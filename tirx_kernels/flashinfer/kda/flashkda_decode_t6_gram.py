# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400),
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's FlashKDA "cake" T=6 coefficient-gram decode kernel.

Source: ``csrc/kda/flashkda_decode_d128_t6_precomputed_gram_split{1,2,4,8}.cu``,
symbol ``kernel_flashinfer_recurrent_kda_wy_vtile_short`` -- the T=6 member of
the same ``coefficient_gram`` generator family as the ported T=5 sibling, and
the last cake variant upstream exports.

Dispatch reaches these bodies whenever ``num_tokens == 6`` with
``num_spec_tokens == 5`` and a precomputed log-space gate. T=5 and T=6 share
one arm of the sm100a value-split policy (``recurrent_kda.py:1181-1191``), so
the split is shape-dependent here too and all four exports are in scope.

Structurally this is the T=5 body at TOKENS=6 with four deltas:

* **The arena is dynamic shared memory.** split1 needs 50560 B, past the
  49152 B static ceiling every earlier cake port used, so all four splits
  allocate through ``T.SMEMPool`` -- which is also what the source does
  (``extern __shared__ __align__(1024)`` plus ``cudaFuncSetAttribute``).
* **Token 5's quad broadcast is elided.** The ``(quad_base + t//2, acc[t%2])``
  map holds for all six tokens, but token 5's source lane is ``quad_base + 2``
  -- the only lane that consumes it -- so the source reads the local registers
  instead and the broadcast count stays at 20.
* **Phase F gained a second tail.** ``lane_quad == 2`` now writes tokens 0, 1,
  4 and 5; token 4's coefficient row needs a clamp because ``sR[29]`` (target
  4, source 5) is never written by the gram block.
* Six token warps: 256/192/192/192 threads, ``barrier.sync 1, 192``, and the
  gram warp is warp 5 at splits 4 and 8.

Helper vocabulary is shared with the T=2 module; only the geometry constants
and the kernel body are per-specialization.
"""

from __future__ import annotations

from typing import Any
from unittest import SkipTest

import torch

from tvm.script import tirx as T

from . import flashkda_decode_t2_precomputed as _t2

# Shared helper vocabulary (identical PTX forms, identical swizzle -- verified
# against this body's exports too; see .porting/.../toolchain_notes.md).
_ptx_un = _t2._ptx_un
_ptx_bin = _t2._ptx_bin
_ptx_ter = _t2._ptx_ter
_mul = _t2._mul
_add = _t2._add
_sub = _t2._sub
_fma = _t2._fma
_rsqrt = _t2._rsqrt
_expf = _t2._expf
_shfl_bfly = _t2._shfl_bfly
_shfl_idx = _t2._shfl_idx
_load_i32 = _t2._load_i32
_load_bf16_f32 = _t2._load_bf16_f32
_widen_lo = _t2._widen_lo
_widen_hi = _t2._widen_hi
_load_u32x2 = _t2._load_u32x2
_load_u32x4 = _t2._load_u32x4
_store_u32x4 = _t2._store_u32x4
_pack_bf16x2 = _t2._pack_bf16x2
_store_f32_as_bf16 = _t2._store_f32_as_bf16
_swz = _t2._swz
_st_shared_f32 = _t2._st_shared_f32
_ld_shared_f32 = _t2._ld_shared_f32
_st_shared_i32 = _t2._st_shared_i32
_ld_shared_i32 = _t2._ld_shared_i32
_ld_shared_f32x4 = _t2._ld_shared_f32x4
_st_shared_u32x4 = _t2._st_shared_u32x4
_st_shared_b16 = _t2._st_shared_b16
_ldmatrix_x4 = _t2._ldmatrix_x4
_mma_zero = _t2._mma_zero
_mma_acc = _t2._mma_acc


def _div(a, b):
    """``div.approx.ftz.f32`` -- the lowering nvcc picks for ``k / prefix``.

    Read off the exported PTX (8 of them in the sVec/sGramA publish, none of
    the rcp+mul or full-range forms); -use_fast_math selects the approximate line.
    """
    return _ptx_bin("div.approx.ftz.f32", a, b)


def _make_warp_uniform(value):
    """``shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF`` -- the source's :36-39.

    Semantically the identity (every lane of a warp already holds the same
    ``tid / 32``), which is why the ported T<=4 siblings spell it ``tid // 32``
    and drop the instruction. It is NOT droppable here: it is the hint that lets
    ptxas prove ``warp_0 < 6`` is warp-uniform. Without it, at split1 -- the only
    geometry where that guard is live -- ptxas cannot assume the shuffles inside
    phase A are collectively executed and wraps them in a
    ``WARPSYNC.COLLECTIVE``/``ENDCOLLECTIVE`` retry region with a back-edge over
    most of the kernel: 14 WARPSYNC, 10 ENDCOLLECTIVE and 10 duplicate
    register-operand ``SHFL.BFLY`` against the export's zero.
    """
    out = T.alloc_local((1,), "uint32")
    T.evaluate(
        T.ptx.shfl_sync.idx.b32(
            out[0], T.reinterpret("uint32", value), T.uint32(0), T.uint32(31), T.uint32(0xFFFFFFFF)
        )
    )
    return T.reinterpret("int32", out[0])


def _named_bar_sync(bar_id: int, threads: int):
    """``barrier.sync <id>, <count>`` -- blocks until `count` threads arrive."""
    T.evaluate(T.ptx["barrier.sync"](bar_id, threads))


def _named_bar_arrive(bar_id: int, threads: int):
    """``barrier.arrive <id>, <count>`` -- releases, does NOT block or acquire."""
    T.evaluate(T.ptx["barrier.arrive"](bar_id, threads))


def _mma_zero_b(acc, a, b, b0: int):
    """``mma.sync...m16n8k16`` with an explicit zero C, B taken at ``b[b0:b0+2]``.

    The gram block issues two products from one pair of ``ldmatrix`` results, so
    unlike the shared helper this one has to select the B half.
    """
    T.evaluate(
        T.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
            acc[0], acc[1], acc[2], acc[3],
            a[0], a[1], a[2], a[3],
            b[b0], b[b0 + 1],
            *_t2._MMA_ZERO_C,
        )
    )  # fmt: skip


def _mma_acc_b(acc, a, b, b0: int):
    """Same, accumulating: C aliases D, matching the source's `+f` tied registers."""
    T.evaluate(
        T.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
            acc[0], acc[1], acc[2], acc[3],
            a[0], a[1], a[2], a[3],
            b[b0], b[b0 + 1],
            acc[0], acc[1], acc[2], acc[3],
        )
    )  # fmt: skip


HEAD_DIM = _t2.HEAD_DIM
NUM_TOKENS = 6
L2_EPS = _t2.L2_EPS
LOG2_E = _t2.LOG2_E

# Per-split geometry, transcribed from the four bodies' `#define` blocks and
# their guard expressions (`.cu:45-88`, `:143-166`, `:180`, `:411`, `:448`,
# `:671`, `:681-684`). `threads` is FLASHKDA_DECODE_LAUNCH_THREADS (the `#define THREADS
# 256` in every body is vestigial -- it only feeds a static_assert in
# binding_impl.cuh:40); it is derived upstream by
# `max(tokens, value_rows/16, ((value_rows/rows_per_group)+1)/2) * 32`
# with rows_per_group = 2 only for gram+split8 (flash_kda_decode.py:115-134).
_SPLIT_GEOMETRY: dict[int, dict[str, int]] = {
    1: {
        "threads": 256,  # 8 warps
        "rows_per_cta": 128,
        "mma_warps": 8,
        "row_groups": 16,
        "rows_per_group": 8,
        "gram_warp": 0,
        "gram_sync_all": 0,
        "su_sync_cta": 0,
        "smem_total": 50560,
        "off_sstate0": 0,
        "off_sstate1": 16384,
        "off_svec": 32768,
        "off_sk": 36864,
        "off_sd": 39936,
        "off_sbeta": 43008,
        "off_sslot": 43032,
        "off_stoken": 43056,
        "off_sinit": 43080,
        "off_sl": 43096,
        "off_sr": 43240,
        "off_su": 43384,
        "off_sgrama0": 46464,
        "off_sgrama1": 48512,
    },
    2: {
        "threads": 192,  # 6 warps
        "rows_per_cta": 64,
        "mma_warps": 4,
        "row_groups": 8,
        "rows_per_group": 8,
        "gram_warp": 0,
        "gram_sync_all": 0,
        "su_sync_cta": 0,
        "smem_total": 32640,
        "off_sstate0": 0,
        "off_sstate1": 8192,
        "off_svec": 16384,
        "off_sk": 20480,
        "off_sd": 23552,
        "off_sbeta": 26624,
        "off_sslot": 26648,
        "off_stoken": 26672,
        "off_sinit": 26696,
        "off_sl": 26712,
        "off_sr": 26856,
        "off_su": 27000,
        "off_sgrama0": 28544,
        "off_sgrama1": 30592,
    },
    4: {
        "threads": 192,  # 6 warps
        "rows_per_cta": 32,
        "mma_warps": 2,
        "row_groups": 4,
        "rows_per_group": 8,
        "gram_warp": 5,
        "gram_sync_all": 1,
        "su_sync_cta": 0,
        "smem_total": 23680,
        "off_sstate0": 0,
        "off_sstate1": 4096,
        "off_svec": 8192,
        "off_sk": 12288,
        "off_sd": 15360,
        "off_sbeta": 18432,
        "off_sslot": 18456,
        "off_stoken": 18480,
        "off_sinit": 18504,
        "off_sl": 18520,
        "off_sr": 18664,
        "off_su": 18808,
        "off_sgrama0": 19584,
        "off_sgrama1": 21632,
    },
    8: {
        "threads": 192,  # 6 warps
        "rows_per_cta": 16,
        "mma_warps": 1,
        "row_groups": 8,
        "rows_per_group": 2,
        "gram_warp": 5,
        "gram_sync_all": 0,
        "su_sync_cta": 1,
        "smem_total": 19200,
        "off_sstate0": 0,
        "off_sstate1": 2048,
        "off_svec": 4096,
        "off_sk": 8192,
        "off_sd": 11264,
        "off_sbeta": 14336,
        "off_sslot": 14360,
        "off_stoken": 14384,
        "off_sinit": 14408,
        "off_sl": 14424,
        "off_sr": 14568,
        "off_su": 14712,
        "off_sgrama0": 15104,
        "off_sgrama1": 17152,
    },
}

K_PER_THREAD = 8


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    """One config row. Defaults mirror FlashInfer's T=6 export bench."""
    config: dict[str, Any] = {
        "label": label,
        "num_seqs": 8,
        "num_heads": 16,
        "num_value_heads": 32,
        "pool_slack": 6,
        "padded_seqs": 0,
        "slot_stride_pad": 0,
        "gate_token_stride_pad": 0,
        "accepted": None,
        "scale": None,
        "sm_count": None,
        "seed": 20260816,
    }
    config.update(overrides)
    return config


# The T=6 split moves with shape: with W = num_seqs * num_value_heads and S SMs
# the bands are W <= 3S/8 -> 8, W <= S/2 -> 2, W <= 3S/4 -> 4, W <= 3S/2 -> 2,
# else 1 (recurrent_kda.py:1181-1191). Every label names the split its shape
# actually selects on a 148-SM B200, and verify_dispatch.py asserts that
# against the real selector. The five hv32h16 rows are FlashInfer's own T=6
# export-bench matrix (all split1); the rest exist to cover the other three
# exports, which are equally reachable in production.
BENCH_CONFIGS = [
    # FlashInfer's own T=6 export bench (W = 256..4096, all split1).
    _case("hv32h16_b8_s1", num_seqs=8),
    _case("hv32h16_b16_s1", num_seqs=16),
    _case("hv32h16_b32_s1", num_seqs=32),
    _case("hv32h16_b64_s1", num_seqs=64),
    _case("hv32h16_b128_s1", num_seqs=128),
    # Small batch at HV=32 walks the first three bands (W = 32, 64, 96).
    _case("hv32h16_b1_s8", num_seqs=1),
    _case("hv32h16_b2_s2", num_seqs=2),
    _case("hv32h16_b3_s4", num_seqs=3),
    # HV == H == 16: W = 32, 80, 128, 1024 -- note b8 lands in the *second*
    # split2 band, the one above the split4 island.
    _case("hv16h16_b2_s8", num_seqs=2, num_value_heads=16),
    _case("hv16h16_b5_s4", num_seqs=5, num_value_heads=16),
    _case("hv16h16_b8_s2", num_seqs=8, num_value_heads=16),
    _case("hv16h16_b64_s1", num_seqs=64, num_value_heads=16),
    # Kimi K3 TP8 per-rank head count (W = 96, 768).
    _case("hv12h12_b8_s4", num_seqs=8, num_heads=12, num_value_heads=12),
    _case("hv12h12_b64_s1", num_seqs=64, num_heads=12, num_value_heads=12),
]

CONFIGS = [dict(cfg) for cfg in BENCH_CONFIGS] + [
    # nat selects the initial checkpoint slot as ssm_idx[n*6 + clamp(nat-1, 0, 5)];
    # the upstream matrix row sweeps 0/1/6/13, covering both clamp arms (.cu:280-286).
    _case("hv32h16_b8_nat0", num_seqs=8, accepted="zeros"),
    _case("hv32h16_b8_nat6", num_seqs=8, accepted="sixes"),
    _case("hv32h16_b8_nat13", num_seqs=8, accepted="thirteens"),
    _case("hv32h16_b8_natmix", num_seqs=8, accepted="mixed"),
    _case("hv32h16_b8_padded", num_seqs=8, padded_seqs=2),
    _case("hv32h16_b8_strided", num_seqs=8, slot_stride_pad=8),
    _case("hv32h16_b8_gstride", num_seqs=8, gate_token_stride_pad=8),
    _case("hv32h16_b8_scale", num_seqs=8, scale=0.05),
    _case("hv64h16_b8_s1", num_seqs=8, num_value_heads=64),
    # Forced splits at one fixed shape (W = 256), via the SM count the policy
    # reads: the same tensors must give the same answer through all four
    # exports. Unlike T=5, upstream has NO forced-split test at T=6 -- its only
    # T=6 GPU coverage is the (6, 8) matrix row, which is split1 -- so these
    # rows and characterize_source.py are the whole story for splits 2/4/8.
    _case("hv32h16_b8_force_s8", num_seqs=8, sm_count=683),
    _case("hv32h16_b8_force_s2", num_seqs=8, sm_count=512),
    _case("hv32h16_b8_force_s4", num_seqs=8, sm_count=342),
    # Selector knife edges on a 148-SM part: W = 222 -> split2, W = 224 -> split1.
    _case("hv16h16_b13_edge_s2", num_seqs=13, num_value_heads=16, sm_count=148),
    _case("hv16h16_b14_edge_s1", num_seqs=14, num_value_heads=16, sm_count=148),
]

KERNEL_META = {
    "name": "flashkda_decode_t6_gram",
    "category": "flashinfer",
    "compute_capability": 10,
}


def _sm_count(kwargs: dict[str, Any]) -> int:
    """SM count driving the split policy; overridable for deterministic tests."""
    override = kwargs.get("sm_count")
    if override is not None:
        return int(override)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required to resolve the FlashKDA decode value split")
    device = kwargs.get("device", "cuda")
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _select_value_split(work: int, sm_count: int) -> int:
    """Reproduce ``_select_flash_kda_decode_value_split_current`` for T = 6.

    recurrent_kda.py:1181-1191. `work` is `num_seqs * num_value_heads`. Note the
    split4 island sits *between* two split2 bands, so this is not monotonic.
    """
    three_wave_ctas = 3 * sm_count
    if 8 * work <= three_wave_ctas:
        return 8
    if 2 * work <= sm_count:
        return 2
    if 4 * work <= three_wave_ctas:
        return 4
    if 2 * work <= three_wave_ctas:
        return 2
    return 1


def _specialization(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Derive the constexpr set, mirroring the source host dispatch."""
    num_seqs = int(kwargs["num_seqs"])
    num_heads = int(kwargs["num_heads"])
    num_value_heads = int(kwargs["num_value_heads"])
    slot_stride_pad = int(kwargs.get("slot_stride_pad", 0))
    gate_token_stride_pad = int(kwargs.get("gate_token_stride_pad", 0))
    pool_slack = int(kwargs.get("pool_slack", 6))

    if num_value_heads % num_heads != 0 or num_value_heads < num_heads:
        raise ValueError("num_value_heads must be a multiple of num_heads and >= it")
    if not 0 < num_seqs <= 65535:
        raise ValueError("num_seqs must fit grid.y (binding_common.cuh:284-287)")
    if slot_stride_pad % 8 != 0:
        raise ValueError("state slot stride padding must stay 8-element aligned")
    if gate_token_stride_pad % 4 != 0:
        raise ValueError("gate token stride padding must stay 4-element aligned")

    value_split = _select_value_split(num_seqs * num_value_heads, _sm_count(kwargs))
    geometry = _SPLIT_GEOMETRY[value_split]

    total_tokens = num_seqs * NUM_TOKENS
    slot_stride = num_value_heads * HEAD_DIM * HEAD_DIM + slot_stride_pad
    gate_token_stride = num_value_heads * HEAD_DIM + gate_token_stride_pad
    state_slots = total_tokens + pool_slack
    spec = {
        "NUM_SEQS": num_seqs,
        "NUM_HEADS": num_heads,
        "NUM_VALUE_HEADS": num_value_heads,
        "HEAD_RATIO": num_value_heads // num_heads,
        "VALUE_SPLIT": value_split,
        "STATE_SLOT_STRIDE": slot_stride,
        "GATE_TOKEN_STRIDE": gate_token_stride,
        "Q_ELEMENTS": total_tokens * num_heads * HEAD_DIM,
        "V_ELEMENTS": total_tokens * num_value_heads * HEAD_DIM,
        "GATE_ELEMENTS": total_tokens * gate_token_stride,
        "BETA_ELEMENTS": total_tokens * num_value_heads,
        "STATE_ELEMENTS": state_slots * slot_stride,
        "CU_SEQLENS_ELEMENTS": num_seqs + 1,
        "STATE_INDEX_ELEMENTS": total_tokens,
        "NAT_ELEMENTS": num_seqs,
    }
    spec.update({key.upper(): value for key, value in geometry.items()})
    return spec


# TIRX_TRANSCRIBE_START flashkda_decode_t6_gram


@T.jit
def _flashkda_decode_t6_gram(
    q_h: T.handle,
    k_h: T.handle,
    v_h: T.handle,
    g_h: T.handle,
    beta_h: T.handle,
    state_h: T.handle,
    out_h: T.handle,
    cu_seqlens_h: T.handle,
    ssm_state_indices_h: T.handle,
    num_accepted_tokens_h: T.handle,
    scale: T.float32,
    *,
    NUM_SEQS: T.constexpr,
    NUM_HEADS: T.constexpr,
    NUM_VALUE_HEADS: T.constexpr,
    HEAD_RATIO: T.constexpr,
    VALUE_SPLIT: T.constexpr,
    STATE_SLOT_STRIDE: T.constexpr,
    GATE_TOKEN_STRIDE: T.constexpr,
    Q_ELEMENTS: T.constexpr,
    V_ELEMENTS: T.constexpr,
    GATE_ELEMENTS: T.constexpr,
    BETA_ELEMENTS: T.constexpr,
    STATE_ELEMENTS: T.constexpr,
    CU_SEQLENS_ELEMENTS: T.constexpr,
    STATE_INDEX_ELEMENTS: T.constexpr,
    NAT_ELEMENTS: T.constexpr,
    THREADS: T.constexpr,
    ROWS_PER_CTA: T.constexpr,
    MMA_WARPS: T.constexpr,
    ROW_GROUPS: T.constexpr,
    ROWS_PER_GROUP: T.constexpr,
    GRAM_WARP: T.constexpr,
    GRAM_SYNC_ALL: T.constexpr,
    SU_SYNC_CTA: T.constexpr,
    SMEM_TOTAL: T.constexpr,
    OFF_SSTATE0: T.constexpr,
    OFF_SSTATE1: T.constexpr,
    OFF_SVEC: T.constexpr,
    OFF_SK: T.constexpr,
    OFF_SD: T.constexpr,
    OFF_SBETA: T.constexpr,
    OFF_SSLOT: T.constexpr,
    OFF_STOKEN: T.constexpr,
    OFF_SINIT: T.constexpr,
    OFF_SL: T.constexpr,
    OFF_SR: T.constexpr,
    OFF_SU: T.constexpr,
    OFF_SGRAMA0: T.constexpr,
    OFF_SGRAMA1: T.constexpr,
):
    """FlashKDA "cake" T=6 coefficient-gram decode; 6 token warps.

    Scaffold only -- the body is written in the kernel-sketch stage, from the
    reviewer-approved sketch at
    `.agents/sketch/flashinfer/kda/flashkda_decode_t6_gram.md`. Bare `:NNN`
    references in the transcribed body are into
    `flashkda_decode_d128_t6_precomputed_gram_split2.cu` unless a split is named.
    """
    q = T.match_buffer(q_h, (Q_ELEMENTS,), "bfloat16", scope="global")
    k = T.match_buffer(k_h, (Q_ELEMENTS,), "bfloat16", scope="global")
    v = T.match_buffer(v_h, (V_ELEMENTS,), "bfloat16", scope="global")
    g = T.match_buffer(g_h, (GATE_ELEMENTS,), "bfloat16", scope="global")
    beta = T.match_buffer(beta_h, (BETA_ELEMENTS,), "bfloat16", scope="global")
    state = T.match_buffer(state_h, (STATE_ELEMENTS,), "bfloat16", scope="global")
    out = T.match_buffer(out_h, (V_ELEMENTS,), "bfloat16", scope="global")
    cu = T.match_buffer(cu_seqlens_h, (CU_SEQLENS_ELEMENTS,), "int32", scope="global")
    ssm_idx = T.match_buffer(ssm_state_indices_h, (STATE_INDEX_ELEMENTS,), "int32", scope="global")
    nat = T.match_buffer(num_accepted_tokens_h, (NAT_ELEMENTS,), "int32", scope="global")
    T.device_entry()

    # The source declares `extern __shared__ __align__(1024) char smem_raw[]`
    # and sizes it with cudaFuncSetAttribute (binding_impl.cuh:59-66). At T=6 the
    # port has to do the same rather than declare a static buffer: split1 needs
    # 50560 B, past the 49152 B ceiling the T<=5 ports fit inside. The pool
    # arena carries the same byte offsets at the same 1024-byte alignment, which
    # is what the swizzled ldmatrix addressing depends on. `get_kernel` adds
    # `tirx.use_dyn_shared_memory` to the launch params; without it the launch
    # reserves zero bytes and every access below faults.
    pool = T.SMEMPool()
    arena = pool.alloc((SMEM_TOTAL,), "uint8", align=1024)
    pool.commit()

    # --- work decomposition and lane roles (:142-178) ----------------------
    work, n = T.cta_id([NUM_VALUE_HEADS * VALUE_SPLIT, NUM_SEQS])
    tid = T.thread_id([THREADS])
    value_tile: T.int32 = work % VALUE_SPLIT
    hv: T.int32 = work // VALUE_SPLIT
    query_head: T.int32 = hv // HEAD_RATIO
    warp: T.int32 = _make_warp_uniform(tid // 32)  # == token index in A and C'
    lane: T.int32 = tid % 32
    lane_quad: T.int32 = lane % 4
    frag_row: T.int32 = lane // 4
    quad_base: T.int32 = lane - lane_quad
    group: T.int32 = tid // 16
    lane_group: T.int32 = tid % 16
    k_start: T.int32 = lane_group * 8
    elem_start: T.int32 = lane * 4
    tile_row_base: T.int32 = value_tile * ROWS_PER_CTA
    owned_row_base: T.int32 = group * ROWS_PER_GROUP
    token_base: T.int32 = _load_i32(cu, n)
    seq_len: T.int32 = _load_i32(cu, n + 1) - token_base

    r_q = T.alloc_local((4,), "float32")
    r_k = T.alloc_local((4,), "float32")
    r_d = T.alloc_local((4,), "float32")

    # =======================================================================
    # Phase A: token preprocess, warp <-> token  (:180-290)
    # =======================================================================
    # Identical to the ported T=4 phase A apart from the token count, the
    # ssm_state_indices stride and the nat clamp ceiling.
    if warp < NUM_TOKENS:
        token: T.int32 = warp
        active_token = token < seq_len
        token_pos: T.int32 = T.if_then_else(active_token, token_base + token, 0)
        qk_base: T.int32 = (token_pos * NUM_HEADS + query_head) * HEAD_DIM + elem_start
        gate_base: T.int32 = token_pos * GATE_TOKEN_STRIDE + hv * HEAD_DIM + elem_start

        q_words = _load_u32x2(q, qk_base)
        k_words = _load_u32x2(k, qk_base)
        g_words = _load_u32x2(g, gate_base)
        for pair in T.unroll(2):
            r_q[2 * pair] = _widen_lo(q_words[pair])
            r_q[2 * pair + 1] = _widen_hi(q_words[pair])
            r_k[2 * pair] = _widen_lo(k_words[pair])
            r_k[2 * pair + 1] = _widen_hi(k_words[pair])
            r_d[2 * pair] = _widen_lo(g_words[pair])
            r_d[2 * pair + 1] = _widen_hi(g_words[pair])

        # Index-ordered accumulation (:241-244); the first term has a zero addend
        # and the two chains interleave, because the source fuses them in one loop.
        q_sq: T.float32 = _fma(
            r_q[3],
            r_q[3],
            _fma(r_q[2], r_q[2], _fma(r_q[1], r_q[1], _fma(r_q[0], r_q[0], T.float32(0.0)))),
        )
        k_sq: T.float32 = _fma(
            r_k[3],
            r_k[3],
            _fma(r_k[2], r_k[2], _fma(r_k[1], r_k[1], _fma(r_k[0], r_k[0], T.float32(0.0)))),
        )
        # Two sequential full-warp butterflies, not interleaved (:245-254).
        for off in T.unroll(5):
            q_sq = _add(q_sq, _shfl_bfly(q_sq, 16 >> off))
        for off in T.unroll(5):
            k_sq = _add(k_sq, _shfl_bfly(k_sq, 16 >> off))
        q_norm: T.float32 = _mul(_rsqrt(_add(q_sq, T.float32(L2_EPS))), scale)
        k_norm: T.float32 = _rsqrt(_add(k_sq, T.float32(L2_EPS)))

        # GATE_KIND == 0: `g` already holds log(gamma), so one exp per element.
        k_pub = T.alloc_local((4,), "uint32")
        d_pub = T.alloc_local((4,), "uint32")
        for i in T.unroll(4):
            r_q[i] = _mul(r_q[i], q_norm)
            r_k[i] = _mul(r_k[i], k_norm)
            r_d[i] = _expf(r_d[i])
            k_pub[i] = T.reinterpret("uint32", r_k[i])
            d_pub[i] = T.reinterpret("uint32", r_d[i])
        # Four contiguous f32 per lane: one 16-byte shared store each, not four
        # scalar ones (:269-270 lower to 2 st.shared.v4.b32).
        _st_shared_u32x4(arena, OFF_SK + (token * HEAD_DIM + elem_start) * 4, k_pub)
        _st_shared_u32x4(arena, OFF_SD + (token * HEAD_DIM + elem_start) * 4, d_pub)

        if lane == 0:
            raw_slot: T.int32 = _load_i32(ssm_idx, n * NUM_TOKENS + token)
            _st_shared_i32(arena, OFF_SSLOT + token * 4, T.if_then_else(active_token, raw_slot, -1))
            _st_shared_i32(arena, OFF_STOKEN + token * 4, token_pos)
            _st_shared_f32(
                arena, OFF_SBETA + token * 4, _load_bf16_f32(beta, token_pos * NUM_VALUE_HEADS + hv)
            )
            if token == 0:
                # nat picks the initial checkpoint slot; at T=5 the clamp ceiling
                # is 4 and both edges are reachable (:279-287).
                accepted: T.int32 = T.min(T.max(_load_i32(nat, n) - 1, 0), NUM_TOKENS - 1)
                initial_slot: T.int32 = _load_i32(ssm_idx, n * NUM_TOKENS + accepted)
                _st_shared_i32(arena, OFF_SINIT, T.max(initial_slot, 0))

    T.cuda.cta_sync()

    # =======================================================================
    # Phase C': sVec columns and the gram operand  (:293-339)
    # =======================================================================
    # Runs BEFORE the state gather at T=5 -- the reverse of the T=4 order.
    if warp < NUM_TOKENS:
        token_c: T.int32 = warp
        for i in T.unroll(4):
            k_idx: T.int32 = elem_start + i
            prefix: T.float32 = T.float32(1.0)
            # Scalar loads: the walk is across tokens at a fixed key, so
            # consecutive iterations are 512 B apart (:296-302).
            for j in T.unroll(NUM_TOKENS):
                if token_c >= j:
                    prefix = _mul(
                        prefix, _ld_shared_f32(arena, OFF_SD + (j * HEAD_DIM + k_idx) * 4)
                    )
            _st_shared_b16(
                arena,
                OFF_SVEC + _swz(k_idx * 32 + token_c * 2),
                _ptx_un("cvt.rn.bf16.f32", _mul(prefix, r_k[i]), dtype="uint16"),
            )
            # The q column is 8 + token at T=5, not 4 + token: the generator
            # emits `c_col = 4 + token` and immediately overrides it (:311-314).
            _st_shared_b16(
                arena,
                OFF_SVEC + _swz(k_idx * 32 + (8 + token_c) * 2),
                _ptx_un("cvt.rn.bf16.f32", _mul(prefix, r_q[i]), dtype="uint16"),
            )
            # The gate-DEFLATED key, the operand that makes the Gram product come
            # out as T<=4's ratio_scan factor. div.approx.ftz.f32 per the PTX.
            deflated = _ptx_un("cvt.rn.bf16.f32", _div(r_k[i], prefix), dtype="uint16")
            if k_idx < 64:
                _st_shared_b16(arena, OFF_SGRAMA0 + _swz(token_c * 128 + k_idx * 2), deflated)
            else:
                _st_shared_b16(
                    arena, OFF_SGRAMA1 + _swz(token_c * 128 + (k_idx - 64) * 2), deflated
                )

    # =======================================================================
    # Phase C'': the coefficient Gram block  (:340-408)  -- new at T=5
    # =======================================================================
    # One warp forms both token x token Gram products on tensor cores, replacing
    # T<=4's inter-token butterfly path entirely. The barrier is partial: only
    # the five token warps take part, which is why its count is 160 in every
    # split, including split1's 256-thread launch.
    if warp < NUM_TOKENS:
        if GRAM_SYNC_ALL == 1:
            # split4 hoists the wait out of the branch: all five token warps
            # block, and the arrive arm below is `else if (0)` -- dead (S4:340-344).
            _named_bar_sync(1, NUM_TOKENS * 32)
        if warp == GRAM_WARP:
            if GRAM_SYNC_ALL == 0:
                _named_bar_sync(1, NUM_TOKENS * 32)
            gram_a = T.alloc_local((4,), "uint32", align=4)
            gram_b = T.alloc_local((4,), "uint32", align=4)
            gram_k_acc = T.alloc_local((4,), "float32", align=4)
            gram_q_acc = T.alloc_local((4,), "float32", align=4)
            for gram_half in T.unroll(2):
                for gram_step in T.unroll(4):
                    gram_k: T.int32 = gram_step * 16
                    global_gram_k: T.int32 = gram_half * 64 + gram_k
                    a_off: T.int32 = (lane % 16 * 64 + (gram_k + lane // 16 * 8)) * 2
                    if gram_half == 0:
                        _ldmatrix_x4(arena, OFF_SGRAMA0 + _swz(a_off), gram_a, False)
                    else:
                        _ldmatrix_x4(arena, OFF_SGRAMA1 + _swz(a_off), gram_a, False)
                    # Same sVec address formula as phase D's operand (:360-361
                    # vs :454-455); frags 0,1 are columns 0..7 (the k side) and
                    # 2,3 are columns 8..15 (the q side).
                    _ldmatrix_x4(
                        arena,
                        OFF_SVEC + _swz(((global_gram_k + lane % 16) * 16 + lane // 16 * 8) * 2),
                        gram_b,
                        True,
                    )
                    if gram_half == 0 and gram_step == 0:
                        _mma_zero_b(gram_k_acc, gram_a, gram_b, 0)
                        _mma_zero_b(gram_q_acc, gram_a, gram_b, 2)
                    else:
                        _mma_acc_b(gram_k_acc, gram_a, gram_b, 0)
                        _mma_acc_b(gram_q_acc, gram_a, gram_b, 2)

            # The transpose of T<=4's roles: the MMA M index is the SOURCE token
            # and the N index (the sVec column) is the TARGET (:387-404). Only
            # acc[0],[1] are read -- acc[2],[3] hold M rows 8..15, past the five
            # real tokens.
            source_token: T.int32 = frag_row
            target0: T.int32 = lane_quad * 2
            target1: T.int32 = target0 + 1
            if source_token < NUM_TOKENS:
                beta_source: T.float32 = _ld_shared_f32(arena, OFF_SBETA + source_token * 4)
                if source_token < target0 and target0 < NUM_TOKENS:
                    _st_shared_f32(
                        arena,
                        OFF_SL + (target0 * NUM_TOKENS + source_token) * 4,
                        _mul(beta_source, gram_k_acc[0]),
                    )
                if source_token < target1 and target1 < NUM_TOKENS:
                    _st_shared_f32(
                        arena,
                        OFF_SL + (target1 * NUM_TOKENS + source_token) * 4,
                        _mul(beta_source, gram_k_acc[1]),
                    )
                if source_token <= target0 and target0 < NUM_TOKENS:
                    _st_shared_f32(
                        arena,
                        OFF_SR + (target0 * NUM_TOKENS + source_token) * 4,
                        _mul(beta_source, gram_q_acc[0]),
                    )
                if source_token <= target1 and target1 < NUM_TOKENS:
                    _st_shared_f32(
                        arena,
                        OFF_SR + (target1 * NUM_TOKENS + source_token) * 4,
                        _mul(beta_source, gram_q_acc[1]),
                    )
        else:
            if GRAM_SYNC_ALL == 0:
                _named_bar_arrive(1, NUM_TOKENS * 32)

    # =======================================================================
    # Phase B: state gather and sState stage  (:410-446)
    # =======================================================================
    init_slot: T.int32 = _ld_shared_i32(arena, OFF_SINIT)
    head_base = T.cast(init_slot, "int64") * T.cast(STATE_SLOT_STRIDE, "int64") + T.cast(
        hv * HEAD_DIM * HEAD_DIM, "int64"
    )
    hist = T.alloc_local((ROWS_PER_GROUP * 8,), "float32")
    if group < ROW_GROUPS:
        for row_local in T.unroll(ROWS_PER_GROUP):
            # Two distinct indices: row_l is CTA-local and addresses sState;
            # tile_row_base + row_l is the global row of `state` (:414-415).
            row_l: T.int32 = owned_row_base + row_local
            pack = _load_u32x4(
                state, head_base + T.cast((tile_row_base + row_l) * HEAD_DIM + k_start, "int64")
            )
            for pr in T.unroll(4):
                hist[row_local * 8 + 2 * pr] = _widen_lo(pack[pr])
                hist[row_local * 8 + 2 * pr + 1] = _widen_hi(pack[pr])
            # An if/ELSE selecting the destination half, not a guard: lanes 8..15
            # stage keys 64..127 into sState1 (:438-442). The bf16 bits go to
            # shared unmodified; the swizzle is on the byte offset.
            if lane_group < 8:
                _st_shared_u32x4(arena, OFF_SSTATE0 + _swz(row_l * 128 + k_start * 2), pack)
            else:
                _st_shared_u32x4(arena, OFF_SSTATE1 + _swz(row_l * 128 + (k_start - 64) * 2), pack)

    T.cuda.cta_sync()
    # Besides sState and sL/sR, this is the sVec publish edge for every MMA warp
    # except the gram warp: `barrier.arrive` releases but does not acquire, so an
    # arriving token warp has synchronized with the others only here.

    # =======================================================================
    # Phase D: the MMA chain, warp <-> 16 value rows  (:448-490)
    # =======================================================================
    # Two issues per step at T=5: the k side needs sVec columns 0..4 and the q
    # side 8..12, which no single n=8 tile covers. `mma_acc_c` and
    # `vec_frag[2],[3]` were dead at every T <= 4.
    acc = T.alloc_local((4,), "float32", align=4)
    acc_c = T.alloc_local((4,), "float32", align=4)
    if warp < MMA_WARPS:
        vec_frag = T.alloc_local((4,), "uint32", align=4)
        state_frag = T.alloc_local((4,), "uint32", align=4)
        for state_half in T.unroll(2):
            for mma_step in T.unroll(4):
                mma_k: T.int32 = mma_step * 16
                global_k: T.int32 = state_half * 64 + mma_k
                _ldmatrix_x4(
                    arena,
                    OFF_SVEC + _swz((global_k + lane % 16) * 32 + lane // 16 * 16),
                    vec_frag,
                    True,
                )
                if state_half == 0:
                    _ldmatrix_x4(
                        arena,
                        OFF_SSTATE0
                        + _swz((warp * 16 + lane % 16) * 128 + (mma_k + lane // 16 * 8) * 2),
                        state_frag,
                        False,
                    )
                else:
                    _ldmatrix_x4(
                        arena,
                        OFF_SSTATE1
                        + _swz((warp * 16 + lane % 16) * 128 + (mma_k + lane // 16 * 8) * 2),
                        state_frag,
                        False,
                    )
                if state_half == 0 and mma_step == 0:
                    _mma_zero_b(acc, state_frag, vec_frag, 0)
                    _mma_zero_b(acc_c, state_frag, vec_frag, 2)
                else:
                    _mma_acc_b(acc, state_frag, vec_frag, 0)
                    _mma_acc_b(acc_c, state_frag, vec_frag, 2)

    # =======================================================================
    # Phase E: quad broadcast and the WY forward substitution  (:491-566)
    # =======================================================================
    u_lo = T.alloc_local((NUM_TOKENS,), "float32")
    u_hi = T.alloc_local((NUM_TOKENS,), "float32")
    # hc_* is declared out here because phase F consumes it from its own block;
    # ha_* never leaves phase E.
    hc_lo = T.alloc_local((NUM_TOKENS,), "float32")
    hc_hi = T.alloc_local((NUM_TOKENS,), "float32")
    if warp < MMA_WARPS:
        # 20 broadcasts in the source's emission order (:491-531). m16n8k16 puts
        # columns 2q, 2q+1 of row frag_row in lane quad_base+q's acc[0],[1] and
        # rows +8 in acc[2],[3], so token t lives at (quad_base + t//2, t%2).
        # ha_* is the k side (the solve), hc_* the q side (the outputs).
        ha_lo = T.alloc_local((NUM_TOKENS,), "float32")
        ha_hi = T.alloc_local((NUM_TOKENS,), "float32")
        for t in T.unroll(4):
            ha_lo[t] = _shfl_idx(acc[t % 2], quad_base + t // 2)
        for t in T.unroll(4):
            ha_hi[t] = _shfl_idx(acc[2 + t % 2], quad_base + t // 2)
        ha_lo[4] = _shfl_idx(acc[0], quad_base + 2)
        ha_hi[4] = _shfl_idx(acc[2], quad_base + 2)
        for t in T.unroll(NUM_TOKENS - 1):
            hc_lo[t] = _shfl_idx(acc_c[t % 2], quad_base + t // 2)
        for t in T.unroll(NUM_TOKENS - 1):
            hc_hi[t] = _shfl_idx(acc_c[2 + t % 2], quad_base + t // 2)

        # Token 5 needs NO shuffle (:532-537). The (quad_base + t//2, acc[t%2])
        # map puts it on lane quad_base + 2 -- which is the only lane that
        # consumes it, since both the solve below and phase F's token-5 tail run
        # under `lane_quad == 2`. So the source reads the local registers, and
        # the broadcast count stays at 20 rather than 24. These four assignments
        # sit OUTSIDE the `lane_quad == 2` guard: every lane executes them, only
        # lane_quad 2's values are meaningful.
        ha_lo[NUM_TOKENS - 1] = acc[1]
        ha_hi[NUM_TOKENS - 1] = acc[3]
        hc_lo[NUM_TOKENS - 1] = acc_c[1]
        hc_hi[NUM_TOKENS - 1] = acc_c[3]

        if lane_quad == 2:
            row_lo: T.int32 = warp * 16 + frag_row
            row_hi: T.int32 = row_lo + 8
            for t in T.unroll(NUM_TOKENS):
                base_t: T.int32 = (
                    _ld_shared_i32(arena, OFF_STOKEN + t * 4) * NUM_VALUE_HEADS + hv
                ) * HEAD_DIM
                solved_lo: T.float32 = _sub(
                    _load_bf16_f32(v, base_t + tile_row_base + row_lo), ha_lo[t]
                )
                solved_hi: T.float32 = _sub(
                    _load_bf16_f32(v, base_t + tile_row_base + row_hi), ha_hi[t]
                )
                for prev in T.unroll(NUM_TOKENS):
                    if prev < t:
                        lts: T.float32 = _ld_shared_f32(arena, OFF_SL + (t * NUM_TOKENS + prev) * 4)
                        solved_lo = _sub(solved_lo, _mul(lts, u_lo[prev]))
                        solved_hi = _sub(solved_hi, _mul(lts, u_hi[prev]))
                u_lo[t] = solved_lo
                u_hi[t] = solved_hi

        # The solve runs on lane_quad == 2 but lane_quad == 3 also writes output,
        # so the residuals cross the quad (:554-560).
        for t in T.unroll(NUM_TOKENS):
            u_lo[t] = _shfl_idx(u_lo[t], quad_base + 2)
            u_hi[t] = _shfl_idx(u_hi[t], quad_base + 2)

    # =======================================================================
    # Phase F: the outputs  (:568-670)
    # =======================================================================
    # The bases come from hc_* (the q-side MMA), not from acc: the source
    # assigns acc[0..3] and then unconditionally overwrites (:567-582). The
    # lane_quad == 3 remap uses STATIC indices, as the source does -- indexing
    # hc_* by a runtime token would spill the register array to local memory.
    if warp < MMA_WARPS and lane_quad >= 2:
        token0: T.int32 = (lane_quad - 2) * 2
        token1: T.int32 = token0 + 1
        row_lo_f: T.int32 = warp * 16 + frag_row
        row_hi_f: T.int32 = row_lo_f + 8
        out0_lo: T.float32 = hc_lo[0]
        out1_lo: T.float32 = hc_lo[1]
        out0_hi: T.float32 = hc_hi[0]
        out1_hi: T.float32 = hc_hi[1]
        if lane_quad == 3:
            out0_lo = hc_lo[2]
            out1_lo = hc_lo[3]
            out0_hi = hc_hi[2]
            out1_hi = hc_hi[3]
        for src in T.unroll(NUM_TOKENS):
            residual_lo: T.float32 = u_lo[src]
            residual_hi: T.float32 = u_hi[src]
            coef0: T.float32 = T.float32(0.0)
            coef1: T.float32 = T.float32(0.0)
            # The masked-out coefficient is a real zero-operand fma, not a
            # skipped iteration (:585-594).
            if token0 >= src:
                coef0 = _ld_shared_f32(arena, OFF_SR + (token0 * NUM_TOKENS + src) * 4)
            if token1 >= src:
                coef1 = _ld_shared_f32(arena, OFF_SR + (token1 * NUM_TOKENS + src) * 4)
            out0_lo = _fma(coef0, residual_lo, out0_lo)
            out1_lo = _fma(coef1, residual_lo, out1_lo)
            out0_hi = _fma(coef0, residual_hi, out0_hi)
            out1_hi = _fma(coef1, residual_hi, out1_hi)

        for half in T.unroll(2):
            token_o: T.int32 = T.if_then_else(half == 0, token0, token1)
            o_lo: T.float32 = T.if_then_else(half == 0, out0_lo, out1_lo)
            o_hi: T.float32 = T.if_then_else(half == 0, out0_hi, out1_hi)
            active_o = _ld_shared_i32(arena, OFF_SSLOT + token_o * 4) >= 0
            base_o: T.int32 = (
                _ld_shared_i32(arena, OFF_STOKEN + token_o * 4) * NUM_VALUE_HEADS + hv
            ) * HEAD_DIM + tile_row_base
            # A padded row writes EXPLICIT zeros; the upstream test asserts them
            # bit-exactly, so this is not an "unwritten" path.
            _store_f32_as_bf16(out, base_o + row_lo_f, o_lo, active_o)
            _store_f32_as_bf16(out, base_o + row_hi_f, o_hi, active_o)
            _store_f32_as_bf16(out, base_o + row_lo_f, T.float32(0.0), T.Not(active_o))
            _store_f32_as_bf16(out, base_o + row_hi_f, T.float32(0.0), T.Not(active_o))

        # Tokens 4 and 5 have no partner lane, so lane_quad == 2 writes BOTH
        # tails at T=6 (:628-668) -- eight output stores against quad 3's four.
        if lane_quad == 2:
            # Token 4 (:629-650). Its coefficient row is CLAMPED: sR[29] is
            # (target 4, source 5), which the gram block never writes -- its
            # predicate is `source <= target` -- so it is in-region but
            # uninitialized, and only the `src <= 4` mask keeps it out of the
            # result. The source's unconditional read at :633 is dead.
            out4_lo: T.float32 = hc_lo[NUM_TOKENS - 2]
            out4_hi: T.float32 = hc_hi[NUM_TOKENS - 2]
            for src4 in T.unroll(NUM_TOKENS):
                coef4: T.float32 = T.float32(0.0)
                if src4 <= NUM_TOKENS - 2:
                    coef4 = _ld_shared_f32(
                        arena, OFF_SR + ((NUM_TOKENS - 2) * NUM_TOKENS + src4) * 4
                    )
                out4_lo = _fma(coef4, u_lo[src4], out4_lo)
                out4_hi = _fma(coef4, u_hi[src4], out4_hi)
            active4 = _ld_shared_i32(arena, OFF_SSLOT + (NUM_TOKENS - 2) * 4) >= 0
            base4: T.int32 = (
                _ld_shared_i32(arena, OFF_STOKEN + (NUM_TOKENS - 2) * 4) * NUM_VALUE_HEADS + hv
            ) * HEAD_DIM + tile_row_base
            _store_f32_as_bf16(out, base4 + row_lo_f, out4_lo, active4)
            _store_f32_as_bf16(out, base4 + row_hi_f, out4_hi, active4)
            _store_f32_as_bf16(out, base4 + row_lo_f, T.float32(0.0), T.Not(active4))
            _store_f32_as_bf16(out, base4 + row_hi_f, T.float32(0.0), T.Not(active4))

            # Token 5 (:651-667). No clamp: the last target accepts every
            # source, so sR[30..35] are all live.
            out5_lo: T.float32 = hc_lo[NUM_TOKENS - 1]
            out5_hi: T.float32 = hc_hi[NUM_TOKENS - 1]
            for src5 in T.unroll(NUM_TOKENS):
                coef5: T.float32 = _ld_shared_f32(
                    arena, OFF_SR + ((NUM_TOKENS - 1) * NUM_TOKENS + src5) * 4
                )
                out5_lo = _fma(coef5, u_lo[src5], out5_lo)
                out5_hi = _fma(coef5, u_hi[src5], out5_hi)
            active5 = _ld_shared_i32(arena, OFF_SSLOT + (NUM_TOKENS - 1) * 4) >= 0
            base5: T.int32 = (
                _ld_shared_i32(arena, OFF_STOKEN + (NUM_TOKENS - 1) * 4) * NUM_VALUE_HEADS + hv
            ) * HEAD_DIM + tile_row_base
            _store_f32_as_bf16(out, base5 + row_lo_f, out5_lo, active5)
            _store_f32_as_bf16(out, base5 + row_hi_f, out5_hi, active5)
            _store_f32_as_bf16(out, base5 + row_lo_f, T.float32(0.0), T.Not(active5))
            _store_f32_as_bf16(out, base5 + row_hi_f, T.float32(0.0), T.Not(active5))

    # =======================================================================
    # Phase G: publish sU  (:671-684)
    # =======================================================================
    if warp < MMA_WARPS:
        if lane_quad == 2:
            row_lo_g: T.int32 = warp * 16 + frag_row
            for t in T.unroll(NUM_TOKENS):
                _st_shared_f32(arena, OFF_SU + (t * ROWS_PER_CTA + row_lo_g) * 4, u_lo[t])
                _st_shared_f32(arena, OFF_SU + (t * ROWS_PER_CTA + row_lo_g + 8) * 4, u_hi[t])
        if SU_SYNC_CTA == 0:
            # Warp w produces rows [16w, 16w+16), exactly the rows its own groups
            # consume in phase H, so a warp barrier suffices (:651-653).
            T.cuda.warp_sync()
    if SU_SYNC_CTA == 1:
        # split8 has ONE MMA warp producing sU for eight consuming groups, so the
        # generator emits a CTA barrier -- and it sits OUTSIDE the warp guard
        # (S8:651 closes it, S8:652-654 follows). Keeping it inside would have
        # one warp of five arrive at a CTA barrier and the kernel would hang.
        T.cuda.cta_sync()

    # =======================================================================
    # Phase H: recurrence and checkpoints  (:798-834)
    # =======================================================================
    words_w = T.alloc_local((4,), "uint32")
    sd_t = T.alloc_local((8,), "float32")
    sk_t = T.alloc_local((8,), "float32")
    if group < ROW_GROUPS:
        for t in T.unroll(NUM_TOKENS):
            slot_t: T.int32 = _ld_shared_i32(arena, OFF_SSLOT + t * 4)
            beta_t: T.float32 = _ld_shared_f32(arena, OFF_SBETA + t * 4)
            # The gate and key slices depend only on (t, k_start), not on the
            # row, so they are hoisted out of the row loop as two 16-byte reads
            # each rather than reloaded per key (:782).
            _ld_shared_f32x4(arena, OFF_SD + (t * HEAD_DIM + k_start) * 4, sd_t, 0)
            _ld_shared_f32x4(arena, OFF_SD + (t * HEAD_DIM + k_start + 4) * 4, sd_t, 4)
            _ld_shared_f32x4(arena, OFF_SK + (t * HEAD_DIM + k_start) * 4, sk_t, 0)
            _ld_shared_f32x4(arena, OFF_SK + (t * HEAD_DIM + k_start + 4) * 4, sk_t, 4)
            # The store predicate is hoisted OUT of the row loop and the loop
            # duplicated, which is the shape nvcc produces for the source's
            # per-row `if (slot_t >= 0)` (:788): its phase H appears 2*TOKENS-1
            # times in the export's SASS, against a single copy when the branch
            # stays inside. Keeping the branch per row costs DRAM utilisation at
            # the largest split1 shapes -- the stores cannot be issued as densely.
            # The recurrence itself is identical in both arms: it advances
            # unconditionally, in FP32, so token t+1 consumes the un-rounded
            # token-t state rather than the bf16 checkpoint.
            if slot_t >= 0:
                for row_local in T.unroll(ROWS_PER_GROUP):
                    row_h: T.int32 = owned_row_base + row_local
                    update: T.float32 = _mul(
                        _ld_shared_f32(arena, OFF_SU + (t * ROWS_PER_CTA + row_h) * 4), beta_t
                    )
                    for i in T.unroll(8):
                        # `hist*sD + update*sK` (:782): the compiler contracts the
                        # FIRST product and rounds update*sK.
                        hist[row_local * 8 + i] = _fma(
                            hist[row_local * 8 + i], sd_t[i], _mul(update, sk_t[i])
                        )
                    for pr in T.unroll(4):
                        words_w[pr] = _pack_bf16x2(
                            hist[row_local * 8 + 2 * pr + 1], hist[row_local * 8 + 2 * pr]
                        )
                    _store_u32x4(
                        state,
                        T.cast(slot_t, "int64") * T.cast(STATE_SLOT_STRIDE, "int64")
                        + T.cast(
                            hv * HEAD_DIM * HEAD_DIM + (tile_row_base + row_h) * HEAD_DIM + k_start,
                            "int64",
                        ),
                        words_w,
                    )
            else:
                for row_local_p in T.unroll(ROWS_PER_GROUP):
                    row_p: T.int32 = owned_row_base + row_local_p
                    update_p: T.float32 = _mul(
                        _ld_shared_f32(arena, OFF_SU + (t * ROWS_PER_CTA + row_p) * 4), beta_t
                    )
                    for i in T.unroll(8):
                        hist[row_local_p * 8 + i] = _fma(
                            hist[row_local_p * 8 + i], sd_t[i], _mul(update_p, sk_t[i])
                        )


# The dynamic-shared arena is only backed at launch when the kernel asks for
# it; without this tag the launch reserves zero bytes and every arena access
# faults at runtime (.porting/flashkda_decode_t6_gram/probe_dyn_smem.py). The
# grid is 2-D (binding_impl.cuh:64).
LAUNCH_TAGS = ("blockIdx.x", "blockIdx.y", "threadIdx.x", "tirx.use_dyn_shared_memory")


def get_kernel(**kwargs: Any):
    """Return the specialized FlashKDA cake T=6 gram decode PrimFunc."""
    kernel = _flashkda_decode_t6_gram.specialize(**_specialization(kwargs))
    return kernel.with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))


_NAT_PATTERNS = {None: None, "zeros": 0, "ones": 1, "sixes": 6, "thirteens": 13}


def _accepted_tensor(kind: Any, num_seqs: int, device: str) -> torch.Tensor:
    """num_accepted_tokens per case; the kernel clamps nat-1 into [0, 5]."""
    if kind is None:
        return torch.ones(num_seqs, device=device, dtype=torch.int32)
    if kind == "mixed":
        # The upstream T=6 matrix test's sweep, tiled to the batch size.
        pattern = torch.tensor([0, 1, 6, 13, 5, 0, 1, 6], dtype=torch.int32)
        reps = (num_seqs + pattern.numel() - 1) // pattern.numel()
        return pattern.repeat(reps)[:num_seqs].to(device)
    if kind in _NAT_PATTERNS:
        return torch.full((num_seqs,), _NAT_PATTERNS[kind], device=device, dtype=torch.int32)
    raise ValueError(f"unknown accepted-token pattern {kind!r}")


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Build one deterministic case with independent TIRx / reference state.

    Same recipe as the ported T=2/T=4 siblings at T=6: packed [1, N*6, ...]
    bf16, a log-space gate (GATE_KIND 0 -- the kernel applies exp()),
    pre-sigmoided beta, flat [N*6] slot indices with slot 0 never a target.
    """
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for FlashKDA cake T=6 decode")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(f"FlashKDA cake decode targets compute capability 10.x, got {capability}")

    spec = _specialization({**kwargs, "device": device})
    num_seqs = spec["NUM_SEQS"]
    num_heads = spec["NUM_HEADS"]
    num_value_heads = spec["NUM_VALUE_HEADS"]
    slot_stride = spec["STATE_SLOT_STRIDE"]
    gate_token_stride = spec["GATE_TOKEN_STRIDE"]
    total_tokens = num_seqs * NUM_TOKENS
    state_slots = spec["STATE_ELEMENTS"] // slot_stride
    padded_seqs = int(kwargs.get("padded_seqs", 0))

    gen = torch.Generator(device=device)
    gen.manual_seed(int(kwargs["seed"]))

    def randn(*shape: int, dtype=torch.bfloat16, gain: float = 1.0):
        raw = torch.randn(shape, device=device, dtype=torch.float32, generator=gen)
        return (gain * raw).to(dtype)

    q = randn(1, total_tokens, num_heads, HEAD_DIM, gain=0.5)
    k = randn(1, total_tokens, num_heads, HEAD_DIM, gain=0.5)
    v = randn(1, total_tokens, num_value_heads, HEAD_DIM, gain=0.5)
    # GATE_KIND == 0: g holds the per-K log-gate; the kernel applies exp().
    gate_logits = torch.randn(
        (1, total_tokens, num_value_heads, HEAD_DIM),
        device=device,
        dtype=torch.float32,
        generator=gen,
    )
    g_dense = torch.nn.functional.logsigmoid(gate_logits).to(torch.bfloat16)
    g_raw = torch.zeros((total_tokens * gate_token_stride,), device=device, dtype=torch.bfloat16)
    g = g_raw.as_strided(
        (1, total_tokens, num_value_heads, HEAD_DIM),
        (total_tokens * gate_token_stride, gate_token_stride, HEAD_DIM, 1),
    )
    g.copy_(g_dense)
    beta = torch.sigmoid(randn(1, total_tokens, num_value_heads, dtype=torch.float32, gain=0.5)).to(
        torch.bfloat16
    )

    cu_seqlens = torch.arange(0, total_tokens + 1, NUM_TOKENS, device=device, dtype=torch.int32)
    # Slot 0 is deliberately never a checkpoint target (upstream bench recipe).
    slots = torch.arange(1, total_tokens + 1, device=device, dtype=torch.int32).reshape(
        num_seqs, NUM_TOKENS
    )
    if padded_seqs:
        slots[num_seqs - padded_seqs :, :] = -1
    ssm_state_indices = slots.contiguous().view(-1)
    num_accepted_tokens = _accepted_tensor(kwargs.get("accepted"), num_seqs, device)

    def make_state_pool() -> tuple[torch.Tensor, torch.Tensor]:
        raw = randn(state_slots * slot_stride, gain=0.01)
        view = raw.as_strided(
            (state_slots, num_value_heads, HEAD_DIM, HEAD_DIM),
            (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
        )
        return raw, view

    tirx_state_raw, tirx_state = make_state_pool()
    reference_state_raw = tirx_state_raw.clone()
    reference_state = reference_state_raw.as_strided(
        (state_slots, num_value_heads, HEAD_DIM, HEAD_DIM),
        (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
    )
    initial_state_raw = tirx_state_raw.clone()

    tirx_out = torch.empty(
        (1, total_tokens, num_value_heads, HEAD_DIM), device=device, dtype=torch.bfloat16
    )

    scale = kwargs.get("scale")
    return {
        "spec": spec,
        "config": dict(kwargs),
        "device": device,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "g_raw": g_raw,
        "beta": beta,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "ssm_state_indices_2d": slots,
        "num_accepted_tokens": num_accepted_tokens,
        "tirx_state_raw": tirx_state_raw,
        "tirx_state": tirx_state,
        "reference_state_raw": reference_state_raw,
        "reference_state": reference_state,
        "initial_state_raw": initial_state_raw,
        "tirx_out": tirx_out,
        "scale": float(scale) if scale is not None else HEAD_DIM**-0.5,
    }


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        case["q"].reshape(-1),
        case["k"].reshape(-1),
        case["v"].reshape(-1),
        case["g_raw"],
        case["beta"].reshape(-1),
        case["tirx_state_raw"],
        case["tirx_out"].reshape(-1),
        case["cu_seqlens"],
        case["ssm_state_indices"],
        case["num_accepted_tokens"],
        float(case["scale"]),
    )


def _variant_name(value_split: int) -> str:
    return f"d128_t6_precomputed_gram_split{value_split}"


def _flashinfer_reference(case: dict[str, Any]) -> torch.Tensor:
    """Run the frozen cake export itself on the reference state pool."""
    from flashinfer.jit.flash_kda_decode import get_flash_kda_decode_module

    device = case["device"]
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == (10, 0):
        target = "sm100f" if torch.version.cuda and torch.version.cuda >= "12.9" else "sm100a"
    elif (major, minor) == (10, 3):
        target = "sm100f"  # non-direct variants are never built for sm103a
    else:
        raise SkipTest(f"no FlashKDA cake export for compute capability {major}.{minor}")

    module = get_flash_kda_decode_module(_variant_name(case["spec"]["VALUE_SPLIT"]), target)
    reference_out = torch.empty_like(case["tirx_out"])
    dummy_f32 = torch.ones(1, device=device, dtype=torch.float32)

    module.run(
        case["q"], case["k"], case["v"], case["g"], case["beta"],
        dummy_f32, dummy_f32,
        case["reference_state"], reference_out,
        case["cu_seqlens"], case["ssm_state_indices"], case["num_accepted_tokens"],
        float(case["scale"]), 0.0,
        int(torch.cuda.current_stream(device).cuda_stream),
    )  # fmt: skip
    torch.cuda.synchronize(device)
    return reference_out


def _torch_reference(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent FP32 oracle of the T=6 delta rule (sequential form)."""
    spec = case["spec"]
    num_seqs = spec["NUM_SEQS"]
    num_value_heads = spec["NUM_VALUE_HEADS"]
    head_ratio = spec["HEAD_RATIO"]
    slot_stride = spec["STATE_SLOT_STRIDE"]

    q = case["q"][0].float()
    k = case["k"][0].float()
    v = case["v"][0].float()
    g = case["g"][0].float()
    beta = case["beta"][0].float()
    slots2d = case["ssm_state_indices_2d"]
    nat = case["num_accepted_tokens"]

    state_raw = case["initial_state_raw"].clone()
    state_slots = state_raw.numel() // slot_stride
    state = state_raw.as_strided(
        (state_slots, num_value_heads, HEAD_DIM, HEAD_DIM),
        (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
    )
    out = torch.zeros(
        (num_seqs * NUM_TOKENS, num_value_heads, HEAD_DIM),
        device=case["device"],
        dtype=torch.float32,
    )

    scale = float(case["scale"])
    for n in range(num_seqs):
        accepted = min(max(int(nat[n].item()) - 1, 0), NUM_TOKENS - 1)
        init_slot = int(slots2d[n, accepted].item())
        if init_slot < 0:
            init_slot = 0
        for hv in range(num_value_heads):
            h = hv // head_ratio
            # FP32 carry across all tokens; only the checkpoint store rounds.
            s = state[init_slot, hv].float()
            for t in range(NUM_TOKENS):
                row = n * NUM_TOKENS + t
                qn = q[row, h] * (torch.rsqrt(q[row, h].pow(2).sum() + L2_EPS) * scale)
                kn = k[row, h] * torch.rsqrt(k[row, h].pow(2).sum() + L2_EPS)
                gamma = torch.exp(g[row, hv])

                decayed = s * gamma.unsqueeze(0)
                delta = (v[row, hv] - decayed @ kn) * beta[row, hv]
                s = decayed + delta.unsqueeze(1) * kn.unsqueeze(0)
                out[row, hv] = s @ qn

                slot = int(slots2d[n, t].item())
                if slot >= 0:
                    state[slot, hv] = s.to(torch.bfloat16)
                else:
                    out[row, hv] = 0.0

    return out.to(torch.bfloat16).unsqueeze(0), state_raw


# The port replicates the source's MMA chain, swizzle and association orders, so
# it should agree with the frozen export to within bf16 rounding.
_RTOL = 2.0**-8
_ATOL = 2.0e-3
# The FP32 oracle is the sequential delta rule; the kernel rounds its MMA
# operands to bf16 -- and at T=6 the WY coefficients themselves come from a
# bf16 Gram product of `k / prefix`, so this band is wider than the T<=4 one and
# is re-measured at the correctness gate rather than inherited.
_ORACLE_RTOL = 2.0**-5
_ORACLE_ATOL = 6.0e-3


def prepare_bench(**kwargs: Any):
    """CPU-compile this workload for same-process GPU execution."""
    from tirx_kernels.runner import prepare_module_bench

    return prepare_module_bench(__name__, kwargs)


def run_test(**kwargs: Any) -> None:
    """Validate one config against the frozen export and an independent oracle."""
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    spec = case["spec"]

    executable = compile_kernel(get_kernel(**kwargs))
    executable(*_tirx_args(case))
    torch.cuda.synchronize(case["device"])

    tirx_out = case["tirx_out"]
    tirx_state = case["tirx_state_raw"].clone()

    # 1. the frozen cake export itself, on an independent state pool
    reference_out = _flashinfer_reference(case)
    torch.testing.assert_close(
        tirx_out.float(),
        reference_out.float(),
        rtol=_RTOL,
        atol=_ATOL,
        msg=lambda m: f"output vs flashinfer cake export (split {spec['VALUE_SPLIT']})\n{m}",
    )
    torch.testing.assert_close(
        tirx_state.float(),
        case["reference_state_raw"].float(),
        rtol=_RTOL,
        atol=_ATOL,
        msg=lambda m: f"state vs flashinfer cake export (split {spec['VALUE_SPLIT']})\n{m}",
    )

    # 2. an independent FP32 oracle of the same recurrence
    oracle_out, oracle_state = _torch_reference(case)
    torch.testing.assert_close(
        tirx_out.float(),
        oracle_out.float(),
        rtol=_ORACLE_RTOL,
        atol=_ORACLE_ATOL,
        msg=lambda m: f"output vs FP32 oracle\n{m}",
    )
    torch.testing.assert_close(
        tirx_state.float(),
        oracle_state.float(),
        rtol=_ORACLE_RTOL,
        atol=_ORACLE_ATOL,
        msg=lambda m: f"state vs FP32 oracle\n{m}",
    )

    # 3. invariants the tolerances cannot express
    num_seqs = spec["NUM_SEQS"]
    slot_stride = spec["STATE_SLOT_STRIDE"]
    payload = spec["NUM_VALUE_HEADS"] * HEAD_DIM * HEAD_DIM
    slots2d = case["ssm_state_indices_2d"]
    initial = case["initial_state_raw"]

    touched: set[int] = set()
    for seq in range(num_seqs):
        for t in range(NUM_TOKENS):
            slot = int(slots2d[seq, t])
            row = seq * NUM_TOKENS + t
            if slot < 0:
                # A padded row writes explicit zeros, so this is exact.
                assert tirx_out[0, row].abs().max().item() == 0.0, (
                    f"padded row {seq} token {t} must have a zeroed output"
                )
            else:
                touched.add(slot)
    n_slots = initial.numel() // slot_stride
    for slot in range(min(n_slots, num_seqs * NUM_TOKENS + 2)):
        if slot in touched:
            continue
        lo, hi = slot * slot_stride, (slot + 1) * slot_stride
        assert torch.equal(initial[lo:hi], tirx_state[lo:hi]), (
            f"untouched state slot {slot} was modified"
        )
    if slot_stride > payload:
        for slot in sorted(touched)[:4]:
            lo = slot * slot_stride + payload
            hi = (slot + 1) * slot_stride
            assert torch.equal(initial[lo:hi], tirx_state[lo:hi]), (
                f"inter-slot padding after slot {slot} was modified"
            )


def run_bench(
    *,
    warmup: float | None = None,
    repeat: float | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Time the port against the frozen cake export on identical inputs."""
    rounds = int(kwargs.pop("rounds", 5))
    cooldown_s = float(kwargs.pop("cooldown_s", 1.0))
    from tirx_kernels.runner import bench, compile_kernel_lazy

    case = prepare_data(**kwargs)
    executable = compile_kernel_lazy(lambda: get_kernel(**kwargs))
    args = _tirx_args(case)

    # Validate once, outside the timed region. Both sides mutate their own state
    # pool in place, so this also proves the pools stayed independent.
    executable(*args)
    reference_out = _flashinfer_reference(case)
    torch.cuda.synchronize(case["device"])
    torch.testing.assert_close(
        case["tirx_out"].float(), reference_out.float(), rtol=_RTOL, atol=_ATOL
    )
    torch.testing.assert_close(
        case["tirx_state_raw"].float(), case["reference_state_raw"].float(), rtol=_RTOL, atol=_ATOL
    )

    def flashinfer_builder():
        # The nvcc JIT build and warmup both happen here, outside the timing.
        for _ in range(2):
            _flashinfer_reference(case)
        torch.cuda.synchronize(case["device"])

        def launch():
            _flashinfer_reference(case)

        return launch

    return bench(
        {"tirx": lambda: executable(*args)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashinfer_cake": flashinfer_builder},
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
