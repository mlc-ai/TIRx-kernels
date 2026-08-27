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
  allocate through Kern's shared-memory pool -- which is also what the source does
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

from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K

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
    out = K.local_scalar("uint32")
    K.ptx.shfl_sync.idx.b32(
        out, K.reinterpret("uint32", value), K.uint32(0), K.uint32(31), K.uint32(0xFFFFFFFF)
    )
    return K.reinterpret("int32", out)


def _named_bar_sync(bar_id: int, threads: int):
    """``barrier.sync <id>, <count>`` -- blocks until `count` threads arrive."""
    K.ptx["barrier.sync"](bar_id, threads)


def _named_bar_arrive(bar_id: int, threads: int):
    """``barrier.arrive <id>, <count>`` -- releases, does NOT block or acquire."""
    K.ptx["barrier.arrive"](bar_id, threads)


def _mma_zero_b(acc, a, b, b0: int):
    """``mma.sync...m16n8k16`` with an explicit zero C, B taken at ``b[b0:b0+2]``.

    The gram block issues two products from one pair of ``ldmatrix`` results, so
    unlike the shared helper this one has to select the B half.
    """
    K.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
            acc[0], acc[1], acc[2], acc[3],
            a[0], a[1], a[2], a[3],
            b[b0], b[b0 + 1],
            *_t2._MMA_ZERO_C,
        )  # fmt: skip


def _mma_acc_b(acc, a, b, b0: int):
    """Same, accumulating: C aliases D, matching the source's `+f` tied registers."""
    K.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
            acc[0], acc[1], acc[2], acc[3],
            a[0], a[1], a[2], a[3],
            b[b0], b[b0 + 1],
            acc[0], acc[1], acc[2], acc[3],
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
    from tirx_kernels.runner import hardware_num_sms

    return hardware_num_sms()


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


def _make_flashkda_decode_t6_gram(spec: dict[str, Any]):
    """Trace the source schedule as the unique K-owned final body."""
    NUM_SEQS = spec["NUM_SEQS"]
    NUM_HEADS = spec["NUM_HEADS"]
    NUM_VALUE_HEADS = spec["NUM_VALUE_HEADS"]
    HEAD_RATIO = spec["HEAD_RATIO"]
    VALUE_SPLIT = spec["VALUE_SPLIT"]
    STATE_SLOT_STRIDE = spec["STATE_SLOT_STRIDE"]
    GATE_TOKEN_STRIDE = spec["GATE_TOKEN_STRIDE"]
    Q_ELEMENTS = spec["Q_ELEMENTS"]
    V_ELEMENTS = spec["V_ELEMENTS"]
    GATE_ELEMENTS = spec["GATE_ELEMENTS"]
    BETA_ELEMENTS = spec["BETA_ELEMENTS"]
    STATE_ELEMENTS = spec["STATE_ELEMENTS"]
    CU_SEQLENS_ELEMENTS = spec["CU_SEQLENS_ELEMENTS"]
    STATE_INDEX_ELEMENTS = spec["STATE_INDEX_ELEMENTS"]
    NAT_ELEMENTS = spec["NAT_ELEMENTS"]
    THREADS = spec["THREADS"]
    ROWS_PER_CTA = spec["ROWS_PER_CTA"]
    MMA_WARPS = spec["MMA_WARPS"]
    ROW_GROUPS = spec["ROW_GROUPS"]
    ROWS_PER_GROUP = spec["ROWS_PER_GROUP"]
    GRAM_WARP = spec["GRAM_WARP"]
    GRAM_SYNC_ALL = spec["GRAM_SYNC_ALL"]
    SU_SYNC_CTA = spec["SU_SYNC_CTA"]
    SMEM_TOTAL = spec["SMEM_TOTAL"]
    OFF_SSTATE0 = spec["OFF_SSTATE0"]
    OFF_SSTATE1 = spec["OFF_SSTATE1"]
    OFF_SVEC = spec["OFF_SVEC"]
    OFF_SK = spec["OFF_SK"]
    OFF_SD = spec["OFF_SD"]
    OFF_SBETA = spec["OFF_SBETA"]
    OFF_SSLOT = spec["OFF_SSLOT"]
    OFF_STOKEN = spec["OFF_STOKEN"]
    OFF_SINIT = spec["OFF_SINIT"]
    OFF_SL = spec["OFF_SL"]
    OFF_SR = spec["OFF_SR"]
    OFF_SU = spec["OFF_SU"]
    OFF_SGRAMA0 = spec["OFF_SGRAMA0"]
    OFF_SGRAMA1 = spec["OFF_SGRAMA1"]

    @K.kernel(warps=THREADS // 32, arch="sm_100a", grid=False)
    def _flashkda_decode_t6_gram(
        q: K.gptr[K.bf16, (Q_ELEMENTS,)],
        k: K.gptr[K.bf16, (Q_ELEMENTS,)],
        v: K.gptr[K.bf16, (V_ELEMENTS,)],
        g: K.gptr[K.bf16, (GATE_ELEMENTS,)],
        beta: K.gptr[K.bf16, (BETA_ELEMENTS,)],
        state: K.gptr[K.bf16, (STATE_ELEMENTS,)],
        out: K.gptr[K.bf16, (V_ELEMENTS,)],
        cu: K.gptr[K.i32, (CU_SEQLENS_ELEMENTS,)],
        ssm_idx: K.gptr[K.i32, (STATE_INDEX_ELEMENTS,)],
        nat: K.gptr[K.i32, (NAT_ELEMENTS,)],
        scale: K.f32,
    ):
        smem = K.smem_pool()
        arena = smem.alloc((SMEM_TOTAL,), K.u8, align=1024)
        smem.commit(SMEM_TOTAL)

        # --- work decomposition and lane roles (:142-178) ----------------------
        work, n = K.cta_id([NUM_VALUE_HEADS * VALUE_SPLIT, NUM_SEQS])
        tid = K.thread_id()
        warp = _make_warp_uniform(tid // 32)  # == token index in A and C'
        lane = tid % 32
        value_tile = K.local_scalar(K.i32)
        hv = K.local_scalar(K.i32)
        query_head = K.local_scalar(K.i32)
        lane_quad = K.local_scalar(K.i32)
        frag_row = K.local_scalar(K.i32)
        quad_base = K.local_scalar(K.i32)
        group = K.local_scalar(K.i32)
        lane_group = K.local_scalar(K.i32)
        k_start = K.local_scalar(K.i32)
        elem_start = K.local_scalar(K.i32)
        tile_row_base = K.local_scalar(K.i32)
        owned_row_base = K.local_scalar(K.i32)
        token_base = K.local_scalar(K.i32)
        seq_len = K.local_scalar(K.i32)
        K.assign(value_tile, work % VALUE_SPLIT)
        K.assign(hv, work // VALUE_SPLIT)
        K.assign(query_head, hv // HEAD_RATIO)
        K.assign(lane_quad, lane % 4)
        K.assign(frag_row, lane // 4)
        K.assign(quad_base, lane - lane_quad)
        K.assign(group, tid // 16)
        K.assign(lane_group, tid % 16)
        K.assign(k_start, lane_group * 8)
        K.assign(elem_start, lane * 4)
        K.assign(tile_row_base, value_tile * ROWS_PER_CTA)
        K.assign(owned_row_base, group * ROWS_PER_GROUP)
        K.assign(token_base, _load_i32(cu, n))
        K.assign(seq_len, _load_i32(cu, n + 1) - token_base)

        r_q = K.alloc_local((4,), "float32")
        r_k = K.alloc_local((4,), "float32")
        r_d = K.alloc_local((4,), "float32")

        # =======================================================================
        # Phase A: token preprocess, warp <-> token  (:180-290)
        # =======================================================================
        # Identical to the ported T=4 phase A apart from the token count, the
        # ssm_state_indices stride and the nat clamp ceiling.
        with K.If(warp < NUM_TOKENS), K.Then():
            token = warp
            active_token = token < seq_len
            token_pos = K.local_scalar(K.i32)
            qk_base = K.local_scalar(K.i32)
            gate_base = K.local_scalar(K.i32)
            K.assign(token_pos, K.if_then_else(active_token, token_base + token, 0))
            K.assign(qk_base, (token_pos * NUM_HEADS + query_head) * HEAD_DIM + elem_start)
            K.assign(gate_base, token_pos * GATE_TOKEN_STRIDE + hv * HEAD_DIM + elem_start)

            q_words = _load_u32x2(q, qk_base)
            k_words = _load_u32x2(k, qk_base)
            g_words = _load_u32x2(g, gate_base)
            for pair in range(2):
                K.ptx.mov.b32(r_q[2 * pair], _widen_lo(q_words[pair]))
                K.ptx.mov.b32(r_q[2 * pair + 1], _widen_hi(q_words[pair]))
                K.ptx.mov.b32(r_k[2 * pair], _widen_lo(k_words[pair]))
                K.ptx.mov.b32(r_k[2 * pair + 1], _widen_hi(k_words[pair]))
                K.ptx.mov.b32(r_d[2 * pair], _widen_lo(g_words[pair]))
                K.ptx.mov.b32(r_d[2 * pair + 1], _widen_hi(g_words[pair]))

            # Index-ordered accumulation (:241-244); the first term has a zero addend
            # and the two chains interleave, because the source fuses them in one loop.
            q_sq = K.local_scalar(K.f32)
            k_sq = K.local_scalar(K.f32)
            K.assign(
                q_sq,
                _fma(
                    r_q[3],
                    r_q[3],
                    _fma(
                        r_q[2], r_q[2], _fma(r_q[1], r_q[1], _fma(r_q[0], r_q[0], K.float32(0.0)))
                    ),
                ),
            )
            K.assign(
                k_sq,
                _fma(
                    r_k[3],
                    r_k[3],
                    _fma(
                        r_k[2], r_k[2], _fma(r_k[1], r_k[1], _fma(r_k[0], r_k[0], K.float32(0.0)))
                    ),
                ),
            )
            # Two sequential full-warp butterflies, not interleaved (:245-254).
            for off in range(5):
                K.assign(q_sq, _add(q_sq, _shfl_bfly(q_sq, 16 >> off)))
            for off in range(5):
                K.assign(k_sq, _add(k_sq, _shfl_bfly(k_sq, 16 >> off)))
            q_norm = _mul(_rsqrt(_add(q_sq, K.float32(L2_EPS))), scale)
            k_norm = _rsqrt(_add(k_sq, K.float32(L2_EPS)))

            # GATE_KIND == 0: `g` already holds log(gamma), so one exp per element.
            k_pub = K.alloc_local((4,), "uint32")
            d_pub = K.alloc_local((4,), "uint32")
            for i in range(4):
                K.ptx.mov.b32(r_q[i], _mul(r_q[i], q_norm))
                K.ptx.mov.b32(r_k[i], _mul(r_k[i], k_norm))
                K.ptx.mov.b32(r_d[i], _expf(r_d[i]))
                K.ptx.mov.b32(k_pub[i], K.reinterpret("uint32", r_k[i]))
                K.ptx.mov.b32(d_pub[i], K.reinterpret("uint32", r_d[i]))
            # Four contiguous f32 per lane: one 16-byte shared store each, not four
            # scalar ones (:269-270 lower to 2 st.shared.v4.b32).
            _st_shared_u32x4(arena, OFF_SK + (token * HEAD_DIM + elem_start) * 4, k_pub)
            _st_shared_u32x4(arena, OFF_SD + (token * HEAD_DIM + elem_start) * 4, d_pub)

            with K.If(lane == 0), K.Then():
                raw_slot = _load_i32(ssm_idx, n * NUM_TOKENS + token)
                _st_shared_i32(
                    arena, OFF_SSLOT + token * 4, K.if_then_else(active_token, raw_slot, -1)
                )
                _st_shared_i32(arena, OFF_STOKEN + token * 4, token_pos)
                _st_shared_f32(
                    arena,
                    OFF_SBETA + token * 4,
                    _load_bf16_f32(beta, token_pos * NUM_VALUE_HEADS + hv),
                )
                with K.If(token == 0), K.Then():
                    # nat picks the initial checkpoint slot; at T=5 the clamp ceiling
                    # is 4 and both edges are reachable (:279-287).
                    accepted = K.min(K.max(_load_i32(nat, n) - 1, 0), NUM_TOKENS - 1)
                    initial_slot = _load_i32(ssm_idx, n * NUM_TOKENS + accepted)
                    _st_shared_i32(arena, OFF_SINIT, K.max(initial_slot, 0))

        K.cuda.cta_sync()

        # =======================================================================
        # Phase C': sVec columns and the gram operand  (:293-339)
        # =======================================================================
        # Runs BEFORE the state gather at T=5 -- the reverse of the T=4 order.
        with K.If(warp < NUM_TOKENS), K.Then():
            token_c = warp
            for i in range(4):
                k_idx = elem_start + i
                prefix = K.local_scalar(K.f32, init=K.float32(1.0))
                # Scalar loads: the walk is across tokens at a fixed key, so
                # consecutive iterations are 512 B apart (:296-302).
                for j in range(NUM_TOKENS):
                    with K.If(token_c >= j), K.Then():
                        K.assign(
                            prefix,
                            _mul(
                                prefix, _ld_shared_f32(arena, OFF_SD + (j * HEAD_DIM + k_idx) * 4)
                            ),
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
                with K.If(k_idx < 64):
                    with K.Then():
                        _st_shared_b16(
                            arena, OFF_SGRAMA0 + _swz(token_c * 128 + k_idx * 2), deflated
                        )
                    with K.Else():
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
        with K.If(warp < NUM_TOKENS), K.Then():
            if GRAM_SYNC_ALL == 1:
                # split4 hoists the wait out of the branch: all five token warps
                # block, and the arrive arm below is `else if (0)` -- dead (S4:340-344).
                _named_bar_sync(1, NUM_TOKENS * 32)
            with K.If(warp == GRAM_WARP):
                with K.Then():
                    if GRAM_SYNC_ALL == 0:
                        _named_bar_sync(1, NUM_TOKENS * 32)
                    gram_a = K.alloc_local((4,), "uint32", align=4)
                    gram_b = K.alloc_local((4,), "uint32", align=4)
                    gram_k_acc = K.alloc_local((4,), "float32", align=4)
                    gram_q_acc = K.alloc_local((4,), "float32", align=4)
                    for gram_half in range(2):
                        for gram_step in range(4):
                            gram_k = gram_step * 16
                            global_gram_k = gram_half * 64 + gram_k
                            a_off = (lane % 16 * 64 + (gram_k + lane // 16 * 8)) * 2
                            gram_a_off = OFF_SGRAMA0 if gram_half == 0 else OFF_SGRAMA1
                            _ldmatrix_x4(arena, gram_a_off + _swz(a_off), gram_a, False)
                            # Same sVec address formula as phase D's operand (:360-361
                            # vs :454-455); frags 0,1 are columns 0..7 (the k side) and
                            # 2,3 are columns 8..15 (the q side).
                            _ldmatrix_x4(
                                arena,
                                OFF_SVEC
                                + _swz(((global_gram_k + lane % 16) * 16 + lane // 16 * 8) * 2),
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
                    source_token = frag_row
                    target0 = lane_quad * 2
                    target1 = target0 + 1
                    with K.If(source_token < NUM_TOKENS), K.Then():
                        beta_source = _ld_shared_f32(arena, OFF_SBETA + source_token * 4)
                        with K.If(K.And(source_token < target0, target0 < NUM_TOKENS)), K.Then():
                            _st_shared_f32(
                                arena,
                                OFF_SL + (target0 * NUM_TOKENS + source_token) * 4,
                                _mul(beta_source, gram_k_acc[0]),
                            )
                        with K.If(K.And(source_token < target1, target1 < NUM_TOKENS)), K.Then():
                            _st_shared_f32(
                                arena,
                                OFF_SL + (target1 * NUM_TOKENS + source_token) * 4,
                                _mul(beta_source, gram_k_acc[1]),
                            )
                        with K.If(K.And(source_token <= target0, target0 < NUM_TOKENS)), K.Then():
                            _st_shared_f32(
                                arena,
                                OFF_SR + (target0 * NUM_TOKENS + source_token) * 4,
                                _mul(beta_source, gram_q_acc[0]),
                            )
                        with K.If(K.And(source_token <= target1, target1 < NUM_TOKENS)), K.Then():
                            _st_shared_f32(
                                arena,
                                OFF_SR + (target1 * NUM_TOKENS + source_token) * 4,
                                _mul(beta_source, gram_q_acc[1]),
                            )
                if GRAM_SYNC_ALL == 0:
                    with K.Else():
                        _named_bar_arrive(1, NUM_TOKENS * 32)

        # =======================================================================
        # Phase B: state gather and sState stage  (:410-446)
        # =======================================================================
        init_slot = _ld_shared_i32(arena, OFF_SINIT)
        head_base = K.cast(init_slot, "int64") * K.cast(STATE_SLOT_STRIDE, "int64") + K.cast(
            hv * HEAD_DIM * HEAD_DIM, "int64"
        )
        hist = K.alloc_local((ROWS_PER_GROUP * 8,), "float32")
        with K.If(group < ROW_GROUPS), K.Then():
            for row_local in range(ROWS_PER_GROUP):
                # Two distinct indices: row_l is CTA-local and addresses sState;
                # tile_row_base + row_l is the global row of `state` (:414-415).
                row_l = owned_row_base + row_local
                pack = _load_u32x4(
                    state, head_base + K.cast((tile_row_base + row_l) * HEAD_DIM + k_start, "int64")
                )
                for pr in range(4):
                    K.ptx.mov.b32(hist[row_local * 8 + 2 * pr], _widen_lo(pack[pr]))
                    K.ptx.mov.b32(hist[row_local * 8 + 2 * pr + 1], _widen_hi(pack[pr]))
                # An if/ELSE selecting the destination half, not a guard: lanes 8..15
                # stage keys 64..127 into sState1 (:438-442). The bf16 bits go to
                # shared unmodified; the swizzle is on the byte offset.
                with K.If(lane_group < 8):
                    with K.Then():
                        _st_shared_u32x4(arena, OFF_SSTATE0 + _swz(row_l * 128 + k_start * 2), pack)
                    with K.Else():
                        _st_shared_u32x4(
                            arena, OFF_SSTATE1 + _swz(row_l * 128 + (k_start - 64) * 2), pack
                        )

        K.cuda.cta_sync()
        # Besides sState and sL/sR, this is the sVec publish edge for every MMA warp
        # except the gram warp: `barrier.arrive` releases but does not acquire, so an
        # arriving token warp has synchronized with the others only here.

        # =======================================================================
        # Phase D: the MMA chain, warp <-> 16 value rows  (:448-490)
        # =======================================================================
        # Two issues per step at T=5: the k side needs sVec columns 0..4 and the q
        # side 8..12, which no single n=8 tile covers. `mma_acc_c` and
        # `vec_frag[2],[3]` were dead at every T <= 4.
        acc = K.alloc_local((4,), "float32", align=4)
        acc_c = K.alloc_local((4,), "float32", align=4)
        with K.If(warp < MMA_WARPS), K.Then():
            vec_frag = K.alloc_local((4,), "uint32", align=4)
            state_frag = K.alloc_local((4,), "uint32", align=4)
            for state_half in range(2):
                for mma_step in range(4):
                    mma_k = mma_step * 16
                    global_k = state_half * 64 + mma_k
                    _ldmatrix_x4(
                        arena,
                        OFF_SVEC + _swz((global_k + lane % 16) * 32 + lane // 16 * 16),
                        vec_frag,
                        True,
                    )
                    state_offset = OFF_SSTATE0 if state_half == 0 else OFF_SSTATE1
                    _ldmatrix_x4(
                        arena,
                        state_offset
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
        u_lo = K.alloc_local((NUM_TOKENS,), "float32")
        u_hi = K.alloc_local((NUM_TOKENS,), "float32")
        # hc_* is declared out here because phase F consumes it from its own block;
        # ha_* never leaves phase E.
        hc_lo = K.alloc_local((NUM_TOKENS,), "float32")
        hc_hi = K.alloc_local((NUM_TOKENS,), "float32")
        with K.If(warp < MMA_WARPS), K.Then():
            # 20 broadcasts in the source's emission order (:491-531). m16n8k16 puts
            # columns 2q, 2q+1 of row frag_row in lane quad_base+q's acc[0],[1] and
            # rows +8 in acc[2],[3], so token t lives at (quad_base + t//2, t%2).
            # ha_* is the k side (the solve), hc_* the q side (the outputs).
            ha_lo = K.alloc_local((NUM_TOKENS,), "float32")
            ha_hi = K.alloc_local((NUM_TOKENS,), "float32")
            for t in range(4):
                K.ptx.mov.b32(ha_lo[t], _shfl_idx(acc[t % 2], quad_base + t // 2))
            for t in range(4):
                K.ptx.mov.b32(ha_hi[t], _shfl_idx(acc[2 + t % 2], quad_base + t // 2))
            K.ptx.mov.b32(ha_lo[4], _shfl_idx(acc[0], quad_base + 2))
            K.ptx.mov.b32(ha_hi[4], _shfl_idx(acc[2], quad_base + 2))
            for t in range(NUM_TOKENS - 1):
                K.ptx.mov.b32(hc_lo[t], _shfl_idx(acc_c[t % 2], quad_base + t // 2))
            for t in range(NUM_TOKENS - 1):
                K.ptx.mov.b32(hc_hi[t], _shfl_idx(acc_c[2 + t % 2], quad_base + t // 2))

            # Token 5 needs NO shuffle (:532-537). The (quad_base + t//2, acc[t%2])
            # map puts it on lane quad_base + 2 -- which is the only lane that
            # consumes it, since both the solve below and phase F's token-5 tail run
            # under `lane_quad == 2`. So the source reads the local registers, and
            # the broadcast count stays at 20 rather than 24. These four assignments
            # sit OUTSIDE the `lane_quad == 2` guard: every lane executes them, only
            # lane_quad 2's values are meaningful.
            K.ptx.mov.b32(ha_lo[NUM_TOKENS - 1], acc[1])
            K.ptx.mov.b32(ha_hi[NUM_TOKENS - 1], acc[3])
            K.ptx.mov.b32(hc_lo[NUM_TOKENS - 1], acc_c[1])
            K.ptx.mov.b32(hc_hi[NUM_TOKENS - 1], acc_c[3])

            with K.If(lane_quad == 2), K.Then():
                row_lo = warp * 16 + frag_row
                row_hi = row_lo + 8
                for t in range(NUM_TOKENS):
                    base_t = (
                        _ld_shared_i32(arena, OFF_STOKEN + t * 4) * NUM_VALUE_HEADS + hv
                    ) * HEAD_DIM
                    solved_lo = K.local_scalar(K.f32)
                    solved_hi = K.local_scalar(K.f32)
                    K.assign(
                        solved_lo,
                        _sub(_load_bf16_f32(v, base_t + tile_row_base + row_lo), ha_lo[t]),
                    )
                    K.assign(
                        solved_hi,
                        _sub(_load_bf16_f32(v, base_t + tile_row_base + row_hi), ha_hi[t]),
                    )
                    for prev in range(t):
                        lts = _ld_shared_f32(arena, OFF_SL + (t * NUM_TOKENS + prev) * 4)
                        K.assign(solved_lo, _sub(solved_lo, _mul(lts, u_lo[prev])))
                        K.assign(solved_hi, _sub(solved_hi, _mul(lts, u_hi[prev])))
                    K.ptx.mov.b32(u_lo[t], solved_lo)
                    K.ptx.mov.b32(u_hi[t], solved_hi)

            # The solve runs on lane_quad == 2 but lane_quad == 3 also writes output,
            # so the residuals cross the quad (:554-560).
            for t in range(NUM_TOKENS):
                K.ptx.mov.b32(u_lo[t], _shfl_idx(u_lo[t], quad_base + 2))
                K.ptx.mov.b32(u_hi[t], _shfl_idx(u_hi[t], quad_base + 2))

        # =======================================================================
        # Phase F: the outputs  (:568-670)
        # =======================================================================
        # The bases come from hc_* (the q-side MMA), not from acc: the source
        # assigns acc[0..3] and then unconditionally overwrites (:567-582). The
        # lane_quad == 3 remap uses STATIC indices, as the source does -- indexing
        # hc_* by a runtime token would spill the register array to local memory.
        with K.If(K.And(warp < MMA_WARPS, lane_quad >= 2)), K.Then():
            token0 = (lane_quad - 2) * 2
            token1 = token0 + 1
            row_lo_f = warp * 16 + frag_row
            row_hi_f = row_lo_f + 8
            out0_lo = K.local_scalar(K.f32)
            out1_lo = K.local_scalar(K.f32)
            out0_hi = K.local_scalar(K.f32)
            out1_hi = K.local_scalar(K.f32)
            K.assign(out0_lo, hc_lo[0])
            K.assign(out1_lo, hc_lo[1])
            K.assign(out0_hi, hc_hi[0])
            K.assign(out1_hi, hc_hi[1])
            with K.If(lane_quad == 3), K.Then():
                K.assign(out0_lo, hc_lo[2])
                K.assign(out1_lo, hc_lo[3])
                K.assign(out0_hi, hc_hi[2])
                K.assign(out1_hi, hc_hi[3])
            for src in range(NUM_TOKENS):
                residual_lo = u_lo[src]
                residual_hi = u_hi[src]
                coef0 = K.local_scalar(K.f32)
                coef1 = K.local_scalar(K.f32)
                K.assign(coef0, K.float32(0.0))
                K.assign(coef1, K.float32(0.0))
                # The masked-out coefficient is a real zero-operand fma, not a
                # skipped iteration (:585-594).
                with K.If(token0 >= src), K.Then():
                    K.assign(coef0, _ld_shared_f32(arena, OFF_SR + (token0 * NUM_TOKENS + src) * 4))
                with K.If(token1 >= src), K.Then():
                    K.assign(coef1, _ld_shared_f32(arena, OFF_SR + (token1 * NUM_TOKENS + src) * 4))
                K.assign(out0_lo, _fma(coef0, residual_lo, out0_lo))
                K.assign(out1_lo, _fma(coef1, residual_lo, out1_lo))
                K.assign(out0_hi, _fma(coef0, residual_hi, out0_hi))
                K.assign(out1_hi, _fma(coef1, residual_hi, out1_hi))

            for half in range(2):
                token_o = token0 if half == 0 else token1
                o_lo = out0_lo if half == 0 else out1_lo
                o_hi = out0_hi if half == 0 else out1_hi
                active_o = _ld_shared_i32(arena, OFF_SSLOT + token_o * 4) >= 0
                base_o = (
                    _ld_shared_i32(arena, OFF_STOKEN + token_o * 4) * NUM_VALUE_HEADS + hv
                ) * HEAD_DIM + tile_row_base
                # A padded row writes EXPLICIT zeros; the upstream test asserts them
                # bit-exactly, so this is not an "unwritten" path.
                _store_f32_as_bf16(out, base_o + row_lo_f, o_lo, active_o)
                _store_f32_as_bf16(out, base_o + row_hi_f, o_hi, active_o)
                _store_f32_as_bf16(out, base_o + row_lo_f, K.float32(0.0), K.Not(active_o))
                _store_f32_as_bf16(out, base_o + row_hi_f, K.float32(0.0), K.Not(active_o))

            # Tokens 4 and 5 have no partner lane, so lane_quad == 2 writes BOTH
            # tails at T=6 (:628-668) -- eight output stores against quad 3's four.
            with K.If(lane_quad == 2), K.Then():
                # Token 4 (:629-650). Its coefficient row is CLAMPED: sR[29] is
                # (target 4, source 5), which the gram block never writes -- its
                # predicate is `source <= target` -- so it is in-region but
                # uninitialized, and only the `src <= 4` mask keeps it out of the
                # result. The source's unconditional read at :633 is dead.
                out4_lo = K.local_scalar(K.f32)
                out4_hi = K.local_scalar(K.f32)
                K.assign(out4_lo, hc_lo[NUM_TOKENS - 2])
                K.assign(out4_hi, hc_hi[NUM_TOKENS - 2])
                for src4 in range(NUM_TOKENS):
                    coef4 = K.float32(0.0)
                    if src4 <= NUM_TOKENS - 2:
                        coef4 = _ld_shared_f32(
                            arena, OFF_SR + ((NUM_TOKENS - 2) * NUM_TOKENS + src4) * 4
                        )
                    K.assign(out4_lo, _fma(coef4, u_lo[src4], out4_lo))
                    K.assign(out4_hi, _fma(coef4, u_hi[src4], out4_hi))
                active4 = _ld_shared_i32(arena, OFF_SSLOT + (NUM_TOKENS - 2) * 4) >= 0
                base4 = (
                    _ld_shared_i32(arena, OFF_STOKEN + (NUM_TOKENS - 2) * 4) * NUM_VALUE_HEADS + hv
                ) * HEAD_DIM + tile_row_base
                _store_f32_as_bf16(out, base4 + row_lo_f, out4_lo, active4)
                _store_f32_as_bf16(out, base4 + row_hi_f, out4_hi, active4)
                _store_f32_as_bf16(out, base4 + row_lo_f, K.float32(0.0), K.Not(active4))
                _store_f32_as_bf16(out, base4 + row_hi_f, K.float32(0.0), K.Not(active4))

                # Token 5 (:651-667). No clamp: the last target accepts every
                # source, so sR[30..35] are all live.
                out5_lo = K.local_scalar(K.f32)
                out5_hi = K.local_scalar(K.f32)
                K.assign(out5_lo, hc_lo[NUM_TOKENS - 1])
                K.assign(out5_hi, hc_hi[NUM_TOKENS - 1])
                for src5 in range(NUM_TOKENS):
                    coef5 = _ld_shared_f32(
                        arena, OFF_SR + ((NUM_TOKENS - 1) * NUM_TOKENS + src5) * 4
                    )
                    K.assign(out5_lo, _fma(coef5, u_lo[src5], out5_lo))
                    K.assign(out5_hi, _fma(coef5, u_hi[src5], out5_hi))
                active5 = _ld_shared_i32(arena, OFF_SSLOT + (NUM_TOKENS - 1) * 4) >= 0
                base5 = (
                    _ld_shared_i32(arena, OFF_STOKEN + (NUM_TOKENS - 1) * 4) * NUM_VALUE_HEADS + hv
                ) * HEAD_DIM + tile_row_base
                _store_f32_as_bf16(out, base5 + row_lo_f, out5_lo, active5)
                _store_f32_as_bf16(out, base5 + row_hi_f, out5_hi, active5)
                _store_f32_as_bf16(out, base5 + row_lo_f, K.float32(0.0), K.Not(active5))
                _store_f32_as_bf16(out, base5 + row_hi_f, K.float32(0.0), K.Not(active5))

        # =======================================================================
        # Phase G: publish sU  (:671-684)
        # =======================================================================
        with K.If(warp < MMA_WARPS), K.Then():
            with K.If(lane_quad == 2), K.Then():
                row_lo_g = warp * 16 + frag_row
                for t in range(NUM_TOKENS):
                    _st_shared_f32(arena, OFF_SU + (t * ROWS_PER_CTA + row_lo_g) * 4, u_lo[t])
                    _st_shared_f32(arena, OFF_SU + (t * ROWS_PER_CTA + row_lo_g + 8) * 4, u_hi[t])
            if SU_SYNC_CTA == 0:
                # Warp w produces rows [16w, 16w+16), exactly the rows its own groups
                # consume in phase H, so a warp barrier suffices (:651-653).
                K.cuda.warp_sync()
        if SU_SYNC_CTA == 1:
            # split8 has ONE MMA warp producing sU for eight consuming groups, so the
            # generator emits a CTA barrier -- and it sits OUTSIDE the warp guard
            # (S8:651 closes it, S8:652-654 follows). Keeping it inside would have
            # one warp of five arrive at a CTA barrier and the kernel would hang.
            K.cuda.cta_sync()

        # =======================================================================
        # Phase H: recurrence and checkpoints  (:798-834)
        # =======================================================================
        words_w = K.alloc_local((4,), "uint32")
        sd_t = K.alloc_local((8,), "float32")
        sk_t = K.alloc_local((8,), "float32")
        with K.If(group < ROW_GROUPS), K.Then():
            for t in range(NUM_TOKENS):
                slot_t = _ld_shared_i32(arena, OFF_SSLOT + t * 4)
                beta_t = _ld_shared_f32(arena, OFF_SBETA + t * 4)
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
                with K.If(slot_t >= 0):
                    with K.Then():
                        for row_local in range(ROWS_PER_GROUP):
                            row_h = owned_row_base + row_local
                            update = _mul(
                                _ld_shared_f32(arena, OFF_SU + (t * ROWS_PER_CTA + row_h) * 4),
                                beta_t,
                            )
                            for i in range(8):
                                # `hist*sD + update*sK` (:782): the compiler contracts the
                                # FIRST product and rounds update*sK.
                                K.ptx.mov.b32(
                                    hist[row_local * 8 + i],
                                    _fma(hist[row_local * 8 + i], sd_t[i], _mul(update, sk_t[i])),
                                )
                            for pr in range(4):
                                K.ptx.mov.b32(
                                    words_w[pr],
                                    _pack_bf16x2(
                                        hist[row_local * 8 + 2 * pr + 1],
                                        hist[row_local * 8 + 2 * pr],
                                    ),
                                )
                            _store_u32x4(
                                state,
                                K.cast(slot_t, "int64") * K.cast(STATE_SLOT_STRIDE, "int64")
                                + K.cast(
                                    hv * HEAD_DIM * HEAD_DIM
                                    + (tile_row_base + row_h) * HEAD_DIM
                                    + k_start,
                                    "int64",
                                ),
                                words_w,
                            )
                    with K.Else():
                        for row_local_p in range(ROWS_PER_GROUP):
                            row_p = owned_row_base + row_local_p
                            update_p = _mul(
                                _ld_shared_f32(arena, OFF_SU + (t * ROWS_PER_CTA + row_p) * 4),
                                beta_t,
                            )
                            for i in range(8):
                                K.ptx.mov.b32(
                                    hist[row_local_p * 8 + i],
                                    _fma(
                                        hist[row_local_p * 8 + i], sd_t[i], _mul(update_p, sk_t[i])
                                    ),
                                )

    return _flashkda_decode_t6_gram


def get_kernel(**kwargs: Any):
    """Return the specialized FlashKDA cake T=6 gram decode PrimFunc."""
    return _make_flashkda_decode_t6_gram(_specialization(kwargs)).func


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


# The port replicates the source's MMA chain, swizzle and association orders, so
# it should agree with the frozen export to within bf16 rounding.
_RTOL = 2.0**-8
_ATOL = 2.0e-3


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


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

    # 1. the frozen cake export (the arbiter) itself, on an independent state pool
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

    # 2. invariants the tolerances cannot express
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


def run_gpu(
    prepared,
    *,
    warmup: float | None = None,
    repeat: float | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Time the port against the frozen cake export on identical inputs."""
    config = dict(prepared["config"])
    config.update(kwargs)
    kwargs = config
    executable = prepared["executable"]
    rounds = int(kwargs.pop("rounds", 5))
    cooldown_s = float(kwargs.pop("cooldown_s", 1.0))
    from tirx_kernels.runner import bench

    case = prepare_data(**kwargs)
    args = _tirx_args(case)

    def flashinfer_builder():
        # Validate once outside the timed region. Both sides mutate independent state pools.
        executable(*args)
        reference_out = _flashinfer_reference(case)
        torch.cuda.synchronize(case["device"])
        torch.testing.assert_close(
            case["tirx_out"].float(), reference_out.float(), rtol=_RTOL, atol=_ATOL
        )
        torch.testing.assert_close(
            case["tirx_state_raw"].float(),
            case["reference_state_raw"].float(),
            rtol=_RTOL,
            atol=_ATOL,
        )
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


def run_bench(
    *,
    warmup: float | None = None,
    repeat: float | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    prepared = prepare_bench(**config)
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
