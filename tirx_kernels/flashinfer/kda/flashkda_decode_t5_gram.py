# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400),
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's FlashKDA "cake" T=5 coefficient-gram decode kernel.

Source: ``csrc/kda/flashkda_decode_d128_t5_precomputed_gram_split{1,2,4,8}.cu``,
symbol ``kernel_flashinfer_recurrent_kda_wy_vtile_short`` -- the same frozen
template family as the ported T=2/T=3/T=4 siblings, re-instantiated at
TOKENS=5 with ``coefficient_gram`` enabled.

Dispatch reaches these bodies whenever ``num_tokens == 5`` with
``num_spec_tokens == 4`` and a precomputed (host-side) log-space gate. Unlike
T=2 (always split4) and T=4 (always split2), the T=5 value split is
**shape-dependent**: with ``work = num_seqs * num_value_heads`` and ``S`` SMs,
``recurrent_kda.py:1181-1191`` returns 8, 2, 4, 2, 1 across five bands, so all
four exports are reachable and all four are in this port's scope. The split is
a constexpr here, exactly as in the two-split T=1 sibling.

What "gram" means: at T<=4 the WY coefficients came from per-warp shuffle
reductions of ``k_t . k_s`` and ``q_t . k_s`` scaled by the gate ratio. At T=5
one designated warp instead forms both 16x16 Gram products on tensor cores
(``ldmatrix`` + ``mma.m16n8k16``) out of a gate-deflated ``k / prefix`` copy in
the new ``sGramA0``/``sGramA1`` shared regions -- regions that existed as dead
aliases of ``sVec`` in the T<=4 bodies. The inter-token butterfly path is gone
entirely, and a partial named barrier (``barrier.sync 1, 160`` /
``barrier.arrive 1, 160``) synchronizes the five token warps around it.

Helper vocabulary is shared with the T=2 module; only the geometry constants
and the kernel body are per-specialization.
"""

from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K

from . import flashkda_decode_t2_precomputed as _t2
from . import flashkda_decode_t4_precomputed as _t4

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
# True low-level leaves shared by the typed K layouts in the T=4/5/6 family.
_store_smem_f32 = _t4._store_smem_f32
_load_smem_f32 = _t4._load_smem_f32
_store_smem_i32 = _t4._store_smem_i32
_load_smem_i32 = _t4._load_smem_i32
_load_smem_f32x4 = _t4._load_smem_f32x4
_store_smem_u32x4_at = _t4._store_smem_u32x4_at
_store_smem_b16_at = _t4._store_smem_b16_at
_ldmatrix_x4_at = _t4._ldmatrix_x4_at
_svec_ptr = _t4._svec_ptr


def _div(a, b):
    """``div.approx.ftz.f32`` -- the lowering nvcc picks for ``k / prefix``.

    Read off the exported PTX (8 of them in the sVec/sGramA publish, none of
    the rcp+mul or full-range forms); -use_fast_math selects the approximate line.
    """
    return _ptx_bin("div.approx.ftz.f32", a, b)


def _named_bar_sync(bar_id: int, threads: int):
    """``barrier.sync <id>, <count>`` -- blocks until `count` threads arrive."""
    K.evaluate(K.ptx["barrier.sync"](bar_id, threads))


def _named_bar_arrive(bar_id: int, threads: int):
    """``barrier.arrive <id>, <count>`` -- releases, does NOT block or acquire."""
    K.evaluate(K.ptx["barrier.arrive"](bar_id, threads))


def _mma_zero_b(acc, a, b, b0: int):
    """``mma.sync...m16n8k16`` with an explicit zero C, B taken at ``b[b0:b0+2]``.

    The gram block issues two products from one pair of ``ldmatrix`` results, so
    unlike the shared helper this one has to select the B half.
    """
    K.evaluate(
        K.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
            acc[0], acc[1], acc[2], acc[3],
            a[0], a[1], a[2], a[3],
            b[b0], b[b0 + 1],
            *_t2._MMA_ZERO_C,
        )
    )  # fmt: skip


def _mma_acc_b(acc, a, b, b0: int):
    """Same, accumulating: C aliases D, matching the source's `+f` tied registers."""
    K.evaluate(
        K.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
            acc[0], acc[1], acc[2], acc[3],
            a[0], a[1], a[2], a[3],
            b[b0], b[b0 + 1],
            acc[0], acc[1], acc[2], acc[3],
        )
    )  # fmt: skip


HEAD_DIM = _t2.HEAD_DIM
NUM_TOKENS = 5
L2_EPS = _t2.L2_EPS
LOG2_E = _t2.LOG2_E

# Per-split geometry, transcribed from the four bodies' `#define` blocks and
# their guard expressions (`.cu:45-88`, `:143-166`, `:180`, `:411`, `:448`,
# `:652`). `threads` is FLASHKDA_DECODE_LAUNCH_THREADS (the `#define THREADS
# 256` in every body is vestigial -- it only feeds a static_assert in
# binding_impl.cuh:40); it is derived upstream by
# `max(tokens, value_rows/16, ((value_rows/rows_per_group)+1)/2) * 32`
# with rows_per_group = 2 only for gram+split8 (flash_kda_decode.py:115-134).
_SPLIT_GEOMETRY: dict[int, dict[str, int]] = {
    1: {
        "threads": 256,  # 8 warps
        "rows_per_cta": 128,
        "mma_warps": 8,  # phases D-G guard `warp_0 < 8`
        "row_groups": 16,  # phases B/H guard `group < 16`
        "rows_per_group": 8,
        "gram_warp": 0,
        "gram_sync_all": 0,  # gram warp syncs, the other token warps arrive
        "su_sync_cta": 0,  # phase G publishes sU behind __syncwarp
    },
    2: {
        "threads": 160,  # 5 warps
        "rows_per_cta": 64,
        "mma_warps": 4,
        "row_groups": 8,
        "rows_per_group": 8,
        "gram_warp": 0,
        "gram_sync_all": 0,
        "su_sync_cta": 0,
    },
    4: {
        "threads": 160,
        "rows_per_cta": 32,
        "mma_warps": 2,
        "row_groups": 4,
        "rows_per_group": 8,
        "gram_warp": 4,
        "gram_sync_all": 1,  # all five token warps sync; the arrive arm is dead
        "su_sync_cta": 0,
    },
    8: {
        "threads": 160,
        "rows_per_cta": 16,
        "mma_warps": 1,
        "row_groups": 8,  # 8 groups x 2 rows
        "rows_per_group": 2,
        "gram_warp": 4,
        "gram_sync_all": 0,
        "su_sync_cta": 1,  # phase G's producer/consumer warps differ -> CTA sync
    },
}


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    """One config row. Defaults mirror FlashInfer's T=5 export bench."""
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
        "seed": 20260815,
    }
    config.update(overrides)
    return config


# The T=5 split moves with shape: with W = num_seqs * num_value_heads and S SMs
# the bands are W <= 3S/8 -> 8, W <= S/2 -> 2, W <= 3S/4 -> 4, W <= 3S/2 -> 2,
# else 1 (recurrent_kda.py:1181-1191). Every label names the split its shape
# actually selects on a 148-SM B200, and verify_dispatch.py asserts that
# against the real selector. The five hv32h16 rows are FlashInfer's own T=5
# export-bench matrix (all split1); the rest exist to cover the other three
# exports, which are equally reachable in production.
BENCH_CONFIGS = [
    # FlashInfer's own T=5 export bench (W = 256..4096, all split1).
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
    # nat selects the initial checkpoint slot as ssm_idx[n*5 + clamp(nat-1, 0, 4)];
    # the upstream test sweeps 0/1/5/12, covering both clamp arms (.cu:280-286).
    _case("hv32h16_b8_nat0", num_seqs=8, accepted="zeros"),
    _case("hv32h16_b8_nat5", num_seqs=8, accepted="fives"),
    _case("hv32h16_b8_nat12", num_seqs=8, accepted="twelves"),
    _case("hv32h16_b8_natmix", num_seqs=8, accepted="mixed"),
    _case("hv32h16_b8_padded", num_seqs=8, padded_seqs=2),
    _case("hv32h16_b8_strided", num_seqs=8, slot_stride_pad=8),
    _case("hv32h16_b8_gstride", num_seqs=8, gate_token_stride_pad=8),
    _case("hv32h16_b8_scale", num_seqs=8, scale=0.05),
    _case("hv64h16_b8_s1", num_seqs=8, num_value_heads=64),
    # Forced splits at one fixed shape (W = 256), via the SM count the policy
    # reads: the same tensors must give the same answer through all four
    # exports (upstream test_..._forced_t5_splits_match_upstream_cute).
    _case("hv32h16_b8_force_s8", num_seqs=8, sm_count=683),
    _case("hv32h16_b8_force_s2", num_seqs=8, sm_count=512),
    _case("hv32h16_b8_force_s4", num_seqs=8, sm_count=342),
    # Selector knife edges on a 148-SM part: W = 222 -> split2, W = 224 -> split1.
    _case("hv16h16_b13_edge_s2", num_seqs=13, num_value_heads=16, sm_count=148),
    _case("hv16h16_b14_edge_s1", num_seqs=14, num_value_heads=16, sm_count=148),
]

KERNEL_META = {
    "name": "flashkda_decode_t5_gram",
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
    """Reproduce ``_select_flash_kda_decode_value_split_current`` for T = 5.

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


# TIRX_TRANSCRIBE_START flashkda_decode_t5_gram


def _make_flashkda_decode_t5_gram(spec: dict[str, Any]):
    """Trace the T=5 Gram schedule with K-owned launch, layout, and context."""
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

    @K.kernel(
        warps=THREADS // 32, arch="sm_100a", grid=(NUM_VALUE_HEADS * VALUE_SPLIT, spec["NUM_SEQS"])
    )
    def flashkda_decode_t5_gram(
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
        s_state0 = smem.alloc((ROWS_PER_CTA, 64), K.bf16, swizzle=K.SW128B, align=1024)
        s_state1 = smem.alloc((ROWS_PER_CTA, 64), K.bf16, swizzle=K.SW128B, align=1024)
        s_vec = smem.alloc((32, 64), K.bf16, swizzle=K.SW128B, align=1024)
        s_k = smem.alloc((NUM_TOKENS * HEAD_DIM,), K.f32)
        s_d = smem.alloc((NUM_TOKENS * HEAD_DIM,), K.f32)
        s_beta = smem.alloc((NUM_TOKENS,), K.f32)
        s_slot = smem.alloc((NUM_TOKENS,), K.i32)
        s_token = smem.alloc((NUM_TOKENS,), K.i32)
        s_init = smem.alloc((4,), K.i32)
        s_l = smem.alloc((NUM_TOKENS * NUM_TOKENS,), K.f32)
        s_r = smem.alloc((NUM_TOKENS * NUM_TOKENS,), K.f32)
        s_u = smem.alloc((NUM_TOKENS * ROWS_PER_CTA,), K.f32)
        s_grama0 = smem.alloc((16, 64), K.bf16, swizzle=K.SW128B, align=128)
        s_grama1 = smem.alloc((16, 64), K.bf16, swizzle=K.SW128B, align=128)
        smem.commit()
        work, n = K.cta_id()

        tid = K.thread_id()
        warp = K.warp_id()
        lane = K.lane_id()
        # --- work decomposition and lane roles (:142-178) ----------------------
        value_tile = K.alloc_local((1,), K.i32)
        hv = K.alloc_local((1,), K.i32)
        query_head = K.alloc_local((1,), K.i32)
        lane_quad = K.alloc_local((1,), K.i32)
        frag_row = K.alloc_local((1,), K.i32)
        quad_base = K.alloc_local((1,), K.i32)
        group = K.alloc_local((1,), K.i32)
        lane_group = K.alloc_local((1,), K.i32)
        k_start = K.alloc_local((1,), K.i32)
        elem_start = K.alloc_local((1,), K.i32)
        tile_row_base = K.alloc_local((1,), K.i32)
        owned_row_base = K.alloc_local((1,), K.i32)
        token_base = K.alloc_local((1,), K.i32)
        seq_len = K.alloc_local((1,), K.i32)
        K.assign(value_tile[0], work % VALUE_SPLIT)
        K.assign(hv[0], work // VALUE_SPLIT)
        K.assign(query_head[0], hv[0] // HEAD_RATIO)
        K.assign(lane_quad[0], lane % 4)
        K.assign(frag_row[0], lane // 4)
        K.assign(quad_base[0], lane - lane_quad[0])
        K.assign(group[0], tid // 16)
        K.assign(lane_group[0], tid % 16)
        K.assign(k_start[0], lane_group[0] * 8)
        K.assign(elem_start[0], lane * 4)
        K.assign(tile_row_base[0], value_tile[0] * ROWS_PER_CTA)
        K.assign(owned_row_base[0], group[0] * ROWS_PER_GROUP)
        K.assign(token_base[0], _load_i32(cu, n))
        K.assign(seq_len[0], _load_i32(cu, n + 1) - token_base[0])
        # These overlapping phase domains are the schedule's single source
        # of truth. They cannot be a K.specialize partition: one warp owns
        # token, Gram, row, and MMA work at different synchronization points.
        token_owner = warp < NUM_TOKENS
        gram_owner = warp == GRAM_WARP
        row_owner = group[0] < ROW_GROUPS
        mma_owner = warp < MMA_WARPS

        r_q = K.alloc_local((4,), "float32")
        r_k = K.alloc_local((4,), "float32")
        r_d = K.alloc_local((4,), "float32")

        # =======================================================================
        # Phase A: token preprocess, warp <-> token  (:180-290)
        # =======================================================================
        # Identical to the ported T=4 phase A apart from the token count, the
        # ssm_state_indices stride and the nat clamp ceiling.
        with K.If(token_owner), K.Then():
            token = warp
            active_token = token < seq_len[0]
            token_pos = K.alloc_local((1,), K.i32)
            qk_base = K.alloc_local((1,), K.i32)
            gate_base = K.alloc_local((1,), K.i32)
            K.assign(token_pos[0], K.if_then_else(active_token, token_base[0] + token, 0))
            K.assign(
                qk_base[0], (token_pos[0] * NUM_HEADS + query_head[0]) * HEAD_DIM + elem_start[0]
            )
            K.assign(
                gate_base[0], token_pos[0] * GATE_TOKEN_STRIDE + hv[0] * HEAD_DIM + elem_start[0]
            )

            q_words = _load_u32x2(q, qk_base[0])
            k_words = _load_u32x2(k, qk_base[0])
            g_words = _load_u32x2(g, gate_base[0])
            for pair in range(2):
                K.ptx.mov.b32(r_q[2 * pair], _widen_lo(q_words[pair]))
                K.ptx.mov.b32(r_q[2 * pair + 1], _widen_hi(q_words[pair]))
                K.ptx.mov.b32(r_k[2 * pair], _widen_lo(k_words[pair]))
                K.ptx.mov.b32(r_k[2 * pair + 1], _widen_hi(k_words[pair]))
                K.ptx.mov.b32(r_d[2 * pair], _widen_lo(g_words[pair]))
                K.ptx.mov.b32(r_d[2 * pair + 1], _widen_hi(g_words[pair]))

            # Index-ordered accumulation (:241-244); the first term has a zero addend
            # and the two chains interleave, because the source fuses them in one loop.
            q_sq = K.alloc_local((1,), K.f32)
            k_sq = K.alloc_local((1,), K.f32)
            K.assign(
                q_sq[0],
                _fma(
                    r_q[3],
                    r_q[3],
                    _fma(
                        r_q[2], r_q[2], _fma(r_q[1], r_q[1], _fma(r_q[0], r_q[0], K.float32(0.0)))
                    ),
                ),
            )
            K.assign(
                k_sq[0],
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
                K.assign(q_sq[0], _add(q_sq[0], _shfl_bfly(q_sq[0], 16 >> off)))
            for off in range(5):
                K.assign(k_sq[0], _add(k_sq[0], _shfl_bfly(k_sq[0], 16 >> off)))
            q_norm = _mul(_rsqrt(_add(q_sq[0], K.float32(L2_EPS))), scale)
            k_norm = _rsqrt(_add(k_sq[0], K.float32(L2_EPS)))

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
            _store_smem_u32x4_at(s_k.ptr_to([token * HEAD_DIM + elem_start[0]]), k_pub)
            _store_smem_u32x4_at(s_d.ptr_to([token * HEAD_DIM + elem_start[0]]), d_pub)

            with K.If(lane == 0), K.Then():
                raw_slot = _load_i32(ssm_idx, n * NUM_TOKENS + token)
                _store_smem_i32(s_slot, token, K.if_then_else(active_token, raw_slot, -1))
                _store_smem_i32(s_token, token, token_pos[0])
                _store_smem_f32(
                    s_beta, token, _load_bf16_f32(beta, token_pos[0] * NUM_VALUE_HEADS + hv[0])
                )
                with K.If(token == 0), K.Then():
                    # nat picks the initial checkpoint slot; at T=5 the clamp ceiling
                    # is 4 and both edges are reachable (:279-287).
                    accepted = K.min(K.max(_load_i32(nat, n) - 1, 0), NUM_TOKENS - 1)
                    initial_slot = _load_i32(ssm_idx, n * NUM_TOKENS + accepted)
                    _store_smem_i32(s_init, 0, K.max(initial_slot, 0))

        K.cuda.cta_sync()

        # =======================================================================
        # Phase C': sVec columns and the gram operand  (:293-339)
        # =======================================================================
        # Runs BEFORE the state gather at T=5 -- the reverse of the T=4 order.
        with K.If(token_owner), K.Then():
            token_c = warp
            for i in range(4):
                k_idx = elem_start[0] + i
                prefix = K.alloc_local((1,), K.f32)
                K.assign(prefix[0], K.float32(1.0))
                # Scalar loads: the walk is across tokens at a fixed key, so
                # consecutive iterations are 512 B apart (:296-302).
                for j in range(NUM_TOKENS):
                    with K.If(token_c >= j), K.Then():
                        K.assign(
                            prefix[0], _mul(prefix[0], _load_smem_f32(s_d, j * HEAD_DIM + k_idx))
                        )
                _store_smem_b16_at(
                    _svec_ptr(s_vec, k_idx, token_c),
                    _ptx_un("cvt.rn.bf16.f32", _mul(prefix[0], r_k[i]), dtype="uint16"),
                )
                # The q column is 8 + token at T=5, not 4 + token: the generator
                # emits `c_col = 4 + token` and immediately overrides it (:311-314).
                _store_smem_b16_at(
                    _svec_ptr(s_vec, k_idx, 8 + token_c),
                    _ptx_un("cvt.rn.bf16.f32", _mul(prefix[0], r_q[i]), dtype="uint16"),
                )
                # The gate-DEFLATED key, the operand that makes the Gram product come
                # out as T<=4's ratio_scan factor. div.approx.ftz.f32 per the PTX.
                deflated = _ptx_un("cvt.rn.bf16.f32", _div(r_k[i], prefix[0]), dtype="uint16")
                with K.If(k_idx < 64):
                    with K.Then():
                        _store_smem_b16_at(s_grama0.ptr_to(token_c, k_idx), deflated)
                    with K.Else():
                        _store_smem_b16_at(s_grama1.ptr_to(token_c, k_idx - 64), deflated)

        # =======================================================================
        # Phase C'': the coefficient Gram block  (:340-408)  -- new at T=5
        # =======================================================================
        # One warp forms both token x token Gram products on tensor cores, replacing
        # T<=4's inter-token butterfly path entirely. The barrier is partial: only
        # the five token warps take part, which is why its count is 160 in every
        # split, including split1's 256-thread launch.
        with K.If(token_owner), K.Then():
            if GRAM_SYNC_ALL == 1:
                # split4 hoists the wait out of the branch: all five token warps
                # block, and the arrive arm below is `else if (0)` -- dead (S4:340-344).
                _named_bar_sync(1, NUM_TOKENS * 32)
            with K.If(gram_owner):
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
                            gram_a_tile = s_grama0 if gram_half == 0 else s_grama1
                            _ldmatrix_x4_at(
                                gram_a_tile.ptr_to(lane % 16, gram_k + lane // 16 * 8),
                                gram_a,
                                False,
                            )
                            # Same sVec address formula as phase D's operand (:360-361
                            # vs :454-455); frags 0,1 are columns 0..7 (the k side) and
                            # 2,3 are columns 8..15 (the q side).
                            _ldmatrix_x4_at(
                                _svec_ptr(s_vec, global_gram_k + lane % 16, lane // 16 * 8),
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
                    source_token = frag_row[0]
                    target0 = lane_quad[0] * 2
                    target1 = target0 + 1
                    with K.If(source_token < NUM_TOKENS), K.Then():
                        beta_source = _load_smem_f32(s_beta, source_token)
                        with K.If(K.And(source_token < target0, target0 < NUM_TOKENS)), K.Then():
                            _store_smem_f32(
                                s_l,
                                target0 * NUM_TOKENS + source_token,
                                _mul(beta_source, gram_k_acc[0]),
                            )
                        with K.If(K.And(source_token < target1, target1 < NUM_TOKENS)), K.Then():
                            _store_smem_f32(
                                s_l,
                                target1 * NUM_TOKENS + source_token,
                                _mul(beta_source, gram_k_acc[1]),
                            )
                        with K.If(K.And(source_token <= target0, target0 < NUM_TOKENS)), K.Then():
                            _store_smem_f32(
                                s_r,
                                target0 * NUM_TOKENS + source_token,
                                _mul(beta_source, gram_q_acc[0]),
                            )
                        with K.If(K.And(source_token <= target1, target1 < NUM_TOKENS)), K.Then():
                            _store_smem_f32(
                                s_r,
                                target1 * NUM_TOKENS + source_token,
                                _mul(beta_source, gram_q_acc[1]),
                            )
                if GRAM_SYNC_ALL == 0:
                    with K.Else():
                        _named_bar_arrive(1, NUM_TOKENS * 32)

        # =======================================================================
        # Phase B: state gather and sState stage  (:410-446)
        # =======================================================================
        init_slot = _load_smem_i32(s_init, 0)
        head_base = K.cast(init_slot, "int64") * K.cast(STATE_SLOT_STRIDE, "int64") + K.cast(
            hv[0] * HEAD_DIM * HEAD_DIM, "int64"
        )
        hist = K.alloc_local((ROWS_PER_GROUP * 8,), "float32")
        with K.If(row_owner), K.Then():
            for row_local in range(ROWS_PER_GROUP):
                # Two distinct indices: row_l is CTA-local and addresses sState;
                # tile_row_base + row_l is the global row of `state` (:414-415).
                row_l = owned_row_base[0] + row_local
                pack = _load_u32x4(
                    state,
                    head_base + K.cast((tile_row_base[0] + row_l) * HEAD_DIM + k_start[0], "int64"),
                )
                for pr in range(4):
                    K.ptx.mov.b32(hist[row_local * 8 + 2 * pr], _widen_lo(pack[pr]))
                    K.ptx.mov.b32(hist[row_local * 8 + 2 * pr + 1], _widen_hi(pack[pr]))
                # An if/ELSE selecting the destination half, not a guard: lanes 8..15
                # stage keys 64..127 into sState1 (:438-442). The bf16 bits go to
                # shared unmodified; the swizzle is on the byte offset.
                with K.If(lane_group[0] < 8):
                    with K.Then():
                        _store_smem_u32x4_at(s_state0.ptr_to(row_l, k_start[0]), pack)
                    with K.Else():
                        _store_smem_u32x4_at(s_state1.ptr_to(row_l, k_start[0] - 64), pack)

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
        with K.If(mma_owner), K.Then():
            vec_frag = K.alloc_local((4,), "uint32", align=4)
            state_frag = K.alloc_local((4,), "uint32", align=4)
            for state_half in range(2):
                for mma_step in range(4):
                    mma_k = mma_step * 16
                    global_k = state_half * 64 + mma_k
                    _ldmatrix_x4_at(
                        _svec_ptr(s_vec, global_k + lane % 16, lane // 16 * 8), vec_frag, True
                    )
                    state_tile = s_state0 if state_half == 0 else s_state1
                    _ldmatrix_x4_at(
                        state_tile.ptr_to(warp * 16 + lane % 16, mma_k + lane // 16 * 8),
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
        # Phase E: quad broadcast and the WY forward substitution  (:491-560)
        # =======================================================================
        u_lo = K.alloc_local((NUM_TOKENS,), "float32")
        u_hi = K.alloc_local((NUM_TOKENS,), "float32")
        # hc_* is declared out here because phase F consumes it from its own block;
        # ha_* never leaves phase E.
        hc_lo = K.alloc_local((NUM_TOKENS,), "float32")
        hc_hi = K.alloc_local((NUM_TOKENS,), "float32")
        with K.If(mma_owner), K.Then():
            # 20 broadcasts in the source's emission order (:491-531). m16n8k16 puts
            # columns 2q, 2q+1 of row frag_row in lane quad_base+q's acc[0],[1] and
            # rows +8 in acc[2],[3], so token t lives at (quad_base + t//2, t%2).
            # ha_* is the k side (the solve), hc_* the q side (the outputs).
            ha_lo = K.alloc_local((NUM_TOKENS,), "float32")
            ha_hi = K.alloc_local((NUM_TOKENS,), "float32")
            for t in range(4):
                K.ptx.mov.b32(ha_lo[t], _shfl_idx(acc[t % 2], quad_base[0] + t // 2))
            for t in range(4):
                K.ptx.mov.b32(ha_hi[t], _shfl_idx(acc[2 + t % 2], quad_base[0] + t // 2))
            K.ptx.mov.b32(ha_lo[4], _shfl_idx(acc[0], quad_base[0] + 2))
            K.ptx.mov.b32(ha_hi[4], _shfl_idx(acc[2], quad_base[0] + 2))
            for t in range(NUM_TOKENS):
                K.ptx.mov.b32(hc_lo[t], _shfl_idx(acc_c[t % 2], quad_base[0] + t // 2))
            for t in range(NUM_TOKENS):
                K.ptx.mov.b32(hc_hi[t], _shfl_idx(acc_c[2 + t % 2], quad_base[0] + t // 2))

            with K.If(lane_quad[0] == 2), K.Then():
                row_lo = warp * 16 + frag_row[0]
                row_hi = row_lo + 8
                v_lo_bits = K.alloc_local((NUM_TOKENS,), K.u16)
                v_hi_bits = K.alloc_local((NUM_TOKENS,), K.u16)
                for t in range(NUM_TOKENS):
                    base_t = (_load_smem_i32(s_token, t) * NUM_VALUE_HEADS + hv[0]) * HEAD_DIM
                    K.ptx.ld.global_.nc.b16(
                        v_lo_bits[t], v.ptr_to([base_t + tile_row_base[0] + row_lo])
                    )
                    K.ptx.ld.global_.nc.b16(
                        v_hi_bits[t], v.ptr_to([base_t + tile_row_base[0] + row_hi])
                    )
                for t in range(NUM_TOKENS):
                    solved_lo = K.alloc_local((1,), K.f32)
                    solved_hi = K.alloc_local((1,), K.f32)
                    K.assign(solved_lo[0], _sub(_ptx_un("cvt.f32.bf16", v_lo_bits[t]), ha_lo[t]))
                    K.assign(solved_hi[0], _sub(_ptx_un("cvt.f32.bf16", v_hi_bits[t]), ha_hi[t]))
                    for prev in range(t):
                        lts = _load_smem_f32(s_l, t * NUM_TOKENS + prev)
                        K.assign(solved_lo[0], _sub(solved_lo[0], _mul(lts, u_lo[prev])))
                        K.assign(solved_hi[0], _sub(solved_hi[0], _mul(lts, u_hi[prev])))
                    K.ptx.mov.b32(u_lo[t], solved_lo[0])
                    K.ptx.mov.b32(u_hi[t], solved_hi[0])

            # The solve runs on lane_quad == 2 but lane_quad == 3 also writes output,
            # so the residuals cross the quad (:554-560).
            for t in range(NUM_TOKENS):
                K.ptx.mov.b32(u_lo[t], _shfl_idx(u_lo[t], quad_base[0] + 2))
                K.ptx.mov.b32(u_hi[t], _shfl_idx(u_hi[t], quad_base[0] + 2))

        # =======================================================================
        # Phase F: the outputs  (:562-640)
        # =======================================================================
        # The bases come from hc_* (the q-side MMA), not from acc: the source
        # assigns acc[0..3] and then unconditionally overwrites (:567-582). The
        # lane_quad == 3 remap uses STATIC indices, as the source does -- indexing
        # hc_* by a runtime token would spill the register array to local memory.
        with K.If(K.And(mma_owner, lane_quad[0] >= 2)), K.Then():
            token0 = (lane_quad[0] - 2) * 2
            token1 = token0 + 1
            row_lo_f = warp * 16 + frag_row[0]
            row_hi_f = row_lo_f + 8
            out0_lo = K.alloc_local((1,), K.f32)
            out1_lo = K.alloc_local((1,), K.f32)
            out0_hi = K.alloc_local((1,), K.f32)
            out1_hi = K.alloc_local((1,), K.f32)
            K.assign(out0_lo[0], hc_lo[0])
            K.assign(out1_lo[0], hc_lo[1])
            K.assign(out0_hi[0], hc_hi[0])
            K.assign(out1_hi[0], hc_hi[1])
            with K.If(lane_quad[0] == 3), K.Then():
                K.assign(out0_lo[0], hc_lo[2])
                K.assign(out1_lo[0], hc_lo[3])
                K.assign(out0_hi[0], hc_hi[2])
                K.assign(out1_hi[0], hc_hi[3])
            for src in range(NUM_TOKENS):
                residual_lo = u_lo[src]
                residual_hi = u_hi[src]
                coef0 = K.alloc_local((1,), K.f32)
                coef1 = K.alloc_local((1,), K.f32)
                K.assign(coef0[0], K.float32(0.0))
                K.assign(coef1[0], K.float32(0.0))
                # The masked-out coefficient is a real zero-operand fma, not a
                # skipped iteration (:585-594).
                with K.If(token0 >= src), K.Then():
                    K.assign(coef0[0], _load_smem_f32(s_r, token0 * NUM_TOKENS + src))
                with K.If(token1 >= src), K.Then():
                    K.assign(coef1[0], _load_smem_f32(s_r, token1 * NUM_TOKENS + src))
                K.assign(out0_lo[0], _fma(coef0[0], residual_lo, out0_lo[0]))
                K.assign(out1_lo[0], _fma(coef1[0], residual_lo, out1_lo[0]))
                K.assign(out0_hi[0], _fma(coef0[0], residual_hi, out0_hi[0]))
                K.assign(out1_hi[0], _fma(coef1[0], residual_hi, out1_hi[0]))

            for half in range(2):
                token_o = token0 if half == 0 else token1
                o_lo = out0_lo[0] if half == 0 else out1_lo[0]
                o_hi = out0_hi[0] if half == 0 else out1_hi[0]
                active_o = _load_smem_i32(s_slot, token_o) >= 0
                base_o = (
                    _load_smem_i32(s_token, token_o) * NUM_VALUE_HEADS + hv[0]
                ) * HEAD_DIM + tile_row_base[0]
                # A padded row writes EXPLICIT zeros; the upstream test asserts them
                # bit-exactly, so this is not an "unwritten" path.
                _store_f32_as_bf16(out, base_o + row_lo_f, o_lo, active_o)
                _store_f32_as_bf16(out, base_o + row_hi_f, o_hi, active_o)
                _store_f32_as_bf16(out, base_o + row_lo_f, K.float32(0.0), K.Not(active_o))
                _store_f32_as_bf16(out, base_o + row_hi_f, K.float32(0.0), K.Not(active_o))

            # The fifth token has no partner lane, so lane_quad == 2 writes it too
            # (:622-639). Row 4 of sR needs no mask -- target 4 >= every source.
            with K.If(lane_quad[0] == 2), K.Then():
                out4_lo = K.alloc_local((1,), K.f32)
                out4_hi = K.alloc_local((1,), K.f32)
                K.assign(out4_lo[0], hc_lo[4])
                K.assign(out4_hi[0], hc_hi[4])
                for src4 in range(NUM_TOKENS):
                    coef4 = _load_smem_f32(s_r, (NUM_TOKENS - 1) * NUM_TOKENS + src4)
                    K.assign(out4_lo[0], _fma(coef4, u_lo[src4], out4_lo[0]))
                    K.assign(out4_hi[0], _fma(coef4, u_hi[src4], out4_hi[0]))
                active4 = _load_smem_i32(s_slot, NUM_TOKENS - 1) >= 0
                base4 = (
                    _load_smem_i32(s_token, NUM_TOKENS - 1) * NUM_VALUE_HEADS + hv[0]
                ) * HEAD_DIM + tile_row_base[0]
                _store_f32_as_bf16(out, base4 + row_lo_f, out4_lo[0], active4)
                _store_f32_as_bf16(out, base4 + row_hi_f, out4_hi[0], active4)
                _store_f32_as_bf16(out, base4 + row_lo_f, K.float32(0.0), K.Not(active4))
                _store_f32_as_bf16(out, base4 + row_hi_f, K.float32(0.0), K.Not(active4))

        # =======================================================================
        # Phase G: publish sU  (:641-654)
        # =======================================================================
        with K.If(mma_owner), K.Then():
            with K.If(lane_quad[0] == 2), K.Then():
                row_lo_g = warp * 16 + frag_row[0]
                for t in range(NUM_TOKENS):
                    _store_smem_f32(s_u, t * ROWS_PER_CTA + row_lo_g, u_lo[t])
                    _store_smem_f32(s_u, t * ROWS_PER_CTA + row_lo_g + 8, u_hi[t])
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
        # Phase H: recurrence and checkpoints  (:768-804)
        # =======================================================================
        words_w = K.alloc_local((4,), "uint32")
        sd_t = K.alloc_local((8,), "float32")
        sk_t = K.alloc_local((8,), "float32")
        with K.If(row_owner), K.Then():
            for t in range(NUM_TOKENS):
                slot_t = _load_smem_i32(s_slot, t)
                beta_t = _load_smem_f32(s_beta, t)
                # The gate and key slices depend only on (t, k_start), not on the
                # row, so they are hoisted out of the row loop as two 16-byte reads
                # each rather than reloaded per key (:782).
                _load_smem_f32x4(s_d, t * HEAD_DIM + k_start[0], sd_t, 0)
                _load_smem_f32x4(s_d, t * HEAD_DIM + k_start[0] + 4, sd_t, 4)
                _load_smem_f32x4(s_k, t * HEAD_DIM + k_start[0], sk_t, 0)
                _load_smem_f32x4(s_k, t * HEAD_DIM + k_start[0] + 4, sk_t, 4)
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
                            row_h = owned_row_base[0] + row_local
                            update = _mul(_load_smem_f32(s_u, t * ROWS_PER_CTA + row_h), beta_t)
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
                                    hv[0] * HEAD_DIM * HEAD_DIM
                                    + (tile_row_base[0] + row_h) * HEAD_DIM
                                    + k_start[0],
                                    "int64",
                                ),
                                words_w,
                            )
                    with K.Else():
                        for row_local_p in range(ROWS_PER_GROUP):
                            row_p = owned_row_base[0] + row_local_p
                            update_p = _mul(_load_smem_f32(s_u, t * ROWS_PER_CTA + row_p), beta_t)
                            for i in range(8):
                                K.ptx.mov.b32(
                                    hist[row_local_p * 8 + i],
                                    _fma(
                                        hist[row_local_p * 8 + i], sd_t[i], _mul(update_p, sk_t[i])
                                    ),
                                )

    return flashkda_decode_t5_gram


def get_kernel(**kwargs: Any):
    """Return the specialized FlashKDA cake T=5 gram decode PrimFunc."""
    return _make_flashkda_decode_t5_gram(_specialization(kwargs)).func


_NAT_PATTERNS = {None: None, "zeros": 0, "ones": 1, "fives": 5, "twelves": 12}


def _accepted_tensor(kind: Any, num_seqs: int, device: str) -> torch.Tensor:
    """num_accepted_tokens per case; the kernel clamps nat-1 into [0, 4]."""
    if kind is None:
        return torch.ones(num_seqs, device=device, dtype=torch.int32)
    if kind == "mixed":
        # The upstream T=5 matrix test's sweep, tiled to the batch size.
        pattern = torch.tensor([0, 1, 5, 12, 4, 0, 1, 5], dtype=torch.int32)
        reps = (num_seqs + pattern.numel() - 1) // pattern.numel()
        return pattern.repeat(reps)[:num_seqs].to(device)
    if kind in _NAT_PATTERNS:
        return torch.full((num_seqs,), _NAT_PATTERNS[kind], device=device, dtype=torch.int32)
    raise ValueError(f"unknown accepted-token pattern {kind!r}")


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Build one deterministic case with independent TIRx / reference state.

    Same recipe as the ported T=2/T=4 siblings at T=5: packed [1, N*5, ...]
    bf16, a log-space gate (GATE_KIND 0 -- the kernel applies exp()),
    pre-sigmoided beta, flat [N*5] slot indices with slot 0 never a target.
    """
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for FlashKDA cake T=5 decode")
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
    return f"d128_t5_precomputed_gram_split{value_split}"


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
    """Independent FP32 oracle of the T=5 delta rule (sequential form)."""
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
# operands to bf16 -- and at T=5 the WY coefficients themselves come from a
# bf16 Gram product of `k / prefix`, so this band is wider than the T<=4 one and
# is re-measured at the correctness gate rather than inherited.
_ORACLE_RTOL = 2.0**-5
_ORACLE_ATOL = 6.0e-3


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
