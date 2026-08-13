# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400),
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's FlashKDA "cake" T=3 lower-bound decode kernel.

Source: ``csrc/kda/flashkda_decode_d128_t3_lower_bound_split4.cu``, symbol
``kernel_flashinfer_recurrent_kda_wy_vtile_short`` -- the same frozen WY
template as the T=2 and T=4 siblings, re-instantiated at TOKENS=3 with
**GATE_KIND = 1**.

T=3 bypasses the value-split selector entirely: the dispatcher early-returns
``d128_t3_lower_bound_split4`` from the ``is_t3_lower_bound`` predicate
(``recurrent_kda.py:1285-1294``, ``:1437-1438``). Its legal domain is narrow and
enforced host-side -- ``num_spec_tokens == 2``, a finite ``lower_bound < 0``,
``A_log``/``dt_bias`` present, ``N in {1,2,4,8,16}`` and ``H == HV == 16``
(``:1352-1359``, ``:1406-1416``).

Two things make this body different from every cake kernel ported so far:

* **the gate is computed in-kernel** (GATE_KIND 1), so ``A_log``, ``dt_bias``
  and ``lower_bound`` are live arguments rather than the dummies the
  precomputed variants pass. Per element the source computes
  ``lower_bound / (1 + exp(-exp(A_log[h]) * (g + dt_bias[h*128 + k])))`` and
  then exponentiates it -- three transcendental sites against T=2's one, and
  the family's only division.
* **warp 2 is a token-only warp.** At 96 threads with 32 value rows the
  state-side bounds are still 2 warps and 4 groups (they derive from rows/16
  and rows/8, not from T), so warp 2 runs the token preprocess and its share of
  the WY coefficients, hits both barriers, and then does nothing: it never
  gathers state, never issues an MMA, never stores. Both CTA barriers are
  therefore genuine three-warp edges.

Helper vocabulary is shared with the T=2 module; only the geometry constants,
the gate and the kernel body are per-specialization.
"""

from __future__ import annotations

from typing import Any
from unittest import SkipTest

import torch

from tvm.script import tirx as T

from . import flashkda_decode_t2_precomputed as _t2

# Shared helper vocabulary (identical PTX forms, identical swizzle -- verified
# against all three bodies).
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


# --- GATE_KIND == 1 additions ---------------------------------------------
# The three forms the precomputed siblings never need. All compile-probed with
# a known-bad control (`mul.ftz.f64` is rejected), and all three appear in the
# frozen export's PTX: 4 div.approx.ftz.f32 (one per element, and NOT
# rcp.approx + mul), 1 neg.ftz.f32 (hoisted), 5 extra ld.global.nc.b32.


def _neg(a):
    """``neg.ftz.f32``."""
    return _ptx_un("neg.ftz.f32", a)


def _div(a, b):
    """``div.approx.ftz.f32`` -- what ``-use_fast_math`` lowers ``/`` to here."""
    return _ptx_bin("div.approx.ftz.f32", a, b)


def _load_f32(buffer, index):
    """``ld.global.nc.b32`` -- one read-only fp32 (A_log / dt_bias).

    dt_bias's four elements per lane are contiguous and nvcc still emits four
    scalar loads, not an ld.global.nc.v4.b32; the port matches that.
    """
    out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.ld.global_.nc.b32(out[0], buffer.ptr_to([index])))
    return T.reinterpret("float32", out[0])


HEAD_DIM = _t2.HEAD_DIM
NUM_TOKENS = 3
VALUE_SPLIT = 4  # T = 3 bypasses the selector; split4 is the whole surface
L2_EPS = _t2.L2_EPS
LOG2_E = _t2.LOG2_E

THREADS = 96  # 3 warps (flash_kda_decode.py:_variant_metadata)
ROWS_PER_CTA = HEAD_DIM // VALUE_SPLIT  # tile_row_base = value_tile * 32 (.cu:159)
MMA_WARPS = HEAD_DIM // VALUE_SPLIT // 16  # phase D-G guard: value_rows/16
ROW_GROUPS = HEAD_DIM // VALUE_SPLIT // 8  # phase B/H guard: value_rows/8
ROWS_PER_THREAD = 8  # groups 0..3 (warps 0,1) own 8 rows x 8 keys each
K_PER_THREAD = 8

# Arena offsets copied from the source's #define block (.cu:45-88). sState and sVec keep the T=2 offsets exactly.
OFF_SSTATE0 = 0
OFF_SSTATE1 = 4096
OFF_SVEC = 8192
OFF_SK = 12288
OFF_SD = 13824
OFF_SBETA = 15360
OFF_SSLOT = 15372
OFF_STOKEN = 15384
OFF_SINIT = 15396
OFF_SL = 15412
OFF_SR = 15448
OFF_SU = 15484
SMEM_TOTAL = 15872
# sGramA0/sGramA1 alias sVec and are t5/t6 machinery -- dead here.

# TIRX_TRANSCRIBE_START flashkda_decode_t3_lower_bound


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    """One config row. Defaults mirror FlashInfer's T=3 export bench.

    H and HV are pinned at 16 and cannot vary: the dispatcher rejects anything
    else (recurrent_kda.py:1352-1359).
    """
    config: dict[str, Any] = {
        "label": label,
        "num_seqs": 8,
        "num_heads": 16,
        "num_value_heads": 16,
        "pool_slack": 6,
        "padded_seqs": 0,
        "slot_stride_pad": 0,
        "gate_token_stride_pad": 0,
        "accepted": None,
        "lower_bound": -5.0,
        "scale": None,
        "seed": 20260814,
    }
    config.update(overrides)
    return config


# T=3's legal domain IS the matrix: the dispatcher enforces H == HV == 16 and
# N in {1,2,4,8,16}, so these five rows are every shape the kernel can serve.
# They match FlashInfer's own T=3 export bench exactly.
BENCH_CONFIGS = [
    _case("hv16h16_b1_t3", num_seqs=1),
    _case("hv16h16_b2_t3", num_seqs=2),
    _case("hv16h16_b4_t3", num_seqs=4),
    _case("hv16h16_b8_t3", num_seqs=8),
    _case("hv16h16_b16_t3", num_seqs=16),
]

CONFIGS = [dict(cfg) for cfg in BENCH_CONFIGS] + [
    # nat selects the initial checkpoint slot as ssm_idx[n*3 + clamp(nat-1, 0, 2)];
    # the upstream test sweeps 0/1/3/10, covering both clamp edges (.cu:287-296).
    _case("hv16h16_b8_t3_nat0", num_seqs=8, accepted="zeros"),
    _case("hv16h16_b8_t3_nat3", num_seqs=8, accepted="threes"),
    _case("hv16h16_b8_t3_nat10", num_seqs=8, accepted="tens"),
    _case("hv16h16_b8_t3_natmix", num_seqs=8, accepted="mixed"),
    _case("hv16h16_b8_t3_padded", num_seqs=8, padded_seqs=2),
    _case("hv16h16_b8_t3_strided", num_seqs=8, slot_stride_pad=8),
    _case("hv16h16_b8_t3_gstride", num_seqs=8, gate_token_stride_pad=8),
    _case("hv16h16_b8_t3_scale", num_seqs=8, scale=0.05),
    # The gate is computed in-kernel, so lower_bound is a real input: a second
    # value must move the output.
    _case("hv16h16_b8_t3_lb1", num_seqs=8, lower_bound=-1.0),
]

KERNEL_META = {
    "name": "flashkda_decode_t3_lower_bound",
    "category": "flashinfer",
    "compute_capability": 10,
}


def _specialization(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Derive the constexpr set, mirroring the source host dispatch."""
    num_seqs = int(kwargs["num_seqs"])
    num_heads = int(kwargs["num_heads"])
    num_value_heads = int(kwargs["num_value_heads"])
    slot_stride_pad = int(kwargs.get("slot_stride_pad", 0))
    gate_token_stride_pad = int(kwargs.get("gate_token_stride_pad", 0))
    pool_slack = int(kwargs.get("pool_slack", 6))

    # The dispatcher rejects anything outside this box (recurrent_kda.py:1352-1359).
    if num_heads != 16 or num_value_heads != 16:
        raise ValueError("t3 lower-bound requires num_heads == num_value_heads == 16")
    if num_seqs not in (1, 2, 4, 8, 16):
        raise ValueError("t3 lower-bound requires num_seqs in {1, 2, 4, 8, 16}")
    if slot_stride_pad % 8 != 0:
        raise ValueError("state slot stride padding must stay 8-element aligned")
    if gate_token_stride_pad % 4 != 0:
        raise ValueError("gate token stride padding must stay 4-element aligned")
    if VALUE_SPLIT != 4:
        raise ValueError("only d128_t3_lower_bound_split4 is in this port's scope")

    total_tokens = num_seqs * NUM_TOKENS
    slot_stride = num_value_heads * HEAD_DIM * HEAD_DIM + slot_stride_pad
    gate_token_stride = num_value_heads * HEAD_DIM + gate_token_stride_pad
    state_slots = total_tokens + pool_slack
    return {
        "NUM_SEQS": num_seqs,
        "NUM_HEADS": num_heads,
        "NUM_VALUE_HEADS": num_value_heads,
        "HEAD_RATIO": num_value_heads // num_heads,
        "STATE_SLOT_STRIDE": slot_stride,
        "GATE_TOKEN_STRIDE": gate_token_stride,
        "Q_ELEMENTS": total_tokens * num_heads * HEAD_DIM,
        "V_ELEMENTS": total_tokens * num_value_heads * HEAD_DIM,
        "GATE_ELEMENTS": total_tokens * gate_token_stride,
        "BETA_ELEMENTS": total_tokens * num_value_heads,
        "STATE_ELEMENTS": state_slots * slot_stride,
        "A_LOG_ELEMENTS": num_heads,
        "DT_BIAS_ELEMENTS": num_heads * HEAD_DIM,
        "CU_SEQLENS_ELEMENTS": num_seqs + 1,
        "STATE_INDEX_ELEMENTS": total_tokens,
        "NAT_ELEMENTS": num_seqs,
    }


@T.jit
def _flashkda_decode_t3_lower_bound(
    q_h: T.handle,
    k_h: T.handle,
    v_h: T.handle,
    g_h: T.handle,
    beta_h: T.handle,
    state_h: T.handle,
    out_h: T.handle,
    a_log_h: T.handle,
    dt_bias_h: T.handle,
    cu_seqlens_h: T.handle,
    ssm_state_indices_h: T.handle,
    num_accepted_tokens_h: T.handle,
    scale: T.float32,
    lower_bound: T.float32,
    *,
    NUM_SEQS: T.constexpr,
    NUM_HEADS: T.constexpr,
    NUM_VALUE_HEADS: T.constexpr,
    HEAD_RATIO: T.constexpr,
    STATE_SLOT_STRIDE: T.constexpr,
    GATE_TOKEN_STRIDE: T.constexpr,
    Q_ELEMENTS: T.constexpr,
    V_ELEMENTS: T.constexpr,
    GATE_ELEMENTS: T.constexpr,
    BETA_ELEMENTS: T.constexpr,
    STATE_ELEMENTS: T.constexpr,
    A_LOG_ELEMENTS: T.constexpr,
    DT_BIAS_ELEMENTS: T.constexpr,
    CU_SEQLENS_ELEMENTS: T.constexpr,
    STATE_INDEX_ELEMENTS: T.constexpr,
    NAT_ELEMENTS: T.constexpr,
):
    """FlashKDA "cake" T=3 lower-bound decode, WY schedule, 3 warps.

    Transcribed from `kernel_flashinfer_recurrent_kda_wy_vtile_short`; bare
    `:NNN` references are into
    `flashkda_decode_d128_t3_lower_bound_split4.cu`. Design sketch:
    `.agents/sketch/flashinfer/kda/flashkda_decode_t3_t4_wy.md`.
    """
    q = T.match_buffer(q_h, (Q_ELEMENTS,), "bfloat16", scope="global")
    k = T.match_buffer(k_h, (Q_ELEMENTS,), "bfloat16", scope="global")
    v = T.match_buffer(v_h, (V_ELEMENTS,), "bfloat16", scope="global")
    g = T.match_buffer(g_h, (GATE_ELEMENTS,), "bfloat16", scope="global")
    beta = T.match_buffer(beta_h, (BETA_ELEMENTS,), "bfloat16", scope="global")
    state = T.match_buffer(state_h, (STATE_ELEMENTS,), "bfloat16", scope="global")
    out = T.match_buffer(out_h, (V_ELEMENTS,), "bfloat16", scope="global")
    a_log = T.match_buffer(a_log_h, (A_LOG_ELEMENTS,), "float32", scope="global")
    dt_bias = T.match_buffer(dt_bias_h, (DT_BIAS_ELEMENTS,), "float32", scope="global")
    cu = T.match_buffer(cu_seqlens_h, (CU_SEQLENS_ELEMENTS,), "int32", scope="global")
    ssm_idx = T.match_buffer(ssm_state_indices_h, (STATE_INDEX_ELEMENTS,), "int32", scope="global")
    nat = T.match_buffer(num_accepted_tokens_h, (NAT_ELEMENTS,), "int32", scope="global")
    T.device_entry()

    # The source uses dynamic smem; 15872 B fits the static limit, so the same
    # bytes are declared statically at the same alignment. The region offsets
    # and the alignment are what the swizzled ldmatrix addressing depends on.
    arena = T.alloc_buffer((SMEM_TOTAL,), "uint8", scope="shared", align=1024)

    # --- work decomposition and lane roles (:142-160) ----------------------
    work, n = T.cta_id([NUM_VALUE_HEADS * VALUE_SPLIT, NUM_SEQS])
    tid = T.thread_id([THREADS])
    value_tile: T.int32 = work % VALUE_SPLIT
    hv: T.int32 = work // VALUE_SPLIT
    query_head: T.int32 = hv // HEAD_RATIO
    warp: T.int32 = tid // 32  # == token index in phases A and C
    lane: T.int32 = tid % 32
    lane_quad: T.int32 = lane % 4
    frag_row: T.int32 = lane // 4
    quad_base: T.int32 = lane - lane_quad
    group: T.int32 = tid // 16
    lane_group: T.int32 = tid % 16
    k_start: T.int32 = lane_group * 8
    elem_start: T.int32 = lane * 4
    tile_row_base: T.int32 = value_tile * ROWS_PER_CTA
    owned_row_base: T.int32 = group * 8
    token_base: T.int32 = _load_i32(cu, n)
    seq_len: T.int32 = _load_i32(cu, n + 1) - token_base

    r_q = T.alloc_local((4,), "float32")
    r_k = T.alloc_local((4,), "float32")
    r_d = T.alloc_local((4,), "float32")

    # =======================================================================
    # Phase A: token preprocess, warp <-> token  (:180-299)
    # =======================================================================
    # Unlike every other body in this family the guards here are LIVE: at 96
    # threads `warp < 3` admits all three warps, but `group < 4` (phases B, H)
    # and `warp < 2` (phases D-G) do not. Warp 2 is a token-only warp -- it
    # preprocesses token 2, builds token 2's sVec columns and sL/sR row, hits
    # both CTA barriers, and then never gathers state, never issues an MMA and
    # never stores.
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

        # Index-ordered accumulation (:250-253); the first term has a zero addend.
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
        # Two sequential full-warp butterflies, not interleaved (:254-263).
        for off in T.unroll(5):
            q_sq = _add(q_sq, _shfl_bfly(q_sq, 16 >> off))
        for off in T.unroll(5):
            k_sq = _add(k_sq, _shfl_bfly(k_sq, 16 >> off))
        q_norm: T.float32 = _mul(_rsqrt(_add(q_sq, T.float32(L2_EPS))), scale)
        k_norm: T.float32 = _rsqrt(_add(k_sq, T.float32(L2_EPS)))

        # ---- GATE_KIND == 1: the gate is derived here (:259-278) ----------
        # `g` is the RAW pre-gate, not a log-gate. Hoisted above the element
        # loop and loaded by EVERY lane -- there is no lane-0 broadcast to
        # reproduce -- and `-gate_a` is loop-invariant, so it is one
        # neg.ftz.f32 for the whole body, not one per element.
        gate_a: T.float32 = _expf(_load_f32(a_log, query_head))
        neg_gate_a: T.float32 = _neg(gate_a)

        k_pub = T.alloc_local((4,), "uint32")
        d_pub = T.alloc_local((4,), "uint32")
        for i in T.unroll(4):
            k_idx_a: T.int32 = elem_start + i
            r_q[i] = _mul(r_q[i], q_norm)
            r_k[i] = _mul(r_k[i], k_norm)
            # biased = g + dt_bias[qh*128 + k]; sigmoid; scale by lower_bound;
            # exponentiate. Operand orders follow the export's PTX exactly:
            # mul(biased, -gate_a), add(sig, 1.0), div(lower_bound, denom).
            biased: T.float32 = _add(r_d[i], _load_f32(dt_bias, query_head * HEAD_DIM + k_idx_a))
            sig: T.float32 = _expf(_mul(biased, neg_gate_a))
            r_d[i] = _expf(_div(lower_bound, _add(sig, T.float32(1.0))))
            k_pub[i] = T.reinterpret("uint32", r_k[i])
            d_pub[i] = T.reinterpret("uint32", r_d[i])
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
                # nat picks the initial checkpoint slot; at T=3 the clamp
                # ceiling is 2 and both edges are reachable (:288-295).
                accepted: T.int32 = T.min(T.max(_load_i32(nat, n) - 1, 0), NUM_TOKENS - 1)
                initial_slot: T.int32 = _load_i32(ssm_idx, n * NUM_TOKENS + accepted)
                _st_shared_i32(arena, OFF_SINIT, T.max(initial_slot, 0))

    T.cuda.cta_sync()
    # A genuine three-warp edge: warp 2 publishes token 2's sK/sD/sBeta/sSlot/
    # sToken here and warps 0,1 consume them in phases C and H.

    # =======================================================================
    # Phase B: state gather and sState stage, groups 0..3 only  (:301-338)
    # =======================================================================
    hist = T.alloc_local((8 * 8,), "float32")
    if group < ROW_GROUPS:
        init_slot: T.int32 = _ld_shared_i32(arena, OFF_SINIT)
        head_base = T.cast(init_slot, "int64") * T.cast(STATE_SLOT_STRIDE, "int64") + T.cast(
            hv * HEAD_DIM * HEAD_DIM, "int64"
        )
        for row_local in T.unroll(8):
            row_l: T.int32 = owned_row_base + row_local
            pack = _load_u32x4(
                state, head_base + T.cast((tile_row_base + row_l) * HEAD_DIM + k_start, "int64")
            )
            for pr in T.unroll(4):
                hist[row_local * 8 + 2 * pr] = _widen_lo(pack[pr])
                hist[row_local * 8 + 2 * pr + 1] = _widen_hi(pack[pr])
            # The bf16 bits go to shared unmodified; the swizzle is on the byte
            # offset. lane_group < 8 lands in sState0, the rest in sState1.
            if lane_group < 8:
                _st_shared_u32x4(arena, OFF_SSTATE0 + _swz(row_l * 128 + k_start * 2), pack)
            else:
                _st_shared_u32x4(arena, OFF_SSTATE1 + _swz(row_l * 128 + (k_start - 64) * 2), pack)

    # =======================================================================
    # Phase C: sVec columns and the WY coefficients  (:339-413)
    # =======================================================================
    if warp < NUM_TOKENS:
        token_c: T.int32 = warp
        for i in T.unroll(4):
            k_idx: T.int32 = elem_start + i
            prefix: T.float32 = T.float32(1.0)
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
            _st_shared_b16(
                arena,
                OFF_SVEC + _swz(k_idx * 32 + (4 + token_c) * 2),
                _ptx_un("cvt.rn.bf16.f32", _mul(prefix, r_q[i]), dtype="uint16"),
            )

        ratio = T.alloc_local((4,), "float32")
        for i in T.unroll(4):
            ratio[i] = T.float32(1.0)
        for source_offset in T.unroll(NUM_TOKENS):
            source_token: T.int32 = token_c - source_offset
            if source_token >= 0:
                dot_kk: T.float32 = T.float32(0.0)
                dot_qk: T.float32 = T.float32(0.0)
                sk_vec = T.alloc_local((4,), "float32")
                _ld_shared_f32x4(
                    arena, OFF_SK + (source_token * HEAD_DIM + elem_start) * 4, sk_vec, 0
                )
                for i in T.unroll(4):
                    # Source order: r * source_k * ratio  (:381-384).
                    dot_kk = _fma(_mul(r_k[i], sk_vec[i]), ratio[i], dot_kk)
                    dot_qk = _fma(_mul(r_q[i], sk_vec[i]), ratio[i], dot_qk)
                for off in T.unroll(5):
                    dot_kk = _add(dot_kk, _shfl_bfly(dot_kk, 16 >> off))
                for off in T.unroll(5):
                    dot_qk = _add(dot_qk, _shfl_bfly(dot_qk, 16 >> off))
                if lane == 0:
                    beta_source: T.float32 = _ld_shared_f32(arena, OFF_SBETA + source_token * 4)
                    if source_token < token_c:
                        _st_shared_f32(
                            arena,
                            OFF_SL + (token_c * NUM_TOKENS + source_token) * 4,
                            _mul(beta_source, dot_kk),
                        )
                    _st_shared_f32(
                        arena,
                        OFF_SR + (token_c * NUM_TOKENS + source_token) * 4,
                        _mul(beta_source, dot_qk),
                    )
                if source_token > 0:
                    sd_vec = T.alloc_local((4,), "float32")
                    _ld_shared_f32x4(
                        arena, OFF_SD + (source_token * HEAD_DIM + elem_start) * 4, sd_vec, 0
                    )
                    for i in T.unroll(4):
                        ratio[i] = _mul(ratio[i], sd_vec[i])

    T.cuda.cta_sync()
    # The second three-warp edge: warp 2 wrote sVec columns 2 and 6, sL row 2
    # and sR row 2 above and then leaves; warps 0,1 depend on that through here.

    # =======================================================================
    # Phase D: the MMA chain, warps 0,1 <-> 16 value rows  (:415-447)
    # =======================================================================
    # Byte-identical to the T=2/T=4 chain: 8 ldmatrix.x4 + 8 .trans + 8 mma.
    acc = T.alloc_local((4,), "float32", align=4)
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
                    _mma_zero(acc, state_frag, vec_frag)
                else:
                    _mma_acc(acc, state_frag, vec_frag)

    # =======================================================================
    # Phase E: quad broadcast and the WY forward substitution  (:448-492)
    # =======================================================================
    u_lo = T.alloc_local((NUM_TOKENS,), "float32")
    u_hi = T.alloc_local((NUM_TOKENS,), "float32")
    if warp < MMA_WARPS:
        # 8 broadcasts, all four ha_lo then all four ha_hi (:448-463). ha_*[3]
        # is MMA column 3 -- token 3, which does not exist at T=3 -- but the
        # source issues its two shuffles anyway, so the port does too.
        ha_lo = T.alloc_local((4,), "float32")
        ha_hi = T.alloc_local((4,), "float32")
        for t in T.unroll(4):
            ha_lo[t] = _shfl_idx(acc[t % 2], quad_base + t // 2)
        for t in T.unroll(4):
            ha_hi[t] = _shfl_idx(acc[2 + t % 2], quad_base + t // 2)

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

        # The residuals must cross the quad: phase F below runs on lane_quad
        # >= 2, so lane 4f+3 consumes what lane 4f+2 solved (:485-491). This
        # broadcast was an identity at T=2 and the T=2 port dropped it; here it
        # is load-bearing for output token 2.
        for t in T.unroll(NUM_TOKENS):
            u_lo[t] = _shfl_idx(u_lo[t], quad_base + 2)
            u_hi[t] = _shfl_idx(u_hi[t], quad_base + 2)

    # =======================================================================
    # Phase F: the outputs, two tokens per writer lane  (:493-541)
    # =======================================================================
    # Byte-identical to the T=4 body's phase F apart from the literal TOKENS.
    # No shuffle is needed to pick the accumulator column pair: the m16n8k16 D
    # layout already keys columns 2*lane_quad, +1 off lane_quad.
    #
    # DELIBERATE DEVIATION, T=3 only. TOKENS is odd, so at lane_quad == 3 the
    # source computes token1 = 3 -- out of range -- and reads sR[9..11] (which
    # aliases the first 12 bytes of sU) and sSlot[3] (which aliases sToken[0])
    # before the `token1 < 3` mask discards the results. Those reads are safe in
    # the source (in-bounds shared memory, provably dead values) but a TIRx port
    # cannot express "read past a declared region and rely on the arena layout".
    # The port predicates them on `token1 < NUM_TOKENS` instead. This is
    # numerically identical: coef1 stays 0.0, so out1_* stays acc[1]/acc[3],
    # which is never stored; slot1's -1 sentinel is never consulted because the
    # store itself is guarded by the same condition.
    if warp < MMA_WARPS and lane_quad >= 2:
        token0: T.int32 = (lane_quad - 2) * 2
        token1: T.int32 = token0 + 1
        row_lo_f: T.int32 = warp * 16 + frag_row
        row_hi_f: T.int32 = row_lo_f + 8
        out0_lo: T.float32 = acc[0]
        out1_lo: T.float32 = acc[1]
        out0_hi: T.float32 = acc[2]
        out1_hi: T.float32 = acc[3]
        for src in T.unroll(NUM_TOKENS):
            residual_lo: T.float32 = u_lo[src]
            residual_hi: T.float32 = u_hi[src]
            coef0: T.float32 = T.float32(0.0)
            coef1: T.float32 = T.float32(0.0)
            # The masked-out coefficient is a real zero-operand fma, not a
            # skipped iteration (:506-514).
            if token0 >= src:
                coef0 = _ld_shared_f32(arena, OFF_SR + (token0 * NUM_TOKENS + src) * 4)
            if token1 < NUM_TOKENS and token1 >= src:
                coef1 = _ld_shared_f32(arena, OFF_SR + (token1 * NUM_TOKENS + src) * 4)
            out0_lo = _fma(coef0, residual_lo, out0_lo)
            out1_lo = _fma(coef1, residual_lo, out1_lo)
            out0_hi = _fma(coef0, residual_hi, out0_hi)
            out1_hi = _fma(coef1, residual_hi, out1_hi)

        for half in T.unroll(2):
            token_o: T.int32 = T.if_then_else(half == 0, token0, token1)
            o_lo: T.float32 = T.if_then_else(half == 0, out0_lo, out1_lo)
            o_hi: T.float32 = T.if_then_else(half == 0, out0_hi, out1_hi)
            if token_o < NUM_TOKENS:
                active_o = _ld_shared_i32(arena, OFF_SSLOT + token_o * 4) >= 0
                base_o: T.int32 = (
                    _ld_shared_i32(arena, OFF_STOKEN + token_o * 4) * NUM_VALUE_HEADS + hv
                ) * HEAD_DIM + tile_row_base
                # A padded row writes EXPLICIT zeros; the upstream test asserts
                # them bit-exactly, so this is not an "unwritten" path.
                _store_f32_as_bf16(out, base_o + row_lo_f, o_lo, active_o)
                _store_f32_as_bf16(out, base_o + row_hi_f, o_hi, active_o)
                _store_f32_as_bf16(out, base_o + row_lo_f, T.float32(0.0), T.Not(active_o))
                _store_f32_as_bf16(out, base_o + row_hi_f, T.float32(0.0), T.Not(active_o))

    # =======================================================================
    # Phase G: publish sU  (:542-555)
    # =======================================================================
    if warp < MMA_WARPS:
        if lane_quad == 2:
            row_lo_g: T.int32 = warp * 16 + frag_row
            for t in T.unroll(NUM_TOKENS):
                _st_shared_f32(arena, OFF_SU + (t * ROWS_PER_CTA + row_lo_g) * 4, u_lo[t])
                _st_shared_f32(arena, OFF_SU + (t * ROWS_PER_CTA + row_lo_g + 8) * 4, u_hi[t])
        # A warp barrier suffices: warp w owns groups 2w and 2w+1, i.e. rows
        # [16w, 16w+16), which is exactly the row set it wrote above (:553).
        T.cuda.warp_sync()

    # =======================================================================
    # Phase H: recurrence and checkpoints, groups 0..3 only  (:669-705)
    # =======================================================================
    if group < ROW_GROUPS:
        words_w = T.alloc_local((4,), "uint32")
        sd_t = T.alloc_local((8,), "float32")
        sk_t = T.alloc_local((8,), "float32")
        for t in T.unroll(NUM_TOKENS):
            slot_t: T.int32 = _ld_shared_i32(arena, OFF_SSLOT + t * 4)
            beta_t: T.float32 = _ld_shared_f32(arena, OFF_SBETA + t * 4)
            # The gate and key slices depend only on (t, k_start), not on the
            # row, so they are loaded once per token as two 16-byte reads
            # rather than reloaded in the row loop (:683).
            _ld_shared_f32x4(arena, OFF_SD + (t * HEAD_DIM + k_start) * 4, sd_t, 0)
            _ld_shared_f32x4(arena, OFF_SD + (t * HEAD_DIM + k_start + 4) * 4, sd_t, 4)
            _ld_shared_f32x4(arena, OFF_SK + (t * HEAD_DIM + k_start) * 4, sk_t, 0)
            _ld_shared_f32x4(arena, OFF_SK + (t * HEAD_DIM + k_start + 4) * 4, sk_t, 4)
            for row_local in T.unroll(8):
                row_h: T.int32 = owned_row_base + row_local
                update: T.float32 = _mul(
                    _ld_shared_f32(arena, OFF_SU + (t * ROWS_PER_CTA + row_h) * 4), beta_t
                )
                for i in T.unroll(8):
                    # The source writes `hist*sD + update*sK` (:683) and the
                    # compiler contracts the FIRST product: update*sK is
                    # rounded, hist*sD is fused. This is the only stateful
                    # accumulation and it feeds both the checkpoint and every
                    # later token's history.
                    hist[row_local * 8 + i] = _fma(
                        hist[row_local * 8 + i], sd_t[i], _mul(update, sk_t[i])
                    )
                for pr in T.unroll(4):
                    words_w[pr] = _pack_bf16x2(
                        hist[row_local * 8 + 2 * pr + 1], hist[row_local * 8 + 2 * pr]
                    )
                # The recurrence advances unconditionally; only the store is
                # slot-predicated, and it stays FP32 so token t+1 consumes the
                # un-rounded token-t state rather than the bf16 checkpoint.
                if slot_t >= 0:
                    _store_u32x4(
                        state,
                        T.cast(slot_t, "int64") * T.cast(STATE_SLOT_STRIDE, "int64")
                        + T.cast(
                            hv * HEAD_DIM * HEAD_DIM + (tile_row_base + row_h) * HEAD_DIM + k_start,
                            "int64",
                        ),
                        words_w,
                    )


def get_kernel(**kwargs: Any):
    """Return the specialized FlashKDA cake T=3 decode PrimFunc."""
    return _flashkda_decode_t3_lower_bound.specialize(**_specialization(kwargs))


_NAT_PATTERNS = {"zeros": 0, "threes": 3, "tens": 10}


def _accepted_tensor(kind: Any, num_seqs: int, device: str) -> torch.Tensor:
    """num_accepted_tokens per case; the kernel clamps nat-1 into [0, 2]."""
    if kind is None:
        return torch.ones(num_seqs, device=device, dtype=torch.int32)
    if kind == "mixed":
        # The upstream test's sweep, tiled to the batch size.
        pattern = torch.tensor([0, 1, 3, 10, 2, 0, 1, 3], dtype=torch.int32)
        reps = (num_seqs + pattern.numel() - 1) // pattern.numel()
        return pattern.repeat(reps)[:num_seqs].to(device)
    if kind in _NAT_PATTERNS:
        return torch.full((num_seqs,), _NAT_PATTERNS[kind], device=device, dtype=torch.int32)
    raise ValueError(f"unknown accepted-token pattern {kind!r}")


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Build one deterministic case with independent TIRx / reference state.

    Follows the upstream T=3 recipe: packed [1, N*3, ...] bf16, a RAW
    pre-gate g (NOT log-space -- the kernel computes the gate itself), plus the
    fp32 A_log / dt_bias the lower-bound gate consumes.
    """
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for FlashKDA cake T=3 decode")
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
    # GATE_KIND == 1: g is the RAW pre-gate input; the kernel derives the gate
    # from it with A_log, dt_bias and lower_bound (upstream recipe
    # test_recurrent_kda_decode_export.py:_make_t3_lower_bound_case).
    g_dense = torch.randn(
        (1, total_tokens, num_value_heads, HEAD_DIM),
        device=device,
        dtype=torch.float32,
        generator=gen,
    ).to(torch.bfloat16)
    g_raw = torch.zeros((total_tokens * gate_token_stride,), device=device, dtype=torch.bfloat16)
    g = g_raw.as_strided(
        (1, total_tokens, num_value_heads, HEAD_DIM),
        (total_tokens * gate_token_stride, gate_token_stride, HEAD_DIM, 1),
    )
    g.copy_(g_dense)
    beta = torch.sigmoid(randn(1, total_tokens, num_value_heads, dtype=torch.float32, gain=0.5)).to(
        torch.bfloat16
    )

    # gate_a = exp(A_log) lands in [1, 2) with this recipe.
    a_log = torch.log(
        torch.rand((num_heads,), device=device, dtype=torch.float32, generator=gen) + 1.0
    )
    dt_bias = torch.randn(
        (num_heads * HEAD_DIM,), device=device, dtype=torch.float32, generator=gen
    )
    lower_bound = float(kwargs.get("lower_bound", -5.0))
    if not lower_bound < 0.0:
        raise ValueError("t3 lower-bound requires a finite negative lower_bound")

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
        "a_log": a_log,
        "dt_bias": dt_bias,
        "lower_bound": lower_bound,
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
        case["a_log"],
        case["dt_bias"],
        case["cu_seqlens"],
        case["ssm_state_indices"],
        case["num_accepted_tokens"],
        float(case["scale"]),
        float(case["lower_bound"]),
    )


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

    module = get_flash_kda_decode_module("d128_t3_lower_bound_split4", target)
    reference_out = torch.empty_like(case["tirx_out"])

    # Unlike the precomputed variants, GATE_KIND == 1 dereferences A_log and
    # dt_bias and uses lower_bound.
    module.run(
        case["q"], case["k"], case["v"], case["g"], case["beta"],
        case["a_log"], case["dt_bias"],
        case["reference_state"], reference_out,
        case["cu_seqlens"], case["ssm_state_indices"], case["num_accepted_tokens"],
        float(case["scale"]), float(case["lower_bound"]),
        int(torch.cuda.current_stream(device).cuda_stream),
    )  # fmt: skip
    torch.cuda.synchronize(device)
    return reference_out


def _torch_reference(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent FP32 oracle of the T=3 delta rule (sequential form)."""
    spec = case["spec"]
    num_seqs = spec["NUM_SEQS"]
    num_value_heads = spec["NUM_VALUE_HEADS"]
    head_ratio = spec["HEAD_RATIO"]
    slot_stride = spec["STATE_SLOT_STRIDE"]

    a_log = case["a_log"].float()
    dt_bias = case["dt_bias"].float().reshape(spec["NUM_HEADS"], HEAD_DIM)
    lower_bound = float(case["lower_bound"])

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
                # GATE_KIND == 1: the gate is derived here, not supplied.
                # log_gate = lower_bound * sigmoid(exp(A_log[h]) * (g + dt_bias))
                gate_a = torch.exp(a_log[h])
                biased = g[row, hv] + dt_bias[h]
                log_gate = lower_bound / (1.0 + torch.exp(-gate_a * biased))
                gamma = torch.exp(log_gate)

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
# operands to bf16, so the band here is genuinely wider and measured, not
# guessed: at scaffold time the export itself sat 3.1e-05 from the oracle.
_ORACLE_RTOL = 2.0**-5
_ORACLE_ATOL = 6.0e-3


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
        msg=lambda m: f"output vs flashinfer cake export\n{m}",
    )
    torch.testing.assert_close(
        tirx_state.float(),
        case["reference_state_raw"].float(),
        rtol=_RTOL,
        atol=_ATOL,
        msg=lambda m: f"state vs flashinfer cake export\n{m}",
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
    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench

    case = prepare_data(**kwargs)
    executable = compile_kernel(get_kernel(**kwargs))
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
