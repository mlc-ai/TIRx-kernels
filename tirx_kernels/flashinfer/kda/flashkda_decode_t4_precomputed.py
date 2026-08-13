# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400),
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's FlashKDA "cake" T=4 precomputed-gate decode kernel.

Source: ``csrc/kda/flashkda_decode_d128_t4_precomputed_split2.cu``, symbol
``kernel_flashinfer_recurrent_kda_wy_vtile_short`` -- the same frozen WY
template as the ported T=2 sibling, re-instantiated at TOKENS=4 and
VALUE_SPLIT=2.

Dispatch reaches this body whenever ``num_tokens == 4`` with
``num_spec_tokens == 3``. The sm100a selector returns value split 2
unconditionally at T=4 (``recurrent_kda.py:1179-1180``), so
``d128_t4_precomputed_split2`` is the only reachable -- and only locally
measurable -- specialization.

Relative to T=2 the geometry doubles (128 threads = 4 warps, 64 value rows per
CTA, a 25856-byte arena) and three things that were dead at T=2 come alive: MMA
accumulator columns 2,3 and 6,7, the ``quad_base + 1`` broadcasts, and the
``u_lo``/``u_hi`` quad broadcasts -- Phase F now runs on ``lane_quad >= 2``,
where lane ``4f+2`` writes tokens 0,1 and lane ``4f+3`` writes tokens 2,3, so
the WY residuals have to cross the quad.

Helper vocabulary is shared with the T=2 module; only the geometry constants,
the frozen digest and the kernel body are per-specialization.
"""

from __future__ import annotations

import hashlib
import os
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

# --- source pinning -------------------------------------------------------
# Raw and normalized digests differ for this export; the hash of the bytes
# between the BEGIN/END markers is the raw one (verified).
FROZEN_FLASHINFER_COMMIT = "f2e04400"
FROZEN_BODY_SHA256 = "6ee50e300a2f20f305252a49e3e84fde957542ccb97b7fcb7ba7f075c7733077"

HEAD_DIM = _t2.HEAD_DIM
NUM_TOKENS = 4
VALUE_SPLIT = 2  # the sm100a selector returns 2 unconditionally at T = 4
L2_EPS = _t2.L2_EPS
LOG2_E = _t2.LOG2_E

THREADS = 128  # 4 warps (flash_kda_decode.py:_variant_metadata)
ROWS_PER_CTA = HEAD_DIM // VALUE_SPLIT  # 64 value rows per CTA
MMA_WARPS = HEAD_DIM // VALUE_SPLIT // 16  # phase D-G guard: value_rows/16
ROW_GROUPS = HEAD_DIM // VALUE_SPLIT // 8  # phase B/H guard: value_rows/8
ROWS_PER_THREAD = 8  # each of the 128 threads owns 8 rows x 8 keys
K_PER_THREAD = 8

# Arena offsets copied from the source's #define block (.cu:45-88).
OFF_SSTATE0 = 0
OFF_SSTATE1 = 8192
OFF_SVEC = 16384
OFF_SK = 20480
OFF_SD = 22528
OFF_SBETA = 24576
OFF_SSLOT = 24592
OFF_STOKEN = 24608
OFF_SINIT = 24624
OFF_SL = 24640
OFF_SR = 24704
OFF_SU = 24768
SMEM_TOTAL = 25856
# sGramA0/sGramA1 alias sVec and are t5/t6 machinery -- dead here.

# TIRX_TRANSCRIBE_START flashkda_decode_t4_precomputed


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    """One config row. Defaults mirror FlashInfer's T=4 export bench."""
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
        "seed": 20260814,
    }
    config.update(overrides)
    return config


# Every T=4 shape selects split2, so the matrix varies only (HV, H, N).
# hv32h16 mirrors FlashInfer's own export bench exactly.
BENCH_CONFIGS = [
    _case("hv32h16_b8_t4", num_seqs=8),
    _case("hv32h16_b16_t4", num_seqs=16),
    _case("hv32h16_b32_t4", num_seqs=32),
    _case("hv32h16_b64_t4", num_seqs=64),
    _case("hv32h16_b128_t4", num_seqs=128),
    _case("hv16h16_b8_t4", num_seqs=8, num_value_heads=16),
    _case("hv16h16_b64_t4", num_seqs=64, num_value_heads=16),
    _case("hv12h12_b8_t4", num_seqs=8, num_heads=12, num_value_heads=12),
    _case("hv12h12_b64_t4", num_seqs=64, num_heads=12, num_value_heads=12),
]

CONFIGS = [dict(cfg) for cfg in BENCH_CONFIGS] + [
    # nat selects the initial checkpoint slot as ssm_idx[n*4 + clamp(nat-1, 0, 3)];
    # the upstream test sweeps 0/1/4/11, covering both clamp edges (.cu:283-286).
    _case("hv32h16_b8_t4_nat0", num_seqs=8, accepted="zeros"),
    _case("hv32h16_b8_t4_nat4", num_seqs=8, accepted="fours"),
    _case("hv32h16_b8_t4_nat11", num_seqs=8, accepted="elevens"),
    _case("hv32h16_b8_t4_natmix", num_seqs=8, accepted="mixed"),
    _case("hv32h16_b8_t4_padded", num_seqs=8, padded_seqs=2),
    _case("hv32h16_b8_t4_strided", num_seqs=8, slot_stride_pad=8),
    _case("hv32h16_b8_t4_gstride", num_seqs=8, gate_token_stride_pad=8),
    _case("hv32h16_b8_t4_scale", num_seqs=8, scale=0.05),
    _case("hv64h16_b8_t4", num_seqs=8, num_value_heads=64),
    _case("hv32h16_b1_t4", num_seqs=1),
]

KERNEL_META = {
    "name": "flashkda_decode_t4_precomputed",
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

    if num_value_heads % num_heads != 0 or num_value_heads < num_heads:
        raise ValueError("num_value_heads must be a multiple of num_heads and >= it")
    if not 0 < num_seqs <= 65535:
        raise ValueError("num_seqs must fit grid.y (binding_common.cuh:284-287)")
    if slot_stride_pad % 8 != 0:
        raise ValueError("state slot stride padding must stay 8-element aligned")
    if gate_token_stride_pad % 4 != 0:
        raise ValueError("gate token stride padding must stay 4-element aligned")
    if VALUE_SPLIT != 2:
        raise ValueError("only d128_t4_precomputed_split2 is in this port's scope")

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
        "CU_SEQLENS_ELEMENTS": num_seqs + 1,
        "STATE_INDEX_ELEMENTS": total_tokens,
        "NAT_ELEMENTS": num_seqs,
    }


@T.jit
def _flashkda_decode_t4_precomputed(
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
):
    """FlashKDA "cake" T=4 precomputed-gate decode, WY schedule, 4 warps.

    Transcribed from `kernel_flashinfer_recurrent_kda_wy_vtile_short`; bare
    `:NNN` references are into
    `flashkda_decode_d128_t4_precomputed_split2.cu`. Design sketch:
    `.agents/sketch/flashinfer/kda/flashkda_decode_t3_t4_wy.md`.
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

    # The source uses dynamic smem; 25856 B fits the static limit, so the same
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
    # Phase A: token preprocess, warp <-> token  (:180-290)
    # =======================================================================
    # `warp < 4` is statically true at 128 threads, as is `group < 8` below:
    # every guard in this body is satisfied by every thread. (The T=3 sibling
    # is the one where these are live.)
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

        # Index-ordered accumulation (:241-244); the first term has a zero addend.
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
                # nat picks the initial checkpoint slot; at T=4 the clamp
                # ceiling is 3 and both edges are reachable (:279-287).
                accepted: T.int32 = T.min(T.max(_load_i32(nat, n) - 1, 0), NUM_TOKENS - 1)
                initial_slot: T.int32 = _load_i32(ssm_idx, n * NUM_TOKENS + accepted)
                _st_shared_i32(arena, OFF_SINIT, T.max(initial_slot, 0))

    T.cuda.cta_sync()

    # =======================================================================
    # Phase B: state gather and sState stage, all 128 threads  (:292-329)
    # =======================================================================
    init_slot: T.int32 = _ld_shared_i32(arena, OFF_SINIT)
    head_base = T.cast(init_slot, "int64") * T.cast(STATE_SLOT_STRIDE, "int64") + T.cast(
        hv * HEAD_DIM * HEAD_DIM, "int64"
    )
    hist = T.alloc_local((8 * 8,), "float32")
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
    # Phase C: sVec columns and the WY coefficients  (:330-404)
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
                    # Source order: r * source_k * ratio  (:372-375).
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

    # =======================================================================
    # Phase D: the MMA chain, warp <-> 16 value rows  (:406-438)
    # =======================================================================
    # The chain is byte-identical at every T: 8 ldmatrix.x4 + 8 .trans + 8 mma.
    # n = 8 was always 8 columns wide; T only decides how many carry live data.
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
    # Phase E: quad broadcast and the WY forward substitution  (:439-483)
    # =======================================================================
    u_lo = T.alloc_local((NUM_TOKENS,), "float32")
    u_hi = T.alloc_local((NUM_TOKENS,), "float32")
    if warp < MMA_WARPS:
        # 8 broadcasts, all four ha_lo then all four ha_hi (:439-454). The
        # `quad_base + 1` half feeds MMA columns 2,3 -- dead at T=2, live here.
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
        # >= 2, so lane 4f+3 consumes what lane 4f+2 solved (:476-482). This
        # broadcast was an identity at T=2 and the T=2 port dropped it; here it
        # is load-bearing for every output token >= 2.
        for t in T.unroll(NUM_TOKENS):
            u_lo[t] = _shfl_idx(u_lo[t], quad_base + 2)
            u_hi[t] = _shfl_idx(u_hi[t], quad_base + 2)

    # =======================================================================
    # Phase F: the outputs, two tokens per writer lane  (:484-532)
    # =======================================================================
    # One generic block keyed off lane_quad, replacing T=2's per-token blocks.
    # No shuffle is needed to pick the right accumulator column pair: the
    # m16n8k16 D layout already keys columns 2*lane_quad, +1 off lane_quad, so
    # lane 4f+2 holds the q-side columns of tokens 0,1 and lane 4f+3 those of
    # tokens 2,3.
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
            # skipped iteration (:497-505).
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

    # =======================================================================
    # Phase G: publish sU  (:533-546)
    # =======================================================================
    if warp < MMA_WARPS:
        if lane_quad == 2:
            row_lo_g: T.int32 = warp * 16 + frag_row
            for t in T.unroll(NUM_TOKENS):
                _st_shared_f32(arena, OFF_SU + (t * ROWS_PER_CTA + row_lo_g) * 4, u_lo[t])
                _st_shared_f32(arena, OFF_SU + (t * ROWS_PER_CTA + row_lo_g + 8) * 4, u_hi[t])
        # A warp barrier suffices: warp w owns groups 2w and 2w+1, i.e. rows
        # [16w, 16w+16), which is exactly the row set it wrote above (:544).
        T.cuda.warp_sync()

    # =======================================================================
    # Phase H: recurrence and checkpoints, all 128 threads  (:660-696)
    # =======================================================================
    words_w = T.alloc_local((4,), "uint32")
    sd_t = T.alloc_local((8,), "float32")
    sk_t = T.alloc_local((8,), "float32")
    for t in T.unroll(NUM_TOKENS):
        slot_t: T.int32 = _ld_shared_i32(arena, OFF_SSLOT + t * 4)
        beta_t: T.float32 = _ld_shared_f32(arena, OFF_SBETA + t * 4)
        # The gate and key slices depend only on (t, k_start), not on the row,
        # so they are loaded once per token as two 16-byte reads rather than
        # reloaded in the row loop (:674).
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
                # The source writes `hist*sD + update*sK` (:674) and the
                # compiler contracts the FIRST product: update*sK is rounded,
                # hist*sD is fused. This is the only stateful accumulation and
                # it feeds both the checkpoint and every later token's history.
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
    """Return the specialized FlashKDA cake T=4 decode PrimFunc."""
    return _flashkda_decode_t4_precomputed.specialize(**_specialization(kwargs))


_NAT_PATTERNS = {None: None, "zeros": 0, "fours": 4, "elevens": 11}


def _accepted_tensor(kind: Any, num_seqs: int, device: str) -> torch.Tensor:
    """num_accepted_tokens per case; the kernel clamps nat-1 into [0, 3]."""
    if kind is None:
        return torch.ones(num_seqs, device=device, dtype=torch.int32)
    if kind == "mixed":
        # The upstream test's sweep, tiled to the batch size.
        pattern = torch.tensor([0, 1, 4, 11, 3, 0, 1, 4], dtype=torch.int32)
        reps = (num_seqs + pattern.numel() - 1) // pattern.numel()
        return pattern.repeat(reps)[:num_seqs].to(device)
    if kind in _NAT_PATTERNS:
        return torch.full((num_seqs,), _NAT_PATTERNS[kind], device=device, dtype=torch.int32)
    raise ValueError(f"unknown accepted-token pattern {kind!r}")


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Build one deterministic case with independent TIRx / reference state.

    Same shape as the T=2 recipe at T=4: packed [1, N*4, ...] bf16, a log-space
    gate, pre-sigmoided beta, flat [N*4] slot indices. Written locally rather
    than parameterizing the already-merged T=2 module.
    """
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for FlashKDA cake T=4 decode")
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


def _source_body_path() -> str:
    """Absolute path of the frozen generated body this port transcribes."""
    import flashinfer

    root = os.path.dirname(os.path.dirname(os.path.abspath(flashinfer.__file__)))
    return os.path.join(root, "csrc", "kda", "flashkda_decode_d128_t4_precomputed_split2.cu")


def assert_frozen_source() -> None:
    """Fail loudly if the upstream generated body was regenerated."""
    path = _source_body_path()
    with open(path, "rb") as handle:
        text = handle.read().decode()
    marker = "// BEGIN FROZEN GENERATED BODY\n"
    start = text.index(marker) + len(marker)
    end = text.index("// END FROZEN GENERATED BODY")
    digest = hashlib.sha256(text[start:end].encode()).hexdigest()
    if digest != FROZEN_BODY_SHA256:
        raise AssertionError(
            f"{path}: frozen body digest {digest} != pinned {FROZEN_BODY_SHA256}; the "
            "upstream export was regenerated and this port must be re-verified"
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

    module = get_flash_kda_decode_module("d128_t4_precomputed_split2", target)
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
    """Independent FP32 oracle of the T=4 delta rule (sequential form)."""
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
# operands to bf16, so the band here is genuinely wider and measured, not
# guessed: at scaffold time the export itself sat 6.1e-05 from the oracle.
_ORACLE_RTOL = 2.0**-5
_ORACLE_ATOL = 6.0e-3


def run_test(**kwargs: Any) -> None:
    """Validate one config against the frozen export and an independent oracle."""
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    spec = case["spec"]
    assert_frozen_source()

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
    "assert_frozen_source",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
