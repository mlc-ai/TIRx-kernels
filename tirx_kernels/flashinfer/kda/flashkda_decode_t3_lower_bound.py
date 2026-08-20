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

from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K

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
    out = K.alloc_local((1,), "uint32")
    K.evaluate(K.ptx.ld.global_.nc.b32(out[0], buffer.ptr_to([index])))
    return K.reinterpret("float32", out[0])


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

# The source requests this exact dynamic-smem size.  K owns the semantic
# regions and their layouts below; the four-byte tail is launch-contract padding.
SMEM_TOTAL = 15872


def _store_smem_f32(buffer, index, value):
    K.ptx.st.shared.b32(buffer.ptr_to([index]), K.reinterpret("uint32", value))


def _load_smem_f32(buffer, index):
    out = K.alloc_local((1,), K.u32)
    K.ptx.ld.shared.b32(out[0], buffer.ptr_to([index]))
    return K.reinterpret("float32", out[0])


def _store_smem_i32(buffer, index, value):
    K.ptx.st.shared.b32(buffer.ptr_to([index]), K.reinterpret("uint32", value))


def _load_smem_i32(buffer, index):
    out = K.alloc_local((1,), K.u32)
    K.ptx.ld.shared.b32(out[0], buffer.ptr_to([index]))
    return K.reinterpret("int32", out[0])


def _load_smem_f32x4(buffer, index, dst, base):
    words = K.alloc_local((4,), K.u32)
    K.ptx.ld.shared.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
    for i in range(4):
        K.buffer_store(dst, K.reinterpret("float32", words[i]), [base + i])


def _store_smem_u32x4_at(ptr, words):
    K.ptx.st.shared.v4.b32(ptr, words[0], words[1], words[2], words[3])


def _store_smem_b16_at(ptr, bits):
    K.ptx.st.shared.b16(ptr, bits)


def _ldmatrix_x4_at(ptr, frag, trans: bool):
    if trans:
        K.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            frag[0], frag[1], frag[2], frag[3], ptr
        )
    else:
        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(frag[0], frag[1], frag[2], frag[3], ptr)


def _svec_ptr(s_vec, k_index, column):
    """Logical ``[HEAD_DIM, 16]`` view over one 128B-swizzled 4 KiB tile."""
    row = K.shift_right(k_index, K.int32(2))
    col = K.bitwise_and(k_index, K.int32(3)) * 16 + column
    return s_vec.ptr_to(row, col)


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


def _make_flashkda_decode_t3_lower_bound(spec: dict[str, Any]):
    """Trace the T=3 WY schedule with an explicit token-only warp role."""
    NUM_SEQS = spec["NUM_SEQS"]
    NUM_HEADS = spec["NUM_HEADS"]
    NUM_VALUE_HEADS = spec["NUM_VALUE_HEADS"]
    HEAD_RATIO = spec["HEAD_RATIO"]
    STATE_SLOT_STRIDE = spec["STATE_SLOT_STRIDE"]
    GATE_TOKEN_STRIDE = spec["GATE_TOKEN_STRIDE"]

    @K.kernel(warps=THREADS // 32, arch="sm_100a", grid=(NUM_VALUE_HEADS * VALUE_SPLIT, NUM_SEQS))
    def _flashkda_decode_t3_lower_bound(
        q: K.gptr[K.bf16],
        k: K.gptr[K.bf16],
        v: K.gptr[K.bf16],
        g: K.gptr[K.bf16],
        beta: K.gptr[K.bf16],
        state: K.gptr[K.bf16],
        out: K.gptr[K.bf16],
        a_log: K.gptr[K.f32],
        dt_bias: K.gptr[K.f32],
        cu: K.gptr[K.i32],
        ssm_idx: K.gptr[K.i32],
        nat: K.gptr[K.i32],
        scale: K.f32,
        lower_bound: K.f32,
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
        smem.commit(SMEM_TOTAL)

        roles = K.specialize()
        compute = roles.role("compute", warps=[0, 1])
        roles.role("token_only", warps=[2])

        work, n = K.cta_id()
        warp = K.warp_id()
        lane = K.lane_id()
        value_tile = K.alloc_local((1,), K.i32)
        hv = K.alloc_local((1,), K.i32)
        query_head = K.alloc_local((1,), K.i32)
        lane_quad = K.alloc_local((1,), K.i32)
        frag_row = K.alloc_local((1,), K.i32)
        quad_base = K.alloc_local((1,), K.i32)
        elem_start = K.alloc_local((1,), K.i32)
        tile_row_base = K.alloc_local((1,), K.i32)
        token_base = K.alloc_local((1,), K.i32)
        seq_len = K.alloc_local((1,), K.i32)
        K.assign(value_tile[0], work % VALUE_SPLIT)
        K.assign(hv[0], work // VALUE_SPLIT)
        K.assign(query_head[0], hv[0] // HEAD_RATIO)
        K.assign(lane_quad[0], lane % 4)
        K.assign(frag_row[0], lane // 4)
        K.assign(quad_base[0], lane - lane_quad[0])
        K.assign(elem_start[0], lane * 4)
        K.assign(tile_row_base[0], value_tile[0] * ROWS_PER_CTA)
        K.assign(token_base[0], _load_i32(cu, K.cast(n, "int64")))
        K.assign(seq_len[0], _load_i32(cu, K.cast(n + 1, "int64")) - token_base[0])

        r_q = K.alloc_local((4,), K.f32)
        r_k = K.alloc_local((4,), K.f32)
        r_d = K.alloc_local((4,), K.f32)

        # Phase A: all three warps preprocess one token each.
        token = warp
        active_token = token < seq_len[0]
        token_pos = K.alloc_local((1,), K.i32)
        qk_base = K.alloc_local((1,), K.i32)
        gate_base = K.alloc_local((1,), K.i32)
        K.assign(token_pos[0], K.if_then_else(active_token, token_base[0] + token, 0))
        K.assign(qk_base[0], (token_pos[0] * NUM_HEADS + query_head[0]) * HEAD_DIM + elem_start[0])
        K.assign(gate_base[0], token_pos[0] * GATE_TOKEN_STRIDE + hv[0] * HEAD_DIM + elem_start[0])
        q_words = _load_u32x2(q, K.cast(qk_base[0], "int64"))
        k_words = _load_u32x2(k, K.cast(qk_base[0], "int64"))
        g_words = _load_u32x2(g, K.cast(gate_base[0], "int64"))
        for pair in range(2):
            K.ptx.mov.b32(r_q[2 * pair], _widen_lo(q_words[pair]))
            K.ptx.mov.b32(r_q[2 * pair + 1], _widen_hi(q_words[pair]))
            K.ptx.mov.b32(r_k[2 * pair], _widen_lo(k_words[pair]))
            K.ptx.mov.b32(r_k[2 * pair + 1], _widen_hi(k_words[pair]))
            K.ptx.mov.b32(r_d[2 * pair], _widen_lo(g_words[pair]))
            K.ptx.mov.b32(r_d[2 * pair + 1], _widen_hi(g_words[pair]))

        q_sq = _fma(
            r_q[3],
            r_q[3],
            _fma(r_q[2], r_q[2], _fma(r_q[1], r_q[1], _fma(r_q[0], r_q[0], K.float32(0.0)))),
        )
        k_sq = _fma(
            r_k[3],
            r_k[3],
            _fma(r_k[2], r_k[2], _fma(r_k[1], r_k[1], _fma(r_k[0], r_k[0], K.float32(0.0)))),
        )
        for off in range(5):
            q_sq = _add(q_sq, _shfl_bfly(q_sq, 16 >> off))
        for off in range(5):
            k_sq = _add(k_sq, _shfl_bfly(k_sq, 16 >> off))
        q_norm = _mul(_rsqrt(_add(q_sq, K.float32(L2_EPS))), scale)
        k_norm = _rsqrt(_add(k_sq, K.float32(L2_EPS)))
        gate_a = _expf(_load_f32(a_log, K.cast(query_head[0], "int64")))
        neg_gate_a = _neg(gate_a)

        k_pub = K.alloc_local((4,), K.u32)
        d_pub = K.alloc_local((4,), K.u32)
        for i in range(4):
            k_idx_a = elem_start[0] + i
            K.ptx.mov.b32(r_q[i], _mul(r_q[i], q_norm))
            K.ptx.mov.b32(r_k[i], _mul(r_k[i], k_norm))
            biased = _add(
                r_d[i], _load_f32(dt_bias, K.cast(query_head[0] * HEAD_DIM + k_idx_a, "int64"))
            )
            sig = _expf(_mul(biased, neg_gate_a))
            K.ptx.mov.b32(r_d[i], _expf(_div(lower_bound, _add(sig, K.float32(1.0)))))
            K.ptx.mov.b32(k_pub[i], K.reinterpret("uint32", r_k[i]))
            K.ptx.mov.b32(d_pub[i], K.reinterpret("uint32", r_d[i]))
        _store_smem_u32x4_at(s_k.ptr_to([token * HEAD_DIM + elem_start[0]]), k_pub)
        _store_smem_u32x4_at(s_d.ptr_to([token * HEAD_DIM + elem_start[0]]), d_pub)

        with K.If(lane == 0), K.Then():
            raw_slot = _load_i32(ssm_idx, K.cast(n * NUM_TOKENS + token, "int64"))
            _store_smem_i32(s_slot, token, K.if_then_else(active_token, raw_slot, -1))
            _store_smem_i32(s_token, token, token_pos[0])
            _store_smem_f32(
                s_beta,
                token,
                _load_bf16_f32(beta, K.cast(token_pos[0] * NUM_VALUE_HEADS + hv[0], "int64")),
            )
            with K.If(token == 0), K.Then():
                accepted = K.min(K.max(_load_i32(nat, K.cast(n, "int64")) - 1, 0), NUM_TOKENS - 1)
                initial_slot = _load_i32(ssm_idx, K.cast(n * NUM_TOKENS + accepted, "int64"))
                _store_smem_i32(s_init, 0, K.max(initial_slot, 0))

        K.cuda.cta_sync()

        # Phase B: only the compute role gathers state.
        hist = K.alloc_local((64,), K.f32)
        with compute:
            compute_group = K.tid_in_role() // 16
            lane_group = K.tid_in_role() % 16
            k_start = lane_group * 8
            owned_row_base = compute_group * 8
            init_slot = _load_smem_i32(s_init, 0)
            head_base = K.alloc_local((1,), K.i64)
            K.assign(
                head_base[0],
                K.cast(init_slot, "int64") * K.cast(STATE_SLOT_STRIDE, "int64")
                + K.cast(hv[0] * HEAD_DIM * HEAD_DIM, "int64"),
            )
            for row_local in range(8):
                row_l = owned_row_base + row_local
                pack = _load_u32x4(
                    state,
                    head_base[0] + K.cast((tile_row_base[0] + row_l) * HEAD_DIM + k_start, "int64"),
                )
                for pr in range(4):
                    K.ptx.mov.b32(hist[row_local * 8 + 2 * pr], _widen_lo(pack[pr]))
                    K.ptx.mov.b32(hist[row_local * 8 + 2 * pr + 1], _widen_hi(pack[pr]))
                with K.If(lane_group < 8):
                    with K.Then():
                        _store_smem_u32x4_at(s_state0.ptr_to(row_l, k_start), pack)
                    with K.Else():
                        _store_smem_u32x4_at(s_state1.ptr_to(row_l, k_start - 64), pack)

        # Phase C: token-only warp 2 remains active through the second edge.
        token_c = warp
        for i in range(4):
            k_idx = elem_start[0] + i
            prefix = K.alloc_local((1,), K.f32)
            K.assign(prefix[0], K.float32(1.0))
            for j in range(NUM_TOKENS):
                with K.If(token_c >= j), K.Then():
                    K.assign(prefix[0], _mul(prefix[0], _load_smem_f32(s_d, j * HEAD_DIM + k_idx)))
            _store_smem_b16_at(
                _svec_ptr(s_vec, k_idx, token_c),
                _ptx_un("cvt.rn.bf16.f32", _mul(prefix[0], r_k[i]), dtype="uint16"),
            )
            _store_smem_b16_at(
                _svec_ptr(s_vec, k_idx, 4 + token_c),
                _ptx_un("cvt.rn.bf16.f32", _mul(prefix[0], r_q[i]), dtype="uint16"),
            )

        ratio = K.alloc_local((4,), K.f32)
        for i in range(4):
            K.ptx.mov.b32(ratio[i], K.float32(1.0))
        for source_offset in range(NUM_TOKENS):
            source_token = token_c - source_offset
            with K.If(source_token >= 0), K.Then():
                dot_kk = K.alloc_local((1,), K.f32)
                dot_qk = K.alloc_local((1,), K.f32)
                sk_vec = K.alloc_local((4,), K.f32)
                K.assign(dot_kk[0], K.float32(0.0))
                K.assign(dot_qk[0], K.float32(0.0))
                _load_smem_f32x4(s_k, source_token * HEAD_DIM + elem_start[0], sk_vec, 0)
                for i in range(4):
                    K.assign(dot_kk[0], _fma(_mul(r_k[i], sk_vec[i]), ratio[i], dot_kk[0]))
                    K.assign(dot_qk[0], _fma(_mul(r_q[i], sk_vec[i]), ratio[i], dot_qk[0]))
                for off in range(5):
                    K.assign(dot_kk[0], _add(dot_kk[0], _shfl_bfly(dot_kk[0], 16 >> off)))
                for off in range(5):
                    K.assign(dot_qk[0], _add(dot_qk[0], _shfl_bfly(dot_qk[0], 16 >> off)))
                with K.If(lane == 0), K.Then():
                    beta_source = _load_smem_f32(s_beta, source_token)
                    with K.If(source_token < token_c), K.Then():
                        _store_smem_f32(
                            s_l, token_c * NUM_TOKENS + source_token, _mul(beta_source, dot_kk[0])
                        )
                    _store_smem_f32(
                        s_r, token_c * NUM_TOKENS + source_token, _mul(beta_source, dot_qk[0])
                    )
                with K.If(source_token > 0), K.Then():
                    sd_vec = K.alloc_local((4,), K.f32)
                    _load_smem_f32x4(s_d, source_token * HEAD_DIM + elem_start[0], sd_vec, 0)
                    for i in range(4):
                        K.ptx.mov.b32(ratio[i], _mul(ratio[i], sd_vec[i]))

        K.cuda.cta_sync()

        # Phases D-H belong solely to the two compute warps.
        with compute:
            compute_warp = K.warp_id_in_role()
            compute_group = K.tid_in_role() // 16
            lane_group = K.tid_in_role() % 16
            k_start = lane_group * 8
            owned_row_base = compute_group * 8

            acc = K.alloc_local((4,), K.f32, align=4)
            vec_frag = K.alloc_local((4,), K.u32, align=4)
            state_frag = K.alloc_local((4,), K.u32, align=4)
            for state_half in range(2):
                for mma_step in range(4):
                    mma_k = mma_step * 16
                    global_k = state_half * 64 + mma_k
                    _ldmatrix_x4_at(
                        _svec_ptr(s_vec, global_k + lane % 16, lane // 16 * 8), vec_frag, True
                    )
                    state_tile = s_state0 if state_half == 0 else s_state1
                    _ldmatrix_x4_at(
                        state_tile.ptr_to(compute_warp * 16 + lane % 16, mma_k + lane // 16 * 8),
                        state_frag,
                        False,
                    )
                    if state_half == 0 and mma_step == 0:
                        _mma_zero(acc, state_frag, vec_frag)
                    else:
                        _mma_acc(acc, state_frag, vec_frag)

            u_lo = K.alloc_local((NUM_TOKENS,), K.f32)
            u_hi = K.alloc_local((NUM_TOKENS,), K.f32)
            ha_lo = K.alloc_local((4,), K.f32)
            ha_hi = K.alloc_local((4,), K.f32)
            for t in range(4):
                K.ptx.mov.b32(ha_lo[t], _shfl_idx(acc[t % 2], quad_base[0] + t // 2))
            for t in range(4):
                K.ptx.mov.b32(ha_hi[t], _shfl_idx(acc[2 + t % 2], quad_base[0] + t // 2))
            with K.If(lane_quad[0] == 2), K.Then():
                row_lo = compute_warp * 16 + frag_row[0]
                row_hi = row_lo + 8
                for t in range(NUM_TOKENS):
                    base_t = K.alloc_local((1,), K.i32)
                    solved_lo = K.alloc_local((1,), K.f32)
                    solved_hi = K.alloc_local((1,), K.f32)
                    K.assign(
                        base_t[0], (_load_smem_i32(s_token, t) * NUM_VALUE_HEADS + hv[0]) * HEAD_DIM
                    )
                    K.assign(
                        solved_lo[0],
                        _sub(
                            _load_bf16_f32(
                                v, K.cast(base_t[0] + tile_row_base[0] + row_lo, "int64")
                            ),
                            ha_lo[t],
                        ),
                    )
                    K.assign(
                        solved_hi[0],
                        _sub(
                            _load_bf16_f32(
                                v, K.cast(base_t[0] + tile_row_base[0] + row_hi, "int64")
                            ),
                            ha_hi[t],
                        ),
                    )
                    for prev in range(t):
                        lts = _load_smem_f32(s_l, t * NUM_TOKENS + prev)
                        K.assign(solved_lo[0], _sub(solved_lo[0], _mul(lts, u_lo[prev])))
                        K.assign(solved_hi[0], _sub(solved_hi[0], _mul(lts, u_hi[prev])))
                    K.ptx.mov.b32(u_lo[t], solved_lo[0])
                    K.ptx.mov.b32(u_hi[t], solved_hi[0])
            for t in range(NUM_TOKENS):
                K.ptx.mov.b32(u_lo[t], _shfl_idx(u_lo[t], quad_base[0] + 2))
                K.ptx.mov.b32(u_hi[t], _shfl_idx(u_hi[t], quad_base[0] + 2))

            with K.If(lane_quad[0] >= 2), K.Then():
                token0 = (lane_quad[0] - 2) * 2
                token1 = token0 + 1
                row_lo_f = compute_warp * 16 + frag_row[0]
                row_hi_f = row_lo_f + 8
                out0_lo = K.alloc_local((1,), K.f32)
                out1_lo = K.alloc_local((1,), K.f32)
                out0_hi = K.alloc_local((1,), K.f32)
                out1_hi = K.alloc_local((1,), K.f32)
                K.assign(out0_lo[0], acc[0])
                K.assign(out1_lo[0], acc[1])
                K.assign(out0_hi[0], acc[2])
                K.assign(out1_hi[0], acc[3])
                for src in range(NUM_TOKENS):
                    coef0 = K.alloc_local((1,), K.f32)
                    coef1 = K.alloc_local((1,), K.f32)
                    K.assign(coef0[0], K.float32(0.0))
                    K.assign(coef1[0], K.float32(0.0))
                    with K.If(token0 >= src), K.Then():
                        K.assign(coef0[0], _load_smem_f32(s_r, token0 * NUM_TOKENS + src))
                    with K.If(K.And(token1 < NUM_TOKENS, token1 >= src)), K.Then():
                        K.assign(coef1[0], _load_smem_f32(s_r, token1 * NUM_TOKENS + src))
                    K.assign(out0_lo[0], _fma(coef0[0], u_lo[src], out0_lo[0]))
                    K.assign(out1_lo[0], _fma(coef1[0], u_lo[src], out1_lo[0]))
                    K.assign(out0_hi[0], _fma(coef0[0], u_hi[src], out0_hi[0]))
                    K.assign(out1_hi[0], _fma(coef1[0], u_hi[src], out1_hi[0]))
                for half in range(2):
                    token_o = token0 if half == 0 else token1
                    o_lo = out0_lo[0] if half == 0 else out1_lo[0]
                    o_hi = out0_hi[0] if half == 0 else out1_hi[0]
                    with K.If(token_o < NUM_TOKENS), K.Then():
                        active_o = _load_smem_i32(s_slot, token_o) >= 0
                        base_o = K.alloc_local((1,), K.i32)
                        K.assign(
                            base_o[0],
                            (_load_smem_i32(s_token, token_o) * NUM_VALUE_HEADS + hv[0]) * HEAD_DIM
                            + tile_row_base[0],
                        )
                        _store_f32_as_bf16(
                            out, K.cast(base_o[0] + row_lo_f, "int64"), o_lo, active_o
                        )
                        _store_f32_as_bf16(
                            out, K.cast(base_o[0] + row_hi_f, "int64"), o_hi, active_o
                        )
                        _store_f32_as_bf16(
                            out,
                            K.cast(base_o[0] + row_lo_f, "int64"),
                            K.float32(0.0),
                            K.Not(active_o),
                        )
                        _store_f32_as_bf16(
                            out,
                            K.cast(base_o[0] + row_hi_f, "int64"),
                            K.float32(0.0),
                            K.Not(active_o),
                        )

            with K.If(lane_quad[0] == 2), K.Then():
                row_lo_g = compute_warp * 16 + frag_row[0]
                for t in range(NUM_TOKENS):
                    _store_smem_f32(s_u, t * ROWS_PER_CTA + row_lo_g, u_lo[t])
                    _store_smem_f32(s_u, t * ROWS_PER_CTA + row_lo_g + 8, u_hi[t])
            K.cuda.warp_sync()

            words_w = K.alloc_local((4,), K.u32)
            sd_t = K.alloc_local((8,), K.f32)
            sk_t = K.alloc_local((8,), K.f32)
            for t in range(NUM_TOKENS):
                slot_t = _load_smem_i32(s_slot, t)
                beta_t = _load_smem_f32(s_beta, t)
                _load_smem_f32x4(s_d, t * HEAD_DIM + k_start, sd_t, 0)
                _load_smem_f32x4(s_d, t * HEAD_DIM + k_start + 4, sd_t, 4)
                _load_smem_f32x4(s_k, t * HEAD_DIM + k_start, sk_t, 0)
                _load_smem_f32x4(s_k, t * HEAD_DIM + k_start + 4, sk_t, 4)
                for row_local in range(8):
                    row_h = owned_row_base + row_local
                    update = _mul(_load_smem_f32(s_u, t * ROWS_PER_CTA + row_h), beta_t)
                    for i in range(8):
                        K.ptx.mov.b32(
                            hist[row_local * 8 + i],
                            _fma(hist[row_local * 8 + i], sd_t[i], _mul(update, sk_t[i])),
                        )
                    for pr in range(4):
                        K.ptx.mov.b32(
                            words_w[pr],
                            _pack_bf16x2(
                                hist[row_local * 8 + 2 * pr + 1], hist[row_local * 8 + 2 * pr]
                            ),
                        )
                    with K.If(slot_t >= 0), K.Then():
                        _store_u32x4(
                            state,
                            K.cast(slot_t, "int64") * K.cast(STATE_SLOT_STRIDE, "int64")
                            + K.cast(
                                hv[0] * HEAD_DIM * HEAD_DIM
                                + (tile_row_base[0] + row_h) * HEAD_DIM
                                + k_start,
                                "int64",
                            ),
                            words_w,
                        )

    return _flashkda_decode_t3_lower_bound


def get_kernel(**kwargs: Any):
    """Return the specialized FlashKDA cake T=3 decode PrimFunc."""
    return _make_flashkda_decode_t3_lower_bound(_specialization(kwargs)).func


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
