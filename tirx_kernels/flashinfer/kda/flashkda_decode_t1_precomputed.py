# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400),
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's FlashKDA "cake" T=1 precomputed-gate decode kernel.

Source: ``csrc/kda/flashkda_decode_d128_t1_precomputed_direct_split16.cu`` (and
its ``_split8`` sibling), symbol ``kernel_flashinfer_recurrent_kda_t1_direct``.
Both bodies are frozen machine-generated exports ("Generated from a recurrent-KDA
Loom schedule"); FlashInfer asserts their SHA256 in
``tests/jit/test_flash_kda_decode_jit.py``.

The two exports differ in exactly three lines -- ``value_splits``,
``tile_row_base``'s multiplier, and the ``row_block`` trip count -- so this module
carries a single ``VALUE_SPLIT`` constexpr and derives the rest.

Dispatch (``flashinfer/kda_kernels/recurrent_kda.py:1443-1447``) routes every
``num_tokens == 1`` precomputed call to ``d128_t1_precomputed_direct_split{16,8}``;
the 679-line non-direct ``t1_precomputed_split{1,2,4,8}`` exports are unreachable
through ``run_recurrent_kda`` and are out of this port's scope.

Shape of the kernel: one warp per CTA, grid ``(HV * VALUE_SPLIT, num_seqs)``, zero
shared memory, no barrier, no atomic, no workspace. The value split partitions the
128 value rows into disjoint slabs, so there is no cross-CTA combine of any kind.
"""

from __future__ import annotations

from typing import Any
from unittest import SkipTest

import torch

from tvm.script import tirx as T

HEAD_DIM = 128
L2_EPS = 1.0e-6
LOG2_E = 1.4426950408889634
# 16 lanes cover the 128-wide K dimension, 8 elements each; the remaining lane
# bit selects one of two interleaved value rows (direct_split16.cu:69-70).
K_LANES = 16
V_LANES = 2
ELEMS_PER_LANE = HEAD_DIM // 32  # 4: the pre-redistribution q/k/g slice
K_PER_LANE = HEAD_DIM // K_LANES  # 8: the post-redistribution slice
ROWS_PER_BLOCK = V_LANES * 4  # 8 value rows per row_block iteration

# TIRX_TRANSCRIBE_START flashkda_decode_t1_precomputed


# ---------------------------------------------------------------------------
# PTX helpers
# ---------------------------------------------------------------------------
# Every float op here is `.ftz`. The source compiles from plain CUDA operators
# under `-use_fast_math` (jit/core.py:545) and its PTX contains 108
# fma.rn.ftz.f32, 69 mul.ftz.f32, 53 add.ftz.f32, 4 sub.ftz.f32 and zero
# plain-.f32 arithmetic. That is the opposite of the CuTe-DSL KDA decode
# siblings, whose source emits no `.ftz` and whose helpers are deliberately
# non-FTZ -- importing theirs here would be a silent divergence.


def _ptx_un(chain: str, a, dtype: str = "float32"):
    out = T.alloc_local((1,), dtype)
    T.evaluate(T.ptx[chain](out[0], a))
    return out[0]


def _ptx_bin(chain: str, a, b, dtype: str = "float32"):
    out = T.alloc_local((1,), dtype)
    T.evaluate(T.ptx[chain](out[0], a, b))
    return out[0]


def _ptx_ter(chain: str, a, b, c, dtype: str = "float32"):
    out = T.alloc_local((1,), dtype)
    T.evaluate(T.ptx[chain](out[0], a, b, c))
    return out[0]


def _mul(a, b):
    """``mul.ftz.f32``."""
    return _ptx_bin("mul.ftz.f32", a, b)


def _add(a, b):
    """``add.ftz.f32``."""
    return _ptx_bin("add.ftz.f32", a, b)


def _sub(a, b):
    """``sub.ftz.f32``."""
    return _ptx_bin("sub.ftz.f32", a, b)


def _fma(a, b, c):
    """``fma.rn.ftz.f32``."""
    return _ptx_ter("fma.rn.ftz.f32", a, b, c)


def _rsqrt(a):
    """``rsqrt.approx.ftz.f32`` -- `rsqrtf` under fast math."""
    return _ptx_un("rsqrt.approx.ftz.f32", a)


def _expf(a):
    """``__expf``: one mul by log2(e), then ``ex2.approx.ftz.f32``.

    The mul is a separate emitted instruction, not an operand fold; the source's
    PTX shows `mul.ftz.f32 %r, %r, 0f3FB8AA3B` immediately before each ex2.
    """
    return _ptx_un("ex2.approx.ftz.f32", _mul(a, T.float32(LOG2_E)))


def _shfl_bfly(value, lane_xor):
    """``shfl.sync.bfly.b32``, clamp 31 and full member mask."""
    out = T.alloc_local((1,), "uint32")
    T.evaluate(
        T.ptx.shfl_sync.bfly.b32(
            out[0],
            T.reinterpret("uint32", value),
            T.uint32(lane_xor),
            T.uint32(31),
            T.uint32(0xFFFFFFFF),
        )
    )
    return T.reinterpret("float32", out[0])


def _shfl_idx(value, source_lane):
    """``shfl.sync.idx.b32``, clamp 31 and full member mask."""
    out = T.alloc_local((1,), "uint32")
    T.evaluate(
        T.ptx.shfl_sync.idx.b32(
            out[0],
            T.reinterpret("uint32", value),
            T.cast(source_lane, "uint32"),
            T.uint32(31),
            T.uint32(0xFFFFFFFF),
        )
    )
    return T.reinterpret("float32", out[0])


def _load_i32(buffer, index):
    """``ld.global.nc.b32`` -- the read-only metadata loads."""
    out = T.alloc_local((1,), "int32")
    T.evaluate(T.ptx.ld.global_.nc.b32(out[0], buffer.ptr_to([index])))
    return out[0]


def _load_bf16_f32(buffer, index):
    """``ld.global.nc.b16`` + ``cvt.f32.bf16`` -- the two scalar loads.

    The scalar path really does use cvt while the vector paths use the shl/and
    pair; both forms are in the source PTX and the asymmetry is deliberate.
    """
    bits = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.ld.global_.nc.b16(bits[0], buffer.ptr_to([index])))
    return _ptx_un("cvt.f32.bf16", bits[0])


# One 32-bit word carries two bf16. The source widens it with an integer pair
# (:96-102), not `cvt.f32.bf16`, and that survives to PTX, so reproducing the
# pair keeps the instruction mix aligned. Split in two because TVMScript has no
# tuple-unpacking assignment.


def _widen_lo(word):
    """``shl.b32 d, word, 16`` -- the low bf16 of the word."""
    return T.reinterpret("float32", T.shift_left(word, T.uint32(16)))


def _widen_hi(word):
    """``and.b32 d, word, 0xffff0000`` -- the high bf16 of the word."""
    return T.reinterpret("float32", T.bitwise_and(word, T.uint32(0xFFFF0000)))


def _load_u32x2(buffer, index):
    """``ld.global.nc.v2.b32`` -- one 8-byte tile (four bf16), left packed."""
    words = T.alloc_local((2,), "uint32")
    T.evaluate(T.ptx.ld.global_.nc.v2.b32(words[0], words[1], buffer.ptr_to([index])))
    return words


def _load_state_row(buffer, index):
    """``ld.global.L1::no_allocate.v4.b32`` -- one 16-byte state row slice.

    The eviction hint is the source's own (:193): the recurrent state streams
    through exactly once, so L1 is left to q/k/g/v.
    """
    words = T.alloc_local((4,), "uint32")
    T.evaluate(
        T.ptx["ld.global.L1::no_allocate.v4.b32"](
            words[0], words[1], words[2], words[3], buffer.ptr_to([index])
        )
    )
    return words


def _pack_bf16x2(hi, lo):
    """``cvt.rn.bf16x2.f32 d, hi, lo`` -- packs (elem[2p+1], elem[2p])."""
    return _ptx_bin("cvt.rn.bf16x2.f32", hi, lo, dtype="uint32")


def _store_u32x4(buffer, index, words):
    """``st.global.v4.b32`` -- one 16-byte state row slice."""
    T.evaluate(
        T.ptx.st.global_.v4.b32(buffer.ptr_to([index]), words[0], words[1], words[2], words[3])
    )


def _store_f32_as_bf16(buffer, index, value, pred):
    """``cvt.rn.bf16.f32`` + predicated ``st.global.b16``.

    The store is predicated at the PTX layer rather than wrapped in `if`: an
    if-wrapped asm block cannot be if-converted by ptxas and costs a real branch.
    """
    bits = _ptx_un("cvt.rn.bf16.f32", value, dtype="uint16")
    T.evaluate(T.ptx.st.global_.b16(buffer.ptr_to([index]), bits, pred=pred))


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    """One config row. Defaults mirror FlashInfer's cake decode export bench."""
    config: dict[str, Any] = {
        "label": label,
        "num_seqs": 8,
        "num_heads": 16,
        "num_value_heads": 16,
        "pool_size": 512,
        "padded_slots": 0,
        "slot_stride_pad": 0,
        "gate_token_stride_pad": 0,
        "scale": None,
        "seed": 20260812,
    }
    config.update(overrides)
    return config


# The split is chosen by `work = num_seqs * num_value_heads` against
# `2 * sm_count` (recurrent_kda.py:1170-1176); on a 148-SM B200 the boundary is
# 296. Every row below names the split its shape actually selects, and
# verify_dispatch.py asserts that against the real selector.
BENCH_CONFIGS = [
    # HV == H == 16, split16 side (work = 16, 64, 128, 256).
    _case("hv16h16_b1_s16", num_seqs=1),
    _case("hv16h16_b4_s16", num_seqs=4),
    _case("hv16h16_b8_s16", num_seqs=8),
    _case("hv16h16_b16_s16", num_seqs=16),
    # HV == H == 16, split8 side (work = 512, 1024, 2048).
    _case("hv16h16_b32_s8", num_seqs=32),
    _case("hv16h16_b64_s8", num_seqs=64),
    _case("hv16h16_b128_s8", num_seqs=128),
    # Kimi K3 TP8 per-rank head count, one row per split side (work = 96, 768).
    _case("hv12h12_b8_s16", num_seqs=8, num_heads=12, num_value_heads=12),
    _case("hv12h12_b64_s8", num_seqs=64, num_heads=12, num_value_heads=12),
    # GQA ratio 2, matching FlashInfer's own export bench (work = 256, 1024).
    _case("hv32h16_b8_s16", num_seqs=8, num_heads=16, num_value_heads=32),
    _case("hv32h16_b32_s8", num_seqs=32, num_heads=16, num_value_heads=32),
]

CONFIGS = [dict(cfg) for cfg in BENCH_CONFIGS] + [
    # Split knife edge on a 148-SM B200: work = 296 takes split16, 304 takes
    # split8 (tests/kda/test_recurrent_kda_decode_export.py:311-320).
    _case("hv16h16_b18_s16_edge", num_seqs=18),
    _case("hv16h16_b19_s8_edge", num_seqs=19),
    # CUDA-graph padding: negative ssm_state_indices rows must leave the state
    # untouched and zero their output (direct_split16.cu:76-78, :251-266).
    _case("hv16h16_b16_s16_padded", num_seqs=16, padded_slots=5),
    # Envelope-strided state pool: slot stride may exceed HV*V*K while staying
    # 8-element aligned (binding_common.cuh:148-161).
    _case("hv16h16_b16_s16_strided", num_seqs=16, slot_stride_pad=16),
    # Envelope-strided gate: g is the only input whose token stride is free
    # (CheckCompactGate, binding_common.cuh:137-146).
    _case("hv16h16_b16_s16_gstride", num_seqs=16, gate_token_stride_pad=256),
    # Non-default scale (the host passes `scale or 1/sqrt(128)`).
    _case("hv16h16_b8_s16_scale", num_seqs=8, scale=0.05),
    # GQA ratio 4 on the split8 side.
    _case("hv64h16_b16_s8", num_seqs=16, num_heads=16, num_value_heads=64),
]

KERNEL_META = {
    "name": "flashkda_decode_t1_precomputed",
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
    """Reproduce ``_select_flash_kda_decode_value_split_current`` for T = 1.

    recurrent_kda.py:1170-1176. `work` is `num_seqs * num_value_heads`; the T = 1
    branch is the whole policy on sm100a, and only 16 and 8 are reachable.
    """
    return 16 if work <= 2 * sm_count else 8


def _specialization(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Derive the constexpr set, mirroring the source host dispatch."""
    num_seqs = int(kwargs["num_seqs"])
    num_heads = int(kwargs["num_heads"])
    num_value_heads = int(kwargs["num_value_heads"])
    pool_size = int(kwargs["pool_size"])
    slot_stride_pad = int(kwargs.get("slot_stride_pad", 0))
    gate_token_stride_pad = int(kwargs.get("gate_token_stride_pad", 0))

    if num_value_heads % num_heads != 0 or num_value_heads < num_heads:
        raise ValueError("num_value_heads must be a multiple of num_heads and >= it")
    if not 0 < num_seqs <= 65535:
        raise ValueError("num_seqs must fit grid.y (binding_common.cuh:284-287)")
    if slot_stride_pad % 8 != 0:
        raise ValueError("state slot stride padding must stay 8-element aligned")
    if gate_token_stride_pad % 4 != 0:
        raise ValueError("gate token stride padding must stay 4-element aligned")

    value_split = _select_value_split(num_seqs * num_value_heads, _sm_count(kwargs))
    rows_per_cta = HEAD_DIM // value_split

    slot_stride = num_value_heads * HEAD_DIM * HEAD_DIM + slot_stride_pad
    gate_token_stride = num_value_heads * HEAD_DIM + gate_token_stride_pad
    return {
        "NUM_SEQS": num_seqs,
        "NUM_HEADS": num_heads,
        "NUM_VALUE_HEADS": num_value_heads,
        "HEAD_RATIO": num_value_heads // num_heads,
        "VALUE_SPLIT": value_split,
        # direct_split{16,8}.cu:79 -- tile_row_base = value_tile * rows_per_cta.
        "TILE_ROW_STRIDE": rows_per_cta,
        # direct_split{16,8}.cu:182 -- the `#pragma unroll 1` row_block trip count.
        "ROW_BLOCKS": rows_per_cta // ROWS_PER_BLOCK,
        "STATE_SLOT_STRIDE": slot_stride,
        "GATE_TOKEN_STRIDE": gate_token_stride,
        "Q_ELEMENTS": num_seqs * num_heads * HEAD_DIM,
        "V_ELEMENTS": num_seqs * num_value_heads * HEAD_DIM,
        "GATE_ELEMENTS": num_seqs * gate_token_stride,
        "BETA_ELEMENTS": num_seqs * num_value_heads,
        "STATE_ELEMENTS": pool_size * slot_stride,
        "CU_SEQLENS_ELEMENTS": num_seqs + 1,
        "STATE_INDEX_ELEMENTS": num_seqs,
    }


@T.jit
def _flashkda_decode_t1_precomputed(
    q_h: T.handle,
    k_h: T.handle,
    v_h: T.handle,
    g_h: T.handle,
    beta_h: T.handle,
    state_h: T.handle,
    out_h: T.handle,
    cu_seqlens_h: T.handle,
    ssm_state_indices_h: T.handle,
    scale: T.float32,
    *,
    NUM_SEQS: T.constexpr,
    NUM_HEADS: T.constexpr,
    NUM_VALUE_HEADS: T.constexpr,
    HEAD_RATIO: T.constexpr,
    VALUE_SPLIT: T.constexpr,
    TILE_ROW_STRIDE: T.constexpr,
    ROW_BLOCKS: T.constexpr,
    STATE_SLOT_STRIDE: T.constexpr,
    GATE_TOKEN_STRIDE: T.constexpr,
    Q_ELEMENTS: T.constexpr,
    V_ELEMENTS: T.constexpr,
    GATE_ELEMENTS: T.constexpr,
    BETA_ELEMENTS: T.constexpr,
    STATE_ELEMENTS: T.constexpr,
    CU_SEQLENS_ELEMENTS: T.constexpr,
    STATE_INDEX_ELEMENTS: T.constexpr,
):
    """FlashKDA "cake" T=1 precomputed-gate decode, one warp per CTA.

    Transcribed from `kernel_flashinfer_recurrent_kda_t1_direct`; line references
    are into `flashkda_decode_d128_t1_precomputed_direct_split16.cu`.
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

    T.device_entry()
    # --- work decomposition and lane roles (:63-71) ------------------------
    work, n = T.cta_id([NUM_VALUE_HEADS * VALUE_SPLIT, NUM_SEQS])
    lane = T.thread_id([32])
    value_tile: T.int32 = work % VALUE_SPLIT
    hv: T.int32 = work // VALUE_SPLIT
    query_head: T.int32 = hv // HEAD_RATIO  # the GQA fold
    k_lane: T.int32 = lane % K_LANES  # 16 lanes x 8 elements = 128 K
    v_lane: T.int32 = lane // K_LANES  # one of two interleaved value rows

    # --- row activity, resolved before any load (:72-79) -------------------
    raw_token_pos: T.int32 = _load_i32(cu, n)
    seq_len: T.int32 = _load_i32(cu, n + 1) - raw_token_pos
    # `raw_token_pos < gridDim.y` is why the body only accepts an identity
    # cu_seqlens; the host forbids a user-supplied T=1 cu_seqlens.
    has_token = T.And(T.And(raw_token_pos >= 0, raw_token_pos < NUM_SEQS), seq_len > 0)
    token_pos: T.int32 = T.if_then_else(has_token, raw_token_pos, 0)
    raw_slot: T.int32 = _load_i32(ssm_idx, n)
    initial_slot: T.int32 = T.max(raw_slot, 0)  # inactive rows clamp to slot 0
    active = T.And(raw_slot >= 0, has_token)
    tile_row_base: T.int32 = value_tile * TILE_ROW_STRIDE

    # --- phase 1: q, k, g vector loads, 4 elements per lane (:88-132) ------
    elem_start: T.int32 = lane * ELEMS_PER_LANE
    q_base: T.int32 = (token_pos * NUM_HEADS + query_head) * HEAD_DIM + elem_start
    gate_base: T.int32 = token_pos * GATE_TOKEN_STRIDE + hv * HEAD_DIM + elem_start

    q_src = T.alloc_local((ELEMS_PER_LANE,), "float32")
    k_src = T.alloc_local((ELEMS_PER_LANE,), "float32")
    gate_src = T.alloc_local((ELEMS_PER_LANE,), "float32")
    q_words = _load_u32x2(q, q_base)
    k_words = _load_u32x2(k, q_base)
    g_words = _load_u32x2(g, gate_base)
    for pair in T.unroll(2):
        q_src[2 * pair] = _widen_lo(q_words[pair])
        q_src[2 * pair + 1] = _widen_hi(q_words[pair])
        k_src[2 * pair] = _widen_lo(k_words[pair])
        k_src[2 * pair + 1] = _widen_hi(k_words[pair])
        gate_src[2 * pair] = _widen_lo(g_words[pair])
        gate_src[2 * pair + 1] = _widen_hi(g_words[pair])

    # --- phase 2: QK L2 norms, full 32-lane butterflies (:133-152) ---------
    # Association order copied from :136: a0*a0 + a1*a1 + (a2*a2 + a3*a3).
    q_sum_sq: T.float32 = _add(
        _fma(q_src[1], q_src[1], _mul(q_src[0], q_src[0])),
        _fma(q_src[3], q_src[3], _mul(q_src[2], q_src[2])),
    )
    k_sum_sq: T.float32 = _add(
        _fma(k_src[1], k_src[1], _mul(k_src[0], k_src[0])),
        _fma(k_src[3], k_src[3], _mul(k_src[2], k_src[2])),
    )
    # Two independent loops, run back to back (:139-143 then :144-148); the
    # source does not interleave them.
    for offset in T.unroll(5):
        q_sum_sq = _add(q_sum_sq, _shfl_bfly(q_sum_sq, 16 >> offset))
    for offset in T.unroll(5):
        k_sum_sq = _add(k_sum_sq, _shfl_bfly(k_sum_sq, 16 >> offset))
    # eps is hardcoded in the source (:149,:151); `scale` multiplies q only.
    q_scale: T.float32 = _mul(_rsqrt(_add(q_sum_sq, T.float32(L2_EPS))), scale)
    k_scale: T.float32 = _rsqrt(_add(k_sum_sq, T.float32(L2_EPS)))

    # --- phase 3: 4-per-lane -> 8-per-lane redistribution + gate (:153-164) -
    q_reg = T.alloc_local((K_PER_LANE,), "float32")
    k_reg = T.alloc_local((K_PER_LANE,), "float32")
    gate_reg = T.alloc_local((K_PER_LANE,), "float32")
    for i in T.unroll(K_PER_LANE):
        source_lane: T.int32 = 2 * k_lane + i // ELEMS_PER_LANE
        q_reg[i] = _mul(_shfl_idx(q_src[i % ELEMS_PER_LANE], source_lane), q_scale)
        k_reg[i] = _mul(_shfl_idx(k_src[i % ELEMS_PER_LANE], source_lane), k_scale)
        gate_reg[i] = _expf(_shfl_idx(gate_src[i % ELEMS_PER_LANE], source_lane))

    # --- phase 4: k_dot_q, 16-lane butterfly (:165-176) --------------------
    # Association order copied from :167: (k0q0 + k1q1 + (k2q2 + k3q3))
    #                                   + (k4q4 + k5q5 + (k6q6 + k7q7)).
    k_dot_q: T.float32 = _add(
        _add(
            _fma(k_reg[1], q_reg[1], _mul(k_reg[0], q_reg[0])),
            _fma(k_reg[3], q_reg[3], _mul(k_reg[2], q_reg[2])),
        ),
        _add(
            _fma(k_reg[5], q_reg[5], _mul(k_reg[4], q_reg[4])),
            _fma(k_reg[7], q_reg[7], _mul(k_reg[6], q_reg[6])),
        ),
    )
    # Offsets stop at 8, so each half-warp reduces within itself; the
    # instruction stays full-warp (clamp 31, mask 0xFFFFFFFF).
    for offset in T.unroll(4):
        k_dot_q = _add(k_dot_q, _shfl_bfly(k_dot_q, 8 >> offset))

    beta_value: T.float32 = _load_bf16_f32(beta, token_pos * NUM_VALUE_HEADS + hv)
    head_base = T.cast(initial_slot, "int64") * T.cast(STATE_SLOT_STRIDE, "int64") + T.cast(
        hv * HEAD_DIM * HEAD_DIM, "int64"
    )
    vg_base: T.int32 = (token_pos * NUM_VALUE_HEADS + hv) * HEAD_DIM

    # --- phase 5: the row loop (:181-268) ----------------------------------
    state_rows = T.alloc_local((4 * K_PER_LANE,), "float32")
    h_decay = T.alloc_local((K_PER_LANE,), "float32")
    words_w = T.alloc_local((4,), "uint32")

    for row_block in T.serial(ROW_BLOCKS, unroll=False):
        # Issue all four row loads before any math (:184-208). The decay pass
        # below depends on every row, so interleaving would serialise the misses.
        for row_local in T.unroll(4):
            row_a: T.int32 = tile_row_base + v_lane + 2 * (row_block * 4 + row_local)
            words = _load_state_row(
                state, head_base + T.cast(row_a * HEAD_DIM + k_lane * 8, "int64")
            )
            for pr in T.unroll(4):
                state_rows[row_local * K_PER_LANE + 2 * pr] = _widen_lo(words[pr])
                state_rows[row_local * K_PER_LANE + 2 * pr + 1] = _widen_hi(words[pr])

        for row_local in T.unroll(4):
            row: T.int32 = tile_row_base + v_lane + 2 * (row_block * 4 + row_local)

            # Decay and both projections share one pass; the source accumulates
            # sequentially (:217-223), not as a tree.
            pred: T.float32 = T.float32(0.0)
            base: T.float32 = T.float32(0.0)
            for i in T.unroll(K_PER_LANE):
                h_decay[i] = _mul(state_rows[row_local * K_PER_LANE + i], gate_reg[i])
                pred = _fma(h_decay[i], k_reg[i], pred)
                base = _fma(h_decay[i], q_reg[i], base)
            for offset in T.unroll(4):
                pred = _add(pred, _shfl_bfly(pred, 8 >> offset))
            for offset in T.unroll(4):
                base = _add(base, _shfl_bfly(base, 8 >> offset))

            # One scalar v load on k_lane == 0, then a broadcast (:240-245).
            v_value: T.float32 = T.float32(0.0)
            if k_lane == 0:
                v_value = _load_bf16_f32(v, vg_base + row)
            v_value = _shfl_idx(v_value, v_lane * K_LANES)

            delta: T.float32 = _mul(_sub(v_value, pred), beta_value)

            # Rank-1 update. The source writes `state*gate + delta*k` (:249) but
            # the decayed product is the h_decay above and the compiler keeps it:
            # .loc :249 emits 8 fma per row and zero mul.
            for i in T.unroll(K_PER_LANE):
                state_rows[row_local * K_PER_LANE + i] = _fma(delta, k_reg[i], h_decay[i])

            if active:
                for pr in T.unroll(4):
                    words_w[pr] = _pack_bf16x2(
                        state_rows[row_local * K_PER_LANE + 2 * pr + 1],
                        state_rows[row_local * K_PER_LANE + 2 * pr],
                    )
                _store_u32x4(
                    state, head_base + T.cast(row * HEAD_DIM + k_lane * 8, "int64"), words_w
                )
            # `o = q.S_new` is folded as base + delta*k_dot_q, so the updated
            # state is never re-read. The source contracts it to one fma (:262);
            # spelling it mul-then-add would round twice.
            _store_f32_as_bf16(
                out, vg_base + row, _fma(delta, k_dot_q, base), T.And(active, k_lane == 0)
            )
            # An in-row but inactive sequence zeroes its output (:264-266).
            _store_f32_as_bf16(
                out,
                vg_base + row,
                T.float32(0.0),
                T.And(T.And(has_token, T.Not(active)), k_lane == 0),
            )


def get_kernel(**kwargs: Any):
    """Return the specialized FlashKDA cake T=1 decode PrimFunc."""
    return _flashkda_decode_t1_precomputed.specialize(**_specialization(kwargs))


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Build one deterministic case with independent TIRx / reference state."""
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for FlashKDA cake T=1 decode")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(f"FlashKDA cake decode targets compute capability 10.x, got {capability}")

    spec = _specialization({**kwargs, "device": device})
    num_seqs = spec["NUM_SEQS"]
    num_heads = spec["NUM_HEADS"]
    num_value_heads = spec["NUM_VALUE_HEADS"]
    slot_stride = spec["STATE_SLOT_STRIDE"]
    gate_token_stride = spec["GATE_TOKEN_STRIDE"]
    pool_size = int(kwargs["pool_size"])
    padded_slots = int(kwargs.get("padded_slots", 0))

    gen = torch.Generator(device=device)
    gen.manual_seed(int(kwargs["seed"]))

    def randn(*shape: int, dtype=torch.bfloat16, gain: float = 1.0):
        raw = torch.randn(shape, device=device, dtype=torch.float32, generator=gen)
        return (gain * raw).to(dtype)

    q = randn(1, num_seqs, num_heads, HEAD_DIM, gain=0.5)
    k = randn(1, num_seqs, num_heads, HEAD_DIM, gain=0.5)
    v = randn(1, num_seqs, num_value_heads, HEAD_DIM, gain=0.5)
    # GATE_KIND == 0: `g` holds the per-K log-gate computed OUTSIDE the kernel,
    # which applies exp() to it. The upstream test builds it exactly this way
    # (tests/kda/test_recurrent_kda_decode_export.py:986-1002).
    gate_logits = torch.randn(
        (1, num_seqs, num_value_heads, HEAD_DIM), device=device, dtype=torch.float32, generator=gen
    )
    g_dense = torch.nn.functional.logsigmoid(gate_logits).to(torch.bfloat16)
    # `g` is the only input whose token stride is free; an envelope-strided case
    # exercises the g_stride_token argument.
    g_raw = torch.zeros((num_seqs * gate_token_stride,), device=device, dtype=torch.bfloat16)
    g = g_raw.as_strided(
        (1, num_seqs, num_value_heads, HEAD_DIM),
        (num_seqs * gate_token_stride, gate_token_stride, HEAD_DIM, 1),
    )
    g.copy_(g_dense)
    # beta arrives pre-sigmoided (kda_decode.py:97-98); the kernel reads it raw.
    beta = torch.sigmoid(randn(1, num_seqs, num_value_heads, dtype=torch.float32, gain=0.5)).to(
        torch.bfloat16
    )

    cu_seqlens = torch.arange(num_seqs + 1, device=device, dtype=torch.int32)
    slots = torch.arange(num_seqs, device=device, dtype=torch.int32)
    if padded_slots:
        slots[num_seqs - padded_slots :] = -1
    num_accepted_tokens = torch.ones(num_seqs, device=device, dtype=torch.int32)

    def make_state_pool() -> tuple[torch.Tensor, torch.Tensor]:
        raw = randn(pool_size * slot_stride, gain=0.01)
        view = raw.as_strided(
            (pool_size, num_value_heads, HEAD_DIM, HEAD_DIM),
            (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
        )
        return raw, view

    tirx_state_raw, tirx_state = make_state_pool()
    reference_state_raw = tirx_state_raw.clone()
    reference_state = reference_state_raw.as_strided(
        (pool_size, num_value_heads, HEAD_DIM, HEAD_DIM),
        (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
    )
    initial_state_raw = tirx_state_raw.clone()

    tirx_out = torch.empty(
        (1, num_seqs, num_value_heads, HEAD_DIM), device=device, dtype=torch.bfloat16
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
        "ssm_state_indices": slots,
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
        float(case["scale"]),
    )


def _flashinfer_reference(case: dict[str, Any]) -> torch.Tensor:
    """Run the frozen cake export itself on the reference state pool.

    Uses the JIT module's direct ABI rather than
    ``kda_decode.recurrent_kda(backend="cake")`` so the comparison is kernel-only
    and so inactive (negative-slot) rows -- which the public T=1 wrapper cannot
    express, since it synthesizes identity metadata -- are reachable.
    """
    from flashinfer.jit.flash_kda_decode import get_flash_kda_decode_module

    spec = case["spec"]
    value_split = spec["VALUE_SPLIT"]
    variant = f"d128_t1_precomputed_direct_split{value_split}"

    device = case["device"]
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == (10, 0):
        target = "sm100f" if torch.version.cuda and torch.version.cuda >= "12.9" else "sm100a"
    elif (major, minor) == (10, 3):
        target = "sm103a"
    else:
        raise SkipTest(f"FlashKDA cake decode has no export for compute capability {major}.{minor}")

    module = get_flash_kda_decode_module(variant, target)
    reference_out = torch.empty_like(case["tirx_out"])
    dummy_f32 = torch.ones(1, device=device, dtype=torch.float32)

    module.run(
        case["q"],
        case["k"],
        case["v"],
        case["g"],
        case["beta"],
        dummy_f32,
        dummy_f32,
        case["reference_state"],
        reference_out,
        case["cu_seqlens"],
        case["ssm_state_indices"],
        case["num_accepted_tokens"],
        float(case["scale"]),
        0.0,
        int(torch.cuda.current_stream(device).cuda_stream),
    )
    torch.cuda.synchronize(device)
    return reference_out


def _torch_reference(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent FP32 oracle of the source math (direct_split16.cu:133-267)."""
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
    slots = case["ssm_state_indices"]

    state_raw = case["initial_state_raw"].clone()
    state = state_raw.as_strided(
        (state_raw.numel() // slot_stride, num_value_heads, HEAD_DIM, HEAD_DIM),
        (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
    )
    out = torch.zeros(
        (num_seqs, num_value_heads, HEAD_DIM), device=case["device"], dtype=torch.float32
    )

    scale = float(case["scale"])
    for n in range(num_seqs):
        slot = int(slots[n].item())
        if slot < 0:
            continue
        for hv in range(num_value_heads):
            h = hv // head_ratio
            qn = q[n, h] * (torch.rsqrt(q[n, h].pow(2).sum() + L2_EPS) * scale)
            kn = k[n, h] * torch.rsqrt(k[n, h].pow(2).sum() + L2_EPS)
            gamma = torch.exp(g[n, hv])
            kq = torch.dot(kn, qn)

            s = state[slot, hv].float()
            decayed = s * gamma.unsqueeze(0)
            pred = decayed @ kn
            base = decayed @ qn
            delta = (v[n, hv] - pred) * beta[n, hv]
            state[slot, hv] = (decayed + delta.unsqueeze(1) * kn.unsqueeze(0)).to(torch.bfloat16)
            out[n, hv] = base + delta * kq

    return out.to(torch.bfloat16).unsqueeze(0), state_raw


# The port reproduces the source's instruction selection and association orders,
# so it should agree with the frozen export to within bf16 rounding, not merely
# to an algorithmic tolerance.
_RTOL = 2.0**-8
_ATOL = 1.0e-4
# The independent FP32 oracle accumulates in a different order, so it gets the
# looser band.
_ORACLE_RTOL = 2.0**-6
_ORACLE_ATOL = 3.0e-4


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
        msg=lambda m: f"output vs flashinfer cake export\n{m}",
    )
    torch.testing.assert_close(
        tirx_state.float(),
        case["reference_state_raw"].float(),
        rtol=_RTOL,
        atol=_ATOL,
        msg=lambda m: f"state vs flashinfer cake export\n{m}",
    )

    # 2. an independent FP32 oracle of the same math
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

    # 3. invariants the tolerance checks above cannot express
    num_seqs = spec["NUM_SEQS"]
    slot_stride = spec["STATE_SLOT_STRIDE"]
    payload = spec["NUM_VALUE_HEADS"] * HEAD_DIM * HEAD_DIM
    slots = case["ssm_state_indices"]
    initial = case["initial_state_raw"]

    inactive = [n for n in range(num_seqs) if int(slots[n]) < 0]
    for n in inactive:
        assert tirx_out[0, n].abs().max().item() == 0.0, (
            f"inactive row {n} must have a zeroed output"
        )
    touched = {int(slots[n]) for n in range(num_seqs) if int(slots[n]) >= 0}
    for slot in range(min(num_seqs + 4, initial.numel() // slot_stride)):
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
    # pool in place, so this also proves the two pools stayed independent.
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
