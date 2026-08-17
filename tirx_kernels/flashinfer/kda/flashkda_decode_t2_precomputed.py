# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400),
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's FlashKDA "cake" T=2 precomputed-gate decode kernel.

Source: ``csrc/kda/flashkda_decode_d128_t2_precomputed_split4.cu``, symbol
``kernel_flashinfer_recurrent_kda_wy_vtile_short`` -- a frozen machine-generated
export ("Generated from a recurrent-KDA Loom schedule") whose SHA256 it declares
in its own header.

This is the WY-transform schedule: 2 warps, a 14720-byte shared-memory arena,
``ldmatrix`` + ``mma.sync`` tensor-core products, and a per-token checkpoint
contract -- structurally unrelated to the one-warp ``t1_direct`` sibling, which
serves only as a contract reference.

Dispatch reaches this body whenever ``num_tokens == 2`` with
``num_spec_tokens == 1`` (MTP with one draft token). The sm100a split selector
returns 4 unconditionally at T=2 (``recurrent_kda.py:1170-1192``), so
``d128_t2_precomputed_split4`` is the only reachable -- and the only locally
measurable -- specialization; ``_specialization`` raises for anything else.
"""

from __future__ import annotations

from typing import Any
from unittest import SkipTest

import torch

from tvm.script import tirx as T

HEAD_DIM = 128
NUM_TOKENS = 2
VALUE_SPLIT = 4  # the sm100a selector returns 4 unconditionally at T = 2
L2_EPS = 1.0e-6
LOG2_E = 1.4426950408889634

THREADS = 64  # 2 warps (flash_kda_decode.py:_variant_metadata)
ROWS_PER_CTA = HEAD_DIM // VALUE_SPLIT  # 32 value rows per CTA
ROWS_PER_THREAD = 8  # each of the 64 threads owns 8 rows x 8 keys
K_PER_THREAD = 8

# Shared-memory arena, byte offsets copied from the source's #define block
# (.cu:45-87). The source declares it `extern __shared__ __align__(1024)` and
# the launcher passes SMEM_TOTAL dynamically; the port declares the same bytes
# with the same alignment as a static arena, which needs no launch tag.
OFF_SSTATE0 = 0
OFF_SSTATE1 = 4096
OFF_SVEC = 8192
OFF_SK = 12288
OFF_SD = 13312
OFF_SBETA = 14336
OFF_SSLOT = 14344
OFF_STOKEN = 14352
OFF_SINIT = 14360
OFF_SL = 14376
OFF_SR = 14392
OFF_SU = 14408
SMEM_TOTAL = 14720
# sGramA0/sGramA1 (offsets 8192/10240) alias sVec and are dead at T=2/split4 --
# they belong to the t5/t6 coefficient-gram schedules. Not declared here.

# TIRX_TRANSCRIBE_START flashkda_decode_t2_precomputed


# ---------------------------------------------------------------------------
# PTX helpers
# ---------------------------------------------------------------------------
# Every float op is `.ftz`: the source compiles from plain CUDA operators under
# -use_fast_math and its PTX contains no plain-.f32 arithmetic at all. Both
# families are registered in the PTX table, so this is an explicit choice --
# importing the CuTe-DSL KDA ports' deliberately non-FTZ helpers would be a
# silent divergence.

_MMA_ZERO_C = (T.float32(0.0), T.float32(0.0), T.float32(0.0), T.float32(0.0))


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
    """``__expf``: one mul by log2(e), then ``ex2.approx.ftz.f32``."""
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
    """``ld.global.nc.b16`` + ``cvt.f32.bf16`` -- the scalar v and beta loads."""
    bits = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.ld.global_.nc.b16(bits[0], buffer.ptr_to([index])))
    return _ptx_un("cvt.f32.bf16", bits[0])


def _widen_lo(word):
    """``shl.b32 d, word, 16`` -- the low bf16 of a packed word."""
    return T.reinterpret("float32", T.shift_left(word, T.uint32(16)))


def _widen_hi(word):
    """``and.b32 d, word, 0xffff0000`` -- the high bf16."""
    return T.reinterpret("float32", T.bitwise_and(word, T.uint32(0xFFFF0000)))


def _load_u32x2(buffer, index):
    """``ld.global.nc.v2.b32`` -- one 8-byte tile (four bf16), left packed."""
    words = T.alloc_local((2,), "uint32")
    T.evaluate(T.ptx.ld.global_.nc.v2.b32(words[0], words[1], buffer.ptr_to([index])))
    return words


def _load_u32x4(buffer, index):
    """``ld.global.v4.b32`` -- one 16-byte state tile.

    Not ``.nc``: the same kernel writes `state`.
    """
    words = T.alloc_local((4,), "uint32")
    T.evaluate(
        T.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
    )
    return words


def _store_u32x4(buffer, index, words):
    """``st.global.v4.b32`` -- one 16-byte state tile."""
    T.evaluate(
        T.ptx.st.global_.v4.b32(buffer.ptr_to([index]), words[0], words[1], words[2], words[3])
    )


def _pack_bf16x2(hi, lo):
    """``cvt.rn.bf16x2.f32 d, hi, lo``."""
    return _ptx_bin("cvt.rn.bf16x2.f32", hi, lo, dtype="uint32")


def _store_f32_as_bf16(buffer, index, value, pred):
    """``cvt.rn.bf16.f32`` + predicated ``st.global.b16``."""
    bits = _ptx_un("cvt.rn.bf16.f32", value, dtype="uint16")
    T.evaluate(T.ptx.st.global_.b16(buffer.ptr_to([index]), bits, pred=pred))


# --- shared memory --------------------------------------------------------
# The swizzle the source applies to every sState/sVec byte offset (:322 etc).


def _swz(byte_off):
    return T.bitwise_xor(byte_off, T.shift_left(T.bitwise_and(T.shift_right(byte_off, 7), 7), 4))


def _st_shared_f32(arena, byte_off, value):
    """``st.shared.b32``."""
    T.evaluate(T.ptx.st.shared.b32(arena.ptr_to([byte_off]), T.reinterpret("uint32", value)))


def _ld_shared_f32(arena, byte_off):
    """``ld.shared.b32``."""
    out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.ld.shared.b32(out[0], arena.ptr_to([byte_off])))
    return T.reinterpret("float32", out[0])


def _st_shared_i32(arena, byte_off, value):
    """``st.shared.b32`` for an int32 payload."""
    T.evaluate(T.ptx.st.shared.b32(arena.ptr_to([byte_off]), T.reinterpret("uint32", value)))


def _ld_shared_i32(arena, byte_off):
    """``ld.shared.b32`` for an int32 payload."""
    out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.ld.shared.b32(out[0], arena.ptr_to([byte_off])))
    return T.reinterpret("int32", out[0])


def _ld_shared_f32x4(arena, byte_off, dst, base):
    """``ld.shared.v4.b32`` -- four contiguous shared floats."""
    words = T.alloc_local((4,), "uint32")
    T.evaluate(
        T.ptx.ld.shared.v4.b32(words[0], words[1], words[2], words[3], arena.ptr_to([byte_off]))
    )
    for i in range(4):
        T.buffer_store(dst, T.reinterpret("float32", words[i]), [base + i])


def _st_shared_u32x4(arena, byte_off, words):
    """``st.shared.v4.b32`` -- the sState stage and the sK/sD publish."""
    T.evaluate(
        T.ptx.st.shared.v4.b32(arena.ptr_to([byte_off]), words[0], words[1], words[2], words[3])
    )


def _st_shared_b16(arena, byte_off, bits):
    """``st.shared.b16`` -- one sVec element."""
    T.evaluate(T.ptx.st.shared.b16(arena.ptr_to([byte_off]), bits))


def _ldmatrix_x4(arena, byte_off, frag, trans: bool):
    """``ldmatrix.sync.aligned.m8n8.x4[.trans].shared.b16`` -- one x4 group."""
    if trans:
        T.evaluate(
            T.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                frag[0], frag[1], frag[2], frag[3], arena.ptr_to([byte_off])
            )
        )
    else:
        T.evaluate(
            T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                frag[0], frag[1], frag[2], frag[3], arena.ptr_to([byte_off])
            )
        )


def _mma_zero(acc, a, b):
    """``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`` with an explicit zero C."""
    T.evaluate(
        T.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
            acc[0], acc[1], acc[2], acc[3],
            a[0], a[1], a[2], a[3],
            b[0], b[1],
            *_MMA_ZERO_C,
        )
    )  # fmt: skip


def _mma_acc(acc, a, b):
    """Same, accumulating: C aliases D, matching the source's `+f` tied registers."""
    T.evaluate(
        T.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
            acc[0], acc[1], acc[2], acc[3],
            a[0], a[1], a[2], a[3],
            b[0], b[1],
            acc[0], acc[1], acc[2], acc[3],
        )
    )  # fmt: skip


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    """One config row. Defaults mirror FlashInfer's T=2 export bench."""
    config: dict[str, Any] = {
        "label": label,
        "num_seqs": 8,
        "num_heads": 16,
        "num_value_heads": 32,
        "pool_slack": 6,  # state_slots = num_seqs*T + slack (upstream bench)
        "padded_seqs": 0,
        "slot_stride_pad": 0,
        "gate_token_stride_pad": 0,
        "accepted": None,  # None -> all ones (initial slot = ssm_idx[n, 0])
        "scale": None,
        "seed": 20260813,
    }
    config.update(overrides)
    return config


# Every T=2 shape selects split4, so the matrix varies only (HV, H, N).
# hv32h16 mirrors FlashInfer's own export bench exactly.
BENCH_CONFIGS = [
    _case("hv32h16_b8_t2", num_seqs=8),
    _case("hv32h16_b16_t2", num_seqs=16),
    _case("hv32h16_b32_t2", num_seqs=32),
    _case("hv32h16_b64_t2", num_seqs=64),
    _case("hv32h16_b128_t2", num_seqs=128),
    _case("hv16h16_b8_t2", num_seqs=8, num_heads=16, num_value_heads=16),
    _case("hv16h16_b64_t2", num_seqs=64, num_heads=16, num_value_heads=16),
    _case("hv12h12_b8_t2", num_seqs=8, num_heads=12, num_value_heads=12),
    _case("hv12h12_b64_t2", num_seqs=64, num_heads=12, num_value_heads=12),
]

CONFIGS = [dict(cfg) for cfg in BENCH_CONFIGS] + [
    # num_accepted_tokens is live at T=2 (unlike t1_direct): it selects the
    # initial checkpoint slot as ssm_idx[n*2 + clamp(nat-1, 0, 1)]. The upstream
    # test sweeps 0/1/2/9, covering both clamp edges (.cu:279-286).
    _case("hv32h16_b8_t2_nat0", num_seqs=8, accepted="zeros"),
    _case("hv32h16_b8_t2_nat2", num_seqs=8, accepted="twos"),
    _case("hv32h16_b8_t2_nat9", num_seqs=8, accepted="nines"),
    _case("hv32h16_b8_t2_natmix", num_seqs=8, accepted="mixed"),
    # CUDA-graph padding: ssm_state_indices == -1 rows must zero their output
    # bit-exactly and leave their state slots untouched (.cu:501-530).
    _case("hv32h16_b8_t2_padded", num_seqs=8, padded_seqs=2),
    # Envelope-strided state pool and gate.
    _case("hv32h16_b8_t2_strided", num_seqs=8, slot_stride_pad=8),
    _case("hv32h16_b8_t2_gstride", num_seqs=8, gate_token_stride_pad=8),
    # Non-default scale; GQA ratio 4; the smallest grid.
    _case("hv32h16_b8_t2_scale", num_seqs=8, scale=0.05),
    _case("hv64h16_b8_t2", num_seqs=8, num_heads=16, num_value_heads=64),
    _case("hv32h16_b1_t2", num_seqs=1),
]

KERNEL_META = {
    "name": "flashkda_decode_t2_precomputed",
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

    # The sm100a policy is `if num_tokens == 2: return 4` with no shape
    # dependence, so there is nothing to select -- but assert it, because a
    # different split would be a different kernel.
    if VALUE_SPLIT != 4:
        raise ValueError("only d128_t2_precomputed_split4 is in this port's scope")

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
        "STATE_INDEX_ELEMENTS": num_seqs * NUM_TOKENS,
        "NAT_ELEMENTS": num_seqs,
    }


@T.jit
def _flashkda_decode_t2_precomputed(
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
    """FlashKDA "cake" T=2 precomputed-gate decode, WY schedule, 2 warps.

    Transcribed from `kernel_flashinfer_recurrent_kda_wy_vtile_short`; bare
    `:NNN` references are into
    `flashkda_decode_d128_t2_precomputed_split4.cu`.
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
    # The source body declares __launch_bounds__(256) -- a vestigial Loom
    # constant, since the binding launches 64 threads -- together with
    # --maxrregcount=128. TIRx would otherwise emit __launch_bounds__(64), which
    # lets ptxas spend 118 registers per thread against the source's 105; at 64
    # threads that is 8 blocks/SM instead of 9. Asking for 9 brings it to 96.
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 9})

    # The source uses dynamic smem; 14720 B fits the static limit, so the same
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
                # nat is live at T=2: it picks the initial checkpoint slot,
                # clamped at both ends (:279-287).
                accepted: T.int32 = T.min(T.max(_load_i32(nat, n) - 1, 0), NUM_TOKENS - 1)
                initial_slot: T.int32 = _load_i32(ssm_idx, n * NUM_TOKENS + accepted)
                _st_shared_i32(arena, OFF_SINIT, T.max(initial_slot, 0))

    T.cuda.cta_sync()

    # =======================================================================
    # Phase B: state gather and sState stage, all 64 threads  (:292-329)
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
    acc = T.alloc_local((4,), "float32", align=4)
    if warp < NUM_TOKENS:
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
    # Phase E/F: quad broadcast, WY solve, and both tokens' outputs (:439-531)
    # =======================================================================
    u_lo = T.alloc_local((NUM_TOKENS,), "float32")
    u_hi = T.alloc_local((NUM_TOKENS,), "float32")
    if warp < NUM_TOKENS:
        ha_lo = T.alloc_local((NUM_TOKENS,), "float32")
        ha_hi = T.alloc_local((NUM_TOKENS,), "float32")
        for t in T.unroll(NUM_TOKENS):
            ha_lo[t] = _shfl_idx(acc[t], quad_base)
            ha_hi[t] = _shfl_idx(acc[2 + t], quad_base)

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

            # ---- outputs, both tokens (:484-531) --------------------------
            for t in T.unroll(NUM_TOKENS):
                out_lo: T.float32 = acc[t]
                out_hi: T.float32 = acc[2 + t]
                for src in T.unroll(NUM_TOKENS):
                    # The s > t coefficient is a real zero-operand fma, not a
                    # skipped iteration (:491-493, :515-517).
                    coef: T.float32 = T.float32(0.0)
                    if src <= t:
                        coef = _ld_shared_f32(arena, OFF_SR + (t * NUM_TOKENS + src) * 4)
                    out_lo = _fma(coef, u_lo[src], out_lo)
                    out_hi = _fma(coef, u_hi[src], out_hi)
                active_t = _ld_shared_i32(arena, OFF_SSLOT + t * 4) >= 0
                base_o: T.int32 = (
                    _ld_shared_i32(arena, OFF_STOKEN + t * 4) * NUM_VALUE_HEADS + hv
                ) * HEAD_DIM + tile_row_base
                _store_f32_as_bf16(out, base_o + row_lo, out_lo, active_t)
                _store_f32_as_bf16(out, base_o + row_hi, out_hi, active_t)
                _store_f32_as_bf16(out, base_o + row_lo, T.float32(0.0), T.Not(active_t))
                _store_f32_as_bf16(out, base_o + row_hi, T.float32(0.0), T.Not(active_t))

            # ---- publish sU (:532-541) ------------------------------------
            for t in T.unroll(NUM_TOKENS):
                _st_shared_f32(arena, OFF_SU + (t * 32 + row_lo) * 4, u_lo[t])
                _st_shared_f32(arena, OFF_SU + (t * 32 + row_hi) * 4, u_hi[t])
        # A warp barrier suffices: the sU rows this warp writes are exactly the
        # rows its own threads read back below (:543).
        T.cuda.warp_sync()

    # =======================================================================
    # Phase H: recurrence and checkpoints, all 64 threads  (:659-695)
    # =======================================================================
    words_w = T.alloc_local((4,), "uint32")
    sd_t = T.alloc_local((8,), "float32")
    sk_t = T.alloc_local((8,), "float32")
    for t in T.unroll(NUM_TOKENS):
        slot_t: T.int32 = _ld_shared_i32(arena, OFF_SSLOT + t * 4)
        beta_t: T.float32 = _ld_shared_f32(arena, OFF_SBETA + t * 4)
        # The gate and key slices depend only on (t, k_start), not on the row,
        # so the source loads each 8-float slice once per token as two 16-byte
        # reads instead of reloading them in the row loop (:673 is 12
        # ld.shared.v4.b32 with no scalar shared loads at all).
        _ld_shared_f32x4(arena, OFF_SD + (t * HEAD_DIM + k_start) * 4, sd_t, 0)
        _ld_shared_f32x4(arena, OFF_SD + (t * HEAD_DIM + k_start + 4) * 4, sd_t, 4)
        _ld_shared_f32x4(arena, OFF_SK + (t * HEAD_DIM + k_start) * 4, sk_t, 0)
        _ld_shared_f32x4(arena, OFF_SK + (t * HEAD_DIM + k_start + 4) * 4, sk_t, 4)
        for row_local in T.unroll(8):
            row_h: T.int32 = owned_row_base + row_local
            update: T.float32 = _mul(_ld_shared_f32(arena, OFF_SU + (t * 32 + row_h) * 4), beta_t)
            for i in T.unroll(8):
                # The source writes `hist*sD + update*sK` (:673) and the
                # compiler contracts the FIRST product: update*sK is rounded,
                # hist*sD is fused. This is the only stateful accumulation and
                # it feeds both the checkpoint and token 1's history.
                hist[row_local * 8 + i] = _fma(
                    hist[row_local * 8 + i], sd_t[i], _mul(update, sk_t[i])
                )
            for pr in T.unroll(4):
                words_w[pr] = _pack_bf16x2(
                    hist[row_local * 8 + 2 * pr + 1], hist[row_local * 8 + 2 * pr]
                )
            # The recurrence advances unconditionally; only the store is
            # slot-predicated.
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
    """Return the specialized FlashKDA cake T=2 decode PrimFunc."""
    return _flashkda_decode_t2_precomputed.specialize(**_specialization(kwargs))


def _accepted_tensor(kind: Any, num_seqs: int, device: str) -> torch.Tensor:
    """num_accepted_tokens per case.

    The kernel clamps `nat - 1` into [0, 1] and uses it to pick the initial
    checkpoint slot, so 0 and 1 both select index 0 while 2 and anything larger
    select index 1 (.cu:279-286).
    """
    if kind is None:
        return torch.ones(num_seqs, device=device, dtype=torch.int32)
    if kind == "zeros":
        return torch.zeros(num_seqs, device=device, dtype=torch.int32)
    if kind == "twos":
        return torch.full((num_seqs,), 2, device=device, dtype=torch.int32)
    if kind == "nines":
        return torch.full((num_seqs,), 9, device=device, dtype=torch.int32)
    if kind == "mixed":
        # The upstream test's sweep, tiled to the batch size
        # (test_recurrent_kda_decode_export.py:1269-1282).
        pattern = torch.tensor([0, 1, 2, 9, 1, 0, 1, 2], dtype=torch.int32)
        reps = (num_seqs + pattern.numel() - 1) // pattern.numel()
        return pattern.repeat(reps)[:num_seqs].to(device)
    raise ValueError(f"unknown accepted-token pattern {kind!r}")


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Build one deterministic case with independent TIRx / reference state."""
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for FlashKDA cake T=2 decode")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(f"FlashKDA cake decode targets compute capability 10.x, got {capability}")

    spec = _specialization(kwargs)
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

    # Packed [1, N*2, ...] layout -- the form the upstream tests and benches use
    # (recurrent_kda.py:1770-1795).
    q = randn(1, total_tokens, num_heads, HEAD_DIM, gain=0.5)
    k = randn(1, total_tokens, num_heads, HEAD_DIM, gain=0.5)
    v = randn(1, total_tokens, num_value_heads, HEAD_DIM, gain=0.5)
    # GATE_KIND == 0: g holds the per-K log-gate computed on the host; the
    # kernel applies exp() to it.
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

    # cu_seqlens steps by exactly NUM_TOKENS per row; the body assumes this
    # stride-T contract and does not bound-check token_base.
    cu_seqlens = torch.arange(0, total_tokens + 1, NUM_TOKENS, device=device, dtype=torch.int32)
    # Slot 0 is deliberately never a checkpoint target, matching the upstream
    # bench (bench_recurrent_kda_decode_export.py:232-236).
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


def _torch_reference(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent FP32 oracle of the T=2 delta rule.

    Written as the sequential recurrence the WY transform computes in blocked
    form, which is the point: it agrees only if the WY algebra is right. It
    keeps the kernel's two structural choices -- the recurrent state advances in
    FP32 across both tokens (only the stored checkpoint is rounded to bf16), and
    the initial slot is selected by the clamped num_accepted_tokens.
    """
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
            # FP32 carry across both tokens; only the checkpoint store rounds.
            s = state[init_slot, hv].float()
            for t in range(NUM_TOKENS):
                row = n * NUM_TOKENS + t
                qn = q[row, h] * (torch.rsqrt(q[row, h].pow(2).sum() + L2_EPS) * scale)
                kn = k[row, h] * torch.rsqrt(k[row, h].pow(2).sum() + L2_EPS)
                gamma = torch.exp(g[row, hv])

                decayed = s * gamma.unsqueeze(0)
                pred = decayed @ kn
                u = v[row, hv] - pred
                delta = u * beta[row, hv]
                s = decayed + delta.unsqueeze(1) * kn.unsqueeze(0)
                out[row, hv] = s @ qn

                slot = int(slots2d[n, t].item())
                if slot >= 0:
                    state[slot, hv] = s.to(torch.bfloat16)
                else:
                    out[row, hv] = 0.0

    return out.to(torch.bfloat16).unsqueeze(0), state_raw


# The FP32 oracle is the sequential delta rule; the kernel rounds its MMA
# operands to bf16, so the band here is genuinely wider and measured, not
# guessed: at scaffold time the export itself sat 6.1e-05 from the oracle.
_ORACLE_RTOL = 2.0**-5
_ORACLE_ATOL = 6.0e-3


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(**kwargs: Any) -> None:
    """Validate one correctness config against the in-tree Torch oracle."""
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    spec = case["spec"]

    executable = compile_kernel(get_kernel(**kwargs))
    executable(*_tirx_args(case))
    torch.cuda.synchronize(case["device"])

    tirx_out = case["tirx_out"]
    tirx_state = case["tirx_state_raw"].clone()

    # Independent FP32 oracle
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
        from tirx_kernels.flashinfer.utils._flashkda_bench import (
            prepare_flashinfer_cake_decode_reference,
        )

        return prepare_flashinfer_cake_decode_reference(case, "d128_t2_precomputed_split4")

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
