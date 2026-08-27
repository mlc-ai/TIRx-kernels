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

from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K

HEAD_DIM = 128
NUM_TOKENS = 2
VALUE_SPLIT = 4  # the sm100a selector returns 4 unconditionally at T = 2
L2_EPS = 1.0e-6
LOG2_E = 1.4426950408889634

THREADS = 64  # 2 warps (flash_kda_decode.py:_variant_metadata)
ROWS_PER_CTA = HEAD_DIM // VALUE_SPLIT  # 32 value rows per CTA
ROWS_PER_THREAD = 8  # each of the 64 threads owns 8 rows x 8 keys
K_PER_THREAD = 8

# The source declares a 1024-byte-aligned 14720-byte dynamic arena.  K's
# typed allocation order below is the single authority for every member offset;
# the explicit total retains the source's final alignment tail.
SMEM_TOTAL = 14720
# sGramA0/sGramA1 alias the sVec region and are dead at T=2/split4.

# TIRX_TRANSCRIBE_START flashkda_decode_t2_precomputed


# ---------------------------------------------------------------------------
# PTX helpers
# ---------------------------------------------------------------------------
# Every float op is `.ftz`: the source compiles from plain CUDA operators under
# -use_fast_math and its PTX contains no plain-.f32 arithmetic at all. Both
# families are registered in the PTX table, so this is an explicit choice --
# importing the CuTe-DSL KDA ports' deliberately non-FTZ helpers would be a
# silent divergence.

_MMA_ZERO_C = (K.float32(0.0), K.float32(0.0), K.float32(0.0), K.float32(0.0))


def _ptx_un(chain: str, a, dtype: str = "float32"):
    out = K.local_scalar(dtype)
    K.ptx[chain](out, a)
    return out


def _ptx_bin(chain: str, a, b, dtype: str = "float32"):
    out = K.local_scalar(dtype)
    K.ptx[chain](out, a, b)
    return out


def _ptx_ter(chain: str, a, b, c, dtype: str = "float32"):
    out = K.local_scalar(dtype)
    K.ptx[chain](out, a, b, c)
    return out


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
    return _ptx_un("ex2.approx.ftz.f32", _mul(a, K.float32(LOG2_E)))


def _shfl_bfly(value, lane_xor):
    """``shfl.sync.bfly.b32``, clamp 31 and full member mask."""
    out = K.local_scalar("uint32")
    K.ptx.shfl_sync.bfly.b32(
        out, K.reinterpret("uint32", value), K.uint32(lane_xor), K.uint32(31), K.uint32(0xFFFFFFFF)
    )
    return K.reinterpret("float32", out)


def _swz(byte_off):
    """Exact SW128B byte transform shared by the T=5/T=6 gram kernels."""
    return K.bitwise_xor(byte_off, K.shift_left(K.bitwise_and(K.shift_right(byte_off, 7), 7), 4))


def _st_shared_f32(region, byte_off, value):
    """Store one FP32 payload through a region-local byte address."""
    K.ptx.st.shared.b32(region.ptr_to([byte_off]), K.reinterpret("uint32", value))


def _ld_shared_f32(region, byte_off):
    """Load one FP32 payload through a region-local byte address."""
    out = K.local_scalar("uint32")
    K.ptx.ld.shared.b32(out, region.ptr_to([byte_off]))
    return K.reinterpret("float32", out)


def _st_shared_i32(region, byte_off, value):
    """Store one int32 payload through a region-local byte address."""
    K.ptx.st.shared.b32(region.ptr_to([byte_off]), K.reinterpret("uint32", value))


def _ld_shared_i32(region, byte_off):
    """Load one int32 payload through a region-local byte address."""
    out = K.local_scalar("uint32")
    K.ptx.ld.shared.b32(out, region.ptr_to([byte_off]))
    return K.reinterpret("int32", out)


def _ld_shared_f32x4(region, byte_off, dst, base):
    """Load four contiguous FP32 values through a region-local byte address."""
    words = K.alloc_local((4,), "uint32")
    K.ptx.ld.shared.v4.b32(words[0], words[1], words[2], words[3], region.ptr_to([byte_off]))
    for index in range(4):
        K.buffer_store(dst, K.reinterpret("float32", words[index]), [base + index])


def _st_shared_u32x4(region, byte_off, words):
    """Store four contiguous words through a region-local byte address."""
    K.ptx.st.shared.v4.b32(region.ptr_to([byte_off]), words[0], words[1], words[2], words[3])


def _st_shared_b16(region, byte_off, bits):
    """Store one 16-bit payload through a region-local byte address."""
    K.ptx.st.shared.b16(region.ptr_to([byte_off]), bits)


def _ldmatrix_x4(region, byte_off, frag, trans: bool):
    """Load one x4 matrix fragment from a region-local byte address."""
    if trans:
        K.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            frag[0], frag[1], frag[2], frag[3], region.ptr_to([byte_off])
        )
    else:
        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
            frag[0], frag[1], frag[2], frag[3], region.ptr_to([byte_off])
        )


def _shfl_idx(value, source_lane):
    """``shfl.sync.idx.b32``, clamp 31 and full member mask."""
    out = K.local_scalar("uint32")
    K.ptx.shfl_sync.idx.b32(
        out,
        K.reinterpret("uint32", value),
        K.cast(source_lane, "uint32"),
        K.uint32(31),
        K.uint32(0xFFFFFFFF),
    )
    return K.reinterpret("float32", out)


def _load_i32(buffer, index):
    """``ld.global.nc.b32`` -- the read-only metadata loads."""
    out = K.local_scalar("int32")
    K.ptx.ld.global_.nc.b32(out, buffer.ptr_to([index]))
    return out


def _load_bf16_f32(buffer, index):
    """``ld.global.nc.b16`` + ``cvt.f32.bf16`` -- the scalar v and beta loads."""
    bits = K.local_scalar("uint16")
    K.ptx.ld.global_.nc.b16(bits, buffer.ptr_to([index]))
    return _ptx_un("cvt.f32.bf16", bits)


def _widen_lo(word):
    """``shl.b32 d, word, 16`` -- the low bf16 of a packed word."""
    return K.reinterpret("float32", K.shift_left(word, K.uint32(16)))


def _widen_hi(word):
    """``and.b32 d, word, 0xffff0000`` -- the high bf16."""
    return K.reinterpret("float32", K.bitwise_and(word, K.uint32(0xFFFF0000)))


def _load_u32x2(buffer, index):
    """``ld.global.nc.v2.b32`` -- one 8-byte tile (four bf16), left packed."""
    words = K.alloc_local((2,), "uint32")
    K.ptx.ld.global_.nc.v2.b32(words[0], words[1], buffer.ptr_to([index]))
    return words


def _load_u32x4(buffer, index):
    """``ld.global.v4.b32`` -- one 16-byte state tile.

    Not ``.nc``: the same kernel writes `state`.
    """
    words = K.alloc_local((4,), "uint32")
    K.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
    return words


def _store_u32x4(buffer, index, words):
    """``st.global.v4.b32`` -- one 16-byte state tile."""
    K.ptx.st.global_.v4.b32(buffer.ptr_to([index]), words[0], words[1], words[2], words[3])


def _pack_bf16x2(hi, lo):
    """``cvt.rn.bf16x2.f32 d, hi, lo``."""
    return _ptx_bin("cvt.rn.bf16x2.f32", hi, lo, dtype="uint32")


def _store_f32_as_bf16(buffer, index, value, pred):
    """``cvt.rn.bf16.f32`` + predicated ``st.global.b16``."""
    bits = _ptx_un("cvt.rn.bf16.f32", value, dtype="uint16")
    K.ptx.st.global_.b16(buffer.ptr_to([index]), bits, pred=pred)


def _mma_zero(acc, a, b):
    """``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`` with an explicit zero C."""
    K.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
            acc[0], acc[1], acc[2], acc[3],
            a[0], a[1], a[2], a[3],
            b[0], b[1],
            *_MMA_ZERO_C,
        )  # fmt: skip


def _mma_acc(acc, a, b):
    """Same, accumulating: C aliases D, matching the source's `+f` tied registers."""
    K.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
            acc[0], acc[1], acc[2], acc[3],
            a[0], a[1], a[2], a[3],
            b[0], b[1],
            acc[0], acc[1], acc[2], acc[3],
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


def _make_flashkda_decode_t2_precomputed(spec: dict[str, Any]):
    NUM_SEQS = spec["NUM_SEQS"]
    NUM_HEADS = spec["NUM_HEADS"]
    NUM_VALUE_HEADS = spec["NUM_VALUE_HEADS"]
    HEAD_RATIO = spec["HEAD_RATIO"]
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

    @K.kernel(
        warps=THREADS // 32,
        arch="sm_100a",
        min_blocks_per_sm=9,
        grid=(NUM_VALUE_HEADS * VALUE_SPLIT, NUM_SEQS),
    )
    def _flashkda_decode_t2_precomputed(
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
        work, n = K.cta_id()
        tid = K.thread_id()
        # The source body declares __launch_bounds__(256) -- a vestigial Loom
        # constant, since the binding launches 64 threads -- together with
        # --maxrregcount=128. TIRx would otherwise emit __launch_bounds__(64), which
        # lets ptxas spend 118 registers per thread against the source's 105; at 64
        # threads that is 8 blocks/SM instead of 9. Asking for 9 brings it to 96.
        # The original byte arena contains three swizzled matrix regions followed
        # by typed scalar/vector regions.  K owns that layout directly; the small
        # padding allocation preserves the source's fixed offsets exactly.
        smem = K.smem_pool()
        s_state0 = smem.alloc((ROWS_PER_CTA, 64), K.bf16, swizzle=K.SW128B)
        s_state1 = smem.alloc((ROWS_PER_CTA, 64), K.bf16, swizzle=K.SW128B)
        # sVec is physically 128x16 bf16.  Represent the same 4096-byte linear
        # surface as 32x64 so K's 128B atom owns the exact source xor layout.
        s_vec = smem.alloc((32, 64), K.bf16, swizzle=K.SW128B)
        s_k = smem.alloc((NUM_TOKENS, HEAD_DIM), K.f32)
        s_d = smem.alloc((NUM_TOKENS, HEAD_DIM), K.f32)
        s_beta = smem.alloc((NUM_TOKENS,), K.f32)
        s_slot = smem.alloc((NUM_TOKENS,), K.i32)
        s_token = smem.alloc((NUM_TOKENS,), K.i32)
        s_init = smem.alloc((1,), K.i32)
        smem.alloc((3,), K.i32)  # source padding: 14364..14375
        s_l = smem.alloc((NUM_TOKENS, NUM_TOKENS), K.f32)
        s_r = smem.alloc((NUM_TOKENS, NUM_TOKENS), K.f32)
        s_u = smem.alloc((NUM_TOKENS, ROWS_PER_CTA), K.f32)
        if smem.bytes != 14664:
            raise AssertionError(f"unexpected T=2 shared layout size {smem.bytes}")
        smem.commit(SMEM_TOTAL)

        def vec_ptr(k_idx, column):
            return s_vec.ptr_to(k_idx // 4, (k_idx % 4) * 16 + column)

        def st_shared_f32(ptr, value):
            K.ptx.st.shared.b32(ptr, K.reinterpret("uint32", value))

        def ld_shared_f32(ptr):
            word = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(word, ptr)
            return K.reinterpret("float32", word)

        def st_shared_i32(ptr, value):
            K.ptx.st.shared.b32(ptr, K.reinterpret("uint32", value))

        def ld_shared_i32(ptr):
            word = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(word, ptr)
            return K.reinterpret("int32", word)

        def ld_shared_f32x4(ptr, dst, base):
            words = K.alloc_local((4,), "uint32")
            K.ptx.ld.shared.v4.b32(words[0], words[1], words[2], words[3], ptr)
            for i in range(4):
                K.assign(dst[base + i], K.reinterpret("float32", words[i]))

        def st_shared_u32x4(ptr, words):
            K.ptx.st.shared.v4.b32(ptr, words[0], words[1], words[2], words[3])

        def st_shared_b16(ptr, bits):
            K.ptx.st.shared.b16(ptr, bits)

        def ldmatrix_x4(ptr, frag, trans):
            if trans:
                K.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                    frag[0], frag[1], frag[2], frag[3], ptr
                )
            else:
                K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    frag[0], frag[1], frag[2], frag[3], ptr
                )

        # --- work decomposition and lane roles (:142-160) ----------------------
        value_tile = K.local_scalar("int32", init=work % VALUE_SPLIT)
        hv = K.local_scalar("int32", init=work // VALUE_SPLIT)
        query_head = K.local_scalar("int32", init=hv // HEAD_RATIO)
        warp = K.local_scalar("int32")
        K.assign(warp, tid // 32)  # == token index in phases A and C
        lane = K.local_scalar("int32", init=tid % 32)
        lane_quad = K.local_scalar("int32", init=lane % 4)
        frag_row = K.local_scalar("int32", init=lane // 4)
        quad_base = K.local_scalar("int32", init=lane - lane_quad)
        group = K.local_scalar("int32", init=tid // 16)
        lane_group = K.local_scalar("int32", init=tid % 16)
        k_start = K.local_scalar("int32", init=lane_group * 8)
        elem_start = K.local_scalar("int32", init=lane * 4)
        tile_row_base = K.local_scalar("int32", init=value_tile * ROWS_PER_CTA)
        owned_row_base = K.local_scalar("int32", init=group * 8)
        token_base = K.local_scalar("int32")
        K.assign(token_base, _load_i32(cu, n))
        seq_len = K.local_scalar("int32")
        K.assign(seq_len, _load_i32(cu, n + 1) - token_base)

        r_q = K.alloc_local((4,), "float32")
        r_k = K.alloc_local((4,), "float32")
        r_d = K.alloc_local((4,), "float32")

        # =======================================================================
        # Phase A: token preprocess, warp <-> token  (:180-290)
        # =======================================================================
        with K.If(warp < NUM_TOKENS), K.Then():
            token = K.local_scalar("int32", init=warp)
            active_token = token < seq_len
            token_pos = K.local_scalar(
                "int32", init=K.if_then_else(active_token, token_base + token, 0)
            )
            qk_base = K.local_scalar(
                "int32", init=(token_pos * NUM_HEADS + query_head) * HEAD_DIM + elem_start
            )
            gate_base = K.local_scalar(
                "int32", init=token_pos * GATE_TOKEN_STRIDE + hv * HEAD_DIM + elem_start
            )

            q_words = _load_u32x2(q, qk_base)
            k_words = _load_u32x2(k, qk_base)
            g_words = _load_u32x2(g, gate_base)
            for pair in range(2):
                K.assign(r_q[2 * pair], _widen_lo(q_words[pair]))
                K.assign(r_q[2 * pair + 1], _widen_hi(q_words[pair]))
                K.assign(r_k[2 * pair], _widen_lo(k_words[pair]))
                K.assign(r_k[2 * pair + 1], _widen_hi(k_words[pair]))
                K.assign(r_d[2 * pair], _widen_lo(g_words[pair]))
                K.assign(r_d[2 * pair + 1], _widen_hi(g_words[pair]))

            # Index-ordered accumulation (:241-244); the first term has a zero addend.
            q_sq = K.local_scalar("float32")
            k_sq = K.local_scalar("float32")
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
            q_norm = K.local_scalar("float32")
            K.assign(q_norm, _mul(_rsqrt(_add(q_sq, K.float32(L2_EPS))), scale))
            k_norm = K.local_scalar("float32")
            K.assign(k_norm, _rsqrt(_add(k_sq, K.float32(L2_EPS))))

            k_pub = K.alloc_local((4,), "uint32")
            d_pub = K.alloc_local((4,), "uint32")
            for i in range(4):
                K.assign(r_q[i], _mul(r_q[i], q_norm))
                K.assign(r_k[i], _mul(r_k[i], k_norm))
                K.assign(r_d[i], _expf(r_d[i]))
                K.assign(k_pub[i], K.reinterpret("uint32", r_k[i]))
                K.assign(d_pub[i], K.reinterpret("uint32", r_d[i]))
            st_shared_u32x4(s_k.ptr_to([token, elem_start]), k_pub)
            st_shared_u32x4(s_d.ptr_to([token, elem_start]), d_pub)

            with K.If(lane == 0), K.Then():
                raw_slot = K.local_scalar("int32")
                K.assign(raw_slot, _load_i32(ssm_idx, n * NUM_TOKENS + token))
                st_shared_i32(s_slot.ptr_to([token]), K.if_then_else(active_token, raw_slot, -1))
                st_shared_i32(s_token.ptr_to([token]), token_pos)
                st_shared_f32(
                    s_beta.ptr_to([token]), _load_bf16_f32(beta, token_pos * NUM_VALUE_HEADS + hv)
                )
                with K.If(token == 0), K.Then():
                    # nat is live at T=2: it picks the initial checkpoint slot,
                    # clamped at both ends (:279-287).
                    accepted = K.local_scalar("int32")
                    K.assign(accepted, K.min(K.max(_load_i32(nat, n) - 1, 0), NUM_TOKENS - 1))
                    initial_slot = K.local_scalar("int32")
                    K.assign(initial_slot, _load_i32(ssm_idx, n * NUM_TOKENS + accepted))
                    st_shared_i32(s_init.ptr_to([0]), K.max(initial_slot, 0))

        K.ptx.bar.sync(K.uint32(0), K.uint32(THREADS))

        # =======================================================================
        # Phase B: state gather and sState stage, all 64 threads  (:292-329)
        # =======================================================================
        init_slot = K.local_scalar("int32")
        K.assign(init_slot, ld_shared_i32(s_init.ptr_to([0])))
        # Materialised, not left as an expression: the eight unrolled row loads
        # below share this 64-bit base, and re-expanding it per row keeps its
        # int32 inputs live all the way to the tail, which costs register
        # spills inside the body.
        head_base = K.local_scalar(
            "int64",
            init=K.cast(init_slot, "int64") * K.cast(STATE_SLOT_STRIDE, "int64")
            + K.cast(hv * HEAD_DIM * HEAD_DIM, "int64"),
        )
        hist = K.alloc_local((8 * 8,), "float32")
        for row_local in range(8):
            row_l = K.local_scalar("int32", init=owned_row_base + row_local)
            pack = _load_u32x4(
                state, head_base + K.cast((tile_row_base + row_l) * HEAD_DIM + k_start, "int64")
            )
            for pr in range(4):
                K.assign(hist[row_local * 8 + 2 * pr], _widen_lo(pack[pr]))
                K.assign(hist[row_local * 8 + 2 * pr + 1], _widen_hi(pack[pr]))
            # The bf16 bits go to shared unmodified; the swizzle is on the byte
            # offset. lane_group < 8 lands in sState0, the rest in sState1.
            with K.If(lane_group < 8):
                with K.Then():
                    st_shared_u32x4(s_state0.ptr_to(row_l, k_start), pack)
                with K.Else():
                    st_shared_u32x4(s_state1.ptr_to(row_l, k_start - 64), pack)

        # =======================================================================
        # Phase C: sVec columns and the WY coefficients  (:330-404)
        # =======================================================================
        with K.If(warp < NUM_TOKENS), K.Then():
            token_c = K.local_scalar("int32", init=warp)
            for i in range(4):
                k_idx = K.local_scalar("int32", init=elem_start + i)
                prefix = K.local_scalar("float32", init=K.float32(1.0))
                for j in range(NUM_TOKENS):
                    with K.If(token_c >= j), K.Then():
                        K.assign(prefix, _mul(prefix, ld_shared_f32(s_d.ptr_to([j, k_idx]))))
                st_shared_b16(
                    vec_ptr(k_idx, token_c),
                    _ptx_un("cvt.rn.bf16.f32", _mul(prefix, r_k[i]), dtype="uint16"),
                )
                st_shared_b16(
                    vec_ptr(k_idx, 4 + token_c),
                    _ptx_un("cvt.rn.bf16.f32", _mul(prefix, r_q[i]), dtype="uint16"),
                )

            ratio = K.alloc_local((4,), "float32")
            for i in range(4):
                K.assign(ratio[i], K.float32(1.0))
            for source_offset in range(NUM_TOKENS):
                source_token = K.local_scalar("int32", init=token_c - source_offset)
                with K.If(source_token >= 0), K.Then():
                    dot_kk = K.local_scalar("float32")
                    dot_qk = K.local_scalar("float32")
                    K.assign(dot_kk, K.float32(0.0))
                    K.assign(dot_qk, K.float32(0.0))
                    sk_vec = K.alloc_local((4,), "float32")
                    ld_shared_f32x4(s_k.ptr_to([source_token, elem_start]), sk_vec, 0)
                    for i in range(4):
                        # Source order: r * source_k * ratio  (:372-375).
                        K.assign(dot_kk, _fma(_mul(r_k[i], sk_vec[i]), ratio[i], dot_kk))
                        K.assign(dot_qk, _fma(_mul(r_q[i], sk_vec[i]), ratio[i], dot_qk))
                    for off in range(5):
                        K.assign(dot_kk, _add(dot_kk, _shfl_bfly(dot_kk, 16 >> off)))
                    for off in range(5):
                        K.assign(dot_qk, _add(dot_qk, _shfl_bfly(dot_qk, 16 >> off)))
                    with K.If(lane == 0), K.Then():
                        beta_source = K.local_scalar("float32")
                        K.assign(beta_source, ld_shared_f32(s_beta.ptr_to([source_token])))
                        with K.If(source_token < token_c), K.Then():
                            st_shared_f32(
                                s_l.ptr_to([token_c, source_token]), _mul(beta_source, dot_kk)
                            )
                        st_shared_f32(
                            s_r.ptr_to([token_c, source_token]), _mul(beta_source, dot_qk)
                        )
                    with K.If(source_token > 0), K.Then():
                        sd_vec = K.alloc_local((4,), "float32")
                        ld_shared_f32x4(s_d.ptr_to([source_token, elem_start]), sd_vec, 0)
                        for i in range(4):
                            K.assign(ratio[i], _mul(ratio[i], sd_vec[i]))

        K.ptx.bar.sync(K.uint32(0), K.uint32(THREADS))

        # =======================================================================
        # Phase D: the MMA chain, warp <-> 16 value rows  (:406-438)
        # =======================================================================
        acc = K.alloc_local((4,), "float32", align=4)
        with K.If(warp < NUM_TOKENS), K.Then():
            vec_frag = K.alloc_local((4,), "uint32", align=4)
            state_frag = K.alloc_local((4,), "uint32", align=4)
            for state_half in range(2):
                for mma_step in range(4):
                    mma_k = K.local_scalar("int32", init=mma_step * 16)
                    global_k = K.local_scalar("int32", init=state_half * 64 + mma_k)
                    ldmatrix_x4(vec_ptr(global_k + lane % 16, lane // 16 * 8), vec_frag, True)
                    if state_half == 0:
                        ldmatrix_x4(
                            s_state0.ptr_to(warp * 16 + lane % 16, mma_k + lane // 16 * 8),
                            state_frag,
                            False,
                        )
                    else:
                        ldmatrix_x4(
                            s_state1.ptr_to(warp * 16 + lane % 16, mma_k + lane // 16 * 8),
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
        u_lo = K.alloc_local((NUM_TOKENS,), "float32")
        u_hi = K.alloc_local((NUM_TOKENS,), "float32")
        with K.If(warp < NUM_TOKENS), K.Then():
            ha_lo = K.alloc_local((NUM_TOKENS,), "float32")
            ha_hi = K.alloc_local((NUM_TOKENS,), "float32")
            for t in range(NUM_TOKENS):
                K.assign(ha_lo[t], _shfl_idx(acc[t], quad_base))
                K.assign(ha_hi[t], _shfl_idx(acc[2 + t], quad_base))

            with K.If(lane_quad == 2), K.Then():
                row_lo = K.local_scalar("int32", init=warp * 16 + frag_row)
                row_hi = K.local_scalar("int32", init=row_lo + 8)
                for t in range(NUM_TOKENS):
                    base_t = K.local_scalar("int32")
                    K.assign(
                        base_t,
                        (ld_shared_i32(s_token.ptr_to([t])) * NUM_VALUE_HEADS + hv) * HEAD_DIM,
                    )
                    solved_lo = K.local_scalar("float32")
                    solved_hi = K.local_scalar("float32")
                    K.assign(
                        solved_lo,
                        _sub(_load_bf16_f32(v, base_t + tile_row_base + row_lo), ha_lo[t]),
                    )
                    K.assign(
                        solved_hi,
                        _sub(_load_bf16_f32(v, base_t + tile_row_base + row_hi), ha_hi[t]),
                    )
                    for prev in range(NUM_TOKENS):
                        if prev < t:
                            lts = K.local_scalar("float32")
                            K.assign(lts, ld_shared_f32(s_l.ptr_to([t, prev])))
                            K.assign(solved_lo, _sub(solved_lo, _mul(lts, u_lo[prev])))
                            K.assign(solved_hi, _sub(solved_hi, _mul(lts, u_hi[prev])))
                    K.assign(u_lo[t], solved_lo)
                    K.assign(u_hi[t], solved_hi)

                # ---- outputs, both tokens (:484-531) --------------------------
                for t in range(NUM_TOKENS):
                    out_lo = K.local_scalar("float32")
                    out_hi = K.local_scalar("float32")
                    K.assign(out_lo, acc[t])
                    K.assign(out_hi, acc[2 + t])
                    for src in range(NUM_TOKENS):
                        # The s > t coefficient is a real zero-operand fma, not a
                        # skipped iteration (:491-493, :515-517).
                        coef = K.local_scalar("float32", init=K.float32(0.0))
                        if src <= t:
                            K.assign(coef, ld_shared_f32(s_r.ptr_to([t, src])))
                        K.assign(out_lo, _fma(coef, u_lo[src], out_lo))
                        K.assign(out_hi, _fma(coef, u_hi[src], out_hi))
                    active_t = ld_shared_i32(s_slot.ptr_to([t])) >= 0
                    base_o = K.local_scalar("int32")
                    K.assign(
                        base_o,
                        (ld_shared_i32(s_token.ptr_to([t])) * NUM_VALUE_HEADS + hv) * HEAD_DIM
                        + tile_row_base,
                    )
                    _store_f32_as_bf16(out, base_o + row_lo, out_lo, active_t)
                    _store_f32_as_bf16(out, base_o + row_hi, out_hi, active_t)
                    _store_f32_as_bf16(out, base_o + row_lo, K.float32(0.0), K.Not(active_t))
                    _store_f32_as_bf16(out, base_o + row_hi, K.float32(0.0), K.Not(active_t))

                # ---- publish sU (:532-541) ------------------------------------
                for t in range(NUM_TOKENS):
                    st_shared_f32(s_u.ptr_to([t, row_lo]), u_lo[t])
                    st_shared_f32(s_u.ptr_to([t, row_hi]), u_hi[t])
            # A warp barrier suffices: the sU rows this warp writes are exactly the
            # rows its own threads read back below (:543).
            K.cuda.warp_sync()

        # =======================================================================
        # Phase H: recurrence and checkpoints, all 64 threads  (:659-695)
        # =======================================================================
        words_w = K.alloc_local((4,), "uint32")
        sd_t = K.alloc_local((8,), "float32")
        sk_t = K.alloc_local((8,), "float32")
        for t in range(NUM_TOKENS):
            slot_t = K.local_scalar("int32")
            K.assign(slot_t, ld_shared_i32(s_slot.ptr_to([t])))
            beta_t = K.local_scalar("float32")
            K.assign(beta_t, ld_shared_f32(s_beta.ptr_to([t])))
            # The gate and key slices depend only on (t, k_start), not on the row,
            # so the source loads each 8-float slice once per token as two 16-byte
            # reads instead of reloading them in the row loop (:673 is 12
            # ld.shared.v4.b32 with no scalar shared loads at all).
            ld_shared_f32x4(s_d.ptr_to([t, k_start]), sd_t, 0)
            ld_shared_f32x4(s_d.ptr_to([t, k_start + 4]), sd_t, 4)
            ld_shared_f32x4(s_k.ptr_to([t, k_start]), sk_t, 0)
            ld_shared_f32x4(s_k.ptr_to([t, k_start + 4]), sk_t, 4)
            for row_local in range(8):
                row_h = K.local_scalar("int32", init=owned_row_base + row_local)
                update = K.local_scalar("float32")
                K.assign(update, _mul(ld_shared_f32(s_u.ptr_to([t, row_h])), beta_t))
                for i in range(8):
                    # The source writes `hist*sD + update*sK` (:673) and the
                    # compiler contracts the FIRST product: update*sK is rounded,
                    # hist*sD is fused. This is the only stateful accumulation and
                    # it feeds both the checkpoint and token 1's history.
                    K.assign(
                        hist[row_local * 8 + i],
                        _fma(hist[row_local * 8 + i], sd_t[i], _mul(update, sk_t[i])),
                    )
                for pr in range(4):
                    K.assign(
                        words_w[pr],
                        _pack_bf16x2(
                            hist[row_local * 8 + 2 * pr + 1], hist[row_local * 8 + 2 * pr]
                        ),
                    )
                # The recurrence advances unconditionally; only the store is
                # slot-predicated.
                with K.If(slot_t >= 0), K.Then():
                    _store_u32x4(
                        state,
                        K.cast(slot_t, "int64") * K.cast(STATE_SLOT_STRIDE, "int64")
                        + K.cast(
                            hv * HEAD_DIM * HEAD_DIM + (tile_row_base + row_h) * HEAD_DIM + k_start,
                            "int64",
                        ),
                        words_w,
                    )

    return _flashkda_decode_t2_precomputed.func


def get_kernel(**kwargs: Any):
    """Return the specialized FlashKDA cake T=2 decode PrimFunc."""
    return _make_flashkda_decode_t2_precomputed(_specialization(kwargs))


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


def _flashinfer_reference(case: dict[str, Any]) -> torch.Tensor:
    """Run the frozen cake export itself on the reference state pool.

    Uses the JIT module's direct ABI rather than
    ``kda_decode.recurrent_kda(backend="cake")`` so the comparison is
    kernel-only and every metadata combination (negative slots, arbitrary
    num_accepted_tokens) is reachable.
    """
    from flashinfer.jit.flash_kda_decode import get_flash_kda_decode_module

    device = case["device"]
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == (10, 0):
        target = "sm100f" if torch.version.cuda and torch.version.cuda >= "12.9" else "sm100a"
    elif (major, minor) == (10, 3):
        # Non-direct variants are not exported for sm103a
        # (flash_kda_decode.py:213-217).
        target = "sm100f"
    else:
        raise SkipTest(f"FlashKDA cake decode has no export for compute capability {major}.{minor}")

    module = get_flash_kda_decode_module("d128_t2_precomputed_split4", target)
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
        msg=lambda m: f"output vs flashinfer cake export\n{m}",
    )
    torch.testing.assert_close(
        tirx_state.float(),
        case["reference_state_raw"].float(),
        rtol=_RTOL,
        atol=_ATOL,
        msg=lambda m: f"state vs flashinfer cake export\n{m}",
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
