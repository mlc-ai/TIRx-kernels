# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Shared kern helpers for the FlashInfer quantization kernel ports.

Every helper mirrors the source CuTe-DSL inline-asm block instruction for
instruction (``flashinfer/quantization/quantization_cute_dsl_utils.py`` and
``flashinfer/cute_dsl/fp4_common.py``); see the kernel sketches under
``.agents/sketch/`` for the validated instruction selections.  All helpers run
while a ``@K.kernel`` body is traced: they build PrimExprs or emit statements
into the active builder frame.  No tile primitives are used.
"""

import tirx_kernels.kern as K

# ---------------------------------------------------------------------------
# Global memory copies (fp4_common.py:134 ld_global_v4_u32, :242 st_global_u64)
# ---------------------------------------------------------------------------


def ld_global_v4_u32(addr):
    """One ``ld.global.v4.u32`` (no ``.nc``); returns the 4-word local tile."""
    v = K.alloc_local([4], "uint32")
    K.ptx.ld.global_.v4.b32(v[0], v[1], v[2], v[3], addr)
    return v


def st_global_u64(addr, val):
    """One ``st.global.u64`` (emitted as st.global.b64)."""
    K.ptx.st.global_.b64(addr, val)


def st_global_u8(addr, val):
    """One byte store (emitted as st.global.b8)."""
    K.ptx.st.global_.b8(addr, val)


# ---------------------------------------------------------------------------
# Packed abs/max (fp4_common.py:583 habs2, :599 hmax2, bf16 variants identical)
# ---------------------------------------------------------------------------


def habs2(x):
    """``and.b32`` with 0x7FFF7FFF: clear both fp16/bf16 sign bits."""
    return K.bitwise_and(x, K.uint32(0x7FFF7FFF))


def hmax2(a, b, dtype):
    """``max.f16x2`` / ``max.bf16x2`` on packed pairs."""
    out = K.local_scalar("uint32")
    if dtype == "float16":
        K.ptx.max.f16x2(out, a, b)
    else:
        K.ptx.max.bf16x2(out, a, b)
    return out


def absmax_8(v, dtype):
    """Tree abs-max over 8 packed words (utils:691 half2_max_abs_8, :728 bf16)."""
    a = [habs2(v[i]) for i in range(8)]
    m01 = hmax2(a[0], a[1], dtype)
    m23 = hmax2(a[2], a[3], dtype)
    m45 = hmax2(a[4], a[5], dtype)
    m67 = hmax2(a[6], a[7], dtype)
    m03 = hmax2(m01, m23, dtype)
    m47 = hmax2(m45, m67, dtype)
    return hmax2(m03, m47, dtype)


def absmax_4(v, dtype):
    """Tree abs-max over 4 packed words (utils:616 half2_max_abs_4, :633 bf16)."""
    a = [habs2(v[i]) for i in range(4)]
    return hmax2(hmax2(a[0], a[1], dtype), hmax2(a[2], a[3], dtype), dtype)


def unpack_lo_f32(word, dtype):
    """Low lane of a packed pair as f32 (mov.b32 {lo,hi} + cvt.f32.f16; bf16:
    and.b32 + shl.b32 + mov.b32)."""
    if dtype == "float16":
        return K.cast(
            K.reinterpret("float16", K.cast(K.bitwise_and(word, K.uint32(0xFFFF)), "uint16")),
            "float32",
        )
    return K.reinterpret(
        "float32", K.shift_left(K.bitwise_and(word, K.uint32(0xFFFF)), K.uint32(16))
    )


def unpack_hi_f32(word, dtype):
    """High lane of a packed pair as f32 (bf16: shr.b32 + shl.b32 + mov.b32)."""
    if dtype == "float16":
        return K.cast(
            K.reinterpret("float16", K.cast(K.shift_right(word, K.uint32(16)), "uint16")), "float32"
        )
    return K.reinterpret("float32", K.shift_left(K.shift_right(word, K.uint32(16)), K.uint32(16)))


def pair_max_to_f32(x, dtype):
    """Max of the two packed lanes as f32, then ``max.f32``
    (utils:95 hmax_reduce_to_f32, :122 bfloat2_hmax_reduce_to_f32)."""
    return fmax_f32(unpack_lo_f32(x, dtype), unpack_hi_f32(x, dtype))


# ---------------------------------------------------------------------------
# f32 max and butterfly reductions (fp4_common.py:514 fmax_f32, utils:505/515)
# ---------------------------------------------------------------------------


def fmax_f32(a, b):
    """``max.f32``."""
    out = K.local_scalar("float32")
    K.ptx.max.f32(out, a, b)
    return out


def shfl_xor_f32(val, lane_xor):
    """``shfl.sync.bfly.b32`` with full membermask (utils:498 shuffle_xor_f32)."""
    out = K.local_scalar("uint32")
    K.ptx.shfl_sync.bfly.b32(
        out, K.reinterpret("uint32", val), K.uint32(lane_xor), K.uint32(31), K.uint32(0xFFFFFFFF)
    )
    return K.reinterpret("float32", out)


def reduce_max_2threads(val):
    """1 butterfly round (utils:505)."""
    return fmax_f32(val, shfl_xor_f32(val, 1))


def reduce_max_4threads(val):
    """2 butterfly rounds, offsets 1 then 2 (utils:515)."""
    val = fmax_f32(val, shfl_xor_f32(val, 1))
    return fmax_f32(val, shfl_xor_f32(val, 2))


# ---------------------------------------------------------------------------
# UE8M0 scale factors (utils:157 float_to_ue8m0_fast, :207 ue8m0_to_inv_scale_fast)
# ---------------------------------------------------------------------------


def mul_f32(a, b):
    """``mul.f32`` (non-ftz, exactly as the source asm blocks)."""
    out = K.local_scalar("float32")
    K.ptx.mul.f32(out, a, b)
    return out


def float_to_ue8m0(value):
    """float_to_ue8m0_fast (utils:157), instruction for instruction.

    setp.le.f32 / mov.b32 / shr.b32 / and.b32 / setp.ne.u32 / selp.u32 /
    setp.eq.u32 / setp.le.u32 / and.pred / @p mov (selp) / add.u32 /
    setp.gt.u32 / selp.u32 x2 -- all explicit so the FTZ-flagged host
    compiler cannot reinterpret the float compare or the selects.
    """
    bits = K.reinterpret("uint32", value)
    exp = K.bitwise_and(K.shift_right(bits, K.uint32(23)), K.uint32(255))
    mant = K.bitwise_and(bits, K.uint32(0x7FFFFF))
    p_zero = K.local_scalar("uint32")
    K.ptx.setp.le.f32(p_zero, value, K.float32(0.0))
    p_has_mant = K.local_scalar("uint32")
    K.ptx.setp.ne.u32(p_has_mant, mant, K.uint32(0))
    bump = K.local_scalar("uint32")
    K.ptx.selp.u32(bump, K.uint32(1), K.uint32(0), K.ptx.pred(p_has_mant))
    p_exp_zero = K.local_scalar("uint32")
    K.ptx.setp.eq.u32(p_exp_zero, exp, K.uint32(0))
    p_tiny = K.local_scalar("uint32")
    K.ptx.setp.le.u32(p_tiny, mant, K.uint32(0x400000))
    K.ptx.and_.pred(p_tiny, K.ptx.pred(p_exp_zero), K.ptx.pred(p_tiny))
    # @p_tiny mov.u32 bump, 0  (selp form)
    K.ptx.selp.u32(bump, K.uint32(0), bump, K.ptx.pred(p_tiny))
    result = exp + bump
    p_ovf = K.local_scalar("uint32")
    K.ptx.setp.gt.u32(p_ovf, result, K.uint32(254))
    out = K.local_scalar("uint32")
    K.ptx.selp.u32(out, K.uint32(254), result, K.ptx.pred(p_ovf))
    K.ptx.selp.u32(out, K.uint32(0), out, K.ptx.pred(p_zero))
    return out


def ue8m0_to_inv_scale(ue8m0_val):
    """ue8m0_to_inv_scale_fast (utils:207), instruction for instruction.

    setp.eq.u32 / sub.s32 / max.s32 / shl.b32 / mov.b32 / @p_zero mov (selp).
    """
    p_zero = K.local_scalar("uint32")
    K.ptx.setp.eq.u32(p_zero, ue8m0_val, K.uint32(0))
    new_exp = K.max(K.int32(254) - K.cast(ue8m0_val, "int32"), K.int32(0))
    inv = K.reinterpret("float32", K.shift_left(K.cast(new_exp, "uint32"), K.uint32(23)))
    out = K.local_scalar("float32")
    K.ptx.selp.f32(out, K.float32(0.0), inv, K.ptx.pred(p_zero))
    return out


# ---------------------------------------------------------------------------
# FP8 E4M3 conversion + packing (utils:249/:286 *_to_fp8x2_scaled, :326 pack)
# ---------------------------------------------------------------------------


def fp8x2_scaled(word, inv_scale, dtype):
    """2 packed fp16/bf16 -> 2 FP8-E4M3 bytes scaled; returns uint32 (zext u16).

    cvt.rn.satfinite.e4m3x2.f32 takes the high element first (source order).
    """
    lo = mul_f32(unpack_lo_f32(word, dtype), inv_scale)
    hi = mul_f32(unpack_hi_f32(word, dtype), inv_scale)
    pair = K.local_scalar("uint16")
    K.ptx.cvt.rn.satfinite.e4m3x2.f32(pair, hi, lo)
    return K.cast(pair, "uint32")


def pack_fp8x8_to_u64(b01, b23, b45, b67):
    """Gather 4 x (2 FP8 bytes in low u16) into one uint64 (utils:326)."""
    lo = K.bitwise_or(
        K.bitwise_and(b01, K.uint32(0xFFFF)),
        K.shift_left(K.bitwise_and(b23, K.uint32(0xFFFF)), K.uint32(16)),
    )
    hi = K.bitwise_or(
        K.bitwise_and(b45, K.uint32(0xFFFF)),
        K.shift_left(K.bitwise_and(b67, K.uint32(0xFFFF)), K.uint32(16)),
    )
    out = K.local_scalar("uint64")
    K.ptx.mov.b64(out, lo, hi)
    return out


def fp8x8_scaled(v, inv_scale, dtype):
    """8 packed elements (4 words) -> one uint64 of 8 FP8 bytes (utils:650/:668)."""
    return pack_fp8x8_to_u64(
        fp8x2_scaled(v[0], inv_scale, dtype),
        fp8x2_scaled(v[1], inv_scale, dtype),
        fp8x2_scaled(v[2], inv_scale, dtype),
        fp8x2_scaled(v[3], inv_scale, dtype),
    )


# ---------------------------------------------------------------------------
# Scale-factor swizzle index math (utils:535 128x4, :569 8x4, :601 linear)
# ---------------------------------------------------------------------------


def sf_offset_128x4(row, col, padded_cols):
    """Swizzled 128x4 SF byte offset; padded_cols is a compile-time int."""
    return (
        K.truncmod(col, K.int32(4))
        + K.truncdiv(col, K.int32(4)) * 512
        + K.truncmod(row, K.int32(32)) * 16
        + K.truncdiv(K.truncmod(row, K.int32(128)), K.int32(32)) * 4
        + K.truncdiv(row, K.int32(128)) * (128 * padded_cols)
    )


def sf_offset_8x4(row, col, padded_cols):
    """Swizzled 8x4 SF byte offset ([mTiles, kTiles, 8, 4] tiles of 32)."""
    num_k_tiles = (padded_cols + 3) // 4
    return (
        K.truncdiv(row, K.int32(8)) * (num_k_tiles * 32)
        + K.truncdiv(col, K.int32(4)) * 32
        + K.truncmod(row, K.int32(8)) * 4
        + K.truncmod(col, K.int32(4))
    )


# ---------------------------------------------------------------------------
# MXFP4/NVFP4 helpers: rcp, scaled unpack, e2m1 packing
# ---------------------------------------------------------------------------


def rcp_approx_ftz(a):
    """``rcp.approx.ftz.f32`` (fp4_common.py:394)."""
    out = K.local_scalar("float32")
    K.ptx.rcp.approx.ftz.f32(out, a)
    return out


def float2_scaled(word, inv_scale, dtype):
    """half2/bfloat2_to_float2_scaled (utils:376/:406): unpack + 2x mul.f32."""
    lo = mul_f32(unpack_lo_f32(word, dtype), inv_scale)
    hi = mul_f32(unpack_hi_f32(word, dtype), inv_scale)
    return lo, hi


def cvt_e2m1x8(vals):
    """cvt_e2m1x8_f32 (utils:442): 4x cvt.rn.satfinite.e2m1x2.f32 + byte pack.

    The source's 4 x b8 ``mov.b32`` pack is not registered in the dialect;
    the byte gather uses b16-pair shifts plus the registered ``mov.b32``
    (2 x b16) -- the native form proven by silu_and_mul_nvfp4_experts_quantize.
    ``vals`` is 8 f32 in element order; the cvt takes the high element first.
    """
    bytes_ = K.alloc_local([4], "uint8")
    for i in range(4):
        K.ptx.cvt.rn.satfinite.e2m1x2.f32(bytes_[i], vals[2 * i + 1], vals[2 * i])
    w0 = K.cast(bytes_[0], "uint16") | (K.cast(bytes_[1], "uint16") << K.uint16(8))
    w1 = K.cast(bytes_[2], "uint16") | (K.cast(bytes_[3], "uint16") << K.uint16(8))
    out = K.local_scalar("uint32")
    K.ptx.mov.b32(out, w0, w1)
    return out


def pack_u32x2_to_u64(lo, hi):
    """(u64(hi) << 32) | u64(lo): plain u64 shift/or (utils:996-997)."""
    return K.bitwise_or(K.shift_left(K.cast(hi, "uint64"), K.uint64(32)), K.cast(lo, "uint64"))


# ---------------------------------------------------------------------------
# NVFP4 helpers: E4M3 scale factors, output scale, SwiGLU fusion
# ---------------------------------------------------------------------------


def add_f32(a, b):
    """``add.f32`` (non-ftz, as the source asm/lowering)."""
    out = K.local_scalar("float32")
    K.ptx.add.f32(out, a, b)
    return out


def div_rn_f32(a, b):
    """``div.rn.f32`` (fp4_common.py fdiv_rn; the silu path is not fast-div)."""
    out = K.local_scalar("float32")
    K.ptx.div.rn.f32(out, a, b)
    return out


def ex2_approx_ftz(a):
    """``ex2.approx.ftz.f32``."""
    out = K.local_scalar("float32")
    K.ptx.ex2.approx.ftz.f32(out, a)
    return out


def silu_f32(g):
    """_silu_f32 (utils:1731): mul.f32 by -log2e (folds -g) + ex2.approx.ftz.f32
    + add.f32(+1.0) + div.rn.f32."""
    e = ex2_approx_ftz(mul_f32(g, K.float32(-1.4426950408889634)))
    return div_rn_f32(g, add_f32(e, K.float32(1.0)))


def cvt_f32x2_to_packed(lo, hi, dtype):
    """cvt_f32x2_to_half2/bfloat2 (fp4_common:880/:910): 2x scalar cvt +
    mov.b32 {h0,h1}."""
    h0 = K.local_scalar("uint16")
    h1 = K.local_scalar("uint16")
    if dtype == "float16":
        K.ptx.cvt.rn.f16.f32(h0, lo)
        K.ptx.cvt.rn.f16.f32(h1, hi)
    else:
        K.ptx.cvt.rn.bf16.f32(h0, lo)
        K.ptx.cvt.rn.bf16.f32(h1, hi)
    out = K.local_scalar("uint32")
    K.ptx.mov.b32(out, h0, h1)
    return out


def silu_and_mul_pair(gate, up, dtype):
    """_silu_and_mul_half2/_bfloat2 (utils:1740/:1752): unpack both pairs with
    scale 1.0, silu(g)*u per scalar in f32, repack."""
    g0 = mul_f32(unpack_lo_f32(gate, dtype), K.float32(1.0))
    g1 = mul_f32(unpack_hi_f32(gate, dtype), K.float32(1.0))
    u0 = mul_f32(unpack_lo_f32(up, dtype), K.float32(1.0))
    u1 = mul_f32(unpack_hi_f32(up, dtype), K.float32(1.0))
    a0 = mul_f32(silu_f32(g0), u0)
    a1 = mul_f32(silu_f32(g1), u1)
    return cvt_f32x2_to_packed(a0, a1, dtype)


def cvt_f32_to_e4m3(a):
    """cvt_f32_to_e4m3 (fp4_common.py:811): mov.f32 0 +
    cvt.rn.satfinite.e4m3x2.f32 + cvt.u32.u16."""
    pair = K.local_scalar("uint16")
    K.ptx.cvt.rn.satfinite.e4m3x2.f32(pair, K.float32(0.0), a)
    return K.cast(pair, "uint32")


def nvfp4_compute_output_scale(sf_u32, global_scale):
    """nvfp4_compute_output_scale (fp4_common.py:973), instruction for
    instruction: decode the E4M3 SF through the f16x2 path, then
    rcp(SF_f32 * rcp(global_scale)), with the zero-SF select."""
    pair16 = K.local_scalar("uint16")
    K.ptx.cvt.u16.u32(pair16, sf_u32)
    h2 = K.local_scalar("uint32")
    K.ptx.cvt.rn.f16x2.e4m3x2(h2, pair16)
    sf_f32 = K.cast(
        K.reinterpret("float16", K.cast(K.bitwise_and(h2, K.uint32(0xFFFF)), "uint16")), "float32"
    )
    product = mul_f32(sf_f32, rcp_approx_ftz(global_scale))
    result = rcp_approx_ftz(product)
    p_zero = K.local_scalar("uint32")
    K.ptx.setp.eq.f32(p_zero, sf_f32, K.float32(0.0))
    out = K.local_scalar("float32")
    K.ptx.selp.f32(out, K.float32(0.0), result, K.ptx.pred(p_zero))
    return out


def opaque_i32(x):
    """Identity ``mov.s32`` that keeps a loop stride opaque to the host
    compiler's strength reduction, so the generated loop recomputes addresses
    per iteration the way the source's own binary does (avoids the heavy
    up-front pointer-induction chains nvcc otherwise builds in the prologue).
    Purely a loop-bookkeeping device; the value is unchanged."""
    out = K.local_scalar("int32")
    K.ptx.mov.s32(out, x)
    return out


# ---------------------------------------------------------------------------
# Scalar loads and reductions (per-token kernel)
# ---------------------------------------------------------------------------


def ld_global_f32(buf, idx):
    """Plain ``ld.global.f32`` scalar load (non-nc), as the source emits."""
    out = K.local_scalar("float32")
    K.ptx.ld.global_.f32(out, K.address_of(buf[idx]))
    return out


def warp_reduce_max(val):
    """warp_reduce (fp4_common.py:1356): 5 butterfly rounds, offsets 1..16."""
    for i in range(5):
        val = fmax_f32(val, shfl_xor_f32(val, 1 << i))
    return val
