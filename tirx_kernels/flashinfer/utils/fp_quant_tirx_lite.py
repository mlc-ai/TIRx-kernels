# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Shared tirx-lite helpers for the FlashInfer quantization kernel ports.

Every helper mirrors the source CuTe-DSL inline-asm block instruction for
instruction (``flashinfer/quantization/quantization_cute_dsl_utils.py`` and
``flashinfer/cute_dsl/fp4_common.py``); see the kernel sketches under
``.agents/sketch/`` for the validated instruction selections.  All helpers run
while a ``@txl.kernel`` body is traced: they build PrimExprs or emit statements
into the active builder frame.  No tile primitives are used.
"""

import tirx_kernels.tirx_lite as txl

# ---------------------------------------------------------------------------
# Global memory copies (fp4_common.py:134 ld_global_v4_u32, :242 st_global_u64)
# ---------------------------------------------------------------------------


def ld_global_v4_u32(addr):
    """One ``ld.global.v4.u32`` (no ``.nc``); returns the 4-word local tile."""
    v = txl.alloc_local([4], "uint32")
    txl.ptx.ld.global_.v4.b32(v[0], v[1], v[2], v[3], addr)
    return v


def st_global_u64(addr, val):
    """One ``st.global.u64`` (emitted as st.global.b64)."""
    txl.ptx.st.global_.b64(addr, val)


def st_global_u8(addr, val):
    """One byte store (emitted as st.global.b8)."""
    txl.ptx.st.global_.b8(addr, val)


# ---------------------------------------------------------------------------
# Packed abs/max (fp4_common.py:583 habs2, :599 hmax2, bf16 variants identical)
# ---------------------------------------------------------------------------


def habs2(x):
    """``and.b32`` with 0x7FFF7FFF: clear both fp16/bf16 sign bits."""
    return txl.bitwise_and(x, txl.uint32(0x7FFF7FFF))


def hmax2(a, b, dtype):
    """``max.f16x2`` / ``max.bf16x2`` on packed pairs."""
    out = txl.local_scalar("uint32")
    if dtype == "float16":
        txl.ptx.max.f16x2(out, a, b)
    else:
        txl.ptx.max.bf16x2(out, a, b)
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
        return txl.cast(
            txl.reinterpret("float16", txl.cast(txl.bitwise_and(word, txl.uint32(0xFFFF)), "uint16")),
            "float32",
        )
    return txl.reinterpret(
        "float32", txl.shift_left(txl.bitwise_and(word, txl.uint32(0xFFFF)), txl.uint32(16))
    )


def unpack_hi_f32(word, dtype):
    """High lane of a packed pair as f32 (bf16: shr.b32 + shl.b32 + mov.b32)."""
    if dtype == "float16":
        return txl.cast(
            txl.reinterpret("float16", txl.cast(txl.shift_right(word, txl.uint32(16)), "uint16")), "float32"
        )
    return txl.reinterpret("float32", txl.shift_left(txl.shift_right(word, txl.uint32(16)), txl.uint32(16)))


def pair_max_to_f32(x, dtype):
    """Max of the two packed lanes as f32, then ``max.f32``
    (utils:95 hmax_reduce_to_f32, :122 bfloat2_hmax_reduce_to_f32)."""
    return fmax_f32(unpack_lo_f32(x, dtype), unpack_hi_f32(x, dtype))


# ---------------------------------------------------------------------------
# f32 max and butterfly reductions (fp4_common.py:514 fmax_f32, utils:505/515)
# ---------------------------------------------------------------------------


def fmax_f32(a, b):
    """``max.f32``."""
    out = txl.local_scalar("float32")
    txl.ptx.max.f32(out, a, b)
    return out


def shfl_xor_f32(val, lane_xor):
    """``shfl.sync.bfly.b32`` with full membermask (utils:498 shuffle_xor_f32)."""
    out = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.bfly.b32(
        out, txl.reinterpret("uint32", val), txl.uint32(lane_xor), txl.uint32(31), txl.uint32(0xFFFFFFFF)
    )
    return txl.reinterpret("float32", out)


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
    out = txl.local_scalar("float32")
    txl.ptx.mul.f32(out, a, b)
    return out


def float_to_ue8m0(value):
    """float_to_ue8m0_fast (utils:157), instruction for instruction.

    setp.le.f32 / mov.b32 / shr.b32 / and.b32 / setp.ne.u32 / selp.u32 /
    setp.eq.u32 / setp.le.u32 / and.pred / @p mov (selp) / add.u32 /
    setp.gt.u32 / selp.u32 x2 -- all explicit so the FTZ-flagged host
    compiler cannot reinterpret the float compare or the selects.
    """
    bits = txl.reinterpret("uint32", value)
    exp = txl.bitwise_and(txl.shift_right(bits, txl.uint32(23)), txl.uint32(255))
    mant = txl.bitwise_and(bits, txl.uint32(0x7FFFFF))
    p_zero = txl.local_scalar("uint32")
    txl.ptx.setp.le.f32(p_zero, value, txl.float32(0.0))
    p_has_mant = txl.local_scalar("uint32")
    txl.ptx.setp.ne.u32(p_has_mant, mant, txl.uint32(0))
    bump = txl.local_scalar("uint32")
    txl.ptx.selp.u32(bump, txl.uint32(1), txl.uint32(0), txl.ptx.pred(p_has_mant))
    p_exp_zero = txl.local_scalar("uint32")
    txl.ptx.setp.eq.u32(p_exp_zero, exp, txl.uint32(0))
    p_tiny = txl.local_scalar("uint32")
    txl.ptx.setp.le.u32(p_tiny, mant, txl.uint32(0x400000))
    txl.ptx.and_.pred(p_tiny, txl.ptx.pred(p_exp_zero), txl.ptx.pred(p_tiny))
    # @p_tiny mov.u32 bump, 0  (selp form)
    txl.ptx.selp.u32(bump, txl.uint32(0), bump, txl.ptx.pred(p_tiny))
    result = exp + bump
    p_ovf = txl.local_scalar("uint32")
    txl.ptx.setp.gt.u32(p_ovf, result, txl.uint32(254))
    out = txl.local_scalar("uint32")
    txl.ptx.selp.u32(out, txl.uint32(254), result, txl.ptx.pred(p_ovf))
    txl.ptx.selp.u32(out, txl.uint32(0), out, txl.ptx.pred(p_zero))
    return out


def ue8m0_to_inv_scale(ue8m0_val):
    """ue8m0_to_inv_scale_fast (utils:207), instruction for instruction.

    setp.eq.u32 / sub.s32 / max.s32 / shl.b32 / mov.b32 / @p_zero mov (selp).
    """
    p_zero = txl.local_scalar("uint32")
    txl.ptx.setp.eq.u32(p_zero, ue8m0_val, txl.uint32(0))
    new_exp = txl.max(txl.int32(254) - txl.cast(ue8m0_val, "int32"), txl.int32(0))
    inv = txl.reinterpret("float32", txl.shift_left(txl.cast(new_exp, "uint32"), txl.uint32(23)))
    out = txl.local_scalar("float32")
    txl.ptx.selp.f32(out, txl.float32(0.0), inv, txl.ptx.pred(p_zero))
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
    pair = txl.local_scalar("uint16")
    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(pair, hi, lo)
    return txl.cast(pair, "uint32")


def pack_fp8x8_to_u64(b01, b23, b45, b67):
    """Gather 4 x (2 FP8 bytes in low u16) into one uint64 (utils:326)."""
    lo = txl.bitwise_or(
        txl.bitwise_and(b01, txl.uint32(0xFFFF)),
        txl.shift_left(txl.bitwise_and(b23, txl.uint32(0xFFFF)), txl.uint32(16)),
    )
    hi = txl.bitwise_or(
        txl.bitwise_and(b45, txl.uint32(0xFFFF)),
        txl.shift_left(txl.bitwise_and(b67, txl.uint32(0xFFFF)), txl.uint32(16)),
    )
    out = txl.local_scalar("uint64")
    txl.ptx.mov.b64(out, lo, hi)
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
        txl.truncmod(col, txl.int32(4))
        + txl.truncdiv(col, txl.int32(4)) * 512
        + txl.truncmod(row, txl.int32(32)) * 16
        + txl.truncdiv(txl.truncmod(row, txl.int32(128)), txl.int32(32)) * 4
        + txl.truncdiv(row, txl.int32(128)) * (128 * padded_cols)
    )


def sf_offset_8x4(row, col, padded_cols):
    """Swizzled 8x4 SF byte offset ([mTiles, kTiles, 8, 4] tiles of 32)."""
    num_k_tiles = (padded_cols + 3) // 4
    return (
        txl.truncdiv(row, txl.int32(8)) * (num_k_tiles * 32)
        + txl.truncdiv(col, txl.int32(4)) * 32
        + txl.truncmod(row, txl.int32(8)) * 4
        + txl.truncmod(col, txl.int32(4))
    )


# ---------------------------------------------------------------------------
# MXFP4/NVFP4 helpers: rcp, scaled unpack, e2m1 packing
# ---------------------------------------------------------------------------


def rcp_approx_ftz(a):
    """``rcp.approx.ftz.f32`` (fp4_common.py:394)."""
    out = txl.local_scalar("float32")
    txl.ptx.rcp.approx.ftz.f32(out, a)
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
    bytes_ = txl.alloc_local([4], "uint8")
    for i in range(4):
        txl.ptx.cvt.rn.satfinite.e2m1x2.f32(bytes_[i], vals[2 * i + 1], vals[2 * i])
    w0 = txl.cast(bytes_[0], "uint16") | (txl.cast(bytes_[1], "uint16") << txl.uint16(8))
    w1 = txl.cast(bytes_[2], "uint16") | (txl.cast(bytes_[3], "uint16") << txl.uint16(8))
    out = txl.local_scalar("uint32")
    txl.ptx.mov.b32(out, w0, w1)
    return out


def pack_u32x2_to_u64(lo, hi):
    """(u64(hi) << 32) | u64(lo): plain u64 shift/or (utils:996-997)."""
    return txl.bitwise_or(txl.shift_left(txl.cast(hi, "uint64"), txl.uint64(32)), txl.cast(lo, "uint64"))


# ---------------------------------------------------------------------------
# NVFP4 helpers: E4M3 scale factors, output scale, SwiGLU fusion
# ---------------------------------------------------------------------------


def add_f32(a, b):
    """``add.f32`` (non-ftz, as the source asm/lowering)."""
    out = txl.local_scalar("float32")
    txl.ptx.add.f32(out, a, b)
    return out


def div_rn_f32(a, b):
    """``div.rn.f32`` (fp4_common.py fdiv_rn; the silu path is not fast-div)."""
    out = txl.local_scalar("float32")
    txl.ptx.div.rn.f32(out, a, b)
    return out


def ex2_approx_ftz(a):
    """``ex2.approx.ftz.f32``."""
    out = txl.local_scalar("float32")
    txl.ptx.ex2.approx.ftz.f32(out, a)
    return out


def silu_f32(g):
    """_silu_f32 (utils:1731): mul.f32 by -log2e (folds -g) + ex2.approx.ftz.f32
    + add.f32(+1.0) + div.rn.f32."""
    e = ex2_approx_ftz(mul_f32(g, txl.float32(-1.4426950408889634)))
    return div_rn_f32(g, add_f32(e, txl.float32(1.0)))


def cvt_f32x2_to_packed(lo, hi, dtype):
    """cvt_f32x2_to_half2/bfloat2 (fp4_common:880/:910): 2x scalar cvt +
    mov.b32 {h0,h1}."""
    h0 = txl.local_scalar("uint16")
    h1 = txl.local_scalar("uint16")
    if dtype == "float16":
        txl.ptx.cvt.rn.f16.f32(h0, lo)
        txl.ptx.cvt.rn.f16.f32(h1, hi)
    else:
        txl.ptx.cvt.rn.bf16.f32(h0, lo)
        txl.ptx.cvt.rn.bf16.f32(h1, hi)
    out = txl.local_scalar("uint32")
    txl.ptx.mov.b32(out, h0, h1)
    return out


def silu_and_mul_pair(gate, up, dtype):
    """_silu_and_mul_half2/_bfloat2 (utils:1740/:1752): unpack both pairs with
    scale 1.0, silu(g)*u per scalar in f32, repack."""
    g0 = mul_f32(unpack_lo_f32(gate, dtype), txl.float32(1.0))
    g1 = mul_f32(unpack_hi_f32(gate, dtype), txl.float32(1.0))
    u0 = mul_f32(unpack_lo_f32(up, dtype), txl.float32(1.0))
    u1 = mul_f32(unpack_hi_f32(up, dtype), txl.float32(1.0))
    a0 = mul_f32(silu_f32(g0), u0)
    a1 = mul_f32(silu_f32(g1), u1)
    return cvt_f32x2_to_packed(a0, a1, dtype)


def cvt_f32_to_e4m3(a):
    """cvt_f32_to_e4m3 (fp4_common.py:811): mov.f32 0 +
    cvt.rn.satfinite.e4m3x2.f32 + cvt.u32.u16."""
    pair = txl.local_scalar("uint16")
    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(pair, txl.float32(0.0), a)
    return txl.cast(pair, "uint32")


def nvfp4_compute_output_scale(sf_u32, global_scale):
    """nvfp4_compute_output_scale (fp4_common.py:973), instruction for
    instruction: decode the E4M3 SF through the f16x2 path, then
    rcp(SF_f32 * rcp(global_scale)), with the zero-SF select."""
    pair16 = txl.local_scalar("uint16")
    txl.ptx.cvt.u16.u32(pair16, sf_u32)
    h2 = txl.local_scalar("uint32")
    txl.ptx.cvt.rn.f16x2.e4m3x2(h2, pair16)
    sf_f32 = txl.cast(
        txl.reinterpret("float16", txl.cast(txl.bitwise_and(h2, txl.uint32(0xFFFF)), "uint16")), "float32"
    )
    product = mul_f32(sf_f32, rcp_approx_ftz(global_scale))
    result = rcp_approx_ftz(product)
    p_zero = txl.local_scalar("uint32")
    txl.ptx.setp.eq.f32(p_zero, sf_f32, txl.float32(0.0))
    out = txl.local_scalar("float32")
    txl.ptx.selp.f32(out, txl.float32(0.0), result, txl.ptx.pred(p_zero))
    return out


def opaque_i32(x):
    """Identity ``mov.s32`` that keeps a loop stride opaque to the host
    compiler's strength reduction, so the generated loop recomputes addresses
    per iteration the way the source's own binary does (avoids the heavy
    up-front pointer-induction chains nvcc otherwise builds in the prologue).
    Purely a loop-bookkeeping device; the value is unchanged."""
    out = txl.local_scalar("int32")
    txl.ptx.mov.s32(out, x)
    return out


# ---------------------------------------------------------------------------
# Scalar loads and reductions (per-token kernel)
# ---------------------------------------------------------------------------


def ld_global_f32(buf, idx):
    """Plain ``ld.global.f32`` scalar load (non-nc), as the source emits."""
    out = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(out, txl.address_of(buf[idx]))
    return out


def warp_reduce_max(val):
    """warp_reduce (fp4_common.py:1356): 5 butterfly rounds, offsets 1..16."""
    for i in range(5):
        val = fmax_f32(val, shfl_xor_f32(val, 1 << i))
    return val
