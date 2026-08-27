# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400e330fb2debe0bf8730d9424a1d37927f), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer CuTe-DSL fused add/RMSNorm with packed FP4 quantization.

The source implementation is ``AddRMSNormFP4QuantKernel`` in
``flashinfer/cute_dsl/add_rmsnorm_fp4quant.py`` with inline PTX helpers from
``flashinfer/cute_dsl/fp4_common.py``. The public entry is
``flashinfer.norm.add_rmsnorm_fp4quant``.
"""

import functools
from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.flashinfer.norm._kern_helpers import _cvt_from_f32, _cvt_pair_from_f32
from tirx_kernels.flashinfer.norm.rmsnorm_fp4quant import (
    _add_f32,
    _add_s32,
    _butterfly_sum_f32,
    _ceil_div,
    _cluster_mbarrier_wait,
    _cvt_f32_to_e4m3,
    _cvt_f32_to_ue8m0,
    _cvt_to_f32,
    _div_rn_f32,
    _fma_rn_f32,
    _load_16_input_words,
    _mapa_u32,
    _minimum_f32,
    _mul_f32,
    _multiply_8_pairs,
    _pair_max_to_f32,
    _quantize_and_pack_16,
    _rcp_approx_ftz,
    _rsqrt_approx_ftz,
    _scale_offset,
    _store_global_u64,
    _threads_per_row,
    _ue8m0_to_output_scale,
    _widen_and_scale_16,
)
from tirx_kernels.flashinfer.utils.fp_quant import absmax_8, hmax2
from tirx_kernels.runner import bench

KERNEL_META = {
    "name": "flashinfer_add_rmsnorm_fp4quant",
    "category": "flashinfer",
    "compute_capability": 10,
}

_DTYPES = ("float16", "bfloat16")
_DEFAULT_EPS = 1e-6
_OPTIN_SMEM_BYTES = 232448


def _abs_f32(value):
    out = K.alloc_local((1,), "float32")
    K.ptx.abs.f32(out[0], value)
    return out[0]


def _maximum_f32(lhs, rhs):
    out = K.alloc_local((1,), "float32")
    K.ptx.max.f32(out[0], lhs, rhs)
    return out[0]


def _e4m3_to_f32_rcp_exact(value):
    pair = K.alloc_local((1,), "uint16")
    halves = K.alloc_local((1,), "uint32")
    half_bits = K.alloc_local((2,), "uint16")
    decoded = K.alloc_local((1,), "float32")
    predicate_zero = K.local_scalar("uint32")
    out = K.alloc_local((1,), "float32")
    K.ptx.cvt.u16.u32(pair[0], value)
    K.ptx.cvt.rn.f16x2.e4m3x2(halves[0], pair[0])
    K.ptx.mov.b32(half_bits[0], half_bits[1], halves[0])
    K.ptx.cvt.f32.f16(decoded[0], half_bits[0])
    reciprocal: K.float32 = _rcp_approx_ftz(decoded[0])
    K.ptx.setp.eq.f32(predicate_zero, decoded[0], K.float32(0.0))
    K.ptx.selp.f32(out[0], K.float32(0.0), reciprocal, K.ptx.pred(predicate_zero))
    return out[0]


def _cluster_bf16_pair_max_to_f32(word):
    low = K.alloc_local((1,), "uint32")
    high = K.alloc_local((1,), "uint32")
    low_shifted = K.alloc_local((1,), "uint32")
    high_shifted = K.alloc_local((1,), "uint32")
    values = K.alloc_local((2,), "float32")
    out = K.alloc_local((1,), "float32")
    K.ptx.and_.b32(low[0], word, K.uint32(0xFFFF))
    K.ptx.shr.b32(high[0], word, K.uint32(16))
    K.ptx.shl.b32(low_shifted[0], low[0], K.uint32(16))
    K.ptx.shl.b32(high_shifted[0], high[0], K.uint32(16))
    K.ptx.mov.b32(values[0], low_shifted[0])
    K.ptx.mov.b32(values[1], high_shifted[0])
    K.ptx.max.f32(out[0], values[0], values[1])
    return out[0]


def _cluster_bf16_widen_and_scale_16(words, scale):
    values = K.alloc_local((16,), "float32")
    for pair in range(8):
        low = K.alloc_local((1,), "uint32")
        high = K.alloc_local((1,), "uint32")
        low_shifted = K.alloc_local((1,), "uint32")
        high_shifted = K.alloc_local((1,), "uint32")
        widened = K.alloc_local((2,), "float32")
        K.ptx.and_.b32(low[0], words[pair], K.uint32(0xFFFF))
        K.ptx.shr.b32(high[0], words[pair], K.uint32(16))
        K.ptx.shl.b32(low_shifted[0], low[0], K.uint32(16))
        K.ptx.shl.b32(high_shifted[0], high[0], K.uint32(16))
        K.ptx.mov.b32(widened[0], low_shifted[0])
        K.ptx.mov.b32(widened[1], high_shifted[0])
        K.ptx.mul.f32(values[pair * 2], widened[0], scale)
        K.ptx.mul.f32(values[pair * 2 + 1], widened[1], scale)
    return values


def _pack_input_pair_scalar(low, high, input_dtype: str):
    low_bits: K.uint16 = _cvt_from_f32(low, input_dtype)
    high_bits: K.uint16 = _cvt_from_f32(high, input_dtype)
    word = K.alloc_local((1,), "uint32")
    K.ptx.mov.b32(word[0], low_bits, high_bits)
    return word[0]


def _store_y_norm_16(y_norm, values, actual_row, block_start, H: int, input_dtype: str):
    for group in range(4):
        low_word: K.uint32 = _pack_input_pair_scalar(
            values[group * 4], values[group * 4 + 1], input_dtype
        )
        high_word: K.uint32 = _pack_input_pair_scalar(
            values[group * 4 + 2], values[group * 4 + 3], input_dtype
        )
        low64 = K.alloc_local((1,), "uint64")
        high64 = K.alloc_local((1,), "uint64")
        packed = K.alloc_local((1,), "uint64")
        K.ptx.cvt.u64.u32(low64[0], low_word)
        K.ptx.cvt.u64.u32(high64[0], high_word)
        K.ptx.shl.b64(high64[0], high64[0], K.uint32(32))
        K.ptx.or_.b64(packed[0], high64[0], low64[0])
        offset: K.int64 = (
            K.cast(actual_row, "int64") * K.int64(H) + K.cast(block_start, "int64") + group * 4
        )
        _store_global_u64(K.address_of(y_norm[offset]), packed[0])


def _process_add_scale_block(
    shared_raw,
    residual,
    weight,
    y,
    scales,
    scales_unswizzled,
    y_norm,
    actual_row,
    row_in_cta,
    sf_index,
    rstd,
    global_scale_value,
    fp4_max_rcp,
    *,
    H: K.constexpr,
    block_size: K.constexpr,
    scale_format: K.constexpr,
    swizzled: K.constexpr,
    output_both_sf_layouts: K.constexpr,
    output_norm: K.constexpr,
    input_dtype: K.constexpr,
    cluster_n: K.constexpr,
    cols: K.constexpr,
    tile_bytes: K.constexpr,
):
    num_scale_blocks = H // block_size
    with K.If(sf_index < num_scale_blocks), K.Then():
        block_start: K.int32 = sf_index * block_size

        if cluster_n == 1:
            h0 = K.alloc_local((16,), "float32")
            w0 = K.alloc_local((16,), "float32")
            values0 = K.alloc_local((16,), "float32")
            for value in range(16):
                h_bits = K.alloc_local((1,), "uint16")
                K.ptx.ld.shared.b16(
                    h_bits[0],
                    shared_raw.ptr_to(
                        [tile_bytes * 3 + (row_in_cta * cols + block_start + value) * 2]
                    ),
                )
                K.assign(h0[value], _cvt_to_f32(h_bits[0], input_dtype))
            for value in range(16):
                w_bits = K.alloc_local((1,), "uint16")
                K.ptx.ld.shared.b16(
                    w_bits[0],
                    shared_raw.ptr_to(
                        [tile_bytes * 2 + (row_in_cta * cols + block_start + value) * 2]
                    ),
                )
                K.assign(w0[value], _cvt_to_f32(w_bits[0], input_dtype))
            K.assign(values0[0], _mul_f32(_mul_f32(h0[0], rstd), w0[0]))
            max0: K.float32 = _abs_f32(values0[0])
            for value in range(15):
                index = value + 1
                K.assign(values0[index], _mul_f32(_mul_f32(h0[index], rstd), w0[index]))
                max0 = _maximum_f32(max0, _abs_f32(values0[index]))
            if block_size == 32:
                h1 = K.alloc_local((16,), "float32")
                w1 = K.alloc_local((16,), "float32")
                values1 = K.alloc_local((16,), "float32")
                for value in range(16):
                    h_bits = K.alloc_local((1,), "uint16")
                    K.ptx.ld.shared.b16(
                        h_bits[0],
                        shared_raw.ptr_to(
                            [tile_bytes * 3 + (row_in_cta * cols + block_start + 16 + value) * 2]
                        ),
                    )
                    K.assign(h1[value], _cvt_to_f32(h_bits[0], input_dtype))
                for value in range(16):
                    w_bits = K.alloc_local((1,), "uint16")
                    K.ptx.ld.shared.b16(
                        w_bits[0],
                        shared_raw.ptr_to(
                            [tile_bytes * 2 + (row_in_cta * cols + block_start + 16 + value) * 2]
                        ),
                    )
                    K.assign(w1[value], _cvt_to_f32(w_bits[0], input_dtype))
                K.assign(values1[0], _mul_f32(_mul_f32(h1[0], rstd), w1[0]))
                max1: K.float32 = _abs_f32(values1[0])
                for value in range(15):
                    index = value + 1
                    K.assign(values1[index], _mul_f32(_mul_f32(h1[index], rstd), w1[index]))
                    max1 = _maximum_f32(max1, _abs_f32(values1[index]))
                max_abs: K.float32 = _maximum_f32(max0, max1)
            else:
                max_abs = max0
        else:
            residual_base: K.int64 = K.cast(actual_row, "int64") * K.int64(H) + K.cast(
                block_start, "int64"
            )
            h0 = _load_16_input_words(residual, residual_base)
            w0 = _load_16_input_words(weight, block_start)
            if block_size == 32:
                h1 = _load_16_input_words(residual, residual_base + 16)
                w1 = _load_16_input_words(weight, block_start + 16)
            products0 = _multiply_8_pairs(h0, w0, input_dtype)
            if block_size == 32:
                products1 = _multiply_8_pairs(h1, w1, input_dtype)
            max_pair = absmax_8(products0, input_dtype)
            if block_size == 32:
                max_pair = hmax2(max_pair, absmax_8(products1, input_dtype), input_dtype)
            if input_dtype == "bfloat16":
                max_xw: K.float32 = _cluster_bf16_pair_max_to_f32(max_pair)
                values0 = _cluster_bf16_widen_and_scale_16(products0, rstd)
                if block_size == 32:
                    values1 = _cluster_bf16_widen_and_scale_16(products1, rstd)
            else:
                max_xw = _pair_max_to_f32(max_pair, input_dtype)
                values0 = _widen_and_scale_16(products0, rstd, input_dtype)
                if block_size == 32:
                    values1 = _widen_and_scale_16(products1, rstd, input_dtype)
            max_abs = _mul_f32(max_xw, rstd)

        if block_size == 16 or scale_format == "e4m3":
            scale_value: K.float32 = _mul_f32(_mul_f32(global_scale_value, max_abs), fp4_max_rcp)
            scale_value = _minimum_f32(scale_value, K.float32(448.0))
            scale_word: K.uint32 = _cvt_f32_to_e4m3(scale_value)
            output_scale: K.float32 = _mul_f32(
                _e4m3_to_f32_rcp_exact(scale_word), global_scale_value
            )
        else:
            scale_word = _cvt_f32_to_ue8m0(_mul_f32(max_abs, fp4_max_rcp))
            output_scale = _ue8m0_to_output_scale(scale_word)

        primary_offset = _scale_offset(
            actual_row, sf_index, H, block_size, swizzled or output_both_sf_layouts
        )
        K.ptx.st.global_.b8(K.address_of(scales[primary_offset]), K.cast(scale_word, "uint8"))
        if output_both_sf_layouts:
            linear_offset: K.int64 = K.cast(actual_row, "int64") * K.int64(
                num_scale_blocks
            ) + K.cast(sf_index, "int64")
            K.ptx.st.global_.b8(
                K.address_of(scales_unswizzled[linear_offset]), K.cast(scale_word, "uint8")
            )

        packed0: K.uint64 = _quantize_and_pack_16(values0, output_scale)
        if block_size == 32:
            packed1: K.uint64 = _quantize_and_pack_16(values1, output_scale)
        y_offset: K.int64 = K.cast(actual_row, "int64") * K.int64(H // 2) + K.cast(
            block_start // 2, "int64"
        )
        _store_global_u64(K.address_of(y[y_offset]), packed0)
        if block_size == 32:
            _store_global_u64(K.address_of(y[y_offset + 8]), packed1)

        if output_norm:
            _store_y_norm_16(y_norm, values0, actual_row, block_start, H, input_dtype)
            if block_size == 32:
                _store_y_norm_16(y_norm, values1, actual_row, block_start + 16, H, input_dtype)


def _dtype_code(dtype: str) -> str:
    return {"float16": "fp16", "bfloat16": "bf16"}[dtype]


def _eps_code(eps: float) -> str:
    return {1e-4: "1e4", 1e-5: "1e5", 1e-6: "1e6"}[eps]


def _derived_config(H: int, cluster_n: int) -> dict[str, int]:
    H_per_cta = H // cluster_n
    tpr = _threads_per_row(H_per_cta)
    threads = 128 if H_per_cta <= 16384 else 256
    rows = threads // tpr
    warps_per_row = max(tpr // 32, 1)
    vec_blocks = max(1, _ceil_div(H_per_cta // 8, tpr))
    cols = 8 * vec_blocks * tpr
    tile_bytes = rows * cols * 2
    return {
        "H_per_cta": H_per_cta,
        "tpr": tpr,
        "threads": threads,
        "rows": rows,
        "warps_per_row": warps_per_row,
        "vec_blocks": vec_blocks,
        "cols": cols,
        "tile_bytes": tile_bytes,
    }


def _estimate_smem(H: int, cluster_n: int) -> int:
    source = _derived_config(H, cluster_n)
    reduction = source["rows"] * source["warps_per_row"] * 4
    if cluster_n == 1:
        return 4 * source["tile_bytes"] + reduction
    return 2 * source["tile_bytes"] + reduction * cluster_n + 8


def _source_config(H: int) -> dict[str, int]:
    cluster_n = 16
    for candidate in (1, 2, 4, 8, 16):
        if H % candidate == 0 and _estimate_smem(H, candidate) <= _OPTIN_SMEM_BYTES:
            cluster_n = candidate
            break
    source = _derived_config(H, cluster_n)
    source["cluster_n"] = cluster_n
    source["smem_bytes"] = _estimate_smem(H, cluster_n)
    return source


def _cfg(
    case: str,
    dtype: str,
    H: int,
    *,
    M: int | None = None,
    B: int | None = None,
    S: int | None = None,
    block_size: int = 16,
    scale_format: str = "e4m3",
    swizzled: bool = False,
    output_both_sf_layouts: bool = False,
    output_norm: bool = False,
    enable_pdl: bool = False,
    eps: float = _DEFAULT_EPS,
    global_scale_mode: str = "none",
    allocation: str = "preallocated",
    data_mode: str = "random",
) -> dict[str, Any]:
    if M is None:
        if B is None or S is None:
            raise ValueError("either M or both B/S must be provided")
        M = B * S
        shape = f"b{B}_s{S}"
        input_ndim = 3
    else:
        if B is not None or S is not None:
            raise ValueError("M and B/S are mutually exclusive")
        shape = f"m{M}"
        input_ndim = 2
    label = (
        f"{case}_{_dtype_code(dtype)}_{shape}_h{H}_b{block_size}_{scale_format}_"
        f"sw{int(swizzled)}_both{int(output_both_sf_layouts)}_"
        f"yn{int(output_norm)}_pdl{int(enable_pdl)}_eps{_eps_code(eps)}_"
        f"gs{global_scale_mode}_{allocation}_{data_mode}"
    )
    return {
        "label": label,
        "case": case,
        "input_dtype": dtype,
        "input_ndim": input_ndim,
        "M": M,
        "B": B,
        "S": S,
        "H": H,
        "block_size": block_size,
        "scale_format": scale_format,
        "swizzled": swizzled,
        "output_both_sf_layouts": output_both_sf_layouts,
        "output_norm": output_norm,
        "enable_pdl": enable_pdl,
        "eps": eps,
        "global_scale_mode": global_scale_mode,
        "allocation": allocation,
        "data_mode": data_mode,
    }


_NV2D_CONFIGS = [
    _cfg("nv2d", dtype, H, M=M, eps=eps)
    for M in (1, 4, 16, 32, 7, 13, 128, 512, 1000, 8192, 16384)
    for H in (64, 128, 256, 512, 1024, 1536, 2048, 4096, 8192)
    for dtype in _DTYPES
    for eps in (1e-5, 1e-6)
]

_NV3D_CONFIGS = [
    _cfg("nv3d", dtype, H, B=B, S=S, eps=1e-5)
    for B in (1, 4, 3, 7, 128)
    for S in (16, 64, 128, 37)
    for H in (128, 256, 1536, 4096, 8192)
    for dtype in _DTYPES
]

_LARGE_BATCH_CONFIGS = [
    _cfg("large_batch", "float16", H, M=M) for M, H in ((512, 4096), (1024, 4096))
]

_MX_BASIC_CONFIGS = [
    _cfg("mx_basic", dtype, H, M=M, block_size=32, scale_format="ue8m0")
    for M in (1, 4, 16, 7, 128, 8192)
    for H in (128, 256, 512, 1536, 2048, 4096)
    for dtype in _DTYPES
]

_CROSS_CONFIGS = [
    *[
        _cfg("fused_vs_separate", "float16", H, M=M)
        for M in (4, 16, 128, 512)
        for H in (256, 512, 1024, 4096, 8192)
    ],
    *[
        _cfg("nv_matches_separate", dtype, H, M=M)
        for M in (1, 4, 16, 128)
        for H in (64, 256, 512, 1024, 2048, 4096)
        for dtype in _DTYPES
    ],
    *[
        _cfg("mx_matches_separate", dtype, H, M=M, block_size=32, scale_format="ue8m0")
        for M in (1, 4, 16, 128)
        for H in (128, 256, 512, 1024, 2048, 4096)
        for dtype in _DTYPES
    ],
    *[
        _cfg("global_scale_consistency", "float16", H, M=M, global_scale_mode="nonunit")
        for M in (1, 16, 64)
        for H in (256, 1024, 4096)
    ],
]

_LARGE_H_CONFIGS = [
    *[
        _cfg("large_h_nv", dtype, H, M=M)
        for M in (1, 16, 128, 1024)
        for H in (16384, 32768)
        for dtype in _DTYPES
    ],
    *[
        _cfg("large_h_mx", dtype, H, M=M, block_size=32, scale_format="ue8m0")
        for M in (1, 16, 128, 1024)
        for H in (16384, 32768)
        for dtype in _DTYPES
    ],
]

_SWIZZLED_CONFIGS = [
    *[
        _cfg("swizzled_nv", dtype, H, M=M, swizzled=True)
        for M in (1, 16, 128, 256)
        for H in (512, 1024, 2048, 4096)
        for dtype in _DTYPES
    ],
    *[
        _cfg("swizzled_mx", dtype, H, M=M, block_size=32, scale_format="ue8m0", swizzled=True)
        for M in (1, 16, 128, 256)
        for H in (512, 1024, 2048, 4096)
        for dtype in _DTYPES
    ],
]

_ALLOCATION_CONFIGS = [
    *[
        _cfg("auto2d_nv", dtype, H, M=M, allocation="auto")
        for M in (1, 16, 128)
        for H in (256, 1024, 4096)
        for dtype in _DTYPES
    ],
    *[
        _cfg("auto3d_nv", "float16", H, B=B, S=S, allocation="auto")
        for B in (1, 4, 16)
        for S in (16, 64)
        for H in (256, 1024)
    ],
    *[
        _cfg("auto_mx", "float16", H, M=M, block_size=32, scale_format="ue8m0", allocation="auto")
        for M in (1, 16, 128)
        for H in (256, 1024)
    ],
    *[
        _cfg("auto_swizzled", "float16", H, M=M, swizzled=True, allocation="auto")
        for M in (16, 128)
        for H in (512, 1024)
    ],
    *[
        _cfg("auto_matches_preallocated", "float16", H, M=M, allocation="compare")
        for M in (16, 128)
        for H in (512, 1024)
    ],
]

_RESIDUAL_CONFIGS = [
    *[
        _cfg("residual2d", dtype, H, M=M)
        for M in (1, 4, 16, 128, 512)
        for H in (256, 512, 1024, 4096)
        for dtype in _DTYPES
    ],
    *[
        _cfg("residual3d", dtype, H, B=B, S=S)
        for B in (1, 4, 16)
        for S in (16, 64, 128)
        for H in (256, 1024, 4096)
        for dtype in _DTYPES
    ],
    *[
        _cfg("residual_mx", dtype, H, M=M, block_size=32, scale_format="ue8m0")
        for M in (1, 16, 128)
        for H in (256, 1024, 2048)
        for dtype in _DTYPES
    ],
    *[
        _cfg("residual_large_h", dtype, H, M=M)
        for M in (1, 16, 128)
        for H in (16384, 32768)
        for dtype in _DTYPES
    ],
    *[_cfg("residual_used_for_norm", "float16", H, M=M) for M in (16, 128) for H in (512, 1024)],
    *[_cfg("residual_preallocated", "float16", H, M=M) for M in (16, 128) for H in (512, 1024)],
    *[
        _cfg("residual_swizzled", "float16", H, M=M, swizzled=True)
        for M in (16, 128)
        for H in (512, 1024)
    ],
    _cfg("residual_not_aliased", "float16", 512, M=16),
]

_BOTH_CONFIGS = [
    *[
        _cfg("both_nv", dtype, H, M=M, output_both_sf_layouts=True)
        for M in (1, 16, 128, 256)
        for H in (256, 512, 1024, 4096)
        for dtype in _DTYPES
    ],
    *[
        _cfg(
            "both_mx",
            dtype,
            H,
            M=M,
            block_size=32,
            scale_format="ue8m0",
            output_both_sf_layouts=True,
        )
        for M in (1, 16, 128, 256)
        for H in (256, 512, 1024, 4096)
        for dtype in _DTYPES
    ],
    *[
        _cfg("both_consistency", "float16", H, M=M, output_both_sf_layouts=True)
        for M in (1, 16, 128)
        for H in (512, 1024, 4096)
    ],
    *[
        _cfg("both3d", "float16", H, B=B, S=S, output_both_sf_layouts=True)
        for B in (1, 4, 8)
        for S in (16, 64, 128)
        for H in (256, 1024)
    ],
    *[
        _cfg(
            "both_global",
            "float16",
            H,
            M=M,
            output_both_sf_layouts=True,
            global_scale_mode="nonunit",
        )
        for M in (16, 128)
        for H in (512, 1024)
    ],
    *[
        _cfg("both_preallocated", "float16", H, M=M, output_both_sf_layouts=True)
        for M in (16, 128)
        for H in (512, 1024)
    ],
    *[
        _cfg("both_large_h", dtype, H, M=M, output_both_sf_layouts=True)
        for M in (1, 16, 128)
        for H in (16384, 32768)
        for dtype in _DTYPES
    ],
    *[
        _cfg("both_residual", "float16", H, M=M, output_both_sf_layouts=True)
        for M in (16, 128)
        for H in (512, 1024)
    ],
    _cfg("both_ignores_swizzle", "float16", 1024, M=64, swizzled=True, output_both_sf_layouts=True),
]

_YOUT_CONFIGS = [
    *[
        _cfg("yout_reference", dtype, H, M=M, output_norm=True)
        for M in (1, 64)
        for H in (256, 4096)
        for dtype in _DTYPES
    ],
    *[
        _cfg("yout_quant_unchanged", dtype, H, M=M, output_norm=True, global_scale_mode="nonunit")
        for M in (1, 64)
        for H in (256, 2048)
        for dtype in _DTYPES
    ],
    *[
        _cfg("yout_both", "bfloat16", H, M=M, output_both_sf_layouts=True, output_norm=True)
        for M in (1, 16, 128)
        for H in (512, 2048)
    ],
    *[
        _cfg("yout3d", "bfloat16", H, B=B, S=37, output_norm=True)
        for B in (1, 4)
        for H in (256, 2048)
    ],
    *[
        _cfg(
            "yout_not_global_scaled",
            dtype,
            2048,
            M=M,
            output_norm=True,
            global_scale_mode="nonunit",
        )
        for M in (1, 64)
        for dtype in _DTYPES
    ],
    *[
        _cfg("yout_mx", dtype, 2048, M=M, block_size=32, scale_format="ue8m0", output_norm=True)
        for M in (1, 64)
        for dtype in _DTYPES
    ],
]

_UPSTREAM_CONFIGS = [
    *_NV2D_CONFIGS,
    *_NV3D_CONFIGS,
    *_LARGE_BATCH_CONFIGS,
    *_MX_BASIC_CONFIGS,
    *_CROSS_CONFIGS,
    *_LARGE_H_CONFIGS,
    *_SWIZZLED_CONFIGS,
    *_ALLOCATION_CONFIGS,
    *_RESIDUAL_CONFIGS,
    *_BOTH_CONFIGS,
    *_YOUT_CONFIGS,
]

_STRUCTURE_CONFIGS = [
    _cfg("guard_subwarp_row_tail", "float16", 64, M=17),
    _cfg("guard_nv_column_tail", "bfloat16", 80, M=129, swizzled=True),
    _cfg("guard_mx_column_tail", "float16", 160, M=3, block_size=32, scale_format="ue8m0"),
    *[
        _cfg("guard_threshold", "bfloat16", H, M=3)
        for H in (3072, 3088, 6144, 6160, 14336, 16384, 16400)
    ],
    *[
        _cfg(f"guard_cluster{cluster}", "bfloat16", H, M=1)
        for cluster, H in ((2, 32768), (4, 131072), (8, 262144), (16, 524288))
    ],
    _cfg("guard_pdl", "bfloat16", 4096, M=3, enable_pdl=True),
    _cfg(
        "guard_block32_e4m3",
        "float16",
        4096,
        M=3,
        block_size=32,
        scale_format="e4m3",
        global_scale_mode="nonunit",
    ),
    _cfg("guard_block16_requested_ue8", "float16", 4096, M=3, scale_format="ue8m0"),
    _cfg("guard_zero_data", "bfloat16", 4096, M=3, data_mode="zero"),
    _cfg("guard_zero_global", "float16", 4096, M=3, global_scale_mode="zero"),
    _cfg("guard_both_column_tail", "bfloat16", 80, M=129, output_both_sf_layouts=True),
    _cfg("guard_cluster16_both", "bfloat16", 524288, M=1, output_both_sf_layouts=True),
    _cfg("guard_cluster16_yout", "bfloat16", 524288, M=1, output_norm=True),
    _cfg(
        "guard_pdl_both_yout",
        "bfloat16",
        4096,
        M=3,
        output_both_sf_layouts=True,
        output_norm=True,
        enable_pdl=True,
    ),
]

CONFIGS = [*_UPSTREAM_CONFIGS, *_STRUCTURE_CONFIGS]

assert len(_NV2D_CONFIGS) == 396
assert len(_NV3D_CONFIGS) == 200
assert len(_LARGE_BATCH_CONFIGS) == 2
assert len(_MX_BASIC_CONFIGS) == 72
assert len(_CROSS_CONFIGS) == 125
assert len(_LARGE_H_CONFIGS) == 32
assert len(_SWIZZLED_CONFIGS) == 64
assert len(_ALLOCATION_CONFIGS) == 44
assert len(_RESIDUAL_CONFIGS) == 137
assert len(_BOTH_CONFIGS) == 116
assert len(_YOUT_CONFIGS) == 34
assert len(_UPSTREAM_CONFIGS) == 1222
assert len(_STRUCTURE_CONFIGS) == 23
assert len(CONFIGS) == 1245
assert len({config["label"] for config in CONFIGS}) == len(CONFIGS)

BENCH_CONFIGS = [
    _cfg("bench_nv", "bfloat16", 4096, M=32),
    _cfg("bench_nv_large", "bfloat16", 8192, M=64),
    _cfg("bench_nv_global", "bfloat16", 4096, M=32, global_scale_mode="one"),
    _cfg("bench_nv_swizzled", "bfloat16", 4096, M=32, swizzled=True),
    _cfg("bench_mx", "bfloat16", 4096, M=32, block_size=32, scale_format="ue8m0"),
    _cfg("bench_nv_3d", "bfloat16", 128, B=32, S=32),
    _cfg("bench_nv_both", "bfloat16", 4096, M=32, output_both_sf_layouts=True),
    _cfg("bench_nv_both_large", "bfloat16", 8192, M=64, output_both_sf_layouts=True),
    _cfg(
        "bench_nv_both_global",
        "bfloat16",
        4096,
        M=32,
        output_both_sf_layouts=True,
        global_scale_mode="one",
    ),
    _cfg(
        "bench_mx_both",
        "bfloat16",
        4096,
        M=32,
        block_size=32,
        scale_format="ue8m0",
        output_both_sf_layouts=True,
    ),
]


def _validate(config: dict[str, Any]) -> None:
    if config["input_dtype"] not in _DTYPES:
        raise ValueError(f"unsupported input_dtype={config['input_dtype']!r}")
    if int(config["H"]) < 64 or int(config["H"]) % int(config["block_size"]):
        raise ValueError("H must be >=64 and divisible by block_size")
    if int(config["block_size"]) not in (16, 32):
        raise ValueError("block_size must be 16 or 32")
    if config["scale_format"] not in ("e4m3", "ue8m0"):
        raise ValueError("scale_format must be e4m3 or ue8m0")


def get_kernel(**config: Any):
    """Return one source-faithful Add/RMSNorm/FP4 specialization."""
    _validate(config)
    input_dtype = str(config["input_dtype"])
    H = int(config["H"])
    block_size = int(config["block_size"])
    scale_format = str(config["scale_format"])
    swizzled = bool(config["swizzled"])
    output_both_sf_layouts = bool(config["output_both_sf_layouts"])
    output_norm = bool(config["output_norm"])
    enable_pdl = bool(config["enable_pdl"])
    source = _source_config(H)
    cluster_n = int(source["cluster_n"])
    tpr = int(source["tpr"])
    threads = int(source["threads"])
    rows = int(source["rows"])
    warps_per_row = int(source["warps_per_row"])
    vec_blocks = int(source["vec_blocks"])
    cols = int(source["cols"])
    tile_bytes = int(source["tile_bytes"])
    smem_bytes = int(source["smem_bytes"])
    total_values = 8 * vec_blocks
    packed_pairs = 4 * vec_blocks
    reduce_base = (4 if cluster_n == 1 else 2) * tile_bytes
    reduce_count = rows * warps_per_row * cluster_n
    mbar_offset = reduce_base + reduce_count * 4
    expected_bytes = reduce_count * 4
    total_partials_per_row = warps_per_row * cluster_n
    full_columns = H == cluster_n * cols
    num_scale_blocks = H // block_size
    scale_blocks_per_thread = _ceil_div(num_scale_blocks, tpr)
    use_e4m3_scale = block_size == 16 or scale_format == "e4m3"
    row_lane_xors = tuple(lane_xor for lane_xor in (1, 2, 4, 8, 16) if lane_xor < min(tpr, 32))
    full_lane_xors = (1, 2, 4, 8, 16)

    def kernel_body(
        x,
        residual,
        weight,
        y,
        scales,
        scales_unswizzled,
        y_norm,
        global_scale,
        runtime_M,
        runtime_eps,
    ):
        # TIRX_TRANSCRIBE_START flashinfer_add_rmsnorm_fp4quant
        tid = K.thread_id()

        if enable_pdl:
            K.ptx.griddepcontrol.wait()

        if cluster_n > 1:
            block_x_raw, block_y_raw = K.cta_id(
                [K.cast(K.ceildiv(runtime_M, K.int32(rows)), "int32"), cluster_n]
            )
            _, cta_rank_raw = K.cta_id_in_cluster([1, cluster_n], preferred=[1, cluster_n])
            block_y: K.int32 = K.cast(block_y_raw, "int32")
            cta_rank: K.int32 = K.cast(cta_rank_raw, "int32")
        else:
            block_x_raw = K.cta_id([K.cast(K.ceildiv(runtime_M, K.int32(rows)), "int32")])
            block_y = K.int32(0)
            cta_rank = K.int32(0)

        fp4_max_rcp: K.float32 = _rcp_approx_ftz(K.float32(6.0))
        block_x: K.int32 = K.cast(block_x_raw, "int32")
        row_in_cta: K.int32 = tid // tpr
        thread_in_row: K.int32 = tid % tpr
        actual_row: K.int32 = block_x * rows + row_in_cta
        row_valid: K.bool = actual_row < runtime_M
        warp: K.int32 = tid // 32
        lane: K.int32 = tid % 32
        row_warp: K.int32 = warp // warps_per_row
        warp_in_row: K.int32 = warp % warps_per_row

        shared_raw = K.smem_pool().alloc((smem_bytes,), "uint8", align=1024)

        if cluster_n > 1:
            with K.If(tid == 0), K.Then():
                K.ptx.mbarrier.init.shared.b64(shared_raw.ptr_to([mbar_offset]), K.uint32(1))
            K.ptx.fence.mbarrier_init.release.cluster()
            K.ptx.barrier.cluster.arrive.relaxed()
            K.ptx.barrier.cluster.wait()

        for vb in range(vec_blocks):
            local_col: K.int32 = (thread_in_row + vb * tpr) * 8
            absolute_col: K.int32 = block_y * cols + local_col
            col_valid: K.bool = absolute_col < H
            source_bytes: K.uint32 = K.uint32(16)
            if not full_columns:
                source_bytes = K.cast(K.if_then_else(col_valid, 16, 0), "uint32")
            with K.If(row_valid), K.Then():
                x_offset: K.int64 = K.cast(actual_row, "int64") * K.int64(H) + K.cast(
                    absolute_col, "int64"
                )
                K.ptx["cp.async.ca.shared.global"](
                    shared_raw.ptr_to([(row_in_cta * cols + local_col) * 2]),
                    x.ptr_to([x_offset]),
                    16,
                    source_bytes,
                )
                K.ptx["cp.async.ca.shared.global"](
                    shared_raw.ptr_to([tile_bytes + (row_in_cta * cols + local_col) * 2]),
                    residual.ptr_to([x_offset]),
                    16,
                    source_bytes,
                )
            if cluster_n == 1:
                K.ptx["cp.async.ca.shared.global"](
                    shared_raw.ptr_to([2 * tile_bytes + (row_in_cta * cols + local_col) * 2]),
                    weight.ptr_to([absolute_col]),
                    16,
                    source_bytes,
                )
        K.ptx.cp.async_.commit_group()
        K.ptx.cp.async_.wait_group(0)

        x_words = K.alloc_local((packed_pairs,), "uint32")
        r_words = K.alloc_local((packed_pairs,), "uint32")
        for vb in range(vec_blocks):
            local_col: K.int32 = (thread_in_row + vb * tpr) * 8
            K.ptx.ld.shared.v4.b32(
                x_words[vb * 4],
                x_words[vb * 4 + 1],
                x_words[vb * 4 + 2],
                x_words[vb * 4 + 3],
                shared_raw.ptr_to([(row_in_cta * cols + local_col) * 2]),
            )
            K.ptx.ld.shared.v4.b32(
                r_words[vb * 4],
                r_words[vb * 4 + 1],
                r_words[vb * 4 + 2],
                r_words[vb * 4 + 3],
                shared_raw.ptr_to([tile_bytes + (row_in_cta * cols + local_col) * 2]),
            )

        x_f32 = K.alloc_local((total_values,), "float32")
        r_f32 = K.alloc_local((total_values,), "float32")
        for pair in range(packed_pairs):
            x_low = K.alloc_local((1,), "uint16")
            x_high = K.alloc_local((1,), "uint16")
            K.ptx.mov.b32(x_low[0], x_high[0], x_words[pair])
            K.assign(x_f32[pair * 2], _cvt_to_f32(x_low[0], input_dtype))
            K.assign(x_f32[pair * 2 + 1], _cvt_to_f32(x_high[0], input_dtype))

        for pair in range(packed_pairs):
            r_low = K.alloc_local((1,), "uint16")
            r_high = K.alloc_local((1,), "uint16")
            K.ptx.mov.b32(r_low[0], r_high[0], r_words[pair])
            K.assign(r_f32[pair * 2], _cvt_to_f32(r_low[0], input_dtype))
            K.assign(r_f32[pair * 2 + 1], _cvt_to_f32(r_high[0], input_dtype))

        h_f32 = K.alloc_local((total_values,), "float32")
        h_sq = K.alloc_local((total_values,), "float32")
        h_words = K.alloc_local((packed_pairs,), "uint32")
        for pair in range(packed_pairs):
            h_pair = K.alloc_local((1,), "uint64")
            K.ptx.add.f32x2(
                h_pair[0],
                K.cuda.make_float2(x_f32[pair * 2], x_f32[pair * 2 + 1]),
                K.cuda.make_float2(r_f32[pair * 2], r_f32[pair * 2 + 1]),
            )
            K.ptx.mov.b64(h_f32[pair * 2], h_f32[pair * 2 + 1], h_pair[0])

        for pair in range(packed_pairs):
            h_pair = K.alloc_local((1,), "uint64")
            sq_pair = K.alloc_local((1,), "uint64")
            K.ptx.mov.b64(h_pair[0], h_f32[pair * 2], h_f32[pair * 2 + 1])
            K.ptx.mul.f32x2(sq_pair[0], h_pair[0], h_pair[0])
            K.ptx.mov.b64(h_sq[pair * 2], h_sq[pair * 2 + 1], sq_pair[0])

        for pair in range(packed_pairs):
            K.assign(
                h_words[pair], _cvt_pair_from_f32(h_f32[pair * 2 + 1], h_f32[pair * 2], input_dtype)
            )

        if cluster_n == 1:
            for vb in range(vec_blocks):
                local_col: K.int32 = (thread_in_row + vb * tpr) * 8
                K.ptx.st.shared.v4.b32(
                    shared_raw.ptr_to([3 * tile_bytes + (row_in_cta * cols + local_col) * 2]),
                    h_words[vb * 4],
                    h_words[vb * 4 + 1],
                    h_words[vb * 4 + 2],
                    h_words[vb * 4 + 3],
                )

        local_sum: K.float32 = K.float32(0.0)
        for value in range(total_values):
            local_sum = _add_f32(local_sum, h_sq[value])
        local_sum = _butterfly_sum_f32(local_sum, row_lane_xors)
        warp_sum: K.float32 = local_sum

        if warps_per_row > 1 and cluster_n == 1:
            with K.If(lane == 0), K.Then():
                reduce_index: K.int32 = row_warp + warp_in_row * rows
                K.ptx.st.shared.b32(
                    shared_raw.ptr_to([reduce_base + reduce_index * 4]),
                    K.reinterpret("uint32", warp_sum),
                )
            K.ptx.bar.sync(K.uint32(0))
            final_sum = K.local_scalar(K.f32, init=K.float32(0.0))
            with K.If(lane < warps_per_row), K.Then():
                reduce_word = K.alloc_local((1,), "uint32")
                reduce_index: K.int32 = row_warp + lane * rows
                K.ptx.ld.shared.b32(
                    reduce_word[0], shared_raw.ptr_to([reduce_base + reduce_index * 4])
                )
                K.assign(final_sum, K.reinterpret("float32", reduce_word[0]))
            K.assign(final_sum, _butterfly_sum_f32(final_sum, full_lane_xors))
            sum_sq: K.float32 = final_sum
        elif cluster_n > 1:
            with K.If(warp == 0), K.Then():
                with K.If(K.cuda.elect_sync()), K.Then():
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        shared_raw.ptr_to([mbar_offset]), K.uint32(expected_bytes)
                    )
            with K.If(lane < cluster_n), K.Then():
                reduce_index: K.int32 = (
                    row_warp + warp_in_row * rows + cta_rank * rows * warps_per_row
                )
                peer_reduce: K.uint32 = _mapa_u32(
                    shared_raw.ptr_to([reduce_base + reduce_index * 4]), lane
                )
                peer_mbar: K.uint32 = _mapa_u32(shared_raw.ptr_to([mbar_offset]), lane)
                K.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.f32(
                    peer_reduce, warp_sum, peer_mbar
                )
            _cluster_mbarrier_wait(shared_raw.ptr_to([mbar_offset]))
            final_sum = K.local_scalar(K.f32, init=K.float32(0.0))
            for iteration in range(_ceil_div(total_partials_per_row, 32)):
                partial: K.int32 = lane + iteration * 32
                with K.If(partial < total_partials_per_row), K.Then():
                    partial_warp: K.int32 = partial % warps_per_row
                    partial_cta: K.int32 = partial // warps_per_row
                    reduce_index: K.int32 = (
                        row_warp + partial_warp * rows + partial_cta * rows * warps_per_row
                    )
                    reduce_word = K.alloc_local((1,), "uint32")
                    K.ptx.ld.shared.b32(
                        reduce_word[0], shared_raw.ptr_to([reduce_base + reduce_index * 4])
                    )
                    K.assign(
                        final_sum, _add_f32(final_sum, K.reinterpret("float32", reduce_word[0]))
                    )
            K.assign(final_sum, _butterfly_sum_f32(final_sum, full_lane_xors))
            sum_sq = final_sum
        else:
            sum_sq = warp_sum

        if H & (H - 1) == 0:
            shifted: K.float32 = _fma_rn_f32(sum_sq, K.float32(1.0 / H), runtime_eps)
        else:
            mean_sq: K.float32 = _div_rn_f32(sum_sq, K.float32(H))
            shifted = _add_f32(mean_sq, runtime_eps)
        rstd: K.float32 = _rsqrt_approx_ftz(shifted)

        global_scale_value: K.float32 = K.float32(0.0)
        if use_e4m3_scale:
            scale_bits = K.alloc_local((1,), "uint32")
            K.ptx.ld.global_.b32(scale_bits[0], global_scale.ptr_to([0]))
            global_scale_value = K.reinterpret("float32", scale_bits[0])

        if cluster_n > 1:
            K.ptx.barrier.cluster.arrive.relaxed()
            K.ptx.barrier.cluster.wait()
        else:
            K.ptx.bar.sync(K.uint32(0))

        with K.If(row_valid), K.Then():
            for vb in range(vec_blocks):
                local_col: K.int32 = (thread_in_row + vb * tpr) * 8
                absolute_col: K.int32 = block_y * cols + local_col
                col_valid: K.bool = absolute_col < H
                with K.If(col_valid), K.Then():
                    residual_offset: K.int64 = K.cast(actual_row, "int64") * K.int64(H) + K.cast(
                        absolute_col, "int64"
                    )
                    K.ptx.st.global_.v4.b32(
                        residual.ptr_to([residual_offset]),
                        h_words[vb * 4],
                        h_words[vb * 4 + 1],
                        h_words[vb * 4 + 2],
                        h_words[vb * 4 + 3],
                    )

        if cluster_n > 1:
            K.ptx.fence.acq_rel.cluster()
            K.ptx.barrier.cluster.arrive.relaxed()
            K.ptx.barrier.cluster.wait()

        with K.If(row_valid), K.Then():
            if cluster_n > 1:
                scale_iteration = K.local_scalar(K.i32, init=K.int32(0))
                with K.While(scale_iteration < scale_blocks_per_thread):
                    sf_index0: K.int32 = thread_in_row + scale_iteration * tpr
                    _process_add_scale_block(
                        shared_raw,
                        residual,
                        weight,
                        y,
                        scales,
                        scales_unswizzled,
                        y_norm,
                        actual_row,
                        row_in_cta,
                        sf_index0,
                        rstd,
                        global_scale_value,
                        fp4_max_rcp,
                        H=H,
                        block_size=block_size,
                        scale_format=scale_format,
                        swizzled=swizzled,
                        output_both_sf_layouts=output_both_sf_layouts,
                        output_norm=output_norm,
                        input_dtype=input_dtype,
                        cluster_n=cluster_n,
                        cols=cols,
                        tile_bytes=tile_bytes,
                    )
                    sf_index1: K.int32 = thread_in_row + (scale_iteration + 1) * tpr
                    _process_add_scale_block(
                        shared_raw,
                        residual,
                        weight,
                        y,
                        scales,
                        scales_unswizzled,
                        y_norm,
                        actual_row,
                        row_in_cta,
                        sf_index1,
                        rstd,
                        global_scale_value,
                        fp4_max_rcp,
                        H=H,
                        block_size=block_size,
                        scale_format=scale_format,
                        swizzled=swizzled,
                        output_both_sf_layouts=output_both_sf_layouts,
                        output_norm=output_norm,
                        input_dtype=input_dtype,
                        cluster_n=cluster_n,
                        cols=cols,
                        tile_bytes=tile_bytes,
                    )
                    K.assign(scale_iteration, _add_s32(scale_iteration, K.int32(2)))
            else:
                static_limit = 1 if output_norm or block_size == 32 else 2
                if scale_blocks_per_thread <= static_limit:
                    for scale_iteration in range(scale_blocks_per_thread):
                        sf_index: K.int32 = thread_in_row + scale_iteration * tpr
                        _process_add_scale_block(
                            shared_raw,
                            residual,
                            weight,
                            y,
                            scales,
                            scales_unswizzled,
                            y_norm,
                            actual_row,
                            row_in_cta,
                            sf_index,
                            rstd,
                            global_scale_value,
                            fp4_max_rcp,
                            H=H,
                            block_size=block_size,
                            scale_format=scale_format,
                            swizzled=swizzled,
                            output_both_sf_layouts=output_both_sf_layouts,
                            output_norm=output_norm,
                            input_dtype=input_dtype,
                            cluster_n=cluster_n,
                            cols=cols,
                            tile_bytes=tile_bytes,
                        )
                else:
                    scale_iteration = K.local_scalar(K.i32, init=K.int32(0))
                    with K.While(scale_iteration < scale_blocks_per_thread):
                        sf_index: K.int32 = thread_in_row + scale_iteration * tpr
                        _process_add_scale_block(
                            shared_raw,
                            residual,
                            weight,
                            y,
                            scales,
                            scales_unswizzled,
                            y_norm,
                            actual_row,
                            row_in_cta,
                            sf_index,
                            rstd,
                            global_scale_value,
                            fp4_max_rcp,
                            H=H,
                            block_size=block_size,
                            scale_format=scale_format,
                            swizzled=swizzled,
                            output_both_sf_layouts=output_both_sf_layouts,
                            output_norm=output_norm,
                            input_dtype=input_dtype,
                            cluster_n=cluster_n,
                            cols=cols,
                            tile_bytes=tile_bytes,
                        )
                        K.assign(scale_iteration, _add_s32(scale_iteration, K.int32(1)))

        if enable_pdl:
            K.ptx.griddepcontrol.launch_dependents()

    @K.kernel(warps=threads // 32, arch="sm_100a", grid=False)
    def flashinfer_add_rmsnorm_fp4quant(
        x: K.gptr[input_dtype],
        residual: K.gptr[input_dtype],
        weight: K.gptr[input_dtype, (H,)],
        y: K.gptr[K.u8],
        scales: K.gptr[K.u8],
        scales_unswizzled: K.gptr[K.u8],
        y_norm: K.gptr[input_dtype],
        global_scale: K.gptr[K.f32, (1,)],
        runtime_M: K.i32,
        runtime_eps: K.f32,
    ):
        with K.attr({"tirx.required_block_size": 1}):
            kernel_body(
                x,
                residual,
                weight,
                y,
                scales,
                scales_unswizzled,
                y_norm,
                global_scale,
                runtime_M,
                runtime_eps,
            )

    launch_params = ["blockIdx.x"]
    if cluster_n > 1:
        launch_params.extend(["blockIdx.y", "clusterCtaIdx.x", "clusterCtaIdx.y"])
    launch_params.append("threadIdx.x")
    if enable_pdl:
        launch_params.append("tirx.use_programtic_dependent_launch")
    launch_params.append("tirx.use_dyn_shared_memory")
    return flashinfer_add_rmsnorm_fp4quant.func.with_attr(
        "tirx.kernel_launch_params", launch_params
    )


def prepare_data(**config: Any):
    """Prepare deterministic inputs and caller-owned fixed-ABI outputs."""
    config = dict(config)
    data = _prepare_tensors(config)
    output = _prepare_output(config)
    return (
        data["x_arg"],
        data["residual_arg"],
        data["weight"],
        output["y_arg"],
        output["scale_arg"],
        output["scale_unswizzled_arg"],
        output["y_norm_arg"],
        data["global_scale"],
    )


_GUARD_ELEMENTS = 128
_GUARD_BYTE = 0xA5


def _torch_input_dtype(dtype: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def _logical_shape(config: dict[str, Any]) -> tuple[int, ...]:
    H = int(config["H"])
    if int(config["input_ndim"]) == 3:
        return int(config["B"]), int(config["S"]), H
    return int(config["M"]), H


def _prepare_tensors(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    M, H = int(config["M"]), int(config["H"])
    dtype = _torch_input_dtype(str(config["input_dtype"]))
    generator = torch.Generator(device="cuda").manual_seed(42)

    x_backing = torch.empty(M * H + _GUARD_ELEMENTS, dtype=dtype, device="cuda")
    residual_backing = torch.empty(M * H + _GUARD_ELEMENTS, dtype=dtype, device="cuda")
    x = x_backing[: M * H].view(_logical_shape(config))
    residual = residual_backing[: M * H].view(_logical_shape(config))
    if str(config.get("data_mode", "random")) == "zero":
        x.zero_()
        residual.zero_()
    elif H >= 65536:
        columns = torch.arange(H, device="cuda")
        x_pattern = torch.where(columns % 2 == 0, 0.75, -0.5).to(dtype)
        residual_pattern = torch.where(columns % 3 == 0, -0.25, 0.5).to(dtype)
        x.view(M, H).copy_(x_pattern.expand(M, H))
        residual.view(M, H).copy_(residual_pattern.expand(M, H))
    else:
        x.normal_(generator=generator)
        residual.normal_(generator=generator)
    x_backing[M * H :].fill_(1.0)
    residual_backing[M * H :].fill_(1.0)

    weight_backing = torch.empty(H + _GUARD_ELEMENTS, dtype=dtype, device="cuda")
    weight = weight_backing[:H]
    weight.normal_(generator=generator)
    weight_backing[H:].fill_(1.0)

    mode = str(config.get("global_scale_mode", "none"))
    if mode in ("none", "one"):
        scale_value = 1.0
    elif mode == "nonunit":
        scale_value = 3.25
    elif mode == "zero":
        scale_value = 0.0
    else:
        raise ValueError(f"unsupported global_scale_mode={mode!r}")
    global_scale = torch.tensor([scale_value], dtype=torch.float32, device="cuda")

    return {
        "x": x,
        "x2d": x.view(M, H),
        "x_arg": x_backing[: M * H],
        "x_backing": x_backing,
        "residual": residual,
        "residual2d": residual.view(M, H),
        "residual_arg": residual_backing[: M * H],
        "residual_backing": residual_backing,
        "weight": weight,
        "weight_backing": weight_backing,
        "global_scale": global_scale,
    }


def _scale_storage_size(config: dict[str, Any]) -> int:
    M, H = int(config["M"]), int(config["H"])
    columns = H // int(config["block_size"])
    if bool(config["swizzled"]) or bool(config["output_both_sf_layouts"]):
        return _ceil_div(M, 128) * _ceil_div(columns, 4) * 512
    return M * columns


def _prepare_output(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    M, H = int(config["M"]), int(config["H"])
    columns = H // int(config["block_size"])
    y_size = M * (H // 2)
    scale_size = _scale_storage_size(config)
    scale_unswizzled_size = M * columns
    y_norm_size = M * H

    y_backing = torch.full(
        (y_size + _GUARD_ELEMENTS,), _GUARD_BYTE, dtype=torch.uint8, device="cuda"
    )
    scale_backing = torch.full(
        (scale_size + _GUARD_ELEMENTS,), _GUARD_BYTE, dtype=torch.uint8, device="cuda"
    )
    scale_unswizzled_backing = torch.full(
        (scale_unswizzled_size + _GUARD_ELEMENTS,), _GUARD_BYTE, dtype=torch.uint8, device="cuda"
    )
    input_dtype = _torch_input_dtype(str(config["input_dtype"]))
    y_norm_backing = torch.empty(y_norm_size + _GUARD_ELEMENTS, dtype=input_dtype, device="cuda")
    y_norm_backing.fill_(1.0)

    y = y_backing[:y_size].view(M, H // 2)
    if bool(config["swizzled"]) or bool(config["output_both_sf_layouts"]):
        scale = scale_backing[:scale_size]
    else:
        scale = scale_backing[:scale_size].view(M, columns)
    scale_unswizzled = scale_unswizzled_backing[:scale_unswizzled_size].view(M, columns)
    y_norm = y_norm_backing[:y_norm_size].view(M, H)
    return {
        "y": y,
        "scale": scale,
        "scale_unswizzled": scale_unswizzled,
        "y_norm": y_norm,
        "y_arg": y_backing[:y_size],
        "scale_arg": scale_backing[:scale_size],
        "scale_unswizzled_arg": scale_unswizzled_backing[:scale_unswizzled_size],
        "y_norm_arg": y_norm_backing[:y_norm_size],
        "y_backing": y_backing,
        "scale_backing": scale_backing,
        "scale_unswizzled_backing": scale_unswizzled_backing,
        "y_norm_backing": y_norm_backing,
        "y_size": y_size,
        "scale_size": scale_size,
        "scale_unswizzled_size": scale_unswizzled_size,
        "y_norm_size": y_norm_size,
        "pointers": {
            "y": y.data_ptr(),
            "scale": scale.data_ptr(),
            "scale_unswizzled": scale_unswizzled.data_ptr(),
            "y_norm": y_norm.data_ptr(),
        },
        "strides": {
            "y": y.stride(),
            "scale": scale.stride(),
            "scale_unswizzled": scale_unswizzled.stride(),
            "y_norm": y_norm.stride(),
        },
    }


def _launch_tirx(executable, data, output, config: dict[str, Any]):
    return executable(
        data["x_arg"],
        data["residual_arg"],
        data["weight"],
        output["y_arg"],
        output["scale_arg"],
        output["scale_unswizzled_arg"],
        output["y_norm_arg"],
        data["global_scale"],
        int(config["M"]),
        float(config["eps"]),
    )


@functools.cache
def _compiled_test_specialization(
    input_dtype: str,
    H: int,
    block_size: int,
    scale_format: str,
    swizzled: bool,
    output_both_sf_layouts: bool,
    enable_pdl: bool,
    output_norm: bool,
):
    from tirx_kernels.runner import compile_kernel

    return compile_kernel(
        get_kernel(
            input_dtype=input_dtype,
            input_ndim=2,
            M=1,
            B=None,
            S=None,
            H=H,
            block_size=block_size,
            scale_format=scale_format,
            swizzled=swizzled,
            output_both_sf_layouts=output_both_sf_layouts,
            output_norm=output_norm,
            enable_pdl=enable_pdl,
            eps=_DEFAULT_EPS,
            global_scale_mode="none",
            allocation="preallocated",
            data_mode="random",
        )
    )


@functools.cache
def _flashinfer_compiled(
    H: int,
    block_size: int,
    input_dtype: str,
    scale_format: str,
    swizzled: bool,
    output_both_sf_layouts: bool,
    enable_pdl: bool,
    output_norm: bool,
):
    from flashinfer.cute_dsl.add_rmsnorm_fp4quant import _get_compiled_kernel

    return _get_compiled_kernel(
        H,
        block_size,
        input_dtype == "float16",
        100,
        scale_format,
        swizzled,
        output_both_sf_layouts,
        enable_pdl,
        output_norm,
    )


def _launch_flashinfer(data, output, config: dict[str, Any]):
    kernel = _flashinfer_compiled(
        int(config["H"]),
        int(config["block_size"]),
        str(config["input_dtype"]),
        str(config["scale_format"]),
        bool(config["swizzled"]),
        bool(config["output_both_sf_layouts"]),
        bool(config["enable_pdl"]),
        bool(config["output_norm"]),
    )
    return kernel(
        data["x2d"],
        data["residual2d"],
        data["weight"],
        output["y"],
        output["scale"],
        output["scale_unswizzled"],
        output["y_norm_arg"],
        data["global_scale"],
        int(config["M"]),
        float(config["eps"]),
    )


def _snapshot_inputs(data):
    return {
        "x": data["x"].clone(),
        "residual": data["residual"].clone(),
        "weight": data["weight"].clone(),
        "global_scale": data["global_scale"].clone(),
    }


def _assert_inputs_and_residual(data, snapshot, config: dict[str, Any], *, name: str):
    import torch

    if not torch.equal(data["x"], snapshot["x"]):
        raise AssertionError(f"{name}: input tensor was modified")
    if not torch.equal(data["weight"], snapshot["weight"]):
        raise AssertionError(f"{name}: weight tensor was modified")
    if not torch.equal(data["global_scale"], snapshot["global_scale"]):
        raise AssertionError(f"{name}: global-scale tensor was modified")
    expected_residual = (snapshot["x"].float() + snapshot["residual"].float()).to(
        _torch_input_dtype(str(config["input_dtype"]))
    )
    if not torch.equal(data["residual"], expected_residual):
        count = int((data["residual"] != expected_residual).sum().item())
        raise AssertionError(f"{name}: {count} residual values differ from in-place add")

    M, H = int(config["M"]), int(config["H"])
    if not torch.equal(data["x_backing"][M * H :], torch.ones_like(data["x_backing"][M * H :])):
        raise AssertionError(f"{name}: input guard was modified")
    if not torch.equal(
        data["residual_backing"][M * H :], torch.ones_like(data["residual_backing"][M * H :])
    ):
        raise AssertionError(f"{name}: residual guard was modified")
    if not torch.equal(data["weight_backing"][H:], torch.ones_like(data["weight_backing"][H:])):
        raise AssertionError(f"{name}: weight guard was modified")


def _logical_scale_bytes(output, config: dict[str, Any]):
    import torch

    M, H = int(config["M"]), int(config["H"])
    columns = H // int(config["block_size"])
    if not (bool(config["swizzled"]) or bool(config["output_both_sf_layouts"])):
        return output["scale"].view(M, columns)
    rows = torch.arange(M, device="cuda", dtype=torch.int64)[:, None]
    cols = torch.arange(columns, device="cuda", dtype=torch.int64)[None, :]
    offsets = (
        (rows // 128) * (_ceil_div(columns, 4) * 512)
        + (cols // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + (cols % 4)
    )
    return output["scale"].view(-1)[offsets]


def _valid_swizzled_offsets(config: dict[str, Any]):
    import torch

    M, H = int(config["M"]), int(config["H"])
    columns = H // int(config["block_size"])
    rows = torch.arange(M, device="cuda", dtype=torch.int64)[:, None]
    cols = torch.arange(columns, device="cuda", dtype=torch.int64)[None, :]
    return (
        (rows // 128) * (_ceil_div(columns, 4) * 512)
        + (cols // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + (cols % 4)
    ).reshape(-1)


def _assert_output_integrity(output, config: dict[str, Any], *, name: str) -> None:
    import torch

    for field in ("y", "scale", "scale_unswizzled", "y_norm"):
        if output[field].data_ptr() != output["pointers"][field]:
            raise AssertionError(f"{name}: {field} pointer identity changed")
        if output[field].stride() != output["strides"][field]:
            raise AssertionError(f"{name}: {field} stride changed")
    for field, size in (
        ("y", output["y_size"]),
        ("scale", output["scale_size"]),
        ("scale_unswizzled", output["scale_unswizzled_size"]),
    ):
        if not torch.all(output[f"{field}_backing"][size:] == _GUARD_BYTE):
            raise AssertionError(f"{name}: {field} trailing guard was modified")
    if not torch.all(output["y_norm_backing"][output["y_norm_size"] :] == 1.0):
        raise AssertionError(f"{name}: y_norm trailing guard was modified")

    if bool(config["swizzled"]) or bool(config["output_both_sf_layouts"]):
        valid = _valid_swizzled_offsets(config)
        padding_mask = torch.ones(output["scale_size"], dtype=torch.bool, device="cuda")
        padding_mask[valid] = False
        if not torch.all(output["scale_arg"][padding_mask] == _GUARD_BYTE):
            raise AssertionError(f"{name}: swizzled scale padding was modified")
    if not bool(config["output_both_sf_layouts"]):
        if not torch.all(output["scale_unswizzled_arg"] == _GUARD_BYTE):
            raise AssertionError(f"{name}: disabled linear scale output was modified")
    if not bool(config["output_norm"]):
        if not torch.all(output["y_norm_arg"] == 1.0):
            raise AssertionError(f"{name}: disabled y_norm output was modified")


def _assert_raw_equal(actual, expected, config: dict[str, Any], *, name: str) -> None:
    import torch

    if not torch.equal(actual["y"], expected["y"]):
        count = int((actual["y"] != expected["y"]).sum().item())
        raise AssertionError(f"{name}: {count} packed FP4 bytes differ")
    actual_scale = _logical_scale_bytes(actual, config)
    expected_scale = _logical_scale_bytes(expected, config)
    if not torch.equal(actual_scale, expected_scale):
        count = int((actual_scale != expected_scale).sum().item())
        raise AssertionError(f"{name}: {count} primary logical scale bytes differ")
    if bool(config["output_both_sf_layouts"]):
        if not torch.equal(actual["scale_unswizzled"], expected["scale_unswizzled"]):
            count = int((actual["scale_unswizzled"] != expected["scale_unswizzled"]).sum().item())
            raise AssertionError(f"{name}: {count} linear scale bytes differ")
        if not torch.equal(actual_scale, actual["scale_unswizzled"]):
            raise AssertionError(f"{name}: dual scale layouts disagree")
    if bool(config["output_norm"]) and not torch.equal(actual["y_norm"], expected["y_norm"]):
        count = int((actual["y_norm"] != expected["y_norm"]).sum().item())
        raise AssertionError(f"{name}: {count} y_norm values differ")


def _assert_math(output, data, snapshot, config: dict[str, Any]) -> None:
    """Zero-global-scale invariant only; FlashInfer's raw bytes are the arbiter.

    A zero global scale must yield signed-zero FP4 payloads and scales, and the
    bitwise FlashInfer comparison cannot distinguish "both wrong the same way"
    for this degenerate input, so the invariant is asserted directly.
    """
    import torch

    del snapshot
    if (int(config["block_size"]) == 16 or str(config["scale_format"]) == "e4m3") and float(
        data["global_scale"].item()
    ) == 0.0:
        if torch.count_nonzero(output["y"] & 0x77) or torch.count_nonzero(
            _logical_scale_bytes(output, config)
        ):
            raise AssertionError("zero global scale must produce signed-zero FP4 and scales")


def _check_public_allocation(reference, reference_data, config: dict[str, Any]) -> None:
    import torch
    from flashinfer.norm import add_rmsnorm_fp4quant

    if str(config.get("allocation")) not in ("auto", "compare"):
        return
    public_data = _prepare_tensors(config)
    global_scale = (
        None if str(config.get("global_scale_mode")) == "none" else public_data["global_scale"]
    )
    result = add_rmsnorm_fp4quant(
        public_data["x"],
        public_data["residual"],
        public_data["weight"],
        global_scale=global_scale,
        eps=float(config["eps"]),
        block_size=int(config["block_size"]),
        scale_format=str(config["scale_format"]),
        is_sf_swizzled_layout=bool(config["swizzled"]),
        output_both_sf_layouts=bool(config["output_both_sf_layouts"]),
        enable_pdl=bool(config["enable_pdl"]),
    )
    public_y = result[0].view(torch.uint8).view_as(reference["y"])
    if not torch.equal(public_y, reference["y"]):
        raise AssertionError("public auto-allocated packed output differs")
    public_output = dict(reference)
    public_output["scale"] = result[1].view(torch.uint8)
    if not torch.equal(
        _logical_scale_bytes(public_output, config), _logical_scale_bytes(reference, config)
    ):
        raise AssertionError("public auto-allocated logical scales differ")
    if not torch.equal(public_data["residual"], reference_data["residual"]):
        raise AssertionError("public residual update differs from direct source kernel")


def run_test(**config: Any) -> None:
    """Compile, launch, and validate one source-domain specialization."""
    import torch

    config = dict(config)
    _validate(config)
    tirx_data = _prepare_tensors(config)
    source_data = _prepare_tensors(config)
    tirx_snapshot = _snapshot_inputs(tirx_data)
    source_snapshot = _snapshot_inputs(source_data)
    tirx_output = _prepare_output(config)
    source_output = _prepare_output(config)
    executable = _compiled_test_specialization(
        str(config["input_dtype"]),
        int(config["H"]),
        int(config["block_size"]),
        str(config["scale_format"]),
        bool(config["swizzled"]),
        bool(config["output_both_sf_layouts"]),
        bool(config["enable_pdl"]),
        bool(config["output_norm"]),
    )
    if _launch_tirx(executable, tirx_data, tirx_output, config) is not None:
        raise AssertionError("TIRx AddRMSNormFP4Quant ABI must return None")
    if _launch_flashinfer(source_data, source_output, config) is not None:
        raise AssertionError("FlashInfer AddRMSNormFP4Quant ABI must return None")
    torch.cuda.synchronize()

    _assert_raw_equal(tirx_output, source_output, config, name="FlashInfer raw-byte oracle")
    if not torch.equal(tirx_data["residual"], source_data["residual"]):
        count = int((tirx_data["residual"] != source_data["residual"]).sum().item())
        raise AssertionError(f"FlashInfer residual oracle: {count} values differ")
    _assert_inputs_and_residual(tirx_data, tirx_snapshot, config, name="TIRx")
    _assert_inputs_and_residual(source_data, source_snapshot, config, name="FlashInfer")
    _assert_math(tirx_output, tirx_data, tirx_snapshot, config)
    _assert_output_integrity(tirx_output, config, name="TIRx output")
    _assert_output_integrity(source_output, config, name="FlashInfer output")
    _check_public_allocation(source_output, source_data, config)


def prepare_bench(**config: Any):
    """Compile the specialization before the bench suite assigns a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(config), "executable": compile_kernel(get_kernel(**config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs: Any
):
    """Build and prevalidate independent TIRx and CuTeDSL launch closures."""
    import torch

    config = dict(prepared["config"])
    config.update(kwargs)
    tirx_data = _prepare_tensors(config)
    tirx_output = _prepare_output(config)
    executable = prepared["executable"]

    def tirx_launch():
        return _launch_tirx(executable, tirx_data, tirx_output, config)

    if tirx_launch() is not None:
        raise AssertionError("TIRx benchmark closure must return None")
    torch.cuda.synchronize()

    def build_flashinfer_reference():
        source_data = _prepare_tensors(config)
        source_output = _prepare_output(config)

        def source_launch():
            return _launch_flashinfer(source_data, source_output, config)

        if source_launch() is not None:
            raise AssertionError("FlashInfer benchmark closure must return None")
        torch.cuda.synchronize()
        _assert_raw_equal(tirx_output, source_output, config, name="benchmark raw-byte precheck")
        if not torch.equal(tirx_data["residual"], source_data["residual"]):
            raise AssertionError("benchmark residual precheck differs")
        return source_launch

    return bench(
        {"tirx": tirx_launch},
        references={"flashinfer_cutedsl": build_flashinfer_reference},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config: Any):
    """Benchmark one specialization against the CuTeDSL kernel reference."""
    prepared = prepare_bench(**config)
    return prepared.run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_gpu",
    "run_test",
]
