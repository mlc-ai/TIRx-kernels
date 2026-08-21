# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400e330fb2debe0bf8730d9424a1d37927f), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer CuTe-DSL RMSNorm with packed FP4 quantization.

The source implementation is ``RMSNormFP4QuantKernel`` in
``flashinfer/cute_dsl/rmsnorm_fp4quant.py`` with inline PTX helpers from
``flashinfer/cute_dsl/fp4_common.py``.  The public entry is
``flashinfer.norm.rmsnorm_fp4quant``.
"""

from __future__ import annotations

import functools
from typing import Any

from tirx_kernels.flashinfer.utils.fp_quant import (
    absmax_8,
    cvt_e2m1x8,
    hmax2,
    pack_u32x2_to_u64,
    sf_offset_128x4,
)
from tirx_kernels.runner import bench
from tvm.script import tirx as T

KERNEL_META = {
    "name": "flashinfer_rmsnorm_fp4quant",
    "category": "flashinfer",
    "compute_capability": 10,
}

_DTYPES = ("float16", "bfloat16")
_DEFAULT_EPS = 1e-6
_OPTIN_SMEM_BYTES = 232448


def _ptx_unary(chain: str, value, dtype: str = "float32"):
    out = T.alloc_local((1,), dtype)
    T.evaluate(T.ptx[chain](out[0], value))
    return out[0]


def _ptx_binary(chain: str, lhs, rhs, dtype: str = "float32"):
    out = T.alloc_local((1,), dtype)
    T.evaluate(T.ptx[chain](out[0], lhs, rhs))
    return out[0]


def _add_f32(lhs, rhs):
    return _ptx_binary("add.f32", lhs, rhs)


def _mul_f32(lhs, rhs):
    return _ptx_binary("mul.f32", lhs, rhs)


def _add_s32(lhs, rhs):
    return _ptx_binary("add.s32", lhs, rhs, dtype="int32")


def _div_rn_f32(lhs, rhs):
    return _ptx_binary("div.rn.f32", lhs, rhs)


def _fma_rn_f32(lhs, rhs, acc):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.fma.rn.f32(out[0], lhs, rhs, acc))
    return out[0]


def _rcp_approx_ftz(value):
    return _ptx_unary("rcp.approx.ftz.f32", value)


def _rsqrt_approx_ftz(value):
    return _ptx_unary("rsqrt.approx.ftz.f32", value)


def _cvt_to_f32(bits, input_dtype: str):
    chain = "cvt.f32.f16" if input_dtype == "float16" else "cvt.f32.bf16"
    return _ptx_unary(chain, T.cast(bits, "uint16"))


def _shfl_bfly_f32(value, lane_xor: int):
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


def _butterfly_sum_f32(value, lane_xors: tuple[int, ...]):
    for lane_xor in lane_xors:
        value = _add_f32(value, _shfl_bfly_f32(value, lane_xor))
    return value


def _mapa_u32(pointer, peer):
    mapped = T.alloc_local((1,), "uint32")
    T.evaluate(
        T.ptx.mapa.shared__cluster.u32(
            mapped[0], T.cuda.cvta_generic_to_shared(pointer), T.cast(peer, "uint32")
        )
    )
    return mapped[0]


@T.inline
def _cluster_mbarrier_wait(pointer):
    # The pinned TVM has no mbarrier_wait_relaxed; the phase-0 wait is the
    # spelling the older norm ports use for this same one-shot cluster
    # rendezvous (see _parser_helpers._cluster_mbarrier_wait).
    T.cuda.mbarrier_wait(pointer, T.int32(0))


def _mul_input2(lhs, rhs, input_dtype: str):
    chain = "mul.f16x2" if input_dtype == "float16" else "mul.bf16x2"
    return _ptx_binary(chain, lhs, rhs, dtype="uint32")


def _minimum_f32(lhs, rhs):
    return _ptx_binary("min.f32", lhs, rhs)


def _cvt_f32_to_e4m3(value):
    pair = T.alloc_local((1,), "uint16")
    word = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.cvt.rn.satfinite.e4m3x2.f32(pair[0], T.float32(0.0), value))
    T.evaluate(T.ptx.cvt.u32.u16(word[0], pair[0]))
    return word[0]


def _e4m3_to_f32_rcp(value):
    pair = T.alloc_local((1,), "uint16")
    halves = T.alloc_local((1,), "uint32")
    half_bits = T.alloc_local((2,), "uint16")
    decoded = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.cvt.u16.u32(pair[0], value))
    T.evaluate(T.ptx.cvt.rn.f16x2.e4m3x2(halves[0], pair[0]))
    T.evaluate(T.ptx.mov.b32(half_bits[0], half_bits[1], halves[0]))
    T.evaluate(T.ptx.cvt.f32.f16(decoded[0], half_bits[0]))
    reciprocal = _rcp_approx_ftz(decoded[0])
    return T.Select(decoded[0] == T.float32(0.0), T.float32(0.0), reciprocal)


def _cvt_f32_to_ue8m0(value):
    predicate_zero = T.local_scalar("uint32")
    predicate_negative = T.local_scalar("uint32")
    predicate_overflow = T.local_scalar("uint32")
    log2_value = T.alloc_local((1,), "float32")
    exponent = T.alloc_local((1,), "int32")
    result = T.alloc_local((1,), "int32")
    T.evaluate(T.ptx.setp.le.f32(predicate_zero, value, T.float32(0.0)))
    T.evaluate(T.ptx.lg2.approx.f32(log2_value[0], value))
    T.evaluate(T.ptx["cvt.rpi.s32.f32"](exponent[0], log2_value[0]))
    T.evaluate(T.ptx.add.s32(result[0], exponent[0], T.int32(127)))
    T.evaluate(T.ptx.setp.lt.s32(predicate_negative, result[0], T.int32(0)))
    T.evaluate(T.ptx.setp.gt.s32(predicate_overflow, result[0], T.int32(255)))
    T.evaluate(T.ptx.selp.s32(result[0], T.int32(0), result[0], T.ptx.pred(predicate_negative)))
    T.evaluate(T.ptx.selp.s32(result[0], T.int32(255), result[0], T.ptx.pred(predicate_overflow)))
    T.evaluate(T.ptx.selp.s32(result[0], T.int32(0), result[0], T.ptx.pred(predicate_zero)))
    return T.cast(result[0], "uint32")


def _ue8m0_to_output_scale(value):
    predicate_zero = T.local_scalar("uint32")
    negative_exponent = T.alloc_local((1,), "int32")
    negative_exponent_f32 = T.alloc_local((1,), "float32")
    candidate = T.alloc_local((1,), "float32")
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.setp.eq.u32(predicate_zero, value, T.uint32(0)))
    T.evaluate(T.ptx.sub.s32(negative_exponent[0], T.int32(127), T.cast(value, "int32")))
    T.evaluate(T.ptx["cvt.rn.f32.s32"](negative_exponent_f32[0], negative_exponent[0]))
    T.evaluate(T.ptx.ex2.approx.f32(candidate[0], negative_exponent_f32[0]))
    T.evaluate(T.ptx.selp.f32(out[0], T.float32(0.0), candidate[0], T.ptx.pred(predicate_zero)))
    return out[0]


def _load_16_input_words(buffer, base_index):
    words = T.alloc_local((8,), "uint32")
    T.evaluate(
        T.ptx.ld.global_.v4.u32(
            words[0], words[1], words[2], words[3], T.address_of(buffer[base_index])
        )
    )
    T.evaluate(
        T.ptx.ld.global_.v4.u32(
            words[4], words[5], words[6], words[7], T.address_of(buffer[base_index + 8])
        )
    )
    return words


def _multiply_8_pairs(lhs, rhs, input_dtype: str):
    products = T.alloc_local((8,), "uint32")
    chain = T.ptx.mul.f16x2 if input_dtype == "float16" else T.ptx.mul.bf16x2
    for pair in range(8):
        T.evaluate(chain(products[pair], lhs[pair], rhs[pair]))
    return products


def _pair_max_to_f32(word, input_dtype: str):
    out = T.alloc_local((1,), "float32")
    if input_dtype == "float16":
        bits = T.alloc_local((2,), "uint16")
        values = T.alloc_local((2,), "float32")
        T.evaluate(T.ptx.mov.b32(bits[0], bits[1], word))
        T.evaluate(T.ptx.cvt.f32.f16(values[0], bits[0]))
        T.evaluate(T.ptx.cvt.f32.f16(values[1], bits[1]))
        T.evaluate(T.ptx.max.f32(out[0], values[0], values[1]))
    else:
        low_bits = T.bitwise_and(word, T.uint32(0xFFFF))
        high_bits = T.shift_right(word, T.uint32(16))
        low = T.reinterpret("float32", T.shift_left(low_bits, T.uint32(16)))
        high = T.reinterpret("float32", T.shift_left(high_bits, T.uint32(16)))
        T.evaluate(T.ptx.max.f32(out[0], low, high))
    return out[0]


def _widen_and_scale_16(words, scale, input_dtype: str):
    values = T.alloc_local((16,), "float32")
    for pair in range(8):
        bits = T.alloc_local((2,), "uint16")
        widened = T.alloc_local((2,), "float32")
        T.evaluate(T.ptx.mov.b32(bits[0], bits[1], words[pair]))
        if input_dtype == "float16":
            T.evaluate(T.ptx.cvt.f32.f16(widened[0], bits[0]))
            T.evaluate(T.ptx.cvt.f32.f16(widened[1], bits[1]))
        else:
            T.evaluate(T.ptx.cvt.f32.bf16(widened[0], bits[0]))
            T.evaluate(T.ptx.cvt.f32.bf16(widened[1], bits[1]))
        T.evaluate(T.ptx.mul.f32(values[pair * 2], widened[0], scale))
        T.evaluate(T.ptx.mul.f32(values[pair * 2 + 1], widened[1], scale))
    return values


def _store_global_u64(pointer, value):
    T.evaluate(T.ptx.st.global_.u64(pointer, value))


def _quantize_and_pack_16(values, output_scale):
    scaled = T.alloc_local((16,), "float32")
    for value in range(16):
        T.evaluate(T.ptx.mul.f32(scaled[value], values[value], output_scale))
    low = cvt_e2m1x8([scaled[index] for index in range(8)])
    high = cvt_e2m1x8([scaled[index] for index in range(8, 16)])
    return pack_u32x2_to_u64(low, high)


def _scale_offset(actual_row, sf_index, H: int, block_size: int, swizzled: bool):
    scale_columns = H // block_size
    if swizzled:
        return sf_offset_128x4(actual_row, sf_index, _ceil_div(scale_columns, 4) * 4)
    return actual_row * scale_columns + sf_index


@T.inline
def _process_scale_block(
    x,
    weight,
    y,
    scales,
    actual_row,
    sf_index,
    rstd,
    global_scale_value,
    fp4_max_rcp,
    *,
    H: T.constexpr,
    block_size: T.constexpr,
    scale_format: T.constexpr,
    swizzled: T.constexpr,
    input_dtype: T.constexpr,
):
    num_scale_blocks = H // block_size
    if sf_index < num_scale_blocks:
        block_start = sf_index * block_size
        x_base = actual_row * H + block_start

        x0 = _load_16_input_words(x, x_base)
        w0 = _load_16_input_words(weight, block_start)

        if block_size == 16:
            products0 = _multiply_8_pairs(x0, w0, input_dtype)
            pair_max = absmax_8(products0, input_dtype)
            max_xw = _pair_max_to_f32(pair_max, input_dtype)
            values0 = _widen_and_scale_16(products0, rstd, input_dtype)
            max_abs = _mul_f32(max_xw, rstd)
            scale_value = _mul_f32(_mul_f32(global_scale_value, max_abs), fp4_max_rcp)
            scale_value = _minimum_f32(scale_value, T.float32(448.0))
            scale_word = _cvt_f32_to_e4m3(scale_value)
            output_scale = _mul_f32(_e4m3_to_f32_rcp(scale_word), global_scale_value)
            scale_offset = _scale_offset(actual_row, sf_index, H, block_size, swizzled)
            T.ptx.st.global_.b8(
                T.address_of(scales[scale_offset]), T.cast(scale_word, "uint8")
            )
            packed0 = _quantize_and_pack_16(values0, output_scale)
            y_offset = T.cast(actual_row, "int64") * (H // 2) + block_start // 2
            _store_global_u64(T.address_of(y[y_offset]), packed0)
        else:
            x1 = _load_16_input_words(x, x_base + 16)
            w1 = _load_16_input_words(weight, block_start + 16)
            products0 = _multiply_8_pairs(x0, w0, input_dtype)
            products1 = _multiply_8_pairs(x1, w1, input_dtype)
            max0 = absmax_8(products0, input_dtype)
            max1 = absmax_8(products1, input_dtype)
            pair_max = hmax2(max0, max1, input_dtype)
            max_xw = _pair_max_to_f32(pair_max, input_dtype)
            values0 = _widen_and_scale_16(products0, rstd, input_dtype)
            values1 = _widen_and_scale_16(products1, rstd, input_dtype)
            max_abs = _mul_f32(max_xw, rstd)
            if scale_format == "ue8m0":
                scale_word = _cvt_f32_to_ue8m0(_mul_f32(max_abs, fp4_max_rcp))
                output_scale = _ue8m0_to_output_scale(scale_word)
            else:
                scale_value = _mul_f32(_mul_f32(global_scale_value, max_abs), fp4_max_rcp)
                scale_value = _minimum_f32(scale_value, T.float32(448.0))
                scale_word = _cvt_f32_to_e4m3(scale_value)
                output_scale = _mul_f32(_e4m3_to_f32_rcp(scale_word), global_scale_value)
            scale_offset = _scale_offset(actual_row, sf_index, H, block_size, swizzled)
            T.ptx.st.global_.b8(
                T.address_of(scales[scale_offset]), T.cast(scale_word, "uint8")
            )
            packed0 = _quantize_and_pack_16(values0, output_scale)
            packed1 = _quantize_and_pack_16(values1, output_scale)
            y_offset = T.cast(actual_row, "int64") * (H // 2) + block_start // 2
            _store_global_u64(T.address_of(y[y_offset]), packed0)
            _store_global_u64(T.address_of(y[y_offset + 8]), packed1)


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _threads_per_row(H_per_cta: int) -> int:
    if H_per_cta <= 64:
        return 8
    if H_per_cta <= 128:
        return 16
    if H_per_cta <= 3072:
        return 32
    if H_per_cta <= 6144:
        return 64
    if H_per_cta <= 16384:
        return 128
    return 256


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
    config = _derived_config(H, cluster_n)
    reduction = config["rows"] * config["warps_per_row"] * 4
    if cluster_n == 1:
        return 2 * config["tile_bytes"] + reduction
    return config["tile_bytes"] + reduction * cluster_n + 8


def _source_config(H: int) -> dict[str, int]:
    cluster_n = 16
    for candidate in (1, 2, 4, 8, 16):
        if H % candidate == 0 and _estimate_smem(H, candidate) <= _OPTIN_SMEM_BYTES:
            cluster_n = candidate
            break
    config = _derived_config(H, cluster_n)
    config["cluster_n"] = cluster_n
    config["smem_bytes"] = (
        config["tile_bytes"]
        + config["rows"] * config["warps_per_row"] * cluster_n * 4
        + (8 if cluster_n > 1 else 0)
    )
    return config


def _dtype_code(dtype: str) -> str:
    return {"float16": "fp16", "bfloat16": "bf16"}[dtype]


def _eps_code(eps: float) -> str:
    return {1e-4: "1e4", 1e-5: "1e5", 1e-6: "1e6"}[eps]


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
    enable_pdl: bool = False,
    eps: float = _DEFAULT_EPS,
    global_scale_mode: str = "computed",
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
        f"sw{int(swizzled)}_pdl{int(enable_pdl)}_eps{_eps_code(eps)}_"
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
        "enable_pdl": enable_pdl,
        "eps": eps,
        "global_scale_mode": global_scale_mode,
        "allocation": allocation,
        "data_mode": data_mode,
    }


_NV2D_CONFIGS = [
    _cfg("nv2d", dtype, H, M=M, eps=eps)
    for M in (1, 4, 16, 32, 7, 13, 33, 100, 128, 8192, 16384)
    for H in (64, 128, 256, 512, 1024, 1536, 2048, 4096, 8192)
    for dtype in _DTYPES
    for eps in (1e-5, 1e-6)
]

_NV3D_CONFIGS = [
    _cfg("nv3d", dtype, H, B=B, S=S, eps=1e-5)
    for B in (1, 4, 3, 7, 128)
    for S in (16, 64, 128, 37, 99)
    for H in (128, 256, 1536, 4096, 8192)
    for dtype in _DTYPES
]

_LARGE_BATCH_CONFIGS = [
    _cfg("large_batch", "float16", H, M=M) for M, H in ((512, 4096), (1024, 4096))
]

_MX_BASIC_CONFIGS = [
    _cfg("mx_basic", dtype, H, M=M, block_size=32, scale_format="ue8m0", global_scale_mode="none")
    for M in (1, 4, 16, 7, 25, 128, 8192)
    for H in (128, 256, 512, 1536, 2048, 4096)
    for dtype in _DTYPES
]

_CROSS_CONFIGS = [
    *[
        _cfg("fused_vs_separate", "float16", H, M=M)
        for M in (4, 16, 128, 512, 8192)
        for H in (256, 512, 1024, 1536, 4096, 8192)
    ],
    *[
        _cfg("nv_matches_separate", dtype, H, M=M)
        for M in (1, 4, 16, 128)
        for H in (64, 256, 512, 1024, 2048, 4096)
        for dtype in _DTYPES
    ],
    *[
        _cfg(
            "mx_matches_separate",
            dtype,
            H,
            M=M,
            block_size=32,
            scale_format="ue8m0",
            global_scale_mode="none",
        )
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
        _cfg(
            "large_h_mx",
            dtype,
            H,
            M=M,
            block_size=32,
            scale_format="ue8m0",
            global_scale_mode="none",
        )
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
        _cfg(
            "swizzled_mx",
            dtype,
            H,
            M=M,
            block_size=32,
            scale_format="ue8m0",
            swizzled=True,
            global_scale_mode="none",
        )
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
        _cfg(
            "auto_mx",
            "float16",
            H,
            M=M,
            block_size=32,
            scale_format="ue8m0",
            global_scale_mode="none",
            allocation="auto",
        )
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

_UPSTREAM_CONFIGS = [
    *_NV2D_CONFIGS,
    *_NV3D_CONFIGS,
    *_LARGE_BATCH_CONFIGS,
    *_MX_BASIC_CONFIGS,
    *_CROSS_CONFIGS,
    *_LARGE_H_CONFIGS,
    *_SWIZZLED_CONFIGS,
    *_ALLOCATION_CONFIGS,
]

_STRUCTURE_CONFIGS = [
    _cfg("subwarp_rows_tail", "float16", 64, M=17),
    _cfg(
        "subwarp_mx",
        "bfloat16",
        64,
        M=17,
        block_size=32,
        scale_format="ue8m0",
        global_scale_mode="none",
    ),
    _cfg("column_tail_nv", "bfloat16", 80, M=129, swizzled=True),
    _cfg(
        "column_tail_mx",
        "float16",
        160,
        M=3,
        block_size=32,
        scale_format="ue8m0",
        global_scale_mode="none",
    ),
    *[
        _cfg("thread_threshold", "bfloat16", H, M=3)
        for H in (3072, 3088, 6144, 6160, 14336, 16384, 16400)
    ],
    _cfg("cluster2", "bfloat16", 65536, M=1),
    _cfg("cluster16", "bfloat16", 1048576, M=1, enable_pdl=True),
    _cfg("pdl", "bfloat16", 4096, M=3, enable_pdl=True),
    _cfg(
        "block32_e4m3",
        "float16",
        4096,
        M=3,
        block_size=32,
        scale_format="e4m3",
        global_scale_mode="nonunit",
    ),
    _cfg("block16_requested_ue8", "float16", 4096, M=3, block_size=16, scale_format="ue8m0"),
    _cfg("zero_input", "bfloat16", 4096, M=3, data_mode="zero"),
    _cfg("zero_global_scale", "float16", 4096, M=3, global_scale_mode="zero"),
]

CONFIGS = [*_UPSTREAM_CONFIGS, *_STRUCTURE_CONFIGS]

assert len(_NV2D_CONFIGS) == 396
assert len(_NV3D_CONFIGS) == 250
assert len(_LARGE_BATCH_CONFIGS) == 2
assert len(_MX_BASIC_CONFIGS) == 84
assert len(_CROSS_CONFIGS) == 135
assert len(_LARGE_H_CONFIGS) == 32
assert len(_SWIZZLED_CONFIGS) == 64
assert len(_ALLOCATION_CONFIGS) == 44
assert len(_UPSTREAM_CONFIGS) == 1007
assert len({config["label"] for config in CONFIGS}) == len(CONFIGS)

BENCH_CONFIGS = [
    _cfg("bench_nv", "bfloat16", 4096, M=32, global_scale_mode="none"),
    _cfg("bench_nv_large", "bfloat16", 8192, M=64, global_scale_mode="none"),
    _cfg("bench_nv_global", "bfloat16", 4096, M=32, global_scale_mode="one"),
    _cfg("bench_nv_swizzled", "bfloat16", 4096, M=32, swizzled=True, global_scale_mode="none"),
    _cfg(
        "bench_mx",
        "bfloat16",
        4096,
        M=32,
        block_size=32,
        scale_format="ue8m0",
        global_scale_mode="none",
    ),
    _cfg("bench_nv_3d", "bfloat16", 128, B=32, S=32, global_scale_mode="none"),
]


def _validate(input_dtype: str, H: int, block_size: int, scale_format: str, M: int) -> None:
    if input_dtype not in _DTYPES:
        raise ValueError(f"unsupported input_dtype={input_dtype!r}")
    if H < 64 or H % block_size != 0:
        raise ValueError(f"H={H} must be >=64 and divisible by block_size={block_size}")
    if block_size not in (16, 32):
        raise ValueError(f"unsupported block_size={block_size}")
    if scale_format not in ("e4m3", "ue8m0"):
        raise ValueError(f"unsupported scale_format={scale_format!r}")
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")


def get_kernel(
    input_dtype: str,
    M: int,
    H: int,
    block_size: int,
    scale_format: str,
    swizzled: bool,
    enable_pdl: bool,
    **kwargs: Any,
):
    """Return the compact source-faithful RMSNorm/FP4 specialization."""
    _validate(input_dtype, H, block_size, scale_format, M)
    del kwargs
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
    packed_pairs = total_values // 2
    reduce_base = tile_bytes
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
    scale_columns = H // block_size

    @T.inline
    def kernel_body(x, weight, y, scales, global_scale, runtime_M, runtime_eps):
        # TIRX_TRANSCRIBE_START flashinfer_rmsnorm_fp4quant
        T.attr({"tirx.required_block_size": 1})
        if cluster_n > 1:
            block_x_raw, block_y_raw = T.cta_id(
                [T.cast(T.ceildiv(runtime_M, T.int32(rows)), "int32"), cluster_n]
            )
            _, cta_rank_raw = T.cta_id_in_cluster([1, cluster_n], preferred=[1, cluster_n])
            block_y: T.int32 = T.cast(block_y_raw, "int32")
            cta_rank: T.int32 = T.cast(cta_rank_raw, "int32")
        else:
            block_x_raw = T.cta_id([T.cast(T.ceildiv(runtime_M, T.int32(rows)), "int32")])
            block_y = T.int32(0)
            cta_rank = T.int32(0)
        tid = T.thread_id([threads])

        if enable_pdl:
            T.ptx.griddepcontrol.wait()

        fp4_max_rcp: T.float32 = _rcp_approx_ftz(T.float32(6.0))
        block_x: T.int32 = T.cast(block_x_raw, "int32")
        row_in_cta: T.int32 = tid // tpr
        thread_in_row: T.int32 = tid % tpr
        actual_row: T.int32 = block_x * rows + row_in_cta
        row_valid: T.bool = actual_row < runtime_M
        warp: T.int32 = tid // 32
        lane: T.int32 = tid % 32
        row_warp: T.int32 = warp // warps_per_row
        warp_in_row: T.int32 = warp % warps_per_row

        shared_raw = T.alloc_buffer((smem_bytes,), "uint8", scope="shared.dyn", align=1024)
        T.attr({"tirx.dyn_smem_bytes": smem_bytes})

        if cluster_n > 1:
            if tid == 0:
                T.ptx.mbarrier.init.shared.b64(shared_raw.ptr_to([mbar_offset]), T.uint32(1))
            T.ptx.fence.mbarrier_init.release.cluster()
            T.ptx.barrier.cluster.arrive.relaxed()
            T.ptx.barrier.cluster.wait()

        for vb in T.unroll(vec_blocks):
            local_col: T.int32 = (thread_in_row + vb * tpr) * 8
            absolute_col: T.int32 = block_y * cols + local_col
            col_valid: T.bool = absolute_col < H
            if row_valid:
                source_bytes: T.uint32 = T.uint32(16)
                if not full_columns:
                    source_bytes = T.cast(T.if_then_else(col_valid, 16, 0), "uint32")
                x_offset: T.int64 = T.cast(actual_row, "int64") * T.int64(H) + T.cast(
                    absolute_col, "int64"
                )
                T.ptx["cp.async.ca.shared.global"](
                    shared_raw.ptr_to([(row_in_cta * cols + local_col) * 2]),
                    x.ptr_to([x_offset]),
                    16,
                    source_bytes,
                )
        T.ptx.cp.async_.commit_group()
        T.ptx.cp.async_.wait_group(0)

        x_words = T.alloc_local((packed_pairs,), "uint32")
        for vb in T.unroll(vec_blocks):
            local_col: T.int32 = (thread_in_row + vb * tpr) * 8
            T.ptx.ld.shared.v4.b32(
                x_words[vb * 4],
                x_words[vb * 4 + 1],
                x_words[vb * 4 + 2],
                x_words[vb * 4 + 3],
                shared_raw.ptr_to([(row_in_cta * cols + local_col) * 2]),
            )

        x_f32_pairs = T.alloc_local((packed_pairs,), "uint64")
        for pair in T.unroll(packed_pairs):
            low_bits = T.alloc_local((1,), "uint16")
            high_bits = T.alloc_local((1,), "uint16")
            T.ptx.mov.b32(low_bits[0], high_bits[0], x_words[pair])
            low_x: T.float32 = _cvt_to_f32(low_bits[0], input_dtype)
            high_x: T.float32 = _cvt_to_f32(high_bits[0], input_dtype)
            T.ptx.mov.b64(x_f32_pairs[pair], low_x, high_x)

        x_sq = T.alloc_local((total_values,), "float32")
        for pair in T.unroll(packed_pairs):
            product = T.alloc_local((1,), "uint64")
            T.ptx.mul.f32x2(product[0], x_f32_pairs[pair], x_f32_pairs[pair])
            T.ptx.mov.b64(x_sq[pair * 2], x_sq[pair * 2 + 1], product[0])

        local_sum: T.float32 = T.float32(0.0)
        for value in T.unroll(total_values):
            local_sum = _add_f32(local_sum, x_sq[value])

        local_sum = _butterfly_sum_f32(local_sum, row_lane_xors)
        warp_sum: T.float32 = local_sum

        if warps_per_row > 1 and cluster_n == 1:
            if lane == 0:
                reduce_index: T.int32 = row_warp + warp_in_row * rows
                T.ptx.st.shared.b32(
                    shared_raw.ptr_to([reduce_base + reduce_index * 4]),
                    T.reinterpret("uint32", warp_sum),
                )
            T.ptx.bar.sync(T.uint32(0))
            final_sum: T.float32 = T.float32(0.0)
            if lane < warps_per_row:
                reduce_word = T.alloc_local((1,), "uint32")
                reduce_index: T.int32 = row_warp + lane * rows
                T.ptx.ld.shared.b32(
                    reduce_word[0], shared_raw.ptr_to([reduce_base + reduce_index * 4])
                )
                final_sum = T.reinterpret("float32", reduce_word[0])
            final_sum = _butterfly_sum_f32(final_sum, full_lane_xors)
            sum_sq: T.float32 = final_sum
        elif cluster_n > 1:
            if warp == 0:
                if T.cuda.elect_sync():
                    T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        shared_raw.ptr_to([mbar_offset]), T.uint32(expected_bytes)
                    )
            if lane < cluster_n:
                reduce_index: T.int32 = (
                    row_warp + warp_in_row * rows + cta_rank * rows * warps_per_row
                )
                peer_reduce: T.uint32 = _mapa_u32(
                    shared_raw.ptr_to([reduce_base + reduce_index * 4]), lane
                )
                peer_mbar: T.uint32 = _mapa_u32(shared_raw.ptr_to([mbar_offset]), lane)
                T.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.f32(
                    peer_reduce, warp_sum, peer_mbar
                )
            _cluster_mbarrier_wait(shared_raw.ptr_to([mbar_offset]))

            final_sum = T.float32(0.0)
            for iteration in T.unroll(_ceil_div(total_partials_per_row, 32)):
                partial: T.int32 = lane + iteration * 32
                if partial < total_partials_per_row:
                    partial_warp: T.int32 = partial % warps_per_row
                    partial_cta: T.int32 = partial // warps_per_row
                    reduce_index: T.int32 = (
                        row_warp + partial_warp * rows + partial_cta * rows * warps_per_row
                    )
                    reduce_word = T.alloc_local((1,), "uint32")
                    T.ptx.ld.shared.b32(
                        reduce_word[0], shared_raw.ptr_to([reduce_base + reduce_index * 4])
                    )
                    final_sum = _add_f32(final_sum, T.reinterpret("float32", reduce_word[0]))
            final_sum = _butterfly_sum_f32(final_sum, full_lane_xors)
            sum_sq = final_sum
        else:
            sum_sq = warp_sum

        if H & (H - 1) == 0:
            shifted: T.float32 = _fma_rn_f32(sum_sq, T.float32(1.0 / H), runtime_eps)
        else:
            mean_sq: T.float32 = _div_rn_f32(sum_sq, T.float32(H))
            shifted = _add_f32(mean_sq, runtime_eps)
        rstd: T.float32 = _rsqrt_approx_ftz(shifted)

        global_scale_value: T.float32 = T.float32(0.0)
        if use_e4m3_scale:
            scale_bits = T.alloc_local((1,), "uint32")
            T.ptx.ld.global_.b32(scale_bits[0], global_scale.ptr_to([0]))
            global_scale_value = T.reinterpret("float32", scale_bits[0])

        if cluster_n > 1:
            T.ptx.barrier.cluster.arrive.relaxed()
            T.ptx.barrier.cluster.wait()
        else:
            T.ptx.bar.sync(T.uint32(0))

        if row_valid:
            if scale_blocks_per_thread <= 7:
                for scale_iteration in T.unroll(scale_blocks_per_thread):
                    sf_index: T.int32 = thread_in_row + scale_iteration * tpr
                    _process_scale_block(
                        x,
                        weight,
                        y,
                        scales,
                        actual_row,
                        sf_index,
                        rstd,
                        global_scale_value,
                        fp4_max_rcp,
                        H=H,
                        block_size=block_size,
                        scale_format=scale_format,
                        swizzled=swizzled,
                        input_dtype=input_dtype,
                    )
            else:
                scale_iteration: T.int32 = T.int32(0)
                while True:
                    sf_index0: T.int32 = thread_in_row + scale_iteration * tpr
                    _process_scale_block(
                        x,
                        weight,
                        y,
                        scales,
                        actual_row,
                        sf_index0,
                        rstd,
                        global_scale_value,
                        fp4_max_rcp,
                        H=H,
                        block_size=block_size,
                        scale_format=scale_format,
                        swizzled=swizzled,
                        input_dtype=input_dtype,
                    )
                    sf_index1: T.int32 = thread_in_row + (scale_iteration + 1) * tpr
                    _process_scale_block(
                        x,
                        weight,
                        y,
                        scales,
                        actual_row,
                        sf_index1,
                        rstd,
                        global_scale_value,
                        fp4_max_rcp,
                        H=H,
                        block_size=block_size,
                        scale_format=scale_format,
                        swizzled=swizzled,
                        input_dtype=input_dtype,
                    )
                    scale_iteration = _add_s32(scale_iteration, T.int32(2))
                    if scale_blocks_per_thread % 2 == 0:
                        if scale_iteration == scale_blocks_per_thread:
                            break
                    else:
                        if scale_iteration >= scale_blocks_per_thread:
                            break

        if enable_pdl:
            T.ptx.griddepcontrol.launch_dependents()

    @T.prim_func
    def flashinfer_rmsnorm_fp4quant(
        x_ptr: T.handle,
        weight_ptr: T.handle,
        y_ptr: T.handle,
        scale_ptr: T.handle,
        global_scale_ptr: T.handle,
        runtime_M: T.int32,
        runtime_eps: T.float32,
    ):
        x = T.match_buffer(
            x_ptr,
            shape=(T.cast(runtime_M, "int64") * T.int64(H),),
            dtype=input_dtype,
            scope="global",
        )
        weight = T.match_buffer(weight_ptr, shape=(H,), dtype=input_dtype, scope="global")
        y = T.match_buffer(
            y_ptr,
            shape=(T.cast(runtime_M, "int64") * T.int64(H // 2),),
            dtype="uint8",
            scope="global",
        )
        if swizzled:
            scales = T.match_buffer(
                scale_ptr,
                shape=(
                    T.cast(T.ceildiv(runtime_M, T.int32(128)), "int64")
                    * T.int64(_ceil_div(scale_columns, 4) * 512),
                ),
                dtype="uint8",
                scope="global",
            )
        else:
            scales = T.match_buffer(
                scale_ptr,
                shape=(T.cast(runtime_M, "int64") * T.int64(scale_columns),),
                dtype="uint8",
                scope="global",
            )
        global_scale = T.match_buffer(global_scale_ptr, shape=(1,), dtype="float32", scope="global")
        T.device_entry()
        kernel_body(x, weight, y, scales, global_scale, runtime_M, runtime_eps)

    launch_params = ["blockIdx.x"]
    if cluster_n > 1:
        launch_params.extend(["blockIdx.y", "clusterCtaIdx.x", "clusterCtaIdx.y"])
    launch_params.append("threadIdx.x")
    if enable_pdl:
        launch_params.append("tirx.use_programtic_dependent_launch")
    launch_params.append("tirx.use_dyn_shared_memory")
    return flashinfer_rmsnorm_fp4quant.with_attr("tirx.kernel_launch_params", launch_params)


def prepare_data(**config: Any):
    """Prepare deterministic inputs and caller-owned packed outputs."""
    data = _prepare_tensors(dict(config))
    output = _prepare_output(dict(config), initialize=True)
    return data["x"], data["weight"], output["y"], output["scale"], data["global_scale"]


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
    x = x_backing[: M * H].view(_logical_shape(config))
    if config.get("data_mode") == "zero":
        x.zero_()
    elif H >= 65536:
        columns = torch.arange(H, device="cuda")
        magnitude = torch.where(columns % 257 < 128, 0.5, 1.0)
        pattern = torch.where(columns % 2 == 0, magnitude, -magnitude).to(dtype)
        x.view(M, H).copy_(pattern.expand(M, H))
    else:
        x.normal_(generator=generator)
    x_backing[M * H :].fill_(1.0)

    weight_backing = torch.empty(H + _GUARD_ELEMENTS, dtype=dtype, device="cuda")
    weight = weight_backing[:H]
    weight.normal_(generator=generator)
    weight_backing[H:].fill_(1.0)

    mode = str(config.get("global_scale_mode", "computed"))
    if mode == "computed":
        x_f32 = x.view(M, H).float()
        rms = x_f32 * torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + float(config["eps"]))
        rms = (rms * weight.float()).to(dtype)
        max_abs = rms.float().abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
        scale_value = 448.0 * 6.0 / float(max_abs.item())
    elif mode in ("none", "one"):
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
        "weight": weight,
        "weight_backing": weight_backing,
        "global_scale": global_scale,
    }


def _scale_storage_size(config: dict[str, Any]) -> int:
    M, H = int(config["M"]), int(config["H"])
    columns = H // int(config["block_size"])
    if bool(config["swizzled"]):
        return _ceil_div(M, 128) * _ceil_div(columns, 4) * 512
    return M * columns


def _prepare_output(config: dict[str, Any], *, initialize: bool) -> dict[str, Any]:
    import torch

    M, H = int(config["M"]), int(config["H"])
    y_size = M * (H // 2)
    scale_size = _scale_storage_size(config)
    y_backing = torch.empty(y_size + _GUARD_ELEMENTS, dtype=torch.uint8, device="cuda")
    scale_backing = torch.empty(scale_size + _GUARD_ELEMENTS, dtype=torch.uint8, device="cuda")
    if initialize:
        y_backing.fill_(_GUARD_BYTE)
        scale_backing.fill_(_GUARD_BYTE)
    else:
        y_backing[y_size:].fill_(_GUARD_BYTE)
        scale_backing[scale_size:].fill_(_GUARD_BYTE)
    y = y_backing[:y_size].view(M, H // 2)
    if bool(config["swizzled"]):
        scale = scale_backing[:scale_size]
    else:
        scale = scale_backing[:scale_size].view(M, H // int(config["block_size"]))
    return {
        "y": y,
        "scale": scale,
        "y_arg": y_backing[:y_size],
        "scale_arg": scale_backing[:scale_size],
        "y_backing": y_backing,
        "scale_backing": scale_backing,
        "y_size": y_size,
        "scale_size": scale_size,
        "y_ptr": y.data_ptr(),
        "scale_ptr": scale.data_ptr(),
        "y_stride": y.stride(),
        "scale_stride": scale.stride(),
    }


def _launch_tirx(executable, data, output, config: dict[str, Any]):
    return executable(
        data["x_arg"],
        data["weight"],
        output["y_arg"],
        output["scale_arg"],
        data["global_scale"],
        int(config["M"]),
        float(config["eps"]),
    )


@functools.cache
def _compiled_test_specialization(
    input_dtype: str, H: int, block_size: int, scale_format: str, swizzled: bool, enable_pdl: bool
):
    from tirx_kernels.runner import compile_kernel

    return compile_kernel(
        get_kernel(
            input_dtype=input_dtype,
            M=1,
            H=H,
            block_size=block_size,
            scale_format=scale_format,
            swizzled=swizzled,
            enable_pdl=enable_pdl,
        )
    )


@functools.cache
def _flashinfer_compiled(
    H: int, block_size: int, input_dtype: str, scale_format: str, swizzled: bool, enable_pdl: bool
):
    from flashinfer.cute_dsl.rmsnorm_fp4quant import _get_compiled_kernel

    return _get_compiled_kernel(
        H, block_size, input_dtype == "float16", 100, scale_format, swizzled, enable_pdl
    )


def _launch_flashinfer(data, output, config: dict[str, Any]):
    kernel = _flashinfer_compiled(
        int(config["H"]),
        int(config["block_size"]),
        str(config["input_dtype"]),
        str(config["scale_format"]),
        bool(config["swizzled"]),
        bool(config["enable_pdl"]),
    )
    return kernel(
        data["x2d"],
        data["weight"],
        output["y"],
        output["scale"],
        data["global_scale"],
        int(config["M"]),
        float(config["eps"]),
    )


def _snapshot_inputs(data):
    return {
        "x": data["x"].clone(),
        "weight": data["weight"].clone(),
        "global_scale": data["global_scale"].clone(),
    }


def _assert_inputs_unchanged(data, snapshot, config: dict[str, Any]) -> None:
    import torch

    if not torch.equal(data["x"], snapshot["x"]):
        raise AssertionError("input tensor was modified")
    if not torch.equal(data["weight"], snapshot["weight"]):
        raise AssertionError("weight tensor was modified")
    if not torch.equal(data["global_scale"], snapshot["global_scale"]):
        raise AssertionError("global-scale tensor was modified")
    M, H = int(config["M"]), int(config["H"])
    if not torch.equal(data["x_backing"][M * H :], torch.ones_like(data["x_backing"][M * H :])):
        raise AssertionError("input guard was modified")
    if not torch.equal(data["weight_backing"][H:], torch.ones_like(data["weight_backing"][H:])):
        raise AssertionError("weight guard was modified")


def _assert_output_integrity(output, *, name: str) -> None:
    import torch

    if output["y"].data_ptr() != output["y_ptr"] or output["y"].stride() != output["y_stride"]:
        raise AssertionError(f"{name} packed-output identity changed")
    if (
        output["scale"].data_ptr() != output["scale_ptr"]
        or output["scale"].stride() != output["scale_stride"]
    ):
        raise AssertionError(f"{name} scale-output identity changed")
    if not torch.all(output["y_backing"][output["y_size"] :] == _GUARD_BYTE):
        raise AssertionError(f"{name} packed-output guard was modified")
    if not torch.all(output["scale_backing"][output["scale_size"] :] == _GUARD_BYTE):
        raise AssertionError(f"{name} scale-output guard was modified")


def _logical_scale_bytes(output, config: dict[str, Any]):
    import torch

    M, H = int(config["M"]), int(config["H"])
    columns = H // int(config["block_size"])
    if not bool(config["swizzled"]):
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


def _dequantize(output, data, config: dict[str, Any]):
    import torch

    M, H = int(config["M"]), int(config["H"])
    packed = output["y"].view(M, H // 2)
    low = packed & 0x0F
    high = packed >> 4
    nibbles = torch.stack((low, high), dim=-1).reshape(M, H).long()
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device="cuda",
        dtype=torch.float32,
    )
    fp4 = table[nibbles]
    scale_bytes = _logical_scale_bytes(output, config)
    if int(config["block_size"]) == 16 or str(config["scale_format"]) == "e4m3":
        scales = scale_bytes.view(torch.float8_e4m3fn).float()
        scales = scales / data["global_scale"].float()
    else:
        scales = torch.pow(2.0, scale_bytes.int() - 127)
    return (fp4.view(M, -1, int(config["block_size"])) * scales[..., None]).reshape(M, H)


def _math_oracle(data, config: dict[str, Any]):
    M, H = int(config["M"]), int(config["H"])
    x = data["x2d"].float()
    normalized = x * (x.square().mean(dim=-1, keepdim=True) + float(config["eps"])).rsqrt()
    return (
        (normalized * data["weight"].float()).to(_torch_input_dtype(config["input_dtype"])).float()
    )


def _assert_raw_equal(actual, expected, config: dict[str, Any], *, name: str) -> None:
    import torch

    if not torch.equal(actual["y"], expected["y"]):
        count = int((actual["y"] != expected["y"]).sum().item())
        raise AssertionError(f"{name}: {count} packed FP4 bytes differ")
    actual_scales = _logical_scale_bytes(actual, config)
    expected_scales = _logical_scale_bytes(expected, config)
    if not torch.equal(actual_scales, expected_scales):
        count = int((actual_scales != expected_scales).sum().item())
        raise AssertionError(f"{name}: {count} logical scale bytes differ")


def _assert_math(output, data, config: dict[str, Any]) -> None:
    import torch

    if (int(config["block_size"]) == 16 or str(config["scale_format"]) == "e4m3") and float(
        data["global_scale"].item()
    ) == 0.0:
        if torch.count_nonzero(output["y"] & 0x77) or torch.count_nonzero(
            _logical_scale_bytes(output, config)
        ):
            raise AssertionError("zero global scale must produce signed-zero FP4 values and scales")
        return

    dequantized = _dequantize(output, data, config)
    oracle = _math_oracle(data, config)
    if not torch.isfinite(dequantized).all():
        raise AssertionError("dequantized TIRx output contains non-finite values")
    torch.testing.assert_close(
        dequantized,
        oracle,
        rtol=0.5,
        atol=2.0,
        msg=lambda message: f"independent FP32 RMSNorm oracle: {message}",
    )


def _check_public_allocation(data, reference, config: dict[str, Any]) -> None:
    import torch
    from flashinfer.norm import rmsnorm_fp4quant

    if str(config.get("allocation")) not in ("auto", "compare"):
        return
    y, scales = rmsnorm_fp4quant(
        data["x"],
        data["weight"],
        global_scale=(None if config.get("global_scale_mode") == "none" else data["global_scale"]),
        eps=float(config["eps"]),
        block_size=int(config["block_size"]),
        scale_format=str(config["scale_format"]),
        is_sf_swizzled_layout=bool(config["swizzled"]),
        enable_pdl=bool(config["enable_pdl"]),
    )
    if tuple(y.shape) != (*_logical_shape(config)[:-1], int(config["H"]) // 2):
        raise AssertionError(f"unexpected public packed output shape {tuple(y.shape)}")
    if not torch.equal(y.view(torch.uint8).view(reference["y"].shape), reference["y"]):
        raise AssertionError("public auto-allocated packed output differs from kernel oracle")
    if bool(config["swizzled"]):
        public_scale = {**reference, "scale": scales.view(torch.uint8)}
        public_logical = _logical_scale_bytes(public_scale, config)
    else:
        public_logical = scales.view(torch.uint8).view_as(reference["scale"])
    if not torch.equal(public_logical, _logical_scale_bytes(reference, config)):
        raise AssertionError("public auto-allocated scale output differs from kernel oracle")


def run_test(**config: Any) -> None:
    """Compile, launch, and validate one source-domain specialization."""
    import torch

    config = dict(config)
    _validate(
        str(config["input_dtype"]),
        int(config["H"]),
        int(config["block_size"]),
        str(config["scale_format"]),
        int(config["M"]),
    )
    data = _prepare_tensors(config)
    snapshot = _snapshot_inputs(data)
    output = _prepare_output(config, initialize=True)
    reference = _prepare_output(config, initialize=True)
    executable = _compiled_test_specialization(
        str(config["input_dtype"]),
        int(config["H"]),
        int(config["block_size"]),
        str(config["scale_format"]),
        bool(config["swizzled"]),
        bool(config["enable_pdl"]),
    )
    if _launch_tirx(executable, data, output, config) is not None:
        raise AssertionError("TIRx RMSNormFP4Quant ABI must return None")
    if _launch_flashinfer(data, reference, config) is not None:
        raise AssertionError("FlashInfer RMSNormFP4Quant ABI must return None")
    torch.cuda.synchronize()
    _assert_raw_equal(output, reference, config, name="FlashInfer raw-byte oracle")
    _assert_math(output, data, config)
    _assert_inputs_unchanged(data, snapshot, config)
    _assert_output_integrity(output, name="TIRx output")
    _assert_output_integrity(reference, name="FlashInfer output")
    _check_public_allocation(data, reference, config)


def prepare_bench(**config: Any):
    """Compile the specialization before the bench suite assigns a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(config), "executable": compile_kernel(get_kernel(**config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs: Any
):
    """Build and prevalidate source and TIRx single-launch closures."""
    import torch

    config = dict(prepared["config"])
    config.update(kwargs)
    data = _prepare_tensors(config)
    tirx_output = _prepare_output(config, initialize=False)
    source_output = _prepare_output(config, initialize=False)
    executable = prepared["executable"]

    def tirx_launch():
        return _launch_tirx(executable, data, tirx_output, config)

    if tirx_launch() is not None:
        raise AssertionError("TIRx benchmark closure must return None")
    torch.cuda.synchronize()

    def build_flashinfer_reference():
        def source_launch():
            return _launch_flashinfer(data, source_output, config)

        if source_launch() is not None:
            raise AssertionError("FlashInfer benchmark closure must return None")
        torch.cuda.synchronize()
        _assert_raw_equal(tirx_output, source_output, config, name="benchmark raw-byte precheck")
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
