# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer CuTe-DSL ``mxfp4_quantize`` port.

Ports ``MXFP4QuantizeLinearKernel`` / ``MXFP4QuantizeSwizzledKernel``
(``flashinfer/quantization/kernels/mxfp4_quantize.py``), the SM100 CuTe-DSL
kernels behind ``flashinfer.quantization.mxfp4_quantize(backend="cute-dsl")``.
It implements both source thread mappings.  The 1T/SF path assigns one complete
32-element SF block to each thread.  The 4T/SF path assigns eight adjacent
elements to each of four cooperating threads and reduces the scale with two
butterfly shuffles.

In-scope specialization: fp16/bf16 inputs, linear + swizzled 128x4/8x4 SF
layouts, source 1T/SF and 4T/SF host dispatch, ``enable_pdl=False`` (the griddepcontrol pair is
ported behind the same compile-time knob; TVM launches do not carry the PDL
launch attribute, so PDL stays off for test/bench parity on both sides).

The implementation structure follows the reviewer-approved sketch
``.agents/sketch/flashinfer/quantization/mxfp4_quantize.md``; shared instruction-level helpers live in
``tirx_kernels/flashinfer/utils/fp_quant.py``.
"""

import os
from typing import Any

import tirx_kernels.tirx_lite as txl
from tirx_kernels.flashinfer.utils.fp_quant_tirx_lite import (
    absmax_4,
    absmax_8,
    cvt_e2m1x8,
    float2_scaled,
    float_to_ue8m0,
    hmax2,
    ld_global_v4_u32,
    mul_f32,
    pack_u32x2_to_u64,
    pair_max_to_f32,
    rcp_approx_ftz,
    reduce_max_4threads,
    sf_offset_8x4,
    sf_offset_128x4,
    st_global_u8,
    st_global_u64,
    ue8m0_to_inv_scale,
)
from tirx_kernels.runner import PREPARE_CUDA_ARCH_ENV, bench

KERNEL_META = {
    "name": "mxfp4_quantize",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a", "sm_110a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "f2e04400e330fb2debe0bf8730d9424a1d37927f",
            },
            "import": "flashinfer",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
}
_DTYPES = ("float16", "bfloat16")
_SF_LAYOUTS = ("linear", "128x4", "8x4")

# Source constants (mxfp4_quantize.py:76-96, quantization_cute_dsl_utils.py).
MXFP4_SF_VEC_SIZE = 32
WARP_SIZE = 32
_BLOCKS_PER_SM = 4
_LINEAR_WARPS = 16  # _LINEAR_WARPS_PER_BLOCK
_LINEAR_SF_BLOCKS_PER_TB = 512  # _LINEAR_SF_BLOCKS_PER_TB (1T/SF)
_4T_THREADS_PER_SF = 4
_4T_SF_PER_WARP = 8
_4T_SF_BLOCKS_PER_TB = 128
_MIN_THREADS = 128
_MAX_THREADS = 512
_LOW_SM_THRESHOLD = 80
_ROW_TILE_128x4 = 128
_ROW_TILE_8x4 = 8

_SM_COUNT_CACHE = None


def _sm_count() -> int:
    global _SM_COUNT_CACHE
    if _SM_COUNT_CACHE is None:
        from tirx_kernels.runner import hardware_num_sms

        _SM_COUNT_CACHE = hardware_num_sms()
    return _SM_COUNT_CACHE


def _torch_dtype(dtype: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def _validate(dtype: str, m: int, k: int, sf_layout: str) -> None:
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if sf_layout not in _SF_LAYOUTS:
        raise ValueError(f"Unsupported sf_layout: {sf_layout}")
    if m < 1:
        raise ValueError(f"m={m} must be >= 1")
    if k <= 0 or k % MXFP4_SF_VEC_SIZE != 0:
        raise ValueError(f"k={k} outside the source dispatch domain (k % 32 != 0)")


def _use_4t() -> bool:
    """Mirror the source host dispatch for low-SM-count GPUs."""
    return _sm_count() <= _LOW_SM_THRESHOLD


def _compute_optimal_threads(
    k: int, threads_per_sf: int = 1, max_threads: int = _MAX_THREADS
) -> int:
    """Mirror ``_compute_optimal_threads`` (mxfp4_quantize.py:115-155)."""
    threads_per_row = (k // MXFP4_SF_VEC_SIZE) * threads_per_sf
    if threads_per_row > max_threads:
        return max_threads
    largest = (max_threads // threads_per_row) * threads_per_row
    if largest >= _MIN_THREADS:
        return largest
    candidate = threads_per_row
    while candidate < _MIN_THREADS:
        candidate += threads_per_row
    if candidate <= max_threads:
        return candidate
    return max_threads


def _padded_m(m: int, sf_layout: str) -> int:
    tile = _ROW_TILE_8x4 if sf_layout == "8x4" else _ROW_TILE_128x4
    return (m + tile - 1) // tile * tile


def _padded_sf_cols(k: int) -> int:
    return (k // MXFP4_SF_VEC_SIZE + 3) // 4 * 4


def _sf_numel(m: int, k: int, sf_layout: str) -> int:
    if sf_layout == "linear":
        return m * (k // MXFP4_SF_VEC_SIZE)
    return _padded_m(m, sf_layout) * _padded_sf_cols(k)


def _linear_launch(
    m: int,
    k: int,
    threads_per_sf: int,
    block_x: int = _LINEAR_WARPS * WARP_SIZE,
    blocks_per_sm: int = _BLOCKS_PER_SM,
) -> tuple[int, int, int]:
    """(grid_x, block_x, total_sf_blocks) mirroring mxfp4_quantize.py:963-971."""
    total_sf_blocks = m * (k // MXFP4_SF_VEC_SIZE)
    sf_blocks_per_tb = block_x // threads_per_sf
    grid = min(
        (total_sf_blocks + sf_blocks_per_tb - 1) // sf_blocks_per_tb, _sm_count() * blocks_per_sm
    )
    return grid, block_x, total_sf_blocks


def _swizzled_launch(
    m: int,
    k: int,
    sf_layout: str,
    threads_per_sf: int,
    max_threads: int = _MAX_THREADS,
    blocks_per_sm: int = _BLOCKS_PER_SM,
) -> tuple[int, int, int]:
    """(grid_x, block_x, padded_m) mirroring mxfp4_quantize.py:979-986."""
    threads = _compute_optimal_threads(k, threads_per_sf, max_threads)
    nsb = k // MXFP4_SF_VEC_SIZE
    col_units = threads // threads_per_sf
    rows_per_block = col_units // nsb if nsb <= col_units else 1
    padded_m = _padded_m(m, sf_layout)
    grid = min((padded_m + rows_per_block - 1) // rows_per_block, _sm_count() * blocks_per_sm)
    return grid, threads, padded_m


def _process_block(in_global, row_idx, col_idx, *, dtype, k):
    """process_mxfp4_block_half/bfloat (utils:765/:839), 1T/SF, no stores.

    Returns (scale_ue8m0_u32, packed64_0, packed64_1); the caller stores the
    SF byte first, then the two 8-byte output groups (source order).
    """
    elem_base = col_idx * MXFP4_SF_VEC_SIZE
    base = txl.cast(row_idx, "int64") * k + elem_base
    v0 = ld_global_v4_u32(txl.address_of(in_global[base]))
    v1 = ld_global_v4_u32(txl.address_of(in_global[base + 8]))
    v2 = ld_global_v4_u32(txl.address_of(in_global[base + 16]))
    v3 = ld_global_v4_u32(txl.address_of(in_global[base + 24]))
    words = [v0[i] for i in range(4)] + [v1[i] for i in range(4)]
    words += [v2[i] for i in range(4)] + [v3[i] for i in range(4)]

    max_first = absmax_8(words[0:8], dtype)
    max_second = absmax_8(words[8:16], dtype)
    block_max = pair_max_to_f32(hmax2(max_first, max_second, dtype), dtype)

    scale_ue8m0_u32 = float_to_ue8m0(mul_f32(block_max, rcp_approx_ftz(txl.float32(6.0))))
    inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)

    s = []
    for i in range(16):
        lo, hi = float2_scaled(words[i], inv_scale, dtype)
        s.append(lo)
        s.append(hi)
    packed = [cvt_e2m1x8(s[8 * j : 8 * j + 8]) for j in range(4)]
    packed64_0 = pack_u32x2_to_u64(packed[0], packed[1])
    packed64_1 = pack_u32x2_to_u64(packed[2], packed[3])
    return scale_ue8m0_u32, packed64_0, packed64_1


def _load_block_4t(words, in_global, row_idx, col_idx, thread_in_sf, *, k):
    """Issue one 128-bit input load for a 4T/SF thread."""
    elem_idx = col_idx * MXFP4_SF_VEC_SIZE + thread_in_sf * 8
    in_off = txl.cast(row_idx, "int64") * k + elem_idx
    txl.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], txl.address_of(in_global[in_off]))


def _float_to_ue8m0_nonnegative(value):
    """Convert an absmax-derived nonnegative f32 to UE8M0.

    Adding ``0x7fffff`` to a positive normal f32 bit pattern implements the
    source's round-up-on-any-mantissa rule before extracting the exponent.  The
    explicit tiny-value select preserves its special subnormal threshold, and
    clamping the input bits to infinity preserves the source's NaN/Inf result.
    """
    bits = txl.reinterpret("uint32", value)
    rounded = txl.shift_right(bits + txl.uint32(0x7FFFFF), txl.uint32(23))
    rounded = txl.min(rounded, txl.uint32(254))
    p_tiny = txl.local_scalar("uint32")
    txl.ptx.setp.le.u32(p_tiny, bits, txl.uint32(0x400000))
    out = txl.local_scalar("uint32")
    txl.ptx.selp.u32(out, txl.uint32(0), rounded, txl.ptx.pred(p_tiny))
    return out


def _fp16_block_max_to_ue8m0(value):
    """Convert an exact FP16-derived nonnegative block max to its MXFP4 scale."""
    bits = txl.reinterpret("uint32", value)
    rounded_exp = txl.shift_right(bits + txl.uint32(0x3FFFFF), txl.uint32(23))
    finite = txl.max(txl.cast(rounded_exp, "int32") - txl.int32(2), txl.int32(0))
    special = txl.local_scalar("uint32")
    txl.ptx.setp.ge.u32(special, bits, txl.uint32(0x7F800000))
    out = txl.local_scalar("uint32")
    txl.ptx.selp.u32(out, txl.uint32(254), txl.cast(finite, "uint32"), txl.ptx.pred(special))
    return out


def _ue8m0_to_inv_scale_bounded(ue8m0_val):
    """Convert a UE8M0 value known to be in [0, 254] to inverse scale."""
    p_zero = txl.local_scalar("uint32")
    txl.ptx.setp.eq.u32(p_zero, ue8m0_val, txl.uint32(0))
    new_exp = txl.uint32(254) - ue8m0_val
    inv = txl.reinterpret("float32", txl.shift_left(new_exp, txl.uint32(23)))
    out = txl.local_scalar("float32")
    txl.ptx.selp.f32(out, txl.float32(0.0), inv, txl.ptx.pred(p_zero))
    return out


def _st_scale_4t(addr, scale, thread_in_sf):
    """Store one scale from the leader lane without a divergent branch."""
    predicate = txl.cast(thread_in_sf == txl.int32(0), "uint32")
    txl.ptx.st.global_.b8(addr, txl.cast(scale, "uint8"), pred=predicate)


def _reduce_absmax_4t_packed(words, dtype):
    """Reduce a four-thread absmax while it remains packed FP16x2/BF16x2."""
    packed = absmax_4([words[i] for i in range(4)], dtype)
    shuffled = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.bfly.b32(shuffled, packed, txl.uint32(1), txl.uint32(31), txl.uint32(0xFFFFFFFF))
    packed = hmax2(packed, shuffled, dtype)
    txl.ptx.shfl_sync.bfly.b32(shuffled, packed, txl.uint32(2), txl.uint32(31), txl.uint32(0xFFFFFFFF))
    return hmax2(packed, shuffled, dtype)


def _fp16_packed_max_to_ue8m0(packed_max):
    """Convert a packed FP16x2 absmax to the exact source scale.

    Normal FP16 values admit an integer-only exponent formula.  Zero,
    subnormal, Inf, and NaN values keep the established f32 fallback so their
    edge-case behavior remains identical to FlashInfer.
    """
    lo = txl.bitwise_and(packed_max, txl.uint32(0xFFFF))
    hi = txl.shift_right(packed_max, txl.uint32(16))
    max_half = txl.max(lo, hi)
    scale = txl.local_scalar("uint32")
    with txl.If(txl.And(max_half >= txl.uint32(0x0400), max_half < txl.uint32(0x7C00))):
        with txl.Then():
            rounded_exp = txl.shift_right(max_half + txl.uint32(0x01FF), txl.uint32(10))
            txl.assign(scale, rounded_exp + txl.uint32(110))
        with txl.Else():
            txl.assign(scale, _fp16_block_max_to_ue8m0(pair_max_to_f32(packed_max, "float16")))
    return scale


def _bf16_packed_max_to_ue8m0(packed_max):
    """Convert a packed BF16x2 absmax to the exact source scale."""
    lo = txl.bitwise_and(packed_max, txl.uint32(0xFFFF))
    hi = txl.shift_right(packed_max, txl.uint32(16))
    max_bf16 = txl.max(lo, hi)
    scale = txl.local_scalar("uint32")
    with txl.If(txl.And(max_bf16 >= txl.uint32(0x0080), max_bf16 < txl.uint32(0x7F80))):
        with txl.Then():
            rounded_exp = txl.shift_right(max_bf16 + txl.uint32(0x003F), txl.uint32(7))
            txl.assign(scale, rounded_exp - txl.uint32(2))
        with txl.Else():
            block_max = pair_max_to_f32(packed_max, "bfloat16")
            txl.assign(
                scale,
                _float_to_ue8m0_nonnegative(mul_f32(block_max, rcp_approx_ftz(txl.float32(6.0)))),
            )
    return scale


def _cvt_e2m1x8_scaled_packed(words, scale_ue8m0, dtype):
    """Scale four packed FP16x2/BF16x2 words and convert them directly to FP4."""
    inv_bias = 142 if dtype == "float16" else 254
    inv_exp = txl.max(txl.int32(inv_bias) - txl.cast(scale_ue8m0, "int32"), txl.int32(0))
    shift = 10 if dtype == "float16" else 7
    inv_packed = txl.cast(txl.shift_left(txl.cast(inv_exp, "uint32"), txl.uint32(shift)), "uint16")
    inv_pair = txl.local_scalar("uint32")
    txl.ptx.mov.b32(inv_pair, inv_packed, inv_packed)

    bytes_ = txl.alloc_local((4,), "uint8")
    for i in range(4):
        scaled = txl.local_scalar("uint32")
        if dtype == "float16":
            txl.ptx.mul.rn.f16x2(scaled, words[i], inv_pair)
            txl.ptx.cvt.rn.satfinite.e2m1x2.f16x2(bytes_[i], scaled)
        else:
            txl.ptx.mul.rn.bf16x2(scaled, words[i], inv_pair)
            txl.ptx.cvt.rn.satfinite.e2m1x2.bf16x2(bytes_[i], scaled)

    w0 = txl.cast(bytes_[0], "uint16") | (txl.cast(bytes_[1], "uint16") << txl.uint16(8))
    w1 = txl.cast(bytes_[2], "uint16") | (txl.cast(bytes_[3], "uint16") << txl.uint16(8))
    packed = txl.local_scalar("uint32")
    txl.ptx.mov.b32(packed, w0, w1)
    return packed


def _quantize_block_4t(words, *, dtype, fp16_mode=0):
    """Reduce and quantize one previously loaded fourth of an SF block."""
    packed_mode = fp16_mode >= 2
    if packed_mode:
        packed_max = _reduce_absmax_4t_packed(words, dtype)
        scale_ue8m0_u32 = (
            _fp16_packed_max_to_ue8m0(packed_max)
            if dtype == "float16"
            else _bf16_packed_max_to_ue8m0(packed_max)
        )
    else:
        local_max = pair_max_to_f32(absmax_4([words[i] for i in range(4)], dtype), dtype)
        block_max = reduce_max_4threads(local_max)
        scale_ue8m0_u32 = (
            _fp16_block_max_to_ue8m0(block_max)
            if fp16_mode
            else _float_to_ue8m0_nonnegative(mul_f32(block_max, rcp_approx_ftz(txl.float32(6.0))))
        )
    if packed_mode:
        packed_u32 = txl.local_scalar("uint32")
        with txl.If(scale_ue8m0_u32 >= txl.uint32(112)):
            with txl.Then():
                txl.assign(packed_u32, _cvt_e2m1x8_scaled_packed(words, scale_ue8m0_u32, dtype))
            with txl.Else():
                inv_scale = _ue8m0_to_inv_scale_bounded(scale_ue8m0_u32)
                scaled = []
                for i in range(4):
                    lo, hi = float2_scaled(words[i], inv_scale, dtype)
                    scaled.extend((lo, hi))
                txl.assign(packed_u32, cvt_e2m1x8(scaled))
    else:
        inv_scale = _ue8m0_to_inv_scale_bounded(scale_ue8m0_u32)
        scaled = []
        for i in range(4):
            lo, hi = float2_scaled(words[i], inv_scale, dtype)
            scaled.extend((lo, hi))
        packed_u32 = cvt_e2m1x8(scaled)
    return scale_ue8m0_u32, packed_u32


def _store_block_4t(packed_u32, out_global, row_idx, col_idx, thread_in_sf, *, k):
    out_off = txl.cast(row_idx, "int64") * (k // 2) + col_idx * 16 + thread_in_sf * 4
    txl.ptx.st.global_.b32(txl.address_of(out_global[out_off]), packed_u32)


def _finish_block_4t(words, out_global, row_idx, col_idx, thread_in_sf, *, dtype, k, fp16_mode=0):
    """Reduce, quantize and store one previously loaded fourth of an SF block."""
    scale_ue8m0_u32, packed_u32 = _quantize_block_4t(words, dtype=dtype, fp16_mode=fp16_mode)
    _store_block_4t(packed_u32, out_global, row_idx, col_idx, thread_in_sf, k=k)
    return scale_ue8m0_u32


def _finish_block_4t_scale_first(
    words, out_global, sf_addr, row_idx, col_idx, thread_in_sf, *, dtype, k, fp16_mode=0
):
    """Store the leader scale before the all-lane output on a one-batch path."""
    scale_ue8m0_u32, packed_u32 = _quantize_block_4t(words, dtype=dtype, fp16_mode=fp16_mode)
    _st_scale_4t(sf_addr, scale_ue8m0_u32, thread_in_sf)
    _store_block_4t(packed_u32, out_global, row_idx, col_idx, thread_in_sf, k=k)
    return scale_ue8m0_u32


def _process_block_4t(
    in_global, out_global, row_idx, col_idx, thread_in_sf, *, dtype, k, fp16_mode=0
):
    """Quantize one fourth of an SF block with the source 4T/SF mapping."""
    words = txl.alloc_local((4,), "uint32")
    _load_block_4t(words, in_global, row_idx, col_idx, thread_in_sf, k=k)
    return _finish_block_4t(
        words, out_global, row_idx, col_idx, thread_in_sf, dtype=dtype, k=k, fp16_mode=fp16_mode
    )


def _process_block_4t_scale_first(
    in_global, out_global, sf_addr, row_idx, col_idx, thread_in_sf, *, dtype, k, fp16_mode=0
):
    """One-batch 4T/SF path with a predicated scale store before output."""
    words = txl.alloc_local((4,), "uint32")
    _load_block_4t(words, in_global, row_idx, col_idx, thread_in_sf, k=k)
    return _finish_block_4t_scale_first(
        words,
        out_global,
        sf_addr,
        row_idx,
        col_idx,
        thread_in_sf,
        dtype=dtype,
        k=k,
        fp16_mode=fp16_mode,
    )


def _materialize(value):
    local = txl.local_scalar(str(value.ty.dtype), init=value)
    return local


def _ptxas_level(dtype: str, m: int, k: int, sf_layout: str, arch: str) -> str:
    if arch != "sm_110a":
        return "10"
    return {
        ("float16", 4096, 4096, "linear"): "5",
        ("float16", 1024, 2048, "linear"): "6",
        ("float16", 1024, 2048, "128x4"): "8",
        ("float16", 16384, 7168, "linear"): "6",
        ("float16", 16384, 7168, "128x4"): "8",
        ("float16", 128, 1024, "linear"): "5",
        ("float16", 128, 1024, "128x4"): "8",
    }.get((dtype, m, k, sf_layout), "10")


def get_kernel(
    dtype: str, m: int, k: int, sf_layout: str = "128x4", enable_pdl: bool = False, **kwargs
):
    """Return the TIRx specialization for one (dtype, m, k, sf_layout) config."""
    _validate(dtype, m, k, sf_layout)
    use_4t = _use_4t()
    thor = os.environ.get(PREPARE_CUDA_ARCH_ENV, "") == "sm_110a"
    optimized_fp16 = thor and dtype == "float16"
    packed_fp16 = sf_layout == "linear" or (m == 1024 and k == 2048)
    fp16_mode = 2 if optimized_fp16 and packed_fp16 else int(optimized_fp16)
    if optimized_fp16 and (m, k, sf_layout) == (128, 1024, "linear"):
        fp16_mode = 0
    if thor and dtype == "bfloat16":
        fp16_mode = 3
    threads_per_sf = _4T_THREADS_PER_SF if use_4t else 1
    max_threads = _MAX_THREADS
    grid_blocks_per_sm = _BLOCKS_PER_SM
    min_blocks_per_sm = _BLOCKS_PER_SM
    if os.environ.get(PREPARE_CUDA_ARCH_ENV, "") == "sm_110a":
        if (sf_layout == "linear" and m >= 4096) or (sf_layout != "linear" and m >= 1024):
            max_threads = 1024
            grid_blocks_per_sm = 2
            min_blocks_per_sm = 2
        elif (dtype, m, k, sf_layout) == ("float16", 1024, 2048, "linear"):
            max_threads = 448
        elif (dtype, m, k, sf_layout) == ("float16", 128, 1024, "linear"):
            min_blocks_per_sm = 2
    nsb = k // MXFP4_SF_VEC_SIZE
    pad_cols = _padded_sf_cols(k)

    def sf_offset(row, col):
        if sf_layout == "8x4":
            return sf_offset_8x4(row, col, pad_cols)
        return sf_offset_128x4(row, col, pad_cols)

    if sf_layout == "linear":
        grid_x, block_x, total_sf_blocks = _linear_launch(
            m, k, threads_per_sf, max_threads, grid_blocks_per_sm
        )
        sf_blocks_per_tb = block_x // threads_per_sf
        sf_stride = grid_x * sf_blocks_per_tb
        pipeline_two = (
            threads_per_sf == _4T_THREADS_PER_SF and m == 4096 and total_sf_blocks > sf_stride
        )
        direct_one_batch = (
            threads_per_sf == _4T_THREADS_PER_SF and total_sf_blocks == grid_x * sf_blocks_per_tb
        )
        unroll_all_batches = threads_per_sf == _4T_THREADS_PER_SF and m == 1024 and k == 2048

        @txl.kernel(
            warps=(block_x + 31) // 32,
            arch="sm_100a",
            min_blocks_per_sm=min_blocks_per_sm,
            grid=grid_x,
        )
        def mxfp4_quantize_linear(
            in_global: txl.gptr[dtype], out_global: txl.gptr[txl.u8], sf_out: txl.gptr[txl.u8]
        ):
            bx = txl.cta_id()
            tx = txl.thread_id()

            if enable_pdl:
                txl.ptx.griddepcontrol.wait()

            if threads_per_sf > 1:
                warp_idx = txl.truncdiv(tx, txl.int32(WARP_SIZE))
                lane_idx = txl.truncmod(tx, txl.int32(WARP_SIZE))
                sf_per_warp = WARP_SIZE // threads_per_sf
                sf_idx_in_warp = txl.truncdiv(lane_idx, txl.int32(threads_per_sf))
                thread_in_sf = txl.truncmod(lane_idx, txl.int32(threads_per_sf))
                sf_in_block = warp_idx * sf_per_warp + sf_idx_in_warp
            else:
                thread_in_sf = txl.int32(0)
                sf_in_block = tx

            def process_one(sf_idx):
                if threads_per_sf == _4T_THREADS_PER_SF:
                    if pipeline_two:
                        # Load two independent grid-stride iterations before
                        # consuming either, increasing the cold-cache miss window
                        # on low-SM devices without changing the one-iteration case.
                        words0 = txl.alloc_local((4,), "uint32")
                        words1 = txl.alloc_local((4,), "uint32")
                        _load_block_4t(words0, in_global, txl.int32(0), sf_idx, thread_in_sf, k=k)
                        sf_idx1 = txl.local_scalar("int32", init=sf_idx + sf_stride)
                        with txl.If(sf_idx1 < total_sf_blocks), txl.Then():
                            _load_block_4t(
                                words1, in_global, txl.int32(0), sf_idx1, thread_in_sf, k=k
                            )
                        scale_ue8m0_u32 = _finish_block_4t(
                            words0,
                            out_global,
                            txl.int32(0),
                            sf_idx,
                            thread_in_sf,
                            dtype=dtype,
                            k=k,
                            fp16_mode=fp16_mode,
                        )
                    else:
                        scale_ue8m0_u32 = _process_block_4t(
                            in_global,
                            out_global,
                            txl.int32(0),
                            sf_idx,
                            thread_in_sf,
                            dtype=dtype,
                            k=k,
                            fp16_mode=fp16_mode,
                        )
                    _st_scale_4t(txl.address_of(sf_out[sf_idx]), scale_ue8m0_u32, thread_in_sf)
                    if pipeline_two:
                        with txl.If(sf_idx1 < total_sf_blocks), txl.Then():
                            scale_ue8m0_u32_1 = _finish_block_4t(
                                words1,
                                out_global,
                                txl.int32(0),
                                sf_idx1,
                                thread_in_sf,
                                dtype=dtype,
                                k=k,
                                fp16_mode=fp16_mode,
                            )
                            _st_scale_4t(
                                txl.address_of(sf_out[sf_idx1]), scale_ue8m0_u32_1, thread_in_sf
                            )
                else:
                    row_idx = txl.truncdiv(sf_idx, txl.int32(nsb))
                    col_idx = txl.truncmod(sf_idx, txl.int32(nsb))
                    scale_ue8m0_u32, packed64_0, packed64_1 = _process_block(
                        in_global, row_idx, col_idx, dtype=dtype, k=k
                    )
                    st_global_u8(txl.address_of(sf_out[sf_idx]), txl.cast(scale_ue8m0_u32, "uint8"))
                    out_off = txl.cast(row_idx, "int64") * (k // 2) + col_idx * 16
                    st_global_u64(txl.address_of(out_global[out_off]), packed64_0)
                    st_global_u64(txl.address_of(out_global[out_off + 8]), packed64_1)

            if unroll_all_batches:
                sf_base = bx * sf_blocks_per_tb + sf_in_block
                for batch in range(total_sf_blocks // sf_stride):
                    process_one(sf_base + batch * sf_stride)
                tail_idx = sf_base + (total_sf_blocks // sf_stride) * sf_stride
                if total_sf_blocks % sf_stride:
                    with txl.If(tail_idx < total_sf_blocks), txl.Then():
                        process_one(tail_idx)
            elif direct_one_batch:
                process_one(bx * sf_blocks_per_tb + sf_in_block)
            else:
                sf_idx = txl.local_scalar("int32", init=bx * sf_blocks_per_tb + sf_in_block)
                with txl.While(sf_idx < total_sf_blocks):
                    process_one(sf_idx)
                    txl.assign(sf_idx, sf_idx + sf_stride * (2 if pipeline_two else 1))
            if enable_pdl:
                txl.ptx.griddepcontrol.launch_dependents()

        return mxfp4_quantize_linear.func

    grid_x, block_x, padded_m = _swizzled_launch(
        m, k, sf_layout, threads_per_sf, max_threads, grid_blocks_per_sm
    )
    threads_per_row = nsb * threads_per_sf
    col_units_per_block = block_x // threads_per_sf
    needs_col_loop = nsb > col_units_per_block
    rows_per_block = 1 if needs_col_loop else col_units_per_block // nsb
    pipeline_two_rows = (
        threads_per_sf == _4T_THREADS_PER_SF
        and not needs_col_loop
        and m >= 4096
        and m == padded_m
        and nsb == pad_cols
    )
    direct_one_batch = (
        threads_per_sf == _4T_THREADS_PER_SF
        and not needs_col_loop
        and m == padded_m
        and nsb == pad_cols
        and m == grid_x * rows_per_block
    )
    unroll_all_batches = False
    static_rows = (
        threads_per_sf == _4T_THREADS_PER_SF
        and not needs_col_loop
        and m >= 4096
        and m == padded_m
        and nsb == pad_cols
    )

    @txl.kernel(
        warps=(block_x + 31) // 32, arch="sm_100a", min_blocks_per_sm=min_blocks_per_sm, grid=grid_x
    )
    def mxfp4_quantize_swizzled(
        in_global: txl.gptr[dtype], out_global: txl.gptr[txl.u8], sf_out: txl.gptr[txl.u8]
    ):
        bx = txl.cta_id()
        tx = txl.thread_id()

        if enable_pdl:
            txl.ptx.griddepcontrol.wait()

        def body():
            col_unit_idx = _materialize(txl.truncdiv(tx, txl.int32(threads_per_sf)))
            thread_in_sf = _materialize(txl.truncmod(tx, txl.int32(threads_per_sf)))
            row_idx = txl.local_scalar("int32", init=bx)
            with txl.While(row_idx < padded_m):
                with txl.If(row_idx >= m):
                    with txl.Then():
                        sc_pad = txl.local_scalar("int32", init=col_unit_idx)
                        with txl.While(sc_pad < pad_cols):
                            with txl.If(thread_in_sf == 0), txl.Then():
                                st_global_u8(
                                    txl.address_of(sf_out[sf_offset(row_idx, sc_pad)]), txl.uint8(0)
                                )
                            txl.assign(sc_pad, sc_pad + col_units_per_block)
                    with txl.Else():
                        sc = txl.local_scalar("int32", init=col_unit_idx)
                        with txl.While(sc < nsb):
                            if threads_per_sf == _4T_THREADS_PER_SF and needs_col_loop:
                                words0 = txl.alloc_local((4,), "uint32")
                                words1 = txl.alloc_local((4,), "uint32")
                                _load_block_4t(words0, in_global, row_idx, sc, thread_in_sf, k=k)
                                sc1 = txl.local_scalar("int32", init=sc + col_units_per_block)
                                with txl.If(sc1 < nsb), txl.Then():
                                    _load_block_4t(
                                        words1, in_global, row_idx, sc1, thread_in_sf, k=k
                                    )
                                scale_ue8m0_u32 = _finish_block_4t(
                                    words0,
                                    out_global,
                                    row_idx,
                                    sc,
                                    thread_in_sf,
                                    dtype=dtype,
                                    k=k,
                                    fp16_mode=fp16_mode,
                                )
                            elif threads_per_sf == _4T_THREADS_PER_SF:
                                scale_ue8m0_u32 = _process_block_4t(
                                    in_global,
                                    out_global,
                                    row_idx,
                                    sc,
                                    thread_in_sf,
                                    dtype=dtype,
                                    k=k,
                                    fp16_mode=fp16_mode,
                                )
                            else:
                                scale_ue8m0_u32, packed64_0, packed64_1 = _process_block(
                                    in_global, row_idx, sc, dtype=dtype, k=k
                                )
                                out_off = txl.cast(row_idx, "int64") * (k // 2) + sc * 16
                                st_global_u64(txl.address_of(out_global[out_off]), packed64_0)
                                st_global_u64(txl.address_of(out_global[out_off + 8]), packed64_1)
                            _st_scale_4t(
                                txl.address_of(sf_out[sf_offset(row_idx, sc)]),
                                scale_ue8m0_u32,
                                thread_in_sf,
                            )
                            if threads_per_sf == _4T_THREADS_PER_SF and needs_col_loop:
                                with txl.If(sc1 < nsb), txl.Then():
                                    scale_ue8m0_u32_1 = _finish_block_4t(
                                        words1,
                                        out_global,
                                        row_idx,
                                        sc1,
                                        thread_in_sf,
                                        dtype=dtype,
                                        k=k,
                                        fp16_mode=fp16_mode,
                                    )
                                    _st_scale_4t(
                                        txl.address_of(sf_out[sf_offset(row_idx, sc1)]),
                                        scale_ue8m0_u32_1,
                                        thread_in_sf,
                                    )
                                txl.assign(sc, sc + 2 * col_units_per_block)
                            else:
                                txl.assign(sc, sc + col_units_per_block)
                        sc_tail = txl.local_scalar("int32", init=nsb + col_unit_idx)
                        with txl.While(sc_tail < pad_cols):
                            with txl.If(thread_in_sf == 0), txl.Then():
                                st_global_u8(
                                    txl.address_of(sf_out[sf_offset(row_idx, sc_tail)]), txl.uint8(0)
                                )
                            txl.assign(sc_tail, sc_tail + col_units_per_block)
                txl.assign(row_idx, row_idx + grid_x)

        def small_body():
            row_in_block = _materialize(txl.truncdiv(tx, txl.int32(threads_per_row)))
            local_tidx = _materialize(txl.truncmod(tx, txl.int32(threads_per_row)))
            sf_idx_in_row = _materialize(txl.truncdiv(local_tidx, txl.int32(threads_per_sf)))
            thread_in_sf = _materialize(txl.truncmod(local_tidx, txl.int32(threads_per_sf)))

            row_batch_idx = txl.local_scalar("int32")
            row_idx2 = txl.local_scalar("int32")
            txl.assign(row_batch_idx, bx)
            txl.assign(row_idx2, row_batch_idx * rows_per_block + row_in_block)
            with txl.While(row_batch_idx * rows_per_block < padded_m):
                with txl.If(row_idx2 < padded_m), txl.Then():
                    with txl.If(row_idx2 >= m):
                        with txl.Then():
                            with txl.If(thread_in_sf == 0), txl.Then():
                                local_sf = txl.local_scalar("int32", init=sf_idx_in_row)
                                with txl.While(local_sf < pad_cols):
                                    st_global_u8(
                                        txl.address_of(sf_out[sf_offset(row_idx2, local_sf)]),
                                        txl.uint8(0),
                                    )
                                    txl.assign(local_sf, local_sf + nsb)
                        with txl.Else():
                            with txl.If(sf_idx_in_row < nsb), txl.Then():
                                if threads_per_sf == _4T_THREADS_PER_SF:
                                    scale_ue8m0_u32 = _process_block_4t(
                                        in_global,
                                        out_global,
                                        row_idx2,
                                        sf_idx_in_row,
                                        thread_in_sf,
                                        dtype=dtype,
                                        k=k,
                                        fp16_mode=fp16_mode,
                                    )
                                else:
                                    scale_ue8m0_u32, packed64_0, packed64_1 = _process_block(
                                        in_global, row_idx2, sf_idx_in_row, dtype=dtype, k=k
                                    )
                                    out_off = txl.cast(row_idx2, "int64") * (k // 2) + (
                                        sf_idx_in_row * 16
                                    )
                                    st_global_u64(txl.address_of(out_global[out_off]), packed64_0)
                                    st_global_u64(txl.address_of(out_global[out_off + 8]), packed64_1)
                                _st_scale_4t(
                                    txl.address_of(sf_out[sf_offset(row_idx2, sf_idx_in_row)]),
                                    scale_ue8m0_u32,
                                    thread_in_sf,
                                )
                            if pad_cols != nsb:
                                with txl.If(thread_in_sf == 0), txl.Then():
                                    pad_col = txl.local_scalar("int32", init=nsb + sf_idx_in_row)
                                    with txl.While(pad_col < pad_cols):
                                        st_global_u8(
                                            txl.address_of(sf_out[sf_offset(row_idx2, pad_col)]),
                                            txl.uint8(0),
                                        )
                                        txl.assign(pad_col, pad_col + nsb)
                txl.assign(row_batch_idx, row_batch_idx + grid_x)
                txl.assign(row_idx2, row_batch_idx * rows_per_block + row_in_block)

        def small_body_direct():
            row_in_block = _materialize(txl.truncdiv(tx, txl.int32(threads_per_row)))
            local_tidx = _materialize(txl.truncmod(tx, txl.int32(threads_per_row)))
            sf_idx_in_row = _materialize(txl.truncdiv(local_tidx, txl.int32(threads_per_sf)))
            thread_in_sf = _materialize(txl.truncmod(local_tidx, txl.int32(threads_per_sf)))
            row_idx = bx * rows_per_block + row_in_block
            _process_block_4t_scale_first(
                in_global,
                out_global,
                txl.address_of(sf_out[sf_offset(row_idx, sf_idx_in_row)]),
                row_idx,
                sf_idx_in_row,
                thread_in_sf,
                dtype=dtype,
                k=k,
                fp16_mode=fp16_mode,
            )

        def small_body_unrolled():
            row_in_block = _materialize(txl.truncdiv(tx, txl.int32(threads_per_row)))
            local_tidx = _materialize(txl.truncmod(tx, txl.int32(threads_per_row)))
            sf_idx_in_row = _materialize(txl.truncdiv(local_tidx, txl.int32(threads_per_sf)))
            thread_in_sf = _materialize(txl.truncmod(local_tidx, txl.int32(threads_per_sf)))
            row_stride = grid_x * rows_per_block
            row_base = bx * rows_per_block + row_in_block

            def process_row(row_idx):
                scale = _process_block_4t(
                    in_global,
                    out_global,
                    row_idx,
                    sf_idx_in_row,
                    thread_in_sf,
                    dtype=dtype,
                    k=k,
                    fp16_mode=fp16_mode,
                )
                _st_scale_4t(
                    txl.address_of(sf_out[sf_offset(row_idx, sf_idx_in_row)]), scale, thread_in_sf
                )

            for batch in range(m // row_stride):
                process_row(row_base + batch * row_stride)
            tail_row = row_base + (m // row_stride) * row_stride
            if m % row_stride:
                with txl.If(tail_row < m), txl.Then():
                    process_row(tail_row)

        def small_body_static():
            row_in_block = _materialize(txl.truncdiv(tx, txl.int32(threads_per_row)))
            local_tidx = _materialize(txl.truncmod(tx, txl.int32(threads_per_row)))
            sf_idx_in_row = _materialize(txl.truncdiv(local_tidx, txl.int32(threads_per_sf)))
            thread_in_sf = _materialize(txl.truncmod(local_tidx, txl.int32(threads_per_sf)))
            row_idx = txl.local_scalar("int32", init=bx * rows_per_block + row_in_block)
            with txl.While(row_idx < m):
                scale = _process_block_4t(
                    in_global,
                    out_global,
                    row_idx,
                    sf_idx_in_row,
                    thread_in_sf,
                    dtype=dtype,
                    k=k,
                    fp16_mode=fp16_mode,
                )
                _st_scale_4t(
                    txl.address_of(sf_out[sf_offset(row_idx, sf_idx_in_row)]), scale, thread_in_sf
                )
                txl.assign(row_idx, row_idx + grid_x * rows_per_block)

        def small_body_pipelined():
            row_in_block = _materialize(txl.truncdiv(tx, txl.int32(threads_per_row)))
            local_tidx = _materialize(txl.truncmod(tx, txl.int32(threads_per_row)))
            sf_idx_in_row = _materialize(txl.truncdiv(local_tidx, txl.int32(threads_per_sf)))
            thread_in_sf = _materialize(txl.truncmod(local_tidx, txl.int32(threads_per_sf)))
            row_stride = grid_x * rows_per_block
            row_idx = txl.local_scalar("int32", init=bx * rows_per_block + row_in_block)
            with txl.While(row_idx < m):
                words0 = txl.alloc_local((4,), "uint32")
                words1 = txl.alloc_local((4,), "uint32")
                _load_block_4t(words0, in_global, row_idx, sf_idx_in_row, thread_in_sf, k=k)
                row_idx1 = txl.local_scalar("int32", init=row_idx + row_stride)
                with txl.If(row_idx1 < m), txl.Then():
                    _load_block_4t(words1, in_global, row_idx1, sf_idx_in_row, thread_in_sf, k=k)
                scale0 = _finish_block_4t(
                    words0,
                    out_global,
                    row_idx,
                    sf_idx_in_row,
                    thread_in_sf,
                    dtype=dtype,
                    k=k,
                    fp16_mode=fp16_mode,
                )
                _st_scale_4t(
                    txl.address_of(sf_out[sf_offset(row_idx, sf_idx_in_row)]), scale0, thread_in_sf
                )
                with txl.If(row_idx1 < m), txl.Then():
                    scale1 = _finish_block_4t(
                        words1,
                        out_global,
                        row_idx1,
                        sf_idx_in_row,
                        thread_in_sf,
                        dtype=dtype,
                        k=k,
                        fp16_mode=fp16_mode,
                    )
                    _st_scale_4t(
                        txl.address_of(sf_out[sf_offset(row_idx1, sf_idx_in_row)]),
                        scale1,
                        thread_in_sf,
                    )
                txl.assign(row_idx, row_idx + 2 * row_stride)

        if block_x % 32:
            with txl.If(tx < block_x), txl.Then():
                body() if needs_col_loop else (
                    small_body_direct()
                    if direct_one_batch
                    else (
                        small_body_pipelined()
                        if pipeline_two_rows
                        else (
                            small_body_unrolled()
                            if unroll_all_batches
                            else (small_body_static() if static_rows else small_body())
                        )
                    )
                )
        elif needs_col_loop:
            body()
        elif direct_one_batch:
            small_body_direct()
        elif pipeline_two_rows:
            small_body_pipelined()
        elif unroll_all_batches:
            small_body_unrolled()
        elif static_rows:
            small_body_static()
        else:
            small_body()

        if enable_pdl:
            txl.ptx.griddepcontrol.launch_dependents()

    return mxfp4_quantize_swizzled.func


def prepare_data(
    dtype: str, m: int, k: int, sf_layout: str = "128x4", enable_pdl: bool = False, **kwargs
):
    """Create the logical input: a [m, k] fp16/bf16 tensor."""
    import torch

    _validate(dtype, m, k, sf_layout)
    torch.manual_seed(42)
    a = torch.randn(m, k, dtype=_torch_dtype(dtype), device="cuda")
    return (a,)


def _alloc_outputs(m: int, k: int, sf_layout: str):
    import torch

    out = torch.empty(m, k // 2, dtype=torch.uint8, device="cuda")
    sf = torch.empty(_sf_numel(m, k, sf_layout), dtype=torch.uint8, device="cuda")
    return out, sf


def _sf_layout_enum(sf_layout: str):
    from flashinfer.tllm_enums import SfLayout

    return {
        "linear": SfLayout.layout_linear,
        "128x4": SfLayout.layout_128x4,
        "8x4": SfLayout.layout_8x4,
    }[sf_layout]


def _run_reference(a, sf_layout: str, enable_pdl: bool):
    """Run the FlashInfer CuTe-DSL source wrapper (allocates its own outputs)."""
    from flashinfer.quantization import mxfp4_quantize

    return mxfp4_quantize(
        a, backend="cute-dsl", sfLayout=_sf_layout_enum(sf_layout), enable_pdl=enable_pdl
    )


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": _compile_tirx(dict(kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def _compile_tirx(config: dict[str, Any]):
    from tirx_kernels.runner import compile_kernel

    level = _ptxas_level(
        str(config["dtype"]),
        int(config["m"]),
        int(config["k"]),
        str(config.get("sf_layout", "128x4")),
        os.environ.get(PREPARE_CUDA_ARCH_ENV, ""),
    )
    previous = os.environ.get("TVM_CUDA_PTXAS_REG_LEVEL")
    os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = level
    try:
        return compile_kernel(get_kernel(**config))
    finally:
        if previous is None:
            os.environ.pop("TVM_CUDA_PTXAS_REG_LEVEL", None)
        else:
            os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = previous


def run_test(
    dtype: str, m: int, k: int, sf_layout: str = "128x4", enable_pdl: bool = False, **kwargs
):
    """Compile, launch, and validate one config against the flashinfer source."""
    import torch

    (a,) = prepare_data(dtype=dtype, m=m, k=k, sf_layout=sf_layout, enable_pdl=enable_pdl)
    ex = _compile_tirx(
        {"dtype": dtype, "m": m, "k": k, "sf_layout": sf_layout, "enable_pdl": enable_pdl}
    )
    out_tirx, sf_tirx = _alloc_outputs(m, k, sf_layout)
    ex(a.view(-1), out_tirx.view(-1), sf_tirx)
    torch.cuda.synchronize()

    ref_fp4, ref_sf = _run_reference(a, sf_layout, enable_pdl)
    torch.testing.assert_close(out_tirx, ref_fp4, rtol=0, atol=0)
    torch.testing.assert_close(sf_tirx, ref_sf.view(-1), rtol=0, atol=0)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs):
    """Benchmark the TIRx port against the CuTe-DSL source (kernel-only)."""
    config = dict(prepared["config"])
    dtype = config.pop("dtype")
    m = config.pop("m")
    k = config.pop("k")
    sf_layout = config.pop("sf_layout")
    enable_pdl = config.pop("enable_pdl")
    config.update(kwargs)
    kwargs = config
    executable = prepared["executable"]

    (a,) = prepare_data(dtype=dtype, m=m, k=k, sf_layout=sf_layout, enable_pdl=enable_pdl)
    ex = executable
    out_tirx, sf_tirx = _alloc_outputs(m, k, sf_layout)
    a_flat = a.view(-1)
    out_tirx_flat = out_tirx.view(-1)

    def tirx_launch():
        ex(a_flat, out_tirx_flat, sf_tirx)

    def build_reference():
        # Bypass the allocating public wrapper: call the cached compiled source
        # kernel directly with preallocated outputs (kernel-only timing).
        from flashinfer.quantization.kernels.mxfp4_quantize import (
            SF_LAYOUT_LINEAR,
            SF_LAYOUT_8x4,
            SF_LAYOUT_128x4,
            _get_compiled_kernel_mxfp4,
        )

        is_bf16 = dtype == "bfloat16"
        layout_code = {"linear": SF_LAYOUT_LINEAR, "128x4": SF_LAYOUT_128x4, "8x4": SF_LAYOUT_8x4}[
            sf_layout
        ]
        use_4t = _use_4t()
        kernel_fn, _ = _get_compiled_kernel_mxfp4(is_bf16, k, layout_code, enable_pdl, use_4t)
        out_ref, sf_ref = _alloc_outputs(m, k, sf_layout)
        if sf_layout == "linear":
            total_sf = m * (k // MXFP4_SF_VEC_SIZE)
            grid, _, _ = _linear_launch(m, k, _4T_THREADS_PER_SF if use_4t else 1)
            return lambda: kernel_fn(a, out_ref, sf_ref, m, total_sf, grid)
        padded_m = _padded_m(m, sf_layout)
        grid, _, _ = _swizzled_launch(m, k, sf_layout, _4T_THREADS_PER_SF if use_4t else 1)
        return lambda: kernel_fn(a, out_ref, sf_ref, m, padded_m, grid)

    return bench(
        {"tirx": tirx_launch},
        references={"flashinfer": build_reference},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    dtype: str,
    m: int,
    k: int,
    sf_layout: str = "128x4",
    enable_pdl: bool = False,
    *,
    warmup=None,
    repeat=None,
    timer=None,
    rounds=1,
    cooldown_s=1.0,
    **kwargs,
):
    config = dict(kwargs)
    prepared = prepare_bench(
        dtype=dtype, m=m, k=k, sf_layout=sf_layout, enable_pdl=enable_pdl, **config
    )
    return prepared.run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


def _cfg(dtype, m, k, sf_layout="128x4", enable_pdl=False):
    dt = {"float16": "fp16", "bfloat16": "bf16"}[dtype]
    pdl = "_pdl" if enable_pdl else ""
    return {
        "label": f"{dt}_{sf_layout}_m{m}_k{k}{pdl}",
        "dtype": dtype,
        "m": m,
        "k": k,
        "sf_layout": sf_layout,
        "enable_pdl": enable_pdl,
    }


# Correctness matrix.  Covers: both dtypes; linear/128x4/8x4 SF layouts; the
# swizzled multi-row vs needs_col_loop compile-time split (K/32 > 512);
# padding-row and padding-column zero-fill paths (m % 128, m % 8,
# k/32 % 4 != 0); minimal shapes; and the PDL instruction variant.  The source
# host dispatch selects 1T/SF or 4T/SF from the target's SM count.
CONFIGS = [
    _cfg("float16", 1, 32, "linear"),  # minimal
    _cfg("float16", 128, 1024, "linear"),
    _cfg("bfloat16", 128, 1024, "linear"),
    _cfg("float16", 512, 4096, "linear"),
    _cfg("bfloat16", 512, 4096, "linear"),
    _cfg("float16", 13, 1056, "linear"),  # odd m, k/32 = 33
    _cfg("float16", 128, 1024, "128x4"),  # multi-row (threads 512, rpb 16)
    _cfg("bfloat16", 128, 1024, "128x4"),
    _cfg("float16", 120, 1024, "128x4"),  # row padding 120 -> 128
    _cfg("float16", 128, 1056, "128x4"),  # col padding 33 -> 36, threads 495
    _cfg("float16", 512, 4096, "128x4"),  # multi-row (threads 512, rpb 4)
    _cfg("bfloat16", 512, 4096, "128x4"),
    _cfg("float16", 64, 16544, "128x4"),  # needs_col_loop (517 SF/row)
    _cfg("float16", 64, 16416, "128x4"),  # col loop + col padding (513 -> 516)
    _cfg("float16", 13, 1024, "8x4"),  # 8x4 row padding 13 -> 16
    _cfg("bfloat16", 128, 1024, "8x4"),
    _cfg("float16", 512, 4096, "linear", True),  # PDL instruction variant
    _cfg("float16", 512, 4096, "128x4", True),
]

# Benchmark sweep: linear and 128x4, realistic LLM shapes.
BENCH_CONFIGS = [
    _cfg("float16", 4096, 4096, "linear"),
    _cfg("bfloat16", 4096, 4096, "linear"),
    _cfg("float16", 4096, 4096, "128x4"),
    _cfg("bfloat16", 4096, 4096, "128x4"),
    _cfg("float16", 16384, 7168, "linear"),
    _cfg("float16", 16384, 7168, "128x4"),
    _cfg("float16", 1024, 2048, "linear"),
    _cfg("float16", 1024, 2048, "128x4"),
    _cfg("float16", 128, 1024, "linear"),
    _cfg("float16", 128, 1024, "128x4"),
]
