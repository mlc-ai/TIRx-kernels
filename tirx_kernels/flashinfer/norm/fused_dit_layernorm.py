# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400e330fb2debe0bf8730d9424a1d37927f), Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer fused DIT LayerNorm port.

The primary source is ``meta_fused_layernorm`` in
``include/flashinfer/norm/fused_dit_layernorm.cuh``.  The source dispatch and
public interfaces are in ``csrc/norm.cu``, ``csrc/flashinfer_norm_binding.cu``,
``flashinfer/norm/__init__.py``, and ``flashinfer/diffusion_ops/__init__.py``.
"""

from __future__ import annotations

import functools
from typing import Any

from tirx_kernels.runner import bench
from tvm.script import tirx as T

KERNEL_META = {
    "name": "flashinfer_fused_dit_layernorm",
    "category": "flashinfer",
    "compute_capability": 10,
}

_HIDDEN_SIZE = 3072
_BLOCK_SIZE = 384
_DEFAULT_EPSILON = 1e-6
_MODES = ("grgb", "rss", "grss")
_OUTPUT_FORMATS = ("bf16", "nvfp4", "mxfp8")
_FULL_MASK = 0xFFFFFFFF
_AUXILIARY_STRIDE = 6 * _HIDDEN_SIZE
_SF_SENTINEL = 0xA5
_GUARD_ELEMENTS = 64
_BF16_GUARD = 7.25


def _load_global_bf16x8(buffer, index):
    words = T.alloc_local((4,), "uint32")
    values = T.alloc_local((8,), "float32")
    bits = T.alloc_local((8,), "uint16")
    T.evaluate(
        T.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
    )
    for pair in range(4):
        T.evaluate(T.ptx.mov.b32(bits[pair * 2], bits[pair * 2 + 1], words[pair]))
        T.evaluate(T.ptx.cvt.f32.bf16(values[pair * 2], bits[pair * 2]))
        T.evaluate(T.ptx.cvt.f32.bf16(values[pair * 2 + 1], bits[pair * 2 + 1]))
    return values


def _load_global_f32x8(buffer, index):
    words = T.alloc_local((4,), "uint64")
    values = T.alloc_local((8,), "float32")
    T.evaluate(
        T.ptx.ld.global_.v4.b64(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
    )
    for pair in range(4):
        T.evaluate(T.ptx.mov.b64(values[pair * 2], values[pair * 2 + 1], words[pair]))
    return values


def _load_global_f32(buffer, index):
    value = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.ld.global_.b32(value[0], buffer.ptr_to([index])))
    return value[0]


def _pack_f32x2(low, high):
    packed = T.alloc_local((1,), "uint64")
    T.evaluate(T.ptx.mov.b64(packed[0], low, high))
    return packed[0]


def _unpack_f32x2(packed):
    values = T.alloc_local((2,), "float32")
    T.evaluate(T.ptx.mov.b64(values[0], values[1], packed))
    return values


def _f32x2_binary(chain: str, lhs_low, lhs_high, rhs_low, rhs_high):
    out = T.alloc_local((1,), "uint64")
    T.evaluate(T.ptx[chain](out[0], _pack_f32x2(lhs_low, lhs_high), _pack_f32x2(rhs_low, rhs_high)))
    return _unpack_f32x2(out[0])


def _f32x2_fma(lhs_low, lhs_high, rhs_low, rhs_high, acc_low, acc_high):
    out = T.alloc_local((1,), "uint64")
    T.evaluate(
        T.ptx.fma.rn.ftz.f32x2(
            out[0],
            _pack_f32x2(lhs_low, lhs_high),
            _pack_f32x2(rhs_low, rhs_high),
            _pack_f32x2(acc_low, acc_high),
        )
    )
    return _unpack_f32x2(out[0])


def _add_f32(lhs, rhs):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.add.rn.ftz.f32(out[0], lhs, rhs))
    return out[0]


def _mul_f32(lhs, rhs):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.mul.ftz.f32(out[0], lhs, rhs))
    return out[0]


def _fma_f32(lhs, rhs, acc):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.fma.rn.ftz.f32(out[0], lhs, rhs, acc))
    return out[0]


def _neg_f32(value):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.neg.ftz.f32(out[0], value))
    return out[0]


def _max_f32(lhs, rhs):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.max.ftz.f32(out[0], lhs, rhs))
    return out[0]


def _rcp_f32(value):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.rcp.approx.ftz.f32(out[0], value))
    return out[0]


@T.inline
def _rsqrt_accurate(value, result):
    """Transcribe the SM100 ``__frsqrt_rn`` expansion from fresh source PTX."""
    bits: T.int32 = T.reinterpret("int32", value)
    adjusted = T.alloc_local((1,), "int32")
    T.evaluate(T.ptx.add.s32(adjusted[0], bits, T.int32(-0x00800000)))
    if T.reinterpret("uint32", adjusted[0]) <= T.uint32(0x7EFFFFFF):
        mantissa = T.alloc_local((1,), "uint32")
        normalized_bits = T.alloc_local((1,), "uint32")
        exponent_adjust = T.alloc_local((1,), "int32")
        T.evaluate(T.ptx.and_.b32(mantissa[0], T.reinterpret("uint32", bits), 0x00FFFFFF))
        T.evaluate(T.ptx.or_.b32(normalized_bits[0], mantissa[0], T.uint32(0x3F000000)))
        T.evaluate(
            T.ptx.sub.s32(exponent_adjust[0], T.reinterpret("int32", normalized_bits[0]), bits)
        )
        normalized: T.float32 = T.reinterpret("float32", normalized_bits[0])
        seed = T.alloc_local((1,), "float32")
        T.evaluate(T.ptx.rsqrt.approx.ftz.f32(seed[0], normalized))
        seed_sq: T.float32 = _mul_f32(seed[0], seed[0])
        correction0: T.float32 = _fma_f32(seed[0], seed[0], _neg_f32(seed_sq))
        neg_normalized: T.float32 = _neg_f32(normalized)
        correction1: T.float32 = _fma_f32(seed_sq, neg_normalized, T.float32(1.0))
        correction2: T.float32 = _fma_f32(correction0, neg_normalized, correction1)
        correction3: T.float32 = _fma_f32(correction2, T.float32(0.375), T.float32(0.5))
        correction4: T.float32 = _mul_f32(seed[0], correction2)
        refined: T.float32 = _fma_f32(correction3, correction4, seed[0])
        half_adjust = T.alloc_local((1,), "int32")
        result_bits = T.alloc_local((1,), "int32")
        T.evaluate(T.ptx.shr.s32(half_adjust[0], exponent_adjust[0], T.uint32(1)))
        T.evaluate(T.ptx.add.s32(result_bits[0], half_adjust[0], T.reinterpret("int32", refined)))
        result[0] = T.reinterpret("float32", T.reinterpret("uint32", result_bits[0]))
    else:
        T.evaluate(T.ptx.rsqrt.approx.ftz.f32(result[0], value))


def _pack_bf16x8(values):
    words = T.alloc_local((4,), "uint32")
    for pair in range(4):
        T.evaluate(T.ptx.cvt.rn.bf16x2.f32(words[pair], values[pair * 2 + 1], values[pair * 2]))
    return words


def _store_global_v4_b32(buffer, index, words):
    T.evaluate(
        T.ptx.st.global_.v4.b32(buffer.ptr_to([index]), words[0], words[1], words[2], words[3])
    )


def _store_generic_v4_b32(buffer, index, words):
    T.evaluate(T.ptx.st.v4.b32(buffer.ptr_to([index]), words[0], words[1], words[2], words[3]))


def _abs_bf16x2(value):
    out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.abs.bf16x2(out[0], value))
    return out[0]


def _max_bf16x2(lhs, rhs):
    out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.max.bf16x2(out[0], lhs, rhs))
    return out[0]


def _shfl_xor_u32(value, lane_xor: int):
    out = T.alloc_local((1,), "uint32")
    T.evaluate(
        T.ptx.shfl_sync.bfly.b32(
            out[0], value, T.uint32(lane_xor), T.uint32(31), T.uint32(_FULL_MASK)
        )
    )
    return out[0]


def _bf16x2_horizontal_max_to_f32(value):
    bits = T.alloc_local((2,), "uint16")
    maximum = T.alloc_local((1,), "uint16")
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.mov.b32(bits[0], bits[1], value))
    T.evaluate(T.ptx.max.bf16(maximum[0], bits[0], bits[1]))
    T.evaluate(T.ptx.cvt.f32.bf16(out[0], maximum[0]))
    return out[0]


def _quant_group_max(words, output_format: str):
    maximum: T.uint32 = _abs_bf16x2(words[0])
    next_value: T.uint32 = _abs_bf16x2(words[1])
    maximum = _max_bf16x2(maximum, next_value)
    next_value = _abs_bf16x2(words[2])
    maximum = _max_bf16x2(maximum, next_value)
    next_value = _abs_bf16x2(words[3])
    maximum = _max_bf16x2(maximum, next_value)
    peer: T.uint32 = _shfl_xor_u32(maximum, 1)
    maximum = _max_bf16x2(peer, maximum)
    if output_format == "mxfp8":
        peer = _shfl_xor_u32(maximum, 2)
        maximum = _max_bf16x2(peer, maximum)
    return _bf16x2_horizontal_max_to_f32(maximum)


def _widen_and_scale_bf16x8(words, scale):
    bits = T.alloc_local((8,), "uint16")
    widened = T.alloc_local((8,), "float32")
    values = T.alloc_local((8,), "float32")
    for pair in range(4):
        T.evaluate(T.ptx.mov.b32(bits[pair * 2], bits[pair * 2 + 1], words[pair]))
        T.evaluate(T.ptx.cvt.f32.bf16(widened[pair * 2], bits[pair * 2]))
        T.evaluate(T.ptx.cvt.f32.bf16(widened[pair * 2 + 1], bits[pair * 2 + 1]))
        T.evaluate(T.ptx.mul.ftz.f32(values[pair * 2], widened[pair * 2], scale))
        T.evaluate(T.ptx.mul.ftz.f32(values[pair * 2 + 1], widened[pair * 2 + 1], scale))
    return values


def _sf_offset(batch, row, col, runtime_num_rows, num_k_tiles: int):
    return (
        T.cast(batch, "int64")
        * T.ceildiv(T.cast(runtime_num_rows, "int64"), T.int64(128))
        * T.int64(num_k_tiles * 512)
        + T.truncdiv(T.cast(row, "int64"), T.int64(128)) * T.int64(num_k_tiles * 512)
        + T.truncdiv(T.cast(col, "int64"), T.int64(4)) * T.int64(512)
        + T.truncmod(T.cast(row, "int64"), T.int64(32)) * T.int64(16)
        + T.truncdiv(T.truncmod(T.cast(row, "int64"), T.int64(128)), T.int64(32)) * T.int64(4)
        + T.truncmod(T.cast(col, "int64"), T.int64(4))
    )


@T.inline
def _store_nvfp4(
    output_words, norm_output, sf_output, output_sf_scale, batch, row, tid, runtime_num_rows
):
    global_scale: T.float32 = _load_global_f32(output_sf_scale, T.int64(0))
    vec_max: T.float32 = _quant_group_max(output_words, "nvfp4")
    sf_value: T.float32 = _mul_f32(global_scale, _mul_f32(vec_max, _rcp_f32(T.float32(6.0))))
    sf_pair = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.cvt.rn.satfinite.e4m3x2.f32(sf_pair[0], T.float32(0.0), sf_value))
    sf_col: T.int32 = tid // 2
    sf_offset: T.int64 = _sf_offset(batch, row, sf_col, runtime_num_rows, 48)
    T.evaluate(T.ptx.st.global_.b8(sf_output.ptr_to([sf_offset]), T.cast(sf_pair[0], "uint8")))

    output_scale = T.alloc_local((1,), "float32")
    output_scale[0] = T.float32(0.0)
    if vec_max == T.float32(0.0):
        T.evaluate(T.uint32(0))
    else:
        sf_low = T.alloc_local((1,), "uint16")
        decoded_pair = T.alloc_local((1,), "uint32")
        decoded_low = T.alloc_local((1,), "uint16")
        decoded = T.alloc_local((1,), "float32")
        T.evaluate(T.ptx.and_.b16(sf_low[0], sf_pair[0], T.uint16(0x00FF)))
        T.evaluate(T.ptx.cvt.rn.f16x2.e4m3x2(decoded_pair[0], sf_low[0]))
        T.evaluate(T.ptx.cvt.u16.u32(decoded_low[0], decoded_pair[0]))
        T.evaluate(T.ptx.cvt.f32.f16(decoded[0], decoded_low[0]))
        decoded_over_global: T.float32 = _mul_f32(decoded[0], _rcp_f32(global_scale))
        T.evaluate(T.ptx.rcp.approx.ftz.f32(output_scale[0], decoded_over_global))

    scaled = _widen_and_scale_bf16x8(output_words, output_scale[0])
    packed: T.uint32 = T.cuda.cvt_e2m1x8_f32(
        scaled[0], scaled[1], scaled[2], scaled[3], scaled[4], scaled[5], scaled[6], scaled[7]
    )
    global_row: T.int64 = T.cast(batch, "int64") * T.cast(runtime_num_rows, "int64") + T.cast(
        row, "int64"
    )
    T.evaluate(T.ptx.st.b32(norm_output.ptr_to([global_row * T.int64(384) + tid]), packed))


@T.inline
def _store_mxfp8(output_words, norm_output, sf_output, batch, row, tid, runtime_num_rows):
    vec_max: T.float32 = _quant_group_max(output_words, "mxfp8")
    sf_value: T.float32 = _mul_f32(vec_max, _rcp_f32(T.float32(448.0)))
    sf_pair = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.cvt.rp.satfinite.ue8m0x2.f32(sf_pair[0], T.float32(0.0), sf_value))

    output_scale = T.alloc_local((1,), "float32")
    output_scale[0] = T.float32(0.0)
    if vec_max == T.float32(0.0):
        T.evaluate(T.uint32(0))
    else:
        sf_low = T.alloc_local((1,), "uint16")
        decoded_pair = T.alloc_local((1,), "uint32")
        decoded_bits = T.alloc_local((1,), "uint32")
        T.evaluate(T.ptx.and_.b16(sf_low[0], sf_pair[0], T.uint16(0x00FF)))
        T.evaluate(T.ptx.cvt.rn.bf16x2.ue8m0x2(decoded_pair[0], sf_low[0]))
        T.evaluate(T.ptx.shl.b32(decoded_bits[0], decoded_pair[0], T.uint32(16)))
        T.evaluate(
            T.ptx.rcp.approx.ftz.f32(output_scale[0], T.reinterpret("float32", decoded_bits[0]))
        )

    sf_col: T.int32 = tid // 4
    sf_offset: T.int64 = _sf_offset(batch, row, sf_col, runtime_num_rows, 24)
    T.evaluate(T.ptx.st.global_.b8(sf_output.ptr_to([sf_offset]), T.cast(sf_pair[0], "uint8")))

    scaled = _widen_and_scale_bf16x8(output_words, output_scale[0])
    pairs = T.alloc_local((4,), "uint16")
    wide = T.alloc_local((4,), "uint64")
    for pair in range(4):
        T.evaluate(
            T.ptx.cvt.rn.satfinite.e4m3x2.f32(pairs[pair], scaled[pair * 2 + 1], scaled[pair * 2])
        )
        T.evaluate(T.ptx.cvt.u64.u16(wide[pair], pairs[pair]))
    shifted = T.alloc_local((3,), "uint64")
    packed = T.alloc_local((1,), "uint64")
    T.evaluate(T.ptx.shl.b64(shifted[0], wide[1], T.uint32(16)))
    T.evaluate(T.ptx.or_.b64(packed[0], wide[0], shifted[0]))
    T.evaluate(T.ptx.shl.b64(shifted[1], wide[2], T.uint32(32)))
    T.evaluate(T.ptx.or_.b64(packed[0], packed[0], shifted[1]))
    T.evaluate(T.ptx.shl.b64(shifted[2], wide[3], T.uint32(48)))
    T.evaluate(T.ptx.or_.b64(packed[0], packed[0], shifted[2]))
    global_row: T.int64 = T.cast(batch, "int64") * T.cast(runtime_num_rows, "int64") + T.cast(
        row, "int64"
    )
    T.evaluate(
        T.ptx.st.b64(
            norm_output.ptr_to([global_row * T.int64(768) + T.cast(tid * 2, "int64")]), packed[0]
        )
    )


def _cfg(
    mode: str,
    batch_size: int,
    num_rows: int,
    output_format: str = "bf16",
    *,
    use_input_sf_scale: bool = False,
    input_sf_scale: float = 1.0,
    has_residual: bool = True,
    destination_passing: bool = False,
    auxiliary_ndim: int = 3,
    bias_ndim: int = 2,
    suffix: str = "",
) -> dict[str, Any]:
    label = f"{mode}_{output_format}_b{batch_size}_r{num_rows}"
    if use_input_sf_scale:
        label += f"_inputsf{str(input_sf_scale).replace('.', 'p')}"
    if not has_residual:
        label += "_noresidual"
    if destination_passing:
        label += "_destination"
    if auxiliary_ndim == 2:
        label += "_aux2d"
    if bias_ndim == 1:
        label += "_bias1d"
    if suffix:
        label += f"_{suffix}"
    return {
        "label": label,
        "mode": mode,
        "batch_size": batch_size,
        "num_rows": num_rows,
        "output_format": output_format,
        "use_input_sf_scale": use_input_sf_scale,
        "input_sf_scale": input_sf_scale,
        "has_residual": has_residual,
        "destination_passing": destination_passing,
        "auxiliary_ndim": auxiliary_ndim,
        "bias_ndim": bias_ndim,
        "epsilon": _DEFAULT_EPSILON,
    }


_OFFICIAL_SHAPES = ((1, 1920), (1, 768), (2, 1920), (2, 768), (4, 1920))

_OFFICIAL_BF16_CONFIGS = [
    _cfg(mode, batch_size, num_rows) for mode in _MODES for batch_size, num_rows in _OFFICIAL_SHAPES
]

_OFFICIAL_QUANT_CONFIGS = [
    _cfg(mode, batch_size, num_rows, output_format)
    for mode in ("grgb", "rss")
    for output_format in ("nvfp4", "mxfp8")
    for batch_size, num_rows in ((1, 768), (2, 1920))
]

_DESTINATION_CONFIGS = [
    _cfg("grgb", 1, 768, destination_passing=True),
    _cfg("rss", 1, 768, destination_passing=True),
]

_BOUNDARY_CONFIGS = [
    _cfg("rss", 1, 768, has_residual=False),
    _cfg("grgb", 1, 769, suffix="odd_rows"),
    _cfg("grgb", 1, 1, suffix="single_row"),
    _cfg("grss", 1, 768, auxiliary_ndim=2),
    _cfg("rss", 1, 768, bias_ndim=1),
]

_GRSS_QUANT_CONFIGS = [_cfg("grss", 1, 768, output_format) for output_format in ("nvfp4", "mxfp8")]

_INPUT_SCALE_CONFIGS = [
    _cfg(mode, 1, 768, output_format, use_input_sf_scale=True, input_sf_scale=0.75)
    for mode in _MODES
    for output_format in _OUTPUT_FORMATS
]

_SF_PADDING_CONFIGS = [
    _cfg("rss", 1, 129, output_format, suffix="sf_padding_guard")
    for output_format in ("nvfp4", "mxfp8")
]

CONFIGS = [
    *_OFFICIAL_BF16_CONFIGS,
    *_OFFICIAL_QUANT_CONFIGS,
    *_DESTINATION_CONFIGS,
    *_BOUNDARY_CONFIGS,
    *_GRSS_QUANT_CONFIGS,
    *_INPUT_SCALE_CONFIGS,
    *_SF_PADDING_CONFIGS,
]

BENCH_CONFIGS = [
    *[dict(config) for config in _OFFICIAL_BF16_CONFIGS],
    *[_cfg(mode, 1, 768, output_format) for mode in _MODES for output_format in ("nvfp4", "mxfp8")],
    *[_cfg(mode, 1, 768, use_input_sf_scale=True, input_sf_scale=0.75) for mode in _MODES],
]

assert len(CONFIGS) == 43
assert len(BENCH_CONFIGS) == 24
assert len({config["label"] for config in CONFIGS}) == len(CONFIGS)
assert len({config["label"] for config in BENCH_CONFIGS}) == len(BENCH_CONFIGS)


def _validate(
    mode: str,
    batch_size: int,
    num_rows: int,
    output_format: str,
    use_input_sf_scale: bool,
    input_sf_scale: float,
    has_residual: bool,
    destination_passing: bool,
    auxiliary_ndim: int,
    bias_ndim: int,
    epsilon: float,
) -> None:
    del destination_passing
    if mode not in _MODES:
        raise ValueError(f"unsupported mode: {mode}")
    if output_format not in _OUTPUT_FORMATS:
        raise ValueError(f"unsupported output format: {output_format}")
    if batch_size <= 0 or num_rows <= 0:
        raise ValueError(
            f"batch_size and num_rows must be positive, got {batch_size} and {num_rows}"
        )
    if mode != "rss" and not has_residual:
        raise ValueError(f"mode {mode} requires a residual input")
    if auxiliary_ndim not in (2, 3):
        raise ValueError(f"auxiliary_ndim must be 2 or 3, got {auxiliary_ndim}")
    if bias_ndim not in (1, 2):
        raise ValueError(f"bias_ndim must be 1 or 2, got {bias_ndim}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if use_input_sf_scale and input_sf_scale <= 0:
        raise ValueError(f"input_sf_scale must be positive, got {input_sf_scale}")


def get_kernel(
    mode: str,
    batch_size: int,
    num_rows: int,
    output_format: str,
    use_input_sf_scale: bool,
    input_sf_scale: float,
    has_residual: bool,
    destination_passing: bool,
    auxiliary_ndim: int,
    bias_ndim: int,
    epsilon: float = _DEFAULT_EPSILON,
):
    """Return one of the eighteen compile-time source specializations."""
    _validate(
        mode,
        batch_size,
        num_rows,
        output_format,
        use_input_sf_scale,
        input_sf_scale,
        has_residual,
        destination_passing,
        auxiliary_ndim,
        bias_ndim,
        epsilon,
    )

    norm_dtype = "bfloat16" if output_format == "bf16" else "int32"
    norm_values_per_row = _HIDDEN_SIZE if output_format == "bf16" else _HIDDEN_SIZE // 8
    if output_format == "mxfp8":
        norm_values_per_row = _HIDDEN_SIZE // 4
    sf_k_tiles = 1
    if output_format == "nvfp4":
        sf_k_tiles = (_HIDDEN_SIZE + 63) // 64
    elif output_format == "mxfp8":
        sf_k_tiles = (_HIDDEN_SIZE + 127) // 128

    @T.prim_func
    def flashinfer_fused_dit_layernorm(
        input_ptr: T.handle,
        residual_ptr: T.handle,
        gate_ptr: T.handle,
        gate_bias_ptr: T.handle,
        gamma_ptr: T.handle,
        beta_ptr: T.handle,
        scale_ptr: T.handle,
        scale_bias_ptr: T.handle,
        shift_ptr: T.handle,
        shift_bias_ptr: T.handle,
        residual_output_ptr: T.handle,
        norm_output_ptr: T.handle,
        sf_output_ptr: T.handle,
        output_sf_scale_ptr: T.handle,
        input_sf_scale_ptr: T.handle,
        runtime_batch_size: T.int32,
        runtime_num_rows: T.int32,
        runtime_epsilon: T.float32,
        runtime_has_residual: T.int32,
    ):
        T.func_attr({"tir.is_entry_func": True})
        input_buffer = T.match_buffer(
            input_ptr,
            shape=(
                T.cast(runtime_batch_size, "int64")
                * T.cast(runtime_num_rows, "int64")
                * T.int64(_HIDDEN_SIZE),
            ),
            dtype="bfloat16",
            scope="global",
        )
        residual_buffer = T.match_buffer(
            residual_ptr,
            shape=(
                T.cast(runtime_batch_size, "int64")
                * T.cast(runtime_num_rows, "int64")
                * T.int64(_HIDDEN_SIZE),
            ),
            dtype="bfloat16",
            scope="global",
        )
        gate_buffer = T.match_buffer(
            gate_ptr,
            shape=(
                (
                    T.cast(runtime_batch_size, "int64") * T.cast(runtime_num_rows, "int64")
                    - T.int64(1)
                )
                * T.int64(_AUXILIARY_STRIDE)
                + T.int64(_HIDDEN_SIZE),
            ),
            dtype="bfloat16",
            scope="global",
        )
        gate_bias_buffer = T.match_buffer(
            gate_bias_ptr, shape=(_HIDDEN_SIZE,), dtype="float32", scope="global"
        )
        gamma_buffer = T.match_buffer(
            gamma_ptr, shape=(_HIDDEN_SIZE,), dtype="float32", scope="global"
        )
        beta_buffer = T.match_buffer(
            beta_ptr, shape=(_HIDDEN_SIZE,), dtype="float32", scope="global"
        )
        scale_buffer = T.match_buffer(
            scale_ptr,
            shape=(
                (
                    T.cast(runtime_batch_size, "int64") * T.cast(runtime_num_rows, "int64")
                    - T.int64(1)
                )
                * T.int64(_AUXILIARY_STRIDE)
                + T.int64(_HIDDEN_SIZE),
            ),
            dtype="bfloat16",
            scope="global",
        )
        scale_bias_buffer = T.match_buffer(
            scale_bias_ptr, shape=(_HIDDEN_SIZE,), dtype="float32", scope="global"
        )
        shift_buffer = T.match_buffer(
            shift_ptr,
            shape=(
                (
                    T.cast(runtime_batch_size, "int64") * T.cast(runtime_num_rows, "int64")
                    - T.int64(1)
                )
                * T.int64(_AUXILIARY_STRIDE)
                + T.int64(_HIDDEN_SIZE),
            ),
            dtype="bfloat16",
            scope="global",
        )
        shift_bias_buffer = T.match_buffer(
            shift_bias_ptr, shape=(_HIDDEN_SIZE,), dtype="float32", scope="global"
        )
        residual_output_buffer = T.match_buffer(
            residual_output_ptr,
            shape=(
                T.cast(runtime_batch_size, "int64")
                * T.cast(runtime_num_rows, "int64")
                * T.int64(_HIDDEN_SIZE),
            ),
            dtype="bfloat16",
            scope="global",
        )
        norm_output_buffer = T.match_buffer(
            norm_output_ptr,
            shape=(
                T.cast(runtime_batch_size, "int64")
                * T.cast(runtime_num_rows, "int64")
                * T.int64(norm_values_per_row),
            ),
            dtype=norm_dtype,
            scope="global",
        )
        sf_output_buffer = T.match_buffer(
            sf_output_ptr,
            shape=(
                T.cast(runtime_batch_size, "int64")
                * T.ceildiv(T.cast(runtime_num_rows, "int64"), T.int64(128))
                * T.int64(sf_k_tiles * 32 * 4 * 4),
            ),
            dtype="uint8",
            scope="global",
        )
        output_sf_scale_buffer = T.match_buffer(
            output_sf_scale_ptr, shape=(1,), dtype="float32", scope="global"
        )
        input_sf_scale_buffer = T.match_buffer(
            input_sf_scale_ptr, shape=(1,), dtype="float32", scope="global"
        )
        T.device_entry()
        # TIRX_TRANSCRIBE_START flashinfer_fused_dit_layernorm
        row = T.cta_id([runtime_num_rows])
        tid = T.thread_id([_BLOCK_SIZE])
        lane: T.int32 = tid % 32
        warp: T.int32 = tid // 32

        reduce_store = T.alloc_buffer((120,), "uint8", scope="shared", align=8)
        shared_mean = T.alloc_buffer((8,), "uint8", scope="shared", align=8)
        shared_inv_std = T.alloc_buffer((8,), "uint8", scope="shared", align=8)

        if use_input_sf_scale:
            input_scale: T.float32 = _load_global_f32(input_sf_scale_buffer, T.int64(0))
        else:
            input_scale: T.float32 = T.float32(1.0)

        if mode == "grgb":
            gamma_values = _load_global_f32x8(gamma_buffer, T.cast(tid * 8, "int64"))
            beta_values = _load_global_f32x8(beta_buffer, T.cast(tid * 8, "int64"))
        if mode in ("grgb", "grss"):
            gate_bias_values = _load_global_f32x8(gate_bias_buffer, T.cast(tid * 8, "int64"))
        if mode in ("rss", "grss"):
            scale_bias_values = _load_global_f32x8(scale_bias_buffer, T.cast(tid * 8, "int64"))
            shift_bias_values = _load_global_f32x8(shift_bias_buffer, T.cast(tid * 8, "int64"))

        for batch_id in T.serial(runtime_batch_size, unroll=False):
            global_row: T.int64 = T.cast(batch_id, "int64") * T.cast(
                runtime_num_rows, "int64"
            ) + T.cast(row, "int64")
            dense_index: T.int64 = global_row * T.int64(_HIDDEN_SIZE) + T.cast(tid * 8, "int64")
            auxiliary_index: T.int64 = global_row * T.int64(_AUXILIARY_STRIDE) + T.cast(
                tid * 8, "int64"
            )

            input_values = _load_global_bf16x8(input_buffer, dense_index)
            residual_values = T.alloc_local((8,), "float32")
            if runtime_has_residual != 0:
                loaded_residual = _load_global_bf16x8(residual_buffer, dense_index)
                for value in T.unroll(8):
                    residual_values[value] = loaded_residual[value]
            else:
                for value in T.unroll(8):
                    residual_values[value] = T.float32(0.0)

            if mode in ("grgb", "grss"):
                gate_values = _load_global_bf16x8(gate_buffer, auxiliary_index)
                if use_input_sf_scale:
                    biased_gate_values = T.alloc_local((8,), "float32")
                    for pair in T.unroll(4):
                        biased_gate = _f32x2_binary(
                            "add.rn.ftz.f32x2",
                            gate_values[pair * 2],
                            gate_values[pair * 2 + 1],
                            gate_bias_values[pair * 2],
                            gate_bias_values[pair * 2 + 1],
                        )
                        biased_gate_values[pair * 2] = biased_gate[0]
                        biased_gate_values[pair * 2 + 1] = biased_gate[1]
                    scaled_gate_values = T.alloc_local((8,), "float32")
                    for pair in T.unroll(4):
                        scaled_gate = _f32x2_binary(
                            "mul.rn.ftz.f32x2",
                            biased_gate_values[pair * 2],
                            biased_gate_values[pair * 2 + 1],
                            input_scale,
                            input_scale,
                        )
                        scaled_gate_values[pair * 2] = scaled_gate[0]
                        scaled_gate_values[pair * 2 + 1] = scaled_gate[1]
                    for pair in T.unroll(4):
                        updated = _f32x2_fma(
                            input_values[pair * 2],
                            input_values[pair * 2 + 1],
                            scaled_gate_values[pair * 2],
                            scaled_gate_values[pair * 2 + 1],
                            residual_values[pair * 2],
                            residual_values[pair * 2 + 1],
                        )
                        input_values[pair * 2] = updated[0]
                        input_values[pair * 2 + 1] = updated[1]
                else:
                    for pair in T.unroll(4):
                        biased_gate = _f32x2_binary(
                            "add.rn.ftz.f32x2",
                            gate_values[pair * 2],
                            gate_values[pair * 2 + 1],
                            gate_bias_values[pair * 2],
                            gate_bias_values[pair * 2 + 1],
                        )
                        updated = _f32x2_fma(
                            input_values[pair * 2],
                            input_values[pair * 2 + 1],
                            biased_gate[0],
                            biased_gate[1],
                            residual_values[pair * 2],
                            residual_values[pair * 2 + 1],
                        )
                        input_values[pair * 2] = updated[0]
                        input_values[pair * 2 + 1] = updated[1]
            else:
                if use_input_sf_scale:
                    for pair in T.unroll(4):
                        updated = _f32x2_fma(
                            input_values[pair * 2],
                            input_values[pair * 2 + 1],
                            input_scale,
                            input_scale,
                            residual_values[pair * 2],
                            residual_values[pair * 2 + 1],
                        )
                        input_values[pair * 2] = updated[0]
                        input_values[pair * 2 + 1] = updated[1]
                else:
                    for pair in T.unroll(4):
                        updated = _f32x2_binary(
                            "add.rn.ftz.f32x2",
                            input_values[pair * 2],
                            input_values[pair * 2 + 1],
                            residual_values[pair * 2],
                            residual_values[pair * 2 + 1],
                        )
                        input_values[pair * 2] = updated[0]
                        input_values[pair * 2 + 1] = updated[1]

            residual_words = _pack_bf16x8(input_values)
            _store_global_v4_b32(residual_output_buffer, dense_index, residual_words)

            thread_sum: T.float32 = T.float32(0.0)
            thread_sum_sq: T.float32 = T.float32(0.0)
            for pair in T.unroll(4):
                thread_sum = _add_f32(
                    _add_f32(thread_sum, input_values[pair * 2]), input_values[pair * 2 + 1]
                )
                thread_sum_sq = _fma_f32(
                    input_values[pair * 2 + 1],
                    input_values[pair * 2 + 1],
                    _fma_f32(input_values[pair * 2], input_values[pair * 2], thread_sum_sq),
                )

            for stage in T.unroll(5):
                delta: T.int32 = 1 << stage
                peer_sum_word = T.alloc_local((1,), "uint32")
                peer_sq_word = T.alloc_local((1,), "uint32")
                T.evaluate(
                    T.ptx.shfl_sync.down.b32(
                        peer_sum_word[0],
                        T.reinterpret("uint32", thread_sum),
                        T.cast(delta, "uint32"),
                        T.uint32(31),
                        T.uint32(_FULL_MASK),
                    )
                )
                T.evaluate(
                    T.ptx.shfl_sync.down.b32(
                        peer_sq_word[0],
                        T.reinterpret("uint32", thread_sum_sq),
                        T.cast(delta, "uint32"),
                        T.uint32(31),
                        T.uint32(_FULL_MASK),
                    )
                )
                if lane <= 31 - delta:
                    reduced = _f32x2_binary(
                        "add.rn.ftz.f32x2",
                        thread_sum,
                        thread_sum_sq,
                        T.reinterpret("float32", peer_sum_word[0]),
                        T.reinterpret("float32", peer_sq_word[0]),
                    )
                    thread_sum = reduced[0]
                    thread_sum_sq = reduced[1]

            if lane == 0:
                T.evaluate(
                    T.ptx.st.shared.v2.b32(
                        reduce_store.ptr_to([16 + warp * 8]),
                        T.reinterpret("uint32", thread_sum),
                        T.reinterpret("uint32", thread_sum_sq),
                    )
                )
            T.ptx.bar.sync(T.uint32(0))

            if tid == 0:
                for partial_index in T.unroll(11):
                    partial_word = T.alloc_local((1,), "uint64")
                    T.evaluate(
                        T.ptx.ld.shared.b64(
                            partial_word[0], reduce_store.ptr_to([24 + partial_index * 8])
                        )
                    )
                    partial = _unpack_f32x2(partial_word[0])
                    reduced = _f32x2_binary(
                        "add.rn.ftz.f32x2", thread_sum, thread_sum_sq, partial[0], partial[1]
                    )
                    thread_sum = reduced[0]
                    thread_sum_sq = reduced[1]

                mean: T.float32 = _mul_f32(thread_sum, T.float32(1.0 / _HIDDEN_SIZE))
                mean_sq: T.float32 = _mul_f32(thread_sum_sq, T.float32(1.0 / _HIDDEN_SIZE))
                variance: T.float32 = _fma_f32(_neg_f32(mean), mean, mean_sq)
                variance = _max_f32(variance, T.float32(0.0))
                inv_std = T.alloc_local((1,), "float32")
                _rsqrt_accurate(_add_f32(variance, runtime_epsilon), inv_std)
                T.evaluate(
                    T.ptx.st.shared.v2.b32(
                        shared_mean.ptr_to([0]),
                        T.reinterpret("uint32", mean),
                        T.reinterpret("uint32", mean),
                    )
                )
                T.evaluate(
                    T.ptx.st.shared.v2.b32(
                        shared_inv_std.ptr_to([0]),
                        T.reinterpret("uint32", inv_std[0]),
                        T.reinterpret("uint32", inv_std[0]),
                    )
                )
            T.ptx.bar.sync(T.uint32(0))

            mean_word = T.alloc_local((1,), "uint64")
            inv_std_word = T.alloc_local((1,), "uint64")
            T.evaluate(T.ptx.ld.shared.b64(mean_word[0], shared_mean.ptr_to([0])))
            T.evaluate(T.ptx.ld.shared.b64(inv_std_word[0], shared_inv_std.ptr_to([0])))
            mean_pair = _unpack_f32x2(mean_word[0])
            inv_std_pair = _unpack_f32x2(inv_std_word[0])

            for pair in T.unroll(4):
                centered = _f32x2_fma(
                    T.float32(-1.0),
                    T.float32(-1.0),
                    mean_pair[0],
                    mean_pair[1],
                    input_values[pair * 2],
                    input_values[pair * 2 + 1],
                )
                input_values[pair * 2] = centered[0]
                input_values[pair * 2 + 1] = centered[1]

            if mode == "grgb":
                for pair in T.unroll(4):
                    scaled_inv = _f32x2_binary(
                        "mul.rn.ftz.f32x2",
                        inv_std_pair[0],
                        inv_std_pair[1],
                        gamma_values[pair * 2],
                        gamma_values[pair * 2 + 1],
                    )
                    normalized = _f32x2_fma(
                        input_values[pair * 2],
                        input_values[pair * 2 + 1],
                        scaled_inv[0],
                        scaled_inv[1],
                        beta_values[pair * 2],
                        beta_values[pair * 2 + 1],
                    )
                    input_values[pair * 2] = normalized[0]
                    input_values[pair * 2 + 1] = normalized[1]
            else:
                for pair in T.unroll(4):
                    normalized = _f32x2_binary(
                        "mul.rn.ftz.f32x2",
                        input_values[pair * 2],
                        input_values[pair * 2 + 1],
                        inv_std_pair[0],
                        inv_std_pair[1],
                    )
                    input_values[pair * 2] = normalized[0]
                    input_values[pair * 2 + 1] = normalized[1]

            if mode in ("rss", "grss"):
                scale_values = _load_global_bf16x8(scale_buffer, auxiliary_index)
                shift_values = _load_global_bf16x8(shift_buffer, auxiliary_index)
                for pair in T.unroll(4):
                    biased_scale = _f32x2_binary(
                        "add.rn.ftz.f32x2",
                        scale_values[pair * 2],
                        scale_values[pair * 2 + 1],
                        scale_bias_values[pair * 2],
                        scale_bias_values[pair * 2 + 1],
                    )
                    affine_scale = _f32x2_binary(
                        "add.rn.ftz.f32x2",
                        T.float32(1.0),
                        T.float32(1.0),
                        biased_scale[0],
                        biased_scale[1],
                    )
                    affine_shift = _f32x2_binary(
                        "add.rn.ftz.f32x2",
                        shift_values[pair * 2],
                        shift_values[pair * 2 + 1],
                        shift_bias_values[pair * 2],
                        shift_bias_values[pair * 2 + 1],
                    )
                    output_pair = _f32x2_fma(
                        input_values[pair * 2],
                        input_values[pair * 2 + 1],
                        affine_scale[0],
                        affine_scale[1],
                        affine_shift[0],
                        affine_shift[1],
                    )
                    input_values[pair * 2] = output_pair[0]
                    input_values[pair * 2 + 1] = output_pair[1]

            output_words = _pack_bf16x8(input_values)
            if output_format == "bf16":
                _store_generic_v4_b32(norm_output_buffer, dense_index, output_words)
            elif output_format == "nvfp4":
                _store_nvfp4(
                    output_words,
                    norm_output_buffer,
                    sf_output_buffer,
                    output_sf_scale_buffer,
                    batch_id,
                    row,
                    tid,
                    runtime_num_rows,
                )
            else:
                _store_mxfp8(
                    output_words,
                    norm_output_buffer,
                    sf_output_buffer,
                    batch_id,
                    row,
                    tid,
                    runtime_num_rows,
                )

    return flashinfer_fused_dit_layernorm.with_attr(
        "tirx.kernel_launch_params", ["blockIdx.x", "threadIdx.x"]
    )


def prepare_data(**config: Any) -> tuple[Any, ...]:
    """Prepare the complete fixed-ABI argument tuple for one TIRx launch."""
    config = dict(config)
    _validate_config(config)
    data = _prepare_tensors(config)
    output = _prepare_output(config)
    return tuple(_tirx_args(data, output, config))


def run_test(**config: Any) -> None:
    """Compile, launch, and validate one source-domain specialization."""
    import torch

    config = dict(config)
    _validate_config(config)
    data = _prepare_tensors(config)
    snapshot = _snapshot_inputs(data)
    tirx_output = _prepare_output(config)
    source_output = _prepare_output(config)
    executable = _compiled_specialization(
        str(config["mode"]), str(config["output_format"]), bool(config["use_input_sf_scale"])
    )

    if _launch_tirx(executable, data, tirx_output, config) is not None:
        raise AssertionError("TIRx fused DIT LayerNorm ABI must return None")
    source_module = _flashinfer_module()
    if _launch_flashinfer(source_module, data, source_output, config) is not None:
        raise AssertionError("FlashInfer fused DIT LayerNorm ABI must return None")
    torch.cuda.synchronize()

    _assert_outputs_match(tirx_output, source_output, config, name="FlashInfer CUDA oracle")
    _assert_math(tirx_output, data, config)
    _assert_output_integrity(tirx_output, config, name="TIRx output")
    _assert_output_integrity(source_output, config, name="FlashInfer output")
    _check_public_wrappers(source_module, data, source_output, config)
    _assert_inputs_unchanged(data, snapshot)


def prepare_bench(**config: Any):
    """Compile the selected specialization before GPU assignment."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    config = dict(config)
    _validate_config(config)
    state = {
        "config": config,
        "executable": _compiled_specialization(
            str(config["mode"]), str(config["output_format"]), bool(config["use_input_sf_scale"])
        ),
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs: Any
):
    """Construct and prevalidate two direct single-kernel launch closures."""
    import torch

    config = dict(prepared["config"])
    config.update(kwargs)
    _validate_config(config)
    data = _prepare_tensors(config)
    snapshot = _snapshot_inputs(data)
    tirx_output = _prepare_output(config)
    source_output = _prepare_output(config)
    executable = prepared["executable"]

    def tirx_launch():
        return _launch_tirx(executable, data, tirx_output, config)

    if tirx_launch() is not None:
        raise AssertionError("TIRx benchmark closure must return None")
    torch.cuda.synchronize()

    source_module = _flashinfer_module()

    def flashinfer_launch():
        return _launch_flashinfer(source_module, data, source_output, config)

    if flashinfer_launch() is not None:
        raise AssertionError("FlashInfer benchmark closure must return None")
    torch.cuda.synchronize()
    _assert_outputs_match(tirx_output, source_output, config, name="benchmark precheck")
    _check_public_wrappers(source_module, data, source_output, config)
    _assert_inputs_unchanged(data, snapshot)

    return bench(
        {"tirx": tirx_launch},
        references={"flashinfer_cuda": lambda: flashinfer_launch},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    *,
    warmup: float | None = None,
    repeat: float | None = None,
    timer: str | None = None,
    rounds: int = 1,
    cooldown_s: float = 1.0,
    **config: Any,
) -> dict[str, Any]:
    """Benchmark the TIRx kernel against one direct FlashInfer CUDA launch."""
    prepared = prepare_bench(**config)
    return prepared.run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _validate_config(config: dict[str, Any]) -> None:
    _validate(
        str(config["mode"]),
        int(config["batch_size"]),
        int(config["num_rows"]),
        str(config["output_format"]),
        bool(config["use_input_sf_scale"]),
        float(config["input_sf_scale"]),
        bool(config["has_residual"]),
        bool(config["destination_passing"]),
        int(config["auxiliary_ndim"]),
        int(config["bias_ndim"]),
        float(config.get("epsilon", _DEFAULT_EPSILON)),
    )


def _mode_id(mode: str) -> int:
    return {"grgb": 0, "rss": 1, "grss": 2}[mode]


def _output_id(output_format: str) -> int:
    return {"bf16": 0, "nvfp4": 1, "mxfp8": 2}[output_format]


def _auxiliary_positions(mode: str) -> dict[str, int]:
    if mode == "grgb":
        return {"gate": 2}
    if mode == "rss":
        return {"scale": 4, "shift": 3}
    return {"gate": 5, "scale": 1, "shift": 0}


def _auxiliary_view(backing, position: int, config: dict[str, Any]):
    import torch

    batch_size = int(config["batch_size"])
    num_rows = int(config["num_rows"])
    offset = position * _HIDDEN_SIZE
    if int(config["auxiliary_ndim"]) == 2:
        return torch.as_strided(
            backing,
            size=(batch_size * num_rows, _HIDDEN_SIZE),
            stride=(_AUXILIARY_STRIDE, 1),
            storage_offset=offset,
        )
    return torch.as_strided(
        backing,
        size=(batch_size, num_rows, _HIDDEN_SIZE),
        stride=(num_rows * _AUXILIARY_STRIDE, _AUXILIARY_STRIDE, 1),
        storage_offset=offset,
    )


def _auxiliary_arg(backing, position: int, config: dict[str, Any]):
    rows = int(config["batch_size"]) * int(config["num_rows"])
    offset = position * _HIDDEN_SIZE
    size = (rows - 1) * _AUXILIARY_STRIDE + _HIDDEN_SIZE
    return backing[offset : offset + size]


def _bias_view(table, position: int, config: dict[str, Any]):
    if int(config["bias_ndim"]) == 1:
        return table[0, position]
    return table[:, position]


def _prepare_tensors(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    batch_size = int(config["batch_size"])
    num_rows = int(config["num_rows"])
    dense_size = batch_size * num_rows * _HIDDEN_SIZE
    auxiliary_size = batch_size * num_rows * 6 * _HIDDEN_SIZE
    generator = torch.Generator(device="cuda").manual_seed(42)

    input_backing = torch.empty(dense_size + _GUARD_ELEMENTS, dtype=torch.bfloat16, device="cuda")
    residual_backing = torch.empty_like(input_backing)
    input_backing.fill_(_BF16_GUARD)
    residual_backing.fill_(_BF16_GUARD)
    input_tensor = input_backing[:dense_size].view(batch_size, num_rows, _HIDDEN_SIZE)
    residual = residual_backing[:dense_size].view(batch_size, num_rows, _HIDDEN_SIZE)
    input_tensor.normal_(generator=generator)
    residual.normal_(generator=generator)

    auxiliary_backing = torch.empty(
        auxiliary_size + _GUARD_ELEMENTS, dtype=torch.bfloat16, device="cuda"
    )
    auxiliary_backing.fill_(_BF16_GUARD)
    auxiliary_backing[:auxiliary_size].normal_(generator=generator)
    bias_backing = torch.empty(
        6 * _HIDDEN_SIZE + _GUARD_ELEMENTS, dtype=torch.float32, device="cuda"
    )
    bias_backing.fill_(_BF16_GUARD)
    bias_backing[: 6 * _HIDDEN_SIZE].normal_(generator=generator)
    bias_table = bias_backing[: 6 * _HIDDEN_SIZE].view(1, 6, _HIDDEN_SIZE)

    gamma_backing = torch.empty(_HIDDEN_SIZE + _GUARD_ELEMENTS, dtype=torch.float32, device="cuda")
    beta_backing = torch.empty_like(gamma_backing)
    gamma_backing.fill_(_BF16_GUARD)
    beta_backing.fill_(_BF16_GUARD)
    gamma = gamma_backing[:_HIDDEN_SIZE]
    beta = beta_backing[:_HIDDEN_SIZE]
    gamma.normal_(generator=generator)
    beta.normal_(generator=generator)

    data: dict[str, Any] = {
        "input": input_tensor,
        "input_arg": input_backing[:dense_size],
        "input_backing": input_backing,
        "residual": residual,
        "residual_arg": residual_backing[:dense_size],
        "residual_backing": residual_backing,
        "auxiliary_backing": auxiliary_backing,
        "auxiliary_size": auxiliary_size,
        "bias_backing": bias_backing,
        "gamma": gamma,
        "gamma_backing": gamma_backing,
        "beta": beta,
        "beta_backing": beta_backing,
        "source_empty_bf16": torch.empty(0, dtype=torch.bfloat16, device="cuda"),
        "source_empty_f32": torch.empty(0, dtype=torch.float32, device="cuda"),
        "input_sf_scale": torch.tensor(
            [float(config["input_sf_scale"])], dtype=torch.float32, device="cuda"
        ),
    }
    for name, position in _auxiliary_positions(str(config["mode"])).items():
        data[name] = _auxiliary_view(auxiliary_backing, position, config)
        data[f"{name}_arg"] = _auxiliary_arg(auxiliary_backing, position, config)
        data[f"{name}_bias"] = _bias_view(bias_table, position, config)
        data[f"{name}_bias_arg"] = bias_backing[
            position * _HIDDEN_SIZE : (position + 1) * _HIDDEN_SIZE
        ]

    # All pointer slots in the fixed TIRx ABI receive valid, sufficiently large
    # storage even when their compile-time mode does not dereference them.
    default_aux = _auxiliary_arg(auxiliary_backing, 0, config)
    default_bias = bias_backing[:_HIDDEN_SIZE]
    for name in ("gate", "scale", "shift"):
        data.setdefault(f"{name}_arg", default_aux)
        data.setdefault(name, _auxiliary_view(auxiliary_backing, 0, config))
        data.setdefault(f"{name}_bias_arg", default_bias)
        data.setdefault(f"{name}_bias", _bias_view(bias_table, 0, config))

    if str(config["output_format"]) == "nvfp4":
        _, oracle_norm = _math_oracle(data, config)
        maximum = max(float(oracle_norm.abs().max().item()), 1e-12)
        output_scale = (448.0 * 6.0) / maximum
    else:
        output_scale = 0.0
    data["output_sf_scale"] = torch.tensor([output_scale], dtype=torch.float32, device="cuda")
    return data


def _output_layout(config: dict[str, Any]):
    import torch

    batch_size = int(config["batch_size"])
    num_rows = int(config["num_rows"])
    output_format = str(config["output_format"])
    if output_format == "bf16":
        norm_shape = (batch_size, num_rows, _HIDDEN_SIZE)
        norm_dtype = torch.bfloat16
        norm_bytes = batch_size * num_rows * _HIDDEN_SIZE * 2
        num_k_tiles = 1
    elif output_format == "nvfp4":
        norm_shape = (batch_size, num_rows, _HIDDEN_SIZE // 8)
        norm_dtype = torch.int32
        norm_bytes = batch_size * num_rows * (_HIDDEN_SIZE // 2)
        num_k_tiles = 48
    else:
        norm_shape = (batch_size, num_rows, _HIDDEN_SIZE // 4)
        norm_dtype = torch.int32
        norm_bytes = batch_size * num_rows * _HIDDEN_SIZE
        num_k_tiles = 24
    sf_shape = (batch_size, _ceil_div(num_rows, 128), num_k_tiles, 32, 4, 4)
    sf_size = batch_size * _ceil_div(num_rows, 128) * num_k_tiles * 512
    return norm_shape, norm_dtype, norm_bytes, sf_shape, sf_size


def _prepare_output(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    batch_size = int(config["batch_size"])
    num_rows = int(config["num_rows"])
    residual_size = batch_size * num_rows * _HIDDEN_SIZE
    norm_shape, norm_dtype, norm_bytes, sf_shape, sf_size = _output_layout(config)

    residual_backing = torch.empty(
        residual_size + _GUARD_ELEMENTS, dtype=torch.bfloat16, device="cuda"
    )
    residual_backing.fill_(_BF16_GUARD)
    residual = residual_backing[:residual_size].view(batch_size, num_rows, _HIDDEN_SIZE)

    if norm_dtype == torch.bfloat16:
        norm_elements = norm_bytes // 2
        norm_backing = torch.empty(
            norm_elements + _GUARD_ELEMENTS, dtype=torch.bfloat16, device="cuda"
        )
        norm_backing.fill_(_BF16_GUARD)
        norm_arg = norm_backing[:norm_elements]
        norm = norm_arg.view(norm_shape)
        norm_guard_size = norm_elements
    else:
        norm_backing = torch.full(
            (norm_bytes + _GUARD_ELEMENTS,), _SF_SENTINEL, dtype=torch.uint8, device="cuda"
        )
        norm_arg = norm_backing[:norm_bytes].view(torch.int32)
        norm = norm_arg.view(norm_shape)
        norm_guard_size = norm_bytes

    sf_backing = torch.full(
        (sf_size + _GUARD_ELEMENTS,), _SF_SENTINEL, dtype=torch.uint8, device="cuda"
    )
    sf_arg = sf_backing[:sf_size]
    sf = sf_arg.view(sf_shape)
    return {
        "residual": residual,
        "residual_arg": residual_backing[:residual_size],
        "residual_backing": residual_backing,
        "residual_size": residual_size,
        "norm": norm,
        "norm_arg": norm_arg,
        "norm_backing": norm_backing,
        "norm_guard_size": norm_guard_size,
        "sf": sf,
        "sf_arg": sf_arg,
        "sf_backing": sf_backing,
        "sf_size": sf_size,
        "pointers": (residual.data_ptr(), norm.data_ptr(), sf.data_ptr()),
        "shapes": (tuple(residual.shape), tuple(norm.shape), tuple(sf.shape)),
        "strides": (residual.stride(), norm.stride(), sf.stride()),
    }


def _tirx_args(data, output, config: dict[str, Any]) -> list[Any]:
    return [
        data["input_arg"],
        data["residual_arg"] if bool(config["has_residual"]) else data["input_arg"],
        data["gate_arg"],
        data["gate_bias_arg"],
        data["gamma"] if str(config["mode"]) == "grgb" else data["gate_bias_arg"],
        data["beta"] if str(config["mode"]) == "grgb" else data["gate_bias_arg"],
        data["scale_arg"],
        data["scale_bias_arg"],
        data["shift_arg"],
        data["shift_bias_arg"],
        output["residual_arg"],
        output["norm_arg"],
        output["sf_arg"],
        data["output_sf_scale"],
        data["input_sf_scale"],
        int(config["batch_size"]),
        int(config["num_rows"]),
        float(config.get("epsilon", _DEFAULT_EPSILON)),
        int(bool(config["has_residual"])),
    ]


def _launch_tirx(executable, data, output, config: dict[str, Any]):
    return executable(*_tirx_args(data, output, config))


@functools.cache
def _compiled_specialization(mode: str, output_format: str, use_input_sf_scale: bool):
    from tirx_kernels.runner import compile_kernel

    return compile_kernel(
        get_kernel(
            mode=mode,
            batch_size=1,
            num_rows=1,
            output_format=output_format,
            use_input_sf_scale=use_input_sf_scale,
            input_sf_scale=1.0,
            has_residual=True,
            destination_passing=False,
            auxiliary_ndim=3,
            bias_ndim=2,
            epsilon=_DEFAULT_EPSILON,
        )
    )


def _flashinfer_module():
    import flashinfer.norm as flashinfer_norm

    return flashinfer_norm.get_norm_module()


def _launch_flashinfer(module, data, output, config: dict[str, Any]):
    mode = str(config["mode"])
    empty_bf16 = data["source_empty_bf16"]
    empty_f32 = data["source_empty_f32"]
    return module.fused_dit_layernorm(
        data["input"],
        data["residual"] if bool(config["has_residual"]) else empty_bf16,
        data["gate"] if mode in ("grgb", "grss") else empty_bf16,
        data["gate_bias"] if mode in ("grgb", "grss") else empty_f32,
        data["gamma"] if mode == "grgb" else empty_f32,
        data["beta"] if mode == "grgb" else empty_f32,
        data["scale"] if mode in ("rss", "grss") else empty_bf16,
        data["scale_bias"] if mode in ("rss", "grss") else empty_f32,
        data["shift"] if mode in ("rss", "grss") else empty_bf16,
        data["shift_bias"] if mode in ("rss", "grss") else empty_f32,
        output["residual"],
        output["norm"],
        output["sf"],
        data["output_sf_scale"],
        data["input_sf_scale"] if bool(config["use_input_sf_scale"]) else empty_f32,
        float(config.get("epsilon", _DEFAULT_EPSILON)),
        _mode_id(mode),
        _output_id(str(config["output_format"])),
    )


def _snapshot_inputs(data: dict[str, Any]) -> dict[str, Any]:
    return {
        name: data[name].clone()
        for name in (
            "input_backing",
            "residual_backing",
            "auxiliary_backing",
            "bias_backing",
            "gamma_backing",
            "beta_backing",
            "output_sf_scale",
            "input_sf_scale",
        )
    }


def _assert_inputs_unchanged(data: dict[str, Any], snapshot: dict[str, Any]) -> None:
    import torch

    for name, expected in snapshot.items():
        if not torch.equal(data[name], expected):
            count = int((data[name] != expected).sum().item())
            raise AssertionError(f"{name} was modified at {count} element(s)")


def _as_auxiliary_3d(value, config: dict[str, Any]):
    return value.reshape(int(config["batch_size"]), int(config["num_rows"]), _HIDDEN_SIZE)


def _math_oracle(data: dict[str, Any], config: dict[str, Any]):
    import torch

    mode = str(config["mode"])
    value = data["input"].float()
    residual = data["residual"].float() if bool(config["has_residual"]) else torch.zeros_like(value)
    input_scale = float(config["input_sf_scale"]) if bool(config["use_input_sf_scale"]) else 1.0
    if mode in ("grgb", "grss"):
        gate = _as_auxiliary_3d(data["gate"], config).float()
        gate = gate + data["gate_bias"].reshape(-1, _HIDDEN_SIZE)[0].float()
        residual_fp32 = residual + value * (gate * input_scale)
    else:
        residual_fp32 = residual + value * input_scale

    mean = residual_fp32.mean(dim=-1, keepdim=True)
    mean_sq = residual_fp32.square().mean(dim=-1, keepdim=True)
    variance = torch.clamp_min(mean_sq - mean.square(), 0.0)
    normalized = (residual_fp32 - mean) * (variance + float(config["epsilon"])).rsqrt()
    if mode == "grgb":
        normalized = normalized * data["gamma"].float() + data["beta"].float()
    else:
        scale = _as_auxiliary_3d(data["scale"], config).float()
        shift = _as_auxiliary_3d(data["shift"], config).float()
        scale = scale + data["scale_bias"].reshape(-1, _HIDDEN_SIZE)[0].float()
        shift = shift + data["shift_bias"].reshape(-1, _HIDDEN_SIZE)[0].float()
        normalized = normalized * (1.0 + scale) + shift
    return residual_fp32.to(torch.bfloat16), normalized.to(torch.bfloat16)


def _logical_sf_bytes(sf, config: dict[str, Any]):
    import torch

    batch_size = int(config["batch_size"])
    num_rows = int(config["num_rows"])
    output_format = str(config["output_format"])
    columns = _HIDDEN_SIZE // (16 if output_format == "nvfp4" else 32)
    num_k_tiles = 48 if output_format == "nvfp4" else 24
    batches = torch.arange(batch_size, device="cuda", dtype=torch.int64)[:, None, None]
    rows = torch.arange(num_rows, device="cuda", dtype=torch.int64)[None, :, None]
    cols = torch.arange(columns, device="cuda", dtype=torch.int64)[None, None, :]
    offsets = (
        batches * (_ceil_div(num_rows, 128) * num_k_tiles * 512)
        + (rows // 128) * (num_k_tiles * 512)
        + (cols // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + (cols % 4)
    )
    return sf.reshape(-1)[offsets]


def _dequantize(output: dict[str, Any], data: dict[str, Any], config: dict[str, Any]):
    import torch

    batch_size = int(config["batch_size"])
    num_rows = int(config["num_rows"])
    output_format = str(config["output_format"])
    payload = output["norm"].view(torch.uint8)
    scale_bytes = _logical_sf_bytes(output["sf"], config)
    if output_format == "nvfp4":
        payload = payload.view(batch_size, num_rows, _HIDDEN_SIZE // 2)
        nibbles = torch.stack((payload & 0x0F, payload >> 4), dim=-1)
        nibbles = nibbles.reshape(batch_size, num_rows, _HIDDEN_SIZE).long()
        table = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
            device="cuda",
        )
        values = table[nibbles].view(batch_size, num_rows, _HIDDEN_SIZE // 16, 16)
        scales = scale_bytes.contiguous().view(torch.float8_e4m3fn).float()
        scales = scales / data["output_sf_scale"].float()
        return (values * scales[..., None]).reshape(batch_size, num_rows, _HIDDEN_SIZE)

    values = payload.contiguous().view(torch.float8_e4m3fn).float()
    values = values.view(batch_size, num_rows, _HIDDEN_SIZE // 32, 32)
    exponent = scale_bytes.int() - 127
    scales = torch.where(
        scale_bytes == 0,
        torch.zeros_like(scale_bytes, dtype=torch.float32),
        torch.pow(torch.tensor(2.0, device="cuda"), exponent),
    )
    return (values * scales[..., None]).reshape(batch_size, num_rows, _HIDDEN_SIZE)


def _assert_outputs_match(
    actual: dict[str, Any], expected: dict[str, Any], config: dict[str, Any], *, name: str
) -> None:
    import torch

    if not torch.equal(actual["residual"], expected["residual"]):
        count = int((actual["residual"] != expected["residual"]).sum().item())
        raise AssertionError(f"{name}: {count} residual BF16 values differ")
    if str(config["output_format"]) == "bf16":
        if not torch.allclose(actual["norm"], expected["norm"], rtol=1.6e-2, atol=1e-5):
            difference = (actual["norm"].float() - expected["norm"].float()).abs()
            count = int(
                (~torch.isclose(actual["norm"], expected["norm"], rtol=1.6e-2, atol=1e-5))
                .sum()
                .item()
            )
            raise AssertionError(
                f"{name} BF16 norm: {count} values differ; "
                f"maximum absolute difference is {float(difference.max().item())}"
            )
        return
    actual_payload = actual["norm"].view(torch.uint8)
    expected_payload = expected["norm"].view(torch.uint8)
    if not torch.equal(actual_payload, expected_payload):
        count = int((actual_payload != expected_payload).sum().item())
        raise AssertionError(f"{name}: {count} packed quantization bytes differ")
    actual_sf = _logical_sf_bytes(actual["sf"], config)
    expected_sf = _logical_sf_bytes(expected["sf"], config)
    if not torch.equal(actual_sf, expected_sf):
        count = int((actual_sf != expected_sf).sum().item())
        raise AssertionError(f"{name}: {count} logical scale-factor bytes differ")


def _assert_math(output, data, config: dict[str, Any]) -> None:
    import torch

    oracle_residual, oracle_norm = _math_oracle(data, config)
    torch.testing.assert_close(
        output["residual"],
        oracle_residual,
        rtol=1.6e-2,
        atol=1e-5,
        msg=lambda message: f"independent FP32 residual oracle: {message}",
    )
    if str(config["output_format"]) == "bf16":
        torch.testing.assert_close(
            output["norm"],
            oracle_norm,
            rtol=1.6e-2,
            atol=1e-5,
            msg=lambda message: f"independent FP32 LayerNorm oracle: {message}",
        )
    else:
        dequantized = _dequantize(output, data, config)
        if not torch.isfinite(dequantized).all():
            raise AssertionError("dequantized norm output contains non-finite values")
        torch.testing.assert_close(
            dequantized,
            oracle_norm.float(),
            rtol=0.5,
            atol=2.0,
            msg=lambda message: f"independent quantized LayerNorm oracle: {message}",
        )


def _assert_output_integrity(output, config: dict[str, Any], *, name: str) -> None:
    import torch

    actual_pointers = (
        output["residual"].data_ptr(),
        output["norm"].data_ptr(),
        output["sf"].data_ptr(),
    )
    actual_shapes = (
        tuple(output["residual"].shape),
        tuple(output["norm"].shape),
        tuple(output["sf"].shape),
    )
    actual_strides = (output["residual"].stride(), output["norm"].stride(), output["sf"].stride())
    if actual_pointers != output["pointers"]:
        raise AssertionError(f"{name}: an output data pointer changed")
    if actual_shapes != output["shapes"] or actual_strides != output["strides"]:
        raise AssertionError(f"{name}: an output shape or stride changed")
    if not torch.all(output["residual_backing"][output["residual_size"] :] == _BF16_GUARD):
        raise AssertionError(f"{name}: residual trailing guard was modified")
    norm_tail = output["norm_backing"][output["norm_guard_size"] :]
    norm_sentinel = _BF16_GUARD if norm_tail.dtype == torch.bfloat16 else _SF_SENTINEL
    if not torch.all(norm_tail == norm_sentinel):
        raise AssertionError(f"{name}: norm trailing guard was modified")
    if not torch.all(output["sf_backing"][output["sf_size"] :] == _SF_SENTINEL):
        raise AssertionError(f"{name}: scale-factor trailing guard was modified")

    output_format = str(config["output_format"])
    if output_format == "bf16":
        if not torch.all(output["sf_arg"] == _SF_SENTINEL):
            raise AssertionError(f"{name}: disabled scale-factor output was modified")
        return
    valid = _logical_sf_offsets(config)
    padding = torch.ones(output["sf_size"], dtype=torch.bool, device="cuda")
    padding[valid] = False
    if not torch.all(output["sf_arg"][padding] == _SF_SENTINEL):
        raise AssertionError(f"{name}: swizzled scale-factor padding was modified")


def _logical_sf_offsets(config: dict[str, Any]):
    import torch

    output_format = str(config["output_format"])
    columns = _HIDDEN_SIZE // (16 if output_format == "nvfp4" else 32)
    num_k_tiles = 48 if output_format == "nvfp4" else 24
    batch_size = int(config["batch_size"])
    num_rows = int(config["num_rows"])
    batches = torch.arange(batch_size, device="cuda", dtype=torch.int64)[:, None, None]
    rows = torch.arange(num_rows, device="cuda", dtype=torch.int64)[None, :, None]
    cols = torch.arange(columns, device="cuda", dtype=torch.int64)[None, None, :]
    return (
        batches * (_ceil_div(num_rows, 128) * num_k_tiles * 512)
        + (rows // 128) * (num_k_tiles * 512)
        + (cols // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + (cols % 4)
    ).reshape(-1)


def _public_call(api, data, output, config: dict[str, Any]):
    mode = str(config["mode"])
    common = {
        "epsilon": float(config["epsilon"]),
        "use_nvfp4": str(config["output_format"]) == "nvfp4",
        "use_mxfp8": str(config["output_format"]) == "mxfp8",
        "global_scaling_factor": (
            data["output_sf_scale"] if str(config["output_format"]) == "nvfp4" else None
        ),
        "input_global_scaling_factor": (
            data["input_sf_scale"] if bool(config["use_input_sf_scale"]) else None
        ),
    }
    if output is not None:
        common.update(
            residual_out=output["residual"],
            norm_out=output["norm"],
            sf_out=(output["sf"] if str(config["output_format"]) != "bf16" else None),
        )
    if mode == "grgb":
        return api(
            data["input"],
            data["residual"],
            data["gate"],
            data["gamma"],
            data["beta"],
            gate_bias=data["gate_bias"],
            **common,
        )
    if mode == "rss":
        return api(
            data["input"],
            data["scale"],
            data["shift"],
            residual=(data["residual"] if bool(config["has_residual"]) else None),
            scale_bias=data["scale_bias"],
            shift_bias=data["shift_bias"],
            **common,
        )
    return api(
        data["input"],
        data["residual"],
        data["gate"],
        data["scale"],
        data["shift"],
        gate_bias=data["gate_bias"],
        scale_bias=data["scale_bias"],
        shift_bias=data["shift_bias"],
        **common,
    )


def _check_public_wrappers(module, data, source_output, config: dict[str, Any]) -> None:
    import flashinfer.norm as flashinfer_norm

    mode = str(config["mode"])
    api = {
        "grgb": flashinfer_norm.fused_dit_gate_residual_layernorm_gamma_beta,
        "rss": flashinfer_norm.fused_dit_residual_layernorm_scale_shift,
        "grss": flashinfer_norm.fused_dit_gate_residual_layernorm_scale_shift,
    }[mode]
    captures: list[tuple[Any, ...]] = []

    class _TrackedNormModule:
        def __getattr__(self, name):
            return getattr(module, name)

        def fused_dit_layernorm(self, *args, **kwargs):
            captures.append(args)
            return module.fused_dit_layernorm(*args, **kwargs)

    original_get_norm_module = flashinfer_norm.get_norm_module
    flashinfer_norm.get_norm_module = lambda: _TrackedNormModule()
    try:
        auto_return = _public_call(api, data, None, config)
        destination = _prepare_output(config)
        destination_return = _public_call(api, data, destination, config)
    finally:
        flashinfer_norm.get_norm_module = original_get_norm_module

    if len(captures) != 2:
        raise AssertionError(f"public {mode} wrapper dispatched {len(captures)} low-level calls")
    for index, returned in enumerate((auto_return, destination_return)):
        if returned[0] is not captures[index][10] or returned[1] is not captures[index][11]:
            raise AssertionError(f"public {mode} wrapper did not return its dispatched outputs")
    if destination_return[0] is not destination["residual"]:
        raise AssertionError(f"public {mode} residual destination identity changed")
    if destination_return[1] is not destination["norm"]:
        raise AssertionError(f"public {mode} norm destination identity changed")

    auto_output = {"residual": auto_return[0], "norm": auto_return[1], "sf": captures[0][12]}
    _assert_outputs_match(auto_output, source_output, config, name=f"public {mode} auto")
    _assert_outputs_match(destination, source_output, config, name=f"public {mode} destination")
    _assert_output_integrity(destination, config, name=f"public {mode} destination")


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
