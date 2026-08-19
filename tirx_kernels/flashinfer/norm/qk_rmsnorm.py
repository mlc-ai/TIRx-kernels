# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400e330fb2debe0bf8730d9424a1d37927f), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer CuTe-DSL 3-D QK RMSNorm and Gemma RMSNorm port.

The source implementation is ``QKRMSNormKernel`` and
``qk_rmsnorm_cute`` in ``flashinfer/norm/kernels/rmsnorm.py``.  The
public dispatch is in ``flashinfer/norm/__init__.py``.
"""

from __future__ import annotations

from typing import Any

from tirx_kernels.runner import bench
from tvm.script import tirx as T

KERNEL_META = {"name": "flashinfer_qk_rmsnorm", "category": "flashinfer", "compute_capability": 10}

_VARIANT_CODE = {"rmsnorm": "rms", "gemma_rmsnorm": "gemma"}
_DTYPE_CODE = {"float16": "f16", "bfloat16": "bf16"}
_LAYOUT_CODE = {"compact": "c", "strided": "s"}
_DEFAULT_EPS = 1e-6
_INT64_STRIDE = 2**31
_OPTIN_SMEM_BYTES = 232448
_ELEM_BYTES = 2
_GUARD_ELEMENTS = 64
_GUARD_VALUE = 123.0


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _zero_half_values(values, count: int) -> None:
    for index in range(count):
        T.evaluate(T.ptx.mov.b16(values[index], T.uint16(0)))


def _zero_packed_half_values(values, count: int) -> None:
    for pair in range(count):
        T.evaluate(T.ptx.mov.b32(values[pair * 2], values[pair * 2 + 1], T.uint32(0)))


def _zero_float_values(values, count: int) -> None:
    for index in range(count):
        T.evaluate(T.ptx.mov.b32(values[index], T.float32(0)))


def _zero_float_pair_values(values, count: int) -> None:
    zero = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.mov.b32(zero[0], T.float32(0)))
    for pair in range(count):
        T.evaluate(T.ptx.mov.b64(values[pair], zero[0], zero[0]))


def _threads_per_row(H: int) -> int:
    if H <= 64:
        return 8
    if H <= 128:
        return 16
    if H <= 3072:
        return 32
    if H <= 6144:
        return 64
    if H <= 16384:
        return 128
    return 256


def _source_config(H: int) -> dict[str, int | bool]:
    tpr = _threads_per_row(H)
    threads = 128 if H <= 16384 else 256
    rows = threads // tpr
    warps_per_row = max(tpr // 32, 1)
    vec = min(H & -H, 8)
    copy_bits = 16 * vec
    vec_blocks = max(1, _ceil_div(H // vec, tpr))
    cols = vec * vec_blocks * tpr
    tile_bytes = rows * cols * _ELEM_BYTES
    use_async = copy_bits >= 32 and tile_bytes <= _OPTIN_SMEM_BYTES // 2
    smem_bytes = (tile_bytes if use_async else 0) + rows * warps_per_row * 4
    return {
        "tpr": tpr,
        "threads": threads,
        "rows": rows,
        "warps_per_row": warps_per_row,
        "vec": vec,
        "copy_bits": copy_bits,
        "vec_blocks": vec_blocks,
        "cols": cols,
        "tile_bytes": tile_bytes,
        "use_async": use_async,
        "smem_bytes": smem_bytes,
    }


def _ptx_unary(chain: str, value, dtype: str = "float32"):
    out = T.alloc_local((1,), dtype)
    T.evaluate(T.ptx[chain](out[0], value))
    return out[0]


def _ptx_binary(chain: str, lhs, rhs, dtype: str = "float32"):
    out = T.alloc_local((1,), dtype)
    T.evaluate(T.ptx[chain](out[0], lhs, rhs))
    return out[0]


def _ptx_ternary(chain: str, lhs, rhs, acc, dtype: str = "float32"):
    out = T.alloc_local((1,), dtype)
    T.evaluate(T.ptx[chain](out[0], lhs, rhs, acc))
    return out[0]


def _add_f32(lhs, rhs):
    return _ptx_binary("add.f32", lhs, rhs)


def _div_rn_f32(lhs, rhs):
    return _ptx_binary("div.rn.f32", lhs, rhs)


def _fma_rn_f32(lhs, rhs, acc):
    return _ptx_ternary("fma.rn.f32", lhs, rhs, acc)


def _fma_half_to_f32(lhs, rhs, acc, dtype: str):
    suffix = "f16" if dtype == "float16" else "bf16"
    return _ptx_ternary(f"fma.rn.f32.{suffix}", lhs, rhs, acc)


def _rsqrt_approx_ftz(value):
    return _ptx_unary("rsqrt.approx.ftz.f32", value)


def _cvt_to_f32(bits, dtype: str):
    if dtype == "float16":
        return _ptx_unary("cvt.f32.f16", T.cast(bits, "uint16"))
    return _ptx_unary("cvt.f32.bf16", T.cast(bits, "uint16"))


def _cvt_from_f32(value, dtype: str):
    if dtype == "float16":
        return _ptx_unary("cvt.rn.f16.f32", value, dtype="uint16")
    return _ptx_unary("cvt.rn.bf16.f32", value, dtype="uint16")


def _cvt_pair_from_f32(high, low, dtype: str):
    if dtype == "float16":
        return _ptx_binary("cvt.rn.f16x2.f32", high, low, dtype="uint32")
    return _ptx_binary("cvt.rn.bf16x2.f32", high, low, dtype="uint32")


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


@T.inline
def _load_global_element(buffer, index, values, value_offset, VEC: T.constexpr):
    if VEC == 1:
        T.ptx.ld.global_.b16(values[value_offset], buffer.ptr_to([index]))
    elif VEC == 2:
        T.ptx.ld.global_.v2.b16(
            values[value_offset], values[value_offset + 1], buffer.ptr_to([index])
        )
    else:
        T.ptx.ld.global_.v4.b16(
            values[value_offset],
            values[value_offset + 1],
            values[value_offset + 2],
            values[value_offset + 3],
            buffer.ptr_to([index]),
        )


@T.inline
def _load_global_packed(buffer, index, values, value_offset, VEC: T.constexpr):
    words = T.alloc_local((VEC // 2,), "uint32")
    if VEC == 2:
        T.ptx.ld.global_.b32(words[0], buffer.ptr_to([index]))
    elif VEC == 4:
        T.ptx.ld.global_.v2.b32(words[0], words[1], buffer.ptr_to([index]))
    else:
        T.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
    for pair in T.unroll(VEC // 2):
        T.ptx.mov.b32(
            values[value_offset + pair * 2], values[value_offset + pair * 2 + 1], words[pair]
        )


@T.inline
def _load_shared_bits(shared_raw, byte_offset, values, value_offset, VEC: T.constexpr):
    if VEC == 2:
        T.ptx.ld.shared.v2.b16(
            values[value_offset], values[value_offset + 1], shared_raw.ptr_to([byte_offset])
        )
    elif VEC == 4:
        T.ptx.ld.shared.v4.b16(
            values[value_offset],
            values[value_offset + 1],
            values[value_offset + 2],
            values[value_offset + 3],
            shared_raw.ptr_to([byte_offset]),
        )
    else:
        words = T.alloc_local((4,), "uint32")
        T.ptx.ld.shared.v4.b32(
            words[0], words[1], words[2], words[3], shared_raw.ptr_to([byte_offset])
        )
        for pair in T.unroll(4):
            T.ptx.mov.b32(
                values[value_offset + pair * 2], values[value_offset + pair * 2 + 1], words[pair]
            )


@T.inline
def _store_global_element(buffer, index, values, value_offset, predicate, VEC: T.constexpr):
    if VEC == 1:
        T.ptx.st.global_.b16(buffer.ptr_to([index]), values[value_offset], pred=predicate)
    elif VEC == 2:
        T.ptx.st.global_.v2.b16(
            buffer.ptr_to([index]), values[value_offset], values[value_offset + 1], pred=predicate
        )
    else:
        T.ptx.st.global_.v4.b16(
            buffer.ptr_to([index]),
            values[value_offset],
            values[value_offset + 1],
            values[value_offset + 2],
            values[value_offset + 3],
            pred=predicate,
        )


@T.inline
def _store_global_packed(buffer, index, words, word_offset, predicate, VEC: T.constexpr):
    if VEC == 2:
        T.ptx.st.global_.b32(buffer.ptr_to([index]), words[word_offset], pred=predicate)
    elif VEC == 4:
        T.ptx.st.global_.v2.b32(
            buffer.ptr_to([index]), words[word_offset], words[word_offset + 1], pred=predicate
        )
    else:
        T.ptx.st.global_.v4.b32(
            buffer.ptr_to([index]),
            words[word_offset],
            words[word_offset + 1],
            words[word_offset + 2],
            words[word_offset + 3],
            pred=predicate,
        )


def _label(
    variant: str,
    dtype: str,
    B: int,
    N: int,
    H: int,
    input_layout: str,
    output_layout: str,
    enable_pdl: bool,
    suffix: str | None = None,
) -> str:
    label = (
        f"{_VARIANT_CODE[variant]}_{_DTYPE_CODE[dtype]}_b{B}_n{N}_h{H}_"
        f"x{_LAYOUT_CODE[input_layout]}_y{_LAYOUT_CODE[output_layout]}_"
        f"pdl{int(enable_pdl)}"
    )
    return f"{label}_{suffix}" if suffix else label


def _config(
    variant: str,
    dtype: str,
    B: int,
    N: int,
    H: int,
    input_layout: str,
    output_layout: str,
    enable_pdl: bool,
    *,
    suffix: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "label": _label(variant, dtype, B, N, H, input_layout, output_layout, enable_pdl, suffix),
        "variant": variant,
        "dtype": dtype,
        "B": B,
        "N": N,
        "H": H,
        "input_layout": input_layout,
        "output_layout": output_layout,
        "enable_pdl": enable_pdl,
        **extra,
    }


def _main_configs() -> list[dict[str, Any]]:
    return [
        _config(variant, "float16", B, N, H, input_layout, "compact", enable_pdl)
        for variant in ("rmsnorm", "gemma_rmsnorm")
        for B in (1, 19, 99, 989)
        for N in (4, 7, 16)
        for H in (64, 128, 256, 512)
        for input_layout in ("compact", "strided")
        for enable_pdl in (False, True)
    ]


BENCH_CONFIGS = [
    _config("rmsnorm", "bfloat16", 32, 32, 128, "compact", "compact", False),
    _config("rmsnorm", "float16", 16, 64, 128, "compact", "compact", False),
    _config("gemma_rmsnorm", "bfloat16", 32, 32, 128, "compact", "compact", False),
]


_REGRESSION_CONFIGS = [
    _config(
        "rmsnorm",
        "bfloat16",
        1,
        4,
        128,
        "strided",
        "compact",
        True,
        suffix="xbs2p31",
        x_batch_stride=_INT64_STRIDE,
        x_head_stride=128,
    ),
    _config(
        "rmsnorm",
        "float16",
        16800,
        1024,
        128,
        "compact",
        "compact",
        True,
        suffix="i32offset_overflow",
    ),
    _config(
        "gemma_rmsnorm",
        "float16",
        16800,
        1024,
        128,
        "compact",
        "compact",
        True,
        suffix="i32offset_overflow",
    ),
]


_STRUCTURAL_CONFIGS = [
    _config("rmsnorm", "bfloat16", 2, 5, 111, "compact", "compact", False),
    _config("gemma_rmsnorm", "float16", 3, 7, 66, "strided", "compact", True),
    _config("rmsnorm", "bfloat16", 2, 3, 68, "compact", "compact", False),
    _config("gemma_rmsnorm", "bfloat16", 2, 3, 4096, "strided", "compact", True),
    _config("rmsnorm", "float16", 2, 5, 8192, "compact", "compact", False),
    _config("gemma_rmsnorm", "bfloat16", 1, 3, 16385, "compact", "compact", True),
    _config("rmsnorm", "bfloat16", 1, 2, 57344, "compact", "compact", False),
    _config("gemma_rmsnorm", "float16", 1, 2, 57352, "strided", "compact", True),
]


_ABI_CONFIGS = [
    _config(
        "rmsnorm",
        "float16",
        19,
        7,
        128,
        "compact",
        "strided",
        True,
        suffix="eps1e4",
        eps=1e-4,
        y_head_stride=256,
        y_batch_stride=1920,
    ),
    _config(
        "gemma_rmsnorm",
        "bfloat16",
        3,
        5,
        66,
        "strided",
        "strided",
        False,
        suffix="independent_strides",
        x_head_stride=70,
        x_batch_stride=364,
        y_head_stride=76,
        y_batch_stride=398,
    ),
]


CONFIGS = _main_configs() + BENCH_CONFIGS + _REGRESSION_CONFIGS + _STRUCTURAL_CONFIGS + _ABI_CONFIGS

if len(CONFIGS) != 400 or len({config["label"] for config in CONFIGS}) != 400:
    raise AssertionError("QK RMSNorm configuration inventory must contain 400 unique labels")


def _validate(variant: str, dtype: str, B: int, N: int, H: int, eps: float) -> None:
    if variant not in _VARIANT_CODE:
        raise ValueError(f"unsupported variant: {variant}")
    if dtype not in _DTYPE_CODE:
        raise ValueError(f"unsupported dtype: {dtype}")
    if B <= 0 or N <= 0 or H <= 0:
        raise ValueError(f"B, N, and H must be positive, got B={B}, N={N}, H={H}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")


def get_kernel(
    *,
    variant: str,
    dtype: str,
    H: int,
    enable_pdl: bool,
    B: int = 1,
    N: int = 1,
    eps: float = _DEFAULT_EPS,
    **kwargs: Any,
):
    """Return the source-shaped dynamic-stride QK RMSNorm specialization."""
    _validate(variant, dtype, B, N, H, eps)
    source = _source_config(H)
    tpr = int(source["tpr"])
    threads = int(source["threads"])
    rows = int(source["rows"])
    warps_per_row = int(source["warps_per_row"])
    vec = int(source["vec"])
    copy_bits = int(source["copy_bits"])
    vec_blocks = int(source["vec_blocks"])
    cols = int(source["cols"])
    tile_bytes = int(source["tile_bytes"])
    use_async = bool(source["use_async"])
    smem_bytes = int(source["smem_bytes"])
    total_values = vec * vec_blocks
    packed_pairs = _ceil_div(total_values, 2)
    pair_values = packed_pairs * 2
    full_tile = H == cols
    vb_pow2 = vec_blocks & (vec_blocks - 1) == 0
    packed_narrow = vec > 1 and (vb_pow2 or full_tile)
    sync_x_packed = vec == 8 or (vec in (2, 4) and vb_pow2)
    output_packed = vec == 8 or (vec in (2, 4) and vb_pow2)
    weight_bias = 0.0 if variant == "rmsnorm" else 1.0
    copy_bytes = copy_bits // 8
    reduce_base = tile_bytes if use_async else 0
    row_lane_xors = tuple(lane_xor for lane_xor in (1, 2, 4, 8, 16) if lane_xor < min(tpr, 32))
    full_lane_xors = (1, 2, 4, 8, 16)

    for name in ("x_batch_stride", "x_head_stride", "y_batch_stride", "y_head_stride"):
        stride = kwargs.get(name)
        if stride is not None and int(stride) % vec != 0:
            raise ValueError(f"{name}={stride} must be divisible by vec={vec}")

    def weight_uses_packed(vb: int) -> bool:
        if vec == 1:
            return False
        if vec == 8:
            return True
        if vec_blocks == 1:
            return False
        if vb_pow2:
            return True
        if total_values < 28:
            return False
        return vb != vec_blocks - 1

    @T.inline
    def kernel_body(
        x,
        weight,
        y,
        runtime_B,
        runtime_N,
        runtime_eps,
        x_batch_stride,
        x_head_stride,
        y_batch_stride,
        y_head_stride,
    ):
        # QK_RMSNORM_KERNEL_START
        block_raw = T.cta_id([T.cast(T.ceildiv(runtime_B * runtime_N, T.int64(rows)), "int32")])
        tid = T.thread_id([threads])

        if enable_pdl:
            T.ptx.griddepcontrol.wait()

        block: T.int32 = T.cast(block_raw, "int32")
        row_in_cta: T.int32 = tid // tpr
        thread_in_row: T.int32 = tid % tpr
        row_i32: T.int32 = block * rows + row_in_cta
        row_i64: T.int64 = T.cast(row_i32, "int64")
        row_valid: T.bool = row_i64 < runtime_B * runtime_N
        batch_idx: T.int64 = row_i64 // runtime_N
        head_idx: T.int64 = row_i64 % runtime_N
        warp: T.int32 = tid // 32
        lane: T.int32 = tid % 32
        row_warp: T.int32 = warp // warps_per_row
        warp_in_row: T.int32 = warp % warps_per_row

        shared_raw = T.alloc_buffer((smem_bytes,), "uint8", scope="shared.dyn")
        T.attr({"tirx.dyn_smem_bytes": smem_bytes})

        x_bits = T.alloc_local((pair_values,), "uint16")
        w_bits = T.alloc_local((pair_values,), "uint16")
        x_f32 = T.alloc_local((pair_values,), "float32")
        x_f32_pairs = T.alloc_local((packed_pairs,), "uint64")
        w_f32 = T.alloc_local((pair_values,), "float32")
        undefined_f32 = T.alloc_local((1,), "float32")

        if not use_async:
            if full_tile and vb_pow2:
                _zero_float_pair_values(x_f32_pairs, packed_pairs)
            elif full_tile:
                _zero_float_values(x_f32, total_values)
            elif vb_pow2 and total_values > 1:
                _zero_packed_half_values(x_bits, total_values // 2)
            elif total_values == 1:
                T.ptx.mov.b16(x_bits[0], T.uint16(0))
            else:
                _zero_half_values(x_bits, total_values)

        for vb in T.unroll(vec_blocks):
            local_col: T.int32 = (thread_in_row + vb * tpr) * vec
            col_valid: T.bool = local_col < H
            x_offset: T.int64 = (
                batch_idx * x_batch_stride + head_idx * x_head_stride + T.cast(local_col, "int64")
            )
            if use_async:
                if row_valid:
                    source_bytes: T.uint32 = T.cast(
                        T.if_then_else(col_valid, copy_bytes, 0), "uint32"
                    )
                    T.ptx.cp.async_.ca.shared.global_(
                        shared_raw.ptr_to([(row_in_cta * cols + local_col) * _ELEM_BYTES]),
                        x.ptr_to([x_offset]),
                        copy_bytes,
                        source_bytes,
                    )
            else:
                if row_valid and col_valid:
                    if sync_x_packed:
                        _load_global_packed(x, x_offset, x_bits, vb * vec, VEC=vec)
                    else:
                        if total_values == 1:
                            T.ptx.mov.b16(x_bits[0], T.uint16(0))
                        _load_global_element(x, x_offset, x_bits, vb * vec, VEC=vec)

        if use_async:
            T.ptx.cp.async_.commit_group()

        for vb in T.unroll(vec_blocks):
            local_col: T.int32 = (thread_in_row + vb * tpr) * vec
            if local_col < H:
                if weight_uses_packed(vb):
                    _load_global_packed(weight, local_col, w_bits, vb * vec, VEC=vec)
                else:
                    _load_global_element(weight, local_col, w_bits, vb * vec, VEC=vec)

        if use_async:
            T.ptx.cp.async_.wait_group(0)
            for vb in T.unroll(vec_blocks):
                local_col: T.int32 = (thread_in_row + vb * tpr) * vec
                _load_shared_bits(
                    shared_raw,
                    (row_in_cta * cols + local_col) * _ELEM_BYTES,
                    x_bits,
                    vb * vec,
                    VEC=vec,
                )
        if not use_async and full_tile and vb_pow2:
            if row_valid:
                for pair in T.unroll(packed_pairs):
                    x_low: T.float32 = _cvt_to_f32(x_bits[pair * 2], dtype)
                    x_high: T.float32 = _cvt_to_f32(x_bits[pair * 2 + 1], dtype)
                    T.ptx.mov.b64(x_f32_pairs[pair], x_low, x_high)
        elif not use_async and full_tile:
            if row_valid:
                for value in T.unroll(total_values):
                    x_f32[value] = _cvt_to_f32(x_bits[value], dtype)
        else:
            for value in T.unroll(total_values):
                x_f32[value] = _cvt_to_f32(x_bits[value], dtype)

        if total_values == 1:
            local_sum: T.float32 = _fma_half_to_f32(x_bits[0], x_bits[0], T.float32(0.0), dtype)
        else:
            x_sq = T.alloc_local((pair_values,), "float32")
            for pair in T.unroll(packed_pairs):
                square = T.alloc_local((1,), "uint64")
                x_pair: T.uint64 = T.cuda.make_float2(x_f32[pair * 2], x_f32[pair * 2 + 1])
                if not use_async and full_tile and vb_pow2:
                    x_pair = x_f32_pairs[pair]
                T.ptx.mul.f32x2(square[0], x_pair, x_pair)
                T.ptx.mov.b64(x_sq[pair * 2], x_sq[pair * 2 + 1], square[0])
            local_sum = T.float32(0.0)
            for value in T.unroll(total_values):
                local_sum = _add_f32(local_sum, x_sq[value])

        local_sum = _butterfly_sum_f32(local_sum, row_lane_xors)
        warp_sum: T.float32 = local_sum

        if warps_per_row > 1:
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
        else:
            sum_sq: T.float32 = warp_sum

        if H == 1:
            shifted: T.float32 = _add_f32(sum_sq, runtime_eps)
        elif H & (H - 1) == 0:
            shifted: T.float32 = _fma_rn_f32(sum_sq, T.float32(1.0 / H), runtime_eps)
        else:
            mean_sq: T.float32 = _div_rn_f32(sum_sq, T.float32(H))
            shifted: T.float32 = _add_f32(mean_sq, runtime_eps)
        rstd: T.float32 = _rsqrt_approx_ftz(shifted)

        T.ptx.bar.sync(T.uint32(0))

        if use_async:
            for vb in T.unroll(vec_blocks):
                local_col: T.int32 = (thread_in_row + vb * tpr) * vec
                _load_shared_bits(
                    shared_raw,
                    (row_in_cta * cols + local_col) * _ELEM_BYTES,
                    x_bits,
                    vb * vec,
                    VEC=vec,
                )
            for value in T.unroll(total_values):
                x_f32[value] = _cvt_to_f32(x_bits[value], dtype)

        for value in T.unroll(total_values):
            w_f32[value] = _cvt_to_f32(w_bits[value], dtype)

        if total_values == 1:
            x_f32[0] = _ptx_binary("mul.f32", x_f32[0], rstd)
            w_f32[0] = _add_f32(w_f32[0], T.float32(weight_bias))
            x_f32[0] = _ptx_binary("mul.f32", x_f32[0], w_f32[0])
        else:
            for pair in T.unroll(packed_pairs):
                high_scale: T.float32 = rstd
                if pair * 2 + 1 >= total_values:
                    high_scale = undefined_f32[0]
                product = T.alloc_local((1,), "uint64")
                x_pair: T.uint64 = T.cuda.make_float2(x_f32[pair * 2], x_f32[pair * 2 + 1])
                if not use_async and full_tile and vb_pow2:
                    x_pair = x_f32_pairs[pair]
                T.ptx.mul.f32x2(product[0], x_pair, T.cuda.make_float2(rstd, high_scale))
                T.ptx.mov.b64(x_f32[pair * 2], x_f32[pair * 2 + 1], product[0])

            for pair in T.unroll(packed_pairs):
                high_bias: T.float32 = T.float32(weight_bias)
                if pair * 2 + 1 >= total_values:
                    high_bias = undefined_f32[0]
                biased = T.alloc_local((1,), "uint64")
                T.ptx.add.f32x2(
                    biased[0],
                    T.cuda.make_float2(w_f32[pair * 2], w_f32[pair * 2 + 1]),
                    T.cuda.make_float2(T.float32(weight_bias), high_bias),
                )
                T.ptx.mov.b64(w_f32[pair * 2], w_f32[pair * 2 + 1], biased[0])

            for pair in T.unroll(packed_pairs):
                product = T.alloc_local((1,), "uint64")
                T.ptx.mul.f32x2(
                    product[0],
                    T.cuda.make_float2(x_f32[pair * 2], x_f32[pair * 2 + 1]),
                    T.cuda.make_float2(w_f32[pair * 2], w_f32[pair * 2 + 1]),
                )
                T.ptx.mov.b64(x_f32[pair * 2], x_f32[pair * 2 + 1], product[0])

        y_bits = T.alloc_local((pair_values,), "uint16")
        y_words = T.alloc_local((packed_pairs,), "uint32")
        if packed_narrow:
            for pair in T.unroll(packed_pairs):
                y_words[pair] = _cvt_pair_from_f32(x_f32[pair * 2 + 1], x_f32[pair * 2], dtype)
        else:
            for value in T.unroll(total_values):
                y_bits[value] = _cvt_from_f32(x_f32[value], dtype)
            if output_packed:
                for pair in T.unroll(total_values // 2):
                    T.ptx.mov.b32(y_words[pair], y_bits[pair * 2], y_bits[pair * 2 + 1])

        for vb in T.unroll(vec_blocks):
            local_col: T.int32 = (thread_in_row + vb * tpr) * vec
            col_valid: T.bool = local_col < H
            y_offset: T.int64 = (
                batch_idx * y_batch_stride + head_idx * y_head_stride + T.cast(local_col, "int64")
            )
            predicate: T.bool = row_valid and col_valid
            if output_packed:
                _store_global_packed(y, y_offset, y_words, vb * vec // 2, predicate, VEC=vec)
            else:
                _store_global_element(y, y_offset, y_bits, vb * vec, predicate, VEC=vec)

        if enable_pdl:
            T.ptx.griddepcontrol.launch_dependents()

    @T.prim_func
    def flashinfer_qk_rmsnorm(
        x_ptr: T.handle,
        weight_ptr: T.handle,
        y_ptr: T.handle,
        runtime_B: T.int64,
        runtime_N: T.int64,
        runtime_eps: T.float32,
        x_batch_stride: T.int64,
        x_head_stride: T.int64,
        y_batch_stride: T.int64,
        y_head_stride: T.int64,
    ):
        T.func_attr({"tir.is_entry_func": True})
        x = T.match_buffer(
            x_ptr,
            shape=(
                (runtime_B - T.int64(1)) * x_batch_stride
                + (runtime_N - T.int64(1)) * x_head_stride
                + T.int64(H),
            ),
            dtype=dtype,
            scope="global",
        )
        weight = T.match_buffer(weight_ptr, shape=(H,), dtype=dtype, scope="global")
        y = T.match_buffer(
            y_ptr,
            shape=(
                (runtime_B - T.int64(1)) * y_batch_stride
                + (runtime_N - T.int64(1)) * y_head_stride
                + T.int64(H),
            ),
            dtype=dtype,
            scope="global",
        )
        T.device_entry()
        kernel_body(
            x,
            weight,
            y,
            runtime_B,
            runtime_N,
            runtime_eps,
            x_batch_stride,
            x_head_stride,
            y_batch_stride,
            y_head_stride,
        )

    launch_params = ["blockIdx.x", "threadIdx.x"]
    if enable_pdl:
        launch_params.append("tirx.use_programtic_dependent_launch")
    launch_params.append("tirx.use_dyn_shared_memory")
    return flashinfer_qk_rmsnorm.with_attr("tirx.kernel_launch_params", launch_params)


def _torch_dtype(dtype: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def _tensor_strides(config: dict[str, Any]) -> tuple[int, int, int, int]:
    N = int(config["N"])
    H = int(config["H"])
    x_head = int(config.get("x_head_stride", H if config["input_layout"] == "compact" else 2 * H))
    y_head = int(config.get("y_head_stride", H if config["output_layout"] == "compact" else 2 * H))
    x_batch = int(config.get("x_batch_stride", N * x_head))
    y_batch = int(config.get("y_batch_stride", N * y_head))
    return x_batch, x_head, y_batch, y_head


def _storage_size(B: int, N: int, H: int, batch_stride: int, head_stride: int) -> int:
    return (B - 1) * batch_stride + (N - 1) * head_stride + H


def _validate_strides(
    B: int, N: int, H: int, vec: int, x_batch: int, x_head: int, y_batch: int, y_head: int
) -> None:
    del B
    row_span_x = (N - 1) * x_head + H
    row_span_y = (N - 1) * y_head + H
    if x_head < H or y_head < H or x_batch < row_span_x or y_batch < row_span_y:
        raise ValueError(
            "batch/head strides must describe non-overlapping rows: "
            f"x=({x_batch},{x_head}), y=({y_batch},{y_head}), N={N}, H={H}"
        )
    for name, stride in (
        ("x_batch_stride", x_batch),
        ("x_head_stride", x_head),
        ("y_batch_stride", y_batch),
        ("y_head_stride", y_head),
    ):
        if stride % vec != 0:
            raise ValueError(f"{name}={stride} must be divisible by vec={vec}")


def _prepare_tensors(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    B = int(config["B"])
    N = int(config["N"])
    H = int(config["H"])
    dtype = str(config["dtype"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    _validate(str(config["variant"]), dtype, B, N, H, eps)
    x_batch, x_head, y_batch, y_head = _tensor_strides(config)
    vec = int(_source_config(H)["vec"])
    _validate_strides(B, N, H, vec, x_batch, x_head, y_batch, y_head)

    torch_dtype = _torch_dtype(dtype)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(42)
    x_size = _storage_size(B, N, H, x_batch, x_head)
    x_backing = torch.full(
        (x_size + _GUARD_ELEMENTS,), _GUARD_VALUE, dtype=torch_dtype, device="cuda"
    )
    x_arg = x_backing[:x_size]
    x = x_arg.as_strided((B, N, H), (x_batch, x_head, 1))
    if (dtype == "bfloat16" and H >= 4096) or H >= 57344:
        columns = torch.arange(H, device="cuda")
        magnitude = torch.where(columns % 257 < 128, 0.5, 1.0)
        pattern = torch.where(columns % 2 == 0, magnitude, -magnitude).to(torch_dtype)
        x.copy_(pattern.expand(B, N, H))
        x[:, 1::2].neg_()
    else:
        x.normal_(generator=generator)

    weight_backing = torch.empty(H + _GUARD_ELEMENTS, dtype=torch_dtype, device="cuda")
    weight = weight_backing[:H]
    weight.normal_(generator=generator)
    weight_backing[H:].fill_(_GUARD_VALUE)
    return {
        "x": x,
        "x_arg": x_arg,
        "x_backing": x_backing,
        "x_size": x_size,
        "weight": weight,
        "weight_backing": weight_backing,
        "x_batch_stride": x_batch,
        "x_head_stride": x_head,
        "y_batch_stride": y_batch,
        "y_head_stride": y_head,
    }


def prepare_data(**config: Any):
    """Create deterministic QK RMSNorm inputs."""
    data = _prepare_tensors(dict(config))
    return data["x"], data["weight"]


def _prepare_output(
    B: int,
    N: int,
    H: int,
    batch_stride: int,
    head_stride: int,
    dtype: str,
    *,
    initialize_padding: bool,
) -> dict[str, Any]:
    import torch

    size = _storage_size(B, N, H, batch_stride, head_stride)
    backing = torch.empty(size + _GUARD_ELEMENTS, dtype=_torch_dtype(dtype), device="cuda")
    if initialize_padding:
        backing.fill_(_GUARD_VALUE)
    else:
        backing[size:].fill_(_GUARD_VALUE)
    arg = backing[:size]
    view = arg.as_strided((B, N, H), (batch_stride, head_stride, 1))
    return {"view": view, "arg": arg, "backing": backing, "size": size}


def _assert_all_guard(values, *, name: str) -> None:
    import torch

    if values.numel() == 0:
        return
    if not torch.equal(values, torch.full_like(values, _GUARD_VALUE)):
        raise AssertionError(f"{name} padding or guard was modified")


def _assert_storage_padding(
    backing, size: int, B: int, N: int, H: int, batch_stride: int, head_stride: int, *, name: str
) -> None:
    _assert_all_guard(backing[size:], name=f"{name} terminal")
    head_gap = head_stride - H
    if head_gap > 0 and N > 1:
        padding = backing.as_strided(
            (B, N - 1, head_gap), (batch_stride, head_stride, 1), storage_offset=H
        )
        _assert_all_guard(padding, name=f"{name} head")
    batch_span = (N - 1) * head_stride + H
    batch_gap = batch_stride - batch_span
    if batch_gap > 0 and B > 1:
        padding = backing.as_strided(
            (B - 1, batch_gap), (batch_stride, 1), storage_offset=batch_span
        )
        _assert_all_guard(padding, name=f"{name} batch")


def _overflow_rows(B: int, N: int, H: int) -> list[int] | None:
    M = B * N
    if M * H <= 2**31 - 1:
        return None
    boundary = _ceil_div(2**31, H)
    return sorted(
        {row for row in (0, 1, boundary - 1, boundary, boundary + 1, M - 1) if 0 <= row < M}
    )


def _checked_rows(tensor, rows: list[int] | None, N: int):
    if rows is None:
        return tensor
    import torch

    row_index = torch.tensor(rows, device=tensor.device, dtype=torch.int64)
    return tensor[row_index // N, row_index % N]


def _math_oracle(x, weight, variant: str, eps: float, rows: list[int] | None, N: int):
    x_checked = _checked_rows(x, rows, N).float()
    variance = x_checked.square().mean(dim=-1, keepdim=True)
    normalized = x_checked * variance.add(eps).rsqrt()
    bias = 0.0 if variant == "rmsnorm" else 1.0
    return (normalized * (weight.float() + bias)).to(dtype=x.dtype)


def _assert_close(actual, expected, *, name: str) -> None:
    import torch

    torch.testing.assert_close(
        actual, expected, rtol=1e-3, atol=1e-3, msg=lambda message: f"{name}: {message}"
    )


def _flashinfer_api(variant: str, device):
    import flashinfer
    import flashinfer.norm as flashinfer_norm

    if flashinfer_norm._use_cuda_norm(device):
        raise AssertionError("FlashInfer QK RMSNorm oracle dispatched to legacy CUDA")
    return getattr(flashinfer, variant), flashinfer_norm


def _launch_tirx(executable, data, output, config: dict[str, Any]) -> None:
    executable(
        data["x_arg"],
        data["weight"],
        output["arg"],
        int(config["B"]),
        int(config["N"]),
        float(config.get("eps", _DEFAULT_EPS)),
        data["x_batch_stride"],
        data["x_head_stride"],
        data["y_batch_stride"],
        data["y_head_stride"],
    )


def _input_snapshot(data, B: int, N: int, H: int) -> dict[str, Any]:
    rows = _overflow_rows(B, N, H)
    if rows is None:
        values = data["x"].clone()
    else:
        values = _checked_rows(data["x"], rows, N)[:, [0, H // 2, H - 1]].clone()
    return {"rows": rows, "values": values, "weight": data["weight"].clone()}


def _assert_inputs_unchanged(data, snapshot, B: int, N: int, H: int) -> None:
    rows = snapshot["rows"]
    if rows is None:
        observed = data["x"]
    else:
        observed = _checked_rows(data["x"], rows, N)[:, [0, H // 2, H - 1]]
    if not observed.equal(snapshot["values"]):
        raise AssertionError("input tensor was modified")
    if not data["weight"].equal(snapshot["weight"]):
        raise AssertionError("weight tensor was modified")
    _assert_storage_padding(
        data["x_backing"],
        data["x_size"],
        B,
        N,
        H,
        data["x_batch_stride"],
        data["x_head_stride"],
        name="input",
    )
    _assert_all_guard(data["weight_backing"][H:], name="weight")


def run_test(**config: Any) -> None:
    """Compile, launch, and validate one QK RMSNorm config."""
    import torch

    from tirx_kernels.runner import compile_kernel

    config = dict(config)
    B = int(config["B"])
    N = int(config["N"])
    H = int(config["H"])
    dtype = str(config["dtype"])
    variant = str(config["variant"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    enable_pdl = bool(config["enable_pdl"])
    data = _prepare_tensors(config)
    snapshot = _input_snapshot(data, B, N, H)
    output = _prepare_output(
        B, N, H, data["y_batch_stride"], data["y_head_stride"], dtype, initialize_padding=True
    )
    reference_out = _prepare_output(
        B, N, H, data["y_batch_stride"], data["y_head_stride"], dtype, initialize_padding=True
    )

    executable = compile_kernel(get_kernel(**config))
    _launch_tirx(executable, data, output, config)

    api, flashinfer_norm = _flashinfer_api(variant, data["x"].device)
    original_cute = flashinfer_norm.qk_rmsnorm_cute
    cute_calls = 0

    def tracked_cute(*args, **kwargs):
        nonlocal cute_calls
        cute_calls += 1
        return original_cute(*args, **kwargs)

    flashinfer_norm.qk_rmsnorm_cute = tracked_cute
    try:
        reference_implicit = api(data["x"], data["weight"], eps, enable_pdl=enable_pdl)
        returned = api(
            data["x"], data["weight"], eps, out=reference_out["view"], enable_pdl=enable_pdl
        )
    finally:
        flashinfer_norm.qk_rmsnorm_cute = original_cute
    if cute_calls != 2:
        raise AssertionError(f"expected two CuTe-DSL oracle dispatches, observed {cute_calls}")
    if returned is not reference_out["view"]:
        raise AssertionError("FlashInfer caller-provided out did not preserve object identity")

    torch.cuda.synchronize()
    rows = _overflow_rows(B, N, H)
    actual_checked = _checked_rows(output["view"], rows, N)
    implicit_checked = _checked_rows(reference_implicit, rows, N)
    explicit_checked = _checked_rows(reference_out["view"], rows, N)
    oracle = _math_oracle(data["x"], data["weight"], variant, eps, rows, N)
    if not torch.isfinite(actual_checked).all():
        raise AssertionError("TIRx output contains non-finite values")
    _assert_close(actual_checked, implicit_checked, name="FlashInfer implicit output")
    _assert_close(actual_checked, explicit_checked, name="FlashInfer explicit output")
    _assert_close(actual_checked, oracle, name="FP32 math oracle")
    _assert_inputs_unchanged(data, snapshot, B, N, H)
    _assert_storage_padding(
        output["backing"],
        output["size"],
        B,
        N,
        H,
        data["y_batch_stride"],
        data["y_head_stride"],
        name="TIRx output",
    )
    _assert_storage_padding(
        reference_out["backing"],
        reference_out["size"],
        B,
        N,
        H,
        data["y_batch_stride"],
        data["y_head_stride"],
        name="FlashInfer output",
    )


def prepare_bench(**config: Any):
    """Compile the selected specialization before GPU assignment."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(config), "executable": compile_kernel(get_kernel(**config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs: Any
):
    """Construct and validate both single-launch timed closures."""
    import torch

    config = dict(prepared["config"])
    config.update(kwargs)
    B = int(config["B"])
    N = int(config["N"])
    H = int(config["H"])
    dtype = str(config["dtype"])
    variant = str(config["variant"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    enable_pdl = bool(config["enable_pdl"])
    data = _prepare_tensors(config)
    tirx_output = _prepare_output(
        B, N, H, data["y_batch_stride"], data["y_head_stride"], dtype, initialize_padding=False
    )
    flashinfer_output = _prepare_output(
        B, N, H, data["y_batch_stride"], data["y_head_stride"], dtype, initialize_padding=False
    )
    executable = prepared["executable"]

    def tirx_launch():
        _launch_tirx(executable, data, tirx_output, config)

    tirx_launch()
    torch.cuda.synchronize()

    def build_flashinfer_reference():
        api, _ = _flashinfer_api(variant, data["x"].device)

        def flashinfer_launch():
            return api(
                data["x"], data["weight"], eps, out=flashinfer_output["view"], enable_pdl=enable_pdl
            )

        returned = flashinfer_launch()
        if returned is not flashinfer_output["view"]:
            raise AssertionError("FlashInfer benchmark out did not preserve object identity")
        torch.cuda.synchronize()
        _assert_close(tirx_output["view"], flashinfer_output["view"], name="benchmark precheck")
        return flashinfer_launch

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
    """Benchmark QK RMSNorm against FlashInfer CuTe-DSL."""
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
