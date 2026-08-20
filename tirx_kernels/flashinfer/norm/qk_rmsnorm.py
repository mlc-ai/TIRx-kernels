# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400e330fb2debe0bf8730d9424a1d37927f), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer CuTe-DSL 3-D QK RMSNorm and Gemma RMSNorm port.

The source implementation is ``QKRMSNormKernel`` and
``qk_rmsnorm_cute`` in ``flashinfer/norm/kernels/rmsnorm.py``.  The
public dispatch is in ``flashinfer/norm/__init__.py``.
"""

from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.runner import bench

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

# The half-precision instruction each dtype spells, selected once per
# specialization and applied at the call sites below.
_CVT_TO_F32 = {"float16": "cvt.f32.f16", "bfloat16": "cvt.f32.bf16"}
_CVT_FROM_F32 = {"float16": "cvt.rn.f16.f32", "bfloat16": "cvt.rn.bf16.f32"}
_CVT_PAIR_FROM_F32 = {"float16": "cvt.rn.f16x2.f32", "bfloat16": "cvt.rn.bf16x2.f32"}
_FMA_HALF_TO_F32 = {"float16": "fma.rn.f32.f16", "bfloat16": "fma.rn.f32.bf16"}
_CP_ASYNC = "cp.async.ca.shared.global"


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


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


def _butterfly_sum_f32(acc, lane_xors: tuple[int, ...]) -> None:
    """``acc`` <- the sum of ``acc`` over the butterfly partners in *lane_xors*.

    One ``shfl.sync.bfly.b32`` followed by one ``add.f32`` per xor distance,
    each round reading the value the previous round produced.  Every lane of
    the warp reaches every round, so this must be called at a convergent point.
    """
    peer = K.local_scalar(K.u32)
    for lane_xor in lane_xors:
        K.ptx.shfl_sync.bfly.b32(
            peer,
            K.reinterpret("uint32", acc),
            K.uint32(lane_xor),
            K.uint32(31),
            K.uint32(0xFFFFFFFF),
        )
        K.ptx.add.f32(acc, acc, K.reinterpret("float32", peer))


def _load_global_element(values, value_offset: int, buffer, index, vec: int) -> None:
    """``values[value_offset : +vec]`` <- *vec* halves at ``buffer[index]``, one b16 load."""
    if vec == 1:
        K.ptx.ld.global_.b16(values[value_offset], buffer.ptr_to([index]))
    elif vec == 2:
        K.ptx.ld.global_.v2.b16(
            values[value_offset], values[value_offset + 1], buffer.ptr_to([index])
        )
    else:
        K.ptx.ld.global_.v4.b16(
            values[value_offset],
            values[value_offset + 1],
            values[value_offset + 2],
            values[value_offset + 3],
            buffer.ptr_to([index]),
        )


def _load_global_packed(values, value_offset: int, buffer, index, vec: int) -> None:
    """``values[value_offset : +vec]`` <- *vec* halves at ``buffer[index]``, via b32 words.

    One ``ld.global`` of packed words followed by one ``mov.b32`` per word to
    split it into its two half lanes.
    """
    words = K.alloc_local([vec // 2], K.u32)
    if vec == 2:
        K.ptx.ld.global_.b32(words[0], buffer.ptr_to([index]))
    elif vec == 4:
        K.ptx.ld.global_.v2.b32(words[0], words[1], buffer.ptr_to([index]))
    else:
        K.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
    for pair in range(vec // 2):
        K.ptx.mov.b32(
            values[value_offset + pair * 2], values[value_offset + pair * 2 + 1], words[pair]
        )


def _load_shared_bits(values, value_offset: int, shared_raw, byte_offset, vec: int) -> None:
    """``values[value_offset : +vec]`` <- *vec* halves staged at ``shared_raw + byte_offset``.

    The two- and four-half shapes read the halves directly; the eight-half
    shape reads four words and splits them with one ``mov.b32`` apiece.
    """
    if vec == 2:
        K.ptx.ld.shared.v2.b16(
            values[value_offset], values[value_offset + 1], shared_raw.ptr_to([byte_offset])
        )
    elif vec == 4:
        K.ptx.ld.shared.v4.b16(
            values[value_offset],
            values[value_offset + 1],
            values[value_offset + 2],
            values[value_offset + 3],
            shared_raw.ptr_to([byte_offset]),
        )
    else:
        words = K.alloc_local([4], K.u32)
        K.ptx.ld.shared.v4.b32(
            words[0], words[1], words[2], words[3], shared_raw.ptr_to([byte_offset])
        )
        for pair in range(4):
            K.ptx.mov.b32(
                values[value_offset + pair * 2], values[value_offset + pair * 2 + 1], words[pair]
            )


def _store_global_element(buffer, index, values, value_offset: int, predicate, vec: int) -> None:
    """Write *vec* halves from ``values[value_offset:]`` to ``buffer[index]`` under *predicate*."""
    if vec == 1:
        K.ptx.st.global_.b16(buffer.ptr_to([index]), values[value_offset], pred=predicate)
    elif vec == 2:
        K.ptx.st.global_.v2.b16(
            buffer.ptr_to([index]), values[value_offset], values[value_offset + 1], pred=predicate
        )
    else:
        K.ptx.st.global_.v4.b16(
            buffer.ptr_to([index]),
            values[value_offset],
            values[value_offset + 1],
            values[value_offset + 2],
            values[value_offset + 3],
            pred=predicate,
        )


def _store_global_packed(buffer, index, words, word_offset: int, predicate, vec: int) -> None:
    """Write *vec* halves from the packed ``words[word_offset:]`` under *predicate*."""
    if vec == 2:
        K.ptx.st.global_.b32(buffer.ptr_to([index]), words[word_offset], pred=predicate)
    elif vec == 4:
        K.ptx.st.global_.v2.b32(
            buffer.ptr_to([index]), words[word_offset], words[word_offset + 1], pred=predicate
        )
    else:
        K.ptx.st.global_.v4.b32(
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
    cvt_to_f32 = _CVT_TO_F32[dtype]
    cvt_from_f32 = _CVT_FROM_F32[dtype]
    cvt_pair_from_f32 = _CVT_PAIR_FROM_F32[dtype]
    fma_half_to_f32 = _FMA_HALF_TO_F32[dtype]
    # The packed x staging is only reachable on the synchronous full-tile
    # power-of-two shape; elsewhere the f32 lanes carry x directly.
    packed_x_pairs = not use_async and full_tile and vb_pow2

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

    @K.kernel(warps=threads // 32, arch="sm_100a", grid=False, thread_layout=False)
    def flashinfer_qk_rmsnorm(
        x: K.gptr[dtype],
        weight: K.gptr[dtype, (H,)],
        y: K.gptr[dtype],
        runtime_B: K.i64,
        runtime_N: K.i64,
        runtime_eps: K.f32,
        x_batch_stride: K.i64,
        x_head_stride: K.i64,
        y_batch_stride: K.i64,
        y_head_stride: K.i64,
    ):
        # QK_RMSNORM_KERNEL_START
        block_raw = K.cta_id([K.cast(K.ceildiv(runtime_B * runtime_N, K.int64(rows)), "int32")])
        tid = K.thread_id([threads])

        if enable_pdl:
            K.ptx.griddepcontrol.wait()

        block = K.cast(block_raw, "int32")
        row_in_cta = tid // tpr
        thread_in_row = tid % tpr
        # The row index is 32-bit in the source. A ``K.gptr`` axis is int64 and
        # the simplifier distributes a cast over a sum, so landing the finished
        # int32 value in a local gives the widening a single Var to sit on.
        row_i32 = K.local_scalar(K.i32, init=block * rows + row_in_cta)
        row_i64 = K.cast(row_i32, "int64")
        row_valid = row_i64 < runtime_B * runtime_N
        # One 64-bit divide and remainder per thread: must run once.
        batch_idx = K.local_scalar(K.i64, init=row_i64 // runtime_N)
        head_idx = K.local_scalar(K.i64, init=row_i64 % runtime_N)
        warp = tid // 32
        lane = tid % 32
        row_warp = warp // warps_per_row
        warp_in_row = warp % warps_per_row

        shared_raw = K.smem_pool().alloc([smem_bytes], K.u8)

        x_bits = K.alloc_local([pair_values], K.u16)
        w_bits = K.alloc_local([pair_values], K.u16)
        x_f32 = K.alloc_local([pair_values], K.f32)
        x_f32_pairs = K.alloc_local([packed_pairs], K.u64)
        w_f32 = K.alloc_local([pair_values], K.f32)
        # The odd upper half of the last pair is a don't-care lane the source
        # leaves uninitialized; reading it keeps the packed shape intact.
        undefined_f32 = K.alloc_local([1], K.f32)
        packed = K.local_scalar(K.u64)

        if not use_async:
            # A masked lane must contribute exact zeros, and the source zeroes
            # whichever staging shape that specialization goes on to read.
            if packed_x_pairs:
                zero = K.local_scalar(K.f32)
                K.ptx.mov.b32(zero, K.float32(0))
                for pair in range(packed_pairs):
                    K.ptx.mov.b64(x_f32_pairs[pair], zero, zero)
            elif full_tile:
                for value in range(total_values):
                    K.ptx.mov.b32(x_f32[value], K.float32(0))
            elif vb_pow2 and total_values > 1:
                for pair in range(total_values // 2):
                    K.ptx.mov.b32(x_bits[pair * 2], x_bits[pair * 2 + 1], K.uint32(0))
            else:
                for value in range(total_values):
                    K.ptx.mov.b16(x_bits[value], K.uint16(0))

        for vb in range(vec_blocks):
            local_col = (thread_in_row + vb * tpr) * vec
            col_valid = local_col < H
            x_offset = (
                batch_idx * x_batch_stride + head_idx * x_head_stride + K.cast(local_col, "int64")
            )
            if use_async:
                with K.If(row_valid), K.Then():
                    # Ignore-src: an out-of-range column copies nothing and the
                    # staged bytes read back as zero.
                    K.ptx[_CP_ASYNC](
                        shared_raw.ptr_to([(row_in_cta * cols + local_col) * _ELEM_BYTES]),
                        x.ptr_to([x_offset]),
                        copy_bytes,
                        K.cast(K.if_then_else(col_valid, copy_bytes, 0), "uint32"),
                    )
            else:
                with K.If(K.And(row_valid, col_valid)), K.Then():
                    if sync_x_packed:
                        _load_global_packed(x_bits, vb * vec, x, x_offset, vec)
                    else:
                        if total_values == 1:
                            K.ptx.mov.b16(x_bits[0], K.uint16(0))
                        _load_global_element(x_bits, vb * vec, x, x_offset, vec)

        if use_async:
            K.ptx.cp.async_.commit_group()

        for vb in range(vec_blocks):
            local_col = (thread_in_row + vb * tpr) * vec
            with K.If(local_col < H), K.Then():
                if weight_uses_packed(vb):
                    _load_global_packed(w_bits, vb * vec, weight, local_col, vec)
                else:
                    _load_global_element(w_bits, vb * vec, weight, local_col, vec)

        if use_async:
            K.ptx.cp.async_.wait_group(0)
            for vb in range(vec_blocks):
                local_col = (thread_in_row + vb * tpr) * vec
                _load_shared_bits(
                    x_bits, vb * vec, shared_raw, (row_in_cta * cols + local_col) * _ELEM_BYTES, vec
                )

        if packed_x_pairs:
            with K.If(row_valid), K.Then():
                for pair in range(packed_pairs):
                    x_low = K.local_scalar(K.f32)
                    x_high = K.local_scalar(K.f32)
                    K.ptx[cvt_to_f32](x_low, x_bits[pair * 2])
                    K.ptx[cvt_to_f32](x_high, x_bits[pair * 2 + 1])
                    K.ptx.mov.b64(x_f32_pairs[pair], x_low, x_high)
        elif not use_async and full_tile:
            with K.If(row_valid), K.Then():
                for value in range(total_values):
                    K.ptx[cvt_to_f32](x_f32[value], x_bits[value])
        else:
            for value in range(total_values):
                K.ptx[cvt_to_f32](x_f32[value], x_bits[value])

        local_sum = K.local_scalar(K.f32)
        if total_values == 1:
            K.ptx[fma_half_to_f32](local_sum, x_bits[0], x_bits[0], K.float32(0.0))
        else:
            x_sq = K.alloc_local([pair_values], K.f32)
            for pair in range(packed_pairs):
                if packed_x_pairs:
                    x_pair = x_f32_pairs[pair]
                else:
                    # Both squaring operands are the same packed value; land it
                    # once instead of emitting the pack twice.
                    x_pair = K.local_scalar(
                        K.u64, init=K.cuda.make_float2(x_f32[pair * 2], x_f32[pair * 2 + 1])
                    )
                K.ptx.mul.f32x2(packed, x_pair, x_pair)
                K.ptx.mov.b64(x_sq[pair * 2], x_sq[pair * 2 + 1], packed)
            K.assign(local_sum, K.float32(0.0))
            for value in range(total_values):
                K.ptx.add.f32(local_sum, local_sum, x_sq[value])

        _butterfly_sum_f32(local_sum, row_lane_xors)
        warp_sum = local_sum

        if warps_per_row > 1:
            with K.If(lane == 0), K.Then():
                K.ptx.st.shared.b32(
                    shared_raw.ptr_to([reduce_base + (row_warp + warp_in_row * rows) * 4]),
                    K.reinterpret("uint32", warp_sum),
                )
            K.ptx.bar.sync(K.uint32(0))
            final_sum = K.local_scalar(K.f32, init=K.float32(0.0))
            with K.If(lane < warps_per_row), K.Then():
                reduce_word = K.local_scalar(K.u32)
                K.ptx.ld.shared.b32(
                    reduce_word, shared_raw.ptr_to([reduce_base + (row_warp + lane * rows) * 4])
                )
                K.assign(final_sum, K.reinterpret("float32", reduce_word))
            _butterfly_sum_f32(final_sum, full_lane_xors)
            sum_sq = final_sum
        else:
            sum_sq = warp_sum

        shifted = K.local_scalar(K.f32)
        if H == 1:
            K.ptx.add.f32(shifted, sum_sq, runtime_eps)
        elif H & (H - 1) == 0:
            K.ptx.fma.rn.f32(shifted, sum_sq, K.float32(1.0 / H), runtime_eps)
        else:
            K.ptx.div.rn.f32(shifted, sum_sq, K.float32(H))
            K.ptx.add.f32(shifted, shifted, runtime_eps)
        rstd = K.local_scalar(K.f32)
        K.ptx.rsqrt.approx.ftz.f32(rstd, shifted)

        K.ptx.bar.sync(K.uint32(0))

        if use_async:
            for vb in range(vec_blocks):
                local_col = (thread_in_row + vb * tpr) * vec
                _load_shared_bits(
                    x_bits, vb * vec, shared_raw, (row_in_cta * cols + local_col) * _ELEM_BYTES, vec
                )
            for value in range(total_values):
                K.ptx[cvt_to_f32](x_f32[value], x_bits[value])

        for value in range(total_values):
            K.ptx[cvt_to_f32](w_f32[value], w_bits[value])

        if total_values == 1:
            K.ptx.mul.f32(x_f32[0], x_f32[0], rstd)
            K.ptx.add.f32(w_f32[0], w_f32[0], K.float32(weight_bias))
            K.ptx.mul.f32(x_f32[0], x_f32[0], w_f32[0])
        else:
            for pair in range(packed_pairs):
                high_scale = rstd if pair * 2 + 1 < total_values else undefined_f32[0]
                x_pair = (
                    x_f32_pairs[pair]
                    if packed_x_pairs
                    else K.cuda.make_float2(x_f32[pair * 2], x_f32[pair * 2 + 1])
                )
                K.ptx.mul.f32x2(packed, x_pair, K.cuda.make_float2(rstd, high_scale))
                K.ptx.mov.b64(x_f32[pair * 2], x_f32[pair * 2 + 1], packed)

            for pair in range(packed_pairs):
                high_bias = (
                    K.float32(weight_bias) if pair * 2 + 1 < total_values else undefined_f32[0]
                )
                K.ptx.add.f32x2(
                    packed,
                    K.cuda.make_float2(w_f32[pair * 2], w_f32[pair * 2 + 1]),
                    K.cuda.make_float2(K.float32(weight_bias), high_bias),
                )
                K.ptx.mov.b64(w_f32[pair * 2], w_f32[pair * 2 + 1], packed)

            for pair in range(packed_pairs):
                K.ptx.mul.f32x2(
                    packed,
                    K.cuda.make_float2(x_f32[pair * 2], x_f32[pair * 2 + 1]),
                    K.cuda.make_float2(w_f32[pair * 2], w_f32[pair * 2 + 1]),
                )
                K.ptx.mov.b64(x_f32[pair * 2], x_f32[pair * 2 + 1], packed)

        y_bits = K.alloc_local([pair_values], K.u16)
        y_words = K.alloc_local([packed_pairs], K.u32)
        if packed_narrow:
            for pair in range(packed_pairs):
                K.ptx[cvt_pair_from_f32](y_words[pair], x_f32[pair * 2 + 1], x_f32[pair * 2])
        else:
            for value in range(total_values):
                K.ptx[cvt_from_f32](y_bits[value], x_f32[value])
            if output_packed:
                for pair in range(total_values // 2):
                    K.ptx.mov.b32(y_words[pair], y_bits[pair * 2], y_bits[pair * 2 + 1])

        for vb in range(vec_blocks):
            local_col = (thread_in_row + vb * tpr) * vec
            col_valid = local_col < H
            y_offset = (
                batch_idx * y_batch_stride + head_idx * y_head_stride + K.cast(local_col, "int64")
            )
            predicate = K.And(row_valid, col_valid)
            if output_packed:
                _store_global_packed(y, y_offset, y_words, vb * vec // 2, predicate, vec)
            else:
                _store_global_element(y, y_offset, y_bits, vb * vec, predicate, vec)

        if enable_pdl:
            K.ptx.griddepcontrol.launch_dependents()

    launch_params = ["blockIdx.x", "threadIdx.x"]
    if enable_pdl:
        launch_params.append("tirx.use_programtic_dependent_launch")
    launch_params.append("tirx.use_dyn_shared_memory")
    return flashinfer_qk_rmsnorm.func.with_attr("tir.is_entry_func", True).with_attr(
        "tirx.kernel_launch_params", launch_params
    )


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
