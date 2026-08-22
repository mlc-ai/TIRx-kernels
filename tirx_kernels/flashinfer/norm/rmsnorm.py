# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400e330fb2debe0bf8730d9424a1d37927f), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer CuTe-DSL 2-D RMSNorm and Gemma RMSNorm port.

The source implementation is ``RMSNormKernel`` plus its 2-D host dispatch in
``flashinfer/norm/kernels/rmsnorm.py``.  The public entry points are
``rmsnorm`` and ``gemma_rmsnorm`` in ``flashinfer/norm/__init__.py``.
"""

import contextlib
from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.runner import bench

KERNEL_META = {"name": "flashinfer_rmsnorm", "category": "flashinfer", "compute_capability": 10}

_VARIANTS = ("rmsnorm", "gemma_rmsnorm")
_DTYPES = ("float16", "bfloat16")
_LAYOUTS = ("compact", "strided")
_INT32_MAX = 2**31 - 1
_DEFAULT_EPS = 1e-6
_OPTIN_SMEM_BYTES = 232448
_ELEM_BYTES = 2

# The half-precision instruction each dtype spells, selected once per
# specialization and applied at the call sites below.
_CVT_TO_F32 = {"float16": "cvt.f32.f16", "bfloat16": "cvt.f32.bf16"}
_CVT_FROM_F32 = {"float16": "cvt.rn.f16.f32", "bfloat16": "cvt.rn.bf16.f32"}
_CVT_PAIR_FROM_F32 = {"float16": "cvt.rn.f16x2.f32", "bfloat16": "cvt.rn.bf16x2.f32"}
_CP_ASYNC = "cp.async.ca.shared.global"


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


def _derived_config(H: int, cluster_n: int) -> dict[str, int | bool]:
    H_per_cta = H // cluster_n
    tpr = _threads_per_row(H_per_cta)
    threads = 128 if H_per_cta <= 16384 else 256
    rows = threads // tpr
    warps_per_row = max(tpr // 32, 1)
    vec = min(H_per_cta & -H_per_cta, 8)
    copy_bits = 16 * vec
    vec_blocks = max(1, _ceil_div(H_per_cta // vec, tpr))
    cols = vec * vec_blocks * tpr
    tile_bytes = rows * cols * _ELEM_BYTES
    use_async = copy_bits >= 32 and tile_bytes <= _OPTIN_SMEM_BYTES // 2
    reduce_bytes = rows * warps_per_row * 4
    if cluster_n > 1:
        reduce_bytes *= cluster_n
    smem_bytes = (tile_bytes if use_async else 0) + reduce_bytes
    if cluster_n > 1:
        smem_bytes += 8
    return {
        "cluster_n": cluster_n,
        "H_per_cta": H_per_cta,
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


def _estimate_smem(H: int, cluster_n: int) -> int:
    config = _derived_config(H, cluster_n)
    rows = int(config["rows"])
    warps_per_row = int(config["warps_per_row"])
    cols = int(config["cols"])
    return (
        rows * cols * _ELEM_BYTES
        + rows * warps_per_row * cluster_n * 4
        + (8 if cluster_n > 1 else 0)
    )


def _source_config(H: int) -> dict[str, int | bool]:
    cluster_n = 16
    for candidate in (1, 2, 4, 8, 16):
        if H % candidate == 0 and _estimate_smem(H, candidate) <= _OPTIN_SMEM_BYTES:
            cluster_n = candidate
            break
    return _derived_config(H, cluster_n)


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


def _load_global_bits(values, value_offset: int, buffer, index, vec: int) -> None:
    """``values[value_offset : +vec]`` <- *vec* halves at ``buffer[index]``.

    One ``ld.global`` of the widest shape the vector width allows; the b32
    forms land two halves per word and are split into the half lanes.
    """
    if vec == 1:
        K.ptx.ld.global_.b16(values[value_offset], buffer.ptr_to([index]))
    elif vec == 2:
        K.ptx.ld.global_.v2.b16(
            values[value_offset], values[value_offset + 1], buffer.ptr_to([index])
        )
    else:
        words = K.alloc_local([vec // 2], K.u32)
        if vec == 4:
            K.ptx.ld.global_.v2.b32(words[0], words[1], buffer.ptr_to([index]))
        else:
            K.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
        for pair in range(vec // 2):
            K.assign(
                values[value_offset + pair * 2],
                K.cast(K.bitwise_and(words[pair], K.uint32(0xFFFF)), "uint16"),
            )
            K.assign(
                values[value_offset + pair * 2 + 1],
                K.cast(K.shift_right(words[pair], K.uint32(16)), "uint16"),
            )


def _load_shared_bits(values, value_offset: int, shared_raw, byte_offset, vec: int) -> None:
    """``values[value_offset : +vec]`` <- *vec* halves staged at ``shared_raw + byte_offset``.

    ``vec == 4`` reads the four halves directly; the eight-half shape reads
    four words and splits them, mirroring the global path.
    """
    if vec == 2:
        K.ptx.ld.shared.v2.b16(
            values[value_offset], values[value_offset + 1], shared_raw.ptr_to([byte_offset])
        )
    elif vec == 4:
        halves = K.alloc_local([4], K.u16)
        K.ptx.ld.shared.v4.b16(
            halves[0], halves[1], halves[2], halves[3], shared_raw.ptr_to([byte_offset])
        )
        for value in range(4):
            K.assign(values[value_offset + value], halves[value])
    else:
        words = K.alloc_local([4], K.u32)
        K.ptx.ld.shared.v4.b32(
            words[0], words[1], words[2], words[3], shared_raw.ptr_to([byte_offset])
        )
        for pair in range(4):
            K.assign(
                values[value_offset + pair * 2],
                K.cast(K.bitwise_and(words[pair], K.uint32(0xFFFF)), "uint16"),
            )
            K.assign(
                values[value_offset + pair * 2 + 1],
                K.cast(K.shift_right(words[pair], K.uint32(16)), "uint16"),
            )


def _store_global_fragment(
    buffer,
    index,
    bits,
    words,
    predicate,
    value_offset: int,
    word_offset: int,
    vec: int,
    packed_narrow: bool,
) -> None:
    """Write one *vec*-wide fragment to ``buffer[index]`` under *predicate*.

    The narrow ``vec == 2`` shape has no b32 store that keeps the source in
    ``words``, so its two halves are unpacked into ``bits`` first.
    """
    if vec == 1:
        K.ptx.st.global_.b16(buffer.ptr_to([index]), bits[value_offset], pred=predicate)
    elif vec == 2:
        if packed_narrow:
            K.assign(
                bits[value_offset],
                K.cast(K.bitwise_and(words[word_offset], K.uint32(0xFFFF)), "uint16"),
            )
            K.assign(
                bits[value_offset + 1],
                K.cast(K.shift_right(words[word_offset], K.uint32(16)), "uint16"),
            )
        K.ptx.st.global_.v2.b16(
            buffer.ptr_to([index]), bits[value_offset], bits[value_offset + 1], pred=predicate
        )
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


def _short_variant(variant: str) -> str:
    return {"rmsnorm": "rms", "gemma_rmsnorm": "gemma"}[variant]


def _short_dtype(dtype: str) -> str:
    return {"float16": "fp16", "bfloat16": "bf16"}[dtype]


def _layout_code(layout: str) -> str:
    return {"compact": "c", "strided": "s"}[layout]


def _cfg(
    variant: str,
    dtype: str,
    M: int,
    H: int,
    input_layout: str,
    output_layout: str,
    enable_pdl: bool,
    *,
    eps: float = _DEFAULT_EPS,
    x_row_stride: int | None = None,
    y_row_stride: int | None = None,
    suffix: str | None = None,
) -> dict[str, Any]:
    label = (
        f"{_short_variant(variant)}_{_short_dtype(dtype)}_m{M}_h{H}_"
        f"x{_layout_code(input_layout)}_y{_layout_code(output_layout)}_"
        f"pdl{int(enable_pdl)}"
    )
    if suffix:
        label += f"_{suffix}"
    config: dict[str, Any] = {
        "label": label,
        "variant": variant,
        "dtype": dtype,
        "M": M,
        "H": H,
        "input_layout": input_layout,
        "output_layout": output_layout,
        "enable_pdl": enable_pdl,
        "eps": eps,
    }
    if x_row_stride is not None:
        config["x_row_stride"] = x_row_stride
    if y_row_stride is not None:
        config["y_row_stride"] = y_row_stride
    return config


# The 256 upstream 2-D pytest combinations.  ``run_test`` executes each entry
# once with implicit FlashInfer output and once with a caller-provided output.
_UPSTREAM_CONFIGS = [
    _cfg(
        variant,
        "float16",
        M,
        H,
        input_layout,
        "compact",
        enable_pdl,
        x_row_stride=2 * H if input_layout == "strided" else None,
    )
    for variant in _VARIANTS
    for M in (1, 19, 99, 989)
    for H in (111, 500, 1024, 3072, 3584, 4096, 8192, 16384)
    for input_layout in _LAYOUTS
    for enable_pdl in (False, True)
]

# Every 2-D performance workload supplied by the pinned FlashInfer source.
BENCH_CONFIGS = [
    _cfg("rmsnorm", "bfloat16", 32, 4096, "compact", "compact", False),
    _cfg("rmsnorm", "bfloat16", 64, 8192, "compact", "compact", False),
    _cfg("rmsnorm", "bfloat16", 32, 4096, "compact", "compact", True),
    _cfg("gemma_rmsnorm", "bfloat16", 32, 4096, "compact", "compact", False),
    _cfg("gemma_rmsnorm", "bfloat16", 64, 8192, "compact", "compact", False),
]

_I64_CONFIGS = [
    _cfg(
        "rmsnorm",
        "bfloat16",
        2,
        128,
        "strided",
        "compact",
        False,
        x_row_stride=2**31,
        suffix="i64_xstride",
    ),
    _cfg(
        "rmsnorm",
        "float16",
        175000,
        12288,
        "compact",
        "compact",
        False,
        suffix="compact_overflow_strided",
    ),
    _cfg(
        "gemma_rmsnorm",
        "float16",
        175000,
        12288,
        "compact",
        "compact",
        False,
        suffix="compact_overflow_strided",
    ),
]

_TRACE_CONFIGS = [
    _cfg("rmsnorm", "bfloat16", 8, 256, "compact", "compact", False),
    _cfg("gemma_rmsnorm", "bfloat16", 8, 256, "compact", "compact", False),
    _cfg("rmsnorm", "bfloat16", 3, 320, "compact", "compact", False),
    _cfg("gemma_rmsnorm", "bfloat16", 3, 320, "compact", "compact", False),
    _cfg("rmsnorm", "bfloat16", 32, 7168, "compact", "compact", False),
    _cfg("gemma_rmsnorm", "bfloat16", 32, 4608, "compact", "compact", False),
]

_STRUCTURE_CONFIGS = [
    _cfg("rmsnorm", "float16", 3, 64, "compact", "compact", False, suffix="tpr8"),
    _cfg("gemma_rmsnorm", "bfloat16", 3, 66, "compact", "compact", True, suffix="vec2"),
    _cfg("rmsnorm", "float16", 3, 16385, "compact", "compact", False, suffix="sync_vec1"),
    _cfg(
        "gemma_rmsnorm",
        "bfloat16",
        3,
        65536,
        "compact",
        "compact",
        True,
        suffix="sync_capacity_cluster1",
    ),
    _cfg("rmsnorm", "float16", 3, 131072, "compact", "compact", False, suffix="cluster2"),
    _cfg("gemma_rmsnorm", "bfloat16", 3, 262144, "compact", "compact", True, suffix="cluster4"),
    _cfg("rmsnorm", "float16", 3, 524288, "compact", "compact", False, suffix="cluster8"),
    _cfg("gemma_rmsnorm", "bfloat16", 3, 1048576, "compact", "compact", True, suffix="cluster16"),
]

_ABI_CONFIGS = [
    _cfg(
        "rmsnorm",
        "float16",
        19,
        500,
        "compact",
        "strided",
        True,
        eps=1e-4,
        y_row_stride=1000,
        suffix="eps1e4_abi",
    )
]

CONFIGS = [
    *_UPSTREAM_CONFIGS,
    *BENCH_CONFIGS,
    *_I64_CONFIGS,
    *_TRACE_CONFIGS,
    *_STRUCTURE_CONFIGS,
    *_ABI_CONFIGS,
]

assert len(CONFIGS) == 279
assert len({config["label"] for config in CONFIGS}) == len(CONFIGS)
assert len(BENCH_CONFIGS) == 5


def _validate(
    variant: str, dtype: str, M: int, H: int, input_layout: str, output_layout: str, eps: float
) -> None:
    if variant not in _VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported dtype: {dtype}")
    if input_layout not in _LAYOUTS or output_layout not in _LAYOUTS:
        raise ValueError(f"unsupported layouts: input={input_layout}, output={output_layout}")
    if M <= 0 or H <= 0:
        raise ValueError(f"M and H must be positive, got M={M}, H={H}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")


def _uses_compact_specialization(M: int, H: int, input_layout: str, output_layout: str) -> bool:
    return input_layout == "compact" and output_layout == "compact" and M * H <= _INT32_MAX


def get_kernel(
    variant: str,
    dtype: str,
    M: int,
    H: int,
    input_layout: str,
    output_layout: str,
    enable_pdl: bool,
    eps: float = _DEFAULT_EPS,
    **kwargs: Any,
):
    """Return the compact or explicit-i64-strided runtime-M specialization."""
    _validate(variant, dtype, M, H, input_layout, output_layout, eps)
    compact = _uses_compact_specialization(M, H, input_layout, output_layout)
    source = _source_config(H)
    cluster_n = int(source["cluster_n"])
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
    packed_narrow = not (vec == 1 or (vec == 2 and vec_blocks == 3))
    weight_bias = 0.0 if variant == "rmsnorm" else 1.0
    copy_bytes = copy_bits // 8
    reduce_base = tile_bytes if use_async else 0
    reduce_count = rows * warps_per_row * cluster_n
    mbar_offset = reduce_base + reduce_count * 4
    expected_bytes = reduce_count * 4
    total_partials_per_row = warps_per_row * cluster_n
    row_lane_xors = tuple(lane_xor for lane_xor in (1, 2, 4, 8, 16) if lane_xor < min(tpr, 32))
    full_lane_xors = (1, 2, 4, 8, 16)
    cvt_to_f32 = _CVT_TO_F32[dtype]
    cvt_from_f32 = _CVT_FROM_F32[dtype]
    cvt_pair_from_f32 = _CVT_PAIR_FROM_F32[dtype]

    x_row_stride_hint = kwargs.get("x_row_stride", H)
    y_row_stride_hint = kwargs.get("y_row_stride", H)
    if x_row_stride_hint is not None and int(x_row_stride_hint) % vec != 0:
        raise ValueError(f"x_row_stride={x_row_stride_hint} must be divisible by vec={vec}")
    if y_row_stride_hint is not None and int(y_row_stride_hint) % vec != 0:
        raise ValueError(f"y_row_stride={y_row_stride_hint} must be divisible by vec={vec}")

    # A 128-thread entry caps its own allocation; the 256-thread entry pins one
    # CTA per SM instead, and the two spellings are mutually exclusive downstream.
    max_registers = None
    if threads == 128:
        max_registers = 64 if enable_pdl else (96 if H == 8192 else 93)

    def entry_registers():
        if max_registers is None:
            return contextlib.nullcontext()
        return K.attr({"tirx.max_registers": max_registers})

    def kernel_body(x, weight, y, runtime_M, runtime_eps, x_row_stride, y_row_stride):
        # TIRX_TRANSCRIBE_START flashinfer_rmsnorm
        if cluster_n > 1:
            block_x_raw, block_y_raw = K.cta_id(
                [K.cast(K.ceildiv(runtime_M, K.int64(rows)), "int32"), cluster_n]
            )
            _, cta_rank_raw = K.cta_id_in_cluster([1, cluster_n], preferred=[1, cluster_n])
            block_y = K.cast(block_y_raw, "int32")
            cta_rank = K.cast(cta_rank_raw, "int32")
        else:
            block_x_raw = K.cta_id([K.cast(K.ceildiv(runtime_M, K.int64(rows)), "int32")])
            block_y = 0
            cta_rank = 0
        tid = K.thread_id()

        if enable_pdl:
            K.ptx.griddepcontrol.wait()

        block_x = K.cast(block_x_raw, "int32")
        row_in_cta = tid // tpr
        thread_in_row = tid % tpr
        # The row index and, in the compact layout, the whole element offset are
        # 32-bit in the source. A ``K.gptr`` axis is int64, and a cast applied to
        # a *sum* is distributed over its terms by the simplifier, so each term
        # would widen on its own (one IMAD.WIDE apiece). Landing the finished
        # int32 value in a local first gives the cast a single Var to sit on.
        row_i32 = K.local_scalar(K.i32, init=block_x * rows + row_in_cta)
        row_i64 = K.cast(row_i32, "int64")
        row_valid = row_i64 < runtime_M
        warp = tid // 32
        lane = tid % 32
        row_warp = warp // warps_per_row
        warp_in_row = warp % warps_per_row

        shared_raw = K.smem_pool().alloc([smem_bytes], K.u8)

        if cluster_n > 1:
            with K.If(tid == 0), K.Then():
                K.ptx.mbarrier.init.shared.b64(shared_raw.ptr_to([mbar_offset]), K.uint32(1))
            K.ptx.fence.mbarrier_init.release.cluster()
            K.ptx.barrier.cluster.arrive.relaxed()
            K.ptx.barrier.cluster.wait()

        x_bits = K.alloc_local([pair_values], K.u16)
        w_bits = K.alloc_local([pair_values], K.u16)
        x_f32 = K.alloc_local([pair_values], K.f32)
        w_f32 = K.alloc_local([pair_values], K.f32)
        x_sq = K.alloc_local([pair_values], K.f32)
        # The odd upper half of the last pair is a don't-care lane the source
        # leaves uninitialized; reading it keeps the packed shape intact.
        undefined_f32 = K.local_scalar(K.f32)
        packed = K.local_scalar(K.u64)

        if compact:
            element = K.local_scalar(K.i32)

            def compact_index(value):
                """One compact element offset, widened to the int64 axis exactly once."""
                K.assign(element, value)
                return K.cast(element, "int64")

        if not use_async:
            for value in range(total_values):
                K.assign(x_bits[value], K.uint16(0))

        for vb in range(vec_blocks):
            local_col = (thread_in_row + vb * tpr) * vec
            absolute_col = block_y * cols + local_col
            col_valid = absolute_col < H
            if compact:
                x_offset = compact_index(row_i32 * H + absolute_col)
            else:
                x_offset = row_i64 * x_row_stride + K.cast(absolute_col, "int64")

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
                    _load_global_bits(x_bits, vb * vec, x, x_offset, vec)

        if use_async:
            K.ptx.cp.async_.commit_group()

        for vb in range(vec_blocks):
            local_col = (thread_in_row + vb * tpr) * vec
            absolute_col = block_y * cols + local_col
            with K.If(absolute_col < H), K.Then():
                _load_global_bits(w_bits, vb * vec, weight, absolute_col, vec)

        if use_async:
            K.ptx.cp.async_.wait_group(0)
            for vb in range(vec_blocks):
                local_col = (thread_in_row + vb * tpr) * vec
                _load_shared_bits(
                    x_bits, vb * vec, shared_raw, (row_in_cta * cols + local_col) * _ELEM_BYTES, vec
                )

        for value in range(total_values):
            K.ptx[cvt_to_f32](x_f32[value], x_bits[value])

        for pair in range(packed_pairs):
            K.ptx.mul.f32x2(
                packed,
                K.cuda.make_float2(x_f32[pair * 2], x_f32[pair * 2 + 1]),
                K.cuda.make_float2(x_f32[pair * 2], x_f32[pair * 2 + 1]),
            )
            K.ptx.mov.b64(x_sq[pair * 2], x_sq[pair * 2 + 1], packed)

        local_sum = K.local_scalar(K.f32, init=K.float32(0.0))
        for value in range(total_values):
            K.ptx.add.f32(local_sum, local_sum, x_sq[value])

        _butterfly_sum_f32(local_sum, row_lane_xors)
        warp_sum = local_sum

        if warps_per_row > 1 and cluster_n == 1:
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
        elif cluster_n > 1:
            with K.If(warp == 0), K.Then():
                with K.If(K.cuda.elect_sync()), K.Then():
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        shared_raw.ptr_to([mbar_offset]), K.uint32(expected_bytes)
                    )
            with K.If(lane < cluster_n), K.Then():
                reduce_index = row_warp + warp_in_row * rows + cta_rank * rows * warps_per_row
                peer_reduce = K.local_scalar(K.u32)
                K.ptx.mapa.shared__cluster.u32(
                    peer_reduce,
                    K.cuda.cvta_generic_to_shared(
                        shared_raw.ptr_to([reduce_base + reduce_index * 4])
                    ),
                    K.cast(lane, "uint32"),
                )
                peer_mbar = K.local_scalar(K.u32)
                K.ptx.mapa.shared__cluster.u32(
                    peer_mbar,
                    K.cuda.cvta_generic_to_shared(shared_raw.ptr_to([mbar_offset])),
                    K.cast(lane, "uint32"),
                )
                K.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.f32(
                    peer_reduce, warp_sum, peer_mbar
                )

            K.cuda.mbarrier_wait(shared_raw.ptr_to([mbar_offset]), K.int32(0))

            final_sum = K.local_scalar(K.f32, init=K.float32(0.0))
            for iteration in range(_ceil_div(total_partials_per_row, 32)):
                partial = lane + iteration * 32
                with K.If(partial < total_partials_per_row), K.Then():
                    reduce_index = (
                        row_warp
                        + (partial % warps_per_row) * rows
                        + (partial // warps_per_row) * rows * warps_per_row
                    )
                    reduce_word = K.local_scalar(K.u32)
                    K.ptx.ld.shared.b32(
                        reduce_word, shared_raw.ptr_to([reduce_base + reduce_index * 4])
                    )
                    K.ptx.add.f32(final_sum, final_sum, K.reinterpret("float32", reduce_word))
            _butterfly_sum_f32(final_sum, full_lane_xors)
            sum_sq = final_sum
        else:
            sum_sq = warp_sum

        shifted = K.local_scalar(K.f32)
        if H & (H - 1) == 0:
            K.ptx.fma.rn.f32(shifted, sum_sq, K.float32(1.0 / H), runtime_eps)
        else:
            K.ptx.div.rn.f32(shifted, sum_sq, K.float32(H))
            K.ptx.add.f32(shifted, shifted, runtime_eps)
        rstd = K.local_scalar(K.f32)
        K.ptx.rsqrt.approx.ftz.f32(rstd, shifted)

        if cluster_n > 1:
            K.ptx.barrier.cluster.arrive.relaxed()
            K.ptx.barrier.cluster.wait()
        else:
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

        for pair in range(packed_pairs):
            high_scale = rstd if pair * 2 + 1 < total_values else undefined_f32
            K.ptx.mul.f32x2(
                packed,
                K.cuda.make_float2(x_f32[pair * 2], x_f32[pair * 2 + 1]),
                K.cuda.make_float2(rstd, high_scale),
            )
            K.ptx.mov.b64(x_f32[pair * 2], x_f32[pair * 2 + 1], packed)

        for pair in range(packed_pairs):
            high_bias = K.float32(weight_bias) if pair * 2 + 1 < total_values else undefined_f32
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

        for vb in range(vec_blocks):
            local_col = (thread_in_row + vb * tpr) * vec
            absolute_col = block_y * cols + local_col
            col_valid = absolute_col < H
            if compact:
                y_offset = compact_index(row_i32 * H + absolute_col)
            else:
                y_offset = row_i64 * y_row_stride + K.cast(absolute_col, "int64")
            _store_global_fragment(
                y,
                y_offset,
                y_bits,
                y_words,
                K.And(row_valid, col_valid),
                vb * vec,
                vb * vec // 2,
                vec,
                packed_narrow,
            )

        if enable_pdl:
            K.ptx.griddepcontrol.launch_dependents()

    entry_kwargs = {
        "warps": threads // 32,
        "arch": "sm_100a",
        "grid": False,
        "min_blocks_per_sm": None if threads == 128 else 1,
    }

    if compact:

        @K.kernel(**entry_kwargs)
        def flashinfer_rmsnorm_compact(
            x: K.gptr[dtype],
            weight: K.gptr[dtype, (H,)],
            y: K.gptr[dtype],
            runtime_M: K.i64,
            runtime_eps: K.f32,
        ):
            with entry_registers():
                kernel_body(x, weight, y, runtime_M, runtime_eps, K.int64(H), K.int64(H))

        kernel = flashinfer_rmsnorm_compact.func
    else:

        @K.kernel(**entry_kwargs)
        def flashinfer_rmsnorm_strided(
            x: K.gptr[dtype],
            weight: K.gptr[dtype, (H,)],
            y: K.gptr[dtype],
            runtime_M: K.i64,
            runtime_eps: K.f32,
            x_row_stride: K.i64,
            y_row_stride: K.i64,
        ):
            with entry_registers():
                kernel_body(x, weight, y, runtime_M, runtime_eps, x_row_stride, y_row_stride)

        kernel = flashinfer_rmsnorm_strided.func

    launch_params = ["blockIdx.x"]
    if cluster_n > 1:
        launch_params.extend(["blockIdx.y", "clusterCtaIdx.x", "clusterCtaIdx.y"])
    launch_params.append("threadIdx.x")
    if enable_pdl:
        launch_params.append("tirx.use_programtic_dependent_launch")
    launch_params.append("tirx.use_dyn_shared_memory")
    return kernel.with_attr("tirx.kernel_launch_params", launch_params)


def prepare_data(**config: Any):
    """Create deterministic tensors for one specialization."""
    data = _prepare_tensors(config)
    return data["x"], data["weight"]


_GUARD_ELEMENTS = 64
_GUARD_VALUE = 123.0


def _torch_dtype(dtype: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def _row_strides(config: dict[str, Any]) -> tuple[int, int]:
    H = int(config["H"])
    x_stride = int(config.get("x_row_stride", 2 * H if config["input_layout"] == "strided" else H))
    y_stride = int(config.get("y_row_stride", 2 * H if config["output_layout"] == "strided" else H))
    return x_stride, y_stride


def _storage_size(M: int, H: int, row_stride: int) -> int:
    return (M - 1) * row_stride + H


def _prepare_tensors(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    M = int(config["M"])
    H = int(config["H"])
    dtype = str(config["dtype"])
    input_layout = str(config["input_layout"])
    output_layout = str(config["output_layout"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    _validate(str(config["variant"]), dtype, M, H, input_layout, output_layout, eps)
    x_stride, y_stride = _row_strides(config)
    vec = int(_source_config(H)["vec"])
    if x_stride < H or y_stride < H:
        raise ValueError(f"row strides must cover H={H}: x={x_stride}, y={y_stride}")
    if x_stride % vec != 0 or y_stride % vec != 0:
        raise ValueError(f"row strides must be divisible by vec={vec}: x={x_stride}, y={y_stride}")

    torch_dtype = _torch_dtype(dtype)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(42)

    x_storage_size = _storage_size(M, H, x_stride)
    x_backing = torch.empty(x_storage_size + _GUARD_ELEMENTS, dtype=torch_dtype, device="cuda")
    x_arg = x_backing[:x_storage_size]
    x = x_arg.as_strided((M, H), (x_stride, 1))
    # The large-H structural cases isolate the DSMEM reduction topology with
    # exactly representable values and an exactly accumulated sum of squares.
    # The 257-column cycle and alternating rows also make X-address, CTA-slice,
    # and async-reload mistakes visible.  Random BF16 values at million-wide
    # shapes can make FlashInfer's approximate-rsqrt result miss its naive FP32
    # oracle by one BF16 bin despite matching the port bit-for-bit.
    if (dtype == "bfloat16" and H >= 4096) or H >= 65536:
        columns = torch.arange(H, device="cuda")
        magnitude = torch.where(columns % 257 < 128, 0.5, 1.0)
        pattern = torch.where(columns % 2 == 0, magnitude, -magnitude).to(torch_dtype)
        x.copy_(pattern.expand(M, H))
        x[1::2].neg_()
    else:
        x.normal_(generator=generator)
    x_backing[x_storage_size:].fill_(_GUARD_VALUE)

    weight_backing = torch.empty(H + _GUARD_ELEMENTS, dtype=torch_dtype, device="cuda")
    weight = weight_backing[:H]
    weight.normal_(generator=generator)
    weight_backing[H:].fill_(_GUARD_VALUE)

    return {
        "x": x,
        "x_arg": x_arg,
        "x_backing": x_backing,
        "x_storage_size": x_storage_size,
        "weight": weight,
        "weight_backing": weight_backing,
        "x_row_stride": x_stride,
        "y_row_stride": y_stride,
    }


def _prepare_output(
    M: int, H: int, row_stride: int, dtype: str, *, initialize_padding: bool
) -> dict[str, Any]:
    import torch

    size = _storage_size(M, H, row_stride)
    backing = torch.empty(size + _GUARD_ELEMENTS, dtype=_torch_dtype(dtype), device="cuda")
    if initialize_padding and row_stride > H:
        for row in range(M - 1):
            backing[row * row_stride + H : (row + 1) * row_stride].fill_(_GUARD_VALUE)
    backing[size:].fill_(_GUARD_VALUE)
    arg = backing[:size]
    view = arg.as_strided((M, H), (row_stride, 1))
    return {"view": view, "arg": arg, "backing": backing, "size": size}


def _assert_guard(backing, size: int, *, name: str) -> None:
    import torch

    expected = torch.full(
        (_GUARD_ELEMENTS,), _GUARD_VALUE, dtype=backing.dtype, device=backing.device
    )
    if not torch.equal(backing[size:], expected):
        raise AssertionError(f"{name} guard was modified")


def _assert_output_padding(
    output: dict[str, Any], M: int, H: int, row_stride: int, *, name: str
) -> None:
    import torch

    _assert_guard(output["backing"], output["size"], name=name)
    if row_stride > H:
        for row in range(M - 1):
            padding = output["backing"][row * row_stride + H : (row + 1) * row_stride]
            expected = torch.full_like(padding, _GUARD_VALUE)
            if not torch.equal(padding, expected):
                raise AssertionError(f"{name} row {row} padding was modified")


def _overflow_rows(M: int, H: int) -> list[int] | None:
    if M * H <= _INT32_MAX:
        return None
    boundary = _ceil_div(2**31, H)
    return sorted(
        {row for row in (0, 1, boundary - 1, boundary, boundary + 1, M - 1) if 0 <= row < M}
    )


def _checked_view(tensor, rows: list[int] | None):
    return tensor if rows is None else tensor[rows]


def _math_oracle(x, weight, variant: str, eps: float, rows: list[int] | None):
    x_checked = _checked_view(x, rows).float()
    weight_f32 = weight.float()
    variance = x_checked.square().mean(dim=-1, keepdim=True)
    normalized = x_checked * variance.add(eps).rsqrt()
    bias = 0.0 if variant == "rmsnorm" else 1.0
    return (normalized * (weight_f32 + bias)).to(dtype=x.dtype)


def _assert_close(actual, expected, *, name: str) -> None:
    import torch

    torch.testing.assert_close(
        actual, expected, rtol=1e-3, atol=1e-3, msg=lambda message: f"{name}: {message}"
    )


def _flashinfer_api(variant: str, device):
    import flashinfer
    import flashinfer.norm as flashinfer_norm

    if flashinfer_norm._use_cuda_norm(device):
        raise AssertionError("FlashInfer RMSNorm oracle dispatched to legacy CUDA")
    api = getattr(flashinfer, variant)
    return api, flashinfer_norm


def _launch_tirx(executable, data, output, config: dict[str, Any]) -> None:
    M = int(config["M"])
    H = int(config["H"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    compact = _uses_compact_specialization(
        M, H, str(config["input_layout"]), str(config["output_layout"])
    )
    if compact:
        executable(data["x_arg"], data["weight"], output["arg"], M, eps)
    else:
        executable(
            data["x_arg"],
            data["weight"],
            output["arg"],
            M,
            eps,
            data["x_row_stride"],
            data["y_row_stride"],
        )


def _input_snapshot(data, M: int, H: int) -> dict[str, Any]:
    rows = _overflow_rows(M, H)
    if rows is None:
        values = data["x"].clone()
    else:
        columns = sorted({0, H // 2, H - 1})
        values = data["x"][rows][:, columns].clone()
    return {"rows": rows, "values": values, "weight": data["weight"].clone()}


def _assert_inputs_unchanged(data, snapshot, M: int, H: int) -> None:
    import torch

    rows = snapshot["rows"]
    if rows is None:
        observed = data["x"]
    else:
        columns = sorted({0, H // 2, H - 1})
        observed = data["x"][rows][:, columns]
    if not torch.equal(observed, snapshot["values"]):
        raise AssertionError("input tensor was modified")
    if not torch.equal(data["weight"], snapshot["weight"]):
        raise AssertionError("weight tensor was modified")
    _assert_guard(data["x_backing"], data["x_storage_size"], name="input")
    _assert_guard(data["weight_backing"], H, name="weight")


def run_test(**config: Any) -> None:
    """Compile, launch, and validate one config."""
    import torch

    from tirx_kernels.runner import compile_kernel

    config = dict(config)
    M = int(config["M"])
    H = int(config["H"])
    dtype = str(config["dtype"])
    variant = str(config["variant"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    enable_pdl = bool(config["enable_pdl"])
    data = _prepare_tensors(config)
    snapshot = _input_snapshot(data, M, H)
    output = _prepare_output(M, H, data["y_row_stride"], dtype, initialize_padding=True)
    reference_out = _prepare_output(M, H, data["y_row_stride"], dtype, initialize_padding=True)

    kernel = get_kernel(**config)
    executable = compile_kernel(kernel)
    _launch_tirx(executable, data, output, config)

    api, flashinfer_norm = _flashinfer_api(variant, data["x"].device)
    original_cute = flashinfer_norm.rmsnorm_cute
    cute_calls = 0

    def tracked_cute(*args, **kwargs):
        nonlocal cute_calls
        cute_calls += 1
        return original_cute(*args, **kwargs)

    flashinfer_norm.rmsnorm_cute = tracked_cute
    try:
        reference_implicit = api(data["x"], data["weight"], eps, enable_pdl=enable_pdl)
        returned = api(
            data["x"], data["weight"], eps, out=reference_out["view"], enable_pdl=enable_pdl
        )
    finally:
        flashinfer_norm.rmsnorm_cute = original_cute
    if cute_calls != 2:
        raise AssertionError(f"expected two CuTe-DSL oracle dispatches, observed {cute_calls}")
    if returned is not reference_out["view"]:
        raise AssertionError("FlashInfer caller-provided out did not preserve object identity")

    torch.cuda.synchronize()
    rows = _overflow_rows(M, H)
    actual_checked = _checked_view(output["view"], rows)
    implicit_checked = _checked_view(reference_implicit, rows)
    explicit_checked = _checked_view(reference_out["view"], rows)
    oracle = _math_oracle(data["x"], data["weight"], variant, eps, rows)
    if not torch.isfinite(actual_checked).all():
        raise AssertionError("TIRx output contains non-finite values")
    _assert_close(actual_checked, implicit_checked, name="FlashInfer implicit output")
    _assert_close(actual_checked, explicit_checked, name="FlashInfer explicit output")
    _assert_close(actual_checked, oracle, name="FP32 math oracle")
    _assert_inputs_unchanged(data, snapshot, M, H)
    _assert_output_padding(output, M, H, data["y_row_stride"], name="TIRx output")
    _assert_output_padding(reference_out, M, H, data["y_row_stride"], name="FlashInfer output")


def prepare_bench(**config: Any):
    """Compile the selected TIRx specialization before GPU assignment."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(config), "executable": compile_kernel(get_kernel(**config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs: Any
):
    """Construct and validate both timed closures before benchmarking."""
    import torch

    config = dict(prepared["config"])
    config.update(kwargs)
    M = int(config["M"])
    H = int(config["H"])
    dtype = str(config["dtype"])
    variant = str(config["variant"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    enable_pdl = bool(config["enable_pdl"])
    data = _prepare_tensors(config)
    tirx_output = _prepare_output(M, H, data["y_row_stride"], dtype, initialize_padding=False)
    flashinfer_output = _prepare_output(M, H, data["y_row_stride"], dtype, initialize_padding=False)
    executable = prepared["executable"]

    def tirx_launch():
        _launch_tirx(executable, data, tirx_output, config)

    # Warm up and validate the TIRx closure before it enters the canonical timer.
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
    """Benchmark a TIRx specialization against FlashInfer CuTe-DSL."""
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
