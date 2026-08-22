# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400e330fb2debe0bf8730d9424a1d37927f), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer CuTe-DSL fused add RMSNorm family.

The source implementation is ``FusedAddRMSNormKernel`` and its 2-D host
dispatch in ``flashinfer/norm/kernels/fused_add_rmsnorm.py``.  The public entry
points are ``fused_add_rmsnorm`` and ``gemma_fused_add_rmsnorm`` in
``flashinfer/norm/__init__.py``.
"""

import contextlib
from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.runner import bench

from ._kern_helpers import (
    _add_f32,
    _butterfly_sum_f32,
    _ceil_div,
    _cluster_mbarrier_wait,
    _cvt_from_f32,
    _cvt_pair_from_f32,
    _cvt_to_f32,
    _div_rn_f32,
    _fma_rn_f32,
    _load_global_bits,
    _load_shared_bits,
    _mapa_u32,
    _rsqrt_approx_ftz,
    _store_global_fragment,
    _threads_per_row,
)

KERNEL_META = {
    "name": "flashinfer_fused_add_rmsnorm",
    "category": "flashinfer",
    "compute_capability": 10,
}

_VARIANTS = ("fused_add_rmsnorm", "gemma_fused_add_rmsnorm")
_DTYPES = ("float16", "bfloat16")
_LAYOUTS = ("compact", "strided")
_INT32_MAX = 2**31 - 1
_DEFAULT_EPS = 1e-6
_OPTIN_SMEM_BYTES = 232448
_ELEM_BYTES = 2


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
    two_tile_bytes = 2 * tile_bytes
    use_async = copy_bits >= 32 and two_tile_bytes <= _OPTIN_SMEM_BYTES // 2
    reduce_bytes = rows * warps_per_row * cluster_n * 4
    smem_bytes = (two_tile_bytes if use_async else 0) + reduce_bytes
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
        "two_tile_bytes": two_tile_bytes,
        "use_async": use_async,
        "smem_bytes": smem_bytes,
    }


def _estimate_smem(H: int, cluster_n: int) -> int:
    config = _derived_config(H, cluster_n)
    return (
        2 * int(config["tile_bytes"])
        + int(config["rows"]) * int(config["warps_per_row"]) * cluster_n * 4
        + (8 if cluster_n > 1 else 0)
    )


def _source_config(H: int) -> dict[str, int | bool]:
    best_fit = 1
    for candidate in (1, 2, 4, 8, 16):
        if H % candidate != 0:
            continue
        required = _estimate_smem(H, candidate)
        if required <= _OPTIN_SMEM_BYTES // 2:
            return _derived_config(H, candidate)
        if required <= _OPTIN_SMEM_BYTES and best_fit == 1:
            best_fit = candidate
    return _derived_config(H, best_fit)


def _uses_rolled_fragment_loops(H: int) -> bool:
    source = _source_config(H)
    return int(source["vec"]) * int(source["vec_blocks"]) > 512


def _short_variant(variant: str) -> str:
    return "fused" if variant == "fused_add_rmsnorm" else "gemma"


def _short_dtype(dtype: str) -> str:
    return "fp16" if dtype == "float16" else "bf16"


def _layout_code(layout: str) -> str:
    return "c" if layout == "compact" else "s"


def _cfg(
    variant: str,
    dtype: str,
    M: int,
    H: int,
    input_layout: str,
    residual_layout: str,
    enable_pdl: bool,
    *,
    eps: float = _DEFAULT_EPS,
    x_row_stride: int | None = None,
    residual_row_stride: int | None = None,
    suffix: str | None = None,
) -> dict[str, Any]:
    label = (
        f"{_short_variant(variant)}_{_short_dtype(dtype)}_m{M}_h{H}_"
        f"x{_layout_code(input_layout)}_r{_layout_code(residual_layout)}_"
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
        "residual_layout": residual_layout,
        "enable_pdl": enable_pdl,
        "eps": eps,
    }
    if x_row_stride is not None:
        config["x_row_stride"] = x_row_stride
    if residual_row_stride is not None:
        config["residual_row_stride"] = residual_row_stride
    return config


_UPSTREAM_CONFIGS = [
    _cfg(
        variant,
        "float16",
        M,
        H,
        layout,
        layout,
        enable_pdl,
        x_row_stride=2 * H if layout == "strided" else None,
        residual_row_stride=2 * H if layout == "strided" else None,
    )
    for variant in _VARIANTS
    for M in (1, 19, 99, 989)
    for H in (111, 500, 1024, 3072, 3584, 4096, 8192, 16384)
    for layout in _LAYOUTS
    for enable_pdl in (False, True)
]

BENCH_CONFIGS = [
    _cfg("fused_add_rmsnorm", "bfloat16", 32, 4096, "compact", "compact", False),
    _cfg("fused_add_rmsnorm", "bfloat16", 64, 8192, "compact", "compact", False),
    _cfg("fused_add_rmsnorm", "bfloat16", 32, 4096, "compact", "compact", True),
    _cfg("gemma_fused_add_rmsnorm", "bfloat16", 32, 4096, "compact", "compact", False),
    _cfg("gemma_fused_add_rmsnorm", "bfloat16", 64, 8192, "compact", "compact", False),
    _cfg("gemma_fused_add_rmsnorm", "bfloat16", 32, 4096, "compact", "compact", True),
]

_I64_CONFIGS = [
    _cfg(
        "fused_add_rmsnorm",
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
        "fused_add_rmsnorm",
        "float16",
        175000,
        12288,
        "compact",
        "compact",
        False,
        suffix="compact_overflow_strided_abi",
    ),
    _cfg(
        "gemma_fused_add_rmsnorm",
        "float16",
        175000,
        12288,
        "compact",
        "compact",
        False,
        suffix="compact_overflow_strided_abi",
    ),
]

_TRACE_CONFIGS = [
    _cfg(variant, "bfloat16", M, H, "compact", "compact", False)
    for variant in _VARIANTS
    for M, H in ((8, 256), (3, 320))
] + [
    _cfg("fused_add_rmsnorm", "bfloat16", 32, 5120, "compact", "compact", False),
    _cfg("gemma_fused_add_rmsnorm", "bfloat16", 32, 4608, "compact", "compact", False),
]

_STRUCTURE_CONFIGS = [
    _cfg("fused_add_rmsnorm", "float16", 3, 64, "compact", "compact", False, suffix="tpr8"),
    _cfg("gemma_fused_add_rmsnorm", "bfloat16", 3, 66, "compact", "compact", True, suffix="vec2"),
    _cfg("fused_add_rmsnorm", "float16", 3, 16385, "compact", "compact", False, suffix="sync_vec1"),
    _cfg("fused_add_rmsnorm", "float16", 3, 32768, "compact", "compact", False, suffix="cluster2"),
    _cfg(
        "gemma_fused_add_rmsnorm",
        "bfloat16",
        3,
        65536,
        "compact",
        "compact",
        True,
        suffix="cluster4",
    ),
    _cfg("fused_add_rmsnorm", "float16", 3, 131072, "compact", "compact", False, suffix="cluster8"),
    _cfg(
        "gemma_fused_add_rmsnorm",
        "bfloat16",
        3,
        262144,
        "compact",
        "compact",
        True,
        suffix="cluster16",
    ),
    _cfg(
        "fused_add_rmsnorm",
        "float16",
        3,
        524288,
        "compact",
        "compact",
        False,
        suffix="cluster16_sync",
    ),
    _cfg(
        "gemma_fused_add_rmsnorm",
        "bfloat16",
        3,
        1048576,
        "compact",
        "compact",
        True,
        suffix="cluster1_sync_fallback",
    ),
]

_ABI_CONFIGS = [
    _cfg(
        "fused_add_rmsnorm",
        "float16",
        19,
        500,
        "strided",
        "strided",
        True,
        eps=1e-4,
        x_row_stride=1000,
        residual_row_stride=1500,
        suffix="eps1e4_full_abi",
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

assert len(CONFIGS) == 281
assert len({config["label"] for config in CONFIGS}) == len(CONFIGS)
assert len(BENCH_CONFIGS) == 6


def _validate(
    variant: str, dtype: str, M: int, H: int, input_layout: str, residual_layout: str, eps: float
) -> None:
    if variant not in _VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported dtype: {dtype}")
    if input_layout not in _LAYOUTS or residual_layout not in _LAYOUTS:
        raise ValueError(f"unsupported layouts: input={input_layout}, residual={residual_layout}")
    if M <= 0 or H <= 0:
        raise ValueError(f"M and H must be positive, got M={M}, H={H}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")


def _uses_compact_specialization(M: int, H: int, input_layout: str, residual_layout: str) -> bool:
    return input_layout == "compact" and residual_layout == "compact" and M * H <= _INT32_MAX


def get_kernel(
    variant: str,
    dtype: str,
    M: int,
    H: int,
    input_layout: str,
    residual_layout: str,
    enable_pdl: bool,
    eps: float = _DEFAULT_EPS,
    **kwargs: Any,
):
    """Return the compact or explicit-i64-strided runtime-M specialization."""
    _validate(variant, dtype, M, H, input_layout, residual_layout, eps)
    compact = _uses_compact_specialization(M, H, input_layout, residual_layout)
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
    two_tile_bytes = int(source["two_tile_bytes"])
    use_async = bool(source["use_async"])
    smem_bytes = int(source["smem_bytes"])
    total_values = vec * vec_blocks
    packed_pairs = _ceil_div(total_values, 2)
    pair_values = packed_pairs * 2
    roll_large_fragments = _uses_rolled_fragment_loops(H)
    packed_narrow = not (vec == 1 or (vec == 2 and vec_blocks == 3))
    weight_bias = 0.0 if variant == "fused_add_rmsnorm" else 1.0
    copy_bytes = copy_bits // 8
    reduce_base = two_tile_bytes if use_async else 0
    reduce_count = rows * warps_per_row * cluster_n
    mbar_offset = reduce_base + reduce_count * 4
    expected_bytes = reduce_count * 4
    total_partials_per_row = warps_per_row * cluster_n
    row_lane_xors = tuple(lane_xor for lane_xor in (1, 2, 4, 8, 16) if lane_xor < min(tpr, 32))
    full_lane_xors = (1, 2, 4, 8, 16)

    def fragment_range(extent: int):
        if roll_large_fragments:
            return K.serial(extent, unroll=False)
        return K.unroll(extent)

    x_row_stride_hint = kwargs.get("x_row_stride", H)
    residual_row_stride_hint = kwargs.get("residual_row_stride", H)
    if x_row_stride_hint is not None and int(x_row_stride_hint) % vec != 0:
        raise ValueError(f"x_row_stride={x_row_stride_hint} must be divisible by vec={vec}")
    if residual_row_stride_hint is not None and int(residual_row_stride_hint) % vec != 0:
        raise ValueError(
            f"residual_row_stride={residual_row_stride_hint} must be divisible by vec={vec}"
        )

    max_registers = 125 if compact and enable_pdl and H == 4096 else None

    def entry_registers():
        if max_registers is None:
            return contextlib.nullcontext()
        return K.attr({"tirx.max_registers": max_registers})

    def kernel_body(
        input_buffer, residual, weight, runtime_M, runtime_eps, x_row_stride, residual_row_stride
    ):
        # TIRX_TRANSCRIBE_START flashinfer_fused_add_rmsnorm
        if cluster_n > 1:
            block_x_raw, block_y_raw = K.cta_id(
                [K.cast(K.ceildiv(runtime_M, K.int64(rows)), "int32"), cluster_n]
            )
            _, cta_rank_raw = K.cta_id_in_cluster([1, cluster_n], preferred=[1, cluster_n])
            block_y = K.local_scalar(K.i32, init=K.cast(block_y_raw, "int32"), name="block_y")
            cta_rank = K.local_scalar(K.i32, init=K.cast(cta_rank_raw, "int32"), name="cta_rank")
        else:
            block_x_raw = K.cta_id([K.cast(K.ceildiv(runtime_M, K.int64(rows)), "int32")])
            block_y = K.local_scalar(K.i32, init=K.int32(0), name="block_y")
            cta_rank = K.local_scalar(K.i32, init=K.int32(0), name="cta_rank")
        tid = K.thread_id()

        if enable_pdl:
            K.ptx.griddepcontrol.wait()

        block_x = K.local_scalar(K.i32, init=K.cast(block_x_raw, "int32"), name="block_x")
        row_in_cta = K.local_scalar(K.i32, init=tid // tpr, name="row_in_cta")
        thread_in_row = K.local_scalar(K.i32, init=tid % tpr, name="thread_in_row")
        row_i32 = K.local_scalar(K.i32, init=block_x * rows + row_in_cta, name="row_i32")
        row_i64 = K.local_scalar(K.i64, init=K.cast(row_i32, "int64"), name="row_i64")
        row_valid = K.local_scalar("bool", init=row_i64 < runtime_M, name="row_valid")
        warp = K.local_scalar(K.i32, init=tid // 32, name="warp")
        lane = K.local_scalar(K.i32, init=tid % 32, name="lane")
        row_warp = K.local_scalar(K.i32, init=warp // warps_per_row, name="row_warp")
        warp_in_row = K.local_scalar(K.i32, init=warp % warps_per_row, name="warp_in_row")

        shared_raw = K.smem_pool().alloc((smem_bytes,), "uint8")

        if cluster_n > 1:
            with K.If(tid == 0), K.Then():
                K.ptx.mbarrier.init.shared.b64(shared_raw.ptr_to([mbar_offset]), K.uint32(1))
            K.ptx.fence.mbarrier_init.release.cluster()
            K.ptx.barrier.cluster.arrive.relaxed()
            K.ptx.barrier.cluster.wait()

        x_bits = K.alloc_local((pair_values,), "uint16")
        r_bits = K.alloc_local((pair_values,), "uint16")
        w_bits = K.alloc_local((pair_values,), "uint16")
        x_f32 = K.alloc_local((pair_values,), "float32")
        r_f32 = K.alloc_local((pair_values,), "float32")
        w_f32 = K.alloc_local((pair_values,), "float32")
        h_f32 = K.alloc_local((pair_values,), "float32")
        h_sq = K.alloc_local((pair_values,), "float32")
        undefined_f32 = K.alloc_local((1,), "float32")

        if not use_async:
            with fragment_range(total_values) as value:
                K.assign(x_bits[value], K.uint16(0))
                K.assign(r_bits[value], K.uint16(0))

        with fragment_range(vec_blocks) as vb:
            local_col = K.local_scalar(
                K.i32, init=(thread_in_row + vb * tpr) * vec, name="local_col"
            )
            absolute_col = K.local_scalar(
                K.i32, init=block_y * cols + local_col, name="absolute_col"
            )
            col_valid = K.local_scalar("bool", init=absolute_col < H, name="col_valid")
            if compact:
                x_offset = K.local_scalar(K.i32, init=row_i32 * H + absolute_col, name="x_offset")
            else:
                x_offset = K.local_scalar(
                    K.i64,
                    init=row_i64 * x_row_stride + K.cast(absolute_col, "int64"),
                    name="x_offset",
                )
            if use_async:
                with K.If(row_valid), K.Then():
                    source_bytes = K.local_scalar(
                        K.u32,
                        init=K.cast(K.if_then_else(col_valid, copy_bytes, 0), "uint32"),
                        name="source_bytes",
                    )
                    K.ptx.cp.async_.ca.shared.global_(
                        shared_raw.ptr_to([(row_in_cta * cols + local_col) * _ELEM_BYTES]),
                        input_buffer.ptr_to([x_offset]),
                        copy_bytes,
                        source_bytes,
                    )
            else:
                with K.If(K.And(row_valid, col_valid)), K.Then():
                    _load_global_bits(input_buffer, x_offset, x_bits, vb * vec, VEC=vec)

        with fragment_range(vec_blocks) as vb:
            local_col = K.local_scalar(
                K.i32, init=(thread_in_row + vb * tpr) * vec, name="local_col"
            )
            absolute_col = K.local_scalar(
                K.i32, init=block_y * cols + local_col, name="absolute_col"
            )
            col_valid = K.local_scalar("bool", init=absolute_col < H, name="col_valid")
            if compact:
                residual_offset = K.local_scalar(
                    K.i32, init=row_i32 * H + absolute_col, name="residual_offset"
                )
            else:
                residual_offset = K.local_scalar(
                    K.i64,
                    init=row_i64 * residual_row_stride + K.cast(absolute_col, "int64"),
                    name="residual_offset",
                )
            if use_async:
                with K.If(row_valid), K.Then():
                    source_bytes = K.local_scalar(
                        K.u32,
                        init=K.cast(K.if_then_else(col_valid, copy_bytes, 0), "uint32"),
                        name="source_bytes",
                    )
                    K.ptx.cp.async_.ca.shared.global_(
                        shared_raw.ptr_to(
                            [tile_bytes + (row_in_cta * cols + local_col) * _ELEM_BYTES]
                        ),
                        residual.ptr_to([residual_offset]),
                        copy_bytes,
                        source_bytes,
                    )
            else:
                with K.If(K.And(row_valid, col_valid)), K.Then():
                    _load_global_bits(residual, residual_offset, r_bits, vb * vec, VEC=vec)

        if use_async:
            K.ptx.cp.async_.commit_group()

        with fragment_range(vec_blocks) as vb:
            local_col = K.local_scalar(
                K.i32, init=(thread_in_row + vb * tpr) * vec, name="local_col"
            )
            absolute_col = K.local_scalar(
                K.i32, init=block_y * cols + local_col, name="absolute_col"
            )
            with K.If(absolute_col < H), K.Then():
                _load_global_bits(weight, absolute_col, w_bits, vb * vec, VEC=vec)

        if use_async:
            K.ptx.cp.async_.wait_group(0)
            with fragment_range(vec_blocks) as vb:
                local_col = K.local_scalar(
                    K.i32, init=(thread_in_row + vb * tpr) * vec, name="local_col"
                )
                _load_shared_bits(
                    shared_raw,
                    (row_in_cta * cols + local_col) * _ELEM_BYTES,
                    x_bits,
                    vb * vec,
                    VEC=vec,
                )
            with fragment_range(vec_blocks) as vb:
                local_col = K.local_scalar(
                    K.i32, init=(thread_in_row + vb * tpr) * vec, name="local_col"
                )
                _load_shared_bits(
                    shared_raw,
                    tile_bytes + (row_in_cta * cols + local_col) * _ELEM_BYTES,
                    r_bits,
                    vb * vec,
                    VEC=vec,
                )

        with fragment_range(total_values) as value:
            K.assign(x_f32[value], _cvt_to_f32(x_bits[value], dtype))
        with fragment_range(total_values) as value:
            K.assign(r_f32[value], _cvt_to_f32(r_bits[value], dtype))

        with fragment_range(packed_pairs) as pair:
            x_high = K.local_scalar(K.f32, init=x_f32[pair * 2 + 1], name="x_high")
            r_high = K.local_scalar(K.f32, init=r_f32[pair * 2 + 1], name="r_high")
            with K.If(pair * 2 + 1 >= total_values), K.Then():
                K.assign(x_high, undefined_f32[0])
                K.assign(r_high, undefined_f32[0])
            packed = K.alloc_local((1,), "uint64")
            K.ptx.add.f32x2(
                packed[0],
                K.cuda.make_float2(x_f32[pair * 2], x_high),
                K.cuda.make_float2(r_f32[pair * 2], r_high),
            )
            K.ptx.mov.b64(h_f32[pair * 2], h_f32[pair * 2 + 1], packed[0])

        residual_bits = K.alloc_local((pair_values,), "uint16")
        residual_words = K.alloc_local((packed_pairs,), "uint32")
        if packed_narrow:
            with fragment_range(packed_pairs) as pair:
                K.assign(
                    residual_words[pair],
                    _cvt_pair_from_f32(h_f32[pair * 2 + 1], h_f32[pair * 2], dtype),
                )
        else:
            with fragment_range(total_values) as value:
                K.assign(residual_bits[value], _cvt_from_f32(h_f32[value], dtype))

        with fragment_range(vec_blocks) as vb:
            local_col = K.local_scalar(
                K.i32, init=(thread_in_row + vb * tpr) * vec, name="local_col"
            )
            absolute_col = K.local_scalar(
                K.i32, init=block_y * cols + local_col, name="absolute_col"
            )
            col_valid = K.local_scalar("bool", init=absolute_col < H, name="col_valid")
            if compact:
                residual_offset = K.local_scalar(
                    K.i32, init=row_i32 * H + absolute_col, name="residual_offset"
                )
            else:
                residual_offset = K.local_scalar(
                    K.i64,
                    init=row_i64 * residual_row_stride + K.cast(absolute_col, "int64"),
                    name="residual_offset",
                )
            _store_global_fragment(
                residual,
                residual_offset,
                residual_bits,
                residual_words,
                K.And(row_valid, col_valid),
                vb * vec,
                vb * vec // 2,
                VEC=vec,
                PACKED_NARROW=packed_narrow,
            )

        with fragment_range(packed_pairs) as pair:
            packed = K.alloc_local((1,), "uint64")
            K.ptx.mul.f32x2(
                packed[0],
                K.cuda.make_float2(h_f32[pair * 2], h_f32[pair * 2 + 1]),
                K.cuda.make_float2(h_f32[pair * 2], h_f32[pair * 2 + 1]),
            )
            K.ptx.mov.b64(h_sq[pair * 2], h_sq[pair * 2 + 1], packed[0])

        local_sum = K.local_scalar(K.f32, init=K.float32(0.0), name="local_sum")
        with fragment_range(total_values) as value:
            K.assign(local_sum, _add_f32(local_sum, h_sq[value]))
        K.assign(local_sum, _butterfly_sum_f32(local_sum, row_lane_xors))
        warp_sum = K.local_scalar(K.f32, init=local_sum, name="warp_sum")

        if warps_per_row > 1 and cluster_n == 1:
            with K.If(lane == 0), K.Then():
                reduce_index = K.local_scalar(
                    K.i32, init=row_warp + warp_in_row * rows, name="reduce_index"
                )
                K.ptx.st.shared.b32(
                    shared_raw.ptr_to([reduce_base + reduce_index * 4]),
                    K.reinterpret("uint32", warp_sum),
                )
            K.ptx.bar.sync(K.uint32(0))
            final_sum = K.local_scalar(K.f32, init=K.float32(0.0), name="final_sum")
            with K.If(lane < warps_per_row), K.Then():
                reduce_word = K.alloc_local((1,), "uint32")
                reduce_index = K.local_scalar(
                    K.i32, init=row_warp + lane * rows, name="reduce_index"
                )
                K.ptx.ld.shared.b32(
                    reduce_word[0], shared_raw.ptr_to([reduce_base + reduce_index * 4])
                )
                K.assign(final_sum, K.reinterpret("float32", reduce_word[0]))
            K.assign(final_sum, _butterfly_sum_f32(final_sum, full_lane_xors))
            sum_sq = K.local_scalar(K.f32, init=final_sum, name="sum_sq")
        elif cluster_n > 1:
            with K.If(warp == 0), K.Then():
                with K.If(K.cuda.elect_sync()), K.Then():
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        shared_raw.ptr_to([mbar_offset]), K.uint32(expected_bytes)
                    )
            with K.If(lane < cluster_n), K.Then():
                reduce_index = K.local_scalar(
                    K.i32,
                    init=row_warp + warp_in_row * rows + cta_rank * rows * warps_per_row,
                    name="reduce_index",
                )
                peer_reduce = K.local_scalar(
                    K.u32,
                    init=_mapa_u32(shared_raw.ptr_to([reduce_base + reduce_index * 4]), lane),
                    name="peer_reduce",
                )
                peer_mbar = K.local_scalar(
                    K.u32, init=_mapa_u32(shared_raw.ptr_to([mbar_offset]), lane), name="peer_mbar"
                )
                K.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.f32(
                    peer_reduce, warp_sum, peer_mbar
                )
            _cluster_mbarrier_wait(shared_raw.ptr_to([mbar_offset]))
            final_sum = K.local_scalar(K.f32, init=K.float32(0.0), name="final_sum")
            for iteration in range(_ceil_div(total_partials_per_row, 32)):
                partial = K.local_scalar(K.i32, init=lane + iteration * 32, name="partial")
                with K.If(partial < total_partials_per_row), K.Then():
                    partial_warp = K.local_scalar(
                        K.i32, init=partial % warps_per_row, name="partial_warp"
                    )
                    partial_cta = K.local_scalar(
                        K.i32, init=partial // warps_per_row, name="partial_cta"
                    )
                    reduce_index = K.local_scalar(
                        K.i32,
                        init=row_warp + partial_warp * rows + partial_cta * rows * warps_per_row,
                        name="reduce_index",
                    )
                    reduce_word = K.alloc_local((1,), "uint32")
                    K.ptx.ld.shared.b32(
                        reduce_word[0], shared_raw.ptr_to([reduce_base + reduce_index * 4])
                    )
                    K.assign(
                        final_sum, _add_f32(final_sum, K.reinterpret("float32", reduce_word[0]))
                    )
            K.assign(final_sum, _butterfly_sum_f32(final_sum, full_lane_xors))
            sum_sq = K.local_scalar(K.f32, init=final_sum, name="sum_sq")
        else:
            sum_sq = K.local_scalar(K.f32, init=warp_sum, name="sum_sq")

        if H & (H - 1) == 0:
            shifted = K.local_scalar(
                K.f32, init=_fma_rn_f32(sum_sq, K.float32(1.0 / H), runtime_eps), name="shifted"
            )
        else:
            mean_sq = K.local_scalar(K.f32, init=_div_rn_f32(sum_sq, K.float32(H)), name="mean_sq")
            shifted = K.local_scalar(K.f32, init=_add_f32(mean_sq, runtime_eps), name="shifted")
        rstd = K.local_scalar(K.f32, init=_rsqrt_approx_ftz(shifted), name="rstd")

        if cluster_n > 1:
            K.ptx.barrier.cluster.arrive.relaxed()
            K.ptx.barrier.cluster.wait()
        else:
            K.ptx.bar.sync(K.uint32(0))

        with fragment_range(total_values) as value:
            K.assign(w_f32[value], _cvt_to_f32(w_bits[value], dtype))

        normalized = K.alloc_local((pair_values,), "float32")
        biased_weight = K.alloc_local((pair_values,), "float32")
        with fragment_range(packed_pairs) as pair:
            high_scale = K.local_scalar(K.f32, init=rstd, name="high_scale")
            with K.If(pair * 2 + 1 >= total_values), K.Then():
                K.assign(high_scale, undefined_f32[0])
            packed = K.alloc_local((1,), "uint64")
            K.ptx.mul.f32x2(
                packed[0],
                K.cuda.make_float2(h_f32[pair * 2], h_f32[pair * 2 + 1]),
                K.cuda.make_float2(rstd, high_scale),
            )
            K.ptx.mov.b64(normalized[pair * 2], normalized[pair * 2 + 1], packed[0])

        with fragment_range(packed_pairs) as pair:
            high_bias = K.local_scalar(K.f32, init=K.float32(weight_bias), name="high_bias")
            with K.If(pair * 2 + 1 >= total_values), K.Then():
                K.assign(high_bias, undefined_f32[0])
            packed = K.alloc_local((1,), "uint64")
            K.ptx.add.f32x2(
                packed[0],
                K.cuda.make_float2(w_f32[pair * 2], w_f32[pair * 2 + 1]),
                K.cuda.make_float2(K.float32(weight_bias), high_bias),
            )
            K.ptx.mov.b64(biased_weight[pair * 2], biased_weight[pair * 2 + 1], packed[0])

        with fragment_range(packed_pairs) as pair:
            packed = K.alloc_local((1,), "uint64")
            K.ptx.mul.f32x2(
                packed[0],
                K.cuda.make_float2(normalized[pair * 2], normalized[pair * 2 + 1]),
                K.cuda.make_float2(biased_weight[pair * 2], biased_weight[pair * 2 + 1]),
            )
            K.ptx.mov.b64(normalized[pair * 2], normalized[pair * 2 + 1], packed[0])

        output_bits = K.alloc_local((pair_values,), "uint16")
        output_words = K.alloc_local((packed_pairs,), "uint32")
        if packed_narrow:
            with fragment_range(packed_pairs) as pair:
                K.assign(
                    output_words[pair],
                    _cvt_pair_from_f32(normalized[pair * 2 + 1], normalized[pair * 2], dtype),
                )
        else:
            with fragment_range(total_values) as value:
                K.assign(output_bits[value], _cvt_from_f32(normalized[value], dtype))

        with fragment_range(vec_blocks) as vb:
            local_col = K.local_scalar(
                K.i32, init=(thread_in_row + vb * tpr) * vec, name="local_col"
            )
            absolute_col = K.local_scalar(
                K.i32, init=block_y * cols + local_col, name="absolute_col"
            )
            col_valid = K.local_scalar("bool", init=absolute_col < H, name="col_valid")
            if compact:
                x_offset = K.local_scalar(K.i32, init=row_i32 * H + absolute_col, name="x_offset")
            else:
                x_offset = K.local_scalar(
                    K.i64,
                    init=row_i64 * x_row_stride + K.cast(absolute_col, "int64"),
                    name="x_offset",
                )
            _store_global_fragment(
                input_buffer,
                x_offset,
                output_bits,
                output_words,
                K.And(row_valid, col_valid),
                vb * vec,
                vb * vec // 2,
                VEC=vec,
                PACKED_NARROW=packed_narrow,
            )

        if enable_pdl:
            K.ptx.griddepcontrol.launch_dependents()

    if compact:

        @K.kernel(warps=threads // 32, arch="sm_100a", grid=False)
        def flashinfer_fused_add_rmsnorm_compact(
            input_buffer: K.gptr[dtype],
            residual: K.gptr[dtype],
            weight: K.gptr[dtype, (H,)],
            runtime_M: K.i64,
            runtime_eps: K.f32,
        ):
            with entry_registers():
                kernel_body(
                    input_buffer, residual, weight, runtime_M, runtime_eps, K.int64(H), K.int64(H)
                )

        kernel = flashinfer_fused_add_rmsnorm_compact.func
    else:

        @K.kernel(warps=threads // 32, arch="sm_100a", grid=False)
        def flashinfer_fused_add_rmsnorm_strided(
            input_buffer: K.gptr[dtype],
            residual: K.gptr[dtype],
            weight: K.gptr[dtype, (H,)],
            runtime_M: K.i64,
            runtime_eps: K.f32,
            x_row_stride: K.i64,
            residual_row_stride: K.i64,
        ):
            with entry_registers():
                kernel_body(
                    input_buffer,
                    residual,
                    weight,
                    runtime_M,
                    runtime_eps,
                    x_row_stride,
                    residual_row_stride,
                )

        kernel = flashinfer_fused_add_rmsnorm_strided.func

    launch_params = ["blockIdx.x"]
    if cluster_n > 1:
        launch_params.extend(["blockIdx.y", "clusterCtaIdx.x", "clusterCtaIdx.y"])
    launch_params.append("threadIdx.x")
    if enable_pdl:
        launch_params.append("tirx.use_programtic_dependent_launch")
    launch_params.append("tirx.use_dyn_shared_memory")
    return kernel.with_attr("tirx.kernel_launch_params", launch_params)


def prepare_data(**config: Any):
    """Create deterministic input, residual, and weight tensors."""
    data = _prepare_tensors(dict(config))
    return data["input"], data["residual"], data["weight"]


_GUARD_ELEMENTS = 64
_GUARD_VALUE = 123.0


def _torch_dtype(dtype: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def _row_strides(config: dict[str, Any]) -> tuple[int, int]:
    H = int(config["H"])
    x_stride = int(config.get("x_row_stride", 2 * H if config["input_layout"] == "strided" else H))
    residual_stride = int(
        config.get("residual_row_stride", 2 * H if config["residual_layout"] == "strided" else H)
    )
    return x_stride, residual_stride


def _storage_size(M: int, H: int, row_stride: int) -> int:
    return (M - 1) * row_stride + H


def _allocate_strided(M: int, H: int, row_stride: int, dtype: str) -> dict[str, Any]:
    import torch

    size = _storage_size(M, H, row_stride)
    backing = torch.empty(size + _GUARD_ELEMENTS, dtype=_torch_dtype(dtype), device="cuda")
    backing.fill_(_GUARD_VALUE)
    arg = backing[:size]
    view = arg.as_strided((M, H), (row_stride, 1))
    return {"view": view, "arg": arg, "backing": backing, "size": size}


def _prepare_tensors(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    M = int(config["M"])
    H = int(config["H"])
    dtype = str(config["dtype"])
    variant = str(config["variant"])
    input_layout = str(config["input_layout"])
    residual_layout = str(config["residual_layout"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    _validate(variant, dtype, M, H, input_layout, residual_layout, eps)
    x_stride, residual_stride = _row_strides(config)
    vec = int(_source_config(H)["vec"])
    if x_stride < H or residual_stride < H:
        raise ValueError(f"row strides must cover H={H}: x={x_stride}, residual={residual_stride}")
    if x_stride % vec != 0 or residual_stride % vec != 0:
        raise ValueError(
            f"row strides must be divisible by vec={vec}: x={x_stride}, residual={residual_stride}"
        )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(42)
    x = _allocate_strided(M, H, x_stride, dtype)
    residual = _allocate_strided(M, H, residual_stride, dtype)

    # Large reductions use exactly representable, nonuniform patterns. This
    # keeps the independent FP32 oracle stable around the source's approximate
    # reciprocal square root while still exposing row/column/cluster mistakes.
    if (dtype == "bfloat16" and H >= 4096) or H >= 65536:
        columns = torch.arange(H, device="cuda")
        x_mag = torch.where(columns % 257 < 128, 0.5, 1.0)
        r_mag = torch.where(columns % 193 < 96, 0.25, 0.75)
        x_pattern = torch.where(columns % 2 == 0, x_mag, -x_mag).to(_torch_dtype(dtype))
        r_pattern = torch.where(columns % 3 == 0, r_mag, -r_mag).to(_torch_dtype(dtype))
        x["view"].copy_(x_pattern.expand(M, H))
        residual["view"].copy_(r_pattern.expand(M, H))
        x["view"][1::2].neg_()
        residual["view"][2::3].neg_()
    else:
        x["view"].normal_(generator=generator)
        residual["view"].normal_(generator=generator)

    weight_backing = torch.empty(H + _GUARD_ELEMENTS, dtype=_torch_dtype(dtype), device="cuda")
    weight = weight_backing[:H]
    weight.normal_(generator=generator)
    weight_backing[H:].fill_(_GUARD_VALUE)
    return {
        "input": x["view"],
        "input_arg": x["arg"],
        "input_backing": x["backing"],
        "input_size": x["size"],
        "residual": residual["view"],
        "residual_arg": residual["arg"],
        "residual_backing": residual["backing"],
        "residual_size": residual["size"],
        "weight": weight,
        "weight_backing": weight_backing,
        "x_row_stride": x_stride,
        "residual_row_stride": residual_stride,
    }


def _assert_guard(backing, size: int, *, name: str) -> None:
    import torch

    expected = torch.full(
        (_GUARD_ELEMENTS,), _GUARD_VALUE, dtype=backing.dtype, device=backing.device
    )
    if not torch.equal(backing[size:], expected):
        raise AssertionError(f"{name} guard was modified")


def _assert_padding(data: dict[str, Any], M: int, H: int, *, prefix: str) -> None:
    import torch

    for name, stride_key in (("input", "x_row_stride"), ("residual", "residual_row_stride")):
        backing = data[f"{name}_backing"]
        size = int(data[f"{name}_size"])
        stride = int(data[stride_key])
        _assert_guard(backing, size, name=f"{prefix} {name}")
        if stride > H:
            for row in range(M - 1):
                padding = backing[row * stride + H : (row + 1) * stride]
                if not torch.equal(padding, torch.full_like(padding, _GUARD_VALUE)):
                    raise AssertionError(f"{prefix} {name} row {row} padding was modified")


def _overflow_rows(M: int, H: int) -> list[int] | None:
    if M * H <= _INT32_MAX:
        return None
    boundary = _ceil_div(2**31, H)
    return sorted(
        {row for row in (0, 1, boundary - 1, boundary, boundary + 1, M - 1) if 0 <= row < M}
    )


def _checked_view(tensor, rows: list[int] | None):
    return tensor if rows is None else tensor[rows]


def _input_snapshot(data: dict[str, Any], M: int, H: int) -> dict[str, Any]:
    rows = _overflow_rows(M, H)
    return {
        "rows": rows,
        "input": _checked_view(data["input"], rows).clone(),
        "residual": _checked_view(data["residual"], rows).clone(),
        "weight": data["weight"].clone(),
    }


def _math_oracle(snapshot: dict[str, Any], variant: str, eps: float):
    h = snapshot["input"].float() + snapshot["residual"].float()
    residual = h.to(snapshot["input"].dtype)
    variance = h.square().mean(dim=-1, keepdim=True)
    bias = 0.0 if variant == "fused_add_rmsnorm" else 1.0
    output = h * variance.add(eps).rsqrt() * (snapshot["weight"].float() + bias)
    return output.to(snapshot["input"].dtype), residual


def _assert_close(actual, expected, *, name: str) -> None:
    import torch

    torch.testing.assert_close(
        actual, expected, rtol=1e-3, atol=1e-3, msg=lambda message: f"{name}: {message}"
    )


def _flashinfer_api(variant: str, device):
    import flashinfer
    import flashinfer.norm as flashinfer_norm

    if flashinfer_norm._use_cuda_norm(device):
        raise AssertionError("FlashInfer fused add RMSNorm oracle dispatched to legacy CUDA")
    return getattr(flashinfer, variant), flashinfer_norm


def _launch_tirx(executable, data: dict[str, Any], config: dict[str, Any]) -> None:
    M = int(config["M"])
    H = int(config["H"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    compact = _uses_compact_specialization(
        M, H, str(config["input_layout"]), str(config["residual_layout"])
    )
    if compact:
        executable(data["input_arg"], data["residual_arg"], data["weight"], M, eps)
    else:
        executable(
            data["input_arg"],
            data["residual_arg"],
            data["weight"],
            M,
            eps,
            data["x_row_stride"],
            data["residual_row_stride"],
        )


def run_test(**config: Any) -> None:
    """Compile, launch, and validate both in-place outputs for one config."""
    import torch

    from tirx_kernels.runner import compile_kernel

    config = dict(config)
    M = int(config["M"])
    H = int(config["H"])
    variant = str(config["variant"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    enable_pdl = bool(config["enable_pdl"])
    data = _prepare_tensors(config)
    # CuTe materializes the million-column fallback as a multi-gigabyte host
    # compilation.  Its cluster-1 synchronous path is compared directly at
    # H=16385; the extreme rolled-loop specialization uses the FP32 oracle.
    use_cute_reference = not _uses_rolled_fragment_loops(H)
    reference = _prepare_tensors(config) if use_cute_reference else None
    snapshot = _input_snapshot(data, M, H)
    executable = compile_kernel(get_kernel(**config))
    _launch_tirx(executable, data, config)

    if reference is not None:
        api, flashinfer_norm = _flashinfer_api(variant, reference["input"].device)
        original_cute = flashinfer_norm.fused_add_rmsnorm_cute
        cute_calls = 0

        def tracked_cute(*args, **kwargs):
            nonlocal cute_calls
            cute_calls += 1
            return original_cute(*args, **kwargs)

        flashinfer_norm.fused_add_rmsnorm_cute = tracked_cute
        try:
            returned = api(
                reference["input"],
                reference["residual"],
                reference["weight"],
                eps,
                enable_pdl=enable_pdl,
            )
        finally:
            flashinfer_norm.fused_add_rmsnorm_cute = original_cute
        if returned is not None:
            raise AssertionError("FlashInfer in-place fused add RMSNorm returned a value")
        if cute_calls != 1:
            raise AssertionError(f"expected one CuTe-DSL oracle dispatch, observed {cute_calls}")

    torch.cuda.synchronize()
    rows = snapshot["rows"]
    actual_input = _checked_view(data["input"], rows)
    actual_residual = _checked_view(data["residual"], rows)
    oracle_input, oracle_residual = _math_oracle(snapshot, variant, eps)
    if not torch.isfinite(actual_input).all() or not torch.isfinite(actual_residual).all():
        raise AssertionError("TIRx fused add RMSNorm produced non-finite values")
    if reference is not None:
        reference_input = _checked_view(reference["input"], rows)
        reference_residual = _checked_view(reference["residual"], rows)
        _assert_close(actual_input, reference_input, name="FlashInfer input output")
        _assert_close(actual_residual, reference_residual, name="FlashInfer residual output")
    _assert_close(actual_input, oracle_input, name="FP32 input oracle")
    _assert_close(actual_residual, oracle_residual, name="FP32 residual oracle")
    if not torch.equal(data["weight"], snapshot["weight"]):
        raise AssertionError("TIRx modified weight")
    _assert_guard(data["weight_backing"], H, name="TIRx weight")
    _assert_padding(data, M, H, prefix="TIRx")
    if reference is not None:
        if not torch.equal(reference["weight"], snapshot["weight"]):
            raise AssertionError("FlashInfer modified weight")
        _assert_guard(reference["weight_backing"], H, name="FlashInfer weight")
        _assert_padding(reference, M, H, prefix="FlashInfer")


def prepare_bench(**config: Any):
    """Compile the selected specialization before GPU assignment."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(config), "executable": compile_kernel(get_kernel(**config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs: Any
):
    """Construct and validate independent mutable timed closures."""
    import torch

    config = dict(prepared["config"])
    config.update(kwargs)
    variant = str(config["variant"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    enable_pdl = bool(config["enable_pdl"])
    tirx_data = _prepare_tensors(config)
    reference_data = _prepare_tensors(config)
    executable = prepared["executable"]

    def tirx_launch():
        _launch_tirx(executable, tirx_data, config)

    tirx_launch()
    torch.cuda.synchronize()

    def build_flashinfer_reference():
        api, _ = _flashinfer_api(variant, reference_data["input"].device)

        def flashinfer_launch():
            return api(
                reference_data["input"],
                reference_data["residual"],
                reference_data["weight"],
                eps,
                enable_pdl=enable_pdl,
            )

        returned = flashinfer_launch()
        if returned is not None:
            raise AssertionError("FlashInfer benchmark in-place API returned a value")
        torch.cuda.synchronize()
        _assert_close(tirx_data["input"], reference_data["input"], name="benchmark input precheck")
        _assert_close(
            tirx_data["residual"], reference_data["residual"], name="benchmark residual precheck"
        )
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
    """Benchmark the TIRx specialization against FlashInfer CuTe-DSL."""
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
