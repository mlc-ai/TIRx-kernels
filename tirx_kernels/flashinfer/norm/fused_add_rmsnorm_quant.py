# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400e330fb2debe0bf8730d9424a1d37927f), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer CuTe-DSL fused add RMSNorm with FP8 quantization.

The source implementation is ``FusedAddRMSNormQuantKernel`` in
``flashinfer/norm/kernels/fused_add_rmsnorm.py`` together with RMSNorm and FP8
helpers from ``flashinfer/norm/kernels/rmsnorm.py`` and
``flashinfer/norm/utils.py``.  The public dispatch is
``flashinfer.norm.fused_add_rmsnorm_quant``.
"""

import functools
import math
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
from .fused_add_rmsnorm import _source_config as _fused_source_config
from .rmsnorm_quant import (
    _cvt_fp8_pair,
    _maximum_f32,
    _minimum_f32,
    _pack_b16_pair,
    _rcp_approx_ftz,
)

KERNEL_META = {
    "name": "flashinfer_fused_add_rmsnorm_quant",
    "category": "flashinfer",
    "compute_capability": 10,
}

_INPUT_DTYPES = ("float16", "bfloat16")
_OUTPUT_DTYPES = ("float8_e4m3fn", "float8_e5m2")
_LAYOUTS = ("compact", "strided")
_SCALES = (0.01, 1.0, 10.0)
_DEFAULT_EPS = 1e-6
_INT32_MAX = 2**31 - 1
_OPTIN_SMEM_BYTES = 232448
_INPUT_ELEM_BYTES = 2


def _derived_config(H: int, cluster_n: int) -> dict[str, int | bool]:
    H_per_cta = H // cluster_n
    tpr = _threads_per_row(H_per_cta)
    threads = 128 if H_per_cta <= 16384 else 256
    if H_per_cta > 8192 and threads < 256:
        threads = 256
    rows = threads // tpr
    warps_per_row = max(tpr // 32, 1)
    vec = min(H_per_cta & -H_per_cta, 8)
    copy_bits = 16 * vec
    vec_blocks = max(1, _ceil_div(H_per_cta // vec, tpr))
    cols = vec * vec_blocks * tpr
    tile_bytes = rows * cols * _INPUT_ELEM_BYTES
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


def _source_config(H: int) -> dict[str, int | bool]:
    cluster_n = int(_fused_source_config(H)["cluster_n"])
    return _derived_config(H, cluster_n)


def _uses_rolled_fragment_loops(H: int) -> bool:
    source = _source_config(H)
    return int(source["vec"]) * int(source["vec_blocks"]) > 512


def _short_dtype(dtype: str) -> str:
    return {"float16": "fp16", "bfloat16": "bf16", "float8_e4m3fn": "e4m3", "float8_e5m2": "e5m2"}[
        dtype
    ]


def _layout_code(layout: str) -> str:
    return "c" if layout == "compact" else "s"


def _scale_code(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def _cfg(
    input_dtype: str,
    output_dtype: str,
    M: int,
    H: int,
    input_layout: str,
    residual_layout: str,
    output_layout: str,
    enable_pdl: bool,
    scale: float,
    *,
    eps: float = _DEFAULT_EPS,
    x_row_stride: int | None = None,
    residual_row_stride: int | None = None,
    y_row_stride: int | None = None,
    suffix: str | None = None,
) -> dict[str, Any]:
    label = (
        f"{_short_dtype(input_dtype)}_{_short_dtype(output_dtype)}_m{M}_h{H}_"
        f"x{_layout_code(input_layout)}_r{_layout_code(residual_layout)}_"
        f"y{_layout_code(output_layout)}_pdl{int(enable_pdl)}_s{_scale_code(scale)}"
    )
    if suffix:
        label += f"_{suffix}"
    config: dict[str, Any] = {
        "label": label,
        "input_dtype": input_dtype,
        "output_dtype": output_dtype,
        "M": M,
        "H": H,
        "input_layout": input_layout,
        "residual_layout": residual_layout,
        "output_layout": output_layout,
        "enable_pdl": enable_pdl,
        "scale": scale,
        "eps": eps,
    }
    if x_row_stride is not None:
        config["x_row_stride"] = x_row_stride
    if residual_row_stride is not None:
        config["residual_row_stride"] = residual_row_stride
    if y_row_stride is not None:
        config["y_row_stride"] = y_row_stride
    return config


_UPSTREAM_CONFIGS = [
    _cfg(
        input_dtype,
        output_dtype,
        M,
        H,
        input_layout,
        "compact",
        "compact",
        enable_pdl,
        scale,
        x_row_stride=2 * H if input_layout == "strided" else None,
    )
    for M in (1, 19, 99, 989)
    for H in (111, 500, 1024, 3072, 3584, 4096, 8192, 16384)
    for input_dtype in _INPUT_DTYPES
    for output_dtype in _OUTPUT_DTYPES
    for scale in _SCALES
    for enable_pdl in (False, True)
    for input_layout in _LAYOUTS
]

_ACTIVE_BENCHMARK_CONFIGS = [
    _cfg("bfloat16", "float8_e4m3fn", 32, 4096, "compact", "compact", "compact", False, 1.0),
    _cfg("bfloat16", "float8_e4m3fn", 64, 8192, "compact", "compact", "compact", False, 1.0),
    _cfg("bfloat16", "float8_e4m3fn", 32, 4096, "compact", "compact", "compact", True, 1.0),
]

_I64_CONFIGS = [
    _cfg(
        "bfloat16",
        "float8_e4m3fn",
        2,
        128,
        "strided",
        "compact",
        "compact",
        False,
        1.0,
        x_row_stride=2**31,
        suffix="i64_xstride",
    ),
    _cfg(
        "float16",
        "float8_e4m3fn",
        175000,
        12288,
        "compact",
        "compact",
        "compact",
        False,
        1.0,
        suffix="compact_overflow_to_strided_i64",
    ),
]

_TRACE_CONFIGS = [
    _cfg("bfloat16", "float8_e4m3fn", M, H, "compact", "compact", "compact", False, 1.0)
    for M, H in ((32, 2048), (7, 1024), (32, 7168))
]

_STRUCTURE_CONFIGS = [
    _cfg(
        "float16",
        "float8_e4m3fn",
        3,
        64,
        "compact",
        "compact",
        "compact",
        False,
        1.0,
        suffix="vec8_tpr8_async",
    ),
    _cfg(
        "bfloat16",
        "float8_e5m2",
        989,
        66,
        "compact",
        "compact",
        "compact",
        True,
        10.0,
        suffix="vec2",
    ),
    _cfg(
        "float16",
        "float8_e4m3fn",
        3,
        16385,
        "compact",
        "compact",
        "compact",
        False,
        1.0,
        suffix="vec1_sync",
    ),
    _cfg(
        "float16",
        "float8_e4m3fn",
        3,
        32768,
        "compact",
        "compact",
        "compact",
        False,
        1.0,
        suffix="cluster2_sync",
    ),
    _cfg(
        "bfloat16",
        "float8_e5m2",
        3,
        65536,
        "compact",
        "compact",
        "compact",
        True,
        1.0,
        suffix="cluster4_sync",
    ),
    _cfg(
        "float16",
        "float8_e4m3fn",
        3,
        131072,
        "compact",
        "compact",
        "compact",
        False,
        1.0,
        suffix="cluster8_sync",
    ),
    _cfg(
        "bfloat16",
        "float8_e5m2",
        3,
        262144,
        "compact",
        "compact",
        "compact",
        True,
        1.0,
        suffix="cluster16_sync",
    ),
    _cfg(
        "float16",
        "float8_e4m3fn",
        3,
        524288,
        "compact",
        "compact",
        "compact",
        False,
        1.0,
        suffix="cluster16_sync_large",
    ),
    _cfg(
        "bfloat16",
        "float8_e5m2",
        3,
        1048576,
        "compact",
        "compact",
        "compact",
        True,
        1.0,
        suffix="cluster1_sync_fallback",
    ),
    _cfg(
        "float16",
        "float8_e4m3fn",
        3,
        131070,
        "compact",
        "compact",
        "compact",
        False,
        1.0,
        suffix="fragment512",
    ),
    _cfg(
        "bfloat16",
        "float8_e5m2",
        3,
        131073,
        "compact",
        "compact",
        "compact",
        True,
        1.0,
        suffix="fragment513_rolled",
    ),
]

_ABI_CONFIGS = [
    _cfg(
        "float16",
        "float8_e5m2",
        19,
        500,
        "strided",
        "strided",
        "strided",
        True,
        1.0,
        eps=1e-4,
        x_row_stride=1000,
        residual_row_stride=1500,
        y_row_stride=2000,
        suffix="eps1e4_full_abi",
    )
]

CONFIGS = [
    *_UPSTREAM_CONFIGS,
    *_ACTIVE_BENCHMARK_CONFIGS,
    *_I64_CONFIGS,
    *_TRACE_CONFIGS,
    *_STRUCTURE_CONFIGS,
    *_ABI_CONFIGS,
]
BENCH_CONFIGS = list(_ACTIVE_BENCHMARK_CONFIGS)

assert len(_UPSTREAM_CONFIGS) == 1536
assert len(CONFIGS) == 1556
assert len(BENCH_CONFIGS) == 3
assert len({config["label"] for config in CONFIGS}) == len(CONFIGS)


def _validate(
    input_dtype: str,
    output_dtype: str,
    M: int,
    H: int,
    input_layout: str,
    residual_layout: str,
    output_layout: str,
    scale: float,
    eps: float,
) -> None:
    if input_dtype not in _INPUT_DTYPES:
        raise ValueError(f"unsupported input dtype: {input_dtype}")
    if output_dtype not in _OUTPUT_DTYPES:
        raise ValueError(f"unsupported output dtype: {output_dtype}")
    for name, layout in (
        ("input", input_layout),
        ("residual", residual_layout),
        ("output", output_layout),
    ):
        if layout not in _LAYOUTS:
            raise ValueError(f"unsupported {name} layout: {layout}")
    if M <= 0 or H <= 0:
        raise ValueError(f"M and H must be positive, got M={M}, H={H}")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be positive and finite, got {scale}")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be positive and finite, got {eps}")


def _uses_compact_specialization(
    M: int, H: int, input_layout: str, residual_layout: str, output_layout: str
) -> bool:
    return input_layout == residual_layout == output_layout == "compact" and M * H <= _INT32_MAX


def get_kernel(
    input_dtype: str,
    output_dtype: str,
    M: int,
    H: int,
    input_layout: str,
    residual_layout: str,
    output_layout: str,
    enable_pdl: bool,
    scale: float,
    eps: float = _DEFAULT_EPS,
    **kwargs: Any,
):
    """Return the compact or explicit-int64-strided source specialization."""
    _validate(
        input_dtype, output_dtype, M, H, input_layout, residual_layout, output_layout, scale, eps
    )
    compact = _uses_compact_specialization(M, H, input_layout, residual_layout, output_layout)
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
    packed_narrow = not (vec == 1 or (vec == 2 and vec_blocks == 3))
    copy_bytes = copy_bits // 8
    reduce_base = two_tile_bytes if use_async else 0
    reduce_count = rows * warps_per_row * cluster_n
    mbar_offset = reduce_base + reduce_count * 4
    expected_bytes = reduce_count * 4
    total_partials_per_row = warps_per_row * cluster_n
    row_lane_xors = tuple(lane_xor for lane_xor in (1, 2, 4, 8, 16) if lane_xor < min(tpr, 32))
    full_lane_xors = (1, 2, 4, 8, 16)
    fp8_max = 448.0 if output_dtype == "float8_e4m3fn" else 57344.0
    roll_large_fragments = _uses_rolled_fragment_loops(H)

    def fragment_range(extent: int):
        if roll_large_fragments:
            return K.serial(extent, unroll=False)
        return K.unroll(extent)

    x_row_stride_hint = int(kwargs.get("x_row_stride", H))
    residual_row_stride_hint = int(kwargs.get("residual_row_stride", H))
    y_row_stride_hint = int(kwargs.get("y_row_stride", H))
    for name, stride_hint in (
        ("x", x_row_stride_hint),
        ("residual", residual_row_stride_hint),
        ("y", y_row_stride_hint),
    ):
        if stride_hint % vec != 0:
            raise ValueError(f"{name}_row_stride={stride_hint} must be divisible by vec={vec}")

    def kernel_body(
        out,
        input_buffer,
        residual,
        weight,
        runtime_M,
        scale_buffer,
        runtime_eps,
        y_row_stride,
        x_row_stride,
        residual_row_stride,
    ):
        # TIRX_TRANSCRIBE_START flashinfer_fused_add_rmsnorm_quant
        if cluster_n > 1:
            block_x_raw, block_y_raw = K.cta_id(
                [K.cast(K.ceildiv(runtime_M, K.int64(rows)), "int32"), cluster_n]
            )
            _, cta_rank_raw = K.cta_id_in_cluster([1, cluster_n], preferred=[1, cluster_n])
            block_y: K.int32 = K.cast(block_y_raw, "int32")
            cta_rank: K.int32 = K.cast(cta_rank_raw, "int32")
        else:
            block_x_raw = K.cta_id([K.cast(K.ceildiv(runtime_M, K.int64(rows)), "int32")])
            block_y = K.int32(0)
            cta_rank = K.int32(0)
        tid = K.thread_id()

        if enable_pdl:
            K.ptx.griddepcontrol.wait()

        scale_bits = K.alloc_local((1,), "uint32")
        K.ptx.ld.global_.b32(scale_bits[0], scale_buffer.ptr_to([0]))
        scale_value: K.float32 = K.reinterpret("float32", scale_bits[0])
        inv_scale: K.float32 = _rcp_approx_ftz(scale_value)
        block_x: K.int32 = K.cast(block_x_raw, "int32")
        row_in_cta: K.int32 = tid // tpr
        thread_in_row: K.int32 = tid % tpr
        compact_row_i32: K.int32 = block_x * rows + row_in_cta
        actual_row: K.int64 = K.cast(block_x, "int64") * K.int64(rows) + K.cast(row_in_cta, "int64")
        row_valid: K.bool = actual_row < runtime_M
        warp: K.int32 = tid // 32
        lane: K.int32 = tid % 32
        row_warp: K.int32 = warp // warps_per_row
        warp_in_row: K.int32 = warp % warps_per_row

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
            local_col: K.int32 = (thread_in_row + vb * tpr) * vec
            absolute_col: K.int32 = block_y * cols + local_col
            col_valid: K.bool = absolute_col < H
            if compact:
                x_offset = compact_row_i32 * H + absolute_col
            else:
                x_offset = actual_row * x_row_stride + K.cast(absolute_col, "int64")
            if use_async:
                with K.If(row_valid), K.Then():
                    source_bytes: K.uint32 = K.cast(
                        K.if_then_else(col_valid, copy_bytes, 0), "uint32"
                    )
                    K.ptx.cp.async_.ca.shared.global_(
                        shared_raw.ptr_to([(row_in_cta * cols + local_col) * _INPUT_ELEM_BYTES]),
                        input_buffer.ptr_to([x_offset]),
                        copy_bytes,
                        source_bytes,
                    )
            else:
                with K.If(K.And(row_valid, col_valid)), K.Then():
                    _load_global_bits(input_buffer, x_offset, x_bits, vb * vec, VEC=vec)

        with fragment_range(vec_blocks) as vb:
            local_col: K.int32 = (thread_in_row + vb * tpr) * vec
            absolute_col: K.int32 = block_y * cols + local_col
            col_valid: K.bool = absolute_col < H
            if compact:
                residual_offset = compact_row_i32 * H + absolute_col
            else:
                residual_offset = actual_row * residual_row_stride + K.cast(absolute_col, "int64")
            if use_async:
                with K.If(row_valid), K.Then():
                    source_bytes: K.uint32 = K.cast(
                        K.if_then_else(col_valid, copy_bytes, 0), "uint32"
                    )
                    K.ptx.cp.async_.ca.shared.global_(
                        shared_raw.ptr_to(
                            [tile_bytes + (row_in_cta * cols + local_col) * _INPUT_ELEM_BYTES]
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
            local_col: K.int32 = (thread_in_row + vb * tpr) * vec
            absolute_col: K.int32 = block_y * cols + local_col
            with K.If(absolute_col < H), K.Then():
                _load_global_bits(weight, absolute_col, w_bits, vb * vec, VEC=vec)

        if use_async:
            K.ptx.cp.async_.wait_group(0)
            with fragment_range(vec_blocks) as vb:
                local_col: K.int32 = (thread_in_row + vb * tpr) * vec
                _load_shared_bits(
                    shared_raw,
                    (row_in_cta * cols + local_col) * _INPUT_ELEM_BYTES,
                    x_bits,
                    vb * vec,
                    VEC=vec,
                )
            with fragment_range(vec_blocks) as vb:
                local_col: K.int32 = (thread_in_row + vb * tpr) * vec
                _load_shared_bits(
                    shared_raw,
                    tile_bytes + (row_in_cta * cols + local_col) * _INPUT_ELEM_BYTES,
                    r_bits,
                    vb * vec,
                    VEC=vec,
                )

        with fragment_range(total_values) as value:
            K.assign(x_f32[value], _cvt_to_f32(x_bits[value], input_dtype))
        with fragment_range(total_values) as value:
            K.assign(r_f32[value], _cvt_to_f32(r_bits[value], input_dtype))

        with fragment_range(packed_pairs) as pair:
            x_high = K.local_scalar(K.f32, init=x_f32[pair * 2 + 1])
            r_high = K.local_scalar(K.f32, init=r_f32[pair * 2 + 1])
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
                    _cvt_pair_from_f32(h_f32[pair * 2 + 1], h_f32[pair * 2], input_dtype),
                )
        else:
            with fragment_range(total_values) as value:
                K.assign(residual_bits[value], _cvt_from_f32(h_f32[value], input_dtype))

        with fragment_range(vec_blocks) as vb:
            local_col: K.int32 = (thread_in_row + vb * tpr) * vec
            absolute_col: K.int32 = block_y * cols + local_col
            col_valid: K.bool = absolute_col < H
            if compact:
                residual_offset = compact_row_i32 * H + absolute_col
            else:
                residual_offset = actual_row * residual_row_stride + K.cast(absolute_col, "int64")
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

        local_sum = K.local_scalar(K.f32, init=K.float32(0.0))
        with fragment_range(total_values) as value:
            K.assign(local_sum, _add_f32(local_sum, h_sq[value]))
        K.assign(local_sum, _butterfly_sum_f32(local_sum, row_lane_xors))
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

        if cluster_n > 1:
            K.ptx.barrier.cluster.arrive.relaxed()
            K.ptx.barrier.cluster.wait()
        else:
            K.ptx.bar.sync(K.uint32(0))

        with fragment_range(total_values) as value:
            K.assign(w_f32[value], _cvt_to_f32(w_bits[value], input_dtype))

        normalized = K.alloc_local((pair_values,), "float32")
        biased_weight = K.alloc_local((pair_values,), "float32")
        weighted = K.alloc_local((pair_values,), "float32")
        y_f32 = K.alloc_local((pair_values,), "float32")

        with fragment_range(packed_pairs) as pair:
            high_scale = K.local_scalar(K.f32, init=rstd)
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
            high_bias = K.local_scalar(K.f32, init=K.float32(0.0))
            with K.If(pair * 2 + 1 >= total_values), K.Then():
                K.assign(high_bias, undefined_f32[0])
            packed = K.alloc_local((1,), "uint64")
            K.ptx.add.f32x2(
                packed[0],
                K.cuda.make_float2(w_f32[pair * 2], w_f32[pair * 2 + 1]),
                K.cuda.make_float2(K.float32(0.0), high_bias),
            )
            K.ptx.mov.b64(biased_weight[pair * 2], biased_weight[pair * 2 + 1], packed[0])

        with fragment_range(packed_pairs) as pair:
            packed = K.alloc_local((1,), "uint64")
            K.ptx.mul.f32x2(
                packed[0],
                K.cuda.make_float2(normalized[pair * 2], normalized[pair * 2 + 1]),
                K.cuda.make_float2(biased_weight[pair * 2], biased_weight[pair * 2 + 1]),
            )
            K.ptx.mov.b64(weighted[pair * 2], weighted[pair * 2 + 1], packed[0])

        with fragment_range(packed_pairs) as pair:
            high_inv_scale = K.local_scalar(K.f32, init=inv_scale)
            with K.If(pair * 2 + 1 >= total_values), K.Then():
                K.assign(high_inv_scale, undefined_f32[0])
            packed = K.alloc_local((1,), "uint64")
            K.ptx.mul.f32x2(
                packed[0],
                K.cuda.make_float2(weighted[pair * 2], weighted[pair * 2 + 1]),
                K.cuda.make_float2(inv_scale, high_inv_scale),
            )
            K.ptx.mov.b64(y_f32[pair * 2], y_f32[pair * 2 + 1], packed[0])

        with fragment_range(vec_blocks) as vb:
            local_col: K.int32 = (thread_in_row + vb * tpr) * vec
            absolute_col: K.int32 = block_y * cols + local_col
            if compact:
                y_offset = actual_row * K.int64(H) + K.cast(absolute_col, "int64")
            else:
                y_offset = actual_row * y_row_stride + K.cast(absolute_col, "int64")

            def store_scalars():
                for element in range(vec):
                    scalar_col: K.int32 = absolute_col + element
                    with K.If(K.And(scalar_col < H, row_valid)), K.Then():
                        clamped_low: K.float32 = _maximum_f32(
                            y_f32[vb * vec + element], K.float32(-fp8_max)
                        )
                        clamped: K.float32 = _minimum_f32(clamped_low, K.float32(fp8_max))
                        pair: K.uint16 = _cvt_fp8_pair(clamped, K.float32(0.0), output_dtype)
                        if compact:
                            scalar_offset = actual_row * K.int64(H) + K.cast(scalar_col, "int64")
                        else:
                            scalar_offset = actual_row * y_row_stride + K.cast(scalar_col, "int64")
                        K.ptx.st.global_.b8(out.ptr_to([scalar_offset]), K.cast(pair, "uint8"))

            vector_guard = K.And(absolute_col + vec <= H, row_valid)
            with K.If(vector_guard):
                with K.Then():
                    if vec == 8:
                        p01: K.uint16 = _cvt_fp8_pair(
                            y_f32[vb * 8], y_f32[vb * 8 + 1], output_dtype
                        )
                        p23: K.uint16 = _cvt_fp8_pair(
                            y_f32[vb * 8 + 2], y_f32[vb * 8 + 3], output_dtype
                        )
                        p45: K.uint16 = _cvt_fp8_pair(
                            y_f32[vb * 8 + 4], y_f32[vb * 8 + 5], output_dtype
                        )
                        p67: K.uint16 = _cvt_fp8_pair(
                            y_f32[vb * 8 + 6], y_f32[vb * 8 + 7], output_dtype
                        )
                        lo_word: K.uint32 = _pack_b16_pair(p01, p23)
                        hi_word: K.uint32 = _pack_b16_pair(p45, p67)
                        K.ptx.st.global_.v2.b32(out.ptr_to([y_offset]), lo_word, hi_word)
                    elif vec == 4:
                        p01 = _cvt_fp8_pair(y_f32[vb * 4], y_f32[vb * 4 + 1], output_dtype)
                        p23 = _cvt_fp8_pair(y_f32[vb * 4 + 2], y_f32[vb * 4 + 3], output_dtype)
                        packed_word: K.uint32 = _pack_b16_pair(p01, p23)
                        K.ptx.st.global_.b32(out.ptr_to([y_offset]), packed_word)
                    elif vec == 2:
                        p01 = _cvt_fp8_pair(y_f32[vb * 2], y_f32[vb * 2 + 1], output_dtype)
                        K.ptx.st.global_.b16(out.ptr_to([y_offset]), p01)
                    else:
                        store_scalars()
                with K.Else():
                    store_scalars()

        if enable_pdl:
            K.ptx.griddepcontrol.launch_dependents()

    if compact:

        @K.kernel(warps=threads // 32, arch="sm_100a", grid=False)
        def flashinfer_fused_add_rmsnorm_quant_compact(
            output: K.gptr[output_dtype],
            input_buffer: K.gptr[input_dtype],
            residual: K.gptr[input_dtype],
            weight: K.gptr[input_dtype, (H,)],
            runtime_M: K.i64,
            scale_buffer: K.gptr[K.f32, (1,)],
            runtime_eps: K.f32,
        ):
            kernel_body(
                output,
                input_buffer,
                residual,
                weight,
                runtime_M,
                scale_buffer,
                runtime_eps,
                K.int64(H),
                K.int64(H),
                K.int64(H),
            )

        kernel = flashinfer_fused_add_rmsnorm_quant_compact.func
    else:

        @K.kernel(warps=threads // 32, arch="sm_100a", grid=False)
        def flashinfer_fused_add_rmsnorm_quant_strided(
            output: K.gptr[output_dtype],
            input_buffer: K.gptr[input_dtype],
            residual: K.gptr[input_dtype],
            weight: K.gptr[input_dtype, (H,)],
            runtime_M: K.i64,
            scale_buffer: K.gptr[K.f32, (1,)],
            runtime_eps: K.f32,
            y_row_stride: K.i64,
            x_row_stride: K.i64,
            residual_row_stride: K.i64,
        ):
            kernel_body(
                output,
                input_buffer,
                residual,
                weight,
                runtime_M,
                scale_buffer,
                runtime_eps,
                y_row_stride,
                x_row_stride,
                residual_row_stride,
            )

        kernel = flashinfer_fused_add_rmsnorm_quant_strided.func

    launch_params = ["blockIdx.x"]
    if cluster_n > 1:
        launch_params.extend(["blockIdx.y", "clusterCtaIdx.x", "clusterCtaIdx.y"])
    launch_params.append("threadIdx.x")
    if enable_pdl:
        launch_params.append("tirx.use_programtic_dependent_launch")
    launch_params.append("tirx.use_dyn_shared_memory")
    return kernel.with_attr("tirx.kernel_launch_params", launch_params)


def prepare_data(**config: Any):
    """Create deterministic source-shaped tensors for the runtime ABI."""
    data = _prepare_tensors(dict(config))
    output = _prepare_output(
        int(config["M"]),
        int(config["H"]),
        data["y_row_stride"],
        str(config["output_dtype"]),
        initialize_padding=False,
    )
    return output["view"], data["input"], data["residual"], data["weight"], data["scale"]


_GUARD_ELEMENTS = 64
_GUARD_VALUE = 1.0


def _torch_input_dtype(dtype: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def _torch_output_dtype(dtype: str):
    import torch

    return {"float8_e4m3fn": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2}[dtype]


def _row_strides(config: dict[str, Any]) -> tuple[int, int, int]:
    H = int(config["H"])
    x_stride = int(config.get("x_row_stride", 2 * H if config["input_layout"] == "strided" else H))
    residual_stride = int(
        config.get("residual_row_stride", 2 * H if config["residual_layout"] == "strided" else H)
    )
    y_stride = int(config.get("y_row_stride", 2 * H if config["output_layout"] == "strided" else H))
    return x_stride, residual_stride, y_stride


def _storage_size(M: int, H: int, row_stride: int) -> int:
    return (M - 1) * row_stride + H


def _allocate_strided(M: int, H: int, row_stride: int, dtype) -> dict[str, Any]:
    import torch

    size = _storage_size(M, H, row_stride)
    backing = torch.empty(size + _GUARD_ELEMENTS, dtype=dtype, device="cuda")
    backing.fill_(_GUARD_VALUE)
    arg = backing[:size]
    view = arg.as_strided((M, H), (row_stride, 1))
    return {
        "view": view,
        "arg": arg,
        "backing": backing,
        "size": size,
        "data_ptr": view.data_ptr(),
        "stride": view.stride(),
    }


def _prepare_tensors(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    M = int(config["M"])
    H = int(config["H"])
    input_dtype = str(config["input_dtype"])
    output_dtype = str(config["output_dtype"])
    scale_value = float(config["scale"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    _validate(
        input_dtype,
        output_dtype,
        M,
        H,
        str(config["input_layout"]),
        str(config["residual_layout"]),
        str(config["output_layout"]),
        scale_value,
        eps,
    )
    x_stride, residual_stride, y_stride = _row_strides(config)
    vec = int(_source_config(H)["vec"])
    for name, stride in (("x", x_stride), ("residual", residual_stride), ("y", y_stride)):
        if stride < H:
            raise ValueError(f"{name} row stride {stride} does not cover H={H}")
        if stride % vec != 0:
            raise ValueError(f"{name} row stride {stride} must be divisible by vec={vec}")

    torch_dtype = _torch_input_dtype(input_dtype)
    generator = torch.Generator(device="cuda").manual_seed(42)
    x = _allocate_strided(M, H, x_stride, torch_dtype)
    residual = _allocate_strided(M, H, residual_stride, torch_dtype)

    if (input_dtype == "bfloat16" and H >= 4096) or H >= 65536:
        columns = torch.arange(H, device="cuda")
        x_mag = torch.where(columns % 257 < 128, 0.5, 1.0)
        r_mag = torch.where(columns % 193 < 96, 0.25, 0.75)
        x_pattern = torch.where(columns % 2 == 0, x_mag, -x_mag).to(torch_dtype)
        r_pattern = torch.where(columns % 3 == 0, r_mag, -r_mag).to(torch_dtype)
        x["view"].copy_(x_pattern.expand(M, H))
        residual["view"].copy_(r_pattern.expand(M, H))
        x["view"][1::2].neg_()
        residual["view"][2::3].neg_()
    else:
        x["view"].normal_(generator=generator)
        residual["view"].normal_(generator=generator)

    weight_backing = torch.empty(H + _GUARD_ELEMENTS, dtype=torch_dtype, device="cuda")
    weight = weight_backing[:H]
    weight.normal_(generator=generator)
    if H >= 3:
        dtype_limit = torch.finfo(torch_dtype).max
        weight[0] = dtype_limit
        weight[1] = torch.tensor(1.0546875 * scale_value, dtype=torch_dtype, device="cuda")
        weight[2] = torch.tensor(1.0703125 * scale_value, dtype=torch_dtype, device="cuda")
    weight_backing[H:].fill_(_GUARD_VALUE)
    scale = torch.tensor([scale_value], dtype=torch.float32, device="cuda")
    return {
        "input": x["view"],
        "input_arg": x["arg"],
        "input_backing": x["backing"],
        "input_size": x["size"],
        "input_data_ptr": x["data_ptr"],
        "input_stride": x["stride"],
        "residual": residual["view"],
        "residual_arg": residual["arg"],
        "residual_backing": residual["backing"],
        "residual_size": residual["size"],
        "residual_data_ptr": residual["data_ptr"],
        "residual_stride": residual["stride"],
        "weight": weight,
        "weight_backing": weight_backing,
        "scale": scale,
        "x_row_stride": x_stride,
        "residual_row_stride": residual_stride,
        "y_row_stride": y_stride,
    }


def _prepare_output(
    M: int, H: int, row_stride: int, output_dtype: str, *, initialize_padding: bool
) -> dict[str, Any]:
    output = _allocate_strided(M, H, row_stride, _torch_output_dtype(output_dtype))
    if not initialize_padding:
        output["backing"][output["size"] :].fill_(_GUARD_VALUE)
    return output


def _raw_bytes(tensor):
    import torch

    return tensor.view(torch.uint8)


def _assert_guard(backing, size: int, *, name: str) -> None:
    import torch

    expected = torch.full(
        (_GUARD_ELEMENTS,), _GUARD_VALUE, dtype=backing.dtype, device=backing.device
    )
    if not torch.equal(_raw_bytes(backing[size:]), _raw_bytes(expected)):
        raise AssertionError(f"{name} guard was modified")


def _padding_sample(backing, M: int, H: int, stride: int):
    import torch

    if stride == H or M <= 1:
        return None
    width = stride - H
    count = (M - 1) * width
    if count <= 1_000_000:
        return backing.as_strided((M - 1, width), (stride, 1), H).reshape(-1)
    rows = sorted(row for row in {0, 1, (M - 2) // 2, M - 2} if 0 <= row < M - 1)
    cols = sorted({0, min(1, width - 1), width // 2, width - 1})
    row_index = torch.tensor(rows, device=backing.device, dtype=torch.int64)
    col_index = torch.tensor(cols, device=backing.device, dtype=torch.int64)
    return backing[row_index[:, None] * stride + H + col_index[None, :]].reshape(-1)


def _assert_padding(backing, size: int, M: int, H: int, stride: int, *, name: str) -> None:
    import torch

    _assert_guard(backing, size, name=name)
    padding = _padding_sample(backing, M, H, stride)
    if padding is not None:
        expected = torch.full_like(padding, _GUARD_VALUE)
        if not torch.equal(_raw_bytes(padding), _raw_bytes(expected)):
            raise AssertionError(f"{name} row padding was modified")


def _overflow_rows(M: int, H: int) -> list[int] | None:
    if M * H <= _INT32_MAX:
        return None
    boundary = _ceil_div(2**31, H)
    return sorted(
        {row for row in (0, 1, boundary - 1, boundary, boundary + 1, M - 1) if 0 <= row < M}
    )


def _checked_view(tensor, rows: list[int] | None):
    return tensor if rows is None else tensor[rows]


def _snapshot(data: dict[str, Any], M: int, H: int) -> dict[str, Any]:
    rows = _overflow_rows(M, H)
    return {
        "rows": rows,
        "input": _checked_view(data["input"], rows).clone(),
        "residual": _checked_view(data["residual"], rows).clone(),
        "weight": data["weight"].clone(),
        "scale": data["scale"].clone(),
    }


def _math_oracle(snapshot: dict[str, Any], eps: float, output_dtype: str):
    h = snapshot["input"].float() + snapshot["residual"].float()
    expected_residual = h.to(snapshot["input"].dtype)
    variance = h.square().mean(dim=-1, keepdim=True)
    y = (
        h
        * variance.add(eps).rsqrt()
        * snapshot["weight"].float()
        * snapshot["scale"].float().reciprocal()
    )
    fp8_max = 448.0 if output_dtype == "float8_e4m3fn" else 57344.0
    expected_output = y.clamp(-fp8_max, fp8_max).to(_torch_output_dtype(output_dtype))
    return expected_output, expected_residual


def _assert_raw_equal(actual, expected, *, name: str) -> None:
    import torch

    if not torch.equal(_raw_bytes(actual), _raw_bytes(expected)):
        mismatch = _raw_bytes(actual) != _raw_bytes(expected)
        count = int(mismatch.sum().item())
        raise AssertionError(f"{name}: {count} FP8 bytes differ")


def _assert_residual_close(actual, expected, *, name: str) -> None:
    import torch

    torch.testing.assert_close(
        actual, expected, rtol=1e-3, atol=1e-3, msg=lambda message: f"{name}: {message}"
    )


def _assert_math_close(actual, expected, *, name: str) -> None:
    import torch

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=1.0,
        atol=1.0,
        msg=lambda message: f"{name}: {message}",
    )


def _flashinfer_api(device):
    import flashinfer.norm as flashinfer_norm

    if flashinfer_norm._use_cuda_norm(device):
        raise AssertionError("FlashInfer fused add RMSNorm quant oracle dispatched to legacy CUDA")
    return flashinfer_norm.fused_add_rmsnorm_quant, flashinfer_norm


def _launch_tirx(executable, data, output, config: dict[str, Any]):
    M = int(config["M"])
    H = int(config["H"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    compact = _uses_compact_specialization(
        M,
        H,
        str(config["input_layout"]),
        str(config["residual_layout"]),
        str(config["output_layout"]),
    )
    if compact:
        return executable(
            output["arg"],
            data["input_arg"],
            data["residual_arg"],
            data["weight"],
            M,
            data["scale"],
            eps,
        )
    return executable(
        output["arg"],
        data["input_arg"],
        data["residual_arg"],
        data["weight"],
        M,
        data["scale"],
        eps,
        data["y_row_stride"],
        data["x_row_stride"],
        data["residual_row_stride"],
    )


@functools.cache
def _compiled_test_specialization(
    input_dtype: str, output_dtype: str, H: int, compact: bool, enable_pdl: bool
):
    from tirx_kernels.runner import compile_kernel

    layout = "compact" if compact else "strided"
    return compile_kernel(
        get_kernel(
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            M=1,
            H=H,
            input_layout=layout,
            residual_layout=layout,
            output_layout=layout,
            enable_pdl=enable_pdl,
            scale=1.0,
            eps=_DEFAULT_EPS,
            x_row_stride=H,
            residual_row_stride=H,
            y_row_stride=H,
        )
    )


def _assert_identity(data: dict[str, Any], output: dict[str, Any]) -> None:
    for name in ("input", "residual"):
        if data[name].data_ptr() != data[f"{name}_data_ptr"]:
            raise AssertionError(f"{name} data pointer changed")
        if data[name].stride() != data[f"{name}_stride"]:
            raise AssertionError(f"{name} stride changed")
    if output["view"].data_ptr() != output["data_ptr"]:
        raise AssertionError("output data pointer changed")
    if output["view"].stride() != output["stride"]:
        raise AssertionError("output stride changed")


def run_test(**config: Any) -> None:
    """Compile, launch, and validate both mutable outputs for one config."""
    import torch

    config = dict(config)
    M = int(config["M"])
    H = int(config["H"])
    output_dtype = str(config["output_dtype"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    enable_pdl = bool(config["enable_pdl"])
    data = _prepare_tensors(config)
    snapshot = _snapshot(data, M, H)
    output = _prepare_output(M, H, data["y_row_stride"], output_dtype, initialize_padding=True)
    compact = _uses_compact_specialization(
        M,
        H,
        str(config["input_layout"]),
        str(config["residual_layout"]),
        str(config["output_layout"]),
    )
    executable = _compiled_test_specialization(
        str(config["input_dtype"]), output_dtype, H, compact, enable_pdl
    )
    returned = _launch_tirx(executable, data, output, config)
    if returned is not None:
        raise AssertionError("TIRx fused add RMSNorm quant ABI must return None")

    reference = None
    reference_output = None
    if not _uses_rolled_fragment_loops(H):
        reference = _prepare_tensors(config)
        reference_output = _prepare_output(
            M, H, reference["y_row_stride"], output_dtype, initialize_padding=True
        )
        api, flashinfer_norm = _flashinfer_api(reference["input"].device)
        original_cute = flashinfer_norm.fused_add_rmsnorm_quant_cute
        cute_calls = 0

        def tracked_cute(*args, **kwargs):
            nonlocal cute_calls
            cute_calls += 1
            return original_cute(*args, **kwargs)

        flashinfer_norm.fused_add_rmsnorm_quant_cute = tracked_cute
        try:
            reference_returned = api(
                reference_output["view"],
                reference["input"],
                reference["residual"],
                reference["weight"],
                reference["scale"],
                eps,
                enable_pdl=enable_pdl,
            )
        finally:
            flashinfer_norm.fused_add_rmsnorm_quant_cute = original_cute
        if cute_calls != 1:
            raise AssertionError(f"expected one CuTe-DSL oracle dispatch, observed {cute_calls}")
        if reference_returned is not None:
            raise AssertionError("FlashInfer fused_add_rmsnorm_quant must return None")

    torch.cuda.synchronize()
    rows = snapshot["rows"]
    actual_output = _checked_view(output["view"], rows)
    actual_residual = _checked_view(data["residual"], rows)
    if not torch.isfinite(actual_output.float()).all():
        raise AssertionError("TIRx FP8 output contains non-finite values")
    if not torch.isfinite(actual_residual.float()).all():
        raise AssertionError("TIRx residual output contains non-finite values")
    if reference is not None and reference_output is not None:
        _assert_raw_equal(
            actual_output,
            _checked_view(reference_output["view"], rows),
            name="FlashInfer raw-byte output oracle",
        )
        _assert_residual_close(
            actual_residual,
            _checked_view(reference["residual"], rows),
            name="FlashInfer residual oracle",
        )
    else:
        # The rolled-fragment rows exceed what the CuTe reference can run, so
        # the FP32 oracle is the only available arbiter for them.
        oracle_output, oracle_residual = _math_oracle(snapshot, eps, output_dtype)
        _assert_math_close(actual_output, oracle_output, name="independent FP32 output oracle")
        _assert_residual_close(
            actual_residual, oracle_residual, name="independent FP32 residual oracle"
        )
    if not torch.equal(_checked_view(data["input"], rows), snapshot["input"]):
        raise AssertionError("TIRx modified input")
    if not torch.equal(data["weight"], snapshot["weight"]):
        raise AssertionError("TIRx modified weight")
    if not torch.equal(data["scale"], snapshot["scale"]):
        raise AssertionError("TIRx modified scale")
    _assert_identity(data, output)
    _assert_padding(
        data["input_backing"], data["input_size"], M, H, data["x_row_stride"], name="TIRx input"
    )
    _assert_padding(
        data["residual_backing"],
        data["residual_size"],
        M,
        H,
        data["residual_row_stride"],
        name="TIRx residual",
    )
    _assert_padding(
        output["backing"], output["size"], M, H, data["y_row_stride"], name="TIRx output"
    )


def prepare_bench(**config: Any):
    """Compile the specialization before the bench suite assigns a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(config), "executable": compile_kernel(get_kernel(**config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    state: dict[str, Any],
    *,
    warmup=None,
    repeat=None,
    timer=None,
    rounds=1,
    cooldown_s=1.0,
    **kwargs: Any,
):
    """Build independent mutable closures and validate them before timing."""
    import torch

    config = dict(state["config"])
    config.update(kwargs)
    M = int(config["M"])
    H = int(config["H"])
    output_dtype = str(config["output_dtype"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    enable_pdl = bool(config["enable_pdl"])
    tirx_data = _prepare_tensors(config)
    reference_data = _prepare_tensors(config)
    tirx_output = _prepare_output(
        M, H, tirx_data["y_row_stride"], output_dtype, initialize_padding=False
    )
    reference_output = _prepare_output(
        M, H, reference_data["y_row_stride"], output_dtype, initialize_padding=False
    )
    executable = state["executable"]

    def tirx_launch():
        return _launch_tirx(executable, tirx_data, tirx_output, config)

    if tirx_launch() is not None:
        raise AssertionError("TIRx benchmark closure must return None")
    torch.cuda.synchronize()

    def build_flashinfer_reference():
        api, flashinfer_norm = _flashinfer_api(reference_data["input"].device)
        original_cute = flashinfer_norm.fused_add_rmsnorm_quant_cute
        cute_calls = 0

        def tracked_cute(*args, **kwargs):
            nonlocal cute_calls
            cute_calls += 1
            return original_cute(*args, **kwargs)

        def flashinfer_launch():
            return api(
                reference_output["view"],
                reference_data["input"],
                reference_data["residual"],
                reference_data["weight"],
                reference_data["scale"],
                eps,
                enable_pdl=enable_pdl,
            )

        flashinfer_norm.fused_add_rmsnorm_quant_cute = tracked_cute
        try:
            returned = flashinfer_launch()
        finally:
            flashinfer_norm.fused_add_rmsnorm_quant_cute = original_cute
        if cute_calls != 1:
            raise AssertionError(f"expected one CuTe-DSL benchmark warmup, observed {cute_calls}")
        if returned is not None:
            raise AssertionError("FlashInfer benchmark closure must return None")
        torch.cuda.synchronize()
        _assert_raw_equal(
            tirx_output["view"], reference_output["view"], name="benchmark raw-byte output precheck"
        )
        _assert_residual_close(
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
    """Benchmark the implemented kernel against FlashInfer CuTe-DSL."""
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
