# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400e330fb2debe0bf8730d9424a1d37927f), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer CuTe-DSL 2-D RMSNorm with FP8 quantization.

The source implementation is ``RMSNormQuantKernel`` and its shared helpers in
``flashinfer/norm/kernels/rmsnorm.py`` and ``flashinfer/norm/utils.py``.  The
public dispatch is ``rmsnorm_quant`` in ``flashinfer/norm/__init__.py``.
"""

import contextlib
import functools
import math
from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.runner import bench

KERNEL_META = {
    "name": "flashinfer_rmsnorm_quant",
    "category": "flashinfer",
    "compute_capability": 10,
}

_INPUT_DTYPES = ("float16", "bfloat16")
_OUTPUT_DTYPES = ("float8_e4m3fn", "float8_e5m2")
_LAYOUTS = ("compact", "strided")
_INT32_MAX = 2**31 - 1
_DEFAULT_EPS = 1e-6
_SCALES = (0.01, 1.0, 10.0)
_OPTIN_SMEM_BYTES = 232448
_INPUT_ELEM_BYTES = 2


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


def _rmsnorm_derived_config(H: int, cluster_n: int) -> dict[str, int]:
    """Return the inherited RMSNorm launch terms used by the smem estimate."""
    H_per_cta = H // cluster_n
    tpr = _threads_per_row(H_per_cta)
    threads = 128 if H_per_cta <= 16384 else 256
    rows = threads // tpr
    warps_per_row = max(tpr // 32, 1)
    vec = min(H_per_cta & -H_per_cta, 8)
    vec_blocks = max(1, _ceil_div(H_per_cta // vec, tpr))
    cols = vec * vec_blocks * tpr
    return {
        "H_per_cta": H_per_cta,
        "tpr": tpr,
        "threads": threads,
        "rows": rows,
        "warps_per_row": warps_per_row,
        "vec": vec,
        "vec_blocks": vec_blocks,
        "cols": cols,
    }


def _estimate_smem(H: int, cluster_n: int) -> int:
    source = _rmsnorm_derived_config(H, cluster_n)
    rows = source["rows"]
    warps_per_row = source["warps_per_row"]
    cols = source["cols"]
    return (
        rows * cols * _INPUT_ELEM_BYTES
        + rows * warps_per_row * cluster_n * 4
        + (8 if cluster_n > 1 else 0)
    )


def _source_config(H: int) -> dict[str, int | bool]:
    cluster_n = 16
    for candidate in (1, 2, 4, 8, 16):
        if H % candidate == 0 and _estimate_smem(H, candidate) <= _OPTIN_SMEM_BYTES:
            cluster_n = candidate
            break

    source = _rmsnorm_derived_config(H, cluster_n)
    H_per_cta = source["H_per_cta"]
    tpr = source["tpr"]
    threads = source["threads"]
    if H_per_cta > 8192 and threads < 256:
        threads = 256
    rows = threads // tpr
    warps_per_row = max(tpr // 32, 1)
    vec = source["vec"]
    copy_bits = 16 * vec
    vec_blocks = source["vec_blocks"]
    cols = source["cols"]
    tile_bytes = rows * cols * _INPUT_ELEM_BYTES
    use_async = copy_bits >= 32 and tile_bytes <= _OPTIN_SMEM_BYTES // 2
    reduce_bytes = rows * warps_per_row * 4 * (cluster_n if cluster_n > 1 else 1)
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


def _ptx_unary(chain: str, value, dtype: str = "float32"):
    out = K.alloc_local((1,), dtype)
    K.ptx[chain](out[0], value)
    return out[0]


def _ptx_binary(chain: str, lhs, rhs, dtype: str = "float32"):
    out = K.alloc_local((1,), dtype)
    K.ptx[chain](out[0], lhs, rhs)
    return out[0]


def _ptx_ternary(chain: str, lhs, rhs, acc, dtype: str = "float32"):
    out = K.alloc_local((1,), dtype)
    K.ptx[chain](out[0], lhs, rhs, acc)
    return out[0]


def _add_f32(lhs, rhs):
    return _ptx_binary("add.f32", lhs, rhs)


def _mul_f32(lhs, rhs):
    return _ptx_binary("mul.f32", lhs, rhs)


def _div_rn_f32(lhs, rhs):
    return _ptx_binary("div.rn.f32", lhs, rhs)


def _fma_rn_f32(lhs, rhs, acc):
    return _ptx_ternary("fma.rn.f32", lhs, rhs, acc)


def _fma_half_inputs_to_f32(lhs, rhs, input_dtype: str):
    out = K.alloc_local((1,), "float32")
    if input_dtype == "float16":
        K.ptx.fma.rn.f32.f16(out[0], lhs, rhs, K.float32(0.0))
    else:
        K.ptx.fma.rn.f32.bf16(out[0], lhs, rhs, K.float32(0.0))
    return out[0]


def _rcp_approx_ftz(value):
    return _ptx_unary("rcp.approx.ftz.f32", value)


def _rsqrt_approx_ftz(value):
    return _ptx_unary("rsqrt.approx.ftz.f32", value)


def _cvt_to_f32(bits, input_dtype: str):
    if input_dtype == "float16":
        return _ptx_unary("cvt.f32.f16", K.cast(bits, "uint16"))
    return _ptx_unary("cvt.f32.bf16", K.cast(bits, "uint16"))


def _cvt_fp8_pair(low, high, output_dtype: str):
    pair = K.alloc_local((1,), "uint16")
    if output_dtype == "float8_e4m3fn":
        K.ptx.cvt.rn.satfinite.e4m3x2.f32(pair[0], high, low)
    else:
        K.ptx.cvt.rn.satfinite.e5m2x2.f32(pair[0], high, low)
    return pair[0]


def _pack_b16_pair(low, high):
    word = K.alloc_local((1,), "uint32")
    K.ptx.mov.b32(word[0], low, high)
    return word[0]


def _maximum_f32(value, lower):
    predicate = K.local_scalar("uint32")
    out = K.alloc_local((1,), "float32")
    K.ptx.setp.le.f32(predicate, value, lower)
    K.ptx.selp.f32(out[0], lower, value, K.ptx.pred(predicate))
    return out[0]


def _minimum_f32(value, upper):
    predicate = K.local_scalar("uint32")
    out = K.alloc_local((1,), "float32")
    K.ptx.setp.ge.f32(predicate, value, upper)
    K.ptx.selp.f32(out[0], upper, value, K.ptx.pred(predicate))
    return out[0]


def _shfl_bfly_f32(value, lane_xor: int):
    out = K.alloc_local((1,), "uint32")
    K.ptx.shfl_sync.bfly.b32(
        out[0],
        K.reinterpret("uint32", value),
        K.uint32(lane_xor),
        K.uint32(31),
        K.uint32(0xFFFFFFFF),
    )
    return K.reinterpret("float32", out[0])


def _butterfly_sum_f32(value, lane_xors: tuple[int, ...]):
    for lane_xor in lane_xors:
        value = _add_f32(value, _shfl_bfly_f32(value, lane_xor))
    return value


def _mapa_u32(pointer, peer):
    mapped = K.alloc_local((1,), "uint32")
    K.ptx.mapa.shared__cluster.u32(
        mapped[0], K.cuda.cvta_generic_to_shared(pointer), K.cast(peer, "uint32")
    )
    return mapped[0]


def _cluster_mbarrier_wait(pointer):
    return K.cuda.mbarrier_wait(pointer, K.int32(0))


@contextlib.contextmanager
def _runtime_guard(predicate):
    """Open a Kern runtime branch, or no branch for a statically true case."""
    if predicate is None:
        yield
    else:
        with K.If(predicate), K.Then():
            yield


def _load_global_bits(buffer, index, values, value_offset, VEC: K.constexpr):
    if VEC == 1:
        K.ptx.ld.global_.b16(values[value_offset], buffer.ptr_to([index]))
    elif VEC == 2:
        K.ptx.ld.global_.v2.b16(
            values[value_offset], values[value_offset + 1], buffer.ptr_to([index])
        )
    else:
        words = K.alloc_local((VEC // 2,), "uint32")
        if VEC == 4:
            K.ptx.ld.global_.v2.b32(words[0], words[1], buffer.ptr_to([index]))
        else:
            K.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
        for pair in range(VEC // 2):
            K.ptx.mov.b32(
                values[value_offset + pair * 2], values[value_offset + pair * 2 + 1], words[pair]
            )


def _load_shared_bits(shared_raw, byte_offset, values, value_offset, VEC: K.constexpr):
    if VEC == 2:
        K.ptx.ld.shared.v2.b16(
            values[value_offset], values[value_offset + 1], shared_raw.ptr_to([byte_offset])
        )
    else:
        words = K.alloc_local((VEC // 2,), "uint32")
        if VEC == 4:
            halves = K.alloc_local((4,), "uint16")
            K.ptx.ld.shared.v4.b16(
                halves[0], halves[1], halves[2], halves[3], shared_raw.ptr_to([byte_offset])
            )
            for value in range(4):
                K.assign(values[value_offset + value], halves[value])
        else:
            K.ptx.ld.shared.v4.b32(
                words[0], words[1], words[2], words[3], shared_raw.ptr_to([byte_offset])
            )
            for pair in range(4):
                K.ptx.mov.b32(
                    values[value_offset + pair * 2],
                    values[value_offset + pair * 2 + 1],
                    words[pair],
                )


def _load_global_words(buffer, index, words, word_offset, VEC: K.constexpr):
    if VEC == 2:
        K.ptx.ld.global_.b32(words[word_offset], buffer.ptr_to([index]))
    elif VEC == 4:
        K.ptx.ld.global_.v2.b32(words[word_offset], words[word_offset + 1], buffer.ptr_to([index]))
    else:
        K.ptx.ld.global_.v4.b32(
            words[word_offset],
            words[word_offset + 1],
            words[word_offset + 2],
            words[word_offset + 3],
            buffer.ptr_to([index]),
        )


def _load_shared_words(shared_raw, byte_offset, words, word_offset, VEC: K.constexpr):
    if VEC == 2:
        K.ptx.ld.shared.b32(words[word_offset], shared_raw.ptr_to([byte_offset]))
    elif VEC == 4:
        K.ptx.ld.shared.v2.b32(
            words[word_offset], words[word_offset + 1], shared_raw.ptr_to([byte_offset])
        )
    else:
        K.ptx.ld.shared.v4.b32(
            words[word_offset],
            words[word_offset + 1],
            words[word_offset + 2],
            words[word_offset + 3],
            shared_raw.ptr_to([byte_offset]),
        )


def _short_dtype(dtype: str) -> str:
    return {"float16": "fp16", "bfloat16": "bf16", "float8_e4m3fn": "e4m3", "float8_e5m2": "e5m2"}[
        dtype
    ]


def _layout_code(layout: str) -> str:
    return {"compact": "c", "strided": "s"}[layout]


def _scale_code(scale: float) -> str:
    return {0.01: "0p01", 1.0: "1", 10.0: "10"}[scale]


def _cfg(
    input_dtype: str,
    output_dtype: str,
    M: int,
    H: int,
    input_layout: str,
    output_layout: str,
    enable_pdl: bool,
    scale: float,
    *,
    eps: float = _DEFAULT_EPS,
    x_row_stride: int | None = None,
    y_row_stride: int | None = None,
    suffix: str | None = None,
) -> dict[str, Any]:
    label = (
        f"{_short_dtype(input_dtype)}_{_short_dtype(output_dtype)}_m{M}_h{H}_"
        f"x{_layout_code(input_layout)}_y{_layout_code(output_layout)}_"
        f"pdl{int(enable_pdl)}_s{_scale_code(scale)}"
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
        "output_layout": output_layout,
        "enable_pdl": enable_pdl,
        "scale": scale,
        "eps": eps,
    }
    if x_row_stride is not None:
        config["x_row_stride"] = x_row_stride
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
    _cfg("bfloat16", "float8_e4m3fn", 32, 4096, "compact", "compact", False, 1.0),
    _cfg("bfloat16", "float8_e4m3fn", 64, 8192, "compact", "compact", False, 1.0),
]

_I64_CONFIGS = [
    _cfg(
        "bfloat16",
        "float8_e4m3fn",
        2,
        128,
        "strided",
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
        False,
        1.0,
        suffix="compact_overflow_to_strided_i64",
    ),
]

_TRACE_CONFIGS = [
    _cfg("bfloat16", "float8_e4m3fn", M, H, "compact", "compact", False, 1.0)
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
        False,
        1.0,
        suffix="vec8_tpr8_async",
    ),
    _cfg("bfloat16", "float8_e5m2", 989, 66, "compact", "compact", True, 10.0, suffix="vec2"),
    _cfg(
        "float16", "float8_e4m3fn", 3, 16385, "compact", "compact", False, 1.0, suffix="vec1_sync"
    ),
    _cfg(
        "float16",
        "float8_e4m3fn",
        3,
        65536,
        "compact",
        "compact",
        False,
        1.0,
        suffix="cluster1_sync_capacity",
    ),
    _cfg(
        "bfloat16",
        "float8_e5m2",
        3,
        131072,
        "compact",
        "compact",
        True,
        1.0,
        suffix="cluster2_sync",
    ),
    _cfg(
        "float16",
        "float8_e4m3fn",
        3,
        262144,
        "compact",
        "compact",
        False,
        1.0,
        suffix="cluster4_sync",
    ),
    _cfg(
        "bfloat16",
        "float8_e5m2",
        3,
        524288,
        "compact",
        "compact",
        True,
        1.0,
        suffix="cluster8_sync",
    ),
    _cfg(
        "bfloat16",
        "float8_e5m2",
        3,
        1048576,
        "compact",
        "compact",
        True,
        1.0,
        suffix="cluster16_sync",
    ),
]

_ABI_CONFIGS = [
    _cfg(
        "float16",
        "float8_e5m2",
        989,
        500,
        "compact",
        "strided",
        True,
        1.0,
        eps=1e-4,
        y_row_stride=1000,
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

assert len(_UPSTREAM_CONFIGS) == 1536
assert len(CONFIGS) == 1552
assert len({config["label"] for config in CONFIGS}) == len(CONFIGS)

_CONFIG_BY_LABEL = {config["label"]: config for config in CONFIGS}
_BENCH_LABELS = (
    "bf16_e4m3_m32_h4096_xc_yc_pdl0_s1",
    "bf16_e4m3_m64_h8192_xc_yc_pdl0_s1",
    "fp16_e5m2_m989_h500_xc_yc_pdl0_s0p01",
    "bf16_e5m2_m989_h66_xc_yc_pdl1_s10_vec2",
    "fp16_e4m3_m989_h111_xs_yc_pdl0_s1",
    "fp16_e5m2_m989_h500_xc_ys_pdl1_s1_eps1e4_full_abi",
    "fp16_e4m3_m3_h65536_xc_yc_pdl0_s1_cluster1_sync_capacity",
    "bf16_e5m2_m3_h1048576_xc_yc_pdl1_s1_cluster16_sync",
)
BENCH_CONFIGS = [_CONFIG_BY_LABEL[label] for label in _BENCH_LABELS]


def _validate(
    input_dtype: str,
    output_dtype: str,
    M: int,
    H: int,
    input_layout: str,
    output_layout: str,
    scale: float,
    eps: float,
    x_row_stride: int | None,
    y_row_stride: int | None,
) -> None:
    if input_dtype not in _INPUT_DTYPES:
        raise ValueError(f"unsupported input dtype: {input_dtype}")
    if output_dtype not in _OUTPUT_DTYPES:
        raise ValueError(f"unsupported output dtype: {output_dtype}")
    if input_layout not in _LAYOUTS or output_layout not in _LAYOUTS:
        raise ValueError(f"unsupported layouts: input={input_layout}, output={output_layout}")
    if M <= 0 or H <= 0:
        raise ValueError(f"M and H must be positive, got M={M}, H={H}")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be positive and finite, got {scale}")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be positive and finite, got {eps}")
    for name, layout, stride in (
        ("x_row_stride", input_layout, x_row_stride),
        ("y_row_stride", output_layout, y_row_stride),
    ):
        if layout == "strided" and stride is not None and stride < H:
            raise ValueError(f"{name} must be at least H={H}, got {stride}")


def _uses_compact_specialization(M: int, H: int, input_layout: str, output_layout: str) -> bool:
    return input_layout == "compact" and output_layout == "compact" and M * H <= _INT32_MAX


def get_kernel(
    input_dtype: str,
    output_dtype: str,
    M: int,
    H: int,
    input_layout: str,
    output_layout: str,
    enable_pdl: bool,
    scale: float,
    eps: float = _DEFAULT_EPS,
    x_row_stride: int | None = None,
    y_row_stride: int | None = None,
    **kwargs: Any,
):
    """Return the compact or explicit-i64-strided source specialization."""
    _validate(
        input_dtype,
        output_dtype,
        M,
        H,
        input_layout,
        output_layout,
        scale,
        eps,
        x_row_stride,
        y_row_stride,
    )
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
    copy_bytes = copy_bits // 8
    reduce_base = tile_bytes if use_async else 0
    reduce_count = rows * warps_per_row * cluster_n
    mbar_offset = reduce_base + reduce_count * 4
    expected_bytes = reduce_count * 4
    total_partials_per_row = warps_per_row * cluster_n
    full_columns = H == cluster_n * cols
    row_lane_xors = tuple(lane_xor for lane_xor in (1, 2, 4, 8, 16) if lane_xor < min(tpr, 32))
    full_lane_xors = (1, 2, 4, 8, 16)
    fp8_max = 448.0 if output_dtype == "float8_e4m3fn" else 57344.0

    del kwargs
    x_row_stride_hint = H if x_row_stride is None else int(x_row_stride)
    y_row_stride_hint = H if y_row_stride is None else int(y_row_stride)
    if x_row_stride_hint % vec != 0:
        raise ValueError(f"x_row_stride={x_row_stride_hint} must be divisible by vec={vec}")
    if y_row_stride_hint % vec != 0:
        raise ValueError(f"y_row_stride={y_row_stride_hint} must be divisible by vec={vec}")

    max_registers = None
    if threads == 128:
        max_registers = 64 if enable_pdl else (80 if H == 8192 else 93)

    def entry_registers():
        if max_registers is None:
            return contextlib.nullcontext()
        return K.attr({"tirx.max_registers": max_registers})

    def kernel_body(x, weight, out, runtime_M, scale_buffer, runtime_eps, x_stride, y_stride):
        # TIRX_TRANSCRIBE_START flashinfer_rmsnorm_quant
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

        scale_bits = K.alloc_local((1,), "uint32")
        K.ptx.ld.global_.b32(scale_bits[0], scale_buffer.ptr_to([0]))
        scale_value = K.local_scalar(
            K.f32, init=K.reinterpret("float32", scale_bits[0]), name="scale_value"
        )
        inv_scale = K.local_scalar(K.f32, init=_rcp_approx_ftz(scale_value), name="inv_scale")

        block_x = K.local_scalar(K.i32, init=K.cast(block_x_raw, "int32"), name="block_x")
        row_in_cta = K.local_scalar(K.i32, init=tid // tpr, name="row_in_cta")
        thread_in_row = K.local_scalar(K.i32, init=tid % tpr, name="thread_in_row")
        row_i64 = K.local_scalar(
            K.i64,
            init=K.cast(block_x, "int64") * K.int64(rows) + K.cast(row_in_cta, "int64"),
            name="row_i64",
        )
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

        x_bits = K.alloc_local((total_values if vec == 1 else 1,), "uint16")
        w_bits = K.alloc_local((total_values if vec == 1 else 1,), "uint16")
        x_words = K.alloc_local((packed_pairs if vec > 1 else 1,), "uint32")
        w_words = K.alloc_local((packed_pairs if vec > 1 else 1,), "uint32")
        x_f32_pairs = K.alloc_local((packed_pairs,), "uint64")
        w_f32_pairs = K.alloc_local((packed_pairs,), "uint64")
        x_f32_scalar = K.alloc_local((1,), "float32")
        w_f32_scalar = K.alloc_local((1,), "float32")
        undefined_f32 = K.alloc_local((1,), "float32")

        if not use_async:
            if vec == 1:
                for value in range(total_values):
                    K.assign(x_bits[value], K.uint16(0))
            else:
                for pair in range(packed_pairs):
                    K.assign(x_words[pair], K.uint32(0))

        if use_async and not enable_pdl and H == 8192:
            with _runtime_guard(None if rows == 1 else row_valid):
                for vb in range(vec_blocks):
                    local_col = K.local_scalar(
                        K.i32, init=(thread_in_row + vb * tpr) * vec, name="local_col"
                    )
                    absolute_col = K.local_scalar(
                        K.i32, init=block_y * cols + local_col, name="absolute_col"
                    )
                    col_valid = K.local_scalar("bool", init=absolute_col < H, name="col_valid")
                    if compact:
                        x_offset = K.local_scalar(
                            K.i32,
                            init=K.cast(
                                row_i64 * K.int64(H) + K.cast(absolute_col, "int64"), "int32"
                            ),
                            name="x_offset",
                        )
                    else:
                        x_offset = K.local_scalar(
                            K.i64,
                            init=row_i64 * x_stride + K.cast(absolute_col, "int64"),
                            name="x_offset",
                        )
                    if full_columns:
                        source_bytes = K.local_scalar(
                            K.u32, init=K.uint32(copy_bytes), name="source_bytes"
                        )
                    else:
                        source_bytes = K.local_scalar(
                            K.u32,
                            init=K.cast(K.if_then_else(col_valid, copy_bytes, 0), "uint32"),
                            name="source_bytes",
                        )
                    K.ptx["cp.async.ca.shared.global"](
                        shared_raw.ptr_to([(row_in_cta * cols + local_col) * _INPUT_ELEM_BYTES]),
                        x.ptr_to([x_offset]),
                        copy_bytes,
                        source_bytes,
                    )
        else:
            for vb in range(vec_blocks):
                local_col = K.local_scalar(
                    K.i32, init=(thread_in_row + vb * tpr) * vec, name="local_col"
                )
                absolute_col = K.local_scalar(
                    K.i32, init=block_y * cols + local_col, name="absolute_col"
                )
                col_valid = K.local_scalar("bool", init=absolute_col < H, name="col_valid")
                if compact:
                    x_offset = K.local_scalar(
                        K.i32,
                        init=K.cast(row_i64 * K.int64(H) + K.cast(absolute_col, "int64"), "int32"),
                        name="x_offset",
                    )
                else:
                    x_offset = K.local_scalar(
                        K.i64,
                        init=row_i64 * x_stride + K.cast(absolute_col, "int64"),
                        name="x_offset",
                    )

                if use_async:
                    with _runtime_guard(None if rows == 1 else row_valid):
                        if full_columns:
                            source_bytes = K.local_scalar(
                                K.u32, init=K.uint32(copy_bytes), name="source_bytes"
                            )
                        else:
                            source_bytes = K.local_scalar(
                                K.u32,
                                init=K.cast(K.if_then_else(col_valid, copy_bytes, 0), "uint32"),
                                name="source_bytes",
                            )
                        K.ptx["cp.async.ca.shared.global"](
                            shared_raw.ptr_to(
                                [(row_in_cta * cols + local_col) * _INPUT_ELEM_BYTES]
                            ),
                            x.ptr_to([x_offset]),
                            copy_bytes,
                            source_bytes,
                        )
                else:
                    if rows == 1 and full_columns:
                        load_guard = None
                    elif rows == 1:
                        load_guard = col_valid
                    elif full_columns:
                        load_guard = row_valid
                    else:
                        load_guard = K.And(row_valid, col_valid)
                    with _runtime_guard(load_guard):
                        if vec == 1:
                            _load_global_bits(x, x_offset, x_bits, vb, VEC=vec)
                        else:
                            _load_global_words(x, x_offset, x_words, vb * (vec // 2), VEC=vec)

        if use_async and not enable_pdl and compact and full_columns and (H == 4096 or H == 8192):
            K.ptx.bar.warp.sync(K.uint32(0xFFFFFFFF))

        if use_async:
            K.ptx.cp.async_.commit_group()

        if use_async and full_columns and compact and H == 4096:
            weight_base_col = K.local_scalar(
                K.i32, init=block_y * cols + thread_in_row * vec, name="weight_base_col"
            )
            for vb in range(vec_blocks):
                K.ptx.ld.global_.v4.b32(
                    w_words[vb * 4],
                    w_words[vb * 4 + 1],
                    w_words[vb * 4 + 2],
                    w_words[vb * 4 + 3],
                    K.ptx.addr(
                        weight.ptr_to([weight_base_col]), vb * tpr * vec * _INPUT_ELEM_BYTES
                    ),
                )
                if vb == vec_blocks // 2 - 1:
                    K.ptx.bar.warp.sync(K.uint32(0xFFFFFFFF))
        else:
            for vb in range(vec_blocks):
                local_col = K.local_scalar(
                    K.i32, init=(thread_in_row + vb * tpr) * vec, name="local_col"
                )
                absolute_col = K.local_scalar(
                    K.i32, init=block_y * cols + local_col, name="absolute_col"
                )
                with _runtime_guard(None if full_columns else absolute_col < H):
                    if vec == 1:
                        _load_global_bits(weight, absolute_col, w_bits, vb, VEC=vec)
                    else:
                        _load_global_words(weight, absolute_col, w_words, vb * (vec // 2), VEC=vec)
                if (
                    use_async
                    and not enable_pdl
                    and compact
                    and full_columns
                    and H == 8192
                    and vb == vec_blocks // 2 - 1
                ):
                    K.ptx.bar.warp.sync(K.uint32(0xFFFFFFFF))

        if use_async:
            K.ptx.cp.async_.wait_group(0)
            if full_columns and compact and H == 4096:
                shared_load_base = K.local_scalar(
                    K.i32,
                    init=(row_in_cta * cols + thread_in_row * vec) * _INPUT_ELEM_BYTES,
                    name="shared_load_base",
                )
                for vb in range(vec_blocks):
                    K.ptx.ld.shared.v4.b32(
                        x_words[vb * 4],
                        x_words[vb * 4 + 1],
                        x_words[vb * 4 + 2],
                        x_words[vb * 4 + 3],
                        K.ptx.addr(
                            shared_raw.ptr_to([shared_load_base]),
                            vb * tpr * vec * _INPUT_ELEM_BYTES,
                        ),
                    )
            else:
                for vb in range(vec_blocks):
                    local_col = K.local_scalar(
                        K.i32, init=(thread_in_row + vb * tpr) * vec, name="local_col"
                    )
                    if vec == 1:
                        _load_shared_bits(
                            shared_raw,
                            (row_in_cta * cols + local_col) * _INPUT_ELEM_BYTES,
                            x_bits,
                            vb,
                            VEC=vec,
                        )
                    else:
                        _load_shared_words(
                            shared_raw,
                            (row_in_cta * cols + local_col) * _INPUT_ELEM_BYTES,
                            x_words,
                            vb * (vec // 2),
                            VEC=vec,
                        )

        if total_values == 1:
            K.assign(x_f32_scalar[0], _cvt_to_f32(x_bits[0], input_dtype))
            local_sum = K.local_scalar(
                K.f32,
                init=_fma_half_inputs_to_f32(x_bits[0], x_bits[0], input_dtype),
                name="local_sum",
            )
        else:
            if vec == 1:
                for pair in range(packed_pairs):
                    low_x = K.local_scalar(
                        K.f32, init=_cvt_to_f32(x_bits[pair * 2], input_dtype), name="low_x"
                    )
                    high_x = K.local_scalar(K.f32, init=undefined_f32[0], name="high_x")
                    if pair * 2 + 1 < total_values:
                        K.assign(high_x, _cvt_to_f32(x_bits[pair * 2 + 1], input_dtype))
                    K.ptx.mov.b64(x_f32_pairs[pair], low_x, high_x)
            else:
                for pair in range(packed_pairs):
                    low_bits = K.alloc_local((1,), "uint16")
                    high_bits = K.alloc_local((1,), "uint16")
                    K.ptx.mov.b32(low_bits[0], high_bits[0], x_words[pair])
                    low_x = K.local_scalar(
                        K.f32, init=_cvt_to_f32(low_bits[0], input_dtype), name="low_x"
                    )
                    high_x = K.local_scalar(
                        K.f32, init=_cvt_to_f32(high_bits[0], input_dtype), name="high_x"
                    )
                    K.ptx.mov.b64(x_f32_pairs[pair], low_x, high_x)

            x_sq = K.alloc_local((total_values,), "float32")
            for reverse_pair in range(packed_pairs):
                pair = packed_pairs - reverse_pair - 1
                product = K.alloc_local((1,), "uint64")
                K.ptx.mul.f32x2(product[0], x_f32_pairs[pair], x_f32_pairs[pair])
                if pair * 2 + 1 < total_values:
                    K.ptx.mov.b64(x_sq[pair * 2], x_sq[pair * 2 + 1], product[0])
                else:
                    discarded = K.alloc_local((1,), "float32")
                    K.ptx.mov.b64(x_sq[pair * 2], discarded[0], product[0])

            local_sum = K.local_scalar(K.f32, init=K.float32(0.0), name="local_sum")
            for value in range(total_values):
                K.assign(local_sum, _add_f32(local_sum, x_sq[value]))

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

        if H == 1:
            shifted = K.local_scalar(K.f32, init=_add_f32(sum_sq, runtime_eps), name="shifted")
        elif H & (H - 1) == 0:
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

        if use_async:
            if full_columns and compact and H == 4096:
                shared_reload_base = K.local_scalar(
                    K.i32,
                    init=(row_in_cta * cols + thread_in_row * vec) * _INPUT_ELEM_BYTES,
                    name="shared_reload_base",
                )
                for vb in range(vec_blocks):
                    K.ptx.ld.shared.v4.b32(
                        x_words[vb * 4],
                        x_words[vb * 4 + 1],
                        x_words[vb * 4 + 2],
                        x_words[vb * 4 + 3],
                        K.ptx.addr(
                            shared_raw.ptr_to([shared_reload_base]),
                            vb * tpr * vec * _INPUT_ELEM_BYTES,
                        ),
                    )
            else:
                for vb in range(vec_blocks):
                    local_col = K.local_scalar(
                        K.i32, init=(thread_in_row + vb * tpr) * vec, name="local_col"
                    )
                    if vec == 1:
                        _load_shared_bits(
                            shared_raw,
                            (row_in_cta * cols + local_col) * _INPUT_ELEM_BYTES,
                            x_bits,
                            vb,
                            VEC=vec,
                        )
                    else:
                        _load_shared_words(
                            shared_raw,
                            (row_in_cta * cols + local_col) * _INPUT_ELEM_BYTES,
                            x_words,
                            vb * (vec // 2),
                            VEC=vec,
                        )
        if total_values == 1:
            if use_async:
                K.assign(x_f32_scalar[0], _cvt_to_f32(x_bits[0], input_dtype))
            K.assign(w_f32_scalar[0], _cvt_to_f32(w_bits[0], input_dtype))
        else:
            if use_async:
                if vec == 1:
                    for pair in range(packed_pairs):
                        low_x = K.local_scalar(
                            K.f32, init=_cvt_to_f32(x_bits[pair * 2], input_dtype), name="low_x"
                        )
                        high_x = K.local_scalar(K.f32, init=undefined_f32[0], name="high_x")
                        if pair * 2 + 1 < total_values:
                            K.assign(high_x, _cvt_to_f32(x_bits[pair * 2 + 1], input_dtype))
                        K.ptx.mov.b64(x_f32_pairs[pair], low_x, high_x)
                else:
                    for pair in range(packed_pairs):
                        low_bits = K.alloc_local((1,), "uint16")
                        high_bits = K.alloc_local((1,), "uint16")
                        K.ptx.mov.b32(low_bits[0], high_bits[0], x_words[pair])
                        low_x = K.local_scalar(
                            K.f32, init=_cvt_to_f32(low_bits[0], input_dtype), name="low_x"
                        )
                        high_x = K.local_scalar(
                            K.f32, init=_cvt_to_f32(high_bits[0], input_dtype), name="high_x"
                        )
                        K.ptx.mov.b64(x_f32_pairs[pair], low_x, high_x)
            if vec == 1:
                for pair in range(packed_pairs):
                    low_w = K.local_scalar(
                        K.f32, init=_cvt_to_f32(w_bits[pair * 2], input_dtype), name="low_w"
                    )
                    high_w = K.local_scalar(K.f32, init=undefined_f32[0], name="high_w")
                    if pair * 2 + 1 < total_values:
                        K.assign(high_w, _cvt_to_f32(w_bits[pair * 2 + 1], input_dtype))
                    K.ptx.mov.b64(w_f32_pairs[pair], low_w, high_w)
            else:
                for pair in range(packed_pairs):
                    low_bits = K.alloc_local((1,), "uint16")
                    high_bits = K.alloc_local((1,), "uint16")
                    K.ptx.mov.b32(low_bits[0], high_bits[0], w_words[pair])
                    low_w = K.local_scalar(
                        K.f32, init=_cvt_to_f32(low_bits[0], input_dtype), name="low_w"
                    )
                    high_w = K.local_scalar(
                        K.f32, init=_cvt_to_f32(high_bits[0], input_dtype), name="high_w"
                    )
                    K.ptx.mov.b64(w_f32_pairs[pair], low_w, high_w)

        if total_values == 1:
            K.assign(x_f32_scalar[0], _mul_f32(x_f32_scalar[0], rstd))
            K.assign(w_f32_scalar[0], _add_f32(w_f32_scalar[0], K.float32(0.0)))
            K.assign(x_f32_scalar[0], _mul_f32(x_f32_scalar[0], w_f32_scalar[0]))
            K.assign(x_f32_scalar[0], _mul_f32(x_f32_scalar[0], inv_scale))
        else:
            for pair in range(packed_pairs):
                high_scale = K.local_scalar(K.f32, init=undefined_f32[0], name="high_scale")
                if pair * 2 + 1 < total_values:
                    K.assign(high_scale, rstd)
                packed = K.alloc_local((1,), "uint64")
                K.ptx.mul.f32x2(packed[0], x_f32_pairs[pair], K.cuda.make_float2(rstd, high_scale))
                K.assign(x_f32_pairs[pair], packed[0])

            for pair in range(packed_pairs):
                high_bias = K.local_scalar(K.f32, init=undefined_f32[0], name="high_bias")
                if pair * 2 + 1 < total_values:
                    K.assign(high_bias, K.float32(0.0))
                packed = K.alloc_local((1,), "uint64")
                K.ptx.add.f32x2(
                    packed[0], w_f32_pairs[pair], K.cuda.make_float2(K.float32(0.0), high_bias)
                )
                K.assign(w_f32_pairs[pair], packed[0])

            for pair in range(packed_pairs):
                packed = K.alloc_local((1,), "uint64")
                K.ptx.mul.f32x2(packed[0], x_f32_pairs[pair], w_f32_pairs[pair])
                K.assign(x_f32_pairs[pair], packed[0])

            for pair in range(packed_pairs):
                high_inv_scale = K.local_scalar(K.f32, init=undefined_f32[0], name="high_inv_scale")
                if pair * 2 + 1 < total_values:
                    K.assign(high_inv_scale, inv_scale)
                packed = K.alloc_local((1,), "uint64")
                K.ptx.mul.f32x2(
                    packed[0], x_f32_pairs[pair], K.cuda.make_float2(inv_scale, high_inv_scale)
                )
                K.assign(x_f32_pairs[pair], packed[0])

        y_f32 = K.alloc_local((total_values,), "float32")
        if total_values == 1:
            K.assign(y_f32[0], x_f32_scalar[0])
        else:
            for pair in range(packed_pairs):
                if pair * 2 + 1 < total_values:
                    K.ptx.mov.b64(y_f32[pair * 2], y_f32[pair * 2 + 1], x_f32_pairs[pair])
                else:
                    discarded = K.alloc_local((1,), "float32")
                    K.ptx.mov.b64(y_f32[pair * 2], discarded[0], x_f32_pairs[pair])

        col_offset = K.local_scalar(K.i32, init=thread_in_row * vec, name="col_offset")
        for vb in range(vec_blocks):
            local_col = K.local_scalar(K.i32, init=col_offset + vb * tpr * vec, name="local_col")
            absolute_col = K.local_scalar(
                K.i32, init=block_y * cols + local_col, name="absolute_col"
            )
            if compact:
                y_offset = K.local_scalar(
                    K.i64,
                    init=row_i64 * K.int64(H) + K.cast(absolute_col, "int64"),
                    name="y_offset",
                )
            else:
                y_offset = K.local_scalar(
                    K.i64, init=row_i64 * y_stride + K.cast(absolute_col, "int64"), name="y_offset"
                )

            row_store_guard = None if rows == 1 else row_i64 < runtime_M

            def store_scalars():
                for element in range(vec):
                    scalar_col = K.local_scalar(
                        K.i32, init=absolute_col + element, name="scalar_col"
                    )
                    if compact:
                        scalar_offset = K.local_scalar(
                            K.i64,
                            init=row_i64 * K.int64(H) + K.cast(scalar_col, "int64"),
                            name="scalar_offset",
                        )
                    else:
                        scalar_offset = K.local_scalar(
                            K.i64,
                            init=row_i64 * y_stride + K.cast(scalar_col, "int64"),
                            name="scalar_offset",
                        )
                    col_guard = scalar_col < H
                    scalar_guard = (
                        col_guard if row_store_guard is None else K.And(col_guard, row_store_guard)
                    )
                    with K.If(scalar_guard), K.Then():
                        clamped_low = K.local_scalar(
                            K.f32,
                            init=_maximum_f32(y_f32[vb * vec + element], K.float32(-fp8_max)),
                            name="clamped_low",
                        )
                        clamped = K.local_scalar(
                            K.f32,
                            init=_minimum_f32(clamped_low, K.float32(fp8_max)),
                            name="clamped",
                        )
                        pair = K.local_scalar(
                            K.u16,
                            init=_cvt_fp8_pair(clamped, K.float32(0.0), output_dtype),
                            name="pair",
                        )
                        K.ptx.st.global_.b8(out.ptr_to([scalar_offset]), K.cast(pair, "uint8"))

            if vec == 8:

                def store_vec8():
                    p01 = K.local_scalar(
                        K.u16,
                        init=_cvt_fp8_pair(y_f32[vb * 8], y_f32[vb * 8 + 1], output_dtype),
                        name="p01",
                    )
                    p23 = K.local_scalar(
                        K.u16,
                        init=_cvt_fp8_pair(y_f32[vb * 8 + 2], y_f32[vb * 8 + 3], output_dtype),
                        name="p23",
                    )
                    p45 = K.local_scalar(
                        K.u16,
                        init=_cvt_fp8_pair(y_f32[vb * 8 + 4], y_f32[vb * 8 + 5], output_dtype),
                        name="p45",
                    )
                    p67 = K.local_scalar(
                        K.u16,
                        init=_cvt_fp8_pair(y_f32[vb * 8 + 6], y_f32[vb * 8 + 7], output_dtype),
                        name="p67",
                    )
                    lo_word = K.local_scalar(K.u32, init=_pack_b16_pair(p01, p23), name="lo_word")
                    hi_word = K.local_scalar(K.u32, init=_pack_b16_pair(p45, p67), name="hi_word")
                    K.ptx.st.global_.v2.b32(out.ptr_to([y_offset]), lo_word, hi_word)

                if full_columns:
                    with _runtime_guard(row_store_guard):
                        store_vec8()
                else:
                    vector_guard = absolute_col + 8 <= H
                    if row_store_guard is not None:
                        vector_guard = K.And(vector_guard, row_store_guard)
                    with K.If(vector_guard):
                        with K.Then():
                            store_vec8()
                        with K.Else():
                            store_scalars()
            elif vec == 4:
                vector_guard = absolute_col + 4 <= H
                if row_store_guard is not None:
                    vector_guard = K.And(vector_guard, row_store_guard)
                with K.If(vector_guard):
                    with K.Then():
                        p01 = _cvt_fp8_pair(y_f32[vb * 4], y_f32[vb * 4 + 1], output_dtype)
                        p23 = _cvt_fp8_pair(y_f32[vb * 4 + 2], y_f32[vb * 4 + 3], output_dtype)
                        packed_word = K.local_scalar(
                            K.u32, init=_pack_b16_pair(p01, p23), name="packed_word"
                        )
                        K.ptx.st.global_.b32(out.ptr_to([y_offset]), packed_word)
                    with K.Else():
                        store_scalars()
            elif vec == 2:
                vector_guard = absolute_col + 2 <= H
                if row_store_guard is not None:
                    vector_guard = K.And(vector_guard, row_store_guard)
                with K.If(vector_guard):
                    with K.Then():
                        p01 = _cvt_fp8_pair(y_f32[vb * 2], y_f32[vb * 2 + 1], output_dtype)
                        K.ptx.st.global_.b16(out.ptr_to([y_offset]), p01)
                    with K.Else():
                        store_scalars()
            else:
                store_scalars()

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
        def flashinfer_rmsnorm_quant_compact(
            x: K.gptr[input_dtype],
            weight: K.gptr[input_dtype, (H,)],
            out: K.gptr[output_dtype],
            runtime_M: K.i64,
            scale_buffer: K.gptr[K.f32, (1,)],
            runtime_eps: K.f32,
        ):
            with entry_registers():
                kernel_body(
                    x, weight, out, runtime_M, scale_buffer, runtime_eps, K.int64(H), K.int64(H)
                )

        kernel = flashinfer_rmsnorm_quant_compact.func
    else:

        @K.kernel(**entry_kwargs)
        def flashinfer_rmsnorm_quant_strided(
            x: K.gptr[input_dtype],
            weight: K.gptr[input_dtype, (H,)],
            out: K.gptr[output_dtype],
            runtime_M: K.i64,
            scale_buffer: K.gptr[K.f32, (1,)],
            runtime_eps: K.f32,
            runtime_x_row_stride: K.i64,
            runtime_y_row_stride: K.i64,
        ):
            with entry_registers():
                kernel_body(
                    x,
                    weight,
                    out,
                    runtime_M,
                    scale_buffer,
                    runtime_eps,
                    runtime_x_row_stride,
                    runtime_y_row_stride,
                )

        kernel = flashinfer_rmsnorm_quant_strided.func

    launch_params = ["blockIdx.x"]
    if cluster_n > 1:
        launch_params.extend(["blockIdx.y", "clusterCtaIdx.x", "clusterCtaIdx.y"])
    launch_params.append("threadIdx.x")
    if enable_pdl:
        launch_params.append("tirx.use_programtic_dependent_launch")
    launch_params.append("tirx.use_dyn_shared_memory")
    return kernel.with_attr("tirx.kernel_launch_params", launch_params)


def prepare_data(**config: Any):
    """Create deterministic tensors, including the caller-owned FP8 output."""
    data = _prepare_tensors(dict(config))
    output = _prepare_output(
        int(config["M"]),
        int(config["H"]),
        data["y_row_stride"],
        str(config["output_dtype"]),
        initialize_padding=False,
    )
    return data["x"], data["weight"], output["view"], data["scale"]


_GUARD_ELEMENTS = 64
_GUARD_VALUE = 1.0


def _torch_input_dtype(dtype: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def _torch_output_dtype(dtype: str):
    import torch

    return {"float8_e4m3fn": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2}[dtype]


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
    input_dtype = str(config["input_dtype"])
    output_dtype = str(config["output_dtype"])
    input_layout = str(config["input_layout"])
    output_layout = str(config["output_layout"])
    scale_value = float(config["scale"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    x_stride, y_stride = _row_strides(config)
    _validate(
        input_dtype,
        output_dtype,
        M,
        H,
        input_layout,
        output_layout,
        scale_value,
        eps,
        x_stride,
        y_stride,
    )
    vec = int(_source_config(H)["vec"])
    if x_stride < H or y_stride < H:
        raise ValueError(f"row strides must cover H={H}: x={x_stride}, y={y_stride}")
    if x_stride % vec != 0 or y_stride % vec != 0:
        raise ValueError(f"row strides must be divisible by vec={vec}: x={x_stride}, y={y_stride}")

    torch_dtype = _torch_input_dtype(input_dtype)
    generator = torch.Generator(device="cuda").manual_seed(42)
    x_storage_size = _storage_size(M, H, x_stride)
    x_backing = torch.empty(x_storage_size + _GUARD_ELEMENTS, dtype=torch_dtype, device="cuda")
    x_arg = x_backing[:x_storage_size]
    x = x_arg.as_strided((M, H), (x_stride, 1))
    if H >= 65536:
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
    if H >= 3:
        dtype_limit = torch.finfo(torch_dtype).max
        weight[0] = dtype_limit
        weight[1] = torch.tensor(1.0546875 * scale_value, dtype=torch_dtype, device="cuda")
        weight[2] = torch.tensor(1.0703125 * scale_value, dtype=torch_dtype, device="cuda")
    weight_backing[H:].fill_(_GUARD_VALUE)

    scale = torch.tensor([scale_value], dtype=torch.float32, device="cuda")
    return {
        "x": x,
        "x_arg": x_arg,
        "x_backing": x_backing,
        "x_storage_size": x_storage_size,
        "weight": weight,
        "weight_backing": weight_backing,
        "scale": scale,
        "x_row_stride": x_stride,
        "y_row_stride": y_stride,
    }


def _prepare_output(
    M: int, H: int, row_stride: int, output_dtype: str, *, initialize_padding: bool
) -> dict[str, Any]:
    import torch

    size = _storage_size(M, H, row_stride)
    backing = torch.empty(
        size + _GUARD_ELEMENTS, dtype=_torch_output_dtype(output_dtype), device="cuda"
    )
    if initialize_padding:
        backing.fill_(_GUARD_VALUE)
    else:
        backing[size:].fill_(_GUARD_VALUE)
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


def _assert_output_padding(
    output: dict[str, Any], M: int, H: int, row_stride: int, *, name: str
) -> None:
    import torch

    _assert_guard(output["backing"], output["size"], name=name)
    if row_stride > H:
        for row in range(M - 1):
            padding = output["backing"][row * row_stride + H : (row + 1) * row_stride]
            expected = torch.full_like(padding, _GUARD_VALUE)
            if not torch.equal(_raw_bytes(padding), _raw_bytes(expected)):
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


def _assert_raw_equal(actual, expected, *, name: str) -> None:
    import torch

    if not torch.equal(_raw_bytes(actual), _raw_bytes(expected)):
        mismatch = _raw_bytes(actual) != _raw_bytes(expected)
        count = int(mismatch.sum().item())
        raise AssertionError(f"{name}: {count} FP8 bytes differ")


def _flashinfer_api(device):
    import flashinfer.norm as flashinfer_norm

    if flashinfer_norm._use_cuda_norm(device):
        raise AssertionError("FlashInfer RMSNormQuant oracle dispatched to legacy CUDA")
    return flashinfer_norm.rmsnorm_quant, flashinfer_norm


def _launch_tirx(executable, data, output, config: dict[str, Any]):
    M = int(config["M"])
    H = int(config["H"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    compact = _uses_compact_specialization(
        M, H, str(config["input_layout"]), str(config["output_layout"])
    )
    if compact:
        return executable(data["x_arg"], data["weight"], output["arg"], M, data["scale"], eps)
    return executable(
        data["x_arg"],
        data["weight"],
        output["arg"],
        M,
        data["scale"],
        eps,
        data["x_row_stride"],
        data["y_row_stride"],
    )


@functools.cache
def _compiled_test_specialization(
    input_dtype: str, output_dtype: str, H: int, compact: bool, enable_pdl: bool
):
    """Compile each runtime-M test specialization once per test process."""
    from tirx_kernels.runner import compile_kernel

    layout = "compact" if compact else "strided"
    return compile_kernel(
        get_kernel(
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            M=1,
            H=H,
            input_layout=layout,
            output_layout=layout,
            enable_pdl=enable_pdl,
            scale=1.0,
            eps=_DEFAULT_EPS,
            x_row_stride=H,
            y_row_stride=H,
        )
    )


def _input_snapshot(data, M: int, H: int) -> dict[str, Any]:
    rows = _overflow_rows(M, H)
    if rows is None:
        values = data["x"].clone()
    else:
        columns = sorted({0, H // 2, H - 1})
        values = data["x"][rows][:, columns].clone()
    return {
        "rows": rows,
        "values": values,
        "weight": data["weight"].clone(),
        "scale": data["scale"].clone(),
    }


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
    if not torch.equal(data["scale"], snapshot["scale"]):
        raise AssertionError("scale tensor was modified")
    _assert_guard(data["x_backing"], data["x_storage_size"], name="input")
    _assert_guard(data["weight_backing"], H, name="weight")


def _assert_output_identity(output: dict[str, Any], *, name: str) -> None:
    if output["view"].data_ptr() != output["data_ptr"]:
        raise AssertionError(f"{name} data pointer changed")
    if output["view"].stride() != output["stride"]:
        raise AssertionError(f"{name} stride changed")


def run_test(**config: Any) -> None:
    """Compile, launch, and validate one RMSNorm-quant specialization."""
    import torch

    config = dict(config)
    M = int(config["M"])
    H = int(config["H"])
    output_dtype = str(config["output_dtype"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    enable_pdl = bool(config["enable_pdl"])
    data = _prepare_tensors(config)
    snapshot = _input_snapshot(data, M, H)
    output = _prepare_output(M, H, data["y_row_stride"], output_dtype, initialize_padding=True)
    reference_output = _prepare_output(
        M, H, data["y_row_stride"], output_dtype, initialize_padding=True
    )

    compact = _uses_compact_specialization(
        M, H, str(config["input_layout"]), str(config["output_layout"])
    )
    executable = _compiled_test_specialization(
        str(config["input_dtype"]), output_dtype, H, compact, enable_pdl
    )
    returned = _launch_tirx(executable, data, output, config)
    if returned is not None:
        raise AssertionError("TIRx RMSNormQuant ABI must return None")

    api, flashinfer_norm = _flashinfer_api(data["x"].device)
    original_cute = flashinfer_norm.rmsnorm_quant_cute
    cute_calls = 0

    def tracked_cute(*args, **kwargs):
        nonlocal cute_calls
        cute_calls += 1
        return original_cute(*args, **kwargs)

    flashinfer_norm.rmsnorm_quant_cute = tracked_cute
    try:
        reference_returned = api(
            reference_output["view"],
            data["x"],
            data["weight"],
            data["scale"],
            eps,
            enable_pdl=enable_pdl,
        )
    finally:
        flashinfer_norm.rmsnorm_quant_cute = original_cute
    if cute_calls != 1:
        raise AssertionError(f"expected one CuTe-DSL oracle dispatch, observed {cute_calls}")
    if reference_returned is not None:
        raise AssertionError("FlashInfer rmsnorm_quant must return None")

    torch.cuda.synchronize()
    rows = _overflow_rows(M, H)
    actual_checked = _checked_view(output["view"], rows)
    reference_checked = _checked_view(reference_output["view"], rows)
    if not torch.isfinite(actual_checked.float()).all():
        raise AssertionError("TIRx FP8 output contains non-finite values")
    _assert_raw_equal(actual_checked, reference_checked, name="FlashInfer raw-byte oracle")
    _assert_inputs_unchanged(data, snapshot, M, H)
    _assert_output_identity(output, name="TIRx output")
    _assert_output_identity(reference_output, name="FlashInfer output")
    _assert_output_padding(output, M, H, data["y_row_stride"], name="TIRx output")
    _assert_output_padding(reference_output, M, H, data["y_row_stride"], name="FlashInfer output")


def prepare_bench(**config: Any):
    """Compile the specialization before the bench suite assigns a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(config), "executable": compile_kernel(get_kernel(**config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs: Any
):
    """Build two single-launch closures and validate them before timing."""
    import torch

    config = dict(prepared["config"])
    config.update(kwargs)
    M = int(config["M"])
    H = int(config["H"])
    output_dtype = str(config["output_dtype"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    enable_pdl = bool(config["enable_pdl"])
    data = _prepare_tensors(config)
    tirx_output = _prepare_output(
        M, H, data["y_row_stride"], output_dtype, initialize_padding=False
    )
    flashinfer_output = _prepare_output(
        M, H, data["y_row_stride"], output_dtype, initialize_padding=False
    )
    executable = prepared["executable"]

    def tirx_launch():
        return _launch_tirx(executable, data, tirx_output, config)

    if tirx_launch() is not None:
        raise AssertionError("TIRx benchmark closure must return None")
    torch.cuda.synchronize()

    def build_flashinfer_reference():
        api, flashinfer_norm = _flashinfer_api(data["x"].device)
        original_cute = flashinfer_norm.rmsnorm_quant_cute
        cute_calls = 0

        def tracked_cute(*args, **kwargs):
            nonlocal cute_calls
            cute_calls += 1
            return original_cute(*args, **kwargs)

        def flashinfer_launch():
            return api(
                flashinfer_output["view"],
                data["x"],
                data["weight"],
                data["scale"],
                eps,
                enable_pdl=enable_pdl,
            )

        flashinfer_norm.rmsnorm_quant_cute = tracked_cute
        try:
            returned = flashinfer_launch()
        finally:
            flashinfer_norm.rmsnorm_quant_cute = original_cute
        if cute_calls != 1:
            raise AssertionError(f"expected one CuTe-DSL benchmark warmup, observed {cute_calls}")
        if returned is not None:
            raise AssertionError("FlashInfer benchmark closure must return None")
        torch.cuda.synchronize()
        _assert_raw_equal(
            tirx_output["view"], flashinfer_output["view"], name="benchmark raw-byte precheck"
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
    """Benchmark one specialization against the lazy FlashInfer CuTe-DSL reference."""
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
