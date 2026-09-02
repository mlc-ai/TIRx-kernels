# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400e330fb2debe0bf8730d9424a1d37927f), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer CuTe-DSL LayerNorm port.

The source implementation is ``LayerNormKernel`` in
``flashinfer/norm/kernels/layernorm.py`` together with the public dispatch in
``flashinfer/norm/__init__.py`` and reduction helpers in
``flashinfer/norm/utils.py``.
"""

from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.runner import bench

KERNEL_META = {
    "name": "flashinfer_layernorm",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a"],
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
_DEFAULT_EPS = 1e-6
_LAYOUTS = ("compact", "strided")
_GUARD_ELEMENTS = 64
_GUARD_VALUE = 123.0


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _source_vec(H: int) -> int:
    for vec in (8, 4, 2, 1):
        if H % vec == 0 and H // vec >= 32:
            return vec
    return 8 if H % 8 == 0 else 4 if H % 4 == 0 else 2 if H % 2 == 0 else 1


def _source_config(H: int) -> dict[str, int | bool]:
    vec = _source_vec(H)
    threads = 32
    while threads < _ceil_div(H, vec) and threads < 1024:
        threads *= 2
    threads = min(threads, 1024)
    warps = max(threads // 32, 1)
    vec_blocks = max(1, _ceil_div(H // vec, threads))
    cols = vec * vec_blocks * threads
    total_values = vec * vec_blocks
    return {
        "vec": vec,
        "threads": threads,
        "warps": warps,
        "vec_blocks": vec_blocks,
        "cols": cols,
        "total_values": total_values,
        "smem_bytes": 2 * warps * 4,
        "mixed_local_sum": total_values > 1 and (cols == H or total_values > 8),
    }


def _ptx_unary(chain: str, value, dtype: str = "float32"):
    out = K.local_scalar(dtype)
    K.ptx[chain](out, value)
    return out


def _ptx_binary(chain: str, lhs, rhs, dtype: str = "float32"):
    out = K.local_scalar(dtype)
    K.ptx[chain](out, lhs, rhs)
    return out


def _ptx_ternary(chain: str, lhs, rhs, acc, dtype: str = "float32"):
    out = K.local_scalar(dtype)
    K.ptx[chain](out, lhs, rhs, acc)
    return out


def _add_f32(lhs, rhs):
    return _ptx_binary("add.f32", lhs, rhs)


def _add_bf16_to_f32(bits, value):
    out = K.local_scalar(K.f32)
    K.ptx.add.rn.f32.bf16(out, K.cast(bits, K.u16), value)
    return out


def _sub_f32(lhs, rhs):
    return _ptx_binary("sub.f32", lhs, rhs)


def _mul_f32(lhs, rhs):
    return _ptx_binary("mul.f32", lhs, rhs)


def _div_rn_f32(lhs, rhs):
    return _ptx_binary("div.rn.f32", lhs, rhs)


def _fma_rn_f32(lhs, rhs, acc):
    return _ptx_ternary("fma.rn.f32", lhs, rhs, acc)


def _rsqrt_approx_ftz(value):
    return _ptx_unary("rsqrt.approx.ftz.f32", value)


def _cvt_bf16_to_f32(bits):
    return _ptx_unary("cvt.f32.bf16", K.cast(bits, K.u16))


def _cvt_f32_to_bf16(value):
    return _ptx_unary("cvt.rn.bf16.f32", value, dtype="uint16")


def _cvt_pair_f32_to_bf16(high, low):
    return _ptx_binary("cvt.rn.bf16x2.f32", high, low, dtype="uint32")


def _shfl_bfly_f32(value, lane_xor: int):
    out = K.local_scalar(K.u32)
    K.ptx.shfl_sync.bfly.b32(
        out, K.reinterpret(K.u32, value), K.uint32(lane_xor), K.uint32(31), K.uint32(0xFFFFFFFF)
    )
    return K.reinterpret(K.f32, out)


def _butterfly_sum_f32(value):
    for lane_xor in (1, 2, 4, 8, 16):
        value = _add_f32(value, _shfl_bfly_f32(value, lane_xor))
    return value


def _load_x(buffer, index, values, value_offset, VEC: int):
    if VEC == 1:
        K.ptx.ld.global_.b16(values[value_offset], buffer.ptr_to([index]))
    elif VEC == 2:
        K.ptx.ld.global_.v2.b16(
            values[value_offset], values[value_offset + 1], buffer.ptr_to([index])
        )
    elif VEC == 4:
        K.ptx.ld.global_.v4.b16(
            values[value_offset],
            values[value_offset + 1],
            values[value_offset + 2],
            values[value_offset + 3],
            buffer.ptr_to([index]),
        )
    else:
        words = K.alloc_local([4], K.u32)
        K.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
        for pair in range(4):
            K.ptx.mov.b32(
                values[value_offset + pair * 2], values[value_offset + pair * 2 + 1], words[pair]
            )


def _store_y(buffer, index, bits, words, value_offset, word_offset, VEC: int):
    if VEC == 1:
        K.ptx.st.global_.b16(buffer.ptr_to([index]), bits[value_offset])
    elif VEC == 2:
        K.ptx.st.global_.b32(buffer.ptr_to([index]), words[word_offset])
    elif VEC == 4:
        K.ptx.st.global_.v2.b32(buffer.ptr_to([index]), words[word_offset], words[word_offset + 1])
    else:
        K.ptx.st.global_.v4.b32(
            buffer.ptr_to([index]),
            words[word_offset],
            words[word_offset + 1],
            words[word_offset + 2],
            words[word_offset + 3],
        )


def _short_layout(layout: str) -> str:
    return "c" if layout == "compact" else "s"


def _cfg(
    M: int,
    H: int,
    input_layout: str = "compact",
    output_layout: str = "compact",
    enable_pdl: bool = False,
    *,
    eps: float = _DEFAULT_EPS,
    x_row_stride: int | None = None,
    y_row_stride: int | None = None,
    suffix: str = "",
) -> dict[str, Any]:
    eps_label = "eps1e6" if eps == _DEFAULT_EPS else "eps1e4"
    label = (
        f"bf16_m{M}_h{H}_x{_short_layout(input_layout)}_y{_short_layout(output_layout)}_"
        f"pdl{int(enable_pdl)}_{eps_label}"
    )
    if suffix:
        label += f"_{suffix}"
    config: dict[str, Any] = {
        "label": label,
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


_UPSTREAM_CONFIGS = [_cfg(M, H) for M in (1, 2, 3, 128) for H in (128, 129, 1024, 16384)]
BENCH_CONFIGS = [dict(config) for config in _UPSTREAM_CONFIGS]
_TRACE_CONFIGS = [
    _cfg(8, 256, suffix="trace"),
    _cfg(3, 320, suffix="trace"),
    _cfg(32, 768, suffix="public_example"),
]
_OVERFLOW_CONFIGS = [_cfg(175000, 12288, suffix="i64_overflow")]
_STRUCTURE_CONFIGS = [
    _cfg(3, 66, enable_pdl=True, suffix="vec2_tail"),
    _cfg(3, 1025, suffix="vec1_two_blocks"),
    _cfg(
        19,
        500,
        "strided",
        "strided",
        eps=1e-4,
        x_row_stride=1000,
        y_row_stride=1500,
        suffix="full_abi",
    ),
]
CONFIGS = [*_UPSTREAM_CONFIGS, *_TRACE_CONFIGS, *_OVERFLOW_CONFIGS, *_STRUCTURE_CONFIGS]

assert len(CONFIGS) == 23
assert len(BENCH_CONFIGS) == 16
assert len({config["label"] for config in CONFIGS}) == len(CONFIGS)


def _validate(M: int, H: int, input_layout: str, output_layout: str, eps: float) -> None:
    if M <= 0 or H <= 0:
        raise ValueError(f"M and H must be positive, got M={M}, H={H}")
    if input_layout not in _LAYOUTS or output_layout not in _LAYOUTS:
        raise ValueError(f"unsupported layouts: input={input_layout}, output={output_layout}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")


def get_kernel(
    M: int,
    H: int,
    input_layout: str,
    output_layout: str,
    enable_pdl: bool,
    eps: float = _DEFAULT_EPS,
    **kwargs: Any,
):
    """Return the source-faithful dynamic-Int64-stride specialization."""
    _validate(M, H, input_layout, output_layout, eps)
    source = _source_config(H)
    vec = int(source["vec"])
    threads = int(source["threads"])
    warps = int(source["warps"])
    vec_blocks = int(source["vec_blocks"])
    total_values = int(source["total_values"])
    smem_bytes = int(source["smem_bytes"])
    mixed_local_sum = bool(source["mixed_local_sum"])
    packed_pairs = _ceil_div(total_values, 2)
    pair_values = packed_pairs * 2

    x_stride_hint = int(kwargs.get("x_row_stride", H if input_layout == "compact" else 2 * H))
    y_stride_hint = int(kwargs.get("y_row_stride", H if output_layout == "compact" else 2 * H))
    if x_stride_hint < H or y_stride_hint < H:
        raise ValueError(f"row strides must cover H={H}: x={x_stride_hint}, y={y_stride_hint}")
    if x_stride_hint % vec != 0 or y_stride_hint % vec != 0:
        raise ValueError(
            f"row strides must be divisible by vec={vec}: x={x_stride_hint}, y={y_stride_hint}"
        )

    @K.kernel(
        warps=warps,
        arch="sm_100a",
        # ``I.cta_id`` owns an int32 block axis; preserve the parser version's
        # explicit runtime-M cast instead of passing the int64 ABI scalar as
        # the extent directly.
        grid=lambda p: K.cast(p["runtime_M"], K.i32),
    )
    def flashinfer_layernorm(
        out: K.gptr[K.bf16],
        x: K.gptr[K.bf16],
        gamma: K.gptr[K.f32],
        beta: K.gptr[K.f32],
        runtime_M: K.i64,
        runtime_eps: K.f32,
        y_row_stride: K.i64,
        x_row_stride: K.i64,
    ):
        row_raw = K.cta_id()
        tid = K.thread_id()
        row = K.cast(row_raw, K.i64)
        lane = tid % 32
        warp = tid // 32

        K.attr({"tirx.dyn_smem_bytes": smem_bytes})
        if warps > 1:
            shared_raw = K.alloc_buffer([smem_bytes], K.u8, scope="shared.dyn", align=1024)
        if enable_pdl:
            K.ptx.griddepcontrol.wait()

        x_bits = K.alloc_local([pair_values], K.u16)
        x_f32 = K.alloc_local([pair_values], K.f32)
        for value in range(total_values):
            K.assign(x_bits[value], K.uint16(0))
        for vb in range(vec_blocks):
            col = (tid + vb * threads) * vec
            x_offset = row * x_row_stride + K.cast(col, K.i64)
            with K.If(col < H), K.Then():
                _load_x(x, x_offset, x_bits, vb * vec, VEC=vec)
        for value in range(total_values):
            K.assign(x_f32[value], _cvt_bf16_to_f32(x_bits[value]))

        local_sum = K.local_scalar(K.f32, init=_add_f32(x_f32[0], K.float32(0.0)))
        if total_values > 1:
            for step in range(total_values - 1):
                value = step + 1
                if mixed_local_sum:
                    K.assign(local_sum, _add_bf16_to_f32(x_bits[value], local_sum))
                else:
                    K.assign(local_sum, _add_f32(local_sum, x_f32[value]))
        warp_sum = _butterfly_sum_f32(local_sum)
        if warps > 1:
            with K.If(lane == 0), K.Then():
                K.ptx.st.shared.b32(shared_raw.ptr_to([warp * 4]), K.reinterpret(K.u32, warp_sum))
            K.ptx.bar.sync(K.uint32(0))
            block_sum = K.local_scalar(K.f32, init=K.float32(0.0))
            with K.If(lane < warps), K.Then():
                sum_word = K.local_scalar(K.u32)
                K.ptx.ld.shared.b32(sum_word, shared_raw.ptr_to([lane * 4]))
                K.assign(block_sum, K.reinterpret(K.f32, sum_word))
            sum_x = _butterfly_sum_f32(block_sum)
        else:
            sum_x = warp_sum

        if H & (H - 1) == 0:
            mean = _mul_f32(sum_x, K.float32(1.0 / H))
        elif total_values > 1:
            mean = _div_rn_f32(sum_x, K.float32(H))
        else:
            negative_mean = _div_rn_f32(sum_x, K.float32(-H))

        diff = K.alloc_local([pair_values], K.f32)
        diff_sq = K.alloc_local([pair_values], K.f32)
        if total_values == 1:
            if H & (H - 1) == 0:
                K.assign(diff[0], _sub_f32(x_f32[0], mean))
            else:
                K.assign(diff[0], _add_f32(x_f32[0], negative_mean))
            K.assign(diff_sq[0], _mul_f32(diff[0], diff[0]))
        else:
            for pair in range(packed_pairs):
                packed = K.local_scalar(K.u64)
                K.ptx.sub.f32x2(
                    packed,
                    K.cuda.make_float2(x_f32[pair * 2], x_f32[pair * 2 + 1]),
                    K.cuda.make_float2(mean, mean),
                )
                K.ptx.mov.b64(diff[pair * 2], diff[pair * 2 + 1], packed)
            for pair in range(packed_pairs):
                packed = K.local_scalar(K.u64)
                diff_pair = K.cuda.make_float2(diff[pair * 2], diff[pair * 2 + 1])
                K.ptx.mul.f32x2(packed, diff_pair, diff_pair)
                K.ptx.mov.b64(diff_sq[pair * 2], diff_sq[pair * 2 + 1], packed)

        for vb in range(vec_blocks):
            col = (tid + vb * threads) * vec
            with K.If(col >= H), K.Then():
                for e in range(vec):
                    K.assign(diff_sq[vb * vec + e], K.float32(0.0))

        local_var = K.local_scalar(K.f32, init=_add_f32(diff_sq[0], K.float32(0.0)))
        if total_values > 1:
            for step in range(total_values - 1):
                value = step + 1
                K.assign(local_var, _add_f32(local_var, diff_sq[value]))
        warp_var = _butterfly_sum_f32(local_var)
        if warps > 1:
            with K.If(lane == 0), K.Then():
                K.ptx.st.shared.b32(
                    shared_raw.ptr_to([warps * 4 + warp * 4]), K.reinterpret(K.u32, warp_var)
                )
            K.ptx.bar.sync(K.uint32(0))
            block_var = K.local_scalar(K.f32, init=K.float32(0.0))
            with K.If(lane < warps), K.Then():
                var_word = K.local_scalar(K.u32)
                K.ptx.ld.shared.b32(var_word, shared_raw.ptr_to([warps * 4 + lane * 4]))
                K.assign(block_var, K.reinterpret(K.f32, var_word))
            sum_diff_sq = _butterfly_sum_f32(block_var)
        else:
            sum_diff_sq = warp_var
        if H & (H - 1) == 0:
            shifted_var = _fma_rn_f32(sum_diff_sq, K.float32(1.0 / H), runtime_eps)
        else:
            variance = _div_rn_f32(sum_diff_sq, K.float32(H))
            shifted_var = _add_f32(variance, runtime_eps)
        rstd = _rsqrt_approx_ftz(shifted_var)
        K.ptx.bar.sync(K.uint32(0))

        gamma_frag = K.alloc_local([pair_values], K.f32)
        beta_frag = K.alloc_local([pair_values], K.f32)
        for value in range(total_values):
            K.assign(gamma_frag[value], K.float32(0.0))
            K.assign(beta_frag[value], K.float32(0.0))
        for vb in range(vec_blocks):
            for e in range(vec):
                value = vb * vec + e
                col = tid * vec + vb * threads * vec + e
                with K.If(col < H), K.Then():
                    K.ptx.ld.global_.b32(gamma_frag[value], gamma.ptr_to([col]))
                    K.ptx.ld.global_.b32(beta_frag[value], beta.ptr_to([col]))

        y_f32 = K.alloc_local([pair_values], K.f32)
        if total_values == 1:
            normalized = _mul_f32(diff[0], rstd)
            K.assign(y_f32[0], _fma_rn_f32(normalized, gamma_frag[0], beta_frag[0]))
        else:
            normalized = K.alloc_local([pair_values], K.f32)
            scaled = K.alloc_local([pair_values], K.f32)
            for pair in range(packed_pairs):
                packed = K.local_scalar(K.u64)
                K.ptx.mul.f32x2(
                    packed,
                    K.cuda.make_float2(diff[pair * 2], diff[pair * 2 + 1]),
                    K.cuda.make_float2(rstd, rstd),
                )
                K.ptx.mov.b64(normalized[pair * 2], normalized[pair * 2 + 1], packed)
            for pair in range(packed_pairs):
                packed = K.local_scalar(K.u64)
                K.ptx.mul.f32x2(
                    packed,
                    K.cuda.make_float2(normalized[pair * 2], normalized[pair * 2 + 1]),
                    K.cuda.make_float2(gamma_frag[pair * 2], gamma_frag[pair * 2 + 1]),
                )
                K.ptx.mov.b64(scaled[pair * 2], scaled[pair * 2 + 1], packed)
            for pair in range(packed_pairs):
                packed = K.local_scalar(K.u64)
                K.ptx.add.f32x2(
                    packed,
                    K.cuda.make_float2(scaled[pair * 2], scaled[pair * 2 + 1]),
                    K.cuda.make_float2(beta_frag[pair * 2], beta_frag[pair * 2 + 1]),
                )
                K.ptx.mov.b64(y_f32[pair * 2], y_f32[pair * 2 + 1], packed)

        y_bits = K.alloc_local([pair_values], K.u16)
        y_words = K.alloc_local([max(packed_pairs, 1)], K.u32)
        if vec == 1:
            for value in range(total_values):
                K.assign(y_bits[value], _cvt_f32_to_bf16(y_f32[value]))
        else:
            for pair in range(packed_pairs):
                K.assign(y_words[pair], _cvt_pair_f32_to_bf16(y_f32[pair * 2 + 1], y_f32[pair * 2]))
        for vb in range(vec_blocks):
            col = (tid + vb * threads) * vec
            y_offset = row * y_row_stride + K.cast(col, K.i64)
            with K.If(col < H), K.Then():
                _store_y(out, y_offset, y_bits, y_words, vb * vec, vb * vec // 2, VEC=vec)
        if enable_pdl:
            K.ptx.griddepcontrol.launch_dependents()

    launch_params = ["blockIdx.x", "threadIdx.x"]
    if enable_pdl:
        launch_params.append("tirx.use_programtic_dependent_launch")
    launch_params.append("tirx.use_dyn_shared_memory")
    return flashinfer_layernorm.func.with_attr("tirx.kernel_launch_params", launch_params)


def _row_strides(config: dict[str, Any]) -> tuple[int, int]:
    H = int(config["H"])
    x_stride = int(config.get("x_row_stride", H if config["input_layout"] == "compact" else 2 * H))
    y_stride = int(config.get("y_row_stride", H if config["output_layout"] == "compact" else 2 * H))
    return x_stride, y_stride


def _storage_size(M: int, H: int, row_stride: int) -> int:
    return (M - 1) * row_stride + H


def _prepare_tensors(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    M, H = int(config["M"]), int(config["H"])
    eps = float(config.get("eps", _DEFAULT_EPS))
    _validate(M, H, str(config["input_layout"]), str(config["output_layout"]), eps)
    x_stride, y_stride = _row_strides(config)
    vec = int(_source_config(H)["vec"])
    if x_stride < H or y_stride < H or x_stride % vec or y_stride % vec:
        raise ValueError(f"invalid row strides for H={H}, vec={vec}: x={x_stride}, y={y_stride}")

    x_size = _storage_size(M, H, x_stride)
    x_backing = torch.full(
        (x_size + _GUARD_ELEMENTS,), _GUARD_VALUE, dtype=torch.bfloat16, device="cuda"
    )
    x_arg = x_backing[:x_size]
    x = x_arg.as_strided((M, H), (x_stride, 1))
    if H >= 4096:
        columns = torch.arange(H, device="cuda")
        magnitude = torch.where(columns % 257 < 128, 0.5, 1.0)
        pattern = torch.where(columns % 2 == 0, magnitude, -magnitude).to(torch.bfloat16)
        x.copy_(pattern.expand(M, H))
        x[1::2].neg_()
    else:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(42)
        x.normal_(generator=generator)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(43)
    gamma_backing = torch.full(
        (H + _GUARD_ELEMENTS,), _GUARD_VALUE, dtype=torch.float32, device="cuda"
    )
    beta_backing = torch.full(
        (H + _GUARD_ELEMENTS,), _GUARD_VALUE, dtype=torch.float32, device="cuda"
    )
    gamma, beta = gamma_backing[:H], beta_backing[:H]
    gamma.normal_(generator=generator)
    beta.normal_(generator=generator)
    return {
        "x": x,
        "x_arg": x_arg,
        "x_backing": x_backing,
        "x_size": x_size,
        "gamma": gamma,
        "gamma_backing": gamma_backing,
        "beta": beta,
        "beta_backing": beta_backing,
        "x_row_stride": x_stride,
        "y_row_stride": y_stride,
    }


def prepare_data(**config: Any):
    """Create deterministic LayerNorm inputs."""
    data = _prepare_tensors(dict(config))
    return data["x"], data["gamma"], data["beta"]


def _prepare_output(M: int, H: int, row_stride: int, *, initialize_padding: bool):
    import torch

    size = _storage_size(M, H, row_stride)
    backing = torch.empty(size + _GUARD_ELEMENTS, dtype=torch.bfloat16, device="cuda")
    if initialize_padding:
        backing.fill_(_GUARD_VALUE)
    else:
        backing[size:].fill_(_GUARD_VALUE)
    arg = backing[:size]
    return {
        "view": arg.as_strided((M, H), (row_stride, 1)),
        "arg": arg,
        "backing": backing,
        "size": size,
    }


def _assert_guard(values, *, name: str) -> None:
    import torch

    if values.numel() and not torch.equal(values, torch.full_like(values, _GUARD_VALUE)):
        raise AssertionError(f"{name} padding or guard was modified")


def _assert_output_padding(output, M: int, H: int, row_stride: int, *, name: str) -> None:
    backing = output["backing"]
    _assert_guard(backing[output["size"] :], name=f"{name} terminal")
    if row_stride > H and M > 1:
        padding = backing.as_strided((M - 1, row_stride - H), (row_stride, 1), storage_offset=H)
        _assert_guard(padding, name=f"{name} row")


def _overflow_rows(M: int, H: int) -> list[int] | None:
    if M * H <= 2**31 - 1:
        return None
    boundary = _ceil_div(2**31, H)
    return sorted(
        {row for row in (0, 1, boundary - 1, boundary, boundary + 1, M - 1) if 0 <= row < M}
    )


def _checked_view(tensor, rows: list[int] | None):
    return tensor if rows is None else tensor[rows]


def _assert_close(actual, expected, *, name: str, rtol: float, atol: float) -> None:
    import torch

    torch.testing.assert_close(
        actual, expected, rtol=rtol, atol=atol, msg=lambda message: f"{name}: {message}"
    )


def _launch_tirx(executable, data, output, config: dict[str, Any]) -> None:
    executable(
        output["arg"],
        data["x_arg"],
        data["gamma"],
        data["beta"],
        int(config["M"]),
        float(config.get("eps", _DEFAULT_EPS)),
        data["y_row_stride"],
        data["x_row_stride"],
    )


def _flashinfer_cute(device):
    import flashinfer.norm as flashinfer_norm

    if flashinfer_norm._use_cuda_norm(device):
        raise AssertionError("FlashInfer LayerNorm oracle dispatched to legacy CUDA")
    return flashinfer_norm, flashinfer_norm.layernorm_cute


def _snapshot_inputs(data, M: int, H: int):
    rows = _overflow_rows(M, H)
    if rows is None:
        x_values = data["x"].clone()
    else:
        columns = sorted({0, H // 2, H - 1})
        x_values = data["x"][rows][:, columns].clone()
    return {
        "rows": rows,
        "x": x_values,
        "gamma": data["gamma"].clone(),
        "beta": data["beta"].clone(),
    }


def _assert_inputs_unchanged(data, snapshot, M: int, H: int) -> None:
    import torch

    rows = snapshot["rows"]
    if rows is None:
        observed = data["x"]
    else:
        columns = sorted({0, H // 2, H - 1})
        observed = data["x"][rows][:, columns]
    if not torch.equal(observed, snapshot["x"]):
        raise AssertionError("input tensor was modified")
    if not torch.equal(data["gamma"], snapshot["gamma"]):
        raise AssertionError("gamma tensor was modified")
    if not torch.equal(data["beta"], snapshot["beta"]):
        raise AssertionError("beta tensor was modified")
    _assert_guard(data["x_backing"][data["x_size"] :], name="input terminal")
    _assert_guard(data["gamma_backing"][H:], name="gamma terminal")
    _assert_guard(data["beta_backing"][H:], name="beta terminal")


def _check_public_dispatch(data, eps: float) -> None:
    import flashinfer

    flashinfer_norm, original_cute = _flashinfer_cute(data["x"].device)
    calls = 0

    def tracked_cute(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_cute(*args, **kwargs)

    flashinfer_norm.layernorm_cute = tracked_cute
    try:
        public_out = flashinfer.layernorm(data["x"], data["gamma"], data["beta"], eps)
    finally:
        flashinfer_norm.layernorm_cute = original_cute
    if calls != 1:
        raise AssertionError(f"expected one CuTe-DSL public dispatch, observed {calls}")
    if public_out.shape != data["x"].shape or public_out.dtype != data["x"].dtype:
        raise AssertionError("FlashInfer public LayerNorm returned an incompatible output")


def run_test(**config: Any) -> None:
    """Compile, launch, and validate one LayerNorm specialization."""
    import torch

    from tirx_kernels.runner import compile_kernel

    config = dict(config)
    M, H = int(config["M"]), int(config["H"])
    eps, enable_pdl = float(config.get("eps", _DEFAULT_EPS)), bool(config["enable_pdl"])
    data = _prepare_tensors(config)
    snapshot = _snapshot_inputs(data, M, H)
    output = _prepare_output(M, H, data["y_row_stride"], initialize_padding=True)
    reference = _prepare_output(M, H, data["y_row_stride"], initialize_padding=True)

    executable = compile_kernel(get_kernel(**config))
    _launch_tirx(executable, data, output, config)
    _, layernorm_cute = _flashinfer_cute(data["x"].device)
    returned = layernorm_cute(
        reference["view"], data["x"], data["gamma"], data["beta"], eps, enable_pdl=enable_pdl
    )
    if returned is not None:
        raise AssertionError("FlashInfer layernorm_cute must return None")
    if M == 32 and H == 768:
        _check_public_dispatch(data, eps)

    torch.cuda.synchronize()
    rows = _overflow_rows(M, H)
    actual_checked = _checked_view(output["view"], rows)
    reference_checked = _checked_view(reference["view"], rows)
    if not torch.isfinite(actual_checked).all():
        raise AssertionError("TIRx output contains non-finite values")
    _assert_close(
        actual_checked, reference_checked, name="FlashInfer CuTe-DSL", rtol=1e-3, atol=1e-3
    )
    _assert_inputs_unchanged(data, snapshot, M, H)
    _assert_output_padding(output, M, H, data["y_row_stride"], name="TIRx output")
    _assert_output_padding(reference, M, H, data["y_row_stride"], name="FlashInfer output")


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
    M, H = int(config["M"]), int(config["H"])
    eps, enable_pdl = float(config.get("eps", _DEFAULT_EPS)), bool(config["enable_pdl"])
    data = _prepare_tensors(config)
    tirx_output = _prepare_output(M, H, data["y_row_stride"], initialize_padding=False)
    flashinfer_output = _prepare_output(M, H, data["y_row_stride"], initialize_padding=False)
    executable = prepared["executable"]

    def tirx_launch():
        _launch_tirx(executable, data, tirx_output, config)

    tirx_launch()
    torch.cuda.synchronize()

    def build_flashinfer_reference():
        flashinfer_norm, layernorm_cute = _flashinfer_cute(data["x"].device)
        calls = 0

        def tracked_cute(*args, **call_kwargs):
            nonlocal calls
            calls += 1
            return layernorm_cute(*args, **call_kwargs)

        flashinfer_norm.layernorm_cute = tracked_cute
        try:
            public_out = flashinfer_norm.layernorm(data["x"], data["gamma"], data["beta"], eps)
        finally:
            flashinfer_norm.layernorm_cute = layernorm_cute
        if calls != 1:
            raise AssertionError(
                f"expected one CuTe-DSL benchmark dispatch proof, observed {calls}"
            )
        del public_out

        def flashinfer_launch():
            return layernorm_cute(
                flashinfer_output["view"],
                data["x"],
                data["gamma"],
                data["beta"],
                eps,
                enable_pdl=enable_pdl,
            )

        returned = flashinfer_launch()
        if returned is not None:
            raise AssertionError("FlashInfer benchmark LayerNorm must return None")
        torch.cuda.synchronize()
        _assert_close(
            tirx_output["view"],
            flashinfer_output["view"],
            name="benchmark precheck",
            rtol=1e-3,
            atol=1e-3,
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
