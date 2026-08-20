# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2024 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer ``act_and_mul_kernel`` port.

Ports ``flashinfer::activation::act_and_mul_kernel<T, Activation>``
(``include/flashinfer/activation.cuh``), the single template kernel behind
``flashinfer.activation.silu_and_mul``, ``gelu_and_mul``, and
``gelu_tanh_and_mul``.  The ``act`` config key mirrors the ``Activation``
function-pointer template parameter; ``dtype`` mirrors the fp16/bf16 runtime
dispatch.  Both TIRx specializations follow the source launch: one CTA per
token, ``min(d / 8, 1024)`` threads, 16-byte vectorized access, scalar
remainder loop, and ``griddepcontrol`` PDL intrinsics.
"""

from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.runner import bench

KERNEL_META = {"name": "act_and_mul", "category": "flashinfer", "compute_capability": 10}

# Source dispatch domain (DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16 + vec_t alignment):
#   dtype in {float16, bfloat16}; d % 8 == 0 (both row halves 16B-aligned); d >= 8.
_ACTS = ("silu", "gelu", "gelu_tanh")
_DTYPES = ("float16", "bfloat16")
_FI_API = {"silu": "silu_and_mul", "gelu": "gelu_and_mul", "gelu_tanh": "gelu_tanh_and_mul"}
VEC_BYTES = 16
ELEM_BYTES = 2  # fp16/bf16
VEC_SIZE = VEC_BYTES // ELEM_BYTES  # 8, matches vec_t<float, 8> in the source


def _block_size(d: int) -> int:
    return min(d // VEC_SIZE, 1024)


def _validate(act: str, dtype: str, d: int) -> None:
    if act not in _ACTS:
        raise ValueError(f"Unsupported act: {act}")
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if d < VEC_SIZE or d % VEC_SIZE != 0:
        raise ValueError(f"d={d} outside the source vectorized dispatch domain (d % 8 != 0)")


def _torch_dtype(dtype: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


# Source constants (jit/activation.py act_func_def_str).
_LOG2E = 1.4426950408889634
_SQRT1_2 = 0.7071067811865476  # M_SQRT1_2
_GELU_TANH_C0 = 0.044715
_GELU_TANH_C1 = 0.7978845608028654


def _tanh_approx(x):
    # tanh.approx.f32 (matches flashinfer math.cuh math::tanh(float))
    out = K.alloc_local([1], "float32")
    K.ptx.tanh.approx.f32(out[0], x)
    return out[0]


def _fmaf_rn(a, b, c):
    # __fmaf_rn under the production -use_fast_math build (fma.rn.ftz.f32)
    out = K.alloc_local([1], "float32")
    K.ptx.fma.rn.ftz.f32(out[0], a, b, c)
    return out[0]


def _unpack_lo(word, dtype):
    return K.cast(
        K.reinterpret(dtype, K.cast(K.bitwise_and(word, K.uint32(0xFFFF)), "uint16")), "float32"
    )


def _unpack_hi(word, dtype):
    return K.cast(
        K.reinterpret(dtype, K.cast(K.shift_right(word, K.uint32(16)), "uint16")), "float32"
    )


def get_kernel(act: str, dtype: str, num_tokens: int, d: int, **kwargs):
    """Return the TIRx specialization for one (act, dtype, num_tokens, d) config."""
    _validate(act, dtype, d)
    block_size = _block_size(d)
    n_vec = d // VEC_SIZE
    rem = d % (block_size * VEC_SIZE)
    rem_off = d - rem

    @K.kernel(warps=(block_size + 31) // 32, arch="sm_100a", grid=num_tokens)
    def act_and_mul(input_global: K.gptr[dtype, 2], out_global: K.gptr[dtype, 2]):
        token = K.cta_id()
        tid = K.thread_id()
        K.ptx.griddepcontrol.wait()

        x_bits = K.alloc_local([4], "uint32")
        y_bits = K.alloc_local([4], "uint32")
        o_bits = K.alloc_local([4], "uint32")
        x_vec = K.alloc_local([8], "float32")
        y_vec = K.alloc_local([8], "float32")
        out_vec = K.alloc_local([8], "float32")
        e_tmp = K.alloc_local([1], "float32")

        # Main vector loop (source: #pragma unroll 1 grid-stride loop).
        idx = K.alloc_local([1], "uint32")
        K.assign(idx[0], tid)
        with K.While(idx[0] < n_vec):
            K.ptx.ld.global_.nc.v4.b32(
                x_bits[0],
                x_bits[1],
                x_bits[2],
                x_bits[3],
                K.address_of(
                    input_global[
                        0, K.cast(token, "int64") * (2 * d) + K.cast(idx[0], "int64") * VEC_SIZE
                    ]
                ),
            )
            K.ptx.ld.global_.nc.v4.b32(
                y_bits[0],
                y_bits[1],
                y_bits[2],
                y_bits[3],
                K.address_of(
                    input_global[
                        0, K.cast(token, "int64") * (2 * d) + K.cast(idx[0], "int64") * VEC_SIZE + d
                    ]
                ),
            )
            for p in range(4):
                K.ptx.mov.b32(x_vec[2 * p], _unpack_lo(x_bits[p], dtype))
                K.ptx.mov.b32(x_vec[2 * p + 1], _unpack_hi(x_bits[p], dtype))
            for p in range(4):
                K.ptx.mov.b32(y_vec[2 * p], _unpack_lo(y_bits[p], dtype))
                K.ptx.mov.b32(y_vec[2 * p + 1], _unpack_hi(y_bits[p], dtype))
            for i in range(8):
                if act == "silu":
                    K.ptx.ex2.approx.ftz.f32(e_tmp[0], x_vec[i] * K.float32(-_LOG2E))
                    K.ptx.mov.b32(out_vec[i], (x_vec[i] / (K.float32(1.0) + e_tmp[0])) * y_vec[i])
                elif act == "gelu":
                    K.ptx.mov.b32(
                        out_vec[i],
                        (
                            (x_vec[i] * K.float32(0.5))
                            * (K.float32(1.0) + K.erf(x_vec[i] * K.float32(_SQRT1_2)))
                        )
                        * y_vec[i],
                    )
                else:  # gelu_tanh
                    t1 = x_vec[i] * K.float32(_GELU_TANH_C0)
                    t2 = x_vec[i] * t1
                    u = _fmaf_rn(x_vec[i], t2, x_vec[i])
                    w = u * K.float32(_GELU_TANH_C1)
                    h = _tanh_approx(w)
                    a = K.float32(1.0) + h
                    c = a * K.float32(0.5)
                    K.ptx.mov.b32(out_vec[i], (x_vec[i] * c) * y_vec[i])
            for p in range(4):
                if dtype == "float16":
                    K.ptx.cvt.rn.f16x2.f32(o_bits[p], out_vec[2 * p + 1], out_vec[2 * p])
                else:
                    K.ptx.cvt.rn.bf16x2.f32(o_bits[p], out_vec[2 * p + 1], out_vec[2 * p])
            K.ptx.st.global_.v4.b32(
                K.address_of(
                    out_global[0, K.cast(token, "int64") * d + K.cast(idx[0], "int64") * VEC_SIZE]
                ),
                o_bits[0],
                o_bits[1],
                o_bits[2],
                o_bits[3],
            )
            K.assign(idx[0], idx[0] + block_size)

        # Scalar remainder loop (source: #pragma unroll 1; dead when REM == 0).
        if rem > 0:
            ridx = K.alloc_local([1], "uint32")
            K.assign(ridx[0], tid)
            with K.While(ridx[0] < rem):
                xr16 = K.alloc_local([1], "uint16")
                yr16 = K.alloc_local([1], "uint16")
                ob16 = K.alloc_local([1], "uint16")
                er = K.alloc_local([1], "float32")
                K.ptx.ld.global_.nc.b16(
                    xr16[0],
                    K.address_of(
                        input_global[
                            0, K.cast(token, "int64") * (2 * d) + K.cast(ridx[0], "int64") + rem_off
                        ]
                    ),
                )
                K.ptx.ld.global_.nc.b16(
                    yr16[0],
                    K.address_of(
                        input_global[
                            0,
                            K.cast(token, "int64") * (2 * d)
                            + K.cast(ridx[0], "int64")
                            + rem_off
                            + d,
                        ]
                    ),
                )
                xr = K.cast(K.reinterpret(dtype, xr16[0]), "float32")
                yr = K.cast(K.reinterpret(dtype, yr16[0]), "float32")
                if act == "silu":
                    K.ptx.ex2.approx.ftz.f32(er[0], xr * K.float32(-_LOG2E))
                    out_r = (xr / (K.float32(1.0) + er[0])) * yr
                elif act == "gelu":
                    out_r = (
                        (xr * K.float32(0.5)) * (K.float32(1.0) + K.erf(xr * K.float32(_SQRT1_2)))
                    ) * yr
                else:  # gelu_tanh
                    t1 = xr * K.float32(_GELU_TANH_C0)
                    t2 = xr * t1
                    u = _fmaf_rn(xr, t2, xr)
                    w = u * K.float32(_GELU_TANH_C1)
                    h = _tanh_approx(w)
                    a = K.float32(1.0) + h
                    c = a * K.float32(0.5)
                    out_r = (xr * c) * yr
                if dtype == "float16":
                    K.ptx.cvt.rn.f16.f32(ob16[0], out_r)
                else:
                    K.ptx.cvt.rn.bf16.f32(ob16[0], out_r)
                K.ptx.st.global_.b16(
                    K.address_of(
                        out_global[
                            0, K.cast(token, "int64") * d + K.cast(ridx[0], "int64") + rem_off
                        ]
                    ),
                    ob16[0],
                )
                K.assign(ridx[0], ridx[0] + block_size)

        K.ptx.griddepcontrol.launch_dependents()

    return act_and_mul.func


def prepare_data(act: str, dtype: str, num_tokens: int, d: int, **kwargs):
    """Create the logical input: (num_tokens, 2 * d) row-major, seeded randn."""
    import torch

    _validate(act, dtype, d)
    torch.manual_seed(42)
    input_data = torch.randn(num_tokens, 2 * d, dtype=_torch_dtype(dtype), device="cuda")
    return (input_data,)


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(act: str, dtype: str, num_tokens: int, d: int, **kwargs):
    """Compile, launch, and validate one config against the flashinfer source."""
    import torch

    from tirx_kernels.runner import compile_kernel

    (input_data,) = prepare_data(act=act, dtype=dtype, num_tokens=num_tokens, d=d)
    kernel = get_kernel(act=act, dtype=dtype, num_tokens=num_tokens, d=d)
    ex = compile_kernel(kernel)
    out_tirx = torch.empty((num_tokens, d), dtype=_torch_dtype(dtype), device="cuda")
    ex(input_data, out_tirx)
    torch.cuda.synchronize()

    import flashinfer

    ref = getattr(flashinfer.activation, _FI_API[act])(input_data, enable_pdl=False)
    # Source test tolerance (tests/utils/test_activation.py): rtol=1e-3, atol=1e-3.
    torch.testing.assert_close(out_tirx, ref, rtol=1e-3, atol=1e-3)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs):
    """Benchmark the TIRx port against the flashinfer source kernel."""
    config = dict(prepared["config"])
    act = config.pop("act")
    dtype = config.pop("dtype")
    num_tokens = config.pop("num_tokens")
    d = config.pop("d")
    config.update(kwargs)
    kwargs = config
    executable = prepared["executable"]
    import torch

    (input_data,) = prepare_data(act=act, dtype=dtype, num_tokens=num_tokens, d=d)
    ex = executable
    out_tirx = torch.empty((num_tokens, d), dtype=_torch_dtype(dtype), device="cuda")

    funcs = {"tirx": lambda: ex(input_data, out_tirx)}

    def build_reference():
        import flashinfer

        out_fi = torch.empty((num_tokens, d), dtype=_torch_dtype(dtype), device="cuda")
        fn = getattr(flashinfer.activation, _FI_API[act])
        return lambda: fn(input_data, out=out_fi, enable_pdl=False)

    return bench(
        funcs,
        references={"flashinfer": build_reference},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    act: str,
    dtype: str,
    num_tokens: int,
    d: int,
    *,
    warmup=None,
    repeat=None,
    timer=None,
    rounds=1,
    cooldown_s=1.0,
    **kwargs,
):
    config = dict(kwargs)
    prepared = prepare_bench(act=act, dtype=dtype, num_tokens=num_tokens, d=d, **config)
    return prepared.run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


def _cfg(act, dtype, d, num_tokens):
    dt = {"float16": "fp16", "bfloat16": "bf16"}[dtype]
    return {
        "label": f"{act}_{dt}_d{d}_t{num_tokens}",
        "act": act,
        "dtype": dtype,
        "d": d,
        "num_tokens": num_tokens,
    }


# Correctness matrix.  Covers: every (act, dtype) instantiation; the three
# block regimes (d/8 < 1024 single-iteration, == 1024 boundary, > 1024
# grid-stride loop); the scalar remainder loop (d % 8192 != 0 with d/8 > 1024);
# and grid extremes (tokens = 1 .. 8192, the source test maximum).
CONFIGS = [
    # act x dtype instantiation coverage on a standard shape (block = 512)
    _cfg("silu", "float16", 4096, 1024),
    _cfg("gelu", "float16", 4096, 1024),
    _cfg("gelu_tanh", "float16", 4096, 1024),
    _cfg("silu", "bfloat16", 4096, 1024),
    _cfg("gelu", "bfloat16", 4096, 1024),
    _cfg("gelu_tanh", "bfloat16", 4096, 1024),
    # small d: block = d/8 < 1024, one vector per thread, no remainder
    _cfg("silu", "float16", 128, 16),
    _cfg("silu", "float16", 2048, 512),
    # block-cap boundary: d/8 == 1024, one vector per thread, no remainder
    _cfg("silu", "float16", 8192, 64),
    # grid-stride vector loop (2 iterations) + 2816-element scalar remainder
    _cfg("silu", "float16", 11008, 1024),
    _cfg("silu", "bfloat16", 11008, 1024),
    # grid-stride vector loop, exact multiple (no remainder); max source tokens
    _cfg("silu", "float16", 16384, 8192),
    # single-token grid
    _cfg("silu", "float16", 11008, 1),
]

# Benchmark sweep: LLM gated-MLP shapes.  d = 4096 / 11008 / 16384 intermediate
# sizes; tokens = 1 (decode), 8192 (source test maximum), 32768 (large prefill).
BENCH_CONFIGS = [
    _cfg("silu", "float16", 4096, 1),
    _cfg("silu", "float16", 4096, 8192),
    _cfg("silu", "float16", 11008, 8192),
    _cfg("silu", "float16", 16384, 32768),
    _cfg("silu", "bfloat16", 4096, 8192),
    _cfg("silu", "bfloat16", 16384, 32768),
    _cfg("gelu", "float16", 11008, 8192),
    _cfg("gelu_tanh", "float16", 11008, 8192),
]
