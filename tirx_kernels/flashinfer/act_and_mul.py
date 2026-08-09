# Copyright (c) 2026 The TIRX Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
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

from tvm.script import tirx as T
from tvm.tirx.bench import bench

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
    out = T.alloc_local([1], "float32")
    T.evaluate(T.ptx.tanh.approx.f32(out[0], x))
    return out[0]


def _fmaf_rn(a, b, c):
    # __fmaf_rn under the production -use_fast_math build (fma.rn.ftz.f32)
    out = T.alloc_local([1], "float32")
    T.evaluate(T.ptx.fma.rn.ftz.f32(out[0], a, b, c))
    return out[0]


def _unpack_lo(word, dtype):
    return T.cast(
        T.reinterpret(dtype, T.cast(T.bitwise_and(word, T.uint32(0xFFFF)), "uint16")), "float32"
    )


def _unpack_hi(word, dtype):
    return T.cast(
        T.reinterpret(dtype, T.cast(T.shift_right(word, T.uint32(16)), "uint16")), "float32"
    )


def get_kernel(act: str, dtype: str, num_tokens: int, d: int, **kwargs):
    """Return the TIRx specialization for one (act, dtype, num_tokens, d) config."""
    _validate(act, dtype, d)
    block_size = _block_size(d)
    n_vec = d // VEC_SIZE
    rem = d % (block_size * VEC_SIZE)
    rem_off = d - rem

    @T.prim_func
    def act_and_mul(input_ptr: T.handle, out_ptr: T.handle):
        input_global = T.match_buffer(
            input_ptr, shape=[num_tokens, 2 * d], dtype=dtype, scope="global"
        )
        out_global = T.match_buffer(out_ptr, shape=[num_tokens, d], dtype=dtype, scope="global")
        T.device_entry()
        token = T.cta_id([num_tokens])
        tid = T.thread_id([block_size], dtype="uint32")
        T.evaluate(T.ptx.griddepcontrol.wait())

        x_bits: T.uint32[4]
        y_bits: T.uint32[4]
        o_bits: T.uint32[4]
        x_vec: T.f32[8]
        y_vec: T.f32[8]
        out_vec: T.f32[8]
        e_tmp: T.f32[1]

        # Main vector loop (source: #pragma unroll 1 grid-stride loop).
        idx: T.uint32 = tid
        while idx < n_vec:
            T.ptx.ld.global_.nc.v4.b32(
                x_bits[0],
                x_bits[1],
                x_bits[2],
                x_bits[3],
                T.address_of(input_global[token, T.cast(idx, "int64") * VEC_SIZE]),
            )
            T.ptx.ld.global_.nc.v4.b32(
                y_bits[0],
                y_bits[1],
                y_bits[2],
                y_bits[3],
                T.address_of(input_global[token, T.cast(idx, "int64") * VEC_SIZE + d]),
            )
            for p in T.unroll(4):
                x_vec[2 * p] = _unpack_lo(x_bits[p], dtype)
                x_vec[2 * p + 1] = _unpack_hi(x_bits[p], dtype)
            for p in T.unroll(4):
                y_vec[2 * p] = _unpack_lo(y_bits[p], dtype)
                y_vec[2 * p + 1] = _unpack_hi(y_bits[p], dtype)
            for i in T.unroll(8):
                if act == "silu":
                    T.evaluate(T.ptx.ex2.approx.ftz.f32(e_tmp[0], x_vec[i] * T.float32(-_LOG2E)))
                    out_vec[i] = (x_vec[i] / (T.float32(1.0) + e_tmp[0])) * y_vec[i]
                elif act == "gelu":
                    out_vec[i] = (
                        (x_vec[i] * T.float32(0.5))
                        * (T.float32(1.0) + T.erf(x_vec[i] * T.float32(_SQRT1_2)))
                    ) * y_vec[i]
                else:  # gelu_tanh
                    t1 = x_vec[i] * T.float32(_GELU_TANH_C0)
                    t2 = x_vec[i] * t1
                    u = _fmaf_rn(x_vec[i], t2, x_vec[i])
                    w = u * T.float32(_GELU_TANH_C1)
                    h = _tanh_approx(w)
                    a = T.float32(1.0) + h
                    c = a * T.float32(0.5)
                    out_vec[i] = (x_vec[i] * c) * y_vec[i]
            for p in T.unroll(4):
                if dtype == "float16":
                    T.evaluate(
                        T.ptx.cvt.rn.f16x2.f32(o_bits[p], out_vec[2 * p + 1], out_vec[2 * p])
                    )
                else:
                    T.evaluate(
                        T.ptx.cvt.rn.bf16x2.f32(o_bits[p], out_vec[2 * p + 1], out_vec[2 * p])
                    )
            T.ptx.st.global_.v4.b32(
                T.address_of(out_global[token, T.cast(idx, "int64") * VEC_SIZE]),
                o_bits[0],
                o_bits[1],
                o_bits[2],
                o_bits[3],
            )
            idx = idx + block_size

        # Scalar remainder loop (source: #pragma unroll 1; dead when REM == 0).
        if rem > 0:
            ridx: T.uint32 = tid
            while ridx < rem:
                xr16 = T.alloc_local([1], "uint16")
                yr16 = T.alloc_local([1], "uint16")
                ob16 = T.alloc_local([1], "uint16")
                er = T.alloc_local([1], "float32")
                T.ptx.ld.global_.nc.b16(
                    xr16[0], T.address_of(input_global[token, T.cast(ridx, "int64") + rem_off])
                )
                T.ptx.ld.global_.nc.b16(
                    yr16[0], T.address_of(input_global[token, T.cast(ridx, "int64") + rem_off + d])
                )
                xr = T.cast(T.reinterpret(dtype, xr16[0]), "float32")
                yr = T.cast(T.reinterpret(dtype, yr16[0]), "float32")
                if act == "silu":
                    T.evaluate(T.ptx.ex2.approx.ftz.f32(er[0], xr * T.float32(-_LOG2E)))
                    out_r = (xr / (T.float32(1.0) + er[0])) * yr
                elif act == "gelu":
                    out_r = (
                        (xr * T.float32(0.5)) * (T.float32(1.0) + T.erf(xr * T.float32(_SQRT1_2)))
                    ) * yr
                else:  # gelu_tanh
                    t1 = xr * T.float32(_GELU_TANH_C0)
                    t2 = xr * t1
                    u = _fmaf_rn(xr, t2, xr)
                    w = u * T.float32(_GELU_TANH_C1)
                    h = _tanh_approx(w)
                    a = T.float32(1.0) + h
                    c = a * T.float32(0.5)
                    out_r = (xr * c) * yr
                if dtype == "float16":
                    T.evaluate(T.ptx.cvt.rn.f16.f32(ob16[0], out_r))
                else:
                    T.evaluate(T.ptx.cvt.rn.bf16.f32(ob16[0], out_r))
                T.ptx.st.global_.b16(
                    T.address_of(out_global[token, T.cast(ridx, "int64") + rem_off]), ob16[0]
                )
                ridx = ridx + block_size

        T.ptx.griddepcontrol.launch_dependents()

    return act_and_mul


def prepare_data(act: str, dtype: str, num_tokens: int, d: int, **kwargs):
    """Create the logical input: (num_tokens, 2 * d) row-major, seeded randn."""
    import torch

    _validate(act, dtype, d)
    torch.manual_seed(42)
    input_data = torch.randn(num_tokens, 2 * d, dtype=_torch_dtype(dtype), device="cuda")
    return (input_data,)


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
    """Benchmark the TIRx port against the flashinfer source kernel."""
    import torch

    from tirx_kernels.runner import compile_kernel

    (input_data,) = prepare_data(act=act, dtype=dtype, num_tokens=num_tokens, d=d)
    kernel = get_kernel(act=act, dtype=dtype, num_tokens=num_tokens, d=d)
    ex = compile_kernel(kernel)
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
