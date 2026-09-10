# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2023 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer ``MergeStateKernel`` port.

Ports ``flashinfer::MergeStateKernel<vec_size, DTypeIn, DTypeO>``
(``include/flashinfer/attention/cascade.cuh``), the kernel behind
``flashinfer.cascade.merge_state``: it merges the partial attention outputs
``v_a``/``v_b`` and their base-2 log-sum-exp values ``s_a``/``s_b`` of two
KV segments into the output of the concatenated segment.  ``dtype`` mirrors
the f16/bf16 runtime dispatch of ``csrc/cascade.cu``; ``head_dim`` selects the
``vec_size`` template argument through ``DISPATCH_HEAD_DIM`` exactly as the
source host launcher ``flashinfer::MergeState`` does.  One CTA per position,
``head_dim / vec_size`` threads per head along ``x`` and one head per ``y``.
"""

from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.runner import bench

KERNEL_META = {
    "name": "merge_state",
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

# Source dispatch domain: DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16 (f16/bf16 only) and
# DISPATCH_HEAD_DIM (utils.cuh) with the MergeState launch formulae.
_DTYPES = ("float16", "bfloat16")
_HEAD_DIMS = (64, 128, 256, 512)
ELEM_BYTES = 2  # fp16/bf16
MAX_THREADS = 1024


def _vec_size(head_dim: int) -> int:
    # MergeState: constexpr vec_size = max(16 / sizeof(DTypeIn), HEAD_DIM / 32)
    return max(16 // ELEM_BYTES, head_dim // 32)


def _bdx(head_dim: int) -> int:
    return head_dim // _vec_size(head_dim)


def _validate(dtype: str, seq_len: int, num_heads: int, head_dim: int) -> None:
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if head_dim not in _HEAD_DIMS:
        raise ValueError(f"head_dim={head_dim} outside the source DISPATCH_HEAD_DIM domain")
    if seq_len < 1 or num_heads < 1:
        raise ValueError("seq_len and num_heads must be positive")
    if _bdx(head_dim) * num_heads > MAX_THREADS:
        raise ValueError(
            f"bdx*bdy = {_bdx(head_dim)}*{num_heads} exceeds the {MAX_THREADS}-thread block limit"
        )


def _torch_dtype(dtype: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def get_kernel(dtype: str, seq_len: int, num_heads: int, head_dim: int, **kwargs):
    """Return the TIRx specialization for one (dtype, seq_len, num_heads, head_dim) config."""
    _validate(dtype, seq_len, num_heads, head_dim)
    vec = _vec_size(head_dim)  # source template vec_size
    bdx = _bdx(head_dim)  # source blockDim.x
    nwords = vec // 2  # packed b32 words per slice
    nthreads = bdx * num_heads  # source blockDim.x * blockDim.y
    warps = (nthreads + 31) // 32
    bdx_shift = bdx.bit_length() - 1  # bdx is 8, 16, or 32
    is_f16 = dtype == "float16"

    def widen_pair(dst, w, word):
        # vec_cast<float, DTypeIn>: one packed pair -> two f32.
        if is_f16:
            # __half22float2: mov.b32 {lo, hi} + cvt.f32.f16 x2 (K.idioms spells the same sequence).
            K.idioms.cast_f16x2_to_f32x2(dst, w, word)
        else:
            # __bfloat1622float2: mov.b32 {lo, hi} + cvt.f32.bf16 x2.
            halves = K.alloc_local([2], "uint16")
            K.ptx.mov.b32(halves[0], halves[1], word)
            K.ptx.cvt.f32.bf16(dst[2 * w], halves[0])
            K.ptx.cvt.f32.bf16(dst[2 * w + 1], halves[1])

    def narrow_pair(dst, lo, hi):
        # vec_cast<DTypeO, float>: __float22half2_rn / __float22bfloat162_rn, hi operand first.
        if is_f16:
            K.ptx.cvt.rn.f16x2.f32(dst, hi, lo)
        else:
            K.ptx.cvt.rn.bf16x2.f32(dst, hi, lo)

    @K.kernel(warps=warps, arch="sm_100a", grid=seq_len)
    def merge_state(
        v_a: K.gptr[dtype],
        s_a: K.gptr[K.f32],
        v_b: K.gptr[dtype],
        s_b: K.gptr[K.f32],
        v_merged: K.gptr[dtype],
        s_merged: K.gptr[K.f32],
    ):
        pos = K.cta_id()  # blockIdx.x
        tid = K.thread_id()
        # The source launches a (bdx, bdy) block and reads threadIdx.x / threadIdx.y.
        # Kern owns one flat thread axis of the same order (tid = ty * bdx + tx).
        if bdx == 32:
            # One warp per head: the lane and warp ids are exactly (tx, ty).
            tx = K.lane_id()
            ty = K.warp_id()  # head_idx, warp-uniform
        else:
            tx = tid & (bdx - 1)
            ty = tid >> bdx_shift  # head_idx

        def body():
            # ---- blend weights from the two log2-sum-exp values (source L53-59) ----
            row = K.local_scalar("int32", init=pos * num_heads + ty)  # pos * num_heads + head_idx
            row64 = K.Cast("int64", row)
            s_a_val = K.local_scalar("float32")
            s_b_val = K.local_scalar("float32")
            K.ptx.ld.global_.nc.b32(s_a_val, s_a.ptr_to([row64]))
            K.ptx.ld.global_.nc.b32(s_b_val, s_b.ptr_to([row64]))
            s_max = K.local_scalar("float32")
            K.ptx.max.ftz.f32(s_max, s_a_val, s_b_val)
            d_a = K.local_scalar("float32")
            d_b = K.local_scalar("float32")
            e_a = K.local_scalar("float32")
            e_b = K.local_scalar("float32")
            K.ptx.sub.ftz.f32(d_a, s_a_val, s_max)
            K.ptx.ex2.approx.ftz.f32(e_a, d_a)  # math::ptx_exp2
            K.ptx.sub.ftz.f32(d_b, s_b_val, s_max)
            K.ptx.ex2.approx.ftz.f32(e_b, d_b)  # math::ptx_exp2
            denom = K.local_scalar("float32")
            K.ptx.add.ftz.f32(denom, e_a, e_b)
            a_scale = K.local_scalar("float32")
            b_scale = K.local_scalar("float32")
            # `/` under the production -use_fast_math build: div.approx.ftz.f32.
            K.ptx.div.approx.ftz.f32(a_scale, e_a, denom)
            K.ptx.div.approx.ftz.f32(b_scale, e_b, denom)

            # ---- cast_load of the vec_size slice: v_a (L61), then v_b (L62) ----
            slice_ = K.local_scalar("int32", init=row * head_dim + tx * vec)
            slice64 = K.Cast("int64", slice_)

            a_bits = K.alloc_local([nwords], "uint32")
            a_vec = K.alloc_local([vec], "float32")
            for k in range(vec // 8):
                K.ptx.ld.global_.nc.v4.b32(
                    a_bits[4 * k],
                    a_bits[4 * k + 1],
                    a_bits[4 * k + 2],
                    a_bits[4 * k + 3],
                    v_a.ptr_to([slice64 + 8 * k]),
                )
            for w in range(nwords):
                widen_pair(a_vec, w, a_bits[w])

            b_bits = K.alloc_local([nwords], "uint32")
            b_vec = K.alloc_local([vec], "float32")
            for k in range(vec // 8):
                K.ptx.ld.global_.nc.v4.b32(
                    b_bits[4 * k],
                    b_bits[4 * k + 1],
                    b_bits[4 * k + 2],
                    b_bits[4 * k + 3],
                    v_b.ptr_to([slice64 + 8 * k]),
                )
            for w in range(nwords):
                widen_pair(b_vec, w, b_bits[w])

            # ---- v_merged[i] = a_scale * v_a[i] + b_scale * v_b[i] (L63-66) ----
            # nvcc contracts the source expression into mul(b_scale, v_b) + fma(a_scale, v_a, .).
            o_vec = K.alloc_local([vec], "float32")
            bt = K.alloc_local([vec], "float32")
            for i in range(vec):
                K.ptx.mul.ftz.f32(bt[i], b_scale, b_vec[i])
                K.ptx.fma.rn.ftz.f32(o_vec[i], a_scale, a_vec[i], bt[i])

            # ---- cast_store of the merged slice (L67) ----
            o_bits = K.alloc_local([nwords], "uint32")
            for w in range(nwords):
                narrow_pair(o_bits[w], o_vec[2 * w], o_vec[2 * w + 1])
            for k in range(vec // 8):
                K.ptx.st.global_.v4.b32(
                    v_merged.ptr_to([slice64 + 8 * k]),
                    o_bits[4 * k],
                    o_bits[4 * k + 1],
                    o_bits[4 * k + 2],
                    o_bits[4 * k + 3],
                )

            # One thread per head owns the scalar output, even though every
            # thread computed the same blend weights. Keep the optional-output guard.
            with K.If(K.And(tx == 0, K.Not(K.isnullptr(s_merged.data)))), K.Then():
                lg = K.local_scalar("float32")
                lse = K.local_scalar("float32")
                K.ptx.lg2.approx.ftz.f32(lg, denom)  # math::ptx_log2
                K.ptx.add.ftz.f32(lse, s_max, lg)
                K.ptx.st.global_.b32(s_merged.ptr_to([row64]), lse)

        if nthreads % 32 == 0:
            body()
        else:
            # The source block is exactly bdx * bdy threads; Kern pads it to whole
            # warps, so the padding threads (which would alias head bdy) must idle.
            with K.If(tid < nthreads), K.Then():
                body()

    return merge_state.func


def prepare_data(dtype: str, seq_len: int, num_heads: int, head_dim: int, **kwargs):
    """Create the logical inputs: v_a/v_b (seq_len, num_heads, head_dim), s_a/s_b (seq_len, num_heads)."""
    import torch

    _validate(dtype, seq_len, num_heads, head_dim)
    torch.manual_seed(42)
    tdt = _torch_dtype(dtype)
    v_a = torch.randn(seq_len, num_heads, head_dim, dtype=tdt, device="cuda")
    s_a = torch.randn(seq_len, num_heads, dtype=torch.float32, device="cuda")
    v_b = torch.randn(seq_len, num_heads, head_dim, dtype=tdt, device="cuda")
    s_b = torch.randn(seq_len, num_heads, dtype=torch.float32, device="cuda")
    return v_a, s_a, v_b, s_b


def _kernel_args(v_a, s_a, v_b, s_b, v_merged, s_merged):
    return (
        v_a.view(-1),
        s_a.view(-1),
        v_b.view(-1),
        s_b.view(-1),
        v_merged.view(-1),
        s_merged.view(-1),
    )


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(dtype: str, seq_len: int, num_heads: int, head_dim: int, **kwargs):
    """Compile, launch, and validate one config against the flashinfer source."""
    import torch

    from tirx_kernels.runner import compile_kernel

    v_a, s_a, v_b, s_b = prepare_data(
        dtype=dtype, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim
    )
    kernel = get_kernel(dtype=dtype, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim)
    ex = compile_kernel(kernel)
    v_out = torch.empty_like(v_a)
    s_out = torch.empty_like(s_a)
    ex(*_kernel_args(v_a, s_a, v_b, s_b, v_out, s_out))
    torch.cuda.synchronize()

    import flashinfer

    v_ref, s_ref = flashinfer.merge_state(v_a, s_a, v_b, s_b)
    torch.cuda.synchronize()
    # Source test tolerance (tests/attention/test_shared_prefix_kernels.py): rtol=1e-3, atol=1e-3.
    torch.testing.assert_close(v_out, v_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(s_out, s_ref, rtol=1e-3, atol=1e-3)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs):
    """Benchmark the TIRx port against the flashinfer source kernel."""
    config = dict(prepared["config"])
    dtype = config.pop("dtype")
    seq_len = config.pop("seq_len")
    num_heads = config.pop("num_heads")
    head_dim = config.pop("head_dim")
    config.update(kwargs)
    kwargs = config
    executable = prepared["executable"]
    import torch

    v_a, s_a, v_b, s_b = prepare_data(
        dtype=dtype, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim
    )
    v_out = torch.empty_like(v_a)
    s_out = torch.empty_like(s_a)
    args = _kernel_args(v_a, s_a, v_b, s_b, v_out, s_out)

    funcs = {"tirx": lambda: executable(*args)}

    def build_reference():
        # The source module op with preallocated outputs, i.e. exactly the launch
        # flashinfer.merge_state performs after its two torch.empty_like calls.
        from flashinfer.cascade import get_cascade_module

        op = get_cascade_module().merge_state
        v_fi = torch.empty_like(v_a)
        s_fi = torch.empty_like(s_a)
        return lambda: op(v_a, s_a, v_b, s_b, v_fi, s_fi)

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
    dtype: str,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    *,
    warmup=None,
    repeat=None,
    timer=None,
    rounds=1,
    cooldown_s=1.0,
    **kwargs,
):
    config = dict(kwargs)
    prepared = prepare_bench(
        dtype=dtype, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim, **config
    )
    return prepared.run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


def _cfg(dtype, seq_len, num_heads, head_dim):
    dt = {"float16": "fp16", "bfloat16": "bf16"}[dtype]
    return {
        "label": f"{dt}_s{seq_len}_h{num_heads}_d{head_dim}",
        "dtype": dtype,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }


# Correctness matrix.  Covers: both dtypes; every DISPATCH_HEAD_DIM branch
# (64/128/256 -> vec_size 8 with bdx 8/16/32, 512 -> vec_size 16); the
# 1024-thread block bound for each bdx; single-position and odd grids; and the
# source test shape (seq_len 512, 32 heads, head_dim 128).
CONFIGS = [
    # head_dim branches on a standard head count
    _cfg("float16", 37, 8, 64),
    _cfg("float16", 37, 8, 128),
    _cfg("float16", 37, 8, 256),
    _cfg("float16", 37, 8, 512),
    _cfg("bfloat16", 37, 8, 128),
    _cfg("bfloat16", 37, 8, 256),
    _cfg("bfloat16", 37, 8, 512),
    # block bound: bdx * num_heads == 1024 for bdx 8 / 16 / 32
    _cfg("float16", 5, 128, 64),
    _cfg("float16", 3, 64, 128),
    _cfg("float16", 3, 32, 256),
    _cfg("bfloat16", 2, 32, 512),
    # single head, single position (one partial warp)
    _cfg("float16", 1, 1, 128),
    _cfg("bfloat16", 1, 1, 64),
    # source test shape and the docstring example
    _cfg("float16", 512, 32, 128),
    _cfg("bfloat16", 2048, 32, 128),
]

# Benchmark sweep: cascade-inference merge shapes.  seq_len = 128 (decode
# batch), 2048 (docstring example), 16384 (large prefill); head_dim 64/128/256
# with 32 heads and 512 with 8 heads; f16 and bf16.
BENCH_CONFIGS = [
    _cfg("float16", 128, 32, 128),
    _cfg("float16", 2048, 32, 128),
    _cfg("float16", 16384, 32, 128),
    _cfg("bfloat16", 2048, 32, 128),
    _cfg("bfloat16", 16384, 32, 128),
    _cfg("float16", 2048, 32, 64),
    _cfg("float16", 2048, 32, 256),
    _cfg("float16", 2048, 8, 512),
]
