# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Batched FP8 GEMM -- `GemmType::Batched`.

Reference: `deep_gemm.fp8_einsum`.  This is the only entry that builds rank-3 TMA
descriptors for A, B and C/D; the scheduler derives the batch index from the
block index and skips the L2 swizzle entirely.  FP8 only.

The three supported expression forms map onto (batch, m, n, k) as:

===========================  ==========================  ==============  ========
expression                   (batch, m, n, k)            majors          output
===========================  ==========================  ==============  ========
``bhr,hdr->bhd``             ``(h, b, d, r)``            K, K            bf16
``bhd,hdr->bhr``             ``(h, b, r, d)``            K, MN           bf16
``bhd,bhr->hdr``             ``(h, d, r, b)``            MN, MN          fp32+acc
===========================  ==========================  ==============  ========

Upstream source: csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:393.
"""

from __future__ import annotations

from ._sm100_fp8_fp4_gemm_1d1d import GemmDesc, GemmType, Major, get_best_config

KERNEL_META = {
    "name": "deepgemm_sm100_fp8_bmm",
    "category": "deepgemm",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a", "sm_110a"],
    "reference_requirements": (
        {
            "package": "deep-gemm",
            "git": {
                "url": "https://github.com/deepseek-ai/DeepGEMM.git",
                "commit": "559d79fb6994a58b8a15b4b93bf13ccc16edf247",
            },
            "import": "deep_gemm",
        },
    ),
}
#: expression -> (major_a, major_b, cd_dtype, with_accumulation)
_EXPRESSIONS = {
    "bhr,hdr->bhd": ("k", "k", "bf16", False),
    "bhd,hdr->bhr": ("k", "mn", "bf16", False),
    "bhd,bhr->hdr": ("mn", "mn", "fp32", True),
}


def _case(expr: str, h: int, r: int, d: int, b: int) -> dict:
    tag = {
        "bhr,hdr->bhd": "bhr_hdr_bhd",
        "bhd,hdr->bhr": "bhd_hdr_bhr",
        "bhd,bhr->hdr": "bhd_bhr_hdr",
    }[expr]
    return {"label": f"{tag}_b{b}_h{h}_r{r}_d{d}", "expr": expr, "H": h, "R": r, "D": d, "B": b}


CONFIGS = [
    _case("bhr,hdr->bhd", 8, 4096, 1024, 4),
    _case("bhr,hdr->bhd", 8, 4096, 1024, 4096),
    _case("bhd,hdr->bhr", 8, 4096, 1024, 128),
    _case("bhd,hdr->bhr", 8, 4096, 1024, 8192),
    _case("bhd,bhr->hdr", 8, 4096, 1024, 4096),
]

BENCH_CONFIGS = [
    _case("bhr,hdr->bhd", 8, 4096, 1024, 4096),
    _case("bhr,hdr->bhd", 8, 4096, 1024, 8192),
    _case("bhd,hdr->bhr", 8, 4096, 1024, 8192),
    _case("bhd,bhr->hdr", 8, 4096, 1024, 4096),
]

_MAJOR = {"k": Major.K, "mn": Major.MN}


def shape_of(expr: str, *, H: int, R: int, D: int, B: int) -> tuple[int, int, int, int]:
    """Map one einsum expression to `(batch, m, n, k)`."""
    if expr == "bhr,hdr->bhd":
        return H, B, D, R
    if expr == "bhd,hdr->bhr":
        return H, B, R, D
    if expr == "bhd,bhr->hdr":
        return H, D, R, B
    raise ValueError(f"unsupported einsum expression: {expr}")


def make_desc(*, expr: str, H: int, R: int, D: int, B: int, num_sms: int | None = None) -> GemmDesc:
    """Build the descriptor `sm100_fp8_bmm` would build."""
    if num_sms is None:
        from tirx_kernels.runner import hardware_num_sms

        num_sms = hardware_num_sms()
    major_a, major_b, cd_dtype, accumulate = _EXPRESSIONS[expr]
    batch, m, n, k = shape_of(expr, H=H, R=R, D=D, B=B)
    return GemmDesc(
        gemm_type=GemmType.BATCHED,
        m=m,
        n=n,
        k=k,
        num_groups=batch,
        cd_dtype=cd_dtype,
        major_a=_MAJOR[major_a],
        major_b=_MAJOR[major_b],
        with_accumulation=accumulate,
        num_sms=num_sms,
    )


def get_kernel(**config):
    from ._sm100_fp8_fp4_gemm_1d1d import build_kernel

    config.pop("label", None)
    return build_kernel(_spec_for(config))


def _spec_for(config: dict):
    from ._sm100_fp8_fp4_gemm_1d1d import make_spec

    desc = make_desc(**config)
    return make_spec(desc, get_best_config(desc), gran_k_a=128, gran_k_b=128, k_alignment=128)


def prepare_data(**config):
    from ._sm100_fp8_fp4_gemm_1d1d.data import prepare_bmm

    config.pop("label", None)
    return prepare_bmm(**config)


def prepare_bench(**config):
    """Compile the exact DeepGEMM specialization without initializing CUDA."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    config.pop("label", None)
    spec = _spec_for(config)
    from ._sm100_fp8_fp4_gemm_1d1d import compile_spec

    state = {"config": dict(config), "executable": compile_spec(spec)}
    return prepared_gpu_benchmark(run_gpu, state)


def _tirx_launch(data, config, executable=None):
    from ._sm100_fp8_fp4_gemm_1d1d import build_launch

    return build_launch(
        _spec_for(config),
        executable=executable,
        a=data["a"],
        b=data["b"],
        sfa=data["sfa"],
        sfb=data["sfb"],
        d=data["d"],
        shape_m=data["M"],
        shape_n=data["N"],
        shape_k=data["K"],
        num_groups=data["batch"],
        sf_num_groups_a=data["batch"],
        sf_num_groups_b=data["batch"],
    )


def run_test(**config):
    """Compile, launch and compare against DeepGEMM on the same operands."""
    import torch

    from tirx_kernels.runner import prepare_cuda_arch

    from ._sm100_fp8_fp4_gemm_1d1d.data import (
        assert_within_threshold,
        calc_diff,
        deepgemm_launch_bmm,
        max_diff_threshold,
    )

    config.pop("label", None)
    data = prepare_data(**config)
    launch = _tirx_launch(data, config)
    if data["c"] is not None:
        # The accumulate path stores with `cp.reduce ... add`, so the output must
        # start out holding C.
        data["z"].copy_(data["z0"])
    else:
        data["z"].zero_()
    launch()
    torch.cuda.synchronize()

    def check(actual, expected, threshold=None):
        return assert_within_threshold(
            calc_diff(actual, expected),
            data,
            kernel="deepgemm_sm100_fp8_bmm",
            detail=(
                f"{data['expr']} batch={data['batch']} M={data['M']} N={data['N']} K={data['K']}"
            ),
            threshold=threshold,
        )

    if prepare_cuda_arch() == "sm_110a":
        actual, expected = data["d"], data["ref"]
        threshold = max_diff_threshold(data["a_dtype"], data["b_dtype"])
        check(actual, expected, threshold)

    _, expected = deepgemm_launch_bmm(
        data, out=data["z0"].clone() if data["c"] is not None else None
    )
    torch.cuda.synchronize()
    actual = data["z"]
    return check(actual, expected)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    config = {**prepared["config"], **config}
    from ._sm100_fp8_fp4_gemm_1d1d.data import bench_against_deepgemm, deepgemm_launch_bmm

    config.pop("label", None)
    data = prepare_data(**config)
    return bench_against_deepgemm(
        _tirx_launch(data, config, executable=prepared["executable"]),
        deepgemm_launch_bmm,
        data,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
        batch=data["batch"],
        M=data["M"],
        N=data["N"],
        K=data["K"],
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    return prepare_bench(**config).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "make_desc",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_test",
    "shape_of",
]
