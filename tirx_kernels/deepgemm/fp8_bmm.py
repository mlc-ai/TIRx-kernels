# Copyright (c) 2025 DeepSeek
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
#
# TIRx port of DeepGEMM's `sm100_fp8_bmm` host entry
# (`csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:393`).
# See NOTICE and THIRD_PARTY_LICENSES.md for upstream attribution.

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
"""

from __future__ import annotations

from ._sm100_fp8_fp4_gemm_1d1d import GemmDesc, GemmType, Major, get_best_config

KERNEL_META = {"name": "deepgemm_sm100_fp8_bmm", "category": "deepgemm", "compute_capability": 10}

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
        import torch

        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
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
    from ._sm100_fp8_fp4_gemm_1d1d_data import prepare_bmm

    config.pop("label", None)
    return prepare_bmm(**config)


def _tirx_launch(data, config):
    from ._sm100_fp8_fp4_gemm_1d1d import build_launch

    return build_launch(
        _spec_for(config),
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
    """Compile, launch and compare against the dequantized-einsum oracle."""
    import torch

    from ._sm100_fp8_fp4_gemm_1d1d_data import assert_within_threshold, calc_diff

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

    return assert_within_threshold(
        calc_diff(data["d"], data["ref"]),
        data,
        kernel="deepgemm_sm100_fp8_bmm",
        detail=(f"{data['expr']} batch={data['batch']} M={data['M']} N={data['N']} K={data['K']}"),
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    from ._sm100_fp8_fp4_gemm_1d1d_data import bench_against_deepgemm, deepgemm_launch_bmm

    config.pop("label", None)
    data = prepare_data(**config)
    return bench_against_deepgemm(
        _tirx_launch(data, config),
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


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "make_desc",
    "prepare_data",
    "run_bench",
    "run_test",
    "shape_of",
]
