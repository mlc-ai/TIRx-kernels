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
# TIRx port of DeepGEMM's `sm100_fp8_fp4_gemm_1d1d` host entry
# (`csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:93`).
# See NOTICE and THIRD_PARTY_LICENSES.md for upstream attribution.

"""Dense FP8/FP4 GEMM -- `GemmType::Normal` instantiation.

Reference: `deep_gemm.fp8_fp4_gemm_nt` (aliased as `deep_gemm.fp8_gemm_nt`).
The kernel body lives in :mod:`tirx_kernels.deepgemm._sm100_fp8_fp4_gemm_1d1d`;
this module only pins the descriptor and the test/benchmark matrices.
"""

from __future__ import annotations

from ._sm100_fp8_fp4_gemm_1d1d import GemmDesc, GemmType, Major, get_best_config

KERNEL_META = {
    "name": "deepgemm_sm100_fp8_gemm_1d1d",
    "category": "deepgemm",
    "compute_capability": 10,
}

#: `tests/generators.py::enumerate_normal` uses these seven (n, k) pairs for
#: BF16 output; they are also the shapes the retired `fp8_blockwise_gemm`
#: benchmarked, so keeping them verbatim preserves baseline continuity.
_NK = [
    (2112, 7168),
    (576, 7168),
    (24576, 1536),
    (32768, 512),
    (7168, 16384),
    (4096, 7168),
    (7168, 2048),
]


def _case(
    m: int,
    n: int,
    k: int,
    *,
    major_a: str = "k",
    major_b: str = "k",
    accumulate: bool = False,
    cd_dtype: str = "bf16",
    b_dtype: str = "fp8",
    suffix: str = "",
) -> dict:
    label = f"m{m}_n{n}_k{k}"
    if major_a != "k" or major_b != "k":
        label += f"_{major_a}{major_b}"
    if accumulate:
        label += "_acc"
    if cd_dtype != "bf16":
        label += f"_{cd_dtype}"
    if b_dtype != "fp8":
        label += f"_b{b_dtype}"
    return {
        "label": label + suffix,
        "M": m,
        "N": n,
        "K": k,
        "major_a": major_a,
        "major_b": major_b,
        "accumulate": accumulate,
        "cd_dtype": cd_dtype,
        "b_dtype": b_dtype,
    }


CONFIGS = [
    # Forward, NT, FP8 x FP8 -> BF16: the full (n, k) sweep.
    *[_case(4096, n, k) for n, k in _NK],
    # Small-M forward: exercises the block-M 32/64 candidate branches.
    *[_case(m, n, k) for m in (1, 128) for n, k in ((4096, 7168), (7168, 2048))],
    # BF16 accumulation (`c` provided) -- SM100-only branch of `enumerate_normal`.
    *[_case(4096, n, k, accumulate=True) for n, k in ((4096, 7168), (7168, 2048))],
    # Dgrad shape/major: `[m, k] @ [k, n]`, B is MN-major.
    *[_case(4096, k, n, major_b="mn") for n, k in ((4096, 7168), (7168, 2048))],
    # Wgrad: both operands MN-major, FP32 output with accumulation.
    *[
        _case(n, 4096, k, major_a="mn", major_b="mn", accumulate=True, cd_dtype="fp32")
        for n, k in ((4096, 7168), (7168, 2048))
    ],
    # FP8(A) x FP4(B), gran_k_b = 32.
    *[_case(4096, n, k, b_dtype="fp4") for n, k in ((4096, 7168), (24576, 1536), (7168, 16384))],
]

BENCH_CONFIGS = [
    *[_case(4096, n, k) for n, k in _NK],
    *[_case(4096, n, k, b_dtype="fp4") for n, k in ((4096, 7168), (24576, 1536), (7168, 16384))],
]

_MAJOR = {"k": Major.K, "mn": Major.MN}


def make_desc(
    *,
    M: int,
    N: int,
    K: int,
    major_a: str = "k",
    major_b: str = "k",
    accumulate: bool = False,
    cd_dtype: str = "bf16",
    b_dtype: str = "fp8",
    num_sms: int | None = None,
) -> GemmDesc:
    """Build the `GemmDesc` DeepGEMM's host entry would build for this config."""
    if num_sms is None:
        import torch

        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    return GemmDesc(
        gemm_type=GemmType.NORMAL,
        m=M,
        n=N,
        k=K,
        num_groups=1,
        b_dtype=b_dtype,
        cd_dtype=cd_dtype,
        major_a=_MAJOR[major_a],
        major_b=_MAJOR[major_b],
        with_accumulation=accumulate,
        num_sms=num_sms,
    )


def get_kernel(**config):
    """Return the TIRx `PrimFunc` for one config."""
    from ._sm100_fp8_fp4_gemm_1d1d import build_kernel

    config.pop("label", None)
    return build_kernel(_spec_for(config))


def _spec_for(config: dict):
    """The `GemmSpec` for one config."""
    from ._sm100_fp8_fp4_gemm_1d1d import gran_k_for, make_spec

    desc = make_desc(**config)
    gran_k_a, gran_k_b = gran_k_for(desc.b_dtype)
    return make_spec(
        desc, get_best_config(desc), gran_k_a=gran_k_a, gran_k_b=gran_k_b, k_alignment=gran_k_a
    )


def prepare_data(*, seed: int = 0, **config):
    """Build one dense-GEMM case; see `_sm100_fp8_fp4_gemm_1d1d_data.prepare_normal`."""
    from ._sm100_fp8_fp4_gemm_1d1d_data import prepare_normal

    config.pop("label", None)
    return prepare_normal(seed=seed, **config)


def _tirx_launch(data, config):
    from ._sm100_fp8_fp4_gemm_1d1d import build_launch

    spec = _spec_for(config)
    return build_launch(
        spec,
        a=data["a"],
        b=data["b"],
        sfa=data["sfa"],
        sfb=data["sfb"],
        d=data["d"],
        shape_m=data["M"],
        shape_n=data["N"],
        shape_k=data["K"],
    )


def run_test(**config):
    """Compile, launch and compare against the dequantized-matmul oracle."""
    import torch

    from ._sm100_fp8_fp4_gemm_1d1d_data import assert_within_threshold, calc_diff

    config.pop("label", None)
    data = prepare_data(seed=17, **config)
    launch = _tirx_launch(data, config)
    data["d"].zero_()
    if data["c"] is not None:
        # The accumulate path stores with `cp.reduce ... add`, so the output must
        # start out holding C.
        data["d"].copy_(data["c"])
    launch()
    torch.cuda.synchronize()

    return assert_within_threshold(
        calc_diff(data["d"], data["ref"]),
        data,
        kernel="deepgemm_sm100_fp8_gemm_1d1d",
        detail=(
            f"M={data['M']} N={data['N']} K={data['K']} "
            f"major={data['major_a']}{data['major_b']} b_dtype={data['b_dtype']} "
            f"cd={data['cd_dtype']} acc={data['accumulate']}"
        ),
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    """Time our launch against `deep_gemm.fp8_fp4_gemm_nt`."""
    from ._sm100_fp8_fp4_gemm_1d1d_data import bench_against_deepgemm, deepgemm_launch_normal

    config.pop("label", None)
    data = prepare_data(seed=17, **config)
    return bench_against_deepgemm(
        _tirx_launch(data, config),
        deepgemm_launch_normal,
        data,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
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
]
