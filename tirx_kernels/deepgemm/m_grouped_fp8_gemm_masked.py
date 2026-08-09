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
# TIRx port of DeepGEMM's `sm100_m_grouped_fp8_fp4_gemm_masked_1d1d` host entry
# (`csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:244`).
# See NOTICE and THIRD_PARTY_LICENSES.md for upstream attribution.

"""M-grouped masked FP8/FP4 GEMM -- `GemmType::MGroupedMasked`.

Reference: `deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked` (aliased as
`m_grouped_fp8_gemm_nt_masked` / `fp8_m_grouped_gemm_nt_masked`).  A is
`[G, max_m, K]` with a per-group valid row count in `masked_m`, so the shape is
static while the work is dynamic -- the decode / CUDA-graph MoE path.
"""

from __future__ import annotations

from ._sm100_fp8_fp4_gemm_1d1d import (
    GemmDesc,
    GemmType,
    get_best_config,
    get_theoretical_mk_alignment,
)

KERNEL_META = {
    "name": "deepgemm_sm100_m_grouped_fp8_gemm_masked",
    "category": "deepgemm",
    "compute_capability": 10,
}

MAX_M = 4096

#: `tests/generators.py::enumerate_m_grouped_masked`.
_GROUPS = [(32, 192), (6, 1024), (32, 20), (6, 20)]
_NK = [(6144, 7168), (7168, 3072), (4096, 4096), (4096, 2048)]


def _case(
    num_groups: int,
    expected_m_per_group: int,
    n: int,
    k: int,
    *,
    b_dtype: str = "fp8",
    seed: int = 0,
) -> dict:
    label = f"g{num_groups}_m{expected_m_per_group}_n{n}_k{k}"
    if b_dtype != "fp8":
        label += f"_b{b_dtype}"
    return {
        "label": label,
        "num_groups": num_groups,
        "expected_m_per_group": expected_m_per_group,
        "N": n,
        "K": k,
        "b_dtype": b_dtype,
        "seed": seed,
    }


CONFIGS = [
    # Two (n, k) shapes for each of the four group profiles; the `m=20` rows are
    # the tiny-M boundary where `get_theoretical_mk_alignment` steps block M down.
    *[
        _case(g, m, n, k, seed=i)
        for i, ((g, m), (n, k)) in enumerate(
            [(gm, nk) for gm in _GROUPS for nk in (_NK[0], _NK[3])], start=1
        )
    ],
    _case(32, 192, 4096, 4096, b_dtype="fp4", seed=21),
    _case(6, 1024, 7168, 3072, b_dtype="fp4", seed=22),
]

BENCH_CONFIGS = [
    # Tiny-M profiles stay correctness-only: they are launch-bound and their
    # ratios are dominated by noise rather than by kernel quality.
    *[
        _case(g, m, n, k, seed=i)
        for i, ((g, m), (n, k)) in enumerate(
            [(gm, nk) for gm in ((32, 192), (6, 1024)) for nk in (_NK[0], _NK[3])], start=1
        )
    ],
    _case(32, 192, 4096, 4096, b_dtype="fp4", seed=21),
]


def make_desc(
    *,
    num_groups: int,
    expected_m_per_group: int,
    N: int,
    K: int,
    b_dtype: str = "fp8",
    seed: int = 0,  # absorbed from the config; only `prepare_data` uses it
    num_sms: int | None = None,
) -> GemmDesc:
    """Build the descriptor `m_grouped_fp8_fp4_gemm_nt_masked` would build.

    DeepGEMM's test passes `expected_m = int(expected_m_per_group * 1.2)` and
    selects the block-M alignment from it (`test_fp8_fp4.py:137`).
    """
    if num_sms is None:
        import torch

        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    expected_m = int(expected_m_per_group * 1.2)
    return GemmDesc(
        gemm_type=GemmType.M_GROUPED_MASKED,
        m=MAX_M,
        n=N,
        k=K,
        num_groups=num_groups,
        b_dtype=b_dtype,
        num_sms=num_sms,
        expected_m=expected_m,
        expected_n=N,
        expected_k=K,
        expected_num_groups=num_groups,
        mk_alignment=get_theoretical_mk_alignment(expected_m),
    )


def get_kernel(**config):
    from ._sm100_fp8_fp4_gemm_1d1d import build_kernel

    config.pop("label", None)
    return build_kernel(_spec_for(config))


def _spec_for(config: dict):
    from ._sm100_fp8_fp4_gemm_1d1d import gran_k_for, make_spec

    desc = make_desc(**config)
    gran_k_a, gran_k_b = gran_k_for(desc.b_dtype)
    return make_spec(
        desc, get_best_config(desc), gran_k_a=gran_k_a, gran_k_b=gran_k_b, k_alignment=gran_k_a
    )


def prepare_data(**config):
    from ._sm100_fp8_fp4_gemm_1d1d_data import prepare_m_grouped_masked

    config.pop("label", None)
    return prepare_m_grouped_masked(max_m=MAX_M, **config)


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
        grouped_layout=data["masked_m"],
        num_groups=data["num_groups"],
        sf_num_groups_a=data["num_groups"],
        sf_num_groups_b=data["num_groups"],
    )


def run_test(**config):
    """Compare only the valid rows of each group against the oracle."""
    import torch

    from ._sm100_fp8_fp4_gemm_1d1d_data import assert_within_threshold, masked_slice_diff

    config.pop("label", None)
    data = prepare_data(**config)
    launch = _tirx_launch(data, config)
    data["d"].zero_()
    launch()
    torch.cuda.synchronize()

    return assert_within_threshold(
        masked_slice_diff(data["d"], data["ref"], data["masked_m"]),
        data,
        kernel="deepgemm_sm100_m_grouped_fp8_gemm_masked",
        detail=(f"g={data['num_groups']} N={data['N']} K={data['K']} b_dtype={data['b_dtype']}"),
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    from ._sm100_fp8_fp4_gemm_1d1d_data import (
        bench_against_deepgemm,
        deepgemm_launch_m_grouped_masked,
    )

    config.pop("label", None)
    data = prepare_data(**config)
    return bench_against_deepgemm(
        _tirx_launch(data, config),
        deepgemm_launch_m_grouped_masked,
        data,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
        N=data["N"],
        K=data["K"],
        num_groups=data["num_groups"],
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "MAX_M",
    "get_kernel",
    "make_desc",
    "prepare_data",
    "run_bench",
    "run_test",
]
