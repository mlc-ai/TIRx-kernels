# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025
# DeepSeek, licensed under the MIT License. The upstream sources carry no
# per-file license header; see licenses/LICENSE.deepgemm.txt for the full
# license text.
#
# Modifications Copyright (c) 2026 The TIRx Authors.
# Modifications are licensed under the Apache License, Version 2.0.
#
# TIRx port of DeepGEMM's sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d host
# entry (csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:161).
# See LICENSE, NOTICE, and licenses/ for the applicable terms.

"""M-grouped contiguous FP8/FP4 GEMM -- `MGroupedContiguous[WithPsumLayout]`.

Reference: `deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous` (aliased as
`m_grouped_fp8_gemm_nt_contiguous`).  `use_psum_layout` switches the descriptor
to `GemmType::MGroupedContiguousWithPsumLayout`, which makes the scheduler walk
per-group psum end offsets instead of a per-row expert id, and makes
`ensure_zero_padding` meaningful (it gates the aligned-effective-M tail).
"""

from __future__ import annotations

from ._sm100_fp8_fp4_gemm_1d1d import (
    GemmDesc,
    GemmType,
    Major,
    get_best_config,
    get_theoretical_mk_alignment,
    make_aligned_ms,
)

KERNEL_META = {
    "name": "deepgemm_sm100_m_grouped_fp8_gemm_contiguous",
    "category": "deepgemm",
    "compute_capability": 10,
}

#: `tests/generators.py::enumerate_m_grouped_contiguous`.  These are also the
#: shapes the retired `grouped_fp8_gemm_contiguous` benchmarked.
_GROUPS = [(4, 8192), (8, 4096)]
_NK = [(6144, 7168), (7168, 3072), (4096, 4096), (4096, 2048)]


def _case(
    num_groups: int,
    expected_m_per_group: int,
    n: int,
    k: int,
    *,
    major_b: str = "k",
    use_psum: bool = False,
    ensure_zero_padding: bool = False,
    b_dtype: str = "fp8",
    seed: int = 0,
) -> dict:
    label = f"g{num_groups}_m{expected_m_per_group}_n{n}_k{k}"
    if major_b != "k":
        label += f"_k{major_b}"
    if use_psum:
        label += "_psum" + ("_zp" if ensure_zero_padding else "")
    if b_dtype != "fp8":
        label += f"_b{b_dtype}"
    return {
        "label": label,
        "num_groups": num_groups,
        "expected_m_per_group": expected_m_per_group,
        "N": n,
        "K": k,
        "major_b": major_b,
        "use_psum_layout": use_psum,
        "ensure_zero_padding": ensure_zero_padding,
        "b_dtype": b_dtype,
        "seed": seed,
    }


CONFIGS = [
    # NT, FP8 x FP8: the full group/shape sweep.
    *[
        _case(g, m, n, k, seed=i)
        for i, ((g, m), (n, k)) in enumerate([(gm, nk) for gm in _GROUPS for nk in _NK], start=1)
    ],
    # NN: B is MN-major (A must stay K-major for every m-grouped GEMM).
    _case(4, 8192, 4096, 4096, major_b="mn", seed=11),
    # FP8(A) x FP4(B).
    _case(4, 8192, 6144, 7168, b_dtype="fp4", seed=12),
    _case(8, 4096, 4096, 2048, b_dtype="fp4", seed=13),
    # PSUM layout, both padding contracts.
    *[
        _case(g, m, n, k, use_psum=True, ensure_zero_padding=zp, seed=20 + i)
        for i, ((g, m), (n, k), zp) in enumerate(
            [
                ((4, 8192), (4096, 4096), False),
                ((4, 8192), (4096, 4096), True),
                ((8, 4096), (7168, 3072), False),
                ((8, 4096), (7168, 3072), True),
            ]
        )
    ],
]

BENCH_CONFIGS = [
    *[
        _case(g, m, n, k, seed=i)
        for i, ((g, m), (n, k)) in enumerate([(gm, nk) for gm in _GROUPS for nk in _NK], start=1)
    ],
    _case(4, 8192, 6144, 7168, b_dtype="fp4", seed=12),
    _case(8, 4096, 4096, 2048, b_dtype="fp4", seed=13),
    _case(4, 8192, 4096, 4096, use_psum=True, ensure_zero_padding=False, seed=20),
    _case(8, 4096, 7168, 3072, use_psum=True, ensure_zero_padding=True, seed=23),
]

_MAJOR = {"k": Major.K, "mn": Major.MN}


def make_desc(
    *,
    num_groups: int,
    expected_m_per_group: int,
    N: int,
    K: int,
    major_b: str = "k",
    use_psum_layout: bool = False,
    ensure_zero_padding: bool = False,
    b_dtype: str = "fp8",
    seed: int = 0,
    num_sms: int | None = None,
) -> GemmDesc:
    """Build the descriptor `m_grouped_fp8_fp4_gemm_nt_contiguous` would build.

    `expected_m_for_psum_layout` is never supplied by DeepGEMM's own test, so
    `expected_m` collapses to the total M and `expected_num_groups` to 1
    (`sm100_fp8_fp4_gemm_1d1d.hpp:193-195`).
    """
    if num_sms is None:
        import torch

        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    alignment = get_theoretical_mk_alignment()
    m = sum(make_aligned_ms(num_groups, expected_m_per_group, seed, alignment))
    return GemmDesc(
        gemm_type=(
            GemmType.M_GROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT
            if use_psum_layout
            else GemmType.M_GROUPED_CONTIGUOUS
        ),
        m=m,
        n=N,
        k=K,
        num_groups=num_groups,
        b_dtype=b_dtype,
        major_b=_MAJOR[major_b],
        num_sms=num_sms,
        ensure_zero_padding=ensure_zero_padding,
        expected_m=m,
        expected_n=N,
        expected_k=K,
        expected_num_groups=1,
        mk_alignment=alignment,
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
    from ._sm100_fp8_fp4_gemm_1d1d_data import prepare_m_grouped_contiguous

    config.pop("label", None)
    return prepare_m_grouped_contiguous(**config)


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
        grouped_layout=data["grouped_layout"],
        num_groups=data["num_groups"],
        sf_num_groups_b=data["num_groups"],
    )


def run_test(**config):
    """Compile, launch and compare against the dequantized-matmul oracle."""
    import torch

    from ._sm100_fp8_fp4_gemm_1d1d_data import assert_within_threshold, calc_diff, psum_slice_diff

    config.pop("label", None)
    data = prepare_data(**config)
    launch = _tirx_launch(data, config)
    data["d"].zero_()
    launch()
    torch.cuda.synchronize()

    if data["use_psum_layout"]:
        diff = psum_slice_diff(
            data["d"],
            data["ref"],
            data["grouped_layout"],
            data["alignment"],
            zero_padding=data["ensure_zero_padding"],
        )
    else:
        diff = calc_diff(data["d"], data["ref"])
    return assert_within_threshold(
        diff,
        data,
        kernel="deepgemm_sm100_m_grouped_fp8_gemm_contiguous",
        detail=(
            f"g={data['num_groups']} M={data['M']} N={data['N']} K={data['K']} "
            f"psum={data['use_psum_layout']} zp={data['ensure_zero_padding']} "
            f"b_dtype={data['b_dtype']}"
        ),
        M=data["M"],
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    from ._sm100_fp8_fp4_gemm_1d1d_data import (
        bench_against_deepgemm,
        deepgemm_launch_m_grouped_contiguous,
    )

    config.pop("label", None)
    data = prepare_data(**config)
    return bench_against_deepgemm(
        _tirx_launch(data, config),
        deepgemm_launch_m_grouped_contiguous,
        data,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
        M=data["M"],
        N=data["N"],
        K=data["K"],
        num_groups=data["num_groups"],
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
