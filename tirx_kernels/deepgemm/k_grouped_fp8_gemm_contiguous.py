# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025
# DeepSeek, licensed under the MIT License. The upstream sources carry no
# per-file license header; see licenses/LICENSE.deepgemm.txt for the full
# license text.
#
# Modifications Copyright (c) 2026 The TIRx Authors.
# Modifications are licensed under the Apache License, Version 2.0.
#
# TIRx port of DeepGEMM's sm100_k_grouped_fp8_gemm_1d1d host entry
# (csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:313).
# See LICENSE, NOTICE, and licenses/ for the applicable terms.

"""K-grouped contiguous FP8 GEMM -- `KGroupedContiguous[WithPsumLayout]`.

Reference: `deep_gemm.k_grouped_fp8_gemm_tn_contiguous`.  Groups are concatenated
along K with per-group lengths, both operands are MN-major, and the output is
always FP32 with accumulation (the MoE weight-gradient path), so the epilogue
takes the `SM90_TMA_REDUCE_ADD` branch.  FP4 is not exercised here -- DeepGEMM's
own enumerator keeps this entry FP8-only.
"""

from __future__ import annotations

from ._sm100_fp8_fp4_gemm_1d1d import (
    GemmDesc,
    GemmType,
    Major,
    align_up,
    ceil_div,
    get_best_config,
    make_actual_ks,
)

KERNEL_META = {
    "name": "deepgemm_sm100_k_grouped_fp8_gemm_contiguous",
    "category": "deepgemm",
    "compute_capability": 10,
}

#: `tests/generators.py::enumerate_k_grouped_contiguous`; the first entry has
#: many small groups, which stresses the scheduler's group walk.
_SHAPES = [
    (8, 768, 2048, 128),
    (4, 4096, 7168, 8192),
    (8, 4096, 7168, 4096),
    (16, 7168, 2048, 2048),
]
#: `(gran_k, k_alignment)`; 160 and 224 are not multiples of BLOCK_K, so they are
#: the pairs that turn on `kMayHaveTailKBlock`.
_QUANT = [(32, 32), (32, 160), (128, 128), (128, 224)]


def _case(
    num_groups: int,
    m: int,
    n: int,
    expected_k_per_group: int,
    gran_k: int,
    k_alignment: int,
    *,
    use_psum: bool = False,
    empty_group: bool = False,
    short_first_group: bool = False,
    seed: int = 0,
) -> dict:
    label = f"g{num_groups}_m{m}_n{n}_k{expected_k_per_group}_gran{gran_k}_al{k_alignment}"
    if use_psum:
        label += "_psum"
    if empty_group:
        label += "_empty"
    if short_first_group:
        label += "_shorthead"
    return {
        "label": label,
        "num_groups": num_groups,
        "M": m,
        "N": n,
        "expected_k_per_group": expected_k_per_group,
        "gran_k": gran_k,
        "k_alignment": k_alignment,
        "use_psum_layout": use_psum,
        "empty_group": empty_group,
        "short_first_group": short_first_group,
        "seed": seed,
    }


CONFIGS = [
    # One quant option per shape, cycling so both granularities and both
    # tail-inducing alignments are covered.
    *[
        _case(g, m, n, k, gran, align, seed=i)
        for i, ((g, m, n, k), (gran, align)) in enumerate(zip(_SHAPES, _QUANT), start=1)
    ],
    # PSUM layout on two shapes.
    _case(4, 4096, 7168, 8192, 32, 160, use_psum=True, seed=11),
    _case(8, 4096, 7168, 4096, 128, 128, use_psum=True, seed=12),
    # Degenerate group layouts (`enumerate_k_grouped_contiguous_test_variants`).
    _case(8, 768, 2048, 128, 32, 32, empty_group=True, seed=21),
    _case(16, 7168, 2048, 2048, 128, 224, short_first_group=True, seed=22),
    # Tail-K with the largest alignment on a compute-heavy shape.
    _case(4, 4096, 7168, 8192, 128, 224, seed=31),
    _case(8, 4096, 7168, 4096, 32, 160, use_psum=True, short_first_group=True, seed=32),
]

BENCH_CONFIGS = [
    _case(4, 4096, 7168, 8192, 32, 32, seed=1),
    _case(4, 4096, 7168, 8192, 128, 128, seed=2),
    _case(8, 4096, 7168, 4096, 32, 160, seed=3),
    _case(8, 4096, 7168, 4096, 128, 128, seed=4),
    _case(16, 7168, 2048, 2048, 128, 128, seed=5),
    _case(4, 4096, 7168, 8192, 128, 128, use_psum=True, seed=6),
]


def make_ks(
    *,
    num_groups: int,
    expected_k_per_group: int,
    k_alignment: int,
    seed: int,
    empty_group: bool = False,
    short_first_group: bool = False,
) -> tuple[list[int], list[int]]:
    """Per-group real and aligned K lengths for one config."""
    real_ks = make_actual_ks(num_groups, expected_k_per_group, seed)
    if empty_group and num_groups > 1:
        real_ks[seed % num_groups] = 0
    if short_first_group and num_groups > 1:
        real_ks[0] = k_alignment
    return real_ks, [align_up(k, k_alignment) for k in real_ks]


def make_desc(
    *,
    num_groups: int,
    M: int,
    N: int,
    expected_k_per_group: int,
    gran_k: int,
    k_alignment: int,
    use_psum_layout: bool = False,
    empty_group: bool = False,
    short_first_group: bool = False,
    seed: int = 0,
    num_sms: int | None = None,
) -> GemmDesc:
    """Build the descriptor `sm100_k_grouped_fp8_gemm_1d1d` would build."""
    if num_sms is None:
        import torch

        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    _, aligned_ks = make_ks(
        num_groups=num_groups,
        expected_k_per_group=expected_k_per_group,
        k_alignment=k_alignment,
        seed=seed,
        empty_group=empty_group,
        short_first_group=short_first_group,
    )
    sum_k = sum(aligned_ks)
    return GemmDesc(
        gemm_type=(
            GemmType.K_GROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT
            if use_psum_layout
            else GemmType.K_GROUPED_CONTIGUOUS
        ),
        m=M,
        n=N,
        k=sum_k,
        num_groups=num_groups,
        cd_dtype="fp32",
        major_a=Major.MN,
        major_b=Major.MN,
        with_accumulation=True,
        num_sms=num_sms,
        expected_m=M,
        expected_n=N,
        expected_k=ceil_div(sum_k, num_groups),
        expected_num_groups=num_groups,
        mk_alignment=k_alignment,
    )


def get_kernel(**config):
    from ._sm100_fp8_fp4_gemm_1d1d import build_kernel

    config.pop("label", None)
    return build_kernel(_spec_for(config))


def _spec_for(config: dict):
    from ._sm100_fp8_fp4_gemm_1d1d import make_spec

    desc = make_desc(**config)
    gran_k = config["gran_k"]
    return make_spec(
        desc,
        get_best_config(desc),
        gran_k_a=gran_k,
        gran_k_b=gran_k,
        k_alignment=config["k_alignment"],
    )


def prepare_data(**config):
    from ._sm100_fp8_fp4_gemm_1d1d_data import prepare_k_grouped

    config.pop("label", None)
    return prepare_k_grouped(**config)


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
    )


def run_test(**config):
    """Compile, launch and compare against the dequantized-matmul oracle."""
    import torch

    from ._sm100_fp8_fp4_gemm_1d1d_data import assert_within_threshold, calc_diff

    config.pop("label", None)
    data = prepare_data(**config)
    launch = _tirx_launch(data, config)
    # The k-grouped store always accumulates, so D starts out holding C.
    data["d"].copy_(data["c"])
    launch()
    torch.cuda.synchronize()

    return assert_within_threshold(
        calc_diff(data["d"], data["ref"]),
        data,
        kernel="deepgemm_sm100_k_grouped_fp8_gemm_contiguous",
        detail=(
            f"g={data['num_groups']} M={data['M']} N={data['N']} K={data['K']} "
            f"gran={data['gran_k_a']} align={data['k_alignment']} "
            f"psum={data['use_psum_layout']}"
        ),
        K=data["K"],
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    from ._sm100_fp8_fp4_gemm_1d1d_data import bench_against_deepgemm, deepgemm_launch_k_grouped

    config.pop("label", None)
    data = prepare_data(**config)
    return bench_against_deepgemm(
        _tirx_launch(data, config),
        deepgemm_launch_k_grouped,
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
    "make_ks",
    "prepare_data",
    "run_bench",
    "run_test",
]
