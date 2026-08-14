# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Fused MoE megakernel (MegaMoE) -- fp8 activations, fp4 weights, SM100.

Reference: ``deep_gemm.fp8_fp4_mega_moe``.  The implementation lives in
:mod:`tirx_kernels.deepgemm._sm100_fp8_fp4_mega_moe`; this module pins the test
and benchmark matrices and the registry surface.

Upstream sources: deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh, csrc/apis/mega.h,
csrc/jit_kernels/heuristics/mega_moe.h.
"""

from __future__ import annotations

from typing import Any
from unittest import SkipTest

from ._sm100_fp8_fp4_mega_moe.data import _run_distributed
from ._sm100_fp8_fp4_mega_moe.kernel import get_kernel as get_kernel
from ._sm100_fp8_fp4_mega_moe.spec import MegaMoeConfig
from ._sm100_fp8_fp4_mega_moe.spec import fp8_fp4_mega_moe as fp8_fp4_mega_moe
from ._sm100_fp8_fp4_mega_moe.spec import (
    launch_prepared_tirx_fp8_fp4_mega_moe as launch_prepared_tirx_fp8_fp4_mega_moe,
)
from ._sm100_fp8_fp4_mega_moe.spec import (
    prepare_tirx_fp8_fp4_mega_moe as prepare_tirx_fp8_fp4_mega_moe,
)

KERNEL_META = {"name": "sm100_fp8_fp4_mega_moe", "category": "deepgemm", "compute_capability": 10}

# One case per block_m bucket in `_get_block_config_for_mega_moe` so a per-PR
# sm100a run covers all heuristic-selected block_m paths (16, 32, 64, 96, 128, 192).
# Each tpe (= tokens * ranks * topk / experts) is set just below the bucket boundary.
CONFIGS = [
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 4,
        "num_tokens": 2,
        "hidden": 1024,
        "intermediate_hidden": 512,
        "num_experts": 2,
        "num_topk": 1,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "p1_tok2_h1024_i512_e2_k1_bm16",
    },
    *[
        {
            "num_processes": num_processes,
            "num_max_tokens_per_rank": 4,
            "num_tokens": 2,
            "hidden": 1024,
            "intermediate_hidden": 512,
            "num_experts": num_processes * 2,
            "num_topk": 1,
            "activation_clamp": 10.0,
            "fast_math": 1,
            "label": f"p{num_processes}_tok2_h1024_i512_e{num_processes * 2}_k1_bm16",
        }
        for num_processes in (2, 4, 6)
    ],
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 16,
        "num_tokens": 16,
        "hidden": 1024,
        "intermediate_hidden": 512,
        "num_experts": 2,
        "num_topk": 2,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "p1_tok16_h1024_i512_e2_k2_bm32",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 32,
        "num_tokens": 32,
        "hidden": 1024,
        "intermediate_hidden": 512,
        "num_experts": 2,
        "num_topk": 2,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "p1_tok32_h1024_i512_e2_k2_bm64",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 64,
        "num_tokens": 64,
        "hidden": 1024,
        "intermediate_hidden": 512,
        "num_experts": 2,
        "num_topk": 2,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "p1_tok64_h1024_i512_e2_k2_bm96",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 96,
        "num_tokens": 96,
        "hidden": 1024,
        "intermediate_hidden": 512,
        "num_experts": 2,
        "num_topk": 2,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "p1_tok96_h1024_i512_e2_k2_bm128",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 192,
        "num_tokens": 192,
        "hidden": 1024,
        "intermediate_hidden": 512,
        "num_experts": 2,
        "num_topk": 2,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "p1_tok192_h1024_i512_e2_k2_bm192",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 64,
        "num_tokens": 64,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t64_m64_h7168_i3072_e384_k6_g1",
    },
    {
        "num_processes": 2,
        "num_max_tokens_per_rank": 64,
        "num_tokens": 64,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t64_m64_h7168_i3072_e384_k6_g2",
    },
    {
        "num_processes": 2,
        "num_max_tokens_per_rank": 8192,
        "num_tokens": 8192,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t8192_m8192_h7168_i3072_e384_k6_g2",
    },
    {
        "num_processes": 4,
        "num_max_tokens_per_rank": 64,
        "num_tokens": 64,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t64_m64_h7168_i3072_e384_k6_g4",
    },
    {
        "num_processes": 4,
        "num_max_tokens_per_rank": 8192,
        "num_tokens": 8192,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t8192_m8192_h7168_i3072_e384_k6_g4",
    },
    {
        "num_processes": 6,
        "num_max_tokens_per_rank": 64,
        "num_tokens": 64,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t64_m64_h7168_i3072_e384_k6_g6",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 8,
        "num_tokens": 8,
        "hidden": 1024,
        "intermediate_hidden": 512,
        "num_experts": 24,
        "num_topk": 2,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t8_h1024_i512_e24_k2_g1",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 16,
        "num_tokens": 16,
        "hidden": 2048,
        "intermediate_hidden": 1024,
        "num_experts": 48,
        "num_topk": 2,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t16_h2048_i1024_e48_k2_g1",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 64,
        "num_tokens": 64,
        "hidden": 4096,
        "intermediate_hidden": 1536,
        "num_experts": 96,
        "num_topk": 4,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t64_h4096_i1536_e96_k4_g1",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 64,
        "num_tokens": 32,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t32_m64_h7168_i3072_e384_k6_g1",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 8192,
        "num_tokens": 8192,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 64,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t8192_h7168_i3072_e64_k6_g1",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 8192,
        "num_tokens": 8192,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 192,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t8192_h7168_i3072_e192_k6_g1",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 8192,
        "num_tokens": 8192,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t8192_m8192_h7168_i3072_e384_k6_g1",
    },
    {
        "num_processes": 4,
        "num_max_tokens_per_rank": 256,
        "num_tokens": 256,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t256_h7168_i3072_e384_k6_g4",
    },
    {
        "num_processes": 4,
        "num_max_tokens_per_rank": 1024,
        "num_tokens": 1024,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t1024_h7168_i3072_e384_k6_g4",
    },
    {
        "num_processes": 6,
        "num_max_tokens_per_rank": 8192,
        "num_tokens": 8192,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t8192_m8192_h7168_i3072_e384_k6_g6",
    },
]


def _make_config(
    num_processes=1,
    num_max_tokens_per_rank=4,
    num_tokens=2,
    hidden=1024,
    intermediate_hidden=512,
    num_experts=2,
    num_topk=1,
    activation_clamp=10.0,
    fast_math=1,
) -> MegaMoeConfig:
    config = MegaMoeConfig(
        num_processes=num_processes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    config.validate()
    return config


def prepare_data(
    num_processes=1,
    num_max_tokens_per_rank=4,
    num_tokens=2,
    hidden=1024,
    intermediate_hidden=512,
    num_experts=2,
    num_topk=1,
    activation_clamp=10.0,
    fast_math=1,
) -> dict[str, Any]:
    return {
        "config": _make_config(
            num_processes=num_processes,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_tokens=num_tokens,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            num_experts=num_experts,
            num_topk=num_topk,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
    }


def _assert_correctness_result(result: dict[str, Any]) -> None:
    if result["status"] == "SKIP":
        raise SkipTest(
            f"{result['reason']} DeepGEMM reference source={result['reference_source']} "
            f"checksum={result['reference_checksum']:.4f}"
        )
    assert result["deepgemm_max_abs_diff"] == 0.0, (
        f"Expected bitwise parity with DeepGEMM reference, got deepgemm_max_abs_diff="
        f"{result['deepgemm_max_abs_diff']}"
    )
    assert result["stats_max_abs_diff"] == 0, (
        "Expected cumulative_local_expert_recv_stats parity with DeepGEMM, got "
        f"stats_max_abs_diff={result['stats_max_abs_diff']}"
    )


def check_correctness(
    outputs: dict[str, Any],
    num_processes=1,
    num_max_tokens_per_rank=4,
    num_tokens=2,
    hidden=1024,
    intermediate_hidden=512,
    num_experts=2,
    num_topk=1,
    activation_clamp=10.0,
    fast_math=1,
) -> None:
    result = outputs.get("result")
    if result is None:
        result = _run_distributed(
            _make_config(
                num_processes=num_processes,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                num_tokens=num_tokens,
                hidden=hidden,
                intermediate_hidden=intermediate_hidden,
                num_experts=num_experts,
                num_topk=num_topk,
                activation_clamp=activation_clamp,
                fast_math=fast_math,
            ),
            "test",
        )
    _assert_correctness_result(result)


def run_test(
    num_processes=1,
    num_max_tokens_per_rank=4,
    num_tokens=2,
    hidden=1024,
    intermediate_hidden=512,
    num_experts=2,
    num_topk=1,
    activation_clamp=10.0,
    fast_math=1,
):
    config = _make_config(
        num_processes=num_processes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    result = _run_distributed(config, "test")
    _assert_correctness_result(result)


def run_bench(
    num_processes=1,
    num_max_tokens_per_rank=128,
    num_tokens=96,
    hidden=1024,
    intermediate_hidden=512,
    num_experts=8,
    num_topk=2,
    activation_clamp=10.0,
    fast_math=1,
    *,
    warmup=None,
    repeat=None,
    timer=None,
    **kwargs,
):
    config = _make_config(
        num_processes=num_processes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    return _run_distributed(config, "bench", warmup=warmup, repeat=repeat, timer=timer, **kwargs)
