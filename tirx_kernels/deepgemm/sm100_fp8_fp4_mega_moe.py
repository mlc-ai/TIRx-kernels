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


def _case(
    label: str, *, tok: int, h: int, i: int, e: int, k: int, g: int = 1, max_tok: int | None = None
) -> dict:
    """One config: `g` ranks routing `tok` tokens over `e` experts, top-`k`.

    `max_tok` is the per-rank capacity when it differs from the token count.
    Every case runs with the upstream default clamp and fast-math setting.
    """
    return {
        "num_processes": g,
        "num_max_tokens_per_rank": tok if max_tok is None else max_tok,
        "num_tokens": tok,
        "hidden": h,
        "intermediate_hidden": i,
        "num_experts": e,
        "num_topk": k,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": label,
    }


# The first nine cases cover one block_m bucket each from
# `_get_block_config_for_mega_moe` (16, 32, 64, 96, 128, 192), so a per-PR
# sm100a run exercises every heuristic-selected path; each token count sits
# just below its bucket boundary. The rest sweep production shapes.
CONFIGS = [
    _case("p1_tok2_h1024_i512_e2_k1_bm16", tok=2, max_tok=4, h=1024, i=512, e=2, k=1),
    _case("p2_tok2_h1024_i512_e4_k1_bm16", g=2, tok=2, max_tok=4, h=1024, i=512, e=4, k=1),
    _case("p4_tok2_h1024_i512_e8_k1_bm16", g=4, tok=2, max_tok=4, h=1024, i=512, e=8, k=1),
    _case("p6_tok2_h1024_i512_e12_k1_bm16", g=6, tok=2, max_tok=4, h=1024, i=512, e=12, k=1),
    _case("p1_tok16_h1024_i512_e2_k2_bm32", tok=16, h=1024, i=512, e=2, k=2),
    _case("p1_tok32_h1024_i512_e2_k2_bm64", tok=32, h=1024, i=512, e=2, k=2),
    _case("p1_tok64_h1024_i512_e2_k2_bm96", tok=64, h=1024, i=512, e=2, k=2),
    _case("p1_tok96_h1024_i512_e2_k2_bm128", tok=96, h=1024, i=512, e=2, k=2),
    _case("p1_tok192_h1024_i512_e2_k2_bm192", tok=192, h=1024, i=512, e=2, k=2),
    _case("t64_m64_h7168_i3072_e384_k6_g1", tok=64, h=7168, i=3072, e=384, k=6),
    _case("t64_m64_h7168_i3072_e384_k6_g2", g=2, tok=64, h=7168, i=3072, e=384, k=6),
    _case("t8192_m8192_h7168_i3072_e384_k6_g2", g=2, tok=8192, h=7168, i=3072, e=384, k=6),
    _case("t64_m64_h7168_i3072_e384_k6_g4", g=4, tok=64, h=7168, i=3072, e=384, k=6),
    _case("t8192_m8192_h7168_i3072_e384_k6_g4", g=4, tok=8192, h=7168, i=3072, e=384, k=6),
    _case("t64_m64_h7168_i3072_e384_k6_g6", g=6, tok=64, h=7168, i=3072, e=384, k=6),
    _case("t8_h1024_i512_e24_k2_g1", tok=8, h=1024, i=512, e=24, k=2),
    _case("t16_h2048_i1024_e48_k2_g1", tok=16, h=2048, i=1024, e=48, k=2),
    _case("t64_h4096_i1536_e96_k4_g1", tok=64, h=4096, i=1536, e=96, k=4),
    _case("t32_m64_h7168_i3072_e384_k6_g1", tok=32, max_tok=64, h=7168, i=3072, e=384, k=6),
    _case("t8192_h7168_i3072_e64_k6_g1", tok=8192, h=7168, i=3072, e=64, k=6),
    _case("t8192_h7168_i3072_e192_k6_g1", tok=8192, h=7168, i=3072, e=192, k=6),
    _case("t8192_m8192_h7168_i3072_e384_k6_g1", tok=8192, h=7168, i=3072, e=384, k=6),
    _case("t256_h7168_i3072_e384_k6_g4", g=4, tok=256, h=7168, i=3072, e=384, k=6),
    _case("t1024_h7168_i3072_e384_k6_g4", g=4, tok=1024, h=7168, i=3072, e=384, k=6),
    _case("t8192_m8192_h7168_i3072_e384_k6_g6", g=6, tok=8192, h=7168, i=3072, e=384, k=6),
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
