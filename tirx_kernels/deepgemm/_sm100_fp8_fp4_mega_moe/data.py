# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Case construction, references, and the distributed bench harness for MegaMoE.

The kernel body lives in :mod:`.kernel`; configuration and launch plumbing in
:mod:`.spec`.

Upstream sources: deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh, csrc/apis/mega.h,
csrc/jit_kernels/heuristics/mega_moe.h.
"""

from __future__ import annotations

import inspect
import math
import os
import random
import socket
import time
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any
from unittest import SkipTest

import torch
import torch.multiprocessing as mp

from tirx_kernels import _torch_quant

from .spec import (
    MegaMoeCase,
    MegaMoeConfig,
    _align_up,
    _launch_tirx_mega_moe,
    _prepare_tirx_invocation,
    create_mega_moe_symm_buffer,
    fp8_fp4_mega_moe,
    get_deepgemm_launch_config,
    get_deepgemm_symm_buffer_layout,
    get_deepgemm_workspace_layout,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _distributed_env(port: int):
    old_master_addr = os.environ.get("MASTER_ADDR")
    old_master_port = os.environ.get("MASTER_PORT")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        yield
    finally:
        if old_master_addr is None:
            os.environ.pop("MASTER_ADDR", None)
        else:
            os.environ["MASTER_ADDR"] = old_master_addr
        if old_master_port is None:
            os.environ.pop("MASTER_PORT", None)
        else:
            os.environ["MASTER_PORT"] = old_master_port


def _cast_grouped_weights_to_fp4(
    bf16_weights: torch.Tensor, *, compute_reference: bool
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
    num_groups, n, k = bf16_weights.shape
    weights = []
    scales = []
    restored = []
    for group_idx in range(num_groups):
        weight, scale = _torch_quant.per_token_cast_to_fp4(bf16_weights[group_idx], gran_k=32)
        weights.append(weight)
        scales.append(scale)
        if compute_reference:
            restored.append(_torch_quant.cast_back_from_fp4(weight, scale, gran_k=32))
    packed_weights = torch.stack(weights, dim=0).contiguous()
    raw_scales = torch.stack(scales, dim=0).contiguous()
    transformed_scales = _torch_quant.transform_sf(
        raw_scales, mn=n, gran_mn=1, num_groups=num_groups
    )
    reference = torch.stack(restored) if compute_reference else None
    return (packed_weights, transformed_scales), reference


def _cast_fp8_for_mega_moe(
    x: torch.Tensor, *, compute_reference: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Construct the packed input and TMA scale layouts consumed by MegaMoE.

    The third entry is the TMA-strided scale view used by shared weights; the
    fourth is the dequantized Torch operand retained only for correctness.
    """
    x_fp8, scales = _torch_quant.per_token_cast_to_fp8(x, gran_k=32)
    x_sf = _torch_quant.pack_ue8m0_words(scales)
    mn, packed_sf_k = x_sf.shape
    x_sf_tma = torch.empty_strided(
        (mn, packed_sf_k), (1, _align_up(mn, 4)), dtype=x_sf.dtype, device=x_sf.device
    )
    x_sf_tma.copy_(x_sf)
    restored = (
        _torch_quant.cast_back_from_fp8(x_fp8, scales, gran_k=32) if compute_reference else None
    )
    return x_fp8, x_sf, x_sf_tma, restored


def _interleave_gate_up(tensor: torch.Tensor, granularity: int = 8) -> torch.Tensor:
    squeeze = tensor.ndim == 2
    source = tensor.unsqueeze(0) if squeeze else tensor
    groups, rows, *tail = source.shape
    half = rows // 2
    gate = source[:, :half].reshape(groups, half // granularity, granularity, *tail)
    up = source[:, half:].reshape(groups, half // granularity, granularity, *tail)
    logical = torch.stack((gate, up), dim=2).reshape_as(source)
    # Packed weights enter contiguous, while their scale tensors enter in the
    # MN-major TMA layout.  Preserve whichever physical strides the input owns.
    result = torch.empty_like(source).copy_(logical)
    return result.squeeze(0) if squeeze else result


def _transpose_sf_for_utccp(scales: torch.Tensor) -> torch.Tensor:
    squeeze = scales.ndim == 2
    source = scales.unsqueeze(0) if squeeze else scales
    groups, rows, packed_k = source.shape
    if rows % 128:
        raise ValueError("MegaMoE weight scale rows must be divisible by 128")
    logical = source.reshape(groups, -1, 4, 32, packed_k).transpose(2, 3).reshape_as(source)
    result = torch.empty_like(source).copy_(logical)
    return result.squeeze(0) if squeeze else result


def _transform_weights_for_mega_moe(
    l1: tuple[torch.Tensor, torch.Tensor], l2: tuple[torch.Tensor, torch.Tensor]
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    return (
        (_interleave_gate_up(l1[0]), _transpose_sf_for_utccp(_interleave_gate_up(l1[1]))),
        (l2[0], _transpose_sf_for_utccp(l2[1])),
    )


def _to_shared_mega_moe_sf_layout(
    sf: torch.Tensor, block_m: int, num_max_sf_tokens: int
) -> torch.Tensor:
    """Port of `_to_shared_mega_moe_sf_layout` (tests/test_mega_moe.py:36-50).

    The shared L1 input SF plane is host-written in the UTCCP-transposed,
    BLOCK_M-dependent layout; the kernel never writes it. Padding rows stay
    ZERO: the plane is allocated at full size, zeroed, and only ``[0,
    num_tokens)`` is filled.
    """
    num_tokens, packed_sf_k = sf.shape
    aligned_block_m = _align_up(block_m, 128)
    num_m_blocks = (num_tokens + block_m - 1) // block_m
    result = torch.empty_strided(
        (num_max_sf_tokens, packed_sf_k), (1, num_max_sf_tokens), dtype=sf.dtype, device=sf.device
    )
    result.zero_()
    for block_idx in range(num_m_blocks):
        num_block_tokens = min(block_m, num_tokens - block_idx * block_m)
        for m_idx in range(num_block_tokens):
            transposed_m_idx = (m_idx // 128) * 128 + (m_idx % 32) * 4 + (m_idx % 128) // 32
            result[block_idx * aligned_block_m + transposed_m_idx].copy_(
                sf[block_idx * block_m + m_idx]
            )
    return result


def _copy_fp8_sf(dst: torch.Tensor, src: torch.Tensor, num_tokens: int) -> None:
    """Port of `_copy_fp8_sf` (tests/test_mega_moe.py:62-70).

    At this call site ``dst.shape == src.shape`` (both are the full shared SF
    plane), so the plain copy path is taken and the replicate-last-row branch is
    dead; it is kept only to mirror the reference helper exactly.
    """
    if num_tokens == 0:
        return
    if dst.shape == src.shape:
        dst.copy_(src)
        return
    dst[:num_tokens].copy_(src)
    if num_tokens < dst.shape[0]:
        dst[num_tokens:].copy_(src[-1:].expand(dst.shape[0] - num_tokens, -1))


def create_case(
    config: MegaMoeConfig, group: Any, rank_idx: int, num_ranks: int, *, compute_reference: bool
) -> MegaMoeCase:
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    symm_buffer = create_mega_moe_symm_buffer(config, group)
    num_tokens = config.num_tokens
    num_experts_per_rank = config.num_experts // num_ranks

    x = torch.randn((num_tokens, config.hidden), dtype=torch.bfloat16, device="cuda")
    l1_weights = torch.randn(
        (num_experts_per_rank, config.intermediate_hidden * 2, config.hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    l2_weights = torch.randn(
        (num_experts_per_rank, config.hidden, config.intermediate_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    shared_intermediate_hidden = config.shared_intermediate_hidden
    if config.has_shared_experts:
        # One fused shared FFN of intermediate width S * I; no expert-group dim.
        shared_l1_weights_bf16 = torch.randn(
            (shared_intermediate_hidden * 2, config.hidden), dtype=torch.bfloat16, device="cuda"
        )
        shared_l2_weights_bf16 = torch.randn(
            (config.hidden, shared_intermediate_hidden), dtype=torch.bfloat16, device="cuda"
        )
    scores = torch.randn((num_tokens, config.num_experts), dtype=torch.float32, device="cuda")
    topk_weights, topk_idx = torch.topk(scores, config.num_topk, dim=-1, largest=True, sorted=False)

    x_fp8_data, x_sf, _x_sf_tma, reference_x = _cast_fp8_for_mega_moe(
        x, compute_reference=compute_reference
    )
    x_fp8 = (x_fp8_data, x_sf)
    transformed_l1_input, reference_l1_weights = _cast_grouped_weights_to_fp4(
        l1_weights, compute_reference=compute_reference
    )
    transformed_l2_input, reference_l2_weights = _cast_grouped_weights_to_fp4(
        l2_weights, compute_reference=compute_reference
    )
    transformed_l1_weights, transformed_l2_weights = _transform_weights_for_mega_moe(
        transformed_l1_input, transformed_l2_input
    )
    transformed_shared_l1_weights = None
    transformed_shared_l2_weights = None
    shared_l1_acts_sf = None
    reference_shared_l1_weights = None
    reference_shared_l2_weights = None
    if config.has_shared_experts:
        shared_l1_cast = _cast_fp8_for_mega_moe(
            shared_l1_weights_bf16, compute_reference=compute_reference
        )
        shared_l2_cast = _cast_fp8_for_mega_moe(
            shared_l2_weights_bf16, compute_reference=compute_reference
        )
        shared_l1_input = (shared_l1_cast[0], shared_l1_cast[2])
        shared_l2_input = (shared_l2_cast[0], shared_l2_cast[2])
        reference_shared_l1_weights = shared_l1_cast[3]
        reference_shared_l2_weights = shared_l2_cast[3]
        transformed_shared_l1_weights, transformed_shared_l2_weights = (
            _transform_weights_for_mega_moe(shared_l1_input, shared_l2_input)
        )
        # The shared L1 SF plane is host-written in a BLOCK_M-dependent layout.
        block_m = get_deepgemm_launch_config(config).block_m
        shared_l1_acts_sf = _to_shared_mega_moe_sf_layout(
            x_sf, block_m, symm_buffer.shared_l1_acts_sf.shape[0]
        )
    workspace_layout = get_deepgemm_workspace_layout(config)
    symm_buffer_layout = get_deepgemm_symm_buffer_layout(config)
    return MegaMoeCase(
        config=config,
        rank_idx=rank_idx,
        num_ranks=num_ranks,
        group=group,
        symm_buffer=symm_buffer,
        x_fp8=x_fp8,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        transformed_l1_weights=transformed_l1_weights,
        transformed_l2_weights=transformed_l2_weights,
        workspace_layout=workspace_layout,
        symm_buffer_layout=symm_buffer_layout,
        transformed_shared_l1_weights=transformed_shared_l1_weights,
        transformed_shared_l2_weights=transformed_shared_l2_weights,
        shared_l1_acts_sf=shared_l1_acts_sf,
        reference_x=reference_x,
        reference_l1_weights=reference_l1_weights,
        reference_l2_weights=reference_l2_weights,
        reference_shared_l1_weights=reference_shared_l1_weights,
        reference_shared_l2_weights=reference_shared_l2_weights,
    )


def _copy_inputs_into_symm_buffer(case: MegaMoeCase) -> None:
    num_tokens = case.config.num_tokens
    case.symm_buffer.x[:num_tokens].copy_(case.x_fp8[0])
    case.symm_buffer.x_sf[:num_tokens].copy_(case.x_fp8[1])
    if case.config.has_shared_experts:
        # Kernel-read, host-written: refreshed on every launch because debug mode
        # zeroes the whole symmetric buffer between calls.
        _copy_fp8_sf(case.symm_buffer.shared_l1_acts_sf, case.shared_l1_acts_sf, num_tokens)
    case.symm_buffer.topk_idx[:num_tokens].copy_(case.topk_idx)
    case.symm_buffer.topk_weights[:num_tokens].copy_(case.topk_weights)


def _quantized_swiglu(
    l1: torch.Tensor,
    *,
    intermediate_hidden: int,
    activation_clamp: float,
    route_weights: torch.Tensor | None,
) -> torch.Tensor:
    values = l1.to(torch.bfloat16).float()
    gate = values[:, :intermediate_hidden].clamp(max=activation_clamp)
    up = values[:, intermediate_hidden:].clamp(min=-activation_clamp, max=activation_clamp)
    activated = torch.nn.functional.silu(gate) * up
    if route_weights is not None:
        activated = activated * route_weights[:, None]
    quantized, scales = _torch_quant.per_token_cast_to_fp8(activated, gran_k=32)
    return _torch_quant.cast_back_from_fp8(quantized, scales, gran_k=32)


def _torch_ffn(
    x: torch.Tensor,
    l1_weights: torch.Tensor,
    l2_weights: torch.Tensor,
    *,
    intermediate_hidden: int,
    activation_clamp: float,
    route_weights: torch.Tensor | None,
) -> torch.Tensor:
    l1 = x.float() @ l1_weights.float().T
    l2_input = _quantized_swiglu(
        l1,
        intermediate_hidden=intermediate_hidden,
        activation_clamp=activation_clamp,
        route_weights=route_weights,
    )
    return l2_input @ l2_weights.float().T


def run_torch_reference(
    case: MegaMoeCase, initial_stats: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate routed and shared experts with the quantized operands TIRx consumes."""

    import torch.distributed as dist

    if any(
        value is None
        for value in (case.reference_x, case.reference_l1_weights, case.reference_l2_weights)
    ):
        raise RuntimeError("Torch reference operands were not retained for this case")

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        parts = [torch.empty_like(tensor) for _ in range(case.num_ranks)]
        dist.all_gather(parts, tensor, group=case.group)
        return torch.cat(parts, dim=0)

    all_x = gather(case.reference_x)
    all_topk_idx = gather(case.topk_idx)
    all_topk_weights = gather(case.topk_weights)
    result = torch.zeros((all_x.shape[0], case.config.hidden), dtype=torch.float32, device="cuda")

    local_expert_begin = case.rank_idx * case.config.num_experts_per_rank
    local_counts = torch.zeros(case.config.num_experts_per_rank, dtype=torch.int32, device="cuda")
    for local_idx in range(case.config.num_experts_per_rank):
        global_idx = local_expert_begin + local_idx
        token_idx, topk_slot = torch.where(all_topk_idx == global_idx)
        local_counts[local_idx] = token_idx.numel()
        if token_idx.numel() == 0:
            continue
        contribution = _torch_ffn(
            all_x.index_select(0, token_idx),
            case.reference_l1_weights[local_idx],
            case.reference_l2_weights[local_idx],
            intermediate_hidden=case.config.intermediate_hidden,
            activation_clamp=case.config.activation_clamp,
            route_weights=all_topk_weights[token_idx, topk_slot],
        )
        result.index_add_(0, token_idx, contribution)

    if case.config.has_shared_experts:
        if case.reference_shared_l1_weights is None or case.reference_shared_l2_weights is None:
            raise RuntimeError("Torch shared-expert operands were not retained for this case")
        begin = case.rank_idx * case.config.num_tokens
        end = begin + case.config.num_tokens
        result[begin:end] += _torch_ffn(
            all_x[begin:end],
            case.reference_shared_l1_weights,
            case.reference_shared_l2_weights,
            intermediate_hidden=case.config.shared_intermediate_hidden,
            activation_clamp=case.config.activation_clamp,
            route_weights=None,
        )

    dist.all_reduce(result, group=case.group)
    begin = case.rank_idx * case.config.num_tokens
    end = begin + case.config.num_tokens
    return result[begin:end].to(torch.bfloat16), initial_stats + local_counts


def _relative_l2_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm(lhs.float() - rhs.float())
    scale = torch.linalg.vector_norm(rhs.float()).clamp_min(1.0e-12)
    return float((difference / scale).item())


def run_tirx_mega_moe(
    case: MegaMoeCase, cumulative_local_expert_recv_stats: torch.Tensor | None = None
) -> torch.Tensor:
    _copy_inputs_into_symm_buffer(case)
    y = torch.empty(
        (case.config.num_tokens, case.config.hidden), dtype=torch.bfloat16, device="cuda"
    )
    fp8_fp4_mega_moe(
        y,
        case.transformed_l1_weights,
        case.transformed_l2_weights,
        case.symm_buffer,
        shared_l1_weights=case.transformed_shared_l1_weights,
        shared_l2_weights=case.transformed_shared_l2_weights,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        activation_clamp=case.config.activation_clamp,
        fast_math=bool(case.config.fast_math),
    )
    return y


def _cleanup_distinct_cases(*cases: MegaMoeCase | None) -> None:
    destroyed: set[int] = set()
    for case in cases:
        if case is None:
            continue
        key = id(case.symm_buffer)
        if key in destroyed:
            continue
        case.symm_buffer.destroy()
        destroyed.add(key)


def _destroy_process_group() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _bench_megamoe_mode(
    funcs: dict[str, Any],
    kernel_names: dict[str, str],
    bench_kineto: Any,
    barrier: Any,
    between_impls: Any,
    *,
    rounds: int,
    cooldown_s: float,
) -> dict[str, Any]:
    """Run the exact benchmark protocol used by latest DeepGEMM MegaMoE."""
    if funcs.keys() != kernel_names.keys():
        raise ValueError("MegaMoE benchmark funcs and kernel_names must have identical keys")
    timer_parameters = inspect.signature(bench_kineto).parameters
    num_tests = int(timer_parameters["num_tests"].default)
    timer_accepts_barrier = "barrier" in timer_parameters
    round_samples: dict[str, list[float]] = {name: [] for name in funcs}
    round_orders: list[list[str]] = []
    items = list(funcs.items())
    for round_idx in range(rounds):
        if round_idx > 0:
            time.sleep(cooldown_s)
        round_items = items if round_idx % 2 == 0 else list(reversed(items))
        round_orders.append([name for name, _ in round_items])

        def run_all() -> None:
            for impl_idx, (_, fn) in enumerate(round_items):
                if impl_idx > 0:
                    between_impls()
                fn()
            if not timer_accepts_barrier:
                barrier()

        round_kernel_names = tuple(kernel_names[name] for name, _ in round_items)
        if timer_accepts_barrier:
            round_times = bench_kineto(run_all, round_kernel_names, barrier=barrier)
        else:
            round_times = bench_kineto(run_all, round_kernel_names)
        for (name, _), seconds in zip(round_items, round_times):
            seconds = float(seconds)
            if not math.isfinite(seconds) or seconds <= 0:
                raise RuntimeError(
                    f"DeepGEMM bench_kineto returned invalid time for {name}: {seconds}"
                )
            round_samples[name].append(seconds * 1e6)

    return {
        "impls": {name: sum(samples) / len(samples) for name, samples in round_samples.items()},
        "round_samples": round_samples,
        "errors": {},
        "timer": "megamoe",
        "benchmark_protocol": {
            "source": "deep_gemm.testing.bench_kineto",
            "kernel_names": kernel_names,
            "num_tests": num_tests,
            "flush_l2": True,
            "flush_l2_bytes": int(8e9),
            "gpu_sleep_cycles": int(2e7),
            "rank_barrier_outside_kernel_timing": True,
            "paired_profile_session": len(funcs) > 1,
            "cold_setup_per_implementation": True,
            "rounds": rounds,
            "round_aggregate": "mean",
            "round_cooldown_s": cooldown_s,
            "round_orders": round_orders,
        },
    }


def _init_dist_on_assigned_device(
    local_rank: int, num_local_ranks: int, physical_device_index: int
) -> tuple[int, int, Any]:
    """Initialize the process-group shape without conflating rank and device."""
    import torch.distributed as dist

    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))
    world_size = num_nodes * num_local_ranks
    rank = node_rank * num_local_ranks + local_rank

    torch.cuda.set_device(physical_device_index)
    params: dict[str, Any] = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": world_size,
        "rank": rank,
    }
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        params["device_id"] = torch.device("cuda", physical_device_index)
    dist.init_process_group(**params)
    torch.set_default_device("cuda")
    torch.cuda.set_device(physical_device_index)
    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(world_size)))


def _run_worker(
    local_rank: int,
    physical_device_index: int,
    physical_device_uuid: str,
    cfg_dict: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    worker_kwargs = dict(cfg_dict)
    warmup = worker_kwargs.pop("warmup", None)
    repeat = worker_kwargs.pop("repeat", None)
    warmup = None if warmup is None else int(warmup)
    repeat = None if repeat is None else int(repeat)
    timer = worker_kwargs.pop("timer", None)
    timer = None if timer is None else str(timer)
    rounds = int(worker_kwargs.pop("rounds", 1))
    cooldown_s = float(worker_kwargs.pop("cooldown_s", 1.0))
    config = MegaMoeConfig(**worker_kwargs)
    config.validate()

    if config.num_processes > torch.cuda.device_count():
        raise SkipTest(
            f"Requested {config.num_processes} processes, but only "
            f"{torch.cuda.device_count()} CUDA devices are visible"
        )

    from tirx_kernels.runner import bind_cuda_assignment, validate_current_cuda_assignment

    bind_cuda_assignment((physical_device_index,), (physical_device_uuid,))
    case = None
    tirx_case = None
    reference_case = None
    default_device_before = torch.get_default_device()
    cuda_device_before = (
        torch.cuda.current_device()
        if torch.cuda.is_available() and torch.cuda.is_initialized()
        else None
    )
    try:
        if (
            hasattr(torch.distributed, "destroy_process_group")
            and torch.distributed.is_initialized()
        ):
            _destroy_process_group()
        rank_idx, num_ranks, group = _init_dist_on_assigned_device(
            local_rank, config.num_processes, physical_device_index
        )
        validate_current_cuda_assignment("after distributed init")

        if mode == "test":
            case = create_case(config, group, rank_idx, num_ranks, compute_reference=True)
            initial_stats = torch.arange(
                config.num_experts_per_rank, dtype=torch.int32, device="cuda"
            )
            tirx_stats = initial_stats.clone()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            y_ref, reference_stats = run_torch_reference(case, initial_stats)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                y_tir = run_tirx_mega_moe(case, tirx_stats)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            except NotImplementedError as exc:
                return {"status": "SKIP", "reason": str(exc), "num_tokens": config.num_tokens}
            torch_relative_l2_diff = _relative_l2_diff(y_tir, y_ref)
            stats_max_abs_diff = int(
                (tirx_stats.to(torch.int64) - reference_stats.to(torch.int64)).abs().max().item()
            )
            return {
                "status": "OK",
                "torch_relative_l2_diff": torch_relative_l2_diff,
                "stats_max_abs_diff": stats_max_abs_diff,
            }

        if mode == "bench":
            from tirx_kernels.runner import bench, external_references_enabled

            if config.num_processes > 1:
                if timer is None:
                    timer = "megamoe"
                elif timer != "megamoe":
                    raise ValueError(
                        "multi-process mega_moe bench requires timer='megamoe' (or omit "
                        f"--timer); {timer!r} is a single-process timer and would "
                        "deadlock the cross-rank kernel collectives"
                    )
            tirx_case = create_case(config, group, rank_idx, num_ranks, compute_reference=False)
            _copy_inputs_into_symm_buffer(tirx_case)
            tirx_stats = None
            reference_stats = None
            if timer == "megamoe":
                initial_stats = torch.zeros(
                    config.num_experts_per_rank, dtype=torch.int32, device="cuda"
                )
                tirx_stats = initial_stats.clone()
                tirx_case.cumulative_local_expert_recv_stats = tirx_stats
            tirx_invocation = _prepare_tirx_invocation(tirx_case)

            deepgemm_step = None
            if external_references_enabled():
                import deep_gemm

                reference_case = create_case(
                    config, group, rank_idx, num_ranks, compute_reference=False
                )
                _copy_inputs_into_symm_buffer(reference_case)
                reference_output = torch.empty(
                    (config.num_tokens, config.hidden), dtype=torch.bfloat16, device="cuda"
                )
                if timer == "megamoe":
                    reference_stats = initial_stats.clone()

                def deepgemm_step() -> None:
                    deep_gemm.fp8_fp4_mega_moe(
                        reference_output,
                        reference_case.transformed_l1_weights,
                        reference_case.transformed_l2_weights,
                        reference_case.symm_buffer,
                        shared_l1_weights=reference_case.transformed_shared_l1_weights,
                        shared_l2_weights=reference_case.transformed_shared_l2_weights,
                        cumulative_local_expert_recv_stats=reference_stats,
                        activation_clamp=config.activation_clamp,
                        fast_math=bool(config.fast_math),
                    )

            def tirx_step() -> None:
                _launch_tirx_mega_moe(tirx_case, tirx_invocation)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            if deepgemm_step is not None:
                deepgemm_step()
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            tirx_step()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            if deepgemm_step is not None:
                max_abs_diff = float(
                    (tirx_invocation.y.float() - reference_output.float()).abs().max().item()
                )
                if max_abs_diff != 0.0:
                    raise AssertionError(f"TIRx diff={max_abs_diff}")
                if timer == "megamoe" and not torch.equal(tirx_stats, reference_stats):
                    raise AssertionError("TIRx cumulative expert stats differ from DeepGEMM")

            if timer == "megamoe":
                if warmup is not None or repeat is not None:
                    raise ValueError(
                        "timer='megamoe' uses DeepGEMM's fixed bench_kineto protocol; "
                        "do not pass warmup/repeat overrides"
                    )

                def tirx_megamoe_step() -> None:
                    _copy_inputs_into_symm_buffer(tirx_case)
                    tirx_invocation.y = torch.empty(
                        (config.num_tokens, config.hidden), dtype=torch.bfloat16, device="cuda"
                    )
                    tirx_step()

                funcs = {"tirx": tirx_megamoe_step}
                kernel_names = {"tirx": "mega_moe_kernel"}
                if deepgemm_step is not None:

                    def deepgemm_megamoe_step() -> None:
                        nonlocal reference_output
                        _copy_inputs_into_symm_buffer(reference_case)
                        reference_output = torch.empty(
                            (config.num_tokens, config.hidden),
                            dtype=torch.bfloat16,
                            device="cuda",
                        )
                        deepgemm_step()

                    funcs["deepgemm"] = deepgemm_megamoe_step
                    kernel_names["deepgemm"] = "sm100_fp8_fp4_mega_moe_impl"

                def reset_between_implementations() -> None:
                    torch.empty(int(8e9 // 4), dtype=torch.int, device="cuda").zero_()
                    torch.cuda._sleep(int(2e7))
                    torch.distributed.barrier()

                # This external import owns the dedicated timer protocol; it is
                # intentionally allowed even when reference implementations are disabled.
                from deep_gemm.testing import bench_kineto

                validate_current_cuda_assignment("before DeepGEMM MegaMoE timing", restore=True)
                result = _bench_megamoe_mode(
                    funcs,
                    kernel_names,
                    bench_kineto,
                    torch.distributed.barrier,
                    reset_between_implementations,
                    rounds=rounds,
                    cooldown_s=cooldown_s,
                )
            else:
                validate_current_cuda_assignment("before TIRx MegaMoE timing", restore=True)
                result = bench(
                    {"tirx": tirx_step},
                    warmup=warmup,
                    repeat=repeat,
                    timer=timer,
                    references=(
                        {"deepgemm": lambda: deepgemm_step}
                        if deepgemm_step is not None
                        else None
                    ),
                    rounds=rounds,
                    cooldown_s=cooldown_s,
                )
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return {"status": "OK", **result}

        raise ValueError(f"Unsupported mode: {mode}")
    finally:
        try:
            _cleanup_distinct_cases(case, tirx_case, reference_case)
            _destroy_process_group()
        finally:
            torch.set_default_device(default_device_before)
            if cuda_device_before is not None:
                torch.cuda.set_device(cuda_device_before)


def _worker_entry(
    local_rank: int,
    device_indices: tuple[int, ...],
    device_uuids: tuple[str, ...],
    cfg_dict: dict[str, Any],
    mode: str,
    result_queue: mp.SimpleQueue | None,
) -> None:
    result = _run_worker(
        local_rank, int(device_indices[local_rank]), str(device_uuids[local_rank]), cfg_dict, mode
    )
    if result_queue is not None:
        result_queue.put((local_rank, result))


def _aggregate_rank_results(rank_results: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    rank_results = sorted(rank_results, key=lambda item: item[0])
    results = [result for _, result in rank_results]
    for result in results:
        if result["status"] == "SKIP":
            return result
    first = results[0]
    if "impls" in first:
        impl_names = sorted({name for result in results for name in result.get("impls", {})})
        round_samples: dict[str, list[float]] = {}
        for name in impl_names:
            per_rank_samples = []
            for result in results:
                samples = result.get("round_samples", {}).get(name)
                if samples is None:
                    raise RuntimeError(f"Rank result is missing round samples for {name!r}")
                per_rank_samples.append([float(value) for value in samples])
            sample_counts = {len(samples) for samples in per_rank_samples}
            if len(sample_counts) != 1:
                raise RuntimeError(
                    f"Ranks reported different round counts for {name!r}: {sorted(sample_counts)}"
                )
            num_rounds = sample_counts.pop()
            round_samples[name] = [
                max(samples[round_idx] for samples in per_rank_samples)
                for round_idx in range(num_rounds)
            ]
        impls = {
            name: sum(samples) / len(samples) for name, samples in round_samples.items() if samples
        }
        errors = {}
        for result in results:
            errors.update(result.get("errors", {}))
        return {
            **first,
            "impls": impls,
            "round_samples": round_samples,
            "errors": errors,
            "rank_results": [
                {
                    "rank": rank,
                    "impls": result.get("impls", {}),
                    "round_samples": result.get("round_samples", {}),
                }
                for rank, result in rank_results
            ],
        }
    if "torch_relative_l2_diff" not in first:
        return first
    return {
        **first,
        "torch_relative_l2_diff": max(
            float(result["torch_relative_l2_diff"]) for result in results
        ),
        "stats_max_abs_diff": max(int(result["stats_max_abs_diff"]) for result in results),
        "rank_results": [
            {
                "rank": rank,
                "torch_relative_l2_diff": float(result["torch_relative_l2_diff"]),
                "stats_max_abs_diff": int(result["stats_max_abs_diff"]),
            }
            for rank, result in rank_results
        ],
    }


def _run_distributed(
    config: MegaMoeConfig,
    mode: str,
    *,
    device_indices: tuple[int, ...] | None = None,
    device_uuids: tuple[str, ...] | None = None,
    **kwargs,
) -> dict[str, Any]:
    cfg_dict = {**asdict(config), **kwargs}
    if device_indices is None:
        device_indices = tuple(range(config.num_processes))
    if device_uuids is None:
        from tirx_kernels.runner import physical_cuda_uuids

        device_uuids = physical_cuda_uuids(device_indices)
    if len(device_indices) != config.num_processes or len(device_uuids) != config.num_processes:
        raise ValueError(
            f"MegaMoE assignment must contain {config.num_processes} devices, got "
            f"indices={device_indices!r}, uuids={device_uuids!r}"
        )
    if config.num_processes > torch.cuda.device_count():
        raise SkipTest(
            f"Requested {config.num_processes} processes, but only "
            f"{torch.cuda.device_count()} CUDA devices are visible"
        )
    if config.num_processes == 1:
        last_error = None
        for _ in range(32):
            port = _find_free_port()
            try:
                with _distributed_env(port):
                    return _run_worker(
                        0, int(device_indices[0]), str(device_uuids[0]), cfg_dict, mode
                    )
            except Exception as exc:
                message = str(exc)
                if "EADDRINUSE" not in message and "address already in use" not in message:
                    raise
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError(
            "Unable to allocate a free TCP port for single-process distributed init."
        )

    port = _find_free_port()
    with _distributed_env(port):
        ctx = mp.get_context("spawn")
        result_queue = ctx.SimpleQueue()
        mp.spawn(
            _worker_entry,
            args=(tuple(device_indices), tuple(device_uuids), cfg_dict, mode, result_queue),
            nprocs=config.num_processes,
            join=True,
        )
        rank_results = [result_queue.get() for _ in range(config.num_processes)]
        return _aggregate_rank_results(rank_results)
