# Copyright (c) 2026 The TIRx Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime isolation for direct old/current benchmark subprocesses.

The migration gate measures only the local TIRx implementation.  The bench
suite copies this module next to a generated ``sitecustomize.py`` outside both
checkouts, so the same hook is installed in the parent benchmark process and
in every rank process created with multiprocessing ``spawn``.
"""

from __future__ import annotations

import os
import sys
from functools import wraps
from importlib import import_module
from typing import Any

LOCAL_ONLY_ENV = "TIRX_BENCH_LOCAL_ONLY"
LOCAL_ONLY_KERNEL_ENV = "TIRX_BENCH_LOCAL_ONLY_KERNEL"
_PATCH_MARKER = "_tirx_bench_local_only"
_SELECTIVE_STATE_KERNELS = {
    "selective_state_update_mtp_horizontal",
    "selective_state_update_mtp_simple",
    "selective_state_update_mtp_vertical",
    "selective_state_update_stp_horizontal",
    "selective_state_update_stp_simple",
    "selective_state_update_stp_vertical",
}


class _LocalOnlyBaselineSuite:
    """No-reference GemmComm suite that preserves collective cleanup order."""

    def __init__(self, runtime: Any):
        self._runtime = runtime

    def references(self) -> dict[str, Any]:
        return {}

    def metadata(self) -> dict[str, Any]:
        return {"local_only": True}

    def close(self) -> None:
        self._runtime.barrier()


def _install_bench_patch() -> None:
    import tvm.tirx.bench as bench_module

    original_bench = bench_module.bench
    if getattr(original_bench, _PATCH_MARKER, False):
        return

    @wraps(original_bench)
    def local_only_bench(funcs, **kwargs):
        kwargs.pop("references", None)
        result = original_bench(funcs, **kwargs)
        result["local_only"] = True
        return result

    setattr(local_only_bench, _PATCH_MARKER, True)
    bench_module.bench = local_only_bench


def _install_gemmcomm_patch() -> None:
    from tirx_kernels.basic.utils import _baselines

    def create_local_only_baseline_suite(runtime, *_args, **_kwargs):
        return _LocalOnlyBaselineSuite(runtime)

    setattr(create_local_only_baseline_suite, _PATCH_MARKER, True)
    _baselines.create_baseline_suite = create_local_only_baseline_suite

    # Normally the kernel modules are imported after sitecustomize.  Keep the
    # patch correct if an embedding imported either module unusually early.
    for module_name in (
        "tirx_kernels.basic.allgather_gemm",
        "tirx_kernels.basic.gemm_reduce_scatter",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            module.create_baseline_suite = create_local_only_baseline_suite


def _local_only_tinygemm_bench(
    module,
    B: int,
    O: int,
    K: int,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 5,
    cooldown_s: float = 1.0,
):
    """Run TinyGEMM2 without importing or launching the FlashInfer oracle."""

    module._require_sm100()
    from tvm.tirx.bench import bench

    case = module.prepare_data(B, O, K)
    torch = module.torch
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    stage = module._select_stage(B, O, K, num_sms)
    executable = module._compile_executable(B, O, stage, False)
    args = module._tirx_args(case)

    # Preserve the ordinary benchmark's compile/setup boundary and one untimed
    # launch preflight, but do not construct the external correctness oracle.
    executable(*args)
    torch.cuda.synchronize()
    return bench(
        {"tirx": lambda: executable(*args)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def _install_tinygemm_patch() -> None:
    from tirx_kernels.flashinfer.gemm import tinygemm2_sm100

    original_run_bench = tinygemm2_sm100.run_bench
    if getattr(original_run_bench, _PATCH_MARKER, False):
        return
    tinygemm2_sm100._tirx_original_run_bench = original_run_bench

    @wraps(original_run_bench)
    def local_only_run_bench(*args, **kwargs):
        return _local_only_tinygemm_bench(tinygemm2_sm100, *args, **kwargs)

    setattr(local_only_run_bench, _PATCH_MARKER, True)
    tinygemm2_sm100.run_bench = local_only_run_bench


def _local_only_selective_state_bench(
    module,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **kwargs: Any,
):
    """Run one selective-state implementation without its FlashInfer preflight."""

    rounds = int(kwargs.pop("rounds", 5))
    cooldown_s = float(kwargs.pop("cooldown_s", 1.0))
    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench

    case = module.prepare_data(**kwargs)
    executable = compile_kernel(module.get_kernel(**kwargs))
    args = module._tirx_args(case)
    executable(*args)
    module.torch.cuda.synchronize()
    return bench(
        {"tirx": lambda: executable(*args)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def _install_selective_state_patch(kernel: str) -> None:
    module = import_module(f"tirx_kernels.flashinfer.mamba.{kernel}")
    original_run_bench = module.run_bench
    if getattr(original_run_bench, _PATCH_MARKER, False):
        return
    module._tirx_original_run_bench = original_run_bench

    @wraps(original_run_bench)
    def local_only_run_bench(*args, **kwargs):
        return _local_only_selective_state_bench(module, *args, **kwargs)

    setattr(local_only_run_bench, _PATCH_MARKER, True)
    module.run_bench = local_only_run_bench


def _local_only_fp8_paged_mqa_bench(module, **kwargs: Any):
    """Run FP8 paged MQA without launching DeepGEMM or SGLang kernels."""

    from tvm.tirx.bench import bench

    timer = kwargs.pop("timer", None)
    warmup = kwargs.pop("warmup", None)
    repeat = kwargs.pop("repeat", None)
    rounds = int(kwargs.pop("rounds", 1))
    cooldown_s = float(kwargs.pop("cooldown_s", 1.0))
    config = module._make_config(**kwargs)
    executable = module._compile_tirx_paged_mqa(config)
    data = module._prepare_data(config, compute_reference=False)
    invocation = module._prepare_tirx_invocation(data, executable=executable)

    module._run_tirx_invocation(data, invocation)
    module.torch.cuda.synchronize()
    module.torch.cuda.empty_cache()
    return bench(
        {"tirx": lambda: module._run_tirx_invocation(data, invocation)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def _install_fp8_paged_mqa_patch() -> None:
    from tirx_kernels.deepgemm import paged_mqa_logits_fp8

    original_run_bench = paged_mqa_logits_fp8.run_bench
    if getattr(original_run_bench, _PATCH_MARKER, False):
        return
    paged_mqa_logits_fp8._tirx_original_run_bench = original_run_bench

    @wraps(original_run_bench)
    def local_only_run_bench(*args, **kwargs):
        return _local_only_fp8_paged_mqa_bench(paged_mqa_logits_fp8, *args, **kwargs)

    setattr(local_only_run_bench, _PATCH_MARKER, True)
    paged_mqa_logits_fp8.run_bench = local_only_run_bench


def _local_only_nvfp4_gemm_bench(
    module,
    M: int = 1024,
    N: int = 1024,
    K: int = 1024,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **kwargs: Any,
):
    """Run NVFP4 GEMM without autotuning or launching external baselines."""

    kernel = module.tir_ws_kernel(M, N, K)
    target = module.tvm.target.Target("cuda")
    with target:
        ir_module = module.tvm.IRModule({"main": kernel})
        executable = module.tvm.compile(ir_module, target=target, tir_pipeline="tirx")

    A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref = module.prepare_data(M, N, K)
    alpha_tensor = module.torch.tensor(
        [float(alpha.item())], device="cuda", dtype=module.torch.float
    )
    out_tir = module.torch.empty_like(C_ref).to("cuda").to(module.torch.bfloat16)

    from tvm.tirx.bench import bench

    return bench(
        {"tir": lambda: executable.mod(A_fp4, B_fp4, A_sf, B_sf, alpha_tensor, out_tir)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        **kwargs,
    )


def _install_nvfp4_gemm_patch() -> None:
    from tirx_kernels.basic import nvfp4_gemm

    original_run_bench = nvfp4_gemm.run_bench
    if getattr(original_run_bench, _PATCH_MARKER, False):
        return
    nvfp4_gemm._tirx_original_run_bench = original_run_bench

    @wraps(original_run_bench)
    def local_only_run_bench(*args, **kwargs):
        return _local_only_nvfp4_gemm_bench(nvfp4_gemm, *args, **kwargs)

    setattr(local_only_run_bench, _PATCH_MARKER, True)
    nvfp4_gemm.run_bench = local_only_run_bench


def _local_only_megamoe_worker(module, local_rank: int, cfg_dict: dict[str, Any], mode: str):
    if mode != "bench":
        return module._tirx_original_run_worker(local_rank, cfg_dict, mode)

    worker_kwargs = dict(cfg_dict)
    warmup = worker_kwargs.pop("warmup", None)
    repeat = worker_kwargs.pop("repeat", None)
    warmup = None if warmup is None else int(warmup)
    repeat = None if repeat is None else int(repeat)
    timer = worker_kwargs.pop("timer", None)
    timer = None if timer is None else str(timer)
    rounds = int(worker_kwargs.pop("rounds", 1))
    cooldown_s = float(worker_kwargs.pop("cooldown_s", 1.0))
    config = module.MegaMoeConfig(**worker_kwargs)
    config.validate()

    torch = module.torch
    if config.num_processes > torch.cuda.device_count():
        raise module.SkipTest(
            f"Requested {config.num_processes} processes, but only "
            f"{torch.cuda.device_count()} CUDA devices are visible"
        )
    if config.num_processes > 1:
        if timer is None:
            timer = "megamoe"
        elif timer != "megamoe":
            raise ValueError(
                "multi-process mega_moe bench requires timer='megamoe' "
                f"(or omit --timer); got {timer!r}"
            )

    deep_gemm, _source = module.load_deep_gemm_mega()
    tirx_case = None
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
            module._destroy_process_group()
        rank_idx, num_ranks, group = deep_gemm.utils.dist.init_dist(
            local_rank, config.num_processes
        )

        tirx_case = module.create_case(deep_gemm, config, group, rank_idx, num_ranks)
        if timer == "megamoe":
            tirx_case.cumulative_local_expert_recv_stats = torch.zeros(
                config.num_experts_per_rank, dtype=torch.int32, device="cuda"
            )
        module._copy_inputs_into_symm_buffer(tirx_case)
        tirx_invocation = module._prepare_tirx_invocation(tirx_case)

        def tirx_step() -> None:
            module._launch_tirx_mega_moe(tirx_case, tirx_invocation)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        tirx_step()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if timer == "megamoe":
            if warmup is not None or repeat is not None:
                raise ValueError(
                    "timer='megamoe' uses DeepGEMM's fixed bench_kineto protocol; "
                    "do not pass warmup/repeat overrides"
                )

            def tirx_megamoe_step() -> None:
                module._copy_inputs_into_symm_buffer(tirx_case)
                tirx_invocation.y = torch.empty(
                    (config.num_tokens, config.hidden), dtype=torch.bfloat16, device="cuda"
                )
                tirx_step()

            from deep_gemm.testing import bench_kineto

            bench_result = module._bench_megamoe_mode(
                {"tirx": tirx_megamoe_step},
                {"tirx": "mega_moe_kernel"},
                bench_kineto,
                torch.distributed.barrier,
                lambda: None,
                rounds=rounds,
                cooldown_s=cooldown_s,
            )
            bench_result["local_only"] = True
        else:
            from tvm.tirx.bench import bench

            bench_result = bench(
                {"tirx": tirx_step},
                warmup=warmup,
                repeat=repeat,
                timer=timer,
                rounds=rounds,
                cooldown_s=cooldown_s,
            )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        impls = bench_result.get("impls") or {}
        if set(impls) != {"tirx"}:
            raise RuntimeError(
                f"Local-only MegaMoE benchmark must report only 'tirx', got {sorted(impls)}"
            )
        protocol = dict(bench_result.get("benchmark_protocol", {}))
        protocol["local_only"] = True
        if timer == "megamoe":
            protocol["paired_profile_session"] = False
            protocol["local_implementation_count"] = 1
        return {
            "status": "OK",
            "impls": {"tirx": float(impls["tirx"])},
            "round_samples": bench_result.get("round_samples", {}),
            "errors": bench_result.get("errors", {}),
            "timer": bench_result.get("timer"),
            "benchmark_protocol": protocol,
            "local_only": True,
        }
    finally:
        try:
            module._cleanup_distinct_cases(tirx_case)
            module._destroy_process_group()
        finally:
            torch.set_default_device(default_device_before)
            if cuda_device_before is not None:
                torch.cuda.set_device(cuda_device_before)


def _install_megamoe_patch() -> None:
    from tirx_kernels.deepgemm import mega_moe

    if getattr(mega_moe._run_worker, _PATCH_MARKER, False):
        return
    mega_moe._tirx_original_run_worker = mega_moe._run_worker

    @wraps(mega_moe._run_worker)
    def local_only_run_worker(local_rank, cfg_dict, mode):
        return _local_only_megamoe_worker(mega_moe, local_rank, cfg_dict, mode)

    setattr(local_only_run_worker, _PATCH_MARKER, True)
    mega_moe._run_worker = local_only_run_worker


def install() -> None:
    """Install the process-wide local-only patches when explicitly enabled."""

    if os.environ.get(LOCAL_ONLY_ENV) != "1":
        return
    kernel = os.environ.get(LOCAL_ONLY_KERNEL_ENV)
    if not kernel:
        raise RuntimeError(f"{LOCAL_ONLY_KERNEL_ENV} is required in local-only mode")

    _install_bench_patch()
    if kernel in {"allgather_gemm", "gemm_reduce_scatter"}:
        _install_gemmcomm_patch()
    elif kernel == "tinygemm2_sm100":
        _install_tinygemm_patch()
    elif kernel in _SELECTIVE_STATE_KERNELS:
        _install_selective_state_patch(kernel)
    elif kernel == "deepgemm_sm100_fp8_paged_mqa_logits":
        _install_fp8_paged_mqa_patch()
    elif kernel == "nvfp4_gemm":
        _install_nvfp4_gemm_patch()
    elif kernel == "deepgemm_fp8_fp4_mega_moe":
        _install_megamoe_patch()


__all__ = ["LOCAL_ONLY_ENV", "LOCAL_ONLY_KERNEL_ENV", "install"]
