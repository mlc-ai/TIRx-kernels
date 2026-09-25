# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Rank-local distributed runtime support for DeepEP kernels.

Trimmed from `tirx_kernels/basic/utils/_runtime.py`: keeps the spawn /
FileStore / DistributedRuntime / DistributedBenchContext pattern, drops the
NVSHMEM initialization and the GemmComm-specific library locks (DeepEP V2
uses NCCL symmetric memory, not NVSHMEM).
"""

from __future__ import annotations

import gc
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest import SkipTest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import tvm
from tvm.tirx.bench import DistributedBenchContext

_PROCESS_GROUP_TIMEOUT = timedelta(seconds=60)


@dataclass
class DistributedRuntime:
    """Rank-local module state shared by DeepEP worker implementations."""

    rank: int
    world_size: int
    device: Any
    compute_stream: int
    timing_stream: torch.cuda.ExternalStream

    def barrier(self) -> None:
        dist.barrier(device_ids=[self.rank])

    def max_reduce(self, value: float) -> float:
        reduced = torch.tensor(value, dtype=torch.float64, device=f"cuda:{self.rank}")
        dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
        return float(reduced.item())

    def bench_context(self) -> DistributedBenchContext:
        return DistributedBenchContext(
            rank=self.rank,
            world_size=self.world_size,
            barrier=self.barrier,
            max_reduce=self.max_reduce,
            stream=self.timing_stream,
        )


def require_sm100(world_size: int) -> None:
    """Reject unsupported hosts before compiling or spawning workers."""

    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    device_count = torch.cuda.device_count()
    if device_count < world_size:
        raise SkipTest(f"requires {world_size} CUDA devices, found {device_count}")
    for device_id in range(world_size):
        major, _minor = torch.cuda.get_device_capability(device_id)
        if major != 10:
            raise SkipTest(f"device {device_id} has compute capability {major}, expected SM100")


def _create_runtime(rank: int, world_size: int) -> DistributedRuntime:
    torch.cuda.set_device(rank)
    device = tvm.cuda(rank)
    compute_stream = int(device.create_raw_stream())
    device.set_raw_stream(compute_stream)
    timing_stream = torch.cuda.ExternalStream(compute_stream, device=rank)
    return DistributedRuntime(
        rank=rank,
        world_size=world_size,
        device=device,
        compute_stream=compute_stream,
        timing_stream=timing_stream,
    )


def _cleanup_runtime(runtime: DistributedRuntime | None) -> None:
    if runtime is None:
        return
    try:
        runtime.device.sync(runtime.compute_stream)
    except Exception:
        pass
    runtime.device.set_raw_stream(0)
    runtime.device.free_raw_stream(runtime.compute_stream)


def _rank_entry(
    rank: int,
    world_size: int,
    init_method: str,
    library_paths: dict[str, str],
    worker: Callable[[DistributedRuntime, dict[str, Any], str, dict[str, Any]], dict[str, Any]],
    mode: str,
    worker_kwargs: dict[str, Any],
    result_queue: Any,
) -> None:
    runtime = None
    modules: dict[str, Any] = {}
    succeeded = False
    result = None
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            device_id=torch.device("cuda", rank),
            timeout=_PROCESS_GROUP_TIMEOUT,
        )
        runtime = _create_runtime(rank, world_size)
        modules = {name: tvm.runtime.load_module(path) for name, path in library_paths.items()}
        result = worker(runtime, modules, mode, worker_kwargs)
        runtime.barrier()
        succeeded = True
    finally:
        if runtime is not None:
            _cleanup_runtime(runtime)
        modules.clear()
        gc.collect()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    if succeeded and rank == 0:
        result_queue.put(result)


def compile_kernels(kernels: dict[str, Any], tmpdir: str) -> dict[str, str]:
    """Compile every kernel once and export loadable libraries into `tmpdir`."""

    library_paths: dict[str, str] = {}
    for name, func in kernels.items():
        library_path = Path(tmpdir) / f"{name}.so"
        executable = tvm.compile(
            tvm.IRModule({"main": func}), target=tvm.target.Target("cuda"), tir_pipeline="tirx"
        )
        executable.export_library(str(library_path))
        library_paths[name] = str(library_path)
    return library_paths


def run_distributed(
    kernels: dict[str, Any],
    *,
    world_size: int,
    worker: Callable[[DistributedRuntime, dict[str, Any], str, dict[str, Any]], dict[str, Any]],
    mode: str,
    worker_kwargs: dict[str, Any],
    prepared_libraries: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compile every kernel once in the parent, then run one rank-local worker per GPU.

    When `prepared_libraries` is given, compilation is skipped and the
    prebuilt `{name: library_path}` mapping is used instead (the two-stage
    bench-suite contract compiles in the CPU-prepare stage).
    """

    require_sm100(world_size)
    if not callable(worker):
        raise TypeError("worker must be callable")

    with tempfile.TemporaryDirectory(prefix="tirx-deepep-") as tmpdir:
        if prepared_libraries is not None:
            library_paths = prepared_libraries
        else:
            library_paths = compile_kernels(kernels, tmpdir)

        context = mp.get_context("spawn")
        result_queue = context.SimpleQueue()
        # Every DeepEP workload is single-host. A unique FileStore avoids the
        # bind-after-probe race of selecting a free TCP port before concurrent
        # rank groups start listening on it.
        init_method = f"file://{Path(tmpdir) / 'torch-distributed-init'}"
        mp.spawn(
            _rank_entry,
            args=(
                world_size,
                init_method,
                library_paths,
                worker,
                mode,
                worker_kwargs,
                result_queue,
            ),
            nprocs=world_size,
            join=True,
        )
        result = result_queue.get()

    if not isinstance(result, dict):
        raise TypeError("distributed worker must return a result dictionary")
    return result


__all__ = ["DistributedRuntime", "compile_kernels", "require_sm100", "run_distributed"]
