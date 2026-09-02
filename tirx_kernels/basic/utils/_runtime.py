# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Rank-local NCCL/NVSHMEM runtime support for distributed GEMM kernels."""

from __future__ import annotations

import ctypes
import gc
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest import SkipTest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tvm_ffi import Shape

import tvm
from tvm.tirx.bench import DistributedBenchContext

_LOCKED_LIBRARY_ENVS = {
    "nccl": "TIRX_NCCL_LIBRARY",
    "cublas": "TIRX_CUBLAS_LIBRARY",
    "cublasmp": "TIRX_CUBLASMP_LIBRARY",
    "nvshmem": "TIRX_NVSHMEM_LIBRARY",
}
_PROCESS_GROUP_TIMEOUT = timedelta(seconds=60)


@dataclass
class DistributedRuntime:
    """Rank-local module state shared by GemmComm worker implementations."""

    rank: int
    world_size: int
    device_index: int
    device: Any
    communication_stream: int
    compute_stream: int
    timing_stream: torch.cuda.ExternalStream

    def barrier(self) -> None:
        dist.barrier(device_ids=[self.device_index])

    def max_reduce(self, value: float) -> float:
        reduced = torch.tensor(value, dtype=torch.float64, device=f"cuda:{self.device_index}")
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


@dataclass
class PreparedDistributedBench:
    """CPU-compiled benchmark whose ranks start only after GPU assignment."""

    executable: Any
    library_path: Path
    temporary_directory: Any
    world_size: int
    worker: Callable[[DistributedRuntime, Any, str, dict[str, Any]], dict[str, Any]]
    worker_kwargs: dict[str, Any]
    required_timer: str

    @property
    def required_num_gpus(self) -> int:
        return self.world_size

    def run_gpu(
        self,
        *,
        warmup: int | None = None,
        repeat: int | None = None,
        timer: str | None = None,
        rounds: int = 1,
        cooldown_s: float = 1.0,
    ) -> dict[str, Any]:
        if timer not in (None, self.required_timer):
            raise ValueError(f"distributed benchmark supports only timer={self.required_timer!r}")
        if warmup is not None or repeat is not None:
            raise ValueError(
                f"timer={self.required_timer!r} uses fixed iteration counts and "
                "rejects warmup/repeat overrides"
            )
        from tirx_kernels.runner import current_cuda_assignment

        device_indices, device_uuids = current_cuda_assignment()
        if len(device_indices) != self.world_size:
            raise RuntimeError(
                f"distributed benchmark requires {self.world_size} GPUs, "
                f"assignment has {len(device_indices)}"
            )
        require_sm100(device_indices)
        worker_kwargs = {
            **self.worker_kwargs,
            "timer": self.required_timer,
            "rounds": rounds,
            "cooldown_s": cooldown_s,
        }
        return _run_distributed_library(
            self.library_path,
            world_size=self.world_size,
            worker=self.worker,
            mode="bench",
            worker_kwargs=worker_kwargs,
            device_indices=device_indices,
            device_uuids=device_uuids,
        )

    def close(self) -> None:
        self.temporary_directory.cleanup()


def require_sm100(device_indices: Sequence[int]) -> None:
    """Reject unsupported hosts before compiling or spawning workers."""

    from tirx_kernels.target import supports_sm100_kernel

    indices = tuple(int(index) for index in device_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("device_indices must be a non-empty set of physical ordinals")
    device_count = torch.cuda.device_count()
    if any(device_id < 0 or device_id >= device_count for device_id in indices):
        raise SkipTest(f"assigned CUDA devices {indices} are outside visible count {device_count}")
    for device_id in indices:
        capability = torch.cuda.get_device_capability(device_id)
        if not supports_sm100_kernel(capability):
            raise SkipTest(
                f"device {device_id} has compute capability {capability}, expected SM100 or prepared Thor"
            )


def symmetric_empty(runtime: DistributedRuntime, shape: Sequence[int], dtype: str):
    """Allocate one explicitly device-local NVSHMEM tensor on every rank."""

    return tvm.get_global_func("runtime.disco.nvshmem.empty")(
        Shape(tuple(shape)), dtype, runtime.device
    )


def torch_view(tensor: Any) -> torch.Tensor:
    """Create a zero-copy torch view of a local TVM runtime Tensor."""

    return torch.from_dlpack(tensor)


def _loaded_nvshmem_mc_ptr():
    """Resolve nvshmemx_mc_ptr from the host library already loaded by TVM."""

    candidates = [None]
    configured = os.environ.get(_LOCKED_LIBRARY_ENVS["nvshmem"])
    if configured:
        candidates.append(configured)
    candidates.extend(("libnvshmem_host.so.3", "libnvshmem_host.so"))
    checked = set()
    for candidate in candidates:
        if candidate in checked:
            continue
        checked.add(candidate)
        try:
            if candidate is None:
                library = ctypes.CDLL(None)
            else:
                library = ctypes.CDLL(candidate, mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
            return getattr(library, "nvshmemx_mc_ptr")
        except (AttributeError, OSError):
            continue
    raise RuntimeError("the loaded NVSHMEM host library does not export nvshmemx_mc_ptr")


def require_nvls_multicast(runtime: DistributedRuntime, tensor: torch.Tensor) -> int:
    """Collectively verify the loaded NVSHMEM library maps tensor through NVLS."""

    runtime.barrier()
    mc_ptr = _loaded_nvshmem_mc_ptr()
    mc_ptr.argtypes = [ctypes.c_int32, ctypes.c_void_p]
    mc_ptr.restype = ctypes.c_void_p
    mapped = mc_ptr(ctypes.c_int32(0), ctypes.c_void_p(int(tensor.data_ptr())))
    local_available = torch.tensor(
        1 if mapped else 0, dtype=torch.int32, device=f"cuda:{runtime.device_index}"
    )
    dist.all_reduce(local_available, op=dist.ReduceOp.MIN)
    all_available = bool(local_available.item())
    runtime.barrier()
    if not all_available:
        raise RuntimeError(
            "NVLS multicast mapping is unavailable for gemm_out on at least one rank"
        )
    return int(mapped)


def barrier_on_compute_stream(runtime: DistributedRuntime) -> None:
    tvm.get_global_func("runtime.disco.nvshmem.barrier_all_on_stream")(runtime.compute_stream)


def barrier_on_communication_stream(runtime: DistributedRuntime) -> None:
    tvm.get_global_func("runtime.disco.nvshmem.barrier_all_on_stream")(runtime.communication_stream)


def _sync_streams(runtime: DistributedRuntime, source: int, destination: int) -> None:
    tvm.get_global_func("runtime.Device_StreamSyncFromTo")(runtime.device, source, destination)


def sync_compute_to_communication(runtime: DistributedRuntime) -> None:
    _sync_streams(runtime, runtime.compute_stream, runtime.communication_stream)


def sync_communication_to_compute(runtime: DistributedRuntime) -> None:
    _sync_streams(runtime, runtime.communication_stream, runtime.compute_stream)


@contextmanager
def _rank_library_preload(*, required: bool = False):
    """Preload explicitly selected communication libraries in spawned ranks."""

    required_libraries = ("nccl", "nvshmem") if required else ()
    libraries = _locked_library_paths(required=required_libraries)
    if not libraries:
        yield
        return

    original = os.environ.get("LD_PRELOAD")
    entries = [] if not original else original.split(":")
    selected = [str(path) for path in libraries.values()]
    os.environ["LD_PRELOAD"] = ":".join(dict.fromkeys((*selected, *entries)))
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("LD_PRELOAD", None)
        else:
            os.environ["LD_PRELOAD"] = original


@contextmanager
def _rank_cuda_visibility(device_indices: Sequence[int]):
    """Map scheduler-owned physical devices to contiguous rank-local ordinals."""
    assigned = tuple(int(index) for index in device_indices)
    if not assigned or len(set(assigned)) != len(assigned):
        raise ValueError(f"rank devices must be unique, got {device_indices!r}")
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = original.split(",") if original else ()
    physical = tuple(
        visible[index] if 0 <= index < len(visible) else str(index) for index in assigned
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical)
    try:
        yield tuple(range(len(physical)))
    finally:
        if original is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original


def _locked_library_paths(*, required: Sequence[str] = ()) -> dict[str, Path]:
    """Resolve the exact shared-library files selected for rank workers."""

    unknown = set(required).difference(_LOCKED_LIBRARY_ENVS)
    if unknown:
        raise ValueError(f"unknown locked libraries: {', '.join(sorted(unknown))}")

    result: dict[str, Path] = {}
    missing = []
    for name, env_name in _LOCKED_LIBRARY_ENVS.items():
        configured = os.environ.get(env_name)
        if not configured:
            if name in required:
                missing.append(env_name)
            continue
        path = Path(configured)
        if not path.is_absolute():
            raise ValueError(f"{env_name} must be an absolute path: {configured}")
        try:
            path = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"{env_name} does not exist: {configured}") from error
        if not path.is_file():
            raise FileNotFoundError(f"{env_name} is not a file: {configured}")
        result[name] = path
    if missing:
        raise RuntimeError(
            "distributed GemmComm benchmarks require explicit library locks: " + ", ".join(missing)
        )
    return result


def _broadcast_nvshmem_uid(rank: int, device_index: int) -> Shape:
    if rank == 0:
        uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")()
        values = [int(value) for value in uid]
    else:
        values = []

    size = torch.tensor([len(values)], dtype=torch.int64, device=f"cuda:{device_index}")
    dist.broadcast(size, src=0)
    if rank == 0:
        encoded = torch.tensor(values, dtype=torch.int64, device=f"cuda:{device_index}")
    else:
        encoded = torch.empty(int(size.item()), dtype=torch.int64, device=f"cuda:{device_index}")
    dist.broadcast(encoded, src=0)
    return Shape(tuple(int(value) for value in encoded.cpu().tolist()))


def _create_runtime(rank: int, world_size: int, device_index: int) -> DistributedRuntime:
    torch.cuda.set_device(device_index)
    device = tvm.cuda(device_index)
    communication_stream = int(device.create_raw_stream())
    compute_stream = int(device.create_raw_stream())
    device.set_raw_stream(compute_stream)
    timing_stream = torch.cuda.ExternalStream(compute_stream, device=device_index)
    return DistributedRuntime(
        rank=rank,
        world_size=world_size,
        device_index=device_index,
        device=device,
        communication_stream=communication_stream,
        compute_stream=compute_stream,
        timing_stream=timing_stream,
    )


def _cleanup_runtime(runtime: DistributedRuntime | None) -> None:
    if runtime is None:
        return
    for stream in (runtime.compute_stream, runtime.communication_stream):
        try:
            runtime.device.sync(stream)
        except Exception:
            pass
    with suppress(Exception):
        runtime.device.set_raw_stream(0)
    for stream in (runtime.communication_stream, runtime.compute_stream):
        with suppress(Exception):
            runtime.device.free_raw_stream(stream)


def _rank_entry(
    rank: int,
    world_size: int,
    init_method: str,
    library_paths: Mapping[str, str],
    worker: Callable[[DistributedRuntime, Any, str, dict[str, Any]], dict[str, Any]],
    mode: str,
    worker_kwargs: dict[str, Any],
    result_queue: Any,
    device_indices: Sequence[int],
    device_uuids: Sequence[str],
    worker_receives_mapping: bool,
) -> None:
    runtime = None
    modules = None
    succeeded = False
    result = None
    try:
        from tirx_kernels.runner import bind_cuda_assignment, validate_current_cuda_assignment

        device_index = int(device_indices[rank])
        bind_cuda_assignment((device_index,), (str(device_uuids[rank]),))
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            device_id=torch.device("cuda", device_index),
            timeout=_PROCESS_GROUP_TIMEOUT,
        )
        validate_current_cuda_assignment("after distributed process-group init", restore=True)
        uid = _broadcast_nvshmem_uid(rank, device_index)
        tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem")(uid, world_size, rank)

        runtime = _create_runtime(rank, world_size, device_index)
        modules = {
            name: tvm.runtime.load_module(library_path)
            for name, library_path in library_paths.items()
        }
        worker_modules = modules if worker_receives_mapping else modules["main"]
        validate_current_cuda_assignment("before distributed worker", restore=True)
        result = worker(runtime, worker_modules, mode, worker_kwargs)
        validate_current_cuda_assignment("after distributed worker", restore=True)
        runtime.barrier()
        succeeded = True
    finally:
        if runtime is not None:
            _cleanup_runtime(runtime)
        modules = None
        gc.collect()
        # Each spawned rank is one-shot. Process teardown owns NVSHMEM cleanup;
        # explicit finalize aborts after cuBLASMp changes the CUDA context stack.
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    if succeeded and rank == 0:
        result_queue.put(result)


def run_distributed(
    ir_module: Any,
    *,
    world_size: int,
    worker: Callable[[DistributedRuntime, Any, str, dict[str, Any]], dict[str, Any]],
    mode: str,
    worker_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Compile once in the parent, then execute one rank-local worker per GPU."""

    return _run_distributed_modules(
        {"main": ir_module},
        world_size=world_size,
        worker=worker,
        mode=mode,
        worker_kwargs=worker_kwargs,
        worker_receives_mapping=False,
    )


def run_distributed_modules(
    ir_modules: Mapping[str, Any],
    *,
    world_size: int,
    worker: Callable[[DistributedRuntime, Mapping[str, Any], str, dict[str, Any]], dict[str, Any]],
    mode: str,
    worker_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Compile named modules and load them together in each rank worker."""

    return _run_distributed_modules(
        ir_modules,
        world_size=world_size,
        worker=worker,
        mode=mode,
        worker_kwargs=worker_kwargs,
        worker_receives_mapping=True,
    )


def _run_distributed_modules(
    ir_modules: Mapping[str, Any],
    *,
    world_size: int,
    worker: Callable[[DistributedRuntime, Any, str, dict[str, Any]], dict[str, Any]],
    mode: str,
    worker_kwargs: dict[str, Any],
    worker_receives_mapping: bool,
) -> dict[str, Any]:
    """Shared synchronous path for one or more named modules."""
    from tirx_kernels.runner import cuda_target, physical_cuda_uuids

    device_indices = tuple(range(world_size))
    device_uuids = physical_cuda_uuids(device_indices)
    require_sm100(device_indices)
    if not callable(worker):
        raise TypeError("worker must be callable")
    ir_modules = dict(ir_modules)
    if not ir_modules:
        raise ValueError("ir_modules must contain at least one named module")
    if any(not isinstance(name, str) or not name for name in ir_modules):
        raise ValueError("ir_modules keys must be non-empty strings")
    if not worker_receives_mapping and set(ir_modules) != {"main"}:
        raise ValueError("single-module workers require exactly the key 'main'")

    with tempfile.TemporaryDirectory(prefix="tirx-gemm-comm-") as tmpdir:
        library_paths = {}
        for index, (name, ir_module) in enumerate(ir_modules.items()):
            library_path = Path(tmpdir) / f"kernel-{index}.so"
            executable = tvm.compile(ir_module, target=cuda_target(), tir_pipeline="tirx")
            executable.export_library(str(library_path))
            library_paths[name] = str(library_path)
        return _run_distributed_libraries(
            library_paths,
            world_size=world_size,
            worker=worker,
            mode=mode,
            worker_kwargs=worker_kwargs,
            device_indices=device_indices,
            device_uuids=device_uuids,
            worker_receives_mapping=worker_receives_mapping,
        )


def prepare_distributed_bench(
    ir_module: Any,
    *,
    world_size: int,
    worker: Callable[[DistributedRuntime, Any, str, dict[str, Any]], dict[str, Any]],
    worker_kwargs: dict[str, Any],
    required_timer: str,
) -> PreparedDistributedBench:
    """Compile/export before assignment and retain the artifact in this process."""
    from tirx_kernels.runner import cuda_target

    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if not callable(worker):
        raise TypeError("worker must be callable")
    temporary_directory = tempfile.TemporaryDirectory(prefix="tirx-gemm-comm-prepared-")
    library_path = Path(temporary_directory.name) / "kernel.so"
    try:
        executable = tvm.compile(ir_module, target=cuda_target(), tir_pipeline="tirx")
        executable.export_library(str(library_path))
    except BaseException:
        temporary_directory.cleanup()
        raise
    return PreparedDistributedBench(
        executable=executable,
        library_path=library_path,
        temporary_directory=temporary_directory,
        world_size=world_size,
        worker=worker,
        worker_kwargs=dict(worker_kwargs),
        required_timer=required_timer,
    )


def _run_distributed_library(
    library_path: Path,
    *,
    world_size: int,
    worker: Callable[[DistributedRuntime, Any, str, dict[str, Any]], dict[str, Any]],
    mode: str,
    worker_kwargs: dict[str, Any],
    device_indices: Sequence[int],
    device_uuids: Sequence[str],
) -> dict[str, Any]:
    """Start rank CUDA/NCCL/NVSHMEM state after the complete claim exists."""

    return _run_distributed_libraries(
        {"main": str(library_path)},
        world_size=world_size,
        worker=worker,
        mode=mode,
        worker_kwargs=worker_kwargs,
        device_indices=device_indices,
        device_uuids=device_uuids,
        worker_receives_mapping=False,
    )


def _run_distributed_libraries(
    library_paths: Mapping[str, str],
    *,
    world_size: int,
    worker: Callable[[DistributedRuntime, Any, str, dict[str, Any]], dict[str, Any]],
    mode: str,
    worker_kwargs: dict[str, Any],
    device_indices: Sequence[int],
    device_uuids: Sequence[str],
    worker_receives_mapping: bool,
) -> dict[str, Any]:
    """Start rank CUDA/NCCL/NVSHMEM state for precompiled named modules."""
    context = mp.get_context("spawn")
    result_queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory(prefix="tirx-gemm-comm-ranks-") as tmpdir:
        init_method = f"file://{Path(tmpdir) / 'torch-distributed-init'}"
        with (
            _rank_library_preload(required=mode == "bench"),
            _rank_cuda_visibility(device_indices) as local_device_indices,
        ):
            mp.spawn(
                _rank_entry,
                args=(
                    world_size,
                    init_method,
                    dict(library_paths),
                    worker,
                    mode,
                    worker_kwargs,
                    result_queue,
                    local_device_indices,
                    tuple(device_uuids),
                    worker_receives_mapping,
                ),
                nprocs=world_size,
                join=True,
            )
        result = result_queue.get()
    if not isinstance(result, dict):
        raise TypeError("distributed worker must return a result dictionary")
    return result


__all__ = [
    "DistributedRuntime",
    "barrier_on_communication_stream",
    "barrier_on_compute_stream",
    "prepare_distributed_bench",
    "require_nvls_multicast",
    "require_sm100",
    "run_distributed",
    "run_distributed_modules",
    "symmetric_empty",
    "sync_communication_to_compute",
    "sync_compute_to_communication",
    "torch_view",
]
