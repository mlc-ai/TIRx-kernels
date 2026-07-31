# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Rank-local NCCL/NVSHMEM runtime support for distributed GEMM kernels."""

from __future__ import annotations

import ctypes
import gc
import os
import tempfile
from collections.abc import Callable, Sequence
from contextlib import contextmanager
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
    device: Any
    communication_stream: int
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
        1 if mapped else 0, dtype=torch.int32, device=f"cuda:{runtime.rank}"
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

    libraries = _locked_library_paths(required=required)
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


def _locked_library_paths(*, required: bool) -> dict[str, Path]:
    """Resolve the exact shared-library files selected for rank workers."""

    result: dict[str, Path] = {}
    missing = []
    for name, env_name in _LOCKED_LIBRARY_ENVS.items():
        configured = os.environ.get(env_name)
        if not configured:
            if required:
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


def _broadcast_nvshmem_uid(rank: int) -> Shape:
    if rank == 0:
        uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")()
        values = [int(value) for value in uid]
    else:
        values = []

    size = torch.tensor([len(values)], dtype=torch.int64, device=f"cuda:{rank}")
    dist.broadcast(size, src=0)
    if rank == 0:
        encoded = torch.tensor(values, dtype=torch.int64, device=f"cuda:{rank}")
    else:
        encoded = torch.empty(int(size.item()), dtype=torch.int64, device=f"cuda:{rank}")
    dist.broadcast(encoded, src=0)
    return Shape(tuple(int(value) for value in encoded.cpu().tolist()))


def _create_runtime(rank: int, world_size: int) -> DistributedRuntime:
    torch.cuda.set_device(rank)
    device = tvm.cuda(rank)
    communication_stream = int(device.create_raw_stream())
    compute_stream = int(device.create_raw_stream())
    device.set_raw_stream(compute_stream)
    timing_stream = torch.cuda.ExternalStream(compute_stream, device=rank)
    return DistributedRuntime(
        rank=rank,
        world_size=world_size,
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
    runtime.device.set_raw_stream(0)
    runtime.device.free_raw_stream(runtime.communication_stream)
    runtime.device.free_raw_stream(runtime.compute_stream)


def _rank_entry(
    rank: int,
    world_size: int,
    init_method: str,
    library_path: str,
    worker: Callable[[DistributedRuntime, Any, str, dict[str, Any]], dict[str, Any]],
    mode: str,
    worker_kwargs: dict[str, Any],
    result_queue: Any,
) -> None:
    runtime = None
    module = None
    nvshmem_initialized = False
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
        uid = _broadcast_nvshmem_uid(rank)
        tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem")(uid, world_size, rank)
        nvshmem_initialized = True

        runtime = _create_runtime(rank, world_size)
        module = tvm.runtime.load_module(library_path)
        result = worker(runtime, module, mode, worker_kwargs)
        runtime.barrier()
        succeeded = True
    finally:
        if runtime is not None:
            _cleanup_runtime(runtime)
        module = None
        gc.collect()
        if nvshmem_initialized:
            tvm.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")()
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

    require_sm100(world_size)
    if not callable(worker):
        raise TypeError("worker must be callable")

    with tempfile.TemporaryDirectory(prefix="tirx-gemm-comm-") as tmpdir:
        library_path = Path(tmpdir) / "kernel.so"
        executable = tvm.compile(ir_module, target=tvm.target.Target("cuda"), tir_pipeline="tirx")
        executable.export_library(str(library_path))

        context = mp.get_context("spawn")
        result_queue = context.SimpleQueue()
        # Every GemmComm workload is single-host. A unique FileStore avoids the
        # bind-after-probe race of selecting a free TCP port before concurrent
        # rank groups start listening on it.
        init_method = f"file://{Path(tmpdir) / 'torch-distributed-init'}"
        with _rank_library_preload(required=mode == "bench"):
            mp.spawn(
                _rank_entry,
                args=(
                    world_size,
                    init_method,
                    str(library_path),
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


__all__ = [
    "DistributedRuntime",
    "barrier_on_communication_stream",
    "barrier_on_compute_stream",
    "require_nvls_multicast",
    "require_sm100",
    "run_distributed",
    "symmetric_empty",
    "sync_communication_to_compute",
    "sync_compute_to_communication",
    "torch_view",
]
