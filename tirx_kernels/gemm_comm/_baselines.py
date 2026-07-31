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

"""Rank-local cuBLASMp and cuBLAS+NCCL GemmComm baselines."""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.distributed as dist

from ._model_shapes import SUPPORTED_WORLD_SIZES
from ._runtime import DistributedRuntime, _locked_library_paths

_ALGORITHMS = {
    "default": "DEFAULT",
    "split_p2p": "SPLIT_P2P",
    "split_multicast": "SPLIT_MULTICAST",
    "no_overlap": "NO_OVERLAP",
}

_LIBRARY_SYMBOLS = {
    "nccl": "ncclGetVersion",
    "cublas": "cublasGetProperty",
    "cublasmp": "cublasMpGetVersion",
    "nvshmem": "nvshmem_info_get_version",
}


class _DlInfo(ctypes.Structure):
    _fields_: ClassVar = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _process_symbol(symbol: str):
    try:
        return getattr(ctypes.CDLL(None), symbol)
    except AttributeError as error:
        raise RuntimeError(f"required library symbol is not loaded: {symbol}") from error


def _symbol_library_path(symbol: str) -> Path:
    """Return the real shared object that currently resolves ``symbol``."""

    libdl = ctypes.CDLL(ctypes.util.find_library("dl") or "libdl.so.2")
    libdl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    libdl.dladdr.restype = ctypes.c_int
    info = _DlInfo()
    address = ctypes.cast(_process_symbol(symbol), ctypes.c_void_p)
    if libdl.dladdr(address, ctypes.byref(info)) == 0 or not info.dli_fname:
        raise RuntimeError(f"unable to resolve the loaded library for symbol {symbol}")
    return Path(os.fsdecode(info.dli_fname)).resolve(strict=True)


def _format_encoded_version(version: int) -> str:
    return f"{version // 10000}.{version % 10000 // 100}.{version % 100}"


def _read_encoded_version(symbol: str) -> tuple[str, int]:
    get_version = _process_symbol(symbol)
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    version = ctypes.c_int()
    status = int(get_version(ctypes.byref(version)))
    if status != 0:
        raise RuntimeError(f"{symbol} failed with status {status}")
    return _format_encoded_version(version.value), int(version.value)


def _read_cublas_version() -> tuple[str, int]:
    get_property = _process_symbol("cublasGetProperty")
    get_property.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    get_property.restype = ctypes.c_int
    parts = []
    for property_id in range(3):
        value = ctypes.c_int()
        status = int(get_property(property_id, ctypes.byref(value)))
        if status != 0:
            raise RuntimeError(f"cublasGetProperty({property_id}) failed with status {status}")
        parts.append(int(value.value))
    major, minor, patch = parts
    return f"{major}.{minor}.{patch}", major * 10000 + minor * 100 + patch


def _read_nvshmem_version() -> tuple[str, str]:
    get_version = _process_symbol("nvshmem_info_get_version")
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    get_version.restype = None
    major = ctypes.c_int()
    minor = ctypes.c_int()
    get_version(ctypes.byref(major), ctypes.byref(minor))
    get_name = _process_symbol("nvshmem_info_get_name")
    get_name.argtypes = [ctypes.c_char_p]
    get_name.restype = None
    name = ctypes.create_string_buffer(256)
    get_name(name)
    release = name.value.decode().removeprefix("NVSHMEM v")
    return release, f"{major.value}.{minor.value}"


def _library_provenance() -> dict[str, dict[str, Any]]:
    """Verify every configured lock against the library resolving its API symbol."""

    configured = _locked_library_paths(required=True)
    result = {}
    for name, symbol in _LIBRARY_SYMBOLS.items():
        actual = _symbol_library_path(symbol)
        expected = configured[name]
        if not actual.samefile(expected):
            raise RuntimeError(
                f"{name} library lock mismatch: configured {expected}, loaded {actual}"
            )
        if name == "cublas":
            version, version_raw = _read_cublas_version()
        elif name == "nvshmem":
            version, api_version = _read_nvshmem_version()
            version_raw = None
        else:
            version, version_raw = _read_encoded_version(symbol)
        record: dict[str, Any] = {"path": str(actual), "version": version}
        if name == "nvshmem":
            record["api_version"] = api_version
        if version_raw is not None:
            record["version_raw"] = version_raw
        result[name] = record
    return result


@dataclass(frozen=True)
class _Problem:
    workload: str
    M: int
    N: int
    K: int
    world_size: int

    @property
    def local_m(self) -> int:
        return self.M // self.world_size

    @property
    def local_n(self) -> int:
        return self.N // self.world_size

    @property
    def local_k(self) -> int:
        return self.K // self.world_size


def _problem(workload: str, *, M: int, N: int, K: int, world_size: int) -> _Problem:
    if workload not in {"allgather_gemm", "gemm_reduce_scatter"}:
        raise ValueError(f"unknown GemmComm baseline workload: {workload!r}")
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"GemmComm baselines require world_size in {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    if any(value <= 0 for value in (M, N, K)):
        raise ValueError("GemmComm baseline dimensions must be positive")
    if M % world_size or N % world_size or K % world_size:
        raise ValueError("GemmComm baseline dimensions must be divisible by world_size")
    return _Problem(workload=workload, M=M, N=N, K=K, world_size=world_size)


class _CublasNcclLaunch:
    """Explicit NCCL collective plus rank-local cuBLAS GEMM baseline."""

    def __init__(
        self,
        runtime: DistributedRuntime,
        problem: _Problem,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        self.runtime = runtime
        self.problem = problem
        self.input = input_tensor
        self.weight = weight
        device = torch.device("cuda", runtime.rank)
        if problem.workload == "allgather_gemm":
            self.collective_buffer = torch.empty(
                (problem.M, problem.K), dtype=torch.float16, device=device
            )
            self.output = torch.empty(
                (problem.M, problem.local_n), dtype=torch.float16, device=device
            )
        else:
            self.collective_buffer = torch.empty(
                (problem.M, problem.N), dtype=torch.float16, device=device
            )
            self.output = torch.empty(
                (problem.local_m, problem.N), dtype=torch.float16, device=device
            )
        self._graph: torch.cuda.CUDAGraph | None = None

    def launch(self) -> None:
        with torch.cuda.stream(self.runtime.timing_stream):
            if self.problem.workload == "allgather_gemm":
                work = dist.all_gather_into_tensor(
                    self.collective_buffer, self.input, async_op=True
                )
                work.block_current_stream()
                torch.mm(self.collective_buffer, self.weight.T, out=self.output)
            else:
                torch.mm(self.input, self.weight.T, out=self.collective_buffer)
                work = dist.reduce_scatter_tensor(
                    self.output, self.collective_buffer, op=dist.ReduceOp.SUM, async_op=True
                )
                # Join ProcessGroupNCCL's auxiliary stream back to the timing stream.
                work.block_current_stream()

    def capture(self) -> Callable[[], None]:
        """Capture the complete cuBLAS+NCCL operation and return graph replay."""

        if self._graph is not None:
            raise RuntimeError("cuBLAS+NCCL CUDA graph has already been captured")

        # Initialize cuBLAS, ProcessGroupNCCL, and their workspaces before capture.
        self.launch()
        self.runtime.timing_stream.synchronize()
        self.runtime.barrier()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.runtime.timing_stream):
            self.launch()
        self.runtime.timing_stream.synchronize()
        self.runtime.barrier()
        self._graph = graph

        # Validate one replay before returning the closure to the event timer.
        self.replay()
        self.runtime.timing_stream.synchronize()
        self.runtime.barrier()
        return self.replay

    def replay(self) -> None:
        if self._graph is None:
            raise RuntimeError("cuBLAS+NCCL CUDA graph has not been captured")
        with torch.cuda.stream(self.runtime.timing_stream):
            self._graph.replay()


def _load_global(env_name: str, sonames: tuple[str, ...]) -> ctypes.CDLL:
    configured = os.environ.get(env_name)
    candidates = [configured] if configured else []
    candidates.extend(sonames)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    library_name = sonames[0].removeprefix("lib").split(".so")[0]
    discovered = ctypes.util.find_library(library_name)
    if discovered is not None:
        return ctypes.CDLL(discovered, mode=ctypes.RTLD_GLOBAL)
    raise RuntimeError(f"unable to load {sonames[0]}; set {env_name} to its absolute path")


def _load_cublasmp_bindings():
    _load_global(
        "TIRX_CUBLASMP_LIBRARY",
        ("libcublasmp.so.0", "libcublasmp.so", "libcublasMp.so.0", "libcublasMp.so"),
    )
    from nvmath import CudaDataType
    from nvmath.bindings import cublas, cublasMp

    return CudaDataType, cublas, cublasMp


def _all_ranks_available(runtime: DistributedRuntime, available: bool) -> bool:
    flag = torch.tensor(int(available), dtype=torch.int32, device=f"cuda:{runtime.rank}")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _broadcast_nccl_unique_id(runtime: DistributedRuntime, nccl):
    if runtime.rank == 0:
        values = list(nccl.get_unique_id().as_bytes)
    else:
        values = []
    size = torch.tensor([len(values)], dtype=torch.int64, device=f"cuda:{runtime.rank}")
    dist.broadcast(size, src=0)
    if runtime.rank == 0:
        encoded = torch.tensor(values, dtype=torch.uint8, device=f"cuda:{runtime.rank}")
    else:
        encoded = torch.empty(int(size.item()), dtype=torch.uint8, device=f"cuda:{runtime.rank}")
    dist.broadcast(encoded, src=0)
    return nccl.UniqueId.from_bytes(bytes(encoded.cpu().tolist()))


class _CublasMpMatmul:
    """One rank's direct cublasMpMatmul setup for a TP block-noncyclic path."""

    def __init__(
        self,
        runtime: DistributedRuntime,
        problem: _Problem,
        comm: Any,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        algo: str,
        bindings: tuple[Any, Any, Any],
    ) -> None:
        if algo not in _ALGORITHMS:
            raise ValueError(f"unsupported cuBLASMp algorithm: {algo!r}")
        CudaDataType, cublas, cublasMp = bindings
        self._cublas_mp = cublasMp
        self._closed = False
        self._descriptors: list[int] = []
        self._grids: list[int] = []
        self._workspace_registered_grids: list[int] = []

        self._handle = cublasMp.create(runtime.timing_stream.cuda_stream)
        self._matmul_desc = cublasMp.matmul_descriptor_create(cublas.ComputeType.COMPUTE_32F)
        self._set_attr(cublasMp.MatmulDescriptorAttribute.TRANSA, cublas.Operation.T)
        self._set_attr(cublasMp.MatmulDescriptorAttribute.TRANSB, cublas.Operation.N)
        self._set_attr(
            cublasMp.MatmulDescriptorAttribute.ALGO_TYPE,
            getattr(cublasMp.MatmulAlgoType, _ALGORITHMS[algo]),
        )
        if problem.workload == "gemm_reduce_scatter":
            self._set_attr(
                cublasMp.MatmulDescriptorAttribute.COMMUNICATION_TYPE, CudaDataType.CUDA_R_16F
            )

        grid_col = cublasMp.grid_create(
            problem.world_size, 1, cublasMp.GridLayout.COL_MAJOR, comm.ptr
        )
        grid_row = cublasMp.grid_create(
            1, problem.world_size, cublasMp.GridLayout.ROW_MAJOR, comm.ptr
        )
        self._grids.extend((grid_col, grid_row))

        fp16 = CudaDataType.CUDA_R_16F
        matmul_m, matmul_n, matmul_k = problem.N, problem.M, problem.K
        if problem.workload == "allgather_gemm":
            desc_a = cublasMp.matrix_descriptor_create(
                problem.K, problem.N, problem.K, problem.local_n, 0, 0, problem.K, fp16, grid_row
            )
            desc_b = cublasMp.matrix_descriptor_create(
                problem.K, problem.M, problem.K, problem.local_m, 0, 0, problem.K, fp16, grid_row
            )
            desc_d = cublasMp.matrix_descriptor_create(
                problem.N,
                problem.M,
                problem.local_n,
                problem.local_m,
                0,
                0,
                problem.local_n,
                fp16,
                grid_col,
            )
            first_workspace_grid, other_workspace_grid = grid_row, grid_col
        else:
            desc_a = cublasMp.matrix_descriptor_create(
                problem.K,
                problem.N,
                problem.local_k,
                problem.N,
                0,
                0,
                problem.local_k,
                fp16,
                grid_col,
            )
            desc_b = cublasMp.matrix_descriptor_create(
                problem.K,
                problem.M,
                problem.local_k,
                problem.local_m,
                0,
                0,
                problem.local_k,
                fp16,
                grid_col,
            )
            desc_d = cublasMp.matrix_descriptor_create(
                problem.N, problem.M, problem.N, problem.local_m, 0, 0, problem.N, fp16, grid_row
            )
            first_workspace_grid, other_workspace_grid = grid_col, grid_row

        self._descriptors.extend((desc_a, desc_b, desc_d))
        self._dimensions = (matmul_m, matmul_n, matmul_k)
        self._pointers = (weight.data_ptr(), input_tensor.data_ptr(), output.data_ptr())
        self._alpha = np.array(1.0, dtype=np.float32)
        self._beta = np.array(0.0, dtype=np.float32)

        workspace_device, workspace_host = cublasMp.matmul_buffer_size(
            self._handle,
            self._matmul_desc,
            *self._dimensions,
            self._alpha.ctypes.data,
            self._pointers[0],
            1,
            1,
            desc_a,
            self._pointers[1],
            1,
            1,
            desc_b,
            self._beta.ctypes.data,
            0,
            1,
            1,
            desc_d,
            self._pointers[2],
            1,
            1,
            desc_d,
        )
        self.workspace_device_bytes = int(workspace_device)
        self.workspace_host_bytes = int(workspace_host)
        if self.workspace_device_bytes <= 0:
            raise RuntimeError("cuBLASMp returned an empty device workspace")

        self._workspace = cublasMp.malloc(first_workspace_grid, self.workspace_device_bytes)
        self._first_workspace_grid = first_workspace_grid
        cublasMp.buffer_register(other_workspace_grid, self._workspace, self.workspace_device_bytes)
        self._workspace_registered_grids.append(other_workspace_grid)
        self._host_workspace = np.empty(self.workspace_host_bytes, dtype=np.uint8)

    def _set_attr(self, attr, value) -> None:
        holder = ctypes.c_int(int(value))
        self._cublas_mp.matmul_descriptor_set_attribute(
            self._matmul_desc, attr, ctypes.addressof(holder), ctypes.sizeof(holder)
        )

    def launch(self) -> None:
        desc_a, desc_b, desc_d = self._descriptors
        host_workspace = self._host_workspace.ctypes.data if self.workspace_host_bytes else 0
        self._cublas_mp.matmul(
            self._handle,
            self._matmul_desc,
            *self._dimensions,
            self._alpha.ctypes.data,
            self._pointers[0],
            1,
            1,
            desc_a,
            self._pointers[1],
            1,
            1,
            desc_b,
            self._beta.ctypes.data,
            0,
            1,
            1,
            desc_d,
            self._pointers[2],
            1,
            1,
            desc_d,
            self._workspace,
            self.workspace_device_bytes,
            host_workspace,
            self.workspace_host_bytes,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for grid in reversed(self._workspace_registered_grids):
            self._cublas_mp.buffer_deregister(grid, self._workspace)
        self._cublas_mp.free(self._first_workspace_grid, self._workspace)
        for descriptor in reversed(self._descriptors):
            self._cublas_mp.matrix_descriptor_destroy(descriptor)
        self._cublas_mp.matmul_descriptor_destroy(self._matmul_desc)
        self._cublas_mp.destroy(self._handle)
        for grid in reversed(self._grids):
            self._cublas_mp.grid_destroy(grid)


class GemmCommBaselineSuite:
    """Own independent baseline state and expose lazy benchmark builders."""

    def __init__(
        self,
        runtime: DistributedRuntime,
        data: Mapping[str, torch.Tensor],
        *,
        workload: str,
        M: int,
        N: int,
        K: int,
        world_size: int,
        cublasmp_algo: str = "split_p2p",
    ) -> None:
        self.runtime = runtime
        self.problem = _problem(workload, M=M, N=N, K=K, world_size=world_size)
        self.cublasmp_algo = cublasmp_algo
        self._cublasmp: _CublasMpMatmul | None = None
        self._cublasmp_comm = None
        self._cublasmp_error: str | None = None
        self._cublas_nccl = _CublasNcclLaunch(runtime, self.problem, data["A"], data["B"])
        self._build_cublasmp(data["A"], data["B"])

    def _build_cublasmp(self, input_tensor: torch.Tensor, weight: torch.Tensor) -> None:
        try:
            bindings = _load_cublasmp_bindings()
            import nccl.core as nccl

            local_error = None
        except BaseException as error:
            bindings = None
            nccl = None
            local_error = f"{type(error).__name__}: {error}"

        if not _all_ranks_available(self.runtime, local_error is None):
            self._cublasmp_error = local_error or "cuBLASMp is unavailable on another rank"
            return
        assert bindings is not None and nccl is not None

        unique_id = _broadcast_nccl_unique_id(self.runtime, nccl)
        self._cublasmp_comm = nccl.Communicator.init(
            self.problem.world_size, self.runtime.rank, unique_id
        )
        output = torch.empty_like(self._cublas_nccl.output)
        self._cublasmp = _CublasMpMatmul(
            self.runtime,
            self.problem,
            self._cublasmp_comm,
            input_tensor,
            weight,
            output,
            self.cublasmp_algo,
            bindings,
        )
        self._cublasmp_output = output

    def _build_cublas_nccl(self) -> Callable[[], None]:
        return self._cublas_nccl.capture()

    def _build_cublasmp_launch(self) -> Callable[[], None]:
        if self._cublasmp is None:
            raise RuntimeError(self._cublasmp_error or "cuBLASMp setup failed")
        self._cublasmp.launch()
        self.runtime.timing_stream.synchronize()
        tolerance = (
            {"rtol": 1e-3, "atol": 2e-2}
            if self.problem.workload == "allgather_gemm"
            else {"rtol": 6e-2, "atol": 6e-2}
        )
        torch.testing.assert_close(self._cublasmp_output, self._cublas_nccl.output, **tolerance)
        return self._cublasmp.launch

    def references(self) -> dict[str, Callable[[], Callable[[], None]]]:
        return {
            "cublas_nccl_cudagraph": self._build_cublas_nccl,
            f"cublasmp_{self.cublasmp_algo}": self._build_cublasmp_launch,
        }

    def metadata(self) -> dict[str, Any]:
        libraries = _library_provenance()
        result: dict[str, Any] = {
            "cublas_nccl_execution": "cuda_graph_replay",
            "cublasmp_algorithm": self.cublasmp_algo,
            "libraries": libraries,
        }
        if self._cublasmp is not None:
            binding_version = int(self._cublasmp._cublas_mp.get_version())
            if binding_version != libraries["cublasmp"]["version_raw"]:
                raise RuntimeError(
                    "cuBLASMp binding/library version mismatch: "
                    f"binding={binding_version}, library={libraries['cublasmp']['version_raw']}"
                )
            result.update(
                {
                    "cublasmp_version": binding_version,
                    "cublasmp_workspace_device_bytes": self._cublasmp.workspace_device_bytes,
                    "cublasmp_workspace_host_bytes": self._cublasmp.workspace_host_bytes,
                }
            )
        elif self._cublasmp_error is not None:
            result["cublasmp_error"] = self._cublasmp_error
        return result

    def close(self) -> None:
        self.runtime.barrier()
        if self._cublasmp is not None:
            self._cublasmp.close()
        if self._cublasmp_comm is not None and self._cublasmp_comm.is_valid:
            self._cublasmp_comm.destroy()
        self.runtime.barrier()


def create_baseline_suite(
    runtime: DistributedRuntime,
    data: Mapping[str, torch.Tensor],
    *,
    workload: str,
    M: int,
    N: int,
    K: int,
    world_size: int,
) -> GemmCommBaselineSuite:
    return GemmCommBaselineSuite(
        runtime, data, workload=workload, M=M, N=N, K=K, world_size=world_size
    )


def ratios(result: Mapping[str, Any], tirx: str = "tirx") -> dict[str, float]:
    """Return ``baseline_us / tirx_us``; values above one favor TIRx."""

    impls = result.get("impls")
    if not isinstance(impls, Mapping) or tirx not in impls:
        raise ValueError(f"benchmark result does not contain {tirx!r}")
    tirx_us = float(impls[tirx])
    if not math.isfinite(tirx_us) or tirx_us <= 0:
        raise ValueError(f"benchmark result contains invalid {tirx!r} timing: {tirx_us}")
    return {
        name: float(value) / tirx_us
        for name, value in impls.items()
        if name != tirx and math.isfinite(float(value)) and float(value) > 0
    }


__all__ = ["GemmCommBaselineSuite", "create_baseline_suite", "ratios"]
