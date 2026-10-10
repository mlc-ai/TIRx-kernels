# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Standard kernel interface protocol.

Native kernels live under ``tirx_kernels/<workload>/``; external ports live
under ``tirx_kernels/ported/<upstream>/``. Every discoverable kernel module
must expose the members below. A category may group its kernels into further
subpackages (``ported/flashinfer/`` groups ports by FlashInfer entry point);
the registry walks into those, skipping private and ``utils`` subpackages.

Module-level constants
----------------------
KERNEL_META : dict
    Required keys:
    - "name" (str): unique kernel name used by CLI (e.g. "rmsnorm")
    - "category" (str): the workload package for native kernels (e.g. ``kda``
      or ``gemm``), or ``ported.<upstream>`` for external ports (e.g.
      ``ported.flashinfer``). This must match the containing category package.
    - "runtime_cuda_archs" (list[str]): exact CUDA architectures on which the
      kernel is allowed to compile and run (e.g. ["sm_100a"]).
    Optional keys:
    - "reference_requirements" (tuple[dict, ...]): correctness-only external
      reference contracts. Each item has distribution "package", Python
      "import", and at least one PEP 440 "specifier" or exact "git" identity
      with a canonical URL and full commit SHA.

CONFIGS : list[dict]
    Each dict has a "label" key (str) plus arbitrary kernel-specific
    parameters.  The same config matrix is used by correctness tests and
    benchmark runs.

Functions
---------
get_kernel(*, compile_config=None, **cfg) -> PrimFunc | IRModule | nested function collection
    Return the pre-lowering TIRx function or functions for this kernel.
    ``compile_config`` is an immutable ``tvm.backend.cuda.CompileConfig``.
    Factories receive it explicitly; architecture-sensitive factories record
    their selected architecture on the device entry. ``prepare_bench`` and
    ``run_test`` accept and forward the same setting to compilation.
    Multi-kernel workloads may return an ``IRModule`` or a nested list,
    tuple, or mapping containing ``tvm.tirx.PrimFunc`` objects.

prepare_data(**cfg) -> dict[str, Any]
    Prepare input/output tensors.  Returns a dict mapping argument names
    to tensors (torch.Tensor or numpy.ndarray).

check_correctness(outputs: dict, **cfg) -> None
    Validate kernel outputs against a reference.
    Raise AssertionError on mismatch.

get_baselines(**cfg) -> dict[str, Callable]   (optional)
    Return {name: callable} for baseline implementations used in
    benchmarking (e.g. cublas, flashinfer).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class KernelModule(Protocol):
    """Structural type that a kernel module must satisfy."""

    KERNEL_META: dict[str, Any]
    CONFIGS: list[dict[str, Any]]

    @staticmethod
    def get_kernel(*, compile_config=None, **kwargs: Any) -> Any: ...

    @staticmethod
    def prepare_data(**kwargs: Any) -> dict[str, Any]: ...

    @staticmethod
    def check_correctness(outputs: dict[str, Any], **kwargs: Any) -> None: ...
