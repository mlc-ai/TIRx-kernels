# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Standard kernel interface protocol.

Every kernel module under ``tirx_kernels/<task_set>/<task>/<device>/`` that wants to be
discoverable by the registry must expose the members below.  Private helper directories are not independently registered. JSON task metadata
is optional; discovery still reads literal KERNEL_META declarations.

Module-level constants
----------------------
KERNEL_META : dict
    Required keys:
    - "name" (str): unique kernel name used by CLI (e.g. "rmsnorm")
    - "category" (str): containing task set, such as ``basic`` or ``flashinfer``.
      Source provenance does not create a separate task or implementation copy.
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
get_kernel(**cfg) -> PrimFunc | IRModule | nested function collection
    Return the pre-lowering TIRx function or functions for this kernel.
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
    def get_kernel(**kwargs: Any) -> Any: ...

    @staticmethod
    def prepare_data(**kwargs: Any) -> dict[str, Any]: ...

    @staticmethod
    def check_correctness(outputs: dict[str, Any], **kwargs: Any) -> None: ...
