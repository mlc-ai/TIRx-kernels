# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Standard kernel interface protocol.

Every kernel module under ``tirx_kernels/<category>/`` that wants to be
discoverable by the registry must expose the members below.  A category may
group its kernels into further subpackages (``flashinfer/`` buckets its ports
by the FlashInfer Python entry point they back); the registry walks into those,
skipping ``utils`` subpackages.

Module-level constants
----------------------
KERNEL_META : dict
    Required keys:
    - "name" (str): unique kernel name used by CLI (e.g. "rmsnorm")
    - "category" (str): the bucket subdirectory the module lives in. Port
      buckets are named for their upstream project; ``basic`` holds native
      TIRx kernels, and ``agent_evolved`` holds curated kernels selected from
      measured agent-evolution runs.
    - "compute_capability" (int): minimum SM version (e.g. 10 for sm100a)

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
