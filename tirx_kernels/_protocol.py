# Copyright (c) 2026 The TIRx Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
    - "category" (str): the bucket subdirectory the module lives in — one of
      basic, deepgemm, flashattention, flashinfer, flashmla.
      Each bucket holds the kernels ported from one upstream project;
      ``basic`` holds native TIRx kernels with no single upstream project.
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
