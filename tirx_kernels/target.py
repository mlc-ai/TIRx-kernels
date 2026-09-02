# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Prepare-time CUDA target selection shared by runners and kernel guards."""

from __future__ import annotations

import os
from collections.abc import Sequence

PREPARE_CUDA_ARCH_ENV = "TIRX_PREPARE_CUDA_ARCH"


def prepare_cuda_arch(declared: str | None = None) -> str | None:
    """Return the explicit prepare arch, falling back to a declared arch."""
    arch = os.environ.get(PREPARE_CUDA_ARCH_ENV) or declared
    if arch is not None and not arch.startswith("sm_"):
        raise ValueError(f"{PREPARE_CUDA_ARCH_ENV} must start with 'sm_', got {arch!r}")
    return arch


def supports_sm100_kernel(capability: Sequence[int], *, declared_arch: str = "sm_100a") -> bool:
    """Whether an SM100 kernel may be attempted on this prepared target.

    The native SM100-family allowlist stays exact.  Thor is admitted only for
    the exact SM11.0 capability and only when the process explicitly prepares
    ``sm_110a`` code; unrelated present or future architectures do not fall
    through a broad major-version check.
    """
    if len(capability) != 2:
        raise ValueError(f"CUDA capability must contain major and minor, got {capability!r}")
    major, minor = (int(part) for part in capability)
    if (major, minor) in {(10, 0), (10, 3), (10, 7)}:
        return True
    return (major, minor) == (11, 0) and prepare_cuda_arch(declared_arch) == "sm_110a"


__all__ = ["PREPARE_CUDA_ARCH_ENV", "prepare_cuda_arch", "supports_sm100_kernel"]
