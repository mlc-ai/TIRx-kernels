# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Prepare-time CUDA target selection shared by runners and kernel guards."""

from __future__ import annotations

import os
from collections.abc import Sequence
from math import prod

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


def prepare_cluster_shape(
    cluster_shape: Sequence[int], *, declared_arch: str = "sm_100a"
) -> tuple[int, ...]:
    """Return a launchable cluster shape for the explicitly prepared target.

    B200 accepts the non-portable 16-block clusters used by a few persistent
    SM100 schedules. Thor accepts at most eight blocks per cluster. Preserve
    the M dimension whenever possible because two-CTA MMA atoms require it to
    remain divisible by two, and shrink only the schedule -- never the problem
    shape or numerical operation.
    """
    shape = tuple(int(extent) for extent in cluster_shape)
    if not shape or any(extent <= 0 for extent in shape):
        raise ValueError(f"cluster shape must contain positive extents, got {shape!r}")
    if prepare_cuda_arch(declared_arch) != "sm_110a" or prod(shape) <= 8:
        return shape
    mutable = list(shape)
    while prod(mutable) > 8:
        shrink_axis = next(
            (axis for axis in range(len(mutable) - 1, -1, -1) if mutable[axis] % 2 == 0),
            None,
        )
        if shrink_axis is None:
            raise ValueError(f"cannot reduce cluster shape {shape!r} to eight blocks")
        mutable[shrink_axis] //= 2
    return tuple(mutable)


__all__ = [
    "PREPARE_CUDA_ARCH_ENV",
    "prepare_cluster_shape",
    "prepare_cuda_arch",
    "supports_sm100_kernel",
]
