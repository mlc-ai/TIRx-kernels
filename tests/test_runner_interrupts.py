# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

import os
import signal

import pytest

from tirx_kernels import runner


class _Interrupted(BaseException):
    pass


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a", "sm_110a"])
def test_prepare_arch_overrides_declared_cuda_arch(monkeypatch, arch):
    monkeypatch.setenv(runner.PREPARE_CUDA_ARCH_ENV, arch)
    assert runner.cuda_target(arch="sm_100a").arch == arch


def test_thor_sm100_compatibility_requires_explicit_sm110a_prepare(monkeypatch):
    from tirx_kernels.runner import supports_sm100_kernel

    monkeypatch.delenv(runner.PREPARE_CUDA_ARCH_ENV, raising=False)
    for capability in ((10, 0), (10, 3), (10, 7)):
        assert supports_sm100_kernel(capability)
    assert not supports_sm100_kernel((11, 0))
    assert not supports_sm100_kernel((12, 0))

    monkeypatch.setenv(runner.PREPARE_CUDA_ARCH_ENV, "sm_110a")
    assert supports_sm100_kernel((11, 0))
    assert not supports_sm100_kernel((12, 0))


@pytest.mark.parametrize("arch", [None, "sm_100a", "sm_103a", "sm_107a"])
def test_thor_cluster_shape_limit(monkeypatch, arch):
    from tirx_kernels.runner import prepare_cluster_shape

    if arch is None:
        monkeypatch.delenv(runner.PREPARE_CUDA_ARCH_ENV, raising=False)
    else:
        monkeypatch.setenv(runner.PREPARE_CUDA_ARCH_ENV, arch)
    assert prepare_cluster_shape((2, 8)) == (2, 8)
    monkeypatch.setenv(runner.PREPARE_CUDA_ARCH_ENV, "sm_110a")
    assert prepare_cluster_shape((2, 8)) == (2, 4)
    assert prepare_cluster_shape((1, 16)) == (1, 8)
    assert prepare_cluster_shape((16, 1)) == (8, 1)
    assert prepare_cluster_shape((2, 4)) == (2, 4)


@pytest.fixture
def bench_child_handler():
    """Install the bench child's SIGUSR1 handler shape for the test's duration."""

    def handler(_signum, _frame):
        if runner.gpu_interrupt_should_defer():
            return
        raise _Interrupted()

    previous = signal.signal(signal.SIGUSR1, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGUSR1, previous)


def test_interrupt_outside_critical_section_raises_immediately(bench_child_handler):
    with pytest.raises(_Interrupted):
        os.kill(os.getpid(), signal.SIGUSR1)


def test_interrupt_inside_critical_section_is_redelivered_at_exit(bench_child_handler):
    """A signal landing mid-region must not unwind the region (an import or JIT
    would be left half-executed); it is redelivered once the region exits."""
    survived = []
    with pytest.raises(_Interrupted):
        with runner.defer_gpu_interrupts():
            os.kill(os.getpid(), signal.SIGUSR1)
            survived.append("region completed")
    assert survived == ["region completed"]


def test_nested_critical_sections_defer_until_outermost_exit(bench_child_handler):
    survived = []
    with pytest.raises(_Interrupted):
        with runner.defer_gpu_interrupts():
            with runner.defer_gpu_interrupts():
                os.kill(os.getpid(), signal.SIGUSR1)
            survived.append("inner exited, still deferred")
    assert survived == ["inner exited, still deferred"]


def test_quiet_critical_section_does_not_redeliver(bench_child_handler):
    with runner.defer_gpu_interrupts():
        pass
    # No signal arrived inside the region: exiting must not synthesize one,
    # and a later signal still raises immediately.
    with pytest.raises(_Interrupted):
        os.kill(os.getpid(), signal.SIGUSR1)
