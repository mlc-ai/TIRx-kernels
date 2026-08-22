# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

import os
import signal

import pytest

from tirx_kernels import runner


class _Interrupted(BaseException):
    pass


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
