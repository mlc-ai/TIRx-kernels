# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import pytest

from tirx_kernels.low_level_ir import (
    LOW_LEVEL_IR_FUNC_CALL_EXCEPTIONS_BY_KERNEL,
    NVSHMEM_RUNTIME_FUNC_CALLS,
    LowLevelIRContractError,
    check_low_level_ir,
)
from tvm.script import tirx as T


def _kernel_with_func_call(callee: str):
    @T.prim_func
    def main():
        T.evaluate(T.cuda.func_call(callee, source_code="__device__ void ignored() {}"))

    return main


def test_func_call_is_rejected_by_default_and_reports_callee():
    with pytest.raises(LowLevelIRContractError) as error:
        check_low_level_ir(_kernel_with_func_call("unexpected_helper"))

    report = error.value.report
    assert [finding.callee for finding in report.func_calls] == ["unexpected_helper"]
    assert "callee=unexpected_helper" in str(error.value)


def test_only_exact_nvshmem_runtime_calls_are_exempt_for_gemm_reduce_scatter():
    assert LOW_LEVEL_IR_FUNC_CALL_EXCEPTIONS_BY_KERNEL == {
        "gemm_reduce_scatter": frozenset(
            {
                "enqueue_remote",
                "exit_barrier_arrive_and_wait",
                "ld_reduce_8_fp16",
                "semaphore_notify_remote",
            }
        )
    }

    for callee in NVSHMEM_RUNTIME_FUNC_CALLS:
        report = check_low_level_ir(
            _kernel_with_func_call(callee), allowed_func_calls=NVSHMEM_RUNTIME_FUNC_CALLS
        )
        assert report.ok
        assert [finding.callee for finding in report.func_calls] == [callee]

    with pytest.raises(LowLevelIRContractError):
        check_low_level_ir(
            _kernel_with_func_call("another_runtime_helper"),
            allowed_func_calls=NVSHMEM_RUNTIME_FUNC_CALLS,
        )
