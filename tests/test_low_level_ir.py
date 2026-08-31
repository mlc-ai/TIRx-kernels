# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import pytest

import tirx_kernels.kern as K
from tirx_kernels.kern.low_level_ir import LowLevelIRContractError, check_low_level_ir
from tirx_kernels.runner import run_kernel_test


def _kernel_with_func_call(callee: str):
    @K.kernel(warps=1, arch="sm_100a", grid=False, check_ir=False)
    def main():
        K.cuda.func_call(callee, source_code="__device__ void ignored() {}")

    return main.func


def test_func_call_is_rejected_by_default_and_reports_callee():
    with pytest.raises(LowLevelIRContractError) as error:
        check_low_level_ir(_kernel_with_func_call("unexpected_helper"))

    report = error.value.report
    assert [finding.callee for finding in report.func_calls] == ["unexpected_helper"]
    assert "callee=unexpected_helper" in str(error.value)


def test_only_exact_kernel_local_helpers_are_exempt():
    allowed_func_calls = frozenset({"expected_runtime_helper"})

    report = check_low_level_ir(
        _kernel_with_func_call("expected_runtime_helper"), allowed_func_calls=allowed_func_calls
    )
    assert report.ok
    assert [finding.callee for finding in report.func_calls] == ["expected_runtime_helper"]

    with pytest.raises(LowLevelIRContractError):
        check_low_level_ir(
            _kernel_with_func_call("another_runtime_helper"), allowed_func_calls=allowed_func_calls
        )


def test_correctness_runner_does_not_rebuild_an_already_checked_kernel():
    class KernelModule:
        @staticmethod
        def get_kernel(**_params):
            raise AssertionError("correctness runner rebuilt the kernel")

        @staticmethod
        def run_test(**params):
            assert params == {"value": 3}

    run_kernel_test("probe", {"label": "case", "value": 3}, registry={"probe": KernelModule})


def test_kern_smem_descriptor_uniformity_stays_in_low_level_contract():
    @K.kernel(warps=1, arch="sm_100a", grid=False)
    def probe():
        descriptor = K.SmemDescriptor()
        descriptor.make_lo_uniform()
        descriptor.add_16B_offset(K.int32(1))

    report = check_low_level_ir(probe.func)

    assert report.ok
    assert report.func_calls == ()
