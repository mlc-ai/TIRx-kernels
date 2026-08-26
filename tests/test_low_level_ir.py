# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import pytest

import tirx_kernels.kern as K
from tirx_kernels.low_level_ir import (
    LOW_LEVEL_IR_FUNC_CALL_EXCEPTIONS_BY_KERNEL,
    NVSHMEM_RUNTIME_FUNC_CALLS,
    LowLevelIRContractError,
    check_low_level_ir,
)


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
    assert LOW_LEVEL_IR_FUNC_CALL_EXCEPTIONS_BY_KERNEL == {
        "cudnn_sm100_bsa_forward_blk64": frozenset({"tirx_bsa_pv_mma_chain"}),
        "cudnn_sm100_csa_compressor_fwd": frozenset(
            {
                "tirx_csa_exp_constants",
                "tirx_csa_load4_bf16x2",
                "tirx_csa_ordered_max",
                "tirx_csa_scan_boundary",
                "tirx_csa_source_exp",
            }
        ),
        "gemm_reduce_scatter": frozenset(
            {
                "enqueue_remote",
                "exit_barrier_arrive_and_wait",
                "ld_reduce_8_fp16",
                "semaphore_notify_remote",
            }
        ),
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


def test_kern_smem_descriptor_uniformity_stays_in_low_level_contract():
    @K.kernel(warps=1, arch="sm_100a", grid=False)
    def probe():
        descriptor = K.SmemDescriptor()
        descriptor.make_lo_uniform()
        descriptor.add_16B_offset(K.int32(1))

    report = check_low_level_ir(probe.func)

    assert report.ok
    assert report.func_calls == ()
