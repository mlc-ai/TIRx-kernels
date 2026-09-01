# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import pytest

import tirx_kernels.kern as K
from tirx_kernels.kern.low_level_ir import LowLevelIRContractError, check_low_level_ir
from tirx_kernels.runner import run_kernel_test


def _build_kernel_with_buffer_access(scope: str, access: str):
    @K.kernel(warps=1, arch="sm_100a", grid=False, check_ir=True)
    def probe(global_buffer: K.gptr("float32")):
        buffer = (
            global_buffer if scope == "global" else K.alloc_buffer([1], "float32", scope="shared")
        )
        if access == "load":
            local = K.alloc_local([1], "float32")
            K.buffer_store(local, buffer[0], [0])
        elif access == "store":
            K.buffer_store(buffer, K.float32(1), [0])
        else:
            K.keep_alive(K.address_of(buffer[0]))

    return probe


def _kernel_with_func_call(callee: str):
    @K.kernel(warps=1, arch="sm_100a", grid=False, check_ir=False)
    def main():
        K.cuda.func_call(callee, source_code="__device__ void ignored() {}")

    return main.func


@pytest.mark.parametrize("scope", ["global", "shared"])
def test_forbidden_tensor_load_is_reported(scope):
    with pytest.raises(LowLevelIRContractError) as error:
        _build_kernel_with_buffer_access(scope, "load")

    assert [
        (finding.kind, finding.node_type, finding.scope)
        for finding in error.value.report.violations
    ] == [("buffer_load", "TensorLoad", scope)]


@pytest.mark.parametrize("scope", ["global", "shared"])
def test_forbidden_buffer_store_is_reported(scope):
    with pytest.raises(LowLevelIRContractError) as error:
        _build_kernel_with_buffer_access(scope, "store")

    assert [
        (finding.kind, finding.node_type, finding.scope)
        for finding in error.value.report.violations
    ] == [("buffer_store", "BufferStore", scope)]


@pytest.mark.parametrize("scope", ["global", "shared"])
def test_address_of_tensor_load_is_not_a_memory_read(scope):
    kernel = _build_kernel_with_buffer_access(scope, "address")
    report = check_low_level_ir(kernel.func)

    assert report.ok
    assert [
        (finding.kind, finding.node_type, finding.scope) for finding in report.address_only_loads
    ] == [("address_only_buffer_load", "TensorLoad", scope)]


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


def test_setmaxnreg_requires_pinned_entry_allocation():
    def build(min_blocks_per_sm):
        @K.kernel(
            warps=4, arch="sm_100a", min_blocks_per_sm=min_blocks_per_sm, grid=False, check_ir=False
        )
        def probe():
            K.ptx.setmaxnreg.dec.sync.aligned.u32(K.uint32(64))

        return probe.func

    with pytest.raises(LowLevelIRContractError) as error:
        check_low_level_ir(build(None))
    assert [finding.kind for finding in error.value.report.violations] == [
        "setmaxnreg_without_min_blocks_per_sm"
    ]
    assert "setmaxnreg_without_min_blocks_per_sm" in str(error.value)

    assert check_low_level_ir(build(1)).ok


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
