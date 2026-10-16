# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Explicit host/device entry boundaries and launch ABI contracts."""

import subprocess
import sys

import numpy as np
import pytest
from tvm_ffi import structural_walk

import tirx_kernels.tirx_lite as txl
import tvm_tirx
from tirx_kernels.tirx_lite import entry
from tvm_tirx import ir
from tvm_tirx.script.ir_builder import IRBuilder


def _nodes(value, kind):
    result = []
    structural_walk(value, (kind, result.append))
    return result


def _split(kernel):
    target = tvm_tirx.target.Target("cuda", host="llvm")
    with target:
        mod = tvm_tirx.tirx.transform.LowerTIRx()(
            tvm_tirx.IRModule({"main": kernel.func.with_attr("target", target)})
        )
        mod = tvm_tirx.tirx.transform.SplitHostDevice()(mod)
    device = [f for f in mod.functions.values() if f.attrs and f.attrs.get("calling_conv") == 2]
    assert len(device) == 1
    return mod["main"], device[0]


def test_tensormap_parameter_uses_addressable_pointer_type():
    @txl.kernel()
    def probe(descriptor: txl.Ptr(txl.TensorMap)):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=32))
        txl.ptx.prefetch.tensormap(txl.address_of(descriptor))

    parameter = probe.func.params[0]
    assert isinstance(parameter.ty, ir.PtrType)
    assert isinstance(parameter.ty.element_type, tvm_tirx.tirx.TensorMapType)
    address = next(
        call for call in _nodes(probe.func, ir.Call) if call.op.name == "tirx.address_of"
    )
    assert address.ty == ir.PrimType("uint64")
    _, device = _split(probe)
    assert isinstance(device.params[0].ty.element_type, tvm_tirx.tirx.TensorMapType)


def test_flat_and_with_entries_generate_the_same_ir():
    def build(managed):
        @txl.kernel()
        def probe(out: txl.Tensor((txl.int64(),), txl.i32), n: txl.i32):
            def body():
                tile = txl.smem_pool().alloc((32,), txl.i32)
                txl.ptx.st.shared.b32(tile.ptr_to([txl.thread_id()]), txl.cta_id())
                txl.ptx.st.global_.b32(out.ptr_to([txl.thread_id()]), txl.cta_id())

            config = txl.cuda.LaunchConfig(grid=n, block=32)
            if managed:
                with txl.device_entry(launch=config):
                    body()
            else:
                txl.device_entry(launch=config)
                body()

        return probe

    flat, managed = build(False), build(True)
    ir.assert_structural_equal(flat.func, managed.func, map_free_vars=True)
    assert flat.warps == managed.warps == 1
    assert flat.session.pool.bytes == managed.session.pool.bytes


def test_host_preparation_and_tail_remain_outside_device_region():
    def prepare(a):
        descriptor = txl.stack_alloca("tensormap", 1)
        txl.cu_tensor_map_encode_tiled(descriptor, "float32", 1, a.data, 32, 32, 1, 0, 0, 0, 0)
        return descriptor

    @txl.kernel()
    def probe(a: txl.Tensor((txl.int64(),), txl.f32), n: txl.i32):
        descriptor = prepare(a)
        with txl.device_entry(launch=txl.cuda.LaunchConfig(grid=n, block=32)):
            txl.keep_alive(txl.address_of(descriptor))
        txl.call_packed("entry_test.host_tail")

    regions = _nodes(probe.func, ir.RegionStmt)
    device = next(r for r in regions if r.op.name == "tirx.device_entry")
    calls = _nodes(probe.func, ir.Call)
    device_calls = _nodes(device.body, ir.Call)
    for name in ("tirx.stack_alloca", "tirx.cuda.tensormap_encode_tiled", "tirx.call_packed"):
        assert any(call.op.name == name for call in calls)
        assert not any(call.op.name == name for call in device_calls)
    host, gpu = _split(probe)
    assert len(gpu.params) == 1  # The descriptor, without a or n.
    assert len(host.params) > 0


@pytest.mark.parametrize("device_uses_n", [False, True])
def test_launch_only_values_stay_on_host(device_uses_n):
    @txl.kernel()
    def probe(
        out: txl.Tensor((txl.int64(),), txl.i32),
        n: txl.i32,
        stream: txl.Tensor((txl.int64(),), txl.u8),
    ):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=(n + 1) // 2, block=32, stream=stream.data)
        )
        txl.ptx.st.global_.b32(out.ptr_to([txl.thread_id()]), n if device_uses_n else txl.cta_id())

    host, gpu = _split(probe)
    assert len(gpu.params) == (2 if device_uses_n else 1)
    launches = [c for c in _nodes(host, ir.Call) if c.op.name == "tirx.call_ffi_kernel"]
    assert len(launches) == 1
    assert launches[0].attrs.num_kernel_args == len(gpu.params)
    assert "stream" in launches[0].attrs.launch_fields


@pytest.mark.parametrize("option", ["launch", "kernel_attrs", "host_prelude"])
def test_decorator_rejects_retired_configuration_options(option):
    with pytest.raises(TypeError, match=option):
        txl.kernel(**{option: None})


def test_device_entry_requires_a_concrete_launch_record():
    with pytest.raises(TypeError, match="LaunchConfig"):

        @txl.kernel()
        def probe():
            txl.device_entry(launch=lambda params: txl.cuda.LaunchConfig(grid=1, block=32))


def test_host_is_an_ordinary_annotated_parameter():
    @txl.kernel()
    def probe(*, host: txl.i32):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=host, block=32))

    assert probe.func.params[0].name == "host"


@pytest.mark.parametrize("block", [0, 16, 33, 1056, (32, 2), "dynamic"])
def test_invalid_blocks_are_rejected(block):
    with pytest.raises((TypeError, ValueError)):

        @txl.kernel()
        def probe(n: txl.i32):
            txl.device_entry(
                launch=txl.cuda.LaunchConfig(grid=1, block=n if block == "dynamic" else block)
            )


@pytest.mark.parametrize("kind", ["missing", "nested", "second", "branch", "loop"])
def test_entry_must_be_unique_and_at_function_scope(kind):
    with pytest.raises(RuntimeError, match=r"exactly one|function scope"):

        @txl.kernel()
        def probe():
            config = txl.cuda.LaunchConfig(grid=1, block=32)
            if kind == "missing":
                return
            if kind == "branch":
                with txl.If(txl.int32(1) == 1), txl.Then():
                    txl.device_entry(launch=config)
            elif kind == "loop":
                with txl.serial(2):
                    txl.device_entry(launch=config)
            else:
                with txl.device_entry(launch=config):
                    if kind == "nested":
                        txl.device_entry(launch=config)
                if kind == "second":
                    txl.device_entry(launch=config)


@pytest.mark.parametrize(
    "helper", [txl.cta_id, txl.thread_id, txl.warp_id, txl.lane_id, txl.specialize, txl.smem_pool]
)
@pytest.mark.parametrize("after", [False, True])
def test_device_helpers_require_an_active_region(helper, after):
    with pytest.raises(RuntimeError, match=r"active txl\.device_entry"):

        @txl.kernel()
        def probe():
            if after:
                with txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=32)):
                    pass
            helper()


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("failure", ["body", "finalize"])
def test_failures_restore_tracing_state(monkeypatch, managed, failure):
    from tirx_kernels.tirx_lite.smem import SmemPool

    previous = sys.gettrace()
    original_commit = SmemPool.commit

    def fail_commit(self, size=None):
        raise ValueError("finalize failed")

    if failure == "finalize":
        monkeypatch.setattr(SmemPool, "commit", fail_commit)
    with pytest.raises(ValueError, match="failed"):

        @txl.kernel()
        def probe():
            def body():
                txl.smem_pool().alloc((32,), txl.i32)
                if failure == "body":
                    raise ValueError("body failed")

            config = txl.cuda.LaunchConfig(grid=1, block=32)
            if managed:
                with txl.device_entry(launch=config):
                    body()
            else:
                txl.device_entry(launch=config)
                body()

    assert sys.gettrace() is previous
    assert entry.current(required=False) is None
    assert not IRBuilder.is_in_scope()
    monkeypatch.setattr(SmemPool, "commit", original_commit)

    @txl.kernel()
    def next_kernel():
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=32))
        txl.smem_pool().alloc((32,), txl.i32)

    assert next_kernel.warps == 1


def test_nested_kernel_tracing_restores_the_outer_device_session():
    @txl.kernel()
    def outer():
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=64))
        before = txl.thread_id()

        @txl.kernel()
        def inner():
            assert entry.current(required=False) is None
            txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=32))

        assert inner.warps == 1
        assert txl.thread_id().same_as(before)

    assert outer.warps == 2


def _compile(kernel, host, tmp_path):
    from tvm_tirx.backend.cuda import export_cuda_host
    from tvm_tirx.testing import env

    arch = env.cuda_arch(0)
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch}, host=host)
    module = tvm_tirx.tirx.compile(kernel.func, target=target, pipeline="tirx").mod
    if host == "llvm":
        return module, None
    import tvm_ffi.cpp

    source = export_cuda_host(module)
    assert "#include <tvm/runtime/" not in source
    library = tvm_ffi.cpp.build_inline(
        name=kernel.name,
        cuda_sources=source,
        extra_cuda_cflags=[f"-arch={arch}"],
        extra_ldflags=["-lcuda"],  # Host TensorMap encoding uses the CUDA Driver API.
        build_directory=str(tmp_path),
        backend="cuda",
    )
    return tvm_ffi.load_module(library), library


@pytest.mark.skipif(not tvm_tirx.cuda().exist, reason="requires CUDA")
@pytest.mark.parametrize("host", ["llvm", "cuda_host"])
def test_native_tensor_arguments_check_shared_shapes(host, tmp_path):
    import torch

    size = txl.Var("size", txl.int64)

    @txl.kernel()
    def copy_tensor(x: txl.Tensor((size,), txl.f32), y: txl.Tensor((size,), txl.f32)):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=(x.shape[0] + 31) // 32, block=32))
        i = txl.cta_id() * 32 + txl.thread_id()
        with txl.If(i < x.shape[0]), txl.Then():
            value = txl.local_scalar("float32")
            txl.ptx.ld.global_.f32(value, x.ptr_to([i]))
            txl.ptx.st.global_.f32(y.ptr_to([i]), value)

    module, _ = _compile(copy_tensor, host, tmp_path)
    copy = module[copy_tensor.name]
    x = torch.arange(37, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    copy(x, y)
    torch.testing.assert_close(y, x)
    for invalid, error, message in (
        (torch.empty(38, device="cuda"), ValueError, "expected to match"),
        (torch.empty(37, dtype=torch.int32, device="cuda"), TypeError, "expected float32"),
        (torch.empty((1, 37), device="cuda"), ValueError, "expected 1"),
    ):
        with pytest.raises(error, match=message):
            copy(x, invalid)


@pytest.mark.skipif(not tvm_tirx.cuda().exist, reason="requires CUDA")
@pytest.mark.parametrize("host", ["llvm", "cuda_host"])
@pytest.mark.parametrize("prepare_map", [False, True])
def test_dynamic_launch_and_host_descriptors_execute(host, prepare_map, tmp_path):
    if prepare_map and int(tvm_tirx.cuda().compute_version.split(".")[0]) < 9:
        pytest.skip("tensor map encoding requires SM90 or newer")

    @txl.kernel()
    def coordinates(out: txl.Tensor((txl.int64(),), txl.i32), n: txl.i32):
        if prepare_map:
            descriptor = txl.stack_alloca("tensormap", 1)
            txl.cu_tensor_map_encode_tiled(
                descriptor, "int32", 1, out.data, n * 32, 32, 1, 0, 0, 0, 0
            )
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=n, block=32))
        if prepare_map:
            txl.keep_alive(txl.address_of(descriptor))
        txl.ptx.st.global_.b32(out.ptr_to([txl.cta_id() * 32 + txl.thread_id()]), txl.cta_id())

    module, library = _compile(coordinates, host, tmp_path)
    for n in (1, 3, 2):
        out = tvm_tirx.runtime.empty((n * 32,), "int32", tvm_tirx.cuda())
        module[coordinates.name](out, n)
        np.testing.assert_array_equal(out.numpy(), np.repeat(np.arange(n, dtype="int32"), 32))
    if library:
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
import torch
import tvm_ffi
assert "tvm" not in sys.modules
assert "tvm_tirx" not in sys.modules
module = tvm_ffi.load_module(sys.argv[1])
out = torch.empty(64, dtype=torch.int32, device="cuda")
module["coordinates"](out, 2)
expected = torch.arange(2, dtype=torch.int32, device="cuda").repeat_interleave(32)
torch.testing.assert_close(out, expected)
assert "tvm" not in sys.modules
assert "tvm_tirx" not in sys.modules
""",
                str(library),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
