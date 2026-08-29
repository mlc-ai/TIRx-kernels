# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Contract tests for the kern kernel-facing API surface."""

# NOTE: no `from __future__ import annotations` — kern kernels trace at
# decoration time and need live annotation objects (PEP 563 breaks them).

import tirx_kernels.kern as K
from tvm.tirx.stmt_functor import StmtExprVisitor


def _tir(build_body):
    @K.kernel(warps=1, arch="sm_100a", grid=False)
    def probe(out: K.gptr("float32")):
        build_body(out)

    return probe.func.script()


def test_local_scalar_init_matches_declare_then_assign():
    def two_statement(out):
        x = K.local_scalar("float32")
        K.assign(x, K.float32(3.0))
        K.ptx.st.global_.f32(out.ptr_to([0]), x)

    def init_form(out):
        x = K.local_scalar("float32", init=K.float32(3.0))
        K.ptx.st.global_.f32(out.ptr_to([0]), x)

    assert _tir(two_statement) == _tir(init_form)


def test_local_scalar_accepts_explicit_trace_name():
    seen = []

    def build(out):
        counter = K.local_scalar("int32", init=K.int32(3), name="counter")
        seen.append(counter.buffer.name)
        K.ptx.st.global_.b32(out.ptr_to([0]), counter)

    _tir(build)
    assert seen == ["counter"]


def test_stack_alloca_is_bound_exactly_once():
    class Scanner(StmtExprVisitor):
        def __init__(self):
            super().__init__()
            self.stack_binds = []

        def visit_bind_(self, op):
            if getattr(getattr(op.value, "op", None), "name", None) == "tirx.tvm_stack_alloca":
                self.stack_binds.append(op)
            super().visit_bind_(op)

    @K.kernel(warps=1, arch="sm_100a", grid=False)
    def probe(out: K.gptr("float32")):
        handle = K.stack_alloca("tensormap", 1)
        K.keep_alive(handle)
        K.ptx.st.global_.f32(out.ptr_to([0]), K.float32(0))

    scanner = Scanner()
    scanner(probe.func.body)
    assert len(scanner.stack_binds) == 1


def test_call_packed_has_statement_semantics():
    class Scanner(StmtExprVisitor):
        def __init__(self):
            super().__init__()
            self.packed_evaluates = []

        def visit_evaluate_(self, op):
            if getattr(getattr(op.value, "op", None), "name", None) == "tirx.tvm_call_packed":
                self.packed_evaluates.append(op)
            super().visit_evaluate_(op)

    @K.kernel(warps=1, arch="sm_100a", grid=False, check_ir=False)
    def probe(out: K.gptr("float32")):
        K.call_packed("runtime.probe", K.int32(1))
        K.ptx.st.global_.f32(out.ptr_to([0]), K.float32(0))

    scanner = Scanner()
    scanner(probe.func.body)
    assert len(scanner.packed_evaluates) == 1


def test_retired_binding_forms_are_rejected_with_guidance():
    import pytest

    for name in ("Bind", "let", "Let"):
        with pytest.raises(AttributeError, match="two spellings"):
            getattr(K, name)
    with pytest.raises(AttributeError, match="emits itself"):
        K.evaluate


def test_kernel_build_runs_low_level_ir_check_by_default():
    import pytest

    from tirx_kernels.low_level_ir import LowLevelIRContractError

    def build(**kw):
        @K.kernel(warps=1, arch="sm_100a", grid=False, **kw)
        def probe(out: K.gptr("float32")):
            # a direct shared-memory buffer store is a contract violation
            smem = K.alloc_buffer([4], "float32", scope="shared")
            K.buffer_store(smem, K.float32(1.0), [0])
            K.ptx.st.global_.f32(out.ptr_to([0]), K.float32(0))

        return probe

    with pytest.raises(LowLevelIRContractError):
        build()
    build(check_ir=False)  # explicit opt-out still traces


def test_unsupported_tmem_buffer_scope_is_rejected():
    import pytest

    with pytest.raises(ValueError, match='scope="tmem"'):
        K.alloc_buffer((1,), K.u32, scope="tmem")
    with pytest.raises(ValueError, match='scope="tmem"'):
        K.decl_buffer((1,), K.u32, scope="tmem")

    with pytest.raises(AttributeError, match="deliberately does not expose"):
        K.TMEMPool


def test_parser_and_raw_builder_entry_points_are_rejected():
    import pytest

    for name in ("parser", "ir"):
        with pytest.raises(AttributeError, match=r"native K\.kernel"):
            getattr(K, name)
    for name in ("jit", "prim_func", "match_buffer", "device_entry"):
        with pytest.raises(AttributeError, match="deliberately does not expose"):
            getattr(K, name)


def test_thread_layout_is_not_a_kernel_entry_option():
    import pytest

    with pytest.raises(TypeError, match="thread_layout"):

        @K.kernel(warps=1, arch="sm_100a", thread_layout=False)
        def probe(out: K.gptr(K.f32)):
            K.ptx.st.global_.f32(out.ptr_to([0]), K.float32(0))

    with pytest.raises(TypeError):
        K.thread_id([32])


def test_gptr_shape_reuses_entry_scalar_parameters():
    @K.kernel(warps=1, arch="sm_100a", grid=False)
    def probe(out: K.gptr(K.f32, shape=lambda p: (p["rows"], p["cols"])), rows: K.i32, cols: K.i32):
        K.ptx.st.global_.f32(out.ptr_to([0, 0]), K.float32(0))

    out = probe.func.params[0]
    assert out.shape[0].same_as(probe.func.params[1])
    assert out.shape[1].same_as(probe.func.params[2])


def test_gptr_shape_rejects_unknown_scalar_parameters():
    import pytest

    with pytest.raises(ValueError, match="unknown scalar parameter 'missing'"):

        @K.kernel(warps=1, arch="sm_100a", grid=False)
        def probe(out: K.gptr(K.f32, shape=lambda p: (p["missing"],))):
            K.ptx.st.global_.f32(out.ptr_to([0]), K.float32(0))


def test_retired_cuda_value_members_are_rejected_with_guidance():
    import pytest

    for name in ("_shfl_xor_sync", "ldg", "any_sync", "atomic_add"):
        with pytest.raises(AttributeError, match="spelled DPS"):
            getattr(K.cuda, name)
    K.cuda.elect_sync  # exempt: pred= idiom
    K.cuda.make_float2  # exempt: pure computation


def test_value_constructors_work_outside_kernel_trace():
    value = K.uint64(0)
    assert str(value.ty.dtype) == "uint64"


def test_kernel_records_python_source_spans():
    import inspect
    import linecache

    def emit(out):
        K.ptx.st.global_.f32(out.ptr_to([0]), K.float32(0))

    @K.kernel(warps=1, arch="sm_100a", grid=False)
    def probe(out: K.gptr(K.f32)):
        K.ptx.st.global_.f32(out.ptr_to([1]), K.float32(1))
        emit(out)

    assert probe.func.span is not None
    assert probe.func.span.source_name.name == inspect.getsourcefile(emit)
    assert "@K.kernel" in linecache.getline(
        probe.func.span.source_name.name, probe.func.span.line
    )

    statements = probe.func.body.body.seq
    stores = [stmt for stmt in statements if type(stmt).__name__ == "Evaluate"]
    assert len(stores) == 2
    source_lines = []
    for store in stores:
        assert store.span is not None
        assert store.span.source_name.name == inspect.getsourcefile(emit)
        assert store.value.span.same_as(store.span)
        source_lines.append(linecache.getline(store.span.source_name.name, store.span.line))
    assert "out.ptr_to([1])" in source_lines[0]
    assert "out.ptr_to([0])" in source_lines[1]


def test_kernel_source_span_tracer_is_restored_after_failure():
    import sys

    import pytest

    previous_trace = sys.gettrace()
    with pytest.raises(RuntimeError, match="trace failed"):

        @K.kernel(warps=1, arch="sm_100a", grid=False)
        def probe(out: K.gptr(K.f32)):
            raise RuntimeError("trace failed")

    assert sys.gettrace() is previous_trace
