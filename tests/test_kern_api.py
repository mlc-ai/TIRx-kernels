# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Contract tests for the kern kernel-facing API surface."""

# NOTE: no `from __future__ import annotations` — kern kernels trace at
# decoration time and need live annotation objects (PEP 563 breaks them).

import tirx_kernels.kern as K


def _tir(build_body):
    @K.kernel(warps=1, arch="sm_100a", grid=False, thread_layout=False)
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


def test_stack_alloca_matches_bound_tvm_stack_alloca():
    from tvm.tirx.script.builder import ir as _I

    def bind_form(out):
        handle = _I.Bind(K.tvm_stack_alloca("tensormap", 1))
        K.keep_alive(handle)
        K.ptx.st.global_.f32(out.ptr_to([0]), K.float32(0))

    def sanctioned_form(out):
        handle = K.stack_alloca("tensormap", 1)
        K.keep_alive(handle)
        K.ptx.st.global_.f32(out.ptr_to([0]), K.float32(0))

    assert _tir(bind_form) == _tir(sanctioned_form)


def test_call_packed_has_statement_semantics():
    from tvm.script import tirx as _T
    from tvm.tirx.script.builder import ir as _I

    def sanctioned_form(out):
        K.call_packed("runtime.probe", K.int32(1))
        K.ptx.st.global_.f32(out.ptr_to([0]), K.float32(0))

    def reference_form(out):
        _I.evaluate(_T.call_packed("runtime.probe", K.int32(1)))
        K.ptx.st.global_.f32(out.ptr_to([0]), K.float32(0))

    assert _tir(sanctioned_form) == _tir(reference_form)


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
        @K.kernel(warps=1, arch="sm_100a", grid=False, thread_layout=False, **kw)
        def probe(out: K.gptr("float32")):
            # a direct shared-memory buffer store is a contract violation
            smem = K.alloc_buffer([4], "float32", scope="shared")
            K.buffer_store(smem, K.float32(1.0), [0])
            K.ptx.st.global_.f32(out.ptr_to([0]), K.float32(0))

        return probe

    with pytest.raises(LowLevelIRContractError):
        build()
    build(check_ir=False)  # explicit opt-out still traces


def test_retired_cuda_value_members_are_rejected_with_guidance():
    import pytest

    for name in ("_shfl_xor_sync", "ldg", "any_sync", "atomic_add"):
        with pytest.raises(AttributeError, match="spelled DPS"):
            getattr(K.cuda, name)
    K.cuda.elect_sync  # exempt: pred= idiom
    K.cuda.make_float2  # exempt: pure computation
