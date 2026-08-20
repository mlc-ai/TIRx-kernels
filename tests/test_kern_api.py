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
    def bind_form(out):
        handle = K.Bind(K.tvm_stack_alloca("tensormap", 1))
        K.evaluate(handle)
        K.ptx.st.global_.f32(out.ptr_to([0]), K.float32(0))

    def sanctioned_form(out):
        handle = K.stack_alloca("tensormap", 1)
        K.evaluate(handle)
        K.ptx.st.global_.f32(out.ptr_to([0]), K.float32(0))

    assert _tir(bind_form) == _tir(sanctioned_form)
