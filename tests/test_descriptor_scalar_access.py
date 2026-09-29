# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Descriptor wrappers convert to scalar IR expressions at FFI boundaries."""

import pytest
from tvm_ffi import structural_walk

import tirx_kernels.tirx_lite as txl
from tvm import ir
from tvm.backend.cuda.cpp.descriptors import encode_instr_descriptor_dense_uint32


@pytest.mark.parametrize("constructor", ["raw", "mma_desc", "encode"])
@pytest.mark.parametrize("operand", ["a", "b"])
@pytest.mark.parametrize("arguments", ["both", "true", "false", "explicit"])
def test_descriptor_selection_is_an_mma_operand(constructor, operand, arguments):
    # This is a tracing contract test; the caller owns TMEM allocation and
    # the MMA completion protocol in an executable kernel.
    descriptors = []
    descriptor_values = []
    idesc = encode_instr_descriptor_dense_uint32(
        M=64,
        N=64,
        K=16,
        d_dtype="float32",
        a_dtype="bfloat16",
        b_dtype="bfloat16",
        trans_a=False,
        trans_b=False,
        cta_group=1,
    )

    @txl.kernel(warps=1, arch="sm_100a", grid=False)
    def probe(choose_first: txl.i32, tmem: txl.u32):
        smem = txl.smem_pool()
        for _ in range(2):
            tile = smem.alloc((64, 64), txl.bf16, align=1024, swizzle=txl.SW128B)
            if constructor == "raw":
                descriptor = txl.SmemDescriptor()
                # This tile is one 128B swizzle atom wide: LBO is zero and
                # the stride of eight rows is 64 units of 16 bytes.
                descriptor.init(tile.ptr_to(0, 0), 0, 64, txl.SW128B.value)
                value = descriptor.desc
            elif constructor == "mma_desc":
                descriptor = tile.mma_desc(major="k", mma_k=16)
                value = descriptor.value
            else:
                descriptor, _offset = tile.encode(major="k", mma_k=16)
                value = descriptor.value
            descriptors.append(descriptor)
            descriptor_values.append(value)
        a = descriptors[0] if arguments in ("both", "true") else descriptor_values[0]
        b = descriptors[1] if arguments in ("both", "false") else descriptor_values[1]
        selected = txl.Select(choose_first != 0, a, b)
        a_value, b_value = (
            (selected, descriptor_values[1]) if operand == "a" else (descriptor_values[0], selected)
        )
        with txl.If(txl.lane_id() == 0), txl.Then():
            txl.ptx["tcgen05.mma.cta_group::1.kind::f16"](
                tmem, a_value, b_value, txl.uint32(idesc), 0, 0, 0, 0, txl.ptx.pred(0)
            )

    calls = []
    structural_walk(probe.func.body, (ir.Call, calls.append))
    mma_calls = [call for call in calls if "tcgen05_mma" in getattr(call.op, "name", "")]
    assert len(mma_calls) == 1, [getattr(call.op, "name", "") for call in calls]
    selections = []
    structural_walk(mma_calls[0], (ir.prim.Select, selections.append))
    assert len(selections) == 1
    selection = selections[0]
    assert str(selection.ty.dtype) == "uint64"
    ir.assert_structural_equal(mma_calls[0].args[1 if operand == "a" else 2], selection)
    ir.assert_structural_equal(selection.true_value, descriptor_values[0])
    ir.assert_structural_equal(selection.false_value, descriptor_values[1])
    assert not descriptor_values[0].source.same_as(descriptor_values[1].source)
