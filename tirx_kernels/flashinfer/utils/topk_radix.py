# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2024 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Instruction-level helpers for the FlashInfer radix top-k ports.

Covers the monotone float->unsigned key mapping of ``RadixTopKTraits``
(``include/flashinfer/topk_common.cuh``) and the shared-memory, warp-shuffle and
barrier primitives ``RadixTopKKernel_Unified`` and its device helpers use
(``include/flashinfer/topk.cuh``).
"""

from tvm.script import tirx as T

WARP_SIZE = 32
FULL_MASK = 0xFFFFFFFF


# --- barriers ---------------------------------------------------------------
def bar_sync():
    """``bar.sync 0`` -- the plain CTA barrier ``__syncthreads()`` lowers to."""
    T.evaluate(T.ptx.bar.sync(T.uint32(0)))


# --- shared memory ----------------------------------------------------------
def ld_shared_u32(buffer, index):
    out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.ld.shared.b32(out[0], buffer.ptr_to([index])))
    return out[0]


def st_shared_u32(buffer, index, value):
    T.evaluate(T.ptx.st.shared.b32(buffer.ptr_to([index]), value))


def ld_shared_u16(buffer, index):
    out = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.ld.shared.b16(out[0], buffer.ptr_to([index])))
    return out[0]


def st_shared_u16(buffer, index, value):
    T.evaluate(T.ptx.st.shared.b16(buffer.ptr_to([index]), value))


def ld_shared_u64(buffer, index):
    """``ld.shared.b64`` -- reads two adjacent u32 scalars as one 64-bit access."""
    out = T.alloc_local((1,), "uint64")
    T.evaluate(T.ptx["ld.shared.b64"](out[0], buffer.ptr_to([index])))
    return out[0]


def u64_lo(value):
    return T.cast(T.bitwise_and(value, T.uint64(0xFFFFFFFF)), "uint32")


def u64_hi(value):
    return T.cast(T.shift_right(value, T.uint64(32)), "uint32")


def ld_shared_pair_u32(buffer, index):
    """``ld.shared.v2.b32``; returns the 2-element register pair."""
    out = T.alloc_local((2,), "uint32", align=8)
    T.evaluate(T.ptx["ld.shared.v2.b32"](out[0], out[1], buffer.ptr_to([index])))
    return out


def st_shared_pair_u32(buffer, index, v0, v1):
    """``st.shared.v2.b32``."""
    T.evaluate(T.ptx["st.shared.v2.b32"](buffer.ptr_to([index]), v0, v1))


def ld_shared_quad_u32(buffer, index):
    """``ld.shared.v4.b32``; returns the 4-element register quad."""
    out = T.alloc_local((4,), "uint32", align=16)
    T.evaluate(T.ptx["ld.shared.v4.b32"](out[0], out[1], out[2], out[3], buffer.ptr_to([index])))
    return out


def st_shared_quad_u32(buffer, index, v0, v1, v2, v3):
    """``st.shared.v4.b32``."""
    T.evaluate(T.ptx["st.shared.v4.b32"](buffer.ptr_to([index]), v0, v1, v2, v3))


def atom_shared_add_u32(buffer, index, value):
    """``atom.shared.add.u32``; returns the value held before the addition."""
    out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.atom.shared.add.u32(out[0], buffer.ptr_to([index]), value))
    return out[0]


# --- warp shuffles ----------------------------------------------------------
def shfl_down_u32(value, delta):
    """``shfl.sync.down.b32 d, a, delta, 31, -1`` (the ``__shfl_down_sync`` form)."""
    out = T.alloc_local((1,), "uint32")
    T.evaluate(
        T.ptx.shfl_sync.down.b32(out[0], value, T.uint32(delta), T.uint32(31), T.uint32(FULL_MASK))
    )
    return out[0]


def shfl_up_u32(value, delta):
    """``shfl.sync.up.b32 d, a, delta, 0, -1`` (cub's warp-scan form)."""
    out = T.alloc_local((1,), "uint32")
    T.evaluate(
        T.ptx.shfl_sync.up.b32(out[0], value, T.uint32(delta), T.uint32(0), T.uint32(FULL_MASK))
    )
    return out[0]


# --- monotone key mapping ---------------------------------------------------
# RadixTopKTraits<float>::ToOrdered  (topk_common.cuh:35-39)
#   (bits & 0x80000000) ? ~bits : (bits ^ 0x80000000)
# nvcc lowers this to setp.gt.s32 + selp.b32(0x80000000, -1) + xor.b32.
def to_ordered_u32(bits):
    signed = T.reinterpret("int32", bits)
    # T.Select, not T.if_then_else: the latter lowers to an if/else statement in
    # generated CUDA, which becomes a real branch with BSSY reconvergence. The
    # source's ternary is a predicated `selp.b32`, which is what the sketch names.
    mask = T.Select(signed > T.int32(-1), T.uint32(0x80000000), T.uint32(0xFFFFFFFF))
    return T.bitwise_xor(mask, bits)


# RadixTopKTraits<float>::FromOrdered (topk_common.cuh:41-44)
#   (ordered & 0x80000000) ? (ordered ^ 0x80000000) : ~ordered
def from_ordered_u32(ordered):
    signed = T.reinterpret("int32", ordered)
    mask = T.Select(signed > T.int32(-1), T.uint32(0xFFFFFFFF), T.uint32(0x80000000))
    return T.bitwise_xor(mask, ordered)


# RadixTopKTraits<half|nv_bfloat16>::ToOrdered (topk_common.cuh:61-64, :87-90)
def to_ordered_u16(bits):
    signed = T.reinterpret("int16", bits)
    mask = T.Select(signed > T.int16(-1), T.uint16(0x8000), T.uint16(0xFFFF))
    return T.bitwise_xor(mask, bits)


# RadixTopKTraits<half|nv_bfloat16>::FromOrdered (topk_common.cuh:66-69, :92-95)
def from_ordered_u16(ordered):
    signed = T.reinterpret("int16", ordered)
    mask = T.Select(signed > T.int16(-1), T.uint16(0xFFFF), T.uint16(0x8000))
    return T.bitwise_xor(mask, ordered)


# --- global memory ----------------------------------------------------------
def st_global_u32(buffer, index, value):
    T.evaluate(T.ptx.st.global_.b32(buffer.ptr_to([index]), value))


def st_global_u16(buffer, index, value):
    T.evaluate(T.ptx.st.global_.b16(buffer.ptr_to([index]), value))


def ld_global_u32(buffer, index):
    out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.ld.global_.b32(out[0], buffer.ptr_to([index])))
    return out[0]


def ld_global_u16(buffer, index):
    out = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.ld.global_.b16(out[0], buffer.ptr_to([index])))
    return out[0]


def warp_inclusive_sum_u32(value, lane):
    """cub ``WarpScanShfl`` inclusive sum: five ``shfl.sync.up.b32`` steps.

    The source uses the shuffle's own out-predicate to guard the add; the
    equivalent lane compare is used here because the TIRx wrapper does not
    expose the ``d|p`` destination form.
    """
    acc = value
    for step in range(5):
        peer = shfl_up_u32(acc, 1 << step)
        acc = T.Select(lane >= T.int32(1 << step), acc + peer, acc)
    return acc
