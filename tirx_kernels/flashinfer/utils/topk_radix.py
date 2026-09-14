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

import tirx_kernels.tirx_lite as txl

WARP_SIZE = 32
FULL_MASK = 0xFFFFFFFF


# --- barriers ---------------------------------------------------------------
def bar_sync():
    """``bar.sync 0`` -- the plain CTA barrier ``__syncthreads()`` lowers to."""
    txl.ptx.bar.sync(txl.uint32(0))


# --- shared memory ----------------------------------------------------------
def ld_shared_u32(buffer, index):
    out = txl.local_scalar("uint32")
    txl.ptx.ld.shared.b32(out, buffer.ptr_to([index]))
    return out


def st_shared_u32(buffer, index, value):
    txl.ptx.st.shared.b32(buffer.ptr_to([index]), value)


def ld_shared_u16(buffer, index):
    out = txl.local_scalar("uint16")
    txl.ptx.ld.shared.b16(out, buffer.ptr_to([index]))
    return out


def st_shared_u16(buffer, index, value):
    txl.ptx.st.shared.b16(buffer.ptr_to([index]), value)


def ld_global_pair_u16(buffer, index):
    """``ld.global.v2.b16``; returns the two 16-bit lanes as separate registers.

    The source's ``vec_t<DType, 2>::cast_load`` lands both halves in 16-bit
    registers directly, so the monotone key flip runs on 16-bit operands with no
    extract or repack arithmetic.  Loading one 32-bit word instead costs an
    ``and``/``shr`` pair to split it and a ``shl``/``or`` pair to reassemble it,
    on every element of the chunk.
    """
    out = txl.alloc_local((2,), "uint16")
    txl.ptx["ld.global.v2.b16"](out[0], out[1], buffer.ptr_to([index]))
    return out[0], out[1]


def st_shared_pair_u16(buffer, index, v0, v1):
    """``st.shared.v2.b16``, the store width the source's element pair coalesces to."""
    txl.ptx["st.shared.v2.b16"](buffer.ptr_to([index]), v0, v1)


def ld_shared_u64(buffer, index):
    """``ld.shared.b64`` -- reads two adjacent u32 scalars as one 64-bit access."""
    out = txl.local_scalar("uint64")
    txl.ptx["ld.shared.b64"](out, buffer.ptr_to([index]))
    return out


def u64_lo(value):
    return txl.cast(txl.bitwise_and(value, txl.uint64(0xFFFFFFFF)), "uint32")


def u64_hi(value):
    return txl.cast(txl.shift_right(value, txl.uint64(32)), "uint32")


def ld_shared_pair_u32(buffer, index):
    """``ld.shared.v2.b32``; returns the 2-element register pair."""
    out = txl.alloc_local((2,), "uint32", align=8)
    txl.ptx["ld.shared.v2.b32"](out[0], out[1], buffer.ptr_to([index]))
    return out


def st_shared_pair_u32(buffer, index, v0, v1):
    """``st.shared.v2.b32``."""
    txl.ptx["st.shared.v2.b32"](buffer.ptr_to([index]), v0, v1)


def ld_shared_quad_u32(buffer, index):
    """``ld.shared.v4.b32``; returns the 4-element register quad."""
    out = txl.alloc_local((4,), "uint32", align=16)
    txl.ptx["ld.shared.v4.b32"](out[0], out[1], out[2], out[3], buffer.ptr_to([index]))
    return out


def st_shared_quad_u32(buffer, index, v0, v1, v2, v3):
    """``st.shared.v4.b32``."""
    txl.ptx["st.shared.v4.b32"](buffer.ptr_to([index]), v0, v1, v2, v3)


def atom_shared_add_u32(buffer, index, value):
    """``atom.shared.add.u32``; returns the value held before the addition."""
    out = txl.local_scalar("uint32")
    txl.ptx.atom.shared.add.u32(out, buffer.ptr_to([index]), value)
    return out


# --- warp shuffles ----------------------------------------------------------
def shfl_down_u32(value, delta):
    """``shfl.sync.down.b32 d, a, delta, 31, -1`` (the ``__shfl_down_sync`` form)."""
    out = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.down.b32(out, value, txl.uint32(delta), txl.uint32(31), txl.uint32(FULL_MASK))
    return out


def shfl_up_u32(value, delta):
    """``shfl.sync.up.b32 d, a, delta, 0, -1`` (cub's warp-scan form)."""
    out = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.up.b32(out, value, txl.uint32(delta), txl.uint32(0), txl.uint32(FULL_MASK))
    return out


# --- monotone key mapping ---------------------------------------------------
# RadixTopKTraits<float>::ToOrdered  (topk_common.cuh:35-39)
#   (bits & 0x80000000) ? ~bits : (bits ^ 0x80000000)
# nvcc lowers this to setp.gt.s32 + selp.b32(0x80000000, -1) + xor.b32.
def to_ordered_u32(bits):
    signed = txl.reinterpret("int32", bits)
    # txl.Select, not txl.if_then_else: the latter lowers to an if/else statement in
    # generated CUDA, which becomes a real branch with BSSY reconvergence. The
    # source's ternary is a predicated `selp.b32`, which is what the sketch names.
    mask = txl.Select(signed > txl.int32(-1), txl.uint32(0x80000000), txl.uint32(0xFFFFFFFF))
    return txl.bitwise_xor(mask, bits)


# RadixTopKTraits<float>::FromOrdered (topk_common.cuh:41-44)
#   (ordered & 0x80000000) ? (ordered ^ 0x80000000) : ~ordered
def from_ordered_u32(ordered):
    signed = txl.reinterpret("int32", ordered)
    mask = txl.Select(signed > txl.int32(-1), txl.uint32(0xFFFFFFFF), txl.uint32(0x80000000))
    return txl.bitwise_xor(mask, ordered)


# RadixTopKTraits<half|nv_bfloat16>::ToOrdered (topk_common.cuh:61-64, :87-90)
def to_ordered_u16(bits):
    signed = txl.reinterpret("int16", bits)
    mask = txl.Select(signed > txl.int16(-1), txl.uint16(0x8000), txl.uint16(0xFFFF))
    return txl.bitwise_xor(mask, bits)


# RadixTopKTraits<half|nv_bfloat16>::FromOrdered (topk_common.cuh:66-69, :92-95)
def from_ordered_u16(ordered):
    signed = txl.reinterpret("int16", ordered)
    mask = txl.Select(signed > txl.int16(-1), txl.uint16(0xFFFF), txl.uint16(0x8000))
    return txl.bitwise_xor(mask, ordered)


# --- global memory ----------------------------------------------------------
def st_global_u32(buffer, index, value):
    txl.ptx.st.global_.b32(buffer.ptr_to([index]), value)


def st_global_u16(buffer, index, value):
    txl.ptx.st.global_.b16(buffer.ptr_to([index]), value)


def ld_global_u32(buffer, index):
    out = txl.local_scalar("uint32")
    txl.ptx.ld.global_.b32(out, buffer.ptr_to([index]))
    return out


def ld_global_u16(buffer, index):
    out = txl.local_scalar("uint16")
    txl.ptx.ld.global_.b16(out, buffer.ptr_to([index]))
    return out


def warp_inclusive_sum_u32(value, lane):
    """cub ``WarpScanShfl`` inclusive sum: five ``shfl.sync.up.b32`` steps.

    The source uses the shuffle's own out-predicate to guard the add; the
    equivalent lane compare is used here because the TIRx wrapper does not
    expose the ``d|p`` destination form.
    """
    acc = value
    for step in range(5):
        peer = shfl_up_u32(acc, 1 << step)
        acc = txl.Select(lane >= txl.int32(1 << step), acc + peer, acc)
    return acc


# --- inter-CTA synchronization -----------------------------------------------
# The multi-CTA radix kernel synchronizes the CTAs of one row group through a
# monotonically increasing arrival counter in global memory, with absolute phase
# targets. These mirror `ld_acquire` / `red_release` / `st_release` /
# `atom_add_release` (topk.cuh:63-121).
def ld_acquire_gpu_s32(buffer, index):
    """``ld.global.acquire.gpu.b32`` -- the acquire half of the group barrier."""
    out = txl.local_scalar("int32")
    txl.ptx.ld.acquire.gpu.global_.b32(out, buffer.ptr_to([index]))
    return out


def fence_acq_rel_gpu():
    """``fence.acq_rel.gpu``; the release half is always fence-then-atomic."""
    txl.ptx.fence.acq_rel.gpu()


def red_release_gpu_add_s32(buffer, index, value):
    """``fence.acq_rel.gpu`` + ``red.relaxed.gpu.global.add.s32`` (no result)."""
    fence_acq_rel_gpu()
    txl.ptx.red.relaxed.gpu.global_.add.s32(buffer.ptr_to([index]), value)


def atom_add_release_gpu_s32(buffer, index, value):
    """``fence.acq_rel.gpu`` + ``atom.relaxed.gpu.global.add.s32``; returns the old value."""
    fence_acq_rel_gpu()
    out = txl.local_scalar("int32")
    txl.ptx.atom.relaxed.gpu.global_.add.s32(out, buffer.ptr_to([index]), value)
    return out


def st_release_gpu_s32(buffer, index, value):
    """``fence.acq_rel.gpu`` + ``st.release.gpu.global.b32``."""
    fence_acq_rel_gpu()
    txl.ptx.st.release.gpu.global_.b32(buffer.ptr_to([index]), value)


def atom_global_add_u32(buffer, index, value):
    """Plain relaxed ``atom.global.add.u32``; returns the value held before the add.

    Used for the non-deterministic collect's output counter, where the source
    calls `atomicAdd` with no ordering qualifier.
    """
    out = txl.local_scalar("uint32")
    txl.ptx.atom.global_.add.u32(out, buffer.ptr_to([index]), value)
    return out


# --- staged-key load/convert/store and collect emit ---------------------------
def ld_global_bits(buf, elem_index, is32):
    """One scalar element's raw bits (``ld.global.b32`` | ``ld.global.b16``)."""
    if is32:
        out = txl.local_scalar("uint32")
        txl.ptx.ld.global_.b32(out, buf.ptr_to([elem_index]))
        return out
    out16 = txl.local_scalar("uint16")
    txl.ptx.ld.global_.b16(out16, buf.ptr_to([elem_index]))
    return out16


def ld_global_words(buf, elem_index, load_bytes):
    """One vector load of ``load_bytes`` bytes, returned as 32-bit words."""
    if load_bytes == 16:
        w = txl.alloc_local((4,), "uint32", align=16)
        txl.ptx["ld.global.v4.b32"](w[0], w[1], w[2], w[3], buf.ptr_to([elem_index]))
        return [w[0], w[1], w[2], w[3]]
    if load_bytes == 8:
        w = txl.alloc_local((2,), "uint32", align=8)
        txl.ptx["ld.global.v2.b32"](w[0], w[1], buf.ptr_to([elem_index]))
        return [w[0], w[1]]
    w = txl.local_scalar("uint32")
    txl.ptx.ld.global_.b32(w, buf.ptr_to([elem_index]))
    return [w]


def stage_vector(buf, s_ordered, row_in, i, vec, load_bytes, is32, to_ordered, st_key):
    """LoadToSharedOrdered's vector body (:612-617).

    One vector load, ``VEC_SIZE`` monotone key conversions, then a single shared
    store as wide as the load. The source stores element-by-element and lets nvcc
    coalesce the contiguous group (its PTX shows one ``st.shared.v4.b32``);
    opaque ``txl.ptx`` intrinsics cannot be merged after the fact, so the wide form
    is requested directly.
    """
    base = row_in + txl.cast(i, "int64")
    if load_bytes == 2:
        st_key(s_ordered, i, to_ordered(ld_global_bits(buf, base, False)))
        return
    if not is32 and load_bytes == 4:
        # VEC_SIZE == 2 on a 16-bit dtype: mirror the source's native 16-bit
        # vector pair (`ld.global.v2.b16` / `st.shared.v2.b16`) so the flip stays
        # in 16-bit registers.  Going through one 32-bit word here would add an
        # extract and a repack per element.
        lo_bits, hi_bits = ld_global_pair_u16(buf, base)
        st_shared_pair_u16(s_ordered, i, to_ordered(lo_bits), to_ordered(hi_bits))
        return
    words = ld_global_words(buf, base, load_bytes)
    if is32:
        keys = [to_ordered(w) for w in words]
    else:
        # Each loaded word holds two 16-bit lanes; convert both and repack so the
        # shared store keeps the same width as the global load.
        keys = []
        for word in words:
            lo = to_ordered(txl.cast(txl.bitwise_and(word, txl.uint32(0xFFFF)), "uint16"))
            hi = to_ordered(txl.cast(txl.shift_right(word, txl.uint32(16)), "uint16"))
            keys.append(
                txl.bitwise_or(txl.cast(lo, "uint32"), txl.shift_left(txl.cast(hi, "uint32"), txl.uint32(16)))
            )
    if len(keys) == 4:
        st_shared_quad_u32(s_ordered, i, keys[0], keys[1], keys[2], keys[3])
    elif len(keys) == 2:
        st_shared_pair_u32(s_ordered, i, keys[0], keys[1])
    else:
        st_shared_u32(s_ordered, i, keys[0])


def emit_selected(out_idx, out_val, row_out, i, key, pos, offset, basic, ragged, is32, dtype):
    """The mode epilogue the collect passes call per selected element (:1339-1375)."""
    slot = row_out + txl.cast(pos, "int64")
    if basic:
        st_global_u32(out_idx, slot, txl.reinterpret("uint32", i))
        if is32:
            st_global_u32(out_val, slot, from_ordered_u32(key))
        else:
            st_global_u16(out_val, slot, from_ordered_u16(txl.cast(key, "uint16")))
    elif ragged:
        st_global_u32(out_idx, slot, txl.reinterpret("uint32", i + offset))
    else:
        st_global_u32(out_idx, slot, txl.reinterpret("uint32", i))
