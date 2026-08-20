# This file is a TIRx port of code from MSA
# (https://github.com/MiniMax-AI/MSA @ 80434d7f), Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Scalar memory and synchronization primitives shared by the MSA ports.

The MSA prepare kernels are scalar integer code: they move one int32 at a time
between registers, global memory and shared memory, and their only cross-thread
traffic is a device-scope atomic, a warp broadcast and a CTA barrier. Each
helper below is one PTX instruction of the family the source export uses.

Both scopes go through ``T.ptx.*`` on ``ptr_to`` rather than a native
``BufferLoad``/``BufferStore``: the repository's low-level IR contract
(:mod:`tirx_kernels.low_level_ir`) rejects the native form for ``global`` and
``shared`` alike.

Upstream source: python/fmha_sm100/cute/src/common/copy_utils.py,
python/fmha_sm100/cute/src/sm100/prepare_scheduler.py.
"""

from tvm.script import tirx as T


# --- global memory ----------------------------------------------------------
def ld_global_i32(buffer, index):
    """``ld.global.b32``; the source's plain scalar load."""
    out = T.alloc_local((1,), "int32")
    T.evaluate(T.ptx.ld.global_.b32(out[0], buffer.ptr_to([index])))
    return out[0]


def st_global_i32(buffer, index, value):
    """``st.global.b32``; one scalar field."""
    T.evaluate(T.ptx.st.global_.b32(buffer.ptr_to([index]), value))


def atom_add_global_i32(buffer, index, value):
    """``atom.global.add.u32``; returns the value held before the addition.

    Relaxed ordering and device scope are PTX's defaults for ``atom.global``, so
    this short encoding -- no ``.relaxed``, no ``.gpu``, no surrounding fence --
    is what both MSA prepare kernels emit, whether they spell the qualifiers out
    (``cute.arch.atomic_add(..., sem="relaxed", scope="gpu")``) or write the
    instruction directly (``copy_utils.atomic_add_i32``).
    """
    out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.atom.global_.add.u32(out[0], buffer.ptr_to([index]), value))
    return T.reinterpret("int32", out[0])


# --- shared memory ----------------------------------------------------------
def ld_shared_i32(buffer, index):
    """``ld.shared.b32``."""
    out = T.alloc_local((1,), "int32")
    T.evaluate(T.ptx.ld.shared.b32(out[0], buffer.ptr_to([index])))
    return out[0]


def st_shared_i32(buffer, index, value):
    """``st.shared.b32``."""
    T.evaluate(T.ptx.st.shared.b32(buffer.ptr_to([index]), value))


# --- synchronization --------------------------------------------------------
def bar_sync():
    """``bar.sync 0`` -- what ``cute.arch.barrier()`` / ``__syncthreads()`` lowers to."""
    T.evaluate(T.ptx.bar.sync(T.uint32(0)))


def shfl_idx_i32(value, source_lane):
    """``shfl.sync.idx.b32 d, a, src, 31, -1``; a full-warp broadcast."""
    out = T.alloc_local((1,), "uint32")
    T.evaluate(
        T.ptx.shfl_sync.idx.b32(
            out[0],
            T.reinterpret("uint32", value),
            T.uint32(source_lane),
            T.uint32(31),
            T.uint32(0xFFFFFFFF),
        )
    )
    return T.reinterpret("int32", out[0])


# --- integer division -------------------------------------------------------
def udiv_i32(x, d):
    """``x / d`` for a non-negative ``x`` and positive ``d``, without the sign fixup.

    These quotients divide a count by a block size or a head count, so no operand
    can be negative -- but they arrive from signed int32 globals and kernel
    arguments, which gives the compiler no such proof, and a signed ``//`` then
    carries the full floordiv correction chain. Routing through unsigned keeps
    the same quotient and drops the correction; the divisors stay runtime values,
    so a real integer divide is still issued.
    """
    return T.cast(T.cast(x, "uint32") // T.cast(d, "uint32"), "int32")


def uceil_div_i32(x, d):
    """``ceil(x / d)`` under the same non-negativity argument as :func:`udiv_i32`."""
    numerator: T.uint32 = T.cast(x, "uint32") + T.cast(d, "uint32") - T.uint32(1)
    return T.cast(numerator // T.cast(d, "uint32"), "int32")


__all__ = [
    "atom_add_global_i32",
    "bar_sync",
    "ld_global_i32",
    "ld_shared_i32",
    "shfl_idx_i32",
    "st_global_i32",
    "st_shared_i32",
    "uceil_div_i32",
    "udiv_i32",
]
