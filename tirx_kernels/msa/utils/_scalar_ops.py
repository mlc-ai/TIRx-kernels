# This file is a TIRx port of code from MSA
# (https://github.com/MiniMax-AI/MSA @ 80434d7f), Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Scalar memory and synchronization primitives shared by the MSA ports.

The MSA prepare kernels are scalar integer code: they move one int32 at a time
between registers, global memory and shared memory, and their only cross-thread
traffic is a device-scope atomic, a warp broadcast and a CTA barrier. Each
helper below is one PTX instruction of the family the source export uses.

Both scopes go through ``txl.ptx.*`` on ``ptr_to`` rather than a native
``TensorLoad``/``BufferStore``: the repository's low-level IR contract
(:mod:`tirx_kernels.tirx_lite.low_level_ir`) rejects the native form for ``global`` and
``shared`` alike.

Upstream source: python/fmha_sm100/cute/src/common/copy_utils.py,
python/fmha_sm100/cute/src/sm100/prepare_scheduler.py.
"""

import tirx_kernels.tirx_lite as txl


# --- global memory ----------------------------------------------------------
def ld_global_i32(buffer, index):
    """``ld.global.b32``; the source's plain scalar load."""
    out = txl.local_scalar(txl.i32)
    txl.ptx.ld.global_.b32(out, buffer.ptr_to([index]))
    return out


def st_global_i32(buffer, index, value):
    """``st.global.b32``; one scalar field."""
    txl.ptx.st.global_.b32(buffer.ptr_to([index]), value)


def atom_add_global_i32(buffer, index, value):
    """``atom.global.add.u32``; returns the value held before the addition.

    Relaxed ordering and device scope are PTX's defaults for ``atom.global``, so
    this short encoding -- no ``.relaxed``, no ``.gpu``, no surrounding fence --
    is what both MSA prepare kernels emit, whether they spell the qualifiers out
    (``cute.arch.atomic_add(..., sem="relaxed", scope="gpu")``) or write the
    instruction directly (``copy_utils.atomic_add_i32``).
    """
    out = txl.local_scalar(txl.u32)
    txl.ptx.atom.global_.add.u32(out, buffer.ptr_to([index]), value)
    return txl.reinterpret(txl.i32, out)


# --- shared memory ----------------------------------------------------------
def ld_shared_i32(buffer, index):
    """``ld.shared.b32``."""
    out = txl.local_scalar(txl.i32)
    txl.ptx.ld.shared.b32(out, buffer.ptr_to([index]))
    return out


def st_shared_i32(buffer, index, value):
    """``st.shared.b32``."""
    txl.ptx.st.shared.b32(buffer.ptr_to([index]), value)


# --- synchronization --------------------------------------------------------
def bar_sync():
    """``bar.sync 0`` -- what ``cute.arch.barrier()`` / ``__syncthreads()`` lowers to."""
    txl.ptx.bar.sync(txl.uint32(0))


def shfl_idx_i32(value, source_lane):
    """``shfl.sync.idx.b32 d, a, src, 31, -1``; a full-warp broadcast."""
    out = txl.local_scalar(txl.u32)
    txl.ptx.shfl_sync.idx.b32(
        out,
        txl.reinterpret(txl.u32, value),
        txl.uint32(source_lane),
        txl.uint32(31),
        txl.uint32(0xFFFFFFFF),
    )
    return txl.reinterpret(txl.i32, out)


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
    return txl.cast(txl.cast(x, txl.u32) // txl.cast(d, txl.u32), txl.i32)


def uceil_div_i32(x, d):
    """``ceil(x / d)`` under the same non-negativity argument as :func:`udiv_i32`."""
    numerator = txl.cast(x, txl.u32) + txl.cast(d, txl.u32) - txl.uint32(1)
    return txl.cast(numerator // txl.cast(d, txl.u32), txl.i32)


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
