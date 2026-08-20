# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors
"""kern — an opinionated PTX-level kernel DSL (``import tirx_kernels.kern as K``).

The base language is tirx script plus the in-tree ``T.ptx`` instruction table,
used as-is: one call is one instruction, spelled the way the ISA spells it.
The protocol layer re-exports the in-tree ``lang`` classes (``Pipeline``,
``PipelineState``, ``MBarrier``, and the one-way barriers ``TMABar`` /
``TCGen05Bar``) so a kernel's source is ``K``-only. On top of that ``K`` adds
three primitive families — the things neither PTX nor ``lang`` has a word for:

specialize
    ``K.kernel`` / ``K.gptr`` / ``K.TensorMap`` entry; ``K.specialize`` +
    ``role(name, warps=[ids], regs=N)``; entry coordinates ``K.cta_id`` /
    ``K.warp_id`` / ``K.lane_id`` / ``K.thread_id`` and role-relative
    ``K.tid_in_role`` / ``K.warp_id_in_role``.

address
    ``K.smem_pool().alloc(..., swizzle=...)``, the stage subview
    ``tile[stage]``, and the four operand constructors it carries —
    ``ptr_to`` / ``m8n8`` / ``m8n8x4`` / ``mma_desc`` (+ ``.k_step``).
    Only ``mma_desc`` emits anything.

ring
    ``K.RingState`` owns an unsigned, optionally topology-strided
    ``(stage, phase)`` cursor when generic signed/unit-stride
    ``PipelineState`` would change lowering or semantics.

Everything else resolves through to ``tvm.script.tirx``.

``K.idioms`` is a *library*, not a fourth primitive family: plain functions over
bare ``K.ptx`` / ``K.cuda`` calls, each documenting the exact instruction
sequence it expands to. It is deliberately left as its own namespace rather
than flattened into ``K``, so a call site says whether it is naming an
instruction or a shape.
"""

from __future__ import annotations

import tvm
from tvm.backend.cuda.lang import MBarrier, Pipeline, PipelineState, TCGen05Bar, TMABar
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode
from tvm.script import tirx as _T
from tvm.tirx.script.builder import ir as _I

from . import idioms
from .entry import Kernel, TensorMap, cta_id, gptr, kernel, lane_id, thread_id, warp_id
from .ring import RingState
from .smem import (
    KDesc,
    KStep,
    KTile,
    KTileView,
    SmemDescriptor,
    smem_desc_add_16B_offset,
    smem_pool,
)
from .specialize import specialize, tid_in_role, warp_id_in_role

# ---------------------------------------------------------------------------
# dtype and swizzle tokens
# ---------------------------------------------------------------------------

f16 = "float16"
bf16 = "bfloat16"
f32 = "float32"
f8e4m3 = "float8_e4m3fn"
i8 = "int8"
i32 = "int32"
i64 = "int64"
u8 = "uint8"
u16 = "uint16"
u32 = "uint32"
u64 = "uint64"

SWNONE = SwizzleMode.SWIZZLE_NONE
SW32B = SwizzleMode.SWIZZLE_32B_ATOM
SW64B = SwizzleMode.SWIZZLE_64B_ATOM
SW128B = SwizzleMode.SWIZZLE_128B_ATOM


# ---------------------------------------------------------------------------
# Statement semantics for instruction calls
# ---------------------------------------------------------------------------
#
# A @K.kernel body is *traced* with an IRBuilder, not parsed. The TVMScript
# parser is what turns a bare expression statement into T.evaluate(...); under
# tracing a bare `K.ptx.barrier.sync.aligned(0, 128)` would build the Call and
# drop it -- silently emitting nothing. These proxies restore statement
# semantics using the engine's own contract ("a call is always a statement",
# destinations are leading operands): a call whose type is void is emitted and
# returns None, anything with a real dtype (T.cuda.elect_sync, get_tmem_addr,
# cvta_generic_to_shared, ...) is passed through untouched.


def _is_void_call(value):
    return (
        isinstance(value, tvm.ir.Expr)
        and isinstance(value.ty, tvm.ir.PrimType)
        and str(value.ty.dtype) == ""
    )


def evaluate(value):
    """Emit a value expression; accept void K calls that already emitted themselves."""
    if value is None:
        return None
    return _I.evaluate(value)


# ---------------------------------------------------------------------------
# The one instruction K makes stricter than the table
# ---------------------------------------------------------------------------
#
# `cp.async.{ca,cg}.shared.global` has three spellings that look alike at the
# call site and differ in what a *skipped* lane leaves in its destination. The
# table accepts all three, as it should -- it is the ISA's shape, and the ISA
# makes the trailing operand optional. K does not: through `K.ptx` the trailing
# operand is REQUIRED, so every call states which of the three it means.
#
#   (dst, src, n, n)              copy n bytes.                  [src-size]
#   (dst, src, n, 0)              copy nothing, ZERO-FILL dst.   [src-size]
#   (dst, src, n, K.ptx.pred(x))  x true -> zero-fill dst.       [ignore-src]
#   (dst, src, n, n, pred=x)      x false -> instruction does NOT run and
#                                 dst keeps its OLD contents.    [@p]
#
# The trap this closes: `pred=` is whole-instruction predication, not
# `ignore-src`. A false lane executes nothing and its destination keeps
# whatever was there -- it is *not* zeroed. The frozen GDN kernel relies on
# exactly that and stores 0.0 to the destination *before* the predicated copy
# (orig:L1252-1262); that pre-store is load-bearing, and dropping it as
# redundant would silently feed stale shared memory into the beta column.
# Omitting the operand is the one spelling that never says which behaviour was
# meant, so it is the one K rejects.

_CP_ASYNC_HELP = """\
`cp.async.{ca,cg}.shared.global` needs its trailing operand through K.ptx, \
because the three spellings differ in what a skipped lane's destination holds \
and the bare form does not say which you mean:

    K.ptx[insn](dst, src, n, n)               -- copy n bytes (the plain copy)
    K.ptx[insn](dst, src, n, 0)               -- copy nothing, ZERO-FILL dst
    K.ptx[insn](dst, src, n, K.ptx.pred(x))   -- ignore-src: x true -> zero-fill
    K.ptx[insn](dst, src, n, n, pred=x)       -- @p: x false -> the instruction
                                                 does not run at all and dst
                                                 KEEPS ITS OLD CONTENTS

The src-size operand must be a trace-time constant; use `K.ptx.pred(...)` for a \
runtime condition. Note that `pred=` alone is NOT ignore-src: if you want \
skipped lanes to read as zero you must either use ignore-src, or pre-store \
zeros yourself before the predicated copy."""


def _cp_async_needs_src_size(obj, args):
    """Is this a g2s ``cp.async`` invoked without its trailing operand?

    Identified off the instruction table rather than by string-matching a
    mnemonic: the guarded family is exactly the one whose candidate set offers
    ``_src_size`` / ``_ignore_src`` siblings beside a bare entry. Anything else
    -- ``cp.async.bulk.*``, ``cp.async.mbarrier.arrive``, every other
    instruction -- has no such sibling and passes straight through.
    """
    cands = getattr(obj, "_cands", None)
    if not cands:
        return False
    names = {getattr(entry, "name", "") for entry, _ in cands}
    if not any(n.endswith(("_src_size", "_ignore_src")) for n in names):
        return False
    # dst, src, cp_size and nothing else: the trailing operand was omitted.
    return len(args) == 3


class _StmtProxy:
    """Attribute/index-chain proxy that emits void calls as statements.

    Also the one place ``K`` is stricter than the instruction table: see
    :func:`_cp_async_needs_src_size`. The table itself is untouched.
    """

    __slots__ = ("_obj",)

    def __init__(self, obj):
        object.__setattr__(self, "_obj", obj)

    def __getattr__(self, name):
        return _StmtProxy(getattr(object.__getattribute__(self, "_obj"), name))

    def __getitem__(self, key):
        return _StmtProxy(object.__getattribute__(self, "_obj")[key])

    def __call__(self, *args, **kwargs):
        obj = object.__getattribute__(self, "_obj")
        if _cp_async_needs_src_size(obj, args):
            raise TypeError(f"{obj!r}: missing the trailing src-size operand.\n\n{_CP_ASYNC_HELP}")
        result = obj(*args, **kwargs)
        if _is_void_call(result):
            _T.evaluate(result)
            return None
        return result

    def __repr__(self):
        return f"<K stmt-proxy {object.__getattribute__(self, '_obj')!r}>"


class _PTXProxy(_StmtProxy):
    def addr(self, ptr, byte_offset):
        """Preserve the pointer's element type while applying a byte offset."""
        return _T.ptr_byte_offset(ptr, byte_offset, str(ptr.ty.element_type.dtype))


ptx = _PTXProxy(_T.ptx)
cuda = _StmtProxy(_T.cuda)


# ---------------------------------------------------------------------------
# tensors: ordinary default layout only
# ---------------------------------------------------------------------------


def alloc_buffer(
    shape,
    dtype="float32",
    data=None,
    strides=None,
    elem_offset=None,
    byte_offset=None,
    scope="global",
    align=-1,
    offset_factor=0,
    allocated_addr=None,
    annotations=None,
):
    """Allocate an ordinary tensor using TIRx's default layout."""
    return _I.alloc_buffer(
        shape,
        dtype,
        data=data,
        strides=strides,
        elem_offset=elem_offset,
        byte_offset=byte_offset,
        scope=scope,
        align=align,
        offset_factor=offset_factor,
        allocated_addr=allocated_addr,
        annotations=annotations,
    )


def decl_buffer(
    shape,
    dtype="float32",
    data=None,
    strides=None,
    elem_offset=None,
    byte_offset=None,
    scope="global",
    align=0,
    offset_factor=0,
    allocated_addr=None,
):
    """Declare an ordinary tensor using TIRx's default layout."""
    return _I.decl_buffer(
        shape,
        dtype,
        data=data,
        strides=strides,
        elem_offset=elem_offset,
        byte_offset=byte_offset,
        scope=scope,
        align=align,
        offset_factor=offset_factor,
        allocated_addr=allocated_addr,
    )


def alloc_local(shape, dtype="float32", *, align=-1, annotations=None):
    """Allocate a default-layout register tensor."""
    return alloc_buffer(shape, dtype, scope="local", align=align, annotations=annotations)


def local_scalar(dtype="float32", init=None):
    """Allocate one default-layout local scalar and return its writable element.

    ``init`` performs one immediate ``K.assign`` into the fresh element; the
    emitted TIR is identical to the two-statement declare-then-assign form.
    """
    element = alloc_local([1], dtype)[0]
    if init is not None:
        assign(element, init)
    return element


def stack_alloca(kind, size=1):
    """Allocate host-stack storage and return its handle, bound exactly once.

    The sanctioned spelling for ``tvm_stack_alloca``: the handle names an
    allocation, so it must be materialized as a single binding rather than
    re-evaluated per use.
    """
    return _I.Bind(_T.tvm_stack_alloca(kind, size))


def assign(dst, value):
    """Store into one writable scalar element of a local register tensor."""
    if not isinstance(dst, tvm.tirx.BufferLoad):
        raise TypeError(f"K.assign destination must be a writable scalar, got {type(dst).__name__}")
    if dst.buffer.scope() != "local":
        raise TypeError("K.assign destination must be a local scalar element")
    return _I.buffer_store(dst.buffer, value, list(dst.indices))


# ---------------------------------------------------------------------------
# warp-uniform values
# ---------------------------------------------------------------------------


def uniform(x, *, width=32, lane=0):
    """Broadcast lane *lane* of *x* to the whole warp.

    A **semantic no-op** on a value that is already warp-uniform, and a real
    broadcast otherwise. Expands to one statement — the collective is emitted
    here, not where the result is read::

        t = K.alloc_local([1], <dtype of x>)
        t[0] = K.cuda._shfl_sync(K.uint32(0xFFFFFFFF), x, lane, width)
        return t[0]

    ``threadIdx.x >> 5`` is already the same in every lane of a warp, but ptxas
    cannot prove it, so the value — and everything predicated on it — stays in
    the vector datapath. Telling the compiler costs one ``__shfl_sync`` and was
    worth **3.7%** end-to-end on the GDN prefill port: 7 configs, per-kernel GPU
    time, the plain shift at 1.014x the frozen hand-written kernel and the
    broadcast form at 0.977x, faster on every config and sign-consistent.
    Output is bit-identical either way. The frozen kernel does exactly this
    (``_make_warp_uniform``).

    ``K.specialize``'s role dispatch already gets this for free — the entry's
    ``cta->warp`` scope id *is* the broadcast. This is for the other values a
    kernel wants uniform.

    Materialising into a local is not optional: ``__shfl_sync`` is a warp
    collective, and a traced body emits a value-returning intrinsic where it is
    *used*, which inside a guard means the excluded lanes never arrive and the
    CTA hangs.
    """
    out = _T.alloc_local([1], str(x.ty.dtype))
    _I.buffer_store(out, _T.cuda._shfl_sync(_T.uint32(0xFFFFFFFF), x, lane, width), [0])
    return out[0]


# ---------------------------------------------------------------------------
# everything else is tirx
# ---------------------------------------------------------------------------


_FORBIDDEN_TIRX_NAMES = {
    "ComposeLayout",
    "Layout",
    "S",
    "SMEMPool",
    "TCol",
    "TLane",
    "TMEMPool",
    "TileLayout",
    "alloc_cast_frag",
    "alloc_shared",
    "alloc_tcgen05_ldst_frag",
    "inline",
    "meta_class",
    "meta_var",
    "smem",
    "tile",
    "tmem",
    "wg_reg_tile",
}

# Binding forms retired by measurement: every kernel migrated to the two direct
# spellings, and the mis-read-prone middle forms are rejected at the API.
_FORBIDDEN_BINDING_NAMES = {"Bind", "Let", "let"}

_BINDING_HELP = (
    "Kern rejects {name!r}: bindings use exactly two spellings.\n"
    "  - cheap pure arithmetic       -> a plain Python variable (re-emitted per use; ptxas CSEs it)\n"
    "  - anything that must run once -> x = K.local_scalar(dtype, init=expr)\n"
    "    (loads, div/rcp/rsqrt, warp collectives, cvta-of-allocation, volatile-ordered reads,\n"
    "     and SNAPSHOTS of mutable state -- e.g. a RingState/PipelineState cursor read before .advance())\n"
    "  - stack allocations           -> K.stack_alloca(kind, size)\n"
    "Annotated `x: K.<dtype> = expr` is inert under tracing (it is a plain variable, NOT a binding)."
)


def __getattr__(name):
    if name in _FORBIDDEN_BINDING_NAMES:
        raise AttributeError(_BINDING_HELP.format(name=name))
    if name in _FORBIDDEN_TIRX_NAMES:
        raise AttributeError(
            f"Kern deliberately does not expose {name!r}: use default-layout tensors, "
            "raw PTX, and K.smem_pool().alloc(..., swizzle=...)"
        )
    try:
        return getattr(_T, name)
    except AttributeError as err:
        raise AttributeError(f"module 'tirx_kernels.kern' has no attribute {name!r}") from err


__all__ = [
    "SW32B",
    "SW64B",
    "SW128B",
    "SWNONE",
    "KDesc",
    "KStep",
    "KTile",
    "KTileView",
    "Kernel",
    "MBarrier",
    "Pipeline",
    "PipelineState",
    "RingState",
    "SmemDescriptor",
    "SwizzleMode",
    "TCGen05Bar",
    "TMABar",
    "TensorMap",
    "alloc_buffer",
    "alloc_local",
    "bf16",
    "cta_id",
    "cuda",
    "decl_buffer",
    "f16",
    "f32",
    "gptr",
    "i32",
    "i64",
    "idioms",
    "kernel",
    "lane_id",
    "ptx",
    "smem_pool",
    "smem_desc_add_16B_offset",
    "specialize",
    "stack_alloca",
    "thread_id",
    "tid_in_role",
    "u8",
    "u32",
    "uniform",
    "warp_id",
    "warp_id_in_role",
]
