# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025
# DeepSeek, licensed under the MIT License. The upstream sources carry no
# per-file license header; see licenses/LICENSE.flashmla.txt for the full
# license text.
#
# Modifications Copyright (c) 2026 The TIRx Authors.
# Modifications are licensed under the Apache License, Version 2.0.
"""Builder-native forms of the SM100 resources used by FlashMLA kernels."""

from __future__ import annotations

import tvm
from tvm.ir import Expr, PointerType, PrimType
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T
from tvm.script.ir_builder.base import IRBuilderFrame
from tvm.tirx import IntImm, IterVar, Layout, Var, is_buffer_var
from tvm.tirx.script.builder.ir import name_meta_class_value
from tvm.tirx.stmt import BufferRegion


def builder_name(name: str, value):
    """Name a directly constructed builder value and return it."""
    try:
        return IRBuilder.name(name, value)
    except (TypeError, ValueError):
        return value


def builder_meta(name: str, value):
    """Name resources owned by a builder-native meta-class instance."""
    name_meta_class_value(name, value)
    return value


def builder_scalar(name: str, value, dtype: str | None = None):
    """Materialize the mutable scalar semantics used by TVMScript."""
    if dtype is None:
        dtype = str(value.ty.dtype)
    scalar = T.alloc_scalar(dtype=dtype, scope="local")
    IRBuilder.name(name, scalar.scalar.buffer)
    T.buffer_store(scalar.scalar.buffer, value, [0])
    return scalar.scalar


def builder_alloc_scalar(name: str, dtype: str):
    """Allocate a mutable scalar without inventing an initializer."""
    scalar = T.alloc_scalar(dtype=dtype, scope="local")
    IRBuilder.name(name, scalar.scalar.buffer)
    return scalar.scalar


def builder_bind(name: str, value, type_annotation=None):
    """Emit an immutable builder Bind with TVMScript naming semantics."""
    result = T.Bind(value, type_annotation)
    IRBuilder.name(name, result)
    return result


def builder_assign(name: str, value):
    """Classify a direct-builder assignment exactly as the TIRx parser does."""
    if isinstance(value, T.scalar_wrapper):
        IRBuilder.name(name, value.scalar.buffer)
        return value.scalar
    if getattr(type(value), "_is_meta_class", False):
        return builder_meta(name, value)
    if isinstance(value, BufferRegion):
        return value
    if isinstance(value, IRBuilderFrame):
        value.add_callback(lambda: value.__exit__(None, None, None))
        result = value.__enter__()
        IRBuilder.name(name, result)
        return result
    if is_buffer_var(value) or isinstance(value, (IterVar, Layout, tvm.ir.Var)):
        builder_name(name, value)
        return value
    is_pointer_expr = isinstance(value, tvm.ir.Expr) and isinstance(
        getattr(value, "ty", None), PointerType
    )
    if is_pointer_expr:
        return builder_bind(name, value)
    if not tvm.ir.is_prim_expr(value) and not isinstance(value, Expr):
        value = tvm.tirx.const(value)
    if isinstance(value, tvm.tirx.StringImm) or not tvm.ir.is_prim_expr(value):
        return builder_bind(name, value)
    return builder_scalar(name, value)


def builder_enter(frame):
    """Enter a flat builder frame until its enclosing PrimFunc completes."""
    frames = frame.frames if hasattr(frame, "frames") else [frame]
    prim_func_frame = next(
        item
        for item in reversed(IRBuilder.current().frames)
        if type(item).__name__ == "PrimFuncFrame"
    )
    for item in frames:
        prim_func_frame.add_callback(lambda item=item: item.__exit__(None, None, None))
        item.__enter__()


def builder_emit(value):
    """Match TVMScript expression-statement emission in direct builder code."""
    if value is None or isinstance(value, tvm.ir.Var):
        return
    if tvm.ir.is_prim_expr(value) or isinstance(value, tvm.ir.Call):
        T.evaluate(value)


def query_cancel_first_ctaid_x(first_ctaid_x, handle, *, use_ld_acquire=True):
    """Decode one cluster-launch-control response with direct builder emission."""
    response = T.alloc_scalar(dtype="uint128", scope="local").scalar
    canceled = T.alloc_scalar(dtype="uint32", scope="local").scalar
    T.evaluate(T.ptx[f"ld{'.acquire.cta' if use_ld_acquire else ''}.shared.b128"](response, handle))
    T.evaluate(T.ptx.clusterlaunchcontrol.query_cancel.is_canceled.pred.b128(canceled, response))
    T.buffer_store(first_ctaid_x.buffer, T.uint32(0xFFFFFFFF), [0])
    T.evaluate(
        T.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__x.b32.b128(
            first_ctaid_x, response, pred=canceled
        )
    )
    T.evaluate(T.ptx.fence.proxy.async_.shared__cta())


class SmemDescriptor:
    """Encoded shared-memory descriptor with direct builder emission."""

    _is_meta_class = True
    __static_attributes__ = ("_buf",)

    def __init__(self):
        self._buf = T.alloc_local([1], "uint64")

    @property
    def desc(self):
        return self._buf[0]

    def init(self, smem_ptr, ldo, sdo, swizzle):
        T.evaluate(
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(self._buf[0]), smem_ptr, ldo, sdo, swizzle
            )
        )

    def add_16B_offset(self, offset):
        from tvm.backend.cuda.tile_primitive.common import smem_desc_add_16B_offset

        result = smem_desc_add_16B_offset(self._buf[0], offset)
        # Keep the descriptor register class explicit for the PTX matcher.
        return T.cast(result, "uint64")

    def make_lo_uniform(self):
        func_name = "smem_desc_make_lo_uniform"
        source_code = f"""
__forceinline__ __device__ void {func_name}(uint64_t* desc) {{
    SmemDescriptor* d = reinterpret_cast<SmemDescriptor*>(desc);
    d->lo = __shfl_sync(0xffffffff, d->lo, 0);
}}
"""
        return T.cuda.func_call(
            func_name, T.address_of(self._buf[0]), source_code=source_code, return_type="void"
        )


class PipelineState:
    """Builder-native stage/phase state for a software-pipelined ring."""

    _is_meta_class = True
    __static_attributes__ = ("depth", "phase", "stage")

    def __init__(self, depth: int, phase=None):
        self.stage = T.alloc_scalar(dtype="int32", scope="local").scalar
        self.phase = T.alloc_scalar(dtype="int32", scope="local").scalar
        self.depth = depth
        if phase is not None:
            self.init(phase)

    def init(self, phase):
        T.buffer_store(self.stage.buffer, 0, [0])
        T.buffer_store(self.phase.buffer, phase, [0])

    def advance(self):
        if self.depth > 1:
            T.buffer_store(self.stage.buffer, self.stage + 1, [0])
            with T.If(self.stage == self.depth):
                with T.Then():
                    T.buffer_store(self.stage.buffer, 0, [0])
                    T.buffer_store(self.phase.buffer, self.phase ^ 1, [0])
        else:
            T.buffer_store(self.phase.buffer, self.phase ^ 1, [0])


def _map_addr_into_cta(ptr, rank):
    mapped = T.alloc_local([1], "uint32")
    T.evaluate(
        T.ptx.mapa.shared__cluster.u32(
            mapped[0], T.cuda.cvta_generic_to_shared(ptr), T.uint32(rank)
        )
    )
    return mapped[0]


def _map_buffer_into_cta(ptr, rank, depth):
    ptr_ty = PointerType(PrimType("uint64"), "shared")
    mapped = T.alloc_local([1], "uint64")
    T.evaluate(T.ptx.mapa.u64(mapped[0], ptr, T.uint32(rank)))
    remote_ptr = Var("remote_mbar_ptr", ptr_ty)
    T.Bind(T.reinterpret(ptr_ty, mapped[0]), var=remote_ptr)
    return T.decl_buffer([depth], "uint64", data=remote_ptr, scope="shared")


def _mbarrier_arrive_remote(bar, pred=None, count=None):
    chain = T.ptx.mbarrier.arrive.shared__cluster.b64
    args = (bar,) if count is None else (bar, T.uint32(count))
    T.evaluate(chain(*args, pred=pred) if pred is not None else chain(*args))


def _mbarrier_arrive_expect_tx_remote(bar, tx_count, pred=None):
    chain = T.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64
    args = (bar, T.uint32(tx_count))
    T.evaluate(chain(*args, pred=pred) if pred is not None else chain(*args))


class MBarrier:
    """Builder-native regular mbarrier."""

    _is_meta_class = True
    __static_attributes__ = ("_remote_cta_id", "buf", "depth", "leader", "phase_offset")

    def __init__(self, pool, depth, phase_offset=0, leader=None):
        self.buf = pool.alloc((depth,), "uint64", align=8)
        self._remote_cta_id = None
        self.depth = depth
        self.phase_offset = phase_offset
        self.leader = leader if leader is not None else T.cuda.thread_rank() == 0

    def init(self, count):
        if self._remote_cta_id is not None:
            raise ValueError("MBarrier.remote_view() cannot be initialized")
        with T.If(self.leader):
            with T.Then():
                with T.unroll(self.depth) as i:
                    T.evaluate(
                        T.ptx.mbarrier.init.shared.b64(self.buf.ptr_to([i]), T.uint32(count))
                    )

    def wait(self, stage, phase):
        if self._remote_cta_id is not None:
            raise ValueError("MBarrier.remote_view() cannot be waited on")
        T.evaluate(T.cuda.mbarrier_wait(self.buf.ptr_to([stage]), phase ^ self.phase_offset))

    def arrive(self, stage, remote=None, pred=None, count=None):
        if self._remote_cta_id is not None:
            if remote is not None:
                raise ValueError("MBarrier.remote_view().arrive() cannot also specify remote")
            _mbarrier_arrive_remote(self.buf.ptr_to([stage]), pred, count)
        elif remote is None:
            T.evaluate(T.ptx.mbarrier.arrive.shared.b64(self.buf.ptr_to([stage]), T.uint32(1)))
        else:
            _mbarrier_arrive_remote(
                _map_addr_into_cta(self.buf.ptr_to([stage]), remote), pred, count
            )

    def ptr_to(self, idx):
        return self.buf.ptr_to(idx)

    def remote_view(self, rank):
        if self._remote_cta_id is not None:
            raise ValueError("MBarrier.remote_view() cannot be applied to a remote view")
        buf = _map_buffer_into_cta(self.buf.ptr_to([0]), rank, self.depth)
        remote = object.__new__(type(self))
        remote.buf = buf
        remote._remote_cta_id = rank
        remote.depth = self.depth
        remote.phase_offset = self.phase_offset
        remote.leader = self.leader
        return remote


class TMABar(MBarrier):
    """Builder-native TMA completion barrier."""

    def arrive(self, stage, tx_count=None, remote=None, pred=None):
        if self._remote_cta_id is not None:
            if remote is not None:
                raise ValueError("TMABar.remote_view().arrive() cannot also specify remote")
            remote_bar = self.buf.ptr_to([stage])
        elif remote is not None:
            remote_bar = _map_addr_into_cta(self.buf.ptr_to([stage]), remote)
        else:
            if tx_count is None:
                T.evaluate(T.ptx.mbarrier.arrive.shared.b64(self.buf.ptr_to([stage]), T.uint32(1)))
            else:
                T.evaluate(
                    T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        self.buf.ptr_to([stage]), T.uint32(tx_count)
                    )
                )
            return
        if tx_count is None:
            _mbarrier_arrive_remote(remote_bar, pred)
        else:
            _mbarrier_arrive_expect_tx_remote(remote_bar, tx_count, pred)


def _tcgen05_commit_is_unicast(cta_mask):
    if cta_mask is None:
        return True
    if isinstance(cta_mask, IntImm):
        cta_mask = cta_mask.value
    return isinstance(cta_mask, int) and bin(cta_mask).count("1") <= 1


class TCGen05Bar(MBarrier):
    """Builder-native tcgen05 completion barrier."""

    def arrive(self, stage, cta_group=1, cta_mask=None, pred=None):
        if _tcgen05_commit_is_unicast(cta_mask):
            T.evaluate(
                T.ptx[
                    f"tcgen05.commit.cta_group::{cta_group}"
                    ".mbarrier::arrive::one.shared::cluster.b64"
                ](self.buf.ptr_to([stage]), pred=pred)
            )
        else:
            T.evaluate(
                T.ptx[
                    f"tcgen05.commit.cta_group::{cta_group}"
                    ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
                ](self.buf.ptr_to([stage]), T.Cast("uint16", cta_mask), pred=pred)
            )


class IketProfiler:
    """Direct-builder IKET annotations used by sparse prefill."""

    _is_meta_class = True
    __static_attributes__ = ()

    def mark(self, name: str, payload=None):
        T.evaluate(T.cuda.iket.mark(name) if payload is None else T.cuda.iket.mark(name, payload))

    def range_start(self, name: str, payload=None):
        return (
            T.cuda.iket.range_start(name)
            if payload is None
            else T.cuda.iket.range_start(name, payload)
        )

    def range_end(self, token, payload=None):
        T.evaluate(
            T.cuda.iket.range_end(token)
            if payload is None
            else T.cuda.iket.range_end(token, payload)
        )

    def range_push(self, name: str, payload=None):
        T.evaluate(
            T.cuda.iket.range_push(name)
            if payload is None
            else T.cuda.iket.range_push(name, payload)
        )

    def range_pop(self):
        T.evaluate(T.cuda.iket.range_pop())

    def sentinel_token(self, name: str):
        return T.cuda.iket.sentinel_token(name)


__all__ = [
    "IketProfiler",
    "MBarrier",
    "PipelineState",
    "SmemDescriptor",
    "TCGen05Bar",
    "TMABar",
    "builder_alloc_scalar",
    "builder_assign",
    "builder_bind",
    "builder_emit",
    "builder_enter",
    "builder_meta",
    "builder_name",
    "builder_scalar",
    "query_cancel_first_ctaid_x",
]
