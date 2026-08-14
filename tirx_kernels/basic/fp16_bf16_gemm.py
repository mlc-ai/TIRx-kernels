# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

from dataclasses import dataclass

import torch

import tvm
from tirx_kernels.runner import bench
from tvm.ir import PointerType, PrimType
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import tirx as T
from tvm.script.ir_builder.base import IRBuilderFrame
from tvm.tirx import IntImm, IterVar, Layout, Var, is_buffer_var
from tvm.tirx.script.builder.ir import name_meta_class_value


def prepare_data(dtype, M, N, K):
    # Pin the ordinal: a bare "cuda" is resolved at each allocation against the
    # then-current device, so every tensor here must name the same one.
    torch_dev = torch.device("cuda", torch.cuda.current_device())
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    A = torch.randn(M, K).to(dtype).to(torch_dev)
    B = torch.randn(N, K).to(dtype).to(torch_dev)
    C = torch.zeros((M, N), dtype=dtype).to(torch_dev)
    return (A, B, C)


_DTYPE_MAP = {"fp16": tvm.DataType("float16"), "bf16": tvm.DataType("bfloat16")}

_TMA_G2S_2SM = (
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
)
_TMA_G2S_3D_2SM = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
)
_TMA_S2G_EVICT_FIRST = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
_MMA_F16_2SM = "tcgen05.mma.cta_group::2.kind::f16"
_MMA_KEEP_ALL_LANES = (0, 0, 0, 0, 0, 0, 0, 0)
_TMEM_LD_16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
_TMEM_LD_32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
_TMEM_LD_64 = "tcgen05.ld.sync.aligned.32x32b.x64.b32"
_EVICT_FIRST_L2_POLICY = 0x12F0000000000000

_BUILDER_MISSING = object()


def _builder_runtime_condition(value):
    return value


def _builder_enter(frame):
    frames = frame.frames if hasattr(frame, "frames") else [frame]
    prim_func = next(
        candidate
        for candidate in reversed(IRBuilder.current().frames)
        if type(candidate).__name__ == "PrimFuncFrame"
    )
    for item in frames:
        prim_func.add_callback(lambda item=item: item.__exit__(None, None, None))
        item.__enter__()


def _builder_emit(value):
    if value is None or isinstance(value, tvm.ir.Var):
        return
    if isinstance(value, IRBuilderFrame) or (
        hasattr(value, "frames") and hasattr(value, "__enter__")
    ):
        _builder_enter(value)
    elif tvm.ir.is_prim_expr(value) or isinstance(value, tvm.ir.Call):
        T.evaluate(value)
    elif isinstance(value, int | bool):
        T.evaluate(tvm.tirx.const(value))


def _builder_alloc_scalar(name, dtype):
    scalar = T.local_scalar(dtype)
    IRBuilder.name(name, scalar.scalar.buffer)
    return scalar.scalar


def _builder_scalar(name, value, dtype):
    scalar = _builder_alloc_scalar(name, dtype)
    T.buffer_store(scalar.buffer, value, scalar.indices)
    return scalar


def _builder_buffer(name, shape, dtype):
    buffer = T.alloc_local(shape, dtype)
    IRBuilder.name(name, buffer)
    return buffer


def _builder_bind(name, value, type_annotation=None):
    result = T.Bind(value, type_annotation)
    IRBuilder.name(name, result)
    return result


def _builder_assign(name, value, previous=_BUILDER_MISSING):
    if isinstance(value, I.meta_var):
        return value.value
    if previous is not _BUILDER_MISSING:
        if isinstance(previous, T.scalar_wrapper | tvm.tirx.expr.BufferLoad):
            target = previous.scalar if isinstance(previous, T.scalar_wrapper) else previous
            T.buffer_store(target.buffer, value, target.indices)
            return target
        if (
            is_buffer_var(previous)
            and len(previous.ty.shape) == 1
            and bool(previous.ty.shape[0] == 1)
        ):
            try:
                T.buffer_store(previous, value, [0])
                return previous
            except TypeError:
                pass
    if getattr(type(value), "_is_meta_class", False):
        name_meta_class_value(name, value)
        return value
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _builder_assign(f"{name}_{index}", item)
        return value
    if is_buffer_var(value) or isinstance(value, IterVar | Layout):
        IRBuilder.name(name, value)
        return value
    if isinstance(value, tvm.ir.Var):
        if isinstance(value.ty, PointerType):
            return _builder_bind(name, value, value.ty)
        IRBuilder.name(name, value)
        return value
    if isinstance(value, tvm.ir.Expr) and isinstance(getattr(value, "ty", None), PointerType):
        return _builder_bind(name, value, value.ty)
    if isinstance(value, tvm.ir.Expr) and tvm.ir.is_prim_expr(value):
        return _builder_scalar(name, value, str(value.ty.dtype))
    if isinstance(value, tvm.tirx.expr.ExprOp):
        return _builder_scalar(name, value, "bool")
    return value


def _builder_assign_many(names, values, previous):
    return tuple(
        _builder_assign(name, value, old) for name, value, old in zip(names, values, previous)
    )


class PipelineState:
    """Builder-native stage and phase state for a software-pipelined ring."""

    _is_meta_class = True
    __static_attributes__ = ("depth", "phase", "stage")

    def __init__(self, depth, phase=None):
        self.stage = T.local_scalar("int32").scalar
        self.phase = T.local_scalar("int32").scalar
        self.depth = depth
        if phase is not None:
            self.init(phase)

    def init(self, phase):
        T.buffer_store(self.stage.buffer, 0, self.stage.indices)
        T.buffer_store(self.phase.buffer, phase, self.phase.indices)

    def advance(self):
        if self.depth > 1:
            T.buffer_store(self.stage.buffer, self.stage + 1, self.stage.indices)
            with T.If(self.stage == self.depth):
                with T.Then():
                    T.buffer_store(self.stage.buffer, 0, self.stage.indices)
                    T.buffer_store(self.phase.buffer, self.phase ^ 1, self.phase.indices)
        else:
            T.buffer_store(self.phase.buffer, self.phase ^ 1, self.phase.indices)


def _map_addr_into_cta(ptr, rank):
    mapped = T.alloc_local((1,), "uint32")
    T.evaluate(
        T.ptx.mapa.shared__cluster.u32(
            mapped[0], T.cuda.cvta_generic_to_shared(ptr), T.uint32(rank)
        )
    )
    return mapped[0]


def _map_buffer_into_cta(ptr, rank, depth):
    ptr_ty = PointerType(PrimType("uint64"), "shared")
    mapped = T.alloc_local((1,), "uint64")
    T.evaluate(T.ptx.mapa.u64(mapped[0], ptr, T.uint32(rank)))
    remote_ptr = Var("remote_mbar_ptr", ptr_ty)
    T.Bind(T.reinterpret(ptr_ty, mapped[0]), var=remote_ptr)
    return T.decl_buffer((depth,), "uint64", data=remote_ptr, scope="shared")


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


class Pipeline:
    """Builder-native full/empty barrier pair."""

    _is_meta_class = True
    __static_attributes__ = ("empty", "full", "stages")

    def __init__(
        self,
        pool,
        stages,
        *,
        full,
        empty,
        init_full=1,
        init_empty=1,
        empty_phase_offset=0,
        leader=None,
    ):
        barrier_kinds = {"tma": TMABar, "tcgen05": TCGen05Bar, "mbar": MBarrier}
        self.stages = stages
        self.full = barrier_kinds[full](pool, stages, leader=leader)
        self.full.init(init_full)
        self.empty = barrier_kinds[empty](
            pool, stages, phase_offset=empty_phase_offset, leader=leader
        )
        self.empty.init(init_empty)


class SmemDescriptor:
    """Builder-native tcgen05 shared-memory descriptor."""

    _is_meta_class = True
    __static_attributes__ = ("_buf",)

    def __init__(self):
        self._buf = T.alloc_local((1,), "uint64")

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

        return smem_desc_add_16B_offset(self._buf[0], offset)


def _query_cancel_first_ctaid_x(first_ctaid_x, handle):
    response = T.local_scalar("uint128").scalar
    canceled = T.local_scalar("uint32").scalar
    T.evaluate(T.ptx.ld.acquire.cta.shared.b128(response, handle))
    T.evaluate(T.ptx.clusterlaunchcontrol.query_cancel.is_canceled.pred.b128(canceled, response))
    T.buffer_store(first_ctaid_x.buffer, T.uint32(0xFFFFFFFF), first_ctaid_x.indices)
    T.evaluate(
        T.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__x.b32.b128(
            first_ctaid_x, response, pred=canceled
        )
    )
    T.evaluate(T.ptx.fence.proxy.async_.shared__cta())


class _CLCWorker:
    """Builder-native worker side of the CLC shared-memory handshake."""

    _is_meta_class = True

    def __init__(self, clc, prefix):
        del prefix
        self._clc = clc
        self.m_idx = T.local_scalar("int32").scalar
        self.n_idx = T.local_scalar("int32").scalar
        self.linear_idx = T.local_scalar("int32").scalar
        self.tile_count = T.local_scalar("int32").scalar
        self._sa = PipelineState(1, 0)
        self._done = T.local_scalar("int32").scalar
        self._nxt = T.local_scalar("uint32").scalar

    def _update_current_m_n_idx(self, work_idx):
        cluster_m_offset = work_idx % 1
        cluster_linear = work_idx // 1
        cluster_n_offset = cluster_linear % 1
        tile_linear = cluster_linear // 1
        group_size = self._clc._l2_group_size
        group_span = group_size * self._clc._num_n_tiles
        full_groups = self._clc._num_m_tiles // group_size
        with T.If((full_groups > 0) & (tile_linear < full_groups * group_span)):
            with T.Then():
                group_id = T.Bind(tile_linear // group_span)
                within_group = T.Bind(tile_linear % group_span)
                tile_row = T.Bind(group_id * group_size + within_group % group_size)
                tile_col = T.Bind(within_group // group_size)
                T.buffer_store(self.m_idx.buffer, tile_row + cluster_m_offset, self.m_idx.indices)
                T.buffer_store(self.n_idx.buffer, tile_col + cluster_n_offset, self.n_idx.indices)
            with T.Else():
                T.buffer_store(self.m_idx.buffer, cluster_m_offset, self.m_idx.indices)
                T.buffer_store(self.n_idx.buffer, cluster_n_offset, self.n_idx.indices)

    def reset(self):
        T.buffer_store(self._done.buffer, 0, self._done.indices)

    def init(self, cluster_id):
        T.buffer_store(self.linear_idx.buffer, cluster_id, self.linear_idx.indices)
        T.buffer_store(self.tile_count.buffer, 0, self.tile_count.indices)
        self._update_current_m_n_idx(cluster_id)
        T.buffer_store(self._done.buffer, 0, self._done.indices)

    def valid(self):
        return self._done == 0

    def consume(self):
        self._clc.sched_arr.full.wait(0, self._sa.phase)
        self._sa.advance()
        _query_cancel_first_ctaid_x(self._nxt, T.address_of(self._clc.clc_handle[0]))
        self._clc.sched_fin.empty.arrive(0, remote=0, pred=True)

    def consume_wg(self, wg_id, warp_id, lane_id):
        self._clc.sched_arr.full.wait(0, self._sa.phase)
        self._sa.advance()
        _query_cancel_first_ctaid_x(self._nxt, T.address_of(self._clc.clc_handle[0]))
        T.evaluate(T.cuda.warpgroup_sync(wg_id + 1))
        with T.If((warp_id == 0) & (lane_id == 0)):
            with T.Then():
                self._clc.sched_fin.empty.arrive(0, remote=0, pred=True)

    def advance_coords(self):
        with T.If(self._nxt != T.uint32(0xFFFFFFFF)):
            with T.Then():
                self._update_current_m_n_idx(self._nxt // self._clc._cta_group)

    def mark_done_if_drained(self):
        with T.If(self._nxt == T.uint32(0xFFFFFFFF)):
            with T.Then():
                T.buffer_store(self._done.buffer, 1, self._done.indices)


class ClusterLaunchControlScheduler:
    """Builder-native Blackwell cluster launch control scheduler."""

    _is_meta_class = True

    def __init__(self, pool, num_m_tiles, num_n_tiles, l2_group_size, cta_group, finish_arrivals):
        self._num_m_tiles = num_m_tiles
        self._num_n_tiles = num_n_tiles
        self._l2_group_size = l2_group_size
        self._cta_group = cta_group
        self.sched_arr = Pipeline(pool, 1, full="tma", empty="mbar", init_empty=1)
        self.sched_fin = Pipeline(pool, 1, full="mbar", empty="mbar", init_empty=finish_arrivals)
        self.clc_handle = pool.alloc((4,), "uint32", align=16)
        self._s_done = T.local_scalar("int32").scalar
        self._s_nxt = T.local_scalar("uint32").scalar

    def worker(self, prefix):
        return _CLCWorker(self, prefix)

    def run_scheduler(self, cbx):
        with T.If(T.cuda.elect_sync()):
            with T.Then():
                sa = PipelineState(1, 0)
                sf = PipelineState(1, 1)
                T.buffer_store(self._s_done.buffer, 0, self._s_done.indices)
                with T.While(self._s_done == 0):
                    with T.If(cbx == 0):
                        with T.Then():
                            self.sched_fin.empty.wait(0, sf.phase)
                            sf.advance()
                            T.evaluate(
                                T.ptx[
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                ](
                                    T.address_of(self.clc_handle[0]),
                                    T.address_of(self.sched_arr.full.buf[0]),
                                )
                            )
                    self.sched_arr.full.arrive(0, 16)
                    self.sched_arr.full.wait(0, sa.phase)
                    sa.advance()
                    _query_cancel_first_ctaid_x(self._s_nxt, T.address_of(self.clc_handle[0]))
                    self.sched_fin.empty.arrive(0, remote=0, pred=True)
                    with T.If(self._s_nxt == T.uint32(0xFFFFFFFF)):
                        with T.Then():
                            T.buffer_store(self._s_done.buffer, 1, self._s_done.indices)


def _swizzle_for_row_bytes(row_bytes):
    """Pick the MMA-shared swizzle atom matching the tile row width (the 128/64/32B
    swizzle is selected from the row byte width)."""
    from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode

    if row_bytes % 128 == 0:
        return SwizzleMode.SWIZZLE_128B_ATOM
    if row_bytes % 64 == 0:
        return SwizzleMode.SWIZZLE_64B_ATOM
    return SwizzleMode.SWIZZLE_32B_ATOM


# Per-shape tuning knobs (CTA_M=256 always): cta_n/cta_k MMA tile, l2_group_size (L2
# supergroup), overlap_epilogue, pipe_depth (A/B smem), wb_pipe_depth (epilogue).
GEMM_CONFIGS = {
    1024: {
        "cta_n": 64,
        "cta_k": 128,
        "l2_group_size": 4,
        "overlap_epilogue": True,
        "pipe_depth": 5,
        "wb_pipe_depth": 2,
    },
    2048: {
        "cta_n": 256,
        "cta_k": 64,
        "l2_group_size": 8,
        "overlap_epilogue": True,
        "pipe_depth": 5,
        "wb_pipe_depth": 4,
    },
    4096: {
        "cta_n": 256,
        "cta_k": 64,
        "l2_group_size": 4,
        "overlap_epilogue": False,
        "pipe_depth": 4,
        "wb_pipe_depth": 8,
    },
    8192: {
        "cta_n": 256,
        "cta_k": 64,
        "l2_group_size": 8,
        "overlap_epilogue": False,
        "pipe_depth": 4,
        "wb_pipe_depth": 8,
    },
    16384: {
        "cta_n": 256,
        "cta_k": 64,
        "l2_group_size": 8,
        "overlap_epilogue": False,
        "pipe_depth": 4,
        "wb_pipe_depth": 8,
    },
}
# Default config for shapes not in the table above.
_DEFAULT_CONFIG = {
    "cta_n": 256,
    "cta_k": 64,
    "l2_group_size": 8,
    "overlap_epilogue": False,
    "pipe_depth": 4,
    "wb_pipe_depth": 8,
}


def _kernel(
    *, M, N, K, ab_type, MMA_N, BLK_K, PIPE_DEPTH, WB_PIPE_DEPTH, L2_GROUP_SIZE, OVERLAP_EPILOGUE
):
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("_kernel")
            A = T.arg("A", T.Buffer((M, K), ab_type))
            B = T.arg("B", T.Buffer((N, K), ab_type))
            D = T.arg("D", T.Buffer((M, N), ab_type))
            NUM_CONSUMER = 1 if OVERLAP_EPILOGUE else 2
            MMA_PIPE = 2 if OVERLAP_EPILOGUE else 1
            TMEM_SLOTS = MMA_PIPE if OVERLAP_EPILOGUE else NUM_CONSUMER
            TMEM_PHASE_DEPTH = MMA_PIPE if OVERLAP_EPILOGUE else 1
            NUM_D_TILES = 2 if WB_PIPE_DEPTH > 1 else 1
            BLK_M = 128
            BLK_N = MMA_N // 2
            EPI_N = MMA_N // WB_PIPE_DEPTH
            AB_DTYPE = str(ab_type)
            D_SWIZZLE = _swizzle_for_row_bytes(EPI_N * (ab_type.bits // 8)).value
            CVT_F32X2 = "cvt.rn.f16x2.f32" if AB_DTYPE == "float16" else "cvt.rn.bf16x2.f32"
            TMEM_LD_OVERLAP = _TMEM_LD_32 if EPI_N == 32 else _TMEM_LD_64
            A_tensor_map = _builder_bind(
                "A_tensor_map", T.tvm_stack_alloca("tensormap", 1), T.TensorMap()
            )
            B_tensor_map = _builder_bind(
                "B_tensor_map", T.tvm_stack_alloca("tensormap", 1), T.TensorMap()
            )
            D_tensor_map = _builder_bind(
                "D_tensor_map", T.tvm_stack_alloca("tensormap", 1), T.TensorMap()
            )
            if BLK_K == 128:
                _builder_emit(
                    T.call_packed(
                        "runtime.cuTensorMapEncodeTiled",
                        A_tensor_map,
                        AB_DTYPE,
                        3,
                        A.data,
                        64,
                        M,
                        K // 64,
                        K * (ab_type.bits // 8),
                        64 * (ab_type.bits // 8),
                        64,
                        BLK_M,
                        BLK_K // 64,
                        1,
                        1,
                        1,
                        0,
                        3,
                        2,
                        0,
                    )
                )
                _builder_emit(
                    T.call_packed(
                        "runtime.cuTensorMapEncodeTiled",
                        B_tensor_map,
                        AB_DTYPE,
                        3,
                        B.data,
                        64,
                        N,
                        K // 64,
                        K * (ab_type.bits // 8),
                        64 * (ab_type.bits // 8),
                        64,
                        BLK_N,
                        BLK_K // 64,
                        1,
                        1,
                        1,
                        0,
                        3,
                        2,
                        0,
                    )
                )
            else:
                _builder_emit(
                    T.call_packed(
                        "runtime.cuTensorMapEncodeTiled",
                        A_tensor_map,
                        AB_DTYPE,
                        2,
                        A.data,
                        K,
                        M,
                        K * (ab_type.bits // 8),
                        BLK_K,
                        BLK_M,
                        1,
                        1,
                        0,
                        3,
                        2,
                        0,
                    )
                )
                _builder_emit(
                    T.call_packed(
                        "runtime.cuTensorMapEncodeTiled",
                        B_tensor_map,
                        AB_DTYPE,
                        2,
                        B.data,
                        K,
                        N,
                        K * (ab_type.bits // 8),
                        BLK_K,
                        BLK_N,
                        1,
                        1,
                        0,
                        3,
                        2,
                        0,
                    )
                )
            _builder_emit(
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    D_tensor_map,
                    AB_DTYPE,
                    2,
                    D.data,
                    N,
                    M,
                    N * (ab_type.bits // 8),
                    EPI_N,
                    BLK_M,
                    1,
                    1,
                    0,
                    D_SWIZZLE,
                    2,
                    0,
                )
            )
            _builder_emit(T.device_entry())
            _builder_emit(T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1}))
            cbx, cby = _builder_assign_many(
                ("cbx", "cby"),
                T.cta_id_in_cluster([2, 1], preferred=[2, 1]),
                (locals().get("cbx", _BUILDER_MISSING), locals().get("cby", _BUILDER_MISSING)),
            )
            bx = _builder_assign(
                "bx",
                T.cta_id([M // (256 * NUM_CONSUMER) * (N // MMA_N) * 2]),
                locals().get("bx", _BUILDER_MISSING),
            )
            wg_id = _builder_assign(
                "wg_id", T.warpgroup_id([NUM_CONSUMER + 1]), locals().get("wg_id", _BUILDER_MISSING)
            )
            warp_id = _builder_assign(
                "warp_id", T.warp_id_in_wg([4]), locals().get("warp_id", _BUILDER_MISSING)
            )
            lane_id = _builder_assign(
                "lane_id", T.lane_id([32]), locals().get("lane_id", _BUILDER_MISSING)
            )
            with T.If((wg_id == 0) & (warp_id == 0)):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            _builder_emit(
                                T.evaluate(T.ptx.prefetch.tensormap(T.address_of(A_tensor_map)))
                            )
                            _builder_emit(
                                T.evaluate(T.ptx.prefetch.tensormap(T.address_of(B_tensor_map)))
                            )
                            _builder_emit(
                                T.evaluate(T.ptx.prefetch.tensormap(T.address_of(D_tensor_map)))
                            )
            pool = _builder_assign("pool", T.SMEMPool(), locals().get("pool", _BUILDER_MISSING))
            tmem_addr = _builder_assign(
                "tmem_addr", pool.alloc((1,), "uint32"), locals().get("tmem_addr", _BUILDER_MISSING)
            )
            tmem_pool = _builder_assign(
                "tmem_pool",
                T.TMEMPool(pool, total_cols=512, cta_group=2, tmem_addr=tmem_addr),
                locals().get("tmem_pool", _BUILDER_MISSING),
            )
            smem_pipe = _builder_assign(
                "smem_pipe",
                Pipeline(pool, PIPE_DEPTH, full="tma", empty="tcgen05", init_empty=NUM_CONSUMER),
                locals().get("smem_pipe", _BUILDER_MISSING),
            )
            tmem_pipe = _builder_assign(
                "tmem_pipe",
                Pipeline(pool, TMEM_SLOTS, full="tcgen05", empty="mbar", init_empty=2 * 128),
                locals().get("tmem_pipe", _BUILDER_MISSING),
            )
            clc_sched = _builder_assign(
                "clc_sched",
                ClusterLaunchControlScheduler(
                    pool,
                    num_m_tiles=M // (256 * NUM_CONSUMER),
                    num_n_tiles=N // MMA_N,
                    l2_group_size=L2_GROUP_SIZE,
                    cta_group=2,
                    finish_arrivals=(2 + NUM_CONSUMER) * 2 + NUM_CONSUMER,
                ),
                locals().get("clc_sched", _BUILDER_MISSING),
            )
            tmem_fin = _builder_assign(
                "tmem_fin",
                Pipeline(pool, 1, full="mbar", empty="mbar", init_full=1),
                locals().get("tmem_fin", _BUILDER_MISSING),
            )
            _builder_emit(pool.move_base_to(1024))
            Asmem = _builder_assign(
                "Asmem",
                pool.alloc_tcgen05_mma_AB((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), ab_type),
                locals().get("Asmem", _BUILDER_MISSING),
            )
            Bsmem = _builder_assign(
                "Bsmem",
                pool.alloc_tcgen05_mma_AB((PIPE_DEPTH, BLK_N, BLK_K), ab_type),
                locals().get("Bsmem", _BUILDER_MISSING),
            )
            Dsmem = _builder_assign(
                "Dsmem",
                pool.alloc_tcgen05_mma_AB(
                    (NUM_CONSUMER, NUM_D_TILES, BLK_M, EPI_N),
                    ab_type,
                    swizzle_mode=_swizzle_for_row_bytes(EPI_N * (ab_type.bits // 8)),
                ),
                locals().get("Dsmem", _BUILDER_MISSING),
            )
            _builder_emit(pool.commit())
            smem_full_cta0 = _builder_assign(
                "smem_full_cta0",
                smem_pipe.full.remote_view(0),
                locals().get("smem_full_cta0", _BUILDER_MISSING),
            )
            tmem = _builder_assign(
                "tmem",
                tmem_pool.alloc((128, 512), "float32"),
                locals().get("tmem", _BUILDER_MISSING),
            )
            _builder_emit(tmem_pool.commit())
            _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
            _builder_emit(T.ptx.fence.mbarrier_init.release.cluster())
            if OVERLAP_EPILOGUE:
                _builder_emit(T.ptx.barrier.cluster.arrive.relaxed.aligned())
            else:
                _builder_emit(T.cuda.cluster_sync())
            with T.If(wg_id == NUM_CONSUMER):
                with T.Then():
                    _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(56))
                    with T.If(warp_id == 3):
                        with T.Then():
                            ld = _builder_assign(
                                "ld",
                                clc_sched.worker("ld_sched"),
                                locals().get("ld", _BUILDER_MISSING),
                            )
                            _builder_emit(ld.init(bx // 2))
                            tma_cur = _builder_assign(
                                "tma_cur",
                                PipelineState(PIPE_DEPTH, 1),
                                locals().get("tma_cur", _BUILDER_MISSING),
                            )
                            if OVERLAP_EPILOGUE:
                                _builder_emit(T.ptx.barrier.cluster.wait.acquire())

                            def tma_load_stage(k_tile, m_idx, n_idx):
                                _builder_emit(smem_pipe.empty.wait(tma_cur.stage, tma_cur.phase))
                                stage = _builder_assign(
                                    "stage", tma_cur.stage, locals().get("stage", _BUILDER_MISSING)
                                )
                                k = k_tile * BLK_K
                                b_n = (n_idx * 2 + cbx) * BLK_N
                                with T.unroll(NUM_CONSUMER) as c:
                                    IRBuilder.name("c", c)
                                    a_m = ((m_idx * 2 + cbx) * NUM_CONSUMER + c) * BLK_M
                                    if BLK_K == 128:
                                        _builder_emit(
                                            T.evaluate(
                                                T.ptx[_TMA_G2S_3D_2SM](
                                                    Asmem.ptr_to([stage, c, 0, 0]),
                                                    T.address_of(A_tensor_map),
                                                    T.int32(0),
                                                    T.cast(a_m, "int32"),
                                                    T.cast(k // 64, "int32"),
                                                    T.cuda.cvta_generic_to_shared(
                                                        smem_full_cta0.ptr_to([stage])
                                                    ),
                                                )
                                            )
                                        )
                                    else:
                                        _builder_emit(
                                            T.evaluate(
                                                T.ptx[_TMA_G2S_2SM](
                                                    Asmem.ptr_to([stage, c, 0, 0]),
                                                    T.address_of(A_tensor_map),
                                                    T.cast(k, "int32"),
                                                    T.cast(a_m, "int32"),
                                                    T.cuda.cvta_generic_to_shared(
                                                        smem_full_cta0.ptr_to([stage])
                                                    ),
                                                )
                                            )
                                        )
                                if BLK_K == 128:
                                    _builder_emit(
                                        T.evaluate(
                                            T.ptx[_TMA_G2S_3D_2SM](
                                                Bsmem.ptr_to([stage, 0, 0]),
                                                T.address_of(B_tensor_map),
                                                T.int32(0),
                                                T.cast(b_n, "int32"),
                                                T.cast(k // 64, "int32"),
                                                T.cuda.cvta_generic_to_shared(
                                                    smem_full_cta0.ptr_to([stage])
                                                ),
                                            )
                                        )
                                    )
                                else:
                                    _builder_emit(
                                        T.evaluate(
                                            T.ptx[_TMA_G2S_2SM](
                                                Bsmem.ptr_to([stage, 0, 0]),
                                                T.address_of(B_tensor_map),
                                                T.cast(k, "int32"),
                                                T.cast(b_n, "int32"),
                                                T.cuda.cvta_generic_to_shared(
                                                    smem_full_cta0.ptr_to([stage])
                                                ),
                                            )
                                        )
                                    )
                                with T.If(cbx == 0):
                                    with T.Then():
                                        _builder_emit(
                                            smem_full_cta0.arrive(
                                                stage,
                                                2
                                                * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K)
                                                * (ab_type.bits // 8),
                                            )
                                        )

                            def tma_load(m_idx, n_idx):
                                with T.serial(K // BLK_K) as k_tile:
                                    IRBuilder.name("k_tile", k_tile)
                                    _builder_emit(tma_load_stage(k_tile, m_idx, n_idx))
                                    _builder_emit(tma_cur.advance())

                            with T.If(T.cuda.elect_sync()):
                                with T.Then():
                                    with T.While(ld.valid()):
                                        m_idx = ld.m_idx
                                        n_idx = ld.n_idx
                                        _builder_emit(tma_load(m_idx, n_idx))
                                        _builder_emit(ld.consume())
                                        _builder_emit(ld.advance_coords())
                                        _builder_emit(ld.mark_done_if_drained())
                        with T.Else():
                            with T.If(warp_id == 2):
                                with T.Then():
                                    if OVERLAP_EPILOGUE:
                                        _builder_emit(T.ptx.barrier.cluster.wait.acquire())
                                    _builder_emit(clc_sched.run_scheduler(cbx))
                                with T.Else():
                                    with T.If((warp_id < NUM_CONSUMER) & (cbx == 0)):
                                        with T.Then():
                                            mma_smem = _builder_assign(
                                                "mma_smem",
                                                PipelineState(PIPE_DEPTH, 0),
                                                locals().get("mma_smem", _BUILDER_MISSING),
                                            )
                                            tmem_buf = _builder_assign(
                                                "tmem_buf",
                                                PipelineState(TMEM_PHASE_DEPTH, 1),
                                                locals().get("tmem_buf", _BUILDER_MISSING),
                                            )
                                            desc_a = SmemDescriptor()
                                            desc_b = SmemDescriptor()
                                            desc_i = _builder_assign(
                                                "desc_i",
                                                T.alloc_local((1,), "uint32"),
                                                locals().get("desc_i", _BUILDER_MISSING),
                                            )
                                            accum = _builder_alloc_scalar("accum", "int32")
                                            if OVERLAP_EPILOGUE:
                                                _builder_emit(T.ptx.barrier.cluster.wait.acquire())

                                            def mma_stage(buf):
                                                nonlocal accum
                                                _builder_emit(
                                                    smem_pipe.full.wait(
                                                        mma_smem.stage, mma_smem.phase
                                                    )
                                                )
                                                stage = _builder_assign(
                                                    "stage",
                                                    mma_smem.stage,
                                                    locals().get("stage", _BUILDER_MISSING),
                                                )
                                                tmem_n = buf * MMA_N
                                                with T.unroll(BLK_K // 16) as ki:
                                                    IRBuilder.name("ki", ki)
                                                    desc_a_ki = desc_a.add_16B_offset(
                                                        (stage * NUM_CONSUMER + warp_id)
                                                        * BLK_M
                                                        * BLK_K
                                                        // 8
                                                        + ki // 4 * BLK_M * 8
                                                        + 2 * (ki % 4)
                                                    )
                                                    desc_b_ki = desc_b.add_16B_offset(
                                                        stage * BLK_N * BLK_K // 8
                                                        + ki // 4 * BLK_N * 8
                                                        + 2 * (ki % 4)
                                                    )
                                                    _builder_emit(
                                                        T.evaluate(
                                                            T.ptx[_MMA_F16_2SM](
                                                                T.cast(tmem_n, "uint32"),
                                                                desc_a_ki,
                                                                desc_b_ki,
                                                                desc_i[0],
                                                                *_MMA_KEEP_ALL_LANES,
                                                                T.ptx.pred(
                                                                    tvm.tirx.any(
                                                                        ki != 0,
                                                                        T.cast(accum, "bool"),
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    )
                                                accum = _builder_assign("accum", 1, accum)
                                                _builder_emit(
                                                    smem_pipe.empty.arrive(
                                                        mma_smem.stage, cta_group=2, cta_mask=3
                                                    )
                                                )

                                            def mma():
                                                nonlocal accum
                                                slot = (
                                                    tmem_buf.stage if OVERLAP_EPILOGUE else warp_id
                                                )
                                                _builder_emit(
                                                    tmem_pipe.empty.wait(slot, tmem_buf.phase)
                                                )
                                                accum = _builder_assign("accum", 0, accum)
                                                with T.serial(K // BLK_K) as k_tile:
                                                    IRBuilder.name("k_tile", k_tile)
                                                    _builder_emit(mma_stage(slot))
                                                    _builder_emit(mma_smem.advance())
                                                _builder_emit(
                                                    tmem_pipe.full.arrive(
                                                        slot, cta_group=2, cta_mask=3
                                                    )
                                                )
                                                _builder_emit(tmem_buf.advance())

                                            mm = _builder_assign(
                                                "mm",
                                                clc_sched.worker("mma_sched"),
                                                locals().get("mm", _BUILDER_MISSING),
                                            )
                                            _builder_emit(mm.reset())
                                            with T.If(T.cuda.elect_sync()):
                                                with T.Then():
                                                    _builder_emit(
                                                        desc_a.init(
                                                            Asmem.ptr_to([0, 0, 0, 0]),
                                                            ldo=BLK_M * 8 if BLK_K == 128 else 0,
                                                            sdo=64,
                                                            swizzle=3,
                                                        )
                                                    )
                                                    _builder_emit(
                                                        desc_b.init(
                                                            Bsmem.ptr_to([0, 0, 0]),
                                                            ldo=BLK_N * 8 if BLK_K == 128 else 0,
                                                            sdo=64,
                                                            swizzle=3,
                                                        )
                                                    )
                                                    _builder_emit(
                                                        T.cuda.tcgen05.encode_instr_descriptor(
                                                            T.address_of(desc_i[0]),
                                                            d_dtype="float32",
                                                            a_dtype=AB_DTYPE,
                                                            b_dtype=AB_DTYPE,
                                                            M=256,
                                                            N=MMA_N,
                                                            K=16,
                                                            trans_a=False,
                                                            trans_b=False,
                                                            n_cta_groups=2,
                                                        )
                                                    )
                                                    with T.While(mm.valid()):
                                                        _builder_emit(mm.consume())
                                                        _builder_emit(mma())
                                                        _builder_emit(mm.mark_done_if_drained())
                with T.Else():
                    with T.If(wg_id < NUM_CONSUMER):
                        with T.Then():
                            if not OVERLAP_EPILOGUE:
                                _builder_emit(T.ptx.setmaxnreg.inc.sync.aligned.u32(224))
                            wb = _builder_assign(
                                "wb",
                                clc_sched.worker("wb_sched"),
                                locals().get("wb", _BUILDER_MISSING),
                            )
                            _builder_emit(wb.init(bx // 2))
                            wb_buf = _builder_assign(
                                "wb_buf",
                                PipelineState(TMEM_PHASE_DEPTH, 0),
                                locals().get("wb_buf", _BUILDER_MISSING),
                            )
                            if OVERLAP_EPILOGUE:
                                _builder_emit(T.ptx.barrier.cluster.wait.acquire())

                            def writeback(m_idx, n_idx):
                                slot = wb_buf.stage if OVERLAP_EPILOGUE else wg_id
                                _builder_emit(tmem_pipe.full.wait(slot, wb_buf.phase))
                                tmem_base = slot * MMA_N
                                if OVERLAP_EPILOGUE:
                                    Dreg_16b = _builder_assign(
                                        "Dreg_16b",
                                        T.alloc_local((EPI_N // 2,), "uint32", align=16),
                                        locals().get("Dreg_16b", _BUILDER_MISSING),
                                    )
                                    with T.unroll(WB_PIPE_DEPTH) as i:
                                        IRBuilder.name("i", i)
                                        Dreg = _builder_assign(
                                            "Dreg",
                                            T.alloc_local((EPI_N,), "float32"),
                                            locals().get("Dreg", _BUILDER_MISSING),
                                        )
                                        tn = tmem_base + i * EPI_N
                                        _builder_emit(
                                            T.evaluate(
                                                T.ptx[TMEM_LD_OVERLAP](
                                                    *[Dreg[j] for j in range(EPI_N)],
                                                    T.cast(tn, "uint32"),
                                                )
                                            )
                                        )
                                        _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                                        with T.unroll(EPI_N // 2) as j:
                                            IRBuilder.name("j", j)
                                            _builder_emit(
                                                T.evaluate(
                                                    T.ptx[CVT_F32X2](
                                                        Dreg_16b[j], Dreg[j * 2 + 1], Dreg[j * 2]
                                                    )
                                                )
                                            )
                                        with T.If(i == WB_PIPE_DEPTH - 1):
                                            with T.Then():
                                                _builder_emit(
                                                    tmem_pipe.empty.arrive(
                                                        slot, remote=0, pred=True
                                                    )
                                                )
                                        db = i % NUM_D_TILES
                                        _builder_emit(
                                            T.ptx.cp.async_.bulk.wait_group.read(NUM_D_TILES - 1)
                                        )
                                        _builder_emit(T.cuda.warpgroup_sync(wg_id + 10))
                                        with T.unroll(EPI_N // 8) as jv:
                                            IRBuilder.name("jv", jv)
                                            r0 = jv * 4
                                            _builder_emit(
                                                T.ptx.st.shared.v4.u32(
                                                    Dsmem.ptr_to(
                                                        [0, db, warp_id * 32 + lane_id, jv * 8]
                                                    ),
                                                    Dreg_16b[r0],
                                                    Dreg_16b[r0 + 1],
                                                    Dreg_16b[r0 + 2],
                                                    Dreg_16b[r0 + 3],
                                                )
                                            )
                                        _builder_emit(T.cuda.warpgroup_sync(wg_id + 10))
                                        with T.If((warp_id == 0) & (lane_id == 0)):
                                            with T.Then():
                                                _builder_emit(
                                                    T.ptx.fence.proxy.async_.shared__cta()
                                                )
                                                d_m = (
                                                    (m_idx * 2 + cbx) * NUM_CONSUMER + wg_id
                                                ) * BLK_M
                                                d_n = n_idx * MMA_N + i * EPI_N
                                                _builder_emit(
                                                    T.evaluate(
                                                        T.ptx[_TMA_S2G_EVICT_FIRST](
                                                            T.address_of(D_tensor_map),
                                                            T.cast(d_n, "int32"),
                                                            T.cast(d_m, "int32"),
                                                            Dsmem.ptr_to([0, db, 0, 0]),
                                                            T.uint64(_EVICT_FIRST_L2_POLICY),
                                                        )
                                                    )
                                                )
                                        _builder_emit(T.ptx.cp.async_.bulk.commit_group())
                                else:
                                    NOL = 16
                                    Dreg_16b = _builder_assign(
                                        "Dreg_16b",
                                        T.alloc_local((MMA_N // 2,), "uint32", align=16),
                                        locals().get("Dreg_16b", _BUILDER_MISSING),
                                    )
                                    with T.unroll(MMA_N // NOL) as i:
                                        IRBuilder.name("i", i)
                                        Dreg = _builder_assign(
                                            "Dreg",
                                            T.alloc_local((NOL,), "float32"),
                                            locals().get("Dreg", _BUILDER_MISSING),
                                        )
                                        tn = tmem_base + i * NOL
                                        _builder_emit(
                                            T.evaluate(
                                                T.ptx[_TMEM_LD_16](
                                                    *[Dreg[j] for j in range(NOL)],
                                                    T.cast(tn, "uint32"),
                                                )
                                            )
                                        )
                                        _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                                        with T.unroll(NOL // 2) as j:
                                            IRBuilder.name("j", j)
                                            _builder_emit(
                                                T.evaluate(
                                                    T.ptx[CVT_F32X2](
                                                        Dreg_16b[i * (NOL // 2) + j],
                                                        Dreg[j * 2 + 1],
                                                        Dreg[j * 2],
                                                    )
                                                )
                                            )
                                    _builder_emit(
                                        tmem_pipe.empty.arrive(wg_id, remote=0, pred=True)
                                    )
                                    with T.unroll(WB_PIPE_DEPTH) as i:
                                        IRBuilder.name("i", i)
                                        db = i % NUM_D_TILES
                                        _builder_emit(
                                            T.ptx.cp.async_.bulk.wait_group.read(NUM_D_TILES - 1)
                                        )
                                        _builder_emit(T.cuda.warpgroup_sync(wg_id + 10))
                                        with T.unroll(EPI_N // 8) as jv:
                                            IRBuilder.name("jv", jv)
                                            c0 = i * EPI_N + jv * 8
                                            r0 = c0 // 2
                                            _builder_emit(
                                                T.ptx.st.shared.v4.u32(
                                                    Dsmem.ptr_to(
                                                        [wg_id, db, warp_id * 32 + lane_id, jv * 8]
                                                    ),
                                                    Dreg_16b[r0],
                                                    Dreg_16b[r0 + 1],
                                                    Dreg_16b[r0 + 2],
                                                    Dreg_16b[r0 + 3],
                                                )
                                            )
                                        _builder_emit(T.cuda.warpgroup_sync(wg_id + 10))
                                        with T.If((warp_id == 0) & (lane_id == 0)):
                                            with T.Then():
                                                _builder_emit(
                                                    T.ptx.fence.proxy.async_.shared__cta()
                                                )
                                                d_m = (
                                                    (m_idx * 2 + cbx) * NUM_CONSUMER + wg_id
                                                ) * BLK_M
                                                d_n = n_idx * MMA_N + i * EPI_N
                                                _builder_emit(
                                                    T.evaluate(
                                                        T.ptx[_TMA_S2G_EVICT_FIRST](
                                                            T.address_of(D_tensor_map),
                                                            T.cast(d_n, "int32"),
                                                            T.cast(d_m, "int32"),
                                                            Dsmem.ptr_to([wg_id, db, 0, 0]),
                                                            T.uint64(_EVICT_FIRST_L2_POLICY),
                                                        )
                                                    )
                                                )
                                        _builder_emit(T.ptx.cp.async_.bulk.commit_group())

                            cur_m = _builder_alloc_scalar("cur_m", "int32")
                            cur_n = _builder_alloc_scalar("cur_n", "int32")
                            with T.While(wb.valid()):
                                cur_m = _builder_assign(
                                    "cur_m", wb.m_idx, locals().get("cur_m", _BUILDER_MISSING)
                                )
                                cur_n = _builder_assign(
                                    "cur_n", wb.n_idx, locals().get("cur_n", _BUILDER_MISSING)
                                )
                                _builder_emit(wb.consume_wg(wg_id, warp_id, lane_id))
                                _builder_emit(wb.advance_coords())
                                cm = cur_m
                                cn = cur_n
                                _builder_emit(writeback(cm, cn))
                                _builder_emit(wb_buf.advance())
                                _builder_emit(wb.mark_done_if_drained())
                            _builder_emit(T.ptx.cp.async_.bulk.wait_group(0))
                            if OVERLAP_EPILOGUE:
                                _builder_emit(T.cuda.warpgroup_sync(wg_id + 10))
                                with T.If((warp_id == 0) & (lane_id == 0)):
                                    with T.Then():
                                        _builder_emit(
                                            tmem_fin.full.arrive(0, remote=1 - cbx, pred=True)
                                        )
                                with T.If(warp_id == 0):
                                    with T.Then():
                                        _builder_emit(tmem_fin.full.wait(0, 0))
            if not OVERLAP_EPILOGUE:
                _builder_emit(T.cuda.cluster_sync())
            with T.If((wg_id == 0) & (warp_id == 0)):
                with T.Then():
                    _builder_emit(T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned())
                    tmem_dealloc_addr = _builder_alloc_scalar("tmem_dealloc_addr", "uint32")
                    _builder_emit(T.ptx.ld.shared.u32(tmem_dealloc_addr, tmem_addr.ptr_to([0])))
                    _builder_emit(
                        T.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](
                            tmem_dealloc_addr, T.uint32(512)
                        )
                    )
    return builder.get()


def tir_kernel(dtype: str, M: int, N: int, K: int):
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}")
    ab_type = _DTYPE_MAP[dtype]
    cfg = GEMM_CONFIGS.get(N, _DEFAULT_CONFIG)
    # Bind only the independent knobs; _kernel derives all geometry from these.
    return _kernel(
        M=M,
        N=N,
        K=K,
        ab_type=ab_type,
        MMA_N=cfg["cta_n"],
        BLK_K=cfg["cta_k"],
        PIPE_DEPTH=cfg["pipe_depth"],
        WB_PIPE_DEPTH=cfg["wb_pipe_depth"],
        L2_GROUP_SIZE=cfg["l2_group_size"],
        OVERLAP_EPILOGUE=cfg["overlap_epilogue"],
    )


KERNEL_META = {"name": "fp16_bf16_gemm", "category": "basic", "compute_capability": 10}
CONFIGS = [
    {"dtype": d, "M": s, "N": s, "K": s, "label": f"{d}_{s}x{s}x{s}"}
    for d in ["fp16", "bf16"]
    for s in [1024, 2048, 4096, 8192, 16384]
]


def get_kernel(dtype, M, N, K, **kwargs):
    return tir_kernel(dtype, M, N, K)


def run_test(dtype, M, N, K, **kwargs):
    """Compile, run, and verify fp16/bf16 GEMM kernel."""
    from tirx_kernels.runner import compile_kernel, cuda_target

    A, B, C = prepare_data(dtype, M, N, K)
    kernel = tir_kernel(dtype, M, N, K)
    C_tvm = torch.zeros_like(C)
    target = cuda_target()
    with target:
        ex = compile_kernel(kernel)
        ex(A, B, C_tvm)
    C_ref = torch.matmul(A, B.T)
    torch.testing.assert_close(C_tvm.cpu(), C_ref.cpu(), rtol=0.001, atol=0.01)


@dataclass(frozen=True)
class PreparedBench:
    """Kernel-owned state produced before GPU assignment."""

    dtype: str
    M: int
    N: int
    K: int
    executable: object


def run_gpu(prepared: PreparedBench, *, warmup=None, repeat=None, timer=None, **kwargs):
    """Allocate inputs/references and run the unchanged GPU timing protocol."""
    A, B, C = prepare_data(prepared.dtype, prepared.M, prepared.N, prepared.K)
    C_tir = torch.zeros_like(C, device="cuda")

    funcs = {"tir": lambda: prepared.executable(A, B, C_tir)}

    def _torch_cublas():
        C_out = torch.zeros_like(C, device="cuda")
        return lambda: torch.matmul(A, B.T, out=C_out)

    references = {"torch-cublas": _torch_cublas}
    if prepared.dtype == "bf16":

        def _deepgemm_cublaslt():
            import deep_gemm

            C_out = torch.zeros(prepared.M, prepared.N, dtype=torch.bfloat16, device="cuda")
            return lambda: deep_gemm.cublaslt_gemm_nt(A, B, C_out, None)

        def _deepgemm_bf16():
            import deep_gemm

            C_out = torch.zeros(prepared.M, prepared.N, dtype=torch.bfloat16, device="cuda")
            return lambda: deep_gemm.bf16_gemm_nt(A, B, C_out)

        references.update(
            {"deepgemm-cublaslt": _deepgemm_cublaslt, "deepgemm-bf16": _deepgemm_bf16}
        )

    return bench(funcs, warmup=warmup, repeat=repeat, timer=timer, references=references, **kwargs)


def prepare_bench(dtype, M, N, K, **kwargs):
    """Specialize and compile the GEMM without initializing CUDA."""
    from tirx_kernels.runner import cuda_initialization_guard, cuda_target, prepared_gpu_benchmark

    with cuda_initialization_guard():
        kernel = tir_kernel(dtype, M, N, K)
        target = cuda_target()
        with target:
            mod = tvm.IRModule({"main": kernel})
            ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
    state = PreparedBench(dtype=dtype, M=M, N=N, K=K, executable=ex)
    return prepared_gpu_benchmark(run_gpu, state)


def run_bench(dtype, M, N, K, warmup=None, repeat=None, timer=None, **kwargs):
    """Benchmark fp16/bf16 GEMM."""
    return prepare_bench(dtype, M, N, K).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, **kwargs
    )
