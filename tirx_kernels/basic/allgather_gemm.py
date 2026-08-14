# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Tensor-parallel AllGather + FP16 GEMM for SM100."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

import tvm
from tvm.ir.type import PointerType, PrimType
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import tirx as T
from tvm.script.ir_builder.base import IRBuilderFrame
from tvm.tirx import IterVar, Layout, is_buffer_var
from tvm.tirx.layout import TCol, TileLayout, TLane
from tvm.tirx.script.builder.ir import name_meta_class_value

from .utils._baselines import create_baseline_suite
from .utils._baselines import ratios as baseline_ratios
from .utils._model_shapes import (
    ALLGATHER_GEMM_MODEL_SHAPES,
    SUPPORTED_WORLD_SIZES,
    make_configs,
    shape_set,
)
from .utils._runtime import (
    DistributedRuntime,
    barrier_on_compute_stream,
    prepare_distributed_bench,
    run_distributed,
    symmetric_empty,
    sync_communication_to_compute,
    sync_compute_to_communication,
    torch_view,
)
from .utils._specialize import load_specialized_module

_BUILDER_MISSING = object()


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


def _builder_bind(name, value, type_annotation=None):
    result = T.Bind(value, type_annotation)
    IRBuilder.name(name, result)
    return result


def _builder_const(name, value):
    return _builder_bind(name, value)


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
    return value


def _builder_assign_many(names, values, previous):
    return tuple(
        _builder_assign(name, value, old) for name, value, old in zip(names, values, previous)
    )


def _builder_setattr(owner, name, value):
    previous = getattr(owner, name, _BUILDER_MISSING)
    if isinstance(previous, T.scalar_wrapper | tvm.tirx.expr.BufferLoad):
        target = previous.scalar if isinstance(previous, T.scalar_wrapper) else previous
        T.buffer_store(target.buffer, value, target.indices)
        return target
    setattr(owner, name, value)
    return getattr(owner, name)


class TaskType(Enum):
    GEMM = 0
    AG = 1


class ProfileEventType(Enum):
    GEMM = 0
    AG = 1
    FETCH = 2


@T.meta_class
class CudaProfiler:
    def __init__(self, profiler_buffer, write_stride, num_groups, profiler_enabled=True):
        self.buffer = profiler_buffer
        self.write_stride = write_stride
        self.num_groups = num_groups
        self.profiler_enabled = (
            T.bool(bool(profiler_enabled))
            if isinstance(profiler_enabled, bool | np.bool_)
            else profiler_enabled
        )
        self.profiler_tag = T.alloc_buffer([1], "uint64", scope="local", align=8)
        self.profiler_write_offset = T.alloc_buffer([1], "uint32", scope="local", align=8)

    def init(self, group_id):
        with T.If(self.profiler_enabled):
            with T.Then():
                T.evaluate(
                    T.cuda.timer_init(
                        self.buffer.data,
                        self.profiler_tag.data,
                        self.profiler_write_offset.data,
                        self.num_groups,
                        group_id,
                    )
                )

    def _leader(self, leader):
        if isinstance(leader, bool | np.bool_):
            return T.bool(bool(leader))
        return T.bool(True) if leader is None else leader

    def start(self, event_type, leader=None):
        with T.If(self.profiler_enabled):
            with T.Then():
                T.evaluate(
                    T.cuda.timer_start(
                        event_type,
                        self.buffer.data,
                        self.profiler_tag.data,
                        self.profiler_write_offset.data,
                        self.write_stride,
                        self._leader(leader),
                    )
                )

    def end(self, event_type, leader=None):
        with T.If(self.profiler_enabled):
            with T.Then():
                T.evaluate(
                    T.cuda.timer_end(
                        event_type,
                        self.buffer.data,
                        self.profiler_tag.data,
                        self.profiler_write_offset.data,
                        self.write_stride,
                        self._leader(leader),
                    )
                )


event_type_names = ["gemm", "ag", "fetch"]
ALLGATHER_HOST_ENTRYPOINT = "runtime.disco.transfer_to_peers_all_gather"
GEMM_DEVICE_ENTRYPOINT = "test_mma_ss_tma_2sm_persistent"

_SPECIALIZATION_M_ENV = "TIRX_INTERNAL_ALLGATHER_GEMM_M"
_SPECIALIZATION_N_ENV = "TIRX_INTERNAL_ALLGATHER_GEMM_N"
_SPECIALIZATION_K_ENV = "TIRX_INTERNAL_ALLGATHER_GEMM_K"
_SPECIALIZATION_WORLD_SIZE_ENV = "TIRX_INTERNAL_ALLGATHER_GEMM_WORLD_SIZE"

# Qwen3-32B TP4 remains the convenient default. Public CONFIGS cover all eight
# model shapes at TP1 and TP4.
M = int(os.environ.get(_SPECIALIZATION_M_ENV, "8192"))
N = int(os.environ.get(_SPECIALIZATION_N_ENV, "51200"))
K = int(os.environ.get(_SPECIALIZATION_K_ENV, "5120"))
WORLD_SIZE = int(os.environ.get(_SPECIALIZATION_WORLD_SIZE_ENV, "4"))
DTYPE = "float16"
_SUPPORTED_SHAPES = shape_set(ALLGATHER_GEMM_MODEL_SHAPES)

M_CLUSTER = 2
N_CLUSTER = 1
WG_NUMBER = 3
WARP_NUMBER = 4
NUM_CONSUMER = 2
NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER
SM_NUMBER = 148

PIPELINE_DEPTH = 4

F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16

d_type, a_type, b_type = DTYPE, DTYPE, DTYPE
LOCAL_M = M // WORLD_SIZE
LOCAL_N = N // WORLD_SIZE
BLK_M, BLK_N, BLK_K = 128, 128, 64
assert LOCAL_M * WORLD_SIZE == M, "M must be divisible by WORLD_SIZE"
assert LOCAL_M % BLK_M == 0, "LOCAL_M must be divisible by BLK_M"
assert LOCAL_N * WORLD_SIZE == N, "N must be divisible by WORLD_SIZE"
assert LOCAL_N % BLK_N == 0, "LOCAL_N must be divisible by BLK_N"

MMA_M, MMA_N, MMA_K = 256, 256, 16
EPI_TILE = 64
SWIZZLE = 3
SMEM_SIZE = (
    PIPELINE_DEPTH * NUM_CONSUMER * BLK_M * BLK_K * F16_BYTES
    + PIPELINE_DEPTH * BLK_N * BLK_K * F16_BYTES
    + NUM_CONSUMER * BLK_M * EPI_TILE * F16_BYTES
    + 1024
)
assert SMEM_SIZE <= 232448

TMEM_LD_SIZE = 64
N_COLS = 512
CTA_GROUP = 2
# TMA instruction spellings emitted by the former ``tma_auto`` calls.
_TMA_G2S_CG2 = (
    "cp.async.bulk.tensor.2d.shared::cluster.global"
    f".mbarrier::complete_tx::bytes.cta_group::{CTA_GROUP}"
)
_TMA_S2G = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
# tcgen05.mma spelling: kind::f16 from the (float32, DTYPE, DTYPE) dtypes.
_MMA_CHAIN = f"tcgen05.mma.cta_group::{CTA_GROUP}.kind::f16"
_MMA_ZERO_MASKS = [0] * (4 if CTA_GROUP == 1 else 8)
_TMEM_LD_64 = "tcgen05.ld.sync.aligned.32x32b.x64.b32"
_CVT_F32X2 = "cvt.rn.f16x2.f32"

PIPE_CYCLE = (K // BLK_K) // PIPELINE_DEPTH
PIPE_REMAIN_NUM = (K // BLK_K) % PIPELINE_DEPTH
assert PIPELINE_DEPTH == 4

assert M % (NUM_CONSUMER * BLK_M * CTA_GROUP) == 0
assert N % (BLK_N * CTA_GROUP) == 0
GEMM_M_CLUSTERS = M // (NUM_CONSUMER * BLK_M * CTA_GROUP)  # gemm tile m: 512
GEMM_N_CLUSTERS = LOCAL_N // (BLK_N * CTA_GROUP)  # gemm tile n: 256
LOCAL_GEMM_M_CLUSTERS = GEMM_M_CLUSTERS // WORLD_SIZE
GROUP_SIZE = LOCAL_GEMM_M_CLUSTERS
assert GROUP_SIZE == LOCAL_GEMM_M_CLUSTERS, (
    "AllGather queue grouping must match the rank-local GEMM M-cluster count"
)

# dyn scheduling
TASK_COUNT = GEMM_M_CLUSTERS * GEMM_N_CLUSTERS
CAPACITY = max(2048, 1 << (TASK_COUNT - 1).bit_length())
assert CAPACITY <= 8192, "AllGather+GEMM queue exceeds the supported capacity"
TASK_IDX_LEN = 2
ENABLE_WARP_BROADCAST = False
C2P_THREAD_COUNT = 12 * 2 if ENABLE_WARP_BROADCAST else NUM_THREADS * 2

# profiling
WARMUP_ITERS = 5
# WARMUP_ITERS = 0
TOTAL_ITERS = 30
# TOTAL_ITERS = 1

PROFILER_ON = False
NUM_GROUPS = 13
PROFILER_BUFFER_SIZE = int(1e7)
PROFILER_WRITE_STRIDE = SM_NUMBER * NUM_GROUPS
CUDA_EVENT_PROFILER = False
if CUDA_EVENT_PROFILER:
    PROFILER_ON = False
VALIDATE = True


@dataclass(frozen=True)
class AllGatherGemmConfig:
    M: int
    N: int
    K: int
    world_size: int
    dtype: str
    local_m: int
    local_n: int
    pipe_cycle: int
    pipe_remainder: int
    group_size: int
    gemm_m_clusters: int
    gemm_n_clusters: int
    local_gemm_m_clusters: int
    task_count: int
    capacity: int


def _mapa_u64_tx(ptr, rank):
    """`mapa.u64` into a declared register, returned as an ordinary value.

    PTX has no defining form, so mapa writes a register the caller declares;
    a one-element local buffer gives both a writable lvalue and an Expr.
    The scratch and the call both go through the full TIRx namespace.
    """
    mapped = T.alloc_local([1], "uint64")
    T.evaluate(T.ptx.mapa.u64(mapped[0], ptr, T.uint32(rank)))
    return mapped[0]


def derive_config(
    M: int = M, N: int = N, K: int = K, world_size: int = WORLD_SIZE, dtype: str = DTYPE
) -> AllGatherGemmConfig:
    """Validate and derive one model-shape/world-size specialization."""

    if (M, N, K) not in _SUPPORTED_SHAPES:
        raise ValueError(f"AllGather+GEMM does not support shape M={M}, N={N}, K={K}")
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"AllGather+GEMM supports world_size in {SUPPORTED_WORLD_SIZES}; got {world_size}"
        )
    if dtype != DTYPE:
        raise ValueError(f"AllGather+GEMM supports only dtype={DTYPE!r}; got {dtype!r}")
    if M % world_size or N % world_size:
        raise ValueError("AllGather+GEMM M and N must be divisible by world_size")
    if K % BLK_K:
        raise ValueError(f"AllGather+GEMM K must be divisible by {BLK_K}")
    if M % (NUM_CONSUMER * BLK_M * CTA_GROUP) or N % (BLK_N * CTA_GROUP):
        raise ValueError("AllGather+GEMM M and N do not satisfy the 2-CTA tile geometry")
    local_m = M // world_size
    local_n = N // world_size
    if local_m % (NUM_CONSUMER * BLK_M * CTA_GROUP) or local_n % (BLK_N * CTA_GROUP):
        raise ValueError("AllGather+GEMM rank-local dimensions do not satisfy tile geometry")
    gemm_m_clusters = M // (NUM_CONSUMER * BLK_M * CTA_GROUP)
    gemm_n_clusters = local_n // (BLK_N * CTA_GROUP)
    local_gemm_m_clusters = gemm_m_clusters // world_size
    task_count = gemm_m_clusters * gemm_n_clusters
    capacity = max(2048, 1 << (task_count - 1).bit_length())
    if capacity > 8192:
        raise AssertionError(
            f"AllGather+GEMM requires queue capacity {capacity}, above the supported 8192"
        )
    return AllGatherGemmConfig(
        M=M,
        N=N,
        K=K,
        world_size=world_size,
        dtype=dtype,
        local_m=local_m,
        local_n=local_n,
        pipe_cycle=(K // BLK_K) // PIPELINE_DEPTH,
        pipe_remainder=(K // BLK_K) % PIPELINE_DEPTH,
        group_size=local_gemm_m_clusters,
        gemm_m_clusters=gemm_m_clusters,
        gemm_n_clusters=gemm_n_clusters,
        local_gemm_m_clusters=local_gemm_m_clusters,
        task_count=task_count,
        capacity=capacity,
    )


semaphore_notify_remote = """
__forceinline__ __device__ uint64_t semaphore_notify_remote(int32_t signal_rank, uint64_t* addr, uint64_t signal_value) {
    auto dst_addr = reinterpret_cast<unsigned long long*>(nvshmem_ptr(addr, signal_rank));
    return atomicAdd_system(dst_addr, signal_value);
}
"""

enqueue_remote = """
__forceinline__ __device__ void enqueue_remote(int32_t* task_types, int32_t* task_idxs, int32_t* tail, int32_t mask,
                                               int32_t signal_rank, int32_t task_type, int32_t task_idx0, int32_t task_idx1) {
    int32_t* remote_task_types = (int32_t*)nvshmem_ptr(task_types, signal_rank);
    int32_t* remote_task_idxs = (int32_t*)nvshmem_ptr(task_idxs, signal_rank);
    int32_t* remote_tail = (int32_t*)nvshmem_ptr(tail, signal_rank);
    int32_t tail_r = atomicAdd(&(remote_tail[0]), 1);
    int32_t masked_pos = tail_r & mask;
    remote_task_types[masked_pos] = task_type;
    remote_task_idxs[masked_pos * 2] = task_idx0;
    remote_task_idxs[masked_pos * 2 + 1] = task_idx1;
    __threadfence();
}
"""


@T.meta_class
class Barriers:
    def __init__(self, shared_buffer_base, shared_buffer_offs, pipe_depth, pipe_width, is_p2c):
        self.mbar: tvm.tir.Buffer = T.decl_buffer(
            (pipe_depth, pipe_width), "uint64", shared_buffer_base, elem_offset=shared_buffer_offs
        )
        self.init_phase = 0 if is_p2c else 1
        self.pipe_depth = pipe_depth
        self.pipe_width = pipe_width

    def init(self, threads_num_wait, initializer):
        with T.If(initializer):
            with T.Then():
                with T.serial(self.pipe_depth) as i:
                    IRBuilder.name("i", i)
                    with T.serial(self.pipe_width) as j:
                        IRBuilder.name("j", j)
                        _builder_emit(
                            T.ptx.mbarrier.init.shared.b64(
                                self.mbar.ptr_to([i, j]), T.uint32(threads_num_wait)
                            )
                        )

    def wait(self, idx_d, idx_w, phase):
        _builder_emit(
            T.cuda.mbarrier_wait(self.mbar.ptr_to([idx_d, idx_w]), self.init_phase ^ phase)
        )


class BarTMA2MMA(Barriers):
    def arrive(self, idx, expected_bytes):
        _builder_emit(
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                self.mbar.ptr_to([idx, 0]), T.uint32(expected_bytes)
            )
        )

    def arrive_only(self, idx):
        _builder_emit(T.ptx.mbarrier.arrive.shared.b64(self.mbar.ptr_to([idx, 0]), T.uint32(1)))


class BarMMA2LD(Barriers):
    def arrive(self, idx):
        _builder_emit(
            T.ptx[
                f"tcgen05.commit.cta_group::{CTA_GROUP}.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
            ](self.mbar.ptr_to([0, idx]), T.uint16(3))
        )


class BarMMA2TMA(Barriers):
    def arrive(self, idx):
        _builder_emit(
            T.ptx[
                f"tcgen05.commit.cta_group::{CTA_GROUP}.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
            ](self.mbar.ptr_to([idx, 0]), T.uint16(3))
        )


class BarLD2MMA(Barriers):
    def arrive(self, idx):
        _rem1 = _builder_assign(
            "_rem1", T.alloc_local([1], "uint64"), locals().get("_rem1", _BUILDER_MISSING)
        )
        _builder_emit(
            T.ptx.mapa.shared__cluster.u64(_rem1[0], self.mbar.ptr_to([0, idx]), T.uint32(0))
        )
        _builder_emit(T.ptx.mbarrier.arrive.b64(_rem1[0], T.uint32(1), pred=T.bool(True)))


@T.meta_class
class Pipeline:
    def __init__(
        self,
        shared_buf,
        base_offset,
        pipeline_depth: int,
        pipeline_num: int,
        p_single_cta: bool = False,
        c_single_cta: bool = False,
    ):
        self.pipeline_depth = pipeline_depth
        self.pipeline_num = pipeline_num
        self.mbar_p2c = T.decl_buffer(
            (pipeline_depth, pipeline_num), "uint64", shared_buf, elem_offset=base_offset
        )
        self.mbar_c2p = T.decl_buffer(
            (pipeline_depth, pipeline_num),
            "uint64",
            shared_buf,
            elem_offset=base_offset + pipeline_depth * pipeline_num,
        )
        self.idx = T.local_scalar("int32")
        self.p2c_phase = T.local_scalar("int32")
        self.c2p_phase = T.local_scalar("int32")
        self.p_single_cta = p_single_cta
        self.c_single_cta = c_single_cta

    def init(self, initializer, p2c_thread_count: int = 1, c2p_thread_count: int = 1):
        _builder_setattr(self, "idx", 0)
        _builder_setattr(self, "p2c_phase", 0)
        _builder_setattr(self, "c2p_phase", 1)
        with T.If(initializer):
            with T.Then():
                with T.thread_binding(M_CLUSTER, "clusterCtaIdx.x") as cbx:
                    IRBuilder.name("cbx", cbx)
                    with T.serial(0, self.pipeline_depth) as i:
                        IRBuilder.name("i", i)
                        with T.serial(0, self.pipeline_num) as j:
                            IRBuilder.name("j", j)
                            _builder_emit(
                                T.ptx.mbarrier.init.shared.b64(
                                    self.mbar_p2c.ptr_to([i, j]), T.uint32(p2c_thread_count)
                                )
                            )
                            with T.If(T.Or(T.bool(False), cbx == 0)):
                                with T.Then():
                                    _builder_emit(
                                        T.ptx.mbarrier.init.shared.b64(
                                            self.mbar_c2p.ptr_to([i, j]), T.uint32(c2p_thread_count)
                                        )
                                    )
        _builder_emit(T.ptx.fence.proxy.async_.shared__cta())

    def advance(self):
        _builder_setattr(self, "idx", (self.idx + 1) % self.pipeline_depth)
        with T.If(self.idx == 0):
            with T.Then():
                _builder_setattr(self, "p2c_phase", self.p2c_phase ^ 1)
                _builder_setattr(self, "c2p_phase", self.c2p_phase ^ 1)

    def producer_wait(self, pipeline_idx):
        with T.thread_binding(M_CLUSTER, "clusterCtaIdx.x") as cbx:
            IRBuilder.name("cbx", cbx)
            with T.If(T.Or(T.bool(False), cbx == 0)):
                with T.Then():
                    _builder_emit(
                        T.cuda.mbarrier_wait(
                            self.mbar_c2p.ptr_to([self.idx, pipeline_idx]), self.c2p_phase
                        )
                    )

    def consumer_wait(self, pipeline_idx):
        with T.thread_binding(M_CLUSTER, "clusterCtaIdx.x") as cbx:
            IRBuilder.name("cbx", cbx)
            _builder_emit(
                T.cuda.mbarrier_wait(self.mbar_p2c.ptr_to([self.idx, pipeline_idx]), self.p2c_phase)
            )


def int_var(name: str, scope="local", dtype="int32", align=4):
    buf = T.alloc_buffer([1], dtype, scope=scope, align=align)
    return buf


@T.meta_class
class Semaphore:
    def __init__(self, cnt, buffer):
        self.cnt = cnt
        self.sem = buffer
        self.state = T.alloc_buffer([1], "uint64", scope="local", align=8)

    def semaphore_wait(self, *coord):
        with T.While(1):
            _builder_emit(
                T.ptx.ld.acquire.gpu.global_.b64(self.state[0], self.sem.ptr_to(list(coord)))
            )
            with T.If(self.state[0] == self.cnt):
                with T.Then():
                    T.evaluate(T.break_loop())
            _builder_emit(T.cuda.nano_sleep(40))


@T.meta_class
class MPMCQueue:
    def __init__(
        self,
        capacity: int,
        task_types: T.Buffer,
        task_idxs: T.Buffer,
        head: T.Buffer,
        tail: T.Buffer,
        num_tot_tasks: int,
    ):
        if capacity & (capacity - 1):
            raise ValueError("capacity must be a power-of-two")
        self.capacity = capacity
        self.mask = capacity - 1
        self.task_types = task_types
        self.task_idxs = task_idxs
        self.head = head
        self.tail = tail
        self.head_r = int_var("head_r")
        self.tail_r = int_var("tail_r")
        self.pos = int_var("pos")
        self.masked_pos = int_var("masked_pos")
        self.num_tot_tasks = num_tot_tasks

    def enqueue(self, signal_rank: int, task_type: int, *task_idx: int):
        _builder_emit(
            T.cuda.func_call(
                "enqueue_remote",
                self.task_types.ptr_to([0]),
                self.task_idxs.ptr_to([0, 0]),
                self.tail.ptr_to([0]),
                self.mask,
                signal_rank,
                task_type,
                *task_idx,
                source_code=enqueue_remote,
            )
        )


class GEMMMPMCQueue(MPMCQueue):
    def dequeue(
        self,
        fetched_task_type: T.Buffer,
        fetched_task_idx0: T.Buffer,
        fetched_task_idx1: T.Buffer,
        sem: Semaphore,
        cbx,
        bx,
        rank,
    ):
        _builder_emit(T.ptx.atom.global_.add.s32(self.head_r[0], self.head.ptr_to([0]), T.int32(1)))
        with T.If(self.head_r[0] < self.num_tot_tasks):
            with T.Then():
                remote_rank = _builder_assign(
                    "remote_rank",
                    (rank + self.head_r[0] // (LOCAL_GEMM_M_CLUSTERS * GEMM_N_CLUSTERS))
                    % WORLD_SIZE,
                    locals().get("remote_rank", _BUILDER_MISSING),
                )
                with T.If(remote_rank != rank):
                    with T.Then():
                        _builder_emit(sem.semaphore_wait(remote_rank))
                T.buffer_store(self.masked_pos, self.head_r[0] & self.mask, [0])
                _builder_emit(
                    T.ptx.ld.global_.acquire.gpu.b32(
                        fetched_task_type[0], self.task_types.ptr_to([self.masked_pos[0]])
                    )
                )
                with T.While(fetched_task_type[0] < 0):
                    _builder_emit(T.cuda.nano_sleep(40))
                    _builder_emit(
                        T.ptx.ld.global_.acquire.gpu.b32(
                            fetched_task_type[0], self.task_types.ptr_to([self.masked_pos[0]])
                        )
                    )
                _builder_emit(
                    T.ptx.st.global_.s32(self.task_types.ptr_to([self.masked_pos[0]]), T.int32(-1))
                )
                _builder_emit(
                    T.ptx.ld.global_.s32(
                        fetched_task_idx0[0], self.task_idxs.ptr_to([self.masked_pos[0], 0])
                    )
                )
                _builder_emit(
                    T.ptx.ld.global_.s32(
                        fetched_task_idx1[0], self.task_idxs.ptr_to([self.masked_pos[0], 1])
                    )
                )
            with T.Else():
                T.buffer_store(fetched_task_type, -1, [0])


# fmt: off
def consumer_fetch(
    sch_pipe,
    packed_value,
    rs_rem,
    fetched_task_type,
    fetched_task_idx0,
    fetched_task_idx1,
):
    _builder_emit(sch_pipe.consumer_wait(0))
    _builder_emit(T.ptx.ld.shared__cluster.v4.b32(rs_rem[0], fetched_task_type[0], fetched_task_idx0[0], fetched_task_idx1[0], packed_value.ptr_to([0])))
    _rem2 = _builder_assign("_rem2", T.alloc_local([1], 'uint64'), locals().get("_rem2", _BUILDER_MISSING))
    _builder_emit(T.ptx.mapa.shared__cluster.u64(_rem2[0], sch_pipe.mbar_c2p.ptr_to([sch_pipe.idx, 0]), T.uint32(0)))
    _builder_emit(T.ptx.mbarrier.arrive.b64(_rem2[0], T.uint32(1), pred=T.bool(True)))
    _builder_setattr(sch_pipe, "p2c_phase", sch_pipe.p2c_phase ^ 1)
# fmt: on


@T.meta_class
class SingleDynamicTileScheduler:
    def __init__(
        self, queue: MPMCQueue, packed_value: T.Buffer, sch_pipe: Pipeline, sem: Semaphore
    ):
        self.queue = queue
        self.sch_pipe = sch_pipe
        self.fetched_task_type = int_var("fetched_task_type")
        self.fetched_task_idx0 = int_var("fetched_task_idx0")
        self.fetched_task_idx1 = int_var("fetched_task_idx1")
        self.sem = sem
        self.rs_rem = int_var("rs_rem")
        self.packed_value = packed_value
        IRBuilder.current().name("packed_value", self.packed_value)

    # fmt: off
    def _fetch_from_queue(
        self,
        cbx,
        bx,
        rank,
        warp_id_in_cta,
        lane_id,
    ):
        # fetch from GEMM queue
        with T.If(T.And(warp_id_in_cta == 11, lane_id == 0)):
            with T.Then():
                with T.If(cbx == 0):
                    with T.Then():
                        _builder_emit(self.sch_pipe.producer_wait(0))
                        _builder_emit(self.queue.dequeue(self.fetched_task_type, self.fetched_task_idx0, self.fetched_task_idx1, self.sem, cbx, bx, rank))
                        _builder_emit(T.ptx.st.shared__cluster.v4.b32(self.packed_value.ptr_to([0]), self.rs_rem[0], self.fetched_task_type[0], self.fetched_task_idx0[0], self.fetched_task_idx1[0]))
                        _rem3 = _builder_assign("_rem3", T.alloc_local([1], 'uint64'), locals().get("_rem3", _BUILDER_MISSING))
                        _builder_emit(T.ptx.mapa.shared__cluster.u64(_rem3[0], self.sch_pipe.mbar_p2c.ptr_to([self.sch_pipe.idx, 0]), T.uint32(0)))
                        _builder_emit(T.ptx.mbarrier.arrive.b64(_rem3[0], T.uint32(1), pred=T.bool(True)))
                        _rem4 = _builder_assign("_rem4", T.alloc_local([1], 'uint64'), locals().get("_rem4", _BUILDER_MISSING))
                        _builder_emit(T.ptx.mapa.shared__cluster.u64(_rem4[0], self.sch_pipe.mbar_p2c.ptr_to([self.sch_pipe.idx, 0]), T.uint32(1)))
                        _builder_emit(T.ptx.mbarrier.arrive.b64(_rem4[0], T.uint32(1), pred=T.bool(True)))
                        _builder_setattr(self.sch_pipe, "c2p_phase", self.sch_pipe.c2p_phase ^ 1)
        if ENABLE_WARP_BROADCAST:
            with T.If(lane_id == 0):
                with T.Then():
                    _builder_emit(consumer_fetch(self.sch_pipe, self.packed_value, self.rs_rem, self.fetched_task_type, self.fetched_task_idx0, self.fetched_task_idx1))
            _builder_emit(T.ptx.shfl_sync.idx.b32(self.rs_rem[0], self.rs_rem[0], T.uint32(0), T.uint32(31), T.uint32(4294967295)))
            _builder_emit(T.ptx.shfl_sync.idx.b32(self.fetched_task_type[0], self.fetched_task_type[0], T.uint32(0), T.uint32(31), T.uint32(4294967295)))
            _builder_emit(T.ptx.shfl_sync.idx.b32(self.fetched_task_idx0[0], self.fetched_task_idx0[0], T.uint32(0), T.uint32(31), T.uint32(4294967295)))
            _builder_emit(T.ptx.shfl_sync.idx.b32(self.fetched_task_idx1[0], self.fetched_task_idx1[0], T.uint32(0), T.uint32(31), T.uint32(4294967295)))
        else:
            _builder_emit(consumer_fetch(self.sch_pipe, self.packed_value, self.rs_rem, self.fetched_task_type, self.fetched_task_idx0, self.fetched_task_idx1))

    def init(self, cbx, bx, rank, warp_id_in_cta, lane_id):
        T.buffer_store(self.rs_rem, -1, [0])
        _builder_emit(self._fetch_from_queue(cbx, bx, rank, warp_id_in_cta, lane_id))

    def next_tile(self, cbx, bx, rank, warp_id_in_cta, lane_id):
        _builder_emit(self._fetch_from_queue(cbx, bx, rank, warp_id_in_cta, lane_id))

    def valid(self):
        return (self.fetched_task_type[0] >= 0) | (self.rs_rem[0] >= 0)
    # fmt: on


def skip():
    pass


def _build_kernel():
    A_layout = T.ComposeLayout(
        3,
        3,
        3,
        T.TileLayout(
            T.S[
                (PIPELINE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K) : (
                    NUM_CONSUMER * BLK_M * BLK_K,
                    BLK_M * BLK_K,
                    BLK_K,
                    1,
                )
            ]
        ),
    )
    B_layout = T.ComposeLayout(
        3, 3, 3, T.TileLayout(T.S[(PIPELINE_DEPTH, BLK_N, BLK_K) : (BLK_N * BLK_K, BLK_K, 1)])
    )
    D_layout = T.ComposeLayout(
        3,
        3,
        3,
        T.TileLayout(T.S[(NUM_CONSUMER, BLK_M, EPI_TILE) : (BLK_M * EPI_TILE, EPI_TILE, 1)]),
    )

    # fmt: off
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("test_mma_ss_tma_2sm_persistent")
            A = T.arg("A", T.Buffer((LOCAL_M, K), a_type))
            B = T.arg("B", T.Buffer((LOCAL_N, K), b_type))
            ag_out = T.arg("ag_out", T.Buffer((M, K), a_type))
            semaphore = T.arg("semaphore", T.Buffer((WORLD_SIZE,), 'uint64'))
            out = T.arg("out", T.Buffer((M, LOCAL_N), d_type))
            profiler_buffer = T.arg("profiler_buffer", T.Buffer((PROFILER_BUFFER_SIZE,), 'uint64'))
            gemm_task_types = T.arg("gemm_task_types", T.Buffer((CAPACITY,), 'int32'))
            gemm_task_idxs = T.arg("gemm_task_idxs", T.Buffer((CAPACITY, 2), 'int32'))
            gemm_head = T.arg("gemm_head", T.Buffer((1,), 'int32'))
            gemm_tail = T.arg("gemm_tail", T.Buffer((1,), 'int32'))
            A_tensor_map = _builder_bind("A_tensor_map", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            A_tensor_map_1 = _builder_bind("A_tensor_map_1", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            ag_out_tensor_map = _builder_bind("ag_out_tensor_map", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            ag_out_tensor_map_1 = _builder_bind("ag_out_tensor_map_1", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            B_tensor_map = _builder_bind("B_tensor_map", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            out_tensor_map = _builder_bind("out_tensor_map", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', A_tensor_map, a_type, 2, A.data, K, LOCAL_M, K * F16_BYTES, BLK_K, BLK_M, 1, 1, 0, SWIZZLE, 2, 0))
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', A_tensor_map_1, a_type, 2, A.data, K, LOCAL_M, K * F16_BYTES, BLK_K, BLK_M, 1, 1, 0, SWIZZLE, 2, 0))
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', ag_out_tensor_map, a_type, 2, ag_out.data, K, M, K * F16_BYTES, BLK_K, BLK_M, 1, 1, 0, SWIZZLE, 2, 0))
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', ag_out_tensor_map_1, a_type, 2, ag_out.data, K, M, K * F16_BYTES, BLK_K, BLK_M, 1, 1, 0, SWIZZLE, 2, 0))
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', B_tensor_map, b_type, 2, B.data, K, LOCAL_N, K * F16_BYTES, BLK_K, BLK_N, 1, 1, 0, SWIZZLE, 2, 0))
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', out_tensor_map, d_type, 2, out.data, LOCAL_N, M, LOCAL_N * F16_BYTES, EPI_TILE, BLK_M, 1, 1, 0, SWIZZLE, 2, 0))
            _builder_emit(T.device_entry())
            cbx, cby = _builder_assign_many(('cbx', 'cby'), T.cta_id_in_cluster([M_CLUSTER, N_CLUSTER]), (locals().get("cbx", _BUILDER_MISSING), locals().get("cby", _BUILDER_MISSING),))
            bx = _builder_assign("bx", T.cta_id([SM_NUMBER]), locals().get("bx", _BUILDER_MISSING))
            wg_id = _builder_assign("wg_id", T.warpgroup_id([WG_NUMBER]), locals().get("wg_id", _BUILDER_MISSING))
            warp_id = _builder_assign("warp_id", T.warp_id_in_wg([WARP_NUMBER]), locals().get("warp_id", _BUILDER_MISSING))
            warp_id_in_cta = _builder_assign("warp_id_in_cta", T.warp_id([WG_NUMBER * WARP_NUMBER]), locals().get("warp_id_in_cta", _BUILDER_MISSING))
            lane_id = _builder_assign("lane_id", T.lane_id([32]), locals().get("lane_id", _BUILDER_MISSING))
            tid = _builder_assign("tid", T.thread_id([NUM_THREADS]), locals().get("tid", _BUILDER_MISSING))
            rank = _builder_scalar("rank", T.nvshmem.my_pe(), "int32")
            buf = _builder_assign("buf", T.alloc_buffer([SMEM_SIZE], 'uint8', scope='shared.dyn'), locals().get("buf", _BUILDER_MISSING))
            _builder_emit(T.attr({'tirx.dyn_smem_bytes': SMEM_SIZE}))
            tmem_addr = _builder_assign("tmem_addr", T.decl_buffer((1,), 'uint32', buf.data, scope='shared.dyn', elem_offset=0), locals().get("tmem_addr", _BUILDER_MISSING))
            A_smem = _builder_assign("A_smem", T.decl_buffer((PIPELINE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, buf.data, layout=A_layout, elem_offset=1024 // F16_BYTES), locals().get("A_smem", _BUILDER_MISSING))
            B_smem = _builder_assign("B_smem", T.decl_buffer((PIPELINE_DEPTH, BLK_N, BLK_K), b_type, buf.data, layout=B_layout, elem_offset=1024 // F16_BYTES + PIPELINE_DEPTH * NUM_CONSUMER * BLK_M * BLK_K), locals().get("B_smem", _BUILDER_MISSING))
            D_smem = _builder_assign("D_smem", T.decl_buffer((NUM_CONSUMER, BLK_M, EPI_TILE), d_type, buf.data, layout=D_layout, elem_offset=1024 // F16_BYTES + PIPELINE_DEPTH * (NUM_CONSUMER * BLK_M + BLK_N) * BLK_K), locals().get("D_smem", _BUILDER_MISSING))
            descA = _builder_alloc_scalar("descA", "uint64")
            descB = _builder_alloc_scalar("descB", "uint64")
            descI = _builder_alloc_scalar("descI", "uint32")
            phase = _builder_assign("phase", T.alloc_buffer((1,), 'int32', scope='local'), locals().get("phase", _BUILDER_MISSING))
            phase_tmem = _builder_assign("phase_tmem", T.alloc_buffer((1,), 'int32', scope='local'), locals().get("phase_tmem", _BUILDER_MISSING))
            stage = _builder_alloc_scalar("stage", "int32")
            tmem_addr_local = _builder_alloc_scalar("tmem_addr_local", "uint32")
            sem = Semaphore(cnt=1, buffer=semaphore)
            gemm_queue = GEMMMPMCQueue(CAPACITY, gemm_task_types, gemm_task_idxs, gemm_head, gemm_tail, GEMM_M_CLUSTERS * GEMM_N_CLUSTERS)
            packed_buf = _builder_assign("packed_buf", T.decl_buffer((1,), 'uint64', buf.data, elem_offset=64), locals().get("packed_buf", _BUILDER_MISSING))
            packed_ptr = _builder_bind("packed_ptr", T.reinterpret(PointerType(PrimType('uint64')), _mapa_u64_tx(packed_buf.ptr_to([0]), 0)), T.Var(name='packed_ptr', ty=PointerType(PrimType('uint64'))))
            packed_value = _builder_assign("packed_value", T.decl_buffer([1], 'uint64', data=packed_ptr, scope='shared'), locals().get("packed_value", _BUILDER_MISSING))
            sch_pipe = Pipeline(buf.data, 64 + 4, pipeline_depth=1, pipeline_num=1, p_single_cta=True, c_single_cta=False)
            tile_scheduler = SingleDynamicTileScheduler(gemm_queue, packed_value, sch_pipe, sem)
            profiler = CudaProfiler(profiler_buffer, write_stride=PROFILER_WRITE_STRIDE, num_groups=NUM_GROUPS, profiler_enabled=PROFILER_ON)
            _builder_emit(profiler.init(warp_id_in_cta))
            tma2mma = BarTMA2MMA(buf.data, 4, PIPELINE_DEPTH, 1, is_p2c=True)
            mma2tma = BarMMA2TMA(buf.data, 4 + PIPELINE_DEPTH, PIPELINE_DEPTH, 1, is_p2c=False)
            mma2ld = BarMMA2LD(buf.data, 4 + 2 * PIPELINE_DEPTH, 1, NUM_CONSUMER, is_p2c=True)
            ld2mma = BarLD2MMA(buf.data, 4 + 2 * PIPELINE_DEPTH + NUM_CONSUMER, 1, NUM_CONSUMER, is_p2c=False)
            _builder_emit(tma2mma.init(1, tid == 0))
            _builder_emit(mma2tma.init(NUM_CONSUMER, tid == 0))
            _builder_emit(mma2ld.init(1, tid == 0))
            _builder_emit(ld2mma.init(128 * NUM_CONSUMER, tid == 0))
            ptr = _builder_bind("ptr", T.reinterpret(PointerType(PrimType('uint64')), _mapa_u64_tx(tma2mma.mbar.ptr_to([0, 0]), 0)), T.Var(name='ptr', ty=PointerType(PrimType('uint64'))))
            tma_finished = _builder_assign("tma_finished", T.decl_buffer([PIPELINE_DEPTH], 'uint64', data=ptr, scope='shared'), locals().get("tma_finished", _BUILDER_MISSING))
            T.buffer_store(phase, 0, [0])
            T.buffer_store(phase_tmem, 0, [0])
            _builder_emit(sch_pipe.init(tid == 0, c2p_thread_count=C2P_THREAD_COUNT, p2c_thread_count=1))
            _builder_emit(T.cuda.tcgen05.encode_instr_descriptor(T.address_of(descI), d_dtype='float32', a_dtype=a_type, b_dtype=b_type, M=MMA_M, N=MMA_N, K=MMA_K, trans_a=False, trans_b=False, n_cta_groups=CTA_GROUP))
            with T.If((wg_id == 0) & (warp_id == 0)):
                with T.Then():
                    _builder_emit(T.ptx[f'tcgen05.alloc.cta_group::{CTA_GROUP}.sync.aligned.shared::cta.b32'](tmem_addr.ptr_to([0]), T.uint32(N_COLS)))
            _builder_emit(T.ptx.barrier.cluster.arrive())
            _builder_emit(T.ptx.barrier.cluster.wait())
            _builder_emit(T.cuda.cta_sync())
            _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
            _builder_emit(T.ptx.fence.mbarrier_init.release.cluster())
            _builder_emit(tile_scheduler.init(cbx, bx, rank, warp_id_in_cta, lane_id))
            _builder_emit(T.ptx.ld.shared.u32(tmem_addr_local, tmem_addr.ptr_to([0])))
            _builder_emit(T.cuda.trap_when_assert_failed(tmem_addr_local == 0))
            tmem = _builder_assign("tmem", T.decl_buffer((128, N_COLS), 'float32', scope='tmem', allocated_addr=0, layout=TileLayout(T.S[(128, N_COLS):(1 @ TLane, 1 @ TCol)])), locals().get("tmem", _BUILDER_MISSING))

            def paritioned_loop(main_loop, epilogue1, epilogue2):
                with T.serial(PIPE_CYCLE) as ko:
                    IRBuilder.name("ko", ko)
                    with T.unroll(PIPELINE_DEPTH) as ks:
                        IRBuilder.name("ks", ks)
                        T.buffer_store(stage.buffer, ko * PIPELINE_DEPTH + ks, [0])
                        _builder_emit(main_loop(False, ks))
                    T.buffer_store(phase, phase[0] ^ 1, [0])
                if PIPE_REMAIN_NUM > 0:
                    with T.unroll(PIPE_REMAIN_NUM) as ks:
                        IRBuilder.name("ks", ks)
                        T.buffer_store(stage.buffer, PIPE_CYCLE * PIPELINE_DEPTH + ks, [0])
                        _builder_emit(main_loop(True, ks))
                    _builder_emit(epilogue1())
                    with T.unroll(PIPE_REMAIN_NUM, PIPELINE_DEPTH) as ks:
                        IRBuilder.name("ks", ks)
                        _builder_emit(epilogue2(ks))
                    T.buffer_store(phase, phase[0] ^ 1, [0])
                else:
                    _builder_emit(epilogue1())
            with T.While(tile_scheduler.valid()):
                with T.If(tile_scheduler.fetched_task_type[0] == TaskType.GEMM.value):
                    with T.Then():
                        _builder_emit(profiler.start(ProfileEventType.GEMM, tid == 0))
                        m_idx = tile_scheduler.fetched_task_idx0[0]
                        n_idx = tile_scheduler.fetched_task_idx1[0]
                        with T.If(wg_id == NUM_CONSUMER):
                            with T.Then():
                                _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(56))
                                with T.If(warp_id == 3):
                                    with T.Then():
                                        with T.If(T.cuda.elect_sync()):
                                            with T.Then():
                                                n_start = (n_idx * CTA_GROUP + cbx) * BLK_N

                                                def tma_load(is_remain, ks):
                                                    stage_k = stage * BLK_K
                                                    _builder_emit(mma2tma.wait(ks, 0, phase[0]))
                                                    with T.If(T.And(rank * LOCAL_GEMM_M_CLUSTERS <= m_idx, m_idx < (rank + 1) * LOCAL_GEMM_M_CLUSTERS)):
                                                        with T.Then():
                                                            m_start0 = (m_idx % LOCAL_GEMM_M_CLUSTERS * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M
                                                            m_start1 = (m_idx % LOCAL_GEMM_M_CLUSTERS * NUM_CONSUMER * CTA_GROUP + CTA_GROUP + cbx) * BLK_M
                                                            _builder_emit(T.ptx[_TMA_G2S_CG2](A_smem.ptr_to([ks, 0, 0, 0]), T.address_of(A_tensor_map), T.cast(stage_k, 'int32'), T.cast(m_start0, 'int32'), T.cuda.cvta_generic_to_shared(tma_finished.ptr_to([ks]))))
                                                            _builder_emit(T.ptx[_TMA_G2S_CG2](A_smem.ptr_to([ks, 1, 0, 0]), T.address_of(A_tensor_map_1), T.cast(stage_k, 'int32'), T.cast(m_start1, 'int32'), T.cuda.cvta_generic_to_shared(tma_finished.ptr_to([ks]))))
                                                        with T.Else():
                                                            m_start0 = (m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M
                                                            m_start1 = (m_idx * NUM_CONSUMER * CTA_GROUP + CTA_GROUP + cbx) * BLK_M
                                                            _builder_emit(T.ptx[_TMA_G2S_CG2](A_smem.ptr_to([ks, 0, 0, 0]), T.address_of(ag_out_tensor_map), T.cast(stage_k, 'int32'), T.cast(m_start0, 'int32'), T.cuda.cvta_generic_to_shared(tma_finished.ptr_to([ks]))))
                                                            _builder_emit(T.ptx[_TMA_G2S_CG2](A_smem.ptr_to([ks, 1, 0, 0]), T.address_of(ag_out_tensor_map_1), T.cast(stage_k, 'int32'), T.cast(m_start1, 'int32'), T.cuda.cvta_generic_to_shared(tma_finished.ptr_to([ks]))))
                                                    _builder_emit(T.ptx[_TMA_G2S_CG2](B_smem.ptr_to([ks, 0, 0]), T.address_of(B_tensor_map), T.cast(stage_k, 'int32'), T.cast(n_start, 'int32'), T.cuda.cvta_generic_to_shared(tma_finished.ptr_to([ks]))))
                                                    with T.If(cbx == 0):
                                                        with T.Then():
                                                            _builder_emit(tma2mma.arrive(ks, NUM_CONSUMER * BLK_K * (BLK_M * NUM_CONSUMER + BLK_N) * F16_BYTES))

                                                def tma_load_epilogue(ks):
                                                    _builder_emit(mma2tma.wait(ks, 0, phase[0]))
                                                    with T.If(cbx == 0):
                                                        with T.Then():
                                                            _builder_emit(tma2mma.arrive_only(ks))
                                                _builder_emit(paritioned_loop(tma_load, skip, tma_load_epilogue))
                                    with T.Else():
                                        with T.If(T.And(warp_id < 2, cbx == 0)):
                                            with T.Then():
                                                with T.If(T.cuda.elect_sync()):
                                                    with T.Then():
                                                        _builder_emit(ld2mma.wait(0, warp_id, phase_tmem[0]))
                                                        _builder_emit(T.ptx.tcgen05.fence__after_thread_sync())

                                                        def mma(is_remain, ks):
                                                            _builder_emit(tma2mma.wait(ks, 0, phase[0]))
                                                            with T.unroll(BLK_K // MMA_K) as ki:
                                                                IRBuilder.name("ki", ki)
                                                                _builder_emit(T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(descA), A_smem.ptr_to([ks, warp_id, 0, ki * MMA_K]), ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE))
                                                                _builder_emit(T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(descB), B_smem.ptr_to([ks, 0, ki * MMA_K]), ldo=1, sdo=8 * BLK_K * F16_BYTES // F128_BYTES, swizzle=SWIZZLE))
                                                                with T.If(T.And(T.And(stage == 0, ki == 0), T.bool(True))):
                                                                    with T.Then():
                                                                        _builder_emit(T.ptx[_MMA_CHAIN](T.cast(warp_id * MMA_N, 'uint32'), descA, descB, descI, *_MMA_ZERO_MASKS, False))
                                                                    with T.Else():
                                                                        _builder_emit(T.ptx[_MMA_CHAIN](T.cast(warp_id * MMA_N, 'uint32'), descA, descB, descI, *_MMA_ZERO_MASKS, True))
                                                            _builder_emit(mma2tma.arrive(ks))

                                                        def mma_epilogue1():
                                                            _builder_emit(mma2ld.arrive(warp_id))

                                                        def mma_epilogue2(ks):
                                                            _builder_emit(tma2mma.wait(ks, 0, phase[0]))
                                                            _builder_emit(mma2tma.arrive(ks))
                                                        _builder_emit(paritioned_loop(mma, mma_epilogue1, mma_epilogue2))
                                                        T.buffer_store(phase_tmem, phase_tmem[0] ^ 1, [0])
                        with T.If(wg_id < NUM_CONSUMER):
                            with T.Then():
                                _builder_emit(T.ptx.setmaxnreg.inc.sync.aligned.u32(224))
                                reg = _builder_assign("reg", T.alloc_buffer((TMEM_LD_SIZE,), 'float32', scope='local'), locals().get("reg", _BUILDER_MISSING))
                                reg_fp16 = _builder_assign("reg_fp16", T.alloc_buffer((TMEM_LD_SIZE // 2,), 'uint32', scope='local', align=16), locals().get("reg_fp16", _BUILDER_MISSING))
                                _builder_emit(mma2ld.wait(0, wg_id, phase_tmem[0]))
                                T.buffer_store(phase_tmem, phase_tmem[0] ^ 1, [0])
                                _builder_emit(T.ptx.tcgen05.fence__after_thread_sync())
                                with T.unroll(MMA_N // TMEM_LD_SIZE) as i:
                                    IRBuilder.name("i", i)
                                    col_st = wg_id * MMA_N + i * TMEM_LD_SIZE
                                    _builder_emit(T.ptx[_TMEM_LD_64](*[reg[j] for j in range(TMEM_LD_SIZE)], T.cast(col_st, 'uint32')))
                                    _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                                    with T.If(i == MMA_N // TMEM_LD_SIZE - 1):
                                        with T.Then():
                                            _builder_emit(ld2mma.arrive(wg_id))
                                    with T.unroll(TMEM_LD_SIZE // 2) as j:
                                        IRBuilder.name("j", j)
                                        _builder_emit(T.ptx[_CVT_F32X2](reg_fp16[j], reg[j * 2 + 1], reg[j * 2]))
                                    with T.If(i > 0):
                                        with T.Then():
                                            with T.If(T.And(lane_id == 0, warp_id == 0)):
                                                with T.Then():
                                                    _builder_emit(T.ptx.cp.async_.bulk.wait_group(0))
                                            _builder_emit(T.cuda.warpgroup_sync(wg_id))
                                    with T.unroll(EPI_TILE // 8) as jv:
                                        IRBuilder.name("jv", jv)
                                        r0 = jv * 4
                                        _builder_emit(T.ptx.st.shared.v4.u32(D_smem.ptr_to([wg_id, warp_id * 32 + lane_id, jv * 8]), reg_fp16[r0], reg_fp16[r0 + 1], reg_fp16[r0 + 2], reg_fp16[r0 + 3]))
                                    _builder_emit(T.cuda.warpgroup_sync(wg_id))
                                    _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                    with T.If(T.And(lane_id == 0, warp_id == 0)):
                                        with T.Then():
                                            m_st = (m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M
                                            n_st = n_idx * BLK_N * CTA_GROUP + i * EPI_TILE
                                            _builder_emit(T.ptx[_TMA_S2G](T.address_of(out_tensor_map), T.cast(n_st, 'int32'), T.cast(m_st, 'int32'), D_smem.ptr_to([wg_id, 0, 0])))
                                            _builder_emit(T.ptx.cp.async_.bulk.commit_group())
                                with T.If(T.And(lane_id == 0, warp_id == 0)):
                                    with T.Then():
                                        _builder_emit(T.ptx.cp.async_.bulk.wait_group(0))
                                _builder_emit(T.cuda.warpgroup_sync(wg_id))
                        _builder_emit(profiler.end(ProfileEventType.GEMM, tid == 0))
                _builder_emit(tile_scheduler.next_tile(cbx, bx, rank, warp_id_in_cta, lane_id))
            _builder_emit(T.ptx.barrier.cluster.arrive())
            _builder_emit(T.ptx.barrier.cluster.wait())
            with T.If((wg_id == 0) & (warp_id == 0)):
                with T.Then():
                    _builder_emit(T.ptx[f'tcgen05.relinquish_alloc_permit.cta_group::{CTA_GROUP}.sync.aligned']())
                    _builder_emit(T.ptx.ld.shared.u32(tmem_addr_local, tmem_addr.ptr_to([0])))
                    _builder_emit(T.ptx[f'tcgen05.dealloc.cta_group::{CTA_GROUP}.sync.aligned.b32'](tmem_addr_local, T.uint32(N_COLS)))
    return builder.get()

    # fmt: on

    return test_mma_ss_tma_2sm_persistent


KERNEL_META = {"name": "allgather_gemm", "category": "basic", "compute_capability": 10}

CONFIGS = make_configs(ALLGATHER_GEMM_MODEL_SHAPES)


def _check_config(M_: int, N_: int, K_: int, world_size: int, dtype: str) -> AllGatherGemmConfig:
    return derive_config(M_, N_, K_, world_size, dtype)


def _check_scheduler(scheduler: str) -> None:
    if scheduler != "dynamic":
        raise ValueError(f"AllGather+GEMM supports only scheduler='dynamic'; got {scheduler!r}")


def get_kernel(
    M: int = M,
    N: int = N,
    K: int = K,
    world_size: int = WORLD_SIZE,
    dtype: str = "float16",
    scheduler: str = "dynamic",
    **_kwargs: Any,
):
    config = _check_config(M, N, K, world_size, dtype)
    _check_scheduler(scheduler)
    requested = (config.M, config.N, config.K, config.world_size)
    active = (globals()["M"], globals()["N"], globals()["K"], WORLD_SIZE)
    if requested != active:
        specialized = load_specialized_module(
            package=__package__,
            stem="allgather_gemm",
            source=__file__,
            key=requested,
            environment={
                _SPECIALIZATION_M_ENV: config.M,
                _SPECIALIZATION_N_ENV: config.N,
                _SPECIALIZATION_K_ENV: config.K,
                _SPECIALIZATION_WORLD_SIZE_ENV: config.world_size,
            },
        )
        return specialized.get_kernel()
    return _build_kernel()


def _get_benchmark_kernel(
    M: int = M,
    N: int = N,
    K: int = K,
    world_size: int = WORLD_SIZE,
    dtype: str = "float16",
    scheduler: str = "dynamic",
):
    return get_kernel(M, N, K, world_size, dtype, scheduler=scheduler)


def prepare_data(
    M: int = M,
    N: int = N,
    K: int = K,
    world_size: int = WORLD_SIZE,
    dtype: str = "float16",
    seed: int = 42,
    scale: float = 0.05,
    rank: int = 0,
    scheduler: str = "dynamic",
    **_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Create deterministic inputs directly on one rank's CUDA device."""

    config = _check_config(M, N, K, world_size, dtype)
    _check_scheduler(scheduler)
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    device = torch.device("cuda", rank)
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    A = torch.randn(
        (config.local_m, config.K), dtype=torch.float16, device=device, generator=generator
    ).mul_(scale)
    B = torch.randn(
        (config.local_n, config.K), dtype=torch.float16, device=device, generator=generator
    ).mul_(scale)
    return {"A": A, "B": B}


def _queue_state(
    config: AllGatherGemmConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = config or derive_config()
    task_types = np.full((config.world_size, config.capacity), -1, dtype=np.int32)
    task_idxs = np.zeros((config.world_size, config.capacity, TASK_IDX_LEN), dtype=np.int32)
    heads = np.zeros((config.world_size, 1), dtype=np.int32)
    tails = np.zeros((config.world_size, 1), dtype=np.int32)

    for rank in range(config.world_size):
        tasks: list[tuple[int, int]] = []
        offset = rank * config.local_gemm_m_clusters
        group_count = math.ceil(config.gemm_m_clusters / config.group_size)
        for group in range(group_count):
            begin = group * config.group_size
            end = min((group + 1) * config.group_size, config.gemm_m_clusters)
            for n_idx in range(config.gemm_n_clusters):
                for m_idx in range(begin, end):
                    tasks.append(((offset + m_idx) % config.gemm_m_clusters, n_idx))
        if len(tasks) != config.task_count:
            raise AssertionError("incomplete AllGather+GEMM queue")
        if len(tasks) > config.capacity:
            raise AssertionError("AllGather+GEMM queue exceeds capacity")
        task_types[rank, : len(tasks)] = TaskType.GEMM.value
        task_idxs[rank, : len(tasks)] = tasks
        tails[rank, 0] = len(tasks)
    return task_types, task_idxs, heads, tails


@dataclass
class _Case:
    runtime: DistributedRuntime
    module: Any
    config: AllGatherGemmConfig
    A: Any
    B: Any
    ag_out: Any
    semaphore: Any
    out: Any
    profiler: Any
    task_types: Any
    task_idxs: Any
    head: Any
    tail: Any
    initial_task_types: Any
    initial_task_idxs: Any
    initial_tail: Any
    ag_out_torch: torch.Tensor
    semaphore_torch: torch.Tensor

    def reset(self) -> None:
        self.semaphore_torch.zero_()
        self.task_types.copy_(self.initial_task_types)
        self.task_idxs.copy_(self.initial_task_idxs)
        self.head.zero_()
        self.tail.copy_(self.initial_tail)

    def prepare(self) -> None:
        barrier_on_compute_stream(self.runtime)
        sync_compute_to_communication(self.runtime)

    def launch(self) -> None:
        tvm.get_global_func(ALLGATHER_HOST_ENTRYPOINT)(
            self.semaphore,
            self.A,
            self.ag_out,
            self.runtime.communication_stream,
            self.config.M,
            self.config.K,
            self.config.world_size,
        )
        self.module[GEMM_DEVICE_ENTRYPOINT](
            self.A,
            self.B,
            self.ag_out,
            self.semaphore,
            self.out,
            self.profiler,
            self.task_types,
            self.task_idxs,
            self.head,
            self.tail,
        )
        sync_communication_to_compute(self.runtime)


def _allocate_case(
    runtime: DistributedRuntime,
    module: Any,
    data: dict[str, torch.Tensor],
    config: AllGatherGemmConfig,
) -> _Case:
    task_types, task_idxs, heads, tails = _queue_state(config)
    device = torch.device("cuda", runtime.device_index)
    ag_out = symmetric_empty(runtime, (config.M, config.K), a_type)
    semaphore = symmetric_empty(runtime, (config.world_size,), "uint64")
    initial_task_types = torch.from_numpy(task_types[runtime.rank].copy()).to(device)
    initial_task_idxs = torch.from_numpy(task_idxs[runtime.rank].copy()).to(device)
    initial_tail = torch.from_numpy(tails[runtime.rank].copy()).to(device)
    case = _Case(
        runtime=runtime,
        module=module,
        config=config,
        A=data["A"],
        B=data["B"],
        ag_out=ag_out,
        semaphore=semaphore,
        out=torch.empty((config.M, config.local_n), dtype=torch.float16, device=device),
        profiler=torch.empty(PROFILER_BUFFER_SIZE, dtype=torch.uint64, device=device),
        task_types=torch.empty_like(initial_task_types),
        task_idxs=torch.empty_like(initial_task_idxs),
        head=torch.from_numpy(heads[runtime.rank].copy()).to(device),
        tail=torch.empty_like(initial_tail),
        initial_task_types=initial_task_types,
        initial_task_idxs=initial_task_idxs,
        initial_tail=initial_tail,
        ag_out_torch=torch_view(ag_out),
        semaphore_torch=torch_view(semaphore),
    )
    with torch.cuda.stream(runtime.timing_stream):
        case.reset()
    torch.cuda.synchronize(runtime.device_index)
    runtime.barrier()
    return case


def _check_correctness(case: _Case) -> None:
    config = case.config
    gathered_A = torch.empty((config.M, config.K), dtype=torch.float16, device=case.A.device)
    with torch.cuda.stream(case.runtime.timing_stream):
        dist.all_gather_into_tensor(gathered_A, case.A)
    case.runtime.timing_stream.synchronize()

    local = slice(case.runtime.rank * config.local_m, (case.runtime.rank + 1) * config.local_m)
    case.ag_out_torch[local].copy_(case.A)
    torch.testing.assert_close(case.ag_out_torch, gathered_A, rtol=0, atol=0)

    reference = torch.matmul(gathered_A, case.B.T)
    torch.testing.assert_close(case.out, reference, rtol=1e-3, atol=2e-2)


def _run_worker(
    runtime: DistributedRuntime, module: Any, mode: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    config = _check_config(
        kwargs["M"], kwargs["N"], kwargs["K"], kwargs["world_size"], kwargs["dtype"]
    )
    _check_scheduler(kwargs.get("scheduler", "dynamic"))
    data = prepare_data(rank=runtime.rank, **kwargs)
    case = _allocate_case(runtime, module, data, config)

    if mode == "test":
        with torch.cuda.stream(runtime.timing_stream):
            case.prepare()
            case.launch()
        runtime.device.sync(runtime.compute_stream)
        _check_correctness(case)
        return {"status": "OK"}
    if mode != "bench":
        raise ValueError(f"unsupported distributed worker mode {mode!r}")

    from tirx_kernels.runner import bench

    def prepare() -> None:
        case.reset()
        case.prepare()

    baselines = create_baseline_suite(
        runtime,
        data,
        workload="allgather_gemm",
        M=kwargs["M"],
        N=kwargs["N"],
        K=kwargs["K"],
        world_size=kwargs["world_size"],
    )
    try:
        result = bench(
            {"tirx": case.launch},
            references=baselines.references(),
            timer=kwargs.get("timer", "kineto"),
            rounds=kwargs.get("rounds", 1),
            cooldown_s=kwargs.get("cooldown_s", 1.0),
            distributed=runtime.bench_context(),
            prepare={"tirx": prepare},
        )
        result["baseline_metadata"] = baselines.metadata()
        result["ratio_definition"] = "baseline_us / tirx_us"
        result["ratios"] = baseline_ratios(result)
        return {"status": "OK", **result}
    finally:
        baselines.close()


def run_test(
    M: int = M,
    N: int = N,
    K: int = K,
    world_size: int = WORLD_SIZE,
    dtype: str = "float16",
    seed: int = 42,
    scheduler: str = "dynamic",
    **_kwargs: Any,
) -> None:
    """Compile, launch on the requested TP ranks, and compare with PyTorch."""

    _check_config(M, N, K, world_size, dtype)
    _check_scheduler(scheduler)
    run_distributed(
        get_kernel(M, N, K, world_size, dtype, scheduler=scheduler),
        world_size=world_size,
        worker=_run_worker,
        mode="test",
        worker_kwargs={
            "M": M,
            "N": N,
            "K": K,
            "world_size": world_size,
            "dtype": dtype,
            "seed": seed,
            "scheduler": scheduler,
        },
    )


def run_bench(
    M: int = M,
    N: int = N,
    K: int = K,
    world_size: int = WORLD_SIZE,
    dtype: str = "float16",
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 1,
    cooldown_s: float = 1.0,
    scheduler: str = "dynamic",
    **_kwargs: Any,
) -> dict[str, Any]:
    """Return cold-cache Kineto full-span timings for TIRx and both baselines."""

    _check_config(M, N, K, world_size, dtype)
    _check_scheduler(scheduler)
    if timer not in {None, "kineto"}:
        raise ValueError("distributed AllGather+GEMM supports only timer='kineto'")
    if warmup is not None or repeat is not None:
        raise ValueError("timer='kineto' uses fixed iteration counts and rejects overrides")
    return prepare_bench(
        M=M, N=N, K=K, world_size=world_size, dtype=dtype, scheduler=scheduler
    ).run_gpu(warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s)


def run_gpu(prepared, **kwargs: Any) -> dict[str, Any]:
    """Start distributed ranks only after the complete GPU claim exists."""
    return prepared.run_gpu(**kwargs)


def prepare_bench(
    M: int = M,
    N: int = N,
    K: int = K,
    world_size: int = WORLD_SIZE,
    dtype: str = "float16",
    *,
    scheduler: str = "dynamic",
    **_kwargs: Any,
):
    """Compile/export before assignment; ranks start CUDA in run_gpu."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    _check_config(M, N, K, world_size, dtype)
    _check_scheduler(scheduler)
    state = prepare_distributed_bench(
        _get_benchmark_kernel(M, N, K, world_size, dtype, scheduler=scheduler),
        world_size=world_size,
        worker=_run_worker,
        worker_kwargs={
            "M": M,
            "N": N,
            "K": K,
            "world_size": world_size,
            "dtype": dtype,
            "scheduler": scheduler,
        },
        required_timer="kineto",
    )
    return prepared_gpu_benchmark(run_gpu, state, required_num_gpus=world_size, close=state.close)


__all__ = [
    "CONFIGS",
    "KERNEL_META",
    "AllGatherGemmConfig",
    "_get_benchmark_kernel",
    "derive_config",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
