# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Direct TP1/TP4 port of the fused persistent dynamic-multimem GemmRS kernel."""

from __future__ import annotations

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
from tvm.tirx.script.builder.ir import name_meta_class_value

from .utils._baselines import create_baseline_suite
from .utils._baselines import ratios as baseline_ratios
from .utils._model_shapes import (
    GEMM_RS_MODEL_SHAPES,
    SUPPORTED_WORLD_SIZES,
    make_configs,
    shape_set,
)
from .utils._runtime import (
    DistributedRuntime,
    barrier_on_compute_stream,
    prepare_distributed_bench,
    require_nvls_multicast,
    run_distributed,
    symmetric_empty,
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
    RS = 1


@T.meta_class
class MBarrier:
    def __init__(self, pool, depth, phase_offset=0, leader=None):
        self.buf = pool.alloc((depth,), "uint64", align=8)
        self.depth = depth
        self.phase_offset = phase_offset
        self.leader = T.cuda.thread_rank() == 0 if leader is None else leader

    def init(self, count):
        with T.If(self.leader):
            with T.Then():
                with T.unroll(self.depth) as i:
                    T.evaluate(
                        T.ptx.mbarrier.init.shared.b64(self.buf.ptr_to([i]), T.uint32(count))
                    )

    def ptr_to(self, indices):
        return self.buf.ptr_to(indices)

    def wait(self, stage, phase):
        T.evaluate(T.cuda.mbarrier_wait(self.buf.ptr_to([stage]), phase ^ self.phase_offset))

    def arrive(self, stage, remote=None):
        if remote is None:
            T.evaluate(T.ptx.mbarrier.arrive.shared.b64(self.buf.ptr_to([stage]), T.uint32(1)))
        else:
            mapped = T.alloc_local([1], "uint32")
            T.evaluate(
                T.ptx.mapa.shared__cluster.u32(
                    mapped[0],
                    T.cuda.cvta_generic_to_shared(self.buf.ptr_to([stage])),
                    T.uint32(remote),
                )
            )
            T.evaluate(T.ptx.mbarrier.arrive.shared__cluster.b64(mapped[0]))


class TMABar(MBarrier):
    def arrive(self, stage, tx_count=None, remote=None):
        if remote is not None:
            raise ValueError("GemmRS does not use remote TMA arrivals")
        if tx_count is None:
            T.evaluate(T.ptx.mbarrier.arrive.shared.b64(self.buf.ptr_to([stage]), T.uint32(1)))
        else:
            T.evaluate(
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                    self.buf.ptr_to([stage]), T.uint32(tx_count)
                )
            )


class TCGen05Bar(MBarrier):
    def arrive(self, stage, cta_group=1, cta_mask=None):
        if cta_mask is None:
            T.evaluate(
                T.ptx[
                    f"tcgen05.commit.cta_group::{cta_group}.mbarrier::arrive::one.shared::cluster.b64"
                ](self.buf.ptr_to([stage]))
            )
        else:
            T.evaluate(
                T.ptx[
                    f"tcgen05.commit.cta_group::{cta_group}.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
                ](self.buf.ptr_to([stage]), T.Cast("uint16", cta_mask))
            )


@T.meta_class
class DataPipeline:
    def __init__(
        self, pool, stages, *, full, empty, init_full=1, init_empty=1, empty_phase_offset=0
    ):
        kinds = {"tma": TMABar, "tcgen05": TCGen05Bar, "mbar": MBarrier}
        self.stages = stages
        self.full = kinds[full](pool, stages)
        self.full.init(init_full)
        self.empty = kinds[empty](pool, stages, phase_offset=empty_phase_offset)
        self.empty.init(init_empty)


_SPECIALIZATION_M_ENV = "TIRX_INTERNAL_GEMMRS_M"
_SPECIALIZATION_N_ENV = "TIRX_INTERNAL_GEMMRS_N"
_SPECIALIZATION_K_ENV = "TIRX_INTERNAL_GEMMRS_K"
_SPECIALIZATION_WORLD_SIZE_ENV = "TIRX_INTERNAL_GEMMRS_WORLD_SIZE"

M = int(os.environ.get(_SPECIALIZATION_M_ENV, "8192"))
N = int(os.environ.get(_SPECIALIZATION_N_ENV, "5120"))
TOTAL_K = int(os.environ.get(_SPECIALIZATION_K_ENV, "25600"))
WORLD_SIZE = int(os.environ.get(_SPECIALIZATION_WORLD_SIZE_ENV, "4"))
if TOTAL_K % WORLD_SIZE:
    raise ValueError("GemmRS total K must be divisible by world_size")
K = TOTAL_K // WORLD_SIZE
DTYPE = "float16"
_SUPPORTED_SHAPES = shape_set(GEMM_RS_MODEL_SHAPES)
M_CLUSTER = 2
N_CLUSTER = 1
WG_NUMBER = 3
WARP_NUMBER = 4
NUM_CONSUMER = 2
NUM_THREADS = 32 * WARP_NUMBER * WG_NUMBER
SM_NUMBER = 148
PIPELINE_DEPTH = 4
F16_BYTES = 2
F128_BYTES = 16
d_type, a_type, b_type = ("float16", "float16", "float16")
LOCAL_M = M // WORLD_SIZE
BLK_M, BLK_N, BLK_K = (128, 128, 64)
assert LOCAL_M * WORLD_SIZE == M, "M must be divisible by WORLD_SIZE"
assert LOCAL_M % BLK_M == 0, "LOCAL_M must be divisible by BLK_M"
MMA_M, MMA_N, MMA_K = (256, 256, 16)
EPI_TILE = 64
SWIZZLE = 3
SMEM_RESERVED_BYTES = 1024
SMEM_SIZE = (
    PIPELINE_DEPTH * NUM_CONSUMER * BLK_M * BLK_K * F16_BYTES
    + PIPELINE_DEPTH * BLK_N * BLK_K * F16_BYTES
    + NUM_CONSUMER * BLK_M * EPI_TILE * F16_BYTES
    + SMEM_RESERVED_BYTES
)
SM100_SMEM_CAPACITY = 232448
assert SMEM_SIZE <= SM100_SMEM_CAPACITY, "GemmRS shared-memory usage exceeds the SM100 limit"
TMEM_LD_SIZE = 64
N_COLS = 512
CTA_GROUP = 2
# TMA instruction spellings (unicast g2s at cluster scope, plain tile s2g).
_TMA_G2S_CG2 = (
    "cp.async.bulk.tensor.2d.shared::cluster.global"
    f".mbarrier::complete_tx::bytes.cta_group::{CTA_GROUP}"
)
_TMA_S2G = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
# tcgen05.mma spelling: kind::f16 from the (float32, float16, float16) dtypes.
_MMA_CHAIN = f"tcgen05.mma.cta_group::{CTA_GROUP}.kind::f16"
_MMA_ZERO_MASKS = [0] * (4 if CTA_GROUP == 1 else 8)
_TMEM_LD_64 = "tcgen05.ld.sync.aligned.32x32b.x64.b32"
_CVT_F32X2 = "cvt.rn.f16x2.f32"
PIPE_CYCLE = K // BLK_K // PIPELINE_DEPTH
PIPE_REMAIN_NUM = K // BLK_K % PIPELINE_DEPTH
assert PIPELINE_DEPTH == 4
GROUP_SIZE = 8
assert M % (NUM_CONSUMER * BLK_M * CTA_GROUP) == 0
assert N % (BLK_N * CTA_GROUP) == 0
GEMM_M_CLUSTERS = M // (NUM_CONSUMER * BLK_M * CTA_GROUP)
GEMM_N_CLUSTERS = N // (BLK_N * CTA_GROUP)
TILE_M, TILE_N = (BLK_M * 2, BLK_N * 2)
RS_M_CLUSTERS = LOCAL_M // (BLK_M * CTA_GROUP)
RS_N_CLUSTERS = N // (BLK_N * CTA_GROUP)
CAPACITY = 2048
TASK_IDX_LEN = 2
C2P_THREAD_COUNT = NUM_THREADS * CTA_GROUP
FUSED_DEVICE_ENTRYPOINT = "test_mma_ss_tma_2sm_persistent"


@dataclass(frozen=True)
class GemmRSConfig:
    M: int
    N: int
    total_k: int
    world_size: int
    dtype: str
    k_local: int
    local_m: int
    pipe_cycle: int
    pipe_remainder: int
    gemm_m_clusters: int
    gemm_n_clusters: int
    rs_m_clusters: int
    rs_n_clusters: int
    gemm_task_count: int
    rs_task_count: int
    completion_count: int


def _mapa_u64_tx(ptr, rank):
    """`mapa.u64` into a declared register, returned as an ordinary value."""

    mapped = T.alloc_local([1], "uint64")
    T.evaluate(T.ptx.mapa.u64(mapped[0], ptr, T.uint32(rank)))
    return mapped[0]


def derive_config(
    M: int = M, N: int = N, K: int = TOTAL_K, world_size: int = WORLD_SIZE, dtype: str = DTYPE
) -> GemmRSConfig:
    """Validate and derive one model-shape/world-size specialization."""

    if (M, N, K) not in _SUPPORTED_SHAPES:
        raise ValueError(f"GemmRS does not support shape M={M}, N={N}, K={K}")
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"GemmRS supports world_size in {SUPPORTED_WORLD_SIZES}; got {world_size}")
    if dtype != DTYPE:
        raise ValueError(f"GemmRS supports only dtype={DTYPE!r}; got {dtype!r}")
    if M % world_size or K % world_size:
        raise ValueError("GemmRS M and K must be divisible by world_size")
    k_local = K // world_size
    local_m = M // world_size
    if k_local % BLK_K:
        raise ValueError(f"GemmRS rank-local K must be divisible by {BLK_K}")
    if M % (NUM_CONSUMER * BLK_M * CTA_GROUP) or N % (BLK_N * CTA_GROUP):
        raise ValueError("GemmRS M and N do not satisfy the 2-CTA tile geometry")
    if local_m % TILE_M:
        raise ValueError(f"GemmRS rank-local M must be divisible by {TILE_M}")
    gemm_m_clusters = M // (NUM_CONSUMER * BLK_M * CTA_GROUP)
    gemm_n_clusters = N // (BLK_N * CTA_GROUP)
    rs_m_clusters = local_m // (BLK_M * CTA_GROUP)
    rs_n_clusters = N // (BLK_N * CTA_GROUP)
    config = GemmRSConfig(
        M=M,
        N=N,
        total_k=K,
        world_size=world_size,
        dtype=dtype,
        k_local=k_local,
        local_m=local_m,
        pipe_cycle=(k_local // BLK_K) // PIPELINE_DEPTH,
        pipe_remainder=(k_local // BLK_K) % PIPELINE_DEPTH,
        gemm_m_clusters=gemm_m_clusters,
        gemm_n_clusters=gemm_n_clusters,
        rs_m_clusters=rs_m_clusters,
        rs_n_clusters=rs_n_clusters,
        gemm_task_count=gemm_m_clusters * gemm_n_clusters,
        rs_task_count=rs_m_clusters * rs_n_clusters,
        completion_count=2 * world_size,
    )
    if config.gemm_task_count > CAPACITY or config.rs_task_count > CAPACITY:
        raise AssertionError(
            f"GemmRS queue capacity {CAPACITY} is too small for "
            f"{config.gemm_task_count} GEMM and {config.rs_task_count} RS tasks"
        )
    return config


ld_reduce_8xfp16 = '\n__forceinline__ __device__ void ld_reduce_8_fp16(void* src_addr, void* dst_addr) {\n    int4* source = (int4*) nvshmemx_mc_ptr(NVSHMEM_TEAM_WORLD, src_addr);\n    int4* dest = (int4*) dst_addr;\n    constexpr int UNROLL = 1;\n    union {\n        uint16_t u2[8 * UNROLL];\n        uint64_t u8[2 * UNROLL];\n    };\n    for (int u = 0; u < UNROLL; u++) {\n        asm("multimem.ld_reduce.global.add.v8.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"\n            : "=h"(u2[8 * u]), "=h"(u2[8 * u + 1]), "=h"(u2[8 * u + 2]), "=h"(u2[8 * u + 3]), "=h"(u2[8 * u + 4]), "=h"(u2[8 * u + 5]), "=h"(u2[8 * u + 6]), "=h"(u2[8 * u + 7])\n            : "l"(source + u));\n    }\n    for (int u = 0; u < UNROLL; u++) {\n        asm("st.global.v2.b64 [%0], {%1, %2};" ::"l"(dest + u), "l"(u8[2 * u]),\n            "l"(u8[2 * u + 1]));\n    }\n}\n'
semaphore_notify_remote = "\n__forceinline__ __device__ uint64_t semaphore_notify_remote(int32_t signal_rank, uint64_t* addr, uint64_t signal_value) {\n    auto dst_addr = reinterpret_cast<unsigned long long*>(nvshmem_ptr(addr, signal_rank));\n    return atomicAdd_system(dst_addr, signal_value);\n}\n"
enqueue_remote = """
__forceinline__ __device__ void enqueue_remote(
        int32_t* task_types, int32_t* task_idxs, int32_t* tail, int32_t mask,
        int32_t signal_rank, int32_t task_type, int32_t task_idx0, int32_t task_idx1) {
    int32_t* remote_task_types = (int32_t*)nvshmem_ptr(task_types, signal_rank);
    int32_t* remote_task_idxs = (int32_t*)nvshmem_ptr(task_idxs, signal_rank);
    int32_t* remote_tail = (int32_t*)nvshmem_ptr(tail, signal_rank);
    int32_t tail_r = atomicAdd_system(remote_tail, 1);
    int32_t masked_pos = tail_r & mask;
    remote_task_idxs[masked_pos * 2] = task_idx0;
    remote_task_idxs[masked_pos * 2 + 1] = task_idx1;
    asm volatile(
        "st.global.release.sys.b32 [%0], %1;"
        :
        : "l"(remote_task_types + masked_pos), "r"(task_type)
        : "memory");
}
"""
# Keep the acquire-load polling loop in one device helper.  Expanding the loop
# into TIR can leave worker CTAs spinning indefinitely on queue publication.
while_ld_global_acquire = """
__forceinline__ __device__ int32_t while_ld_global_acquire(int32_t* addr) {
    int32_t value;
    asm volatile(
        "ld.global.acquire.sys.b32 %0, [%1];"
        : "=r"(value)
        : "l"(addr)
        : "memory");
    while (value < 0) {
        __nanosleep(40);
        asm volatile(
            "ld.global.acquire.sys.b32 %0, [%1];"
            : "=r"(value)
            : "l"(addr)
            : "memory");
    }
    return value;
}
"""


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

    def init(self, p2c_thread_count: int = 1, c2p_thread_count: int = 1):
        tid = _builder_assign(
            "tid", T.thread_id([NUM_THREADS]), locals().get("tid", _BUILDER_MISSING)
        )
        _builder_setattr(self, "idx", 0)
        _builder_setattr(self, "p2c_phase", 0)
        _builder_setattr(self, "c2p_phase", 1)
        with T.If(tid == 0):
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


def int_var(scope="local", dtype="int32", align=4):
    buf = T.alloc_buffer([1], dtype, scope=scope, align=align)
    return buf


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
        if capacity & capacity - 1:
            raise ValueError("capacity must be a power-of-two")
        self.mask = capacity - 1
        self.task_types = task_types
        self.task_idxs = task_idxs
        self.head = head
        self.tail = tail
        self.head_r = int_var()
        self.masked_pos = int_var()
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
        rs_rem: T.Buffer,
        cbx,
        bx,
        rank,
    ):
        _builder_emit(T.ptx.atom.global_.add.s32(self.head_r[0], self.head.ptr_to([0]), T.int32(1)))
        with T.If(self.head_r[0] < self.num_tot_tasks):
            with T.Then():
                T.buffer_store(self.masked_pos, self.head_r[0] & self.mask, [0])
                T.buffer_store(
                    fetched_task_type,
                    T.cuda.func_call(
                        "while_ld_global_acquire",
                        self.task_types.ptr_to([self.masked_pos[0]]),
                        source_code=while_ld_global_acquire,
                        return_type="int32",
                    ),
                    [0],
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


class RSMPMCQueue(MPMCQueue):
    def dequeue(
        self,
        fetched_task_type: T.Buffer,
        fetched_task_idx0: T.Buffer,
        fetched_task_idx1: T.Buffer,
        rs_rem: T.Buffer,
        cbx,
        bx,
        rank,
    ):
        with T.If(rs_rem[0] >= 0):
            with T.Then():
                T.buffer_store(self.head_r, rs_rem[0], [0])
                T.buffer_store(rs_rem, -1, [0])
            with T.Else():
                _builder_emit(
                    T.ptx.atom.global_.add.s32(self.head_r[0], self.head.ptr_to([0]), T.int32(1))
                )
        with T.If(self.head_r[0] < self.num_tot_tasks):
            with T.Then():
                T.buffer_store(self.masked_pos, self.head_r[0] & self.mask, [0])
                _builder_emit(
                    T.ptx.ld.global_.acquire.sys.b32(
                        fetched_task_type[0], self.task_types.ptr_to([self.masked_pos[0]])
                    )
                )
                with T.If(fetched_task_type[0] < 0):
                    with T.Then():
                        T.buffer_store(rs_rem, self.head_r[0], [0])
                    with T.Else():
                        _builder_emit(
                            T.ptx.st.global_.s32(
                                self.task_types.ptr_to([self.masked_pos[0]]), T.int32(-1)
                            )
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


def consumer_fetch(
    sch_pipe, packed_value, rs_rem, fetched_task_type, fetched_task_idx0, fetched_task_idx1
):
    _builder_emit(sch_pipe.consumer_wait(0))
    _builder_emit(
        T.ptx.ld.shared__cluster.v4.b32(
            rs_rem[0],
            fetched_task_type[0],
            fetched_task_idx0[0],
            fetched_task_idx1[0],
            packed_value.ptr_to([0]),
        )
    )
    mapped = _builder_assign(
        "mapped", T.alloc_local([1], "uint64"), locals().get("mapped", _BUILDER_MISSING)
    )
    _builder_emit(
        T.ptx.mapa.shared__cluster.u64(
            mapped[0], sch_pipe.mbar_c2p.ptr_to([sch_pipe.idx, 0]), T.uint32(0)
        )
    )
    _builder_emit(T.ptx.mbarrier.arrive.b64(mapped[0], T.uint32(1), pred=T.bool(True)))
    _builder_setattr(sch_pipe, "p2c_phase", sch_pipe.p2c_phase ^ 1)


@T.meta_class
class MixedDynamicTileScheduler:
    def __init__(
        self,
        gemm_queue: GEMMMPMCQueue,
        rs_queue: RSMPMCQueue,
        packed_value: T.Buffer,
        sch_pipe: Pipeline,
    ):
        self.gemm_queue = gemm_queue
        self.rs_queue = rs_queue
        self.sch_pipe = sch_pipe
        self.fetched_task_type = int_var()
        self.fetched_task_idx0 = int_var()
        self.fetched_task_idx1 = int_var()
        self.rs_rem = int_var()
        self.packed_value = packed_value

    def _fetch_from_queue(self, cbx, bx, rank, warp_id_in_cta, lane_id):
        with T.If((warp_id_in_cta == 11) & (lane_id == 0)):
            with T.Then():
                with T.If(cbx == 0):
                    with T.Then():
                        _builder_emit(self.sch_pipe.producer_wait(0))
                        _builder_emit(
                            self.rs_queue.dequeue(
                                self.fetched_task_type,
                                self.fetched_task_idx0,
                                self.fetched_task_idx1,
                                self.rs_rem,
                                cbx,
                                bx,
                                rank,
                            )
                        )
                        with T.If(self.fetched_task_type[0] < 0):
                            with T.Then():
                                _builder_emit(
                                    self.gemm_queue.dequeue(
                                        self.fetched_task_type,
                                        self.fetched_task_idx0,
                                        self.fetched_task_idx1,
                                        self.rs_rem,
                                        cbx,
                                        bx,
                                        rank,
                                    )
                                )
                        _builder_emit(
                            T.ptx.st.shared__cluster.v4.b32(
                                self.packed_value.ptr_to([0]),
                                self.rs_rem[0],
                                self.fetched_task_type[0],
                                self.fetched_task_idx0[0],
                                self.fetched_task_idx1[0],
                            )
                        )
                        _builder_emit(T.cuda.thread_fence())
                        mapped0 = _builder_assign(
                            "mapped0",
                            T.alloc_local([1], "uint64"),
                            locals().get("mapped0", _BUILDER_MISSING),
                        )
                        _builder_emit(
                            T.ptx.mapa.shared__cluster.u64(
                                mapped0[0],
                                self.sch_pipe.mbar_p2c.ptr_to([self.sch_pipe.idx, 0]),
                                T.uint32(0),
                            )
                        )
                        _builder_emit(
                            T.ptx.mbarrier.arrive.b64(mapped0[0], T.uint32(1), pred=T.bool(True))
                        )
                        mapped1 = _builder_assign(
                            "mapped1",
                            T.alloc_local([1], "uint64"),
                            locals().get("mapped1", _BUILDER_MISSING),
                        )
                        _builder_emit(
                            T.ptx.mapa.shared__cluster.u64(
                                mapped1[0],
                                self.sch_pipe.mbar_p2c.ptr_to([self.sch_pipe.idx, 0]),
                                T.uint32(1),
                            )
                        )
                        _builder_emit(
                            T.ptx.mbarrier.arrive.b64(mapped1[0], T.uint32(1), pred=T.bool(True))
                        )
                        _builder_setattr(self.sch_pipe, "c2p_phase", self.sch_pipe.c2p_phase ^ 1)
        _builder_emit(
            consumer_fetch(
                self.sch_pipe,
                self.packed_value,
                self.rs_rem,
                self.fetched_task_type,
                self.fetched_task_idx0,
                self.fetched_task_idx1,
            )
        )

    def init(self, cbx, bx, rank, warp_id_in_cta, lane_id):
        T.buffer_store(self.rs_rem, -1, [0])
        _builder_emit(self._fetch_from_queue(cbx, bx, rank, warp_id_in_cta, lane_id))

    def next_tile(self, cbx, bx, rank, warp_id_in_cta, lane_id):
        _builder_emit(self._fetch_from_queue(cbx, bx, rank, warp_id_in_cta, lane_id))

    def valid(self):
        return tvm.tirx.any(self.fetched_task_type[0] >= 0, self.rs_rem[0] >= 0)


@T.meta_class
class Semaphore:
    def __init__(self, cnt, buffer):
        self.cnt = cnt
        self.sem = buffer
        self.state = T.alloc_buffer([1], "uint64", scope="local", align=8)

    def semaphore_notify(self, signal_rank, tid, m_idx, n_idx, rs_queue):
        with T.If(tid % 128 == 0):
            with T.Then():
                _builder_emit(T.ptx.fence.sc.sys())
                T.buffer_store(
                    self.state,
                    T.cuda.func_call(
                        "semaphore_notify_remote",
                        signal_rank,
                        self.sem.ptr_to([m_idx, n_idx]),
                        T.uint64(1),
                        source_code=semaphore_notify_remote,
                        return_type="uint64",
                    )
                    + 1,
                    [0],
                )
                with T.If(self.state[0] == self.cnt):
                    with T.Then():
                        _builder_emit(
                            rs_queue.enqueue(signal_rank, TaskType.RS.value, m_idx, n_idx)
                        )


def _build_active_prim_func():
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("test_mma_ss_tma_2sm_persistent")
            A = T.arg("A", T.Buffer((M, K), a_type))
            B = T.arg("B", T.Buffer((N, K), b_type))
            gemm_out = T.arg("gemm_out", T.Buffer((M, N), d_type))
            semaphore = T.arg("semaphore", T.Buffer((LOCAL_M // TILE_M, N // TILE_N), "uint64"))
            out = T.arg("out", T.Buffer((LOCAL_M, N), d_type))
            gemm_task_types = T.arg("gemm_task_types", T.Buffer((CAPACITY,), "int32"))
            gemm_task_idxs = T.arg("gemm_task_idxs", T.Buffer((CAPACITY, 2), "int32"))
            gemm_head = T.arg("gemm_head", T.Buffer((1,), "int32"))
            gemm_tail = T.arg("gemm_tail", T.Buffer((1,), "int32"))
            rs_task_types = T.arg("rs_task_types", T.Buffer((CAPACITY,), "int32"))
            rs_task_idxs = T.arg("rs_task_idxs", T.Buffer((CAPACITY, 2), "int32"))
            rs_head = T.arg("rs_head", T.Buffer((1,), "int32"))
            rs_tail = T.arg("rs_tail", T.Buffer((1,), "int32"))
            A_tensor_map = _builder_bind(
                "A_tensor_map", T.tvm_stack_alloca("tensormap", 1), T.handle("tensormap")
            )
            B_tensor_map = _builder_bind(
                "B_tensor_map", T.tvm_stack_alloca("tensormap", 1), T.handle("tensormap")
            )
            D_tensor_map = _builder_bind(
                "D_tensor_map", T.tvm_stack_alloca("tensormap", 1), T.handle("tensormap")
            )
            _builder_emit(
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    A_tensor_map,
                    a_type,
                    2,
                    A.data,
                    K,
                    M,
                    K * F16_BYTES,
                    BLK_K,
                    BLK_M,
                    1,
                    1,
                    0,
                    SWIZZLE,
                    0,
                    0,
                )
            )
            _builder_emit(
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    B_tensor_map,
                    b_type,
                    2,
                    B.data,
                    K,
                    N,
                    K * F16_BYTES,
                    BLK_K,
                    BLK_N,
                    1,
                    1,
                    0,
                    SWIZZLE,
                    0,
                    0,
                )
            )
            _builder_emit(
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    D_tensor_map,
                    d_type,
                    2,
                    gemm_out.data,
                    N,
                    M,
                    N * F16_BYTES,
                    EPI_TILE,
                    BLK_M,
                    1,
                    1,
                    0,
                    SWIZZLE,
                    0,
                    0,
                )
            )
            _builder_emit(T.device_entry())
            cbx, cby = _builder_assign_many(
                ("cbx", "cby"),
                T.cta_id_in_cluster([M_CLUSTER, N_CLUSTER]),
                (locals().get("cbx", _BUILDER_MISSING), locals().get("cby", _BUILDER_MISSING)),
            )
            bx = _builder_assign("bx", T.cta_id([SM_NUMBER]), locals().get("bx", _BUILDER_MISSING))
            wg_id = _builder_assign(
                "wg_id", T.warpgroup_id([WG_NUMBER]), locals().get("wg_id", _BUILDER_MISSING)
            )
            warp_id = _builder_assign(
                "warp_id", T.warp_id_in_wg([WARP_NUMBER]), locals().get("warp_id", _BUILDER_MISSING)
            )
            warp_id_in_cta = _builder_assign(
                "warp_id_in_cta",
                T.warp_id([WG_NUMBER * WARP_NUMBER]),
                locals().get("warp_id_in_cta", _BUILDER_MISSING),
            )
            lane_id = _builder_assign(
                "lane_id", T.lane_id([32]), locals().get("lane_id", _BUILDER_MISSING)
            )
            tid = _builder_assign(
                "tid", T.thread_id([NUM_THREADS]), locals().get("tid", _BUILDER_MISSING)
            )
            rank = _builder_assign(
                "rank", T.nvshmem.my_pe(), locals().get("rank", _BUILDER_MISSING)
            )
            pool = _builder_assign("pool", T.SMEMPool(), locals().get("pool", _BUILDER_MISSING))
            tmem_addr = _builder_assign(
                "tmem_addr",
                pool.alloc([1], "uint32", align=4),
                locals().get("tmem_addr", _BUILDER_MISSING),
            )
            tmem_pool = _builder_assign(
                "tmem_pool",
                T.TMEMPool(pool, total_cols=N_COLS, cta_group=CTA_GROUP, tmem_addr=tmem_addr),
                locals().get("tmem_pool", _BUILDER_MISSING),
            )
            smem_pipe = _builder_assign(
                "smem_pipe",
                DataPipeline(
                    pool,
                    PIPELINE_DEPTH,
                    full="tma",
                    empty="tcgen05",
                    init_empty=NUM_CONSUMER,
                    empty_phase_offset=1,
                ),
                locals().get("smem_pipe", _BUILDER_MISSING),
            )
            tmem_pipe = _builder_assign(
                "tmem_pipe",
                DataPipeline(
                    pool,
                    NUM_CONSUMER,
                    full="tcgen05",
                    empty="mbar",
                    init_empty=128 * NUM_CONSUMER,
                    empty_phase_offset=1,
                ),
                locals().get("tmem_pipe", _BUILDER_MISSING),
            )
            packed_buf = _builder_assign(
                "packed_buf",
                pool.alloc((1,), "uint64", align=16),
                locals().get("packed_buf", _BUILDER_MISSING),
            )
            sch_pipe_base = _builder_scalar("sch_pipe_base", pool.offset // 8, "int32")
            _builder_emit(pool.move_base_to(pool.offset + 2 * 1 * 1 * 8))
            _builder_emit(pool.move_base_to(SMEM_RESERVED_BYTES))
            A_smem = _builder_assign(
                "A_smem",
                pool.alloc_tcgen05_mma_AB((PIPELINE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type),
                locals().get("A_smem", _BUILDER_MISSING),
            )
            B_smem = _builder_assign(
                "B_smem",
                pool.alloc_tcgen05_mma_AB((PIPELINE_DEPTH, BLK_N, BLK_K), b_type),
                locals().get("B_smem", _BUILDER_MISSING),
            )
            D_smem = _builder_assign(
                "D_smem",
                pool.alloc_tcgen05_mma_AB((NUM_CONSUMER, BLK_M, EPI_TILE), d_type),
                locals().get("D_smem", _BUILDER_MISSING),
            )
            _builder_emit(pool.commit())
            reg = _builder_assign(
                "reg",
                T.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local"),
                locals().get("reg", _BUILDER_MISSING),
            )
            reg_fp16 = _builder_assign(
                "reg_fp16",
                T.alloc_buffer((TMEM_LD_SIZE // 2,), "uint32", scope="local", align=16),
                locals().get("reg_fp16", _BUILDER_MISSING),
            )
            copy_word0 = _builder_alloc_scalar("copy_word0", "uint32")
            copy_word1 = _builder_alloc_scalar("copy_word1", "uint32")
            copy_word2 = _builder_alloc_scalar("copy_word2", "uint32")
            copy_word3 = _builder_alloc_scalar("copy_word3", "uint32")
            descA = _builder_alloc_scalar("descA", "uint64")
            descB = _builder_alloc_scalar("descB", "uint64")
            descI = _builder_alloc_scalar("descI", "uint32")
            phase = _builder_alloc_scalar("phase", "int32")
            phase_tmem = _builder_alloc_scalar("phase_tmem", "int32")
            stage = _builder_alloc_scalar("stage", "int32")
            tmem_addr_local = _builder_alloc_scalar("tmem_addr_local", "uint32")
            sem = Semaphore(cnt=2 * WORLD_SIZE, buffer=semaphore)
            offset = _builder_alloc_scalar("offset", "int32")
            gemm_queue = GEMMMPMCQueue(
                CAPACITY,
                gemm_task_types,
                gemm_task_idxs,
                gemm_head,
                gemm_tail,
                GEMM_M_CLUSTERS * GEMM_N_CLUSTERS,
            )
            rs_queue = RSMPMCQueue(
                CAPACITY,
                rs_task_types,
                rs_task_idxs,
                rs_head,
                rs_tail,
                RS_M_CLUSTERS * RS_N_CLUSTERS,
            )
            packed_ptr = _builder_bind(
                "packed_ptr",
                T.reinterpret(
                    PointerType(PrimType("uint64")), _mapa_u64_tx(packed_buf.ptr_to([0]), 0)
                ),
                T.Var(name="packed_ptr", ty=PointerType(PrimType("uint64"))),
            )
            packed_value = _builder_assign(
                "packed_value",
                T.decl_buffer([1], "uint64", data=packed_ptr, scope="shared"),
                locals().get("packed_value", _BUILDER_MISSING),
            )
            sch_pipe = _builder_assign(
                "sch_pipe",
                Pipeline(
                    pool.ptr,
                    sch_pipe_base,
                    pipeline_depth=1,
                    pipeline_num=1,
                    p_single_cta=True,
                    c_single_cta=False,
                ),
                locals().get("sch_pipe", _BUILDER_MISSING),
            )
            tile_scheduler = _builder_assign(
                "tile_scheduler",
                MixedDynamicTileScheduler(gemm_queue, rs_queue, packed_value, sch_pipe),
                locals().get("tile_scheduler", _BUILDER_MISSING),
            )
            ptr = _builder_bind(
                "ptr",
                T.reinterpret(
                    PointerType(PrimType("uint64")), _mapa_u64_tx(smem_pipe.full.ptr_to([0]), 0)
                ),
                T.Var(name="ptr", ty=PointerType(PrimType("uint64"))),
            )
            tma_finished = _builder_assign(
                "tma_finished",
                T.decl_buffer([PIPELINE_DEPTH], "uint64", data=ptr, scope="shared"),
                locals().get("tma_finished", _BUILDER_MISSING),
            )
            phase = _builder_assign("phase", 0, locals().get("phase", _BUILDER_MISSING))
            phase_tmem = _builder_assign(
                "phase_tmem", 0, locals().get("phase_tmem", _BUILDER_MISSING)
            )
            _builder_emit(sch_pipe.init(c2p_thread_count=C2P_THREAD_COUNT, p2c_thread_count=1))
            _builder_emit(
                T.cuda.tcgen05.encode_instr_descriptor(
                    T.address_of(descI),
                    d_dtype="float32",
                    a_dtype=a_type,
                    b_dtype=b_type,
                    M=MMA_M,
                    N=MMA_N,
                    K=MMA_K,
                    trans_a=False,
                    trans_b=False,
                    n_cta_groups=CTA_GROUP,
                )
            )
            tmem = _builder_assign(
                "tmem",
                tmem_pool.alloc((128, N_COLS), "float32"),
                locals().get("tmem", _BUILDER_MISSING),
            )
            _builder_emit(tmem_pool.commit())
            _builder_emit(T.ptx.barrier.cluster.arrive())
            _builder_emit(T.ptx.barrier.cluster.wait())
            _builder_emit(T.cuda.cta_sync())
            _builder_emit(T.ptx.ld.shared.u32(tmem_addr_local, tmem_addr.ptr_to([0])))
            _builder_emit(T.cuda.trap_when_assert_failed(tmem_addr_local == 0))
            _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
            _builder_emit(T.ptx.fence.mbarrier_init.release.cluster())
            _builder_emit(tile_scheduler.init(cbx, bx, rank, warp_id_in_cta, lane_id))
            with T.While(tile_scheduler.valid()):
                with T.If(tile_scheduler.fetched_task_type[0] == TaskType.RS.value):
                    with T.Then():
                        m_idx = tile_scheduler.fetched_task_idx0[0]
                        n_idx = tile_scheduler.fetched_task_idx1[0]
                        offset = _builder_assign(
                            "offset", tid, locals().get("offset", _BUILDER_MISSING)
                        )
                        with T.While(True):
                            with T.If(offset < TILE_M // 2 * TILE_N // 8):
                                with T.Then():
                                    m_start = offset // (TILE_N // 8)
                                    n_start = offset % (TILE_N // 8) * 8
                                    if WORLD_SIZE == 1:
                                        _builder_emit(
                                            T.ptx.ld.global_.v4.b32(
                                                copy_word0,
                                                copy_word1,
                                                copy_word2,
                                                copy_word3,
                                                gemm_out.ptr_to(
                                                    [
                                                        TILE_M * m_idx
                                                        + TILE_M // 2 * cbx
                                                        + m_start,
                                                        TILE_N * n_idx + n_start,
                                                    ]
                                                ),
                                            )
                                        )
                                        _builder_emit(
                                            T.ptx.st.global_.v4.b32(
                                                out.ptr_to(
                                                    [
                                                        TILE_M * m_idx
                                                        + TILE_M // 2 * cbx
                                                        + m_start,
                                                        TILE_N * n_idx + n_start,
                                                    ]
                                                ),
                                                copy_word0,
                                                copy_word1,
                                                copy_word2,
                                                copy_word3,
                                            )
                                        )
                                    else:
                                        _builder_emit(
                                            T.cuda.func_call(
                                                "ld_reduce_8_fp16",
                                                gemm_out.ptr_to(
                                                    [
                                                        rank * LOCAL_M
                                                        + TILE_M * m_idx
                                                        + TILE_M // 2 * cbx
                                                        + m_start,
                                                        TILE_N * n_idx + n_start,
                                                    ]
                                                ),
                                                out.ptr_to(
                                                    [
                                                        TILE_M * m_idx
                                                        + TILE_M // 2 * cbx
                                                        + m_start,
                                                        TILE_N * n_idx + n_start,
                                                    ]
                                                ),
                                                source_code=ld_reduce_8xfp16,
                                            )
                                        )
                                    offset = _builder_assign(
                                        "offset",
                                        offset + NUM_THREADS,
                                        locals().get("offset", _BUILDER_MISSING),
                                    )
                                with T.Else():
                                    T.evaluate(T.break_loop())
                    with T.Else():
                        with T.If(tile_scheduler.fetched_task_type[0] == TaskType.GEMM.value):
                            with T.Then():
                                m_idx = tile_scheduler.fetched_task_idx0[0]
                                n_idx = tile_scheduler.fetched_task_idx1[0]
                                with T.If(T.bitwise_and(T.LE(2, wg_id), wg_id < 3)):
                                    with T.Then():
                                        _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(56))
                                        with T.If(warp_id == 3):
                                            with T.Then():
                                                with T.If(T.filter(lane_id, T.cuda.elect_sync())):
                                                    with T.Then():
                                                        with T.serial(PIPE_CYCLE) as ko:
                                                            IRBuilder.name("ko", ko)
                                                            with T.unroll(PIPELINE_DEPTH) as ks:
                                                                IRBuilder.name("ks", ks)
                                                                stage = _builder_assign(
                                                                    "stage",
                                                                    ko * PIPELINE_DEPTH + ks,
                                                                    locals().get(
                                                                        "stage", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                _builder_emit(
                                                                    smem_pipe.empty.wait(ks, phase)
                                                                )
                                                                _builder_emit(
                                                                    T.ptx[_TMA_G2S_CG2](
                                                                        A_smem.ptr_to(
                                                                            [ks, 0, 0, 0]
                                                                        ),
                                                                        T.address_of(A_tensor_map),
                                                                        stage * BLK_K,
                                                                        (
                                                                            m_idx
                                                                            * NUM_CONSUMER
                                                                            * CTA_GROUP
                                                                            + cbx
                                                                        )
                                                                        * BLK_M,
                                                                        tma_finished.ptr_to([ks]),
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    T.ptx[_TMA_G2S_CG2](
                                                                        A_smem.ptr_to(
                                                                            [ks, 1, 0, 0]
                                                                        ),
                                                                        T.address_of(A_tensor_map),
                                                                        stage * BLK_K,
                                                                        (
                                                                            m_idx
                                                                            * NUM_CONSUMER
                                                                            * CTA_GROUP
                                                                            + CTA_GROUP
                                                                            + cbx
                                                                        )
                                                                        * BLK_M,
                                                                        tma_finished.ptr_to([ks]),
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    T.ptx[_TMA_G2S_CG2](
                                                                        B_smem.ptr_to([ks, 0, 0]),
                                                                        T.address_of(B_tensor_map),
                                                                        stage * BLK_K,
                                                                        (n_idx * CTA_GROUP + cbx)
                                                                        * BLK_N,
                                                                        tma_finished.ptr_to([ks]),
                                                                    )
                                                                )
                                                                with T.If(cbx == 0):
                                                                    with T.Then():
                                                                        _builder_emit(
                                                                            smem_pipe.full.arrive(
                                                                                ks,
                                                                                NUM_CONSUMER
                                                                                * BLK_K
                                                                                * (
                                                                                    BLK_M
                                                                                    * NUM_CONSUMER
                                                                                    + BLK_N
                                                                                )
                                                                                * F16_BYTES,
                                                                            )
                                                                        )
                                                            phase = _builder_assign(
                                                                "phase",
                                                                phase ^ 1,
                                                                locals().get(
                                                                    "phase", _BUILDER_MISSING
                                                                ),
                                                            )
                                                        if PIPE_REMAIN_NUM > 0:
                                                            with T.unroll(PIPE_REMAIN_NUM) as ks:
                                                                IRBuilder.name("ks", ks)
                                                                stage = _builder_assign(
                                                                    "stage",
                                                                    PIPE_CYCLE * PIPELINE_DEPTH
                                                                    + ks,
                                                                    locals().get(
                                                                        "stage", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                _builder_emit(
                                                                    smem_pipe.empty.wait(ks, phase)
                                                                )
                                                                _builder_emit(
                                                                    T.ptx[_TMA_G2S_CG2](
                                                                        A_smem.ptr_to(
                                                                            [ks, 0, 0, 0]
                                                                        ),
                                                                        T.address_of(A_tensor_map),
                                                                        stage * BLK_K,
                                                                        (
                                                                            m_idx
                                                                            * NUM_CONSUMER
                                                                            * CTA_GROUP
                                                                            + cbx
                                                                        )
                                                                        * BLK_M,
                                                                        tma_finished.ptr_to([ks]),
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    T.ptx[_TMA_G2S_CG2](
                                                                        A_smem.ptr_to(
                                                                            [ks, 1, 0, 0]
                                                                        ),
                                                                        T.address_of(A_tensor_map),
                                                                        stage * BLK_K,
                                                                        (
                                                                            m_idx
                                                                            * NUM_CONSUMER
                                                                            * CTA_GROUP
                                                                            + CTA_GROUP
                                                                            + cbx
                                                                        )
                                                                        * BLK_M,
                                                                        tma_finished.ptr_to([ks]),
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    T.ptx[_TMA_G2S_CG2](
                                                                        B_smem.ptr_to([ks, 0, 0]),
                                                                        T.address_of(B_tensor_map),
                                                                        stage * BLK_K,
                                                                        (n_idx * CTA_GROUP + cbx)
                                                                        * BLK_N,
                                                                        tma_finished.ptr_to([ks]),
                                                                    )
                                                                )
                                                                with T.If(cbx == 0):
                                                                    with T.Then():
                                                                        _builder_emit(
                                                                            smem_pipe.full.arrive(
                                                                                ks,
                                                                                NUM_CONSUMER
                                                                                * BLK_K
                                                                                * (
                                                                                    BLK_M
                                                                                    * NUM_CONSUMER
                                                                                    + BLK_N
                                                                                )
                                                                                * F16_BYTES,
                                                                            )
                                                                        )
                                                            with T.unroll(
                                                                PIPE_REMAIN_NUM, PIPELINE_DEPTH
                                                            ) as ks:
                                                                IRBuilder.name("ks", ks)
                                                                _builder_emit(
                                                                    smem_pipe.empty.wait(ks, phase)
                                                                )
                                                                with T.If(cbx == 0):
                                                                    with T.Then():
                                                                        _builder_emit(
                                                                            smem_pipe.full.arrive(
                                                                                ks, remote=0
                                                                            )
                                                                        )
                                                            phase = _builder_assign(
                                                                "phase",
                                                                phase ^ 1,
                                                                locals().get(
                                                                    "phase", _BUILDER_MISSING
                                                                ),
                                                            )
                                            with T.Else():
                                                with T.If((warp_id < 2) & (cbx == 0)):
                                                    with T.Then():
                                                        with T.If(
                                                            T.filter(lane_id, T.cuda.elect_sync())
                                                        ):
                                                            with T.Then():
                                                                _builder_emit(
                                                                    tmem_pipe.empty.wait(
                                                                        warp_id, phase_tmem
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    T.ptx.tcgen05.fence__after_thread_sync()
                                                                )
                                                                with T.serial(PIPE_CYCLE) as ko:
                                                                    IRBuilder.name("ko", ko)
                                                                    with T.unroll(
                                                                        PIPELINE_DEPTH
                                                                    ) as ks:
                                                                        IRBuilder.name("ks", ks)
                                                                        stage = _builder_assign(
                                                                            "stage",
                                                                            ko * PIPELINE_DEPTH
                                                                            + ks,
                                                                            locals().get(
                                                                                "stage",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        _builder_emit(
                                                                            smem_pipe.full.wait(
                                                                                ks, phase
                                                                            )
                                                                        )
                                                                        with T.unroll(
                                                                            BLK_K // MMA_K
                                                                        ) as ki:
                                                                            IRBuilder.name("ki", ki)
                                                                            _builder_emit(
                                                                                T.cuda.tcgen05.encode_matrix_descriptor(
                                                                                    T.address_of(
                                                                                        descA
                                                                                    ),
                                                                                    A_smem.ptr_to(
                                                                                        [
                                                                                            ks,
                                                                                            warp_id,
                                                                                            0,
                                                                                            ki
                                                                                            * MMA_K,
                                                                                        ]
                                                                                    ),
                                                                                    ldo=1,
                                                                                    sdo=8
                                                                                    * BLK_K
                                                                                    * F16_BYTES
                                                                                    // F128_BYTES,
                                                                                    swizzle=SWIZZLE,
                                                                                )
                                                                            )
                                                                            _builder_emit(
                                                                                T.cuda.tcgen05.encode_matrix_descriptor(
                                                                                    T.address_of(
                                                                                        descB
                                                                                    ),
                                                                                    B_smem.ptr_to(
                                                                                        [
                                                                                            ks,
                                                                                            0,
                                                                                            ki
                                                                                            * MMA_K,
                                                                                        ]
                                                                                    ),
                                                                                    ldo=1,
                                                                                    sdo=8
                                                                                    * BLK_K
                                                                                    * F16_BYTES
                                                                                    // F128_BYTES,
                                                                                    swizzle=SWIZZLE,
                                                                                )
                                                                            )
                                                                            with T.If(
                                                                                T.And(
                                                                                    stage == 0,
                                                                                    ki == 0,
                                                                                )
                                                                            ):
                                                                                with T.Then():
                                                                                    _builder_emit(
                                                                                        T.ptx[
                                                                                            _MMA_CHAIN
                                                                                        ](
                                                                                            T.cast(
                                                                                                warp_id
                                                                                                * MMA_N,
                                                                                                "uint32",
                                                                                            ),
                                                                                            descA,
                                                                                            descB,
                                                                                            descI,
                                                                                            *_MMA_ZERO_MASKS,
                                                                                            False,
                                                                                        )
                                                                                    )
                                                                                with T.Else():
                                                                                    _builder_emit(
                                                                                        T.ptx[
                                                                                            _MMA_CHAIN
                                                                                        ](
                                                                                            T.cast(
                                                                                                warp_id
                                                                                                * MMA_N,
                                                                                                "uint32",
                                                                                            ),
                                                                                            descA,
                                                                                            descB,
                                                                                            descI,
                                                                                            *_MMA_ZERO_MASKS,
                                                                                            True,
                                                                                        )
                                                                                    )
                                                                        _builder_emit(
                                                                            smem_pipe.empty.arrive(
                                                                                ks,
                                                                                cta_group=CTA_GROUP,
                                                                                cta_mask=3,
                                                                            )
                                                                        )
                                                                    phase = _builder_assign(
                                                                        "phase",
                                                                        phase ^ 1,
                                                                        locals().get(
                                                                            "phase",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                if PIPE_REMAIN_NUM > 0:
                                                                    with T.unroll(
                                                                        PIPE_REMAIN_NUM
                                                                    ) as ks:
                                                                        IRBuilder.name("ks", ks)
                                                                        _builder_emit(
                                                                            smem_pipe.full.wait(
                                                                                ks, phase
                                                                            )
                                                                        )
                                                                        with T.unroll(
                                                                            BLK_K // MMA_K
                                                                        ) as ki:
                                                                            IRBuilder.name("ki", ki)
                                                                            _builder_emit(
                                                                                T.cuda.tcgen05.encode_matrix_descriptor(
                                                                                    T.address_of(
                                                                                        descA
                                                                                    ),
                                                                                    A_smem.ptr_to(
                                                                                        [
                                                                                            ks,
                                                                                            warp_id,
                                                                                            0,
                                                                                            ki
                                                                                            * MMA_K,
                                                                                        ]
                                                                                    ),
                                                                                    ldo=1,
                                                                                    sdo=8
                                                                                    * BLK_K
                                                                                    * F16_BYTES
                                                                                    // F128_BYTES,
                                                                                    swizzle=SWIZZLE,
                                                                                )
                                                                            )
                                                                            _builder_emit(
                                                                                T.cuda.tcgen05.encode_matrix_descriptor(
                                                                                    T.address_of(
                                                                                        descB
                                                                                    ),
                                                                                    B_smem.ptr_to(
                                                                                        [
                                                                                            ks,
                                                                                            0,
                                                                                            ki
                                                                                            * MMA_K,
                                                                                        ]
                                                                                    ),
                                                                                    ldo=1,
                                                                                    sdo=8
                                                                                    * BLK_K
                                                                                    * F16_BYTES
                                                                                    // F128_BYTES,
                                                                                    swizzle=SWIZZLE,
                                                                                )
                                                                            )
                                                                            with T.If(
                                                                                T.And(
                                                                                    T.And(
                                                                                        PIPE_CYCLE
                                                                                        == 0,
                                                                                        ks == 0,
                                                                                    ),
                                                                                    ki == 0,
                                                                                )
                                                                            ):
                                                                                with T.Then():
                                                                                    _builder_emit(
                                                                                        T.ptx[
                                                                                            _MMA_CHAIN
                                                                                        ](
                                                                                            T.cast(
                                                                                                warp_id
                                                                                                * MMA_N,
                                                                                                "uint32",
                                                                                            ),
                                                                                            descA,
                                                                                            descB,
                                                                                            descI,
                                                                                            *_MMA_ZERO_MASKS,
                                                                                            False,
                                                                                        )
                                                                                    )
                                                                                with T.Else():
                                                                                    _builder_emit(
                                                                                        T.ptx[
                                                                                            _MMA_CHAIN
                                                                                        ](
                                                                                            T.cast(
                                                                                                warp_id
                                                                                                * MMA_N,
                                                                                                "uint32",
                                                                                            ),
                                                                                            descA,
                                                                                            descB,
                                                                                            descI,
                                                                                            *_MMA_ZERO_MASKS,
                                                                                            True,
                                                                                        )
                                                                                    )
                                                                        _builder_emit(
                                                                            smem_pipe.empty.arrive(
                                                                                ks,
                                                                                cta_group=CTA_GROUP,
                                                                                cta_mask=3,
                                                                            )
                                                                        )
                                                                    _builder_emit(
                                                                        tmem_pipe.full.arrive(
                                                                            warp_id,
                                                                            cta_group=CTA_GROUP,
                                                                            cta_mask=3,
                                                                        )
                                                                    )
                                                                    with T.unroll(
                                                                        PIPE_REMAIN_NUM,
                                                                        PIPELINE_DEPTH,
                                                                    ) as ks:
                                                                        IRBuilder.name("ks", ks)
                                                                        _builder_emit(
                                                                            smem_pipe.full.wait(
                                                                                ks, phase
                                                                            )
                                                                        )
                                                                        _builder_emit(
                                                                            smem_pipe.empty.arrive(
                                                                                ks,
                                                                                cta_group=CTA_GROUP,
                                                                                cta_mask=3,
                                                                            )
                                                                        )
                                                                    phase = _builder_assign(
                                                                        "phase",
                                                                        phase ^ 1,
                                                                        locals().get(
                                                                            "phase",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                else:
                                                                    _builder_emit(
                                                                        tmem_pipe.full.arrive(
                                                                            warp_id,
                                                                            cta_group=CTA_GROUP,
                                                                            cta_mask=3,
                                                                        )
                                                                    )
                                                                phase_tmem = _builder_assign(
                                                                    "phase_tmem",
                                                                    phase_tmem ^ 1,
                                                                    locals().get(
                                                                        "phase_tmem",
                                                                        _BUILDER_MISSING,
                                                                    ),
                                                                )
                                with T.If(T.bitwise_and(T.LE(0, wg_id), wg_id < 2)):
                                    with T.Then():
                                        _builder_emit(T.ptx.setmaxnreg.inc.sync.aligned.u32(224))
                                        _builder_emit(tmem_pipe.full.wait(wg_id, phase_tmem))
                                        phase_tmem = _builder_assign(
                                            "phase_tmem",
                                            phase_tmem ^ 1,
                                            locals().get("phase_tmem", _BUILDER_MISSING),
                                        )
                                        _builder_emit(T.ptx.tcgen05.fence__after_thread_sync())
                                        with T.unroll(MMA_N // TMEM_LD_SIZE) as i:
                                            IRBuilder.name("i", i)
                                            col_st = wg_id * MMA_N + i * TMEM_LD_SIZE
                                            _builder_emit(
                                                T.ptx[_TMEM_LD_64](
                                                    *[reg[j] for j in range(TMEM_LD_SIZE)],
                                                    T.cast(col_st, "uint32"),
                                                )
                                            )
                                            _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                                            with T.unroll(TMEM_LD_SIZE // 2) as j:
                                                IRBuilder.name("j", j)
                                                _builder_emit(
                                                    T.ptx[_CVT_F32X2](
                                                        reg_fp16[j], reg[j * 2 + 1], reg[j * 2]
                                                    )
                                                )
                                            with T.unroll(EPI_TILE // 8) as jv:
                                                IRBuilder.name("jv", jv)
                                                r0 = jv * 4
                                                _builder_emit(
                                                    T.ptx.st.shared.v4.u32(
                                                        D_smem.ptr_to(
                                                            [wg_id, warp_id * 32 + lane_id, jv * 8]
                                                        ),
                                                        reg_fp16[r0],
                                                        reg_fp16[r0 + 1],
                                                        reg_fp16[r0 + 2],
                                                        reg_fp16[r0 + 3],
                                                    )
                                                )
                                            _builder_emit(T.cuda.warpgroup_sync(wg_id))
                                            _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                            with T.If((lane_id == 0) & (warp_id == 0)):
                                                with T.Then():
                                                    _builder_emit(
                                                        T.ptx[_TMA_S2G](
                                                            T.address_of(D_tensor_map),
                                                            n_idx * BLK_N * CTA_GROUP
                                                            + i * EPI_TILE,
                                                            (
                                                                m_idx * NUM_CONSUMER * CTA_GROUP
                                                                + wg_id * CTA_GROUP
                                                                + cbx
                                                            )
                                                            * BLK_M,
                                                            D_smem.ptr_to([wg_id, 0, 0]),
                                                        )
                                                    )
                                                    _builder_emit(
                                                        T.ptx.cp.async_.bulk.commit_group()
                                                    )
                                                    _builder_emit(
                                                        T.ptx.cp.async_.bulk.wait_group(0)
                                                    )
                                            _builder_emit(T.cuda.warpgroup_sync(wg_id))
                                        _builder_emit(tmem_pipe.empty.arrive(wg_id, remote=0))
                                        comm_m_idx = m_idx * 2 + wg_id
                                        comm_m_idx_local = comm_m_idx % (LOCAL_M // TILE_M)
                                        signal_rank = comm_m_idx // (LOCAL_M // TILE_M)
                                        _builder_emit(
                                            sem.semaphore_notify(
                                                signal_rank, tid, comm_m_idx_local, n_idx, rs_queue
                                            )
                                        )
                _builder_emit(tile_scheduler.next_tile(cbx, bx, rank, warp_id_in_cta, lane_id))
            _builder_emit(T.ptx.barrier.cluster.arrive())
            _builder_emit(T.ptx.barrier.cluster.wait())
            with T.If((wg_id == 0) & (warp_id == 0)):
                with T.Then():
                    _builder_emit(
                        T.ptx[
                            f"tcgen05.relinquish_alloc_permit.cta_group::{CTA_GROUP}.sync.aligned"
                        ]()
                    )
                    _builder_emit(T.ptx.ld.shared.u32(tmem_addr_local, tmem_addr.ptr_to([0])))
                    _builder_emit(
                        T.ptx[f"tcgen05.dealloc.cta_group::{CTA_GROUP}.sync.aligned.b32"](
                            tmem_addr_local, T.uint32(N_COLS)
                        )
                    )
    return builder.get()


def build_kernel(config: GemmRSConfig | None = None) -> tvm.IRModule:
    """Return the directly ported fused persistent GemmRS kernel."""

    config = config or derive_config()
    requested = (config.M, config.N, config.total_k, config.world_size)
    active = (M, N, TOTAL_K, WORLD_SIZE)
    if requested != active:
        specialized = load_specialized_module(
            package=__package__,
            stem="gemm_reduce_scatter",
            source=__file__,
            key=requested,
            environment={
                _SPECIALIZATION_M_ENV: config.M,
                _SPECIALIZATION_N_ENV: config.N,
                _SPECIALIZATION_K_ENV: config.total_k,
                _SPECIALIZATION_WORLD_SIZE_ENV: config.world_size,
            },
        )
        return specialized.build_kernel()
    return tvm.IRModule({FUSED_DEVICE_ENTRYPOINT: _build_active_prim_func()})


KERNEL_META = {"name": "gemm_reduce_scatter", "category": "basic", "compute_capability": 10}
_RELAUNCH_COUNT = 20

CONFIGS = make_configs(GEMM_RS_MODEL_SHAPES)


def _config(M: int, N: int, K: int, world_size: int, dtype: str, scheduler: str) -> GemmRSConfig:
    if scheduler != "dynamic":
        raise ValueError(f"GEMM+ReduceScatter supports only scheduler='dynamic'; got {scheduler!r}")
    return derive_config(M, N, K, world_size, dtype)


def get_kernel(
    M: int = M,
    N: int = N,
    K: int = TOTAL_K,
    world_size: int = 4,
    dtype: str = DTYPE,
    scheduler: str = "dynamic",
    **_kwargs: Any,
) -> tvm.IRModule:
    """Build the hand-transcribed fused kernel directly, without the megakernel DSL."""

    config = _config(M, N, K, world_size, dtype, scheduler)
    return build_kernel(config)


def _get_benchmark_kernel(
    M: int = M,
    N: int = N,
    K: int = TOTAL_K,
    world_size: int = 4,
    dtype: str = DTYPE,
    scheduler: str = "dynamic",
) -> tvm.IRModule:
    return get_kernel(M, N, K, world_size, dtype, scheduler=scheduler)


def prepare_data(
    M: int = M,
    N: int = N,
    K: int = TOTAL_K,
    world_size: int = 4,
    dtype: str = DTYPE,
    scheduler: str = "dynamic",
    seed: int = 42,
    scale: float = 0.02,
    rank: int = 0,
    **_kwargs: Any,
) -> dict[str, torch.Tensor]:
    config = _config(M, N, K, world_size, dtype, scheduler)
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    device = torch.device("cuda", rank)
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    A = torch.randn(
        (config.M, config.k_local), dtype=torch.float16, device=device, generator=generator
    ).mul_(scale)
    B = torch.randn(
        (config.N, config.k_local), dtype=torch.float16, device=device, generator=generator
    ).mul_(scale)
    return {"A": A, "B": B}


def _queue_state(config: GemmRSConfig) -> tuple[np.ndarray, ...]:
    tasks = [
        (m_idx, n_idx)
        for group_begin in range(0, config.gemm_n_clusters, GROUP_SIZE)
        for m_idx in range(config.gemm_m_clusters)
        for n_idx in range(group_begin, min(group_begin + GROUP_SIZE, config.gemm_n_clusters))
    ]
    if len(tasks) != config.gemm_task_count:
        raise AssertionError("GEMM queue does not have exact group-major coverage")

    gemm_types = np.full((config.world_size, CAPACITY), -1, dtype=np.int32)
    gemm_indices = np.zeros((config.world_size, CAPACITY, TASK_IDX_LEN), dtype=np.int32)
    gemm_heads = np.zeros((config.world_size, 1), dtype=np.int32)
    gemm_tails = np.full((config.world_size, 1), config.gemm_task_count, dtype=np.int32)
    rs_types = np.full((config.world_size, CAPACITY), -1, dtype=np.int32)
    rs_indices = np.zeros((config.world_size, CAPACITY, TASK_IDX_LEN), dtype=np.int32)
    rs_heads = np.zeros((config.world_size, 1), dtype=np.int32)
    rs_tails = np.zeros((config.world_size, 1), dtype=np.int32)
    for rank in range(config.world_size):
        gemm_types[rank, : config.gemm_task_count] = TaskType.GEMM.value
        gemm_indices[rank, : config.gemm_task_count] = tasks
    return (
        gemm_types,
        gemm_indices,
        gemm_heads,
        gemm_tails,
        rs_types,
        rs_indices,
        rs_heads,
        rs_tails,
    )


_manual_queue_state = _queue_state


@dataclass
class _Case:
    runtime: DistributedRuntime
    module: Any
    config: GemmRSConfig
    A: torch.Tensor
    B: torch.Tensor
    gemm_out: Any
    semaphore: Any
    out: torch.Tensor
    gemm_task_types: Any
    gemm_task_idxs: Any
    gemm_head: Any
    gemm_tail: Any
    rs_task_types: Any
    rs_task_idxs: Any
    rs_head: Any
    rs_tail: Any
    gemm_out_torch: torch.Tensor
    semaphore_torch: torch.Tensor
    gemm_types_torch: torch.Tensor
    gemm_head_torch: torch.Tensor
    gemm_tail_torch: torch.Tensor
    rs_types_torch: torch.Tensor
    rs_head_torch: torch.Tensor
    rs_tail_torch: torch.Tensor
    initial_queues: tuple[torch.Tensor, ...]

    def reset(self) -> None:
        # A completed local launch does not imply that peer kernels have stopped
        # writing this rank's symmetric queue state.  Join every rank before
        # overwriting it; prepare() supplies the matching post-reset barrier.
        barrier_on_compute_stream(self.runtime)
        (
            gemm_types,
            gemm_indices,
            gemm_heads,
            gemm_tails,
            rs_types,
            rs_indices,
            rs_heads,
            rs_tails,
        ) = self.initial_queues
        torch_view(self.gemm_task_types).copy_(gemm_types)
        torch_view(self.gemm_task_idxs).copy_(gemm_indices)
        self.gemm_head_torch.copy_(gemm_heads)
        self.gemm_tail_torch.copy_(gemm_tails)
        torch_view(self.rs_task_types).copy_(rs_types)
        torch_view(self.rs_task_idxs).copy_(rs_indices)
        self.rs_head_torch.copy_(rs_heads)
        self.rs_tail_torch.copy_(rs_tails)
        self.semaphore_torch.zero_()
        self.gemm_out_torch.fill_(float("nan"))
        self.out.fill_(float("nan"))

    def prepare(self) -> None:
        barrier_on_compute_stream(self.runtime)

    def launch(self) -> None:
        self.module[FUSED_DEVICE_ENTRYPOINT](
            self.A,
            self.B,
            self.gemm_out,
            self.semaphore,
            self.out,
            self.gemm_task_types,
            self.gemm_task_idxs,
            self.gemm_head,
            self.gemm_tail,
            self.rs_task_types,
            self.rs_task_idxs,
            self.rs_head,
            self.rs_tail,
        )

    def assert_terminal_state(self) -> None:
        if int(self.gemm_tail_torch.item()) != self.config.gemm_task_count:
            raise AssertionError("GEMM queue tail changed unexpectedly")
        if int(self.rs_tail_torch.item()) != self.config.rs_task_count:
            raise AssertionError("RS queue did not publish every tile")
        if int(self.gemm_head_torch.item()) < self.config.gemm_task_count:
            raise AssertionError("GEMM queue did not consume every task")
        if int(self.rs_head_torch.item()) < self.config.rs_task_count:
            raise AssertionError("RS queue did not consume every task")
        if torch.any(self.gemm_types_torch[: self.config.gemm_task_count] != -1):
            raise AssertionError("GEMM queue retains an unconsumed task")
        if torch.any(self.rs_types_torch[: self.config.rs_task_count] != -1):
            raise AssertionError("RS queue retains an unconsumed task")
        torch.testing.assert_close(
            self.semaphore_torch,
            torch.full_like(self.semaphore_torch, self.config.completion_count),
        )
        if torch.isnan(self.gemm_out_torch).any() or torch.isnan(self.out).any():
            raise AssertionError("GemmRS output contains an uncovered tile")


def _allocate_case(
    runtime: DistributedRuntime, module: Any, data: dict[str, torch.Tensor], config: GemmRSConfig
) -> _Case:
    queue_state = _queue_state(config)
    device = torch.device("cuda", runtime.device_index)
    gemm_out = symmetric_empty(runtime, (config.M, config.N), config.dtype)
    semaphore = symmetric_empty(runtime, (config.rs_m_clusters, config.rs_n_clusters), "uint64")
    gemm_task_types = torch.empty((CAPACITY,), dtype=torch.int32, device=device)
    gemm_task_idxs = torch.empty((CAPACITY, TASK_IDX_LEN), dtype=torch.int32, device=device)
    gemm_head = torch.empty((1,), dtype=torch.int32, device=device)
    gemm_tail = torch.empty((1,), dtype=torch.int32, device=device)
    rs_task_types = symmetric_empty(runtime, (CAPACITY,), "int32")
    rs_task_idxs = symmetric_empty(runtime, (CAPACITY, TASK_IDX_LEN), "int32")
    rs_head = symmetric_empty(runtime, (1,), "int32")
    rs_tail = symmetric_empty(runtime, (1,), "int32")
    initial_queues = tuple(
        torch.from_numpy(array[runtime.rank].copy()).to(device) for array in queue_state
    )
    case = _Case(
        runtime=runtime,
        module=module,
        config=config,
        A=data["A"],
        B=data["B"],
        gemm_out=gemm_out,
        semaphore=semaphore,
        out=torch.empty((config.local_m, config.N), dtype=torch.float16, device=device),
        gemm_task_types=gemm_task_types,
        gemm_task_idxs=gemm_task_idxs,
        gemm_head=gemm_head,
        gemm_tail=gemm_tail,
        rs_task_types=rs_task_types,
        rs_task_idxs=rs_task_idxs,
        rs_head=rs_head,
        rs_tail=rs_tail,
        gemm_out_torch=torch_view(gemm_out),
        semaphore_torch=torch_view(semaphore),
        gemm_types_torch=torch_view(gemm_task_types),
        gemm_head_torch=torch_view(gemm_head),
        gemm_tail_torch=torch_view(gemm_tail),
        rs_types_torch=torch_view(rs_task_types),
        rs_head_torch=torch_view(rs_head),
        rs_tail_torch=torch_view(rs_tail),
        initial_queues=initial_queues,
    )
    if config.world_size > 1:
        require_nvls_multicast(runtime, case.gemm_out_torch)
    with torch.cuda.stream(runtime.timing_stream):
        case.reset()
    runtime.timing_stream.synchronize()
    runtime.barrier()
    return case


def _reference_outputs(
    runtime: DistributedRuntime, data: dict[str, torch.Tensor], config: GemmRSConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    partial = torch.mm(data["A"], data["B"].T)
    expected = torch.empty(
        (config.local_m, config.N), dtype=torch.float16, device=f"cuda:{runtime.device_index}"
    )
    dist.reduce_scatter_tensor(expected, partial, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(runtime.device_index)
    return partial, expected


def _check_correctness(case: _Case, partial: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(case.gemm_out_torch, partial, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(case.out, expected, rtol=1e-2, atol=1e-2)
    case.assert_terminal_state()


def _run_worker(
    runtime: DistributedRuntime, module: Any, mode: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    config = _config(
        kwargs["M"],
        kwargs["N"],
        kwargs["K"],
        kwargs["world_size"],
        kwargs["dtype"],
        kwargs.get("scheduler", "dynamic"),
    )
    data = prepare_data(rank=runtime.rank, **kwargs)
    case = _allocate_case(runtime, module, data, config)

    if mode == "test":
        partial, expected = _reference_outputs(runtime, data, config)
        for _ in range(_RELAUNCH_COUNT):
            with torch.cuda.stream(runtime.timing_stream):
                case.reset()
                case.prepare()
                case.launch()
            runtime.timing_stream.synchronize()
            _check_correctness(case, partial, expected)
            runtime.barrier()
        return {
            "status": "OK",
            "relaunches": _RELAUNCH_COUNT,
            "gemm_tasks": config.gemm_task_count,
            "rs_tasks": config.rs_task_count,
        }
    if mode != "bench":
        raise ValueError(f"unsupported distributed worker mode {mode!r}")

    from tirx_kernels.runner import bench

    def prepare() -> None:
        case.reset()
        case.prepare()

    baselines = create_baseline_suite(
        runtime,
        data,
        workload="gemm_reduce_scatter",
        M=config.M,
        N=config.N,
        K=config.total_k,
        world_size=config.world_size,
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
        result["ratios"] = baseline_ratios(result, tirx="tirx")
        result["performance_gate"] = {
            "required_ratio": "> 1",
            "passed": all(
                ratio > 1 for name, ratio in result["ratios"].items() if name.startswith("cublas")
            ),
        }
        return {"status": "OK", **result}
    finally:
        baselines.close()


def run_test(
    M: int = M,
    N: int = N,
    K: int = TOTAL_K,
    world_size: int = 4,
    dtype: str = DTYPE,
    seed: int = 42,
    scheduler: str = "dynamic",
    **_kwargs: Any,
) -> None:
    """Validate the direct port for 20 reset/relaunch cycles."""

    _config(M, N, K, world_size, dtype, scheduler)
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
    K: int = TOTAL_K,
    world_size: int = 4,
    dtype: str = DTYPE,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 1,
    cooldown_s: float = 1.0,
    scheduler: str = "dynamic",
    **_kwargs: Any,
) -> dict[str, Any]:
    """Benchmark the direct port and external baselines."""

    _config(M, N, K, world_size, dtype, scheduler)
    if timer not in {None, "kineto"}:
        raise ValueError("distributed GemmRS supports only timer='kineto'")
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
    K: int = TOTAL_K,
    world_size: int = WORLD_SIZE,
    dtype: str = DTYPE,
    *,
    scheduler: str = "dynamic",
    **_kwargs: Any,
):
    """Compile/export before assignment; ranks start CUDA in run_gpu."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    _config(M, N, K, world_size, dtype, scheduler)
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
    "CAPACITY",
    "CONFIGS",
    "DTYPE",
    "FUSED_DEVICE_ENTRYPOINT",
    "GROUP_SIZE",
    "KERNEL_META",
    "SUPPORTED_WORLD_SIZES",
    "TASK_IDX_LEN",
    "TOTAL_K",
    "GemmRSConfig",
    "M",
    "N",
    "TaskType",
    "_get_benchmark_kernel",
    "_manual_queue_state",
    "_queue_state",
    "build_kernel",
    "derive_config",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
