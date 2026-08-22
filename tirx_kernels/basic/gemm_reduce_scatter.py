# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Direct TP1/TP4 port of the fused persistent dynamic-multimem GemmRS kernel."""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

import tirx_kernels.kern as Kern
import tvm
from tirx_kernels.low_level_ir import NVSHMEM_RUNTIME_FUNC_CALLS
from tvm.ir.type import PointerType, PrimType

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


class TaskType(Enum):
    GEMM = 0
    RS = 1


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
NUM_CLUSTERS = SM_NUMBER // M_CLUSTER
assert NUM_CLUSTERS * M_CLUSTER == SM_NUMBER
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
SM100_SMEM_CAPACITY = 232448
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

    mapped = Kern.local_scalar("uint64")
    Kern.ptx.mapa.u64(mapped, ptr, Kern.uint32(rank))
    return mapped


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
exit_barrier_arrive_and_wait = """
__forceinline__ __device__ void exit_barrier_arrive_and_wait(
        uint32_t* flags, int32_t local_expected, int32_t rank_expected) {
    __threadfence_system();
    uint32_t local_arrivals = atomicAdd_system(flags, 1) + 1;
    if (local_arrivals != static_cast<uint32_t>(local_expected)) {
        return;
    }
    uint32_t* rank_arrivals = flags + 1;
    auto mc_rank_arrivals = reinterpret_cast<uint32_t*>(
        nvshmemx_mc_ptr(NVSHMEM_TEAM_WORLD, rank_arrivals));
    asm volatile(
        "multimem.red.release.sys.global.add.u32 [%0], 1;"
        :
        : "l"(mc_rank_arrivals)
        : "memory");
    uint32_t value;
    do {
        asm volatile(
            "ld.global.acquire.sys.b32 %0, [%1];"
            : "=r"(value)
            : "l"(rank_arrivals)
            : "memory");
    } while (value != static_cast<uint32_t>(rank_expected));
}
"""
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


class SchedulerPipeline:
    """One CTA0 producer publishes each task to both cluster CTAs."""

    def __init__(self, smem, cbx):
        self.p2c = Kern.MBarrier(smem.pool, 1)
        self.p2c.init(1)
        self.c2p = Kern.MBarrier(smem.pool, 1, leader=(Kern.thread_id() == 0) & (cbx == 0))
        self.c2p.init(C2P_THREAD_COUNT)
        self.p2c_state = Kern.PipelineState(1, phase=0)
        self.c2p_state = Kern.PipelineState(1, phase=1)


def int_var(scope="local", dtype="int32", align=4):
    buf = Kern.alloc_buffer([1], dtype, scope=scope, align=align)
    return buf


class MPMCQueue:
    def __init__(
        self,
        capacity: int,
        task_types: Kern.Buffer,
        task_idxs: Kern.Buffer,
        head: Kern.Buffer,
        tail: Kern.Buffer,
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
        Kern.cuda.func_call(
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


class GEMMMPMCQueue(MPMCQueue):
    def dequeue(
        self,
        fetched_task_type: Kern.Buffer,
        fetched_task_idx0: Kern.Buffer,
        fetched_task_idx1: Kern.Buffer,
    ):
        Kern.ptx.atom.global_.add.s32(self.head_r[0], self.head.ptr_to([0]), Kern.int32(1))
        with Kern.If(self.head_r[0] < self.num_tot_tasks):
            with Kern.Then():
                Kern.assign(self.masked_pos[0], self.head_r[0] & self.mask)
                Kern.ptx.ld.acquire.sys.global_.b32(
                    fetched_task_type[0], self.task_types.ptr_to([self.masked_pos[0]])
                )
                with Kern.While(fetched_task_type[0] < 0):
                    Kern.cuda.nano_sleep(40)
                    Kern.ptx.ld.acquire.sys.global_.b32(
                        fetched_task_type[0], self.task_types.ptr_to([self.masked_pos[0]])
                    )
                Kern.ptx.st.global_.s32(
                    self.task_types.ptr_to([self.masked_pos[0]]), Kern.int32(-1)
                )
                Kern.ptx.ld.global_.s32(
                    fetched_task_idx0[0], self.task_idxs.ptr_to([self.masked_pos[0], 0])
                )
                Kern.ptx.ld.global_.s32(
                    fetched_task_idx1[0], self.task_idxs.ptr_to([self.masked_pos[0], 1])
                )
            with Kern.Else():
                Kern.assign(fetched_task_type[0], -1)


class RSMPMCQueue(MPMCQueue):
    def dequeue(
        self,
        fetched_task_type: Kern.Buffer,
        fetched_task_idx0: Kern.Buffer,
        fetched_task_idx1: Kern.Buffer,
        rs_rem: Kern.Buffer,
    ):
        with Kern.If(rs_rem[0] >= 0):
            with Kern.Then():
                Kern.assign(self.head_r[0], rs_rem[0])
                Kern.assign(rs_rem[0], -1)
            with Kern.Else():
                Kern.ptx.atom.global_.add.s32(self.head_r[0], self.head.ptr_to([0]), Kern.int32(1))
        with Kern.If(self.head_r[0] < self.num_tot_tasks):
            with Kern.Then():
                Kern.assign(self.masked_pos[0], self.head_r[0] & self.mask)
                Kern.ptx.ld.global_.acquire.sys.b32(
                    fetched_task_type[0], self.task_types.ptr_to([self.masked_pos[0]])
                )
                with Kern.If(fetched_task_type[0] < 0):
                    with Kern.Then():
                        Kern.assign(rs_rem[0], self.head_r[0])
                    with Kern.Else():
                        Kern.ptx.st.global_.s32(
                            self.task_types.ptr_to([self.masked_pos[0]]), Kern.int32(-1)
                        )
                        Kern.ptx.ld.global_.s32(
                            fetched_task_idx0[0], self.task_idxs.ptr_to([self.masked_pos[0], 0])
                        )
                        Kern.ptx.ld.global_.s32(
                            fetched_task_idx1[0], self.task_idxs.ptr_to([self.masked_pos[0], 1])
                        )
            with Kern.Else():
                Kern.assign(fetched_task_type[0], -1)


class MixedDynamicTileScheduler:
    def __init__(
        self,
        gemm_queue: GEMMMPMCQueue,
        rs_queue: RSMPMCQueue,
        packed_value: Kern.Buffer,
        sch_pipe: SchedulerPipeline,
    ):
        self.gemm_queue = gemm_queue
        self.rs_queue = rs_queue
        self.sch_pipe = sch_pipe
        self.fetched_task_type = int_var()
        self.fetched_task_idx0 = int_var()
        self.fetched_task_idx1 = int_var()
        self.rs_rem = int_var()
        self.packed_value = packed_value

    def publish(self, cbx, lane_id):
        with Kern.If(lane_id == 0):
            with Kern.Then():
                with Kern.If(cbx == 0):
                    with Kern.Then():
                        self.sch_pipe.c2p.wait(0, self.sch_pipe.c2p_state.phase)
                        self.rs_queue.dequeue(
                            self.fetched_task_type,
                            self.fetched_task_idx0,
                            self.fetched_task_idx1,
                            self.rs_rem,
                        )
                        with Kern.If(self.fetched_task_type[0] < 0):
                            with Kern.Then():
                                self.gemm_queue.dequeue(
                                    self.fetched_task_type,
                                    self.fetched_task_idx0,
                                    self.fetched_task_idx1,
                                )
                        Kern.ptx.st.shared__cluster.v4.b32(
                            self.packed_value.ptr_to([0]),
                            self.rs_rem[0],
                            self.fetched_task_type[0],
                            self.fetched_task_idx0[0],
                            self.fetched_task_idx1[0],
                        )
                        Kern.cuda.thread_fence()
                        self.sch_pipe.p2c.arrive(0, remote=0)
                        self.sch_pipe.p2c.arrive(0, remote=1)
                        self.sch_pipe.c2p_state.advance()

    def receive(self):
        self.sch_pipe.p2c.wait(0, self.sch_pipe.p2c_state.phase)
        Kern.ptx.ld.shared__cluster.v4.b32(
            self.rs_rem[0],
            self.fetched_task_type[0],
            self.fetched_task_idx0[0],
            self.fetched_task_idx1[0],
            self.packed_value.ptr_to([0]),
        )
        self.sch_pipe.c2p.arrive(0, remote=0)
        self.sch_pipe.p2c_state.advance()

    def init(self):
        Kern.assign(self.rs_rem[0], -1)

    def valid(self):
        return tvm.tirx.any(self.fetched_task_type[0] >= 0, self.rs_rem[0] >= 0)


class Semaphore:
    def __init__(self, cnt, buffer):
        self.cnt = cnt
        self.sem = buffer
        self.state = Kern.alloc_buffer([1], "uint64", scope="local", align=8)

    def semaphore_notify(self, signal_rank, tid, m_idx, n_idx, rs_queue):
        with Kern.If(tid % 128 == 0):
            with Kern.Then():
                Kern.ptx.fence.sc.sys()
                Kern.assign(
                    self.state[0],
                    (
                        Kern.cuda.func_call(
                            "semaphore_notify_remote",
                            signal_rank,
                            self.sem.ptr_to([m_idx, n_idx]),
                            Kern.uint64(1),
                            source_code=semaphore_notify_remote,
                            return_type="uint64",
                        )
                        + 1
                    ),
                )
                with Kern.If(self.state[0] == self.cnt):
                    with Kern.Then():
                        rs_queue.enqueue(signal_rank, TaskType.RS.value, m_idx, n_idx)


def _make_device_kernel(config: GemmRSConfig, *, chain_dispatch: bool = False):
    """Trace one direct K specialization with its frozen host ABI."""

    M = config.M
    N = config.N
    K_LOCAL = config.k_local
    WORLD_SIZE = config.world_size
    LOCAL_M = config.local_m
    PIPE_CYCLE = config.pipe_cycle
    PIPE_REMAIN_NUM = config.pipe_remainder
    GEMM_M_CLUSTERS = config.gemm_m_clusters
    GEMM_N_CLUSTERS = config.gemm_n_clusters
    RS_M_CLUSTERS = config.rs_m_clusters
    RS_N_CLUSTERS = config.rs_n_clusters

    def host_prelude(params):
        A_tensor_map = Kern.stack_alloca("tensormap", 1)
        B_tensor_map = Kern.stack_alloca("tensormap", 1)
        D_tensor_map = Kern.stack_alloca("tensormap", 1)
        Kern.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            A_tensor_map,
            a_type,
            2,
            params["A"].data,
            K_LOCAL,
            M,
            K_LOCAL * F16_BYTES,
            BLK_K,
            BLK_M,
            1,
            1,
            0,
            SWIZZLE,
            0,
            0,
        )
        Kern.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            B_tensor_map,
            b_type,
            2,
            params["B"].data,
            K_LOCAL,
            N,
            K_LOCAL * F16_BYTES,
            BLK_K,
            BLK_N,
            1,
            1,
            0,
            SWIZZLE,
            0,
            0,
        )
        Kern.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            D_tensor_map,
            d_type,
            2,
            params["gemm_out"].data,
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
        return A_tensor_map, B_tensor_map, D_tensor_map

    @Kern.kernel(
        warps=NUM_THREADS // 32,
        arch="sm_100a",
        grid=SM_NUMBER,
        host_prelude=host_prelude,
        allowed_func_calls=tuple(NVSHMEM_RUNTIME_FUNC_CALLS),
    )
    def test_mma_ss_tma_2sm_persistent(
        A: Kern.gptr[Kern.f16, (M, K_LOCAL)],
        B: Kern.gptr[Kern.f16, (N, K_LOCAL)],
        gemm_out: Kern.gptr[Kern.f16, (M, N)],
        semaphore: Kern.gptr[Kern.u64, (LOCAL_M // TILE_M, N // TILE_N)],
        out: Kern.gptr[Kern.f16, (LOCAL_M, N)],
        gemm_task_types: Kern.gptr[Kern.i32, (CAPACITY,)],
        gemm_task_idxs: Kern.gptr[Kern.i32, (CAPACITY, 2)],
        gemm_head: Kern.gptr[Kern.i32, (1,)],
        gemm_tail: Kern.gptr[Kern.i32, (1,)],
        rs_task_types: Kern.gptr[Kern.i32, (CAPACITY,)],
        rs_task_idxs: Kern.gptr[Kern.i32, (CAPACITY, 2)],
        rs_head: Kern.gptr[Kern.i32, (1,)],
        rs_tail: Kern.gptr[Kern.i32, (1,)],
        exit_barrier: Kern.gptr[Kern.u32, (2,)],
        *,
        host,
    ):
        A_tensor_map, B_tensor_map, D_tensor_map = host
        gemm_out = gemm_out.view(M, N)
        semaphore = semaphore.view(LOCAL_M // TILE_M, N // TILE_N)
        out = out.view(LOCAL_M, N)
        gemm_task_types = gemm_task_types.view(CAPACITY)
        gemm_task_idxs = gemm_task_idxs.view(CAPACITY, 2)
        gemm_head = gemm_head.view(1)
        gemm_tail = gemm_tail.view(1)
        rs_task_types = rs_task_types.view(CAPACITY)
        rs_task_idxs = rs_task_idxs.view(CAPACITY, 2)
        rs_head = rs_head.view(1)
        rs_tail = rs_tail.view(1)
        exit_barrier = exit_barrier.view(2)

        cbx_expr, _ = Kern.cta_id_in_cluster([M_CLUSTER, N_CLUSTER])
        cbx = cbx_expr
        bx = Kern.cta_id()
        warp_id_in_cta = Kern.warp_id()
        wg_id = warp_id_in_cta // WARP_NUMBER
        warp_id = warp_id_in_cta % WARP_NUMBER
        tid = Kern.thread_id()
        lane_id = Kern.lane_id()
        rank = Kern.alloc_local((1,), Kern.i32)
        Kern.assign(rank[0], Kern.nvshmem.my_pe())

        smem = Kern.smem_pool()
        pool = smem.pool
        tmem_addr = smem.alloc((1,), Kern.u32, align=4)
        smem_pipe = Kern.Pipeline(
            smem,
            PIPELINE_DEPTH,
            full="tma",
            empty="tcgen05",
            init_empty=NUM_CONSUMER,
            empty_phase_offset=1,
        )
        tmem_pipe = Kern.Pipeline(
            smem,
            NUM_CONSUMER,
            full="tcgen05",
            empty="mbar",
            init_empty=128 * NUM_CONSUMER,
            empty_phase_offset=1,
        )
        packed_buf = smem.alloc((4,), Kern.u32, align=16)
        sch_pipe = SchedulerPipeline(smem, cbx)
        pool.move_base_to(SMEM_RESERVED_BYTES)
        A_smem = smem.alloc(
            (PIPELINE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, swizzle=Kern.SW128B
        )
        B_smem = smem.alloc((PIPELINE_DEPTH, BLK_N, BLK_K), b_type, swizzle=Kern.SW128B).buf
        D_smem = smem.alloc((NUM_CONSUMER, BLK_M, EPI_TILE), d_type, swizzle=Kern.SW128B).buf
        if smem.bytes > SM100_SMEM_CAPACITY:
            raise ValueError(
                f"GemmRS shared-memory usage {smem.bytes} exceeds {SM100_SMEM_CAPACITY} bytes"
            )

        reg = Kern.alloc_local((TMEM_LD_SIZE,), Kern.f32)
        reg_fp16 = Kern.alloc_local((TMEM_LD_SIZE // 2,), Kern.u32, align=16)
        copy_words = Kern.alloc_local((4,), Kern.u32)
        descA = Kern.alloc_local((1,), Kern.u64)
        descB = Kern.alloc_local((1,), Kern.u64)
        descI = Kern.alloc_local((1,), Kern.u32)
        stage = Kern.alloc_local((1,), Kern.i32)
        tmem_addr_local = Kern.alloc_local((1,), Kern.u32)
        offset = Kern.alloc_local((1,), Kern.i32)
        sem = Semaphore(cnt=2 * WORLD_SIZE, buffer=semaphore)
        gemm_queue = GEMMMPMCQueue(
            CAPACITY,
            gemm_task_types,
            gemm_task_idxs,
            gemm_head,
            gemm_tail,
            GEMM_M_CLUSTERS * GEMM_N_CLUSTERS,
        )
        rs_queue = RSMPMCQueue(
            CAPACITY, rs_task_types, rs_task_idxs, rs_head, rs_tail, RS_M_CLUSTERS * RS_N_CLUSTERS
        )

        packed_ptr = Kern.reinterpret(
            PointerType(PrimType("uint32")), _mapa_u64_tx(packed_buf.ptr_to([0]), 0)
        )
        packed_value = Kern.decl_buffer((4,), "uint32", data=packed_ptr, scope="shared")
        tile_scheduler = MixedDynamicTileScheduler(gemm_queue, rs_queue, packed_value, sch_pipe)
        smem_full_cta0 = smem_pipe.full.remote_view(0)
        smem_cycle = Kern.PipelineState(1, phase=0)
        tmem_cycle = Kern.PipelineState(1, phase=0)

        specialization = Kern.specialize(chain_dispatch=chain_dispatch)
        consumer = specialization.role("consumer", range(8), regs=None)
        mma_role = specialization.role("mma", [8, 9], regs=None, when=cbx == 0)
        specialization.role("idle", [10], regs=None)
        tma_scheduler_role = specialization.role("tma_scheduler", [11], regs=None)
        producer_regs = specialization.register_scope(
            "producer_regs", warps=range(8, 12), regs=56, direction="dec"
        )
        consumer_regs = specialization.register_scope(
            "consumer_regs", warps=range(8), regs=224, direction="inc"
        )

        def fetch_next():
            with tma_scheduler_role:
                tile_scheduler.publish(cbx, lane_id)
            tile_scheduler.receive()

        Kern.cuda.tcgen05.encode_instr_descriptor(
            Kern.address_of(descI[0]),
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
        with Kern.If((wg_id == 0) & (warp_id == 0)), Kern.Then():
            Kern.ptx[f"tcgen05.alloc.cta_group::{CTA_GROUP}.sync.aligned.shared::cta.b32"](
                Kern.address_of(tmem_addr[0]), Kern.uint32(N_COLS)
            )
            Kern.cuda.warp_sync()
        Kern.ptx.barrier.cluster.arrive()
        Kern.ptx.barrier.cluster.wait()
        Kern.cuda.cta_sync()
        Kern.ptx.ld.shared.u32(tmem_addr_local[0], tmem_addr.ptr_to([0]))
        Kern.cuda.trap_when_assert_failed(tmem_addr_local[0] == 0)
        Kern.ptx.fence.proxy.async_.shared__cta()
        Kern.ptx.fence.mbarrier_init.release.cluster()
        tile_scheduler.init()
        fetch_next()

        def partitioned_loop(main_loop, epilogue1, epilogue2):
            with Kern.serial(PIPE_CYCLE) as ko:
                with Kern.unroll(PIPELINE_DEPTH) as ks:
                    Kern.assign(stage[0], ko * PIPELINE_DEPTH + ks)
                    main_loop(False, ks)
                smem_cycle.advance()
            if PIPE_REMAIN_NUM > 0:
                with Kern.unroll(PIPE_REMAIN_NUM) as ks:
                    Kern.assign(stage[0], PIPE_CYCLE * PIPELINE_DEPTH + ks)
                    main_loop(True, ks)
                epilogue1()
                with Kern.unroll(PIPE_REMAIN_NUM, PIPELINE_DEPTH) as ks:
                    epilogue2(ks)
                smem_cycle.advance()
            else:
                epilogue1()

        with Kern.While(tile_scheduler.valid()):
            with Kern.If(tile_scheduler.fetched_task_type[0] == TaskType.RS.value), Kern.Then():
                m_idx = tile_scheduler.fetched_task_idx0[0]
                n_idx = tile_scheduler.fetched_task_idx1[0]
                Kern.assign(offset[0], tid)
                with Kern.While(offset[0] < TILE_M // 2 * TILE_N // 8):
                    m_start = offset[0] // (TILE_N // 8)
                    n_start = offset[0] % (TILE_N // 8) * 8
                    if WORLD_SIZE == 1:
                        Kern.ptx.ld.global_.v4.b32(
                            copy_words[0],
                            copy_words[1],
                            copy_words[2],
                            copy_words[3],
                            gemm_out.ptr_to(
                                [
                                    TILE_M * m_idx + TILE_M // 2 * cbx + m_start,
                                    TILE_N * n_idx + n_start,
                                ]
                            ),
                        )
                        Kern.ptx.st.global_.v4.b32(
                            out.ptr_to(
                                [
                                    TILE_M * m_idx + TILE_M // 2 * cbx + m_start,
                                    TILE_N * n_idx + n_start,
                                ]
                            ),
                            copy_words[0],
                            copy_words[1],
                            copy_words[2],
                            copy_words[3],
                        )
                    else:
                        Kern.cuda.func_call(
                            "ld_reduce_8_fp16",
                            gemm_out.ptr_to(
                                [
                                    rank[0] * LOCAL_M
                                    + TILE_M * m_idx
                                    + TILE_M // 2 * cbx
                                    + m_start,
                                    TILE_N * n_idx + n_start,
                                ]
                            ),
                            out.ptr_to(
                                [
                                    TILE_M * m_idx + TILE_M // 2 * cbx + m_start,
                                    TILE_N * n_idx + n_start,
                                ]
                            ),
                            source_code=ld_reduce_8xfp16,
                        )
                    Kern.assign(offset[0], offset[0] + NUM_THREADS)

            with Kern.If(tile_scheduler.fetched_task_type[0] == TaskType.GEMM.value), Kern.Then():
                m_idx = tile_scheduler.fetched_task_idx0[0]
                n_idx = tile_scheduler.fetched_task_idx1[0]

                def emit_producer_roles():
                    def tma_body():
                        with Kern.If(Kern.cuda.elect_sync()), Kern.Then():

                            def tma_load(_is_remain, ks):
                                smem_pipe.empty.wait(ks, smem_cycle.phase)
                                # Materialized, not plain: `stage` is a mutable
                                # cursor the pipeline rewrites per iteration, and
                                # this value feeds three TMA coordinates. A plain
                                # name would re-load stage[0] at each of them.
                                stage_k = Kern.local_scalar("int32", init=stage[0] * BLK_K)
                                Kern.ptx[_TMA_G2S_CG2](
                                    A_smem.ptr_to([ks, 0, 0, 0]),
                                    Kern.address_of(A_tensor_map),
                                    stage_k,
                                    (m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M,
                                    smem_full_cta0.ptr_to([ks]),
                                )
                                Kern.ptx[_TMA_G2S_CG2](
                                    A_smem.ptr_to([ks, 1, 0, 0]),
                                    Kern.address_of(A_tensor_map),
                                    stage_k,
                                    (m_idx * NUM_CONSUMER * CTA_GROUP + CTA_GROUP + cbx) * BLK_M,
                                    smem_full_cta0.ptr_to([ks]),
                                )
                                Kern.ptx[_TMA_G2S_CG2](
                                    B_smem.ptr_to([ks, 0, 0]),
                                    Kern.address_of(B_tensor_map),
                                    stage_k,
                                    (n_idx * CTA_GROUP + cbx) * BLK_N,
                                    smem_full_cta0.ptr_to([ks]),
                                )
                                with Kern.If(cbx == 0):
                                    with Kern.Then():
                                        smem_pipe.full.arrive(
                                            ks,
                                            NUM_CONSUMER
                                            * BLK_K
                                            * (BLK_M * NUM_CONSUMER + BLK_N)
                                            * F16_BYTES,
                                        )

                            def tma_done():
                                pass

                            def tma_drain(ks):
                                smem_pipe.empty.wait(ks, smem_cycle.phase)
                                with Kern.If(cbx == 0):
                                    with Kern.Then():
                                        smem_pipe.full.arrive(ks, remote=0)

                            partitioned_loop(tma_load, tma_done, tma_drain)

                    def mma_body():
                        # Preserve the source's warpgroup-local lowering in the
                        # descriptor/TMEM address path.
                        pw = warp_id
                        with Kern.If(Kern.cuda.elect_sync()), Kern.Then():
                            tmem_pipe.empty.wait(pw, tmem_cycle.phase)
                            Kern.ptx.tcgen05.fence__after_thread_sync()

                            def mma(_is_remain, ks):
                                smem_pipe.full.wait(ks, smem_cycle.phase)
                                with Kern.unroll(BLK_K // MMA_K) as ki:
                                    Kern.cuda.tcgen05.encode_matrix_descriptor(
                                        Kern.address_of(descA[0]),
                                        A_smem.ptr_to([ks, pw, 0, ki * MMA_K]),
                                        ldo=1,
                                        sdo=8 * BLK_K * F16_BYTES // F128_BYTES,
                                        swizzle=SWIZZLE,
                                    )
                                    Kern.cuda.tcgen05.encode_matrix_descriptor(
                                        Kern.address_of(descB[0]),
                                        B_smem.ptr_to([ks, 0, ki * MMA_K]),
                                        ldo=1,
                                        sdo=8 * BLK_K * F16_BYTES // F128_BYTES,
                                        swizzle=SWIZZLE,
                                    )
                                    with Kern.If(ki == 0):
                                        with Kern.Then():
                                            with Kern.If(stage[0] == 0):
                                                with Kern.Then():
                                                    Kern.ptx[_MMA_CHAIN](
                                                        Kern.cast(pw * MMA_N, "uint32"),
                                                        descA[0],
                                                        descB[0],
                                                        descI[0],
                                                        *_MMA_ZERO_MASKS,
                                                        False,
                                                    )
                                                with Kern.Else():
                                                    Kern.ptx[_MMA_CHAIN](
                                                        Kern.cast(pw * MMA_N, "uint32"),
                                                        descA[0],
                                                        descB[0],
                                                        descI[0],
                                                        *_MMA_ZERO_MASKS,
                                                        True,
                                                    )
                                        with Kern.Else():
                                            Kern.ptx[_MMA_CHAIN](
                                                Kern.cast(pw * MMA_N, "uint32"),
                                                descA[0],
                                                descB[0],
                                                descI[0],
                                                *_MMA_ZERO_MASKS,
                                                True,
                                            )
                                smem_pipe.empty.arrive(ks, cta_group=CTA_GROUP, cta_mask=3)

                            def mma_done():
                                tmem_pipe.full.arrive(pw, cta_group=CTA_GROUP, cta_mask=3)

                            def mma_drain(ks):
                                smem_pipe.full.wait(ks, smem_cycle.phase)
                                smem_pipe.empty.arrive(ks, cta_group=CTA_GROUP, cta_mask=3)

                            partitioned_loop(mma, mma_done, mma_drain)
                            tmem_cycle.advance()

                    with tma_scheduler_role:
                        tma_body()
                    with mma_role:
                        mma_body()

                # setmaxnreg is warpgroup-collective. Preserve the original
                # producer scope, with K owning its TMA/scheduler/MMA/idle partition.
                with Kern.If(wg_id == NUM_CONSUMER), Kern.Then():
                    producer_regs.emit()
                    emit_producer_roles()

                with consumer:
                    consumer_regs.emit()
                    consumer_warp = Kern.warp_id_in_role()
                    consumer_wg = consumer_warp >> 2
                    consumer_wid = consumer_warp & 3
                    consumer_lane = Kern.lane_id()
                    tmem_pipe.full.wait(consumer_wg, tmem_cycle.phase)
                    tmem_cycle.advance()
                    Kern.ptx.tcgen05.fence__after_thread_sync()
                    for i in range(MMA_N // TMEM_LD_SIZE):
                        col_st = consumer_wg * MMA_N + i * TMEM_LD_SIZE
                        Kern.ptx[_TMEM_LD_64](
                            *[reg[j] for j in range(TMEM_LD_SIZE)], Kern.cast(col_st, "uint32")
                        )
                        Kern.ptx.tcgen05.wait__ld.sync.aligned()
                        for j in range(TMEM_LD_SIZE // 2):
                            Kern.ptx[_CVT_F32X2](reg_fp16[j], reg[j * 2 + 1], reg[j * 2])
                        for jv in range(EPI_TILE // 8):
                            r0 = jv * 4
                            Kern.ptx.st.shared.v4.u32(
                                D_smem.ptr_to(
                                    [consumer_wg, consumer_wid * 32 + consumer_lane, jv * 8]
                                ),
                                reg_fp16[r0],
                                reg_fp16[r0 + 1],
                                reg_fp16[r0 + 2],
                                reg_fp16[r0 + 3],
                            )
                        Kern.cuda.warpgroup_sync(consumer_wg)
                        Kern.ptx.fence.proxy.async_.shared__cta()
                        with Kern.If((consumer_lane == 0) & (consumer_wid == 0)), Kern.Then():
                            Kern.ptx[_TMA_S2G](
                                Kern.address_of(D_tensor_map),
                                n_idx * BLK_N * CTA_GROUP + i * EPI_TILE,
                                (m_idx * NUM_CONSUMER * CTA_GROUP + consumer_wg * CTA_GROUP + cbx)
                                * BLK_M,
                                D_smem.ptr_to([consumer_wg, 0, 0]),
                            )
                            Kern.ptx.cp.async_.bulk.commit_group()
                            Kern.ptx.cp.async_.bulk.wait_group(0)
                        Kern.cuda.warpgroup_sync(consumer_wg)
                    tmem_pipe.empty.arrive(consumer_wg, remote=0)
                    comm_m_idx = m_idx * 2 + consumer_wg
                    comm_m_idx_local = comm_m_idx % (LOCAL_M // TILE_M)
                    signal_rank = comm_m_idx // (LOCAL_M // TILE_M)
                    sem.semaphore_notify(signal_rank, tid, comm_m_idx_local, n_idx, rs_queue)

            fetch_next()

        Kern.ptx.barrier.cluster.arrive()
        Kern.ptx.barrier.cluster.wait()
        if WORLD_SIZE > 1:
            with Kern.If((cbx == 0) & (tid == 0)), Kern.Then():
                Kern.cuda.func_call(
                    "exit_barrier_arrive_and_wait",
                    exit_barrier.ptr_to([0]),
                    Kern.int32(NUM_CLUSTERS),
                    Kern.int32(WORLD_SIZE),
                    source_code=exit_barrier_arrive_and_wait,
                )
            Kern.cuda.cta_sync()
            Kern.ptx.barrier.cluster.arrive()
            Kern.ptx.barrier.cluster.wait()
        # Spell the teardown boundary exactly as the original: shared load to a
        # register, then collective relinquish/dealloc after every local and peer user.
        with Kern.If((wg_id == 0) & (warp_id == 0)), Kern.Then():
            Kern.ptx[f"tcgen05.relinquish_alloc_permit.cta_group::{CTA_GROUP}.sync.aligned"]()
            Kern.ptx.ld.shared.u32(tmem_addr_local[0], tmem_addr.ptr_to([0]))
            Kern.ptx[f"tcgen05.dealloc.cta_group::{CTA_GROUP}.sync.aligned.b32"](
                tmem_addr_local[0], Kern.uint32(N_COLS)
            )

    return test_mma_ss_tma_2sm_persistent


def build_kernel(config: GemmRSConfig | None = None) -> tvm.IRModule:
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
    device = _make_device_kernel(config)
    return tvm.IRModule({FUSED_DEVICE_ENTRYPOINT: device.func})


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
    exit_barrier: Any
    gemm_out_torch: torch.Tensor
    semaphore_torch: torch.Tensor
    gemm_types_torch: torch.Tensor
    gemm_head_torch: torch.Tensor
    gemm_tail_torch: torch.Tensor
    rs_types_torch: torch.Tensor
    rs_head_torch: torch.Tensor
    rs_tail_torch: torch.Tensor
    exit_barrier_torch: torch.Tensor
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
        self.exit_barrier_torch.zero_()
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
            self.exit_barrier,
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
        expected_exit_state = (
            (NUM_CLUSTERS, self.config.world_size) if self.config.world_size > 1 else (0, 0)
        )
        torch.testing.assert_close(
            self.exit_barrier_torch,
            torch.tensor(
                expected_exit_state,
                dtype=self.exit_barrier_torch.dtype,
                device=self.exit_barrier_torch.device,
            ),
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
    exit_barrier = symmetric_empty(runtime, (2,), "uint32")
    exit_barrier_torch = torch_view(exit_barrier)
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
        exit_barrier=exit_barrier,
        gemm_out_torch=torch_view(gemm_out),
        semaphore_torch=torch_view(semaphore),
        gemm_types_torch=torch_view(gemm_task_types),
        gemm_head_torch=torch_view(gemm_head),
        gemm_tail_torch=torch_view(gemm_tail),
        rs_types_torch=torch_view(rs_task_types),
        rs_head_torch=torch_view(rs_head),
        rs_tail_torch=torch_view(rs_tail),
        exit_barrier_torch=exit_barrier_torch,
        initial_queues=initial_queues,
    )
    if config.world_size > 1:
        require_nvls_multicast(runtime, case.gemm_out_torch)
        require_nvls_multicast(runtime, exit_barrier_torch)
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

    from tirx_kernels.runner import bench, external_references_enabled

    def prepare() -> None:
        case.reset()
        case.prepare()

    baselines = None
    if external_references_enabled():
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
            references=baselines.references() if baselines is not None else None,
            timer=kwargs.get("timer", "kineto"),
            rounds=kwargs.get("rounds", 1),
            cooldown_s=kwargs.get("cooldown_s", 1.0),
            distributed=runtime.bench_context(),
            prepare={"tirx": prepare},
        )
        if baselines is not None:
            result["baseline_metadata"] = baselines.metadata()
            result["ratio_definition"] = "baseline_us / tirx_us"
            result["ratios"] = baseline_ratios(result, tirx="tirx")
            result["performance_gate"] = {
                "required_ratio": "> 1",
                "passed": all(
                    ratio > 1
                    for name, ratio in result["ratios"].items()
                    if name.startswith("cublas")
                ),
            }
        return {"status": "OK", **result}
    finally:
        if baselines is not None:
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
