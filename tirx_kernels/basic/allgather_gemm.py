# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Tensor-parallel AllGather + FP16 GEMM for SM100."""

import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

import tirx_kernels.kern as Kern
import tvm
from tvm.ir.type import PointerType, PrimType

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


class TaskType(Enum):
    GEMM = 0
    AG = 1


class ProfileEventType(Enum):
    GEMM = 0
    AG = 1
    FETCH = 2


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
    mapped = Kern.local_scalar("uint64")
    Kern.ptx.mapa.u64(mapped, ptr, Kern.uint32(rank))
    return mapped


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


def _arrive_remote_u64(barrier_ptr, remote_cta):
    """Preserve the source's mapa.u64 + implicit-count remote arrive sequence."""
    mapped = Kern.local_scalar("uint64")
    Kern.ptx.mapa.shared__cluster.u64(mapped, barrier_ptr, Kern.uint32(remote_cta))
    Kern.ptx.mbarrier.arrive.b64(mapped, Kern.uint32(1), pred=Kern.bool(True))


def int_var(name: str, scope="local", dtype="int32", align=4):
    buf = Kern.alloc_buffer([1], dtype, scope=scope, align=align)
    return buf


class Semaphore:
    def __init__(self, cnt, buffer):
        self.cnt = cnt
        self.sem = buffer
        self.state = Kern.alloc_buffer([1], "uint64", scope="local", align=8)

    def semaphore_wait(self, *coord):
        with Kern.While(1):
            Kern.ptx.ld.acquire.gpu.global_.b64(self.state[0], self.sem.ptr_to(list(coord)))
            with Kern.If(self.state[0] == self.cnt):
                with Kern.Then():
                    Kern.Break()
            Kern.cuda.nano_sleep(40)


class MPMCQueue:
    def __init__(
        self,
        capacity: int,
        task_types: Kern.Buffer,
        task_idxs: Kern.Buffer,
        head: Kern.Buffer,
        num_tot_tasks: int,
    ):
        if capacity & (capacity - 1):
            raise ValueError("capacity must be a power-of-two")
        self.mask = capacity - 1
        self.task_types = task_types
        self.task_idxs = task_idxs
        self.head = head
        self.head_r = int_var("head_r")
        self.masked_pos = int_var("masked_pos")
        self.num_tot_tasks = num_tot_tasks


class GEMMMPMCQueue(MPMCQueue):
    def dequeue(
        self,
        fetched_task_type: Kern.Buffer,
        fetched_task_idx0: Kern.Buffer,
        fetched_task_idx1: Kern.Buffer,
        sem: Semaphore,
        rank,
    ):
        Kern.ptx.atom.global_.add.s32(self.head_r[0], self.head.ptr_to([0]), Kern.int32(1))
        with Kern.If(self.head_r[0] < self.num_tot_tasks):
            with Kern.Then():
                # TODO: modify the wait logic to make it faster
                remote_rank = (
                    rank + (self.head_r[0] // (LOCAL_GEMM_M_CLUSTERS * GEMM_N_CLUSTERS))
                ) % WORLD_SIZE
                with Kern.If(remote_rank != rank):
                    with Kern.Then():
                        sem.semaphore_wait(remote_rank)

                Kern.assign(self.masked_pos[0], self.head_r[0] & self.mask)
                Kern.ptx.ld.global_.acquire.gpu.b32(
                    fetched_task_type[0], self.task_types.ptr_to([self.masked_pos[0]])
                )
                with Kern.While(fetched_task_type[0] < 0):
                    Kern.cuda.nano_sleep(40)
                    Kern.ptx.ld.global_.acquire.gpu.b32(
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


class SingleDynamicTileScheduler:
    def __init__(
        self,
        queue: MPMCQueue,
        packed_value: Kern.Buffer,
        sch_pipe,
        producer_state,
        consumer_state,
        sem: Semaphore,
    ):
        self.queue = queue
        self.sch_pipe = sch_pipe
        self.producer_state = producer_state
        self.consumer_state = consumer_state
        self.fetched_task_type = int_var("fetched_task_type")
        self.fetched_task_idx0 = int_var("fetched_task_idx0")
        self.fetched_task_idx1 = int_var("fetched_task_idx1")
        self.sem = sem
        self.rs_rem = int_var("rs_rem")
        self.packed_value = packed_value

    # fmt: off
    def publish(
        self,
        cbx,
        rank,
        lane_id,
    ):
        with Kern.If(lane_id == 0):
            with Kern.Then():
                with Kern.If(cbx == 0):
                    with Kern.Then():
                        self.sch_pipe.empty.wait(
                            self.producer_state.stage, self.producer_state.phase
                        )
                        self.queue.dequeue(
                            self.fetched_task_type,
                            self.fetched_task_idx0,
                            self.fetched_task_idx1,
                            self.sem,
                            rank,
                        )
                        Kern.ptx.st.shared__cluster.v4.b32(
                            self.packed_value.ptr_to([0]),
                            self.rs_rem[0],
                            self.fetched_task_type[0],
                            self.fetched_task_idx0[0],
                            self.fetched_task_idx1[0],
                        )
                        _arrive_remote_u64(
                            self.sch_pipe.full.ptr_to([self.producer_state.stage]), 0
                        )
                        _arrive_remote_u64(
                            self.sch_pipe.full.ptr_to([self.producer_state.stage]), 1
                        )
                        self.producer_state.advance()

    def _fetch(self):
        self.sch_pipe.full.wait(
            self.consumer_state.stage, self.consumer_state.phase
        )
        Kern.ptx.ld.shared__cluster.v4.b32(
            self.rs_rem[0],
            self.fetched_task_type[0],
            self.fetched_task_idx0[0],
            self.fetched_task_idx1[0],
            self.packed_value.ptr_to([0]),
        )
        _arrive_remote_u64(
            self.sch_pipe.empty.ptr_to([self.consumer_state.stage]), 0
        )
        self.consumer_state.advance()

    def receive(self, lane_id):
        if ENABLE_WARP_BROADCAST:
            with Kern.If(lane_id == 0):
                with Kern.Then():
                    self._fetch()
            Kern.ptx.shfl_sync.idx.b32(
                self.rs_rem[0],
                self.rs_rem[0],
                Kern.uint32(0),
                Kern.uint32(31),
                Kern.uint32(0xFFFFFFFF),
            )
            Kern.ptx.shfl_sync.idx.b32(
                self.fetched_task_type[0],
                self.fetched_task_type[0],
                Kern.uint32(0),
                Kern.uint32(31),
                Kern.uint32(0xFFFFFFFF),
            )
            Kern.ptx.shfl_sync.idx.b32(
                self.fetched_task_idx0[0],
                self.fetched_task_idx0[0],
                Kern.uint32(0),
                Kern.uint32(31),
                Kern.uint32(0xFFFFFFFF),
            )
            Kern.ptx.shfl_sync.idx.b32(
                self.fetched_task_idx1[0],
                self.fetched_task_idx1[0],
                Kern.uint32(0),
                Kern.uint32(31),
                Kern.uint32(0xFFFFFFFF),
            )
        else:
            self._fetch()

    def init(self):
        Kern.assign(self.rs_rem[0], -1)

    def valid(self):
        return (self.fetched_task_type[0] >= 0) | (self.rs_rem[0] >= 0)
    # fmt: on


def skip():
    pass


def _host_prelude(params):
    """Encode the six TensorMaps promised by the public PrimFunc ABI."""

    A_tensor_map = Kern.stack_alloca("tensormap", 1)
    A_tensor_map_1 = Kern.stack_alloca("tensormap", 1)
    ag_out_tensor_map = Kern.stack_alloca("tensormap", 1)
    ag_out_tensor_map_1 = Kern.stack_alloca("tensormap", 1)
    B_tensor_map = Kern.stack_alloca("tensormap", 1)
    out_tensor_map = Kern.stack_alloca("tensormap", 1)

    def encode(descriptor, data, dtype, dim0, dim1, stride, box0, box1):
        Kern.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            descriptor,
            dtype,
            2,
            data,
            dim0,
            dim1,
            stride,
            box0,
            box1,
            1,
            1,
            0,
            SWIZZLE,
            2,
            0,
        )

    A = params["A"]
    B = params["B"]
    ag_out = params["ag_out"]
    out = params["out"]
    encode(A_tensor_map, A.data, a_type, K, LOCAL_M, K * F16_BYTES, BLK_K, BLK_M)
    encode(A_tensor_map_1, A.data, a_type, K, LOCAL_M, K * F16_BYTES, BLK_K, BLK_M)
    encode(ag_out_tensor_map, ag_out.data, a_type, K, M, K * F16_BYTES, BLK_K, BLK_M)
    encode(ag_out_tensor_map_1, ag_out.data, a_type, K, M, K * F16_BYTES, BLK_K, BLK_M)
    encode(B_tensor_map, B.data, b_type, K, LOCAL_N, K * F16_BYTES, BLK_K, BLK_N)
    encode(out_tensor_map, out.data, d_type, LOCAL_N, M, LOCAL_N * F16_BYTES, EPI_TILE, BLK_M)
    return (
        A_tensor_map,
        A_tensor_map_1,
        ag_out_tensor_map,
        ag_out_tensor_map_1,
        B_tensor_map,
        out_tensor_map,
    )


def _make_device_kernel():
    def test_mma_ss_tma_2sm_persistent(
        A: Kern.gptr[Kern.f16, (LOCAL_M, K)],
        B: Kern.gptr[Kern.f16, (LOCAL_N, K)],
        ag_out: Kern.gptr[Kern.f16, (M, K)],
        semaphore: Kern.gptr[Kern.u64, (WORLD_SIZE,)],
        out: Kern.gptr[Kern.f16, (M, LOCAL_N)],
        profiler_buffer: Kern.gptr[Kern.u64, (PROFILER_BUFFER_SIZE,)],
        gemm_task_types: Kern.gptr[Kern.i32, (CAPACITY,)],
        gemm_task_idxs: Kern.gptr[Kern.i32, (CAPACITY, 2)],
        gemm_head: Kern.gptr[Kern.i32, (1,)],
        gemm_tail: Kern.gptr[Kern.i32, (1,)],
        *,
        host,
    ):
        (
            A_tensor_map,
            A_tensor_map_1,
            ag_out_tensor_map,
            ag_out_tensor_map_1,
            B_tensor_map,
            out_tensor_map,
        ) = host
        semaphore = semaphore.view(WORLD_SIZE)
        gemm_task_types = gemm_task_types.view(CAPACITY)
        gemm_task_idxs = gemm_task_idxs.view(CAPACITY, 2)
        gemm_head = gemm_head.view(1)
        cbx_expr, cby_expr = Kern.cta_id_in_cluster([M_CLUSTER, N_CLUSTER])
        cbx = cbx_expr
        cby = cby_expr
        bx = Kern.cta_id()
        warp_id_in_cta = Kern.warp_id()
        wg_id = warp_id_in_cta // WARP_NUMBER
        warp_id = warp_id_in_cta % WARP_NUMBER
        tid = Kern.thread_id()
        lane_id = tid % 32
        # The original parser materialises this assignment.  A plain traced
        # Python name would re-emit nvshmem_my_pe at every use site.
        rank = Kern.alloc_local((1,), "int32")
        Kern.assign(rank[0], Kern.nvshmem.my_pe())
        # Shared-memory ownership. Padding preserves the source's externally
        # visible cluster addresses while each live region has one typed owner.
        smem = Kern.smem_pool()
        tmem_addr = smem.alloc((1,), Kern.u32, align=4)
        smem.alloc((32 - smem.bytes,), Kern.u8)
        ab_pipe = Kern.Pipeline(
            smem,
            PIPELINE_DEPTH,
            full="tma",
            empty="tcgen05",
            init_full=1,
            init_empty=NUM_CONSUMER,
            empty_phase_offset=1,
            leader=False,
        )
        out_pipe = Kern.Pipeline(
            smem,
            NUM_CONSUMER,
            full="tcgen05",
            empty="mbar",
            init_full=1,
            init_empty=128 * NUM_CONSUMER,
            empty_phase_offset=1,
            leader=False,
        )
        if smem.bytes != 128:
            raise AssertionError(f"unexpected main-pipeline header size: {smem.bytes}")
        smem.alloc((512 - smem.bytes,), Kern.u8)
        packed_buf = smem.alloc((1,), Kern.u64, align=8)
        smem.alloc((544 - smem.bytes,), Kern.u8)
        sch_pipe = Kern.Pipeline(
            smem,
            1,
            full="mbar",
            empty="mbar",
            init_full=1,
            init_empty=C2P_THREAD_COUNT,
            empty_phase_offset=1,
            leader=False,
        )
        if smem.bytes != 560:
            raise AssertionError(f"unexpected scheduler header size: {smem.bytes}")
        smem.alloc((1024 - smem.bytes,), Kern.u8)
        A_smem = smem.alloc((PIPELINE_DEPTH, NUM_CONSUMER * BLK_M, BLK_K), a_type, swizzle=SWIZZLE)
        B_smem = smem.alloc((PIPELINE_DEPTH, BLK_N, BLK_K), b_type, swizzle=SWIZZLE)
        D_smem = smem.alloc((NUM_CONSUMER * BLK_M, EPI_TILE), d_type, swizzle=SWIZZLE)
        if smem.bytes != SMEM_SIZE:
            raise AssertionError(f"unexpected shared-memory size: {smem.bytes} != {SMEM_SIZE}")
        smem.commit(SMEM_SIZE)

        # Local state.
        descA = Kern.alloc_local((1,), "uint64")
        descB = Kern.alloc_local((1,), "uint64")
        descI = Kern.alloc_local((1,), "uint32")
        tmem_addr_local = Kern.alloc_local((1,), "uint32")

        # ag + gemm
        sem = Semaphore(cnt=1, buffer=semaphore)
        gemm_queue = GEMMMPMCQueue(
            CAPACITY, gemm_task_types, gemm_task_idxs, gemm_head, GEMM_M_CLUSTERS * GEMM_N_CLUSTERS
        )
        # rank: 0 -- _mapa_u64_tx already materializes the mapa into a local
        # scalar, so the reinterpret over it is a pure type re-tag.
        packed_ptr = Kern.reinterpret(
            PointerType(PrimType("uint64")), _mapa_u64_tx(packed_buf.ptr_to([0]), 0)
        )
        packed_value = Kern.decl_buffer([1], "uint64", data=packed_ptr, scope="shared")
        # Initialize in source order after the packed-value mapa.
        ab_pipe.full.leader = tid == 0
        ab_pipe.empty.leader = tid == 0
        out_pipe.full.leader = tid == 0
        out_pipe.empty.leader = tid == 0
        ab_pipe.full.init(1)
        ab_pipe.empty.init(NUM_CONSUMER)
        out_pipe.full.init(1)
        out_pipe.empty.init(128 * NUM_CONSUMER)
        ptr = Kern.reinterpret(
            PointerType(PrimType("uint64")), _mapa_u64_tx(ab_pipe.full.ptr_to([0]), 0)
        )
        tma_finished = Kern.decl_buffer([PIPELINE_DEPTH], "uint64", data=ptr, scope="shared")
        ab_state = Kern.PipelineState(1, phase=0)
        out_state = Kern.PipelineState(1, phase=0)
        sch_producer_state = Kern.PipelineState(1, phase=0)
        sch_consumer_state = Kern.PipelineState(1, phase=0)
        sch_pipe.full.leader = tid == 0
        sch_pipe.empty.leader = (tid == 0) & (cbx == 0)
        sch_pipe.full.init(1)
        sch_pipe.empty.init(C2P_THREAD_COUNT)
        Kern.ptx.fence.proxy.async_.shared__cta()
        tile_scheduler = SingleDynamicTileScheduler(
            gemm_queue, packed_value, sch_pipe, sch_producer_state, sch_consumer_state, sem
        )
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

        # The frozen kernel dispatches independent role predicates inside every
        # scheduled task. Keep that non-chained shape and exact register-switch
        # timing while K owns both allocation metadata and functional roles.
        sp = Kern.specialize(chain_dispatch=False)
        mma_role = sp.role("mma", warps=[8, 9], regs=None, when=cbx == 0)
        sp.role("idle", warps=[10], regs=None)
        tma_scheduler_role = sp.role("tma_scheduler", warps=[11], regs=None)
        consumer = sp.role("consumer", warps=[0, 1, 2, 3, 4, 5, 6, 7], regs=None)
        producer_regs = sp.register_scope("producer_regs", warps=range(8, 12), regs=56)
        consumer_regs = sp.register_scope("consumer_regs", warps=range(8), regs=224)

        def fetch_next():
            with tma_scheduler_role:
                tile_scheduler.publish(cbx, rank[0], lane_id)
            tile_scheduler.receive(lane_id)

        # alloc TMEM
        with Kern.If((wg_id == 0) & (warp_id == 0)):
            with Kern.Then():
                Kern.ptx[f"tcgen05.alloc.cta_group::{CTA_GROUP}.sync.aligned.shared::cta.b32"](
                    tmem_addr.ptr_to([0]), Kern.uint32(N_COLS)
                )

        Kern.ptx.barrier.cluster.arrive()
        Kern.ptx.barrier.cluster.wait()
        Kern.cuda.cta_sync()
        Kern.ptx.fence.proxy.async_.shared__cta()
        Kern.ptx.fence.mbarrier_init.release.cluster()
        tile_scheduler.init()
        fetch_next()

        Kern.ptx.ld.shared.u32(tmem_addr_local[0], tmem_addr.ptr_to([0]))
        Kern.cuda.trap_when_assert_failed(tmem_addr_local[0] == 0)

        def partitioned_loop(pipe_state, main_loop, epilogue1, epilogue2):
            with Kern.serial(PIPE_CYCLE) as ko:
                with Kern.unroll(PIPELINE_DEPTH) as ks:
                    main_loop(False, ks, ko * PIPELINE_DEPTH + ks)
                pipe_state.advance()
            if PIPE_REMAIN_NUM > 0:
                # last remained loop
                with Kern.unroll(PIPE_REMAIN_NUM) as ks:
                    main_loop(True, ks, PIPE_CYCLE * PIPELINE_DEPTH + ks)
                epilogue1()
                # for unaligned cases
                with Kern.unroll(PIPE_REMAIN_NUM, PIPELINE_DEPTH) as ks:
                    epilogue2(ks)
                pipe_state.advance()
            else:
                epilogue1()

        with Kern.While(tile_scheduler.valid()):
            with Kern.If(tile_scheduler.fetched_task_type[0] == TaskType.GEMM.value):
                with Kern.Then():
                    m_idx = tile_scheduler.fetched_task_idx0[0]
                    n_idx = tile_scheduler.fetched_task_idx1[0]

                    def emit_producer_roles():
                        def tma_body():
                            # GMEM -> SMEM  (tma)
                            with Kern.If(Kern.cuda.elect_sync()):
                                with Kern.Then():
                                    # Materialized, not plain: this is bound out
                                    # here but read inside `tma_load`, which the
                                    # pipeline emits once per unrolled stage. A
                                    # plain name would re-emit the expression --
                                    # re-loading n_idx -- in every iteration.
                                    n_start = Kern.local_scalar(
                                        "int32", init=(n_idx * CTA_GROUP + cbx) * BLK_N
                                    )

                                    def tma_load(is_remain, ks, tile_idx):
                                        stage_k = tile_idx * BLK_K
                                        ab_pipe.empty.wait(ks, ab_state.phase)
                                        with Kern.If(
                                            Kern.And(
                                                rank[0] * LOCAL_GEMM_M_CLUSTERS <= m_idx,
                                                m_idx < (rank[0] + 1) * LOCAL_GEMM_M_CLUSTERS,
                                            )
                                        ):
                                            with Kern.Then():
                                                m_start0 = (
                                                    (m_idx % LOCAL_GEMM_M_CLUSTERS)
                                                    * NUM_CONSUMER
                                                    * CTA_GROUP
                                                    + cbx
                                                ) * BLK_M
                                                m_start1 = (
                                                    (m_idx % LOCAL_GEMM_M_CLUSTERS)
                                                    * NUM_CONSUMER
                                                    * CTA_GROUP
                                                    + CTA_GROUP
                                                    + cbx
                                                ) * BLK_M
                                                Kern.ptx[_TMA_G2S_CG2](
                                                    A_smem[ks].ptr_to(0, 0),
                                                    Kern.address_of(A_tensor_map),
                                                    Kern.cast(stage_k, "int32"),
                                                    Kern.cast(m_start0, "int32"),
                                                    Kern.cuda.cvta_generic_to_shared(
                                                        tma_finished.ptr_to([ks])
                                                    ),
                                                )
                                                Kern.ptx[_TMA_G2S_CG2](
                                                    A_smem[ks].ptr_to(BLK_M, 0),
                                                    Kern.address_of(A_tensor_map_1),
                                                    Kern.cast(stage_k, "int32"),
                                                    Kern.cast(m_start1, "int32"),
                                                    Kern.cuda.cvta_generic_to_shared(
                                                        tma_finished.ptr_to([ks])
                                                    ),
                                                )
                                            with Kern.Else():
                                                m_start0 = (
                                                    m_idx * NUM_CONSUMER * CTA_GROUP + cbx
                                                ) * BLK_M
                                                m_start1 = (
                                                    m_idx * NUM_CONSUMER * CTA_GROUP
                                                    + CTA_GROUP
                                                    + cbx
                                                ) * BLK_M
                                                Kern.ptx[_TMA_G2S_CG2](
                                                    A_smem[ks].ptr_to(0, 0),
                                                    Kern.address_of(ag_out_tensor_map),
                                                    Kern.cast(stage_k, "int32"),
                                                    Kern.cast(m_start0, "int32"),
                                                    Kern.cuda.cvta_generic_to_shared(
                                                        tma_finished.ptr_to([ks])
                                                    ),
                                                )
                                                Kern.ptx[_TMA_G2S_CG2](
                                                    A_smem[ks].ptr_to(BLK_M, 0),
                                                    Kern.address_of(ag_out_tensor_map_1),
                                                    Kern.cast(stage_k, "int32"),
                                                    Kern.cast(m_start1, "int32"),
                                                    Kern.cuda.cvta_generic_to_shared(
                                                        tma_finished.ptr_to([ks])
                                                    ),
                                                )
                                        Kern.ptx[_TMA_G2S_CG2](
                                            B_smem[ks].ptr_to(0, 0),
                                            Kern.address_of(B_tensor_map),
                                            Kern.cast(stage_k, "int32"),
                                            Kern.cast(n_start, "int32"),
                                            Kern.cuda.cvta_generic_to_shared(
                                                tma_finished.ptr_to([ks])
                                            ),
                                        )
                                        with Kern.If(cbx == 0):
                                            with Kern.Then():
                                                ab_pipe.full.arrive(
                                                    ks,
                                                    tx_count=NUM_CONSUMER
                                                    * BLK_K
                                                    * (BLK_M * NUM_CONSUMER + BLK_N)
                                                    * F16_BYTES,
                                                )

                                    def tma_load_epilogue(ks):
                                        ab_pipe.empty.wait(ks, ab_state.phase)
                                        with Kern.If(cbx == 0):
                                            with Kern.Then():
                                                ab_pipe.full.arrive(ks)

                                    partitioned_loop(ab_state, tma_load, skip, tma_load_epilogue)

                        def mma_body():
                            with Kern.If(Kern.cuda.elect_sync()):
                                with Kern.Then():
                                    out_pipe.empty.wait(warp_id, out_state.phase)
                                    Kern.ptx.tcgen05.fence__after_thread_sync()

                                    def mma(is_remain, ks, tile_idx):
                                        # wait tma
                                        ab_pipe.full.wait(ks, ab_state.phase)
                                        with Kern.unroll(BLK_K // MMA_K) as ki:
                                            Kern.cuda.tcgen05.encode_matrix_descriptor(
                                                Kern.address_of(descA[0]),
                                                A_smem[ks].ptr_to(warp_id * BLK_M, ki * MMA_K),
                                                ldo=1,
                                                sdo=8 * BLK_K * F16_BYTES // F128_BYTES,
                                                swizzle=SWIZZLE,
                                            )
                                            Kern.cuda.tcgen05.encode_matrix_descriptor(
                                                Kern.address_of(descB[0]),
                                                B_smem[ks].ptr_to(0, ki * MMA_K),
                                                ldo=1,
                                                sdo=8 * BLK_K * F16_BYTES // F128_BYTES,
                                                swizzle=SWIZZLE,
                                            )

                                            with Kern.If(
                                                Kern.And(
                                                    Kern.And(tile_idx == 0, ki == 0),
                                                    Kern.Or(
                                                        Kern.Not(is_remain),
                                                        Kern.And(is_remain, PIPE_CYCLE == 0),
                                                    ),
                                                )
                                            ):
                                                with Kern.Then():
                                                    Kern.ptx[_MMA_CHAIN](
                                                        Kern.cast(warp_id * MMA_N, "uint32"),
                                                        descA[0],
                                                        descB[0],
                                                        descI[0],
                                                        *_MMA_ZERO_MASKS,
                                                        False,
                                                    )
                                                with Kern.Else():
                                                    Kern.ptx[_MMA_CHAIN](
                                                        Kern.cast(warp_id * MMA_N, "uint32"),
                                                        descA[0],
                                                        descB[0],
                                                        descI[0],
                                                        *_MMA_ZERO_MASKS,
                                                        True,
                                                    )
                                        ab_pipe.empty.arrive(ks, cta_group=CTA_GROUP, cta_mask=3)

                                    def mma_epilogue1():
                                        out_pipe.full.arrive(
                                            warp_id, cta_group=CTA_GROUP, cta_mask=3
                                        )

                                    def mma_epilogue2(ks):
                                        ab_pipe.full.wait(ks, ab_state.phase)
                                        ab_pipe.empty.arrive(ks, cta_group=CTA_GROUP, cta_mask=3)

                                    partitioned_loop(ab_state, mma, mma_epilogue1, mma_epilogue2)
                                    out_state.advance()

                        with tma_scheduler_role:
                            tma_body()
                        with mma_role:
                            mma_body()

                    # setmaxnreg is warpgroup-collective, and the original keeps
                    # the functional warp dispatch inside that same reconvergence
                    # scope. K owns the transition without changing its predicate.
                    with Kern.If(
                        (NUM_CONSUMER * WARP_NUMBER <= warp_id_in_cta)
                        & (warp_id_in_cta < (NUM_CONSUMER + 1) * WARP_NUMBER)
                    ):
                        with Kern.Then():
                            producer_regs.emit()
                            emit_producer_roles()

                    with consumer:
                        consumer_regs.emit()
                        reg = Kern.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
                        reg_fp16 = Kern.alloc_buffer(
                            (TMEM_LD_SIZE // 2,), "uint32", scope="local", align=16
                        )

                        out_pipe.full.wait(wg_id, out_state.phase)
                        out_state.advance()
                        Kern.ptx.tcgen05.fence__after_thread_sync()
                        # TMEM -> RF (ld)
                        for i in range(MMA_N // TMEM_LD_SIZE):  # load (MMA_M // 2, MMA_N)
                            col_st = wg_id * MMA_N + i * TMEM_LD_SIZE
                            Kern.ptx[_TMEM_LD_64](
                                *[reg[j] for j in range(TMEM_LD_SIZE)], Kern.cast(col_st, "uint32")
                            )
                            Kern.ptx.tcgen05.wait__ld.sync.aligned()

                            # Once the final asynchronous load has completed, no
                            # later instruction consumes TMEM: conversion and
                            # output staging read only the register payload.  Let
                            # the producer reuse TMEM while that tail executes.
                            if i == MMA_N // TMEM_LD_SIZE - 1:
                                _arrive_remote_u64(out_pipe.empty.ptr_to([wg_id]), 0)

                            for j in range(TMEM_LD_SIZE // 2):
                                Kern.ptx[_CVT_F32X2](reg_fp16[j], reg[j * 2 + 1], reg[j * 2])

                            # Keep the preceding TMA store in flight while this
                            # independent TMEM chunk is loaded and converted.  The
                            # staging buffer is not overwritten until the elected
                            # thread has observed completion and the warpgroup has
                            # rendezvoused.
                            if i > 0:
                                with Kern.If((lane_id == 0) & (warp_id == 0)):
                                    with Kern.Then():
                                        Kern.ptx.cp.async_.bulk.wait_group(0)
                                Kern.cuda.warpgroup_sync(wg_id)

                            for jv in range(EPI_TILE // 8):
                                r0 = jv * 4
                                Kern.ptx.st.shared.v4.u32(
                                    D_smem.ptr_to(wg_id * BLK_M + warp_id * 32 + lane_id, jv * 8),
                                    reg_fp16[r0],
                                    reg_fp16[r0 + 1],
                                    reg_fp16[r0 + 2],
                                    reg_fp16[r0 + 3],
                                )
                            Kern.cuda.warpgroup_sync(wg_id)
                            Kern.ptx.fence.proxy.async_.shared__cta()
                            # st to gmem
                            with Kern.If((lane_id == 0) & (warp_id == 0)):
                                with Kern.Then():
                                    m_st = (
                                        m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx
                                    ) * BLK_M
                                    n_st = n_idx * BLK_N * CTA_GROUP + i * EPI_TILE
                                    Kern.ptx[_TMA_S2G](
                                        Kern.address_of(out_tensor_map),
                                        Kern.cast(n_st, "int32"),
                                        Kern.cast(m_st, "int32"),
                                        D_smem.ptr_to(wg_id * BLK_M, 0),
                                    )
                                    Kern.ptx.cp.async_.bulk.commit_group()

                        # The last store has no following TMEM load to hide its
                        # latency, but it still must complete before this staging
                        # buffer is reused by the next scheduled tile.
                        with Kern.If((lane_id == 0) & (warp_id == 0)):
                            with Kern.Then():
                                Kern.ptx.cp.async_.bulk.wait_group(0)
                        Kern.cuda.warpgroup_sync(wg_id)

            fetch_next()

        # All local and peer-CTA TMEM users must finish before collective deallocation.
        Kern.ptx.barrier.cluster.arrive()
        Kern.ptx.barrier.cluster.wait()

        # dealloc TMEM
        with Kern.If((wg_id == 0) & (warp_id == 0)):
            with Kern.Then():
                Kern.ptx[f"tcgen05.relinquish_alloc_permit.cta_group::{CTA_GROUP}.sync.aligned"]()
                Kern.ptx.ld.shared.u32(tmem_addr_local[0], tmem_addr.ptr_to([0]))
                Kern.ptx[f"tcgen05.dealloc.cta_group::{CTA_GROUP}.sync.aligned.b32"](
                    tmem_addr_local[0], Kern.uint32(N_COLS)
                )

    return Kern.kernel(
        warps=12, arch="sm_100a", min_blocks_per_sm=1, grid=SM_NUMBER, host_prelude=_host_prelude
    )(test_mma_ss_tma_2sm_persistent)


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
    return _make_device_kernel().func


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

    # cuBLAS baseline: torch.matmul dispatches to cuBLAS, so this IS the
    # library comparison.
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

    from tirx_kernels.runner import bench, external_references_enabled

    def prepare() -> None:
        case.reset()
        case.prepare()

    baselines = None
    if external_references_enabled():
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
            result["ratios"] = baseline_ratios(result)
        return {"status": "OK", **result}
    finally:
        if baselines is not None:
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
