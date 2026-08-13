# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

from dataclasses import dataclass

import torch

import tvm
from tvm.script import tirx as T
from tvm.tirx.bench import bench
from tvm.tirx.lang.pipeline import Pipeline, PipelineState
from tvm.tirx.lang.smem_desc import SmemDescriptor
from tvm.tirx.lang.tile_scheduler import ClusterLaunchControlScheduler


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


@T.jit
def _kernel(
    A: T.Buffer((M, K), ab_type),
    B: T.Buffer((N, K), ab_type),
    D: T.Buffer((M, N), ab_type),
    *,
    M: T.constexpr,
    N: T.constexpr,
    K: T.constexpr,
    ab_type: T.constexpr,
    # Independent tuning knobs only (from GEMM_CONFIGS, selected by N). Everything
    # else is a constant or derived from these.
    MMA_N: T.constexpr,
    BLK_K: T.constexpr,
    PIPE_DEPTH: T.constexpr,
    WB_PIPE_DEPTH: T.constexpr,
    L2_GROUP_SIZE: T.constexpr,
    OVERLAP_EPILOGUE: T.constexpr,
):
    # Named locals: knob-branching or heavily-reused geometry; single-use values and
    # constants (CTA_GROUP=2, the cluster grid, ...) are inlined at their use-sites.
    NUM_CONSUMER = T.meta_var(1 if OVERLAP_EPILOGUE else 2)
    MMA_PIPE = T.meta_var(2 if OVERLAP_EPILOGUE else 1)
    TMEM_SLOTS = T.meta_var(MMA_PIPE if OVERLAP_EPILOGUE else NUM_CONSUMER)
    TMEM_PHASE_DEPTH = T.meta_var(MMA_PIPE if OVERLAP_EPILOGUE else 1)
    NUM_D_TILES = T.meta_var(2 if WB_PIPE_DEPTH > 1 else 1)
    BLK_M = T.meta_var(128)  # CTA_M/2: per-CTA A rows (2-SM MMA combines the cluster)
    BLK_N = T.meta_var(MMA_N // 2)  # per-CTA B rows (cluster covers MMA_N)
    EPI_N = T.meta_var(MMA_N // WB_PIPE_DEPTH)  # epilogue N tile
    AB_DTYPE = T.meta_var(str(ab_type))
    D_SWIZZLE = T.meta_var(_swizzle_for_row_bytes(EPI_N * (ab_type.bits // 8)).value)
    CVT_F32X2 = T.meta_var("cvt.rn.f16x2.f32" if AB_DTYPE == "float16" else "cvt.rn.bf16x2.f32")
    TMEM_LD_OVERLAP = T.meta_var(_TMEM_LD_32 if EPI_N == 32 else _TMEM_LD_64)

    A_tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    B_tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    D_tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    if BLK_K == 128:
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
    else:
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

    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    cbx, cby = T.cta_id_in_cluster([2, 1], preferred=[2, 1])
    bx = T.cta_id([(M // (256 * NUM_CONSUMER) * (N // MMA_N) * 2)])
    wg_id = T.warpgroup_id([(NUM_CONSUMER + 1)])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])

    if (wg_id == 0) & (warp_id == 0):
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(A_tensor_map)))
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(B_tensor_map)))
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(D_tensor_map)))

    pool = T.SMEMPool()
    tmem_addr = pool.alloc((1,), "uint32")
    tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=2, tmem_addr=tmem_addr)
    # Input smem pipeline (full=tma expect_tx, empty=tcgen05 consumed).
    smem_pipe = Pipeline(pool, PIPE_DEPTH, full="tma", empty="tcgen05", init_empty=NUM_CONSUMER)
    # Accumulator tmem pipeline (full=tcgen05 commit, empty=mbar consumed).
    tmem_pipe = Pipeline(pool, TMEM_SLOTS, full="tcgen05", empty="mbar", init_empty=2 * 128)
    # CLC tile scheduler: owns the work-stealing handshake + scheduling barriers.
    clc_sched = ClusterLaunchControlScheduler(
        pool,
        num_m_tiles=(M // (256 * NUM_CONSUMER)),
        num_n_tiles=(N // MMA_N),
        l2_group_size=L2_GROUP_SIZE,
        cta_group=2,
        finish_arrivals=((2 + NUM_CONSUMER) * 2 + NUM_CONSUMER),
    )
    # Teardown handshake: 1-arrival cross-CTA mbarrier (OVERLAP) vs the full cluster_sync.
    tmem_fin = Pipeline(pool, 1, full="mbar", empty="mbar", init_full=1)
    pool.move_base_to(1024)
    Asmem = pool.alloc_tcgen05_mma_AB((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), ab_type)
    Bsmem = pool.alloc_tcgen05_mma_AB((PIPE_DEPTH, BLK_N, BLK_K), ab_type)
    Dsmem = pool.alloc_tcgen05_mma_AB(
        (NUM_CONSUMER, NUM_D_TILES, BLK_M, EPI_N),
        ab_type,
        swizzle_mode=_swizzle_for_row_bytes(EPI_N * (ab_type.bits // 8)),
    )
    pool.commit()
    smem_full_cta0 = smem_pipe.full.remote_view(0)
    tmem = tmem_pool.alloc((128, 512), "float32")
    # Keep pool allocation/layout bookkeeping; teardown is emitted explicitly below.
    tmem_pool.commit()
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    # OVERLAP shapes split the prologue cluster barrier: arrive (relaxed) here, then each
    # active role waits(acquire) after its own setup so the latency overlaps it (idle warps
    # skip the wait). No-overlap shapes keep the cheaper fused cluster_sync.
    if OVERLAP_EPILOGUE:
        T.ptx.barrier.cluster.arrive.relaxed.aligned()
    else:
        T.cuda.cluster_sync()

    if wg_id == NUM_CONSUMER:
        # ==================== PRODUCER warpgroup ====================
        T.ptx.setmaxnreg.dec.sync.aligned.u32(56)  # reduce the producer's per-warp register budget
        if warp_id == 3:
            # -------- LOADER (TMA) --------
            ld = clc_sched.worker("ld_sched")
            ld.init(bx // 2)
            tma_cur = PipelineState(PIPE_DEPTH, 1)
            if OVERLAP_EPILOGUE:
                T.ptx.barrier.cluster.wait.acquire()  # split cluster barrier (loader)

            @T.inline
            def tma_load_stage(k_tile, m_idx, n_idx):
                smem_pipe.empty.wait(tma_cur.stage, tma_cur.phase)
                stage = tma_cur.stage
                k = T.meta_var(k_tile * BLK_K)
                b_n = T.meta_var((n_idx * 2 + cbx) * BLK_N)
                # Each CTA loads its OWN A rows / B cols (depend on cbx); cta_group=2
                # routes the completion mbarrier to the cluster, not a data multicast.
                for c in T.unroll(NUM_CONSUMER):
                    a_m = T.meta_var(((m_idx * 2 + cbx) * NUM_CONSUMER + c) * BLK_M)
                    if BLK_K == 128:
                        T.evaluate(
                            T.ptx[_TMA_G2S_3D_2SM](
                                Asmem.ptr_to([stage, c, 0, 0]),
                                T.address_of(A_tensor_map),
                                T.int32(0),
                                T.cast(a_m, "int32"),
                                T.cast(k // 64, "int32"),
                                T.cuda.cvta_generic_to_shared(smem_full_cta0.ptr_to([stage])),
                            )
                        )
                    else:
                        T.evaluate(
                            T.ptx[_TMA_G2S_2SM](
                                Asmem.ptr_to([stage, c, 0, 0]),
                                T.address_of(A_tensor_map),
                                T.cast(k, "int32"),
                                T.cast(a_m, "int32"),
                                T.cuda.cvta_generic_to_shared(smem_full_cta0.ptr_to([stage])),
                            )
                        )
                if BLK_K == 128:
                    T.evaluate(
                        T.ptx[_TMA_G2S_3D_2SM](
                            Bsmem.ptr_to([stage, 0, 0]),
                            T.address_of(B_tensor_map),
                            T.int32(0),
                            T.cast(b_n, "int32"),
                            T.cast(k // 64, "int32"),
                            T.cuda.cvta_generic_to_shared(smem_full_cta0.ptr_to([stage])),
                        )
                    )
                else:
                    T.evaluate(
                        T.ptx[_TMA_G2S_2SM](
                            Bsmem.ptr_to([stage, 0, 0]),
                            T.address_of(B_tensor_map),
                            T.cast(k, "int32"),
                            T.cast(b_n, "int32"),
                            T.cuda.cvta_generic_to_shared(smem_full_cta0.ptr_to([stage])),
                        )
                    )
                # Loader-side expect_tx for the whole stage's bytes; cbx==0 owns the mbar.
                if cbx == 0:
                    smem_full_cta0.arrive(
                        stage,
                        2 * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * (ab_type.bits // 8),
                    )

            @T.inline
            def tma_load(m_idx, n_idx):
                for k_tile in T.serial(K // BLK_K):
                    tma_load_stage(k_tile, m_idx, n_idx)
                    tma_cur.advance()

            # CLC loader: load the current tile, then consume the schedule for the next.
            if T.cuda.elect_sync():
                while ld.valid():
                    m_idx = T.meta_var(ld.m_idx)
                    n_idx = T.meta_var(ld.n_idx)
                    tma_load(m_idx, n_idx)
                    ld.consume()
                    ld.advance_coords()
                    ld.mark_done_if_drained()
        elif warp_id == 2:
            # -------- CLC SCHEDULER --------
            if OVERLAP_EPILOGUE:
                T.ptx.barrier.cluster.wait.acquire()  # split barrier (scheduler)
            clc_sched.run_scheduler(cbx)
        elif (warp_id < NUM_CONSUMER) & (cbx == 0):
            # -------- MMA (tcgen05) --------
            mma_smem = PipelineState(PIPE_DEPTH, 0)
            # tmem wait state: double-buffer (overlap, depth=MMA_PIPE) or a single
            # slot toggled per tile (no-overlap, depth=1).
            tmem_buf = PipelineState(TMEM_PHASE_DEPTH, 1)
            desc_a = T.meta_var(SmemDescriptor())
            desc_b = T.meta_var(SmemDescriptor())
            desc_i = T.alloc_local((1,), "uint32")
            accum: T.int32
            if OVERLAP_EPILOGUE:
                T.ptx.barrier.cluster.wait.acquire()  # split cluster barrier (MMA)

            @T.inline
            def mma_stage(buf):
                smem_pipe.full.wait(mma_smem.stage, mma_smem.phase)
                stage = mma_smem.stage
                tmem_n = T.meta_var(buf * MMA_N)
                # 2-SM tcgen05 A@B^T (B stored (N,K), transB=False); accum=0 on the
                # first k-tile (overwrite), 1 thereafter.
                for ki in T.unroll(BLK_K // 16):
                    desc_a_ki = T.meta_var(
                        desc_a.add_16B_offset(
                            ((stage * NUM_CONSUMER + warp_id) * BLK_M * BLK_K) // 8
                            + (ki // 4) * BLK_M * 8
                            + 2 * (ki % 4)
                        )
                    )
                    desc_b_ki = T.meta_var(
                        desc_b.add_16B_offset(
                            (stage * BLK_N * BLK_K) // 8 + (ki // 4) * BLK_N * 8 + 2 * (ki % 4)
                        )
                    )
                    T.evaluate(
                        T.ptx[_MMA_F16_2SM](
                            T.cast(tmem_n, "uint32"),
                            desc_a_ki,
                            desc_b_ki,
                            desc_i[0],
                            *_MMA_KEEP_ALL_LANES,
                            T.ptx.pred(tvm.tirx.any(ki != 0, T.cast(accum, "bool"))),
                        )
                    )
                accum = 1
                smem_pipe.empty.arrive(mma_smem.stage, cta_group=2, cta_mask=3)

            @T.inline
            def mma():
                slot = T.meta_var(tmem_buf.stage if OVERLAP_EPILOGUE else warp_id)
                tmem_pipe.empty.wait(slot, tmem_buf.phase)
                accum = 0
                for k_tile in T.serial(K // BLK_K):
                    mma_stage(slot)
                    mma_smem.advance()
                tmem_pipe.full.arrive(slot, cta_group=2, cta_mask=3)
                tmem_buf.advance()

            # CLC MMA: consume the schedule, then accumulate. mma() ignores the tile
            # coords (it MMAs whatever the loader staged), so reset() not init().
            mm = clc_sched.worker("mma_sched")
            mm.reset()
            if T.cuda.elect_sync():
                desc_a.init(
                    Asmem.ptr_to([0, 0, 0, 0]),
                    ldo=BLK_M * 8 if BLK_K == 128 else 0,
                    sdo=64,
                    swizzle=3,
                )
                desc_b.init(
                    Bsmem.ptr_to([0, 0, 0]), ldo=BLK_N * 8 if BLK_K == 128 else 0, sdo=64, swizzle=3
                )
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
                while mm.valid():
                    mm.consume()
                    mma()
                    mm.mark_done_if_drained()
    elif wg_id < NUM_CONSUMER:
        # ==================== CONSUMER / EPILOGUE warpgroup(s) ====================
        if not OVERLAP_EPILOGUE:
            T.ptx.setmaxnreg.inc.sync.aligned.u32(
                224
            )  # raise the consumer's per-warp register budget
        wb = clc_sched.worker("wb_sched")
        wb.init(bx // 2)
        wb_buf = PipelineState(TMEM_PHASE_DEPTH, 0)
        if OVERLAP_EPILOGUE:
            T.ptx.barrier.cluster.wait.acquire()  # split cluster barrier (consumer)

        @T.inline
        def writeback(m_idx, n_idx):
            slot = T.meta_var(wb_buf.stage if OVERLAP_EPILOGUE else wg_id)
            tmem_pipe.full.wait(slot, wb_buf.phase)
            tmem_base = T.meta_var(slot * MMA_N)
            if OVERLAP_EPILOGUE:
                # Fused per-chunk load+store, overlapping the next MMA. Keep Dreg_16b
                # exactly EPI_N wide: a wider fragment spills registers (measured).
                Dreg_16b = T.alloc_local((EPI_N // 2,), "uint32", align=16)
                for i in T.unroll(WB_PIPE_DEPTH):
                    Dreg = T.alloc_local((EPI_N,), "float32")
                    tn = T.meta_var(tmem_base + i * EPI_N)
                    T.evaluate(
                        T.ptx[TMEM_LD_OVERLAP](
                            *[Dreg[j] for j in range(EPI_N)], T.cast(tn, "uint32")
                        )
                    )
                    T.ptx.tcgen05.wait__ld.sync.aligned()
                    for j in T.unroll(EPI_N // 2):
                        # Packed cvt puts its second source in the low halfword.
                        T.evaluate(T.ptx[CVT_F32X2](Dreg_16b[j], Dreg[j * 2 + 1], Dreg[j * 2]))
                    if i == WB_PIPE_DEPTH - 1:
                        tmem_pipe.empty.arrive(slot, remote=0, pred=True)
                    db = T.meta_var(i % NUM_D_TILES)
                    T.ptx.cp.async_.bulk.wait_group.read(NUM_D_TILES - 1)
                    T.cuda.warpgroup_sync(wg_id + 10)
                    # consumer is wg_id==0 here; each thread writes one 128-bit row slice.
                    for jv in T.unroll(EPI_N // 8):
                        r0 = T.meta_var(jv * 4)
                        T.ptx.st.shared.v4.u32(
                            Dsmem.ptr_to([0, db, warp_id * 32 + lane_id, jv * 8]),
                            Dreg_16b[r0],
                            Dreg_16b[r0 + 1],
                            Dreg_16b[r0 + 2],
                            Dreg_16b[r0 + 3],
                        )
                    T.cuda.warpgroup_sync(wg_id + 10)
                    if (warp_id == 0) & (lane_id == 0):
                        # Proxy fence by the single TMA-issuing thread (warpgroup_sync above
                        # made writes CTA-visible; an all-128-thread fence was the dominant stall).
                        T.ptx.fence.proxy.async_.shared__cta()
                        d_m = T.meta_var(((m_idx * 2 + cbx) * NUM_CONSUMER + wg_id) * BLK_M)
                        d_n = T.meta_var(n_idx * MMA_N + i * EPI_N)
                        T.evaluate(
                            T.ptx[_TMA_S2G_EVICT_FIRST](
                                T.address_of(D_tensor_map),
                                T.cast(d_n, "int32"),
                                T.cast(d_m, "int32"),
                                Dsmem.ptr_to([0, db, 0, 0]),
                                T.uint64(_EVICT_FIRST_L2_POLICY),
                            )
                        )
                    # commit_group collectively reconverges the warpgroup (no post-sync).
                    T.ptx.cp.async_.bulk.commit_group()
            else:
                # No-overlap: load+cast all chunks, free the accumulator, then store. Stage
                # the tmem->reg f32 load in 16-col sub-chunks so the f32 footprint stays 16
                # (not EPI_N), else the consumer spills (LDL/STL).
                NOL = T.meta_var(16)
                Dreg_16b = T.alloc_local((MMA_N // 2,), "uint32", align=16)
                for i in T.unroll(MMA_N // NOL):
                    Dreg = T.alloc_local((NOL,), "float32")
                    tn = T.meta_var(tmem_base + i * NOL)
                    T.evaluate(
                        T.ptx[_TMEM_LD_16](*[Dreg[j] for j in range(NOL)], T.cast(tn, "uint32"))
                    )
                    T.ptx.tcgen05.wait__ld.sync.aligned()
                    for j in T.unroll(NOL // 2):
                        # Keep the logical (lo, hi) pair in low/high halfword order.
                        T.evaluate(
                            T.ptx[CVT_F32X2](
                                Dreg_16b[i * (NOL // 2) + j], Dreg[j * 2 + 1], Dreg[j * 2]
                            )
                        )
                tmem_pipe.empty.arrive(wg_id, remote=0, pred=True)
                for i in T.unroll(WB_PIPE_DEPTH):
                    db = T.meta_var(i % NUM_D_TILES)
                    T.ptx.cp.async_.bulk.wait_group.read(NUM_D_TILES - 1)
                    T.cuda.warpgroup_sync(wg_id + 10)
                    # Store reg->smem in 8x16-bit (128b) sub-slices -> st.128 (one swizzle
                    # chunk each), avoiding the scalar 16b loop / bank conflicts.
                    for jv in T.unroll(EPI_N // 8):
                        c0 = T.meta_var(i * EPI_N + jv * 8)
                        r0 = T.meta_var(c0 // 2)
                        T.ptx.st.shared.v4.u32(
                            Dsmem.ptr_to([wg_id, db, warp_id * 32 + lane_id, jv * 8]),
                            Dreg_16b[r0],
                            Dreg_16b[r0 + 1],
                            Dreg_16b[r0 + 2],
                            Dreg_16b[r0 + 3],
                        )
                    T.cuda.warpgroup_sync(wg_id + 10)
                    if (warp_id == 0) & (lane_id == 0):
                        # Single-thread proxy fence after the CTA sync (see overlap path).
                        T.ptx.fence.proxy.async_.shared__cta()
                        d_m = T.meta_var(((m_idx * 2 + cbx) * NUM_CONSUMER + wg_id) * BLK_M)
                        d_n = T.meta_var(n_idx * MMA_N + i * EPI_N)
                        T.evaluate(
                            T.ptx[_TMA_S2G_EVICT_FIRST](
                                T.address_of(D_tensor_map),
                                T.cast(d_n, "int32"),
                                T.cast(d_m, "int32"),
                                Dsmem.ptr_to([wg_id, db, 0, 0]),
                                T.uint64(_EVICT_FIRST_L2_POLICY),
                            )
                        )
                    # commit_group collectively reconverges the warpgroup (no post-sync).
                    T.ptx.cp.async_.bulk.commit_group()

        # CLC consumer: capture the current tile, consume the schedule for the next
        # (overlapping it with the MMA-output wait), then store the captured tile.
        cur_m: T.int32
        cur_n: T.int32
        while wb.valid():
            cur_m = wb.m_idx
            cur_n = wb.n_idx
            wb.consume_wg(wg_id, warp_id, lane_id)
            wb.advance_coords()
            cm = T.meta_var(cur_m)
            cn = T.meta_var(cur_n)
            writeback(cm, cn)
            wb_buf.advance()
            wb.mark_done_if_drained()
        # Drain any in-flight TMA stores before tmem teardown.
        T.ptx.cp.async_.bulk.wait_group(0)
        if OVERLAP_EPILOGUE:
            # Teardown: warpgroup_sync (all tmem reads done), then warp0 does a 1-arrival
            # cross-CTA mbarrier handshake before dealloc -- lighter than a full cluster_sync.
            T.cuda.warpgroup_sync(wg_id + 10)
            if (warp_id == 0) & (lane_id == 0):
                tmem_fin.full.arrive(0, remote=1 - cbx, pred=True)
            if warp_id == 0:
                tmem_fin.full.wait(0, 0)
    if not OVERLAP_EPILOGUE:
        # No-overlap keeps the full cluster_sync teardown.
        T.cuda.cluster_sync()
    if (wg_id == 0) & (warp_id == 0):
        # tcgen05 allocation/deallocation are warp-uniform. Read the allocator's
        # shared slot explicitly so the low-level IR contains a real shared load.
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
        tmem_dealloc_addr: T.uint32
        T.ptx.ld.shared.u32(tmem_dealloc_addr, tmem_addr.ptr_to([0]))
        T.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](tmem_dealloc_addr, T.uint32(512))


def tir_kernel(dtype: str, M: int, N: int, K: int):
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}")
    ab_type = _DTYPE_MAP[dtype]
    cfg = GEMM_CONFIGS.get(N, _DEFAULT_CONFIG)
    # Bind only the independent knobs; _kernel derives all geometry from these.
    return _kernel.specialize(
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
    from tirx_kernels.runner import compile_kernel

    A, B, C = prepare_data(dtype, M, N, K)
    kernel = tir_kernel(dtype, M, N, K)
    C_tvm = torch.zeros_like(C)
    target = tvm.target.Target("cuda")
    with target:
        ex = compile_kernel(kernel)
        ex(A, B, C_tvm)
    C_ref = torch.matmul(A, B.T)
    torch.testing.assert_close(C_tvm.cpu(), C_ref.cpu(), rtol=0.001, atol=0.01)


@dataclass(frozen=True)
class PreparedBench:
    """CPU-prepared GEMM benchmark whose remaining work requires a GPU."""

    dtype: str
    M: int
    N: int
    K: int
    executable: object

    def run_gpu(self, warmup=None, repeat=None, timer=None, **kwargs):
        """Allocate inputs/references and run the unchanged GPU timing protocol."""
        A, B, C = prepare_data(self.dtype, self.M, self.N, self.K)
        C_tir = torch.zeros_like(C, device="cuda")

        funcs = {"tir": lambda: self.executable(A, B, C_tir)}

        def _torch_cublas():
            C_out = torch.zeros_like(C, device="cuda")
            return lambda: torch.matmul(A, B.T, out=C_out)

        references = {"torch-cublas": _torch_cublas}
        if self.dtype == "bf16":

            def _deepgemm_cublaslt():
                import deep_gemm

                C_out = torch.zeros(self.M, self.N, dtype=torch.bfloat16, device="cuda")
                return lambda: deep_gemm.cublaslt_gemm_nt(A, B, C_out, None)

            def _deepgemm_bf16():
                import deep_gemm

                C_out = torch.zeros(self.M, self.N, dtype=torch.bfloat16, device="cuda")
                return lambda: deep_gemm.bf16_gemm_nt(A, B, C_out)

            references.update(
                {"deepgemm-cublaslt": _deepgemm_cublaslt, "deepgemm-bf16": _deepgemm_bf16}
            )

        return bench(
            funcs,
            warmup=warmup,
            repeat=repeat,
            timer=timer,
            references=references,
            **kwargs,
        )


def prepare_bench(dtype, M, N, K, **kwargs):
    """Specialize and compile the GEMM without initializing CUDA."""
    from tirx_kernels.runner import cuda_initialization_guard

    with cuda_initialization_guard():
        kernel = tir_kernel(dtype, M, N, K)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule({"main": kernel})
            ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
    return PreparedBench(dtype=dtype, M=M, N=N, K=K, executable=ex)


def run_bench(dtype, M, N, K, warmup=None, repeat=None, timer=None, **kwargs):
    """Benchmark fp16/bf16 GEMM."""
    return prepare_bench(dtype, M, N, K).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, **kwargs
    )
