# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

from dataclasses import dataclass

import torch

import tirx_kernels.kern as K
import tvm
from tirx_kernels.runner import bench


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
    if row_bytes % 128 == 0:
        return K.SW128B
    if row_bytes % 64 == 0:
        return K.SW64B
    return K.SW32B


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


def _cfg_for(N):
    return GEMM_CONFIGS.get(N, _DEFAULT_CONFIG)


def _make_device_kernel(dtype: str, M: int, N: int, Kdim: int):
    """Trace the kernel for one (dtype, M, N, K). Every knob is baked."""
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}")
    ab_type = _DTYPE_MAP[dtype]
    cfg = _cfg_for(N)
    MMA_N = cfg["cta_n"]
    BLK_K = cfg["cta_k"]
    PIPE_DEPTH = cfg["pipe_depth"]
    WB_PIPE_DEPTH = cfg["wb_pipe_depth"]
    L2_GROUP_SIZE = cfg["l2_group_size"]
    OVERLAP = cfg["overlap_epilogue"]

    # orig:L134-145 — everything below is derived from the knobs above.
    NUM_CONSUMER = 1 if OVERLAP else 2
    MMA_PIPE = 2 if OVERLAP else 1
    TMEM_SLOTS = MMA_PIPE if OVERLAP else NUM_CONSUMER
    TMEM_PHASE_DEPTH = MMA_PIPE if OVERLAP else 1
    NUM_D_TILES = 2 if WB_PIPE_DEPTH > 1 else 1
    BLK_M = 128  # CTA_M/2: per-CTA A rows (the 2-SM MMA combines the cluster)
    BLK_N = MMA_N // 2  # per-CTA B rows (the cluster covers MMA_N)
    EPI_N = MMA_N // WB_PIPE_DEPTH  # epilogue N tile
    AB_DTYPE = str(ab_type)
    ELEM_BYTES = ab_type.bits // 8
    D_SWIZZLE = _swizzle_for_row_bytes(EPI_N * ELEM_BYTES)
    CVT_F32X2 = "cvt.rn.f16x2.f32" if AB_DTYPE == "float16" else "cvt.rn.bf16x2.f32"
    TMEM_LD_OVERLAP = _TMEM_LD_32 if EPI_N == 32 else _TMEM_LD_64

    NUM_M_TILES = M // (256 * NUM_CONSUMER)
    NUM_N_TILES = N // MMA_N
    WARPS = (NUM_CONSUMER + 1) * 4
    CONSUMER_WARPS = list(range(NUM_CONSUMER * 4))

    def host_prelude(params):
        a = params["a"]
        b = params["b"]
        d = params["d"]
        a_map = K.stack_alloca("tensormap", 1)
        b_map = K.stack_alloca("tensormap", 1)
        d_map = K.stack_alloca("tensormap", 1)

        def encode(descriptor, rank, data, *shape):
            K.call_packed(
                "runtime.cuTensorMapEncodeTiled", descriptor, AB_DTYPE, rank, data, *shape
            )

        # A and B encode identically apart from their map, data pointer, row count
        # and block tile.
        ab_operands = ((a_map, a.data, M, BLK_M), (b_map, b.data, N, BLK_N))
        if BLK_K == 128:
            for _map, _data, _dim, _blk in ab_operands:
                encode(
                    _map, 3, _data, 64, _dim, Kdim // 64, Kdim * ELEM_BYTES,
                    64 * ELEM_BYTES, 64, _blk, BLK_K // 64, 1, 1, 1, 0, 3, 2, 0,
                )  # fmt: skip
        else:
            for _map, _data, _dim, _blk in ab_operands:
                encode(
                    _map, 2, _data, Kdim, _dim, Kdim * ELEM_BYTES, BLK_K, _blk, 1, 1, 0, 3, 2, 0,
                )  # fmt: skip
        encode(d_map, 2, d.data, N, M, N * ELEM_BYTES, EPI_N, BLK_M, 1, 1, 0, D_SWIZZLE.value, 2, 0)
        return a_map, b_map, d_map

    def gemm(
        # A/B/D are never dereferenced by the device code — every access goes
        # through a tensor map. They stay in the signature because the launch
        # is what keeps the tensors the maps point at alive.
        a,
        b,
        d,
        *,
        host,
    ):
        a_map, b_map, d_map = host
        cbx, _cby = K.cta_id_in_cluster([2, 1], preferred=[2, 1])
        bx = K.cta_id()
        warp_in_cta = K.warp_id()  # the entry's warp-uniform cta->warp scope id

        def elected():
            """The original's ``if K.cuda.elect_sync():`` — one lane per warp.

            Read straight into the ``If`` *condition*, not materialised into a
            local first. The G3 hazard (port notes §4) is a value-returning
            collective emitted at a use site *inside* a guard, where the
            excluded lanes never execute it; a condition is evaluated by every
            thread that reaches the ``If``, so this placement is convergent.
            Spelling it any other way costs a live register, and this kernel's
            consumer is allocated at exactly the 255-register ceiling
            (``fp16_bf16_gemm_kern_NOTES.md``).
            """
            return K.cuda.elect_sync() != K.uint32(0)

        # orig:L259-263 — warp 0 prefetches the three descriptors.
        with K.If(warp_in_cta == 0), K.Then():
            with K.If(elected()), K.Then():
                K.ptx.prefetch.tensormap(K.address_of(a_map))
                K.ptx.prefetch.tensormap(K.address_of(b_map))
                K.ptx.prefetch.tensormap(K.address_of(d_map))

        # ---------------- smem plan + protocol — orig:L265-295 --------------
        # Declaration order reproduces the original's byte layout. The pool's
        # commit is emitted for us at trace end (same high-water mark: nothing
        # is allocated after this point).
        smem = K.smem_pool()
        tmem_addr = smem.alloc((1,), K.u32)
        # Input smem pipeline (full=tma expect_tx, empty=tcgen05 consumed).
        smem_pipe = K.Pipeline(
            smem, PIPE_DEPTH, full="tma", empty="tcgen05", init_empty=NUM_CONSUMER
        )
        # Accumulator tmem pipeline (full=tcgen05 commit, empty=mbar consumed).
        tmem_pipe = K.Pipeline(smem, TMEM_SLOTS, full="tcgen05", empty="mbar", init_empty=2 * 128)
        # CLC tile scheduler: owns the work-stealing handshake + scheduling barriers.
        clc = K.ClusterLaunchControlScheduler(
            smem.pool,
            num_m_tiles=NUM_M_TILES,
            num_n_tiles=NUM_N_TILES,
            l2_group_size=L2_GROUP_SIZE,
            cta_group=2,
            finish_arrivals=((2 + NUM_CONSUMER) * 2 + NUM_CONSUMER),
        )
        # Teardown handshake: 1-arrival cross-CTA mbarrier (OVERLAP) vs cluster_sync.
        tmem_fin = K.Pipeline(smem, 1, full="mbar", empty="mbar", init_full=1)
        smem.pool.move_base_to(1024)
        Asmem = smem.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), ab_type, swizzle=K.SW128B)
        Bsmem = smem.alloc((PIPE_DEPTH, BLK_N, BLK_K), ab_type, swizzle=K.SW128B).buf
        Dsmem = smem.alloc((NUM_CONSUMER, NUM_D_TILES, BLK_M, EPI_N), ab_type, swizzle=D_SWIZZLE)
        smem_full_cta0 = smem_pipe.full.remote_view(0)
        with K.If(warp_in_cta == 0), K.Then():
            K.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                K.address_of(tmem_addr[0]), K.uint32(512)
            )
            K.cuda.warp_sync()
        K.ptx.fence.proxy.async_.shared__cta()
        K.ptx.fence.mbarrier_init.release.cluster()
        # OVERLAP shapes split the prologue cluster barrier: arrive (relaxed) here,
        # then each active role waits(acquire) after its own setup so the latency
        # overlaps it. No-overlap shapes keep the cheaper fused cluster_sync.
        if OVERLAP:
            K.ptx.barrier.cluster.arrive.relaxed.aligned()
        else:
            K.cuda.cluster_sync()

        def split_barrier_wait():
            if OVERLAP:
                K.ptx.barrier.cluster.wait.acquire()

        # ---------------- roles — orig:L306/L483 ----------------------------
        # K owns the functional partition, including the producer warpgroup's
        # loader/scheduler/MMA split. The register instruction remains on the
        # enclosing warpgroup scope because it is collective across all four
        # producer warps, including CTA-local idle participants.
        sp = K.specialize()
        producer_first = NUM_CONSUMER * 4
        mma_role = sp.role(
            "mma",
            warps=range(producer_first, producer_first + NUM_CONSUMER),
            regs=None,
            when=cbx == 0,
        )
        if NUM_CONSUMER == 1:
            idle_role = sp.role("idle", warps=[producer_first + 1], regs=None)
        scheduler_role = sp.role("scheduler", warps=[producer_first + 2], regs=None)
        loader_role = sp.role("loader", warps=[producer_first + 3], regs=None)
        consumer = sp.role("consumer", warps=CONSUMER_WARPS, regs=None)
        producer_regs = sp.register_scope(
            "producer_regs", warps=range(producer_first, producer_first + 4), regs=56
        )
        consumer_regs = (
            sp.register_scope("consumer_regs", warps=CONSUMER_WARPS, regs=224)
            if not OVERLAP
            else None
        )

        # Emit the functional role guards as one adjacent run. Chaining is the
        # original if/elif shape and avoids the measured N=2048 register cliff.
        def emit_roles():
            def loader_body():
                # -------- LOADER (TMA) -------- orig:L309-390
                ld = clc.worker("ld_sched")
                ld.init(bx // 2)
                tma_cur = K.PipelineState(PIPE_DEPTH, 1)
                split_barrier_wait()

                def tma_load_stage(k_tile, m_idx, n_idx):
                    smem_pipe.empty.wait(tma_cur.stage, tma_cur.phase)
                    stage = tma_cur.stage
                    k = k_tile * BLK_K
                    b_n = (n_idx * 2 + cbx) * BLK_N
                    mbar = K.cuda.cvta_generic_to_shared(smem_full_cta0.ptr_to([stage]))
                    # Each CTA loads its OWN A rows / B cols (they depend on cbx);
                    # cta_group=2 routes the completion mbarrier to the cluster,
                    # it is not a data multicast.
                    for c in range(NUM_CONSUMER):
                        a_m = ((m_idx * 2 + cbx) * NUM_CONSUMER + c) * BLK_M
                        if BLK_K == 128:
                            K.ptx[_TMA_G2S_3D_2SM](
                                Asmem.ptr_to([stage, c, 0, 0]),
                                K.address_of(a_map),
                                K.int32(0),
                                K.Cast("int32", a_m),
                                K.Cast("int32", k // 64),
                                mbar,
                            )
                        else:
                            K.ptx[_TMA_G2S_2SM](
                                Asmem.ptr_to([stage, c, 0, 0]),
                                K.address_of(a_map),
                                K.Cast("int32", k),
                                K.Cast("int32", a_m),
                                mbar,
                            )
                    if BLK_K == 128:
                        K.ptx[_TMA_G2S_3D_2SM](
                            Bsmem.ptr_to([stage, 0, 0]),
                            K.address_of(b_map),
                            K.int32(0),
                            K.Cast("int32", b_n),
                            K.Cast("int32", k // 64),
                            mbar,
                        )
                    else:
                        K.ptx[_TMA_G2S_2SM](
                            Bsmem.ptr_to([stage, 0, 0]),
                            K.address_of(b_map),
                            K.Cast("int32", k),
                            K.Cast("int32", b_n),
                            mbar,
                        )
                    # Loader-side expect_tx for the whole stage; cbx==0 owns the mbar.
                    with K.If(cbx == 0), K.Then():
                        smem_full_cta0.arrive(
                            stage, 2 * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * ELEM_BYTES
                        )

                def tma_load(m_idx, n_idx):
                    with K.serial(Kdim // BLK_K) as k_tile:
                        tma_load_stage(k_tile, m_idx, n_idx)
                        tma_cur.advance()

                # CLC loader: load the current tile, then consume the schedule
                # for the next one.
                with K.If(elected()), K.Then():
                    with K.While(ld.valid()):
                        tma_load(ld.m_idx, ld.n_idx)
                        ld.consume()
                        ld.advance_coords()
                        ld.mark_done_if_drained()

            def scheduler_body():
                # -------- CLC SCHEDULER -------- orig:L391-395
                split_barrier_wait()
                clc.run_scheduler(cbx)

            def mma_body():
                # -------- MMA (tcgen05) -------- orig:L396-482
                # Preserve the source's warpgroup-local lowering; subtracting
                # the role's global first warp expands every descriptor address.
                pw = warp_in_cta % 4
                mma_smem = K.PipelineState(PIPE_DEPTH, 0)
                # tmem wait state: double-buffered (overlap, depth=MMA_PIPE) or a
                # single slot toggled per tile (no-overlap, depth=1).
                tmem_buf = K.PipelineState(TMEM_PHASE_DEPTH, 1)
                desc_a = K.SmemDescriptor()
                desc_b = K.SmemDescriptor()
                desc_i = K.alloc_local((1,), "uint32")
                accum = K.alloc_local((1,), "int32")
                split_barrier_wait()

                def mma_stage(buf):
                    smem_pipe.full.wait(mma_smem.stage, mma_smem.phase)
                    stage = mma_smem.stage
                    tmem_n = buf * MMA_N
                    # 2-SM tcgen05 A@B^T (B is stored (N,K), so transB=False).
                    for ki in range(BLK_K // 16):
                        # Descriptor stepping is a plain 64-bit add, not
                        # `add_16B_offset`: the helper's unpack/add/pack round
                        # trip needs three extra locals per operand and, with
                        # eight of them live in this loop, ptxas spills the MMA
                        # warp (STACK 72 -> 168 B, +25% static SASS, measured).
                        # The arithmetic is identical here — the low half of a
                        # tcgen05 descriptor holds the 14-bit address and the
                        # 14-bit leading-dim offset with bits 31:30 clear, so no
                        # offset this kernel forms can carry out of it.
                        desc_a_ki = desc_a.desc + K.Cast(
                            "uint64",
                            ((stage * NUM_CONSUMER + pw) * BLK_M * BLK_K) // 8
                            + (ki // 4) * BLK_M * 8
                            + 2 * (ki % 4),
                        )
                        desc_b_ki = desc_b.desc + K.Cast(
                            "uint64",
                            (stage * BLK_N * BLK_K) // 8 + (ki // 4) * BLK_N * 8 + 2 * (ki % 4),
                        )
                        # orig: pred = any(ki != 0, accum). `ki` is a trace-time
                        # int here, so the disjunction folds: every k-phase after
                        # the first accumulates unconditionally, and only phase 0
                        # asks the runtime flag (0 on the first k-tile of a
                        # tile => overwrite, 1 after).
                        acc_pred = K.ptx.pred(1) if ki else K.ptx.pred(K.Cast("bool", accum[0]))
                        K.ptx[_MMA_F16_2SM](
                            K.Cast("uint32", tmem_n),
                            desc_a_ki,
                            desc_b_ki,
                            desc_i[0],
                            *_MMA_KEEP_ALL_LANES,
                            acc_pred,
                        )
                    K.assign(accum[0], 1)
                    smem_pipe.empty.arrive(mma_smem.stage, cta_group=2, cta_mask=3)

                def mma():
                    slot = tmem_buf.stage if OVERLAP else pw
                    tmem_pipe.empty.wait(slot, tmem_buf.phase)
                    K.assign(accum[0], 0)
                    with K.serial(Kdim // BLK_K) as _k_tile:
                        mma_stage(slot)
                        mma_smem.advance()
                    tmem_pipe.full.arrive(slot, cta_group=2, cta_mask=3)
                    tmem_buf.advance()

                # CLC MMA: consume the schedule, then accumulate. mma() ignores
                # the tile coords (it MMAs whatever the loader staged), so this
                # worker resets rather than inits.
                mm = clc.worker("mma_sched")
                mm.reset()
                with K.If(elected()), K.Then():
                    desc_a.init(
                        Asmem.ptr_to([0, 0, 0, 0]),
                        ldo=BLK_M * 8 if BLK_K == 128 else 0,
                        sdo=64,
                        swizzle=3,
                    )
                    desc_b.init(
                        Bsmem.ptr_to([0, 0, 0]),
                        ldo=BLK_N * 8 if BLK_K == 128 else 0,
                        sdo=64,
                        swizzle=3,
                    )
                    K.cuda.tcgen05.encode_instr_descriptor(
                        K.address_of(desc_i[0]),
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
                    with K.While(mm.valid()):
                        mm.consume()
                        mma()
                        mm.mark_done_if_drained()

            with loader_role:
                loader_body()
            with scheduler_role:
                scheduler_body()
            with mma_role:
                mma_body()
            if NUM_CONSUMER == 1:
                with idle_role:
                    pass

        def consumer_body():
            # ============== CONSUMER / EPILOGUE warpgroup(s) ==============
            wg = K.warp_id_in_role() >> 2  # the original's wg_id
            wid = K.warp_id_in_role() & 3  # the original's warp_id (in warpgroup)
            lane = K.lane_id()
            wb = clc.worker("wb_sched")
            wb.init(bx // 2)
            wb_buf = K.PipelineState(TMEM_PHASE_DEPTH, 0)
            split_barrier_wait()

            def store_slice(dtile, db, regs, r0):
                """One 128-bit row slice reg->smem — orig:L522-530/L576-585.

                Eight 16-bit values per store keeps each store inside a single
                swizzle chunk (st.128), avoiding the scalar 16b loop and its
                bank conflicts. Keep the slice loop in IR so layout lowering
                shares its swizzle address algebra.
                """
                with K.unroll(EPI_N // 8) as jv:
                    base = r0 + jv * 4
                    K.ptx.st.shared.v4.u32(
                        Dsmem.ptr_to([dtile, db, wid * 32 + lane, jv * 8]),
                        regs[base],
                        regs[base + 1],
                        regs[base + 2],
                        regs[base + 3],
                    )

            def tma_store(dtile, db, m_idx, n_idx, i):
                """The single-thread S2G store — orig:L532-548/L587-602."""
                with K.If((wid == 0) & (lane == 0)), K.Then():
                    # Proxy fence by the single TMA-issuing thread; the
                    # warpgroup_sync above already made the writes CTA-visible,
                    # and an all-128-thread fence was the dominant stall.
                    K.ptx.fence.proxy.async_.shared__cta()
                    d_m = ((m_idx * 2 + cbx) * NUM_CONSUMER + wg) * BLK_M
                    d_n = n_idx * MMA_N + i * EPI_N
                    K.ptx[_TMA_S2G_EVICT_FIRST](
                        K.address_of(d_map),
                        K.Cast("int32", d_n),
                        K.Cast("int32", d_m),
                        Dsmem.ptr_to([dtile, db, 0, 0]),
                        K.uint64(_EVICT_FIRST_L2_POLICY),
                    )
                # commit_group collectively reconverges the warpgroup (no post-sync).
                K.ptx.cp.async_.bulk.commit_group()

            def writeback(m_idx, n_idx):
                slot = wb_buf.stage if OVERLAP else wg
                tmem_pipe.full.wait(slot, wb_buf.phase)
                tmem_base = slot * MMA_N
                if OVERLAP:
                    # Fused per-chunk load+store, overlapping the next MMA. Keep
                    # Dreg_16b exactly EPI_N wide: a wider fragment spills
                    # registers (measured, orig:L501-503).
                    Dreg_16b = K.alloc_local((EPI_N // 2,), "uint32", align=16)
                    for i in range(WB_PIPE_DEPTH):
                        Dreg = K.alloc_local((EPI_N,), "float32")
                        tn = tmem_base + i * EPI_N
                        K.ptx[TMEM_LD_OVERLAP](
                            *[Dreg[j] for j in range(EPI_N)], K.Cast("uint32", tn)
                        )
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                        for j in range(EPI_N // 2):
                            # The packed cvt puts its second source in the low halfword.
                            K.ptx[CVT_F32X2](Dreg_16b[j], Dreg[j * 2 + 1], Dreg[j * 2])
                        if i == WB_PIPE_DEPTH - 1:
                            tmem_pipe.empty.arrive(slot, remote=0, pred=True)
                        db = i % NUM_D_TILES
                        K.ptx.cp.async_.bulk.wait_group.read(NUM_D_TILES - 1)
                        K.cuda.warpgroup_sync(wg + 10)
                        # the consumer is wg 0 here, so its D tile index is 0
                        store_slice(0, db, Dreg_16b, 0)
                        K.cuda.warpgroup_sync(wg + 10)
                        tma_store(0, db, m_idx, n_idx, i)
                else:
                    # No-overlap: load+cast every chunk, free the accumulator,
                    # then store. The tmem->reg f32 load is staged in 16-column
                    # sub-chunks so the f32 footprint stays 16 (not MMA_N) --
                    # otherwise the consumer spills (LDL/STL). orig:L549-568.
                    NOL = 16
                    Dreg_16b = K.alloc_local((MMA_N // 2,), "uint32", align=16)
                    for i in range(MMA_N // NOL):
                        Dreg = K.alloc_local((NOL,), "float32")
                        tn = tmem_base + i * NOL
                        K.ptx[_TMEM_LD_16](*[Dreg[j] for j in range(NOL)], K.Cast("uint32", tn))
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                        for j in range(NOL // 2):
                            # Keep the logical (lo, hi) pair in low/high halfword order.
                            K.ptx[CVT_F32X2](
                                Dreg_16b[i * (NOL // 2) + j], Dreg[j * 2 + 1], Dreg[j * 2]
                            )
                    tmem_pipe.empty.arrive(wg, remote=0, pred=True)
                    for i in range(WB_PIPE_DEPTH):
                        db = i % NUM_D_TILES
                        K.ptx.cp.async_.bulk.wait_group.read(NUM_D_TILES - 1)
                        K.cuda.warpgroup_sync(wg + 10)
                        store_slice(wg, db, Dreg_16b, (i * EPI_N) // 2)
                        K.cuda.warpgroup_sync(wg + 10)
                        tma_store(wg, db, m_idx, n_idx, i)

            # CLC consumer: capture the current tile, consume the schedule for
            # the next (overlapping it with the MMA-output wait), then store the
            # captured tile. The capture must be a real local: `wb.m_idx` is a
            # live slot that `advance_coords()` rewrites.
            cur_m = K.alloc_local((1,), "int32")
            cur_n = K.alloc_local((1,), "int32")
            with K.While(wb.valid()):
                K.assign(cur_m[0], wb.m_idx)
                K.assign(cur_n[0], wb.n_idx)
                wb.consume_wg(wg, wid, lane)
                wb.advance_coords()
                writeback(cur_m[0], cur_n[0])
                wb_buf.advance()
                wb.mark_done_if_drained()
            # Drain any in-flight TMA stores before the tmem teardown.
            K.ptx.cp.async_.bulk.wait_group(0)
            if OVERLAP:
                # Teardown: warpgroup_sync (all tmem reads done), then warp 0
                # does a 1-arrival cross-CTA handshake before dealloc — lighter
                # than a full cluster_sync.
                K.cuda.warpgroup_sync(wg + 10)
                with K.If((wid == 0) & (lane == 0)), K.Then():
                    tmem_fin.full.arrive(0, remote=1 - cbx, pred=True)
                with K.If(wid == 0), K.Then():
                    tmem_fin.full.wait(0, 0)

        producer_group = warp_in_cta >> 2
        with K.If(producer_group == NUM_CONSUMER):
            with K.Then():
                producer_regs.emit()
                emit_roles()
            with K.Else():
                with consumer:
                    if not OVERLAP:
                        consumer_regs.emit()
                    consumer_body()

        if not OVERLAP:
            # No-overlap keeps the full cluster_sync teardown.
            K.cuda.cluster_sync()
        with K.If(warp_in_cta == 0), K.Then():
            # tcgen05 allocation/deallocation are warp-uniform. Read the
            # allocator's shared slot explicitly so the low-level IR contains a
            # real shared load.
            K.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
            dealloc = K.alloc_local((1,), "uint32")
            K.ptx.ld.shared.u32(dealloc[0], tmem_addr.ptr_to([0]))
            K.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](dealloc[0], K.uint32(512))

    gemm.__annotations__ = {
        "a": K.gptr[AB_DTYPE, (M, Kdim)],
        "b": K.gptr[AB_DTYPE, (N, Kdim)],
        "d": K.gptr[AB_DTYPE, (M, N)],
    }
    return K.kernel(
        warps=WARPS,
        arch="sm_100a",
        min_blocks_per_sm=1,
        grid=NUM_M_TILES * NUM_N_TILES * 2,
        host_prelude=host_prelude,
    )(gemm)


def make_kernel(dtype: str, M: int, N: int, Kdim: int):
    return _make_device_kernel(dtype, M, N, Kdim).func


KERNEL_META = {
    "name": "fp16_bf16_gemm",
    "category": "basic",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a", "sm_110a"],
}
CONFIGS = [
    {"dtype": d, "M": s, "N": s, "K": s, "label": f"{d}_{s}x{s}x{s}"}
    for d in ["fp16", "bf16"]
    for s in [1024, 2048, 4096, 8192, 16384]
]


def get_kernel(dtype, M, N, K, **kwargs):
    return make_kernel(dtype, M, N, K)


def run_test(dtype, M, N, K, **kwargs):
    """Compile, run, and verify fp16/bf16 GEMM kernel."""
    from tirx_kernels.runner import compile_kernel, cuda_target

    A, B, C = prepare_data(dtype, M, N, K)
    kernel = get_kernel(dtype, M, N, K)
    C_tvm = torch.zeros_like(C)
    target = cuda_target()
    with target:
        ex = compile_kernel(kernel)
        ex(A, B, C_tvm)
    # cuBLAS baseline: torch.matmul dispatches to cuBLAS, so this IS the
    # library comparison.
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
    # DeepGEMM's BF16 entry point rejects compute capability 11, while the
    # cuBLAS path is architecture-generic and remains the exact Thor baseline.
    # Keep the additional DeepGEMM diagnostics unchanged on their supported
    # architectures instead of making an otherwise valid Thor comparison fail.
    from tirx_kernels.target import prepare_cuda_arch

    if prepared.dtype == "bf16" and prepare_cuda_arch() != "sm_110a":

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
        kernel = get_kernel(dtype, M, N, K)
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
