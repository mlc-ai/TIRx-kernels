# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved Alpha-MoE FP8 block-scale megakernel for B200 (Qwen3-Next TP4).

This module covers the SIX official Alpha-MoE rows -- ``M`` in
{1, 8, 16, 32, 64, 128} tokens -- at hidden size 2048, intermediate size 128,
512 experts, top-10 routing, no shared experts, block-128 FP32 scales,
``float8_e4m3fn`` weights and BF16 in/out.  Routes are SUPPLIED rather than
computed: the kernel consumes the given ``topk_ids`` and their paired FP32
``topk_weights`` in the given order and never recomputes or renormalizes them.
Activation quantization, expert alignment, both projections and the BF16
accumulation all happen inside the timed call.

**This replaces the previous single-shape (M=128 only) kernel, and M=128
regresses from 59.984 us to 60.960 us -- about 1.6% -- in exchange for covering
all six rows.**  That trade was taken deliberately: the task scores the
geometric mean over the six rows, where this kernel reaches 2.3437x, and it is
3.53x at M=1 where the replaced kernel did not run at all.  A reviewer who
cares only about M=128 should know that the superseded kernel was faster on
that one row.

The selected kernel is ``cluster-split`` from evolution run
``alphamoe-20260916-193749``.  Two tirx-lite kernels are dispatched by shape
inside ``setup``; both are persistent and both are reached only through
``@txl.kernel``:

``M >= 8`` -- one persistent 2-CTA-cluster launch, 12 warps per CTA.  A work
item is one (expert, chunk of <= 16 tokens); both CTAs of a cluster process the
same item and every cluster walks a static round-robin item sequence, so there
is no work counter and no cross-cluster data handoff.  Gate/up is N-split (64
intermediate channels per CTA), the down projection is H-split (8 output tiles
per CTA), and the next item's gate/up stream precedes the current item's down
stream so the last item's exchange does not sit in the tail.  M=8/16/64
exchange rank-major FP8 activation halves through DSMEM; M=32/128 retain the
FP32-slice exchange.  An idle warp owns cluster-slot release, and M <= 32
publishes token lists early.  Shape-only grids are M8=80, M32=128, M64=124 and
M128=134 CTAs, with M=16 using the full SM count.  At M=64 each adjacent pair
of down tiles is fetched by one rank-3 32 KiB transaction over an eight-stage
ring; at M=16 and M=128 gate/up promotion packs adjacent independent FP32 lanes
into ``fma.rn.f32x2`` while preserving each lane's accumulation order.

``M == 1`` -- one 8-CTA cluster per supplied route (K-split gate/up, DSMEM
all-reduce of the partial sums, replicated SwiGLU and group-128 FP8
quantization, H-split down).  Each CTA fetches gate/up with two rank-3 32 KiB
transactions and its two down tiles with one more, instead of six rank-2 16 KiB
transactions.  The ten preselected routes are the ten cluster work items, which
preserves route order and duplicate expert IDs without the generic 512-expert
planner.

Routed BF16 contributions are accumulated with vector ``red.global.add.bf16x2``
so that each update rounds to BF16, as the task contract requires.

Measured on GB200 against the packaged FlashInfer
``trtllm_fp8_block_scale_routed_moe`` baseline.  The locked evaluation harness,
which times with CUPTI and flushes L2 between iterations, scores (ours / sota /
speedup, microseconds): M=1 6.232/22.016/3.5326x, M=8 16.088/44.552/2.7693x,
M=16 23.488/54.960/2.3399x, M=32 35.576/74.360/2.0902x, M=64
49.665/93.416/1.8809x, M=128 60.960/112.256/1.8415x -- geometric mean 2.3437x.
``README.md`` records the pairs a reviewer can reproduce from this module with
``run_bench(timer="event")``, which measures a 2.2725x geometric mean (30.907
us versus 70.237 us) across the same six rows; see those rows for the per-shape
reproducible numbers.

As with the superseded kernel, ``run_bench`` defaults to the Proton timer,
which reports a flatteringly higher speedup because its per-kernel
instrumentation costs the many-kernel baseline far more than these
single-kernel candidates.  Prefer the event timer when re-measuring.

Two real data races were found by Racecheck during the evolution run and fixed
here: missing free-arrives that left no synchronization edge ordering a buffer
read before a peer's later write, and a scheduler token-list re-store that
raced the quantizer's early read.  The selected kernel is Synccheck- and
Racecheck-clean.
"""

import ctypes
from dataclasses import dataclass, fields
from typing import Any
from unittest import SkipTest

import numpy as np
import torch

import tirx_kernels.tirx_lite as txl
import tvm
from tvm.backend.cuda.cpp.descriptors import encode_instr_descriptor_dense_uint32

BK = 128
BM = 128
NT = 16
NWARPS = 12
MATH_WARP0 = 4
QUANT_WARP0 = 8
NMATH = 4 * 32
NQUANT = 4 * 32
NTB = 32
TASK_RING = 4
TASK_W = 4 + 2 * NT + 3 * 16
TOK_OFF = 4
RW_OFF = 4 + NT
W1_OFF = 4 + 2 * NT
W2_OFF = 4 + 2 * NT + 32
TMEM_COLS = NTB * NT
FP8_MAX = 448.0
INV_FP8_MAX = float(np.float32(1.0) / np.float32(FP8_MAX))
LOG2E = 1.4426950408889634
EVICT_FIRST = 0x12F0000000000000
EVICT_LAST = 0x14F0000000000000
TMA_G2S_2D = (
    "cp.async.bulk.tensor.2d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
)
TMA_G2S_3D = (
    "cp.async.bulk.tensor.3d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
)
MMA = "tcgen05.mma.cta_group::1.kind::f8f6f4"
TMEM_LD64 = "tcgen05.ld.sync.aligned.32x32b.x64.b32"
FMA_F32 = "fma.rn.f32"
BULK_S2C = "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
BAR_MATH = 1
BAR_QUANT = 2
BAR_ROWS = 3
DONE = 32
ZERO_DONE = 64
WORK_SLOTS = 64
N_TASK_CONSUMERS = 1 + 1 + 4 + 4
REGS_WG0 = 112
REGS_MATH = 224
REGS_QUANT = 152
CS = 2
W1_TMA3D = True
W2_TMA3D_MS = (16, 64, 128)
CH = 128 // CS
HBC = 16 // CS


def _f32(v):
    return txl.local_scalar(txl.f32, init=v)


def _i32(v):
    return txl.local_scalar(txl.i32, init=v)


def _u32(v):
    return txl.local_scalar(txl.u32, init=v)


def _bf16_lo(word):
    return txl.reinterpret("float32", txl.shift_left(word, txl.uint32(16)))


def _bf16_hi(word):
    return txl.reinterpret("float32", txl.bitwise_and(word, txl.uint32(0xFFFF0000)))


def _rng(name):
    token = txl.alloc_local([1], "uint32")
    txl.assign(token[0], txl.cuda.iket.range_start(name))
    return token


def _rng_end(token):
    txl.cuda.iket.range_end(token[0])


def build_kernel(G, M, TOPK, E, HID, INTER, cs=CS):
    assert cs == CS
    NGU = 2 * INTER
    KB = HID // BK
    assert INTER == BK and HID % BK == 0 and KB == 16
    assert G % CS == 0
    NCL = G // CS
    SB_MAX = 8
    SHARED_B = M <= SB_MAX
    GLOBAL_Q = M > SB_MAX
    PAIR_D = M in (16, 64, 128)
    RANK_ACT = M in (8, 16, 64)
    ROWS_PER_CTA = (M + G - 1) // G
    STAGES = 8 if M == 64 else (6 if PAIR_D else (8 if SHARED_B else 7))
    NB = 1 if SHARED_B else 2
    P = M * TOPK
    NTHREADS = NWARPS * 32
    MW = (M + 31) // 32
    IDS_PER_T = (P + NTHREADS - 1) // NTHREADS
    EPT = (E + NTHREADS - 1) // NTHREADS
    ROUTE_IDS = 0 if M == 1 else IDS_PER_T
    ROUTE_EPT = 0 if M == 1 else EPT
    MASK_COUNT = 0 if M == 1 else E * MW
    TILE_A = BM * BK
    HALF_A = CH * BK
    B_BYTES = KB * NT * BK
    AMX_BYTES = NT * 4
    ACT_BYTES = CH * NT
    HV_BYTES = CH * NT * 4
    OUT_WORDS = M * HID // 2
    ZERO_PER_T = (OUT_WORDS // 4 + G * NTHREADS - 1) // (G * NTHREADS)
    RQ_T = 2
    assert RQ_T * KB == 32
    # Early token-list / idle-warp row publication only where it measured faster (M <= 32); the
    # M >= 64 shapes keep the quantizer-warp row publication and task_full-driven quantizers.
    FIRST_TILE = GLOBAL_Q and M <= 32
    assert TOPK <= 32 and NT <= 32 and HBC % 4 == 0 and KB % 4 == 0
    if SHARED_B:
        assert MW == 1
    IDESC = encode_instr_descriptor_dense_uint32(
        M=BM, N=NT, K=32, d_dtype="float32", a_dtype="float8_e4m3fn",
        b_dtype="float8_e4m3fn", trans_a=False, trans_b=False, cta_group=1,
    )

    @txl.kernel(warps=NWARPS, arch="sm_100a", min_blocks_per_sm=1, grid=G)
    def alphamoe_cluster(
        topk_ids: txl.gptr[txl.i32],
        topk_w: txl.gptr[txl.f32],
        hidden: txl.gptr[txl.i32],
        w1s: txl.gptr[txl.f32],
        w2s: txl.gptr[txl.f32],
        out: txl.gptr[txl.i32],
        sync_ctr: txl.gptr[txl.u32],
        xq_g: txl.gptr[txl.u32],
        xs_g: txl.gptr[txl.f32],
        xflag_g: txl.gptr[txl.u32],
        tm_w1h: txl.TensorMap,
        tm_w2: txl.TensorMap,
        rsf: txl.f32,
        epoch: txl.u32,
    ):
        cta = txl.cta_id()
        rank = txl.cta_id_in_cluster([CS])
        peer = txl.int32(CS - 1) - rank
        warp = txl.warp_id()
        lane = txl.lane_id()
        tid = txl.thread_id()
        cl = cta // CS

        smem = txl.smem_pool()
        a_tile = smem.alloc((STAGES, BM, BK), txl.f8e4m3, align=1024, swizzle=txl.SW128B)
        b_gu = smem.alloc((NB * KB, NT, BK), txl.f8e4m3, align=1024, swizzle=txl.SW128B)
        if RANK_ACT:


            b_act = smem.alloc((2 * CS, NT, CH), txl.f8e4m3, align=1024, swizzle=txl.SW64B)
            s_amx = smem.alloc((2, CS, NT), txl.f32, align=16)
        else:
            b_act = smem.alloc((2, NT, BK), txl.f8e4m3, align=1024, swizzle=txl.SW128B)
            s_hvx = smem.alloc((2, CH, NT), txl.f32, align=16)
        s_xs = smem.alloc((NB, NT * KB), txl.f32, align=16)
        s_prod = smem.alloc((2, 2 * KB, NT), txl.f32, align=16)
        s_ids = smem.alloc((P + 16,), txl.i32, align=16)
        s_mask = smem.alloc((E * MW,), txl.u32, align=16)
        s_cnt = smem.alloc((E,), txl.i32, align=16)
        s_chunk = smem.alloc((P + 16,), txl.i32, align=16)
        s_wsum = smem.alloc((NWARPS,), txl.i32, align=16)
        s_task = smem.alloc((TASK_RING, TASK_W), txl.i32, align=16)
        s_amax = smem.alloc((NT, 4), txl.f32, align=16)
        s_scl = smem.alloc((NT, 2), txl.f32, align=16)
        s_misc = smem.alloc((8,), txl.i32, align=16)
        s_stg = smem.alloc((2, NT, BM), txl.u16, align=128)


        def s_up_ptr(ch, t4):
            return txl.ptx.addr(s_stg.ptr_to([0, 0, 0]), (ch * NT + t4) * 4)
        s_ridx = smem.alloc((TASK_RING,), txl.i32, align=16)
        tmem_slot = smem.alloc((1,), txl.u32, align=4)

        full_bar = txl.TMABar(smem, STAGES)
        empty_bar = txl.TCGen05Bar(smem, STAGES)
        tfull = txl.TCGen05Bar(smem, NTB)
        tempty = txl.MBarrier(smem, NTB)
        task_full = txl.MBarrier(smem, TASK_RING)
        task_hdr = txl.MBarrier(smem, TASK_RING)
        task_empty = txl.MBarrier(smem, TASK_RING)
        bq_full = txl.MBarrier(smem, 2)
        bq_empty = txl.TCGen05Bar(smem, 2)
        prod_empty = txl.MBarrier(smem, 2)
        bready = txl.MBarrier(smem, 1)
        if RANK_ACT:
            act_empty = txl.TCGen05Bar(smem, 2)
            afull = txl.TMABar(smem, 2)
            actfull = txl.TMABar(smem, 2)
            xchg_free = txl.MBarrier(smem, 2)
            slot_free = xchg_free
        else:
            hvfull = txl.TMABar(smem, 2)
            hvfree = txl.MBarrier(smem, 2)
            slot_free = hvfree
        aq_full = txl.MBarrier(smem, 2)
        task_tok = txl.MBarrier(smem, TASK_RING)
        ridx_full = txl.TMABar(smem, TASK_RING)
        ridx_free = txl.MBarrier(smem, TASK_RING)

        full_bar.init(1)
        empty_bar.init(1)
        tfull.init(1)
        tempty.init(4)
        task_full.init(1)
        task_hdr.init(1)
        task_empty.init(N_TASK_CONSUMERS)
        bq_full.init(4)
        bq_empty.init(1)
        prod_empty.init(4)
        bready.init(8)
        if RANK_ACT:
            act_empty.init(1)
            afull.init(1)
            actfull.init(1)
            xchg_free.init(1)
        else:
            hvfull.init(1)
            hvfree.init(1)
        aq_full.init(1)
        task_tok.init(1)
        ridx_full.init(1)
        ridx_free.init(1)
        txl.ptx.fence.mbarrier_init.release.cluster()

        with txl.If(txl.And(cta == 0, tid == 0)), txl.Then():
            txl.ptx.st.global_.u32(sync_ctr.ptr_to([txl.cast((epoch + txl.uint32(1)) % txl.uint32(WORK_SLOTS), "int32")]), txl.uint32(0))
        with txl.If(warp == 1), txl.Then():
            txl.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                txl.address_of(tmem_slot[0]), txl.uint32(TMEM_COLS)
            )
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
        with txl.If(txl.And(warp == 0, lane == 0)), txl.Then():
            txl.ptx.prefetch.tensormap(txl.address_of(tm_w1h))
            txl.ptx.prefetch.tensormap(txl.address_of(tm_w2))
        with txl.If(tid < (M + G - 1) // G), txl.Then():
            row = cta + G * tid
            with txl.If(row < M), txl.Then():
                txl.ptx["cp.async.bulk.prefetch.L2.global"](hidden.ptr_to([row * (HID // 2)]), txl.uint32(HID * 2))


        for i in range(ZERO_PER_T):
            wi = (cta + G * (tid + NTHREADS * i)) * 4
            with txl.If(wi < OUT_WORDS), txl.Then():
                txl.ptx.st.global_.v4.b32(out.ptr_to([wi]), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0))


        for i in range((MASK_COUNT + NTHREADS - 1) // NTHREADS):
            idx = tid + NTHREADS * i
            with txl.If(idx < MASK_COUNT), txl.Then():
                txl.ptx.st.shared.u32(s_mask.ptr_to([idx]), txl.uint32(0))
        for i in range((NB * B_BYTES + NTHREADS * 16 - 1) // (NTHREADS * 16)):
            with txl.If((tid + NTHREADS * i) * 16 < NB * B_BYTES), txl.Then():
                txl.ptx.st.shared.v4.b32(
                    txl.ptx.addr(b_gu[0].ptr_to(0, 0), (tid + NTHREADS * i) * 16),
                    txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0),
                )
        idv = txl.alloc_local((IDS_PER_T,), txl.i32)
        for i in range(ROUTE_IDS):
            txl.assign(idv[i], txl.int32(0))
            p = tid + NTHREADS * i
            with txl.If(p < P), txl.Then():
                txl.ptx.ld.global_.nc.s32(idv[i], topk_ids.ptr_to([p]))
        txl.cuda.cta_sync()
        for i in range(ROUTE_IDS):
            p = tid + NTHREADS * i
            with txl.If(p < P), txl.Then():
                v = idv[i]
                txl.ptx.st.shared.s32(s_ids.ptr_to([p]), v)
                t_p = p // TOPK
                bit = txl.shift_left(txl.uint32(1), txl.cast(t_p % 32, "uint32"))
                old_m = txl.local_scalar(txl.u32)
                txl.ptx.atom.shared.or_.b32(old_m, s_mask.ptr_to([v * MW + t_p // 32]), bit)
        txl.cuda.cta_sync()
        cnt_l = txl.alloc_local((EPT,), txl.i32)
        nch_l = txl.alloc_local((EPT,), txl.i32)
        local_sum = _i32(txl.int32(0))
        for i in range(ROUTE_EPT):
            e_i = tid * EPT + i
            txl.assign(cnt_l[i], txl.int32(0))
            with txl.If(e_i < E), txl.Then():
                for w_ in range(MW):
                    mwv = txl.local_scalar(txl.u32)
                    txl.ptx.ld.shared.u32(mwv, s_mask.ptr_to([e_i * MW + w_]))
                    txl.assign(cnt_l[i], cnt_l[i] + txl.cast(txl.popcount(mwv), "int32"))
            txl.assign(nch_l[i], (cnt_l[i] + (NT - 1)) // NT)
            txl.assign(local_sum, local_sum + nch_l[i])
        incl = _i32(local_sum)
        if M != 1:
            for d in (1, 2, 4, 8, 16):
                o = txl.local_scalar(txl.i32)
                txl.ptx.shfl_sync.up.b32(o, incl, txl.uint32(d), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                txl.assign(incl, incl + txl.Select(lane >= d, o, txl.int32(0)))
            with txl.If(lane == 31), txl.Then():
                txl.ptx.st.shared.s32(s_wsum.ptr_to([warp]), incl)
            txl.cuda.cta_sync()
        pre = _i32(txl.int32(0))
        tot = _i32(txl.int32(0))
        for w in range(0 if M == 1 else NWARPS):
            v_w = txl.local_scalar(txl.i32)
            txl.ptx.ld.shared.s32(v_w, s_wsum.ptr_to([w]))
            txl.assign(pre, pre + txl.Select(warp > w, v_w, txl.int32(0)))
            txl.assign(tot, tot + v_w)
        excl = _i32(pre + incl - local_sum)
        for i in range(ROUTE_EPT):
            e_i = tid * EPT + i
            with txl.If(e_i < E), txl.Then():
                txl.ptx.st.shared.s32(s_cnt.ptr_to([e_i]), cnt_l[i])
                with txl.serial(0, nch_l[i]) as c_:
                    txl.ptx.st.shared.s32(
                        s_chunk.ptr_to([excl + c_]), e_i + txl.shift_left(c_, txl.int32(16))
                    )
            txl.assign(excl, excl + nch_l[i])
        if M == 1:
            with txl.If(tid < TOPK), txl.Then():
                route_e = txl.local_scalar(txl.i32)
                txl.ptx.ld.global_.nc.s32(route_e, topk_ids.ptr_to([tid]))
                txl.ptx.st.shared.s32(s_chunk.ptr_to([tid]), route_e)
            with txl.If(tid == 0), txl.Then():
                txl.ptx.st.shared.s32(s_misc.ptr_to([0]), txl.int32(TOPK))
        else:
            with txl.If(tid == 0), txl.Then():
                txl.ptx.st.shared.s32(s_misc.ptr_to([0]), tot)
        txl.cuda.cta_sync()



        txl.ptx.barrier.cluster.arrive.relaxed()
        with txl.If(warp == 0), txl.Then():
            txl.cuda.iket.mark("tables-ready")
        C = txl.local_scalar(txl.i32)
        txl.ptx.ld.shared.s32(C, s_misc.ptr_to([0]))
        # Shape-only round-robin ownership removes the global work counter and
        # the per-item peer-index handoff at every routed shape.
        dyn = txl.int32(0) != txl.int32(0)
        tmem_base = txl.local_scalar(txl.u32)
        txl.ptx.ld.shared.u32(tmem_base, tmem_slot.ptr_to([0]))


        def rcp_refined(scale):
            rcp0 = txl.local_scalar(txl.f32)
            txl.ptx.rcp.approx.ftz.f32(rcp0, scale)
            rerr = txl.local_scalar(txl.f32)
            txl.ptx[FMA_F32](rerr, txl.float32(0.0) - scale, rcp0, txl.float32(1.0))
            rcp1 = txl.local_scalar(txl.f32)
            txl.ptx[FMA_F32](rcp1, rerr, rcp0, rcp0)
            return rcp1

        def div_rn(x, scale, rcp1):
            q0 = _f32(x * rcp1)
            err = txl.local_scalar(txl.f32)
            txl.ptx[FMA_F32](err, txl.float32(0.0) - q0, scale, x)
            q = txl.local_scalar(txl.f32)
            txl.ptx[FMA_F32](q, err, rcp1, q0)
            return q

        def read_task(ts, e_dst, ntok_dst):
            task_full.wait(ts.stage, ts.phase)
            txl.ptx.ld.shared.s32(e_dst, s_task.ptr_to([ts.stage, 0]))
            txl.ptx.ld.shared.s32(ntok_dst, s_task.ptr_to([ts.stage, 2]))

        def release_task(ts, single_lane):
            if single_lane:
                task_empty.arrive(ts.stage)
            else:
                txl.cuda.warp_sync()
                with txl.If(lane == 0), txl.Then():
                    task_empty.arrive(ts.stage)
            ts.advance()

        def remote_u32(ptr, peer_):
            """shared::cluster address of this CTA-local pointer in CTA `peer_` of the cluster."""
            r = txl.local_scalar(txl.u32)
            txl.ptx.mapa.shared__cluster.u32(r, txl.cuda.cvta_generic_to_shared(ptr), txl.cast(peer_, "uint32"))
            return r

        def cluster_wait(bar, stage, phase):
            """mbarrier phase wait with cluster-scope acquire (tracked bytes were written by the peer)."""
            ok = txl.local_scalar(txl.u32, init=txl.uint32(0))
            with txl.While(ok == txl.uint32(0)):
                txl.ptx.mbarrier.try_wait.parity.acquire.cluster.shared.b64(
                    ok, bar.ptr_to([stage]), txl.cast(phase, "uint32")
                )

        def quant_unit_T(T, n, kb, gbuf, tok, sub, to_global=False):
            """T consecutive lanes quantize one (token row, k-block) unit; lane `sub` owns 128/T values.
            Destination: B row `n` of buffer `gbuf` (shared) or the global fp8 row buffer."""
            NW = 64 // T
            w = txl.alloc_local((NW,), txl.u32)
            src = hidden.ptr_to([tok * (HID // 2) + kb * (BK // 2) + sub * NW])
            for i in range(NW // 4):
                txl.ptx.ld.global_.nc.v4.b32(w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3], txl.ptx.addr(src, 16 * i))
            am4 = [_f32(txl.float32(0.0)) for _ in range(4)]
            for j in range(NW):
                txl.assign(am4[j % 4], txl.max(am4[j % 4], txl.max(txl.fabs(_bf16_lo(w[j])), txl.fabs(_bf16_hi(w[j])))))
            amax = _f32(txl.max(txl.max(am4[0], am4[1]), txl.max(am4[2], am4[3])))
            m_ = T // 2
            while m_ >= 1:
                o = txl.local_scalar(txl.f32)
                txl.ptx.shfl_sync.bfly.b32(o, amax, txl.uint32(m_), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF))
                txl.assign(amax, txl.max(amax, o))
                m_ //= 2
            scale = _f32(txl.max(amax, txl.float32(1.0e-8)) * txl.float32(INV_FP8_MAX))
            rcp1 = rcp_refined(scale)


            for i in range(NW // 8):
                qw = txl.alloc_local((4,), txl.u32)
                for jx in range(4):
                    a = 8 * i + 2 * jx
                    h_lo = txl.local_scalar(txl.u16)
                    h_hi = txl.local_scalar(txl.u16)
                    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(h_lo, _bf16_hi(w[a]) * rcp1, _bf16_lo(w[a]) * rcp1)
                    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(h_hi, _bf16_hi(w[a + 1]) * rcp1, _bf16_lo(w[a + 1]) * rcp1)
                    txl.assign(qw[jx], txl.bitwise_or(txl.cast(h_lo, "uint32"),
                                                     txl.shift_left(txl.cast(h_hi, "uint32"), txl.uint32(16))))
                if to_global:
                    txl.ptx.st.global_.v4.b32(xq_g.ptr_to([tok * (HID // 4) + kb * 32 + sub * (32 // T) + 4 * i]),
                                              qw[0], qw[1], qw[2], qw[3])
                else:
                    txl.ptx.st.shared.v4.b32(b_gu[gbuf * KB + kb].ptr_to(n, sub * (128 // T) + 16 * i), qw[0], qw[1], qw[2], qw[3])
            with txl.If(sub == 0), txl.Then():
                if to_global:
                    txl.ptx.st.global_.f32(xs_g.ptr_to([tok * KB + kb]), scale)
                else:
                    txl.ptx.st.shared.f32(s_xs.ptr_to([gbuf, n * KB + kb]), scale)


        def prologue_quant(tq8):
            """M <= 16: quantize all token rows into B buffer 0 (row n = token n); 256 threads (tq8 in [0, 256))."""
            T = 1
            while T * 2 <= min(8, 256 // (M * KB)):
                T *= 2
            UNITS = M * KB
            for ps in range((UNITS * T + 255) // 256):
                u = (tq8 + 256 * ps) // T
                sub = tq8 % T
                if (ps + 1) * (256 // T) <= UNITS:
                    quant_unit_T(T, u // KB, u % KB, txl.int32(0), u // KB, sub)
                else:
                    with txl.If(u < UNITS), txl.Then():
                        quant_unit_T(T, u // KB, u % KB, txl.int32(0), u // KB, sub)
            for n in range(M, NT):
                with txl.If(tq8 < KB), txl.Then():
                    txl.ptx.st.shared.f32(s_xs.ptr_to([0, n * KB + tq8]), txl.float32(0.0))
            txl.ptx.fence.proxy.async_.shared__cta()
            txl.cuda.warp_sync()
            with txl.If(lane == 0), txl.Then():
                bready.arrive(0)


        def first_item_quant(slot, ntok_):
            """Chunk mode, item 0: the 4 quantizer warps quantize the chunk's rows straight into B buffer 0
            (lane count per (row, k-block) unit chosen from the token count), so the first item never waits
            for other CTAs' row flags."""
            def variant(T):
                units = ntok_ * KB
                max_rows = {8: 2, 4: 4, 2: 8, 1: NT}[T]
                for ps in range((max_rows * KB * T + NQUANT - 1) // NQUANT):
                    u = (tq + NQUANT * ps) // T
                    sub = tq % T
                    with txl.If(u < units), txl.Then():
                        tok = txl.local_scalar(txl.i32)
                        txl.ptx.ld.shared.s32(tok, s_task.ptr_to([slot, TOK_OFF + u // KB]))
                        quant_unit_T(T, u // KB, u % KB, txl.int32(0), tok, sub)
            with txl.If(ntok_ <= 2):
                with txl.Then():
                    variant(8)
                with txl.Else():
                    with txl.If(ntok_ <= 4):
                        with txl.Then():
                            variant(4)
                        with txl.Else():
                            with txl.If(ntok_ <= 8):
                                with txl.Then():
                                    variant(2)
                                with txl.Else():
                                    variant(1)
            for n in range(NT):
                with txl.If(txl.And(n >= ntok_, tq < KB)), txl.Then():
                    txl.ptx.st.shared.f32(s_xs.ptr_to([0, n * KB + tq]), txl.float32(0.0))

        roles = txl.specialize()
        wg0 = roles.warpgroup("wg0", warps=[0, 1, 2, 3], regs=REGS_WG0)
        prod_role = roles.role("prod", warps=[0], group=wg0)
        mma_role = roles.role("mma", warps=[1], group=wg0)
        sched_role = roles.role("sched", warps=[2], group=wg0)
        idle_role = roles.role("idle", warps=[3], group=wg0)
        math_role = roles.role("math", warps=list(range(MATH_WARP0, QUANT_WARP0)), regs=REGS_MATH)
        quant_role = roles.role("quant", warps=list(range(QUANT_WARP0, NWARPS)), regs=REGS_QUANT)

        with wg0:

            with sched_role:
                ts = txl.PipelineState(TASK_RING, phase=0)
                rst = txl.PipelineState(TASK_RING, phase=0)
                idx = _i32(cl)
                nxt_idx = _i32(txl.int32(0))
                nxt = txl.local_scalar(txl.u32, init=txl.uint32(0))
                work_slot = txl.cast(epoch % txl.uint32(WORK_SLOTS), "int32")
                lane_lt = txl.shift_left(txl.uint32(1), txl.cast(lane, "uint32")) - txl.uint32(1)
                running = _i32(txl.int32(1))
                txl.ptx.barrier.cluster.wait()
                txl.cuda.iket.mark("sched-start")

                with txl.If(txl.And(dyn, txl.And(rank == 0, lane == 0))), txl.Then():
                    txl.ptx.atom.relaxed.gpu.global_.add.u32(nxt, sync_ctr.ptr_to([work_slot]), txl.uint32(1))
                with txl.While(running == 1):
                    tk_s = _rng("s-decode")
                    e = _i32(txl.int32(-1))
                    c = _i32(txl.int32(0))
                    ntok = _i32(txl.int32(0))
                    mask0 = _u32(txl.uint32(0))
                    with txl.If(idx < C), txl.Then():
                        packed = txl.local_scalar(txl.i32)
                        txl.ptx.ld.shared.s32(packed, s_chunk.ptr_to([idx]))
                        txl.assign(e, txl.bitwise_and(packed, txl.int32(0xFFFF)))
                        txl.assign(c, txl.shift_right(packed, txl.int32(16)))
                    task_empty.wait(ts.stage, ts.phase ^ 1)
                    with txl.If(e >= 0), txl.Then():
                        if M == 1:
                            txl.assign(ntok, txl.int32(1))
                            txl.assign(mask0, txl.uint32(1))
                        else:
                            cnt_e = txl.local_scalar(txl.i32)
                            txl.ptx.ld.shared.s32(cnt_e, s_cnt.ptr_to([e]))
                            txl.assign(ntok, txl.min(cnt_e - c * NT, txl.int32(NT)))
                            txl.ptx.ld.shared.u32(mask0, s_mask.ptr_to([e * MW]))

                    with txl.If(lane == 0), txl.Then():
                        txl.ptx.st.shared.s32(s_task.ptr_to([ts.stage, 0]), e)
                        txl.ptx.st.shared.s32(s_task.ptr_to([ts.stage, 1]), c)
                        txl.ptx.st.shared.s32(s_task.ptr_to([ts.stage, 2]), ntok)
                        txl.ptx.st.shared.u32(s_task.ptr_to([ts.stage, 3]), mask0)
                        task_hdr.arrive(ts.stage)
                    with txl.If(e >= 0), txl.Then():
                        t_l = _i32(txl.int32(0))
                        rw = _f32(txl.float32(0.0))
                        if SHARED_B:
                            if M == 1:


                                with txl.If(lane == 0), txl.Then():
                                    txl.ptx.ld.global_.nc.f32(rw, topk_w.ptr_to([idx]))
                            else:

                                with txl.If(lane < M), txl.Then():
                                    txl.assign(t_l, lane)
                                    routed = txl.bitwise_and(txl.shift_right(mask0, txl.cast(lane, "uint32")), txl.uint32(1)) == txl.uint32(1)
                                    k_l = _i32(txl.int32(0))
                                    for kk in range(TOPK):
                                        idk = txl.local_scalar(txl.i32)
                                        txl.ptx.ld.shared.s32(idk, s_ids.ptr_to([lane * TOPK + kk]))
                                        txl.assign(k_l, txl.Select(idk == e, txl.int32(kk), k_l))
                                    with txl.If(routed), txl.Then():
                                        txl.ptx.ld.global_.nc.f32(rw, topk_w.ptr_to([lane * TOPK + k_l]))
                        else:
                            base = _i32(txl.int32(0))
                            for w_ in range(MW):
                                mwv = txl.local_scalar(txl.u32)
                                txl.ptx.ld.shared.u32(mwv, s_mask.ptr_to([e * MW + w_]))
                                mine_t = txl.bitwise_and(txl.shift_right(mwv, txl.cast(lane, "uint32")), txl.uint32(1)) == txl.uint32(1)
                                r = base + txl.cast(txl.popcount(txl.bitwise_and(mwv, lane_lt)), "int32") - c * NT
                                with txl.If(txl.And(mine_t, txl.And(r >= 0, r < NT))), txl.Then():
                                    txl.ptx.st.shared.s32(s_task.ptr_to([ts.stage, TOK_OFF + r]), txl.int32(32 * w_) + lane)
                                txl.assign(base, base + txl.cast(txl.popcount(mwv), "int32"))
                            txl.cuda.warp_sync()
                            if FIRST_TILE:
                                # The token list is complete: publish it before the global route-weight
                                # and scale loads so the quantizer warps start the item's B tile at once.
                                with txl.If(lane == 0), txl.Then():
                                    task_tok.arrive(ts.stage)
                            with txl.If(lane < NT), txl.Then():
                                txl.ptx.ld.shared.s32(t_l, s_task.ptr_to([ts.stage, TOK_OFF + txl.min(lane, ntok - 1)]))
                                k_l = _i32(txl.int32(0))
                                for kk in range(TOPK):
                                    idk = txl.local_scalar(txl.i32)
                                    txl.ptx.ld.shared.s32(idk, s_ids.ptr_to([t_l * TOPK + kk]))
                                    txl.assign(k_l, txl.Select(idk == e, txl.int32(kk), k_l))
                                txl.ptx.ld.global_.nc.f32(rw, topk_w.ptr_to([t_l * TOPK + k_l]))
                        w1v = txl.local_scalar(txl.f32)
                        txl.ptx.ld.global_.nc.f32(w1v, w1s.ptr_to([e * (2 * KB) + lane]))
                        w2v = _f32(txl.float32(0.0))
                        with txl.If(lane < KB), txl.Then():
                            txl.ptx.ld.global_.nc.f32(w2v, w2s.ptr_to([e * KB + lane]))
                        txl.cuda.warp_sync()
                        txl.ptx.st.shared.f32(s_task.ptr_to([ts.stage, W1_OFF + lane]), w1v)
                        with txl.If(lane < KB), txl.Then():
                            txl.ptx.st.shared.f32(s_task.ptr_to([ts.stage, W2_OFF + lane]), w2v)
                        with txl.If(lane < NT), txl.Then():
                            if not FIRST_TILE:
                                txl.ptx.st.shared.s32(s_task.ptr_to([ts.stage, TOK_OFF + lane]), t_l)
                            else:
                                # Entries [0, ntok) were published with task_tok and may already be
                                # read by the quantizer warps; only fill the padding rows here.
                                with txl.If(lane >= ntok), txl.Then():
                                    txl.ptx.st.shared.s32(s_task.ptr_to([ts.stage, TOK_OFF + lane]), t_l)
                            txl.ptx.st.shared.f32(s_task.ptr_to([ts.stage, RW_OFF + lane]), rw)
                    txl.cuda.warp_sync()
                    with txl.If(lane == 0), txl.Then():
                        if FIRST_TILE:
                            with txl.If(e < 0), txl.Then():
                                task_tok.arrive(ts.stage)
                        task_full.arrive(ts.stage)
                    ts.advance()


                    with txl.If(dyn):
                        with txl.Then():
                            with txl.If(rank == 0):
                                with txl.Then():
                                    nxt_b = txl.local_scalar(txl.u32)
                                    txl.ptx.shfl_sync.idx.b32(nxt_b, nxt, txl.uint32(0), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF))
                                    txl.assign(nxt_idx, txl.int32(NCL) + txl.cast(nxt_b, "int32"))
                                    with txl.If(lane == 0), txl.Then():
                                        ridx_free.wait(rst.stage, rst.phase ^ 1)
                                        r_slot = remote_u32(s_ridx.ptr_to([rst.stage]), peer)
                                        r_bar = remote_u32(ridx_full.ptr_to([rst.stage]), peer)
                                        txl.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.b32(r_slot, txl.cast(nxt_idx, "uint32"), r_bar)
                                        with txl.If(nxt_idx < C), txl.Then():
                                            txl.ptx.atom.relaxed.gpu.global_.add.u32(nxt, sync_ctr.ptr_to([work_slot]), txl.uint32(1))
                                with txl.Else():
                                    with txl.If(lane == 0), txl.Then():
                                        ridx_full.arrive(rst.stage, tx_count=4)
                                    cluster_wait(ridx_full, rst.stage, rst.phase)
                                    txl.ptx.ld.shared.s32(nxt_idx, s_ridx.ptr_to([rst.stage]))
                                    txl.cuda.warp_sync()
                                    with txl.If(lane == 0), txl.Then():
                                        txl.ptx.mbarrier.arrive.release.cluster.shared__cluster.b64(remote_u32(ridx_free.ptr_to([rst.stage]), peer))
                            rst.advance()
                        with txl.Else():
                            txl.assign(nxt_idx, idx + txl.int32(NCL))
                    _rng_end(tk_s)
                    with txl.If(e < 0):
                        with txl.Then():
                            txl.assign(running, txl.int32(0))
                        with txl.Else():
                            txl.assign(idx, nxt_idx)

            with idle_role:
                if FIRST_TILE:
                    # Quantize and publish this CTA's own token row(s) for other CTAs' gathers (two
                    # lanes per k-block), so the quantizer warps start the first item's B tile at once.
                    tk_i = _rng("i-rowpub")
                    for r_ in range(ROWS_PER_CTA):
                        row = cta + G * r_
                        with txl.If(row < M), txl.Then():
                            quant_unit_T(RQ_T, txl.int32(0), lane // RQ_T, txl.int32(0), row, lane % RQ_T, to_global=True)
                        txl.cuda.warp_sync()
                        with txl.If(txl.And(row < M, lane == 0)), txl.Then():
                            txl.ptx.st.release.gpu.global_.u32(xflag_g.ptr_to([row]), epoch)
                    _rng_end(tk_i)
                with txl.If(lane == 0), txl.Then():
                    txl.ptx.red.release.gpu.global_.add.u32(sync_ctr.ptr_to([ZERO_DONE]), txl.uint32(1))
                if GLOBAL_Q and not FIRST_TILE:
                    for r_ in range(ROWS_PER_CTA):
                        row = cta + G * r_
                        txl.ptx.bar.sync(txl.uint32(BAR_ROWS), txl.uint32(NQUANT + 32))
                        with txl.If(txl.And(row < M, lane == 0)), txl.Then():
                            txl.ptx.st.release.gpu.global_.u32(xflag_g.ptr_to([row]), epoch)
                # Cluster slot-release agent (see v47 notes): release the peer's slot k%2 once this
                # CTA published item k's activation (aq_full) and, in the rank path, its down MMAs
                # finished reading the activation slot (act_empty).  Only items with a successor k+2
                # are released, so the peer is always alive when the remote arrive lands.
                n_items = _i32(txl.Select(cl < C, (C - cl + txl.int32(NCL - 1)) // txl.int32(NCL), txl.int32(0)))
                k_i = _i32(txl.int32(0))
                with txl.While(k_i + txl.int32(2) < n_items):
                    par_i = _i32(k_i % 2)
                    ph_i = _i32((k_i // 2) % 2)
                    aq_full.wait(par_i, ph_i)
                    if RANK_ACT:
                        act_empty.wait(par_i, ph_i)
                        txl.ptx.tcgen05.fence__after_thread_sync()
                    txl.cuda.warp_sync()
                    with txl.If(lane == 0), txl.Then():
                        txl.ptx.mbarrier.arrive.release.cluster.shared__cluster.b64(
                            remote_u32(slot_free.ptr_to([par_i]), peer)
                        )
                    txl.assign(k_i, k_i + txl.int32(1))
                txl.ptx.barrier.cluster.wait()


            with prod_role:
                with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                    st = txl.PipelineState(STAGES, phase=0)
                    ts = txl.PipelineState(TASK_RING, phase=0)
                    e = _i32(txl.int32(0))

                    def issue_gu(kb):
                        """One MMA tile = my 64 gate rows (tile rows 0-63) + my 64 up rows (rows 64-127)."""
                        empty_bar.wait(st.stage, st.phase ^ 1)
                        full_bar.arrive(st.stage, tx_count=TILE_A)
                        txl.ptx[TMA_G2S_3D](
                            a_tile[st.stage].ptr_to(0, 0), txl.address_of(tm_w1h),
                            kb * BK, rank * CH, e * 2,
                            full_bar.ptr_to([st.stage]), txl.uint64(EVICT_FIRST),
                        )
                        st.advance()

                    def issue_w2_pair(h):


                        empty_bar.wait(st.stage, st.phase ^ 1)
                        empty_bar.wait(st.stage + 1, st.phase ^ 1)
                        full_bar.arrive(st.stage, tx_count=2 * TILE_A)
                        txl.ptx[TMA_G2S_3D](
                            a_tile[st.stage].ptr_to(0, 0), txl.address_of(tm_w2),
                            txl.int32(0), txl.int32(0), e * (HID // BM) + h,
                            full_bar.ptr_to([st.stage]), txl.uint64(EVICT_FIRST),
                        )
                        st.advance()
                        st.advance()

                    def issue_w2(h):
                        empty_bar.wait(st.stage, st.phase ^ 1)
                        full_bar.arrive(st.stage, tx_count=TILE_A)
                        txl.ptx[TMA_G2S_2D](
                            a_tile[st.stage].ptr_to(0, 0), txl.address_of(tm_w2),
                            txl.int32(0), e * HID + h * BM,
                            full_bar.ptr_to([st.stage]), txl.uint64(EVICT_FIRST),
                        )
                        st.advance()

                    def peek_next(dst):
                        nst = _i32(ts.stage + 1)
                        nph = _i32(ts.phase)
                        with txl.If(nst == TASK_RING), txl.Then():
                            txl.assign(nst, txl.int32(0))
                            txl.assign(nph, nph ^ 1)
                        task_hdr.wait(nst, nph)
                        txl.ptx.ld.shared.s32(dst, s_task.ptr_to([nst, 0]))

                    e_cur = _i32(txl.int32(-1))
                    e_nxt = _i32(txl.int32(-1))
                    txl.cuda.iket.mark("prod-start")

                    with txl.If(cl < C), txl.Then():
                        packed0 = txl.local_scalar(txl.i32)
                        txl.ptx.ld.shared.s32(packed0, s_chunk.ptr_to([cl]))
                        txl.assign(e_cur, txl.bitwise_and(packed0, txl.int32(0xFFFF)))
                    with txl.If(e_cur >= 0), txl.Then():
                        txl.assign(e, e_cur)
                        tk_p = _rng("p-gu")
                        with txl.serial(0, KB) as kb:
                            issue_gu(kb)
                        _rng_end(tk_p)
                    task_hdr.wait(ts.stage, ts.phase)
                    with txl.While(e_cur >= 0):


                        peek_next(e_nxt)
                        with txl.If(e_nxt >= 0), txl.Then():
                            txl.assign(e, e_nxt)
                            tk_p = _rng("p-gu")
                            with txl.serial(0, KB) as kb:
                                issue_gu(kb)
                            _rng_end(tk_p)
                        txl.assign(e, e_cur)
                        tk_p = _rng("p-d")
                        if PAIR_D:
                            with txl.serial(0, HBC // 2) as hp:
                                issue_w2_pair(rank * HBC + 2 * hp)
                        else:
                            with txl.serial(0, HBC) as h:
                                issue_w2(rank * HBC + h)
                        _rng_end(tk_p)
                        release_task(ts, True)
                        txl.assign(e_cur, e_nxt)
                    release_task(ts, True)
                txl.ptx.barrier.cluster.wait()


            with mma_role:
                with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                    sst = txl.PipelineState(STAGES, phase=0)
                    tst = txl.PipelineState(NTB, phase=0)
                    ts = txl.PipelineState(TASK_RING, phase=0)
                    gst = txl.PipelineState(2, phase=0)
                    par = _i32(txl.int32(0))
                    ast = txl.PipelineState(2, phase=0)
                    e = _i32(txl.int32(0))
                    ntok = _i32(txl.int32(0))

                    def mma_issue(bview):
                        a_desc, a_off = a_tile[sst.stage].encode(major="k", mma_k=32)
                        b_desc, b_off = bview.encode(major="k", mma_k=32)
                        d_addr = tmem_base + txl.Cast("uint32", tst.stage) * txl.uint32(NT)
                        for ki in range(BK // 32):
                            txl.ptx[MMA](
                                d_addr, a_desc + a_off(ki), b_desc + b_off(ki), txl.uint32(IDESC),
                                txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0),
                                txl.ptx.pred(txl.uint32(1 if ki > 0 else 0)),
                            )

                    def mma_issue_d(act_par):
                        a_desc, a_off = a_tile[sst.stage].encode(major="k", mma_k=32)
                        b0_desc, b0_off = b_act[act_par * CS].encode(major="k", mma_k=32)
                        b1_desc, b1_off = b_act[act_par * CS + 1].encode(major="k", mma_k=32)
                        d_addr = tmem_base + txl.Cast("uint32", tst.stage) * txl.uint32(NT)
                        for ki in range(CH // 32):
                            txl.ptx[MMA](
                                d_addr, a_desc + a_off(ki), b0_desc + b0_off(ki), txl.uint32(IDESC),
                                txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0),
                                txl.ptx.pred(txl.uint32(1 if ki > 0 else 0)),
                            )
                        for ki in range(CH // 32):
                            txl.ptx[MMA](
                                d_addr, a_desc + a_off(CH // 32 + ki), b1_desc + b1_off(ki), txl.uint32(IDESC),
                                txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0),
                                txl.ptx.pred(txl.uint32(1)),
                            )

                    def mma_tile(bview, wait_full=True):
                        if wait_full:
                            full_bar.wait(sst.stage, sst.phase)
                        tempty.wait(tst.stage, tst.phase ^ 1)
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        mma_issue(bview)
                        txl.ptx.tcgen05.fence__before_thread_sync()
                        empty_bar.arrive(sst.stage)
                        with txl.If(tst.stage % 4 == 3), txl.Then():
                            tfull.arrive(tst.stage)
                        sst.advance()
                        tst.advance()

                    def mma_pair(bview):



                        full_bar.wait(sst.stage, sst.phase)
                        full_bar.arrive(sst.stage + 1)
                        mma_tile(bview, wait_full=False)
                        mma_tile(bview)

                    def mma_tile_d(act_par, wait_full=True):
                        if wait_full:
                            full_bar.wait(sst.stage, sst.phase)
                        tempty.wait(tst.stage, tst.phase ^ 1)
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        mma_issue_d(act_par)
                        txl.ptx.tcgen05.fence__before_thread_sync()
                        empty_bar.arrive(sst.stage)
                        with txl.If(tst.stage % 4 == 3), txl.Then():
                            tfull.arrive(tst.stage)
                        sst.advance()
                        tst.advance()

                    def mma_pair_d(act_par):
                        full_bar.wait(sst.stage, sst.phase)
                        full_bar.arrive(sst.stage + 1)
                        mma_tile_d(act_par, wait_full=False)
                        mma_tile_d(act_par)

                    def peek_next(dst):
                        nst = _i32(ts.stage + 1)
                        nph = _i32(ts.phase)
                        with txl.If(nst == TASK_RING), txl.Then():
                            txl.assign(nst, txl.int32(0))
                            txl.assign(nph, nph ^ 1)
                        task_hdr.wait(nst, nph)
                        txl.ptx.ld.shared.s32(dst, s_task.ptr_to([nst, 0]))

                    def gu_mmas():
                        gbuf = _i32(txl.int32(0))
                        if not SHARED_B:
                            tk_m = _rng("mma-wait-b")
                            bq_full.wait(gst.stage, gst.phase)
                            txl.assign(gbuf, gst.stage)
                            _rng_end(tk_m)
                        tk_m = _rng("mma-gu")
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        with txl.serial(0, KB) as kb:
                            mma_tile(b_gu[gbuf * KB + kb])
                        if not SHARED_B:
                            bq_empty.arrive(gst.stage)
                            gst.advance()
                        _rng_end(tk_m)

                    if SHARED_B:
                        bready.wait(0, 0)
                        txl.ptx.tcgen05.fence__after_thread_sync()
                    e_cur = _i32(txl.int32(-1))
                    e_nxt = _i32(txl.int32(-1))
                    task_hdr.wait(ts.stage, ts.phase)
                    txl.ptx.ld.shared.s32(e_cur, s_task.ptr_to([ts.stage, 0]))
                    with txl.If(e_cur >= 0), txl.Then():
                        gu_mmas()
                    with txl.While(e_cur >= 0):
                        peek_next(e_nxt)
                        with txl.If(e_nxt >= 0), txl.Then():
                            gu_mmas()
                        tk_m = _rng("mma-wait-act")
                        aq_full.wait(ast.stage, ast.phase)
                        ast.advance()
                        _rng_end(tk_m)
                        tk_m = _rng("mma-d")
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        if RANK_ACT:
                            if PAIR_D:
                                with txl.serial(0, HBC // 2) as _hp:
                                    mma_pair_d(par)
                            else:
                                with txl.serial(0, HBC) as _h:
                                    mma_tile_d(par)
                            act_empty.arrive(par)
                        else:
                            if PAIR_D:
                                with txl.serial(0, HBC // 2) as _hp:
                                    mma_pair(b_act[par])
                            else:
                                with txl.serial(0, HBC) as _h:
                                    mma_tile(b_act[par])
                        txl.assign(par, par ^ 1)
                        _rng_end(tk_m)
                        release_task(ts, True)
                        txl.assign(e_cur, e_nxt)
                    release_task(ts, True)
                txl.ptx.barrier.cluster.wait()




        with quant_role:
            tq = tid - QUANT_WARP0 * 32
            ts = txl.PipelineState(TASK_RING, phase=0)
            gst = txl.PipelineState(2, phase=0)
            e = _i32(txl.int32(0))
            ntok = _i32(txl.int32(0))
            running = _i32(txl.int32(1))

            def gather_rows(gbuf, nrows, tok_of):
                """B rows n < nrows <- quantized global rows tok_of(n) (16-byte chunks, loads issued first)."""
                cch = tq % 8
                kbg = tq // 8

                with txl.If(tq < NT), txl.Then():
                    with txl.If(txl.int32(0) + tq < nrows), txl.Then():
                        tokf = tok_of(tq)
                        fl = txl.local_scalar(txl.u32, init=txl.uint32(0))
                        txl.ptx.ld.acquire.gpu.global_.b32(fl, xflag_g.ptr_to([tokf]))
                        with txl.While(fl != epoch):
                            txl.ptx.ld.acquire.gpu.global_.b32(fl, xflag_g.ptr_to([tokf]))
                txl.ptx.bar.sync(txl.uint32(BAR_QUANT), txl.uint32(NQUANT))
                wv = txl.alloc_local((4 * NT,), txl.u32)
                xv2 = [txl.local_scalar(txl.f32) for _ in range(2)]
                for n in range(NT):
                    if isinstance(nrows, int) and n >= nrows:
                        continue
                    cond = txl.int32(n) < nrows
                    with txl.If(cond), txl.Then():
                        tok = tok_of(n)
                        txl.ptx.ld.global_.nc.v4.b32(wv[4 * n], wv[4 * n + 1], wv[4 * n + 2], wv[4 * n + 3],
                                                     xq_g.ptr_to([tok * (HID // 4) + kbg * 32 + cch * 4]))

                for half in range(2):
                    n = tq // KB + 8 * half
                    kb2 = tq % KB
                    txl.assign(xv2[half], txl.float32(0.0))
                    if isinstance(nrows, int) and 8 * half >= nrows:
                        continue
                    with txl.If(n < nrows), txl.Then():
                        tok = tok_of(n)
                        txl.ptx.ld.global_.nc.f32(xv2[half], xs_g.ptr_to([tok * KB + kb2]))
                for n in range(NT):
                    if isinstance(nrows, int) and n >= nrows:
                        continue
                    with txl.If(txl.int32(n) < nrows), txl.Then():
                        txl.ptx.st.shared.v4.b32(b_gu[gbuf * KB + kbg].ptr_to(n, cch * 16), wv[4 * n], wv[4 * n + 1], wv[4 * n + 2], wv[4 * n + 3])
                for half in range(2):
                    n = tq // KB + 8 * half
                    kb2 = tq % KB
                    txl.ptx.st.shared.f32(s_xs.ptr_to([gbuf, n * KB + kb2]), xv2[half])

            def scale_products(gbuf, xbuf):
                """s_prod[gbuf][tile*KB + kb][n] = w1s[e][tile][kb] * xs[n][kb] for every row n."""
                for i in range((2 * KB * NT) // NQUANT):
                    idx = tq + NQUANT * i
                    tile = idx // (KB * NT)
                    rem = idx % (KB * NT)
                    kb2 = rem // NT
                    n2 = rem % NT
                    wv = txl.local_scalar(txl.f32)
                    txl.ptx.ld.shared.f32(wv, s_task.ptr_to([ts.stage, W1_OFF + tile * KB + kb2]))
                    xv = txl.local_scalar(txl.f32)
                    txl.ptx.ld.shared.f32(xv, s_xs.ptr_to([xbuf, n2 * KB + kb2]))
                    txl.ptx.st.shared.f32(s_prod.ptr_to([gbuf, tile * KB + kb2, n2]), wv * xv)

            if GLOBAL_Q and not FIRST_TILE:
                tk_q = _rng("q-quant")
                for r_ in range(ROWS_PER_CTA):
                    row = cta + G * r_
                    with txl.If(row < M), txl.Then():
                        quant_unit_T(8, txl.int32(0), tq // 8, txl.int32(0), row, tq % 8, to_global=True)
                    txl.ptx.bar.sync(txl.uint32(BAR_ROWS), txl.uint32(NQUANT + 32))
                _rng_end(tk_q)

            if SHARED_B:
                tk_q = _rng("q-quant")
                prologue_quant(tq)
                bready.wait(0, 0)
                _rng_end(tk_q)
                with txl.While(running == 1):
                    read_task(ts, e, ntok)
                    with txl.If(e < 0):
                        with txl.Then():
                            txl.assign(running, txl.int32(0))
                        with txl.Else():
                            tk_q = _rng("q-prod")
                            prod_empty.wait(gst.stage, gst.phase ^ 1)
                            scale_products(_i32(gst.stage), txl.int32(0))
                            txl.ptx.bar.sync(txl.uint32(BAR_QUANT), txl.uint32(NQUANT))
                            with txl.If(lane == 0), txl.Then():
                                bq_full.arrive(gst.stage)
                            gst.advance()
                            _rng_end(tk_q)
                    release_task(ts, False)
            else:
                def quantize_item():
                    gbuf = _i32(gst.stage)
                    tk_q = _rng("q-wait-bempty")
                    bq_empty.wait(gst.stage, gst.phase ^ 1)
                    prod_empty.wait(gst.stage, gst.phase ^ 1)
                    txl.ptx.fence.proxy.async_.shared__cta()
                    _rng_end(tk_q)
                    tk_q = _rng("q-gather")

                    def tok_of(n):
                        t_ = txl.local_scalar(txl.i32)
                        txl.ptx.ld.shared.s32(t_, s_task.ptr_to([ts.stage, TOK_OFF + n]))
                        return t_

                    gather_rows(gbuf, ntok, tok_of)
                    if FIRST_TILE:
                        task_full.wait(ts.stage, ts.phase)
                    txl.ptx.bar.sync(txl.uint32(BAR_QUANT), txl.uint32(NQUANT))
                    scale_products(gbuf, gbuf)
                    txl.ptx.fence.proxy.async_.shared__cta()
                    txl.ptx.bar.sync(txl.uint32(BAR_QUANT), txl.uint32(NQUANT))
                    with txl.If(lane == 0), txl.Then():
                        bq_full.arrive(gst.stage)
                    gst.advance()
                    _rng_end(tk_q)

                first_q = _i32(txl.int32(1))
                with txl.While(running == 1):
                    if FIRST_TILE:
                        # The token list (task_tok) is published before the scheduler's global scale
                        # loads; task_full (scales) is only required by scale_products below.
                        task_tok.wait(ts.stage, ts.phase)
                        txl.ptx.ld.shared.s32(e, s_task.ptr_to([ts.stage, 0]))
                        txl.ptx.ld.shared.s32(ntok, s_task.ptr_to([ts.stage, 2]))
                    else:
                        read_task(ts, e, ntok)
                    with txl.If(e < 0):
                        with txl.Then():
                            if FIRST_TILE:
                                task_full.wait(ts.stage, ts.phase)
                            txl.assign(running, txl.int32(0))
                        with txl.Else():
                            with txl.If(first_q == 1):
                                with txl.Then():
                                    tk_q = _rng("q-quant0")
                                    first_item_quant(ts.stage, ntok)
                                    if FIRST_TILE:
                                        task_full.wait(ts.stage, ts.phase)
                                    txl.ptx.bar.sync(txl.uint32(BAR_QUANT), txl.uint32(NQUANT))
                                    scale_products(_i32(gst.stage), _i32(gst.stage))
                                    txl.ptx.fence.proxy.async_.shared__cta()
                                    txl.ptx.bar.sync(txl.uint32(BAR_QUANT), txl.uint32(NQUANT))
                                    with txl.If(lane == 0), txl.Then():
                                        bq_full.arrive(gst.stage)
                                    gst.advance()
                                    _rng_end(tk_q)
                                with txl.Else():
                                    quantize_item()
                            txl.assign(first_q, txl.int32(0))
                    release_task(ts, False)
            txl.ptx.barrier.cluster.wait()


        with math_role:
            mw = warp - MATH_WARP0
            tm = mw * 32 + lane
            is_gate = mw < 2
            ch_l = tm % CH
            tst = txl.PipelineState(NTB, phase=0)
            ts = txl.PipelineState(TASK_RING, phase=0)
            gst = txl.PipelineState(2, phase=0)
            xst = txl.PipelineState(2, phase=0)
            e = _i32(txl.int32(0))
            ntok = _i32(txl.int32(0))
            zero_ok = _i32(txl.int32(0))
            if RANK_ACT:
                items = _i32(txl.int32(0))
            acc = txl.alloc_local((NT,), txl.f32)
            pv = txl.alloc_local((4 * NT,), txl.f32)
            acts = txl.alloc_local((NT,), txl.f32)
            toks = txl.alloc_local((NT,), txl.i32)
            hv = txl.alloc_local((NT,), txl.f32)

            def named_bar():
                txl.ptx.bar.sync(txl.uint32(BAR_MATH), txl.uint32(NMATH))

            def drain_quad():
                s0 = _i32(tst.stage)
                tfull.wait(s0 + 3, tst.phase)
                for _ in range(4):
                    tst.advance()
                txl.ptx.tcgen05.fence__after_thread_sync()
                taddr = tmem_base + txl.Cast("uint32", s0) * txl.uint32(NT)
                txl.ptx[TMEM_LD64](*[pv[i] for i in range(4 * NT)], taddr)
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                txl.ptx.tcgen05.fence__before_thread_sync()
                txl.cuda.warp_sync()
                with txl.If(lane == 0), txl.Then():
                    for q in range(4):
                        tempty.arrive(s0 + q)

            def promote(gbuf, prow, off):
                for q4 in range(NT // 4):
                    x0 = txl.local_scalar(txl.f32)
                    x1 = txl.local_scalar(txl.f32)
                    x2 = txl.local_scalar(txl.f32)
                    x3 = txl.local_scalar(txl.f32)
                    txl.ptx.ld.shared.v4.f32(x0, x1, x2, x3, s_prod.ptr_to([gbuf, prow, 4 * q4]))
                    if M in (16, 128):
                        pair01 = txl.local_scalar("uint64")
                        pair23 = txl.local_scalar("uint64")
                        txl.ptx.fma.rn.f32x2(
                            pair01,
                            txl.cuda.make_float2(pv[off + 4 * q4], pv[off + 4 * q4 + 1]),
                            txl.cuda.make_float2(x0, x1),
                            txl.cuda.make_float2(acc[4 * q4], acc[4 * q4 + 1]),
                        )
                        txl.ptx.fma.rn.f32x2(
                            pair23,
                            txl.cuda.make_float2(pv[off + 4 * q4 + 2], pv[off + 4 * q4 + 3]),
                            txl.cuda.make_float2(x2, x3),
                            txl.cuda.make_float2(acc[4 * q4 + 2], acc[4 * q4 + 3]),
                        )
                        txl.ptx.mov.b64(acc[4 * q4], acc[4 * q4 + 1], pair01)
                        txl.ptx.mov.b64(acc[4 * q4 + 2], acc[4 * q4 + 3], pair23)
                    else:
                        txl.assign(acc[4 * q4], acc[4 * q4] + pv[off + 4 * q4] * x0)
                        txl.assign(acc[4 * q4 + 1], acc[4 * q4 + 1] + pv[off + 4 * q4 + 1] * x1)
                        txl.assign(acc[4 * q4 + 2], acc[4 * q4 + 2] + pv[off + 4 * q4 + 2] * x2)
                        txl.assign(acc[4 * q4 + 3], acc[4 * q4 + 3] + pv[off + 4 * q4 + 3] * x3)

            def gu_drain():
                tk_w = _rng("m-wait-b")
                bq_full.wait(gst.stage, gst.phase)
                gbuf = _i32(gst.stage)
                _rng_end(tk_w)
                tk_w = _rng("m-gu")
                for t_ in range(NT):
                    txl.assign(acc[t_], txl.float32(0.0))
                tile_row = txl.Select(is_gate, txl.int32(0), txl.int32(KB))
                with txl.serial(0, KB // 4) as kk:
                    drain_quad()
                    for q in range(4):
                        promote(gbuf, tile_row + 4 * kk + q, q * NT)
                txl.cuda.warp_sync()
                with txl.If(lane == 0), txl.Then():
                    prod_empty.arrive(gst.stage)
                gst.advance()
                _rng_end(tk_w)

            def swiglu_exchange_rank(par, slot, acts_dst):
                """Exchange half-amax values, quantize owned channels once, and exchange fp8 slices.
                `slot` is the task slot of this item; the epilogue scales land in `acts_dst`."""
                tk_w = _rng("m-swiglu")


                with txl.If(items >= txl.int32(2)), txl.Then():
                    act_empty.wait(par, xst.phase ^ 1)


                    txl.ptx.tcgen05.fence__after_thread_sync()
                with txl.If(tm == 0), txl.Then():
                    afull.arrive(par, tx_count=AMX_BYTES)
                    actfull.arrive(par, tx_count=ACT_BYTES)

                with txl.If(txl.Not(is_gate)), txl.Then():
                    for q4 in range(NT // 4):
                        txl.ptx.st.shared.v4.f32(s_up_ptr(ch_l, 4 * q4), acc[4 * q4], acc[4 * q4 + 1], acc[4 * q4 + 2], acc[4 * q4 + 3])
                named_bar()
                with txl.If(is_gate), txl.Then():
                    for q4 in range(NT // 4):
                        u4 = [txl.local_scalar(txl.f32) for _ in range(4)]
                        txl.ptx.ld.shared.v4.f32(u4[0], u4[1], u4[2], u4[3], s_up_ptr(ch_l, 4 * q4))
                        for q in range(4):
                            t_ = 4 * q4 + q
                            ex = txl.local_scalar(txl.f32)
                            txl.ptx.ex2.approx.ftz.f32(ex, acc[t_] * txl.float32(-LOG2E))
                            sig = txl.local_scalar(txl.f32)
                            txl.ptx.rcp.approx.ftz.f32(sig, ex + txl.float32(1.0))
                            txl.assign(hv[t_], acc[t_] * sig * u4[q])
                    for t_ in range(NT):
                        am = txl.local_scalar(txl.f32)
                        txl.ptx.redux_sync.max.abs.f32(am, hv[t_], txl.uint32(0xFFFFFFFF))
                        with txl.If(lane == 0), txl.Then():
                            txl.ptx.st.shared.f32(s_amax.ptr_to([t_, mw]), am)
                named_bar()
                # The peer's idle warp releases slot `par` once the peer consumed our item k-2 push
                # and its down MMAs finished reading the activation slot (see idle_role).
                cluster_wait(xchg_free, par, xst.phase ^ 1)
                with txl.If(txl.And(mw == 0, lane < NT)), txl.Then():
                    am0 = txl.local_scalar(txl.f32)
                    am1 = txl.local_scalar(txl.f32)
                    txl.ptx.ld.shared.v2.f32(am0, am1, s_amax.ptr_to([lane, 0]))
                    am = _f32(txl.max(am0, am1))
                    txl.ptx.st.shared.f32(s_amx.ptr_to([par, rank, lane]), am)
                    txl.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.f32(
                        remote_u32(s_amx.ptr_to([par, rank, lane]), peer), am,
                        remote_u32(afull.ptr_to([par]), peer),
                    )
                cluster_wait(afull, par, xst.phase)
                named_bar()
                with txl.If(txl.And(mw == 0, lane < NT)), txl.Then():
                    am0 = txl.local_scalar(txl.f32)
                    am1 = txl.local_scalar(txl.f32)
                    txl.ptx.ld.shared.f32(am0, s_amx.ptr_to([par, rank, lane]))
                    txl.ptx.ld.shared.f32(am1, s_amx.ptr_to([par, peer, lane]))
                    am = _f32(txl.max(am0, am1))
                    sc = _f32(txl.max(am, txl.float32(1.0e-8)) * txl.float32(INV_FP8_MAX))
                    rw = txl.local_scalar(txl.f32)
                    txl.ptx.ld.shared.f32(rw, s_task.ptr_to([slot, RW_OFF + lane]))
                    txl.ptx.st.shared.v2.f32(s_scl.ptr_to([lane, 0]), sc, (sc * rw) * rsf)
                named_bar()




                txl.ptx.fence.proxy.async_.shared__cta()
                col = ch_l
                scs = txl.alloc_local((NT,), txl.f32)
                rcps = txl.alloc_local((NT,), txl.f32)
                qbs = txl.alloc_local((NT,), txl.u16)
                for t_ in range(NT):
                    txl.ptx.ld.shared.v2.f32(scs[t_], acts_dst[t_], s_scl.ptr_to([t_, 0]))
                with txl.If(is_gate), txl.Then():
                    for t_ in range(NT):
                        txl.ptx.rcp.approx.ftz.f32(rcps[t_], scs[t_])
                    for t_ in range(NT):
                        rerr = txl.local_scalar(txl.f32)
                        txl.ptx[FMA_F32](rerr, txl.float32(0.0) - scs[t_], rcps[t_], txl.float32(1.0))
                        txl.ptx[FMA_F32](rcps[t_], rerr, rcps[t_], rcps[t_])
                    for t_ in range(NT):
                        txl.ptx.cvt.rn.satfinite.e4m3x2.f32(qbs[t_], txl.float32(0.0), hv[t_] * rcps[t_])
                    for t_ in range(NT):
                        w0 = _u32(txl.cast(qbs[t_], "uint32"))
                        w1 = txl.local_scalar(txl.u32)
                        txl.ptx.shfl_sync.down.b32(w1, w0, txl.uint32(1), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF))
                        txl.assign(w0, txl.bitwise_or(w0, txl.shift_left(w1, txl.uint32(8))))
                        txl.ptx.shfl_sync.down.b32(w1, w0, txl.uint32(2), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF))
                        txl.assign(w0, txl.bitwise_or(w0, txl.shift_left(w1, txl.uint32(16))))
                        with txl.If(lane % 4 == 0), txl.Then():
                            txl.ptx.st.shared.b32(b_act[par * CS + rank].ptr_to(t_, col), w0)
                txl.ptx.fence.proxy.async_.shared__cta()
                named_bar()
                with txl.If(tm == 0), txl.Then():
                    r_bar = remote_u32(actfull.ptr_to([par]), peer)
                    src = b_act[par * CS + rank].ptr_to(0, 0)
                    txl.ptx[BULK_S2C](remote_u32(src, peer), src, txl.uint32(ACT_BYTES), r_bar)
                cluster_wait(actfull, par, xst.phase)
                named_bar()
                with txl.If(tm == 0), txl.Then():
                    aq_full.arrive(par)
                _rng_end(tk_w)

            def swiglu_exchange_legacy(par, slot, acts_dst):
                """Legacy fp32 slice exchange used on shapes where paired timing favors it."""
                tk_w = _rng("m-swiglu")
                with txl.If(tm == 0), txl.Then():
                    hvfull.arrive(par, tx_count=HV_BYTES)


                with txl.If(txl.Not(is_gate)), txl.Then():
                    for q4 in range(NT // 4):
                        txl.ptx.st.shared.v4.f32(
                            s_up_ptr(ch_l, 4 * q4), acc[4 * q4], acc[4 * q4 + 1],
                            acc[4 * q4 + 2], acc[4 * q4 + 3],
                        )
                named_bar()
                with txl.If(is_gate), txl.Then():
                    for q4 in range(NT // 4):
                        u4 = [txl.local_scalar(txl.f32) for _ in range(4)]
                        txl.ptx.ld.shared.v4.f32(
                            u4[0], u4[1], u4[2], u4[3], s_up_ptr(ch_l, 4 * q4)
                        )
                        for q in range(4):
                            t_ = 4 * q4 + q
                            ex = txl.local_scalar(txl.f32)
                            txl.ptx.ex2.approx.ftz.f32(ex, acc[t_] * txl.float32(-LOG2E))
                            sig = txl.local_scalar(txl.f32)
                            txl.ptx.rcp.approx.ftz.f32(sig, ex + txl.float32(1.0))
                            txl.assign(hv[t_], acc[t_] * sig * u4[q])
                    # The peer's idle warp releases s_hvx slot `par` once the peer consumed our
                    # item k-2 push (see idle_role).
                    cluster_wait(hvfree, par, xst.phase ^ 1)
                    r_dst = remote_u32(s_hvx.ptr_to([par, ch_l, 0]), peer)
                    r_bar = remote_u32(hvfull.ptr_to([par]), peer)
                    for q4 in range(NT // 4):
                        txl.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.f32(
                            r_dst + txl.uint32(16 * q4), hv[4 * q4], hv[4 * q4 + 1],
                            hv[4 * q4 + 2], hv[4 * q4 + 3], r_bar,
                        )
                with txl.If(txl.Not(is_gate)), txl.Then():
                    cluster_wait(hvfull, par, xst.phase)
                    for q4 in range(NT // 4):
                        txl.ptx.ld.shared.v4.f32(
                            hv[4 * q4], hv[4 * q4 + 1], hv[4 * q4 + 2], hv[4 * q4 + 3],
                            s_hvx.ptr_to([par, ch_l, 4 * q4]),
                        )
                for t_ in range(NT):
                    am = txl.local_scalar(txl.f32)
                    txl.ptx.redux_sync.max.abs.f32(am, hv[t_], txl.uint32(0xFFFFFFFF))
                    with txl.If(lane == 0), txl.Then():
                        txl.ptx.st.shared.f32(s_amax.ptr_to([t_, mw]), am)
                named_bar()
                with txl.If(txl.And(mw == 0, lane < NT)), txl.Then():
                    am0 = txl.local_scalar(txl.f32)
                    am1 = txl.local_scalar(txl.f32)
                    am2 = txl.local_scalar(txl.f32)
                    am3 = txl.local_scalar(txl.f32)
                    txl.ptx.ld.shared.v4.f32(am0, am1, am2, am3, s_amax.ptr_to([lane, 0]))
                    am = _f32(txl.max(txl.max(am0, am1), txl.max(am2, am3)))
                    sc = _f32(txl.max(am, txl.float32(1.0e-8)) * txl.float32(INV_FP8_MAX))
                    rw = txl.local_scalar(txl.f32)
                    txl.ptx.ld.shared.f32(rw, s_task.ptr_to([slot, RW_OFF + lane]))
                    txl.ptx.st.shared.v2.f32(s_scl.ptr_to([lane, 0]), sc, (sc * rw) * rsf)
                named_bar()


                txl.ptx.fence.proxy.async_.shared__cta()
                col = txl.Select(is_gate, rank * CH, peer * CH) + ch_l
                scs = txl.alloc_local((NT,), txl.f32)
                rcps = txl.alloc_local((NT,), txl.f32)
                qbs = txl.alloc_local((NT,), txl.u16)
                for t_ in range(NT):
                    txl.ptx.ld.shared.v2.f32(scs[t_], acts_dst[t_], s_scl.ptr_to([t_, 0]))
                for t_ in range(NT):
                    txl.ptx.rcp.approx.ftz.f32(rcps[t_], scs[t_])
                for t_ in range(NT):
                    rerr = txl.local_scalar(txl.f32)
                    txl.ptx[FMA_F32](rerr, txl.float32(0.0) - scs[t_], rcps[t_], txl.float32(1.0))
                    txl.ptx[FMA_F32](rcps[t_], rerr, rcps[t_], rcps[t_])
                for t_ in range(NT):
                    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(
                        qbs[t_], txl.float32(0.0), hv[t_] * rcps[t_]
                    )
                for t_ in range(NT):
                    txl.ptx.st.shared.u8(
                        b_act[par].ptr_to(t_, col), txl.cast(qbs[t_], "uint8")
                    )
                txl.ptx.fence.proxy.async_.shared__cta()
                named_bar()
                with txl.If(tm == 0), txl.Then():
                    aq_full.arrive(par)
                _rng_end(tk_w)

            def d_epilogue():
                with txl.If(zero_ok == 0), txl.Then():
                    with txl.If(tm == 0), txl.Then():
                        zd = txl.local_scalar(txl.u32, init=txl.uint32(0))
                        txl.ptx.ld.acquire.gpu.global_.b32(zd, sync_ctr.ptr_to([ZERO_DONE]))
                        with txl.While(zd < epoch * txl.uint32(G)):
                            txl.ptx.ld.acquire.gpu.global_.b32(zd, sync_ctr.ptr_to([ZERO_DONE]))
                    named_bar()
                    txl.assign(zero_ok, txl.int32(1))
                mask_w = _u32(txl.uint32(0))
                if SHARED_B:
                    txl.ptx.ld.shared.u32(mask_w, s_task.ptr_to([ts.stage, 3]))

                def chunk_dest(chunk, h):
                    """16-byte chunk `chunk` of the staged tile: row r (token), columns [8*c8, 8*c8+8)."""
                    r = chunk // 16
                    c8 = chunk % 16
                    if SHARED_B:
                        tok_r = txl.int32(0) + r
                        valid = txl.bitwise_and(txl.shift_right(mask_w, txl.cast(r, "uint32")), txl.uint32(1)) == txl.uint32(1)
                    else:
                        tok_r = txl.local_scalar(txl.i32)
                        txl.ptx.ld.shared.s32(tok_r, s_task.ptr_to([ts.stage, TOK_OFF + r]))
                        valid = r < ntok
                    return r, c8, (tok_r * HID + h * BM) // 2 + 4 * c8, valid

                def quad_loop(nblk):
                    with txl.serial(0, HBC // 4) as hh:
                        drain_quad()
                        w2s4 = [txl.local_scalar(txl.f32) for _ in range(4)]
                        for q in range(4):
                            txl.ptx.ld.shared.f32(w2s4[q], s_task.ptr_to([ts.stage, W2_OFF + rank * HBC + 4 * hh + q]))
                        for q in range(4):
                            h = rank * HBC + 4 * hh + q
                            sb = q % 2
                            rd = txl.alloc_local((nblk,), txl.u16)
                            for t_ in range(nblk):
                                txl.ptx.cvt.rn.bf16.f32(rd[t_], pv[q * NT + t_] * (w2s4[q] * acts[t_]))
                            for t_ in range(nblk):
                                txl.ptx.st.shared.b16(s_stg.ptr_to([sb, t_, tm]), rd[t_])
                            named_bar()
                            nchunks = nblk * 16
                            for ci in range((nchunks + NMATH - 1) // NMATH):
                                chunk = tm + NMATH * ci
                                cond = (chunk < nchunks) if (nchunks - NMATH * ci) < NMATH else (txl.int32(1) == 1)
                                with txl.If(cond), txl.Then():
                                    r, c8, wbase, valid = chunk_dest(chunk, h)
                                    v4 = [txl.local_scalar(txl.u32) for _ in range(4)]
                                    txl.ptx.ld.shared.v4.b32(v4[0], v4[1], v4[2], v4[3], s_stg.ptr_to([sb, r, 8 * c8]))
                                    ok = txl.local_scalar("bool", init=valid)
                                    txl.ptx.red.relaxed.gpu.global_.add.noftz.v4.bf16x2(
                                        out.ptr_to([wbase]), v4[0], v4[1], v4[2], v4[3], pred=ok,
                                    )

                if SHARED_B:
                    quad_loop(M)
                else:
                    with txl.If(ntok <= 4):
                        with txl.Then():
                            quad_loop(4)
                        with txl.Else():
                            quad_loop(NT)

            def peek_next(dst):
                nst = _i32(ts.stage + 1)
                nph = _i32(ts.phase)
                with txl.If(nst == TASK_RING), txl.Then():
                    txl.assign(nst, txl.int32(0))
                    txl.assign(nph, nph ^ 1)
                task_hdr.wait(nst, nph)
                txl.ptx.ld.shared.s32(dst, s_task.ptr_to([nst, 0]))

            if SHARED_B:
                tk_w = _rng("m-quant")
                prologue_quant(128 + tm)
                _rng_end(tk_w)
            acts_n = txl.alloc_local((NT,), txl.f32)
            for t_ in range(NT):
                txl.assign(acts_n[t_], txl.float32(0.0))
            e_nxt = _i32(txl.int32(-1))
            ntok_n = _i32(txl.int32(0))

            def swiglu(slot, acts_dst):
                """SwiGLU + activation exchange of the item whose gate/up sums are in `acc`."""
                par = _i32(xst.stage)
                if RANK_ACT:
                    swiglu_exchange_rank(par, slot, acts_dst)
                else:
                    swiglu_exchange_legacy(par, slot, acts_dst)
                xst.advance()
                if RANK_ACT:
                    txl.assign(items, items + txl.int32(1))

            txl.ptx.barrier.cluster.wait()
            read_task(ts, e, ntok)
            with txl.If(e >= 0), txl.Then():
                gu_drain()
                swiglu(ts.stage, acts)
            with txl.While(e >= 0):

                nst = _i32(ts.stage + 1)
                nph = _i32(ts.phase)
                with txl.If(nst == TASK_RING), txl.Then():
                    txl.assign(nst, txl.int32(0))
                    txl.assign(nph, nph ^ 1)
                task_full.wait(nst, nph)
                txl.ptx.ld.shared.s32(e_nxt, s_task.ptr_to([nst, 0]))
                txl.ptx.ld.shared.s32(ntok_n, s_task.ptr_to([nst, 2]))
                with txl.If(e_nxt >= 0), txl.Then():
                    gu_drain()
                    swiglu(nst, acts_n)
                tk_w = _rng("m-d")
                d_epilogue()
                _rng_end(tk_w)
                release_task(ts, False)
                txl.assign(e, e_nxt)
                txl.assign(ntok, ntok_n)
                for t_ in range(NT):
                    txl.assign(acts[t_], acts_n[t_])
            release_task(ts, False)


        txl.cuda.cta_sync()
        with txl.If(warp == 1), txl.Then():
            txl.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](tmem_base, txl.uint32(TMEM_COLS))


        with txl.If(dyn), txl.Then():
            txl.ptx.barrier.cluster.arrive()
            txl.ptx.barrier.cluster.wait()

    return alphamoe_cluster



def build_wide_kernel(G, M, TOPK, E, HID, INTER):
    """M=1: one 8-CTA cluster per route (K-split gate/up, DSMEM all-reduce, H-split down); loads first (v2)."""

    CS8 = 8
    NWARPS = 6
    MATH_WARP0 = 2
    NMATH = 4 * 32
    NTHREADS = NWARPS * 32
    KBC = 2
    HBC = 2
    STAGES = 2 * KBC + HBC
    TILE_A = BM * BK
    TMEM_COLS = 128
    TMEM_LD2 = "tcgen05.ld.sync.aligned.32x32b.x2.b32"
    RED_ROW_BYTES = 2 * 4
    RED_BYTES = BM * RED_ROW_BYTES
    assert M == 1
    KB = HID // BK
    assert INTER == BK and KB == CS8 * KBC and HID // BM == CS8 * HBC
    assert G == TOPK * CS8
    OUT_WORDS = M * HID // 2
    OUT_V4 = OUT_WORDS // 4
    NZ = (OUT_V4 + G - 1) // G
    IDESC = encode_instr_descriptor_dense_uint32(
        M=BM, N=NT, K=32, d_dtype="float32", a_dtype="float8_e4m3fn",
        b_dtype="float8_e4m3fn", trans_a=False, trans_b=False, cta_group=1,
    )

    @txl.kernel(warps=NWARPS, arch="sm_100a", min_blocks_per_sm=1, grid=G)
    def alphamoe_wide_m1(
        topk_ids: txl.gptr[txl.i32],
        topk_w: txl.gptr[txl.f32],
        hidden: txl.gptr[txl.i32],
        w1s: txl.gptr[txl.f32],
        w2s: txl.gptr[txl.f32],
        out: txl.gptr[txl.i32],
        sync_ctr: txl.gptr[txl.u32],
        tm_w1: txl.TensorMap,
        tm_w2: txl.TensorMap,
        rsf: txl.f32,
        epoch: txl.u32,
    ):
        cta = txl.cta_id()
        rank = txl.cta_id_in_cluster([CS8])
        warp = txl.warp_id()
        lane = txl.lane_id()
        tid = txl.thread_id()
        cl = cta // CS8




        e_pre = _i32(txl.int32(0))
        rw_pre = _f32(txl.float32(0.0))
        with txl.If(txl.And(warp == 0, lane == 0)), txl.Then():
            txl.ptx.ld.global_.nc.s32(e_pre, topk_ids.ptr_to([cl]))
            txl.ptx.ld.global_.nc.f32(rw_pre, topk_w.ptr_to([cl]))
        QT = 16
        QNW = 64 // QT
        wpre = txl.alloc_local((QNW,), txl.u32)
        for j in range(QNW):
            txl.assign(wpre[j], txl.uint32(0))
        with txl.If(warp == MATH_WARP0 + 2), txl.Then():
            q_kb = rank * KBC + lane // QT
            txl.ptx.ld.global_.nc.v4.b32(wpre[0], wpre[1], wpre[2], wpre[3],
                                         hidden.ptr_to([q_kb * (BK // 2) + (lane % QT) * QNW]))

        smem = txl.smem_pool()
        a_tile = smem.alloc((STAGES, BM, BK), txl.f8e4m3, align=1024, swizzle=txl.SW128B)
        b_gu = smem.alloc((KBC, NT, BK), txl.f8e4m3, align=1024, swizzle=txl.SW128B)
        b_act = smem.alloc((NT, BK), txl.f8e4m3, align=1024, swizzle=txl.SW128B)
        s_red = smem.alloc((CS8, BM, 2), txl.f32, align=16)
        s_xs = smem.alloc((4,), txl.f32, align=16)
        s_amax = smem.alloc((4,), txl.f32, align=16)
        s_misc = smem.alloc((4,), txl.i32, align=16)
        s_miscf = smem.alloc((4,), txl.f32, align=16)
        s_stg = smem.alloc((HBC, BM), txl.u16, align=16)
        tmem_slot = smem.alloc((1,), txl.u32, align=4)

        full_bar = txl.TMABar(smem, STAGES)
        tfull = txl.TCGen05Bar(smem, STAGES)
        bready = txl.MBarrier(smem, 1)
        e_ready = txl.MBarrier(smem, 1)
        red_full = txl.TMABar(smem, 1)
        act_full = txl.MBarrier(smem, 1)
        full_bar.init(1)
        tfull.init(1)
        bready.init(4)
        e_ready.init(1)
        red_full.init(1)
        act_full.init(4)
        txl.ptx.fence.mbarrier_init.release.cluster()
        with txl.If(warp == 1), txl.Then():
            txl.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                txl.address_of(tmem_slot[0]), txl.uint32(TMEM_COLS)
            )
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
        with txl.If(txl.And(warp == 0, lane == 0)), txl.Then():
            txl.ptx.prefetch.tensormap(txl.address_of(tm_w1))
            txl.ptx.prefetch.tensormap(txl.address_of(tm_w2))



        with txl.If(txl.And(warp == 1, lane < NZ)), txl.Then():
            j = cta + G * lane
            with txl.If(j < OUT_V4), txl.Then():
                txl.ptx.st.global_.v4.b32(out.ptr_to([j * 4]), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0))
        txl.cuda.cta_sync()
        with txl.If(tid == 0), txl.Then():

            red_full.arrive(0, tx_count=(CS8 - 1) * RED_BYTES)
        with txl.If(txl.And(warp == 1, lane == 0)), txl.Then():

            txl.ptx.red.release.gpu.global_.add.u32(sync_ctr.ptr_to([ZERO_DONE]), txl.uint32(1))

        txl.ptx.barrier.cluster.arrive.relaxed()

        def remote_u32(ptr, peer_):
            r = txl.local_scalar(txl.u32)
            txl.ptx.mapa.shared__cluster.u32(r, txl.cuda.cvta_generic_to_shared(ptr), txl.cast(peer_, "uint32"))
            return r

        def cluster_wait(bar, stage, phase):
            ok = txl.local_scalar(txl.u32, init=txl.uint32(0))
            with txl.While(ok == txl.uint32(0)):
                txl.ptx.mbarrier.try_wait.parity.acquire.cluster.shared.b64(
                    ok, bar.ptr_to([stage]), txl.cast(phase, "uint32")
                )

        def rcp_refined(scale):
            rcp0 = txl.local_scalar(txl.f32)
            txl.ptx.rcp.approx.ftz.f32(rcp0, scale)
            rerr = txl.local_scalar(txl.f32)
            txl.ptx[FMA_F32](rerr, txl.float32(0.0) - scale, rcp0, txl.float32(1.0))
            rcp1 = txl.local_scalar(txl.f32)
            txl.ptx[FMA_F32](rcp1, rerr, rcp0, rcp0)
            return rcp1

        def named_bar():
            txl.ptx.bar.sync(txl.uint32(BAR_MATH), txl.uint32(NMATH))


        with txl.If(warp == 0), txl.Then():
            with txl.If(lane == 0), txl.Then():
                e = e_pre
                rw = rw_pre
                tk_p = _rng("p-gu")
                for u in range(KBC):
                    s = 2 * u
                    kb = rank * KBC + u
                    full_bar.arrive(s, tx_count=2 * TILE_A)
                    txl.ptx[TMA_G2S_3D](
                        a_tile[s].ptr_to(0, 0), txl.address_of(tm_w1),
                        kb * BK, txl.int32(0), e * 2,
                        full_bar.ptr_to([s]), txl.uint64(EVICT_FIRST),
                    )
                _rng_end(tk_p)
                tk_p = _rng("p-d")
                s = 2 * KBC
                h = rank * HBC
                full_bar.arrive(s, tx_count=2 * TILE_A)
                txl.ptx[TMA_G2S_3D](
                    a_tile[s].ptr_to(0, 0), txl.address_of(tm_w2),
                    txl.int32(0), txl.int32(0), e * (HID // BM) + h,
                    full_bar.ptr_to([s]), txl.uint64(EVICT_FIRST),
                )
                _rng_end(tk_p)
                txl.ptx.st.shared.s32(s_misc.ptr_to([0]), e)
                txl.ptx.st.shared.f32(s_miscf.ptr_to([0]), rw)
                e_ready.arrive(0)
                txl.cuda.iket.mark("e-ready")
            txl.ptx.barrier.cluster.wait()


        with txl.If(warp == 1), txl.Then():
            with txl.If(lane == 0), txl.Then():
                tmem_base = txl.local_scalar(txl.u32)
                txl.ptx.ld.shared.u32(tmem_base, tmem_slot.ptr_to([0]))

                def mma_stage(s, bview, wait_full=True):
                    if wait_full:
                        full_bar.wait(s, 0)
                    txl.ptx.tcgen05.fence__after_thread_sync()
                    a_desc, a_off = a_tile[s].encode(major="k", mma_k=32)
                    b_desc, b_off = bview.encode(major="k", mma_k=32)
                    d_addr = tmem_base + txl.uint32(s * NT)
                    for ki in range(BK // 32):
                        txl.ptx[MMA](
                            d_addr, a_desc + a_off(ki), b_desc + b_off(ki), txl.uint32(IDESC),
                            txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0),
                            txl.ptx.pred(txl.uint32(1 if ki > 0 else 0)),
                        )
                    txl.ptx.tcgen05.fence__before_thread_sync()
                    tfull.arrive(s)

                bready.wait(0, 0)
                tk_m = _rng("mma-gu")
                for u in range(KBC):
                    s = 2 * u
                    full_bar.wait(s, 0)
                    # The one TMA completion covers both contiguous stages.
                    full_bar.arrive(s + 1)
                    mma_stage(s, b_gu[u], wait_full=False)
                    mma_stage(s + 1, b_gu[u], wait_full=False)
                _rng_end(tk_m)
                tk_m = _rng("mma-wait-act")
                act_full.wait(0, 0)
                _rng_end(tk_m)
                tk_m = _rng("mma-d")
                s = 2 * KBC
                full_bar.wait(s, 0)
                full_bar.arrive(s + 1)
                mma_stage(s, b_act, wait_full=False)
                mma_stage(s + 1, b_act, wait_full=False)
                _rng_end(tk_m)
            txl.ptx.barrier.cluster.wait()


        with txl.If(warp >= MATH_WARP0), txl.Then():
            tm = (warp % 4) * 32 + lane
            tmem_base = txl.local_scalar(txl.u32)
            txl.ptx.ld.shared.u32(tmem_base, tmem_slot.ptr_to([0]))



            tk_w = _rng("m-quant")
            T = QT
            NW = QNW
            assert KBC * T == 32
            with txl.If(tm < KBC * T), txl.Then():
                u = tm // T
                sub = tm % T
                w = wpre
                amax = _f32(txl.float32(0.0))
                for j in range(NW):
                    txl.assign(amax, txl.max(amax, txl.max(txl.fabs(_bf16_lo(w[j])), txl.fabs(_bf16_hi(w[j])))))
                m_ = T // 2
                while m_ >= 1:
                    o = txl.local_scalar(txl.f32)
                    txl.ptx.shfl_sync.bfly.b32(o, amax, txl.uint32(m_), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF))
                    txl.assign(amax, txl.max(amax, o))
                    m_ //= 2
                scale = _f32(txl.max(amax, txl.float32(1.0e-8)) * txl.float32(INV_FP8_MAX))
                rcp1 = rcp_refined(scale)
                qw = txl.alloc_local((2,), txl.u32)
                for jx in range(2):
                    a = 2 * jx
                    h_lo = txl.local_scalar(txl.u16)
                    h_hi = txl.local_scalar(txl.u16)
                    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(h_lo, _bf16_hi(w[a]) * rcp1, _bf16_lo(w[a]) * rcp1)
                    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(h_hi, _bf16_hi(w[a + 1]) * rcp1, _bf16_lo(w[a + 1]) * rcp1)
                    txl.assign(qw[jx], txl.bitwise_or(txl.cast(h_lo, "uint32"),
                                                     txl.shift_left(txl.cast(h_hi, "uint32"), txl.uint32(16))))
                txl.ptx.st.shared.v2.b32(b_gu[u].ptr_to(0, sub * (BK // T)), qw[0], qw[1])
                with txl.If(sub == 0), txl.Then():
                    txl.ptx.st.shared.f32(s_xs.ptr_to([u]), scale)

            ZCH = (KBC + 1) * (NT - 1) * 8
            for i in range((ZCH + NMATH - 1) // NMATH):
                j = tm + NMATH * i
                tile = j // ((NT - 1) * 8)
                rem = j % ((NT - 1) * 8)
                n_ = 1 + rem // 8
                c_ = rem % 8
                cond = txl.int32(1) == 1 if (i + 1) * NMATH <= ZCH else j < ZCH
                with txl.If(cond), txl.Then():
                    base = txl.Select(tile == 0, txl.cuda.cvta_generic_to_shared(b_gu[0].ptr_to(0, 0)),
                                      txl.Select(tile == 1, txl.cuda.cvta_generic_to_shared(b_gu[1].ptr_to(0, 0)),
                                                 txl.cuda.cvta_generic_to_shared(b_act.ptr_to(0, 0))))
                    txl.ptx.st.shared.v4.b32(base + txl.cast(n_ * BK + 16 * c_, "uint32"),
                                             txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0))
            txl.ptx.fence.proxy.async_.shared__cta()
            txl.cuda.warp_sync()
            with txl.If(lane == 0), txl.Then():
                bready.arrive(0)
            bready.wait(0, 0)
            _rng_end(tk_w)


            e_ready.wait(0, 0)
            e = txl.local_scalar(txl.i32)
            rw = txl.local_scalar(txl.f32)
            txl.ptx.ld.shared.s32(e, s_misc.ptr_to([0]))
            txl.ptx.ld.shared.f32(rw, s_miscf.ptr_to([0]))
            w1v = txl.alloc_local((2 * KBC,), txl.f32)
            for t_ in range(2):
                for u in range(KBC):
                    txl.ptx.ld.global_.nc.f32(w1v[t_ * KBC + u], w1s.ptr_to([e * (2 * KB) + t_ * KB + rank * KBC + u]))
            w2v = txl.alloc_local((HBC,), txl.f32)
            for hh in range(HBC):
                txl.ptx.ld.global_.nc.f32(w2v[hh], w2s.ptr_to([e * KB + rank * HBC + hh]))
            xs = txl.alloc_local((KBC,), txl.f32)
            for u in range(KBC):
                txl.ptx.ld.shared.f32(xs[u], s_xs.ptr_to([u]))


            tk_w = _rng("m-gu")
            pv0 = txl.local_scalar(txl.f32)
            pv1 = txl.local_scalar(txl.f32)
            part = txl.alloc_local((2,), txl.f32)
            for t_ in range(2):
                txl.assign(part[t_], txl.float32(0.0))
            for s in range(2 * KBC):
                t_ = s % 2
                u = s // 2
                tfull.wait(s, 0)
                txl.ptx.tcgen05.fence__after_thread_sync()
                txl.ptx[TMEM_LD2](pv0, pv1, tmem_base + txl.uint32(s * NT))
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                txl.assign(part[t_], part[t_] + pv0 * (w1v[t_ * KBC + u] * xs[u]))
            txl.ptx.tcgen05.fence__before_thread_sync()
            _rng_end(tk_w)


            tk_w = _rng("m-reduce")
            txl.ptx.barrier.cluster.wait()
            txl.ptx.st.shared.v2.f32(s_red.ptr_to([rank, tm, 0]), part[0], part[1])
            g1 = txl.local_scalar(txl.f32)
            u1 = txl.local_scalar(txl.f32)
            txl.ptx.shfl_sync.down.b32(g1, part[0], txl.uint32(1), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF))
            txl.ptx.shfl_sync.down.b32(u1, part[1], txl.uint32(1), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF))
            with txl.If(lane % 2 == 0), txl.Then():

                for p in range(CS8):
                    with txl.If(p != rank), txl.Then():
                        r_dst = remote_u32(s_red.ptr_to([rank, tm, 0]), p)
                        r_bar = remote_u32(red_full.ptr_to([0]), p)
                        txl.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
                            r_dst,
                            txl.reinterpret("uint32", part[0]), txl.reinterpret("uint32", part[1]),
                            txl.reinterpret("uint32", g1), txl.reinterpret("uint32", u1),
                            r_bar,
                        )
            cluster_wait(red_full, 0, 0)
            gsum = _f32(txl.float32(0.0))
            usum = _f32(txl.float32(0.0))
            for r_ in range(CS8):
                gr = txl.local_scalar(txl.f32)
                ur = txl.local_scalar(txl.f32)
                txl.ptx.ld.shared.v2.f32(gr, ur, s_red.ptr_to([r_, tm, 0]))
                txl.assign(gsum, gsum + gr)
                txl.assign(usum, usum + ur)
            _rng_end(tk_w)


            tk_w = _rng("m-swiglu")
            ex = txl.local_scalar(txl.f32)
            txl.ptx.ex2.approx.ftz.f32(ex, gsum * txl.float32(-LOG2E))
            sig = txl.local_scalar(txl.f32)
            txl.ptx.rcp.approx.ftz.f32(sig, ex + txl.float32(1.0))
            hv = _f32(gsum * sig * usum)
            am = txl.local_scalar(txl.f32)
            txl.ptx.redux_sync.max.abs.f32(am, hv, txl.uint32(0xFFFFFFFF))
            with txl.If(lane == 0), txl.Then():
                txl.ptx.st.shared.f32(s_amax.ptr_to([warp % 4]), am)
            named_bar()
            am0 = txl.local_scalar(txl.f32)
            am1 = txl.local_scalar(txl.f32)
            am2 = txl.local_scalar(txl.f32)
            am3 = txl.local_scalar(txl.f32)
            txl.ptx.ld.shared.v4.f32(am0, am1, am2, am3, s_amax.ptr_to([0]))
            amx = _f32(txl.max(txl.max(am0, am1), txl.max(am2, am3)))
            sc = _f32(txl.max(amx, txl.float32(1.0e-8)) * txl.float32(INV_FP8_MAX))
            rcp_a = rcp_refined(sc)
            qb = txl.local_scalar(txl.u16)
            txl.ptx.cvt.rn.satfinite.e4m3x2.f32(qb, txl.float32(0.0), hv * rcp_a)
            txl.ptx.st.shared.u8(b_act.ptr_to(0, tm), txl.cast(qb, "uint8"))
            txl.ptx.fence.proxy.async_.shared__cta()
            txl.cuda.warp_sync()
            with txl.If(lane == 0), txl.Then():
                act_full.arrive(0)
            afac = _f32((sc * rw) * rsf)
            _rng_end(tk_w)


            tk_w = _rng("m-d")
            with txl.If(tm == 0), txl.Then():
                zd = txl.local_scalar(txl.u32, init=txl.uint32(0))
                txl.ptx.ld.acquire.gpu.global_.b32(zd, sync_ctr.ptr_to([ZERO_DONE]))
                with txl.While(zd < epoch * txl.uint32(G)):
                    txl.ptx.ld.acquire.gpu.global_.b32(zd, sync_ctr.ptr_to([ZERO_DONE]))
            named_bar()
            for hh in range(HBC):
                s = 2 * KBC + hh
                h = rank * HBC + hh
                tfull.wait(s, 0)
                txl.ptx.tcgen05.fence__after_thread_sync()
                txl.ptx[TMEM_LD2](pv0, pv1, tmem_base + txl.uint32(s * NT))
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                rd = txl.local_scalar(txl.u16)
                txl.ptx.cvt.rn.bf16.f32(rd, pv0 * (w2v[hh] * afac))
                txl.ptx.st.shared.b16(s_stg.ptr_to([hh, tm]), rd)
                named_bar()
                with txl.If(tm < BM // 8), txl.Then():
                    v4 = [txl.local_scalar(txl.u32) for _ in range(4)]
                    txl.ptx.ld.shared.v4.b32(v4[0], v4[1], v4[2], v4[3], s_stg.ptr_to([hh, 8 * tm]))
                    txl.ptx.red.relaxed.gpu.global_.add.noftz.v4.bf16x2(
                        out.ptr_to([h * (BM // 2) + 4 * tm]), v4[0], v4[1], v4[2], v4[3],
                    )
            txl.ptx.tcgen05.fence__before_thread_sync()
            _rng_end(tk_w)


        txl.cuda.cta_sync()
        with txl.If(warp == 1), txl.Then():
            tb = txl.local_scalar(txl.u32)
            txl.ptx.ld.shared.u32(tb, tmem_slot.ptr_to([0]))
            txl.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](tb, txl.uint32(TMEM_COLS))

    return alphamoe_wide_m1






class _TensorMap:
    def __init__(self):
        self._buf = ctypes.create_string_buffer(256)
        self.ptr = ctypes.c_void_p((ctypes.addressof(self._buf) + 127) & ~127)


def _encode_2d(tensor, dtype_name, inner, outer, stride_bytes, box_inner, box_outer, swizzle):
    enc = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    tm = _TensorMap()
    enc(
        tm.ptr, dtype_name, 2, ctypes.c_void_p(int(tensor.data_ptr())),
        int(inner), int(outer), int(stride_bytes), int(box_inner), int(box_outer),
        1, 1, 0, int(swizzle), 3, 0,
    )
    return tm


def _encode_3d(tensor, dtype_name, dims, strides_bytes, box, swizzle):
    tm = _TensorMap()
    enc = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    enc(
        tm.ptr, dtype_name, 3, ctypes.c_void_p(int(tensor.data_ptr())),
        int(dims[0]), int(dims[1]), int(dims[2]),
        int(strides_bytes[0]), int(strides_bytes[1]),
        int(box[0]), int(box[1]), int(box[2]),
        1, 1, 1, 0, int(swizzle), 3, 0,
    )
    return tm


def choose_splits(M):
    return (CS,)


_COMPILED = {}


def _executable(G, M, TOPK, E, HID, INTER, cs):
    key = (G, M, TOPK, E, HID, INTER, cs)
    if key not in _COMPILED:
        _COMPILED[key] = build_kernel(G, M, TOPK, E, HID, INTER, cs).compile()
    return _COMPILED[key]


SHAPES = (1, 8, 16, 32, 64, 128)
_SHAPE_GRID = {1: 20, 8: 80, 32: 128, 64: 124, 128: 134}


def _grid_for(M, topk, sms):
    """The shape-only persistent grid.  M=1 runs one 8-CTA cluster per route."""
    if int(M) == 1:
        return int(topk) * 8
    G = _SHAPE_GRID.get(int(M), int(sms))
    (cs,) = choose_splits(int(M))
    return G - G % cs


def _precompile(M, topk, E, HID, INTER, sms):
    """Populate the compile cache with the exact keys ``setup`` will look up."""
    G = _grid_for(M, topk, sms)
    if int(M) == 1:
        key = ("wide", G, 1, topk, int(E), int(HID), INTER)
        if key not in _COMPILED:
            _COMPILED[key] = build_wide_kernel(G, 1, topk, int(E), int(HID), INTER).compile()
        return _COMPILED[key]
    (cs,) = choose_splits(int(M))
    return _executable(G, int(M), topk, int(E), int(HID), INTER, cs)


def setup(data, M):
    hidden = data["hidden_states"]
    w1 = data["gemm1_weights"]
    w2 = data["gemm2_weights"]
    topk = int(data["top_k"])
    M_h, HID = hidden.shape
    E, NGU, HID_w = w1.shape
    INTER = int(w2.shape[2])
    assert int(M_h) == int(M) and HID_w == HID and NGU == 2 * INTER
    device = hidden.device
    G_full = int(torch.cuda.get_device_properties(device).multi_processor_count)
    G = _grid_for(int(M), topk, G_full)
    if int(M) == 1:
        key = ("wide", G, 1, topk, int(E), int(HID), INTER)
        if key not in _COMPILED:
            _COMPILED[key] = build_wide_kernel(G, 1, topk, int(E), int(HID), INTER).compile()
        return make_wide_runner(_COMPILED[key], data, M)
    (cs,) = choose_splits(int(M))
    executable = _executable(G, int(M), topk, int(E), int(HID), INTER, cs)
    return make_runner(executable, data, M, cs)


def make_runner(executable, data, M, cs):
    hidden = data["hidden_states"]
    topk_ids = data["topk_ids"]
    topk_w = data["topk_weights"]
    w1 = data["gemm1_weights"]
    w1s = data["gemm1_weights_scale"]
    w2 = data["gemm2_weights"]
    w2s = data["gemm2_weights_scale"]
    out = data["output"]
    topk = int(data["top_k"])
    rsf = float(data["routed_scaling_factor"])
    device = hidden.device
    M_h, HID = hidden.shape
    E, NGU, _ = w1.shape
    INTER = int(w2.shape[2])
    assert INTER == BK and HID % BK == 0
    assert hidden.dtype == torch.bfloat16 and w1.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn and out.dtype == torch.bfloat16
    assert topk_ids.dtype == torch.int32 and topk_w.dtype == torch.float32
    assert tuple(topk_ids.shape) == (M, topk) and tuple(topk_w.shape) == (M, topk)
    assert tuple(w1s.shape) == (E, NGU // BK, HID // BK)
    assert tuple(w2s.shape) == (E, HID // BK, INTER // BK)
    for t in (hidden, topk_ids, topk_w, w1, w1s, w2, w2s, out):
        assert t.is_contiguous()

    sync_ctr = torch.zeros(128, dtype=torch.uint32, device=device)
    xq_g = torch.zeros(int(M) * HID // 4, dtype=torch.uint32, device=device)
    xs_g = torch.zeros(int(M) * (HID // BK), dtype=torch.float32, device=device)
    xflag_g = torch.zeros(int(M), dtype=torch.uint32, device=device)


    tm_w1h = _encode_3d(
        w1, "float8_e4m3fn", (HID, INTER, 2 * E), (HID, INTER * HID),
        (BK, CH, 2), 3,
    )
    if int(M) in W2_TMA3D_MS:

        tm_w2 = _encode_3d(
            w2, "float8_e4m3fn", (INTER, BM, E * (HID // BM)), (INTER, BM * INTER),
            (BK, BM, 2), 3,
        )
    else:
        tm_w2 = _encode_2d(w2, "float8_e4m3fn", INTER, E * HID, INTER, BK, BM, 3)

    state = {"epoch": 0}
    fixed = (
        topk_ids.view(-1),
        topk_w.view(-1),
        hidden.view(torch.int32).view(-1),
        w1s.view(-1),
        w2s.view(-1),
        out.view(torch.int32).view(-1),
        sync_ctr,
        xq_g,
        xs_g,
        xflag_g,
        tm_w1h.ptr,
        tm_w2.ptr,
        rsf,
    )

    def run():

        state["epoch"] = (state["epoch"] % 0x00FFFFFF) + 1
        executable(*fixed, state["epoch"])

    run._keep_alive = (fixed, tm_w1h, tm_w2, xq_g, xs_g, xflag_g)
    run()
    torch.cuda.synchronize(device)
    return run


def make_wide_runner(executable, data, M):
    hidden = data["hidden_states"]
    topk_ids = data["topk_ids"]
    topk_w = data["topk_weights"]
    w1 = data["gemm1_weights"]
    w1s = data["gemm1_weights_scale"]
    w2 = data["gemm2_weights"]
    w2s = data["gemm2_weights_scale"]
    out = data["output"]
    topk = int(data["top_k"])
    rsf = float(data["routed_scaling_factor"])
    device = hidden.device
    M_h, HID = hidden.shape
    E, NGU, _ = w1.shape
    INTER = int(w2.shape[2])
    assert INTER == BK and HID % BK == 0
    assert hidden.dtype == torch.bfloat16 and w1.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn and out.dtype == torch.bfloat16
    assert topk_ids.dtype == torch.int32 and topk_w.dtype == torch.float32
    assert tuple(topk_ids.shape) == (M, topk) and tuple(topk_w.shape) == (M, topk)
    assert tuple(w1s.shape) == (E, NGU // BK, HID // BK)
    assert tuple(w2s.shape) == (E, HID // BK, INTER // BK)
    for t in (hidden, topk_ids, topk_w, w1, w1s, w2, w2s, out):
        assert t.is_contiguous()

    sync_ctr = torch.zeros(128, dtype=torch.uint32, device=device)
    tm_w1 = _encode_3d(
        w1, "float8_e4m3fn", (HID, INTER, 2 * E), (HID, INTER * HID),
        (BK, BM, 2), 3,
    )
    tm_w2 = _encode_3d(
        w2, "float8_e4m3fn", (INTER, BM, E * (HID // BM)), (INTER, BM * INTER),
        (BK, BM, 2), 3,
    )
    state = {"epoch": 0}
    fixed = (
        topk_ids.view(-1),
        topk_w.view(-1),
        hidden.view(torch.int32).view(-1),
        w1s.view(-1),
        w2s.view(-1),
        out.view(torch.int32).view(-1),
        sync_ctr,
        tm_w1.ptr,
        tm_w2.ptr,
        rsf,
    )

    def run():
        state["epoch"] = (state["epoch"] % 0x00FFFFFF) + 1
        executable(*fixed, state["epoch"])

    run._keep_alive = (fixed, tm_w1, tm_w2, sync_ctr)
    run()
    torch.cuda.synchronize(device)
    return run


# ---------------------------------------------------------------------------
# Registered benchmark surface
# ---------------------------------------------------------------------------

KERNEL_META = {
    "name": "agent_evolved_alphamoe_fp8_blockscale_qwen3next",
    "category": "agent_evolved",
    "runtime_cuda_archs": ["sm_100a"],
    "provenance": {
        "generator": "kda_flow",
        "run": "alphamoe-20260916-193749",
        "selected_version": "cluster-split",
    },
}

HIDDEN = 2048
INTERMEDIATE = 128
NUM_EXPERTS = 512
NUM_SHARED_EXPERTS = 0
TOPK = 10
BLOCK_SIZE = BK
MAX_CTAS = 148

# The official rows' input distribution (flashinfer-bench-evolve alphamoe task).
HIDDEN_STATES_STD = 0.25
WEIGHT_STD = 0.125
OFFICIAL_SEED = 42
OFFICIAL_ROW = "alphamoe-qwen3-next-tp4-m{m}-e512-top10-k2048-i128-fp8"


@dataclass(frozen=True, slots=True)
class AlphaMoEConfig:
    label: str = "m128_official"
    num_tokens: int = 128
    seed: int = OFFICIAL_SEED
    routed_scaling_factor: float = 1.0
    balancedness: float = 1.0

    def validate(self) -> None:
        if self.num_tokens not in SHAPES:
            raise ValueError(
                f"num_tokens must be one of the official rows {SHAPES}, got {self.num_tokens}"
            )
        if self.routed_scaling_factor <= 0:
            raise ValueError("routed_scaling_factor must be positive")
        if not 0.0 < self.balancedness <= 1.0:
            raise ValueError(f"balancedness must be in (0, 1], got {self.balancedness}")


def _official(num_tokens: int) -> dict[str, Any]:
    return {
        "label": f"m{num_tokens}_official",
        "num_tokens": num_tokens,
        "seed": OFFICIAL_SEED,
        "routed_scaling_factor": 1.0,
        "balancedness": 1.0,
    }


CONFIGS = [_official(m) for m in SHAPES] + [
    # A hot routed expert 0 and a non-unit scaling factor exercise the
    # multi-chunk expert path and the route-weight multiplication that the
    # evolution run's correctness failures came from.  M=128 covers the
    # chunked M >= 8 kernel, M=1 the per-route wide-cluster kernel.
    {
        "label": "m128_hot_expert",
        "num_tokens": 128,
        "seed": 7,
        "routed_scaling_factor": 2.5,
        "balancedness": 0.25,
    },
    {
        "label": "m1_hot_expert",
        "num_tokens": 1,
        "seed": 7,
        "routed_scaling_factor": 2.5,
        "balancedness": 0.25,
    },
]

BENCH_CONFIGS = [_official(m) for m in SHAPES]


def _cfg(**kwargs: Any) -> AlphaMoEConfig:
    names = {field.name for field in fields(AlphaMoEConfig)}
    cfg = AlphaMoEConfig(**{name: value for name, value in kwargs.items() if name in names})
    cfg.validate()
    return cfg


def _num_ctas(**kwargs: Any) -> int:
    """The SM budget.  Each row's persistent grid is derived from it by shape."""
    if "num_ctas" in kwargs:
        value = int(kwargs["num_ctas"])
    else:
        from tirx_kernels.runner import hardware_num_sms

        value = min(hardware_num_sms(), MAX_CTAS)
    if not 1 <= value <= MAX_CTAS:
        raise ValueError(f"num_ctas must be in [1, {MAX_CTAS}], got {value}")
    return value


def get_kernel(**kwargs: Any):
    """The tirx-lite PrimFunc for the configured row (wide cluster at M=1)."""
    cfg = _cfg(**kwargs)
    num_tokens = int(cfg.num_tokens)
    grid = _grid_for(num_tokens, TOPK, _num_ctas(**kwargs))
    if num_tokens == 1:
        return build_wide_kernel(grid, 1, TOPK, NUM_EXPERTS, HIDDEN, INTERMEDIATE).func
    (cs,) = choose_splits(num_tokens)
    return build_kernel(
        grid, num_tokens, TOPK, NUM_EXPERTS, HIDDEN, INTERMEDIATE, cs
    ).func


def _assert_supported_arch() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved Alpha-MoE")
    capability = torch.cuda.get_device_capability()
    runtime_arch = f"sm_{capability[0]}{capability[1]}a"
    if runtime_arch not in KERNEL_META["runtime_cuda_archs"]:
        raise SkipTest(
            "agent-evolved Alpha-MoE requires one of "
            f"{KERNEL_META['runtime_cuda_archs']}, got {runtime_arch}"
        )


# ---------------------------------------------------------------------------
# Inputs (the official rows' generator)
# ---------------------------------------------------------------------------


def _quantize_block_2d(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert 128x128 block FP8 E4M3 quantization of ``[E, rows, cols]``."""
    experts, rows, columns = values.shape
    groups = values.float().reshape(
        experts, rows // BLOCK_SIZE, BLOCK_SIZE, columns // BLOCK_SIZE, BLOCK_SIZE
    )
    scales = groups.abs().amax(dim=(2, 4)).clamp_min(1.0e-8) / FP8_MAX
    quantized = (groups / scales[:, :, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    return (
        quantized.to(torch.float8_e4m3fn).reshape(experts, rows, columns).contiguous(),
        scales.contiguous(),
    )


def _make_routing(
    generator: torch.Generator, device: torch.device, num_tokens: int, balancedness: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded top-k routing; ``balancedness`` < 1 makes routed expert 0 hot."""
    routed_top_k = TOPK - NUM_SHARED_EXPERTS
    routed_experts = NUM_EXPERTS - NUM_SHARED_EXPERTS
    scores = torch.randn(
        (num_tokens, routed_experts), dtype=torch.float32, device=device, generator=generator
    )
    scores[:, 0] += (1.0 - balancedness) * 6.0
    topk_ids = torch.topk(scores, routed_top_k, dim=-1).indices.to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(
            (num_tokens, TOPK), dtype=torch.float32, device=device, generator=generator
        ),
        dim=-1,
    )
    return topk_ids.contiguous(), topk_weights.contiguous()


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    device = torch.device(kwargs.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved Alpha-MoE")
    num_tokens = int(cfg.num_tokens)
    num_ctas = _num_ctas(**kwargs)
    actual_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if num_ctas > actual_sms:
        raise ValueError(
            f"kernel was built for {num_ctas} CTAs but the GPU has only {actual_sms} SMs"
        )

    generator = torch.Generator(device=device).manual_seed(int(cfg.seed))
    hidden_states = (
        torch.randn(
            (num_tokens, HIDDEN), dtype=torch.bfloat16, device=device, generator=generator
        )
        * HIDDEN_STATES_STD
    ).contiguous()
    gemm1 = (
        torch.randn(
            (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * WEIGHT_STD
    )
    gemm2 = (
        torch.randn(
            (NUM_EXPERTS, HIDDEN, INTERMEDIATE),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * WEIGHT_STD
    )
    gemm1_weights, gemm1_weights_scale = _quantize_block_2d(gemm1)
    gemm2_weights, gemm2_weights_scale = _quantize_block_2d(gemm2)
    del gemm1, gemm2
    topk_ids, topk_weights = _make_routing(generator, device, num_tokens, cfg.balancedness)

    # The nine contract inputs plus the preallocated output, exactly as the
    # task's ``setup(data, M)`` entry point consumes them.
    data = {
        "hidden_states": hidden_states,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "gemm1_weights": gemm1_weights,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
        "top_k": TOPK,
        "routed_scaling_factor": float(cfg.routed_scaling_factor),
        "output": torch.empty(num_tokens, HIDDEN, dtype=torch.bfloat16, device=device),
    }
    case: dict[str, Any] = {"config": cfg, "num_ctas": num_ctas, "data": data}
    case.update(data)
    return case


def _launcher(case: dict[str, Any]):
    """``setup`` compiles for the row's shape and returns the timed callable."""
    launch = setup(case["data"], int(case["config"].num_tokens))
    launch._keep_alive_case = case
    return launch


# ---------------------------------------------------------------------------
# Correctness (the task's independent oracle and element-wise bound)
# ---------------------------------------------------------------------------


def _quantize_per_token_group(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = values.shape
    groups = values.float().reshape(rows, columns // BLOCK_SIZE, BLOCK_SIZE)
    scales = groups.abs().amax(dim=-1).clamp_min(1.0e-8) / FP8_MAX
    quantized = (groups / scales.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return quantized.to(torch.float8_e4m3fn).reshape(rows, columns), scales


def _expand_block_scales(scales: torch.Tensor) -> torch.Tensor:
    return scales.float().repeat_interleave(BLOCK_SIZE, dim=1).repeat_interleave(BLOCK_SIZE, dim=2)


@torch.no_grad()
def _torch_reference(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent expert oracle on explicit FP8 bytes and the supplied routes.

    Gate/up and SwiGLU stay FP32; the intermediate is requantized per row in
    groups of 128; each routed group-128 down contribution and each accumulator
    update rounds to BF16.  ``abs_sum`` carries the absolute contribution mass
    for the accumulation-order bound.  Weights use the logical [gate; up]
    layout.
    """
    cfg: AlphaMoEConfig = case["config"]
    device = case["hidden_states"].device
    num_tokens = int(cfg.num_tokens)
    x_q, x_scale = _quantize_per_token_group(case["hidden_states"])
    topk_ids = case["topk_ids"]
    topk_weights = case["topk_weights"]

    pair_expert = topk_ids.reshape(-1).to(torch.int64)
    flat_weights = topk_weights.reshape(-1)
    x = x_q.float() * x_scale.repeat_interleave(BLOCK_SIZE, dim=1)
    w1 = case["gemm1_weights"].float() * _expand_block_scales(case["gemm1_weights_scale"])
    w2 = case["gemm2_weights"].float() * _expand_block_scales(case["gemm2_weights_scale"])

    output = torch.zeros((num_tokens, HIDDEN), dtype=torch.bfloat16, device=device)
    abs_sum = torch.zeros((num_tokens, HIDDEN), dtype=torch.float32, device=device)
    for expert in range(NUM_EXPERTS):
        pair_indices = torch.nonzero(pair_expert == expert, as_tuple=False).flatten()
        if pair_indices.numel() == 0:
            continue
        tokens = torch.div(pair_indices, TOPK, rounding_mode="floor")
        gate_up = x[tokens] @ w1[expert].t()
        gate, up = gate_up[:, :INTERMEDIATE], gate_up[:, INTERMEDIATE:]
        activated = torch.nn.functional.silu(gate) * up
        act_q, act_scale = _quantize_per_token_group(activated)
        activated = act_q.float() * act_scale.repeat_interleave(BLOCK_SIZE, dim=1)
        for base in range(0, INTERMEDIATE, BLOCK_SIZE):
            down = activated[:, base : base + BLOCK_SIZE] @ w2[
                expert, :, base : base + BLOCK_SIZE
            ].t()
            down *= flat_weights[pair_indices, None] * float(cfg.routed_scaling_factor)
            contribution = down.to(torch.bfloat16)
            output[tokens] = (output[tokens].float() + contribution.float()).to(torch.bfloat16)
            abs_sum[tokens] += contribution.float().abs()
    return output, abs_sum


def check_correctness(outputs: dict[str, Any], **kwargs: Any) -> None:
    """The task's pass/fail gate: the element-wise accumulation-order bound.

    Every launch must land inside ``max(0.1 + 0.1|ref|, 2 * abs_sum * 2**-7)``
    of the oracle, and two launches must agree with each other to within the
    same bound.  Note that repeated launches are NOT bitwise identical here:
    the down projection accumulates with ``red.global.add.bf16x2``, so the
    order in which a token's contributions land varies between launches and
    BF16 addition is not associative.  That is what the ``2 * abs_sum * 2**-7``
    term of the task's bound exists to cover.  The superseded M=128-only kernel
    used a deterministic last-arriver finalize and so could be checked with
    exact equality; this one cannot, and asserting bitwise repeatability here
    would be stricter than the task contract rather than more correct.
    """
    _cfg(**kwargs)
    first, actual = outputs["first"], outputs["actual"]
    reference, abs_sum = outputs["reference"], outputs["abs_sum"]
    for name, tensor in (("first", first), ("actual", actual), ("reference", reference)):
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} output contains non-finite values")
    ref = reference.float()
    bound = torch.maximum(0.1 + 0.1 * ref.abs(), 2.0 * abs_sum * 2.0**-7)
    for name, tensor in (("first", first), ("actual", actual)):
        diff = (tensor.float() - ref).abs()
        outside = int((diff > bound).sum())
        if outside:
            raise AssertionError(
                f"{name}: {outside} of {diff.numel()} elements exceed the "
                f"accumulation-order bound; max abs diff={float(diff.max())}"
            )
    spread = (first.float() - actual.float()).abs()
    unstable = int((spread > bound).sum())
    if unstable:
        raise AssertionError(
            f"{unstable} of {spread.numel()} elements differ between identical launches by "
            f"more than the accumulation-order bound; max abs diff={float(spread.max())}"
        )


def run_test(**kwargs: Any) -> None:
    _assert_supported_arch()
    config = dict(kwargs)
    config.pop("num_ctas", None)
    num_ctas = _num_ctas(**config)
    case = prepare_data(**config, num_ctas=num_ctas)
    launch = _launcher(case)
    case["output"].fill_(float("nan"))
    launch()
    torch.cuda.synchronize()
    first = case["output"].clone()
    case["output"].fill_(42.0)
    launch()
    torch.cuda.synchronize()
    actual = case["output"].clone()
    reference, abs_sum = _torch_reference(case)
    torch.cuda.synchronize()
    check_correctness(
        {"first": first, "actual": actual, "reference": reference, "abs_sum": abs_sum}, **config
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _flashinfer_builder(case: dict[str, Any]):
    """The packaged TRT-LLM baseline, called exactly as the locked harness does.

    ``trtllm_fp8_block_scale_routed_moe`` consumes preselected routes, so the
    routed scaling factor is folded into the supplied weights and the call is
    made with ``routed_scaling_factor=1.0``.  TRT-LLM wants [up; gate] and a
    multiple of four experts; the activation quantization stays inside the
    timed region, as the contract requires.

    The baseline is autotuned, warmed up on a side stream and then captured
    into a CUDA graph, so the timed callable is one ``graph.replay()``.  This
    mirrors the evaluation harness, whose alphamoe baseline captures the whole
    quantize-plus-expert operator in ``prepare()`` and times ``run_prepared``.
    Timing an uncaptured baseline instead charges it per-launch CPU dispatch
    that the harness does not, which inflates the reported speedup.
    """

    def build():
        from flashinfer import autotune
        from flashinfer.fused_moe import trtllm_fp8_block_scale_routed_moe

        cfg: AlphaMoEConfig = case["config"]
        num_tokens = int(cfg.num_tokens)
        gemm1_weights = case["gemm1_weights"]
        gemm1_weights_scale = case["gemm1_weights_scale"]
        gate, up = gemm1_weights.chunk(2, dim=1)
        gemm1_weights = torch.cat((up, gate), dim=1).contiguous()
        gate_s, up_s = gemm1_weights_scale.chunk(2, dim=1)
        gemm1_weights_scale = torch.cat((up_s, gate_s), dim=1).contiguous()
        weights = case["topk_weights"]
        if float(cfg.routed_scaling_factor) != 1.0:
            weights = (weights * float(cfg.routed_scaling_factor)).contiguous()

        def launch() -> None:
            x_q, x_scale = _quantize_per_token_group(case["hidden_states"])
            trtllm_fp8_block_scale_routed_moe(
                topk_ids=(case["topk_ids"], weights),
                routing_bias=None,
                hidden_states=x_q,
                hidden_states_scale=x_scale.t().contiguous(),
                gemm1_weights=gemm1_weights,
                gemm1_weights_scale=gemm1_weights_scale,
                gemm2_weights=case["gemm2_weights"],
                gemm2_weights_scale=case["gemm2_weights_scale"],
                num_experts=NUM_EXPERTS,
                top_k=TOPK,
                n_group=None,
                topk_group=None,
                intermediate_size=INTERMEDIATE,
                local_expert_offset=0,
                local_num_experts=NUM_EXPERTS,
                routed_scaling_factor=1.0,
                routing_method_type=5,
                use_shuffled_weight=False,
                tune_max_num_tokens=num_tokens,
            )

        with autotune(tuning_buckets=(max(1, num_tokens),)):
            launch()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            launch()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        return graph.replay

    return build


def prepare_bench(**kwargs: Any):
    """Trace and compile before the bench suite assigns a GPU."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    num_ctas = _num_ctas(**kwargs)
    config = dict(kwargs)
    config.pop("num_ctas", None)
    cfg = _cfg(**config)
    _precompile(
        int(cfg.num_tokens), TOPK, NUM_EXPERTS, HIDDEN, INTERMEDIATE, num_ctas
    )
    state = {"config": config, "num_ctas": num_ctas}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    _assert_supported_arch()
    config = dict(prepared["config"])
    config.update(kwargs)
    rounds = config.pop("rounds", 5)
    cooldown_s = config.pop("cooldown_s", 1.0)
    config.pop("num_ctas", None)
    case = prepare_data(**config, num_ctas=prepared["num_ctas"])
    launch = _launcher(case)
    launch()
    torch.cuda.synchronize()

    from tirx_kernels.runner import bench

    return bench(
        {"tirx": launch},
        references={"flashinfer_trtllm_fp8_block_scale_routed_moe": _flashinfer_builder(case)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    prepared = prepare_bench(**config)
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "check_correctness",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_test",
]
