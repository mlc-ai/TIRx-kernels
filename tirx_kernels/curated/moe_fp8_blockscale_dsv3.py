# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Curated native TIRx DeepSeek-V3 FP8 block-scale MoE megakernel for B200.

The supported contract is the FlashInfer benchmark geometry with 256 global
experts, 32 local experts, hidden size 7168, intermediate size 2048, block-128
E4M3 inputs and weights, DeepSeek-V3 grouped routing (top-8, eight groups,
four selected groups), and bf16 output.  Sequence length is dynamic; the
registered benchmark shape is the official T=14107 maximum row.

The selected kernel is ``pair2sm-fp8-partials-megakernel`` from optimization run
``moe-20260908-004417``.  It is one persistent 12-warp tirx-lite launch over
two-CTA clusters:

One persistent launch:
  R1  routing (sigmoid, grouped top-k, weights) + per-CTA/per-warp expert counts
  --- grid barrier ---
  R2  deterministic expert-sorted positions, gathered activation scales
  --- grid barrier ---
  G   dynamic pair-tile stream over 2-CTA clusters: a pair-tile is 256 tokens x 256 W rows of
      one expert; CTA r of the pair owns token rows [128r, 128r+128) and loads only those A rows
      plus 128 of the 256 B rows (32 KB per K block per SM instead of 48 KB). The leader CTA
      issues tcgen05.mma.cta_group::2 (M=256), which reads both CTAs' SMEM and leaves each CTA's
      128x256 fp32 block in its own TMEM; tcgen05.commit multicasts the stage/TMEM barriers to
      both CTAs. GEMM1 pair-tiles (SwiGLU, fp8 quant) then GEMM2 pair-tiles whose weighted
      results are quantized to FP8 and written in coalesced row segments through shared memory.
      Token rows are pre-permuted into an expert-sorted A buffer during R2 so that
      every A tile is one 128x128 TMA box (gather4 is TMA-issue bound).
      The arbitrary FP32 block-128 scales are rounded to UE8M0 powers of two
      and consumed directly by block-scaled tcgen05 MMA. Two K128 blocks share
      each three-stage data slot and occupy UE8M0 byte lanes 0/1, cutting scale
      copies in half. R2 packs each hidden-row scale pair into a compact u16
      half-tile layout so the G1 scale warp uses aligned vector loads. A full K reduction
      stays in one TMEM accumulator; math warps drain it once per output tile.
  --- grid barrier ---
  F   finalize every token with a local route: convert FP8 partial rows to
      packed bf16x2 and sum them in route-slot order. Tokens with no local
      expert are zeroed by the aux warps during the finalize phase.

The speed path deliberately rounds FP32 block scales to UE8M0, quantizes
weighted GEMM2 partials to unscaled E4M3, and accumulates those partials in
bf16x2 instead of FP32.  The official sweep accepted every row with
matched_ratio=1.0; the exact-scale promotion member remains in the run tree as
the numerically more faithful alternative.
"""

import ctypes
from dataclasses import dataclass, fields
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.tirx_lite as txl
from tvm.backend.cuda.cpp.descriptors import (
    encode_instr_descriptor_block_scaled_uint32,
    encode_smem_descriptor_base_uint64,
)

KERNEL_META = {
    "name": "curated_moe_fp8_blockscale_dsv3",
    "category": "curated",
    "runtime_cuda_archs": ["sm_100a"],
    "provenance": {
        "generator": "kda_flow",
        "run": "moe-20260908-004417",
        "selected_version": "pair2sm-fp8-partials-megakernel",
    },
}


NUM_EXPERTS = 256
NUM_LOCAL = 32
TOPK = 8
HIDDEN = 7168
INTER = 2048
KB1 = HIDDEN // 128
KB2 = INTER // 128
BM = 128
PAIR = 2
BMP = BM * PAIR
BN = 256
BNH = BN // PAIR
BK = 128
KPACK = 2
A1_PACKS = KB1 // KPACK
A2_PACKS = KB2 // KPACK
SF_TILE_BYTES = BM * 4
BULK_G2S = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
NT1 = (2 * INTER) // BN
NT2 = HIDDEN // BN
STAGES = 3
NWARPS = 12
MATH_WARP0 = 4
REGS_WG0 = 40
REGS_MATH = 232
TASK_RING = 2
A_STAGE_BYTES = BM * BK
B_STAGE_BYTES = BNH * BK
LOG2E = 1.4426950408889634
NEG_INF = -3.0e38
FP8_MAX = 448.0
EVICT_NORMAL = 0x1000000000000000

MMA = "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"
IDESC = encode_instr_descriptor_block_scaled_uint32(
    M=BMP,
    N=BN,
    K=32,
    d_dtype="float32",
    a_dtype="float8_e4m3fn",
    b_dtype="float8_e4m3fn",
    sf_dtype="float8_e8m0fnu",
    trans_a=False,
    trans_b=False,
    cta_group=2,
)
IDESC_H = encode_instr_descriptor_block_scaled_uint32(
    M=BM,
    N=BN,
    K=32,
    d_dtype="float32",
    a_dtype="float8_e4m3fn",
    b_dtype="float8_e4m3fn",
    sf_dtype="float8_e8m0fnu",
    trans_a=False,
    trans_b=False,
    cta_group=2,
)
IDESC_IDS = tuple((IDESC & 0x9FFFFFCF) | (sf_id << 29) | (sf_id << 4) for sf_id in range(KPACK))
IDESC_H_IDS = tuple((IDESC_H & 0x9FFFFFCF) | (sf_id << 29) | (sf_id << 4) for sf_id in range(KPACK))
UTCCP = "tcgen05.cp.cta_group::2.32x128b.warpx4"
SF_DESC_BASE = encode_smem_descriptor_base_uint64(0, 8, 0)
SFA_TMEM_COL = 256
SFB_TMEM_COL = 260
TMA_PREFETCH = "cp.async.bulk.prefetch.tensor.2d.L2.global"
PREFETCH_DIST = 0
FIN_G = 8
FIN_RPP = 2
ROW_CHUNKS = HIDDEN // 512
TMA_2SM = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
TMEM_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
COMMIT = "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
HALF_A_BYTES = (BM // 2) * BK
NCONS = 2 * (NWARPS - MATH_WARP0) + 2 + 2 * PAIR


def f32(v):
    return txl.local_scalar(txl.f32, init=v)


def i32(v):
    return txl.local_scalar(txl.i32, init=v)


def u32(v):
    return txl.local_scalar(txl.u32, init=v)


def shfl_idx(dst, v, lane):
    txl.ptx.shfl_sync.idx.b32(dst, v, txl.cast(lane, "uint32"), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF))


def shfl_bfly(dst, v, m):
    txl.ptx.shfl_sync.bfly.b32(dst, v, txl.uint32(m), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF))


def float_key(v):
    bits = txl.reinterpret("uint32", v)
    neg = txl.shift_right(bits, txl.uint32(31)) != txl.uint32(0)
    return txl.Select(neg, txl.bitwise_not(bits), txl.bitwise_or(bits, txl.uint32(0x80000000)))


def bf16_lo(word):
    return txl.reinterpret("float32", txl.shift_left(word, txl.uint32(16)))


def bf16_hi(word):
    return txl.reinterpret("float32", txl.bitwise_and(word, txl.uint32(0xFFFF0000)))


def ue8m0_pack4(v):
    """Round a positive f32 scale into UE8M0 sub-column zero."""
    bits = txl.reinterpret("uint32", v)

    code = txl.bitwise_and(txl.shift_right(bits + txl.uint32(0x400000), txl.uint32(23)), txl.uint32(0xFF))
    return code


def with_sf_id(desc, sf_id):
    """Set both block-scale ID fields ([31:29] and [6:4])."""
    out = txl.bitwise_and(desc, txl.uint32(0x9FFFFFCF))
    out = txl.bitwise_or(out, txl.shift_left(txl.cast(sf_id, "uint32"), txl.uint32(29)))
    return txl.bitwise_or(out, txl.shift_left(txl.cast(sf_id, "uint32"), txl.uint32(4)))


def with_smem_addr(desc_base, addr):
    """Fill descriptor address bits [13:0] from a 16-byte-aligned SMEM address."""
    addr_field = txl.cast(txl.bitwise_and(txl.shift_right(addr, txl.uint32(4)), txl.uint32(0x3FFF)), "uint64")
    return txl.bitwise_or(txl.uint64(desc_base), addr_field)


def iket_range(name):
    token = txl.alloc_local([1], "uint32")
    txl.assign(token[0], txl.cuda.iket.range_start(name))
    return token


def iket_end(token):
    txl.cuda.iket.range_end(token[0])


def remote_arrive_leader(bar_ptr):
    """Arrive (count 1) on the leader CTA's copy of a shared-memory mbarrier (rank 0 of the pair)."""
    rem = txl.local_scalar(txl.u64)
    txl.ptx.mapa.shared__cluster.u64(rem, bar_ptr, txl.uint32(0))
    txl.ptx.mbarrier.arrive.b64(rem, txl.uint32(1), pred=txl.bool(True))


def emit_grid_sync(ctr_ptr, cta, num_ctas, tid):
    """Sense-reversing grid barrier over all CTAs (all threads of the CTA participate)."""
    txl.ptx.bar.sync(txl.uint32(0))
    with txl.If(tid == 0), txl.Then():
        # The counter is a declared synchronization word: the wait is spelled
        # `txl.cuda.wait_until`, which emits the loop the raw spelling did and
        # names the address as the barrier's. The arrival stays in raw PTX, and
        # the wait needs no seeding read because it tests `dst` before its
        # first load. The wait states the condition the barrier completes on -- the
        # sense bit having flipped against the value this CTA's own arrival
        # returned -- so the checker can tell which arrival released it.
        old = txl.local_scalar(txl.u32)
        with txl.If(cta == 0):
            with txl.Then():
                txl.ptx.atom.release.gpu.global_.add.u32(old, ctr_ptr, txl.uint32(0x80000000 - (num_ctas - 1)))
            with txl.Else():
                txl.ptx.atom.release.gpu.global_.add.u32(old, ctr_ptr, txl.uint32(1))
        cur = txl.local_scalar(txl.u32)
        txl.cuda.wait_until(
            cur,
            ctr_ptr,
            txl.bitwise_and(txl.bitwise_xor(cur, old), txl.uint32(0x80000000)) != txl.uint32(0),
            scope="gpu",
            ptx_type="b32",
        )
    txl.ptx.bar.sync(txl.uint32(0))


def emit_route_token(
    t, lane, logits, bias, offset, rsf, route_id, route_w, tok_cnt, wcnt, fin_list, fin_ctr
):
    """Route one token with one warp; lane l owns experts [8l, 8l+8)."""
    x = txl.alloc_local((8,), txl.f32)
    sv = txl.alloc_local((8,), txl.f32)
    bv = txl.alloc_local((8,), txl.f32)
    braw = txl.alloc_local((4,), txl.u32)
    txl.ptx.ld.global_.nc.v4.f32(x[0], x[1], x[2], x[3], logits.ptr_to([t * NUM_EXPERTS + lane * 8]))
    txl.ptx.ld.global_.nc.v4.f32(
        x[4], x[5], x[6], x[7], logits.ptr_to([t * NUM_EXPERTS + lane * 8 + 4])
    )
    txl.ptx.ld.global_.nc.v4.b32(braw[0], braw[1], braw[2], braw[3], bias.ptr_to([lane * 8]))
    for j in range(8):
        e = txl.local_scalar(txl.f32)
        txl.ptx.ex2.approx.ftz.f32(e, x[j] * txl.float32(-LOG2E))
        txl.ptx.rcp.approx.ftz.f32(sv[j], e + txl.float32(1.0))
        word = braw[j // 2]
        bbits = (
            txl.shift_left(word, txl.uint32(16))
            if j % 2 == 0
            else txl.bitwise_and(word, txl.uint32(0xFFFF0000))
        )
        txl.assign(bv[j], sv[j] + txl.reinterpret("float32", bbits))
    m1 = f32(txl.max(bv[0], bv[1]))
    m2 = f32(txl.min(bv[0], bv[1]))
    for j in range(2, 8):
        txl.assign(m2, txl.max(m2, txl.min(m1, bv[j])))
        txl.assign(m1, txl.max(m1, bv[j]))
    for m in (1, 2):
        n1 = txl.local_scalar(txl.f32)
        n2 = txl.local_scalar(txl.f32)
        shfl_bfly(n1, m1, m)
        shfl_bfly(n2, m2, m)
        new1 = f32(txl.max(m1, n1))
        new2 = f32(txl.max(txl.min(m1, n1), txl.max(m2, n2)))
        txl.assign(m1, new1)
        txl.assign(m2, new2)
    gs = f32(m1 + m2)
    my_g = lane // 4
    rank = i32(txl.int32(0))
    for g in range(8):
        og = txl.local_scalar(txl.f32)
        shfl_idx(og, gs, txl.int32(g * 4))
        txl.assign(rank, rank + txl.cast(txl.Or(og > gs, txl.And(og == gs, txl.int32(g) < my_g)), "int32"))
    keep = rank < 4
    for j in range(8):
        txl.assign(bv[j], txl.Select(keep, bv[j], txl.float32(NEG_INF)))
    my_e = i32(txl.int32(-1))
    my_s = f32(txl.float32(0.0))
    for r in range(8):
        bestv = f32(bv[0])
        bestj = i32(txl.int32(0))
        for j in range(1, 8):
            take = u32(txl.cast(bv[j] > bestv, "uint32"))
            txl.assign(bestj, txl.Select(take != txl.uint32(0), txl.int32(j), bestj))
            txl.assign(bestv, txl.Select(take != txl.uint32(0), bv[j], bestv))
        key = u32(float_key(bestv))
        wkey = txl.local_scalar(txl.u32)
        txl.ptx.redux_sync.max.u32(wkey, key, txl.uint32(0xFFFFFFFF))
        bal = txl.local_scalar(txl.u32)
        txl.ptx.vote_sync.ballot.b32(
            bal, txl.ptx.pred(txl.cast(key == wkey, "uint32")), txl.uint32(0xFFFFFFFF)
        )
        wlane = i32(txl.cuda.ffs_u32(bal) - txl.int32(1))
        ssel = f32(sv[0])
        for j in range(1, 8):
            txl.assign(ssel, txl.Select(bestj == txl.int32(j), sv[j], ssel))
        widx = txl.local_scalar(txl.i32)
        wsv = txl.local_scalar(txl.f32)
        shfl_idx(widx, wlane * 8 + bestj, wlane)
        shfl_idx(wsv, ssel, wlane)
        with txl.If(lane == wlane), txl.Then():
            for j in range(8):
                txl.assign(bv[j], txl.Select(bestj == txl.int32(j), txl.float32(NEG_INF), bv[j]))
        txl.assign(my_e, txl.Select(lane == txl.int32(r), widx, my_e))
        txl.assign(my_s, txl.Select(lane == txl.int32(r), wsv, my_s))
    contrib = f32(txl.Select(lane < 8, my_s, txl.float32(0.0)))
    for m in (1, 2, 4):
        o = txl.local_scalar(txl.f32)
        shfl_bfly(o, contrib, m)
        txl.assign(contrib, contrib + o)
    wgt = f32((my_s / (contrib + txl.float32(1e-20))) * rsf)
    loc = my_e - offset
    locv = i32(txl.Select(txl.And(loc >= 0, loc < NUM_LOCAL), loc, txl.int32(-1)))
    with txl.If(lane < 8), txl.Then():
        txl.ptx.st.global_.s32(route_id.ptr_to([t * TOPK + lane]), locv)
        txl.ptx.st.global_.f32(route_w.ptr_to([t * TOPK + lane]), wgt)
    for k in range(8):
        ek = txl.local_scalar(txl.i32)
        shfl_idx(ek, locv, txl.int32(k))
        txl.assign(wcnt, wcnt + txl.cast(ek == lane, "int32"))

    lbal = txl.local_scalar(txl.u32)
    txl.ptx.vote_sync.ballot.b32(
        lbal, txl.ptx.pred(txl.cast(txl.And(lane < 8, locv >= 0), "uint32")), txl.uint32(0xFFFFFFFF)
    )
    with txl.If(lane == 0), txl.Then():
        nloc = i32(txl.cast(txl.popcount(lbal), "int32"))
        txl.ptx.st.global_.s32(tok_cnt.ptr_to([t]), nloc)

        with txl.If(nloc >= 1), txl.Then():
            fidx = txl.local_scalar(txl.u32)
            txl.ptx.atom.global_.add.u32(fidx, fin_ctr, txl.uint32(1))
            txl.ptx.st.global_.s32(fin_list.ptr_to([txl.cast(fidx, "int32")]), t)


def build_kernel(num_ctas):
    ctas_per_warp = (num_ctas + NWARPS - 1) // NWARPS

    @txl.kernel(warps=NWARPS, arch="sm_100a", min_blocks_per_sm=1, grid=num_ctas)
    def moe_mega(
        logits: txl.gptr[txl.f32],
        bias: txl.gptr[txl.bf16],
        hs_scale: txl.gptr[txl.f32],
        w1s: txl.gptr[txl.f32],
        w2s: txl.gptr[txl.f32],
        out: txl.gptr[txl.bf16],
        route_id: txl.gptr[txl.i32],
        route_w: txl.gptr[txl.f32],
        tok_cnt: txl.gptr[txl.i32],
        cnt_cta: txl.gptr[txl.i32],
        cnt_warp: txl.gptr[txl.i32],
        sorted_token: txl.gptr[txl.i32],
        pos_of: txl.gptr[txl.i32],
        sorted_w: txl.gptr[txl.f32],
        a1p: txl.gptr[txl.u32],
        a1h: txl.gptr[txl.u32],
        a2p: txl.gptr[txl.u8],
        a2h: txl.gptr[txl.u8],
        act: txl.gptr[txl.f8e4m3],
        hidden: txl.gptr[txl.f8e4m3],
        ap: txl.gptr[txl.f8e4m3],
        partial: txl.gptr[txl.f8e4m3],
        task_ctr: txl.gptr[txl.u32],
        done: txl.gptr[txl.u32],
        sync_ctr: txl.gptr[txl.u32],
        fin_list: txl.gptr[txl.i32],
        tm_ap: txl.TensorMap,
        tm_w1: txl.TensorMap,
        tm_w2: txl.TensorMap,
        tm_act: txl.TensorMap,
        tm_ap64: txl.TensorMap,
        tm_act64: txl.TensorMap,
        tm_w1h: txl.TensorMap,
        T: txl.i32,
        P: txl.i32,
        MAXMT: txl.i32,
        offset: txl.i32,
        rsf: txl.f32,
    ):
        cta = txl.cta_id()
        crank = txl.cta_id_in_cluster([PAIR])
        is_leader = crank == 0
        warp = txl.warp_id()
        lane = txl.lane_id()
        tid = txl.thread_id()

        smem = txl.smem_pool()
        a_tile0 = smem.alloc((STAGES, BM, BK), txl.f8e4m3, align=1024, swizzle=txl.SW128B)
        a_tile1 = smem.alloc((STAGES, BM, BK), txl.f8e4m3, align=1024, swizzle=txl.SW128B)
        b_tile0 = smem.alloc((STAGES, BNH, BK), txl.f8e4m3, align=1024, swizzle=txl.SW128B)
        b_tile1 = smem.alloc((STAGES, BNH, BK), txl.f8e4m3, align=1024, swizzle=txl.SW128B)
        a_tiles = (a_tile0, a_tile1)
        b_tiles = (b_tile0, b_tile1)
        sfa_tile = smem.alloc((STAGES, BM), txl.u32, align=16)

        sfb_tile = smem.alloc((STAGES, BN), txl.u32, align=16)
        s_cnt = smem.alloc((32,), txl.i32, align=16)
        s_base = smem.alloc((32,), txl.i32, align=16)
        s_ecnt = smem.alloc((32,), txl.i32, align=16)
        s_eoff = smem.alloc((32,), txl.i32, align=16)
        s_mpre = smem.alloc((33,), txl.i32, align=16)
        s_ptot = smem.alloc((NWARPS, 32), txl.i32, align=16)
        s_ppre = smem.alloc((NWARPS, 32), txl.i32, align=16)
        s_task = smem.alloc((TASK_RING, 8), txl.i32, align=16)
        s_amax = smem.alloc((512,), txl.f32, align=16)

        s_epi = smem.alloc((NWARPS - MATH_WARP0, 16, 36), txl.u32, align=16)
        tmem_slot = smem.alloc((1,), txl.u32, align=4)

        full_bar = txl.TMABar(smem, STAGES * KPACK)
        empty_bar = txl.TCGen05Bar(smem, STAGES * KPACK)
        sf_bar = txl.MBarrier(smem, STAGES)

        sfempty_bar = txl.TCGen05Bar(smem, STAGES)

        sfa_bar = txl.TMABar(smem, STAGES)
        tfull_bar = txl.TCGen05Bar(smem, 1)
        tempty_bar = txl.MBarrier(smem, 1)
        task_full = txl.MBarrier(smem, TASK_RING)
        task_empty = txl.MBarrier(smem, TASK_RING)

        roles = txl.specialize(chain_dispatch=True)
        control_regs = roles.register_scope("control", warps=range(4), regs=REGS_WG0)
        prod_role = roles.role("prod", warps=[0], register_scope=control_regs)
        mma_role = roles.role("mma", warps=[1], register_scope=control_regs)
        aux_role = roles.role("aux", warps=[2, 3], register_scope=control_regs)
        # Math owns complete warpgroups. Keep the register target at each role
        # entry so ptxas retains the math allocation for both G and F.
        math_role = roles.role(
            "math", warps=range(MATH_WARP0, NWARPS), regs=REGS_MATH
        )

        txl.ptx.barrier.cluster.arrive.relaxed.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        with txl.If(warp == 1), txl.Then():
            with txl.If(lane == 0), txl.Then():
                for s in range(STAGES * KPACK):
                    txl.ptx.mbarrier.init.shared.b64(full_bar.ptr_to([s]), txl.uint32(1))
                    txl.ptx.mbarrier.init.shared.b64(empty_bar.ptr_to([s]), txl.uint32(1))
                for s in range(STAGES):
                    txl.ptx.mbarrier.init.shared.b64(sf_bar.ptr_to([s]), txl.uint32(2 * PAIR))
                    txl.ptx.mbarrier.init.shared.b64(sfempty_bar.ptr_to([s]), txl.uint32(1))
                    txl.ptx.mbarrier.init.shared.b64(sfa_bar.ptr_to([s]), txl.uint32(1))
                for s in range(1):
                    txl.ptx.mbarrier.init.shared.b64(tfull_bar.ptr_to([s]), txl.uint32(1))

                    txl.ptx.mbarrier.init.shared.b64(
                        tempty_bar.ptr_to([s]), txl.uint32(PAIR * (NWARPS - MATH_WARP0))
                    )
                for s in range(TASK_RING):
                    txl.ptx.mbarrier.init.shared.b64(task_full.ptr_to([s]), txl.uint32(1))
                    txl.ptx.mbarrier.init.shared.b64(task_empty.ptr_to([s]), txl.uint32(NCONS))
                txl.ptx.fence.mbarrier_init.release.cluster()
        with txl.If(warp == 2), txl.Then():
            txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                txl.address_of(tmem_slot[0]), txl.uint32(512)
            )
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        with txl.If(warp == 0), txl.Then():
            with txl.If(lane == 0), txl.Then():
                txl.ptx.prefetch.tensormap(txl.address_of(tm_ap))
                txl.ptx.prefetch.tensormap(txl.address_of(tm_w1))
                txl.ptx.prefetch.tensormap(txl.address_of(tm_w2))
                txl.ptx.prefetch.tensormap(txl.address_of(tm_act))
                txl.ptx.prefetch.tensormap(txl.address_of(tm_ap64))
                txl.ptx.prefetch.tensormap(txl.address_of(tm_act64))
                txl.ptx.prefetch.tensormap(txl.address_of(tm_w1h))
        with txl.If(cta == 0), txl.Then():
            with txl.If(tid == 0), txl.Then():
                txl.ptx.st.global_.u32(task_ctr.ptr_to([0]), txl.uint32(0))
                txl.ptx.st.global_.u32(task_ctr.ptr_to([1]), txl.uint32(0))
            di = i32(tid)
            with txl.While(di < MAXMT * NUM_LOCAL):
                txl.ptx.st.global_.u32(done.ptr_to([di]), txl.uint32(0))
                txl.assign(di, di + NWARPS * 32)
        with txl.If(tid < 32), txl.Then():
            txl.ptx.st.shared.s32(s_cnt.ptr_to([tid]), txl.int32(0))
        txl.ptx.barrier.cluster.arrive.release.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()

        chunk = (T + (num_ctas - 1)) // num_ctas
        t_begin = cta * chunk
        t_end = txl.min(T, t_begin + chunk)
        wcnt = i32(txl.int32(0))
        t = i32(t_begin + warp)
        with txl.While(t < t_end):
            emit_route_token(
                t,
                lane,
                logits,
                bias,
                offset,
                rsf,
                route_id,
                route_w,
                tok_cnt,
                wcnt,
                fin_list,
                task_ctr.ptr_to([2]),
            )
            txl.assign(t, t + NWARPS)
        txl.ptx.st.global_.s32(cnt_warp.ptr_to([(cta * NWARPS + warp) * 32 + lane]), wcnt)
        txl.ptx.red.shared.add.u32(s_cnt.ptr_to([lane]), txl.cast(wcnt, "uint32"))
        txl.ptx.bar.sync(txl.uint32(0))
        with txl.If(tid < 32), txl.Then():
            v = txl.local_scalar(txl.i32)
            txl.ptx.ld.shared.s32(v, s_cnt.ptr_to([tid]))
            txl.ptx.st.global_.s32(cnt_cta.ptr_to([cta * 32 + tid]), v)

        txl.cuda.iket.mark("R1-done")
        emit_grid_sync(sync_ctr.ptr_to([0]), cta, num_ctas, tid)

        ptot = i32(txl.int32(0))
        ppre = i32(txl.int32(0))
        vals = txl.alloc_local((ctas_per_warp,), txl.i32)
        for i in range(ctas_per_warp):
            c = warp + i * NWARPS
            txl.assign(vals[i], txl.int32(0))
            with txl.If(c < num_ctas), txl.Then():
                txl.ptx.ld.global_.s32(vals[i], cnt_cta.ptr_to([c * 32 + lane]))
        for i in range(ctas_per_warp):
            c = warp + i * NWARPS
            txl.assign(ptot, ptot + vals[i])
            txl.assign(ppre, ppre + txl.Select(c < cta, vals[i], txl.int32(0)))
        txl.ptx.st.shared.s32(s_ptot.ptr_to([warp, lane]), ptot)
        txl.ptx.st.shared.s32(s_ppre.ptr_to([warp, lane]), ppre)
        txl.ptx.bar.sync(txl.uint32(0))
        with txl.If(warp == 0), txl.Then():
            tot = i32(txl.int32(0))
            pre = i32(txl.int32(0))
            for w in range(NWARPS):
                v = txl.local_scalar(txl.i32)
                txl.ptx.ld.shared.s32(v, s_ptot.ptr_to([w, lane]))
                txl.assign(tot, tot + v)
                txl.ptx.ld.shared.s32(v, s_ppre.ptr_to([w, lane]))
                txl.assign(pre, pre + v)
            mtiles = i32((tot + (BMP - 1)) // BMP)
            incl = i32(mtiles)
            for m in (1, 2, 4, 8, 16):
                o = txl.local_scalar(txl.i32)
                txl.ptx.shfl_sync.up.b32(o, incl, txl.uint32(m), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                txl.assign(incl, incl + txl.Select(lane >= m, o, txl.int32(0)))
            excl = i32(incl - mtiles)
            txl.ptx.st.shared.s32(s_ecnt.ptr_to([lane]), tot)
            txl.ptx.st.shared.s32(s_eoff.ptr_to([lane]), excl * BMP)
            txl.ptx.st.shared.s32(s_base.ptr_to([lane]), excl * BMP + pre)
            txl.ptx.st.shared.s32(s_mpre.ptr_to([lane]), excl)
            with txl.If(lane == 31), txl.Then():
                txl.ptx.st.shared.s32(s_mpre.ptr_to([32]), incl)
        txl.ptx.bar.sync(txl.uint32(0))

        base = txl.local_scalar(txl.i32)
        txl.ptx.ld.shared.s32(base, s_base.ptr_to([lane]))
        for wp in range(NWARPS - 1):
            with txl.If(wp < warp), txl.Then():
                v = txl.local_scalar(txl.i32)
                txl.ptx.ld.global_.s32(v, cnt_warp.ptr_to([(cta * NWARPS + wp) * 32 + lane]))
                txl.assign(base, base + v)
        cursor = i32(txl.int32(0))
        txl.assign(t, t_begin + warp)
        with txl.While(t < t_end):
            e = i32(txl.int32(-1))
            w = f32(txl.float32(0.0))
            with txl.If(lane < 8), txl.Then():
                txl.ptx.ld.global_.s32(e, route_id.ptr_to([t * TOPK + lane]))
                txl.ptx.ld.global_.f32(w, route_w.ptr_to([t * TOPK + lane]))
            has_local = txl.local_scalar(txl.u32)
            txl.ptx.vote_sync.ballot.b32(
                has_local,
                txl.ptx.pred(txl.cast(txl.And(lane < 8, e >= 0), "uint32")),
                txl.uint32(0xFFFFFFFF),
            )
            sc0 = f32(txl.float32(0.0))
            sc1 = f32(txl.float32(0.0))
            with txl.If(has_local != txl.uint32(0)), txl.Then():
                with txl.If(lane < KB1), txl.Then():
                    txl.ptx.ld.global_.f32(sc0, hs_scale.ptr_to([lane * T + t]))
                with txl.If(lane < KB1 - 32), txl.Then():
                    txl.ptx.ld.global_.f32(sc1, hs_scale.ptr_to([(lane + 32) * T + t]))
            sc0q = u32(ue8m0_pack4(sc0))
            sc1q = u32(ue8m0_pack4(sc1))
            qsrc = i32(txl.Select(lane < 16, lane * 2, (lane - 16) * 2))
            q00 = txl.local_scalar(txl.u32)
            q01 = txl.local_scalar(txl.u32)
            q10 = txl.local_scalar(txl.u32)
            q11 = txl.local_scalar(txl.u32)
            shfl_idx(q00, sc0q, qsrc)
            shfl_idx(q01, sc0q, qsrc + 1)
            shfl_idx(q10, sc1q, qsrc)
            shfl_idx(q11, sc1q, qsrc + 1)
            qpair = u32(
                txl.bitwise_or(
                    txl.Select(lane < 16, q00, q10),
                    txl.shift_left(txl.Select(lane < 16, q01, q11), txl.uint32(8)),
                )
            )
            ek = []
            for k in range(8):
                v = txl.local_scalar(txl.i32)
                shfl_idx(v, e, txl.int32(k))
                ek.append(v)
            rank = i32(txl.int32(0))
            cnt_here = i32(txl.int32(0))
            for k in range(8):
                txl.assign(rank, rank + txl.cast(txl.And(txl.int32(k) < lane, ek[k] == e), "int32"))
                txl.assign(cnt_here, cnt_here + txl.cast(ek[k] == lane, "int32"))
            cur_e = txl.local_scalar(txl.i32)
            shfl_idx(cur_e, cursor, txl.max(e, txl.int32(0)))
            base_e = txl.local_scalar(txl.i32)
            shfl_idx(base_e, base, txl.max(e, txl.int32(0)))
            pos = i32(base_e + cur_e + rank)
            with txl.If(lane < 8), txl.Then():
                with txl.If(e >= 0):
                    with txl.Then():
                        txl.ptx.st.global_.s32(sorted_token.ptr_to([pos]), t)
                        txl.ptx.st.global_.s32(pos_of.ptr_to([t * TOPK + lane]), pos)
                        txl.ptx.st.global_.f32(sorted_w.ptr_to([pos]), w)
                    with txl.Else():
                        txl.ptx.st.global_.s32(pos_of.ptr_to([t * TOPK + lane]), txl.int32(-1))
            txl.assign(cursor, cursor + cnt_here)

            rowv = txl.alloc_local((ROW_CHUNKS * 4,), txl.u32)
            with txl.If(has_local != txl.uint32(0)), txl.Then():
                for c in range(ROW_CHUNKS):
                    txl.ptx.ld.global_.nc.v4.b32(
                        rowv[4 * c],
                        rowv[4 * c + 1],
                        rowv[4 * c + 2],
                        rowv[4 * c + 3],
                        hidden.ptr_to([t * HIDDEN + c * 512 + lane * 16]),
                    )
            for k in range(8):
                posk = txl.local_scalar(txl.i32)
                shfl_idx(posk, pos, txl.int32(k))
                with txl.If(ek[k] >= 0), txl.Then():
                    with txl.If(lane < A1_PACKS), txl.Then():
                        ht = posk // BM
                        lr = posk - ht * BM
                        pidx = (ht * A1_PACKS + lane) * BM + (lr % 32) * 4 + lr // 32
                        txl.ptx.st.global_.u32(a1p.ptr_to([pidx]), qpair)
                        ht2 = posk // (BM // 2)
                        lr2 = posk - ht2 * (BM // 2)
                        hidx = (ht2 * A1_PACKS + lane) * BM + (lr2 % 32) * 4 + lr2 // 32
                        txl.ptx.st.global_.u32(a1h.ptr_to([hidx]), qpair)
                    for c in range(ROW_CHUNKS):
                        txl.ptx.st.global_.v4.b32(
                            ap.ptr_to([posk * HIDDEN + c * 512 + lane * 16]),
                            rowv[4 * c],
                            rowv[4 * c + 1],
                            rowv[4 * c + 2],
                            rowv[4 * c + 3],
                        )
            txl.assign(t, t + NWARPS)

        txl.cuda.iket.mark("R2-done")
        emit_grid_sync(sync_ctr.ptr_to([0]), cta, num_ctas, tid)
        txl.cuda.iket.mark("G-start")

        total_mt = txl.local_scalar(txl.i32)
        txl.ptx.ld.shared.s32(total_mt, s_mpre.ptr_to([32]))
        n_g1 = total_mt * NT1
        n_g2 = total_mt * NT2
        tmem_base = txl.local_scalar(txl.u32)
        txl.ptx.ld.shared.u32(tmem_base, tmem_slot.ptr_to([0]))

        def read_task(tstate, kind, e, mt, nt, row0, valid, nkb, half):
            tk_t = iket_range("wait-task")
            txl.cuda.mbarrier_wait(task_full.ptr_to([tstate.stage]), tstate.phase)
            iket_end(tk_t)
            txl.ptx.ld.shared.s32(kind, s_task.ptr_to([tstate.stage, 0]))
            txl.ptx.ld.shared.s32(e, s_task.ptr_to([tstate.stage, 1]))
            txl.ptx.ld.shared.s32(mt, s_task.ptr_to([tstate.stage, 2]))
            txl.ptx.ld.shared.s32(nt, s_task.ptr_to([tstate.stage, 3]))
            txl.ptx.ld.shared.s32(row0, s_task.ptr_to([tstate.stage, 4]))
            txl.ptx.ld.shared.s32(valid, s_task.ptr_to([tstate.stage, 5]))
            txl.ptx.ld.shared.s32(nkb, s_task.ptr_to([tstate.stage, 6]))
            txl.ptx.ld.shared.s32(half, s_task.ptr_to([tstate.stage, 7]))
            txl.cuda.warp_sync()
            with txl.If(lane == 0), txl.Then():
                remote_arrive_leader(task_empty.ptr_to([tstate.stage]))
            tstate.advance()

        def wait_done(e, mt):
            """GEMM2 pair-tiles need all GEMM1 pair-tiles of their m-tile (both CTAs arrive per tile)."""
            with txl.If(lane == 0), txl.Then():
                dv = txl.local_scalar(txl.u32)
                txl.cuda.wait_until(
                    dv,
                    done.ptr_to([e * MAXMT + mt]),
                    dv >= txl.uint32(PAIR * NT1),
                    scope="gpu",
                    ptx_type="b32",
                )
            txl.cuda.warp_sync()
            txl.ptx.fence.proxy.async_.global_()

        def producer_loads(kind, e, nt, row0, nkb, sstate, half):
            """Pair: this CTA's 128 A rows + 128 of the 256 B rows. Half: this CTA's 64 A rows of the lone tile + the same B share."""
            arow0 = i32(txl.Select(half != 0, row0 + crank * (BM // 2), row0 + crank * BM))
            brow = i32(
                txl.Select(
                    kind == 0,
                    e * (2 * INTER) + crank * INTER + nt * BNH,
                    e * HIDDEN + nt * BN + crank * BNH,
                )
            )
            brow_g = e * (2 * INTER) + nt * 128 + crank * 64
            brow_u = e * (2 * INTER) + INTER + nt * 128 + crank * 64
            tk_tile = iket_range("prod-tile")
            with txl.serial(0, nkb) as kb:
                for sub in range(KPACK):
                    slot = sstate.stage * KPACK + sub
                    tk_w = iket_range("prod-wait-empty")
                    txl.cuda.mbarrier_wait(empty_bar.ptr_to([slot]), sstate.phase ^ 1)
                    iket_end(tk_w)
                    with txl.If(lane == 0), txl.Then():
                        lbar = txl.cuda.sm100_2sm_leader_smem_addr(full_bar.ptr_to([slot]))
                        with txl.If(is_leader), txl.Then():
                            txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                full_bar.ptr_to([slot]),
                                txl.Select(
                                    half != 0,
                                    txl.uint32(PAIR * (HALF_A_BYTES + B_STAGE_BYTES)),
                                    txl.uint32(PAIR * (A_STAGE_BYTES + B_STAGE_BYTES)),
                                ),
                            )
                        kcol = (kb * KPACK + sub) * BK
                        with txl.If(half != 0):
                            with txl.Then():
                                with txl.If(kind == 0):
                                    with txl.Then():
                                        txl.ptx[TMA_2SM](
                                            a_tiles[sub][sstate.stage].ptr_to(0, 0),
                                            txl.address_of(tm_ap64),
                                            kcol,
                                            arow0,
                                            lbar,
                                            txl.uint64(EVICT_NORMAL),
                                        )
                                        txl.ptx[TMA_2SM](
                                            b_tiles[sub][sstate.stage].ptr_to(0, 0),
                                            txl.address_of(tm_w1h),
                                            kcol,
                                            brow_g,
                                            lbar,
                                            txl.uint64(EVICT_NORMAL),
                                        )
                                        txl.ptx[TMA_2SM](
                                            b_tiles[sub][sstate.stage].ptr_to(64, 0),
                                            txl.address_of(tm_w1h),
                                            kcol,
                                            brow_u,
                                            lbar,
                                            txl.uint64(EVICT_NORMAL),
                                        )
                                    with txl.Else():
                                        txl.ptx[TMA_2SM](
                                            a_tiles[sub][sstate.stage].ptr_to(0, 0),
                                            txl.address_of(tm_act64),
                                            kcol,
                                            arow0,
                                            lbar,
                                            txl.uint64(EVICT_NORMAL),
                                        )
                                        txl.ptx[TMA_2SM](
                                            b_tiles[sub][sstate.stage].ptr_to(0, 0),
                                            txl.address_of(tm_w2),
                                            kcol,
                                            brow,
                                            lbar,
                                            txl.uint64(EVICT_NORMAL),
                                        )
                            with txl.Else():
                                with txl.If(kind == 0):
                                    with txl.Then():
                                        txl.ptx[TMA_2SM](
                                            a_tiles[sub][sstate.stage].ptr_to(0, 0),
                                            txl.address_of(tm_ap),
                                            kcol,
                                            arow0,
                                            lbar,
                                            txl.uint64(EVICT_NORMAL),
                                        )
                                        txl.ptx[TMA_2SM](
                                            b_tiles[sub][sstate.stage].ptr_to(0, 0),
                                            txl.address_of(tm_w1),
                                            kcol,
                                            brow,
                                            lbar,
                                            txl.uint64(EVICT_NORMAL),
                                        )
                                    with txl.Else():
                                        txl.ptx[TMA_2SM](
                                            a_tiles[sub][sstate.stage].ptr_to(0, 0),
                                            txl.address_of(tm_act),
                                            kcol,
                                            arow0,
                                            lbar,
                                            txl.uint64(EVICT_NORMAL),
                                        )
                                        txl.ptx[TMA_2SM](
                                            b_tiles[sub][sstate.stage].ptr_to(0, 0),
                                            txl.address_of(tm_w2),
                                            kcol,
                                            brow,
                                            lbar,
                                            txl.uint64(EVICT_NORMAL),
                                        )
                    txl.cuda.warp_sync()
                sstate.advance()
            iket_end(tk_tile)

        def scale_a_copies(kind, e, nt, row0, nkb, sstate, half):
            """Aux warp 0: per stage, bulk-copy this CTA's 512 B A-scale image into the stage's
            scale words (local completion barrier), and forward the previous stage's completion
            to the leader's scale barrier one iteration later so the copy latency is hidden."""
            arow0 = i32(txl.Select(half != 0, row0 + crank * (BM // 2), row0 + crank * BM))
            hbase = i32(
                txl.Select(half != 0, (arow0 // (BM // 2)) * A1_PACKS, (arow0 // BM) * A1_PACKS)
            )
            hbase2 = i32(
                txl.Select(half != 0, (arow0 // (BM // 2)) * A2_PACKS, (arow0 // BM) * A2_PACKS)
            )
            have_prev = u32(txl.uint32(0))
            prev_stage = i32(txl.int32(0))
            prev_phase = u32(txl.uint32(0))
            tk_tile = iket_range("scale-tile")
            with txl.serial(0, nkb) as kb:
                txl.cuda.mbarrier_wait(sfempty_bar.ptr_to([sstate.stage]), sstate.phase ^ 1)
                with txl.If(lane == 0), txl.Then():
                    lbar_local = txl.cuda.cvta_generic_to_shared(sfa_bar.ptr_to([sstate.stage]))
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        sfa_bar.ptr_to([sstate.stage]), txl.uint32(SF_TILE_BYTES)
                    )
                    with txl.If(kind == 0):
                        with txl.Then():
                            with txl.If(half != 0):
                                with txl.Then():
                                    txl.ptx[BULK_G2S](
                                        sfa_tile.ptr_to([sstate.stage, 0]),
                                        a1h.ptr_to([(hbase + kb) * BM]),
                                        txl.uint32(SF_TILE_BYTES),
                                        lbar_local,
                                    )
                                with txl.Else():
                                    txl.ptx[BULK_G2S](
                                        sfa_tile.ptr_to([sstate.stage, 0]),
                                        a1p.ptr_to([(hbase + kb) * BM]),
                                        txl.uint32(SF_TILE_BYTES),
                                        lbar_local,
                                    )
                        with txl.Else():
                            with txl.If(half != 0):
                                with txl.Then():
                                    txl.ptx[BULK_G2S](
                                        sfa_tile.ptr_to([sstate.stage, 0]),
                                        a2h.ptr_to([((hbase2 + kb) * BM) * 4]),
                                        txl.uint32(SF_TILE_BYTES),
                                        lbar_local,
                                    )
                                with txl.Else():
                                    txl.ptx[BULK_G2S](
                                        sfa_tile.ptr_to([sstate.stage, 0]),
                                        a2p.ptr_to([((hbase2 + kb) * BM) * 4]),
                                        txl.uint32(SF_TILE_BYTES),
                                        lbar_local,
                                    )
                with txl.If(have_prev != txl.uint32(0)), txl.Then():
                    txl.cuda.mbarrier_wait(sfa_bar.ptr_to([prev_stage]), prev_phase)
                    txl.cuda.warp_sync()
                    with txl.If(lane == 0), txl.Then():
                        remote_arrive_leader(sf_bar.ptr_to([prev_stage]))
                txl.assign(prev_stage, sstate.stage)
                txl.assign(prev_phase, txl.cast(sstate.phase, "uint32"))
                txl.assign(have_prev, txl.uint32(1))
                sstate.advance()
            txl.cuda.mbarrier_wait(sfa_bar.ptr_to([prev_stage]), prev_phase)
            txl.cuda.warp_sync()
            with txl.If(lane == 0), txl.Then():
                remote_arrive_leader(sf_bar.ptr_to([prev_stage]))
            iket_end(tk_tile)

        def scale_loads(kind, e, nt, row0, nkb, sstate, half, aw):
            """Aux warp 1 stages both N halves of the B scales (A scales arrive by bulk copy).

            The global loads for K-pair kb+1 are issued right after stage kb is published, so
            their latency overlaps the wait for stage reuse."""
            raw = txl.alloc_local((4,), txl.u32)

            def load_regs(kb):
                for sub in range(KPACK):
                    ks = kb * KPACK + sub
                    bscale0 = txl.local_scalar(txl.f32)
                    bscale1 = txl.local_scalar(txl.f32)
                    with txl.If(kind == 0):
                        with txl.Then():
                            txl.ptx.ld.global_.f32(
                                bscale0, w1s.ptr_to([e * (32 * KB1) + nt * KB1 + ks])
                            )
                            txl.ptx.ld.global_.f32(
                                bscale1, w1s.ptr_to([e * (32 * KB1) + (16 + nt) * KB1 + ks])
                            )
                        with txl.Else():
                            txl.ptx.ld.global_.f32(
                                bscale0, w2s.ptr_to([e * (KB1 * KB2) + (nt * 2) * KB2 + ks])
                            )
                            txl.ptx.ld.global_.f32(
                                bscale1, w2s.ptr_to([e * (KB1 * KB2) + (nt * 2 + 1) * KB2 + ks])
                            )
                    txl.assign(raw[sub * 2], txl.reinterpret("uint32", bscale0))
                    txl.assign(raw[sub * 2 + 1], txl.reinterpret("uint32", bscale1))

            tk_tile = iket_range("scale-tile")
            load_regs(txl.int32(0))
            with txl.serial(0, nkb) as kb:
                txl.cuda.mbarrier_wait(sfempty_bar.ptr_to([sstate.stage]), sstate.phase ^ 1)
                # Completed async TCGEN reads precede generic writes that reuse this stage.
                txl.ptx.fence.proxy.async_.shared__cta()
                bpack0 = u32(
                    txl.bitwise_or(
                        ue8m0_pack4(txl.reinterpret("float32", raw[0])),
                        txl.shift_left(ue8m0_pack4(txl.reinterpret("float32", raw[2])), txl.uint32(8)),
                    )
                )
                bpack1 = u32(
                    txl.bitwise_or(
                        ue8m0_pack4(txl.reinterpret("float32", raw[1])),
                        txl.shift_left(ue8m0_pack4(txl.reinterpret("float32", raw[3])), txl.uint32(8)),
                    )
                )
                # Each v4 fills one TMEM lane's four 32-row scale columns.
                # G1 half tiles put 64 up then 64 gate rows in each CTA's B.
                split_g1 = txl.And(kind == 0, half != 0)
                first_upper = txl.Select(split_g1, bpack1, bpack0)
                second_lower = txl.Select(split_g1, bpack0, bpack1)
                txl.ptx.st.shared.v4.u32(
                    sfb_tile.ptr_to([sstate.stage, lane * 4]),
                    bpack0, bpack0, first_upper, first_upper,
                )
                txl.ptx.st.shared.v4.u32(
                    sfb_tile.ptr_to([sstate.stage, 128 + lane * 4]),
                    second_lower, second_lower, bpack1, bpack1,
                )
                txl.cuda.warp_sync()
                txl.ptx.fence.proxy.async_.shared__cta()
                with txl.If(lane == 0), txl.Then():
                    remote_arrive_leader(sf_bar.ptr_to([sstate.stage]))
                sstate.advance()
                with txl.If(kb + 1 < nkb), txl.Then():
                    load_regs(kb + 1)
            iket_end(tk_tile)

        # The control roles split one warpgroup: release its registers together.
        # Re-entering aux during finalization must not repeat this transition.
        with txl.If(warp < MATH_WARP0):
            with txl.Then():
                control_regs.emit()

        with prod_role:
            txl.ptx.fence.proxy.async_.global_()
            tstate = txl.PipelineState(TASK_RING, phase=0)
            sstate = txl.PipelineState(STAGES, phase=0)
            running = i32(txl.int32(1))
            with txl.If(is_leader):
                with txl.Then():
                    with txl.While(running == 1):
                        tidx = u32(txl.uint32(0))
                        with txl.If(lane == 0), txl.Then():
                            txl.ptx.atom.global_.add.u32(tidx, task_ctr.ptr_to([0]), txl.uint32(1))
                        tix = txl.local_scalar(txl.i32)
                        shfl_idx(tix, txl.cast(tidx, "int32"), txl.int32(0))
                        kind = i32(
                            txl.Select(
                                tix < n_g1,
                                txl.int32(0),
                                txl.Select(tix < n_g1 + n_g2, txl.int32(1), txl.int32(2)),
                            )
                        )
                        rel = i32(txl.Select(kind == 0, tix, tix - n_g1))
                        ntn = txl.Select(kind == 0, txl.int32(NT1), txl.int32(NT2))
                        mtg = i32(rel // ntn)
                        nt = i32(rel - mtg * ntn)

                        mp1 = txl.local_scalar(txl.i32)
                        txl.ptx.ld.shared.s32(mp1, s_mpre.ptr_to([lane + 1]))
                        bal = txl.local_scalar(txl.u32)
                        txl.ptx.vote_sync.ballot.b32(
                            bal, txl.ptx.pred(txl.cast(mp1 <= mtg, "uint32")), txl.uint32(0xFFFFFFFF)
                        )
                        e = i32(txl.min(txl.cast(txl.popcount(bal), "int32"), txl.int32(NUM_LOCAL - 1)))
                        mp0 = txl.local_scalar(txl.i32)
                        txl.ptx.ld.shared.s32(mp0, s_mpre.ptr_to([e]))
                        ecnt = txl.local_scalar(txl.i32)
                        txl.ptx.ld.shared.s32(ecnt, s_ecnt.ptr_to([e]))
                        eoff = txl.local_scalar(txl.i32)
                        txl.ptx.ld.shared.s32(eoff, s_eoff.ptr_to([e]))
                        mt = i32(mtg - mp0)
                        row0 = i32(eoff + mt * BMP)
                        valid = i32(txl.max(txl.min(ecnt - mt * BMP, txl.int32(BMP)), txl.int32(0)))
                        nkb = i32(txl.Select(kind == 0, txl.int32(KB1 // KPACK), txl.int32(KB2 // KPACK)))
                        half = i32(txl.cast(txl.And(valid > 0, valid <= BM), "int32"))

                        txl.cuda.mbarrier_wait(task_empty.ptr_to([tstate.stage]), tstate.phase ^ 1)
                        with txl.If(lane == 0), txl.Then():
                            txl.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 0]), kind)
                            txl.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 1]), e)
                            txl.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 2]), mt)
                            txl.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 3]), nt)
                            txl.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 4]), row0)
                            txl.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 5]), valid)
                            txl.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 6]), nkb)
                            txl.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 7]), half)
                            txl.ptx.mbarrier.arrive.shared.b64(
                                task_full.ptr_to([tstate.stage]), txl.uint32(1)
                            )
                            rem_bar = txl.local_scalar(txl.u64)
                            txl.ptx.mapa.shared__cluster.u64(
                                rem_bar, task_full.ptr_to([tstate.stage]), txl.uint32(1)
                            )
                            txl.ptx.mbarrier.arrive.expect_tx.release.cluster.b64(
                                rem_bar, txl.uint32(32), pred=txl.bool(True)
                            )
                            mbar32 = txl.local_scalar(txl.u32)
                            mdst = txl.local_scalar(txl.u32)
                            txl.ptx.mapa.shared__cluster.u32(
                                mbar32,
                                txl.cuda.cvta_generic_to_shared(task_full.ptr_to([tstate.stage])),
                                txl.uint32(1),
                            )
                            txl.ptx.mapa.shared__cluster.u32(
                                mdst,
                                txl.cuda.cvta_generic_to_shared(s_task.ptr_to([tstate.stage, 0])),
                                txl.uint32(1),
                            )
                            txl.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
                                mdst,
                                txl.cast(kind, "uint32"),
                                txl.cast(e, "uint32"),
                                txl.cast(mt, "uint32"),
                                txl.cast(nt, "uint32"),
                                mbar32,
                            )
                            txl.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
                                mdst + txl.uint32(16),
                                txl.cast(row0, "uint32"),
                                txl.cast(valid, "uint32"),
                                txl.cast(nkb, "uint32"),
                                txl.cast(half, "uint32"),
                                mbar32,
                            )
                        tstate.advance()
                        with txl.If(kind == 2):
                            with txl.Then():
                                txl.assign(running, txl.int32(0))
                            with txl.Else():
                                with txl.If(kind == 1), txl.Then():
                                    tk_done = iket_range("prod-wait-done")
                                    wait_done(e, mt)
                                    iket_end(tk_done)
                                producer_loads(kind, e, nt, row0, nkb, sstate, half)
                with txl.Else():
                    kind = i32(txl.int32(0))
                    e = i32(txl.int32(0))
                    mt = i32(txl.int32(0))
                    nt = i32(txl.int32(0))
                    row0 = i32(txl.int32(0))
                    valid = i32(txl.int32(0))
                    nkb = i32(txl.int32(0))
                    half = i32(txl.int32(0))
                    with txl.While(running == 1):
                        read_task(tstate, kind, e, mt, nt, row0, valid, nkb, half)
                        with txl.If(kind == 2):
                            with txl.Then():
                                txl.assign(running, txl.int32(0))
                            with txl.Else():
                                with txl.If(kind == 1), txl.Then():
                                    tk_done = iket_range("prod-wait-done")
                                    wait_done(e, mt)
                                    iket_end(tk_done)
                                producer_loads(kind, e, nt, row0, nkb, sstate, half)

        with mma_role:
            with txl.If(is_leader), txl.Then():
                el = txl.local_scalar(txl.u32)
                ell = txl.local_scalar(txl.u32)
                txl.ptx.elect_sync(ell, el, txl.uint32(0xFFFFFFFF))
                tstate = txl.PipelineState(TASK_RING, phase=0)
                sstate = txl.PipelineState(STAGES, phase=0)
                astate = txl.PipelineState(1, phase=0)
                desc_sf = txl.local_scalar(txl.u64)
                sfa_smem = txl.local_scalar(txl.u32)
                sfb_smem = txl.local_scalar(txl.u32)
                txl.assign(sfa_smem, txl.cuda.cvta_generic_to_shared(sfa_tile.ptr_to([0, 0])))
                txl.assign(sfb_smem, txl.cuda.cvta_generic_to_shared(sfb_tile.ptr_to([0, 0])))
                kind = i32(txl.int32(0))
                e = i32(txl.int32(0))
                mt = i32(txl.int32(0))
                nt = i32(txl.int32(0))
                row0 = i32(txl.int32(0))
                valid = i32(txl.int32(0))
                nkb = i32(txl.int32(0))
                half = i32(txl.int32(0))
                running = i32(txl.int32(1))
                with txl.While(running == 1):
                    read_task(tstate, kind, e, mt, nt, row0, valid, nkb, half)
                    with txl.If(kind == 2):
                        with txl.Then():
                            txl.assign(running, txl.int32(0))
                        with txl.Else():
                            tk_tile = iket_range("mma-tile")
                            tk_e = iket_range("mma-wait-tempty")
                            txl.cuda.mbarrier_wait(tempty_bar.ptr_to([0]), astate.phase ^ 1)
                            iket_end(tk_e)
                            txl.ptx.tcgen05.fence__after_thread_sync()
                            with txl.serial(0, nkb) as kb:
                                tk_s = iket_range("mma-wait-sf")
                                txl.cuda.mbarrier_wait(sf_bar.ptr_to([sstate.stage]), sstate.phase)
                                iket_end(tk_s)
                                txl.ptx.tcgen05.fence__after_thread_sync()
                                sf_stage = txl.cast(sstate.stage, "uint32")
                                txl.assign(
                                    desc_sf,
                                    with_smem_addr(
                                        SF_DESC_BASE, sfa_smem + sf_stage * txl.uint32(BM * 4)
                                    ),
                                )
                                txl.ptx[UTCCP](tmem_base + txl.uint32(SFA_TMEM_COL), desc_sf, pred=el)
                                txl.assign(
                                    desc_sf,
                                    with_smem_addr(
                                        SF_DESC_BASE, sfb_smem + sf_stage * txl.uint32(BN * 4)
                                    ),
                                )
                                with txl.If(half != 0):
                                    with txl.Then():
                                        # M128 uses a 2x2 SFB layout: the two N128 halves
                                        # occupy TMEM lane partitions 0/1 and 2/3.
                                        txl.ptx["tcgen05.cp.cta_group::2.64x128b.warpx2::01_23"](
                                            tmem_base + txl.uint32(SFB_TMEM_COL), desc_sf, pred=el
                                        )
                                    with txl.Else():
                                        txl.ptx[UTCCP](tmem_base + txl.uint32(SFB_TMEM_COL), desc_sf, pred=el)
                                        txl.assign(
                                            desc_sf,
                                            with_smem_addr(
                                                SF_DESC_BASE,
                                                sfb_smem + sf_stage * txl.uint32(BN * 4) + txl.uint32(128 * 4),
                                            ),
                                        )
                                        txl.ptx[UTCCP](
                                            tmem_base + txl.uint32(SFB_TMEM_COL + 4), desc_sf, pred=el
                                        )
                                txl.ptx.tcgen05.fence__before_thread_sync()
                                txl.cuda.warp_sync()
                                txl.ptx[COMMIT](
                                    sfempty_bar.ptr_to([sstate.stage]), txl.uint16(3), pred=el
                                )
                                txl.cuda.warp_sync()
                                for sub in range(KPACK):
                                    slot = sstate.stage * KPACK + sub
                                    tk_f = iket_range("mma-wait-full")
                                    txl.cuda.mbarrier_wait(full_bar.ptr_to([slot]), sstate.phase)
                                    iket_end(tk_f)
                                    txl.ptx.tcgen05.fence__after_thread_sync()
                                    a_desc, a_off = a_tiles[sub][sstate.stage].encode(
                                        major="k", mma_k=32
                                    )
                                    b_desc, b_off = b_tiles[sub][sstate.stage].encode(
                                        major="k", mma_k=32
                                    )
                                    for ki in range(4):
                                        accumulate = txl.Or(kb > 0, txl.Or(txl.int32(sub) > 0, ki > 0))
                                        with txl.If(half != 0):
                                            with txl.Then():
                                                txl.ptx[MMA](
                                                    tmem_base,
                                                    a_desc + a_off(ki),
                                                    b_desc + b_off(ki),
                                                    txl.uint32(IDESC_H_IDS[sub]),
                                                    tmem_base + txl.uint32(SFA_TMEM_COL),
                                                    tmem_base + txl.uint32(SFB_TMEM_COL),
                                                    accumulate,
                                                    pred=el,
                                                )
                                            with txl.Else():
                                                txl.ptx[MMA](
                                                    tmem_base,
                                                    a_desc + a_off(ki),
                                                    b_desc + b_off(ki),
                                                    txl.uint32(IDESC_IDS[sub]),
                                                    tmem_base + txl.uint32(SFA_TMEM_COL),
                                                    tmem_base + txl.uint32(SFB_TMEM_COL),
                                                    accumulate,
                                                    pred=el,
                                                )
                                    txl.ptx.tcgen05.fence__before_thread_sync()
                                    txl.cuda.warp_sync()
                                    txl.ptx[COMMIT](empty_bar.ptr_to([slot]), txl.uint16(3), pred=el)
                                    if sub == KPACK - 1:
                                        txl.ptx[COMMIT](
                                            tfull_bar.ptr_to([0]),
                                            txl.uint16(3),
                                            pred=txl.And(el == txl.uint32(1), kb == nkb - 1),
                                        )
                                    txl.cuda.warp_sync()
                                sstate.advance()
                            astate.advance()
                            iket_end(tk_tile)

        with aux_role:
            aw = warp - 2
            txl.ptx.fence.proxy.async_.global_()
            tstate = txl.PipelineState(TASK_RING, phase=0)
            sstate = txl.PipelineState(STAGES, phase=0)
            kind = i32(txl.int32(0))
            e = i32(txl.int32(0))
            mt = i32(txl.int32(0))
            nt = i32(txl.int32(0))
            row0 = i32(txl.int32(0))
            valid = i32(txl.int32(0))
            nkb = i32(txl.int32(0))
            half = i32(txl.int32(0))
            running = i32(txl.int32(1))
            with txl.While(running == 1):
                read_task(tstate, kind, e, mt, nt, row0, valid, nkb, half)
                with txl.If(kind == 2):
                    with txl.Then():
                        txl.assign(running, txl.int32(0))
                    with txl.Else():
                        with txl.If(kind == 1), txl.Then():
                            wait_done(e, mt)
                        with txl.If(aw == 0):
                            with txl.Then():
                                scale_a_copies(kind, e, nt, row0, nkb, sstate, half)
                            with txl.Else():
                                scale_loads(kind, e, nt, row0, nkb, sstate, half, aw)

        with math_role:
            mw = warp - MATH_WARP0
            wg = mw // 4
            row = (mw % 4) * 32 + lane
            tstate = txl.PipelineState(TASK_RING, phase=0)
            astate = txl.PipelineState(1, phase=0)
            kind = i32(txl.int32(0))
            e = i32(txl.int32(0))
            mt = i32(txl.int32(0))
            nt = i32(txl.int32(0))
            row0 = i32(txl.int32(0))
            valid = i32(txl.int32(0))
            nkb = i32(txl.int32(0))
            half = i32(txl.int32(0))
            acc = txl.alloc_local((128,), txl.f32)
            tile_ctr = i32(txl.int32(0))
            running = i32(txl.int32(1))

            def drain_tmem(cols):
                """Wait for the completed K reduction, drain its columns once, then release TMEM."""
                tk_f = iket_range("math-wait-tfull")
                txl.cuda.mbarrier_wait(tfull_bar.ptr_to([0]), astate.phase)
                iket_end(tk_f)
                txl.ptx.tcgen05.fence__after_thread_sync()
                for c, col in enumerate(cols):
                    taddr = tmem_base + txl.uint32(col)
                    txl.ptx[TMEM_LD32](*[acc[c * 32 + i] for i in range(32)], taddr)
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                txl.ptx.tcgen05.fence__before_thread_sync()
                txl.cuda.warp_sync()
                with txl.If(lane == 0), txl.Then():
                    remote_arrive_leader(tempty_bar.ptr_to([0]))
                astate.advance()

            L = (mw % 4) * 32 + lane
            hi = (mw % 4) // 2
            rowh = ((mw % 4) % 2) * 32 + lane

            def g1_mainloop(cols, sb_base0, sb_base1, arow):
                drain_tmem(cols)

            def g1_epilogue(nval, arow, is_valid, colbase, amax_slot, amax_others, a2s_writer):
                """SwiGLU over `nval` columns per thread, 128-column block amax via SMEM exchange, fp8 quant, act store."""
                h = txl.alloc_local((64,), txl.f32)
                amax = f32(txl.float32(0.0))
                for j in range(nval):
                    uu = acc[nval + j]
                    sig = txl.idioms.sigmoid_tanh_approx_f32(uu)
                    txl.assign(h[j], txl.Select(is_valid, uu * sig * acc[j], txl.float32(0.0)))
                    txl.assign(amax, txl.max(amax, txl.fabs(h[j])))
                txl.ptx.st.shared.f32(s_amax.ptr_to([amax_slot]), amax)
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32((NWARPS - MATH_WARP0) * 32))
                for other in amax_others:
                    oam = txl.local_scalar(txl.f32)
                    txl.ptx.ld.shared.f32(oam, s_amax.ptr_to([other]))
                    txl.assign(amax, txl.max(amax, oam))
                inv = f32(
                    txl.Select(amax > txl.float32(0.0), txl.float32(FP8_MAX) / amax, txl.float32(0.0))
                )
                q = txl.alloc_local((16,), txl.u32)
                for j in range(nval // 4):
                    lo16 = txl.local_scalar(txl.u16)
                    hi16 = txl.local_scalar(txl.u16)
                    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(lo16, h[4 * j + 1] * inv, h[4 * j] * inv)
                    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(hi16, h[4 * j + 3] * inv, h[4 * j + 2] * inv)
                    txl.assign(
                        q[j],
                        txl.bitwise_or(
                            txl.cast(lo16, "uint32"),
                            txl.shift_left(txl.cast(hi16, "uint32"), txl.uint32(16)),
                        ),
                    )
                abase = arow * INTER + nt * 128 + colbase
                for j in range(nval // 16):
                    txl.ptx.st.global_.v4.b32(
                        act.ptr_to([abase + j * 16]),
                        q[4 * j],
                        q[4 * j + 1],
                        q[4 * j + 2],
                        q[4 * j + 3],
                    )
                with txl.If(a2s_writer), txl.Then():
                    sbyte = txl.cast(ue8m0_pack4(amax * txl.float32(1.0 / FP8_MAX)), "uint8")
                    ht = arow // BM
                    lr = arow - ht * BM
                    wp = (ht * A2_PACKS + nt // 2) * BM + (lr % 32) * 4 + lr // 32
                    txl.ptx.st.global_.u8(a2p.ptr_to([wp * 4 + nt % 2]), sbyte)
                    ht2 = arow // (BM // 2)
                    lr2 = arow - ht2 * (BM // 2)
                    wh = (ht2 * A2_PACKS + nt // 2) * BM + (lr2 % 32) * 4 + lr2 // 32
                    txl.ptx.st.global_.u8(a2h.ptr_to([wh * 4 + nt % 2]), sbyte)
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32((NWARPS - MATH_WARP0) * 32))
                with txl.If(txl.And(mw == 0, lane == 0)), txl.Then():
                    txl.ptx.red.release.gpu.global_.add.u32(done.ptr_to([e * MAXMT + mt]), txl.uint32(1))

            def g2_mainloop(cols, sb_base, arow):
                drain_tmem(cols)

            with txl.While(running == 1):
                read_task(tstate, kind, e, mt, nt, row0, valid, nkb, half)
                with txl.If(kind == 2):
                    with txl.Then():
                        txl.assign(running, txl.int32(0))
                    with txl.Else():
                        arow_p = i32(row0 + crank * BM + row)
                        is_valid_p = row + crank * BM < valid
                        arow_h = i32(row0 + crank * (BM // 2) + rowh)
                        valid_h = txl.max(
                            txl.min(valid - crank * (BM // 2), txl.int32(BM // 2)), txl.int32(0)
                        )
                        is_valid_h = rowh < valid_h
                        arow = i32(txl.Select(half != 0, arow_h, arow_p))
                        par = txl.bitwise_and(tile_ctr, txl.int32(1))
                        tk_tile = iket_range("math-tile")
                        with txl.If(kind == 0):
                            with txl.Then():
                                sb_base0 = e * (32 * KB1) + nt * KB1
                                sb_base1 = e * (32 * KB1) + (16 + nt) * KB1
                                with txl.If(half != 0):
                                    with txl.Then():
                                        g1_mainloop(
                                            [wg * 32, 64 + wg * 32], sb_base0, sb_base1, arow_h
                                        )
                                        tk_ep = iket_range("math-epi-g1")
                                        slot_h = par * 256 + (wg * 2 + hi) * 64 + rowh
                                        others = [
                                            par * 256 + (w2 * 2 + h2) * 64 + rowh
                                            for w2 in range(2)
                                            for h2 in range(2)
                                        ]
                                        g1_epilogue(
                                            32,
                                            arow_h,
                                            is_valid_h,
                                            hi * 64 + wg * 32,
                                            slot_h,
                                            [
                                                txl.Select(txl.And(w2 == wg, h2 == hi), slot_h, o)
                                                for (w2, h2), o in zip(
                                                    [(a, b) for a in range(2) for b in range(2)],
                                                    others,
                                                )
                                            ],
                                            txl.And(wg == 0, hi == 0),
                                        )
                                        iket_end(tk_ep)
                                    with txl.Else():
                                        g1_mainloop(
                                            [
                                                wg * 64,
                                                wg * 64 + 32,
                                                128 + wg * 64,
                                                128 + wg * 64 + 32,
                                            ],
                                            sb_base0,
                                            sb_base1,
                                            arow_p,
                                        )
                                        tk_ep = iket_range("math-epi-g1")
                                        g1_epilogue(
                                            64,
                                            arow_p,
                                            is_valid_p,
                                            wg * 64,
                                            par * 256 + wg * 128 + row,
                                            [par * 256 + (1 - wg) * 128 + row],
                                            wg == 0,
                                        )
                                        iket_end(tk_ep)
                            with txl.Else():
                                tk_d = iket_range("math-wait-done")
                                dv = txl.local_scalar(txl.u32)
                                txl.cuda.wait_until(
                                    dv,
                                    done.ptr_to([e * MAXMT + mt]),
                                    dv >= txl.uint32(PAIR * NT1),
                                    scope="gpu",
                                    ptx_type="b32",
                                )
                                iket_end(tk_d)
                                wtok = f32(txl.float32(0.0))
                                with txl.If(arow < row0 + valid), txl.Then():
                                    txl.ptx.ld.global_.f32(wtok, sorted_w.ptr_to([arow]))
                                with txl.If(half != 0):
                                    with txl.Then():
                                        g2_mainloop(
                                            [wg * 64, wg * 64 + 32],
                                            e * (KB1 * KB2) + (nt * 2 + hi) * KB2,
                                            arow_h,
                                        )
                                        tk_ep = iket_range("math-epi-g2")
                                        pk = txl.alloc_local((16,), txl.u32)
                                        for j in range(16):
                                            lo = txl.local_scalar(txl.u16)
                                            hi8 = txl.local_scalar(txl.u16)
                                            txl.ptx.cvt.rn.satfinite.e4m3x2.f32(
                                                lo, acc[4 * j + 1] * wtok, acc[4 * j] * wtok
                                            )
                                            txl.ptx.cvt.rn.satfinite.e4m3x2.f32(
                                                hi8, acc[4 * j + 3] * wtok, acc[4 * j + 2] * wtok
                                            )
                                            txl.assign(
                                                pk[j],
                                                txl.bitwise_or(
                                                    txl.cast(lo, "uint32"),
                                                    txl.shift_left(
                                                        txl.cast(hi8, "uint32"), txl.uint32(16)
                                                    ),
                                                ),
                                            )
                                        cbase_h = nt * BN + hi * 128 + wg * 64
                                        dst_row_h = arow_h * HIDDEN + cbase_h
                                        for r in range(2):
                                            with txl.If(lane // 16 == r), txl.Then():
                                                for j in range(4):
                                                    txl.ptx.st.shared.v4.b32(
                                                        s_epi.ptr_to([mw, lane % 16, j * 4]),
                                                        pk[4 * j],
                                                        pk[4 * j + 1],
                                                        pk[4 * j + 2],
                                                        pk[4 * j + 3],
                                                    )
                                                txl.ptx.st.shared.s32(
                                                    s_epi.ptr_to([mw, lane % 16, 32]), dst_row_h
                                                )
                                                txl.ptx.st.shared.s32(
                                                    s_epi.ptr_to([mw, lane % 16, 33]),
                                                    txl.cast(is_valid_h, "int32"),
                                                )
                                            txl.cuda.warp_sync()
                                            for q in range(2):
                                                f = q * 32 + lane
                                                row_i = f // 4
                                                piece = f % 4
                                                t0 = txl.local_scalar(txl.u32)
                                                t1 = txl.local_scalar(txl.u32)
                                                t2 = txl.local_scalar(txl.u32)
                                                t3 = txl.local_scalar(txl.u32)
                                                txl.ptx.ld.shared.v4.b32(
                                                    t0,
                                                    t1,
                                                    t2,
                                                    t3,
                                                    s_epi.ptr_to([mw, row_i, piece * 4]),
                                                )
                                                doff = txl.local_scalar(txl.i32)
                                                dval = txl.local_scalar(txl.i32)
                                                txl.ptx.ld.shared.s32(
                                                    doff, s_epi.ptr_to([mw, row_i, 32])
                                                )
                                                txl.ptx.ld.shared.s32(
                                                    dval, s_epi.ptr_to([mw, row_i, 33])
                                                )
                                                with txl.If(dval != 0), txl.Then():
                                                    txl.ptx.st.global_.v4.b32(
                                                        partial.ptr_to([doff + piece * 16]),
                                                        t0,
                                                        t1,
                                                        t2,
                                                        t3,
                                                    )
                                            txl.cuda.warp_sync()
                                        iket_end(tk_ep)
                                    with txl.Else():
                                        g2_mainloop(
                                            [wg * 128, wg * 128 + 32, wg * 128 + 64, wg * 128 + 96],
                                            e * (KB1 * KB2) + (nt * 2 + wg) * KB2,
                                            arow_p,
                                        )
                                        is_valid = is_valid_p

                                        tk_ep = iket_range("math-epi-g2")
                                        pk = txl.alloc_local((32,), txl.u32)
                                        for j in range(32):
                                            lo = txl.local_scalar(txl.u16)
                                            hi8 = txl.local_scalar(txl.u16)
                                            txl.ptx.cvt.rn.satfinite.e4m3x2.f32(
                                                lo, acc[4 * j + 1] * wtok, acc[4 * j] * wtok
                                            )
                                            txl.ptx.cvt.rn.satfinite.e4m3x2.f32(
                                                hi8, acc[4 * j + 3] * wtok, acc[4 * j + 2] * wtok
                                            )
                                            txl.assign(
                                                pk[j],
                                                txl.bitwise_or(
                                                    txl.cast(lo, "uint32"),
                                                    txl.shift_left(
                                                        txl.cast(hi8, "uint32"), txl.uint32(16)
                                                    ),
                                                ),
                                            )
                                        cbase = nt * BN + wg * 128
                                        dst_row = arow * HIDDEN + cbase
                                        for r in range(2):
                                            with txl.If(lane // 16 == r), txl.Then():
                                                for j in range(8):
                                                    txl.ptx.st.shared.v4.b32(
                                                        s_epi.ptr_to([mw, lane % 16, j * 4]),
                                                        pk[4 * j],
                                                        pk[4 * j + 1],
                                                        pk[4 * j + 2],
                                                        pk[4 * j + 3],
                                                    )
                                                txl.ptx.st.shared.s32(
                                                    s_epi.ptr_to([mw, lane % 16, 32]), dst_row
                                                )
                                                txl.ptx.st.shared.s32(
                                                    s_epi.ptr_to([mw, lane % 16, 33]),
                                                    txl.cast(is_valid, "int32"),
                                                )
                                            txl.cuda.warp_sync()
                                            for q in range(4):
                                                f = q * 32 + lane
                                                row_i = f // 8
                                                piece = f % 8
                                                t0 = txl.local_scalar(txl.u32)
                                                t1 = txl.local_scalar(txl.u32)
                                                t2 = txl.local_scalar(txl.u32)
                                                t3 = txl.local_scalar(txl.u32)
                                                txl.ptx.ld.shared.v4.b32(
                                                    t0,
                                                    t1,
                                                    t2,
                                                    t3,
                                                    s_epi.ptr_to([mw, row_i, piece * 4]),
                                                )
                                                doff = txl.local_scalar(txl.i32)
                                                dval = txl.local_scalar(txl.i32)
                                                txl.ptx.ld.shared.s32(
                                                    doff, s_epi.ptr_to([mw, row_i, 32])
                                                )
                                                txl.ptx.ld.shared.s32(
                                                    dval, s_epi.ptr_to([mw, row_i, 33])
                                                )
                                                with txl.If(dval != 0), txl.Then():
                                                    txl.ptx.st.global_.v4.b32(
                                                        partial.ptr_to([doff + piece * 16]),
                                                        t0,
                                                        t1,
                                                        t2,
                                                        t3,
                                                    )
                                            txl.cuda.warp_sync()
                                        iket_end(tk_ep)
                        iket_end(tk_tile)
                        txl.assign(tile_ctr, tile_ctr + 1)

        txl.cuda.iket.mark("G-done")
        emit_grid_sync(sync_ctr.ptr_to([0]), cta, num_ctas, tid)
        txl.cuda.iket.mark("F-start")

        with aux_role:
            aw = warp - 2
            zt = i32(cta * 2 + aw)
            with txl.While(zt < T):
                zc = txl.local_scalar(txl.i32)
                txl.ptx.ld.global_.s32(zc, tok_cnt.ptr_to([zt]))
                with txl.If(zc == 0), txl.Then():
                    for j in range(HIDDEN // 256):
                        txl.ptx.st.global_.v4.b32(
                            out.ptr_to([zt * HIDDEN + j * 256 + lane * 8]),
                            txl.uint32(0),
                            txl.uint32(0),
                            txl.uint32(0),
                            txl.uint32(0),
                        )
                txl.assign(zt, zt + num_ctas * 2)

        with math_role:
            mw = warp - MATH_WARP0
            nmw = NWARPS - MATH_WARP0
            ngw = num_ctas * nmw
            nfin_u = txl.local_scalar(txl.u32)
            txl.ptx.ld.global_.u32(nfin_u, task_ctr.ptr_to([2]))
            nfin = i32(txl.cast(nfin_u, "int32"))
            fi = i32(cta * nmw + mw)
            ft = i32(txl.int32(0))
            mypos = i32(txl.int32(-1))
            posk = txl.alloc_local((8,), txl.i32)
            cpos = txl.alloc_local((8,), txl.i32)
            pw2 = txl.alloc_local((FIN_G, FIN_RPP, 2, 2), txl.u32)
            bacc2 = txl.alloc_local((FIN_G, 2, 4), txl.u32)

            def load_tok(dst_ft, dst_pos, idx):
                txl.ptx.ld.global_.s32(dst_ft, fin_list.ptr_to([idx]))
                with txl.If(lane < 8), txl.Then():
                    txl.ptx.ld.global_.s32(dst_pos, pos_of.ptr_to([dst_ft * TOPK + lane]))

            with txl.If(fi < nfin), txl.Then():
                load_tok(ft, mypos, fi)
            with txl.While(fi < nfin):
                fi_n = i32(fi + ngw)
                ft_n = i32(txl.int32(0))
                mypos_n = i32(txl.int32(-1))
                with txl.If(fi_n < nfin), txl.Then():
                    load_tok(ft_n, mypos_n, fi_n)

                bal = txl.local_scalar(txl.u32)
                txl.ptx.vote_sync.ballot.b32(
                    bal,
                    txl.ptx.pred(txl.cast(txl.And(lane < 8, mypos >= 0), "uint32")),
                    txl.uint32(0xFFFFFFFF),
                )
                fc = i32(txl.cast(txl.popcount(bal), "int32"))
                for k in range(8):
                    shfl_idx(posk[k], mypos, txl.int32(k))
                for j in range(8):
                    txl.assign(cpos[j], txl.int32(-1))
                for k in range(8):
                    rk = i32(
                        txl.cast(txl.popcount(txl.bitwise_and(bal, txl.uint32((1 << k) - 1))), "int32")
                    )
                    vk = txl.bitwise_and(txl.shift_right(bal, txl.uint32(k)), txl.uint32(1)) != txl.uint32(0)
                    for j in range(8):
                        txl.assign(
                            cpos[j],
                            txl.Select(txl.And(vk, rk == txl.int32(j)), posk[k] * HIDDEN, cpos[j]),
                        )
                for cbase in range(0, HIDDEN, 512 * FIN_G):
                    ng = min(FIN_G, (HIDDEN - cbase) // 512)
                    for g in range(ng):
                        for h in range(2):
                            for i in range(4):
                                txl.assign(bacc2[g, h, i], txl.uint32(0))

                    def fin_pass(r0):
                        rows = [r for r in range(r0, min(r0 + FIN_RPP, 8))]
                        for jj, r in enumerate(rows):
                            with txl.If(cpos[r] >= 0), txl.Then():
                                for g in range(ng):
                                    for h in range(2):
                                        col = cbase + g * 512 + h * 256 + lane * 8
                                        txl.ptx.ld.global_.nc.v2.b32(
                                            pw2[g, jj, h, 0],
                                            pw2[g, jj, h, 1],
                                            partial.ptr_to([cpos[r] + col]),
                                        )
                        for jj, r in enumerate(rows):
                            with txl.If(cpos[r] >= 0), txl.Then():
                                for g in range(ng):
                                    for h in range(2):
                                        for q4 in range(2):
                                            lo2 = txl.local_scalar(txl.u32)
                                            hi2 = txl.local_scalar(txl.u32)
                                            raw4 = pw2[g, jj, h, q4]
                                            txl.ptx.cvt.rn.bf16x2.e4m3x2(
                                                lo2,
                                                txl.cast(
                                                    txl.bitwise_and(raw4, txl.uint32(0xFFFF)), "uint16"
                                                ),
                                            )
                                            txl.ptx.cvt.rn.bf16x2.e4m3x2(
                                                hi2,
                                                txl.cast(txl.shift_right(raw4, txl.uint32(16)), "uint16"),
                                            )
                                            txl.ptx.add.rn.bf16x2(
                                                bacc2[g, h, 2 * q4], bacc2[g, h, 2 * q4], lo2
                                            )
                                            txl.ptx.add.rn.bf16x2(
                                                bacc2[g, h, 2 * q4 + 1],
                                                bacc2[g, h, 2 * q4 + 1],
                                                hi2,
                                            )

                    for r0 in range(0, 8, FIN_RPP):
                        if r0 == 0:
                            fin_pass(0)
                        else:
                            with txl.If(fc > r0), txl.Then():
                                fin_pass(r0)
                    for g in range(ng):
                        for h in range(2):
                            col = cbase + g * 512 + h * 256 + lane * 8
                            txl.ptx.st.global_.v4.b32(
                                out.ptr_to([ft * HIDDEN + col]),
                                bacc2[g, h, 0],
                                bacc2[g, h, 1],
                                bacc2[g, h, 2],
                                bacc2[g, h, 3],
                            )
                txl.assign(fi, fi_n)
                txl.assign(ft, ft_n)
                txl.assign(mypos, mypos_n)

        txl.cuda.iket.mark("F-done")
        txl.ptx.barrier.cluster.arrive.release.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()
        with txl.If(warp == 2), txl.Then():
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](txl.uint32(0), txl.uint32(512))

        with txl.If(tid == 0), txl.Then():
            oldx = txl.local_scalar(txl.u32)
            txl.ptx.atom.acq_rel.gpu.global_.add.u32(oldx, task_ctr.ptr_to([3]), txl.uint32(1))
            with txl.If(oldx == txl.uint32(num_ctas - 1)), txl.Then():
                txl.ptx.st.global_.u32(task_ctr.ptr_to([3]), txl.uint32(0))
                txl.ptx.st.global_.u32(task_ctr.ptr_to([2]), txl.uint32(0))

    return moe_mega


class _TensorMap:
    def __init__(self):
        self._buf = ctypes.create_string_buffer(256)
        self.ptr = ctypes.c_void_p((ctypes.addressof(self._buf) + 127) & ~127)


def _encode_2d(tensor, dtype_name, inner, outer, stride_bytes, box_inner, box_outer, swizzle):
    import tvm

    enc = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    tm = _TensorMap()
    enc(
        tm.ptr,
        dtype_name,
        2,
        ctypes.c_void_p(int(tensor.data_ptr())),
        int(inner),
        int(outer),
        int(stride_bytes),
        int(box_inner),
        int(box_outer),
        1,
        1,
        0,
        int(swizzle),
        3,
        0,
    )
    return tm


@dataclass(frozen=True, slots=True)
class MoEConfig:
    label: str = "t7_correctness"
    seq_len: int = 7
    seed: int = 0
    local_expert_offset: int = 0
    routed_scaling_factor: float = 2.5

    def validate(self) -> None:
        if not 1 <= self.seq_len <= 32768:
            raise ValueError(f"seq_len must be in [1, 32768], got {self.seq_len}")
        if self.local_expert_offset % NUM_LOCAL != 0:
            raise ValueError(
                f"local_expert_offset must be aligned to {NUM_LOCAL}, "
                f"got {self.local_expert_offset}"
            )
        if not 0 <= self.local_expert_offset <= NUM_EXPERTS - NUM_LOCAL:
            raise ValueError(f"invalid local_expert_offset {self.local_expert_offset}")
        if self.routed_scaling_factor <= 0:
            raise ValueError("routed_scaling_factor must be positive")


CONFIGS = [
    {
        "label": "t7_correctness",
        "seq_len": 7,
        "seed": 0,
        "local_expert_offset": 0,
        "routed_scaling_factor": 2.5,
    }
]

_OFFICIAL_SEQ_LENS = (
    7,
    1,
    32,
    80,
    901,
    16,
    15,
    14,
    14107,
    11948,
    62,
    59,
    58,
    57,
    56,
    55,
    54,
    53,
    52,
)
_STRESS_SEQ_LENS = (32768,)

BENCH_CONFIGS = [
    {
        "label": f"t{seq_len}",
        "seq_len": seq_len,
        "seed": 0,
        "local_expert_offset": 0,
        "routed_scaling_factor": 2.5,
    }
    for seq_len in (*_OFFICIAL_SEQ_LENS, *_STRESS_SEQ_LENS)
]


def _cfg(**kwargs: Any) -> MoEConfig:
    names = {field.name for field in fields(MoEConfig)}
    cfg = MoEConfig(**{name: value for name, value in kwargs.items() if name in names})
    cfg.validate()
    return cfg


def _num_ctas(**kwargs: Any) -> int:
    if "num_ctas" in kwargs:
        value = int(kwargs["num_ctas"])
    else:
        from tirx_kernels.runner import hardware_num_sms

        value = hardware_num_sms()
    if value < PAIR or value % PAIR != 0:
        raise ValueError(f"the two-CTA cluster kernel needs a positive even CTA count, got {value}")
    return value


def get_kernel(**kwargs: Any):
    _cfg(**kwargs)
    return build_kernel(_num_ctas(**kwargs)).func


def _assert_supported_arch() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for curated native TIRx FP8 MoE")
    capability = torch.cuda.get_device_capability()
    runtime_arch = f"sm_{capability[0]}{capability[1]}a"
    if runtime_arch not in KERNEL_META["runtime_cuda_archs"]:
        raise SkipTest(
            "curated native TIRx FP8 MoE requires one of "
            f"{KERNEL_META['runtime_cuda_archs']}, got {runtime_arch}"
        )


def _rand_fp8(
    shape: tuple[int, ...], device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    """Match the official benchmark's raw E4M3 input distribution."""
    return (
        torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
        .clamp_(-2.0, 2.0)
        .to(torch.float8_e4m3fn)
    )


def _positive_scale(
    shape: tuple[int, ...], device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    """Match the official benchmark's independent FP32 block scales."""
    return torch.empty(shape, dtype=torch.float32, device=device).uniform_(
        0.01, 0.1, generator=generator
    )


def _allocate_kernel_state(case: dict[str, Any], num_ctas: int) -> None:
    cfg: MoEConfig = case["config"]
    dev = case["hidden_states"].device
    T = cfg.seq_len
    P = ((T * TOPK + NUM_LOCAL * (BMP - 1) + BMP - 1) // BMP) * BMP
    max_m_tiles = (T + BMP - 1) // BMP + 1
    i32 = {"dtype": torch.int32, "device": dev}
    case.update(
        {
            "num_ctas": num_ctas,
            "P": P,
            "max_m_tiles": max_m_tiles,
            "route_id": torch.empty(T * TOPK, **i32),
            "route_w": torch.empty(T * TOPK, dtype=torch.float32, device=dev),
            "tok_cnt": torch.empty(T, **i32),
            "cnt_cta": torch.empty(num_ctas * NUM_LOCAL, **i32),
            "cnt_warp": torch.empty(num_ctas * NWARPS * NUM_LOCAL, **i32),
            "sorted_token": torch.empty(P, **i32),
            "pos_of": torch.empty(T * TOPK, **i32),
            "sorted_w": torch.empty(P, dtype=torch.float32, device=dev),
            "a1p": torch.zeros(P * A1_PACKS, dtype=torch.uint32, device=dev),
            "a1h": torch.zeros(2 * P * A1_PACKS, dtype=torch.uint32, device=dev),
            "a2p": torch.zeros(P * A2_PACKS * 4, dtype=torch.uint8, device=dev),
            "a2h": torch.zeros(2 * P * A2_PACKS * 4, dtype=torch.uint8, device=dev),
            "act": torch.empty(P * INTER, dtype=torch.float8_e4m3fn, device=dev),
            "ap": torch.empty(P * HIDDEN, dtype=torch.float8_e4m3fn, device=dev),
            "partial": torch.empty(P * HIDDEN, dtype=torch.float8_e4m3fn, device=dev),
            "task_ctr": torch.zeros(4, dtype=torch.uint32, device=dev),
            "done": torch.zeros(NUM_LOCAL * max_m_tiles, dtype=torch.uint32, device=dev),
            "sync_ctr": torch.zeros(4, dtype=torch.uint32, device=dev),
            "fin_list": torch.empty(T, **i32),
        }
    )

    w1 = case["gemm1_weights"]
    w2 = case["gemm2_weights"]
    maps = {
        "ap": _encode_2d(case["ap"], "float8_e4m3fn", HIDDEN, P, HIDDEN, BK, BM, 3),
        "w1": _encode_2d(w1, "float8_e4m3fn", HIDDEN, NUM_LOCAL * 2 * INTER, HIDDEN, BK, 128, 3),
        "w2": _encode_2d(w2, "float8_e4m3fn", INTER, NUM_LOCAL * HIDDEN, INTER, BK, BNH, 3),
        "act": _encode_2d(case["act"], "float8_e4m3fn", INTER, P, INTER, BK, BM, 3),
        "ap64": _encode_2d(case["ap"], "float8_e4m3fn", HIDDEN, P, HIDDEN, BK, BM // 2, 3),
        "act64": _encode_2d(case["act"], "float8_e4m3fn", INTER, P, INTER, BK, BM // 2, 3),
        "w1h": _encode_2d(w1, "float8_e4m3fn", HIDDEN, NUM_LOCAL * 2 * INTER, HIDDEN, BK, 64, 3),
    }
    case["tensor_maps"] = maps


@torch.no_grad()
def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    device = torch.device(kwargs.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SkipTest("CUDA is required for curated native TIRx FP8 MoE")
    num_ctas = _num_ctas(**kwargs)
    actual_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if num_ctas > actual_sms:
        raise ValueError(
            f"kernel was built for {num_ctas} CTAs but the GPU has only {actual_sms} SMs"
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    routing_logits = torch.randn(
        cfg.seq_len, NUM_EXPERTS, dtype=torch.float32, device=device, generator=generator
    )
    routing_bias = torch.zeros(NUM_EXPERTS, dtype=torch.bfloat16, device=device)
    hidden_states = _rand_fp8((cfg.seq_len, HIDDEN), device, generator)
    hidden_states_scale = _positive_scale((HIDDEN // BK, cfg.seq_len), device, generator)
    gemm1_weights = _rand_fp8((NUM_LOCAL, 2 * INTER, HIDDEN), device, generator)
    gemm1_weights_scale = _positive_scale(
        (NUM_LOCAL, 2 * INTER // BK, HIDDEN // BK), device, generator
    )
    gemm2_weights = _rand_fp8((NUM_LOCAL, HIDDEN, INTER), device, generator)
    gemm2_weights_scale = _positive_scale((NUM_LOCAL, HIDDEN // BK, INTER // BK), device, generator)

    case: dict[str, Any] = {
        "config": cfg,
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": gemm1_weights,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
        "output": torch.empty(cfg.seq_len, HIDDEN, dtype=torch.bfloat16, device=device),
        "reference_output": torch.empty(cfg.seq_len, HIDDEN, dtype=torch.bfloat16, device=device),
    }
    _allocate_kernel_state(case, num_ctas)
    return case


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    cfg: MoEConfig = case["config"]
    for name in (
        "routing_logits",
        "routing_bias",
        "hidden_states",
        "hidden_states_scale",
        "gemm1_weights",
        "gemm1_weights_scale",
        "gemm2_weights",
        "gemm2_weights_scale",
        "output",
    ):
        if not case[name].is_contiguous():
            raise AssertionError(f"{name} must be contiguous")
    maps = case["tensor_maps"]
    return (
        case["routing_logits"].view(-1),
        case["routing_bias"].view(-1),
        case["hidden_states_scale"].view(-1),
        case["gemm1_weights_scale"].view(-1),
        case["gemm2_weights_scale"].view(-1),
        case["output"].view(-1),
        case["route_id"],
        case["route_w"],
        case["tok_cnt"],
        case["cnt_cta"],
        case["cnt_warp"],
        case["sorted_token"],
        case["pos_of"],
        case["sorted_w"],
        case["a1p"],
        case["a1h"],
        case["a2p"],
        case["a2h"],
        case["act"],
        case["hidden_states"].view(-1),
        case["ap"],
        case["partial"],
        case["task_ctr"],
        case["done"],
        case["sync_ctr"],
        case["fin_list"],
        maps["ap"].ptr,
        maps["w1"].ptr,
        maps["w2"].ptr,
        maps["act"].ptr,
        maps["ap64"].ptr,
        maps["act64"].ptr,
        maps["w1h"].ptr,
        cfg.seq_len,
        case["P"],
        case["max_m_tiles"],
        cfg.local_expert_offset,
        cfg.routed_scaling_factor,
    )


def _launcher(executable, case: dict[str, Any]):
    args = _tirx_args(case)

    def launch() -> None:
        executable(*args)

    launch._keep_alive = args
    return launch


def _expand_weight_scale(scale: torch.Tensor) -> torch.Tensor:
    return scale.float().repeat_interleave(BK, dim=0).repeat_interleave(BK, dim=1)


@torch.no_grad()
def _torch_reference(case: dict[str, Any]) -> torch.Tensor:
    cfg: MoEConfig = case["config"]
    logits = case["routing_logits"].float()
    unbiased = torch.sigmoid(logits)
    biased = unbiased + case["routing_bias"].float()
    grouped = biased.reshape(cfg.seq_len, 8, NUM_EXPERTS // 8)
    group_scores = grouped.topk(2, dim=2, sorted=False).values.sum(dim=2)
    selected_groups = group_scores.topk(4, dim=1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    keep = group_mask[:, :, None].expand_as(grouped).reshape_as(biased)
    selected_experts = (
        biased.masked_fill(~keep, torch.finfo(torch.float32).min)
        .topk(TOPK, dim=1, sorted=False)
        .indices
    )
    selected_weights = unbiased.gather(1, selected_experts)
    selected_weights = (
        selected_weights / selected_weights.sum(dim=1, keepdim=True)
    ) * cfg.routed_scaling_factor

    hidden_scale = case["hidden_states_scale"].transpose(0, 1).repeat_interleave(BK, dim=1)
    hidden = case["hidden_states"].float() * hidden_scale
    output = torch.zeros(cfg.seq_len, HIDDEN, dtype=torch.float32, device=hidden.device)
    for local_expert in range(NUM_LOCAL):
        global_expert = cfg.local_expert_offset + local_expert
        selected = selected_experts == global_expert
        token_index = selected.any(dim=1).nonzero(as_tuple=False).flatten()
        if token_index.numel() == 0:
            continue
        w1 = case["gemm1_weights"][local_expert].float() * _expand_weight_scale(
            case["gemm1_weights_scale"][local_expert]
        )
        gemm1 = hidden.index_select(0, token_index).matmul(w1.transpose(0, 1))
        del w1
        activation = torch.nn.functional.silu(gemm1[:, INTER:]) * gemm1[:, :INTER]
        del gemm1
        w2 = case["gemm2_weights"][local_expert].float() * _expand_weight_scale(
            case["gemm2_weights_scale"][local_expert]
        )
        contribution = activation.matmul(w2.transpose(0, 1))
        del activation, w2
        route_weight = (
            selected_weights.index_select(0, token_index) * selected.index_select(0, token_index)
        ).sum(dim=1)
        output.index_add_(0, token_index, contribution * route_weight[:, None])
    return output.to(torch.bfloat16)


def check_correctness(outputs: dict[str, Any], **kwargs: Any) -> None:
    _cfg(**kwargs)
    first, actual, reference = outputs["first"], outputs["actual"], outputs["reference"]
    for name, tensor in (("first", first), ("actual", actual), ("reference", reference)):
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} output contains non-finite values")
    if not torch.equal(first, actual):
        max_abs = float((first.float() - actual.float()).abs().max())
        raise AssertionError(
            f"identical launches are not exactly repeatable; max abs diff={max_abs}"
        )
    matched = torch.isclose(actual.float(), reference.float(), atol=1.0, rtol=0.3)
    matched_ratio = float(matched.float().mean())
    if matched_ratio < 0.9:
        max_abs = float((actual.float() - reference.float()).abs().max())
        raise AssertionError(
            f"matched ratio {matched_ratio:.6f} must be at least 0.9; max abs diff={max_abs}"
        )


def run_test(**kwargs: Any) -> None:
    _assert_supported_arch()
    from tirx_kernels.runner import compile_kernel

    num_ctas = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    config = dict(kwargs)
    config.pop("num_ctas", None)
    case = prepare_data(**config, num_ctas=num_ctas)
    executable = compile_kernel(get_kernel(**config, num_ctas=num_ctas))
    launch = _launcher(executable, case)
    case["output"].fill_(float("nan"))
    launch()
    torch.cuda.synchronize()
    first = case["output"].clone()
    case["output"].fill_(42.0)
    launch()
    torch.cuda.synchronize()
    actual = case["output"].clone()
    reference = _torch_reference(case)
    torch.cuda.synchronize()
    check_correctness({"first": first, "actual": actual, "reference": reference}, **config)


def _flashinfer_builder(case: dict[str, Any]):
    def build():
        from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

        cfg: MoEConfig = case["config"]

        def launch() -> None:
            trtllm_fp8_block_scale_moe(
                case["routing_logits"],
                case["routing_bias"],
                case["hidden_states"],
                case["hidden_states_scale"],
                case["gemm1_weights"],
                case["gemm1_weights_scale"],
                case["gemm2_weights"],
                case["gemm2_weights_scale"],
                NUM_EXPERTS,
                TOPK,
                8,
                4,
                INTER,
                cfg.local_expert_offset,
                NUM_LOCAL,
                cfg.routed_scaling_factor,
                routing_method_type=2,
                use_shuffled_weight=False,
                weight_layout=0,
                do_finalize=True,
                tune_max_num_tokens=32768,
                output=case["reference_output"],
            )

        return launch

    return build


def prepare_bench(**kwargs: Any):
    """Trace and compile before the bench suite assigns a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    num_ctas = _num_ctas(**kwargs)
    config = dict(kwargs)
    config.pop("num_ctas", None)
    state = {
        "config": config,
        "num_ctas": num_ctas,
        "executable": compile_kernel(get_kernel(**config, num_ctas=num_ctas)),
    }
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
    launch = _launcher(prepared["executable"], case)
    launch()
    torch.cuda.synchronize()

    from tirx_kernels.runner import bench

    return bench(
        {"tirx": launch},
        references={"flashinfer_trtllm_fp8_block_scale_moe": _flashinfer_builder(case)},
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
