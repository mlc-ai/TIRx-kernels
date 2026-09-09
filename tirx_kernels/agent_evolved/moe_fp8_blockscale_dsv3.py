# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved DeepSeek-V3 FP8 block-scale MoE megakernel for B200.

The supported contract is the FlashInfer benchmark geometry with 256 global
experts, 32 local experts, hidden size 7168, intermediate size 2048, block-128
E4M3 inputs and weights, DeepSeek-V3 grouped routing (top-8, eight groups,
four selected groups), and bf16 output.  Sequence length is dynamic; the
registered benchmark shape is the official T=14107 maximum row.

The selected kernel is ``pair2sm-fp8-partials-megakernel`` from evolution run
``moe-20260908-004417``.  It is one persistent 12-warp Kern launch over
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

import tirx_kernels.kern as K
from tvm.backend.cuda.cpp.descriptors import (
    encode_instr_descriptor_block_scaled_uint32,
    encode_smem_descriptor_base_uint64,
)

KERNEL_META = {
    "name": "agent_evolved_moe_fp8_blockscale_dsv3",
    "category": "agent_evolved",
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
    return K.local_scalar(K.f32, init=v)


def i32(v):
    return K.local_scalar(K.i32, init=v)


def u32(v):
    return K.local_scalar(K.u32, init=v)


def shfl_idx(dst, v, lane):
    K.ptx.shfl_sync.idx.b32(dst, v, K.cast(lane, "uint32"), K.uint32(0x1F), K.uint32(0xFFFFFFFF))


def shfl_bfly(dst, v, m):
    K.ptx.shfl_sync.bfly.b32(dst, v, K.uint32(m), K.uint32(0x1F), K.uint32(0xFFFFFFFF))


def float_key(v):
    bits = K.reinterpret("uint32", v)
    neg = K.shift_right(bits, K.uint32(31)) != K.uint32(0)
    return K.Select(neg, K.bitwise_not(bits), K.bitwise_or(bits, K.uint32(0x80000000)))


def bf16_lo(word):
    return K.reinterpret("float32", K.shift_left(word, K.uint32(16)))


def bf16_hi(word):
    return K.reinterpret("float32", K.bitwise_and(word, K.uint32(0xFFFF0000)))


def ue8m0_pack4(v):
    """Round a positive f32 scale into UE8M0 sub-column zero."""
    bits = K.reinterpret("uint32", v)

    code = K.bitwise_and(K.shift_right(bits + K.uint32(0x400000), K.uint32(23)), K.uint32(0xFF))
    return code


def with_sf_id(desc, sf_id):
    """Set both block-scale ID fields ([31:29] and [6:4])."""
    out = K.bitwise_and(desc, K.uint32(0x9FFFFFCF))
    out = K.bitwise_or(out, K.shift_left(K.cast(sf_id, "uint32"), K.uint32(29)))
    return K.bitwise_or(out, K.shift_left(K.cast(sf_id, "uint32"), K.uint32(4)))


def with_smem_addr(desc_base, addr):
    """Fill descriptor address bits [13:0] from a 16-byte-aligned SMEM address."""
    addr_field = K.cast(K.bitwise_and(K.shift_right(addr, K.uint32(4)), K.uint32(0x3FFF)), "uint64")
    return K.bitwise_or(K.uint64(desc_base), addr_field)


def iket_range(name):
    token = K.alloc_local([1], "uint32")
    K.assign(token[0], K.cuda.iket.range_start(name))
    return token


def iket_end(token):
    K.cuda.iket.range_end(token[0])


def remote_arrive_leader(bar_ptr):
    """Arrive (count 1) on the leader CTA's copy of a shared-memory mbarrier (rank 0 of the pair)."""
    rem = K.local_scalar(K.u64)
    K.ptx.mapa.shared__cluster.u64(rem, bar_ptr, K.uint32(0))
    K.ptx.mbarrier.arrive.b64(rem, K.uint32(1), pred=K.bool(True))


def emit_grid_sync(ctr_ptr, cta, num_ctas, tid):
    """Sense-reversing grid barrier over all CTAs (all threads of the CTA participate)."""
    K.ptx.bar.sync(K.uint32(0))
    with K.If(tid == 0), K.Then():
        old = K.local_scalar(K.u32)
        with K.If(cta == 0):
            with K.Then():
                K.ptx.atom.release.gpu.global_.add.u32(
                    old, ctr_ptr, K.uint32(0x80000000 - (num_ctas - 1))
                )
            with K.Else():
                K.ptx.atom.release.gpu.global_.add.u32(old, ctr_ptr, K.uint32(1))
        cur = K.local_scalar(K.u32)
        K.ptx.ld.acquire.gpu.global_.b32(cur, ctr_ptr)
        with K.While(K.bitwise_and(K.bitwise_xor(cur, old), K.uint32(0x80000000)) == K.uint32(0)):
            K.ptx.ld.acquire.gpu.global_.b32(cur, ctr_ptr)
    K.ptx.bar.sync(K.uint32(0))


def emit_route_token(
    t, lane, logits, bias, offset, rsf, route_id, route_w, tok_cnt, wcnt, fin_list, fin_ctr
):
    """Route one token with one warp; lane l owns experts [8l, 8l+8)."""
    x = K.alloc_local((8,), K.f32)
    sv = K.alloc_local((8,), K.f32)
    bv = K.alloc_local((8,), K.f32)
    braw = K.alloc_local((4,), K.u32)
    K.ptx.ld.global_.nc.v4.f32(x[0], x[1], x[2], x[3], logits.ptr_to([t * NUM_EXPERTS + lane * 8]))
    K.ptx.ld.global_.nc.v4.f32(
        x[4], x[5], x[6], x[7], logits.ptr_to([t * NUM_EXPERTS + lane * 8 + 4])
    )
    K.ptx.ld.global_.nc.v4.b32(braw[0], braw[1], braw[2], braw[3], bias.ptr_to([lane * 8]))
    for j in range(8):
        e = K.local_scalar(K.f32)
        K.ptx.ex2.approx.ftz.f32(e, x[j] * K.float32(-LOG2E))
        K.ptx.rcp.approx.ftz.f32(sv[j], e + K.float32(1.0))
        word = braw[j // 2]
        bbits = (
            K.shift_left(word, K.uint32(16))
            if j % 2 == 0
            else K.bitwise_and(word, K.uint32(0xFFFF0000))
        )
        K.assign(bv[j], sv[j] + K.reinterpret("float32", bbits))
    m1 = f32(K.max(bv[0], bv[1]))
    m2 = f32(K.min(bv[0], bv[1]))
    for j in range(2, 8):
        K.assign(m2, K.max(m2, K.min(m1, bv[j])))
        K.assign(m1, K.max(m1, bv[j]))
    for m in (1, 2):
        n1 = K.local_scalar(K.f32)
        n2 = K.local_scalar(K.f32)
        shfl_bfly(n1, m1, m)
        shfl_bfly(n2, m2, m)
        new1 = f32(K.max(m1, n1))
        new2 = f32(K.max(K.min(m1, n1), K.max(m2, n2)))
        K.assign(m1, new1)
        K.assign(m2, new2)
    gs = f32(m1 + m2)
    my_g = lane // 4
    rank = i32(K.int32(0))
    for g in range(8):
        og = K.local_scalar(K.f32)
        shfl_idx(og, gs, K.int32(g * 4))
        K.assign(rank, rank + K.cast(K.Or(og > gs, K.And(og == gs, K.int32(g) < my_g)), "int32"))
    keep = rank < 4
    for j in range(8):
        K.assign(bv[j], K.Select(keep, bv[j], K.float32(NEG_INF)))
    my_e = i32(K.int32(-1))
    my_s = f32(K.float32(0.0))
    for r in range(8):
        bestv = f32(bv[0])
        bestj = i32(K.int32(0))
        for j in range(1, 8):
            take = u32(K.cast(bv[j] > bestv, "uint32"))
            K.assign(bestj, K.Select(take != K.uint32(0), K.int32(j), bestj))
            K.assign(bestv, K.Select(take != K.uint32(0), bv[j], bestv))
        key = u32(float_key(bestv))
        wkey = K.local_scalar(K.u32)
        K.ptx.redux_sync.max.u32(wkey, key, K.uint32(0xFFFFFFFF))
        bal = K.local_scalar(K.u32)
        K.ptx.vote_sync.ballot.b32(
            bal, K.ptx.pred(K.cast(key == wkey, "uint32")), K.uint32(0xFFFFFFFF)
        )
        wlane = i32(K.cuda.ffs_u32(bal) - K.int32(1))
        ssel = f32(sv[0])
        for j in range(1, 8):
            K.assign(ssel, K.Select(bestj == K.int32(j), sv[j], ssel))
        widx = K.local_scalar(K.i32)
        wsv = K.local_scalar(K.f32)
        shfl_idx(widx, wlane * 8 + bestj, wlane)
        shfl_idx(wsv, ssel, wlane)
        with K.If(lane == wlane), K.Then():
            for j in range(8):
                K.assign(bv[j], K.Select(bestj == K.int32(j), K.float32(NEG_INF), bv[j]))
        K.assign(my_e, K.Select(lane == K.int32(r), widx, my_e))
        K.assign(my_s, K.Select(lane == K.int32(r), wsv, my_s))
    contrib = f32(K.Select(lane < 8, my_s, K.float32(0.0)))
    for m in (1, 2, 4):
        o = K.local_scalar(K.f32)
        shfl_bfly(o, contrib, m)
        K.assign(contrib, contrib + o)
    wgt = f32((my_s / (contrib + K.float32(1e-20))) * rsf)
    loc = my_e - offset
    locv = i32(K.Select(K.And(loc >= 0, loc < NUM_LOCAL), loc, K.int32(-1)))
    with K.If(lane < 8), K.Then():
        K.ptx.st.global_.s32(route_id.ptr_to([t * TOPK + lane]), locv)
        K.ptx.st.global_.f32(route_w.ptr_to([t * TOPK + lane]), wgt)
    for k in range(8):
        ek = K.local_scalar(K.i32)
        shfl_idx(ek, locv, K.int32(k))
        K.assign(wcnt, wcnt + K.cast(ek == lane, "int32"))

    lbal = K.local_scalar(K.u32)
    K.ptx.vote_sync.ballot.b32(
        lbal, K.ptx.pred(K.cast(K.And(lane < 8, locv >= 0), "uint32")), K.uint32(0xFFFFFFFF)
    )
    with K.If(lane == 0), K.Then():
        nloc = i32(K.cast(K.popcount(lbal), "int32"))
        K.ptx.st.global_.s32(tok_cnt.ptr_to([t]), nloc)

        with K.If(nloc >= 1), K.Then():
            fidx = K.local_scalar(K.u32)
            K.ptx.atom.global_.add.u32(fidx, fin_ctr, K.uint32(1))
            K.ptx.st.global_.s32(fin_list.ptr_to([K.cast(fidx, "int32")]), t)


def build_kernel(num_ctas):
    ctas_per_warp = (num_ctas + NWARPS - 1) // NWARPS

    @K.kernel(warps=NWARPS, arch="sm_100a", min_blocks_per_sm=1, grid=num_ctas)
    def moe_mega(
        logits: K.gptr[K.f32],
        bias: K.gptr[K.bf16],
        hs_scale: K.gptr[K.f32],
        w1s: K.gptr[K.f32],
        w2s: K.gptr[K.f32],
        out: K.gptr[K.bf16],
        route_id: K.gptr[K.i32],
        route_w: K.gptr[K.f32],
        tok_cnt: K.gptr[K.i32],
        cnt_cta: K.gptr[K.i32],
        cnt_warp: K.gptr[K.i32],
        sorted_token: K.gptr[K.i32],
        pos_of: K.gptr[K.i32],
        sorted_w: K.gptr[K.f32],
        a1p: K.gptr[K.u32],
        a1h: K.gptr[K.u32],
        a2p: K.gptr[K.u8],
        a2h: K.gptr[K.u8],
        act: K.gptr[K.f8e4m3],
        hidden: K.gptr[K.f8e4m3],
        ap: K.gptr[K.f8e4m3],
        partial: K.gptr[K.f8e4m3],
        task_ctr: K.gptr[K.u32],
        done: K.gptr[K.u32],
        sync_ctr: K.gptr[K.u32],
        fin_list: K.gptr[K.i32],
        tm_ap: K.TensorMap,
        tm_w1: K.TensorMap,
        tm_w2: K.TensorMap,
        tm_act: K.TensorMap,
        tm_ap64: K.TensorMap,
        tm_act64: K.TensorMap,
        tm_w1h: K.TensorMap,
        T: K.i32,
        P: K.i32,
        MAXMT: K.i32,
        offset: K.i32,
        rsf: K.f32,
    ):
        cta = K.cta_id()
        crank = K.cta_id_in_cluster([PAIR])
        is_leader = crank == 0
        warp = K.warp_id()
        lane = K.lane_id()
        tid = K.thread_id()

        smem = K.smem_pool()
        a_tile0 = smem.alloc((STAGES, BM, BK), K.f8e4m3, align=1024, swizzle=K.SW128B)
        a_tile1 = smem.alloc((STAGES, BM, BK), K.f8e4m3, align=1024, swizzle=K.SW128B)
        b_tile0 = smem.alloc((STAGES, BNH, BK), K.f8e4m3, align=1024, swizzle=K.SW128B)
        b_tile1 = smem.alloc((STAGES, BNH, BK), K.f8e4m3, align=1024, swizzle=K.SW128B)
        a_tiles = (a_tile0, a_tile1)
        b_tiles = (b_tile0, b_tile1)
        sfa_tile = smem.alloc((STAGES, BM), K.u32, align=16)

        sfb_tile = smem.alloc((STAGES, BN), K.u32, align=16)
        s_cnt = smem.alloc((32,), K.i32, align=16)
        s_base = smem.alloc((32,), K.i32, align=16)
        s_ecnt = smem.alloc((32,), K.i32, align=16)
        s_eoff = smem.alloc((32,), K.i32, align=16)
        s_mpre = smem.alloc((33,), K.i32, align=16)
        s_ptot = smem.alloc((NWARPS, 32), K.i32, align=16)
        s_ppre = smem.alloc((NWARPS, 32), K.i32, align=16)
        s_task = smem.alloc((TASK_RING, 8), K.i32, align=16)
        s_amax = smem.alloc((512,), K.f32, align=16)

        s_epi = smem.alloc((NWARPS - MATH_WARP0, 16, 36), K.u32, align=16)
        tmem_slot = smem.alloc((1,), K.u32, align=4)

        full_bar = K.TMABar(smem, STAGES * KPACK)
        empty_bar = K.TCGen05Bar(smem, STAGES * KPACK)
        sf_bar = K.MBarrier(smem, STAGES)

        sfempty_bar = K.TCGen05Bar(smem, STAGES)

        sfa_bar = K.TMABar(smem, STAGES)
        tfull_bar = K.TCGen05Bar(smem, 1)
        tempty_bar = K.MBarrier(smem, 1)
        task_full = K.MBarrier(smem, TASK_RING)
        task_empty = K.MBarrier(smem, TASK_RING)

        roles = K.specialize(chain_dispatch=True)
        prod_role = roles.role("prod", warps=[0], regs=REGS_WG0)
        mma_role = roles.role("mma", warps=[1], regs=REGS_WG0)
        aux_role = roles.role("aux", warps=[2, 3], regs=REGS_WG0)
        math_role = roles.role("math", warps=range(MATH_WARP0, NWARPS), regs=REGS_MATH)

        K.ptx.barrier.cluster.arrive.relaxed.aligned()
        K.ptx.barrier.cluster.wait.acquire.aligned()
        with K.If(warp == 1), K.Then():
            with K.If(lane == 0), K.Then():
                for s in range(STAGES * KPACK):
                    K.ptx.mbarrier.init.shared.b64(full_bar.ptr_to([s]), K.uint32(1))
                    K.ptx.mbarrier.init.shared.b64(empty_bar.ptr_to([s]), K.uint32(1))
                for s in range(STAGES):
                    K.ptx.mbarrier.init.shared.b64(sf_bar.ptr_to([s]), K.uint32(2 * PAIR))
                    K.ptx.mbarrier.init.shared.b64(sfempty_bar.ptr_to([s]), K.uint32(1))
                    K.ptx.mbarrier.init.shared.b64(sfa_bar.ptr_to([s]), K.uint32(1))
                for s in range(1):
                    K.ptx.mbarrier.init.shared.b64(tfull_bar.ptr_to([s]), K.uint32(1))

                    K.ptx.mbarrier.init.shared.b64(
                        tempty_bar.ptr_to([s]), K.uint32(PAIR * (NWARPS - MATH_WARP0))
                    )
                for s in range(TASK_RING):
                    K.ptx.mbarrier.init.shared.b64(task_full.ptr_to([s]), K.uint32(1))
                    K.ptx.mbarrier.init.shared.b64(task_empty.ptr_to([s]), K.uint32(NCONS))
                K.ptx.fence.mbarrier_init.release.cluster()
        with K.If(warp == 2), K.Then():
            K.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                K.address_of(tmem_slot[0]), K.uint32(512)
            )
            K.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
        with K.If(warp == 0), K.Then():
            with K.If(lane == 0), K.Then():
                K.ptx.prefetch.tensormap(K.address_of(tm_ap))
                K.ptx.prefetch.tensormap(K.address_of(tm_w1))
                K.ptx.prefetch.tensormap(K.address_of(tm_w2))
                K.ptx.prefetch.tensormap(K.address_of(tm_act))
                K.ptx.prefetch.tensormap(K.address_of(tm_ap64))
                K.ptx.prefetch.tensormap(K.address_of(tm_act64))
                K.ptx.prefetch.tensormap(K.address_of(tm_w1h))
        with K.If(cta == 0), K.Then():
            with K.If(tid == 0), K.Then():
                K.ptx.st.global_.u32(task_ctr.ptr_to([0]), K.uint32(0))
                K.ptx.st.global_.u32(task_ctr.ptr_to([1]), K.uint32(0))
            di = i32(tid)
            with K.While(di < MAXMT * NUM_LOCAL):
                K.ptx.st.global_.u32(done.ptr_to([di]), K.uint32(0))
                K.assign(di, di + NWARPS * 32)
        with K.If(tid < 32), K.Then():
            K.ptx.st.shared.s32(s_cnt.ptr_to([tid]), K.int32(0))
        K.ptx.barrier.cluster.arrive.release.aligned()
        K.ptx.barrier.cluster.wait.acquire.aligned()

        chunk = (T + (num_ctas - 1)) // num_ctas
        t_begin = cta * chunk
        t_end = K.min(T, t_begin + chunk)
        wcnt = i32(K.int32(0))
        t = i32(t_begin + warp)
        with K.While(t < t_end):
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
            K.assign(t, t + NWARPS)
        K.ptx.st.global_.s32(cnt_warp.ptr_to([(cta * NWARPS + warp) * 32 + lane]), wcnt)
        K.ptx.red.shared.add.u32(s_cnt.ptr_to([lane]), K.cast(wcnt, "uint32"))
        K.ptx.bar.sync(K.uint32(0))
        with K.If(tid < 32), K.Then():
            v = K.local_scalar(K.i32)
            K.ptx.ld.shared.s32(v, s_cnt.ptr_to([tid]))
            K.ptx.st.global_.s32(cnt_cta.ptr_to([cta * 32 + tid]), v)

        K.cuda.iket.mark("R1-done")
        emit_grid_sync(sync_ctr.ptr_to([0]), cta, num_ctas, tid)

        ptot = i32(K.int32(0))
        ppre = i32(K.int32(0))
        vals = K.alloc_local((ctas_per_warp,), K.i32)
        for i in range(ctas_per_warp):
            c = warp + i * NWARPS
            K.assign(vals[i], K.int32(0))
            with K.If(c < num_ctas), K.Then():
                K.ptx.ld.global_.s32(vals[i], cnt_cta.ptr_to([c * 32 + lane]))
        for i in range(ctas_per_warp):
            c = warp + i * NWARPS
            K.assign(ptot, ptot + vals[i])
            K.assign(ppre, ppre + K.Select(c < cta, vals[i], K.int32(0)))
        K.ptx.st.shared.s32(s_ptot.ptr_to([warp, lane]), ptot)
        K.ptx.st.shared.s32(s_ppre.ptr_to([warp, lane]), ppre)
        K.ptx.bar.sync(K.uint32(0))
        with K.If(warp == 0), K.Then():
            tot = i32(K.int32(0))
            pre = i32(K.int32(0))
            for w in range(NWARPS):
                v = K.local_scalar(K.i32)
                K.ptx.ld.shared.s32(v, s_ptot.ptr_to([w, lane]))
                K.assign(tot, tot + v)
                K.ptx.ld.shared.s32(v, s_ppre.ptr_to([w, lane]))
                K.assign(pre, pre + v)
            mtiles = i32((tot + (BMP - 1)) // BMP)
            incl = i32(mtiles)
            for m in (1, 2, 4, 8, 16):
                o = K.local_scalar(K.i32)
                K.ptx.shfl_sync.up.b32(o, incl, K.uint32(m), K.uint32(0), K.uint32(0xFFFFFFFF))
                K.assign(incl, incl + K.Select(lane >= m, o, K.int32(0)))
            excl = i32(incl - mtiles)
            K.ptx.st.shared.s32(s_ecnt.ptr_to([lane]), tot)
            K.ptx.st.shared.s32(s_eoff.ptr_to([lane]), excl * BMP)
            K.ptx.st.shared.s32(s_base.ptr_to([lane]), excl * BMP + pre)
            K.ptx.st.shared.s32(s_mpre.ptr_to([lane]), excl)
            with K.If(lane == 31), K.Then():
                K.ptx.st.shared.s32(s_mpre.ptr_to([32]), incl)
        K.ptx.bar.sync(K.uint32(0))

        base = K.local_scalar(K.i32)
        K.ptx.ld.shared.s32(base, s_base.ptr_to([lane]))
        for wp in range(NWARPS - 1):
            with K.If(wp < warp), K.Then():
                v = K.local_scalar(K.i32)
                K.ptx.ld.global_.s32(v, cnt_warp.ptr_to([(cta * NWARPS + wp) * 32 + lane]))
                K.assign(base, base + v)
        cursor = i32(K.int32(0))
        K.assign(t, t_begin + warp)
        with K.While(t < t_end):
            e = i32(K.int32(-1))
            w = f32(K.float32(0.0))
            with K.If(lane < 8), K.Then():
                K.ptx.ld.global_.s32(e, route_id.ptr_to([t * TOPK + lane]))
                K.ptx.ld.global_.f32(w, route_w.ptr_to([t * TOPK + lane]))
            has_local = K.local_scalar(K.u32)
            K.ptx.vote_sync.ballot.b32(
                has_local,
                K.ptx.pred(K.cast(K.And(lane < 8, e >= 0), "uint32")),
                K.uint32(0xFFFFFFFF),
            )
            sc0 = f32(K.float32(0.0))
            sc1 = f32(K.float32(0.0))
            with K.If(has_local != K.uint32(0)), K.Then():
                with K.If(lane < KB1), K.Then():
                    K.ptx.ld.global_.f32(sc0, hs_scale.ptr_to([lane * T + t]))
                with K.If(lane < KB1 - 32), K.Then():
                    K.ptx.ld.global_.f32(sc1, hs_scale.ptr_to([(lane + 32) * T + t]))
            sc0q = u32(ue8m0_pack4(sc0))
            sc1q = u32(ue8m0_pack4(sc1))
            qsrc = i32(K.Select(lane < 16, lane * 2, (lane - 16) * 2))
            q00 = K.local_scalar(K.u32)
            q01 = K.local_scalar(K.u32)
            q10 = K.local_scalar(K.u32)
            q11 = K.local_scalar(K.u32)
            shfl_idx(q00, sc0q, qsrc)
            shfl_idx(q01, sc0q, qsrc + 1)
            shfl_idx(q10, sc1q, qsrc)
            shfl_idx(q11, sc1q, qsrc + 1)
            qpair = u32(
                K.bitwise_or(
                    K.Select(lane < 16, q00, q10),
                    K.shift_left(K.Select(lane < 16, q01, q11), K.uint32(8)),
                )
            )
            ek = []
            for k in range(8):
                v = K.local_scalar(K.i32)
                shfl_idx(v, e, K.int32(k))
                ek.append(v)
            rank = i32(K.int32(0))
            cnt_here = i32(K.int32(0))
            for k in range(8):
                K.assign(rank, rank + K.cast(K.And(K.int32(k) < lane, ek[k] == e), "int32"))
                K.assign(cnt_here, cnt_here + K.cast(ek[k] == lane, "int32"))
            cur_e = K.local_scalar(K.i32)
            shfl_idx(cur_e, cursor, K.max(e, K.int32(0)))
            base_e = K.local_scalar(K.i32)
            shfl_idx(base_e, base, K.max(e, K.int32(0)))
            pos = i32(base_e + cur_e + rank)
            with K.If(lane < 8), K.Then():
                with K.If(e >= 0):
                    with K.Then():
                        K.ptx.st.global_.s32(sorted_token.ptr_to([pos]), t)
                        K.ptx.st.global_.s32(pos_of.ptr_to([t * TOPK + lane]), pos)
                        K.ptx.st.global_.f32(sorted_w.ptr_to([pos]), w)
                    with K.Else():
                        K.ptx.st.global_.s32(pos_of.ptr_to([t * TOPK + lane]), K.int32(-1))
            K.assign(cursor, cursor + cnt_here)

            rowv = K.alloc_local((ROW_CHUNKS * 4,), K.u32)
            with K.If(has_local != K.uint32(0)), K.Then():
                for c in range(ROW_CHUNKS):
                    K.ptx.ld.global_.nc.v4.b32(
                        rowv[4 * c],
                        rowv[4 * c + 1],
                        rowv[4 * c + 2],
                        rowv[4 * c + 3],
                        hidden.ptr_to([t * HIDDEN + c * 512 + lane * 16]),
                    )
            for k in range(8):
                posk = K.local_scalar(K.i32)
                shfl_idx(posk, pos, K.int32(k))
                with K.If(ek[k] >= 0), K.Then():
                    with K.If(lane < A1_PACKS), K.Then():
                        ht = posk // BM
                        lr = posk - ht * BM
                        pidx = (ht * A1_PACKS + lane) * BM + (lr % 32) * 4 + lr // 32
                        K.ptx.st.global_.u32(a1p.ptr_to([pidx]), qpair)
                        ht2 = posk // (BM // 2)
                        lr2 = posk - ht2 * (BM // 2)
                        hidx = (ht2 * A1_PACKS + lane) * BM + (lr2 % 32) * 4 + lr2 // 32
                        K.ptx.st.global_.u32(a1h.ptr_to([hidx]), qpair)
                    for c in range(ROW_CHUNKS):
                        K.ptx.st.global_.v4.b32(
                            ap.ptr_to([posk * HIDDEN + c * 512 + lane * 16]),
                            rowv[4 * c],
                            rowv[4 * c + 1],
                            rowv[4 * c + 2],
                            rowv[4 * c + 3],
                        )
            K.assign(t, t + NWARPS)

        K.cuda.iket.mark("R2-done")
        emit_grid_sync(sync_ctr.ptr_to([0]), cta, num_ctas, tid)
        K.cuda.iket.mark("G-start")

        total_mt = K.local_scalar(K.i32)
        K.ptx.ld.shared.s32(total_mt, s_mpre.ptr_to([32]))
        n_g1 = total_mt * NT1
        n_g2 = total_mt * NT2
        tmem_base = K.local_scalar(K.u32)
        K.ptx.ld.shared.u32(tmem_base, tmem_slot.ptr_to([0]))

        def read_task(tstate, kind, e, mt, nt, row0, valid, nkb, half):
            tk_t = iket_range("wait-task")
            K.cuda.mbarrier_wait(task_full.ptr_to([tstate.stage]), tstate.phase)
            iket_end(tk_t)
            K.ptx.ld.shared.s32(kind, s_task.ptr_to([tstate.stage, 0]))
            K.ptx.ld.shared.s32(e, s_task.ptr_to([tstate.stage, 1]))
            K.ptx.ld.shared.s32(mt, s_task.ptr_to([tstate.stage, 2]))
            K.ptx.ld.shared.s32(nt, s_task.ptr_to([tstate.stage, 3]))
            K.ptx.ld.shared.s32(row0, s_task.ptr_to([tstate.stage, 4]))
            K.ptx.ld.shared.s32(valid, s_task.ptr_to([tstate.stage, 5]))
            K.ptx.ld.shared.s32(nkb, s_task.ptr_to([tstate.stage, 6]))
            K.ptx.ld.shared.s32(half, s_task.ptr_to([tstate.stage, 7]))
            K.cuda.warp_sync()
            with K.If(lane == 0), K.Then():
                remote_arrive_leader(task_empty.ptr_to([tstate.stage]))
            tstate.advance()

        def wait_done(e, mt):
            """GEMM2 pair-tiles need all GEMM1 pair-tiles of their m-tile (both CTAs arrive per tile)."""
            with K.If(lane == 0), K.Then():
                dv = K.local_scalar(K.u32)
                K.ptx.ld.acquire.gpu.global_.b32(dv, done.ptr_to([e * MAXMT + mt]))
                with K.While(dv < K.uint32(PAIR * NT1)):
                    K.ptx.ld.acquire.gpu.global_.b32(dv, done.ptr_to([e * MAXMT + mt]))
            K.cuda.warp_sync()
            K.ptx.fence.proxy.async_.global_()

        def producer_loads(kind, e, nt, row0, nkb, sstate, half):
            """Pair: this CTA's 128 A rows + 128 of the 256 B rows. Half: this CTA's 64 A rows of the lone tile + the same B share."""
            arow0 = i32(K.Select(half != 0, row0 + crank * (BM // 2), row0 + crank * BM))
            brow = i32(
                K.Select(
                    kind == 0,
                    e * (2 * INTER) + crank * INTER + nt * BNH,
                    e * HIDDEN + nt * BN + crank * BNH,
                )
            )
            brow_g = e * (2 * INTER) + nt * 128 + crank * 64
            brow_u = e * (2 * INTER) + INTER + nt * 128 + crank * 64
            tk_tile = iket_range("prod-tile")
            with K.serial(0, nkb) as kb:
                for sub in range(KPACK):
                    slot = sstate.stage * KPACK + sub
                    tk_w = iket_range("prod-wait-empty")
                    K.cuda.mbarrier_wait(empty_bar.ptr_to([slot]), sstate.phase ^ 1)
                    iket_end(tk_w)
                    with K.If(lane == 0), K.Then():
                        lbar = K.cuda.sm100_2sm_leader_smem_addr(full_bar.ptr_to([slot]))
                        with K.If(is_leader), K.Then():
                            K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                full_bar.ptr_to([slot]),
                                K.Select(
                                    half != 0,
                                    K.uint32(PAIR * (HALF_A_BYTES + B_STAGE_BYTES)),
                                    K.uint32(PAIR * (A_STAGE_BYTES + B_STAGE_BYTES)),
                                ),
                            )
                        kcol = (kb * KPACK + sub) * BK
                        with K.If(half != 0):
                            with K.Then():
                                with K.If(kind == 0):
                                    with K.Then():
                                        K.ptx[TMA_2SM](
                                            a_tiles[sub][sstate.stage].ptr_to(0, 0),
                                            K.address_of(tm_ap64),
                                            kcol,
                                            arow0,
                                            lbar,
                                            K.uint64(EVICT_NORMAL),
                                        )
                                        K.ptx[TMA_2SM](
                                            b_tiles[sub][sstate.stage].ptr_to(0, 0),
                                            K.address_of(tm_w1h),
                                            kcol,
                                            brow_g,
                                            lbar,
                                            K.uint64(EVICT_NORMAL),
                                        )
                                        K.ptx[TMA_2SM](
                                            b_tiles[sub][sstate.stage].ptr_to(64, 0),
                                            K.address_of(tm_w1h),
                                            kcol,
                                            brow_u,
                                            lbar,
                                            K.uint64(EVICT_NORMAL),
                                        )
                                    with K.Else():
                                        K.ptx[TMA_2SM](
                                            a_tiles[sub][sstate.stage].ptr_to(0, 0),
                                            K.address_of(tm_act64),
                                            kcol,
                                            arow0,
                                            lbar,
                                            K.uint64(EVICT_NORMAL),
                                        )
                                        K.ptx[TMA_2SM](
                                            b_tiles[sub][sstate.stage].ptr_to(0, 0),
                                            K.address_of(tm_w2),
                                            kcol,
                                            brow,
                                            lbar,
                                            K.uint64(EVICT_NORMAL),
                                        )
                            with K.Else():
                                with K.If(kind == 0):
                                    with K.Then():
                                        K.ptx[TMA_2SM](
                                            a_tiles[sub][sstate.stage].ptr_to(0, 0),
                                            K.address_of(tm_ap),
                                            kcol,
                                            arow0,
                                            lbar,
                                            K.uint64(EVICT_NORMAL),
                                        )
                                        K.ptx[TMA_2SM](
                                            b_tiles[sub][sstate.stage].ptr_to(0, 0),
                                            K.address_of(tm_w1),
                                            kcol,
                                            brow,
                                            lbar,
                                            K.uint64(EVICT_NORMAL),
                                        )
                                    with K.Else():
                                        K.ptx[TMA_2SM](
                                            a_tiles[sub][sstate.stage].ptr_to(0, 0),
                                            K.address_of(tm_act),
                                            kcol,
                                            arow0,
                                            lbar,
                                            K.uint64(EVICT_NORMAL),
                                        )
                                        K.ptx[TMA_2SM](
                                            b_tiles[sub][sstate.stage].ptr_to(0, 0),
                                            K.address_of(tm_w2),
                                            kcol,
                                            brow,
                                            lbar,
                                            K.uint64(EVICT_NORMAL),
                                        )
                    K.cuda.warp_sync()
                sstate.advance()
            iket_end(tk_tile)

        def scale_a_copies(kind, e, nt, row0, nkb, sstate, half):
            """Aux warp 0: per stage, bulk-copy this CTA's 512 B A-scale image into the stage's
            scale words (local completion barrier), and forward the previous stage's completion
            to the leader's scale barrier one iteration later so the copy latency is hidden."""
            arow0 = i32(K.Select(half != 0, row0 + crank * (BM // 2), row0 + crank * BM))
            hbase = i32(
                K.Select(half != 0, (arow0 // (BM // 2)) * A1_PACKS, (arow0 // BM) * A1_PACKS)
            )
            hbase2 = i32(
                K.Select(half != 0, (arow0 // (BM // 2)) * A2_PACKS, (arow0 // BM) * A2_PACKS)
            )
            have_prev = u32(K.uint32(0))
            prev_stage = i32(K.int32(0))
            prev_phase = u32(K.uint32(0))
            tk_tile = iket_range("scale-tile")
            with K.serial(0, nkb) as kb:
                K.cuda.mbarrier_wait(sfempty_bar.ptr_to([sstate.stage]), sstate.phase ^ 1)
                with K.If(lane == 0), K.Then():
                    lbar_local = K.cuda.cvta_generic_to_shared(sfa_bar.ptr_to([sstate.stage]))
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        sfa_bar.ptr_to([sstate.stage]), K.uint32(SF_TILE_BYTES)
                    )
                    with K.If(kind == 0):
                        with K.Then():
                            with K.If(half != 0):
                                with K.Then():
                                    K.ptx[BULK_G2S](
                                        sfa_tile.ptr_to([sstate.stage, 0]),
                                        a1h.ptr_to([(hbase + kb) * BM]),
                                        K.uint32(SF_TILE_BYTES),
                                        lbar_local,
                                    )
                                with K.Else():
                                    K.ptx[BULK_G2S](
                                        sfa_tile.ptr_to([sstate.stage, 0]),
                                        a1p.ptr_to([(hbase + kb) * BM]),
                                        K.uint32(SF_TILE_BYTES),
                                        lbar_local,
                                    )
                        with K.Else():
                            with K.If(half != 0):
                                with K.Then():
                                    K.ptx[BULK_G2S](
                                        sfa_tile.ptr_to([sstate.stage, 0]),
                                        a2h.ptr_to([((hbase2 + kb) * BM) * 4]),
                                        K.uint32(SF_TILE_BYTES),
                                        lbar_local,
                                    )
                                with K.Else():
                                    K.ptx[BULK_G2S](
                                        sfa_tile.ptr_to([sstate.stage, 0]),
                                        a2p.ptr_to([((hbase2 + kb) * BM) * 4]),
                                        K.uint32(SF_TILE_BYTES),
                                        lbar_local,
                                    )
                with K.If(have_prev != K.uint32(0)), K.Then():
                    K.cuda.mbarrier_wait(sfa_bar.ptr_to([prev_stage]), prev_phase)
                    K.cuda.warp_sync()
                    with K.If(lane == 0), K.Then():
                        remote_arrive_leader(sf_bar.ptr_to([prev_stage]))
                K.assign(prev_stage, sstate.stage)
                K.assign(prev_phase, K.cast(sstate.phase, "uint32"))
                K.assign(have_prev, K.uint32(1))
                sstate.advance()
            K.cuda.mbarrier_wait(sfa_bar.ptr_to([prev_stage]), prev_phase)
            K.cuda.warp_sync()
            with K.If(lane == 0), K.Then():
                remote_arrive_leader(sf_bar.ptr_to([prev_stage]))
            iket_end(tk_tile)

        def scale_loads(kind, e, nt, row0, nkb, sstate, half, aw):
            """Aux warp 1 stages both N halves of the B scales (A scales arrive by bulk copy).

            The global loads for K-pair kb+1 are issued right after stage kb is published, so
            their latency overlaps the wait for stage reuse."""
            raw = K.alloc_local((4,), K.u32)

            def load_regs(kb):
                for sub in range(KPACK):
                    ks = kb * KPACK + sub
                    bscale0 = K.local_scalar(K.f32)
                    bscale1 = K.local_scalar(K.f32)
                    with K.If(kind == 0):
                        with K.Then():
                            K.ptx.ld.global_.f32(
                                bscale0, w1s.ptr_to([e * (32 * KB1) + nt * KB1 + ks])
                            )
                            K.ptx.ld.global_.f32(
                                bscale1, w1s.ptr_to([e * (32 * KB1) + (16 + nt) * KB1 + ks])
                            )
                        with K.Else():
                            K.ptx.ld.global_.f32(
                                bscale0, w2s.ptr_to([e * (KB1 * KB2) + (nt * 2) * KB2 + ks])
                            )
                            K.ptx.ld.global_.f32(
                                bscale1, w2s.ptr_to([e * (KB1 * KB2) + (nt * 2 + 1) * KB2 + ks])
                            )
                    K.assign(raw[sub * 2], K.reinterpret("uint32", bscale0))
                    K.assign(raw[sub * 2 + 1], K.reinterpret("uint32", bscale1))

            tk_tile = iket_range("scale-tile")
            load_regs(K.int32(0))
            with K.serial(0, nkb) as kb:
                K.cuda.mbarrier_wait(sfempty_bar.ptr_to([sstate.stage]), sstate.phase ^ 1)
                bpack0 = u32(
                    K.bitwise_or(
                        ue8m0_pack4(K.reinterpret("float32", raw[0])),
                        K.shift_left(ue8m0_pack4(K.reinterpret("float32", raw[2])), K.uint32(8)),
                    )
                )
                bpack1 = u32(
                    K.bitwise_or(
                        ue8m0_pack4(K.reinterpret("float32", raw[1])),
                        K.shift_left(ue8m0_pack4(K.reinterpret("float32", raw[3])), K.uint32(8)),
                    )
                )
                K.ptx.st.shared.v4.u32(
                    sfb_tile.ptr_to([sstate.stage, lane * 4]), bpack0, bpack0, bpack0, bpack0
                )
                K.ptx.st.shared.v4.u32(
                    sfb_tile.ptr_to([sstate.stage, 128 + lane * 4]), bpack1, bpack1, bpack1, bpack1
                )
                K.cuda.warp_sync()
                K.ptx.fence.proxy.async_.shared__cta()
                with K.If(lane == 0), K.Then():
                    remote_arrive_leader(sf_bar.ptr_to([sstate.stage]))
                sstate.advance()
                with K.If(kb + 1 < nkb), K.Then():
                    load_regs(kb + 1)
            iket_end(tk_tile)

        with prod_role:
            K.ptx.fence.proxy.async_.global_()
            tstate = K.PipelineState(TASK_RING, phase=0)
            sstate = K.PipelineState(STAGES, phase=0)
            running = i32(K.int32(1))
            with K.If(is_leader):
                with K.Then():
                    with K.While(running == 1):
                        tidx = u32(K.uint32(0))
                        with K.If(lane == 0), K.Then():
                            K.ptx.atom.global_.add.u32(tidx, task_ctr.ptr_to([0]), K.uint32(1))
                        tix = K.local_scalar(K.i32)
                        shfl_idx(tix, K.cast(tidx, "int32"), K.int32(0))
                        kind = i32(
                            K.Select(
                                tix < n_g1,
                                K.int32(0),
                                K.Select(tix < n_g1 + n_g2, K.int32(1), K.int32(2)),
                            )
                        )
                        rel = i32(K.Select(kind == 0, tix, tix - n_g1))
                        ntn = K.Select(kind == 0, K.int32(NT1), K.int32(NT2))
                        mtg = i32(rel // ntn)
                        nt = i32(rel - mtg * ntn)

                        mp1 = K.local_scalar(K.i32)
                        K.ptx.ld.shared.s32(mp1, s_mpre.ptr_to([lane + 1]))
                        bal = K.local_scalar(K.u32)
                        K.ptx.vote_sync.ballot.b32(
                            bal, K.ptx.pred(K.cast(mp1 <= mtg, "uint32")), K.uint32(0xFFFFFFFF)
                        )
                        e = i32(K.min(K.cast(K.popcount(bal), "int32"), K.int32(NUM_LOCAL - 1)))
                        mp0 = K.local_scalar(K.i32)
                        K.ptx.ld.shared.s32(mp0, s_mpre.ptr_to([e]))
                        ecnt = K.local_scalar(K.i32)
                        K.ptx.ld.shared.s32(ecnt, s_ecnt.ptr_to([e]))
                        eoff = K.local_scalar(K.i32)
                        K.ptx.ld.shared.s32(eoff, s_eoff.ptr_to([e]))
                        mt = i32(mtg - mp0)
                        row0 = i32(eoff + mt * BMP)
                        valid = i32(K.max(K.min(ecnt - mt * BMP, K.int32(BMP)), K.int32(0)))
                        nkb = i32(K.Select(kind == 0, K.int32(KB1 // KPACK), K.int32(KB2 // KPACK)))
                        half = i32(K.cast(K.And(valid > 0, valid <= BM), "int32"))

                        K.cuda.mbarrier_wait(task_empty.ptr_to([tstate.stage]), tstate.phase ^ 1)
                        with K.If(lane == 0), K.Then():
                            K.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 0]), kind)
                            K.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 1]), e)
                            K.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 2]), mt)
                            K.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 3]), nt)
                            K.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 4]), row0)
                            K.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 5]), valid)
                            K.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 6]), nkb)
                            K.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 7]), half)
                            K.ptx.mbarrier.arrive.shared.b64(
                                task_full.ptr_to([tstate.stage]), K.uint32(1)
                            )
                            rem_bar = K.local_scalar(K.u64)
                            K.ptx.mapa.shared__cluster.u64(
                                rem_bar, task_full.ptr_to([tstate.stage]), K.uint32(1)
                            )
                            K.ptx.mbarrier.arrive.expect_tx.release.cluster.b64(
                                rem_bar, K.uint32(32), pred=K.bool(True)
                            )
                            mbar32 = K.local_scalar(K.u32)
                            mdst = K.local_scalar(K.u32)
                            K.ptx.mapa.shared__cluster.u32(
                                mbar32,
                                K.cuda.cvta_generic_to_shared(task_full.ptr_to([tstate.stage])),
                                K.uint32(1),
                            )
                            K.ptx.mapa.shared__cluster.u32(
                                mdst,
                                K.cuda.cvta_generic_to_shared(s_task.ptr_to([tstate.stage, 0])),
                                K.uint32(1),
                            )
                            K.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
                                mdst,
                                K.cast(kind, "uint32"),
                                K.cast(e, "uint32"),
                                K.cast(mt, "uint32"),
                                K.cast(nt, "uint32"),
                                mbar32,
                            )
                            K.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
                                mdst + K.uint32(16),
                                K.cast(row0, "uint32"),
                                K.cast(valid, "uint32"),
                                K.cast(nkb, "uint32"),
                                K.cast(half, "uint32"),
                                mbar32,
                            )
                        tstate.advance()
                        with K.If(kind == 2):
                            with K.Then():
                                K.assign(running, K.int32(0))
                            with K.Else():
                                with K.If(kind == 1), K.Then():
                                    tk_done = iket_range("prod-wait-done")
                                    wait_done(e, mt)
                                    iket_end(tk_done)
                                producer_loads(kind, e, nt, row0, nkb, sstate, half)
                with K.Else():
                    kind = i32(K.int32(0))
                    e = i32(K.int32(0))
                    mt = i32(K.int32(0))
                    nt = i32(K.int32(0))
                    row0 = i32(K.int32(0))
                    valid = i32(K.int32(0))
                    nkb = i32(K.int32(0))
                    half = i32(K.int32(0))
                    with K.While(running == 1):
                        read_task(tstate, kind, e, mt, nt, row0, valid, nkb, half)
                        with K.If(kind == 2):
                            with K.Then():
                                K.assign(running, K.int32(0))
                            with K.Else():
                                with K.If(kind == 1), K.Then():
                                    tk_done = iket_range("prod-wait-done")
                                    wait_done(e, mt)
                                    iket_end(tk_done)
                                producer_loads(kind, e, nt, row0, nkb, sstate, half)

        with mma_role:
            with K.If(is_leader), K.Then():
                el = K.local_scalar(K.u32)
                ell = K.local_scalar(K.u32)
                K.ptx.elect_sync(ell, el, K.uint32(0xFFFFFFFF))
                tstate = K.PipelineState(TASK_RING, phase=0)
                sstate = K.PipelineState(STAGES, phase=0)
                astate = K.PipelineState(1, phase=0)
                desc_sf = K.local_scalar(K.u64)
                sfa_smem = K.local_scalar(K.u32)
                sfb_smem = K.local_scalar(K.u32)
                K.assign(sfa_smem, K.cuda.cvta_generic_to_shared(sfa_tile.ptr_to([0, 0])))
                K.assign(sfb_smem, K.cuda.cvta_generic_to_shared(sfb_tile.ptr_to([0, 0])))
                kind = i32(K.int32(0))
                e = i32(K.int32(0))
                mt = i32(K.int32(0))
                nt = i32(K.int32(0))
                row0 = i32(K.int32(0))
                valid = i32(K.int32(0))
                nkb = i32(K.int32(0))
                half = i32(K.int32(0))
                running = i32(K.int32(1))
                with K.While(running == 1):
                    read_task(tstate, kind, e, mt, nt, row0, valid, nkb, half)
                    with K.If(kind == 2):
                        with K.Then():
                            K.assign(running, K.int32(0))
                        with K.Else():
                            tk_tile = iket_range("mma-tile")
                            tk_e = iket_range("mma-wait-tempty")
                            K.cuda.mbarrier_wait(tempty_bar.ptr_to([0]), astate.phase ^ 1)
                            iket_end(tk_e)
                            K.ptx.tcgen05.fence__after_thread_sync()
                            with K.serial(0, nkb) as kb:
                                tk_s = iket_range("mma-wait-sf")
                                K.cuda.mbarrier_wait(sf_bar.ptr_to([sstate.stage]), sstate.phase)
                                iket_end(tk_s)
                                K.ptx.tcgen05.fence__after_thread_sync()
                                sf_stage = K.cast(sstate.stage, "uint32")
                                K.assign(
                                    desc_sf,
                                    with_smem_addr(
                                        SF_DESC_BASE, sfa_smem + sf_stage * K.uint32(BM * 4)
                                    ),
                                )
                                K.ptx[UTCCP](tmem_base + K.uint32(SFA_TMEM_COL), desc_sf, pred=el)
                                K.assign(
                                    desc_sf,
                                    with_smem_addr(
                                        SF_DESC_BASE, sfb_smem + sf_stage * K.uint32(BN * 4)
                                    ),
                                )
                                K.ptx[UTCCP](tmem_base + K.uint32(SFB_TMEM_COL), desc_sf, pred=el)
                                K.assign(
                                    desc_sf,
                                    with_smem_addr(
                                        SF_DESC_BASE,
                                        sfb_smem + sf_stage * K.uint32(BN * 4) + K.uint32(128 * 4),
                                    ),
                                )
                                K.ptx[UTCCP](
                                    tmem_base + K.uint32(SFB_TMEM_COL + 4), desc_sf, pred=el
                                )
                                K.ptx.tcgen05.fence__before_thread_sync()
                                K.cuda.warp_sync()
                                K.ptx[COMMIT](
                                    sfempty_bar.ptr_to([sstate.stage]), K.uint16(3), pred=el
                                )
                                K.cuda.warp_sync()
                                for sub in range(KPACK):
                                    slot = sstate.stage * KPACK + sub
                                    tk_f = iket_range("mma-wait-full")
                                    K.cuda.mbarrier_wait(full_bar.ptr_to([slot]), sstate.phase)
                                    iket_end(tk_f)
                                    K.ptx.tcgen05.fence__after_thread_sync()
                                    a_desc, a_off = a_tiles[sub][sstate.stage].encode(
                                        major="k", mma_k=32
                                    )
                                    b_desc, b_off = b_tiles[sub][sstate.stage].encode(
                                        major="k", mma_k=32
                                    )
                                    for ki in range(4):
                                        accumulate = K.Or(kb > 0, K.Or(K.int32(sub) > 0, ki > 0))
                                        with K.If(half != 0):
                                            with K.Then():
                                                K.ptx[MMA](
                                                    tmem_base,
                                                    a_desc + a_off(ki),
                                                    b_desc + b_off(ki),
                                                    K.uint32(IDESC_H_IDS[sub]),
                                                    tmem_base + K.uint32(SFA_TMEM_COL),
                                                    tmem_base + K.uint32(SFB_TMEM_COL),
                                                    accumulate,
                                                    pred=el,
                                                )
                                            with K.Else():
                                                K.ptx[MMA](
                                                    tmem_base,
                                                    a_desc + a_off(ki),
                                                    b_desc + b_off(ki),
                                                    K.uint32(IDESC_IDS[sub]),
                                                    tmem_base + K.uint32(SFA_TMEM_COL),
                                                    tmem_base + K.uint32(SFB_TMEM_COL),
                                                    accumulate,
                                                    pred=el,
                                                )
                                    K.ptx.tcgen05.fence__before_thread_sync()
                                    K.cuda.warp_sync()
                                    K.ptx[COMMIT](empty_bar.ptr_to([slot]), K.uint16(3), pred=el)
                                    if sub == KPACK - 1:
                                        K.ptx[COMMIT](
                                            tfull_bar.ptr_to([0]),
                                            K.uint16(3),
                                            pred=K.And(el == K.uint32(1), kb == nkb - 1),
                                        )
                                    K.cuda.warp_sync()
                                sstate.advance()
                            astate.advance()
                            iket_end(tk_tile)

        with aux_role:
            aw = warp - 2
            K.ptx.fence.proxy.async_.global_()
            tstate = K.PipelineState(TASK_RING, phase=0)
            sstate = K.PipelineState(STAGES, phase=0)
            kind = i32(K.int32(0))
            e = i32(K.int32(0))
            mt = i32(K.int32(0))
            nt = i32(K.int32(0))
            row0 = i32(K.int32(0))
            valid = i32(K.int32(0))
            nkb = i32(K.int32(0))
            half = i32(K.int32(0))
            running = i32(K.int32(1))
            with K.While(running == 1):
                read_task(tstate, kind, e, mt, nt, row0, valid, nkb, half)
                with K.If(kind == 2):
                    with K.Then():
                        K.assign(running, K.int32(0))
                    with K.Else():
                        with K.If(kind == 1), K.Then():
                            wait_done(e, mt)
                        with K.If(aw == 0):
                            with K.Then():
                                scale_a_copies(kind, e, nt, row0, nkb, sstate, half)
                            with K.Else():
                                scale_loads(kind, e, nt, row0, nkb, sstate, half, aw)

        with math_role:
            mw = warp - MATH_WARP0
            wg = mw // 4
            row = (mw % 4) * 32 + lane
            tstate = K.PipelineState(TASK_RING, phase=0)
            astate = K.PipelineState(1, phase=0)
            kind = i32(K.int32(0))
            e = i32(K.int32(0))
            mt = i32(K.int32(0))
            nt = i32(K.int32(0))
            row0 = i32(K.int32(0))
            valid = i32(K.int32(0))
            nkb = i32(K.int32(0))
            half = i32(K.int32(0))
            acc = K.alloc_local((128,), K.f32)
            tile_ctr = i32(K.int32(0))
            running = i32(K.int32(1))

            def drain_tmem(cols):
                """Wait for the completed K reduction, drain its columns once, then release TMEM."""
                tk_f = iket_range("math-wait-tfull")
                K.cuda.mbarrier_wait(tfull_bar.ptr_to([0]), astate.phase)
                iket_end(tk_f)
                K.ptx.tcgen05.fence__after_thread_sync()
                for c, col in enumerate(cols):
                    taddr = tmem_base + K.uint32(col)
                    K.ptx[TMEM_LD32](*[acc[c * 32 + i] for i in range(32)], taddr)
                K.ptx.tcgen05.wait__ld.sync.aligned()
                K.ptx.tcgen05.fence__before_thread_sync()
                K.cuda.warp_sync()
                with K.If(lane == 0), K.Then():
                    remote_arrive_leader(tempty_bar.ptr_to([0]))
                astate.advance()

            L = (mw % 4) * 32 + lane
            hi = (mw % 4) // 2
            rowh = ((mw % 4) % 2) * 32 + lane

            def g1_mainloop(cols, sb_base0, sb_base1, arow):
                drain_tmem(cols)

            def g1_epilogue(nval, arow, is_valid, colbase, amax_slot, amax_others, a2s_writer):
                """SwiGLU over `nval` columns per thread, 128-column block amax via SMEM exchange, fp8 quant, act store."""
                h = K.alloc_local((64,), K.f32)
                amax = f32(K.float32(0.0))
                for j in range(nval):
                    uu = acc[nval + j]
                    sig = K.idioms.sigmoid_tanh_approx_f32(uu)
                    K.assign(h[j], K.Select(is_valid, uu * sig * acc[j], K.float32(0.0)))
                    K.assign(amax, K.max(amax, K.fabs(h[j])))
                K.ptx.st.shared.f32(s_amax.ptr_to([amax_slot]), amax)
                K.ptx.bar.sync(K.uint32(1), K.uint32((NWARPS - MATH_WARP0) * 32))
                for other in amax_others:
                    oam = K.local_scalar(K.f32)
                    K.ptx.ld.shared.f32(oam, s_amax.ptr_to([other]))
                    K.assign(amax, K.max(amax, oam))
                inv = f32(
                    K.Select(amax > K.float32(0.0), K.float32(FP8_MAX) / amax, K.float32(0.0))
                )
                q = K.alloc_local((16,), K.u32)
                for j in range(nval // 4):
                    lo16 = K.local_scalar(K.u16)
                    hi16 = K.local_scalar(K.u16)
                    K.ptx.cvt.rn.satfinite.e4m3x2.f32(lo16, h[4 * j + 1] * inv, h[4 * j] * inv)
                    K.ptx.cvt.rn.satfinite.e4m3x2.f32(hi16, h[4 * j + 3] * inv, h[4 * j + 2] * inv)
                    K.assign(
                        q[j],
                        K.bitwise_or(
                            K.cast(lo16, "uint32"),
                            K.shift_left(K.cast(hi16, "uint32"), K.uint32(16)),
                        ),
                    )
                abase = arow * INTER + nt * 128 + colbase
                for j in range(nval // 16):
                    K.ptx.st.global_.v4.b32(
                        act.ptr_to([abase + j * 16]),
                        q[4 * j],
                        q[4 * j + 1],
                        q[4 * j + 2],
                        q[4 * j + 3],
                    )
                with K.If(a2s_writer), K.Then():
                    sbyte = K.cast(ue8m0_pack4(amax * K.float32(1.0 / FP8_MAX)), "uint8")
                    ht = arow // BM
                    lr = arow - ht * BM
                    wp = (ht * A2_PACKS + nt // 2) * BM + (lr % 32) * 4 + lr // 32
                    K.ptx.st.global_.u8(a2p.ptr_to([wp * 4 + nt % 2]), sbyte)
                    ht2 = arow // (BM // 2)
                    lr2 = arow - ht2 * (BM // 2)
                    wh = (ht2 * A2_PACKS + nt // 2) * BM + (lr2 % 32) * 4 + lr2 // 32
                    K.ptx.st.global_.u8(a2h.ptr_to([wh * 4 + nt % 2]), sbyte)
                K.ptx.bar.sync(K.uint32(1), K.uint32((NWARPS - MATH_WARP0) * 32))
                with K.If(K.And(mw == 0, lane == 0)), K.Then():
                    K.ptx.red.release.gpu.global_.add.u32(
                        done.ptr_to([e * MAXMT + mt]), K.uint32(1)
                    )

            def g2_mainloop(cols, sb_base, arow):
                drain_tmem(cols)

            with K.While(running == 1):
                read_task(tstate, kind, e, mt, nt, row0, valid, nkb, half)
                with K.If(kind == 2):
                    with K.Then():
                        K.assign(running, K.int32(0))
                    with K.Else():
                        arow_p = i32(row0 + crank * BM + row)
                        is_valid_p = row + crank * BM < valid
                        arow_h = i32(row0 + crank * (BM // 2) + rowh)
                        valid_h = K.max(
                            K.min(valid - crank * (BM // 2), K.int32(BM // 2)), K.int32(0)
                        )
                        is_valid_h = rowh < valid_h
                        arow = i32(K.Select(half != 0, arow_h, arow_p))
                        par = K.bitwise_and(tile_ctr, K.int32(1))
                        tk_tile = iket_range("math-tile")
                        with K.If(kind == 0):
                            with K.Then():
                                sb_base0 = e * (32 * KB1) + nt * KB1
                                sb_base1 = e * (32 * KB1) + (16 + nt) * KB1
                                with K.If(half != 0):
                                    with K.Then():
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
                                                K.Select(K.And(w2 == wg, h2 == hi), slot_h, o)
                                                for (w2, h2), o in zip(
                                                    [(a, b) for a in range(2) for b in range(2)],
                                                    others,
                                                )
                                            ],
                                            K.And(wg == 0, hi == 0),
                                        )
                                        iket_end(tk_ep)
                                    with K.Else():
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
                            with K.Else():
                                tk_d = iket_range("math-wait-done")
                                dv = K.local_scalar(K.u32)
                                K.ptx.ld.acquire.gpu.global_.b32(dv, done.ptr_to([e * MAXMT + mt]))
                                with K.While(dv < K.uint32(PAIR * NT1)):
                                    K.ptx.ld.acquire.gpu.global_.b32(
                                        dv, done.ptr_to([e * MAXMT + mt])
                                    )
                                iket_end(tk_d)
                                wtok = f32(K.float32(0.0))
                                with K.If(arow < row0 + valid), K.Then():
                                    K.ptx.ld.global_.f32(wtok, sorted_w.ptr_to([arow]))
                                with K.If(half != 0):
                                    with K.Then():
                                        g2_mainloop(
                                            [wg * 64, wg * 64 + 32],
                                            e * (KB1 * KB2) + (nt * 2 + hi) * KB2,
                                            arow_h,
                                        )
                                        tk_ep = iket_range("math-epi-g2")
                                        pk = K.alloc_local((16,), K.u32)
                                        for j in range(16):
                                            lo = K.local_scalar(K.u16)
                                            hi8 = K.local_scalar(K.u16)
                                            K.ptx.cvt.rn.satfinite.e4m3x2.f32(
                                                lo, acc[4 * j + 1] * wtok, acc[4 * j] * wtok
                                            )
                                            K.ptx.cvt.rn.satfinite.e4m3x2.f32(
                                                hi8, acc[4 * j + 3] * wtok, acc[4 * j + 2] * wtok
                                            )
                                            K.assign(
                                                pk[j],
                                                K.bitwise_or(
                                                    K.cast(lo, "uint32"),
                                                    K.shift_left(
                                                        K.cast(hi8, "uint32"), K.uint32(16)
                                                    ),
                                                ),
                                            )
                                        cbase_h = nt * BN + hi * 128 + wg * 64
                                        dst_row_h = arow_h * HIDDEN + cbase_h
                                        for r in range(2):
                                            with K.If(lane // 16 == r), K.Then():
                                                for j in range(4):
                                                    K.ptx.st.shared.v4.b32(
                                                        s_epi.ptr_to([mw, lane % 16, j * 4]),
                                                        pk[4 * j],
                                                        pk[4 * j + 1],
                                                        pk[4 * j + 2],
                                                        pk[4 * j + 3],
                                                    )
                                                K.ptx.st.shared.s32(
                                                    s_epi.ptr_to([mw, lane % 16, 32]), dst_row_h
                                                )
                                                K.ptx.st.shared.s32(
                                                    s_epi.ptr_to([mw, lane % 16, 33]),
                                                    K.cast(is_valid_h, "int32"),
                                                )
                                            K.cuda.warp_sync()
                                            for q in range(2):
                                                f = q * 32 + lane
                                                row_i = f // 4
                                                piece = f % 4
                                                t0 = K.local_scalar(K.u32)
                                                t1 = K.local_scalar(K.u32)
                                                t2 = K.local_scalar(K.u32)
                                                t3 = K.local_scalar(K.u32)
                                                K.ptx.ld.shared.v4.b32(
                                                    t0,
                                                    t1,
                                                    t2,
                                                    t3,
                                                    s_epi.ptr_to([mw, row_i, piece * 4]),
                                                )
                                                doff = K.local_scalar(K.i32)
                                                dval = K.local_scalar(K.i32)
                                                K.ptx.ld.shared.s32(
                                                    doff, s_epi.ptr_to([mw, row_i, 32])
                                                )
                                                K.ptx.ld.shared.s32(
                                                    dval, s_epi.ptr_to([mw, row_i, 33])
                                                )
                                                with K.If(dval != 0), K.Then():
                                                    K.ptx.st.global_.v4.b32(
                                                        partial.ptr_to([doff + piece * 16]),
                                                        t0,
                                                        t1,
                                                        t2,
                                                        t3,
                                                    )
                                            K.cuda.warp_sync()
                                        iket_end(tk_ep)
                                    with K.Else():
                                        g2_mainloop(
                                            [wg * 128, wg * 128 + 32, wg * 128 + 64, wg * 128 + 96],
                                            e * (KB1 * KB2) + (nt * 2 + wg) * KB2,
                                            arow_p,
                                        )
                                        is_valid = is_valid_p

                                        tk_ep = iket_range("math-epi-g2")
                                        pk = K.alloc_local((32,), K.u32)
                                        for j in range(32):
                                            lo = K.local_scalar(K.u16)
                                            hi8 = K.local_scalar(K.u16)
                                            K.ptx.cvt.rn.satfinite.e4m3x2.f32(
                                                lo, acc[4 * j + 1] * wtok, acc[4 * j] * wtok
                                            )
                                            K.ptx.cvt.rn.satfinite.e4m3x2.f32(
                                                hi8, acc[4 * j + 3] * wtok, acc[4 * j + 2] * wtok
                                            )
                                            K.assign(
                                                pk[j],
                                                K.bitwise_or(
                                                    K.cast(lo, "uint32"),
                                                    K.shift_left(
                                                        K.cast(hi8, "uint32"), K.uint32(16)
                                                    ),
                                                ),
                                            )
                                        cbase = nt * BN + wg * 128
                                        dst_row = arow * HIDDEN + cbase
                                        for r in range(2):
                                            with K.If(lane // 16 == r), K.Then():
                                                for j in range(8):
                                                    K.ptx.st.shared.v4.b32(
                                                        s_epi.ptr_to([mw, lane % 16, j * 4]),
                                                        pk[4 * j],
                                                        pk[4 * j + 1],
                                                        pk[4 * j + 2],
                                                        pk[4 * j + 3],
                                                    )
                                                K.ptx.st.shared.s32(
                                                    s_epi.ptr_to([mw, lane % 16, 32]), dst_row
                                                )
                                                K.ptx.st.shared.s32(
                                                    s_epi.ptr_to([mw, lane % 16, 33]),
                                                    K.cast(is_valid, "int32"),
                                                )
                                            K.cuda.warp_sync()
                                            for q in range(4):
                                                f = q * 32 + lane
                                                row_i = f // 8
                                                piece = f % 8
                                                t0 = K.local_scalar(K.u32)
                                                t1 = K.local_scalar(K.u32)
                                                t2 = K.local_scalar(K.u32)
                                                t3 = K.local_scalar(K.u32)
                                                K.ptx.ld.shared.v4.b32(
                                                    t0,
                                                    t1,
                                                    t2,
                                                    t3,
                                                    s_epi.ptr_to([mw, row_i, piece * 4]),
                                                )
                                                doff = K.local_scalar(K.i32)
                                                dval = K.local_scalar(K.i32)
                                                K.ptx.ld.shared.s32(
                                                    doff, s_epi.ptr_to([mw, row_i, 32])
                                                )
                                                K.ptx.ld.shared.s32(
                                                    dval, s_epi.ptr_to([mw, row_i, 33])
                                                )
                                                with K.If(dval != 0), K.Then():
                                                    K.ptx.st.global_.v4.b32(
                                                        partial.ptr_to([doff + piece * 16]),
                                                        t0,
                                                        t1,
                                                        t2,
                                                        t3,
                                                    )
                                            K.cuda.warp_sync()
                                        iket_end(tk_ep)
                        iket_end(tk_tile)
                        K.assign(tile_ctr, tile_ctr + 1)

        K.cuda.iket.mark("G-done")
        emit_grid_sync(sync_ctr.ptr_to([0]), cta, num_ctas, tid)
        K.cuda.iket.mark("F-start")

        with aux_role:
            aw = warp - 2
            zt = i32(cta * 2 + aw)
            with K.While(zt < T):
                zc = K.local_scalar(K.i32)
                K.ptx.ld.global_.s32(zc, tok_cnt.ptr_to([zt]))
                with K.If(zc == 0), K.Then():
                    for j in range(HIDDEN // 256):
                        K.ptx.st.global_.v4.b32(
                            out.ptr_to([zt * HIDDEN + j * 256 + lane * 8]),
                            K.uint32(0),
                            K.uint32(0),
                            K.uint32(0),
                            K.uint32(0),
                        )
                K.assign(zt, zt + num_ctas * 2)

        with math_role:
            mw = warp - MATH_WARP0
            nmw = NWARPS - MATH_WARP0
            ngw = num_ctas * nmw
            nfin_u = K.local_scalar(K.u32)
            K.ptx.ld.global_.u32(nfin_u, task_ctr.ptr_to([2]))
            nfin = i32(K.cast(nfin_u, "int32"))
            fi = i32(cta * nmw + mw)
            ft = i32(K.int32(0))
            mypos = i32(K.int32(-1))
            posk = K.alloc_local((8,), K.i32)
            cpos = K.alloc_local((8,), K.i32)
            pw2 = K.alloc_local((FIN_G, FIN_RPP, 2, 2), K.u32)
            bacc2 = K.alloc_local((FIN_G, 2, 4), K.u32)

            def load_tok(dst_ft, dst_pos, idx):
                K.ptx.ld.global_.s32(dst_ft, fin_list.ptr_to([idx]))
                with K.If(lane < 8), K.Then():
                    K.ptx.ld.global_.s32(dst_pos, pos_of.ptr_to([dst_ft * TOPK + lane]))

            with K.If(fi < nfin), K.Then():
                load_tok(ft, mypos, fi)
            with K.While(fi < nfin):
                fi_n = i32(fi + ngw)
                ft_n = i32(K.int32(0))
                mypos_n = i32(K.int32(-1))
                with K.If(fi_n < nfin), K.Then():
                    load_tok(ft_n, mypos_n, fi_n)

                bal = K.local_scalar(K.u32)
                K.ptx.vote_sync.ballot.b32(
                    bal,
                    K.ptx.pred(K.cast(K.And(lane < 8, mypos >= 0), "uint32")),
                    K.uint32(0xFFFFFFFF),
                )
                fc = i32(K.cast(K.popcount(bal), "int32"))
                for k in range(8):
                    shfl_idx(posk[k], mypos, K.int32(k))
                for j in range(8):
                    K.assign(cpos[j], K.int32(-1))
                for k in range(8):
                    rk = i32(
                        K.cast(K.popcount(K.bitwise_and(bal, K.uint32((1 << k) - 1))), "int32")
                    )
                    vk = K.bitwise_and(K.shift_right(bal, K.uint32(k)), K.uint32(1)) != K.uint32(0)
                    for j in range(8):
                        K.assign(
                            cpos[j],
                            K.Select(K.And(vk, rk == K.int32(j)), posk[k] * HIDDEN, cpos[j]),
                        )
                for cbase in range(0, HIDDEN, 512 * FIN_G):
                    ng = min(FIN_G, (HIDDEN - cbase) // 512)
                    for g in range(ng):
                        for h in range(2):
                            for i in range(4):
                                K.assign(bacc2[g, h, i], K.uint32(0))

                    def fin_pass(r0):
                        rows = [r for r in range(r0, min(r0 + FIN_RPP, 8))]
                        for jj, r in enumerate(rows):
                            with K.If(cpos[r] >= 0), K.Then():
                                for g in range(ng):
                                    for h in range(2):
                                        col = cbase + g * 512 + h * 256 + lane * 8
                                        K.ptx.ld.global_.nc.v2.b32(
                                            pw2[g, jj, h, 0],
                                            pw2[g, jj, h, 1],
                                            partial.ptr_to([cpos[r] + col]),
                                        )
                        for jj, r in enumerate(rows):
                            with K.If(cpos[r] >= 0), K.Then():
                                for g in range(ng):
                                    for h in range(2):
                                        for q4 in range(2):
                                            lo2 = K.local_scalar(K.u32)
                                            hi2 = K.local_scalar(K.u32)
                                            raw4 = pw2[g, jj, h, q4]
                                            K.ptx.cvt.rn.bf16x2.e4m3x2(
                                                lo2,
                                                K.cast(
                                                    K.bitwise_and(raw4, K.uint32(0xFFFF)), "uint16"
                                                ),
                                            )
                                            K.ptx.cvt.rn.bf16x2.e4m3x2(
                                                hi2,
                                                K.cast(K.shift_right(raw4, K.uint32(16)), "uint16"),
                                            )
                                            K.ptx.add.rn.bf16x2(
                                                bacc2[g, h, 2 * q4], bacc2[g, h, 2 * q4], lo2
                                            )
                                            K.ptx.add.rn.bf16x2(
                                                bacc2[g, h, 2 * q4 + 1],
                                                bacc2[g, h, 2 * q4 + 1],
                                                hi2,
                                            )

                    for r0 in range(0, 8, FIN_RPP):
                        if r0 == 0:
                            fin_pass(0)
                        else:
                            with K.If(fc > r0), K.Then():
                                fin_pass(r0)
                    for g in range(ng):
                        for h in range(2):
                            col = cbase + g * 512 + h * 256 + lane * 8
                            K.ptx.st.global_.v4.b32(
                                out.ptr_to([ft * HIDDEN + col]),
                                bacc2[g, h, 0],
                                bacc2[g, h, 1],
                                bacc2[g, h, 2],
                                bacc2[g, h, 3],
                            )
                K.assign(fi, fi_n)
                K.assign(ft, ft_n)
                K.assign(mypos, mypos_n)

        K.cuda.iket.mark("F-done")
        K.ptx.barrier.cluster.arrive.release.aligned()
        K.ptx.barrier.cluster.wait.acquire.aligned()
        with K.If(warp == 2), K.Then():
            K.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](K.uint32(0), K.uint32(512))

        with K.If(tid == 0), K.Then():
            oldx = K.local_scalar(K.u32)
            K.ptx.atom.acq_rel.gpu.global_.add.u32(oldx, task_ctr.ptr_to([3]), K.uint32(1))
            with K.If(oldx == K.uint32(num_ctas - 1)), K.Then():
                K.ptx.st.global_.u32(task_ctr.ptr_to([3]), K.uint32(0))
                K.ptx.st.global_.u32(task_ctr.ptr_to([2]), K.uint32(0))

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
        raise SkipTest("CUDA is required for agent-evolved FP8 MoE")
    capability = torch.cuda.get_device_capability()
    runtime_arch = f"sm_{capability[0]}{capability[1]}a"
    if runtime_arch not in KERNEL_META["runtime_cuda_archs"]:
        raise SkipTest(
            "agent-evolved FP8 MoE requires one of "
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
        raise SkipTest("CUDA is required for agent-evolved FP8 MoE")
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
