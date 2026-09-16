# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved Alpha-MoE FP8 block-scale megakernel for B200 (Qwen3-Next TP4).

Unlike the neighbouring ``*_multishape`` kernels this module targets a SINGLE
pinned geometry, the official row
``alphamoe-qwen3-next-tp4-m128-e512-top10-k2048-i128-fp8``: 128 tokens, hidden
size 2048, intermediate size 128, 512 experts, top-10 routing, no shared
experts, block-128 FP32 scales, ``float8_e4m3fn`` weights and BF16 in/out.
Routes are SUPPLIED rather than computed: the kernel consumes the given
``topk_ids`` and their paired FP32 ``topk_weights`` in the given order and
never recomputes or renormalizes them.  Activation quantization, expert
alignment, both projections and the BF16 accumulation all happen inside the
timed call.

The selected kernel is ``split-grid-self-first`` from evolution run
``alphamoe_m128_e512_topk10_k2048_i128_fp8-20260915-183022``.  It is one
persistent 148-CTA tirx-lite launch (8 warps per CTA, one CTA per SM) split
into two roles:

Token CTAs (the last NTOK = 33 CTAs)
    Quantize ALL tokens (group-128 FP8, the exact Torch recipe with a fast-path
    correctly rounded division: reciprocal plus two FMA corrections) into the
    token-order ``xq[M, HID]`` / ``xs[M, KB]`` buffers, and are the only
    arrivers on the first grid barrier.  They hold no TMA loads while doing so,
    so their GPU-scope release does not queue behind a weight burst.

Compute CTAs (the first GCOMP = 115 CTAs)
    Store nothing before the first barrier.  Compute CTA ``c`` speculatively
    issues the first SPEC = 4 gate/up weight tiles of expert ``c`` right after
    launch, before the routes have landed; if that expert turns out to be
    unrouted a discard item consumes those ring stages and the CTA falls back
    to expert ``c + GCOMP`` or to a dynamic first item.  The static first item
    is self-sufficient: its BF16 token rows are bulk-copied into spare ring
    stages and quantized in-CTA into a resident 16-K-block FP8 B region, so the
    tcgen05 MMA can use it as the N=16 operand without waiting on any other
    CTA.  Route tables and the sorted token list are built by warps 2-7 after
    the static-item prologue, and a dynamic work queue hands out the remaining
    gate/up chunks followed by the half-size down items.

Phase G is warp specialized with dynamic work stealing over an 8-stage TMA
weight ring, gather4 token tiles and a 16-stage TMEM ring (weights are the
M=128 operand, tokens the N=16 operand), with exact FP32 scale products and a
SwiGLU/requantization epilogue.  Phase F performs a route-ordered packed
BF16x2 finalization of one token per CTA after a sense-reversal barrier.

Measured on GB200 through the locked evaluation harness against the packaged
FlashInfer ``trtllm_fp8_block_scale_routed_moe`` baseline: 59.984 us versus
112.537 us, a 1.8761x speedup, verdict STABLE over 22 correctness checks with
``max_rms_ratio`` 9.038e-05.  See ``README.md`` for the recorded row.
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

KERNEL_META = {
    "name": "agent_evolved_alphamoe_fp8_blockscale_qwen3next",
    "category": "agent_evolved",
    "runtime_cuda_archs": ["sm_100a"],
    "provenance": {
        "generator": "kda_flow",
        "run": "alphamoe_m128_e512_topk10_k2048_i128_fp8-20260915-183022",
        "selected_version": "split-grid-self-first",
    },
}

BK = 128
NT = 16
BM = 128
NWARPS = 8
MATH_WARP0 = 4
NMATH = 4 * 32
STAGES = 8
NTB = 16
TASK_RING = 4
TASK_W = 5 + NT
NCONS = 6
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
TMA_GATHER4 = (
    "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
    ".mbarrier::complete_tx::bytes.L2::cache_hint"
)
BULK_G2S = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
MMA = "tcgen05.mma.cta_group::1.kind::f8f6f4"
TMEM_LD16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
FMA_F32 = "fma.rn.f32"
BAR_MATH = 1
BAR_TAB = 2
BAR_QUANT = 3
NTABW = NWARPS - 2                          # warps 2-7 build the tables; producer/MMA warps keep streaming
GCOMP_DELTA = 33
DSPLIT = 2
POLL_WARP = 3
ROWS_PER_STAGE = 4                          # 4 KiB bf16 token rows per 16 KiB ring stage
SPEC = 4                                    # weight tiles of expert `cta` issued before the routes are known
SENSE_BIT = 0x80000000
# sync_ctr layout (one 128-byte line each so pollers of one counter do not slow the others):
# [FLAG] barrier-1 release flag, [WORK] dynamic work counter, [BAR2] barrier-2 sense counter,
# [TOKA] token-CTA arrival count.
FLAG = 0
WORK = 32
BAR2 = 64
TOKA = 96


def _f32(v):
    return txl.local_scalar(txl.f32, init=v)


def _i32(v):
    return txl.local_scalar(txl.i32, init=v)


def _bf16_lo(word):
    return txl.reinterpret("float32", txl.shift_left(word, txl.uint32(16)))


def _bf16_hi(word):
    return txl.reinterpret("float32", txl.bitwise_and(word, txl.uint32(0xFFFF0000)))


def build_kernel(G, M, TOPK, E, HID, INTER):
    NGU = 2 * INTER
    KB = HID // BK
    assert INTER == BK and HID % BK == 0 and KB % 4 == 0
    GCOMP = max(1, G - GCOMP_DELTA)
    NTOK = G - GCOMP
    assert NTOK >= 1
    CPT = HID // 256
    assert CPT % 2 == 0 and CPT >= 2
    P = M * TOPK
    PADP = P + NT
    MW = (M + 31) // 32
    IDS_PER_T = (P + 255) // 256
    NTABT = NTABW * 32
    BINS_PER_T = (E + NTABT - 1) // NTABT
    MASK_PER_T = (E * MW + 255) // 256
    TPC = (M + NTOK - 1) // NTOK
    NRND = (TPC + 1) // 2
    ROWB = HID * 2                                        # bytes of one bf16 token row
    NSTG_MAX = (NT + ROWS_PER_STAGE - 1) // ROWS_PER_STAGE
    assert ROWS_PER_STAGE * ROWB == BM * BK and NSTG_MAX < STAGES
    # the speculated stages must never overlap the staging stages of a 16-token static item
    # (SPEC=6 deadlocked the full row: the producer's tile/stage accounting desynchronizes)
    assert SPEC <= STAGES - NSTG_MAX
    assert PADP <= G * 256 and TOPK <= 32
    NT_GU = 2 * KB
    assert KB % DSPLIT == 0 and (KB // DSPLIT) % 2 == 0
    NT_D = KB // DSPLIT
    TILE_A = BM * BK
    TILE_B = NT * BK
    IDESC = encode_instr_descriptor_dense_uint32(
        M=BM, N=NT, K=32, d_dtype="float32", a_dtype="float8_e4m3fn",
        b_dtype="float8_e4m3fn", trans_a=False, trans_b=False, cta_group=1,
    )

    @txl.kernel(warps=NWARPS, arch="sm_100a", min_blocks_per_sm=1, grid=G)
    def alphamoe_kernel(
        topk_ids: txl.gptr[txl.i32],
        topk_w: txl.gptr[txl.f32],
        hidden: txl.gptr[txl.i32],
        w1s: txl.gptr[txl.f32],
        w2s: txl.gptr[txl.f32],
        out: txl.gptr[txl.i32],
        xq: txl.gptr[txl.i32],
        xs: txl.gptr[txl.f32],
        actq: txl.gptr[txl.u8],
        acts: txl.gptr[txl.f32],
        contrib: txl.gptr[txl.bf16],
        done: txl.gptr[txl.u32],
        sync_ctr: txl.gptr[txl.u32],
        tm_w1: txl.TensorMap,
        tm_w2: txl.TensorMap,
        tm_xq: txl.TensorMap,
        tm_act: txl.TensorMap,
        rsf: txl.f32,
    ):
        cta = txl.cta_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        tid = txl.thread_id()
        is_tok_cta = cta >= GCOMP

        # ---- cold loads: every CTA needs the routes; token CTAs also fetch their rows ----
        idv = txl.alloc_local((IDS_PER_T,), txl.i32)
        for i in range(IDS_PER_T):
            txl.assign(idv[i], txl.int32(0))
            p = tid + 256 * i
            with txl.If(p < P), txl.Then():
                txl.ptx.ld.global_.nc.s32(idv[i], topk_ids.ptr_to([p]))
        tq = tid % 128
        half = tid // 128
        tok_base = cta - GCOMP
        xw_all = txl.alloc_local((NRND, 8), txl.u32)
        for r in range(NRND):
            for i_ in range(8):
                txl.assign(xw_all[r, i_], txl.uint32(0))
        with txl.If(is_tok_cta), txl.Then():
            for r in range(NRND):
                t_r = tok_base + NTOK * (2 * r + half)
                with txl.If(t_r < M), txl.Then():
                    txl.ptx.ld.global_.nc.v4.b32(
                        xw_all[r, 0], xw_all[r, 1], xw_all[r, 2], xw_all[r, 3],
                        hidden.ptr_to([t_r * (HID // 2) + tq * 8]),
                    )
                    txl.ptx.ld.global_.nc.v4.b32(
                        xw_all[r, 4], xw_all[r, 5], xw_all[r, 6], xw_all[r, 7],
                        hidden.ptr_to([t_r * (HID // 2) + tq * 8 + 4]),
                    )
        sense0 = txl.local_scalar(txl.u32, init=txl.uint32(0))
        with txl.If(tid == 0), txl.Then():
            txl.ptx.ld.acquire.gpu.global_.b32(sense0, sync_ctr.ptr_to([BAR2]))

        smem = txl.smem_pool()
        a_tile = smem.alloc((STAGES, BM, BK), txl.f8e4m3, align=1024, swizzle=txl.SW128B)
        b_tile = smem.alloc((STAGES, NT, BK), txl.f8e4m3, align=1024, swizzle=txl.SW128B)
        fitem_b = smem.alloc((KB, NT, BK), txl.f8e4m3, align=1024, swizzle=txl.SW128B)
        s_xsl = smem.alloc((NT * KB,), txl.f32, align=16)
        s_ids = smem.alloc((IDS_PER_T * 256,), txl.i32, align=16)
        s_mask = smem.alloc((MASK_PER_T * 256,), txl.u32, align=16)
        s_cnt = smem.alloc((E,), txl.i32, align=16)
        s_off = smem.alloc((E,), txl.i32, align=16)
        s_cb = smem.alloc((E,), txl.i32, align=16)
        s_chunk_e = smem.alloc((PADP,), txl.i32, align=16)
        s_sorted = smem.alloc((PADP,), txl.i32, align=16)
        s_dyn = smem.alloc((PADP,), txl.i32, align=16)
        s_wsum = smem.alloc((3 * NTABW,), txl.i32, align=16)
        s_pos = smem.alloc((32,), txl.i32, align=16)
        s_task = smem.alloc((TASK_RING, TASK_W), txl.i32, align=16)
        s_misc = smem.alloc((8,), txl.i32, align=16)
        s_prod = smem.alloc((TASK_RING, 2 * KB, NT), txl.f32, align=16)
        s_amax = smem.alloc((NT, 4), txl.f32, align=16)
        tmem_slot = smem.alloc((1,), txl.u32, align=4)
        full_bar = txl.TMABar(smem, STAGES)
        empty_bar = txl.TCGen05Bar(smem, STAGES)
        tfull = txl.TCGen05Bar(smem, NTB)
        tempty = txl.MBarrier(smem, NTB)
        task_full = txl.MBarrier(smem, TASK_RING)
        task_empty = txl.MBarrier(smem, TASK_RING)
        scale_full = txl.MBarrier(smem, TASK_RING)
        scale_empty = txl.MBarrier(smem, TASK_RING)
        rel_bar = txl.MBarrier(smem, 1)
        stage_bar = txl.TMABar(smem, 1)                   # staged bf16 rows of the static item landed
        fitem_bar = txl.MBarrier(smem, 1)                 # fp8 B region + local scales written
        stage_free = txl.MBarrier(smem, 1)                # staging stages may be reused for weights
        tables_bar = txl.MBarrier(smem, 1)                # route tables complete (for the producer)

        full_bar.init(1)
        empty_bar.init(1)
        tfull.init(1)
        tempty.init(4)
        task_full.init(1)
        task_empty.init(NCONS)
        scale_full.init(1)
        scale_empty.init(4)
        rel_bar.init(1)
        stage_bar.init(1)
        fitem_bar.init(1)
        stage_free.init(1)
        tables_bar.init(1)
        txl.ptx.fence.mbarrier_init.release.cluster()
        with txl.If(warp == 1), txl.Then():
            txl.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                txl.address_of(tmem_slot[0]), txl.uint32(TMEM_COLS)
            )
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
        with txl.If(txl.And(warp == 0, cta < GCOMP)), txl.Then():
            with txl.If(lane == 0), txl.Then():
                txl.ptx.prefetch.tensormap(txl.address_of(tm_w1))
                txl.ptx.prefetch.tensormap(txl.address_of(tm_w2))
                txl.ptx.prefetch.tensormap(txl.address_of(tm_xq))
                txl.ptx.prefetch.tensormap(txl.address_of(tm_act))

        st = txl.PipelineState(STAGES, phase=0)
        tstate = txl.PipelineState(TASK_RING, phase=0)
        cur_e = _i32(txl.int32(0))
        cur_j = _i32(txl.int32(0))
        cur_kind = _i32(txl.int32(-1))
        cur_row0 = _i32(txl.int32(0))
        cur_ntok = _i32(txl.int32(0))
        nxt_idx = txl.local_scalar(txl.u32)
        rows = txl.alloc_local((NT,), txl.i32)
        toks = txl.alloc_local((NT,), txl.i32)
        for i_ in range(NT):
            txl.assign(rows[i_], txl.int32(0))
            txl.assign(toks[i_], txl.int32(0))
        static_ok = _i32(txl.int32(0))
        n_stage = _i32(txl.int32(0))

        # ---------------- ring issue helpers ----------------
        def issue_b(stage_idx, tile):
            """Token-side (N=16) tile via four gather4 requests: x_q token rows for GU items,
            compact activation rows for D items; rows beyond n_tok repeat the last valid row."""
            base = b_tile[stage_idx].ptr_to(0, 0)
            with txl.If(cur_kind == 0):
                with txl.Then():
                    for g in range(NT // 4):
                        txl.ptx[TMA_GATHER4](
                            txl.ptx.addr(base, 512 * g),
                            txl.address_of(tm_xq),
                            (tile % KB) * BK,
                            rows[4 * g], rows[4 * g + 1], rows[4 * g + 2], rows[4 * g + 3],
                            full_bar.ptr_to([stage_idx]),
                            txl.uint64(EVICT_LAST),
                        )
                with txl.Else():
                    for g in range(NT // 4):
                        txl.ptx[TMA_GATHER4](
                            txl.ptx.addr(base, 512 * g),
                            txl.address_of(tm_act),
                            txl.int32(0),
                            rows[4 * g], rows[4 * g + 1], rows[4 * g + 2], rows[4 * g + 3],
                            full_bar.ptr_to([stage_idx]),
                            txl.uint64(EVICT_LAST),
                        )

        def issue_a(tile, with_b, expect_b=False):
            """One ring stage: expect the stage bytes, the weight tile, and optionally the token tile
            (with_b=False: static item, whose B operand is the resident fitem_b region;
            expect_b=True: the token tile is issued later into the same stage)."""
            empty_bar.wait(st.stage, st.phase ^ 1)
            full_bar.arrive(st.stage, tx_count=TILE_A + (TILE_B if (with_b or expect_b) else 0))
            with txl.If(cur_kind == 0):
                with txl.Then():
                    txl.ptx[TMA_G2S_2D](
                        a_tile[st.stage].ptr_to(0, 0),
                        txl.address_of(tm_w1),
                        (tile % KB) * BK,
                        cur_e * NGU + (tile // KB) * BM,
                        full_bar.ptr_to([st.stage]),
                        txl.uint64(EVICT_FIRST),
                    )
                with txl.Else():
                    txl.ptx[TMA_G2S_2D](
                        a_tile[st.stage].ptr_to(0, 0),
                        txl.address_of(tm_w2),
                        txl.int32(0),
                        cur_e * HID + ((cur_kind - 1) * NT_D + tile) * BM,
                        full_bar.ptr_to([st.stage]),
                        txl.uint64(EVICT_FIRST),
                    )
            if with_b:
                issue_b(st.stage, tile)
            st.advance()

        def publish_current(with_toks=True):
            """Publish the current item (and its token ids) to the task ring."""
            if with_toks:
                task_empty.wait(tstate.stage, tstate.phase ^ 1)
            txl.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 0]), cur_e)
            txl.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 1]), cur_j)
            txl.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 2]), cur_kind)
            txl.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 3]), cur_row0)
            txl.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 4]), cur_ntok)
            if with_toks:
                for i in range(NT):
                    txl.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 5 + i]), toks[i])
            task_full.arrive(tstate.stage)
            tstate.advance()

        def mask_count(e, dst):
            txl.assign(dst, txl.int32(0))
            for w_ in range(MW):
                mwv0 = txl.local_scalar(txl.u32)
                txl.ptx.ld.shared.u32(mwv0, s_mask.ptr_to([e * MW + w_]))
                txl.assign(dst, dst + txl.cast(txl.popcount(mwv0), "int32"))


        def fetch_next():
            """Claim a dynamic item; its latency overlaps the current work."""
            txl.ptx.atom.relaxed.gpu.global_.add.u32(nxt_idx, sync_ctr.ptr_to([WORK]), txl.uint32(1))

        spec_done = _i32(txl.int32(0))                      # warp-uniform in warp 0 (read by all lanes)
        disc_stages = _i32(txl.int32(0))                    # ring stages consumed by a discard item
        with txl.If(txl.And(warp == 0, cta < GCOMP)), txl.Then():
            with txl.If(cta < E), txl.Then():
                txl.assign(spec_done, txl.int32(SPEC))
                txl.assign(cur_e, cta)
                txl.assign(cur_kind, txl.int32(0))
            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                fetch_next()
                with txl.If(cta < E), txl.Then():
                    # speculate that expert `cta` is routed (~91% of experts are): its first weight
                    # tiles go out before the routes even land; an unrouted expert costs one discard item
                    for tile in range(SPEC):
                        issue_a(txl.int32(tile), False)

        for i in range(MASK_PER_T):
            txl.ptx.st.shared.u32(s_mask.ptr_to([tid + 256 * i]), txl.uint32(0))
        with txl.If(tid < 8), txl.Then():
            txl.ptx.st.shared.s32(s_misc.ptr_to([tid]), txl.int32(0))
        # the resident B region and its scales: padding rows must not be uninitialized
        for i in range(KB * NT * BK // (256 * 16)):
            txl.ptx.st.shared.v4.b32(
                txl.ptx.addr(fitem_b[0].ptr_to(0, 0), (tid + 256 * i) * 16),
                txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0),
            )
        for i in range(NT * KB // 256):
            txl.ptx.st.shared.f32(s_xsl.ptr_to([tid + 256 * i]), txl.float32(0.0))
        txl.cuda.cta_sync()

        for i in range(IDS_PER_T):
            p = tid + 256 * i
            with txl.If(p < P), txl.Then():
                v = idv[i]
                txl.ptx.st.shared.s32(s_ids.ptr_to([p]), v)
                t_p = p // TOPK
                bit = txl.shift_left(txl.uint32(1), txl.cast(t_p % 32, "uint32"))
                old_m = txl.local_scalar(txl.u32)
                txl.ptx.atom.shared.or_.b32(old_m, s_mask.ptr_to([v * MW + t_p // 32]), bit)
        txl.cuda.cta_sync()                                              # routes landed, mask complete

        # ---------------- group-128 quantization of one 2048-wide row held in 8 u32 per thread ----------------
        def quantize_words(xw8, dst_fp8, dst_scale):
            """128 threads per row (tq = 0..127, 16 elements each, 8 threads per group).
            dst_fp8(qw[4]) stores the 16 fp8 codes, dst_scale(scale) runs on the group leader."""
            xf = txl.alloc_local((16,), txl.f32)
            for jx in range(8):
                txl.assign(xf[2 * jx], _bf16_lo(xw8[jx]))
                txl.assign(xf[2 * jx + 1], _bf16_hi(xw8[jx]))
            amax = _f32(txl.fabs(xf[0]))
            for jx in range(1, 16):
                txl.assign(amax, txl.max(amax, txl.fabs(xf[jx])))
            for m_ in (4, 2, 1):
                o = txl.local_scalar(txl.f32)
                txl.ptx.shfl_sync.bfly.b32(o, amax, txl.uint32(m_), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF))
                txl.assign(amax, txl.max(amax, o))
            scale = _f32(txl.max(amax, txl.float32(1.0e-8)) * txl.float32(INV_FP8_MAX))
            # x / scale: one reciprocal per group plus a residual correction; in this kernel's
            # operand range this is the correctly rounded quotient (the fast path of div.rn).
            nscale = _f32(txl.float32(0.0) - scale)
            rcp0 = txl.local_scalar(txl.f32)
            txl.ptx.rcp.approx.ftz.f32(rcp0, scale)
            rerr = txl.local_scalar(txl.f32)
            txl.ptx[FMA_F32](rerr, nscale, rcp0, txl.float32(1.0))
            rcp1 = txl.local_scalar(txl.f32)
            txl.ptx[FMA_F32](rcp1, rerr, rcp0, rcp0)
            yq = txl.alloc_local((16,), txl.f32)
            for jx in range(16):
                q0 = _f32(xf[jx] * rcp1)
                nq0 = _f32(txl.float32(0.0) - q0)
                err = txl.local_scalar(txl.f32)
                txl.ptx[FMA_F32](err, nq0, scale, xf[jx])
                y = txl.local_scalar(txl.f32)
                txl.ptx[FMA_F32](y, err, rcp1, q0)
                txl.assign(yq[jx], txl.min(txl.max(y, txl.float32(-FP8_MAX)), txl.float32(FP8_MAX)))
            h16 = txl.alloc_local((8,), txl.u16)
            for jx in range(8):
                txl.ptx.cvt.rn.satfinite.e4m3x2.f32(h16[jx], yq[2 * jx + 1], yq[2 * jx])
            qw = txl.alloc_local((4,), txl.u32)
            for jx in range(4):
                txl.assign(
                    qw[jx],
                    txl.bitwise_or(
                        txl.cast(h16[2 * jx], "uint32"),
                        txl.shift_left(txl.cast(h16[2 * jx + 1], "uint32"), txl.uint32(16)),
                    ),
                )
            dst_fp8(qw)
            with txl.If(tq % 8 == 0), txl.Then():
                dst_scale(scale)

        # ---------------- token CTAs: quantize their tokens, reset counters, arrive ----------------
        with txl.If(is_tok_cta), txl.Then():
            xw_c = [txl.local_scalar(txl.u32) for _ in range(8)]
            t_c = _i32(txl.int32(0))
            with txl.serial(0, NRND) as r_:                 # one copy of the quantization body
                for i_ in range(8):
                    v = xw_all[0, i_]
                    for r in range(1, NRND):
                        v = txl.Select(r_ == r, xw_all[r, i_], v)
                    txl.assign(xw_c[i_], v)
                txl.assign(t_c, tok_base + NTOK * (2 * r_ + half))
                with txl.If(t_c < M), txl.Then():

                    def st_row(qw):
                        txl.ptx.st.global_.L2__cache_hint.v4.b32(
                            xq.ptr_to([t_c * (HID // 4) + tq * 4]), qw[0], qw[1], qw[2], qw[3], txl.uint64(EVICT_LAST)
                        )

                    def st_scale(scale):
                        txl.ptx.st.global_.L2__cache_hint.f32(xs.ptr_to([t_c * KB + tq // 8]), scale, txl.uint64(EVICT_LAST))

                    quantize_words(xw_c, st_row, st_scale)
            txl.ptx.fence.proxy.async_.global_()
            txl.cuda.cta_sync()
            with txl.If(tid == 0), txl.Then():
                old_a = txl.local_scalar(txl.u32)
                txl.ptx.atom.acq_rel.gpu.global_.add.u32(old_a, sync_ctr.ptr_to([TOKA]), txl.uint32(1))
                with txl.If(old_a == txl.uint32(NTOK - 1)), txl.Then():
                    txl.ptx.red.release.gpu.global_.add.u32(sync_ctr.ptr_to([FLAG]), txl.uint32(1))

        # ---------------- compute CTAs, warp 0: static first item straight from the bitmask ----------------
        with txl.If(txl.And(warp == 0, cta < GCOMP)), txl.Then():
            e_s = _i32(txl.int32(-1))
            cnt_s = _i32(txl.int32(0))
            with txl.If(cta < E), txl.Then():
                mask_count(cta, cnt_s)
                txl.assign(e_s, txl.Select(cnt_s > 0, cta, txl.int32(-1)))
            with txl.If(txl.And(e_s < 0, spec_done > 0)), txl.Then():
                # speculated tiles of an unrouted expert: the MMA warp consumes their stages without work
                with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                    txl.assign(cur_kind, txl.int32(-2))
                    txl.assign(cur_ntok, spec_done)
                    txl.assign(cur_row0, txl.int32(0))
                    txl.assign(cur_j, txl.int32(0))
                    for i in range(NT):
                        txl.ptx.st.shared.s32(s_task.ptr_to([tstate.stage, 5 + i]), txl.int32(0))
                    publish_current(with_toks=False)
                txl.assign(disc_stages, spec_done)
                txl.assign(spec_done, txl.int32(0))
            with txl.If(txl.And(e_s < 0, cta + GCOMP < E)), txl.Then():
                mask_count(cta + GCOMP, cnt_s)
                # the fallback expert's staging rows must fit above the discarded stages
                with txl.If(txl.min(cnt_s, txl.int32(NT)) > (STAGES - SPEC - 1) * ROWS_PER_STAGE), txl.Then():
                    txl.assign(cnt_s, txl.int32(0))
                txl.assign(e_s, txl.Select(cnt_s > 0, cta + GCOMP, txl.int32(-1)))
            with txl.If(e_s >= 0), txl.Then():
                txl.assign(static_ok, txl.int32(1))
                txl.assign(cur_e, e_s)
                txl.assign(cur_j, txl.int32(-1))                    # resolved by consumers from s_cb
                txl.assign(cur_kind, txl.int32(0))
                txl.assign(cur_row0, txl.int32(-1))                 # marks the static item
                txl.assign(cur_ntok, txl.min(cnt_s, txl.int32(NT)))
                txl.assign(n_stage, (cur_ntok + (ROWS_PER_STAGE - 1)) // ROWS_PER_STAGE)
                # first chunk = the NT lowest token ids of the expert: warp-wide bit scan into the slot the
                # static item is published to (slot 1 after a discard item, else slot 0); warp-uniform
                static_slot = txl.Select(disc_stages > 0, txl.int32(1), txl.int32(0))
                lane_lt = txl.shift_left(txl.uint32(1), txl.cast(lane, "uint32")) - txl.uint32(1)
                rbase = _i32(txl.int32(0))
                for w_ in range(MW):
                    mwv = txl.local_scalar(txl.u32)
                    txl.ptx.ld.shared.u32(mwv, s_mask.ptr_to([e_s * MW + w_]))
                    mine = txl.bitwise_and(txl.shift_right(mwv, txl.cast(lane, "uint32")), txl.uint32(1)) == txl.uint32(1)
                    rk = rbase + txl.cast(txl.popcount(txl.bitwise_and(mwv, lane_lt)), "int32")
                    with txl.If(txl.And(mine, rk < NT)), txl.Then():
                        txl.ptx.st.shared.s32(s_task.ptr_to([static_slot, 5 + rk]), txl.int32(32 * w_) + lane)
                    txl.assign(rbase, rbase + txl.cast(txl.popcount(mwv), "int32"))
                txl.cuda.warp_sync()
                for i in range(NT):
                    txl.ptx.ld.shared.s32(toks[i], s_task.ptr_to([static_slot, 5 + txl.min(txl.int32(i), cur_ntok - 1)]))
                    txl.assign(rows[i], toks[i])
                with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                    txl.ptx.st.shared.s32(s_misc.ptr_to([2]), cur_ntok)
                    txl.ptx.st.shared.s32(s_misc.ptr_to([3]), txl.int32(1))
                    txl.ptx.st.shared.s32(s_misc.ptr_to([4]), n_stage)
                    txl.ptx.st.shared.s32(s_misc.ptr_to([5]), txl.Select(e_s == cta, spec_done, txl.int32(0)))
                    # the chunk's bf16 rows ride ahead of the weight burst into the top ring stages
                    stage_bar.arrive(0, tx_count=cur_ntok * ROWB)
                    for i in range(NT):
                        with txl.If(i < cur_ntok), txl.Then():
                            stg = STAGES - n_stage + i // ROWS_PER_STAGE
                            txl.ptx[BULK_G2S](
                                txl.ptx.addr(a_tile[stg].ptr_to(0, 0), (i % ROWS_PER_STAGE) * ROWB),
                                hidden.ptr_to([toks[i] * (HID // 2)]),
                                txl.uint32(ROWB),
                                stage_bar.ptr_to([0]),
                            )
                    publish_current(with_toks=False)
            with txl.If(e_s < 0), txl.Then():
                with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                    txl.ptx.st.shared.s32(s_misc.ptr_to([3]), txl.int32(0))
        txl.cuda.cta_sync()                                              # static decision visible

        # ---- warp 0: the static item's weight tiles fill the ring (staging stages after they are free) ----
        pref_issued = _i32(txl.int32(0))
        with txl.If(txl.And(warp == 0, static_ok == 1)), txl.Then():
            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                already = txl.local_scalar(txl.i32)
                txl.ptx.ld.shared.s32(already, s_misc.ptr_to([5]))
                # ring stages 0..SPEC-1 hold either this item's first tiles (already > 0) or a discard
                # item's tiles (disc_stages); the staging rows sit in the top n_stage stages; fill between
                n_pref = STAGES - n_stage - txl.Select(already > 0, txl.int32(0), disc_stages)
                for tile in range(STAGES):
                    with txl.If(txl.And(tile >= already, tile < n_pref)), txl.Then():
                        issue_a(txl.int32(tile), False)
                txl.assign(pref_issued, n_pref)

        # ---- warps 4-7: quantize the staged rows into the resident B region + local scales ----
        with txl.If(warp >= MATH_WARP0), txl.Then():
            has_static = txl.local_scalar(txl.i32)
            txl.ptx.ld.shared.s32(has_static, s_misc.ptr_to([3]))
            with txl.If(txl.And(has_static == 1, cta < GCOMP)), txl.Then():
                ntok_s = txl.local_scalar(txl.i32)
                nstg_s = txl.local_scalar(txl.i32)
                txl.ptx.ld.shared.s32(ntok_s, s_misc.ptr_to([2]))
                txl.ptx.ld.shared.s32(nstg_s, s_misc.ptr_to([4]))
                stage_bar.wait(0, 0)
                kb_q = tq // 8
                c_q = tq % 8
                # one runtime loop over the chunk's rows keeps a single copy of the quantization body
                with txl.serial(0, ntok_s) as i_r:
                    stg = STAGES - nstg_s + i_r // ROWS_PER_STAGE
                    xw8 = [txl.local_scalar(txl.u32) for _ in range(8)]
                    src = txl.ptx.addr(a_tile[stg].ptr_to(0, 0), (i_r % ROWS_PER_STAGE) * ROWB + tq * 32)
                    txl.ptx.ld.shared.v4.b32(xw8[0], xw8[1], xw8[2], xw8[3], src)
                    txl.ptx.ld.shared.v4.b32(xw8[4], xw8[5], xw8[6], xw8[7], txl.ptx.addr(src, 16))

                    def st_b(qw):
                        txl.ptx.st.shared.v4.b32(fitem_b[kb_q].ptr_to(i_r, c_q * 16), qw[0], qw[1], qw[2], qw[3])

                    def st_s(scale):
                        txl.ptx.st.shared.f32(s_xsl.ptr_to([i_r * KB + kb_q]), scale)

                    quantize_words(xw8, st_b, st_s)
                txl.ptx.fence.proxy.async_.shared__cta()
                txl.ptx.bar.sync(txl.uint32(BAR_QUANT), txl.uint32(NMATH))
                with txl.If(tq == 0), txl.Then():
                    fitem_bar.arrive(0)
                    stage_free.arrive(0)

        # ---------------- route tables (warps 2-7 under a named barrier; warps 0/1 keep streaming) ----------------
        def tab_bar():
            txl.ptx.bar.sync(txl.uint32(BAR_TAB), txl.uint32(NTABT))

        C = txl.local_scalar(txl.i32, init=txl.int32(0))
        CDYN = txl.local_scalar(txl.i32, init=txl.int32(0))
        with txl.If(warp >= 2), txl.Then():
            vt = tid - 64
            vw = warp - 2
            cbin = txl.alloc_local((BINS_PER_T,), txl.i32)
            nbin = txl.alloc_local((BINS_PER_T,), txl.i32)
            sbin = txl.alloc_local((BINS_PER_T,), txl.i32)
            tok_sum = _i32(txl.int32(0))
            ch_sum = _i32(txl.int32(0))
            dy_sum = _i32(txl.int32(0))
            for b in range(BINS_PER_T):
                e_b = vt * BINS_PER_T + b
                txl.assign(cbin[b], txl.int32(0))
                txl.assign(sbin[b], txl.int32(0))
                with txl.If(e_b < E), txl.Then():
                    mask_count(e_b, cbin[b])
                    # static items: chunk 0 of routed experts below GCOMP, and of routed experts
                    # GCOMP <= e < 2*GCOMP whose primary compute CTA (e - GCOMP) found its own expert unrouted
                    with txl.If(cbin[b] > 0), txl.Then():
                        with txl.If(e_b < GCOMP):
                            with txl.Then():
                                txl.assign(sbin[b], txl.int32(1))
                            with txl.Else():
                                with txl.If(e_b < 2 * GCOMP), txl.Then():
                                    cnt_lo = txl.local_scalar(txl.i32)
                                    mask_count(e_b - GCOMP, cnt_lo)
                                    txl.assign(sbin[b], txl.Select(cnt_lo == 0, txl.int32(1), txl.int32(0)))
                txl.assign(nbin[b], (cbin[b] + (NT - 1)) // NT)
                txl.assign(tok_sum, tok_sum + cbin[b])
                txl.assign(ch_sum, ch_sum + nbin[b])
                txl.assign(dy_sum, dy_sum + nbin[b] - sbin[b])
            incl_t = _i32(tok_sum)
            incl_c = _i32(ch_sum)
            incl_d = _i32(dy_sum)
            for d in (1, 2, 4, 8, 16):
                o_t = txl.local_scalar(txl.i32)
                o_c = txl.local_scalar(txl.i32)
                o_d = txl.local_scalar(txl.i32)
                txl.ptx.shfl_sync.up.b32(o_t, incl_t, txl.uint32(d), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                txl.ptx.shfl_sync.up.b32(o_c, incl_c, txl.uint32(d), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                txl.ptx.shfl_sync.up.b32(o_d, incl_d, txl.uint32(d), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                txl.assign(incl_t, incl_t + txl.Select(lane >= d, o_t, txl.int32(0)))
                txl.assign(incl_c, incl_c + txl.Select(lane >= d, o_c, txl.int32(0)))
                txl.assign(incl_d, incl_d + txl.Select(lane >= d, o_d, txl.int32(0)))
            with txl.If(lane == 31), txl.Then():
                txl.ptx.st.shared.s32(s_wsum.ptr_to([vw]), incl_t)
                txl.ptx.st.shared.s32(s_wsum.ptr_to([NTABW + vw]), incl_c)
                txl.ptx.st.shared.s32(s_wsum.ptr_to([2 * NTABW + vw]), incl_d)
            tab_bar()
            pre_t = _i32(txl.int32(0))
            pre_c = _i32(txl.int32(0))
            pre_d = _i32(txl.int32(0))
            tot_c = _i32(txl.int32(0))
            tot_d = _i32(txl.int32(0))
            for w in range(NTABW):
                v_t = txl.local_scalar(txl.i32)
                v_c = txl.local_scalar(txl.i32)
                v_d = txl.local_scalar(txl.i32)
                txl.ptx.ld.shared.s32(v_t, s_wsum.ptr_to([w]))
                txl.ptx.ld.shared.s32(v_c, s_wsum.ptr_to([NTABW + w]))
                txl.ptx.ld.shared.s32(v_d, s_wsum.ptr_to([2 * NTABW + w]))
                txl.assign(pre_t, pre_t + txl.Select(vw > w, v_t, txl.int32(0)))
                txl.assign(pre_c, pre_c + txl.Select(vw > w, v_c, txl.int32(0)))
                txl.assign(pre_d, pre_d + txl.Select(vw > w, v_d, txl.int32(0)))
                txl.assign(tot_c, tot_c + v_c)
                txl.assign(tot_d, tot_d + v_d)
            excl_t = _i32(pre_t + incl_t - tok_sum)
            excl_c = _i32(pre_c + incl_c - ch_sum)
            excl_d = _i32(pre_d + incl_d - dy_sum)
            for b in range(BINS_PER_T):
                e_b = vt * BINS_PER_T + b
                with txl.If(e_b < E), txl.Then():
                    txl.ptx.st.shared.s32(s_cnt.ptr_to([e_b]), cbin[b])
                    txl.ptx.st.shared.s32(s_off.ptr_to([e_b]), excl_t)
                    txl.ptx.st.shared.s32(s_cb.ptr_to([e_b]), excl_c)
                    with txl.serial(0, nbin[b]) as c_:
                        txl.ptx.st.shared.s32(s_chunk_e.ptr_to([excl_c + c_]), e_b)
                        with txl.If(c_ >= sbin[b]), txl.Then():
                            txl.ptx.st.shared.s32(s_dyn.ptr_to([excl_d + c_ - sbin[b]]), excl_c + c_)
                    # the expert's token list in increasing token id: walk the set bits of its mask
                    k_s = _i32(excl_t)
                    for w_ in range(MW):
                        xm = txl.local_scalar(txl.u32)
                        txl.ptx.ld.shared.u32(xm, s_mask.ptr_to([e_b * MW + w_]))
                        with txl.While(xm != txl.uint32(0)):
                            lowb = txl.bitwise_and(xm, txl.uint32(0) - xm)
                            txl.ptx.st.shared.s32(
                                s_sorted.ptr_to([k_s]), txl.int32(32 * w_) + txl.cast(txl.popcount(lowb - txl.uint32(1)), "int32")
                            )
                            txl.assign(k_s, k_s + 1)
                            txl.assign(xm, txl.bitwise_and(xm, xm - txl.uint32(1)))
                txl.assign(excl_t, excl_t + cbin[b])
                txl.assign(excl_c, excl_c + nbin[b])
                txl.assign(excl_d, excl_d + nbin[b] - sbin[b])
            tab_bar()
            with txl.If(vt == 0), txl.Then():
                txl.ptx.st.shared.s32(s_misc.ptr_to([0]), tot_c)
                txl.ptx.st.shared.s32(s_misc.ptr_to([1]), tot_d)
                tables_bar.arrive(0)

        def rank_of(t, e, dst):
            """dst = number of tokens < t routed to expert e (popcount of the mask bits below t)."""
            txl.assign(dst, txl.int32(0))
            tw = t // 32
            tb = txl.cast(t % 32, "uint32")
            for w_ in range(MW):
                mwv = txl.local_scalar(txl.u32)
                txl.ptx.ld.shared.u32(mwv, s_mask.ptr_to([e * MW + w_]))
                low = txl.Select(
                    tw > w_, txl.uint32(0xFFFFFFFF),
                    txl.Select(tw == w_, txl.shift_left(txl.uint32(1), tb) - txl.uint32(1), txl.uint32(0)),
                )
                txl.assign(dst, dst + txl.cast(txl.popcount(txl.bitwise_and(mwv, low)), "int32"))

        def token_positions(t, posr):
            """posr[r] = compact position of token t's r-th route in increasing expert order."""
            ek = txl.alloc_local((TOPK,), txl.i32)
            for k in range(TOPK):
                txl.ptx.ld.shared.s32(ek[k], s_ids.ptr_to([t * TOPK + k]))
            for k in range(TOPK):
                rk = txl.local_scalar(txl.i32)
                rank_of(t, ek[k], rk)
                oe = txl.local_scalar(txl.i32)
                txl.ptx.ld.shared.s32(oe, s_off.ptr_to([ek[k]]))
                r = _i32(txl.int32(0))
                for k2 in range(TOPK):
                    txl.assign(r, r + txl.cast(ek[k2] < ek[k], "int32"))
                for r_ in range(TOPK):
                    txl.assign(posr[r_], txl.Select(r == r_, oe + rk, posr[r_]))

        def chunk_row0(e, j, row0):
            oe = txl.local_scalar(txl.i32)
            cbe = txl.local_scalar(txl.i32)
            txl.ptx.ld.shared.s32(oe, s_off.ptr_to([e]))
            txl.ptx.ld.shared.s32(cbe, s_cb.ptr_to([e]))
            txl.assign(row0, oe + NT * (j - cbe))

        def item_ntok(e, row0):
            cnt_e = txl.local_scalar(txl.i32)
            txl.ptx.ld.shared.s32(cnt_e, s_cnt.ptr_to([e]))
            off_e = txl.local_scalar(txl.i32)
            txl.ptx.ld.shared.s32(off_e, s_off.ptr_to([e]))
            return txl.min(cnt_e - (row0 - off_e), txl.int32(NT))

        def resolve_static(e, j, row0):
            """The static item is published before the tables exist: j / row0 are -1 in its slot."""
            with txl.If(row0 < 0), txl.Then():
                txl.ptx.ld.shared.s32(row0, s_off.ptr_to([e]))
                txl.ptx.ld.shared.s32(j, s_cb.ptr_to([e]))

        def decode_item(idx):
            """Dynamic item ``idx``: idx < CDYN -> GU of chunk s_dyn[idx] (kind 0);
            else D split h of chunk (idx-CDYN)//DSPLIT (kind 1+h) over all chunks."""
            valid = idx < CDYN + DSPLIT * C
            is_gu = idx < CDYN
            txl.assign(cur_j, txl.int32(0))
            with txl.If(valid), txl.Then():
                with txl.If(is_gu):
                    with txl.Then():
                        txl.ptx.ld.shared.s32(cur_j, s_dyn.ptr_to([txl.max(idx, txl.int32(0))]))
                    with txl.Else():
                        txl.assign(cur_j, (idx - CDYN) // DSPLIT)
            txl.assign(cur_kind, txl.Select(valid, txl.Select(is_gu, txl.int32(0), txl.int32(1) + (idx - CDYN) % DSPLIT), txl.int32(-1)))
            txl.assign(cur_e, txl.int32(0))
            txl.assign(cur_row0, txl.int32(0))
            txl.assign(cur_ntok, txl.int32(0))
            with txl.If(valid), txl.Then():
                txl.ptx.ld.shared.s32(cur_e, s_chunk_e.ptr_to([cur_j]))
                chunk_row0(cur_e, cur_j, cur_row0)
                txl.assign(cur_ntok, item_ntok(cur_e, cur_row0))

        def rows_from_mask():
            """Rows / token ids of the current item: a GU chunk's tokens from the sorted list built in
            the table pass, a D item's rows as its compact activation positions."""
            for i in range(NT):
                pidx = cur_row0 + txl.min(txl.int32(i), cur_ntok - 1)
                txl.ptx.ld.shared.s32(toks[i], s_sorted.ptr_to([pidx]))
                txl.assign(rows[i], txl.Select(cur_kind == 0, toks[i], pidx))

        tmem_base = txl.local_scalar(txl.u32)
        txl.ptx.ld.shared.u32(tmem_base, tmem_slot.ptr_to([0]))

        roles = txl.specialize()
        prod_role = roles.role("prod", warps=[0])
        mma_role = roles.role("mma", warps=[1])
        aux_role = roles.role("aux", warps=[2])
        idle_role = roles.role("idle", warps=[POLL_WARP])
        math_role = roles.role("math", warps=list(range(MATH_WARP0, NWARPS)))

        def read_task(ts, e, j, kind, row0, ntok, single_lane=False, tok_dst=None):
            """Consumer side of the task ring: one arrive per consumer warp."""
            task_full.wait(ts.stage, ts.phase)
            txl.ptx.ld.shared.s32(e, s_task.ptr_to([ts.stage, 0]))
            txl.ptx.ld.shared.s32(j, s_task.ptr_to([ts.stage, 1]))
            txl.ptx.ld.shared.s32(kind, s_task.ptr_to([ts.stage, 2]))
            txl.ptx.ld.shared.s32(row0, s_task.ptr_to([ts.stage, 3]))
            txl.ptx.ld.shared.s32(ntok, s_task.ptr_to([ts.stage, 4]))
            if tok_dst is not None:
                txl.ptx.ld.shared.s32(tok_dst, s_task.ptr_to([ts.stage, 5 + txl.min(txl.int32(lane % NT), txl.max(ntok - 1, txl.int32(0)))]))
            if single_lane:
                task_empty.arrive(ts.stage)
            else:
                txl.cuda.warp_sync()
                with txl.If(lane == 0), txl.Then():
                    task_empty.arrive(ts.stage)
            ts.advance()

        # ---------------- idle warp: sorted list, release poll (compute CTAs), Phase-F positions ----------------
        with idle_role:
            with txl.If(cta < GCOMP), txl.Then():
                with txl.If(lane == 0), txl.Then():
                    flag = txl.local_scalar(txl.u32, init=txl.uint32(0))
                    txl.ptx.ld.acquire.gpu.global_.b32(flag, sync_ctr.ptr_to([FLAG]))
                    with txl.While(flag == txl.uint32(0)):
                        txl.ptx.ld.acquire.gpu.global_.b32(flag, sync_ctr.ptr_to([FLAG]))
                    rel_bar.arrive(0)
                txl.cuda.warp_sync()
            with txl.If(cta < M), txl.Then():
                posr_w = txl.alloc_local((TOPK,), txl.i32)
                for r_ in range(TOPK):
                    txl.assign(posr_w[r_], txl.int32(0))
                token_positions(cta, posr_w)
                with txl.If(lane == 0), txl.Then():
                    for r_ in range(TOPK):
                        txl.ptx.st.shared.s32(s_pos.ptr_to([r_]), posr_w[r_])

        # ---------------- auxiliary warp: exact scale products ----------------
        with aux_role:
            ats = txl.PipelineState(TASK_RING, phase=0)
            e = _i32(txl.int32(0))
            j = _i32(txl.int32(0))
            kind = _i32(txl.int32(0))
            row0 = _i32(txl.int32(0))
            running = _i32(txl.int32(1))
            with txl.While(running == 1):
                prod_stage = _i32(ats.stage)
                prod_phase = _i32(ats.phase)
                task_full.wait(ats.stage, ats.phase)
                txl.ptx.ld.shared.s32(e, s_task.ptr_to([ats.stage, 0]))
                txl.ptx.ld.shared.s32(j, s_task.ptr_to([ats.stage, 1]))
                txl.ptx.ld.shared.s32(kind, s_task.ptr_to([ats.stage, 2]))
                txl.ptx.ld.shared.s32(row0, s_task.ptr_to([ats.stage, 3]))
                ntok_slot = txl.local_scalar(txl.i32)
                txl.ptx.ld.shared.s32(ntok_slot, s_task.ptr_to([ats.stage, 4]))
                is_static = _i32(txl.Select(row0 < 0, txl.int32(1), txl.int32(0)))
                n_tok = _i32(txl.Select(kind >= 0, ntok_slot, txl.int32(1)))
                tokid = txl.local_scalar(txl.i32)
                txl.ptx.ld.shared.s32(
                    tokid, s_task.ptr_to([ats.stage, 5 + txl.min(txl.int32(lane % NT), n_tok - 1)])
                )
                txl.cuda.warp_sync()
                with txl.If(lane == 0), txl.Then():
                    task_empty.arrive(ats.stage)
                ats.advance()
                with txl.If(kind == -1):
                    with txl.Then():
                        txl.assign(running, txl.int32(0))
                    with txl.Else():
                        scale_empty.wait(prod_stage, prod_phase ^ 1)
                        tok = lane % NT
                        halfw = lane // NT
                        with txl.If(kind >= 0), txl.Then():
                            with txl.If(kind == 0):
                                with txl.Then():
                                    wv = _f32(txl.float32(0.0))
                                    txl.ptx.ld.global_.nc.f32(
                                        wv, w1s.ptr_to([e * (2 * KB) + lane])
                                    )
                                    xb = txl.alloc_local((4,), txl.u32)
                                    with txl.If(is_static == 1):
                                        with txl.Then():
                                            fitem_bar.wait(0, 0)
                                        with txl.Else():
                                            rel_bar.wait(0, 0)      # token rows / scales of other CTAs visible
                                    for q4 in range(KB // 4):
                                        with txl.If(is_static == 1):
                                            with txl.Then():
                                                txl.ptx.ld.shared.v4.b32(
                                                    xb[0], xb[1], xb[2], xb[3],
                                                    s_xsl.ptr_to([tok * KB + 4 * q4]),
                                                )
                                            with txl.Else():
                                                txl.ptx.ld.global_.nc.v4.b32(
                                                    xb[0], xb[1], xb[2], xb[3],
                                                    xs.ptr_to([tokid * KB + 4 * q4]),
                                                )
                                        for qi in range(4):
                                            kb = 4 * q4 + qi
                                            wb = _f32(txl.float32(0.0))
                                            txl.ptx.shfl_sync.idx.b32(
                                                wb, wv, txl.cast(halfw * NT + kb, "uint32"),
                                                txl.uint32(0x1F), txl.uint32(0xFFFFFFFF),
                                            )
                                            txl.ptx.st.shared.f32(
                                                s_prod.ptr_to([prod_stage, 2 * kb + halfw, tok]),
                                                wb * txl.reinterpret("float32", xb[qi]),
                                            )
                                with txl.Else():
                                    with txl.If(lane == 0), txl.Then():
                                        dv = txl.local_scalar(txl.u32)
                                        txl.ptx.ld.acquire.gpu.global_.b32(dv, done.ptr_to([j]))
                                        with txl.While(dv == txl.uint32(0)):
                                            txl.ptx.ld.acquire.gpu.global_.b32(dv, done.ptr_to([j]))
                                    txl.cuda.warp_sync()
                                    base = _f32(txl.float32(0.0))
                                    wsv = _f32(txl.float32(0.0))
                                    av = _f32(txl.float32(0.0))
                                    with txl.If(lane < NT), txl.Then():
                                        txl.ptx.ld.global_.nc.f32(wsv, w2s.ptr_to([e * KB + lane]))
                                        with txl.If(lane < n_tok), txl.Then():
                                            txl.ptx.ld.global_.f32(av, acts.ptr_to([row0 + lane]))
                                            txl.assign(base, av)
                                    with txl.serial(0, NT_D // 2) as rr:
                                        rb = (kind - 1) * NT_D + 2 * rr + halfw
                                        baseb = _f32(txl.float32(0.0))
                                        wsb = _f32(txl.float32(0.0))
                                        txl.ptx.shfl_sync.idx.b32(
                                            baseb, base, txl.cast(tok, "uint32"),
                                            txl.uint32(0x1F), txl.uint32(0xFFFFFFFF),
                                        )
                                        txl.ptx.shfl_sync.idx.b32(
                                            wsb, wsv, txl.cast(rb, "uint32"),
                                            txl.uint32(0x1F), txl.uint32(0xFFFFFFFF),
                                        )
                                        txl.ptx.st.shared.f32(
                                            s_prod.ptr_to([prod_stage, rb, tok]), baseb * wsb
                                        )
                        txl.cuda.warp_sync()
                        with txl.If(lane == 0), txl.Then():
                            scale_full.arrive(prod_stage)

        # ---------------- producer warp ----------------
        with prod_role:
            txl.ptx.fence.proxy.async_.global_()
            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                start_tile = _i32(txl.int32(0))
                # only a CTA without a static item can reach its first gather item before the release
                # flag is visible; a static CTA's second item starts ~10 us later, so it just waits
                need_rel = _i32(txl.Select(static_ok == 1, txl.int32(0), txl.int32(1)))
                with txl.If(cta >= GCOMP):
                    with txl.Then():
                        decode_item(txl.int32(0x7FFFFFFF))
                        publish_current()
                    with txl.Else():
                        with txl.If(static_ok == 1):
                            with txl.Then():
                                stage_free.wait(0, 0)                      # staging stages reusable
                                txl.assign(start_tile, pref_issued)
                            with txl.Else():
                                tables_bar.wait(0, 0)
                                txl.ptx.ld.shared.s32(C, s_misc.ptr_to([0]))
                                txl.ptx.ld.shared.s32(CDYN, s_misc.ptr_to([1]))
                                decode_item(txl.cast(nxt_idx, "int32"))
                                with txl.If(cur_kind >= 0), txl.Then():
                                    rows_from_mask()
                                publish_current()
                                fetch_next()
                running = _i32(txl.Select(cur_kind >= 0, txl.int32(1), txl.int32(0)))
                with txl.While(running == 1):
                    with txl.If(cur_kind >= 1), txl.Then():
                        dv = txl.local_scalar(txl.u32)
                        txl.ptx.ld.acquire.gpu.global_.b32(dv, done.ptr_to([cur_j]))
                        with txl.While(dv == txl.uint32(0)):
                            txl.ptx.ld.acquire.gpu.global_.b32(dv, done.ptr_to([cur_j]))
                        txl.ptx.fence.proxy.async_.global_()
                    n_tiles = txl.Select(cur_kind == 0, txl.int32(NT_GU), txl.int32(NT_D))
                    with txl.If(cur_row0 < 0):
                        with txl.Then():                                    # static item: resident B
                            with txl.serial(start_tile, n_tiles) as tile:
                                issue_a(tile, False)
                        with txl.Else():
                            with txl.If(need_rel == 1):
                                with txl.Then():
                                    # first gather item: its weight tiles fill the ring while the
                                    # release flag (seen by the idle warp) is awaited; token tiles follow
                                    npre = txl.min(n_tiles, txl.int32(STAGES))
                                    s0 = _i32(st.stage)
                                    with txl.serial(0, npre) as tile:
                                        issue_a(tile, False, expect_b=True)
                                    rel_bar.wait(0, 0)                      # other CTAs' token rows visible
                                    txl.ptx.fence.proxy.async_.global_()
                                    with txl.serial(0, npre) as tile:
                                        issue_b((s0 + tile) % STAGES, tile)
                                    with txl.serial(npre, n_tiles) as tile:
                                        issue_a(tile, True)
                                    txl.assign(need_rel, txl.int32(0))
                                with txl.Else():
                                    rel_bar.wait(0, 0)                      # other CTAs' token rows visible
                                    txl.ptx.fence.proxy.async_.global_()
                                    with txl.serial(start_tile, n_tiles) as tile:
                                        issue_a(tile, True)
                    txl.assign(start_tile, txl.int32(0))
                    tables_bar.wait(0, 0)
                    txl.ptx.ld.shared.s32(C, s_misc.ptr_to([0]))
                    txl.ptx.ld.shared.s32(CDYN, s_misc.ptr_to([1]))
                    decode_item(txl.cast(nxt_idx, "int32"))
                    with txl.If(cur_kind >= 0), txl.Then():
                        rows_from_mask()
                    publish_current()
                    fetch_next()
                    with txl.If(cur_kind < 0), txl.Then():
                        txl.assign(running, txl.int32(0))

        # ---------------- MMA warp ----------------
        with mma_role:
            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                sst = txl.PipelineState(STAGES, phase=0)
                tst = txl.PipelineState(NTB, phase=0)
                ts = txl.PipelineState(TASK_RING, phase=0)
                e = _i32(txl.int32(0))
                j = _i32(txl.int32(0))
                kind = _i32(txl.int32(0))
                row0 = _i32(txl.int32(0))
                ntok_m = _i32(txl.int32(0))

                def mma_issue(bview):
                    a_desc, a_off = a_tile[sst.stage].encode(major="k", mma_k=32)
                    b_desc, b_off = bview.encode(major="k", mma_k=32)
                    d_addr = tmem_base + txl.Cast("uint32", tst.stage) * txl.uint32(NT)
                    for ki in range(BK // 32):
                        txl.ptx[MMA](
                            d_addr,
                            a_desc + a_off(ki),
                            b_desc + b_off(ki),
                            txl.uint32(IDESC),
                            txl.uint32(0),
                            txl.uint32(0),
                            txl.uint32(0),
                            txl.uint32(0),
                            txl.ptx.pred(txl.uint32(1 if ki > 0 else 0)),
                        )

                def mma_tile(tile, static_b):
                    """static_b is a Python bool: the two item kinds get separate tile loops so the
                    per-tile path has a single B-descriptor computation and no branch."""
                    full_bar.wait(sst.stage, sst.phase)
                    tempty.wait(tst.stage, tst.phase ^ 1)
                    txl.ptx.tcgen05.fence__after_thread_sync()
                    if static_b:
                        mma_issue(fitem_b[tile % KB])                        # static item: resident B region
                    else:
                        mma_issue(b_tile[sst.stage])
                    txl.ptx.tcgen05.fence__before_thread_sync()
                    empty_bar.arrive(sst.stage)
                    tfull.arrive(tst.stage)
                    sst.advance()
                    tst.advance()

                running = _i32(txl.int32(1))
                with txl.While(running == 1):
                    read_task(ts, e, j, kind, row0, ntok_m, single_lane=True)
                    with txl.If(kind == -2), txl.Then():                       # discard: free the stages
                        with txl.serial(0, ntok_m) as tile:
                            full_bar.wait(sst.stage, sst.phase)
                            txl.ptx.tcgen05.fence__after_thread_sync()
                            txl.ptx.tcgen05.fence__before_thread_sync()
                            empty_bar.arrive(sst.stage)
                            sst.advance()
                    with txl.If(kind == -1):
                        with txl.Then():
                            txl.assign(running, txl.int32(0))
                        with txl.Else():
                            n_tiles = txl.Select(kind == 0, txl.int32(NT_GU), txl.int32(NT_D))
                            with txl.If(kind >= 0), txl.Then():
                                with txl.If(row0 < 0):
                                    with txl.Then():                            # static item
                                        fitem_bar.wait(0, 0)
                                        with txl.serial(0, n_tiles) as tile:
                                            mma_tile(tile, True)
                                    with txl.Else():
                                        with txl.serial(0, n_tiles) as tile:
                                            mma_tile(tile, False)

        # ---------------- math warps ----------------
        with math_role:
            mw = warp - MATH_WARP0
            tm = mw * 32 + lane
            tst = txl.PipelineState(NTB, phase=0)
            ts = txl.PipelineState(TASK_RING, phase=0)
            e = _i32(txl.int32(0))
            j = _i32(txl.int32(0))
            kind = _i32(txl.int32(0))
            row0 = _i32(txl.int32(0))
            ntok_slot = _i32(txl.int32(0))
            tok_lane = _i32(txl.int32(0))
            accg = txl.alloc_local((NT,), txl.f32)
            accu = txl.alloc_local((NT,), txl.f32)
            pv = txl.alloc_local((NT,), txl.f32)
            xsv = txl.alloc_local((NT,), txl.f32)

            def named_bar():
                txl.ptx.bar.sync(txl.uint32(BAR_MATH), txl.uint32(NMATH))

            pv2 = txl.alloc_local((NT,), txl.f32)

            def drain_pair():
                tfull.wait(tst.stage, tst.phase)
                txl.ptx.tcgen05.fence__after_thread_sync()
                st0 = _i32(tst.stage)
                taddr0 = tmem_base + txl.Cast("uint32", st0) * txl.uint32(NT)
                txl.ptx[TMEM_LD16](*[pv[i] for i in range(NT)], taddr0)
                tst.advance()
                tfull.wait(tst.stage, tst.phase)
                txl.ptx.tcgen05.fence__after_thread_sync()
                taddr1 = tmem_base + txl.Cast("uint32", tst.stage) * txl.uint32(NT)
                txl.ptx[TMEM_LD16](*[pv2[i] for i in range(NT)], taddr1)
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                txl.ptx.tcgen05.fence__before_thread_sync()
                txl.cuda.warp_sync()
                with txl.If(lane == 0), txl.Then():
                    tempty.arrive(st0)
                    tempty.arrive(tst.stage)
                tst.advance()

            def promote(acc, kb, regs, prod_stage):
                for q4 in range(NT // 4):
                    txl.ptx.ld.shared.v4.f32(
                        xsv[4 * q4], xsv[4 * q4 + 1], xsv[4 * q4 + 2], xsv[4 * q4 + 3],
                        s_prod.ptr_to([prod_stage, kb, 4 * q4]),
                    )
                for t_ in range(NT):
                    txl.assign(acc[t_], acc[t_] + regs[t_] * xsv[t_])

            running = _i32(txl.int32(1))
            with txl.While(running == 1):
                prod_stage = _i32(ts.stage)
                prod_phase = _i32(ts.phase)
                read_task(ts, e, j, kind, row0, ntok_slot, tok_dst=tok_lane)
                with txl.If(kind == -1):
                    with txl.Then():
                        txl.assign(running, txl.int32(0))
                    with txl.Else():
                        scale_full.wait(prod_stage, prod_phase)
                        n_tok = ntok_slot
                        with txl.If(kind == -2), txl.Then():
                            pass                                                # discard: keep the scale ring aligned
                        with txl.If(kind == 0):
                            with txl.Then():
                                for t_ in range(NT):
                                    txl.assign(accg[t_], txl.float32(0.0))
                                    txl.assign(accu[t_], txl.float32(0.0))
                                with txl.serial(0, KB // 2) as kk:
                                    drain_pair()
                                    promote(accg, 4 * kk, pv, prod_stage)
                                    promote(accg, 4 * kk + 2, pv2, prod_stage)
                                with txl.serial(0, KB // 2) as kk:
                                    drain_pair()
                                    promote(accu, 4 * kk + 1, pv, prod_stage)
                                    promote(accu, 4 * kk + 3, pv2, prod_stage)
                                resolve_static(e, j, row0)      # static item: tables are complete by now
                                hv = txl.alloc_local((NT,), txl.f32)
                                for t_ in range(NT):
                                    ex = txl.local_scalar(txl.f32)
                                    txl.ptx.ex2.approx.ftz.f32(ex, accg[t_] * txl.float32(-LOG2E))
                                    sig = txl.local_scalar(txl.f32)
                                    txl.ptx.rcp.approx.ftz.f32(sig, ex + txl.float32(1.0))
                                    txl.assign(hv[t_], accg[t_] * sig * accu[t_])
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
                                    txl.ptx.ld.shared.v4.f32(
                                        am0, am1, am2, am3, s_amax.ptr_to([lane, 0])
                                    )
                                    am = _f32(txl.max(txl.max(am0, am1), txl.max(am2, am3)))
                                    sc = _f32(
                                        txl.max(am, txl.float32(1.0e-8))
                                        * txl.float32(INV_FP8_MAX)
                                    )
                                    inv = txl.local_scalar(txl.f32)
                                    txl.ptx.rcp.approx.ftz.f32(inv, sc)
                                    txl.ptx.st.shared.v2.f32(
                                        s_amax.ptr_to([lane, 0]), sc, inv
                                    )
                                    with txl.If(lane < n_tok), txl.Then():
                                        tokid = tok_lane
                                        kk = _i32(txl.int32(0))
                                        for k in range(TOPK):
                                            idk = txl.local_scalar(txl.i32)
                                            txl.ptx.ld.shared.s32(idk, s_ids.ptr_to([tokid * TOPK + k]))
                                            txl.assign(kk, txl.Select(idk == e, txl.int32(k), kk))
                                        rw = txl.local_scalar(txl.f32)
                                        txl.ptx.ld.global_.nc.f32(
                                            rw, topk_w.ptr_to([tokid * TOPK + kk])
                                        )
                                        txl.ptx.st.global_.L2__cache_hint.f32(
                                            acts.ptr_to([row0 + lane]), (sc * rw) * rsf,
                                            txl.uint64(EVICT_LAST),
                                        )
                                named_bar()
                                for t_ in range(NT):
                                    sc = txl.local_scalar(txl.f32)
                                    inv = txl.local_scalar(txl.f32)
                                    txl.ptx.ld.shared.v2.f32(
                                        sc, inv, s_amax.ptr_to([t_, 0])
                                    )
                                    with txl.If(t_ < n_tok), txl.Then():
                                        y = _f32(hv[t_] * inv)
                                        yc = txl.min(txl.max(y, txl.float32(-FP8_MAX)), txl.float32(FP8_MAX))
                                        qb = txl.local_scalar(txl.u16)
                                        txl.ptx.cvt.rn.satfinite.e4m3x2.f32(qb, txl.float32(0.0), yc)
                                        txl.ptx.st.global_.L2__cache_hint.u8(
                                            actq.ptr_to([(row0 + t_) * INTER + tm]), txl.cast(qb, "uint8"),
                                            txl.uint64(EVICT_LAST),
                                        )
                                txl.ptx.fence.proxy.async_.global_()
                                named_bar()
                                with txl.If(tm == 0), txl.Then():
                                    txl.ptx.red.release.gpu.global_.add.u32(done.ptr_to([j]), txl.uint32(1))
                            with txl.Else():
                              with txl.If(kind >= 1), txl.Then():
                                rb0 = (kind - 1) * NT_D

                                def d_store(rb, regs):
                                    for q4 in range(NT // 4):
                                        txl.ptx.ld.shared.v4.f32(
                                            xsv[4 * q4], xsv[4 * q4 + 1],
                                            xsv[4 * q4 + 2], xsv[4 * q4 + 3],
                                            s_prod.ptr_to([prod_stage, rb0 + rb, 4 * q4]),
                                        )
                                    for t_ in range(NT):
                                        with txl.If(t_ < n_tok), txl.Then():
                                            rounded = txl.local_scalar(txl.u16)
                                            txl.ptx.cvt.rn.bf16.f32(
                                                rounded, regs[t_] * xsv[t_]
                                            )
                                            txl.ptx.st.global_.L2__cache_hint.b16(
                                                contrib.ptr_to([(row0 + t_) * HID + (rb0 + rb) * BM + tm]),
                                                rounded,
                                                txl.uint64(EVICT_LAST),
                                            )

                                with txl.serial(0, NT_D // 2) as rr:
                                    drain_pair()
                                    d_store(2 * rr, pv)
                                    d_store(2 * rr + 1, pv2)
                        txl.cuda.warp_sync()
                        with txl.If(lane == 0), txl.Then():
                            scale_empty.arrive(prod_stage)

        # ---------------- second grid barrier (sense reversal on sync_ctr[2]) and Phase F ----------------
        def arrive_release():
            with txl.If(cta == 0):
                with txl.Then():
                    txl.ptx.red.release.gpu.global_.add.u32(
                        sync_ctr.ptr_to([BAR2]), txl.uint32(SENSE_BIT - (G - 1))
                    )
                with txl.Else():
                    txl.ptx.red.release.gpu.global_.add.u32(sync_ctr.ptr_to([BAR2]), txl.uint32(1))

        def poll_release(target_bit):
            cur = txl.local_scalar(txl.u32)
            txl.ptx.ld.acquire.gpu.global_.b32(cur, sync_ctr.ptr_to([BAR2]))
            with txl.While(txl.bitwise_and(cur, txl.uint32(SENSE_BIT)) != target_bit):
                with txl.If(is_tok_cta), txl.Then():
                    txl.cuda.nano_sleep(txl.uint64(256))
                txl.ptx.ld.acquire.gpu.global_.b32(cur, sync_ctr.ptr_to([BAR2]))

        target2 = txl.bitwise_xor(txl.bitwise_and(sense0, txl.uint32(SENSE_BIT)), txl.uint32(SENSE_BIT))
        txl.cuda.cta_sync()
        with txl.If(tid == 0), txl.Then():
            arrive_release()
            poll_release(target2)
        txl.cuda.cta_sync()
        with txl.If(warp == 1), txl.Then():
            txl.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](tmem_base, txl.uint32(TMEM_COLS))
        # every item of this launch is complete: reset the per-chunk completion counters for the next launch
        with txl.If(cta + G * tid < PADP), txl.Then():
            txl.ptx.st.global_.u32(done.ptr_to([cta + G * tid]), txl.uint32(0))
        with txl.If(txl.And(cta == 0, tid == 0)), txl.Then():
            # every CTA has passed barrier 1 and its dynamic claims: reset for the next launch
            txl.ptx.st.global_.u32(sync_ctr.ptr_to([FLAG]), txl.uint32(0))
            txl.ptx.st.global_.u32(sync_ctr.ptr_to([WORK]), txl.uint32(0))
            txl.ptx.st.global_.u32(sync_ctr.ptr_to([TOKA]), txl.uint32(0))
        posr = txl.alloc_local((TOPK,), txl.i32)
        for r_ in range(TOPK):
            txl.ptx.ld.shared.s32(posr[r_], s_pos.ptr_to([r_]))
        t_var = _i32(cta)
        with txl.While(t_var < M):
            t = t_var
            if M > G:
                with txl.If(t != cta), txl.Then():
                    token_positions(t, posr)
            col0 = tid * CPT
            racc = txl.alloc_local((CPT // 2,), txl.u32)
            for c2 in range(CPT // 2):
                txl.assign(racc[c2], txl.uint32(0))
            cw = txl.alloc_local((TOPK, CPT // 2), txl.u32)
            for r_ in range(TOPK):
                for c4 in range(CPT // 8):
                    txl.ptx.ld.global_.v4.b32(
                        cw[r_, 4 * c4], cw[r_, 4 * c4 + 1], cw[r_, 4 * c4 + 2], cw[r_, 4 * c4 + 3],
                        contrib.ptr_to([posr[r_] * HID + col0 + 8 * c4]),
                    )
                if CPT % 8 == 4:
                    txl.ptx.ld.global_.v2.b32(
                        cw[r_, CPT // 2 - 2], cw[r_, CPT // 2 - 1],
                        contrib.ptr_to([posr[r_] * HID + col0 + CPT - 4]),
                    )
            for r_ in range(TOPK):
                for c2 in range(CPT // 2):
                    txl.ptx.add.rn.bf16x2(racc[c2], racc[c2], cw[r_, c2])
            obase = t * (HID // 2) + col0 // 2
            for c4 in range(CPT // 8):
                txl.ptx.st.global_.v4.b32(out.ptr_to([obase + 4 * c4]), racc[4 * c4], racc[4 * c4 + 1], racc[4 * c4 + 2], racc[4 * c4 + 3])
            if CPT % 8 == 4:
                txl.ptx.st.global_.v2.b32(out.ptr_to([obase + CPT // 2 - 2]), racc[CPT // 2 - 2], racc[CPT // 2 - 1])
            txl.assign(t_var, t_var + G)

    return alphamoe_kernel


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


# ---------------------------------------------------------------------------
# Registered benchmark surface
# ---------------------------------------------------------------------------

NUM_TOKENS = 128
HIDDEN = 2048
INTERMEDIATE = 128
NUM_EXPERTS = 512
NUM_SHARED_EXPERTS = 0
TOPK = 10
BLOCK_SIZE = 128
FP8_MAX = 448.0
MAX_CTAS = 148

# The official row's input distribution (flashinfer-bench-evolve alphamoe task).
HIDDEN_STATES_STD = 0.25
WEIGHT_STD = 0.125
OFFICIAL_SEED = 42
OFFICIAL_ROW = "alphamoe-qwen3-next-tp4-m128-e512-top10-k2048-i128-fp8"


@dataclass(frozen=True, slots=True)
class AlphaMoEConfig:
    label: str = "m128_official"
    num_tokens: int = NUM_TOKENS
    seed: int = OFFICIAL_SEED
    routed_scaling_factor: float = 1.0
    balancedness: float = 1.0

    def validate(self) -> None:
        if self.num_tokens != NUM_TOKENS:
            raise ValueError(
                f"this kernel is specialized for num_tokens={NUM_TOKENS}, got {self.num_tokens}"
            )
        if self.routed_scaling_factor <= 0:
            raise ValueError("routed_scaling_factor must be positive")
        if not 0.0 < self.balancedness <= 1.0:
            raise ValueError(f"balancedness must be in (0, 1], got {self.balancedness}")


CONFIGS = [
    {
        "label": "m128_official",
        "num_tokens": NUM_TOKENS,
        "seed": OFFICIAL_SEED,
        "routed_scaling_factor": 1.0,
        "balancedness": 1.0,
    },
    # A hot routed expert 0 and a non-unit scaling factor: both exercise the
    # multi-chunk expert path and the route-weight multiplication that the
    # evolution run's correctness failures came from.
    {
        "label": "m128_hot_expert",
        "num_tokens": NUM_TOKENS,
        "seed": 7,
        "routed_scaling_factor": 2.5,
        "balancedness": 0.25,
    },
]

BENCH_CONFIGS = [
    {
        "label": "m128_official",
        "num_tokens": NUM_TOKENS,
        "seed": OFFICIAL_SEED,
        "routed_scaling_factor": 1.0,
        "balancedness": 1.0,
    }
]


def _cfg(**kwargs: Any) -> AlphaMoEConfig:
    names = {field.name for field in fields(AlphaMoEConfig)}
    cfg = AlphaMoEConfig(**{name: value for name, value in kwargs.items() if name in names})
    cfg.validate()
    return cfg


def _num_ctas(**kwargs: Any) -> int:
    """The persistent grid is one CTA per SM, capped at MAX_CTAS."""
    if "num_ctas" in kwargs:
        value = int(kwargs["num_ctas"])
    else:
        from tirx_kernels.runner import hardware_num_sms

        value = min(hardware_num_sms(), MAX_CTAS)
    if not 1 <= value <= MAX_CTAS:
        raise ValueError(f"num_ctas must be in [1, {MAX_CTAS}], got {value}")
    return value


def get_kernel(**kwargs: Any):
    _cfg(**kwargs)
    return build_kernel(
        _num_ctas(**kwargs), NUM_TOKENS, TOPK, NUM_EXPERTS, HIDDEN, INTERMEDIATE
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
# Inputs (the official row's generator)
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
    generator: torch.Generator, device: torch.device, balancedness: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded top-k routing; ``balancedness`` < 1 makes routed expert 0 hot."""
    routed_top_k = TOPK - NUM_SHARED_EXPERTS
    routed_experts = NUM_EXPERTS - NUM_SHARED_EXPERTS
    scores = torch.randn(
        (NUM_TOKENS, routed_experts), dtype=torch.float32, device=device, generator=generator
    )
    scores[:, 0] += (1.0 - balancedness) * 6.0
    topk_ids = torch.topk(scores, routed_top_k, dim=-1).indices.to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(
            (NUM_TOKENS, TOPK), dtype=torch.float32, device=device, generator=generator
        ),
        dim=-1,
    )
    return topk_ids.contiguous(), topk_weights.contiguous()


def _allocate_kernel_state(case: dict[str, Any]) -> None:
    """Scratch buffers and tensor maps; allocated once and reused per launch."""
    device = case["hidden_states"].device
    kb = HIDDEN // BLOCK_SIZE
    padded_pairs = NUM_TOKENS * TOPK + NT

    case["xq"] = torch.zeros(NUM_TOKENS * HIDDEN, dtype=torch.float8_e4m3fn, device=device)
    case["xs"] = torch.zeros(NUM_TOKENS * kb, dtype=torch.float32, device=device)
    case["actq"] = torch.zeros(
        padded_pairs * INTERMEDIATE, dtype=torch.float8_e4m3fn, device=device
    )
    case["acts"] = torch.zeros(padded_pairs, dtype=torch.float32, device=device)
    case["contrib"] = torch.zeros(padded_pairs * HIDDEN, dtype=torch.bfloat16, device=device)
    case["done"] = torch.zeros(padded_pairs, dtype=torch.uint32, device=device)
    case["sync_ctr"] = torch.zeros(128, dtype=torch.uint32, device=device)

    case["tensor_maps"] = {
        "w1": _encode_2d(
            case["gemm1_weights"],
            "float8_e4m3fn",
            HIDDEN,
            NUM_EXPERTS * 2 * INTERMEDIATE,
            HIDDEN,
            BLOCK_SIZE,
            BM,
            3,
        ),
        "w2": _encode_2d(
            case["gemm2_weights"],
            "float8_e4m3fn",
            INTERMEDIATE,
            NUM_EXPERTS * HIDDEN,
            INTERMEDIATE,
            BLOCK_SIZE,
            BM,
            3,
        ),
        "xq": _encode_2d(
            case["xq"], "float8_e4m3fn", HIDDEN, NUM_TOKENS, HIDDEN, BLOCK_SIZE, 1, 3
        ),
        "act": _encode_2d(
            case["actq"],
            "float8_e4m3fn",
            INTERMEDIATE,
            padded_pairs,
            INTERMEDIATE,
            BLOCK_SIZE,
            1,
            3,
        ),
    }


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    device = torch.device(kwargs.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved Alpha-MoE")
    num_ctas = _num_ctas(**kwargs)
    actual_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if num_ctas > actual_sms:
        raise ValueError(
            f"kernel was built for {num_ctas} CTAs but the GPU has only {actual_sms} SMs"
        )

    generator = torch.Generator(device=device).manual_seed(int(cfg.seed))
    hidden_states = (
        torch.randn(
            (NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device, generator=generator
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
    topk_ids, topk_weights = _make_routing(generator, device, cfg.balancedness)

    case: dict[str, Any] = {
        "config": cfg,
        "num_ctas": num_ctas,
        "hidden_states": hidden_states,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "gemm1_weights": gemm1_weights,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
        "output": torch.empty(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device),
    }
    _allocate_kernel_state(case)
    return case


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    cfg: AlphaMoEConfig = case["config"]
    for name in (
        "hidden_states",
        "topk_ids",
        "topk_weights",
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
        case["topk_ids"].view(-1),
        case["topk_weights"].view(-1),
        case["hidden_states"].view(torch.int32).view(-1),
        case["gemm1_weights_scale"].view(-1),
        case["gemm2_weights_scale"].view(-1),
        case["output"].view(torch.int32).view(-1),
        case["xq"].view(torch.int32),
        case["xs"],
        case["actq"].view(torch.uint8),
        case["acts"],
        case["contrib"],
        case["done"],
        case["sync_ctr"],
        maps["w1"].ptr,
        maps["w2"].ptr,
        maps["xq"].ptr,
        maps["act"].ptr,
        float(cfg.routed_scaling_factor),
    )


def _launcher(executable, case: dict[str, Any]):
    args = _tirx_args(case)

    def launch() -> None:
        executable(*args)

    launch._keep_alive = (args, case)
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
    x_q, x_scale = _quantize_per_token_group(case["hidden_states"])
    topk_ids = case["topk_ids"]
    topk_weights = case["topk_weights"]

    pair_expert = topk_ids.reshape(-1).to(torch.int64)
    flat_weights = topk_weights.reshape(-1)
    x = x_q.float() * x_scale.repeat_interleave(BLOCK_SIZE, dim=1)
    w1 = case["gemm1_weights"].float() * _expand_block_scales(case["gemm1_weights_scale"])
    w2 = case["gemm2_weights"].float() * _expand_block_scales(case["gemm2_weights_scale"])

    output = torch.zeros((NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device)
    abs_sum = torch.zeros((NUM_TOKENS, HIDDEN), dtype=torch.float32, device=device)
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
    """The task's pass/fail gate: repeatability plus the element-wise bound."""
    _cfg(**kwargs)
    first, actual = outputs["first"], outputs["actual"]
    reference, abs_sum = outputs["reference"], outputs["abs_sum"]
    for name, tensor in (("first", first), ("actual", actual), ("reference", reference)):
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} output contains non-finite values")
    if not torch.equal(first, actual):
        max_abs = float((first.float() - actual.float()).abs().max())
        raise AssertionError(
            f"identical launches are not exactly repeatable; max abs diff={max_abs}"
        )
    ref = reference.float()
    diff = (actual.float() - ref).abs()
    bound = torch.maximum(0.1 + 0.1 * ref.abs(), 2.0 * abs_sum * 2.0**-7)
    outside = int((diff > bound).sum())
    if outside:
        raise AssertionError(
            f"{outside} of {diff.numel()} elements exceed the accumulation-order bound; "
            f"max abs diff={float(diff.max())}"
        )


def run_test(**kwargs: Any) -> None:
    _assert_supported_arch()
    from tirx_kernels.runner import compile_kernel

    config = dict(kwargs)
    config.pop("num_ctas", None)
    num_ctas = _num_ctas(**config)
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
    """

    def build():
        from flashinfer.fused_moe import trtllm_fp8_block_scale_routed_moe

        cfg: AlphaMoEConfig = case["config"]
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
                tune_max_num_tokens=NUM_TOKENS,
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
