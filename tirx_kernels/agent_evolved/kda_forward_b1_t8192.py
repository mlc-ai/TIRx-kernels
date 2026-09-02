# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved SM100a KDA forward for the Kimi K3 prefill workloads.

The supported contract is B=1, 8192 total tokens, H in {64, 96}, K=V=128,
BF16 q/k/v/g/beta and output, FP32 gate parameters and per-sequence V-first
initial state, and either one fixed sequence, six mixed-length sequences, or
eight 1024-token sequences. No final state is produced.

The selected kernel is candidate v68 from the 2026-09-01 KDA-forward evolution
run. It is a fused persistent tcgen05 implementation with device-side dynamic
LPT work assignment, TMEM recurrent state, shape-specific compact paths, and
tail-safe packed sequence handling. Evolution records and profiler artifacts
remain in the run tree; this module owns the supported implementation and
registry interface.
"""

import ctypes
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K
import tvm

D_HEAD = 128
CHUNK = 64
LOG2E = 1.4426950408889634
NEG5LOG2E = -5.0 * LOG2E
TMEM_COLS = 512

# TMEM column map (f32 columns; 128 lanes each)
TM_S = 0  # S^T fp32 running state              [128 lanes v x 128 cols k]
TM_ST = 128  # S^T bf16 A-operand copy              (64 cols)
TM_U = 192  # U^T fp32 accumulator (G3 + G4)       (64 cols)
TM_UB = 256  # U^T bf16 A-operand copy              (32 cols)
TM_O = 288  # O^T fp32 accumulator (G5 + G6)       (64 cols)
TM_V1 = 352  # V1 = S~'^T X0^T fp32 accumulator     (64 cols; bf16 copy goes to TM_UB)
TM_D0 = 416  # G1_0: [Akk;Aqk] columns 0..31        (32 cols; chunk parity 1 uses TM_D0 + 64 = 480)
TM_D1 = 448  # G1_1: [Akk;Aqk] columns 32..63       (32 cols)

# tcgen05 instruction descriptors (kind::f16, bf16 x bf16 -> f32, dense)
ID_G1 = 0x08080490  # M128 N32  K-major A, K-major B
ID_G1B = 0x04080490  # M64 N32  K-major A, K-major B (G1_1: J1 is a 64-row tile)
ID_G23 = 0x08108490  # M128 N64  MN-major A (smem), K-major B
ID_G56 = 0x08100490  # M128 N64  tmem A, K-major B
ID_G4B = 0x08104490  # M128 N64  tmem A, K-major B, negate B (G4'': U^T -= V1b TB^T)
ID_G7 = 0x08210490  # M128 N128 tmem A, MN-major B

MMA_SS = "tcgen05.mma.cta_group::1.kind::f16"
TMA_LD = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
TMA_ST = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
TMA_PREFETCH = "cp.async.bulk.prefetch.tensor.3d.L2.global.tile"
TC_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TC_ST32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"
TC_LD256 = "tcgen05.ld.sync.aligned.16x256b.x8.b32"
TC_ST16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
WAIT_LD = "tcgen05.wait::ld.sync.aligned"
WAIT_ST = "tcgen05.wait::st.sync.aligned"
STM_X4T = "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
MMA_K8 = "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
MMA_K16 = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
LDM_X1 = "ldmatrix.sync.aligned.m8n8.x1.shared.b16"
LDM_X1T = "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16"
LDM_X4 = "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
LDM_X4T = "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
STM_X1 = "stmatrix.sync.aligned.m8n8.x1.shared.b16"
STM_X4 = "stmatrix.sync.aligned.m8n8.x4.shared.b16"
FENCE_ASYNC = "fence.proxy.async.shared::cta"
BULK_COMMIT = "cp.async.bulk.commit_group"
BULK_WAIT_READ = "cp.async.bulk.wait_group.read"
BULK_WAIT = "cp.async.bulk.wait_group"
BULK_S2C = "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
LDG_V4 = "ld.global.L1::no_allocate.v4.b32"

BAR_INV, BAR_INV_N = 2, 128  # cg0 inverse stages
BAR_PREP, BAR_PREP_N = 3, 256  # prep gate-scan exchange (8 warps)
BAR_CG1, BAR_CG1_N = 4, 128  # cg1 tmem dealloc rendezvous
BAR_INV2, BAR_INV2_N = 5, 64  # cg0 32->64 merge (warps 0,1)
BAR_RNORM, BAR_RNORM_N = 6, 256  # chain prep: all remote norm reads completed
BAR_HELP, BAR_HELP_N = 7, 512  # helper: both eight-warp norm teams

STAGE_ROWS = 3 * CHUNK  # k rows 0..63, q rows 64..127, g rows 128..191
STAGE_UNITS = STAGE_ROWS * D_HEAD * 2 // 16  # one stage in 16-byte descriptor units
KG_UNITS = CHUNK * D_HEAD * 2 // 16
STAGE_BYTES = STAGE_ROWS * D_HEAD * 2
KQ_BYTES = 2 * CHUNK * D_HEAD * 2  # TMA bytes per stage: k and q rows only (g comes from global)
V_BYTES = CHUNK * D_HEAD * 2
NORM_BYTES = 2 * CHUNK * 4
BETA_BYTES = CHUNK * 4


MAX_ORDER = 512  # sequences beyond this count are dispatched in natural order
WORK_RING = (
    16  # work-id ring depth (loader fetches ahead of the slowest role by at most this many items)
)


def make_kernel(
    H: int,
    iket: bool = False,
    static_grid: int | None = None,
    compact: bool = False,
    fixed: bool = False,
    cluster_norm: bool = False,
    packed_tail_hoist: bool = False,
    uniform: bool = False,
    mixed: bool = False,
):
    """Trace the packed-varlen KDA forward kernel for one head count (iket=True adds timeline ranges;
    static_grid pins the CTA count for CPU analysis tools that need a static launch extent; compact=True
    selects the smaller-code variant used by the high-CTA-count configs)."""
    if cluster_norm and not fixed:
        raise ValueError("cluster_norm requires the fixed single-sequence specialization")
    if uniform and fixed:
        raise ValueError("uniform specialization requires packed input")
    mixed = mixed and H == 96 and not fixed and not uniform
    derive_cg1_vo = mixed or (uniform and H == 64)
    HK = H * D_HEAD

    @K.kernel(
        warps=20,
        arch="sm_100a",
        min_blocks_per_sm=1,
        grid="num_ctas" if static_grid is None else static_grid,
    )
    def kda_fwd(
        q: K.gptr[K.bf16],
        k: K.gptr[K.bf16],
        v: K.gptr[K.bf16],
        g: K.gptr[K.bf16],
        beta: K.gptr[K.bf16],
        a_log: K.gptr[K.f32],
        dt_bias: K.gptr[K.f32],
        h0: K.gptr[K.f32],
        cu_seqlens: K.gptr[K.i64],
        sched: K.gptr[K.i32],
        o: K.gptr[K.bf16],
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        g_map: K.TensorMap,
        o_map: K.TensorMap,
        scale: K.f32,
        num_seqs: K.i32,
        total_tokens: K.i32,
        num_ctas: K.i32,
    ):
        # q/k/v/o are only reached through the tensor maps.
        K.keep_alive(q.data)
        K.keep_alive(k.data)
        K.keep_alive(v.data)
        K.keep_alive(g.data)
        K.keep_alive(o.data)
        num_work = H if fixed else (8 * H if uniform else num_seqs * H)
        HK64 = K.int64(HK)
        cluster_rank = K.cta_id_in_cluster([3], preferred=[3]) if cluster_norm else K.int32(0)
        is_chain = cluster_rank < K.int32(2) if cluster_norm else K.bool(True)

        sp = K.specialize()
        # setmaxnreg redistributes the LAUNCH allocation (96 regs x 640 threads = 61440), not the
        # 64K file: the role targets must sum to 480 per 128-thread column (Kern validates against 65536).
        prep = sp.role("prep", warps=list(range(8)), regs=104)  # 2 warpgroups
        cg0 = sp.role("cg0", warps=[8, 9, 10, 11], regs=112)
        cg1 = sp.role("cg1", warps=[12, 13, 14, 15], regs=112)
        auxg = sp.warpgroup("aux", warps=[16, 17, 18, 19], regs=48)
        mma0 = sp.role("mma0", warps=[16], group=auxg)  # idle (every MMA is issued by mma1 or cg0)
        loader = sp.role("loader", warps=[17], group=auxg)
        storer = sp.role("storer", warps=[18], group=auxg)
        mma1 = sp.role("mma1", warps=[19], group=auxg)  # state-chain GEMMs G5,G4,G6,G7

        smem = K.smem_pool()
        s_tmem_addr = smem.alloc((1,), K.i32, align=4)
        # ---- pipelines / barriers ---------------------------------------
        p_stage = K.Pipeline(
            smem, 2, full="tma", empty="tcgen05", init_empty=2
        )  # loader -> prep ; G1_0 (mma0) + V1 (after G5, mma1) -> loader
        p_v = K.Pipeline(smem, 1, full="tma", empty="tcgen05")  # loader -> mma(G3) ; G3 -> loader
        p_prep = K.Pipeline(
            smem, 2, full="mbar", empty="tcgen05", init_full=256
        )  # prep(8 warps) -> mma ; G7 -> prep (KG)
        p_j1 = K.Pipeline(
            smem, 1, full="mbar", empty="tcgen05", init_full=256
        )  # prep(8 warps) -> mma ; G1_1 -> prep
        p_beta = K.Pipeline(
            smem, 2, full="mbar", empty="mbar", init_full=1 if cluster_norm else 32, init_empty=128
        )  # loader/helper -> cg0
        p_g1 = K.Pipeline(
            smem, 2, full="tcgen05", empty="mbar", init_empty=128
        )  # G1_0 (D0, two TMEM buffers) -> cg0 ; cg0 (D0 read) -> mma0
        m_g1b = K.TCGen05Bar(smem, 1)  # G1_1 (D1) done -> cg0
        m_g1b.init(1)
        p_g1b = K.MBarrier(
            smem, 1
        )  # D1 consumed by cg0 (Akk and Aqk halves) -> mma1 may overwrite it with the next G1_1
        p_g1b.init(128)
        p_t = K.Pipeline(
            smem, 1, full="mbar", empty="tcgen05", init_full=128
        )  # cg0 -> mma ; G4'' (mma1, after G3) -> cg0
        p_aqk = K.Pipeline(
            smem, 1, full="mbar", empty="tcgen05", init_full=64
        )  # cg0 warps 2/3 -> mma1 (Aqk rows in the stage Y region; the region is recycled by prep after G7)
        p_u = K.Pipeline(
            smem, 1, full="tcgen05", empty="mbar", init_empty=128
        )  # G4 -> cg1 ; cg1 -> mma(G3)
        p_o = K.Pipeline(
            smem, 1, full="tcgen05", empty="mbar", init_empty=128
        )  # G6 -> cg1 ; cg1 -> mma(G5)
        p_osm = K.Pipeline(
            smem, 1, full="mbar", empty="mbar", init_full=128, init_empty=1
        )  # cg1 -> storer -> cg1
        m_v1 = K.TCGen05Bar(smem, 1)  # V1 done (mma1) -> cg1
        m_v1.init(1)
        m_v1b = K.MBarrier(smem, 1)  # V1 bf16 copy in TM_UB ready (cg1) -> mma1 (G4'')
        m_v1b.init(128)
        m_sacc = K.TCGen05Bar(smem, 1)  # G7 done -> cg1
        m_sacc.init(1)
        m_ub = K.MBarrier(smem, 1)  # U^T bf16 ready (cg1 -> mma)
        m_ub.init(128)
        m_s = K.MBarrier(smem, 1)  # S^T bf16 ready (cg1 -> mma)
        m_s.init(128)
        m_g3 = K.TCGen05Bar(smem, 1)  # G3 done (cg0) -> mma1 may accumulate G4'' into U^T
        m_g3.init(1)

        # ---- shared memory plan ------------------------------------------
        s_stage = smem.alloc((2, STAGE_ROWS, D_HEAD), K.bf16, swizzle=K.SW128B)  # 96 KB
        s_j1 = smem.alloc((1, CHUNK, D_HEAD), K.bf16, swizzle=K.SW128B)[0]  # 16 KB [X1(32); Q1(32)]
        s_T = smem.alloc((1, CHUNK, CHUNK), K.bf16, swizzle=K.SW128B)[0]  # 8 KB  TB = T*beta_j
        s_v = smem.alloc((1, CHUNK, D_HEAD), K.bf16, swizzle=K.SW128B)[0]  # 16 KB
        s_kg = smem.alloc((2, CHUNK, D_HEAD), K.bf16, swizzle=K.SW128B)  # 32 KB
        s_A = smem.alloc((CHUNK, CHUNK), K.f16, swizzle=K.SW128B)  # 8 KB inverse workspace
        s_o = smem.alloc((CHUNK, D_HEAD), K.bf16, swizzle=K.SW128B)  # 16 KB O [t][v]
        s_beta = smem.alloc((2, CHUNK), K.f32, align=16)
        s_hbeta = smem.alloc((2, 2, CHUNK), K.f32, align=16) if cluster_norm else None
        # 4-deep ring: chunk n's vectors live in slot n&3; prep(n) writes it after the stage n-1 TMA, which
        # the loader issued only after G5(n-3), i.e. after cg1's state epilogue n-4 consumed chunk n-3's
        # vectors and (in program order) epilogue n-5 consumed chunk n-4's, the slot's previous tenant.
        # The midpoint-scaled state copy is consumed as bf16, so its factors are stored in that
        # destination precision.  End-of-chunk decay still uses fp32 factors and fp32 state.
        s_gate_r = smem.alloc((4, D_HEAD), K.bf16, align=16)  # [chunk&3]=bf16(2^R0)
        s_gate_e = smem.alloc((4, D_HEAD), K.f32, align=16)  # [chunk&3]=2^G2_63
        s_part = smem.alloc(
            (2, 8, D_HEAD), K.f32, align=16
        )  # gate-scan partial totals, [chunk&1][group]
        s_norm = smem.alloc(
            (2, 2, CHUNK), K.f32, align=16
        )  # [stage][0]=rstd_k[t], [1]=rstd_q[t]*scale (prep-private per warp)
        s_order = (
            None if uniform else smem.alloc((MAX_ORDER,), K.i32, align=16)
        )  # sequence index by descending-length rank
        s_work = smem.alloc(
            (WORK_RING,), K.i32, align=16
        )  # work-id ring: loader -> every other warp
        s_cu = (
            None if uniform else smem.alloc((MAX_ORDER + 1,), K.i64, align=16)
        )  # cu_seqlens copy (num_seqs <= MAX_ORDER)
        # full: the loader's lane 0 publishes one id; empty: all 32 lanes of the 19 consumer warps
        p_work = (
            None
            if fixed
            else K.Pipeline(
                smem, WORK_RING, full="mbar", empty="mbar", init_full=1, init_empty=19 * 32
            )
        )
        if cluster_norm:
            # One private q/k stage per helper team.  Eight norm warps release each stage.
            p_hstage0 = K.Pipeline(smem, 1, full="tma", empty="mbar", init_empty=8 * 32)
            p_hstage1 = K.Pipeline(smem, 1, full="tma", empty="mbar", init_empty=8 * 32)
            # Helper bulk copies complete on the owning chain's local barrier.  The chain then
            # remotely releases that team's single local norm source after all eight warps read it.
            rnorm_full = K.MBarrier(smem, 2)
            rnorm_full.init(1)
            helper_release = K.MBarrier(smem, 2)
            helper_release.init(1)
            # Per-team, per-destination-stage acknowledgement after warp 0 observes
            # p_t.full, which proves that all 128 cg0 threads completed their beta reads.
            helper_beta_release = K.MBarrier(smem, 4)
            helper_beta_release.init(1)
        else:
            p_hstage0 = p_hstage1 = rnorm_full = helper_release = helper_beta_release = None

        # Item order: sequences by descending length (ties by index) so the longest items
        # are dispatched first.  Natural order beyond MAX_ORDER sequences.
        if not uniform:
            tid0 = K.thread_id()
            with K.If(num_seqs <= MAX_ORDER), K.Then():
                with K.If(tid0 <= num_seqs), K.Then():
                    cuv = K.local_scalar("int64")
                    K.ptx.ld.global_.s64(cuv, cu_seqlens.ptr_to([tid0]))
                    K.ptx.st.shared.s64(K.address_of(s_cu[tid0]), cuv)
                with K.If(tid0 < num_seqs), K.Then():
                    cs0 = K.alloc_local([2], "int64")
                    K.ptx.ld.global_.s64(cs0[0], cu_seqlens.ptr_to([tid0]))
                    K.ptx.ld.global_.s64(cs0[1], cu_seqlens.ptr_to([tid0 + 1]))
                    len_i = K.local_scalar("int64", init=cs0[1] - cs0[0])
                    rank = K.local_scalar("int32", init=K.int32(0))
                    with K.serial(num_seqs) as j:
                        cs1 = K.alloc_local([2], "int64")
                        K.ptx.ld.global_.s64(cs1[0], cu_seqlens.ptr_to([j]))
                        K.ptx.ld.global_.s64(cs1[1], cu_seqlens.ptr_to([j + 1]))
                        len_j = K.local_scalar("int64", init=cs1[1] - cs1[0])
                        K.assign(
                            rank,
                            rank
                            + K.Select(
                                len_j > len_i,
                                K.int32(1),
                                K.Select(
                                    len_j == len_i,
                                    K.Select(j < tid0, K.int32(1), K.int32(0)),
                                    K.int32(0),
                                ),
                            ),
                        )
                    K.ptx.st.shared.s32(K.address_of(s_order[rank]), tid0)
        # The inverse never writes TB's strictly-upper 32x32 block; zero s_T once
        # so that block stays the required zero operand in every chunk.
        with K.If(K.thread_id() < 512), K.Then():
            zpad = K.alloc_local([4], "uint32")
            for i in range(4):
                K.assign(zpad[i], K.uint32(0))
            K.ptx["st.shared.v4.b32"](
                s_T.ptr_to(K.thread_id() >> 3, (K.thread_id() & 7) * 8),
                zpad[0],
                zpad[1],
                zpad[2],
                zpad[3],
            )
        K.ptx[FENCE_ASYNC]()
        with K.If(K.thread_id() == 0), K.Then():
            K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()
        if cluster_norm:
            K.ptx.barrier.cluster.arrive.release.aligned()
            K.ptx.barrier.cluster.wait.acquire()
        # TMEM allocation by warp 8 (cg1 warp 0); the address is read by all after cta_sync.
        with K.If(is_chain), K.Then():
            with K.If(K.warp_id() == 12), K.Then():
                K.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                    K.address_of(s_tmem_addr[0]), K.uint32(TMEM_COLS)
                )
        if cluster_norm:
            with K.If((cluster_rank == K.int32(2)) & (K.thread_id() == 0)), K.Then():
                K.ptx.st.shared.s32(K.address_of(s_tmem_addr[0]), K.int32(0))
        K.cuda.cta_sync()

        # ---- trace-time layout facts ------------------------------------
        def row_off16(tile, rows):
            """16B-unit offset of row `rows` (column 0) inside one stage of `tile`."""
            t = tile.tile if isinstance(tile, K.KTileView) else tile
            c0 = t._coord(0, 0, 0) if t.stages is not None else t._coord(0, 0)
            c1 = t._coord(0, rows, 0) if t.stages is not None else t._coord(rows, 0)
            delta = t._phys(c1) - t._phys(c0)
            assert delta * 2 % 16 == 0
            return delta * 2 // 16

        OFF_Q = row_off16(s_stage, 64)
        OFF_Y0 = row_off16(s_stage, 128)
        OFF_Y1 = row_off16(s_stage, 160)
        assert (OFF_Q, OFF_Y0, OFF_Y1) == (64 * 8, 128 * 8, 160 * 8)

        # ---- shared helpers ------------------------------------------------
        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def rng(name):
            if not iket:
                return None
            tok = K.alloc_local([1], "uint32")
            K.assign(tok[0], K.cuda.iket.range_start(name))
            return tok

        def rng_end(tok):
            if tok is not None:
                K.cuda.iket.range_end(tok[0])

        def elect_local():
            e = K.local_scalar("uint32")
            K.assign(e, K.cuda.elect_sync())
            return e

        def tmem_preamble():
            tm = K.alloc_local([1], "int32")
            K.ptx.ld.volatile.shared.s32(tm[0], K.address_of(s_tmem_addr[0]))
            return tm

        def pack_bf16x2(dst, lo, hi):
            # cvt.rn.bf16x2.f32 d, a, b puts b in the LOW half.
            K.ptx.cvt.rn.bf16x2.f32(dst, hi, lo)

        def pack_f16x2(dst, lo, hi):
            K.ptx.cvt.rn.f16x2.f32(dst, hi, lo)

        def bf16_lo(w):
            return K.reinterpret("float32", w << K.uint32(16))

        def bf16_hi(w):
            return K.reinterpret("float32", w & K.uint32(0xFFFF0000))

        def lds128(view, row, col, words, base):
            K.ptx["ld.shared.v4.b32"](
                words[base],
                words[base + 1],
                words[base + 2],
                words[base + 3],
                view.ptr_to(row, col),
            )

        def sts128(view, row, col, words, base):
            K.ptx["st.shared.v4.b32"](
                view.ptr_to(row, col),
                words[base],
                words[base + 1],
                words[base + 2],
                words[base + 3],
            )

        def f2(a, b):
            return K.cuda.make_float2(a, b)

        def runtime_off(off_fn, n_k):
            """Runtime 16B-unit descriptor offset of k-tile kp: the trace-time offset table is either
            linear or two linear runs (one per 64-column swizzle atom)."""
            offs = [int(off_fn(k)) for k in range(n_k)]
            if n_k == 1:
                return lambda kp: K.uint64(0)
            d1 = offs[1] - offs[0]
            if all(offs[k] == offs[0] + k * d1 for k in range(n_k)):
                return lambda kp: K.Cast("uint64", kp) * K.uint64(d1) + K.uint64(offs[0])
            half = n_k // 2
            d_atom = offs[half] - offs[0]
            assert half & (half - 1) == 0 and all(
                offs[k] == offs[0] + (k // half) * d_atom + (k % half) * d1 for k in range(n_k)
            ), offs
            sh = half.bit_length() - 1
            return lambda kp: (
                K.Cast("uint64", kp >> sh) * K.uint64(d_atom)
                + K.Cast("uint64", kp & (half - 1)) * K.uint64(d1)
                + K.uint64(offs[0])
            )

        def mma_issue(dcol, a_of, b_base, b_off, n_k, idesc, accumulate, tmem):
            """One tcgen05 k-chain from an elected lane. `a_of(kp)` yields the A operand (smem descriptor
            or tmem address). Unrolled for the fast shape; a rolled loop for the compact shape."""

            def one(kp, acc_pred):
                K.ptx[MMA_SS](
                    K.Cast("uint32", tmem[0] + dcol),
                    a_of(kp),
                    b_base + b_off(kp),
                    K.uint32(idesc),
                    K.uint32(0),
                    K.uint32(0),
                    K.uint32(0),
                    K.uint32(0),
                    K.ptx.pred(acc_pred),
                )

            with K.If(elected()), K.Then():
                if compact:
                    with K.serial(n_k, unroll=False) as kp:
                        one(kp, K.uint32(1) if accumulate else K.Cast("uint32", kp != 0))
                else:
                    for kp in range(n_k):
                        one(kp, K.uint32(1) if (accumulate or kp != 0) else K.uint32(0))

        def consume_work(st):
            """Next work id for a consumer role (waits for the loader's publication)."""
            p_work.full.wait(st.stage, st.phase)
            w = K.local_scalar("int32")
            K.ptx.ld.shared.s32(w, K.address_of(s_work[st.stage]))
            p_work.empty.arrive(st.stage)
            st.advance()
            return w

        def prefetch_initial_state(h0_base):
            for part in range(4):
                K.ptx["cp.async.bulk.prefetch.L2.global"](
                    h0.ptr_to([h0_base + K.int64(part * (D_HEAD * D_HEAD // 4))]),
                    K.uint32(D_HEAD * D_HEAD),
                )

        def fetch_work(st):
            """Loader only: grab the next work id from the global counter and publish it."""
            p_work.empty.wait(st.stage, st.phase)
            w = K.local_scalar("int32")
            with K.If(K.lane_id() == 0), K.Then():
                old = K.local_scalar("uint32")
                K.ptx.atom.global_.add.u32(old, sched.ptr_to([0]), K.uint32(1))
                published = K.local_scalar("int32", init=K.Cast("int32", old))
                if uniform and H == 96 and static_grid is None:
                    # Logical uniform work IDs are the flattened (sequence, head) state index.
                    # Enqueue the state prefetch before publishing the ID to cg1.
                    with K.If(K.Cast("int32", old) < K.int32(num_work)), K.Then():
                        prefetch_initial_state(K.Cast("int64", old) * K.int64(D_HEAD * D_HEAD))
                if uniform and H == 96:
                    # Preserve the sequence-major scheduler order, but publish the valid work
                    # item as (head << 3) | sequence.  One loader lane pays the /H conversion;
                    # all 19 consumer warps recover the coordinates with a shift and mask.
                    seq = K.local_scalar("int32", init=K.Cast("int32", old // K.uint32(H)))
                    K.assign(
                        published,
                        K.Select(
                            K.Cast("int32", old) < K.int32(num_work),
                            (K.Cast("int32", old) - seq * K.int32(H)) * K.int32(8) + seq,
                            K.int32(num_work),
                        ),
                    )
                K.ptx.st.shared.s32(K.address_of(s_work[st.stage]), published)
                p_work.full.arrive(st.stage)
            K.ptx["bar.warp.sync"](K.uint32(0xFFFFFFFF))
            K.ptx.ld.shared.s32(w, K.address_of(s_work[st.stage]))
            st.advance()
            return w

        def begin_work(producer=False):
            """Create one role's work cursor; the fixed shape maps CTA id directly to head."""
            if fixed:
                if cluster_norm:
                    cid = K.Cast("int32", K.cta_id()) // K.int32(3)
                    mapped = K.Select(is_chain, cid * K.int32(2) + cluster_rank, K.int32(H))
                    return None, K.local_scalar("int32", init=mapped)
                return None, K.local_scalar("int32", init=K.Cast("int32", K.cta_id()))
            st = K.PipelineState(WORK_RING, phase=1 if producer else 0)
            work = K.local_scalar("int32", init=K.int32(0))
            K.assign(work, fetch_work(st) if producer else consume_work(st))
            return st, work

        def advance_work(work, st, producer=False):
            if fixed:
                K.assign(work, work + num_ctas)
            else:
                K.assign(work, fetch_work(st) if producer else consume_work(st))

        def work_coords(work):
            """(seq, head, seq_start:int64, seq_len:int32, nch:int32) of work item `work`."""
            if fixed:
                return (
                    K.int32(0),
                    work,
                    K.int64(0),
                    total_tokens,
                    (total_tokens + K.int32(CHUNK - 1)) >> 6,
                )
            if uniform:
                wu = K.Cast("uint32", work)
                if H == 96:
                    r_idx = K.Cast("int32", wu & K.uint32(7))
                    h_idx = K.Cast("int32", wu >> 3)
                else:
                    r_idx = K.Cast("int32", wu // K.uint32(H))
                    h_idx = K.local_scalar("int32", init=work - r_idx * H)
                return (
                    r_idx,
                    h_idx,
                    K.Cast("int64", r_idx) * K.int64(1024),
                    K.int32(1024),
                    K.int32(16),
                )
            wu = K.Cast("uint32", work)
            r_idx = K.Cast("int32", wu // K.uint32(H))
            h_idx = K.local_scalar("int32", init=work - r_idx * H)
            so = K.local_scalar("int32")
            K.ptx.ld.shared.s32(so, K.address_of(s_order[K.min(r_idx, K.int32(MAX_ORDER - 1))]))
            seq_idx = K.local_scalar("int32", init=K.Select(num_seqs <= MAX_ORDER, so, r_idx))
            cs = K.alloc_local([2], "int64")
            with K.If(num_seqs <= MAX_ORDER):
                with K.Then():
                    K.ptx.ld.shared.s64(cs[0], K.address_of(s_cu[seq_idx]))
                    K.ptx.ld.shared.s64(cs[1], K.address_of(s_cu[seq_idx + 1]))
                with K.Else():
                    K.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([seq_idx]))
                    K.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([seq_idx + 1]))
            seq_start = K.local_scalar("int64", init=cs[0])
            seq_len = K.local_scalar("int32", init=K.Cast("int32", cs[1] - cs[0]))
            nch = K.local_scalar("int32", init=(seq_len + K.int32(CHUNK - 1)) >> 6)
            return seq_idx, h_idx, seq_start, seq_len, nch

        if cluster_norm:

            def helper_load(team):
                """One helper loader warp TMA-loads q/k and publishes beta for one chain."""
                pipe = p_hstage0 if team == 0 else p_hstage1
                st = K.PipelineState(1, phase=1)
                cid = K.Cast("int32", K.cta_id()) // K.int32(3)
                h_idx = cid * K.int32(2) + K.int32(team)
                nch = (total_tokens + K.int32(CHUNK - 1)) >> 6
                tok_last = total_tokens - K.int32(1)
                with K.If(elected()), K.Then():
                    K.ptx.prefetch.tensormap(K.address_of(k_map))
                    K.ptx.prefetch.tensormap(K.address_of(q_map))
                with K.serial(nch) as n:
                    _t = rng(f"HL{team}.wait")
                    pipe.empty.wait(st.stage, st.phase)
                    rng_end(_t)
                    _t = rng(f"HL{team}.issue")
                    tok0 = n * CHUNK
                    with K.If(elected()), K.Then():
                        pipe.full.arrive(st.stage, tx_count=KQ_BYTES)
                        mb = K.cuda.cvta_generic_to_shared(pipe.full.ptr_to([st.stage]))
                        for tmap, row0 in ((k_map, 0), (q_map, 64)):
                            for d0 in (0, 64):
                                K.ptx[TMA_LD](
                                    s_stage[team].ptr_to(row0, d0),
                                    K.address_of(tmap),
                                    K.int32(d0),
                                    K.Cast("int32", tok0),
                                    h_idx,
                                    mb,
                                )
                    st.advance()
                    rng_end(_t)
                    # The source and destination are double-buffered.  Before reuse at n+2,
                    # wait until all four cg0 warps have completed their reads of chunk n.
                    bstage = n & 1
                    with K.If(n >= 2), K.Then():
                        K.cuda.mbarrier_wait_acquire_cluster(
                            helper_beta_release.ptr_to([team * 2 + bstage]), ((n >> 1) - 1) & 1
                        )
                    _t = rng(f"HB{team}.calc")
                    for i in range(2):
                        t = K.lane_id() + 32 * i
                        bu = K.local_scalar("uint16")
                        tokc = K.min(tok0 + t, tok_last)
                        K.ptx.ld.global_.nc.u16(
                            bu,
                            beta.ptr_to(
                                [K.Cast("int64", tokc) * K.int64(H) + K.Cast("int64", h_idx)]
                            ),
                        )
                        bfv = K.reinterpret("float32", K.Cast("uint32", bu) << K.uint32(16))
                        sg = K.idioms.sigmoid_tanh_approx_f32(bfv)
                        K.ptx.st.shared.f32(K.address_of(s_hbeta[team, bstage, t]), sg)
                    K.ptx[FENCE_ASYNC]()
                    with K.If(elected()), K.Then():
                        rem_bar = K.local_scalar("uint32")
                        rem_dst = K.local_scalar("uint32")
                        K.ptx.mapa.shared__cluster.u32(
                            rem_bar,
                            K.cuda.cvta_generic_to_shared(p_beta.full.ptr_to([bstage])),
                            K.uint32(team),
                        )
                        K.ptx.mapa.shared__cluster.u32(
                            rem_dst,
                            K.cuda.cvta_generic_to_shared(s_beta.ptr_to([bstage, 0])),
                            K.uint32(team),
                        )
                        K.ptx["mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64"](
                            rem_bar, K.uint32(BETA_BYTES)
                        )
                        K.ptx[BULK_S2C](
                            rem_dst,
                            s_hbeta.ptr_to([team, bstage, 0]),
                            K.uint32(BETA_BYTES),
                            rem_bar,
                        )
                    rng_end(_t)

            def helper_norm(team, r_grp):
                """Eight helper warps compute and publish one chain's 64 q/k row norms."""
                pipe = p_hstage0 if team == 0 else p_hstage1
                st = K.PipelineState(1, phase=0)
                lane = K.lane_id()
                base_k = s_stage[team].ptr_to(0, 0)
                norm_row = (lane >> 4) * CHUNK + r_grp * 8 + ((lane & 15) >> 1)
                norm_off = K.local_scalar(
                    "int32", init=(lane & 1) * (STAGE_ROWS * 128) + norm_row * 128
                )
                norm_x = K.local_scalar("int32", init=((norm_row & 7) ^ ((lane & 1) << 2)) << 4)
                norm_is_q = lane >= 16
                norm_wr = (lane & 1) == 0
                nch = (total_tokens + K.int32(CHUNK - 1)) >> 6
                with K.serial(nch) as n:
                    # The previous chain release proves both that the prior DSMEM copy read this
                    # team's single local norm tile and that every chain warp finished reading it.
                    with K.If(n > 0), K.Then():
                        K.cuda.mbarrier_wait_acquire_cluster(
                            helper_release.ptr_to([team]), (n - 1) & 1
                        )
                    _t = rng(f"HN{team}.wait")
                    pipe.full.wait(st.stage, st.phase)
                    rng_end(_t)
                    _t = rng(f"HN{team}.norm")
                    nacc = K.alloc_local([4], "uint64")
                    for pp in range(4):
                        K.assign(nacc[pp], f2(K.float32(0.0), K.float32(0.0)))
                    nw = K.alloc_local([4], "uint32")
                    for j in range(8):
                        K.ptx["ld.shared.v4.b32"](
                            nw[0],
                            nw[1],
                            nw[2],
                            nw[3],
                            K.ptx.addr(base_k, norm_off + (K.int32(16 * j) ^ norm_x)),
                        )
                        for pp in range(4):
                            xv = f2(bf16_lo(nw[pp]), bf16_hi(nw[pp]))
                            K.ptx.fma.rn.f32x2(nacc[pp], xv, xv, nacc[pp])
                    nhalf = K.local_scalar(
                        "float32",
                        init=(K.cuda.float2_x(nacc[0]) + K.cuda.float2_y(nacc[0]))
                        + (K.cuda.float2_x(nacc[1]) + K.cuda.float2_y(nacc[1]))
                        + (K.cuda.float2_x(nacc[2]) + K.cuda.float2_y(nacc[2]))
                        + (K.cuda.float2_x(nacc[3]) + K.cuda.float2_y(nacc[3])),
                    )
                    nother = K.local_scalar("uint32")
                    K.ptx.shfl_sync.bfly.b32(
                        nother,
                        K.reinterpret("uint32", nhalf),
                        K.uint32(1),
                        K.uint32(0x1F),
                        K.uint32(0xFFFFFFFF),
                    )
                    nr = K.local_scalar("float32")
                    K.ptx.rsqrt.approx.ftz.f32(
                        nr, (nhalf + K.reinterpret("float32", nother)) + K.float32(1e-6)
                    )
                    with K.If(norm_is_q), K.Then():
                        K.assign(nr, nr * scale)
                    with K.If(norm_wr), K.Then():
                        K.ptx.st.shared.f32(
                            K.address_of(s_norm[team, lane >> 4, r_grp * 8 + ((lane & 15) >> 1)]),
                            nr,
                        )
                    K.ptx[FENCE_ASYNC]()
                    rng_end(_t)
                    K.ptx.bar.sync(K.uint32(BAR_HELP), K.uint32(BAR_HELP_N))
                    with K.If(r_grp == 0), K.Then():
                        with K.If(elected()), K.Then():
                            dst_stage = n & 1
                            rem_bar = K.local_scalar("uint32")
                            rem_dst = K.local_scalar("uint32")
                            K.ptx.mapa.shared__cluster.u32(
                                rem_bar,
                                K.cuda.cvta_generic_to_shared(rnorm_full.ptr_to([dst_stage])),
                                K.uint32(team),
                            )
                            K.ptx.mapa.shared__cluster.u32(
                                rem_dst,
                                K.cuda.cvta_generic_to_shared(s_norm.ptr_to([dst_stage, 0, 0])),
                                K.uint32(team),
                            )
                            K.ptx["mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64"](
                                rem_bar, K.uint32(NORM_BYTES)
                            )
                            K.ptx[BULK_S2C](
                                rem_dst, s_norm.ptr_to([team, 0, 0]), K.uint32(NORM_BYTES), rem_bar
                            )
                    K.ptx.bar.sync(K.uint32(BAR_HELP), K.uint32(BAR_HELP_N))
                    pipe.empty.arrive(st.stage)
                    st.advance()

        # =====================================================================
        # PREP: gates, norms, gated bf16 operand tiles.       warps 0-3
        # =====================================================================
        with prep:
            st_stage = K.PipelineState(2, phase=0)  # consumer of p_stage.full
            st_prep = K.PipelineState(2, phase=1)  # producer of p_prep
            st_j1 = K.PipelineState(1, phase=1)  # producer of p_j1
            lane = K.lane_id()
            r_grp = K.local_scalar(
                "int32", init=K.warp_id_in_role()
            )  # token group 0..7 (one warp each)
            col0 = lane * 4  # channels 4*lane .. 4*lane+3
            hi_block = K.local_scalar("int32", init=K.Cast("int32", (K.warp_id_in_role() >= 4)))
            st_w, work = begin_work()

            # Byte offsets of this thread's 8-byte cell in token row 8r+i of a SW128B bf16 tile with
            # `rows` rows: atom column (lane/16) * rows*128 + row*128 + ((chunk ^ i) << 4) + (lane&1)*8,
            # chunk = (lane%16)/2.  Precomputed once; every tile access is then one add.
            def row_offsets(rows):
                offs = K.alloc_local([8], "int32")
                for i in range(8):
                    K.assign(
                        offs[i],
                        (lane >> 4) * (rows * 128)
                        + (r_grp * 8 + i) * 128
                        + ((((lane & 15) >> 1) ^ i) << 4)
                        + (lane & 1) * 8,
                    )
                return offs

            xo_stage = row_offsets(STAGE_ROWS)
            xo_adj = (lane >> 4) * (
                (STAGE_ROWS - CHUNK) * 128
            )  # xo_64[i] == xo_stage[i] - xo_adj (64-row tiles)
            # row norms: lanes 0..15 own this warp's eight k rows, lanes 16..31 its eight q rows, two lanes per
            # row, each lane one 64-column swizzle atom.  The atom's eight 16-byte chunks are summed in the
            # order j ^ (row & 7) ^ (atom << 2) (the sum does not care), which keeps the eight lanes of a
            # quarter-warp (four rows x two atoms) on eight distinct chunk positions: conflict-free LDS.128.
            norm_row = (lane >> 4) * CHUNK + r_grp * 8 + ((lane & 15) >> 1)
            norm_off = K.local_scalar(
                "int32", init=(lane & 1) * (STAGE_ROWS * 128) + norm_row * 128
            )
            norm_x = K.local_scalar("int32", init=((norm_row & 7) ^ ((lane & 1) << 2)) << 4)
            norm_is_q = lane >= 16
            norm_wr = (lane & 1) == 0

            def lds64(base, off, w0, w1):
                K.ptx["ld.shared.v2.b32"](w0, w1, K.ptx.addr(base, off))

            def sts64(base, off, w0, w1):
                K.ptx["st.shared.v2.b32"](K.ptx.addr(base, off), w0, w1)

            def mul2(dst_pair, a0, a1, b0, b1):
                K.ptx.mul.rn.f32x2(dst_pair, f2(a0, a1), f2(b0, b1))

            cyc = K.local_scalar("int32", init=K.int32(0))  # running chunk counter (ring slots)
            with K.While(work < num_work):
                seq_idx, h_idx, seq_start, seq_len, nch = work_coords(work)
                # every per-item global load first (independent latencies overlap): gate parameters
                # and, below, the first two chunks' gate rows
                ea = K.local_scalar("float32")
                K.ptx.ld.global_.f32(ea, a_log.ptr_to([h_idx]))
                bw = K.alloc_local([4], "uint32")
                K.ptx[LDG_V4](bw[0], bw[1], bw[2], bw[3], dt_bias.ptr_to([h_idx * D_HEAD + col0]))
                # gate logits of this thread's 8 tokens x 4 channels, fetched from global one chunk
                # ahead (they only feed pass 1, which therefore no longer waits for the TMA stage)
                gw = K.alloc_local([16], "uint32")
                g_hoff = K.local_scalar(
                    "int64", init=K.Cast("int64", h_idx) * K.int64(D_HEAD) + K.Cast("int64", col0)
                )
                tok_last = K.local_scalar("int32", init=total_tokens - K.int32(1))
                seq_start32 = K.local_scalar("int32", init=K.Cast("int32", seq_start))

                def g_addr(nn, i):
                    # token nn*64+t of this sequence; rows past the sequence end are masked later, the
                    # clamp only keeps the address inside the packed buffer
                    t = r_grp * 8 + i
                    tok = (
                        seq_start32 + (nn * CHUNK + t)
                        if uniform
                        else K.min(seq_start32 + (nn * CHUNK + t), tok_last)
                    )
                    return g.ptr_to([K.Cast("int64", tok) * HK64 + g_hoff])

                def issue_g_loads(nn):
                    for i in range(8):
                        K.ptx["ld.global.L1::no_allocate.v2.b32"](
                            gw[2 * i], gw[2 * i + 1], g_addr(nn, i)
                        )

                def issue_g_load_token(nn, i):
                    K.ptx["ld.global.L1::no_allocate.v2.b32"](
                        gw[2 * i], gw[2 * i + 1], g_addr(nn, i)
                    )

                # gate sigmoid + chunk-local cumsum of one token into G[4i..4i+3] (registers only)
                G = K.alloc_local([32], "float32")

                def pass1_token(nn, i):
                    # tokens at or past the sequence end contribute no decay (their log-gate is 0)
                    if uniform:
                        tvalid = K.float32(NEG5LOG2E)
                    else:
                        tvalid = K.local_scalar(
                            "float32",
                            init=K.Select(
                                nn * CHUNK + (r_grp * 8 + i) < seq_len,
                                K.float32(NEG5LOG2E),
                                K.float32(0.0),
                            ),
                        )
                    for j in range(4):
                        wj = gw[2 * i] if j < 2 else gw[2 * i + 1]
                        gv = bf16_lo(wj) if j % 2 == 0 else bf16_hi(wj)
                        sg = K.idioms.sigmoid_tanh_approx_f32(tanh_input=gv * hea + biash[j])
                        if i == 0:
                            K.assign(G[j], sg * tvalid)
                        else:
                            K.assign(G[i * 4 + j], G[(i - 1) * 4 + j] + sg * tvalid)

                # chunk 0's pass 1 runs here; every later chunk's pass 1 is fused into the previous chunk's pass 2
                with K.If(nch > 0), K.Then():
                    issue_g_loads(0)
                ea_l2 = K.local_scalar("float32")
                K.ptx.ex2.approx.ftz.f32(ea_l2, ea * K.float32(LOG2E))  # exp(A_log)
                hea = K.local_scalar(
                    "float32", init=ea_l2 * K.float32(0.5)
                )  # exp(A_log)/2 for tanh sigmoid
                bias = K.alloc_local([4], "float32")
                biash = K.alloc_local(
                    [4], "float32"
                )  # bias * exp(A_log)/2: the tanh argument is one fma
                for j in range(4):
                    K.assign(bias[j], K.reinterpret("float32", bw[j]))
                    K.assign(biash[j], bias[j] * hea)
                with K.If(nch > 0), K.Then():
                    for i in range(8):
                        pass1_token(K.int32(0), i)
                    with K.If(nch > 1), K.Then():
                        issue_g_loads(1)

                with K.serial(nch) as n:
                    sv = s_stage[st_stage.stage]
                    base_k = sv.ptr_to(0, 0)  # rows 0..63 (k -> X0)
                    base_q = sv.ptr_to(64, 0)  # rows 64..127 (q -> Q0)
                    base_g = sv.ptr_to(128, 0)  # rows 128..191 (g -> Y)
                    base_j1 = s_j1.ptr_to(0, 0)
                    base_kg = s_kg[st_stage.stage].ptr_to(0, 0)
                    w0 = K.local_scalar("uint32")
                    w1 = K.local_scalar("uint32")
                    # (pass 1 of this chunk already ran, fused into the previous chunk's pass 2: G = cumsum)
                    _t = rng("P.scan_a")
                    # block totals -> smem partials [r][k]
                    K.ptx["st.shared.v4.f32"](
                        K.address_of(s_part[cyc & 1, r_grp, col0]), G[28], G[29], G[30], G[31]
                    )
                    rng_end(_t)
                    _t = rng("P.scan_bar1")
                    K.ptx.bar.sync(K.uint32(BAR_PREP), K.uint32(BAR_PREP_N))
                    rng_end(_t)
                    _t = rng("P.scan_b")
                    # exclusive prefix over token groups; R0=P(2), R1=P(6), e63=P(8)
                    P = K.alloc_local([4], "float32")
                    R0 = K.alloc_local([4], "float32")
                    R1 = K.alloc_local([4], "float32")
                    E63 = K.alloc_local([4], "float32")
                    for j in range(4):
                        K.assign(P[j], K.float32(0.0))
                        K.assign(E63[j], K.float32(0.0))
                    pw = K.alloc_local([4], "float32")
                    for rr in range(8):
                        K.ptx["ld.shared.v4.f32"](
                            pw[0], pw[1], pw[2], pw[3], K.address_of(s_part[cyc & 1, rr, col0])
                        )
                        if rr == 2:
                            for j in range(4):
                                K.assign(R0[j], E63[j])
                        if rr == 6:
                            for j in range(4):
                                K.assign(R1[j], E63[j])
                        with K.If(rr == r_grp), K.Then():
                            for j in range(4):
                                K.assign(P[j], E63[j])
                        for j in range(4):
                            K.assign(E63[j], E63[j] + pw[j])
                    rng_end(_t)
                    # (no second barrier: the partials are double-buffered, and the barrier of the next
                    #  chunk separates this chunk's reads from the write two chunks later)
                    _t = rng("P.scan_c")
                    # gate vectors for cg1's state epilogue (warp 0 covers all 128 channels)
                    with K.If(r_grp == 0), K.Then():
                        gv8 = K.alloc_local([8], "float32")
                        for j in range(4):
                            K.ptx.ex2.approx.ftz.f32(gv8[j], R0[j])
                            K.ptx.ex2.approx.ftz.f32(gv8[4 + j], E63[j])
                        gvr = K.alloc_local([2], "uint32")
                        pack_bf16x2(gvr[0], gv8[0], gv8[1])
                        pack_bf16x2(gvr[1], gv8[2], gv8[3])
                        K.ptx["st.shared.v2.b32"](
                            K.address_of(s_gate_r[cyc & 3, col0]), gvr[0], gvr[1]
                        )
                        K.ptx["st.shared.v4.f32"](
                            K.address_of(s_gate_e[cyc & 3, col0]), gv8[4], gv8[5], gv8[6], gv8[7]
                        )
                    # per-thread block factors
                    Rsel = K.alloc_local([4], "float32")
                    fpr = K.alloc_local([4], "float32")  # 2^(G2_63 - R)   (KG factor)
                    f10 = K.alloc_local([4], "float32")  # 2^(R1-R0), pre-squared once per chunk
                    for j in range(4):
                        K.assign(Rsel[j], K.Select(hi_block != 0, R1[j], R0[j]))
                        K.ptx.ex2.approx.ftz.f32(fpr[j], E63[j] - Rsel[j])
                    for j in range(
                        4
                    ):  # lo threads: f10 = 1 (their "own-reference" tile is the stage tile)
                        K.ptx.ex2.approx.ftz.f32(
                            f10[j],
                            K.Select(
                                hi_block != 0, (R1[j] - R0[j]) * K.float32(0.5), K.float32(0.0)
                            ),
                        )
                        K.assign(f10[j], f10[j] * f10[j])
                    fprb = K.alloc_local([2], "uint32")
                    f10b = K.alloc_local([2], "uint32")
                    for p in range(2):
                        pack_bf16x2(fprb[p], fpr[2 * p], fpr[2 * p + 1])
                        pack_bf16x2(f10b[p], f10[2 * p], f10[2 * p + 1])
                    # G2_t - R (log2 exponent of this thread's 32 (t,k) cells) as one add per cell with the
                    # per-channel P - R precomputed: P, Rsel, R0/R1/E63 die here
                    PmR = K.alloc_local([4], "float32")
                    for j in range(4):
                        K.assign(PmR[j], P[j] - Rsel[j])
                    for i in range(8):
                        for j in range(4):
                            K.assign(G[i * 4 + j], G[i * 4 + j] + PmR[j])
                    rng_end(_t)
                    _t = rng("P.wait_stage")
                    p_stage.full.wait(st_stage.stage, st_stage.phase)
                    rng_end(_t)
                    _t = rng("P.norm")
                    if cluster_norm:
                        # Helper rank 2 publishes this chain rank's norm tile into the same two-stage
                        # slots used by the local implementation.
                        K.cuda.mbarrier_wait_acquire_cluster(
                            rnorm_full.ptr_to([st_stage.stage]), (n >> 1) & 1
                        )
                    else:
                        # Per-token 1/sqrt(sum x^2 + eps) of this warp's eight k and eight q rows.
                        nacc = K.alloc_local([4], "uint32")
                        for pp in range(4):
                            K.assign(nacc[pp], K.uint32(0))
                        nw = K.alloc_local([4], "uint32")
                        for j in range(8):
                            K.ptx["ld.shared.v4.b32"](
                                nw[0],
                                nw[1],
                                nw[2],
                                nw[3],
                                K.ptx.addr(base_k, norm_off + (K.int32(16 * j) ^ norm_x)),
                            )
                            for pp in range(4):
                                K.ptx.fma.rn.bf16x2(nacc[pp], nw[pp], nw[pp], nacc[pp])
                        K.ptx.add.rn.bf16x2(nacc[0], nacc[0], nacc[1])
                        K.ptx.add.rn.bf16x2(nacc[2], nacc[2], nacc[3])
                        K.ptx.add.rn.bf16x2(nacc[0], nacc[0], nacc[2])
                        nhalf = K.local_scalar("float32", init=bf16_lo(nacc[0]) + bf16_hi(nacc[0]))
                        nother = K.local_scalar("uint32")
                        K.ptx.shfl_sync.bfly.b32(
                            nother,
                            K.reinterpret("uint32", nhalf),
                            K.uint32(1),
                            K.uint32(0x1F),
                            K.uint32(0xFFFFFFFF),
                        )
                        nr = K.local_scalar("float32")
                        K.ptx.rsqrt.approx.ftz.f32(
                            nr, (nhalf + K.reinterpret("float32", nother)) + K.float32(1e-6)
                        )
                        with K.If(norm_is_q), K.Then():
                            K.assign(nr, nr * scale)
                        with K.If(norm_wr), K.Then():
                            K.ptx.st.shared.f32(
                                K.address_of(
                                    s_norm[
                                        st_stage.stage, lane >> 4, r_grp * 8 + ((lane & 15) >> 1)
                                    ]
                                ),
                                nr,
                            )
                        K.ptx["bar.warp.sync"](K.uint32(0xFFFFFFFF))
                    rq = K.alloc_local([8], "float32")
                    rk = K.alloc_local([8], "float32")
                    K.ptx["ld.shared.v4.f32"](
                        rk[0],
                        rk[1],
                        rk[2],
                        rk[3],
                        K.address_of(s_norm[st_stage.stage, 0, r_grp * 8]),
                    )
                    K.ptx["ld.shared.v4.f32"](
                        rk[4],
                        rk[5],
                        rk[6],
                        rk[7],
                        K.address_of(s_norm[st_stage.stage, 0, r_grp * 8 + 4]),
                    )
                    K.ptx["ld.shared.v4.f32"](
                        rq[0],
                        rq[1],
                        rq[2],
                        rq[3],
                        K.address_of(s_norm[st_stage.stage, 1, r_grp * 8]),
                    )
                    K.ptx["ld.shared.v4.f32"](
                        rq[4],
                        rq[5],
                        rq[6],
                        rq[7],
                        K.address_of(s_norm[st_stage.stage, 1, r_grp * 8 + 4]),
                    )
                    # rows past the sequence end: zero every derived operand tile row (X0/Q0/Y/KG/J1)
                    if not fixed and not uniform:
                        if packed_tail_hoist:
                            # Only the final chunk can contain invalid rows.  H96 benefits from one
                            # uniform guard in the eight-sequence dispatch; other packed shapes retain
                            # the straight-line selects below because they did not reproduce a gain.
                            with K.If((n + 1) * CHUNK > seq_len), K.Then():
                                for i in range(8):
                                    rvalid = n * CHUNK + (r_grp * 8 + i) < seq_len
                                    K.assign(rk[i], K.Select(rvalid, rk[i], K.float32(0.0)))
                                    K.assign(rq[i], K.Select(rvalid, rq[i], K.float32(0.0)))
                        else:
                            for i in range(8):
                                rvalid = n * CHUNK + (r_grp * 8 + i) < seq_len
                                K.assign(rk[i], K.Select(rvalid, rk[i], K.float32(0.0)))
                                K.assign(rq[i], K.Select(rvalid, rq[i], K.float32(0.0)))
                    if cluster_norm:
                        # Do not release the helper's single local source until all eight chain
                        # warps have materialized their row factors into registers.
                        K.ptx.bar.sync(K.uint32(BAR_RNORM), K.uint32(BAR_RNORM_N))
                        with K.If(r_grp == 0), K.Then():
                            with K.If(elected()), K.Then():
                                rem_release = K.local_scalar("uint32")
                                K.ptx.mapa.shared__cluster.u32(
                                    rem_release,
                                    K.cuda.cvta_generic_to_shared(
                                        helper_release.ptr_to([cluster_rank])
                                    ),
                                    K.uint32(2),
                                )
                                K.ptx["mbarrier.arrive.release.cluster.shared::cluster.b64"](
                                    rem_release
                                )
                    rng_end(_t)
                    _t = rng("P.wait_bufs")
                    # buffers this chunk writes: KG[stage] (after G7 two chunks ago), J1 (after G1_1 last chunk)
                    p_prep.empty.wait(st_prep.stage, st_prep.phase)
                    p_j1.empty.wait(st_j1.stage, st_j1.phase)
                    K.ptx[
                        FENCE_ASYNC
                    ]()  # async-proxy (MMA) reads of those tiles precede our generic writes
                    rng_end(_t)
                    _t = rng("P.pass2")
                    # ---------------- pass 2: gated operand tiles ---------------
                    is_hi = hi_block != 0
                    ev = K.alloc_local([4], "float32")
                    eiv = K.alloc_local([4], "float32")
                    sf = K.alloc_local([2], "uint64")  # combined norm/exponent scale pairs
                    ow = K.alloc_local([4], "uint32")
                    for i in range(8):
                        t = r_grp * 8 + i
                        for j in range(4):
                            K.ptx.ex2.approx.ftz.f32(ev[j], G[i * 4 + j])
                            K.ptx.ex2.approx.ftz.f32(eiv[j], K.float32(0.0) - G[i * 4 + j])
                        # Round each combined row-normalization/exponent factor once, then scale the
                        # original packed BF16 words.  This avoids widening q/k and keeps only one
                        # pair of FP32x2 temporaries live across the three operand variants.
                        lds64(base_q, xo_stage[i], w0, w1)
                        for p in range(2):
                            mul2(sf[p], rq[i], rq[i], ev[2 * p], ev[2 * p + 1])
                            pack_bf16x2(ow[2 + p], K.cuda.float2_x(sf[p]), K.cuda.float2_y(sf[p]))
                            K.ptx.mul.rn.bf16x2(ow[2 + p], (w0, w1)[p], ow[2 + p])
                        lds64(base_k, xo_stage[i], w0, w1)
                        # Y = kh * einv -> g region row 128+t ; KG = Y * fpr -> s_kg[stage]
                        for p in range(2):
                            mul2(sf[p], rk[i], rk[i], eiv[2 * p], eiv[2 * p + 1])
                            pack_bf16x2(ow[p], K.cuda.float2_x(sf[p]), K.cuda.float2_y(sf[p]))
                            K.ptx.mul.rn.bf16x2(ow[p], (w0, w1)[p], ow[p])
                        sts64(base_g, xo_stage[i], ow[0], ow[1])
                        for p in range(2):
                            K.ptx.mul.rn.bf16x2(ow[p], ow[p], fprb[p])
                        sts64(base_kg, xo_stage[i] - xo_adj, ow[0], ow[1])
                        # X = kh * e ; Q = qh * e
                        for p in range(2):
                            mul2(sf[p], rk[i], rk[i], ev[2 * p], ev[2 * p + 1])
                            pack_bf16x2(ow[p], K.cuda.float2_x(sf[p]), K.cuda.float2_y(sf[p]))
                            K.ptx.mul.rn.bf16x2(ow[p], (w0, w1)[p], ow[p])
                        # one straight-line path for both sub-blocks: hi threads store their own-reference
                        # X1/Q1 to J1 (single predicated stores), then everybody applies f10 (= 1 for lo
                        # threads) and stores the R0-referenced X0/Q0 to the stage
                        with K.If(is_hi), K.Then():
                            sts64(base_j1, xo_stage[i] - xo_adj - 32 * 128, ow[0], ow[1])
                        with K.If(is_hi), K.Then():
                            sts64(base_j1, xo_stage[i] - xo_adj, ow[2], ow[3])
                        for p in range(2):
                            K.ptx.mul.rn.bf16x2(ow[p], ow[p], f10b[p])
                        sts64(base_k, xo_stage[i], ow[0], ow[1])
                        for p in range(2):
                            K.ptx.mul.rn.bf16x2(ow[2 + p], ow[2 + p], f10b[p])
                        sts64(base_q, xo_stage[i], ow[2], ow[3])
                        # ---- pass 1 of chunk n+1 for this token: its gate sigmoids/cumsum go into the G slots
                        # freed by this token's exponents; then the token's gate logits of chunk n+2 are fetched
                        # into the freed gw words (for the last chunk this computes unused values) ----
                        pass1_token(n + 1, i)
                        with K.If(n + 2 < nch), K.Then():
                            issue_g_load_token(n + 2, i)
                    K.ptx[FENCE_ASYNC]()
                    rng_end(_t)
                    p_prep.full.arrive(st_prep.stage)
                    p_j1.full.arrive(st_j1.stage)
                    st_prep.advance()
                    st_j1.advance()
                    st_stage.advance()
                    K.assign(cyc, cyc + K.int32(1))
                advance_work(work, st_w)
            if cluster_norm:
                with K.If(cluster_rank == K.int32(2)), K.Then():
                    helper_norm(0, r_grp)

        # =====================================================================
        # CG0: Akk epilogue, hierarchical inverse (-> TB bf16), G3 issue.      warps 8-11
        # =====================================================================
        with cg0:
            if not mixed:
                st_g1 = K.PipelineState(
                    2, phase=0
                )  # consumer p_g1.full ; producer p_g1.empty (same index)
                st_g1b = K.PipelineState(1, phase=0)  # consumer of m_g1b (D1 done)
                st_t = K.PipelineState(1, phase=1)  # producer p_t
                st_beta = K.PipelineState(2, phase=0)  # consumer p_beta
                st_tc = K.PipelineState(
                    1, phase=0
                )  # consumer of p_t.full (this warpgroup's own T, all 128 arrivals)
                st_v = K.PipelineState(1, phase=0)  # consumer of p_v.full (G3)
                st_ue = K.PipelineState(1, phase=1)  # waits p_u.empty before G3 overwrites U^T
                st_aqk0 = K.PipelineState(1, phase=1)  # producer of p_aqk (warps 2/3)
            dT0, offT0 = s_T.encode(major="k", mma_k=16)
            dV0, offV0 = s_v.encode(major="mn", mma_k=16)

            def mma_ss0(dcol, a_desc, a_off, b_desc, b_off, n_k, idesc, accumulate):
                a_base = K.local_scalar("uint64", init=a_desc)
                b_base = K.local_scalar("uint64", init=b_desc)
                fa = runtime_off(a_off, n_k) if compact else (lambda kp: K.uint64(a_off(kp)))
                fb = runtime_off(b_off, n_k) if compact else (lambda kp: K.uint64(b_off(kp)))
                mma_issue(
                    dcol, lambda kp: a_base + fa(kp), b_base, fb, n_k, idesc, accumulate, tmem
                )

            tmem = tmem_preamble()
            tid1 = K.tid_in_role()
            lane = K.lane_id()
            lw = K.warp_id_in_role()
            rowbits = (tid1 << 16) & 0x600000

            def tmem_at(col, rb=None):
                return K.Cast("uint32", tmem[0] + col + (rowbits if rb is None else rb))

            def ld32(regs, col):
                K.ptx[TC_LD32](*(regs[i] for i in range(32)), tmem_at(col))

            def bar_inv():
                K.ptx.bar.sync(K.uint32(BAR_INV), K.uint32(BAR_INV_N))

            # ---- GDN-style hierarchical inverse of (I + A) on the f16 tile s_A ----
            def neg_pack(dst_word, a, b):
                neg = K.local_scalar("uint64")
                K.ptx.sub.rn.f32x2(neg, f2(K.float32(0.0), K.float32(0.0)), f2(a, b))
                pack_f16x2(dst_word, K.cuda.float2_x(neg), K.cuda.float2_y(neg))

            def invert_diag_8x8(av, block8):
                r = block8 + (lane & 7)
                wds = K.alloc_local([4], "uint32")
                K.ptx["ld.shared.v4.b32"](wds[0], wds[1], wds[2], wds[3], av.ptr_to(r, block8))
                row = [K.local_scalar("float32") for _ in range(8)]
                for p in range(4):
                    K.idioms.cast_f16x2_to_f32x2(row, p, wds[p])
                for i in range(8):
                    with K.If((lane & 7) == i), K.Then():
                        K.assign(row[i], K.float32(1.0))
                rs = K.local_scalar("float32")
                pv = K.local_scalar("uint32")
                for src in range(7):
                    K.ptx.neg.f32(rs, row[src])
                    for i in range(7):
                        if i < src:
                            K.ptx.shfl_sync.idx.b32(
                                pv,
                                K.reinterpret("uint32", row[i]),
                                K.uint32(src),
                                K.uint32(0x181F),
                                K.uint32(0xFFFFFFFF),
                            )
                            with K.If((lane & 7) > src), K.Then():
                                K.assign(row[i], row[i] + rs * K.reinterpret("float32", pv))
                    with K.If((lane & 7) > src), K.Then():
                        K.assign(row[src], rs)
                for p in range(4):
                    pack_f16x2(wds[p], row[2 * p], row[2 * p + 1])
                K.ptx["st.shared.v4.b32"](av.ptr_to(r, block8), wds[0], wds[1], wds[2], wds[3])

            def ldm_x4(insn, dst, av, base_row, base_col):
                lm = lane >> 3
                row = base_row + (lane & 7) + (lm & 1) * 8
                col = base_col + (lm >> 1) * 8
                K.ptx[insn](dst[0], dst[1], dst[2], dst[3], av.ptr_to(row, col))

            def stm_x4(src, av, base_row, base_col):
                lm = lane >> 3
                row = base_row + (lane & 7) + (lm & 1) * 8
                col = base_col + (lm >> 1) * 8
                K.ptx[STM_X4](av.ptr_to(row, col), src[0], src[1], src[2], src[3])

            def mma_k8_zero(acc, a, b):
                K.ptx[MMA_K8](
                    acc[0], acc[1], acc[2], acc[3], a[0], a[1], b[0],
                    K.float32(0.0), K.float32(0.0), K.float32(0.0), K.float32(0.0),
                )  # fmt: skip

            def mma_k16(acc, a, b, acc_off, b_off, accumulate):
                c = [acc[acc_off + i] for i in range(4)] if accumulate else [K.float32(0.0)] * 4
                K.ptx[MMA_K16](
                    *(acc[acc_off + i] for i in range(4)),
                    a[0],
                    a[1],
                    a[2],
                    a[3],
                    b[b_off],
                    b[b_off + 1],
                    *c,
                )

            def inverse_8_to_16(av, b16):
                a = K.alloc_local([2], "uint32")
                b = K.alloc_local([1], "uint32")
                acc = K.alloc_local([4], "float32")
                dm = K.local_scalar("uint32")
                cm = K.local_scalar("uint32")
                K.ptx[LDM_X1](dm, av.ptr_to(b16 + 8 + (lane & 7), b16 + 8))
                K.ptx[LDM_X1T](cm, av.ptr_to(b16 + 8 + (lane & 7), b16))
                K.assign(a[0], dm)
                K.assign(a[1], dm)
                K.assign(b[0], cm)
                mma_k8_zero(acc, a, b)
                neg_pack(a[0], acc[0], acc[1])
                neg_pack(a[1], acc[2], acc[3])
                K.ptx[LDM_X1T](b[0], av.ptr_to(b16 + (lane & 7), b16))
                mma_k8_zero(acc, a, b)
                pack_f16x2(dm, acc[0], acc[1])
                K.ptx[STM_X1](av.ptr_to(b16 + 8 + (lane & 7), b16), dm)

            def inverse_16_to_32(av, b32):
                a = K.alloc_local([4], "uint32")
                b = K.alloc_local([4], "uint32")
                acc = K.alloc_local([8], "float32")
                out = K.alloc_local([4], "uint32")
                ldm_x4(LDM_X4, a, av, b32 + 16, b32 + 16)
                ldm_x4(LDM_X4T, b, av, b32 + 16, b32)
                mma_k16(acc, a, b, 0, 0, False)
                mma_k16(acc, a, b, 4, 2, False)
                for p in range(4):
                    neg_pack(a[p], acc[2 * p], acc[2 * p + 1])
                ldm_x4(LDM_X4T, b, av, b32, b32)
                mma_k16(acc, a, b, 0, 0, False)
                mma_k16(acc, a, b, 4, 2, False)
                for p in range(4):
                    pack_f16x2(out[p], acc[2 * p], acc[2 * p + 1])
                stm_x4(out, av, b32 + 16, b32)

            def inverse_32_to_64_tb(av, tb, half_warp, stg):
                """T21 = -T22 A21 T11 (f16 mma.sync), scaled by beta_j and written as bf16 TB rows
                32+16*half_warp.. of `tb`; s_A keeps A21 (never overwritten -> no intra-stage barrier)."""
                rb = 32 + half_warp * 16
                a0 = K.alloc_local([4], "uint32")
                a1 = K.alloc_local([4], "uint32")
                b00 = K.alloc_local([4], "uint32")
                b01 = K.alloc_local([4], "uint32")
                b10 = K.alloc_local([4], "uint32")
                b11 = K.alloc_local([4], "uint32")
                acc = K.alloc_local([16], "float32")
                pa = K.alloc_local([8], "uint32")
                o0 = K.alloc_local([4], "uint32")
                o1 = K.alloc_local([4], "uint32")
                ldm_x4(LDM_X4, a0, av, rb, 32)
                ldm_x4(LDM_X4, a1, av, rb, 48)
                ldm_x4(LDM_X4T, b00, av, 32, 0)
                ldm_x4(LDM_X4T, b01, av, 32, 16)
                ldm_x4(LDM_X4T, b10, av, 48, 0)
                ldm_x4(LDM_X4T, b11, av, 48, 16)
                mma_k16(acc, a0, b00, 0, 0, False)
                mma_k16(acc, a0, b00, 4, 2, False)
                mma_k16(acc, a0, b01, 8, 0, False)
                mma_k16(acc, a0, b01, 12, 2, False)
                mma_k16(acc, a1, b10, 0, 0, True)
                mma_k16(acc, a1, b10, 4, 2, True)
                mma_k16(acc, a1, b11, 8, 0, True)
                mma_k16(acc, a1, b11, 12, 2, True)
                for p in range(8):
                    neg_pack(pa[p], acc[2 * p], acc[2 * p + 1])
                ldm_x4(LDM_X4T, b00, av, 0, 0)
                ldm_x4(LDM_X4T, b01, av, 0, 16)
                ldm_x4(LDM_X4T, b10, av, 16, 0)
                ldm_x4(LDM_X4T, b11, av, 16, 16)
                for i in range(4):
                    K.assign(a0[i], pa[i])
                    K.assign(a1[i], pa[4 + i])
                mma_k16(acc, a0, b00, 0, 0, False)
                mma_k16(acc, a0, b00, 4, 2, False)
                mma_k16(acc, a0, b01, 8, 0, False)
                mma_k16(acc, a0, b01, 12, 2, False)
                mma_k16(acc, a1, b10, 0, 0, True)
                mma_k16(acc, a1, b10, 4, 2, True)
                mma_k16(acc, a1, b11, 8, 0, True)
                mma_k16(acc, a1, b11, 12, 2, True)
                # accumulator pair (acc[2pp], acc[2pp+1]) sits at column 8*(pp>>1) + 2*(lane&3) + {0,1}
                bq = K.alloc_local([8], "float32")
                for qq in range(4):
                    K.ptx["ld.shared.v2.f32"](
                        bq[2 * qq],
                        bq[2 * qq + 1],
                        K.address_of(s_beta[stg, 8 * qq + 2 * (lane & 3)]),
                    )
                scaled = K.local_scalar("uint64")
                for p in range(4):
                    q0 = 2 * (p >> 1)
                    mul2(scaled, acc[2 * p], acc[2 * p + 1], bq[q0], bq[q0 + 1])
                    pack_bf16x2(o0[p], K.cuda.float2_x(scaled), K.cuda.float2_y(scaled))
                    mul2(scaled, acc[8 + 2 * p], acc[8 + 2 * p + 1], bq[4 + q0], bq[4 + q0 + 1])
                    pack_bf16x2(o1[p], K.cuda.float2_x(scaled), K.cuda.float2_y(scaled))
                stm_x4(o0, tb, rb, 0)
                stm_x4(o1, tb, rb, 16)

            def diag_blocks_to_tb(stg):
                """warps 2/3: T11 / T22 (f16, s_A) * beta_j -> bf16 TB blocks of s_T (one row per lane)."""
                blk = lw - 2
                trow = blk * 32 + lane
                tc0 = blk * 32
                for m in range(4):
                    lds128(s_A, trow, tc0 + 8 * m, wds, 4 * m)
                for p in range(16):
                    K.idioms.cast_f16x2_to_f32x2(acc32, p, wds[p])
                bcol = K.alloc_local([4], "float32")
                scaled = K.local_scalar("uint64")
                for m in range(8):
                    K.ptx["ld.shared.v4.f32"](
                        bcol[0], bcol[1], bcol[2], bcol[3], K.address_of(s_beta[stg, tc0 + 4 * m])
                    )
                    for p in range(2):
                        mul2(
                            scaled,
                            acc32[4 * m + 2 * p],
                            acc32[4 * m + 2 * p + 1],
                            bcol[2 * p],
                            bcol[2 * p + 1],
                        )
                        pack_bf16x2(
                            wds[2 * m + p], K.cuda.float2_x(scaled), K.cuda.float2_y(scaled)
                        )
                for m in range(4):
                    sts128(s_T, trow, tc0 + 8 * m, wds, 4 * m)

            # zero the never-written upper 32x32 block of s_A once
            zw = K.alloc_local([4], "uint32")
            for i in range(4):
                K.assign(zw[i], K.uint32(0))
            with K.If(lw < 2), K.Then():  # 64 threads: rows 0..31, cols 32..63
                zr = lw * 16 + (lane >> 1)
                zc = 32 + (lane & 1) * 16
                sts128(s_A, zr, zc, zw, 0)
                sts128(s_A, zr, zc + 8, zw, 0)
            K.ptx[FENCE_ASYNC]()
            bar_inv()

            acc32 = K.alloc_local([32], "float32")
            wds = K.alloc_local(
                [32], "uint32"
            )  # [0..15] Akk/TB/W words, [16..31] Aqk D1 words (warps 2/3)
            st_w, work = begin_work()
            cyc = K.local_scalar("int32", init=K.int32(0))
            with K.While(work < num_work):
                _sq, _h, _ss, _sl, nch = work_coords(work)
                with K.serial(nch) as n:
                    stg = cyc & 1
                    ph2 = (cyc >> 1) & 1
                    _t = rng("C0.wait_g1")
                    if mixed:
                        p_beta.full.wait(stg, ph2)
                        p_g1.full.wait(stg, ph2)
                    else:
                        p_beta.full.wait(st_beta.stage, st_beta.phase)
                        p_g1.full.wait(st_g1.stage, st_g1.phase)
                    bar_inv()  # every warp finished reading s_A / s_T sources of the previous chunk
                    rng_end(_t)
                    _t = rng("C0.akk")
                    # ---- Akk -> s_A (f16, strictly-lower * beta_t) on warps 0/1; Aqk (causal, bf16)
                    # -> the dead Y rows 128..191 of this chunk's stage on warps 2/3 -------------
                    # D0 is M128: warp 0 owns Akk rows 0..31, warp 1 rows 32..63, warps 2/3 the
                    # same rows of Aqk (lanes 64..127).
                    # D1 is M64 layout F: lanes 0..15 of warps 0/1 (2/3) own rows 32..47/48..63.
                    acc_b = K.alloc_local([32], "float32")
                    ld32(acc32, TM_D0 + stg * 64)
                    K.ptx[WAIT_LD]()
                    if mixed:
                        p_g1.empty.arrive(stg)
                    else:
                        p_g1.empty.arrive(
                            st_g1.stage
                        )  # D0 buffer consumed: mma0 may run G1_0 of chunk n+2 into it
                        st_g1.advance()

                    def akk_rows(acc, row_base, col_base, diag_lim):
                        """Scale one 32-column Akk row block into s_A; entries with column >= diag_lim are
                        zeroed (diag_lim None: full block; an int: lane + shift; compact shape: a runtime
                        limit shared by both warps)."""
                        t = row_base + lane
                        bt = K.local_scalar("float32")
                        K.ptx.ld.shared.f32(bt, K.address_of(s_beta[stg, t]))
                        scaled = K.local_scalar("uint64")
                        for p in range(16):
                            mul2(scaled, acc[2 * p], acc[2 * p + 1], bt, bt)
                            lo = K.cuda.float2_x(scaled)
                            hi = K.cuda.float2_y(scaled)
                            if diag_lim is not None:
                                lim = lane + diag_lim if isinstance(diag_lim, int) else diag_lim
                                lo = K.Select(K.int32(2 * p) < lim, lo, K.float32(0.0))
                                hi = K.Select(K.int32(2 * p + 1) < lim, hi, K.float32(0.0))
                            pack_f16x2(wds[p], lo, hi)
                        for m in range(4):
                            sts128(s_A, t, col_base + 8 * m, wds, 4 * m)

                    if compact:
                        with K.If(lw < 2), K.Then():
                            # warp 0: rows 0..31 strictly lower (limit lane); warp 1: rows 32..63 full block
                            akk_rows(
                                acc32, lw * 32, K.int32(0), K.Select(lw == 0, lane, K.int32(64))
                            )
                    else:
                        with K.If(lw == 0), K.Then():
                            akk_rows(acc32, 0, 0, 0)
                        with K.If(lw == 1), K.Then():
                            akk_rows(acc32, 32, 0, None)
                    with K.If(lw >= 2), K.Then():
                        with K.If(lw == 2), K.Then():
                            for j in range(32):
                                K.assign(acc32[j], K.Select(j <= lane, acc32[j], K.float32(0.0)))
                        for p in range(16):
                            pack_bf16x2(wds[p], acc32[2 * p], acc32[2 * p + 1])
                    with K.If(lw < 2), K.Then():
                        m_g1b.wait(0, stg if mixed else st_g1b.phase)
                        ld32(acc_b, TM_D1)
                        K.ptx[WAIT_LD]()
                        with K.If(lane < 16), K.Then():
                            if compact:
                                # D1 rows: warp 0 rows 32..47 (limit lane), warp 1 rows 48..63 (limit lane+16)
                                akk_rows(acc_b, 32 + lw * 16, K.int32(32), lane + lw * 16)
                            else:
                                with K.If(lw == 0):
                                    with K.Then():
                                        akk_rows(acc_b, 32, 32, 0)
                                    with K.Else():
                                        akk_rows(acc_b, 48, 32, 16)
                        p_g1b.arrive(0)  # Akk half of D1 consumed
                    bar_inv()
                    rng_end(_t)
                    _t = rng("C0.inv")
                    # warps 2/3 finish the Aqk epilogue in the windows where only warps 0/1 work:
                    # the D1 rows during the 8x8 diagonal inverse, the stores during the 16->32 merge
                    with K.If(lw < 2):
                        with K.Then():
                            invert_diag_8x8(s_A, ((lw * 32 + lane) >> 3) * 8)
                        with K.Else():
                            m_g1b.wait(0, stg if mixed else st_g1b.phase)
                            ld32(acc_b, TM_D1)
                            K.ptx[WAIT_LD]()
                            with K.If(lane < 16), K.Then():
                                dshift = K.Select(lw == 2, K.int32(0), K.int32(16))
                                for j in range(32):
                                    K.assign(
                                        acc_b[j],
                                        K.Select(j <= lane + dshift, acc_b[j], K.float32(0.0)),
                                    )
                                for p in range(16):
                                    pack_bf16x2(wds[16 + p], acc_b[2 * p], acc_b[2 * p + 1])
                            p_g1b.arrive(0)  # Aqk half of D1 consumed
                    if not mixed:
                        st_g1b.advance()
                    bar_inv()
                    inverse_8_to_16(s_A, lw * 16)
                    bar_inv()
                    with K.If(lw < 2):
                        with K.Then():
                            inverse_16_to_32(s_A, lw * 32)
                        with K.Else():
                            # G1_1 (async proxy) finished reading the Y rows: publish Aqk there.
                            K.ptx[FENCE_ASYNC]()
                            svy = s_stage[stg]
                            arow = K.Select(lw == 2, lane, 32 + lane)
                            for m in range(4):
                                sts128(svy, 128 + arow, 8 * m, wds, 4 * m)
                            with K.If(lane < 16), K.Then():
                                arow2 = K.Select(lw == 2, 32 + lane, 48 + lane)
                                for m in range(4):
                                    sts128(svy, 128 + arow2, 32 + 8 * m, wds, 16 + 4 * m)
                            # the never-computed upper-right 32x32 block must be zero for G6
                            zi = (lw - 2) * 32 + lane
                            zr = zi >> 1
                            zc = 32 + (zi & 1) * 16
                            sts128(svy, 128 + zr, zc, zw, 0)
                            sts128(svy, 128 + zr, zc + 8, zw, 0)
                            K.ptx[FENCE_ASYNC]()
                            p_aqk.full.arrive(K.int32(0) if mixed else st_aqk0.stage)
                    if not mixed:
                        st_aqk0.advance()
                    bar_inv()
                    rng_end(_t)
                    _t = rng("C0.t64")
                    # ---- last merge writes TB = T * beta_j (bf16) straight into s_T: warps 0/1 compute
                    # T21 with mma.sync, warps 2/3 convert the diagonal blocks T11 / T22 meanwhile ----
                    if mixed:
                        p_t.empty.wait(K.int32(0), stg ^ K.int32(1))
                    else:
                        p_t.empty.wait(
                            st_t.stage, st_t.phase
                        )  # G3 of the previous chunk finished reading s_T
                    K.ptx[FENCE_ASYNC]()
                    with K.If(lw < 2):
                        with K.Then():
                            inverse_32_to_64_tb(s_A, s_T, lw, stg)
                        with K.Else():
                            diag_blocks_to_tb(stg)
                    K.ptx[FENCE_ASYNC]()
                    p_t.full.arrive(K.int32(0) if mixed else st_t.stage)
                    if not mixed:
                        st_t.advance()
                    if not cluster_norm:
                        p_beta.empty.arrive(
                            stg if mixed else st_beta.stage
                        )  # last beta read of this chunk
                    if not mixed:
                        st_beta.advance()
                    rng_end(_t)
                    # ---- G3: U^T = V^T TB^T, issued here by warp 0 (no handoff); TB is released by G4'' ----
                    _t = rng("C0.g3")
                    with K.If(lw == 0), K.Then():
                        if mixed:
                            p_t.full.wait(K.int32(0), stg)
                        else:
                            p_t.full.wait(
                                st_tc.stage, st_tc.phase
                            )  # every cg0 thread's TB stores are published
                            st_tc.advance()
                        if cluster_norm:
                            with K.If(elected()), K.Then():
                                rem_beta_release = K.local_scalar("uint32")
                                K.ptx.mapa.shared__cluster.u32(
                                    rem_beta_release,
                                    K.cuda.cvta_generic_to_shared(
                                        helper_beta_release.ptr_to([cluster_rank * 2 + stg])
                                    ),
                                    K.uint32(2),
                                )
                                K.ptx["mbarrier.arrive.release.cluster.shared::cluster.b64"](
                                    rem_beta_release
                                )
                        if mixed:
                            p_v.full.wait(K.int32(0), stg)
                            p_u.empty.wait(K.int32(0), stg ^ K.int32(1))
                        else:
                            p_v.full.wait(st_v.stage, st_v.phase)
                            p_u.empty.wait(st_ue.stage, st_ue.phase)
                            st_ue.advance()
                        mma_ss0(TM_U, dV0.value, offV0, dT0.value, offT0, 4, ID_G23, False)
                        pr = elect_local()
                        p_v.empty.arrive(K.int32(0) if mixed else st_v.stage, pred=pr)
                        if not mixed:
                            st_v.advance()
                        m_g3.arrive(0, pred=pr)
                    rng_end(_t)
                    K.assign(cyc, cyc + K.int32(1))
                advance_work(work, st_w)
            if cluster_norm:
                with K.If(cluster_rank == K.int32(2)), K.Then():
                    helper_norm(1, lw)

        # =====================================================================
        # CG1: U / O / state epilogues.                                 warps 12-15
        # =====================================================================
        with cg1:
            st_u = K.PipelineState(1, phase=0)
            if not derive_cg1_vo:
                st_o = K.PipelineState(1, phase=0)
            st_sacc = K.PipelineState(1, phase=0)
            st_osm = K.PipelineState(1, phase=1)  # producer of p_osm
            st_pn = K.PipelineState(2, phase=0)  # consumer of p_prep.full, one chunk ahead
            if not derive_cg1_vo:
                st_v1 = K.PipelineState(1, phase=0)  # consumer of m_v1 (V1 done)
            tmem = tmem_preamble()
            tid2 = K.tid_in_role()
            lane = K.lane_id()
            lw = K.warp_id_in_role()
            rowbits = (tid2 << 16) & 0x600000

            def tmem_at2(col, extra_rows=0):
                return K.Cast("uint32", tmem[0] + col + rowbits + (extra_rows << 16))

            def ld32b(regs, col):
                K.ptx[TC_LD32](*(regs[i] for i in range(32)), tmem_at2(col))

            def st32b(col, regs):
                K.ptx[TC_ST32](tmem_at2(col), *(regs[i] for i in range(32)))

            fr = K.alloc_local([64], "float32")
            fr2 = K.alloc_local([32], "float32")
            wds = K.alloc_local([32], "uint32")
            fac = K.alloc_local([8], "float32")
            bfac = K.alloc_local([4], "uint32")

            def state_pass(nstg, gbase):
                """S~'(next) = bf16(S * 2^R0(next)) -> TM_ST ; S *= 2^(G2_63 of the next chunk) -> TM_S.

                `nstg` selects the next chunk's gate vectors.  `gbase` None reads the resident fp32
                state from TMEM; otherwise it is this lane's fp32 initial-state row (V-first layout:
                row v = TMEM lane, 128 contiguous k) in global memory."""
                state_pair = K.local_scalar("uint64")
                # Two 64-column rounds: both 32-column loads of a round are in flight before the one
                # wait::ld; each 32-column group is packed/scaled with only 16 packed words live.
                with K.serial(2) as half:
                    hc = half * 64
                    hs = half * 32
                    if gbase is None:
                        ld32b(fr, TM_S + hc)
                        K.ptx[TC_LD32](*(fr[32 + i] for i in range(32)), tmem_at2(TM_S + hc + 32))
                        K.ptx[WAIT_LD]()
                    else:
                        # Issue each aligned 32-byte initial-state span with one v8.b32 load. These
                        # rows are streaming, and the loader also issues L2 prefetches, so request
                        # L1::no_allocate here.
                        for m in range(8):
                            K.ptx["ld.global.L1::no_allocate.v8.b32"](
                                fr[8 * m],
                                fr[8 * m + 1],
                                fr[8 * m + 2],
                                fr[8 * m + 3],
                                fr[8 * m + 4],
                                fr[8 * m + 5],
                                fr[8 * m + 6],
                                fr[8 * m + 7],
                                h0.ptr_to([gbase + K.Cast("int64", hc + 8 * m)]),
                            )
                    for sub in range(2):
                        qc = hc + 32 * sub
                        fb = 32 * sub
                        # One packed load supplies eight bf16 midpoint factors; two fp32 loads
                        # supply the eight resident-state decay factors.
                        for m in range(4):
                            K.ptx["ld.shared.v4.b32"](
                                bfac[0],
                                bfac[1],
                                bfac[2],
                                bfac[3],
                                K.address_of(s_gate_r[nstg, qc + 8 * m]),
                            )
                            K.ptx["ld.shared.v4.f32"](
                                fac[0],
                                fac[1],
                                fac[2],
                                fac[3],
                                K.address_of(s_gate_e[nstg, qc + 8 * m]),
                            )
                            K.ptx["ld.shared.v4.f32"](
                                fac[4],
                                fac[5],
                                fac[6],
                                fac[7],
                                K.address_of(s_gate_e[nstg, qc + 8 * m + 4]),
                            )
                            for p in range(4):
                                pack_bf16x2(
                                    wds[4 * m + p],
                                    fr[fb + 8 * m + 2 * p],
                                    fr[fb + 8 * m + 2 * p + 1],
                                )
                                K.ptx.mul.rn.bf16x2(wds[4 * m + p], wds[4 * m + p], bfac[p])
                                mul2(
                                    state_pair,
                                    fr[fb + 8 * m + 2 * p],
                                    fr[fb + 8 * m + 2 * p + 1],
                                    fac[2 * p],
                                    fac[2 * p + 1],
                                )
                                K.assign(fr[fb + 8 * m + 2 * p], K.cuda.float2_x(state_pair))
                                K.assign(fr[fb + 8 * m + 2 * p + 1], K.cuda.float2_y(state_pair))
                        K.ptx[TC_ST16](
                            tmem_at2(TM_ST + hs + 16 * sub), *(wds[i] for i in range(16))
                        )
                    st32b(TM_S + hc, fr)
                    K.ptx[TC_ST32](tmem_at2(TM_S + hc + 32), *(fr[32 + i] for i in range(32)))
                K.ptx[WAIT_ST]()
                m_s.arrive(0)

            st_w, work = begin_work()
            cyc = K.local_scalar("int32", init=K.int32(0))
            with K.While(work < num_work):
                seq_idx, h_idx, seq_start, seq_len, nch = work_coords(work)
                with K.If(nch > 0), K.Then():
                    # initial state S0 (fp32, V-first) -> S~'(0) and the chunk-0-decayed resident state
                    p_prep.full.wait(st_pn.stage, st_pn.phase)  # chunk 0's gate vectors
                    gbase = K.local_scalar(
                        "int64",
                        init=(
                            (K.Cast("int64", seq_idx) * K.int64(H) + K.Cast("int64", h_idx))
                            * K.int64(D_HEAD)
                            + K.Cast("int64", tid2)
                        )
                        * K.int64(D_HEAD),
                    )
                    state_pass(cyc & 3, gbase)
                    st_pn.advance()
                with K.serial(nch) as n:
                    stg = cyc & 1
                    _t = rng("C1.wait_v1")
                    # ---- V1 = S~'^T X0^T f32 -> bf16 into the UB columns (A operand of G4''); UB(n-1) was
                    # consumed by G7(n-1), whose completion this warpgroup awaited before its state pass ----
                    m_v1.wait(0, stg if derive_cg1_vo else st_v1.phase)
                    if not derive_cg1_vo:
                        st_v1.advance()
                    rng_end(_t)
                    _t = rng("C1.v1")
                    ld32b(fr, TM_V1)
                    K.ptx[TC_LD32](*(fr[32 + i] for i in range(32)), tmem_at2(TM_V1 + 32))
                    K.ptx[WAIT_LD]()
                    for p in range(32):
                        pack_bf16x2(wds[p], fr[2 * p], fr[2 * p + 1])
                    st32b(TM_UB, wds)
                    K.ptx[WAIT_ST]()
                    m_v1b.arrive(0)
                    rng_end(_t)
                    _t = rng("C1.wait_u")
                    # ---- U^T f32 -> bf16 (A operand for G6/G7) -------------------
                    p_u.full.wait(st_u.stage, st_u.phase)
                    rng_end(_t)
                    _t = rng("C1.u")
                    ld32b(fr, TM_U)
                    K.ptx[TC_LD32](*(fr[32 + i] for i in range(32)), tmem_at2(TM_U + 32))
                    K.ptx[WAIT_LD]()
                    for p in range(32):
                        pack_bf16x2(wds[p], fr[2 * p], fr[2 * p + 1])
                    st32b(TM_UB, wds)
                    K.ptx[WAIT_ST]()
                    p_u.empty.arrive(st_u.stage)
                    st_u.advance()
                    m_ub.arrive(0)
                    rng_end(_t)
                    _t = rng("C1.wait_o")
                    # ---- O^T f32 -> bf16 -> s_o[t][v] via stmatrix.trans ------------
                    if derive_cg1_vo:
                        p_o.full.wait(0, stg)
                    else:
                        p_o.full.wait(st_o.stage, st_o.phase)
                    p_osm.empty.wait(st_osm.stage, st_osm.phase)
                    rng_end(_t)
                    _t = rng("C1.o")
                    for hh in range(2):
                        K.ptx[TC_LD256](*(fr[i] for i in range(32)), tmem_at2(TM_O, 16 * hh))
                        K.ptx[WAIT_LD]()
                        # regs[4i+2q+e] = O^T[v = 32w + 16hh + 8q + lane/4][t = 8i + 2(lane%4) + e]
                        for i in range(8):
                            for qq in range(2):
                                pack_bf16x2(
                                    wds[16 * hh + 2 * i + qq],
                                    fr[4 * i + 2 * qq],
                                    fr[4 * i + 2 * qq + 1],
                                )
                    # 8 stmatrix.x4.trans calls: call ci covers hh=ci//4, i in {2(ci%4), 2(ci%4)+1}, q in {0,1}
                    mm = lane >> 3
                    jj = lane & 7
                    for ci in range(8):
                        hh = ci // 4
                        ib = 2 * (ci % 4)
                        # matrix m = (i - ib)*2 + q  ->  i = ib + m//2, q = m%2
                        trow = (ib + (mm >> 1)) * 8 + jj
                        tcol = lw * 32 + 16 * hh + 8 * (mm & 1)
                        K.ptx[STM_X4T](
                            s_o.ptr_to(trow, tcol),
                            wds[16 * hh + 2 * ib + 0],
                            wds[16 * hh + 2 * ib + 1],
                            wds[16 * hh + 2 * ib + 2],
                            wds[16 * hh + 2 * ib + 3],
                        )
                    K.ptx[FENCE_ASYNC]()
                    p_o.empty.arrive(0 if derive_cg1_vo else st_o.stage)
                    if not derive_cg1_vo:
                        st_o.advance()
                    p_osm.full.arrive(st_osm.stage)
                    st_osm.advance()
                    rng_end(_t)
                    _t = rng("C1.wait_s")
                    # ---- S~'(n+1) = bf16(S_n * 2^R0(n+1)) ; S *= 2^(G2_63 of chunk n+1) — one TMEM pass --
                    # (decay must precede the next chunk's G7 accumulation; the bf16 operand copy is the
                    #  undecayed end-of-chunk state scaled per column k by the next chunk's 2^R0, which
                    #  makes G5's Q0 and G4's W' operands reference-free)
                    m_sacc.wait(0, st_sacc.phase)
                    st_sacc.advance()
                    with K.If(n + 1 < nch), K.Then():
                        p_prep.full.wait(st_pn.stage, st_pn.phase)  # chunk n+1's gate vectors
                    rng_end(_t)
                    _t = rng("C1.s")
                    with K.If(n + 1 < nch), K.Then():
                        state_pass((cyc + 1) & 3, None)
                        st_pn.advance()
                    K.assign(cyc, cyc + K.int32(1))
                    rng_end(_t)
                advance_work(work, st_w)
            # TMEM belongs only to the two chain CTAs.  The helper's cg1 warps join
            # this local rendezvous, then become norm-team lanes 4..7.
            K.ptx.bar.sync(K.uint32(BAR_CG1), K.uint32(BAR_CG1_N))
            with K.If(is_chain & (lw == 0)), K.Then():
                K.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
                K.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                    K.Cast("uint32", tmem[0]), K.uint32(TMEM_COLS)
                )
            if cluster_norm:
                with K.If(cluster_rank == K.int32(2)), K.Then():
                    helper_norm(1, lw + K.int32(4))

        with auxg:
            # =====================================================================
            # MMA issuer 0: chunk-local GEMMs G1 (Akk/Aqk), G2 (W'^T), G3 (U_pre^T).  warp 12
            # =====================================================================
            with mma0:
                # G1_0 of every chunk, issued as soon as prep publishes the chunk (double-buffered D0)
                st_prep = K.PipelineState(2, phase=0)
                st_g1 = K.PipelineState(2, phase=1)  # producer p_g1
                st_stg = K.PipelineState(
                    2, phase=0
                )  # ledger for p_stage.empty commits (G1_0 reads X0/Q0 rows)
                tmem = tmem_preamble()

                def mma_ss(dcol, a_desc, a_off, b_desc, b_off, n_k, idesc, accumulate):
                    a_base = K.local_scalar("uint64", init=a_desc)
                    b_base = K.local_scalar("uint64", init=b_desc)
                    fa = runtime_off(a_off, n_k) if compact else (lambda kp: K.uint64(a_off(kp)))
                    fb = runtime_off(b_off, n_k) if compact else (lambda kp: K.uint64(b_off(kp)))
                    mma_issue(
                        dcol, lambda kp: a_base + fa(kp), b_base, fb, n_k, idesc, accumulate, tmem
                    )

                def shifted(off_fn, extra):
                    return lambda kp: off_fn(kp) + extra

                dA0, offA = s_stage[0].encode(major="k", mma_k=16)
                st_w, work = begin_work()
                cyc = K.local_scalar("int32", init=K.int32(0))
                with K.While(work < num_work):
                    _sq, _h, _ss, _sl, nch = work_coords(work)
                    with K.serial(nch) as n:
                        stg = cyc & 1
                        dA = dA0.value + K.Cast("uint64", stg) * K.uint64(STAGE_UNITS)
                        _t = rng("M0.wait_g1")
                        p_g1.empty.wait(st_g1.stage, st_g1.phase)
                        p_prep.full.wait(st_prep.stage, st_prep.phase)
                        st_prep.advance()
                        rng_end(_t)
                        _t = rng("M0.g1")
                        mma_ss(
                            TM_D0 + stg * 64, dA, offA, dA, shifted(offA, OFF_Y0), 8, ID_G1, False
                        )
                        pr = elect_local()
                        p_g1.full.arrive(st_g1.stage, pred=pr)  # D0 done (G1_0)
                        st_g1.advance()
                        p_stage.empty.arrive(
                            st_stg.stage, pred=pr
                        )  # G1_0 finished reading the stage's X0/Q0 rows
                        st_stg.advance()
                        rng_end(_t)
                        K.assign(cyc, cyc + K.int32(1))
                    advance_work(work, st_w)

            # =====================================================================
            # MMA issuer 1: G5 (O^T=S~Q~), V1 (S~X0), G4'' (U^T-=V1b TB^T), G1 of the next chunk, G6, G7.   warp 19
            # =====================================================================
            with mma1:
                st_stage = K.PipelineState(2, phase=0)
                st_prep = K.PipelineState(2, phase=0)  # ledger for p_prep.empty commits
                st_j1 = K.PipelineState(1, phase=0)
                st_g1b_e = K.PipelineState(1, phase=1)  # producer-side wait on p_g1b (D1 consumed)
                st_v1b = K.PipelineState(1, phase=0)  # consumer of m_v1b (V1 bf16 copy ready)
                st_t1 = K.PipelineState(
                    1, phase=0
                )  # ledger for p_t.empty commits (G4'' is TB's last reader)
                st_aqk = K.PipelineState(1, phase=0)
                st_u = K.PipelineState(1, phase=1)  # producer p_u.full
                st_o = K.PipelineState(1, phase=1)  # producer p_o
                st_ub = K.PipelineState(1, phase=0)
                st_s = K.PipelineState(1, phase=0)
                st_g3 = K.PipelineState(1, phase=0)
                tmem = tmem_preamble()

                def mma_ts(dcol, a_col, b_desc, b_off, n_k, idesc, accumulate):
                    b_base = K.local_scalar("uint64", init=b_desc)
                    fb = runtime_off(b_off, n_k) if compact else (lambda kp: K.uint64(b_off(kp)))
                    mma_issue(
                        dcol,
                        lambda kp: K.Cast("uint32", tmem[0] + a_col + kp * 8),
                        b_base,
                        fb,
                        n_k,
                        idesc,
                        accumulate,
                        tmem,
                    )

                def mma_ss(dcol, a_desc, a_off, b_desc, b_off, n_k, idesc, accumulate):
                    """SS-form chain issued by one elected lane; descriptor phases are plain 64-bit adds
                    (the 16B-unit offsets never carry out of the 14-bit start-address field)."""
                    a_base = K.local_scalar("uint64", init=a_desc)
                    b_base = K.local_scalar("uint64", init=b_desc)
                    fa = runtime_off(a_off, n_k) if compact else (lambda kp: K.uint64(a_off(kp)))
                    fb = runtime_off(b_off, n_k) if compact else (lambda kp: K.uint64(b_off(kp)))
                    mma_issue(
                        dcol, lambda kp: a_base + fa(kp), b_base, fb, n_k, idesc, accumulate, tmem
                    )

                def shifted(off_fn, extra):
                    return lambda kp: off_fn(kp) + extra

                dT1, offT1 = s_T.encode(major="k", mma_k=16)
                dJ, offJ = s_j1.encode(major="k", mma_k=16)
                dA0, offA = s_stage[0].encode(major="k", mma_k=16)
                dKG0, offKG = s_kg[0].encode(major="mn", mma_k=16)

                def issue_g1(cc):
                    """G1_1 of the chunk with running index cc: [Akk;Aqk] column block D1 (M64, J1 rows x Y1)."""
                    dA1 = dA0.value + K.Cast("uint64", cc & 1) * K.uint64(STAGE_UNITS)
                    _t = rng("M1.wait_g1")
                    p_g1b.wait(0, st_g1b_e.phase)  # cg0 consumed the previous D1
                    st_g1b_e.advance()
                    p_j1.full.wait(st_j1.stage, st_j1.phase)
                    rng_end(_t)
                    _t = rng("M1.g1")
                    mma_ss(TM_D1, dJ.value, offJ, dA1, shifted(offA, OFF_Y1), 8, ID_G1B, False)
                    pr = elect_local()
                    m_g1b.arrive(0, pred=pr)  # D1 done (G1_1)
                    p_j1.empty.arrive(st_j1.stage, pred=pr)
                    st_j1.advance()
                    rng_end(_t)

                st_w, work = begin_work()
                cyc = K.local_scalar("int32", init=K.int32(0))
                with K.While(work < num_work):
                    _sq, _h, _ss, _sl, nch = work_coords(work)
                    with K.If(nch > 0), K.Then():
                        issue_g1(cyc)
                    with K.serial(nch) as n:
                        stg = cyc & 1
                        dA = dA0.value + K.Cast("uint64", stg) * K.uint64(STAGE_UNITS)
                        dKG = dKG0.value + K.Cast("uint64", stg) * K.uint64(KG_UNITS)
                        # ---- G5: O^T = S~'^T Q0^T, then V1 = S~'^T X0^T (V1 releases the stage) -------
                        _t = rng("M1.wait_g5")
                        m_s.wait(0, st_s.phase)
                        st_s.advance()
                        p_prep.full.wait(
                            st_prep.stage, st_prep.phase
                        )  # Q0/X0 rows of this chunk published
                        p_o.empty.wait(st_o.stage, st_o.phase)
                        rng_end(_t)
                        _t = rng("M1.g5")
                        mma_ts(TM_O, TM_ST, dA, shifted(offA, OFF_Q), 8, ID_G56, False)
                        # V1's accumulator (TM_V1) was converted by cg1 before it published S~ of this chunk
                        mma_ts(TM_V1, TM_ST, dA, offA, 8, ID_G56, False)
                        pr = elect_local()
                        m_v1.arrive(0, pred=pr)  # V1 done -> cg1 converts it
                        p_stage.empty.arrive(
                            st_stage.stage, pred=pr
                        )  # stage fully consumed (G5 + V1)
                        st_stage.advance()
                        rng_end(_t)
                        # ---- G4'': U^T -= V1b TB^T (after G3 landed in U^T and cg1 published V1b) --------
                        _t = rng("M1.wait_g4")
                        m_v1b.wait(0, st_v1b.phase)
                        st_v1b.advance()
                        m_g3.wait(0, st_g3.phase)
                        st_g3.advance()
                        rng_end(_t)
                        _t = rng("M1.g4")
                        mma_ts(TM_U, TM_UB, dT1.value, offT1, 4, ID_G4B, True)
                        pr = elect_local()
                        p_u.full.arrive(st_u.stage, pred=pr)
                        st_u.advance()
                        p_t.empty.arrive(st_t1.stage, pred=pr)  # TB's last reader
                        st_t1.advance()
                        rng_end(_t)
                        # Aqk(n) was published by cg0 long ago; waiting for it before G1(n+1) keeps
                        # cg0's Aqk(n+1) arrival from ever running a phase ahead of this wait.
                        p_aqk.full.wait(st_aqk.stage, st_aqk.phase)
                        # ---- G1 of the next chunk sits between G4 and G6 in tensor-pipe order ----
                        with K.If(n + 1 < nch), K.Then():
                            issue_g1(cyc + 1)
                        # ---- G6: O^T += U^T Aqk^T ; G7: S^T += U^T KG ----------------------
                        _t = rng("M1.wait_g6")
                        m_ub.wait(0, st_ub.phase)
                        st_ub.advance()
                        rng_end(_t)
                        _t = rng("M1.g6g7")
                        mma_ts(TM_O, TM_UB, dA, shifted(offA, OFF_Y0), 4, ID_G56, True)
                        pr = elect_local()
                        p_o.full.arrive(st_o.stage, pred=pr)
                        st_o.advance()
                        st_aqk.advance()
                        mma_ts(TM_S, TM_UB, dKG, offKG, 4, ID_G7, True)
                        pr = elect_local()
                        m_sacc.arrive(0, pred=pr)
                        p_prep.empty.arrive(st_prep.stage, pred=pr)
                        st_prep.advance()
                        rng_end(_t)
                        K.assign(cyc, cyc + K.int32(1))
                    advance_work(work, st_w)

            # =====================================================================
            # TMA loader: q/k/g stages only (runs two chunks ahead).            warp 17
            # =====================================================================
            with loader:
                st_stage = K.PipelineState(2, phase=1)
                st_beta = K.PipelineState(2, phase=1)
                lane = K.lane_id()
                with K.If(elected()), K.Then():
                    for m in (q_map, k_map, g_map):
                        K.ptx.prefetch.tensormap(K.address_of(m))
                st_wp, work = begin_work(producer=True)
                with K.While(work < num_work):
                    seq_idx, h_idx, seq_start, seq_len, nch = work_coords(work)
                    seq_start32 = K.local_scalar("int32", init=K.Cast("int32", seq_start))
                    tok_last = K.local_scalar("int32", init=total_tokens - K.int32(1))
                    # This performance-only hint is omitted from static-grid analysis specializations:
                    # NumSim does not yet model it, and it has no synchronization role.
                    if static_grid is None and not (uniform and H == 96):
                        with K.If(elected()), K.Then():
                            h0_base = K.local_scalar(
                                "int64",
                                init=(
                                    K.Cast("int64", seq_idx) * K.int64(H) + K.Cast("int64", h_idx)
                                )
                                * K.int64(D_HEAD * D_HEAD),
                            )
                            prefetch_initial_state(h0_base)
                    with K.serial(nch) as n:
                        tok0 = seq_start32 + n * CHUNK
                        sv = s_stage[st_stage.stage]
                        _t = rng("L.wait_stage")
                        p_stage.empty.wait(st_stage.stage, st_stage.phase)
                        rng_end(_t)
                        _t = rng("L.issue")
                        with K.If(elected()), K.Then():
                            p_stage.full.arrive(st_stage.stage, tx_count=KQ_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(
                                p_stage.full.ptr_to([st_stage.stage])
                            )
                            for tmap, row0 in ((k_map, 0), (q_map, 64)):
                                for d0 in (0, 64):
                                    K.ptx[TMA_LD](
                                        sv.ptr_to(row0, d0),
                                        K.address_of(tmap),
                                        K.int32(d0),
                                        K.Cast("int32", tok0),
                                        K.Cast("int32", h_idx),
                                        mb,
                                    )
                            # Request L2 prefetches for the next chunk's k/q/g boxes.
                            with K.If(n + 1 < nch), K.Then():
                                for tmap in (k_map, q_map, g_map):
                                    for d0 in (0, 64):
                                        K.ptx[TMA_PREFETCH](
                                            K.address_of(tmap),
                                            K.int32(d0),
                                            K.Cast("int32", tok0 + CHUNK),
                                            K.Cast("int32", h_idx),
                                        )
                        st_stage.advance()
                        rng_end(_t)
                        if not cluster_norm:
                            # beta: sigmoid(bf16 logits) for the 64 tokens of this chunk
                            p_beta.empty.wait(st_beta.stage, st_beta.phase)
                            for i in range(2):
                                t = lane + 32 * i
                                bu = K.local_scalar("uint16")
                                tokc = tok0 + t if uniform else K.min(tok0 + t, tok_last)
                                K.ptx.ld.global_.nc.u16(
                                    bu,
                                    beta.ptr_to(
                                        [
                                            K.Cast("int64", tokc) * K.int64(H)
                                            + K.Cast("int64", h_idx)
                                        ]
                                    ),
                                )
                                bfv = K.reinterpret("float32", K.Cast("uint32", bu) << K.uint32(16))
                                sg = K.idioms.sigmoid_tanh_approx_f32(bfv)
                                K.ptx.st.shared.f32(K.address_of(s_beta[st_beta.stage, t]), sg)
                            p_beta.full.arrive(st_beta.stage)
                            st_beta.advance()
                    advance_work(work, st_wp, producer=True)
                if not fixed:
                    # Every CTA fetched its terminal id; the last CTA to get here resets both counters
                    # for the next launch.
                    with K.If(K.lane_id() == 0), K.Then():
                        K.ptx[
                            "fence.acq_rel.gpu"
                        ]()  # release: our fetches precede our done increment
                        done = K.local_scalar("uint32")
                        K.ptx["atom.acq_rel.gpu.global.add.u32"](
                            done, sched.ptr_to([1]), K.uint32(1)
                        )
                        with K.If(K.Cast("int32", done) == num_ctas - K.int32(1)), K.Then():
                            K.ptx[
                                "fence.acq_rel.gpu"
                            ]()  # acquire: every CTA's fetches/increments precede reset
                            K.ptx.st.global_.s32(sched.ptr_to([0]), K.int32(0))
                            K.ptx.st.global_.s32(sched.ptr_to([1]), K.int32(0))
                if cluster_norm:
                    with K.If(cluster_rank == K.int32(2)), K.Then():
                        helper_load(0)

            # =====================================================================
            # v loads (one chunk ahead of the O store) + TMA store of O.        warp 18
            # =====================================================================
            with storer:
                st_osm = K.PipelineState(1, phase=0)
                st_v = K.PipelineState(1, phase=1)
                lane = K.lane_id()
                with K.If(elected()), K.Then():
                    K.ptx.prefetch.tensormap(K.address_of(o_map))
                    K.ptx.prefetch.tensormap(K.address_of(v_map))

                def issue_v(tok0, h_idx):
                    _t = rng("L.wait_v")
                    p_v.empty.wait(
                        st_v.stage, st_v.phase
                    )  # single v stage: free once G3 of the previous chunk completed
                    rng_end(_t)
                    with K.If(elected()), K.Then():
                        p_v.full.arrive(st_v.stage, tx_count=V_BYTES)
                        mbv = K.cuda.cvta_generic_to_shared(p_v.full.ptr_to([st_v.stage]))
                        for d0 in (0, 64):
                            K.ptx[TMA_LD](
                                s_v.ptr_to(0, d0),
                                K.address_of(v_map),
                                K.int32(d0),
                                K.Cast("int32", tok0),
                                K.Cast("int32", h_idx),
                                mbv,
                            )
                        for d0 in (0, 64):  # the chunk after this one into L2
                            K.ptx[TMA_PREFETCH](
                                K.address_of(v_map),
                                K.int32(d0),
                                K.Cast("int32", tok0 + CHUNK),
                                K.Cast("int32", h_idx),
                            )
                    st_v.advance()

                st_w, work = begin_work()
                with K.While(work < num_work):
                    seq_idx, h_idx, seq_start, seq_len, nch = work_coords(work)
                    seq_start32 = K.local_scalar("int32", init=K.Cast("int32", seq_start))
                    with K.If(nch > 0), K.Then():
                        issue_v(seq_start32, h_idx)  # v(0)
                    with K.serial(nch) as n:
                        tok0 = seq_start32 + n * CHUNK
                        with K.If(n + 1 < nch), K.Then():
                            issue_v(tok0 + CHUNK, h_idx)  # v(n+1) ahead of the O(n) store
                        # O(n) store
                        _t = rng("S.wait")
                        p_osm.full.wait(st_osm.stage, st_osm.phase)
                        rng_end(_t)
                        _t = rng("S.store")
                        if uniform:
                            with K.If(elected()), K.Then():
                                K.ptx[FENCE_ASYNC]()
                                for d0 in (0, 64):
                                    K.ptx[TMA_ST](
                                        K.address_of(o_map),
                                        K.int32(d0),
                                        K.Cast("int32", tok0),
                                        K.Cast("int32", h_idx),
                                        s_o.ptr_to(0, d0),
                                    )
                                K.ptx[BULK_COMMIT]()
                                K.ptx[BULK_WAIT_READ](0)
                                p_osm.empty.arrive(st_osm.stage)
                        else:
                            n_valid = K.local_scalar(
                                "int32", init=K.min(seq_len - n * CHUNK, K.int32(CHUNK))
                            )
                            with K.If(n_valid == CHUNK):
                                with K.Then():
                                    with K.If(elected()), K.Then():
                                        K.ptx[FENCE_ASYNC]()
                                        for d0 in (0, 64):
                                            K.ptx[TMA_ST](
                                                K.address_of(o_map),
                                                K.int32(d0),
                                                K.Cast("int32", tok0),
                                                K.Cast("int32", h_idx),
                                                s_o.ptr_to(0, d0),
                                            )
                                        K.ptx[BULK_COMMIT]()
                                        K.ptx[BULK_WAIT_READ](0)
                                        p_osm.empty.arrive(st_osm.stage)
                                with K.Else():
                                    # Partial tail chunk: copy exactly the valid rows with generic stores.
                                    trow = K.local_scalar(
                                        "int64",
                                        init=(K.Cast("int64", tok0) + K.Cast("int64", lane >> 4))
                                        * HK64
                                        + K.Cast("int64", h_idx) * K.int64(D_HEAD)
                                        + K.Cast("int64", (lane & 15) * 8),
                                    )
                                    ow = K.alloc_local([4], "uint32")
                                    with K.serial((n_valid + 1) >> 1) as i2:
                                        r = i2 * 2 + (lane >> 4)
                                        with K.If(r < n_valid), K.Then():
                                            K.ptx["ld.shared.v4.b32"](
                                                ow[0],
                                                ow[1],
                                                ow[2],
                                                ow[3],
                                                s_o.ptr_to(r, (lane & 15) * 8),
                                            )
                                            K.ptx["st.global.v4.b32"](
                                                o.ptr_to([trow + K.Cast("int64", i2 * 2) * HK64]),
                                                ow[0],
                                                ow[1],
                                                ow[2],
                                                ow[3],
                                            )
                                    K.ptx["bar.warp.sync"](K.uint32(0xFFFFFFFF))
                                    with K.If(elected()), K.Then():
                                        p_osm.empty.arrive(st_osm.stage)
                        st_osm.advance()
                        rng_end(_t)
                    advance_work(work, st_w)
                with K.If(elected()), K.Then():
                    K.ptx[BULK_WAIT](0)
                if cluster_norm:
                    with K.If(cluster_rank == K.int32(2)), K.Then():
                        helper_load(1)

        if cluster_norm:
            # Keeps every DSMEM source/destination live until both chains and the helper
            # have completed the final publish/release handshake.
            K.ptx.barrier.cluster.arrive.release.aligned()
            K.ptx.barrier.cluster.wait.acquire()

    return kda_fwd


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------


class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_map(tensor, T_total, H, box_rows=64):
    desc = _AlignedTensorMap()
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    encode(
        desc.ptr,
        "bfloat16",
        3,
        ctypes.c_void_p(int(tensor.data_ptr())),
        D_HEAD, T_total, H,  # dims (innermost first)
        2 * D_HEAD * H, 2 * D_HEAD,  # strides in bytes (dims 1, 2)
        64, box_rows, 1,  # box
        1, 1, 1,  # element strides
        0,  # interleave none
        3,  # swizzle 128B
        2,  # L2 promotion 128B
        0,  # oob fill none
    )  # fmt: skip
    return desc


COMPACT_MIN_CTAS = 128


_FIXED_SEQ_LENS = (8192,)
_MIXED_SEQ_LENS = (1300, 547, 2048, 963, 271, 3063)
_UNIFORM_SEQ_LENS = (1024,) * 8
_SUPPORTED_SEQ_LENS = {_FIXED_SEQ_LENS, _MIXED_SEQ_LENS, _UNIFORM_SEQ_LENS}


@dataclass(frozen=True, slots=True)
class KDAForwardConfig:
    label: str
    num_heads: int
    seq_lens: tuple[int, ...]
    seed: int = 0
    scale: float = 1.0 / math.sqrt(D_HEAD)
    lower_bound: float = -5.0

    def validate(self) -> None:
        if self.num_heads not in (64, 96):
            raise ValueError(f"num_heads must be 64 or 96, got {self.num_heads}")
        if self.seq_lens not in _SUPPORTED_SEQ_LENS:
            raise ValueError(f"unsupported KDA sequence layout {self.seq_lens}")
        if sum(self.seq_lens) != 8192:
            raise ValueError(f"total tokens must be 8192, got {sum(self.seq_lens)}")
        if not math.isclose(self.scale, 1.0 / math.sqrt(D_HEAD), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"scale must be 1/sqrt({D_HEAD}), got {self.scale}")
        if self.lower_bound != -5.0:
            raise ValueError(f"lower_bound must be -5.0, got {self.lower_bound}")

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def num_seqs(self) -> int:
        return len(self.seq_lens)

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)

    @property
    def packed(self) -> bool:
        return self.num_seqs > 1

    @property
    def use_initial_state(self) -> bool:
        return True

    @property
    def store_final_state(self) -> bool:
        return False


CONFIGS = [
    {"label": f"h{heads}_{layout}", "num_heads": heads, "seq_lens": seq_lens, "seed": seed}
    for heads, layout, seq_lens, seed in (
        (96, "fixed", _FIXED_SEQ_LENS, 2858210354),
        (96, "mixed", _MIXED_SEQ_LENS, 2858210355),
        (96, "uniform", _UNIFORM_SEQ_LENS, 2858210356),
        (64, "fixed", _FIXED_SEQ_LENS, 2858210357),
        (64, "mixed", _MIXED_SEQ_LENS, 2858210358),
        (64, "uniform", _UNIFORM_SEQ_LENS, 2858210359),
    )
]

KERNEL_META = {
    "name": "agent_evolved_kda_forward_b1_t8192",
    "category": "agent_evolved",
    "runtime_cuda_archs": ["sm_100a"],
    "reference_requirements": (
        {
            "package": "flash-linear-attention",
            "git": {
                "url": "https://github.com/fla-org/flash-linear-attention.git",
                "commit": "9c8e42e762fce087c27b673af4922795d9edb85e",
            },
            "import": "fla",
        },
    ),
    "provenance": {
        "generator": "hmz",
        "run": "kda_forward-20260901-235533",
        "selected_version": "v68",
    },
}


def _cfg(**kwargs: Any) -> KDAForwardConfig:
    names = {field.name for field in fields(KDAForwardConfig)}
    values = {name: value for name, value in kwargs.items() if name in names}
    if "seq_lens" in values:
        values["seq_lens"] = tuple(int(length) for length in values["seq_lens"])
    values.setdefault("label", "custom")
    cfg = KDAForwardConfig(**values)
    cfg.validate()
    return cfg


def _kernel_traits(cfg: KDAForwardConfig) -> dict[str, Any]:
    from tirx_kernels.runner import hardware_num_sms

    fixed = not cfg.packed
    uniform = cfg.seq_lens == _UNIFORM_SEQ_LENS
    mixed = cfg.num_heads == 96 and cfg.seq_lens == _MIXED_SEQ_LENS
    cluster_norm = fixed and cfg.num_heads == 64
    if cluster_norm:
        num_ctas = 3 * (cfg.num_heads // 2)
    elif uniform and cfg.num_heads == 64:
        num_ctas = 128
    else:
        num_ctas = min(cfg.num_seqs * cfg.num_heads, hardware_num_sms())
    return {
        "static_grid": num_ctas,
        "compact": num_ctas >= COMPACT_MIN_CTAS,
        "fixed": fixed,
        "cluster_norm": cluster_norm,
        "packed_tail_hoist": False,
        "uniform": uniform,
        "mixed": mixed,
    }


def get_kernel(**kwargs: Any):
    cfg = _cfg(**kwargs)
    return make_kernel(cfg.num_heads, **_kernel_traits(cfg)).func


def _gate_parameters(
    cfg: KDAForwardConfig, *, device: torch.device, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    a_log = torch.log(
        torch.empty(cfg.num_heads, dtype=torch.float32, device=device).uniform_(
            1.0, 16.0, generator=generator
        )
    )
    dt = torch.exp(
        torch.rand(cfg.num_heads * D_HEAD, dtype=torch.float32, device=device, generator=generator)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    ).clamp_(min=1e-4)
    return a_log, dt + torch.log(-torch.expm1(-dt))


def _randn(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    device: torch.device,
    generator: torch.Generator,
    scale: float,
) -> torch.Tensor:
    return (torch.randn(shape, dtype=torch.float32, device=device, generator=generator) * scale).to(
        dtype
    )


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    device = torch.device(kwargs.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved KDA forward")

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    a_log, dt_bias = _gate_parameters(cfg, device=device, generator=generator)
    vector_shape = (1, cfg.total_tokens, cfg.num_heads, D_HEAD)
    q = _randn(vector_shape, torch.bfloat16, device=device, generator=generator, scale=0.5)
    k = _randn(vector_shape, torch.bfloat16, device=device, generator=generator, scale=0.5)
    v = _randn(vector_shape, torch.bfloat16, device=device, generator=generator, scale=0.5)
    g = _randn(vector_shape, torch.bfloat16, device=device, generator=generator, scale=0.5)
    beta = _randn(
        (1, cfg.total_tokens, cfg.num_heads),
        torch.bfloat16,
        device=device,
        generator=generator,
        scale=0.5,
    )
    initial_state = _randn(
        (cfg.num_seqs, cfg.num_heads, D_HEAD, D_HEAD),
        torch.float32,
        device=device,
        generator=generator,
        scale=0.25,
    )
    offsets = [0]
    for length in cfg.seq_lens:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device=device)
    sched = torch.zeros(2, dtype=torch.int32, device=device)
    out = torch.empty_like(v)
    tensors = {"q": q, "k": k, "v": v, "g": g, "out": out}
    tensor_maps = {
        name: _encode_map(tensor, cfg.total_tokens, cfg.num_heads)
        for name, tensor in tensors.items()
    }
    return {
        "config": cfg,
        **tensors,
        "beta": beta,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "sched": sched,
        "scale": cfg.scale,
        "tensor_maps": tensor_maps,
    }


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    cfg: KDAForwardConfig = case["config"]
    maps = case["tensor_maps"]
    num_ctas = _kernel_traits(cfg)["static_grid"]
    return (
        case["q"].view(-1),
        case["k"].view(-1),
        case["v"].view(-1),
        case["g"].view(-1),
        case["beta"].view(-1),
        case["A_log"],
        case["dt_bias"],
        case["initial_state"].view(-1),
        case["cu_seqlens"],
        case["sched"],
        case["out"].view(-1),
        maps["q"].ptr,
        maps["k"].ptr,
        maps["v"].ptr,
        maps["g"].ptr,
        maps["out"].ptr,
        cfg.scale,
        cfg.num_seqs,
        cfg.total_tokens,
        num_ctas,
    )


def _assert_sm100() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved KDA forward")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 0):
        raise SkipTest(
            f"agent-evolved KDA forward requires compute capability 10.0, got {capability}"
        )


@contextmanager
def _native_fla_backend():
    values = {"FLA_FLASH_KDA": "0", "FLA_TILELANG": "0"}
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@lru_cache(maxsize=1)
def _load_fla_chunk_kda():
    from fla.ops.kda import chunk_kda

    return chunk_kda


def _run_fla_reference(case: dict[str, Any]) -> torch.Tensor:
    cfg: KDAForwardConfig = case["config"]
    with _native_fla_backend(), torch.inference_mode():
        output, _ = _load_fla_chunk_kda()(
            q=case["q"],
            k=case["k"],
            v=case["v"],
            g=case["g"],
            beta=case["beta"],
            scale=case["scale"],
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            state_v_first=True,
            safe_gate=True,
            lower_bound=cfg.lower_bound,
            A_log=case["A_log"],
            dt_bias=case["dt_bias"],
            initial_state=case["initial_state"],
            cu_seqlens=case["cu_seqlens"] if cfg.packed else None,
        )
    return output


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
    torch.testing.assert_close(actual, reference, atol=5e-2, rtol=5e-2)
    diff_rms = torch.sqrt(torch.mean((actual.float() - reference.float()).square()))
    reference_rms = torch.sqrt(torch.mean(reference.float().square()))
    rms_ratio = float(diff_rms / (reference_rms + 1e-8))
    if rms_ratio >= 1e-2:
        raise AssertionError(f"normalized RMS error ratio {rms_ratio:.6e} must be below 1e-2")


def run_test(**kwargs: Any) -> None:
    _assert_sm100()
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    executable = compile_kernel(get_kernel(**kwargs))
    args = _tirx_args(case)
    case["out"].fill_(float("nan"))
    executable(*args)
    torch.cuda.synchronize()
    first = case["out"].clone()
    case["out"].fill_(42.0)
    executable(*args)
    torch.cuda.synchronize()
    actual = case["out"].clone()
    reference = _run_fla_reference(case)
    torch.cuda.synchronize()
    check_correctness({"first": first, "actual": actual, "reference": reference}, **kwargs)


def prepare_bench(**kwargs: Any):
    """Trace and compile before bench-suite assigns a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    _assert_sm100()
    config = dict(prepared["config"])
    config.update(kwargs)
    rounds = config.pop("rounds", 5)
    cooldown_s = config.pop("cooldown_s", 1.0)
    case = prepare_data(**config)
    executable = prepared["executable"]
    args = _tirx_args(case)
    executable(*args)
    torch.cuda.synchronize()

    def _flashkda_builder():
        from tirx_kernels.flashinfer.utils._flashkda_bench import prepare_flashkda_raw_reference

        reference_case = dict(case)
        reference_case["dt_bias"] = case["dt_bias"].view(case["config"].num_heads, D_HEAD)
        return prepare_flashkda_raw_reference(reference_case).launch

    from tirx_kernels.runner import bench

    return bench(
        {"tirx": lambda: executable(*args)},
        references={"flash_kda": _flashkda_builder},
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
    "CONFIGS",
    "KERNEL_META",
    "check_correctness",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_test",
]
