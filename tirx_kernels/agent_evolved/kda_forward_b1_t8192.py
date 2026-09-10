# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors
"""Fused KDA forward in Kern with one warp-specialized CTA per head.

Each 64-token chunk factors its log2 decay around the chunk start so the same
scaled K tile serves the round and recurrent-state MMAs; the full chunk decay
is applied once after the delta update.
"""

import ctypes
import math
import os
from dataclasses import dataclass
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K
import tvm
from tvm.backend.cuda.cpp.descriptors import encode_instr_descriptor_dense_uint32 as _idesc

BT, BC, D = 64, 16, 128
RCP_LN2 = 1.0 / math.log(2.0)
GATE_C = -2.5 * RCP_LN2  # gl = GATE_C * (tanh(x/2) + 1) = -5*RCP_LN2*sigmoid(x)
EPS = 1e-6

MMA = "tcgen05.mma.cta_group::1.kind::f16"
TMA3 = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
TMA2 = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
PREF3 = "cp.async.bulk.prefetch.tensor.3d.L2.global.tile"
COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
LD32x64 = "tcgen05.ld.sync.aligned.32x32b.x64.b32"
ST32x64 = "tcgen05.st.sync.aligned.32x32b.x64.b32"
ST32x32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"
ST32x16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
LD16x8 = "tcgen05.ld.sync.aligned.16x256b.x8.b32"
TMA_S2G3 = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
LD16x2 = "tcgen05.ld.sync.aligned.16x256b.x2.b32"
LD32x32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
TMEM_RELINQ = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
FENCE_AFTER = "tcgen05.fence::after_thread_sync"
FENCE_BEFORE = "tcgen05.fence::before_thread_sync"

# TMEM column map (512 columns); AA buffers hold the round accumulators, later u^T / v_new^T; GACC = (S k2^T)^T, GBF its bf16 copy
C_SACC, C_SBF, C_VNBF, C_OACC, C_GACC, C_AA0, C_GBF = 0, 128, 192, 224, 288, 352, 480
N_COLS = 512

# instruction descriptors (M, N, K, d, a, b, trans_a, trans_b)
F32, BF = "float32", "bfloat16"
ID_AQK = _idesc(64, 64, 16, F32, BF, BF, False, False)
ID_U = _idesc(128, 64, 16, F32, BF, BF, True, True)
ID_VN = _idesc(128, 64, 16, F32, BF, BF, False, True, neg_a=True)
ID_G64 = _idesc(128, 64, 16, F32, BF, BF, False, False)
ID_DL = _idesc(128, 128, 16, F32, BF, BF, False, True)
ID_OX = _idesc(128, 64, 16, F32, BF, BF, False, True)

W_STATE = list(range(0, 4))
W_LD, W_MMI, W_MMC, W_RND = 4, 5, 6, 7
W_INTRA = list(range(8, 12))
W_PREP = list(range(12, 20))
NWARPS = 20

NB_PREP, NB_INTRA, NB_STATE = 1, 2, 3
# mbarrier waits poll; try_wait already parks the warp until the phase flips,
# so the old nanosleep back-off only added latency after a near-miss.
WAIT_HINT = 0x989680
SLEEP_NS = 0
SLEEP_CRIT = 0


DBG_PREP = 0
DBG_INTRA = DBG_PREP + 256 * 160
DBG_AQK, DBG_LOFF, DBG_TD, DBG_TP = [DBG_INTRA + i * 4096 for i in range(4)]
DBG_G = DBG_INTRA + 4 * 4096
DBG_STATE = DBG_G + 8192
DBG_VN, DBG_S = DBG_STATE, DBG_STATE + 8192
DBG_TOTAL = DBG_S + 16384


EV = {
    n: i
    for i, n in enumerate(
        [
            "ld_issue",
            "ld_v",
            "rd_start",
            "rd_aa",
            "mmc_g",
            "mmi_u",
            "mmc_oi",
            "mmc_vnew",
            "mmc_delta",
            "mmc_ox",
            "st_delta_seen",
            "st_S_ready",
            "st_epi_start",
            "st_epi_end",
            "st_vnew_seen",
            "st_vn_ready",
            "in_aa_seen",
            "in_L_done",
            "in_diag_done",
            "in_s2_done",
            "in_s3_done",
            "in_s4_done",
            "in_aqk",
            "in_T_ready",
            "st_g_seen",
            "st_g_ready",
            "pr_ring",
            "pr_A_end",
            "pr_rfull",
            "pr_done",
            "in_diagst",
            "in_l0",
            "in_l1",
            "pr_gate",
            "pr_ef",
            "pr_norm",
            "pr_main",
            "pr_cross",
            "in_s1a",
            "in_s1b",
            "in_s1c",
            "rd_r0",
            "rd_r1",
            "st_epi_ld",
            "st_q01",
            "st_q2",
            "st_q3",
            "mmc_ofree",
            "mmc_sready",
            "cta_start",
            "cta_end",
        ]
    )
}
NEV = len(EV)


def build_kernel(
    H: int,
    debug: bool = False,
    trace: bool = False,
    NT_TR: int = 128,
    direct: bool = False,
    state_regs: int = 112,
    wg_regs: int = 56,
    intra_regs: int = 88,
    prep_regs: int = 112,
    fixed_nt: int = 0,
    full_chunks: bool = False,
    unroll_chunks: bool = False,
    fixed_tokens: int = 0,
    state_f32x2: bool = False,
    iket_trace: bool = False,
    event_mode: int = 0,
    sready_stages: int = 1,
    split_rounds: bool = False,
    probe: int = 0,
    pad: tuple = (0, 0, 0, 0),
):
    """One CTA per (sequence, head); CTA b handles head b % H of the (b // H)-th longest sequence (longest-first dispatch)."""
    DBG = debug
    TRACE = trace
    IKET = iket_trace
    MERGE_DELTA = event_mode >= 1
    MERGE_GVN = event_mode >= 2
    SPLIT_RD = bool(split_rounds)
    assert sready_stages in (1, 2, 4)
    SR_ST = int(sready_stages)
    # `probe` is a measurement-only bitmask that drops prep's input waits
    # (1 kA_free, 2 aa_r, 4 oi_done, 8 dvec_free, 16 T_ready) or one of its
    # transcendental streams (32 the 2^-cs reciprocal, 64 the gate tanh, 128 the
    # 2^cs exponential, 256 the whole |q|/|k| L2-norm reduction) or a state-chain
    # stage (512 the state decay and its fp32 write-back).  The results
    # are wrong with any bit set; it exists to
    # separate "prep is waiting" from "prep is working" and to measure which
    # pipe its span is actually bound by.
    assert 0 <= probe < 1024

    assert not fixed_tokens or direct
    assert not fixed_nt or (
        fixed_tokens > 0 and fixed_tokens % BT == 0 and fixed_nt == fixed_tokens // BT
    )
    assert not full_chunks or (fixed_tokens > 0 and fixed_tokens % BT == 0)

    def kda_fwd(
        q_map,
        k_map,
        v_map,
        g_map,
        beta_map,
        o_map,
        o,
        A_log,
        dt_bias,
        h0,
        scale,
        cu_seqlens,
        num_seqs,
        num_ctas,
        dbg,
        dbg_c,
    ):
        cta = K.cta_id()
        warp = K.warp_id()
        lane = K.lane_id()
        tid = K.thread_id()
        h = K.local_scalar("int32", init=cta % H)
        rank = K.local_scalar("int32", init=cta // H)

        def shfl_idx_i32(x, src):
            r = K.local_scalar("int32")
            K.ptx.shfl_sync.idx.b32(r, x, K.Cast("uint32", src), K.uint32(31), K.uint32(0xFFFFFFFF))
            return r

        def shfl_xor_i32(x, xr):
            r = K.local_scalar("int32")
            K.ptx.shfl_sync.bfly.b32(r, x, K.uint32(xr), K.uint32(31), K.uint32(0xFFFFFFFF))
            return r

        # ---- sequence selection: lane s < num_seqs owns sequence s; its rank counts longer (or equal, earlier) sequences.
        # Sequences beyond lane 31 fall back to a serial scan; every warp evaluates this redundantly (warp-uniform result).
        seq = K.local_scalar("int32", init=K.int32(0))
        c0 = K.local_scalar("int64", init=K.int64(0))
        c1 = K.local_scalar("int64", init=K.int64(0))
        bos = K.local_scalar("int32", init=K.int32(0))
        Lv = K.local_scalar("int32", init=K.int32(0))
        mylen = K.local_scalar("int32", init=K.int32(-1))
        if direct:
            # Fixed-length launches have one sequence, so rank is already the only
            # valid sequence.  Omitting the general ranking prologue keeps its
            # temporaries out of the entry register allocation and spill set.
            K.assign(seq, rank)
            if fixed_tokens:
                K.assign(bos, K.int32(0))
                K.assign(Lv, K.int32(fixed_tokens))
            else:
                K.ptx.ld.global_.s64(c0, cu_seqlens.ptr_to([rank]))
                K.ptx.ld.global_.s64(c1, cu_seqlens.ptr_to([rank + 1]))
                K.assign(bos, K.Cast("int32", c0))
                K.assign(Lv, K.Cast("int32", c1 - c0))
        else:
            with K.If(num_seqs <= 32):
                with K.Then():
                    lane_ok = lane < num_seqs
                    with K.If(lane_ok), K.Then():
                        K.ptx.ld.global_.s64(c0, cu_seqlens.ptr_to([lane]))
                        K.ptx.ld.global_.s64(c1, cu_seqlens.ptr_to([lane + 1]))
                    K.assign(mylen, K.Select(lane_ok, K.Cast("int32", c1 - c0), K.int32(-1)))
                    r = K.local_scalar("int32", init=K.int32(0))
                    with K.serial(num_seqs, unroll=False) as j:
                        lj = shfl_idx_i32(mylen, j)
                        longer = tvm.tirx.any(lj > mylen, tvm.tirx.all(lj == mylen, j < lane))
                        K.assign(
                            r, r + K.Select(tvm.tirx.all(lane_ok, longer), K.int32(1), K.int32(0))
                        )
                    sel = K.local_scalar(
                        "int32", init=K.Select(tvm.tirx.all(lane_ok, r == rank), lane, K.int32(0))
                    )
                    for xr in (16, 8, 4, 2, 1):
                        K.assign(sel, K.max(sel, shfl_xor_i32(sel, xr)))
                    K.assign(seq, sel)
                    K.assign(bos, shfl_idx_i32(K.Cast("int32", c0), sel))
                    K.assign(Lv, shfl_idx_i32(mylen, sel))
                with K.Else():
                    # Longest-first is a dispatch heuristic, not a correctness requirement, so past 32
                    # sequences the CTA simply takes its own rank -- the O(n^2) ordering scan it replaces
                    # cost registers in the entry region, which is the tightest register scope.
                    K.assign(seq, rank)
                    K.ptx.ld.global_.s64(c0, cu_seqlens.ptr_to([rank]))
                    K.ptx.ld.global_.s64(c1, cu_seqlens.ptr_to([rank + 1]))
                    K.assign(bos, K.Cast("int32", c0))
                    K.assign(Lv, K.Cast("int32", c1 - c0))
        L = Lv
        NT = int(fixed_nt) if fixed_nt else K.local_scalar("int32", init=(Lv + (BT - 1)) // BT)

        def tmark(name, c, guard=True):
            """Trace milestone (head, event, chunk) -> clock64; guard=False inside single-lane code."""
            if IKET:
                K.cuda.iket.mark(name)
            if not TRACE:
                return
            idx = (cta * NEV + EV[name]) * NT_TR + c
            if guard:
                with K.If(lane == 0), K.Then():
                    K.ptx.st.global_.u64(dbg.ptr_to([idx]), K.cuda.clock64())
            else:
                K.ptx.st.global_.u64(dbg.ptr_to([idx]), K.cuda.clock64())

        with K.If(warp == 0), K.Then():
            tmark("cta_start", 0)
        smem = K.smem_pool()
        tmem_addr = smem.alloc((1,), K.u32)
        # (h, seq, bos, L, NT) published once; every role reloads them into its own registers.
        # Keeping them live across the role bodies spilled them to local memory, and the reloads
        # ran with one active lane per warp (32B sector, 4B used) inside the single-lane MMA loops.
        ctx_s = None if fixed_tokens else smem.alloc((8,), K.u32)
        rsq = smem.alloc((2 * 4 * 64,), K.f32, align=16)
        beta_s = smem.alloc((2 * 64 * 8,), K.bf16, align=128)
        bsig = smem.alloc((2 * 64,), K.f32, align=16)
        dvec = smem.alloc((3 * 128,), K.f32, align=16)
        tot = smem.alloc((2 * 4 * 128,), K.f32, align=16)
        adt_s = smem.alloc((128 + 4,), K.f32, align=16)

        def mbar(count, depth=1):
            b = K.MBarrier(smem, depth)
            b.init(count)
            return b

        ring_full = mbar(1, 2)
        stage_free = mbar(8, 2)
        v_full = mbar(1)
        rfull = mbar(8, 2)
        aa_r = mbar(1, 2)
        akk_r = mbar(1, 2) if SPLIT_RD else None
        aa_free = mbar(4, 2)
        T_ready = mbar(4, 2)
        dvec_ready = mbar(8, 3)
        dvec_free = mbar(4, 3)
        g_done = mbar(1)
        g_ready = mbar(4)
        g_free = None if MERGE_GVN else mbar(4)
        u_done = mbar(1, 2)
        Aqk_ready = mbar(4)
        # One arrival per SBF quarter-group: the next chunk's S-operand MMAs consume the
        # decayed state incrementally instead of waiting for the whole 128x128 update.
        S_ready = mbar(4, SR_ST)
        vnew_done = mbar(1, 2)
        vn_ready = None if MERGE_GVN else mbar(4)
        delta_done = None if MERGE_DELTA else mbar(1)
        kA_free = mbar(1, 2)
        oi_done = mbar(1)
        o_done = mbar(1)
        o_free = mbar(4)

        pool = smem.pool

        TOFF = {}

        def tile_alloc(name, shape):
            view = smem.alloc(shape, K.bf16, swizzle=K.SW128B)
            nbytes = 2
            for d_ in shape:
                nbytes *= d_
            TOFF[name] = pool.offset - nbytes
            return view

        pool.move_base_to((pool.offset + 1023) // 1024 * 1024)
        # q/k/g stages hold only the raw tiles; kA = A rows of the rounds and B rows of the delta MMA (k * 2^-cs); Bt / qg_s = q and k rows * 2^cs (tokens 0-31 / 32-63)
        q_t = tile_alloc("q", (2, BT, D))
        k_t = tile_alloc("k", (2, BT, D))
        g_t = tile_alloc("g", (2, BT, D))
        STAGE_UNITS = BT * D * 2 // 16
        v_t = tile_alloc("v", (BT, D))
        pool.move_base_to((pool.offset + 1023) // 1024 * 1024)
        kA = tile_alloc("kA", (2, BT, D))
        q2t = tile_alloc("q2", (BT, D))
        k2t = tile_alloc("k2", (BT, D))
        Aqk_s = tile_alloc("Aqk", (64, 64))
        LTT = tile_alloc("LTT", (64, 64))
        TpT_t = tile_alloc("TpT", (64, 64))
        o_st_off = pool.offset
        o_st = tile_alloc(
            "ost", (2, BT, 64)
        )  # output staging tile [v-half][token][64 v] for the TMA store
        end_off = pool.offset
        pool.move_base_to(o_st_off)
        h0_s = smem.alloc((64 * 64,), K.f32, align=16)  # aliases o_st: prologue-only staging
        pool.move_base_to(end_off)
        base0 = TOFF["q"]
        for name in TOFF:
            TOFF[name] -= base0
        assert min(TOFF.values()) == 0

        if not fixed_tokens:
            with K.If(tid == 0), K.Then():
                for i_, v_ in enumerate((h, seq, bos, L, NT)):
                    K.ptx.st.shared.u32(ctx_s.ptr_to([i_]), K.Cast("uint32", v_))
            K.ptx.fence.proxy.async_.shared__cta()
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        with K.If(warp == 0), K.Then():
            K.ptx[TMEM_ALLOC](K.address_of(tmem_addr[0]), K.uint32(N_COLS))
            K.cuda.warp_sync()
        K.cuda.cta_sync()
        tbase = K.local_scalar("uint32")
        K.ptx.ld.shared.u32(tbase, tmem_addr.ptr_to([0]))

        def ctx(*names):
            """Role-local copies of the published CTA scalars, in the order (h, seq, bos, L, NT)."""
            if fixed_tokens:
                out = []
                for n in names:
                    if n == "h":
                        value = cta % H
                    elif n == "seq":
                        value = rank
                    elif n == "bos":
                        value = K.int32(0)
                    elif n == "L":
                        value = K.int32(fixed_tokens)
                    elif n == "NT":
                        value = (fixed_tokens + BT - 1) // BT
                    else:
                        raise ValueError(n)
                    out.append(K.local_scalar("int32", init=value))
                return out[0] if len(out) == 1 else out
            out = []
            for n in names:
                u = K.local_scalar("uint32")
                K.ptx.ld.shared.u32(u, ctx_s.ptr_to([("h", "seq", "bos", "L", "NT").index(n)]))
                out.append(K.local_scalar("int32", init=K.Cast("int32", u)))
            return out[0] if len(out) == 1 else out

        def role_nt():
            """Chunk count, frozen only for an implicit single sequence."""
            return int(fixed_nt) if fixed_nt else ctx("NT")

        def tmem(col, lane_off=0):
            if isinstance(col, int):
                return K.cuda.get_tmem_addr(tbase, lane_off, col)
            return tbase + K.uint32(lane_off << 16) + K.Cast("uint32", col)

        def aa_col(b2, extra=0):
            """column of AA buffer b2 (runtime) plus a static offset"""
            return C_AA0 + extra + 64 * b2

        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def warrive(b, idx, pred=None):
            """One arrival per warp (after a warp sync) instead of one per thread."""
            K.cuda.warp_sync()
            with K.If(lane == 0), K.Then():
                if pred is None:
                    b.arrive(idx)
                else:
                    b.arrive(idx, pred=pred)

        def fwait(b, stage, parity, ns=SLEEP_NS):
            """mbarrier phase wait; non-critical waits (ns > 0) back off with nanosleep after a failed poll so
            waiting warps do not steal issue slots; chain-critical waits (ns == 0) poll without sleeping."""
            if IKET:
                wait_token = K.alloc_local((1,), "uint32")
                K.assign(wait_token[0], K.cuda.iket.range_start("mbarrier-wait"))
            # One try_wait site per wait, not two.  The peeled first poll cost a
            # second PHASECHK/NANOSLEEP pair at every one of the ~54 wait sites,
            # and this kernel is instruction-fetch bound at high CTA counts.
            ready = K.local_scalar("uint32", init=K.uint32(0))
            with K.While(ready == K.uint32(0)):
                if ns:
                    K.cuda.nano_sleep(K.uint64(ns))
                K.ptx.mbarrier.try_wait.parity.shared.b64(
                    ready, b.ptr_to([stage]), K.Cast("uint32", parity), K.uint32(WAIT_HINT)
                )
            if IKET:
                K.cuda.iket.range_end(wait_token[0])

        ZERO4 = (K.uint32(0),) * 4

        def mma(d, aop, bop, idesc, acc, dol=ZERO4):
            K.ptx[MMA](K.Cast("uint32", d), aop, bop, K.uint32(idesc), *dol, K.ptx.pred(acc))

        def commit(ptr):
            K.ptx[COMMIT](ptr)

        PARENT_ROWS = {
            "q": BT,
            "k": BT,
            "g": BT,
            "v": BT,
            "kA": BT,
            "q2": BT,
            "k2": BT,
            "Aqk": 64,
            "LTT": 64,
            "TpT": 64,
        }
        LBO_UNITS = sorted({r * 8 for r in PARENT_ROWS.values()} | {0})

        def make_templates():
            """tcgen05 smem descriptor template per LBO (16B units): SBO = 64 (8-row group), 128B swizzle."""
            t = {}
            for ldo in LBO_UNITS:
                d = K.SmemDescriptor()
                d.init(q_t[0].ptr_to(0, 0), ldo=ldo, sdo=64, swizzle=3)
                t[ldo] = d.desc
            return t

        def tile_desc(tmpl, name, rows, cols, major, kp=0, row_off=0, stage_units=None, khalf=0):
            """Operand descriptor in tile `name` (TOFF = stage-0 byte offset from the ring); LBO = parent K-half stride, zeroed for one-atom-wide K-major operands."""
            lbo = PARENT_ROWS[name] * 8
            ldo = 0 if (major == "k" and cols <= 64) else lbo
            if major == "k":
                step = (kp % 4) * 2 + (kp // 4) * lbo
            else:
                step = kp * 128
            off = (TOFF[name] >> 4) + step + row_off * 8 + khalf * lbo
            d = tmpl[ldo] + K.uint64(off) if off else tmpl[ldo]
            if stage_units is not None:
                d = d + K.Cast("uint64", stage_units)
            return d

        def bf16x2(lo, hi):
            r = K.local_scalar("uint32")
            K.ptx.cvt.rn.bf16x2.f32(r, hi, lo)
            return r

        def unpack(u):
            lo = K.reinterpret("float32", K.shift_left(u, K.uint32(16)))
            hi = K.reinterpret("float32", K.bitwise_and(u, K.uint32(0xFFFF0000)))
            return lo, hi

        def hmul2(a, b):
            r = K.local_scalar("uint32")
            K.ptx.mul.rn.bf16x2(r, a, b)
            return r

        def hfma2(a, b, c):
            r = K.local_scalar("uint32")
            K.ptx.fma.rn.bf16x2(r, a, b, c)
            return r

        def ex2(x):
            r = K.local_scalar("float32")
            K.ptx.ex2.approx.ftz.f32(r, x)
            return r

        def is_dbg(c):
            return tvm.tirx.all(h == 0, c == dbg_c) if DBG else None

        PAD = (*tuple(int(x) for x in pad), 0, 0, 0, 0)

        def padblock(role_idx, outside=False):
            """Measurement-only: `pad[role_idx]` never-executed FFMA in this role's chunk loop.

            dbg_c is -1 in every production launch, so the guard is always false and the
            block only costs instruction-cache footprint -- which is what it measures."""
            n = PAD[role_idx + (4 if outside else 0)] if len(PAD) > 4 or not outside else 0
            if not n:
                return
            with K.If(dbg_c == K.int32(0x5F5F5F)), K.Then():
                acc = K.local_scalar("float32", init=scale)
                for _ in range(n):
                    K.assign(acc, acc * K.float32(1.0000001) + K.float32(0.5))
                K.ptx.st.global_.f32(dbg.ptr_to([0]), acc)

        def baddr(ptr, off):
            return K.ptx.addr(ptr, K.int32(off) if isinstance(off, int) else off)

        sp = K.specialize()
        r_state = sp.role("state", warps=W_STATE, regs=state_regs)
        wg1 = sp.warpgroup("wg1", warps=range(4, 8), regs=wg_regs)
        r_ld = sp.role("load", warps=[W_LD], group=wg1)
        r_mmi = sp.role("mma_i", warps=[W_MMI], group=wg1)
        r_mmc = sp.role("mma_c", warps=[W_MMC], group=wg1)
        r_rnd = sp.role("rounds", warps=[W_RND], group=wg1)
        r_intra = sp.role("intra", warps=W_INTRA, regs=intra_regs)
        r_prep = sp.role("prep", warps=W_PREP, regs=prep_regs)

        def load_body():
            lane_i = K.lane_id()
            h, bos = ctx("h", "bos")
            NT = role_nt()

            def issue_loads(cc):
                sc = cc % 2
                with K.If(elected()), K.Then():
                    mb = K.cuda.cvta_generic_to_shared(ring_full.ptr_to([sc]))
                    for tile, tmap in ((q_t, q_map), (k_t, k_map), (g_t, g_map)):
                        K.ptx[TMA3](
                            tile[sc].ptr_to(0, 0),
                            K.address_of(tmap),
                            K.int32(0),
                            K.Cast("int32", bos + cc * BT),
                            K.Cast("int32", 2 * h),
                            mb,
                        )
                    K.ptx[TMA2](
                        beta_s.ptr_to([sc * 512]),
                        K.address_of(beta_map),
                        K.Cast("int32", (h // 8) * 8),
                        K.Cast("int32", bos + cc * BT),
                        mb,
                    )
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        ring_full.ptr_to([sc]), K.uint32(3 * BT * D * 2 + 1024)
                    )
                    # No L2 tensor prefetch: the next chunk's TMA is issued one
                    # stage ahead already, so the prefetch only added twelve
                    # UTMAPF issues and ~60 instructions of code to a kernel that
                    # is instruction-fetch bound at high CTA counts.
                tmark("ld_issue", cc)

            issue_loads(K.int32(0))
            with K.If(NT > 1), K.Then():
                issue_loads(K.int32(1))
            with K.serial(NT, unroll=unroll_chunks) as c:
                padblock(1)
                # stage (c+1)%2 is free once prep has consumed the raw tiles of chunk c-1; v_t once u^T(c-1) is done
                with K.If(tvm.tirx.all(c >= 1, c + 1 < NT)), K.Then():
                    fwait(stage_free, (c + 1) % 2, ((c - 1) // 2) % 2)
                    # stage_free retires prep's generic reads of the raw ring and
                    # beta tiles.  Bridge that into the async proxy before the TMA
                    # refills the same stage: the mbarrier release/acquire pair
                    # orders the accesses within the generic proxy only.
                    K.ptx.fence.proxy.async_.shared__cta()
                    issue_loads(c + 1)
                with K.If(c >= 1), K.Then():
                    fwait(u_done, (c + 1) % 2, ((c - 1) // 2) % 2)
                with K.If(elected()), K.Then():
                    mbv = K.cuda.cvta_generic_to_shared(v_full.ptr_to([0]))
                    K.ptx[TMA3](
                        v_t.ptr_to(0, 0),
                        K.address_of(v_map),
                        K.int32(0),
                        K.Cast("int32", bos + c * BT),
                        K.Cast("int32", 2 * h),
                        mbv,
                    )
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        v_full.ptr_to([0]), K.uint32(BT * D * 2)
                    )
                tmark("ld_v", c)

        def rounds_body():
            """Round R (32-token block of i): D[j, 0:32] = Aqk^T, D[j, 32:64] = Akk^T; A = kA (k rows * 2^-cs), B = Bt (round 0) / qg_s (round 1)."""
            NT = role_nt()
            with K.If(elected()), K.Then():
                tmpl = make_templates()
                with K.serial(NT, unroll=unroll_chunks) as c:
                    padblock(1)
                    b2 = c % 2
                    su = b2 * STAGE_UNITS

                    def td(name, rows, cols, major, kp=0, row_off=0, staged=False):
                        return tile_desc(
                            tmpl, name, rows, cols, major, kp, row_off, su if staged else None, 0
                        )

                    with K.If(c >= 2), K.Then():
                        fwait(aa_free, b2, (c // 2 + 1) % 2)
                    tmark("rd_start", c, False)
                    fwait(rfull, b2, (c // 2) % 2)
                    tmark("rd_r0", c, False)
                    # With split_rounds, Akk^T is issued first and gets its own
                    # completion: the intra transform's first dependency is the
                    # diagonal block of Akk, so signalling it separately starts the
                    # 16x16 inverse a full round of MMAs earlier.  aa_r still marks
                    # all sixteen done, which is what prep needs before it may
                    # overwrite q2/k2.  The extra commit costs a round trip in the
                    # rounds warp, which is why the dispatch picks per shape.
                    for R in (1, 0) if SPLIT_RD else (0, 1):
                        b_name = "q2" if R == 0 else "k2"
                        for kp in range(8):
                            mma(
                                tmem(aa_col(b2), 16 * R),
                                td("kA", 64, 128, "k", kp, 0, True),
                                td(b_name, 64, 128, "k", kp),
                                ID_AQK,
                                kp != 0,
                            )
                        if SPLIT_RD:
                            commit((akk_r if R == 1 else aa_r).ptr_to([b2]))
                    if not SPLIT_RD:
                        commit(aa_r.ptr_to([b2]))
                    tmark("rd_aa", c, False)

        def mmi_body():
            """The u^T MMA moved into mmc_body (same-warp ordering removes a barrier hop); this role idles."""
            return

        def mmc_body():
            """Recurrence MMAs: G^T = S^T k2^T and O_ACC = S^T q2^T from the k / q rows of Bt and qg_s (two N=32 halves); v_new^T = u^T - G_bf^T T'^T (on u^T in AA[b2]); S^T += v_new^T kA (decay applied afterwards by the state warps); O_ACC += v_new^T Aqk_s^T."""
            NT = role_nt()
            with K.If(elected()), K.Then():
                tmpl = make_templates()
                with K.serial(NT, unroll=unroll_chunks) as c:
                    padblock(1)
                    b2 = c % 2
                    su = b2 * STAGE_UNITS

                    def td(name, rows, cols, major, kp=0, row_off=0, staged=False):
                        return tile_desc(
                            tmpl, name, rows, cols, major, kp, row_off, su if staged else None, 0
                        )

                    fwait(rfull, b2, (c // 2) % 2)
                    if not MERGE_GVN:
                        with K.If(c >= 1), K.Then():
                            fwait(g_free, 0, (c + 1) % 2)
                    fwait(S_ready, 0, c % 2, SLEEP_CRIT)
                    tmark("mmc_sready", c, False)
                    tmark("mmc_g", c, False)
                    kp_stage = 8 // SR_ST
                    for st_ in range(SR_ST):
                        if st_:
                            fwait(S_ready, st_, c % 2, SLEEP_CRIT)
                        for kp in range(st_ * kp_stage, (st_ + 1) * kp_stage):
                            mma(
                                tmem(C_GACC, 0),
                                K.Cast("uint32", tmem(C_SBF + 8 * kp, 0)),
                                td("k2", 64, 128, "k", kp),
                                ID_G64,
                                kp != 0,
                            )
                    commit(g_done.ptr_to([0]))
                    with K.If(c >= 1), K.Then():
                        fwait(o_free, 0, (c + 1) % 2)
                    tmark("mmc_ofree", c, False)
                    tmark("mmc_oi", c, False)
                    for kp in range(8):
                        mma(
                            tmem(C_OACC, 0),
                            K.Cast("uint32", tmem(C_SBF + 8 * kp, 0)),
                            td("q2", 64, 128, "k", kp),
                            ID_G64,
                            kp != 0,
                        )
                    commit(oi_done.ptr_to([0]))
                    fwait(T_ready, b2, (c // 2) % 2, SLEEP_CRIT)
                    fwait(v_full, 0, c % 2, SLEEP_CRIT)
                    tmark("mmi_u", c, False)
                    for kp in range(4):
                        mma(
                            tmem(aa_col(b2), 0),
                            td("v", 64, 128, "mn", kp),
                            td("TpT", 64, 64, "mn", kp),
                            ID_U,
                            kp != 0,
                        )
                    commit(u_done.ptr_to([b2]))
                    fwait(g_ready, 0, c % 2, SLEEP_CRIT)
                    tmark("mmc_vnew", c, False)
                    for kp in range(4):
                        mma(
                            tmem(aa_col(b2), 0),
                            K.Cast("uint32", tmem(C_GBF + 8 * kp, 0)),
                            td("TpT", 64, 64, "mn", kp),
                            ID_VN,
                            True,
                        )
                    commit(vnew_done.ptr_to([b2]))
                    if MERGE_GVN:
                        fwait(aa_free, b2, (c // 2) % 2, SLEEP_CRIT)
                    else:
                        fwait(vn_ready, 0, c % 2, SLEEP_CRIT)
                    tmark("mmc_delta", c, False)
                    for kp in range(4):
                        mma(
                            tmem(C_SACC, 0),
                            K.Cast("uint32", tmem(C_VNBF + 8 * kp, 0)),
                            td("kA", 64, 128, "mn", kp, 0, True),
                            ID_DL,
                            True,
                        )
                    if MERGE_DELTA:
                        # The same completion releases kA to prep(c+2) and
                        # the updated accumulator to state(c); mbarrier waits
                        # are non-consuming, so both share one tcgen commit.
                        commit(kA_free.ptr_to([b2]))
                    else:
                        commit(delta_done.ptr_to([0]))
                        commit(kA_free.ptr_to([b2]))
                    fwait(Aqk_ready, 0, c % 2, SLEEP_CRIT)
                    tmark("mmc_ox", c, False)
                    for kp in range(4):
                        mma(
                            tmem(C_OACC, 0),
                            K.Cast("uint32", tmem(C_VNBF + 8 * kp, 0)),
                            td("Aqk", 64, 64, "mn", kp),
                            ID_OX,
                            True,
                        )
                    commit(o_done.ptr_to([0]))

        def state_body():
            h, seq, bos, L = ctx("h", "seq", "bos", "L")
            NT = role_nt()
            v_idx = tid
            regs = K.alloc_local((64,), "float32")
            packed = K.alloc_local((16,), "uint32")

            def st_packed(col, base):
                """regs[base : base + 32] -> bf16 -> TMEM columns col .. col + 15"""
                for j in range(16):
                    K.assign(packed[j], bf16x2(regs[base + 2 * j], regs[base + 2 * j + 1]))
                K.ptx[ST32x16](tmem(col), *[packed[j] for j in range(16)])

            # ---- initial state (V-first layout, as FLA's state_v_first): S^T[v][k] = h0[h][v][k].
            # One 512 B row per thread makes the direct read 32 sectors per instruction, and every
            # CTA pays it inside the pipeline fill.  Stage it through shared memory instead: the
            # global side is 256 B contiguous per 16 lanes, and the shared side is XOR-swizzled on
            # 16 B units so neither the fill nor the drain has bank conflicts.  The staging buffer
            # aliases o_st, which the same warps first touch in the epilogue.
            hbase = K.local_scalar("int32", init=(seq * H + h) * D * D)
            for half in range(2):
                for blk in range(2):
                    t4 = K.alloc_local((4,), "float32")
                    for i in range(8):
                        rr = i * 8 + tid // 16
                        cc = (tid % 16) * 4
                        K.ptx.ld.global_.v4.f32(
                            t4[0],
                            t4[1],
                            t4[2],
                            t4[3],
                            h0.ptr_to([hbase + (64 * blk + rr) * D + 64 * half + cc]),
                        )
                        K.ptx.st.shared.v4.f32(
                            h0_s.ptr_to([rr * 64 + K.bitwise_xor(tid % 16, rr % 16) * 4]),
                            t4[0],
                            t4[1],
                            t4[2],
                            t4[3],
                        )
                    K.ptx.bar.sync(K.uint32(NB_STATE), K.uint32(128))
                    with K.If(v_idx // 64 == blk), K.Then():
                        rl = v_idx % 64
                        for u in range(16):
                            K.ptx.ld.shared.v4.f32(
                                regs[4 * u],
                                regs[4 * u + 1],
                                regs[4 * u + 2],
                                regs[4 * u + 3],
                                h0_s.ptr_to([rl * 64 + K.bitwise_xor(K.int32(u), rl % 16) * 4]),
                            )
                    K.ptx.bar.sync(K.uint32(NB_STATE), K.uint32(128))
                K.ptx[ST32x64](tmem(C_SACC + 64 * half), *[regs[j] for j in range(64)])
                st_packed(C_SBF + 32 * half, 0)
                st_packed(C_SBF + 32 * half + 16, 32)
            K.ptx.tcgen05.wait__st.sync.aligned()
            K.ptx[FENCE_BEFORE]()
            for _st in range(SR_ST):
                warrive(S_ready, _st)

            def epilogue(c1, tail=not full_chunks):
                """o(c1) = O_ACC (bf16). Full chunks: TMEM fragments -> stmatrix.trans into the staging tile -> one TMA store
                (scalar global stores contend with the G MMA's operand traffic); partial tail chunks use predicated scalar stores.

                `tail=False` emits only the full-tile path.  Only a sequence's last chunk
                can be partial, so the varlen build peels that chunk out of the loop and
                leaves the predicated scalar path as cold code, where the instruction
                cache does not pay for it."""
                whole = full_chunks or not tail
                fwait(o_done, 0, c1 % 2, SLEEP_CRIT)
                with K.If(warp == 0), K.Then():
                    tmark("st_epi_start", c1)
                    with K.If(lane == 0), K.Then():
                        K.ptx.cp.async_.bulk.wait_group.read(0)
                K.ptx.bar.sync(K.uint32(NB_STATE), K.uint32(128))
                K.ptx[FENCE_AFTER]()
                nvalid = BT if whole else K.local_scalar("int32", init=L - c1 * BT)
                c4 = lane % 4
                r16 = lane // 4
                # Both OACC halves are pulled before any packing so o_free -- which gates the next
                # chunk's o_inter MMA, and through it prep's phase B -- is released as early as possible.
                for half in range(2):
                    K.ptx[LD16x8](
                        *[regs[32 * half + j] for j in range(32)], tmem(C_OACC, 16 * half)
                    )
                K.ptx.tcgen05.wait__ld.sync.aligned()
                with K.If(warp == 0), K.Then():
                    tmark("st_epi_ld", c1)
                K.ptx[FENCE_BEFORE]()
                warrive(o_free, 0)
                for half in range(2):
                    b = 32 * half
                    if whole:
                        for u in range(8):
                            for hh in range(2):
                                K.assign(
                                    packed[2 * u + hh],
                                    bf16x2(regs[b + 4 * u + 2 * hh], regs[b + 4 * u + 2 * hh + 1]),
                                )
                        for hh in range(2):
                            for g_ in range(2):
                                K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                                    o_st[warp // 2].ptr_to(
                                        32 * g_ + lane, 32 * (warp % 2) + 16 * half + 8 * hh
                                    ),
                                    packed[2 * (4 * g_) + hh],
                                    packed[2 * (4 * g_ + 1) + hh],
                                    packed[2 * (4 * g_ + 2) + hh],
                                    packed[2 * (4 * g_ + 3) + hh],
                                )
                    else:
                        with K.If(nvalid >= BT):
                            with K.Then():
                                for u in range(8):
                                    for hh in range(2):
                                        K.assign(
                                            packed[2 * u + hh],
                                            bf16x2(
                                                regs[b + 4 * u + 2 * hh],
                                                regs[b + 4 * u + 2 * hh + 1],
                                            ),
                                        )
                                for hh in range(2):
                                    for g_ in range(2):
                                        K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                                            o_st[warp // 2].ptr_to(
                                                32 * g_ + lane, 32 * (warp % 2) + 16 * half + 8 * hh
                                            ),
                                            packed[2 * (4 * g_) + hh],
                                            packed[2 * (4 * g_ + 1) + hh],
                                            packed[2 * (4 * g_ + 2) + hh],
                                            packed[2 * (4 * g_ + 3) + hh],
                                        )
                            with K.Else():
                                obase = o.ptr_to(
                                    [
                                        ((bos + c1 * BT + 2 * c4) * H + h) * D
                                        + 32 * warp
                                        + 16 * half
                                        + r16
                                    ]
                                )
                                for u in range(8):
                                    for hh in range(2):
                                        for cc in range(2):
                                            hb = K.local_scalar("uint16")
                                            K.ptx.cvt.rn.bf16.f32(hb, regs[b + 4 * u + 2 * hh + cc])
                                            K.ptx.st.global_.b16(
                                                baddr(obase, (8 * u + cc) * H * D * 2 + 16 * hh),
                                                hb,
                                                pred=(8 * u + cc + 2 * c4 < nvalid),
                                            )
                if whole:
                    K.ptx.fence.proxy.async_.shared__cta()
                    K.ptx.bar.sync(K.uint32(NB_STATE), K.uint32(128))
                    with K.If(warp == 0), K.Then():
                        with K.If(lane == 0), K.Then():
                            K.ptx[TMA_S2G3](
                                K.address_of(o_map),
                                K.int32(0),
                                K.Cast("int32", bos + c1 * BT),
                                K.Cast("int32", 2 * h),
                                o_st[0].ptr_to(0, 0),
                            )
                            K.ptx.cp.async_.bulk.commit_group()
                else:
                    with K.If(nvalid >= BT), K.Then():
                        K.ptx.fence.proxy.async_.shared__cta()
                        K.ptx.bar.sync(K.uint32(NB_STATE), K.uint32(128))
                        with K.If(warp == 0), K.Then():
                            with K.If(lane == 0), K.Then():
                                K.ptx[TMA_S2G3](
                                    K.address_of(o_map),
                                    K.int32(0),
                                    K.Cast("int32", bos + c1 * BT),
                                    K.Cast("int32", 2 * h),
                                    o_st[0].ptr_to(0, 0),
                                )
                                K.ptx.cp.async_.bulk.commit_group()
                with K.If(warp == 0), K.Then():
                    tmark("st_epi_end", c1)

            def gconv(c):
                """G^T = (S k2^T)^T (fp32, GACC) -> bf16 GBF, the A operand of v_new^T = u^T - G_bf^T T'^T."""
                fwait(g_done, 0, c % 2, SLEEP_CRIT)
                with K.If(warp == 0), K.Then():
                    tmark("st_g_seen", c)
                K.ptx[FENCE_AFTER]()
                K.ptx[LD32x64](*[regs[j] for j in range(64)], tmem(C_GACC))
                K.ptx.tcgen05.wait__ld.sync.aligned()
                if DBG:
                    with K.If(is_dbg(c)), K.Then():
                        for j in range(64):
                            K.ptx.st.global_.f32(dbg.ptr_to([DBG_G + v_idx * 64 + j]), regs[j])
                st_packed(C_GBF, 0)
                st_packed(C_GBF + 16, 32)
                K.ptx.tcgen05.wait__st.sync.aligned()
                K.ptx[FENCE_BEFORE]()
                warrive(g_ready, 0)
                if not MERGE_GVN:
                    warrive(g_free, 0)
                with K.If(warp == 0), K.Then():
                    tmark("st_g_ready", c)

            padblock(0, outside=True)
            with K.serial(NT, unroll=unroll_chunks) as c:
                padblock(0)
                s = c % 2
                gconv(c)
                fwait(vnew_done, s, (c // 2) % 2, SLEEP_CRIT)
                with K.If(warp == 0), K.Then():
                    tmark("st_vnew_seen", c)
                K.ptx[FENCE_AFTER]()
                K.ptx[LD32x64](*[regs[j] for j in range(64)], tmem(aa_col(s)))
                K.ptx.tcgen05.wait__ld.sync.aligned()
                if DBG:
                    with K.If(is_dbg(c)), K.Then():
                        for j in range(64):
                            K.ptx.st.global_.f32(dbg.ptr_to([DBG_VN + v_idx * 64 + j]), regs[j])
                st_packed(C_VNBF, 0)
                st_packed(C_VNBF + 16, 32)
                K.ptx.tcgen05.wait__st.sync.aligned()
                K.ptx[FENCE_BEFORE]()
                if not MERGE_GVN:
                    warrive(vn_ready, 0)
                warrive(aa_free, s)
                with K.If(warp == 0), K.Then():
                    tmark("st_vn_ready", c)
                # ---- S_{c+1} = (S_c + v_new^T kA) * 2^gc_last: fp32 in SACC, bf16 copy in SBF (A operand of G / o_inter of chunk c+1)
                if MERGE_DELTA:
                    fwait(kA_free, s, (c // 2) % 2, SLEEP_CRIT)
                else:
                    fwait(delta_done, 0, c % 2, SLEEP_CRIT)
                with K.If(warp == 0), K.Then():
                    tmark("st_delta_seen", c)
                K.ptx[FENCE_AFTER]()
                fwait(dvec_ready, c % 3, (c // 3) % 2)

                def ld_q(qi):
                    """quarter qi (32 fp32 columns) of S^T -> regs[32*(qi%2) : +32]."""
                    b = 32 * (qi % 2)
                    K.ptx[LD32x32](*[regs[b + j] for j in range(32)], tmem(C_SACC + 32 * qi))

                def proc_q(qi):
                    """decay by dvec(c), bf16 copy -> SBF, fp32 write back."""
                    b = 32 * (qi % 2)
                    dv4 = K.alloc_local((4,), "float32")
                    for j in range(0, 32, 4):
                        if probe & 512:
                            continue
                        K.ptx.ld.shared.v4.f32(
                            dv4[0],
                            dv4[1],
                            dv4[2],
                            dv4[3],
                            dvec.ptr_to([(c % 3) * 128 + 32 * qi + j]),
                        )
                        if state_f32x2:
                            for m in range(0, 4, 2):
                                pair = K.local_scalar("uint64")
                                K.ptx.mov.b64(pair, regs[b + j + m], regs[b + j + m + 1])
                                K.ptx.mul.rn.ftz.f32x2(
                                    pair, pair, K.cuda.make_float2(dv4[m], dv4[m + 1])
                                )
                                K.ptx.mov.b64(regs[b + j + m], regs[b + j + m + 1], pair)
                        else:
                            for m in range(4):
                                K.assign(regs[b + j + m], regs[b + j + m] * dv4[m])
                    for j in range(16):
                        K.assign(packed[j], bf16x2(regs[b + 2 * j], regs[b + 2 * j + 1]))
                    K.ptx[ST32x16](tmem(C_SBF + 16 * qi), *[packed[j] for j in range(16)])
                    if DBG:
                        with K.If(is_dbg(c)), K.Then():
                            for j in range(32):
                                K.ptx.st.global_.f32(
                                    dbg.ptr_to([DBG_S + v_idx * 128 + 32 * qi + j]), regs[b + j]
                                )
                    if not (probe & 512):
                        K.ptx[ST32x32](tmem(C_SACC + 32 * qi), *[regs[b + j] for j in range(32)])

                per_stage = 4 // SR_ST

                def sr_signal(qi):
                    """Publish SBF through quarter qi once it closes a stage group."""
                    if (qi + 1) % per_stage:
                        return
                    K.ptx.tcgen05.wait__st.sync.aligned()
                    K.ptx[FENCE_BEFORE]()
                    warrive(S_ready, (qi + 1) // per_stage - 1)

                ld_q(0)
                ld_q(1)
                K.ptx.tcgen05.wait__ld.sync.aligned()
                with K.If(warp == 0), K.Then():
                    tmark("st_q01", c)
                proc_q(0)
                ld_q(2)
                sr_signal(0)
                K.ptx.tcgen05.wait__ld.sync.aligned()
                with K.If(warp == 0), K.Then():
                    tmark("st_q2", c)
                proc_q(1)
                ld_q(3)
                sr_signal(1)
                K.ptx.tcgen05.wait__ld.sync.aligned()
                with K.If(warp == 0), K.Then():
                    tmark("st_q3", c)
                proc_q(2)
                sr_signal(2)
                proc_q(3)
                warrive(dvec_free, c % 3)
                sr_signal(3)
                with K.If(warp == 0), K.Then():
                    tmark("st_S_ready", c)
                if full_chunks:
                    epilogue(c)
                else:
                    with K.If(c + 1 < NT), K.Then():
                        epilogue(c, tail=False)
            if not full_chunks:
                # The peeled last chunk: nothing waits on its o_free, so running its
                # epilogue after the loop only moves the partial-tile path out of the
                # chunk loop's instruction footprint.
                epilogue(K.local_scalar("int32", init=NT - 1))
            with K.If(warp == 0), K.Then():
                with K.If(lane == 0), K.Then():
                    K.ptx.cp.async_.bulk.wait_group(0)

        def intra_body():
            """Intra-chunk transform: L = strict-lower(Akk) beta -> T = (I+L)^-1 -> T' = T diag(beta), Aqk_s; blocks live in mma fragments (movmatrix turns a C fragment into a B operand), only T'^T / T^T diagonal / Aqk^T reach smem."""
            NT = role_nt()
            q = warp - W_INTRA[0]
            r16 = lane // 4
            c4 = lane % 4
            aqk = K.alloc_local((32,), "float32")
            acc = K.alloc_local((8,), "float32")
            a_frag = K.alloc_local((4,), "uint32")
            b_frag = K.alloc_local((4,), "uint32")
            aM = K.alloc_local((4,), "uint32")
            bM = K.alloc_local((4,), "uint32")
            aP = K.alloc_local((4,), "uint32")
            bP4 = K.alloc_local((4,), "uint32")
            bP8 = K.alloc_local((4,), "uint32")
            tA = [K.alloc_local((4,), "uint32") for _ in range(3)]
            bL = [K.alloc_local((4,), "uint32") for _ in range(3)]

            def round_coords(L, reg):
                """Round-L accumulator register `reg` (lane offset 16L) -> (j, i, is_aqk, block of i): round 0 = Aqk^T, round 1 = Akk^T, 64 columns = i."""
                rep, rem = divmod(reg, 4)
                hh, cc = divmod(rem, 2)
                i = 8 * rep + 2 * c4 + cc
                j = q * 16 + r16 + 8 * hh
                return j, i, L == 0, rep // 2

            def isync():
                K.ptx.bar.sync(K.uint32(NB_INTRA), K.uint32(128))

            def imark(name, c):
                with K.If(warp == W_INTRA[0]), K.Then():
                    tmark(name, c)

            def frag_addr(tile, rb, cb):
                return tile.ptr_to(rb + lane % 16, cb + (lane // 16) * 8)

            def ld_b(B, br, bc, dst):
                K.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                    dst[0], dst[1], dst[2], dst[3], frag_addr(B, br, bc)
                )

            def st_frag(tile, rb, cb, src):
                K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                    frag_addr(tile, rb, cb), src[0], src[1], src[2], src[3]
                )

            def movm(dst, src):
                for z in range(4):
                    K.ptx.movmatrix.sync.aligned.m8n8.trans.b16(dst[z], src[z])

            def mma_frag(a, b, clear):
                if clear:
                    for z in range(8):
                        K.assign(acc[z], K.float32(0.0))
                for nh in range(2):
                    z = 4 * nh
                    K.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
                        acc[z],
                        acc[z + 1],
                        acc[z + 2],
                        acc[z + 3],
                        a[0],
                        a[1],
                        a[2],
                        a[3],
                        b[2 * nh],
                        b[2 * nh + 1],
                        acc[z],
                        acc[z + 1],
                        acc[z + 2],
                        acc[z + 3],
                    )

            def pack_acc(dst, neg=False, rs=None):
                """C fragment -> bf16 A fragment; rs=(b_row, b_row8) scales rows r16 / r16+8, neg flips the sign bits."""
                for z in range(4):
                    v0, v1 = acc[2 * z], acc[2 * z + 1]
                    if rs is not None:
                        v0, v1 = v0 * rs[z % 2], v1 * rs[z % 2]
                    p = bf16x2(v0, v1)
                    K.assign(dst[z], K.bitwise_xor(p, K.uint32(0x80008000)) if neg else p)

            def add_identity():
                """acc += I in the C-fragment layout: rows (r16, r16+8), cols (8nh + 2c4, +1)."""
                for nh in range(2):
                    z = 4 * nh
                    K.assign(
                        acc[z],
                        acc[z] + K.Select(r16 == 8 * nh + 2 * c4, K.float32(1.0), K.float32(0.0)),
                    )
                    K.assign(
                        acc[z + 1],
                        acc[z + 1]
                        + K.Select(r16 == 8 * nh + 2 * c4 + 1, K.float32(1.0), K.float32(0.0)),
                    )
                    K.assign(
                        acc[z + 2],
                        acc[z + 2]
                        + K.Select(r16 + 8 == 8 * nh + 2 * c4, K.float32(1.0), K.float32(0.0)),
                    )
                    K.assign(
                        acc[z + 3],
                        acc[z + 3]
                        + K.Select(r16 + 8 == 8 * nh + 2 * c4 + 1, K.float32(1.0), K.float32(0.0)),
                    )

            def beta_rows(s, J):
                """-beta of rows 16J + r16 and 16J + r16 + 8 (T'^T = T^T beta_j, sign folded so acc = -T^T packs to +T'^T)."""
                b0 = K.local_scalar("float32")
                b1 = K.local_scalar("float32")
                K.ptx.ld.shared.f32(b0, bsig.ptr_to([s * 64 + 16 * J + r16]))
                K.ptx.ld.shared.f32(b1, bsig.ptr_to([s * 64 + 16 * J + r16 + 8]))
                return (K.float32(0.0) - b0, K.float32(0.0) - b1)

            def dbg_T(c, J, I, rs):
                """DBG: acc holds -T_IJ^T block (J, I); dump T^T and T'^T entries."""
                if DBG:
                    with K.If(is_dbg(c)), K.Then():
                        for nh in range(2):
                            z = 4 * nh
                            for m in range(4):
                                ii = 16 * I + 8 * nh + 2 * c4 + (m % 2)
                                jj = 16 * J + r16 + 8 * (m // 2)
                                vv = K.float32(0.0) - acc[z + m]
                                K.ptx.st.global_.f32(dbg.ptr_to([DBG_TD + ii * 64 + jj]), vv)
                                K.ptx.st.global_.f32(
                                    dbg.ptr_to([DBG_TP + ii * 64 + jj]),
                                    K.float32(0.0) - vv * rs[m // 2],
                                )

            # LTT holds the off-diagonal L^T blocks (row block < column block) and the diagonal T^T blocks; TpT_t = T'^T (MN-major B operand of u / v_new), its lower blocks stay zero
            LT = LTT
            TT = LTT
            TpT = TpT_t
            with K.If(q == 3), K.Then():
                z = K.uint32(0)
                for m in range(6):
                    n = m * 32 + lane
                    rr, hf = (n % 32) // 2, n % 2
                    UJ = [1, 2, 3, 2, 3, 3][m]
                    UI = [0, 0, 0, 1, 1, 2][m]
                    K.ptx.st.shared.v4.u32(TpT.ptr_to(16 * UJ + rr, 16 * UI + 8 * hf), z, z, z, z)
            K.ptx.fence.proxy.async_.shared__cta()
            with K.serial(NT, unroll=unroll_chunks) as c:
                padblock(2)
                s = c % 2

                def dbg_sq(off, i, j, val):
                    if DBG:
                        with K.If(is_dbg(c)), K.Then():
                            K.ptx.st.global_.f32(dbg.ptr_to([off + i * 64 + j]), val)

                # ---- S0a: diagonal block M = L_qq^T from round q//2's Akk^T columns, straight into an A fragment (rows j = 16q + r16 + 8hh, cols i = 16q + 8rep + 2c4 + cc)
                fwait(akk_r if SPLIT_RD else aa_r, s, (c // 2) % 2)
                imark("in_aa_seen", c)
                K.ptx[FENCE_AFTER]()
                bcol = K.alloc_local((16,), "float32")
                for I in range(4):
                    for r2 in range(2):
                        K.ptx.ld.shared.v2.f32(
                            bcol[I * 4 + r2 * 2],
                            bcol[I * 4 + r2 * 2 + 1],
                            bsig.ptr_to([s * 64 + 16 * I + 8 * r2 + 2 * c4]),
                        )
                dblk = K.alloc_local((8,), "float32")
                K.ptx[LD16x2](*[dblk[j] for j in range(8)], tmem(aa_col(s, 16 * q), 16))
                K.ptx.tcgen05.wait__ld.sync.aligned()
                # beta of the two diagonal-block columns this lane owns; reg 0/2 and 4/6
                # share them, so load each pair once as one v2 instead of four scalars.
                bdiag = K.alloc_local((4,), "float32")
                for rep_ in range(2):
                    K.ptx.ld.shared.v2.f32(
                        bdiag[2 * rep_],
                        bdiag[2 * rep_ + 1],
                        bsig.ptr_to([s * 64 + 16 * q + 8 * rep_ + 2 * c4]),
                    )
                for reg in range(0, 8, 2):
                    rep_, rem = divmod(reg, 4)
                    hh = rem // 2
                    j = q * 16 + r16 + 8 * hh
                    i = 16 * q + 8 * rep_ + 2 * c4
                    lv = []
                    for cc in range(2):
                        lv.append(
                            K.local_scalar(
                                "float32",
                                init=K.Select(
                                    i + cc > j,
                                    dblk[reg + cc] * bdiag[2 * rep_ + cc],
                                    K.float32(0.0),
                                ),
                            )
                        )
                        dbg_sq(DBG_LOFF, i + cc, j, lv[cc])
                    K.assign(aM[reg // 2], bf16x2(lv[0], lv[1]))
                movm(bM, aM)
                imark("in_L_done", c)
                # ---- S1: T_qq^T = (I+M)^-1 = (I-M)(I+M^2)(I+M^4)(I+M^8), M nilpotent; every operand stays in fragments
                mma_frag(aM, bM, True)
                pack_acc(aP)
                movm(b_frag, aP)
                mma_frag(aP, b_frag, True)
                pack_acc(a_frag)
                movm(bP4, a_frag)
                mma_frag(a_frag, bP4, True)
                pack_acc(a_frag)
                movm(bP8, a_frag)
                imark("in_s1a", c)
                for z in range(4):
                    lo, hi = unpack(aP[z])
                    K.assign(acc[2 * z], lo)
                    K.assign(acc[2 * z + 1], hi)
                add_identity()
                pack_acc(a_frag)
                mma_frag(a_frag, bP4, False)
                pack_acc(a_frag)
                mma_frag(a_frag, bP8, False)
                imark("in_s1b", c)
                pack_acc(a_frag)
                movm(b_frag, a_frag)
                for z in range(8):
                    K.assign(acc[z], K.float32(0.0) - acc[z])
                mma_frag(aM, b_frag, False)
                imark("in_s1c", c)
                # T_qq^T: own A fragment + TT block (B operand of the other warps' chains); T'^T diagonal block -> TpT
                pack_acc(tA[0], neg=True)
                with K.If(c >= 1), K.Then():
                    fwait(vnew_done, (c + 1) % 2, ((c - 1) // 2) % 2)
                    # vnew_done retires the prior async reads of the shared
                    # TpT tile.  Re-enter the generic proxy before stmatrix
                    # overwrites that single-buffered tile for this chunk.
                    K.ptx.fence.proxy.async_.shared__cta()
                st_frag(TT, 16 * q, 16 * q, tA[0])
                rs = beta_rows(s, q)
                pack_acc(a_frag, rs=rs)
                st_frag(TpT, 16 * q, 16 * q, a_frag)
                imark("in_diagst", c)
                dbg_T(c, q, q, rs)
                # ---- S0b: off-diagonal L^T blocks (rows of block q, columns of blocks I > q) and the Aqk capture
                aqk_pk = K.alloc_local((16,), "uint32")
                npk = 0
                # Under split_rounds the L = 1 pass (Akk^T -> off-diagonal L blocks)
                # is already complete, so the full-round wait sits between the two
                # passes, just before the only one that reads Aqk^T.
                for L in (1, 0) if SPLIT_RD else (0, 1):
                    if SPLIT_RD and L == 0:
                        fwait(aa_r, s, (c // 2) % 2)
                        K.ptx[FENCE_AFTER]()
                    K.ptx[LD16x8](*[aqk[j] for j in range(32)], tmem(aa_col(s), 16 * L))
                    K.ptx.tcgen05.wait__ld.sync.aligned()
                    for reg in range(0, 32, 2):
                        j, i, is_aqk, I = round_coords(L, reg)
                        rep_ = reg // 4
                        if is_aqk:
                            av = [
                                K.local_scalar(
                                    "float32",
                                    init=K.Select(i + cc >= j, aqk[reg + cc], K.float32(0.0)),
                                )
                                for cc in range(2)
                            ]
                            K.assign(aqk_pk[npk], bf16x2(av[0], av[1]))
                            npk += 1
                            continue
                        lv = []
                        for cc in range(2):
                            bidx = I * 4 + (rep_ % 2) * 2 + cc
                            # The store below is predicated on q < I, and every column of
                            # block I then lies strictly right of every row of block q, so
                            # the strict-lower select this used to carry was always true on
                            # the rows that reach memory.  Dropping it removes an ISETP and
                            # an FSEL per element from intra's chunk loop.
                            lv.append(K.local_scalar("float32", init=aqk[reg + cc] * bcol[bidx]))
                            dbg_sq(DBG_LOFF, i + cc, j, lv[cc])
                        K.ptx.st.shared.b32(LT.ptr_to(j, i), bf16x2(lv[0], lv[1]), pred=(q < I))
                    imark(("in_l0", "in_l1")[L], c)
                isync()
                imark("in_diag_done", c)

                def chain(I, J, terms):
                    """T_IJ^T = -(sum_K T_KJ^T L_IK^T) T_II^T with T_KJ^T from tA[K-J]; result kept in tA[I-J], T'^T block (J, I) -> TpT."""
                    for n, Kk in enumerate(terms):
                        ld_b(LT, 16 * Kk, 16 * I, bL[n])
                    ld_b(TT, 16 * I, 16 * I, b_frag)
                    for n, Kk in enumerate(terms):
                        mma_frag(tA[Kk - J], bL[n], n == 0)
                    pack_acc(a_frag)
                    mma_frag(a_frag, b_frag, True)
                    if I - J < 3:
                        pack_acc(tA[I - J], neg=True)
                    rs = beta_rows(s, J)
                    pack_acc(a_frag, rs=rs)
                    st_frag(TpT, 16 * J, 16 * I, a_frag)
                    dbg_T(c, J, I, rs)

                with K.If(q == 0), K.Then():
                    chain(1, 0, [0])
                    imark("in_s2_done", c)
                    chain(2, 0, [0, 1])
                    imark("in_s3_done", c)
                    chain(3, 0, [0, 1, 2])
                with K.If(q == 1), K.Then():
                    chain(2, 1, [1])
                    chain(3, 1, [1, 2])
                with K.If(q == 2), K.Then():
                    chain(3, 2, [2])
                imark("in_s4_done", c)
                K.ptx.fence.proxy.async_.shared__cta()
                K.ptx[FENCE_BEFORE]()
                warrive(T_ready, s)
                imark("in_T_ready", c)
                # ---- Aqk^T (unscaled, i >= j) -> AqkT[j][i] (MN-major B operand of o_intra), from the captured pairs
                with K.If(c >= 1), K.Then():
                    fwait(o_done, 0, (c + 1) % 2)
                    # o_done retires mmc's prior async read of Aqk_s.
                    K.ptx.fence.proxy.async_.shared__cta()
                npk = 0
                for L in range(2):
                    for reg in range(0, 32, 2):
                        j, i, is_aqk, I = round_coords(L, reg)
                        if not is_aqk:
                            continue
                        K.ptx.st.shared.b32(Aqk_s.ptr_to(j, i), aqk_pk[npk])
                        npk += 1
                K.ptx.fence.proxy.async_.shared__cta()
                warrive(Aqk_ready, 0)
                imark("in_aqk", c)

        def prep_body():
            """Gate/cumsum/normalize/scaled-copy producer: thread = 4 tokens x 8 channels (lane = cg8 + 8*j4, one 128B row per 8-lane phase); warp pair (2I, 2I+1) = 16-token block I, one channel half each; every scaling references the chunk start (E2 = 2^cs, F2 = 2^-cs)."""
            h, L = ctx("h", "L")
            NT = role_nt()
            w = warp - W_PREP[0]
            I = w // 2
            half = w % 2
            cg8 = lane % 8
            j4 = lane // 8
            d0 = (half * 8 + cg8) * 8
            tl0 = I * 16 + 4 * j4
            # E2/F2 = 2^(+-cs) per [token][pair] (F2 later holds the A rows), csv = inclusive local cumsum, sq = |q|^2 / |k|^2 partials, qraw/kraw = raw bf16 pairs, offp = gate prefix at the block start (producer lane)
            E2 = K.alloc_local((16,), "uint32")
            F2 = K.alloc_local((16,), "uint32")
            csv = K.alloc_local((32,), "float32")
            sq = K.alloc_local((8,), "float32")
            qraw = K.alloc_local((16,), "uint32")
            kraw = K.alloc_local((16,), "uint32")
            offv = K.alloc_local((8,), "float32")
            offp = K.alloc_local((2,), "float32")

            def shfl_xor(x, xr):
                peer = K.local_scalar("uint32")
                K.ptx.shfl_sync.bfly.b32(
                    peer,
                    K.reinterpret("uint32", x),
                    K.uint32(xr),
                    K.uint32(31),
                    K.uint32(0xFFFFFFFF),
                )
                return K.reinterpret("float32", peer)

            def shfl_up(x, delta):
                peer = K.local_scalar("uint32")
                K.ptx.shfl_sync.up.b32(
                    peer,
                    K.reinterpret("uint32", x),
                    K.uint32(delta),
                    K.uint32(0),
                    K.uint32(0xFFFFFFFF),
                )
                return K.reinterpret("float32", peer)

            def gather_off():
                """offv[2p + m] = gate prefix at the block start of channel 2*(4*cg8 + p) + m (fp32, from producer lane 4*cg8 + p)."""
                for p in range(4):
                    for m in range(2):
                        v = K.local_scalar("uint32")
                        K.ptx.shfl_sync.idx.b32(
                            v,
                            K.reinterpret("uint32", offp[m]),
                            K.Cast("uint32", 4 * cg8 + p),
                            K.uint32(31),
                            K.uint32(0xFFFFFFFF),
                        )
                        K.assign(offv[2 * p + m], K.reinterpret("float32", v))

            def ld4u(dst, off, ptr):
                K.ptx.ld.shared.v4.b32(dst[off], dst[off + 1], dst[off + 2], dst[off + 3], ptr)

            def st4u(ptr, src, off):
                K.ptx.st.shared.v4.b32(ptr, src[off], src[off + 1], src[off + 2], src[off + 3])

            def rcp(x):
                """1/x as one MUFU op; the magic-seed + Newton form it replaces cost five ALU slots for worse accuracy."""
                r = K.local_scalar("float32")
                K.ptx.rcp.approx.ftz.f32(r, x)
                return r

            def pmark(name, c):
                with K.If(w == 0), K.Then():
                    tmark(name, c)

            def dbg_tile(c, nm, t, vals):
                """DBG: vals[4] packed pairs of tile nm row t, channels d0..d0+7."""
                if DBG:
                    with K.If(is_dbg(c)), K.Then():
                        for p in range(4):
                            lo, hi = unpack(vals[p])
                            K.ptx.st.global_.f32(
                                dbg.ptr_to([DBG_PREP + (nm * 64 + t) * 128 + d0 + 2 * p]), lo
                            )
                            K.ptx.st.global_.f32(
                                dbg.ptr_to([DBG_PREP + (nm * 64 + t) * 128 + d0 + 2 * p + 1]), hi
                            )

            # one-time gate parameter table: adt_s[d] = exp(A_log) * dt_bias[d] / 2, adt_s[128] = exp(A_log) / 2
            with K.If(I == 0), K.Then():
                a_h = K.local_scalar("float32")
                K.ptx.ld.global_.f32(a_h, A_log.ptr_to([h]))
                a_e2 = K.local_scalar(
                    "float32", init=ex2(a_h * K.float32(RCP_LN2)) * K.float32(0.5)
                )
                with K.If(j4 == 0), K.Then():
                    dtb = K.alloc_local((8,), "float32")
                    K.ptx.ld.global_.v4.f32(
                        dtb[0], dtb[1], dtb[2], dtb[3], dt_bias.ptr_to([h * D + d0])
                    )
                    K.ptx.ld.global_.v4.f32(
                        dtb[4], dtb[5], dtb[6], dtb[7], dt_bias.ptr_to([h * D + d0 + 4])
                    )
                    K.ptx.st.shared.v4.f32(
                        adt_s.ptr_to([d0]),
                        a_e2 * dtb[0],
                        a_e2 * dtb[1],
                        a_e2 * dtb[2],
                        a_e2 * dtb[3],
                    )
                    K.ptx.st.shared.v4.f32(
                        adt_s.ptr_to([d0 + 4]),
                        a_e2 * dtb[4],
                        a_e2 * dtb[5],
                        a_e2 * dtb[6],
                        a_e2 * dtb[7],
                    )
                with K.If(tvm.tirx.all(half == 0, lane == 0)), K.Then():
                    K.ptx.st.shared.f32(adt_s.ptr_to([128]), a_e2)
            K.ptx.bar.sync(K.uint32(NB_PREP), K.uint32(256))
            adt2 = K.alloc_local((8,), "float32")
            K.ptx.ld.shared.v4.f32(adt2[0], adt2[1], adt2[2], adt2[3], adt_s.ptr_to([d0]))
            K.ptx.ld.shared.v4.f32(adt2[4], adt2[5], adt2[6], adt2[7], adt_s.ptr_to([d0 + 4]))
            a_e2 = K.local_scalar("float32")
            K.ptx.ld.shared.f32(a_e2, adt_s.ptr_to([128]))
            sc2 = K.local_scalar("float32", init=scale)

            with K.serial(NT, unroll=unroll_chunks) as c:
                padblock(3)
                s = c % 2
                fwait(ring_full, s, (c // 2) % 2)
                # The raw TMA ring is free after phase A, but bsig shares this
                # two-stage lifetime with the intra transform.  Do not lap the
                # transform and overwrite bsig(c-2) while it is still live.
                with K.If(c >= 2), K.Then():
                    if not (probe & 16):
                        fwait(T_ready, s, (c // 2 + 1) % 2)
                pmark("pr_ring", c)
                nval = K.local_scalar("int32", init=L - c * BT)
                qs, ks, gs_ = q_t[s], k_t[s], g_t[s]

                # ---- phase A: raw loads (q/k stay in registers for phase B), gate, local cumsum, sums of squares
                def phase_a(masked):
                    # Issue the independent shared-memory reads as one window so their latency overlaps
                    # the gate dependency chains below.  q/k remain live for phase B; g is consumed here.
                    g16 = K.alloc_local((16,), "uint32")
                    for i in range(4):
                        ld4u(qraw, 4 * i, qs.ptr_to(tl0 + i, d0))
                        ld4u(kraw, 4 * i, ks.ptr_to(tl0 + i, d0))
                        ld4u(g16, 4 * i, gs_.ptr_to(tl0 + i, d0))
                    for i in range(4):
                        t = tl0 + i
                        g8 = [g16[4 * i + z] for z in range(4)]
                        # |q|^2 / |k|^2 partials accumulate as packed bf16 pairs: one FMA slot per
                        # channel pair instead of two unpacks plus two FFMA.
                        sqa = K.local_scalar("uint32", init=K.uint32(0))
                        ska = K.local_scalar("uint32", init=K.uint32(0))
                        for p in range(4):
                            g0, g1 = unpack(g8[p])
                            for ch, gv in enumerate((g0, g1)):
                                m = 2 * p + ch
                                x = K.local_scalar("float32", init=gv * a_e2 + adt2[m])
                                th = K.local_scalar("float32")
                                if probe & 64:
                                    K.assign(th, x)
                                else:
                                    K.ptx.tanh.approx.f32(th, x)
                                gl = th * K.float32(GATE_C) + K.float32(GATE_C)
                                if i == 0:
                                    K.assign(csv[m], gl)
                                else:
                                    K.assign(csv[i * 8 + m], csv[(i - 1) * 8 + m] + gl)
                            if not (probe & 256):
                                K.assign(sqa, hfma2(qraw[4 * i + p], qraw[4 * i + p], sqa))
                                K.assign(ska, hfma2(kraw[4 * i + p], kraw[4 * i + p], ska))
                        if probe & 256:
                            K.assign(sq[i], K.float32(1.0))
                            K.assign(sq[4 + i], K.float32(1.0))
                        else:
                            q0, q1 = unpack(sqa)
                            k0, k1 = unpack(ska)
                            K.assign(sq[i], q0 + q1)
                            K.assign(sq[4 + i], k0 + k1)

                phase_a(not full_chunks)
                pmark("pr_gate", c)
                xin = [K.local_scalar("float32", init=csv[24 + m]) for m in range(8)]
                for step in (1, 2):
                    for m in range(8):
                        y = shfl_up(xin[m], 8 * step)
                        K.assign(xin[m], K.Select(j4 >= step, xin[m] + y, xin[m]))
                with K.If(j4 == 3), K.Then():
                    K.ptx.st.shared.v4.f32(
                        tot.ptr_to([s * 512 + I * 128 + d0]), xin[0], xin[1], xin[2], xin[3]
                    )
                    K.ptx.st.shared.v4.f32(
                        tot.ptr_to([s * 512 + I * 128 + d0 + 4]), xin[4], xin[5], xin[6], xin[7]
                    )
                excl = [K.local_scalar("float32", init=xin[m] - csv[24 + m]) for m in range(8)]
                # |q|^2, |k|^2 partial sums over the 8 lanes sharing j4 (transpose-reduce: lane keeps index cg8)
                cur = [sq[m] for m in range(8)]
                for xr in () if probe & 256 else (4, 2, 1):
                    nb = len(cur) // 2
                    hi = K.Cast("bool", K.bitwise_and(lane, K.int32(xr)))
                    nxt = []
                    for m in range(nb):
                        send = K.local_scalar("float32", init=K.Select(hi, cur[m], cur[nb + m]))
                        keep = K.Select(hi, cur[nb + m], cur[m])
                        nxt.append(K.local_scalar("float32", init=keep + shfl_xor(send, xr)))
                    cur = nxt
                K.ptx.st.shared.f32(
                    rsq.ptr_to([s * 256 + (cg8 // 4) * 128 + half * 64 + tl0 + cg8 % 4]), cur[0]
                )
                K.ptx.bar.sync(K.uint32(NB_PREP), K.uint32(256))
                # beta is only read by the intra transform, which cannot start before
                # rfull, so computing it here instead of ahead of phase A keeps warp 0 --
                # the only prep warp with extra work -- off the phase-A barrier that
                # every other prep warp waits on.  stage_free (which releases beta_s to
                # the next TMA) therefore moves below it; the refill it gates has two
                # chunk periods of slack.
                with K.If(w == 0), K.Then():
                    for hf in range(2):
                        rawb = K.local_scalar("uint16")
                        K.ptx.ld.shared.b16(
                            rawb, beta_s.ptr_to([s * 512 + (lane + 32 * hf) * 8 + h % 8])
                        )
                        bv = K.reinterpret(
                            "float32", K.shift_left(K.Cast("uint32", rawb), K.uint32(16))
                        )
                        sg = K.idioms.sigmoid_tanh_approx_f32(bv)
                        K.ptx.st.shared.f32(
                            bsig.ptr_to([s * 64 + lane + 32 * hf]),
                            K.Select(lane + 32 * hf < nval, sg, K.float32(0.0)),
                        )
                warrive(stage_free, s)
                pmark("pr_A_end", c)
                # ---- per-channel constants: lane l of warp (I, half) produces channel pair cp = 32*half + l; pref[J] = gate prefix at the start of block J, pref[4] = chunk total (dvec)
                cp = half * 32 + lane
                tt = K.alloc_local((8,), "float32")
                for J in range(4):
                    K.ptx.ld.shared.v2.f32(
                        tt[2 * J], tt[2 * J + 1], tot.ptr_to([s * 512 + J * 128 + 2 * cp])
                    )
                pref = [[K.float32(0.0)] + [None] * 4 for _ in range(2)]
                for m in range(2):
                    for J in range(4):
                        pref[m][J + 1] = K.local_scalar("float32", init=pref[m][J] + tt[2 * J + m])
                for m in range(2):
                    r = pref[m][3]
                    for J in (2, 1, 0):
                        r = K.Select(I == J, pref[m][J], r)
                    K.assign(offp[m], r)
                # dvec has 3 stages; stage c % 3 is free once the state consumed dvec(c-3)
                with K.If(c >= 3), K.Then():
                    if not (probe & 8):
                        fwait(dvec_free, c % 3, ((c - 3) // 3) % 2)
                with K.If(I == 0), K.Then():
                    K.ptx.st.shared.v2.f32(
                        dvec.ptr_to([(c % 3) * 128 + 2 * cp]), ex2(pref[0][4]), ex2(pref[1][4])
                    )
                warrive(dvec_ready, c % 3)
                gather_off()
                for m in range(8):
                    K.assign(excl[m], excl[m] + offv[m])
                pmark("pr_cross", c)
                for i in range(4):
                    for p in range(4):
                        if probe & 128:
                            e0 = K.local_scalar("float32", init=csv[i * 8 + 2 * p] + excl[2 * p])
                            e1 = K.local_scalar(
                                "float32", init=csv[i * 8 + 2 * p + 1] + excl[2 * p + 1]
                            )
                        else:
                            e0 = ex2(csv[i * 8 + 2 * p] + excl[2 * p])
                            e1 = ex2(csv[i * 8 + 2 * p + 1] + excl[2 * p + 1])
                        K.assign(E2[4 * i + p], bf16x2(e0, e1))
                        if probe & 32:
                            K.assign(F2[4 * i + p], bf16x2(e0, e1))
                        else:
                            K.assign(F2[4 * i + p], bf16x2(rcp(e0), rcp(e1)))
                pmark("pr_norm", c)
                # ---- phase B
                with K.If(c >= 2), K.Then():
                    if not (probe & 1):
                        fwait(kA_free, s, (c // 2 + 1) % 2)
                with K.If(c >= 1), K.Then():
                    if not (probe & 2):
                        fwait(aa_r, (c + 1) % 2, ((c - 1) // 2) % 2)
                    if not (probe & 4):
                        fwait(oi_done, 0, (c + 1) % 2)
                    # aa_r/oi_done (and kA_free above) complete asynchronous
                    # tcgen reads.  Bridge that proxy before these generic
                    # shared stores reuse q2/k2/kA.
                    K.ptx.fence.proxy.async_.shared__cta()
                pmark("pr_ef", c)
                sm = K.alloc_local((16,), "float32")
                if not (probe & 256):
                    for which in range(2):
                        for hf in range(2):
                            K.ptx.ld.shared.v4.f32(
                                sm[which * 8 + hf * 4],
                                sm[which * 8 + hf * 4 + 1],
                                sm[which * 8 + hf * 4 + 2],
                                sm[which * 8 + hf * 4 + 3],
                                rsq.ptr_to([s * 256 + which * 128 + hf * 64 + tl0]),
                            )
                rq2 = K.alloc_local((4,), "uint32")
                rk2 = K.alloc_local((4,), "uint32")
                for i in range(4):
                    rqv = K.local_scalar("float32")
                    rkv = K.local_scalar("float32")
                    if probe & 256:
                        K.assign(rqv, sc2)
                        K.assign(rkv, sc2)
                    else:
                        K.ptx.rsqrt.approx.ftz.f32(rqv, sm[i] + sm[4 + i] + K.float32(EPS))
                        K.ptx.rsqrt.approx.ftz.f32(rkv, sm[8 + i] + sm[12 + i] + K.float32(EPS))
                    rqs = rqv * sc2
                    K.assign(rq2[i], bf16x2(rqs, rqs))
                    K.assign(rk2[i], bf16x2(rkv, rkv))
                pmark("pr_main", c)
                for i in range(4):
                    t = tl0 + i
                    il = 4 * j4 + i
                    qgI = K.alloc_local((4,), "uint32")
                    knI = K.alloc_local((4,), "uint32")
                    # Tail masking, once per token instead of 48 selects per thread.
                    # A padded row only has to leave kA zero: it is the *rows* of kA
                    # that enter A^T and the state update, beta is already zeroed for
                    # padded tokens (so u and v_new vanish there), q2/k2 rows for
                    # padded tokens reach only their own discarded output, and the
                    # chunk decay they corrupt belongs to a state no one reads -- a
                    # partial chunk is always a sequence's last.  That lets the gate
                    # and the raw k tile stay unmasked even though 2^-cs then
                    # overflows on the padded rows.
                    for p in range(4):
                        qn = hmul2(qraw[4 * i + p], rq2[i])
                        kn = hmul2(kraw[4 * i + p], rk2[i])
                        K.assign(qgI[p], hmul2(qn, E2[4 * i + p]))
                        K.assign(knI[p], hmul2(kn, E2[4 * i + p]))
                        kav = hmul2(kn, F2[4 * i + p])
                        K.assign(
                            F2[4 * i + p],
                            K.Select(t < nval, kav, K.uint32(0)) if not full_chunks else kav,
                        )
                    st4u(q2t.ptr_to(t, d0), qgI, 0)
                    st4u(k2t.ptr_to(t, d0), knI, 0)
                    st4u(kA[s].ptr_to(t, d0), F2, 4 * i)
                    dbg_tile(c, 0, t, [qgI[p] for p in range(4)])
                    dbg_tile(c, 1, t, [F2[4 * i + p] for p in range(4)])
                    dbg_tile(c, 2, t, [knI[p] for p in range(4)])
                K.ptx.fence.proxy.async_.shared__cta()
                warrive(rfull, s)
                pmark("pr_rfull", c)

        with K.If(Lv > 0), K.Then():
            with r_state:
                state_body()
            with wg1:
                with r_ld:
                    load_body()
                with r_mmi:
                    mmi_body()
                with r_mmc:
                    mmc_body()
                with r_rnd:
                    rounds_body()
            with r_intra:
                intra_body()
            with r_prep:
                prep_body()

        K.cuda.cta_sync()
        with K.If(warp == 0), K.Then():
            tmark("cta_end", 0)
        with K.If(warp == 0), K.Then():
            K.ptx[TMEM_RELINQ]()
            K.ptx[TMEM_DEALLOC](tbase, K.uint32(N_COLS))

    kda_fwd.__annotations__ = {
        "q_map": K.TensorMap,
        "k_map": K.TensorMap,
        "v_map": K.TensorMap,
        "g_map": K.TensorMap,
        "beta_map": K.TensorMap,
        "o_map": K.TensorMap,
        "o": K.gptr[K.bf16],
        "A_log": K.gptr[K.f32, (H,)],
        "dt_bias": K.gptr[K.f32, (H * D,)],
        "h0": K.gptr[K.f32],
        "scale": K.f32,
        "cu_seqlens": K.gptr[K.i64],
        "num_seqs": K.i32,
        "num_ctas": K.i32,
        "dbg": (K.gptr[K.u64] if trace else K.gptr[K.f32, (DBG_TOTAL if debug else 8,)]),
        "dbg_c": K.i32,
    }
    return K.kernel(warps=NWARPS, arch="sm_100a", min_blocks_per_sm=1, grid="num_ctas")(kda_fwd)


_KERNELS = {}
_DBG_DUMMY = {}
_CU_DEFAULT = {}
_TMAP_ENCODE = None
# Retuned after the instruction-fetch work: intra gains the 16 registers the MMA
# issue warpgroup no longer needs (+2.3% on both uniform shapes, +1.4% on h96 mixed).
_REG_TUNE = tuple(int(x) for x in os.environ.get("KDA_REG_TUNE", "112,40,104,112").split(","))
_FIXED_REG_TUNE = tuple(
    int(x) for x in os.environ.get("KDA_FIXED_REG_TUNE", "112,40,104,112").split(",")
)
_FIXED_REG_TUNE_H64 = tuple(
    int(x) for x in os.environ.get("KDA_FIXED_REG_TUNE_H64", "112,32,112,112").split(",")
)
_MIXED_REG_TUNE_H64 = tuple(
    int(x) for x in os.environ.get("KDA_MIXED_REG_TUNE_H64", "112,48,96,112").split(",")
)
_FULL_MODE = os.environ.get("KDA_FULL_MODE", "all")
_UNROLL_CHUNKS = bool(int(os.environ.get("KDA_UNROLL_CHUNKS", "0")))
_STATE_F32X2 = bool(int(os.environ.get("KDA_STATE_F32X2", "1")))
_SREADY = int(os.environ.get("KDA_SREADY", "2"))


class _TensorMap:
    __slots__ = ("_storage", "ptr")

    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_map(tensor, T, H):
    """3D map over a [T, H, 128] bf16 tensor; args: dims (innermost first), byte strides of dims 1-2, box, element strides, interleave/swizzle/L2 promo/oob fill."""
    global _TMAP_ENCODE
    if _TMAP_ENCODE is None:
        _TMAP_ENCODE = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    m = _TensorMap()
    _TMAP_ENCODE(
        m.ptr,
        "bfloat16",
        3,
        ctypes.c_void_p(int(tensor.data_ptr())),
        64,
        T,
        2 * H,
        H * 256,
        128,
        64,
        64,
        2,
        1,
        1,
        1,
        0,
        3,
        2,
        0,
    )
    return m


def _encode_beta_map(tensor, T, H):
    """2D map over the [T, H] bf16 beta logits: dims (H, T), row stride 2H bytes, box (8 heads, 64 tokens)."""
    global _TMAP_ENCODE
    if _TMAP_ENCODE is None:
        _TMAP_ENCODE = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    m = _TensorMap()
    _TMAP_ENCODE(
        m.ptr,
        "bfloat16",
        2,
        ctypes.c_void_p(int(tensor.data_ptr())),
        H,
        T,
        H * 2,
        8,
        64,
        1,
        1,
        0,
        0,
        2,
        0,
    )
    return m


def _get_kernel(H, direct=False, num_seqs=1, T=None, *, fixed_shape=False, mixed_shape=False):
    fixed_shape = bool(fixed_shape)
    mixed_shape = bool(mixed_shape)
    # Only the no-cu_seqlens single-sequence case has a layout implied by the
    # call itself.  Packed launches must consume their live cu_seqlens: the
    # verifier deliberately salts sequence lengths and ordering, so treating
    # ``num_seqs == 6/8`` as the production layout is incorrect.
    full = fixed_shape and _FULL_MODE in ("all", "fixed")
    fixed_nt = T // 64 if fixed_shape and T % 64 == 0 else 0
    fixed_tokens = T if fixed_shape else 0
    full = full and T % 64 == 0
    tune = (
        (_FIXED_REG_TUNE_H64 if H == 64 else _FIXED_REG_TUNE)
        if fixed_shape
        else (_MIXED_REG_TUNE_H64 if mixed_shape and H == 64 else _REG_TUNE)
    )
    # Equivalent readiness events have shape-dependent scheduling costs.  The
    # full merge is repeatably profitable on both mixed shapes and on H=96
    # fixed; keep H=64 fixed, uniform, and arbitrary shapes on the conservative
    # protocol (re-swept after the instruction-fetch work: 0.998x and 0.999x
    # respectively there, 1.007x on H=64 mixed).
    event_mode = 2 if (mixed_shape or (H == 96 and fixed_shape)) else 0
    # Publishing the decayed state in halves lets the next chunk's S-operand MMA
    # start inside the state update instead of after it.  The extra tcgen05 store
    # wait pays for itself only where the state warps are not already the ones
    # holding up gconv: measured 1.020x on H=64 fixed, but 0.96-0.98x on H=96 and
    # on every uniform layout, and -2.6% on H=64 mixed once that shape moved to
    # the merged event protocol and the split round commit.
    sready = _SREADY if (H == 64 and fixed_shape) else 1
    # Splitting the round commit so the intra transform can start on Akk's
    # diagonal block one round of MMAs earlier is worth 1.0-2.6% everywhere
    # except H=96 mixed, where the rounds warp's extra commit round trip
    # reproducibly costs 2.4%.
    split_rounds = not (H == 96 and mixed_shape)
    key = (
        H,
        bool(direct),
        num_seqs,
        T,
        tune,
        full,
        fixed_nt,
        _UNROLL_CHUNKS,
        fixed_tokens,
        _STATE_F32X2,
        event_mode,
        sready,
        split_rounds,
    )
    if key not in _KERNELS:
        kern = build_kernel(
            H,
            debug=False,
            direct=direct,
            state_regs=tune[0],
            wg_regs=tune[1],
            intra_regs=tune[2],
            prep_regs=tune[3],
            fixed_nt=fixed_nt,
            full_chunks=full,
            unroll_chunks=_UNROLL_CHUNKS,
            fixed_tokens=fixed_tokens,
            state_f32x2=_STATE_F32X2,
            event_mode=event_mode,
            sready_stages=sready,
            split_rounds=split_rounds,
        )
        target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
        with target:
            _KERNELS[key] = tvm.compile(kern.mod, target=target, tir_pipeline="tirx")
    return _KERNELS[key]


@torch.no_grad()
def run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None):
    B, T, H, Dh = q.shape
    assert B == 1 and Dh == 128 and v.shape[-1] == 128
    for t in (q, k, v, g, beta, A_log, dt_bias, initial_state):
        assert t.is_contiguous()
    dev = q.device
    fixed_shape = False
    mixed_shape = False
    if cu_seqlens is None:
        key = (dev, T)
        if key not in _CU_DEFAULT:
            _CU_DEFAULT[key] = torch.tensor([0, T], dtype=torch.int64, device=dev)
        cu = _CU_DEFAULT[key]
        num_seqs = 1
        direct = True
        fixed_shape = True
    else:
        cu = cu_seqlens
        if cu.dtype != torch.int64 or not cu.is_contiguous() or cu.device != dev:
            cu = cu.to(device=dev, dtype=torch.int64).contiguous()
        num_seqs = int(cu.shape[0]) - 1
        direct = num_seqs == 1 or num_seqs == 8 or num_seqs > 32
        mixed_shape = num_seqs == 6
    assert initial_state.dtype == torch.float32 and initial_state.shape == (num_seqs, H, 128, 128)
    o = torch.empty((1, T, H, 128), dtype=torch.bfloat16, device=dev)
    if T == 0:
        return o
    ex = _get_kernel(
        H, direct=direct, num_seqs=num_seqs, T=T, fixed_shape=fixed_shape, mixed_shape=mixed_shape
    )
    assert H % 8 == 0
    maps = [_encode_map(t, T, H) for t in (q, k, v, g)] + [
        _encode_beta_map(beta, T, H),
        _encode_map(o, T, H),
    ]
    if dev not in _DBG_DUMMY:
        _DBG_DUMMY[dev] = torch.empty(8, dtype=torch.float32, device=dev)
    ex(
        *[m.ptr for m in maps],
        o.view(-1),
        A_log,
        dt_bias.view(-1),
        initial_state.view(-1),
        float(scale),
        cu,
        num_seqs,
        num_seqs * H,
        _DBG_DUMMY[dev],
        -1,
    )
    return o


@dataclass(frozen=True, slots=True)
class KDAForwardConfig:
    label: str
    num_heads: int
    seq_lens: tuple[int, ...]
    seed: int = 0
    scale: float = 1.0 / math.sqrt(128)
    lower_bound: float = -5.0

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)

    @property
    def num_seqs(self) -> int:
        return len(self.seq_lens)

    @property
    def packed(self) -> bool:
        return self.num_seqs > 1

    @property
    def use_initial_state(self) -> bool:
        return True

    @property
    def store_final_state(self) -> bool:
        return False


_FIXED = (8192,)
_MIXED = (1300, 547, 2048, 963, 271, 3063)
_UNIFORM = (1024,) * 8
_LAYOUTS = {_FIXED, _MIXED, _UNIFORM}
CONFIGS = [
    {"label": f"h{heads}_{label}", "num_heads": heads, "seq_lens": lengths, "seed": seed}
    for heads, label, lengths, seed in (
        (96, "fixed", _FIXED, 2858210354),
        (96, "mixed", _MIXED, 2858210355),
        (96, "uniform", _UNIFORM, 2858210356),
        (64, "fixed", _FIXED, 2858210357),
        (64, "mixed", _MIXED, 2858210358),
        (64, "uniform", _UNIFORM, 2858210359),
    )
]
KERNEL_META = {
    "name": "agent_evolved_kda_forward_b1_t8192",
    "category": "agent_evolved",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a", "sm_110a"],
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
        "run": "kda-forward-handoff-20260906",
        "selected_version": "f280a7902118d21fb1e8f0f34a349238a4203b81",
    },
}


def _cfg(**kwargs: Any) -> KDAForwardConfig:
    values = {
        key: kwargs[key]
        for key in ("label", "num_heads", "seq_lens", "seed", "scale", "lower_bound")
        if key in kwargs
    }
    if "seq_lens" in values:
        values["seq_lens"] = tuple(int(length) for length in values["seq_lens"])
    values.setdefault("label", "custom")
    cfg = KDAForwardConfig(**values)
    if (
        cfg.num_heads not in (64, 96)
        or tuple(cfg.seq_lens) not in _LAYOUTS
        or cfg.total_tokens != 8192
    ):
        raise ValueError(f"unsupported KDA config: {cfg}")
    return cfg


def get_kernel(**kwargs: Any):
    cfg = _cfg(**kwargs)
    fixed = not cfg.packed
    mixed = cfg.seq_lens == _MIXED
    tune = (
        (112, 40, 104, 112)
        if fixed
        else (112, 48, 96, 112)
        if mixed and cfg.num_heads == 64
        else (112, 40, 104, 112)
    )
    kern = build_kernel(
        cfg.num_heads,
        direct=fixed or cfg.seq_lens == _UNIFORM,
        state_regs=tune[0],
        wg_regs=tune[1],
        intra_regs=tune[2],
        prep_regs=tune[3],
        fixed_nt=128 if fixed else 0,
        full_chunks=fixed,
        fixed_tokens=8192 if fixed else 0,
        state_f32x2=True,
        event_mode=2 if (mixed or (cfg.num_heads == 96 and fixed)) else 0,
        sready_stages=2 if (cfg.num_heads == 64 and fixed) else 1,
        split_rounds=not mixed,
    )
    return kern.func


def _make_case(cfg: KDAForwardConfig, device: torch.device) -> dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    shape = (1, cfg.total_tokens, cfg.num_heads, 128)

    def rand(
        shape: tuple[int, ...], scale: float, dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        return (
            torch.randn(shape, dtype=torch.float32, device=device, generator=generator) * scale
        ).to(dtype)

    q, k, v, g = (rand(shape, 0.5) for _ in range(4))
    beta = rand((1, cfg.total_tokens, cfg.num_heads), 0.5)
    a_log = torch.log(
        torch.empty(cfg.num_heads, device=device).uniform_(1.0, 16.0, generator=generator)
    )
    dt = torch.exp(
        torch.rand(cfg.num_heads * 128, device=device, generator=generator)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    ).clamp_(min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    initial = rand((cfg.num_seqs, cfg.num_heads, 128, 128), 0.25, torch.float32)
    cu = (
        None
        if not cfg.packed
        else torch.tensor(
            [0, *__import__("itertools").accumulate(cfg.seq_lens)], dtype=torch.int64, device=device
        )
    )
    return {
        "config": cfg,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "scale": cfg.scale,
        "initial_state": initial,
        "cu_seqlens": cu,
    }


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved KDA forward")
    cfg = _cfg(**kwargs)
    case = _make_case(cfg, torch.device(kwargs.get("device", "cuda")))
    case["out"] = torch.empty_like(case["q"])
    return case


def run_test(**kwargs: Any) -> None:
    cfg = _cfg(**kwargs)
    case = prepare_data(**kwargs)
    args = (
        case["q"],
        case["k"],
        case["v"],
        case["g"],
        case["beta"],
        case["A_log"],
        case["dt_bias"],
        cfg.scale,
        case["initial_state"],
        case["cu_seqlens"],
    )
    first = run(*args)
    actual = run(*args)
    torch.cuda.synchronize()
    os.environ["FLA_FLASH_KDA"] = "0"
    os.environ["FLA_TILELANG"] = "0"
    from fla.ops.kda import chunk_kda

    reference, _ = chunk_kda(
        q=case["q"],
        k=case["k"],
        v=case["v"],
        g=case["g"],
        beta=case["beta"],
        scale=cfg.scale,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
        safe_gate=True,
        lower_bound=-5.0,
        A_log=case["A_log"],
        dt_bias=case["dt_bias"],
        initial_state=case["initial_state"],
        cu_seqlens=case["cu_seqlens"],
    )
    if not torch.equal(first, actual):
        raise AssertionError("KDA output is not repeatable")
    torch.testing.assert_close(actual, reference, atol=5e-2, rtol=5e-2)


def prepare_bench(**kwargs: Any):
    from tirx_kernels.runner import prepared_gpu_benchmark

    return prepared_gpu_benchmark(run_gpu, {"config": dict(kwargs)})


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, **kwargs: Any) -> dict[str, Any]:
    from tirx_kernels.runner import bench

    config = dict(prepared["config"])
    config.update(kwargs)
    case = prepare_data(**config)
    args = (
        case["q"],
        case["k"],
        case["v"],
        case["g"],
        case["beta"],
        case["A_log"],
        case["dt_bias"],
        case["scale"],
        case["initial_state"],
        case["cu_seqlens"],
    )
    case["out"] = run(*args)
    torch.cuda.synchronize()

    def _flashkda_builder():
        from tirx_kernels.flashinfer.utils._flashkda_bench import prepare_flashkda_raw_reference

        return prepare_flashkda_raw_reference(case).launch

    return bench(
        {"tirx": lambda: run(*args)},
        references={"flash_kda": _flashkda_builder},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=config.get("rounds", 5),
        cooldown_s=config.get("cooldown_s", 1.0),
    )


def run_bench(*, warmup=None, repeat=None, timer=None, **kwargs: Any) -> dict[str, Any]:
    return run_gpu({"config": kwargs}, warmup=warmup, repeat=repeat, timer=timer)


__all__ = [
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_test",
]
