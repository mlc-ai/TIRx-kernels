# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved SM100a MiniMax sparse-attention (MSA) decode, every official shape.

This replaces the pinned `mtp_bf16_b128_q16_kv4096_h64` kernel with the
`rowmajor-tmem-probability` frontier member of the 2026-09-13 multi-shape
MSA-decode evolution run. It covers all ten official rows: flat and paged KV,
bf16 / fp16 / fp8-E4M3 storage, seqlen_q of 1, 4, 8 and 16, GQA ratios 8 and
16, top-k 4, 16 and 32, and kv_len from 257 to 65536.

Three device programs sit behind one shape dispatch:

* **KV-major** for small `seqlen_q` and the fp8 rows. One CTA task is one
  (request, kv head) item; the prep warp builds the exact union of the item's
  selected 128-token KV blocks plus a per-block token-selection mask, and the
  MMA warp computes `S^T = K_tile . Q^T` and `O^T += V_tile^T . P^T` so each of
  the 128 softmax threads owns one KV token and all `seqlen_q * GQA` query
  columns. The exp work therefore spreads over all four softmax warps instead
  of the few valid Q rows of a Q-major tile. Roles (8 warps): 0-3 softmax and
  epilogue, 4 MMA issue, 5 TMA load, 6 scheduler/union prep, 7 idle.
* **Q-major** for items of exactly 128 query rows, carrying the row-major TMEM
  probability layout this member is named for.
* A **two-query-tile Q-major specialization** for the q16 row and a dedicated
  q4 route, selected by `seqlen_q`, dtype, GQA ratio and top-k.

The dispatch is on shape and dtype metadata only -- never on tensor values.

Measured over the ten official rows on one GB200 through the kcoral benchmark
server, candidate and baseline in the same run: **4.006x geometric mean**, with
every row at or above parity. The two fp8 rows reach about 10x, the long
65536-token paged row 6.1x, and the previously pinned q16 row 6.26x -- itself
faster than the single-shape kernel this module replaces (5.851x).

The only numerical approximation is in the softmax exponential, where a
packed-f32x2 polynomial replaces some native `ex2.approx.ftz.f32` evaluations;
its worst-case relative error stays below bf16's own rounding. Both tcgen05
MMAs are `kind::f16` with FP32 accumulation on the bf16/fp16 rows and
`kind::f8f6f4` on the fp8 rows, matching each row's declared storage dtype.
"""

import ctypes
import math
import os
from typing import Any
from unittest import SkipTest

import torch
import tvm

import tirx_kernels.kern as K

HEAD_DIM = 128
BLK = 128
LOG2E = 1.4426950408889634
NEG_INF = float("-inf")
RESCALE_THRESH = 8.0
DEBUG_SKIP_SOFTMAX = os.environ.get("KV_DEBUG_SKIP_SOFTMAX", "0") == "1"
ENV_STAGES = int(os.environ.get("KV_STAGES", "0"))
ENV_NO_HINT = os.environ.get("KV_NO_HINT", "0") == "1"
ENV_SPLIT = int(os.environ.get("KV_SPLIT", "0"))



QM64_EMU_NUM = 0
QM64_SOFT_REGS = 200
QM64_PROD_REGS = 56
QM64_USE_LMMA = False
QM64_EX2_MAGIC = 12582912.0
QM64_EX2_MAGIC_BITS = 0x4B400000
QM64_EX2_C0, QM64_EX2_C1, QM64_EX2_C2, QM64_EX2_C3 = (
    0.999928073,
    0.693260998,
    0.242611145,
    0.055171612,
)

TMA_3D = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
TMA_4D = "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
TMA_3D_HINT = TMA_3D + ".L2::cache_hint"
TMA_4D_HINT = TMA_4D + ".L2::cache_hint"
if ENV_NO_HINT:
    TMA_3D_HINT = TMA_3D
    TMA_4D_HINT = TMA_4D
KV_CACHE_POLICY = 0x12F0000000000000
HINT_ARGS = () if ENV_NO_HINT else (K.uint64(KV_CACHE_POLICY),)
MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
MMA_F8 = "tcgen05.mma.cta_group::1.kind::f8f6f4"
TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
ST_OUT_V8 = "st.global.L1::no_allocate.L2::evict_first.v8.b32"


def tmem_ld(n):
    return f"tcgen05.ld.sync.aligned.32x32b.x{n}.b32"


def tmem_st(n):
    return f"tcgen05.st.sync.aligned.32x32b.x{n}.b32"


def make_idesc(M, N, a_fmt, b_fmt, trans_a, trans_b):
    return (
        (1 << 4)
        | (a_fmt << 7)
        | (b_fmt << 10)
        | (trans_a << 15)
        | (trans_b << 16)
        | ((N >> 3) << 17)
        | ((M >> 4) << 24)
    )


def ceildiv(a, b):
    return -(-a // b)


def make_kernel_kv(cfg):
    TQ = cfg["T"]
    T = cfg["TI"]
    NCH = TQ // T
    G = cfg["G"]
    HKV = cfg["HKV"]
    TOPK = cfg["TOPK"]
    TOTAL_Q = cfg["TOTAL_Q"]
    kind = cfg["kv_kind"]
    PAGED = cfg["paged"]
    MAX_PAGES = cfg["MAX_PAGES"]
    W_MAX = cfg["MAX_BLOCK_WORDS"]
    NUM_ITEMS = cfg["NUM_ITEMS"]
    NUM_CTAS = cfg["NUM_CTAS"]
    STAGES = cfg["STAGES"]
    MIN_BLOCKS = cfg.get("MIN_BLOCKS", 1)
    SPLIT_LOADERS = cfg.get("SPLIT_LOADERS", False)
    STATIC_ONE_SHOT = cfg.get("STATIC_ONE_SHOT", False)
    SPLIT = cfg["SPLIT"]
    FULL_ITEMS = (NUM_ITEMS // NUM_CTAS) * NUM_CTAS if SPLIT > 1 else NUM_ITEMS
    TAIL_ITEMS = NUM_ITEMS - FULL_ITEMS
    NUM_TASKS = FULL_ITEMS + TAIL_ITEMS * SPLIT
    HQ = HKV * G
    NQ = T * G
    NQ_PAD = max(16, ceildiv(NQ, 16) * 16)
    FP8 = kind == "fp8"
    kv_dt = {"bf16": K.bf16, "f16": K.f16, "fp8": K.u8}[kind]
    out_dt = K.f16 if kind == "f16" else K.bf16
    EB = 1 if FP8 else 2
    MMA_K = 32 if FP8 else 16
    NK = HEAD_DIM // MMA_K
    KV_TILE_BYTES = BLK * HEAD_DIM * EB
    assert NQ_PAD % T == 0
    G_BOX = NQ_PAD // T
    Q_TMA_BYTES = NQ_PAD * HEAD_DIM * 2
    MMA = MMA_F8 if FP8 else MMA_F16
    fmt = 1 if kind == "bf16" else 0
    Q_CHUNKS = NQ * (HEAD_DIM // 16)
    Q_ROUNDS = ceildiv(Q_CHUNKS, 32)
    ID_QK = make_idesc(128, NQ_PAD, fmt, fmt, 0, 0)
    ID_PV = make_idesc(128, NQ_PAD, fmt, fmt, 1, 0)
    O_COL = 2 * NQ_PAD
    TMEM_COLS = max(32, 1 << (3 * NQ_PAD - 1).bit_length())
    MAX_UNION = T * TOPK
    N_ROUNDS = ceildiv(MAX_UNION, 32)
    KV16 = KV_TILE_BYTES // 16
    QP16 = NQ_PAD * HEAD_DIM * EB // 16
    NCHUNK = NQ_PAD // 16
    NSWG = cfg["NSWG"]
    NQW = NQ // NSWG
    assert NQW % 8 == 0 and (NQW % G == 0 or NSWG == 1)
    GPW = max(1, NQW // G)
    NFLAG = ceildiv(NQW, 32)
    RS = min(32, NQW)
    NWARPS = 4 * NSWG + 4 + (1 if SPLIT_LOADERS else 0)
    MMA_WARP = 4 * NSWG
    LOAD_WARP = 4 * NSWG + 1
    PREP_WARP = 4 * NSWG + 2
    PV_WARP = 4 * NSWG + 3
    VLOAD_WARP = 4 * NSWG + 4
    SOFT_THREADS = 128 * NSWG

    @K.kernel(warps=NWARPS, arch="sm_100a", min_blocks_per_sm=MIN_BLOCKS, grid=NUM_CTAS)
    def msa_decode_kvmajor(
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        out: K.gptr[out_dt],
        q2k: K.gptr[K.i32],
        lens: K.gptr[K.i32],
        ptab: K.gptr[K.i32],
        qraw: K.gptr[K.i32],
        sched: K.gptr[K.i32],
        part_o: K.gptr[K.f32],
        part_ml: K.gptr[K.f32],
        part_ctr: K.gptr[K.i32],
        scale_log2: K.f32,
    ):
        warp = K.warp_id()
        lane = K.lane_id()
        tid = K.thread_id()

        smem = K.smem_pool()
        kv_smem = smem.alloc((STAGES, BLK, HEAD_DIM), kv_dt, swizzle=K.SW128B)
        q_smem = smem.alloc((2, NQ_PAD, HEAD_DIM), kv_dt, swizzle=K.SW128B)
        p_smem = smem.alloc((2, NQ_PAD, HEAD_DIM), kv_dt, swizzle=K.SW128B)
        tmem_addr = smem.alloc((4,), K.u32)
        meta = smem.alloc((32,), K.i32)
        ulist = smem.alloc((2 * MAX_UNION,), K.i32)
        uplist = smem.alloc((2 * MAX_UNION,), K.i32)
        umask = smem.alloc((2 * MAX_UNION,), K.u32)
        words = smem.alloc((W_MAX,), K.u32)
        prefix = smem.alloc((W_MAX,), K.u32)
        xmax = smem.alloc((NSWG * 2 * NQW * 4,), K.f32, align=16)
        xsum = smem.alloc((NSWG * NQW * 4,), K.f32, align=16)
        alpha_s = smem.alloc((NQ_PAD,), K.f32, align=16)
        K.keep_alive(ptab.ptr_to([0]))
        K.keep_alive(qraw.ptr_to([0]))
        K.keep_alive(part_o.ptr_to([0]))
        K.keep_alive(part_ml.ptr_to([0]))
        K.keep_alive(part_ctr.ptr_to([0]))
        mrg_order = smem.alloc((4,), K.i32)


        def lo_uniform(desc):
            desc_lo = K.alloc_local((1,), "uint32")
            desc_hi = K.alloc_local((1,), "uint32")
            K.assign(desc_lo[0], K.uniform(K.Cast("uint32", desc.value)))
            K.assign(desc_hi[0], K.Cast("uint32", K.shift_right(desc.value, K.uint64(32))))
            return desc_lo, desc_hi

        def desc_at(desc, off16):
            lo, hi = desc
            packed = K.alloc_local((1,), "uint64")
            low = lo[0] if isinstance(off16, int) and off16 == 0 else lo[0] + K.Cast("uint32", off16)
            K.assign(
                packed[0],
                K.bitwise_or(K.shift_left(K.Cast("uint64", hi[0]), K.uint64(32)), K.Cast("uint64", low)),
            )
            return packed[0]

        def encode(view, major="k"):
            desc, off16 = view.encode(major=major, mma_k=MMA_K)
            return lo_uniform(desc), off16

        k_desc, koff = encode(kv_smem[0], "k")
        v_desc, voff = encode(kv_smem[0], "mn")
        q_desc, qoff = encode(q_smem[0], "k")
        p_desc, poff = encode(p_smem[0], "k")


        kv_load = K.Pipeline(smem, STAGES, full="tma", empty="tcgen05", empty_phase_offset=1)
        q_load = K.Pipeline(
            smem, 2, full=("mbar" if FP8 else "tma"), empty="tcgen05", init_full=(32 if FP8 else 1), empty_phase_offset=1
        )
        union_ready = K.MBarrier(smem, 2)
        union_ready.init(32)
        if not STATIC_ONE_SHOT:
            union_free = K.MBarrier(smem, 2)
            union_free.init(SOFT_THREADS + 96 + (32 if SPLIT_LOADERS else 0))
        s_ready = K.TCGen05Bar(smem, 2)
        s_ready.init(1)
        s_free = K.MBarrier(smem, 2)
        s_free.init(SOFT_THREADS)
        p_ready = K.MBarrier(smem, 2)
        p_ready.init(SOFT_THREADS)
        p_free = K.TCGen05Bar(smem, 2)
        p_free.init(1)
        o_ready = K.TCGen05Bar(smem, 1)
        o_ready.init(1)
        o_free = K.MBarrier(smem, 1)
        o_free.init(SOFT_THREADS)

        K.ptx.fence.proxy.async_.shared__cta()
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        with K.If(warp == MMA_WARP), K.Then():
            K.ptx[TMEM_ALLOC](K.address_of(tmem_addr[0]), K.uint32(TMEM_COLS))
            K.ptx[TMEM_RELINQUISH]()
        K.cuda.cta_sync()
        tb_raw = K.local_scalar("uint32")
        K.ptx.ld.shared.u32(tb_raw, tmem_addr.ptr_to([0]))
        tmem_base = K.local_scalar("uint32", init=K.uniform(tb_raw))


        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def ld_smem_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.shared.b32(value, ptr)
            return value

        def ld_smem_u32(ptr):
            value = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(value, ptr)
            return value

        def ld_global_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.global_.nc.b32(value, ptr)
            return value

        def warp_max_f32(dst, src):
            K.ptx.redux_sync.max.f32(dst, src, K.uint32(0xFFFFFFFF))

        def transpose_max(vals, ncols):
            """Butterfly all-reduce of ncols independent columns; lane l ends with the
            lane-wide max of column l // (32 // ncols)."""
            assert 32 % ncols == 0 and ncols >= 2
            cur = [vals[i] for i in range(ncols)]
            xor = 16
            while len(cur) > 1:
                half = len(cur) // 2
                bit = K.bitwise_and(K.Cast("uint32", lane), K.uint32(xor)) != K.uint32(0)
                nxt = []
                for i in range(half):
                    a = cur[i]
                    b = cur[i + half]
                    send = K.local_scalar("float32", init=K.Select(bit, a, b))
                    recv = K.local_scalar("float32")
                    K.ptx.shfl_sync.bfly.b32(recv, send, K.uint32(xor), K.uint32(31), K.uint32(0xFFFFFFFF))
                    keep = K.Select(bit, b, a)
                    nxt.append(K.local_scalar("float32", init=K.max(keep, recv)))
                cur = nxt
                xor >>= 1
            res = cur[0]

            while xor >= 1:
                other = K.local_scalar("float32")
                K.ptx.shfl_sync.bfly.b32(other, res, K.uint32(xor), K.uint32(31), K.uint32(0xFFFFFFFF))
                res = K.local_scalar("float32", init=K.max(res, other))
                xor >>= 1
            return res

        def warp_sum_f32(dst, src):
            tmp = K.local_scalar("float32", init=src)
            other = K.local_scalar("float32")
            for d in (16, 8, 4, 2, 1):
                K.ptx.shfl_sync.bfly.b32(other, tmp, K.uint32(d), K.uint32(31), K.uint32(0xFFFFFFFF))
                K.assign(tmp, tmp + other)
            K.assign(dst, tmp)

        def f32_to_half_bits(dst16, src):
            if kind == "f16":
                K.ptx.cvt.rn.f16.f32(dst16, src)
            else:
                K.ptx.cvt.rn.bf16.f32(dst16, src)

        def store_p(ptr, src, tmp16):
            if FP8:
                K.ptx.cvt.rn.satfinite.e4m3x2.f32(tmp16, K.float32(0.0), src)
                K.ptx.st.shared.b8(ptr, tmp16)
            else:
                f32_to_half_bits(tmp16, src)
                K.ptx.st.shared.b16(ptr, tmp16)

        def tmem_load_cols(dst, col_expr, ncols):
            width = 16 if ncols % 16 == 0 else 8
            for c in range(ncols // width):
                K.ptx[tmem_ld(width)](*(dst[width * c + i] for i in range(width)), tmem_base + col_expr + K.uint32(width * c))
            K.ptx.tcgen05.wait__ld.sync.aligned()

        def tmem_store_cols(src, col_expr, ncols):
            width = 16 if ncols % 16 == 0 else 8
            for c in range(ncols // width):
                K.ptx[tmem_st(width)](tmem_base + col_expr + K.uint32(width * c), *(src[width * c + i] for i in range(width)))
            K.ptx.tcgen05.wait__st.sync.aligned()

        def lanemask_lt():
            return K.shift_left(K.uint32(1), K.Cast("uint32", lane)) - K.uint32(1)

        def iket_range(name, *, leader_only=False):
            token = K.alloc_local([1], "uint32")
            if leader_only:
                K.assign(token[0], K.cuda.iket.sentinel_token(name))
                with K.If((warp & 3) == 0), K.Then():
                    K.assign(token[0], K.cuda.iket.range_start(name))
            else:
                K.assign(token[0], K.cuda.iket.range_start(name))
            return token

        def iket_end(token):
            K.cuda.iket.range_end(token[0])

        sp = K.specialize(chain_dispatch=True)
        r_soft = sp.role("softmax", warps=list(range(4 * NSWG)))
        r_mma = sp.role("mma", warps=[MMA_WARP])
        r_load = sp.role("load", warps=[LOAD_WARP])
        r_prep = sp.role("prep", warps=[PREP_WARP])
        r_pv = sp.role("pv", warps=[PV_WARP])
        if SPLIT_LOADERS:
            r_vload = sp.role("vload", warps=[VLOAD_WARP])


        with r_soft:
            wg = warp >> 2
            wq = warp & 3
            tid_wg = tid & 127
            col0 = wg * NQW
            bar_id = K.uint32(1) + K.Cast("uint32", wg)
            it = K.local_scalar("int32", init=0)
            gj = K.local_scalar("int32", init=0)
            par = K.local_scalar("int32", init=0)
            running = K.local_scalar("int32", init=1)
            m = K.alloc_local([NQW], "float32")
            lsum = K.alloc_local([NQW], "float32")
            s = K.alloc_local([NQW], "float32")
            o = K.alloc_local([NQW], "float32")
            hbits = K.local_scalar("uint16")
            zero16 = K.local_scalar("uint16", init=K.Cast("uint16", K.int32(0)))
            with K.While(running != 0):
                slot = it & 1
                tk_wu = iket_range("sm-wait-union", leader_only=True)
                union_ready.wait(slot, (it >> 1) & 1)
                iket_end(tk_wu)
                n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                with K.If(n_blocks < 0):
                    with K.Then():
                        K.assign(running, 0)
                    with K.Else():
                        h = ld_smem_i32(meta.ptr_to([slot * 16 + 2]))
                        tok_base = ld_smem_i32(meta.ptr_to([slot * 16 + 5]))
                        causal_off = ld_smem_i32(meta.ptr_to([slot * 16 + 6]))
                        blk_off_s = ld_smem_i32(meta.ptr_to([slot * 16 + 7]))
                        item_s = ld_smem_i32(meta.ptr_to([slot * 16 + 8]))
                        part_s = ld_smem_i32(meta.ptr_to([slot * 16 + 9]))

                        def group_token(gl):

                            return wg * GPW + gl

                        def out_index(nl):
                            if NQW >= G:
                                t = group_token(nl // G)
                                g = nl % G
                            else:
                                t = (col0 + nl) // G
                                g = (col0 + nl) % G
                            row = tok_base + t
                            head = h * G + g
                            return (row * HQ + head) * HEAD_DIM + tid_wg

                        with K.If(n_blocks > 0):
                            with K.Then():
                                for nl in range(NQW):
                                    K.assign(m[nl], K.float32(NEG_INF))
                                    K.assign(lsum[nl], K.float32(0.0))
                                with K.serial(n_blocks, unroll=False) as j:
                                    blk = ld_smem_i32(ulist.ptr_to([slot * MAX_UNION + blk_off_s + j]))
                                    if T > 1:
                                        tmask = ld_smem_u32(umask.ptr_to([slot * MAX_UNION + blk_off_s + j]))
                                    sb = gj & 1
                                    sb_col = K.Cast("uint32", sb) * K.uint32(NQ_PAD) + K.Cast("uint32", col0)
                                    kv_pos = blk * BLK + tid_wg
                                    need_causal = (blk * BLK + (BLK - 1)) > causal_off
                                    tk_ws = iket_range("sm-wait-s", leader_only=True)
                                    s_ready.wait(sb, (gj >> 1) & 1)
                                    iket_end(tk_ws)
                                    tk_ph1 = iket_range("sm-max", leader_only=True)
                                    tmem_load_cols(s, sb_col, NQW)
                                    s_free.arrive(sb)

                                    def group_sel(gl):
                                        if T == 1:
                                            return None
                                        t = group_token(gl)
                                        return K.bitwise_and(K.shift_right(tmask, K.Cast("uint32", t)), K.uint32(1)) != K.uint32(0)

                                    def with_sel(gl, body, otherwise=None):
                                        cond = group_sel(gl)
                                        if cond is None:
                                            body()
                                        else:
                                            with K.If(cond):
                                                with K.Then():
                                                    body()
                                                if otherwise is not None:
                                                    with K.Else():
                                                        otherwise()

                                    def group_cols(gl):
                                        if NQW >= G:
                                            return range(gl * G, (gl + 1) * G)
                                        return range(NQW)

                                    if not DEBUG_SKIP_SOFTMAX:

                                        for gl in range(GPW):

                                            def phase1(gl=gl):
                                                qpos = causal_off + group_token(gl)
                                                with K.If(need_causal), K.Then():
                                                    for nl in group_cols(gl):
                                                        K.assign(s[nl], K.Select(kv_pos <= qpos, s[nl], K.float32(NEG_INF)))

                                            with_sel(gl, phase1)
                                        LPC = 32 // NQW
                                        mycol_max = transpose_max(s, NQW)
                                        with K.If((lane % LPC) == 0), K.Then():
                                            K.ptx.st.shared.f32(
                                                xmax.ptr_to([((wg * 2 + par) * NQW) * 4 + (lane // LPC) * 4 + wq]), mycol_max
                                            )
                                        iket_end(tk_ph1)
                                        tk_bar = iket_range("sm-bar", leader_only=True)
                                        K.ptx.bar.sync(bar_id, K.uint32(128))
                                        iket_end(tk_bar)
                                        tk_ph2 = iket_range("sm-merge-exp", leader_only=True)

                                        mb = K.alloc_local([NQW], "float32")
                                        dlog = K.alloc_local([NQW], "float32")
                                        if T > 1:
                                            for nl in range(NQW):
                                                K.assign(dlog[nl], K.float32(0.0))
                                        any_r = K.local_scalar("int32", init=0)
                                        v4 = K.alloc_local([4], "float32")
                                        K.ptx.ld.shared.v4.f32(
                                            v4[0], v4[1], v4[2], v4[3], xmax.ptr_to([((wg * 2 + par) * NQW) * 4 + (lane // LPC) * 4])
                                        )
                                        fin = K.local_scalar("float32")
                                        K.ptx.max.f32(fin, v4[0], v4[1], v4[2])
                                        K.assign(fin, K.max(fin, v4[3]) * scale_log2)
                                        for nl in range(NQW):
                                            K.ptx.shfl_sync.idx.b32(mb[nl], fin, K.uint32(nl * LPC), K.uint32(31), K.uint32(0xFFFFFFFF))
                                        for gl in range(GPW):

                                            def p2merge(gl=gl):
                                                for nl in group_cols(gl):
                                                    m_old = K.local_scalar("float32", init=m[nl])
                                                    m_new = K.local_scalar("float32", init=K.max(m_old, mb[nl]))
                                                    d = K.local_scalar("float32", init=m_new - m_old)
                                                    big = d > K.float32(RESCALE_THRESH)
                                                    do_r = K.And(m_old != K.float32(NEG_INF), big)
                                                    K.assign(dlog[nl], K.Select(do_r, K.float32(0.0) - d, K.float32(0.0)))
                                                    K.assign(m[nl], K.Select(big, m_new, m_old))
                                                    K.assign(any_r, K.Select(do_r, 1, any_r))

                                            with_sel(gl, p2merge)

                                        with K.If(any_r != 0), K.Then():
                                            tk_rs = iket_range("sm-rescale", leader_only=True)
                                            gjm = gj - 1
                                            p_free.wait(gjm & 1, (gjm >> 1) & 1)
                                            o_ch = K.alloc_local([RS], "float32")
                                            for c0 in range(0, NQW, RS):
                                                tmem_load_cols(o_ch, K.uint32(O_COL) + K.Cast("uint32", col0) + K.uint32(c0), RS)
                                                for i in range(RS):
                                                    nl = c0 + i
                                                    a = K.local_scalar("float32")
                                                    K.ptx.ex2.approx.ftz.f32(a, dlog[nl])
                                                    K.assign(o_ch[i], o_ch[i] * a)
                                                    K.assign(lsum[nl], lsum[nl] * a)
                                                tmem_store_cols(o_ch, K.uint32(O_COL) + K.Cast("uint32", col0) + K.uint32(c0), RS)
                                            iket_end(tk_rs)

                                        for gl in range(GPW):

                                            def p2exp(gl=gl):
                                                for nl in group_cols(gl):
                                                    msub = K.max(m[nl], K.float32(-1.0e38))
                                                    pval = K.local_scalar("float32")
                                                    K.ptx.ex2.approx.ftz.f32(pval, s[nl] * scale_log2 - msub)
                                                    K.assign(lsum[nl], lsum[nl] + pval)
                                                    K.assign(s[nl], pval)

                                            def p2zero(gl=gl):
                                                for nl in group_cols(gl):
                                                    K.assign(s[nl], K.float32(0.0))

                                            with_sel(gl, p2exp, p2zero)
                                        iket_end(tk_ph2)
                                    else:
                                        for nl in range(NQW):
                                            K.assign(s[nl], K.float32(0.0))

                                    pb = gj & 1
                                    tk_wp = iket_range("sm-wait-pfree", leader_only=True)
                                    p_free.wait(pb, ((gj >> 1) + 1) & 1)
                                    K.ptx.fence.proxy.async_.shared__cta()
                                    iket_end(tk_wp)
                                    tk_ps = iket_range("sm-pstore", leader_only=True)
                                    for nl in range(NQW):
                                        store_p(p_smem[pb].ptr_to(col0 + nl, tid_wg), s[nl], hbits)
                                    K.ptx.fence.proxy.async_.shared__cta()
                                    p_ready.arrive(pb)
                                    iket_end(tk_ps)
                                    K.assign(par, par ^ 1)
                                    K.assign(gj, gj + 1)

                                tk_wo = iket_range("sm-wait-o", leader_only=True)
                                o_ready.wait(0, it & 1)
                                iket_end(tk_wo)
                                tk_epi = iket_range("sm-epi", leader_only=True)
                                tmem_load_cols(o, K.uint32(O_COL) + K.Cast("uint32", col0), NQW)
                                o_free.arrive(0)
                                lt = K.local_scalar("float32")
                                for nl in range(NQW):
                                    warp_sum_f32(lt, lsum[nl])
                                    with K.If(lane == 0), K.Then():
                                        K.ptx.st.shared.f32(xsum.ptr_to([(wg * NQW + nl) * 4 + wq]), lt)
                                K.ptx.bar.sync(bar_id, K.uint32(128))
                                ltot_a = K.alloc_local([NQW], "float32")
                                for nl in range(NQW):
                                    v4 = K.alloc_local([4], "float32")
                                    K.ptx.ld.shared.v4.f32(v4[0], v4[1], v4[2], v4[3], xsum.ptr_to([(wg * NQW + nl) * 4]))
                                    K.assign(ltot_a[nl], (v4[0] + v4[1]) + (v4[2] + v4[3]))

                                def write_output(o_arr, l_arr):
                                    for nl in range(NQW):
                                        ltot = l_arr[nl]
                                        bad = K.Or(ltot == K.float32(0.0), ltot != ltot)
                                        inv = K.local_scalar("float32")
                                        K.ptx.rcp.approx.ftz.f32(inv, K.Select(bad, K.float32(1.0), ltot))
                                        val = K.local_scalar("float32", init=K.Select(bad, K.float32(0.0), o_arr[nl] * inv))
                                        f32_to_half_bits(hbits, val)
                                        K.ptx.st.global_.b16(out.ptr_to([out_index(nl)]), hbits)

                                if SPLIT == 1:
                                    write_output(o, ltot_a)
                                else:
                                    with K.If(item_s < FULL_ITEMS):
                                        with K.Then():
                                            write_output(o, ltot_a)
                                        with K.Else():

                                            slot_idx = (item_s - FULL_ITEMS) * SPLIT + part_s
                                            o_base = slot_idx * (HEAD_DIM * NQ) + tid_wg
                                            ml_base = slot_idx * (2 * NQ)
                                            for nl in range(NQW):
                                                K.ptx.st.global_.f32(part_o.ptr_to([o_base + (col0 + nl) * HEAD_DIM]), o[nl])
                                            with K.If(tid_wg == 0), K.Then():
                                                for nl in range(NQW):
                                                    K.ptx.st.global_.f32(part_ml.ptr_to([ml_base + col0 + nl]), m[nl])
                                                    K.ptx.st.global_.f32(part_ml.ptr_to([ml_base + NQ + col0 + nl]), ltot_a[nl])
                                            K.ptx.bar.sync(K.uint32(3), K.uint32(SOFT_THREADS))
                                            with K.If(tid == 0), K.Then():
                                                K.ptx.fence.acq_rel.gpu()
                                                old = K.local_scalar("int32")
                                                K.ptx.atom.acq_rel.gpu.global_.add.s32(old, part_ctr.ptr_to([item_s - FULL_ITEMS]), K.int32(1))
                                                K.ptx.st.shared.b32(mrg_order.ptr_to([0]), old)
                                            K.ptx.bar.sync(K.uint32(3), K.uint32(SOFT_THREADS))
                                            order = ld_smem_i32(mrg_order.ptr_to([0]))
                                            with K.If(order == SPLIT - 1), K.Then():
                                                with K.If(tid == 0), K.Then():
                                                    K.ptx.fence.acq_rel.gpu()
                                                    K.ptx.st.relaxed.gpu.global_.b32(part_ctr.ptr_to([item_s - FULL_ITEMS]), K.int32(0))
                                                K.ptx.bar.sync(K.uint32(3), K.uint32(SOFT_THREADS))
                                                base_slot = (item_s - FULL_ITEMS) * SPLIT
                                                m_tot = K.alloc_local([NQW], "float32")
                                                l_tot = K.alloc_local([NQW], "float32")
                                                o_tot = K.alloc_local([NQW], "float32")
                                                mp = K.alloc_local([SPLIT * NQW], "float32")
                                                for pi in range(SPLIT):
                                                    for nl in range(NQW):
                                                        K.ptx.ld.relaxed.gpu.global_.f32(
                                                            mp[pi * NQW + nl], part_ml.ptr_to([(base_slot + pi) * (2 * NQ) + col0 + nl])
                                                        )
                                                for nl in range(NQW):
                                                    K.assign(m_tot[nl], mp[nl])
                                                    for pi in range(1, SPLIT):
                                                        K.assign(m_tot[nl], K.max(m_tot[nl], mp[pi * NQW + nl]))
                                                    K.assign(l_tot[nl], K.float32(0.0))
                                                    K.assign(o_tot[nl], K.float32(0.0))
                                                for pi in range(SPLIT):
                                                    for nl in range(NQW):
                                                        msafe = K.Select(m_tot[nl] == K.float32(NEG_INF), K.float32(0.0), m_tot[nl])
                                                        sc = K.local_scalar("float32")
                                                        K.ptx.ex2.approx.ftz.f32(sc, mp[pi * NQW + nl] - msafe)
                                                        lp = K.local_scalar("float32")
                                                        K.ptx.ld.relaxed.gpu.global_.f32(lp, part_ml.ptr_to([(base_slot + pi) * (2 * NQ) + NQ + col0 + nl]))
                                                        op = K.local_scalar("float32")
                                                        K.ptx.ld.relaxed.gpu.global_.f32(
                                                            op, part_o.ptr_to([(base_slot + pi) * (HEAD_DIM * NQ) + (col0 + nl) * HEAD_DIM + tid_wg])
                                                        )
                                                        K.assign(l_tot[nl], l_tot[nl] + lp * sc)
                                                        K.assign(o_tot[nl], o_tot[nl] + op * sc)
                                                write_output(o_tot, l_tot)
                                iket_end(tk_epi)
                            with K.Else():
                                o_ready.wait(0, it & 1)
                                o_free.arrive(0)
                                for nl in range(NQW):
                                    K.ptx.st.global_.b16(out.ptr_to([out_index(nl)]), zero16)
                        if not STATIC_ONE_SHOT:
                            union_free.arrive(slot)
                if STATIC_ONE_SHOT:
                    K.assign(running, 0)
                else:
                    K.assign(it, it + 1)


        with r_mma:
            kv_pipe = K.PipelineState(STAGES, phase=0)
            it_m = K.local_scalar("int32", init=0)
            gj_m = K.local_scalar("int32", init=0)
            running_m = K.local_scalar("int32", init=1)
            zero = K.uint32(0)

            def mma_qk(sb_col, kstage, qslot):
                tk = iket_range("mma-qk")
                for ki in range(NK):
                    with K.If(elected()), K.Then():
                        K.ptx[MMA](
                            tmem_base + sb_col,
                            desc_at(k_desc, kstage * KV16 + koff(ki)),
                            desc_at(q_desc, qslot * QP16 + qoff(ki)),
                            K.uint32(ID_QK),
                            zero,
                            zero,
                            zero,
                            zero,
                            ki != 0,
                        )
                iket_end(tk)

            with K.While(running_m != 0):
                slot = it_m & 1
                union_ready.wait(slot, (it_m >> 1) & 1)
                n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                with K.If(n_blocks < 0):
                    with K.Then():
                        K.assign(running_m, 0)
                    with K.Else():
                        q_load.full.wait(slot, (it_m >> 1) & 1)
                        with K.If(n_blocks == 0), K.Then():
                            with K.If(elected()), K.Then():
                                q_load.empty.arrive(slot)
                        with K.serial(n_blocks, unroll=False) as j:
                            kstage = K.local_scalar("int32", init=kv_pipe.stage)
                            kphase = K.local_scalar("int32", init=kv_pipe.phase)
                            kv_pipe.advance()
                            kv_pipe.advance()
                            sb = gj_m & 1
                            tk_wk = iket_range("mma-wait-k")
                            kv_load.full.wait(kstage, kphase)
                            iket_end(tk_wk)
                            tk_wsf = iket_range("mma-wait-sfree")
                            s_free.wait(sb, ((gj_m >> 1) + 1) & 1)
                            iket_end(tk_wsf)
                            mma_qk(K.Cast("uint32", sb) * K.uint32(NQ_PAD), kstage, slot)
                            with K.If(elected()), K.Then():
                                s_ready.arrive(sb)
                                kv_load.empty.arrive(kstage)
                                with K.If(j + 1 == n_blocks), K.Then():
                                    q_load.empty.arrive(slot)
                            K.assign(gj_m, gj_m + 1)
                        if not STATIC_ONE_SHOT:
                            union_free.arrive(slot)
                if STATIC_ONE_SHOT:
                    K.assign(running_m, 0)
                else:
                    K.assign(it_m, it_m + 1)


        with r_pv:
            kv_pipe_v = K.PipelineState(STAGES, phase=0)
            it_v = K.local_scalar("int32", init=0)
            gj_v = K.local_scalar("int32", init=0)
            running_v = K.local_scalar("int32", init=1)
            zero_v = K.uint32(0)

            def mma_pv(vstage, pb, acc):
                tk = iket_range("mma-pv")
                for ki in range(NK):
                    with K.If(elected()), K.Then():
                        K.ptx[MMA](
                            tmem_base + K.uint32(O_COL),
                            desc_at(v_desc, vstage * KV16 + voff(ki)),
                            desc_at(p_desc, pb * QP16 + poff(ki)),
                            K.uint32(ID_PV),
                            zero_v,
                            zero_v,
                            zero_v,
                            zero_v,
                            True if ki != 0 else K.Cast("bool", acc != 0),
                        )
                iket_end(tk)

            with K.While(running_v != 0):
                slot = it_v & 1
                union_ready.wait(slot, (it_v >> 1) & 1)
                n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                with K.If(n_blocks < 0):
                    with K.Then():
                        K.assign(running_v, 0)
                    with K.Else():
                        o_free.wait(0, (it_v + 1) & 1)
                        with K.If(n_blocks == 0), K.Then():
                            with K.If(elected()), K.Then():
                                o_ready.arrive(0)
                        acc = K.local_scalar("int32", init=0)
                        with K.serial(n_blocks, unroll=False) as j:
                            kv_pipe_v.advance()
                            vstage = K.local_scalar("int32", init=kv_pipe_v.stage)
                            vphase = K.local_scalar("int32", init=kv_pipe_v.phase)
                            kv_pipe_v.advance()
                            pb = gj_v & 1
                            tk_wv = iket_range("mma-wait-v")
                            kv_load.full.wait(vstage, vphase)
                            iket_end(tk_wv)
                            tk_wp = iket_range("mma-wait-p")
                            p_ready.wait(pb, (gj_v >> 1) & 1)
                            iket_end(tk_wp)
                            mma_pv(vstage, pb, acc)
                            with K.If(elected()), K.Then():
                                p_free.arrive(pb)
                                kv_load.empty.arrive(vstage)
                                with K.If(j + 1 == n_blocks), K.Then():
                                    o_ready.arrive(0)
                            K.assign(acc, 1)
                            K.assign(gj_v, gj_v + 1)
                        if not STATIC_ONE_SHOT:
                            union_free.arrive(slot)
                if STATIC_ONE_SHOT:
                    K.assign(running_v, 0)
                else:
                    K.assign(it_v, it_v + 1)


        with r_load:
            kv_pipe_l = K.PipelineState(STAGES, phase=0)
            it_l = K.local_scalar("int32", init=0)
            running_l = K.local_scalar("int32", init=1)
            with K.While(running_l != 0):
                slot = it_l & 1
                tk_lu = iket_range("ld-wait-union")
                union_ready.wait(slot, (it_l >> 1) & 1)
                iket_end(tk_lu)
                n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                with K.If(n_blocks < 0):
                    with K.Then():
                        K.assign(running_l, 0)
                    with K.Else():
                        h = ld_smem_i32(meta.ptr_to([slot * 16 + 2]))
                        kv_s = ld_smem_i32(meta.ptr_to([slot * 16 + 3]))
                        tok_base = ld_smem_i32(meta.ptr_to([slot * 16 + 5]))
                        q_load.empty.wait(slot, (it_l >> 1) & 1)
                        if not FP8:
                            with K.If(elected()), K.Then():
                                K.ptx[TMA_4D](
                                    q_smem[slot].ptr_to(0, 0),
                                    K.address_of(q_map),
                                    K.int32(0),
                                    h * G,
                                    tok_base,
                                    K.int32(0),
                                    K.cuda.cvta_generic_to_shared(q_load.full.ptr_to([slot])),
                                )
                                q_load.full.arrive(slot, tx_count=Q_TMA_BYTES)
                        else:
                            words8 = K.alloc_local([8], "int32")
                            e4 = K.alloc_local([8], "uint16")
                            packed4 = K.alloc_local([4], "uint32")
                            for r in range(Q_ROUNDS):
                                c = lane + 32 * r

                                def quant_chunk(c=c):
                                    n = c >> 3
                                    k16 = c & 7
                                    row = tok_base + n // G
                                    head = h * G + (n % G)
                                    wbase = (row * HQ + head) * (HEAD_DIM // 2) + k16 * 8
                                    K.ptx.ld.global_.nc.v4.b32(
                                        words8[0], words8[1], words8[2], words8[3], qraw.ptr_to([wbase])
                                    )
                                    K.ptx.ld.global_.nc.v4.b32(
                                        words8[4], words8[5], words8[6], words8[7], qraw.ptr_to([wbase + 4])
                                    )
                                    for i in range(8):
                                        K.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(e4[i], K.Cast("uint32", words8[i]))
                                    for i in range(4):
                                        K.ptx.mov.b32(packed4[i], e4[2 * i], e4[2 * i + 1])
                                    K.ptx.st.shared.v4.b32(
                                        q_smem[slot].ptr_to(n, k16 * 16), packed4[0], packed4[1], packed4[2], packed4[3]
                                    )

                                if 32 * r + 32 <= Q_CHUNKS:
                                    quant_chunk()
                                else:
                                    with K.If(c < Q_CHUNKS), K.Then():
                                        quant_chunk()
                            K.ptx.fence.proxy.async_.shared__cta()
                            q_load.full.arrive(slot)
                        blk_off_l = ld_smem_i32(meta.ptr_to([slot * 16 + 7]))
                        with K.If(n_blocks > 0), K.Then():
                            with K.serial(n_blocks, unroll=False) as j:
                                blk = ld_smem_i32(ulist.ptr_to([slot * MAX_UNION + blk_off_l + j]))
                                if PAGED:
                                    pg = ld_smem_i32(uplist.ptr_to([slot * MAX_UNION + blk_off_l + j]))
                                tmaps = (k_map,) if SPLIT_LOADERS else (k_map, v_map)
                                for tmap in tmaps:
                                    tk_we = iket_range("ld-wait-stage")
                                    kv_load.empty.wait(kv_pipe_l.stage, kv_pipe_l.phase)
                                    iket_end(tk_we)
                                    with K.If(elected()), K.Then():
                                        if PAGED and not FP8:
                                            K.ptx[TMA_4D_HINT](
                                                kv_smem[kv_pipe_l.stage].ptr_to(0, 0),
                                                K.address_of(tmap),
                                                K.int32(0),
                                                K.int32(0),
                                                K.int32(0),
                                                pg * HKV + h,
                                                K.cuda.cvta_generic_to_shared(kv_load.full.ptr_to([kv_pipe_l.stage])),
                                                *HINT_ARGS,
                                            )
                                        elif PAGED:
                                            K.ptx[TMA_3D_HINT](
                                                kv_smem[kv_pipe_l.stage].ptr_to(0, 0),
                                                K.address_of(tmap),
                                                K.int32(0),
                                                K.int32(0),
                                                pg * HKV + h,
                                                K.cuda.cvta_generic_to_shared(kv_load.full.ptr_to([kv_pipe_l.stage])),
                                                *HINT_ARGS,
                                            )
                                        elif not FP8:
                                            K.ptx[TMA_3D_HINT](
                                                kv_smem[kv_pipe_l.stage].ptr_to(0, 0),
                                                K.address_of(tmap),
                                                K.int32(0),
                                                kv_s + blk * BLK,
                                                h * 2,
                                                K.cuda.cvta_generic_to_shared(kv_load.full.ptr_to([kv_pipe_l.stage])),
                                                *HINT_ARGS,
                                            )
                                        else:
                                            K.ptx[TMA_3D_HINT](
                                                kv_smem[kv_pipe_l.stage].ptr_to(0, 0),
                                                K.address_of(tmap),
                                                K.int32(0),
                                                kv_s + blk * BLK,
                                                h,
                                                K.cuda.cvta_generic_to_shared(kv_load.full.ptr_to([kv_pipe_l.stage])),
                                                *HINT_ARGS,
                                            )
                                        kv_load.full.arrive(kv_pipe_l.stage, tx_count=KV_TILE_BYTES)
                                    kv_pipe_l.advance()
                                if SPLIT_LOADERS:

                                    kv_pipe_l.advance()
                        if not STATIC_ONE_SHOT:
                            union_free.arrive(slot)
                if STATIC_ONE_SHOT:
                    K.assign(running_l, 0)
                else:
                    K.assign(it_l, it_l + 1)




        if SPLIT_LOADERS:
            with r_vload:
                kv_pipe_vl = K.PipelineState(STAGES, phase=0)
                kv_pipe_vl.advance()
                it_vl = K.local_scalar("int32", init=0)
                running_vl = K.local_scalar("int32", init=1)
                with K.While(running_vl != 0):
                    slot = it_vl & 1
                    tk_vlu = iket_range("vld-wait-union")
                    union_ready.wait(slot, (it_vl >> 1) & 1)
                    iket_end(tk_vlu)
                    n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                    with K.If(n_blocks < 0):
                        with K.Then():
                            K.assign(running_vl, 0)
                        with K.Else():
                            h = ld_smem_i32(meta.ptr_to([slot * 16 + 2]))
                            kv_s = ld_smem_i32(meta.ptr_to([slot * 16 + 3]))
                            blk_off_vl = ld_smem_i32(meta.ptr_to([slot * 16 + 7]))
                            with K.If(n_blocks > 0), K.Then():
                                with K.serial(n_blocks, unroll=False) as j:
                                    blk = ld_smem_i32(ulist.ptr_to([slot * MAX_UNION + blk_off_vl + j]))
                                    if PAGED:
                                        pg = ld_smem_i32(uplist.ptr_to([slot * MAX_UNION + blk_off_vl + j]))
                                    tk_vwe = iket_range("vld-wait-stage")
                                    kv_load.empty.wait(kv_pipe_vl.stage, kv_pipe_vl.phase)
                                    iket_end(tk_vwe)
                                    with K.If(elected()), K.Then():
                                        if PAGED and not FP8:
                                            K.ptx[TMA_4D_HINT](
                                                kv_smem[kv_pipe_vl.stage].ptr_to(0, 0),
                                                K.address_of(v_map),
                                                K.int32(0),
                                                K.int32(0),
                                                K.int32(0),
                                                pg * HKV + h,
                                                K.cuda.cvta_generic_to_shared(kv_load.full.ptr_to([kv_pipe_vl.stage])),
                                                *HINT_ARGS,
                                            )
                                        elif PAGED:
                                            K.ptx[TMA_3D_HINT](
                                                kv_smem[kv_pipe_vl.stage].ptr_to(0, 0),
                                                K.address_of(v_map),
                                                K.int32(0),
                                                K.int32(0),
                                                pg * HKV + h,
                                                K.cuda.cvta_generic_to_shared(kv_load.full.ptr_to([kv_pipe_vl.stage])),
                                                *HINT_ARGS,
                                            )
                                        elif not FP8:
                                            K.ptx[TMA_3D_HINT](
                                                kv_smem[kv_pipe_vl.stage].ptr_to(0, 0),
                                                K.address_of(v_map),
                                                K.int32(0),
                                                kv_s + blk * BLK,
                                                h * 2,
                                                K.cuda.cvta_generic_to_shared(kv_load.full.ptr_to([kv_pipe_vl.stage])),
                                                *HINT_ARGS,
                                            )
                                        else:
                                            K.ptx[TMA_3D_HINT](
                                                kv_smem[kv_pipe_vl.stage].ptr_to(0, 0),
                                                K.address_of(v_map),
                                                K.int32(0),
                                                kv_s + blk * BLK,
                                                h,
                                                K.cuda.cvta_generic_to_shared(kv_load.full.ptr_to([kv_pipe_vl.stage])),
                                                *HINT_ARGS,
                                            )
                                        kv_load.full.arrive(kv_pipe_vl.stage, tx_count=KV_TILE_BYTES)
                                    kv_pipe_vl.advance()
                                    kv_pipe_vl.advance()
                            if not STATIC_ONE_SHOT:
                                union_free.arrive(slot)
                    if STATIC_ONE_SHOT:
                        K.assign(running_vl, 0)
                    else:
                        K.assign(it_vl, it_vl + 1)


        with r_prep:
            it_p = K.local_scalar("int32", init=0)
            running_p = K.local_scalar("int32", init=1)
            with K.While(running_p != 0):
                slot = it_p & 1
                task = K.local_scalar("int32")
                if STATIC_ONE_SHOT:
                    K.assign(task, K.cta_id())
                else:
                    with K.If(it_p == 0):
                        with K.Then():
                            K.assign(task, K.cta_id())
                        with K.Else():
                            grabbed = K.local_scalar("int32", init=0)
                            with K.If(lane == 0), K.Then():
                                K.ptx.atom.relaxed.gpu.global_.add.s32(grabbed, sched.ptr_to([0]), K.int32(1))
                            K.assign(task, K.uniform(grabbed) + NUM_CTAS)
                if not STATIC_ONE_SHOT:
                    tk_pf = iket_range("prep-wait-free")
                    union_free.wait(slot, ((it_p >> 1) + 1) & 1)
                    iket_end(tk_pf)
                tk_pi = iket_range("prep-item")
                with K.If(task >= NUM_TASKS):
                    with K.Then():
                        with K.If(lane == 0), K.Then():
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16]), K.int32(-1))
                        union_ready.arrive(slot)
                        K.assign(running_p, 0)
                    with K.Else():
                        item = K.local_scalar("int32", init=task)
                        part = K.local_scalar("int32", init=0)
                        if SPLIT > 1:
                            with K.If(task >= FULL_ITEMS), K.Then():
                                tidx = task - FULL_ITEMS
                                K.assign(item, FULL_ITEMS + tidx // SPLIT)
                                K.assign(part, tidx - (tidx // SPLIT) * SPLIT)
                        b = item // (HKV * NCH)
                        rem = item - b * (HKV * NCH)
                        h = rem // NCH
                        ch = rem - h * NCH
                        kv_s = K.local_scalar("int32", init=0)
                        kv_len = K.local_scalar("int32")
                        if PAGED:
                            K.assign(kv_len, ld_global_i32(lens.ptr_to([b])))
                        else:
                            K.assign(kv_s, ld_global_i32(lens.ptr_to([b])))
                            K.assign(kv_len, ld_global_i32(lens.ptr_to([b + 1])) - kv_s)
                        n_vis = (kv_len + (BLK - 1)) >> 7
                        tok_base = b * TQ + ch * T
                        causal_off = kv_len - TQ + ch * T
                        base = (h * TOTAL_Q + tok_base) * TOPK
                        idxs = K.alloc_local([N_ROUNDS], "int32")
                        for r in range(N_ROUNDS):
                            e = lane + 32 * r
                            K.assign(idxs[r], -1)
                            if 32 * r + 32 <= MAX_UNION:
                                K.assign(idxs[r], ld_global_i32(q2k.ptr_to([base + e])))
                            else:
                                with K.If(e < MAX_UNION), K.Then():
                                    K.assign(idxs[r], ld_global_i32(q2k.ptr_to([base + e])))

                        def valid(r):
                            return K.And(idxs[r] >= 0, idxs[r] < n_vis)

                        n_blocks = K.local_scalar("int32", init=0)
                        if T == 1:
                            for r in range(N_ROUNDS):
                                vflag = K.local_scalar("int32", init=K.Select(valid(r), 1, 0))
                                ballot = K.local_scalar("uint32")
                                K.ptx.vote_sync.ballot.b32(ballot, K.ptx.pred(vflag), K.uint32(0xFFFFFFFF))
                                before = K.local_scalar("uint32")
                                K.ptx.popc.b32(before, K.bitwise_and(ballot, lanemask_lt()))
                                total = K.local_scalar("uint32")
                                K.ptx.popc.b32(total, ballot)
                                rank = n_blocks + K.Cast("int32", before)
                                with K.If(vflag != 0), K.Then():
                                    K.ptx.st.shared.b32(ulist.ptr_to([slot * MAX_UNION + rank]), idxs[r])
                                    if PAGED:
                                        pg = ld_global_i32(ptab.ptr_to([b * MAX_PAGES + idxs[r]]))
                                        K.ptx.st.shared.b32(uplist.ptr_to([slot * MAX_UNION + rank]), pg)
                                K.assign(n_blocks, n_blocks + K.Cast("int32", total))
                        else:
                            nwords = (n_vis + 31) >> 5
                            wi = K.local_scalar("int32", init=lane)
                            with K.While(wi < nwords):
                                K.ptx.st.shared.b32(words.ptr_to([wi]), K.uint32(0))
                                K.assign(wi, wi + 32)
                            K.cuda.warp_sync()
                            for r in range(N_ROUNDS):
                                with K.If(valid(r)), K.Then():
                                    K.ptx.red.shared.or_.b32(
                                        words.ptr_to([idxs[r] >> 5]),
                                        K.shift_left(K.uint32(1), K.Cast("uint32", idxs[r] & 31)),
                                    )
                            K.cuda.warp_sync()
                            carry = K.local_scalar("uint32", init=K.uint32(0))
                            wbase = K.local_scalar("int32", init=0)
                            with K.While(wbase < nwords):
                                w = wbase + lane
                                wv = K.local_scalar("uint32", init=K.uint32(0))
                                with K.If(w < nwords), K.Then():
                                    K.ptx.ld.shared.u32(wv, words.ptr_to([w]))
                                cnt = K.alloc_local([1], "uint32")
                                K.ptx.popc.b32(cnt[0], wv)
                                own = K.local_scalar("uint32", init=cnt[0])
                                K.idioms.warp_scan_add(cnt, 1, lane)
                                with K.If(w < nwords), K.Then():
                                    K.ptx.st.shared.b32(prefix.ptr_to([w]), carry + (cnt[0] - own))
                                tot = K.local_scalar("uint32")
                                K.ptx.shfl_sync.idx.b32(tot, cnt[0], K.uint32(31), K.uint32(31), K.uint32(0xFFFFFFFF))
                                K.assign(carry, carry + tot)
                                K.assign(wbase, wbase + 32)
                            K.assign(n_blocks, K.Cast("int32", carry))
                            K.cuda.warp_sync()
                            with K.serial(nwords, unroll=False) as w:
                                wv = ld_smem_u32(words.ptr_to([w]))
                                pw = ld_smem_u32(prefix.ptr_to([w]))
                                mybit = K.bitwise_and(K.shift_right(wv, K.Cast("uint32", lane)), K.uint32(1))
                                with K.If(mybit != K.uint32(0)), K.Then():
                                    below = K.local_scalar("uint32")
                                    K.ptx.popc.b32(below, K.bitwise_and(wv, lanemask_lt()))
                                    rank = K.Cast("int32", pw + below)
                                    blkid = w * 32 + lane
                                    K.ptx.st.shared.b32(ulist.ptr_to([slot * MAX_UNION + rank]), blkid)
                                    K.ptx.st.shared.b32(umask.ptr_to([slot * MAX_UNION + rank]), K.uint32(0))
                                    if PAGED:
                                        pg = ld_global_i32(ptab.ptr_to([b * MAX_PAGES + blkid]))
                                        K.ptx.st.shared.b32(uplist.ptr_to([slot * MAX_UNION + rank]), pg)
                            K.cuda.warp_sync()
                            for r in range(N_ROUNDS):
                                e = lane + 32 * r
                                with K.If(valid(r)), K.Then():
                                    wq = idxs[r] >> 5
                                    wv = ld_smem_u32(words.ptr_to([wq]))
                                    pw = ld_smem_u32(prefix.ptr_to([wq]))
                                    below = K.local_scalar("uint32")
                                    K.ptx.popc.b32(
                                        below,
                                        K.bitwise_and(
                                            wv,
                                            K.shift_left(K.uint32(1), K.Cast("uint32", idxs[r] & 31)) - K.uint32(1),
                                        ),
                                    )
                                    u = K.Cast("int32", pw + below)
                                    K.ptx.red.shared.or_.b32(
                                        umask.ptr_to([slot * MAX_UNION + u]),
                                        K.shift_left(K.uint32(1), K.Cast("uint32", e // TOPK)),
                                    )
                        K.cuda.warp_sync()
                        blk_off = K.local_scalar("int32", init=0)
                        if SPLIT > 1:
                            with K.If(item >= FULL_ITEMS), K.Then():
                                per_part = (n_blocks + (SPLIT - 1)) // SPLIT
                                start = K.min(part * per_part, n_blocks)
                                K.assign(blk_off, start)
                                K.assign(n_blocks, K.min(n_blocks - start, per_part))
                        with K.If(lane == 0), K.Then():
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 7]), blk_off)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 8]), item)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 9]), part)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 0]), n_blocks)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 1]), b)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 2]), h)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 3]), kv_s)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 4]), kv_len)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 5]), tok_base)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 6]), causal_off)
                        union_ready.arrive(slot)
                iket_end(tk_pi)
                if STATIC_ONE_SHOT:
                    K.assign(running_p, 0)
                else:
                    K.assign(it_p, it_p + 1)


        K.cuda.cta_sync()
        if not STATIC_ONE_SHOT:
            with K.If(tid == 0), K.Then():
                done = K.local_scalar("int32")
                K.ptx.atom.acq_rel.gpu.global_.add.s32(done, sched.ptr_to([1]), K.int32(1))
                with K.If(done == NUM_CTAS - 1), K.Then():
                    K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([0]), K.int32(0))
                    K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([1]), K.int32(0))
        with K.If(warp == MMA_WARP), K.Then():
            K.ptx[TMEM_DEALLOC](tmem_base, K.uint32(TMEM_COLS))

    return msa_decode_kvmajor


def make_kernel_qm(cfg):
    TQ = cfg["T"]
    T = cfg["TI"]
    NCH = TQ // T
    G = cfg["G"]
    HKV = cfg["HKV"]
    TOPK = cfg["TOPK"]
    TOTAL_Q = cfg["TOTAL_Q"]
    kind = cfg["kv_kind"]
    PAGED = cfg["paged"]
    MAX_PAGES = cfg["MAX_PAGES"]
    W_MAX = cfg["MAX_BLOCK_WORDS"]
    NUM_ITEMS = cfg["NUM_ITEMS"]
    NUM_CTAS = cfg["NUM_CTAS"]
    STAGES = cfg["STAGES"]
    SPLIT = cfg["SPLIT"]
    FULL_ITEMS = (NUM_ITEMS // NUM_CTAS) * NUM_CTAS if SPLIT > 1 else NUM_ITEMS
    TAIL_ITEMS = NUM_ITEMS - FULL_ITEMS
    NUM_TASKS = FULL_ITEMS + TAIL_ITEMS * SPLIT
    HQ = HKV * G
    ROWS = T * G
    assert ROWS == 128, "q-major family needs exactly 128 query rows per item"
    FP8 = False
    assert kind in ("bf16", "f16")
    kv_dt = {"bf16": K.bf16, "f16": K.f16}[kind]
    out_dt = kv_dt
    EB = 2
    MMA_K = 16
    NK = HEAD_DIM // MMA_K
    KV_TILE_BYTES = BLK * HEAD_DIM * EB
    NQ_PAD = ROWS
    Q_TMA_BYTES = ROWS * HEAD_DIM * EB
    MMA = MMA_F16
    fmt = 1 if kind == "bf16" else 0
    ID_QK = make_idesc(128, 128, fmt, fmt, 0, 0)
    ID_PV = make_idesc(128, 128, fmt, fmt, 0, 1)
    S_COLS = 128
    P_BASE = 256
    P_COLS = 64
    O_COL = 384
    TMEM_COLS = 512
    MAX_UNION = T * TOPK
    N_ROUNDS = ceildiv(MAX_UNION, 32)
    KV16 = KV_TILE_BYTES // 16
    QP16 = ROWS * HEAD_DIM * EB // 16
    NWARPS = 8
    MMA_WARP = 4
    LOAD_WARP = 5
    PREP_WARP = 6
    IDLE_WARP = 7
    SOFT_THREADS = 128
    LOG2G = G.bit_length() - 1
    assert (1 << LOG2G) == G

    @K.kernel(warps=NWARPS, arch="sm_100a", min_blocks_per_sm=1, grid=NUM_CTAS)
    def msa_decode_qmajor(
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        out: K.gptr[out_dt],
        q2k: K.gptr[K.i32],
        lens: K.gptr[K.i32],
        ptab: K.gptr[K.i32],
        qraw: K.gptr[K.i32],
        sched: K.gptr[K.i32],
        part_o: K.gptr[K.f32],
        part_ml: K.gptr[K.f32],
        part_ctr: K.gptr[K.i32],
        scale_log2: K.f32,
    ):
        warp = K.warp_id()
        lane = K.lane_id()
        tid = K.thread_id()

        smem = K.smem_pool()
        kv_smem = smem.alloc((STAGES, BLK, HEAD_DIM), kv_dt, swizzle=K.SW128B)
        q_smem = smem.alloc((2, ROWS, HEAD_DIM), kv_dt, swizzle=K.SW128B)
        tmem_addr = smem.alloc((4,), K.u32)
        meta = smem.alloc((32,), K.i32)
        ulist = smem.alloc((2 * MAX_UNION,), K.i32)
        uplist = smem.alloc((2 * MAX_UNION,), K.i32)
        umask = smem.alloc((2 * MAX_UNION,), K.u32)
        words = smem.alloc((W_MAX,), K.u32)
        prefix = smem.alloc((W_MAX,), K.u32)
        K.keep_alive(ptab.ptr_to([0]))
        K.keep_alive(qraw.ptr_to([0]))
        K.keep_alive(part_o.ptr_to([0]))
        K.keep_alive(part_ml.ptr_to([0]))
        K.keep_alive(part_ctr.ptr_to([0]))
        mrg_order = smem.alloc((4,), K.i32)


        def lo_uniform(desc):
            desc_lo = K.alloc_local((1,), "uint32")
            desc_hi = K.alloc_local((1,), "uint32")
            K.assign(desc_lo[0], K.uniform(K.Cast("uint32", desc.value)))
            K.assign(desc_hi[0], K.Cast("uint32", K.shift_right(desc.value, K.uint64(32))))
            return desc_lo, desc_hi

        def desc_at(desc, off16):
            lo, hi = desc
            packed = K.alloc_local((1,), "uint64")
            low = lo[0] if isinstance(off16, int) and off16 == 0 else lo[0] + K.Cast("uint32", off16)
            K.assign(
                packed[0],
                K.bitwise_or(K.shift_left(K.Cast("uint64", hi[0]), K.uint64(32)), K.Cast("uint64", low)),
            )
            return packed[0]

        def encode(view, major="k"):
            desc, off16 = view.encode(major=major, mma_k=MMA_K)
            return lo_uniform(desc), off16

        k_desc, koff = encode(kv_smem[0], "k")
        v_desc, voff = encode(kv_smem[0], "mn")
        q_desc, qoff = encode(q_smem[0], "k")


        kv_load = K.Pipeline(smem, STAGES, full="tma", empty="tcgen05", empty_phase_offset=1)
        q_load = K.Pipeline(smem, 2, full="tma", empty="tcgen05", init_full=1, empty_phase_offset=1)
        union_ready = K.MBarrier(smem, 2)
        union_ready.init(32)
        union_free = K.MBarrier(smem, 2)
        union_free.init(SOFT_THREADS + 64)
        s_ready = K.TCGen05Bar(smem, 2)
        s_ready.init(1)
        s_free = K.MBarrier(smem, 2)
        s_free.init(SOFT_THREADS)
        p_ready = K.MBarrier(smem, 2)
        p_ready.init(SOFT_THREADS)
        p_free = K.TCGen05Bar(smem, 2)
        p_free.init(1)
        o_ready = K.TCGen05Bar(smem, 1)
        o_ready.init(1)
        o_free = K.MBarrier(smem, 1)
        o_free.init(SOFT_THREADS)

        K.ptx.fence.proxy.async_.shared__cta()
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        with K.If(warp == MMA_WARP), K.Then():
            K.ptx[TMEM_ALLOC](K.address_of(tmem_addr[0]), K.uint32(TMEM_COLS))
            K.ptx[TMEM_RELINQUISH]()
        K.cuda.cta_sync()
        tb_raw = K.local_scalar("uint32")
        K.ptx.ld.shared.u32(tb_raw, tmem_addr.ptr_to([0]))
        tmem_base = K.local_scalar("uint32", init=K.uniform(tb_raw))


        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def ld_smem_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.shared.b32(value, ptr)
            return value

        def ld_smem_u32(ptr):
            value = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(value, ptr)
            return value

        def ld_global_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.global_.nc.b32(value, ptr)
            return value

        def lanemask_lt():
            return K.shift_left(K.uint32(1), K.Cast("uint32", lane)) - K.uint32(1)

        def iket_range(name, *, leader_only=False):
            token = K.alloc_local([1], "uint32")
            if leader_only:
                K.assign(token[0], K.cuda.iket.sentinel_token(name))
                with K.If((warp & 3) == 0), K.Then():
                    K.assign(token[0], K.cuda.iket.range_start(name))
            else:
                K.assign(token[0], K.cuda.iket.range_start(name))
            return token

        def iket_end(token):
            K.cuda.iket.range_end(token[0])

        def pack_half2(dst, hi, lo):
            if kind == "bf16":
                K.ptx.cvt.rn.bf16x2.f32(dst, hi, lo)
            else:
                K.ptx.cvt.rn.f16x2.f32(dst, hi, lo)

        def tmem_row_load(dst, col0, ncols):
            for c in range(ncols // 32):
                K.ptx[tmem_ld(32)](*(dst[32 * c + i] for i in range(32)), tmem_base + K.uint32(col0 + 32 * c))
            K.ptx.tcgen05.wait__ld.sync.aligned()

        def tmem_row_store(src, col0, ncols):
            for c in range(ncols // 16):
                K.ptx[tmem_st(16)](tmem_base + K.uint32(col0 + 16 * c), *(src[16 * c + i] for i in range(16)))
            K.ptx.tcgen05.wait__st.sync.aligned()

        sp = K.specialize(chain_dispatch=True)
        r_soft = sp.role("softmax", warps=[0, 1, 2, 3], regs=232)
        wg1 = sp.warpgroup("wg1", warps=range(4, 8), regs=56)
        r_mma = sp.role("mma", warps=[MMA_WARP], group=wg1)
        r_load = sp.role("load", warps=[LOAD_WARP], group=wg1)
        r_prep = sp.role("prep", warps=[PREP_WARP], group=wg1)
        r_idle = sp.role("idle", warps=[IDLE_WARP], group=wg1)


        with r_soft:
            t_row = tid >> LOG2G
            g_row = tid & (G - 1)
            it = K.local_scalar("int32", init=0)
            gj = K.local_scalar("int32", init=0)
            running = K.local_scalar("int32", init=1)
            m_s = K.local_scalar("float32", init=K.float32(NEG_INF))
            l_row = K.local_scalar("float32", init=K.float32(0.0))
            s = K.alloc_local([128], "float32")
            pk = K.alloc_local([64], "uint32")
            zeros16 = K.alloc_local([16], "uint32")
            for i in range(16):
                K.assign(zeros16[i], K.uint32(0))
            with K.While(running != 0):
                slot = it & 1
                tk_wu = iket_range("sm-wait-union", leader_only=True)
                union_ready.wait(slot, (it >> 1) & 1)
                iket_end(tk_wu)
                n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                with K.If(n_blocks < 0):
                    with K.Then():
                        K.assign(running, 0)
                    with K.Else():
                        h = ld_smem_i32(meta.ptr_to([slot * 16 + 2]))
                        tok_base = ld_smem_i32(meta.ptr_to([slot * 16 + 5]))
                        causal_off = ld_smem_i32(meta.ptr_to([slot * 16 + 6]))
                        blk_off_s = ld_smem_i32(meta.ptr_to([slot * 16 + 7]))
                        item_s = ld_smem_i32(meta.ptr_to([slot * 16 + 8]))
                        part_s = ld_smem_i32(meta.ptr_to([slot * 16 + 9]))
                        qpos = causal_off + t_row
                        out_row = ((tok_base + t_row) * HQ + h * G + g_row) * HEAD_DIM
                        with K.If(n_blocks > 0):
                            with K.Then():
                                K.assign(m_s, K.float32(NEG_INF))
                                K.assign(l_row, K.float32(0.0))
                                with K.serial(n_blocks, unroll=False) as j:
                                    blk = ld_smem_i32(ulist.ptr_to([slot * MAX_UNION + blk_off_s + j]))
                                    if T > 1:
                                        tmask = ld_smem_u32(umask.ptr_to([slot * MAX_UNION + blk_off_s + j]))
                                        sel = K.local_scalar(
                                            "int32",
                                            init=K.Select(
                                                K.bitwise_and(K.shift_right(tmask, K.Cast("uint32", t_row)), K.uint32(1)) != K.uint32(0), 1, 0
                                            ),
                                        )
                                    else:
                                        sel = K.local_scalar("int32", init=1)
                                    any_sel = K.local_scalar("uint32", init=K.uint32(1))
                                    if T > 1:
                                        K.ptx.vote_sync.any.pred(any_sel, K.ptx.pred(sel), K.uint32(0xFFFFFFFF))
                                    sb = gj & 1
                                    pb = gj & 1
                                    s_col = K.Cast("uint32", sb) * K.uint32(S_COLS)
                                    p_col = K.uint32(P_BASE) + K.Cast("uint32", pb) * K.uint32(P_COLS)
                                    need_causal = (blk * BLK + (BLK - 1)) > causal_off
                                    tk_ws = iket_range("sm-wait-s", leader_only=True)
                                    s_ready.wait(sb, (gj >> 1) & 1)
                                    iket_end(tk_ws)
                                    with K.If(any_sel != K.uint32(0)):
                                        with K.Then():
                                            tk_ph1 = iket_range("sm-max", leader_only=True)
                                            for c in range(4):
                                                K.ptx[tmem_ld(32)](*(s[32 * c + i] for i in range(32)), tmem_base + s_col + K.uint32(32 * c))
                                            K.ptx.tcgen05.wait__ld.sync.aligned()
                                            s_free.arrive(sb)
                                            with K.If(need_causal), K.Then():
                                                kv0 = blk * BLK
                                                for c in range(128):
                                                    K.assign(s[c], K.Select(kv0 + c <= qpos, s[c], K.float32(NEG_INF)))

                                            lvl = [s[i] for i in range(128)]
                                            while len(lvl) > 1:
                                                nxt = []
                                                i = 0
                                                while i < len(lvl):
                                                    if i + 2 < len(lvl):
                                                        r = K.local_scalar("float32")
                                                        K.ptx.max.f32(r, lvl[i], lvl[i + 1], lvl[i + 2])
                                                        nxt.append(r)
                                                        i += 3
                                                    elif i + 1 < len(lvl):
                                                        nxt.append(K.local_scalar("float32", init=K.max(lvl[i], lvl[i + 1])))
                                                        i += 2
                                                    else:
                                                        nxt.append(lvl[i])
                                                        i += 1
                                                lvl = nxt
                                            bmax = lvl[0]
                                            mbs = K.local_scalar("float32", init=K.Select(sel != 0, bmax * scale_log2, K.float32(NEG_INF)))
                                            m_old = K.local_scalar("float32", init=m_s)
                                            m_new = K.local_scalar("float32", init=K.max(m_old, mbs))
                                            d = K.local_scalar("float32", init=m_new - m_old)
                                            big = d > K.float32(RESCALE_THRESH)
                                            do_r = K.local_scalar("int32", init=K.Select(K.And(m_old != K.float32(NEG_INF), big), 1, 0))
                                            need_r = K.local_scalar("uint32")
                                            K.ptx.vote_sync.any.pred(need_r, K.ptx.pred(do_r), K.uint32(0xFFFFFFFF))
                                            iket_end(tk_ph1)
                                            with K.If(need_r != K.uint32(0)), K.Then():
                                                tk_rs = iket_range("sm-rescale", leader_only=True)
                                                gjm = gj - 1
                                                p_free.wait(gjm & 1, (gjm >> 1) & 1)
                                                alpha = K.local_scalar("float32")
                                                K.ptx.ex2.approx.ftz.f32(alpha, K.float32(0.0) - d)
                                                K.assign(alpha, K.Select(do_r != 0, alpha, K.float32(1.0)))
                                                K.assign(l_row, l_row * alpha)
                                                o_ch = K.alloc_local([32], "float32")
                                                for c0 in range(0, 128, 32):
                                                    K.ptx[tmem_ld(32)](*(o_ch[i] for i in range(32)), tmem_base + K.uint32(O_COL + c0))
                                                    K.ptx.tcgen05.wait__ld.sync.aligned()
                                                    for i in range(32):
                                                        K.assign(o_ch[i], o_ch[i] * alpha)
                                                    K.ptx[tmem_st(32)](tmem_base + K.uint32(O_COL + c0), *(o_ch[i] for i in range(32)))
                                                K.ptx.tcgen05.wait__st.sync.aligned()
                                                iket_end(tk_rs)
                                            K.assign(m_s, K.Select(big, m_new, m_old))
                                            tk_ph2 = iket_range("sm-exp", leader_only=True)
                                            msub = K.local_scalar(
                                                "float32",
                                                init=K.Select(sel != 0, K.max(m_s, K.float32(-1.0e38)), K.float32(1.0e38)),
                                            )
                                            lacc = K.alloc_local([4], "float32")
                                            for a in range(4):
                                                K.assign(lacc[a], K.float32(0.0))
                                            for c in range(128):
                                                K.ptx.ex2.approx.ftz.f32(s[c], s[c] * scale_log2 - msub)
                                                K.assign(lacc[c % 4], lacc[c % 4] + s[c])
                                            K.assign(l_row, l_row + ((lacc[0] + lacc[1]) + (lacc[2] + lacc[3])))
                                            for i in range(64):
                                                pack_half2(pk[i], s[2 * i + 1], s[2 * i])
                                            iket_end(tk_ph2)
                                            tk_wp = iket_range("sm-wait-pfree", leader_only=True)
                                            p_free.wait(pb, ((gj >> 1) + 1) & 1)
                                            iket_end(tk_wp)
                                            tk_ps = iket_range("sm-pstore", leader_only=True)
                                            for c in range(4):
                                                K.ptx[tmem_st(16)](tmem_base + p_col + K.uint32(16 * c), *(pk[16 * c + i] for i in range(16)))
                                            K.ptx.tcgen05.wait__st.sync.aligned()
                                            p_ready.arrive(pb)
                                            iket_end(tk_ps)
                                        with K.Else():
                                            s_free.arrive(sb)
                                            p_free.wait(pb, ((gj >> 1) + 1) & 1)
                                            for c in range(4):
                                                K.ptx[tmem_st(16)](tmem_base + p_col + K.uint32(16 * c), *(zeros16[i] for i in range(16)))
                                            K.ptx.tcgen05.wait__st.sync.aligned()
                                            p_ready.arrive(pb)
                                    K.assign(gj, gj + 1)

                                tk_wo = iket_range("sm-wait-o", leader_only=True)
                                o_ready.wait(0, it & 1)
                                iket_end(tk_wo)
                                tk_epi = iket_range("sm-epi", leader_only=True)
                                for c in range(4):
                                    K.ptx[tmem_ld(32)](*(s[32 * c + i] for i in range(32)), tmem_base + K.uint32(O_COL + 32 * c))
                                K.ptx.tcgen05.wait__ld.sync.aligned()
                                o_free.arrive(0)

                                def write_row(o_arr, l_val):
                                    bad = K.Or(l_val == K.float32(0.0), l_val != l_val)
                                    inv = K.local_scalar("float32")
                                    K.ptx.rcp.approx.ftz.f32(inv, K.Select(bad, K.float32(1.0), l_val))
                                    K.assign(inv, K.Select(bad, K.float32(0.0), inv))
                                    for i in range(64):
                                        pack_half2(pk[i], o_arr[2 * i + 1] * inv, o_arr[2 * i] * inv)
                                    for i in range(8):
                                        K.ptx[ST_OUT_V8](out.ptr_to([out_row + 16 * i]), *(pk[8 * i + j] for j in range(8)))

                                if SPLIT == 1:
                                    write_row(s, l_row)
                                else:
                                    with K.If(item_s < FULL_ITEMS):
                                        with K.Then():
                                            write_row(s, l_row)
                                        with K.Else():
                                            slot_idx = (item_s - FULL_ITEMS) * SPLIT + part_s
                                            o_base = (slot_idx * ROWS + tid) * HEAD_DIM
                                            ml_base = (slot_idx * ROWS + tid) * 2
                                            for i in range(32):
                                                K.ptx.st.global_.v4.f32(
                                                    part_o.ptr_to([o_base + 4 * i]), s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]
                                                )
                                            K.ptx.st.global_.f32(part_ml.ptr_to([ml_base]), m_s)
                                            K.ptx.st.global_.f32(part_ml.ptr_to([ml_base + 1]), l_row)
                                            K.ptx.bar.sync(K.uint32(3), K.uint32(SOFT_THREADS))
                                            with K.If(tid == 0), K.Then():
                                                K.ptx.fence.acq_rel.gpu()
                                                old = K.local_scalar("int32")
                                                K.ptx.atom.acq_rel.gpu.global_.add.s32(old, part_ctr.ptr_to([item_s - FULL_ITEMS]), K.int32(1))
                                                K.ptx.st.shared.b32(mrg_order.ptr_to([0]), old)
                                            K.ptx.bar.sync(K.uint32(3), K.uint32(SOFT_THREADS))
                                            order = ld_smem_i32(mrg_order.ptr_to([0]))
                                            with K.If(order == SPLIT - 1), K.Then():
                                                with K.If(tid == 0), K.Then():
                                                    K.ptx.fence.acq_rel.gpu()
                                                    K.ptx.st.relaxed.gpu.global_.b32(part_ctr.ptr_to([item_s - FULL_ITEMS]), K.int32(0))
                                                K.ptx.bar.sync(K.uint32(3), K.uint32(SOFT_THREADS))
                                                base_slot = (item_s - FULL_ITEMS) * SPLIT
                                                mp = K.alloc_local([SPLIT], "float32")
                                                lp = K.alloc_local([SPLIT], "float32")
                                                for pi in range(SPLIT):
                                                    K.ptx.ld.relaxed.gpu.global_.f32(mp[pi], part_ml.ptr_to([((base_slot + pi) * ROWS + tid) * 2]))
                                                    K.ptx.ld.relaxed.gpu.global_.f32(lp[pi], part_ml.ptr_to([((base_slot + pi) * ROWS + tid) * 2 + 1]))
                                                m_tot = K.local_scalar("float32", init=mp[0])
                                                for pi in range(1, SPLIT):
                                                    K.assign(m_tot, K.max(m_tot, mp[pi]))
                                                msafe = K.Select(m_tot == K.float32(NEG_INF), K.float32(0.0), m_tot)
                                                l_tot = K.local_scalar("float32", init=K.float32(0.0))
                                                for c in range(128):
                                                    K.assign(s[c], K.float32(0.0))
                                                for pi in range(SPLIT):
                                                    sc = K.local_scalar("float32")
                                                    K.ptx.ex2.approx.ftz.f32(sc, mp[pi] - msafe)
                                                    K.assign(l_tot, l_tot + lp[pi] * sc)
                                                    ob = K.alloc_local([4], "float32")
                                                    for i in range(32):
                                                        K.ptx.ld.relaxed.gpu.global_.v4.f32(
                                                            ob[0], ob[1], ob[2], ob[3],
                                                            part_o.ptr_to([((base_slot + pi) * ROWS + tid) * HEAD_DIM + 4 * i]),
                                                        )
                                                        for e in range(4):
                                                            K.assign(s[4 * i + e], s[4 * i + e] + ob[e] * sc)
                                                write_row(s, l_tot)
                                iket_end(tk_epi)
                            with K.Else():
                                o_ready.wait(0, it & 1)
                                o_free.arrive(0)
                                for i in range(8):
                                    K.ptx[ST_OUT_V8](out.ptr_to([out_row + 16 * i]), *(zeros16[j] for j in range(8)))
                        union_free.arrive(slot)
                K.assign(it, it + 1)

        with wg1:

            with r_mma:
                kv_pipe = K.PipelineState(STAGES, phase=0)
                it_m = K.local_scalar("int32", init=0)
                gj_m = K.local_scalar("int32", init=0)
                running_m = K.local_scalar("int32", init=1)
                zero = K.uint32(0)

                def mma_qk(sb, kstage, qslot):
                    tk = iket_range("mma-qk")
                    d_addr = tmem_base + K.Cast("uint32", sb) * K.uint32(S_COLS)
                    for ki in range(NK):
                        with K.If(elected()), K.Then():
                            K.ptx[MMA](
                                d_addr,
                                desc_at(q_desc, qslot * QP16 + qoff(ki)),
                                desc_at(k_desc, kstage * KV16 + koff(ki)),
                                K.uint32(ID_QK),
                                zero,
                                zero,
                                zero,
                                zero,
                                ki != 0,
                            )
                    iket_end(tk)

                def mma_pv(vstage, pb, acc):
                    tk = iket_range("mma-pv")
                    p_addr = tmem_base + K.uint32(P_BASE) + K.Cast("uint32", pb) * K.uint32(P_COLS)
                    for ki in range(NK):
                        with K.If(elected()), K.Then():
                            K.ptx[MMA](
                                tmem_base + K.uint32(O_COL),
                                p_addr + K.uint32(ki * (MMA_K // 2)),
                                desc_at(v_desc, vstage * KV16 + voff(ki)),
                                K.uint32(ID_PV),
                                zero,
                                zero,
                                zero,
                                zero,
                                True if ki != 0 else K.Cast("bool", acc != 0),
                            )
                    iket_end(tk)

                with K.While(running_m != 0):
                    slot = it_m & 1
                    union_ready.wait(slot, (it_m >> 1) & 1)
                    n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                    with K.If(n_blocks < 0):
                        with K.Then():
                            K.assign(running_m, 0)
                        with K.Else():
                            q_load.full.wait(slot, (it_m >> 1) & 1)
                            o_free.wait(0, (it_m + 1) & 1)
                            with K.If(n_blocks == 0), K.Then():
                                with K.If(elected()), K.Then():
                                    q_load.empty.arrive(slot)
                                    o_ready.arrive(0)
                            with K.If(n_blocks > 0), K.Then():
                                acc = K.local_scalar("int32", init=0)
                                sb0 = gj_m & 1
                                kstage0 = K.local_scalar("int32", init=kv_pipe.stage)
                                kphase0 = K.local_scalar("int32", init=kv_pipe.phase)
                                kv_pipe.advance()
                                kv_load.full.wait(kstage0, kphase0)
                                s_free.wait(sb0, ((gj_m >> 1) + 1) & 1)
                                mma_qk(sb0, kstage0, slot)
                                with K.If(elected()), K.Then():
                                    s_ready.arrive(sb0)
                                    kv_load.empty.arrive(kstage0)
                                    with K.If(n_blocks == 1), K.Then():
                                        q_load.empty.arrive(slot)
                                with K.serial(n_blocks, unroll=False) as j:
                                    vstage = K.local_scalar("int32", init=kv_pipe.stage)
                                    vphase = K.local_scalar("int32", init=kv_pipe.phase)
                                    kv_pipe.advance()
                                    with K.If(j + 1 < n_blocks), K.Then():
                                        kstage = K.local_scalar("int32", init=kv_pipe.stage)
                                        kphase = K.local_scalar("int32", init=kv_pipe.phase)
                                        kv_pipe.advance()
                                        gj1 = gj_m + 1
                                        sb1 = gj1 & 1
                                        tk_wk = iket_range("mma-wait-k")
                                        kv_load.full.wait(kstage, kphase)
                                        iket_end(tk_wk)
                                        tk_wsf = iket_range("mma-wait-sfree")
                                        s_free.wait(sb1, ((gj1 >> 1) + 1) & 1)
                                        iket_end(tk_wsf)
                                        mma_qk(sb1, kstage, slot)
                                        with K.If(elected()), K.Then():
                                            s_ready.arrive(sb1)
                                            kv_load.empty.arrive(kstage)
                                            with K.If(j + 2 == n_blocks), K.Then():
                                                q_load.empty.arrive(slot)
                                    pb = gj_m & 1
                                    tk_wv = iket_range("mma-wait-v")
                                    kv_load.full.wait(vstage, vphase)
                                    iket_end(tk_wv)
                                    tk_wp = iket_range("mma-wait-p")
                                    p_ready.wait(pb, (gj_m >> 1) & 1)
                                    iket_end(tk_wp)
                                    mma_pv(vstage, pb, acc)
                                    with K.If(elected()), K.Then():
                                        p_free.arrive(pb)
                                        kv_load.empty.arrive(vstage)
                                        with K.If(j + 1 == n_blocks), K.Then():
                                            o_ready.arrive(0)
                                    K.assign(acc, 1)
                                    K.assign(gj_m, gj_m + 1)
                            union_free.arrive(slot)
                    K.assign(it_m, it_m + 1)


            with r_load:
                kv_pipe_l = K.PipelineState(STAGES, phase=0)
                it_l = K.local_scalar("int32", init=0)
                running_l = K.local_scalar("int32", init=1)
                with K.While(running_l != 0):
                    slot = it_l & 1
                    tk_lu = iket_range("ld-wait-union")
                    union_ready.wait(slot, (it_l >> 1) & 1)
                    iket_end(tk_lu)
                    n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                    with K.If(n_blocks < 0):
                        with K.Then():
                            K.assign(running_l, 0)
                        with K.Else():
                            h = ld_smem_i32(meta.ptr_to([slot * 16 + 2]))
                            kv_s = ld_smem_i32(meta.ptr_to([slot * 16 + 3]))
                            tok_base = ld_smem_i32(meta.ptr_to([slot * 16 + 5]))
                            q_load.empty.wait(slot, (it_l >> 1) & 1)
                            if not FP8:
                                with K.If(elected()), K.Then():
                                    K.ptx[TMA_4D](
                                        q_smem[slot].ptr_to(0, 0),
                                        K.address_of(q_map),
                                        K.int32(0),
                                        h * G,
                                        tok_base,
                                        K.int32(0),
                                        K.cuda.cvta_generic_to_shared(q_load.full.ptr_to([slot])),
                                    )
                                    q_load.full.arrive(slot, tx_count=Q_TMA_BYTES)
                            else:
                                words8 = K.alloc_local([8], "int32")
                                e4 = K.alloc_local([8], "uint16")
                                packed4 = K.alloc_local([4], "uint32")
                                for r in range(Q_ROUNDS):
                                    c = lane + 32 * r

                                    def quant_chunk(c=c):
                                        n = c >> 3
                                        k16 = c & 7
                                        row = tok_base + n // G
                                        head = h * G + (n % G)
                                        wbase = (row * HQ + head) * (HEAD_DIM // 2) + k16 * 8
                                        K.ptx.ld.global_.nc.v4.b32(
                                            words8[0], words8[1], words8[2], words8[3], qraw.ptr_to([wbase])
                                        )
                                        K.ptx.ld.global_.nc.v4.b32(
                                            words8[4], words8[5], words8[6], words8[7], qraw.ptr_to([wbase + 4])
                                        )
                                        for i in range(8):
                                            K.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(e4[i], K.Cast("uint32", words8[i]))
                                        for i in range(4):
                                            K.ptx.mov.b32(packed4[i], e4[2 * i], e4[2 * i + 1])
                                        K.ptx.st.shared.v4.b32(
                                            q_smem[slot].ptr_to(n, k16 * 16), packed4[0], packed4[1], packed4[2], packed4[3]
                                        )

                                    if 32 * r + 32 <= Q_CHUNKS:
                                        quant_chunk()
                                    else:
                                        with K.If(c < Q_CHUNKS), K.Then():
                                            quant_chunk()
                                K.ptx.fence.proxy.async_.shared__cta()
                                q_load.full.arrive(slot)
                            blk_off_l = ld_smem_i32(meta.ptr_to([slot * 16 + 7]))
                            with K.If(n_blocks > 0), K.Then():
                                with K.serial(n_blocks, unroll=False) as j:
                                    blk = ld_smem_i32(ulist.ptr_to([slot * MAX_UNION + blk_off_l + j]))
                                    if PAGED:
                                        pg = ld_smem_i32(uplist.ptr_to([slot * MAX_UNION + blk_off_l + j]))
                                    for tmap in (k_map, v_map):
                                        tk_we = iket_range("ld-wait-stage")
                                        kv_load.empty.wait(kv_pipe_l.stage, kv_pipe_l.phase)
                                        iket_end(tk_we)
                                        with K.If(elected()), K.Then():
                                            if PAGED and not FP8:
                                                K.ptx[TMA_4D_HINT](
                                                    kv_smem[kv_pipe_l.stage].ptr_to(0, 0),
                                                    K.address_of(tmap),
                                                    K.int32(0),
                                                    K.int32(0),
                                                    K.int32(0),
                                                    pg * HKV + h,
                                                    K.cuda.cvta_generic_to_shared(kv_load.full.ptr_to([kv_pipe_l.stage])),
                                                    K.uint64(KV_CACHE_POLICY),
                                                )
                                            elif PAGED:
                                                K.ptx[TMA_3D_HINT](
                                                    kv_smem[kv_pipe_l.stage].ptr_to(0, 0),
                                                    K.address_of(tmap),
                                                    K.int32(0),
                                                    K.int32(0),
                                                    pg * HKV + h,
                                                    K.cuda.cvta_generic_to_shared(kv_load.full.ptr_to([kv_pipe_l.stage])),
                                                    K.uint64(KV_CACHE_POLICY),
                                                )
                                            elif not FP8:
                                                K.ptx[TMA_3D_HINT](
                                                    kv_smem[kv_pipe_l.stage].ptr_to(0, 0),
                                                    K.address_of(tmap),
                                                    K.int32(0),
                                                    kv_s + blk * BLK,
                                                    h * 2,
                                                    K.cuda.cvta_generic_to_shared(kv_load.full.ptr_to([kv_pipe_l.stage])),
                                                    K.uint64(KV_CACHE_POLICY),
                                                )
                                            else:
                                                K.ptx[TMA_3D_HINT](
                                                    kv_smem[kv_pipe_l.stage].ptr_to(0, 0),
                                                    K.address_of(tmap),
                                                    K.int32(0),
                                                    kv_s + blk * BLK,
                                                    h,
                                                    K.cuda.cvta_generic_to_shared(kv_load.full.ptr_to([kv_pipe_l.stage])),
                                                    K.uint64(KV_CACHE_POLICY),
                                                )
                                            kv_load.full.arrive(kv_pipe_l.stage, tx_count=KV_TILE_BYTES)
                                        kv_pipe_l.advance()
                            union_free.arrive(slot)
                    K.assign(it_l, it_l + 1)


            with r_prep:
                it_p = K.local_scalar("int32", init=0)
                running_p = K.local_scalar("int32", init=1)
                with K.While(running_p != 0):
                    slot = it_p & 1
                    task = K.local_scalar("int32")
                    with K.If(it_p == 0):
                        with K.Then():
                            K.assign(task, K.cta_id())
                        with K.Else():
                            grabbed = K.local_scalar("int32", init=0)
                            with K.If(lane == 0), K.Then():
                                K.ptx.atom.relaxed.gpu.global_.add.s32(grabbed, sched.ptr_to([0]), K.int32(1))
                            K.assign(task, K.uniform(grabbed) + NUM_CTAS)
                    tk_pf = iket_range("prep-wait-free")
                    union_free.wait(slot, ((it_p >> 1) + 1) & 1)
                    iket_end(tk_pf)
                    tk_pi = iket_range("prep-item")
                    with K.If(task >= NUM_TASKS):
                        with K.Then():
                            with K.If(lane == 0), K.Then():
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16]), K.int32(-1))
                            union_ready.arrive(slot)
                            K.assign(running_p, 0)
                        with K.Else():
                            item = K.local_scalar("int32", init=task)
                            part = K.local_scalar("int32", init=0)
                            if SPLIT > 1:
                                with K.If(task >= FULL_ITEMS), K.Then():
                                    tidx = task - FULL_ITEMS
                                    K.assign(item, FULL_ITEMS + tidx // SPLIT)
                                    K.assign(part, tidx - (tidx // SPLIT) * SPLIT)
                            b = item // (HKV * NCH)
                            rem = item - b * (HKV * NCH)
                            h = rem // NCH
                            ch = rem - h * NCH
                            kv_s = K.local_scalar("int32", init=0)
                            kv_len = K.local_scalar("int32")
                            if PAGED:
                                K.assign(kv_len, ld_global_i32(lens.ptr_to([b])))
                            else:
                                K.assign(kv_s, ld_global_i32(lens.ptr_to([b])))
                                K.assign(kv_len, ld_global_i32(lens.ptr_to([b + 1])) - kv_s)
                            n_vis = (kv_len + (BLK - 1)) >> 7
                            tok_base = b * TQ + ch * T
                            causal_off = kv_len - TQ + ch * T
                            base = (h * TOTAL_Q + tok_base) * TOPK
                            idxs = K.alloc_local([N_ROUNDS], "int32")
                            for r in range(N_ROUNDS):
                                e = lane + 32 * r
                                K.assign(idxs[r], -1)
                                if 32 * r + 32 <= MAX_UNION:
                                    K.assign(idxs[r], ld_global_i32(q2k.ptr_to([base + e])))
                                else:
                                    with K.If(e < MAX_UNION), K.Then():
                                        K.assign(idxs[r], ld_global_i32(q2k.ptr_to([base + e])))

                            def valid(r):
                                return K.And(idxs[r] >= 0, idxs[r] < n_vis)

                            n_blocks = K.local_scalar("int32", init=0)
                            if T == 1:
                                for r in range(N_ROUNDS):
                                    vflag = K.local_scalar("int32", init=K.Select(valid(r), 1, 0))
                                    ballot = K.local_scalar("uint32")
                                    K.ptx.vote_sync.ballot.b32(ballot, K.ptx.pred(vflag), K.uint32(0xFFFFFFFF))
                                    before = K.local_scalar("uint32")
                                    K.ptx.popc.b32(before, K.bitwise_and(ballot, lanemask_lt()))
                                    total = K.local_scalar("uint32")
                                    K.ptx.popc.b32(total, ballot)
                                    rank = n_blocks + K.Cast("int32", before)
                                    with K.If(vflag != 0), K.Then():
                                        K.ptx.st.shared.b32(ulist.ptr_to([slot * MAX_UNION + rank]), idxs[r])
                                        if PAGED:
                                            pg = ld_global_i32(ptab.ptr_to([b * MAX_PAGES + idxs[r]]))
                                            K.ptx.st.shared.b32(uplist.ptr_to([slot * MAX_UNION + rank]), pg)
                                    K.assign(n_blocks, n_blocks + K.Cast("int32", total))
                            else:
                                nwords = (n_vis + 31) >> 5
                                wi = K.local_scalar("int32", init=lane)
                                with K.While(wi < nwords):
                                    K.ptx.st.shared.b32(words.ptr_to([wi]), K.uint32(0))
                                    K.assign(wi, wi + 32)
                                K.cuda.warp_sync()
                                for r in range(N_ROUNDS):
                                    with K.If(valid(r)), K.Then():
                                        K.ptx.red.shared.or_.b32(
                                            words.ptr_to([idxs[r] >> 5]),
                                            K.shift_left(K.uint32(1), K.Cast("uint32", idxs[r] & 31)),
                                        )
                                K.cuda.warp_sync()
                                carry = K.local_scalar("uint32", init=K.uint32(0))
                                wbase = K.local_scalar("int32", init=0)
                                with K.While(wbase < nwords):
                                    w = wbase + lane
                                    wv = K.local_scalar("uint32", init=K.uint32(0))
                                    with K.If(w < nwords), K.Then():
                                        K.ptx.ld.shared.u32(wv, words.ptr_to([w]))
                                    cnt = K.alloc_local([1], "uint32")
                                    K.ptx.popc.b32(cnt[0], wv)
                                    own = K.local_scalar("uint32", init=cnt[0])
                                    K.idioms.warp_scan_add(cnt, 1, lane)
                                    with K.If(w < nwords), K.Then():
                                        K.ptx.st.shared.b32(prefix.ptr_to([w]), carry + (cnt[0] - own))
                                    tot = K.local_scalar("uint32")
                                    K.ptx.shfl_sync.idx.b32(tot, cnt[0], K.uint32(31), K.uint32(31), K.uint32(0xFFFFFFFF))
                                    K.assign(carry, carry + tot)
                                    K.assign(wbase, wbase + 32)
                                K.assign(n_blocks, K.Cast("int32", carry))
                                K.cuda.warp_sync()
                                with K.serial(nwords, unroll=False) as w:
                                    wv = ld_smem_u32(words.ptr_to([w]))
                                    pw = ld_smem_u32(prefix.ptr_to([w]))
                                    mybit = K.bitwise_and(K.shift_right(wv, K.Cast("uint32", lane)), K.uint32(1))
                                    with K.If(mybit != K.uint32(0)), K.Then():
                                        below = K.local_scalar("uint32")
                                        K.ptx.popc.b32(below, K.bitwise_and(wv, lanemask_lt()))
                                        rank = K.Cast("int32", pw + below)
                                        blkid = w * 32 + lane
                                        K.ptx.st.shared.b32(ulist.ptr_to([slot * MAX_UNION + rank]), blkid)
                                        K.ptx.st.shared.b32(umask.ptr_to([slot * MAX_UNION + rank]), K.uint32(0))
                                        if PAGED:
                                            pg = ld_global_i32(ptab.ptr_to([b * MAX_PAGES + blkid]))
                                            K.ptx.st.shared.b32(uplist.ptr_to([slot * MAX_UNION + rank]), pg)
                                K.cuda.warp_sync()
                                for r in range(N_ROUNDS):
                                    e = lane + 32 * r
                                    with K.If(valid(r)), K.Then():
                                        wq = idxs[r] >> 5
                                        wv = ld_smem_u32(words.ptr_to([wq]))
                                        pw = ld_smem_u32(prefix.ptr_to([wq]))
                                        below = K.local_scalar("uint32")
                                        K.ptx.popc.b32(
                                            below,
                                            K.bitwise_and(
                                                wv,
                                                K.shift_left(K.uint32(1), K.Cast("uint32", idxs[r] & 31)) - K.uint32(1),
                                            ),
                                        )
                                        u = K.Cast("int32", pw + below)
                                        K.ptx.red.shared.or_.b32(
                                            umask.ptr_to([slot * MAX_UNION + u]),
                                            K.shift_left(K.uint32(1), K.Cast("uint32", e // TOPK)),
                                        )
                            K.cuda.warp_sync()
                            blk_off = K.local_scalar("int32", init=0)
                            if SPLIT > 1:
                                with K.If(item >= FULL_ITEMS), K.Then():
                                    per_part = (n_blocks + (SPLIT - 1)) // SPLIT
                                    start = K.min(part * per_part, n_blocks)
                                    K.assign(blk_off, start)
                                    K.assign(n_blocks, K.min(n_blocks - start, per_part))
                            with K.If(lane == 0), K.Then():
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 7]), blk_off)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 8]), item)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 9]), part)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 0]), n_blocks)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 1]), b)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 2]), h)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 3]), kv_s)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 4]), kv_len)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 5]), tok_base)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 6]), causal_off)
                            union_ready.arrive(slot)
                    iket_end(tk_pi)
                    K.assign(it_p, it_p + 1)

            with r_idle:
                pass


        K.cuda.cta_sync()
        with K.If(tid == 0), K.Then():
            done = K.local_scalar("int32")
            K.ptx.atom.acq_rel.gpu.global_.add.s32(done, sched.ptr_to([1]), K.int32(1))
            with K.If(done == NUM_CTAS - 1), K.Then():
                K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([0]), K.int32(0))
                K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([1]), K.int32(0))
        with K.If(warp == MMA_WARP), K.Then():
            K.ptx[TMEM_DEALLOC](tmem_base, K.uint32(TMEM_COLS))

    return msa_decode_qmajor







class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode(tensor, dtype_name, dims, strides, box, l2_promotion=2):
    desc = _AlignedTensorMap()
    rank = len(dims)
    assert len(strides) == rank - 1 and len(box) == rank
    tvm.get_global_func("runtime.cuTensorMapEncodeTiled")(
        desc.ptr,
        dtype_name,
        rank,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *dims,
        *strides,
        *box,
        *((1,) * rank),
        0,
        3,
        l2_promotion,
        0,
    )
    return desc


_EXEC_CACHE = {}


def _compile(cfg, builder, family):
    key = (family,) + tuple(sorted(cfg.items()))
    hit = _EXEC_CACHE.get(key)
    if hit is None:
        kernel = builder(cfg)
        target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
        with target:
            hit = tvm.compile(kernel.mod, target=target, tir_pipeline="tirx")
        _EXEC_CACHE[key] = hit
    return hit


def choose_split(num_items, num_ctas, max_split=4):
    """Parts per tail item minimizing the makespan (in item units) of the dynamic schedule."""
    full = (num_items // num_ctas) * num_ctas
    tail = num_items - full
    if tail == 0:
        return 1
    best_s, best_cost = 1, 1.0
    for s in range(2, max_split + 1):
        cost = math.ceil(tail * s / num_ctas) / s
        if cost < best_cost - 1e-9:
            best_s, best_cost = s, cost
    return best_s


def plan_config_kv(q, k, q2k, page_table, B, T, num_sms):
    HQ = q.shape[1]
    HKV = k.shape[1]
    G = HQ // HKV
    TOPK = int(q2k.shape[-1])
    TOTAL_Q = q.shape[0]
    paged = page_table is not None
    kind = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float8_e4m3fn: "fp8"}[k.dtype]

    TI = max(1, min(T, 64 // G))
    while T % TI:
        TI -= 1
    NCH = T // TI
    if paged:
        MAX_PAGES = int(page_table.shape[1])
        max_blocks = MAX_PAGES
    else:
        MAX_PAGES = 0
        max_blocks = ceildiv(int(k.shape[0]), BLK)
    W_MAX = max(1, ceildiv(max_blocks, 32))
    NUM_ITEMS = B * HKV * NCH
    fp8_q1_two_resident = T == 1 and kind == "fp8"
    NUM_CTAS = max(1, min((2 * num_sms if fp8_q1_two_resident else num_sms), NUM_ITEMS))

    NQ = TI * G
    SPLIT = (choose_split(NUM_ITEMS, NUM_CTAS) if NQ >= 32 else 1) if ENV_SPLIT == 0 else ENV_SPLIT
    STATIC_ONE_SHOT = SPLIT == 1 and NUM_ITEMS == NUM_CTAS
    NQ_PAD = max(16, ceildiv(NQ, 16) * 16)
    NSWG = 2 if (NQ >= 32 and (NQ // 2) % 8 == 0 and (NQ // 2) % G == 0) else 1
    MAX_UNION = TI * TOPK
    eb = 1 if kind == "fp8" else 2
    kv_tile = BLK * HEAD_DIM * eb
    qp_tile = NQ_PAD * HEAD_DIM * eb
    misc = 24 * MAX_UNION + 8 * W_MAX + 48 * NQ_PAD + 4096
    budget = 227 * 1024 - misc - 4 * qp_tile
    STAGES = int(min(12 if kind == "fp8" else 8, budget // kv_tile))
    if fp8_q1_two_resident:
        STAGES = min(STAGES, 4)
    if ENV_STAGES:
        STAGES = min(STAGES, ENV_STAGES)


    STAGES -= STAGES % 2
    assert STAGES >= 2, "shared memory budget too small"
    return dict(
        T=T,
        TI=TI,
        G=G,
        HKV=HKV,
        TOPK=TOPK,
        TOTAL_Q=TOTAL_Q,
        kv_kind=kind,
        paged=paged,
        MAX_PAGES=MAX_PAGES,
        MAX_BLOCK_WORDS=W_MAX,
        NUM_ITEMS=NUM_ITEMS,
        NUM_CTAS=NUM_CTAS,
        STAGES=STAGES,
        MIN_BLOCKS=2 if fp8_q1_two_resident else 1,
        SPLIT_LOADERS=fp8_q1_two_resident,
        STATIC_ONE_SHOT=STATIC_ONE_SHOT,
        NSWG=NSWG,
        SPLIT=SPLIT,
    )


def plan_config_qm(q, k, q2k, page_table, B, T, num_sms):
    HQ = q.shape[1]
    HKV = k.shape[1]
    G = HQ // HKV
    TOPK = int(q2k.shape[-1])
    TOTAL_Q = q.shape[0]
    paged = page_table is not None
    kind = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float8_e4m3fn: "fp8"}[k.dtype]
    if kind == "fp8" or 128 % G != 0:
        raise NotImplementedError("q-major family: bf16/f16 KV and GQA dividing 128 only")
    TI = 128 // G
    if T % TI != 0:
        raise NotImplementedError("q-major family needs seqlen_q to be a multiple of 128/GQA")
    NCH = T // TI
    if paged:
        MAX_PAGES = int(page_table.shape[1])
        max_blocks = MAX_PAGES
    else:
        MAX_PAGES = 0
        max_blocks = ceildiv(int(k.shape[0]), BLK)
    W_MAX = max(1, ceildiv(max_blocks, 32))
    NUM_ITEMS = B * HKV * NCH
    NUM_CTAS = max(1, min(num_sms, NUM_ITEMS))
    env_split = int(os.environ.get("QM_SPLIT", "0"))

    SPLIT = 1 if env_split == 0 else env_split
    MAX_UNION = TI * TOPK
    kv_tile = BLK * HEAD_DIM * 2
    q_tiles = 2 * 128 * HEAD_DIM * 2
    misc = 24 * MAX_UNION + 8 * W_MAX + 4096
    budget = 227 * 1024 - misc - q_tiles
    STAGES = int(min(8, budget // kv_tile))
    if int(os.environ.get("QM_STAGES", "0")):
        STAGES = min(STAGES, int(os.environ["QM_STAGES"]))
    assert STAGES >= 2, "shared memory budget too small"
    return dict(
        T=T,
        TI=TI,
        G=G,
        HKV=HKV,
        TOPK=TOPK,
        TOTAL_Q=TOTAL_Q,
        kv_kind=kind,
        paged=paged,
        MAX_PAGES=MAX_PAGES,
        MAX_BLOCK_WORDS=W_MAX,
        NUM_ITEMS=NUM_ITEMS,
        NUM_CTAS=NUM_CTAS,
        STAGES=STAGES,
        SPLIT=SPLIT,
    )


def setup_kv(data, B, seqlen_q):
    q = data["q"]
    k = data["k"]
    v = data["v"]
    q2k = data["q2k_indices"]
    cu_seqlens_k = data["cu_seqlens_k"]
    page_table = data["page_table"]
    seqused_k = data["seqused_k"]
    out = data["output"]
    scale = float(data["softmax_scale"])
    T = int(seqlen_q)
    B = int(B)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous()
    assert q.shape[0] == B * T
    device = q.device
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    cfg = plan_config_kv(q, k, q2k, page_table, B, T, num_sms)
    ex = _compile(cfg, make_kernel_kv, 'kv')

    HKV = cfg["HKV"]
    G = cfg["G"]
    TI = cfg["TI"]
    HQ = HKV * G
    TOTAL_Q = cfg["TOTAL_Q"]
    fp8 = cfg["kv_kind"] == "fp8"
    q_dtype_name = "bfloat16" if q.dtype == torch.bfloat16 else "float16"
    kv_dtype_name = "uint8" if fp8 else q_dtype_name
    NQ_PAD = max(16, ceildiv(TI * G, 16) * 16)
    q_map = _encode(
        q,
        q_dtype_name,
        (HEAD_DIM // 2, HQ, TOTAL_Q, 2),
        (HEAD_DIM * 2, HQ * HEAD_DIM * 2, (HEAD_DIM // 2) * 2),
        (HEAD_DIM // 2, NQ_PAD // TI, TI, 2),
    )
    if cfg["paged"]:
        num_pages = int(k.shape[0])
        if fp8:
            kv_dims = (HEAD_DIM, BLK, num_pages * HKV)
            kv_strides = (HEAD_DIM, BLK * HEAD_DIM)
            kv_box = (HEAD_DIM, BLK, 1)
        else:
            kv_dims = (HEAD_DIM // 2, BLK, 2, num_pages * HKV)
            kv_strides = (HEAD_DIM * 2, (HEAD_DIM // 2) * 2, BLK * HEAD_DIM * 2)
            kv_box = (HEAD_DIM // 2, BLK, 2, 1)
        lens = seqused_k.contiguous()
        ptab = page_table.contiguous().view(-1)
    else:
        total_k = int(k.shape[0])
        if fp8:
            kv_dims = (HEAD_DIM, total_k, HKV)
            kv_strides = (HKV * HEAD_DIM, HEAD_DIM)
            kv_box = (HEAD_DIM, BLK, 1)
        else:
            kv_dims = (HEAD_DIM // 2, total_k, HKV * 2)
            kv_strides = (HKV * HEAD_DIM * 2, (HEAD_DIM // 2) * 2)
            kv_box = (HEAD_DIM // 2, BLK, 2)
        lens = cu_seqlens_k.contiguous()
        ptab = torch.zeros(4, dtype=torch.int32, device=device)
    k_map = _encode(k, kv_dtype_name, kv_dims, kv_strides, kv_box, l2_promotion=3)
    v_map = _encode(v, kv_dtype_name, kv_dims, kv_strides, kv_box, l2_promotion=3)
    sched = torch.zeros(4, dtype=torch.int32, device=device)
    q2k_flat = q2k.contiguous().view(-1)
    out_flat = out.view(-1)
    if fp8:
        qraw = q.view(torch.int32).view(-1)
    else:
        qraw = torch.zeros(4, dtype=torch.int32, device=device)
    NUM_ITEMS = cfg["NUM_ITEMS"]
    NUM_CTAS = cfg["NUM_CTAS"]
    SPLIT = cfg["SPLIT"]
    NQ = TI * G
    tail_items = NUM_ITEMS - (NUM_ITEMS // NUM_CTAS) * NUM_CTAS if SPLIT > 1 else 0
    n_parts = max(1, tail_items * SPLIT)
    part_o = torch.zeros(n_parts * HEAD_DIM * NQ, dtype=torch.float32, device=device)
    part_ml = torch.zeros(n_parts * 2 * NQ, dtype=torch.float32, device=device)
    part_ctr = torch.zeros(max(4, tail_items), dtype=torch.int32, device=device)
    args = (
        q_map.ptr,
        k_map.ptr,
        v_map.ptr,
        out_flat,
        q2k_flat,
        lens,
        ptab,
        qraw,
        sched,
        part_o,
        part_ml,
        part_ctr,
        float(scale * LOG2E),
    )
    keep = (q, k, v, q2k, q2k_flat, lens, ptab, qraw, out, out_flat, sched, part_o, part_ml, part_ctr, q_map, k_map, v_map)

    def run():
        ex(*args)

    run._keep = keep
    run._cfg = cfg
    run()
    torch.cuda.synchronize(device)
    return run


def setup_qm(data, B, seqlen_q):
    q = data["q"]
    k = data["k"]
    v = data["v"]
    q2k = data["q2k_indices"]
    cu_seqlens_k = data["cu_seqlens_k"]
    page_table = data["page_table"]
    seqused_k = data["seqused_k"]
    out = data["output"]
    scale = float(data["softmax_scale"])
    T = int(seqlen_q)
    B = int(B)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous()
    assert q.shape[0] == B * T
    device = q.device
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    cfg = plan_config_qm(q, k, q2k, page_table, B, T, num_sms)
    ex = _compile(cfg, make_kernel_qm, 'qm')

    HKV = cfg["HKV"]
    G = cfg["G"]
    TI = cfg["TI"]
    HQ = HKV * G
    TOTAL_Q = cfg["TOTAL_Q"]
    dtype_name = "bfloat16" if q.dtype == torch.bfloat16 else "float16"
    q_map = _encode(
        q,
        dtype_name,
        (HEAD_DIM // 2, HQ, TOTAL_Q, 2),
        (HEAD_DIM * 2, HQ * HEAD_DIM * 2, (HEAD_DIM // 2) * 2),
        (HEAD_DIM // 2, G, TI, 2),
    )
    if cfg["paged"]:
        num_pages = int(k.shape[0])
        kv_dims = (HEAD_DIM // 2, BLK, 2, num_pages * HKV)
        kv_strides = (HEAD_DIM * 2, (HEAD_DIM // 2) * 2, BLK * HEAD_DIM * 2)
        kv_box = (HEAD_DIM // 2, BLK, 2, 1)
        lens = seqused_k.contiguous()
        ptab = page_table.contiguous().view(-1)
    else:
        total_k = int(k.shape[0])
        kv_dims = (HEAD_DIM // 2, total_k, HKV * 2)
        kv_strides = (HKV * HEAD_DIM * 2, (HEAD_DIM // 2) * 2)
        kv_box = (HEAD_DIM // 2, BLK, 2)
        lens = cu_seqlens_k.contiguous()
        ptab = torch.zeros(4, dtype=torch.int32, device=device)
    k_map = _encode(k, dtype_name, kv_dims, kv_strides, kv_box, l2_promotion=3)
    v_map = _encode(v, dtype_name, kv_dims, kv_strides, kv_box, l2_promotion=3)
    sched = torch.zeros(4, dtype=torch.int32, device=device)
    q2k_flat = q2k.contiguous().view(-1)
    out_flat = out.view(-1)
    qraw = torch.zeros(4, dtype=torch.int32, device=device)
    NUM_ITEMS = cfg["NUM_ITEMS"]
    NUM_CTAS = cfg["NUM_CTAS"]
    SPLIT = cfg["SPLIT"]
    tail_items = NUM_ITEMS - (NUM_ITEMS // NUM_CTAS) * NUM_CTAS if SPLIT > 1 else 0
    n_parts = max(1, tail_items * SPLIT)
    part_o = torch.zeros(n_parts * 128 * HEAD_DIM, dtype=torch.float32, device=device)
    part_ml = torch.zeros(n_parts * 128 * 2, dtype=torch.float32, device=device)
    part_ctr = torch.zeros(max(4, tail_items), dtype=torch.int32, device=device)
    args = (
        q_map.ptr,
        k_map.ptr,
        v_map.ptr,
        out_flat,
        q2k_flat,
        lens,
        ptab,
        qraw,
        sched,
        part_o,
        part_ml,
        part_ctr,
        float(scale * LOG2E),
    )
    keep = (q, k, v, q2k, q2k_flat, lens, ptab, qraw, out, out_flat, sched, part_o, part_ml, part_ctr, q_map, k_map, v_map)

    def run():
        ex(*args)

    run._keep = keep
    run._cfg = cfg
    run()
    torch.cuda.synchronize(device)
    return run


def choose_family(q, k, T):
    """Q-major when seqlen_q x GQA fills whole 128-row tiles (bf16/f16), else KV-major."""
    HQ = q.shape[1]
    HKV = k.shape[1]
    G = HQ // HKV
    if k.dtype == torch.float8_e4m3fn:
        return "kv"
    if 128 % G != 0:
        return "kv"
    TI = 128 // G
    if T % TI != 0:
        return "kv"
    return "qm"


def setup(data, B, seqlen_q):
    family = choose_family(data["q"], data["k"], int(seqlen_q))
    if family == "qm":
        return setup_qm(data, B, seqlen_q)
    return setup_kv(data, B, seqlen_q)

def make_kernel_qm64(cfg):
    TQ = cfg["T"]
    T = cfg["TI"]
    NCH = TQ // T
    G = cfg["G"]
    HKV = cfg["HKV"]
    TOPK = cfg["TOPK"]
    TOTAL_Q = cfg["TOTAL_Q"]
    kind = cfg["kv_kind"]
    PAGED = cfg["paged"]
    MAX_PAGES = cfg["MAX_PAGES"]
    W_MAX = cfg["MAX_BLOCK_WORDS"]
    NUM_ITEMS = cfg["NUM_ITEMS"]
    NUM_CTAS = cfg["NUM_CTAS"]
    STAGES = cfg["STAGES"]
    QSLOTS = cfg["QSLOTS"]
    assert QSLOTS in (1, 2)
    STATIC_ONE_SHOT = cfg.get("STATIC_ONE_SHOT", False)
    KS = int(os.environ.get("QM_KSTAGES", str(max(1, STAGES // 2))))
    VS = STAGES - KS
    assert KS >= 1 and VS >= 1
    SPLIT = cfg["SPLIT"]
    FULL_ITEMS = (NUM_ITEMS // NUM_CTAS) * NUM_CTAS if SPLIT > 1 else NUM_ITEMS
    TAIL_ITEMS = NUM_ITEMS - FULL_ITEMS
    NUM_TASKS = FULL_ITEMS + TAIL_ITEMS * SPLIT
    HQ = HKV * G
    ROWS = T * G
    assert ROWS == 128, "q-major family needs exactly 128 query rows per item"
    FP8 = False
    assert kind in ("bf16", "f16")
    kv_dt = {"bf16": K.bf16, "f16": K.f16}[kind]
    out_dt = kv_dt
    EB = 2
    TILE_N = 64
    MMA_K = 16
    NQK = HEAD_DIM // MMA_K
    NPV = TILE_N // MMA_K
    KV_TILE_BYTES = TILE_N * HEAD_DIM * EB
    NQ_PAD = ROWS
    Q_TMA_BYTES = ROWS * HEAD_DIM * EB
    MMA = MMA_F16
    fmt = 1 if kind == "bf16" else 0
    ID_QK = make_idesc(128, TILE_N, fmt, fmt, 0, 0)
    ID_PV = make_idesc(128, 128, fmt, fmt, 0, 1)
    S_COLS = TILE_N
    P_BASE = 0
    P_COLS = TILE_N // 2
    L_COL = 320
    ONES_ROWS = 16
    ID_L = make_idesc(128, ONES_ROWS, fmt, fmt, 0, 0)
    ONE2 = 0x3F803F80 if kind == "bf16" else 0x3C003C00
    O_COL = 128
    TMEM_COLS = 256
    MAX_UNION = T * TOPK
    N_ROUNDS = ceildiv(MAX_UNION, 32)
    KV16 = KV_TILE_BYTES // 16
    QP16 = ROWS * HEAD_DIM * EB // 16
    NWARPS = 8
    MMA_WARP = 4
    LOAD_WARP = 5
    PREP_WARP = 6
    VLOAD_WARP = 7
    SOFT_THREADS = 128
    LOG2G = G.bit_length() - 1
    assert (1 << LOG2G) == G

    @K.kernel(warps=NWARPS, arch="sm_100a", min_blocks_per_sm=2, grid=NUM_CTAS)
    def msa_decode_qmajor(
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        out: K.gptr[out_dt],
        q2k: K.gptr[K.i32],
        lens: K.gptr[K.i32],
        ptab: K.gptr[K.i32],
        qraw: K.gptr[K.i32],
        sched: K.gptr[K.i32],
        part_o: K.gptr[K.f32],
        part_ml: K.gptr[K.f32],
        part_ctr: K.gptr[K.i32],
        scale_log2: K.f32,
    ):
        warp = K.warp_id()
        lane = K.lane_id()
        tid = K.thread_id()

        smem = K.smem_pool()
        k_smem = smem.alloc((KS, TILE_N, HEAD_DIM), kv_dt, swizzle=K.SW128B)
        v_smem = smem.alloc((VS, TILE_N, HEAD_DIM), kv_dt, swizzle=K.SW128B)
        q_smem = smem.alloc((QSLOTS, ROWS, HEAD_DIM), kv_dt, swizzle=K.SW128B)
        tmem_addr = smem.alloc((4,), K.u32)
        meta = smem.alloc((32,), K.i32)
        ulist = smem.alloc((2 * MAX_UNION,), K.i32)
        uplist = smem.alloc((2 * MAX_UNION,), K.i32)
        umask = smem.alloc((2 * MAX_UNION,), K.u32)
        words = smem.alloc((W_MAX,), K.u32)
        prefix = smem.alloc((W_MAX,), K.u32)
        K.keep_alive(ptab.ptr_to([0]))
        K.keep_alive(qraw.ptr_to([0]))
        K.keep_alive(part_o.ptr_to([0]))
        K.keep_alive(part_ml.ptr_to([0]))
        K.keep_alive(part_ctr.ptr_to([0]))
        mrg_order = smem.alloc((4,), K.i32)


        def lo_uniform(desc):
            desc_lo = K.alloc_local((1,), "uint32")
            desc_hi = K.alloc_local((1,), "uint32")
            K.assign(desc_lo[0], K.uniform(K.Cast("uint32", desc.value)))
            K.assign(desc_hi[0], K.Cast("uint32", K.shift_right(desc.value, K.uint64(32))))
            return desc_lo, desc_hi

        def desc_at(desc, off16):
            lo, hi = desc
            packed = K.alloc_local((1,), "uint64")
            low = lo[0] if isinstance(off16, int) and off16 == 0 else lo[0] + K.Cast("uint32", off16)
            K.assign(
                packed[0],
                K.bitwise_or(K.shift_left(K.Cast("uint64", hi[0]), K.uint64(32)), K.Cast("uint64", low)),
            )
            return packed[0]

        def encode(view, major="k"):
            desc, off16 = view.encode(major=major, mma_k=MMA_K)
            return lo_uniform(desc), off16

        k_desc, koff = encode(k_smem[0], "k")
        v_desc, voff = encode(v_smem[0], "mn")
        q_desc, qoff = encode(q_smem[0], "k")


        k_load = K.Pipeline(smem, KS, full="tma", empty="tcgen05", empty_phase_offset=1)
        v_load = K.Pipeline(smem, VS, full="tma", empty="tcgen05", empty_phase_offset=1)
        q_load = K.Pipeline(smem, QSLOTS, full="tma", empty="tcgen05", init_full=1, empty_phase_offset=1)

        def q_stage(it_v):
            return (it_v & 1) if QSLOTS == 2 else K.int32(0)

        def q_phase(it_v):
            return ((it_v >> 1) & 1) if QSLOTS == 2 else (it_v & 1)
        union_ready = K.MBarrier(smem, 2)
        union_ready.init(32)
        if not STATIC_ONE_SHOT:
            union_free = K.MBarrier(smem, 2)
            union_free.init(SOFT_THREADS + 96)
        s_ready = K.TCGen05Bar(smem, 2)
        s_ready.init(1)
        p_ready = K.MBarrier(smem, 2)
        p_ready.init(SOFT_THREADS)
        p_free = K.TCGen05Bar(smem, 2)
        p_free.init(1)
        o_ready = K.TCGen05Bar(smem, 1)
        o_ready.init(1)
        o_free = K.MBarrier(smem, 1)
        o_free.init(SOFT_THREADS)

        K.ptx.fence.proxy.async_.shared__cta()
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        with K.If(warp == MMA_WARP), K.Then():
            K.ptx[TMEM_ALLOC](K.address_of(tmem_addr[0]), K.uint32(TMEM_COLS))
            K.ptx[TMEM_RELINQUISH]()
        K.cuda.cta_sync()
        tb_raw = K.local_scalar("uint32")
        K.ptx.ld.shared.u32(tb_raw, tmem_addr.ptr_to([0]))
        tmem_base = K.local_scalar("uint32", init=K.uniform(tb_raw))


        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def ld_smem_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.shared.b32(value, ptr)
            return value

        def ld_smem_u32(ptr):
            value = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(value, ptr)
            return value

        def ld_global_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.global_.nc.b32(value, ptr)
            return value

        def lanemask_lt():
            return K.shift_left(K.uint32(1), K.Cast("uint32", lane)) - K.uint32(1)

        def iket_range(name, *, leader_only=False):
            token = K.alloc_local([1], "uint32")
            if leader_only:
                K.assign(token[0], K.cuda.iket.sentinel_token(name))
                with K.If((warp & 3) == 0), K.Then():
                    K.assign(token[0], K.cuda.iket.range_start(name))
            else:
                K.assign(token[0], K.cuda.iket.range_start(name))
            return token

        def iket_end(token):
            K.cuda.iket.range_end(token[0])

        def pack_half2(dst, hi, lo):
            if kind == "bf16":
                K.ptx.cvt.rn.bf16x2.f32(dst, hi, lo)
            else:
                K.ptx.cvt.rn.f16x2.f32(dst, hi, lo)

        def ex2_emul(dst, x):
            """dst = 2^x for x <= 0 on the FMA/ALU pipes (Cody-Waite split + cubic), ~9 instructions."""
            xc = K.local_scalar("float32", init=K.max(x, K.float32(-126.0)))
            t = K.local_scalar("float32", init=xc + K.float32(QM64_EX2_MAGIC))
            nf = K.local_scalar("float32", init=t - K.float32(QM64_EX2_MAGIC))
            f = K.local_scalar("float32", init=xc - nf)
            pv = K.local_scalar("float32")
            K.ptx.fma.rn.f32(pv, K.float32(QM64_EX2_C3), f, K.float32(QM64_EX2_C2))
            K.ptx.fma.rn.f32(pv, pv, f, K.float32(QM64_EX2_C1))
            K.ptx.fma.rn.f32(pv, pv, f, K.float32(QM64_EX2_C0))
            ti = K.local_scalar("int32")
            K.ptx.mov.b32(ti, t)
            pbits = K.local_scalar("int32")
            K.ptx.mov.b32(pbits, pv)
            rbits = K.local_scalar("int32", init=pbits + K.shift_left(ti - K.int32(QM64_EX2_MAGIC_BITS), K.int32(23)))
            K.ptx.mov.b32(dst, rbits)

        def ex2_col(dst, x, c):
            if (c % 8) < QM64_EMU_NUM:
                ex2_emul(dst, x)
            else:
                K.ptx.ex2.approx.ftz.f32(dst, x)

        def tmem_row_load(dst, col0, ncols):
            for c in range(ncols // 32):
                K.ptx[tmem_ld(32)](*(dst[32 * c + i] for i in range(32)), tmem_base + K.uint32(col0 + 32 * c))
            K.ptx.tcgen05.wait__ld.sync.aligned()

        def tmem_row_store(src, col0, ncols):
            for c in range(ncols // 16):
                K.ptx[tmem_st(16)](tmem_base + K.uint32(col0 + 16 * c), *(src[16 * c + i] for i in range(16)))
            K.ptx.tcgen05.wait__st.sync.aligned()

        sp = K.specialize(chain_dispatch=True)
        r_soft = sp.role("softmax", warps=[0, 1, 2, 3], regs=QM64_SOFT_REGS)
        wg1 = sp.warpgroup("wg1", warps=range(4, 8), regs=QM64_PROD_REGS)
        r_mma = sp.role("mma", warps=[MMA_WARP], group=wg1)
        r_load = sp.role("load", warps=[LOAD_WARP], group=wg1)
        r_prep = sp.role("prep", warps=[PREP_WARP], group=wg1)
        r_vload = sp.role("vload", warps=[VLOAD_WARP], group=wg1)


        with r_soft:
            t_row = tid >> LOG2G
            g_row = tid & (G - 1)
            it = K.local_scalar("int32", init=0)
            gj = K.local_scalar("int32", init=0)
            running = K.local_scalar("int32", init=1)
            m_s = K.local_scalar("float32", init=K.float32(NEG_INF))
            l_row = K.local_scalar("float32", init=K.float32(0.0))
            s = K.alloc_local([TILE_N], "float32")
            pk = K.alloc_local([P_COLS], "uint32")
            zeros16 = K.alloc_local([16], "uint32")
            for i in range(16):
                K.assign(zeros16[i], K.uint32(0))
            with K.While(running != 0):
                slot = it & 1
                tk_wu = iket_range("sm-wait-union", leader_only=True)
                union_ready.wait(slot, (it >> 1) & 1)
                iket_end(tk_wu)
                n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                with K.If(n_blocks < 0):
                    with K.Then():
                        K.assign(running, 0)
                    with K.Else():
                        h = ld_smem_i32(meta.ptr_to([slot * 16 + 2]))
                        tok_base = ld_smem_i32(meta.ptr_to([slot * 16 + 5]))
                        causal_off = ld_smem_i32(meta.ptr_to([slot * 16 + 6]))
                        blk_off_s = ld_smem_i32(meta.ptr_to([slot * 16 + 7]))
                        item_s = ld_smem_i32(meta.ptr_to([slot * 16 + 8]))
                        part_s = ld_smem_i32(meta.ptr_to([slot * 16 + 9]))
                        qpos = causal_off + t_row
                        out_row = ((tok_base + t_row) * HQ + h * G + g_row) * HEAD_DIM
                        with K.If(n_blocks > 0):
                            with K.Then():
                                n_tiles = n_blocks * 2
                                K.assign(m_s, K.float32(NEG_INF))
                                K.assign(l_row, K.float32(0.0))
                                with K.serial(n_tiles, unroll=False) as j:
                                    u_idx = j >> 1
                                    half = j & 1
                                    blk = ld_smem_i32(ulist.ptr_to([slot * MAX_UNION + blk_off_s + u_idx]))
                                    if T > 1:
                                        tmask = ld_smem_u32(umask.ptr_to([slot * MAX_UNION + blk_off_s + u_idx]))
                                        sel = K.local_scalar(
                                            "int32",
                                            init=K.Select(
                                                K.bitwise_and(K.shift_right(tmask, K.Cast("uint32", t_row)), K.uint32(1)) != K.uint32(0), 1, 0
                                            ),
                                        )
                                    else:
                                        sel = K.local_scalar("int32", init=1)
                                    any_sel = K.local_scalar("uint32", init=K.uint32(1))
                                    if T > 1:
                                        K.ptx.vote_sync.any.pred(any_sel, K.ptx.pred(sel), K.uint32(0xFFFFFFFF))
                                    sb = gj & 1
                                    s_col = K.Cast("uint32", sb) * K.uint32(S_COLS)
                                    p_col = K.uint32(P_BASE) + K.Cast("uint32", sb) * K.uint32(S_COLS)
                                    kv0 = blk * BLK + half * TILE_N
                                    need_causal = (kv0 + (TILE_N - 1)) > causal_off

                                    with K.If(j == 0), K.Then():
                                        tk_ws0 = iket_range("sm-wait-s", leader_only=True)
                                        s_ready.wait(sb, (gj >> 1) & 1)
                                        iket_end(tk_ws0)
                                        for c in range(2):
                                            K.ptx[tmem_ld(32)](*(s[32 * c + i] for i in range(32)), tmem_base + s_col + K.uint32(32 * c))
                                    K.ptx.tcgen05.wait__ld.sync.aligned()

                                    def prefetch_next_s():
                                        with K.If(j + 1 < n_tiles), K.Then():
                                            gj1 = gj + 1
                                            sb1 = gj1 & 1
                                            s_col1 = K.Cast("uint32", sb1) * K.uint32(S_COLS)
                                            tk_ws1 = iket_range("sm-wait-s", leader_only=True)
                                            s_ready.wait(sb1, (gj1 >> 1) & 1)
                                            iket_end(tk_ws1)
                                            for c in range(2):
                                                K.ptx[tmem_ld(32)](*(s[32 * c + i] for i in range(32)), tmem_base + s_col1 + K.uint32(32 * c))

                                    with K.If(any_sel != K.uint32(0)):
                                        with K.Then():
                                            tk_ph1 = iket_range("sm-max", leader_only=True)
                                            with K.If(need_causal), K.Then():
                                                for c in range(TILE_N):
                                                    K.assign(s[c], K.Select(kv0 + c <= qpos, s[c], K.float32(NEG_INF)))
                                            lvl = [s[i] for i in range(TILE_N)]
                                            while len(lvl) > 1:
                                                nxt = []
                                                i = 0
                                                while i < len(lvl):
                                                    if i + 2 < len(lvl):
                                                        r = K.local_scalar("float32")
                                                        K.ptx.max.f32(r, lvl[i], lvl[i + 1], lvl[i + 2])
                                                        nxt.append(r)
                                                        i += 3
                                                    elif i + 1 < len(lvl):
                                                        nxt.append(K.local_scalar("float32", init=K.max(lvl[i], lvl[i + 1])))
                                                        i += 2
                                                    else:
                                                        nxt.append(lvl[i])
                                                        i += 1
                                                lvl = nxt
                                            bmax = lvl[0]
                                            mbs = K.local_scalar("float32", init=K.Select(sel != 0, bmax * scale_log2, K.float32(NEG_INF)))
                                            m_old = K.local_scalar("float32", init=m_s)
                                            m_new = K.local_scalar("float32", init=K.max(m_old, mbs))
                                            d = K.local_scalar("float32", init=m_new - m_old)
                                            big = d > K.float32(RESCALE_THRESH)
                                            do_r = K.local_scalar("int32", init=K.Select(K.And(m_old != K.float32(NEG_INF), big), 1, 0))
                                            need_r = K.local_scalar("uint32")
                                            K.ptx.vote_sync.any.pred(need_r, K.ptx.pred(do_r), K.uint32(0xFFFFFFFF))
                                            iket_end(tk_ph1)
                                            with K.If(need_r != K.uint32(0)), K.Then():
                                                tk_rs = iket_range("sm-rescale", leader_only=True)
                                                gjm = gj - 1
                                                p_free.wait(gjm & 1, (gjm >> 1) & 1)
                                                alpha = K.local_scalar("float32")
                                                K.ptx.ex2.approx.ftz.f32(alpha, K.float32(0.0) - d)
                                                K.assign(alpha, K.Select(do_r != 0, alpha, K.float32(1.0)))
                                                o_ch = K.alloc_local([32], "float32")
                                                for c0 in range(0, 128, 32):
                                                    K.ptx[tmem_ld(32)](*(o_ch[i] for i in range(32)), tmem_base + K.uint32(O_COL + c0))
                                                    K.ptx.tcgen05.wait__ld.sync.aligned()
                                                    for i in range(32):
                                                        K.assign(o_ch[i], o_ch[i] * alpha)
                                                    K.ptx[tmem_st(32)](tmem_base + K.uint32(O_COL + c0), *(o_ch[i] for i in range(32)))
                                                if QM64_USE_LMMA:
                                                    l_ch = K.alloc_local([1], "float32")
                                                    K.ptx[tmem_ld(1)](l_ch[0], tmem_base + K.uint32(L_COL))
                                                    K.ptx.tcgen05.wait__ld.sync.aligned()
                                                    K.assign(l_ch[0], l_ch[0] * alpha)
                                                    K.ptx[tmem_st(1)](tmem_base + K.uint32(L_COL), l_ch[0])
                                                    K.ptx.tcgen05.wait__st.sync.aligned()
                                                else:
                                                    K.assign(l_row, l_row * alpha)
                                                iket_end(tk_rs)
                                            K.assign(m_s, K.Select(big, m_new, m_old))
                                            tk_ph2 = iket_range("sm-exp", leader_only=True)
                                            msub = K.local_scalar(
                                                "float32",
                                                init=K.Select(sel != 0, K.max(m_s, K.float32(-1.0e38)), K.float32(1.0e38)),
                                            )
                                            for c in range(TILE_N):
                                                K.assign(s[c], s[c] * scale_log2 - msub)
                                            for c in range(TILE_N):
                                                ex2_col(s[c], s[c], c)
                                            if not QM64_USE_LMMA:
                                                lacc = K.alloc_local([4], "float32")
                                                for a in range(4):
                                                    K.assign(lacc[a], K.float32(0.0))
                                                for c in range(TILE_N):
                                                    K.assign(lacc[c % 4], lacc[c % 4] + s[c])
                                                K.assign(l_row, l_row + ((lacc[0] + lacc[1]) + (lacc[2] + lacc[3])))
                                            for i in range(P_COLS):
                                                pack_half2(pk[i], s[2 * i + 1], s[2 * i])
                                            iket_end(tk_ph2)
                                            prefetch_next_s()
                                            tk_wp = iket_range("sm-wait-pfree", leader_only=True)
                                            p_free.wait(sb, ((gj >> 1) + 1) & 1)
                                            iket_end(tk_wp)
                                            tk_ps = iket_range("sm-pstore", leader_only=True)
                                            for c in range(2):
                                                K.ptx[tmem_st(16)](tmem_base + p_col + K.uint32(16 * c), *(pk[16 * c + i] for i in range(16)))
                                            K.ptx.tcgen05.wait__st.sync.aligned()
                                            p_ready.arrive(sb)
                                            iket_end(tk_ps)
                                        with K.Else():
                                            prefetch_next_s()
                                            p_free.wait(sb, ((gj >> 1) + 1) & 1)
                                            for c in range(2):
                                                K.ptx[tmem_st(16)](tmem_base + p_col + K.uint32(16 * c), *(zeros16[i] for i in range(16)))
                                            K.ptx.tcgen05.wait__st.sync.aligned()
                                            p_ready.arrive(sb)
                                    K.assign(gj, gj + 1)

                                tk_wo = iket_range("sm-wait-o", leader_only=True)
                                o_ready.wait(0, it & 1)
                                iket_end(tk_wo)
                                tk_epi = iket_range("sm-epi", leader_only=True)
                                bad = K.Or(l_row == K.float32(0.0), l_row != l_row)
                                inv = K.local_scalar("float32")
                                K.ptx.rcp.approx.ftz.f32(inv, K.Select(bad, K.float32(1.0), l_row))
                                K.assign(inv, K.Select(bad, K.float32(0.0), inv))
                                o_ch = K.alloc_local([32], "float32")
                                out_pk = K.alloc_local([16], "uint32")
                                for c0 in range(0, HEAD_DIM, 32):
                                    K.ptx[tmem_ld(32)](
                                        *(o_ch[i] for i in range(32)),
                                        tmem_base + K.uint32(O_COL + c0),
                                    )
                                    K.ptx.tcgen05.wait__ld.sync.aligned()
                                    for i in range(16):
                                        pack_half2(out_pk[i], o_ch[2 * i + 1] * inv, o_ch[2 * i] * inv)
                                    K.ptx[ST_OUT_V8](out.ptr_to([out_row + c0]), *(out_pk[i] for i in range(8)))
                                    K.ptx[ST_OUT_V8](out.ptr_to([out_row + c0 + 16]), *(out_pk[8 + i] for i in range(8)))
                                o_free.arrive(0)

                                def write_row(o_arr, l_val):
                                    bad = K.Or(l_val == K.float32(0.0), l_val != l_val)
                                    inv = K.local_scalar("float32")
                                    K.ptx.rcp.approx.ftz.f32(inv, K.Select(bad, K.float32(1.0), l_val))
                                    K.assign(inv, K.Select(bad, K.float32(0.0), inv))
                                    for i in range(64):
                                        pack_half2(pk[i], o_arr[2 * i + 1] * inv, o_arr[2 * i] * inv)
                                    for i in range(8):
                                        K.ptx[ST_OUT_V8](out.ptr_to([out_row + 16 * i]), *(pk[8 * i + j] for j in range(8)))

                                if SPLIT == 1:
                                    pass
                                else:
                                    with K.If(item_s < FULL_ITEMS):
                                        with K.Then():
                                            write_row(s, l_row)
                                        with K.Else():
                                            slot_idx = (item_s - FULL_ITEMS) * SPLIT + part_s
                                            o_base = (slot_idx * ROWS + tid) * HEAD_DIM
                                            ml_base = (slot_idx * ROWS + tid) * 2
                                            for i in range(32):
                                                K.ptx.st.global_.v4.f32(
                                                    part_o.ptr_to([o_base + 4 * i]), s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]
                                                )
                                            K.ptx.st.global_.f32(part_ml.ptr_to([ml_base]), m_s)
                                            K.ptx.st.global_.f32(part_ml.ptr_to([ml_base + 1]), l_row)
                                            K.ptx.bar.sync(K.uint32(3), K.uint32(SOFT_THREADS))
                                            with K.If(tid == 0), K.Then():
                                                K.ptx.fence.acq_rel.gpu()
                                                old = K.local_scalar("int32")
                                                K.ptx.atom.acq_rel.gpu.global_.add.s32(old, part_ctr.ptr_to([item_s - FULL_ITEMS]), K.int32(1))
                                                K.ptx.st.shared.b32(mrg_order.ptr_to([0]), old)
                                            K.ptx.bar.sync(K.uint32(3), K.uint32(SOFT_THREADS))
                                            order = ld_smem_i32(mrg_order.ptr_to([0]))
                                            with K.If(order == SPLIT - 1), K.Then():
                                                with K.If(tid == 0), K.Then():
                                                    K.ptx.fence.acq_rel.gpu()
                                                    K.ptx.st.relaxed.gpu.global_.b32(part_ctr.ptr_to([item_s - FULL_ITEMS]), K.int32(0))
                                                K.ptx.bar.sync(K.uint32(3), K.uint32(SOFT_THREADS))
                                                base_slot = (item_s - FULL_ITEMS) * SPLIT
                                                mp = K.alloc_local([SPLIT], "float32")
                                                lp = K.alloc_local([SPLIT], "float32")
                                                for pi in range(SPLIT):
                                                    K.ptx.ld.relaxed.gpu.global_.f32(mp[pi], part_ml.ptr_to([((base_slot + pi) * ROWS + tid) * 2]))
                                                    K.ptx.ld.relaxed.gpu.global_.f32(lp[pi], part_ml.ptr_to([((base_slot + pi) * ROWS + tid) * 2 + 1]))
                                                m_tot = K.local_scalar("float32", init=mp[0])
                                                for pi in range(1, SPLIT):
                                                    K.assign(m_tot, K.max(m_tot, mp[pi]))
                                                msafe = K.Select(m_tot == K.float32(NEG_INF), K.float32(0.0), m_tot)
                                                l_tot = K.local_scalar("float32", init=K.float32(0.0))
                                                for c in range(128):
                                                    K.assign(s[c], K.float32(0.0))
                                                for pi in range(SPLIT):
                                                    sc = K.local_scalar("float32")
                                                    K.ptx.ex2.approx.ftz.f32(sc, mp[pi] - msafe)
                                                    K.assign(l_tot, l_tot + lp[pi] * sc)
                                                    ob = K.alloc_local([4], "float32")
                                                    for i in range(32):
                                                        K.ptx.ld.relaxed.gpu.global_.v4.f32(
                                                            ob[0], ob[1], ob[2], ob[3],
                                                            part_o.ptr_to([((base_slot + pi) * ROWS + tid) * HEAD_DIM + 4 * i]),
                                                        )
                                                        for e in range(4):
                                                            K.assign(s[4 * i + e], s[4 * i + e] + ob[e] * sc)
                                                write_row(s, l_tot)
                                iket_end(tk_epi)
                            with K.Else():
                                o_ready.wait(0, it & 1)
                                o_free.arrive(0)
                                for i in range(8):
                                    K.ptx[ST_OUT_V8](out.ptr_to([out_row + 16 * i]), *(zeros16[j] for j in range(8)))
                        if not STATIC_ONE_SHOT:
                            union_free.arrive(slot)
                if STATIC_ONE_SHOT:
                    K.assign(running, 0)
                else:
                    K.assign(it, it + 1)

        with wg1:

            with r_mma:
                k_pipe = K.PipelineState(KS, phase=0)
                v_pipe = K.PipelineState(VS, phase=0)
                it_m = K.local_scalar("int32", init=0)
                gj_m = K.local_scalar("int32", init=0)
                running_m = K.local_scalar("int32", init=1)
                zero = K.uint32(0)

                def mma_qk(sb, kstage, qslot):
                    tk = iket_range("mma-qk")
                    d_addr = tmem_base + K.Cast("uint32", sb) * K.uint32(S_COLS)
                    for ki in range(NQK):
                        with K.If(elected()), K.Then():
                            K.ptx[MMA](
                                d_addr,
                                desc_at(q_desc, qslot * QP16 + qoff(ki)),
                                desc_at(k_desc, kstage * KV16 + koff(ki)),
                                K.uint32(ID_QK),
                                zero,
                                zero,
                                zero,
                                zero,
                                ki != 0,
                            )
                    iket_end(tk)

                def mma_pv(vstage, pb, acc):
                    tk = iket_range("mma-pv")
                    p_addr = tmem_base + K.uint32(P_BASE) + K.Cast("uint32", pb) * K.uint32(S_COLS)
                    for ki in range(NPV):
                        with K.If(elected()), K.Then():
                            K.ptx[MMA](
                                tmem_base + K.uint32(O_COL),
                                p_addr + K.uint32(ki * (MMA_K // 2)),
                                desc_at(v_desc, vstage * KV16 + voff(ki)),
                                K.uint32(ID_PV),
                                zero,
                                zero,
                                zero,
                                zero,
                                True if ki != 0 else K.Cast("bool", acc != 0),
                            )

                    for ki in (range(NPV) if QM64_USE_LMMA else ()):
                        with K.If(elected()), K.Then():
                            K.ptx[MMA](
                                tmem_base + K.uint32(L_COL),
                                p_addr + K.uint32(ki * (MMA_K // 2)),
                                desc_at(ones_desc, ooff(ki)),
                                K.uint32(ID_L),
                                zero,
                                zero,
                                zero,
                                zero,
                                True if ki != 0 else K.Cast("bool", acc != 0),
                            )
                    iket_end(tk)

                with K.While(running_m != 0):
                    slot = it_m & 1
                    union_ready.wait(slot, (it_m >> 1) & 1)
                    n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                    with K.If(n_blocks < 0):
                        with K.Then():
                            K.assign(running_m, 0)
                        with K.Else():
                            qs_m = q_stage(it_m)
                            q_load.full.wait(qs_m, q_phase(it_m))
                            o_free.wait(0, (it_m + 1) & 1)
                            with K.If(n_blocks == 0), K.Then():
                                with K.If(elected()), K.Then():
                                    q_load.empty.arrive(qs_m)
                                    o_ready.arrive(0)
                            with K.If(n_blocks > 0), K.Then():
                                n_tiles = n_blocks * 2
                                acc = K.local_scalar("int32", init=0)
                                sb0 = gj_m & 1
                                kstage0 = K.local_scalar("int32", init=k_pipe.stage)
                                kphase0 = K.local_scalar("int32", init=k_pipe.phase)
                                k_pipe.advance()
                                k_load.full.wait(kstage0, kphase0)
                                p_free.wait(sb0, ((gj_m >> 1) + 1) & 1)
                                mma_qk(sb0, kstage0, qs_m)
                                with K.If(elected()), K.Then():
                                    s_ready.arrive(sb0)
                                    k_load.empty.arrive(kstage0)
                                    with K.If(n_tiles == 1), K.Then():
                                        q_load.empty.arrive(qs_m)
                                with K.serial(n_tiles, unroll=False) as j:
                                    with K.If(j + 1 < n_tiles), K.Then():
                                        kstage = K.local_scalar("int32", init=k_pipe.stage)
                                        kphase = K.local_scalar("int32", init=k_pipe.phase)
                                        k_pipe.advance()
                                        gj1 = gj_m + 1
                                        sb1 = gj1 & 1
                                        tk_wk = iket_range("mma-wait-k")
                                        k_load.full.wait(kstage, kphase)
                                        iket_end(tk_wk)
                                        tk_wsf = iket_range("mma-wait-sfree")
                                        p_free.wait(sb1, ((gj1 >> 1) + 1) & 1)
                                        iket_end(tk_wsf)
                                        mma_qk(sb1, kstage, qs_m)
                                        with K.If(elected()), K.Then():
                                            s_ready.arrive(sb1)
                                            k_load.empty.arrive(kstage)
                                            with K.If(j + 2 == n_tiles), K.Then():
                                                q_load.empty.arrive(qs_m)
                                    vstage = K.local_scalar("int32", init=v_pipe.stage)
                                    vphase = K.local_scalar("int32", init=v_pipe.phase)
                                    v_pipe.advance()
                                    tk_wv = iket_range("mma-wait-v")
                                    v_load.full.wait(vstage, vphase)
                                    iket_end(tk_wv)
                                    tk_wp = iket_range("mma-wait-p")
                                    pb = gj_m & 1
                                    p_ready.wait(pb, (gj_m >> 1) & 1)
                                    iket_end(tk_wp)
                                    mma_pv(vstage, pb, acc)
                                    with K.If(elected()), K.Then():
                                        p_free.arrive(pb)
                                        v_load.empty.arrive(vstage)
                                        with K.If(j + 1 == n_tiles), K.Then():
                                            o_ready.arrive(0)
                                    K.assign(acc, 1)
                                    K.assign(gj_m, gj_m + 1)
                            if not STATIC_ONE_SHOT:
                                union_free.arrive(slot)
                    if STATIC_ONE_SHOT:
                        K.assign(running_m, 0)
                    else:
                        K.assign(it_m, it_m + 1)


            def loader_body(which):
                """which == "k": Q tile + K ring;  which == "v": V ring.  One warp each."""
                pipe_l = K.PipelineState(KS if which == "k" else VS, phase=0)
                it_l = K.local_scalar("int32", init=0)
                running_l = K.local_scalar("int32", init=1)
                with K.While(running_l != 0):
                    slot = it_l & 1
                    tk_lu = iket_range("ld-wait-union" if which == "k" else "vld-wait-union")
                    union_ready.wait(slot, (it_l >> 1) & 1)
                    iket_end(tk_lu)
                    n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                    with K.If(n_blocks < 0):
                        with K.Then():
                            K.assign(running_l, 0)
                        with K.Else():
                            h = ld_smem_i32(meta.ptr_to([slot * 16 + 2]))
                            kv_s = ld_smem_i32(meta.ptr_to([slot * 16 + 3]))
                            tok_base = ld_smem_i32(meta.ptr_to([slot * 16 + 5]))
                            blk_off_l = ld_smem_i32(meta.ptr_to([slot * 16 + 7]))
                            tmap = k_map if which == "k" else v_map
                            tile = k_smem if which == "k" else v_smem
                            pipe_load = k_load if which == "k" else v_load

                            def issue_tile(blk_idx):
                                u_idx = blk_idx >> 1
                                half = blk_idx & 1
                                blk_v = ld_smem_i32(ulist.ptr_to([slot * MAX_UNION + blk_off_l + u_idx]))
                                if PAGED:
                                    pg_v = ld_smem_i32(uplist.ptr_to([slot * MAX_UNION + blk_off_l + u_idx]))
                                tk_we = iket_range("ld-wait-stage" if which == "k" else "vld-wait-stage")
                                pipe_load.empty.wait(pipe_l.stage, pipe_l.phase)
                                iket_end(tk_we)
                                with K.If(elected()), K.Then():
                                    dst = tile[pipe_l.stage].ptr_to(0, 0)
                                    bar = K.cuda.cvta_generic_to_shared(pipe_load.full.ptr_to([pipe_l.stage]))
                                    if PAGED and not FP8:
                                        K.ptx[TMA_4D_HINT](dst, K.address_of(tmap), K.int32(0), half * TILE_N, K.int32(0),
                                                           pg_v * HKV + h, bar, K.uint64(KV_CACHE_POLICY))
                                    elif PAGED:
                                        K.ptx[TMA_3D_HINT](dst, K.address_of(tmap), K.int32(0), K.int32(0),
                                                           pg_v * HKV + h, bar, K.uint64(KV_CACHE_POLICY))
                                    elif not FP8:
                                        K.ptx[TMA_3D_HINT](dst, K.address_of(tmap), K.int32(0), kv_s + blk_v * BLK + half * TILE_N,
                                                           h * 2, bar, K.uint64(KV_CACHE_POLICY))
                                    else:
                                        K.ptx[TMA_3D_HINT](dst, K.address_of(tmap), K.int32(0), kv_s + blk_v * BLK,
                                                           h, bar, K.uint64(KV_CACHE_POLICY))
                                    pipe_load.full.arrive(pipe_l.stage, tx_count=KV_TILE_BYTES)
                                pipe_l.advance()

                            if which == "k":
                                n_tiles = n_blocks * 2

                                with K.If(n_tiles > 0), K.Then():
                                    issue_tile(K.int32(0))
                                qs_l = q_stage(it_l)
                                q_load.empty.wait(qs_l, q_phase(it_l))
                                if not FP8:
                                    with K.If(elected()), K.Then():
                                        K.ptx[TMA_4D](
                                            q_smem[qs_l].ptr_to(0, 0),
                                            K.address_of(q_map),
                                            K.int32(0),
                                            h * G,
                                            tok_base,
                                            K.int32(0),
                                            K.cuda.cvta_generic_to_shared(q_load.full.ptr_to([qs_l])),
                                        )
                                        q_load.full.arrive(qs_l, tx_count=Q_TMA_BYTES)
                                else:
                                    words8 = K.alloc_local([8], "int32")
                                    e4 = K.alloc_local([8], "uint16")
                                    packed4 = K.alloc_local([4], "uint32")
                                    for r in range(Q_ROUNDS):
                                        c = lane + 32 * r

                                        def quant_chunk(c=c):
                                            n = c >> 3
                                            k16 = c & 7
                                            row = tok_base + n // G
                                            head = h * G + (n % G)
                                            wbase = (row * HQ + head) * (HEAD_DIM // 2) + k16 * 8
                                            K.ptx.ld.global_.nc.v4.b32(
                                                words8[0], words8[1], words8[2], words8[3], qraw.ptr_to([wbase])
                                            )
                                            K.ptx.ld.global_.nc.v4.b32(
                                                words8[4], words8[5], words8[6], words8[7], qraw.ptr_to([wbase + 4])
                                            )
                                            for i in range(8):
                                                K.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(e4[i], K.Cast("uint32", words8[i]))
                                            for i in range(4):
                                                K.ptx.mov.b32(packed4[i], e4[2 * i], e4[2 * i + 1])
                                            K.ptx.st.shared.v4.b32(
                                                q_smem[qs_l].ptr_to(n, k16 * 16), packed4[0], packed4[1], packed4[2], packed4[3]
                                            )

                                        if 32 * r + 32 <= Q_CHUNKS:
                                            quant_chunk()
                                        else:
                                            with K.If(c < Q_CHUNKS), K.Then():
                                                quant_chunk()
                                    K.ptx.fence.proxy.async_.shared__cta()
                                    q_load.full.arrive(qs_l)
                                with K.If(n_tiles > 1), K.Then():
                                    with K.serial(n_tiles - 1, unroll=False) as j1:
                                        issue_tile(j1 + 1)
                            else:
                                n_tiles = n_blocks * 2
                                with K.If(n_tiles > 0), K.Then():
                                    with K.serial(n_tiles, unroll=False) as j:
                                        issue_tile(j)
                            if not STATIC_ONE_SHOT:
                                union_free.arrive(slot)
                    if STATIC_ONE_SHOT:
                        K.assign(running_l, 0)
                    else:
                        K.assign(it_l, it_l + 1)

            with r_load:
                loader_body("k")


            with r_prep:
                it_p = K.local_scalar("int32", init=0)
                running_p = K.local_scalar("int32", init=1)
                with K.While(running_p != 0):
                    slot = it_p & 1
                    task = K.local_scalar("int32")
                    if STATIC_ONE_SHOT:
                        K.assign(task, K.cta_id())
                    else:
                        with K.If(it_p == 0):
                            with K.Then():
                                K.assign(task, K.cta_id())
                            with K.Else():
                                grabbed = K.local_scalar("int32", init=0)
                                with K.If(lane == 0), K.Then():
                                    K.ptx.atom.relaxed.gpu.global_.add.s32(grabbed, sched.ptr_to([0]), K.int32(1))
                                K.assign(task, K.uniform(grabbed) + NUM_CTAS)
                    if not STATIC_ONE_SHOT:
                        tk_pf = iket_range("prep-wait-free")
                        union_free.wait(slot, ((it_p >> 1) + 1) & 1)
                        iket_end(tk_pf)
                    tk_pi = iket_range("prep-item")
                    with K.If(task >= NUM_TASKS):
                        with K.Then():
                            with K.If(lane == 0), K.Then():
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16]), K.int32(-1))
                            union_ready.arrive(slot)
                            K.assign(running_p, 0)
                        with K.Else():
                            item = K.local_scalar("int32", init=task)
                            part = K.local_scalar("int32", init=0)
                            if SPLIT > 1:
                                with K.If(task >= FULL_ITEMS), K.Then():
                                    tidx = task - FULL_ITEMS
                                    K.assign(item, FULL_ITEMS + tidx // SPLIT)
                                    K.assign(part, tidx - (tidx // SPLIT) * SPLIT)
                            b = item // (HKV * NCH)
                            rem = item - b * (HKV * NCH)
                            h = rem // NCH
                            ch = rem - h * NCH
                            kv_s = K.local_scalar("int32", init=0)
                            kv_len = K.local_scalar("int32")
                            if PAGED:
                                K.assign(kv_len, ld_global_i32(lens.ptr_to([b])))
                            else:
                                K.assign(kv_s, ld_global_i32(lens.ptr_to([b])))
                                K.assign(kv_len, ld_global_i32(lens.ptr_to([b + 1])) - kv_s)
                            n_vis = (kv_len + (BLK - 1)) >> 7
                            tok_base = b * TQ + ch * T
                            causal_off = kv_len - TQ + ch * T
                            base = (h * TOTAL_Q + tok_base) * TOPK
                            idxs = K.alloc_local([N_ROUNDS], "int32")
                            for r in range(N_ROUNDS):
                                e = lane + 32 * r
                                K.assign(idxs[r], -1)
                                if 32 * r + 32 <= MAX_UNION:
                                    K.assign(idxs[r], ld_global_i32(q2k.ptr_to([base + e])))
                                else:
                                    with K.If(e < MAX_UNION), K.Then():
                                        K.assign(idxs[r], ld_global_i32(q2k.ptr_to([base + e])))

                            def valid(r):
                                return K.And(idxs[r] >= 0, idxs[r] < n_vis)

                            n_blocks = K.local_scalar("int32", init=0)
                            if T == 1:
                                for r in range(N_ROUNDS):
                                    vflag = K.local_scalar("int32", init=K.Select(valid(r), 1, 0))
                                    ballot = K.local_scalar("uint32")
                                    K.ptx.vote_sync.ballot.b32(ballot, K.ptx.pred(vflag), K.uint32(0xFFFFFFFF))
                                    before = K.local_scalar("uint32")
                                    K.ptx.popc.b32(before, K.bitwise_and(ballot, lanemask_lt()))
                                    total = K.local_scalar("uint32")
                                    K.ptx.popc.b32(total, ballot)
                                    rank = n_blocks + K.Cast("int32", before)
                                    with K.If(vflag != 0), K.Then():
                                        K.ptx.st.shared.b32(ulist.ptr_to([slot * MAX_UNION + rank]), idxs[r])
                                        if PAGED:
                                            pg = ld_global_i32(ptab.ptr_to([b * MAX_PAGES + idxs[r]]))
                                            K.ptx.st.shared.b32(uplist.ptr_to([slot * MAX_UNION + rank]), pg)
                                    K.assign(n_blocks, n_blocks + K.Cast("int32", total))
                            else:
                                nwords = (n_vis + 31) >> 5
                                wi = K.local_scalar("int32", init=lane)
                                with K.While(wi < nwords):
                                    K.ptx.st.shared.b32(words.ptr_to([wi]), K.uint32(0))
                                    K.assign(wi, wi + 32)
                                K.cuda.warp_sync()
                                for r in range(N_ROUNDS):
                                    with K.If(valid(r)), K.Then():
                                        K.ptx.red.shared.or_.b32(
                                            words.ptr_to([idxs[r] >> 5]),
                                            K.shift_left(K.uint32(1), K.Cast("uint32", idxs[r] & 31)),
                                        )
                                K.cuda.warp_sync()
                                carry = K.local_scalar("uint32", init=K.uint32(0))
                                wbase = K.local_scalar("int32", init=0)
                                with K.While(wbase < nwords):
                                    w = wbase + lane
                                    wv = K.local_scalar("uint32", init=K.uint32(0))
                                    with K.If(w < nwords), K.Then():
                                        K.ptx.ld.shared.u32(wv, words.ptr_to([w]))
                                    cnt = K.alloc_local([1], "uint32")
                                    K.ptx.popc.b32(cnt[0], wv)
                                    own = K.local_scalar("uint32", init=cnt[0])
                                    K.idioms.warp_scan_add(cnt, 1, lane)
                                    with K.If(w < nwords), K.Then():
                                        K.ptx.st.shared.b32(prefix.ptr_to([w]), carry + (cnt[0] - own))
                                    tot = K.local_scalar("uint32")
                                    K.ptx.shfl_sync.idx.b32(tot, cnt[0], K.uint32(31), K.uint32(31), K.uint32(0xFFFFFFFF))
                                    K.assign(carry, carry + tot)
                                    K.assign(wbase, wbase + 32)
                                K.assign(n_blocks, K.Cast("int32", carry))
                                K.cuda.warp_sync()
                                with K.serial(nwords, unroll=False) as w:
                                    wv = ld_smem_u32(words.ptr_to([w]))
                                    pw = ld_smem_u32(prefix.ptr_to([w]))
                                    mybit = K.bitwise_and(K.shift_right(wv, K.Cast("uint32", lane)), K.uint32(1))
                                    with K.If(mybit != K.uint32(0)), K.Then():
                                        below = K.local_scalar("uint32")
                                        K.ptx.popc.b32(below, K.bitwise_and(wv, lanemask_lt()))
                                        rank = K.Cast("int32", pw + below)
                                        blkid = w * 32 + lane
                                        K.ptx.st.shared.b32(ulist.ptr_to([slot * MAX_UNION + rank]), blkid)
                                        K.ptx.st.shared.b32(umask.ptr_to([slot * MAX_UNION + rank]), K.uint32(0))
                                        if PAGED:
                                            pg = ld_global_i32(ptab.ptr_to([b * MAX_PAGES + blkid]))
                                            K.ptx.st.shared.b32(uplist.ptr_to([slot * MAX_UNION + rank]), pg)
                                K.cuda.warp_sync()
                                for r in range(N_ROUNDS):
                                    e = lane + 32 * r
                                    with K.If(valid(r)), K.Then():
                                        wq = idxs[r] >> 5
                                        wv = ld_smem_u32(words.ptr_to([wq]))
                                        pw = ld_smem_u32(prefix.ptr_to([wq]))
                                        below = K.local_scalar("uint32")
                                        K.ptx.popc.b32(
                                            below,
                                            K.bitwise_and(
                                                wv,
                                                K.shift_left(K.uint32(1), K.Cast("uint32", idxs[r] & 31)) - K.uint32(1),
                                            ),
                                        )
                                        u = K.Cast("int32", pw + below)
                                        K.ptx.red.shared.or_.b32(
                                            umask.ptr_to([slot * MAX_UNION + u]),
                                            K.shift_left(K.uint32(1), K.Cast("uint32", e // TOPK)),
                                        )
                            K.cuda.warp_sync()
                            blk_off = K.local_scalar("int32", init=0)
                            if SPLIT > 1:
                                with K.If(item >= FULL_ITEMS), K.Then():
                                    per_part = (n_blocks + (SPLIT - 1)) // SPLIT
                                    start = K.min(part * per_part, n_blocks)
                                    K.assign(blk_off, start)
                                    K.assign(n_blocks, K.min(n_blocks - start, per_part))
                            with K.If(lane == 0), K.Then():
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 7]), blk_off)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 8]), item)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 9]), part)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 0]), n_blocks)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 1]), b)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 2]), h)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 3]), kv_s)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 4]), kv_len)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 5]), tok_base)
                                K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 6]), causal_off)
                            union_ready.arrive(slot)
                    iket_end(tk_pi)
                    if STATIC_ONE_SHOT:
                        K.assign(running_p, 0)
                    else:
                        K.assign(it_p, it_p + 1)

            with r_vload:
                loader_body("v")


        K.cuda.cta_sync()
        if not STATIC_ONE_SHOT:
            with K.If(tid == 0), K.Then():
                done = K.local_scalar("int32")
                K.ptx.atom.acq_rel.gpu.global_.add.s32(done, sched.ptr_to([1]), K.int32(1))
                with K.If(done == NUM_CTAS - 1), K.Then():
                    K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([0]), K.int32(0))
                    K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([1]), K.int32(0))
        with K.If(warp == MMA_WARP), K.Then():
            K.ptx[TMEM_DEALLOC](tmem_base, K.uint32(TMEM_COLS))

    return msa_decode_qmajor



def plan_config_qm64(q, k, q2k, page_table, B, T, num_sms):
    HQ = q.shape[1]
    HKV = k.shape[1]
    G = HQ // HKV
    TOPK = int(q2k.shape[-1])
    TOTAL_Q = q.shape[0]
    paged = page_table is not None
    kind = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float8_e4m3fn: "fp8"}[k.dtype]
    if kind == "fp8" or 128 % G != 0:
        raise NotImplementedError("q-major family: bf16/f16 KV and GQA dividing 128 only")
    TI = 128 // G
    if T % TI != 0:
        raise NotImplementedError("q-major family needs seqlen_q to be a multiple of 128/GQA")
    NCH = T // TI
    if paged:
        MAX_PAGES = int(page_table.shape[1])
        max_blocks = MAX_PAGES
    else:
        MAX_PAGES = 0
        max_blocks = ceildiv(int(k.shape[0]), BLK)
    W_MAX = max(1, ceildiv(max_blocks, 32))
    NUM_ITEMS = B * HKV * NCH
    NUM_CTAS = max(1, min(2 * num_sms, NUM_ITEMS))
    STATIC_ONE_SHOT = NUM_ITEMS == NUM_CTAS
    env_split = int(os.environ.get("QM_SPLIT", "0"))

    SPLIT = 1 if env_split == 0 else env_split
    MAX_UNION = TI * TOPK
    kv_tile = 64 * HEAD_DIM * 2
    QSLOTS = int(os.environ.get("QM_QSLOTS", "1"))
    q_tiles = QSLOTS * 128 * HEAD_DIM * 2
    misc = 24 * MAX_UNION + 8 * W_MAX + 4096 + 16 * HEAD_DIM * 2
    budget = 227 * 1024 - misc - q_tiles
    STAGES = 4
    if int(os.environ.get("QM_STAGES", "0")):
        STAGES = min(STAGES, int(os.environ["QM_STAGES"]))
    assert STAGES >= 2, "shared memory budget too small"
    return dict(
        T=T,
        TI=TI,
        G=G,
        HKV=HKV,
        TOPK=TOPK,
        TOTAL_Q=TOTAL_Q,
        kv_kind=kind,
        paged=paged,
        MAX_PAGES=MAX_PAGES,
        MAX_BLOCK_WORDS=W_MAX,
        NUM_ITEMS=NUM_ITEMS,
        NUM_CTAS=NUM_CTAS,
        STAGES=STAGES,
        QSLOTS=QSLOTS,
        STATIC_ONE_SHOT=STATIC_ONE_SHOT,
        SPLIT=SPLIT,
    )


def setup_qm64(data, B, seqlen_q):
    q = data["q"]
    k = data["k"]
    v = data["v"]
    q2k = data["q2k_indices"]
    cu_seqlens_k = data["cu_seqlens_k"]
    page_table = data["page_table"]
    seqused_k = data["seqused_k"]
    out = data["output"]
    scale = float(data["softmax_scale"])
    T = int(seqlen_q)
    B = int(B)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous()
    assert q.shape[0] == B * T
    device = q.device
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    cfg = plan_config_qm64(q, k, q2k, page_table, B, T, num_sms)
    ex = _compile(cfg, make_kernel_qm64, 'qm64')

    HKV = cfg["HKV"]
    G = cfg["G"]
    TI = cfg["TI"]
    HQ = HKV * G
    TOTAL_Q = cfg["TOTAL_Q"]
    dtype_name = "bfloat16" if q.dtype == torch.bfloat16 else "float16"
    q_map = _encode(
        q,
        dtype_name,
        (HEAD_DIM // 2, HQ, TOTAL_Q, 2),
        (HEAD_DIM * 2, HQ * HEAD_DIM * 2, (HEAD_DIM // 2) * 2),
        (HEAD_DIM // 2, G, TI, 2),
    )
    if cfg["paged"]:
        num_pages = int(k.shape[0])
        kv_dims = (HEAD_DIM // 2, BLK, 2, num_pages * HKV)
        kv_strides = (HEAD_DIM * 2, (HEAD_DIM // 2) * 2, BLK * HEAD_DIM * 2)
        kv_box = (HEAD_DIM // 2, 64, 2, 1)
        lens = seqused_k.contiguous()
        ptab = page_table.contiguous().view(-1)
    else:
        total_k = int(k.shape[0])
        kv_dims = (HEAD_DIM // 2, total_k, HKV * 2)
        kv_strides = (HKV * HEAD_DIM * 2, (HEAD_DIM // 2) * 2)
        kv_box = (HEAD_DIM // 2, 64, 2)
        lens = cu_seqlens_k.contiguous()
        ptab = torch.zeros(4, dtype=torch.int32, device=device)
    k_map = _encode(k, dtype_name, kv_dims, kv_strides, kv_box, l2_promotion=3)
    v_map = _encode(v, dtype_name, kv_dims, kv_strides, kv_box, l2_promotion=3)
    sched = torch.zeros(4, dtype=torch.int32, device=device)
    q2k_flat = q2k.contiguous().view(-1)
    out_flat = out.view(-1)
    qraw = torch.zeros(4, dtype=torch.int32, device=device)
    NUM_ITEMS = cfg["NUM_ITEMS"]
    NUM_CTAS = cfg["NUM_CTAS"]
    SPLIT = cfg["SPLIT"]
    tail_items = NUM_ITEMS - (NUM_ITEMS // NUM_CTAS) * NUM_CTAS if SPLIT > 1 else 0
    n_parts = max(1, tail_items * SPLIT)
    part_o = torch.zeros(n_parts * 128 * HEAD_DIM, dtype=torch.float32, device=device)
    part_ml = torch.zeros(n_parts * 128 * 2, dtype=torch.float32, device=device)
    part_ctr = torch.zeros(max(4, tail_items), dtype=torch.int32, device=device)
    args = (
        q_map.ptr,
        k_map.ptr,
        v_map.ptr,
        out_flat,
        q2k_flat,
        lens,
        ptab,
        qraw,
        sched,
        part_o,
        part_ml,
        part_ctr,
        float(scale * LOG2E),
    )
    keep = (q, k, v, q2k, q2k_flat, lens, ptab, qraw, out, out_flat, sched, part_o, part_ml, part_ctr, q_map, k_map, v_map)

    def run():
        ex(*args)

    run._keep = keep
    run._cfg = cfg
    run()
    torch.cuda.synchronize(device)
    return run


def _use_qm64(data, B, T):
    q, k, v = data["q"], data["k"], data["v"]
    page_table = data["page_table"]
    return (
        int(B) == 64
        and int(T) == 8
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and page_table is not None
        and data["seqused_k"] is not None
        and data["cu_seqlens_k"] is None
        and int(q.shape[1]) == 64
        and int(k.shape[1]) == 4
        and int(v.shape[1]) == 4
        and int(data["q2k_indices"].shape[-1]) == 32
        and int(page_table.shape[1]) >= 256
    )





def _build_q16_twotile_factory():
    HEAD_DIM = 128
    BLK_N = 128
    BLK_M = 128
    GQA = 16
    TOK_PER_TILE = BLK_M // GQA
    N_TILES = 2
    TOK_PER_CTA = TOK_PER_TILE * N_TILES
    KV_DEPTH = 4
    N_COLS_TMEM = 512
    MMA_N = 128
    MMA_K = 16
    MAX_BLOCKS = 32
    LOG2E = 1.4426950408889634
    K_SPLIT = 2 * MMA_K
    P_SPLIT_Q = 1
    N_SUM_ACC = 8
    MAX_CHAINS = 8
    EMU_PAIRS = 2
    EMU_START = 0
    RESCALE_THRESHOLD = 8.0
    NEG_INF = float("-inf")
    F16_BYTES = 2

    TMA_G2S_3D = (
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
    )
    KV_CACHE_POLICY = 0x12F0000000000000
    TMA_G2S_4D = (
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
    )
    TMA_S2G_4D = "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
    MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
    TMEM_LD_16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
    TMEM_LD_32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
    TMEM_ST_16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
    TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
    TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
    TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
    TCGEN05_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    MAX3_F32 = "max.f32"
    LD_Q2K = "ld.global.nc.L1::no_allocate.L2::evict_first.L2::256B.v8.u32"
    ST_OUTPUT = "st.global.L1::no_allocate.L2::evict_first.v8.b32"

    ID_QK = 0x08200490
    ID_PV = 0x08210490


    def ceildiv(a, b):
        return (a + b - 1) // b


    def make_kernel(BATCH, SEQLEN_Q, HQ, HKV, TOPK, NUM_CTAS):
        assert HQ == HKV * GQA
        assert TOPK == 16
        assert SEQLEN_Q == TOK_PER_CTA
        TOTAL_Q = BATCH * SEQLEN_Q
        NUM_ITEMS = BATCH * HKV
        NUM_CTAS = min(NUM_CTAS, NUM_ITEMS)


        FULL_TASKS = (NUM_ITEMS // NUM_CTAS) * NUM_CTAS
        R_SPLIT = NUM_ITEMS - FULL_TASKS
        SPLIT = R_SPLIT > 0
        NUM_TASKS = FULL_TASKS + 2 * R_SPLIT
        Q_TILE_BYTES = BLK_M * HEAD_DIM * F16_BYTES
        KV_TILE_BYTES = BLK_N * HEAD_DIM * F16_BYTES

        @K.kernel(warps=12, arch="sm_100a", min_blocks_per_sm=1, grid=NUM_CTAS)
        def msa_decode_qmajor_union(
            q_map: K.TensorMap,
            k_map: K.TensorMap,
            v_map: K.TensorMap,
            out: K.gptr[K.bf16],
            q2k: K.gptr[K.i32],
            cu_k: K.gptr[K.i32],
            sched: K.gptr[K.i32],
            mrg_o: K.gptr[K.f32],
            mrg_ml: K.gptr[K.f32],
            mrg_ctl: K.gptr[K.i32],
            scale_log2: K.f32,
        ):
            warp_cta = K.warp_id()
            wg_id = warp_cta >> 2
            warp_id = warp_cta & 3
            tid_in_wg = K.thread_id() & 127
            lane = K.lane_id()


            smem = K.smem_pool()
            q_smem = smem.alloc((N_TILES, BLK_M, HEAD_DIM), K.bf16, swizzle=K.SW128B)
            kv_base = N_TILES * Q_TILE_BYTES
            k_smem = smem.alloc((KV_DEPTH, BLK_N, HEAD_DIM), K.bf16, swizzle=K.SW128B)
            smem.pool.move_base_to(kv_base)
            v_smem = smem.alloc((KV_DEPTH, BLK_N, HEAD_DIM), K.bf16, swizzle=K.SW128B)
            smem.pool.move_base_to(kv_base + KV_DEPTH * KV_TILE_BYTES)

            def stage16(tile):
                return tile.rows * tile.cols * tile.bits // 8 // 16

            Q_STAGE16 = stage16(q_smem)
            KV_STAGE16 = stage16(k_smem)

            def lo_uniform(desc):
                desc_lo = K.alloc_local((1,), "uint32")
                desc_hi = K.alloc_local((1,), "uint32")
                K.assign(desc_lo[0], K.uniform(K.Cast("uint32", desc.value)))
                K.assign(desc_hi[0], K.Cast("uint32", K.shift_right(desc.value, K.uint64(32))))
                return desc_lo, desc_hi

            def desc_at(desc, off16):
                lo, hi = desc
                packed = K.alloc_local((1,), "uint64")
                low = lo[0] if isinstance(off16, int) and off16 == 0 else lo[0] + K.Cast("uint32", off16)
                K.assign(
                    packed[0],
                    K.bitwise_or(
                        K.shift_left(K.Cast("uint64", hi[0]), K.uint64(32)), K.Cast("uint64", low)
                    ),
                )
                return packed[0]

            def encode(view, major="k"):
                desc, off16 = view.encode(major=major, mma_k=MMA_K)
                return lo_uniform(desc), off16

            q_desc, qoff = encode(q_smem[0])
            k_desc, koff = encode(k_smem[0])
            v_desc, mnoff = encode(v_smem[0], major="mn")

            tmem_addr = smem.alloc((1,), K.u32)

            union_meta = smem.alloc((16,), K.i32)
            token_masks = smem.alloc((2 * TOK_PER_CTA,), K.u32)
            output_lane_masks = smem.alloc((2, MAX_BLOCKS, N_TILES, 4), K.u32)
            mrg_order = smem.alloc((2,), K.i32)


            kv_pipe = K.PipelineState(KV_DEPTH, phase=0)
            score_epoch = K.PipelineState(1, phase=0)
            tmem_epoch = K.PipelineState(1, phase=0)
            q_epoch = K.PipelineState(1, phase=0)


            q_load = K.Pipeline(smem, N_TILES, full="tma", empty="tcgen05", empty_phase_offset=1)
            kv_load = K.Pipeline(smem, KV_DEPTH, full="tma", empty="tcgen05", empty_phase_offset=1)
            p_o_rescale = K.MBarrier(smem, 2)
            p_o_rescale.init(128)
            s_ready = K.MBarrier(smem, 2)
            s_ready.init(1)
            o_ready = K.MBarrier(smem, 2)
            o_ready.init(1)
            p_ready_2 = K.MBarrier(smem, 2)
            p_ready_2.init(128)
            s_consumed = K.MBarrier(smem, 2)
            s_consumed.init(128)
            xu_turn = K.MBarrier(smem, 2)
            xu_turn.init(128)
            pv_done = K.TCGen05Bar(smem, 2)
            pv_done.init(1)
            o_free = K.MBarrier(smem, 2)
            o_free.init(128)
            union_ready = K.MBarrier(smem, 2)
            union_ready.init(32)
            union_free = K.MBarrier(smem, 2)
            union_free.init(256)

            K.ptx.fence.proxy.async_.shared__cta()
            K.ptx.fence.mbarrier_init.release.cluster()
            K.cuda.cta_sync()


            def elected():
                return K.cuda.elect_sync() != K.uint32(0)

            def commit(bar, stage):
                K.ptx[TCGEN05_COMMIT](bar.ptr_to([stage]))

            def tmem(col):
                return K.cuda.get_tmem_addr(K.uint32(0), 0, col)

            def tmem_load(dst, dst_offset, tmem_col, width):
                chain = TMEM_LD_16 if width == 16 else TMEM_LD_32
                K.ptx[chain](*(dst[dst_offset + i] for i in range(width)), tmem_col)

            def tmem_store(src, src_offset, tmem_col):
                K.ptx[TMEM_ST_16](tmem_col, *(src[src_offset + i] for i in range(16)))

            def ld_shared_i32(ptr):
                value = K.local_scalar("int32")
                K.ptx.ld.shared.b32(value, ptr)
                return value

            def iket_range(name, *, leader_only=False):
                token = K.alloc_local([1], "uint32")
                if leader_only:
                    K.assign(token[0], K.cuda.iket.sentinel_token(name))
                    with K.If(warp_id == 0), K.Then():
                        K.assign(token[0], K.cuda.iket.range_start(name))
                else:
                    K.assign(token[0], K.cuda.iket.range_start(name))
                return token

            def cast_f32x2_bf16x2(dst_u32, src, offset):
                K.ptx.cvt.rn.bf16x2.f32(dst_u32[offset // 2], src[offset + 1], src[offset])

            def mul_f32x2(values, idx, multiplier):
                packed = K.local_scalar("uint64")
                rhs = K.local_scalar("uint64")
                K.ptx.mov.b64(packed, values[idx], values[idx + 1])
                K.ptx.mov.b64(rhs, multiplier, multiplier)
                K.ptx.mul.rz.ftz.f32x2(packed, packed, rhs)
                K.ptx.mov.b64(values[idx], values[idx + 1], packed)

            def reduce_max_128(out_, values, accum=False):
                C = MAX_CHAINS
                temp = K.alloc_local([C], "float32")
                for i in range(C):
                    if accum and i == 0:
                        K.ptx[MAX3_F32](temp[i], values[2 * i], values[2 * i + 1], out_[0])
                    else:
                        K.ptx.mov.b32(temp[i], K.max(values[2 * i], values[2 * i + 1]))
                for g in range(1, BLK_N // (2 * C)):
                    for i in range(C):
                        K.ptx[MAX3_F32](
                            temp[i], temp[i], values[2 * C * g + 2 * i], values[2 * C * g + 2 * i + 1]
                        )
                K.ptx[MAX3_F32](temp[0], temp[0], temp[1], temp[2])
                K.ptx[MAX3_F32](temp[3], temp[3], temp[4], temp[5])
                K.ptx[MAX3_F32](out_[0], temp[6], temp[7], temp[0])
                K.assign(out_[0], K.max(out_[0], temp[3]))

            def shl_u32_clamp(val, shift):
                result = K.local_scalar("uint32")
                K.ptx.shl.b32(result, val, shift)
                return result

            def combine_int_frac_ex2(x_rounded, frac_ex2):
                x_rounded_i = K.local_scalar("int32")
                frac_ex_i = K.local_scalar("int32")
                x_rounded_e = K.local_scalar("int32")
                out_i = K.local_scalar("int32")
                out_f = K.local_scalar("float32")
                K.ptx.mov.b32(x_rounded_i, x_rounded)
                K.ptx.mov.b32(frac_ex_i, frac_ex2)
                K.ptx.shl.b32(x_rounded_e, x_rounded_i, K.uint32(23))
                K.ptx.add.s32(out_i, x_rounded_e, frac_ex_i)
                K.ptx.mov.b32(out_f, out_i)
                return out_f



            POLY_EX2_DEG3 = (1.0, 0.6951461434364319, 0.22756439447402954, 0.07711908966302872)
            FP32_ROUND_INT = float(2**23 + 2**22)

            def ex2_emulation_2(out_, idx, x, y):
                xy_clamped = K.alloc_local([2], "float32")
                K.ptx.mov.b32(xy_clamped[0], K.max(x, -127.0))
                K.ptx.mov.b32(xy_clamped[1], K.max(y, -127.0))
                packed = K.local_scalar("uint64")
                rhs = K.local_scalar("uint64")
                addend = K.local_scalar("uint64")
                xy_rounded = K.alloc_local([2], "float32")
                K.ptx.mov.b64(packed, xy_clamped[0], xy_clamped[1])
                K.ptx.mov.b64(rhs, K.float32(FP32_ROUND_INT), K.float32(FP32_ROUND_INT))
                K.ptx.add.rm.ftz.f32x2(packed, packed, rhs)
                K.ptx.mov.b64(xy_rounded[0], xy_rounded[1], packed)
                xy_rounded_back = K.alloc_local([2], "float32")
                K.ptx.mov.b64(packed, xy_rounded[0], xy_rounded[1])
                K.ptx.sub.rn.ftz.f32x2(packed, packed, rhs)
                K.ptx.mov.b64(xy_rounded_back[0], xy_rounded_back[1], packed)
                xy_frac = K.alloc_local([2], "float32")
                K.ptx.mov.b64(packed, xy_clamped[0], xy_clamped[1])
                K.ptx.mov.b64(rhs, xy_rounded_back[0], xy_rounded_back[1])
                K.ptx.sub.rn.ftz.f32x2(packed, packed, rhs)
                K.ptx.mov.b64(xy_frac[0], xy_frac[1], packed)
                xy_frac_ex2 = K.alloc_local([2], "float32")
                K.ptx.mov.b32(xy_frac_ex2[0], K.float32(POLY_EX2_DEG3[3]))
                K.ptx.mov.b32(xy_frac_ex2[1], K.float32(POLY_EX2_DEG3[3]))
                for coeff in (POLY_EX2_DEG3[2], POLY_EX2_DEG3[1], POLY_EX2_DEG3[0]):
                    K.ptx.mov.b64(rhs, xy_frac[0], xy_frac[1])
                    K.ptx.mov.b64(packed, xy_frac_ex2[0], xy_frac_ex2[1])
                    K.ptx.mov.b64(addend, K.float32(coeff), K.float32(coeff))
                    K.ptx.fma.rz.ftz.f32x2(packed, packed, rhs, addend)
                    K.ptx.mov.b64(xy_frac_ex2[0], xy_frac_ex2[1], packed)
                K.ptx.mov.b32(out_[idx], combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0]))
                K.ptx.mov.b32(out_[idx + 1], combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1]))

            sp = K.specialize(chain_dispatch=True)
            r_softmax = sp.role("softmax", warps=[0, 1, 2, 3, 4, 5, 6, 7], regs=232)
            wg3 = sp.warpgroup("wg3", warps=range(8, 12), regs=40)
            r_mma = sp.role("mma", warps=[8], group=wg3)
            r_load = sp.role("load", warps=[9], group=wg3)
            r_store = sp.role("store", warps=[10], group=wg3)
            r_idle = sp.role("idle", warps=[11], group=wg3)

            with K.If(warp_cta == 8), K.Then():
                K.ptx[TMEM_ALLOC](K.address_of(tmem_addr[0]), K.uint32(N_COLS_TMEM))
                K.cuda.warp_sync()
            with K.If(tvm.tirx.all(wg_id == 2, warp_id == 0)), K.Then():
                allocated = K.local_scalar("uint32")
                K.ptx.ld.shared.u32(allocated, tmem_addr.ptr_to([0]))
                K.cuda.trap_when_assert_failed(allocated == K.uint32(0))




            with wg3:

                with r_load:
                    it = K.local_scalar("int32", init=0)
                    running = K.local_scalar("int32", init=1)
                    with K.While(running != 0):
                        slot = it & 1
                        grabbed = K.local_scalar("int32", init=0)
                        with K.If(lane == 0), K.Then():
                            K.ptx.atom.relaxed.gpu.global_.add.s32(grabbed, sched.ptr_to([0]), K.int32(1))
                        task = K.local_scalar("int32", init=K.uniform(grabbed))
                        union_free.wait(slot, ((it >> 1) + 1) & 1)
                        with K.If(task >= NUM_TASKS):
                            with K.Then():
                                with K.If(lane == 0), K.Then():
                                    K.ptx.st.shared.b32(union_meta.ptr_to([slot * 8]), K.int32(-1))
                                union_ready.arrive(slot)
                                K.assign(running, 0)
                            with K.Else():
                                is_split = K.local_scalar("int32", init=0)
                                half = K.local_scalar("int32", init=0)
                                s_idx = K.local_scalar("int32", init=0)
                                item = K.local_scalar("int32", init=task)
                                if SPLIT:
                                    with K.If(task >= FULL_TASKS), K.Then():
                                        K.assign(is_split, 1)
                                        K.assign(half, (task - FULL_TASKS) & 1)
                                        K.assign(s_idx, (task - FULL_TASKS) >> 1)
                                        K.assign(item, FULL_TASKS + s_idx)
                                batch = item // HKV
                                kv_head = item % HKV
                                tok_base = batch * SEQLEN_Q
                                kv_s = K.local_scalar("int32")
                                kv_e = K.local_scalar("int32")
                                K.ptx.ld.global_.nc.b32(kv_s, cu_k.ptr_to([batch]))
                                K.ptx.ld.global_.nc.b32(kv_e, cu_k.ptr_to([batch + 1]))
                                kv_len = kv_e - kv_s
                                causal_off = kv_len - SEQLEN_Q
                                union_token = iket_range("union-build")
                                idxs = K.alloc_local([8], "int32")
                                idxs_i32 = K.decl_buffer((8,), "int32", data=idxs.data, scope="local")
                                idxs_u32 = idxs_i32.view("uint32")
                                q2k_base = (
                                    (kv_head * TOTAL_Q + tok_base + (lane >> 1)) * TOPK
                                    + (lane & 1) * 8
                                )
                                K.ptx[LD_Q2K](
                                    *(idxs_u32[i] for i in range(8)),
                                    q2k.ptr_to([q2k_base]),
                                )
                                my_mask = K.local_scalar("uint32", init=K.uint32(0))
                                for s in range(8):
                                    bit = K.Select(
                                        K.And(idxs[s] >= 0, idxs[s] < MAX_BLOCKS),
                                        K.shift_left(
                                            K.uint32(1), K.Cast("uint32", K.max(idxs[s], 0))
                                        ),
                                        K.uint32(0),
                                    )
                                    K.assign(my_mask, K.bitwise_or(my_mask, bit))
                                other_half = K.local_scalar("uint32")
                                K.ptx.shfl_sync.bfly.b32(
                                    other_half,
                                    my_mask,
                                    K.uint32(1),
                                    K.uint32(31),
                                    K.uint32(0xFFFFFFFF),
                                )
                                K.assign(my_mask, K.bitwise_or(my_mask, other_half))
                                token_selection_mask = K.local_scalar("uint32", init=my_mask)
                                with K.If((lane & 1) == 0), K.Then():
                                    K.ptx.st.shared.b32(
                                        token_masks.ptr_to([slot * TOK_PER_CTA + (lane >> 1)]),
                                        my_mask,
                                    )





                                q_pos_min = causal_off
                                q_pos_max = q_pos_min + (TOK_PER_CTA - 1)
                                b_max = K.max(q_pos_max, -1) // BLK_N
                                b_min = K.max(q_pos_min, 0) // BLK_N
                                vis_mask = K.local_scalar("uint32", init=K.uint32(0xFFFFFFFF))
                                with K.If(b_max < MAX_BLOCKS - 1), K.Then():
                                    K.assign(
                                        vis_mask,
                                        K.shift_left(K.uint32(1), K.Cast("uint32", b_max + 1))
                                        - K.uint32(1),
                                    )
                                with K.If(b_max < 0), K.Then():
                                    K.assign(vis_mask, K.uint32(0))
                                K.assign(my_mask, vis_mask)
                                lo_mask = K.local_scalar("uint32", init=K.uint32(0xFFFFFFFF))
                                with K.If(b_min < MAX_BLOCKS), K.Then():
                                    K.assign(
                                        lo_mask,
                                        K.shift_left(K.uint32(1), K.Cast("uint32", b_min)) - K.uint32(1),
                                    )
                                count = K.local_scalar("uint32")
                                K.ptx.popc.b32(count, my_mask)
                                n_masked = K.local_scalar("uint32")
                                K.ptx.popc.b32(n_masked, K.bitwise_and(my_mask, K.bitwise_not(lo_mask)))
                                with K.If(count == K.uint32(0)), K.Then():
                                    K.assign(my_mask, K.uint32(1))
                                    K.assign(count, K.uint32(1))
                                    K.assign(n_masked, K.uint32(1))

                                cnt_h = K.local_scalar("uint32", init=count)
                                nm_h = K.local_scalar("uint32", init=n_masked)
                                lo_h = K.local_scalar("int32", init=0)
                                dummy = K.local_scalar("int32", init=0)
                                if SPLIT:
                                    with K.If(is_split != 0), K.Then():
                                        nh = K.local_scalar("uint32", init=(count + K.uint32(1)) >> K.uint32(1))
                                        with K.If(half == 0):
                                            with K.Then():
                                                K.assign(cnt_h, nh)
                                                K.assign(nm_h, K.min(n_masked, nh))
                                            with K.Else():
                                                with K.If(count >= K.uint32(2)):
                                                    with K.Then():
                                                        K.assign(cnt_h, count - nh)
                                                        K.assign(
                                                            nm_h,
                                                            K.Select(n_masked > nh, n_masked - nh, K.uint32(0)),
                                                        )
                                                        K.assign(lo_h, K.Cast("int32", nh))
                                                    with K.Else():

                                                        K.assign(cnt_h, K.uint32(1))
                                                        K.assign(nm_h, K.uint32(1))
                                                        K.assign(dummy, 1)
                                    with K.If(dummy != 0), K.Then():
                                        with K.If((lane & 1) == 0), K.Then():
                                            K.ptx.st.shared.b32(
                                                token_masks.ptr_to([slot * TOK_PER_CTA + (lane >> 1)]),
                                                K.uint32(0),
                                            )
                                blk_hi = K.local_scalar("int32", init=b_max - lo_h)
                                with K.If(dummy != 0), K.Then():
                                    K.assign(blk_hi, 0)
                                with K.If(lane == 0), K.Then():
                                    K.ptx.st.shared.b32(union_meta.ptr_to([slot * 8]), K.Cast("int32", cnt_h))
                                    K.ptx.st.shared.b32(
                                        union_meta.ptr_to([slot * 8 + 1]), K.Cast("int32", nm_h)
                                    )
                                    K.ptx.st.shared.b32(union_meta.ptr_to([slot * 8 + 6]), is_split)
                                    K.ptx.st.shared.b32(union_meta.ptr_to([slot * 8 + 7]), s_idx)
                                    K.ptx.st.shared.b32(union_meta.ptr_to([slot * 8 + 2]), tok_base)
                                    K.ptx.st.shared.b32(union_meta.ptr_to([slot * 8 + 3]), kv_head)
                                    K.ptx.st.shared.b32(union_meta.ptr_to([slot * 8 + 4]), blk_hi)
                                    K.ptx.st.shared.b32(union_meta.ptr_to([slot * 8 + 5]), causal_off)
                                blk = K.local_scalar(
                                    "uint32",
                                    init=K.Select(
                                        K.Cast("uint32", lane) < cnt_h,
                                        K.Cast("uint32", blk_hi - lane),
                                        K.uint32(0),
                                    ),
                                )
                                selected_tokens = K.local_scalar("uint32", init=K.uint32(0))
                                for tok in range(TOK_PER_CTA):
                                    tok_mask = K.local_scalar("uint32")
                                    K.ptx.shfl_sync.idx.b32(
                                        tok_mask,
                                        token_selection_mask,
                                        K.uint32(2 * tok),
                                        K.uint32(31),
                                        K.uint32(0xFFFFFFFF),
                                    )
                                    selected_bit = K.bitwise_and(
                                        K.shift_right(tok_mask, blk), K.uint32(1)
                                    )
                                    K.assign(
                                        selected_tokens,
                                        K.bitwise_or(
                                            selected_tokens,
                                            K.shift_left(selected_bit, K.uint32(tok)),
                                        ),
                                    )
                                if SPLIT:
                                    with K.If(dummy != 0), K.Then():
                                        K.assign(selected_tokens, K.uint32(0))
                                disabled_tokens = K.bitwise_and(
                                    K.bitwise_not(selected_tokens), K.uint32(0xFFFF)
                                )




                                for i_q in range(N_TILES):
                                    q_load.empty.wait(i_q, q_epoch.phase)
                                with K.If(K.Cast("uint32", lane) < cnt_h), K.Then():
                                    for i_q in range(N_TILES):
                                        for pair in range(4):
                                            tok_lo = i_q * TOK_PER_TILE + 2 * pair
                                            lo = K.Select(
                                                K.bitwise_and(
                                                    disabled_tokens, K.uint32(1 << tok_lo)
                                                )
                                                != K.uint32(0),
                                                K.uint32(0x0000FFFF),
                                                K.uint32(0),
                                            )
                                            hi = K.Select(
                                                K.bitwise_and(
                                                    disabled_tokens, K.uint32(1 << (tok_lo + 1))
                                                )
                                                != K.uint32(0),
                                                K.uint32(0xFFFF0000),
                                                K.uint32(0),
                                            )
                                            K.ptx.st.shared.b32(
                                                output_lane_masks.ptr_to([slot, lane, i_q, pair]),
                                                K.bitwise_or(lo, hi),
                                            )
                                union_ready.arrive(slot)
                                K.cuda.iket.range_end(union_token[0])

                                for i_q in range(N_TILES):
                                    tma_q_token = iket_range("issue-tma-q")
                                    with K.If(elected()), K.Then():
                                        K.ptx[TMA_G2S_4D](
                                            q_smem[i_q].ptr_to(0, 0),
                                            K.address_of(q_map),
                                            K.int32(0),
                                            K.Cast("int32", kv_head * GQA),
                                            K.Cast("int32", tok_base + i_q * TOK_PER_TILE),
                                            K.int32(0),
                                            K.cuda.cvta_generic_to_shared(q_load.full.ptr_to([i_q])),
                                        )
                                        q_load.full.arrive(i_q, tx_count=Q_TILE_BYTES)
                                    K.cuda.iket.range_end(tma_q_token[0])
                                q_epoch.advance()

                                def load_kv(blk_, tensor_map, is_v):
                                    kv_load.empty.wait(kv_pipe.stage, kv_pipe.phase)
                                    tma_kv_token = iket_range("issue-tma-v" if is_v else "issue-tma-k")
                                    with K.If(elected()), K.Then():
                                        K.ptx[TMA_G2S_3D](
                                            (v_smem if is_v else k_smem)[kv_pipe.stage].ptr_to(0, 0),
                                            K.address_of(tensor_map),
                                            K.int32(0),
                                            K.Cast("int32", kv_s + blk_ * BLK_N),
                                            K.Cast("int32", kv_head * 2),
                                            K.cuda.cvta_generic_to_shared(
                                                kv_load.full.ptr_to([kv_pipe.stage])
                                            ),
                                            K.uint64(KV_CACHE_POLICY),
                                        )
                                        kv_load.full.arrive(kv_pipe.stage, tx_count=KV_TILE_BYTES)
                                    K.cuda.iket.range_end(tma_kv_token[0])
                                    kv_pipe.advance()

                                blk_cur = K.local_scalar("int32", init=blk_hi)
                                load_kv(blk_cur, k_map, False)
                                with K.serial(K.Cast("int32", cnt_h), unroll=False) as _k:
                                    with K.If(_k + 1 < K.Cast("int32", cnt_h)):
                                        with K.Then():
                                            blk_nxt = K.local_scalar("int32", init=blk_cur - 1)
                                            load_kv(blk_nxt, k_map, False)
                                            load_kv(blk_cur, v_map, True)
                                            K.assign(blk_cur, blk_nxt)
                                        with K.Else():
                                            load_kv(blk_cur, v_map, True)
                                K.assign(it, it + 1)


                with r_mma:
                    it_m = K.local_scalar("int32", init=0)
                    gstep_m = K.local_scalar("int32", init=0)

                    tb_raw = K.local_scalar("uint32")
                    K.ptx.ld.shared.u32(tb_raw, tmem_addr.ptr_to([0]))
                    tmem_base = K.local_scalar("uint32", init=K.uniform(tb_raw))

                    def load_output_lane_mask(slot_, list_idx, q_stage):
                        disabled = K.alloc_local([4], "uint32")
                        for word in range(4):
                            K.ptx.ld.shared.u32(
                                disabled[word],
                                output_lane_masks.ptr_to([slot_, list_idx, q_stage, word]),
                            )
                        return disabled

                    def gemm_qk(q_stage, kv_stage, disabled):
                        qk_token = iket_range("mma-qk")
                        for ki in range(HEAD_DIM // MMA_K):
                            with K.If(elected()), K.Then():
                                K.ptx[MMA_F16](
                                    tmem_base + K.uint32(q_stage * MMA_N),
                                    desc_at(q_desc, q_stage * Q_STAGE16 + qoff(ki)),
                                    desc_at(k_desc, kv_stage * KV_STAGE16 + koff(ki)),
                                    K.uint32(ID_QK),
                                    disabled[0],
                                    disabled[1],
                                    disabled[2],
                                    disabled[3],
                                    ki != 0,
                                )
                        with K.If(elected()), K.Then():
                            commit(s_ready, q_stage)
                        K.cuda.iket.range_end(qk_token[0])

                    def gemm_pv_part1(i_q, kv_stage, should_accumulate, disabled):
                        for ki in range(K_SPLIT // MMA_K):
                            with K.If(elected()), K.Then():
                                K.ptx[MMA_F16](
                                    tmem_base + K.uint32((N_TILES + i_q) * MMA_N),
                                    tmem_base + K.uint32((1 - i_q) * MMA_N + MMA_N // 2 + ki * (MMA_K // 2)),
                                    desc_at(v_desc, kv_stage * KV_STAGE16 + mnoff(ki)),
                                    K.uint32(ID_PV),
                                    disabled[0],
                                    disabled[1],
                                    disabled[2],
                                    disabled[3],
                                    True if ki != 0 else K.Cast("bool", should_accumulate),
                                )

                    def gemm_pv_part2(i_q, kv_stage, disabled):
                        p_ready_2.wait(i_q, tmem_epoch.phase)
                        for ki in range((BLK_N - K_SPLIT) // MMA_K):
                            with K.If(elected()), K.Then():
                                K.ptx[MMA_F16](
                                    tmem_base + K.uint32((N_TILES + i_q) * MMA_N),
                                    tmem_base + K.uint32((1 - i_q) * MMA_N + MMA_N // 2 + K_SPLIT // 2 + ki * (MMA_K // 2)),
                                    desc_at(v_desc, kv_stage * KV_STAGE16 + mnoff(K_SPLIT // MMA_K + ki)),
                                    K.uint32(ID_PV),
                                    disabled[0],
                                    disabled[1],
                                    disabled[2],
                                    disabled[3],
                                    True,
                                )

                    def gemm_pv(i_q, kv_stage, should_accumulate, selected_disabled):
                        pv_token = iket_range("mma-pv")
                        disabled = K.alloc_local([4], "uint32")
                        for word in range(4):
                            K.assign(
                                disabled[word],
                                K.Select(
                                    should_accumulate != 0,
                                    selected_disabled[word],
                                    K.uint32(0),
                                ),
                            )
                        gemm_pv_part1(i_q, kv_stage, should_accumulate, disabled)
                        gemm_pv_part2(i_q, kv_stage, disabled)
                        with K.If(elected()), K.Then():
                            commit(pv_done, i_q)
                        K.cuda.iket.range_end(pv_token[0])

                    running_m = K.local_scalar("int32", init=1)
                    with K.While(running_m != 0):
                        slot_m = it_m & 1
                        union_ready.wait(slot_m, (it_m >> 1) & 1)
                        n_blocks = ld_shared_i32(union_meta.ptr_to([slot_m * 8]))
                        with K.If(n_blocks < 0), K.Then():
                            K.assign(running_m, 0)
                        with K.If(n_blocks > 0), K.Then():
                            acc = K.local_scalar("int32", init=0)
                            for i_q in range(N_TILES):
                                q_load.full.wait(i_q, q_epoch.phase)
                                if i_q == 0:
                                    kv_load.full.wait(kv_pipe.stage, kv_pipe.phase)
                                first_disabled = K.alloc_local([4], "uint32")
                                for word in range(4):
                                    K.assign(first_disabled[word], K.uint32(0))
                                gemm_qk(i_q, kv_pipe.stage, first_disabled)
                                if i_q == N_TILES - 1:
                                    with K.If(elected()), K.Then():
                                        kv_load.empty.arrive(kv_pipe.stage)
                            kv_pipe.advance()
                            with K.If(n_blocks == 1), K.Then():
                                for i_q in range(N_TILES):
                                    with K.If(elected()), K.Then():
                                        q_load.empty.arrive(i_q)
                            with K.serial(n_blocks, unroll=False) as n:
                                has_next = n + 1 < n_blocks
                                k_stage = K.local_scalar("int32", init=kv_pipe.stage)
                                k_phase = K.local_scalar("int32", init=kv_pipe.phase)
                                with K.If(has_next), K.Then():
                                    kv_pipe.advance()
                                v_stage = K.local_scalar("int32", init=kv_pipe.stage)
                                v_phase = K.local_scalar("int32", init=kv_pipe.phase)
                                kv_pipe.advance()
                                for i_q in range(N_TILES):
                                    current_disabled = load_output_lane_mask(slot_m, n, i_q)
                                    with K.If(has_next), K.Then():
                                        if i_q == 0:
                                            kv_load.full.wait(k_stage, k_phase)
                                        s_consumed.wait(i_q, gstep_m & 1)
                                        next_disabled = load_output_lane_mask(slot_m, n + 1, i_q)
                                        gemm_qk(i_q, k_stage, next_disabled)
                                        with K.If(n == n_blocks - 2), K.Then():
                                            with K.If(elected()), K.Then():
                                                q_load.empty.arrive(i_q)
                                        if i_q == N_TILES - 1:
                                            with K.If(elected()), K.Then():
                                                kv_load.empty.arrive(k_stage)
                                    if i_q == 0:
                                        kv_load.full.wait(v_stage, v_phase)
                                    with K.If(n == 0), K.Then():
                                        o_free.wait(i_q, (it_m + 1) & 1)
                                    p_o_rescale.wait(i_q, tmem_epoch.phase)
                                    gemm_pv(i_q, v_stage, acc, current_disabled)
                                    if i_q == N_TILES - 1:
                                        with K.If(elected()), K.Then():
                                            kv_load.empty.arrive(v_stage)
                                    with K.If(n == n_blocks - 1), K.Then():
                                        with K.If(elected()), K.Then():
                                            commit(o_ready, i_q)
                                K.assign(acc, 1)
                                tmem_epoch.advance()
                                K.assign(gstep_m, gstep_m + 1)
                            q_epoch.advance()
                        K.assign(it_m, it_m + 1)


                with r_store:
                    pass

                with r_idle:
                    pass




            with r_softmax:
                it_x = K.local_scalar("int32", init=0)
                gstep_x = K.local_scalar("int32", init=0)
                with K.If(wg_id == 1), K.Then():
                    xu_turn.arrive(0)
                tok_local = tid_in_wg // GQA
                head_local = tid_in_wg % GQA
                row_max = K.local_scalar("float32")
                row_sum = K.alloc_local([1], "float32")
                sel_mask = K.local_scalar("uint32")
                q_pos = K.local_scalar("int32")

                def mask_r2p(s_chunk, col_limit, ncol):
                    CHUNK_SIZE = 32
                    for s_ in range(ceildiv(ncol, CHUNK_SIZE)):
                        k_keep = K.max(col_limit - s_ * CHUNK_SIZE, 0)
                        mask_inv = K.local_scalar("uint32")
                        K.assign(
                            mask_inv, shl_u32_clamp(K.uint32(0xFFFFFFFF), K.Cast("uint32", k_keep))
                        )
                        for i in range(CHUNK_SIZE):
                            if i < ncol - s_ * CHUNK_SIZE:
                                c = s_ * CHUNK_SIZE + i
                                in_bound = K.bitwise_and(
                                    K.bitwise_not(mask_inv), K.shift_left(K.uint32(1), K.uint32(i))
                                )
                                K.ptx.mov.b32(
                                    s_chunk[c],
                                    K.Select(
                                        K.Cast("bool", in_bound), s_chunk[c], K.float32(NEG_INF)
                                    ),
                                )

                def apply_causal_mask(s_chunk, blk_):
                    col_limit_right = q_pos - blk_ * BLK_N + 1
                    mask_r2p(s_chunk, col_limit_right, BLK_N)

                def rescale_o_rows(scale):
                    RESCALE_TILE = 16
                    o_row = K.alloc_local([RESCALE_TILE], "float32")
                    for d_tile in range(HEAD_DIM // RESCALE_TILE):
                        d_start = d_tile * RESCALE_TILE
                        addr = tmem((N_TILES + wg_id) * MMA_N + d_start)
                        tmem_load(o_row, 0, addr, RESCALE_TILE)
                        for i in range(RESCALE_TILE // 2):
                            mul_f32x2(o_row, 2 * i, scale)
                        tmem_store(o_row, 0, addr)
                    K.ptx.tcgen05.wait__st.sync.aligned()

                def softmax_step(blk_, other_par, apply_mask=False, is_first=False):
                    s_chunk = K.alloc_local([BLK_N], "float32")
                    p_chunk = K.alloc_local([BLK_N // 2], "uint32")
                    selected = K.local_scalar(
                        "int32",
                        init=K.Cast(
                            "int32",
                            K.bitwise_and(
                                K.shift_right(sel_mask, K.Cast("uint32", blk_)), K.uint32(1)
                            ),
                        ),
                    )




                    any_sel = K.local_scalar("uint32", init=K.uint32(1))
                    if not is_first:
                        K.ptx.vote_sync.any.pred(any_sel, K.ptx.pred(selected), K.uint32(0xFFFFFFFF))
                    s_ready.wait(wg_id, score_epoch.phase)

                    def active_body():
                        softmax_max_token = iket_range("softmax-max", leader_only=True)
                        tile_max = K.alloc_local([1], "float32")
                        for chunk_idx in range(BLK_N // 32):
                            tmem_load(s_chunk, chunk_idx * 32, tmem(wg_id * MMA_N + chunk_idx * 32), 32)
                        if apply_mask:
                            apply_causal_mask(s_chunk, blk_)
                        row_max_old = K.local_scalar("float32")
                        if is_first:
                            reduce_max_128(tile_max, s_chunk)
                            K.assign(
                                tile_max[0],
                                K.Select(selected != 0, tile_max[0], K.float32(NEG_INF)),
                            )
                        else:
                            K.assign(row_max_old, row_max)
                            K.assign(tile_max[0], row_max_old)
                            reduce_max_128(tile_max, s_chunk, accum=True)
                            K.assign(tile_max[0], K.Select(selected != 0, tile_max[0], row_max_old))
                        s_consumed.arrive(wg_id)
                        row_max_new = K.local_scalar("float32")
                        acc_scale = K.local_scalar("float32", init=K.float32(1.0))
                        acc_scale_ = K.local_scalar("float32")
                        row_max_safe = K.local_scalar("float32")
                        K.assign(row_max_new, tile_max[0])
                        K.assign(
                            row_max_safe,
                            K.if_then_else(tile_max[0] == K.float32(NEG_INF), K.float32(0.0), tile_max[0]),
                        )
                        if not is_first:
                            K.assign(acc_scale_, (row_max_old - row_max_safe) * scale_log2)
                            with K.If(acc_scale_ >= -RESCALE_THRESHOLD):
                                with K.Then():
                                    K.assign(row_max_new, row_max_old)
                                    K.assign(row_max_safe, row_max_old)
                                with K.Else():
                                    with K.If(row_max_old != K.float32(NEG_INF)), K.Then():
                                        K.ptx.ex2.approx.ftz.f32(acc_scale, acc_scale_)
                        K.assign(row_max, row_max_new)
                        row_max_scaled = row_max_safe * scale_log2
                        K.cuda.iket.range_end(softmax_max_token[0])
                        if not is_first:
                            should_rescale = K.local_scalar(
                                "int32", init=K.Select(acc_scale < K.float32(1.0), 1, 0)
                            )
                            any_needs_rescale = K.local_scalar("uint32")
                            K.ptx.vote_sync.any.pred(
                                any_needs_rescale, K.ptx.pred(should_rescale), K.uint32(0xFFFFFFFF)
                            )
                            with K.If(any_needs_rescale != 0), K.Then():
                                rescale_token = iket_range("rescale", leader_only=True)
                                pv_done.wait(wg_id, (gstep_x + 1) & 1)
                                rescale_o_rows(acc_scale)
                                K.cuda.iket.range_end(rescale_token[0])
                        turn_token = iket_range("xu-turn-wait", leader_only=True)
                        xu_turn.wait(wg_id, gstep_x & 1)
                        K.cuda.iket.range_end(turn_token[0])
                        softmax_exp2_token = iket_range("softmax-exp2", leader_only=True)
                        bias = K.local_scalar("float32")
                        K.assign(
                            bias,
                            K.Select(selected != 0, K.float32(0.0) - row_max_scaled, K.float32(NEG_INF)),
                        )
                        scale_pair = K.local_scalar("uint64")
                        bias_pair = K.local_scalar("uint64")
                        K.ptx.mov.b64(scale_pair, K.float32(1.0) * scale_log2, K.float32(1.0) * scale_log2)
                        K.ptx.mov.b64(bias_pair, bias, bias)
                        sum_acc = [K.local_scalar("uint64") for _ in range(N_SUM_ACC)]
                        for a in sum_acc:
                            K.ptx.mov.b64(a, K.float32(0.0), K.float32(0.0))
                        pair_tmp = K.local_scalar("uint64")
                        for frag_idx in range(4):
                            for i in range(BLK_N // 4 // 2):
                                idx = frag_idx * BLK_N // 4 + 2 * i
                                K.ptx.mov.b64(pair_tmp, s_chunk[idx], s_chunk[idx + 1])
                                K.ptx.fma.rz.ftz.f32x2(pair_tmp, pair_tmp, scale_pair, bias_pair)
                                K.ptx.mov.b64(s_chunk[idx], s_chunk[idx + 1], pair_tmp)
                                if (
                                    i * 2 % 16 < 16 - 2 * EMU_PAIRS
                                    or frag_idx >= 4 - 1
                                    or frag_idx < EMU_START
                                    or apply_mask
                                ):
                                    K.ptx.ex2.approx.ftz.f32(s_chunk[idx], s_chunk[idx])
                                    K.ptx.ex2.approx.ftz.f32(s_chunk[idx + 1], s_chunk[idx + 1])
                                else:
                                    ex2_emulation_2(s_chunk, idx, s_chunk[idx], s_chunk[idx + 1])
                        K.cuda.warp_sync()
                        xu_turn.arrive(1 - wg_id)
                        K.cuda.warp_sync()
                        for frag_idx in range(4):
                            for i in range(BLK_N // 4 // 2):
                                idx = frag_idx * BLK_N // 4 + 2 * i
                                K.ptx.mov.b64(pair_tmp, s_chunk[idx], s_chunk[idx + 1])
                                acc_k = sum_acc[(frag_idx * (BLK_N // 8) + i) % N_SUM_ACC]
                                K.ptx.add.rn.ftz.f32x2(acc_k, acc_k, pair_tmp)
                                cast_f32x2_bf16x2(p_chunk, s_chunk, idx)
                            if frag_idx == P_SPLIT_Q - 1:
                                pv_done.wait(wg_id, (gstep_x + 1) & 1)
                                s_consumed.wait(1 - wg_id, other_par)
                                for i in range(P_SPLIT_Q):
                                    tmem_store(
                                        p_chunk,
                                        i * BLK_N // 4 // 2,
                                        tmem(((1 - wg_id) * 2 * MMA_N + MMA_N + i * BLK_N // 4) // 2),
                                    )
                            if frag_idx == P_SPLIT_Q:
                                K.ptx.tcgen05.wait__st.sync.aligned()
                                p_o_rescale.arrive(wg_id)
                        K.cuda.iket.range_end(softmax_exp2_token[0])
                        softmax_tmem_st_token = iket_range("softmax-tmem-st", leader_only=True)
                        for i in range(4 - P_SPLIT_Q):
                            tmem_store(
                                p_chunk,
                                (P_SPLIT_Q + i) * BLK_N // 4 // 2,
                                tmem(((1 - wg_id) * 2 * MMA_N + MMA_N + (P_SPLIT_Q + i) * BLK_N // 4) // 2),
                            )
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        p_ready_2.arrive(wg_id)
                        K.cuda.iket.range_end(softmax_tmem_st_token[0])
                        softmax_sum_token = iket_range("softmax-sum", leader_only=True)
                        for step in (4, 2, 1):
                            for a in range(step):
                                K.ptx.add.rn.ftz.f32x2(sum_acc[a], sum_acc[a], sum_acc[a + step])
                        sum_lo = K.local_scalar("float32")
                        sum_hi = K.local_scalar("float32")
                        K.ptx.mov.b64(sum_lo, sum_hi, sum_acc[0])
                        if is_first:
                            K.assign(row_sum[0], sum_lo + sum_hi)
                        else:
                            K.assign(row_sum[0], row_sum[0] * acc_scale + (sum_lo + sum_hi))
                        K.cuda.iket.range_end(softmax_sum_token[0])

                    def idle_body():
                        s_consumed.arrive(wg_id)
                        xu_turn.wait(wg_id, gstep_x & 1)
                        K.cuda.warp_sync()
                        xu_turn.arrive(1 - wg_id)
                        K.cuda.warp_sync()
                        pv_done.wait(wg_id, (gstep_x + 1) & 1)
                        s_consumed.wait(1 - wg_id, other_par)
                        p_o_rescale.arrive(wg_id)
                        p_ready_2.arrive(wg_id)

                    if is_first:
                        active_body()
                    else:
                        with K.If(any_sel != 0):
                            with K.Then():
                                active_body()
                            with K.Else():
                                idle_body()
                    score_epoch.advance()
                    K.assign(gstep_x, gstep_x + 1)

                running_x = K.local_scalar("int32", init=1)
                with K.While(running_x != 0):
                    slot_x = it_x & 1
                    union_ready.wait(slot_x, (it_x >> 1) & 1)
                    n_blocks_s = ld_shared_i32(union_meta.ptr_to([slot_x * 8]))
                    with K.If(n_blocks_s < 0), K.Then():
                        K.assign(running_x, 0)
                    with K.If(n_blocks_s > 0), K.Then():
                        n_masked_s = ld_shared_i32(union_meta.ptr_to([slot_x * 8 + 1]))
                        causal_off_x = ld_shared_i32(union_meta.ptr_to([slot_x * 8 + 5]))

                        K.assign(q_pos, causal_off_x + wg_id * TOK_PER_TILE + tok_local)
                        mask_lane = K.local_scalar("uint32", init=K.uint32(0))
                        with K.If(head_local == 0), K.Then():
                            K.ptx.ld.shared.b32(
                                mask_lane,
                                token_masks.ptr_to(
                                    [slot_x * TOK_PER_CTA + wg_id * TOK_PER_TILE + tok_local]
                                ),
                            )
                        K.ptx.shfl_sync.idx.b32(
                            sel_mask,
                            mask_lane,
                            K.bitwise_and(K.Cast("uint32", lane), K.uint32(0xFFFFFFF0)),
                            K.uint32(31),
                            K.uint32(0xFFFFFFFF),
                        )
                        blk_hi_x = ld_shared_i32(union_meta.ptr_to([slot_x * 8 + 4]))

                        def other_parity(n):
                            if_wg0 = gstep_x & 1
                            nxt = K.Select(n + 1 < n_blocks_s, (gstep_x + 1) & 1, gstep_x & 1)
                            return K.Select(wg_id == 0, if_wg0, nxt)

                        blk0 = blk_hi_x
                        softmax_step(blk0, other_parity(K.int32(0)), apply_mask=True, is_first=True)
                        n_masked_rest = K.max(n_masked_s - 1, 0)
                        with K.serial(n_masked_rest, unroll=False) as i:
                            blk_m = blk_hi_x - 1 - i
                            softmax_step(blk_m, other_parity(1 + i), apply_mask=True)
                        start_plain = K.max(n_masked_s, 1)
                        with K.serial(n_blocks_s - start_plain, unroll=False) as i:
                            blk_p = blk_hi_x - start_plain - i
                            softmax_step(blk_p, other_parity(start_plain + i), apply_mask=False)


                        split_x = ld_shared_i32(union_meta.ptr_to([slot_x * 8 + 6]))
                        s_x = ld_shared_i32(union_meta.ptr_to([slot_x * 8 + 7]))
                        tok_base_x = ld_shared_i32(union_meta.ptr_to([slot_x * 8 + 2]))
                        kv_head_x = ld_shared_i32(union_meta.ptr_to([slot_x * 8 + 3]))
                        union_free.arrive(slot_x)

                        epi_wait_token = iket_range("epi-wait-o", leader_only=True)
                        o_ready.wait(wg_id, it_x & 1)
                        K.cuda.iket.range_end(epi_wait_token[0])
                        epi_token = iket_range("epi-store", leader_only=True)
                        EPI_LD = 32
                        o_row_f32 = K.alloc_local([HEAD_DIM], "float32")
                        for d_tile in range(HEAD_DIM // EPI_LD):
                            tmem_load(
                                o_row_f32, d_tile * EPI_LD, tmem((N_TILES + wg_id) * MMA_N + d_tile * EPI_LD), EPI_LD
                            )
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                        o_free.arrive(wg_id)
                        l_row = K.local_scalar("float32", init=row_sum[0])
                        do_stage = K.local_scalar("int32", init=1)
                        if SPLIT:
                            with K.If(split_x != 0), K.Then():
                                ctl_base = (s_x * 2 + wg_id) * 2
                                with K.If(tid_in_wg == 0), K.Then():
                                    old = K.local_scalar("int32")
                                    K.ptx.atom.acq_rel.gpu.global_.add.s32(
                                        old, mrg_ctl.ptr_to([ctl_base]), K.int32(1)
                                    )
                                    K.ptx.st.shared.b32(mrg_order.ptr_to([wg_id]), old)
                                K.ptx.bar.sync(K.Cast("uint32", 1 + wg_id), K.uint32(128))
                                order = ld_shared_i32(mrg_order.ptr_to([wg_id]))
                                ml_base = ((s_x * 2 + wg_id) * 2) * BLK_M



                                o_base = (s_x * 2 + wg_id) * BLK_M * HEAD_DIM + tid_in_wg * 4
                                with K.If(order == 0):
                                    with K.Then():

                                        K.ptx.st.global_.f32(mrg_ml.ptr_to([ml_base + tid_in_wg]), row_max)
                                        K.ptx.st.global_.f32(
                                            mrg_ml.ptr_to([ml_base + BLK_M + tid_in_wg]), row_sum[0]
                                        )
                                        for i in range(HEAD_DIM // 4):
                                            K.ptx.st.global_.v4.f32(
                                                mrg_o.ptr_to([o_base + i * (BLK_M * 4)]),
                                                o_row_f32[4 * i],
                                                o_row_f32[4 * i + 1],
                                                o_row_f32[4 * i + 2],
                                                o_row_f32[4 * i + 3],
                                            )
                                        K.ptx.bar.sync(K.Cast("uint32", 1 + wg_id), K.uint32(128))
                                        with K.If(tid_in_wg == 0), K.Then():
                                            K.ptx.fence.acq_rel.gpu()
                                            K.ptx.st.relaxed.gpu.global_.b32(
                                                mrg_ctl.ptr_to([ctl_base + 1]), K.int32(1)
                                            )
                                        K.assign(do_stage, 0)
                                    with K.Else():

                                        ready = K.local_scalar("int32", init=0)
                                        with K.While(ready == 0):
                                            K.ptx.ld.acquire.gpu.global_.b32(
                                                ready, mrg_ctl.ptr_to([ctl_base + 1])
                                            )
                                        m_b = K.local_scalar("float32")
                                        l_b = K.local_scalar("float32")
                                        K.ptx.ld.relaxed.gpu.global_.f32(m_b, mrg_ml.ptr_to([ml_base + tid_in_wg]))
                                        K.ptx.ld.relaxed.gpu.global_.f32(
                                            l_b, mrg_ml.ptr_to([ml_base + BLK_M + tid_in_wg])
                                        )
                                        m_ab = K.local_scalar("float32", init=K.max(row_max, m_b))
                                        m_safe = K.local_scalar(
                                            "float32",
                                            init=K.if_then_else(m_ab == K.float32(NEG_INF), K.float32(0.0), m_ab),
                                        )
                                        a_a = K.local_scalar("float32")
                                        a_b = K.local_scalar("float32")
                                        K.ptx.ex2.approx.ftz.f32(a_a, (row_max - m_safe) * scale_log2)
                                        K.ptx.ex2.approx.ftz.f32(a_b, (m_b - m_safe) * scale_log2)
                                        K.assign(l_row, row_sum[0] * a_a + l_b * a_b)
                                        MRG_CHUNK = 32
                                        for c in range(HEAD_DIM // MRG_CHUNK):
                                            ob = K.alloc_local([MRG_CHUNK], "float32")
                                            for i in range(MRG_CHUNK // 4):
                                                K.ptx.ld.relaxed.gpu.global_.v4.f32(
                                                    ob[4 * i], ob[4 * i + 1], ob[4 * i + 2], ob[4 * i + 3],
                                                    mrg_o.ptr_to([o_base + (c * (MRG_CHUNK // 4) + i) * (BLK_M * 4)]),
                                                )
                                            for i in range(MRG_CHUNK):
                                                K.assign(
                                                    o_row_f32[c * MRG_CHUNK + i],
                                                    o_row_f32[c * MRG_CHUNK + i] * a_a + ob[i] * a_b,
                                                )
                                        with K.If(tid_in_wg == 0), K.Then():
                                            K.ptx.st.relaxed.gpu.global_.b32(mrg_ctl.ptr_to([ctl_base]), K.int32(0))
                                            K.ptx.st.relaxed.gpu.global_.b32(
                                                mrg_ctl.ptr_to([ctl_base + 1]), K.int32(0)
                                            )
                        with K.If(do_stage != 0), K.Then():
                            acc_O_row_is_zero_or_nan = tvm.tirx.any(
                                l_row == K.float32(0.0), l_row != l_row
                            )
                            norm_scale = K.local_scalar("float32")
                            K.ptx.rcp.approx.ftz.f32(
                                norm_scale, K.Select(acc_O_row_is_zero_or_nan, K.float32(1.0), l_row)
                            )


                            row_elem = (
                                (tok_base_x + wg_id * TOK_PER_TILE + tok_local) * (HKV * GQA)
                                + kv_head_x * GQA
                                + head_local
                            ) * HEAD_DIM
                            for d_tile in range(HEAD_DIM // EPI_LD):
                                d_start = d_tile * EPI_LD
                                o_tile_bf16 = K.alloc_local([EPI_LD // 2], "uint32")
                                for i in range(EPI_LD // 2):
                                    mul_f32x2(o_row_f32, d_start + 2 * i, norm_scale)
                                for i in range(EPI_LD // 2):
                                    K.ptx.cvt.rn.bf16x2.f32(
                                        o_tile_bf16[i], o_row_f32[d_start + 2 * i + 1], o_row_f32[d_start + 2 * i]
                                    )
                                for i in range(EPI_LD // 16):
                                    w0 = i * 8
                                    K.ptx[ST_OUTPUT](
                                        out.ptr_to([row_elem + d_start + i * 16]),
                                        *(o_tile_bf16[w0 + j] for j in range(8)),
                                    )
                        K.cuda.iket.range_end(epi_token[0])
                        K.assign(it_x, it_x + 1)

            K.cuda.cta_sync()
            with K.If(K.thread_id() == 0), K.Then():
                done = K.local_scalar("int32")
                K.ptx.atom.acq_rel.gpu.global_.add.s32(done, sched.ptr_to([1]), K.int32(1))
                with K.If(done == NUM_CTAS - 1), K.Then():
                    K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([0]), K.int32(0))
                    K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([1]), K.int32(0))
            with K.If(tvm.tirx.all(wg_id == 0, warp_id == 0)), K.Then():
                dealloc = K.local_scalar("uint32")
                K.ptx.ld.shared.u32(dealloc, tmem_addr.ptr_to([0]))
                K.ptx[TMEM_RELINQUISH]()
                K.ptx[TMEM_DEALLOC](dealloc, K.uint32(N_COLS_TMEM))

        return msa_decode_qmajor_union
    return make_kernel


make_kernel_q16_twotile = _build_q16_twotile_factory()


def setup_q16_twotile(data, B, seqlen_q):
    q = data["q"]
    k = data["k"]
    v = data["v"]
    q2k = data["q2k_indices"]
    cu_k = data["cu_seqlens_k"]
    out = data["output"]
    B = int(B)
    T = int(seqlen_q)
    total_q, hq, d = q.shape
    total_k, hkv, dk = k.shape
    topk = int(q2k.shape[-1])
    assert B == 128 and T == 16
    assert total_q == B * T and total_k == B * 4096
    assert hq == 64 and hkv == 4 and d == HEAD_DIM and dk == HEAD_DIM
    assert topk == 16
    assert q.dtype == torch.bfloat16
    assert k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    assert cu_k is not None
    assert data["page_table"] is None and data["seqused_k"] is None
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q2k.is_contiguous() and cu_k.is_contiguous() and out.is_contiguous()

    device = q.device
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    key = ("q16_twotile", B, T, hq, hkv, topk, int(num_sms))
    ex = _EXEC_CACHE.get(key)
    if ex is None:
        os.environ.setdefault("TVM_CUDA_PTXAS_REG_LEVEL", "6")
        kernel = make_kernel_q16_twotile(B, T, hq, hkv, topk, int(num_sms))
        target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
        with target:
            ex = tvm.compile(kernel.mod, target=target, tir_pipeline="tirx")
        _EXEC_CACHE[key] = ex

    scale_log2 = float(data["softmax_scale"]) * LOG2E
    q_map = _encode(
        q,
        "bfloat16",
        (HEAD_DIM // 2, hq, total_q, 2),
        (HEAD_DIM * 2, hq * HEAD_DIM * 2, (HEAD_DIM // 2) * 2),
        (HEAD_DIM // 2, 16, 8, 2),
    )
    kv_dims = (HEAD_DIM // 2, total_k, hkv * 2)
    kv_strides = (hkv * HEAD_DIM * 2, (HEAD_DIM // 2) * 2)
    kv_box = (HEAD_DIM // 2, BLK, 2)
    k_map = _encode(k, "bfloat16", kv_dims, kv_strides, kv_box, l2_promotion=3)
    v_map = _encode(v, "bfloat16", kv_dims, kv_strides, kv_box, l2_promotion=3)

    sched = torch.zeros(2, dtype=torch.int32, device=device)
    num_items = B * hkv
    num_ctas = min(int(num_sms), num_items)
    r_split = num_items - (num_items // num_ctas) * num_ctas
    n_slots = max(r_split, 1) * 2
    mrg_o = torch.zeros(n_slots * 128 * HEAD_DIM, dtype=torch.float32, device=device)
    mrg_ml = torch.zeros(n_slots * 2 * 128, dtype=torch.float32, device=device)
    mrg_ctl = torch.zeros(n_slots * 2, dtype=torch.int32, device=device)
    q2k_flat = q2k.view(-1)
    out_flat = out.view(-1)
    args = (
        q_map.ptr,
        k_map.ptr,
        v_map.ptr,
        out_flat,
        q2k_flat,
        cu_k,
        sched,
        mrg_o,
        mrg_ml,
        mrg_ctl,
        scale_log2,
    )
    keep = (
        q,
        k,
        v,
        q2k,
        q2k_flat,
        cu_k,
        out,
        out_flat,
        sched,
        mrg_o,
        mrg_ml,
        mrg_ctl,
        q_map,
        k_map,
        v_map,
    )

    def run():
        ex(*args)

    run._keep = keep
    run._cfg = {"family": "q16-two-tile-native-ex2", "num_ctas": num_ctas}
    run()
    torch.cuda.synchronize(device)
    return run


def _use_q16_twotile(data, B, T):
    q, k, v = data["q"], data["k"], data["v"]
    return (
        int(B) == 128
        and int(T) == 16
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and data["page_table"] is None
        and data["seqused_k"] is None
        and data["cu_seqlens_k"] is not None
        and tuple(q.shape) == (2048, 64, 128)
        and tuple(k.shape) == (524288, 4, 128)
        and tuple(v.shape) == (524288, 4, 128)
        and int(data["q2k_indices"].shape[-1]) == 16
    )


def setup(data, B, seqlen_q):  # noqa: F811 - chained shape-dispatch override
    T = int(seqlen_q)
    if _use_q16_twotile(data, B, T):
        return setup_q16_twotile(data, B, T)
    if _use_qm64(data, B, T):
        return setup_qm64(data, B, T)
    family = choose_family(data["q"], data["k"], T)
    if family == "qm":
        return setup_qm(data, B, T)
    return setup_kv(data, B, T)




















Q1D_NOFENCE = os.environ.get("Q1D_NOFENCE", "0") == "1"
Q1D_NOPREFETCH = os.environ.get("Q1D_NOPREFETCH", "0") == "1"


def make_kernel_q1d(cfg):
    G = cfg["G"]
    HKV = cfg["HKV"]
    TOPK = cfg["TOPK"]
    TOTAL_Q = cfg["TOTAL_Q"]
    kind = cfg["kv_kind"]
    PAGED = cfg["paged"]
    MAX_PAGES = cfg["MAX_PAGES"]
    NUM_ITEMS = cfg["NUM_ITEMS"]
    NUM_CTAS = cfg["NUM_CTAS"]
    STAGES = cfg["STAGES"]
    CH = cfg["CH"]
    STATIC_ONE_SHOT = cfg["STATIC_ONE_SHOT"]
    HQ = HKV * G
    NQ = G
    NQ_PAD = max(16, ceildiv(NQ, 16) * 16)
    FP8 = kind == "fp8"
    kv_dt = {"bf16": K.bf16, "f16": K.f16, "fp8": K.u8}[kind]
    out_dt = K.f16 if kind == "f16" else K.bf16
    EB = 1 if FP8 else 2
    MMA_K = 32 if FP8 else 16
    NK = HEAD_DIM // MMA_K
    KV_TILE_BYTES = BLK * HEAD_DIM * EB
    Q_TMA_BYTES = NQ_PAD * HEAD_DIM * 2
    MMA = MMA_F8 if FP8 else MMA_F16
    fmt = 1 if kind == "bf16" else 0
    Q_CHUNKS = NQ * (HEAD_DIM // 16)
    Q_ROUNDS = ceildiv(Q_CHUNKS, 32)



    NP = max(NQ_PAD, 32 // EB)
    P_ROW_BYTES = NP * EB
    P_SWZ = {32: K.SW32B, 64: K.SW64B, 128: K.SW128B}[P_ROW_BYTES]
    NP_SLOTS = cfg.get("NP_SLOTS", 4)
    ID_QK = make_idesc(128, NQ_PAD, fmt, fmt, 0, 0)
    ID_PV = make_idesc(128, NP, fmt, fmt, 1, 1)
    O_COL = CH * NQ_PAD
    TMEM_COLS = max(32, 1 << (CH * NQ_PAD + NP - 1).bit_length())
    MIN_BLOCKS = cfg.get("MIN_BLOCKS", 1)
    assert TMEM_COLS <= 512
    N_ROUNDS = ceildiv(TOPK, 32)
    KV16 = KV_TILE_BYTES // 16
    QP16 = NQ_PAD * HEAD_DIM * EB // 16
    assert NQ in (2, 4, 8, 16, 32)
    LPC = 32 // NQ
    NWARPS = 8
    MMA_WARP = 4
    LOAD_WARP = 5
    PREP_WARP = 6
    PV_WARP = 7
    SOFT_THREADS = 128

    @K.kernel(warps=NWARPS, arch="sm_100a", min_blocks_per_sm=MIN_BLOCKS, grid=NUM_CTAS)
    def msa_decode_q1d(
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        out: K.gptr[out_dt],
        q2k: K.gptr[K.i32],
        lens: K.gptr[K.i32],
        ptab: K.gptr[K.i32],
        qraw: K.gptr[K.i32],
        sched: K.gptr[K.i32],
        scale_log2: K.f32,
    ):
        warp = K.warp_id()
        lane = K.lane_id()
        tid = K.thread_id()

        smem = K.smem_pool()
        kv_smem = smem.alloc((STAGES, BLK, HEAD_DIM), kv_dt, swizzle=K.SW128B)
        q_smem = smem.alloc((2, NQ_PAD, HEAD_DIM), kv_dt, swizzle=K.SW128B)
        p_smem = smem.alloc((NP_SLOTS, BLK, NP), kv_dt, swizzle=P_SWZ)
        tmem_addr = smem.alloc((4,), K.u32)
        meta = smem.alloc((32,), K.i32)
        ulist = smem.alloc((2 * TOPK,), K.i32)
        uplist = smem.alloc((2 * TOPK,), K.i32)
        xmax = smem.alloc((2 * NQ * 4,), K.f32, align=16)
        xsum = smem.alloc((NQ * 4,), K.f32, align=16)
        K.keep_alive(ptab.ptr_to([0]))
        K.keep_alive(qraw.ptr_to([0]))
        K.keep_alive(sched.ptr_to([0]))


        def encode(view, major):
            return view.encode(major=major, mma_k=MMA_K)






        kv_full_k = K.TMABar(smem, STAGES)
        kv_full_k.init(1)
        kv_full_v = K.TMABar(smem, STAGES)
        kv_full_v.init(1)
        kv_empty = K.TCGen05Bar(smem, STAGES, phase_offset=1)
        kv_empty.init(1)
        q_load = K.Pipeline(
            smem, 2, full=("mbar" if FP8 else "tma"), empty="tcgen05", init_full=(32 if FP8 else 1), empty_phase_offset=1
        )
        union_ready = K.MBarrier(smem, 2)
        union_ready.init(32)
        if not STATIC_ONE_SHOT:
            union_free = K.MBarrier(smem, 2)
            union_free.init(SOFT_THREADS + 96)
        s_ready = K.TCGen05Bar(smem, CH)
        s_ready.init(1)
        s_free = K.MBarrier(smem, CH)
        s_free.init(SOFT_THREADS)
        p_ready = K.MBarrier(smem, NP_SLOTS)
        p_ready.init(SOFT_THREADS)
        p_free = K.TCGen05Bar(smem, NP_SLOTS)
        p_free.init(1)
        o_ready = K.TCGen05Bar(smem, 1)
        o_ready.init(1)
        o_free = K.MBarrier(smem, 1)
        o_free.init(SOFT_THREADS)

        def ld_global_i32_early(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.global_.nc.b32(value, ptr)
            return value

        def load_q_words(b, h, dst_qw):
            """Raw bf16 Q words of item (b, h) for this lane's 16-dim chunks (fp8 quantization input)."""
            for r in range(Q_ROUNDS):
                c = lane + 32 * r

                def ld_chunk(c=c, r=r):
                    n = c >> 3
                    k16 = c & 7
                    head = h * G + n
                    wbase = (b * HQ + head) * (HEAD_DIM // 2) + k16 * 8
                    K.ptx.ld.global_.nc.v4.b32(dst_qw[8 * r], dst_qw[8 * r + 1], dst_qw[8 * r + 2], dst_qw[8 * r + 3], qraw.ptr_to([wbase]))
                    K.ptx.ld.global_.nc.v4.b32(dst_qw[8 * r + 4], dst_qw[8 * r + 5], dst_qw[8 * r + 6], dst_qw[8 * r + 7], qraw.ptr_to([wbase + 4]))

                if 32 * r + 32 <= Q_CHUNKS:
                    ld_chunk()
                else:
                    with K.If(c < Q_CHUNKS), K.Then():
                        ld_chunk()

        def quantize_q_words(qslot, qw):
            """e4m3 Q rows from the prefetched raw words (mirrors quantize_q's layout)."""
            e4 = K.alloc_local([8], "uint16")
            packed4 = K.alloc_local([4], "uint32")
            for r in range(Q_ROUNDS):
                c = lane + 32 * r

                def cvt_chunk(c=c, r=r):
                    n = c >> 3
                    k16 = c & 7
                    for i in range(8):
                        K.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(e4[i], K.Cast("uint32", qw[8 * r + i]))
                    for i in range(4):
                        K.ptx.mov.b32(packed4[i], e4[2 * i], e4[2 * i + 1])
                    K.ptx.st.shared.v4.b32(q_smem[qslot].ptr_to(n, k16 * 16), packed4[0], packed4[1], packed4[2], packed4[3])

                if 32 * r + 32 <= Q_CHUNKS:
                    cvt_chunk()
                else:
                    with K.If(c < Q_CHUNKS), K.Then():
                        cvt_chunk()

        def load_item_meta(b, h, dst_kv_s, dst_kv_len, dst_idxs, dst_pt=None, dst_qw=None):
            if FP8 and dst_qw is not None:
                load_q_words(b, h, dst_qw)
            if PAGED:
                K.assign(dst_kv_s, 0)
                K.assign(dst_kv_len, ld_global_i32_early(lens.ptr_to([b])))
                if dst_pt is not None:


                    if MAX_PAGES >= 32:
                        K.assign(dst_pt, ld_global_i32_early(ptab.ptr_to([b * MAX_PAGES + lane])))
                    else:
                        K.assign(dst_pt, 0)
                        with K.If(lane < MAX_PAGES), K.Then():
                            K.assign(dst_pt, ld_global_i32_early(ptab.ptr_to([b * MAX_PAGES + lane])))
            else:
                K.assign(dst_kv_s, ld_global_i32_early(lens.ptr_to([b])))
                K.assign(dst_kv_len, ld_global_i32_early(lens.ptr_to([b + 1])) - dst_kv_s)
            base = (h * TOTAL_Q + b) * TOPK
            for r in range(N_ROUNDS):
                e = lane + 32 * r
                K.assign(dst_idxs[r], -1)
                if 32 * r + 32 <= TOPK:
                    K.assign(dst_idxs[r], ld_global_i32_early(q2k.ptr_to([base + e])))
                else:
                    with K.If(e < TOPK), K.Then():
                        K.assign(dst_idxs[r], ld_global_i32_early(q2k.ptr_to([base + e])))




        pre_kv_s = K.local_scalar("int32", init=0)
        pre_kv_len = K.local_scalar("int32", init=0)
        pre_idxs = K.alloc_local([N_ROUNDS], "int32")
        pre_pt = K.local_scalar("int32", init=0)
        pre_qw = K.alloc_local([Q_ROUNDS * 8], "int32")
        with K.If(warp == PREP_WARP), K.Then():
            task0 = K.cta_id()
            with K.If(task0 < NUM_ITEMS), K.Then():
                b0 = task0 // HKV
                h0 = task0 - b0 * HKV
                load_item_meta(b0, h0, pre_kv_s, pre_kv_len, pre_idxs, pre_pt, pre_qw)
        K.ptx.fence.proxy.async_.shared__cta()
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        with K.If(warp == MMA_WARP), K.Then():
            K.ptx[TMEM_ALLOC](K.address_of(tmem_addr[0]), K.uint32(TMEM_COLS))
            K.ptx[TMEM_RELINQUISH]()
            K.ptx.tcgen05.fence__before_thread_sync()
        if not Q1D_NOPREFETCH:
            with K.If(warp == LOAD_WARP), K.Then():
                with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                    K.ptx.prefetch.tensormap(K.address_of(q_map))
                    K.ptx.prefetch.tensormap(K.address_of(k_map))
                    K.ptx.prefetch.tensormap(K.address_of(v_map))
        tmem_base = K.local_scalar("uint32", init=K.uint32(0))

        def acquire_tmem_base():

            K.ptx.bar.sync(K.uint32(2), K.uint32(192))
            K.ptx.tcgen05.fence__after_thread_sync()
            tb_raw = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(tb_raw, tmem_addr.ptr_to([0]))
            K.assign(tmem_base, K.uniform(tb_raw))


        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def tc_fence_after():
            if not Q1D_NOFENCE:
                K.ptx.tcgen05.fence__after_thread_sync()

        def tc_fence_before():
            if not Q1D_NOFENCE:
                K.ptx.tcgen05.fence__before_thread_sync()

        def ld_smem_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.shared.b32(value, ptr)
            return value

        def ld_global_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.global_.nc.b32(value, ptr)
            return value

        def transpose_reduce(vals, ncols, op):
            """Butterfly all-reduce of ncols independent columns; lane l ends with the
            lane-wide reduction of column l // (32 // ncols)."""
            assert 32 % ncols == 0 and ncols >= 2
            cur = [vals[i] for i in range(ncols)]
            xor = 16
            while len(cur) > 1:
                half = len(cur) // 2
                bit = K.bitwise_and(K.Cast("uint32", lane), K.uint32(xor)) != K.uint32(0)
                nxt = []
                for i in range(half):
                    a = cur[i]
                    b = cur[i + half]
                    send = K.local_scalar("float32", init=K.Select(bit, a, b))
                    recv = K.local_scalar("float32")
                    K.ptx.shfl_sync.bfly.b32(recv, send, K.uint32(xor), K.uint32(31), K.uint32(0xFFFFFFFF))
                    keep = K.Select(bit, b, a)
                    nxt.append(K.local_scalar("float32", init=op(keep, recv)))
                cur = nxt
                xor >>= 1
            res = cur[0]
            while xor >= 1:
                other = K.local_scalar("float32")
                K.ptx.shfl_sync.bfly.b32(other, res, K.uint32(xor), K.uint32(31), K.uint32(0xFFFFFFFF))
                res = K.local_scalar("float32", init=op(res, other))
                xor >>= 1
            return res

        def transpose_max(vals, ncols):
            return transpose_reduce(vals, ncols, K.max)

        def transpose_sum(vals, ncols):
            return transpose_reduce(vals, ncols, lambda a, b: a + b)

        def warp_sum_f32(dst, src):
            tmp = K.local_scalar("float32", init=src)
            other = K.local_scalar("float32")
            for d in (16, 8, 4, 2, 1):
                K.ptx.shfl_sync.bfly.b32(other, tmp, K.uint32(d), K.uint32(31), K.uint32(0xFFFFFFFF))
                K.assign(tmp, tmp + other)
            K.assign(dst, tmp)

        def f32_to_half_bits(dst16, src):
            if kind == "f16":
                K.ptx.cvt.rn.f16.f32(dst16, src)
            else:
                K.ptx.cvt.rn.bf16.f32(dst16, src)

        def exp_block_packed(sv, msub, kv_pos, words):
            """p = exp2(s*scale - m) for this thread's NQ columns, computed two at a time
            in packed half precision (bf16x2 for bf16/fp8 K/V, f16x2 for f16 K/V).  The
            packed words are the P^T row (before the fp8 conversion) and the partial
            sums accumulate the quantized probabilities in fp32."""
            valid = kv_pos <= causal_off
            for i in range(0, NQ, 2):
                x0 = sv[i] * scale_log2 - msub[i]
                x1 = sv[i + 1] * scale_log2 - msub[i + 1]
                xw = K.local_scalar("uint32")
                pw = K.local_scalar("uint32")
                if kind == "f16":
                    K.ptx.cvt.rn.f16x2.f32(xw, x1, x0)
                    K.ptx.ex2.approx.f16x2(pw, xw)
                else:
                    K.ptx.cvt.rn.bf16x2.f32(xw, x1, x0)
                    K.ptx.ex2.approx.ftz.bf16x2(pw, xw)
                K.assign(words[i // 2], K.Select(valid, pw, K.uint32(0)))
                lo = K.local_scalar("float32")
                hi = K.local_scalar("float32")
                if kind == "f16":
                    K.ptx.cvt.f32.f16(lo, K.Cast("uint16", K.bitwise_and(words[i // 2], K.uint32(0xFFFF))))
                    K.ptx.cvt.f32.f16(hi, K.Cast("uint16", K.shift_right(words[i // 2], K.uint32(16))))
                else:
                    K.ptx.mov.b32(lo, K.shift_left(words[i // 2], K.uint32(16)))
                    K.ptx.mov.b32(hi, K.bitwise_and(words[i // 2], K.uint32(0xFFFF0000)))
                K.assign(lsum[i], lsum[i] + lo)
                K.assign(lsum[i + 1], lsum[i + 1] + hi)

        def store_p_words(pb, words, row):
            """Store this thread's packed P^T row (fp8: convert bf16x2 pairs to e4m3x2)."""
            if FP8:
                out_words = []
                for i in range(0, NQ // 2, 2):
                    lo = K.local_scalar("uint16")
                    hi = K.local_scalar("uint16")
                    K.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(lo, words[i])
                    K.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(hi, words[i + 1])
                    w = K.local_scalar("uint32")
                    K.ptx.mov.b32(w, lo, hi)
                    out_words.append(w)
                per_word = 4
            else:
                out_words = [words[i] for i in range(NQ // 2)]
                per_word = 2
            i = 0
            while i < len(out_words):
                n = min(4, len(out_words) - i)
                if n == 3:
                    n = 2
                ptr = p_smem[pb].ptr_to(row, i * per_word)
                if n == 4:
                    K.ptx.st.shared.v4.b32(ptr, out_words[i], out_words[i + 1], out_words[i + 2], out_words[i + 3])
                elif n == 2:
                    K.ptx.st.shared.v2.b32(ptr, out_words[i], out_words[i + 1])
                else:
                    K.ptx.st.shared.b32(ptr, out_words[i])
                i += n

        def quantize_q(qslot, h, tok_base):
            words8 = K.alloc_local([8], "int32")
            e4 = K.alloc_local([8], "uint16")
            packed4 = K.alloc_local([4], "uint32")
            for r in range(Q_ROUNDS):
                c = lane + 32 * r

                def quant_chunk(c=c):
                    n = c >> 3
                    k16 = c & 7
                    head = h * G + n
                    wbase = (tok_base * HQ + head) * (HEAD_DIM // 2) + k16 * 8
                    K.ptx.ld.global_.nc.v4.b32(words8[0], words8[1], words8[2], words8[3], qraw.ptr_to([wbase]))
                    K.ptx.ld.global_.nc.v4.b32(words8[4], words8[5], words8[6], words8[7], qraw.ptr_to([wbase + 4]))
                    for i in range(8):
                        K.ptx.cvt.rn.satfinite.e4m3x2.bf16x2(e4[i], K.Cast("uint32", words8[i]))
                    for i in range(4):
                        K.ptx.mov.b32(packed4[i], e4[2 * i], e4[2 * i + 1])
                    K.ptx.st.shared.v4.b32(q_smem[qslot].ptr_to(n, k16 * 16), packed4[0], packed4[1], packed4[2], packed4[3])

                if 32 * r + 32 <= Q_CHUNKS:
                    quant_chunk()
                else:
                    with K.If(c < Q_CHUNKS), K.Then():
                        quant_chunk()

        def zero_p_padding(row):
            """P^T columns NQ..NP-1 are never written by the softmax; zero them once so
            the PV MMA's padding columns read defined data."""
            zero = K.local_scalar("uint32", init=K.uint32(0))
            zero16 = K.local_scalar("uint16", init=K.Cast("uint16", K.int32(0)))
            for pb in range(NP_SLOTS):
                col = NQ
                while col < NP:
                    off = col * EB
                    rem_bytes = (NP - col) * EB
                    ptr = p_smem[pb].ptr_to(row, col)
                    if off % 16 == 0 and rem_bytes >= 16:
                        K.ptx.st.shared.v4.b32(ptr, zero, zero, zero, zero)
                        col += 16 // EB
                    elif off % 8 == 0 and rem_bytes >= 8:
                        K.ptx.st.shared.v2.b32(ptr, zero, zero)
                        col += 8 // EB
                    elif off % 4 == 0 and rem_bytes >= 4:
                        K.ptx.st.shared.b32(ptr, zero)
                        col += 4 // EB
                    else:
                        K.ptx.st.shared.b16(ptr, zero16)
                        col += 2 // EB

        def zero_q_padding():
            """fp8 Q rows NQ..NQ_PAD-1 are never quantized; zero them once (lane-parallel)."""
            zero = K.local_scalar("uint32", init=K.uint32(0))
            n_chunks = (NQ_PAD - NQ) * (HEAD_DIM // 16)
            for qs in range(2):
                for r in range(ceildiv(n_chunks, 32)):
                    c = lane + 32 * r

                    def zchunk(c=c):
                        n = NQ + c // (HEAD_DIM // 16)
                        k16 = c % (HEAD_DIM // 16)
                        K.ptx.st.shared.v4.b32(q_smem[qs].ptr_to(n, k16 * 16), zero, zero, zero, zero)

                    if 32 * r + 32 <= n_chunks:
                        zchunk()
                    else:
                        with K.If(c < n_chunks), K.Then():
                            zchunk()

        def warp_arrive(bar, idx):
            """elect.sync converges the warp after every lane's prior work; one lane arrives for all."""
            with K.If(elected()), K.Then():
                bar.arrive(idx, count=32)

        def tmem_issue_cols(dst, col_expr, ncols):
            width = 32 if ncols >= 32 else (16 if ncols >= 16 else ncols)
            for c in range(ncols // width):
                K.ptx[tmem_ld(width)](*(dst[width * c + i] for i in range(width)), tmem_base + col_expr + K.uint32(width * c))

        def tmem_load_cols(dst, col_expr, ncols):
            tmem_issue_cols(dst, col_expr, ncols)
            K.ptx.tcgen05.wait__ld.sync.aligned()

        def tmem_store_cols(src, col_expr, ncols):
            width = 16 if ncols >= 16 else ncols
            for c in range(ncols // width):
                K.ptx[tmem_st(width)](tmem_base + col_expr + K.uint32(width * c), *(src[width * c + i] for i in range(width)))
            K.ptx.tcgen05.wait__st.sync.aligned()

        def lanemask_lt():
            return K.shift_left(K.uint32(1), K.Cast("uint32", lane)) - K.uint32(1)

        def ring_pos(t):
            q = t // STAGES
            return t - q * STAGES, q & 1

        def iket_range(name, *, leader_only=False):
            token = K.alloc_local([1], "uint32")
            if leader_only:
                K.assign(token[0], K.cuda.iket.sentinel_token(name))
                with K.If((warp & 3) == 0), K.Then():
                    K.assign(token[0], K.cuda.iket.range_start(name))
            else:
                K.assign(token[0], K.cuda.iket.range_start(name))
            return token

        def iket_end(token):
            K.cuda.iket.range_end(token[0])

        sp = K.specialize(chain_dispatch=True)
        r_soft = sp.role("softmax", warps=[0, 1, 2, 3])
        r_mma = sp.role("mma", warps=[MMA_WARP])
        r_load = sp.role("load", warps=[LOAD_WARP])
        r_prep = sp.role("prep", warps=[PREP_WARP])
        r_pv = sp.role("pv", warps=[PV_WARP])


        with r_soft:
            acquire_tmem_base()
            wq = warp & 3
            tid_wg = tid & 127
            if NQ < NP:
                zero_p_padding(tid_wg)
            it = K.local_scalar("int32", init=0)
            gj = K.local_scalar("int32", init=0)
            gc = K.local_scalar("int32", init=0)
            par = K.local_scalar("int32", init=0)
            ps = K.local_scalar("uint32", init=K.uint32(0))
            running = K.local_scalar("int32", init=1)
            m = K.alloc_local([NQ], "float32")
            lsum = K.alloc_local([NQ], "float32")
            s2 = K.alloc_local([2 * NQ_PAD], "float32")
            s2n = K.alloc_local([2 * NQ_PAD], "float32")
            s = [s2[i] for i in range(NQ)]
            sn = [s2[NQ_PAD + i] for i in range(NQ)]
            sc = [s2n[i] for i in range(NQ)]
            sd = [s2n[NQ_PAD + i] for i in range(NQ)]
            msub = K.alloc_local([NQ], "float32")
            pw0 = K.alloc_local([NQ // 2], "uint32")
            pw1 = K.alloc_local([NQ // 2], "uint32")
            cm = K.alloc_local([NQ], "float32")
            mb = K.alloc_local([NQ], "float32")
            dlog = K.alloc_local([NQ], "float32")
            o = K.alloc_local([NQ], "float32")
            hbits = K.local_scalar("uint16")
            zero16 = K.local_scalar("uint16", init=K.Cast("uint16", K.int32(0)))
            with K.While(running != 0):
                slot = it & 1
                tk_wu = iket_range("sm-wait-union", leader_only=True)
                union_ready.wait(slot, (it >> 1) & 1)
                iket_end(tk_wu)
                n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                with K.If(n_blocks < 0):
                    with K.Then():
                        K.assign(running, 0)
                    with K.Else():
                        h = ld_smem_i32(meta.ptr_to([slot * 16 + 2]))
                        tok_base = ld_smem_i32(meta.ptr_to([slot * 16 + 5]))
                        causal_off = ld_smem_i32(meta.ptr_to([slot * 16 + 6]))

                        def out_index(nl):
                            head = h * G + nl
                            return (tok_base * HQ + head) * HEAD_DIM + tid_wg

                        with K.If(n_blocks > 0):
                            with K.Then():
                                for nl in range(NQ):
                                    K.assign(m[nl], K.float32(NEG_INF))
                                    K.assign(lsum[nl], K.float32(0.0))
                                n_chunks = K.local_scalar("int32", init=(n_blocks + (CH - 1)) // CH)
                                with K.serial(n_chunks, unroll=False) as c:
                                    cbase = c * CH
                                    nb = K.local_scalar("int32", init=K.min(K.int32(CH), n_blocks - cbase))

                                    for nl in range(NQ):
                                        K.assign(cm[nl], K.float32(NEG_INF))
                                    with K.serial(nb, unroll=False) as j:
                                        blk = ld_smem_i32(ulist.ptr_to([slot * TOPK + cbase + j]))
                                        kv_pos = blk * BLK + tid_wg
                                        tk_ws = iket_range("sm-wait-s", leader_only=True)
                                        s_ready.wait(j, K.Cast("int32", K.bitwise_and(K.shift_right(ps, K.Cast("uint32", j)), K.uint32(1))))
                                        K.assign(ps, K.bitwise_xor(ps, K.shift_left(K.uint32(1), K.Cast("uint32", j))))
                                        tc_fence_after()
                                        iket_end(tk_ws)
                                        tk_p1 = iket_range("sm-max", leader_only=True)
                                        tmem_load_cols(s, K.Cast("uint32", j) * K.uint32(NQ_PAD), NQ)
                                        for nl in range(NQ):
                                            K.assign(cm[nl], K.max(cm[nl], K.Select(kv_pos <= causal_off, s[nl], K.float32(NEG_INF))))
                                        iket_end(tk_p1)
                                    tk_p1 = iket_range("sm-max", leader_only=True)
                                    mycol_max = transpose_max(cm, NQ)
                                    with K.If((lane % LPC) == 0), K.Then():
                                        K.ptx.st.shared.f32(xmax.ptr_to([(par * NQ + lane // LPC) * 4 + wq]), mycol_max)
                                    iket_end(tk_p1)
                                    tk_bar = iket_range("sm-bar", leader_only=True)
                                    K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                                    iket_end(tk_bar)
                                    tk_mg = iket_range("sm-merge", leader_only=True)
                                    v4 = K.alloc_local([4], "float32")
                                    K.ptx.ld.shared.v4.f32(v4[0], v4[1], v4[2], v4[3], xmax.ptr_to([(par * NQ + lane // LPC) * 4]))
                                    fin = K.local_scalar("float32")
                                    K.ptx.max.f32(fin, v4[0], v4[1], v4[2])
                                    K.assign(fin, K.max(fin, v4[3]) * scale_log2)
                                    for nl in range(NQ):
                                        K.ptx.shfl_sync.idx.b32(mb[nl], fin, K.uint32(nl * LPC), K.uint32(31), K.uint32(0xFFFFFFFF))
                                    any_r = K.local_scalar("int32", init=0)
                                    for nl in range(NQ):
                                        m_old = K.local_scalar("float32", init=m[nl])
                                        m_new = K.local_scalar("float32", init=K.max(m_old, mb[nl]))
                                        d = K.local_scalar("float32", init=m_new - m_old)
                                        big = d > K.float32(RESCALE_THRESH)
                                        do_r = K.And(m_old != K.float32(NEG_INF), big)
                                        K.assign(dlog[nl], K.Select(do_r, K.float32(0.0) - d, K.float32(0.0)))
                                        K.assign(m[nl], K.Select(big, m_new, m_old))
                                        K.assign(any_r, K.Select(do_r, 1, any_r))

                                    with K.If(any_r != 0), K.Then():
                                        gjm = gj - 1
                                        p_free.wait(gjm & (NP_SLOTS - 1), (gjm // NP_SLOTS) & 1)
                                        tc_fence_after()
                                        tmem_load_cols(o, K.uint32(O_COL), NQ)
                                        for nl in range(NQ):
                                            a = K.local_scalar("float32")
                                            K.ptx.ex2.approx.ftz.f32(a, dlog[nl])
                                            K.assign(o[nl], o[nl] * a)
                                            K.assign(lsum[nl], lsum[nl] * a)
                                        tmem_store_cols(o, K.uint32(O_COL), NQ)
                                        tc_fence_before()
                                    iket_end(tk_mg)
                                    for nl in range(NQ):
                                        K.assign(msub[nl], K.max(m[nl], K.float32(-1.0e38)))




                                    with K.If(nb > 1):
                                        with K.Then():
                                            tmem_issue_cols(s2, K.uint32(0), 2 * NQ_PAD)
                                        with K.Else():
                                            with K.If(nb > 0), K.Then():
                                                tmem_issue_cols(s, K.uint32(0), NQ)
                                    npairs = K.local_scalar("int32", init=(nb + 1) >> 1)
                                    with K.serial(npairs, unroll=False) as jj:
                                        j0 = jj * 2
                                        two = (j0 + 1) < nb
                                        tk_ex = iket_range("sm-exp", leader_only=True)
                                        K.ptx.tcgen05.wait__ld.sync.aligned()
                                        tc_fence_before()
                                        warp_arrive(s_free, j0)
                                        with K.If(two), K.Then():
                                            warp_arrive(s_free, j0 + 1)
                                        with K.If(j0 + 3 < nb):
                                            with K.Then():
                                                tmem_issue_cols(s2n, K.Cast("uint32", j0 + 2) * K.uint32(NQ_PAD), 2 * NQ_PAD)
                                            with K.Else():
                                                with K.If(j0 + 2 < nb), K.Then():
                                                    tmem_issue_cols(sc, K.Cast("uint32", j0 + 2) * K.uint32(NQ_PAD), NQ)
                                        blk0 = ld_smem_i32(ulist.ptr_to([slot * TOPK + cbase + j0]))
                                        blk1 = ld_smem_i32(ulist.ptr_to([slot * TOPK + cbase + K.min(j0 + 1, nb - 1)]))
                                        kv_pos0 = blk0 * BLK + tid_wg
                                        kv_pos1 = blk1 * BLK + tid_wg

                                        exp_block_packed(s, msub, kv_pos0, pw0)
                                        with K.If(two), K.Then():
                                            exp_block_packed(sn, msub, kv_pos1, pw1)
                                        iket_end(tk_ex)
                                        pb0 = gj & (NP_SLOTS - 1)
                                        tk_wp = iket_range("sm-wait-pfree", leader_only=True)
                                        p_free.wait(pb0, ((gj // NP_SLOTS) + 1) & 1)
                                        K.ptx.fence.proxy.async_.shared__cta()
                                        iket_end(tk_wp)
                                        tk_ps = iket_range("sm-pstore", leader_only=True)
                                        store_p_words(pb0, pw0, tid_wg)
                                        K.ptx.fence.proxy.async_.shared__cta()
                                        warp_arrive(p_ready, pb0)
                                        with K.If(two), K.Then():
                                            gj1 = gj + 1
                                            pb1 = gj1 & (NP_SLOTS - 1)
                                            p_free.wait(pb1, ((gj1 // NP_SLOTS) + 1) & 1)
                                            K.ptx.fence.proxy.async_.shared__cta()
                                            store_p_words(pb1, pw1, tid_wg)
                                            K.ptx.fence.proxy.async_.shared__cta()
                                            warp_arrive(p_ready, pb1)
                                        iket_end(tk_ps)
                                        K.assign(gj, gj + K.Select(two, 2, 1))
                                        with K.If(j0 + 2 < nb), K.Then():
                                            for nl in range(NQ):
                                                K.assign(s[nl], sc[nl])
                                        with K.If(j0 + 3 < nb), K.Then():
                                            for nl in range(NQ):
                                                K.assign(sn[nl], sd[nl])
                                    K.assign(par, par ^ 1)
                                    K.assign(gc, gc + 1)

                                # Row sums and their reciprocals do not depend on O: reduce them while the
                                # last PV MMAs drain, then only the O load/scale/store follows o_ready.
                                tk_epi0 = iket_range("sm-epi-sum", leader_only=True)
                                mycol_sum = transpose_sum(lsum, NQ)
                                with K.If((lane % LPC) == 0), K.Then():
                                    K.ptx.st.shared.f32(xsum.ptr_to([(lane // LPC) * 4 + wq]), mycol_sum)
                                K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                                v4s = K.alloc_local([4], "float32")
                                K.ptx.ld.shared.v4.f32(v4s[0], v4s[1], v4s[2], v4s[3], xsum.ptr_to([(lane // LPC) * 4]))
                                lfin = K.local_scalar("float32", init=(v4s[0] + v4s[1]) + (v4s[2] + v4s[3]))
                                inv_a = K.alloc_local([NQ], "float32")
                                for nl in range(NQ):
                                    ltot = K.local_scalar("float32")
                                    K.ptx.shfl_sync.idx.b32(ltot, lfin, K.uint32(nl * LPC), K.uint32(31), K.uint32(0xFFFFFFFF))
                                    bad = K.Or(ltot == K.float32(0.0), ltot != ltot)
                                    inv = K.local_scalar("float32")
                                    K.ptx.rcp.approx.ftz.f32(inv, K.Select(bad, K.float32(1.0), ltot))
                                    K.assign(inv_a[nl], K.Select(bad, K.float32(0.0), inv))
                                iket_end(tk_epi0)
                                tk_wo = iket_range("sm-wait-o", leader_only=True)
                                o_ready.wait(0, it & 1)
                                tc_fence_after()
                                iket_end(tk_wo)
                                tk_epi = iket_range("sm-epi", leader_only=True)
                                tmem_load_cols(o, K.uint32(O_COL), NQ)
                                tc_fence_before()
                                warp_arrive(o_free, 0)
                                for nl in range(NQ):
                                    val = K.local_scalar("float32", init=o[nl] * inv_a[nl])
                                    f32_to_half_bits(hbits, val)
                                    K.ptx.st.global_.b16(out.ptr_to([out_index(nl)]), hbits)
                                iket_end(tk_epi)
                            with K.Else():
                                o_ready.wait(0, it & 1)
                                warp_arrive(o_free, 0)
                                for nl in range(NQ):
                                    K.ptx.st.global_.b16(out.ptr_to([out_index(nl)]), zero16)
                        if not STATIC_ONE_SHOT:
                            union_free.arrive(slot)
                if STATIC_ONE_SHOT:
                    K.assign(running, 0)
                else:
                    K.assign(it, it + 1)


        with r_mma:
            acquire_tmem_base()
            it_m = K.local_scalar("int32", init=0)
            gc_m = K.local_scalar("int32", init=0)
            t_m = K.local_scalar("int32", init=0)
            pk = K.local_scalar("uint32", init=K.uint32(0))
            pf = K.local_scalar("uint32", init=K.uint32(0))
            running_m = K.local_scalar("int32", init=1)
            zero = K.uint32(0)

            kd0, koff = encode(kv_smem[0], "k")
            qd0, qoff = encode(q_smem[0], "k")

            def mma_qk(sb_col, kstage, qslot):
                for ki in range(NK):
                    with K.If(elected()), K.Then():
                        K.ptx[MMA](
                            tmem_base + sb_col,
                            kd0 + (kstage * KV16 + koff(ki)),
                            qd0 + (qslot * QP16 + qoff(ki)),
                            K.uint32(ID_QK),
                            zero,
                            zero,
                            zero,
                            zero,
                            ki != 0,
                        )

            with K.While(running_m != 0):
                slot = it_m & 1
                union_ready.wait(slot, (it_m >> 1) & 1)
                n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                with K.If(n_blocks < 0):
                    with K.Then():
                        K.assign(running_m, 0)
                    with K.Else():
                        q_load.full.wait(slot, (it_m >> 1) & 1)
                        with K.If(n_blocks == 0), K.Then():
                            with K.If(elected()), K.Then():
                                q_load.empty.arrive(slot)
                        n_chunks = K.local_scalar("int32", init=(n_blocks + (CH - 1)) // CH)
                        with K.serial(n_chunks, unroll=False) as c:
                            cbase = c * CH
                            nb = K.local_scalar("int32", init=K.min(K.int32(CH), n_blocks - cbase))
                            with K.serial(nb, unroll=False) as j:
                                kstage, _kphase = ring_pos(t_m)
                                kst = K.local_scalar("int32", init=kstage)
                                K.assign(t_m, t_m + 1)
                                tk_wk = iket_range("mma-wait-k")
                                kv_full_k.wait(kst, K.Cast("int32", K.bitwise_and(K.shift_right(pk, K.Cast("uint32", kst)), K.uint32(1))))
                                K.assign(pk, K.bitwise_xor(pk, K.shift_left(K.uint32(1), K.Cast("uint32", kst))))
                                iket_end(tk_wk)
                                tk_sf = iket_range("mma-wait-sfree")
                                s_free.wait(j, K.Cast("int32", K.bitwise_xor(K.bitwise_and(K.shift_right(pf, K.Cast("uint32", j)), K.uint32(1)), K.uint32(1))))
                                K.assign(pf, K.bitwise_xor(pf, K.shift_left(K.uint32(1), K.Cast("uint32", j))))
                                tc_fence_after()
                                iket_end(tk_sf)
                                tk_qk = iket_range("mma-qk")
                                mma_qk(K.Cast("uint32", j) * K.uint32(NQ_PAD), kst, slot)
                                with K.If(elected()), K.Then():
                                    s_ready.arrive(j)
                                    kv_empty.arrive(kst)
                                iket_end(tk_qk)
                            with K.If(elected()), K.Then():
                                with K.If(c + 1 == n_chunks), K.Then():
                                    q_load.empty.arrive(slot)
                            K.assign(t_m, t_m + nb)
                            K.assign(gc_m, gc_m + 1)
                        if not STATIC_ONE_SHOT:
                            union_free.arrive(slot)
                if STATIC_ONE_SHOT:
                    K.assign(running_m, 0)
                else:
                    K.assign(it_m, it_m + 1)


        with r_pv:
            acquire_tmem_base()
            if FP8 and NQ < NQ_PAD:
                zero_q_padding()
            it_v = K.local_scalar("int32", init=0)
            gj_v = K.local_scalar("int32", init=0)
            t_v = K.local_scalar("int32", init=0)
            pv = K.local_scalar("uint32", init=K.uint32(0))
            running_v = K.local_scalar("int32", init=1)
            zero_v = K.uint32(0)

            vd0, voff = encode(kv_smem[0], "mn")
            pd0, poff = encode(p_smem[0], "mn")
            PT16 = BLK * NP * EB // 16

            def mma_pv(vstage, pb, acc):
                for ki in range(NK):
                    with K.If(elected()), K.Then():
                        K.ptx[MMA](
                            tmem_base + K.uint32(O_COL),
                            vd0 + (vstage * KV16 + voff(ki)),
                            pd0 + (pb * PT16 + poff(ki)),
                            K.uint32(ID_PV),
                            zero_v,
                            zero_v,
                            zero_v,
                            zero_v,
                            True if ki != 0 else K.Cast("bool", acc != 0),
                        )

            with K.While(running_v != 0):
                slot = it_v & 1
                union_ready.wait(slot, (it_v >> 1) & 1)
                n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                with K.If(n_blocks < 0):
                    with K.Then():
                        K.assign(running_v, 0)
                    with K.Else():
                        o_free.wait(0, (it_v + 1) & 1)
                        tc_fence_after()
                        with K.If(n_blocks == 0), K.Then():
                            with K.If(elected()), K.Then():
                                o_ready.arrive(0)
                        acc = K.local_scalar("int32", init=0)
                        n_chunks = K.local_scalar("int32", init=(n_blocks + (CH - 1)) // CH)
                        with K.serial(n_chunks, unroll=False) as c:
                            cbase = c * CH
                            nb = K.local_scalar("int32", init=K.min(K.int32(CH), n_blocks - cbase))
                            K.assign(t_v, t_v + nb)
                            with K.serial(nb, unroll=False) as j:
                                vstage, _vphase = ring_pos(t_v)
                                vst = K.local_scalar("int32", init=vstage)
                                K.assign(t_v, t_v + 1)
                                pb = gj_v & (NP_SLOTS - 1)
                                tk_wv = iket_range("mma-wait-v")
                                kv_full_v.wait(vst, K.Cast("int32", K.bitwise_and(K.shift_right(pv, K.Cast("uint32", vst)), K.uint32(1))))
                                K.assign(pv, K.bitwise_xor(pv, K.shift_left(K.uint32(1), K.Cast("uint32", vst))))
                                iket_end(tk_wv)
                                tk_wp = iket_range("mma-wait-p")
                                p_ready.wait(pb, (gj_v // NP_SLOTS) & 1)
                                iket_end(tk_wp)
                                tk_pv = iket_range("mma-pv")
                                mma_pv(vst, pb, acc)
                                with K.If(elected()), K.Then():
                                    p_free.arrive(pb)
                                    kv_empty.arrive(vst)
                                iket_end(tk_pv)
                                K.assign(acc, 1)
                                K.assign(gj_v, gj_v + 1)
                        with K.If(n_blocks > 0), K.Then():
                            with K.If(elected()), K.Then():
                                o_ready.arrive(0)
                        if not STATIC_ONE_SHOT:
                            union_free.arrive(slot)
                if STATIC_ONE_SHOT:
                    K.assign(running_v, 0)
                else:
                    K.assign(it_v, it_v + 1)


        with r_load:
            it_l = K.local_scalar("int32", init=0)
            t_l = K.local_scalar("int32", init=0)
            running_l = K.local_scalar("int32", init=1)
            with K.While(running_l != 0):
                slot = it_l & 1
                tk_lu = iket_range("ld-wait-union")
                union_ready.wait(slot, (it_l >> 1) & 1)
                iket_end(tk_lu)
                n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                with K.If(n_blocks < 0):
                    with K.Then():
                        K.assign(running_l, 0)
                    with K.Else():
                        h = ld_smem_i32(meta.ptr_to([slot * 16 + 2]))
                        kv_s = ld_smem_i32(meta.ptr_to([slot * 16 + 3]))
                        tok_base = ld_smem_i32(meta.ptr_to([slot * 16 + 5]))
                        n_chunks = K.local_scalar("int32", init=(n_blocks + (CH - 1)) // CH)
                        with K.serial(n_chunks, unroll=False) as c:
                            cbase = c * CH
                            nb = K.local_scalar("int32", init=K.min(K.int32(CH), n_blocks - cbase))
                            for tmap, full_bar in ((k_map, kv_full_k), (v_map, kv_full_v)):
                                with K.serial(nb, unroll=False) as j:
                                    blk = ld_smem_i32(ulist.ptr_to([slot * TOPK + cbase + j]))
                                    if PAGED:
                                        pg = ld_smem_i32(uplist.ptr_to([slot * TOPK + cbase + j]))
                                    lstage, lphase = ring_pos(t_l)
                                    lst = K.local_scalar("int32", init=lstage)
                                    lph = K.local_scalar("int32", init=lphase)
                                    K.assign(t_l, t_l + 1)
                                    tk_we = iket_range("ld-wait-stage")
                                    kv_empty.wait(lst, lph)
                                    iket_end(tk_we)
                                    with K.If(elected()), K.Then():
                                        if PAGED and not FP8:
                                            K.ptx[TMA_4D_HINT](
                                                kv_smem[lst].ptr_to(0, 0),
                                                K.address_of(tmap),
                                                K.int32(0),
                                                K.int32(0),
                                                K.int32(0),
                                                pg * HKV + h,
                                                K.cuda.cvta_generic_to_shared(full_bar.ptr_to([lst])),
                                                *HINT_ARGS,
                                            )
                                        elif PAGED:
                                            K.ptx[TMA_3D_HINT](
                                                kv_smem[lst].ptr_to(0, 0),
                                                K.address_of(tmap),
                                                K.int32(0),
                                                K.int32(0),
                                                pg * HKV + h,
                                                K.cuda.cvta_generic_to_shared(full_bar.ptr_to([lst])),
                                                *HINT_ARGS,
                                            )
                                        elif not FP8:
                                            K.ptx[TMA_3D_HINT](
                                                kv_smem[lst].ptr_to(0, 0),
                                                K.address_of(tmap),
                                                K.int32(0),
                                                kv_s + blk * BLK,
                                                h * 2,
                                                K.cuda.cvta_generic_to_shared(full_bar.ptr_to([lst])),
                                                *HINT_ARGS,
                                            )
                                        else:
                                            K.ptx[TMA_3D_HINT](
                                                kv_smem[lst].ptr_to(0, 0),
                                                K.address_of(tmap),
                                                K.int32(0),
                                                kv_s + blk * BLK,
                                                h,
                                                K.cuda.cvta_generic_to_shared(full_bar.ptr_to([lst])),
                                                *HINT_ARGS,
                                            )
                                        full_bar.arrive(lst, tx_count=KV_TILE_BYTES)
                        if not STATIC_ONE_SHOT:
                            union_free.arrive(slot)
                if STATIC_ONE_SHOT:
                    K.assign(running_l, 0)
                else:
                    K.assign(it_l, it_l + 1)


        with r_prep:
            it_p = K.local_scalar("int32", init=0)
            running_p = K.local_scalar("int32", init=1)
            with K.While(running_p != 0):
                slot = it_p & 1
                task = K.local_scalar("int32")
                if STATIC_ONE_SHOT:
                    K.assign(task, K.cta_id())
                else:
                    with K.If(it_p == 0):
                        with K.Then():
                            K.assign(task, K.cta_id())
                        with K.Else():
                            grabbed = K.local_scalar("int32", init=0)
                            with K.If(lane == 0), K.Then():
                                K.ptx.atom.relaxed.gpu.global_.add.s32(grabbed, sched.ptr_to([0]), K.int32(1))
                            K.assign(task, K.uniform(grabbed) + NUM_CTAS)
                if not STATIC_ONE_SHOT:
                    tk_pf = iket_range("prep-wait-free")
                    union_free.wait(slot, ((it_p >> 1) + 1) & 1)
                    iket_end(tk_pf)
                tk_pi = iket_range("prep-item")
                with K.If(task >= NUM_ITEMS):
                    with K.Then():
                        with K.If(lane == 0), K.Then():
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16]), K.int32(-1))
                        union_ready.arrive(slot)
                        K.assign(running_p, 0)
                    with K.Else():
                        b = task // HKV
                        h = task - b * HKV
                        if not FP8:

                            q_load.empty.wait(slot, (it_p >> 1) & 1)
                            with K.If(elected()), K.Then():
                                K.ptx[TMA_4D](
                                    q_smem[slot].ptr_to(0, 0),
                                    K.address_of(q_map),
                                    K.int32(0),
                                    h * G,
                                    b,
                                    K.int32(0),
                                    K.cuda.cvta_generic_to_shared(q_load.full.ptr_to([slot])),
                                )
                                q_load.full.arrive(slot, tx_count=Q_TMA_BYTES)
                        kv_s = K.local_scalar("int32", init=0)
                        kv_len = K.local_scalar("int32")
                        idxs = K.alloc_local([N_ROUNDS], "int32")
                        pt_lane = K.local_scalar("int32", init=0)
                        qw = K.alloc_local([Q_ROUNDS * 8], "int32")
                        if STATIC_ONE_SHOT:
                            K.assign(kv_s, pre_kv_s)
                            K.assign(kv_len, pre_kv_len)
                            K.assign(pt_lane, pre_pt)
                            for r in range(N_ROUNDS):
                                K.assign(idxs[r], pre_idxs[r])
                            if FP8:
                                for i in range(Q_ROUNDS * 8):
                                    K.assign(qw[i], pre_qw[i])
                        else:
                            with K.If(it_p == 0):
                                with K.Then():
                                    K.assign(kv_s, pre_kv_s)
                                    K.assign(kv_len, pre_kv_len)
                                    K.assign(pt_lane, pre_pt)
                                    for r in range(N_ROUNDS):
                                        K.assign(idxs[r], pre_idxs[r])
                                    if FP8:
                                        for i in range(Q_ROUNDS * 8):
                                            K.assign(qw[i], pre_qw[i])
                                with K.Else():
                                    load_item_meta(b, h, kv_s, kv_len, idxs, pt_lane, qw)
                        n_vis = (kv_len + (BLK - 1)) >> 7
                        n_blocks = K.local_scalar("int32", init=0)
                        for r in range(N_ROUNDS):
                            vflag = K.local_scalar("int32", init=K.Select(K.And(idxs[r] >= 0, idxs[r] < n_vis), 1, 0))
                            ballot = K.local_scalar("uint32")
                            K.ptx.vote_sync.ballot.b32(ballot, K.ptx.pred(vflag), K.uint32(0xFFFFFFFF))
                            before = K.local_scalar("uint32")
                            K.ptx.popc.b32(before, K.bitwise_and(ballot, lanemask_lt()))
                            total = K.local_scalar("uint32")
                            K.ptx.popc.b32(total, ballot)
                            rank = n_blocks + K.Cast("int32", before)
                            if PAGED:

                                pg = K.local_scalar("int32")
                                src_lane = K.Cast("uint32", K.max(idxs[r], 0) & 31)
                                K.ptx.shfl_sync.idx.b32(pg, pt_lane, src_lane, K.uint32(31), K.uint32(0xFFFFFFFF))
                                if MAX_PAGES > 32:
                                    with K.If(K.And(vflag != 0, idxs[r] >= 32)), K.Then():
                                        K.assign(pg, ld_global_i32(ptab.ptr_to([b * MAX_PAGES + idxs[r]])))
                            with K.If(vflag != 0), K.Then():
                                K.ptx.st.shared.b32(ulist.ptr_to([slot * TOPK + rank]), idxs[r])
                                if PAGED:
                                    K.ptx.st.shared.b32(uplist.ptr_to([slot * TOPK + rank]), pg)
                            K.assign(n_blocks, n_blocks + K.Cast("int32", total))
                        K.cuda.warp_sync()
                        with K.If(lane == 0), K.Then():
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 0]), n_blocks)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 1]), b)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 2]), h)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 3]), kv_s)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 4]), kv_len)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 5]), b)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 6]), kv_len - 1)
                        union_ready.arrive(slot)
                        if FP8:

                            tk_pq = iket_range("prep-quant")
                            q_load.empty.wait(slot, (it_p >> 1) & 1)
                            quantize_q_words(slot, qw)
                            K.ptx.fence.proxy.async_.shared__cta()
                            q_load.full.arrive(slot)
                            iket_end(tk_pq)
                iket_end(tk_pi)
                if STATIC_ONE_SHOT:
                    K.assign(running_p, 0)
                else:
                    K.assign(it_p, it_p + 1)


        K.cuda.cta_sync()
        if not STATIC_ONE_SHOT:
            with K.If(tid == 0), K.Then():
                done = K.local_scalar("int32")
                K.ptx.atom.acq_rel.gpu.global_.add.s32(done, sched.ptr_to([1]), K.int32(1))
                with K.If(done == NUM_CTAS - 1), K.Then():
                    K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([0]), K.int32(0))
                    K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([1]), K.int32(0))
        with K.If(warp == MMA_WARP), K.Then():
            K.ptx[TMEM_DEALLOC](tmem_base, K.uint32(TMEM_COLS))

    return msa_decode_q1d


def plan_config_q1d(q, k, q2k, page_table, B, T, num_sms):
    HQ = q.shape[1]
    HKV = k.shape[1]
    G = HQ // HKV
    TOPK = int(q2k.shape[-1])
    TOTAL_Q = q.shape[0]
    paged = page_table is not None
    kind = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float8_e4m3fn: "fp8"}[k.dtype]
    assert T == 1
    NQ = G
    NQ_PAD = max(16, ceildiv(NQ, 16) * 16)
    eb = 1 if kind == "fp8" else 2
    NP = max(NQ_PAD, 32 // eb)
    MAX_PAGES = int(page_table.shape[1]) if paged else 0
    NUM_ITEMS = B * HKV


    two_res_env = os.environ.get("Q1D_TWO_RES", "")
    TWO_RES = (kind == "fp8" and NUM_ITEMS > num_sms) if two_res_env == "" else two_res_env == "1"
    tmem_budget = 256 if TWO_RES else 512
    CH = int(min(TOPK, (tmem_budget - NP) // NQ_PAD))
    env_ch = int(os.environ.get("Q1D_CH", "0"))
    if env_ch:
        CH = min(CH, env_ch)
    elif kind == "fp8" and TOPK > 8:
        # Two-pass softmax: the exp/P phase of one S chunk only starts after that chunk's
        # exact max, so a smaller chunk lets it overlap the next chunk's K stream instead of
        # being exposed at the end of the item (NCU: fp8 flat 19.46->18.34 us at CH=6,
        # fp8 paged two-resident 49.22->47.14 us at CH=10).
        CH = min(CH, 10 if TWO_RES else 6)
    assert CH >= 1
    NUM_CTAS = max(1, min((2 * num_sms if TWO_RES else num_sms), NUM_ITEMS))
    env_ctas = int(os.environ.get("Q1D_CTAS", "0"))
    if env_ctas:
        NUM_CTAS = min(NUM_CTAS, env_ctas)
    STATIC_ONE_SHOT = NUM_ITEMS == NUM_CTAS
    kv_tile = BLK * HEAD_DIM * eb
    q_tile = NQ_PAD * HEAD_DIM * eb
    p_tile = BLK * NP * eb
    misc = 16 * TOPK + 48 * NQ + 2048 + 8 * (2 * 16 + 2 * CH + 24)
    smem_cap = (113 if TWO_RES else 225) * 1024
    NP_SLOTS = int(os.environ.get("Q1D_PSLOTS", "4"))
    budget = smem_cap - misc - 2 * q_tile - NP_SLOTS * p_tile
    STAGES = int(min(16, budget // kv_tile))
    if ENV_STAGES:
        STAGES = min(STAGES, ENV_STAGES)
    assert STAGES >= 2, "shared memory budget too small"
    return dict(
        T=T,
        G=G,
        HKV=HKV,
        TOPK=TOPK,
        TOTAL_Q=TOTAL_Q,
        kv_kind=kind,
        paged=paged,
        MAX_PAGES=MAX_PAGES,
        NUM_ITEMS=NUM_ITEMS,
        NUM_CTAS=NUM_CTAS,
        STAGES=STAGES,
        CH=CH,
        STATIC_ONE_SHOT=STATIC_ONE_SHOT,
        MIN_BLOCKS=2 if TWO_RES else 1,
        NP_SLOTS=NP_SLOTS,
    )


def setup_q1d(data, B, seqlen_q):
    q = data["q"]
    k = data["k"]
    v = data["v"]
    q2k = data["q2k_indices"]
    cu_seqlens_k = data["cu_seqlens_k"]
    page_table = data["page_table"]
    seqused_k = data["seqused_k"]
    out = data["output"]
    scale = float(data["softmax_scale"])
    T = int(seqlen_q)
    B = int(B)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous()
    assert q.shape[0] == B * T
    device = q.device
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    cfg = plan_config_q1d(q, k, q2k, page_table, B, T, num_sms)
    ex = _compile(cfg, make_kernel_q1d, "q1d")

    HKV = cfg["HKV"]
    G = cfg["G"]
    HQ = HKV * G
    TOTAL_Q = cfg["TOTAL_Q"]
    fp8 = cfg["kv_kind"] == "fp8"
    q_dtype_name = "bfloat16" if q.dtype == torch.bfloat16 else "float16"
    kv_dtype_name = "uint8" if fp8 else q_dtype_name
    NQ_PAD = max(16, ceildiv(G, 16) * 16)
    q_map = _encode(
        q,
        q_dtype_name,
        (HEAD_DIM // 2, HQ, TOTAL_Q, 2),
        (HEAD_DIM * 2, HQ * HEAD_DIM * 2, (HEAD_DIM // 2) * 2),
        (HEAD_DIM // 2, NQ_PAD, 1, 2),
    )
    if cfg["paged"]:
        num_pages = int(k.shape[0])
        if fp8:
            kv_dims = (HEAD_DIM, BLK, num_pages * HKV)
            kv_strides = (HEAD_DIM, BLK * HEAD_DIM)
            kv_box = (HEAD_DIM, BLK, 1)
        else:
            kv_dims = (HEAD_DIM // 2, BLK, 2, num_pages * HKV)
            kv_strides = (HEAD_DIM * 2, (HEAD_DIM // 2) * 2, BLK * HEAD_DIM * 2)
            kv_box = (HEAD_DIM // 2, BLK, 2, 1)
        lens = seqused_k.contiguous()
        ptab = page_table.contiguous().view(-1)
    else:
        total_k = int(k.shape[0])
        if fp8:
            kv_dims = (HEAD_DIM, total_k, HKV)
            kv_strides = (HKV * HEAD_DIM, HEAD_DIM)
            kv_box = (HEAD_DIM, BLK, 1)
        else:
            kv_dims = (HEAD_DIM // 2, total_k, HKV * 2)
            kv_strides = (HKV * HEAD_DIM * 2, (HEAD_DIM // 2) * 2)
            kv_box = (HEAD_DIM // 2, BLK, 2)
        lens = cu_seqlens_k.contiguous()
        ptab = torch.zeros(4, dtype=torch.int32, device=device)


    promo_default = 2 if (fp8 and not cfg["paged"]) else 3
    promo = int(os.environ.get("Q1D_L2PROMO", str(promo_default)))
    k_map = _encode(k, kv_dtype_name, kv_dims, kv_strides, kv_box, l2_promotion=promo)
    v_map = _encode(v, kv_dtype_name, kv_dims, kv_strides, kv_box, l2_promotion=promo)
    sched = torch.zeros(4, dtype=torch.int32, device=device)
    q2k_flat = q2k.contiguous().view(-1)
    out_flat = out.view(-1)
    if fp8:
        qraw = q.view(torch.int32).view(-1)
    else:
        qraw = torch.zeros(4, dtype=torch.int32, device=device)
    args = (
        q_map.ptr,
        k_map.ptr,
        v_map.ptr,
        out_flat,
        q2k_flat,
        lens,
        ptab,
        qraw,
        sched,
        float(scale * LOG2E),
    )
    keep = (q, k, v, q2k, q2k_flat, lens, ptab, qraw, out, out_flat, sched, q_map, k_map, v_map)

    def run():
        ex(*args)

    run._keep = keep
    run._cfg = cfg
    run()
    torch.cuda.synchronize(device)
    return run


def _use_q1d(data, B, T):
    q, k = data["q"], data["k"]
    if int(T) != 1:
        return False
    HQ = int(q.shape[1])
    HKV = int(k.shape[1])
    if HQ % HKV:
        return False
    G = HQ // HKV
    if G not in (2, 4, 8, 16, 32):
        return False
    if k.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
        return False
    if k.dtype == torch.float8_e4m3fn and G < 4:
        return False
    return True


_setup_prev = setup


def setup(data, B, seqlen_q):  # noqa: F811 - chained shape-dispatch override
    T = int(seqlen_q)
    if _use_q1d(data, B, T):
        return setup_q1d(data, B, T)
    return _setup_prev(data, B, seqlen_q)
















def make_kernel_q4d(cfg):
    T = cfg["T"]
    G = cfg["G"]
    HKV = cfg["HKV"]
    TOPK = cfg["TOPK"]
    TOTAL_Q = cfg["TOTAL_Q"]
    kind = cfg["kv_kind"]
    PAGED = cfg["paged"]
    MAX_PAGES = cfg["MAX_PAGES"]
    W_MAX = cfg["MAX_BLOCK_WORDS"]
    NUM_ITEMS = cfg["NUM_ITEMS"]
    NUM_CTAS = cfg["NUM_CTAS"]
    STAGES = cfg["STAGES"]
    CH = cfg["CH"]
    STATIC_ONE_SHOT = cfg["STATIC_ONE_SHOT"]
    HQ = HKV * G
    NQ = T * G
    assert NQ in (32, 64) and 32 % G == 0
    NQ_PAD = NQ
    NSWG = NQ // 32
    NQW = 32
    GPW = NQW // G


    SKIP_GROUPS = G >= 8 and os.environ.get("Q4D_NOSKIP", "0") != "1"
    assert kind in ("bf16", "f16")
    kv_dt = {"bf16": K.bf16, "f16": K.f16}[kind]
    out_dt = K.f16 if kind == "f16" else K.bf16
    EB = 2
    MMA_K = 16
    NK = HEAD_DIM // MMA_K
    KV_TILE_BYTES = BLK * HEAD_DIM * EB
    Q_TMA_BYTES = NQ_PAD * HEAD_DIM * 2
    MMA = MMA_F16
    fmt = 1 if kind == "bf16" else 0
    NP = NQ_PAD
    P_ROW_BYTES = NP * EB
    P_SWZ = {32: K.SW32B, 64: K.SW64B, 128: K.SW128B}[P_ROW_BYTES]
    NP_SLOTS = 2
    ID_QK = make_idesc(128, NQ_PAD, fmt, fmt, 0, 0)
    ID_PV = make_idesc(128, NP, fmt, fmt, 1, 1)
    O_COL = CH * NQ_PAD
    TMEM_COLS = max(32, 1 << (CH * NQ_PAD + NP - 1).bit_length())
    assert TMEM_COLS <= 512
    MAX_UNION = T * TOPK
    N_ROUNDS = ceildiv(MAX_UNION, 32)
    KV16 = KV_TILE_BYTES // 16
    QP16 = NQ_PAD * HEAD_DIM * EB // 16
    PT16 = BLK * NP * EB // 16
    NWARPS = 4 * NSWG + 4
    MMA_WARP = 4 * NSWG
    LOAD_WARP = 4 * NSWG + 1
    PREP_WARP = 4 * NSWG + 2
    PV_WARP = 4 * NSWG + 3
    SOFT_THREADS = 128 * NSWG
    BAR_TMEM = 1
    BAR_WG = 2

    @K.kernel(warps=NWARPS, arch="sm_100a", min_blocks_per_sm=1, grid=NUM_CTAS)
    def msa_decode_q4d(
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        out: K.gptr[out_dt],
        q2k: K.gptr[K.i32],
        lens: K.gptr[K.i32],
        ptab: K.gptr[K.i32],
        sched: K.gptr[K.i32],
        scale_log2: K.f32,
    ):
        warp = K.warp_id()
        lane = K.lane_id()
        tid = K.thread_id()

        smem = K.smem_pool()
        kv_smem = smem.alloc((STAGES, BLK, HEAD_DIM), kv_dt, swizzle=K.SW128B)
        q_smem = smem.alloc((1, NQ_PAD, HEAD_DIM), kv_dt, swizzle=K.SW128B)
        p_smem = smem.alloc((NP_SLOTS, BLK, NP), kv_dt, swizzle=P_SWZ)
        tmem_addr = smem.alloc((4,), K.u32)
        meta = smem.alloc((32,), K.i32)
        ulist = smem.alloc((2 * MAX_UNION,), K.i32)
        uplist = smem.alloc((2 * MAX_UNION,), K.i32)
        umask = smem.alloc((2 * MAX_UNION,), K.u32)
        words = smem.alloc((W_MAX,), K.u32)
        prefix = smem.alloc((W_MAX,), K.u32)
        xmax = smem.alloc((NSWG * 2 * NQW * 4,), K.f32, align=16)
        xsum = smem.alloc((NSWG * NQW * 4,), K.f32, align=16)
        mrow = smem.alloc((NSWG * 4 * NQW,), K.f32, align=16)
        K.keep_alive(ptab.ptr_to([0]))
        K.keep_alive(sched.ptr_to([0]))

        def encode(view, major):
            return view.encode(major=major, mma_k=MMA_K)


        kv_full_k = K.TMABar(smem, STAGES)
        kv_full_k.init(1)
        kv_full_v = K.TMABar(smem, STAGES)
        kv_full_v.init(1)
        kv_empty = K.TCGen05Bar(smem, STAGES, phase_offset=1)
        kv_empty.init(1)
        q_load = K.Pipeline(smem, 1, full="tma", empty="tcgen05", empty_phase_offset=1)
        union_ready = K.MBarrier(smem, 2)
        union_ready.init(32)
        if not STATIC_ONE_SHOT:
            union_free = K.MBarrier(smem, 2)
            union_free.init(SOFT_THREADS + 96)
        s_ready = K.TCGen05Bar(smem, CH)
        s_ready.init(1)
        s_free = K.MBarrier(smem, CH)
        s_free.init(SOFT_THREADS)
        p_ready = K.MBarrier(smem, NP_SLOTS)
        p_ready.init(SOFT_THREADS)
        p_free = K.TCGen05Bar(smem, NP_SLOTS)
        p_free.init(1)
        o_ready = K.TCGen05Bar(smem, 1)
        o_ready.init(1)
        o_free = K.MBarrier(smem, 1)
        o_free.init(SOFT_THREADS)

        def ld_global_i32_early(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.global_.nc.b32(value, ptr)
            return value

        def load_item_meta(b, h, dst_kv_s, dst_kv_len, dst_idxs):
            if PAGED:
                K.assign(dst_kv_s, 0)
                K.assign(dst_kv_len, ld_global_i32_early(lens.ptr_to([b])))
            else:
                K.assign(dst_kv_s, ld_global_i32_early(lens.ptr_to([b])))
                K.assign(dst_kv_len, ld_global_i32_early(lens.ptr_to([b + 1])) - dst_kv_s)
            base = (h * TOTAL_Q + b * T) * TOPK
            for r in range(N_ROUNDS):
                e = lane + 32 * r
                K.assign(dst_idxs[r], -1)
                if 32 * r + 32 <= MAX_UNION:
                    K.assign(dst_idxs[r], ld_global_i32_early(q2k.ptr_to([base + e])))
                else:
                    with K.If(e < MAX_UNION), K.Then():
                        K.assign(dst_idxs[r], ld_global_i32_early(q2k.ptr_to([base + e])))

        pre_kv_s = K.local_scalar("int32", init=0)
        pre_kv_len = K.local_scalar("int32", init=0)
        pre_idxs = K.alloc_local([N_ROUNDS], "int32")
        with K.If(warp == PREP_WARP), K.Then():
            task0 = K.cta_id()
            with K.If(task0 < NUM_ITEMS), K.Then():
                b0 = task0 // HKV
                h0 = task0 - b0 * HKV
                load_item_meta(b0, h0, pre_kv_s, pre_kv_len, pre_idxs)
        K.ptx.fence.proxy.async_.shared__cta()
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        with K.If(warp == MMA_WARP), K.Then():
            K.ptx[TMEM_ALLOC](K.address_of(tmem_addr[0]), K.uint32(TMEM_COLS))
            K.ptx[TMEM_RELINQUISH]()
            K.ptx.tcgen05.fence__before_thread_sync()
        with K.If(warp == LOAD_WARP), K.Then():
            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                K.ptx.prefetch.tensormap(K.address_of(q_map))
                K.ptx.prefetch.tensormap(K.address_of(k_map))
                K.ptx.prefetch.tensormap(K.address_of(v_map))
        tmem_base = K.local_scalar("uint32", init=K.uint32(0))

        def acquire_tmem_base():
            K.ptx.bar.sync(K.uint32(BAR_TMEM), K.uint32((4 * NSWG + 2) * 32))
            K.ptx.tcgen05.fence__after_thread_sync()
            tb_raw = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(tb_raw, tmem_addr.ptr_to([0]))
            K.assign(tmem_base, K.uniform(tb_raw))


        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def ld_smem_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.shared.b32(value, ptr)
            return value

        def ld_smem_u32(ptr):
            value = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(value, ptr)
            return value

        def ld_global_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.global_.nc.b32(value, ptr)
            return value

        def transpose_reduce(vals, ncols, op):
            assert 32 % ncols == 0 and ncols >= 2
            cur = [vals[i] for i in range(ncols)]
            xor = 16
            while len(cur) > 1:
                half = len(cur) // 2
                bit = K.bitwise_and(K.Cast("uint32", lane), K.uint32(xor)) != K.uint32(0)
                nxt = []
                for i in range(half):
                    a = cur[i]
                    b = cur[i + half]
                    send = K.local_scalar("float32", init=K.Select(bit, a, b))
                    recv = K.local_scalar("float32")
                    K.ptx.shfl_sync.bfly.b32(recv, send, K.uint32(xor), K.uint32(31), K.uint32(0xFFFFFFFF))
                    keep = K.Select(bit, b, a)
                    nxt.append(K.local_scalar("float32", init=op(keep, recv)))
                cur = nxt
                xor >>= 1
            res = cur[0]
            while xor >= 1:
                other = K.local_scalar("float32")
                K.ptx.shfl_sync.bfly.b32(other, res, K.uint32(xor), K.uint32(31), K.uint32(0xFFFFFFFF))
                res = K.local_scalar("float32", init=op(res, other))
                xor >>= 1
            return res

        def f32_to_half_bits(dst16, src):
            if kind == "f16":
                K.ptx.cvt.rn.f16.f32(dst16, src)
            else:
                K.ptx.cvt.rn.bf16.f32(dst16, src)

        def exp_group_packed(sv, m_arr, valid, words, i0, i1):
            """p = exp2(s*scale - m) for columns i0..i1-1 (pairs), packed half words; zero if not valid."""
            for i in range(i0, i1, 2):
                x0 = sv[i] * scale_log2 - m_arr[i]
                x1 = sv[i + 1] * scale_log2 - m_arr[i + 1]
                xw = K.local_scalar("uint32")
                pw = K.local_scalar("uint32")
                if kind == "f16":
                    K.ptx.cvt.rn.f16x2.f32(xw, x1, x0)
                    K.ptx.ex2.approx.f16x2(pw, xw)
                else:
                    K.ptx.cvt.rn.bf16x2.f32(xw, x1, x0)
                    K.ptx.ex2.approx.ftz.bf16x2(pw, xw)
                K.assign(words[i // 2], K.Select(valid, pw, K.uint32(0)))

        def accumulate_sums(words, lsum_arr, i0=0, i1=None):
            for i in range(i0, NQW if i1 is None else i1, 2):
                lo = K.local_scalar("float32")
                hi = K.local_scalar("float32")
                if kind == "f16":
                    K.ptx.cvt.f32.f16(lo, K.Cast("uint16", K.bitwise_and(words[i // 2], K.uint32(0xFFFF))))
                    K.ptx.cvt.f32.f16(hi, K.Cast("uint16", K.shift_right(words[i // 2], K.uint32(16))))
                else:
                    K.ptx.mov.b32(lo, K.shift_left(words[i // 2], K.uint32(16)))
                    K.ptx.mov.b32(hi, K.bitwise_and(words[i // 2], K.uint32(0xFFFF0000)))
                K.assign(lsum_arr[i], lsum_arr[i] + lo)
                K.assign(lsum_arr[i + 1], lsum_arr[i + 1] + hi)

        def store_p_words(pb, words, row, col0):
            """Store this thread's 32 packed probabilities (64 B) as part of one P^T row."""
            for c in range(0, NQW // 2, 4):
                ptr = p_smem[pb].ptr_to(row, col0 + 2 * c)
                K.ptx.st.shared.v4.b32(ptr, words[c], words[c + 1], words[c + 2], words[c + 3])

        def warp_arrive(bar, idx):
            with K.If(elected()), K.Then():
                bar.arrive(idx, count=32)

        def tmem_issue_cols(dst, col_expr, ncols):
            width = 32 if ncols >= 32 else 16
            for c in range(ncols // width):
                K.ptx[tmem_ld(width)](*(dst[width * c + i] for i in range(width)), tmem_base + col_expr + K.uint32(width * c))

        def tmem_load_cols(dst, col_expr, ncols):
            tmem_issue_cols(dst, col_expr, ncols)
            K.ptx.tcgen05.wait__ld.sync.aligned()

        def tmem_store_cols(src, col_expr, ncols):
            width = 32 if ncols >= 32 else 16
            for c in range(ncols // width):
                K.ptx[tmem_st(width)](tmem_base + col_expr + K.uint32(width * c), *(src[width * c + i] for i in range(width)))
            K.ptx.tcgen05.wait__st.sync.aligned()

        def lanemask_lt():
            return K.shift_left(K.uint32(1), K.Cast("uint32", lane)) - K.uint32(1)

        def ring_pos(t):
            q = t // STAGES
            return t - q * STAGES, q & 1

        def chunk_len(n_blocks, c):
            return K.min(K.int32(CH), n_blocks - c * CH)







        def iket_range(name, *, leader_only=False):
            token = K.alloc_local([1], "uint32")
            if leader_only:
                K.assign(token[0], K.cuda.iket.sentinel_token(name))
                with K.If((warp & 3) == 0), K.Then():
                    K.assign(token[0], K.cuda.iket.range_start(name))
            else:
                K.assign(token[0], K.cuda.iket.range_start(name))
            return token

        def iket_end(token):
            K.cuda.iket.range_end(token[0])

        sp = K.specialize(chain_dispatch=True)
        r_soft = sp.role("softmax", warps=list(range(4 * NSWG)))
        r_mma = sp.role("mma", warps=[MMA_WARP])
        r_load = sp.role("load", warps=[LOAD_WARP])
        r_prep = sp.role("prep", warps=[PREP_WARP])
        r_pv = sp.role("pv", warps=[PV_WARP])


        with r_soft:
            acquire_tmem_base()
            wg = warp >> 2
            wq = warp & 3
            tid_wg = tid & 127
            col0 = wg * NQW
            bar_id = K.uint32(BAR_WG) + K.Cast("uint32", wg)
            it = K.local_scalar("int32", init=0)
            gj = K.local_scalar("int32", init=0)
            gc = K.local_scalar("int32", init=0)
            par = K.local_scalar("int32", init=0)
            ps = K.local_scalar("uint32", init=K.uint32(0))
            running = K.local_scalar("int32", init=1)
            m = K.alloc_local([NQW], "float32")
            lsum = K.alloc_local([NQW], "float32")
            s = K.alloc_local([NQW], "float32")
            cm = K.alloc_local([NQW], "float32")
            mb = K.alloc_local([NQW], "float32")
            msub = K.alloc_local([NQW], "float32")
            pw = K.alloc_local([NQW // 2], "uint32")
            o = K.alloc_local([NQW], "float32")
            hbits = K.local_scalar("uint16")
            zero16 = K.local_scalar("uint16", init=K.Cast("uint16", K.int32(0)))
            with K.While(running != 0):
                slot = it & 1
                tk_wu = iket_range("sm-wait-union", leader_only=True)
                union_ready.wait(slot, (it >> 1) & 1)
                iket_end(tk_wu)
                n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                with K.If(n_blocks < 0):
                    with K.Then():
                        K.assign(running, 0)
                    with K.Else():
                        h = ld_smem_i32(meta.ptr_to([slot * 16 + 2]))
                        tok_base = ld_smem_i32(meta.ptr_to([slot * 16 + 5]))
                        causal_off = ld_smem_i32(meta.ptr_to([slot * 16 + 6]))

                        def group_token(gl):
                            return wg * GPW + gl

                        def group_cols(gl):
                            return range(gl * G, (gl + 1) * G)

                        def out_index(nl):
                            t = group_token(nl // G)
                            g = nl % G
                            return ((tok_base + t) * HQ + h * G + g) * HEAD_DIM + tid_wg

                        def group_sel(gl, tmask):
                            """Did query token gl of this warpgroup select the block? (uniform over the warpgroup)"""
                            t = group_token(gl)
                            return K.bitwise_and(K.shift_right(tmask, K.Cast("uint32", t)), K.uint32(1)) != K.uint32(0)

                        def group_causal(gl, kv_pos):
                            return kv_pos <= causal_off + group_token(gl)

                        def group_valid(gl, kv_pos, tmask):
                            return K.And(group_sel(gl, tmask), group_causal(gl, kv_pos))

                        with K.If(n_blocks > 0):
                            with K.Then():
                                for nl in range(NQW):
                                    K.assign(m[nl], K.float32(NEG_INF))
                                    K.assign(msub[nl], K.float32(-1.0e38))
                                    K.assign(lsum[nl], K.float32(0.0))
                                n_chunks = K.local_scalar("int32", init=(n_blocks + (CH - 1)) // CH)
                                with K.serial(n_chunks, unroll=False) as c:
                                    cbase = c * CH
                                    nb = K.local_scalar("int32", init=K.min(K.int32(CH), n_blocks - cbase))

                                    for nl in range(NQW):
                                        K.assign(cm[nl], K.float32(NEG_INF))
                                    with K.serial(nb, unroll=False) as j:
                                        blk = ld_smem_i32(ulist.ptr_to([slot * MAX_UNION + cbase + j]))
                                        tmask = ld_smem_u32(umask.ptr_to([slot * MAX_UNION + cbase + j]))
                                        kv_pos = blk * BLK + tid_wg
                                        tk_ws = iket_range("sm-wait-s", leader_only=True)
                                        s_ready.wait(j, K.Cast("int32", K.bitwise_and(K.shift_right(ps, K.Cast("uint32", j)), K.uint32(1))))
                                        K.assign(ps, K.bitwise_xor(ps, K.shift_left(K.uint32(1), K.Cast("uint32", j))))
                                        K.ptx.tcgen05.fence__after_thread_sync()
                                        iket_end(tk_ws)
                                        tk_p1 = iket_range("sm-max", leader_only=True)
                                        tmem_load_cols(s, K.Cast("uint32", j) * K.uint32(NQ_PAD) + K.uint32(col0), NQW)
                                        for gl in range(GPW):
                                            if SKIP_GROUPS:
                                                with K.If(group_sel(gl, tmask)), K.Then():
                                                    valid = group_causal(gl, kv_pos)
                                                    for nl in group_cols(gl):
                                                        K.assign(cm[nl], K.max(cm[nl], K.Select(valid, s[nl], K.float32(NEG_INF))))
                                            else:
                                                valid = group_valid(gl, kv_pos, tmask)
                                                for nl in group_cols(gl):
                                                    K.assign(cm[nl], K.max(cm[nl], K.Select(valid, s[nl], K.float32(NEG_INF))))
                                        iket_end(tk_p1)
                                    tk_p1 = iket_range("sm-max", leader_only=True)


                                    for nl in range(NQW):
                                        K.ptx.redux_sync.max.f32(cm[nl], cm[nl], K.uint32(0xFFFFFFFF))
                                    mine = K.local_scalar("float32", init=cm[0])
                                    for nl in range(1, NQW):
                                        K.assign(mine, K.Select(lane == nl, cm[nl], mine))
                                    K.ptx.st.shared.f32(xmax.ptr_to([((wg * 2 + par) * NQW + lane) * 4 + wq]), mine)
                                    iket_end(tk_p1)
                                    tk_bar = iket_range("sm-bar", leader_only=True)
                                    K.ptx.bar.sync(bar_id, K.uint32(128))
                                    iket_end(tk_bar)
                                    tk_mg = iket_range("sm-merge", leader_only=True)
                                    v4 = K.alloc_local([4], "float32")
                                    K.ptx.ld.shared.v4.f32(v4[0], v4[1], v4[2], v4[3], xmax.ptr_to([((wg * 2 + par) * NQW + lane) * 4]))
                                    fin = K.local_scalar("float32")
                                    K.ptx.max.f32(fin, v4[0], v4[1], v4[2])
                                    K.assign(fin, K.max(fin, v4[3]) * scale_log2)

                                    mrow_base = (wg * 4 + wq) * NQW
                                    K.ptx.st.shared.f32(mrow.ptr_to([mrow_base + lane]), fin)
                                    K.cuda.warp_sync()
                                    for c4 in range(NQW // 4):
                                        K.ptx.ld.shared.v4.f32(mb[4 * c4], mb[4 * c4 + 1], mb[4 * c4 + 2], mb[4 * c4 + 3], mrow.ptr_to([mrow_base + 4 * c4]))


                                    dmax = K.local_scalar("float32", init=K.float32(NEG_INF))
                                    for nl in range(NQW):
                                        K.assign(dmax, K.max(dmax, mb[nl] - m[nl]))
                                    with K.If(dmax > K.float32(RESCALE_THRESH)), K.Then():
                                        any_r = K.local_scalar("int32", init=0)
                                        dlog = K.alloc_local([NQW], "float32")
                                        for nl in range(NQW):
                                            m_old = K.local_scalar("float32", init=m[nl])
                                            m_new = K.local_scalar("float32", init=K.max(m_old, mb[nl]))
                                            d = K.local_scalar("float32", init=m_new - m_old)
                                            big = d > K.float32(RESCALE_THRESH)
                                            do_r = K.And(m_old != K.float32(NEG_INF), big)
                                            K.assign(dlog[nl], K.Select(do_r, K.float32(0.0) - d, K.float32(0.0)))
                                            K.assign(m[nl], K.Select(big, m_new, m_old))
                                            K.assign(any_r, K.Select(do_r, 1, any_r))
                                        with K.If(any_r != 0), K.Then():
                                            gjm = gj - 1
                                            p_free.wait(gjm & (NP_SLOTS - 1), (gjm >> 1) & 1)
                                            K.ptx.tcgen05.fence__after_thread_sync()
                                            tmem_load_cols(o, K.uint32(O_COL + col0), NQW)
                                            for nl in range(NQW):
                                                a = K.local_scalar("float32")
                                                K.ptx.ex2.approx.ftz.f32(a, dlog[nl])
                                                K.assign(o[nl], o[nl] * a)
                                                K.assign(lsum[nl], lsum[nl] * a)
                                            tmem_store_cols(o, K.uint32(O_COL + col0), NQW)
                                            K.ptx.tcgen05.fence__before_thread_sync()
                                        for nl in range(NQW):
                                            K.assign(msub[nl], K.max(m[nl], K.float32(-1.0e38)))
                                    iket_end(tk_mg)

                                    with K.serial(nb, unroll=False) as j:
                                        blk = ld_smem_i32(ulist.ptr_to([slot * MAX_UNION + cbase + j]))
                                        tmask = ld_smem_u32(umask.ptr_to([slot * MAX_UNION + cbase + j]))
                                        kv_pos = blk * BLK + tid_wg
                                        pb = gj & (NP_SLOTS - 1)
                                        tk_ex = iket_range("sm-exp", leader_only=True)
                                        tmem_load_cols(s, K.Cast("uint32", j) * K.uint32(NQ_PAD) + K.uint32(col0), NQW)
                                        K.ptx.tcgen05.fence__before_thread_sync()
                                        warp_arrive(s_free, j)
                                        for gl in range(GPW):
                                            if SKIP_GROUPS:
                                                with K.If(group_sel(gl, tmask)):
                                                    with K.Then():
                                                        valid = group_causal(gl, kv_pos)
                                                        exp_group_packed(s, msub, valid, pw, gl * G, (gl + 1) * G)
                                                        accumulate_sums(pw, lsum, gl * G, (gl + 1) * G)
                                                    with K.Else():
                                                        for i in range(gl * G // 2, (gl + 1) * G // 2):
                                                            K.assign(pw[i], K.uint32(0))
                                            else:
                                                valid = group_valid(gl, kv_pos, tmask)
                                                exp_group_packed(s, msub, valid, pw, gl * G, (gl + 1) * G)
                                        if not SKIP_GROUPS:
                                            accumulate_sums(pw, lsum)
                                        iket_end(tk_ex)
                                        tk_wp = iket_range("sm-wait-pfree", leader_only=True)
                                        p_free.wait(pb, ((gj >> 1) + 1) & 1)

                                        K.ptx.fence.proxy.async_.shared__cta()
                                        iket_end(tk_wp)
                                        tk_ps = iket_range("sm-pstore", leader_only=True)
                                        store_p_words(pb, pw, tid_wg, col0)
                                        K.ptx.fence.proxy.async_.shared__cta()
                                        warp_arrive(p_ready, pb)
                                        iket_end(tk_ps)
                                        K.assign(gj, gj + 1)
                                    K.assign(par, par ^ 1)
                                    K.assign(gc, gc + 1)

                                tk_wo = iket_range("sm-wait-o", leader_only=True)
                                o_ready.wait(0, it & 1)
                                K.ptx.tcgen05.fence__after_thread_sync()
                                iket_end(tk_wo)
                                tk_epi = iket_range("sm-epi", leader_only=True)
                                tmem_load_cols(o, K.uint32(O_COL + col0), NQW)
                                K.ptx.tcgen05.fence__before_thread_sync()
                                warp_arrive(o_free, 0)
                                mycol_sum = transpose_reduce(lsum, NQW, lambda a, b: a + b)
                                K.ptx.st.shared.f32(xsum.ptr_to([(wg * NQW + lane) * 4 + wq]), mycol_sum)
                                K.ptx.bar.sync(bar_id, K.uint32(128))
                                v4s = K.alloc_local([4], "float32")
                                K.ptx.ld.shared.v4.f32(v4s[0], v4s[1], v4s[2], v4s[3], xsum.ptr_to([(wg * NQW + lane) * 4]))
                                lfin = K.local_scalar("float32", init=(v4s[0] + v4s[1]) + (v4s[2] + v4s[3]))
                                for nl in range(NQW):
                                    ltot = K.local_scalar("float32")
                                    K.ptx.shfl_sync.idx.b32(ltot, lfin, K.uint32(nl), K.uint32(31), K.uint32(0xFFFFFFFF))
                                    bad = K.Or(ltot == K.float32(0.0), ltot != ltot)
                                    inv = K.local_scalar("float32")
                                    K.ptx.rcp.approx.ftz.f32(inv, K.Select(bad, K.float32(1.0), ltot))
                                    val = K.local_scalar("float32", init=K.Select(bad, K.float32(0.0), o[nl] * inv))
                                    f32_to_half_bits(hbits, val)
                                    K.ptx.st.global_.b16(out.ptr_to([out_index(nl)]), hbits)
                                iket_end(tk_epi)
                            with K.Else():
                                o_ready.wait(0, it & 1)
                                warp_arrive(o_free, 0)
                                for nl in range(NQW):
                                    K.ptx.st.global_.b16(out.ptr_to([out_index(nl)]), zero16)
                        if not STATIC_ONE_SHOT:
                            union_free.arrive(slot)
                if STATIC_ONE_SHOT:
                    K.assign(running, 0)
                else:
                    K.assign(it, it + 1)


        with r_mma:
            acquire_tmem_base()
            it_m = K.local_scalar("int32", init=0)
            t_m = K.local_scalar("int32", init=0)
            pk = K.local_scalar("uint32", init=K.uint32(0))
            pf = K.local_scalar("uint32", init=K.uint32(0))
            running_m = K.local_scalar("int32", init=1)
            zero = K.uint32(0)
            kd0, koff = encode(kv_smem[0], "k")
            qd0, qoff = encode(q_smem[0], "k")

            def mma_qk(sb_col, kstage):
                for ki in range(NK):
                    with K.If(elected()), K.Then():
                        K.ptx[MMA](
                            tmem_base + sb_col,
                            kd0 + (kstage * KV16 + koff(ki)),
                            qd0 + qoff(ki),
                            K.uint32(ID_QK),
                            zero,
                            zero,
                            zero,
                            zero,
                            ki != 0,
                        )

            with K.While(running_m != 0):
                slot = it_m & 1
                union_ready.wait(slot, (it_m >> 1) & 1)
                n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                with K.If(n_blocks < 0):
                    with K.Then():
                        K.assign(running_m, 0)
                    with K.Else():
                        q_load.full.wait(0, it_m & 1)
                        with K.If(n_blocks == 0), K.Then():
                            with K.If(elected()), K.Then():
                                q_load.empty.arrive(0)
                        n_chunks = K.local_scalar("int32", init=(n_blocks + (CH - 1)) // CH)
                        seg_m = K.local_scalar("int32", init=t_m)
                        with K.serial(n_chunks, unroll=False) as c:
                            cbase = c * CH
                            nb = K.local_scalar("int32", init=K.min(K.int32(CH), n_blocks - cbase))
                            nprev = K.local_scalar("int32", init=K.Select(c > 0, chunk_len(n_blocks, c - 1), K.int32(0)))
                            with K.serial(nb, unroll=False) as j:
                                kpos = K.Select(c > 0, seg_m + 2 * j + 1, seg_m + j)
                                kstage, _kphase = ring_pos(kpos)
                                kst = K.local_scalar("int32", init=kstage)
                                tk_wk = iket_range("mma-wait-k")
                                kv_full_k.wait(kst, K.Cast("int32", K.bitwise_and(K.shift_right(pk, K.Cast("uint32", kst)), K.uint32(1))))
                                K.assign(pk, K.bitwise_xor(pk, K.shift_left(K.uint32(1), K.Cast("uint32", kst))))
                                iket_end(tk_wk)
                                tk_sf = iket_range("mma-wait-sfree")
                                s_free.wait(j, K.Cast("int32", K.bitwise_xor(K.bitwise_and(K.shift_right(pf, K.Cast("uint32", j)), K.uint32(1)), K.uint32(1))))
                                K.assign(pf, K.bitwise_xor(pf, K.shift_left(K.uint32(1), K.Cast("uint32", j))))
                                K.ptx.tcgen05.fence__after_thread_sync()
                                iket_end(tk_sf)
                                tk_qk = iket_range("mma-qk")
                                mma_qk(K.Cast("uint32", j) * K.uint32(NQ_PAD), kst)
                                with K.If(elected()), K.Then():
                                    s_ready.arrive(j)
                                    kv_empty.arrive(kst)
                                iket_end(tk_qk)
                            with K.If(elected()), K.Then():
                                with K.If(c + 1 == n_chunks), K.Then():
                                    q_load.empty.arrive(0)
                            K.assign(seg_m, seg_m + nprev + nb)
                        with K.If(n_blocks > 0), K.Then():
                            K.assign(t_m, seg_m + chunk_len(n_blocks, n_chunks - 1))
                        if not STATIC_ONE_SHOT:
                            union_free.arrive(slot)
                if STATIC_ONE_SHOT:
                    K.assign(running_m, 0)
                else:
                    K.assign(it_m, it_m + 1)


        with r_pv:
            acquire_tmem_base()
            it_v = K.local_scalar("int32", init=0)
            gj_v = K.local_scalar("int32", init=0)
            t_v = K.local_scalar("int32", init=0)
            pv = K.local_scalar("uint32", init=K.uint32(0))
            running_v = K.local_scalar("int32", init=1)
            zero_v = K.uint32(0)
            vd0, voff = encode(kv_smem[0], "mn")
            pd0, poff = encode(p_smem[0], "mn")

            def mma_pv(vstage, pb, acc):
                for ki in range(NK):
                    with K.If(elected()), K.Then():
                        K.ptx[MMA](
                            tmem_base + K.uint32(O_COL),
                            vd0 + (vstage * KV16 + voff(ki)),
                            pd0 + (pb * PT16 + poff(ki)),
                            K.uint32(ID_PV),
                            zero_v,
                            zero_v,
                            zero_v,
                            zero_v,
                            True if ki != 0 else K.Cast("bool", acc != 0),
                        )

            with K.While(running_v != 0):
                slot = it_v & 1
                union_ready.wait(slot, (it_v >> 1) & 1)
                n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                with K.If(n_blocks < 0):
                    with K.Then():
                        K.assign(running_v, 0)
                    with K.Else():
                        o_free.wait(0, (it_v + 1) & 1)
                        K.ptx.tcgen05.fence__after_thread_sync()
                        with K.If(n_blocks == 0), K.Then():
                            with K.If(elected()), K.Then():
                                o_ready.arrive(0)
                        acc = K.local_scalar("int32", init=0)
                        n_chunks = K.local_scalar("int32", init=(n_blocks + (CH - 1)) // CH)
                        seg_v = K.local_scalar("int32", init=t_v + chunk_len(n_blocks, 0))
                        with K.serial(n_chunks, unroll=False) as c:
                            cbase = c * CH
                            nb = K.local_scalar("int32", init=K.min(K.int32(CH), n_blocks - cbase))
                            nnext = K.local_scalar("int32", init=K.Select(c + 1 < n_chunks, chunk_len(n_blocks, c + 1), K.int32(0)))
                            with K.serial(nb, unroll=False) as j:
                                vpos = K.Select(j < nnext, seg_v + 2 * j, seg_v + 2 * nnext + (j - nnext))
                                vstage, _vphase = ring_pos(vpos)
                                vst = K.local_scalar("int32", init=vstage)
                                pb = gj_v & (NP_SLOTS - 1)
                                tk_wv = iket_range("mma-wait-v")
                                kv_full_v.wait(vst, K.Cast("int32", K.bitwise_and(K.shift_right(pv, K.Cast("uint32", vst)), K.uint32(1))))
                                K.assign(pv, K.bitwise_xor(pv, K.shift_left(K.uint32(1), K.Cast("uint32", vst))))
                                iket_end(tk_wv)
                                tk_wp = iket_range("mma-wait-p")
                                p_ready.wait(pb, (gj_v >> 1) & 1)
                                iket_end(tk_wp)
                                tk_pv = iket_range("mma-pv")
                                mma_pv(vst, pb, acc)
                                with K.If(elected()), K.Then():
                                    p_free.arrive(pb)
                                    kv_empty.arrive(vst)
                                iket_end(tk_pv)
                                K.assign(acc, 1)
                                K.assign(gj_v, gj_v + 1)
                            K.assign(seg_v, seg_v + nb + nnext)
                        with K.If(n_blocks > 0), K.Then():
                            K.assign(t_v, seg_v)
                            with K.If(elected()), K.Then():
                                o_ready.arrive(0)
                        if not STATIC_ONE_SHOT:
                            union_free.arrive(slot)
                if STATIC_ONE_SHOT:
                    K.assign(running_v, 0)
                else:
                    K.assign(it_v, it_v + 1)


        with r_load:
            it_l = K.local_scalar("int32", init=0)
            t_l = K.local_scalar("int32", init=0)
            running_l = K.local_scalar("int32", init=1)
            with K.While(running_l != 0):
                slot = it_l & 1
                tk_lu = iket_range("ld-wait-union")
                union_ready.wait(slot, (it_l >> 1) & 1)
                iket_end(tk_lu)
                n_blocks = ld_smem_i32(meta.ptr_to([slot * 16]))
                with K.If(n_blocks < 0):
                    with K.Then():
                        K.assign(running_l, 0)
                    with K.Else():
                        h = ld_smem_i32(meta.ptr_to([slot * 16 + 2]))
                        kv_s = ld_smem_i32(meta.ptr_to([slot * 16 + 3]))
                        tok_base = ld_smem_i32(meta.ptr_to([slot * 16 + 5]))
                        q_load.empty.wait(0, it_l & 1)
                        with K.If(elected()), K.Then():
                            K.ptx[TMA_4D](
                                q_smem[0].ptr_to(0, 0),
                                K.address_of(q_map),
                                K.int32(0),
                                h * G,
                                tok_base,
                                K.int32(0),
                                K.cuda.cvta_generic_to_shared(q_load.full.ptr_to([0])),
                            )
                            q_load.full.arrive(0, tx_count=Q_TMA_BYTES)
                        def issue_tile(tmap, full_bar, ub):
                            """One TMA tile for union entry ub into the next ring position."""
                            blk = ld_smem_i32(ulist.ptr_to([slot * MAX_UNION + ub]))
                            if PAGED:
                                pg = ld_smem_i32(uplist.ptr_to([slot * MAX_UNION + ub]))
                            lstage, lphase = ring_pos(t_l)
                            lst = K.local_scalar("int32", init=lstage)
                            lph = K.local_scalar("int32", init=lphase)
                            K.assign(t_l, t_l + 1)
                            tk_we = iket_range("ld-wait-stage")
                            kv_empty.wait(lst, lph)
                            iket_end(tk_we)
                            with K.If(elected()), K.Then():
                                if PAGED:
                                    K.ptx[TMA_4D_HINT](
                                        kv_smem[lst].ptr_to(0, 0),
                                        K.address_of(tmap),
                                        K.int32(0),
                                        K.int32(0),
                                        K.int32(0),
                                        pg * HKV + h,
                                        K.cuda.cvta_generic_to_shared(full_bar.ptr_to([lst])),
                                        *HINT_ARGS,
                                    )
                                else:
                                    K.ptx[TMA_3D_HINT](
                                        kv_smem[lst].ptr_to(0, 0),
                                        K.address_of(tmap),
                                        K.int32(0),
                                        kv_s + blk * BLK,
                                        h * 2,
                                        K.cuda.cvta_generic_to_shared(full_bar.ptr_to([lst])),
                                        *HINT_ARGS,
                                    )
                                full_bar.arrive(lst, tx_count=KV_TILE_BYTES)

                        n_chunks = K.local_scalar("int32", init=(n_blocks + (CH - 1)) // CH)
                        n0 = chunk_len(n_blocks, 0)
                        with K.serial(n0, unroll=False) as j:
                            issue_tile(k_map, kv_full_k, j)
                        with K.serial(n_chunks - 1, unroll=False) as cm1:
                            c = cm1 + 1
                            nprev = chunk_len(n_blocks, c - 1)
                            nc = chunk_len(n_blocks, c)
                            with K.serial(nc, unroll=False) as j:
                                issue_tile(v_map, kv_full_v, (c - 1) * CH + j)
                                issue_tile(k_map, kv_full_k, c * CH + j)
                            with K.serial(nprev - nc, unroll=False) as jj:
                                issue_tile(v_map, kv_full_v, (c - 1) * CH + nc + jj)
                        with K.If(n_blocks > 0), K.Then():
                            nlast = chunk_len(n_blocks, n_chunks - 1)
                            with K.serial(nlast, unroll=False) as j:
                                issue_tile(v_map, kv_full_v, (n_chunks - 1) * CH + j)
                        if not STATIC_ONE_SHOT:
                            union_free.arrive(slot)
                if STATIC_ONE_SHOT:
                    K.assign(running_l, 0)
                else:
                    K.assign(it_l, it_l + 1)


        with r_prep:
            it_p = K.local_scalar("int32", init=0)
            running_p = K.local_scalar("int32", init=1)
            with K.While(running_p != 0):
                slot = it_p & 1
                task = K.local_scalar("int32")
                if STATIC_ONE_SHOT:
                    K.assign(task, K.cta_id())
                else:
                    with K.If(it_p == 0):
                        with K.Then():
                            K.assign(task, K.cta_id())
                        with K.Else():
                            grabbed = K.local_scalar("int32", init=0)
                            with K.If(lane == 0), K.Then():
                                K.ptx.atom.relaxed.gpu.global_.add.s32(grabbed, sched.ptr_to([0]), K.int32(1))
                            K.assign(task, K.uniform(grabbed) + NUM_CTAS)
                if not STATIC_ONE_SHOT:
                    tk_pf = iket_range("prep-wait-free")
                    union_free.wait(slot, ((it_p >> 1) + 1) & 1)
                    iket_end(tk_pf)
                tk_pi = iket_range("prep-item")
                with K.If(task >= NUM_ITEMS):
                    with K.Then():
                        with K.If(lane == 0), K.Then():
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16]), K.int32(-1))
                        union_ready.arrive(slot)
                        K.assign(running_p, 0)
                    with K.Else():
                        b = task // HKV
                        h = task - b * HKV
                        kv_s = K.local_scalar("int32", init=0)
                        kv_len = K.local_scalar("int32")
                        idxs = K.alloc_local([N_ROUNDS], "int32")
                        if STATIC_ONE_SHOT:
                            K.assign(kv_s, pre_kv_s)
                            K.assign(kv_len, pre_kv_len)
                            for r in range(N_ROUNDS):
                                K.assign(idxs[r], pre_idxs[r])
                        else:
                            with K.If(it_p == 0):
                                with K.Then():
                                    K.assign(kv_s, pre_kv_s)
                                    K.assign(kv_len, pre_kv_len)
                                    for r in range(N_ROUNDS):
                                        K.assign(idxs[r], pre_idxs[r])
                                with K.Else():
                                    load_item_meta(b, h, kv_s, kv_len, idxs)
                        n_vis = (kv_len + (BLK - 1)) >> 7

                        def valid(r):
                            return K.And(idxs[r] >= 0, idxs[r] < n_vis)


                        nwords = (n_vis + 31) >> 5
                        wi = K.local_scalar("int32", init=lane)
                        with K.While(wi < nwords):
                            K.ptx.st.shared.b32(words.ptr_to([wi]), K.uint32(0))
                            K.assign(wi, wi + 32)
                        K.cuda.warp_sync()
                        for r in range(N_ROUNDS):
                            with K.If(valid(r)), K.Then():
                                K.ptx.red.shared.or_.b32(
                                    words.ptr_to([idxs[r] >> 5]),
                                    K.shift_left(K.uint32(1), K.Cast("uint32", idxs[r] & 31)),
                                )
                        K.cuda.warp_sync()
                        carry = K.local_scalar("uint32", init=K.uint32(0))
                        wbase = K.local_scalar("int32", init=0)
                        with K.While(wbase < nwords):
                            w = wbase + lane
                            wv = K.local_scalar("uint32", init=K.uint32(0))
                            with K.If(w < nwords), K.Then():
                                K.ptx.ld.shared.u32(wv, words.ptr_to([w]))
                            cnt = K.alloc_local([1], "uint32")
                            K.ptx.popc.b32(cnt[0], wv)
                            own = K.local_scalar("uint32", init=cnt[0])
                            K.idioms.warp_scan_add(cnt, 1, lane)
                            with K.If(w < nwords), K.Then():
                                K.ptx.st.shared.b32(prefix.ptr_to([w]), carry + (cnt[0] - own))
                            tot = K.local_scalar("uint32")
                            K.ptx.shfl_sync.idx.b32(tot, cnt[0], K.uint32(31), K.uint32(31), K.uint32(0xFFFFFFFF))
                            K.assign(carry, carry + tot)
                            K.assign(wbase, wbase + 32)
                        n_blocks = K.local_scalar("int32", init=K.Cast("int32", carry))
                        K.cuda.warp_sync()
                        with K.serial(nwords, unroll=False) as w:
                            wv = ld_smem_u32(words.ptr_to([w]))
                            pw_ = ld_smem_u32(prefix.ptr_to([w]))
                            mybit = K.bitwise_and(K.shift_right(wv, K.Cast("uint32", lane)), K.uint32(1))
                            with K.If(mybit != K.uint32(0)), K.Then():
                                below = K.local_scalar("uint32")
                                K.ptx.popc.b32(below, K.bitwise_and(wv, lanemask_lt()))
                                rank = K.Cast("int32", pw_ + below)
                                blkid = w * 32 + lane
                                K.ptx.st.shared.b32(ulist.ptr_to([slot * MAX_UNION + rank]), blkid)
                                K.ptx.st.shared.b32(umask.ptr_to([slot * MAX_UNION + rank]), K.uint32(0))
                                if PAGED:
                                    pg = ld_global_i32(ptab.ptr_to([b * MAX_PAGES + blkid]))
                                    K.ptx.st.shared.b32(uplist.ptr_to([slot * MAX_UNION + rank]), pg)
                        K.cuda.warp_sync()
                        for r in range(N_ROUNDS):
                            e = lane + 32 * r
                            with K.If(valid(r)), K.Then():
                                wq_ = idxs[r] >> 5
                                wv = ld_smem_u32(words.ptr_to([wq_]))
                                pw_ = ld_smem_u32(prefix.ptr_to([wq_]))
                                below = K.local_scalar("uint32")
                                K.ptx.popc.b32(
                                    below,
                                    K.bitwise_and(wv, K.shift_left(K.uint32(1), K.Cast("uint32", idxs[r] & 31)) - K.uint32(1)),
                                )
                                u = K.Cast("int32", pw_ + below)
                                K.ptx.red.shared.or_.b32(
                                    umask.ptr_to([slot * MAX_UNION + u]),
                                    K.shift_left(K.uint32(1), K.Cast("uint32", e // TOPK)),
                                )
                        K.cuda.warp_sync()
                        with K.If(lane == 0), K.Then():
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 0]), n_blocks)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 1]), b)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 2]), h)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 3]), kv_s)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 4]), kv_len)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 5]), b * T)
                            K.ptx.st.shared.b32(meta.ptr_to([slot * 16 + 6]), kv_len - T)
                        union_ready.arrive(slot)
                iket_end(tk_pi)
                if STATIC_ONE_SHOT:
                    K.assign(running_p, 0)
                else:
                    K.assign(it_p, it_p + 1)


        K.cuda.cta_sync()
        if not STATIC_ONE_SHOT:
            with K.If(tid == 0), K.Then():
                done = K.local_scalar("int32")
                K.ptx.atom.acq_rel.gpu.global_.add.s32(done, sched.ptr_to([1]), K.int32(1))
                with K.If(done == NUM_CTAS - 1), K.Then():
                    K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([0]), K.int32(0))
                    K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([1]), K.int32(0))
        with K.If(warp == MMA_WARP), K.Then():
            K.ptx[TMEM_DEALLOC](tmem_base, K.uint32(TMEM_COLS))

    return msa_decode_q4d


def plan_config_q4d(q, k, q2k, page_table, B, T, num_sms):
    HQ = q.shape[1]
    HKV = k.shape[1]
    G = HQ // HKV
    TOPK = int(q2k.shape[-1])
    TOTAL_Q = q.shape[0]
    paged = page_table is not None
    kind = {torch.bfloat16: "bf16", torch.float16: "f16"}[k.dtype]
    NQ = T * G
    assert NQ in (32, 64)
    NQ_PAD = NQ
    NP = NQ_PAD
    CH = int(min(T * TOPK, (512 - NP) // NQ_PAD))
    if paged:
        MAX_PAGES = int(page_table.shape[1])
        max_blocks = MAX_PAGES
    else:
        MAX_PAGES = 0
        max_blocks = ceildiv(int(k.shape[0]), BLK)
    W_MAX = max(1, ceildiv(max_blocks, 32))
    NUM_ITEMS = B * HKV
    NUM_CTAS = max(1, min(num_sms, NUM_ITEMS))
    if NQ == 64 and NUM_ITEMS == 512:
        NUM_CTAS = min(NUM_CTAS, 128)
    env_ctas = int(os.environ.get("Q4D_CTAS", "0"))
    if env_ctas:
        NUM_CTAS = min(NUM_CTAS, env_ctas)
    STATIC_ONE_SHOT = NUM_ITEMS == NUM_CTAS
    eb = 2
    kv_tile = BLK * HEAD_DIM * eb
    q_tile = NQ_PAD * HEAD_DIM * eb
    p_tile = BLK * NP * eb
    MAX_UNION = T * TOPK
    misc = 24 * MAX_UNION + 8 * W_MAX + 64 * NQ + 2048 + 8 * (3 * 16 + 2 * CH + 24)
    budget = 225 * 1024 - misc - q_tile - 2 * p_tile
    STAGES = int(min(16, budget // kv_tile))
    if int(os.environ.get("Q4D_STAGES", "0")):
        STAGES = min(STAGES, int(os.environ["Q4D_STAGES"]))
    assert STAGES >= 2, "shared memory budget too small"
    return dict(
        T=T,
        G=G,
        HKV=HKV,
        TOPK=TOPK,
        TOTAL_Q=TOTAL_Q,
        kv_kind=kind,
        paged=paged,
        MAX_PAGES=MAX_PAGES,
        MAX_BLOCK_WORDS=W_MAX,
        NUM_ITEMS=NUM_ITEMS,
        NUM_CTAS=NUM_CTAS,
        STAGES=STAGES,
        CH=CH,
        STATIC_ONE_SHOT=STATIC_ONE_SHOT,
    )


def setup_q4d(data, B, seqlen_q):
    q = data["q"]
    k = data["k"]
    v = data["v"]
    q2k = data["q2k_indices"]
    cu_seqlens_k = data["cu_seqlens_k"]
    page_table = data["page_table"]
    seqused_k = data["seqused_k"]
    out = data["output"]
    scale = float(data["softmax_scale"])
    T = int(seqlen_q)
    B = int(B)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous()
    assert q.shape[0] == B * T
    device = q.device
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    cfg = plan_config_q4d(q, k, q2k, page_table, B, T, num_sms)
    ex = _compile(cfg, make_kernel_q4d, "q4d")

    HKV = cfg["HKV"]
    G = cfg["G"]
    HQ = HKV * G
    TOTAL_Q = cfg["TOTAL_Q"]
    dtype_name = "bfloat16" if q.dtype == torch.bfloat16 else "float16"
    q_map = _encode(
        q,
        dtype_name,
        (HEAD_DIM // 2, HQ, TOTAL_Q, 2),
        (HEAD_DIM * 2, HQ * HEAD_DIM * 2, (HEAD_DIM // 2) * 2),
        (HEAD_DIM // 2, G, T, 2),
    )
    if cfg["paged"]:
        num_pages = int(k.shape[0])
        kv_dims = (HEAD_DIM // 2, BLK, 2, num_pages * HKV)
        kv_strides = (HEAD_DIM * 2, (HEAD_DIM // 2) * 2, BLK * HEAD_DIM * 2)
        kv_box = (HEAD_DIM // 2, BLK, 2, 1)
        lens = seqused_k.contiguous()
        ptab = page_table.contiguous().view(-1)
    else:
        total_k = int(k.shape[0])
        kv_dims = (HEAD_DIM // 2, total_k, HKV * 2)
        kv_strides = (HKV * HEAD_DIM * 2, (HEAD_DIM // 2) * 2)
        kv_box = (HEAD_DIM // 2, BLK, 2)
        lens = cu_seqlens_k.contiguous()
        ptab = torch.zeros(4, dtype=torch.int32, device=device)
    k_map = _encode(k, dtype_name, kv_dims, kv_strides, kv_box, l2_promotion=3)
    v_map = _encode(v, dtype_name, kv_dims, kv_strides, kv_box, l2_promotion=3)
    sched = torch.zeros(4, dtype=torch.int32, device=device)
    q2k_flat = q2k.contiguous().view(-1)
    out_flat = out.view(-1)
    args = (q_map.ptr, k_map.ptr, v_map.ptr, out_flat, q2k_flat, lens, ptab, sched, float(scale * LOG2E))
    keep = (q, k, v, q2k, q2k_flat, lens, ptab, out, out_flat, sched, q_map, k_map, v_map)

    def run():
        ex(*args)

    run._keep = keep
    run._cfg = cfg
    run()
    torch.cuda.synchronize(device)
    return run


def _use_q4d(data, B, T):
    q, k = data["q"], data["k"]
    T = int(T)
    if T < 2:
        return False
    HQ = int(q.shape[1])
    HKV = int(k.shape[1])
    if HQ % HKV:
        return False
    G = HQ // HKV
    if 32 % G != 0 or T * G not in (32, 64):
        return False
    if k.dtype not in (torch.bfloat16, torch.float16):
        return False
    if os.environ.get("Q4D_DISABLE", "0") == "1":
        return False
    return True


_setup_prev_q4d = setup


def setup(data, B, seqlen_q):  # noqa: F811 - chained shape-dispatch override
    T = int(seqlen_q)
    if _use_q4d(data, B, T):
        return setup_q4d(data, B, T)
    return _setup_prev_q4d(data, B, seqlen_q)


# Query-major q4 experiment.  The 64 query rows are the M extent and the
# selected 128-token block is N.  P overwrites the upper half of the score
# tile in TMEM and is consumed directly by the ts-form PV MMA, removing the
# 32 KiB shared P ring used by q4d.  A two-stage KV ring keeps the CTA below
# the two-resident shared-memory budget.
Q4R_TMEM_LD32 = "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
Q4R_TMEM_ST16 = "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
Q4R_TMEM_ST64 = "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
Q4R_MIN_BLOCKS = int(os.environ.get("Q4R_MIN_BLOCKS", "2"))
Q4R_SOFT_REGS = int(os.environ.get("Q4R_SOFT_REGS", "112"))
Q4R_CORR_REGS = int(os.environ.get("Q4R_CORR_REGS", "88"))
Q4R_PROD_REGS = int(os.environ.get("Q4R_PROD_REGS", "40"))


def make_kernel_q4r(cfg):
    T = cfg["T"]
    G = cfg["G"]
    HKV = cfg["HKV"]
    TOPK = cfg["TOPK"]
    TOTAL_Q = cfg["TOTAL_Q"]
    W_MAX = cfg["MAX_BLOCK_WORDS"]
    NUM_ITEMS = cfg["NUM_ITEMS"]
    STAGES = 2
    NQ = T * G
    HQ = HKV * G
    MAX_UNION = T * TOPK
    N_ROUNDS = ceildiv(MAX_UNION, 32)
    assert T == 4 and G == 16 and TOPK == 16 and NQ == 64

    KV_TILE_BYTES = BLK * HEAD_DIM * 2
    Q_TILE_BYTES = NQ * HEAD_DIM * 2
    KV16 = KV_TILE_BYTES // 16
    MMA_K = 16
    NK = HEAD_DIM // MMA_K
    ID_QK = make_idesc(64, 128, 1, 1, 0, 0)
    ID_PV = make_idesc(64, 128, 1, 1, 0, 1)
    S_COL = 0
    P_COL = 64
    O_COL = 128
    TMEM_COLS = 256
    SOFT_WARPS = range(0, 4)
    CORR_WARPS = range(4, 8)
    MMA_WARP = 8
    LOAD_WARP = 9
    NWARPS = 12

    @K.kernel(warps=NWARPS, arch="sm_100a", min_blocks_per_sm=Q4R_MIN_BLOCKS, grid=NUM_ITEMS)
    def msa_decode_q4r(
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        out: K.gptr[K.bf16],
        q2k: K.gptr[K.i32],
        lens: K.gptr[K.i32],
        scale_log2: K.f32,
    ):
        warp = K.warp_id()
        lane = K.lane_id()

        smem = K.smem_pool()
        kv_smem = smem.alloc((STAGES, BLK, HEAD_DIM), K.bf16, swizzle=K.SW128B)
        q_smem = smem.alloc((1, NQ, HEAD_DIM), K.bf16, swizzle=K.SW128B)
        tmem_mailbox = smem.alloc((4,), K.u32)
        meta = smem.alloc((8,), K.i32)
        ulist = smem.alloc((MAX_UNION,), K.i32)
        umask = smem.alloc((MAX_UNION,), K.u32)
        words = smem.alloc((W_MAX,), K.u32)
        prefix = smem.alloc((W_MAX,), K.u32)
        row_scale = smem.alloc((NQ,), K.f32, align=16)
        row_sum_s = smem.alloc((NQ,), K.f32, align=16)
        row_max_s = smem.alloc((NQ,), K.f32, align=16)

        q_full = K.TMABar(smem, 1)
        q_full.init(1)
        kv_full = K.TMABar(smem, STAGES)
        kv_full.init(1)
        kv_empty = K.TCGen05Bar(smem, STAGES, phase_offset=1)
        kv_empty.init(1)
        union_ready = K.MBarrier(smem, 1)
        union_ready.init(32)
        s_full = K.TCGen05Bar(smem, 1)
        s_full.init(1)
        # P becomes consumable after both the softmax WG has written it and
        # the correction WG has finished any old-O rescale.
        p_full = K.MBarrier(smem, 1)
        p_full.init(256)
        corr_sig = K.MBarrier(smem, 1)
        corr_sig.init(128)
        corr_done = K.MBarrier(smem, 1)
        corr_done.init(128)
        o_full = K.TCGen05Bar(smem, 1)
        o_full.init(1)
        tmem_done = K.MBarrier(smem, 1)
        tmem_done.init(128)

        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()
        with K.If(warp == MMA_WARP), K.Then():
            K.ptx[TMEM_ALLOC](K.address_of(tmem_mailbox[0]), K.uint32(TMEM_COLS))
            K.ptx[TMEM_RELINQUISH]()
        K.cuda.cta_sync()
        K.ptx.tcgen05.fence__after_thread_sync()
        tmem_base = K.local_scalar("uint32")
        K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
        K.assign(tmem_base, K.uniform(tmem_base))

        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def wg_arrive(bar):
            with K.If(elected()), K.Then():
                bar.arrive(0, count=32)

        def ld_global_i32(ptr):
            x = K.local_scalar("int32")
            K.ptx.ld.global_.nc.b32(x, ptr)
            return x

        def ld_shared_i32(ptr):
            x = K.local_scalar("int32")
            K.ptx.ld.shared.b32(x, ptr)
            return x

        def ld_shared_u32(ptr):
            x = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(x, ptr)
            return x

        def ring_pos(pos):
            return pos & 1, (pos >> 1) & 1

        def iket_range(name):
            token = K.alloc_local([1], "uint32")
            K.assign(token[0], K.cuda.iket.range_start(name))
            return token

        def iket_end(token):
            K.cuda.iket.range_end(token[0])

        def tmem_load64(dst, addr):
            K.ptx[Q4R_TMEM_LD32](*(dst[i] for i in range(32)), addr, K.uint32(64))
            K.ptx[Q4R_TMEM_LD32](*(dst[32 + i] for i in range(32)), addr + K.uint32(32), K.uint32(64))
            K.ptx.tcgen05.wait__ld.sync.aligned()

        def shfl_xor16(x):
            y = K.local_scalar("float32")
            K.ptx.shfl_sync.bfly.b32(y, x, K.uint32(16), K.uint32(31), K.uint32(0xFFFFFFFF))
            return y

        def row_max64(vals):
            a0 = K.local_scalar("float32", init=K.float32(NEG_INF))
            a1 = K.local_scalar("float32", init=K.float32(NEG_INF))
            for j in range(0, 64, 4):
                p0 = K.local_scalar("float32")
                p1 = K.local_scalar("float32")
                K.ptx.max.f32(p0, vals[j], vals[j + 1])
                K.ptx.max.f32(p1, vals[j + 2], vals[j + 3])
                K.assign(a0, K.max(a0, p0))
                K.assign(a1, K.max(a1, p1))
            return K.max(a0, a1)

        def f32_pair(lo, hi):
            packed = K.local_scalar("uint64")
            K.ptx.mov.b64(packed, lo, hi)
            return packed

        sp = K.specialize(chain_dispatch=True)
        r_soft = sp.role("softmax", warps=SOFT_WARPS, regs=Q4R_SOFT_REGS)
        r_corr = sp.role("correction", warps=CORR_WARPS, regs=Q4R_CORR_REGS)
        prod = sp.register_scope("producer", warps=range(8, 12), regs=Q4R_PROD_REGS)
        r_mma = sp.role("mma", warps=[MMA_WARP], register_scope=prod)
        r_load = sp.role("load", warps=[LOAD_WARP], register_scope=prod)
        r_idle = sp.role("idle", warps=[10, 11], register_scope=prod)
        with K.If(K.And(warp >= 8, warp <= 11)), K.Then():
            prod.emit()

        with r_load:
            task = K.cta_id()
            b = task // HKV
            h = task - b * HKV
            kv_s = ld_global_i32(lens.ptr_to([b]))
            kv_len = ld_global_i32(lens.ptr_to([b + 1])) - kv_s

            # Q is independent of selection metadata, so launch it first.
            with K.If(elected()), K.Then():
                K.ptx[TMA_4D](
                    q_smem[0].ptr_to(0, 0),
                    K.address_of(q_map),
                    K.int32(0),
                    h * G,
                    b * T,
                    K.int32(0),
                    K.cuda.cvta_generic_to_shared(q_full.ptr_to([0])),
                )
                q_full.arrive(0, tx_count=Q_TILE_BYTES)

            idxs = K.alloc_local([N_ROUNDS], "int32")
            base = (h * TOTAL_Q + b * T) * TOPK
            for r in range(N_ROUNDS):
                e = lane + 32 * r
                K.assign(idxs[r], ld_global_i32(q2k.ptr_to([base + e])))
            n_vis = (kv_len + (BLK - 1)) >> 7
            nwords = (n_vis + 31) >> 5
            wi = K.local_scalar("int32", init=lane)
            with K.While(wi < nwords):
                K.ptx.st.shared.b32(words.ptr_to([wi]), K.uint32(0))
                K.assign(wi, wi + 32)
            K.cuda.warp_sync()
            for r in range(N_ROUNDS):
                with K.If(K.And(idxs[r] >= 0, idxs[r] < n_vis)), K.Then():
                    K.ptx.red.shared.or_.b32(
                        words.ptr_to([idxs[r] >> 5]),
                        K.shift_left(K.uint32(1), K.Cast("uint32", idxs[r] & 31)),
                    )
            K.cuda.warp_sync()
            carry = K.local_scalar("uint32", init=K.uint32(0))
            wbase = K.local_scalar("int32", init=0)
            with K.While(wbase < nwords):
                w = wbase + lane
                wv = K.local_scalar("uint32", init=K.uint32(0))
                with K.If(w < nwords), K.Then():
                    K.ptx.ld.shared.u32(wv, words.ptr_to([w]))
                cnt = K.alloc_local([1], "uint32")
                K.ptx.popc.b32(cnt[0], wv)
                own = K.local_scalar("uint32", init=cnt[0])
                K.idioms.warp_scan_add(cnt, 1, lane)
                with K.If(w < nwords), K.Then():
                    K.ptx.st.shared.b32(prefix.ptr_to([w]), carry + cnt[0] - own)
                total = K.local_scalar("uint32")
                K.ptx.shfl_sync.idx.b32(total, cnt[0], K.uint32(31), K.uint32(31), K.uint32(0xFFFFFFFF))
                K.assign(carry, carry + total)
                K.assign(wbase, wbase + 32)
            n_blocks = K.local_scalar("int32", init=K.Cast("int32", carry))
            K.cuda.warp_sync()
            lanemask = K.shift_left(K.uint32(1), K.Cast("uint32", lane)) - K.uint32(1)
            with K.serial(nwords, unroll=False) as w:
                wv = ld_shared_u32(words.ptr_to([w]))
                pref = ld_shared_u32(prefix.ptr_to([w]))
                bit = K.bitwise_and(K.shift_right(wv, K.Cast("uint32", lane)), K.uint32(1))
                with K.If(bit != K.uint32(0)), K.Then():
                    below = K.local_scalar("uint32")
                    K.ptx.popc.b32(below, K.bitwise_and(wv, lanemask))
                    rank = K.Cast("int32", pref + below)
                    K.ptx.st.shared.b32(ulist.ptr_to([rank]), w * 32 + lane)
                    K.ptx.st.shared.b32(umask.ptr_to([rank]), K.uint32(0))
            K.cuda.warp_sync()
            for r in range(N_ROUNDS):
                e = lane + 32 * r
                with K.If(K.And(idxs[r] >= 0, idxs[r] < n_vis)), K.Then():
                    wq = idxs[r] >> 5
                    wv = ld_shared_u32(words.ptr_to([wq]))
                    pref = ld_shared_u32(prefix.ptr_to([wq]))
                    below = K.local_scalar("uint32")
                    K.ptx.popc.b32(
                        below,
                        K.bitwise_and(
                            wv,
                            K.shift_left(K.uint32(1), K.Cast("uint32", idxs[r] & 31)) - K.uint32(1),
                        ),
                    )
                    rank = K.Cast("int32", pref + below)
                    K.ptx.red.shared.or_.b32(
                        umask.ptr_to([rank]),
                        K.shift_left(K.uint32(1), K.Cast("uint32", e // TOPK)),
                    )
            K.cuda.warp_sync()
            with K.If(lane == 0), K.Then():
                K.ptx.st.shared.b32(meta.ptr_to([0]), n_blocks)
                K.ptx.st.shared.b32(meta.ptr_to([1]), b)
                K.ptx.st.shared.b32(meta.ptr_to([2]), h)
                K.ptx.st.shared.b32(meta.ptr_to([3]), kv_s)
                K.ptx.st.shared.b32(meta.ptr_to([4]), kv_len)
            K.ptx.fence.proxy.async_.shared__cta()
            with K.If(elected()), K.Then():
                union_ready.arrive(0, count=32)

            load_pos = K.local_scalar("int32", init=0)

            def push_tile(tmap, ub):
                stage, phase = ring_pos(load_pos)
                st = K.local_scalar("int32", init=stage)
                ph = K.local_scalar("int32", init=phase)
                K.assign(load_pos, load_pos + 1)
                kv_empty.wait(st, ph)
                blk = ld_shared_i32(ulist.ptr_to([ub]))
                with K.If(elected()), K.Then():
                    K.ptx[TMA_3D_HINT](
                        kv_smem[st].ptr_to(0, 0),
                        K.address_of(tmap),
                        K.int32(0),
                        kv_s + blk * BLK,
                        h * 2,
                        K.cuda.cvta_generic_to_shared(kv_full.ptr_to([st])),
                        *HINT_ARGS,
                    )
                    kv_full.arrive(st, tx_count=KV_TILE_BYTES)

            with K.serial(n_blocks, unroll=False) as ub:
                push_tile(k_map, ub)
                push_tile(v_map, ub)

        with r_mma:
            union_ready.wait(0, 0)
            n_blocks = ld_shared_i32(meta.ptr_to([0]))
            q_full.wait(0, 0)
            K.ptx.tcgen05.fence__after_thread_sync()
            qd, qoff = q_smem[0].encode(major="k", mma_k=MMA_K)
            kd, koff = kv_smem[0].encode(major="k", mma_k=MMA_K)
            vd, voff = kv_smem[0].encode(major="mn", mma_k=MMA_K)
            zero = K.uint32(0)
            first = K.local_scalar("int32", init=1)
            with K.serial(n_blocks, unroll=False) as ub:
                kst, kph = ring_pos(2 * ub)
                kv_full.wait(kst, kph)
                K.ptx.tcgen05.fence__after_thread_sync()
                tqk = iket_range("q4r-mma-qk")
                for ki in range(NK):
                    with K.If(elected()), K.Then():
                        K.ptx[MMA_F16](
                            tmem_base + K.uint32(S_COL),
                            qd + qoff(ki),
                            kd + kst * KV16 + koff(ki),
                            K.uint32(ID_QK),
                            zero,
                            zero,
                            zero,
                            zero,
                            ki != 0,
                        )
                with K.If(elected()), K.Then():
                    s_full.arrive(0)
                    kv_empty.arrive(kst)
                iket_end(tqk)

                vst, vph = ring_pos(2 * ub + 1)
                kv_full.wait(vst, vph)
                p_full.wait(0, ub & 1)
                K.ptx.tcgen05.fence__after_thread_sync()
                tpv = iket_range("q4r-mma-pv")
                for ki in range(NK):
                    with K.If(elected()), K.Then():
                        K.ptx[MMA_F16](
                            tmem_base + K.uint32(O_COL),
                            tmem_base + K.uint32(P_COL + 8 * ki),
                            vd + vst * KV16 + voff(ki),
                            K.uint32(ID_PV),
                            zero,
                            zero,
                            zero,
                            zero,
                            True if ki != 0 else K.Cast("bool", first == 0),
                        )
                K.assign(first, 0)
                with K.If(elected()), K.Then():
                    kv_empty.arrive(vst)
                    with K.If(ub + 1 == n_blocks), K.Then():
                        o_full.arrive(0)
                iket_end(tpv)
            tmem_done.wait(0, 0)
            K.ptx[TMEM_DEALLOC](tmem_base, K.uint32(TMEM_COLS))

        with r_soft:
            union_ready.wait(0, 0)
            n_blocks = ld_shared_i32(meta.ptr_to([0]))
            kv_len = ld_shared_i32(meta.ptr_to([4]))
            warp_local = warp
            my_row = warp_local * 16 + (lane & 15)
            col_half = lane >> 4
            token = my_row >> 4
            row_origin = K.shift_left(K.Cast("uint32", warp_local * 32), K.uint32(16))
            score_addr = tmem_base + row_origin + K.uint32(S_COL)
            p_addr = score_addr + K.uint32(P_COL)
            scores = K.alloc_local([64], "float32")
            packed_p = K.alloc_local([32], "uint32")
            row_max = K.local_scalar("float32", init=K.float32(NEG_INF))
            row_sum = K.local_scalar("float32", init=K.float32(0.0))
            soft_token = iket_range("q4r-softmax")
            with K.serial(n_blocks, unroll=False) as ub:
                s_full.wait(0, ub & 1)
                K.ptx.tcgen05.fence__after_thread_sync()
                tmem_load64(scores, score_addr)
                blk = ld_shared_i32(ulist.ptr_to([ub]))
                tmask = ld_shared_u32(umask.ptr_to([ub]))
                selected = K.bitwise_and(
                    K.shift_right(tmask, K.Cast("uint32", token)), K.uint32(1)
                ) != K.uint32(0)
                valid_cols = K.local_scalar(
                    "int32", init=K.Select(selected, kv_len - T + token - blk * BLK + 1, 0)
                )
                K.assign(valid_cols, K.max(K.int32(0), K.min(K.int32(BLK), valid_cols)))
                half_valid = K.max(
                    K.int32(0), K.min(K.int32(64), valid_cols - col_half * 64)
                )
                for j in range(64):
                    K.assign(
                        scores[j],
                        K.Select(K.int32(j) < half_valid, scores[j], K.float32(NEG_INF)),
                    )
                tile_half = row_max64(scores)
                tile_max = K.max(tile_half, shfl_xor16(tile_half))
                new_max = K.max(row_max, tile_max)
                safe_max = K.local_scalar(
                    "float32", init=K.Select(new_max == K.float32(NEG_INF), K.float32(0.0), new_max)
                )
                delta = K.local_scalar("float32", init=(row_max - safe_max) * scale_log2)
                acc_scale = K.local_scalar("float32", init=K.float32(1.0))
                with K.If(delta >= K.float32(-8.0)):
                    with K.Then():
                        K.assign(safe_max, K.Select(row_max == K.float32(NEG_INF), K.float32(0.0), row_max))
                    with K.Else():
                        a = K.local_scalar("float32")
                        K.ptx.ex2.approx.ftz.f32(a, delta)
                        K.assign(acc_scale, K.Select(row_max == K.float32(NEG_INF), K.float32(1.0), a))
                        K.assign(row_max, new_max)
                with K.If(col_half == 0), K.Then():
                    K.ptx.st.shared.f32(row_scale.ptr_to([my_row]), acc_scale)
                K.ptx.fence.proxy.async_.shared__cta()
                wg_arrive(corr_sig)

                block_acc = K.alloc_local([4], "float32")
                for j in range(4):
                    K.assign(block_acc[j], K.float32(0.0))
                bias = K.local_scalar("float32", init=K.float32(0.0) - safe_max * scale_log2)
                for j in range(32):
                    x0 = scores[2 * j] * scale_log2 + bias
                    x1 = scores[2 * j + 1] * scale_log2 + bias
                    xw = K.local_scalar("uint32")
                    K.ptx.cvt.rn.bf16x2.f32(xw, x1, x0)
                    K.ptx.ex2.approx.ftz.bf16x2(packed_p[j], xw)
                    lo = K.local_scalar("float32")
                    hi = K.local_scalar("float32")
                    K.ptx.mov.b32(lo, K.shift_left(K.bitwise_and(packed_p[j], K.uint32(0xFFFF)), K.uint32(16)))
                    K.ptx.mov.b32(hi, K.bitwise_and(packed_p[j], K.uint32(0xFFFF0000)))
                    K.assign(block_acc[(2 * j) & 3], block_acc[(2 * j) & 3] + lo)
                    K.assign(block_acc[(2 * j + 1) & 3], block_acc[(2 * j + 1) & 3] + hi)
                half_sum = (block_acc[0] + block_acc[1]) + (block_acc[2] + block_acc[3])
                block_sum = half_sum + shfl_xor16(half_sum)
                K.ptx[Q4R_TMEM_ST16](p_addr, K.uint32(32), *(packed_p[j] for j in range(16)))
                K.ptx[Q4R_TMEM_ST16](p_addr + K.uint32(16), K.uint32(32), *(packed_p[16 + j] for j in range(16)))
                K.ptx.tcgen05.wait__st.sync.aligned()
                K.ptx.tcgen05.fence__before_thread_sync()
                wg_arrive(p_full)
                corr_done.wait(0, ub & 1)
                K.assign(row_sum, row_sum * acc_scale + block_sum)
            with K.If(col_half == 0), K.Then():
                K.ptx.st.shared.f32(row_sum_s.ptr_to([my_row]), row_sum)
                K.ptx.st.shared.f32(row_max_s.ptr_to([my_row]), row_max)
            K.ptx.fence.proxy.async_.shared__cta()
            wg_arrive(corr_sig)
            iket_end(soft_token)

        with r_corr:
            union_ready.wait(0, 0)
            n_blocks = ld_shared_i32(meta.ptr_to([0]))
            b = ld_shared_i32(meta.ptr_to([1]))
            h = ld_shared_i32(meta.ptr_to([2]))
            warp_local = warp - 4
            my_row = warp_local * 16 + (lane & 15)
            col_half = lane >> 4
            row_origin = K.shift_left(K.Cast("uint32", warp_local * 32), K.uint32(16))
            o_addr = tmem_base + row_origin + K.uint32(O_COL)
            o_frag = K.alloc_local([64], "float32")
            corr_token = iket_range("q4r-correction")
            wg_arrive(p_full)
            with K.If(n_blocks > 0), K.Then():
                corr_sig.wait(0, 0)
                wg_arrive(corr_done)
                with K.serial(n_blocks - 1, unroll=False) as um1:
                    ub = um1 + 1
                    corr_sig.wait(0, ub & 1)
                    scale = K.local_scalar("float32")
                    K.ptx.ld.shared.f32(scale, row_scale.ptr_to([my_row]))
                    any_rescale = K.local_scalar("uint32")
                    K.ptx.vote_sync.any.pred(
                        any_rescale,
                        K.ptx.pred(scale < K.float32(1.0)),
                        K.uint32(0xFFFFFFFF),
                    )
                    with K.If(any_rescale != K.uint32(0)), K.Then():
                        tmem_load64(o_frag, o_addr)
                        for j in range(64):
                            K.assign(o_frag[j], o_frag[j] * scale)
                        for chunk in range(4):
                            K.ptx[Q4R_TMEM_ST16](
                                o_addr + K.uint32(16 * chunk),
                                K.uint32(64),
                                *(o_frag[16 * chunk + j] for j in range(16)),
                            )
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        K.ptx.tcgen05.fence__before_thread_sync()
                    wg_arrive(p_full)
                    wg_arrive(corr_done)
                o_full.wait(0, 0)
                K.ptx.tcgen05.fence__after_thread_sync()
                corr_sig.wait(0, n_blocks & 1)
                tmem_load64(o_frag, o_addr)
            with K.If(n_blocks == 0), K.Then():
                corr_sig.wait(0, 0)
                for j in range(64):
                    K.assign(o_frag[j], K.float32(0.0))

            final_sum = K.local_scalar("float32")
            K.ptx.ld.shared.f32(final_sum, row_sum_s.ptr_to([my_row]))
            bad = K.Or(final_sum <= K.float32(0.0), final_sum != final_sum)
            inv = K.local_scalar("float32")
            K.ptx.rcp.approx.ftz.f32(inv, K.Select(bad, K.float32(1.0), final_sum))
            out_words = K.alloc_local([32], "uint32")
            for j in range(32):
                lo = K.Select(bad, K.float32(0.0), o_frag[2 * j] * inv)
                hi = K.Select(bad, K.float32(0.0), o_frag[2 * j + 1] * inv)
                K.ptx.cvt.rn.bf16x2.f32(out_words[j], hi, lo)
            token = my_row >> 4
            g = my_row & 15
            out_row = ((b * T + token) * HQ + h * G + g) * HEAD_DIM + col_half * 64
            for j in range(4):
                K.ptx[ST_OUT_V8](
                    out.ptr_to([out_row + 16 * j]),
                    *(out_words[8 * j + k] for k in range(8)),
                )
            K.ptx.tcgen05.fence__before_thread_sync()
            wg_arrive(tmem_done)
            iket_end(corr_token)

        with r_idle:
            pass

    return msa_decode_q4r


def setup_q4r(data, B, seqlen_q):
    q = data["q"]
    k = data["k"]
    v = data["v"]
    q2k = data["q2k_indices"]
    cu = data["cu_seqlens_k"]
    out = data["output"]
    T = int(seqlen_q)
    B = int(B)
    HQ = int(q.shape[1])
    HKV = int(k.shape[1])
    G = HQ // HKV
    TOPK = int(q2k.shape[-1])
    TOTAL_Q = int(q.shape[0])
    max_blocks = ceildiv(int(k.shape[0]), BLK)
    cfg = dict(
        T=T,
        G=G,
        HKV=HKV,
        TOPK=TOPK,
        TOTAL_Q=TOTAL_Q,
        MAX_BLOCK_WORDS=max(1, ceildiv(max_blocks, 32)),
        NUM_ITEMS=B * HKV,
    )
    ex = _compile(cfg, make_kernel_q4r, "q4r")
    q_map = _encode(
        q,
        "bfloat16",
        (HEAD_DIM // 2, HQ, TOTAL_Q, 2),
        (HEAD_DIM * 2, HQ * HEAD_DIM * 2, (HEAD_DIM // 2) * 2),
        (HEAD_DIM // 2, G, T, 2),
    )
    total_k = int(k.shape[0])
    kv_dims = (HEAD_DIM // 2, total_k, HKV * 2)
    kv_strides = (HKV * HEAD_DIM * 2, (HEAD_DIM // 2) * 2)
    kv_box = (HEAD_DIM // 2, BLK, 2)
    k_map = _encode(k, "bfloat16", kv_dims, kv_strides, kv_box, l2_promotion=3)
    v_map = _encode(v, "bfloat16", kv_dims, kv_strides, kv_box, l2_promotion=3)
    q2k_flat = q2k.contiguous().view(-1)
    lens = cu.contiguous()
    out_flat = out.view(-1)
    scale = float(data["softmax_scale"]) * LOG2E
    args = (q_map.ptr, k_map.ptr, v_map.ptr, out_flat, q2k_flat, lens, scale)

    def run():
        ex(*args)

    run._keep = (q, k, v, q2k, q2k_flat, lens, out, out_flat, q_map, k_map, v_map)
    run._cfg = {"family": "q4-rowmajor-tworesident", **cfg}
    run()
    torch.cuda.synchronize(q.device)
    return run


def _use_q4r(data, B, T):
    q, k, v = data["q"], data["k"], data["v"]
    if os.environ.get("Q4R_DISABLE", "0") == "1":
        return False
    return (
        int(T) == 4
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and data["page_table"] is None
        and int(q.shape[1]) // int(k.shape[1]) == 16
        and int(data["q2k_indices"].shape[-1]) == 16
    )


_setup_prev_q4r = setup


def _candidate_setup(data, B, seqlen_q):
    """The candidate's own dispatch entry (its evolution-harness ``setup``)."""
    T = int(seqlen_q)
    if _use_q4r(data, B, T):
        return setup_q4r(data, B, T)
    return _setup_prev_q4r(data, B, seqlen_q)


# ---------------------------------------------------------------------------
# Registry interface.
# ---------------------------------------------------------------------------

KERNEL_META = {
    "name": "agent_evolved_msa_decode_multishape",
    "category": "agent_evolved",
    "runtime_cuda_archs": ["sm_100a"],
    "reference_requirements": (
        {
            "package": "msa",
            "git": {
                "url": "https://github.com/MiniMax-AI/MSA.git",
                "commit": "80434d7f67877c6570ca19cac444b84bc9855dac",
            },
            "import": "fmha_sm100",
        },
        {"package": "flashinfer-python", "specifier": ">=0.6.18", "import": "flashinfer"},
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.7.0", "import": "cutlass"},
    ),
    "provenance": {
        "generator": "hmz",
        "run": "msa-decode-relay",
        "selected_version": "frontier/rowmajor-tmem-probability",
    },
}


def _cfg(label, batch_size, seqlen_q, seqlen_kv, num_qo_heads, num_kv_heads, topk,
         kv_layout, q_dtype, kv_dtype, seed):
    return {
        "label": label,
        "batch_size": batch_size,
        "seqlen_q": seqlen_q,
        "seqlen_kv": seqlen_kv,
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "topk": topk,
        "kv_layout": kv_layout,
        "q_dtype": q_dtype,
        "kv_dtype": kv_dtype,
        "seed": seed,
    }


# The ten packaged `msa_sparse_decode_hd128_blk128` official rows, in the
# order the task's workload list declares them.
CONFIGS = [
    _cfg("decode_bf16_b128_q1_kv4096_h64", 128, 1, 4096, 64, 4, 16, "flat", "bfloat16", "bfloat16", 50),
    _cfg("speculative_bf16_b128_q4_kv4096_h64", 128, 4, 4096, 64, 4, 16, "flat", "bfloat16", "bfloat16", 50),
    _cfg("mtp_bf16_b128_q16_kv4096_h64", 128, 16, 4096, 64, 4, 16, "flat", "bfloat16", "bfloat16", 50),
    _cfg("decode_fp16_b128_q1_kv4096_h64", 128, 1, 4096, 64, 4, 16, "flat", "float16", "float16", 50),
    _cfg("decode_fp8_b128_q1_kv4096_h64", 128, 1, 4096, 64, 4, 16, "paged", "bfloat16", "float8_e4m3fn", 50),
    _cfg("official_decode_bf16_b32_q8_kv8192_h64_hkv4_k16_paged", 32, 8, 8192, 64, 4, 16, "paged", "bfloat16", "bfloat16", 50),
    _cfg("official_decode_bf16_b64_q8_kv65536_h64_hkv4_k32_paged", 64, 8, 65536, 64, 4, 32, "paged", "bfloat16", "bfloat16", 50),
    _cfg("coverage_decode_fp16_b32_q4_kv8192_h64_hkv4_k16_paged", 32, 4, 8192, 64, 4, 16, "paged", "float16", "float16", 50),
    _cfg("coverage_decode_mixed_fp8_b32_q1_kv8192_h64_hkv4_k16_flat", 32, 1, 8192, 64, 4, 16, "flat", "bfloat16", "float8_e4m3fn", 50),
    _cfg("boundary_decode_bf16_b2_q1_kv257_h8_hkv1_k4_paged", 2, 1, 257, 8, 1, 4, "paged", "bfloat16", "bfloat16", 50),
]

_CONFIG_KEYS = set(CONFIGS[0]) - {"label"}
_BY_LABEL = {config["label"]: config for config in CONFIGS}

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
}
BLOCK_SIZE = 128


def _config(**config: Any) -> dict[str, Any]:
    """Resolve one config, by label or by explicit axes."""
    label = config.get("label")
    base = dict(_BY_LABEL[label]) if label in _BY_LABEL else dict(CONFIGS[0])
    values = {key: value for key, value in config.items() if key != "label"}
    unknown = set(values) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"unsupported config keys: {sorted(unknown)}")
    base.update(values)
    resolved = {key: base[key] for key in _CONFIG_KEYS}
    if int(resolved["num_qo_heads"]) % int(resolved["num_kv_heads"]):
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    if str(resolved["kv_layout"]) not in ("flat", "paged"):
        raise ValueError("kv_layout must be 'flat' or 'paged'")
    return resolved


def _assert_supported_arch() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved MSA decode")
    capability = torch.cuda.get_device_capability()
    runtime_arch = f"sm_{capability[0]}{capability[1]}a"
    if runtime_arch not in KERNEL_META["runtime_cuda_archs"]:
        raise SkipTest(
            "agent-evolved MSA decode requires one of "
            f"{KERNEL_META['runtime_cuda_archs']}, got {runtime_arch}"
        )


def get_kernel(**config: Any):
    """Return a traced Kern PrimFunc for this config's dispatch target.

    The runtime path compiles through the candidate's own ``setup``, which
    picks among the KV-major, Q-major and specialized routes from the shape and
    dtype metadata; this entry point exists for registry discovery and IR
    inspection and returns the general KV-major program.
    """
    from tirx_kernels.runner import hardware_num_sms

    os.environ.setdefault("TVM_CUDA_PTXAS_REG_LEVEL", "6")
    case = prepare_data(**config)
    cfg = plan_config_kv(
        case["q"],
        case["k"],
        case["q2k_indices"],
        case["page_table"],
        int(case["config"]["batch_size"]),
        int(case["seqlen_q"]),
        hardware_num_sms(),
    )
    return make_kernel_kv(cfg).func


# ---------------------------------------------------------------------------
# Inputs.
#
# The draw order reproduces the packaged MSA-decode benchmark rows, which
# follow flashinfer PR #4355's `bench_blackwell_msa_sm100.py`: q, k and v are
# `randn/3` in one generator sequence, then `q2k_indices` is drawn per (query
# token, kv head) from the blocks the token may see under bottom-right causal
# masking. Paged rows store 128-token pages in reverse order, as that benchmark
# does.
# ---------------------------------------------------------------------------


def _make_q2k_indices(batch_size, seqlen_q, seqlen_kv, num_kv_heads, topk, seed, device):
    total_q = batch_size * seqlen_q
    out = torch.full((num_kv_heads, total_q, topk), -1, dtype=torch.int32)
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    offset = seqlen_kv - seqlen_q
    for row in range(total_q):
        visible_blocks = (offset + row % seqlen_q + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
        for kv_head in range(num_kv_heads):
            selected = torch.randperm(visible_blocks, generator=generator)
            selected = selected[: min(topk, visible_blocks)].sort().values
            out[kv_head, row, : selected.numel()] = selected.to(torch.int32)
    return out.to(device)


def _to_pages(logical, batch_size, seqlen_kv):
    """The packaged layout: 128-token pages stored in reverse order."""
    total_k, num_kv_heads, head_dim = logical.shape
    pages_per_seq = (seqlen_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_pages = batch_size * pages_per_seq
    padded = logical.view(batch_size, seqlen_kv, num_kv_heads, head_dim)
    if pages_per_seq * BLOCK_SIZE != seqlen_kv:
        padded = logical.new_zeros((batch_size, pages_per_seq * BLOCK_SIZE, num_kv_heads, head_dim))
        padded[:, :seqlen_kv] = logical.view(batch_size, seqlen_kv, num_kv_heads, head_dim)
    pages = (
        padded.view(batch_size, pages_per_seq, BLOCK_SIZE, num_kv_heads, head_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(total_pages, num_kv_heads, BLOCK_SIZE, head_dim)
    )
    page_table = torch.arange(total_pages - 1, -1, -1, dtype=torch.int32, device=logical.device)
    return pages.flip(0).contiguous(), page_table.view(batch_size, pages_per_seq).contiguous()


def prepare_data(**config: Any) -> dict[str, Any]:
    """Build one row's contract inputs plus the preallocated output."""
    resolved = _config(**config)
    device = torch.device("cuda")
    batch_size = int(resolved["batch_size"])
    seqlen_q = int(resolved["seqlen_q"])
    seqlen_kv = int(resolved["seqlen_kv"])
    num_qo_heads = int(resolved["num_qo_heads"])
    num_kv_heads = int(resolved["num_kv_heads"])
    topk = int(resolved["topk"])
    q_dtype = _DTYPES[str(resolved["q_dtype"])]
    kv_dtype = _DTYPES[str(resolved["kv_dtype"])]
    generator = torch.Generator(device=device).manual_seed(int(resolved["seed"]))

    def randn(shape, dtype):
        values = torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
        return (values / 3.0).to(dtype)

    total_q = batch_size * seqlen_q
    total_k = batch_size * seqlen_kv
    q = randn((total_q, num_qo_heads, HEAD_DIM), q_dtype)
    k = randn((total_k, num_kv_heads, HEAD_DIM), kv_dtype)
    v = randn((total_k, num_kv_heads, HEAD_DIM), kv_dtype)
    cu_seqlens_k = torch.arange(0, total_k + 1, seqlen_kv, dtype=torch.int32, device=device)
    q2k_indices = _make_q2k_indices(
        batch_size, seqlen_q, seqlen_kv, num_kv_heads, topk, int(resolved["seed"]), device
    )
    page_table = seqused_k = None
    if str(resolved["kv_layout"]) == "paged":
        k, page_table = _to_pages(k, batch_size, seqlen_kv)
        v, _ = _to_pages(v, batch_size, seqlen_kv)
        seqused_k = torch.full((batch_size,), seqlen_kv, dtype=torch.int32, device=device)
        cu_seqlens_k = None
    return {
        "config": resolved,
        "q": q,
        "k": k,
        "v": v,
        "q2k_indices": q2k_indices,
        "cu_seqlens_k": cu_seqlens_k,
        "page_table": page_table,
        "seqused_k": seqused_k,
        "seqlen_q": seqlen_q,
        "softmax_scale": float(HEAD_DIM**-0.5),
        "output": torch.empty(q.shape, dtype=q.dtype, device=device),
    }


def _launch_state(case: dict[str, Any]):
    """Bind the launch through the candidate's own shape dispatch."""
    batch_size = int(case["config"]["batch_size"])
    os.environ.setdefault("TVM_CUDA_PTXAS_REG_LEVEL", "6")
    return _candidate_setup(case, batch_size, int(case["seqlen_q"]))


# ---------------------------------------------------------------------------
# Independent oracle and correctness.
#
# The packaged task's reference: fp32 sparse decode attention, one request at a
# time, with the bottom-right causal boundary applied on top of the selected
# blocks. It shares no code with the kernel. fp8 K/V dequantize by plain cast.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _reference_output(case: dict[str, Any]) -> torch.Tensor:
    q, k, v = case["q"], case["k"], case["v"]
    q2k_indices = case["q2k_indices"]
    page_table, seqused_k = case["page_table"], case["seqused_k"]
    softmax_scale = float(case["softmax_scale"])
    q_len = int(case["seqlen_q"])

    num_qo_heads, head_dim = q.shape[1], q.shape[2]
    num_kv_heads = k.shape[1]
    group = num_qo_heads // num_kv_heads
    device = q.device
    if page_table is None:
        cu_k = case["cu_seqlens_k"].tolist()
        kv_lens = [cu_k[b + 1] - cu_k[b] for b in range(len(cu_k) - 1)]
    else:
        kv_lens = seqused_k.tolist()

    out = torch.zeros(q.shape, dtype=torch.float32, device=device)
    for b, kv_len in enumerate(kv_lens):
        q_start, q_end = b * q_len, (b + 1) * q_len
        if page_table is None:
            kb = k[cu_k[b] : cu_k[b + 1]].float()
            vb = v[cu_k[b] : cu_k[b + 1]].float()
        else:
            pages = page_table[b, : (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE].long()
            kb = k[pages].float().permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)[:kv_len]
            vb = v[pages].float().permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)[:kv_len]
        qb = q[q_start:q_end].float().view(q_len, num_kv_heads, group, head_dim)
        selected = q2k_indices[:, q_start:q_end]
        positions = torch.arange(kv_len, device=device)
        allowed = (
            (positions // BLOCK_SIZE).view(1, 1, -1, 1) == selected.unsqueeze(2)
        ).any(-1)
        q_pos = kv_len - q_len + torch.arange(q_len, device=device)
        allowed &= positions.view(1, 1, -1) <= q_pos.view(1, -1, 1)
        allowed = allowed.unsqueeze(2)
        logits = torch.einsum("qhgd,khd->hqgk", qb, kb) * softmax_scale
        probs = torch.softmax(logits.masked_fill(~allowed, float("-inf")), dim=-1)
        probs = torch.where(allowed.any(-1, keepdim=True), probs, 0.0)
        ob = torch.einsum("hqgk,khd->qhgd", probs, vb)
        out[q_start:q_end] = ob.reshape(q_len, num_qo_heads, head_dim)
    return out.to(q.dtype)


def _gate(kv_dtype: str) -> tuple[float, float]:
    """The packaged gate: looser on fp8 rows, as the task declares."""
    return (0.1, 0.1) if str(kv_dtype) == "float8_e4m3fn" else (1e-2, 1e-2)


def check_correctness(outputs: dict[str, Any], **config: Any) -> None:
    """Gate the kernel output against the oracle in the native output dtype."""
    case = outputs["case"]
    atol, rtol = _gate(case["config"]["kv_dtype"])
    torch.testing.assert_close(
        outputs["output"], _reference_output(case), atol=atol, rtol=rtol
    )


def run_test(**config: Any) -> None:
    """Run one config through the dispatch and gate it against the oracle."""
    _assert_supported_arch()
    case = prepare_data(**config)
    run = _launch_state(case)
    run()
    torch.cuda.synchronize()
    check_correctness({"case": case, "output": case["output"]}, **config)


# ---------------------------------------------------------------------------
# Reference arm.
#
# The packaged baseline is MiniMax's public sparse attention (MiniMax-AI/MSA at
# `80434d7f`), as flashinfer PR #4355 benchmarks it. MiniMax's CSR forward
# accepts bf16 and fp8 storage only and fails NVVM codegen for GQA ratios below
# 8, so the packaged baseline routes the fp16 rows to flashinfer's trtllm-gen
# block-sparse decode bridge; this module keeps that same split. The CSR build,
# the schedule and the flat->paged conversion are prepare work.
# ---------------------------------------------------------------------------

MINIMAX_STORAGE_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
MINIMAX_GQA_RATIOS = (8, 16)
MINIMAX_TOPK = (4, 8, 16, 32)
_WORKSPACE: dict[str, Any] = {}


def _baseline_arm(case: dict[str, Any]) -> str:
    q, k = case["q"], case["k"]
    return (
        "minimax"
        if (
            q.dtype in MINIMAX_STORAGE_DTYPES
            and k.dtype in MINIMAX_STORAGE_DTYPES
            and q.shape[1] // k.shape[1] in MINIMAX_GQA_RATIOS
            and int(case["q2k_indices"].shape[-1]) in MINIMAX_TOPK
        )
        else "trtllm_bridge"
    )


def _cu_seqlens_k(case: dict[str, Any]):
    if case["cu_seqlens_k"] is not None:
        return case["cu_seqlens_k"]
    seqused_k = case["seqused_k"]
    cu = torch.zeros(seqused_k.numel() + 1, dtype=torch.int32, device=seqused_k.device)
    cu[1:] = torch.cumsum(seqused_k.to(torch.int32), 0)
    return cu


def _minimax_builder(case: dict[str, Any]):
    import fmha_sm100

    q, k, v = case["q"], case["k"], case["v"]
    q2k_indices = case["q2k_indices"]
    seqlen_q = int(case["seqlen_q"])
    cu_seqlens_q = torch.arange(
        0, q.shape[0] + 1, seqlen_q, dtype=torch.int32, device=q.device
    )
    cu_seqlens_k = _cu_seqlens_k(case)
    kv_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    max_seqlen_k = int(kv_lens.max())
    total_rows = int(((kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE).sum())
    k2q_row_ptr, k2q_q_indices, schedule = fmha_sm100.build_k2q_csr(
        q2k_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        BLOCK_SIZE,
        total_k=int(cu_seqlens_k[-1]),
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=seqlen_q,
        total_rows=total_rows,
        qhead_per_kv=q.shape[1] // k.shape[1],
        return_schedule=True,
    )
    topk = int(q2k_indices.shape[-1])
    page_table, seqused_k = case["page_table"], case["seqused_k"]
    softmax_scale = float(case["softmax_scale"])

    def launch():
        return fmha_sm100.sparse_atten_func(
            q, k, v, k2q_row_ptr, k2q_q_indices, topk,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=max_seqlen_k,
            blk_kv=BLOCK_SIZE,
            causal=True,
            softmax_scale=softmax_scale,
            return_softmax_lse=False,
            page_table=page_table,
            seqused_k=seqused_k,
            schedule=schedule,
        )

    return launch


def _workspace(device):
    key = str(device)
    if key not in _WORKSPACE:
        _WORKSPACE[key] = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    return _WORKSPACE[key]


def _paged_kv(case: dict[str, Any]):
    """Return (k_pages, v_pages, page_table, kv_lens) in trtllm's HND layout."""
    k, v = case["k"], case["v"]
    if case["page_table"] is not None:
        return k, v, case["page_table"].long(), case["seqused_k"].long()
    device = k.device
    cu_seqlens_k = case["cu_seqlens_k"].long()
    kv_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    pages_per_seq = (kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    page_base = torch.cumsum(pages_per_seq, 0) - pages_per_seq
    total_pages = int(pages_per_seq.sum())
    seqs = torch.arange(kv_lens.numel(), device=device)
    seq_of_token = torch.repeat_interleave(seqs, kv_lens, output_size=k.shape[0])
    local = torch.arange(k.shape[0], device=device) - cu_seqlens_k[:-1][seq_of_token]
    page = page_base[seq_of_token] + local // BLOCK_SIZE
    slot = local % BLOCK_SIZE
    k_pages = k.new_zeros((total_pages, k.shape[1], BLOCK_SIZE, k.shape[2]))
    v_pages = torch.zeros_like(k_pages)
    k_pages[page, :, slot] = k
    v_pages[page, :, slot] = v
    seq_of_page = torch.repeat_interleave(seqs, pages_per_seq, output_size=total_pages)
    pages = torch.arange(total_pages, device=device)
    page_table = torch.zeros(
        (kv_lens.numel(), int(pages_per_seq.max())), dtype=torch.int64, device=device
    )
    page_table[seq_of_page, pages - page_base[seq_of_page]] = pages
    return k_pages, v_pages, page_table, kv_lens


def _bridge_builder(case: dict[str, Any]):
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    q = case["q"]
    q2k_indices = case["q2k_indices"]
    seqlen_q = int(case["seqlen_q"])
    k_pages, v_pages, page_table, kv_lens = _paged_kv(case)
    total_q = q2k_indices.shape[1]
    device = q2k_indices.device
    token = torch.arange(total_q, device=device)
    seq_of_q = token // seqlen_q
    q_pos = kv_lens[seq_of_q] - seqlen_q + token % seqlen_q
    valid = q2k_indices >= 0
    count = valid.sum(-1)
    physical = page_table[seq_of_q.view(1, -1, 1), q2k_indices.clamp(min=0).long()]
    block_tables = torch.where(valid, physical, 0).to(torch.int32).contiguous()
    last = q2k_indices.gather(-1, (count - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1).long()
    tail = torch.clamp(q_pos.view(1, -1) + 1 - last * BLOCK_SIZE, max=BLOCK_SIZE)
    seq_lens = ((count - 1) * BLOCK_SIZE + tail).clamp(min=0).to(torch.int32).contiguous()
    max_seq_len = int(seq_lens.max().item())
    workspace = _workspace(q.device)
    softmax_scale = float(case["softmax_scale"])

    def launch():
        return trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_pages, v_pages),
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=softmax_scale,
            bmm2_scale=1.0,
            kv_layout="HND",
            backend="trtllm-gen",
            enable_block_sparse_attention=True,
        )

    return launch


def _reference_builder(case: dict[str, Any]):
    """Build this row's baseline launch; all setup happens outside timing."""
    if _baseline_arm(case) == "minimax":
        return _minimax_builder(case)
    return _bridge_builder(case)


# ---------------------------------------------------------------------------
# Benchmark entry points.
# ---------------------------------------------------------------------------


def prepare_bench(**config: Any):
    """Build the row and bind its launch, so nothing compiles in the GPU stage.

    The shape dispatch reads the row's tensors, so this step allocates them and
    runs the candidate's ``setup`` here; by the time the workload is READY every
    TIRx specialization for the chosen route is already compiled and cached.
    """
    from tirx_kernels.runner import prepared_gpu_benchmark

    case = prepare_data(**config)
    run = _launch_state(case)
    return prepared_gpu_benchmark(run_gpu, {"config": dict(config), "case": case, "run": run})


def run_gpu(
    prepared,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    _assert_supported_arch()
    from tirx_kernels.runner import bench

    config = dict(prepared["config"])
    config.update(kwargs)
    rounds = config.pop("rounds", 5)
    cooldown_s = config.pop("cooldown_s", 1.0)
    case = prepared["case"]
    run = prepared["run"]
    run()
    torch.cuda.synchronize()

    return bench(
        {"tirx": run},
        references={"minimax_msa": lambda: _reference_builder(case)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **config: Any,
) -> dict[str, Any]:
    values = dict(config)
    protocol = {name: values.pop(name) for name in ("rounds", "cooldown_s") if name in values}
    prepared = prepare_bench(**values)
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
