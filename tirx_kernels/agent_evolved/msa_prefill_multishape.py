# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved SM100a MiniMax sparse-attention (MSA) prefill, all official shapes.

This complements `msa_prefill_b1_q4096` rather than replacing it: it covers
every official MSA prefill row, including the FP8-KV and paged layouts the
single-shape port does not implement, but it is about 14% slower on the one
flat bf16 row they share (scored back to back through the evolution harness,
3.157x/3.168x for the specialized kernel against 2.725x/2.736x for this one).
Prefer the specialized kernel for that exact shape.

Registered rows: flat bf16 (B=1, Q=KV=4096, Hq=64, Hkv=4, top-k 16) and paged
bf16 (B=3, Q=4096, KV=8192, Hq=8, Hkv=2, top-k 4). Selection is
`q2k_indices` int32[Hkv, total_q, topk] of ascending sequence-local KV block
ids padded with -1, under bottom-right causal masking.

The selected kernel is the `dispatch-qmajor-xalias-csr-kvmajor` frontier member
of the 2026-09-11 multi-shape MSA-prefill evolution run. Everything from the
module docstring's mechanism notes down to `setup_qm` is that candidate's
source; this module adds the registry interface, input generation, the
independent oracle, and the MiniMax reference arm.

Only the two bf16 rows are registered. The candidate also implements the
official flat-FP8 row through a third route, but that route carries its block
partials in E2M1 and lands 0.304 normalized RMS error against an exact FP32
oracle over the same dequantized K/V, against 0.002 and 0.029 for the two bf16
routes. It clears that row's official gate only because the gate cannot
distinguish it: the row's reference magnitudes peak at 0.0525, below the 0.1
absolute bound, and the criterion fails an element only when it exceeds the
absolute AND the relative bound, so even an all-zero output passes. The
partial format is not independently switchable -- the combine kernel unpacks
E2M1 (`cvt.rn.bf16x2.e2m1x2`), so feeding it E4M3 partials misreads the bits
(10.5 RMS, or an illegal access) rather than improving precision. Registering
that row would need producer and combine reworked together.

Candidate mechanism notes, carried over from the evolution run:

(1) q-major union route with the cross-aliased-P / exp-ping-pong / partial-exp2-emulation pipeline
(bf16 K/V, GQA >= 8, <= 64 blocks); (2) scan-based kv-major producer with E2M1 partials and the
quarter-warp combine (flat FP8 K/V); (3) CSR-list kv-major v2 producer (fixed tile-chunk work items,
aux-warpgroup epilogue, cross-aliased P, split K/V pipelines) with the packed combine, launched as one
CUDA graph (paged bf16 K/V and everything else).
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
MMA_K = 16
NSLOT = 2
LOG2E = 1.4426950408889634
NEG_INF = float("-inf")

TMA_G2S_3D = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
TMA_G2S_4D = (
    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
TMA_G2S_2D = (
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
TMEM_LD_32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TMEM_ST_16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
TCGEN05_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
ID_QK = 0x08200490                                                  
ID_PV = 0x08210490                                                  
N_COLS_TMEM = 512


def ceildiv(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# KV-major v2 producer: CSR-built token lists, epilogue on the aux warpgroup,
# fixed-size tile chunks as work items, K/V fp8 conversion on the load warp.
# ---------------------------------------------------------------------------
CSR_TOKEN_BITS = 24


def make_kernel_csr(*, TOPK, HKV, B, TOTAL_Q, MAXB, CAP, PAGED, NUM_CTAS):
    """One thread per (kv head, token). Valid selections are counted in a shared-memory
    histogram first, so each CTA reserves one contiguous range per block with a single
    global atomic (no per-entry same-address atomic serialization), then scatters
    u32 = token | slot << 24 into the block's token list."""
    PAIRS = HKV * TOTAL_Q
    TOPK_V4 = TOPK % 4 == 0
    NGROUPS = B * HKV * MAXB
    CSR_THREADS = 256

    @K.kernel(warps=CSR_THREADS // 32, arch="sm_100a", grid=NUM_CTAS)
    def msa_csr_build(
        q2k: K.gptr[K.i32],
        cu_q: K.gptr[K.i32],
        kv_meta: K.gptr[K.i32],
        csr_count: K.gptr[K.i32],
        csr_list: K.gptr[K.i32],
    ):
        tid = K.thread_id()
        smem = K.smem_pool()
        hist = smem.alloc((NGROUPS,), K.i32)
        base = smem.alloc((NGROUPS,), K.i32)
        with K.serial(ceildiv(NGROUPS, CSR_THREADS), unroll=False) as r:
            gz = r * CSR_THREADS + tid
            with K.If(gz < NGROUPS), K.Then():
                K.ptx.st.shared.b32(hist.ptr_to([gz]), K.int32(0))
        K.cuda.cta_sync()
        idx = K.cta_id() * CSR_THREADS + tid
        valid_pair = idx < PAIRS
        gk = K.alloc_local([TOPK], "int32")
        rk = K.alloc_local([TOPK], "int32")
        code = K.local_scalar("int32", init=0)
        for kk in range(TOPK):
            K.assign(gk[kk], K.int32(-1))
            K.assign(rk[kk], K.int32(0))
        with K.If(valid_pair), K.Then():
            h = idx // TOTAL_Q
            t = idx % TOTAL_Q
            cuq = K.alloc_local([B + 1], "int32")
            for i in range(B + 1):
                K.ptx.ld.global_.nc.b32(cuq[i], cu_q.ptr_to([i]))
            s = K.local_scalar("int32", init=0)
            for i in range(B - 1):
                K.assign(s, s + K.Select(t >= cuq[i + 1], 1, 0))
            q_base = K.local_scalar("int32", init=0)
            q_end = K.local_scalar("int32", init=0)
            for i in range(B):
                K.assign(q_base, K.Select(s == i, cuq[i], q_base))
                K.assign(q_end, K.Select(s == i, cuq[i + 1], q_end))
            kv_len = K.local_scalar("int32")
            if PAGED:
                K.ptx.ld.global_.nc.b32(kv_len, kv_meta.ptr_to([s]))
            else:
                k0 = K.local_scalar("int32")
                k1 = K.local_scalar("int32")
                K.ptx.ld.global_.nc.b32(k0, kv_meta.ptr_to([s]))
                K.ptx.ld.global_.nc.b32(k1, kv_meta.ptr_to([s + 1]))
                K.assign(kv_len, k1 - k0)
            q_len = q_end - q_base
            t_local = t - q_base
            K.assign(code, t_local)
            nblocks = (kv_len + (BLK - 1)) // BLK
            pos = kv_len - q_len + t_local
            idxs = K.alloc_local([TOPK], "int32")
            row_base = (h * TOTAL_Q + t) * TOPK
            if TOPK_V4:
                for v in range(TOPK // 4):
                    K.ptx.ld.global_.nc.v4.b32(
                        idxs[4 * v], idxs[4 * v + 1], idxs[4 * v + 2], idxs[4 * v + 3],
                        q2k.ptr_to([row_base + 4 * v]),
                    )
            else:
                for v in range(TOPK):
                    K.ptx.ld.global_.nc.b32(idxs[v], q2k.ptr_to([row_base + v]))
            for kk in range(TOPK):
                bk = idxs[kk]
                ok = K.And(K.And(K.And(bk >= 0, bk < nblocks), bk < MAXB), bk * BLK <= pos)
                with K.If(ok), K.Then():
                    g = (s * HKV + h) * MAXB + K.max(bk, 0)
                    K.assign(gk[kk], g)
                    K.ptx.atom.shared.add.s32(rk[kk], hist.ptr_to([g]), K.int32(1))
        K.cuda.cta_sync()
        with K.serial(ceildiv(NGROUPS, CSR_THREADS), unroll=False) as r:
            gz = r * CSR_THREADS + tid
            with K.If(gz < NGROUPS), K.Then():
                cnt = K.local_scalar("int32")
                K.ptx.ld.shared.b32(cnt, hist.ptr_to([gz]))
                with K.If(cnt > 0), K.Then():
                    old = K.local_scalar("int32")
                    K.ptx.atom.relaxed.gpu.global_.add.s32(old, csr_count.ptr_to([gz]), cnt)
                    K.ptx.st.shared.b32(base.ptr_to([gz]), old)
        K.cuda.cta_sync()
        for kk in range(TOPK):
            with K.If(gk[kk] >= 0), K.Then():
                b0 = K.local_scalar("int32")
                K.ptx.ld.shared.b32(b0, base.ptr_to([gk[kk]]))
                K.ptx.st.global_.b32(
                    csr_list.ptr_to([K.Cast("int64", gk[kk]) * CAP + K.Cast("int64", b0 + rk[kk])]),
                    K.bitwise_or(code, K.shift_left(K.int32(kk), CSR_TOKEN_BITS)),
                )

    return msa_csr_build


def make_kernel_kv2(*, GQA, TOPK, HKV, B, TOTAL_Q, MAXB, CAP, PAGED, MAX_PAGES, KV_FP8,
                    NUM_CTAS, TPI, Q_STAGES, PARTIAL_FMT, SM_REGS=184, AUX_REGS=96):
    """Persistent kv-major producer. Work item = (group, chunk of TPI tiles)."""
    HQ = HKV * GQA
    assert 128 % GQA == 0
    T = 128 // GQA                         # tokens per 128-row tile
    NGROUPS = B * HKV * MAXB
    NTOK_SLOT = TPI * T                    # tokens of one work item
    Q_TILE_BYTES = 128 * HEAD_DIM * 2
    KV_TILE_BYTES = 128 * HEAD_DIM * 2
    RAW_TILE_BYTES = 128 * HEAD_DIM
    META_N = 16
    NSLOT = 3                              # metadata ring: the aux publishes two items ahead
    ROWS_TOTAL = TOTAL_Q * HQ
    NGROUPS_PER_LANE = ceildiv(NGROUPS, 32)
    assert PARTIAL_FMT in ("e2m1", "e4m3")
    assert 512 - 2 * SM_REGS - AUX_REGS >= 40
    WG3_REGS = 512 - 2 * SM_REGS - AUX_REGS
    Q_GATHER_WARPS = 2 if KV_FP8 else 3    # fp8: TMA boxes from w14/15; bf16: cp.async from w13-15
    Q_GATHER_THREADS = 32 * Q_GATHER_WARPS
    V_STAGES = 2 if KV_FP8 else 1          # bf16: single V stage (loaded by the MMA warp after the
                                           # item's first QKs) buys a third Q stage within SMEM
    PINGPONG = int(os.environ.get("MSA_PINGPONG", "0"))

    @K.kernel(warps=16, arch="sm_100a", min_blocks_per_sm=1, grid=NUM_CTAS)
    def msa_kvmajor2(
        q_g: K.gptr[K.bf16],
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        cu_q: K.gptr[K.i32],
        kv_meta: K.gptr[K.i32],
        page_table: K.gptr[K.i32],
        csr_count: K.gptr[K.i32],
        csr_list: K.gptr[K.i32],
        o_part: K.gptr[K.bf16],
        lse_part: K.gptr[K.f32],
        sched: K.gptr[K.i32],
        scale_log2: K.f32,
    ):
        cta = K.cta_id()
        warp_cta = K.warp_id()
        wg_id = warp_cta >> 2
        warp_in_wg = warp_cta & 3
        tid_in_wg = K.thread_id() & 127
        tid = K.thread_id()
        lane = K.lane_id()

        smem = K.smem_pool()
        q_smem = smem.alloc((Q_STAGES, 128, HEAD_DIM), K.bf16, swizzle=K.SW128B)
        k_smem = smem.alloc((2, 128, HEAD_DIM), K.bf16, swizzle=K.SW128B)
        v_smem = smem.alloc((V_STAGES, 128, HEAD_DIM), K.bf16, swizzle=K.SW128B)
        if KV_FP8:
            raw_smem = smem.alloc((128, HEAD_DIM), K.u8, swizzle=K.SW128B)
        meta = smem.alloc((NSLOT * META_N,), K.i32)
        tok_smem = smem.alloc((NSLOT * NTOK_SLOT,), K.i32)
        msum_smem = smem.alloc((4 * 2 * 128,), K.f32)      # [slot=stage+2*par][m|sum][row]
        prefix_smem = smem.alloc((NGROUPS + 1,), K.i32)
        wscratch = smem.alloc((16,), K.i32)
        tmem_addr = smem.alloc((1,), K.u32)

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

        # ---- barriers
        list_full = K.MBarrier(smem, NSLOT)
        list_full.init(128)
        list_free = K.MBarrier(smem, NSLOT)
        list_free.init(256 + 32 + 32 + 64)
        if KV_FP8:
            q_load = K.Pipeline(smem, Q_STAGES, full="tma", empty="tcgen05", empty_phase_offset=1)
            k_load = K.Pipeline(smem, 2, full="mbar", empty="tcgen05", init_full=128, empty_phase_offset=1)
            v_load = K.Pipeline(smem, V_STAGES, full="mbar", empty="tcgen05", init_full=128, empty_phase_offset=1)
            raw_load = K.Pipeline(smem, 1, full="tma", empty="mbar", init_empty=128, empty_phase_offset=1)
        else:
            q_load = K.Pipeline(
                smem, Q_STAGES, full="mbar", empty="tcgen05", init_full=Q_GATHER_THREADS,
                empty_phase_offset=1,
            )
            k_load = K.Pipeline(smem, 2, full="tma", empty="tcgen05", empty_phase_offset=1)
            v_load = K.Pipeline(smem, V_STAGES, full="tma", empty="tcgen05", empty_phase_offset=1)
        s_full = K.TCGen05Bar(smem, 2)
        s_full.init(1)
        p_full = K.MBarrier(smem, 2)
        p_full.init(128)
        # cross-aliased P: P(x) lives in the upper 64 columns of the OTHER stage's S region.
        # s_consumed[w]: softmax w has read S into registers (lets the MMA issue QK two tiles ahead);
        # pv_done[w]: PV of softmax w's previous tile finished reading its P (region reuse).
        s_consumed = K.MBarrier(smem, 2)
        s_consumed.init(128)
        pv_done = K.TCGen05Bar(smem, 2)
        pv_done.init(1)
        # exp ping-pong: the two softmax warpgroups alternate their MUFU-heavy exponential
        # phases so that neither runs at half MUFU rate; WG1 arrives turn[0] once at start.
        xu_turn = K.MBarrier(smem, 2)
        xu_turn.init(128)
        o_full = K.TCGen05Bar(smem, 2)
        o_full.init(1)
        o_empty = K.MBarrier(smem, 2)
        o_empty.init(128)
        # four slots (tile & 3): the softmax may run two tiles ahead of the aux on one stage,
        # so a two-slot barrier could complete twice before the aux waits (parity aliasing -> deadlock)
        msum_full = K.MBarrier(smem, 4)
        msum_full.init(128)

        K.ptx.fence.proxy.async_.shared__cta()
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        # ---- helpers
        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def tmem(col):
            return K.cuda.get_tmem_addr(K.uint32(0), 0, col)

        def tmem_load32(dst, dst_offset, tmem_col):
            K.ptx[TMEM_LD_32](*(dst[dst_offset + i] for i in range(32)), tmem_col)

        def tmem_store16(src, src_offset, tmem_col):
            K.ptx[TMEM_ST_16](tmem_col, *(src[src_offset + i] for i in range(16)))

        def ld_shared_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.shared.b32(value, ptr)
            return value

        def ld_shared_f32(ptr):
            value = K.local_scalar("float32")
            K.ptx.ld.shared.f32(value, ptr)
            return value

        def ld_global_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.global_.nc.b32(value, ptr)
            return value

        def ld_global_cg_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.global_.cg.b32(value, ptr)
            return value

        def iket_range(name, *, leader_only=False):
            token = K.alloc_local([1], "uint32")
            if leader_only:
                K.assign(token[0], K.cuda.iket.sentinel_token(name))
                with K.If(warp_in_wg == 0), K.Then():
                    K.assign(token[0], K.cuda.iket.range_start(name))
            else:
                K.assign(token[0], K.cuda.iket.range_start(name))
            return token

        def iket_end(token):
            K.cuda.iket.range_end(token[0])

        def meta_ptr(slot, idx):
            return meta.ptr_to([slot * META_N + idx])

        def ring_advance(slot, use):
            K.assign(slot, slot + 1)
            with K.If(slot == NSLOT), K.Then():
                K.assign(slot, 0)
                K.assign(use, use + 1)

        def read_meta(slot):
            return [ld_shared_i32(meta_ptr(slot, i)) for i in range(12)]

        def seq_lengths(s):
            q0 = ld_global_i32(cu_q.ptr_to([s]))
            q1 = ld_global_i32(cu_q.ptr_to([s + 1]))
            if PAGED:
                kv_len = ld_global_i32(kv_meta.ptr_to([s]))
                k_base = K.int32(0)
            else:
                k0 = ld_global_i32(kv_meta.ptr_to([s]))
                k1 = ld_global_i32(kv_meta.ptr_to([s + 1]))
                kv_len = k1 - k0
                k_base = k0
            return q0, q1 - q0, kv_len, k_base

        def cast_f32x2_bf16x2(dst_u32, src, offset):
            K.ptx.cvt.rn.bf16x2.f32(dst_u32[offset // 2], src[offset + 1], src[offset])

        # ---- role layout
        sp = K.specialize(chain_dispatch=True)
        r_sm0 = sp.role("softmax0", warps=[0, 1, 2, 3], regs=SM_REGS)
        r_sm1 = sp.role("softmax1", warps=[4, 5, 6, 7], regs=SM_REGS)
        r_aux = sp.role("aux", warps=[8, 9, 10, 11], regs=AUX_REGS)
        wg3 = sp.warpgroup("wg3", warps=range(12, 16), regs=WG3_REGS)
        r_mma = sp.role("mma", warps=[12], group=wg3)
        if KV_FP8:
            r_load = sp.role("load", warps=[13], group=wg3)
            r_qload = sp.role("qload", warps=[14, 15], group=wg3)
        else:
            r_qload = sp.role("qload", warps=[13, 14, 15], group=wg3)

        with K.If(warp_cta == 12), K.Then():
            K.ptx[TMEM_ALLOC](K.address_of(tmem_addr[0]), K.uint32(N_COLS_TMEM))
            K.ptx[TMEM_RELINQUISH]()
            K.cuda.warp_sync()

        # ---- prologue: exclusive prefix of per-group chunk counts (warp 0)
        with K.If(warp_cta == 0), K.Then():
            cnts = K.alloc_local([NGROUPS_PER_LANE], "int32")
            for j in range(NGROUPS_PER_LANE):
                gidx = lane * NGROUPS_PER_LANE + j
                K.assign(cnts[j], K.int32(0))
                with K.If(gidx < NGROUPS), K.Then():
                    n_tok_g = ld_global_cg_i32(csr_count.ptr_to([gidx]))
                    n_tiles_g = (n_tok_g + (T - 1)) // T
                    K.assign(cnts[j], (n_tiles_g + (TPI - 1)) // TPI)
            total = K.alloc_local([1], "int32")
            K.assign(total[0], K.int32(0))
            for j in range(NGROUPS_PER_LANE):
                K.assign(total[0], total[0] + cnts[j])
            K.idioms.warp_scan_add(total, 1, lane)
            run = K.local_scalar("int32", init=total[0])
            for j in range(NGROUPS_PER_LANE):
                K.assign(run, run - cnts[NGROUPS_PER_LANE - 1 - j])
            # run is now the exclusive prefix of this lane's first group
            for j in range(NGROUPS_PER_LANE):
                gidx = lane * NGROUPS_PER_LANE + j
                with K.If(gidx < NGROUPS), K.Then():
                    K.ptx.st.shared.b32(prefix_smem.ptr_to([gidx]), run)
                K.assign(run, run + cnts[j])
            with K.If(lane == 31), K.Then():
                K.ptx.st.shared.b32(prefix_smem.ptr_to([NGROUPS]), total[0])
        K.cuda.cta_sync()
        with K.If(K.thread_id() == 0), K.Then():
            allocated = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(allocated, tmem_addr.ptr_to([0]))
            K.cuda.trap_when_assert_failed(allocated == K.uint32(0))

        def role_tail(kind):
            K.cuda.cta_sync()
            if kind == "mma":
                dealloc = K.local_scalar("uint32")
                K.ptx.ld.shared.u32(dealloc, tmem_addr.ptr_to([0]))
                K.ptx[TMEM_DEALLOC](dealloc, K.uint32(N_COLS_TMEM))
            if kind == "aux":
                # last CTA resets the scheduler and the CSR counters for the next call
                with K.If(tid_in_wg == 0), K.Then():
                    done = K.local_scalar("int32")
                    K.ptx.atom.acq_rel.gpu.global_.add.s32(done, sched.ptr_to([1]), K.int32(1))
                    K.ptx.st.shared.b32(wscratch.ptr_to([9]), done)
                K.cuda.warpgroup_sync(1)
                done_v = ld_shared_i32(wscratch.ptr_to([9]))
                with K.If(done_v == NUM_CTAS - 1), K.Then():
                    with K.serial(ceildiv(NGROUPS, 128), unroll=False) as r:
                        gz = r * 128 + tid_in_wg
                        with K.If(gz < NGROUPS), K.Then():
                            K.ptx.st.relaxed.gpu.global_.b32(csr_count.ptr_to([gz]), K.int32(0))
                    with K.If(tid_in_wg == 0), K.Then():
                        K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([0]), K.int32(0))
                        K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([1]), K.int32(0))

        # =====================================================================
        # AUX: work-item scheduler + meta/token publication + per-tile epilogue
        # =====================================================================
        with r_aux:
            slot = K.local_scalar("int32", init=0)               # publish ring position
            use_a = K.local_scalar("int32", init=0)
            cur_slot = K.local_scalar("int32", init=0)           # slot of the item being epilogued
            g_e = K.local_scalar("int32", init=0)                # tiles epilogued so far
            total_items = ld_shared_i32(prefix_smem.ptr_to([NGROUPS]))
            running_a = K.local_scalar("int32", init=1)
            published_real = K.local_scalar("int32", init=0)    # last publish_next produced a real item
            pub_count = K.local_scalar("int32", init=0)         # real items published so far
            pending_cvt = K.local_scalar("int32", init=0)       # fp8: raw tiles of the next item still to convert
            cvt_item = K.local_scalar("int32", init=0)          # fp8: index of the item being converted
            raw_it_a = K.local_scalar("int32", init=0)

            def convert_step(which):
                """fp8: convert one raw 128x128 e4m3 tile (K if which == 0 else V) of item
                `cvt_item` into its bf16 stage, then publish it to the MMA."""
                dst_tile = k_smem if which == 0 else v_smem
                pipe_ = k_load if which == 0 else v_load
                kvslot = cvt_item & 1
                tk_cv = iket_range("aux-convert", leader_only=True)
                raw_load.full.wait(0, raw_it_a & 1)
                K.assign(raw_it_a, raw_it_a + 1)
                K.ptx.fence.proxy.async_.shared__cta()
                for rep_ in range(8):
                    idx = rep_ * 128 + tid_in_wg
                    row = idx >> 3
                    chunk = idx & 7
                    words = K.alloc_local([4], "uint32")
                    K.ptx.ld.shared.v4.b32(
                        words[0], words[1], words[2], words[3], raw_smem.ptr_to(row, chunk * 16),
                    )
                    outw = K.alloc_local([8], "uint32")
                    for w4 in range(4):
                        lo16 = K.local_scalar("uint16")
                        hi16 = K.local_scalar("uint16")
                        K.ptx.mov.b32(lo16, hi16, words[w4])
                        K.ptx.cvt.rn.bf16x2.e4m3x2(outw[2 * w4], lo16)
                        K.ptx.cvt.rn.bf16x2.e4m3x2(outw[2 * w4 + 1], hi16)
                    K.ptx.st.shared.v4.b32(
                        dst_tile[kvslot].ptr_to(row, chunk * 16), outw[0], outw[1], outw[2], outw[3],
                    )
                    K.ptx.st.shared.v4.b32(
                        dst_tile[kvslot].ptr_to(row, chunk * 16 + 8), outw[4], outw[5], outw[6], outw[7],
                    )
                K.ptx.fence.proxy.async_.shared__cta()
                raw_load.empty.arrive(0)
                pipe_.full.arrive(kvslot)
                iket_end(tk_cv)

            def publish_next():
                """Grab one work item and publish its meta + token slice (or the terminator)."""
                with K.If(tid_in_wg == 0), K.Then():
                    grabbed = K.local_scalar("int32")
                    K.ptx.atom.relaxed.gpu.global_.add.s32(grabbed, sched.ptr_to([0]), K.int32(1))
                    K.ptx.st.shared.b32(wscratch.ptr_to([8]), grabbed)
                K.cuda.warpgroup_sync(1)
                item = ld_shared_i32(wscratch.ptr_to([8]))
                tk_free = iket_range("aux-wait-free", leader_only=True)
                list_free.wait(slot, (use_a + 1) & 1)
                iket_end(tk_free)
                K.assign(published_real, K.Select(item >= total_items, 0, 1))
                with K.If(item >= total_items):
                    with K.Then():
                        with K.If(tid_in_wg == 0), K.Then():
                            K.ptx.st.shared.b32(meta_ptr(slot, 0), K.int32(-1))
                            for mi in range(1, 12):
                                K.ptx.st.shared.b32(meta_ptr(slot, mi), K.int32(0))
                    with K.Else():
                        K.assign(pub_count, pub_count + 1)
                        # binary search: largest g with prefix[g] <= item
                        lo = K.local_scalar("int32", init=0)
                        hi = K.local_scalar("int32", init=NGROUPS)       # prefix[hi] > item
                        with K.While(hi - lo > 1):
                            mid = (lo + hi) >> 1
                            pm = ld_shared_i32(prefix_smem.ptr_to([mid]))
                            with K.If(pm <= item):
                                with K.Then():
                                    K.assign(lo, mid)
                                with K.Else():
                                    K.assign(hi, mid)
                        g = lo
                        c = item - ld_shared_i32(prefix_smem.ptr_to([g]))
                        b = g % MAXB
                        sh = g // MAXB
                        h = sh % HKV
                        s = sh // HKV
                        q_base, q_len, kv_len, k_base = seq_lengths(s)
                        n_tok = ld_global_cg_i32(csr_count.ptr_to([g]))
                        n_tiles_all = (n_tok + (T - 1)) // T
                        tile0 = c * TPI
                        n_tiles = K.local_scalar("int32", init=K.min(TPI, n_tiles_all - tile0))
                        # diagnostic: an impossible decode must never look like the terminator
                        with K.If(n_tiles <= 0), K.Then():
                            with K.If(tid_in_wg == 0), K.Then():
                                dbg = K.local_scalar("int32")
                                K.ptx.atom.relaxed.gpu.global_.add.s32(dbg, sched.ptr_to([2]), K.int32(1))
                                K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([3]), item)
                            K.assign(n_tiles, K.int32(0))
                        # token slice of this item into shared memory
                        list_base = K.Cast("int64", g) * CAP
                        with K.serial(ceildiv(NTOK_SLOT, 128), unroll=False) as r:
                            j = r * 128 + tid_in_wg
                            with K.If(j < NTOK_SLOT), K.Then():
                                li = K.min(tile0 * T + j, K.max(n_tok - 1, 0))
                                tokw = ld_global_cg_i32(csr_list.ptr_to([list_base + K.Cast("int64", li)]))
                                K.ptx.st.shared.b32(tok_smem.ptr_to([slot * NTOK_SLOT + j]), tokw)
                        with K.If(tid_in_wg == 0), K.Then():
                            K.ptx.st.shared.b32(meta_ptr(slot, 0), n_tiles)
                            K.ptx.st.shared.b32(meta_ptr(slot, 1), n_tok)
                            K.ptx.st.shared.b32(meta_ptr(slot, 2), b)
                            K.ptx.st.shared.b32(meta_ptr(slot, 3), h)
                            K.ptx.st.shared.b32(meta_ptr(slot, 4), s)
                            K.ptx.st.shared.b32(meta_ptr(slot, 5), q_base)
                            K.ptx.st.shared.b32(meta_ptr(slot, 6), kv_len)
                            K.ptx.st.shared.b32(meta_ptr(slot, 7), q_len)
                            K.ptx.st.shared.b32(meta_ptr(slot, 8), tile0)
                            if PAGED:
                                page = ld_global_i32(page_table.ptr_to([s * MAX_PAGES + b]))
                                K.ptx.st.shared.b32(meta_ptr(slot, 9), page)
                            else:
                                K.ptx.st.shared.b32(meta_ptr(slot, 9), k_base + b * BLK)
                            K.ptx.st.shared.b32(meta_ptr(slot, 10), kv_len - q_len)
                            K.ptx.st.shared.b32(meta_ptr(slot, 11), K.int32(0))
                K.cuda.warpgroup_sync(1)
                list_full.arrive(slot)
                ring_advance(slot, use_a)

            def epilogue_item(m):
                n_tiles, n_tok, b, h, s, q_base, kv_len, q_len, tile0 = (
                    m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]
                )
                row = tid_in_wg
                j_tok = row // GQA
                head = row % GQA
                with K.serial(n_tiles, unroll=False) as i:
                    stage = g_e & 1
                    par = (g_e >> 1) & 1
                    tok_valid = (tile0 + i) * T + j_tok < n_tok
                    tokw = ld_shared_i32(tok_smem.ptr_to([cur_slot * NTOK_SLOT + i * T + j_tok]))
                    t_local = K.bitwise_and(tokw, K.int32((1 << CSR_TOKEN_BITS) - 1))
                    sel = K.shift_right(tokw, CSR_TOKEN_BITS)
                    tk_wo = iket_range("ep-wait-o", leader_only=True)
                    ms_slot = stage + 2 * par
                    msum_full.wait(ms_slot, (g_e >> 2) & 1)
                    row_max = ld_shared_f32(msum_smem.ptr_to([(ms_slot * 2) * 128 + row]))
                    row_sum = ld_shared_f32(msum_smem.ptr_to([(ms_slot * 2 + 1) * 128 + row]))
                    o_full.wait(stage, par)
                    iket_end(tk_wo)
                    tk_ep = iket_range("ep-store", leader_only=True)
                    inv = K.local_scalar("float32", init=K.float32(0.0))
                    with K.If(row_sum > K.float32(0.0)), K.Then():
                        K.ptx.rcp.approx.ftz.f32(inv, row_sum)
                    if PARTIAL_FMT == "e2m1":
                        K.assign(inv, inv * K.float32(16.0))
                    grow = (q_base + t_local) * HQ + h * GQA + head
                    prow = K.Cast("int64", grow * TOPK + sel)
                    with K.If(tok_valid), K.Then():
                        lg = K.local_scalar("float32")
                        K.ptx.lg2.approx.ftz.f32(lg, K.max(row_sum, K.float32(1e-30)))
                        lse2 = row_max * scale_log2 + lg
                        K.ptx.st.global_.f32(lse_part.ptr_to([prow]), lse2)
                    NQ = 4                                   # 32-column quarters
                    QW = 32 // 8 if PARTIAL_FMT == "e2m1" else 32 // 4   # packed words per quarter
                    packed = K.alloc_local([NQ * QW], "uint32")
                    K.ptx.tcgen05.fence__after_thread_sync()
                    for qi in range(NQ):
                        o_q = K.alloc_local([32], "float32")
                        tmem_load32(o_q, 0, tmem(256 + stage * 128 + qi * 32))
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                        if qi == NQ - 1:
                            K.ptx.tcgen05.fence__before_thread_sync()
                            o_empty.arrive(stage)
                        for d in range(32):
                            K.assign(o_q[d], o_q[d] * inv)
                        if PARTIAL_FMT == "e2m1":
                            for d8 in range(4):
                                pair_bytes = K.alloc_local([4], "uint8")
                                for pair in range(4):
                                    d = 8 * d8 + 2 * pair
                                    K.ptx.cvt.rn.satfinite.e2m1x2.f32(pair_bytes[pair], o_q[d + 1], o_q[d])
                                lo16 = K.local_scalar(
                                    "uint16",
                                    init=K.bitwise_or(
                                        K.Cast("uint16", pair_bytes[0]),
                                        K.shift_left(K.Cast("uint16", pair_bytes[1]), K.uint16(8)),
                                    ),
                                )
                                hi16 = K.local_scalar(
                                    "uint16",
                                    init=K.bitwise_or(
                                        K.Cast("uint16", pair_bytes[2]),
                                        K.shift_left(K.Cast("uint16", pair_bytes[3]), K.uint16(8)),
                                    ),
                                )
                                K.ptx.mov.b32(packed[qi * QW + d8], lo16, hi16)
                        else:
                            for d4 in range(8):
                                lo16 = K.local_scalar("uint16")
                                hi16 = K.local_scalar("uint16")
                                K.ptx.cvt.rn.satfinite.e4m3x2.f32(lo16, o_q[4 * d4 + 1], o_q[4 * d4])
                                K.ptx.cvt.rn.satfinite.e4m3x2.f32(hi16, o_q[4 * d4 + 3], o_q[4 * d4 + 2])
                                K.assign(
                                    packed[qi * QW + d4],
                                    K.bitwise_or(
                                        K.Cast("uint32", lo16),
                                        K.shift_left(K.Cast("uint32", hi16), K.uint32(16)),
                                    ),
                                )
                    with K.If(tok_valid), K.Then():
                        if PARTIAL_FMT == "e2m1":
                            obase = prow * (HEAD_DIM // 4)
                            for v in range(HEAD_DIM // 64):
                                K.ptx.st.global_.v8.b32(
                                    o_part.ptr_to([obase + v * 16]),
                                    *(packed[8 * v + q8] for q8 in range(8)),
                                )
                        else:
                            obase = prow * (HEAD_DIM // 2)
                            for v in range(HEAD_DIM // 32):
                                K.ptx.st.global_.v8.b32(
                                    o_part.ptr_to([obase + v * 16]),
                                    *(packed[8 * v + q8] for q8 in range(8)),
                                )
                    iket_end(tk_ep)
                    K.assign(g_e, g_e + 1)
                    if KV_FP8:
                        # one conversion step per tile: V (pending == 1) is checked first so that
                        # K (pending == 2) and V land in consecutive tiles
                        with K.If(pending_cvt == 1), K.Then():
                            convert_step(1)
                            K.assign(pending_cvt, 0)
                        with K.If(pending_cvt == 2), K.Then():
                            convert_step(0)
                            K.assign(pending_cvt, 1)

            # publish the first item (and, for fp8, convert its K/V before anything can start),
            # then alternate: publish next / epilogue current (converting the next item's K/V
            # between epilogues)
            publish_next()                                       # item 0
            first_real = K.local_scalar("int32", init=published_real)
            publish_next()                                       # item 1
            if KV_FP8:
                with K.If(first_real != 0), K.Then():           # item 0's K/V before anything can start
                    convert_step(0)
                    convert_step(1)
                    K.assign(cvt_item, 1)
            with K.While(running_a != 0):
                m_cur = read_meta(cur_slot)
                with K.If(m_cur[0] < 0):
                    with K.Then():
                        K.assign(running_a, 0)
                    with K.Else():
                        publish_next()                           # item i+2
                        if KV_FP8:
                            nxt_slot = K.local_scalar("int32", init=cur_slot + 1)
                            with K.If(nxt_slot == NSLOT), K.Then():
                                K.assign(nxt_slot, 0)
                            nxt_real = ld_shared_i32(meta_ptr(nxt_slot, 0)) >= 0
                            K.assign(pending_cvt, K.Select(nxt_real, 2, 0))
                        epilogue_item(m_cur)
                        if KV_FP8:
                            # drain: the next item's conversions not covered by this item's tiles
                            with K.If(pending_cvt == 2), K.Then():
                                convert_step(0)
                                K.assign(pending_cvt, 1)
                            with K.If(pending_cvt == 1), K.Then():
                                convert_step(1)
                                K.assign(pending_cvt, 0)
                            with K.If(nxt_real), K.Then():
                                K.assign(cvt_item, cvt_item + 1)
                        K.assign(cur_slot, cur_slot + 1)
                        with K.If(cur_slot == NSLOT), K.Then():
                            K.assign(cur_slot, 0)
            role_tail("aux")

        # =====================================================================
        # WG3: MMA issue, K/V load (+fp8 conversion), Q gather
        # =====================================================================
        with wg3:
            with r_mma:
                slot = K.local_scalar("int32", init=0)
                use_m = K.local_scalar("int32", init=0)
                running_m = K.local_scalar("int32", init=1)
                g_m = K.local_scalar("int32", init=0)
                k_pipe_m = K.PipelineState(2, phase=0)
                v_pipe_m = K.PipelineState(V_STAGES, phase=0)
                q_pipe_m = K.PipelineState(Q_STAGES, phase=0)
                tb_raw = K.local_scalar("uint32")
                K.ptx.ld.shared.u32(tb_raw, tmem_addr.ptr_to([0]))
                tmem_base = K.local_scalar("uint32", init=K.uniform(tb_raw))
                q_desc, qoff = encode(q_smem[0])
                k_desc, koff = encode(k_smem[0])
                v_desc, mnoff = encode(v_smem[0], major="mn")

                def gemm_qk(q_stage, kv_stage, stage):
                    for ki in range(HEAD_DIM // MMA_K):
                        K.ptx[MMA_F16](
                            tmem_base + K.uint32(stage * 128),
                            desc_at(q_desc, q_stage * Q_STAGE16 + qoff(ki)),
                            desc_at(k_desc, kv_stage * KV_STAGE16 + koff(ki)),
                            K.uint32(ID_QK), K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0), ki != 0,
                        )

                def gemm_pv(kv_stage, stage):
                    for ki in range(BLK // MMA_K):
                        K.ptx[MMA_F16](
                            tmem_base + K.uint32(256 + stage * 128),
                            tmem_base + K.uint32((1 - stage) * 128 + 64 + ki * (MMA_K // 2)),
                            desc_at(v_desc, kv_stage * KV_STAGE16 + mnoff(ki)),
                            K.uint32(ID_PV), K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0), ki != 0,
                        )

                with K.While(running_m != 0):
                    tk_ml = iket_range("mma-wait-list")
                    list_full.wait(slot, use_m & 1)
                    iket_end(tk_ml)
                    n_tiles = ld_shared_i32(meta_ptr(slot, 0))
                    with K.If(n_tiles < 0), K.Then():
                        K.assign(running_m, 0)
                    with K.If(n_tiles > 0), K.Then():
                        ks = K.local_scalar("int32", init=k_pipe_m.stage)
                        vs = K.local_scalar("int32", init=v_pipe_m.stage)
                        tk_wkv = iket_range("mma-wait-kv")
                        k_load.full.wait(ks, k_pipe_m.phase)
                        iket_end(tk_wkv)
                        K.ptx.tcgen05.fence__after_thread_sync()

                        def issue_qk(stage_expr, is_last):
                            qs = K.local_scalar("int32", init=q_pipe_m.stage)
                            tk_wq = iket_range("mma-wait-q")
                            q_load.full.wait(qs, q_pipe_m.phase)
                            iket_end(tk_wq)
                            K.ptx.fence.proxy.async_.shared__cta()
                            K.ptx.tcgen05.fence__after_thread_sync()
                            tk_iq = iket_range("mma-issue-qk")
                            with K.If(elected()), K.Then():
                                gemm_qk(qs, ks, stage_expr)
                                s_full.arrive(stage_expr)
                                q_load.empty.arrive(qs)
                                with K.If(is_last), K.Then():
                                    k_load.empty.arrive(ks)      # K of this item fully consumed
                            iket_end(tk_iq)
                            q_pipe_m.advance()

                        # the first two QKs of an item: their S stages were consumed during the
                        # previous item (implied by the p_full waits already performed)
                        issue_qk(g_m & 1, n_tiles == 1)
                        with K.If(n_tiles > 1), K.Then():
                            issue_qk((g_m + 1) & 1, n_tiles == 2)
                        if not KV_FP8:
                            # bf16: single V stage, issued here so the load lands under QK/softmax
                            m_h = ld_shared_i32(meta_ptr(slot, 3))
                            m_kv = ld_shared_i32(meta_ptr(slot, 9))
                            tk_wv = iket_range("mma-wait-vempty")
                            v_load.empty.wait(vs, v_pipe_m.phase)
                            iket_end(tk_wv)
                            with K.If(elected()), K.Then():
                                if PAGED:
                                    K.ptx[TMA_G2S_3D](
                                        v_smem[vs].ptr_to(0, 0), K.address_of(v_map), K.int32(0),
                                        (m_kv * HKV + m_h) * BLK, K.int32(0),
                                        K.cuda.cvta_generic_to_shared(v_load.full.ptr_to([vs])),
                                    )
                                else:
                                    K.ptx[TMA_G2S_3D](
                                        v_smem[vs].ptr_to(0, 0), K.address_of(v_map), K.int32(0),
                                        m_kv, m_h * 2,
                                        K.cuda.cvta_generic_to_shared(v_load.full.ptr_to([vs])),
                                    )
                                v_load.full.arrive(vs, tx_count=KV_TILE_BYTES)
                        with K.serial(n_tiles, unroll=False) as i:
                            stage = g_m & 1
                            par = (g_m >> 1) & 1
                            with K.If(i + 2 < n_tiles), K.Then():
                                # QK(x+2) reuses S stage `stage`: wait until softmax read S(x)
                                tk_wc = iket_range("mma-wait-consumed")
                                s_consumed.wait(stage, par)
                                iket_end(tk_wc)
                                issue_qk(stage, i + 3 == n_tiles)
                            with K.If(i == 0), K.Then():
                                tk_wvf = iket_range("mma-wait-v")
                                v_load.full.wait(vs, v_pipe_m.phase)
                                iket_end(tk_wvf)
                                K.ptx.tcgen05.fence__after_thread_sync()
                            tk_wp = iket_range("mma-wait-p")
                            p_full.wait(stage, par)
                            o_empty.wait(stage, par ^ 1)
                            iket_end(tk_wp)
                            K.ptx.tcgen05.fence__after_thread_sync()
                            tk_ip = iket_range("mma-issue-pv")
                            with K.If(elected()), K.Then():
                                gemm_pv(vs, stage)
                                o_full.arrive(stage)
                                pv_done.arrive(stage)
                            iket_end(tk_ip)
                            K.assign(g_m, g_m + 1)
                        with K.If(elected()), K.Then():
                            v_load.empty.arrive(vs)
                        k_pipe_m.advance()
                        v_pipe_m.advance()
                    list_free.arrive(slot)
                    ring_advance(slot, use_m)
                role_tail("mma")

            def k_issue_bf16(k_stage, kv_coord, h):
                """One elected lane: TMA K of the item's block into k stage."""
                if PAGED:
                    K.ptx[TMA_G2S_3D](
                        k_smem[k_stage].ptr_to(0, 0), K.address_of(k_map), K.int32(0),
                        (kv_coord * HKV + h) * BLK, K.int32(0),
                        K.cuda.cvta_generic_to_shared(k_load.full.ptr_to([k_stage])),
                    )
                else:
                    K.ptx[TMA_G2S_3D](
                        k_smem[k_stage].ptr_to(0, 0), K.address_of(k_map), K.int32(0),
                        kv_coord, h * 2,
                        K.cuda.cvta_generic_to_shared(k_load.full.ptr_to([k_stage])),
                    )
                k_load.full.arrive(k_stage, tx_count=KV_TILE_BYTES)

            if KV_FP8:
                with r_load:
                    slot = K.local_scalar("int32", init=0)
                    use_l = K.local_scalar("int32", init=0)
                    running_l = K.local_scalar("int32", init=1)
                    k_pipe = K.PipelineState(2, phase=0)
                    v_pipe = K.PipelineState(V_STAGES, phase=0)
                    raw_it = K.local_scalar("int32", init=0)
                    with K.While(running_l != 0):
                        tk_wl = iket_range("ld-wait-list")
                        list_full.wait(slot, use_l & 1)
                        iket_end(tk_wl)
                        m = read_meta(slot)
                        n_tiles, h, kv_coord = m[0], m[3], m[9]
                        with K.If(n_tiles < 0), K.Then():
                            K.assign(running_l, 0)
                        with K.If(n_tiles > 0), K.Then():
                            tk_kv = iket_range("ld-kv")
                            for which, tmap, pipe_, pipe_state in (
                                (0, k_map, k_load, k_pipe), (1, v_map, v_load, v_pipe)
                            ):
                                # stage free (MMA done with it) and raw buffer free (aux converted it)
                                pipe_.empty.wait(pipe_state.stage, pipe_state.phase)
                                raw_load.empty.wait(0, raw_it & 1)
                                with K.If(elected()), K.Then():
                                    if PAGED:
                                        K.ptx[TMA_G2S_2D](
                                            raw_smem.ptr_to(0, 0), K.address_of(tmap), K.int32(0),
                                            (kv_coord * HKV + h) * BLK,
                                            K.cuda.cvta_generic_to_shared(raw_load.full.ptr_to([0])),
                                        )
                                    else:
                                        K.ptx[TMA_G2S_3D](
                                            raw_smem.ptr_to(0, 0), K.address_of(tmap), K.int32(0),
                                            kv_coord, h,
                                            K.cuda.cvta_generic_to_shared(raw_load.full.ptr_to([0])),
                                        )
                                    raw_load.full.arrive(0, tx_count=RAW_TILE_BYTES)
                                K.assign(raw_it, raw_it + 1)
                                pipe_state.advance()
                            iket_end(tk_kv)
                        list_free.arrive(slot)
                        ring_advance(slot, use_l)
                    role_tail("load")

            with r_qload:
                slot = K.local_scalar("int32", init=0)
                use_q = K.local_scalar("int32", init=0)
                running_q = K.local_scalar("int32", init=1)
                q_pipe_l = K.PipelineState(Q_STAGES, phase=0)
                k_pipe_q = K.PipelineState(2, phase=0)
                wq = warp_cta - (16 - Q_GATHER_WARPS)              # 0..Q_GATHER_WARPS-1
                tq = wq * 32 + lane                                # thread index among gather threads
                with K.While(running_q != 0):
                    tk_qw = iket_range("ql-wait-list")
                    list_full.wait(slot, use_q & 1)
                    iket_end(tk_qw)
                    m = read_meta(slot)
                    n_tiles, n_tok, b, h, s, q_base, kv_len, q_len, tile0, kv_coord = (
                        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9]
                    )
                    with K.If(n_tiles < 0), K.Then():
                        K.assign(running_q, 0)
                    with K.If(n_tiles > 0), K.Then():
                        if not KV_FP8:
                            # K of this item: one elected lane of warp 13 issues the TMA
                            with K.If(wq == 0), K.Then():
                                tk_kv = iket_range("ld-kv")
                                k_load.empty.wait(k_pipe_q.stage, k_pipe_q.phase)
                                with K.If(elected()), K.Then():
                                    k_issue_bf16(k_pipe_q.stage, kv_coord, h)
                                iket_end(tk_kv)
                            k_pipe_q.advance()
                        tk_qi = iket_range("ql-issue")
                        with K.serial(n_tiles, unroll=False) as i:
                            q_load.empty.wait(q_pipe_l.stage, q_pipe_l.phase)
                            stage_now = K.local_scalar("int32", init=q_pipe_l.stage)
                            if KV_FP8:
                                with K.If(elected()), K.Then():
                                    qbase = q_smem[stage_now].ptr_to(0, 0)
                                    for j in range(T):
                                        tokw = ld_shared_i32(tok_smem.ptr_to([slot * NTOK_SLOT + i * T + j]))
                                        t_local = K.bitwise_and(tokw, K.int32((1 << CSR_TOKEN_BITS) - 1))
                                        dst = K.ptx.addr(
                                            qbase,
                                            wq * (Q_TILE_BYTES // 2) + j * GQA * (HEAD_DIM // 2) * 2,
                                        )
                                        K.ptx[TMA_G2S_4D](
                                            dst, K.address_of(q_map), K.int32(0), h * GQA,
                                            q_base + t_local, wq,
                                            K.cuda.cvta_generic_to_shared(q_load.full.ptr_to([stage_now])),
                                        )
                                with K.If(wq == 0), K.Then():
                                    with K.If(elected()), K.Then():
                                        q_load.full.arrive(stage_now, tx_count=Q_TILE_BYTES)
                            else:
                                K.ptx.fence.proxy.async_.shared__cta()
                                base_ptr = q_smem[stage_now].ptr_to(0, 0)
                                n_valid = K.min(n_tok - (tile0 + i) * T, T)
                                # 2048 16-byte chunks per tile spread over the gather threads
                                NCHUNK = 128 * (HEAD_DIM * 2 // 16)
                                for jj in range(ceildiv(NCHUNK, Q_GATHER_THREADS)):
                                    cidx = jj * Q_GATHER_THREADS + tq
                                    with K.If(cidx < NCHUNK), K.Then():
                                        row = cidx >> 4
                                        c16 = cidx & 15
                                        tidx = row // GQA
                                        tokw = ld_shared_i32(tok_smem.ptr_to([slot * NTOK_SLOT + i * T + tidx]))
                                        t_local = K.bitwise_and(tokw, K.int32((1 << CSR_TOKEN_BITS) - 1))
                                        grow = (q_base + t_local) * HQ + h * GQA + (row % GQA)
                                        src = q_g.ptr_to([K.Cast("int64", grow) * HEAD_DIM + c16 * 8])
                                        dst = K.ptx.addr(
                                            base_ptr,
                                            (c16 >> 3) * (Q_TILE_BYTES // 2) + row * 128
                                            + K.shift_left(K.bitwise_xor(c16 & 7, row & 7), 4),
                                        )
                                        # invalid tokens: zero-fill (ignore-src) so no stale Q leaks
                                        K.ptx["cp.async.cg.shared.global"](
                                            dst, src, 16, K.ptx.pred(tidx >= n_valid)
                                        )
                                K.ptx.cp.async_.mbarrier.arrive.noinc.shared.b64(
                                    q_load.full.ptr_to([stage_now])
                                )
                            q_pipe_l.advance()
                        iket_end(tk_qi)
                    list_free.arrive(slot)
                    ring_advance(slot, use_q)
                role_tail("qload")

        # =====================================================================
        # SOFTMAX warpgroups: single-block softmax, P -> TMEM, (m, sum) -> SMEM
        # =====================================================================
        def softmax_role(role, w, kind):
            with role:
                slot = K.local_scalar("int32", init=0)
                use_x = K.local_scalar("int32", init=0)
                running_x = K.local_scalar("int32", init=1)
                g0 = K.local_scalar("int32", init=0)
                xturn = K.local_scalar("int32", init=0)
                if PINGPONG and w == 1:
                    xu_turn.arrive(0)
                with K.While(running_x != 0):
                    tk_sl = iket_range("sm-wait-list", leader_only=True)
                    list_full.wait(slot, use_x & 1)
                    iket_end(tk_sl)
                    m = read_meta(slot)
                    n_tiles, n_tok, b, h, s, q_base, kv_len, q_len, tile0 = (
                        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]
                    )
                    causal_off = m[10]
                    with K.If(n_tiles < 0), K.Then():
                        K.assign(running_x, 0)
                    with K.If(n_tiles > 0), K.Then():
                        start = (w - g0) & 1
                        n_iter = K.max((n_tiles - start + 1) // 2, 0)
                        kv_valid = K.min(kv_len - b * BLK, BLK)
                        row = tid_in_wg
                        j_tok = row // GQA
                        with K.serial(n_iter, unroll=False) as ii:
                            i = start + 2 * ii
                            g = g0 + i
                            par = (g >> 1) & 1
                            tokw = ld_shared_i32(tok_smem.ptr_to([slot * NTOK_SLOT + i * T + j_tok]))
                            t_local = K.bitwise_and(tokw, K.int32((1 << CSR_TOKEN_BITS) - 1))
                            pos = causal_off + t_local
                            col_limit = K.min(kv_valid, pos - b * BLK + 1)
                            s_chunk = K.alloc_local([BLK], "float32")
                            tk_ws = iket_range("sm-wait-s", leader_only=True)
                            s_full.wait(w, par)
                            iket_end(tk_ws)
                            tk_sm = iket_range("sm-softmax", leader_only=True)
                            tk_sl = iket_range("sm-sload", leader_only=True)
                            K.ptx.tcgen05.fence__after_thread_sync()
                            for ci in range(BLK // 32):
                                tmem_load32(s_chunk, ci * 32, tmem(w * 128 + ci * 32))
                            K.ptx.tcgen05.wait__ld.sync.aligned()
                            K.ptx.tcgen05.fence__before_thread_sync()
                            s_consumed.arrive(w)
                            iket_end(tk_sl)
                            tk_mx = iket_range("sm-max", leader_only=True)
                            with K.If(col_limit < BLK), K.Then():
                                for cidx in range(BLK):
                                    K.ptx.mov.b32(
                                        s_chunk[cidx],
                                        K.Select(cidx < col_limit, s_chunk[cidx], K.float32(NEG_INF)),
                                    )
                            mx = K.alloc_local([8], "float32")
                            for ch in range(8):
                                K.ptx.mov.b32(mx[ch], K.max(s_chunk[2 * ch], s_chunk[2 * ch + 1]))
                            for grp in range(1, BLK // 16):
                                for ch in range(8):
                                    K.ptx["max.f32"](mx[ch], mx[ch], s_chunk[16 * grp + 2 * ch], s_chunk[16 * grp + 2 * ch + 1])
                            K.ptx["max.f32"](mx[0], mx[0], mx[1], mx[2])
                            K.ptx["max.f32"](mx[3], mx[3], mx[4], mx[5])
                            K.ptx["max.f32"](mx[0], mx[0], mx[6], mx[7])
                            row_max = K.local_scalar("float32")
                            K.assign(row_max, K.max(mx[0], mx[3]))
                            m_safe = K.Select(row_max == K.float32(NEG_INF), K.float32(0.0), row_max)
                            neg_bias = K.local_scalar("float32")
                            K.assign(neg_bias, K.float32(0.0) - m_safe * scale_log2)
                            scale_pair = K.local_scalar("uint64")
                            bias_pair = K.local_scalar("uint64")
                            K.ptx.mov.b64(scale_pair, scale_log2, scale_log2)
                            K.ptx.mov.b64(bias_pair, neg_bias, neg_bias)
                            sum_acc = [K.local_scalar("uint64") for _ in range(8)]
                            for acc_pair in sum_acc:
                                K.ptx.mov.b64(acc_pair, K.float32(0.0), K.float32(0.0))
                            pair_tmp = K.local_scalar("uint64")
                            # P(x) goes to the other stage's upper half. It may be stored only after
                            # (a) the other warpgroup has read the S tile currently in that stage
                            #     (S(x+1) if QK(x+1) was issued within this item, else S(x-1)), and
                            # (b) PV(x-2) has finished reading P(x-2) from the same region.
                            kx = g >> 1
                            has_next = i + 1 < n_tiles
                            if w == 0:
                                other_par = K.Select(has_next, kx & 1, (kx + 1) & 1)
                            else:
                                other_par = K.Select(has_next, (kx + 1) & 1, kx & 1)
                            iket_end(tk_mx)
                            tk_pw = iket_range("sm-wait-pstore", leader_only=True)
                            s_consumed.wait(1 - w, other_par)
                            pv_done.wait(w, (kx + 1) & 1)
                            iket_end(tk_pw)
                            if PINGPONG:
                                tk_xt = iket_range("sm-wait-turn", leader_only=True)
                                xu_turn.wait(w, xturn & 1)
                                iket_end(tk_xt)
                            tk_ex = iket_range("sm-exp", leader_only=True)
                            K.ptx.tcgen05.fence__after_thread_sync()
                            for pi in range(4):
                                p_chunk = K.alloc_local([16], "uint32")
                                for pair in range(16):
                                    cidx = pi * 32 + pair * 2
                                    K.ptx.mov.b64(pair_tmp, s_chunk[cidx], s_chunk[cidx + 1])
                                    K.ptx.fma.rz.ftz.f32x2(pair_tmp, pair_tmp, scale_pair, bias_pair)
                                    K.ptx.mov.b64(s_chunk[cidx], s_chunk[cidx + 1], pair_tmp)
                                    K.ptx.ex2.approx.ftz.f32(s_chunk[cidx], s_chunk[cidx])
                                    K.ptx.ex2.approx.ftz.f32(s_chunk[cidx + 1], s_chunk[cidx + 1])
                                    K.ptx.mov.b64(pair_tmp, s_chunk[cidx], s_chunk[cidx + 1])
                                    K.ptx.add.rn.ftz.f32x2(sum_acc[pair % 8], sum_acc[pair % 8], pair_tmp)
                                    K.ptx.cvt.rn.bf16x2.f32(p_chunk[pair], s_chunk[cidx + 1], s_chunk[cidx])
                                tmem_store16(p_chunk, 0, tmem((1 - w) * 128 + 64 + pi * 16))
                            if PINGPONG:
                                K.cuda.warp_sync()
                                xu_turn.arrive(1 - w)
                                K.assign(xturn, xturn + 1)
                            for step in (4, 2, 1):
                                for a in range(step):
                                    K.ptx.add.rn.ftz.f32x2(sum_acc[a], sum_acc[a], sum_acc[a + step])
                            sum_lo = K.local_scalar("float32")
                            sum_hi = K.local_scalar("float32")
                            K.ptx.mov.b64(sum_lo, sum_hi, sum_acc[0])
                            row_sum = K.local_scalar("float32", init=sum_lo + sum_hi)
                            ms_slot = w + 2 * par
                            K.ptx.st.shared.f32(msum_smem.ptr_to([(ms_slot * 2) * 128 + row]), m_safe)
                            K.ptx.st.shared.f32(msum_smem.ptr_to([(ms_slot * 2 + 1) * 128 + row]), row_sum)
                            K.ptx.tcgen05.wait__st.sync.aligned()
                            K.ptx.tcgen05.fence__before_thread_sync()
                            p_full.arrive(w)
                            msum_full.arrive(ms_slot)
                            iket_end(tk_ex)
                            iket_end(tk_sm)
                        K.assign(g0, g0 + n_tiles)
                    list_free.arrive(slot)
                    ring_advance(slot, use_x)
                role_tail(kind)

        softmax_role(r_sm0, 0, "sm0")
        softmax_role(r_sm1, 1, "sm1")

    return msa_kvmajor2


def make_kernel_kv(*, GQA, TOPK, HKV, B, TOTAL_Q, MAXB, MAXC, PAGED, MAX_PAGES,
                   KV_FP8, NUM_CTAS, CHUNK, FUSED_COMBINE=True,
                   PACK_FP8_PARTIAL=False, DYNAMIC_CHUNKS=False,
                   PACKED_TOKEN_LIST=False):
    HQ = HKV * GQA
    assert 128 % GQA == 0
    DIAG = os.environ.get("MSA_KV_DIAG", "")
    T = 128 // GQA                           
    NUM_ITEMS = MAXB * HKV * B * MAXC
    Q_TILE_BYTES = 128 * HEAD_DIM * 2
    KV_TILE_BYTES = 128 * HEAD_DIM * 2
    RAW_TILE_BYTES = 128 * HEAD_DIM
    ROW_BYTES = HEAD_DIM * 2
    TOPK_V4 = TOPK % 4 == 0
    META_N = 16
    ROWS_TOTAL = TOTAL_Q * HQ
    NCHUNK = B * MAXC * HKV
    SCHED_DONE = 2
    SCHED_CURSOR = 2 + NCHUNK
    TOKEN_BITS = (CHUNK - 1).bit_length()
    TOKEN_MASK = (1 << TOKEN_BITS) - 1
    if PACKED_TOKEN_LIST:
        assert TOKEN_BITS + max(1, (TOPK - 1).bit_length()) <= 16

    @K.kernel(warps=16, arch="sm_100a", min_blocks_per_sm=1, grid=NUM_CTAS)
    def msa_kvmajor(
        q_g: K.gptr[K.bf16],
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        q2k: K.gptr[K.i32],
        cu_q: K.gptr[K.i32],
        kv_meta: K.gptr[K.i32],
        page_table: K.gptr[K.i32],
        o_part: K.gptr[K.bf16],
        lse_part: K.gptr[K.f32],
        out: K.gptr[K.bf16],
        sched: K.gptr[K.i32],
        scale_log2: K.f32,
    ):
        cta = K.cta_id()
        warp_cta = K.warp_id()
        wg_id = warp_cta >> 2
        warp_in_wg = warp_cta & 3
        tid_in_wg = K.thread_id() & 127
        lane = K.lane_id()

                                                         
        smem = K.smem_pool()
        q_smem = smem.alloc((2, 128, HEAD_DIM), K.bf16, swizzle=K.SW128B)
        k_smem = smem.alloc((2, 128, HEAD_DIM), K.bf16, swizzle=K.SW128B)
        v_smem = smem.alloc((2, 128, HEAD_DIM), K.bf16, swizzle=K.SW128B)
        if KV_FP8:
            raw_smem = smem.alloc((128, HEAD_DIM), K.u8, swizzle=K.SW128B)
        if PACKED_TOKEN_LIST:
            tok_list = smem.alloc((NSLOT * CHUNK,), K.u16)
        else:
            tok_list = smem.alloc((NSLOT * CHUNK,), K.i32)
        meta = smem.alloc((NSLOT * META_N,), K.i32)
        wcount = smem.alloc((8,), K.i32)
        tmem_addr = smem.alloc((1,), K.u32)

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

                                                    
        list_full = K.MBarrier(smem, NSLOT)
        list_full.init(128)
        list_free = K.MBarrier(smem, NSLOT)
        list_free.init(256 + 32 + 32 + 64)
        if KV_FP8:
            q_load = K.Pipeline(smem, 2, full="tma", empty="tcgen05", empty_phase_offset=1)
        else:
            q_load = K.Pipeline(
                smem, 2, full="mbar", empty="tcgen05", init_full=64, empty_phase_offset=1
            )
        if KV_FP8:
            kv_load = K.Pipeline(
                smem, 2, full="mbar", empty="tcgen05", init_full=128, empty_phase_offset=1
            )
            raw_load = K.Pipeline(
                smem, 1, full="tma", empty="mbar", init_empty=128, empty_phase_offset=1
            )
        else:
            kv_load = K.Pipeline(smem, 2, full="tma", empty="tcgen05", empty_phase_offset=1)
        s_full = K.TCGen05Bar(smem, 2)
        s_full.init(1)
        p_full = K.MBarrier(smem, 2)
        p_full.init(128)
        o_full = K.TCGen05Bar(smem, 2)
        o_full.init(1)
        o_empty = K.MBarrier(smem, 2)
        o_empty.init(128)

        K.ptx.fence.proxy.async_.shared__cta()
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

                                                   
        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def tmem(col):
            return K.cuda.get_tmem_addr(K.uint32(0), 0, col)

        def tmem_load32(dst, dst_offset, tmem_col):
            K.ptx[TMEM_LD_32](*(dst[dst_offset + i] for i in range(32)), tmem_col)

        def tmem_store16(src, src_offset, tmem_col):
            K.ptx[TMEM_ST_16](tmem_col, *(src[src_offset + i] for i in range(16)))

        def ld_shared_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.shared.b32(value, ptr)
            return value

        def ld_shared_tok(ptr):
            if PACKED_TOKEN_LIST:
                value = K.local_scalar("uint16")
                K.ptx.ld.shared.u16(value, ptr)
                return K.Cast("int32", value)
            return ld_shared_i32(ptr)

        def ld_global_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.global_.nc.b32(value, ptr)
            return value

        def iket_range(name, *, leader_only=False):
            token = K.alloc_local([1], "uint32")
            if leader_only:
                K.assign(token[0], K.cuda.iket.sentinel_token(name))
                with K.If(warp_in_wg == 0), K.Then():
                    K.assign(token[0], K.cuda.iket.range_start(name))
            else:
                K.assign(token[0], K.cuda.iket.range_start(name))
            return token

        def iket_end(token):
            K.cuda.iket.range_end(token[0])

        def meta_ptr(slot, idx):
            return meta.ptr_to([slot * META_N + idx])

        def ring_advance(slot, use):
            K.assign(slot, slot + 1)
            with K.If(slot == NSLOT), K.Then():
                K.assign(slot, 0)
                K.assign(use, use + 1)

        def read_meta(slot):
            vals = [ld_shared_i32(meta_ptr(slot, i)) for i in range(12)]
            return vals

        def item_decode(item):
            b = item % MAXB
            t1 = item // MAXB
            h = t1 % HKV
            t2 = t1 // HKV
            c = t2 % MAXC
            s = t2 // MAXC
            return b, h, s, c

        def chunk_id(s, c, h):
            return (s * MAXC + c) * HKV + h

        def seq_lengths(s):
            """(q_base, q_len, kv_len, kv_coord_or_page) for sequence s and block b later."""
            q0 = ld_global_i32(cu_q.ptr_to([s]))
            q1 = ld_global_i32(cu_q.ptr_to([s + 1]))
            if PAGED:
                kv_len = ld_global_i32(kv_meta.ptr_to([s]))
                k_base = K.int32(0)
            else:
                k0 = ld_global_i32(kv_meta.ptr_to([s]))
                k1 = ld_global_i32(kv_meta.ptr_to([s + 1]))
                kv_len = k1 - k0
                k_base = k0
            return q0, q1 - q0, kv_len, k_base

        def cast_f32x2_bf16x2(dst_u32, src, offset):
            K.ptx.cvt.rn.bf16x2.f32(dst_u32[offset // 2], src[offset + 1], src[offset])

                                                                               
                                                                                  
                                                                               
        def combine_rows(rows_spec):
            """Merge the topk partials of up to W rows: one warp per row, lane = 4 dims.

            rows_spec: list of (valid_pred, row_index_expr).  Every global load of the
            batch (q2k, lengths, lse, partial rows) is issued before any value is
            consumed; invalid slots are masked afterwards."""
            cuq = K.alloc_local([B + 1], "int32")
            for i in range(B + 1):
                K.ptx.ld.global_.nc.b32(cuq[i], cu_q.ptr_to([i]))
            rows = []
            for (ok_row, row_e) in rows_spec:
                row = K.local_scalar("int32", init=K.min(K.max(row_e, 0), ROWS_TOTAL - 1))
                t = K.local_scalar("int32", init=row // HQ)
                head = row % HQ
                h = head // GQA
                s_ = K.local_scalar("int32", init=0)
                for i in range(B - 1):
                    K.assign(s_, s_ + K.Select(t >= cuq[i + 1], 1, 0))
                q_base = K.local_scalar("int32", init=0)
                q_end = K.local_scalar("int32", init=0)
                for i in range(B):
                    K.assign(q_base, K.Select(s_ == i, cuq[i], q_base))
                    K.assign(q_end, K.Select(s_ == i, cuq[i + 1], q_end))
                kvl = K.alloc_local([2], "int32")
                K.ptx.ld.global_.nc.b32(kvl[0], kv_meta.ptr_to([s_]))
                if not PAGED:
                    K.ptx.ld.global_.nc.b32(kvl[1], kv_meta.ptr_to([s_ + 1]))
                idxs = K.alloc_local([TOPK], "int32")
                rowk = (h * TOTAL_Q + t) * TOPK
                if TOPK_V4:
                    for v in range(TOPK // 4):
                        K.ptx.ld.global_.nc.v4.b32(
                            idxs[4 * v], idxs[4 * v + 1], idxs[4 * v + 2], idxs[4 * v + 3],
                            q2k.ptr_to([rowk + 4 * v]),
                        )
                else:
                    for v in range(TOPK):
                        K.ptx.ld.global_.nc.b32(idxs[v], q2k.ptr_to([rowk + v]))
                prow = K.Cast("int64", row) * TOPK
                lse = K.alloc_local([TOPK], "float32")
                words = K.alloc_local([TOPK if KV_FP8 else TOPK * 2], "uint32")
                for kk in range(TOPK):
                    K.ptx.ld.global_.cg.f32(lse[kk], lse_part.ptr_to([prow + kk]))
                    if KV_FP8:
                                                                                          
                                                                               
                        K.ptx.ld.global_.cg.b32(
                            words[kk],
                            o_part.ptr_to([((prow + kk) * HEAD_DIM + lane * 4) // 2]),
                        )
                    else:
                        K.ptx.ld.global_.cg.v2.b32(
                            words[2 * kk], words[2 * kk + 1],
                            o_part.ptr_to([(prow + kk) * HEAD_DIM + lane * 4]),
                        )
                rows.append((ok_row, row, t, q_base, q_end, kvl, idxs, lse, words))
            for (ok_row, row, t, q_base, q_end, kvl, idxs, lse, words) in rows:
                kv_len = kvl[0] if PAGED else kvl[1] - kvl[0]
                q_len = q_end - q_base
                nblocks = (kv_len + (BLK - 1)) // BLK
                pos = kv_len - q_len + (t - q_base)
                mval = K.local_scalar("float32", init=K.float32(NEG_INF))
                for kk in range(TOPK):
                    bk = idxs[kk]
                    ok = K.And(K.And(bk >= 0, bk < nblocks), bk * BLK <= pos)
                    K.assign(lse[kk], K.Select(ok, lse[kk], K.float32(NEG_INF)))
                    K.assign(mval, K.max(mval, lse[kk]))
                wsum = K.local_scalar("float32", init=K.float32(0.0))
                acc = K.alloc_local([4], "float32")
                for d in range(4):
                    K.assign(acc[d], K.float32(0.0))
                safe_m = K.max(mval, K.float32(-1e30))
                for kk in range(TOPK):
                    wk = K.local_scalar("float32")
                    K.ptx.ex2.approx.ftz.f32(wk, lse[kk] - safe_m)
                    K.assign(wsum, wsum + wk)
                    if KV_FP8:
                        bf_pairs = K.alloc_local([2], "uint32")
                        raw_lo = K.local_scalar("uint16", init=K.Cast("uint16", words[kk]))
                        raw_hi = K.local_scalar(
                            "uint16", init=K.Cast("uint16", K.shift_right(words[kk], K.uint32(16)))
                        )
                        K.ptx.cvt.rn.bf16x2.e4m3x2(bf_pairs[0], raw_lo)
                        K.ptx.cvt.rn.bf16x2.e4m3x2(bf_pairs[1], raw_hi)
                    for wi in range(2):
                        lo = K.local_scalar("float32")
                        hi = K.local_scalar("float32")
                        packed = bf_pairs[wi] if KV_FP8 else words[2 * kk + wi]
                        K.ptx.mov.b32(lo, K.shift_left(packed, K.uint32(16)))
                        K.ptx.mov.b32(hi, K.bitwise_and(packed, K.uint32(0xFFFF0000)))
                        K.assign(lo, K.Select(wk > K.float32(0.0), lo, K.float32(0.0)))
                        K.assign(hi, K.Select(wk > K.float32(0.0), hi, K.float32(0.0)))
                        K.ptx.fma.rn.f32(acc[2 * wi], wk, lo, acc[2 * wi])
                        K.ptx.fma.rn.f32(acc[2 * wi + 1], wk, hi, acc[2 * wi + 1])
                inv = K.local_scalar("float32", init=K.float32(0.0))
                with K.If(wsum > K.float32(0.0)), K.Then():
                    K.ptx.rcp.approx.ftz.f32(inv, wsum)
                outw = K.alloc_local([2], "uint32")
                for d in range(4):
                    K.assign(acc[d], acc[d] * inv)
                    if d % 2 == 1:
                        cast_f32x2_bf16x2(outw, acc, d - 1)
                with K.If(ok_row), K.Then():
                    K.ptx.st.global_.v2.b32(
                        out.ptr_to([K.Cast("int64", row) * HEAD_DIM + lane * 4]), outw[0], outw[1]
                    )

        def progressive_combine():
            """After this CTA's items: merge rows of chunks as they complete (all CTAs share
            each chunk's row cursor), waiting only for the earliest incomplete chunk."""
            W = max(1, 32 // TOPK)
            with K.serial(NCHUNK, unroll=False) as ch:
                h = ch % HKV
                sc = ch // HKV
                c = sc % MAXC
                s_ = sc // MAXC
                q_base, q_len, kv_len, k_base = seq_lengths(s_)
                chunk0 = c * CHUNK
                n_tok_c = K.max(K.min(q_len - chunk0, CHUNK), 0)
                nrows = n_tok_c * GQA
                with K.If(nrows > 0), K.Then():
                    target = 2 * ((kv_len + (BLK - 1)) // BLK)
                    seen = K.local_scalar("int32")
                    K.ptx.ld.acquire.gpu.global_.s32(seen, sched.ptr_to([SCHED_DONE + ch]))
                    with K.While(seen < target):
                        K.cuda.nano_sleep(K.uint32(200))
                        K.ptx.ld.acquire.gpu.global_.s32(seen, sched.ptr_to([SCHED_DONE + ch]))
                    K.ptx.fence.acq_rel.gpu()
                    more = K.local_scalar("int32", init=1)
                    with K.While(more != 0):
                        r0_raw = K.local_scalar("int32", init=0)
                        with K.If(lane == 0), K.Then():
                            K.ptx.atom.relaxed.gpu.global_.add.s32(
                                r0_raw, sched.ptr_to([SCHED_CURSOR + ch]), K.int32(W)
                            )
                        r0 = K.local_scalar("int32", init=K.uniform(r0_raw))
                        with K.If(r0 >= nrows):
                            with K.Then():
                                K.assign(more, 0)
                            with K.Else():
                                spec = []
                                for wi in range(W):
                                    r = r0 + wi
                                    tok = q_base + chunk0 + r // GQA
                                    row_e = tok * HQ + h * GQA + (r % GQA)
                                    spec.append((r < nrows, row_e))
                                combine_rows(spec)

        def role_tail(kind):
            K.cuda.cta_sync()
            if kind == "mma":
                dealloc = K.local_scalar("uint32")
                K.ptx.ld.shared.u32(dealloc, tmem_addr.ptr_to([0]))
                K.ptx[TMEM_DEALLOC](dealloc, K.uint32(N_COLS_TMEM))
            if FUSED_COMBINE and kind in ("sm0", "sm1") and DIAG != "nocombine":
                tk_cb = iket_range("combine", leader_only=True)
                progressive_combine()
                iket_end(tk_cb)
            K.cuda.cta_sync()
            if kind == "sm0":
                with K.If(K.thread_id() == 0), K.Then():
                    done = K.local_scalar("int32")
                    K.ptx.atom.acq_rel.gpu.global_.add.s32(done, sched.ptr_to([1]), K.int32(1))
                    with K.If(done == NUM_CTAS - 1), K.Then():
                        for i in range(2 + 2 * NCHUNK):
                            K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([i]), K.int32(0))

                                                 
        sp = K.specialize(chain_dispatch=True)
        r_sm0 = sp.role("softmax0", warps=[0, 1, 2, 3], regs=200)
        r_sm1 = sp.role("softmax1", warps=[4, 5, 6, 7], regs=200)
        r_aux = sp.role("aux", warps=[8, 9, 10, 11], regs=64)
        wg3 = sp.warpgroup("wg3", warps=range(12, 16), regs=48)
        r_mma = sp.role("mma", warps=[12], group=wg3)
        r_load = sp.role("load", warps=[13], group=wg3)
        r_qload = sp.role("qload", warps=[14, 15], group=wg3)

        with K.If(warp_cta == 12), K.Then():
            K.ptx[TMEM_ALLOC](K.address_of(tmem_addr[0]), K.uint32(N_COLS_TMEM))
            K.ptx[TMEM_RELINQUISH]()
            K.cuda.warp_sync()
        K.cuda.cta_sync()
        with K.If(K.thread_id() == 0), K.Then():
            allocated = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(allocated, tmem_addr.ptr_to([0]))
            K.cuda.trap_when_assert_failed(allocated == K.uint32(0))

                                                                               
                                                                           
                                                                               
        with r_aux:
            slot = K.local_scalar("int32", init=0)
            use_a = K.local_scalar("int32", init=0)
            running_a = K.local_scalar("int32", init=1)
            kv_it = K.local_scalar("int32", init=0)
            if DYNAMIC_CHUNKS:
                                                                          
                                                                            
                                                                             
                                                                          
                q_bounds = K.alloc_local([B + 1], "int32")
                for bi in range(B + 1):
                    K.ptx.ld.global_.nc.b32(q_bounds[bi], cu_q.ptr_to([bi]))
                nchunks = K.alloc_local([B], "int32")
                active_chunks = K.local_scalar("int32", init=0)
                for bi in range(B):
                    qlen_b = K.max(q_bounds[bi + 1] - q_bounds[bi], 0)
                    K.assign(nchunks[bi], (qlen_b + CHUNK - 1) // CHUNK)
                    K.assign(active_chunks, active_chunks + nchunks[bi])
                active_items = active_chunks * HKV * MAXB
            else:
                active_items = K.int32(NUM_ITEMS)
            with K.While(running_a != 0):
                with K.If(tid_in_wg == 0), K.Then():
                    grabbed = K.local_scalar("int32")
                    K.ptx.atom.relaxed.gpu.global_.add.s32(grabbed, sched.ptr_to([0]), K.int32(1))
                    K.ptx.st.shared.b32(wcount.ptr_to([4]), grabbed)
                K.cuda.warpgroup_sync(1)
                item = ld_shared_i32(wcount.ptr_to([4]))
                with K.If(item >= (0 if DIAG == "onlycombine" else active_items)):
                    with K.Then():
                        tk_free = iket_range("aux-wait-free", leader_only=True)
                        list_free.wait(slot, (use_a + 1) & 1)
                        iket_end(tk_free)
                        with K.If(tid_in_wg == 0), K.Then():
                            K.ptx.st.shared.b32(meta_ptr(slot, 0), K.int32(-1))
                                                                                    
                                                                                     
                            for mi in range(1, 12):
                                K.ptx.st.shared.b32(meta_ptr(slot, mi), K.int32(0))
                        list_full.arrive(slot)
                        K.assign(running_a, 0)
                    with K.Else():
                        if DYNAMIC_CHUNKS:
                                                                          
                                                                              
                                                                           
                                                                            
                                                                             
                                                                              
                                                                     
                            inner = K.local_scalar("int32", init=HKV * active_chunks)
                            b = item // inner
                            r = item % inner
                            h = r % HKV
                            c = K.local_scalar("int32", init=r // HKV)
                            s = K.local_scalar("int32", init=0)
                            for bi in range(B - 1):
                                past = c >= nchunks[bi]
                                K.assign(s, s + K.Select(past, 1, 0))
                                K.assign(c, c - K.Select(past, nchunks[bi], 0))
                        else:
                            b, h, s, c = item_decode(item)
                        q_base, q_len, kv_len, k_base = seq_lengths(s)
                        nblocks = (kv_len + (BLK - 1)) // BLK
                        chunk0 = c * CHUNK
                        valid_item = K.And(b < nblocks, chunk0 < q_len)
                        n_tok = K.local_scalar("int32", init=0)
                        ch_id = chunk_id(s, c, h)
                        tk_scan = iket_range("aux-scan", leader_only=True)
                                                                               
                        with K.If(valid_item), K.Then():
                            tk_free = iket_range("aux-wait-free", leader_only=True)
                            list_free.wait(slot, (use_a + 1) & 1)
                            iket_end(tk_free)
                            causal_off = kv_len - q_len
                            blk_lo = b * BLK
                            chunk_end = K.min(chunk0 + CHUNK, q_len)
                            BATCH = max(1, min(8, 32 // TOPK))
                            N_ROUNDS = CHUNK // 128
                            assert N_ROUNDS % BATCH == 0
                            with K.serial(N_ROUNDS // BATCH, unroll=False) as bt:
                                idxs = K.alloc_local([BATCH * TOPK], "int32")
                                for r in range(BATCH):
                                    for v in range(TOPK):
                                        K.assign(idxs[r * TOPK + v], K.int32(-1))
                                for r in range(BATCH):
                                    t_local = chunk0 + (bt * BATCH + r) * 128 + tid_in_wg
                                    with K.If(
                                        K.And(t_local < chunk_end, blk_lo <= causal_off + t_local)
                                    ), K.Then():
                                        row_base = (h * TOTAL_Q + q_base + t_local) * TOPK
                                        if TOPK_V4:
                                            for v in range(TOPK // 4):
                                                K.ptx.ld.global_.nc.v4.b32(
                                                    idxs[r * TOPK + 4 * v],
                                                    idxs[r * TOPK + 4 * v + 1],
                                                    idxs[r * TOPK + 4 * v + 2],
                                                    idxs[r * TOPK + 4 * v + 3],
                                                    q2k.ptr_to([row_base + 4 * v]),
                                                )
                                        else:
                                            for v in range(TOPK):
                                                K.ptx.ld.global_.nc.b32(
                                                    idxs[r * TOPK + v], q2k.ptr_to([row_base + v])
                                                )
                                for r in range(BATCH):
                                    t_local = chunk0 + (bt * BATCH + r) * 128 + tid_in_wg
                                    found = K.local_scalar("int32", init=0)
                                    slotk = K.local_scalar("int32", init=0)
                                    for v in range(TOPK):
                                        with K.If(idxs[r * TOPK + v] == b), K.Then():
                                            K.assign(found, 1)
                                            K.assign(slotk, v)
                                    ballot = K.local_scalar("uint32")
                                    K.ptx.vote_sync.ballot.b32(
                                        ballot, K.ptx.pred(found), K.uint32(0xFFFFFFFF)
                                    )
                                    cnt_w = K.local_scalar("uint32")
                                    K.ptx.popc.b32(cnt_w, ballot)
                                    lower = K.bitwise_and(
                                        ballot,
                                        K.shift_left(K.uint32(1), K.Cast("uint32", lane)) - K.uint32(1),
                                    )
                                    rank = K.local_scalar("uint32")
                                    K.ptx.popc.b32(rank, lower)
                                    with K.If(lane == 0), K.Then():
                                        K.ptx.st.shared.b32(
                                            wcount.ptr_to([warp_in_wg]), K.Cast("int32", cnt_w)
                                        )
                                    K.cuda.warpgroup_sync(1)
                                    c0 = ld_shared_i32(wcount.ptr_to([0]))
                                    c1 = ld_shared_i32(wcount.ptr_to([1]))
                                    c2 = ld_shared_i32(wcount.ptr_to([2]))
                                    c3 = ld_shared_i32(wcount.ptr_to([3]))
                                    prefix = K.local_scalar("int32", init=0)
                                    with K.If(warp_in_wg >= 1), K.Then():
                                        K.assign(prefix, prefix + c0)
                                    with K.If(warp_in_wg >= 2), K.Then():
                                        K.assign(prefix, prefix + c1)
                                    with K.If(warp_in_wg >= 3), K.Then():
                                        K.assign(prefix, prefix + c2)
                                    with K.If(found != 0), K.Then():
                                        pos = n_tok + prefix + K.Cast("int32", rank)
                                        packed_tok = K.bitwise_or(
                                            t_local - chunk0,
                                            K.shift_left(slotk, TOKEN_BITS),
                                        )
                                        if PACKED_TOKEN_LIST:
                                            K.ptx.st.shared.b16(
                                                tok_list.ptr_to([slot * CHUNK + pos]),
                                                K.Cast("uint16", packed_tok),
                                            )
                                        else:
                                            K.ptx.st.shared.b32(
                                                tok_list.ptr_to([slot * CHUNK + pos]),
                                                packed_tok,
                                            )
                                    K.assign(n_tok, n_tok + c0 + c1 + c2 + c3)
                                    K.cuda.warpgroup_sync(1)
                        iket_end(tk_scan)
                        n_tiles = (n_tok + (T - 1)) // T
                                                                                            
                        if FUSED_COMBINE:
                            with K.If(K.And(valid_item, n_tiles == 0)), K.Then():
                                with K.If(tid_in_wg == 0), K.Then():
                                    dummy = K.local_scalar("int32")
                                    K.ptx.atom.relaxed.gpu.global_.add.s32(
                                        dummy, sched.ptr_to([SCHED_DONE + ch_id]), K.int32(2)
                                    )
                        with K.If(K.And(valid_item, n_tiles > 0)), K.Then():
                          with K.If(tid_in_wg == 0), K.Then():
                            K.ptx.st.shared.b32(meta_ptr(slot, 0), n_tiles)
                            K.ptx.st.shared.b32(meta_ptr(slot, 1), n_tok)
                            K.ptx.st.shared.b32(meta_ptr(slot, 2), b)
                            K.ptx.st.shared.b32(meta_ptr(slot, 3), h)
                            K.ptx.st.shared.b32(meta_ptr(slot, 4), s)
                            K.ptx.st.shared.b32(meta_ptr(slot, 5), q_base)
                            K.ptx.st.shared.b32(meta_ptr(slot, 6), kv_len)
                            K.ptx.st.shared.b32(meta_ptr(slot, 7), q_len)
                            K.ptx.st.shared.b32(meta_ptr(slot, 8), chunk0)
                            if PAGED:
                                page = K.local_scalar("int32", init=0)
                                with K.If(valid_item), K.Then():
                                    K.assign(
                                        page,
                                        ld_global_i32(page_table.ptr_to([s * MAX_PAGES + b])),
                                    )
                                K.ptx.st.shared.b32(meta_ptr(slot, 9), page)
                            else:
                                K.ptx.st.shared.b32(meta_ptr(slot, 9), k_base + b * BLK)
                            K.ptx.st.shared.b32(meta_ptr(slot, 10), kv_len - q_len)
                            K.ptx.st.shared.b32(meta_ptr(slot, 11), ch_id)
                          list_full.arrive(slot)
                          ring_advance(slot, use_a)
                        if KV_FP8:
                            with K.If(K.And(valid_item, n_tiles > 0)), K.Then():
                                tk_cvt = iket_range("aux-fp8-convert", leader_only=True)
                                kvslot = kv_it & 1
                                for which, dst_tile in ((0, k_smem), (1, v_smem)):
                                    raw_load.full.wait(0, kv_it * 2 + which & 1)
                                    for rep in range(8):
                                        idx = rep * 128 + tid_in_wg
                                        row = idx >> 3
                                        chunk = idx & 7
                                        words = K.alloc_local([4], "uint32")
                                        K.ptx.ld.shared.v4.b32(
                                            words[0], words[1], words[2], words[3],
                                            raw_smem.ptr_to(row, chunk * 16),
                                        )
                                        outw = K.alloc_local([8], "uint32")
                                        for w in range(4):
                                            lo16 = K.local_scalar("uint16")
                                            hi16 = K.local_scalar("uint16")
                                            K.ptx.mov.b32(lo16, hi16, words[w])
                                            K.ptx.cvt.rn.bf16x2.e4m3x2(outw[2 * w], lo16)
                                            K.ptx.cvt.rn.bf16x2.e4m3x2(outw[2 * w + 1], hi16)
                                        K.ptx.st.shared.v4.b32(
                                            dst_tile[kvslot].ptr_to(row, chunk * 16),
                                            outw[0], outw[1], outw[2], outw[3],
                                        )
                                        K.ptx.st.shared.v4.b32(
                                            dst_tile[kvslot].ptr_to(row, chunk * 16 + 8),
                                            outw[4], outw[5], outw[6], outw[7],
                                        )
                                    K.ptx.fence.proxy.async_.shared__cta()
                                    raw_load.empty.arrive(0)
                                kv_load.full.arrive(kvslot)
                                iket_end(tk_cvt)
                                K.assign(kv_it, kv_it + 1)
            role_tail("aux")

                                                                               
                                           
                                                                               
        with wg3:
            with r_load:
                slot = K.local_scalar("int32", init=0)
                use_l = K.local_scalar("int32", init=0)
                running_l = K.local_scalar("int32", init=1)
                kv_pipe = K.PipelineState(2, phase=0)
                raw_it = K.local_scalar("int32", init=0)
                with K.While(running_l != 0):
                    tk_wl = iket_range("ld-wait-list")
                    list_full.wait(slot, use_l & 1)
                    iket_end(tk_wl)
                    m = read_meta(slot)
                    n_tiles, n_tok, b, h, s, q_base, kv_len, q_len, chunk0, kv_coord = (
                        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9]
                    )
                    with K.If(n_tiles < 0), K.Then():
                        K.assign(running_l, 0)
                    with K.If(n_tiles > 0), K.Then():
                                                           
                        tk_kv = iket_range("ld-kv-issue")
                        kv_load.empty.wait(kv_pipe.stage, kv_pipe.phase)
                        if KV_FP8:
                            for which, tmap in ((0, k_map), (1, v_map)):
                                raw_load.empty.wait(0, raw_it & 1)
                                with K.If(elected()), K.Then():
                                    if PAGED:
                                        K.ptx[TMA_G2S_2D](
                                            raw_smem.ptr_to(0, 0),
                                            K.address_of(tmap),
                                            K.int32(0),
                                            (kv_coord * HKV + h) * BLK,
                                            K.cuda.cvta_generic_to_shared(raw_load.full.ptr_to([0])),
                                        )
                                    else:
                                        K.ptx[TMA_G2S_3D](
                                            raw_smem.ptr_to(0, 0),
                                            K.address_of(tmap),
                                            K.int32(0),
                                            kv_coord,
                                            h,
                                            K.cuda.cvta_generic_to_shared(raw_load.full.ptr_to([0])),
                                        )
                                    raw_load.full.arrive(0, tx_count=RAW_TILE_BYTES)
                                K.assign(raw_it, raw_it + 1)
                        else:
                            with K.If(elected()), K.Then():
                                for tmap, tile in ((k_map, k_smem), (v_map, v_smem)):
                                    if PAGED:
                                        K.ptx[TMA_G2S_3D](
                                            tile[kv_pipe.stage].ptr_to(0, 0),
                                            K.address_of(tmap),
                                            K.int32(0),
                                            (kv_coord * HKV + h) * BLK,
                                            K.int32(0),
                                            K.cuda.cvta_generic_to_shared(
                                                kv_load.full.ptr_to([kv_pipe.stage])
                                            ),
                                        )
                                    else:
                                        K.ptx[TMA_G2S_3D](
                                            tile[kv_pipe.stage].ptr_to(0, 0),
                                            K.address_of(tmap),
                                            K.int32(0),
                                            kv_coord,
                                            h * 2,
                                            K.cuda.cvta_generic_to_shared(
                                                kv_load.full.ptr_to([kv_pipe.stage])
                                            ),
                                        )
                                kv_load.full.arrive(kv_pipe.stage, tx_count=2 * KV_TILE_BYTES)
                        kv_pipe.advance()
                        iket_end(tk_kv)
                    list_free.arrive(slot)
                    ring_advance(slot, use_l)
                role_tail("load")

            with r_mma:
                slot = K.local_scalar("int32", init=0)
                use_m = K.local_scalar("int32", init=0)
                running_m = K.local_scalar("int32", init=1)
                g_m = K.local_scalar("int32", init=0)
                kv_pipe_m = K.PipelineState(2, phase=0)
                q_pipe_m = K.PipelineState(2, phase=0)
                tb_raw = K.local_scalar("uint32")
                K.ptx.ld.shared.u32(tb_raw, tmem_addr.ptr_to([0]))
                tmem_base = K.local_scalar("uint32", init=K.uniform(tb_raw))
                q_desc, qoff = encode(q_smem[0])
                k_desc, koff = encode(k_smem[0])
                v_desc, mnoff = encode(v_smem[0], major="mn")

                def gemm_qk(q_stage, kv_stage, stage):
                    for ki in range(HEAD_DIM // MMA_K):
                        K.ptx[MMA_F16](
                            tmem_base + K.uint32(stage * 128),
                            desc_at(q_desc, q_stage * Q_STAGE16 + qoff(ki)),
                            desc_at(k_desc, kv_stage * KV_STAGE16 + koff(ki)),
                            K.uint32(ID_QK),
                            K.uint32(0),
                            K.uint32(0),
                            K.uint32(0),
                            K.uint32(0),
                            ki != 0,
                        )

                def gemm_pv(kv_stage, stage):
                    for ki in range(BLK // MMA_K):
                        K.ptx[MMA_F16](
                            tmem_base + K.uint32(256 + stage * 128),
                            tmem_base + K.uint32(stage * 128 + 64 + ki * (MMA_K // 2)),
                            desc_at(v_desc, kv_stage * KV_STAGE16 + mnoff(ki)),
                            K.uint32(ID_PV),
                            K.uint32(0),
                            K.uint32(0),
                            K.uint32(0),
                            K.uint32(0),
                            ki != 0,
                        )

                with K.While(running_m != 0):
                    tk_ml = iket_range("mma-wait-list")
                    list_full.wait(slot, use_m & 1)
                    iket_end(tk_ml)
                    n_tiles = ld_shared_i32(meta_ptr(slot, 0))
                    with K.If(n_tiles < 0), K.Then():
                        K.assign(running_m, 0)
                    with K.If(n_tiles > 0), K.Then():
                        kvs = K.local_scalar("int32", init=kv_pipe_m.stage)
                        tk_wkv = iket_range("mma-wait-kv")
                        kv_load.full.wait(kvs, kv_pipe_m.phase)
                        iket_end(tk_wkv)
                        K.ptx.tcgen05.fence__after_thread_sync()

                        def issue_qk(stage_expr):
                            qs = K.local_scalar("int32", init=q_pipe_m.stage)
                            tk_wq = iket_range("mma-wait-q")
                            q_load.full.wait(qs, q_pipe_m.phase)
                            iket_end(tk_wq)
                            K.ptx.fence.proxy.async_.shared__cta()
                            K.ptx.tcgen05.fence__after_thread_sync()
                            with K.If(elected()), K.Then():
                                gemm_qk(qs, kvs, stage_expr)
                                s_full.arrive(stage_expr)
                                q_load.empty.arrive(qs)
                            q_pipe_m.advance()

                                                               
                        issue_qk(g_m & 1)
                        with K.serial(n_tiles, unroll=False) as i:
                            stage = g_m & 1
                            par = (g_m >> 1) & 1
                                                                                      
                                                                                        
                                                                                      
                            with K.If(i + 1 < n_tiles), K.Then():
                                issue_qk(1 - stage)
                            tk_wp = iket_range("mma-wait-p")
                            p_full.wait(stage, par)
                            o_empty.wait(stage, par ^ 1)
                            iket_end(tk_wp)
                            K.ptx.tcgen05.fence__after_thread_sync()
                            with K.If(elected()), K.Then():
                                gemm_pv(kvs, stage)
                                o_full.arrive(stage)
                            K.assign(g_m, g_m + 1)
                        with K.If(elected()), K.Then():
                            kv_load.empty.arrive(kvs)
                        kv_pipe_m.advance()
                    list_free.arrive(slot)
                    ring_advance(slot, use_m)
                role_tail("mma")

            with r_qload:
                slot = K.local_scalar("int32", init=0)
                use_q = K.local_scalar("int32", init=0)
                running_q = K.local_scalar("int32", init=1)
                q_pipe_l = K.PipelineState(2, phase=0)
                wq = warp_cta - 14
                with K.While(running_q != 0):
                    tk_qw = iket_range("ql-wait-list")
                    list_full.wait(slot, use_q & 1)
                    iket_end(tk_qw)
                    m = read_meta(slot)
                    n_tiles, n_tok, b, h, s, q_base, kv_len, q_len, chunk0 = (
                        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]
                    )
                    with K.If(n_tiles < 0), K.Then():
                        K.assign(running_q, 0)
                    with K.If(n_tiles > 0), K.Then():
                        tk_qi = iket_range("ql-issue")
                        with K.serial(n_tiles, unroll=False) as i:
                            if KV_FP8:
                                                                            
                                                                          
                                                                               
                                                                             
                                q_load.empty.wait(q_pipe_l.stage, q_pipe_l.phase)
                                stage_now = K.local_scalar(
                                    "int32", init=q_pipe_l.stage
                                )
                                with K.If(elected()), K.Then():
                                    qbase = q_smem[stage_now].ptr_to(0, 0)
                                    for j in range(T):
                                        list_idx = K.min(i * T + j, n_tok - 1)
                                        tokw = ld_shared_tok(
                                            tok_list.ptr_to(
                                                [slot * CHUNK + list_idx]
                                            )
                                        )
                                        t_local = K.bitwise_and(
                                            tokw, K.int32(TOKEN_MASK)
                                        )
                                        dst = K.ptx.addr(
                                            qbase,
                                            wq * (Q_TILE_BYTES // 2)
                                            + j * GQA * (HEAD_DIM // 2) * 2,
                                        )
                                        K.ptx[TMA_G2S_4D](
                                            dst,
                                            K.address_of(q_map),
                                            K.int32(0),
                                            h * GQA,
                                            q_base + chunk0 + t_local,
                                            wq,
                                            K.cuda.cvta_generic_to_shared(
                                                q_load.full.ptr_to([stage_now])
                                            ),
                                        )
                                with K.If(warp_cta == 14), K.Then():
                                    with K.If(elected()), K.Then():
                                        q_load.full.arrive(
                                            stage_now, tx_count=Q_TILE_BYTES
                                        )
                            else:
                                q_load.empty.wait(q_pipe_l.stage, q_pipe_l.phase)
                                                                                     
                                                                                    
                                                                                      
                                K.ptx.fence.proxy.async_.shared__cta()
                                stage_now = K.local_scalar("int32", init=q_pipe_l.stage)
                                base_ptr = q_smem[stage_now].ptr_to(0, 0)
                                n_valid = K.min(n_tok - i * T, T)
                                toks = K.alloc_local([T], "int32")
                                if PACKED_TOKEN_LIST:
                                    words = K.alloc_local([T // 2], "uint32")
                                    for v in range(T // 8):
                                        K.ptx.ld.shared.v4.b32(
                                            words[4 * v], words[4 * v + 1],
                                            words[4 * v + 2], words[4 * v + 3],
                                            tok_list.ptr_to([slot * CHUNK + i * T + 8 * v]),
                                        )
                                    for v in range(T // 2):
                                        K.assign(
                                            toks[2 * v],
                                            K.Cast(
                                                "int32",
                                                K.bitwise_and(words[v], K.uint32(0xFFFF)),
                                            ),
                                        )
                                        K.assign(
                                            toks[2 * v + 1],
                                            K.Cast(
                                                "int32",
                                                K.shift_right(words[v], K.uint32(16)),
                                            ),
                                        )
                                else:
                                    for v in range(T // 4):
                                        K.ptx.ld.shared.v4.b32(
                                            toks[4 * v], toks[4 * v + 1],
                                            toks[4 * v + 2], toks[4 * v + 3],
                                            tok_list.ptr_to([slot * CHUNK + i * T + 4 * v]),
                                        )
                                x_lane = 2 * wq + (lane >> 4)
                                c = lane & 15
                                for j in range(32):
                                    row = 4 * j + x_lane
                                    if GQA >= 4:
                                        tidx = (4 * j) // GQA
                                        tok_valid = K.int32(tidx) < n_valid
                                        tokw = toks[tidx]
                                    else:
                                        tidx_e = row // GQA
                                        tok_valid = tidx_e < n_valid
                                        tokw = K.local_scalar("int32", init=toks[(4 * j) // GQA])
                                        for x in range(1, 4):
                                            cand = (4 * j + x) // GQA
                                            if cand != (4 * j) // GQA:
                                                K.assign(tokw, K.Select(x_lane == x, toks[cand], tokw))
                                    with K.If(tok_valid), K.Then():
                                        t_local = K.bitwise_and(tokw, K.int32(TOKEN_MASK))
                                        grow = (q_base + chunk0 + t_local) * HQ + h * GQA + (row % GQA)
                                        src = q_g.ptr_to([K.Cast("int64", grow) * HEAD_DIM + c * 8])
                                        dst = K.ptx.addr(
                                            base_ptr,
                                            (c >> 3) * (Q_TILE_BYTES // 2)
                                            + row * 128
                                            + K.shift_left(K.bitwise_xor(c & 7, row & 7), 4),
                                        )
                                        K.ptx["cp.async.cg.shared.global"](dst, src, 16, 16)
                                K.ptx.cp.async_.mbarrier.arrive.noinc.shared.b64(
                                    q_load.full.ptr_to([stage_now])
                                )
                            q_pipe_l.advance()
                        iket_end(tk_qi)
                    list_free.arrive(slot)
                    ring_advance(slot, use_q)
                role_tail("qload")

                                                                               
                                                    
                                                                               
        def softmax_role(role, w, kind):
            with role:
                slot = K.local_scalar("int32", init=0)
                use_x = K.local_scalar("int32", init=0)
                running_x = K.local_scalar("int32", init=1)
                g0 = K.local_scalar("int32", init=0)
                with K.While(running_x != 0):
                    tk_sl = iket_range("sm-wait-list", leader_only=True)
                    list_full.wait(slot, use_x & 1)
                    iket_end(tk_sl)
                    m = read_meta(slot)
                    n_tiles, n_tok, b, h, s, q_base, kv_len, q_len, chunk0 = (
                        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]
                    )
                    causal_off = m[10]
                    with K.If(n_tiles < 0), K.Then():
                        K.assign(running_x, 0)
                    with K.If(n_tiles > 0), K.Then():
                        start = (w - g0) & 1
                        n_iter = K.max((n_tiles - start + 1) // 2, 0)
                        kv_valid = K.min(kv_len - b * BLK, BLK)
                        row = tid_in_wg
                        j_tok = row // GQA
                        head = row % GQA
                        with K.serial(n_iter, unroll=False) as ii:
                            i = start + 2 * ii
                            g = g0 + i
                            par = (g >> 1) & 1
                            tok_valid = i * T + j_tok < n_tok
                            list_idx = K.min(i * T + j_tok, n_tok - 1)
                            tokw = ld_shared_tok(tok_list.ptr_to([slot * CHUNK + list_idx]))
                            t_local = K.bitwise_and(tokw, K.int32(TOKEN_MASK))
                            sel = K.shift_right(tokw, TOKEN_BITS)
                            pos = causal_off + chunk0 + t_local
                            col_limit = K.min(kv_valid, pos - b * BLK + 1)
                            s_chunk = K.alloc_local([BLK], "float32")
                            tk_ws = iket_range("sm-wait-s", leader_only=True)
                            s_full.wait(w, par)
                            iket_end(tk_ws)
                            tk_sm = iket_range("sm-softmax", leader_only=True)
                            K.ptx.tcgen05.fence__after_thread_sync()
                            for ci in range(BLK // 32):
                                tmem_load32(s_chunk, ci * 32, tmem(w * 128 + ci * 32))
                            K.ptx.tcgen05.wait__ld.sync.aligned()
                            with K.If(col_limit < BLK), K.Then():
                                for cidx in range(BLK):
                                    K.ptx.mov.b32(
                                        s_chunk[cidx],
                                        K.Select(cidx < col_limit, s_chunk[cidx], K.float32(NEG_INF)),
                                    )
                                               
                            mx = K.alloc_local([8], "float32")
                            for ch in range(8):
                                K.ptx.mov.b32(mx[ch], K.max(s_chunk[2 * ch], s_chunk[2 * ch + 1]))
                            for grp in range(1, BLK // 16):
                                for ch in range(8):
                                    K.ptx["max.f32"](
                                        mx[ch], mx[ch], s_chunk[16 * grp + 2 * ch], s_chunk[16 * grp + 2 * ch + 1]
                                    )
                            K.ptx["max.f32"](mx[0], mx[0], mx[1], mx[2])
                            K.ptx["max.f32"](mx[3], mx[3], mx[4], mx[5])
                            K.ptx["max.f32"](mx[0], mx[0], mx[6], mx[7])
                            row_max = K.local_scalar("float32")
                            K.assign(row_max, K.max(mx[0], mx[3]))
                            neg_bias = K.local_scalar("float32")
                            K.assign(neg_bias, K.float32(0.0) - row_max * scale_log2)
                                          
                            scale_pair = K.local_scalar("uint64")
                            bias_pair = K.local_scalar("uint64")
                            K.ptx.mov.b64(scale_pair, scale_log2, scale_log2)
                            K.ptx.mov.b64(bias_pair, neg_bias, neg_bias)
                            sum_acc = [K.local_scalar("uint64") for _ in range(8)]
                            for acc_pair in sum_acc:
                                K.ptx.mov.b64(
                                    acc_pair, K.float32(0.0), K.float32(0.0)
                                )
                            pair_tmp = K.local_scalar("uint64")
                            for pi in range(4):
                                p_chunk = K.alloc_local([16], "uint32")
                                for pair in range(16):
                                    cidx = pi * 32 + pair * 2
                                    K.ptx.mov.b64(
                                        pair_tmp, s_chunk[cidx], s_chunk[cidx + 1]
                                    )
                                    K.ptx.fma.rz.ftz.f32x2(
                                        pair_tmp, pair_tmp, scale_pair, bias_pair
                                    )
                                    K.ptx.mov.b64(
                                        s_chunk[cidx], s_chunk[cidx + 1], pair_tmp
                                    )
                                    K.ptx.ex2.approx.ftz.f32(
                                        s_chunk[cidx], s_chunk[cidx]
                                    )
                                    K.ptx.ex2.approx.ftz.f32(
                                        s_chunk[cidx + 1], s_chunk[cidx + 1]
                                    )
                                    K.ptx.mov.b64(
                                        pair_tmp, s_chunk[cidx], s_chunk[cidx + 1]
                                    )
                                    K.ptx.add.rn.ftz.f32x2(
                                        sum_acc[pair % 8],
                                        sum_acc[pair % 8],
                                        pair_tmp,
                                    )
                                    K.ptx.cvt.rn.bf16x2.f32(
                                        p_chunk[pair],
                                        s_chunk[cidx + 1],
                                        s_chunk[cidx],
                                    )
                                tmem_store16(
                                    p_chunk, 0, tmem(w * 128 + 64 + pi * 16)
                                )
                            for step in (4, 2, 1):
                                for a in range(step):
                                    K.ptx.add.rn.ftz.f32x2(
                                        sum_acc[a], sum_acc[a], sum_acc[a + step]
                                    )
                            sum_lo = K.local_scalar("float32")
                            sum_hi = K.local_scalar("float32")
                            K.ptx.mov.b64(sum_lo, sum_hi, sum_acc[0])
                            row_sum = K.local_scalar(
                                "float32", init=sum_lo + sum_hi
                            )
                            K.ptx.tcgen05.wait__st.sync.aligned()
                            K.ptx.tcgen05.fence__before_thread_sync()
                            p_full.arrive(w)
                            iket_end(tk_sm)
                                       
                            tk_wo = iket_range("sm-wait-o", leader_only=True)
                            o_full.wait(w, par)
                            iket_end(tk_wo)
                            tk_ep = iket_range("sm-epilogue", leader_only=True)
                            K.ptx.tcgen05.fence__after_thread_sync()
                            o_row = K.alloc_local([HEAD_DIM], "float32")
                            for ci in range(HEAD_DIM // 32):
                                tmem_load32(o_row, ci * 32, tmem(256 + w * 128 + ci * 32))
                            K.ptx.tcgen05.wait__ld.sync.aligned()
                            K.ptx.tcgen05.fence__before_thread_sync()
                            o_empty.arrive(w)
                            with K.If(tok_valid), K.Then():
                                inv = K.local_scalar("float32")
                                K.ptx.rcp.approx.ftz.f32(inv, row_sum)
                                lg = K.local_scalar("float32")
                                K.ptx.lg2.approx.ftz.f32(lg, row_sum)
                                lse2 = row_max * scale_log2 + lg
                                grow = (q_base + chunk0 + t_local) * HQ + h * GQA + head
                                prow = K.Cast("int64", grow * TOPK + sel)
                                K.ptx.st.global_.f32(lse_part.ptr_to([prow]), lse2)
                                if KV_FP8:
                                                                                      
                                                                                       
                                                                                        
                                                                                        
                                                                                       
                                                                              
                                    fp4_inv = K.local_scalar(
                                        "float32", init=inv * K.float32(16.0)
                                    )
                                    o_f4 = K.alloc_local([HEAD_DIM // 8], "uint32")
                                    for d8 in range(HEAD_DIM // 8):
                                        pair_bytes = K.alloc_local([4], "uint8")
                                        for pair in range(4):
                                            d = 8 * d8 + 2 * pair
                                            K.assign(o_row[d], o_row[d] * fp4_inv)
                                            K.assign(o_row[d + 1], o_row[d + 1] * fp4_inv)
                                            K.ptx.cvt.rn.satfinite.e2m1x2.f32(
                                                pair_bytes[pair],
                                                o_row[d + 1],
                                                o_row[d],
                                            )
                                        lo16 = K.local_scalar(
                                            "uint16",
                                            init=K.bitwise_or(
                                                K.Cast("uint16", pair_bytes[0]),
                                                K.shift_left(
                                                    K.Cast("uint16", pair_bytes[1]),
                                                    K.uint16(8),
                                                ),
                                            ),
                                        )
                                        hi16 = K.local_scalar(
                                            "uint16",
                                            init=K.bitwise_or(
                                                K.Cast("uint16", pair_bytes[2]),
                                                K.shift_left(
                                                    K.Cast("uint16", pair_bytes[3]),
                                                    K.uint16(8),
                                                ),
                                            ),
                                        )
                                        K.ptx.mov.b32(o_f4[d8], lo16, hi16)
                                    obase = prow * (HEAD_DIM // 4)
                                    for v in range(HEAD_DIM // 64):
                                        K.ptx.st.global_.v8.b32(
                                            o_part.ptr_to([obase + v * 16]),
                                            *(o_f4[8 * v + q8] for q8 in range(8)),
                                        )
                                elif PACK_FP8_PARTIAL:
                                    o_f8 = K.alloc_local([HEAD_DIM // 4], "uint32")
                                    for d4 in range(HEAD_DIM // 4):
                                        lo16 = K.local_scalar("uint16")
                                        hi16 = K.local_scalar("uint16")
                                        for dd in range(4):
                                            K.assign(o_row[4 * d4 + dd], o_row[4 * d4 + dd] * inv)
                                        K.ptx.cvt.rn.satfinite.e4m3x2.f32(
                                            lo16, o_row[4 * d4 + 1], o_row[4 * d4]
                                        )
                                        K.ptx.cvt.rn.satfinite.e4m3x2.f32(
                                            hi16, o_row[4 * d4 + 3], o_row[4 * d4 + 2]
                                        )
                                        K.assign(
                                            o_f8[d4],
                                            K.bitwise_or(
                                                K.Cast("uint32", lo16),
                                                K.shift_left(K.Cast("uint32", hi16), K.uint32(16)),
                                            ),
                                        )
                                                                                         
                                                                                         
                                    obase = prow * (HEAD_DIM // 2)
                                    for v in range(HEAD_DIM // 32):
                                        K.ptx.st.global_.v8.b32(
                                            o_part.ptr_to([obase + v * 16]),
                                            *(o_f8[8 * v + q8] for q8 in range(8)),
                                        )
                                else:
                                    o_bf = K.alloc_local([HEAD_DIM // 2], "uint32")
                                    for d in range(HEAD_DIM):
                                        K.assign(o_row[d], o_row[d] * inv)
                                        if d % 2 == 1:
                                            cast_f32x2_bf16x2(o_bf, o_row, d - 1)
                                    obase = prow * HEAD_DIM
                                    for v in range(HEAD_DIM // 16):
                                        K.ptx.st.global_.v8.b32(
                                            o_part.ptr_to([obase + v * 16]),
                                            *(o_bf[8 * v + q8] for q8 in range(8)),
                                        )
                            iket_end(tk_ep)
                        K.assign(g0, g0 + n_tiles)
                        if FUSED_COMBINE:
                                                                                          
                            K.cuda.warpgroup_sync(2 + w)
                            with K.If(tid_in_wg == 0), K.Then():
                                K.ptx.fence.acq_rel.gpu()
                                dummy = K.local_scalar("int32")
                                K.ptx.atom.release.gpu.global_.add.s32(
                                    dummy, sched.ptr_to([SCHED_DONE + m[11]]), K.int32(1)
                                )
                    list_free.arrive(slot)
                    ring_advance(slot, use_x)
                role_tail(kind)


        softmax_role(r_sm0, 0, "sm0")
        softmax_role(r_sm1, 1, "sm1")

    return msa_kvmajor

def make_kernel_combine_fp8(*, GQA, TOPK, HKV, B, TOTAL_Q, MAXB, NUM_CTAS):
    """Standalone flat-FP8 reduction: one warp owns one output row.

    Lanes 0..TOPK-1 each own one selection slot while the warp computes the
    softmax weights.  This turns the TOPK scalar metadata loads and exponentials
    into lane-parallel operations; the resulting weights are then broadcast to
    all 32 lanes for the coalesced 128-byte partial-vector loads.
    """
    HQ = HKV * GQA
    ROWS_TOTAL = TOTAL_Q * HQ
    assert 1 <= TOPK <= 32
    reduce_width = 1
    while reduce_width < TOPK:
        reduce_width *= 2
    REDUCE_STEPS = []
    step = reduce_width // 2
    while step:
        REDUCE_STEPS.append(step)
        step //= 2
    WARPS = 24
    IKET = bool(os.environ.get("MSA_KV_IKET"))

    @K.kernel(warps=WARPS, arch="sm_100a", min_blocks_per_sm=1, grid=NUM_CTAS)
    def msa_combine_fp8(
        q2k: K.gptr[K.i32],
        cu_q: K.gptr[K.i32],
        cu_k: K.gptr[K.i32],
        o_part: K.gptr[K.bf16],
        lse_part: K.gptr[K.f32],
        out: K.gptr[K.bf16],
    ):
        lane = K.lane_id()
        row = K.local_scalar("int32", init=K.cta_id() * WARPS + K.warp_id())
        stride = NUM_CTAS * WARPS
        cuq = K.alloc_local([B + 1], "int32")
        for i in range(B + 1):
            K.ptx.ld.global_.nc.b32(cuq[i], cu_q.ptr_to([i]))

        token = K.alloc_local([1], "uint32")
        if IKET:
            K.assign(token[0], K.cuda.iket.range_start("split-combine"))

        def shfl_bfly_f32(value, delta):
            bits = K.local_scalar("uint32")
            K.ptx.shfl_sync.bfly.b32(
                bits,
                K.reinterpret("uint32", value),
                K.uint32(delta),
                K.uint32(31),
                K.uint32(0xFFFFFFFF),
            )
            return K.reinterpret("float32", bits)

        def shfl_idx_f32(value, source_lane):
            bits = K.local_scalar("uint32")
            K.ptx.shfl_sync.idx.b32(
                bits,
                K.reinterpret("uint32", value),
                K.uint32(source_lane),
                K.uint32(31),
                K.uint32(0xFFFFFFFF),
            )
            return K.reinterpret("float32", bits)

        with K.While(row < ROWS_TOTAL):
            t = K.local_scalar("int32", init=row // HQ)
            head = row % HQ
            h = head // GQA
            s = K.local_scalar("int32", init=0)
            for i in range(B - 1):
                K.assign(s, s + K.Select(t >= cuq[i + 1], 1, 0))
            q_base = K.local_scalar("int32", init=0)
            q_end = K.local_scalar("int32", init=0)
            for i in range(B):
                K.assign(q_base, K.Select(s == i, cuq[i], q_base))
                K.assign(q_end, K.Select(s == i, cuq[i + 1], q_end))
            k0 = K.local_scalar("int32")
            k1 = K.local_scalar("int32")
            K.ptx.ld.global_.nc.b32(k0, cu_k.ptr_to([s]))
            K.ptx.ld.global_.nc.b32(k1, cu_k.ptr_to([s + 1]))
            kv_len = k1 - k0
            q_len = q_end - q_base
            nblocks = (kv_len + (BLK - 1)) // BLK
            pos = kv_len - q_len + (t - q_base)

            rowk = (h * TOTAL_Q + t) * TOPK
            prow = K.Cast("int64", row) * TOPK
            bk = K.local_scalar("int32", init=K.int32(-1))
            lse_lane = K.local_scalar("float32", init=K.float32(NEG_INF))
            with K.If(lane < TOPK), K.Then():
                K.ptx.ld.global_.nc.b32(bk, q2k.ptr_to([rowk + lane]))
                K.ptx.ld.global_.cg.f32(lse_lane, lse_part.ptr_to([prow + lane]))
            ok = K.And(K.And(bk >= 0, bk < nblocks), bk * BLK <= pos)
            K.assign(lse_lane, K.Select(ok, lse_lane, K.float32(NEG_INF)))

            mval_lane = K.local_scalar("float32", init=lse_lane)
            for delta in REDUCE_STEPS:
                K.assign(mval_lane, K.max(mval_lane, shfl_bfly_f32(mval_lane, delta)))
            mval = shfl_idx_f32(mval_lane, 0)
            safe_m = K.max(mval, K.float32(-1e30))
            weight_lane = K.local_scalar("float32", init=K.float32(0.0))
            with K.If(lane < TOPK), K.Then():
                K.ptx.ex2.approx.ftz.f32(weight_lane, lse_lane - safe_m)
            wsum_lane = K.local_scalar("float32", init=weight_lane)
            for delta in REDUCE_STEPS:
                K.assign(wsum_lane, wsum_lane + shfl_bfly_f32(wsum_lane, delta))
            wsum = shfl_idx_f32(wsum_lane, 0)

                                                                            
                                                                             
                                                                             
            acc = K.alloc_local([2], "uint32")
            for wi in range(2):
                K.assign(acc[wi], K.uint32(0))
            for kk in range(TOPK):
                wk = shfl_idx_f32(weight_lane, kk)
                weight_pair = K.local_scalar("uint32")
                K.ptx.cvt.rn.bf16x2.f32(weight_pair, wk, wk)
                word = K.local_scalar("uint32")
                K.ptx.ld.global_.cg.b32(
                    word, o_part.ptr_to([((prow + kk) * HEAD_DIM + lane * 4) // 2])
                )
                bf_pairs = K.alloc_local([2], "uint32")
                raw_lo = K.local_scalar("uint16", init=K.Cast("uint16", word))
                raw_hi = K.local_scalar(
                    "uint16", init=K.Cast("uint16", K.shift_right(word, K.uint32(16)))
                )
                K.ptx.cvt.rn.bf16x2.e4m3x2(bf_pairs[0], raw_lo)
                K.ptx.cvt.rn.bf16x2.e4m3x2(bf_pairs[1], raw_hi)
                for wi in range(2):
                                                                              
                                                                             
                    K.assign(
                        bf_pairs[wi],
                        K.Select(wk > K.float32(0.0), bf_pairs[wi], K.uint32(0)),
                    )
                    K.ptx.fma.rn.bf16x2(
                        acc[wi], bf_pairs[wi], weight_pair, acc[wi]
                    )

            inv = K.local_scalar("float32", init=K.float32(0.0))
            with K.If(wsum > K.float32(0.0)), K.Then():
                K.ptx.rcp.approx.ftz.f32(inv, wsum)
            outw = K.alloc_local([2], "uint32")
            inv_pair = K.local_scalar("uint32")
            K.ptx.cvt.rn.bf16x2.f32(inv_pair, inv, inv)
            for wi in range(2):
                K.ptx.mul.rn.bf16x2(outw[wi], acc[wi], inv_pair)
            K.ptx.st.global_.v2.b32(
                out.ptr_to([K.Cast("int64", row) * HEAD_DIM + lane * 4]), outw[0], outw[1]
            )
            K.assign(row, row + stride)

        if IKET:
            K.cuda.iket.range_end(token[0])

    return msa_combine_fp8


def make_kernel_combine_bf16_paged(*, GQA, TOPK, HKV, B, TOTAL_Q, MAXB, NUM_CTAS):
    """Standalone paged-BF16 reduction: one warp owns one output row.

    Lanes 0..TOPK-1 load and reduce the selection metadata, then all lanes
    stream four BF16 partial values per selected block.  The producer and this
    kernel run consecutively on the same stream, so no in-kernel grid barrier
    or completion counters are needed.
    """
    HQ = HKV * GQA
    ROWS_TOTAL = TOTAL_Q * HQ
    assert 1 <= TOPK <= 32
    reduce_width = 1
    while reduce_width < TOPK:
        reduce_width *= 2
    REDUCE_STEPS = []
    step = reduce_width // 2
    while step:
        REDUCE_STEPS.append(step)
        step //= 2
    WARPS = 16
    IKET = bool(os.environ.get("MSA_KV_IKET"))

    @K.kernel(warps=WARPS, arch="sm_100a", min_blocks_per_sm=1, grid=NUM_CTAS)
    def msa_combine_bf16_paged(
        q2k: K.gptr[K.i32],
        cu_q: K.gptr[K.i32],
        seqused_k: K.gptr[K.i32],
        o_part: K.gptr[K.bf16],
        lse_part: K.gptr[K.f32],
        out: K.gptr[K.bf16],
    ):
        lane = K.lane_id()
        row = K.local_scalar("int32", init=K.cta_id() * WARPS + K.warp_id())
        stride = NUM_CTAS * WARPS
        cuq = K.alloc_local([B + 1], "int32")
        for i in range(B + 1):
            K.ptx.ld.global_.nc.b32(cuq[i], cu_q.ptr_to([i]))

        token = K.alloc_local([1], "uint32")
        if IKET:
            K.assign(token[0], K.cuda.iket.range_start("split-combine-bf16"))

        def shfl_bfly_f32(value, delta):
            bits = K.local_scalar("uint32")
            K.ptx.shfl_sync.bfly.b32(
                bits,
                K.reinterpret("uint32", value),
                K.uint32(delta),
                K.uint32(31),
                K.uint32(0xFFFFFFFF),
            )
            return K.reinterpret("float32", bits)

        def shfl_idx_f32(value, source_lane):
            bits = K.local_scalar("uint32")
            K.ptx.shfl_sync.idx.b32(
                bits,
                K.reinterpret("uint32", value),
                K.uint32(source_lane),
                K.uint32(31),
                K.uint32(0xFFFFFFFF),
            )
            return K.reinterpret("float32", bits)

        with K.While(row < ROWS_TOTAL):
            t = K.local_scalar("int32", init=row // HQ)
            head = row % HQ
            h = head // GQA
            s = K.local_scalar("int32", init=0)
            for i in range(B - 1):
                K.assign(s, s + K.Select(t >= cuq[i + 1], 1, 0))
            q_base = K.local_scalar("int32", init=0)
            q_end = K.local_scalar("int32", init=0)
            for i in range(B):
                K.assign(q_base, K.Select(s == i, cuq[i], q_base))
                K.assign(q_end, K.Select(s == i, cuq[i + 1], q_end))
            kv_len = K.local_scalar("int32")
            K.ptx.ld.global_.nc.b32(kv_len, seqused_k.ptr_to([s]))
            q_len = q_end - q_base
            nblocks = (kv_len + (BLK - 1)) // BLK
            pos = kv_len - q_len + (t - q_base)

            rowk = (h * TOTAL_Q + t) * TOPK
            prow = K.Cast("int64", row) * TOPK
            bk = K.local_scalar("int32", init=K.int32(-1))
            lse_lane = K.local_scalar("float32", init=K.float32(NEG_INF))
            with K.If(lane < TOPK), K.Then():
                K.ptx.ld.global_.nc.b32(bk, q2k.ptr_to([rowk + lane]))
                K.ptx.ld.global_.cg.f32(lse_lane, lse_part.ptr_to([prow + lane]))
            ok = K.And(K.And(bk >= 0, bk < nblocks), bk * BLK <= pos)
            K.assign(lse_lane, K.Select(ok, lse_lane, K.float32(NEG_INF)))

            mval_lane = K.local_scalar("float32", init=lse_lane)
            for delta in REDUCE_STEPS:
                K.assign(mval_lane, K.max(mval_lane, shfl_bfly_f32(mval_lane, delta)))
            mval = shfl_idx_f32(mval_lane, 0)
            safe_m = K.max(mval, K.float32(-1e30))
            weight_lane = K.local_scalar("float32", init=K.float32(0.0))
            with K.If(lane < TOPK), K.Then():
                K.ptx.ex2.approx.ftz.f32(weight_lane, lse_lane - safe_m)
            wsum_lane = K.local_scalar("float32", init=weight_lane)
            for delta in REDUCE_STEPS:
                K.assign(wsum_lane, wsum_lane + shfl_bfly_f32(wsum_lane, delta))
            wsum = shfl_idx_f32(wsum_lane, 0)

                                                                          
                                                                  
            acc = K.alloc_local([2], "uint32")
            for wi in range(2):
                K.assign(acc[wi], K.uint32(0))
            for kk in range(TOPK):
                wk = shfl_idx_f32(weight_lane, kk)
                weight_pair = K.local_scalar("uint32")
                K.ptx.cvt.rn.bf16x2.f32(weight_pair, wk, wk)
                word = K.local_scalar("uint32")
                                                                          
                                                                              
                K.ptx.ld.global_.cg.b32(
                    word, o_part.ptr_to([((prow + kk) * HEAD_DIM + lane * 4) // 2])
                )
                bf_pairs = K.alloc_local([2], "uint32")
                raw_lo = K.local_scalar("uint16", init=K.Cast("uint16", word))
                raw_hi = K.local_scalar(
                    "uint16", init=K.Cast("uint16", K.shift_right(word, K.uint32(16)))
                )
                K.ptx.cvt.rn.bf16x2.e4m3x2(bf_pairs[0], raw_lo)
                K.ptx.cvt.rn.bf16x2.e4m3x2(bf_pairs[1], raw_hi)
                for wi in range(2):
                    K.assign(
                        bf_pairs[wi],
                        K.Select(wk > K.float32(0.0), bf_pairs[wi], K.uint32(0)),
                    )
                    K.ptx.fma.rn.bf16x2(
                        acc[wi], bf_pairs[wi], weight_pair, acc[wi]
                    )

            inv = K.local_scalar("float32", init=K.float32(0.0))
            with K.If(wsum > K.float32(0.0)), K.Then():
                K.ptx.rcp.approx.ftz.f32(inv, wsum)
            outw = K.alloc_local([2], "uint32")
            inv_pair = K.local_scalar("uint32")
            K.ptx.cvt.rn.bf16x2.f32(inv_pair, inv, inv)
            for wi in range(2):
                K.ptx.mul.rn.bf16x2(outw[wi], acc[wi], inv_pair)
            K.ptx.st.global_.v2.b32(
                out.ptr_to([K.Cast("int64", row) * HEAD_DIM + lane * 4]), outw[0], outw[1]
            )
            K.assign(row, row + stride)

        if IKET:
            K.cuda.iket.range_end(token[0])

    return msa_combine_bf16_paged


def make_kernel_combine_packed_halfwarp(
    *, GQA, TOPK, HKV, B, TOTAL_Q, MAXB, NUM_CTAS, PAGED
):
    """Merge packed partials in independent 8- or 16-lane row groups."""
    HQ = HKV * GQA
    ROWS_TOTAL = TOTAL_Q * HQ
    assert 1 <= TOPK <= 16
    reduce_width = 1
    while reduce_width < TOPK:
        reduce_width *= 2
    REDUCE_STEPS = []
    step = reduce_width // 2
    while step:
        REDUCE_STEPS.append(step)
        step //= 2
    WARPS = 12
    ROWS_PER_WARP = 2 if PAGED else 4
    GROUP_SHIFT = 4 if PAGED else 3
    GROUP_MASK = 15 if PAGED else 7
                                                                 
                                                 
    SHFL_SEGMENT = 0x100F if PAGED else 0x1807
    IKET = bool(os.environ.get("MSA_KV_IKET"))

    @K.kernel(warps=WARPS, arch="sm_100a", min_blocks_per_sm=1, grid=NUM_CTAS)
    def msa_combine_packed_halfwarp(
        q2k: K.gptr[K.i32],
        cu_q: K.gptr[K.i32],
        kv_meta: K.gptr[K.i32],
        o_part: K.gptr[K.bf16],
        lse_part: K.gptr[K.f32],
        out: K.gptr[K.bf16],
    ):
        lane = K.lane_id()
        lane_row = lane >> GROUP_SHIFT
        lane_group = lane & GROUP_MASK
        row_base = K.local_scalar(
            "int32",
            init=(K.cta_id() * WARPS + K.warp_id()) * ROWS_PER_WARP,
        )
        stride = NUM_CTAS * WARPS * ROWS_PER_WARP
        cuq = K.alloc_local([B + 1], "int32")
        for i in range(B + 1):
            K.ptx.ld.global_.nc.b32(cuq[i], cu_q.ptr_to([i]))

        token = K.alloc_local([1], "uint32")
        if IKET:
            K.assign(token[0], K.cuda.iket.range_start("split-combine-packed-halfwarp"))

        def shfl_bfly_f32(value, delta):
            bits = K.local_scalar("uint32")
            K.ptx.shfl_sync.bfly.b32(
                bits,
                K.reinterpret("uint32", value),
                K.uint32(delta),
                K.uint32(SHFL_SEGMENT),
                K.uint32(0xFFFFFFFF),
            )
            return K.reinterpret("float32", bits)

        def shfl_idx_f32(value, source_lane):
            bits = K.local_scalar("uint32")
            K.ptx.shfl_sync.idx.b32(
                bits,
                K.reinterpret("uint32", value),
                K.uint32(source_lane),
                K.uint32(SHFL_SEGMENT),
                K.uint32(0xFFFFFFFF),
            )
            return K.reinterpret("float32", bits)

        with K.While(row_base < ROWS_TOTAL):
            valid_row = row_base + lane_row < ROWS_TOTAL
            row = K.local_scalar(
                "int32", init=K.min(row_base + lane_row, ROWS_TOTAL - 1)
            )
            t = K.local_scalar("int32", init=row // HQ)
            head = row % HQ
            h = head // GQA
            s = K.local_scalar("int32", init=0)
            for i in range(B - 1):
                K.assign(s, s + K.Select(t >= cuq[i + 1], 1, 0))
            q_base = K.local_scalar("int32", init=0)
            q_end = K.local_scalar("int32", init=0)
            for i in range(B):
                K.assign(q_base, K.Select(s == i, cuq[i], q_base))
                K.assign(q_end, K.Select(s == i, cuq[i + 1], q_end))
            if PAGED:
                kv_len = K.local_scalar("int32")
                K.ptx.ld.global_.nc.b32(kv_len, kv_meta.ptr_to([s]))
            else:
                k0 = K.local_scalar("int32")
                k1 = K.local_scalar("int32")
                K.ptx.ld.global_.nc.b32(k0, kv_meta.ptr_to([s]))
                K.ptx.ld.global_.nc.b32(k1, kv_meta.ptr_to([s + 1]))
                kv_len = k1 - k0
            q_len = q_end - q_base
            nblocks = (kv_len + (BLK - 1)) // BLK
            pos = kv_len - q_len + (t - q_base)

            rowk = (h * TOTAL_Q + t) * TOPK
            prow = K.Cast("int64", row) * TOPK
            bk = K.local_scalar("int32", init=K.int32(-1))
            lse_lane = K.local_scalar("float32", init=K.float32(NEG_INF))
            with K.If(lane_group < TOPK), K.Then():
                K.ptx.ld.global_.nc.b32(bk, q2k.ptr_to([rowk + lane_group]))
                K.ptx.ld.global_.cg.f32(lse_lane, lse_part.ptr_to([prow + lane_group]))
            ok = K.And(K.And(bk >= 0, bk < nblocks), bk * BLK <= pos)
            K.assign(lse_lane, K.Select(ok, lse_lane, K.float32(NEG_INF)))

            mval_lane = K.local_scalar("float32", init=lse_lane)
            for delta in REDUCE_STEPS:
                K.assign(mval_lane, K.max(mval_lane, shfl_bfly_f32(mval_lane, delta)))
            mval = shfl_idx_f32(mval_lane, 0)
            safe_m = K.max(mval, K.float32(-1e30))
            weight_lane = K.local_scalar("float32", init=K.float32(0.0))
            with K.If(lane_group < TOPK), K.Then():
                K.ptx.ex2.approx.ftz.f32(weight_lane, lse_lane - safe_m)
            wsum_lane = K.local_scalar("float32", init=weight_lane)
            for delta in REDUCE_STEPS:
                K.assign(wsum_lane, wsum_lane + shfl_bfly_f32(wsum_lane, delta))
            wsum = shfl_idx_f32(wsum_lane, 0)

            NPAIR = 4 if PAGED else 8
            acc = K.alloc_local([NPAIR], "uint32")
            for wi in range(NPAIR):
                K.assign(acc[wi], K.uint32(0))
            for kk in range(TOPK):
                wk = shfl_idx_f32(weight_lane, kk)
                weight_pair = K.local_scalar("uint32")
                K.ptx.cvt.rn.bf16x2.f32(weight_pair, wk, wk)
                bf_pairs = K.alloc_local([NPAIR], "uint32")
                if PAGED:
                    words = K.alloc_local([2], "uint32")
                    K.ptx.ld.global_.cg.v2.b32(
                        words[0],
                        words[1],
                        o_part.ptr_to(
                            [((prow + kk) * HEAD_DIM + lane_group * 8) // 2]
                        ),
                    )
                    for wi in range(2):
                        raw_lo = K.local_scalar(
                            "uint16", init=K.Cast("uint16", words[wi])
                        )
                        raw_hi = K.local_scalar(
                            "uint16",
                            init=K.Cast(
                                "uint16",
                                K.shift_right(words[wi], K.uint32(16)),
                            ),
                        )
                        K.ptx.cvt.rn.bf16x2.e4m3x2(
                            bf_pairs[2 * wi], raw_lo
                        )
                        K.ptx.cvt.rn.bf16x2.e4m3x2(
                            bf_pairs[2 * wi + 1], raw_hi
                        )
                else:
                    words = K.alloc_local([2], "uint32")
                    K.ptx.ld.global_.cg.v2.b32(
                        words[0],
                        words[1],
                        o_part.ptr_to(
                            [(prow + kk) * (HEAD_DIM // 4) + lane_group * 4]
                        ),
                    )
                    for wi in range(8):
                        raw_pair = K.local_scalar(
                            "uint8",
                            init=K.Cast(
                                "uint8",
                                K.shift_right(
                                    words[wi // 4], K.uint32(8 * (wi % 4))
                                ),
                            ),
                        )
                        K.ptx.cvt.rn.bf16x2.e2m1x2(
                            bf_pairs[wi], raw_pair
                        )
                for wi in range(NPAIR):
                    K.assign(
                        bf_pairs[wi],
                        K.Select(wk > K.float32(0.0), bf_pairs[wi], K.uint32(0)),
                    )
                    K.ptx.fma.rn.bf16x2(
                        acc[wi], bf_pairs[wi], weight_pair, acc[wi]
                    )

            inv = K.local_scalar("float32", init=K.float32(0.0))
            with K.If(wsum > K.float32(0.0)), K.Then():
                K.ptx.rcp.approx.ftz.f32(inv, wsum)
            if not PAGED:
                                                                           
                                                                            
                                                                         
                                                                        
                K.assign(inv, inv * K.float32(0.0625))
            inv_pair = K.local_scalar("uint32")
            K.ptx.cvt.rn.bf16x2.f32(inv_pair, inv, inv)
            outw = K.alloc_local([NPAIR], "uint32")
            for wi in range(NPAIR):
                K.ptx.mul.rn.bf16x2(outw[wi], acc[wi], inv_pair)
            with K.If(valid_row), K.Then():
                if PAGED:
                    K.ptx.st.global_.v4.b32(
                        out.ptr_to(
                            [K.Cast("int64", row) * HEAD_DIM + lane_group * 8]
                        ),
                        outw[0],
                        outw[1],
                        outw[2],
                        outw[3],
                    )
                else:
                    K.ptx.st.global_.v8.b32(
                        out.ptr_to(
                            [K.Cast("int64", row) * HEAD_DIM + lane_group * 16]
                        ),
                        outw[0],
                        outw[1],
                        outw[2],
                        outw[3],
                        outw[4],
                        outw[5],
                        outw[6],
                        outw[7],
                    )
            K.assign(row_base, row_base + stride)

        if IKET:
            K.cuda.iket.range_end(token[0])

    return msa_combine_packed_halfwarp


                                                                             
           
                                                                             


class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode(tensor, dtype, dims, strides, box):
    desc = _AlignedTensorMap()
    rank = len(dims)
    assert len(strides) == rank - 1 and len(box) == rank
    tvm.get_global_func("runtime.cuTensorMapEncodeTiled")(
        desc.ptr,
        dtype,
        rank,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *dims,
        *strides,
        *box,
        *((1,) * rank),
        0,                   
        3,                
        2,                     
        0,                 
    )
    return desc


_KERNEL_CACHE = {}


def _compile_kv(key, **cfg):
    if key not in _KERNEL_CACHE:
        kernel = make_kernel_kv(**cfg)
        if os.environ.get("MSA_KV_IKET"):
            from tvm.tirx.cuda import iket as _iket

            _KERNEL_CACHE[key] = _iket.IketProfiler().compile(
                kernel.mod, target=kernel.target(), tir_pipeline="tirx"
            )
        else:
            _KERNEL_CACHE[key] = kernel.compile()
    return _KERNEL_CACHE[key]


def _compile_combine_fp8(key, **cfg):
    if key not in _KERNEL_CACHE:
        kernel = make_kernel_combine_fp8(**cfg)
        if os.environ.get("MSA_KV_IKET"):
            from tvm.tirx.cuda import iket as _iket

            _KERNEL_CACHE[key] = _iket.IketProfiler().compile(
                kernel.mod, target=kernel.target(), tir_pipeline="tirx"
            )
        else:
            _KERNEL_CACHE[key] = kernel.compile()
    return _KERNEL_CACHE[key]


def _compile_combine_bf16_paged(key, **cfg):
    if key not in _KERNEL_CACHE:
        kernel = make_kernel_combine_bf16_paged(**cfg)
        if os.environ.get("MSA_KV_IKET"):
            from tvm.tirx.cuda import iket as _iket

            _KERNEL_CACHE[key] = _iket.IketProfiler().compile(
                kernel.mod, target=kernel.target(), tir_pipeline="tirx"
            )
        else:
            _KERNEL_CACHE[key] = kernel.compile()
    return _KERNEL_CACHE[key]


def _compile_combine_packed_halfwarp(key, **cfg):
    if key not in _KERNEL_CACHE:
        kernel = make_kernel_combine_packed_halfwarp(**cfg)
        if os.environ.get("MSA_KV_IKET"):
            from tvm.tirx.cuda import iket as _iket

            _KERNEL_CACHE[key] = _iket.IketProfiler().compile(
                kernel.mod, target=kernel.target(), tir_pipeline="tirx"
            )
        else:
            _KERNEL_CACHE[key] = kernel.compile()
    return _KERNEL_CACHE[key]


def setup_kv(data, total_q, B):
    q = data["q"]
    k = data["k"]
    v = data["v"]
    q2k = data["q2k_indices"]
    cu_q = data["cu_seqlens_q"]
    cu_k = data["cu_seqlens_k"]
    page_table = data["page_table"]
    seqused_k = data["seqused_k"]
    out = data["output"]
    device = q.device
    assert q.dtype == torch.bfloat16, "this candidate implements bf16 queries"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous()
    assert q2k.is_contiguous() and q2k.dtype == torch.int32
    HQ = q.shape[1]
    HKV = k.shape[1]
    GQA = HQ // HKV
    TOPK = q2k.shape[2]
    PAGED = page_table is not None
    KV_FP8 = k.dtype == torch.float8_e4m3fn
    if not KV_FP8:
        assert k.dtype == torch.bfloat16
    if PAGED:
        num_pages = k.shape[0]
        MAX_PAGES = page_table.shape[1]
        MAXB = MAX_PAGES
        kv_meta = seqused_k
        assert page_table.is_contiguous() and seqused_k.is_contiguous()
        page_table = page_table.view(-1)
    else:
        total_k = k.shape[0]
        MAX_PAGES = 1
        MAXB = ceildiv(total_k, BLK)
        kv_meta = cu_k
        page_table = torch.zeros(1, dtype=torch.int32, device=device)
    CHUNK = 512 if KV_FP8 else (4096 if PAGED and TOPK <= 4 else 2048)   # finer FP8 items: measured tail reduction
    MAXC = ceildiv(total_q, CHUNK)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    rows_total = total_q * HQ
                                                                              
                                                                     
    SPLIT_FP8 = KV_FP8 and not PAGED and TOPK <= 16
    SPLIT_PAGED_BF16 = PAGED and not KV_FP8 and TOPK <= 4
    SPLIT_COMBINE = SPLIT_FP8 or SPLIT_PAGED_BF16
    NUM_ITEMS = MAXB * HKV * B * MAXC
    NUM_CTAS = max(1, min(num_sms, NUM_ITEMS))
    cfg = dict(
        GQA=GQA, TOPK=TOPK, HKV=HKV, B=B, TOTAL_Q=total_q, MAXB=MAXB, MAXC=MAXC,
        PAGED=PAGED, MAX_PAGES=MAX_PAGES, KV_FP8=KV_FP8, NUM_CTAS=NUM_CTAS, CHUNK=CHUNK,
        FUSED_COMBINE=not SPLIT_COMBINE,
        PACK_FP8_PARTIAL=KV_FP8 or SPLIT_PAGED_BF16,
        DYNAMIC_CHUNKS=SPLIT_COMBINE,
        PACKED_TOKEN_LIST=SPLIT_COMBINE,
    )
    key = ("kv",) + tuple(sorted(cfg.items()))
    executable = _compile_kv(key, **cfg)
    combine_executable = None
    if SPLIT_COMBINE:
                                                                            
                                                       
        rows_per_combine_cta = 48 if KV_FP8 else 24
        resident_combine_ctas = 3 if KV_FP8 else 4
        NUM_COMBINE_CTAS = max(
            1,
            min(
                ceildiv(rows_total, rows_per_combine_cta),
                num_sms * resident_combine_ctas,
            ),
        )
        combine_cfg = dict(
            GQA=GQA, TOPK=TOPK, HKV=HKV, B=B, TOTAL_Q=total_q, MAXB=MAXB,
            NUM_CTAS=NUM_COMBINE_CTAS, PAGED=PAGED,
        )
        combine_key = ("combine_packed_halfwarp",) + tuple(sorted(combine_cfg.items()))
        combine_executable = _compile_combine_packed_halfwarp(
            combine_key, **combine_cfg
        )

                 
    q_flat = q.view(-1)
    q_map = _encode(
        q,
        "bfloat16",
        (HEAD_DIM // 2, HQ, total_q, 2),
        (HEAD_DIM * 2, HQ * HEAD_DIM * 2, (HEAD_DIM // 2) * 2),
        (HEAD_DIM // 2, GQA, 1, 1),
    )
    if KV_FP8:
        if PAGED:
            rows = num_pages * HKV * BLK
            k_map = _encode(k, "float8_e4m3fn", (HEAD_DIM, rows), (HEAD_DIM,), (HEAD_DIM, BLK))
            v_map = _encode(v, "float8_e4m3fn", (HEAD_DIM, rows), (HEAD_DIM,), (HEAD_DIM, BLK))
        else:
            k_map = _encode(
                k, "float8_e4m3fn", (HEAD_DIM, total_k, HKV), (HKV * HEAD_DIM, HEAD_DIM), (HEAD_DIM, BLK, 1)
            )
            v_map = _encode(
                v, "float8_e4m3fn", (HEAD_DIM, total_k, HKV), (HKV * HEAD_DIM, HEAD_DIM), (HEAD_DIM, BLK, 1)
            )
    else:
        if PAGED:
            rows = num_pages * HKV * BLK
            k_map = _encode(
                k, "bfloat16", (HEAD_DIM // 2, rows, 2), (HEAD_DIM * 2, (HEAD_DIM // 2) * 2), (HEAD_DIM // 2, BLK, 2)
            )
            v_map = _encode(
                v, "bfloat16", (HEAD_DIM // 2, rows, 2), (HEAD_DIM * 2, (HEAD_DIM // 2) * 2), (HEAD_DIM // 2, BLK, 2)
            )
        else:
            k_map = _encode(
                k, "bfloat16", (HEAD_DIM // 2, total_k, HKV * 2), (HKV * HEAD_DIM * 2, (HEAD_DIM // 2) * 2),
                (HEAD_DIM // 2, BLK, 2),
            )
            v_map = _encode(
                v, "bfloat16", (HEAD_DIM // 2, total_k, HKV * 2), (HKV * HEAD_DIM * 2, (HEAD_DIM // 2) * 2),
                (HEAD_DIM // 2, BLK, 2),
            )
                                                                            
                                                                              
    pack_fp8_partial = KV_FP8 or SPLIT_PAGED_BF16
    if KV_FP8:
        o_part_elems = rows_total * TOPK * (HEAD_DIM // 4)
    else:
        o_part_elems = rows_total * TOPK * (
            HEAD_DIM // 2 if pack_fp8_partial else HEAD_DIM
        )
    o_part = torch.empty(o_part_elems, dtype=torch.bfloat16, device=device)
    lse_part = torch.empty(rows_total * TOPK, dtype=torch.float32, device=device)
    NCHUNK = B * MAXC * HKV
    sched = torch.zeros(2 + 2 * NCHUNK, dtype=torch.int32, device=device)
    scale_log2 = float(data["softmax_scale"]) * LOG2E
    q2k_flat = q2k.view(-1)
    out_flat = out.view(-1)
    args = (
        q_flat, q_map.ptr, k_map.ptr, v_map.ptr, q2k_flat, cu_q, kv_meta, page_table,
        o_part, lse_part, out_flat, sched, scale_log2,
    )
    combine_args = (q2k_flat, cu_q, kv_meta, o_part, lse_part, out_flat)
    keep = (
        q, k, v, q2k, cu_q, kv_meta, page_table, o_part, lse_part, out, sched,
        q_map, k_map, v_map,
    )

    def run(_args=args, _combine_args=combine_args, _keep=keep,
            _exe=executable, _combine=combine_executable):
        _exe(*_args)
        if _combine is not None:
            _combine(*_combine_args)

    run()          
    torch.cuda.synchronize(device)
    return run


                                                                       
                     
                                                                       


def _compile_any(key, maker, **cfg):
    if key not in _KERNEL_CACHE:
        kernel = maker(**cfg)
        if os.environ.get("MSA_KV_IKET"):
            from tvm.tirx.cuda import iket as _iket

            _KERNEL_CACHE[key] = _iket.IketProfiler().compile(
                kernel.mod, target=kernel.target(), tir_pipeline="tirx"
            )
        else:
            _KERNEL_CACHE[key] = kernel.compile()
    return _KERNEL_CACHE[key]


def setup_kv2(data, total_q, B):
    """CSR build -> kv-major v2 producer -> standalone packed combine."""
    q = data["q"]; k = data["k"]; v = data["v"]
    q2k = data["q2k_indices"]; cu_q = data["cu_seqlens_q"]; cu_k = data["cu_seqlens_k"]
    page_table = data["page_table"]; seqused_k = data["seqused_k"]; out = data["output"]
    device = q.device
    assert q.dtype == torch.bfloat16, "this candidate implements bf16 queries"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous()
    assert q2k.is_contiguous() and q2k.dtype == torch.int32
    HQ = q.shape[1]; HKV = k.shape[1]; GQA = HQ // HKV; TOPK = q2k.shape[2]
    PAGED = page_table is not None
    KV_FP8 = k.dtype == torch.float8_e4m3fn
    if not KV_FP8:
        assert k.dtype == torch.bfloat16
    if PAGED:
        num_pages = k.shape[0]; MAX_PAGES = page_table.shape[1]; MAXB = MAX_PAGES
        kv_meta = seqused_k
        assert page_table.is_contiguous() and seqused_k.is_contiguous()
        page_table = page_table.view(-1)
    else:
        total_k = k.shape[0]; MAX_PAGES = 1; MAXB = ceildiv(total_k, BLK)
        kv_meta = cu_k
        page_table = torch.zeros(1, dtype=torch.int32, device=device)
    assert TOPK <= 256
    NGROUPS = B * HKV * MAXB
    CAP = ceildiv(total_q, 128) * 128
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    rows_total = total_q * HQ
    PARTIAL_FMT = "e2m1" if KV_FP8 else "e4m3"
    TPI = int(os.environ.get("MSA_TPI", 8 if KV_FP8 else 4))
    Q_STAGES = int(os.environ.get("MSA_QSTAGES", 2 if KV_FP8 else 3))
    NUM_CTAS = int(os.environ.get("MSA_NUM_CTAS", num_sms))
    cfg = dict(
        GQA=GQA, TOPK=TOPK, HKV=HKV, B=B, TOTAL_Q=total_q, MAXB=MAXB, CAP=CAP, PAGED=PAGED,
        MAX_PAGES=MAX_PAGES, KV_FP8=KV_FP8, NUM_CTAS=NUM_CTAS, TPI=TPI, Q_STAGES=Q_STAGES,
        PARTIAL_FMT=PARTIAL_FMT,
        SM_REGS=int(os.environ.get("MSA_SM_REGS", 184)),
        AUX_REGS=int(os.environ.get("MSA_AUX_REGS", 104 if KV_FP8 else 96)),
    )
    executable = _compile_any(("kv2",) + tuple(sorted(cfg.items())), make_kernel_kv2, **cfg)
    csr_cfg = dict(TOPK=TOPK, HKV=HKV, B=B, TOTAL_Q=total_q, MAXB=MAXB, CAP=CAP, PAGED=PAGED,
                   NUM_CTAS=ceildiv(HKV * total_q, 256))
    csr_executable = _compile_any(("csr",) + tuple(sorted(csr_cfg.items())), make_kernel_csr, **csr_cfg)
    rows_per_combine_cta = 48 if KV_FP8 else 24
    resident_combine_ctas = 3 if KV_FP8 else 4
    NUM_COMBINE_CTAS = max(1, min(ceildiv(rows_total, rows_per_combine_cta), num_sms * resident_combine_ctas))
    combine_cfg = dict(GQA=GQA, TOPK=TOPK, HKV=HKV, B=B, TOTAL_Q=total_q, MAXB=MAXB,
                       NUM_CTAS=NUM_COMBINE_CTAS, PAGED=PAGED)
    combine_executable = _compile_any(
        ("combine_packed_halfwarp",) + tuple(sorted(combine_cfg.items())),
        make_kernel_combine_packed_halfwarp, **combine_cfg,
    )

    q_flat = q.view(-1)
    q_map = _encode(
        q, "bfloat16",
        (HEAD_DIM // 2, HQ, total_q, 2),
        (HEAD_DIM * 2, HQ * HEAD_DIM * 2, (HEAD_DIM // 2) * 2),
        (HEAD_DIM // 2, GQA, 1, 1),
    )
    if KV_FP8:
        if PAGED:
            rows = num_pages * HKV * BLK
            k_map = _encode(k, "float8_e4m3fn", (HEAD_DIM, rows), (HEAD_DIM,), (HEAD_DIM, BLK))
            v_map = _encode(v, "float8_e4m3fn", (HEAD_DIM, rows), (HEAD_DIM,), (HEAD_DIM, BLK))
        else:
            k_map = _encode(k, "float8_e4m3fn", (HEAD_DIM, total_k, HKV), (HKV * HEAD_DIM, HEAD_DIM), (HEAD_DIM, BLK, 1))
            v_map = _encode(v, "float8_e4m3fn", (HEAD_DIM, total_k, HKV), (HKV * HEAD_DIM, HEAD_DIM), (HEAD_DIM, BLK, 1))
    else:
        if PAGED:
            rows = num_pages * HKV * BLK
            k_map = _encode(k, "bfloat16", (HEAD_DIM // 2, rows, 2), (HEAD_DIM * 2, (HEAD_DIM // 2) * 2), (HEAD_DIM // 2, BLK, 2))
            v_map = _encode(v, "bfloat16", (HEAD_DIM // 2, rows, 2), (HEAD_DIM * 2, (HEAD_DIM // 2) * 2), (HEAD_DIM // 2, BLK, 2))
        else:
            k_map = _encode(k, "bfloat16", (HEAD_DIM // 2, total_k, HKV * 2), (HKV * HEAD_DIM * 2, (HEAD_DIM // 2) * 2), (HEAD_DIM // 2, BLK, 2))
            v_map = _encode(v, "bfloat16", (HEAD_DIM // 2, total_k, HKV * 2), (HKV * HEAD_DIM * 2, (HEAD_DIM // 2) * 2), (HEAD_DIM // 2, BLK, 2))
    o_part_elems = rows_total * TOPK * (HEAD_DIM // 4 if KV_FP8 else HEAD_DIM // 2)
    o_part = torch.empty(o_part_elems, dtype=torch.bfloat16, device=device)
    lse_part = torch.empty(rows_total * TOPK, dtype=torch.float32, device=device)
    csr_count = torch.zeros(NGROUPS, dtype=torch.int32, device=device)
    csr_list = torch.zeros(NGROUPS * CAP, dtype=torch.int32, device=device)
    sched = torch.zeros(4, dtype=torch.int32, device=device)
    scale_log2 = float(data["softmax_scale"]) * LOG2E
    q2k_flat = q2k.view(-1)
    out_flat = out.view(-1)
    csr_args = (q2k_flat, cu_q, kv_meta, csr_count, csr_list)
    args = (
        q_flat, q_map.ptr, k_map.ptr, v_map.ptr, cu_q, kv_meta, page_table, csr_count, csr_list,
        o_part, lse_part, sched, scale_log2,
    )
    combine_args = (q2k_flat, cu_q, kv_meta, o_part, lse_part, out_flat)
    keep = (q, k, v, q2k, cu_q, kv_meta, page_table, o_part, lse_part, out, sched, csr_count, csr_list,
            q_map, k_map, v_map)

    def run_eager(_csr_args=csr_args, _args=args, _combine_args=combine_args, _keep=keep,
                  _csr=csr_executable, _exe=executable, _combine=combine_executable):
        _csr(*_csr_args)
        _exe(*_args)
        _combine(*_combine_args)

    run_eager()
    torch.cuda.synchronize(device)
    if not int(os.environ.get("MSA_GRAPH", "1")):
        return run_eager
    # Capture the three Kern launches in one CUDA graph (as the packaged baseline does for its
    # kernels): every timed call replays exactly these kernels on exactly these buffers, so it
    # consumes the current selection metadata and overwrites the output; only the host-side
    # launch gaps between the short CSR kernel and the producer disappear.
    import tvm_ffi
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with tvm_ffi.use_torch_stream(torch.cuda.stream(side)):
        run_eager()
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with tvm_ffi.use_torch_stream(torch.cuda.graph(graph)):
        run_eager()
    torch.cuda.synchronize(device)

    def run(_graph=graph, _keep=keep, _eager=run_eager):
        _graph.replay()

    run()
    torch.cuda.synchronize(device)
    return run


KV_DEPTH = 4
MAXB_QM = 64
N_TILES = 2
RESCALE_THRESHOLD = 8.0

TMA_G2S_3D = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
TMA_G2S_4D = (
    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
TMEM_LD_32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TMEM_LD_16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
TMEM_LD_64 = "tcgen05.ld.sync.aligned.32x32b.x64.b32"
TMEM_ST_16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
ID_QK = 0x08200490
ID_PV = 0x08210490
N_COLS_TMEM = 512


def ceildiv(a, b):
    return (a + b - 1) // b


def make_kernel_qm(*, GQA, TOPK, HKV, B, TOTAL_Q, MAXG, PAGED, MAX_PAGES, NUM_CTAS):
    HQ = HKV * GQA
    assert GQA in (8, 16, 32, 64, 128)
    TPT = 128 // GQA                                  
    G = N_TILES * TPT                                 
    NUM_TASKS = B * HKV * MAXG
    Q_TILE_BYTES = 128 * HEAD_DIM * 2
    KV_TILE_BYTES = 128 * HEAD_DIM * 2
    TOPK_V4 = TOPK % 4 == 0
    META_N = 16
    NSLOT = 2
    DIAG = os.environ.get("MSA_QM_DIAG", "")
    # ---- v2 MMA/softmax pipeline constants (ported from the canonical
    # ---- qmajor-persistent kernel: cross-aliased P, split P publish, XU
    # ---- turn ping-pong, partial exp2 emulation) ----
    K_SPLIT = 4 * MMA_K                 # PV part 1 consumes P's first 64 keys (4 MMA_K steps)
    P_SPLIT_FRAGS = 2                   # 32-column fragments published in the first P half
    N_FRAGS = BLK // 32
    # 4 packed-f32x2 row-sum accumulators (the canonical kernel uses 8): with the separated
    # exp / convert passes, 8 accumulators push the 216-register softmax role into a 6-word
    # spill; 4 compile spill-free with the exp2 emulation kept.
    N_SUM_ACC = int(os.environ.get("MSA_QM_SUM_ACC", "4"))
    MAX_CHAINS = int(os.environ.get("MSA_QM_MAX_CHAINS", "16"))  # independent row-max chains (8 or 16)
    assert N_SUM_ACC in (4, 8) and MAX_CHAINS in (8, 16)
    USE_EMU = os.environ.get("MSA_QM_EMU", "0") != "0"
    EMU_PAIRS = 0                       # all-native exp2: the exp ping-pong already spreads MUFU across the two warpgroups, so FMA-pipe emulation only adds critical-path work here
    EMU_START = 0
    POLY_EX2_DEG1 = (1.0290300065, 0.6860200044)
    FP32_ROUND_INT = float(2**23 + 2**22)

    @K.kernel(warps=16, arch="sm_100a", min_blocks_per_sm=1, grid=NUM_CTAS)
    def msa_qmajor(
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        q2k: K.gptr[K.i32],
        cu_q: K.gptr[K.i32],
        kv_meta: K.gptr[K.i32],
        page_table: K.gptr[K.i32],
        out: K.gptr[K.bf16],
        sched: K.gptr[K.i32],
        scale_log2: K.f32,
    ):
        cta = K.cta_id()
        warp_cta = K.warp_id()
        wg_id = warp_cta >> 2
        warp_in_wg = warp_cta & 3
        tid_in_wg = K.thread_id() & 127
        lane = K.lane_id()

                                                         
        smem = K.smem_pool()
        q_smem = smem.alloc((N_TILES, 128, HEAD_DIM), K.bf16, swizzle=K.SW128B)
        kv_smem = smem.alloc((KV_DEPTH, 128, HEAD_DIM), K.bf16, swizzle=K.SW128B)
        blk_list = smem.alloc((NSLOT * MAXB_QM,), K.i32)
        tok_mask = smem.alloc((NSLOT * G * 2,), K.u32)                         
                                                                                  
                                                                               
                                                          
        lane_mask = smem.alloc((NSLOT * MAXB_QM * N_TILES * 4,), K.u32)
        meta = smem.alloc((NSLOT * META_N,), K.i32)
        wscratch = smem.alloc((16,), K.u32)
        tmem_addr = smem.alloc((1,), K.u32)

        def stage16(tile):
            return tile.rows * tile.cols * tile.bits // 8 // 16

        Q_STAGE16 = stage16(q_smem)
        KV_STAGE16 = stage16(kv_smem)

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

                                                    
        list_full = K.MBarrier(smem, NSLOT)
        list_full.init(128)
        list_free = K.MBarrier(smem, NSLOT)
        list_free.init(256 + 32 + 32)
        q_load = K.Pipeline(smem, N_TILES, full="tma", empty="tcgen05", empty_phase_offset=1)
        kv_load = K.Pipeline(smem, KV_DEPTH, full="tma", empty="tcgen05", empty_phase_offset=1)
        # TMEM: S_i = cols [i*128, i*128+128); bf16 P_i = cols [(1-i)*128+64, (1-i)*128+128)
        # (upper half of the OTHER tile's S region); O_i = cols [256+i*128, 256+i*128+128).
        s_full = K.TCGen05Bar(smem, N_TILES)          # QK_i committed: S_i(n) readable
        s_full.init(1)
        s_consumed = K.MBarrier(smem, N_TILES)        # WG i copied S_i(n) to registers
        s_consumed.init(128)
        p_o_rescale = K.MBarrier(smem, N_TILES)       # WG i: O_i rescale done + first P_i half stored
        p_o_rescale.init(128)
        p_ready_2 = K.MBarrier(smem, N_TILES)         # WG i: second P_i half stored
        p_ready_2.init(128)
        pv_done = K.TCGen05Bar(smem, N_TILES)         # PV_i committed: O_i stable, P_i region free
        pv_done.init(1)
        xu_turn = K.MBarrier(smem, N_TILES)           # exp-phase (MUFU) ping-pong token per WG
        xu_turn.init(128)

        K.ptx.fence.proxy.async_.shared__cta()
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

                                                   
        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def tmem(col):
            return K.cuda.get_tmem_addr(K.uint32(0), 0, col)

        def tmem_load(dst, dst_offset, tmem_col, width):
            chain = TMEM_LD_16 if width == 16 else (TMEM_LD_64 if width == 64 else TMEM_LD_32)
            K.ptx[chain](*(dst[dst_offset + i] for i in range(width)), tmem_col)

        def tmem_store16(src, src_offset, tmem_col):
            K.ptx[TMEM_ST_16](tmem_col, *(src[src_offset + i] for i in range(16)))

        def ld_shared_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.shared.b32(value, ptr)
            return value

        def ld_shared_u32(ptr):
            value = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(value, ptr)
            return value

        def ld_global_i32(ptr):
            value = K.local_scalar("int32")
            K.ptx.ld.global_.nc.b32(value, ptr)
            return value

        def iket_range(name, *, leader_only=False):
            token = K.alloc_local([1], "uint32")
            if leader_only:
                K.assign(token[0], K.cuda.iket.sentinel_token(name))
                with K.If(warp_in_wg == 0), K.Then():
                    K.assign(token[0], K.cuda.iket.range_start(name))
            else:
                K.assign(token[0], K.cuda.iket.range_start(name))
            return token

        def iket_end(token):
            K.cuda.iket.range_end(token[0])

        def meta_ptr(slot, idx):
            return meta.ptr_to([slot * META_N + idx])

        def lane_mask_ptr(slot, list_idx, tile, word):
            return lane_mask.ptr_to([((slot * MAXB_QM + list_idx) * N_TILES + tile) * 4 + word])

        def ring_advance(slot, use):
            K.assign(slot, slot + 1)
            with K.If(slot == NSLOT), K.Then():
                K.assign(slot, 0)
                K.assign(use, use + 1)

        def read_meta(slot):
            return [ld_shared_i32(meta_ptr(slot, i)) for i in range(10)]

        def task_decode(task):
                                                                                             
                                                                                              
                                                                                                
            g_rev = task // (B * HKV)
            t1 = task % (B * HKV)
            h = t1 % HKV
            s = t1 // HKV
            return s, h, (MAXG - 1) - g_rev

        def seq_lengths(s):
            q0 = ld_global_i32(cu_q.ptr_to([s]))
            q1 = ld_global_i32(cu_q.ptr_to([s + 1]))
            if PAGED:
                kv_len = ld_global_i32(kv_meta.ptr_to([s]))
                k_base = K.int32(0)
            else:
                k0 = ld_global_i32(kv_meta.ptr_to([s]))
                k1 = ld_global_i32(kv_meta.ptr_to([s + 1]))
                kv_len = k1 - k0
                k_base = k0
            return q0, q1 - q0, kv_len, k_base

        def cast_f32x2_bf16x2(dst_u32, src, offset):
            K.ptx.cvt.rn.bf16x2.f32(dst_u32[offset // 2], src[offset + 1], src[offset])

                                                 
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

        def ex2_emulation_2(out_, idx, x, y):
            """Packed-f32x2 polynomial exp2 for two values (canonical ``ex2_emulation_2``)."""
            xy_clamped = K.alloc_local([2], "float32")
            K.ptx.mov.b32(xy_clamped[0], K.max(x, -127.0))
            K.ptx.mov.b32(xy_clamped[1], K.max(y, -127.0))
            packed = K.local_scalar("uint64")
            rhs = K.local_scalar("uint64")
            addend = K.local_scalar("uint64")
            xy_rounded = K.alloc_local([2], "float32")
            K.ptx.mov.b64(packed, xy_clamped[0], xy_clamped[1])
            K.ptx.mov.b64(rhs, K.float32(FP32_ROUND_INT), K.float32(FP32_ROUND_INT))
            K.ptx.add.rn.ftz.f32x2(packed, packed, rhs)   # round-to-nearest: frac in [-0.5, 0.5], matching the fit
            K.ptx.mov.b64(xy_rounded[0], xy_rounded[1], packed)
            xy_rounded_back = K.alloc_local([2], "float32")
            K.ptx.mov.b64(packed, xy_rounded[0], xy_rounded[1])
            K.ptx.mov.b64(rhs, K.float32(FP32_ROUND_INT), K.float32(FP32_ROUND_INT))
            K.ptx.sub.rn.ftz.f32x2(packed, packed, rhs)
            K.ptx.mov.b64(xy_rounded_back[0], xy_rounded_back[1], packed)
            xy_frac = K.alloc_local([2], "float32")
            K.ptx.mov.b64(packed, xy_clamped[0], xy_clamped[1])
            K.ptx.mov.b64(rhs, xy_rounded_back[0], xy_rounded_back[1])
            K.ptx.sub.rn.ftz.f32x2(packed, packed, rhs)
            K.ptx.mov.b64(xy_frac[0], xy_frac[1], packed)
            xy_frac_ex2 = K.alloc_local([2], "float32")
            K.ptx.mov.b32(xy_frac_ex2[0], K.float32(POLY_EX2_DEG1[1]))
            K.ptx.mov.b32(xy_frac_ex2[1], K.float32(POLY_EX2_DEG1[1]))
            for coeff in (POLY_EX2_DEG1[0],):
                K.ptx.mov.b64(rhs, xy_frac[0], xy_frac[1])
                K.ptx.mov.b64(packed, xy_frac_ex2[0], xy_frac_ex2[1])
                K.ptx.mov.b64(addend, K.float32(coeff), K.float32(coeff))
                K.ptx.fma.rz.ftz.f32x2(packed, packed, rhs, addend)
                K.ptx.mov.b64(xy_frac_ex2[0], xy_frac_ex2[1], packed)
            K.ptx.mov.b32(out_[idx], combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0]))
            K.ptx.mov.b32(out_[idx + 1], combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1]))

        sp = K.specialize(chain_dispatch=True)
        r_sm0 = sp.role("softmax0", warps=[0, 1, 2, 3], regs=216)
        r_sm1 = sp.role("softmax1", warps=[4, 5, 6, 7], regs=216)
        r_aux = sp.role("aux", warps=[8, 9, 10, 11], regs=40)
        wg3 = sp.warpgroup("wg3", warps=range(12, 16), regs=40)
        r_mma = sp.role("mma", warps=[12], group=wg3)
        r_load = sp.role("load", warps=[13], group=wg3)
        r_idle = sp.role("idle", warps=[14, 15], group=wg3)

        with K.If(warp_cta == 12), K.Then():
            K.ptx[TMEM_ALLOC](K.address_of(tmem_addr[0]), K.uint32(N_COLS_TMEM))
            K.ptx[TMEM_RELINQUISH]()
            K.cuda.warp_sync()
        K.cuda.cta_sync()
        with K.If(K.thread_id() == 0), K.Then():
            allocated = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(allocated, tmem_addr.ptr_to([0]))
            K.cuda.trap_when_assert_failed(allocated == K.uint32(0))

        def role_tail(kind):
            K.cuda.cta_sync()
            if kind == "mma":
                dealloc = K.local_scalar("uint32")
                K.ptx.ld.shared.u32(dealloc, tmem_addr.ptr_to([0]))
                K.ptx[TMEM_DEALLOC](dealloc, K.uint32(N_COLS_TMEM))
            if kind == "sm0":
                with K.If(K.thread_id() == 0), K.Then():
                    done = K.local_scalar("int32")
                    K.ptx.atom.acq_rel.gpu.global_.add.s32(done, sched.ptr_to([1]), K.int32(1))
                    with K.If(done == NUM_CTAS - 1), K.Then():
                        K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([0]), K.int32(0))
                        K.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([1]), K.int32(0))

                                                                               
                                                                       
                                                                               
        with r_aux:
            slot = K.local_scalar("int32", init=0)
            use_a = K.local_scalar("int32", init=0)
            running_a = K.local_scalar("int32", init=1)

            def store_lane_mask(list_idx, blk_value, valid_entry):
                """Transpose per-token selection bits into tcgen05 lane masks."""
                blk_safe = K.max(blk_value, 0)
                for tile in range(N_TILES):
                    for word in range(4):
                        disabled = K.local_scalar("uint32", init=K.uint32(0))
                        word_lo = word * 32
                        word_hi = word_lo + 32
                        for t in range(TPT):
                            lane_lo = t * GQA
                            lane_hi = lane_lo + GQA
                            overlap_lo = max(lane_lo, word_lo)
                            overlap_hi = min(lane_hi, word_hi)
                            if overlap_lo < overlap_hi:
                                tok = tile * TPT + t
                                tm_lo = ld_shared_u32(tok_mask.ptr_to([(slot * G + tok) * 2]))
                                tm_hi = ld_shared_u32(tok_mask.ptr_to([(slot * G + tok) * 2 + 1]))
                                selected = K.Select(
                                    blk_value < 32,
                                    K.bitwise_and(
                                        K.shift_right(tm_lo, K.Cast("uint32", K.bitwise_and(blk_safe, 31))),
                                        K.uint32(1),
                                    ),
                                    K.bitwise_and(
                                        K.shift_right(tm_hi, K.Cast("uint32", K.bitwise_and(blk_safe, 31))),
                                        K.uint32(1),
                                    ),
                                ) != K.uint32(0)
                                width = overlap_hi - overlap_lo
                                bits = ((1 << width) - 1) << (overlap_lo - word_lo)
                                K.assign(
                                    disabled,
                                    K.bitwise_or(
                                        disabled,
                                        K.Select(K.And(valid_entry, selected), K.uint32(0), K.uint32(bits)),
                                    ),
                                )
                        K.ptx.st.shared.b32(lane_mask_ptr(slot, list_idx, tile, word), disabled)

            with K.While(running_a != 0):
                with K.If(tid_in_wg == 0), K.Then():
                    grabbed = K.local_scalar("int32")
                    K.ptx.atom.relaxed.gpu.global_.add.s32(grabbed, sched.ptr_to([0]), K.int32(1))
                    K.ptx.st.shared.b32(wscratch.ptr_to([8]), K.Cast("uint32", grabbed))
                K.cuda.warpgroup_sync(1)
                task = K.Cast("int32", ld_shared_u32(wscratch.ptr_to([8])))
                with K.If(task >= NUM_TASKS):
                    with K.Then():
                        list_free.wait(slot, (use_a + 1) & 1)
                        with K.If(tid_in_wg == 0), K.Then():
                            K.ptx.st.shared.b32(meta_ptr(slot, 0), K.int32(-1))
                        list_full.arrive(slot)
                        K.assign(running_a, 0)
                    with K.Else():
                        s, h, g = task_decode(task)
                        q_base, q_len, kv_len, k_base = seq_lengths(s)
                        tok0 = g * G
                        nblocks = K.min((kv_len + (BLK - 1)) // BLK, MAXB_QM)
                        with K.If(tok0 < q_len), K.Then():
                            list_free.wait(slot, (use_a + 1) & 1)
                            tk_u = iket_range("aux-union", leader_only=True)
                            n_valid_tok = K.min(q_len - tok0, G)
                            causal_off = kv_len - q_len
                                                                                                  
                            m_lo = K.local_scalar("uint32", init=K.uint32(0))
                            m_hi = K.local_scalar("uint32", init=K.uint32(0))
                            with K.If(tid_in_wg < n_valid_tok), K.Then():
                                t_glob = q_base + tok0 + tid_in_wg
                                pos = causal_off + tok0 + tid_in_wg
                                row_base = (h * TOTAL_Q + t_glob) * TOPK
                                idxs = K.alloc_local([TOPK], "int32")
                                if TOPK_V4:
                                    for v in range(TOPK // 4):
                                        K.ptx.ld.global_.nc.v4.b32(
                                            idxs[4 * v], idxs[4 * v + 1], idxs[4 * v + 2], idxs[4 * v + 3],
                                            q2k.ptr_to([row_base + 4 * v]),
                                        )
                                else:
                                    for v in range(TOPK):
                                        K.ptx.ld.global_.nc.b32(idxs[v], q2k.ptr_to([row_base + v]))
                                for v in range(TOPK):
                                    bk = idxs[v]
                                    ok = K.And(K.And(bk >= 0, bk < nblocks), bk * BLK <= pos)
                                    bit = K.shift_left(K.uint32(1), K.Cast("uint32", K.bitwise_and(K.max(bk, 0), 31)))
                                    K.assign(m_lo, K.Select(K.And(ok, bk < 32), K.bitwise_or(m_lo, bit), m_lo))
                                    K.assign(m_hi, K.Select(K.And(ok, bk >= 32), K.bitwise_or(m_hi, bit), m_hi))
                            with K.If(tid_in_wg < G), K.Then():
                                K.ptx.st.shared.b32(tok_mask.ptr_to([(slot * G + tid_in_wg) * 2]), m_lo)
                                K.ptx.st.shared.b32(tok_mask.ptr_to([(slot * G + tid_in_wg) * 2 + 1]), m_hi)
                                                        
                            u_lo = K.local_scalar("uint32")
                            u_hi = K.local_scalar("uint32")
                            K.ptx.redux_sync.or_.b32(u_lo, m_lo, K.uint32(0xFFFFFFFF))
                            K.ptx.redux_sync.or_.b32(u_hi, m_hi, K.uint32(0xFFFFFFFF))
                            with K.If(lane == 0), K.Then():
                                K.ptx.st.shared.b32(wscratch.ptr_to([warp_in_wg * 2]), u_lo)
                                K.ptx.st.shared.b32(wscratch.ptr_to([warp_in_wg * 2 + 1]), u_hi)
                            K.cuda.warpgroup_sync(1)
                            for w4 in range(4):
                                K.assign(u_lo, K.bitwise_or(u_lo, ld_shared_u32(wscratch.ptr_to([w4 * 2]))))
                                K.assign(u_hi, K.bitwise_or(u_hi, ld_shared_u32(wscratch.ptr_to([w4 * 2 + 1]))))
                            cnt_lo = K.local_scalar("uint32")
                            cnt_hi = K.local_scalar("uint32")
                            K.ptx.popc.b32(cnt_lo, u_lo)
                            K.ptx.popc.b32(cnt_hi, u_hi)
                            n_blocks = K.Cast("int32", cnt_lo + cnt_hi)
                                                                                          
                            with K.If(warp_in_wg == 0), K.Then():
                                pos_i = K.local_scalar("int32", init=lane)
                                blk_val = K.local_scalar("int32", init=K.int32(-1))
                                with K.If(K.Cast("uint32", lane) < cnt_hi), K.Then():
                                    bv = K.local_scalar("uint32")
                                    K.ptx.fns.b32(bv, u_hi, K.uint32(31), -K.Cast("int32", lane) - K.int32(1))
                                    K.assign(blk_val, K.Cast("int32", bv) + 32)
                                with K.If(K.And(K.Cast("uint32", lane) >= cnt_hi, K.Cast("uint32", lane) < cnt_lo + cnt_hi)), K.Then():
                                    bv = K.local_scalar("uint32")
                                    K.ptx.fns.b32(bv, u_lo, K.uint32(31), -(K.Cast("int32", lane) - K.Cast("int32", cnt_hi)) - K.int32(1))
                                    K.assign(blk_val, K.Cast("int32", bv))
                                with K.If(lane < 32), K.Then():
                                    K.ptx.st.shared.b32(blk_list.ptr_to([slot * MAXB_QM + lane]), blk_val)
                                store_lane_mask(lane, blk_val, K.Cast("uint32", lane) < cnt_lo + cnt_hi)
                                                                    
                                blk_val2 = K.local_scalar("int32", init=K.int32(-1))
                                r2 = lane + 32
                                with K.If(K.Cast("uint32", r2) < cnt_hi), K.Then():
                                    bv = K.local_scalar("uint32")
                                    K.ptx.fns.b32(bv, u_hi, K.uint32(31), -r2 - K.int32(1))
                                    K.assign(blk_val2, K.Cast("int32", bv) + 32)
                                with K.If(K.And(K.Cast("uint32", r2) >= cnt_hi, K.Cast("uint32", r2) < cnt_lo + cnt_hi)), K.Then():
                                    bv = K.local_scalar("uint32")
                                    K.ptx.fns.b32(bv, u_lo, K.uint32(31), -(r2 - K.Cast("int32", cnt_hi)) - K.int32(1))
                                    K.assign(blk_val2, K.Cast("int32", bv))
                                K.ptx.st.shared.b32(blk_list.ptr_to([slot * MAXB_QM + lane + 32]), blk_val2)
                                store_lane_mask(
                                    lane + 32,
                                    blk_val2,
                                    K.Cast("uint32", lane + 32) < cnt_lo + cnt_hi,
                                )
                                                                                                   
                            pos_min = causal_off + tok0
                            b_first_masked = K.max(pos_min // BLK, 0)                                       
                            masked_bits_lo = K.local_scalar("uint32", init=K.uint32(0))
                            masked_bits_hi = K.local_scalar("uint32", init=K.uint32(0))
                            with K.If(b_first_masked < 32), K.Then():
                                K.assign(masked_bits_lo, K.bitwise_and(u_lo, K.bitwise_not(K.shift_left(K.uint32(1), K.Cast("uint32", b_first_masked)) - K.uint32(1))))
                                K.assign(masked_bits_hi, u_hi)
                            with K.If(K.And(b_first_masked >= 32, b_first_masked < 64)), K.Then():
                                K.assign(masked_bits_hi, K.bitwise_and(u_hi, K.bitwise_not(K.shift_left(K.uint32(1), K.Cast("uint32", b_first_masked - 32)) - K.uint32(1))))
                            nm_lo = K.local_scalar("uint32")
                            nm_hi = K.local_scalar("uint32")
                            K.ptx.popc.b32(nm_lo, masked_bits_lo)
                            K.ptx.popc.b32(nm_hi, masked_bits_hi)
                            n_masked = K.Cast("int32", nm_lo + nm_hi)
                            with K.If(tid_in_wg == 0), K.Then():
                                K.ptx.st.shared.b32(meta_ptr(slot, 0), n_blocks)
                                K.ptx.st.shared.b32(meta_ptr(slot, 1), n_masked)
                                K.ptx.st.shared.b32(meta_ptr(slot, 2), tok0)
                                K.ptx.st.shared.b32(meta_ptr(slot, 3), h)
                                K.ptx.st.shared.b32(meta_ptr(slot, 4), s)
                                K.ptx.st.shared.b32(meta_ptr(slot, 5), q_base)
                                K.ptx.st.shared.b32(meta_ptr(slot, 6), kv_len)
                                K.ptx.st.shared.b32(meta_ptr(slot, 7), q_len)
                                K.ptx.st.shared.b32(meta_ptr(slot, 8), n_valid_tok)
                                K.ptx.st.shared.b32(meta_ptr(slot, 9), k_base)
                            iket_end(tk_u)
                            list_full.arrive(slot)
                            ring_advance(slot, use_a)
            role_tail("aux")

                                                                               
                                                           
                                                                               
        with wg3:
            with r_load:
                slot = K.local_scalar("int32", init=0)
                use_l = K.local_scalar("int32", init=0)
                running_l = K.local_scalar("int32", init=1)
                kv_pipe = K.PipelineState(KV_DEPTH, phase=0)
                q_pipe = K.PipelineState(1, phase=0)
                with K.While(running_l != 0):
                    list_full.wait(slot, use_l & 1)
                    m = read_meta(slot)
                    n_blocks, n_masked, tok0, h, s, q_base, kv_len, q_len, n_valid_tok, k_base = m
                    with K.If(n_blocks < 0), K.Then():
                        K.assign(running_l, 0)
                    with K.If(n_blocks >= 0), K.Then():
                                                                                           
                        for i_q in range(N_TILES):
                            q_load.empty.wait(i_q, q_pipe.phase)
                            with K.If(elected()), K.Then():
                                K.ptx[TMA_G2S_4D](
                                    q_smem[i_q].ptr_to(0, 0),
                                    K.address_of(q_map),
                                    K.int32(0),
                                    h * GQA,
                                    q_base + tok0 + i_q * TPT,
                                    K.int32(0),
                                    K.cuda.cvta_generic_to_shared(q_load.full.ptr_to([i_q])),
                                )
                                q_load.full.arrive(i_q, tx_count=Q_TILE_BYTES)
                        q_pipe.advance()

                        def load_kv(blk, tmap):
                            kv_load.empty.wait(kv_pipe.stage, kv_pipe.phase)
                            with K.If(elected()), K.Then():
                                if PAGED:
                                    page = ld_global_i32(page_table.ptr_to([s * MAX_PAGES + blk]))
                                    K.ptx[TMA_G2S_3D](
                                        kv_smem[kv_pipe.stage].ptr_to(0, 0),
                                        K.address_of(tmap),
                                        K.int32(0),
                                        (page * HKV + h) * BLK,
                                        K.int32(0),
                                        K.cuda.cvta_generic_to_shared(kv_load.full.ptr_to([kv_pipe.stage])),
                                    )
                                else:
                                    K.ptx[TMA_G2S_3D](
                                        kv_smem[kv_pipe.stage].ptr_to(0, 0),
                                        K.address_of(tmap),
                                        K.int32(0),
                                        k_base + blk * BLK,
                                        h * 2,
                                        K.cuda.cvta_generic_to_shared(kv_load.full.ptr_to([kv_pipe.stage])),
                                    )
                                kv_load.full.arrive(kv_pipe.stage, tx_count=KV_TILE_BYTES)
                            kv_pipe.advance()

                                              
                        with K.If(n_blocks > 0), K.Then():
                            blk_cur = K.local_scalar("int32", init=ld_shared_i32(blk_list.ptr_to([slot * MAXB_QM])))
                            load_kv(blk_cur, k_map)
                            with K.serial(n_blocks, unroll=False) as n:
                                with K.If(n + 1 < n_blocks):
                                    with K.Then():
                                        blk_nxt = ld_shared_i32(blk_list.ptr_to([slot * MAXB_QM + n + 1]))
                                        load_kv(blk_nxt, k_map)
                                        load_kv(blk_cur, v_map)
                                        K.assign(blk_cur, blk_nxt)
                                    with K.Else():
                                        load_kv(blk_cur, v_map)
                    list_free.arrive(slot)
                    ring_advance(slot, use_l)
                role_tail("load")

            with r_mma:
                slot = K.local_scalar("int32", init=0)
                use_m = K.local_scalar("int32", init=0)
                running_m = K.local_scalar("int32", init=1)
                kv_pipe_m = K.PipelineState(KV_DEPTH, phase=0)
                q_pipe_m = K.PipelineState(1, phase=0)
                gstep = K.local_scalar("int32", init=0)       # global KV-block step counter (never reset)
                tb_raw = K.local_scalar("uint32")
                K.ptx.ld.shared.u32(tb_raw, tmem_addr.ptr_to([0]))
                tmem_base = K.local_scalar("uint32", init=K.uniform(tb_raw))
                q_desc, qoff = encode(q_smem[0])
                k_desc, koff = encode(kv_smem[0])
                v_desc, mnoff = encode(kv_smem[0], major="mn")

                def load_lane_mask(slot_, list_idx, i_q):
                    disabled = K.alloc_local([4], "uint32")
                    for word in range(4):
                        K.ptx.ld.shared.u32(disabled[word], lane_mask_ptr(slot_, list_idx, i_q, word))
                    return disabled

                def gemm_qk(i_q, kv_stage, disabled):
                    for ki in range(HEAD_DIM // MMA_K):
                        K.ptx[MMA_F16](
                            tmem_base + K.uint32(i_q * 128),
                            desc_at(q_desc, i_q * Q_STAGE16 + qoff(ki)),
                            desc_at(k_desc, kv_stage * KV_STAGE16 + koff(ki)),
                            K.uint32(ID_QK), disabled[0], disabled[1], disabled[2], disabled[3], ki != 0,
                        )

                def p_operand(i_q, ki):
                    # bf16 P_i lives in the upper 64 columns of the OTHER tile's S region.
                    return tmem_base + K.uint32((1 - i_q) * 128 + 64 + ki * (MMA_K // 2))

                def pv_disabled(accumulate, selected_disabled):
                    # The first (overwriting) PV stays unmasked so every O row is defined.
                    disabled = K.alloc_local([4], "uint32")
                    for word in range(4):
                        K.assign(
                            disabled[word],
                            K.Select(accumulate != 0, selected_disabled[word], K.uint32(0)),
                        )
                    return disabled

                def gemm_pv_part1(i_q, kv_stage, accumulate, disabled):
                    for ki in range(K_SPLIT // MMA_K):
                        K.ptx[MMA_F16](
                            tmem_base + K.uint32(256 + i_q * 128),
                            p_operand(i_q, ki),
                            desc_at(v_desc, kv_stage * KV_STAGE16 + mnoff(ki)),
                            K.uint32(ID_PV), disabled[0], disabled[1], disabled[2], disabled[3],
                            True if ki != 0 else K.Cast("bool", accumulate),
                        )

                def gemm_pv_part2(i_q, kv_stage, disabled):
                    for ki in range(K_SPLIT // MMA_K, BLK // MMA_K):
                        K.ptx[MMA_F16](
                            tmem_base + K.uint32(256 + i_q * 128),
                            p_operand(i_q, ki),
                            desc_at(v_desc, kv_stage * KV_STAGE16 + mnoff(ki)),
                            K.uint32(ID_PV), disabled[0], disabled[1], disabled[2], disabled[3], True,
                        )

                with K.While(running_m != 0):
                    list_full.wait(slot, use_m & 1)
                    n_blocks = ld_shared_i32(meta_ptr(slot, 0))
                    with K.If(n_blocks < 0), K.Then():
                        K.assign(running_m, 0)
                    with K.If(n_blocks == 0), K.Then():
                        # Empty union: the load warp still streamed this task's Q tiles, so pass the
                        # q_load ring through (the frontier left it unbalanced here).
                        for i_q in range(N_TILES):
                            q_load.full.wait(i_q, q_pipe_m.phase)
                        with K.If(elected()), K.Then():
                            for i_q in range(N_TILES):
                                q_load.empty.arrive(i_q)
                        q_pipe_m.advance()
                    with K.If(n_blocks > 0), K.Then():
                        for i_q in range(N_TILES):
                            q_load.full.wait(i_q, q_pipe_m.phase)
                        K.ptx.tcgen05.fence__after_thread_sync()
                        k_stage = K.local_scalar("int32", init=kv_pipe_m.stage)
                        k_phase = K.local_scalar("int32", init=kv_pipe_m.phase)
                        kv_pipe_m.advance()
                        kv_load.full.wait(k_stage, k_phase)
                        K.ptx.tcgen05.fence__after_thread_sync()
                        with K.If(elected()), K.Then():
                            # QK(i_q, 0) is unmasked: it only has to precede PV(i_q, 0), which is
                            # unmasked too, and S_i is fully overwritten by QK(i_q, 1).
                            for i_q in range(N_TILES):
                                first_disabled = K.alloc_local([4], "uint32")
                                for word in range(4):
                                    K.assign(first_disabled[word], K.uint32(0))
                                gemm_qk(i_q, k_stage, first_disabled)
                                s_full.arrive(i_q)
                            kv_load.empty.arrive(k_stage)
                            with K.If(n_blocks == 1), K.Then():
                                # No further QK reads Q: release both Q tiles right away.
                                for i_q in range(N_TILES):
                                    q_load.empty.arrive(i_q)
                        acc = K.local_scalar("int32", init=0)
                        with K.serial(n_blocks, unroll=False) as n:
                            has_next = n + 1 < n_blocks
                            kn_stage = K.local_scalar("int32", init=kv_pipe_m.stage)
                            kn_phase = K.local_scalar("int32", init=kv_pipe_m.phase)
                            with K.If(has_next), K.Then():
                                kv_pipe_m.advance()
                            v_stage = K.local_scalar("int32", init=kv_pipe_m.stage)
                            v_phase = K.local_scalar("int32", init=kv_pipe_m.phase)
                            kv_pipe_m.advance()
                            # Issue order per block n:  QK_0(n+1)  PV_0(n)  QK_1(n+1)  PV_1(n).
                            # QK_i(n+1) needs only WG_i's register copy of S_i(n) (s_consumed); its
                            # write into S_i[64:128) = P_{1-i}(n-1)'s home is ordered after PV_{1-i}(n-1)
                            # by in-order tcgen05.mma execution from this thread.
                            for i_q in range(N_TILES):
                                current_disabled = load_lane_mask(slot, n, i_q)
                                with K.If(has_next), K.Then():
                                    if i_q == 0:
                                        kv_load.full.wait(kn_stage, kn_phase)
                                    tk_sc = iket_range("mma-wait-s-consumed")
                                    s_consumed.wait(i_q, gstep & 1)
                                    iket_end(tk_sc)
                                    K.ptx.tcgen05.fence__after_thread_sync()
                                    next_disabled = load_lane_mask(slot, n + 1, i_q)
                                    with K.If(elected()), K.Then():
                                        gemm_qk(i_q, kn_stage, next_disabled)
                                        s_full.arrive(i_q)
                                        with K.If(n == n_blocks - 2), K.Then():
                                            # Last QK of this Q tile: let the load warp prefetch the
                                            # next task's Q behind the tail PVs.
                                            q_load.empty.arrive(i_q)
                                        if i_q == N_TILES - 1:
                                            kv_load.empty.arrive(kn_stage)
                                if i_q == 0:
                                    kv_load.full.wait(v_stage, v_phase)
                                tk_pw = iket_range("mma-wait-p1")
                                p_o_rescale.wait(i_q, gstep & 1)
                                iket_end(tk_pw)
                                K.ptx.tcgen05.fence__after_thread_sync()
                                disabled = pv_disabled(acc, current_disabled)
                                with K.If(elected()), K.Then():
                                    gemm_pv_part1(i_q, v_stage, acc, disabled)
                                tk_p2 = iket_range("mma-wait-p2")
                                p_ready_2.wait(i_q, gstep & 1)
                                iket_end(tk_p2)
                                K.ptx.tcgen05.fence__after_thread_sync()
                                with K.If(elected()), K.Then():
                                    gemm_pv_part2(i_q, v_stage, disabled)
                                    pv_done.arrive(i_q)
                                    if i_q == N_TILES - 1:
                                        kv_load.empty.arrive(v_stage)
                            K.assign(acc, 1)
                            K.assign(gstep, gstep + 1)
                        q_pipe_m.advance()
                    list_free.arrive(slot)
                    ring_advance(slot, use_m)
                role_tail("mma")

            with r_idle:
                role_tail("idle")

                                                                               
                                                                                        
                                                                               
        def softmax_role(role, w, kind):
            with role:
                slot = K.local_scalar("int32", init=0)
                use_x = K.local_scalar("int32", init=0)
                running_x = K.local_scalar("int32", init=1)
                gstep_x = K.local_scalar("int32", init=0)     # global KV-block step counter (never reset)
                tok_local = tid_in_wg // GQA
                head = tid_in_wg % GQA
                p_col = (1 - w) * 128 + 64                   # bf16 P_w home: upper half of S_{1-w}
                if w == 1:
                    # Hand the first exponential turn to WG0 (xu_turn[0] completion #0).
                    xu_turn.arrive(0)
                with K.While(running_x != 0):
                    list_full.wait(slot, use_x & 1)
                    m = read_meta(slot)
                    n_blocks, n_masked, tok0, h, s, q_base, kv_len, q_len, n_valid_tok, k_base = m
                    with K.If(n_blocks < 0), K.Then():
                        K.assign(running_x, 0)
                    with K.If(n_blocks >= 0), K.Then():
                        my_tok = w * TPT + tok_local
                        row_valid = my_tok < n_valid_tok
                        pos = (kv_len - q_len) + tok0 + my_tok
                        sel_lo = ld_shared_u32(tok_mask.ptr_to([(slot * G + my_tok) * 2]))
                        sel_hi = ld_shared_u32(tok_mask.ptr_to([(slot * G + my_tok) * 2 + 1]))
                        row_max = K.local_scalar("float32", init=K.float32(NEG_INF))
                        row_sum = K.local_scalar("float32", init=K.float32(0.0))

                        def softmax_step(n, apply_mask):
                            blk = ld_shared_i32(blk_list.ptr_to([slot * MAXB_QM + n]))
                            selected = K.Select(
                                blk < 32,
                                K.bitwise_and(K.shift_right(sel_lo, K.Cast("uint32", K.bitwise_and(blk, 31))), K.uint32(1)),
                                K.bitwise_and(K.shift_right(sel_hi, K.Cast("uint32", K.bitwise_and(blk, 31))), K.uint32(1)),
                            ) != K.uint32(0)
                            # Parity of the other warpgroup's s_consumed completion that must precede this
                            # step's P store into S_{1-w}[64:128).  WG0 needs WG1's copy of S_1(n) (step n);
                            # WG1 needs WG0's copy of S_0(n+1) (step n+1) because QK_0(n+1) is issued before
                            # PV_1(n) and overwrites S_0[64:128).  On the task's last block only S_0(n).
                            if w == 0:
                                other_par = gstep_x & 1
                            else:
                                other_par = K.Select(n + 1 < n_blocks, (gstep_x + 1) & 1, gstep_x & 1)
                            s_chunk = K.alloc_local([BLK], "float32")
                            tk_ws = iket_range("sm-wait-s", leader_only=True)
                            s_full.wait(w, gstep_x & 1)
                            iket_end(tk_ws)
                            K.ptx.tcgen05.fence__after_thread_sync()
                            for ci in range(BLK // 64):
                                tmem_load(s_chunk, ci * 64, tmem(w * 128 + ci * 64), 64)
                            K.ptx.tcgen05.wait__ld.sync.aligned()
                            K.ptx.tcgen05.fence__before_thread_sync()
                            s_consumed.arrive(w)          # S_w(n) is in registers: QK_w(n+1) may be issued
                            tk_mx = iket_range("sm-max", leader_only=True)
                            if apply_mask:
                                col_limit = K.min(K.min(kv_len - blk * BLK, BLK), pos - blk * BLK + 1)
                                with K.If(col_limit < BLK), K.Then():
                                    for cidx in range(BLK):
                                        K.ptx.mov.b32(
                                            s_chunk[cidx],
                                            K.Select(cidx < col_limit, s_chunk[cidx], K.float32(NEG_INF)),
                                        )
                            C = MAX_CHAINS
                            mx = K.alloc_local([C], "float32")
                            for ch in range(C):
                                K.ptx.mov.b32(mx[ch], K.max(s_chunk[2 * ch], s_chunk[2 * ch + 1]))
                            for grp in range(1, BLK // (2 * C)):
                                for ch in range(C):
                                    K.ptx["max.f32"](mx[ch], mx[ch], s_chunk[2 * C * grp + 2 * ch], s_chunk[2 * C * grp + 2 * ch + 1])
                            if C == 16:
                                for base in (0, 3, 6, 9, 12):
                                    K.ptx["max.f32"](mx[base], mx[base], mx[base + 1], mx[base + 2])
                                K.ptx["max.f32"](mx[0], mx[0], mx[3], mx[6])
                                K.ptx["max.f32"](mx[9], mx[9], mx[12], mx[15])
                                tile_max = K.local_scalar("float32", init=K.max(mx[0], mx[9]))
                            else:
                                K.ptx["max.f32"](mx[0], mx[0], mx[1], mx[2])
                                K.ptx["max.f32"](mx[3], mx[3], mx[4], mx[5])
                                K.ptx["max.f32"](mx[0], mx[0], mx[6], mx[7])
                                tile_max = K.local_scalar("float32", init=K.max(mx[0], mx[3]))
                            m_old = K.local_scalar("float32", init=row_max)
                            m_new = K.local_scalar("float32", init=K.Select(selected, K.max(m_old, tile_max), m_old))
                            # Online softmax: rescale O only when the max grows by more than
                            # RESCALE_THRESHOLD (in log2 units); otherwise keep the old max.
                            acc_scale = K.local_scalar("float32", init=K.float32(1.0))
                            need = K.local_scalar("int32", init=0)
                            with K.If(K.And(m_new != K.float32(NEG_INF), m_old != K.float32(NEG_INF))), K.Then():
                                delta = (m_old - m_new) * scale_log2
                                with K.If(delta < -RESCALE_THRESHOLD):
                                    with K.Then():
                                        K.ptx.ex2.approx.ftz.f32(acc_scale, delta)
                                        K.assign(need, 1)
                                    with K.Else():
                                        K.assign(m_new, m_old)
                            with K.If(K.And(m_old == K.float32(NEG_INF), m_new != K.float32(NEG_INF))), K.Then():
                                K.assign(row_sum, K.float32(0.0))
                            K.assign(row_max, m_new)
                            any_need = K.local_scalar("uint32")
                            K.ptx.vote_sync.any.pred(any_need, K.ptx.pred(need), K.uint32(0xFFFFFFFF))
                            iket_end(tk_mx)
                            with K.If(any_need != 0), K.Then():
                                # Rare path: PV_w(n-1) must have finished accumulating before O_w is scaled.
                                tk_rs = iket_range("rescale", leader_only=True)
                                pv_done.wait(w, (gstep_x + 1) & 1)
                                K.ptx.tcgen05.fence__after_thread_sync()
                                o_row = K.alloc_local([16], "float32")
                                for d_tile in range(HEAD_DIM // 16):
                                    addr = tmem(256 + w * 128 + d_tile * 16)
                                    tmem_load(o_row, 0, addr, 16)
                                    K.ptx.tcgen05.wait__ld.sync.aligned()
                                    for i in range(16):
                                        K.assign(o_row[i], o_row[i] * acc_scale)
                                    tmem_store16(o_row, 0, addr)
                                K.ptx.tcgen05.wait__st.sync.aligned()
                                K.ptx.tcgen05.fence__before_thread_sync()
                                iket_end(tk_rs)
                            K.assign(row_sum, row_sum * acc_scale)
                            m_safe = K.Select(m_new == K.float32(NEG_INF), K.float32(0.0), m_new)
                            neg_bias = K.local_scalar("float32", init=K.Select(selected, K.float32(0.0) - m_safe * scale_log2, K.float32(NEG_INF)))
                            scale_pair = K.local_scalar("uint64")
                            bias_pair = K.local_scalar("uint64")
                            K.ptx.mov.b64(scale_pair, scale_log2, scale_log2)
                            K.ptx.mov.b64(bias_pair, neg_bias, neg_bias)
                            pair_tmp = K.local_scalar("uint64")
                            # ---- exponential pass, gated by the XU turn; no other waits inside ----
                            tk_turn = iket_range("xu-turn-wait", leader_only=True)
                            xu_turn.wait(w, gstep_x & 1)
                            iket_end(tk_turn)
                            tk_exp = iket_range("sm-exp2", leader_only=True)
                            for frag in range(N_FRAGS):
                                for i in range(16):
                                    idx = frag * 32 + 2 * i
                                    K.ptx.mov.b64(pair_tmp, s_chunk[idx], s_chunk[idx + 1])
                                    K.ptx.fma.rz.ftz.f32x2(pair_tmp, pair_tmp, scale_pair, bias_pair)
                                    K.ptx.mov.b64(s_chunk[idx], s_chunk[idx + 1], pair_tmp)
                                    native = (
                                        not USE_EMU
                                        or i * 2 % 16 < 16 - 2 * EMU_PAIRS
                                        or frag >= N_FRAGS - 1
                                        or frag < EMU_START
                                        or apply_mask
                                    )
                                    if native:
                                        K.ptx.ex2.approx.ftz.f32(s_chunk[idx], s_chunk[idx])
                                        K.ptx.ex2.approx.ftz.f32(s_chunk[idx + 1], s_chunk[idx + 1])
                                    else:
                                        ex2_emulation_2(s_chunk, idx, s_chunk[idx], s_chunk[idx + 1])
                            K.cuda.warp_sync()
                            xu_turn.arrive(1 - w)         # hand the XU to the other warpgroup
                            K.cuda.warp_sync()
                            iket_end(tk_exp)
                            # ---- row-sum / bf16 convert / split P publish ----
                            tk_cv = iket_range("sm-sum-cvt", leader_only=True)
                            sum_acc = [K.local_scalar("uint64") for _ in range(N_SUM_ACC)]
                            for acc_pair in sum_acc:
                                K.ptx.mov.b64(acc_pair, K.float32(0.0), K.float32(0.0))
                            p_chunk = K.alloc_local([BLK // 2], "uint32")
                            for frag in range(N_FRAGS):
                                for i in range(16):
                                    idx = frag * 32 + 2 * i
                                    K.ptx.mov.b64(pair_tmp, s_chunk[idx], s_chunk[idx + 1])
                                    acc_k = sum_acc[(frag * 16 + i) % N_SUM_ACC]
                                    K.ptx.add.rn.ftz.f32x2(acc_k, acc_k, pair_tmp)
                                    K.ptx.cvt.rn.bf16x2.f32(p_chunk[idx // 2], s_chunk[idx + 1], s_chunk[idx])
                                if frag == P_SPLIT_FRAGS - 1:
                                    # P_w(n) overwrites P_w(n-1) (read by PV_w(n-1)) and the upper half of
                                    # S_{1-w}, which the other warpgroup must have copied out first.
                                    tk_pw = iket_range("sm-p-wait", leader_only=True)
                                    pv_done.wait(w, (gstep_x + 1) & 1)
                                    s_consumed.wait(1 - w, other_par)
                                    iket_end(tk_pw)
                                    K.ptx.tcgen05.fence__after_thread_sync()
                                    for f in range(P_SPLIT_FRAGS):
                                        tmem_store16(p_chunk, f * 16, tmem(p_col + f * 16))
                                if frag == P_SPLIT_FRAGS:
                                    K.ptx.tcgen05.wait__st.sync.aligned()
                                    K.ptx.tcgen05.fence__before_thread_sync()
                                    p_o_rescale.arrive(w)     # PV_w(n) part 1 (keys 0..63) may start
                            for f in range(P_SPLIT_FRAGS, N_FRAGS):
                                tmem_store16(p_chunk, f * 16, tmem(p_col + f * 16))
                            K.ptx.tcgen05.wait__st.sync.aligned()
                            K.ptx.tcgen05.fence__before_thread_sync()
                            p_ready_2.arrive(w)               # PV_w(n) part 2 (keys 64..127) may start
                            step = N_SUM_ACC // 2
                            while step:
                                for a_ in range(step):
                                    K.ptx.add.rn.ftz.f32x2(sum_acc[a_], sum_acc[a_], sum_acc[a_ + step])
                                step //= 2
                            sum_lo = K.local_scalar("float32")
                            sum_hi = K.local_scalar("float32")
                            K.ptx.mov.b64(sum_lo, sum_hi, sum_acc[0])
                            K.assign(row_sum, row_sum + sum_lo + sum_hi)
                            iket_end(tk_cv)
                            K.assign(gstep_x, gstep_x + 1)

                        with K.If(n_blocks > 0), K.Then():
                            tk_sm = iket_range("softmax-task", leader_only=True)
                            nm = K.min(n_masked, n_blocks)
                            with K.serial(nm, unroll=False) as n:
                                softmax_step(n, True)
                            with K.serial(n_blocks - nm, unroll=False) as n2:
                                softmax_step(nm + n2, False)
                            iket_end(tk_sm)
                        # Epilogue: normalized bf16 O row straight from TMEM to global (row_valid guards
                        # tail tiles that straddle sequences).
                        tk_ep = iket_range("epilogue", leader_only=True)
                        grow = K.Cast("int64", (q_base + tok0 + my_tok) * HQ + h * GQA + head) * HEAD_DIM
                        with K.If(n_blocks > 0):
                            with K.Then():
                                pv_done.wait(w, (gstep_x + 1) & 1)
                                K.ptx.tcgen05.fence__after_thread_sync()
                                o_row = K.alloc_local([HEAD_DIM], "float32")
                                for ci in range(HEAD_DIM // 64):
                                    tmem_load(o_row, ci * 64, tmem(256 + w * 128 + ci * 64), 64)
                                K.ptx.tcgen05.wait__ld.sync.aligned()
                                K.ptx.tcgen05.fence__before_thread_sync()
                                with K.If(row_valid), K.Then():
                                    inv = K.local_scalar("float32", init=K.float32(0.0))
                                    with K.If(row_sum > K.float32(0.0)), K.Then():
                                        K.ptx.rcp.approx.ftz.f32(inv, row_sum)
                                    o_bf = K.alloc_local([HEAD_DIM // 2], "uint32")
                                    for d in range(HEAD_DIM):
                                        K.assign(o_row[d], o_row[d] * inv)
                                        if d % 2 == 1:
                                            cast_f32x2_bf16x2(o_bf, o_row, d - 1)
                                    for v in range(HEAD_DIM // 16):
                                        K.ptx.st.global_.v8.b32(
                                            out.ptr_to([grow + v * 16]), *(o_bf[8 * v + q8] for q8 in range(8))
                                        )
                            with K.Else():
                                with K.If(row_valid), K.Then():
                                    zero = K.alloc_local([8], "uint32")
                                    for q8 in range(8):
                                        K.assign(zero[q8], K.uint32(0))
                                    for v in range(HEAD_DIM // 16):
                                        K.ptx.st.global_.v8.b32(
                                            out.ptr_to([grow + v * 16]), *(zero[q8] for q8 in range(8))
                                        )
                        iket_end(tk_ep)
                    list_free.arrive(slot)
                    ring_advance(slot, use_x)
                role_tail(kind)

        softmax_role(r_sm0, 0, "sm0")
        softmax_role(r_sm1, 1, "sm1")

    return msa_qmajor


                                                                             
           
                                                                             


def _compile_qm(key, **cfg):
    if key not in _KERNEL_CACHE:
        kernel = make_kernel_qm(**cfg)
        if os.environ.get("MSA_KV_IKET"):
            from tvm.tirx.cuda import iket as _iket

            _KERNEL_CACHE[key] = _iket.IketProfiler().compile(
                kernel.mod, target=kernel.target(), tir_pipeline="tirx"
            )
        else:
            _KERNEL_CACHE[key] = kernel.compile()
    return _KERNEL_CACHE[key]


def setup_qm(data, total_q, B):
    q = data["q"]; k = data["k"]; v = data["v"]
    q2k = data["q2k_indices"]; cu_q = data["cu_seqlens_q"]; cu_k = data["cu_seqlens_k"]
    page_table = data["page_table"]; seqused_k = data["seqused_k"]; out = data["output"]
    device = q.device
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous()
    HQ = q.shape[1]; HKV = k.shape[1]; GQA = HQ // HKV; TOPK = q2k.shape[2]
    PAGED = page_table is not None
    if PAGED:
        num_pages = k.shape[0]; MAX_PAGES = page_table.shape[1]; kv_meta = seqused_k
        page_table = page_table.view(-1)
        assert MAX_PAGES <= MAXB_QM
    else:
        total_k = k.shape[0]; MAX_PAGES = 1; kv_meta = cu_k
        page_table = torch.zeros(1, dtype=torch.int32, device=device)
        assert ceildiv(total_k, BLK) <= MAXB_QM or B > 1
    TPT = 128 // GQA; G = 2 * TPT
    MAXG = ceildiv(total_q, G)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    NUM_TASKS = B * HKV * MAXG
    NUM_CTAS = max(1, min(num_sms, NUM_TASKS))
    cfg = dict(GQA=GQA, TOPK=TOPK, HKV=HKV, B=B, TOTAL_Q=total_q, MAXG=MAXG, PAGED=PAGED,
               MAX_PAGES=MAX_PAGES, NUM_CTAS=NUM_CTAS)
    executable = _compile_qm(("qm",) + tuple(sorted(cfg.items())), **cfg)
    q_map = _encode(q, "bfloat16", (HEAD_DIM // 2, HQ, total_q, 2),
                    (HEAD_DIM * 2, HQ * HEAD_DIM * 2, (HEAD_DIM // 2) * 2), (HEAD_DIM // 2, GQA, TPT, 2))
    if PAGED:
        rows = num_pages * HKV * BLK
        k_map = _encode(k, "bfloat16", (HEAD_DIM // 2, rows, 2), (HEAD_DIM * 2, (HEAD_DIM // 2) * 2), (HEAD_DIM // 2, BLK, 2))
        v_map = _encode(v, "bfloat16", (HEAD_DIM // 2, rows, 2), (HEAD_DIM * 2, (HEAD_DIM // 2) * 2), (HEAD_DIM // 2, BLK, 2))
    else:
        k_map = _encode(k, "bfloat16", (HEAD_DIM // 2, total_k, HKV * 2), (HKV * HEAD_DIM * 2, (HEAD_DIM // 2) * 2), (HEAD_DIM // 2, BLK, 2))
        v_map = _encode(v, "bfloat16", (HEAD_DIM // 2, total_k, HKV * 2), (HKV * HEAD_DIM * 2, (HEAD_DIM // 2) * 2), (HEAD_DIM // 2, BLK, 2))
    sched = torch.zeros(2, dtype=torch.int32, device=device)
    scale_log2 = float(data["softmax_scale"]) * LOG2E
    args = (q_map.ptr, k_map.ptr, v_map.ptr, q2k.view(-1), cu_q, kv_meta, page_table, out.view(-1), sched, scale_log2)
    keep = (q, k, v, q2k, cu_q, kv_meta, page_table, out, sched, q_map, k_map, v_map)

    def run(_args=args, _keep=keep, _exe=executable):
        _exe(*_args)

    run()
    torch.cuda.synchronize(device)
    return run


                                                                             
                                                                         
                                                                   
                                                                             





# ---------------------------------------------------------------------------
# Registry interface.
# ---------------------------------------------------------------------------

KERNEL_META = {
    "name": "agent_evolved_msa_prefill_multishape",
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
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.7.0", "import": "cutlass"},
        {"package": "flashinfer-python", "specifier": ">=0.6.18", "import": "flashinfer"},
    ),
    "provenance": {
        "generator": "hmz",
        "run": "msa_prefill-20260911-201609",
        "selected_version": "frontier/dispatch-qmajor-xalias-csr-kvmajor",
    },
}

CONFIGS = [
    {
        "label": "flat_bf16_b1_q4096_kv4096_h64",
        "batch_size": 1,
        "seqlen_q": 4096,
        "seqlen_kv": 4096,
        "num_qo_heads": 64,
        "num_kv_heads": 4,
        "topk": 16,
        "kv_layout": "flat",
        "kv_dtype": "bfloat16",
        "seed": 43,
    },
    {
        "label": "paged_bf16_b3_q4096_kv8192_h8",
        "batch_size": 3,
        "seqlen_q": 4096,
        "seqlen_kv": 8192,
        "num_qo_heads": 8,
        "num_kv_heads": 2,
        "topk": 4,
        "kv_layout": "paged",
        "kv_dtype": "bfloat16",
        "seed": 73,
    },
]

_CONFIG_KEYS = {
    "batch_size",
    "seqlen_q",
    "seqlen_kv",
    "num_qo_heads",
    "num_kv_heads",
    "topk",
    "kv_layout",
    "kv_dtype",
    "seed",
}
_BY_LABEL = {config["label"]: config for config in CONFIGS}
_DTYPES = {"bfloat16": torch.bfloat16, "float8_e4m3fn": torch.float8_e4m3fn}


def _config(**config: Any) -> dict[str, Any]:
    """Resolve and validate one config against the contract this kernel implements."""
    label = config.get("label")
    base = _BY_LABEL.get(label, CONFIGS[0]) if label else CONFIGS[0]
    values = {key: value for key, value in config.items() if key != "label"}
    unknown = set(values) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"unsupported config keys: {sorted(unknown)}")
    resolved = {key: value for key, value in base.items() if key != "label"}
    resolved.update(values)
    if resolved["kv_layout"] not in ("flat", "paged"):
        raise ValueError("kv_layout must be 'flat' or 'paged'")
    if resolved["kv_dtype"] not in _DTYPES:
        raise ValueError(f"kv_dtype must be one of {sorted(_DTYPES)}")
    if int(resolved["num_qo_heads"]) % int(resolved["num_kv_heads"]):
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    return resolved


def _assert_supported_arch() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved MSA prefill")
    capability = torch.cuda.get_device_capability()
    runtime_arch = f"sm_{capability[0]}{capability[1]}a"
    if runtime_arch not in KERNEL_META["runtime_cuda_archs"]:
        raise SkipTest(
            "agent-evolved MSA prefill requires one of "
            f"{KERNEL_META['runtime_cuda_archs']}, got {runtime_arch}"
        )


def _route(resolved: dict[str, Any]) -> str:
    """The candidate's own shape dispatch, expressed on the config alone.

    q-major union takes bf16 K/V with GQA >= 8 and a short block list; the
    scan-based kv-major route takes flat FP8 K/V; the CSR kv-major v2 route
    takes everything else, which here is the paged bf16 row.
    """
    gqa = int(resolved["num_qo_heads"]) // int(resolved["num_kv_heads"])
    paged = resolved["kv_layout"] == "paged"
    max_blocks = ceildiv(int(resolved["seqlen_kv"]), BLK)
    if (
        resolved["kv_dtype"] == "bfloat16"
        and gqa in (8, 16, 32, 64, 128)
        and max_blocks <= MAXB_QM
        and int(resolved["topk"]) <= 32
    ):
        return "qmajor"
    if resolved["kv_dtype"] == "float8_e4m3fn" and not paged and int(resolved["topk"]) <= 16:
        return "kvmajor_scan"
    return "kvmajor_csr"


def get_kernel(**config: Any):
    """Return the traced Kern functions this config's route builds.

    Each route is one producer plus, for the two kv-major routes, a combine
    kernel. Tracing needs the launch geometry the candidate derives in
    `setup`, so this mirrors that derivation rather than compiling.
    """
    from tirx_kernels.runner import hardware_num_sms

    resolved = _config(**config)
    route = _route(resolved)
    batch = int(resolved["batch_size"])
    total_q = batch * int(resolved["seqlen_q"])
    hkv = int(resolved["num_kv_heads"])
    gqa = int(resolved["num_qo_heads"]) // hkv
    topk = int(resolved["topk"])
    max_blocks = ceildiv(int(resolved["seqlen_kv"]), BLK)
    paged = resolved["kv_layout"] == "paged"
    max_pages = max_blocks if paged else 0
    num_ctas = hardware_num_sms()
    if route == "qmajor":
        return {
            "qmajor": make_kernel_qm(
                GQA=gqa,
                TOPK=topk,
                HKV=hkv,
                B=batch,
                TOTAL_Q=total_q,
                MAXG=max_blocks,
                PAGED=paged,
                MAX_PAGES=max_pages,
                NUM_CTAS=num_ctas,
            ).func
        }
    if route == "kvmajor_scan":
        return {
            "producer": make_kernel_kv(
                GQA=gqa,
                TOPK=topk,
                HKV=hkv,
                B=batch,
                TOTAL_Q=total_q,
                MAXB=max_blocks,
                MAXC=max_blocks,
                PAGED=paged,
                MAX_PAGES=max_pages,
                KV_FP8=True,
                NUM_CTAS=num_ctas,
            ).func,
            "combine": make_kernel_combine_fp8(
                GQA=gqa,
                TOPK=topk,
                HKV=hkv,
                B=batch,
                TOTAL_Q=total_q,
                MAXB=max_blocks,
                NUM_CTAS=num_ctas,
            ).func,
        }
    return {
        "producer": make_kernel_kv2(
            GQA=gqa,
            TOPK=topk,
            HKV=hkv,
            B=batch,
            TOTAL_Q=total_q,
            MAXB=max_blocks,
            CAP=max_blocks,
            PAGED=paged,
            MAX_PAGES=max_pages,
            KV_FP8=False,
            NUM_CTAS=num_ctas,
        ).func,
        "combine": make_kernel_combine_bf16_paged(
            GQA=gqa,
            TOPK=topk,
            HKV=hkv,
            B=batch,
            TOTAL_Q=total_q,
            MAXB=max_blocks,
            NUM_CTAS=num_ctas,
        ).func,
    }


# ---------------------------------------------------------------------------
# Inputs.
#
# The draw order reproduces the packaged MSA-prefill rows, which follow
# flashinfer PR #4355's `bench_blackwell_msa_sm100.py`: q, k and v come from one
# `randn/3` generator sequence in that order, then `q2k_indices` is drawn per
# (query token, kv head) from the blocks the token may see under bottom-right
# causal masking. Paged rows then scatter K/V into 128-token pages stored in
# reverse order, which is the layout that benchmark builds.
# ---------------------------------------------------------------------------


def _make_q2k_indices(batch_size, seqlen_q, seqlen_kv, num_kv_heads, topk, seed, device):
    total_q = batch_size * seqlen_q
    out = torch.full((num_kv_heads, total_q, topk), -1, dtype=torch.int32)
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    offset = seqlen_kv - seqlen_q
    for row in range(total_q):
        visible_blocks = ceildiv(offset + row % seqlen_q + 1, BLK)
        for kv_head in range(num_kv_heads):
            selected = torch.randperm(visible_blocks, generator=generator)
            selected = selected[: min(topk, visible_blocks)].sort().values
            out[kv_head, row, : selected.numel()] = selected.to(torch.int32)
    return out.to(device)


def _to_pages(logical, batch_size, seqlen_kv):
    """128-token pages stored in reverse order, as the packaged benchmark builds them."""
    _total_k, num_kv_heads, head_dim = logical.shape
    pages_per_seq = ceildiv(seqlen_kv, BLK)
    total_pages = batch_size * pages_per_seq
    padded = logical.view(batch_size, seqlen_kv, num_kv_heads, head_dim)
    if pages_per_seq * BLK != seqlen_kv:
        padded = logical.new_zeros((batch_size, pages_per_seq * BLK, num_kv_heads, head_dim))
        padded[:, :seqlen_kv] = logical.view(batch_size, seqlen_kv, num_kv_heads, head_dim)
    pages = (
        padded.view(batch_size, pages_per_seq, BLK, num_kv_heads, head_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(total_pages, num_kv_heads, BLK, head_dim)
    )
    page_table = torch.arange(total_pages - 1, -1, -1, dtype=torch.int32, device=logical.device)
    return pages.flip(0).contiguous(), page_table.view(batch_size, pages_per_seq).contiguous()


def prepare_data(**config: Any) -> dict[str, Any]:
    """Build one row's contract tensors plus the preallocated output."""
    resolved = _config(**config)
    device = torch.device("cuda")
    batch_size = int(resolved["batch_size"])
    seqlen_q = int(resolved["seqlen_q"])
    seqlen_kv = int(resolved["seqlen_kv"])
    num_qo_heads = int(resolved["num_qo_heads"])
    num_kv_heads = int(resolved["num_kv_heads"])
    topk = int(resolved["topk"])
    seed = int(resolved["seed"])
    kv_dtype = _DTYPES[resolved["kv_dtype"]]
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(shape, dtype):
        values = torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
        return (values / 3.0).to(dtype)

    total_q = batch_size * seqlen_q
    total_k = batch_size * seqlen_kv
    q = randn((total_q, num_qo_heads, HEAD_DIM), torch.bfloat16)
    k = randn((total_k, num_kv_heads, HEAD_DIM), kv_dtype)
    v = randn((total_k, num_kv_heads, HEAD_DIM), kv_dtype)
    cu_seqlens_q = torch.arange(0, total_q + 1, seqlen_q, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.arange(0, total_k + 1, seqlen_kv, dtype=torch.int32, device=device)
    q2k_indices = _make_q2k_indices(
        batch_size, seqlen_q, seqlen_kv, num_kv_heads, topk, seed, device
    )
    page_table = seqused_k = None
    if resolved["kv_layout"] == "paged":
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
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "page_table": page_table,
        "seqused_k": seqused_k,
        "softmax_scale": HEAD_DIM**-0.5,
        "output": torch.empty_like(q),
        "total_q": total_q,
        "batch_size": batch_size,
    }


# ---------------------------------------------------------------------------
# Independent oracle.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _logical_kv(case: dict[str, Any]):
    """Undo the paged layout so the oracle always sees flat [total_k, Hkv, D]."""
    resolved = case["config"]
    if resolved["kv_layout"] != "paged":
        return case["k"].float(), case["v"].float()
    batch = int(resolved["batch_size"])
    seqlen_kv = int(resolved["seqlen_kv"])
    page_table = case["page_table"].long()
    out = []
    for pages in (case["k"], case["v"]):
        gathered = pages[page_table.reshape(-1)]  # [B*pages, Hkv, BLK, D]
        gathered = gathered.view(batch, -1, gathered.shape[1], BLK, gathered.shape[-1])
        flat = gathered.permute(0, 1, 3, 2, 4).reshape(batch, -1, gathered.shape[2], gathered.shape[-1])
        out.append(flat[:, :seqlen_kv].reshape(-1, gathered.shape[2], gathered.shape[-1]).float())
    return out[0], out[1]


@torch.no_grad()
def _reference_output(case: dict[str, Any], *, chunk: int = 256) -> torch.Tensor:
    """Masked FP32 sparse attention, computed independently of the kernel.

    A query token may attend only tokens of its selected blocks that also sit
    at or before its bottom-right causal position; -1 padding never selects a
    block, and an empty selection yields a zero row.
    """
    resolved = case["config"]
    q = case["q"]
    k_flat, v_flat = _logical_kv(case)
    q2k = case["q2k_indices"]
    scale = float(case["softmax_scale"])
    batch = int(resolved["batch_size"])
    seqlen_q = int(resolved["seqlen_q"])
    seqlen_kv = int(resolved["seqlen_kv"])
    hq = int(resolved["num_qo_heads"])
    hkv = int(resolved["num_kv_heads"])
    gqa = hq // hkv
    offset = seqlen_kv - seqlen_q
    device = q.device
    out = torch.empty_like(q)
    key_block = torch.arange(seqlen_kv, device=device) // BLK
    key_pos = torch.arange(seqlen_kv, device=device)
    for sequence in range(batch):
        q_base = sequence * seqlen_q
        k_base = sequence * seqlen_kv
        for kv_head in range(hkv):
            k_head = k_flat[k_base : k_base + seqlen_kv, kv_head]
            v_head = v_flat[k_base : k_base + seqlen_kv, kv_head]
            for start in range(0, seqlen_q, chunk):
                stop = min(start + chunk, seqlen_q)
                rows = torch.arange(start, stop, device=device)
                selected = q2k[kv_head, q_base + start : q_base + stop]
                valid_sel = selected >= 0
                block_ok = (
                    (selected.unsqueeze(-1) == key_block.view(1, 1, -1)) & valid_sel.unsqueeze(-1)
                ).any(1)
                causal_ok = key_pos.view(1, -1) <= (offset + rows).view(-1, 1)
                allowed = block_ok & causal_ok
                q_chunk = q[
                    q_base + start : q_base + stop, kv_head * gqa : (kv_head + 1) * gqa
                ].float()
                scores = torch.einsum("tgd,kd->tgk", q_chunk, k_head) * scale
                scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))
                empty = ~allowed.any(-1)
                weights = torch.softmax(scores, dim=-1)
                weights = torch.where(
                    empty.unsqueeze(1).unsqueeze(-1), torch.zeros_like(weights), weights
                )
                out[q_base + start : q_base + stop, kv_head * gqa : (kv_head + 1) * gqa] = (
                    torch.einsum("tgk,kd->tgd", weights, v_head).to(q.dtype)
                )
    return out


def _gate(reference, actual, atol: float, rtol: float) -> None:
    """The packaged task's acceptance criterion, applied element-wise.

    An element fails only when it exceeds BOTH the absolute and the relative
    bound, which is the harness's rule
    (`exceeds = (abs_error > atol) & (rel_error > rtol)`) and is looser than
    torch's additive `atol + rtol*|reference|` form.
    """
    got = actual.float()
    ref = reference.float()
    abs_error = (got - ref).abs()
    rel_error = abs_error / (ref.abs() + 1e-8)
    exceeds = (abs_error > atol) & (rel_error > rtol)
    failures = int(exceeds.sum())
    if failures:
        raise AssertionError(
            f"{failures} of {exceeds.numel()} elements exceed atol={atol} and rtol={rtol}; "
            f"max abs {float(abs_error.max()):.6e}, max rel {float(rel_error.max()):.6e}"
        )


def check_correctness(outputs: dict[str, Any], **config: Any) -> None:
    resolved = _config(**config)
    first, actual, reference = outputs["first"], outputs["actual"], outputs["reference"]
    for name, tensor in (("first", first), ("actual", actual), ("reference", reference)):
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} output contains non-finite values")
    if not torch.equal(first, actual):
        max_abs = float((first.float() - actual.float()).abs().max())
        raise AssertionError(
            f"identical launches are not exactly repeatable; max abs diff={max_abs}"
        )
    # The packaged task gates FP8-KV rows at 0.1 and every other row at 0.01,
    # because an FP8 K/V row's own storage already costs more than the tighter
    # bound allows.
    atol = rtol = 0.1 if resolved["kv_dtype"] == "float8_e4m3fn" else 1e-2
    _gate(reference, actual, atol, rtol)
    diff_rms = torch.sqrt(torch.mean((actual.float() - reference.float()).square()))
    reference_rms = torch.sqrt(torch.mean(reference.float().square()))
    rms_ratio = float(diff_rms / (reference_rms + 1e-8))
    if rms_ratio >= 5e-2:
        raise AssertionError(f"normalized RMS error ratio {rms_ratio:.6e} must be below 5e-2")


def _launch(case: dict[str, Any]):
    """Build the candidate's own dispatch closure for this shape.

    This is the candidate's `setup` dispatch, keyed off the resolved config
    instead of re-deriving the same predicates from the tensors.
    """
    route = _route(case["config"])
    builder = {
        "qmajor": setup_qm,
        "kvmajor_scan": setup_kv,
        "kvmajor_csr": setup_kv2,
    }[route]
    return builder(case, case["total_q"], case["batch_size"])


def run_test(**config: Any) -> None:
    _assert_supported_arch()
    case = prepare_data(**config)
    run = _launch(case)
    case["output"].fill_(float("nan"))
    run()
    torch.cuda.synchronize()
    first = case["output"].clone()
    # Poison the buffer so a kernel that skips rows cannot pass by leaving them.
    case["output"].fill_(42.0)
    run()
    torch.cuda.synchronize()
    actual = case["output"].clone()
    reference = _reference_output(case)
    torch.cuda.synchronize()
    check_correctness({"first": first, "actual": actual, "reference": reference}, **config)


# ---------------------------------------------------------------------------
# Benchmark: the MiniMax reference arm and the timed dispatch.
# ---------------------------------------------------------------------------


# MiniMax's forward accepts bf16 and FP8-E4M3 storage only, and its CSR kernel
# fails NVVM codegen for GQA ratios below 8 on every cutlass-dsl 4.6-4.8 build
# the packaged baseline tried. `baseline_arm` therefore routes those rows to
# flashinfer's trtllm-gen block-sparse bridge instead, and this port follows it
# so each row is compared against the arm the harness actually scores.
_MINIMAX_GQA_RATIOS = (8, 16)
_MINIMAX_TOPK = (4, 8, 16, 32)
_BRIDGE_WORKSPACE: dict[str, Any] = {}


def _baseline_arm(case: dict[str, Any]) -> str:
    resolved = case["config"]
    gqa = int(resolved["num_qo_heads"]) // int(resolved["num_kv_heads"])
    if gqa in _MINIMAX_GQA_RATIOS and int(resolved["topk"]) in _MINIMAX_TOPK:
        return "minimax"
    return "trtllm_bridge"


def _bridge_reference(case: dict[str, Any]):
    """flashinfer's trtllm-gen block-sparse decode, one request per query token.

    trtllm-gen takes per-(kv head, request) page lists where MSA has
    per-(kv head, query token) block ids, so every query token becomes one
    single-token request. The conversion, the flat-to-paged copy and the
    `max_seq_len` sync are prepare work, as the packaged baseline does them.
    """
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    resolved = case["config"]
    q, k, v = case["q"], case["k"], case["v"]
    device = q.device
    batch = int(resolved["batch_size"])
    seqlen_kv = int(resolved["seqlen_kv"])
    cu_q = case["cu_seqlens_q"].long()
    if case["page_table"] is not None:
        k_pages, v_pages = k, v
        page_table = case["page_table"].long()
        kv_lens = case["seqused_k"].long()
    else:
        cu_k = case["cu_seqlens_k"].long()
        kv_lens = cu_k[1:] - cu_k[:-1]
        pages_per_seq = (kv_lens + BLK - 1) // BLK
        page_base = torch.cumsum(pages_per_seq, 0) - pages_per_seq
        total_pages = int(pages_per_seq.sum())
        sequences = torch.arange(kv_lens.numel(), device=device)
        seq_of_token = torch.repeat_interleave(sequences, kv_lens, output_size=k.shape[0])
        local = torch.arange(k.shape[0], device=device) - cu_k[:-1][seq_of_token]
        page = page_base[seq_of_token] + local // BLK
        slot = local % BLK
        k_pages = k.new_zeros((total_pages, k.shape[1], BLK, k.shape[2]))
        v_pages = torch.zeros_like(k_pages)
        k_pages[page, :, slot] = k
        v_pages[page, :, slot] = v
        seq_of_page = torch.repeat_interleave(sequences, pages_per_seq, output_size=total_pages)
        pages = torch.arange(total_pages, device=device)
        page_table = torch.zeros(
            (kv_lens.numel(), int(pages_per_seq.max())), dtype=torch.int64, device=device
        )
        page_table[seq_of_page, pages - page_base[seq_of_page]] = pages

    q2k = case["q2k_indices"]
    total_q = q2k.shape[1]
    q_lens = cu_q[1:] - cu_q[:-1]
    seq_of_q = torch.repeat_interleave(
        torch.arange(batch, device=device), q_lens, output_size=total_q
    )
    local_q = torch.arange(total_q, device=device) - cu_q[:-1][seq_of_q]
    q_pos = kv_lens[seq_of_q] - q_lens[seq_of_q] + local_q
    valid = q2k >= 0
    count = valid.sum(-1)
    physical = page_table[seq_of_q.view(1, -1, 1), q2k.clamp(min=0).long()]
    block_tables = torch.where(valid, physical, 0).to(torch.int32).contiguous()
    last = q2k.gather(-1, (count - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1).long()
    tail = torch.clamp(q_pos.view(1, -1) + 1 - last * BLK, max=BLK)
    seq_lens = ((count - 1) * BLK + tail).clamp(min=0).to(torch.int32).contiguous()
    max_seq_len = int(seq_lens.max().item())
    key = str(device)
    if key not in _BRIDGE_WORKSPACE:
        _BRIDGE_WORKSPACE[key] = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
    workspace = _BRIDGE_WORKSPACE[key]
    scale = float(case["softmax_scale"])

    def launch():
        return trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_pages, v_pages),
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=scale,
            bmm2_scale=1.0,
            kv_layout="HND",
            backend="trtllm-gen",
            enable_block_sparse_attention=True,
        )

    launch()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        launch()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    state = (graph, k_pages, v_pages, block_tables, seq_lens)

    def replay(_state=state):
        _state[0].replay()

    replay._keep_alive = state
    return replay


def _reference_builder(case: dict[str, Any]):
    """The arm the packaged harness scores this row against."""
    if _baseline_arm(case) == "minimax":
        return _minimax_reference(case)
    return _bridge_reference(case)


def _minimax_reference(case: dict[str, Any]):
    """Capture MiniMax's sparse forward in a CUDA graph and return its replay.

    `build_k2q_csr` turns the contract's `q2k_indices` into the kernel's CSR
    reverse index and forward schedule; that build, the JIT and the workspace
    are prepare work, exactly as flashinfer PR #4355's
    `bench_blackwell_msa_sm100.py` does it with `baseline_mode="minimax_public"`.
    The capture removes the host gap between MiniMax's forward and its combine
    so the timed span is the kernels alone.

    MiniMax's forward wants cumulative KV lengths on paged rows too, which its
    own bench builds from `seqused_k`.
    """
    import fmha_sm100

    q, k, v = case["q"], case["k"], case["v"]
    q2k = case["q2k_indices"]
    cu_q = case["cu_seqlens_q"]
    cu_k = case["cu_seqlens_k"]
    if cu_k is None:
        seqused = case["seqused_k"]
        cu_k = torch.zeros(seqused.numel() + 1, dtype=torch.int32, device=seqused.device)
        cu_k[1:] = torch.cumsum(seqused.to(torch.int32), 0)
    q_lens = cu_q[1:] - cu_q[:-1]
    kv_lens = cu_k[1:] - cu_k[:-1]
    max_seqlen_q = int(q_lens.max())
    max_seqlen_k = int(kv_lens.max())
    total_rows = int(((kv_lens + BLK - 1) // BLK).sum())
    k2q_row_ptr, k2q_q_indices, schedule = fmha_sm100.build_k2q_csr(
        q2k,
        cu_q,
        cu_k,
        BLK,
        total_k=int(cu_k[-1]),
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        total_rows=total_rows,
        qhead_per_kv=int(case["config"]["num_qo_heads"]) // int(case["config"]["num_kv_heads"]),
        return_schedule=True,
    )

    def launch():
        return fmha_sm100.sparse_atten_func(
            q,
            k,
            v,
            k2q_row_ptr,
            k2q_q_indices,
            int(q2k.shape[-1]),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            blk_kv=BLK,
            causal=True,
            softmax_scale=float(case["softmax_scale"]),
            return_softmax_lse=False,
            page_table=case["page_table"],
            seqused_k=case["seqused_k"],
            schedule=schedule,
        )

    launch()  # JIT / extension load and workspace allocation
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        launch()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    state = (graph, k2q_row_ptr, k2q_q_indices, schedule, cu_k)

    def replay(_state=state):
        _state[0].replay()

    replay._keep_alive = state
    return replay


def prepare_bench(**config: Any):
    """Resolve the config before bench-suite assigns a GPU."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    return prepared_gpu_benchmark(run_gpu, {"config": dict(config)})


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
    case = prepare_data(**config)
    run = _launch(case)
    run()
    torch.cuda.synchronize()

    return bench(
        {"tirx": run},
        references={_baseline_arm(case): lambda: _reference_builder(case)},
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
