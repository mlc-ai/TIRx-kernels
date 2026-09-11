# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved SM100a KDA backward for the packed Kimi K3 workload portfolio.

The supported contract is the prepared-tensor contract of
``fla.ops.kda.chunk_bwd.chunk_kda_bwd`` with B=1, K=V=128, chunk_size=64,
``Hv % Hqk == 0`` (grouped value heads) and arbitrary packed sequence lengths
with masked partial trailing chunks. It covers all seventeen official
KDA-backward workloads: ``kda-bwd-packed-1024x8-h96`` (8192 tokens as eight
1024-token sequences, Hqk=Hv=96) and the sixteen packed portfolios that cross
T in {18432, 32768} and four sequence layouts with (Hqk, Hv) in
{(2,4), (2,8), (4,4), (4,8)}.

Inputs are the saved L2-normalized q/k, v, the activated update gate beta, the
saved Aqk/Akk interaction matrices, the chunk-local base-2 cumulative gate g,
the per-sequence K-first initial state, the upstream do and dht, and
cu_seqlens. The kernel writes dq, dk, dv, db, dg and dh0; dA and dbias are
absent from this contract.

The selected kernel is the ``fused-zfold`` global-best frontier member of the
2026-09-10 KDA-backward evolution run ``kda-bwd-portfolio-carry-3``. It is a
shape-aware dispatcher over two persistent tcgen05 designs:

* Fused chain kernel, used when ``Hqk == Hv``, ``Hv % 8 == 0`` and every
  sequence is a whole number of chunks. One 12-warp CTA per SM runs pass 1
  (forward chunk-state recompute into TMEM, bf16 snapshots and a bf16 2^g
  cache) for a host-scheduled chain set, publishes per-chain readiness with an
  epoch-stamped gpu-scope release flag, then runs pass 2 (backward) for its own
  chains. Pass 2 folds the W/U/Vn/dw chain into a single TMEM intermediate
  ``Z = vb - h^T kbg``. Pass 1 builds a channel-major TMEM tile with four
  tcgen05 identity MMAs per raw-K stage; the M=64 accumulators are read out
  through 16-lane half-lane readouts.
* Persistent megakernel, used for every other layout (grouped value heads,
  partial trailing chunks). Each (sequence, v-head) chain runs a forward and a
  backward recurrence stream that publish per-chunk progress flags, and every
  (chunk, qk-head) work item is then independent: it loops over the grouped
  v-heads, reduces dq/dk in TMEM, and produces dv, db and dg in one pass. Work
  units are dealt from a host-built readiness-ordered table.

Numerical notes. All MMAs are ``kind::f16`` bf16 x bf16 -> fp32, the same class
the FLA reference feeds ``tl.dot``; no tf32 or fp8 path is used. Two choices
are lower precision than the reference and are deliberate: ``2^g`` is cached to
global memory in bf16, and inverse gates are recovered with
``rcp.approx.ftz.f32`` instead of the reference's fp32 ``exp2(gn - g)``.
Measured across the official suite this keeps the maximum normalized RMS error
ratio at 5.959e-3, inside the tightest per-output limit of 8e-3.
"""

import ctypes
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K

D = 128
CHUNK = 64
MMA_SS = "tcgen05.mma.cta_group::1.kind::f16"
TMA_LD = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
TMA_ST = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
TMA_PREFETCH = "cp.async.bulk.prefetch.tensor.3d.L2.global.tile"
TC_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TC_ST32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"
TC_LD8 = "tcgen05.ld.sync.aligned.32x32b.x8.b32"
TC_LD4 = "tcgen05.ld.sync.aligned.32x32b.x4.b32"
TC_LD_HALF32 = "tcgen05.ld.sync.aligned.16x256b.x4.b32"
TC_ST16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
TC_ST8 = "tcgen05.st.sync.aligned.32x32b.x8.b32"
WAIT_LD = "tcgen05.wait::ld.sync.aligned"
WAIT_ST = "tcgen05.wait::st.sync.aligned"
FENCE_ASYNC = "fence.proxy.async.shared::cta"
BULK_COMMIT = "cp.async.bulk.commit_group"
BULK_WAIT_READ = "cp.async.bulk.wait_group.read"
BULK_WAIT = "cp.async.bulk.wait_group"
TC_FENCE_BEFORE = "tcgen05.fence::before_thread_sync"
TC_FENCE_AFTER = "tcgen05.fence::after_thread_sync"


_CUDA_MBAR_WAIT = K.MBarrier._wait





def _ptx_mbarrier_wait(self, stage, phase):
    ready = K.local_scalar("uint32", init=K.uint32(0))
    barrier = K.cuda.cvta_generic_to_shared(self.buf.ptr_to([stage]))
    target_phase = K.cast(phase ^ self.phase_offset, "uint32")
    with K.While(ready == K.uint32(0)):
        K.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
            ready, barrier, target_phase, K.uint32(10_000_000)
        )

UNITS_PER_STAGE = 512
SBO_UNITS = 64


def idesc(M, N, *, ta=0, tb=0, na=0, nb=0):
    """Dense tcgen05 instruction descriptor: bf16 x bf16 -> f32."""
    return (
        (1 << 4) | (1 << 7) | (1 << 10) | (na << 13) | (nb << 14) | (ta << 15) | (tb << 16)
        | ((N >> 3) << 17) | ((M >> 4) << 24)
    )


class Op:
    """A tcgen05 matrix-descriptor operand at trace-time stage `base` of the staged pool.

    The pool's base descriptor is encoded once (stage 0, 64-row column-atom stride); every
    operand is that descriptor plus a compile-time immediate (stage offset in 16 B units and,
    for 128-row tiles, the larger LBO field).
    """

    LBO_BASE = 64 * 8

    def __init__(self, base_desc, base, rows, kdim, major):
        self.rows = rows
        self.major = major
        self.n_k = kdim // 16
        ldo = rows * 8
        assert ldo >= self.LBO_BASE
        self._bd = base_desc
        self._imm = base * UNITS_PER_STAGE + ((ldo - self.LBO_BASE) << 16)

    def off(self, kp):
        if self.major == "k":
            return (kp % 4) * 2 + (kp // 4) * self.rows * 8
        return kp * 128

    def desc(self, kp, units=None):
        d = self._bd[0] + K.uint64(self._imm + self.off(kp))
        if units is not None:
            d = d + units
        return d


def mma_chain(tm, dcol, a, b, idesc_val, accumulate, a_units=None, b_units=None):
    """One k-chain of tcgen05.mma from the calling (single, elected) thread."""
    n_k = b.n_k
    assert a.n_k == n_k, (a.n_k, n_k)
    for kp in range(n_k):
        K.ptx[MMA_SS](
            K.Cast("uint32", tm[0] + dcol),
            a.desc(kp, a_units),
            b.desc(kp, b_units),
            K.uint32(idesc_val),
            K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0),
            K.ptx.pred(1 if (accumulate or kp > 0) else 0),
        )


def mma_chain_ta(tm, dcol, a_col, b, idesc_val, accumulate, b_units=None):
    """One k-chain of tcgen05.mma whose A operand is a K-major bf16 tile in Tensor Memory.

    A[M=128 lanes x K] is packed two bf16 per 32-bit column starting at column `a_col`, so each
    16-element k-step advances the A address by 8 columns (the FlashAttention-4 P-in-TMEM form).
    """
    for kp in range(b.n_k):
        K.ptx[MMA_SS](
            K.Cast("uint32", tm[0] + dcol),
            K.Cast("uint32", tm[0] + a_col + 8 * kp),
            b.desc(kp, b_units),
            K.uint32(idesc_val),
            K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0),
            K.ptx.pred(1 if (accumulate or kp > 0) else 0),
        )


class AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def encode_tensor_map(tensor, dtype: str, dims, strides_bytes, box, swizzle=3, l2promo=2):
    """cuTensorMapEncodeTiled through TVM's runtime function (dims innermost first)."""
    import tvm

    desc = AlignedTensorMap()
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    rank = len(dims)
    assert len(strides_bytes) == rank - 1 and len(box) == rank
    encode(
        desc.ptr, dtype, rank, ctypes.c_void_p(int(tensor.data_ptr())),
        *[int(d) for d in dims], *[int(s) for s in strides_bytes], *[int(b) for b in box],
        *([1] * rank), 0, swizzle, l2promo, 0,
    )
    return desc


def token_map(tensor, T, H, inner, box_inner, box_rows=CHUNK, swizzle=3):
    """[T, H, inner] tensor viewed as dims (inner, T, H): coordinates (d0, token, head)."""
    esz = tensor.element_size()
    dtype = {2: "bfloat16", 4: "float32"}[esz]
    return encode_tensor_map(tensor, dtype, (inner, T, H), (esz * inner * H, esz * inner),
                             (box_inner, box_rows, 1), swizzle=swizzle)


def state_map(tensor, n_states):
    """[n_states, 128, 128] bf16 states viewed as dims (128 v, 128 k, n): coordinates (v0, 0, idx)."""
    return encode_tensor_map(tensor, "bfloat16", (D, D, n_states), (2 * D, 2 * D * D), (64, D, 1))


KV_BYTES = 2 * CHUNK * D * 2
A_BYTES = CHUNK * CHUNK * 2
G_BYTES = CHUNK * D * 4
IN3_BYTES = 3 * CHUNK * D * 2 + G_BYTES






def make_mega_kernel(HQ: int, HV: int, static_grid=None):
    K.MBarrier._wait = _ptx_mbarrier_wait
    G = HV // HQ
    HALF_DA_READOUT = HQ < 96
    HALF_XY_READOUT = HQ < 96
    HQK64 = K.int64(HQ * D)
    G = HV // HQ
    HQK = HQ * D
    HVK = HV * D
    HVK64 = K.int64(HVK)






    F_KV, F_AKK, F_HS, F_G, F_KG, F_KBG, F_VB = 0, 4, 6, 10, 14, 18, 22





    B_QK, B_DO, B_G, B_AQK, B_AKK, B_T2, B_DHB, B_DV2 = 0, 4, 8, 12, 13, 14, 16, 20

    MAXSEQ = 64
    TM_H, TM_W0, TM_U0, TM_KT, TM_VT = 0, 128, 256, 384, 448
    TM_DH, TM_BW, TM_DV2, TM_QT, TM_KTB, TM_T1, TM_KB = 0, 128, 192, 256, 320, 384, 448



    T1, T2, T3, T5, T6, DHB = 0, 2, 4, 8, 10, 12
    DV2, ZT, DVB, DAM = 6, 8, 6, 9
    PB0, PB1 = 12, 14
    ST_Q, ST_K, ST_V, ST_G = 12, 14, 16, 8
    S_DO, S_H, S_AQK, S_AKK = 18, 20, 24, 25
    IN_BYTES = 3 * CHUNK * D * 2
    EG_BYTES = CHUNK * D * 2
    DO_BYTES = CHUNK * D * 2
    H_BYTES = D * D * 2

    TM_ADQ, TM_ADK = 0, 64
    S1, S2, S3, S4, S6, S5 = 128, 192, 256, 320, 384, 448
    TMEM_COLS = 512

    @K.kernel(warps=12, arch="sm_100a", min_blocks_per_sm=1,
              grid="num_ctas" if static_grid is None else static_grid)
    def kda_bwd_mega(
        q: K.gptr[K.bf16],
        k: K.gptr[K.bf16],
        v: K.gptr[K.bf16],
        beta: K.gptr[K.bf16],
        aqk: K.gptr[K.bf16],
        akk: K.gptr[K.bf16],
        g: K.gptr[K.f32],
        egcache: K.gptr[K.bf16],
        do: K.gptr[K.bf16],
        dht: K.gptr[K.f32],
        h0: K.gptr[K.f32],
        hsnap: K.gptr[K.bf16],
        dhsnap: K.gptr[K.bf16],
        cu_seqlens: K.gptr[K.i64],
        dq: K.gptr[K.f32],
        dk: K.gptr[K.f32],
        dv: K.gptr[K.bf16],
        db: K.gptr[K.f32],
        dg: K.gptr[K.f32],
        dh0: K.gptr[K.f32],
        stream_counter: K.gptr[K.i32],
        flags: K.gptr[K.i64],
        stream_tab: K.gptr[K.i32],
        item_tab: K.gptr[K.i32],
        seq_tab: K.gptr[K.i32],
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        g_map: K.TensorMap,
        eg_map: K.TensorMap,
        do_map: K.TensorMap,
        aqk_map: K.TensorMap,
        akk_map: K.TensorMap,
        h_map: K.TensorMap,
        dh_map: K.TensorMap,
        scale: K.f32,
        num_seqs: K.i32,
        num_items: K.i32,
        num_ctas: K.i32,
        epoch: K.i32,
    ):
        for buf in (q, k, v, aqk, akk, g, egcache, do, hsnap, dhsnap):
            K.keep_alive(buf.data)
        num_chains = num_seqs * K.int32(HV)
        num_streams = num_chains * K.int32(2)
        total_work = num_streams + num_items
        cta = K.local_scalar("int32", init=K.Cast("int32", K.cta_id()))
        ITEM_RING = 4


        ep64 = K.local_scalar("int64", init=K.Cast("int64", epoch) * K.int64(1 << 32))

        sp = K.specialize()
        cg = sp.role("cg", warps=list(range(8)), regs=208)
        auxg = sp.warpgroup("aux", warps=[8, 9, 10, 11], regs=88)
        loader = sp.role("loader", warps=[8], group=auxg)
        mma = sp.role("mma", warps=[9], group=auxg)
        w10 = sp.role("w10", warps=[10], group=auxg)
        w11 = sp.role("w11", warps=[11], group=auxg)

        smem = K.smem_pool()
        s_tmem = smem.alloc((4,), K.i32, align=16)

        p_kv = K.Pipeline(smem, 1, full="tma", empty="tcgen05")
        p_akk1 = K.Pipeline(smem, 2, full="tma", empty="tcgen05")
        p_tiles = K.Pipeline(smem, 2, full="mbar", empty="tcgen05", init_full=256)
        p_hs = K.Pipeline(smem, 1, full="mbar", empty="mbar", init_full=256, init_empty=9)
        p_w = K.Pipeline(smem, 2, full="tcgen05", empty="mbar", init_empty=256)
        p_vn = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=256)
        p_g = K.Pipeline(smem, 1, full="tma", empty="mbar", init_full=33, init_empty=256)
        b_kvT_done = K.TCGen05Bar(smem, 1); b_kvT_done.init(1)
        b_kv_read = K.MBarrier(smem, 1); b_kv_read.init(256)

        p_qk = K.Pipeline(smem, 1, full="tma", empty="tcgen05")
        b_qkT_done = K.TCGen05Bar(smem, 1); b_qkT_done.init(1)
        b_qk_read = K.MBarrier(smem, 1); b_qk_read.init(256)
        b_g_full = K.TMABar(smem, 1); b_g_full.init(33)
        b_g_free = K.MBarrier(smem, 1); b_g_free.init(256)
        b_bdo_full = K.TMABar(smem, 2); b_bdo_full.init(1)
        b_baqk_full = K.TMABar(smem, 1); b_baqk_full.init(1)
        b_baqk_masked = K.MBarrier(smem, 1); b_baqk_masked.init(32)
        b_baqk_empty = K.TCGen05Bar(smem, 1); b_baqk_empty.init(1)
        b_bakk_full = K.TMABar(smem, 1); b_bakk_full.init(1)
        b_bakk_empty = K.TCGen05Bar(smem, 1); b_bakk_empty.init(1)
        b_dhb_stored = K.MBarrier(smem, 1); b_dhb_stored.init(1)
        MB = {}
        for nm in ("prep_ready", "wT_ready", "dhb_ready", "dv2T_ready"):
            MB[nm] = K.MBarrier(smem, 1)
            MB[nm].init(256)
        TC = {}
        for nm in ("W_done", "dv2_done", "dh_done"):
            TC[nm] = K.TCGen05Bar(smem, 1)
            TC[nm].init(1)

        b_in_full = K.TMABar(smem, 1); b_in_full.init(33)
        b_eg_full = K.TMABar(smem, 1); b_eg_full.init(1)
        b_mid_free = K.MBarrier(smem, 1); b_mid_free.init(256)
        b_qk_free = K.MBarrier(smem, 1); b_qk_free.init(256)
        b_do_full = K.TMABar(smem, 1); b_do_full.init(1)
        b_h_full = K.TMABar(smem, 1); b_h_full.init(1)
        b_dhb_full = K.TMABar(smem, 1); b_dhb_full.init(1)
        b_aqk_full = K.TMABar(smem, 1); b_aqk_full.init(1)
        b_akk_full = K.TMABar(smem, 2); b_akk_full.init(1)
        b_do_empty = K.TCGen05Bar(smem, 1); b_do_empty.init(1)
        b_h_free = K.MBarrier(smem, 1); b_h_free.init(256)
        b_aqk_empty = K.TCGen05Bar(smem, 1); b_aqk_empty.init(1)
        b_akk_empty = K.TCGen05Bar(smem, 2); b_akk_empty.init(1)
        mbg_names = ["t_early", "zT_ready", "vnT_ready", "dv2T_ready",
                    "dAqk_tile_ready", "dAm_ready", "X_ready", "intra_ready", "dv_epi_done"]
        MBG = {}
        for nm in mbg_names:
            MBG[nm] = K.MBarrier(smem, 1)
            MBG[nm].init(256)
        b_dg0_ready = K.MBarrier(smem, 1); b_dg0_ready.init(256)
        b_aqk_masked = K.MBarrier(smem, 1); b_aqk_masked.init(64)
        b_akk_masked = K.MBarrier(smem, 2); b_akk_masked.init(64)
        tcg_names = ["Z_done", "Vn_done", "dv2_done", "dAqk_done", "dk_done",
                    "dAs_done", "dvb_done", "X_done", "Y_done",
                    "dq2_done", "dkt_done", "chunk_done", "xT_done"]
        TCG = {}
        for nm in tcg_names:
            TCG[nm] = K.TCGen05Bar(smem, 1)
            TCG[nm].init(1)

        TT = smem.alloc((27, 64, 64), K.bf16, swizzle=K.SW128B)
        s_beta1 = smem.alloc((2, CHUNK), K.f32, align=16)
        s_bbeta = smem.alloc((2, CHUNK), K.f32, align=16)

        s_beta = smem.alloc((2, 64), K.f32, align=16)
        s_dgk = smem.alloc((2, 128), K.f32, align=16)

        s_seq = smem.alloc((MAXSEQ, 4), K.i32, align=16)

        s_work = smem.alloc((ITEM_RING,), K.i32, align=16)
        b_work = K.MBarrier(smem, ITEM_RING); b_work.init(1)
        s_ident = smem.alloc((256,), K.bf16, align=128)

        with K.If(K.thread_id() == 0), K.Then():
            K.ptx.st.shared.s32(K.address_of(s_tmem[1]), K.int32(0))
            K.ptx.fence.mbarrier_init.release.cluster()
        with K.If(K.thread_id() < K.int32(256)), K.Then():
            tid_i = K.thread_id()
            n_i = tid_i >> 4
            k_i = tid_i & K.int32(15)
            K.ptx.st.shared.u16(
                s_ident.ptr_to([(n_i >> 3) * K.int32(128) + (k_i >> 3) * K.int32(64) + (n_i & K.int32(7)) * K.int32(8) + (k_i & K.int32(7))]),
                K.Cast("uint16", K.Select(n_i == k_i, K.int32(0x3F80), K.int32(0))))
            K.ptx[FENCE_ASYNC]()
        K.cuda.cta_sync()
        with K.If(K.warp_id() == 8), K.Then():
            K.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                K.address_of(s_tmem[0]), K.uint32(TMEM_COLS))
        with K.If((K.warp_id() == 0) & (num_seqs <= K.int32(MAXSEQ))), K.Then():
            lane0 = K.lane_id()
            with K.serial((num_seqs + K.int32(31)) >> 5) as blk:
                i = blk * K.int32(32) + lane0
                with K.If(i < num_seqs), K.Then():
                    st4 = K.alloc_local([4], "int32")
                    K.ptx["ld.global.nc.v4.s32"](st4[0], st4[1], st4[2], st4[3],
                                                 seq_tab.ptr_to([i * K.int32(4)]))
                    for j in range(4):
                        K.ptx.st.shared.s32(K.address_of(s_seq[i, j]), st4[j])
        with K.If(K.thread_id() == 0), K.Then():
            K.ptx.st.shared.s32(K.address_of(s_work[0]), cta)
            b_work.arrive(0)
        K.cuda.cta_sync()

        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def tmem_preamble():
            tmv = K.alloc_local([1], "int32")
            K.ptx.ld.volatile.shared.s32(tmv[0], K.address_of(s_tmem[0]))
            return tmv

        def pack_bf16x2(dst, lo, hi):
            K.ptx.cvt.rn.bf16x2.f32(dst, hi, lo)

        def make_phaser():
            """Sequential IKET ranges for one role: phase(name) ends the current range and starts the next."""
            tok = K.alloc_local([1], "uint32")
            K.assign(tok[0], K.cuda.iket.sentinel_token("idle"))

            def phase(name):
                K.cuda.iket.range_end(tok[0])
                K.assign(tok[0], K.cuda.iket.range_start(name))

            def phase_end():
                K.cuda.iket.range_end(tok[0])
                K.assign(tok[0], K.cuda.iket.sentinel_token("idle"))

            return phase, phase_end

        def bf16_bits_to_f32(u16val):
            return K.reinterpret("float32", K.Cast("uint32", u16val) << K.uint32(16))

        def lo(w):
            return K.reinterpret("float32", w << K.uint32(16))

        def hi(w):
            return K.reinterpret("float32", w & K.uint32(0xFFFF0000))

        def seq_info(seq):
            """(bos, seq_len, nch) of a sequence from the SMEM table, or from cu_seqlens when it does not fit."""
            bos = K.local_scalar("int64", init=K.int64(0))
            seq_len = K.local_scalar("int32", init=K.int32(0))
            with K.If(num_seqs <= K.int32(MAXSEQ)):
                with K.Then():
                    b32 = K.local_scalar("int32")
                    K.ptx.ld.shared.s32(b32, K.address_of(s_seq[seq, 0]))
                    K.assign(bos, K.Cast("int64", b32))
                    K.ptx.ld.shared.s32(seq_len, K.address_of(s_seq[seq, 1]))
                with K.Else():
                    cs = K.alloc_local([2], "int64")
                    K.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([seq]))
                    K.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([seq + K.int32(1)]))
                    K.assign(bos, cs[0])
                    K.assign(seq_len, K.Cast("int32", cs[1] - cs[0]))
            nch = K.local_scalar("int32", init=(seq_len + K.int32(CHUNK - 1)) >> 6)
            return bos, seq_len, nch

        def stream_coords(s):
            """Stream rank s -> (is_fwd, seq, hv, hq, bos, seq_len, nch) from the host-built stream table."""
            sv = K.local_scalar("int32")
            K.ptx.ld.global_.nc.s32(sv, stream_tab.ptr_to([s]))
            is_fwd = K.local_scalar("int32", init=sv >> K.int32(30))
            seq = K.local_scalar("int32", init=(sv >> K.int32(15)) & K.int32(0x7FFF))
            hv = K.local_scalar("int32", init=sv & K.int32(0x7FFF))
            hq = K.local_scalar("int32", init=hv // K.int32(G))
            bos, seq_len, nch = seq_info(seq)
            return is_fwd, seq, hv, hq, bos, seq_len, nch

        def chunk_base(seq):
            cb = K.local_scalar("int32", init=K.int32(0))
            with K.If(num_seqs <= K.int32(MAXSEQ)):
                with K.Then():
                    K.ptx.ld.shared.s32(cb, K.address_of(s_seq[seq, 3]))
                with K.Else():
                    with K.serial(seq) as i:
                        cs = K.alloc_local([2], "int64")
                        K.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([i]))
                        K.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([i + 1]))
                        K.assign(cb, cb + ((K.Cast("int32", cs[1] - cs[0]) + K.int32(CHUNK - 1)) >> 6))
            return cb

        def chunk_rows(seq_len, n):
            return K.min(K.int32(CHUNK), seq_len - n * K.int32(CHUNK))

        def load_beta_lanes(dst_ptr_fn, bos, hv, n, rows):
            """Loader warp: every lane fetches beta for tokens lane and lane+32 of the chunk."""
            lane = K.lane_id()
            for j in range(2):
                t = lane + K.int32(32 * j)
                tokc = bos + K.Cast("int64", n * K.int32(CHUNK) + K.min(t, rows - K.int32(1)))
                u = K.local_scalar("uint16")
                K.ptx.ld.global_.nc.u16(u, beta.ptr_to([tokc * K.int64(HV) + K.Cast("int64", hv)]))
                val = K.Select(t < rows, bf16_bits_to_f32(u), K.float32(0.0))
                K.ptx.st.shared.f32(dst_ptr_fn(t), val)

        def seq_len_of(sq):
            cs = K.alloc_local([2], "int64")
            K.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([sq]))
            K.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([sq + 1]))
            return cs[0], K.Cast("int32", cs[1] - cs[0])

        def item_coords(item):
            """Item index -> (c, hq, seq, n, bos, rows, nch).

            (seq, n) comes from the host-built table, which orders chunks by their predicted
            recurrence readiness (both streams of the sequence have published the chunk); the
            qk-head index is the fastest-varying component.
            """
            pos = K.local_scalar("int32", init=item // K.int32(HQ))
            hq = K.local_scalar("int32", init=item - pos * K.int32(HQ))
            iv = K.local_scalar("int32")
            K.ptx.ld.global_.nc.s32(iv, item_tab.ptr_to([pos]))
            seq = K.local_scalar("int32", init=iv >> K.int32(16))
            n = K.local_scalar("int32", init=iv & K.int32(0xFFFF))
            bos, seq_len, nch = seq_info(seq)
            cb = chunk_base(seq)
            c = K.local_scalar("int32", init=cb + n)
            rows = K.local_scalar("int32", init=K.min(K.int32(CHUNK), seq_len - n * K.int32(CHUNK)))
            return c, hq, seq, n, bos, rows, nch

        def load_beta_lanes_g(bos, hv, n, rows, slot):
            lane = K.lane_id()
            for j in range(2):
                t = lane + K.int32(32 * j)
                tokc = bos + K.Cast("int64", n * K.int32(CHUNK) + K.min(t, rows - K.int32(1)))
                u = K.local_scalar("uint16")
                K.ptx.ld.global_.nc.u16(u, beta.ptr_to([tokc * K.int64(HV) + K.Cast("int64", hv)]))
                val = K.Select(t < rows, K.reinterpret("float32", K.Cast("uint32", u) << K.uint32(16)), K.float32(0.0))
                K.ptx.st.shared.f32(K.address_of(s_beta[slot, t]), val)


        def work_wait(j):
            """The j-th work unit of this CTA (published by the loader); >= total_work means done."""
            slot = j % K.int32(ITEM_RING)
            b_work.wait(slot, (j // K.int32(ITEM_RING)) & K.int32(1))
            v_ = K.local_scalar("int32")
            K.ptx.ld.shared.s32(v_, K.address_of(s_work[slot]))
            return v_

        def claim_publish(j):
            """Loader: claim work unit j for the CTA and publish it in the ring."""
            slot = j % K.int32(ITEM_RING)
            with K.If(elected()), K.Then():
                nxt = K.local_scalar("int32")
                K.ptx["atom.acq_rel.gpu.global.add.s32"](nxt, stream_counter.ptr_to([0]), K.int32(1))
                K.ptx.st.shared.s32(K.address_of(s_work[slot]), nxt + num_ctas)
                b_work.arrive(slot)


        kk_ = K.local_scalar("int32", init=K.int32(0))
        cur = K.local_scalar("int32", init=work_wait(kk_))

        def g_masker(MROW):
            cyc = K.local_scalar("int32", init=K.int32(0))
            lane = K.lane_id()
            rowc = K.local_scalar("int32", init=MROW * K.int32(32) + lane)
            item = K.local_scalar("int32", init=cur - num_streams)
            with K.While(cur < total_work):
                K.assign(item, cur - num_streams)
                c, hq, seq, n, bos, rows, nch_i = item_coords(item)
                with K.serial(G) as gi:
                    par = cyc & K.int32(1)
                    b_aqk_full.wait(0, par)
                    diag = K.alloc_local([4], "uint32")
                    dmat = lane >> K.int32(3)
                    dblk = MROW * K.int32(4) + dmat
                    dptr = TT[S_AQK].ptr_to(dblk * K.int32(8) + (lane & K.int32(7)), dblk * K.int32(8))
                    K.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](diag[0], diag[1], diag[2], diag[3], dptr)
                    drow = lane >> K.int32(2)
                    dcol = (lane & K.int32(3)) * K.int32(2)
                    dmask = K.Select(dcol > drow, K.uint32(0),
                                     K.Select(dcol == drow, K.uint32(0x0000FFFF), K.uint32(0xFFFFFFFF)))
                    for e in range(4):
                        blk_row = (MROW * K.int32(4) + K.int32(e)) * K.int32(8) + drow
                        K.assign(diag[e], K.Select(blk_row < rows, diag[e] & dmask, K.uint32(0)))
                    K.ptx["stmatrix.sync.aligned.m8n8.x4.shared.b16"](dptr, diag[0], diag[1], diag[2], diag[3])
                    for u in range(1, 8):
                        with K.If(K.int32(8 * u) > rowc), K.Then():
                            K.ptx["st.shared.v4.b32"](TT[S_AQK].ptr_to(rowc, 8 * u),
                                                      K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0))
                    with K.If(rowc >= rows), K.Then():
                        for u in range(8):
                            K.ptx["st.shared.v4.b32"](TT[S_AQK].ptr_to(rowc, 8 * u),
                                                      K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0))
                    K.ptx[FENCE_ASYNC]()
                    b_aqk_masked.arrive(0)

                    b_akk_full.wait(par, (cyc >> 1) & K.int32(1))
                    with K.If(rowc >= rows), K.Then():
                        for u in range(8):
                            K.ptx["st.shared.v4.b32"](TT[S_AKK + par].ptr_to(rowc, 8 * u),
                                                      K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0))
                    K.ptx[FENCE_ASYNC]()
                    b_akk_masked.arrive(par)
                    K.assign(cyc, cyc + K.int32(1))
                K.assign(kk_, kk_ + K.int32(1))
                K.assign(cur, work_wait(kk_))


        with cg:
            tm = tmem_preamble()
            wr = K.warp_id_in_role()
            lane = K.lane_id()
            wg = K.local_scalar("int32", init=wr >> 2)
            quad = K.local_scalar("int32", init=wr & 3)
            x = K.local_scalar("int32", init=quad * 32 + lane)
            row0 = K.local_scalar("int32", init=wg * 32)
            x64 = K.Cast("int64", x)
            xs = K.local_scalar("int32", init=x >> 6)
            xr = K.local_scalar("int32", init=x & 63)
            xg = K.local_scalar("int32", init=x >> 5)
            xgc = K.local_scalar("int32", init=(x & 31) * 2)

            def tmem_at(col):
                return K.Cast("uint32", tm[0] + col + (quad << 21))

            def bar_all():
                K.ptx.bar.sync(K.uint32(1), K.uint32(256))


            st_te = K.PipelineState(2, phase=0)
            st_hs = K.PipelineState(1, phase=1)
            st_g = K.PipelineState(1, phase=0)
            st_w = K.PipelineState(2, phase=0)
            st_vn = K.PipelineState(1, phase=0)
            fkv = K.local_scalar("int32", init=K.int32(0))
            bqk = K.local_scalar("int32", init=K.int32(0))
            st_bg = K.PipelineState(1, phase=0)
            bcyc = K.local_scalar("int32", init=K.int32(0))

            gv = K.alloc_local([32], "float32")
            kk = K.alloc_local([32], "float32")
            vv = K.alloc_local([32], "float32")
            bb = K.alloc_local([32], "float32")
            acc = K.alloc_local([64], "float32")
            wds = K.alloc_local([32], "uint32")
            gn = K.local_scalar("float32")
            egn = K.local_scalar("float32")
            eg = K.local_scalar("float32")
            egng = K.local_scalar("float32")
            bu = K.local_scalar("uint16")
            ku = K.local_scalar("uint16")
            vu = K.local_scalar("uint16")
            qu = K.local_scalar("uint16")
            phase, phase_end = make_phaser()

            def fwd_body(seq, hv, hq, bos, seq_len, nch):
                """Forward state recurrence, software-pipelined: while the tensor core runs chunk n's
                Vn / state-update MMAs, the compute warps prepare chunk n+1's tiles (from K/V transposed
                into TMEM by the MMA warp) and read chunk n+2's gate."""
                hv64 = K.Cast("int64", hv)
                gcol = K.local_scalar("int64", init=hv64 * K.int64(D) + x64)
                gn_c = K.local_scalar("float32", init=K.float32(0.0))
                gn_1 = K.local_scalar("float32", init=K.float32(0.0))
                gn_2 = K.local_scalar("float32", init=K.float32(0.0))
                bpair = K.alloc_local([2], "float32")

                def f_g(m, gn_dst):
                    """g of chunk m (32 rows of this channel) into gv; its last valid row into gn_dst."""
                    rows = chunk_rows(seq_len, m)
                    phase("fw-g")
                    p_g.full.wait(0, st_g.phase)
                    phase("f-g")
                    gst = K.local_scalar("int32", init=F_G + xg)
                    for i in range(32):
                        K.ptx.ld.shared.f32(gv[i], TT[gst].ptr_to(row0 + i, xgc))
                    K.ptx.ld.shared.f32(gn_dst, TT[gst].ptr_to(rows - K.int32(1), xgc))
                    K.ptx[FENCE_ASYNC]()
                    p_g.empty.arrive(0)
                    st_g.advance()

                def f_tiles(m, gn_m, u_lo, u_hi):
                    """K/V of chunk m from their TMEM transposes (first half only), then token blocks
                    [u_lo, u_hi) of its kg / kbg / vb tiles into set m & 1 (the last half publishes)."""
                    rows = K.local_scalar("int32", init=chunk_rows(seq_len, m))
                    tok0 = K.local_scalar("int64", init=bos + K.Cast("int64", m * K.int32(CHUNK)))
                    sset = K.local_scalar("int32", init=m & K.int32(1))
                    if u_lo == 0:
                        phase("fw-kv")
                        b_kvT_done.wait(0, fkv & K.int32(1))
                        K.ptx[TC_FENCE_AFTER]()
                        phase("f-kv")
                        K.ptx[TC_LD32](*(kk[i] for i in range(32)), tmem_at(TM_KT + wg * 32))
                        K.ptx[TC_LD32](*(vv[i] for i in range(32)), tmem_at(TM_VT + wg * 32))
                        K.ptx[WAIT_LD]()
                        K.ptx[TC_FENCE_BEFORE]()
                        b_kv_read.arrive(0)
                        K.assign(fkv, fkv + K.int32(1))
                    phase("f-tiles")
                    kg_t = K.local_scalar("int32", init=F_KG + sset * K.int32(2) + xs)
                    kbg_t = K.local_scalar("int32", init=F_KBG + sset * K.int32(2) + xs)
                    vb_t = K.local_scalar("int32", init=F_VB + sset * K.int32(2) + xs)
                    for u in range(u_lo, u_hi):
                        wkg = K.alloc_local([4], "uint32")
                        wkbg = K.alloc_local([4], "uint32")
                        wvb = K.alloc_local([4], "uint32")
                        for p in range(4):
                            i = 8 * u + 2 * p
                            valid0 = row0 + K.int32(i) < rows
                            valid1 = row0 + K.int32(i + 1) < rows
                            m0 = K.Select(valid0, K.float32(1.0), K.float32(0.0))
                            m1 = K.Select(valid1, K.float32(1.0), K.float32(0.0))
                            eg0 = K.local_scalar("float32")
                            eg1 = K.local_scalar("float32")
                            en0 = K.local_scalar("float32")
                            en1 = K.local_scalar("float32")
                            K.ptx.ex2.approx.ftz.f32(eg0, gv[i])
                            K.ptx.ex2.approx.ftz.f32(eg1, gv[i + 1])
                            K.ptx.ex2.approx.ftz.f32(en0, gn_m - gv[i])
                            K.ptx.ex2.approx.ftz.f32(en1, gn_m - gv[i + 1])
                            bu0 = K.local_scalar("uint16")
                            bu1 = K.local_scalar("uint16")
                            K.ptx.cvt.rn.bf16.f32(bu0, eg0)
                            K.ptx.cvt.rn.bf16.f32(bu1, eg1)
                            egidx0 = (tok0 + K.Cast("int64", row0 + K.int32(i))) * HVK64 + gcol
                            egidx1 = (tok0 + K.Cast("int64", row0 + K.int32(i + 1))) * HVK64 + gcol
                            with K.If(valid0), K.Then():
                                K.ptx["st.global.L1::no_allocate.b16"](egcache.ptr_to([egidx0]), bu0)
                            with K.If(valid1), K.Then():
                                K.ptx["st.global.L1::no_allocate.b16"](egcache.ptr_to([egidx1]), bu1)
                            K.ptx["ld.shared.v2.f32"](bpair[0], bpair[1], K.address_of(s_beta1[sset, row0 + i]))
                            pair0 = K.local_scalar("uint64")
                            pair1 = K.local_scalar("uint64")
                            pair2 = K.local_scalar("uint64")
                            K.ptx["mul.rn.f32x2"](
                                pair0, K.cuda.make_float2(kk[i], kk[i + 1]),
                                K.cuda.make_float2(en0, en1),
                            )
                            K.ptx["mul.rn.f32x2"](pair0, pair0, K.cuda.make_float2(m0, m1))
                            K.ptx["mul.rn.f32x2"](
                                pair1, K.cuda.make_float2(kk[i], kk[i + 1]),
                                K.cuda.make_float2(bpair[0], bpair[1]),
                            )
                            K.ptx["mul.rn.f32x2"](pair1, pair1, K.cuda.make_float2(eg0, eg1))
                            K.ptx["mul.rn.f32x2"](
                                pair2, K.cuda.make_float2(vv[i], vv[i + 1]),
                                K.cuda.make_float2(bpair[0], bpair[1]),
                            )
                            pack_bf16x2(wkg[p], K.cuda.float2_x(pair0), K.cuda.float2_y(pair0))
                            pack_bf16x2(wkbg[p], K.cuda.float2_x(pair1), K.cuda.float2_y(pair1))
                            pack_bf16x2(wvb[p], K.cuda.float2_x(pair2), K.cuda.float2_y(pair2))
                        col = row0 + 8 * u
                        K.ptx["st.shared.v4.b32"](TT[kg_t].ptr_to(xr, col), wkg[0], wkg[1], wkg[2], wkg[3])
                        K.ptx["st.shared.v4.b32"](TT[kbg_t].ptr_to(xr, col), wkbg[0], wkbg[1], wkbg[2], wkbg[3])
                        K.ptx["st.shared.v4.b32"](TT[vb_t].ptr_to(xr, col), wvb[0], wvb[1], wvb[2], wvb[3])
                    if u_hi == 4:
                        K.ptx[FENCE_ASYNC]()

                        K.ptx["fence.proxy.async.global"]()
                        p_tiles.full.arrive(sset)

                f_g(K.int32(0), gn_c)
                f_tiles(K.int32(0), gn_c, 0, 4)
                with K.If(nch > K.int32(1)), K.Then():
                    f_g(K.int32(1), gn_1)
                with K.serial(nch) as n:
                    sn = K.local_scalar("int32", init=n & K.int32(1))
                    K.ptx.ex2.approx.ftz.f32(egn, gn_c)
                    phase("fw-hupd")

                    with K.If(n > K.int32(0)), K.Then():
                        p_tiles.empty.wait(st_te.stage, st_te.phase)
                        st_te.advance()
                    K.ptx[TC_FENCE_AFTER]()
                    K.ptx[FENCE_ASYNC]()
                    phase("fw-hs")
                    p_hs.empty.wait(st_hs.stage, st_hs.phase)
                    K.ptx[TC_FENCE_AFTER]()
                    phase("f-hdecay")
                    hc0 = wg * 64
                    hsst = K.local_scalar("int32", init=F_HS + wg * K.int32(2) + xs)
                    with K.If(n == K.int32(0)):
                        with K.Then():
                            h0base = ((K.Cast("int64", seq) * K.int64(HV) + hv64) * K.int64(D) + x64) * K.int64(D) \
                                + K.Cast("int64", hc0)
                            for m8 in range(8):
                                K.ptx["ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.f32"](
                                    *(acc[8 * m8 + i] for i in range(8)),
                                    h0.ptr_to([h0base + K.int64(8 * m8)]))
                        with K.Else():
                            K.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_H + hc0))
                            K.ptx[TC_LD32](*(acc[32 + i] for i in range(32)), tmem_at(TM_H + hc0 + 32))
                            K.ptx[WAIT_LD]()
                    for p in range(32):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    for u in range(8):
                        K.ptx["st.shared.v4.b32"](TT[hsst].ptr_to(xr, 8 * u),
                                                  wds[4 * u], wds[4 * u + 1], wds[4 * u + 2], wds[4 * u + 3])
                    for p in range(32):
                        dpair = K.local_scalar("uint64")
                        K.ptx["mul.rn.f32x2"](dpair, K.cuda.make_float2(acc[2 * p], acc[2 * p + 1]),
                                              K.cuda.make_float2(egn, egn))
                        K.assign(acc[2 * p], K.cuda.float2_x(dpair))
                        K.assign(acc[2 * p + 1], K.cuda.float2_y(dpair))
                    K.ptx[TC_ST32](tmem_at(TM_H + hc0), *(acc[i] for i in range(32)))
                    K.ptx[TC_ST32](tmem_at(TM_H + hc0 + 32), *(acc[32 + i] for i in range(32)))
                    K.ptx[WAIT_ST]()
                    K.ptx[TC_FENCE_BEFORE]()
                    K.ptx[FENCE_ASYNC]()
                    p_hs.full.arrive(st_hs.stage)
                    phase("fw-W")
                    p_w.full.wait(st_w.stage, st_w.phase)
                    K.ptx[TC_FENCE_AFTER]()
                    K.ptx[FENCE_ASYNC]()
                    phase("f-wT")
                    K.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_W0 + sn * 64 + wg * 32))
                    K.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    wt_t = K.local_scalar("int32", init=F_KBG + sn * K.int32(2) + xs)
                    for u in range(4):
                        K.ptx["st.shared.v4.b32"](TT[wt_t].ptr_to(xr, row0 + 8 * u),
                                                  wds[4 * u], wds[4 * u + 1], wds[4 * u + 2], wds[4 * u + 3])
                    K.ptx[TC_FENCE_BEFORE]()
                    K.ptx[FENCE_ASYNC]()
                    p_w.empty.arrive(st_w.stage)
                    st_w.advance()
                    with K.If(n + K.int32(1) < nch), K.Then():
                        f_tiles(n + K.int32(1), gn_1, 0, 2)
                    phase("fw-Vn")
                    p_vn.full.wait(0, st_vn.phase)
                    st_vn.advance()
                    K.ptx[TC_FENCE_AFTER]()
                    K.ptx[FENCE_ASYNC]()
                    phase("f-vnT")
                    K.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_U0 + sn * 64 + wg * 32))
                    K.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    vn_t = K.local_scalar("int32", init=F_VB + sn * K.int32(2) + xs)
                    for u in range(4):
                        K.ptx["st.shared.v4.b32"](TT[vn_t].ptr_to(xr, row0 + 8 * u),
                                                  wds[4 * u], wds[4 * u + 1], wds[4 * u + 2], wds[4 * u + 3])
                    K.ptx[TC_FENCE_BEFORE]()
                    K.ptx[FENCE_ASYNC]()
                    p_vn.empty.arrive(0)
                    with K.If(elected()), K.Then():
                        p_hs.empty.arrive(st_hs.stage)
                    st_hs.advance()
                    with K.If(n + K.int32(1) < nch), K.Then():
                        f_tiles(n + K.int32(1), gn_1, 2, 4)
                    with K.If(n + K.int32(2) < nch), K.Then():
                        f_g(n + K.int32(2), gn_2)
                    K.assign(gn_c, gn_1)
                    K.assign(gn_1, gn_2)
                    phase_end()

                p_tiles.empty.wait(st_te.stage, st_te.phase)
                st_te.advance()
                K.ptx[TC_FENCE_AFTER]()

            def bwd_body(seq, hv, hq, bos, seq_len, nch):
                """Backward state-gradient recurrence, software-pipelined: chunk n-1's operand
                preparation (q^T/k^T from TMEM transposes, T1/kbg into TMEM, T2 into shared memory)
                overlaps chunk n's dv2 and dh MMAs."""
                hv64 = K.Cast("int64", hv)
                gn_c = K.local_scalar("float32", init=K.float32(0.0))
                gn_1 = K.local_scalar("float32", init=K.float32(0.0))
                bpair = K.alloc_local([2], "float32")

                def b_prep(m, gn_dst):
                    """g of chunk m into gv (its last valid row into gn_dst), then q^T / k^T from TMEM."""
                    rows = chunk_rows(seq_len, m)
                    phase("bw-in")
                    b_g_full.wait(0, st_bg.phase)
                    phase("b-prep")
                    gst = K.local_scalar("int32", init=B_G + xg)
                    for i in range(32):
                        K.ptx.ld.shared.f32(gv[i], TT[gst].ptr_to(row0 + i, xgc))
                    K.ptx.ld.shared.f32(gn_dst, TT[gst].ptr_to(rows - K.int32(1), xgc))
                    K.ptx[FENCE_ASYNC]()
                    b_g_free.arrive(0)
                    st_bg.advance()
                    phase("bw-qk")
                    b_qkT_done.wait(0, bqk & K.int32(1))
                    K.ptx[TC_FENCE_AFTER]()
                    phase("b-qk")
                    K.ptx[TC_LD32](*(vv[i] for i in range(32)), tmem_at(TM_QT + wg * 32))
                    K.ptx[TC_LD32](*(kk[i] for i in range(32)), tmem_at(TM_KTB + wg * 32))
                    K.ptx[WAIT_LD]()
                    K.ptx[TC_FENCE_BEFORE]()
                    b_qk_read.arrive(0)
                    K.assign(bqk, bqk + K.int32(1))

                def b_prep2(m, sset, bslot):
                    """T1 (TMEM set sset), T2 (shared) and kbg (TMEM set sset) of chunk m."""
                    rows = K.local_scalar("int32", init=chunk_rows(seq_len, m))
                    phase("b-prep2")
                    w1 = K.alloc_local([16], "uint32")
                    w3 = K.alloc_local([16], "uint32")
                    for u in range(4):
                        w2 = K.alloc_local([4], "uint32")
                        for p in range(4):
                            i = 8 * u + 2 * p
                            m0 = K.Select(row0 + K.int32(i) < rows, K.float32(1.0), K.float32(0.0))
                            m1 = K.Select(row0 + K.int32(i + 1) < rows, K.float32(1.0), K.float32(0.0))
                            eg0 = K.local_scalar("float32")
                            eg1 = K.local_scalar("float32")
                            en0 = K.local_scalar("float32")
                            en1 = K.local_scalar("float32")
                            K.ptx.ex2.approx.ftz.f32(eg0, gv[i])
                            K.ptx.ex2.approx.ftz.f32(eg1, gv[i + 1])
                            K.ptx.ex2.approx.ftz.f32(en0, K.float32(0.0) - gv[i])
                            K.ptx.ex2.approx.ftz.f32(en1, K.float32(0.0) - gv[i + 1])
                            K.ptx["ld.shared.v2.f32"](bpair[0], bpair[1], K.address_of(s_bbeta[bslot, row0 + i]))
                            pair0 = K.local_scalar("uint64")
                            pair1 = K.local_scalar("uint64")
                            pair2 = K.local_scalar("uint64")
                            K.ptx["mul.rn.f32x2"](
                                pair0, K.cuda.make_float2(vv[i], vv[i + 1]),
                                K.cuda.make_float2(eg0, eg1),
                            )
                            K.ptx["mul.rn.f32x2"](
                                pair0, pair0, K.cuda.make_float2(scale, scale)
                            )
                            K.ptx["mul.rn.f32x2"](pair0, pair0, K.cuda.make_float2(m0, m1))
                            K.ptx["mul.rn.f32x2"](
                                pair1, K.cuda.make_float2(kk[i], kk[i + 1]),
                                K.cuda.make_float2(en0, en1),
                            )
                            K.ptx["mul.rn.f32x2"](pair1, pair1, K.cuda.make_float2(m0, m1))
                            K.ptx["mul.rn.f32x2"](
                                pair2, K.cuda.make_float2(kk[i], kk[i + 1]),
                                K.cuda.make_float2(bpair[0], bpair[1]),
                            )
                            K.ptx["mul.rn.f32x2"](pair2, pair2, K.cuda.make_float2(eg0, eg1))
                            pack_bf16x2(w1[4 * u + p], K.cuda.float2_x(pair0), K.cuda.float2_y(pair0))
                            pack_bf16x2(w2[p], K.cuda.float2_x(pair1), K.cuda.float2_y(pair1))
                            pack_bf16x2(w3[4 * u + p], K.cuda.float2_x(pair2), K.cuda.float2_y(pair2))
                        K.ptx["st.shared.v4.b32"](TT[B_T2 + xs].ptr_to(xr, row0 + 8 * u), w2[0], w2[1], w2[2], w2[3])
                    K.ptx[TC_ST16](tmem_at(TM_T1 + sset * 32 + wg * 16), *(w1[j] for j in range(16)))
                    K.ptx[TC_ST16](tmem_at(TM_KB + sset * 32 + wg * 16), *(w3[j] for j in range(16)))
                    K.ptx[WAIT_ST]()
                    K.ptx[TC_FENCE_BEFORE]()
                    K.ptx[FENCE_ASYNC]()
                    MB["prep_ready"].arrive(0)

                b_prep(nch - K.int32(1), gn_c)
                b_prep2(nch - K.int32(1), bcyc & K.int32(1), bcyc & K.int32(1))
                with K.serial(nch) as rn:
                    n = nch - K.int32(1) - rn
                    par = K.local_scalar("int32", init=bcyc & K.int32(1))
                    rows = K.local_scalar("int32", init=chunk_rows(seq_len, n))
                    K.ptx.ex2.approx.ftz.f32(egn, gn_c)
                    phase("bw-dh")
                    with K.If(rn > K.int32(0)), K.Then():
                        TC["dh_done"].wait(0, par ^ K.int32(1))
                    K.ptx[TC_FENCE_AFTER]()
                    K.ptx[FENCE_ASYNC]()
                    phase("b-dhb")
                    with K.If(rn == K.int32(0)):
                        with K.Then():
                            dbase = ((K.Cast("int64", seq) * K.int64(HV) + hv64) * K.int64(D) + x64) * K.int64(D) \
                                + K.Cast("int64", wg * 64)
                            for m8 in range(8):
                                K.ptx["ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.f32"](
                                    *(acc[8 * m8 + i] for i in range(8)),
                                    dht.ptr_to([dbase + K.int64(8 * m8)]))
                        with K.Else():
                            K.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_DH + wg * 64))
                            K.ptx[TC_LD32](*(acc[32 + i] for i in range(32)), tmem_at(TM_DH + wg * 64 + 32))
                            K.ptx[WAIT_LD]()
                    for p in range(32):
                        dpair = K.local_scalar("uint64")
                        K.ptx["mul.rn.f32x2"](dpair, K.cuda.make_float2(acc[2 * p], acc[2 * p + 1]),
                                              K.cuda.make_float2(egn, egn))
                        K.assign(acc[2 * p], K.cuda.float2_x(dpair))
                        K.assign(acc[2 * p + 1], K.cuda.float2_y(dpair))
                    K.ptx[TC_ST32](tmem_at(TM_DH + wg * 64), *(acc[i] for i in range(32)))
                    K.ptx[TC_ST32](tmem_at(TM_DH + wg * 64 + 32), *(acc[32 + i] for i in range(32)))
                    phase("bw-stored")
                    with K.If(rn > K.int32(0)), K.Then():
                        b_dhb_stored.wait(0, par ^ K.int32(1))
                    phase("b-dhb2")
                    for p in range(32):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    dhst = K.local_scalar("int32", init=B_DHB + wg * K.int32(2) + xs)
                    for u in range(8):
                        K.ptx["st.shared.v4.b32"](TT[dhst].ptr_to(xr, 8 * u),
                                                  wds[4 * u], wds[4 * u + 1], wds[4 * u + 2], wds[4 * u + 3])
                    K.ptx[WAIT_ST]()
                    K.ptx[TC_FENCE_BEFORE]()
                    K.ptx[FENCE_ASYNC]()
                    MB["dhb_ready"].arrive(0)
                    phase("bw-W")
                    TC["W_done"].wait(0, par)
                    K.ptx[TC_FENCE_AFTER]()
                    phase("b-wT")
                    K.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_BW + wg * 32))
                    K.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])

                    K.ptx[TC_ST16](tmem_at(TM_KB + par * 32 + wg * 16), *(wds[j] for j in range(16)))
                    K.ptx[WAIT_ST]()
                    K.ptx[TC_FENCE_BEFORE]()
                    MB["wT_ready"].arrive(0)
                    with K.If(rn + K.int32(1) < nch), K.Then():
                        b_prep(n - K.int32(1), gn_1)
                    phase("bw-dv2")
                    TC["dv2_done"].wait(0, par)
                    K.ptx[TC_FENCE_AFTER]()
                    K.ptx[FENCE_ASYNC]()
                    phase("b-dv2T")
                    K.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_DV2 + wg * 32))
                    K.ptx[WAIT_LD]()
                    for p in range(16):
                        i = 2 * p
                        m0 = K.Select(row0 + K.int32(i) < rows, acc[i], K.float32(0.0))
                        m1 = K.Select(row0 + K.int32(i + 1) < rows, acc[i + 1], K.float32(0.0))
                        pack_bf16x2(wds[p], m0, m1)
                    for u in range(4):
                        K.ptx["st.shared.v4.b32"](TT[B_DV2 + xs].ptr_to(xr, row0 + 8 * u),
                                                  wds[4 * u], wds[4 * u + 1], wds[4 * u + 2], wds[4 * u + 3])
                    K.ptx[TC_FENCE_BEFORE]()
                    K.ptx[FENCE_ASYNC]()
                    MB["dv2T_ready"].arrive(0)
                    with K.If(rn + K.int32(1) < nch), K.Then():
                        b_prep2(n - K.int32(1), par ^ K.int32(1), par ^ K.int32(1))
                    K.assign(gn_c, gn_1)
                    phase_end()
                    K.assign(bcyc, bcyc + K.int32(1))
                TC["dh_done"].wait(0, (bcyc & K.int32(1)) ^ K.int32(1))
                K.ptx[TC_FENCE_AFTER]()
                K.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_DH + wg * 64))
                K.ptx[TC_LD32](*(acc[32 + i] for i in range(32)), tmem_at(TM_DH + wg * 64 + 32))
                K.ptx[WAIT_LD]()
                obase = ((K.Cast("int64", seq) * K.int64(HV) + hv64) * K.int64(D) + x64) * K.int64(D) \
                    + K.Cast("int64", wg * 64)
                for m8 in range(8):
                    K.ptx["st.global.L1::no_allocate.v8.f32"](
                        dh0.ptr_to([obase + K.int64(8 * m8)]),
                        *(acc[8 * m8 + i] for i in range(8)))
                b_dhb_stored.wait(0, (bcyc & K.int32(1)) ^ K.int32(1))

            with K.While(cur < num_streams):
                is_fwd, seq, hv, hq, bos, seq_len, nch = stream_coords(cur)
                with K.If(is_fwd == K.int32(1)):
                    with K.Then():
                        fwd_body(seq, hv, hq, bos, seq_len, nch)
                    with K.Else():
                        bwd_body(seq, hv, hq, bos, seq_len, nch)

                K.ptx.bar.sync(K.uint32(5), K.uint32(384))
                K.assign(kk_, kk_ + K.int32(1))
                K.assign(cur, work_wait(kk_))




            # Converge even when this CTA starts with an item and skips streams.
            K.ptx.bar.sync(K.uint32(6), K.uint32(256))

        with cg:
            tm = tmem_preamble()
            wr = K.warp_id_in_role()
            lane = K.lane_id()
            wg = K.local_scalar("int32", init=wr >> 2)
            quad = K.local_scalar("int32", init=wr & 3)
            x = K.local_scalar("int32", init=quad * 32 + lane)
            row0 = K.local_scalar("int32", init=wg * 32)
            x64 = K.Cast("int64", x)
            xs = K.local_scalar("int32", init=x >> 6)
            xr = K.local_scalar("int32", init=x & 63)
            phalf = K.local_scalar("int32", init=x & 1)
            pcol = K.local_scalar("int32", init=x & ~1)
            prow0 = K.local_scalar("int32", init=row0 + phalf * 16)
            ps = K.local_scalar("int32", init=pcol >> 6)
            pr = K.local_scalar("int32", init=pcol & 63)
            cyc = K.local_scalar("int32", init=K.int32(0))

            def tmem_at(col):
                return K.Cast("uint32", tm[0] + col + (quad << 21))

            def ld32(regs, col, base=0):
                K.ptx[TC_LD32](*(regs[base + i] for i in range(32)), tmem_at(col))

            def ld8(regs, col, base=0):
                K.ptx[TC_LD8](*(regs[base + i] for i in range(8)), tmem_at(col))

            def ld4(regs, col, base=0):
                K.ptx[TC_LD4](*(regs[base + i] for i in range(4)), tmem_at(col))

            def st8(col, regs, base=0):
                K.ptx[TC_ST8](tmem_at(col), *(regs[base + i] for i in range(8)))

            def st_row(stage0, col0, words, wbase=0, nunits=4):
                for u in range(nunits):
                    K.ptx["st.shared.v4.b32"](TT[stage0 + xs].ptr_to(xr, col0 + 8 * u),
                                              words[wbase + 4 * u], words[wbase + 4 * u + 1],
                                              words[wbase + 4 * u + 2], words[wbase + 4 * u + 3])

            def bar_all():
                K.ptx.bar.sync(K.uint32(1), K.uint32(256))

            def bar_wg():
                K.ptx.bar.sync(K.uint32(2) + K.Cast("uint32", wg), K.uint32(128))

            def twait(nm):
                TCG[nm].wait(0, cyc & K.int32(1))
                K.ptx[TC_FENCE_AFTER]()

                K.ptx[FENCE_ASYNC]()
                K.ptx["bar.warp.sync"](K.uint32(0xFFFFFFFF))

            def marrive(nm):
                K.ptx[TC_FENCE_BEFORE]()
                MBG[nm].arrive(0)

            def e_ptr(c, col):
                return TT[ST_G + (col >> 6)].ptr_to(c, col & 63)

            def load_transpose_frag(base, frag):
                col0 = (quad & K.int32(1)) * K.int32(32)
                tile = TT[base + xs]
                for rb in range(2):
                    for cb_ in range(2):
                        o = 4 * (2 * rb + cb_)
                        K.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](
                            frag[o], frag[o + 1], frag[o + 2], frag[o + 3],
                            tile.m8n8x4(row0 + K.int32(16 * rb), col0 + K.int32(16 * cb_), lane),
                        )

            def store_transpose_frag(base, frag):
                col0 = (quad & K.int32(1)) * K.int32(32)
                tile = TT[base + xs]
                mm = lane >> K.int32(3)
                jj = lane & K.int32(7)
                for rb in range(2):
                    for cb_ in range(2):
                        o = 4 * (2 * rb + cb_)
                        ptr = tile.ptr_to(
                            col0 + K.int32(16 * cb_) + (mm >> K.int32(1)) * K.int32(8) + jj,
                            row0 + K.int32(16 * rb) + (mm & K.int32(1)) * K.int32(8),
                        )
                        K.ptx["stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"](
                            ptr, frag[o], frag[o + 1], frag[o + 2], frag[o + 3]
                        )

            egA = K.alloc_local([16], "float32")
            egB = K.alloc_local([16], "float32")
            enA = K.alloc_local([16], "float32")
            enB = K.alloc_local([16], "float32")
            egcw = K.alloc_local([16], "uint32")
            t4 = K.alloc_local([4], "float32")
            acc = K.alloc_local([64], "float32")
            wds = K.alloc_local([32], "uint32")
            dgv = K.alloc_local([32], "float32")
            oq = K.alloc_local([8], "float32")
            ok8 = K.alloc_local([8], "float32")
            egn = K.local_scalar("float32")
            dgk = K.local_scalar("float32")
            dgk_k = K.local_scalar("float32")
            t0 = K.local_scalar("float32")
            t1 = K.local_scalar("float32")
            u16 = K.local_scalar("uint16")

            def rcp(dst, val):
                K.ptx.rcp.approx.ftz.f32(dst, val)

            def s_beta_row(c):
                b = K.local_scalar("float32")
                K.ptx.ld.shared.f32(b, K.address_of(s_beta[cyc & K.int32(1), c]))
                return b

            phase, phase_end = make_phaser()
            item = K.local_scalar("int32", init=cur - num_streams)
            with K.While(cur < total_work):
                K.assign(item, cur - num_streams)
                c, hq, seq, n, bos, rows, nch_i = item_coords(item)
                hq64 = K.Cast("int64", hq)
                tok0 = K.local_scalar("int64", init=bos + K.Cast("int64", n * K.int32(CHUNK)))
                last = K.local_scalar("int32", init=rows - K.int32(1))
                xq_base = K.local_scalar("int64", init=(tok0 + K.Cast("int64", row0)) * HQK64 + hq64 * K.int64(D) + x64)
                with K.serial(G) as gi:
                    hv = K.local_scalar("int32", init=hq * K.int32(G) + gi)
                    hv64 = K.Cast("int64", hv)
                    par = cyc & K.int32(1)
                    x_base = K.local_scalar("int64", init=(tok0 + K.Cast("int64", row0)) * HVK64 + hv64 * K.int64(D) + x64)

                    phase("w-in")
                    b_in_full.wait(0, par)
                    phase("w-xT")
                    twait("xT_done")
                    phase("c0")

                    xf = K.alloc_local([32], "float32")
                    egf = K.alloc_local([32], "float32")
                    qw = K.alloc_local([16], "uint32")
                    kw = K.alloc_local([16], "uint32")
                    t3w = K.alloc_local([16], "uint32")
                    qc = K.alloc_local([16], "uint32")
                    kc = K.alloc_local([16], "uint32")
                    vc = K.alloc_local([16], "uint32")
                    prep0 = K.local_scalar("uint64")
                    prep1 = K.local_scalar("uint64")
                    scale_pair = K.local_scalar("uint64", init=K.cuda.make_float2(scale, scale))
                    bpair = K.alloc_local([2], "float32")

                    K.ptx[TC_LD4](t4[0], t4[1], t4[2], t4[3], tmem_at(S1 + ((last >> 2) << 2)))
                    ld32(xf, S2 + wg * 32)
                    K.ptx[WAIT_LD]()
                    lq = last & K.int32(3)
                    K.assign(egn, K.Select(lq == K.int32(0), t4[0], K.Select(lq == K.int32(1), t4[1],
                                                                          K.Select(lq == K.int32(2), t4[2], t4[3]))))
                    for half in range(2):
                        vb32 = K.alloc_local([16], "float32")
                        for p in range(8):
                            i = 16 * half + 2 * p
                            K.ptx["ld.shared.v2.f32"](bpair[0], bpair[1], K.address_of(s_beta[par, row0 + i]))
                            K.assign(vb32[2 * p], xf[i] * bpair[0])
                            K.assign(vb32[2 * p + 1], xf[i + 1] * bpair[1])
                            pack_bf16x2(vc[i >> 1], xf[i], xf[i + 1])
                        K.ptx[TC_ST16](tmem_at(S2 + wg * 32 + 16 * half), *(vb32[j] for j in range(16)))
                    K.ptx[WAIT_ST]()
                    st_row(ST_V, row0, vc, 0, 4)
                    ld32(egf, S1 + wg * 32)
                    ld32(xf, S3 + wg * 32)
                    K.ptx[WAIT_LD]()


                    for i in range(32):
                        K.assign(egf[i], K.Select((row0 + K.int32(i) < rows) & (egf[i] != K.float32(0.0)),
                                                  egf[i], K.float32(1.0)))
                    for i in range(16):
                        m0 = K.Select(row0 + K.int32(2 * i) < rows, K.float32(1.0), K.float32(0.0))
                        m1 = K.Select(row0 + K.int32(2 * i + 1) < rows, K.float32(1.0), K.float32(0.0))
                        K.ptx["mul.rn.f32x2"](prep0, K.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                                              K.cuda.make_float2(egf[2 * i], egf[2 * i + 1]))
                        K.ptx["mul.rn.f32x2"](prep0, prep0, scale_pair)
                        K.ptx["mul.rn.f32x2"](prep0, prep0, K.cuda.make_float2(m0, m1))
                        pack_bf16x2(qw[i], K.cuda.float2_x(prep0), K.cuda.float2_y(prep0))
                        pack_bf16x2(qc[i], xf[2 * i], xf[2 * i + 1])
                        pack_bf16x2(egcw[i], egf[2 * i], egf[2 * i + 1])
                    for half in range(2):
                        K.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                            *(xf[16 * half + j] for j in range(16)), tmem_at(S4 + wg * 32 + 16 * half))
                        K.ptx[WAIT_LD]()
                        for pp in range(8):
                            i = 8 * half + pp
                            m0 = K.Select(row0 + K.int32(2 * i) < rows, K.float32(1.0), K.float32(0.0))
                            m1 = K.Select(row0 + K.int32(2 * i + 1) < rows, K.float32(1.0), K.float32(0.0))
                            K.ptx["ld.shared.v2.f32"](bpair[0], bpair[1], K.address_of(s_beta[par, row0 + 2 * i]))
                            rcp(t0, egf[2 * i])
                            rcp(t1, egf[2 * i + 1])
                            K.ptx["mul.rn.f32x2"](prep0, K.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                                                  K.cuda.make_float2(t0, t1))
                            K.ptx["mul.rn.f32x2"](prep0, prep0, K.cuda.make_float2(m0, m1))
                            pack_bf16x2(kw[i], K.cuda.float2_x(prep0), K.cuda.float2_y(prep0))
                            K.ptx["mul.rn.f32x2"](prep1, K.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                                                  K.cuda.make_float2(egf[2 * i], egf[2 * i + 1]))
                            K.ptx["mul.rn.f32x2"](prep1, prep1, K.cuda.make_float2(bpair[0], bpair[1]))
                            K.ptx["mul.rn.f32x2"](prep1, prep1, K.cuda.make_float2(m0, m1))
                            pack_bf16x2(t3w[i], K.cuda.float2_x(prep1), K.cuda.float2_y(prep1))
                            pack_bf16x2(kc[i], xf[2 * i], xf[2 * i + 1])

                    phase("w-chunk")
                    TCG["chunk_done"].wait(0, par ^ K.int32(1))
                    K.ptx[TC_FENCE_AFTER]()
                    K.ptx[FENCE_ASYNC]()
                    phase("c1")
                    st_row(T1, row0, qw, 0, 4)
                    st_row(T2, row0, kw, 0, 4)
                    st_row(T3, row0, t3w, 0, 4)
                    K.ptx[FENCE_ASYNC]()
                    marrive("t_early")

                    phase("w-h")
                    b_h_full.wait(0, par)
                    b_dhb_full.wait(0, par)
                    phase("c2")
                    dgk2 = K.local_scalar("uint64", init=K.cuda.make_float2(K.float32(0.0), K.float32(0.0)))
                    hst = K.local_scalar("int32", init=wg * 2 + xs)
                    for u in range(8):
                        hw = K.alloc_local([4], "uint32")
                        dw = K.alloc_local([4], "uint32")
                        K.ptx["ld.shared.v4.b32"](hw[0], hw[1], hw[2], hw[3], TT[S_H + hst].ptr_to(xr, 8 * u))
                        K.ptx["ld.shared.v4.b32"](dw[0], dw[1], dw[2], dw[3], TT[DHB + hst].ptr_to(xr, 8 * u))
                        for p in range(4):
                            K.ptx["fma.rn.f32x2"](dgk2, K.cuda.make_float2(lo(hw[p]), hi(hw[p])),
                                                  K.cuda.make_float2(lo(dw[p]), hi(dw[p])), dgk2)
                    K.assign(dgk, K.cuda.float2_x(dgk2) + K.cuda.float2_y(dgk2))

                    def readout_to_tile(slot, stage0):
                        ld32(acc, slot + wg * 32)
                        K.ptx[WAIT_LD]()
                        for p in range(16):
                            pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                        st_row(stage0, row0, wds, 0, 4)
                        K.ptx[FENCE_ASYNC]()

                    phase("w-Z")
                    twait("Z_done")
                    phase("c3")
                    readout_to_tile(S2, ZT)
                    marrive("zT_ready")
                    phase("w-dv2")
                    twait("dv2_done")
                    phase("c5")
                    readout_to_tile(S3, DV2)
                    marrive("dv2T_ready")
                    phase("w-Vn")
                    twait("Vn_done")
                    phase("c4")
                    readout_to_tile(S2, T6)
                    marrive("vnT_ready")

                    def readout64(slot, stage, mask, negate=False):
                        ld32(acc, slot + wg * 32)
                        K.ptx[WAIT_LD]()
                        cc = quad * 16 + lane
                        with K.If(lane < K.int32(16)), K.Then():
                            for p in range(16):
                                vv2 = []
                                for e in range(2):
                                    jj = row0 + 2 * p + e
                                    val = acc[2 * p + e]
                                    if negate:
                                        val = K.float32(0.0) - val
                                    vv2.append(K.Select(mask(cc, jj), val, K.float32(0.0)))
                                pack_bf16x2(wds[p], vv2[0], vv2[1])
                            for u in range(4):
                                K.ptx["st.shared.v4.b32"](TT[stage].ptr_to(cc, row0 + 8 * u),
                                                          wds[4 * u], wds[4 * u + 1], wds[4 * u + 2], wds[4 * u + 3])
                        K.ptx[FENCE_ASYNC]()

                    def readout64_half(slot, stage, mask, negate=False):
                        """Read one live 16-lane half of an M=64 accumulator.

                        `.16x256b.x4` maps each thread to two rows and eight
                        adjacent columns per row.  Pairwise bf16 conversion then
                        matches two non-transposed stmatrix.x4 stores exactly.
                        """
                        K.ptx[TC_LD_HALF32](*(acc[i] for i in range(16)), tmem_at(slot + wg * 32))
                        K.ptx[WAIT_LD]()
                        cc0 = quad * 16 + (lane >> K.int32(2))
                        cc1 = cc0 + K.int32(8)
                        for rep in range(4):
                            jj0 = row0 + K.int32(8 * rep) + (lane & K.int32(3)) * K.int32(2)
                            jj1 = jj0 + K.int32(1)
                            v00 = acc[4 * rep]
                            v01 = acc[4 * rep + 1]
                            v10 = acc[4 * rep + 2]
                            v11 = acc[4 * rep + 3]
                            if negate:
                                v00 = K.float32(0.0) - v00
                                v01 = K.float32(0.0) - v01
                                v10 = K.float32(0.0) - v10
                                v11 = K.float32(0.0) - v11
                            pack_bf16x2(
                                wds[2 * rep],
                                K.Select(mask(cc0, jj0), v00, K.float32(0.0)),
                                K.Select(mask(cc0, jj1), v01, K.float32(0.0)),
                            )
                            pack_bf16x2(
                                wds[2 * rep + 1],
                                K.Select(mask(cc1, jj0), v10, K.float32(0.0)),
                                K.Select(mask(cc1, jj1), v11, K.float32(0.0)),
                            )
                        tile = TT[stage]
                        for half in range(2):
                            K.ptx["stmatrix.sync.aligned.m8n8.x4.shared.b16"](
                                tile.m8n8x4(quad * K.int32(16), row0 + K.int32(16 * half), lane),
                                wds[4 * half], wds[4 * half + 1], wds[4 * half + 2], wds[4 * half + 3],
                            )
                        K.ptx[FENCE_ASYNC]()

                    phase("w-dAs")
                    twait("dAs_done")
                    phase("c8")
                    if HALF_DA_READOUT:
                        readout64_half(S4, DAM, lambda cc, jj: (jj < cc) & (cc < rows))
                    else:
                        readout64(S4, DAM, lambda cc, jj: (jj < cc) & (cc < rows))
                    marrive("dAm_ready")
                    phase("w-dAqk")
                    twait("dAqk_done")
                    phase("c7")
                    if HALF_DA_READOUT:
                        readout64_half(S4 + (16 << 16), T5, lambda cc, jj: (jj <= cc) & (cc < rows))
                    else:
                        readout64(S1, T5, lambda cc, jj: (jj <= cc) & (cc < rows))
                    marrive("dAqk_tile_ready")
                    phase("w-dk")
                    twait("dk_done")
                    twait("dvb_done")
                    phase("passA")

                    pbx = K.local_scalar("int32", init=K.int32(PB0) + wg * K.int32(PB1 - PB0))

                    def pass_a():
                        pa_acc = K.local_scalar("uint64")
                        pa_v = K.local_scalar("uint64")
                        pa_db = K.local_scalar("uint64")
                        pa_dv = K.local_scalar("uint64")
                        pa_word = K.local_scalar("uint32")
                        ld8(acc, S3 + wg * 32, 0)
                        K.ptx[WAIT_LD]()
                        for b in range(4):
                            ab = 8 * (b % 2)
                            if b < 3:
                                ld8(acc, S3 + wg * 32 + 8 * (b + 1), 8 * ((b + 1) % 2))
                            vq = K.alloc_local([4], "uint32")
                            K.ptx["ld.shared.v4.b32"](vq[0], vq[1], vq[2], vq[3],
                                                      TT[ST_V + xs].ptr_to(xr, row0 + 8 * b))
                            dbp = K.alloc_local([8], "float32")
                            for p in range(4):
                                i = 8 * b + 2 * p
                                K.assign(pa_acc, K.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]))
                                K.assign(pa_v, K.cuda.make_float2(lo(vq[p]), hi(vq[p])))
                                K.ptx["mul.rn.f32x2"](pa_db, pa_acc, pa_v)
                                K.assign(dbp[2 * p], K.cuda.float2_x(pa_db))
                                K.assign(dbp[2 * p + 1], K.cuda.float2_y(pa_db))
                                K.ptx["ld.shared.v2.f32"](
                                    t4[0], t4[1], K.address_of(s_beta[par, row0 + i])
                                )
                                K.ptx["mul.rn.f32x2"](
                                    pa_dv, pa_acc, K.cuda.make_float2(t4[0], t4[1])
                                )
                                pack_bf16x2(pa_word, K.cuda.float2_x(pa_dv), K.cuda.float2_y(pa_dv))
                                with K.If(row0 + K.int32(i) < rows), K.Then():
                                    K.ptx["st.global.L1::no_allocate.b16"](
                                        dv.ptr_to([x_base + K.int64(i * HVK)]), K.Cast("uint16", pa_word)
                                    )
                                with K.If(row0 + K.int32(i + 1) < rows), K.Then():
                                    K.ptx["st.global.L1::no_allocate.b16"](
                                        dv.ptr_to([x_base + K.int64((i + 1) * HVK)]),
                                        K.Cast("uint16", pa_word >> K.uint32(16)),
                                    )
                            for e in range(8):
                                i = 8 * b + e
                                K.ptx.st.shared.f32(TT[pbx + (i >> 4)].ptr_to(4 * (i & 15) + quad, 2 * lane), dbp[e])
                            dvw = K.alloc_local([4], "uint32")
                            for p in range(4):
                                pack_bf16x2(dvw[p], acc[ab + 2 * p], acc[ab + 2 * p + 1])
                            K.ptx["st.shared.v4.b32"](TT[DVB + xs].ptr_to(xr, row0 + 8 * b),
                                                      dvw[0], dvw[1], dvw[2], dvw[3])
                            if b < 3:
                                K.ptx[WAIT_LD]()

                    pass_a()
                    K.ptx[FENCE_ASYNC]()
                    marrive("dv_epi_done")
                    phase("w-X")
                    twait("X_done")
                    phase("c9")
                    if HALF_XY_READOUT:
                        readout64_half(S1, T6, lambda cc, jj: cc < rows)
                    else:
                        readout64(S1, T6, lambda cc, jj: cc < rows)
                    marrive("X_ready")
                    phase("dbv")
                    bar_wg()
                    tq = lane & K.int32(3)
                    ti = quad * 8 + (lane >> 2)
                    srow = (quad & K.int32(1)) * 32 + lane
                    dsum_v2 = K.local_scalar(
                        "uint64", init=K.cuda.make_float2(K.float32(0.0), K.float32(0.0))
                    )
                    dsum2 = K.local_scalar("uint64")
                    sum_pair0 = K.local_scalar("uint64")
                    sum_pair1 = K.local_scalar("uint64")
                    for u in range(8):
                        K.ptx["ld.shared.v4.f32"](t4[0], t4[1], t4[2], t4[3], TT[pbx + (quad >> 1)].ptr_to(srow, 8 * u))
                        K.assign(sum_pair0, K.cuda.make_float2(t4[0], t4[1]))
                        K.assign(sum_pair1, K.cuda.make_float2(t4[2], t4[3]))
                        K.ptx["add.rn.f32x2"](sum_pair0, sum_pair0, sum_pair1)
                        K.ptx["add.rn.f32x2"](dsum_v2, dsum_v2, sum_pair0)
                    K.ptx[FENCE_ASYNC]()
                    b_mid_free.arrive(0)
                    phase("w-Y")
                    twait("Y_done")
                    phase("c10")
                    if HALF_XY_READOUT:
                        readout64_half(S2, T5 + 1, lambda cc, jj: (jj < cc) & (cc < rows), negate=True)
                    else:
                        readout64(S2, T5 + 1, lambda cc, jj: (jj < cc) & (cc < rows), negate=True)
                    marrive("intra_ready")
                    phase("w-epi")
                    twait("dq2_done")
                    phase("epi-q")
                    dgk_k2 = K.local_scalar(
                        "uint64", init=K.cuda.make_float2(K.float32(0.0), K.float32(0.0))
                    )

                    def q_loads(b, base):
                        ld8(acc, S4 + wg * 32 + 8 * b, base)

                    def k_loads(b, base):
                        ld4(acc, S5 + wg * 32 + 4 * b, base + 0)
                        ld4(acc, S6 + wg * 32 + 4 * b, base + 4)
                        ld4(acc, S3 + wg * 32 + 4 * b, base + 8)

                    def emit_group_output(b, vals, tm_col, out, obase):
                        """dq/dk for 8 tokens of this channel: accumulate over the head group in TMEM."""
                        if G > 1:
                            with K.If(gi > K.int32(0)), K.Then():
                                ld8(oq, tm_col + wg * 32 + 8 * b)
                                K.ptx[WAIT_LD]()
                                for e in range(8):
                                    K.assign(vals[e], vals[e] + oq[e])
                        with K.If(gi == K.int32(G - 1)):
                            with K.Then():
                                for e in range(8):
                                    i = 8 * b + e
                                    with K.If(row0 + K.int32(i) < rows), K.Then():
                                        K.ptx["st.global.L1::no_allocate.f32"](out.ptr_to([obase + K.int64(i * HQK)]), vals[e])
                            if G > 1:
                                with K.Else():
                                    st8(tm_col + wg * 32 + 8 * b, vals)
                                    K.ptx[WAIT_ST]()

                    def epilogue():
                        pair0 = K.local_scalar("uint64")
                        pair1 = K.local_scalar("uint64")
                        pair2 = K.local_scalar("uint64")
                        pair3 = K.local_scalar("uint64")
                        pair4 = K.local_scalar("uint64")
                        pair5 = K.local_scalar("uint64")
                        q_loads(0, 0)
                        K.ptx[WAIT_LD]()
                        for b in range(4):
                            ab = 8 * (b % 2)
                            if b < 3:
                                q_loads(b + 1, 8 * ((b + 1) % 2))
                            for p in range(4):
                                i = 8 * b + 2 * p
                                rcp(enA[i >> 1], lo(egcw[i >> 1]))
                                rcp(enB[i >> 1], hi(egcw[i >> 1]))
                                K.ptx["mul.rn.f32x2"](
                                    pair1,
                                    K.cuda.make_float2(lo(egcw[i >> 1]), hi(egcw[i >> 1])),
                                    K.cuda.make_float2(scale, scale),
                                )
                                K.ptx["mul.rn.f32x2"](
                                    pair0,
                                    K.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    pair1,
                                )
                                K.assign(ok8[2 * p], K.cuda.float2_x(pair0))
                                K.assign(ok8[2 * p + 1], K.cuda.float2_y(pair0))
                                K.ptx["mul.rn.f32x2"](pair1, K.cuda.make_float2(lo(qc[i >> 1]), hi(qc[i >> 1])), pair0)
                                K.assign(dgv[i], K.cuda.float2_x(pair1))
                                K.assign(dgv[i + 1], K.cuda.float2_y(pair1))
                            emit_group_output(b, ok8, TM_ADQ, dq, xq_base)
                            if b < 3:
                                K.ptx[WAIT_LD]()
                        phase("w-dkt")
                        twait("dkt_done")
                        phase("epi-k")
                        dbx = 2 * wg
                        k_loads(0, 0)
                        K.ptx[WAIT_LD]()
                        for b in range(8):
                            ab = 12 * (b % 2)
                            if b < 7:
                                k_loads(b + 1, 12 * ((b + 1) % 2))
                            for p in range(2):
                                i = 4 * b + 2 * p
                                K.assign(pair0, K.cuda.make_float2(enA[i >> 1], enB[i >> 1]))
                                K.assign(pair1, K.cuda.make_float2(lo(kc[i >> 1]), hi(kc[i >> 1])))
                                K.ptx["add.rn.f32x2"](
                                    pair2,
                                    K.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    K.cuda.make_float2(acc[ab + 8 + 2 * p], acc[ab + 8 + 2 * p + 1]),
                                )
                                K.ptx["mul.rn.f32x2"](pair2, pair2, pair0)
                                K.ptx["mul.rn.f32x2"](
                                    pair3,
                                    K.cuda.make_float2(acc[ab + 4 + 2 * p], acc[ab + 4 + 2 * p + 1]),
                                    K.cuda.make_float2(lo(egcw[i >> 1]), hi(egcw[i >> 1])),
                                )
                                K.ptx["mul.rn.f32x2"](pair4, pair1, pair3)
                                K.ptx.st.shared.f32(TT[dbx + (i >> 4)].ptr_to(4 * (i & 15) + quad, 2 * lane), K.cuda.float2_x(pair4))
                                K.ptx.st.shared.f32(TT[dbx + ((i + 1) >> 4)].ptr_to(4 * ((i + 1) & 15) + quad, 2 * lane), K.cuda.float2_y(pair4))
                                K.ptx["mul.rn.f32x2"](pair4, K.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]), pair0)
                                K.ptx["mul.rn.f32x2"](pair5, pair1, pair4)
                                K.ptx["add.rn.f32x2"](dgk_k2, dgk_k2, pair5)
                                K.ptx["mul.rn.f32x2"](pair3, pair3, K.cuda.make_float2(s_beta_row(row0 + i), s_beta_row(row0 + i + 1)))
                                K.ptx["add.rn.f32x2"](pair5, pair2, pair3)
                                K.assign(ok8[4 * (b % 2) + 2 * p], K.cuda.float2_x(pair5))
                                K.assign(ok8[4 * (b % 2) + 2 * p + 1], K.cuda.float2_y(pair5))
                                K.ptx["sub.rn.f32x2"](pair3, pair3, pair2)
                                K.ptx["fma.rn.f32x2"](pair5, pair1, pair3, K.cuda.make_float2(dgv[i], dgv[i + 1]))
                                K.assign(dgv[i], K.cuda.float2_x(pair5))
                                K.assign(dgv[i + 1], K.cuda.float2_y(pair5))
                            if b % 2 == 1:
                                emit_group_output(b // 2, ok8, TM_ADK, dk, xq_base)
                            if b < 7:
                                K.ptx[WAIT_LD]()
                        bar_wg()
                        K.assign(dsum2, dsum_v2)
                        for u in range(8):
                            K.ptx["ld.shared.v4.f32"](t4[0], t4[1], t4[2], t4[3], TT[dbx + (quad >> 1)].ptr_to(srow, 8 * u))
                            K.assign(sum_pair0, K.cuda.make_float2(t4[0], t4[1]))
                            K.assign(sum_pair1, K.cuda.make_float2(t4[2], t4[3]))
                            K.ptx["add.rn.f32x2"](sum_pair0, sum_pair0, sum_pair1)
                            K.ptx["add.rn.f32x2"](dsum2, dsum2, sum_pair0)
                        dsum = K.local_scalar(
                            "float32", init=K.cuda.float2_x(dsum2) + K.cuda.float2_y(dsum2)
                        )
                        K.ptx[FENCE_ASYNC]()
                        b_h_free.arrive(0)
                        for s in (1, 2):
                            r = K.local_scalar("uint32")
                            K.ptx.shfl_sync.bfly.b32(r, K.reinterpret("uint32", dsum), K.uint32(s), K.uint32(0x1F), K.uint32(0xFFFFFFFF))
                            K.assign(dsum, dsum + K.reinterpret("float32", r))
                        with K.If((tq == K.int32(0)) & (row0 + ti < rows)), K.Then():
                            K.ptx["st.global.L1::no_allocate.f32"](
                                db.ptr_to([(tok0 + K.Cast("int64", row0 + ti)) * K.int64(HV) + hv64]), dsum)

                    epilogue()
                    K.assign(dgk_k, K.cuda.float2_x(dgk_k2) + K.cuda.float2_y(dgk_k2))
                    phase("cumsum")
                    for i in range(32):
                        K.assign(dgv[i], K.Select(row0 + K.int32(i) < rows, dgv[i], K.float32(0.0)))


                    tot = K.alloc_local([16], "float32")
                    for i in range(16):
                        K.assign(tot[i], dgv[2 * i] + dgv[2 * i + 1])
                    for w in (8, 4, 2, 1):
                        for i in range(w):
                            K.assign(tot[i], tot[i] + tot[i + w])
                    K.ptx.st.shared.f32(K.address_of(s_dgk[wg, x]),
                                        dgk + dgk_k + K.Select(wg == K.int32(0), K.float32(0.0), tot[0]))
                    b_dg0_ready.arrive(0)
                    for i in range(30, -1, -1):
                        K.assign(dgv[i], dgv[i] + dgv[i + 1])
                    b_dg0_ready.wait(0, cyc & K.int32(1))
                    K.ptx.ld.shared.f32(t0, K.address_of(s_dgk[K.int32(1) - wg, x]))
                    K.assign(t1, t0 + dgk + dgk_k)
                    for i in range(32):
                        with K.If(row0 + K.int32(i) < rows), K.Then():
                            K.ptx["st.global.L1::no_allocate.f32"](dg.ptr_to([x_base + K.int64(i * HVK)]), dgv[i] + t1)
                    phase_end()
                    K.assign(cyc, cyc + K.int32(1))
                K.assign(kk_, kk_ + K.int32(1))
                K.assign(cur, work_wait(kk_))

        with auxg:
            with mma:
                tm = tmem_preamble()
                bd1 = K.alloc_local([1], "uint64")
                zq1 = K.alloc_local([1], "int32")

                op_kbg_k = Op(bd1, F_KBG, 128, 64, "k")
                op_vb_k = Op(bd1, F_VB, 128, 64, "k")
                op_kg_k = Op(bd1, F_KG, 128, 64, "k")
                op_akk1_k = Op(bd1, F_AKK, 64, 64, "k")
                op_hs_mn = Op(bd1, F_HS, 128, 128, "mn")
                op_w_mn = Op(bd1, F_KBG, 128, 128, "mn")
                op_kraw = Op(bd1, F_KV, 128, 64, "mn")
                op_vraw = Op(bd1, F_KV + 1, 128, 64, "mn")
                ID_T1 = idesc(128, 16, ta=1)
                bdI1 = K.alloc_local([1], "uint64")
                K.cuda.tcgen05.encode_matrix_descriptor(
                    K.address_of(bdI1[0]), s_ident.ptr_to([0]), ldo=8, sdo=16, swizzle=0)
                st_kv1 = K.PipelineState(1, phase=0)
                p1m = K.local_scalar("int32", init=K.int32(0))
                SET_UNITS = 2 * UNITS_PER_STAGE

                op_bakk_k = Op(bd1, B_AKK, 64, 64, "k")
                op_bdo_mn = Op(bd1, B_DO, 64, 64, "mn")
                DO2_UNITS = 2 * UNITS_PER_STAGE
                op_baqk_mn = Op(bd1, B_AQK, 64, 64, "mn")
                op_dhb_mn = Op(bd1, B_DHB, 128, 128, "mn")
                op_t2_mn = Op(bd1, B_T2, 128, 128, "mn")
                op_dv2_k = Op(bd1, B_DV2, 128, 64, "k")
                op_qraw = Op(bd1, B_QK, 128, 64, "mn")
                op_kraw_b = Op(bd1, B_QK + 1, 128, 64, "mn")
                st_qk1 = K.PipelineState(1, phase=0)
                bqm = K.local_scalar("int32", init=K.int32(0))
                ID_M128N64 = idesc(128, 64)
                ID_VN = idesc(128, 64, ta=1, tb=1, nb=1)
                ID_HUPD = idesc(128, 128)
                ID_128x64_TATB = idesc(128, 64, ta=1, tb=1)
                ID_128x128_TB = idesc(128, 128, tb=1)
                ID_128x128_NB = idesc(128, 128, nb=1)
                st_tiles = K.PipelineState(2, phase=0)
                st_akk = K.PipelineState(2, phase=0)
                st_hs = K.PipelineState(1, phase=0)
                st_w = K.PipelineState(2, phase=0)
                st_vn = K.PipelineState(1, phase=0)
                bcyc = K.local_scalar("int32", init=K.int32(0))
                mphase, mphase_end = make_phaser()

                def encode_base():
                    K.ptx.ld.volatile.shared.s32(zq1[0], K.address_of(s_tmem[1]))
                    K.cuda.tcgen05.encode_matrix_descriptor(
                        K.address_of(bd1[0]), TT[zq1[0]].ptr_to(0, 0), ldo=Op.LBO_BASE, sdo=SBO_UNITS,
                        swizzle=K.SW128B.value)

                def kv_transpose():
                    """Raw K and V chunk tiles -> channel-major fp32 K^T / V^T in TMEM (eight identity MMAs)."""
                    mphase("fmw-kv")
                    p_kv.full.wait(0, st_kv1.phase)
                    b_kv_read.wait(0, (p1m & K.int32(1)) ^ K.int32(1))
                    K.ptx[TC_FENCE_AFTER]()
                    mphase("fm-kvT")
                    with K.If(elected()), K.Then():
                        for src, dst in ((op_kraw, TM_KT), (op_vraw, TM_VT)):
                            for j in range(4):
                                K.ptx[MMA_SS](
                                    K.Cast("uint32", tm[0] + dst + 16 * j),
                                    src.desc(j),
                                    bdI1[0],
                                    K.uint32(ID_T1),
                                    K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0),
                                    K.ptx.pred(0),
                                )
                        b_kvT_done.arrive(0)
                        p_kv.empty.arrive(0)
                    st_kv1.advance()
                    K.assign(p1m, p1m + K.int32(1))

                def qk_transpose():
                    """Raw q and k chunk tiles -> channel-major fp32 q^T / k^T in TMEM (eight identity MMAs)."""
                    mphase("bmw-qk")
                    p_qk.full.wait(0, st_qk1.phase)
                    b_qk_read.wait(0, (bqm & K.int32(1)) ^ K.int32(1))
                    K.ptx[TC_FENCE_AFTER]()
                    mphase("bm-qkT")
                    with K.If(elected()), K.Then():
                        for src, dst in ((op_qraw, TM_QT), (op_kraw_b, TM_KTB)):
                            for j in range(4):
                                K.ptx[MMA_SS](
                                    K.Cast("uint32", tm[0] + dst + 16 * j),
                                    src.desc(j),
                                    bdI1[0],
                                    K.uint32(ID_T1),
                                    K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0),
                                    K.ptx.pred(0),
                                )
                        b_qkT_done.arrive(0)
                        p_qk.empty.arrive(0)
                    st_qk1.advance()
                    K.assign(bqm, bqm + K.int32(1))

                with K.While(cur < num_streams):
                    is_fwd, seq, hv, hq, bos, seq_len, nch = stream_coords(cur)
                    with K.If(is_fwd == K.int32(1)):
                        with K.Then():
                            encode_base()
                            kv_transpose()
                            with K.If(nch > K.int32(1)), K.Then():
                                kv_transpose()
                            with K.serial(nch) as n:
                                set_u = K.local_scalar("uint64", init=K.Cast("uint64", n & K.int32(1)) * K.uint64(SET_UNITS))
                                akk_u = K.local_scalar("uint64", init=K.Cast("uint64", st_akk.stage) * K.uint64(UNITS_PER_STAGE))
                                dW = (n & K.int32(1)) * 64
                                mphase("fmw-tiles")
                                p_tiles.full.wait(st_tiles.stage, st_tiles.phase)
                                p_akk1.full.wait(st_akk.stage, st_akk.phase)
                                K.ptx[TC_FENCE_AFTER]()
                                mphase("fm-WU")
                                with K.If(elected()), K.Then():
                                    mma_chain(tm, TM_W0 + dW, op_kbg_k, op_akk1_k, ID_M128N64, False, a_units=set_u, b_units=akk_u)
                                    p_w.full.arrive(st_w.stage)
                                    mma_chain(tm, TM_U0 + dW, op_vb_k, op_akk1_k, ID_M128N64, False, a_units=set_u, b_units=akk_u)
                                    p_akk1.empty.arrive(st_akk.stage)
                                st_akk.advance()
                                mphase("fmw-wT")
                                p_w.empty.wait(st_w.stage, st_w.phase)
                                st_w.advance()
                                mphase("fmw-hs")
                                p_hs.full.wait(st_hs.stage, st_hs.phase)
                                K.ptx[TC_FENCE_AFTER]()
                                mphase("fm-Vn")
                                with K.If(elected()), K.Then():
                                    mma_chain(tm, TM_U0 + dW, op_hs_mn, op_w_mn, ID_VN, True, b_units=set_u)
                                    p_vn.full.arrive(0)
                                st_hs.advance()
                                mphase("fmw-vnT")
                                p_vn.empty.wait(0, st_vn.phase)
                                st_vn.advance()
                                K.ptx[TC_FENCE_AFTER]()
                                mphase("fm-hupd")
                                with K.If(elected()), K.Then():
                                    mma_chain(tm, TM_H, op_kg_k, op_vb_k, ID_HUPD, True, a_units=set_u, b_units=set_u)
                                    p_tiles.empty.arrive(st_tiles.stage)
                                st_tiles.advance()
                                with K.If(n + K.int32(2) < nch), K.Then():
                                    kv_transpose()
                                mphase_end()
                        with K.Else():
                            encode_base()
                            qk_transpose()
                            with K.serial(nch) as rn:
                                par = K.local_scalar("int32", init=bcyc & K.int32(1))
                                t1_col = TM_T1 + par * 32
                                kb_col = TM_KB + par * 32
                                do_u = K.local_scalar("uint64", init=K.Cast("uint64", par) * K.uint64(DO2_UNITS))
                                mphase("bmw-prep")
                                MB["prep_ready"].wait(0, par)
                                b_bakk_full.wait(0, par)
                                K.ptx[TC_FENCE_AFTER]()
                                mphase("bm-W")
                                with K.If(elected()), K.Then():
                                    mma_chain_ta(tm, TM_BW, kb_col, op_bakk_k, ID_M128N64, False)
                                    TC["W_done"].arrive(0)
                                    b_bakk_empty.arrive(0)
                                with K.If(rn + K.int32(1) < nch), K.Then():
                                    qk_transpose()
                                mphase("bmw-dhb")
                                MB["dhb_ready"].wait(0, par)
                                b_baqk_masked.wait(0, par)
                                b_bdo_full.wait(par, (bcyc >> 1) & K.int32(1))
                                K.ptx[TC_FENCE_AFTER]()
                                mphase("bm-dv2")
                                with K.If(elected()), K.Then():
                                    mma_chain(tm, TM_DV2, op_bdo_mn, op_baqk_mn, ID_128x64_TATB, False, a_units=do_u)
                                    b_baqk_empty.arrive(0)
                                    mma_chain(tm, TM_DV2, op_dhb_mn, op_t2_mn, ID_128x64_TATB, True)
                                    TC["dv2_done"].arrive(0)
                                mphase("bmw-rd")
                                MB["wT_ready"].wait(0, par)
                                MB["dv2T_ready"].wait(0, par)
                                K.ptx[TC_FENCE_AFTER]()
                                mphase("bm-dh")
                                with K.If(elected()), K.Then():
                                    mma_chain_ta(tm, TM_DH, t1_col, op_bdo_mn, ID_128x128_TB, True, b_units=do_u)
                                    mma_chain_ta(tm, TM_DH, kb_col, op_dv2_k, ID_128x128_NB, True)
                                    TC["dh_done"].arrive(0)
                                mphase_end()
                                K.assign(bcyc, bcyc + K.int32(1))
                    K.ptx.bar.sync(K.uint32(5), K.uint32(384))
                    K.assign(kk_, kk_ + K.int32(1))
                    K.assign(cur, work_wait(kk_))

            with loader:
                with K.If(elected()), K.Then():
                    for m in (q_map, k_map, v_map, g_map, do_map, aqk_map, akk_map, h_map, dh_map):
                        K.ptx.prefetch.tensormap(K.address_of(m))
                st_kv = K.PipelineState(1, phase=1)
                st_akk = K.PipelineState(2, phase=1)
                st_g = K.PipelineState(1, phase=1)
                st_qk = K.PipelineState(1, phase=1)
                st_bg = K.PipelineState(1, phase=1)
                bcyc = K.local_scalar("int32", init=K.int32(0))
                lphase, lphase_end = make_phaser()
                with K.While(cur < num_streams):
                    is_fwd, seq, hv, hq, bos, seq_len, nch = stream_coords(cur)
                    bos32 = K.local_scalar("int32", init=K.Cast("int32", bos))
                    with K.If(is_fwd == K.int32(1)):
                        with K.Then():
                            with K.serial(nch) as n:
                                tok0 = bos32 + n * K.int32(CHUNK)
                                rows = K.local_scalar("int32", init=chunk_rows(seq_len, n))
                                p_kv.empty.wait(0, st_kv.phase)
                                with K.If(elected()), K.Then():
                                    p_kv.full.arrive(0, tx_count=KV_BYTES)
                                    mb = K.cuda.cvta_generic_to_shared(p_kv.full.ptr_to([0]))
                                    for d0 in (0, 64):
                                        K.ptx[TMA_LD](TT[F_KV + K.int32((d0 // 64) * 2)].ptr_to(0, 0),
                                                      K.address_of(k_map), K.int32(d0), tok0, hq, mb)
                                        K.ptx[TMA_LD](TT[F_KV + K.int32((d0 // 64) * 2 + 1)].ptr_to(0, 0),
                                                      K.address_of(v_map), K.int32(d0), tok0, hv, mb)
                                    with K.If(n + K.int32(1) < nch), K.Then():
                                        for d0 in (0, 64):
                                            K.ptx[TMA_PREFETCH](K.address_of(k_map), K.int32(d0), tok0 + K.int32(CHUNK), hq)
                                            K.ptx[TMA_PREFETCH](K.address_of(v_map), K.int32(d0), tok0 + K.int32(CHUNK), hv)
                                        for d0 in (0, 32, 64, 96):
                                            K.ptx[TMA_PREFETCH](K.address_of(g_map), K.int32(d0), tok0 + K.int32(CHUNK), hv)
                                        K.ptx[TMA_PREFETCH](K.address_of(akk_map), K.int32(0), tok0 + K.int32(CHUNK), hv)
                                st_kv.advance()
                                p_g.empty.wait(0, st_g.phase)
                                with K.If(elected()), K.Then():
                                    p_g.full.arrive(0, tx_count=G_BYTES)
                                    mbg = K.cuda.cvta_generic_to_shared(p_g.full.ptr_to([0]))
                                    for j in range(4):
                                        K.ptx[TMA_LD](TT[F_G + K.int32(j)].ptr_to(0, 0),
                                                      K.address_of(g_map), K.int32(32 * j), tok0, hv, mbg)


                                bslot = K.local_scalar("int32", init=n & K.int32(1))
                                load_beta_lanes(lambda t: K.address_of(s_beta1[bslot, t]), bos, hv, n, rows)
                                p_g.full.arrive(0)
                                st_g.advance()
                                p_akk1.empty.wait(st_akk.stage, st_akk.phase)
                                with K.If(elected()), K.Then():
                                    p_akk1.full.arrive(st_akk.stage, tx_count=A_BYTES)
                                    mb2 = K.cuda.cvta_generic_to_shared(p_akk1.full.ptr_to([st_akk.stage]))
                                    K.ptx[TMA_LD](TT[F_AKK + st_akk.stage].ptr_to(0, 0), K.address_of(akk_map), K.int32(0), tok0, hv, mb2)
                                st_akk.advance()
                        with K.Else():
                            with K.serial(nch) as rn:
                                n = nch - K.int32(1) - rn
                                par = K.local_scalar("int32", init=bcyc & K.int32(1))
                                tok0 = bos32 + n * K.int32(CHUNK)
                                rows = K.local_scalar("int32", init=chunk_rows(seq_len, n))

                                lphase("blw-qk")
                                p_qk.empty.wait(0, st_qk.phase)
                                lphase("bl-qk")
                                with K.If(elected()), K.Then():
                                    p_qk.full.arrive(0, tx_count=KV_BYTES)
                                    mb = K.cuda.cvta_generic_to_shared(p_qk.full.ptr_to([0]))
                                    for d0 in (0, 64):
                                        K.ptx[TMA_LD](TT[B_QK + K.int32((d0 // 64) * 2)].ptr_to(0, 0), K.address_of(q_map), K.int32(d0), tok0, hq, mb)
                                        K.ptx[TMA_LD](TT[B_QK + K.int32((d0 // 64) * 2 + 1)].ptr_to(0, 0), K.address_of(k_map), K.int32(d0), tok0, hq, mb)
                                    with K.If(n > K.int32(0)), K.Then():
                                        tokp = tok0 - K.int32(CHUNK)
                                        for d0 in (0, 64):
                                            K.ptx[TMA_PREFETCH](K.address_of(q_map), K.int32(d0), tokp, hq)
                                            K.ptx[TMA_PREFETCH](K.address_of(k_map), K.int32(d0), tokp, hq)
                                            K.ptx[TMA_PREFETCH](K.address_of(do_map), K.int32(d0), tokp, hv)
                                        for d0 in (0, 32, 64, 96):
                                            K.ptx[TMA_PREFETCH](K.address_of(g_map), K.int32(d0), tokp, hv)
                                        K.ptx[TMA_PREFETCH](K.address_of(aqk_map), K.int32(0), tokp, hv)
                                        K.ptx[TMA_PREFETCH](K.address_of(akk_map), K.int32(0), tokp, hv)
                                st_qk.advance()

                                lphase("blw-g")
                                b_g_free.wait(0, st_bg.phase)
                                lphase("bl-g")
                                with K.If(elected()), K.Then():
                                    b_g_full.arrive(0, tx_count=G_BYTES)
                                    mbg = K.cuda.cvta_generic_to_shared(b_g_full.ptr_to([0]))
                                    for j in range(4):
                                        K.ptx[TMA_LD](TT[B_G + j].ptr_to(0, 0), K.address_of(g_map), K.int32(32 * j), tok0, hv, mbg)
                                load_beta_lanes(lambda t: K.address_of(s_bbeta[par, t]), bos, hv, n, rows)
                                b_g_full.arrive(0)
                                st_bg.advance()

                                lphase("blw-do")
                                with K.If(rn > K.int32(1)), K.Then():
                                    TC["dh_done"].wait(0, par)
                                lphase("bl-do")
                                with K.If(elected()), K.Then():
                                    b_bdo_full.arrive(par, tx_count=DO_BYTES)
                                    mbd = K.cuda.cvta_generic_to_shared(b_bdo_full.ptr_to([par]))
                                    for d0 in (0, 64):
                                        K.ptx[TMA_LD](TT[B_DO + par * K.int32(2) + K.int32(d0 // 64)].ptr_to(0, 0), K.address_of(do_map), K.int32(d0), tok0, hv, mbd)
                                lphase("blw-a")
                                with K.If(rn > K.int32(0)), K.Then():
                                    b_baqk_empty.wait(0, par ^ K.int32(1))
                                with K.If(elected()), K.Then():
                                    b_baqk_full.arrive(0, tx_count=A_BYTES)
                                    mb = K.cuda.cvta_generic_to_shared(b_baqk_full.ptr_to([0]))
                                    K.ptx[TMA_LD](TT[B_AQK].ptr_to(0, 0), K.address_of(aqk_map), K.int32(0), tok0, hv, mb)
                                with K.If(rn > K.int32(0)), K.Then():
                                    b_bakk_empty.wait(0, par ^ K.int32(1))
                                with K.If(elected()), K.Then():
                                    b_bakk_full.arrive(0, tx_count=A_BYTES)
                                    mb = K.cuda.cvta_generic_to_shared(b_bakk_full.ptr_to([0]))
                                    K.ptx[TMA_LD](TT[B_AKK].ptr_to(0, 0), K.address_of(akk_map), K.int32(0), tok0, hv, mb)
                                lphase_end()
                                K.assign(bcyc, bcyc + K.int32(1))
                    claim_publish(kk_ + K.int32(1))
                    K.ptx.bar.sync(K.uint32(5), K.uint32(384))
                    K.assign(kk_, kk_ + K.int32(1))
                    K.assign(cur, work_wait(kk_))

            with w10:
                st_hs = K.PipelineState(1, phase=0)
                bcyc = K.local_scalar("int32", init=K.int32(0))
                with K.While(cur < num_streams):
                    is_fwd, seq, hv, hq, bos, seq_len, nch = stream_coords(cur)
                    cb = chunk_base(seq)

                    fidx_s = K.local_scalar("int32", init=seq * K.int32(HV) + hv)
                    with K.If(is_fwd == K.int32(1)):
                        with K.Then():
                            with K.serial(nch) as n:
                                p_hs.full.wait(st_hs.stage, st_hs.phase)
                                with K.If(elected()), K.Then():
                                    K.ptx[FENCE_ASYNC]()
                                    idx = (cb + n) * K.int32(HV) + hv
                                    for d0 in (0, 64):
                                        K.ptx[TMA_ST](K.address_of(h_map), K.int32(d0), K.int32(0), idx,
                                                      TT[F_HS + st_hs.stage * K.int32(4) + K.int32((d0 // 64) * 2)].ptr_to(0, 0))
                                    K.ptx[BULK_COMMIT]()
                                    K.ptx[BULK_WAIT_READ](0)
                                    p_hs.empty.arrive(st_hs.stage)


                                    K.ptx[BULK_WAIT](0)
                                    K.ptx["fence.proxy.async.global"]()
                                    K.ptx["st.release.gpu.global.s64"](
                                        flags.ptr_to([fidx_s]), ep64 + K.Cast("int64", n + K.int32(1)))
                                st_hs.advance()
                        with K.Else():
                            with K.serial(nch) as rn:
                                n = nch - K.int32(1) - rn
                                par = bcyc & K.int32(1)
                                MB["dhb_ready"].wait(0, par)
                                with K.If(elected()), K.Then():
                                    K.ptx[FENCE_ASYNC]()
                                    idx = (cb + n) * K.int32(HV) + hv
                                    for d0 in (0, 64):
                                        K.ptx[TMA_ST](K.address_of(dh_map), K.int32(d0), K.int32(0), idx,
                                                      TT[B_DHB + K.int32((d0 // 64) * 2)].ptr_to(0, 0))
                                    K.ptx[BULK_COMMIT]()
                                    K.ptx[BULK_WAIT_READ](0)
                                    b_dhb_stored.arrive(0)
                                    K.ptx[BULK_WAIT](0)
                                    K.ptx["fence.proxy.async.global"]()
                                    K.ptx["st.release.gpu.global.s64"](
                                        flags.ptr_to([num_chains + fidx_s]), ep64 + K.Cast("int64", rn + K.int32(1)))
                                K.assign(bcyc, bcyc + K.int32(1))
                    with K.If(elected()), K.Then():
                        K.ptx[BULK_WAIT](0)
                        K.ptx["fence.proxy.async.global"]()

                    K.ptx.bar.sync(K.uint32(5), K.uint32(384))
                    K.assign(kk_, kk_ + K.int32(1))
                    K.assign(cur, work_wait(kk_))

            with w11:
                bcyc = K.local_scalar("int32", init=K.int32(0))
                lane = K.lane_id()
                with K.While(cur < num_streams):
                    is_fwd, seq, hv, hq, bos, seq_len, nch = stream_coords(cur)
                    with K.If(is_fwd == K.int32(0)), K.Then():
                        with K.serial(nch) as rn:
                            n = nch - K.int32(1) - rn
                            par = bcyc & K.int32(1)
                            rows = K.local_scalar("int32", init=chunk_rows(seq_len, n))
                            b_baqk_full.wait(0, par)

                            for half in range(2):
                                diag = K.alloc_local([4], "uint32")
                                dmat = lane >> K.int32(3)
                                dblk = K.int32(4 * half) + dmat
                                dptr = TT[B_AQK].ptr_to(dblk * K.int32(8) + (lane & K.int32(7)), dblk * K.int32(8))
                                K.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](diag[0], diag[1], diag[2], diag[3], dptr)
                                drow = lane >> K.int32(2)
                                dcol = (lane & K.int32(3)) * K.int32(2)
                                dmask = K.Select(dcol > drow, K.uint32(0),
                                                 K.Select(dcol == drow, K.uint32(0x0000FFFF), K.uint32(0xFFFFFFFF)))
                                for e in range(4):
                                    blk_row = K.int32(8 * (4 * half + e)) + drow
                                    K.assign(diag[e], K.Select(blk_row < rows, diag[e] & dmask, K.uint32(0)))
                                K.ptx["stmatrix.sync.aligned.m8n8.x4.shared.b16"](dptr, diag[0], diag[1], diag[2], diag[3])
                            for r in range(2):
                                rowc = lane + K.int32(32 * r)
                                for u in range(1, 8):
                                    with K.If(K.int32(8 * u) > rowc), K.Then():
                                        K.ptx["st.shared.v4.b32"](TT[B_AQK].ptr_to(rowc, 8 * u),
                                                                  K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0))
                                with K.If(rowc >= rows), K.Then():
                                    for u in range(0, 8):
                                        K.ptx["st.shared.v4.b32"](TT[B_AQK].ptr_to(rowc, 8 * u),
                                                                  K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0))
                            K.ptx[FENCE_ASYNC]()
                            b_baqk_masked.arrive(0)
                            K.assign(bcyc, bcyc + K.int32(1))
                    K.ptx.bar.sync(K.uint32(5), K.uint32(384))
                    K.assign(kk_, kk_ + K.int32(1))
                    K.assign(cur, work_wait(kk_))

            # All four auxiliary warps must synchronize before reallocating.
            K.ptx.bar.sync(K.uint32(7), K.uint32(128))

        with auxg:
            with mma:
                tm = tmem_preamble()
                cyc = K.local_scalar("int32", init=K.int32(0))

                def mwait(nm):
                    MBG[nm].wait(0, cyc & K.int32(1))
                    K.ptx[TC_FENCE_AFTER]()

                mphase, mphase_end = make_phaser()
                bd = K.alloc_local([1], "uint64")
                zq = K.alloc_local([1], "int32")
                op_T1k = Op(bd, T1, 128, 64, "k")
                op_T2k = Op(bd, T2, 128, 64, "k")
                op_T3k = Op(bd, T3, 128, 64, "k")
                op_T3mn = Op(bd, T3, 128, 128, "mn")
                op_ZTk = Op(bd, ZT, 128, 64, "k")
                op_ZTmn = Op(bd, ZT, 128, 128, "mn")
                op_dAqk_k = Op(bd, T5, 64, 64, "k")
                op_dAkk_k = Op(bd, T5 + 1, 64, 64, "k")
                op_T6mn = Op(bd, T6, 128, 128, "mn")
                op_dAqk_mn = Op(bd, T5, 64, 64, "mn")
                op_dAm_k = Op(bd, DAM, 64, 64, "k")
                op_X_mn = Op(bd, T6, 64, 64, "mn")
                op_dAkk_mn = Op(bd, T5 + 1, 64, 64, "mn")
                op_DHBk = Op(bd, DHB, 128, 128, "k")
                op_DHBmn = Op(bd, DHB, 128, 128, "mn")
                op_DV2k = Op(bd, DV2, 128, 64, "k")
                op_DV2mn = Op(bd, DV2, 128, 128, "mn")
                op_DVBmn = Op(bd, DVB, 128, 128, "mn")
                op_do_k128 = Op(bd, S_DO, 64, 128, "k")
                op_do_mn64 = Op(bd, S_DO, 64, 64, "mn")
                op_h_k = Op(bd, S_H, 128, 128, "k")
                op_h_mn = Op(bd, S_H, 128, 128, "mn")
                op_aqk_mn = Op(bd, S_AQK, 64, 64, "mn")
                op_akk_k = Op(bd, S_AKK, 64, 64, "k")
                op_akk_mn = Op(bd, S_AKK, 64, 64, "mn")
                ID_128x64 = idesc(128, 64)
                ID_128x64_TATB_NB = idesc(128, 64, ta=1, tb=1, nb=1)
                ID_128x64_TATB = idesc(128, 64, ta=1, tb=1)
                ID_128x64_TB = idesc(128, 64, tb=1)
                ID_128x64_TB_NA = idesc(128, 64, tb=1, na=1)
                ID_64x64_TB = idesc(64, 64, tb=1)
                ID_64x64_TATB = idesc(64, 64, ta=1, tb=1)
                ID_64x64 = idesc(64, 64)


                op_egT = Op(bd, ST_G, 64, 64, "mn")
                op_vT = Op(bd, ST_V, 64, 64, "mn")
                op_qT = Op(bd, ST_Q, 64, 64, "mn")
                op_kT = Op(bd, ST_K, 64, 64, "mn")
                ID_T = idesc(128, 16, ta=1)
                bdI = K.alloc_local([1], "uint64")
                K.cuda.tcgen05.encode_matrix_descriptor(
                    K.address_of(bdI[0]), s_ident.ptr_to([0]), ldo=8, sdo=16, swizzle=0)

                item = K.local_scalar("int32", init=cur - num_streams)
                with K.While(cur < total_work):
                    K.assign(item, cur - num_streams)
                    with K.serial(G) as gi:
                        par = cyc & K.int32(1)
                        K.ptx.ld.volatile.shared.s32(zq[0], K.address_of(s_tmem[1]))
                        K.cuda.tcgen05.encode_matrix_descriptor(
                            K.address_of(bd[0]), TT[zq[0]].ptr_to(0, 0), ldo=Op.LBO_BASE, sdo=SBO_UNITS,
                            swizzle=K.SW128B.value)
                        akk_u = K.local_scalar("uint64", init=K.Cast("uint64", par) * K.uint64(UNITS_PER_STAGE))
                        mphase("mw-xT")
                        b_in_full.wait(0, par)
                        b_eg_full.wait(0, par)

                        b_h_free.wait(0, par ^ K.int32(1))
                        K.ptx[TC_FENCE_AFTER]()
                        mphase("m-xT")
                        with K.If(elected()), K.Then():
                            for src, dst in ((op_egT, S1), (op_vT, S2), (op_qT, S3), (op_kT, S4)):
                                for j in range(4):
                                    K.ptx[MMA_SS](
                                        K.Cast("uint32", tm[0] + dst + 16 * j),
                                        src.desc(j),
                                        bdI[0],
                                        K.uint32(ID_T),
                                        K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0),
                                        K.ptx.pred(0),
                                    )
                            TCG["xT_done"].arrive(0)
                        mphase("mw-early")
                        mwait("t_early")
                        b_akk_full.wait(par, (cyc >> 1) & K.int32(1))
                        b_akk_masked.wait(par, (cyc >> 1) & K.int32(1))
                        b_h_full.wait(0, par)
                        K.ptx[TC_FENCE_AFTER]()
                        mphase("m-Z")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S2, op_h_mn, op_T3mn, ID_128x64_TATB_NB, True)
                            TCG["Z_done"].arrive(0)
                        mphase("mw-aqk")
                        b_aqk_masked.wait(0, par)
                        b_do_full.wait(0, par)
                        K.ptx[TC_FENCE_AFTER]()
                        mphase("m-dvp")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S3, op_do_mn64, op_aqk_mn, ID_128x64_TATB, False)
                            b_aqk_empty.arrive(0)
                        mphase("mw-dhb")
                        b_dhb_full.wait(0, par)
                        K.ptx[TC_FENCE_AFTER]()
                        mphase("m-dv2")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S3, op_DHBmn, op_T2k if False else Op(bd, T2, 128, 128, "mn"), ID_128x64_TATB, True)
                            TCG["dv2_done"].arrive(0)
                        mphase("mw-zT")
                        mwait("zT_ready")
                        mphase("m-Vn")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S2, op_ZTk, op_akk_k, ID_128x64, False, b_units=akk_u)
                            TCG["Vn_done"].arrive(0)
                        mphase("mw-dv2T")
                        mwait("dv2T_ready")
                        mphase("m-dAs")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S4, op_DV2mn, op_ZTmn, ID_64x64_TATB, False)
                            TCG["dAs_done"].arrive(0)
                            mma_chain(tm, S3, op_DV2k, op_akk_mn, ID_128x64_TB, False, b_units=akk_u)
                            TCG["dvb_done"].arrive(0)
                        mphase("mw-vnT")
                        mwait("vnT_ready")
                        mphase("m-dAqk")
                        with K.If(elected()), K.Then():
                            if HALF_DA_READOUT:
                                mma_chain(tm, S4 + (16 << 16), op_do_k128, op_T6mn, ID_64x64_TB, False)
                            else:
                                mma_chain(tm, S1, op_do_k128, op_T6mn, ID_64x64_TB, False)
                            TCG["dAqk_done"].arrive(0)
                            mma_chain(tm, S5, op_DHBk, op_T6mn, ID_128x64_TB, False)
                            TCG["dk_done"].arrive(0)
                        mphase("mw-dAm")
                        mwait("dAm_ready")
                        mwait("dAqk_tile_ready")
                        mphase("m-X")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S1, op_dAm_k, op_akk_k, ID_64x64, False, b_units=akk_u)
                            TCG["X_done"].arrive(0)
                            mma_chain(tm, S4, op_h_k, op_do_k128, ID_128x64, False)
                            b_do_empty.arrive(0)
                            mma_chain(tm, S4, op_T2k, op_dAqk_k, ID_128x64, True)
                            TCG["dq2_done"].arrive(0)
                        mphase("mw-dvepi")
                        mwait("dv_epi_done")
                        mphase("m-dwb")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S6, op_h_k, op_DVBmn, ID_128x64_TB_NA, False)
                        mphase("mw-X")
                        mwait("X_ready")
                        mphase("m-Y")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S2, op_akk_mn, op_X_mn, ID_64x64_TATB, False, a_units=akk_u)
                            TCG["Y_done"].arrive(0)
                            b_akk_empty.arrive(par)
                        mphase("mw-intra")
                        mwait("intra_ready")
                        mphase("m-dkt")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S6, op_T2k, op_dAkk_k, ID_128x64, True)
                            mma_chain(tm, S3, op_T1k, op_dAqk_mn, ID_128x64_TB, False)
                            mma_chain(tm, S3, op_T3k, op_dAkk_mn, ID_128x64_TB, True)
                            TCG["dkt_done"].arrive(0)
                            TCG["chunk_done"].arrive(0)
                        mphase_end()
                        K.assign(cyc, cyc + K.int32(1))
                    K.assign(kk_, kk_ + K.int32(1))
                    K.assign(cur, work_wait(kk_))

            with loader:
                with K.If(elected()), K.Then():
                    for m in (q_map, k_map, v_map, eg_map, do_map, aqk_map, akk_map, h_map, dh_map):
                        K.ptx.prefetch.tensormap(K.address_of(m))
                cyc = K.local_scalar("int32", init=K.int32(0))
                lphase, lphase_end = make_phaser()
                item = K.local_scalar("int32", init=cur - num_streams)
                with K.While(cur < total_work):
                    K.assign(item, cur - num_streams)
                    c, hq, seq, n, bos, rows, nch_i = item_coords(item)
                    bos32 = K.local_scalar("int32", init=K.Cast("int32", bos))
                    tok0 = K.local_scalar("int32", init=bos32 + n * K.int32(CHUNK))


                    lphase("lw-flags")
                    with K.If(elected()), K.Then():



                        tgt_f = K.local_scalar("int64", init=ep64 + K.Cast("int64", n + K.int32(1)))
                        tgt_b = K.local_scalar("int64", init=ep64 + K.Cast("int64", nch_i - n))
                        for gi_ in range(G):
                            fidx = seq * K.int32(HV) + hq * K.int32(G) + K.int32(gi_)
                            flf = K.local_scalar("int64", init=K.int64(0))
                            with K.While(flf < tgt_f):
                                K.ptx.ld.acquire.gpu.global_.s64(flf, flags.ptr_to([fidx]))
                            flb = K.local_scalar("int64", init=K.int64(0))
                            with K.While(flb < tgt_b):
                                K.ptx.ld.acquire.gpu.global_.s64(flb, flags.ptr_to([num_chains + fidx]))


                        with K.If(rows < K.int32(CHUNK)), K.Then():
                            tgt_1 = K.local_scalar("int64", init=ep64 + K.int64(1))
                            s2 = K.local_scalar("int32", init=seq + K.int32(1))
                            b2 = K.local_scalar("int32", init=tok0 + rows)
                            with K.While((s2 < num_seqs) & (b2 < tok0 + K.int32(CHUNK))):
                                for gi_ in range(G):
                                    fl2 = K.local_scalar("int64", init=K.int64(0))
                                    with K.While(fl2 < tgt_1):
                                        K.ptx.ld.acquire.gpu.global_.s64(
                                            fl2, flags.ptr_to([s2 * K.int32(HV) + hq * K.int32(G) + K.int32(gi_)]))
                                _, l2 = seq_len_of(s2)
                                K.assign(b2, b2 + l2)
                                K.assign(s2, s2 + K.int32(1))
                    K.ptx["bar.warp.sync"](K.uint32(0xFFFFFFFF))
                    K.ptx["fence.proxy.async.global"]()
                    lphase_end()
                    with K.serial(G) as gi:
                        hv = K.local_scalar("int32", init=hq * K.int32(G) + gi)
                        par = cyc & K.int32(1)
                        npar = par ^ K.int32(1)
                        hidx = c * K.int32(HV) + hv
                        lphase("lw-mid")
                        with K.If(cyc > K.int32(0)), K.Then():
                            b_mid_free.wait(0, npar)
                        lphase("l-issue")
                        with K.If(elected()), K.Then():
                            b_in_full.arrive(0, tx_count=IN_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(b_in_full.ptr_to([0]))
                            for d0 in (0, 64):
                                K.ptx[TMA_LD](TT[ST_Q + d0 // 64].ptr_to(0, 0), K.address_of(q_map), K.int32(d0), tok0, hq, mb)
                                K.ptx[TMA_LD](TT[ST_K + d0 // 64].ptr_to(0, 0), K.address_of(k_map), K.int32(d0), tok0, hq, mb)
                                K.ptx[TMA_LD](TT[ST_V + d0 // 64].ptr_to(0, 0), K.address_of(v_map), K.int32(d0), tok0, hv, mb)
                        load_beta_lanes_g(bos, hv, n, rows, par)
                        b_in_full.arrive(0)
                        lphase("lw-chunk")
                        with K.If(cyc > K.int32(0)), K.Then():
                            TCG["chunk_done"].wait(0, npar)
                        lphase("l-issue2")
                        with K.If(elected()), K.Then():
                            b_eg_full.arrive(0, tx_count=EG_BYTES)
                            mbe = K.cuda.cvta_generic_to_shared(b_eg_full.ptr_to([0]))
                            for d0 in (0, 64):
                                K.ptx[TMA_LD](TT[ST_G + d0 // 64].ptr_to(0, 0), K.address_of(eg_map), K.int32(d0), tok0, hv, mbe)
                        with K.If(cyc > K.int32(0)), K.Then():
                            b_do_empty.wait(0, npar)
                        with K.If(elected()), K.Then():
                            b_do_full.arrive(0, tx_count=DO_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(b_do_full.ptr_to([0]))
                            for d0 in (0, 64):
                                K.ptx[TMA_LD](TT[S_DO + d0 // 64].ptr_to(0, 0), K.address_of(do_map), K.int32(d0), tok0, hv, mb)
                        with K.If(cyc > K.int32(0)), K.Then():
                            b_h_free.wait(0, npar)
                        with K.If(elected()), K.Then():
                            b_h_full.arrive(0, tx_count=H_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(b_h_full.ptr_to([0]))
                            for d0 in (0, 64):
                                K.ptx[TMA_LD](TT[S_H + (d0 // 64) * 2].ptr_to(0, 0), K.address_of(h_map), K.int32(d0), K.int32(0), hidx, mb)
                        with K.If(cyc > K.int32(0)), K.Then():
                            b_aqk_empty.wait(0, npar)
                        with K.If(elected()), K.Then():
                            b_aqk_full.arrive(0, tx_count=A_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(b_aqk_full.ptr_to([0]))
                            K.ptx[TMA_LD](TT[S_AQK].ptr_to(0, 0), K.address_of(aqk_map), K.int32(0), tok0, hv, mb)
                        with K.If(cyc > K.int32(1)), K.Then():
                            b_akk_empty.wait(par, ((cyc >> 1) & K.int32(1)) ^ K.int32(1))
                        with K.If(elected()), K.Then():
                            b_akk_full.arrive(par, tx_count=A_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(b_akk_full.ptr_to([par]))
                            K.ptx[TMA_LD](TT[S_AKK + par].ptr_to(0, 0), K.address_of(akk_map), K.int32(0), tok0, hv, mb)

                        lphase("lw-qkfree")
                        TCG["xT_done"].wait(0, par)
                        with K.If(cyc > K.int32(0)), K.Then():
                            TCG["dk_done"].wait(0, npar)
                        lphase("l-dhb")
                        with K.If(elected()), K.Then():
                            b_dhb_full.arrive(0, tx_count=H_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(b_dhb_full.ptr_to([0]))
                            for d0 in (0, 64):
                                K.ptx[TMA_LD](TT[DHB + (d0 // 64) * 2].ptr_to(0, 0), K.address_of(dh_map), K.int32(d0), K.int32(0), hidx, mb)

                            with K.If(gi + K.int32(1) < K.int32(G)):
                                with K.Then():
                                    hvn = hv + K.int32(1)
                                    for tmap in (v_map, do_map, eg_map):
                                        for d0 in (0, 64):
                                            K.ptx[TMA_PREFETCH](K.address_of(tmap), K.int32(d0), tok0, hvn)
                                    K.ptx[TMA_PREFETCH](K.address_of(aqk_map), K.int32(0), tok0, hvn)
                                    K.ptx[TMA_PREFETCH](K.address_of(akk_map), K.int32(0), tok0, hvn)
                                    for d0 in (0, 64):
                                        K.ptx[TMA_PREFETCH](K.address_of(h_map), K.int32(d0), K.int32(0), hidx + K.int32(1))
                                        K.ptx[TMA_PREFETCH](K.address_of(dh_map), K.int32(d0), K.int32(0), hidx + K.int32(1))
                        lphase_end()
                        K.assign(cyc, cyc + K.int32(1))
                    claim_publish(kk_ + K.int32(1))
                    K.assign(kk_, kk_ + K.int32(1))
                    K.assign(cur, work_wait(kk_))

            with w10:
                g_masker(K.int32(0))
            with w11:
                g_masker(K.int32(1))
        K.cuda.cta_sync()
        with K.If(K.warp_id() == 8), K.Then():
            K.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
            K.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                K.Cast("uint32", K.local_scalar("int32", init=tmem_preamble()[0])), K.uint32(TMEM_COLS))

        with K.If(K.thread_id() == K.int32(0)), K.Then():
            done = K.local_scalar("int32")
            K.ptx["atom.acq_rel.gpu.global.add.s32"](done, stream_counter.ptr_to([1]), K.int32(1))
            with K.If(done == num_ctas - K.int32(1)), K.Then():
                K.ptx["st.release.gpu.global.s32"](stream_counter.ptr_to([0]), K.int32(0))
                K.ptx["st.release.gpu.global.s32"](stream_counter.ptr_to([1]), K.int32(0))

    K.MBarrier._wait = _CUDA_MBAR_WAIT
    return kda_bwd_mega


AQK_BYTES = CHUNK * CHUNK * 2




def make_fused_kernel(H: int, sched_maxp2: int, sched_maxp1: int, static_grid=None):
    K.MBarrier._wait = _CUDA_MBAR_WAIT
    TM_DH = 0
    S1, S2, S3, S4, S6, S5 = 128, 192, 256, 320, 384, 448
    DO_BYTES = CHUNK * D * 2
    H_BYTES = D * D * 2
    SCHED_MAXP2, SCHED_MAXP1 = sched_maxp2, sched_maxp1
    SCHED_STRIDE = 2 + SCHED_MAXP2 + SCHED_MAXP1
    HK = H * D
    HK64 = K.int64(HK)
    QKVE_BYTES = 4 * CHUNK * D * 2









    T1, T2, T3, T5, T6, DHB = 0, 2, 4, 8, 10, 12
    DV2, ZT, DVB, DAM = 6, 8, 6, 9
    PB0, PB1 = 12, 14
    ST_Q, ST_K, ST_V, ST_G = 12, 14, 16, 8


    S_DO, S_H, S_AQK, S_AKK = 18, 20, 24, 25
    IN_BYTES = 3 * CHUNK * D * 2 + CHUNK * 8 * 2
    EG_BYTES = CHUNK * D * 2

    @K.kernel(warps=12, arch="sm_100a", min_blocks_per_sm=1,
              grid="num_ctas" if static_grid is None else static_grid)
    def kda_bwd_fused(
        q: K.gptr[K.bf16],
        k: K.gptr[K.bf16],
        v: K.gptr[K.bf16],
        beta: K.gptr[K.bf16],
        aqk: K.gptr[K.bf16],
        akk: K.gptr[K.bf16],
        g: K.gptr[K.f32],
        egcache: K.gptr[K.bf16],
        do: K.gptr[K.bf16],
        dht: K.gptr[K.f32],
        h0: K.gptr[K.f32],
        hsnap: K.gptr[K.bf16],
        cu_seqlens: K.gptr[K.i64],
        flags: K.gptr[K.i32],
        sched: K.gptr[K.i32],
        dq: K.gptr[K.f32],
        dk: K.gptr[K.f32],
        dv: K.gptr[K.bf16],
        db: K.gptr[K.f32],
        dg: K.gptr[K.f32],
        dh0: K.gptr[K.f32],
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        g_map: K.TensorMap,
        eg_map: K.TensorMap,
        beta_map: K.TensorMap,
        do_map: K.TensorMap,
        aqk_map: K.TensorMap,
        akk_map: K.TensorMap,
        h_map: K.TensorMap,
        scale: K.f32,
        num_seqs: K.i32,
        num_ctas: K.i32,
        epoch: K.i32,
    ):
        for buf in (q, k, v, beta, aqk, akk, g, do, hsnap, egcache):
            K.keep_alive(buf.data)
        num_work = num_seqs * K.int32(H)




        cta = K.local_scalar("int32", init=K.Cast("int32", K.cta_id()))
        sbase = K.local_scalar("int32", init=cta * K.int32(SCHED_STRIDE))
        n_p2 = K.local_scalar("int32")
        K.ptx.ld.global_.s32(n_p2, sched.ptr_to([sbase]))
        n_p1 = K.local_scalar("int32")
        K.ptx.ld.global_.s32(n_p1, sched.ptr_to([sbase + K.int32(1)]))

        def p2_chain(i):
            c = K.local_scalar("int32")
            K.ptx.ld.global_.s32(c, sched.ptr_to([sbase + K.int32(2) + i]))
            return c

        def p1_chain(i):
            c = K.local_scalar("int32")
            K.ptx.ld.global_.s32(c, sched.ptr_to([sbase + K.int32(2 + SCHED_MAXP2) + i]))
            return c

        sp = K.specialize()
        cg = sp.role("cg", warps=list(range(8)), regs=208)
        auxg = sp.warpgroup("aux", warps=[8, 9, 10, 11], regs=88)
        loader = sp.role("loader", warps=[8], group=auxg)
        mma = sp.role("mma", warps=[9], group=auxg)
        idle = sp.role("idle", warps=[10, 11], group=auxg)

        smem = K.smem_pool()
        s_tmem = smem.alloc((4,), K.i32, align=16)
        b_in_full = K.TMABar(smem, 1); b_in_full.init(1)
        b_eg_full = K.TMABar(smem, 1); b_eg_full.init(1)
        b_mid_free = K.MBarrier(smem, 1); b_mid_free.init(256)
        b_do_full = K.TMABar(smem, 1); b_do_full.init(1)
        b_h_full = K.TMABar(smem, 1); b_h_full.init(1)
        b_aqk_full = K.TMABar(smem, 1); b_aqk_full.init(1)
        b_akk_full = K.TMABar(smem, 2); b_akk_full.init(1)
        b_do_empty = K.TCGen05Bar(smem, 1); b_do_empty.init(1)
        b_h_free = K.MBarrier(smem, 1); b_h_free.init(256)
        b_aqk_empty = K.TCGen05Bar(smem, 1); b_aqk_empty.init(1)
        b_akk_empty = K.TCGen05Bar(smem, 2); b_akk_empty.init(1)
        mb_names = ["t_early", "dhb_ready", "zT_ready", "vnT_ready", "dv2T_ready",
                    "dAqk_tile_ready", "dAm_ready", "X_ready", "intra_ready", "dv_epi_done"]
        MB = {}
        for nm in mb_names:
            MB[nm] = K.MBarrier(smem, 1)
            MB[nm].init(256)
        b_dg0_ready = K.MBarrier(smem, 1); b_dg0_ready.init(256)
        b_aqk_masked = K.MBarrier(smem, 1); b_aqk_masked.init(64)

        p_kv = K.Pipeline(smem, 2, full="tma", empty="mbar", init_empty=256)
        b_kvT_done = K.TCGen05Bar(smem, 1); b_kvT_done.init(1)
        b_kv_read = K.MBarrier(smem, 1); b_kv_read.init(256)
        p_akk1 = K.Pipeline(smem, 1, full="tma", empty="tcgen05")
        p_tiles = K.Pipeline(smem, 1, full="mbar", empty="tcgen05", init_full=256)
        p_hs = K.Pipeline(smem, 1, full="mbar", empty="mbar", init_full=256, init_empty=9)
        p_w = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=256)
        p_vn = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=256)
        p_g = K.Pipeline(smem, 2, full="tma", empty="mbar", init_empty=256)
        tc_names = ["Z_done", "Vn_done", "dv2_done", "dAqk_done", "dk_done",
                    "dAs_done", "dvb_done", "X_done", "Y_done",
                    "dq2_done", "dkt_done", "chunk_done", "xT_done"]
        TC = {}
        for nm in tc_names:
            TC[nm] = K.TCGen05Bar(smem, 1)
            TC[nm].init(1)

        TT = smem.alloc((27, 64, 64), K.bf16, swizzle=K.SW128B)
        s_beta = smem.alloc((64,), K.f32, align=16)
        s_beta_in = smem.alloc((CHUNK, 8), K.bf16, align=128)
        s_dgk = smem.alloc((2, 128), K.f32, align=16)
        s_cs = smem.alloc((128,), K.f32, align=16)
        s_beta_g = smem.alloc((2, CHUNK, 8), K.bf16, align=128)
        s_beta1 = smem.alloc((2, CHUNK), K.f32, align=16)


        s_ident = smem.alloc((256,), K.bf16, align=128)

        with K.If(K.thread_id() == 0), K.Then():
            K.ptx.st.shared.s32(K.address_of(s_tmem[1]), K.int32(0))
            K.ptx.fence.mbarrier_init.release.cluster()
        with K.If(K.thread_id() < K.int32(256)), K.Then():
            tid_i = K.thread_id()
            n_i = tid_i >> 4
            k_i = tid_i & K.int32(15)
            K.ptx.st.shared.u16(
                s_ident.ptr_to([(n_i >> 3) * K.int32(128) + (k_i >> 3) * K.int32(64) + (n_i & K.int32(7)) * K.int32(8) + (k_i & K.int32(7))]),
                K.Cast("uint16", K.Select(n_i == k_i, K.int32(0x3F80), K.int32(0))))
            K.ptx[FENCE_ASYNC]()
        K.cuda.cta_sync()
        with K.If(K.warp_id() == 8), K.Then():
            K.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                K.address_of(s_tmem[0]), K.uint32(512))
        K.cuda.cta_sync()

        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def make_phaser():
            """Sequential IKET ranges for one role: phase(name) ends the current range and starts the next."""
            tok = K.alloc_local([1], "uint32")
            K.assign(tok[0], K.cuda.iket.sentinel_token("idle"))

            def phase(name):
                K.cuda.iket.range_end(tok[0])
                K.assign(tok[0], K.cuda.iket.range_start(name))

            def phase_end():
                K.cuda.iket.range_end(tok[0])
                K.assign(tok[0], K.cuda.iket.sentinel_token("idle"))

            return phase, phase_end

        def tmem_preamble():
            tmv = K.alloc_local([1], "int32")
            K.ptx.ld.volatile.shared.s32(tmv[0], K.address_of(s_tmem[0]))
            return tmv

        def pack_bf16x2(dst, lo, hi):
            K.ptx.cvt.rn.bf16x2.f32(dst, hi, lo)

        def work_coords(work):

            seq = K.local_scalar("int32", init=work // K.int32(H))
            head = K.local_scalar("int32", init=work - seq * K.int32(H))
            cs = K.alloc_local([2], "int64")
            K.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([seq]))
            K.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([seq + K.int32(1)]))
            bos = K.local_scalar("int64", init=cs[0])
            seq_len = K.local_scalar("int32", init=K.Cast("int32", cs[1] - cs[0]))
            nch = K.local_scalar("int32", init=(seq_len + K.int32(CHUNK - 1)) >> 6)
            return seq, head, bos, seq_len, nch

        def chunk_base(seq):
            cb = K.local_scalar("int32", init=K.int32(0))
            with K.serial(seq) as i:
                cs = K.alloc_local([2], "int64")
                K.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([i]))
                K.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([i + 1]))
                K.assign(cb, cb + ((K.Cast("int32", cs[1] - cs[0]) + K.int32(CHUNK - 1)) >> 6))
            return cb









        P1_KV, P1_AKK, P1_HS, P1_G, P1_KG, P1_KBG, P1_VB = 0, 8, 9, 13, 21, 23, 25
        G_BYTES = CHUNK * D * 4 + CHUNK * 8 * 2
        TM_H, TM_W, TM_U = 0, 128, 192
        TM_KT, TM_VT = 256, 320

        def bf16_bits_to_f32(u16val):
            return K.reinterpret("float32", K.Cast("uint32", u16val) << K.uint32(16))

        def p1_compute():
            tm = tmem_preamble()
            wr = K.warp_id_in_role()
            lane = K.lane_id()
            wg = K.local_scalar("int32", init=wr >> 2)
            quad = K.local_scalar("int32", init=wr & 3)
            x = K.local_scalar("int32", init=quad * 32 + lane)
            row0 = K.local_scalar("int32", init=wg * 32)
            x64 = K.Cast("int64", x)
            xs = K.local_scalar("int32", init=x >> 6)
            xr = K.local_scalar("int32", init=x & 63)
            xg = K.local_scalar("int32", init=x >> 5)
            xgc = K.local_scalar("int32", init=(x & 31) * 2)

            def tmem_at(col):
                return K.Cast("uint32", tm[0] + col + (quad << 21))

            st_kv = K.PipelineState(2, phase=0)
            st_te = K.PipelineState(1, phase=1)
            st_hs = K.PipelineState(1, phase=1)
            st_g = K.PipelineState(2, phase=0)
            st_w = K.PipelineState(1, phase=0)
            st_vn = K.PipelineState(1, phase=0)
            p1c = K.local_scalar("int32", init=K.int32(0))
            gv = K.alloc_local([32], "float32")
            kk = K.alloc_local([32], "float32")
            vv = K.alloc_local([32], "float32")
            bb = K.alloc_local([32], "float32")
            acc = K.alloc_local([64], "float32")
            wds = K.alloc_local([32], "uint32")
            gn = K.local_scalar("float32")
            egn = K.local_scalar("float32")
            eg = K.local_scalar("float32")
            egng = K.local_scalar("float32")
            bu = K.local_scalar("uint16")
            ku = K.local_scalar("uint16")
            vu = K.local_scalar("uint16")
            phase, phase_end = make_phaser()
            tid_all = K.local_scalar("int32", init=wr * 32 + lane)
            with K.serial(n_p1) as it:
                chain = K.local_scalar("int32", init=p1_chain(it))
                seq, head, bos, seq_len, nch = work_coords(chain)
                head64 = K.Cast("int64", head)
                gcol = K.local_scalar("int64", init=head64 * K.int64(D) + x64)
                def h_c0():
                    rows = K.int32(CHUNK)
                    p_g.full.wait(st_g.stage, st_g.phase)
                    gst = K.local_scalar("int32", init=P1_G + st_g.stage * K.int32(4) + xg)
                    with K.If(lane < K.int32(8)), K.Then():
                        btok = wr * K.int32(8) + lane
                        K.ptx.ld.shared.u16(bu, s_beta_g.ptr_to([st_g.stage, btok, head & K.int32(7)]))
                        K.ptx.st.shared.f32(K.address_of(s_beta1[st_g.stage, btok]), bf16_bits_to_f32(bu))
                    for i in range(32):
                        K.ptx.ld.shared.f32(gv[i], TT[gst].ptr_to(row0 + i, xgc))
                    K.ptx.ld.shared.f32(gn, TT[gst].ptr_to(rows - K.int32(1), xgc))
                    K.ptx.bar.sync(K.uint32(1), K.uint32(256))
                    for u in range(8):
                        K.ptx["ld.shared.v4.f32"](bb[4 * u], bb[4 * u + 1], bb[4 * u + 2], bb[4 * u + 3],
                                                  K.address_of(s_beta1[st_g.stage, row0 + 4 * u]))
                    K.ptx[FENCE_ASYNC]()
                    p_g.empty.arrive(st_g.stage)
                    st_g.advance()

                phase("h-c0")
                h_c0()
                with K.serial(nch) as n:
                    rows = K.int32(CHUNK)
                    tok0 = K.local_scalar("int64", init=bos + K.Cast("int64", n * K.int32(CHUNK)))
                    K.ptx.ex2.approx.ftz.f32(egn, gn)
                    phase("hw-kv")
                    b_kvT_done.wait(0, p1c & K.int32(1))
                    K.ptx[TC_FENCE_AFTER]()
                    phase("h-kv")
                    kst = K.local_scalar(
                        "int32",
                        init=P1_KV + st_kv.stage * K.int32(4) + xs * K.int32(2),
                    )
                    K.ptx[TC_LD32](*(kk[i] for i in range(32)), tmem_at(TM_KT + wg * 32))
                    for i in range(32):
                        K.ptx.ld.shared.u16(vu, TT[kst + K.int32(1)].ptr_to(row0 + i, xr))
                        K.assign(vv[i], bf16_bits_to_f32(vu))
                    K.ptx[WAIT_LD]()
                    K.ptx[FENCE_ASYNC]()
                    p_kv.empty.arrive(st_kv.stage)
                    st_kv.advance()
                    K.ptx[TC_FENCE_BEFORE]()
                    b_kv_read.arrive(0)
                    K.assign(p1c, p1c + K.int32(1))
                    phase("hw-tiles")
                    p_tiles.empty.wait(0, st_te.phase)
                    st_te.advance()
                    K.ptx[TC_FENCE_AFTER]()
                    K.ptx[FENCE_ASYNC]()
                    phase("h-tiles")
                    for u in range(4):
                        wkg = K.alloc_local([4], "uint32")
                        wkbg = K.alloc_local([4], "uint32")
                        wvb = K.alloc_local([4], "uint32")
                        vals = K.alloc_local([24], "float32")
                        for e in range(8):
                            i = 8 * u + e
                            K.ptx.ex2.approx.ftz.f32(eg, gv[i])
                            K.ptx.ex2.approx.ftz.f32(egng, gn - gv[i])
                            K.ptx.cvt.rn.bf16.f32(bu, eg)
                            egidx = (tok0 + K.Cast("int64", row0 + K.int32(i))) * HK64 + gcol
                            K.ptx["st.global.L1::no_allocate.b16"](egcache.ptr_to([egidx]), bu)
                            K.assign(vals[e], kk[i] * egng)
                            K.assign(vals[8 + e], kk[i] * bb[i] * eg)
                            K.assign(vals[16 + e], vv[i] * bb[i])
                        for p in range(4):
                            pack_bf16x2(wkg[p], vals[2 * p], vals[2 * p + 1])
                            pack_bf16x2(wkbg[p], vals[8 + 2 * p], vals[8 + 2 * p + 1])
                            pack_bf16x2(wvb[p], vals[16 + 2 * p], vals[16 + 2 * p + 1])
                        col = row0 + 8 * u
                        K.ptx["st.shared.v4.b32"](TT[P1_KG + xs].ptr_to(xr, col), wkg[0], wkg[1], wkg[2], wkg[3])
                        K.ptx["st.shared.v4.b32"](TT[P1_KBG + xs].ptr_to(xr, col), wkbg[0], wkbg[1], wkbg[2], wkbg[3])
                        K.ptx["st.shared.v4.b32"](TT[P1_VB + xs].ptr_to(xr, col), wvb[0], wvb[1], wvb[2], wvb[3])

                    K.ptx[FENCE_ASYNC]()
                    p_tiles.full.arrive(0)
                    phase("hw-hs")
                    p_hs.empty.wait(st_hs.stage, st_hs.phase)
                    K.ptx[TC_FENCE_AFTER]()
                    phase("h-decay")
                    hc0 = wg * 64
                    hsst = K.local_scalar("int32", init=P1_HS + st_hs.stage * K.int32(4) + wg * K.int32(2) + xs)
                    with K.If(n == K.int32(0)):
                        with K.Then():
                            h0base = ((K.Cast("int64", seq) * K.int64(H) + head64) * K.int64(D) + x64) * K.int64(D) \
                                + K.Cast("int64", hc0)
                            for m in range(8):
                                K.ptx["ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.f32"](
                                    *(acc[8 * m + i] for i in range(8)),
                                    h0.ptr_to([h0base + K.int64(8 * m)]))
                        with K.Else():
                            K.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_H + hc0))
                            K.ptx[TC_LD32](*(acc[32 + i] for i in range(32)), tmem_at(TM_H + hc0 + 32))
                            K.ptx[WAIT_LD]()
                    for p in range(32):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    for u in range(8):
                        K.ptx["st.shared.v4.b32"](TT[hsst].ptr_to(xr, 8 * u),
                                                  wds[4 * u], wds[4 * u + 1], wds[4 * u + 2], wds[4 * u + 3])
                    for p in range(32):
                        dpair = K.local_scalar("uint64")
                        K.ptx["mul.rn.f32x2"](dpair, K.cuda.make_float2(acc[2 * p], acc[2 * p + 1]),
                                              K.cuda.make_float2(egn, egn))
                        K.assign(acc[2 * p], K.cuda.float2_x(dpair))
                        K.assign(acc[2 * p + 1], K.cuda.float2_y(dpair))
                    K.ptx[TC_ST32](tmem_at(TM_H + hc0), *(acc[i] for i in range(32)))
                    K.ptx[TC_ST32](tmem_at(TM_H + hc0 + 32), *(acc[32 + i] for i in range(32)))
                    K.ptx[WAIT_ST]()
                    K.ptx[TC_FENCE_BEFORE]()
                    K.ptx[FENCE_ASYNC]()
                    p_hs.full.arrive(st_hs.stage)
                    phase("hw-W")
                    p_w.full.wait(0, st_w.phase)
                    st_w.advance()
                    K.ptx[TC_FENCE_AFTER]()
                    K.ptx[FENCE_ASYNC]()
                    phase("h-wT")
                    K.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_W + wg * 32))
                    K.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    for u in range(4):
                        K.ptx["st.shared.v4.b32"](TT[P1_KBG + xs].ptr_to(xr, row0 + 8 * u),
                                                  wds[4 * u], wds[4 * u + 1], wds[4 * u + 2], wds[4 * u + 3])
                    K.ptx[TC_FENCE_BEFORE]()
                    K.ptx[FENCE_ASYNC]()
                    p_w.empty.arrive(0)
                    phase("h-c0")
                    with K.If(n + K.int32(1) < nch), K.Then():
                        h_c0()
                    phase("hw-Vn")
                    p_vn.full.wait(0, st_vn.phase)
                    st_vn.advance()
                    K.ptx[TC_FENCE_AFTER]()
                    K.ptx[FENCE_ASYNC]()
                    phase("h-vnT")
                    K.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_U + wg * 32))
                    K.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    for u in range(4):
                        K.ptx["st.shared.v4.b32"](TT[P1_VB + xs].ptr_to(xr, row0 + 8 * u),
                                                  wds[4 * u], wds[4 * u + 1], wds[4 * u + 2], wds[4 * u + 3])
                    K.ptx[TC_FENCE_BEFORE]()
                    K.ptx[FENCE_ASYNC]()
                    p_vn.empty.arrive(0)

                    with K.If(elected()), K.Then():
                        p_hs.empty.arrive(st_hs.stage)
                    st_hs.advance()
                    phase_end()

            p_tiles.empty.wait(0, st_te.phase)
            K.ptx[TC_FENCE_AFTER]()

        def p1_mma():
            tm = tmem_preamble()



            bd1 = K.alloc_local([1], "uint64")
            zq1 = K.alloc_local([1], "int32")
            op_kbg_k = Op(bd1, P1_KBG, 128, 64, "k")
            op_vb_k = Op(bd1, P1_VB, 128, 64, "k")
            op_kg_k = Op(bd1, P1_KG, 128, 64, "k")
            op_akk1_k = Op(bd1, P1_AKK, 64, 64, "k")
            op_hs_mn = Op(bd1, P1_HS, 128, 128, "mn")
            op_w_mn = Op(bd1, P1_KBG, 128, 128, "mn")
            op_kraw = Op(bd1, P1_KV, 128, 64, "mn")
            ID_T1 = idesc(128, 16, ta=1)
            bdI1 = K.alloc_local([1], "uint64")
            K.cuda.tcgen05.encode_matrix_descriptor(
                K.address_of(bdI1[0]), s_ident.ptr_to([0]), ldo=8, sdo=16, swizzle=0)
            st_kv1 = K.PipelineState(2, phase=0)
            p1m = K.local_scalar("int32", init=K.int32(0))
            ID_M128N64 = idesc(128, 64)

            def kv_transpose():
                mphase("hmw-kv")
                p_kv.full.wait(st_kv1.stage, st_kv1.phase)
                b_kv_read.wait(0, (p1m & K.int32(1)) ^ K.int32(1))
                K.ptx[TC_FENCE_AFTER]()
                mphase("hm-kvT")
                kv_u = K.local_scalar("uint64", init=K.Cast("uint64", st_kv1.stage) * K.uint64(4 * UNITS_PER_STAGE))
                with K.If(elected()), K.Then():
                    for j in range(4):
                        K.ptx[MMA_SS](
                            K.Cast("uint32", tm[0] + TM_KT + 16 * j),
                            op_kraw.desc(j, kv_u),
                            bdI1[0],
                            K.uint32(ID_T1),
                            K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0),
                            K.ptx.pred(0),
                        )
                    b_kvT_done.arrive(0)
                st_kv1.advance()
                K.assign(p1m, p1m + K.int32(1))
            ID_VN = idesc(128, 64, ta=1, tb=1, nb=1)
            ID_HUPD = idesc(128, 128)
            st_tiles = K.PipelineState(1, phase=0)
            st_akk = K.PipelineState(1, phase=0)
            st_hs = K.PipelineState(1, phase=0)
            st_w = K.PipelineState(1, phase=0)
            st_vn = K.PipelineState(1, phase=0)
            mphase, mphase_end = make_phaser()
            with K.serial(n_p1) as it:
                chain = K.local_scalar("int32", init=p1_chain(it))
                seq, head, bos, seq_len, nch = work_coords(chain)
                K.ptx.ld.volatile.shared.s32(zq1[0], K.address_of(s_tmem[1]))
                K.cuda.tcgen05.encode_matrix_descriptor(
                    K.address_of(bd1[0]), TT[zq1[0]].ptr_to(0, 0), ldo=Op.LBO_BASE, sdo=SBO_UNITS,
                    swizzle=K.SW128B.value)
                kv_transpose()
                with K.serial(nch) as n:
                    K.ptx.ld.volatile.shared.s32(zq1[0], K.address_of(s_tmem[1]))
                    K.cuda.tcgen05.encode_matrix_descriptor(
                        K.address_of(bd1[0]), TT[zq1[0]].ptr_to(0, 0), ldo=Op.LBO_BASE, sdo=SBO_UNITS,
                        swizzle=K.SW128B.value)
                    mphase("hmw-tiles")
                    p_tiles.full.wait(0, st_tiles.phase)
                    p_akk1.full.wait(st_akk.stage, st_akk.phase)
                    K.ptx[TC_FENCE_AFTER]()
                    akk_u = K.local_scalar("uint64", init=K.Cast("uint64", st_akk.stage) * K.uint64(UNITS_PER_STAGE))
                    mphase("hm-WU")
                    with K.If(elected()), K.Then():

                        mma_chain(tm, TM_W, op_kbg_k, op_akk1_k, ID_M128N64, False, b_units=akk_u)
                        p_w.full.arrive(0)

                        mma_chain(tm, TM_U, op_vb_k, op_akk1_k, ID_M128N64, False, b_units=akk_u)
                        p_akk1.empty.arrive(st_akk.stage)
                    st_akk.advance()
                    with K.If(n + K.int32(1) < nch), K.Then():
                        kv_transpose()
                    mphase("hmw-wT")
                    p_w.empty.wait(0, st_w.phase)
                    st_w.advance()
                    mphase("hmw-hs")
                    p_hs.full.wait(st_hs.stage, st_hs.phase)
                    K.ptx[TC_FENCE_AFTER]()
                    hs_u = K.local_scalar("uint64", init=K.Cast("uint64", st_hs.stage) * K.uint64(4 * UNITS_PER_STAGE))
                    mphase("hm-Vn")
                    with K.If(elected()), K.Then():

                        mma_chain(tm, TM_U, op_hs_mn, op_w_mn, ID_VN, True, a_units=hs_u)
                        p_vn.full.arrive(0)
                    st_hs.advance()
                    mphase("hmw-vnT")
                    p_vn.empty.wait(0, st_vn.phase)
                    st_vn.advance()
                    K.ptx[TC_FENCE_AFTER]()
                    mphase("hm-hupd")
                    with K.If(elected()), K.Then():

                        mma_chain(tm, TM_H, op_kg_k, op_vb_k, ID_HUPD, True)
                        p_tiles.empty.arrive(0)
                    st_tiles.advance()
                    mphase_end()

        def p1_loader():
            st_kv = K.PipelineState(2, phase=1)
            st_akk = K.PipelineState(1, phase=1)
            st_g = K.PipelineState(2, phase=1)
            with K.serial(n_p1) as it:
                chain = K.local_scalar("int32", init=p1_chain(it))
                seq, head, bos, seq_len, nch = work_coords(chain)
                bos32 = K.local_scalar("int32", init=K.Cast("int32", bos))
                head8 = K.local_scalar("int32", init=head >> K.int32(3))
                with K.serial(nch) as n:
                    tok0 = bos32 + n * K.int32(CHUNK)
                    p_g.empty.wait(st_g.stage, st_g.phase)
                    with K.If(elected()), K.Then():
                        p_g.full.arrive(st_g.stage, tx_count=G_BYTES)
                        mbg = K.cuda.cvta_generic_to_shared(p_g.full.ptr_to([st_g.stage]))
                        for j in range(4):
                            K.ptx[TMA_LD](TT[P1_G + st_g.stage * K.int32(4) + K.int32(j)].ptr_to(0, 0),
                                          K.address_of(g_map), K.int32(32 * j), tok0, head, mbg)
                        K.ptx[TMA_LD](s_beta_g.ptr_to([st_g.stage, 0, 0]), K.address_of(beta_map),
                                      K.int32(0), tok0, head8, mbg)
                    st_g.advance()
                    p_kv.empty.wait(st_kv.stage, st_kv.phase)
                    with K.If(elected()), K.Then():
                        p_kv.full.arrive(st_kv.stage, tx_count=KV_BYTES)
                        mb = K.cuda.cvta_generic_to_shared(p_kv.full.ptr_to([st_kv.stage]))
                        for tmap, half in ((k_map, 0), (v_map, 1)):
                            for d0 in (0, 64):
                                K.ptx[TMA_LD](TT[P1_KV + st_kv.stage * K.int32(4) + K.int32((d0 // 64) * 2 + half)].ptr_to(0, 0),
                                              K.address_of(tmap), K.int32(d0), tok0, head, mb)
                        with K.If(n + K.int32(1) < nch), K.Then():
                            for tmap in (k_map, v_map):
                                for d0 in (0, 64):
                                    K.ptx[TMA_PREFETCH](K.address_of(tmap), K.int32(d0), tok0 + K.int32(CHUNK), head)
                            for d0 in (0, 32, 64, 96):
                                K.ptx[TMA_PREFETCH](K.address_of(g_map), K.int32(d0), tok0 + K.int32(CHUNK), head)
                            K.ptx[TMA_PREFETCH](K.address_of(akk_map), K.int32(0), tok0 + K.int32(CHUNK), head)
                    st_kv.advance()
                    p_akk1.empty.wait(st_akk.stage, st_akk.phase)
                    with K.If(elected()), K.Then():
                        p_akk1.full.arrive(st_akk.stage, tx_count=AQK_BYTES)
                        mb2 = K.cuda.cvta_generic_to_shared(p_akk1.full.ptr_to([st_akk.stage]))
                        K.ptx[TMA_LD](TT[P1_AKK + st_akk.stage].ptr_to(0, 0), K.address_of(akk_map), K.int32(0), tok0, head, mb2)
                    st_akk.advance()

        def p1_storer():
            st_hs = K.PipelineState(1, phase=0)
            with K.serial(n_p1) as it:
                chain = K.local_scalar("int32", init=p1_chain(it))
                seq, head, bos, seq_len, nch = work_coords(chain)
                cb = chunk_base(seq)
                with K.serial(nch) as n:
                    p_hs.full.wait(st_hs.stage, st_hs.phase)
                    with K.If(elected()), K.Then():
                        K.ptx[FENCE_ASYNC]()
                        idx = (cb + n) * K.int32(H) + head
                        for d0 in (0, 64):
                            K.ptx[TMA_ST](K.address_of(h_map), K.int32(d0), K.int32(0), idx,
                                          TT[P1_HS + st_hs.stage * K.int32(4) + K.int32((d0 // 64) * 2)].ptr_to(0, 0))
                        K.ptx[BULK_COMMIT]()
                        K.ptx[BULK_WAIT_READ](0)
                        p_hs.empty.arrive(st_hs.stage)
                    st_hs.advance()



                with K.If(elected()), K.Then():
                    K.ptx[BULK_WAIT](0)
                    K.ptx["fence.proxy.async.global"]()
                    K.ptx["st.release.gpu.global.s32"](flags.ptr_to([chain]), epoch)




        with cg:
            p1_compute()
            K.ptx.bar.sync(K.uint32(5), K.uint32(384))
            tm = tmem_preamble()
            wr = K.warp_id_in_role()
            lane = K.lane_id()
            wg = K.local_scalar("int32", init=wr >> 2)
            quad = K.local_scalar("int32", init=wr & 3)
            x = K.local_scalar("int32", init=quad * 32 + lane)
            row0 = K.local_scalar("int32", init=wg * 32)
            x64 = K.Cast("int64", x)
            xs = K.local_scalar("int32", init=x >> 6)
            xr = K.local_scalar("int32", init=x & 63)
            tid_all = K.local_scalar("int32", init=wr * 32 + lane)
            phalf = K.local_scalar("int32", init=x & 1)
            pcol = K.local_scalar("int32", init=x & ~1)
            prow0 = K.local_scalar("int32", init=row0 + phalf * 16)
            ps = K.local_scalar("int32", init=pcol >> 6)
            pr = K.local_scalar("int32", init=pcol & 63)
            is_odd = phalf != K.int32(0)
            cyc = K.local_scalar("int32", init=K.int32(0))

            def tmem_at(col):
                return K.Cast("uint32", tm[0] + col + (quad << 21))

            def ld32(regs, col, base=0):
                K.ptx[TC_LD32](*(regs[base + i] for i in range(32)), tmem_at(col))

            def ld8(regs, col, base=0):
                K.ptx[TC_LD8](*(regs[base + i] for i in range(8)), tmem_at(col))

            def ld4(regs, col, base=0):
                K.ptx[TC_LD4](*(regs[base + i] for i in range(4)), tmem_at(col))

            def st_row(stage0, col0, words, wbase=0, nunits=4):
                """Write this thread's row x, columns [col0, col0 + 8*nunits) of the [128][64] tile at stage0/stage0+1."""
                for u in range(nunits):
                    K.ptx["st.shared.v4.b32"](TT[stage0 + xs].ptr_to(xr, col0 + 8 * u),
                                              words[wbase + 4 * u], words[wbase + 4 * u + 1],
                                              words[wbase + 4 * u + 2], words[wbase + 4 * u + 3])

            def st_pair_rows(stage0, words):
                """Pair layout: rows pcol and pcol+1, columns [prow0, prow0+16): words[0:8] row pcol, words[8:16] row pcol+1."""
                for r in range(2):
                    for u in range(2):
                        K.ptx["st.shared.v4.b32"](TT[stage0 + ps].ptr_to(pr + r, prow0 + 8 * u),
                                                  words[8 * r + 4 * u], words[8 * r + 4 * u + 1],
                                                  words[8 * r + 4 * u + 2], words[8 * r + 4 * u + 3])

            def bar_all():
                K.ptx.bar.sync(K.uint32(1), K.uint32(256))

            def bar_wg():
                K.ptx.bar.sync(K.uint32(2) + K.Cast("uint32", wg), K.uint32(128))

            def twait(nm):
                TC[nm].wait(0, cyc & K.int32(1))
                K.ptx[TC_FENCE_AFTER]()

                K.ptx[FENCE_ASYNC]()
                K.ptx["bar.warp.sync"](K.uint32(0xFFFFFFFF))

            def marrive(nm):
                K.ptx[TC_FENCE_BEFORE]()
                MB[nm].arrive(0)

            def lo(w):
                return K.reinterpret("float32", w << K.uint32(16))

            def hi(w):
                return K.reinterpret("float32", w & K.uint32(0xFFFF0000))

            def shfl_xor1(val):
                r = K.local_scalar("uint32")
                K.ptx.shfl_sync.bfly.b32(r, K.reinterpret("uint32", val), K.uint32(1), K.uint32(0x1F),
                                         K.uint32(0xFFFFFFFF))
                return K.reinterpret("float32", r)


            def q_ptr(c, col):
                return TT[ST_Q + (col >> 6)].ptr_to(c, col & 63)

            def k_ptr(c, col):
                return TT[ST_K + (col >> 6)].ptr_to(c, col & 63)

            def v_ptr(c, col):
                return TT[ST_V + (col >> 6)].ptr_to(c, col & 63)

            def e_ptr(c, col):
                return TT[ST_G + (col >> 6)].ptr_to(c, col & 63)

            def load_transpose_frag(base, frag):
                """Load this warp's 32x32 block as four groups of four 8x8 fragments."""
                col0 = (quad & K.int32(1)) * K.int32(32)
                tile = TT[base + xs]
                for rb in range(2):
                    for cb in range(2):
                        o = 4 * (2 * rb + cb)
                        K.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](
                            frag[o], frag[o + 1], frag[o + 2], frag[o + 3],
                            tile.m8n8x4(row0 + K.int32(16 * rb), col0 + K.int32(16 * cb), lane),
                        )

            def store_transpose_frag(base, frag):
                """Transpose those fragments in place, turning [token,channel] into [channel,token]."""
                col0 = (quad & K.int32(1)) * K.int32(32)
                tile = TT[base + xs]
                mm = lane >> K.int32(3)
                jj = lane & K.int32(7)
                for rb in range(2):
                    for cb in range(2):
                        o = 4 * (2 * rb + cb)


                        ptr = tile.ptr_to(
                            col0 + K.int32(16 * cb) + (mm >> K.int32(1)) * K.int32(8) + jj,
                            row0 + K.int32(16 * rb) + (mm & K.int32(1)) * K.int32(8),
                        )
                        K.ptx["stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"](
                            ptr, frag[o], frag[o + 1], frag[o + 2], frag[o + 3]
                        )


            enA = K.alloc_local([16], "float32")
            enB = K.alloc_local([16], "float32")
            egcw = K.alloc_local([16], "uint32")
            t4 = K.alloc_local([4], "float32")
            acc = K.alloc_local([64], "float32")
            wds = K.alloc_local([32], "uint32")
            dgv = K.alloc_local([32], "float32")
            gn = K.local_scalar("float32")
            egn = K.local_scalar("float32")
            dgk = K.local_scalar("float32")
            dgk_k = K.local_scalar("float32")
            t0 = K.local_scalar("float32")
            t1 = K.local_scalar("float32")
            u16 = K.local_scalar("uint16")
            u16b = K.local_scalar("uint16")

            def ex2(dst, val):
                K.ptx.ex2.approx.ftz.f32(dst, val)

            def rcp(dst, val):
                K.ptx.rcp.approx.ftz.f32(dst, val)

            def load_u16_pair(words, i, ptr):
                if i % 2 == 0:
                    K.ptx.ld.global_.nc.u16(u16, ptr)
                else:
                    K.ptx.ld.global_.nc.u16(u16b, ptr)
                    K.ptx.mov.b32(words[i >> 1], u16, u16b)

            def load_u16_pair_sh(words, i, ptr):
                if i % 2 == 0:
                    K.ptx.ld.shared.u16(u16, ptr)
                else:
                    K.ptx.ld.shared.u16(u16b, ptr)
                    K.ptx.mov.b32(words[i >> 1], u16, u16b)

            def shfl_xor1_u32(val):
                r = K.local_scalar("uint32")
                K.ptx.shfl_sync.bfly.b32(r, val, K.uint32(1), K.uint32(0x1F), K.uint32(0xFFFFFFFF))
                return r

            def gcol_ptr(tensor, i):
                """Global pointer to row (row0+i) of this thread's column (clamped to the last valid row)."""
                tokc = tok0 + K.Cast("int64", K.min(row0 + K.int32(i), rows - K.int32(1)))
                return tensor.ptr_to([tokc * HK64 + gcol])

            def s_beta_row(c):
                b = K.local_scalar("float32")
                K.ptx.ld.shared.f32(b, K.address_of(s_beta[c]))
                return b

            def wsel(cond, a, b):
                return K.Select(cond, a, b)

            phase, phase_end = make_phaser()

            with K.serial(n_p2) as i2:
                work = K.local_scalar("int32", init=p2_chain(i2))
                seq, head, bos, seq_len, nch = work_coords(work)
                head64 = K.Cast("int64", head)
                gcol = K.local_scalar("int64", init=head64 * K.int64(D) + x64)
                with K.serial(nch) as rn:
                    n = nch - K.int32(1) - rn
                    par = cyc & K.int32(1)



                    rows = K.int32(CHUNK)
                    last = K.int32(CHUNK - 1)
                    tok0 = K.local_scalar("int64", init=bos + K.Cast("int64", n * K.int32(CHUNK)))
                    x_base = K.local_scalar("int64", init=(tok0 + K.Cast("int64", row0)) * HK64 + gcol)

                    phase("w-in")
                    b_in_full.wait(0, par)
                    b_eg_full.wait(0, par)
                    phase("w-xT")
                    twait("xT_done")
                    phase("c0")
                    with K.If(lane < K.int32(8)), K.Then():
                        btok = wr * K.int32(8) + lane
                        K.ptx.ld.shared.u16(u16, s_beta_in.ptr_to([btok, head & K.int32(7)]))
                        K.ptx.st.shared.f32(K.address_of(s_beta[btok]), lo(K.Cast("uint32", u16)))



                    egf = K.alloc_local([32], "float32")
                    xf = K.alloc_local([32], "float32")
                    qw = K.alloc_local([16], "uint32")
                    kw = K.alloc_local([16], "uint32")
                    t3w = K.alloc_local([16], "uint32")
                    qc = K.alloc_local([16], "uint32")
                    kc = K.alloc_local([16], "uint32")
                    vc = K.alloc_local([16], "uint32")
                    prep0 = K.local_scalar("uint64")
                    prep1 = K.local_scalar("uint64")
                    scale_pair = K.local_scalar("uint64", init=K.cuda.make_float2(scale, scale))
                    bpair = K.alloc_local([2], "float32")


                    ld32(xf, S2 + wg * 32)
                    K.ptx[WAIT_LD]()
                    bar_all()
                    for half in range(2):
                        vb32 = K.alloc_local([16], "float32")
                        for p in range(8):
                            i = 16 * half + 2 * p
                            K.ptx["ld.shared.v2.f32"](bpair[0], bpair[1], K.address_of(s_beta[row0 + i]))
                            K.assign(vb32[2 * p], xf[i] * bpair[0])
                            K.assign(vb32[2 * p + 1], xf[i + 1] * bpair[1])
                            pack_bf16x2(vc[i >> 1], xf[i], xf[i + 1])
                        K.ptx[TC_ST16](tmem_at(S2 + wg * 32 + 16 * half), *(vb32[j] for j in range(16)))
                    K.ptx[WAIT_ST]()
                    st_row(ST_V, row0, vc, 0, 4)


                    ld4(t4, S1 + 60)
                    ld32(egf, S1 + wg * 32)
                    ld32(xf, S3 + wg * 32)
                    K.ptx[WAIT_LD]()
                    K.assign(egn, t4[3])
                    for i in range(16):
                        K.ptx["mul.rn.f32x2"](prep0, K.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                                              K.cuda.make_float2(egf[2 * i], egf[2 * i + 1]))
                        K.ptx["mul.rn.f32x2"](prep0, prep0, scale_pair)
                        pack_bf16x2(qw[i], K.cuda.float2_x(prep0), K.cuda.float2_y(prep0))
                        pack_bf16x2(qc[i], xf[2 * i], xf[2 * i + 1])
                        pack_bf16x2(egcw[i], egf[2 * i], egf[2 * i + 1])

                    for half in range(2):
                        K.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                            *(xf[16 * half + j] for j in range(16)), tmem_at(S4 + wg * 32 + 16 * half))
                        K.ptx[WAIT_LD]()
                        for pp in range(8):
                            i = 8 * half + pp
                            K.ptx["ld.shared.v2.f32"](bpair[0], bpair[1], K.address_of(s_beta[row0 + 2 * i]))
                            rcp(t0, egf[2 * i])
                            rcp(t1, egf[2 * i + 1])
                            K.ptx["mul.rn.f32x2"](prep0, K.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                                                  K.cuda.make_float2(t0, t1))
                            pack_bf16x2(kw[i], K.cuda.float2_x(prep0), K.cuda.float2_y(prep0))
                            K.ptx["mul.rn.f32x2"](prep1, K.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                                                  K.cuda.make_float2(egf[2 * i], egf[2 * i + 1]))
                            K.ptx["mul.rn.f32x2"](prep1, prep1, K.cuda.make_float2(bpair[0], bpair[1]))
                            pack_bf16x2(t3w[i], K.cuda.float2_x(prep1), K.cuda.float2_y(prep1))
                            pack_bf16x2(kc[i], xf[2 * i], xf[2 * i + 1])

                    phase("w-chunk")
                    TC["chunk_done"].wait(0, par ^ K.int32(1))
                    K.ptx[TC_FENCE_AFTER]()
                    K.ptx[FENCE_ASYNC]()
                    phase("c1")
                    st_row(T1, row0, qw, 0, 4)
                    st_row(T2, row0, kw, 0, 4)
                    st_row(T3, row0, t3w, 0, 4)
                    K.ptx[FENCE_ASYNC]()
                    marrive("t_early")
                    phase("c1c")

                    phase("w-h")
                    b_h_full.wait(0, par)
                    K.ptx[TC_FENCE_AFTER]()
                    phase("c2")
                    dgk2 = K.local_scalar("uint64", init=K.cuda.make_float2(K.float32(0.0), K.float32(0.0)))
                    hst = K.local_scalar("int32", init=wg * 2 + xs)
                    dbase = ((K.Cast("int64", seq) * K.int64(H) + head64) * K.int64(D) + x64) * K.int64(D) \
                        + K.Cast("int64", wg * 64)
                    with K.If(rn == K.int32(0)):
                        with K.Then():
                            for m in range(8):
                                K.ptx["ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.f32"](
                                    *(acc[8 * m + i] for i in range(8)),
                                    dht.ptr_to([dbase + K.int64(8 * m)]))
                        with K.Else():
                            ld32(acc, TM_DH + wg * 64)
                            ld32(acc, TM_DH + wg * 64 + 32, 32)
                            K.ptx[WAIT_LD]()
                    for half in range(2):
                        hc = wg * 64 + 32 * half
                        a0 = 32 * half


                        for p in range(16):
                            dpair = K.local_scalar("uint64")
                            K.ptx["mul.rn.f32x2"](dpair, K.cuda.make_float2(acc[a0 + 2 * p], acc[a0 + 2 * p + 1]),
                                                  K.cuda.make_float2(egn, egn))
                            K.assign(acc[a0 + 2 * p], K.cuda.float2_x(dpair))
                            K.assign(acc[a0 + 2 * p + 1], K.cuda.float2_y(dpair))
                        for u in range(4):
                            K.ptx["ld.shared.v4.b32"](wds[0], wds[1], wds[2], wds[3],
                                                      TT[S_H + hst].ptr_to(xr, 32 * half + 8 * u))
                            for p in range(4):
                                K.ptx["fma.rn.f32x2"](
                                    dgk2,
                                    K.cuda.make_float2(lo(wds[p]), hi(wds[p])),
                                    K.cuda.make_float2(acc[a0 + 8 * u + 2 * p], acc[a0 + 8 * u + 2 * p + 1]),
                                    dgk2,
                                )
                        for p in range(16):
                            pack_bf16x2(wds[p], acc[a0 + 2 * p], acc[a0 + 2 * p + 1])
                        for u in range(4):
                            K.ptx["st.shared.v4.b32"](TT[DHB + hst].ptr_to(xr, 32 * half + 8 * u),
                                                      wds[4 * u], wds[4 * u + 1], wds[4 * u + 2], wds[4 * u + 3])
                        K.ptx[TC_ST32](tmem_at(TM_DH + hc), *(acc[a0 + i] for i in range(32)))
                    K.assign(dgk, K.cuda.float2_x(dgk2) + K.cuda.float2_y(dgk2))
                    K.ptx[WAIT_ST]()
                    K.ptx[FENCE_ASYNC]()
                    marrive("dhb_ready")


                    def readout_to_tile(slot, stage0):
                        ld32(acc, slot + wg * 32)
                        K.ptx[WAIT_LD]()
                        for p in range(16):
                            pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                        st_row(stage0, row0, wds, 0, 4)
                        K.ptx[FENCE_ASYNC]()

                    phase("w-Z")
                    twait("Z_done")
                    phase("c3")
                    readout_to_tile(S2, ZT)
                    marrive("zT_ready")
                    phase("w-dv2")
                    twait("dv2_done")
                    phase("c5")
                    readout_to_tile(S3, DV2)
                    marrive("dv2T_ready")
                    phase("w-Vn")
                    twait("Vn_done")
                    phase("c4")
                    readout_to_tile(S2, T6)
                    marrive("vnT_ready")


                    def readout64(slot, stage, mask, scale_by=None, negate=False):
                        ld32(acc, slot + wg * 32)
                        K.ptx[WAIT_LD]()
                        cc = quad * 16 + lane
                        with K.If(lane < K.int32(16)), K.Then():
                            for p in range(16):
                                vv2 = []
                                for e in range(2):
                                    jj = row0 + 2 * p + e
                                    val = acc[2 * p + e]
                                    if scale_by is not None:
                                        val = val * scale_by
                                    if negate:
                                        val = K.float32(0.0) - val
                                    vv2.append(val if mask is None else K.Select(mask(cc, jj), val, K.float32(0.0)))
                                pack_bf16x2(wds[p], vv2[0], vv2[1])
                            for u in range(4):
                                K.ptx["st.shared.v4.b32"](TT[stage].ptr_to(cc, row0 + 8 * u),
                                                          wds[4 * u], wds[4 * u + 1], wds[4 * u + 2], wds[4 * u + 3])
                        K.ptx[FENCE_ASYNC]()

                    def readout64_half(slot, stage, mask, negate=False):
                        K.ptx[TC_LD_HALF32](*(acc[i] for i in range(16)), tmem_at(slot + wg * 32))
                        K.ptx[WAIT_LD]()
                        cc0 = quad * 16 + (lane >> K.int32(2))
                        cc1 = cc0 + K.int32(8)
                        for rep in range(4):
                            jj0 = row0 + K.int32(8 * rep) + (lane & K.int32(3)) * K.int32(2)
                            jj1 = jj0 + K.int32(1)
                            v00 = acc[4 * rep]
                            v01 = acc[4 * rep + 1]
                            v10 = acc[4 * rep + 2]
                            v11 = acc[4 * rep + 3]
                            if negate:
                                v00 = K.float32(0.0) - v00
                                v01 = K.float32(0.0) - v01
                                v10 = K.float32(0.0) - v10
                                v11 = K.float32(0.0) - v11
                            if mask is not None:
                                v00 = K.Select(mask(cc0, jj0), v00, K.float32(0.0))
                                v01 = K.Select(mask(cc0, jj1), v01, K.float32(0.0))
                                v10 = K.Select(mask(cc1, jj0), v10, K.float32(0.0))
                                v11 = K.Select(mask(cc1, jj1), v11, K.float32(0.0))
                            pack_bf16x2(wds[2 * rep], v00, v01)
                            pack_bf16x2(wds[2 * rep + 1], v10, v11)
                        tile = TT[stage]
                        for half in range(2):
                            K.ptx["stmatrix.sync.aligned.m8n8.x4.shared.b16"](
                                tile.m8n8x4(quad * K.int32(16), row0 + K.int32(16 * half), lane),
                                wds[4 * half], wds[4 * half + 1], wds[4 * half + 2], wds[4 * half + 3],
                            )
                        K.ptx[FENCE_ASYNC]()

                    phase("w-dAs")
                    twait("dAs_done")
                    phase("c8")

                    readout64_half(S4, DAM, lambda cc, jj: jj < cc)
                    marrive("dAm_ready")
                    phase("w-dAqk")
                    twait("dAqk_done")
                    phase("c7")

                    readout64_half(S1, T5, lambda cc, jj: jj <= cc)
                    marrive("dAqk_tile_ready")
                    twait("dk_done")
                    phase("w-dvb")
                    twait("dvb_done")
                    phase("passA")

                    pbx = K.local_scalar("int32", init=K.int32(PB0) + wg * K.int32(PB1 - PB0))

                    def pass_a(full):
                        assert full
                        pa_acc = K.local_scalar("uint64")
                        pa_v = K.local_scalar("uint64")
                        pa_db = K.local_scalar("uint64")
                        pa_dv = K.local_scalar("uint64")
                        pa_word = K.local_scalar("uint32")
                        ld8(acc, S3 + wg * 32, 0)
                        K.ptx[WAIT_LD]()
                        for b in range(4):
                            ab = 8 * (b % 2)
                            if b < 3:
                                ld8(acc, S3 + wg * 32 + 8 * (b + 1), 8 * ((b + 1) % 2))
                            vq = K.alloc_local([4], "uint32")
                            K.ptx["ld.shared.v4.b32"](
                                vq[0], vq[1], vq[2], vq[3],
                                TT[ST_V + xs].ptr_to(xr, row0 + 8 * b),
                            )
                            dbp = K.alloc_local([8], "float32")
                            for p in range(4):
                                i = 8 * b + 2 * p
                                K.assign(pa_acc, K.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]))
                                K.assign(pa_v, K.cuda.make_float2(lo(vq[p]), hi(vq[p])))
                                K.ptx["mul.rn.f32x2"](pa_db, pa_acc, pa_v)
                                K.assign(dbp[2 * p], K.cuda.float2_x(pa_db))
                                K.assign(dbp[2 * p + 1], K.cuda.float2_y(pa_db))
                                K.ptx["ld.shared.v2.f32"](
                                    t4[0], t4[1], K.address_of(s_beta[row0 + i])
                                )
                                K.ptx["mul.rn.f32x2"](
                                    pa_dv, pa_acc, K.cuda.make_float2(t4[0], t4[1])
                                )
                                pack_bf16x2(pa_word, K.cuda.float2_x(pa_dv), K.cuda.float2_y(pa_dv))
                                K.ptx["st.global.L1::no_allocate.b16"](
                                    dv.ptr_to([x_base + K.int64(i * HK)]), K.Cast("uint16", pa_word)
                                )
                                K.ptx["st.global.L1::no_allocate.b16"](
                                    dv.ptr_to([x_base + K.int64((i + 1) * HK)]),
                                    K.Cast("uint16", pa_word >> K.uint32(16)),
                                )




                            for e in range(8):
                                i = 8 * b + e
                                K.ptx.st.shared.f32(TT[pbx + (i >> 4)].ptr_to(4 * (i & 15) + quad, 2 * lane), dbp[e])

                            dvw = K.alloc_local([4], "uint32")
                            for p in range(4):
                                pack_bf16x2(dvw[p], acc[ab + 2 * p], acc[ab + 2 * p + 1])
                            K.ptx["st.shared.v4.b32"](TT[DVB + xs].ptr_to(xr, row0 + 8 * b),
                                                      dvw[0], dvw[1], dvw[2], dvw[3])
                            if b < 3:
                                K.ptx[WAIT_LD]()

                    pass_a(True)
                    K.ptx[FENCE_ASYNC]()
                    marrive("dv_epi_done")
                    phase("w-X")
                    twait("X_done")
                    phase("c9")
                    readout64_half(S1, T6, None)
                    marrive("X_ready")
                    phase("dbv")
                    bar_wg()
                    tq = lane & K.int32(3)
                    ti = quad * 8 + (lane >> 2)
                    srow = (quad & K.int32(1)) * 32 + lane
                    dsum_v = K.local_scalar("float32", init=K.float32(0.0))
                    for u in range(8):
                        K.ptx["ld.shared.v4.f32"](t4[0], t4[1], t4[2], t4[3], TT[pbx + (quad >> 1)].ptr_to(srow, 8 * u))
                        K.assign(dsum_v, dsum_v + ((t4[0] + t4[1]) + (t4[2] + t4[3])))

                    K.ptx[FENCE_ASYNC]()
                    b_mid_free.arrive(0)
                    phase("w-Y")
                    twait("Y_done")
                    phase("c10")
                    readout64_half(S2, T5 + 1, lambda cc, jj: jj < cc, negate=True)
                    marrive("intra_ready")


                    phase("w-epi")
                    twait("dq2_done")
                    phase("epi")
                    dgk_k2 = K.local_scalar(
                        "uint64", init=K.cuda.make_float2(K.float32(0.0), K.float32(0.0))
                    )

                    def q_loads(b, base):
                        ld8(acc, S4 + wg * 32 + 8 * b, base)

                    def k_loads(b, base):
                        ld4(acc, S5 + wg * 32 + 4 * b, base + 0)
                        ld4(acc, S6 + wg * 32 + 4 * b, base + 4)
                        ld4(acc, S3 + wg * 32 + 4 * b, base + 8)

                    def epilogue(full):
                        assert full
                        pair0 = K.local_scalar("uint64")
                        pair1 = K.local_scalar("uint64")
                        pair2 = K.local_scalar("uint64")
                        pair3 = K.local_scalar("uint64")
                        pair4 = K.local_scalar("uint64")
                        pair5 = K.local_scalar("uint64")

                        q_loads(0, 0)
                        K.ptx[WAIT_LD]()
                        for b in range(4):
                            ab = 8 * (b % 2)
                            if b < 3:
                                q_loads(b + 1, 8 * ((b + 1) % 2))
                            for p in range(4):
                                i = 8 * b + 2 * p



                                rcp(enA[i >> 1], lo(egcw[i >> 1]))
                                rcp(enB[i >> 1], hi(egcw[i >> 1]))
                                K.ptx["mul.rn.f32x2"](
                                    pair1,
                                    K.cuda.make_float2(lo(egcw[i >> 1]), hi(egcw[i >> 1])),
                                    K.cuda.make_float2(scale, scale),
                                )
                                K.ptx["mul.rn.f32x2"](
                                    pair0,
                                    K.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    pair1,
                                )
                                K.ptx["st.global.L1::no_allocate.f32"](
                                    dq.ptr_to([x_base + K.int64(i * HK)]), K.cuda.float2_x(pair0))
                                K.ptx["st.global.L1::no_allocate.f32"](
                                    dq.ptr_to([x_base + K.int64((i + 1) * HK)]), K.cuda.float2_y(pair0))
                                K.ptx["mul.rn.f32x2"](pair1, K.cuda.make_float2(lo(qc[i >> 1]), hi(qc[i >> 1])), pair0)
                                K.assign(dgv[i], K.cuda.float2_x(pair1))
                                K.assign(dgv[i + 1], K.cuda.float2_y(pair1))
                            if b < 3:
                                K.ptx[WAIT_LD]()
                        twait("dkt_done")


                        dbx = 2 * wg
                        k_loads(0, 0)
                        K.ptx[WAIT_LD]()
                        for b in range(8):
                            ab = 12 * (b % 2)
                            if b < 7:
                                k_loads(b + 1, 12 * ((b + 1) % 2))
                            for p in range(2):
                                i = 4 * b + 2 * p
                                K.assign(pair0, K.cuda.make_float2(enA[i >> 1], enB[i >> 1]))
                                K.assign(pair1, K.cuda.make_float2(lo(kc[i >> 1]), hi(kc[i >> 1])))
                                K.ptx["add.rn.f32x2"](
                                    pair2,
                                    K.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    K.cuda.make_float2(acc[ab + 8 + 2 * p], acc[ab + 8 + 2 * p + 1]),
                                )
                                K.ptx["mul.rn.f32x2"](pair2, pair2, pair0)
                                K.ptx["mul.rn.f32x2"](
                                    pair3,
                                    K.cuda.make_float2(acc[ab + 4 + 2 * p], acc[ab + 4 + 2 * p + 1]),
                                    K.cuda.make_float2(lo(egcw[i >> 1]), hi(egcw[i >> 1])),
                                )
                                K.ptx["mul.rn.f32x2"](pair4, pair1, pair3)
                                K.ptx.st.shared.f32(
                                    TT[dbx + (i >> 4)].ptr_to(4 * (i & 15) + quad, 2 * lane), K.cuda.float2_x(pair4))
                                K.ptx.st.shared.f32(
                                    TT[dbx + ((i + 1) >> 4)].ptr_to(4 * ((i + 1) & 15) + quad, 2 * lane),
                                    K.cuda.float2_y(pair4))
                                K.ptx["mul.rn.f32x2"](
                                    pair4,
                                    K.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    pair0,
                                )
                                K.ptx["mul.rn.f32x2"](pair5, pair1, pair4)
                                K.ptx["add.rn.f32x2"](dgk_k2, dgk_k2, pair5)
                                beta_pair = K.cuda.make_float2(
                                    s_beta_row(row0 + i), s_beta_row(row0 + i + 1)
                                )
                                K.ptx["fma.rn.f32x2"](pair5, pair3, beta_pair, pair2)
                                K.ptx["st.global.L1::no_allocate.f32"](
                                    dk.ptr_to([x_base + K.int64(i * HK)]), K.cuda.float2_x(pair5))
                                K.ptx["st.global.L1::no_allocate.f32"](
                                    dk.ptr_to([x_base + K.int64((i + 1) * HK)]), K.cuda.float2_y(pair5))
                                K.ptx["fma.rn.f32x2"](
                                    pair3,
                                    pair2,
                                    K.cuda.make_float2(K.float32(-2.0), K.float32(-2.0)),
                                    pair5,
                                )
                                K.ptx["fma.rn.f32x2"](
                                    pair5, pair1, pair3, K.cuda.make_float2(dgv[i], dgv[i + 1]),
                                )
                                K.assign(dgv[i], K.cuda.float2_x(pair5))
                                K.assign(dgv[i + 1], K.cuda.float2_y(pair5))
                            if b < 7:
                                K.ptx[WAIT_LD]()

                        K.assign(dgk_k, K.cuda.float2_x(dgk_k2) + K.cuda.float2_y(dgk_k2))
                        bar_wg()
                        dsum = K.local_scalar("float32", init=dsum_v)
                        for u in range(8):
                            K.ptx["ld.shared.v4.f32"](t4[0], t4[1], t4[2], t4[3], TT[dbx + (quad >> 1)].ptr_to(srow, 8 * u))
                            K.assign(dsum, dsum + ((t4[0] + t4[1]) + (t4[2] + t4[3])))
                        K.ptx[FENCE_ASYNC]()
                        b_h_free.arrive(0)
                        for s in (1, 2):
                            r = K.local_scalar("uint32")
                            K.ptx.shfl_sync.bfly.b32(r, K.reinterpret("uint32", dsum), K.uint32(s), K.uint32(0x1F), K.uint32(0xFFFFFFFF))
                            K.assign(dsum, dsum + K.reinterpret("float32", r))
                        with K.If(tq == K.int32(0)), K.Then():
                            K.ptx["st.global.L1::no_allocate.f32"](
                                db.ptr_to([(tok0 + K.Cast("int64", row0 + ti)) * K.int64(H) + head64]), dsum)

                    epilogue(True)
                    phase("cumsum")


                    for i in range(30, -1, -1):
                        K.assign(dgv[i], dgv[i] + dgv[i + 1])
                    K.ptx.st.shared.f32(K.address_of(s_dgk[wg, x]),
                                        dgk + dgk_k + K.Select(wg == K.int32(0), K.float32(0.0), dgv[0]))
                    b_dg0_ready.arrive(0)
                    b_dg0_ready.wait(0, cyc & K.int32(1))
                    K.ptx.ld.shared.f32(t0, K.address_of(s_dgk[K.int32(1) - wg, x]))
                    K.assign(t1, t0 + dgk + dgk_k)
                    for i in range(32):
                        K.assign(dgv[i], dgv[i] + t1)
                    for i in range(32):
                        K.ptx["st.global.L1::no_allocate.f32"](
                            dg.ptr_to([x_base + K.int64(i * HK)]), dgv[i])
                    phase_end()
                    K.assign(cyc, cyc + K.int32(1))

                phase("dh0")
                TC["chunk_done"].wait(0, (cyc & K.int32(1)) ^ K.int32(1))
                K.ptx[TC_FENCE_AFTER]()
                ld32(acc, TM_DH + wg * 64)
                ld32(acc, TM_DH + wg * 64 + 32, 32)
                K.ptx[WAIT_LD]()
                obase = ((K.Cast("int64", seq) * K.int64(H) + head64) * K.int64(D) + x64) * K.int64(D) \
                    + K.Cast("int64", wg * 64)
                for m in range(8):
                    K.ptx["st.global.L1::no_allocate.v8.f32"](
                        dh0.ptr_to([obase + K.int64(8 * m)]),
                        *(acc[8 * m + i] for i in range(8)))
                phase_end()

        with auxg:



            with mma:
                p1_mma()
                K.ptx.bar.sync(K.uint32(5), K.uint32(384))
                tm = tmem_preamble()
                cyc = K.local_scalar("int32", init=K.int32(0))

                def mwait(nm):
                    MB[nm].wait(0, cyc & K.int32(1))
                    K.ptx[TC_FENCE_AFTER]()

                mphase, mphase_end = make_phaser()

                bd = K.alloc_local([1], "uint64")
                zq = K.alloc_local([1], "int32")
                op_T1k = Op(bd, T1, 128, 64, "k")
                op_T1mn = Op(bd, T1, 128, 128, "mn")
                op_T2k = Op(bd, T2, 128, 64, "k")
                op_T2mn = Op(bd, T2, 128, 128, "mn")
                op_T3k = Op(bd, T3, 128, 64, "k")
                op_T3mn = Op(bd, T3, 128, 128, "mn")
                op_T5k = Op(bd, T5, 128, 64, "k")
                op_ZTk = Op(bd, ZT, 128, 64, "k")
                op_ZTmn = Op(bd, ZT, 128, 128, "mn")
                op_dAqk_k = Op(bd, T5, 64, 64, "k")
                op_dAkk_k = Op(bd, T5 + 1, 64, 64, "k")
                op_T6mn = Op(bd, T6, 128, 128, "mn")
                op_dAqk_mn = Op(bd, T5, 64, 64, "mn")
                op_dAm_k = Op(bd, DAM, 64, 64, "k")
                op_X_mn = Op(bd, T6, 64, 64, "mn")
                op_dAkk_mn = Op(bd, T5 + 1, 64, 64, "mn")
                op_DHBk = Op(bd, DHB, 128, 128, "k")
                op_DHBmn = Op(bd, DHB, 128, 128, "mn")
                op_DV2k = Op(bd, DV2, 128, 64, "k")
                op_DV2mn = Op(bd, DV2, 128, 128, "mn")
                op_DVBk = Op(bd, DVB, 128, 64, "k")
                op_DVBmn = Op(bd, DVB, 128, 128, "mn")
                op_do_k128 = Op(bd, S_DO, 64, 128, "k")
                op_do_mn64 = Op(bd, S_DO, 64, 64, "mn")
                op_h_k = Op(bd, S_H, 128, 128, "k")
                op_h_mn = Op(bd, S_H, 128, 128, "mn")
                op_aqk_mn = Op(bd, S_AQK, 64, 64, "mn")
                op_akk_k = Op(bd, S_AKK, 64, 64, "k")
                op_akk_mn = Op(bd, S_AKK, 64, 64, "mn")
                ID_128x64 = idesc(128, 64)
                ID_128x64_TATB_NB = idesc(128, 64, ta=1, tb=1, nb=1)
                ID_128x64_TATB = idesc(128, 64, ta=1, tb=1)
                ID_128x64_TB = idesc(128, 64, tb=1)
                ID_128x64_TB_NA = idesc(128, 64, tb=1, na=1)
                ID_128x128_TB = idesc(128, 128, tb=1)
                ID_128x128_NB = idesc(128, 128, nb=1)
                ID_128x128 = idesc(128, 128)
                ID_64x64_TB = idesc(64, 64, tb=1)
                ID_64x64_TATB = idesc(64, 64, ta=1, tb=1)
                ID_64x64 = idesc(64, 64)


                op_egT = Op(bd, ST_G, 64, 64, "mn")
                op_vT = Op(bd, ST_V, 64, 64, "mn")
                op_qT = Op(bd, ST_Q, 64, 64, "mn")
                op_kT = Op(bd, ST_K, 64, 64, "mn")
                ID_T = idesc(128, 16, ta=1)
                bdI = K.alloc_local([1], "uint64")
                K.cuda.tcgen05.encode_matrix_descriptor(
                    K.address_of(bdI[0]), s_ident.ptr_to([0]), ldo=8, sdo=16, swizzle=0)

                with K.serial(n_p2) as i2:
                    work = K.local_scalar("int32", init=p2_chain(i2))
                    seq, head, bos, seq_len, nch = work_coords(work)
                    with K.serial(nch) as rn:
                        par = cyc & K.int32(1)
                        K.ptx.ld.volatile.shared.s32(zq[0], K.address_of(s_tmem[1]))
                        K.cuda.tcgen05.encode_matrix_descriptor(
                            K.address_of(bd[0]), TT[zq[0]].ptr_to(0, 0), ldo=Op.LBO_BASE, sdo=SBO_UNITS,
                            swizzle=K.SW128B.value)
                        akk_u = K.local_scalar("uint64", init=K.Cast("uint64", par) * K.uint64(UNITS_PER_STAGE))
                        mphase("mw-xT")
                        b_in_full.wait(0, par)
                        b_eg_full.wait(0, par)


                        b_h_free.wait(0, par ^ K.int32(1))
                        K.ptx[TC_FENCE_AFTER]()
                        mphase("m-xT")
                        with K.If(elected()), K.Then():
                            for src, dst in ((op_egT, S1), (op_vT, S2), (op_qT, S3), (op_kT, S4)):
                                for j in range(4):
                                    K.ptx[MMA_SS](
                                        K.Cast("uint32", tm[0] + dst + 16 * j),
                                        src.desc(j),
                                        bdI[0],
                                        K.uint32(ID_T),
                                        K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0),
                                        K.ptx.pred(0),
                                    )
                            TC["xT_done"].arrive(0)
                        mphase("mw-early")
                        mwait("t_early")
                        b_akk_full.wait(par, (cyc >> 1) & K.int32(1))
                        b_h_full.wait(0, par)
                        K.ptx[TC_FENCE_AFTER]()
                        mphase("m-Z")
                        with K.If(elected()), K.Then():

                            mma_chain(tm, S2, op_h_mn, op_T3mn, ID_128x64_TATB_NB, True)
                            TC["Z_done"].arrive(0)
                        mphase("mw-aqk")
                        b_aqk_masked.wait(0, par)
                        b_do_full.wait(0, par)
                        K.ptx[TC_FENCE_AFTER]()
                        mphase("m-dvp")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S3, op_do_mn64, op_aqk_mn, ID_128x64_TATB, False)
                            b_aqk_empty.arrive(0)
                        mphase("mw-dhb")
                        mwait("dhb_ready")
                        mphase("m-dv2")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S3, op_DHBmn, op_T2mn, ID_128x64_TATB, True)
                            TC["dv2_done"].arrive(0)
                        mphase("mw-zT")
                        mwait("zT_ready")
                        mphase("m-Vn")
                        with K.If(elected()), K.Then():

                            mma_chain(tm, S2, op_ZTk, op_akk_k, ID_128x64, False, b_units=akk_u)
                            TC["Vn_done"].arrive(0)
                        mphase("mw-dv2T")
                        mwait("dv2T_ready")
                        mphase("m-dAs")
                        with K.If(elected()), K.Then():

                            mma_chain(tm, S4, op_DV2mn, op_ZTmn, ID_64x64_TATB, False)
                            TC["dAs_done"].arrive(0)
                            mma_chain(tm, S3, op_DV2k, op_akk_mn, ID_128x64_TB, False, b_units=akk_u)
                            TC["dvb_done"].arrive(0)
                        mphase("mw-vnT")
                        mwait("vnT_ready")
                        mphase("m-dAqk")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S1, op_do_k128, op_T6mn, ID_64x64_TB, False)
                            TC["dAqk_done"].arrive(0)
                            mma_chain(tm, S5, op_DHBk, op_T6mn, ID_128x64_TB, False)
                            TC["dk_done"].arrive(0)
                        mphase("mw-dAm")
                        mwait("dAm_ready")
                        mwait("dAqk_tile_ready")
                        mphase("m-X")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S1, op_dAm_k, op_akk_k, ID_64x64, False, b_units=akk_u)
                            TC["X_done"].arrive(0)


                            mma_chain(tm, S4, op_h_k, op_do_k128, ID_128x64, False)
                            mma_chain(tm, S4, op_T2k, op_dAqk_k, ID_128x64, True)
                            TC["dq2_done"].arrive(0)
                            mma_chain(tm, TM_DH, op_T1k, op_do_mn64, ID_128x128_TB, True)
                            b_do_empty.arrive(0)
                        mphase("mw-dvepi")
                        mwait("dv_epi_done")
                        mphase("m-dwb")
                        with K.If(elected()), K.Then():

                            mma_chain(tm, S6, op_h_k, op_DVBmn, ID_128x64_TB_NA, False)
                        mphase("mw-X")
                        mwait("X_ready")
                        mphase("m-Y")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S2, op_akk_mn, op_X_mn, ID_64x64_TATB, False, a_units=akk_u)
                            TC["Y_done"].arrive(0)
                            b_akk_empty.arrive(par)
                        mphase("mw-intra")
                        mwait("intra_ready")
                        mphase("m-dk2")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S6, op_T2k, op_dAkk_k, ID_128x64, True)
                        mphase("m-dkt")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S3, op_T1k, op_dAqk_mn, ID_128x64_TB, False)
                            mma_chain(tm, S3, op_T3k, op_dAkk_mn, ID_128x64_TB, True)
                            TC["dkt_done"].arrive(0)

                            mma_chain(tm, TM_DH, op_T3k, op_DVBk, ID_128x128_NB, True)
                            TC["chunk_done"].arrive(0)
                        mphase_end()
                        K.assign(cyc, cyc + K.int32(1))




            with loader:
                with K.If(elected()), K.Then():
                    for m in (q_map, k_map, v_map, g_map, eg_map, beta_map, do_map, aqk_map, akk_map, h_map):
                        K.ptx.prefetch.tensormap(K.address_of(m))
                p1_loader()
                K.ptx.bar.sync(K.uint32(5), K.uint32(384))
                cyc = K.local_scalar("int32", init=K.int32(0))
                lphase, lphase_end = make_phaser()
                with K.serial(n_p2) as i2:
                    work = K.local_scalar("int32", init=p2_chain(i2))
                    seq, head, bos, seq_len, nch = work_coords(work)
                    bos32 = K.local_scalar("int32", init=K.Cast("int32", bos))
                    cb = chunk_base(seq)
                    head8 = K.local_scalar("int32", init=head >> K.int32(3))


                    lphase("lw-flag")
                    with K.If(elected()), K.Then():
                        fl = K.local_scalar("int32", init=K.int32(0))
                        with K.While(fl != epoch):
                            K.ptx.ld.acquire.gpu.global_.s32(fl, flags.ptr_to([work]))
                    K.ptx["bar.warp.sync"](K.uint32(0xFFFFFFFF))
                    K.ptx["fence.proxy.async.global"]()
                    lphase_end()
                    with K.serial(nch) as rn:
                        n = nch - K.int32(1) - rn
                        par = cyc & K.int32(1)
                        npar = par ^ K.int32(1)
                        tok0 = bos32 + n * K.int32(CHUNK)
                        hidx = (cb + n) * K.int32(H) + head

                        lphase("lw-mid")
                        b_mid_free.wait(0, npar)
                        lphase("l-issue")
                        with K.If(elected()), K.Then():
                            b_in_full.arrive(0, tx_count=IN_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(b_in_full.ptr_to([0]))
                            K.ptx[TMA_LD](s_beta_in.ptr_to([0, 0]), K.address_of(beta_map), K.int32(0), tok0, head8, mb)
                            for d0 in (0, 64):
                                K.ptx[TMA_LD](TT[ST_Q + d0 // 64].ptr_to(0, 0), K.address_of(q_map), K.int32(d0), tok0, head, mb)
                                K.ptx[TMA_LD](TT[ST_K + d0 // 64].ptr_to(0, 0), K.address_of(k_map), K.int32(d0), tok0, head, mb)
                                K.ptx[TMA_LD](TT[ST_V + d0 // 64].ptr_to(0, 0), K.address_of(v_map), K.int32(d0), tok0, head, mb)
                        lphase("lw-chunk")
                        TC["chunk_done"].wait(0, npar)
                        lphase("l-issue-eg")
                        with K.If(elected()), K.Then():
                            b_eg_full.arrive(0, tx_count=EG_BYTES)
                            mbe = K.cuda.cvta_generic_to_shared(b_eg_full.ptr_to([0]))
                            for d0 in (0, 64):
                                K.ptx[TMA_LD](TT[ST_G + d0 // 64].ptr_to(0, 0), K.address_of(eg_map), K.int32(d0), tok0, head, mbe)
                        b_do_empty.wait(0, npar)
                        with K.If(elected()), K.Then():
                            b_do_full.arrive(0, tx_count=DO_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(b_do_full.ptr_to([0]))
                            for d0 in (0, 64):
                                K.ptx[TMA_LD](TT[S_DO + d0 // 64].ptr_to(0, 0), K.address_of(do_map), K.int32(d0), tok0, head, mb)
                        b_h_free.wait(0, npar)
                        with K.If(elected()), K.Then():
                            b_h_full.arrive(0, tx_count=H_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(b_h_full.ptr_to([0]))
                            for d0 in (0, 64):
                                K.ptx[TMA_LD](TT[S_H + (d0 // 64) * 2].ptr_to(0, 0), K.address_of(h_map), K.int32(d0), K.int32(0), hidx, mb)
                        b_aqk_empty.wait(0, npar)
                        with K.If(elected()), K.Then():
                            b_aqk_full.arrive(0, tx_count=AQK_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(b_aqk_full.ptr_to([0]))
                            K.ptx[TMA_LD](TT[S_AQK].ptr_to(0, 0), K.address_of(aqk_map), K.int32(0), tok0, head, mb)
                        b_akk_empty.wait(par, ((cyc >> 1) & K.int32(1)) ^ K.int32(1))
                        with K.If(elected()), K.Then():
                            b_akk_full.arrive(par, tx_count=AQK_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(b_akk_full.ptr_to([par]))
                            K.ptx[TMA_LD](TT[S_AKK + par].ptr_to(0, 0), K.address_of(akk_map), K.int32(0), tok0, head, mb)
                            with K.If(n == K.int32(0)), K.Then():
                                with K.If(i2 + K.int32(1) < n_p2), K.Then():
                                    nxt = p2_chain(i2 + K.int32(1))
                                    seq2, head2, bos2, seq_len2, nch2 = work_coords(nxt)
                                    tokn = K.Cast("int32", bos2) + (nch2 - K.int32(1)) * K.int32(CHUNK)
                                    hidn = (chunk_base(seq2) + nch2 - K.int32(1)) * K.int32(H) + head2
                                    for tmap in (q_map, k_map, v_map, do_map, eg_map):
                                        for d0 in (0, 64):
                                            K.ptx[TMA_PREFETCH](K.address_of(tmap), K.int32(d0), tokn, head2)
                                    K.ptx[TMA_PREFETCH](K.address_of(aqk_map), K.int32(0), tokn, head2)
                                    K.ptx[TMA_PREFETCH](K.address_of(akk_map), K.int32(0), tokn, head2)
                                    for d0 in (0, 64):
                                        K.ptx[TMA_PREFETCH](K.address_of(h_map), K.int32(d0), K.int32(0), hidn)
                            with K.If(n > K.int32(0)), K.Then():
                                tokp = tok0 - K.int32(CHUNK)
                                for tmap in (q_map, k_map, v_map, do_map):
                                    for d0 in (0, 64):
                                        K.ptx[TMA_PREFETCH](K.address_of(tmap), K.int32(d0), tokp, head)
                                for d0 in (0, 64):
                                    K.ptx[TMA_PREFETCH](K.address_of(eg_map), K.int32(d0), tokp, head)
                                K.ptx[TMA_PREFETCH](K.address_of(aqk_map), K.int32(0), tokp, head)
                                K.ptx[TMA_PREFETCH](K.address_of(akk_map), K.int32(0), tokp, head)
                                for d0 in (0, 64):
                                    K.ptx[TMA_PREFETCH](K.address_of(h_map), K.int32(d0), K.int32(0), hidx - K.int32(H))
                        lphase_end()
                        K.assign(cyc, cyc + K.int32(1))





            with idle:
                with K.If(K.warp_id_in_role() == K.int32(0)), K.Then():
                    p1_storer()
                K.ptx.bar.sync(K.uint32(5), K.uint32(384))
                cyc = K.local_scalar("int32", init=K.int32(0))
                rowc = K.local_scalar("int32", init=K.warp_id_in_role() * K.int32(32) + K.lane_id())
                with K.serial(n_p2) as i2:
                    work = K.local_scalar("int32", init=p2_chain(i2))
                    seq, head, bos, seq_len, nch = work_coords(work)
                    with K.serial(nch) as rn:
                        par = cyc & K.int32(1)
                        b_aqk_full.wait(0, par)




                        diag = K.alloc_local([4], "uint32")
                        dmat = K.lane_id() >> K.int32(3)
                        dblk = K.warp_id_in_role() * K.int32(4) + dmat
                        dptr = TT[S_AQK].ptr_to(
                            dblk * K.int32(8) + (K.lane_id() & K.int32(7)),
                            dblk * K.int32(8),
                        )
                        K.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](
                            diag[0], diag[1], diag[2], diag[3], dptr
                        )
                        drow = K.lane_id() >> K.int32(2)
                        dcol = (K.lane_id() & K.int32(3)) * K.int32(2)
                        dmask = K.Select(
                            dcol > drow,
                            K.uint32(0),
                            K.Select(dcol == drow, K.uint32(0x0000FFFF), K.uint32(0xFFFFFFFF)),
                        )
                        for e in range(4):
                            K.assign(diag[e], diag[e] & dmask)
                        K.ptx["stmatrix.sync.aligned.m8n8.x4.shared.b16"](
                            dptr, diag[0], diag[1], diag[2], diag[3]
                        )
                        for u in range(1, 8):
                            with K.If(K.int32(8 * u) > rowc), K.Then():
                                K.ptx["st.shared.v4.b32"](TT[S_AQK].ptr_to(rowc, 8 * u),
                                                          K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0))
                        K.ptx[FENCE_ASYNC]()
                        b_aqk_masked.arrive(0)
                        K.assign(cyc, cyc + K.int32(1))

        K.cuda.cta_sync()
        with K.If(K.warp_id() == 8), K.Then():
            K.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
            K.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                K.Cast("uint32", K.local_scalar("int32", init=tmem_preamble()[0])), K.uint32(512))

    return kda_bwd_fused

SCHEDULE_CLASSES = [(40, 6, 2), (80, 5, 5), (32, 4, 9)]


def build_schedule(num_chains, num_ctas, classes):
    """Per-CTA (pass-2 chains, pass-1 chains) lists.

    classes == "legacy": base = num_chains // num_ctas; the `extra` lowest CTAs own base+1 pass-2 chains
    (strided) and run pass 1 for their first chain only; the remaining CTAs absorb the other heavy
    chains' pass-1 work first, then their own.  Otherwise classes is a list of (count, n_p2, n_p1):
    pass-2 chains are dealt in rounds; each CTA runs pass 1 for its own first n_p1 chains and, if it has
    spare pass-1 capacity, for other CTAs' later chains first (so those are published early).
    """
    if classes == "legacy":
        base = num_chains // num_ctas
        extra = num_chains - base * num_ctas
        p2 = [[c + r * num_ctas for r in range(base + (1 if c < extra else 0))] for c in range(num_ctas)]
        n_light = num_ctas - extra
        extra1 = max(extra, 1)
        heavy_items = [(t % extra1) + (t // extra1 + 1) * num_ctas for t in range(extra * base)]
        p1 = []
        for c in range(num_ctas):
            if c < extra:
                p1.append([c])
            else:
                lidx = c - extra
                p1.append([heavy_items[t] for t in range(lidx, extra * base, n_light)]
                          + [c + r * num_ctas for r in range(base)])
        return p2, p1
    cta_p2, cta_p1 = [], []
    for count, a, b in classes:
        cta_p2 += [a] * count
        cta_p1 += [b] * count
    assert len(cta_p2) == num_ctas and sum(cta_p2) == num_chains and sum(cta_p1) == num_chains, \
        (len(cta_p2), sum(cta_p2), sum(cta_p1))
    p2 = [[] for _ in range(num_ctas)]
    chain = 0
    for r in range(max(cta_p2)):
        for c in range(num_ctas):
            if cta_p2[c] > r:
                p2[c].append(chain)
                chain += 1
    leftover = []
    p1 = []
    for c in range(num_ctas):
        take = min(cta_p1[c], len(p2[c]))
        p1.append(p2[c][:take])
        leftover += p2[c][take:]
    li = 0
    for c in range(num_ctas):
        spare = cta_p1[c] - len(p1[c])
        if spare > 0:
            p1[c] = leftover[li:li + spare] + p1[c]
            li += spare
    assert li == len(leftover), (li, len(leftover))
    return p2, p1


def build_schedule_tensor(num_chains, num_ctas, classes, dev):
    """Per-CTA [n_p2, n_p1, p2 chains..., p1 chains...] table plus its two list capacities."""
    p2, p1 = build_schedule(num_chains, num_ctas, classes)
    assert sorted(c for l in p2 for c in l) == list(range(num_chains))
    assert sorted(c for l in p1 for c in l) == list(range(num_chains))
    maxp2 = max(1, max(len(l) for l in p2))
    maxp1 = max(1, max(len(l) for l in p1))
    stride = 2 + maxp2 + maxp1
    table = torch.full((num_ctas, stride), -1, dtype=torch.int32)
    for c in range(num_ctas):
        table[c, 0] = len(p2[c])
        table[c, 1] = len(p1[c])
        for i, ch in enumerate(p2[c]):
            table[c, 2 + i] = ch
        for i, ch in enumerate(p1[c]):
            table[c, 2 + maxp2 + i] = ch
    return table.reshape(-1).to(dev), maxp2, maxp1






_TUNED_NUM_CTAS = 152
_TUNED_NUM_CHAINS = 768


def build_mega_tables(lens, HQ, HV, num_ctas):
    """Host-side schedule tables for the mega kernel.

    stream_tab: work-unit order of the recurrence streams (backward streams first, longest
    sequences first, value heads ascending; then the forward streams in the same order), packed as
    (is_fwd << 30) | (seq << 15) | hv.
    item_tab: (seq << 16) | chunk for every chunk, sorted by the predicted step at which both of its
    streams have published it.  Streams are assumed to start when a CTA of the initial wave frees
    up (list-scheduled in stream order) and to take one step per chunk; a chunk is published by the
    forward stream after n + 1 steps and by the backward stream after nch - n steps.  A partial
    trailing chunk also waits for the first chunk of the following sequence(s) that share its tile.
    seq_tab: [N, 4] int32 rows (bos, len, nch, chunk_base).
    """
    import heapq

    N = len(lens)
    nchs = [(l + CHUNK - 1) // CHUNK for l in lens]
    cbs, boss = [], []
    cb = bo = 0
    for s in range(N):
        cbs.append(cb)
        boss.append(bo)
        cb += nchs[s]
        bo += lens[s]
    order = sorted(range(N), key=lambda s: (-lens[s], s))
    rank = {s: r for r, s in enumerate(order)}
    streams = [(0, s, hv) for s in order for hv in range(HV)] + [(1, s, hv) for s in order for hv in range(HV)]
    free = [0] * max(1, min(num_ctas, len(streams)))
    heapq.heapify(free)
    start = {}
    for key in streams:
        t0 = heapq.heappop(free)
        start[key] = t0
        heapq.heappush(free, t0 + nchs[key[1]])
    fwd_start = [max(start[(1, s, hv)] for hv in range(HV)) for s in range(N)]
    bwd_start = [max(start[(0, s, hv)] for hv in range(HV)) for s in range(N)]
    items = []
    for s in range(N):
        for n in range(nchs[s]):
            r = max(fwd_start[s] + n + 1, bwd_start[s] + nchs[s] - n)
            if n == nchs[s] - 1 and lens[s] % CHUNK != 0:
                tok0 = boss[s] + n * CHUNK
                b2 = boss[s] + lens[s]
                s2 = s + 1
                while s2 < N and b2 < tok0 + CHUNK:
                    r = max(r, fwd_start[s2] + 1)
                    b2 += lens[s2]
                    s2 += 1
            items.append((r, rank[s], n, s))
    items.sort()
    assert N < (1 << 15) and max(nchs) < (1 << 15) and HV < (1 << 15)
    stream_tab = torch.tensor([(d << 30) | (s << 15) | hv for (d, s, hv) in streams], dtype=torch.int32)
    item_tab = torch.tensor([(s << 16) | n for (_, _, n, s) in items], dtype=torch.int32)
    seq_tab = torch.tensor([[boss[s], lens[s], nchs[s], cbs[s]] for s in range(N)], dtype=torch.int32).reshape(-1)
    return stream_tab, item_tab, seq_tab






_KERNEL_CACHE = {}
_DEBUG = {}


def _target():
    import tvm

    cap = torch.cuda.get_device_capability()
    return tvm.target.Target({"kind": "cuda", "arch": f"sm_{cap[0]}{cap[1]}a"})


def _compile(kind, *key_args):
    key = (kind, *key_args)
    if key not in _KERNEL_CACHE:
        import tvm

        kernel = {"mega": make_mega_kernel, "fused": make_fused_kernel}[kind](*key_args)
        target = _target()
        with target:
            _KERNEL_CACHE[key] = tvm.compile(kernel.mod, target=target, tir_pipeline="tirx")
    return _KERNEL_CACHE[key]


def build_kernels_for_shape(H, num_chains=768, num_ctas=152, HV=None):
    """Trace-only entry for offline tooling."""
    HV = H if HV is None else HV
    out = {"kda_bwd_mega": make_mega_kernel(H, HV)}
    if H == HV and H % 8 == 0:
        classes = SCHEDULE_CLASSES if (num_ctas == _TUNED_NUM_CTAS and num_chains == _TUNED_NUM_CHAINS) else "legacy"
        p2, p1 = build_schedule(num_chains, num_ctas, classes)
        out["kda_bwd_fused"] = make_fused_kernel(H, max(1, max(len(l) for l in p2)), max(1, max(len(l) for l in p1)))
    return out


def _count_chunks(cu_seqlens):
    """Total 64-token chunks over the packed sequences (a launch-extent fact, like FLA's chunk_indices)."""
    lens = torch.diff(cu_seqlens.cpu())
    return int(((lens + CHUNK - 1) // CHUNK).sum()), bool(((lens % CHUNK) == 0).all())


def setup(data, B, T, H):
    q, k, v, beta = data["q"], data["k"], data["v"], data["beta"]
    Aqk, Akk, g = data["Aqk"], data["Akk"], data["g"]
    h0, do, dht = data["initial_state"], data["do"], data["dht"]
    scale = float(data["scale"])
    chunk_size = int(data["chunk_size"])
    cu_seqlens = data["cu_seqlens"]
    dq, dk, dv, db, dg, dh0 = (data[n] for n in ("dq", "dk", "dv", "db", "dg", "dh0"))
    device = q.device
    HQ = H
    HV = v.shape[2]
    N = h0.shape[0]
    if B != 1 or chunk_size != CHUNK or q.shape[-1] != D or v.shape[-1] != D or HV % HQ != 0:
        raise ValueError("unsupported KDA backward shape")
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, T], dtype=torch.int64, device=device)
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int64).contiguous()
    for name in ("q", "k", "v", "beta", "Aqk", "Akk", "g", "initial_state", "do", "dht",
                 "dq", "dk", "dv", "db", "dg", "dh0"):
        if not data[name].is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    total_chunks, full_chunks = _count_chunks(cu_seqlens)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    num_chunks_max = (T + CHUNK - 1) // CHUNK + N
    hsnap = torch.empty((num_chunks_max, HV, D, D), dtype=torch.bfloat16, device=device)
    egcache = torch.empty((1, T, HV, D), dtype=torch.bfloat16, device=device)
    stream_counter = torch.zeros((4,), dtype=torch.int32, device=device)
    maps = {
        "q": token_map(q, T, HQ, D, 64),
        "k": token_map(k, T, HQ, D, 64),
        "v": token_map(v, T, HV, D, 64),
        "g": token_map(g, T, HV, D, 32),
        "do": token_map(do, T, HV, D, 64),
        "eg": token_map(egcache, T, HV, D, 64),
        "aqk": token_map(Aqk, T, HV, CHUNK, CHUNK),
        "akk": token_map(Akk, T, HV, CHUNK, CHUNK),
        "h": state_map(hsnap, num_chunks_max * HV),
    }
    _DEBUG.clear()

    if HQ == HV and HV % 8 == 0 and full_chunks:

        num_chains = N * HV
        num_ctas = min(num_sms, num_chains)
        classes = SCHEDULE_CLASSES if (num_ctas == _TUNED_NUM_CTAS and num_chains == _TUNED_NUM_CHAINS) else "legacy"
        sched, maxp2, maxp1 = build_schedule_tensor(num_chains, num_ctas, classes, device)
        flags = torch.zeros((num_chains,), dtype=torch.int32, device=device)
        maps["beta"] = token_map(beta, T, HV // 8, 8, 8, swizzle=0)
        fused = _compile("fused", HV, maxp2, maxp1)
        args = (
            q.view(-1), k.view(-1), v.view(-1), beta.view(-1), Aqk.view(-1), Akk.view(-1), g.view(-1),
            egcache.view(-1), do.view(-1), dht.view(-1), h0.view(-1), hsnap.view(-1), cu_seqlens, flags,
            sched, dq.view(-1), dk.view(-1), dv.view(-1), db.view(-1), dg.view(-1), dh0.view(-1),
            maps["q"].ptr, maps["k"].ptr, maps["v"].ptr, maps["g"].ptr, maps["eg"].ptr, maps["beta"].ptr,
            maps["do"].ptr, maps["aqk"].ptr, maps["akk"].ptr, maps["h"].ptr,
            scale, N, num_ctas,
        )
        state = {"epoch": 0}

        def run():
            state["epoch"] += 1
            fused(*args, state["epoch"])

        run._keep_alive = (args, maps, hsnap, egcache, flags, sched, cu_seqlens)
        _DEBUG.update(family="fused", h=hsnap)
    else:

        dhsnap = torch.empty((num_chunks_max, HV, D, D), dtype=torch.bfloat16, device=device)
        maps["dh"] = state_map(dhsnap, num_chunks_max * HV)
        num_chains = N * HV
        flags = torch.zeros((2 * num_chains,), dtype=torch.int64, device=device)
        num_items = total_chunks * HQ
        num_ctas = min(num_sms, 2 * num_chains + num_items)
        lens = torch.diff(cu_seqlens.cpu()).tolist()
        stream_tab, item_tab, seq_tab = build_mega_tables(lens, HQ, HV, num_ctas)
        assert item_tab.numel() == total_chunks
        stream_tab = stream_tab.to(device)
        item_tab = item_tab.to(device)
        seq_tab = seq_tab.to(device)
        mega = _compile("mega", HQ, HV)
        args = (
            q.view(-1), k.view(-1), v.view(-1), beta.view(-1), Aqk.view(-1), Akk.view(-1), g.view(-1),
            egcache.view(-1), do.view(-1), dht.view(-1), h0.view(-1), hsnap.view(-1), dhsnap.view(-1),
            cu_seqlens, dq.view(-1), dk.view(-1), dv.view(-1), db.view(-1), dg.view(-1), dh0.view(-1),
            stream_counter, flags, stream_tab, item_tab, seq_tab,
            maps["q"].ptr, maps["k"].ptr, maps["v"].ptr, maps["g"].ptr, maps["eg"].ptr, maps["do"].ptr,
            maps["aqk"].ptr, maps["akk"].ptr, maps["h"].ptr, maps["dh"].ptr,
            scale, N, num_items, num_ctas,
        )
        state = {"epoch": 0}

        def run():
            state["epoch"] += 1
            mega(*args, state["epoch"])

        run._keep_alive = (args, maps, hsnap, dhsnap, egcache, cu_seqlens, stream_counter, flags,
                           stream_tab, item_tab, seq_tab)
        _DEBUG.update(family="mega", h=hsnap, dhb=dhsnap)

    run()
    torch.cuda.synchronize(device)
    return run


def debug_state():
    """Kernel-owned intermediates for the remote test harness."""
    return {k: v for k, v in _DEBUG.items() if k in ("h", "dhb")}


# ----------------------------------------------------------------------------
# Public kernel identity and supported configurations
# ----------------------------------------------------------------------------

KERNEL_META = {
    "name": "agent_evolved_kda_backward_packed",
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
        "run": "kda-bwd-portfolio-carry-3",
        "selected_version": "fused-zfold",
    },
}

# The seventeen official workloads: (label, total tokens, Hqk, Hv, sequence lengths).
_OFFICIAL_WORKLOADS = (
    ("packed_1024x8_h96", 8192, 96, 96, (1024,) * 8),
    ("p01_hq4_hv8_t32768", 32768, 4, 8, (3200,) * 9 + (3968,)),
    ("p02_hq2_hv8_t18432", 18432, 2, 8, (2000,) * 8 + (2432,)),
    ("p03_hq4_hv4_t32768", 32768, 4, 4, (2656,) * 11 + (3552,)),
    ("p04_hq2_hv4_t18432", 18432, 2, 4, (1648,) * 10 + (1952,)),
    ("p05_hq4_hv8_t18432", 18432, 4, 8, (2000,) * 8 + (2432,)),
    ("p06_hq2_hv8_t32768", 32768, 2, 8, (3200,) * 9 + (3968,)),
    ("p07_hq2_hv4_t18432", 18432, 2, 4, (2000,) * 8 + (2432,)),
    ("p08_hq4_hv4_t18432", 18432, 4, 4, (1648,) * 10 + (1952,)),
    ("p09_hq2_hv8_t32768", 32768, 2, 8, (2656,) * 11 + (3552,)),
    ("p10_hq4_hv8_t18432", 18432, 4, 8, (1648,) * 10 + (1952,)),
    ("p11_hq4_hv4_t32768", 32768, 4, 4, (3200,) * 9 + (3968,)),
    ("p12_hq2_hv4_t32768", 32768, 2, 4, (2656,) * 11 + (3552,)),
    ("p13_hq4_hv4_t18432", 18432, 4, 4, (2000,) * 8 + (2432,)),
    ("p14_hq2_hv4_t32768", 32768, 2, 4, (3200,) * 9 + (3968,)),
    ("p15_hq4_hv8_t32768", 32768, 4, 8, (2656,) * 11 + (3552,)),
    ("p16_hq2_hv8_t18432", 18432, 2, 8, (1648,) * 10 + (1952,)),
)

_SUPPORTED = {
    label: (total, hq, hv, lens) for label, total, hq, hv, lens in _OFFICIAL_WORKLOADS
}


@dataclass(frozen=True, slots=True)
class KDABackwardConfig:
    label: str
    num_qk_heads: int
    num_v_heads: int
    seq_lens: tuple[int, ...]
    seed: int = 0
    scale: float = 1.0 / math.sqrt(D)

    def validate(self) -> None:
        if self.num_v_heads % self.num_qk_heads != 0:
            raise ValueError(
                f"Hv must be a multiple of Hqk, got {self.num_v_heads} % {self.num_qk_heads}"
            )
        if not self.seq_lens or any(length <= 0 for length in self.seq_lens):
            raise ValueError(f"invalid packed sequence layout {self.seq_lens}")
        if not math.isclose(self.scale, 1.0 / math.sqrt(D), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"scale must be 1/sqrt({D}), got {self.scale}")

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
    def uses_fused_path(self) -> bool:
        """Mirror ``setup``'s dispatch rule without touching CUDA."""
        whole_chunks = all(length % CHUNK == 0 for length in self.seq_lens)
        return self.num_qk_heads == self.num_v_heads and self.num_v_heads % 8 == 0 and whole_chunks


CONFIGS = [
    {
        "label": label,
        "num_qk_heads": hq,
        "num_v_heads": hv,
        "seq_lens": lens,
        "seed": 2858210400 + index,
    }
    for index, (label, _total, hq, hv, lens) in enumerate(_OFFICIAL_WORKLOADS)
]


def _cfg(**kwargs: Any) -> KDABackwardConfig:
    names = {field.name for field in fields(KDABackwardConfig)}
    values = {name: value for name, value in kwargs.items() if name in names}
    if "seq_lens" in values:
        values["seq_lens"] = tuple(int(length) for length in values["seq_lens"])
    values.setdefault("label", "custom")
    cfg = KDABackwardConfig(**values)
    cfg.validate()
    return cfg


def get_kernel(**kwargs: Any):
    """The PrimFunc this configuration actually launches."""
    cfg = _cfg(**kwargs)
    if cfg.uses_fused_path:
        from tirx_kernels.runner import hardware_num_sms

        num_chains = cfg.num_seqs * cfg.num_v_heads
        num_ctas = min(hardware_num_sms(), num_chains)
        classes = (
            SCHEDULE_CLASSES
            if (num_ctas == _TUNED_NUM_CTAS and num_chains == _TUNED_NUM_CHAINS)
            else "legacy"
        )
        p2, p1 = build_schedule(num_chains, num_ctas, classes)
        maxp2 = max(1, max(len(entries) for entries in p2))
        maxp1 = max(1, max(len(entries) for entries in p1))
        return make_fused_kernel(cfg.num_v_heads, maxp2, maxp1).func
    return make_mega_kernel(cfg.num_qk_heads, cfg.num_v_heads).func


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------


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


def _l2_normalize_bf16(tensor: torch.Tensor) -> torch.Tensor:
    values = tensor.float()
    return (values * torch.rsqrt(values.square().sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)


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


def _saved_interaction_matrices(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Aqk/Akk exactly as FLA's varlen intra path saves them for the backward."""
    with _native_fla_backend(), torch.no_grad():
        from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra

        outputs = chunk_kda_fwd_intra(
            q=case["q"],
            k=case["k"],
            v=case["v"],
            gk=case["g"],
            beta=case["beta"],
            scale=case["scale"],
            cu_seqlens=case["cu_seqlens"],
            chunk_size=CHUNK,
        )
    return outputs[4], outputs[5]


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    device = torch.device(kwargs.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved KDA backward")

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    total, hq, hv = cfg.total_tokens, cfg.num_qk_heads, cfg.num_v_heads
    qk_shape = (1, total, hq, D)
    v_shape = (1, total, hv, D)
    state_shape = (cfg.num_seqs, hv, D, D)

    q = _l2_normalize_bf16(
        _randn(qk_shape, torch.bfloat16, device=device, generator=generator, scale=0.5)
    )
    k = _l2_normalize_bf16(
        _randn(qk_shape, torch.bfloat16, device=device, generator=generator, scale=0.5)
    )
    v = _randn(v_shape, torch.bfloat16, device=device, generator=generator, scale=0.5)
    beta = torch.sigmoid(
        _randn(
            (1, total, hv), torch.float32, device=device, generator=generator, scale=0.5
        )
    ).to(torch.bfloat16)
    gate_increments = -(
        0.01 + 0.04 * torch.rand(v_shape, dtype=torch.float32, device=device, generator=generator)
    )
    offsets = [0]
    for length in cfg.seq_lens:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device=device)

    with _native_fla_backend():
        from fla.ops.utils import chunk_local_cumsum

        g = chunk_local_cumsum(gate_increments, chunk_size=CHUNK, cu_seqlens=cu_seqlens)

    case: dict[str, Any] = {
        "config": cfg,
        "q": q,
        "k": k,
        "v": v,
        "beta": beta,
        "g": g,
        "scale": cfg.scale,
        "chunk_size": CHUNK,
        "cu_seqlens": cu_seqlens,
        "initial_state": _randn(
            state_shape, torch.float32, device=device, generator=generator, scale=0.01
        ),
        "do": _randn(v_shape, torch.bfloat16, device=device, generator=generator, scale=0.25),
        "dht": _randn(state_shape, torch.float32, device=device, generator=generator, scale=0.01),
    }
    case["Aqk"], case["Akk"] = _saved_interaction_matrices(case)
    case["dq"] = torch.empty(qk_shape, dtype=torch.float32, device=device)
    case["dk"] = torch.empty(qk_shape, dtype=torch.float32, device=device)
    case["dv"] = torch.empty_like(v)
    case["db"] = torch.empty((1, total, hv), dtype=torch.float32, device=device)
    case["dg"] = torch.empty(v_shape, dtype=torch.float32, device=device)
    case["dh0"] = torch.empty(state_shape, dtype=torch.float32, device=device)
    return case


def _launcher(case: dict[str, Any]):
    """Bind the kernel-owned scratch, schedule tables and tensor maps."""
    cfg: KDABackwardConfig = case["config"]
    return setup(case, cfg.batch_size, cfg.total_tokens, cfg.num_qk_heads)


# ----------------------------------------------------------------------------
# Correctness
# ----------------------------------------------------------------------------

_OUTPUT_NAMES = ("dq", "dk", "dv", "db", "dg", "dh0")
# FLA's test_kda.py normalized RMS error-ratio limits, in return order.
_RMS_LIMITS = (8e-3, 8e-3, 8e-3, 2e-2, 2e-2, 8e-3)


def _assert_supported_arch() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved KDA backward")
    capability = torch.cuda.get_device_capability()
    runtime_arch = f"sm_{capability[0]}{capability[1]}a"
    if runtime_arch not in KERNEL_META["runtime_cuda_archs"]:
        raise SkipTest(
            "agent-evolved KDA backward requires one of "
            f"{KERNEL_META['runtime_cuda_archs']}, got {runtime_arch}"
        )


def _run_fla_reference(case: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    with _native_fla_backend(), torch.no_grad():
        from fla.ops.kda.chunk_bwd import chunk_kda_bwd

        dq, dk, dv, db, dg, dh0, dA, dbias = chunk_kda_bwd(
            q=case["q"],
            k=case["k"],
            v=case["v"],
            beta=case["beta"],
            Aqk=case["Aqk"],
            Akk=case["Akk"],
            g=case["g"],
            initial_state=case["initial_state"],
            do=case["do"],
            dht=case["dht"],
            scale=case["scale"],
            chunk_size=CHUNK,
            cu_seqlens=case["cu_seqlens"],
        )
    if dA is not None or dbias is not None:
        raise AssertionError("prepared-gate chunk_kda_bwd unexpectedly returned dA/dbias")
    return dq, dk, dv, db, dg, dh0


def check_correctness(outputs: dict[str, Any], **kwargs: Any) -> None:
    _cfg(**kwargs)
    first, actual, reference = outputs["first"], outputs["actual"], outputs["reference"]
    for name, tensor in zip(_OUTPUT_NAMES, actual):
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} contains non-finite values")
    for name, one, two in zip(_OUTPUT_NAMES, first, actual):
        if not torch.equal(one, two):
            max_abs = float((one.float() - two.float()).abs().max())
            raise AssertionError(
                f"identical launches are not exactly repeatable for {name}; max abs diff={max_abs}"
            )
    for name, got, want, limit in zip(_OUTPUT_NAMES, actual, reference, _RMS_LIMITS):
        torch.testing.assert_close(
            got, want, atol=1e-1, rtol=1e-1, msg=lambda message, n=name: f"{n}: {message}"
        )
        diff_rms = torch.sqrt(torch.mean((got.float() - want.float()).square()))
        reference_rms = torch.sqrt(torch.mean(want.float().square()))
        rms_ratio = float(diff_rms / (reference_rms + 1e-8))
        if rms_ratio >= limit:
            raise AssertionError(
                f"{name} normalized RMS error ratio {rms_ratio:.6e} must be below {limit:.0e}"
            )


def _clone_outputs(case: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    return tuple(case[name].clone() for name in _OUTPUT_NAMES)


def _poison_outputs(case: dict[str, Any], value: float) -> None:
    for name in _OUTPUT_NAMES:
        case[name].fill_(value)


def run_test(**kwargs: Any) -> None:
    _assert_supported_arch()
    case = prepare_data(**kwargs)
    launch = _launcher(case)
    _poison_outputs(case, float("nan"))
    launch()
    torch.cuda.synchronize()
    first = _clone_outputs(case)
    _poison_outputs(case, 42.0)
    launch()
    torch.cuda.synchronize()
    actual = _clone_outputs(case)
    reference = _run_fla_reference(case)
    torch.cuda.synchronize()
    check_correctness({"first": first, "actual": actual, "reference": reference}, **kwargs)


# ----------------------------------------------------------------------------
# Benchmarking
# ----------------------------------------------------------------------------


def prepare_bench(**kwargs: Any):
    """Trace and compile before bench-suite assigns a GPU.

    ``setup`` compiles through this module's own cache, so priming that cache
    here is what keeps the timed path free of compilation.
    """
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    compile_kernel(get_kernel(**kwargs))
    return prepared_gpu_benchmark(run_gpu, {"config": dict(kwargs)})


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
    case = prepare_data(**config)
    launch = _launcher(case)
    torch.cuda.synchronize()

    def _fla_builder():
        return lambda: _run_fla_reference(case)

    from tirx_kernels.runner import bench

    return bench(
        {"tirx": launch},
        references={"fla_chunk_kda_bwd": _fla_builder},
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
