# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved SM100a recurrent Kimi Delta Attention (KDA) decode, every official shape.

This replaces the pinned `kda-decode-d128-t1-b128-h16-hv32` kernel with the
`colreg-clc-ring` frontier member of the 2026-09-13 multi-shape KDA-decode
evolution run. It covers all thirty official rows: tokens per sequence T = 1..6,
sequence counts N = 1..128, H = 16 query/key heads, HV in {16, 32} value heads
and K = V = 128 -- standard single-token decode, speculative decode with
per-token state checkpoints and acceptance semantics, and the T = 3 rows whose
gate is a runtime lower-bound sigmoid rather than a precomputed log-space g.

Four device programs sit behind one shape dispatch, chosen from T and N only:

* **colreg-persist** (T <= 5): a persistent register-resident recurrence. A unit
  is (sequence, value head, column slice of CPT rows); a persistent 4-warp CTA
  walks units and keeps the fp32 state of its CPT rows x 128 k in registers
  across all T tokens, with the register chunk permuted by the k-slice so
  quarter-warp shared accesses on the staging tiles are bank-conflict free.
  Every warp runs its phases autonomously and issues its own `cp.async.bulk`
  checkpoint store; CTA barriers remain only at unit boundaries.
* **deferred publication** (T = 6): the same compute path with checkpoint
  publication deferred, which wins on the longest speculative rows.
* **dynamic ticket scheduler** (T = 4 with N >= 64, T = 2 with N = 128): again
  the same compute path, but every item beyond a CTA's first is taken from a
  global counter two items ahead, so fast CTAs absorb the late second launch
  wave and the per-SM speed variance that leave a 6-12% finish spread under a
  static stride. Paired A/B on the server: T=4 N=128 -6.7..-7.6%, T=4 N=64
  -0.7..-2.3%, T=2 N=128 -0.2..-1.6%; T=5 and T=6 lose 3-6% with the same code
  and stay on the static kernels.
* **CLC unit ring** (T = 1, N in {8, 32, 64, 128}): the grid has one CTA per
  state unit but only the resident CTAs run, each obtaining its next unit by
  cancelling a not-yet-launched CTA with `clusterlaunchcontrol.try_cancel`. The
  response lands asynchronously on an mbarrier, so the next unit's bulk load is
  issued while the current one computes and no L2 atomic sits on the issue path;
  load balancing is done by hardware at unit granularity.

Every threshold is a function of the row's token and sequence counts, never of
input values.

Measured over the thirty official rows on one GB200 through the kcoral
benchmark server, candidate and baseline in the same run: **1.325x geometric
mean, with no row below parity**. The arithmetic is fp32 throughout and the only
approximations are the hardware `rsqrt.approx.ftz.f32` used for the contract's
q/k L2 normalization and `ex2.approx.ftz.f32` for the decay; there is no
polynomial exponential on this path.
"""

from typing import Any
from unittest import SkipTest

import torch

def _make_base():
    """KDA recurrent decode, family "colreg-persist" (v4, warp-autonomous phases): persistent register-resident recurrence.

    Unit = (sequence n, value head hv, column slice of CPT rows). A persistent CTA (4 warps)
    walks units u = cta, cta + num_main, ... . For each unit the fp32 state of its CPT rows x
    128 k stays in registers across all T tokens: lane (ks = lane & 7, cw = lane >> 3) holds
    k-slice [16 ks, 16 ks + 16) of the J = CPT / 16 rows v = warp * CPT/4 + j * 4 + cw, with
    register chunk c holding k-chunk c ^ (ks >> 2) so quarter-warp shared accesses are
    bank-conflict free on the plain row-major staging tiles.

    Pipeline per unit (tile A = input, tiles B/C = checkpoint staging):
      1. wait raw vectors (TMA-prefetched during the previous unit) -> preprocess into the
         per-phase vector table; then issue the raw-vector prefetch of the next unit.
      2. wait the state tile (TMA-prefetched during the previous unit) -> registers; then
         issue the next unit's state load into tile A.
      3. phase 0: S <- alpha_0 S ; d1 += k_0 S ; d2 += q_0 S ; reduce-scatter -> u_0, out_0.
      4. phase p: S <- S + k_{p-1} u_{p-1} (checkpoint p-1 packed to bf16 into a staging tile
         and written with cp.async.bulk), S <- alpha_p S, dots, reduce -> u_p, out_p.
      5. final update + checkpoint T-1.
    Within a unit every warp runs its phases autonomously: its rows are contiguous in the staging
    tile and in the pool slot, so lane 0 of each warp issues the warp's own cp.async.bulk checkpoint
    store after a warp-level sync; CTA barriers remain only at unit boundaries (vector table, tile A).
    Untouched pool slots (the harness poisons final_state before every check) are copied by the
    same persistent CTAs: after its recurrence units a CTA processes tile-sized copy units
    (TMA load into tile A, TMA store out), prefetched exactly like state loads. Each CTA resolves
    the pool slots of its copy units once at start with a warp-parallel ballot scan over the
    touched-slot flags (built in tile B from ssm_state_indices).
    """

    import torch
    import tvm

    import tirx_kernels.kern as K

    D = 128
    LOG2E = 1.4426950408889634
    FULL = 0xFFFFFFFF
    _BULK_G2S = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
    _BULK_S2G = "cp.async.bulk.global.shared::cta.bulk_group"
    FST = 64 * 8 + 8 * 4
    RAW_TOK_B = 4 * 256 + 16
    GATE_B = 512 + 16
    MAXC = 16
    MAXH = 6
    IDX_PRE = 8


    def _shfl_bfly(val_f32, xor):
        out = K.local_scalar("uint32")
        K.ptx.shfl_sync.bfly.b32(out, K.reinterpret("uint32", val_f32), K.uint32(xor), K.uint32(31), K.uint32(FULL))
        return K.reinterpret("float32", out)


    def _shfl_idx(val_f32, src_lane):
        out = K.local_scalar("uint32")
        K.ptx.shfl_sync.idx.b32(out, K.reinterpret("uint32", val_f32), K.cast(src_lane, "uint32"), K.uint32(31), K.uint32(FULL))
        return K.reinterpret("float32", out)


    def _f2(a, b):
        return K.cuda.make_float2(a, b)


    def _rng(name):
        """IKET range token (stripped by the production pipeline)."""
        token = K.alloc_local([1], "uint32")
        K.assign(token[0], K.cuda.iket.range_start(name))
        return token


    def _rng_end(token):
        K.cuda.iket.range_end(token[0])


    def build_kernel(*, T, H, HV, gate_lb, cpt, num_units, num_main, per_sm, copy_est=0, allreduce=None, bar_arrive=False, warp_auto=True, vec_split=True, nw=4, l2pf=False):
        assert D % cpt == 0 and cpt % (4 * nw) == 0, "each warp needs a multiple of 4 rows"
        G = HV // H
        NT = 32 * nw
        rows_per_warp = cpt // nw
        J = rows_per_warp // 4
        M = 2 * J
        R = max(1, M // 8)
        split = D // cpt
        tile_elems = cpt * D
        tile_bytes = tile_elems * 2
        raw_bytes = T * RAW_TOK_B + (GATE_B if gate_lb else 0)
        raw_elems = raw_bytes // 2
        nphase = T + 1
        load_piece = min(tile_bytes, 16384)
        assert tile_bytes % load_piece == 0
        n_pieces = tile_bytes // load_piece
        num_ctas = num_main
        pmax = tile_bytes // 4
        spec = T > 1
        DCU = HV * split
        if allreduce is None:
            allreduce = J <= 2
        if num_main > num_units:
            hole_base0 = num_units
        else:
            hole_base0 = num_units % num_main
        hole_cnt0 = num_main - hole_base0


        if hole_cnt0 == 0 or copy_est > hole_cnt0 * max(MAXH, T + 1):
            hole_base0, hole_cnt0 = 0, num_main

        @K.kernel(warps=nw, arch="sm_100a", grid=num_ctas, min_blocks_per_sm=per_sm)
        def kda_decode_persist(
            q: K.gptr[K.bf16],
            k: K.gptr[K.bf16],
            v: K.gptr[K.bf16],
            g: K.gptr[K.bf16],
            beta: K.gptr[K.bf16],
            a_log: K.gptr[K.f32],
            dt_bias: K.gptr[K.f32],
            state_in: K.gptr[K.bf16],
            state_out: K.gptr[K.bf16],
            output: K.gptr[K.bf16],
            ssm_idx: K.gptr[K.i32],
            num_accepted: K.gptr[K.i32],
            scale: K.f32,
            lower_bound: K.f32,
            num_slots: K.i32,
            num_seqs: K.i32,
        ):
            cta = K.cta_id()
            tid = K.thread_id()
            warp = K.warp_id()
            lane = K.lane_id()
            smem = K.smem_pool()
            tiles = smem.alloc((3 * tile_elems,), K.bf16, align=1024)
            raw = smem.alloc((raw_elems,), K.bf16, align=128)
            ftab = smem.alloc((nphase * FST,), K.f32, align=16)
            vtab = smem.alloc((T * D,), K.f32, align=16)
            svec = smem.alloc((2 * T + 2,), K.f32, align=16)
            cpslots = smem.alloc((MAXC + 4,), K.i32, align=16)
            bar_state = K.MBarrier(smem, 1)
            bar_state.init(1)
            bar_vec = K.MBarrier(smem, 1)
            bar_vec.init(nw if vec_split else 1)
            K.ptx.fence.proxy.async_.shared__cta()
            K.cuda.cta_sync()


            def main_role():
                ks = lane & 7
                cw = lane >> 3
                nmain_me = K.max((num_units - cta + num_main - 1) // num_main, K.int32(0))
                ck = K.local_scalar("int32", init=K.int32(0))
                scount = K.local_scalar("int32", init=K.int32(0))
                vcount = K.local_scalar("int32", init=K.int32(0))
                slot0_next = K.local_scalar("int32", init=K.int32(0))
                acc_next = K.local_scalar("int32", init=K.int32(1))
                acc_next2 = K.local_scalar("int32", init=K.int32(1))
                slot0_next2 = K.local_scalar("int32", init=K.int32(0))

                def unit_coords(uu):
                    head = uu // split
                    part = uu % split
                    hv = head % HV
                    n = head // HV
                    return head, part, hv, n, hv // G

                def issue_vec(uu):
                    """Lane 0 of warp w: TMA-prefetch tokens t = w, w+4, ... of unit uu's raw q/k/g/v/beta into `raw`;
                    warp 0 also fetches the gate parameters. Each issuing lane arrives with its own byte count."""
                    _, _, hv, n, h = unit_coords(uu)
                    nwi = nw if vec_split else 1
                    for w in range(nwi):
                        toks = [t for t in range(T) if t % nwi == w]
                        nbytes = len(toks) * RAW_TOK_B + ((GATE_B if gate_lb else 0) if w == 0 else 0)
                        with K.If(warp == w), K.Then():
                            K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_vec.ptr_to([0]), K.uint32(nbytes))
                            for t in toks:
                                _issue_vec_token(n, hv, h, t)
                            if gate_lb and w == 0:
                                gb = T * (RAW_TOK_B // 2)
                                K.ptx[_BULK_G2S](raw.ptr_to([gb]), dt_bias.ptr_to([h * D]), K.uint32(512), bar_vec.ptr_to([0]))
                                K.ptx[_BULK_G2S](raw.ptr_to([gb + 256]), a_log.ptr_to([h - (h % 4)]), K.uint32(16), bar_vec.ptr_to([0]))

                def _issue_vec_token(n, hv, h, t):
                    if True:
                        tok = n * T + t
                        qk_off = (K.cast(tok, "int64") * H + h) * D
                        vg_off = (K.cast(tok, "int64") * HV + hv) * D
                        rb = t * (RAW_TOK_B // 2)
                        K.ptx[_BULK_G2S](raw.ptr_to([rb]), q.ptr_to([qk_off]), K.uint32(256), bar_vec.ptr_to([0]))
                        K.ptx[_BULK_G2S](raw.ptr_to([rb + 128]), k.ptr_to([qk_off]), K.uint32(256), bar_vec.ptr_to([0]))
                        K.ptx[_BULK_G2S](raw.ptr_to([rb + 256]), g.ptr_to([vg_off]), K.uint32(256), bar_vec.ptr_to([0]))
                        K.ptx[_BULK_G2S](raw.ptr_to([rb + 384]), v.ptr_to([vg_off]), K.uint32(256), bar_vec.ptr_to([0]))
                        bidx = K.cast(tok, "int64") * HV + hv
                        K.ptx[_BULK_G2S](raw.ptr_to([rb + 512]), beta.ptr_to([bidx - (bidx % 8)]), K.uint32(16), bar_vec.ptr_to([0]))

                def tile_load(tile, slot, hv, part):
                    """Thread 0: TMA load of cpt rows of (slot, hv) from initial_state into `tile` (no expect_tx)."""
                    src0 = (K.cast(slot, "int64") * HV + hv) * (D * D) + part * cpt * D
                    for pc in range(n_pieces):
                        K.ptx[_BULK_G2S](
                            tiles.ptr_to([tile * tile_elems + pc * (load_piece // 2)]),
                            state_in.ptr_to([src0 + pc * (load_piece // 2)]),
                            K.uint32(load_piece),
                            bar_state.ptr_to([0]),
                        )


                hole_base = hole_base0
                hole_cnt = hole_cnt0

                def copy_coords(kc):
                    c = kc * hole_cnt + (cta - hole_base)
                    return (c % DCU) // split, c % split

                def issue_state(uu, slot0):
                    """Thread 0: TMA load of the unit's state rows into tile A."""
                    _, part, hv, n, _ = unit_coords(uu)
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.uint32(tile_bytes))
                    src0 = (K.cast(slot0, "int64") * HV + hv) * (D * D) + part * cpt * D
                    for pc in range(n_pieces):
                        K.ptx[_BULK_G2S](
                            tiles.ptr_to([pc * (load_piece // 2)]),
                            state_in.ptr_to([src0 + pc * (load_piece // 2)]),
                            K.uint32(load_piece),
                            bar_state.ptr_to([0]),
                        )

                def start_slot(uu, acc_val):
                    _, _, _, n, _ = unit_coords(uu)
                    if not spec:
                        return n
                    acc_c = K.max(K.min(acc_val, K.int32(T)), K.int32(1))
                    s = K.local_scalar("int32")
                    K.ptx.ld.global_.s32(s, ssm_idx.ptr_to([n * T + acc_c - 1]))
                    return s

                def load_acc(uu):
                    _, _, _, n, _ = unit_coords(uu)
                    K.ptx.ld.global_.s32(acc_next, num_accepted.ptr_to([n]))

                def load_acc2(uu):
                    _, _, _, n, _ = unit_coords(uu)
                    K.ptx.ld.global_.s32(acc_next2, num_accepted.ptr_to([n]))

                def prefetch_l2(uu, slot0):
                    """Thread 0: pull the state rows of unit uu (two units ahead) into L2 so the later TMA load
                    does not queue behind the checkpoint write stream in DRAM."""
                    if not l2pf:
                        return
                    _, part, hv, n, _ = unit_coords(uu)
                    src0 = (K.cast(slot0, "int64") * HV + hv) * (D * D) + part * cpt * D
                    K.ptx["cp.async.bulk.prefetch.L2.global"](state_in.ptr_to([src0]), K.uint32(tile_bytes))


                if spec:
                    total_idx = num_seqs * T
                    idx_pre = K.alloc_local((IDX_PRE,), "int32")
                    for i in range(IDX_PRE):
                        K.assign(idx_pre[i], K.int32(-1))
                        with K.If(i * NT + tid < total_idx), K.Then():
                            K.ptx.ld.global_.s32(idx_pre[i], ssm_idx.ptr_to([i * NT + tid]))


                with K.If((lane == 0) & (cta < num_units) & (vec_split or (warp == 0))), K.Then():
                    issue_vec(cta)
                with K.If((tid == 0) & (cta < num_units)), K.Then():
                    if spec:

                        n0 = unit_coords(cta)[3]
                        a0 = K.local_scalar("int32")
                        K.ptx.ld.global_.s32(a0, num_accepted.ptr_to([n0]))
                        cands = K.alloc_local((T,), "int32")
                        for t in range(T):
                            K.ptx.ld.global_.s32(cands[t], ssm_idx.ptr_to([n0 * T + t]))
                        acc_c = K.max(K.min(a0, K.int32(T)), K.int32(1))
                        sel = cands[T - 1]
                        for t in range(T - 2, -1, -1):
                            sel = K.if_then_else(acc_c == t + 1, cands[t], sel)
                        issue_state(cta, sel)
                        with K.If(cta + num_main < num_units), K.Then():
                            load_acc(cta + num_main)
                        if l2pf:
                            with K.If(cta + 2 * num_main < num_units), K.Then():
                                load_acc2(cta + 2 * num_main)
                    else:
                        issue_state(cta, unit_coords(cta)[3])

                def build_flags(cb):
                    """All threads: touched-slot flags for slots [cb, cb + pmax) as u32 in tile B."""
                    nz = K.min(K.int32(pmax), num_slots - cb)
                    with K.serial(0, (nz + NT - 1) // NT, unroll=False) as i:
                        pz = i * NT + tid
                        with K.If(pz < nz), K.Then():
                            K.ptx.st.shared.b32(tiles.ptr_to([tile_elems + 2 * pz]), K.uint32(0))
                    K.cuda.cta_sync()
                    total = num_seqs * T
                    for i in range(IDX_PRE):
                        rel = idx_pre[i] - cb
                        with K.If((i * NT + tid < total) & (rel >= 0) & (rel < pmax)), K.Then():
                            K.ptx.st.shared.b32(tiles.ptr_to([tile_elems + 2 * rel]), K.uint32(1))
                    with K.serial(IDX_PRE, (total + NT - 1) // NT, unroll=False) as i:
                        e = i * NT + tid
                        with K.If(e < total), K.Then():
                            sidx = K.local_scalar("int32")
                            K.ptx.ld.global_.s32(sidx, ssm_idx.ptr_to([e]))
                            rel = sidx - cb
                            with K.If((rel >= 0) & (rel < pmax)), K.Then():
                                K.ptx.st.shared.b32(tiles.ptr_to([tile_elems + 2 * rel]), K.uint32(1))
                    K.cuda.cta_sync()

                def scan_untouched(cb, cnt, want_rank, dst_idx):
                    """Warp 0: walk the flags of chunk cb; record the slots of this CTA's copy units
                    (fast path, want_rank None), or the slot of `want_rank` into cpslots[dst_idx]."""
                    pend = K.min(pmax, num_slots - cb)
                    with K.serial(0, (pend + 31) // 32, unroll=False) as b:
                        pl = b * 32 + lane
                        valid = pl < pend
                        flag = K.local_scalar("uint32", init=K.uint32(1))
                        with K.If(valid), K.Then():
                            K.ptx.ld.shared.b32(flag, tiles.ptr_to([tile_elems + 2 * pl]))
                        untouched = flag == K.uint32(0)
                        upred = K.local_scalar("uint32", init=K.if_then_else(untouched, K.uint32(1), K.uint32(0)))
                        mask = K.local_scalar("uint32")
                        K.ptx.vote_sync.ballot.b32(mask, K.ptx.pred(upred), K.uint32(FULL))
                        below = K.bitwise_and(mask, (K.uint32(1) << K.cast(lane, "uint32")) - K.uint32(1))
                        rank = cnt + K.cast(K.popcount(below), "int32")
                        with K.If(untouched), K.Then():
                            slot = cb + pl
                            if want_rank is None:
                                lo = rank * DCU
                                hi = lo + DCU
                                me = cta - hole_base

                                c0 = K.local_scalar("int32", init=lo + ((me - lo) % hole_cnt + hole_cnt) % hole_cnt)
                                with K.If(me >= 0), K.Then():
                                    with K.While(c0 < hi):
                                        kk = (c0 - me) // hole_cnt
                                        with K.If(kk < MAXC), K.Then():
                                            K.ptx.st.shared.b32(cpslots.ptr_to([kk]), K.reinterpret("uint32", slot))
                                        K.assign(c0, c0 + hole_cnt)
                            else:
                                with K.If((want_rank >= 0) & (rank == want_rank)), K.Then():
                                    K.ptx.st.shared.b32(cpslots.ptr_to([dst_idx]), K.reinterpret("uint32", slot))
                        K.assign(cnt, cnt + K.cast(K.popcount(mask), "int32"))

                ncopy_me = K.local_scalar("int32", init=K.int32(0))
                copy_armed = K.local_scalar("int32", init=K.int32(0))
                if spec:
                    with K.If(tid < MAXC + 4), K.Then():
                        K.ptx.st.shared.b32(cpslots.ptr_to([tid]), K.uint32(0))
                    cnt = K.local_scalar("int32", init=K.int32(0))
                    with K.serial(0, (num_slots + pmax - 1) // pmax, unroll=False) as chunk:
                        cb = chunk * pmax
                        build_flags(cb)
                        with K.If(warp == 0), K.Then():
                            scan_untouched(cb, cnt, None, 0)
                        K.cuda.cta_sync()
                    with K.If(tid == 0), K.Then():
                        K.ptx.st.shared.b32(cpslots.ptr_to([MAXC]), K.reinterpret("uint32", cnt))
                    K.cuda.cta_sync()
                    ucnt = K.local_scalar("int32")
                    K.ptx.ld.shared.b32(ucnt, cpslots.ptr_to([MAXC]))
                    total_copy = ucnt * DCU
                    me0 = cta - hole_base
                    K.assign(ncopy_me, K.if_then_else((me0 >= 0) & (total_copy > me0), (total_copy - me0 + hole_cnt - 1) // hole_cnt, K.int32(0)))

                def preprocess(t, hv, h):
                    """Warp-level: normalize q/k, decay, per-token scalars of token t from `raw`."""
                    rb = t * (RAW_TOK_B // 2)
                    qw = K.alloc_local((2,), "uint32", align=8)
                    kw = K.alloc_local((2,), "uint32", align=8)
                    gw = K.alloc_local((2,), "uint32", align=8)
                    vw = K.alloc_local((2,), "uint32", align=8)
                    K.ptx.ld.shared.v2.b32(qw[0], qw[1], raw.ptr_to([rb + lane * 4]))
                    K.ptx.ld.shared.v2.b32(kw[0], kw[1], raw.ptr_to([rb + 128 + lane * 4]))
                    K.ptx.ld.shared.v2.b32(gw[0], gw[1], raw.ptr_to([rb + 256 + lane * 4]))
                    K.ptx.ld.shared.v2.b32(vw[0], vw[1], raw.ptr_to([rb + 384 + lane * 4]))
                    bbits = K.local_scalar("uint16")
                    K.ptx.ld.shared.b16(bbits, raw.ptr_to([rb + 512 + (hv % 8)]))
                    qf = K.alloc_local((4,), "float32")
                    kf = K.alloc_local((4,), "float32")
                    gf = K.alloc_local((4,), "float32")
                    vf = K.alloc_local((4,), "float32")
                    for src, dst in ((qw, qf), (kw, kf), (gw, gf), (vw, vf)):
                        for p2 in range(2):
                            K.assign(dst[2 * p2], K.cuda.uint_as_float(K.shift_left(src[p2], K.uint32(16))))
                            K.assign(dst[2 * p2 + 1], K.cuda.uint_as_float(K.bitwise_and(src[p2], K.uint32(0xFFFF0000))))

                    sq = K.local_scalar("float32", init=K.float32(0.0))
                    sk = K.local_scalar("float32", init=K.float32(0.0))
                    cpart = K.local_scalar("float32", init=K.float32(0.0))
                    for e in range(4):
                        K.ptx["fma.rn.f32"](sq, qf[e], qf[e], sq)
                        K.ptx["fma.rn.f32"](sk, kf[e], kf[e], sk)
                        K.ptx["fma.rn.f32"](cpart, qf[e], kf[e], cpart)
                    for x in (16, 8, 4, 2, 1):
                        K.ptx["add.f32"](sq, sq, _shfl_bfly(sq, x))
                        K.ptx["add.f32"](sk, sk, _shfl_bfly(sk, x))
                        K.ptx["add.f32"](cpart, cpart, _shfl_bfly(cpart, x))
                    rq = K.local_scalar("float32")
                    rk = K.local_scalar("float32")
                    tmp = K.local_scalar("float32")
                    K.ptx["add.f32"](tmp, sq, K.float32(1e-6))
                    K.ptx["rsqrt.approx.ftz.f32"](rq, tmp)
                    K.ptx["mul.f32"](rq, rq, scale)
                    K.ptx["add.f32"](tmp, sk, K.float32(1e-6))
                    K.ptx["rsqrt.approx.ftz.f32"](rk, tmp)

                    K.ptx["mul.f32"](cpart, cpart, rq)
                    K.ptx["mul.f32"](cpart, cpart, rk)
                    for e in range(4):
                        K.ptx["mul.f32"](qf[e], qf[e], rq)
                        K.ptx["mul.f32"](kf[e], kf[e], rk)
                    if gate_lb:
                        gb = T * (RAW_TOK_B // 2)
                        ea = K.local_scalar("float32")
                        alog = K.local_scalar("uint32")
                        K.ptx.ld.shared.b32(alog, raw.ptr_to([gb + 256 + (h % 4) * 2]))
                        K.ptx["mul.f32"](tmp, K.reinterpret("float32", alog), K.float32(LOG2E))
                        K.ptx["ex2.approx.ftz.f32"](ea, tmp)
                        bias = K.alloc_local((4,), "uint32", align=16)
                        K.ptx.ld.shared.v4.b32(bias[0], bias[1], bias[2], bias[3], raw.ptr_to([gb + lane * 8]))
                        for e in range(4):
                            x = K.local_scalar("float32")
                            K.ptx["add.f32"](x, gf[e], K.reinterpret("float32", bias[e]))
                            K.ptx["mul.f32"](x, x, ea)
                            K.ptx["mul.f32"](x, x, K.float32(-LOG2E))
                            K.ptx["ex2.approx.ftz.f32"](x, x)
                            K.ptx["add.f32"](x, x, K.float32(1.0))
                            K.ptx["rcp.approx.ftz.f32"](x, x)
                            K.ptx["mul.f32"](x, x, lower_bound)
                            K.ptx["mul.f32"](x, x, K.float32(LOG2E))
                            K.ptx["ex2.approx.ftz.f32"](gf[e], x)
                    else:
                        for e in range(4):
                            K.ptx["mul.f32"](gf[e], gf[e], K.float32(LOG2E))
                            K.ptx["ex2.approx.ftz.f32"](gf[e], gf[e])
                    for pp in range(2):
                        pair = lane * 2 + pp
                        ent = t * FST + pair * 8 + (pair >> 3) * 4
                        K.ptx.st.shared.v2.b32(ftab.ptr_to([ent + 2]), K.reinterpret("uint32", gf[2 * pp]), K.reinterpret("uint32", gf[2 * pp + 1]))
                        K.ptx.st.shared.v4.b32(
                            ftab.ptr_to([ent + 4]),
                            K.reinterpret("uint32", kf[2 * pp]), K.reinterpret("uint32", kf[2 * pp + 1]),
                            K.reinterpret("uint32", qf[2 * pp]), K.reinterpret("uint32", qf[2 * pp + 1]),
                        )
                        K.ptx.st.shared.v2.b32(ftab.ptr_to([ent + FST]), K.reinterpret("uint32", kf[2 * pp]), K.reinterpret("uint32", kf[2 * pp + 1]))
                    K.ptx.st.shared.v4.b32(
                        vtab.ptr_to([t * D + lane * 4]),
                        K.reinterpret("uint32", vf[0]), K.reinterpret("uint32", vf[1]),
                        K.reinterpret("uint32", vf[2]), K.reinterpret("uint32", vf[3]),
                    )
                    with K.If(lane == 0), K.Then():
                        bf = K.local_scalar("float32")
                        K.ptx.cvt.f32.bf16(bf, K.cast(bbits, "uint16"))
                        K.ptx.st.shared.v2.b32(svec.ptr_to([2 * t]), K.reinterpret("uint32", bf), K.reinterpret("uint32", cpart))

                S = K.alloc_local((J * 8,), "uint64", align=8)
                d1 = K.alloc_local((J,), "uint64", align=8)
                d2 = K.alloc_local((J,), "uint64", align=8)
                ub = K.alloc_local((J,), "float32")
                zero2 = _f2(K.float32(0.0), K.float32(0.0))

                def vec_ptr(phase, i):
                    cs = K.bitwise_xor(K.int32(i // 4), ks >> 2)
                    pair = ks * 8 + cs * 4 + (i % 4)
                    return ftab.ptr_to([phase * FST + pair * 8 + ks * 4])

                def reduce_and_finish(t, hv, n, row0):
                    """Reduce-scatter d1/d2 over the 8 k-slice lanes; lane owns R columns; broadcast u."""
                    bc = K.alloc_local((2,), "uint32", align=8)
                    K.ptx.ld.shared.v2.b32(bc[0], bc[1], svec.ptr_to([2 * t]))
                    beta_t = K.reinterpret("float32", bc[0])
                    c_t = K.reinterpret("float32", bc[1])
                    vals = []
                    for j in range(J):
                        r = K.local_scalar("float32")
                        K.ptx["add.f32"](r, K.cuda.float2_x(d1[j]), K.cuda.float2_y(d1[j]))
                        vals.append(r)
                    for j in range(J):
                        r = K.local_scalar("float32")
                        K.ptx["add.f32"](r, K.cuda.float2_x(d2[j]), K.cuda.float2_y(d2[j]))
                        vals.append(r)
                    if allreduce:
                        for s in (1, 2, 4):
                            for i in range(M):
                                K.ptx["add.f32"](vals[i], vals[i], _shfl_bfly(vals[i], s))
                        tok = n * T + t
                        out_base = (K.cast(tok, "int64") * HV + hv) * D + row0
                        for j in range(J):
                            col = warp * rows_per_warp + j * 4 + cw
                            vv = K.local_scalar("uint32")
                            K.ptx.ld.shared.b32(vv, vtab.ptr_to([t * D + row0 + col]))
                            diff = K.local_scalar("float32")
                            K.ptx["sub.f32"](diff, K.reinterpret("float32", vv), vals[j])
                            K.ptx["mul.f32"](ub[j], diff, beta_t)
                            o = K.local_scalar("float32")
                            K.ptx["fma.rn.f32"](o, c_t, ub[j], vals[J + j])
                            ob = K.local_scalar("uint16")
                            K.ptx.cvt.rn.bf16.f32(ob, o)
                            K.ptx.st.global_.b16(output.ptr_to([out_base + col]), ob, pred=K.EQ(ks, 0))
                        return
                    for s in (4, 2, 1):
                        if len(vals) > 1:
                            half = len(vals) // 2
                            b = ((ks // s) & 1) != 0
                            nxt = []
                            for i in range(half):
                                send = K.local_scalar("float32", init=K.if_then_else(b, vals[i], vals[i + half]))
                                keep = K.local_scalar("float32", init=K.if_then_else(b, vals[i + half], vals[i]))
                                r = K.local_scalar("float32")
                                K.ptx["add.f32"](r, keep, _shfl_bfly(send, s))
                                nxt.append(r)
                            vals = nxt
                        else:
                            K.ptx["add.f32"](vals[0], vals[0], _shfl_bfly(vals[0], s))
                    assert len(vals) == R

                    is_d2 = (ks >> 2) != 0
                    jbase = ((ks & 3) * M) // 8
                    tok = n * T + t
                    out_base = (K.cast(tok, "int64") * HV + hv) * D + row0
                    uloc = K.alloc_local((R,), "float32")
                    for i in range(R):
                        other = K.local_scalar("float32", init=_shfl_bfly(vals[i], 4))
                        dd1 = K.if_then_else(is_d2, other, vals[i])
                        dd2 = K.if_then_else(is_d2, vals[i], other)
                        col = warp * rows_per_warp + (jbase + i) * 4 + cw
                        vv = K.local_scalar("uint32")
                        K.ptx.ld.shared.b32(vv, vtab.ptr_to([t * D + row0 + col]))
                        diff = K.local_scalar("float32")
                        K.ptx["sub.f32"](diff, K.reinterpret("float32", vv), dd1)
                        K.ptx["mul.f32"](uloc[i], diff, beta_t)
                        o = K.local_scalar("float32")
                        K.ptx["fma.rn.f32"](o, c_t, uloc[i], dd2)
                        ob = K.local_scalar("uint16")
                        K.ptx.cvt.rn.bf16.f32(ob, o)
                        K.ptx.st.global_.b16(output.ptr_to([out_base + col]), ob, pred=K.LT(ks, 4))
                    for j in range(J):

                        src = (lane & 0x18) | ((j * 8) // M)
                        K.assign(ub[j], _shfl_idx(uloc[j % R], src))

                def update_pass(phase, tile, with_dots):
                    if with_dots:
                        for j in range(J):
                            K.assign(d1[j], zero2)
                            K.assign(d2[j], zero2)
                    u2 = K.alloc_local((J,), "uint64", align=8)
                    for j in range(J):
                        K.assign(u2[j], _f2(ub[j], ub[j]))
                    for c in range(2):
                        kp2 = K.alloc_local((4,), "uint64", align=8)
                        a2 = K.alloc_local((4,), "uint64", align=8)
                        k2 = K.alloc_local((4,), "uint64", align=8)
                        q2 = K.alloc_local((4,), "uint64", align=8)
                        for ii in range(4):
                            i = c * 4 + ii
                            if with_dots:
                                K.ptx.ld.shared.v2.b64(kp2[ii], a2[ii], vec_ptr(phase, i))
                                K.ptx.ld.shared.v2.b64(k2[ii], q2[ii], K.ptx.addr(vec_ptr(phase, i), 16))
                            else:
                                K.ptx.ld.shared.b64(kp2[ii], vec_ptr(phase, i))
                        for j in range(J):
                            packed = K.alloc_local((4,), "uint32", align=16)
                            for ii in range(4):
                                i = c * 4 + ii
                                K.ptx.fma.rn.f32x2(S[j * 8 + i], kp2[ii], u2[j], S[j * 8 + i])
                                K.assign(packed[ii], K.cuda.float22bfloat162_rn_from_float2(S[j * 8 + i]))
                                if with_dots:
                                    K.ptx.mul.rn.f32x2(S[j * 8 + i], a2[ii], S[j * 8 + i])
                                    K.ptx.fma.rn.f32x2(d1[j], k2[ii], S[j * 8 + i], d1[j])
                                    K.ptx.fma.rn.f32x2(d2[j], q2[ii], S[j * 8 + i], d2[j])
                            row = warp * rows_per_warp + j * 4 + cw
                            cs = K.bitwise_xor(K.int32(c), ks >> 2)
                            K.ptx.st.shared.v4.b32(
                                tiles.ptr_to([tile * tile_elems + row * D + ks * 16 + cs * 8]),
                                packed[0], packed[1], packed[2], packed[3],
                            )
                    K.ptx.fence.proxy.async_.shared__cta()

                def store_barrier():
                    """All warps' staging writes complete -> warp 0 may issue the TMA store.

                    Warps 1..3 only arrive (named barrier 1, 128 threads) and continue; warp 0 waits.
                    """
                    if bar_arrive:
                        with K.If(warp == 0):
                            with K.Then():
                                K.ptx.bar.sync(K.uint32(1), K.uint32(NT))
                            with K.Else():
                                K.ptx.bar.arrive(K.uint32(1), K.uint32(NT))
                    else:
                        K.cuda.cta_sync()

                def store_checkpoint(tile, dst_slot, hv, row0):
                    dst0 = (K.cast(dst_slot, "int64") * HV + hv) * (D * D) + row0 * D
                    for pc in range(n_pieces):
                        K.ptx[_BULK_S2G](
                            state_out.ptr_to([dst0 + pc * (load_piece // 2)]),
                            tiles.ptr_to([tile * tile_elems + pc * (load_piece // 2)]),
                            K.uint32(load_piece),
                        )
                    K.ptx.cp.async_.bulk.commit_group()

                def store_warp_rows(tile, dst_slot, hv, row0):
                    """Lane 0 of the calling warp: bulk store of this warp's rows_per_warp rows of `tile`."""
                    wrow = row0 + warp * rows_per_warp
                    dst0 = (K.cast(dst_slot, "int64") * HV + hv) * (D * D) + wrow * D
                    K.ptx[_BULK_S2G](
                        state_out.ptr_to([dst0]),
                        tiles.ptr_to([tile * tile_elems + warp * rows_per_warp * D]),
                        K.uint32(rows_per_warp * D * 2),
                    )
                    K.ptx.cp.async_.bulk.commit_group()

                with K.serial(0, nmain_me, unroll=False) as sm:
                    u = cta + sm * num_main
                    head, part, hv, n, h = unit_coords(u)
                    row0 = part * cpt
                    has_next = u + num_main < num_units
                    cur_slot = K.local_scalar("int32")
                    nxt_slot = K.local_scalar("int32")


                    K.cuda.iket.mark("unit-start")
                    tk = _rng("wait-vec")
                    bar_vec.wait(0, vcount & 1)
                    K.assign(vcount, vcount + K.int32(1))
                    _rng_end(tk)
                    tk = _rng("preprocess")
                    with K.If(lane == 0 if warp_auto else tid == 0), K.Then():
                        if spec:
                            K.ptx.ld.global_.s32(nxt_slot, ssm_idx.ptr_to([n * T]))
                        else:
                            K.assign(nxt_slot, n)
                    with K.If(tid == 0), K.Then():
                        if spec:
                            with K.If(has_next), K.Then():
                                K.assign(slot0_next, start_slot(u + num_main, acc_next))
                            if l2pf:
                                with K.If(u + 2 * num_main < num_units), K.Then():
                                    K.assign(slot0_next2, start_slot(u + 2 * num_main, acc_next2))
                                    prefetch_l2(u + 2 * num_main, slot0_next2)
                    for rnd in range((T + nw - 1) // nw):
                        t = rnd * nw + warp
                        if rnd * nw + nw <= T:
                            preprocess(t, hv, h)
                        else:
                            with K.If(t < T), K.Then():
                                preprocess(t, hv, h)

                    K.ptx.fence.proxy.async_.shared__cta()
                    K.cuda.cta_sync()
                    _rng_end(tk)
                    with K.If((lane == 0) & has_next & (vec_split or (warp == 0))), K.Then():
                        issue_vec(u + num_main)


                    tk = _rng("wait-state")
                    bar_state.wait(0, scount & 1)
                    K.assign(scount, scount + K.int32(1))
                    _rng_end(tk)
                    tk = _rng("ld-state")
                    for j in range(J):
                        row = warp * rows_per_warp + j * 4 + cw
                        for c in range(2):
                            wds = K.alloc_local((4,), "uint32", align=16)
                            cs = K.bitwise_xor(K.int32(c), ks >> 2)
                            K.ptx.ld.shared.v4.b32(wds[0], wds[1], wds[2], wds[3], tiles.ptr_to([row * D + ks * 16 + cs * 8]))
                            for i in range(4):
                                K.assign(
                                    S[j * 8 + c * 4 + i],
                                    _f2(
                                        K.cuda.uint_as_float(K.shift_left(wds[i], K.uint32(16))),
                                        K.cuda.uint_as_float(K.bitwise_and(wds[i], K.uint32(0xFFFF0000))),
                                    ),
                                )

                    K.ptx.fence.proxy.async_.shared__cta()
                    K.cuda.cta_sync()
                    _rng_end(tk)
                    with K.If(tid == 0), K.Then():
                        with K.If(has_next):
                            with K.Then():
                                if spec:
                                    issue_state(u + num_main, slot0_next)
                                    with K.If(u + 2 * num_main < num_units), K.Then():
                                        load_acc(u + 2 * num_main)
                                    if l2pf:
                                        with K.If(u + 3 * num_main < num_units), K.Then():
                                            load_acc2(u + 3 * num_main)
                                else:
                                    issue_state(u + num_main, unit_coords(u + num_main)[3])
                                    with K.If(u + 2 * num_main < num_units), K.Then():
                                        prefetch_l2(u + 2 * num_main, unit_coords(u + 2 * num_main)[3])
                            if spec:
                                with K.Else():


                                    with K.If(ncopy_me > 0), K.Then():
                                        nb0 = K.min(K.int32(3), K.min(ncopy_me, K.int32(MAXC)))
                                        K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.cast(nb0, "uint32") * K.uint32(tile_bytes))
                                        hv_c, part_c = copy_coords(0)
                                        slot_c = K.local_scalar("int32")
                                        K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([0]))
                                        tile_load(0, slot_c, hv_c, part_c)
                                        K.assign(copy_armed, K.int32(1))


                    tk = _rng("phase0")
                    for j in range(J):
                        K.assign(d1[j], zero2)
                        K.assign(d2[j], zero2)
                    for i in range(8):
                        a2 = K.local_scalar("uint64")
                        k2 = K.local_scalar("uint64")
                        q2 = K.local_scalar("uint64")
                        K.ptx.ld.shared.b64(a2, K.ptx.addr(vec_ptr(0, i), 8))
                        K.ptx.ld.shared.v2.b64(k2, q2, K.ptx.addr(vec_ptr(0, i), 16))
                        for j in range(J):
                            K.ptx.mul.rn.f32x2(S[j * 8 + i], a2, S[j * 8 + i])
                            K.ptx.fma.rn.f32x2(d1[j], k2, S[j * 8 + i], d1[j])
                            K.ptx.fma.rn.f32x2(d2[j], q2, S[j * 8 + i], d2[j])
                    reduce_and_finish(0, hv, n, row0)
                    _rng_end(tk)


                    issuer = lane == 0 if warp_auto else tid == 0

                    def tile_free_wait():
                        with K.If(issuer), K.Then():
                            K.ptx.cp.async_.bulk.wait_group.read(1)
                        if warp_auto:
                            K.cuda.warp_sync()
                        else:
                            K.cuda.cta_sync()

                    def store_done_sync():
                        if warp_auto:
                            K.cuda.warp_sync()
                        else:
                            K.cuda.cta_sync()

                    def issue_store(tile, slot):
                        if warp_auto:
                            store_warp_rows(tile, slot, hv, row0)
                        else:
                            store_checkpoint(tile, slot, hv, row0)

                    if T > 1:
                        with K.serial(1, T, unroll=False) as ph:
                            tile = 1 + (ck & 1)
                            tk = _rng("wait-tile")
                            tile_free_wait()
                            _rng_end(tk)
                            tk = _rng("update")
                            update_pass(ph, tile, True)
                            _rng_end(tk)
                            tk = _rng("sync-store")
                            store_done_sync()
                            with K.If(issuer), K.Then():
                                K.assign(cur_slot, nxt_slot)
                                if spec:
                                    K.ptx.ld.global_.s32(nxt_slot, ssm_idx.ptr_to([n * T + ph]))
                                issue_store(tile, cur_slot)
                            K.assign(ck, ck + K.int32(1))
                            _rng_end(tk)
                            tk = _rng("reduce")
                            reduce_and_finish(ph, hv, n, row0)
                            _rng_end(tk)

                    tile = 1 + (ck & 1)
                    tk = _rng("wait-tile")
                    tile_free_wait()
                    _rng_end(tk)
                    tk = _rng("update")
                    update_pass(T, tile, False)
                    _rng_end(tk)
                    tk = _rng("sync-store")
                    store_done_sync()
                    with K.If(issuer), K.Then():
                        issue_store(tile, nxt_slot)
                    K.assign(ck, ck + K.int32(1))


                    K.cuda.cta_sync()
                    _rng_end(tk)


                if spec:
                    with K.If(ncopy_me > 0), K.Then():
                        with K.If(lane == 0), K.Then():
                            K.ptx.cp.async_.bulk.wait_group.read(0)
                        K.cuda.cta_sync()
                        nfast = K.min(ncopy_me, K.int32(MAXC))
                        with K.serial(0, (nfast + 2) // 3, unroll=False) as kb:
                            tk = _rng("copy-batch")
                            k0 = kb * 3
                            nb = K.min(K.int32(3), nfast - k0)
                            with K.If(tid == 0), K.Then():
                                pre = (kb == 0) & (copy_armed == 1)
                                with K.If(pre == False), K.Then():
                                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.cast(nb, "uint32") * K.uint32(tile_bytes))
                                for i in range(3):

                                    with K.If((i < nb) & ((i > 0) | (pre == False))), K.Then():
                                        hv_c, part_c = copy_coords(k0 + i)
                                        slot_c = K.local_scalar("int32")
                                        K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([k0 + i]))
                                        tile_load(i, slot_c, hv_c, part_c)
                            bar_state.wait(0, scount & 1)
                            K.assign(scount, scount + K.int32(1))
                            with K.If(tid == 0), K.Then():
                                for i in range(3):
                                    with K.If(i < nb), K.Then():
                                        hv_c, part_c = copy_coords(k0 + i)
                                        slot_c = K.local_scalar("int32")
                                        K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([k0 + i]))
                                        store_checkpoint(i, slot_c, hv_c, part_c * cpt)
                                K.ptx.cp.async_.bulk.wait_group.read(0)

                            K.cuda.cta_sync()
                            _rng_end(tk)

                        with K.If(ncopy_me > MAXC), K.Then():
                            K.cuda.cta_sync()
                            with K.serial(MAXC, ncopy_me, unroll=False) as kc:
                                rank = (kc * hole_cnt + (cta - hole_base)) // DCU
                                cnt3 = K.local_scalar("int32", init=K.int32(0))
                                with K.serial(0, (num_slots + pmax - 1) // pmax, unroll=False) as chunk:
                                    cb = chunk * pmax
                                    build_flags(cb)
                                    with K.If(warp == 0), K.Then():
                                        scan_untouched(cb, cnt3, rank, MAXC + 1)
                                    K.cuda.cta_sync()
                                with K.If(tid == 0), K.Then():
                                    hv_c, part_c = copy_coords(kc)
                                    slot_c = K.local_scalar("int32")
                                    K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([MAXC + 1]))
                                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.uint32(tile_bytes))
                                    tile_load(0, slot_c, hv_c, part_c)
                                bar_state.wait(0, scount & 1)
                                K.assign(scount, scount + K.int32(1))
                                with K.If(tid == 0), K.Then():
                                    hv_c, part_c = copy_coords(kc)
                                    slot_c = K.local_scalar("int32")
                                    K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([MAXC + 1]))
                                    store_checkpoint(0, slot_c, hv_c, part_c * cpt)
                                    K.ptx.cp.async_.bulk.wait_group.read(0)
                                K.cuda.cta_sync()
                tk = _rng("drain")
                with K.If(lane == 0), K.Then():
                    K.ptx.cp.async_.bulk.wait_group.read(0)
                _rng_end(tk)

            main_role()

        return kda_decode_persist


    OVERRIDE = {}
    _COMPILED = {}


    def _pick_cpt(N, T, HV):


        heads = N * HV
        if heads <= 16:
            return 32
        return 64


    def _ctas_per_sm(cpt, T, gate_lb, nw=4):
        smem = 3 * cpt * D * 2 + T * RAW_TOK_B + (GATE_B if gate_lb else 0) + (T + 1) * FST * 4 + T * D * 4 + 256
        by_smem = (220 * 1024) // smem
        if nw == 4:
            by_regs = {16: 6, 32: 5, 64: 3, 128: 2}[cpt]
            if T == 1:
                by_regs = {16: 8, 32: 8, 64: 4, 128: 2}[cpt]
        else:
            by_regs = {32: 3, 64: 2, 128: 1}[cpt]
        return max(1, min(by_smem, by_regs))


    def _compile(**kw):
        key = tuple(sorted(kw.items()))
        exe = _COMPILED.get(key)
        if exe is None:
            kernel = build_kernel(**kw)
            target = kernel.target()
            with target:
                exe = tvm.compile(kernel.mod, target=target, tir_pipeline="tirx")
            _COMPILED[key] = exe
        return exe


    def setup(data, N, T):
        q, k, v, g, beta = data["q"], data["k"], data["v"], data["g"], data["beta"]
        initial_state, final_state, output = data["initial_state"], data["final_state"], data["output"]
        H, HV = q.shape[-2], v.shape[-2]
        P = initial_state.shape[0]
        device = q.device
        gate_lb = data["A_log"] is not None
        spec = T > 1
        cpt = int(OVERRIDE.get("cpt", _pick_cpt(N, T, HV)))
        split = D // cpt
        num_units = N * HV * split
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        nw = int(OVERRIDE.get("nw", 4))
        per_sm = int(OVERRIDE.get("ctas_per_sm", _ctas_per_sm(cpt, T, gate_lb, nw)))
        copy_est = max(P - N * T, 0) * HV * split if spec else 0
        num_main = min(num_units + copy_est, per_sm * sms)
        per_sm = min(per_sm, max(1, -(-num_main // sms)))
        extra = {k: bool(OVERRIDE[k]) for k in ("allreduce", "bar_arrive", "warp_auto", "vec_split") if k in OVERRIDE}
        if "warp_auto" not in extra:
            extra["warp_auto"] = (num_units + copy_est) <= 6 * num_main
        if "vec_split" not in extra:
            extra["vec_split"] = extra["warp_auto"] or (
                T == 6 and num_units + copy_est <= 10 * num_main
            )
        extra["nw"] = nw
        if "l2pf" in OVERRIDE:
            extra["l2pf"] = bool(OVERRIDE["l2pf"])
        else:


            extra["l2pf"] = T == 2 and num_units >= 8 * num_main
        exe = _compile(T=T, H=H, HV=HV, gate_lb=gate_lb, cpt=cpt, num_units=num_units, num_main=num_main, per_sm=per_sm, copy_est=copy_est, **extra)

        dummy_f32 = torch.zeros(4, dtype=torch.float32, device=device)
        dummy_i32 = torch.zeros(1, dtype=torch.int32, device=device)
        a_log = data["A_log"].contiguous().view(-1).to(torch.float32) if gate_lb else dummy_f32
        dt_bias = data["dt_bias"].contiguous().view(-1).to(torch.float32) if gate_lb else dummy_f32
        ssm = data["ssm_state_indices"].contiguous().view(-1) if spec else dummy_i32
        acc = data["num_accepted_tokens"].contiguous().view(-1) if spec else dummy_i32
        lower_bound = float(data["lower_bound"]) if data["lower_bound"] is not None else 0.0
        scale = float(data["scale"])
        args = (
            q.view(-1), k.view(-1), v.view(-1), g.view(-1), beta.view(-1),
            a_log, dt_bias, initial_state.view(-1), final_state.view(-1), output.view(-1),
            ssm, acc, scale, lower_bound, int(P), int(N),
        )
        keep = (q, k, v, g, beta, a_log, dt_bias, initial_state, final_state, output, ssm, acc)

        def run():
            exe(*args)

        run._keep = keep
        run()
        torch.cuda.synchronize(device)
        return run
    return setup

_BASE_SETUP = _make_base()
del _make_base

def _make_defer():
    """KDA recurrent decode, family "colreg-persist" (v10: safe deferred checkpoint hand-off): persistent register-resident recurrence.

    Unit = (sequence n, value head hv, column slice of CPT rows). A persistent CTA (4 warps)
    walks units u = cta, cta + num_main, ... . For each unit the fp32 state of its CPT rows x
    128 k stays in registers across all T tokens: lane (ks = lane & 7, cw = lane >> 3) holds
    k-slice [16 ks, 16 ks + 16) of the J = CPT / 16 rows v = warp * CPT/4 + j * 4 + cw, with
    register chunk c holding k-chunk c ^ (ks >> 2) so quarter-warp shared accesses are
    bank-conflict free on the plain row-major tiles.

    Pipeline per unit (tile A = input, tiles B/C = checkpoint staging):
      1. wait raw vectors (TMA-prefetched during the previous unit) -> preprocess into the
         per-phase vector table; then issue the raw-vector prefetch of the next unit.
      2. wait the state tile (TMA-prefetched during the previous unit) -> registers; then
         issue the next unit's state load into tile A.
      3. phase 0: S <- alpha_0 S ; d1 += k_0 S ; d2 += q_0 S ; reduce -> u_0, out_0.
      4. phase p: hand off checkpoint p-2 (proxy fence + cp.async.bulk store of its staging tile,
         issued now that its shared-memory writes have long drained; wait_group.read(1) frees the
         tile of checkpoint p); S <- S + k_{p-1} u_{p-1} packed to bf16 into the staging tile;
         S <- alpha_p S, dots, reduce -> u_p, out_p.
      5. final update + hand-off of checkpoints T-2 and T-1.
    Deferring the proxy fence and the store issue by one phase takes them off the per-phase
    latency chain (measured: the fence right after the staging writes dominated the update range
    of a CTA running alone on its SM). Unlike v9, the running checkpoint counter advances only
    after the final hand-off: Kern scalar expressions are lazy, so advancing it first made the
    store read the opposite, uninitialized ping-pong tile. Warps run their phases autonomously
    (warp-level syncs, per-warp contiguous-row stores) in the latency-bound regime; CTA-wide
    stores otherwise.
    Untouched pool slots (the harness poisons final_state before every check) are copied by the
    same persistent CTAs after their recurrence units as tile-sized copy units (TMA load into
    tiles A/B/C, TMA store out) in batches of three. Each CTA resolves the pool slots of its copy
    units in its prologue (touched-slot bitset from ssm_state_indices, built while the first
    index / state / vector loads are in flight) with a warp-parallel scan of the bitset.
    """

    import torch
    import tvm

    import tirx_kernels.kern as K

    D = 128
    LOG2E = 1.4426950408889634
    FULL = 0xFFFFFFFF
    _BULK_G2S = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
    _BULK_S2G = "cp.async.bulk.global.shared::cta.bulk_group"
    FST = 64 * 8 + 8 * 4
    RAW_TOK_B = 4 * 256 + 16
    GATE_B = 512 + 16
    MAXC = 16
    MAXH = 6
    IDX_PRE = 8
    FLAG_WORDS = 128


    def _shfl_bfly(val_f32, xor):
        out = K.local_scalar("uint32")
        K.ptx.shfl_sync.bfly.b32(out, K.reinterpret("uint32", val_f32), K.uint32(xor), K.uint32(31), K.uint32(FULL))
        return K.reinterpret("float32", out)


    def _shfl_idx(val_f32, src_lane):
        out = K.local_scalar("uint32")
        K.ptx.shfl_sync.idx.b32(out, K.reinterpret("uint32", val_f32), K.cast(src_lane, "uint32"), K.uint32(31), K.uint32(FULL))
        return K.reinterpret("float32", out)


    def _f2(a, b):
        return K.cuda.make_float2(a, b)


    def _rng(name):
        """IKET range token (stripped by the production pipeline)."""
        token = K.alloc_local([1], "uint32")
        K.assign(token[0], K.cuda.iket.range_start(name))
        return token


    def _rng_end(token):
        K.cuda.iket.range_end(token[0])


    def _ntiles(T):
        """Tiles: A = input state, B/C = checkpoint staging (also the copy batch buffers)."""
        return 3


    def _smem_bytes(T, cpt, gate_lb, ntiles):
        return ntiles * cpt * D * 2 + T * RAW_TOK_B + (GATE_B if gate_lb else 0) + (T + 1) * FST * 4 + T * D * 4 + FLAG_WORDS * 4 + 512


    def build_kernel(*, T, H, HV, gate_lb, cpt, num_units, num_main, per_sm, copy_est=0, allreduce=None, vec_split=True, nw=4, ntiles=None, warp_auto=True, bar_arrive=False, l2pf=False):
        assert D % cpt == 0 and cpt % (4 * nw) == 0, "each warp needs a multiple of 4 rows"
        del bar_arrive
        G = HV // H
        NT = 32 * nw
        rows_per_warp = cpt // nw
        J = rows_per_warp // 4
        M = 2 * J
        R = max(1, M // 8)
        split = D // cpt
        tile_elems = cpt * D
        tile_bytes = tile_elems * 2
        raw_bytes = T * RAW_TOK_B + (GATE_B if gate_lb else 0)
        raw_elems = raw_bytes // 2
        nphase = T + 1
        load_piece = min(tile_bytes, 16384)
        assert tile_bytes % load_piece == 0
        n_pieces = tile_bytes // load_piece
        if ntiles is None:
            ntiles = _ntiles(T)
        assert ntiles == 3
        num_ctas = num_main
        pmax = FLAG_WORDS * 32
        spec = T > 1
        DCU = HV * split
        if allreduce is None:
            allreduce = J <= 2
        if num_main > num_units:
            hole_base0 = num_units
        else:
            hole_base0 = num_units % num_main
        hole_cnt0 = num_main - hole_base0

        if hole_cnt0 == 0 or copy_est > hole_cnt0 * max(MAXH, T + 1):
            hole_base0, hole_cnt0 = 0, num_main

        @K.kernel(warps=nw, arch="sm_100a", grid=num_ctas, min_blocks_per_sm=per_sm)
        def kda_decode_persist(
            q: K.gptr[K.bf16],
            k: K.gptr[K.bf16],
            v: K.gptr[K.bf16],
            g: K.gptr[K.bf16],
            beta: K.gptr[K.bf16],
            a_log: K.gptr[K.f32],
            dt_bias: K.gptr[K.f32],
            state_in: K.gptr[K.bf16],
            state_out: K.gptr[K.bf16],
            output: K.gptr[K.bf16],
            ssm_idx: K.gptr[K.i32],
            num_accepted: K.gptr[K.i32],
            scale: K.f32,
            lower_bound: K.f32,
            num_slots: K.i32,
            num_seqs: K.i32,
        ):
            cta = K.cta_id()
            tid = K.thread_id()
            warp = K.warp_id()
            lane = K.lane_id()
            smem = K.smem_pool()
            tiles = smem.alloc((ntiles * tile_elems,), K.bf16, align=1024)
            raw = smem.alloc((raw_elems,), K.bf16, align=128)
            ftab = smem.alloc((nphase * FST,), K.f32, align=16)
            vtab = smem.alloc((T * D,), K.f32, align=16)
            svec = smem.alloc((2 * T + 2,), K.f32, align=16)
            cpslots = smem.alloc((MAXC + 4,), K.i32, align=16)
            flags = smem.alloc((FLAG_WORDS,), K.u32, align=16)
            bar_state = K.MBarrier(smem, 1)
            bar_state.init(1)
            bar_vec = K.MBarrier(smem, 1)
            bar_vec.init(nw if vec_split else 1)
            K.ptx.fence.proxy.async_.shared__cta()
            K.cuda.cta_sync()


            def main_role():
                ks = lane & 7
                cw = lane >> 3
                nmain_me = K.max((num_units - cta + num_main - 1) // num_main, K.int32(0))
                ck = K.local_scalar("int32", init=K.int32(0))
                scount = K.local_scalar("int32", init=K.int32(0))
                vcount = K.local_scalar("int32", init=K.int32(0))
                slot0_next = K.local_scalar("int32", init=K.int32(0))
                acc_next = K.local_scalar("int32", init=K.int32(1))
                acc_next2 = K.local_scalar("int32", init=K.int32(1))
                slot0_next2 = K.local_scalar("int32", init=K.int32(0))

                def unit_coords(uu):
                    head = uu // split
                    part = uu % split
                    hv = head % HV
                    n = head // HV
                    return head, part, hv, n, hv // G

                def issue_vec(uu):
                    """Lane 0 of warp w: TMA-prefetch tokens t = w, w+4, ... of unit uu's raw q/k/g/v/beta into `raw`;
                    warp 0 also fetches the gate parameters. Each issuing lane arrives with its own byte count."""
                    _, _, hv, n, h = unit_coords(uu)
                    nwi = nw if vec_split else 1
                    for w in range(nwi):
                        toks = [t for t in range(T) if t % nwi == w]
                        nbytes = len(toks) * RAW_TOK_B + ((GATE_B if gate_lb else 0) if w == 0 else 0)
                        with K.If(warp == w), K.Then():
                            K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_vec.ptr_to([0]), K.uint32(nbytes))
                            for t in toks:
                                _issue_vec_token(n, hv, h, t)
                            if gate_lb and w == 0:
                                gb = T * (RAW_TOK_B // 2)
                                K.ptx[_BULK_G2S](raw.ptr_to([gb]), dt_bias.ptr_to([h * D]), K.uint32(512), bar_vec.ptr_to([0]))
                                K.ptx[_BULK_G2S](raw.ptr_to([gb + 256]), a_log.ptr_to([h - (h % 4)]), K.uint32(16), bar_vec.ptr_to([0]))

                def _issue_vec_token(n, hv, h, t):
                    tok = n * T + t
                    qk_off = (K.cast(tok, "int64") * H + h) * D
                    vg_off = (K.cast(tok, "int64") * HV + hv) * D
                    rb = t * (RAW_TOK_B // 2)
                    K.ptx[_BULK_G2S](raw.ptr_to([rb]), q.ptr_to([qk_off]), K.uint32(256), bar_vec.ptr_to([0]))
                    K.ptx[_BULK_G2S](raw.ptr_to([rb + 128]), k.ptr_to([qk_off]), K.uint32(256), bar_vec.ptr_to([0]))
                    K.ptx[_BULK_G2S](raw.ptr_to([rb + 256]), g.ptr_to([vg_off]), K.uint32(256), bar_vec.ptr_to([0]))
                    K.ptx[_BULK_G2S](raw.ptr_to([rb + 384]), v.ptr_to([vg_off]), K.uint32(256), bar_vec.ptr_to([0]))
                    bidx = K.cast(tok, "int64") * HV + hv
                    K.ptx[_BULK_G2S](raw.ptr_to([rb + 512]), beta.ptr_to([bidx - (bidx % 8)]), K.uint32(16), bar_vec.ptr_to([0]))

                def tile_load(tile, slot, hv, part):
                    """Thread 0: TMA load of cpt rows of (slot, hv) from initial_state into `tile` (no expect_tx)."""
                    src0 = (K.cast(slot, "int64") * HV + hv) * (D * D) + part * cpt * D
                    for pc in range(n_pieces):
                        K.ptx[_BULK_G2S](
                            tiles.ptr_to([tile * tile_elems + pc * (load_piece // 2)]),
                            state_in.ptr_to([src0 + pc * (load_piece // 2)]),
                            K.uint32(load_piece),
                            bar_state.ptr_to([0]),
                        )

                def tile_store(tile, dst_slot, hv, row0):
                    """One thread: bulk store of `tile` (cpt rows) to (dst_slot, hv) rows [row0, row0 + cpt)."""
                    dst0 = (K.cast(dst_slot, "int64") * HV + hv) * (D * D) + row0 * D
                    for pc in range(n_pieces):
                        K.ptx[_BULK_S2G](
                            state_out.ptr_to([dst0 + pc * (load_piece // 2)]),
                            tiles.ptr_to([tile * tile_elems + pc * (load_piece // 2)]),
                            K.uint32(load_piece),
                        )
                    K.ptx.cp.async_.bulk.commit_group()


                hole_base = hole_base0
                hole_cnt = hole_cnt0

                def copy_coords(kc):
                    c = kc * hole_cnt + (cta - hole_base)
                    return (c % DCU) // split, c % split

                def issue_state(uu, slot0):
                    """Thread 0: arm bar_state and TMA-load the unit's state rows into tile A."""
                    _, part, hv, n, _ = unit_coords(uu)
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.uint32(tile_bytes))
                    tile_load(0, slot0, hv, part)

                def start_slot(uu, acc_val):
                    _, _, _, n, _ = unit_coords(uu)
                    if not spec:
                        return n
                    acc_c = K.max(K.min(acc_val, K.int32(T)), K.int32(1))
                    s = K.local_scalar("int32")
                    K.ptx.ld.global_.s32(s, ssm_idx.ptr_to([n * T + acc_c - 1]))
                    return s

                def load_acc(uu):
                    _, _, _, n, _ = unit_coords(uu)
                    K.ptx.ld.global_.s32(acc_next, num_accepted.ptr_to([n]))

                def load_acc2(uu):
                    _, _, _, n, _ = unit_coords(uu)
                    K.ptx.ld.global_.s32(acc_next2, num_accepted.ptr_to([n]))

                def prefetch_l2(uu, slot0):
                    """Thread 0: pull the state rows of unit uu (two units ahead) into L2 so the later TMA load
                    does not queue behind the checkpoint write stream in DRAM."""
                    if not l2pf:
                        return
                    _, part, hv, n, _ = unit_coords(uu)
                    src0 = (K.cast(slot0, "int64") * HV + hv) * (D * D) + part * cpt * D
                    K.ptx["cp.async.bulk.prefetch.L2.global"](state_in.ptr_to([src0]), K.uint32(tile_bytes))


                def build_flags(cb):
                    """All threads: bitset of the touched slots [cb, cb + pmax) in `flags` (rescanning fallback:
                    reloads every index so the prologue's prefetched registers stay dead here)."""
                    with K.If(tid < FLAG_WORDS), K.Then():
                        K.ptx.st.shared.b32(flags.ptr_to([tid]), K.uint32(0))
                    K.cuda.cta_sync()
                    total = num_seqs * T
                    with K.serial(0, (total + NT - 1) // NT, unroll=False) as i:
                        e = i * NT + tid
                        with K.If(e < total), K.Then():
                            sidx = K.local_scalar("int32")
                            K.ptx.ld.global_.s32(sidx, ssm_idx.ptr_to([e]))
                            rel = sidx - cb
                            with K.If((rel >= 0) & (rel < pmax)), K.Then():
                                K.ptx.red.shared.or_.b32(flags.ptr_to([rel >> 5]), K.uint32(1) << K.cast(rel & 31, "uint32"))
                    K.cuda.cta_sync()

                def scan_untouched(cb, cnt, want_rank, dst_idx):
                    """Warp 0: walk the bitset of chunk cb (one word of 32 slots per step, lane = slot within the word);
                    record the slots of this CTA's copy units (fast path, want_rank None), or the slot of `want_rank`
                    into cpslots[dst_idx]."""
                    pend = K.min(pmax, num_slots - cb)
                    with K.serial(0, (pend + 31) // 32, unroll=False) as b:
                        word = K.local_scalar("uint32")
                        K.ptx.ld.shared.b32(word, flags.ptr_to([b]))
                        nvalid = K.min(K.int32(32), pend - b * 32)
                        vmask = K.if_then_else(nvalid >= 32, K.uint32(FULL), (K.uint32(1) << K.cast(nvalid, "uint32")) - K.uint32(1))
                        mask = K.bitwise_and(K.bitwise_not(word), vmask)
                        untouched = K.bitwise_and(mask >> K.cast(lane, "uint32"), K.uint32(1)) != K.uint32(0)
                        below = K.bitwise_and(mask, (K.uint32(1) << K.cast(lane, "uint32")) - K.uint32(1))
                        rank = cnt + K.cast(K.popcount(below), "int32")
                        with K.If(untouched), K.Then():
                            slot = cb + b * 32 + lane
                            if want_rank is None:
                                lo = rank * DCU
                                hi = lo + DCU
                                me = cta - hole_base

                                c0 = K.local_scalar("int32", init=lo + ((me - lo) % hole_cnt + hole_cnt) % hole_cnt)
                                with K.If(me >= 0), K.Then():
                                    with K.While(c0 < hi):
                                        kk = (c0 - me) // hole_cnt
                                        with K.If(kk < MAXC), K.Then():
                                            K.ptx.st.shared.b32(cpslots.ptr_to([kk]), K.reinterpret("uint32", slot))
                                        K.assign(c0, c0 + hole_cnt)
                            else:
                                with K.If((want_rank >= 0) & (rank == want_rank)), K.Then():
                                    K.ptx.st.shared.b32(cpslots.ptr_to([dst_idx]), K.reinterpret("uint32", slot))
                        K.assign(cnt, cnt + K.cast(K.popcount(mask), "int32"))

                ncopy_me = K.local_scalar("int32", init=K.int32(0))
                copy_armed = K.local_scalar("int32", init=K.int32(0))




                if spec:
                    total_idx = num_seqs * T
                    idx_pre = K.alloc_local((IDX_PRE,), "int32")
                    for i in range(IDX_PRE):
                        K.assign(idx_pre[i], K.int32(-1))
                        with K.If(i * NT + tid < total_idx), K.Then():
                            K.ptx.ld.global_.s32(idx_pre[i], ssm_idx.ptr_to([i * NT + tid]))
                    a0 = K.local_scalar("int32", init=K.int32(1))
                    cands = K.alloc_local((T,), "int32")
                    for t in range(T):
                        K.assign(cands[t], K.int32(0))
                    with K.If((tid == 0) & (cta < num_units)), K.Then():
                        n0 = unit_coords(cta)[3]
                        K.ptx.ld.global_.s32(a0, num_accepted.ptr_to([n0]))
                        for t in range(T):
                            K.ptx.ld.global_.s32(cands[t], ssm_idx.ptr_to([n0 * T + t]))
                with K.If((lane == 0) & (cta < num_units) & (vec_split or (warp == 0))), K.Then():
                    issue_vec(cta)

                def issue_first_state():
                    if spec:
                        acc_c = K.max(K.min(a0, K.int32(T)), K.int32(1))
                        sel = cands[T - 1]
                        for t in range(T - 2, -1, -1):
                            sel = K.if_then_else(acc_c == t + 1, cands[t], sel)
                        issue_state(cta, sel)
                        with K.If(cta + num_main < num_units), K.Then():
                            load_acc(cta + num_main)
                        if l2pf:
                            with K.If(cta + 2 * num_main < num_units), K.Then():
                                load_acc2(cta + 2 * num_main)
                    else:
                        issue_state(cta, unit_coords(cta)[3])



                if spec:
                    with K.If(tid < MAXC + 4), K.Then():
                        K.ptx.st.shared.b32(cpslots.ptr_to([tid]), K.uint32(0))
                    cnt = K.local_scalar("int32", init=K.int32(0))
                    with K.serial(0, (num_slots + pmax - 1) // pmax, unroll=False) as chunk:
                        cb = chunk * pmax
                        with K.If(tid < FLAG_WORDS), K.Then():
                            K.ptx.st.shared.b32(flags.ptr_to([tid]), K.uint32(0))
                        K.cuda.cta_sync()
                        with K.If((chunk == 0) & (tid == 0) & (cta < num_units)), K.Then():
                            issue_first_state()
                        total = num_seqs * T
                        for i in range(IDX_PRE):
                            rel = idx_pre[i] - cb
                            with K.If((i * NT + tid < total) & (rel >= 0) & (rel < pmax)), K.Then():
                                K.ptx.red.shared.or_.b32(flags.ptr_to([rel >> 5]), K.uint32(1) << K.cast(rel & 31, "uint32"))
                        with K.serial(IDX_PRE, (total + NT - 1) // NT, unroll=False) as i:
                            e = i * NT + tid
                            with K.If(e < total), K.Then():
                                sidx = K.local_scalar("int32")
                                K.ptx.ld.global_.s32(sidx, ssm_idx.ptr_to([e]))
                                rel = sidx - cb
                                with K.If((rel >= 0) & (rel < pmax)), K.Then():
                                    K.ptx.red.shared.or_.b32(flags.ptr_to([rel >> 5]), K.uint32(1) << K.cast(rel & 31, "uint32"))
                        K.cuda.cta_sync()
                        with K.If(warp == 0), K.Then():
                            scan_untouched(cb, cnt, None, 0)
                        K.cuda.cta_sync()
                    with K.If(tid == 0), K.Then():
                        K.ptx.st.shared.b32(cpslots.ptr_to([MAXC]), K.reinterpret("uint32", cnt))
                    K.cuda.cta_sync()
                    ucnt = K.local_scalar("int32")
                    K.ptx.ld.shared.b32(ucnt, cpslots.ptr_to([MAXC]))
                    total_copy = ucnt * DCU
                    me0 = cta - hole_base
                    K.assign(ncopy_me, K.if_then_else((me0 >= 0) & (total_copy > me0), (total_copy - me0 + hole_cnt - 1) // hole_cnt, K.int32(0)))
                else:
                    with K.If((tid == 0) & (cta < num_units)), K.Then():
                        issue_first_state()

                def preprocess(t, hv, h):
                    """Warp-level: normalize q/k, decay, per-token scalars of token t from `raw`."""
                    rb = t * (RAW_TOK_B // 2)
                    qw = K.alloc_local((2,), "uint32", align=8)
                    kw = K.alloc_local((2,), "uint32", align=8)
                    gw = K.alloc_local((2,), "uint32", align=8)
                    vw = K.alloc_local((2,), "uint32", align=8)
                    K.ptx.ld.shared.v2.b32(qw[0], qw[1], raw.ptr_to([rb + lane * 4]))
                    K.ptx.ld.shared.v2.b32(kw[0], kw[1], raw.ptr_to([rb + 128 + lane * 4]))
                    K.ptx.ld.shared.v2.b32(gw[0], gw[1], raw.ptr_to([rb + 256 + lane * 4]))
                    K.ptx.ld.shared.v2.b32(vw[0], vw[1], raw.ptr_to([rb + 384 + lane * 4]))
                    bbits = K.local_scalar("uint16")
                    K.ptx.ld.shared.b16(bbits, raw.ptr_to([rb + 512 + (hv % 8)]))
                    qf = K.alloc_local((4,), "float32")
                    kf = K.alloc_local((4,), "float32")
                    gf = K.alloc_local((4,), "float32")
                    vf = K.alloc_local((4,), "float32")
                    for src, dst in ((qw, qf), (kw, kf), (gw, gf), (vw, vf)):
                        for p2 in range(2):
                            K.assign(dst[2 * p2], K.cuda.uint_as_float(K.shift_left(src[p2], K.uint32(16))))
                            K.assign(dst[2 * p2 + 1], K.cuda.uint_as_float(K.bitwise_and(src[p2], K.uint32(0xFFFF0000))))

                    sq = K.local_scalar("float32", init=K.float32(0.0))
                    sk = K.local_scalar("float32", init=K.float32(0.0))
                    cpart = K.local_scalar("float32", init=K.float32(0.0))
                    for e in range(4):
                        K.ptx["fma.rn.f32"](sq, qf[e], qf[e], sq)
                        K.ptx["fma.rn.f32"](sk, kf[e], kf[e], sk)
                        K.ptx["fma.rn.f32"](cpart, qf[e], kf[e], cpart)
                    for x in (16, 8, 4, 2, 1):
                        K.ptx["add.f32"](sq, sq, _shfl_bfly(sq, x))
                        K.ptx["add.f32"](sk, sk, _shfl_bfly(sk, x))
                        K.ptx["add.f32"](cpart, cpart, _shfl_bfly(cpart, x))
                    rq = K.local_scalar("float32")
                    rk = K.local_scalar("float32")
                    tmp = K.local_scalar("float32")
                    K.ptx["add.f32"](tmp, sq, K.float32(1e-6))
                    K.ptx["rsqrt.approx.ftz.f32"](rq, tmp)
                    K.ptx["mul.f32"](rq, rq, scale)
                    K.ptx["add.f32"](tmp, sk, K.float32(1e-6))
                    K.ptx["rsqrt.approx.ftz.f32"](rk, tmp)

                    K.ptx["mul.f32"](cpart, cpart, rq)
                    K.ptx["mul.f32"](cpart, cpart, rk)
                    for e in range(4):
                        K.ptx["mul.f32"](qf[e], qf[e], rq)
                        K.ptx["mul.f32"](kf[e], kf[e], rk)
                    if gate_lb:
                        gb = T * (RAW_TOK_B // 2)
                        ea = K.local_scalar("float32")
                        alog = K.local_scalar("uint32")
                        K.ptx.ld.shared.b32(alog, raw.ptr_to([gb + 256 + (h % 4) * 2]))
                        K.ptx["mul.f32"](tmp, K.reinterpret("float32", alog), K.float32(LOG2E))
                        K.ptx["ex2.approx.ftz.f32"](ea, tmp)
                        bias = K.alloc_local((4,), "uint32", align=16)
                        K.ptx.ld.shared.v4.b32(bias[0], bias[1], bias[2], bias[3], raw.ptr_to([gb + lane * 8]))
                        for e in range(4):
                            x = K.local_scalar("float32")
                            K.ptx["add.f32"](x, gf[e], K.reinterpret("float32", bias[e]))
                            K.ptx["mul.f32"](x, x, ea)
                            K.ptx["mul.f32"](x, x, K.float32(-LOG2E))
                            K.ptx["ex2.approx.ftz.f32"](x, x)
                            K.ptx["add.f32"](x, x, K.float32(1.0))
                            K.ptx["rcp.approx.ftz.f32"](x, x)
                            K.ptx["mul.f32"](x, x, lower_bound)
                            K.ptx["mul.f32"](x, x, K.float32(LOG2E))
                            K.ptx["ex2.approx.ftz.f32"](gf[e], x)
                    else:
                        for e in range(4):
                            K.ptx["mul.f32"](gf[e], gf[e], K.float32(LOG2E))
                            K.ptx["ex2.approx.ftz.f32"](gf[e], gf[e])
                    for pp in range(2):
                        pair = lane * 2 + pp
                        ent = t * FST + pair * 8 + (pair >> 3) * 4
                        K.ptx.st.shared.v2.b32(ftab.ptr_to([ent + 2]), K.reinterpret("uint32", gf[2 * pp]), K.reinterpret("uint32", gf[2 * pp + 1]))
                        K.ptx.st.shared.v4.b32(
                            ftab.ptr_to([ent + 4]),
                            K.reinterpret("uint32", kf[2 * pp]), K.reinterpret("uint32", kf[2 * pp + 1]),
                            K.reinterpret("uint32", qf[2 * pp]), K.reinterpret("uint32", qf[2 * pp + 1]),
                        )
                        K.ptx.st.shared.v2.b32(ftab.ptr_to([ent + FST]), K.reinterpret("uint32", kf[2 * pp]), K.reinterpret("uint32", kf[2 * pp + 1]))
                    K.ptx.st.shared.v4.b32(
                        vtab.ptr_to([t * D + lane * 4]),
                        K.reinterpret("uint32", vf[0]), K.reinterpret("uint32", vf[1]),
                        K.reinterpret("uint32", vf[2]), K.reinterpret("uint32", vf[3]),
                    )
                    with K.If(lane == 0), K.Then():
                        bf = K.local_scalar("float32")
                        K.ptx.cvt.f32.bf16(bf, K.cast(bbits, "uint16"))
                        K.ptx.st.shared.v2.b32(svec.ptr_to([2 * t]), K.reinterpret("uint32", bf), K.reinterpret("uint32", cpart))


                S = K.alloc_local((J * 8,), "uint64", align=8)
                d1 = K.alloc_local((J,), "uint64", align=8)
                d2 = K.alloc_local((J,), "uint64", align=8)
                u2 = K.alloc_local((J,), "uint64", align=8)
                zero2 = _f2(K.float32(0.0), K.float32(0.0))

                def vec_ptr(phase, i):
                    cs = K.bitwise_xor(K.int32(i // 4), ks >> 2)
                    pair = ks * 8 + cs * 4 + (i % 4)
                    return ftab.ptr_to([phase * FST + pair * 8 + ks * 4])

                def ld_state():
                    for j in range(J):
                        row = warp * rows_per_warp + j * 4 + cw
                        for c in range(2):
                            wds = K.alloc_local((4,), "uint32", align=16)
                            cs = K.bitwise_xor(K.int32(c), ks >> 2)
                            K.ptx.ld.shared.v4.b32(wds[0], wds[1], wds[2], wds[3], tiles.ptr_to([row * D + ks * 16 + cs * 8]))
                            for i in range(4):
                                K.assign(
                                    S[j * 8 + c * 4 + i],
                                    _f2(
                                        K.cuda.uint_as_float(K.shift_left(wds[i], K.uint32(16))),
                                        K.cuda.uint_as_float(K.bitwise_and(wds[i], K.uint32(0xFFFF0000))),
                                    ),
                                )

                def phase0_dots():
                    for j in range(J):
                        K.assign(d1[j], zero2)
                        K.assign(d2[j], zero2)
                    for i in range(8):
                        a2 = K.local_scalar("uint64")
                        k2 = K.local_scalar("uint64")
                        q2 = K.local_scalar("uint64")
                        K.ptx.ld.shared.b64(a2, K.ptx.addr(vec_ptr(0, i), 8))
                        K.ptx.ld.shared.v2.b64(k2, q2, K.ptx.addr(vec_ptr(0, i), 16))
                        for j in range(J):
                            K.ptx.mul.rn.f32x2(S[j * 8 + i], a2, S[j * 8 + i])
                            K.ptx.fma.rn.f32x2(d1[j], k2, S[j * 8 + i], d1[j])
                            K.ptx.fma.rn.f32x2(d2[j], q2, S[j * 8 + i], d2[j])

                def reduce_and_finish(t, hv, n, row0):
                    """Reduce d1/d2 over the 8 k-slice lanes (all-reduce or reduce-scatter); u_t (into u2) and out_t."""
                    bc = K.alloc_local((2,), "uint32", align=8)
                    K.ptx.ld.shared.v2.b32(bc[0], bc[1], svec.ptr_to([2 * t]))
                    beta_t = K.reinterpret("float32", bc[0])
                    c_t = K.reinterpret("float32", bc[1])
                    vals = []
                    for j in range(J):
                        r = K.local_scalar("float32")
                        K.ptx["add.f32"](r, K.cuda.float2_x(d1[j]), K.cuda.float2_y(d1[j]))
                        vals.append(r)
                    for j in range(J):
                        r = K.local_scalar("float32")
                        K.ptx["add.f32"](r, K.cuda.float2_x(d2[j]), K.cuda.float2_y(d2[j]))
                        vals.append(r)
                    tok = n * T + t
                    out_base = (K.cast(tok, "int64") * HV + hv) * D + row0
                    if allreduce:
                        for s in (1, 2, 4):
                            for i in range(M):
                                K.ptx["add.f32"](vals[i], vals[i], _shfl_bfly(vals[i], s))
                        for j in range(J):
                            col = warp * rows_per_warp + j * 4 + cw
                            vv = K.local_scalar("uint32")
                            K.ptx.ld.shared.b32(vv, vtab.ptr_to([t * D + row0 + col]))
                            diff = K.local_scalar("float32")
                            K.ptx["sub.f32"](diff, K.reinterpret("float32", vv), vals[j])
                            uj = K.local_scalar("float32")
                            K.ptx["mul.f32"](uj, diff, beta_t)
                            K.assign(u2[j], _f2(uj, uj))
                            o = K.local_scalar("float32")
                            K.ptx["fma.rn.f32"](o, c_t, uj, vals[J + j])
                            ob = K.local_scalar("uint16")
                            K.ptx.cvt.rn.bf16.f32(ob, o)
                            K.ptx.st.global_.b16(output.ptr_to([out_base + col]), ob, pred=K.EQ(ks, 0))
                        return
                    for s in (4, 2, 1):
                        if len(vals) > 1:
                            half = len(vals) // 2
                            b = ((ks // s) & 1) != 0
                            nxt = []
                            for i in range(half):
                                send = K.local_scalar("float32", init=K.if_then_else(b, vals[i], vals[i + half]))
                                keep = K.local_scalar("float32", init=K.if_then_else(b, vals[i + half], vals[i]))
                                r = K.local_scalar("float32")
                                K.ptx["add.f32"](r, keep, _shfl_bfly(send, s))
                                nxt.append(r)
                            vals = nxt
                        else:
                            K.ptx["add.f32"](vals[0], vals[0], _shfl_bfly(vals[0], s))
                    assert len(vals) == R

                    is_d2 = (ks >> 2) != 0
                    jbase = ((ks & 3) * M) // 8
                    uloc = K.alloc_local((R,), "float32")
                    for i in range(R):
                        other = K.local_scalar("float32", init=_shfl_bfly(vals[i], 4))
                        dd1 = K.if_then_else(is_d2, other, vals[i])
                        dd2 = K.if_then_else(is_d2, vals[i], other)
                        col = warp * rows_per_warp + (jbase + i) * 4 + cw
                        vv = K.local_scalar("uint32")
                        K.ptx.ld.shared.b32(vv, vtab.ptr_to([t * D + row0 + col]))
                        diff = K.local_scalar("float32")
                        K.ptx["sub.f32"](diff, K.reinterpret("float32", vv), dd1)
                        K.ptx["mul.f32"](uloc[i], diff, beta_t)
                        o = K.local_scalar("float32")
                        K.ptx["fma.rn.f32"](o, c_t, uloc[i], dd2)
                        ob = K.local_scalar("uint16")
                        K.ptx.cvt.rn.bf16.f32(ob, o)
                        K.ptx.st.global_.b16(output.ptr_to([out_base + col]), ob, pred=K.LT(ks, 4))
                    for j in range(J):

                        src = (lane & 0x18) | ((j * 8) // M)
                        uj = K.local_scalar("float32", init=_shfl_idx(uloc[j % R], src))
                        K.assign(u2[j], _f2(uj, uj))

                def update_pass(phase, with_dots, tile):
                    """S += k_{phase-1} u_{phase-1}; checkpoint phase-1 packed to bf16 into staging `tile`; then, unless
                    this is the final pass, S <- alpha_phase S and the two dots of token `phase`. (No proxy fence here:
                    the hand-off of this tile happens one phase later.)"""
                    if with_dots:
                        for j in range(J):
                            K.assign(d1[j], zero2)
                            K.assign(d2[j], zero2)
                    for c in range(2):
                        kp2 = K.alloc_local((4,), "uint64", align=8)
                        a2 = K.alloc_local((4,), "uint64", align=8)
                        k2 = K.alloc_local((4,), "uint64", align=8)
                        q2 = K.alloc_local((4,), "uint64", align=8)
                        for ii in range(4):
                            i = c * 4 + ii
                            if with_dots:
                                K.ptx.ld.shared.v2.b64(kp2[ii], a2[ii], vec_ptr(phase, i))
                                K.ptx.ld.shared.v2.b64(k2[ii], q2[ii], K.ptx.addr(vec_ptr(phase, i), 16))
                            else:
                                K.ptx.ld.shared.b64(kp2[ii], vec_ptr(phase, i))
                        for j in range(J):
                            packed = K.alloc_local((4,), "uint32", align=16)
                            for ii in range(4):
                                i = c * 4 + ii
                                K.ptx.fma.rn.f32x2(S[j * 8 + i], kp2[ii], u2[j], S[j * 8 + i])
                                K.assign(packed[ii], K.cuda.float22bfloat162_rn_from_float2(S[j * 8 + i]))
                                if with_dots:
                                    K.ptx.mul.rn.f32x2(S[j * 8 + i], a2[ii], S[j * 8 + i])
                                    K.ptx.fma.rn.f32x2(d1[j], k2[ii], S[j * 8 + i], d1[j])
                                    K.ptx.fma.rn.f32x2(d2[j], q2[ii], S[j * 8 + i], d2[j])
                            row = warp * rows_per_warp + j * 4 + cw
                            cs = K.bitwise_xor(K.int32(c), ks >> 2)
                            K.ptx.st.shared.v4.b32(
                                tiles.ptr_to([tile * tile_elems + row * D + ks * 16 + cs * 8]),
                                packed[0], packed[1], packed[2], packed[3],
                            )

                def store_warp_rows(tile, dst_slot, hv, row0):
                    """Lane 0 of the calling warp: bulk store of this warp's rows_per_warp rows of `tile`."""
                    wrow = row0 + warp * rows_per_warp
                    dst0 = (K.cast(dst_slot, "int64") * HV + hv) * (D * D) + wrow * D
                    K.ptx[_BULK_S2G](
                        state_out.ptr_to([dst0]),
                        tiles.ptr_to([tile * tile_elems + warp * rows_per_warp * D]),
                        K.uint32(rows_per_warp * D * 2),
                    )
                    K.ptx.cp.async_.bulk.commit_group()

                def early_copy_arm():
                    """Thread 0 after the CTA's last recurrence unit freed tile A: arm the first copy batch as a
                    whole and fetch its first tile now, so the copies overlap the unit's remaining phases."""
                    with K.If(ncopy_me > 0), K.Then():
                        nb0 = K.min(K.int32(ntiles), K.min(ncopy_me, K.int32(MAXC)))
                        K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.cast(nb0, "uint32") * K.uint32(tile_bytes))
                        hv_c, part_c = copy_coords(0)
                        slot_c = K.local_scalar("int32")
                        K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([0]))
                        tile_load(0, slot_c, hv_c, part_c)
                        K.assign(copy_armed, K.int32(1))


                with K.serial(0, nmain_me, unroll=False) as sm:
                    u = cta + sm * num_main
                    head, part, hv, n, h = unit_coords(u)
                    row0 = part * cpt
                    has_next = u + num_main < num_units
                    nxt_slot = K.local_scalar("int32")


                    K.cuda.iket.mark("unit-start")
                    tk = _rng("wait-vec")
                    bar_vec.wait(0, vcount & 1)
                    K.assign(vcount, vcount + K.int32(1))
                    _rng_end(tk)
                    tk = _rng("preprocess")
                    if spec:
                        K.ptx.ld.global_.s32(nxt_slot, ssm_idx.ptr_to([n * T]))
                        with K.If((tid == 0) & has_next), K.Then():
                            K.assign(slot0_next, start_slot(u + num_main, acc_next))
                        if l2pf:
                            with K.If((tid == 0) & (u + 2 * num_main < num_units)), K.Then():
                                K.assign(slot0_next2, start_slot(u + 2 * num_main, acc_next2))
                                prefetch_l2(u + 2 * num_main, slot0_next2)
                    else:
                        K.assign(nxt_slot, n)
                    for rnd in range((T + nw - 1) // nw):
                        t = rnd * nw + warp
                        if rnd * nw + nw <= T:
                            preprocess(t, hv, h)
                        else:
                            with K.If(t < T), K.Then():
                                preprocess(t, hv, h)

                    K.ptx.fence.proxy.async_.shared__cta()
                    K.cuda.cta_sync()
                    _rng_end(tk)
                    with K.If((lane == 0) & has_next & (vec_split or (warp == 0))), K.Then():
                        issue_vec(u + num_main)


                    tk = _rng("wait-state")
                    bar_state.wait(0, scount & 1)
                    K.assign(scount, scount + K.int32(1))
                    _rng_end(tk)
                    tk = _rng("ld-state")
                    ld_state()

                    K.ptx.fence.proxy.async_.shared__cta()
                    K.cuda.cta_sync()
                    _rng_end(tk)
                    with K.If(tid == 0), K.Then():
                        with K.If(has_next):
                            with K.Then():
                                if spec:
                                    issue_state(u + num_main, slot0_next)
                                    with K.If(u + 2 * num_main < num_units), K.Then():
                                        load_acc(u + 2 * num_main)
                                    if l2pf:
                                        with K.If(u + 3 * num_main < num_units), K.Then():
                                            load_acc2(u + 3 * num_main)
                                else:
                                    issue_state(u + num_main, unit_coords(u + num_main)[3])
                                    with K.If(u + 2 * num_main < num_units), K.Then():
                                        prefetch_l2(u + 2 * num_main, unit_coords(u + 2 * num_main)[3])
                            if spec:
                                with K.Else():
                                    early_copy_arm()


                    tk = _rng("phase0")
                    phase0_dots()
                    reduce_and_finish(0, hv, n, row0)
                    _rng_end(tk)

                    issuer = lane == 0 if warp_auto else tid == 0
                    st_slot = K.local_scalar("int32")

                    def group_sync():
                        if warp_auto:
                            K.cuda.warp_sync()
                        else:
                            K.cuda.cta_sync()

                    def issue_store(tile):
                        if warp_auto:
                            store_warp_rows(tile, st_slot, hv, row0)
                        else:
                            tile_store(tile, st_slot, hv, row0)

                    def handoff(prev_tile, wait_free):
                        """Hand off the staging tile written last phase (fence + bulk store); optionally wait until the
                        tile written two checkpoints ago has been read out (it is about to be rewritten)."""
                        K.ptx.fence.proxy.async_.shared__cta()
                        group_sync()
                        with K.If(issuer), K.Then():
                            if prev_tile is not None:
                                issue_store(prev_tile)
                            if wait_free:
                                K.ptx.cp.async_.bulk.wait_group.read(1)
                        if wait_free:
                            group_sync()


                    if T > 1:
                        with K.serial(1, T, unroll=False) as ph:
                            tile = 1 + (ck & 1)
                            tk = _rng("handoff")
                            with K.If(ph >= 2):
                                with K.Then():
                                    handoff(1 + ((ck - 1) & 1), True)
                                with K.Else():
                                    handoff(None, True)
                            _rng_end(tk)
                            tk = _rng("update")
                            update_pass(ph, True, tile)
                            K.assign(st_slot, nxt_slot)
                            if spec:
                                K.ptx.ld.global_.s32(nxt_slot, ssm_idx.ptr_to([n * T + ph]))
                            K.assign(ck, ck + K.int32(1))
                            _rng_end(tk)
                            tk = _rng("reduce")
                            reduce_and_finish(ph, hv, n, row0)
                            _rng_end(tk)

                    tile = 1 + (ck & 1)
                    tk = _rng("handoff")
                    if T > 1:
                        handoff(1 + ((ck - 1) & 1), True)
                    else:
                        handoff(None, True)
                    _rng_end(tk)
                    tk = _rng("update")
                    update_pass(T, False, tile)
                    K.assign(st_slot, nxt_slot)
                    _rng_end(tk)
                    tk = _rng("handoff")
                    handoff(tile, False)
                    K.assign(ck, ck + K.int32(1))
                    _rng_end(tk)


                    K.cuda.cta_sync()


                if spec:
                    with K.If(ncopy_me > 0), K.Then():



                        with K.If(lane == 0), K.Then():
                            K.ptx.cp.async_.bulk.wait_group.read(0)
                        K.cuda.cta_sync()

                        K.ptx.fence.proxy.async_.shared__cta()
                        K.cuda.cta_sync()
                        nfast = K.min(ncopy_me, K.int32(MAXC))
                        with K.serial(0, (nfast + ntiles - 1) // ntiles, unroll=False) as kb:
                            tk = _rng("copy-batch")
                            k0 = kb * ntiles
                            nb = K.min(K.int32(ntiles), nfast - k0)
                            with K.If(tid == 0), K.Then():
                                pre = (kb == 0) & (copy_armed == 1)
                                with K.If(pre == False), K.Then():
                                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.cast(nb, "uint32") * K.uint32(tile_bytes))
                                for i in range(ntiles):

                                    with K.If((i < nb) & ((i > 0) | (pre == False))), K.Then():
                                        hv_c, part_c = copy_coords(k0 + i)
                                        slot_c = K.local_scalar("int32")
                                        K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([k0 + i]))
                                        tile_load(i, slot_c, hv_c, part_c)
                            bar_state.wait(0, scount & 1)
                            K.assign(scount, scount + K.int32(1))
                            with K.If(tid == 0), K.Then():
                                for i in range(ntiles):
                                    with K.If(i < nb), K.Then():
                                        hv_c, part_c = copy_coords(k0 + i)
                                        slot_c = K.local_scalar("int32")
                                        K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([k0 + i]))
                                        tile_store(i, slot_c, hv_c, part_c * cpt)
                                K.ptx.cp.async_.bulk.wait_group.read(0)

                            K.cuda.cta_sync()
                            _rng_end(tk)

                        with K.If(ncopy_me > MAXC), K.Then():
                            K.cuda.cta_sync()
                            with K.serial(MAXC, ncopy_me, unroll=False) as kc:
                                rank = (kc * hole_cnt + (cta - hole_base)) // DCU
                                cnt3 = K.local_scalar("int32", init=K.int32(0))
                                with K.serial(0, (num_slots + pmax - 1) // pmax, unroll=False) as chunk:
                                    cb = chunk * pmax
                                    build_flags(cb)
                                    with K.If(warp == 0), K.Then():
                                        scan_untouched(cb, cnt3, rank, MAXC + 1)
                                    K.cuda.cta_sync()
                                with K.If(tid == 0), K.Then():
                                    hv_c, part_c = copy_coords(kc)
                                    slot_c = K.local_scalar("int32")
                                    K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([MAXC + 1]))
                                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.uint32(tile_bytes))
                                    tile_load(0, slot_c, hv_c, part_c)
                                bar_state.wait(0, scount & 1)
                                K.assign(scount, scount + K.int32(1))
                                with K.If(tid == 0), K.Then():
                                    hv_c, part_c = copy_coords(kc)
                                    slot_c = K.local_scalar("int32")
                                    K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([MAXC + 1]))
                                    tile_store(0, slot_c, hv_c, part_c * cpt)
                                    K.ptx.cp.async_.bulk.wait_group.read(0)
                                K.cuda.cta_sync()
                tk = _rng("drain")
                with K.If(tid == 0), K.Then():
                    K.ptx.cp.async_.bulk.wait_group.read(0)
                _rng_end(tk)

            main_role()

        return kda_decode_persist


    OVERRIDE = {}
    _COMPILED = {}


    def _pick_cpt(N, T, HV):


        heads = N * HV
        if heads <= 16:
            return 32
        return 64


    def _ctas_per_sm(cpt, T, gate_lb, nw=4):
        by_smem = (227 * 1024) // (_smem_bytes(T, cpt, gate_lb, _ntiles(T)) + 1024)
        if nw == 4:
            by_regs = {16: 6, 32: 5, 64: 3, 128: 2}[cpt]
            if T == 1:
                by_regs = {16: 8, 32: 8, 64: 4, 128: 2}[cpt]
        else:
            by_regs = {32: 3, 64: 2, 128: 1}[cpt]
        return max(1, min(by_smem, by_regs))


    def _compile(**kw):
        key = tuple(sorted(kw.items()))
        exe = _COMPILED.get(key)
        if exe is None:
            kernel = build_kernel(**kw)
            target = kernel.target()
            with target:
                exe = tvm.compile(kernel.mod, target=target, tir_pipeline="tirx")
            _COMPILED[key] = exe
        return exe


    def setup(data, N, T):
        q, k, v, g, beta = data["q"], data["k"], data["v"], data["g"], data["beta"]
        initial_state, final_state, output = data["initial_state"], data["final_state"], data["output"]
        H, HV = q.shape[-2], v.shape[-2]
        P = initial_state.shape[0]
        device = q.device
        gate_lb = data["A_log"] is not None
        spec = T > 1
        cpt = int(OVERRIDE.get("cpt", _pick_cpt(N, T, HV)))
        split = D // cpt
        num_units = N * HV * split
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        nw = int(OVERRIDE.get("nw", 4))
        per_sm = int(OVERRIDE.get("ctas_per_sm", _ctas_per_sm(cpt, T, gate_lb, nw)))
        copy_est = max(P - N * T, 0) * HV * split if spec else 0
        num_main = min(num_units + copy_est, per_sm * sms)
        per_sm = min(per_sm, max(1, -(-num_main // sms)))
        extra = {k: bool(OVERRIDE[k]) for k in ("allreduce", "vec_split", "warp_auto") if k in OVERRIDE}
        if "warp_auto" not in extra:
            extra["warp_auto"] = (num_units + copy_est) <= 6 * num_main
        if "vec_split" not in extra:
            extra["vec_split"] = extra["warp_auto"] or (
                T == 6 and num_units + copy_est <= 10 * num_main
            )
        if "ntiles" in OVERRIDE:
            extra["ntiles"] = int(OVERRIDE["ntiles"])
        extra["nw"] = nw
        if "l2pf" in OVERRIDE:
            extra["l2pf"] = bool(OVERRIDE["l2pf"])
        else:


            extra["l2pf"] = T == 2 and num_units >= 8 * num_main
        exe = _compile(T=T, H=H, HV=HV, gate_lb=gate_lb, cpt=cpt, num_units=num_units, num_main=num_main, per_sm=per_sm, copy_est=copy_est, **extra)

        dummy_f32 = torch.zeros(4, dtype=torch.float32, device=device)
        dummy_i32 = torch.zeros(1, dtype=torch.int32, device=device)
        a_log = data["A_log"].contiguous().view(-1).to(torch.float32) if gate_lb else dummy_f32
        dt_bias = data["dt_bias"].contiguous().view(-1).to(torch.float32) if gate_lb else dummy_f32
        ssm = data["ssm_state_indices"].contiguous().view(-1) if spec else dummy_i32
        acc = data["num_accepted_tokens"].contiguous().view(-1) if spec else dummy_i32
        lower_bound = float(data["lower_bound"]) if data["lower_bound"] is not None else 0.0
        scale = float(data["scale"])
        args = (
            q.view(-1), k.view(-1), v.view(-1), g.view(-1), beta.view(-1),
            a_log, dt_bias, initial_state.view(-1), final_state.view(-1), output.view(-1),
            ssm, acc, scale, lower_bound, int(P), int(N),
        )
        keep = (q, k, v, g, beta, a_log, dt_bias, initial_state, final_state, output, ssm, acc)

        def run():
            exe(*args)

        run._keep = keep
        run()
        torch.cuda.synchronize(device)
        return run
    return setup


def _make_dyn():
    """KDA recurrent decode, family "colreg-persist" (v4, warp-autonomous phases): persistent register-resident recurrence.

    Unit = (sequence n, value head hv, column slice of CPT rows). A persistent CTA (4 warps)
    walks units u = cta, cta + num_main, ... . For each unit the fp32 state of its CPT rows x
    128 k stays in registers across all T tokens: lane (ks = lane & 7, cw = lane >> 3) holds
    k-slice [16 ks, 16 ks + 16) of the J = CPT / 16 rows v = warp * CPT/4 + j * 4 + cw, with
    register chunk c holding k-chunk c ^ (ks >> 2) so quarter-warp shared accesses are
    bank-conflict free on the plain row-major staging tiles.

    Pipeline per unit (tile A = input, tiles B/C = checkpoint staging):
      1. wait raw vectors (TMA-prefetched during the previous unit) -> preprocess into the
         per-phase vector table; then issue the raw-vector prefetch of the next unit.
      2. wait the state tile (TMA-prefetched during the previous unit) -> registers; then
         issue the next unit's state load into tile A.
      3. phase 0: S <- alpha_0 S ; d1 += k_0 S ; d2 += q_0 S ; reduce-scatter -> u_0, out_0.
      4. phase p: S <- S + k_{p-1} u_{p-1} (checkpoint p-1 packed to bf16 into a staging tile
         and written with cp.async.bulk), S <- alpha_p S, dots, reduce -> u_p, out_p.
      5. final update + checkpoint T-1.
    Within a unit every warp runs its phases autonomously: its rows are contiguous in the staging
    tile and in the pool slot, so lane 0 of each warp issues the warp's own cp.async.bulk checkpoint
    store after a warp-level sync; CTA barriers remain only at unit boundaries (vector table, tile A).
    Untouched pool slots (the harness poisons final_state before every check) are copied by the
    same persistent CTAs: after its recurrence units a CTA processes tile-sized copy units
    (TMA load into tile A, TMA store out), prefetched exactly like state loads. Each CTA resolves
    the pool slots of its copy units once at start with a warp-parallel ballot scan over the
    touched-slot flags (built in tile B from ssm_state_indices).
    """

    import torch
    import tvm

    import tirx_kernels.kern as K

    D = 128
    LOG2E = 1.4426950408889634
    FULL = 0xFFFFFFFF
    _BULK_G2S = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
    _BULK_S2G = "cp.async.bulk.global.shared::cta.bulk_group"
    FST = 64 * 8 + 8 * 4
    RAW_TOK_B = 4 * 256 + 16
    GATE_B = 512 + 16
    MAXC = 16
    MAXR = 64
    MAXH = 6
    IDX_PRE = 8


    def _shfl_bfly(val_f32, xor):
        out = K.local_scalar("uint32")
        K.ptx.shfl_sync.bfly.b32(out, K.reinterpret("uint32", val_f32), K.uint32(xor), K.uint32(31), K.uint32(FULL))
        return K.reinterpret("float32", out)


    def _shfl_idx(val_f32, src_lane):
        out = K.local_scalar("uint32")
        K.ptx.shfl_sync.idx.b32(out, K.reinterpret("uint32", val_f32), K.cast(src_lane, "uint32"), K.uint32(31), K.uint32(FULL))
        return K.reinterpret("float32", out)


    def _f2(a, b):
        return K.cuda.make_float2(a, b)


    def _rng(name):
        """IKET range token (stripped by the production pipeline)."""
        token = K.alloc_local([1], "uint32")
        K.assign(token[0], K.cuda.iket.range_start(name))
        return token


    def _rng_end(token):
        K.cuda.iket.range_end(token[0])


    def build_kernel(*, T, H, HV, gate_lb, cpt, num_units, num_main, per_sm, copy_est=0, allreduce=None, bar_arrive=False, warp_auto=True, vec_split=True, nw=4, l2pf=False):
        l2pf = False
        static_tickets = bool(OVERRIDE.get("static_tickets", False))
        assert D % cpt == 0 and cpt % (4 * nw) == 0, "each warp needs a multiple of 4 rows"
        G = HV // H
        NT = 32 * nw
        rows_per_warp = cpt // nw
        J = rows_per_warp // 4
        M = 2 * J
        R = max(1, M // 8)
        split = D // cpt
        tile_elems = cpt * D
        tile_bytes = tile_elems * 2
        raw_bytes = T * RAW_TOK_B + (GATE_B if gate_lb else 0)
        raw_elems = raw_bytes // 2
        nphase = T + 1
        load_piece = min(tile_bytes, 16384)
        assert tile_bytes % load_piece == 0
        n_pieces = tile_bytes // load_piece
        num_ctas = num_main
        pmax = tile_bytes // 4
        spec = T > 1
        DCU = HV * split
        if allreduce is None:
            allreduce = J <= 2
        if num_main > num_units:
            hole_base0 = num_units
        else:
            hole_base0 = num_units % num_main
        hole_cnt0 = num_main - hole_base0


        if hole_cnt0 == 0 or copy_est > hole_cnt0 * max(MAXH, T + 1):
            hole_base0, hole_cnt0 = 0, num_main

        @K.kernel(warps=nw, arch="sm_100a", grid=num_ctas, min_blocks_per_sm=per_sm)
        def kda_decode_persist(
            q: K.gptr[K.bf16],
            k: K.gptr[K.bf16],
            v: K.gptr[K.bf16],
            g: K.gptr[K.bf16],
            beta: K.gptr[K.bf16],
            a_log: K.gptr[K.f32],
            dt_bias: K.gptr[K.f32],
            state_in: K.gptr[K.bf16],
            state_out: K.gptr[K.bf16],
            output: K.gptr[K.bf16],
            ssm_idx: K.gptr[K.i32],
            num_accepted: K.gptr[K.i32],
            sched: K.gptr[K.i32],
            scale: K.f32,
            lower_bound: K.f32,
            num_slots: K.i32,
            num_seqs: K.i32,
        ):
            cta = K.cta_id()
            tid = K.thread_id()
            warp = K.warp_id()
            lane = K.lane_id()
            smem = K.smem_pool()
            tiles = smem.alloc((3 * tile_elems,), K.bf16, align=1024)
            raw = smem.alloc((raw_elems,), K.bf16, align=128)
            ftab = smem.alloc((nphase * FST,), K.f32, align=16)
            vtab = smem.alloc((T * D,), K.f32, align=16)
            svec = smem.alloc((2 * T + 2,), K.f32, align=16)
            cpslots = smem.alloc((MAXR + 4,), K.i32, align=16)
            mailbox = smem.alloc((4,), K.i32, align=16)
            bar_state = K.MBarrier(smem, 1)
            bar_state.init(1)
            bar_vec = K.MBarrier(smem, 1)
            bar_vec.init(nw if vec_split else 1)
            K.ptx.fence.proxy.async_.shared__cta()
            K.cuda.cta_sync()


            def main_role():
                ks = lane & 7
                cw = lane >> 3
                nmain_me = K.max((num_units - cta + num_main - 1) // num_main, K.int32(0))
                ck = K.local_scalar("int32", init=K.int32(0))
                scount = K.local_scalar("int32", init=K.int32(0))
                vcount = K.local_scalar("int32", init=K.int32(0))
                slot0_next = K.local_scalar("int32", init=K.int32(0))
                acc_next = K.local_scalar("int32", init=K.int32(1))
                acc_next2 = K.local_scalar("int32", init=K.int32(1))
                slot0_next2 = K.local_scalar("int32", init=K.int32(0))

                def unit_coords(uu):
                    head = uu // split
                    part = uu % split
                    hv = head % HV
                    n = head // HV
                    return head, part, hv, n, hv // G

                def issue_vec(uu):
                    """Lane 0 of warp w: TMA-prefetch tokens t = w, w+4, ... of unit uu's raw q/k/g/v/beta into `raw`;
                    warp 0 also fetches the gate parameters. Each issuing lane arrives with its own byte count."""
                    _, _, hv, n, h = unit_coords(uu)
                    nwi = nw if vec_split else 1
                    for w in range(nwi):
                        toks = [t for t in range(T) if t % nwi == w]
                        nbytes = len(toks) * RAW_TOK_B + ((GATE_B if gate_lb else 0) if w == 0 else 0)
                        with K.If(warp == w), K.Then():
                            K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_vec.ptr_to([0]), K.uint32(nbytes))
                            for t in toks:
                                _issue_vec_token(n, hv, h, t)
                            if gate_lb and w == 0:
                                gb = T * (RAW_TOK_B // 2)
                                K.ptx[_BULK_G2S](raw.ptr_to([gb]), dt_bias.ptr_to([h * D]), K.uint32(512), bar_vec.ptr_to([0]))
                                K.ptx[_BULK_G2S](raw.ptr_to([gb + 256]), a_log.ptr_to([h - (h % 4)]), K.uint32(16), bar_vec.ptr_to([0]))

                def _issue_vec_token(n, hv, h, t):
                    if True:
                        tok = n * T + t
                        qk_off = (K.cast(tok, "int64") * H + h) * D
                        vg_off = (K.cast(tok, "int64") * HV + hv) * D
                        rb = t * (RAW_TOK_B // 2)
                        K.ptx[_BULK_G2S](raw.ptr_to([rb]), q.ptr_to([qk_off]), K.uint32(256), bar_vec.ptr_to([0]))
                        K.ptx[_BULK_G2S](raw.ptr_to([rb + 128]), k.ptr_to([qk_off]), K.uint32(256), bar_vec.ptr_to([0]))
                        K.ptx[_BULK_G2S](raw.ptr_to([rb + 256]), g.ptr_to([vg_off]), K.uint32(256), bar_vec.ptr_to([0]))
                        K.ptx[_BULK_G2S](raw.ptr_to([rb + 384]), v.ptr_to([vg_off]), K.uint32(256), bar_vec.ptr_to([0]))
                        bidx = K.cast(tok, "int64") * HV + hv
                        K.ptx[_BULK_G2S](raw.ptr_to([rb + 512]), beta.ptr_to([bidx - (bidx % 8)]), K.uint32(16), bar_vec.ptr_to([0]))

                def tile_load(tile, slot, hv, part):
                    """Thread 0: TMA load of cpt rows of (slot, hv) from initial_state into `tile` (no expect_tx)."""
                    src0 = (K.cast(slot, "int64") * HV + hv) * (D * D) + part * cpt * D
                    for pc in range(n_pieces):
                        K.ptx[_BULK_G2S](
                            tiles.ptr_to([tile * tile_elems + pc * (load_piece // 2)]),
                            state_in.ptr_to([src0 + pc * (load_piece // 2)]),
                            K.uint32(load_piece),
                            bar_state.ptr_to([0]),
                        )


                def copy_item(cid):
                    """Copy item id (0-based over total_copy) -> (untouched-slot rank, hv, part)."""
                    return cid // DCU, (cid % DCU) // split, cid % split

                def issue_state(uu, slot0):
                    """Thread 0: TMA load of the unit's state rows into tile A."""
                    _, part, hv, n, _ = unit_coords(uu)
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.uint32(tile_bytes))
                    src0 = (K.cast(slot0, "int64") * HV + hv) * (D * D) + part * cpt * D
                    for pc in range(n_pieces):
                        K.ptx[_BULK_G2S](
                            tiles.ptr_to([pc * (load_piece // 2)]),
                            state_in.ptr_to([src0 + pc * (load_piece // 2)]),
                            K.uint32(load_piece),
                            bar_state.ptr_to([0]),
                        )

                def start_slot(uu, acc_val):
                    _, _, _, n, _ = unit_coords(uu)
                    if not spec:
                        return n
                    acc_c = K.max(K.min(acc_val, K.int32(T)), K.int32(1))
                    s = K.local_scalar("int32")
                    K.ptx.ld.global_.s32(s, ssm_idx.ptr_to([n * T + acc_c - 1]))
                    return s

                def load_acc(uu):
                    _, _, _, n, _ = unit_coords(uu)
                    K.ptx.ld.global_.s32(acc_next, num_accepted.ptr_to([n]))

                def load_acc2(uu):
                    _, _, _, n, _ = unit_coords(uu)
                    K.ptx.ld.global_.s32(acc_next2, num_accepted.ptr_to([n]))


                def prefetch_copy(cid):
                    """Thread 0: TMA load of copy item cid's tile into tile A (only for cached ranks)."""
                    rank_c, hv_c, part_c = copy_item(cid)
                    with K.If(rank_c < MAXR), K.Then():
                        slot_c = K.local_scalar("int32")
                        K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([rank_c]))
                        K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.uint32(tile_bytes))
                        tile_load(0, slot_c, hv_c, part_c)

                # ---- dynamic tickets: two items ahead of the (static) first item ------------------------------
                tk1 = K.local_scalar("int32", init=K.int32(0))
                tk2 = K.local_scalar("int32", init=K.int32(0))
                if static_tickets:
                    K.assign(tk1, cta)
                    K.assign(tk2, cta + num_main)
                else:
                    with K.If(tid == 0), K.Then():
                        K.ptx.atom.global_.add.s32(tk1, sched.ptr_to([0]), K.int32(1))
                        K.ptx.atom.global_.add.s32(tk2, sched.ptr_to([0]), K.int32(1))

                if spec:
                    total_idx = num_seqs * T
                    idx_pre = K.alloc_local((IDX_PRE,), "int32")
                    for i in range(IDX_PRE):
                        K.assign(idx_pre[i], K.int32(-1))
                        with K.If(i * NT + tid < total_idx), K.Then():
                            K.ptx.ld.global_.s32(idx_pre[i], ssm_idx.ptr_to([i * NT + tid]))

                with K.If((lane == 0) & (cta < num_units) & (vec_split or (warp == 0))), K.Then():
                    issue_vec(cta)
                with K.If((tid == 0) & (cta < num_units)), K.Then():
                    if spec:
                        n0 = unit_coords(cta)[3]
                        a0 = K.local_scalar("int32")
                        K.ptx.ld.global_.s32(a0, num_accepted.ptr_to([n0]))
                        cands = K.alloc_local((T,), "int32")
                        for t in range(T):
                            K.ptx.ld.global_.s32(cands[t], ssm_idx.ptr_to([n0 * T + t]))
                        acc_c = K.max(K.min(a0, K.int32(T)), K.int32(1))
                        sel = cands[T - 1]
                        for t in range(T - 2, -1, -1):
                            sel = K.if_then_else(acc_c == t + 1, cands[t], sel)
                        issue_state(cta, sel)
                    else:
                        issue_state(cta, unit_coords(cta)[3])

                def build_flags(cb):
                    """All threads: touched-slot flags for slots [cb, cb + pmax) as u32 in tile B."""
                    nz = K.min(K.int32(pmax), num_slots - cb)
                    with K.serial(0, (nz + NT - 1) // NT, unroll=False) as i:
                        pz = i * NT + tid
                        with K.If(pz < nz), K.Then():
                            K.ptx.st.shared.b32(tiles.ptr_to([tile_elems + 2 * pz]), K.uint32(0))
                    K.cuda.cta_sync()
                    total = num_seqs * T
                    for i in range(IDX_PRE):
                        rel = idx_pre[i] - cb
                        with K.If((i * NT + tid < total) & (rel >= 0) & (rel < pmax)), K.Then():
                            K.ptx.st.shared.b32(tiles.ptr_to([tile_elems + 2 * rel]), K.uint32(1))
                    with K.serial(IDX_PRE, (total + NT - 1) // NT, unroll=False) as i:
                        e = i * NT + tid
                        with K.If(e < total), K.Then():
                            sidx = K.local_scalar("int32")
                            K.ptx.ld.global_.s32(sidx, ssm_idx.ptr_to([e]))
                            rel = sidx - cb
                            with K.If((rel >= 0) & (rel < pmax)), K.Then():
                                K.ptx.st.shared.b32(tiles.ptr_to([tile_elems + 2 * rel]), K.uint32(1))
                    K.cuda.cta_sync()

                def scan_untouched(cb, cnt, want_rank, dst_idx):
                    """Warp 0: walk the flags of chunk cb.  want_rank None: cache the slots of ranks < MAXR in
                    cpslots[rank]; otherwise record the slot of `want_rank` into cpslots[dst_idx]."""
                    pend = K.min(pmax, num_slots - cb)
                    with K.serial(0, (pend + 31) // 32, unroll=False) as b:
                        pl = b * 32 + lane
                        valid = pl < pend
                        flag = K.local_scalar("uint32", init=K.uint32(1))
                        with K.If(valid), K.Then():
                            K.ptx.ld.shared.b32(flag, tiles.ptr_to([tile_elems + 2 * pl]))
                        untouched = flag == K.uint32(0)
                        upred = K.local_scalar("uint32", init=K.if_then_else(untouched, K.uint32(1), K.uint32(0)))
                        mask = K.local_scalar("uint32")
                        K.ptx.vote_sync.ballot.b32(mask, K.ptx.pred(upred), K.uint32(FULL))
                        below = K.bitwise_and(mask, (K.uint32(1) << K.cast(lane, "uint32")) - K.uint32(1))
                        rank = cnt + K.cast(K.popcount(below), "int32")
                        with K.If(untouched), K.Then():
                            slot = cb + pl
                            if want_rank is None:
                                with K.If(rank < MAXR), K.Then():
                                    K.ptx.st.shared.b32(cpslots.ptr_to([rank]), K.reinterpret("uint32", slot))
                            else:
                                with K.If((want_rank >= 0) & (rank == want_rank)), K.Then():
                                    K.ptx.st.shared.b32(cpslots.ptr_to([dst_idx]), K.reinterpret("uint32", slot))
                        K.assign(cnt, cnt + K.cast(K.popcount(mask), "int32"))

                total_copy = K.local_scalar("int32", init=K.int32(0))
                if spec:
                    cnt = K.local_scalar("int32", init=K.int32(0))
                    with K.serial(0, (num_slots + pmax - 1) // pmax, unroll=False) as chunk:
                        cb = chunk * pmax
                        build_flags(cb)
                        with K.If(warp == 0), K.Then():
                            scan_untouched(cb, cnt, None, 0)
                        K.cuda.cta_sync()
                    with K.If(tid == 0), K.Then():
                        K.ptx.st.shared.b32(cpslots.ptr_to([MAXR + 1]), K.reinterpret("uint32", cnt))
                        K.ptx.st.shared.b32(mailbox.ptr_to([0]), K.reinterpret("uint32", tk1))
                        K.ptx.st.shared.b32(mailbox.ptr_to([1]), K.reinterpret("uint32", tk2))
                    K.cuda.cta_sync()
                    ucnt = K.local_scalar("int32")
                    K.ptx.ld.shared.b32(ucnt, cpslots.ptr_to([MAXR + 1]))
                    K.assign(total_copy, K.uniform(ucnt) * DCU)
                else:
                    with K.If(tid == 0), K.Then():
                        K.ptx.st.shared.b32(mailbox.ptr_to([0]), K.reinterpret("uint32", tk1))
                        K.ptx.st.shared.b32(mailbox.ptr_to([1]), K.reinterpret("uint32", tk2))
                    K.cuda.cta_sync()
                total_items = num_units + total_copy
                nxt = K.local_scalar("int32")
                nxt2 = K.local_scalar("int32")
                K.ptx.ld.shared.b32(nxt, mailbox.ptr_to([0]))
                K.ptx.ld.shared.b32(nxt2, mailbox.ptr_to([1]))
                K.assign(nxt, K.uniform(nxt) + num_main)
                K.assign(nxt2, K.uniform(nxt2) + num_main)
                cur = K.local_scalar("int32", init=K.uniform(cta))
                with K.If(tid == 0), K.Then():
                    if spec:
                        with K.If(nxt < num_units), K.Then():
                            load_acc(nxt)
                        with K.If((cta >= num_units) & (cta < total_items)), K.Then():
                            prefetch_copy(cta - num_units)

                def preprocess(t, hv, h):
                    """Warp-level: normalize q/k, decay, per-token scalars of token t from `raw`."""
                    rb = t * (RAW_TOK_B // 2)
                    qw = K.alloc_local((2,), "uint32", align=8)
                    kw = K.alloc_local((2,), "uint32", align=8)
                    gw = K.alloc_local((2,), "uint32", align=8)
                    vw = K.alloc_local((2,), "uint32", align=8)
                    K.ptx.ld.shared.v2.b32(qw[0], qw[1], raw.ptr_to([rb + lane * 4]))
                    K.ptx.ld.shared.v2.b32(kw[0], kw[1], raw.ptr_to([rb + 128 + lane * 4]))
                    K.ptx.ld.shared.v2.b32(gw[0], gw[1], raw.ptr_to([rb + 256 + lane * 4]))
                    K.ptx.ld.shared.v2.b32(vw[0], vw[1], raw.ptr_to([rb + 384 + lane * 4]))
                    bbits = K.local_scalar("uint16")
                    K.ptx.ld.shared.b16(bbits, raw.ptr_to([rb + 512 + (hv % 8)]))
                    qf = K.alloc_local((4,), "float32")
                    kf = K.alloc_local((4,), "float32")
                    gf = K.alloc_local((4,), "float32")
                    vf = K.alloc_local((4,), "float32")
                    for src, dst in ((qw, qf), (kw, kf), (gw, gf), (vw, vf)):
                        for p2 in range(2):
                            K.assign(dst[2 * p2], K.cuda.uint_as_float(K.shift_left(src[p2], K.uint32(16))))
                            K.assign(dst[2 * p2 + 1], K.cuda.uint_as_float(K.bitwise_and(src[p2], K.uint32(0xFFFF0000))))

                    sq = K.local_scalar("float32", init=K.float32(0.0))
                    sk = K.local_scalar("float32", init=K.float32(0.0))
                    cpart = K.local_scalar("float32", init=K.float32(0.0))
                    for e in range(4):
                        K.ptx["fma.rn.f32"](sq, qf[e], qf[e], sq)
                        K.ptx["fma.rn.f32"](sk, kf[e], kf[e], sk)
                        K.ptx["fma.rn.f32"](cpart, qf[e], kf[e], cpart)
                    for x in (16, 8, 4, 2, 1):
                        K.ptx["add.f32"](sq, sq, _shfl_bfly(sq, x))
                        K.ptx["add.f32"](sk, sk, _shfl_bfly(sk, x))
                        K.ptx["add.f32"](cpart, cpart, _shfl_bfly(cpart, x))
                    rq = K.local_scalar("float32")
                    rk = K.local_scalar("float32")
                    tmp = K.local_scalar("float32")
                    K.ptx["add.f32"](tmp, sq, K.float32(1e-6))
                    K.ptx["rsqrt.approx.ftz.f32"](rq, tmp)
                    K.ptx["mul.f32"](rq, rq, scale)
                    K.ptx["add.f32"](tmp, sk, K.float32(1e-6))
                    K.ptx["rsqrt.approx.ftz.f32"](rk, tmp)

                    K.ptx["mul.f32"](cpart, cpart, rq)
                    K.ptx["mul.f32"](cpart, cpart, rk)
                    for e in range(4):
                        K.ptx["mul.f32"](qf[e], qf[e], rq)
                        K.ptx["mul.f32"](kf[e], kf[e], rk)
                    if gate_lb:
                        gb = T * (RAW_TOK_B // 2)
                        ea = K.local_scalar("float32")
                        alog = K.local_scalar("uint32")
                        K.ptx.ld.shared.b32(alog, raw.ptr_to([gb + 256 + (h % 4) * 2]))
                        K.ptx["mul.f32"](tmp, K.reinterpret("float32", alog), K.float32(LOG2E))
                        K.ptx["ex2.approx.ftz.f32"](ea, tmp)
                        bias = K.alloc_local((4,), "uint32", align=16)
                        K.ptx.ld.shared.v4.b32(bias[0], bias[1], bias[2], bias[3], raw.ptr_to([gb + lane * 8]))
                        for e in range(4):
                            x = K.local_scalar("float32")
                            K.ptx["add.f32"](x, gf[e], K.reinterpret("float32", bias[e]))
                            K.ptx["mul.f32"](x, x, ea)
                            K.ptx["mul.f32"](x, x, K.float32(-LOG2E))
                            K.ptx["ex2.approx.ftz.f32"](x, x)
                            K.ptx["add.f32"](x, x, K.float32(1.0))
                            K.ptx["rcp.approx.ftz.f32"](x, x)
                            K.ptx["mul.f32"](x, x, lower_bound)
                            K.ptx["mul.f32"](x, x, K.float32(LOG2E))
                            K.ptx["ex2.approx.ftz.f32"](gf[e], x)
                    else:
                        for e in range(4):
                            K.ptx["mul.f32"](gf[e], gf[e], K.float32(LOG2E))
                            K.ptx["ex2.approx.ftz.f32"](gf[e], gf[e])
                    for pp in range(2):
                        pair = lane * 2 + pp
                        ent = t * FST + pair * 8 + (pair >> 3) * 4
                        K.ptx.st.shared.v2.b32(ftab.ptr_to([ent + 2]), K.reinterpret("uint32", gf[2 * pp]), K.reinterpret("uint32", gf[2 * pp + 1]))
                        K.ptx.st.shared.v4.b32(
                            ftab.ptr_to([ent + 4]),
                            K.reinterpret("uint32", kf[2 * pp]), K.reinterpret("uint32", kf[2 * pp + 1]),
                            K.reinterpret("uint32", qf[2 * pp]), K.reinterpret("uint32", qf[2 * pp + 1]),
                        )
                        K.ptx.st.shared.v2.b32(ftab.ptr_to([ent + FST]), K.reinterpret("uint32", kf[2 * pp]), K.reinterpret("uint32", kf[2 * pp + 1]))
                    K.ptx.st.shared.v4.b32(
                        vtab.ptr_to([t * D + lane * 4]),
                        K.reinterpret("uint32", vf[0]), K.reinterpret("uint32", vf[1]),
                        K.reinterpret("uint32", vf[2]), K.reinterpret("uint32", vf[3]),
                    )
                    with K.If(lane == 0), K.Then():
                        bf = K.local_scalar("float32")
                        K.ptx.cvt.f32.bf16(bf, K.cast(bbits, "uint16"))
                        K.ptx.st.shared.v2.b32(svec.ptr_to([2 * t]), K.reinterpret("uint32", bf), K.reinterpret("uint32", cpart))

                S = K.alloc_local((J * 8,), "uint64", align=8)
                d1 = K.alloc_local((J,), "uint64", align=8)
                d2 = K.alloc_local((J,), "uint64", align=8)
                ub = K.alloc_local((J,), "float32")
                zero2 = _f2(K.float32(0.0), K.float32(0.0))

                def vec_ptr(phase, i):
                    cs = K.bitwise_xor(K.int32(i // 4), ks >> 2)
                    pair = ks * 8 + cs * 4 + (i % 4)
                    return ftab.ptr_to([phase * FST + pair * 8 + ks * 4])

                def reduce_and_finish(t, hv, n, row0):
                    """Reduce-scatter d1/d2 over the 8 k-slice lanes; lane owns R columns; broadcast u."""
                    bc = K.alloc_local((2,), "uint32", align=8)
                    K.ptx.ld.shared.v2.b32(bc[0], bc[1], svec.ptr_to([2 * t]))
                    beta_t = K.reinterpret("float32", bc[0])
                    c_t = K.reinterpret("float32", bc[1])
                    vals = []
                    for j in range(J):
                        r = K.local_scalar("float32")
                        K.ptx["add.f32"](r, K.cuda.float2_x(d1[j]), K.cuda.float2_y(d1[j]))
                        vals.append(r)
                    for j in range(J):
                        r = K.local_scalar("float32")
                        K.ptx["add.f32"](r, K.cuda.float2_x(d2[j]), K.cuda.float2_y(d2[j]))
                        vals.append(r)
                    if allreduce:
                        for s in (1, 2, 4):
                            for i in range(M):
                                K.ptx["add.f32"](vals[i], vals[i], _shfl_bfly(vals[i], s))
                        tok = n * T + t
                        out_base = (K.cast(tok, "int64") * HV + hv) * D + row0
                        for j in range(J):
                            col = warp * rows_per_warp + j * 4 + cw
                            vv = K.local_scalar("uint32")
                            K.ptx.ld.shared.b32(vv, vtab.ptr_to([t * D + row0 + col]))
                            diff = K.local_scalar("float32")
                            K.ptx["sub.f32"](diff, K.reinterpret("float32", vv), vals[j])
                            K.ptx["mul.f32"](ub[j], diff, beta_t)
                            o = K.local_scalar("float32")
                            K.ptx["fma.rn.f32"](o, c_t, ub[j], vals[J + j])
                            ob = K.local_scalar("uint16")
                            K.ptx.cvt.rn.bf16.f32(ob, o)
                            K.ptx.st.global_.b16(output.ptr_to([out_base + col]), ob, pred=K.EQ(ks, 0))
                        return
                    for s in (4, 2, 1):
                        if len(vals) > 1:
                            half = len(vals) // 2
                            b = ((ks // s) & 1) != 0
                            nxt = []
                            for i in range(half):
                                send = K.local_scalar("float32", init=K.if_then_else(b, vals[i], vals[i + half]))
                                keep = K.local_scalar("float32", init=K.if_then_else(b, vals[i + half], vals[i]))
                                r = K.local_scalar("float32")
                                K.ptx["add.f32"](r, keep, _shfl_bfly(send, s))
                                nxt.append(r)
                            vals = nxt
                        else:
                            K.ptx["add.f32"](vals[0], vals[0], _shfl_bfly(vals[0], s))
                    assert len(vals) == R

                    is_d2 = (ks >> 2) != 0
                    jbase = ((ks & 3) * M) // 8
                    tok = n * T + t
                    out_base = (K.cast(tok, "int64") * HV + hv) * D + row0
                    uloc = K.alloc_local((R,), "float32")
                    for i in range(R):
                        other = K.local_scalar("float32", init=_shfl_bfly(vals[i], 4))
                        dd1 = K.if_then_else(is_d2, other, vals[i])
                        dd2 = K.if_then_else(is_d2, vals[i], other)
                        col = warp * rows_per_warp + (jbase + i) * 4 + cw
                        vv = K.local_scalar("uint32")
                        K.ptx.ld.shared.b32(vv, vtab.ptr_to([t * D + row0 + col]))
                        diff = K.local_scalar("float32")
                        K.ptx["sub.f32"](diff, K.reinterpret("float32", vv), dd1)
                        K.ptx["mul.f32"](uloc[i], diff, beta_t)
                        o = K.local_scalar("float32")
                        K.ptx["fma.rn.f32"](o, c_t, uloc[i], dd2)
                        ob = K.local_scalar("uint16")
                        K.ptx.cvt.rn.bf16.f32(ob, o)
                        K.ptx.st.global_.b16(output.ptr_to([out_base + col]), ob, pred=K.LT(ks, 4))
                    for j in range(J):

                        src = (lane & 0x18) | ((j * 8) // M)
                        K.assign(ub[j], _shfl_idx(uloc[j % R], src))

                def update_pass(phase, tile, with_dots):
                    if with_dots:
                        for j in range(J):
                            K.assign(d1[j], zero2)
                            K.assign(d2[j], zero2)
                    u2 = K.alloc_local((J,), "uint64", align=8)
                    for j in range(J):
                        K.assign(u2[j], _f2(ub[j], ub[j]))
                    for c in range(2):
                        kp2 = K.alloc_local((4,), "uint64", align=8)
                        a2 = K.alloc_local((4,), "uint64", align=8)
                        k2 = K.alloc_local((4,), "uint64", align=8)
                        q2 = K.alloc_local((4,), "uint64", align=8)
                        for ii in range(4):
                            i = c * 4 + ii
                            if with_dots:
                                K.ptx.ld.shared.v2.b64(kp2[ii], a2[ii], vec_ptr(phase, i))
                                K.ptx.ld.shared.v2.b64(k2[ii], q2[ii], K.ptx.addr(vec_ptr(phase, i), 16))
                            else:
                                K.ptx.ld.shared.b64(kp2[ii], vec_ptr(phase, i))
                        for j in range(J):
                            packed = K.alloc_local((4,), "uint32", align=16)
                            for ii in range(4):
                                i = c * 4 + ii
                                K.ptx.fma.rn.f32x2(S[j * 8 + i], kp2[ii], u2[j], S[j * 8 + i])
                                K.assign(packed[ii], K.cuda.float22bfloat162_rn_from_float2(S[j * 8 + i]))
                                if with_dots:
                                    K.ptx.mul.rn.f32x2(S[j * 8 + i], a2[ii], S[j * 8 + i])
                                    K.ptx.fma.rn.f32x2(d1[j], k2[ii], S[j * 8 + i], d1[j])
                                    K.ptx.fma.rn.f32x2(d2[j], q2[ii], S[j * 8 + i], d2[j])
                            row = warp * rows_per_warp + j * 4 + cw
                            cs = K.bitwise_xor(K.int32(c), ks >> 2)
                            K.ptx.st.shared.v4.b32(
                                tiles.ptr_to([tile * tile_elems + row * D + ks * 16 + cs * 8]),
                                packed[0], packed[1], packed[2], packed[3],
                            )
                    K.ptx.fence.proxy.async_.shared__cta()

                def store_barrier():
                    """All warps' staging writes complete -> warp 0 may issue the TMA store.

                    Warps 1..3 only arrive (named barrier 1, 128 threads) and continue; warp 0 waits.
                    """
                    if bar_arrive:
                        with K.If(warp == 0):
                            with K.Then():
                                K.ptx.bar.sync(K.uint32(1), K.uint32(NT))
                            with K.Else():
                                K.ptx.bar.arrive(K.uint32(1), K.uint32(NT))
                    else:
                        K.cuda.cta_sync()

                def store_checkpoint(tile, dst_slot, hv, row0):
                    dst0 = (K.cast(dst_slot, "int64") * HV + hv) * (D * D) + row0 * D
                    for pc in range(n_pieces):
                        K.ptx[_BULK_S2G](
                            state_out.ptr_to([dst0 + pc * (load_piece // 2)]),
                            tiles.ptr_to([tile * tile_elems + pc * (load_piece // 2)]),
                            K.uint32(load_piece),
                        )
                    K.ptx.cp.async_.bulk.commit_group()

                def store_warp_rows(tile, dst_slot, hv, row0):
                    """Lane 0 of the calling warp: bulk store of this warp's rows_per_warp rows of `tile`."""
                    wrow = row0 + warp * rows_per_warp
                    dst0 = (K.cast(dst_slot, "int64") * HV + hv) * (D * D) + wrow * D
                    K.ptx[_BULK_S2G](
                        state_out.ptr_to([dst0]),
                        tiles.ptr_to([tile * tile_elems + warp * rows_per_warp * D]),
                        K.uint32(rows_per_warp * D * 2),
                    )
                    K.ptx.cp.async_.bulk.commit_group()

                with K.While(cur < total_items):
                    with K.If(cur < num_units):
                        with K.Then():
                            u = cur
                            head, part, hv, n, h = unit_coords(u)
                            row0 = part * cpt
                            has_next = nxt < num_units
                            has_copy_next = (nxt >= num_units) & (nxt < total_items)
                            tk3 = K.local_scalar("int32", init=K.int32(0))
                            if static_tickets:
                                K.assign(tk3, nxt2)
                            else:
                                with K.If(tid == 0), K.Then():
                                    K.ptx.atom.global_.add.s32(tk3, sched.ptr_to([0]), K.int32(1))
                            cur_slot = K.local_scalar("int32")
                            nxt_slot = K.local_scalar("int32")


                            K.cuda.iket.mark("unit-start")
                            tk = _rng("wait-vec")
                            bar_vec.wait(0, vcount & 1)
                            K.assign(vcount, vcount + K.int32(1))
                            _rng_end(tk)
                            tk = _rng("preprocess")
                            with K.If(lane == 0 if warp_auto else tid == 0), K.Then():
                                if spec:
                                    K.ptx.ld.global_.s32(nxt_slot, ssm_idx.ptr_to([n * T]))
                                else:
                                    K.assign(nxt_slot, n)
                            with K.If(tid == 0), K.Then():
                                if spec:
                                    with K.If(has_next), K.Then():
                                        K.assign(slot0_next, start_slot(nxt, acc_next))
                            for rnd in range((T + nw - 1) // nw):
                                t = rnd * nw + warp
                                if rnd * nw + nw <= T:
                                    preprocess(t, hv, h)
                                else:
                                    with K.If(t < T), K.Then():
                                        preprocess(t, hv, h)

                            K.ptx.fence.proxy.async_.shared__cta()
                            K.cuda.cta_sync()
                            _rng_end(tk)
                            with K.If(tid == 0), K.Then():
                                if spec:
                                    with K.If(nxt2 < num_units), K.Then():
                                        load_acc2(nxt2)
                            with K.If((lane == 0) & has_next & (vec_split or (warp == 0))), K.Then():
                                issue_vec(nxt)


                            tk = _rng("wait-state")
                            bar_state.wait(0, scount & 1)
                            K.assign(scount, scount + K.int32(1))
                            _rng_end(tk)
                            tk = _rng("ld-state")
                            for j in range(J):
                                row = warp * rows_per_warp + j * 4 + cw
                                for c in range(2):
                                    wds = K.alloc_local((4,), "uint32", align=16)
                                    cs = K.bitwise_xor(K.int32(c), ks >> 2)
                                    K.ptx.ld.shared.v4.b32(wds[0], wds[1], wds[2], wds[3], tiles.ptr_to([row * D + ks * 16 + cs * 8]))
                                    for i in range(4):
                                        K.assign(
                                            S[j * 8 + c * 4 + i],
                                            _f2(
                                                K.cuda.uint_as_float(K.shift_left(wds[i], K.uint32(16))),
                                                K.cuda.uint_as_float(K.bitwise_and(wds[i], K.uint32(0xFFFF0000))),
                                            ),
                                        )

                            K.ptx.fence.proxy.async_.shared__cta()
                            K.cuda.cta_sync()
                            _rng_end(tk)
                            with K.If(tid == 0), K.Then():
                                with K.If(has_next):
                                    with K.Then():
                                        if spec:
                                            issue_state(nxt, slot0_next)
                                        else:
                                            issue_state(nxt, unit_coords(nxt)[3])
                                    if spec:
                                        with K.Else():
                                            with K.If(has_copy_next), K.Then():
                                                prefetch_copy(nxt - num_units)


                            tk = _rng("phase0")
                            for j in range(J):
                                K.assign(d1[j], zero2)
                                K.assign(d2[j], zero2)
                            for i in range(8):
                                a2 = K.local_scalar("uint64")
                                k2 = K.local_scalar("uint64")
                                q2 = K.local_scalar("uint64")
                                K.ptx.ld.shared.b64(a2, K.ptx.addr(vec_ptr(0, i), 8))
                                K.ptx.ld.shared.v2.b64(k2, q2, K.ptx.addr(vec_ptr(0, i), 16))
                                for j in range(J):
                                    K.ptx.mul.rn.f32x2(S[j * 8 + i], a2, S[j * 8 + i])
                                    K.ptx.fma.rn.f32x2(d1[j], k2, S[j * 8 + i], d1[j])
                                    K.ptx.fma.rn.f32x2(d2[j], q2, S[j * 8 + i], d2[j])
                            reduce_and_finish(0, hv, n, row0)
                            _rng_end(tk)


                            issuer = lane == 0 if warp_auto else tid == 0

                            def tile_free_wait():
                                with K.If(issuer), K.Then():
                                    K.ptx.cp.async_.bulk.wait_group.read(1)
                                if warp_auto:
                                    K.cuda.warp_sync()
                                else:
                                    K.cuda.cta_sync()

                            def store_done_sync():
                                if warp_auto:
                                    K.cuda.warp_sync()
                                else:
                                    K.cuda.cta_sync()

                            def issue_store(tile, slot):
                                if warp_auto:
                                    store_warp_rows(tile, slot, hv, row0)
                                else:
                                    store_checkpoint(tile, slot, hv, row0)

                            if T > 1:
                                with K.serial(1, T, unroll=False) as ph:
                                    tile = 1 + (ck & 1)
                                    tk = _rng("wait-tile")
                                    tile_free_wait()
                                    _rng_end(tk)
                                    tk = _rng("update")
                                    update_pass(ph, tile, True)
                                    _rng_end(tk)
                                    tk = _rng("sync-store")
                                    store_done_sync()
                                    with K.If(issuer), K.Then():
                                        K.assign(cur_slot, nxt_slot)
                                        if spec:
                                            K.ptx.ld.global_.s32(nxt_slot, ssm_idx.ptr_to([n * T + ph]))
                                        issue_store(tile, cur_slot)
                                    K.assign(ck, ck + K.int32(1))
                                    _rng_end(tk)
                                    tk = _rng("reduce")
                                    reduce_and_finish(ph, hv, n, row0)
                                    _rng_end(tk)

                            tile = 1 + (ck & 1)
                            tk = _rng("wait-tile")
                            tile_free_wait()
                            _rng_end(tk)
                            tk = _rng("update")
                            update_pass(T, tile, False)
                            _rng_end(tk)
                            tk = _rng("sync-store")
                            store_done_sync()
                            with K.If(issuer), K.Then():
                                issue_store(tile, nxt_slot)
                            K.assign(ck, ck + K.int32(1))
                            with K.If(tid == 0), K.Then():
                                K.ptx.st.shared.b32(mailbox.ptr_to([2]), K.reinterpret("uint32", tk3))
                            K.cuda.cta_sync()
                            _rng_end(tk)


                        if spec:
                            with K.Else():
                                cid = cur - num_units
                                rank_c, hv_c, part_c = copy_item(cid)
                                tk3 = K.local_scalar("int32", init=K.int32(0))
                                if static_tickets:
                                    K.assign(tk3, nxt2)
                                else:
                                    with K.If(tid == 0), K.Then():
                                        K.ptx.atom.global_.add.s32(tk3, sched.ptr_to([0]), K.int32(1))
                                tk = _rng("copy-batch")
                                with K.If(rank_c >= MAXR), K.Then():
                                    cnt3 = K.local_scalar("int32", init=K.int32(0))
                                    with K.serial(0, (num_slots + pmax - 1) // pmax, unroll=False) as chunk:
                                        cb = chunk * pmax
                                        build_flags(cb)
                                        with K.If(warp == 0), K.Then():
                                            scan_untouched(cb, cnt3, rank_c, MAXR)
                                        K.cuda.cta_sync()
                                    with K.If(tid == 0), K.Then():
                                        slot_c = K.local_scalar("int32")
                                        K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([MAXR]))
                                        K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.uint32(tile_bytes))
                                        tile_load(0, slot_c, hv_c, part_c)
                                bar_state.wait(0, scount & 1)
                                K.assign(scount, scount + K.int32(1))
                                K.cuda.cta_sync()
                                with K.If(tid == 0), K.Then():
                                    slot_c = K.local_scalar("int32")
                                    K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([K.min(rank_c, K.int32(MAXR))]))
                                    store_checkpoint(0, slot_c, hv_c, part_c * cpt)
                                    K.ptx.cp.async_.bulk.wait_group.read(0)
                                    with K.If(nxt < total_items), K.Then():
                                        prefetch_copy(nxt - num_units)
                                    K.ptx.st.shared.b32(mailbox.ptr_to([2]), K.reinterpret("uint32", tk3))
                                K.cuda.cta_sync()
                                _rng_end(tk)
                    K.assign(cur, nxt)
                    K.assign(nxt, nxt2)
                    K.ptx.ld.shared.b32(nxt2, mailbox.ptr_to([2]))
                    K.assign(nxt2, K.uniform(nxt2) + num_main)
                    if spec:
                        K.assign(acc_next, acc_next2)
                tk = _rng("drain")
                with K.If(lane == 0), K.Then():
                    K.ptx.cp.async_.bulk.wait_group.read(0)
                _rng_end(tk)
                K.cuda.cta_sync()
                with K.If(tid == 0), K.Then():
                    done = K.local_scalar("int32")
                    K.ptx.atom.acq_rel.gpu.global_.add.s32(done, sched.ptr_to([1]), K.int32(1))
                    with K.If(done == K.int32(num_main - 1)), K.Then():
                        K.ptx.st.global_.s32(sched.ptr_to([0]), K.int32(0))
                        K.ptx.st.global_.s32(sched.ptr_to([1]), K.int32(0))

            main_role()

        return kda_decode_persist


    OVERRIDE = {}
    _COMPILED = {}


    def _pick_cpt(N, T, HV):


        heads = N * HV
        if heads <= 16:
            return 32
        return 64


    def _ctas_per_sm(cpt, T, gate_lb, nw=4):
        smem = 3 * cpt * D * 2 + T * RAW_TOK_B + (GATE_B if gate_lb else 0) + (T + 1) * FST * 4 + T * D * 4 + 256
        by_smem = (220 * 1024) // smem
        if nw == 4:
            by_regs = {16: 6, 32: 5, 64: 3, 128: 2}[cpt]
            if T == 1:
                by_regs = {16: 8, 32: 8, 64: 4, 128: 2}[cpt]
        else:
            by_regs = {32: 3, 64: 2, 128: 1}[cpt]
        return max(1, min(by_smem, by_regs))


    def _compile(**kw):
        key = tuple(sorted(kw.items()))
        exe = _COMPILED.get(key)
        if exe is None:
            kernel = build_kernel(**kw)
            target = kernel.target()
            with target:
                exe = tvm.compile(kernel.mod, target=target, tir_pipeline="tirx")
            _COMPILED[key] = exe
        return exe


    def setup(data, N, T):
        q, k, v, g, beta = data["q"], data["k"], data["v"], data["g"], data["beta"]
        initial_state, final_state, output = data["initial_state"], data["final_state"], data["output"]
        H, HV = q.shape[-2], v.shape[-2]
        P = initial_state.shape[0]
        device = q.device
        gate_lb = data["A_log"] is not None
        spec = T > 1
        cpt = int(OVERRIDE.get("cpt", _pick_cpt(N, T, HV)))
        split = D // cpt
        num_units = N * HV * split
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        nw = int(OVERRIDE.get("nw", 4))
        per_sm = int(OVERRIDE.get("ctas_per_sm", _ctas_per_sm(cpt, T, gate_lb, nw)))
        copy_est = max(P - N * T, 0) * HV * split if spec else 0
        num_main = min(num_units + copy_est, per_sm * sms)
        per_sm = min(per_sm, max(1, -(-num_main // sms)))
        extra = {k: bool(OVERRIDE[k]) for k in ("allreduce", "bar_arrive", "warp_auto", "vec_split") if k in OVERRIDE}
        if "warp_auto" not in extra:
            extra["warp_auto"] = (num_units + copy_est) <= 6 * num_main
        if "vec_split" not in extra:
            extra["vec_split"] = extra["warp_auto"] or (
                T == 6 and num_units + copy_est <= 10 * num_main
            )
        extra["nw"] = nw
        extra["l2pf"] = False
        exe = _compile(T=T, H=H, HV=HV, gate_lb=gate_lb, cpt=cpt, num_units=num_units, num_main=num_main, per_sm=per_sm, copy_est=copy_est, **extra)

        dummy_f32 = torch.zeros(4, dtype=torch.float32, device=device)
        dummy_i32 = torch.zeros(1, dtype=torch.int32, device=device)
        a_log = data["A_log"].contiguous().view(-1).to(torch.float32) if gate_lb else dummy_f32
        dt_bias = data["dt_bias"].contiguous().view(-1).to(torch.float32) if gate_lb else dummy_f32
        ssm = data["ssm_state_indices"].contiguous().view(-1) if spec else dummy_i32
        acc = data["num_accepted_tokens"].contiguous().view(-1) if spec else dummy_i32
        sched = torch.zeros(4, dtype=torch.int32, device=device)
        lower_bound = float(data["lower_bound"]) if data["lower_bound"] is not None else 0.0
        scale = float(data["scale"])
        args = (
            q.view(-1), k.view(-1), v.view(-1), g.view(-1), beta.view(-1),
            a_log, dt_bias, initial_state.view(-1), final_state.view(-1), output.view(-1),
            ssm, acc, sched, scale, lower_bound, int(P), int(N),
        )
        keep = (q, k, v, g, beta, a_log, dt_bias, initial_state, final_state, output, ssm, acc, sched)

        def run():
            exe(*args)

        run._keep = keep
        run()
        torch.cuda.synchronize(device)
        return run
    return setup




_DEFER_SETUP = _make_defer()
del _make_defer
_DYN_SETUP = _make_dyn()
del _make_dyn

DYN_RULE = {4: 8.0, 2: 16.0}


def _items_per_cta(data, N, T):
    import torch
    HV = data["v"].shape[-2]
    P = data["initial_state"].shape[0]
    cpt = 32 if N * HV <= 16 else 64
    split = 128 // cpt
    units = N * HV * split
    copies = max(P - N * T, 0) * HV * split if T > 1 else 0
    sms = torch.cuda.get_device_properties(data["q"].device).multi_processor_count
    num_main = min(units + copies, 3 * sms)
    return (units + copies) / max(1, num_main)


def _incumbent_setup(data, N, T):
    if T in DYN_RULE and _items_per_cta(data, N, T) >= DYN_RULE[T]:
        return _DYN_SETUP(data, N, T)
    return (_DEFER_SETUP if T == 6 else _BASE_SETUP)(data, N, T)


"""KDA recurrent decode, family colreg-persist (split-tail candidate): the v4c persistent column-register
schedule where the partial last round of 64-row units is processed as 32-row half units (same tokens, same
vector tables, half the rows and registers), so no CTA carries a whole extra unit as its tail; copies go to the
CTAs without a half unit.  Half units reuse the phase code through a geometry factory (J rows per lane group).
"""

def _make_split():
    """KDA recurrent decode, family "colreg-persist" (v4, warp-autonomous phases): persistent register-resident recurrence.

    Unit = (sequence n, value head hv, column slice of CPT rows). A persistent CTA (4 warps)
    walks units u = cta, cta + num_main, ... . For each unit the fp32 state of its CPT rows x
    128 k stays in registers across all T tokens: lane (ks = lane & 7, cw = lane >> 3) holds
    k-slice [16 ks, 16 ks + 16) of the J = CPT / 16 rows v = warp * CPT/4 + j * 4 + cw, with
    register chunk c holding k-chunk c ^ (ks >> 2) so quarter-warp shared accesses are
    bank-conflict free on the plain row-major staging tiles.

    Pipeline per unit (tile A = input, tiles B/C = checkpoint staging):
      1. wait raw vectors (TMA-prefetched during the previous unit) -> preprocess into the
         per-phase vector table; then issue the raw-vector prefetch of the next unit.
      2. wait the state tile (TMA-prefetched during the previous unit) -> registers; then
         issue the next unit's state load into tile A.
      3. phase 0: S <- alpha_0 S ; d1 += k_0 S ; d2 += q_0 S ; reduce-scatter -> u_0, out_0.
      4. phase p: S <- S + k_{p-1} u_{p-1} (checkpoint p-1 packed to bf16 into a staging tile
         and written with cp.async.bulk), S <- alpha_p S, dots, reduce -> u_p, out_p.
      5. final update + checkpoint T-1.
    Within a unit every warp runs its phases autonomously: its rows are contiguous in the staging
    tile and in the pool slot, so lane 0 of each warp issues the warp's own cp.async.bulk checkpoint
    store after a warp-level sync; CTA barriers remain only at unit boundaries (vector table, tile A).
    Untouched pool slots (the harness poisons final_state before every check) are copied by the
    same persistent CTAs: after its recurrence units a CTA processes tile-sized copy units
    (TMA load into tile A, TMA store out), prefetched exactly like state loads. Each CTA resolves
    the pool slots of its copy units once at start with a warp-parallel ballot scan over the
    touched-slot flags (built in tile B from ssm_state_indices).
    """

    import torch
    import tvm

    import tirx_kernels.kern as K

    D = 128
    LOG2E = 1.4426950408889634
    FULL = 0xFFFFFFFF
    _BULK_G2S = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
    _BULK_S2G = "cp.async.bulk.global.shared::cta.bulk_group"
    FST = 64 * 8 + 8 * 4
    RAW_TOK_B = 4 * 256 + 16
    GATE_B = 512 + 16
    MAXC = 16
    MAXH = 6
    IDX_PRE = 8


    def _shfl_bfly(val_f32, xor):
        out = K.local_scalar("uint32")
        K.ptx.shfl_sync.bfly.b32(out, K.reinterpret("uint32", val_f32), K.uint32(xor), K.uint32(31), K.uint32(FULL))
        return K.reinterpret("float32", out)


    def _shfl_idx(val_f32, src_lane):
        out = K.local_scalar("uint32")
        K.ptx.shfl_sync.idx.b32(out, K.reinterpret("uint32", val_f32), K.cast(src_lane, "uint32"), K.uint32(31), K.uint32(FULL))
        return K.reinterpret("float32", out)


    def _f2(a, b):
        return K.cuda.make_float2(a, b)


    def _rng(name):
        """IKET range token (stripped by the production pipeline)."""
        token = K.alloc_local([1], "uint32")
        K.assign(token[0], K.cuda.iket.range_start(name))
        return token


    def _rng_end(token):
        K.cuda.iket.range_end(token[0])


    def build_kernel(*, T, H, HV, gate_lb, cpt, num_units, num_main, per_sm, copy_est=0, allreduce=None, bar_arrive=False, warp_auto=True, vec_split=True, nw=4, l2pf=False):
        assert D % cpt == 0 and cpt % (4 * nw) == 0, "each warp needs a multiple of 4 rows"
        G = HV // H
        NT = 32 * nw
        rows_per_warp = cpt // nw
        J = rows_per_warp // 4
        M = 2 * J
        R = max(1, M // 8)
        JH = J // 2
        assert JH >= 1
        RPW_H = JH * 4
        HALF_ROWS = RPW_H * nw
        HALF_BYTES = HALF_ROWS * D * 2
        MH = 2 * JH
        RH = max(1, MH // 8)
        k_full = num_units // num_main
        F_units = k_full * num_main
        NHALF = 2 * (num_units - F_units)
        l2pf = False
        split = D // cpt
        tile_elems = cpt * D
        tile_bytes = tile_elems * 2
        raw_bytes = T * RAW_TOK_B + (GATE_B if gate_lb else 0)
        raw_elems = raw_bytes // 2
        nphase = T + 1
        load_piece = min(tile_bytes, 16384)
        assert tile_bytes % load_piece == 0
        n_pieces = tile_bytes // load_piece
        num_ctas = num_main
        pmax = tile_bytes // 4
        spec = T > 1
        DCU = HV * split
        if allreduce is None:
            allreduce = J <= 2
        hole_base0 = NHALF % num_main
        hole_cnt0 = num_main - hole_base0


        if hole_cnt0 == 0 or copy_est > hole_cnt0 * max(MAXH, T + 1):
            hole_base0, hole_cnt0 = 0, num_main

        @K.kernel(warps=nw, arch="sm_100a", grid=num_ctas, min_blocks_per_sm=per_sm)
        def kda_decode_persist(
            q: K.gptr[K.bf16],
            k: K.gptr[K.bf16],
            v: K.gptr[K.bf16],
            g: K.gptr[K.bf16],
            beta: K.gptr[K.bf16],
            a_log: K.gptr[K.f32],
            dt_bias: K.gptr[K.f32],
            state_in: K.gptr[K.bf16],
            state_out: K.gptr[K.bf16],
            output: K.gptr[K.bf16],
            ssm_idx: K.gptr[K.i32],
            num_accepted: K.gptr[K.i32],
            scale: K.f32,
            lower_bound: K.f32,
            num_slots: K.i32,
            num_seqs: K.i32,
        ):
            cta = K.cta_id()
            tid = K.thread_id()
            warp = K.warp_id()
            lane = K.lane_id()
            smem = K.smem_pool()
            tiles = smem.alloc((3 * tile_elems,), K.bf16, align=1024)
            raw = smem.alloc((raw_elems,), K.bf16, align=128)
            ftab = smem.alloc((nphase * FST,), K.f32, align=16)
            vtab = smem.alloc((T * D,), K.f32, align=16)
            svec = smem.alloc((2 * T + 2,), K.f32, align=16)
            cpslots = smem.alloc((MAXC + 4,), K.i32, align=16)
            bar_state = K.MBarrier(smem, 1)
            bar_state.init(1)
            bar_vec = K.MBarrier(smem, 1)
            bar_vec.init(nw if vec_split else 1)
            K.ptx.fence.proxy.async_.shared__cta()
            K.cuda.cta_sync()


            def main_role():
                ks = lane & 7
                cw = lane >> 3
                nhalf_me = K.max((K.int32(NHALF) - cta + num_main - 1) // num_main, K.int32(0))
                ck = K.local_scalar("int32", init=K.int32(0))
                scount = K.local_scalar("int32", init=K.int32(0))
                vcount = K.local_scalar("int32", init=K.int32(0))
                slot0_next = K.local_scalar("int32", init=K.int32(0))
                acc_next = K.local_scalar("int32", init=K.int32(1))
                acc_next2 = K.local_scalar("int32", init=K.int32(1))
                slot0_next2 = K.local_scalar("int32", init=K.int32(0))

                def unit_coords(uu):
                    head = uu // split
                    part = uu % split
                    hv = head % HV
                    n = head // HV
                    return head, part, hv, n, hv // G

                def issue_vec(uu):
                    """Lane 0 of warp w: TMA-prefetch tokens t = w, w+4, ... of unit uu's raw q/k/g/v/beta into `raw`;
                    warp 0 also fetches the gate parameters. Each issuing lane arrives with its own byte count."""
                    _, _, hv, n, h = unit_coords(uu)
                    nwi = nw if vec_split else 1
                    for w in range(nwi):
                        toks = [t for t in range(T) if t % nwi == w]
                        nbytes = len(toks) * RAW_TOK_B + ((GATE_B if gate_lb else 0) if w == 0 else 0)
                        with K.If(warp == w), K.Then():
                            K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_vec.ptr_to([0]), K.uint32(nbytes))
                            for t in toks:
                                _issue_vec_token(n, hv, h, t)
                            if gate_lb and w == 0:
                                gb = T * (RAW_TOK_B // 2)
                                K.ptx[_BULK_G2S](raw.ptr_to([gb]), dt_bias.ptr_to([h * D]), K.uint32(512), bar_vec.ptr_to([0]))
                                K.ptx[_BULK_G2S](raw.ptr_to([gb + 256]), a_log.ptr_to([h - (h % 4)]), K.uint32(16), bar_vec.ptr_to([0]))

                def _issue_vec_token(n, hv, h, t):
                    if True:
                        tok = n * T + t
                        qk_off = (K.cast(tok, "int64") * H + h) * D
                        vg_off = (K.cast(tok, "int64") * HV + hv) * D
                        rb = t * (RAW_TOK_B // 2)
                        K.ptx[_BULK_G2S](raw.ptr_to([rb]), q.ptr_to([qk_off]), K.uint32(256), bar_vec.ptr_to([0]))
                        K.ptx[_BULK_G2S](raw.ptr_to([rb + 128]), k.ptr_to([qk_off]), K.uint32(256), bar_vec.ptr_to([0]))
                        K.ptx[_BULK_G2S](raw.ptr_to([rb + 256]), g.ptr_to([vg_off]), K.uint32(256), bar_vec.ptr_to([0]))
                        K.ptx[_BULK_G2S](raw.ptr_to([rb + 384]), v.ptr_to([vg_off]), K.uint32(256), bar_vec.ptr_to([0]))
                        bidx = K.cast(tok, "int64") * HV + hv
                        K.ptx[_BULK_G2S](raw.ptr_to([rb + 512]), beta.ptr_to([bidx - (bidx % 8)]), K.uint32(16), bar_vec.ptr_to([0]))

                def tile_load(tile, slot, hv, part):
                    """Thread 0: TMA load of cpt rows of (slot, hv) from initial_state into `tile` (no expect_tx)."""
                    src0 = (K.cast(slot, "int64") * HV + hv) * (D * D) + part * cpt * D
                    for pc in range(n_pieces):
                        K.ptx[_BULK_G2S](
                            tiles.ptr_to([tile * tile_elems + pc * (load_piece // 2)]),
                            state_in.ptr_to([src0 + pc * (load_piece // 2)]),
                            K.uint32(load_piece),
                            bar_state.ptr_to([0]),
                        )


                hole_base = hole_base0
                hole_cnt = hole_cnt0

                def copy_coords(kc):
                    c = kc * hole_cnt + (cta - hole_base)
                    return (c % DCU) // split, c % split

                def issue_state(uu, slot0):
                    """Thread 0: TMA load of the unit's state rows into tile A."""
                    _, part, hv, n, _ = unit_coords(uu)
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.uint32(tile_bytes))
                    src0 = (K.cast(slot0, "int64") * HV + hv) * (D * D) + part * cpt * D
                    for pc in range(n_pieces):
                        K.ptx[_BULK_G2S](
                            tiles.ptr_to([pc * (load_piece // 2)]),
                            state_in.ptr_to([src0 + pc * (load_piece // 2)]),
                            K.uint32(load_piece),
                            bar_state.ptr_to([0]),
                        )

                def half_coords(hq_id):
                    """Half-unit id hq_id -> (head unit id, part, hv, n, h, row0)."""
                    uu = F_units + hq_id // 2
                    head, part, hv, n, h = unit_coords(uu)
                    return uu, part, hv, n, h, part * cpt + (hq_id % 2) * HALF_ROWS

                def issue_state_half(hq_id, slot0):
                    """Thread 0: TMA load of half-unit hq_id's HALF_ROWS rows into the front of tile A."""
                    _, part, hv, n, _, row0h = half_coords(hq_id)
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.uint32(HALF_BYTES))
                    src0 = (K.cast(slot0, "int64") * HV + hv) * (D * D) + row0h * D
                    K.ptx[_BULK_G2S](tiles.ptr_to([0]), state_in.ptr_to([src0]), K.uint32(HALF_BYTES), bar_state.ptr_to([0]))

                def start_slot(uu, acc_val):
                    _, _, _, n, _ = unit_coords(uu)
                    if not spec:
                        return n
                    acc_c = K.max(K.min(acc_val, K.int32(T)), K.int32(1))
                    s = K.local_scalar("int32")
                    K.ptx.ld.global_.s32(s, ssm_idx.ptr_to([n * T + acc_c - 1]))
                    return s

                def load_acc(uu):
                    _, _, _, n, _ = unit_coords(uu)
                    K.ptx.ld.global_.s32(acc_next, num_accepted.ptr_to([n]))

                def load_acc2(uu):
                    _, _, _, n, _ = unit_coords(uu)
                    K.ptx.ld.global_.s32(acc_next2, num_accepted.ptr_to([n]))

                def prefetch_l2(uu, slot0):
                    """Thread 0: pull the state rows of unit uu (two units ahead) into L2 so the later TMA load
                    does not queue behind the checkpoint write stream in DRAM."""
                    if not l2pf:
                        return
                    _, part, hv, n, _ = unit_coords(uu)
                    src0 = (K.cast(slot0, "int64") * HV + hv) * (D * D) + part * cpt * D
                    K.ptx["cp.async.bulk.prefetch.L2.global"](state_in.ptr_to([src0]), K.uint32(tile_bytes))


                if spec:
                    total_idx = num_seqs * T
                    idx_pre = K.alloc_local((IDX_PRE,), "int32")
                    for i in range(IDX_PRE):
                        K.assign(idx_pre[i], K.int32(-1))
                        with K.If(i * NT + tid < total_idx), K.Then():
                            K.ptx.ld.global_.s32(idx_pre[i], ssm_idx.ptr_to([i * NT + tid]))


                first_head = K.local_scalar("int32", init=cta if k_full > 0 else (F_units + cta // 2))
                first_is_item = (cta < num_units) if k_full > 0 else (cta < NHALF)
                with K.If((lane == 0) & first_is_item & (vec_split or (warp == 0))), K.Then():
                    issue_vec(first_head)
                with K.If((tid == 0) & first_is_item), K.Then():
                    if spec:

                        n0 = unit_coords(first_head)[3]
                        a0 = K.local_scalar("int32")
                        K.ptx.ld.global_.s32(a0, num_accepted.ptr_to([n0]))
                        cands = K.alloc_local((T,), "int32")
                        for t in range(T):
                            K.ptx.ld.global_.s32(cands[t], ssm_idx.ptr_to([n0 * T + t]))
                        acc_c = K.max(K.min(a0, K.int32(T)), K.int32(1))
                        sel = cands[T - 1]
                        for t in range(T - 2, -1, -1):
                            sel = K.if_then_else(acc_c == t + 1, cands[t], sel)
                        if k_full > 0:
                            issue_state(cta, sel)
                        if k_full > 1:
                            with K.If(cta + num_main < num_units), K.Then():
                                load_acc(cta + num_main)
                        elif k_full == 1:
                            with K.If(nhalf_me > 0), K.Then():
                                load_acc(F_units + cta // 2)
                        else:
                            issue_state_half(cta, sel)
                            with K.If(nhalf_me > 1), K.Then():
                                load_acc(F_units + (cta + num_main) // 2)
                    else:
                        if k_full > 0:
                            issue_state(cta, unit_coords(cta)[3])
                        else:
                            issue_state_half(cta, half_coords(cta)[3])

                def build_flags(cb):
                    """All threads: touched-slot flags for slots [cb, cb + pmax) as u32 in tile B."""
                    nz = K.min(K.int32(pmax), num_slots - cb)
                    with K.serial(0, (nz + NT - 1) // NT, unroll=False) as i:
                        pz = i * NT + tid
                        with K.If(pz < nz), K.Then():
                            K.ptx.st.shared.b32(tiles.ptr_to([tile_elems + 2 * pz]), K.uint32(0))
                    K.cuda.cta_sync()
                    total = num_seqs * T
                    for i in range(IDX_PRE):
                        rel = idx_pre[i] - cb
                        with K.If((i * NT + tid < total) & (rel >= 0) & (rel < pmax)), K.Then():
                            K.ptx.st.shared.b32(tiles.ptr_to([tile_elems + 2 * rel]), K.uint32(1))
                    with K.serial(IDX_PRE, (total + NT - 1) // NT, unroll=False) as i:
                        e = i * NT + tid
                        with K.If(e < total), K.Then():
                            sidx = K.local_scalar("int32")
                            K.ptx.ld.global_.s32(sidx, ssm_idx.ptr_to([e]))
                            rel = sidx - cb
                            with K.If((rel >= 0) & (rel < pmax)), K.Then():
                                K.ptx.st.shared.b32(tiles.ptr_to([tile_elems + 2 * rel]), K.uint32(1))
                    K.cuda.cta_sync()

                def scan_untouched(cb, cnt, want_rank, dst_idx):
                    """Warp 0: walk the flags of chunk cb; record the slots of this CTA's copy units
                    (fast path, want_rank None), or the slot of `want_rank` into cpslots[dst_idx]."""
                    pend = K.min(pmax, num_slots - cb)
                    with K.serial(0, (pend + 31) // 32, unroll=False) as b:
                        pl = b * 32 + lane
                        valid = pl < pend
                        flag = K.local_scalar("uint32", init=K.uint32(1))
                        with K.If(valid), K.Then():
                            K.ptx.ld.shared.b32(flag, tiles.ptr_to([tile_elems + 2 * pl]))
                        untouched = flag == K.uint32(0)
                        upred = K.local_scalar("uint32", init=K.if_then_else(untouched, K.uint32(1), K.uint32(0)))
                        mask = K.local_scalar("uint32")
                        K.ptx.vote_sync.ballot.b32(mask, K.ptx.pred(upred), K.uint32(FULL))
                        below = K.bitwise_and(mask, (K.uint32(1) << K.cast(lane, "uint32")) - K.uint32(1))
                        rank = cnt + K.cast(K.popcount(below), "int32")
                        with K.If(untouched), K.Then():
                            slot = cb + pl
                            if want_rank is None:
                                lo = rank * DCU
                                hi = lo + DCU
                                me = cta - hole_base

                                c0 = K.local_scalar("int32", init=lo + ((me - lo) % hole_cnt + hole_cnt) % hole_cnt)
                                with K.If(me >= 0), K.Then():
                                    with K.While(c0 < hi):
                                        kk = (c0 - me) // hole_cnt
                                        with K.If(kk < MAXC), K.Then():
                                            K.ptx.st.shared.b32(cpslots.ptr_to([kk]), K.reinterpret("uint32", slot))
                                        K.assign(c0, c0 + hole_cnt)
                            else:
                                with K.If((want_rank >= 0) & (rank == want_rank)), K.Then():
                                    K.ptx.st.shared.b32(cpslots.ptr_to([dst_idx]), K.reinterpret("uint32", slot))
                        K.assign(cnt, cnt + K.cast(K.popcount(mask), "int32"))

                ncopy_me = K.local_scalar("int32", init=K.int32(0))
                copy_armed = K.local_scalar("int32", init=K.int32(0))
                if spec:
                    with K.If(tid < MAXC + 4), K.Then():
                        K.ptx.st.shared.b32(cpslots.ptr_to([tid]), K.uint32(0))
                    cnt = K.local_scalar("int32", init=K.int32(0))
                    with K.serial(0, (num_slots + pmax - 1) // pmax, unroll=False) as chunk:
                        cb = chunk * pmax
                        build_flags(cb)
                        with K.If(warp == 0), K.Then():
                            scan_untouched(cb, cnt, None, 0)
                        K.cuda.cta_sync()
                    with K.If(tid == 0), K.Then():
                        K.ptx.st.shared.b32(cpslots.ptr_to([MAXC]), K.reinterpret("uint32", cnt))
                    K.cuda.cta_sync()
                    K.ptx.fence.proxy.async_.shared__cta()
                    ucnt = K.local_scalar("int32")
                    K.ptx.ld.shared.b32(ucnt, cpslots.ptr_to([MAXC]))
                    total_copy = ucnt * DCU
                    me0 = cta - hole_base
                    K.assign(ncopy_me, K.if_then_else((me0 >= 0) & (total_copy > me0), (total_copy - me0 + hole_cnt - 1) // hole_cnt, K.int32(0)))

                def preprocess(t, hv, h):
                    """Warp-level: normalize q/k, decay, per-token scalars of token t from `raw`."""
                    rb = t * (RAW_TOK_B // 2)
                    qw = K.alloc_local((2,), "uint32", align=8)
                    kw = K.alloc_local((2,), "uint32", align=8)
                    gw = K.alloc_local((2,), "uint32", align=8)
                    vw = K.alloc_local((2,), "uint32", align=8)
                    K.ptx.ld.shared.v2.b32(qw[0], qw[1], raw.ptr_to([rb + lane * 4]))
                    K.ptx.ld.shared.v2.b32(kw[0], kw[1], raw.ptr_to([rb + 128 + lane * 4]))
                    K.ptx.ld.shared.v2.b32(gw[0], gw[1], raw.ptr_to([rb + 256 + lane * 4]))
                    K.ptx.ld.shared.v2.b32(vw[0], vw[1], raw.ptr_to([rb + 384 + lane * 4]))
                    bbits = K.local_scalar("uint16")
                    K.ptx.ld.shared.b16(bbits, raw.ptr_to([rb + 512 + (hv % 8)]))
                    qf = K.alloc_local((4,), "float32")
                    kf = K.alloc_local((4,), "float32")
                    gf = K.alloc_local((4,), "float32")
                    vf = K.alloc_local((4,), "float32")
                    for src, dst in ((qw, qf), (kw, kf), (gw, gf), (vw, vf)):
                        for p2 in range(2):
                            K.assign(dst[2 * p2], K.cuda.uint_as_float(K.shift_left(src[p2], K.uint32(16))))
                            K.assign(dst[2 * p2 + 1], K.cuda.uint_as_float(K.bitwise_and(src[p2], K.uint32(0xFFFF0000))))

                    sq = K.local_scalar("float32", init=K.float32(0.0))
                    sk = K.local_scalar("float32", init=K.float32(0.0))
                    cpart = K.local_scalar("float32", init=K.float32(0.0))
                    for e in range(4):
                        K.ptx["fma.rn.f32"](sq, qf[e], qf[e], sq)
                        K.ptx["fma.rn.f32"](sk, kf[e], kf[e], sk)
                        K.ptx["fma.rn.f32"](cpart, qf[e], kf[e], cpart)
                    for x in (16, 8, 4, 2, 1):
                        K.ptx["add.f32"](sq, sq, _shfl_bfly(sq, x))
                        K.ptx["add.f32"](sk, sk, _shfl_bfly(sk, x))
                        K.ptx["add.f32"](cpart, cpart, _shfl_bfly(cpart, x))
                    rq = K.local_scalar("float32")
                    rk = K.local_scalar("float32")
                    tmp = K.local_scalar("float32")
                    K.ptx["add.f32"](tmp, sq, K.float32(1e-6))
                    K.ptx["rsqrt.approx.ftz.f32"](rq, tmp)
                    K.ptx["mul.f32"](rq, rq, scale)
                    K.ptx["add.f32"](tmp, sk, K.float32(1e-6))
                    K.ptx["rsqrt.approx.ftz.f32"](rk, tmp)

                    K.ptx["mul.f32"](cpart, cpart, rq)
                    K.ptx["mul.f32"](cpart, cpart, rk)
                    for e in range(4):
                        K.ptx["mul.f32"](qf[e], qf[e], rq)
                        K.ptx["mul.f32"](kf[e], kf[e], rk)
                    if gate_lb:
                        gb = T * (RAW_TOK_B // 2)
                        ea = K.local_scalar("float32")
                        alog = K.local_scalar("uint32")
                        K.ptx.ld.shared.b32(alog, raw.ptr_to([gb + 256 + (h % 4) * 2]))
                        K.ptx["mul.f32"](tmp, K.reinterpret("float32", alog), K.float32(LOG2E))
                        K.ptx["ex2.approx.ftz.f32"](ea, tmp)
                        bias = K.alloc_local((4,), "uint32", align=16)
                        K.ptx.ld.shared.v4.b32(bias[0], bias[1], bias[2], bias[3], raw.ptr_to([gb + lane * 8]))
                        for e in range(4):
                            x = K.local_scalar("float32")
                            K.ptx["add.f32"](x, gf[e], K.reinterpret("float32", bias[e]))
                            K.ptx["mul.f32"](x, x, ea)
                            K.ptx["mul.f32"](x, x, K.float32(-LOG2E))
                            K.ptx["ex2.approx.ftz.f32"](x, x)
                            K.ptx["add.f32"](x, x, K.float32(1.0))
                            K.ptx["rcp.approx.ftz.f32"](x, x)
                            K.ptx["mul.f32"](x, x, lower_bound)
                            K.ptx["mul.f32"](x, x, K.float32(LOG2E))
                            K.ptx["ex2.approx.ftz.f32"](gf[e], x)
                    else:
                        for e in range(4):
                            K.ptx["mul.f32"](gf[e], gf[e], K.float32(LOG2E))
                            K.ptx["ex2.approx.ftz.f32"](gf[e], gf[e])
                    for pp in range(2):
                        pair = lane * 2 + pp
                        ent = t * FST + pair * 8 + (pair >> 3) * 4
                        K.ptx.st.shared.v2.b32(ftab.ptr_to([ent + 2]), K.reinterpret("uint32", gf[2 * pp]), K.reinterpret("uint32", gf[2 * pp + 1]))
                        K.ptx.st.shared.v4.b32(
                            ftab.ptr_to([ent + 4]),
                            K.reinterpret("uint32", kf[2 * pp]), K.reinterpret("uint32", kf[2 * pp + 1]),
                            K.reinterpret("uint32", qf[2 * pp]), K.reinterpret("uint32", qf[2 * pp + 1]),
                        )
                        K.ptx.st.shared.v2.b32(ftab.ptr_to([ent + FST]), K.reinterpret("uint32", kf[2 * pp]), K.reinterpret("uint32", kf[2 * pp + 1]))
                    K.ptx.st.shared.v4.b32(
                        vtab.ptr_to([t * D + lane * 4]),
                        K.reinterpret("uint32", vf[0]), K.reinterpret("uint32", vf[1]),
                        K.reinterpret("uint32", vf[2]), K.reinterpret("uint32", vf[3]),
                    )
                    with K.If(lane == 0), K.Then():
                        bf = K.local_scalar("float32")
                        K.ptx.cvt.f32.bf16(bf, K.cast(bbits, "uint16"))
                        K.ptx.st.shared.v2.b32(svec.ptr_to([2 * t]), K.reinterpret("uint32", bf), K.reinterpret("uint32", cpart))

                S = K.alloc_local((J * 8,), "uint64", align=8)
                d1 = K.alloc_local((J,), "uint64", align=8)
                d2 = K.alloc_local((J,), "uint64", align=8)
                ub = K.alloc_local((J,), "float32")
                zero2 = _f2(K.float32(0.0), K.float32(0.0))

                def make_unit_fns(J, rows_per_warp, M, R, allreduce, ubytes):
                    """Phase code for one unit geometry (J rows per lane group)."""
                    n_pieces_u = (ubytes + load_piece - 1) // load_piece
                    piece_u = min(load_piece, ubytes)
                    def vec_ptr(phase, i):
                        cs = K.bitwise_xor(K.int32(i // 4), ks >> 2)
                        pair = ks * 8 + cs * 4 + (i % 4)
                        return ftab.ptr_to([phase * FST + pair * 8 + ks * 4])

                    def reduce_and_finish(t, hv, n, row0):
                        """Reduce-scatter d1/d2 over the 8 k-slice lanes; lane owns R columns; broadcast u."""
                        bc = K.alloc_local((2,), "uint32", align=8)
                        K.ptx.ld.shared.v2.b32(bc[0], bc[1], svec.ptr_to([2 * t]))
                        beta_t = K.reinterpret("float32", bc[0])
                        c_t = K.reinterpret("float32", bc[1])
                        vals = []
                        for j in range(J):
                            r = K.local_scalar("float32")
                            K.ptx["add.f32"](r, K.cuda.float2_x(d1[j]), K.cuda.float2_y(d1[j]))
                            vals.append(r)
                        for j in range(J):
                            r = K.local_scalar("float32")
                            K.ptx["add.f32"](r, K.cuda.float2_x(d2[j]), K.cuda.float2_y(d2[j]))
                            vals.append(r)
                        if allreduce:
                            for s in (1, 2, 4):
                                for i in range(M):
                                    K.ptx["add.f32"](vals[i], vals[i], _shfl_bfly(vals[i], s))
                            tok = n * T + t
                            out_base = (K.cast(tok, "int64") * HV + hv) * D + row0
                            for j in range(J):
                                col = warp * rows_per_warp + j * 4 + cw
                                vv = K.local_scalar("uint32")
                                K.ptx.ld.shared.b32(vv, vtab.ptr_to([t * D + row0 + col]))
                                diff = K.local_scalar("float32")
                                K.ptx["sub.f32"](diff, K.reinterpret("float32", vv), vals[j])
                                K.ptx["mul.f32"](ub[j], diff, beta_t)
                                o = K.local_scalar("float32")
                                K.ptx["fma.rn.f32"](o, c_t, ub[j], vals[J + j])
                                ob = K.local_scalar("uint16")
                                K.ptx.cvt.rn.bf16.f32(ob, o)
                                K.ptx.st.global_.b16(output.ptr_to([out_base + col]), ob, pred=K.EQ(ks, 0))
                            return
                        for s in (4, 2, 1):
                            if len(vals) > 1:
                                half = len(vals) // 2
                                b = ((ks // s) & 1) != 0
                                nxt = []
                                for i in range(half):
                                    send = K.local_scalar("float32", init=K.if_then_else(b, vals[i], vals[i + half]))
                                    keep = K.local_scalar("float32", init=K.if_then_else(b, vals[i + half], vals[i]))
                                    r = K.local_scalar("float32")
                                    K.ptx["add.f32"](r, keep, _shfl_bfly(send, s))
                                    nxt.append(r)
                                vals = nxt
                            else:
                                K.ptx["add.f32"](vals[0], vals[0], _shfl_bfly(vals[0], s))
                        assert len(vals) == R

                        is_d2 = (ks >> 2) != 0
                        jbase = ((ks & 3) * M) // 8
                        tok = n * T + t
                        out_base = (K.cast(tok, "int64") * HV + hv) * D + row0
                        uloc = K.alloc_local((R,), "float32")
                        for i in range(R):
                            other = K.local_scalar("float32", init=_shfl_bfly(vals[i], 4))
                            dd1 = K.if_then_else(is_d2, other, vals[i])
                            dd2 = K.if_then_else(is_d2, vals[i], other)
                            col = warp * rows_per_warp + (jbase + i) * 4 + cw
                            vv = K.local_scalar("uint32")
                            K.ptx.ld.shared.b32(vv, vtab.ptr_to([t * D + row0 + col]))
                            diff = K.local_scalar("float32")
                            K.ptx["sub.f32"](diff, K.reinterpret("float32", vv), dd1)
                            K.ptx["mul.f32"](uloc[i], diff, beta_t)
                            o = K.local_scalar("float32")
                            K.ptx["fma.rn.f32"](o, c_t, uloc[i], dd2)
                            ob = K.local_scalar("uint16")
                            K.ptx.cvt.rn.bf16.f32(ob, o)
                            K.ptx.st.global_.b16(output.ptr_to([out_base + col]), ob, pred=K.LT(ks, 4))
                        for j in range(J):

                            src = (lane & 0x18) | ((j * 8) // M)
                            K.assign(ub[j], _shfl_idx(uloc[j % R], src))

                    def update_pass(phase, tile, with_dots):
                        if with_dots:
                            for j in range(J):
                                K.assign(d1[j], zero2)
                                K.assign(d2[j], zero2)
                        u2 = K.alloc_local((J,), "uint64", align=8)
                        for j in range(J):
                            K.assign(u2[j], _f2(ub[j], ub[j]))
                        for c in range(2):
                            kp2 = K.alloc_local((4,), "uint64", align=8)
                            a2 = K.alloc_local((4,), "uint64", align=8)
                            k2 = K.alloc_local((4,), "uint64", align=8)
                            q2 = K.alloc_local((4,), "uint64", align=8)
                            for ii in range(4):
                                i = c * 4 + ii
                                if with_dots:
                                    K.ptx.ld.shared.v2.b64(kp2[ii], a2[ii], vec_ptr(phase, i))
                                    K.ptx.ld.shared.v2.b64(k2[ii], q2[ii], K.ptx.addr(vec_ptr(phase, i), 16))
                                else:
                                    K.ptx.ld.shared.b64(kp2[ii], vec_ptr(phase, i))
                            for j in range(J):
                                packed = K.alloc_local((4,), "uint32", align=16)
                                for ii in range(4):
                                    i = c * 4 + ii
                                    K.ptx.fma.rn.f32x2(S[j * 8 + i], kp2[ii], u2[j], S[j * 8 + i])
                                    K.assign(packed[ii], K.cuda.float22bfloat162_rn_from_float2(S[j * 8 + i]))
                                    if with_dots:
                                        K.ptx.mul.rn.f32x2(S[j * 8 + i], a2[ii], S[j * 8 + i])
                                        K.ptx.fma.rn.f32x2(d1[j], k2[ii], S[j * 8 + i], d1[j])
                                        K.ptx.fma.rn.f32x2(d2[j], q2[ii], S[j * 8 + i], d2[j])
                                row = warp * rows_per_warp + j * 4 + cw
                                cs = K.bitwise_xor(K.int32(c), ks >> 2)
                                K.ptx.st.shared.v4.b32(
                                    tiles.ptr_to([tile * tile_elems + row * D + ks * 16 + cs * 8]),
                                    packed[0], packed[1], packed[2], packed[3],
                                )
                        K.ptx.fence.proxy.async_.shared__cta()

                    def store_barrier():
                        """All warps' staging writes complete -> warp 0 may issue the TMA store.

                        Warps 1..3 only arrive (named barrier 1, 128 threads) and continue; warp 0 waits.
                        """
                        if bar_arrive:
                            with K.If(warp == 0):
                                with K.Then():
                                    K.ptx.bar.sync(K.uint32(1), K.uint32(NT))
                                with K.Else():
                                    K.ptx.bar.arrive(K.uint32(1), K.uint32(NT))
                        else:
                            K.cuda.cta_sync()

                    def store_checkpoint(tile, dst_slot, hv, row0):
                        dst0 = (K.cast(dst_slot, "int64") * HV + hv) * (D * D) + row0 * D
                        for pc in range(n_pieces_u):
                            K.ptx[_BULK_S2G](
                                state_out.ptr_to([dst0 + pc * (piece_u // 2)]),
                                tiles.ptr_to([tile * tile_elems + pc * (piece_u // 2)]),
                                K.uint32(piece_u),
                            )
                        K.ptx.cp.async_.bulk.commit_group()

                    def store_warp_rows(tile, dst_slot, hv, row0):
                        """Lane 0 of the calling warp: bulk store of this warp's rows_per_warp rows of `tile`."""
                        wrow = row0 + warp * rows_per_warp
                        dst0 = (K.cast(dst_slot, "int64") * HV + hv) * (D * D) + wrow * D
                        K.ptx[_BULK_S2G](
                            state_out.ptr_to([dst0]),
                            tiles.ptr_to([tile * tile_elems + warp * rows_per_warp * D]),
                            K.uint32(rows_per_warp * D * 2),
                        )
                        K.ptx.cp.async_.bulk.commit_group()


                    def ld_state():
                        for j in range(J):
                            row = warp * rows_per_warp + j * 4 + cw
                            for c in range(2):
                                wds = K.alloc_local((4,), "uint32", align=16)
                                cs = K.bitwise_xor(K.int32(c), ks >> 2)
                                K.ptx.ld.shared.v4.b32(wds[0], wds[1], wds[2], wds[3], tiles.ptr_to([row * D + ks * 16 + cs * 8]))
                                for i in range(4):
                                    K.assign(
                                        S[j * 8 + c * 4 + i],
                                        _f2(
                                            K.cuda.uint_as_float(K.shift_left(wds[i], K.uint32(16))),
                                            K.cuda.uint_as_float(K.bitwise_and(wds[i], K.uint32(0xFFFF0000))),
                                        ),
                                    )

                    def phase0_dots():
                        for j in range(J):
                            K.assign(d1[j], zero2)
                            K.assign(d2[j], zero2)
                        for i in range(8):
                            a2 = K.local_scalar("uint64")
                            k2 = K.local_scalar("uint64")
                            q2 = K.local_scalar("uint64")
                            K.ptx.ld.shared.b64(a2, K.ptx.addr(vec_ptr(0, i), 8))
                            K.ptx.ld.shared.v2.b64(k2, q2, K.ptx.addr(vec_ptr(0, i), 16))
                            for j in range(J):
                                K.ptx.mul.rn.f32x2(S[j * 8 + i], a2, S[j * 8 + i])
                                K.ptx.fma.rn.f32x2(d1[j], k2, S[j * 8 + i], d1[j])
                                K.ptx.fma.rn.f32x2(d2[j], q2, S[j * 8 + i], d2[j])

                    return dict(update_pass=update_pass, reduce_and_finish=reduce_and_finish, store_checkpoint=store_checkpoint,
                                store_warp_rows=store_warp_rows, ld_state=ld_state, phase0_dots=phase0_dots)

                GF = make_unit_fns(J, rows_per_warp, M, R, allreduce, tile_bytes)
                GH = make_unit_fns(JH, RPW_H, MH, RH, JH <= 2, HALF_BYTES)
                store_checkpoint = GF["store_checkpoint"]

                def run_unit(G, uu, row0, hv, n, h, is_half, nxt_kind, nxt_full_id, nxt_half_q, nxt2_kind, nxt2_full_id, nxt2_half_q):
                    """One unit (full or half).  nxt_kind/nxt2_kind: 0 none, 1 full unit, 2 half unit (runtime ints)."""
                    has_next = nxt_kind != 0
                    nxt_head = K.if_then_else(nxt_kind == 2, F_units + nxt_half_q // 2, nxt_full_id)
                    nxt2_head = K.if_then_else(nxt2_kind == 2, F_units + nxt2_half_q // 2, nxt2_full_id)
                    cur_slot = K.local_scalar("int32")
                    nxt_slot = K.local_scalar("int32")
                    K.cuda.iket.mark("unit-start")
                    tk = _rng("wait-vec")
                    bar_vec.wait(0, vcount & 1)
                    K.assign(vcount, vcount + K.int32(1))
                    _rng_end(tk)
                    tk = _rng("preprocess")
                    with K.If(lane == 0 if warp_auto else tid == 0), K.Then():
                        if spec:
                            K.ptx.ld.global_.s32(nxt_slot, ssm_idx.ptr_to([n * T]))
                        else:
                            K.assign(nxt_slot, n)
                    with K.If(tid == 0), K.Then():
                        if spec:
                            with K.If(has_next), K.Then():
                                K.assign(slot0_next, start_slot(nxt_head, acc_next))
                    for rnd in range((T + nw - 1) // nw):
                        t = rnd * nw + warp
                        if rnd * nw + nw <= T:
                            preprocess(t, hv, h)
                        else:
                            with K.If(t < T), K.Then():
                                preprocess(t, hv, h)
                    K.ptx.fence.proxy.async_.shared__cta()
                    K.cuda.cta_sync()
                    _rng_end(tk)
                    with K.If((lane == 0) & has_next & (vec_split or (warp == 0))), K.Then():
                        issue_vec(nxt_head)
                    tk = _rng("wait-state")
                    bar_state.wait(0, scount & 1)
                    K.assign(scount, scount + K.int32(1))
                    _rng_end(tk)
                    tk = _rng("ld-state")
                    G["ld_state"]()
                    K.ptx.fence.proxy.async_.shared__cta()
                    K.cuda.cta_sync()
                    _rng_end(tk)
                    with K.If(tid == 0), K.Then():
                        with K.If(has_next):
                            with K.Then():
                                with K.If(nxt_kind == 1):
                                    with K.Then():
                                        if spec:
                                            issue_state(nxt_full_id, slot0_next)
                                        else:
                                            issue_state(nxt_full_id, unit_coords(nxt_full_id)[3])
                                    with K.Else():
                                        if spec:
                                            issue_state_half(nxt_half_q, slot0_next)
                                        else:
                                            issue_state_half(nxt_half_q, half_coords(nxt_half_q)[3])
                                if spec:
                                    with K.If(nxt2_kind != 0), K.Then():
                                        load_acc(nxt2_head)
                            if spec:
                                with K.Else():
                                    with K.If(ncopy_me > 0), K.Then():
                                        nb0 = K.min(K.int32(3), K.min(ncopy_me, K.int32(MAXC)))
                                        K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.cast(nb0, "uint32") * K.uint32(tile_bytes))
                                        hv_c, part_c = copy_coords(0)
                                        slot_c = K.local_scalar("int32")
                                        K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([0]))
                                        tile_load(0, slot_c, hv_c, part_c)
                                        K.assign(copy_armed, K.int32(1))
                    tk = _rng("phase0")
                    G["phase0_dots"]()
                    G["reduce_and_finish"](0, hv, n, row0)
                    _rng_end(tk)
                    issuer = lane == 0 if warp_auto else tid == 0

                    def tile_free_wait():
                        with K.If(issuer), K.Then():
                            K.ptx.cp.async_.bulk.wait_group.read(1)
                            # This lane issued the S2G copy, so bridge its
                            # completed async read before publishing completion
                            # to the ordinary shared-memory writers below.
                            K.ptx.fence.proxy.async_.shared__cta()
                        if warp_auto:
                            K.cuda.warp_sync()
                        else:
                            K.cuda.cta_sync()

                    def store_done_sync():
                        if warp_auto:
                            K.cuda.warp_sync()
                        else:
                            K.cuda.cta_sync()

                    def issue_store(tile, slot):
                        if warp_auto:
                            G["store_warp_rows"](tile, slot, hv, row0)
                        else:
                            G["store_checkpoint"](tile, slot, hv, row0)

                    if T > 1:
                        with K.serial(1, T, unroll=False) as ph:
                            tile = 1 + (ck & 1)
                            tk = _rng("wait-tile")
                            tile_free_wait()
                            _rng_end(tk)
                            tk = _rng("update")
                            G["update_pass"](ph, tile, True)
                            _rng_end(tk)
                            tk = _rng("sync-store")
                            store_done_sync()
                            with K.If(issuer), K.Then():
                                K.assign(cur_slot, nxt_slot)
                                if spec:
                                    K.ptx.ld.global_.s32(nxt_slot, ssm_idx.ptr_to([n * T + ph]))
                                issue_store(tile, cur_slot)
                            K.assign(ck, ck + K.int32(1))
                            _rng_end(tk)
                            tk = _rng("reduce")
                            G["reduce_and_finish"](ph, hv, n, row0)
                            _rng_end(tk)
                    tile = 1 + (ck & 1)
                    tk = _rng("wait-tile")
                    tile_free_wait()
                    _rng_end(tk)
                    tk = _rng("update")
                    G["update_pass"](T, tile, False)
                    _rng_end(tk)
                    tk = _rng("sync-store")
                    store_done_sync()
                    with K.If(issuer), K.Then():
                        issue_store(tile, nxt_slot)
                    K.assign(ck, ck + K.int32(1))
                    K.cuda.cta_sync()
                    _rng_end(tk)
                    # Full and half geometries assign the same shared rows to
                    # different warps (16 versus 8 rows/warp).  Before the one
                    # full->half handoff, every issuing lane must therefore
                    # retire its S2G reads and publish that completion CTA-wide;
                    # the ordinary per-warp ping-pong wait cannot order a new
                    # owner against the old owner's async queue.
                    if not is_half:
                        with K.If(nxt_kind == 2), K.Then():
                            with K.If(issuer), K.Then():
                                K.ptx.cp.async_.bulk.wait_group.read(0)
                                K.ptx.fence.proxy.async_.shared__cta()
                            K.cuda.cta_sync()

                def kind_after_full(sm_next):
                    """Item kind at full-round index sm_next (>= 0): 1 full, 2 half (first half of this CTA), 0 none."""
                    return K.if_then_else(sm_next < k_full, K.int32(1), K.if_then_else((sm_next == k_full) & (nhalf_me > 0), K.int32(2), K.int32(0)))

                with K.serial(0, k_full, unroll=False) as sm:
                    u = cta + sm * num_main
                    head, part, hv, n, h = unit_coords(u)
                    row0 = part * cpt
                    nk1 = kind_after_full(sm + 1)
                    nk2 = K.if_then_else(sm + 2 < k_full, K.int32(1),
                                         K.if_then_else((sm + 2 == k_full) & (nhalf_me > 0), K.int32(2),
                                                        K.if_then_else((sm + 1 == k_full) & (nhalf_me > 1), K.int32(2), K.int32(0))))
                    nh2 = K.if_then_else(sm + 2 == k_full, cta, cta + num_main)
                    run_unit(GF, u, row0, hv, n, h, False, nk1, u + num_main, cta, nk2, u + 2 * num_main, nh2)
                with K.serial(0, nhalf_me, unroll=False) as hq:
                    half_id = cta + hq * num_main
                    uu, part, hv, n, h, row0h = half_coords(half_id)
                    nk1 = K.if_then_else(hq + 1 < nhalf_me, K.int32(2), K.int32(0))
                    nk2 = K.if_then_else(hq + 2 < nhalf_me, K.int32(2), K.int32(0))
                    run_unit(GH, uu, row0h, hv, n, h, True, nk1, K.int32(0), half_id + num_main, nk2, K.int32(0), half_id + 2 * num_main)

                if spec:
                    with K.If(ncopy_me > 0), K.Then():
                        with K.If(lane == 0), K.Then():
                            K.ptx.cp.async_.bulk.wait_group.read(0)
                        K.cuda.cta_sync()
                        nfast = K.min(ncopy_me, K.int32(MAXC))
                        with K.serial(0, (nfast + 2) // 3, unroll=False) as kb:
                            tk = _rng("copy-batch")
                            k0 = kb * 3
                            nb = K.min(K.int32(3), nfast - k0)
                            with K.If(tid == 0), K.Then():
                                pre = (kb == 0) & (copy_armed == 1)
                                with K.If(pre == False), K.Then():
                                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.cast(nb, "uint32") * K.uint32(tile_bytes))
                                for i in range(3):

                                    with K.If((i < nb) & ((i > 0) | (pre == False))), K.Then():
                                        hv_c, part_c = copy_coords(k0 + i)
                                        slot_c = K.local_scalar("int32")
                                        K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([k0 + i]))
                                        tile_load(i, slot_c, hv_c, part_c)
                            bar_state.wait(0, scount & 1)
                            K.assign(scount, scount + K.int32(1))
                            with K.If(tid == 0), K.Then():
                                for i in range(3):
                                    with K.If(i < nb), K.Then():
                                        hv_c, part_c = copy_coords(k0 + i)
                                        slot_c = K.local_scalar("int32")
                                        K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([k0 + i]))
                                        store_checkpoint(i, slot_c, hv_c, part_c * cpt)
                                K.ptx.cp.async_.bulk.wait_group.read(0)

                            K.cuda.cta_sync()
                            _rng_end(tk)

                        with K.If(ncopy_me > MAXC), K.Then():
                            K.cuda.cta_sync()
                            with K.serial(MAXC, ncopy_me, unroll=False) as kc:
                                rank = (kc * hole_cnt + (cta - hole_base)) // DCU
                                cnt3 = K.local_scalar("int32", init=K.int32(0))
                                with K.serial(0, (num_slots + pmax - 1) // pmax, unroll=False) as chunk:
                                    cb = chunk * pmax
                                    build_flags(cb)
                                    with K.If(warp == 0), K.Then():
                                        scan_untouched(cb, cnt3, rank, MAXC + 1)
                                    K.cuda.cta_sync()
                                with K.If(tid == 0), K.Then():
                                    hv_c, part_c = copy_coords(kc)
                                    slot_c = K.local_scalar("int32")
                                    K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([MAXC + 1]))
                                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.uint32(tile_bytes))
                                    tile_load(0, slot_c, hv_c, part_c)
                                bar_state.wait(0, scount & 1)
                                K.assign(scount, scount + K.int32(1))
                                with K.If(tid == 0), K.Then():
                                    hv_c, part_c = copy_coords(kc)
                                    slot_c = K.local_scalar("int32")
                                    K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([MAXC + 1]))
                                    store_checkpoint(0, slot_c, hv_c, part_c * cpt)
                                    K.ptx.cp.async_.bulk.wait_group.read(0)
                                K.cuda.cta_sync()
                tk = _rng("drain")
                with K.If(lane == 0), K.Then():
                    K.ptx.cp.async_.bulk.wait_group.read(0)
                _rng_end(tk)

            main_role()

        return kda_decode_persist


    OVERRIDE = {}
    _COMPILED = {}


    def _pick_cpt(N, T, HV):


        heads = N * HV
        if heads <= 16:
            return 32
        return 64


    def _ctas_per_sm(cpt, T, gate_lb, nw=4):
        smem = 3 * cpt * D * 2 + T * RAW_TOK_B + (GATE_B if gate_lb else 0) + (T + 1) * FST * 4 + T * D * 4 + 256
        by_smem = (220 * 1024) // smem
        if nw == 4:
            by_regs = {16: 6, 32: 5, 64: 3, 128: 2}[cpt]
            if T == 1:
                by_regs = {16: 8, 32: 8, 64: 4, 128: 2}[cpt]
        else:
            by_regs = {32: 3, 64: 2, 128: 1}[cpt]
        return max(1, min(by_smem, by_regs))


    def _compile(**kw):
        key = tuple(sorted(kw.items()))
        exe = _COMPILED.get(key)
        if exe is None:
            kernel = build_kernel(**kw)
            target = kernel.target()
            with target:
                exe = tvm.compile(kernel.mod, target=target, tir_pipeline="tirx")
            _COMPILED[key] = exe
        return exe


    def setup(data, N, T):
        q, k, v, g, beta = data["q"], data["k"], data["v"], data["g"], data["beta"]
        initial_state, final_state, output = data["initial_state"], data["final_state"], data["output"]
        H, HV = q.shape[-2], v.shape[-2]
        P = initial_state.shape[0]
        device = q.device
        gate_lb = data["A_log"] is not None
        spec = T > 1
        cpt = int(OVERRIDE.get("cpt", _pick_cpt(N, T, HV)))
        split = D // cpt
        num_units = N * HV * split
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        nw = int(OVERRIDE.get("nw", 4))
        per_sm = int(OVERRIDE.get("ctas_per_sm", _ctas_per_sm(cpt, T, gate_lb, nw)))
        copy_est = max(P - N * T, 0) * HV * split if spec else 0
        num_main = min(num_units + copy_est, per_sm * sms)
        per_sm = min(per_sm, max(1, -(-num_main // sms)))
        extra = {k: bool(OVERRIDE[k]) for k in ("allreduce", "bar_arrive", "warp_auto", "vec_split") if k in OVERRIDE}
        if "warp_auto" not in extra:
            extra["warp_auto"] = (num_units + copy_est) <= 6 * num_main
        if "vec_split" not in extra:
            extra["vec_split"] = extra["warp_auto"] or (
                T == 6 and num_units + copy_est <= 10 * num_main
            )
        extra["nw"] = nw
        extra["l2pf"] = False
        exe = _compile(T=T, H=H, HV=HV, gate_lb=gate_lb, cpt=cpt, num_units=num_units, num_main=num_main, per_sm=per_sm, copy_est=copy_est, **extra)

        dummy_f32 = torch.zeros(4, dtype=torch.float32, device=device)
        dummy_i32 = torch.zeros(1, dtype=torch.int32, device=device)
        a_log = data["A_log"].contiguous().view(-1).to(torch.float32) if gate_lb else dummy_f32
        dt_bias = data["dt_bias"].contiguous().view(-1).to(torch.float32) if gate_lb else dummy_f32
        ssm = data["ssm_state_indices"].contiguous().view(-1) if spec else dummy_i32
        acc = data["num_accepted_tokens"].contiguous().view(-1) if spec else dummy_i32
        lower_bound = float(data["lower_bound"]) if data["lower_bound"] is not None else 0.0
        scale = float(data["scale"])
        args = (
            q.view(-1), k.view(-1), v.view(-1), g.view(-1), beta.view(-1),
            a_log, dt_bias, initial_state.view(-1), final_state.view(-1), output.view(-1),
            ssm, acc, scale, lower_bound, int(P), int(N),
        )
        keep = (q, k, v, g, beta, a_log, dt_bias, initial_state, final_state, output, ssm, acc)

        def run():
            exe(*args)

        run._keep = keep
        run()
        torch.cuda.synchronize(device)
        return run
    return setup



_SPLIT_SETUP = _make_split()


def _split_public_setup(data, N, T):
    return _SPLIT_SETUP(data, N, T)


# Same-input paired measurements select only rows where the half-unit route
# reproducibly beats the incumbent.  All other shapes retain the incumbent's
# base, deferred-publication, or ticket-scheduled route.
SPLIT_ROWS = frozenset({(3, 2), (3, 4), (4, 32), (5, 32), (6, 8)})


def _frontier_setup(data, N, T):
    if (T, N) in SPLIT_ROWS:
        return _SPLIT_SETUP(data, N, T)
    return _incumbent_setup(data, N, T)


def _make_clc():
    """KDA recurrent decode, family "colreg-persist" (v4, warp-autonomous phases): persistent register-resident recurrence.

    Unit = (sequence n, value head hv, column slice of CPT rows). A persistent CTA (4 warps)
    walks units u = cta, cta + num_main, ... . For each unit the fp32 state of its CPT rows x
    128 k stays in registers across all T tokens: lane (ks = lane & 7, cw = lane >> 3) holds
    k-slice [16 ks, 16 ks + 16) of the J = CPT / 16 rows v = warp * CPT/4 + j * 4 + cw, with
    register chunk c holding k-chunk c ^ (ks >> 2) so quarter-warp shared accesses are
    bank-conflict free on the plain row-major staging tiles.

    Pipeline per unit (tile A = input, tiles B/C = checkpoint staging):
      1. wait raw vectors (TMA-prefetched during the previous unit) -> preprocess into the
         per-phase vector table; then issue the raw-vector prefetch of the next unit.
      2. wait the state tile (TMA-prefetched during the previous unit) -> registers; then
         issue the next unit's state load into tile A.
      3. phase 0: S <- alpha_0 S ; d1 += k_0 S ; d2 += q_0 S ; reduce-scatter -> u_0, out_0.
      4. phase p: S <- S + k_{p-1} u_{p-1} (checkpoint p-1 packed to bf16 into a staging tile
         and written with cp.async.bulk), S <- alpha_p S, dots, reduce -> u_p, out_p.
      5. final update + checkpoint T-1.
    Within a unit every warp runs its phases autonomously: its rows are contiguous in the staging
    tile and in the pool slot, so lane 0 of each warp issues the warp's own cp.async.bulk checkpoint
    store after a warp-level sync; CTA barriers remain only at unit boundaries (vector table, tile A).
    Untouched pool slots (the harness poisons final_state before every check) are copied by the
    same persistent CTAs: after its recurrence units a CTA processes tile-sized copy units
    (TMA load into tile A, TMA store out), prefetched exactly like state loads. Each CTA resolves
    the pool slots of its copy units once at start with a warp-parallel ballot scan over the
    touched-slot flags (built in tile B from ssm_state_indices).
    """

    import torch
    import tvm

    import tirx_kernels.kern as K

    D = 128
    LOG2E = 1.4426950408889634
    FULL = 0xFFFFFFFF
    _BULK_G2S = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
    _BULK_S2G = "cp.async.bulk.global.shared::cta.bulk_group"
    _TRY_CANCEL = "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128"
    CLC_SENTINEL = 0xFFFFFFFF
    FST = 64 * 8 + 8 * 4
    RAW_TOK_B = 4 * 256 + 16
    GATE_B = 512 + 16
    MAXC = 16
    MAXH = 6
    IDX_PRE = 8


    def _shfl_bfly(val_f32, xor):
        out = K.local_scalar("uint32")
        K.ptx.shfl_sync.bfly.b32(out, K.reinterpret("uint32", val_f32), K.uint32(xor), K.uint32(31), K.uint32(FULL))
        return K.reinterpret("float32", out)


    def _shfl_idx(val_f32, src_lane):
        out = K.local_scalar("uint32")
        K.ptx.shfl_sync.idx.b32(out, K.reinterpret("uint32", val_f32), K.cast(src_lane, "uint32"), K.uint32(31), K.uint32(FULL))
        return K.reinterpret("float32", out)


    def _f2(a, b):
        return K.cuda.make_float2(a, b)


    def _rng(name):
        """IKET range token (stripped by the production pipeline)."""
        token = K.alloc_local([1], "uint32")
        K.assign(token[0], K.cuda.iket.range_start(name))
        return token


    def _rng_end(token):
        K.cuda.iket.range_end(token[0])


    def build_kernel(*, T, H, HV, gate_lb, cpt, num_units, num_main, per_sm, copy_est=0, allreduce=None, bar_arrive=False, warp_auto=True, vec_split=True, nw=4, l2pf=False):
        assert D % cpt == 0 and cpt % (4 * nw) == 0, "each warp needs a multiple of 4 rows"
        G = HV // H
        NT = 32 * nw
        rows_per_warp = cpt // nw
        J = rows_per_warp // 4
        M = 2 * J
        R = max(1, M // 8)
        split = D // cpt
        tile_elems = cpt * D
        tile_bytes = tile_elems * 2
        raw_bytes = T * RAW_TOK_B + (GATE_B if gate_lb else 0)
        raw_elems = raw_bytes // 2
        nphase = T + 1
        load_piece = min(tile_bytes, 16384)
        assert tile_bytes % load_piece == 0
        n_pieces = tile_bytes // load_piece
        num_ctas = num_units
        assert T == 1 and not vec_split
        pmax = tile_bytes // 4
        spec = T > 1
        DCU = HV * split
        if allreduce is None:
            allreduce = J <= 2
        if num_main > num_units:
            hole_base0 = num_units
        else:
            hole_base0 = num_units % num_main
        hole_cnt0 = num_main - hole_base0


        if hole_cnt0 == 0 or copy_est > hole_cnt0 * max(MAXH, T + 1):
            hole_base0, hole_cnt0 = 0, num_main

        @K.kernel(warps=nw, arch="sm_100a", grid=num_ctas, min_blocks_per_sm=per_sm)
        def kda_decode_persist(
            q: K.gptr[K.bf16],
            k: K.gptr[K.bf16],
            v: K.gptr[K.bf16],
            g: K.gptr[K.bf16],
            beta: K.gptr[K.bf16],
            a_log: K.gptr[K.f32],
            dt_bias: K.gptr[K.f32],
            state_in: K.gptr[K.bf16],
            state_out: K.gptr[K.bf16],
            output: K.gptr[K.bf16],
            ssm_idx: K.gptr[K.i32],
            num_accepted: K.gptr[K.i32],
            scale: K.f32,
            lower_bound: K.f32,
            num_slots: K.i32,
            num_seqs: K.i32,
        ):
            cta = K.cta_id()
            tid = K.thread_id()
            warp = K.warp_id()
            lane = K.lane_id()
            smem = K.smem_pool()
            tiles = smem.alloc((3 * tile_elems,), K.bf16, align=1024)
            raw = smem.alloc((2 * raw_elems,), K.bf16, align=128)
            ftab = smem.alloc((nphase * FST,), K.f32, align=16)
            vtab = smem.alloc((T * D,), K.f32, align=16)
            svec = smem.alloc((2 * T + 2,), K.f32, align=16)
            cpslots = smem.alloc((MAXC + 4,), K.i32, align=16)
            bar_state = K.MBarrier(smem, 3)
            bar_state.init(1)
            bar_vec = K.MBarrier(smem, 2)
            bar_vec.init(nw if vec_split else 1)
            clc_handle = smem.pool.alloc((4,), K.u32, align=16)
            clc_mailbox = smem.pool.alloc((1,), K.u32, align=4)
            bar_clc = K.MBarrier(smem, 1)
            bar_clc.init(1)
            K.ptx.fence.proxy.async_.shared__cta()
            K.cuda.cta_sync()


            def main_role():
                ks = lane & 7
                cw = lane >> 3
                nmain_me = K.max((num_units - cta + num_main - 1) // num_main, K.int32(0))
                ck = K.local_scalar("int32", init=K.int32(0))
                scount = K.local_scalar("int32", init=K.int32(0))
                vcount = K.local_scalar("int32", init=K.int32(0))
                slot0_next = K.local_scalar("int32", init=K.int32(0))
                acc_next = K.local_scalar("int32", init=K.int32(1))
                acc_next2 = K.local_scalar("int32", init=K.int32(1))
                slot0_next2 = K.local_scalar("int32", init=K.int32(0))
                clc_phase = K.local_scalar("int32", init=K.int32(0))

                def unit_coords(uu):
                    head = uu // split
                    part = uu % split
                    hv = head % HV
                    n = head // HV
                    return head, part, hv, n, hv // G

                def issue_vec(uu, stage):
                    """Lane 0 of warp w: TMA-prefetch tokens t = w, w+4, ... of unit uu's raw q/k/g/v/beta into `raw`;
                    warp 0 also fetches the gate parameters. Each issuing lane arrives with its own byte count."""
                    _, _, hv, n, h = unit_coords(uu)
                    nwi = nw if vec_split else 1
                    for w in range(nwi):
                        toks = [t for t in range(T) if t % nwi == w]
                        nbytes = len(toks) * RAW_TOK_B + ((GATE_B if gate_lb else 0) if w == 0 else 0)
                        with K.If(warp == w), K.Then():
                            K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_vec.ptr_to([stage]), K.uint32(nbytes))
                            for t in toks:
                                _issue_vec_token(n, hv, h, t, stage)
                            if gate_lb and w == 0:
                                gb = stage * raw_elems + T * (RAW_TOK_B // 2)
                                K.ptx[_BULK_G2S](raw.ptr_to([gb]), dt_bias.ptr_to([h * D]), K.uint32(512), bar_vec.ptr_to([stage]))
                                K.ptx[_BULK_G2S](raw.ptr_to([gb + 256]), a_log.ptr_to([h - (h % 4)]), K.uint32(16), bar_vec.ptr_to([stage]))

                def _issue_vec_token(n, hv, h, t, stage):
                    if True:
                        tok = n * T + t
                        qk_off = (K.cast(tok, "int64") * H + h) * D
                        vg_off = (K.cast(tok, "int64") * HV + hv) * D
                        rb = stage * raw_elems + t * (RAW_TOK_B // 2)
                        K.ptx[_BULK_G2S](raw.ptr_to([rb]), q.ptr_to([qk_off]), K.uint32(256), bar_vec.ptr_to([stage]))
                        K.ptx[_BULK_G2S](raw.ptr_to([rb + 128]), k.ptr_to([qk_off]), K.uint32(256), bar_vec.ptr_to([stage]))
                        K.ptx[_BULK_G2S](raw.ptr_to([rb + 256]), g.ptr_to([vg_off]), K.uint32(256), bar_vec.ptr_to([stage]))
                        K.ptx[_BULK_G2S](raw.ptr_to([rb + 384]), v.ptr_to([vg_off]), K.uint32(256), bar_vec.ptr_to([stage]))
                        bidx = K.cast(tok, "int64") * HV + hv
                        K.ptx[_BULK_G2S](raw.ptr_to([rb + 512]), beta.ptr_to([bidx - (bidx % 8)]), K.uint32(16), bar_vec.ptr_to([stage]))

                def tile_load(tile, slot, hv, part):
                    """Thread 0: TMA load of cpt rows of (slot, hv) from initial_state into `tile` (no expect_tx)."""
                    src0 = (K.cast(slot, "int64") * HV + hv) * (D * D) + part * cpt * D
                    for pc in range(n_pieces):
                        K.ptx[_BULK_G2S](
                            tiles.ptr_to([tile * tile_elems + pc * (load_piece // 2)]),
                            state_in.ptr_to([src0 + pc * (load_piece // 2)]),
                            K.uint32(load_piece),
                            bar_state.ptr_to([0]),
                        )


                hole_base = hole_base0
                hole_cnt = hole_cnt0

                def copy_coords(kc):
                    c = kc * hole_cnt + (cta - hole_base)
                    return (c % DCU) // split, c % split

                def issue_state(uu, slot0, state_stage):
                    """Thread 0: TMA load of the unit's state rows into tile A."""
                    _, part, hv, n, _ = unit_coords(uu)
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([state_stage]), K.uint32(tile_bytes))
                    src0 = (K.cast(slot0, "int64") * HV + hv) * (D * D) + part * cpt * D
                    for pc in range(n_pieces):
                        K.ptx[_BULK_G2S](
                            tiles.ptr_to([state_stage * tile_elems + pc * (load_piece // 2)]),
                            state_in.ptr_to([src0 + pc * (load_piece // 2)]),
                            K.uint32(load_piece),
                            bar_state.ptr_to([state_stage]),
                        )

                def start_slot(uu, acc_val):
                    _, _, _, n, _ = unit_coords(uu)
                    if not spec:
                        return n
                    acc_c = K.max(K.min(acc_val, K.int32(T)), K.int32(1))
                    s = K.local_scalar("int32")
                    K.ptx.ld.global_.s32(s, ssm_idx.ptr_to([n * T + acc_c - 1]))
                    return s

                def load_acc(uu):
                    _, _, _, n, _ = unit_coords(uu)
                    K.ptx.ld.global_.s32(acc_next, num_accepted.ptr_to([n]))

                def load_acc2(uu):
                    _, _, _, n, _ = unit_coords(uu)
                    K.ptx.ld.global_.s32(acc_next2, num_accepted.ptr_to([n]))

                def prefetch_l2(uu, slot0):
                    """Thread 0: pull the state rows of unit uu (two units ahead) into L2 so the later TMA load
                    does not queue behind the checkpoint write stream in DRAM."""
                    if not l2pf:
                        return
                    _, part, hv, n, _ = unit_coords(uu)
                    src0 = (K.cast(slot0, "int64") * HV + hv) * (D * D) + part * cpt * D
                    K.ptx["cp.async.bulk.prefetch.L2.global"](state_in.ptr_to([src0]), K.uint32(tile_bytes))


                if spec:
                    total_idx = num_seqs * T
                    idx_pre = K.alloc_local((IDX_PRE,), "int32")
                    for i in range(IDX_PRE):
                        K.assign(idx_pre[i], K.int32(-1))
                        with K.If(i * NT + tid < total_idx), K.Then():
                            K.ptx.ld.global_.s32(idx_pre[i], ssm_idx.ptr_to([i * NT + tid]))


                with K.If((lane == 0) & (cta < num_units) & (vec_split or (warp == 0))), K.Then():
                    issue_vec(cta, 0)
                with K.If((tid == 0) & (cta < num_units)), K.Then():
                    if spec:

                        n0 = unit_coords(cta)[3]
                        a0 = K.local_scalar("int32")
                        K.ptx.ld.global_.s32(a0, num_accepted.ptr_to([n0]))
                        cands = K.alloc_local((T,), "int32")
                        for t in range(T):
                            K.ptx.ld.global_.s32(cands[t], ssm_idx.ptr_to([n0 * T + t]))
                        acc_c = K.max(K.min(a0, K.int32(T)), K.int32(1))
                        sel = cands[T - 1]
                        for t in range(T - 2, -1, -1):
                            sel = K.if_then_else(acc_c == t + 1, cands[t], sel)
                        issue_state(cta, sel)
                        with K.If(cta + num_main < num_units), K.Then():
                            load_acc(cta + num_main)
                        if l2pf:
                            with K.If(cta + 2 * num_main < num_units), K.Then():
                                load_acc2(cta + 2 * num_main)
                    else:
                        issue_state(cta, unit_coords(cta)[3], 0)
                    K.ptx.fence.proxy.async_.shared__cta()
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_clc.ptr_to([0]), K.uint32(16))
                    K.ptx[_TRY_CANCEL](K.address_of(clc_handle[0]), bar_clc.ptr_to([0]))

                def build_flags(cb):
                    """All threads: touched-slot flags for slots [cb, cb + pmax) as u32 in tile B."""
                    nz = K.min(K.int32(pmax), num_slots - cb)
                    with K.serial(0, (nz + NT - 1) // NT, unroll=False) as i:
                        pz = i * NT + tid
                        with K.If(pz < nz), K.Then():
                            K.ptx.st.shared.b32(tiles.ptr_to([tile_elems + 2 * pz]), K.uint32(0))
                    K.cuda.cta_sync()
                    total = num_seqs * T
                    for i in range(IDX_PRE):
                        rel = idx_pre[i] - cb
                        with K.If((i * NT + tid < total) & (rel >= 0) & (rel < pmax)), K.Then():
                            K.ptx.st.shared.b32(tiles.ptr_to([tile_elems + 2 * rel]), K.uint32(1))
                    with K.serial(IDX_PRE, (total + NT - 1) // NT, unroll=False) as i:
                        e = i * NT + tid
                        with K.If(e < total), K.Then():
                            sidx = K.local_scalar("int32")
                            K.ptx.ld.global_.s32(sidx, ssm_idx.ptr_to([e]))
                            rel = sidx - cb
                            with K.If((rel >= 0) & (rel < pmax)), K.Then():
                                K.ptx.st.shared.b32(tiles.ptr_to([tile_elems + 2 * rel]), K.uint32(1))
                    K.cuda.cta_sync()

                def scan_untouched(cb, cnt, want_rank, dst_idx):
                    """Warp 0: walk the flags of chunk cb; record the slots of this CTA's copy units
                    (fast path, want_rank None), or the slot of `want_rank` into cpslots[dst_idx]."""
                    pend = K.min(pmax, num_slots - cb)
                    with K.serial(0, (pend + 31) // 32, unroll=False) as b:
                        pl = b * 32 + lane
                        valid = pl < pend
                        flag = K.local_scalar("uint32", init=K.uint32(1))
                        with K.If(valid), K.Then():
                            K.ptx.ld.shared.b32(flag, tiles.ptr_to([tile_elems + 2 * pl]))
                        untouched = flag == K.uint32(0)
                        upred = K.local_scalar("uint32", init=K.if_then_else(untouched, K.uint32(1), K.uint32(0)))
                        mask = K.local_scalar("uint32")
                        K.ptx.vote_sync.ballot.b32(mask, K.ptx.pred(upred), K.uint32(FULL))
                        below = K.bitwise_and(mask, (K.uint32(1) << K.cast(lane, "uint32")) - K.uint32(1))
                        rank = cnt + K.cast(K.popcount(below), "int32")
                        with K.If(untouched), K.Then():
                            slot = cb + pl
                            if want_rank is None:
                                lo = rank * DCU
                                hi = lo + DCU
                                me = cta - hole_base

                                c0 = K.local_scalar("int32", init=lo + ((me - lo) % hole_cnt + hole_cnt) % hole_cnt)
                                with K.If(me >= 0), K.Then():
                                    with K.While(c0 < hi):
                                        kk = (c0 - me) // hole_cnt
                                        with K.If(kk < MAXC), K.Then():
                                            K.ptx.st.shared.b32(cpslots.ptr_to([kk]), K.reinterpret("uint32", slot))
                                        K.assign(c0, c0 + hole_cnt)
                            else:
                                with K.If((want_rank >= 0) & (rank == want_rank)), K.Then():
                                    K.ptx.st.shared.b32(cpslots.ptr_to([dst_idx]), K.reinterpret("uint32", slot))
                        K.assign(cnt, cnt + K.cast(K.popcount(mask), "int32"))

                ncopy_me = K.local_scalar("int32", init=K.int32(0))
                copy_armed = K.local_scalar("int32", init=K.int32(0))
                if spec:
                    with K.If(tid < MAXC + 4), K.Then():
                        K.ptx.st.shared.b32(cpslots.ptr_to([tid]), K.uint32(0))
                    cnt = K.local_scalar("int32", init=K.int32(0))
                    with K.serial(0, (num_slots + pmax - 1) // pmax, unroll=False) as chunk:
                        cb = chunk * pmax
                        build_flags(cb)
                        with K.If(warp == 0), K.Then():
                            scan_untouched(cb, cnt, None, 0)
                        K.cuda.cta_sync()
                    with K.If(tid == 0), K.Then():
                        K.ptx.st.shared.b32(cpslots.ptr_to([MAXC]), K.reinterpret("uint32", cnt))
                    K.cuda.cta_sync()
                    ucnt = K.local_scalar("int32")
                    K.ptx.ld.shared.b32(ucnt, cpslots.ptr_to([MAXC]))
                    total_copy = ucnt * DCU
                    me0 = cta - hole_base
                    K.assign(ncopy_me, K.if_then_else((me0 >= 0) & (total_copy > me0), (total_copy - me0 + hole_cnt - 1) // hole_cnt, K.int32(0)))

                def preprocess(t, hv, h, stage):
                    """Warp-level: normalize q/k, decay, per-token scalars of token t from `raw`."""
                    rb = stage * raw_elems + t * (RAW_TOK_B // 2)
                    qw = K.alloc_local((2,), "uint32", align=8)
                    kw = K.alloc_local((2,), "uint32", align=8)
                    gw = K.alloc_local((2,), "uint32", align=8)
                    vw = K.alloc_local((2,), "uint32", align=8)
                    K.ptx.ld.shared.v2.b32(qw[0], qw[1], raw.ptr_to([rb + lane * 4]))
                    K.ptx.ld.shared.v2.b32(kw[0], kw[1], raw.ptr_to([rb + 128 + lane * 4]))
                    K.ptx.ld.shared.v2.b32(gw[0], gw[1], raw.ptr_to([rb + 256 + lane * 4]))
                    K.ptx.ld.shared.v2.b32(vw[0], vw[1], raw.ptr_to([rb + 384 + lane * 4]))
                    bbits = K.local_scalar("uint16")
                    K.ptx.ld.shared.b16(bbits, raw.ptr_to([rb + 512 + (hv % 8)]))
                    qf = K.alloc_local((4,), "float32")
                    kf = K.alloc_local((4,), "float32")
                    gf = K.alloc_local((4,), "float32")
                    vf = K.alloc_local((4,), "float32")
                    for src, dst in ((qw, qf), (kw, kf), (gw, gf), (vw, vf)):
                        for p2 in range(2):
                            K.assign(dst[2 * p2], K.cuda.uint_as_float(K.shift_left(src[p2], K.uint32(16))))
                            K.assign(dst[2 * p2 + 1], K.cuda.uint_as_float(K.bitwise_and(src[p2], K.uint32(0xFFFF0000))))

                    sq = K.local_scalar("float32", init=K.float32(0.0))
                    sk = K.local_scalar("float32", init=K.float32(0.0))
                    cpart = K.local_scalar("float32", init=K.float32(0.0))
                    for e in range(4):
                        K.ptx["fma.rn.f32"](sq, qf[e], qf[e], sq)
                        K.ptx["fma.rn.f32"](sk, kf[e], kf[e], sk)
                        K.ptx["fma.rn.f32"](cpart, qf[e], kf[e], cpart)
                    for x in (16, 8, 4, 2, 1):
                        K.ptx["add.f32"](sq, sq, _shfl_bfly(sq, x))
                        K.ptx["add.f32"](sk, sk, _shfl_bfly(sk, x))
                        K.ptx["add.f32"](cpart, cpart, _shfl_bfly(cpart, x))
                    rq = K.local_scalar("float32")
                    rk = K.local_scalar("float32")
                    tmp = K.local_scalar("float32")
                    K.ptx["add.f32"](tmp, sq, K.float32(1e-6))
                    K.ptx["rsqrt.approx.ftz.f32"](rq, tmp)
                    K.ptx["mul.f32"](rq, rq, scale)
                    K.ptx["add.f32"](tmp, sk, K.float32(1e-6))
                    K.ptx["rsqrt.approx.ftz.f32"](rk, tmp)

                    K.ptx["mul.f32"](cpart, cpart, rq)
                    K.ptx["mul.f32"](cpart, cpart, rk)
                    for e in range(4):
                        K.ptx["mul.f32"](qf[e], qf[e], rq)
                        K.ptx["mul.f32"](kf[e], kf[e], rk)
                    if gate_lb:
                        gb = stage * raw_elems + T * (RAW_TOK_B // 2)
                        ea = K.local_scalar("float32")
                        alog = K.local_scalar("uint32")
                        K.ptx.ld.shared.b32(alog, raw.ptr_to([gb + 256 + (h % 4) * 2]))
                        K.ptx["mul.f32"](tmp, K.reinterpret("float32", alog), K.float32(LOG2E))
                        K.ptx["ex2.approx.ftz.f32"](ea, tmp)
                        bias = K.alloc_local((4,), "uint32", align=16)
                        K.ptx.ld.shared.v4.b32(bias[0], bias[1], bias[2], bias[3], raw.ptr_to([gb + lane * 8]))
                        for e in range(4):
                            x = K.local_scalar("float32")
                            K.ptx["add.f32"](x, gf[e], K.reinterpret("float32", bias[e]))
                            K.ptx["mul.f32"](x, x, ea)
                            K.ptx["mul.f32"](x, x, K.float32(-LOG2E))
                            K.ptx["ex2.approx.ftz.f32"](x, x)
                            K.ptx["add.f32"](x, x, K.float32(1.0))
                            K.ptx["rcp.approx.ftz.f32"](x, x)
                            K.ptx["mul.f32"](x, x, lower_bound)
                            K.ptx["mul.f32"](x, x, K.float32(LOG2E))
                            K.ptx["ex2.approx.ftz.f32"](gf[e], x)
                    else:
                        for e in range(4):
                            K.ptx["mul.f32"](gf[e], gf[e], K.float32(LOG2E))
                            K.ptx["ex2.approx.ftz.f32"](gf[e], gf[e])
                    for pp in range(2):
                        pair = lane * 2 + pp
                        ent = t * FST + pair * 8 + (pair >> 3) * 4
                        K.ptx.st.shared.v2.b32(ftab.ptr_to([ent + 2]), K.reinterpret("uint32", gf[2 * pp]), K.reinterpret("uint32", gf[2 * pp + 1]))
                        K.ptx.st.shared.v4.b32(
                            ftab.ptr_to([ent + 4]),
                            K.reinterpret("uint32", kf[2 * pp]), K.reinterpret("uint32", kf[2 * pp + 1]),
                            K.reinterpret("uint32", qf[2 * pp]), K.reinterpret("uint32", qf[2 * pp + 1]),
                        )
                        K.ptx.st.shared.v2.b32(ftab.ptr_to([ent + FST]), K.reinterpret("uint32", kf[2 * pp]), K.reinterpret("uint32", kf[2 * pp + 1]))
                    K.ptx.st.shared.v4.b32(
                        vtab.ptr_to([t * D + lane * 4]),
                        K.reinterpret("uint32", vf[0]), K.reinterpret("uint32", vf[1]),
                        K.reinterpret("uint32", vf[2]), K.reinterpret("uint32", vf[3]),
                    )
                    with K.If(lane == 0), K.Then():
                        bf = K.local_scalar("float32")
                        K.ptx.cvt.f32.bf16(bf, K.cast(bbits, "uint16"))
                        K.ptx.st.shared.v2.b32(svec.ptr_to([2 * t]), K.reinterpret("uint32", bf), K.reinterpret("uint32", cpart))

                S = K.alloc_local((J * 8,), "uint64", align=8)
                d1 = K.alloc_local((J,), "uint64", align=8)
                d2 = K.alloc_local((J,), "uint64", align=8)
                ub = K.alloc_local((J,), "float32")
                zero2 = _f2(K.float32(0.0), K.float32(0.0))

                def vec_ptr(phase, i):
                    cs = K.bitwise_xor(K.int32(i // 4), ks >> 2)
                    pair = ks * 8 + cs * 4 + (i % 4)
                    return ftab.ptr_to([phase * FST + pair * 8 + ks * 4])

                def reduce_and_finish(t, hv, n, row0):
                    """Reduce-scatter d1/d2 over the 8 k-slice lanes; lane owns R columns; broadcast u."""
                    bc = K.alloc_local((2,), "uint32", align=8)
                    K.ptx.ld.shared.v2.b32(bc[0], bc[1], svec.ptr_to([2 * t]))
                    beta_t = K.reinterpret("float32", bc[0])
                    c_t = K.reinterpret("float32", bc[1])
                    vals = []
                    for j in range(J):
                        r = K.local_scalar("float32")
                        K.ptx["add.f32"](r, K.cuda.float2_x(d1[j]), K.cuda.float2_y(d1[j]))
                        vals.append(r)
                    for j in range(J):
                        r = K.local_scalar("float32")
                        K.ptx["add.f32"](r, K.cuda.float2_x(d2[j]), K.cuda.float2_y(d2[j]))
                        vals.append(r)
                    if allreduce:
                        for s in (1, 2, 4):
                            for i in range(M):
                                K.ptx["add.f32"](vals[i], vals[i], _shfl_bfly(vals[i], s))
                        tok = n * T + t
                        out_base = (K.cast(tok, "int64") * HV + hv) * D + row0
                        for j in range(J):
                            col = warp * rows_per_warp + j * 4 + cw
                            vv = K.local_scalar("uint32")
                            K.ptx.ld.shared.b32(vv, vtab.ptr_to([t * D + row0 + col]))
                            diff = K.local_scalar("float32")
                            K.ptx["sub.f32"](diff, K.reinterpret("float32", vv), vals[j])
                            K.ptx["mul.f32"](ub[j], diff, beta_t)
                            o = K.local_scalar("float32")
                            K.ptx["fma.rn.f32"](o, c_t, ub[j], vals[J + j])
                            ob = K.local_scalar("uint16")
                            K.ptx.cvt.rn.bf16.f32(ob, o)
                            K.ptx.st.global_.b16(output.ptr_to([out_base + col]), ob, pred=K.EQ(ks, 0))
                        return
                    for s in (4, 2, 1):
                        if len(vals) > 1:
                            half = len(vals) // 2
                            b = ((ks // s) & 1) != 0
                            nxt = []
                            for i in range(half):
                                send = K.local_scalar("float32", init=K.if_then_else(b, vals[i], vals[i + half]))
                                keep = K.local_scalar("float32", init=K.if_then_else(b, vals[i + half], vals[i]))
                                r = K.local_scalar("float32")
                                K.ptx["add.f32"](r, keep, _shfl_bfly(send, s))
                                nxt.append(r)
                            vals = nxt
                        else:
                            K.ptx["add.f32"](vals[0], vals[0], _shfl_bfly(vals[0], s))
                    assert len(vals) == R

                    is_d2 = (ks >> 2) != 0
                    jbase = ((ks & 3) * M) // 8
                    tok = n * T + t
                    out_base = (K.cast(tok, "int64") * HV + hv) * D + row0
                    uloc = K.alloc_local((R,), "float32")
                    for i in range(R):
                        other = K.local_scalar("float32", init=_shfl_bfly(vals[i], 4))
                        dd1 = K.if_then_else(is_d2, other, vals[i])
                        dd2 = K.if_then_else(is_d2, vals[i], other)
                        col = warp * rows_per_warp + (jbase + i) * 4 + cw
                        vv = K.local_scalar("uint32")
                        K.ptx.ld.shared.b32(vv, vtab.ptr_to([t * D + row0 + col]))
                        diff = K.local_scalar("float32")
                        K.ptx["sub.f32"](diff, K.reinterpret("float32", vv), dd1)
                        K.ptx["mul.f32"](uloc[i], diff, beta_t)
                        o = K.local_scalar("float32")
                        K.ptx["fma.rn.f32"](o, c_t, uloc[i], dd2)
                        ob = K.local_scalar("uint16")
                        K.ptx.cvt.rn.bf16.f32(ob, o)
                        K.ptx.st.global_.b16(output.ptr_to([out_base + col]), ob, pred=K.LT(ks, 4))
                    for j in range(J):

                        src = (lane & 0x18) | ((j * 8) // M)
                        K.assign(ub[j], _shfl_idx(uloc[j % R], src))

                def update_pass(phase, tile, with_dots):
                    if with_dots:
                        for j in range(J):
                            K.assign(d1[j], zero2)
                            K.assign(d2[j], zero2)
                    u2 = K.alloc_local((J,), "uint64", align=8)
                    for j in range(J):
                        K.assign(u2[j], _f2(ub[j], ub[j]))
                    for c in range(2):
                        kp2 = K.alloc_local((4,), "uint64", align=8)
                        a2 = K.alloc_local((4,), "uint64", align=8)
                        k2 = K.alloc_local((4,), "uint64", align=8)
                        q2 = K.alloc_local((4,), "uint64", align=8)
                        for ii in range(4):
                            i = c * 4 + ii
                            if with_dots:
                                K.ptx.ld.shared.v2.b64(kp2[ii], a2[ii], vec_ptr(phase, i))
                                K.ptx.ld.shared.v2.b64(k2[ii], q2[ii], K.ptx.addr(vec_ptr(phase, i), 16))
                            else:
                                K.ptx.ld.shared.b64(kp2[ii], vec_ptr(phase, i))
                        for j in range(J):
                            packed = K.alloc_local((4,), "uint32", align=16)
                            for ii in range(4):
                                i = c * 4 + ii
                                K.ptx.fma.rn.f32x2(S[j * 8 + i], kp2[ii], u2[j], S[j * 8 + i])
                                K.assign(packed[ii], K.cuda.float22bfloat162_rn_from_float2(S[j * 8 + i]))
                                if with_dots:
                                    K.ptx.mul.rn.f32x2(S[j * 8 + i], a2[ii], S[j * 8 + i])
                                    K.ptx.fma.rn.f32x2(d1[j], k2[ii], S[j * 8 + i], d1[j])
                                    K.ptx.fma.rn.f32x2(d2[j], q2[ii], S[j * 8 + i], d2[j])
                            row = warp * rows_per_warp + j * 4 + cw
                            cs = K.bitwise_xor(K.int32(c), ks >> 2)
                            K.ptx.st.shared.v4.b32(
                                tiles.ptr_to([tile * tile_elems + row * D + ks * 16 + cs * 8]),
                                packed[0], packed[1], packed[2], packed[3],
                            )
                    K.ptx.fence.proxy.async_.shared__cta()

                def store_barrier():
                    """All warps' staging writes complete -> warp 0 may issue the TMA store.

                    Warps 1..3 only arrive (named barrier 1, 128 threads) and continue; warp 0 waits.
                    """
                    if bar_arrive:
                        with K.If(warp == 0):
                            with K.Then():
                                K.ptx.bar.sync(K.uint32(1), K.uint32(NT))
                            with K.Else():
                                K.ptx.bar.arrive(K.uint32(1), K.uint32(NT))
                    else:
                        K.cuda.cta_sync()

                def store_checkpoint(tile, dst_slot, hv, row0):
                    dst0 = (K.cast(dst_slot, "int64") * HV + hv) * (D * D) + row0 * D
                    for pc in range(n_pieces):
                        K.ptx[_BULK_S2G](
                            state_out.ptr_to([dst0 + pc * (load_piece // 2)]),
                            tiles.ptr_to([tile * tile_elems + pc * (load_piece // 2)]),
                            K.uint32(load_piece),
                        )
                    K.ptx.cp.async_.bulk.commit_group()

                def store_warp_rows(tile, dst_slot, hv, row0):
                    """Lane 0 of the calling warp: bulk store of this warp's rows_per_warp rows of `tile`."""
                    wrow = row0 + warp * rows_per_warp
                    dst0 = (K.cast(dst_slot, "int64") * HV + hv) * (D * D) + wrow * D
                    K.ptx[_BULK_S2G](
                        state_out.ptr_to([dst0]),
                        tiles.ptr_to([tile * tile_elems + warp * rows_per_warp * D]),
                        K.uint32(rows_per_warp * D * 2),
                    )
                    K.ptx.cp.async_.bulk.commit_group()

                u = K.local_scalar("int32", init=cta)
                with K.While((u >= 0) & (u < num_units)):
                    head, part, hv, n, h = unit_coords(u)
                    row0 = part * cpt
                    cur_slot = K.local_scalar("int32")
                    nxt_slot = K.local_scalar("int32")


                    K.cuda.iket.mark("unit-start")
                    tk = _rng("wait-vec")
                    vec_stage = K.local_scalar("int32", init=vcount & 1)
                    bar_vec.wait(vec_stage, (vcount >> 1) & 1)
                    K.assign(vcount, vcount + K.int32(1))
                    _rng_end(tk)
                    tk = _rng("preprocess")
                    with K.If(lane == 0 if warp_auto else tid == 0), K.Then():
                        if spec:
                            K.ptx.ld.global_.s32(nxt_slot, ssm_idx.ptr_to([n * T]))
                        else:
                            K.assign(nxt_slot, n)
                    with K.If(tid == 0), K.Then():
                        if spec:
                            with K.If(has_next), K.Then():
                                K.assign(slot0_next, start_slot(u + num_main, acc_next))
                            if l2pf:
                                with K.If(u + 2 * num_main < num_units), K.Then():
                                    K.assign(slot0_next2, start_slot(u + 2 * num_main, acc_next2))
                                    prefetch_l2(u + 2 * num_main, slot0_next2)
                    for rnd in range((T + nw - 1) // nw):
                        t = rnd * nw + warp
                        if rnd * nw + nw <= T:
                            preprocess(t, hv, h, vec_stage)
                        else:
                            with K.If(t < T), K.Then():
                                preprocess(t, hv, h, vec_stage)

                    K.ptx.fence.proxy.async_.shared__cta()
                    K.cuda.cta_sync()
                    _rng_end(tk)


                    tk = _rng("wait-state")
                    state_stage = K.local_scalar("int32", init=scount % 3)
                    bar_state.wait(state_stage, (scount // 3) & 1)
                    K.assign(scount, scount + K.int32(1))
                    _rng_end(tk)
                    next_state_stage = K.local_scalar("int32", init=scount % 3)
                    nxt_u = K.local_scalar("int32", init=K.int32(-1))
                    with K.If(tid == 0), K.Then():
                        bar_clc.wait(0, clc_phase)
                        K.assign(clc_phase, clc_phase ^ K.int32(1))
                        cancelled = K.local_scalar("uint32")
                        K.query_cancel_first_ctaid_x(cancelled, K.address_of(clc_handle[0]))
                        K.assign(nxt_u, K.reinterpret("int32", cancelled))
                        K.ptx.st.shared.b32(clc_mailbox.ptr_to([0]), cancelled)
                        with K.If(cancelled != K.uint32(CLC_SENTINEL)), K.Then():
                            issue_vec(K.reinterpret("int32", cancelled), vcount & 1)
                            K.ptx.cp.async_.bulk.wait_group.read(0)
                            issue_state(
                                K.reinterpret("int32", cancelled),
                                unit_coords(K.reinterpret("int32", cancelled))[3],
                                next_state_stage,
                            )
                            K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_clc.ptr_to([0]), K.uint32(16))
                            K.ptx[_TRY_CANCEL](K.address_of(clc_handle[0]), bar_clc.ptr_to([0]))
                    tk = _rng("ld-state")
                    for j in range(J):
                        row = warp * rows_per_warp + j * 4 + cw
                        for c in range(2):
                            wds = K.alloc_local((4,), "uint32", align=16)
                            cs = K.bitwise_xor(K.int32(c), ks >> 2)
                            K.ptx.ld.shared.v4.b32(wds[0], wds[1], wds[2], wds[3], tiles.ptr_to([state_stage * tile_elems + row * D + ks * 16 + cs * 8]))
                            for i in range(4):
                                K.assign(
                                    S[j * 8 + c * 4 + i],
                                    _f2(
                                        K.cuda.uint_as_float(K.shift_left(wds[i], K.uint32(16))),
                                        K.cuda.uint_as_float(K.bitwise_and(wds[i], K.uint32(0xFFFF0000))),
                                    ),
                                )

                    _rng_end(tk)
                    tk = _rng("phase0")
                    for j in range(J):
                        K.assign(d1[j], zero2)
                        K.assign(d2[j], zero2)
                    for i in range(8):
                        a2 = K.local_scalar("uint64")
                        k2 = K.local_scalar("uint64")
                        q2 = K.local_scalar("uint64")
                        K.ptx.ld.shared.b64(a2, K.ptx.addr(vec_ptr(0, i), 8))
                        K.ptx.ld.shared.v2.b64(k2, q2, K.ptx.addr(vec_ptr(0, i), 16))
                        for j in range(J):
                            K.ptx.mul.rn.f32x2(S[j * 8 + i], a2, S[j * 8 + i])
                            K.ptx.fma.rn.f32x2(d1[j], k2, S[j * 8 + i], d1[j])
                            K.ptx.fma.rn.f32x2(d2[j], q2, S[j * 8 + i], d2[j])
                    reduce_and_finish(0, hv, n, row0)
                    _rng_end(tk)


                    issuer = lane == 0 if warp_auto else tid == 0

                    def tile_free_wait():
                        with K.If(issuer), K.Then():
                            K.ptx.cp.async_.bulk.wait_group.read(1)
                        if warp_auto:
                            K.cuda.warp_sync()
                        else:
                            K.cuda.cta_sync()

                    def store_done_sync():
                        if warp_auto:
                            K.cuda.warp_sync()
                        else:
                            K.cuda.cta_sync()

                    def issue_store(tile, slot):
                        if warp_auto:
                            store_warp_rows(tile, slot, hv, row0)
                        else:
                            store_checkpoint(tile, slot, hv, row0)

                    if T > 1:
                        with K.serial(1, T, unroll=False) as ph:
                            tile = 1 + (ck & 1)
                            tk = _rng("wait-tile")
                            tile_free_wait()
                            _rng_end(tk)
                            tk = _rng("update")
                            update_pass(ph, tile, True)
                            _rng_end(tk)
                            tk = _rng("sync-store")
                            store_done_sync()
                            with K.If(issuer), K.Then():
                                K.assign(cur_slot, nxt_slot)
                                if spec:
                                    K.ptx.ld.global_.s32(nxt_slot, ssm_idx.ptr_to([n * T + ph]))
                                issue_store(tile, cur_slot)
                            K.assign(ck, ck + K.int32(1))
                            _rng_end(tk)
                            tk = _rng("reduce")
                            reduce_and_finish(ph, hv, n, row0)
                            _rng_end(tk)

                    tile = state_stage
                    tk = _rng("update")
                    update_pass(T, tile, False)
                    _rng_end(tk)
                    tk = _rng("sync-store")
                    store_done_sync()
                    with K.If(issuer), K.Then():
                        issue_store(tile, nxt_slot)


                    K.cuda.cta_sync()
                    _rng_end(tk)
                    next_bits = K.local_scalar("uint32")
                    K.ptx.ld.shared.b32(next_bits, clc_mailbox.ptr_to([0]))
                    K.assign(nxt_u, K.reinterpret("int32", next_bits))
                    K.assign(u, nxt_u)

                if spec:
                    with K.If(ncopy_me > 0), K.Then():
                        with K.If(lane == 0), K.Then():
                            K.ptx.cp.async_.bulk.wait_group.read(0)
                        K.cuda.cta_sync()
                        nfast = K.min(ncopy_me, K.int32(MAXC))
                        with K.serial(0, (nfast + 2) // 3, unroll=False) as kb:
                            tk = _rng("copy-batch")
                            k0 = kb * 3
                            nb = K.min(K.int32(3), nfast - k0)
                            with K.If(tid == 0), K.Then():
                                pre = (kb == 0) & (copy_armed == 1)
                                with K.If(pre == False), K.Then():
                                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.cast(nb, "uint32") * K.uint32(tile_bytes))
                                for i in range(3):

                                    with K.If((i < nb) & ((i > 0) | (pre == False))), K.Then():
                                        hv_c, part_c = copy_coords(k0 + i)
                                        slot_c = K.local_scalar("int32")
                                        K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([k0 + i]))
                                        tile_load(i, slot_c, hv_c, part_c)
                            bar_state.wait(0, scount & 1)
                            K.assign(scount, scount + K.int32(1))
                            with K.If(tid == 0), K.Then():
                                for i in range(3):
                                    with K.If(i < nb), K.Then():
                                        hv_c, part_c = copy_coords(k0 + i)
                                        slot_c = K.local_scalar("int32")
                                        K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([k0 + i]))
                                        store_checkpoint(i, slot_c, hv_c, part_c * cpt)
                                K.ptx.cp.async_.bulk.wait_group.read(0)

                            K.cuda.cta_sync()
                            _rng_end(tk)

                        with K.If(ncopy_me > MAXC), K.Then():
                            K.cuda.cta_sync()
                            with K.serial(MAXC, ncopy_me, unroll=False) as kc:
                                rank = (kc * hole_cnt + (cta - hole_base)) // DCU
                                cnt3 = K.local_scalar("int32", init=K.int32(0))
                                with K.serial(0, (num_slots + pmax - 1) // pmax, unroll=False) as chunk:
                                    cb = chunk * pmax
                                    build_flags(cb)
                                    with K.If(warp == 0), K.Then():
                                        scan_untouched(cb, cnt3, rank, MAXC + 1)
                                    K.cuda.cta_sync()
                                with K.If(tid == 0), K.Then():
                                    hv_c, part_c = copy_coords(kc)
                                    slot_c = K.local_scalar("int32")
                                    K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([MAXC + 1]))
                                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar_state.ptr_to([0]), K.uint32(tile_bytes))
                                    tile_load(0, slot_c, hv_c, part_c)
                                bar_state.wait(0, scount & 1)
                                K.assign(scount, scount + K.int32(1))
                                with K.If(tid == 0), K.Then():
                                    hv_c, part_c = copy_coords(kc)
                                    slot_c = K.local_scalar("int32")
                                    K.ptx.ld.shared.b32(slot_c, cpslots.ptr_to([MAXC + 1]))
                                    store_checkpoint(0, slot_c, hv_c, part_c * cpt)
                                    K.ptx.cp.async_.bulk.wait_group.read(0)
                                K.cuda.cta_sync()
                tk = _rng("drain")
                with K.If(lane == 0), K.Then():
                    K.ptx.cp.async_.bulk.wait_group.read(0)
                _rng_end(tk)

            main_role()

        return kda_decode_persist


    OVERRIDE = {"vec_split": False}
    _COMPILED = {}


    def _pick_cpt(N, T, HV):


        heads = N * HV
        if heads <= 16:
            return 32
        return 64


    def _ctas_per_sm(cpt, T, gate_lb, nw=4):
        smem = 3 * cpt * D * 2 + T * RAW_TOK_B + (GATE_B if gate_lb else 0) + (T + 1) * FST * 4 + T * D * 4 + 256
        by_smem = (220 * 1024) // smem
        if nw == 4:
            by_regs = {16: 6, 32: 5, 64: 3, 128: 2}[cpt]
            if T == 1:
                by_regs = {16: 8, 32: 8, 64: 4, 128: 2}[cpt]
        else:
            by_regs = {32: 3, 64: 2, 128: 1}[cpt]
        return max(1, min(by_smem, by_regs))


    def _compile(**kw):
        key = tuple(sorted(kw.items()))
        exe = _COMPILED.get(key)
        if exe is None:
            kernel = build_kernel(**kw)
            target = kernel.target()
            with target:
                exe = tvm.compile(kernel.mod, target=target, tir_pipeline="tirx")
            _COMPILED[key] = exe
        return exe


    def setup(data, N, T):
        q, k, v, g, beta = data["q"], data["k"], data["v"], data["g"], data["beta"]
        initial_state, final_state, output = data["initial_state"], data["final_state"], data["output"]
        H, HV = q.shape[-2], v.shape[-2]
        P = initial_state.shape[0]
        device = q.device
        gate_lb = data["A_log"] is not None
        spec = T > 1
        cpt = int(OVERRIDE.get("cpt", _pick_cpt(N, T, HV)))
        split = D // cpt
        num_units = N * HV * split
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        nw = int(OVERRIDE.get("nw", 4))
        per_sm = int(OVERRIDE.get("ctas_per_sm", _ctas_per_sm(cpt, T, gate_lb, nw)))
        copy_est = max(P - N * T, 0) * HV * split if spec else 0
        num_main = min(num_units + copy_est, per_sm * sms)
        per_sm = min(per_sm, max(1, -(-num_main // sms)))
        extra = {k: bool(OVERRIDE[k]) for k in ("allreduce", "bar_arrive", "warp_auto", "vec_split") if k in OVERRIDE}
        if "warp_auto" not in extra:
            extra["warp_auto"] = (num_units + copy_est) <= 6 * num_main
        if "vec_split" not in extra:
            extra["vec_split"] = extra["warp_auto"] or (
                T == 6 and num_units + copy_est <= 10 * num_main
            )
        extra["nw"] = nw
        if "l2pf" in OVERRIDE:
            extra["l2pf"] = bool(OVERRIDE["l2pf"])
        else:


            extra["l2pf"] = T == 2 and num_units >= 8 * num_main
        exe = _compile(T=T, H=H, HV=HV, gate_lb=gate_lb, cpt=cpt, num_units=num_units, num_main=num_main, per_sm=per_sm, copy_est=copy_est, **extra)

        dummy_f32 = torch.zeros(4, dtype=torch.float32, device=device)
        dummy_i32 = torch.zeros(1, dtype=torch.int32, device=device)
        a_log = data["A_log"].contiguous().view(-1).to(torch.float32) if gate_lb else dummy_f32
        dt_bias = data["dt_bias"].contiguous().view(-1).to(torch.float32) if gate_lb else dummy_f32
        ssm = data["ssm_state_indices"].contiguous().view(-1) if spec else dummy_i32
        acc = data["num_accepted_tokens"].contiguous().view(-1) if spec else dummy_i32
        lower_bound = float(data["lower_bound"]) if data["lower_bound"] is not None else 0.0
        scale = float(data["scale"])
        args = (
            q.view(-1), k.view(-1), v.view(-1), g.view(-1), beta.view(-1),
            a_log, dt_bias, initial_state.view(-1), final_state.view(-1), output.view(-1),
            ssm, acc, scale, lower_bound, int(P), int(N),
        )
        keep = (q, k, v, g, beta, a_log, dt_bias, initial_state, final_state, output, ssm, acc)

        def run():
            exe(*args)

        run._keep = keep
        run()
        torch.cuda.synchronize(device)
        return run
    return setup



_CLC_SETUP = _make_clc()
del _make_clc


def _candidate_setup(data, N, T):
    """The candidate's own dispatch entry (its evolution-harness ``setup``)."""
    if T == 1 and N in (8, 32, 64, 128):
        return _CLC_SETUP(data, N, T)
    return _frontier_setup(data, N, T)


# ---------------------------------------------------------------------------
# Registry interface.
# ---------------------------------------------------------------------------

KERNEL_META = {
    "name": "agent_evolved_kda_decode_multishape",
    "category": "agent_evolved",
    "runtime_cuda_archs": ["sm_100a"],
    "reference_requirements": (
        {"package": "flashinfer-python", "specifier": ">=0.6.18", "import": "flashinfer"},
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.7.0", "import": "cutlass"},
    ),
    "provenance": {
        "generator": "hmz",
        "run": "kda-decode-relay",
        "selected_version": "frontier/colreg-clc-ring",
    },
}

HEAD_DIM = 128
NUM_QK_HEADS = 16
STATE_STD = 0.01
LOWER_BOUND = -5.0
RANDOM_SEED = 42


def _cfg(label, num_tokens, num_seqs, num_v_heads, lower_bound_gate):
    return {
        "label": label,
        "num_tokens": num_tokens,
        "num_seqs": num_seqs,
        "num_qk_heads": NUM_QK_HEADS,
        "num_v_heads": num_v_heads,
        "lower_bound_gate": lower_bound_gate,
        "seed": RANDOM_SEED,
    }


# The thirty packaged official rows: T=1 standard decode, T=2/4/5/6 speculative
# decode with precomputed gates, and the T=3 lower-bound-gate rows at HV=16.
CONFIGS = (
    [_cfg(f"t1_b{n}_hv32_standard", 1, n, 32, False) for n in (8, 16, 32, 64, 128)]
    + [_cfg(f"t2_b{n}_hv32_spec", 2, n, 32, False) for n in (8, 16, 32, 64, 128)]
    + [_cfg(f"t3_b{n}_hv16_lower_bound", 3, n, 16, True) for n in (1, 2, 4, 8, 16)]
    + [_cfg(f"t4_b{n}_hv32_spec", 4, n, 32, False) for n in (8, 16, 32, 64, 128)]
    + [_cfg(f"t5_b{n}_hv32_spec", 5, n, 32, False) for n in (8, 16, 32, 64, 128)]
    + [_cfg(f"t6_b{n}_hv32_spec", 6, n, 32, False) for n in (8, 16, 32, 64, 128)]
)

_CONFIG_KEYS = set(CONFIGS[0]) - {"label"}
_BY_LABEL = {config["label"]: config for config in CONFIGS}


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
    if int(resolved["num_qk_heads"]) != NUM_QK_HEADS:
        raise ValueError(f"num_qk_heads must be {NUM_QK_HEADS}")
    if int(resolved["num_v_heads"]) % int(resolved["num_qk_heads"]):
        raise ValueError("num_v_heads must be a multiple of num_qk_heads")
    return resolved


def _assert_supported_arch() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved KDA decode")
    capability = torch.cuda.get_device_capability()
    runtime_arch = f"sm_{capability[0]}{capability[1]}a"
    if runtime_arch not in KERNEL_META["runtime_cuda_archs"]:
        raise SkipTest(
            "agent-evolved KDA decode requires one of "
            f"{KERNEL_META['runtime_cuda_archs']}, got {runtime_arch}"
        )


def get_kernel(**config: Any):
    """Return the traced Kern PrimFunc this config dispatches to.

    The runtime path builds and compiles through the candidate's own ``setup``,
    which selects among the four routes from the row's token and sequence counts;
    this entry point exists for registry discovery and IR inspection.
    """
    resolved = _config(**config)
    raise SkipTest(
        "agent-evolved KDA decode builds its device programs inside the shape "
        f"dispatch; inspect them through prepare_bench for {resolved['num_tokens']} "
        f"token(s) x {resolved['num_seqs']} sequence(s)"
    )


# ---------------------------------------------------------------------------
# Inputs.
#
# The draw order reproduces the packaged KDA-decode benchmark rows, which follow
# flashinfer PR #4279's benchmark: uniform q/k/v, a sigmoid'd normal beta, a
# log-sigmoid precomputed gate (or a raw normal gate plus fp32 A_log/dt_bias on
# the lower-bound rows), and a normal state pool scaled by 0.01. Speculative rows
# carry cu_seqlens, per-token checkpoint indices and an all-ones acceptance
# vector, with a state pool of total_tokens + 6 slots.
# ---------------------------------------------------------------------------


def prepare_data(**config: Any) -> dict[str, Any]:
    """Build one row's contract inputs plus the preallocated outputs."""
    import torch.nn.functional as F

    resolved = _config(**config)
    device = torch.device("cuda")
    num_tokens = int(resolved["num_tokens"])
    num_seqs = int(resolved["num_seqs"])
    num_qk_heads = int(resolved["num_qk_heads"])
    num_v_heads = int(resolved["num_v_heads"])
    lower_bound_gate = bool(resolved["lower_bound_gate"])
    head_dim = HEAD_DIM
    standard_decode = num_tokens == 1
    total_tokens = num_seqs * num_tokens
    token_shape = (num_seqs, 1) if standard_decode else (1, total_tokens)
    generator = torch.Generator(device=device).manual_seed(int(resolved["seed"]))

    def rand(shape, dtype=torch.bfloat16):
        return torch.rand(shape, dtype=dtype, device=device, generator=generator)

    def randn(shape, dtype=torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)

    q = rand((*token_shape, num_qk_heads, head_dim))
    k = rand((*token_shape, num_qk_heads, head_dim))
    v = rand((*token_shape, num_v_heads, head_dim))
    beta = torch.sigmoid(randn((*token_shape, num_v_heads)))
    if lower_bound_gate:
        g = randn((*token_shape, num_v_heads, head_dim))
        A_log = torch.log(rand((num_qk_heads,), torch.float32) + 1.0)
        dt_bias = randn((num_qk_heads * head_dim,), torch.float32)
        lower_bound = LOWER_BOUND
    else:
        g = F.logsigmoid(randn((*token_shape, num_v_heads, head_dim), torch.float32))
        g = g.to(torch.bfloat16)
        A_log = dt_bias = lower_bound = None
    if standard_decode:
        cu_seqlens = ssm_state_indices = num_accepted_tokens = num_spec_tokens = None
        num_state_slots = num_seqs
    else:
        cu_seqlens = torch.arange(
            0, total_tokens + 1, num_tokens, dtype=torch.int32, device=device
        )
        ssm_state_indices = torch.arange(
            1, total_tokens + 1, dtype=torch.int32, device=device
        ).reshape(num_seqs, num_tokens)
        num_accepted_tokens = torch.ones(num_seqs, dtype=torch.int32, device=device)
        num_spec_tokens = num_tokens - 1
        num_state_slots = total_tokens + 6
    initial_state = (
        randn((num_state_slots, num_v_heads, head_dim, head_dim), torch.float32) * STATE_STD
    ).to(torch.bfloat16)
    return {
        "config": resolved,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": float(head_dim**-0.5),
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "num_spec_tokens": num_spec_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "lower_bound": lower_bound,
        "output": torch.empty_like(v),
        # Only the checkpoint slots are written; the rest must carry over.
        "final_state": initial_state.clone(),
    }


def _launch_state(case: dict[str, Any]):
    """Bind the launch through the candidate's own shape dispatch."""
    q, cu_seqlens = case["q"], case["cu_seqlens"]
    num_seqs = q.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
    tokens = q.shape[0] * q.shape[1] // num_seqs
    return _candidate_setup(case, int(num_seqs), int(tokens))


# ---------------------------------------------------------------------------
# Independent oracle and correctness.
#
# The packaged task's reference: the fp32 recurrence, one sequence at a time,
# from a k-first working state. It shares no code with the kernel and checks the
# complete final-state pool as well as the output.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _reference_output(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v, g, beta = case["q"], case["k"], case["v"], case["g"], case["beta"]
    A_log, dt_bias = case["A_log"], case["dt_bias"]
    scale = float(case["scale"])
    initial_state = case["initial_state"]
    cu_seqlens = case["cu_seqlens"]
    ssm_state_indices = case["ssm_state_indices"]
    num_accepted_tokens = case["num_accepted_tokens"]
    lower_bound = case["lower_bound"]

    num_qk_heads, head_dim = q.shape[-2], q.shape[-1]
    num_v_heads = v.shape[-2]
    group = num_v_heads // num_qk_heads
    standard = cu_seqlens is None
    if standard:
        num_seqs, num_tokens = q.shape[0], 1
    else:
        num_seqs = int(cu_seqlens.numel()) - 1
        num_tokens = int(case["num_spec_tokens"]) + 1

    def normalize(x):
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)

    qf = normalize(q.float()).reshape(num_seqs, num_tokens, num_qk_heads, head_dim) * scale
    kf = normalize(k.float()).reshape(num_seqs, num_tokens, num_qk_heads, head_dim)
    vf = v.float().reshape(num_seqs, num_tokens, num_v_heads, head_dim)
    bf = beta.float().reshape(num_seqs, num_tokens, num_v_heads)
    gf = g.float().reshape(num_seqs, num_tokens, num_v_heads, head_dim)
    if A_log is not None:
        head_of_v = torch.arange(num_v_heads, device=q.device) // group
        a = A_log.float().exp()[head_of_v]
        bias = dt_bias.float().view(num_qk_heads, head_dim)[head_of_v]
        gf = float(lower_bound) * torch.sigmoid(a[:, None] * (gf + bias))
    decay = gf.exp()

    output = torch.empty_like(v, dtype=torch.float32)
    output_view = output.reshape(num_seqs, num_tokens, num_v_heads, head_dim)
    pool = initial_state.clone()
    if not standard:
        indices = ssm_state_indices.to(torch.int64).cpu()
        accepted = (
            torch.ones(num_seqs, dtype=torch.int64)
            if num_accepted_tokens is None
            else num_accepted_tokens.to(torch.int64).cpu().clamp(1, num_tokens)
        )
    for n in range(num_seqs):
        start_slot = n if standard else int(indices[n, accepted[n] - 1])
        S = pool[start_slot].float().transpose(-1, -2).clone()
        for t in range(num_tokens):
            q_t = qf[n, t].repeat_interleave(group, dim=0)
            k_t = kf[n, t].repeat_interleave(group, dim=0)
            v_t, b_t = vf[n, t], bf[n, t]
            S = S * decay[n, t][:, :, None]
            kS = torch.einsum("hk,hkv->hv", k_t, S)
            S = S + torch.einsum("hk,hv->hkv", b_t[:, None] * k_t, v_t - kS)
            output_view[n, t] = torch.einsum("hk,hkv->hv", q_t, S)
            slot = n if standard else int(indices[n, t])
            pool[slot] = S.transpose(-1, -2).to(pool.dtype)
    return output.to(v.dtype), pool


def check_correctness(outputs: dict[str, Any], **config: Any) -> None:
    """Gate the output and the complete final-state pool against the oracle."""
    case = outputs["case"]
    expected_output, expected_state = _reference_output(case)
    torch.testing.assert_close(
        outputs["output"].float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        outputs["final_state"].float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )


def run_test(**config: Any) -> None:
    """Run one config through the dispatch and gate it against the oracle."""
    _assert_supported_arch()
    case = prepare_data(**config)
    run = _launch_state(case)
    run()
    torch.cuda.synchronize()
    check_correctness(
        {"case": case, "output": case["output"], "final_state": case["final_state"]},
        **config,
    )


# ---------------------------------------------------------------------------
# FlashInfer reference arm.
#
# The packaged baseline is `flashinfer.kda_decode.recurrent_kda` on its CuTe-DSL
# backend. The kernel updates its state pool in place while the contract returns
# a new pool, so the state copy and the caller-owned output buffer are prepare
# work, as PR #4279's upstream arm does; the timed span is the kernel alone.
# ---------------------------------------------------------------------------


def _recurrent_kda_builder(case: dict[str, Any]):
    from flashinfer.kda_decode import recurrent_kda

    args = (
        case["q"], case["k"], case["v"], case["g"], case["beta"],
        case["A_log"], case["dt_bias"], float(case["scale"]),
        case["initial_state"].clone(), case["cu_seqlens"],
        case["ssm_state_indices"], case["num_spec_tokens"],
        case["num_accepted_tokens"], case["lower_bound"],
        torch.empty_like(case["v"]),
    )

    def launch():
        (q, k, v, g, beta, A_log, dt_bias, scale, state, cu_seqlens,
         ssm_state_indices, num_spec_tokens, num_accepted_tokens,
         lower_bound, output) = args
        out, _ = recurrent_kda(
            q, k, v, g, beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=state,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=A_log is not None,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens=num_accepted_tokens,
            output=output,
            backend="cute-dsl",
        )
        return out, state

    launch()  # JIT / extension load before timing
    return launch


# ---------------------------------------------------------------------------
# Benchmark entry points.
# ---------------------------------------------------------------------------


def prepare_bench(**config: Any):
    """Build the row and bind its launch, so nothing compiles in the GPU stage."""
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
        references={"flashinfer_recurrent_kda": lambda: _recurrent_kda_builder(case)},
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
