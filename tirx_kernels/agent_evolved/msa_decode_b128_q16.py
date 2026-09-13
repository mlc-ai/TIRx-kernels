# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved SM100a MiniMax sparse-attention (MSA) decode.

The supported contract is the flat B=128 MSA multi-token decode row
``mtp_bf16_b128_q16_kv4096_h64``: batch 128, ``seqlen_q`` = 16, ``kv_len`` =
4096 per request, Hq = 64, Hkv = 4 (GQA = 16), D = 128, top-k = 16, bf16
q/k/v and output, ``cu_seqlens_k`` of length 129, and ``page_table`` and
``seqused_k`` both ``None``. Selection is ``q2k_indices`` int32[Hkv, B *
seqlen_q, 16] holding ascending sequence-local KV block ids padded with -1.
Token ``i`` of request ``b`` sits at KV position ``kv_len - seqlen_q + i``,
so the bottom-right causal boundary applies on top of the selected blocks.

The selected kernel is the ``qmajor-union`` frontier member of the
2026-09-12 MSA-decode evolution run. Everything from ``import ctypes`` down
to ``_encode`` is that candidate's source unchanged; this module adds the
registry interface, input generation, the independent oracle, and the
MiniMax reference arm.

It is pure bf16: q/k/v/P are bf16 and both tcgen05 MMAs are ``kind::f16``
with FP32 TMEM accumulation. The only numerical approximation is in the
softmax exponential, where three packed-f32x2 quadratic pairs per unmasked
fragment replace native ``ex2.approx.ftz.f32``; its range reduction clamps
at -127.

Candidate mechanism notes, carried over from the evolution run:

Approach family: **qmajor dense-valid interval** (Q-major, sparse-row-masked).
One CTA task owns one (request b, kv head h) item: the 16 query tokens x 16
GQA heads form two 128-row Q tiles (8 tokens x 16 heads each). For this
pinned 32-block row, the load warp streams the valid block interval through a
5-stage TMA ring while retaining each token's exact 32-bit sparse-selection
mask; both Q tiles share every K/V tile. Per row (token, head) the softmax
applies the token's own selection bit for the current block (unselected ->
P = 0 for that row) plus the bottom-right causal column mask on the diagonal
block(s). Online softmax with TMEM-resident O, 2^-8 rescale skip,
cross-aliased bf16 P in the other tile's S region, exp ping-pong between the
two softmax warpgroups, persistent CTAs with a dynamic (atomic) task
scheduler. Items beyond the last full round of CTAs are split into two
KV-half tasks (tail balancing); the second half to finish merges the
partner's (max, sum, O) partial from global scratch.

Roles (12 warps): 0-7 softmax (two warpgroups, one per Q tile), warp 8
issues MMA, warp 9 performs TMA and selection preparation, and warps 10-11
are idle. The softmax warpgroups receive 232 registers/thread while the
producer group uses 40. O is normalized in registers and stored straight to
global memory (no smem staging). Softmax warps whose two tokens both skip
the current block bypass the exp/sum/P work for that block (their O rows are
protected by the PV disable-output-lane mask).
"""

import ctypes
import os
from typing import Any
from unittest import SkipTest

import torch
import tvm

import tirx_kernels.kern as K

HEAD_DIM = 128
BLK_N = 128
BLK_M = 128
GQA = 16
TOK_PER_TILE = BLK_M // GQA
N_TILES = 2
TOK_PER_CTA = TOK_PER_TILE * N_TILES
KV_DEPTH = 5
N_COLS_TMEM = 512
MMA_N = 128
MMA_K = 16
MAX_BLOCKS = 32
LOG2E = 1.4426950408889634
K_SPLIT = 2 * MMA_K
P_SPLIT_Q = 1
N_SUM_ACC = 8
MAX_CHAINS = 8
EMU_PAIRS = 3
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


                                                                             
           
                                                                             
class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode(tensor, dims, strides, box, l2_promotion=2):
    desc = _AlignedTensorMap()
    rank = len(dims)
    assert len(strides) == rank - 1 and len(box) == rank
    tvm.get_global_func("runtime.cuTensorMapEncodeTiled")(
        desc.ptr,
        "bfloat16",
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



# ---------------------------------------------------------------------------
# Registry interface.
# ---------------------------------------------------------------------------

KERNEL_META = {
    "name": "agent_evolved_msa_decode_b128_q16",
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
    ),
    "provenance": {
        "generator": "hmz",
        "run": "msa-decode-b128",
        "selected_version": "frontier/qmajor-union",
    },
}

CONFIGS = [
    {
        "label": "b128_q16_kv4096_h64",
        "batch_size": 128,
        "seqlen_q": 16,
        "seqlen_kv": 4096,
        "num_qo_heads": 64,
        "num_kv_heads": 4,
        "topk": 16,
        "seed": 50,
    }
]


def _config(**config: Any) -> dict[str, Any]:
    """Validate one config against the contract this kernel implements."""
    values = {key: value for key, value in config.items() if key != "label"}
    unknown = set(values) - {
        "batch_size",
        "seqlen_q",
        "seqlen_kv",
        "num_qo_heads",
        "num_kv_heads",
        "topk",
        "seed",
    }
    if unknown:
        raise ValueError(f"unsupported config keys: {sorted(unknown)}")
    resolved = {**CONFIGS[0], **values}
    resolved.pop("label", None)
    if int(resolved["num_qo_heads"]) != int(resolved["num_kv_heads"]) * GQA:
        raise ValueError(f"GQA must be {GQA}")
    if int(resolved["seqlen_kv"]) > MAX_BLOCKS * BLK_N:
        raise ValueError(f"kv_len must be <= {MAX_BLOCKS * BLK_N}")
    if int(resolved["seqlen_q"]) != TOK_PER_CTA:
        raise ValueError(f"seqlen_q must be {TOK_PER_CTA}")
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
    """Return the traced Kern PrimFunc for one compile key."""
    from tirx_kernels.runner import hardware_num_sms

    resolved = _config(**config)
    os.environ.setdefault("TVM_CUDA_PTXAS_REG_LEVEL", "6")
    kernel = make_kernel(
        int(resolved["batch_size"]),
        int(resolved["seqlen_q"]),
        int(resolved["num_qo_heads"]),
        int(resolved["num_kv_heads"]),
        int(resolved["topk"]),
        hardware_num_sms(),
    )
    return kernel.func


# ---------------------------------------------------------------------------
# Inputs.
#
# The draw order reproduces the packaged MSA-decode benchmark row
# `mtp_bf16_b128_q16_kv4096_h64`, which itself follows flashinfer PR #4355's
# `bench_blackwell_msa_sm100.py`: q, k and v are `randn/3` in one generator
# sequence, then `q2k_indices` is drawn per (query token, kv head) from the
# blocks the token may see under bottom-right causal masking.
# ---------------------------------------------------------------------------


def _make_q2k_indices(batch_size, seqlen_q, seqlen_kv, num_kv_heads, topk, seed, device):
    """The packaged benchmark's random-valid, bottom-right-causal selection."""
    total_q = batch_size * seqlen_q
    out = torch.full((num_kv_heads, total_q, topk), -1, dtype=torch.int32)
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    offset = seqlen_kv - seqlen_q
    for row in range(total_q):
        visible_blocks = (offset + row % seqlen_q + 1 + BLK_N - 1) // BLK_N
        for kv_head in range(num_kv_heads):
            selected = torch.randperm(visible_blocks, generator=generator)
            selected = selected[: min(topk, visible_blocks)].sort().values
            out[kv_head, row, : selected.numel()] = selected.to(torch.int32)
    return out.to(device)


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
    seed = int(resolved["seed"])
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(shape):
        values = torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
        return (values / 3.0).to(torch.bfloat16)

    total_q = batch_size * seqlen_q
    total_k = batch_size * seqlen_kv
    q = randn((total_q, num_qo_heads, HEAD_DIM))
    k = randn((total_k, num_kv_heads, HEAD_DIM))
    v = randn((total_k, num_kv_heads, HEAD_DIM))
    cu_seqlens_k = torch.arange(0, total_k + 1, seqlen_kv, dtype=torch.int32, device=device)
    q2k_indices = _make_q2k_indices(
        batch_size, seqlen_q, seqlen_kv, num_kv_heads, topk, seed, device
    )
    return {
        "config": resolved,
        "q": q,
        "k": k,
        "v": v,
        "q2k_indices": q2k_indices,
        "cu_seqlens_k": cu_seqlens_k,
        "page_table": None,
        "seqused_k": None,
        "seqlen_q": seqlen_q,
        "softmax_scale": float(HEAD_DIM**-0.5),
        "output": torch.empty_like(q),
    }


def _tirx_args(case: dict[str, Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Bind one case to the kernel's argument list (the candidate's ``setup``)."""
    q, k, v = case["q"], case["k"], case["v"]
    q2k = case["q2k_indices"]
    cu_k = case["cu_seqlens_k"]
    out = case["output"]
    batch_size = int(case["config"]["batch_size"])
    seqlen_q = int(case["seqlen_q"])
    total_q, hq, d = q.shape
    total_k, hkv, dk = k.shape
    assert d == HEAD_DIM and dk == HEAD_DIM
    assert seqlen_q == TOK_PER_CTA and total_q == batch_size * seqlen_q
    assert hq == hkv * GQA
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert out.is_contiguous() and q2k.is_contiguous()
    assert cu_k.is_contiguous() and cu_k.dtype == torch.int32

    scale_log2 = float(case["softmax_scale"]) * LOG2E
    qo_dims = (HEAD_DIM // 2, hq, total_q, 2)
    qo_strides = (HEAD_DIM * F16_BYTES, hq * HEAD_DIM * F16_BYTES, (HEAD_DIM // 2) * F16_BYTES)
    qo_box = (HEAD_DIM // 2, GQA, TOK_PER_TILE, 2)
    q_map = _encode(q, qo_dims, qo_strides, qo_box)
    kv_dims = (HEAD_DIM // 2, total_k, hkv * 2)
    kv_strides = (hkv * HEAD_DIM * F16_BYTES, (HEAD_DIM // 2) * F16_BYTES)
    kv_box = (HEAD_DIM // 2, BLK_N, 2)
    k_map = _encode(k, kv_dims, kv_strides, kv_box, l2_promotion=3)
    v_map = _encode(v, kv_dims, kv_strides, kv_box, l2_promotion=3)

    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    sched = torch.zeros(2, dtype=torch.int32, device=q.device)
    num_items = batch_size * hkv
    num_ctas = min(int(num_sms), num_items)
    r_split = num_items - (num_items // num_ctas) * num_ctas
    n_slots = max(r_split, 1) * 2
    mrg_o = torch.zeros(n_slots * BLK_M * HEAD_DIM, dtype=torch.float32, device=q.device)
    mrg_ml = torch.zeros(n_slots * 2 * BLK_M, dtype=torch.float32, device=q.device)
    mrg_ctl = torch.zeros(n_slots * 2, dtype=torch.int32, device=q.device)
    q2k_flat = q2k.view(-1)
    out_flat = out.view(-1)
    args = (
        q_map.ptr, k_map.ptr, v_map.ptr, out_flat, q2k_flat, cu_k,
        sched, mrg_o, mrg_ml, mrg_ctl, scale_log2,
    )
    keep = (
        q, k, v, q2k, q2k_flat, cu_k, out, out_flat,
        sched, mrg_o, mrg_ml, mrg_ctl, q_map, k_map, v_map,
    )
    return args, keep


# ---------------------------------------------------------------------------
# Independent oracle and correctness.
#
# This is the packaged task's reference: fp32 sparse decode attention, one
# request at a time, with the bottom-right causal boundary applied on top of
# the selected blocks. It shares no code with the kernel.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _reference_output(case: dict[str, Any]) -> torch.Tensor:
    q, k, v = case["q"], case["k"], case["v"]
    q2k_indices = case["q2k_indices"]
    cu_seqlens_k = case["cu_seqlens_k"]
    softmax_scale = float(case["softmax_scale"])
    q_len = int(case["seqlen_q"])

    num_qo_heads, head_dim = q.shape[1], q.shape[2]
    num_kv_heads = k.shape[1]
    group = num_qo_heads // num_kv_heads
    device = q.device
    cu_k = cu_seqlens_k.tolist()
    kv_lens = [cu_k[b + 1] - cu_k[b] for b in range(len(cu_k) - 1)]

    out = torch.zeros(q.shape, dtype=torch.float32, device=device)
    for b, kv_len in enumerate(kv_lens):
        q_start, q_end = b * q_len, (b + 1) * q_len
        kb = k[cu_k[b] : cu_k[b + 1]].float()
        vb = v[cu_k[b] : cu_k[b + 1]].float()
        qb = q[q_start:q_end].float().view(q_len, num_kv_heads, group, head_dim)
        selected = q2k_indices[:, q_start:q_end]
        positions = torch.arange(kv_len, device=device)
        allowed = (
            (positions // BLK_N).view(1, 1, -1, 1) == selected.unsqueeze(2)
        ).any(-1)
        q_pos = kv_len - q_len + torch.arange(q_len, device=device)
        allowed &= positions.view(1, 1, -1) <= q_pos.view(1, -1, 1)
        allowed = allowed.unsqueeze(2)
        logits = torch.einsum("qhgd,khd->hqgk", qb, kb) * softmax_scale
        probs = torch.softmax(logits.masked_fill(~allowed, NEG_INF), dim=-1)
        probs = torch.where(allowed.any(-1, keepdim=True), probs, 0.0)
        ob = torch.einsum("hqgk,khd->qhgd", probs, vb)
        out[q_start:q_end] = ob.reshape(q_len, num_qo_heads, head_dim)
    return out.to(q.dtype)


def check_correctness(outputs: dict[str, Any], **config: Any) -> None:
    """Gate the kernel output against the oracle in the native bf16 dtype."""
    case = outputs["case"]
    actual = outputs["output"]
    expected = _reference_output(case)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def run_test(**config: Any) -> None:
    """Compile, run once and gate against the oracle."""
    _assert_supported_arch()
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**config)
    executable = compile_kernel(get_kernel(**config))
    args, keep = _tirx_args(case)
    executable(*args)
    torch.cuda.synchronize()
    check_correctness({"case": case, "output": case["output"]}, **config)
    del keep


# ---------------------------------------------------------------------------
# MiniMax reference arm.
#
# The packaged baseline for this row is MiniMax's public sparse attention
# (MiniMax-AI/MSA at 80434d7f, `benchmarks/bench_blackwell_msa_sm100.py`,
# `baseline_mode="minimax_public"`). `build_k2q_csr` turns the contract's
# `q2k_indices` into the kernel's CSR reverse index and forward schedule; that
# build is prepare work and the timed span is `sparse_atten_func` alone.
# ---------------------------------------------------------------------------


def _minimax_args(case: dict[str, Any]):
    import fmha_sm100

    q, k, v = case["q"], case["k"], case["v"]
    q2k_indices = case["q2k_indices"]
    cu_seqlens_k = case["cu_seqlens_k"]
    seqlen_q = int(case["seqlen_q"])
    cu_seqlens_q = torch.arange(
        0, q.shape[0] + 1, seqlen_q, dtype=torch.int32, device=q.device
    )
    kv_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    max_seqlen_k = int(kv_lens.max())
    total_rows = int(((kv_lens + BLK_N - 1) // BLK_N).sum())
    k2q_row_ptr, k2q_q_indices, schedule = fmha_sm100.build_k2q_csr(
        q2k_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        BLK_N,
        total_k=int(cu_seqlens_k[-1]),
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=seqlen_q,
        total_rows=total_rows,
        qhead_per_kv=q.shape[1] // k.shape[1],
        return_schedule=True,
    )
    return (
        q, k, v, k2q_row_ptr, k2q_q_indices, int(q2k_indices.shape[-1]),
        cu_seqlens_q, cu_seqlens_k, seqlen_q, max_seqlen_k, schedule,
        float(case["softmax_scale"]),
    )


def _minimax_launch(
    q, k, v, k2q_row_ptr, k2q_q_indices, topk, cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k, schedule, softmax_scale,
):
    import fmha_sm100

    return fmha_sm100.sparse_atten_func(
        q,
        k,
        v,
        k2q_row_ptr,
        k2q_q_indices,
        topk,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        blk_kv=BLK_N,
        causal=True,
        softmax_scale=softmax_scale,
        return_softmax_lse=False,
        page_table=None,
        seqused_k=None,
        schedule=schedule,
    )


def _minimax_reference(case: dict[str, Any]):
    return _minimax_launch(*_minimax_args(case))


# ---------------------------------------------------------------------------
# Benchmark entry points.
# ---------------------------------------------------------------------------


def prepare_bench(**config: Any):
    """Trace and compile before bench-suite assigns a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(config), "executable": compile_kernel(get_kernel(**config))}
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
    from tirx_kernels.runner import bench

    config = dict(prepared["config"])
    config.update(kwargs)
    rounds = config.pop("rounds", 5)
    cooldown_s = config.pop("cooldown_s", 1.0)
    case = prepare_data(**config)
    executable = prepared["executable"]
    args, keep = _tirx_args(case)
    executable(*args)
    torch.cuda.synchronize()

    def _minimax_builder():
        # The CSR reverse index and forward schedule are prepare work, as the
        # packaged benchmark builds them outside timing.
        minimax_args = _minimax_args(case)
        return lambda: _minimax_launch(*minimax_args)

    results = bench(
        {"tirx": lambda: executable(*args)},
        references={"minimax_msa": _minimax_builder},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )
    del keep
    return results


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
