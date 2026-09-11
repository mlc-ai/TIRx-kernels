# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved SM100a MiniMax sparse-attention (MSA) prefill.

The supported contract is the flat B=1 MSA prefill row: total_q = total_k =
4096, Hq = 64, Hkv = 4 (GQA = 16), D = 128, top-k = 16, bf16 q/k/v and output,
``cu_seqlens_q``/``cu_seqlens_k`` of length two, and ``page_table`` and
``seqused_k`` both ``None``. Selection is ``q2k_indices`` int32[Hkv, total_q,
16] holding ascending sequence-local KV block ids padded with -1, under
bottom-right causal masking.

The selected kernel is the ``qmajor-persistent`` frontier member of the
2026-09-10 MSA-prefill evolution run. Everything from the module docstring's
mechanism notes down to ``_compiled`` is that candidate's source; this module
adds the registry interface, input generation, the independent oracle, and the
MiniMax reference arm.

It is pure bf16: q/k/v/P are bf16 and both tcgen05 MMAs are ``kind::f16`` with
FP32 TMEM accumulation. The only numerical approximation is in the softmax
exponential, where 48 of the 128 exponentials in each unmasked non-final
fragment pattern use a packed-f32x2 quadratic approximation and the rest use
native ``ex2.approx.ftz.f32``.

Candidate mechanism notes, carried over from the evolution run:

The selection/union warp transposes the sixteen per-token block masks into
four ready-to-issue disable-output-lane words for each query tile and union
entry.  QK always suppresses unselected 16-row token groups; accumulated PV
does too.  The initial overwriting PV remains unmasked so every O row is
defined.  This keeps mask construction off the saturated MMA issuer.

Approach family: **qmajor-union**.  One CTA owns (kv head, 16 consecutive query
tokens).  GQA folds the 16 query heads of a kv head into the MMA M dimension, so
the 16 tokens form two 128-row Q tiles (8 tokens x 16 heads each).  The load
warp builds the *union* of the 16 tokens' selected KV blocks (a 32-bit mask,
kv_len <= 4096), sorted high-block-first, and streams K/V of those blocks
through a 4-stage 32 KB SMEM ring with TMA.  Both Q tiles share every K/V tile.
Per row (token, head) the softmax applies the token's own selection bit for the
current block (unselected -> the whole row of that block is -inf, i.e. P = 0)
plus the bottom-right causal column mask on the diagonal block(s).  The online
softmax, TMEM-resident O accumulator, 2^-8 rescale-skip, correction warpgroup
and MMA/softmax handshakes follow the canonical FlashAttention-4 port.

Roles (16 warps): 0-7 softmax (two warpgroups, one per Q tile), 8-11 idle,
12 MMA issue, 13 TMA load + union build, 14-15 idle.

v8 (fused softmax): the scale FMA, exp2 (MUFU plus polynomial emulation),
row-sum accumulation and bf16 conversion run in one interleaved loop; the
first half of P is published mid-loop so PV part 1 overlaps the remaining
exponentials and the P -> PV -> QK(n+1) -> S(n+1) round trip shortens; the
row-max tree uses 8 chains.  No correction warpgroup: a softmax warp rescales
its own O rows on the rare path (PV(n-1) completion is implied by s_ready(n),
whose QK(n) was issued after PV(n-1)); rows whose running max is still -inf
never rescale.

v9 (persistent): grid = number of SMs; each CTA walks a heavy-first snake
task list.  The load warp runs into the next task (double-buffered union
list, Q released right after the task's last QK), the MMA warp starts the next
task's QK behind the tail PVs (O reuse gated by o_free), and the epilogue
stages the normalized bf16 O tile in shared memory for a coalesced TMA store
issued by a store warp.  K/V ring: 3 stages; SMEM = Q 64 KB + KV 96 KB +
O staging 64 KB.

v10: dynamic task scheduler (one global atomic counter, self-reset by the last
CTA so every launch starts at task 0; the load warp grabs tasks and publishes
the task id with the union list), union build before the Q wait so it overlaps
the previous task's tail, and an optional split of QK into two N=64 halves so
the first half is issued before PV part 2 (MSA_QK_SPLIT=1).

v11 (cross-aliased P): tile i's bf16 P lives in the upper half of the OTHER
tile's S region.  QK_i(n+1) then only has to follow PV_{1-i}(n-1) (ISA
pipelined-pair rule) and WG_i's read of S_i(n) (s_consumed barrier), so it is
issued early in the step and the P -> PV -> QK -> S round trip is hidden
behind the other tile's softmax.  A pv_done commit per PV serves the rare O
rescale path; K/V are loaded K-ahead (K(n+1) before V(n)).

v13 (exp ping-pong): the two softmax warpgroups alternate their MUFU-heavy
exponential phases through a pair of turn barriers (WG0 exp, WG1 exp, WG0
exp, ...), so they never contend for the XU pipe and each warpgroup's row-max,
P store and bookkeeping run under the other's exponentials.

v15: the exponential pass (scale FMA + MUFU/emulated exp2 for all 128 values)
is separated from the sum/convert pass, and the XU turn is handed over between
them, so the other warpgroup's MUFU block runs while this one does its
adds/conversions/P stores.

v21: the load warp publishes each token's already-computed 32-bit selection
mask with the task metadata.  Each 16-head subgroup loads it once and broadcasts
it with a warp shuffle, eliminating the softmax threads' redundant q2k scans.

v22: the union warp reads the fixed 16-entry q2k row as two aligned 128-bit
halves, one half per lane pair.  A shuffle joins each pair and ``redux.sync.or``
forms the task union.  This turns sixteen sparse scalar warp loads into two
vector warp loads while retaining one independently published mask per token.

v24: all active lanes extract the descending union list in parallel with the
PTX ``fns`` instruction.  This replaces the lane-zero serial CLZ/clear loop
without changing the union order consumed by the K/V pipeline.

Frontier revision (2026-09-11): the MMA warp derives every tcgen05 tmem
address from the runtime TMEM allocation base broadcast once per role
(K.uniform), which keeps those operands in uniform registers instead of a
vector-register constant plus R2UR move inside each elected block (78 -> 15
R2UR in SASS; paired official runs 0.1842/0.1829 ms vs 0.1849/0.1894 ms for the
previous revision).  Exponentials: 48 of the 128 exponentials in each unmasked
non-final-fragment pattern use a packed-f32x2 quadratic approximation; the
remaining values use native ``ex2.approx.ftz.f32``.  Its range reduction
clamps at -127.  For an unselected row, the reconstructed value from -inf has
f32 bits 0x00003884: the ftz row-sum add consumes it as zero and the subsequent
bf16 conversion rounds it to exact zero, so an unselected block does not enter
either the normalization sum or PV.
"""

import ctypes
import math
import os
from functools import lru_cache
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
KV_DEPTH = 3                                
N_COLS_TMEM = 512
MMA_N = 128
MMA_K = 16
MAX_BLOCKS = 32                                          
LOG2E = 1.4426950408889634
K_SPLIT = 4 * MMA_K                                        
P_SPLIT_Q = 2                                                      
N_SUM_ACC = 8                                                       
QK_SPLIT = 0
ID_QK64 = 0x08100490                                            
MAX_CHAINS = 8                                               
EMU_PAIRS = 5
EMU_START = 0
SUM_EARLY = 1
RESCALE_THRESHOLD = 8.0
NEG_INF = float("-inf")
F16_BYTES = 2

TMA_G2S_3D = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
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
                                                                          
ID_QK = 0x08200490                           
ID_PV = 0x08210490                  


def ceildiv(a, b):
    return (a + b - 1) // b


def make_kernel(TOTAL_Q, HQ, HKV, TOPK, NUM_CTAS):
    assert HQ == HKV * GQA
    assert TOPK == 16
    assert TOTAL_Q % TOK_PER_CTA == 0
    NUM_GROUPS = TOTAL_Q // TOK_PER_CTA
    NUM_TASKS = NUM_GROUPS * HKV
    NUM_CTAS = min(NUM_CTAS, NUM_TASKS)
    Q_TILE_BYTES = BLK_M * HEAD_DIM * F16_BYTES
    KV_TILE_BYTES = BLK_N * HEAD_DIM * F16_BYTES

    @K.kernel(warps=16, arch="sm_100a", min_blocks_per_sm=1, grid=NUM_CTAS)
    def msa_prefill_qmajor_union(
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        o_map: K.TensorMap,
        q2k: K.gptr[K.i32],
        cu_q: K.gptr[K.i32],
        cu_k: K.gptr[K.i32],
        sched: K.gptr[K.i32],
        scale_log2: K.f32,
    ):
        cta = K.cta_id()
        warp_cta = K.warp_id()
        wg_id = warp_cta >> 2
        warp_id = warp_cta & 3
        tid_in_wg = K.thread_id() & 127
        lane = K.lane_id()

                                                                                 
        def task_coords(task):
            grp = (NUM_GROUPS - 1) - task // HKV
            return grp * TOK_PER_CTA, task % HKV

                                                                               
        cu_q0 = K.local_scalar("int32")
        cu_q1 = K.local_scalar("int32")
        cu_k0 = K.local_scalar("int32")
        cu_k1 = K.local_scalar("int32")
        K.ptx.ld.global_.nc.b32(cu_q0, cu_q.ptr_to([0]))
        K.ptx.ld.global_.nc.b32(cu_q1, cu_q.ptr_to([1]))
        K.ptx.ld.global_.nc.b32(cu_k0, cu_k.ptr_to([0]))
        K.ptx.ld.global_.nc.b32(cu_k1, cu_k.ptr_to([1]))
        q_len = cu_q1 - cu_q0
        kv_len = cu_k1 - cu_k0
        causal_off = kv_len - q_len                                      

                                                                              
        smem = K.smem_pool()
        q_smem = smem.alloc((N_TILES, BLK_M, HEAD_DIM), K.bf16, swizzle=K.SW128B)
        kv_base = N_TILES * Q_TILE_BYTES
        k_smem = smem.alloc((KV_DEPTH, BLK_N, HEAD_DIM), K.bf16, swizzle=K.SW128B)
        smem.pool.move_base_to(kv_base)
        v_smem = smem.alloc((KV_DEPTH, BLK_N, HEAD_DIM), K.bf16, swizzle=K.SW128B)
        smem.pool.move_base_to(kv_base + KV_DEPTH * KV_TILE_BYTES)
        o_smem = smem.alloc((N_TILES, BLK_M, HEAD_DIM), K.bf16, swizzle=K.SW128B)

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
        union_list = smem.alloc((2 * MAX_BLOCKS,), K.i32)                  
        union_meta = smem.alloc((8,), K.i32)                                                  
        token_masks = smem.alloc((2 * TOK_PER_CTA,), K.u32)                                          
        output_lane_masks = smem.alloc((2, MAX_BLOCKS, N_TILES, 4), K.u32)

                                                                              
        kv_pipe = K.PipelineState(KV_DEPTH, phase=0)
        score_epoch = K.PipelineState(1, phase=0)
        tmem_epoch = K.PipelineState(1, phase=0)
        q_epoch = K.PipelineState(1, phase=0)
        o_epi_epoch = K.PipelineState(1, phase=0)

                                                                              
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
        o_staged = K.MBarrier(smem, 2)                                                
        o_staged.init(128)
        o_smem_free = K.MBarrier(smem, 2)                                 
        o_smem_free.init(1)
        union_ready = K.MBarrier(smem, 2)                 
        union_ready.init(32)
        union_free = K.MBarrier(smem, 2)                                                          
        union_free.init(256 + 32)

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

        def fma_f32x2(values, idx, multiplier, addend_value):
            packed = K.local_scalar("uint64")
            rhs = K.local_scalar("uint64")
            addend = K.local_scalar("uint64")
            K.ptx.mov.b64(packed, values[idx], values[idx + 1])
            K.ptx.mov.b64(rhs, multiplier, multiplier)
            K.ptx.mov.b64(addend, addend_value, addend_value)
            K.ptx.fma.rz.ftz.f32x2(packed, packed, rhs, addend)
            K.ptx.mov.b64(values[idx], values[idx + 1], packed)

        def mul_f32x2(values, idx, multiplier):
            packed = K.local_scalar("uint64")
            rhs = K.local_scalar("uint64")
            K.ptx.mov.b64(packed, values[idx], values[idx + 1])
            K.ptx.mov.b64(rhs, multiplier, multiplier)
            K.ptx.mul.rz.ftz.f32x2(packed, packed, rhs)
            K.ptx.mov.b64(values[idx], values[idx + 1], packed)

        def reduce_max_128(out_, values, accum=False):
            """Row max over 128 values with MAX_CHAINS independent max3 chains."""
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

        def reduce_sum_128(out_, values, accum=False):
            local_sum = K.alloc_local([8], "float32")
            packed = K.local_scalar("uint64")
            rhs = K.local_scalar("uint64")
            for i in range(8):
                if accum and i == 0:
                    K.ptx.mov.b32(local_sum[i], values[i] + out_[0])
                else:
                    K.ptx.mov.b32(local_sum[i], values[i])
            with K.serial(15) as outer:
                for i in range(4):
                    K.ptx.mov.b64(packed, local_sum[2 * i], local_sum[2 * i + 1])
                    K.ptx.mov.b64(
                        rhs, values[8 * (outer + 1) + 2 * i], values[8 * (outer + 1) + 2 * i + 1]
                    )
                    K.ptx.add.rn.ftz.f32x2(packed, packed, rhs)
                    K.ptx.mov.b64(local_sum[2 * i], local_sum[2 * i + 1], packed)
            for lo, hi in ((0, 2), (4, 6), (0, 4)):
                K.ptx.mov.b64(packed, local_sum[lo], local_sum[lo + 1])
                K.ptx.mov.b64(rhs, local_sum[hi], local_sum[hi + 1])
                K.ptx.add.rn.ftz.f32x2(packed, packed, rhs)
                K.ptx.mov.b64(local_sum[lo], local_sum[lo + 1], packed)
            K.assign(out_[0], local_sum[0] + local_sum[1])

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

        POLY_EX2_DEG1 = (1.0290300065, 0.6860200044)
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

        def row_select_mask(kv_head, token):
            """32-bit mask of the blocks selected for (kv_head, token)."""
            mask = K.local_scalar("uint32", init=K.uint32(0))
            base = (kv_head * TOTAL_Q + token) * TOPK
            idxs = K.alloc_local([TOPK], "int32")
            for s in range(TOPK):
                K.ptx.ld.global_.nc.b32(idxs[s], q2k.ptr_to([base + s]))
            for s in range(TOPK):
                bit = K.Select(
                    K.And(idxs[s] >= 0, idxs[s] < MAX_BLOCKS),
                    K.shift_left(K.uint32(1), K.Cast("uint32", K.max(idxs[s], 0))),
                    K.uint32(0),
                )
                K.assign(mask, K.bitwise_or(mask, bit))
            return mask

                                                                              
        sp = K.specialize(chain_dispatch=True)
        r_softmax = sp.role("softmax", warps=[0, 1, 2, 3, 4, 5, 6, 7], regs=216)
        r_idle2 = sp.role("idle2", warps=[8, 9, 10, 11], regs=32)
        wg3 = sp.warpgroup("wg3", warps=range(12, 16), regs=48)
        r_mma = sp.role("mma", warps=[12], group=wg3)
        r_load = sp.role("load", warps=[13], group=wg3)
        r_store = sp.role("store", warps=[14], group=wg3)
        r_idle = sp.role("idle", warps=[15], group=wg3)

                                                                               
                                                                         
        with K.If(warp_cta == 12), K.Then():
            K.ptx[TMEM_ALLOC](K.address_of(tmem_addr[0]), K.uint32(N_COLS_TMEM))
            K.cuda.warp_sync()
        with K.If(tvm.tirx.all(wg_id == 3, warp_id == 0)), K.Then():
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
                                K.ptx.st.shared.b32(union_meta.ptr_to([slot * 4]), K.int32(-1))
                            union_ready.arrive(slot)
                            K.assign(running, 0)
                        with K.Else():
                            tok_base, kv_head = task_coords(task)
                                                                                
                            union_token = iket_range("union-build")
                                                                                
                                                                                 
                            idxs = K.alloc_local([8], "int32")
                            q2k_base = (
                                (kv_head * TOTAL_Q + tok_base + (lane >> 1)) * TOPK
                                + (lane & 1) * 8
                            )
                            for vec in range(2):
                                K.ptx.ld.global_.nc.v4.b32(
                                    idxs[vec * 4],
                                    idxs[vec * 4 + 1],
                                    idxs[vec * 4 + 2],
                                    idxs[vec * 4 + 3],
                                    q2k.ptr_to([q2k_base + vec * 4]),
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
                            union_mask = K.local_scalar("uint32")
                            K.ptx.redux_sync.or_.b32(
                                union_mask, my_mask, K.uint32(0xFFFFFFFF)
                            )
                            K.assign(my_mask, union_mask)
                            q_pos_min = tok_base + causal_off
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
                            K.assign(my_mask, K.bitwise_and(my_mask, vis_mask))
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
                            with K.If(lane == 0), K.Then():
                                K.ptx.st.shared.b32(union_meta.ptr_to([slot * 4]), K.Cast("int32", count))
                                K.ptx.st.shared.b32(
                                    union_meta.ptr_to([slot * 4 + 1]), K.Cast("int32", n_masked)
                                )
                                K.ptx.st.shared.b32(union_meta.ptr_to([slot * 4 + 2]), tok_base)
                                K.ptx.st.shared.b32(union_meta.ptr_to([slot * 4 + 3]), kv_head)
                            blk = K.local_scalar("uint32", init=K.uint32(0))
                            with K.If(K.Cast("uint32", lane) < count), K.Then():
                                K.ptx.fns.b32(
                                    blk,
                                    my_mask,
                                    K.uint32(31),
                                    -K.Cast("int32", lane) - K.int32(1),
                                )
                                K.ptx.st.shared.b32(
                                    union_list.ptr_to([slot * MAX_BLOCKS + lane]),
                                    K.Cast("int32", blk),
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
                            disabled_tokens = K.bitwise_and(
                                K.bitwise_not(selected_tokens), K.uint32(0xFFFF)
                            )
                            with K.If(K.Cast("uint32", lane) < count), K.Then():
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
                                q_load.empty.wait(i_q, q_epoch.phase)
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

                            def load_kv(blk, tensor_map, is_v):
                                kv_load.empty.wait(kv_pipe.stage, kv_pipe.phase)
                                tma_kv_token = iket_range("issue-tma-v" if is_v else "issue-tma-k")
                                with K.If(elected()), K.Then():
                                    K.ptx[TMA_G2S_3D](
                                        (v_smem if is_v else k_smem)[kv_pipe.stage].ptr_to(0, 0),
                                        K.address_of(tensor_map),
                                        K.int32(0),
                                        K.Cast("int32", blk * BLK_N),
                                        K.Cast("int32", kv_head * 2),
                                        K.cuda.cvta_generic_to_shared(
                                            kv_load.full.ptr_to([kv_pipe.stage])
                                        ),
                                    )
                                    kv_load.full.arrive(kv_pipe.stage, tx_count=KV_TILE_BYTES)
                                K.cuda.iket.range_end(tma_kv_token[0])
                                kv_pipe.advance()

                                                                                         
                            rest_l = K.local_scalar("uint32", init=my_mask)
                            lz0 = K.local_scalar("uint32")
                            K.ptx.clz.b32(lz0, rest_l)
                            blk_cur = K.local_scalar("uint32", init=K.uint32(31) - lz0)
                            K.assign(rest_l, K.bitwise_xor(rest_l, K.shift_left(K.uint32(1), blk_cur)))
                            load_kv(K.Cast("int32", blk_cur), k_map, False)
                            with K.serial(K.Cast("int32", count), unroll=False) as _k:
                                with K.If(rest_l != K.uint32(0)):
                                    with K.Then():
                                        lz_l = K.local_scalar("uint32")
                                        K.ptx.clz.b32(lz_l, rest_l)
                                        blk_nxt = K.local_scalar("uint32", init=K.uint32(31) - lz_l)
                                        K.assign(
                                            rest_l,
                                            K.bitwise_xor(rest_l, K.shift_left(K.uint32(1), blk_nxt)),
                                        )
                                        load_kv(K.Cast("int32", blk_nxt), k_map, False)
                                        load_kv(K.Cast("int32", blk_cur), v_map, True)
                                        K.assign(blk_cur, blk_nxt)
                                    with K.Else():
                                        load_kv(K.Cast("int32", blk_cur), v_map, True)
                            K.assign(it, it + 1)

                                                                             
            with r_mma:
                it_m = K.local_scalar("int32", init=0)
                gstep_m = K.local_scalar("int32", init=0)                       

                tb_raw = K.local_scalar("uint32")
                K.ptx.ld.shared.u32(tb_raw, tmem_addr.ptr_to([0]))
                tmem_base = K.local_scalar("uint32", init=K.uniform(tb_raw))

                def load_output_lane_mask(slot, list_idx, q_stage):
                    disabled = K.alloc_local([4], "uint32")
                    for word in range(4):
                        K.ptx.ld.shared.u32(
                            disabled[word],
                            output_lane_masks.ptr_to([slot, list_idx, q_stage, word]),
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

                def gemm_qk_half(q_stage, kv_stage, half, do_commit):
                    """S[q_stage][:, 64*half:+64] = Q @ K[keys 64*half:+64]^T (N=64)."""
                    qk_token = iket_range("mma-qk")
                    for ki in range(HEAD_DIM // MMA_K):
                        with K.If(elected()), K.Then():
                            K.ptx[MMA_F16](
                                K.Cast("uint32", q_stage * MMA_N + 64 * half),
                                desc_at(q_desc, q_stage * Q_STAGE16 + qoff(ki)),
                                desc_at(k_desc, kv_stage * KV_STAGE16 + 512 * half + koff(ki)),
                                K.uint32(ID_QK64),
                                K.uint32(0),
                                K.uint32(0),
                                K.uint32(0),
                                K.uint32(0),
                                ki != 0,
                            )
                    if do_commit:
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
                  n_blocks = ld_shared_i32(union_meta.ptr_to([slot_m * 4]))
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
                it_s = K.local_scalar("int32", init=0)
                running_s = K.local_scalar("int32", init=1)
                with K.While(running_s != 0):
                  slot_s = it_s & 1
                  union_ready.wait(slot_s, (it_s >> 1) & 1)
                  n_blocks_st = ld_shared_i32(union_meta.ptr_to([slot_s * 4]))
                  tok_base_s = ld_shared_i32(union_meta.ptr_to([slot_s * 4 + 2]))
                  kv_head_s = ld_shared_i32(union_meta.ptr_to([slot_s * 4 + 3]))
                  with K.If(n_blocks_st < 0), K.Then():
                    K.assign(running_s, 0)
                  with K.If(n_blocks_st > 0), K.Then():
                                                                                    
                      union_free.arrive(slot_s)
                      store_token = iket_range("tma-store")
                      for i_q in range(N_TILES):
                          o_staged.wait(i_q, it_s & 1)
                          with K.If(elected()), K.Then():
                              K.ptx[TMA_S2G_4D](
                                  K.address_of(o_map),
                                  K.int32(0),
                                  K.Cast("int32", kv_head_s * GQA),
                                  K.Cast("int32", tok_base_s + i_q * TOK_PER_TILE),
                                  K.int32(0),
                                  o_smem[i_q].ptr_to(0, 0),
                              )
                              K.ptx.cp.async_.bulk.commit_group()
                      K.ptx.cp.async_.bulk.wait_group(0)
                      with K.If(elected()), K.Then():
                          for i_q in range(N_TILES):
                              o_smem_free.arrive(i_q)
                      K.cuda.iket.range_end(store_token[0])
                  K.assign(it_s, it_s + 1)

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

            def apply_causal_mask(s_chunk, blk):
                col_limit_right = q_pos - blk * BLK_N + 1
                mask_r2p(s_chunk, col_limit_right, BLK_N)

            def rescale_o_rows(scale):
                """Multiply this thread's O row (128 f32 in TMEM) by ``scale``."""
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

            def softmax_step(blk, other_par, apply_mask=False, is_first=False):
                """One KV block of the online softmax with the fused exp loop.

                ``other_par`` is the parity of the s_consumed completion of the other
                warpgroup that must precede this step's P store into S_{1-i}.
                """
                s_chunk = K.alloc_local([BLK_N], "float32")
                p_chunk = K.alloc_local([BLK_N // 2], "uint32")
                selected = K.local_scalar(
                    "int32",
                    init=K.Cast(
                        "int32",
                        K.bitwise_and(
                            K.shift_right(sel_mask, K.Cast("uint32", blk)), K.uint32(1)
                        ),
                    ),
                )
                s_ready.wait(wg_id, score_epoch.phase)
                with K.If(warp_id == 0), K.Then():
                    K.cuda.iket.mark("softmax-phase-0")
                softmax_max_token = iket_range("softmax-max", leader_only=True)
                tile_max = K.alloc_local([1], "float32")
                for chunk_idx in range(BLK_N // 32):
                    tmem_load(s_chunk, chunk_idx * 32, tmem(wg_id * MMA_N + chunk_idx * 32), 32)
                if apply_mask:
                    apply_causal_mask(s_chunk, blk)
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
                score_epoch.advance()
                                                 
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
                K.assign(gstep_x, gstep_x + 1)

            running_x = K.local_scalar("int32", init=1)
            with K.While(running_x != 0):
              slot_x = it_x & 1
              union_ready.wait(slot_x, (it_x >> 1) & 1)
              n_blocks_s = ld_shared_i32(union_meta.ptr_to([slot_x * 4]))
              with K.If(n_blocks_s < 0), K.Then():
                K.assign(running_x, 0)
              with K.If(n_blocks_s > 0), K.Then():
                n_masked_s = ld_shared_i32(union_meta.ptr_to([slot_x * 4 + 1]))
                tok_base_x = ld_shared_i32(union_meta.ptr_to([slot_x * 4 + 2]))
                kv_head_x = ld_shared_i32(union_meta.ptr_to([slot_x * 4 + 3]))
                my_token = tok_base_x + wg_id * TOK_PER_TILE + tok_local
                K.assign(q_pos, my_token + causal_off)
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
                list_base = slot_x * MAX_BLOCKS
                def other_parity(n):
                                                                                    
                                                                             
                    if_wg0 = gstep_x & 1
                    nxt = K.Select(n + 1 < n_blocks_s, (gstep_x + 1) & 1, gstep_x & 1)
                    return K.Select(wg_id == 0, if_wg0, nxt)

                blk0 = ld_shared_i32(union_list.ptr_to([list_base]))
                softmax_step(blk0, other_parity(K.int32(0)), apply_mask=True, is_first=True)
                n_masked_rest = K.max(n_masked_s - 1, 0)
                with K.serial(n_masked_rest, unroll=False) as i:
                    blk_m = ld_shared_i32(union_list.ptr_to([list_base + 1 + i]))
                    softmax_step(blk_m, other_parity(1 + i), apply_mask=True)
                start_plain = K.max(n_masked_s, 1)
                with K.serial(n_blocks_s - start_plain, unroll=False) as i:
                    blk_p = ld_shared_i32(union_list.ptr_to([list_base + start_plain + i]))
                    softmax_step(blk_p, other_parity(start_plain + i), apply_mask=False)
                union_free.arrive(slot_x)

                                                                                     
                epi_wait_token = iket_range("epi-wait-o", leader_only=True)
                o_ready.wait(wg_id, it_x & 1)
                K.cuda.iket.range_end(epi_wait_token[0])
                epi_token = iket_range("epi-store", leader_only=True)
                acc_O_row_is_zero_or_nan = tvm.tirx.any(
                    row_sum[0] == K.float32(0.0), row_sum[0] != row_sum[0]
                )
                norm_scale = K.local_scalar("float32")
                K.ptx.rcp.approx.ftz.f32(
                    norm_scale, K.Select(acc_O_row_is_zero_or_nan, K.float32(1.0), row_sum[0])
                )
                EPI_LD = 32
                o_row_f32 = K.alloc_local([HEAD_DIM], "float32")
                o_row_bf16 = K.alloc_local([HEAD_DIM // 2], "uint32")
                for d_tile in range(HEAD_DIM // EPI_LD):
                    tmem_load(
                        o_row_f32, d_tile * EPI_LD, tmem((N_TILES + wg_id) * MMA_N + d_tile * EPI_LD), EPI_LD
                    )
                K.ptx.tcgen05.wait__ld.sync.aligned()
                                                                                
                o_free.arrive(wg_id)
                                                                        
                o_smem_free.wait(wg_id, (it_x + 1) & 1)
                for d_tile in range(HEAD_DIM // EPI_LD):
                    d_start = d_tile * EPI_LD
                    for i in range(EPI_LD // 2):
                        mul_f32x2(o_row_f32, d_start + 2 * i, norm_scale)
                    for i in range(EPI_LD // 2):
                        cast_f32x2_bf16x2(o_row_bf16, o_row_f32, d_start + 2 * i)
                    for i in range(EPI_LD // 8):
                        w0 = d_start // 2 + i * 4
                        K.ptx.st.shared.v4.u32(
                            o_smem[wg_id].ptr_to(tid_in_wg, d_start + i * 8),
                            o_row_bf16[w0],
                            o_row_bf16[w0 + 1],
                            o_row_bf16[w0 + 2],
                            o_row_bf16[w0 + 3],
                        )
                K.ptx.fence.proxy.async_.shared__cta()
                o_staged.arrive(wg_id)
                K.cuda.iket.range_end(epi_token[0])
                K.assign(it_x, it_x + 1)

        with r_idle2:
            pass

                                                                                
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

    return msa_prefill_qmajor_union


                                                                             
           
                                                                             
class _AlignedTensorMap:
    """Host storage for one 64-byte-aligned, 128-byte CUtensorMap."""

    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode(tensor, dims, strides, box):
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
        2,                                      
        0,                                     
    )
    return desc


# ---------------------------------------------------------------------------
# Registry interface.
# ---------------------------------------------------------------------------

KERNEL_META = {
    "name": "agent_evolved_msa_prefill_b1_q4096",
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
        "run": "msa_prefill_b1_q4096_kv4096_hq64_hkv4_d128_topk16_bf16_flat-20260910-221452",
        "selected_version": "frontier/qmajor-persistent",
    },
}

CONFIGS = [
    {
        "label": "b1_q4096_kv4096_h64",
        "batch_size": 1,
        "seqlen_q": 4096,
        "seqlen_kv": 4096,
        "num_qo_heads": 64,
        "num_kv_heads": 4,
        "topk": 16,
        "seed": 43,
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
    if int(resolved["batch_size"]) != 1:
        raise ValueError("this kernel implements the flat B=1 contract")
    if int(resolved["num_qo_heads"]) != int(resolved["num_kv_heads"]) * GQA:
        raise ValueError(f"GQA must be {GQA}")
    if int(resolved["seqlen_kv"]) > MAX_BLOCKS * BLK_N:
        raise ValueError(f"kv_len must be <= {MAX_BLOCKS * BLK_N}")
    if int(resolved["seqlen_q"]) % TOK_PER_CTA != 0:
        raise ValueError(f"total_q must be a multiple of {TOK_PER_CTA}")
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


def get_kernel(**config: Any):
    """Return the traced Kern PrimFunc for one compile key."""
    from tirx_kernels.runner import hardware_num_sms

    resolved = _config(**config)
    os.environ.setdefault("TVM_CUDA_PTXAS_REG_LEVEL", "6")
    kernel = make_kernel(
        int(resolved["seqlen_q"]) * int(resolved["batch_size"]),
        int(resolved["num_qo_heads"]),
        int(resolved["num_kv_heads"]),
        int(resolved["topk"]),
        hardware_num_sms(),
    )
    return kernel.func


# ---------------------------------------------------------------------------
# Inputs.
#
# The draw order reproduces the packaged MSA-prefill benchmark row
# `prefill_bf16_b1_q4096_kv4096_h64`, which itself follows flashinfer PR #4355's
# `bench_blackwell_msa_sm100.py`: q, k and v are `randn/3` in one generator
# sequence, then `q2k_indices` is drawn per (query token, kv head) from the
# blocks the token may see under bottom-right causal masking.
# ---------------------------------------------------------------------------


def _make_q2k_indices(seqlen_q, seqlen_kv, num_kv_heads, topk, seed, device):
    total_q = seqlen_q
    out = torch.full((num_kv_heads, total_q, topk), -1, dtype=torch.int32)
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    offset = seqlen_kv - seqlen_q
    for row in range(total_q):
        visible_blocks = ceildiv(offset + row % seqlen_q + 1, BLK_N)
        for kv_head in range(num_kv_heads):
            selected = torch.randperm(visible_blocks, generator=generator)
            selected = selected[: min(topk, visible_blocks)].sort().values
            out[kv_head, row, : selected.numel()] = selected.to(torch.int32)
    return out.to(device)


def prepare_data(**config: Any) -> dict[str, Any]:
    """Build one row's contract tensors plus the preallocated output."""
    resolved = _config(**config)
    device = torch.device("cuda")
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

    q = randn((seqlen_q, num_qo_heads, HEAD_DIM))
    k = randn((seqlen_kv, num_kv_heads, HEAD_DIM))
    v = randn((seqlen_kv, num_kv_heads, HEAD_DIM))
    cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, seqlen_kv], dtype=torch.int32, device=device)
    q2k_indices = _make_q2k_indices(seqlen_q, seqlen_kv, num_kv_heads, topk, seed, device)
    return {
        "config": resolved,
        "q": q,
        "k": k,
        "v": v,
        "q2k_indices": q2k_indices,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "page_table": None,
        "seqused_k": None,
        "softmax_scale": HEAD_DIM**-0.5,
        "output": torch.empty_like(q),
    }


def _tirx_args(case: dict[str, Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Encode the tensor maps and assemble the launch argument tuple."""
    q, k, v = case["q"], case["k"], case["v"]
    q2k = case["q2k_indices"]
    cu_q, cu_k = case["cu_seqlens_q"], case["cu_seqlens_k"]
    out = case["output"]
    total_q, hq, _ = q.shape
    total_k, hkv, _ = k.shape
    scale_log2 = float(case["softmax_scale"]) * LOG2E
    qo_dims = (HEAD_DIM // 2, hq, total_q, 2)
    qo_strides = (
        HEAD_DIM * F16_BYTES,
        hq * HEAD_DIM * F16_BYTES,
        (HEAD_DIM // 2) * F16_BYTES,
    )
    qo_box = (HEAD_DIM // 2, GQA, TOK_PER_TILE, 2)
    q_map = _encode(q, qo_dims, qo_strides, qo_box)
    o_map = _encode(out, qo_dims, qo_strides, qo_box)
    kv_dims = (HEAD_DIM // 2, total_k, hkv * 2)
    kv_strides = (hkv * HEAD_DIM * F16_BYTES, (HEAD_DIM // 2) * F16_BYTES)
    kv_box = (HEAD_DIM // 2, BLK_N, 2)
    k_map = _encode(k, kv_dims, kv_strides, kv_box)
    v_map = _encode(v, kv_dims, kv_strides, kv_box)
    sched = torch.zeros(2, dtype=torch.int32, device=q.device)
    args = (q_map.ptr, k_map.ptr, v_map.ptr, o_map.ptr, q2k.view(-1), cu_q, cu_k, sched, scale_log2)
    keep = (q, k, v, q2k, out, cu_q, cu_k, sched, q_map, k_map, v_map, o_map)
    return args, keep


# ---------------------------------------------------------------------------
# Independent oracle.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _reference_output(case: dict[str, Any], *, chunk: int = 256) -> torch.Tensor:
    """Masked FP32 sparse attention, computed independently of the kernel.

    A query token may attend only tokens of its selected blocks that also sit
    at or before its bottom-right causal position. An empty selection yields a
    zero row.
    """
    q, k, v = case["q"], case["k"], case["v"]
    q2k = case["q2k_indices"]
    scale = float(case["softmax_scale"])
    total_q, hq, _ = q.shape
    total_k, hkv, _ = k.shape
    gqa = hq // hkv
    offset = total_k - total_q
    device = q.device
    out = torch.empty_like(q)
    key_block = torch.arange(total_k, device=device) // BLK_N
    key_pos = torch.arange(total_k, device=device)
    for kv_head in range(hkv):
        k_head = k[:, kv_head].float()
        v_head = v[:, kv_head].float()
        for start in range(0, total_q, chunk):
            stop = min(start + chunk, total_q)
            rows = torch.arange(start, stop, device=device)
            selected = q2k[kv_head, start:stop]
            block_ok = (selected.unsqueeze(-1) == key_block.view(1, 1, -1)).any(1)
            causal_ok = key_pos.view(1, -1) <= (offset + rows).view(-1, 1)
            allowed = block_ok & causal_ok
            q_chunk = q[start:stop, kv_head * gqa : (kv_head + 1) * gqa].float()
            scores = torch.einsum("tgd,kd->tgk", q_chunk, k_head) * scale
            scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))
            empty = ~allowed.any(-1)
            weights = torch.softmax(scores, dim=-1)
            weights = torch.where(empty.unsqueeze(1).unsqueeze(-1), torch.zeros_like(weights), weights)
            out[start:stop, kv_head * gqa : (kv_head + 1) * gqa] = torch.einsum(
                "tgk,kd->tgd", weights, v_head
            ).to(q.dtype)
    return out


def check_correctness(outputs: dict[str, Any], **config: Any) -> None:
    _config(**config)
    first, actual, reference = outputs["first"], outputs["actual"], outputs["reference"]
    for name, tensor in (("first", first), ("actual", actual), ("reference", reference)):
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} output contains non-finite values")
    if not torch.equal(first, actual):
        max_abs = float((first.float() - actual.float()).abs().max())
        raise AssertionError(
            f"identical launches are not exactly repeatable; max abs diff={max_abs}"
        )
    # The contract's acceptance criterion: elementwise atol + rtol*|reference|
    # with atol = rtol = 0.01, evaluated in the native bf16 dtype.
    torch.testing.assert_close(actual, reference, atol=1e-2, rtol=1e-2)
    diff_rms = torch.sqrt(torch.mean((actual.float() - reference.float()).square()))
    reference_rms = torch.sqrt(torch.mean(reference.float().square()))
    rms_ratio = float(diff_rms / (reference_rms + 1e-8))
    if rms_ratio >= 5e-2:
        raise AssertionError(f"normalized RMS error ratio {rms_ratio:.6e} must be below 5e-2")


def run_test(**config: Any) -> None:
    _assert_supported_arch()
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**config)
    executable = compile_kernel(get_kernel(**config))
    args, keep = _tirx_args(case)
    case["output"].fill_(float("nan"))
    executable(*args)
    torch.cuda.synchronize()
    first = case["output"].clone()
    # Poison the buffer so a kernel that skips rows cannot pass by leaving them.
    case["output"].fill_(42.0)
    executable(*args)
    torch.cuda.synchronize()
    actual = case["output"].clone()
    reference = _reference_output(case)
    torch.cuda.synchronize()
    del keep
    check_correctness({"first": first, "actual": actual, "reference": reference}, **config)


# ---------------------------------------------------------------------------
# Benchmark: the MiniMax reference arm and the timed dispatch.
# ---------------------------------------------------------------------------


def _minimax_reference(case: dict[str, Any]):
    """Capture MiniMax's sparse forward in a CUDA graph and return its replay.

    ``build_k2q_csr`` turns the contract's ``q2k_indices`` into the kernel's CSR
    reverse index and forward schedule; that build, the JIT and the workspace
    are prepare work, exactly as flashinfer PR #4355's
    ``bench_blackwell_msa_sm100.py`` does it with
    ``baseline_mode="minimax_public"``. The capture removes the host gap
    between MiniMax's forward and its combine so the timed span is the kernels
    alone; the replay is bitwise equal to the eager call.
    """
    import fmha_sm100

    q, k, v = case["q"], case["k"], case["v"]
    q2k = case["q2k_indices"]
    cu_q, cu_k = case["cu_seqlens_q"], case["cu_seqlens_k"]
    q_lens = cu_q[1:] - cu_q[:-1]
    kv_lens = cu_k[1:] - cu_k[:-1]
    max_seqlen_q = int(q_lens.max())
    max_seqlen_k = int(kv_lens.max())
    total_rows = int(((kv_lens + BLK_N - 1) // BLK_N).sum())
    k2q_row_ptr, k2q_q_indices, schedule = fmha_sm100.build_k2q_csr(
        q2k,
        cu_q,
        cu_k,
        BLK_N,
        total_k=int(cu_k[-1]),
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        total_rows=total_rows,
        qhead_per_kv=q.shape[1] // k.shape[1],
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
            blk_kv=BLK_N,
            causal=True,
            softmax_scale=float(case["softmax_scale"]),
            return_softmax_lse=False,
            page_table=None,
            seqused_k=None,
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
    state = (graph, k2q_row_ptr, k2q_q_indices, schedule)

    def replay(_state=state):
        _state[0].replay()

    replay._keep_alive = state
    return replay


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

    results = bench(
        {"tirx": lambda: executable(*args)},
        references={"minimax_msa": lambda: _minimax_reference(case)},
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
