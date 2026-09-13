# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved SM100a Video Sparse Attention (VSA) forward, all official shapes.

This supersedes the single-shape `vsa_s80000_h8_topk156` port: it covers every
official VSA row rather than the one pinned blk128 shape, at the same measured
speedup on that shape (1.426x vs 1.411x through the evolution harness, inside
run-to-run noise).

Supported contract: contiguous bf16 q/k/v/output of shape [S, H, 128],
`q2k_indices` int32[H, S/block_size, topk] holding a sorted list of distinct KV
block ids per (head, query block), `block_size` of 64 or 128, and
`kv_block_lens` int32[S/block_size] or None (block_size=64 only). There is no
causal mask and no GQA.

The selected kernel is the `cross-aliased-tmem` frontier member of the
2026-09-11 multi-shape VSA evolution run, which generalized the merged
single-shape kernel to 18 official rows with no row below parity. Everything
from the module docstring's mechanism notes down to `_get_executable` is that
candidate's source; this module adds the registry interface, input generation,
the independent oracle, and the FlashInfer reference arms.

It is pure bf16: q/k/v and P are bf16 and every tcgen05 MMA is `kind::f16` with
FP32 accumulation. No operand is quantized.

Candidate mechanism notes, carried over from the evolution run:


block_size=128 ("dual-stream 128-row" family): one CTA owns one 128-row query
block at a time and walks its KV block list in 128-key stages that alternate
between two online-softmax streams (two S/P TMEM regions, two O accumulators);
a correction warpgroup rescales O and combines the two streams per tile.

block_size=64 ("ws64 quad" family): one CTA owns one 64-row query block; each
stage packs four 64-key KV blocks into one N=256 ``tcgen05.mma.ws`` (M=64,
tensor-memory layout E), so the two 64-lane halves compute independent online
softmaxes over disjoint key halves and are combined at the end of the task.
No preprocessing kernel: both kernels read q2k_indices (and kv_block_lens)
directly, so one launch per call.

Layout contract: q/k/v/out contiguous bf16[S, H, 128]; q2k_indices
int32[H, S/block_size, topk]; kv_block_lens int32[S/block_size] or None.
"""

import ctypes
import math
from typing import Any
from unittest import SkipTest

import torch
import tvm

import tirx_kernels.kern as K

D = 128
TILE_ROWS = 128
TILE_KEYS = 128
KV_STAGES = 5
TMEM_COLS = 512
NEG_INF = -float("inf")
LOG2_E = 1.4426950408889634
TILE_BYTES = TILE_ROWS * D * 2
STAGE16 = TILE_BYTES // 16
REGS_SOFTMAX = 184
REGS_CORRECTION = 96
REGS_OTHER = 48

QK_IDESC = 0x08200490                                            
PV_IDESC = 0x08210490                                                         
MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
TCGEN_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
TMEM_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TMEM_LD16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
TMEM_ST16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
TMA_G2S = "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
TMA_S2G = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint"
TMA_PREFETCH_L2 = "cp.async.bulk.prefetch.tensor.3d.L2.global.tile"
POLICY_EVICT_FIRST = 0x12F0000000000000
POLICY_EVICT_LAST = 0x14F0000000000000

                                             
ENTRY_ID_BITS = 20
ENTRY_LEN_SHIFT = 20
ENTRY_MASK_SHIFT = 27
PREP_MAX_TOPK = 2048                                               


def _arch():
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}a"


                                                                             
                                                                                 
                                                                             

EMU_MODE = "none"                                                                            
POLY_EX2_3 = (1.0, 0.695146143436431884765625, 0.227564394474029541015625, 0.077119089663028717041015625)
FP32_ROUND_INT = float(2**23 + 2**22)


_EMU = {"mode": EMU_MODE}


def set_emu_mode(mode):
    _EMU["mode"] = mode


def emulate_pair(fragment, pair):
    EMU_MODE = _EMU["mode"]
    if EMU_MODE == "quarter":
        return pair % 4 == 3
    if EMU_MODE == "half":
        return pair % 2 == 1
    if EMU_MODE == "cudnn":
        return (pair * 2) % 10 >= 6 and fragment < 3
    return False


def iket_range(name):
    token = K.alloc_local([1], "uint32")
    K.assign(token[0], K.cuda.iket.range_start(name))
    return token


def iket_end(token):
    K.cuda.iket.range_end(token[0])


def exp2(value):
    out = K.local_scalar("float32")
    K.ptx.ex2.approx.ftz.f32(out, value)
    return out


def rcp(value):
    out = K.local_scalar("float32")
    K.ptx.rcp.approx.ftz.f32(out, value)
    return out


def packed(op, dst, base, a0, a1, b0, b1, c0=None, c1=None):
    lhs = K.local_scalar("uint64")
    rhs = K.local_scalar("uint64")
    result = K.local_scalar("uint64")
    K.ptx.mov.b64(lhs, a0, a1)
    K.ptx.mov.b64(rhs, b0, b1)
    if c0 is None:
        K.ptx[op](result, lhs, rhs)
    else:
        addend = K.local_scalar("uint64")
        K.ptx.mov.b64(addend, c0, c1)
        K.ptx[op](result, lhs, rhs, addend)
    K.ptx.mov.b64(dst[base], dst[base + 1], result)


def reduce_max_128(values, initial=None):
    acc = K.alloc_local((4,), "float32")
    if initial is None:
        K.ptx.max.f32(acc[0], values[0], values[1])
    else:
        K.ptx.max.f32(acc[0], initial, values[0], values[1])
    K.ptx.max.f32(acc[1], values[2], values[3])
    K.ptx.max.f32(acc[2], values[4], values[5])
    K.ptx.max.f32(acc[3], values[6], values[7])
    for group in range(1, 16):
        base = group * 8
        K.ptx.max.f32(acc[0], acc[0], values[base], values[base + 1])
        K.ptx.max.f32(acc[1], acc[1], values[base + 2], values[base + 3])
        K.ptx.max.f32(acc[2], acc[2], values[base + 4], values[base + 5])
        K.ptx.max.f32(acc[3], acc[3], values[base + 6], values[base + 7])
    K.ptx.max.f32(acc[0], acc[0], acc[1])
    K.ptx.max.f32(acc[0], acc[0], acc[2], acc[3])
    return acc[0]


def packed_sum_128(values, old_sum, old_scale, first):
    acc = K.alloc_local((8,), "float32")
    for j in range(8):
        K.assign(acc[j], values[j])
    if not first:
        scaled_old = K.local_scalar("float32", init=old_sum * old_scale)
        packed("add.rn.f32x2", acc, 0, values[0], values[1], scaled_old, K.float32(0.0))
    for group in range(1, 16):
        base = group * 8
        for pair in range(4):
            packed(
                "add.rn.f32x2",
                acc,
                pair * 2,
                acc[pair * 2],
                acc[pair * 2 + 1],
                values[base + pair * 2],
                values[base + pair * 2 + 1],
            )
    for lo, hi in ((0, 2), (4, 6), (0, 4)):
        packed("add.rn.f32x2", acc, lo, acc[lo], acc[lo + 1], acc[hi], acc[hi + 1])
    return acc[0] + acc[1]


def combine_int_frac_ex2(x_rounded, frac_ex2):
    rounded_i = K.local_scalar("int32")
    frac_i = K.local_scalar("int32")
    exponent = K.local_scalar("int32")
    bits = K.local_scalar("int32")
    out = K.local_scalar("float32")
    K.ptx.mov.b32(rounded_i, x_rounded)
    K.ptx.mov.b32(frac_i, frac_ex2)
    K.ptx.shl.b32(exponent, rounded_i, K.uint32(23))
    K.ptx.add.s32(bits, exponent, frac_i)
    K.ptx.mov.b32(out, bits)
    return out


def ex2_emulation_2(values, base):
    """Two-lane exp2 on the FMA datapath (degree-3 polynomial on the fraction)."""
    clamped = K.alloc_local((2,), "float32")
    K.ptx.max.f32(clamped[0], values[base], K.float32(-127.0))
    K.ptx.max.f32(clamped[1], values[base + 1], K.float32(-127.0))
    packed_v = K.local_scalar("uint64")
    rhs = K.local_scalar("uint64")
    addend = K.local_scalar("uint64")
    rounded = K.alloc_local((2,), "float32")
    K.ptx.mov.b64(packed_v, clamped[0], clamped[1])
    K.ptx.mov.b64(rhs, K.float32(FP32_ROUND_INT), K.float32(FP32_ROUND_INT))
    K.ptx.add.rm.f32x2(packed_v, packed_v, rhs)
    K.ptx.mov.b64(rounded[0], rounded[1], packed_v)
    rounded_back = K.alloc_local((2,), "float32")
    K.ptx.mov.b64(packed_v, rounded[0], rounded[1])
    K.ptx.sub.rn.f32x2(packed_v, packed_v, rhs)
    K.ptx.mov.b64(rounded_back[0], rounded_back[1], packed_v)
    frac = K.alloc_local((2,), "float32")
    K.ptx.mov.b64(packed_v, clamped[0], clamped[1])
    K.ptx.mov.b64(rhs, rounded_back[0], rounded_back[1])
    K.ptx.sub.rn.f32x2(packed_v, packed_v, rhs)
    K.ptx.mov.b64(frac[0], frac[1], packed_v)
    poly = K.alloc_local((2,), "float32")
    K.assign(poly[0], K.float32(POLY_EX2_3[3]))
    K.assign(poly[1], K.float32(POLY_EX2_3[3]))
    for coeff in (POLY_EX2_3[2], POLY_EX2_3[1], POLY_EX2_3[0]):
        K.ptx.mov.b64(packed_v, poly[0], poly[1])
        K.ptx.mov.b64(rhs, frac[0], frac[1])
        K.ptx.mov.b64(addend, K.float32(coeff), K.float32(coeff))
        K.ptx.fma.rn.f32x2(packed_v, packed_v, rhs, addend)
        K.ptx.mov.b64(poly[0], poly[1], packed_v)
    K.assign(values[base], combine_int_frac_ex2(rounded[0], poly[0]))
    K.assign(values[base + 1], combine_int_frac_ex2(rounded[1], poly[1]))


def tmem_load32(dst, address):
    K.ptx[TMEM_LD32](*(dst[i] for i in range(32)), K.cast(address, "uint32"))


def tmem_load16(dst, address):
    K.ptx[TMEM_LD16](*(dst[i] for i in range(16)), K.cast(address, "uint32"))


def tmem_load32_at(dst, base, address):
    K.ptx[TMEM_LD32](*(dst[base + i] for i in range(32)), K.cast(address, "uint32"))


def tmem_load16_at(dst, base, address):
    K.ptx[TMEM_LD16](*(dst[base + i] for i in range(16)), K.cast(address, "uint32"))


def tmem_store16(src, address):
    K.ptx[TMEM_ST16](K.cast(address, "uint32"), *(src[i] for i in range(16)))


def encode(view, major):
    """Hoisted tcgen05 smem descriptor as warp-uniform (lo, hi) halves plus the k-step map."""
    desc, off16 = view.encode(major=major, mma_k=16)
    lo = K.alloc_local((1,), "uint32")
    hi = K.alloc_local((1,), "uint32")
    K.assign(lo[0], K.uniform(K.Cast("uint32", desc.value)))
    K.assign(hi[0], K.Cast("uint32", K.shift_right(desc.value, K.uint64(32))))
    return (lo, hi), off16


def desc_at(desc, off16):
    lo, hi = desc
    low = lo[0] if isinstance(off16, int) and off16 == 0 else lo[0] + K.Cast("uint32", off16)
    packed_desc = K.alloc_local((1,), "uint64")
    K.assign(
        packed_desc[0],
        K.bitwise_or(K.shift_left(K.Cast("uint64", hi[0]), K.uint64(32)), K.Cast("uint64", low)),
    )
    return packed_desc[0]


def apply_column_mask(score, len0, len1):
    """Keep column c of a 128-wide score row iff c - 64*half < len_half; others become -inf."""
    for quarter in range(4):
        length = len0 if quarter < 2 else len1
        shift = K.max((quarter % 2 + 1) * 32 - length, 0)
        mask = K.local_scalar("uint32")
        K.ptx.shr.u32(mask, K.uint32(0xFFFFFFFF), K.cast(shift, "uint32"))
        for bit in range(32):
            live = K.bitwise_and(mask, K.shift_left(K.uint32(1), K.uint32(bit))) != K.uint32(0)
            index = quarter * 32 + bit
            K.assign(score[index], K.if_then_else(live, score[index], K.float32(NEG_INF)))


                                                                             
                                                                            
                                                                             


def make_attention_kernel_blk128(
    arch, static_grid=None, complete_pairs=False, sid_fifo=False, static_shape=None
):
    """``static_grid`` pins the blockIdx extent for the pre-GPU checkers; production uses
    the ``num_ctas`` parameter as the grid."""
    HALF = False

    def body(
        q_map,
        k_map,
        v_map,
        o_map,
        blocks,
        counts,
        num_tasks,
        tiles_per_head,
        topk,
        entry_stride,
        num_ctas,
        scale_log2,
    ):
        if static_shape is None:
            task_count = num_tasks
            tile_count = tiles_per_head
            list_topk = topk
            cta_stride = num_ctas
        else:
            task_count, tile_count, list_topk, cta_stride = (
                int(value) for value in static_shape
            )
        cta = K.cta_id()
        warp = K.warp_id()
        tid = K.thread_id()
        lane = tid & 31

        with K.If(warp == 0), K.Then():
            K.ptx.prefetch.tensormap(K.address_of(q_map))
            K.ptx.prefetch.tensormap(K.address_of(k_map))
            K.ptx.prefetch.tensormap(K.address_of(v_map))
            K.ptx.prefetch.tensormap(K.address_of(o_map))
        if PREFETCH_IDS and not HALF:
            with K.If((warp == 14) & (cta < task_count)), K.Then():
                first_line = K.local_scalar("int32", init=lane)
                with K.While(first_line * 32 < list_topk + 32):
                    K.ptx.prefetch.global_.L2(blocks.ptr_to([cta * list_topk + first_line * 32]))
                    K.assign(first_line, first_line + 32)

        smem = K.smem_pool()
        q_smem = smem.alloc((TILE_ROWS, D), K.bf16, swizzle=K.SW128B)
        kv_smem = smem.alloc((KV_STAGES, TILE_KEYS, D), K.bf16, swizzle=K.SW128B)
        o_smem = smem.alloc((TILE_ROWS, D), K.bf16, swizzle=K.SW128B)
        stats_smem = smem.alloc((512,), K.f32, align=16)
        tmem_mailbox = smem.alloc((1,), K.u32, align=8)
        pool = smem.pool
        pre_leader = tid == 14 * 32
        pre_leader_u = K.local_scalar("uint32", init=K.if_then_else(tid == 14 * 32, K.uint32(1), K.uint32(0)))
        q_full = K.TMABar(pool, 1, leader=pre_leader)
        q_empty = K.TCGen05Bar(pool, 1, leader=pre_leader)
        kv_full = K.TMABar(pool, KV_STAGES, leader=pre_leader)
        kv_empty = K.TCGen05Bar(pool, KV_STAGES, leader=pre_leader)
        spo_full = K.TCGen05Bar(pool, 2, leader=pre_leader)
        spo_empty = K.MBarrier(pool, 2, leader=pre_leader)
        plast_full = K.MBarrier(pool, 2, leader=pre_leader)
        s_consumed = K.MBarrier(pool, 4, leader=pre_leader)
        pv_done = K.TCGen05Bar(pool, 2, leader=pre_leader)
        oacc_full = K.TCGen05Bar(pool, 2, leader=pre_leader)
        stats_empty = K.MBarrier(pool, 2, leader=pre_leader)
        oepi_full = K.MBarrier(pool, 1, leader=pre_leader)
        oepi_empty = K.MBarrier(pool, 1, leader=pre_leader)

        q_full.init(1)
        q_empty.init(1)
        kv_full.init(1)
        kv_empty.init(1)
        spo_full.init(1)
        spo_empty.init(256)
        plast_full.init(4)
        s_consumed.init(128)
        pv_done.init(1)
        oacc_full.init(1)
        stats_empty.init(128)
        oepi_full.init(128)
        oepi_empty.init(1)
        K.ptx.fence.proxy.async_.shared__cta()
        K.ptx.fence.mbarrier_init.release.cluster()
        if EARLY_ISSUE and not HALF:
            sid0_pre = K.local_scalar("int32", init=0)
            with K.If(warp == 14), K.Then():
                K.cuda.warp_sync()
                with K.If(cta < task_count), K.Then():
                    head_pre = cta // tile_count
                    tile_pre = cta - head_pre * tile_count
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        q_full.ptr_to([0]), K.uint32(TILE_BYTES), pred=pre_leader_u
                    )
                    for band in range(2):
                        K.ptx[TMA_G2S](
                            q_smem.ptr_to(0, band * 64),
                            K.address_of(q_map),
                            K.int32(0),
                            tile_pre * TILE_ROWS,
                            head_pre * 2 + band,
                            K.cuda.cvta_generic_to_shared(q_full.ptr_to([0])),
                            K.uint64(POLICY_EVICT_FIRST),
                            pred=pre_leader_u,
                        )
                    K.ptx.ld.global_.s32(sid0_pre, blocks.ptr_to([cta * list_topk]))
        K.cuda.cta_sync()

        roles = K.specialize(chain_dispatch=True)
        r_softmax = roles.role("softmax", warps=range(0, 8), regs=REGS_SOFTMAX)
        r_correction = roles.role("correction", warps=range(8, 12), regs=REGS_CORRECTION)
        r_mma = roles.role("mma", warps=[12], regs=REGS_OTHER)
        r_epilogue = roles.role("epilogue", warps=[13], regs=REGS_OTHER)
        r_load = roles.role("load", warps=[14], regs=REGS_OTHER)
        r_idle = roles.role("idle", warps=[15], regs=REGS_OTHER)

                                                                              

        def exp2(value):
            out = K.local_scalar("float32")
            K.ptx.ex2.approx.ftz.f32(out, value)
            return out

        def rcp(value):
            out = K.local_scalar("float32")
            K.ptx.rcp.approx.ftz.f32(out, value)
            return out

        def packed(op, dst, base, a0, a1, b0, b1, c0=None, c1=None):
            lhs = K.local_scalar("uint64")
            rhs = K.local_scalar("uint64")
            result = K.local_scalar("uint64")
            K.ptx.mov.b64(lhs, a0, a1)
            K.ptx.mov.b64(rhs, b0, b1)
            if c0 is None:
                K.ptx[op](result, lhs, rhs)
            else:
                addend = K.local_scalar("uint64")
                K.ptx.mov.b64(addend, c0, c1)
                K.ptx[op](result, lhs, rhs, addend)
            K.ptx.mov.b64(dst[base], dst[base + 1], result)

        def reduce_max_128(values, initial=None):
            acc = K.alloc_local((4,), "float32")
            if initial is None:
                K.ptx.max.f32(acc[0], values[0], values[1])
            else:
                K.ptx.max.f32(acc[0], initial, values[0], values[1])
            K.ptx.max.f32(acc[1], values[2], values[3])
            K.ptx.max.f32(acc[2], values[4], values[5])
            K.ptx.max.f32(acc[3], values[6], values[7])
            for group in range(1, 16):
                base = group * 8
                K.ptx.max.f32(acc[0], acc[0], values[base], values[base + 1])
                K.ptx.max.f32(acc[1], acc[1], values[base + 2], values[base + 3])
                K.ptx.max.f32(acc[2], acc[2], values[base + 4], values[base + 5])
                K.ptx.max.f32(acc[3], acc[3], values[base + 6], values[base + 7])
            K.ptx.max.f32(acc[0], acc[0], acc[1])
            K.ptx.max.f32(acc[0], acc[0], acc[2], acc[3])
            return acc[0]

        def packed_sum_128(values, old_sum, old_scale, first):
            acc = K.alloc_local((8,), "float32")
            for j in range(8):
                K.assign(acc[j], values[j])
            if not first:
                scaled_old = K.local_scalar("float32", init=old_sum * old_scale)
                packed("add.rn.f32x2", acc, 0, values[0], values[1], scaled_old, K.float32(0.0))
            for group in range(1, 16):
                base = group * 8
                for pair in range(4):
                    packed(
                        "add.rn.f32x2",
                        acc,
                        pair * 2,
                        acc[pair * 2],
                        acc[pair * 2 + 1],
                        values[base + pair * 2],
                        values[base + pair * 2 + 1],
                    )
            for lo, hi in ((0, 2), (4, 6), (0, 4)):
                packed("add.rn.f32x2", acc, lo, acc[lo], acc[lo + 1], acc[hi], acc[hi + 1])
            return acc[0] + acc[1]

        def tmem_load32(dst, address):
            K.ptx[TMEM_LD32](*(dst[i] for i in range(32)), K.cast(address, "uint32"))

        def tmem_load16(dst, address):
            K.ptx[TMEM_LD16](*(dst[i] for i in range(16)), K.cast(address, "uint32"))

        def tmem_store16(src, address):
            K.ptx[TMEM_ST16](K.cast(address, "uint32"), *(src[i] for i in range(16)))

        def stats_arrive(stage, warp_in_group):
            K.ptx.bar.arrive(K.cast(3 + stage * 4 + warp_in_group, "uint32"), K.uint32(64))

        def stats_sync(stage, warp_in_group):
            K.ptx.bar.sync(K.cast(3 + stage * 4 + warp_in_group, "uint32"), K.uint32(64))

        def ld_stats(index):
            value = K.local_scalar("float32")
            K.ptx.ld.shared.f32(value, stats_smem.ptr_to([index]))
            return value

        def st_stats(index, value):
            K.ptx.st.shared.f32(stats_smem.ptr_to([index]), value)

        def encode(view, major):
            desc, off16 = view.encode(major=major, mma_k=16)
            lo = K.alloc_local((1,), "uint32")
            hi = K.alloc_local((1,), "uint32")
            K.assign(lo[0], K.uniform(K.Cast("uint32", desc.value)))
            K.assign(hi[0], K.Cast("uint32", K.shift_right(desc.value, K.uint64(32))))
            return (lo, hi), off16

        def desc_at(desc, off16):
            lo, hi = desc
            low = lo[0] if isinstance(off16, int) and off16 == 0 else lo[0] + K.Cast("uint32", off16)
            packed_desc = K.alloc_local((1,), "uint64")
            K.assign(
                packed_desc[0],
                K.bitwise_or(
                    K.shift_left(K.Cast("uint64", hi[0]), K.uint64(32)), K.Cast("uint64", low)
                ),
            )
            return packed_desc[0]

        def iket_range(name):
            token = K.alloc_local([1], "uint32")
            K.assign(token[0], K.cuda.iket.range_start(name))
            return token

        def iket_end(token):
            K.cuda.iket.range_end(token[0])

        def ld_i32(buffer, index, pred=None):
            value = K.local_scalar("int32", init=0)
            if pred is None:
                K.ptx.ld.global_.s32(value, buffer.ptr_to([index]))
            else:
                K.ptx.ld.global_.s32(value, buffer.ptr_to([index]), pred=pred)
            return value

        def task_coords(task):
            head = K.local_scalar("int32", init=task // tile_count)
            tile = K.local_scalar("int32", init=task - head * tile_count)
            return head, tile

        def task_iters(task, head, tile):
            """Even number of 128-key stages for this task."""
            if HALF:
                return K.local_scalar("int32", init=ld_i32(counts, task))
            if complete_pairs:
                return K.local_scalar("int32", init=list_topk)
            return K.local_scalar("int32", init=((list_topk + 1) >> 1) * 2)

        def list_base(task, head, tile):
            if HALF:
                return K.local_scalar("int32", init=task * entry_stride)
            return K.local_scalar("int32", init=(head * tile_count + tile) * list_topk)

                                                                                
        with r_idle:
            pass

                                                                                
        with r_load:
            leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
            q_phase = K.local_scalar("int32", init=1)
            kv_stage = K.local_scalar("int32", init=0)
            kv_phase = K.local_scalar("int32", init=1)
            task = K.local_scalar("int32", init=cta)

            def advance_kv():
                K.assign(kv_stage, kv_stage + 1)
                with K.If(kv_stage == KV_STAGES), K.Then():
                    K.assign(kv_stage, 0)
                    K.assign(kv_phase, kv_phase ^ 1)

            with K.While(task < task_count):
                head, tile = task_coords(task)
                n_iter = task_iters(task, head, tile)
                base = list_base(task, head, tile)
                head2 = head * 2

                def load_sid(i):
                    if complete_pairs:
                        return ld_i32(blocks, base + i, pred=leader)
                    return ld_i32(blocks, base + K.min(i, list_topk - 1), pred=leader)

                def load_kv(is_v, i, sid=None):
                    tok = iket_range("load-wait-empty")
                    kv_empty.wait(kv_stage, kv_phase)
                    iket_end(tok)
                    tmap = v_map if is_v else k_map
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        kv_full.ptr_to([kv_stage]), K.uint32(TILE_BYTES), pred=leader
                    )
                    bar = K.cuda.cvta_generic_to_shared(kv_full.ptr_to([kv_stage]))
                    if HALF:
                        for c in range(2):
                            entry = ld_i32(blocks, base + i * 2 + c, pred=leader)
                            sid = K.bitwise_and(entry, (1 << ENTRY_ID_BITS) - 1)
                            for band in range(2):
                                K.ptx[TMA_G2S](
                                    kv_smem[kv_stage].ptr_to(c * 64, band * 64),
                                    K.address_of(tmap),
                                    K.int32(0),
                                    sid * 64,
                                    head2 + band,
                                    bar,
                                    K.uint64(POLICY_EVICT_LAST),
                                    pred=leader,
                                )
                    else:
                        if not sid_fifo:
                            sid = load_sid(i)
                        for band in range(2):
                            K.ptx[TMA_G2S](
                                kv_smem[kv_stage].ptr_to(0, band * 64),
                                K.address_of(tmap),
                                K.int32(0),
                                sid * 128,
                                head2 + band,
                                bar,
                                K.uint64(POLICY_EVICT_LAST),
                                pred=leader,
                            )
                    advance_kv()

                def issue_q():
                    q_empty.wait(0, q_phase)
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        q_full.ptr_to([0]), K.uint32(TILE_BYTES), pred=leader
                    )
                    for band in range(2):
                        K.ptx[TMA_G2S](
                            q_smem.ptr_to(0, band * 64),
                            K.address_of(q_map),
                            K.int32(0),
                            tile * TILE_ROWS,
                            head2 + band,
                            K.cuda.cvta_generic_to_shared(q_full.ptr_to([0])),
                            K.uint64(POLICY_EVICT_FIRST),
                            pred=leader,
                        )

                if EARLY_ISSUE and not HALF:
                    with K.If(task != cta), K.Then():
                        issue_q()
                    K.assign(q_phase, q_phase ^ 1)
                    if sid_fifo:
                        sid0 = K.local_scalar("int32", init=0)
                        with K.If(task == cta):
                            with K.Then():
                                K.assign(sid0, sid0_pre)
                            with K.Else():
                                K.assign(sid0, load_sid(K.int32(0)))
                        load_kv(False, K.int32(0), sid0)
                    else:
                        with K.If(task == cta):
                            with K.Then():
                                load_kv(False, K.int32(0), sid0_pre)
                            with K.Else():
                                load_kv(False, K.int32(0))
                else:
                    issue_q()
                    K.assign(q_phase, q_phase ^ 1)
                    if sid_fifo:
                        sid0 = load_sid(K.int32(0))
                        load_kv(False, K.int32(0), sid0)
                    else:
                        load_kv(False, K.int32(0))
                                                                                    
                                                                                     
                next_task = K.local_scalar("int32", init=task + cta_stride)
                with K.If((next_task < task_count) & (leader != K.uint32(0))), K.Then():
                    next_head, next_tile = task_coords(next_task)
                    for band in range(2):
                        K.ptx[TMA_PREFETCH_L2](
                            K.address_of(q_map),
                            K.int32(0),
                            next_tile * TILE_ROWS,
                            next_head * 2 + band,
                        )
                    if PREFETCH_IDS and not HALF:
                        line = K.local_scalar("int32", init=0)
                        with K.While(line * 32 < list_topk + 32):
                            K.ptx.prefetch.global_.L2(blocks.ptr_to([next_task * list_topk + line * 32]))
                            K.assign(line, line + 1)
                if sid_fifo:
                    sid1 = load_sid(K.int32(1))
                    load_kv(False, K.int32(1), sid1)
                    i = K.local_scalar("int32", init=0)
                    with K.While(i < n_iter - 2):
                        load_kv(True, i, sid0)
                        next_sid0 = load_sid(i + 2)
                        load_kv(False, i + 2, next_sid0)
                        K.assign(sid0, next_sid0)
                        load_kv(True, i + 1, sid1)
                        next_sid1 = load_sid(i + 3)
                        load_kv(False, i + 3, next_sid1)
                        K.assign(sid1, next_sid1)
                        K.assign(i, i + 2)
                    load_kv(True, n_iter - 2, sid0)
                    load_kv(True, n_iter - 1, sid1)
                else:
                    load_kv(False, K.int32(1))
                    i = K.local_scalar("int32", init=0)
                    with K.While(i < n_iter - 2):
                        load_kv(True, i)
                        load_kv(False, i + 2)
                        K.assign(i, i + 1)
                    load_kv(True, n_iter - 2)
                    load_kv(True, n_iter - 1)
                K.assign(task, task + cta_stride)

                                                                                
        with r_mma:
            K.ptx[TMEM_ALLOC](K.address_of(tmem_mailbox[0]), K.uint32(TMEM_COLS))
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            tmem_base = K.local_scalar("uint32", init=K.uint32(0))
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
            q_desc, qoff = encode(q_smem, "k")
            k_desc, koff = encode(kv_smem[0], "k")
            v_desc, voff = encode(kv_smem[0], "mn")
            q_phase = K.local_scalar("int32", init=0)
            kv_stage = K.local_scalar("int32", init=0)
            kv_phase = K.local_scalar("int32", init=0)
            spo_phase0 = K.local_scalar("int32", init=0)
            spo_phase1 = K.local_scalar("int32", init=0)
                                                                                    
                                                                                   
                                                                     
            cbase = [K.local_scalar("int32", init=0) for _ in range(4)]
            acc0 = K.local_scalar("int32", init=0)
            acc1 = K.local_scalar("int32", init=0)
            task = K.local_scalar("int32", init=cta)

            def advance_kv():
                K.assign(kv_stage, kv_stage + 1)
                with K.If(kv_stage == KV_STAGES), K.Then():
                    K.assign(kv_stage, 0)
                    K.assign(kv_phase, kv_phase ^ 1)

            def commit(bar, stage):
                K.ptx[TCGEN_COMMIT](bar.ptr_to([stage]), pred=leader)

            def issue_qk(s, k_stage):
                for k16 in range(D // 16):
                    K.ptx[MMA_F16](
                        K.cast(tmem_base + s * 128, "uint32"),
                        desc_at(q_desc, qoff(k16)),
                        desc_at(k_desc, k_stage * STAGE16 + koff(k16)),
                        K.uint32(QK_IDESC),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.cast(k16 != 0, "bool"),
                        pred=leader,
                    )

            def issue_pv(s, v_stage, accumulate, phase):
                for k16 in range(TILE_KEYS // 16):
                    if k16 == 6:
                        plast_full.wait(s, phase)
                    K.ptx[MMA_F16](
                        K.cast(tmem_base + 256 + s * 128, "uint32"),
                        K.cast(tmem_base + 64 + (1 - s) * 128 + k16 * 8, "uint32"),
                        desc_at(v_desc, v_stage * STAGE16 + voff(k16)),
                        K.uint32(PV_IDESC),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.cast(accumulate != 0 if k16 == 0 else True, "bool"),
                        pred=leader,
                    )

            with K.While(task < task_count):
                head, tile = task_coords(task)
                n_iter = task_iters(task, head, tile)
                K.assign(acc0, 0)
                K.assign(acc1, 0)
                tok = iket_range("mma-wait-q")
                q_full.wait(0, q_phase)
                iket_end(tok)
                K.assign(q_phase, q_phase ^ 1)
                for s in range(2):
                    tok = iket_range("mma-wait-kv")
                    kv_full.wait(kv_stage, kv_phase)
                    iket_end(tok)
                    issue_qk(s, kv_stage)
                    commit(spo_full, s)
                    commit(kv_empty, kv_stage)
                    advance_kv()
                pair_count = (n_iter - 2) >> 1
                pair = K.local_scalar("int32", init=0)
                with K.While(pair < pair_count):
                    for s in range(2):
                        phase = spo_phase0 if s == 0 else spo_phase1
                        acc = acc0 if s == 0 else acc1
                        tok = iket_range("mma-wait-kv")
                        kv_full.wait(kv_stage, kv_phase)
                        v_stage = K.local_scalar("int32", init=kv_stage)
                        advance_kv()
                        kv_full.wait(kv_stage, kv_phase)
                        iket_end(tok)
                                                                            
                                                                               
                                                                               
                        s_consumed.wait(
                            s * 2 + (pair & 1),
                            (
                                K.if_then_else((pair & 1) == 0, cbase[s * 2], cbase[s * 2 + 1])
                                + (pair >> 1)
                            )
                            & 1,
                        )
                        tok = iket_range("mma-issue-qk")
                        issue_qk(s, kv_stage)
                        commit(spo_full, s)
                        commit(kv_empty, kv_stage)
                        iket_end(tok)
                        advance_kv()
                        tok = iket_range("mma-wait-p")
                        spo_empty.wait(s, phase)
                        iket_end(tok)
                        tok = iket_range("mma-issue-pv")
                        issue_pv(s, v_stage, acc, phase)
                        commit(pv_done, s)
                        iket_end(tok)
                        K.assign(phase, phase ^ 1)
                        K.assign(acc, 1)
                        commit(kv_empty, v_stage)
                    K.assign(pair, pair + 1)
                commit(q_empty, 0)
                for s in range(2):
                    phase = spo_phase0 if s == 0 else spo_phase1
                    acc = acc0 if s == 0 else acc1
                    kv_full.wait(kv_stage, kv_phase)
                    spo_empty.wait(s, phase)
                    issue_pv(s, kv_stage, acc, phase)
                    commit(pv_done, s)
                    commit(oacc_full, s)
                    commit(kv_empty, kv_stage)
                    advance_kv()
                K.assign(spo_phase0, spo_phase0 ^ 1)
                K.assign(spo_phase1, spo_phase1 ^ 1)
                                                                             
                                                                               
                gens = n_iter >> 1
                for s in range(2):
                    K.assign(cbase[s * 2], cbase[s * 2] + ((gens + 1) >> 1))
                    K.assign(cbase[s * 2 + 1], cbase[s * 2 + 1] + (gens >> 1))
                K.assign(task, task + cta_stride)

            K.ptx[TMEM_RELINQUISH]()
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            allocated = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(allocated, tmem_mailbox.ptr_to([0]))
            K.ptx[TMEM_DEALLOC](allocated, K.uint32(TMEM_COLS))

                                                                               
        with r_epilogue:
            leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
            oepi_phase = K.local_scalar("int32", init=0)
            task = K.local_scalar("int32", init=cta)
            with K.While(task < task_count):
                head, tile = task_coords(task)
                tok = iket_range("epi-wait-full")
                oepi_full.wait(0, oepi_phase)
                iket_end(tok)
                for band in range(2):
                    K.ptx[TMA_S2G](
                        K.address_of(o_map),
                        K.int32(0),
                        tile * TILE_ROWS,
                        head * 2 + band,
                        o_smem.ptr_to(0, band * 64),
                        K.uint64(POLICY_EVICT_FIRST),
                        pred=leader,
                    )
                K.ptx.cp.async_.bulk.commit_group()
                K.ptx.cp.async_.bulk.wait_group.read(0)
                K.ptx.mbarrier.arrive.shared.b64(oepi_empty.ptr_to([0]), K.uint32(1), pred=leader)
                K.assign(oepi_phase, oepi_phase ^ 1)
                K.assign(task, task + cta_stride)

                                                                               
        with r_softmax:
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            tmem_base = K.local_scalar("uint32", init=K.uint32(0))
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            s = K.if_then_else(warp < 4, 0, 1)
            local_warp = warp & 3
            row_half = local_warp >> 1
            tid128 = tid & 127
            row_hi = K.shift_left(K.cast(local_warp * 32, "uint32"), K.uint32(16))
            score_phase = K.local_scalar("int32", init=0)
            stats_phase = K.local_scalar("int32", init=1)
                                                                            
                                                               
            pv_phase = K.local_scalar("int32", init=1)
                                                                            
                                                                             
            pbase = [K.local_scalar("int32", init=0) for _ in range(2)]
            row_max = K.local_scalar("float32", init=K.float32(NEG_INF))
            row_sum = K.local_scalar("float32", init=K.float32(0.0))
            task = K.local_scalar("int32", init=cta)

            def apply_column_mask(score, len0, len1):
                """Keep column c iff c - 64*half < len_half; the rest become -inf."""
                for quarter in range(4):
                    length = len0 if quarter < 2 else len1
                    shift = K.max((quarter % 2 + 1) * 32 - length, 0)
                    mask = K.local_scalar("uint32")
                    K.ptx.shr.u32(mask, K.uint32(0xFFFFFFFF), K.cast(shift, "uint32"))
                    for bit in range(32):
                        live = (
                            K.bitwise_and(mask, K.shift_left(K.uint32(1), K.uint32(bit)))
                            != K.uint32(0)
                        )
                        index = quarter * 32 + bit
                        K.assign(
                            score[index],
                            K.if_then_else(live, score[index], K.float32(NEG_INF)),
                        )

            def consume(i, first, base, n_iter):
                tok = iket_range("sm-wait-s")
                spo_full.wait(s, score_phase)
                iket_end(tok)
                tok_c = iket_range("sm-compute")
                score = K.alloc_local((128,), "float32")
                for chunk in range(4):
                    tmem_load32_at(score, chunk * 32, tmem_base + s * 128 + chunk * 32 + row_hi)
                if HALF:
                    lens = []
                    for c in range(2):
                        entry = ld_i32(blocks, base + i * 2 + c)
                        member = (
                            K.bitwise_and(
                                K.shift_right(entry, ENTRY_MASK_SHIFT + row_half), K.int32(1)
                            )
                            != 0
                        )
                        length = K.bitwise_and(K.shift_right(entry, ENTRY_LEN_SHIFT), K.int32(0x7F))
                        lens.append(
                            K.local_scalar("int32", init=K.if_then_else(member, length, 0))
                        )
                    with K.If((lens[0] < 64) | (lens[1] < 64)), K.Then():
                        apply_column_mask(score, lens[0], lens[1])
                else:
                    if not complete_pairs:
                        with K.If(i >= list_topk), K.Then():
                            for j in range(128):
                                K.assign(score[j], K.float32(NEG_INF))

                old_scale = K.local_scalar("float32", init=K.float32(0.0))
                new_max = K.local_scalar("float32")
                max_safe = K.local_scalar("float32")
                if first:
                    tile_max = reduce_max_128(score)
                    K.assign(new_max, tile_max)
                    K.assign(
                        max_safe, K.if_then_else(tile_max != K.float32(NEG_INF), tile_max, 0.0)
                    )
                else:
                    tile_max = reduce_max_128(score, row_max)
                    K.assign(new_max, tile_max)
                    K.assign(
                        max_safe, K.if_then_else(new_max != K.float32(NEG_INF), new_max, 0.0)
                    )
                    delta = K.local_scalar("float32")
                    K.ptx.sub.f32(delta, row_max, max_safe)
                    delta_scaled = K.local_scalar("float32")
                    K.ptx.mul.f32(delta_scaled, delta, scale_log2)
                    K.assign(old_scale, exp2(delta_scaled))
                    with K.If(delta_scaled >= K.float32(-8.0)), K.Then():
                        K.assign(new_max, row_max)
                        K.assign(max_safe, row_max)
                        K.assign(old_scale, K.float32(1.0))
                                                                            
                                                                                 
                                                                                    
                K.ptx.tcgen05.wait__ld.sync.aligned()
                s_consumed.arrive(s * 2 + ((i >> 1) & 1))
                tok = iket_range("sm-wait-stats")
                stats_empty.wait(s, stats_phase)
                iket_end(tok)
                K.assign(stats_phase, stats_phase ^ 1)
                if not first:
                    st_stats(s * 128 + tid128, old_scale)
                stats_arrive(s, local_warp)

                negative_max = K.local_scalar("float32")
                K.ptx.mul.f32(negative_max, max_safe, -scale_log2)
                for pair in range(64):
                    base_idx = pair * 2
                    packed(
                        "fma.rn.f32x2",
                        score,
                        base_idx,
                        score[base_idx],
                        score[base_idx + 1],
                        scale_log2,
                        scale_log2,
                        negative_max,
                        negative_max,
                    )
                for fragment in range(4):
                    for pair in range(16):
                        base_idx = fragment * 32 + pair * 2
                        if emulate_pair(fragment, pair):
                            ex2_emulation_2(score, base_idx)
                        else:
                            K.assign(score[base_idx], exp2(score[base_idx]))
                            K.assign(score[base_idx + 1], exp2(score[base_idx + 1]))
                                                                           
                                                                          
                new_row_sum = packed_sum_128(score, row_sum, old_scale, first)
                                                                          
                                                                            
                                                                           
                for fragment in range(4):
                    packed_p = K.alloc_local((16,), "uint32")
                    for pair in range(16):
                        base_idx = fragment * 32 + pair * 2
                        K.ptx.cvt.rn.bf16x2.f32(packed_p[pair], score[base_idx + 1], score[base_idx])
                    if fragment == 0:
                                                                               
                        pv_done.wait(s, pv_phase)
                        K.assign(pv_phase, pv_phase ^ 1)
                                                                           
                                                                          
                                                                              
                                                                             
                                        
                        peer_gen = K.local_scalar(
                            "int32",
                            init=K.if_then_else(
                                (s == 0) | (i + 2 >= n_iter), i >> 1, (i >> 1) + 1
                            ),
                        )
                        peer_par = peer_gen & 1
                        s_consumed.wait(
                            (1 - s) * 2 + peer_par,
                            (
                                K.if_then_else(peer_par == 0, pbase[0], pbase[1])
                                + (peer_gen >> 1)
                            )
                            & 1,
                        )
                    tmem_store16(
                        packed_p,
                        tmem_base + 64 + (1 - s) * 128 + fragment * 16 + row_hi,
                    )
                    if fragment == 2:
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        spo_empty.arrive(s)
                K.ptx.tcgen05.wait__st.sync.aligned()
                K.cuda.warp_sync()
                with K.If(K.cuda.elect_sync()), K.Then():
                    plast_full.arrive(s)
                iket_end(tok_c)
                K.assign(row_sum, new_row_sum)
                K.assign(row_max, new_max)
                K.assign(score_phase, score_phase ^ 1)

            with K.While(task < task_count):
                head, tile = task_coords(task)
                n_iter = task_iters(task, head, tile)
                base = list_base(task, head, tile)
                K.assign(row_max, K.float32(NEG_INF))
                K.assign(row_sum, K.float32(0.0))
                consume(K.local_scalar("int32", init=s), True, base, n_iter)
                iteration = K.local_scalar("int32", init=1)
                with K.While(iteration < (n_iter >> 1)):
                    consume(K.local_scalar("int32", init=iteration * 2 + s), False, base, n_iter)
                    K.assign(iteration, iteration + 1)
                stats_empty.wait(s, stats_phase)
                K.assign(stats_phase, stats_phase ^ 1)
                st_stats(s * 128 + tid128, row_sum)
                st_stats(256 + s * 128 + tid128, row_max)
                stats_arrive(s, local_warp)
                gens = n_iter >> 1
                K.assign(pbase[0], pbase[0] + ((gens + 1) >> 1))
                K.assign(pbase[1], pbase[1] + (gens >> 1))
                K.assign(task, task + cta_stride)
            K.ptx.bar.arrive(K.uint32(2), K.uint32(416))

                                                                               
        with r_correction:
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            tmem_base = K.local_scalar("uint32", init=K.uint32(0))
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            corr_warp = warp - 8
            tid128 = tid & 127
            row_hi = K.shift_left(K.cast(corr_warp * 32, "uint32"), K.uint32(16))
            oacc_phase = K.local_scalar("int32", init=0)
            oepi_phase = K.local_scalar("int32", init=1)
            pv_corr_phase0 = K.local_scalar("int32", init=0)
            pv_corr_phase1 = K.local_scalar("int32", init=0)
            task = K.local_scalar("int32", init=cta)
            spo_empty.arrive(0)
            spo_empty.arrive(1)

            def rescale_o(s, scale):
                for chunk in range(D // 16):
                    values = K.alloc_local((16,), "float32")
                    address = tmem_base + 256 + s * 128 + chunk * 16 + row_hi
                    tmem_load16(values, address)
                    for pair in range(8):
                        base_idx = pair * 2
                        packed(
                            "mul.rn.f32x2",
                            values,
                            base_idx,
                            values[base_idx],
                            values[base_idx + 1],
                            scale,
                            scale,
                        )
                    tmem_store16(values, address)
                K.ptx.tcgen05.wait__st.sync.aligned()

            def store_combined(scale0, scale1):
                for chunk in range(D // 16):
                    values0 = K.alloc_local((16,), "float32")
                    values1 = K.alloc_local((16,), "float32")
                    combined = K.alloc_local((16,), "float32")
                    tmem_load16(values0, tmem_base + 256 + chunk * 16 + row_hi)
                    tmem_load16(values1, tmem_base + 384 + chunk * 16 + row_hi)
                    for pair in range(8):
                        base_idx = pair * 2
                        scaled0 = K.alloc_local((2,), "float32")
                        scaled1 = K.alloc_local((2,), "float32")
                        packed(
                            "mul.rn.f32x2",
                            scaled0,
                            0,
                            values0[base_idx],
                            values0[base_idx + 1],
                            scale0,
                            scale0,
                        )
                        packed(
                            "mul.rn.f32x2",
                            scaled1,
                            0,
                            values1[base_idx],
                            values1[base_idx + 1],
                            scale1,
                            scale1,
                        )
                        packed(
                            "add.rn.f32x2",
                            combined,
                            base_idx,
                            scaled0[0],
                            scaled0[1],
                            scaled1[0],
                            scaled1[1],
                        )
                    words = K.alloc_local((8,), "uint32")
                    for pair in range(8):
                        K.ptx.cvt.rn.bf16x2.f32(
                            words[pair], combined[pair * 2 + 1], combined[pair * 2]
                        )
                    K.ptx.st.shared.v4.u32(
                        o_smem.ptr_to(tid128, chunk * 16), words[0], words[1], words[2], words[3]
                    )
                    K.ptx.st.shared.v4.u32(
                        o_smem.ptr_to(tid128, chunk * 16 + 8),
                        words[4],
                        words[5],
                        words[6],
                        words[7],
                    )

            with K.While(task < task_count):
                head, tile = task_coords(task)
                n_iter = task_iters(task, head, tile)
                for s in range(2):
                    stats_sync(s, corr_warp)
                    stats_empty.arrive(s)
                pair_count = (n_iter - 2) >> 1
                pair = K.local_scalar("int32", init=0)
                with K.While(pair < pair_count):
                    for s in range(2):
                        pv_corr_phase = pv_corr_phase0 if s == 0 else pv_corr_phase1
                        stats_sync(s, corr_warp)
                        scale = ld_stats(s * 128 + tid128)
                                                                           
                                                                             
                                                                         
                        pv_done.wait(s, pv_corr_phase)
                        K.assign(pv_corr_phase, pv_corr_phase ^ 1)
                        ballot = K.local_scalar("uint32")
                        K.ptx.vote_sync.ballot.b32(
                            ballot, K.ptx.pred(scale < K.float32(1.0)), K.uint32(0xFFFFFFFF)
                        )
                        with K.If(ballot != 0), K.Then():
                            tok = iket_range("corr-rescale")
                            rescale_o(s, scale)
                            iket_end(tok)
                        spo_empty.arrive(s)
                        stats_empty.arrive(s)
                    K.assign(pair, pair + 1)

                sum0 = K.local_scalar("float32")
                sum1 = K.local_scalar("float32")
                maximum0 = K.local_scalar("float32")
                maximum1 = K.local_scalar("float32")
                for s in range(2):
                    stats_sync(s, corr_warp)
                    if s == 0:
                        K.assign(sum0, ld_stats(tid128))
                        K.assign(maximum0, ld_stats(256 + tid128))
                    else:
                        K.assign(sum1, ld_stats(128 + tid128))
                        K.assign(maximum1, ld_stats(384 + tid128))
                    stats_empty.arrive(s)
                valid0 = (sum0 != K.float32(0.0)) & (sum0 == sum0)
                valid1 = (sum1 != K.float32(0.0)) & (sum1 == sum1)
                rm0 = K.if_then_else(valid0, maximum0, K.float32(NEG_INF))
                rm1 = K.if_then_else(valid1, maximum1, K.float32(NEG_INF))
                maximum = K.local_scalar("float32")
                K.ptx.max.f32(maximum, rm0, rm1)
                safe_max = K.local_scalar("float32")
                K.assign(safe_max, K.if_then_else(maximum != K.float32(NEG_INF), maximum, 0.0))
                scale0 = K.local_scalar("float32")
                scale1 = K.local_scalar("float32")
                K.assign(
                    scale0, K.if_then_else(valid0, exp2((rm0 - safe_max) * scale_log2), 0.0)
                )
                K.assign(
                    scale1, K.if_then_else(valid1, exp2((rm1 - safe_max) * scale_log2), 0.0)
                )
                total_sum = K.local_scalar("float32", init=sum0 * scale0 + sum1 * scale1)
                valid_total = (total_sum != K.float32(0.0)) & (total_sum == total_sum)
                inv_sum = rcp(K.if_then_else(valid_total, total_sum, 1.0))
                final_scale0 = K.local_scalar("float32", init=scale0 * inv_sum)
                final_scale1 = K.local_scalar("float32", init=scale1 * inv_sum)
                tok = iket_range("corr-wait-oacc")
                oacc_full.wait(0, oacc_phase)
                oacc_full.wait(1, oacc_phase)
                oepi_empty.wait(0, oepi_phase)
                iket_end(tok)
                tok = iket_range("corr-store")
                store_combined(final_scale0, final_scale1)
                iket_end(tok)
                K.ptx.fence.proxy.async_.shared__cta()
                K.ptx.tcgen05.wait__ld.sync.aligned()
                if EARLY_TMEM_RELEASE:
                    with K.If(task + cta_stride >= task_count), K.Then():
                        K.ptx.bar.arrive(K.uint32(2), K.uint32(416))
                spo_empty.arrive(0)
                spo_empty.arrive(1)
                oepi_full.arrive(0)
                K.assign(oacc_phase, oacc_phase ^ 1)
                K.assign(oepi_phase, oepi_phase ^ 1)
                                                                             
                                                                   
                K.assign(pv_corr_phase0, pv_corr_phase0 ^ 1)
                K.assign(pv_corr_phase1, pv_corr_phase1 ^ 1)
                K.assign(task, task + cta_stride)
            oepi_empty.wait(0, oepi_phase)
            if EARLY_TMEM_RELEASE:
                with K.If(cta >= task_count), K.Then():
                    K.ptx.bar.arrive(K.uint32(2), K.uint32(416))
            else:
                K.ptx.bar.arrive(K.uint32(2), K.uint32(416))

    if HALF:

        def vsa_attn_blk64(
            q_map: K.TensorMap,
            k_map: K.TensorMap,
            v_map: K.TensorMap,
            o_map: K.TensorMap,
            blocks: K.gptr[K.i32],
            counts: K.gptr[K.i32],
            num_tasks: K.i32,
            tiles_per_head: K.i32,
            topk: K.i32,
            entry_stride: K.i32,
            num_ctas: K.i32,
            scale_log2: K.f32,
        ):
            body(
                q_map,
                k_map,
                v_map,
                o_map,
                blocks,
                counts,
                num_tasks,
                tiles_per_head,
                topk,
                entry_stride,
                num_ctas,
                scale_log2,
            )

        fn = vsa_attn_blk64
    else:

        def vsa_attn_blk128(
            q_map: K.TensorMap,
            k_map: K.TensorMap,
            v_map: K.TensorMap,
            o_map: K.TensorMap,
            blocks: K.gptr[K.i32],
            num_tasks: K.i32,
            tiles_per_head: K.i32,
            topk: K.i32,
            num_ctas: K.i32,
            scale_log2: K.f32,
        ):
            body(
                q_map,
                k_map,
                v_map,
                o_map,
                blocks,
                None,
                num_tasks,
                tiles_per_head,
                topk,
                None,
                num_ctas,
                scale_log2,
            )

        fn = vsa_attn_blk128
    grid = "num_ctas" if static_grid is None else int(static_grid)
    return K.kernel(warps=16, arch=arch, min_blocks_per_sm=1, grid=grid)(fn)




                                                                             
                                                                                
                                                                             

WS_GROUP = 4                               
WS_STAGE_BYTES = WS_GROUP * 64 * D * 2                                     
WS_STAGE16 = WS_STAGE_BYTES // 16
                                                                                      
                                                                                          
                                                                                           
                                                                           
WS_PIECES = 6
WS_PIECE_BYTES = WS_STAGE_BYTES // 2
WS_PIECE16 = WS_PIECE_BYTES // 16
WS_Q_BYTES = 64 * D * 2
WS_BLOCK16 = 64 * D * 2 // 16                                                 
QK_WS_IDESC = 0x04400490                                             
PV_WS_IDESC = 0x04410490                                            
MMA_WS_F16 = "tcgen05.mma.ws.cta_group::1.kind::f16"
K_SLOT_OF_BLOCK = (0, 2, 1, 3)                                                                


def make_attention_kernel_blk64(
    arch,
    static_grid=None,
    cross_alias=True,
    full_blocks=False,
    complete_groups=False,
    static_shape=None,
):
    """One 64-row query block per task; each stage covers four 64-key KV blocks.

    Tensor-memory layout E (M=64 .ws): lanes 0-63 hold rows 0-63 of the first
    N half, lanes 64-127 the second N half.  K blocks are placed in slots
    (0,2,1,3) so that lane half ``h`` sees list blocks ``4g+h`` (columns 0-63)
    and ``4g+h+2`` (columns 64-127); V blocks stay in list order because the
    MN-major descriptor's leading offset (one 8 KB band) walks block pairs the
    same way.  Every 64-lane half runs an independent online softmax; the two
    halves' partial outputs are combined at the end of the task through a
    shared-memory exchange between partner correction warps ``c`` and ``c^2``.

    ``full_blocks`` specializes away kv_block_lens traffic when setup receives
    ``kv_block_lens=None``.  ``complete_groups`` additionally proves that topk
    is a multiple of four, eliminating the padded-tail column mask entirely.
    """
    if complete_groups and not full_blocks:
        raise ValueError("complete_groups requires full_blocks")
    grid = "num_ctas" if static_grid is None else int(static_grid)

    def body(q_map, k_map, v_map, out, q2k, lens, num_tasks, num_blocks, num_heads, topk, num_ctas, scale_log2):
        if static_shape is None:
            task_count = num_tasks
            block_count = num_blocks
            head_count = num_heads
            list_topk = topk
            cta_stride = num_ctas
        else:
            task_count, block_count, head_count, list_topk, cta_stride = (
                int(value) for value in static_shape
            )
        cta = K.cta_id()
        warp = K.warp_id()
        tid = K.thread_id()
        lane = tid & 31
        n_groups = (list_topk + 3) >> 2
        n_stream0 = (n_groups + 1) >> 1
        n_stream1 = n_groups >> 1

        with K.If(warp == 0), K.Then():
            K.ptx.prefetch.tensormap(K.address_of(q_map))
            K.ptx.prefetch.tensormap(K.address_of(k_map))
            K.ptx.prefetch.tensormap(K.address_of(v_map))
        if PREFETCH_IDS:
                                                                                       
                                                                          
            with K.If((warp == 14) & (cta < task_count)), K.Then():
                first_line = K.local_scalar("int32", init=lane)
                with K.While(first_line * 32 < list_topk + 32):
                    K.ptx.prefetch.global_.L2(q2k.ptr_to([cta * list_topk + first_line * 32]))
                    K.assign(first_line, first_line + 32)

        smem = K.smem_pool()
        q_smem = smem.alloc((64, D), K.bf16, swizzle=K.SW128B)
        kv_base = smem.pool.offset
        k_smem = smem.alloc((WS_PIECES // 2, WS_GROUP * 64, D), K.bf16, swizzle=K.SW128B)
        smem.pool.move_base_to(kv_base)
        v_smem = smem.alloc((WS_PIECES * 2, 64, D), K.bf16, swizzle=K.SW128B)
        smem.pool.move_base_to(kv_base + WS_PIECES * WS_PIECE_BYTES)
        xchg = smem.alloc((4 * 32 * 32,), K.f32, align=16)                                
        stats_smem = smem.alloc((512,), K.f32, align=16)
        tmem_mailbox = smem.alloc((1,), K.u32, align=8)
        pool = smem.pool
                                                                                             
                                                                         
        pre_leader = tid == 14 * 32
        pre_leader_u = K.local_scalar("uint32", init=K.if_then_else(tid == 14 * 32, K.uint32(1), K.uint32(0)))
        q_full = K.TMABar(pool, 1, leader=pre_leader)
        q_empty = K.TCGen05Bar(pool, 1, leader=pre_leader)
        kv_full = K.TMABar(pool, WS_PIECES, leader=pre_leader)
        kv_empty = K.TCGen05Bar(pool, WS_PIECES, leader=pre_leader)
        spo_full = K.TCGen05Bar(pool, 2, leader=pre_leader)
        spo_empty = K.MBarrier(pool, 2, leader=pre_leader)
        plast_full = K.MBarrier(pool, 2, leader=pre_leader)
        if cross_alias:
            s_consumed = K.MBarrier(pool, 4, leader=pre_leader)
            pv_done = K.TCGen05Bar(pool, 2, leader=pre_leader)
        oacc_full = K.TCGen05Bar(pool, 2, leader=pre_leader)
        stats_empty = K.MBarrier(pool, 2, leader=pre_leader)
        q_full.init(1)
        q_empty.init(1)
        kv_full.init(1)
        kv_empty.init(1)
        spo_full.init(1)
        spo_empty.init(256)
        plast_full.init(4)
        if cross_alias:
            s_consumed.init(128)
            pv_done.init(1)
        oacc_full.init(1)
        stats_empty.init(128)
        K.ptx.fence.proxy.async_.shared__cta()
        K.ptx.fence.mbarrier_init.release.cluster()
        if EARLY_ISSUE:
            sids0 = [K.local_scalar("int32", init=0) for _ in range(WS_GROUP)]
            with K.If(warp == 14), K.Then():
                K.cuda.warp_sync()
                with K.If(cta < task_count), K.Then():
                    head_pre = cta // block_count
                    qb_pre = cta - head_pre * block_count
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        q_full.ptr_to([0]), K.uint32(WS_Q_BYTES), pred=pre_leader_u
                    )
                    K.ptx[TMA_G2S](
                        q_smem.ptr_to(0, 0),
                        K.address_of(q_map),
                        K.int32(0),
                        qb_pre * 64,
                        head_pre * 2,
                        K.cuda.cvta_generic_to_shared(q_full.ptr_to([0])),
                        K.uint64(POLICY_EVICT_FIRST),
                        pred=pre_leader_u,
                    )
                    for b in range(WS_GROUP):
                        K.ptx.ld.global_.s32(
                            sids0[b], q2k.ptr_to([cta * list_topk + K.min(b, list_topk - 1)])
                        )
        K.cuda.cta_sync()

        roles = K.specialize(chain_dispatch=True)
        r_softmax = roles.role("softmax", warps=range(0, 8), regs=REGS_SOFTMAX)
        r_correction = roles.role("correction", warps=range(8, 12), regs=80)
        r_mma = roles.role("mma", warps=[12], regs=64)
        r_idle0 = roles.role("idle0", warps=[13], regs=64)
        r_load = roles.role("load", warps=[14], regs=64)
        r_idle1 = roles.role("idle1", warps=[15], regs=64)

        def ld_i32(buffer, index, pred=None):
            value = K.local_scalar("int32", init=0)
            if pred is None:
                K.ptx.ld.global_.s32(value, buffer.ptr_to([index]))
            else:
                K.ptx.ld.global_.s32(value, buffer.ptr_to([index]), pred=pred)
            return value

        def stats_arrive(stage, warp_in_group):
            K.ptx.bar.arrive(K.cast(3 + stage * 4 + warp_in_group, "uint32"), K.uint32(64))

        def stats_sync(stage, warp_in_group):
            K.ptx.bar.sync(K.cast(3 + stage * 4 + warp_in_group, "uint32"), K.uint32(64))

        def ld_stats(index):
            value = K.local_scalar("float32")
            K.ptx.ld.shared.f32(value, stats_smem.ptr_to([index]))
            return value

        def st_stats(index, value):
            K.ptx.st.shared.f32(stats_smem.ptr_to([index]), value)

        def task_coords(task):
            head = K.local_scalar("int32", init=task // block_count)
            qblock = K.local_scalar("int32", init=task - head * block_count)
            return head, qblock

        with r_idle0:
            pass

        with r_idle1:
            pass

                                                                                
        with r_load:
            leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
            q_phase = K.local_scalar("int32", init=1)
            piece = K.local_scalar("int32", init=0)
            piece_phase = K.local_scalar("int32", init=1)
            task = K.local_scalar("int32", init=cta)

            def advance_pieces():
                                                                                   
                K.assign(piece, piece + 2)
                with K.If(piece == WS_PIECES), K.Then():
                    K.assign(piece, 0)
                    K.assign(piece_phase, piece_phase ^ 1)

            def wait_pair_empty():
                tok = iket_range("load-wait-empty")
                kv_empty.wait(piece, piece_phase)
                kv_empty.wait(piece + 1, piece_phase)
                iket_end(tok)

            def expect_piece(idx):
                K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                    kv_full.ptr_to([idx]), K.uint32(WS_PIECE_BYTES), pred=leader
                )
                return K.cuda.cvta_generic_to_shared(kv_full.ptr_to([idx]))

            with K.While(task < task_count):
                head, qblock = task_coords(task)
                head2 = head * 2
                list_base = K.local_scalar("int32", init=task * list_topk)

                def block_id(i):
                    return ld_i32(q2k, list_base + K.min(i, list_topk - 1), pred=leader)

                def load_k_group(g, sids=None):
                    if sids is None:
                        sids = [block_id(g * WS_GROUP + b) for b in range(WS_GROUP)]
                    stage_view = k_smem[K.shift_right(piece, 1)]
                    for band in range(2):
                        tok = iket_range("load-wait-empty")
                        kv_empty.wait(piece + band, piece_phase)
                        iket_end(tok)
                        bar = expect_piece(piece + band)
                        for b in range(WS_GROUP):
                            slot = K_SLOT_OF_BLOCK[b]
                            K.ptx[TMA_G2S](
                                stage_view.ptr_to(slot * 64, band * 64),
                                K.address_of(k_map),
                                K.int32(0),
                                sids[b] * 64,
                                head2 + band,
                                bar,
                                K.uint64(POLICY_EVICT_LAST),
                                pred=leader,
                            )
                    advance_pieces()

                def load_v_group(g):
                    sids = [block_id(g * WS_GROUP + b) for b in range(WS_GROUP)]
                    for half in range(2):
                        tok = iket_range("load-wait-empty")
                        kv_empty.wait(piece + half, piece_phase)
                        iket_end(tok)
                        bar = expect_piece(piece + half)
                        for b in (2 * half, 2 * half + 1):
                            K.ptx[TMA_G2S](
                                v_smem[piece * 2 + b].ptr_to(0, 0),
                                K.address_of(v_map),
                                K.int32(0),
                                sids[b] * 64,
                                head2,
                                bar,
                                K.uint64(POLICY_EVICT_LAST),
                                pred=leader,
                            )
                    advance_pieces()

                def issue_q():
                    q_empty.wait(0, q_phase)
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        q_full.ptr_to([0]), K.uint32(WS_Q_BYTES), pred=leader
                    )
                    K.ptx[TMA_G2S](
                        q_smem.ptr_to(0, 0),
                        K.address_of(q_map),
                        K.int32(0),
                        qblock * 64,
                        head2,
                        K.cuda.cvta_generic_to_shared(q_full.ptr_to([0])),
                        K.uint64(POLICY_EVICT_FIRST),
                        pred=leader,
                    )

                if EARLY_ISSUE:
                    with K.If(task != cta), K.Then():
                        issue_q()
                    K.assign(q_phase, q_phase ^ 1)
                    with K.If(task == cta):
                        with K.Then():
                            load_k_group(K.int32(0), sids0)
                        with K.Else():
                            load_k_group(K.int32(0))
                else:
                    issue_q()
                    K.assign(q_phase, q_phase ^ 1)
                    load_k_group(K.int32(0))
                                                                                            
                next_task = K.local_scalar("int32", init=task + cta_stride)
                with K.If((next_task < task_count) & (leader != K.uint32(0))), K.Then():
                    next_head, next_qblock = task_coords(next_task)
                    K.ptx[TMA_PREFETCH_L2](
                        K.address_of(q_map),
                        K.int32(0),
                        next_qblock * 64,
                        next_head * 2,
                    )
                    if PREFETCH_IDS:
                                                                                       
                                                                          
                        line = K.local_scalar("int32", init=0)
                        with K.While(line * 32 < list_topk + 32):
                            K.ptx.prefetch.global_.L2(q2k.ptr_to([next_task * list_topk + line * 32]))
                            K.assign(line, line + 1)
                with K.If(n_groups >= 2), K.Then():
                    load_k_group(K.int32(1))
                g = K.local_scalar("int32", init=2)
                with K.While(g < n_groups):
                    if cross_alias:
                                                                           
                                                                             
                        load_k_group(g)
                        load_v_group(g - 2)
                    else:
                        load_v_group(g - 2)
                        load_k_group(g)
                    K.assign(g, g + 1)
                with K.If(n_groups >= 2), K.Then():
                    load_v_group(n_groups - 2)
                load_v_group(n_groups - 1)
                K.assign(task, task + cta_stride)

                                                                                
        with r_mma:
            K.ptx[TMEM_ALLOC](K.address_of(tmem_mailbox[0]), K.uint32(TMEM_COLS))
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            tmem_base = K.local_scalar("uint32", init=K.uint32(0))
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
            q_desc, qoff = encode(q_smem, "k")
            k_desc, koff = encode(k_smem[0], "k")
            v_desc, voff = encode(v_smem[0], "mn")
            q_phase = K.local_scalar("int32", init=0)
            piece = K.local_scalar("int32", init=0)
            piece_phase = K.local_scalar("int32", init=0)
            spo_phase = [K.local_scalar("int32", init=0), K.local_scalar("int32", init=0)]
            if cross_alias:
                                                                                
                                                                              
                                                                             
                cbase = [K.local_scalar("int32", init=0) for _ in range(4)]
                qk_gen = [K.local_scalar("int32", init=0), K.local_scalar("int32", init=0)]
            acc = [K.local_scalar("int32", init=0), K.local_scalar("int32", init=0)]
            task = K.local_scalar("int32", init=cta)

            def piece_at(k):
                """(index, phase) of the piece k positions ahead of the current one (k < 4)."""
                ahead = piece + k
                wrapped = ahead >= WS_PIECES
                idx = K.local_scalar("int32", init=K.if_then_else(wrapped, ahead - WS_PIECES, ahead))
                ph = K.local_scalar("int32", init=K.if_then_else(wrapped, piece_phase ^ 1, piece_phase))
                return idx, ph

            def advance_pieces(n):
                K.assign(piece, piece + n)
                with K.If(piece >= WS_PIECES), K.Then():
                    K.assign(piece, piece - WS_PIECES)
                    K.assign(piece_phase, piece_phase ^ 1)

            def commit(bar, stage):
                K.ptx[TCGEN_COMMIT](bar.ptr_to([stage]), pred=leader)

            def v_off(k16):
                                                                                           
                return voff(k16 % 4) + (k16 // 4) * 2 * WS_BLOCK16

            def issue_qk_half(s, k_piece, lo, hi):
                                                                                     
                                                                                    
                                              
                with K.If(leader != K.uint32(0)), K.Then():
                    for k16 in range(lo, hi):
                        K.ptx[MMA_WS_F16](
                            K.cast(tmem_base + s * 128, "uint32"),
                            desc_at(q_desc, qoff(k16)),
                            desc_at(k_desc, k_piece * WS_PIECE16 + koff(k16)),
                            K.uint32(QK_WS_IDESC),
                            K.cast(k16 != 0, "bool"),
                            K.uint64(0),
                        )

            def issue_pv_part(s, v_piece, lo, hi, accumulate):
                with K.If(leader != K.uint32(0)), K.Then():
                    for k16 in range(lo, hi):
                        p_addr = (
                            tmem_base + 64 + (1 - s) * 128 + k16 * 8
                            if cross_alias
                            else tmem_base + s * 128 + k16 * 8
                        )
                        K.ptx[MMA_WS_F16](
                            K.cast(tmem_base + 256 + s * 128, "uint32"),
                            K.cast(p_addr, "uint32"),
                            desc_at(v_desc, v_piece * WS_PIECE16 + v_off(k16)),
                            K.uint32(PV_WS_IDESC),
                            K.cast(accumulate != 0 if k16 == 0 else True, "bool"),
                            K.uint64(0),
                        )

            def issue_qk(s, k0, k0p, k1, k1p):
                """QK over the K group in pieces k0 (band 0) and k1 (band 1).  Each band is
                waited for right before its four k-steps, so band 0's MMAs run while band 1
                is still landing, and each piece is released as soon as its k-steps issue."""
                tok = iket_range("mma-wait-kv")
                kv_full.wait(k0, k0p)
                iket_end(tok)
                issue_qk_half(s, k0, 0, 4)
                commit(kv_empty, k0)
                tok = iket_range("mma-wait-kv")
                kv_full.wait(k1, k1p)
                iket_end(tok)
                issue_qk_half(s, k0, 4, 8)
                commit(spo_full, s)
                commit(kv_empty, k1)

            def issue_pv(s, v0, v0p, v1, v1p, phase):
                """PV over the V group in pieces v0 (blocks 0/1) and v1 (blocks 2/3)."""
                issue_pv_part(s, v0, 0, 4, acc[s])
                commit(kv_empty, v0)
                tok = iket_range("mma-wait-kv")
                kv_full.wait(v1, v1p)
                iket_end(tok)
                issue_pv_part(s, v0, 4, 6, acc[s])
                plast_full.wait(s, phase)
                issue_pv_part(s, v0, 6, 8, acc[s])

            def prologue_qk(s):
                k0, k0p = piece_at(0)
                k1, k1p = piece_at(1)
                issue_qk(s, k0, k0p, k1, k1p)
                advance_pieces(2)

            if cross_alias:
                def step(s):
                    """QK for stream s's next stage, then its pending PV."""
                    k0, k0p = piece_at(0)
                    k1, k1p = piece_at(1)
                    v0, v0p = piece_at(2)
                    v1, v1p = piece_at(3)
                    k = qk_gen[s]
                    s_consumed.wait(
                        s * 2 + (k & 1),
                        (
                            K.if_then_else((k & 1) == 0, cbase[s * 2], cbase[s * 2 + 1])
                            + (k >> 1)
                        )
                        & 1,
                    )
                    tok = iket_range("mma-issue-qk")
                    issue_qk(s, k0, k0p, k1, k1p)
                    iket_end(tok)
                    K.assign(qk_gen[s], qk_gen[s] + 1)
                    tok = iket_range("mma-wait-kv")
                    kv_full.wait(v0, v0p)
                    iket_end(tok)
                    tok = iket_range("mma-wait-p")
                    spo_empty.wait(s, spo_phase[s])
                    iket_end(tok)
                    tok = iket_range("mma-issue-pv")
                    issue_pv(s, v0, v0p, v1, v1p, spo_phase[s])
                    commit(pv_done, s)
                    commit(kv_empty, v1)
                    iket_end(tok)
                    K.assign(spo_phase[s], spo_phase[s] ^ 1)
                    K.assign(acc[s], 1)
                    advance_pieces(4)
            else:
                def step(s):
                    """PV for stream s's pending stage, then QK for its next stage."""
                    v0, v0p = piece_at(0)
                    v1, v1p = piece_at(1)
                    k0, k0p = piece_at(2)
                    k1, k1p = piece_at(3)
                    tok = iket_range("mma-wait-kv")
                    kv_full.wait(v0, v0p)
                    iket_end(tok)
                    tok = iket_range("mma-wait-p")
                    spo_empty.wait(s, spo_phase[s])
                    iket_end(tok)
                    tok = iket_range("mma-issue")
                    issue_pv(s, v0, v0p, v1, v1p, spo_phase[s])
                    commit(kv_empty, v1)
                    iket_end(tok)
                    tok = iket_range("mma-issue")
                    issue_qk(s, k0, k0p, k1, k1p)
                    iket_end(tok)
                    K.assign(spo_phase[s], spo_phase[s] ^ 1)
                    K.assign(acc[s], 1)
                    advance_pieces(4)

            def tail(s):
                v0, v0p = piece_at(0)
                v1, v1p = piece_at(1)
                tok = iket_range("mma-wait-kv")
                kv_full.wait(v0, v0p)
                iket_end(tok)
                tok = iket_range("mma-wait-p")
                spo_empty.wait(s, spo_phase[s])
                iket_end(tok)
                issue_pv(s, v0, v0p, v1, v1p, spo_phase[s])
                if cross_alias:
                    commit(pv_done, s)
                commit(oacc_full, s)
                commit(kv_empty, v1)
                advance_pieces(2)
                K.assign(spo_phase[s], spo_phase[s] ^ 1)

            with K.While(task < task_count):
                K.assign(acc[0], 0)
                K.assign(acc[1], 0)
                if cross_alias:
                    K.assign(qk_gen[0], 0)
                    K.assign(qk_gen[1], 0)
                tok = iket_range("mma-wait-q")
                q_full.wait(0, q_phase)
                iket_end(tok)
                K.assign(q_phase, q_phase ^ 1)
                prologue_qk(0)
                with K.If(n_groups >= 2), K.Then():
                    prologue_qk(1)
                pairs = (n_groups - 2) >> 1
                pair = K.local_scalar("int32", init=0)
                with K.While(pair < pairs):
                    step(0)
                    step(1)
                    K.assign(pair, pair + 1)
                                                                         
                with K.If((n_groups >= 3) & (((n_groups - 2) & 1) != 0)), K.Then():
                    step(0)
                commit(q_empty, 0)
                with K.If(n_groups >= 2):
                    with K.Then():
                                                                                                  
                        with K.If((n_groups & 1) == 0):
                            with K.Then():
                                tail(0)
                                tail(1)
                            with K.Else():
                                tail(1)
                                tail(0)
                    with K.Else():
                        tail(0)
                if cross_alias:
                                                                              
                                                                                
                    for s_, n_s_ in ((0, n_stream0), (1, n_stream1)):
                        K.assign(cbase[s_ * 2], cbase[s_ * 2] + ((n_s_ + 1) >> 1))
                        K.assign(cbase[s_ * 2 + 1], cbase[s_ * 2 + 1] + (n_s_ >> 1))
                K.assign(task, task + cta_stride)

            K.ptx[TMEM_RELINQUISH]()
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            allocated = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(allocated, tmem_mailbox.ptr_to([0]))
            K.ptx[TMEM_DEALLOC](allocated, K.uint32(TMEM_COLS))

                                                                                 
        with r_softmax:
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            tmem_base = K.local_scalar("uint32", init=K.uint32(0))
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            s = K.if_then_else(warp < 4, 0, 1)
            local_warp = warp & 3
            half = local_warp >> 1
            tid128 = tid & 127
            row_hi = K.shift_left(K.cast(local_warp * 32, "uint32"), K.uint32(16))
            n_s = K.if_then_else(s == 0, n_stream0, n_stream1)
            score_phase = K.local_scalar("int32", init=0)
            if cross_alias:
                                                                                
                                                                                   
                pbase = [K.local_scalar("int32", init=0) for _ in range(2)]
                stats_phase = K.local_scalar("int32", init=1)
                                                          
                pv_phase = K.local_scalar("int32", init=1)
            else:
                stats_phase = K.local_scalar("int32", init=1)
            row_max = K.local_scalar("float32", init=K.float32(NEG_INF))
            row_sum = K.local_scalar("float32", init=K.float32(0.0))
            task = K.local_scalar("int32", init=cta)
                                                                                      
                                                                                      
                                                                                    
            if not full_blocks:
                cur_sid = [K.local_scalar("int32", init=0) for _ in range(2)]
                cur_len = [K.local_scalar("int32", init=64) for _ in range(2)]

                def prefetch_sid(g_next, list_base_next):
                    for c in range(2):
                        b = g_next * WS_GROUP + half + 2 * c
                        K.assign(
                            cur_sid[c],
                            ld_i32(q2k, list_base_next + K.min(b, list_topk - 1)),
                        )

                def prefetch_len():
                    for c in range(2):
                        K.assign(cur_len[c], ld_i32(lens, cur_sid[c]))

            def consume(g, first, list_base):
                tok = iket_range("sm-wait-s")
                spo_full.wait(s, score_phase)
                iket_end(tok)
                tok_c = iket_range("sm-compute")
                tok_a = iket_range("sm-ldmax")
                score = K.alloc_local((128,), "float32")
                for chunk in range(4):
                    tmem_load32_at(score, chunk * 32, tmem_base + s * 128 + chunk * 32 + row_hi)
                if not complete_groups:
                    lens_local = []
                    for c in range(2):
                        b = g * WS_GROUP + half + 2 * c
                        if full_blocks:
                            length = K.if_then_else(b < list_topk, 64, 0)
                        else:
                            length = K.if_then_else(b < list_topk, cur_len[c], 0)
                        lens_local.append(K.local_scalar("int32", init=length))
                if not full_blocks:
                    prefetch_sid(g + 2, list_base)
                if not complete_groups:
                    with K.If((lens_local[0] < 64) | (lens_local[1] < 64)), K.Then():
                        apply_column_mask(score, lens_local[0], lens_local[1])

                old_scale = K.local_scalar("float32", init=K.float32(0.0))
                new_max = K.local_scalar("float32")
                max_safe = K.local_scalar("float32")
                if first:
                    tile_max = reduce_max_128(score)
                    K.assign(new_max, tile_max)
                    K.assign(
                        max_safe, K.if_then_else(tile_max != K.float32(NEG_INF), tile_max, 0.0)
                    )
                else:
                    tile_max = reduce_max_128(score, row_max)
                    K.assign(new_max, tile_max)
                    K.assign(
                        max_safe, K.if_then_else(new_max != K.float32(NEG_INF), new_max, 0.0)
                    )
                    delta = K.local_scalar("float32")
                    K.ptx.sub.f32(delta, row_max, max_safe)
                    delta_scaled = K.local_scalar("float32")
                    K.ptx.mul.f32(delta_scaled, delta, scale_log2)
                    K.assign(old_scale, exp2(delta_scaled))
                    with K.If(delta_scaled >= K.float32(-8.0)), K.Then():
                        K.assign(new_max, row_max)
                        K.assign(max_safe, row_max)
                        K.assign(old_scale, K.float32(1.0))
                if cross_alias:
                                                                             
                                                                            
                    K.ptx.tcgen05.wait__ld.sync.aligned()
                    s_consumed.arrive(s * 2 + ((g >> 1) & 1))
                iket_end(tok_a)
                tok = iket_range("sm-wait-stats")
                stats_empty.wait(s, stats_phase)
                iket_end(tok)
                K.assign(stats_phase, stats_phase ^ 1)
                if not first:
                    st_stats(s * 128 + tid128, old_scale)
                stats_arrive(s, local_warp)
                if not full_blocks:
                    prefetch_len()
                tok_b = iket_range("sm-exp-pstore")

                negative_max = K.local_scalar("float32")
                K.ptx.mul.f32(negative_max, max_safe, -scale_log2)
                for pair in range(64):
                    base_idx = pair * 2
                    packed(
                        "fma.rn.f32x2",
                        score,
                        base_idx,
                        score[base_idx],
                        score[base_idx + 1],
                        scale_log2,
                        scale_log2,
                        negative_max,
                        negative_max,
                    )
                if cross_alias:
                                                                           
                                                                     
                    for fragment in range(4):
                        for pair in range(16):
                            base_idx = fragment * 32 + pair * 2
                            if emulate_pair(fragment, pair):
                                ex2_emulation_2(score, base_idx)
                            else:
                                K.assign(score[base_idx], exp2(score[base_idx]))
                                K.assign(score[base_idx + 1], exp2(score[base_idx + 1]))
                    for fragment in range(4):
                        packed_p = K.alloc_local((16,), "uint32")
                        for pair in range(16):
                            base_idx = fragment * 32 + pair * 2
                            K.ptx.cvt.rn.bf16x2.f32(
                                packed_p[pair], score[base_idx + 1], score[base_idx]
                            )
                        if fragment == 0:
                            pv_done.wait(s, pv_phase)
                            K.assign(pv_phase, pv_phase ^ 1)
                                                                               
                                                                            
                            with K.If((s == 1) | (g + 1 < n_groups)), K.Then():
                                peer_gen = K.local_scalar(
                                    "int32",
                                    init=(g >> 1)
                                    + K.if_then_else(
                                        (s == 1) & (g + 1 < n_groups), 1, 0
                                    ),
                                )
                                peer_par = peer_gen & 1
                                s_consumed.wait(
                                    (1 - s) * 2 + peer_par,
                                    (
                                        K.if_then_else(peer_par == 0, pbase[0], pbase[1])
                                        + (peer_gen >> 1)
                                    )
                                    & 1,
                                )
                        tmem_store16(
                            packed_p,
                            tmem_base + 64 + (1 - s) * 128 + fragment * 16 + row_hi,
                        )
                        if fragment == 2:
                            K.ptx.tcgen05.wait__st.sync.aligned()
                            spo_empty.arrive(s)
                else:
                    for fragment in range(4):
                        for pair in range(16):
                            base_idx = fragment * 32 + pair * 2
                            if emulate_pair(fragment, pair):
                                ex2_emulation_2(score, base_idx)
                            else:
                                K.assign(score[base_idx], exp2(score[base_idx]))
                                K.assign(score[base_idx + 1], exp2(score[base_idx + 1]))
                        packed_p = K.alloc_local((16,), "uint32")
                        for pair in range(16):
                            base_idx = fragment * 32 + pair * 2
                            K.ptx.cvt.rn.bf16x2.f32(
                                packed_p[pair], score[base_idx + 1], score[base_idx]
                            )
                        tmem_store16(
                            packed_p, tmem_base + s * 128 + fragment * 16 + row_hi
                        )
                        if fragment == 2:
                            K.ptx.tcgen05.wait__st.sync.aligned()
                            spo_empty.arrive(s)
                K.ptx.tcgen05.wait__st.sync.aligned()
                iket_end(tok_b)
                K.cuda.warp_sync()
                with K.If(K.cuda.elect_sync()), K.Then():
                    plast_full.arrive(s)
                iket_end(tok_c)
                tok_s = iket_range("sm-sum")
                K.assign(row_sum, packed_sum_128(score, row_sum, old_scale, first))
                iket_end(tok_s)
                K.assign(row_max, new_max)
                K.assign(score_phase, score_phase ^ 1)

            if not full_blocks:
                prefetch_sid(s, K.min(task, task_count - 1) * list_topk)
                prefetch_len()
            with K.While(task < task_count):
                list_base = K.local_scalar("int32", init=task * list_topk)
                K.assign(row_max, K.float32(NEG_INF))
                K.assign(row_sum, K.float32(0.0))
                with K.If(n_s > 0), K.Then():
                    consume(K.local_scalar("int32", init=s), True, list_base)
                    k = K.local_scalar("int32", init=1)
                    with K.While(k < n_s):
                        consume(K.local_scalar("int32", init=s + 2 * k), False, list_base)
                        K.assign(k, k + 1)
                    stats_empty.wait(s, stats_phase)
                    K.assign(stats_phase, stats_phase ^ 1)
                    st_stats(s * 128 + tid128, row_sum)
                    st_stats(256 + s * 128 + tid128, row_max)
                    stats_arrive(s, local_warp)
                if cross_alias:
                    peer_count = K.if_then_else(s == 0, n_stream1, n_stream0)
                    K.assign(pbase[0], pbase[0] + ((peer_count + 1) >> 1))
                    K.assign(pbase[1], pbase[1] + (peer_count >> 1))
                                                                                          
                if not full_blocks:
                    prefetch_sid(
                        s,
                        K.min(task + cta_stride, task_count - 1) * list_topk,
                    )
                    prefetch_len()
                K.assign(task, task + cta_stride)
            K.ptx.bar.arrive(K.uint32(2), K.uint32(416))

                                                                                 
        with r_correction:
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            tmem_base = K.local_scalar("uint32", init=K.uint32(0))
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            c = warp - 8
            tid128 = tid & 127
            row_hi = K.shift_left(K.cast(c * 32, "uint32"), K.uint32(16))
            pair_bar = K.cast(11 + (c & 1), "uint32")
            partner = c ^ 2
            oacc_phase = K.local_scalar("int32", init=0)
            if cross_alias:
                pv_corr_phase = K.alloc_local((2,), "int32")
                K.assign(pv_corr_phase[0], 0)
                K.assign(pv_corr_phase[1], 0)
            task = K.local_scalar("int32", init=cta)

            def o_free_arrive():
                K.ptx.tcgen05.wait__ld.sync.aligned()
                spo_empty.arrive(0)
                with K.If(n_stream1 > 0), K.Then():
                    spo_empty.arrive(1)

            def rescale_o(s, scale):
                for chunk in range(D // 16):
                    values = K.alloc_local((16,), "float32")
                    address = tmem_base + 256 + s * 128 + chunk * 16 + row_hi
                    tmem_load16(values, address)
                    for pair in range(8):
                        base_idx = pair * 2
                        packed(
                            "mul.rn.f32x2",
                            values,
                            base_idx,
                            values[base_idx],
                            values[base_idx + 1],
                            scale,
                            scale,
                        )
                    tmem_store16(values, address)
                K.ptx.tcgen05.wait__st.sync.aligned()

            o_free_arrive()
            with K.While(task < task_count):
                head, qblock = task_coords(task)
                g = K.local_scalar("int32", init=0)
                with K.While(g < n_groups):
                    s = g & 1
                    stats_sync(s, c)
                    with K.If(g >= 2), K.Then():
                        scale = ld_stats(s * 128 + tid128)
                        if cross_alias:
                                                                          
                                                                           
                            pv_done.wait(s, pv_corr_phase[s])
                            K.assign(pv_corr_phase[s], pv_corr_phase[s] ^ 1)
                        ballot = K.local_scalar("uint32")
                        K.ptx.vote_sync.ballot.b32(
                            ballot, K.ptx.pred(scale < K.float32(1.0)), K.uint32(0xFFFFFFFF)
                        )
                        with K.If(ballot != 0), K.Then():
                            tok = iket_range("corr-rescale")
                            rescale_o(s, scale)
                            iket_end(tok)
                        spo_empty.arrive(s)
                    stats_empty.arrive(s)
                    K.assign(g, g + 1)

                sums = [K.local_scalar("float32", init=K.float32(0.0)) for _ in range(2)]
                maxes = [K.local_scalar("float32", init=K.float32(NEG_INF)) for _ in range(2)]
                stats_sync(0, c)
                K.assign(sums[0], ld_stats(tid128))
                K.assign(maxes[0], ld_stats(256 + tid128))
                stats_empty.arrive(0)
                with K.If(n_stream1 > 0), K.Then():
                    stats_sync(1, c)
                    K.assign(sums[1], ld_stats(128 + tid128))
                    K.assign(maxes[1], ld_stats(384 + tid128))
                    stats_empty.arrive(1)
                valid = [(sums[i] != K.float32(0.0)) & (sums[i] == sums[i]) for i in range(2)]
                rm = [K.if_then_else(valid[i], maxes[i], K.float32(NEG_INF)) for i in range(2)]
                m_half = K.local_scalar("float32")
                K.ptx.max.f32(m_half, rm[0], rm[1])
                m_half_safe = K.local_scalar(
                    "float32", init=K.if_then_else(m_half != K.float32(NEG_INF), m_half, 0.0)
                )
                w = [
                    K.local_scalar(
                        "float32",
                        init=K.if_then_else(valid[i], exp2((rm[i] - m_half_safe) * scale_log2), 0.0),
                    )
                    for i in range(2)
                ]
                sum_half = K.local_scalar("float32", init=sums[0] * w[0] + sums[1] * w[1])
                                                                                          
                                                                                          
                                                                                               
                                                                                              
                                                                                               
                                                                                            
                K.ptx.st.shared.f32(xchg.ptr_to([partner * 1024 + lane * 2]), sum_half)
                K.ptx.st.shared.f32(xchg.ptr_to([partner * 1024 + lane * 2 + 1]), m_half)
                K.ptx.bar.sync(pair_bar, K.uint32(64))
                peer_sum = K.local_scalar("float32")
                peer_max = K.local_scalar("float32")
                K.ptx.ld.shared.f32(peer_sum, xchg.ptr_to([c * 1024 + lane * 2]))
                K.ptx.ld.shared.f32(peer_max, xchg.ptr_to([c * 1024 + lane * 2 + 1]))
                m_total = K.local_scalar("float32")
                K.ptx.max.f32(m_total, m_half, peer_max)
                m_total_safe = K.if_then_else(m_total != K.float32(NEG_INF), m_total, 0.0)
                own_w = K.local_scalar(
                    "float32",
                    init=K.if_then_else(
                        sum_half > K.float32(0.0), exp2((m_half - m_total_safe) * scale_log2), 0.0
                    ),
                )
                peer_w = K.local_scalar(
                    "float32",
                    init=K.if_then_else(
                        peer_sum > K.float32(0.0), exp2((peer_max - m_total_safe) * scale_log2), 0.0
                    ),
                )
                total = K.local_scalar("float32", init=sum_half * own_w + peer_sum * peer_w)
                inv_total = K.local_scalar(
                    "float32", init=K.if_then_else(total > K.float32(0.0), rcp(total), 0.0)
                )
                f = [K.local_scalar("float32", init=w[i] * own_w * inv_total) for i in range(2)]
                                                                                               
                                                                                    
                K.cuda.warp_sync()

                tok = iket_range("corr-wait-oacc")
                oacc_full.wait(0, oacc_phase)
                with K.If(n_stream1 > 0), K.Then():
                    oacc_full.wait(1, oacc_phase)
                iket_end(tok)
                K.assign(oacc_phase, oacc_phase ^ 1)
                tok = iket_range("corr-store")
                row = qblock * 64 + (c & 1) * 32 + lane
                out_row_base = (row * head_count + head) * (D // 2)
                                                                                            
                                                                                         
                own_lo = (c >> 1) * 64
                oth_lo = 64 - own_lo
                                                                                                
                                                                                                 
                my_slot = c * 1024 + lane * 4
                peer_slot = partner * 1024 + lane * 4

                def load_combine(dim0, last):
                    """acc[0:32] = O0[dim0:dim0+32]*f0 (+ O1[...]*f1).  Both streams' 32-column
                    chunks are loaded back to back so each chunk costs one exposed TMEM round
                    trip (four per task instead of eight); the stream-0 chunk is scaled in place."""
                    acc = K.alloc_local((32,), "float32")
                    tmem_load32(acc, tmem_base + 256 + dim0 + row_hi)
                    with K.If(n_stream1 > 0):
                        with K.Then():
                            values1 = K.alloc_local((32,), "float32")
                            tmem_load32(values1, tmem_base + 384 + dim0 + row_hi)
                            for pair in range(16):
                                idx = pair * 2
                                packed("mul.rn.f32x2", acc, idx, acc[idx], acc[idx + 1], f[0], f[0])
                                packed(
                                    "fma.rn.f32x2",
                                    acc,
                                    idx,
                                    values1[idx],
                                    values1[idx + 1],
                                    f[1],
                                    f[1],
                                    acc[idx],
                                    acc[idx + 1],
                                )
                        with K.Else():
                            for pair in range(16):
                                idx = pair * 2
                                packed("mul.rn.f32x2", acc, idx, acc[idx], acc[idx + 1], f[0], f[0])
                    if last:
                        o_free_arrive()
                        if EARLY_TMEM_RELEASE:
                            with K.If(task + cta_stride >= task_count), K.Then():
                                K.ptx.bar.arrive(K.uint32(2), K.uint32(416))
                    return acc

                for half in range(2):
                                                                                              
                                                                                                
                    acc_oth = load_combine(oth_lo + half * 32, False)
                    for j in range(8):
                        K.ptx.st.shared.v4.f32(
                            xchg.ptr_to([my_slot + j * 128]),
                            acc_oth[j * 4],
                            acc_oth[j * 4 + 1],
                            acc_oth[j * 4 + 2],
                            acc_oth[j * 4 + 3],
                        )
                                                                                        
                                                                                        
                                                                                         
                                                         
                    acc_own = load_combine(own_lo + half * 32, half == 1)
                    K.ptx.bar.sync(pair_bar, K.uint32(64))
                    for half16 in range(2):
                        peer = K.alloc_local((16,), "float32")
                        for j in range(4):
                            K.ptx.ld.shared.v4.f32(
                                peer[j * 4],
                                peer[j * 4 + 1],
                                peer[j * 4 + 2],
                                peer[j * 4 + 3],
                                xchg.ptr_to([peer_slot + (half16 * 4 + j) * 128]),
                            )
                        for pair in range(8):
                            idx = half16 * 16 + pair * 2
                            packed(
                                "add.rn.f32x2",
                                acc_own,
                                idx,
                                acc_own[idx],
                                acc_own[idx + 1],
                                peer[pair * 2],
                                peer[pair * 2 + 1],
                            )
                        words = K.alloc_local((8,), "uint32")
                        for pair in range(8):
                            idx = half16 * 16 + pair * 2
                            K.ptx.cvt.rn.bf16x2.f32(words[pair], acc_own[idx + 1], acc_own[idx])
                        for j in range(2):
                            K.ptx.st.global_.v4.b32(
                                out.ptr_to([out_row_base + (own_lo + half * 32 + half16 * 16) // 2 + j * 4]),
                                words[j * 4],
                                words[j * 4 + 1],
                                words[j * 4 + 2],
                                words[j * 4 + 3],
                            )
                    if half == 0:
                        K.ptx.bar.sync(pair_bar, K.uint32(64))                                 
                iket_end(tok)
                if cross_alias:
                                                                       
                                                                            
                    K.assign(pv_corr_phase[0], pv_corr_phase[0] ^ 1)
                    with K.If(n_stream1 > 0), K.Then():
                        K.assign(pv_corr_phase[1], pv_corr_phase[1] ^ 1)
                K.assign(task, task + cta_stride)
            if EARLY_TMEM_RELEASE:
                with K.If(cta >= task_count), K.Then():
                    K.ptx.bar.arrive(K.uint32(2), K.uint32(416))
            else:
                K.ptx.bar.arrive(K.uint32(2), K.uint32(416))

    def vsa_attn_blk64_ws(
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        out: K.gptr[K.i32],
        q2k: K.gptr[K.i32],
        lens: K.gptr[K.i32],
        num_tasks: K.i32,
        num_blocks: K.i32,
        num_heads: K.i32,
        topk: K.i32,
        num_ctas: K.i32,
        scale_log2: K.f32,
    ):
        body(q_map, k_map, v_map, out, q2k, lens, num_tasks, num_blocks, num_heads, topk, num_ctas, scale_log2)

    return K.kernel(warps=16, arch=arch, min_blocks_per_sm=1, grid=grid)(vsa_attn_blk64_ws)


                                                                             
           
                                                                             


class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_map(tensor, S, H, box_rows, box_bands=1):
    """Rank-3 map over bf16[S, H, 128]: dim0 = 64-col band, dim1 = token, dim2 = head*2+band."""
    desc = _AlignedTensorMap()
    dims = (64, S, H * 2)
    strides = (H * D * 2, 128)
    box = (64, box_rows, box_bands)
    tvm.get_global_func("runtime.cuTensorMapEncodeTiled")(
        desc.ptr,
        "bfloat16",
        3,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *dims,
        *strides,
        *box,
        *((1,) * 3),
        0,                   
        3,                
        2,                     
        0,                         
    )
    return desc


_COMPILED = {}

                                                                                            
                                                                                      
                                                                                   
PTXAS_REG_LEVEL = {128: "4", 64: "10"}
                                                                                             
                                                                                
EMU_BY_BLOCK = {128: "quarter", 64: "none"}
import os as _os
PREFETCH_IDS = _os.environ.get("VSA_PF_IDS", "1") != "0"
EARLY_TMEM_RELEASE = _os.environ.get("VSA_EARLY_TMEM", "1") != "0"
EARLY_ISSUE = _os.environ.get("VSA_EARLY_ISSUE", "1") != "0"


def _compile(kernel, reg_level=None):
    import os

    previous = os.environ.get("TVM_CUDA_PTXAS_REG_LEVEL")
    if reg_level is not None:
        os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = str(reg_level)
    try:
        target = tvm.target.Target({"kind": "cuda", "arch": kernel.arch})
        with target:
            return tvm.compile(kernel.mod, target=target, tir_pipeline="tirx")
    finally:
        if reg_level is not None:
            if previous is None:
                os.environ.pop("TVM_CUDA_PTXAS_REG_LEVEL", None)
            else:
                os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = previous


def _get_executable(
    block_size,
    blk64_cross=False,
    blk64_full=False,
    blk64_complete_groups=False,
    blk64_shape=None,
    blk128_complete_pairs=False,
    blk128_sid_fifo=False,
    blk128_shape=None,
):
    key = (
        (
            128,
            bool(blk128_complete_pairs),
            bool(blk128_sid_fifo),
            None if blk128_shape is None else tuple(blk128_shape),
        )
        if block_size == 128
        else (
            64,
            bool(blk64_cross),
            bool(blk64_full),
            bool(blk64_complete_groups),
            None if blk64_shape is None else tuple(blk64_shape),
        )
    )
    if key not in _COMPILED:
        arch = _arch()
        level = PTXAS_REG_LEVEL.get(block_size)
        set_emu_mode(EMU_BY_BLOCK.get(block_size, "none"))
        try:
            if block_size == 128:
                _COMPILED[key] = _compile(
                    make_attention_kernel_blk128(
                        arch,
                        static_grid=None if blk128_shape is None else blk128_shape[-1],
                        complete_pairs=blk128_complete_pairs,
                        sid_fifo=blk128_sid_fifo,
                        static_shape=blk128_shape,
                    ),
                    level,
                )
            else:
                _COMPILED[key] = _compile(
                    make_attention_kernel_blk64(
                        arch,
                        cross_alias=blk64_cross,
                        full_blocks=blk64_full,
                        complete_groups=blk64_complete_groups,
                        static_grid=None if blk64_shape is None else blk64_shape[-1],
                        static_shape=blk64_shape,
                    ),
                    level,
                )
        finally:
            set_emu_mode(EMU_MODE)
    return _COMPILED[key]




# ---------------------------------------------------------------------------
# Registry interface.
# ---------------------------------------------------------------------------

KERNEL_META = {
    "name": "agent_evolved_vsa_multishape",
    "category": "agent_evolved",
    "runtime_cuda_archs": ["sm_100a"],
    "reference_requirements": (
        {"package": "flashinfer-python", "specifier": ">=0.6.18", "import": "flashinfer"},
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.7.0", "import": "cutlass"},
    ),
    "provenance": {
        "generator": "hmz",
        "run": "vsa-20260911-201703",
        "selected_version": "frontier/cross-aliased-tmem",
    },
}


def _cfg(label, seq_len, num_heads, block_size, topk, *, shared=False, partial=(0, 0)):
    return {
        "label": label,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "block_size": block_size,
        "topk": topk,
        "head_shared_mask": shared,
        "num_partial_blocks": partial[0],
        "partial_block_len": partial[1],
        "seed": 0,
    }


# The eighteen official VSA rows: seven pooled blk128, nine head-shared BSR
# blk64, and two FastWan blk64 rows (the second carries partial blocks).
CONFIGS = [
    _cfg("pooled_blk128_s16384_topk64", 16384, 8, 128, 64),
    _cfg("pooled_blk128_s16384_topk32", 16384, 8, 128, 32),
    _cfg("pooled_blk128_s16384_topk13", 16384, 8, 128, 13),
    _cfg("pooled_blk128_s32768_topk64", 32768, 8, 128, 64),
    _cfg("pooled_blk128_s32768_topk26", 32768, 8, 128, 26),
    _cfg("pooled_blk128_s80000_topk156", 80000, 8, 128, 156),
    _cfg("pooled_blk128_s80000_topk62", 80000, 8, 128, 62),
    _cfg("bsr_blk64_s1024_topk4", 1024, 8, 64, 4, shared=True),
    _cfg("bsr_blk64_s1024_topk8", 1024, 8, 64, 8, shared=True),
    _cfg("bsr_blk64_s1024_topk12", 1024, 8, 64, 12, shared=True),
    _cfg("bsr_blk64_s2048_topk8", 2048, 8, 64, 8, shared=True),
    _cfg("bsr_blk64_s2048_topk16", 2048, 8, 64, 16, shared=True),
    _cfg("bsr_blk64_s2048_topk24", 2048, 8, 64, 24, shared=True),
    _cfg("bsr_blk64_s4096_topk16", 4096, 8, 64, 16, shared=True),
    _cfg("bsr_blk64_s4096_topk32", 4096, 8, 64, 32, shared=True),
    _cfg("bsr_blk64_s4096_topk48", 4096, 8, 64, 48, shared=True),
    _cfg("fastwan_blk64_s23296_h12_topk73", 23296, 12, 64, 73),
    _cfg("fastwan_blk64_s26624_h12_topk84_partial", 26624, 12, 64, 84, partial=(52, 32)),
]

_CONFIG_KEYS = {
    "seq_len",
    "num_heads",
    "block_size",
    "topk",
    "head_shared_mask",
    "num_partial_blocks",
    "partial_block_len",
    "seed",
}
_BY_LABEL = {config["label"]: config for config in CONFIGS}


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
    block_size = int(resolved["block_size"])
    if block_size not in (64, 128):
        raise ValueError("this kernel implements block_size 64 or 128")
    seq_len = int(resolved["seq_len"])
    if seq_len % block_size != 0:
        raise ValueError(f"seq_len must be a multiple of {block_size}")
    num_blocks = seq_len // block_size
    if not 1 <= int(resolved["topk"]) <= num_blocks:
        raise ValueError(f"topk must be in [1, {num_blocks}]")
    if int(resolved["num_partial_blocks"]) and block_size != 64:
        raise ValueError("partial blocks are defined for block_size=64 only")
    return resolved


def _assert_supported_arch() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved VSA forward")
    capability = torch.cuda.get_device_capability()
    runtime_arch = f"sm_{capability[0]}{capability[1]}a"
    if runtime_arch not in KERNEL_META["runtime_cuda_archs"]:
        raise SkipTest(
            "agent-evolved VSA forward requires one of "
            f"{KERNEL_META['runtime_cuda_archs']}, got {runtime_arch}"
        )


def get_kernel(**config: Any):
    """Return the traced Kern PrimFunc this config dispatches to.

    The runtime path compiles through `_get_executable`, which pins a
    per-block-size ptxas register level and exponent-emulation mode; this entry
    point exists for registry discovery and IR inspection.
    """
    resolved = _config(**config)
    state = _dispatch_shape(resolved, num_sms=hardware_sms())
    arch = _arch()
    if int(resolved["block_size"]) == 128:
        kernel = make_attention_kernel_blk128(
            arch,
            static_grid=None if state["blk128_shape"] is None else state["blk128_shape"][-1],
            complete_pairs=state["blk128_complete_pairs"],
            sid_fifo=state["blk128_sid_fifo"],
            static_shape=state["blk128_shape"],
        )
    else:
        kernel = make_attention_kernel_blk64(
            arch,
            cross_alias=state["blk64_cross"],
            full_blocks=state["blk64_full"],
            complete_groups=state["blk64_complete_groups"],
            static_grid=None if state["blk64_shape"] is None else state["blk64_shape"][-1],
            static_shape=state["blk64_shape"],
        )
    return kernel.func


def hardware_sms() -> int:
    from tirx_kernels.runner import hardware_num_sms

    return hardware_num_sms()


def _dispatch_shape(resolved: dict[str, Any], *, num_sms: int) -> dict[str, Any]:
    """The candidate's own shape dispatch, lifted out of its `setup`.

    Which specialization runs is a property of the shape alone, so the decision
    is shared by tracing, compilation and launch.
    """
    block_size = int(resolved["block_size"])
    topk = int(resolved["topk"])
    num_heads = int(resolved["num_heads"])
    num_blocks = int(resolved["seq_len"]) // block_size
    has_lens = bool(int(resolved["num_partial_blocks"]))

    def choose_num_ctas(task_count):
        natural = min(num_sms, task_count)
        if (
            block_size == 64
            and task_count % 128 == 0
            and task_count // 128 == (task_count + num_sms - 1) // num_sms
        ):
            return min(128, task_count)
        if task_count >= 2048 and topk >= 64:
            critical_tasks = (task_count + natural - 1) // natural
            balanced = (task_count + critical_tasks - 1) // critical_tasks
            if natural - balanced <= 5:
                return balanced
        return natural

    blk64_full = block_size == 64 and not has_lens
    state = {
        "num_blocks": num_blocks,
        "blk64_cross": block_size == 64
        and (
            ((topk + 3) // 4 >= 6)
            or (((topk + 3) // 4 >= 3) and num_heads * num_blocks <= num_sms)
        ),
        "blk64_full": blk64_full,
        "blk64_complete_groups": blk64_full and topk % 4 == 0,
        "blk128_complete_pairs": block_size == 128 and topk % 2 == 0,
        "blk128_sid_fifo": block_size == 128 and topk >= 128,
        "blk64_shape": None,
        "blk128_shape": None,
    }
    if block_size == 64 and not has_lens and topk in (4, 8, 12, 24, 32, 48):
        tasks = num_heads * num_blocks
        state["blk64_shape"] = (tasks, num_blocks, num_heads, topk, choose_num_ctas(tasks))
    if block_size == 128:
        tasks = num_heads * num_blocks
        state["blk128_shape"] = (tasks, num_blocks, topk, choose_num_ctas(tasks))
    state["num_tasks"] = num_heads * num_blocks
    state["num_ctas"] = choose_num_ctas(num_heads * num_blocks)
    return state


# ---------------------------------------------------------------------------
# Inputs.
#
# The three input families reproduce the packaged VSA benchmark rows, which
# follow flashinfer PR #4612: pooled top-k selection for the blk128 rows, a
# head-shared random BSR mask for the blk64 `-shared-bsr` rows, and FastWan's
# flattened (t, h, w) tiling for the two blk64 FastWan rows, whose second row
# gives every period-th block a short length.
# ---------------------------------------------------------------------------


def _pooled_topk_indices(q, k, block_size, topk, sm_scale):
    seq_len, num_heads, head_dim = q.shape
    query_blocks = seq_len // block_size
    key_blocks = k.shape[0] // block_size
    q_pooled = q.view(query_blocks, block_size, num_heads, head_dim).float().mean(1).permute(1, 0, 2)
    k_pooled = k.view(key_blocks, block_size, num_heads, head_dim).float().mean(1).permute(1, 0, 2)
    scores = torch.softmax(q_pooled @ k_pooled.transpose(-1, -2) * sm_scale, dim=-1)
    return torch.topk(scores, topk, dim=-1).indices.sort(dim=-1).values.to(torch.int32).contiguous()


def _shared_random_indices(num_heads, num_blocks, topk, seed):
    generator = torch.Generator().manual_seed(int(seed))
    rows = [
        torch.randperm(num_blocks, generator=generator)[:topk].sort().values
        for _ in range(num_blocks)
    ]
    selected = torch.stack(rows).to(torch.int32)
    return selected.unsqueeze(0).expand(num_heads, -1, -1).contiguous()


def prepare_data(**config: Any) -> dict[str, Any]:
    """Build one row's contract tensors plus the preallocated output."""
    resolved = _config(**config)
    device = torch.device("cuda")
    seq_len = int(resolved["seq_len"])
    num_heads = int(resolved["num_heads"])
    block_size = int(resolved["block_size"])
    topk = int(resolved["topk"])
    num_blocks = seq_len // block_size
    sm_scale = 1.0 / math.sqrt(D)
    generator = torch.Generator(device=device).manual_seed(int(resolved["seed"]))

    def randn():
        return torch.randn(
            (seq_len, num_heads, D), dtype=torch.bfloat16, device=device, generator=generator
        )

    q, k, v = randn(), randn(), randn()
    if resolved["head_shared_mask"]:
        q2k_indices = _shared_random_indices(
            num_heads, num_blocks, topk, int(resolved["seed"])
        ).to(device)
    else:
        q2k_indices = _pooled_topk_indices(q, k, block_size, topk, sm_scale)
    kv_block_lens = None
    num_partial = int(resolved["num_partial_blocks"])
    if num_partial:
        # FastWan flattens (t, h, w) tiles, so the partial last w-tile of every
        # (t, h) row recurs once per `num_blocks // num_partial` blocks.
        period = num_blocks // num_partial
        kv_block_lens = torch.full(
            (num_blocks,), block_size, dtype=torch.int32, device=device
        )
        kv_block_lens[period - 1 :: period] = int(resolved["partial_block_len"])
    return {
        "config": resolved,
        "q": q,
        "k": k,
        "v": v,
        "q2k_indices": q2k_indices,
        "kv_block_lens": kv_block_lens,
        "block_size": block_size,
        "sm_scale": sm_scale,
        "output": torch.empty_like(q),
    }


def _launch_state(case: dict[str, Any]):
    """Encode the tensor maps and bind the launch arguments for this shape."""
    resolved = case["config"]
    q, k, v, out = case["q"], case["k"], case["v"], case["output"]
    block_size = int(resolved["block_size"])
    seq_len = int(resolved["seq_len"])
    num_heads = int(resolved["num_heads"])
    topk = int(resolved["topk"])
    device = q.device
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    state = _dispatch_shape(resolved, num_sms=num_sms)
    scale_log2 = float(case["sm_scale"]) * LOG2_E
    attn = _get_executable(
        block_size,
        state["blk64_cross"],
        state["blk64_full"],
        state["blk64_complete_groups"],
        state["blk64_shape"],
        state["blk128_complete_pairs"],
        state["blk128_sid_fifo"],
        state["blk128_shape"],
    )
    q2k_flat = case["q2k_indices"].contiguous().view(-1)
    if q2k_flat.dtype != torch.int32:
        q2k_flat = q2k_flat.to(torch.int32)
    num_blocks = state["num_blocks"]
    if block_size == 128:
        maps = [_encode_map(t, seq_len, num_heads, 128) for t in (q, k, v, out)]
        args = (
            maps[0].ptr,
            maps[1].ptr,
            maps[2].ptr,
            maps[3].ptr,
            q2k_flat,
            state["num_tasks"],
            num_blocks,
            topk,
            state["num_ctas"],
            scale_log2,
        )
        keep = (*maps, q2k_flat, q, k, v, out)
    else:
        lens = case["kv_block_lens"]
        if lens is None:
            lens_buf = torch.full((num_blocks,), block_size, dtype=torch.int32, device=device)
        else:
            lens_buf = lens.contiguous()
            if lens_buf.dtype != torch.int32:
                lens_buf = lens_buf.to(torch.int32)
        q_map = _encode_map(q, seq_len, num_heads, 64, 2)
        k_map = _encode_map(k, seq_len, num_heads, 64, 1)
        v_map = _encode_map(v, seq_len, num_heads, 64, 2)
        out_words = out.view(torch.int32).view(-1)
        args = (
            q_map.ptr,
            k_map.ptr,
            v_map.ptr,
            out_words,
            q2k_flat,
            lens_buf,
            state["num_tasks"],
            num_blocks,
            num_heads,
            topk,
            state["num_ctas"],
            scale_log2,
        )
        keep = (q_map, k_map, v_map, out_words, q2k_flat, lens_buf, q, k, v, out)

    def run():
        attn(*args)

    run._keep_alive = keep
    return run


# ---------------------------------------------------------------------------
# Independent oracle.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _reference_output(case: dict[str, Any]) -> torch.Tensor:
    """Masked FP32 block-sparse attention, computed independently of the kernel.

    A query block attends exactly the KV blocks its `q2k_indices` row lists,
    and within a selected block only the first `kv_block_lens[block]` tokens
    are valid. There is no causal mask.
    """
    q, k, v = case["q"], case["k"], case["v"]
    q2k = case["q2k_indices"]
    lens = case["kv_block_lens"]
    scale = float(case["sm_scale"])
    block_size = int(case["block_size"])
    seq_len, num_heads, head_dim = q.shape
    query_blocks = seq_len // block_size
    device = q.device
    out = torch.empty_like(q)
    offsets = torch.arange(block_size, device=device)
    for head in range(num_heads):
        k_head = k[:, head].float()
        v_head = v[:, head].float()
        for block in range(query_blocks):
            rows = slice(block * block_size, (block + 1) * block_size)
            selected = q2k[head, block].long()
            token_ids = (selected.unsqueeze(-1) * block_size + offsets).reshape(-1)
            scores = (q[rows, head].float() @ k_head[token_ids].transpose(0, 1)) * scale
            if lens is not None:
                valid = offsets.view(1, -1) < lens[selected].view(-1, 1)
                scores = scores.masked_fill(~valid.reshape(1, -1), float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            out[rows, head] = (weights @ v_head[token_ids]).to(q.dtype)
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
    if rms_ratio >= 1e-2:
        raise AssertionError(f"normalized RMS error ratio {rms_ratio:.6e} must be below 1e-2")


def run_test(**config: Any) -> None:
    _assert_supported_arch()
    case = prepare_data(**config)
    run = _launch_state(case)
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
# Benchmark: the FlashInfer reference arms and the timed dispatch.
# ---------------------------------------------------------------------------


def _flashinfer_reference(case: dict[str, Any]):
    """Capture FlashInfer's block-sparse forward in a CUDA graph and replay it.

    blk128 rows use `bsa_attn_fwd`, the arm `BlockSparseAttentionWrapper`
    compiles, with the library default `allow_empty_block_nums=True`. blk64
    rows use `bsa_attn_blk64_fwd`; with counts and no block lengths it fills
    the lengths itself, which is the wrapper's path, while the FastWan partial
    row passes its lengths. Building the batched views, the block counts and
    the CuTe-DSL compile are prepare work; the capture removes the host gap
    between the blk64 extension's layout copies and its attention kernel.
    """
    from flashinfer.cute_dsl.sparse import bsa_attn_blk64_fwd, bsa_attn_fwd

    q, k, v = case["q"], case["k"], case["v"]
    lens = case["kv_block_lens"]
    block_size = int(case["block_size"])
    q2k = case["q2k_indices"].to(torch.int32).unsqueeze(0).contiguous()
    counts = torch.full(q2k.shape[:3], q2k.shape[-1], dtype=torch.int32, device=q.device)
    batched = (q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
    scale = float(case["sm_scale"])
    if block_size == 64:
        kernel = bsa_attn_blk64_fwd
        block_sparse_num = 1
        block_sizes = None if lens is None else lens.to(torch.int32).contiguous()
    else:
        kernel = bsa_attn_fwd
        block_sparse_num = 2
        block_sizes = None

    def launch():
        output, _lse = kernel(
            *batched,
            q2k,
            block_sparse_num=block_sparse_num,
            block_sizes=block_sizes,
            q2k_block_nums=counts,
            softmax_scale=scale,
            return_lse=True,
        )
        return output[0]

    launch()  # extension load / CuTe-DSL compile and workspace allocation
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        launch()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    state = (graph, q2k, counts, batched, block_sizes)

    def replay(_state=state):
        _state[0].replay()

    replay._keep_alive = state
    return replay


def prepare_bench(**config: Any):
    """Resolve the dispatch before bench-suite assigns a GPU."""
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
    run = _launch_state(case)
    run()
    torch.cuda.synchronize()

    return bench(
        {"tirx": run},
        references={"flashinfer_bsa": lambda: _flashinfer_reference(case)},
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
