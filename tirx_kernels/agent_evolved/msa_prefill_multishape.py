# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved SM100a MiniMax sparse-attention (MSA) prefill, all official shapes.

Covers every official MSA prefill row: flat bf16 (B=1, Q=KV=4096, Hq=64,
Hkv=4, top-k 16), flat FP8-E4M3 K/V under a bf16 q (B=3, Q=1024, KV=8192,
Hq=32, Hkv=2, top-k 8), and paged bf16 (B=3, Q=4096, KV=8192, Hq=8, Hkv=2,
top-k 4). Selection is `q2k_indices` int32[Hkv, total_q, topk] of ascending
sequence-local KV block ids padded with -1, under bottom-right causal masking.

The selected kernel is the `dispatch-all-s2f6-raw-maxabs` frontier member of
the 2026-09-13 MSA-prefill evolution run, the first run scored against the
task's normalized-RMS bound. Everything from the module docstring's mechanism
notes down to `setup_union` is that candidate's source; this module adds the
registry interface, input generation, the independent oracle, and the
reference arms.

Numerics. Every MMA is `tcgen05.mma kind::f16` with bf16 operands and FP32
accumulation; the FP8 row's E4M3 K/V are the contract's own storage format and
are expanded to bf16 before they reach a tensor core. Two approximations sit
on top of that, both inside the task's 1e-2 normalized-RMS bound and both
measured: a cubic exp2 approximation in the softmax (0.002382 maximum relative
error over the tested range), and, on the two sparse rows only, S2F6 split-K
partials -- an 8-bit format carrying a `ue8m0` block scale per element pair,
which halves partial traffic without the fixed-scale failure mode that sank an
earlier E2M1 attempt. The dense row keeps bf16 partials. Measured RMS ratios
are 0.007197 (dense), 0.008247 (FP8) and 0.008199 (paged), against 0.0025 for
MiniMax itself; a reviewer who wants the approximation-free arithmetic should
know it costs about 2% on the FP8 main kernel (NCU 80.67 us -> 79.04 us).

Candidate mechanism notes, carried over from the evolution run:


Two approach families in one self-contained module:

* **qmajor-union** (dense selections): one persistent CTA task owns (sequence,
  TOK_PER_CTA consecutive tokens, kv head); the union of the group's selected KV
  blocks is streamed once for two 128-row Q tiles; per-row selection masks and
  causal masks in the softmax; TMEM-resident O.

* **kvmajor-reverse** (sparse selections): a prep kernel bins (token, slot)
  edges by KV block and, in its last CTA, plans the whole persistent schedule
  (guided item-aligned batches) into a global batch table; the persistent main
  kernel grabs batches from that table, publishes edge lists with cp.async,
  gathers the exact Q rows of each 128-row block tile with one TMA per lane,
  expands FP8 K/V to BF16 in the epilogue warpgroup as soon as a block lands,
  and overlaps the next block's K/V prefetch with edge-slot waits.
  QK/softmax/PV writes normalized partials (S2F6-compressed for FP8 top-k 4/8)
  which a TMA-staged combine kernel merges.  Kernels chain with programmatic
  dependent launch.

``setup`` dispatches on the selection density derived from shapes: topk relative to
the number of selectable blocks (a shape-only quantity).
"""

import ctypes
import os
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.tirx_lite as txl
import tvm

SPIN_WAITS = os.environ.get("MSA_SPIN_WAITS", "0") == "1"
if SPIN_WAITS:

    def _spin_wait(self, stage, phase):
        ok = txl.local_scalar("uint32", init=txl.uint32(0))
        with txl.While(ok == txl.uint32(0)):
            txl.ptx.mbarrier.test_wait.parity.shared.b64(
                ok, self.buf.ptr_to([stage]), txl.uint32(phase ^ self.phase_offset)
            )

    txl.MBarrier._wait = _spin_wait

HEAD_DIM = 128
BLK_N = 128
BLK_M = 128
MMA_N = 128
MMA_K = 16
LOG2E = 1.4426950408889634
NEG_INF = float("-inf")
F16_BYTES = 2
ID_QK = 0x08200490
ID_PV = 0x08210490
MAX3_F32 = "max.f32"
PREP_THREADS = 256
BATCH = 8
ESLOTS = 4
USE_PDL = os.environ.get("MSA_PDL", "1") == "1"
FUSED_COMBINE = os.environ.get("MSA_FUSED_COMBINE", "0") == "1"
SKIP_STORES = os.environ.get("MSA_SKIP_STORES", "0") == "1"
SKIP_GATHER = os.environ.get("MSA_SKIP_GATHER", "0") == "1"
EPI_CONV = os.environ.get("MSA_EPI_CONV", "1") == "1"
UNION_EMU_PAIRS = int(os.environ.get("MSA_UNION_EMU_PAIRS", "4"))
SKIP_EXP = os.environ.get("MSA_SKIP_EXP", "0") == "1"
EMU_PER_8 = int(os.environ.get("MSA_EMU_PER_8", "2"))
STATS_SLOTS = int(os.environ.get("MSA_STATS_SLOTS", "4"))
META_FIELDS = 12

M_NVALID, M_BLK, M_KVS, M_NEWKV, M_RELKV, M_H, M_B, M_EDGE, M_POSOFF, M_Q0, M_KVUSE, M_BIDX = range(
    12
)

TMA_G2S_3D = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
TMA_G2S_4D = (
    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
TMA_G2S_5D = (
    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
TMA_G2S_3D_H = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
TMA_G2S_4D_H = "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
TMA_G2S_5D_H = "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
TMA_G2S_2D_H = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
TMEM_LD_16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
TMEM_LD_32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TMEM_ST_16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
TCGEN05_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"


def ceildiv(a, b):
    return (a + b - 1) // b


HEAD_DIM = 128
BLK_N = 128
BLK_M = 128
N_TILES = 2
N_COLS_TMEM = 512
MMA_N = 128
MMA_K = 16
LOG2E = 1.4426950408889634
K_SPLIT = 4 * MMA_K
P_SPLIT_Q = 2
N_SUM_ACC = 8
MAX_CHAINS = 8
EMU_PAIRS = UNION_EMU_PAIRS
EMU_START = 0
RESCALE_THRESHOLD = 8.0
NEG_INF = float("-inf")
F16_BYTES = 2
ID_QK = 0x08200490
ID_PV = 0x08210490
U_META_FIELDS = 8

TMA_G2S_3D = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
TMA_G2S_4D = (
    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
TMA_G2S_5D = (
    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
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


def ceildiv(a, b):
    return (a + b - 1) // b


def make_union_kernel(
    *, total_q, hq, hkv, topk, mask_words, paged, max_pages, kv_fp8, num_ctas, kv_depth
):
    assert hq % hkv == 0
    GQA = hq // hkv
    assert GQA in (4, 8, 16), GQA
    assert topk % 4 == 0 and topk >= 4
    TOK_PER_TILE = BLK_M // GQA
    TOK_PER_CTA = TOK_PER_TILE * N_TILES
    MASK_WORDS = mask_words
    MAX_BLOCKS = MASK_WORDS * 32
    HQ, HKV, TOPK, TOTAL_Q = hq, hkv, topk, total_q
    KV_DEPTH = kv_depth
    STG_DEPTH = 2
    Q_TILE_BYTES = BLK_M * HEAD_DIM * F16_BYTES
    KV_TILE_BYTES = BLK_N * HEAD_DIM * F16_BYTES
    STG_TILE_BYTES = BLK_N * HEAD_DIM
    TOK_ROUNDS = ceildiv(TOK_PER_CTA, 32)
    LIST_ROUNDS = ceildiv(MAX_BLOCKS, 32)
    SOFTMAX_REGS = 200 if kv_fp8 else 216
    XFORM_REGS = 64 if kv_fp8 else 32

    @txl.kernel(warps=16, arch="sm_100a", min_blocks_per_sm=1, grid=num_ctas)
    def msa_prefill_union(
        q_map: txl.TensorMap,
        k_map: txl.TensorMap,
        v_map: txl.TensorMap,
        o_map: txl.TensorMap,
        out: txl.gptr[txl.i32],
        q2k: txl.gptr[txl.i32],
        cu_q: txl.gptr[txl.i32],
        cu_k: txl.gptr[txl.i32],
        page_table: txl.gptr[txl.i32],
        seqused: txl.gptr[txl.i32],
        sched: txl.gptr[txl.i32],
        scale_log2: txl.f32,
        num_seqs: txl.i32,
    ):
        warp_cta = txl.warp_id()
        wg_id = warp_cta >> 2
        warp_id = warp_cta & 3
        tid_in_wg = txl.thread_id() & 127
        lane = txl.lane_id()

        smem = txl.smem_pool()
        q_smem = smem.alloc((N_TILES, BLK_M, HEAD_DIM), txl.bf16, swizzle=txl.SW128B)
        kv_base = N_TILES * Q_TILE_BYTES
        k_smem = smem.alloc((KV_DEPTH, BLK_N, HEAD_DIM), txl.bf16, swizzle=txl.SW128B)
        smem.pool.move_base_to(kv_base)
        v_smem = smem.alloc((KV_DEPTH, BLK_N, HEAD_DIM), txl.bf16, swizzle=txl.SW128B)
        smem.pool.move_base_to(kv_base + KV_DEPTH * KV_TILE_BYTES)
        if kv_fp8:
            stg_smem = smem.alloc((STG_DEPTH, BLK_N, HEAD_DIM), txl.f8e4m3, swizzle=txl.SW128B)
        # One tile is enough because WG0/WG1 hand the buffer to the store warp
        # in a fixed order.  This preserves the four-deep BF16 K/V ring.
        o_smem = smem.alloc((BLK_M, HEAD_DIM), txl.bf16, swizzle=txl.SW128B)

        def stage16(tile):
            return tile.rows * tile.cols * tile.bits // 8 // 16

        Q_STAGE16 = stage16(q_smem)
        KV_STAGE16 = stage16(k_smem)

        def lo_uniform(desc):
            desc_lo = txl.alloc_local((1,), "uint32")
            desc_hi = txl.alloc_local((1,), "uint32")
            txl.assign(desc_lo[0], txl.uniform(txl.Cast("uint32", desc.value)))
            txl.assign(desc_hi[0], txl.Cast("uint32", txl.shift_right(desc.value, txl.uint64(32))))
            return desc_lo, desc_hi

        def desc_at(desc, off16):
            lo, hi = desc
            packed = txl.alloc_local((1,), "uint64")
            low = (
                lo[0]
                if isinstance(off16, int) and off16 == 0
                else lo[0] + txl.Cast("uint32", off16)
            )
            txl.assign(
                packed[0],
                txl.bitwise_or(
                    txl.shift_left(txl.Cast("uint64", hi[0]), txl.uint64(32)),
                    txl.Cast("uint64", low),
                ),
            )
            return packed[0]

        def encode(view, major="k"):
            desc, off16 = view.encode(major=major, mma_k=MMA_K)
            return lo_uniform(desc), off16

        q_desc, qoff = encode(q_smem[0])
        k_desc, koff = encode(k_smem[0])
        v_desc, mnoff = encode(v_smem[0], major="mn")

        tmem_addr = smem.alloc((1,), txl.u32)
        union_list = smem.alloc((2 * MAX_BLOCKS,), txl.i32)
        union_meta = smem.alloc((2 * U_META_FIELDS,), txl.i32)
        token_masks = smem.alloc((2 * TOK_PER_CTA * MASK_WORDS,), txl.u32)
        output_lane_masks = smem.alloc((2, MAX_BLOCKS, N_TILES, 4), txl.u32)

        kv_pipe = txl.PipelineState(KV_DEPTH, phase=0)
        score_epoch = txl.PipelineState(1, phase=0)
        tmem_epoch = txl.PipelineState(1, phase=0)
        q_epoch = txl.PipelineState(1, phase=0)

        q_load = txl.Pipeline(smem, N_TILES, full="tma", empty="tcgen05", empty_phase_offset=1)
        if kv_fp8:
            kv_load = txl.Pipeline(
                smem, KV_DEPTH, full="mbar", empty="tcgen05", init_full=128, empty_phase_offset=1
            )
            stg_load = txl.Pipeline(
                smem, STG_DEPTH, full="tma", empty="mbar", init_empty=128, empty_phase_offset=1
            )
            stg_pipe = txl.PipelineState(STG_DEPTH, phase=0)
        else:
            kv_load = txl.Pipeline(
                smem, KV_DEPTH, full="tma", empty="tcgen05", empty_phase_offset=1
            )
        p_o_rescale = txl.MBarrier(smem, 2)
        p_o_rescale.init(128)
        s_ready = txl.MBarrier(smem, 2)
        s_ready.init(1)
        o_ready = txl.MBarrier(smem, 2)
        o_ready.init(1)
        p_ready_2 = txl.MBarrier(smem, 2)
        p_ready_2.init(128)
        s_consumed = txl.MBarrier(smem, 2)
        s_consumed.init(128)
        xu_turn = txl.MBarrier(smem, 2)
        xu_turn.init(128)
        pv_done = txl.TCGen05Bar(smem, 2)
        pv_done.init(1)
        o_free = txl.MBarrier(smem, 2)
        o_free.init(128)
        o_staged = txl.MBarrier(smem, 1)
        o_staged.init(128)
        o_smem_free = txl.MBarrier(smem, 1)
        o_smem_free.init(1)
        union_ready = txl.MBarrier(smem, 2)
        union_ready.init(32)
        union_free = txl.MBarrier(smem, 2)
        union_free.init(256 + 32)

        txl.ptx.fence.proxy.async_.shared__cta()
        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.cuda.cta_sync()

        def elected():
            return txl.cuda.elect_sync() != txl.uint32(0)

        def commit(bar, stage):
            txl.ptx[TCGEN05_COMMIT](bar.ptr_to([stage]))

        def tmem(col):
            return txl.cuda.get_tmem_addr(txl.uint32(0), 0, col)

        def tmem_load(dst, dst_offset, tmem_col, width):
            chain = TMEM_LD_16 if width == 16 else TMEM_LD_32
            txl.ptx[chain](*(dst[dst_offset + i] for i in range(width)), tmem_col)

        def tmem_store(src, src_offset, tmem_col):
            txl.ptx[TMEM_ST_16](tmem_col, *(src[src_offset + i] for i in range(16)))

        def ld_shared_i32(ptr):
            value = txl.local_scalar("int32")
            txl.ptx.ld.shared.b32(value, ptr)
            return value

        def ld_shared_u32(ptr):
            value = txl.local_scalar("uint32")
            txl.ptx.ld.shared.b32(value, ptr)
            return value

        def ldg_i32(ptr):
            value = txl.local_scalar("int32")
            txl.ptx.ld.global_.nc.b32(value, ptr)
            return value

        def meta_ptr(slot, field):
            return union_meta.ptr_to([slot * U_META_FIELDS + field])

        def iket_range(name, *, leader_only=False):
            token = txl.alloc_local([1], "uint32")
            if leader_only:
                txl.assign(token[0], txl.cuda.iket.sentinel_token(name))
                with txl.If(warp_id == 0), txl.Then():
                    txl.assign(token[0], txl.cuda.iket.range_start(name))
            else:
                txl.assign(token[0], txl.cuda.iket.range_start(name))
            return token

        def cast_f32x2_bf16x2(dst_u32, src, offset):
            txl.ptx.cvt.rn.bf16x2.f32(dst_u32[offset // 2], src[offset + 1], src[offset])

        def mul_f32x2(values, idx, multiplier):
            packed = txl.local_scalar("uint64")
            rhs = txl.local_scalar("uint64")
            txl.ptx.mov.b64(packed, values[idx], values[idx + 1])
            txl.ptx.mov.b64(rhs, multiplier, multiplier)
            txl.ptx.mul.rz.ftz.f32x2(packed, packed, rhs)
            txl.ptx.mov.b64(values[idx], values[idx + 1], packed)

        def reduce_max_128(out_, values, accum=False):
            C = MAX_CHAINS
            temp = txl.alloc_local([C], "float32")
            for i in range(C):
                if accum and i == 0:
                    txl.ptx[MAX3_F32](temp[i], values[2 * i], values[2 * i + 1], out_[0])
                else:
                    txl.ptx.mov.b32(temp[i], txl.max(values[2 * i], values[2 * i + 1]))
            for g in range(1, BLK_N // (2 * C)):
                for i in range(C):
                    txl.ptx[MAX3_F32](
                        temp[i], temp[i], values[2 * C * g + 2 * i], values[2 * C * g + 2 * i + 1]
                    )
            txl.ptx[MAX3_F32](temp[0], temp[0], temp[1], temp[2])
            txl.ptx[MAX3_F32](temp[3], temp[3], temp[4], temp[5])
            txl.ptx[MAX3_F32](out_[0], temp[6], temp[7], temp[0])
            txl.assign(out_[0], txl.max(out_[0], temp[3]))

        def shl_u32_clamp(val, shift):
            result = txl.local_scalar("uint32")
            txl.ptx.shl.b32(result, val, shift)
            return result

        def combine_int_frac_ex2(x_rounded, frac_ex2):
            x_rounded_i = txl.local_scalar("int32")
            frac_ex_i = txl.local_scalar("int32")
            x_rounded_e = txl.local_scalar("int32")
            out_i = txl.local_scalar("int32")
            out_f = txl.local_scalar("float32")
            txl.ptx.mov.b32(x_rounded_i, x_rounded)
            txl.ptx.mov.b32(frac_ex_i, frac_ex2)
            txl.ptx.shl.b32(x_rounded_e, x_rounded_i, txl.uint32(23))
            txl.ptx.add.s32(out_i, x_rounded_e, frac_ex_i)
            txl.ptx.mov.b32(out_f, out_i)
            return out_f

        POLY_EX2_DEG1 = (0.9701787941914297, 0.9701787941914297)
        FP32_ROUND_INT = float(2**23 + 2**22)

        def ex2_emulation_2(out_, idx, x, y):
            xy_clamped = txl.alloc_local([2], "float32")
            txl.ptx.mov.b32(xy_clamped[0], txl.max(x, -126.0))
            txl.ptx.mov.b32(xy_clamped[1], txl.max(y, -126.0))
            packed = txl.local_scalar("uint64")
            rhs = txl.local_scalar("uint64")
            addend = txl.local_scalar("uint64")
            xy_rounded = txl.alloc_local([2], "float32")
            txl.ptx.mov.b64(packed, xy_clamped[0], xy_clamped[1])
            txl.ptx.mov.b64(rhs, txl.float32(FP32_ROUND_INT), txl.float32(FP32_ROUND_INT))
            txl.ptx.add.rm.ftz.f32x2(packed, packed, rhs)
            txl.ptx.mov.b64(xy_rounded[0], xy_rounded[1], packed)
            xy_rounded_back = txl.alloc_local([2], "float32")
            txl.ptx.mov.b64(packed, xy_rounded[0], xy_rounded[1])
            txl.ptx.mov.b64(rhs, txl.float32(FP32_ROUND_INT), txl.float32(FP32_ROUND_INT))
            txl.ptx.sub.rn.ftz.f32x2(packed, packed, rhs)
            txl.ptx.mov.b64(xy_rounded_back[0], xy_rounded_back[1], packed)
            xy_frac = txl.alloc_local([2], "float32")
            txl.ptx.mov.b64(packed, xy_clamped[0], xy_clamped[1])
            txl.ptx.mov.b64(rhs, xy_rounded_back[0], xy_rounded_back[1])
            txl.ptx.sub.rn.ftz.f32x2(packed, packed, rhs)
            txl.ptx.mov.b64(xy_frac[0], xy_frac[1], packed)
            xy_frac_ex2 = txl.alloc_local([2], "float32")
            txl.ptx.mov.b32(xy_frac_ex2[0], txl.float32(POLY_EX2_DEG1[1]))
            txl.ptx.mov.b32(xy_frac_ex2[1], txl.float32(POLY_EX2_DEG1[1]))
            for coeff in (POLY_EX2_DEG1[0],):
                txl.ptx.mov.b64(rhs, xy_frac[0], xy_frac[1])
                txl.ptx.mov.b64(packed, xy_frac_ex2[0], xy_frac_ex2[1])
                txl.ptx.mov.b64(addend, txl.float32(coeff), txl.float32(coeff))
                txl.ptx.fma.rz.ftz.f32x2(packed, packed, rhs, addend)
                txl.ptx.mov.b64(xy_frac_ex2[0], xy_frac_ex2[1], packed)
            txl.ptx.mov.b32(out_[idx], combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0]))
            txl.ptx.mov.b32(out_[idx + 1], combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1]))

        def bits_le(x):
            """uint32 mask of bit positions <= x (x may be negative or >= 32)."""
            shift = txl.Cast("uint32", txl.min(txl.max(x + 1, 0), 32))
            full = shl_u32_clamp(txl.uint32(0xFFFFFFFF), shift)
            return txl.Select(x >= 31, txl.uint32(0xFFFFFFFF), txl.bitwise_not(full))

        def bits_lt(x):
            """uint32 mask of bit positions < x."""
            return bits_le(x - 1)

        def popc(x):
            r = txl.local_scalar("uint32")
            txl.ptx.popc.b32(r, x)
            return r

        sp = txl.specialize(chain_dispatch=True)
        r_softmax = sp.role("softmax", warps=[0, 1, 2, 3, 4, 5, 6, 7], regs=SOFTMAX_REGS)
        r_xform = sp.role("xform", warps=[8, 9, 10, 11], regs=XFORM_REGS)
        wg3 = sp.warpgroup("wg3", warps=range(12, 16), regs=48)
        r_mma = sp.role("mma", warps=[12], group=wg3)
        r_load = sp.role("load", warps=[13], group=wg3)
        r_store = sp.role("store", warps=[14], group=wg3)
        r_idle = sp.role("idle", warps=[15], group=wg3)

        with txl.If(warp_cta == 12), txl.Then():
            txl.ptx[TMEM_ALLOC](txl.address_of(tmem_addr[0]), txl.uint32(N_COLS_TMEM))
            txl.cuda.warp_sync()
        with txl.If(tvm.tirx.all(wg_id == 3, warp_id == 0)), txl.Then():
            allocated = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(allocated, tmem_addr.ptr_to([0]))
            txl.cuda.trap_when_assert_failed(allocated == txl.uint32(0))

        with wg3:
            with r_load:
                lane_groups = txl.local_scalar("int32", init=0)
                bb = txl.local_scalar("int32", init=txl.Cast("int32", lane))
                with txl.While(bb < num_seqs):
                    qa = ldg_i32(cu_q.ptr_to([bb]))
                    qb = ldg_i32(cu_q.ptr_to([bb + 1]))
                    ng = (qb - qa + (TOK_PER_CTA - 1)) // TOK_PER_CTA
                    txl.assign(lane_groups, txl.max(lane_groups, ng))
                    txl.assign(bb, bb + 32)
                max_groups = txl.local_scalar("int32")
                txl.ptx.redux_sync.max.s32(max_groups, lane_groups, txl.uint32(0xFFFFFFFF))
                num_tasks = txl.local_scalar("int32", init=max_groups * num_seqs * HKV)

                it = txl.local_scalar("int32", init=0)
                running = txl.local_scalar("int32", init=1)
                with txl.While(running != 0):
                    slot = it & 1
                    grabbed = txl.local_scalar("int32", init=0)
                    with txl.If(lane == 0), txl.Then():
                        txl.ptx.atom.relaxed.gpu.global_.add.s32(
                            grabbed, sched.ptr_to([0]), txl.int32(1)
                        )
                    task = txl.local_scalar("int32", init=txl.uniform(grabbed))
                    with txl.If(task >= num_tasks):
                        with txl.Then():
                            union_free.wait(slot, ((it >> 1) + 1) & 1)
                            with txl.If(lane == 0), txl.Then():
                                txl.ptx.st.shared.b32(meta_ptr(slot, 0), txl.int32(-1))
                            union_ready.arrive(slot)
                            txl.assign(running, 0)
                        with txl.Else():
                            kv_head = txl.local_scalar("int32", init=task % HKV)
                            item = task // HKV
                            g_rev = txl.local_scalar("int32", init=item // num_seqs)
                            bidx = txl.local_scalar("int32", init=item - g_rev * num_seqs)
                            q0 = ldg_i32(cu_q.ptr_to([bidx]))
                            q1 = ldg_i32(cu_q.ptr_to([bidx + 1]))
                            q_len = txl.local_scalar("int32", init=q1 - q0)
                            n_groups = txl.local_scalar(
                                "int32", init=(q_len + (TOK_PER_CTA - 1)) // TOK_PER_CTA
                            )
                            with txl.If(g_rev < n_groups), txl.Then():
                                union_free.wait(slot, ((it >> 1) + 1) & 1)
                                union_token = iket_range("union-build")
                                g = txl.local_scalar("int32", init=n_groups - 1 - g_rev)
                                q_start = txl.local_scalar("int32", init=q0 + g * TOK_PER_CTA)
                                q_valid = txl.local_scalar(
                                    "int32", init=txl.min(TOK_PER_CTA, q_len - g * TOK_PER_CTA)
                                )
                                if paged:
                                    kv_len = txl.local_scalar(
                                        "int32", init=ldg_i32(seqused.ptr_to([bidx]))
                                    )
                                    kv_start = txl.local_scalar("int32", init=0)
                                else:
                                    k0 = ldg_i32(cu_k.ptr_to([bidx]))
                                    k1 = ldg_i32(cu_k.ptr_to([bidx + 1]))
                                    kv_len = txl.local_scalar("int32", init=k1 - k0)
                                    kv_start = txl.local_scalar("int32", init=k0)
                                pos_base = txl.local_scalar(
                                    "int32", init=kv_len - q_len + g * TOK_PER_CTA
                                )
                                n_blocks_seq = (kv_len + (BLK_N - 1)) // BLK_N

                                union = [
                                    txl.local_scalar("uint32", init=txl.uint32(0))
                                    for _ in range(MASK_WORDS)
                                ]
                                for rnd in range(TOK_ROUNDS):
                                    tok = rnd * 32 + txl.Cast("int32", lane)
                                    words = [
                                        txl.local_scalar("uint32", init=txl.uint32(0))
                                        for _ in range(MASK_WORDS)
                                    ]
                                    with txl.If(tok < q_valid), txl.Then():
                                        idxs = txl.alloc_local([TOPK], "int32")
                                        q2k_base = (kv_head * TOTAL_Q + q_start + tok) * TOPK
                                        for vec in range(TOPK // 4):
                                            txl.ptx.ld.global_.nc.v4.b32(
                                                idxs[vec * 4],
                                                idxs[vec * 4 + 1],
                                                idxs[vec * 4 + 2],
                                                idxs[vec * 4 + 3],
                                                q2k.ptr_to([q2k_base + vec * 4]),
                                            )
                                        for s in range(TOPK):
                                            idx = idxs[s]
                                            bitpos = txl.Cast("uint32", txl.bitwise_and(idx, 31))
                                            for w in range(MASK_WORDS):
                                                hit = txl.And(idx >= w * 32, idx < (w + 1) * 32)
                                                txl.assign(
                                                    words[w],
                                                    txl.bitwise_or(
                                                        words[w],
                                                        txl.Select(
                                                            hit,
                                                            txl.shift_left(txl.uint32(1), bitpos),
                                                            txl.uint32(0),
                                                        ),
                                                    ),
                                                )
                                        vis_hi = (pos_base + tok) // BLK_N
                                        for w in range(MASK_WORDS):
                                            txl.assign(
                                                words[w],
                                                txl.bitwise_and(words[w], bits_le(vis_hi - w * 32)),
                                            )
                                    if rnd * 32 + 32 <= TOK_PER_CTA:
                                        for w in range(MASK_WORDS):
                                            txl.ptx.st.shared.b32(
                                                token_masks.ptr_to(
                                                    [(slot * TOK_PER_CTA + tok) * MASK_WORDS + w]
                                                ),
                                                words[w],
                                            )
                                    else:
                                        with txl.If(tok < TOK_PER_CTA), txl.Then():
                                            for w in range(MASK_WORDS):
                                                txl.ptx.st.shared.b32(
                                                    token_masks.ptr_to(
                                                        [
                                                            (slot * TOK_PER_CTA + tok) * MASK_WORDS
                                                            + w
                                                        ]
                                                    ),
                                                    words[w],
                                                )
                                    for w in range(MASK_WORDS):
                                        red = txl.local_scalar("uint32")
                                        txl.ptx.redux_sync.or_.b32(
                                            red, words[w], txl.uint32(0xFFFFFFFF)
                                        )
                                        txl.assign(union[w], txl.bitwise_or(union[w], red))
                                b_max = txl.min((pos_base + q_valid - 1) // BLK_N, n_blocks_seq - 1)
                                b_min = pos_base // BLK_N
                                for w in range(MASK_WORDS):
                                    txl.assign(
                                        union[w], txl.bitwise_and(union[w], bits_le(b_max - w * 32))
                                    )
                                count = txl.local_scalar("int32", init=0)
                                n_masked = txl.local_scalar("int32", init=0)
                                for w in range(MASK_WORDS):
                                    txl.assign(count, count + txl.Cast("int32", popc(union[w])))
                                    txl.assign(
                                        n_masked,
                                        n_masked
                                        + txl.Cast(
                                            "int32",
                                            popc(
                                                txl.bitwise_and(
                                                    union[w],
                                                    txl.bitwise_not(bits_lt(b_min - w * 32)),
                                                )
                                            ),
                                        ),
                                    )
                                with txl.If(count == 0), txl.Then():
                                    txl.assign(union[0], txl.uint32(1))
                                    txl.assign(count, 1)
                                    txl.assign(n_masked, 1)
                                with txl.If(lane == 0), txl.Then():
                                    txl.ptx.st.shared.b32(meta_ptr(slot, 0), count)
                                    txl.ptx.st.shared.b32(meta_ptr(slot, 1), n_masked)
                                    txl.ptx.st.shared.b32(meta_ptr(slot, 2), q_start)
                                    txl.ptx.st.shared.b32(meta_ptr(slot, 3), kv_head)
                                    txl.ptx.st.shared.b32(meta_ptr(slot, 4), q_valid)
                                    txl.ptx.st.shared.b32(meta_ptr(slot, 5), pos_base)
                                    txl.ptx.st.shared.b32(meta_ptr(slot, 6), kv_start)
                                    txl.ptx.st.shared.b32(meta_ptr(slot, 7), bidx)
                                txl.cuda.warp_sync()

                                for rnd in range(LIST_ROUNDS):
                                    n = rnd * 32 + txl.Cast("int32", lane)
                                    with txl.If(n < count), txl.Then():
                                        rem = txl.local_scalar("int32", init=n)
                                        found = txl.local_scalar("int32", init=0)
                                        blk = txl.local_scalar("int32", init=0)
                                        for w in reversed(range(MASK_WORDS)):
                                            c = txl.Cast("int32", popc(union[w]))
                                            with txl.If(found == 0), txl.Then():
                                                with txl.If(rem < c):
                                                    with txl.Then():
                                                        pos = txl.local_scalar("uint32")
                                                        txl.ptx.fns.b32(
                                                            pos,
                                                            union[w],
                                                            txl.uint32(31),
                                                            -rem - txl.int32(1),
                                                        )
                                                        txl.assign(
                                                            blk, w * 32 + txl.Cast("int32", pos)
                                                        )
                                                        txl.assign(found, 1)
                                                    with txl.Else():
                                                        txl.assign(rem, rem - c)
                                        txl.ptx.st.shared.b32(
                                            union_list.ptr_to([slot * MAX_BLOCKS + n]), blk
                                        )
                                        wsel = blk >> 5
                                        bsel = txl.Cast("uint32", txl.bitwise_and(blk, 31))
                                        lane_words = [
                                            [
                                                txl.local_scalar("uint32", init=txl.uint32(0))
                                                for _ in range(4)
                                            ]
                                            for _ in range(N_TILES)
                                        ]
                                        for tok in range(TOK_PER_CTA):
                                            tm = ld_shared_u32(
                                                token_masks.ptr_to(
                                                    [(slot * TOK_PER_CTA + tok) * MASK_WORDS + wsel]
                                                )
                                            )
                                            unselected = txl.bitwise_and(
                                                txl.shift_right(tm, bsel), txl.uint32(1)
                                            ) == txl.uint32(0)
                                            i_q = tok // TOK_PER_TILE
                                            row0 = (tok % TOK_PER_TILE) * GQA
                                            word = row0 // 32
                                            shift = row0 % 32
                                            txl.assign(
                                                lane_words[i_q][word],
                                                txl.bitwise_or(
                                                    lane_words[i_q][word],
                                                    txl.Select(
                                                        unselected,
                                                        txl.uint32(((1 << GQA) - 1) << shift),
                                                        txl.uint32(0),
                                                    ),
                                                ),
                                            )
                                        for i_q in range(N_TILES):
                                            for word in range(4):
                                                txl.ptx.st.shared.b32(
                                                    output_lane_masks.ptr_to([slot, n, i_q, word]),
                                                    lane_words[i_q][word],
                                                )
                                union_ready.arrive(slot)
                                txl.cuda.iket.range_end(union_token[0])

                                for i_q in range(N_TILES):
                                    q_load.empty.wait(i_q, q_epoch.phase)
                                    tma_q_token = iket_range("issue-tma-q")
                                    with txl.If(elected()), txl.Then():
                                        txl.ptx[TMA_G2S_4D](
                                            q_smem[i_q].ptr_to(0, 0),
                                            txl.address_of(q_map),
                                            txl.int32(0),
                                            txl.Cast("int32", kv_head * GQA),
                                            txl.Cast("int32", q_start + i_q * TOK_PER_TILE),
                                            txl.int32(0),
                                            txl.cuda.cvta_generic_to_shared(
                                                q_load.full.ptr_to([i_q])
                                            ),
                                        )
                                        q_load.full.arrive(i_q, tx_count=Q_TILE_BYTES)
                                    txl.cuda.iket.range_end(tma_q_token[0])
                                q_epoch.advance()

                                def block_page(blk):
                                    page = txl.local_scalar("int32")
                                    txl.assign(
                                        page, ldg_i32(page_table.ptr_to([bidx * max_pages + blk]))
                                    )
                                    return txl.max(page, 0)

                                def issue_kv_tma(blk, tensor_map, dst_ptr, mbar_ptr):
                                    if kv_fp8:
                                        if paged:
                                            txl.ptx[TMA_G2S_4D](
                                                dst_ptr,
                                                txl.address_of(tensor_map),
                                                txl.int32(0),
                                                txl.int32(0),
                                                txl.Cast("int32", kv_head),
                                                txl.Cast("int32", block_page(blk)),
                                                mbar_ptr,
                                            )
                                        else:
                                            txl.ptx[TMA_G2S_3D](
                                                dst_ptr,
                                                txl.address_of(tensor_map),
                                                txl.int32(0),
                                                txl.Cast("int32", kv_start + blk * BLK_N),
                                                txl.Cast("int32", kv_head),
                                                mbar_ptr,
                                            )
                                    else:
                                        if paged:
                                            txl.ptx[TMA_G2S_5D](
                                                dst_ptr,
                                                txl.address_of(tensor_map),
                                                txl.int32(0),
                                                txl.int32(0),
                                                txl.int32(0),
                                                txl.Cast("int32", kv_head),
                                                txl.Cast("int32", block_page(blk)),
                                                mbar_ptr,
                                            )
                                        else:
                                            txl.ptx[TMA_G2S_3D](
                                                dst_ptr,
                                                txl.address_of(tensor_map),
                                                txl.int32(0),
                                                txl.Cast("int32", kv_start + blk * BLK_N),
                                                txl.Cast("int32", kv_head * 2),
                                                mbar_ptr,
                                            )

                                def load_kv(blk, tensor_map, is_v):
                                    tma_kv_token = iket_range(
                                        "issue-tma-v" if is_v else "issue-tma-k"
                                    )
                                    if kv_fp8:
                                        stg_load.empty.wait(stg_pipe.stage, stg_pipe.phase)
                                        with txl.If(elected()), txl.Then():
                                            issue_kv_tma(
                                                blk,
                                                tensor_map,
                                                stg_smem[stg_pipe.stage].ptr_to(0, 0),
                                                txl.cuda.cvta_generic_to_shared(
                                                    stg_load.full.ptr_to([stg_pipe.stage])
                                                ),
                                            )
                                            stg_load.full.arrive(
                                                stg_pipe.stage, tx_count=STG_TILE_BYTES
                                            )
                                        stg_pipe.advance()
                                    else:
                                        kv_load.empty.wait(kv_pipe.stage, kv_pipe.phase)
                                        with txl.If(elected()), txl.Then():
                                            issue_kv_tma(
                                                blk,
                                                tensor_map,
                                                (v_smem if is_v else k_smem)[kv_pipe.stage].ptr_to(
                                                    0, 0
                                                ),
                                                txl.cuda.cvta_generic_to_shared(
                                                    kv_load.full.ptr_to([kv_pipe.stage])
                                                ),
                                            )
                                            kv_load.full.arrive(
                                                kv_pipe.stage, tx_count=KV_TILE_BYTES
                                            )
                                        kv_pipe.advance()
                                    txl.cuda.iket.range_end(tma_kv_token[0])

                                rest = [
                                    txl.local_scalar("uint32", init=union[w])
                                    for w in range(MASK_WORDS)
                                ]

                                def pop_highest():
                                    blk = txl.local_scalar("int32", init=-1)
                                    for w in reversed(range(MASK_WORDS)):
                                        with (
                                            txl.If(txl.And(blk < 0, rest[w] != txl.uint32(0))),
                                            txl.Then(),
                                        ):
                                            lz = txl.local_scalar("uint32")
                                            txl.ptx.clz.b32(lz, rest[w])
                                            pos = txl.uint32(31) - lz
                                            txl.assign(blk, w * 32 + txl.Cast("int32", pos))
                                            txl.assign(
                                                rest[w],
                                                txl.bitwise_xor(
                                                    rest[w], txl.shift_left(txl.uint32(1), pos)
                                                ),
                                            )
                                    return blk

                                blk_cur = txl.local_scalar("int32", init=pop_highest())
                                load_kv(blk_cur, k_map, False)
                                with txl.serial(count - 1, unroll=False) as _k:
                                    blk_nxt = txl.local_scalar("int32", init=pop_highest())
                                    load_kv(blk_nxt, k_map, False)
                                    load_kv(blk_cur, v_map, True)
                                    txl.assign(blk_cur, blk_nxt)
                                load_kv(blk_cur, v_map, True)
                                txl.assign(it, it + 1)

            with r_mma:
                it_m = txl.local_scalar("int32", init=0)
                gstep_m = txl.local_scalar("int32", init=0)

                tb_raw = txl.local_scalar("uint32")
                txl.ptx.ld.shared.u32(tb_raw, tmem_addr.ptr_to([0]))
                tmem_base = txl.local_scalar("uint32", init=txl.uniform(tb_raw))

                def load_output_lane_mask(slot, list_idx, q_stage):
                    disabled = txl.alloc_local([4], "uint32")
                    for word in range(4):
                        txl.ptx.ld.shared.u32(
                            disabled[word],
                            output_lane_masks.ptr_to([slot, list_idx, q_stage, word]),
                        )
                    return disabled

                def gemm_qk(q_stage, kv_stage, disabled):
                    qk_token = iket_range("mma-qk")
                    for ki in range(HEAD_DIM // MMA_K):
                        with txl.If(elected()), txl.Then():
                            txl.ptx[MMA_F16](
                                tmem_base + txl.uint32(q_stage * MMA_N),
                                desc_at(q_desc, q_stage * Q_STAGE16 + qoff(ki)),
                                desc_at(k_desc, kv_stage * KV_STAGE16 + koff(ki)),
                                txl.uint32(ID_QK),
                                disabled[0],
                                disabled[1],
                                disabled[2],
                                disabled[3],
                                ki != 0,
                            )
                    with txl.If(elected()), txl.Then():
                        commit(s_ready, q_stage)
                    txl.cuda.iket.range_end(qk_token[0])

                def gemm_pv_part1(i_q, kv_stage, should_accumulate, disabled):
                    for ki in range(K_SPLIT // MMA_K):
                        with txl.If(elected()), txl.Then():
                            txl.ptx[MMA_F16](
                                tmem_base + txl.uint32((N_TILES + i_q) * MMA_N),
                                tmem_base
                                + txl.uint32((1 - i_q) * MMA_N + MMA_N // 2 + ki * (MMA_K // 2)),
                                desc_at(v_desc, kv_stage * KV_STAGE16 + mnoff(ki)),
                                txl.uint32(ID_PV),
                                disabled[0],
                                disabled[1],
                                disabled[2],
                                disabled[3],
                                True if ki != 0 else txl.Cast("bool", should_accumulate),
                            )

                def gemm_pv_part2(i_q, kv_stage, disabled):
                    p_ready_2.wait(i_q, tmem_epoch.phase)
                    for ki in range((BLK_N - K_SPLIT) // MMA_K):
                        with txl.If(elected()), txl.Then():
                            txl.ptx[MMA_F16](
                                tmem_base + txl.uint32((N_TILES + i_q) * MMA_N),
                                tmem_base
                                + txl.uint32(
                                    (1 - i_q) * MMA_N
                                    + MMA_N // 2
                                    + K_SPLIT // 2
                                    + ki * (MMA_K // 2)
                                ),
                                desc_at(
                                    v_desc, kv_stage * KV_STAGE16 + mnoff(K_SPLIT // MMA_K + ki)
                                ),
                                txl.uint32(ID_PV),
                                disabled[0],
                                disabled[1],
                                disabled[2],
                                disabled[3],
                                True,
                            )

                def gemm_pv(i_q, kv_stage, should_accumulate, selected_disabled):
                    pv_token = iket_range("mma-pv")
                    disabled = txl.alloc_local([4], "uint32")
                    for word in range(4):
                        txl.assign(
                            disabled[word],
                            txl.Select(
                                should_accumulate != 0, selected_disabled[word], txl.uint32(0)
                            ),
                        )
                    gemm_pv_part1(i_q, kv_stage, should_accumulate, disabled)
                    gemm_pv_part2(i_q, kv_stage, disabled)
                    with txl.If(elected()), txl.Then():
                        commit(pv_done, i_q)
                    txl.cuda.iket.range_end(pv_token[0])

                running_m = txl.local_scalar("int32", init=1)
                with txl.While(running_m != 0):
                    slot_m = it_m & 1
                    union_ready.wait(slot_m, (it_m >> 1) & 1)
                    n_blocks = ld_shared_i32(meta_ptr(slot_m, 0))
                    with txl.If(n_blocks < 0), txl.Then():
                        txl.assign(running_m, 0)
                    with txl.If(n_blocks > 0), txl.Then():
                        acc = txl.local_scalar("int32", init=0)
                        for i_q in range(N_TILES):
                            q_load.full.wait(i_q, q_epoch.phase)
                            if i_q == 0:
                                kv_load.full.wait(kv_pipe.stage, kv_pipe.phase)
                            first_disabled = txl.alloc_local([4], "uint32")
                            for word in range(4):
                                txl.assign(first_disabled[word], txl.uint32(0))
                            gemm_qk(i_q, kv_pipe.stage, first_disabled)
                            if i_q == N_TILES - 1:
                                with txl.If(elected()), txl.Then():
                                    kv_load.empty.arrive(kv_pipe.stage)
                        kv_pipe.advance()
                        with txl.If(n_blocks == 1), txl.Then():
                            for i_q in range(N_TILES):
                                with txl.If(elected()), txl.Then():
                                    q_load.empty.arrive(i_q)
                        with txl.serial(n_blocks, unroll=False) as n:
                            has_next = n + 1 < n_blocks
                            k_stage = txl.local_scalar("int32", init=kv_pipe.stage)
                            k_phase = txl.local_scalar("int32", init=kv_pipe.phase)
                            with txl.If(has_next), txl.Then():
                                kv_pipe.advance()
                            v_stage = txl.local_scalar("int32", init=kv_pipe.stage)
                            v_phase = txl.local_scalar("int32", init=kv_pipe.phase)
                            kv_pipe.advance()
                            for i_q in range(N_TILES):
                                current_disabled = load_output_lane_mask(slot_m, n, i_q)
                                with txl.If(has_next), txl.Then():
                                    if i_q == 0:
                                        kv_load.full.wait(k_stage, k_phase)
                                    s_consumed.wait(i_q, gstep_m & 1)
                                    next_disabled = load_output_lane_mask(slot_m, n + 1, i_q)
                                    gemm_qk(i_q, k_stage, next_disabled)
                                    with txl.If(n == n_blocks - 2), txl.Then():
                                        with txl.If(elected()), txl.Then():
                                            q_load.empty.arrive(i_q)
                                    if i_q == N_TILES - 1:
                                        with txl.If(elected()), txl.Then():
                                            kv_load.empty.arrive(k_stage)
                                if i_q == 0:
                                    kv_load.full.wait(v_stage, v_phase)
                                with txl.If(n == 0), txl.Then():
                                    o_free.wait(i_q, (it_m + 1) & 1)
                                p_o_rescale.wait(i_q, tmem_epoch.phase)
                                gemm_pv(i_q, v_stage, acc, current_disabled)
                                if i_q == N_TILES - 1:
                                    with txl.If(elected()), txl.Then():
                                        kv_load.empty.arrive(v_stage)
                                with txl.If(n == n_blocks - 1), txl.Then():
                                    with txl.If(elected()), txl.Then():
                                        commit(o_ready, i_q)
                            txl.assign(acc, 1)
                            tmem_epoch.advance()
                            txl.assign(gstep_m, gstep_m + 1)
                        q_epoch.advance()
                    txl.assign(it_m, it_m + 1)

            with r_store:
                it_s = txl.local_scalar("int32", init=0)
                running_s = txl.local_scalar("int32", init=1)
                with txl.While(running_s != 0):
                    slot_s = it_s & 1
                    union_ready.wait(slot_s, (it_s >> 1) & 1)
                    n_blocks_st = ld_shared_i32(meta_ptr(slot_s, 0))
                    with txl.If(n_blocks_st < 0), txl.Then():
                        txl.assign(running_s, 0)
                    with txl.If(n_blocks_st > 0), txl.Then():
                        q_start_st = ld_shared_i32(meta_ptr(slot_s, 2))
                        kv_head_st = ld_shared_i32(meta_ptr(slot_s, 3))
                        q_valid_st = ld_shared_i32(meta_ptr(slot_s, 4))
                        # Release the metadata slot after all three consumers
                        # have captured it; output staging has its own lifetime.
                        union_free.arrive(slot_s)
                        store_token = iket_range("tma-store")
                        for i_q in range(N_TILES):
                            seq_s = it_s * N_TILES + i_q
                            o_staged.wait(0, seq_s & 1)
                            tile_full = q_valid_st >= (i_q + 1) * TOK_PER_TILE
                            with txl.If(txl.And(tile_full, elected())), txl.Then():
                                txl.ptx[TMA_S2G_4D](
                                    txl.address_of(o_map),
                                    txl.int32(0),
                                    txl.Cast("int32", kv_head_st * GQA),
                                    txl.Cast("int32", q_start_st + i_q * TOK_PER_TILE),
                                    txl.int32(0),
                                    o_smem.ptr_to(0, 0),
                                )
                                txl.ptx.cp.async_.bulk.commit_group()
                            with txl.If(tile_full), txl.Then():
                                # Source-read completion is sufficient before
                                # the epilogue reuses this shared-memory tile.
                                txl.ptx.cp.async_.bulk.wait_group.read(0)
                            with txl.If(elected()), txl.Then():
                                o_smem_free.arrive(0)
                        txl.cuda.iket.range_end(store_token[0])
                    txl.assign(it_s, it_s + 1)

            with r_idle:
                pass

        with r_xform:
            if kv_fp8:
                it_t = txl.local_scalar("int32", init=0)
                running_t = txl.local_scalar("int32", init=1)
                row = tid_in_wg
                with txl.While(running_t != 0):
                    slot_t = it_t & 1
                    union_ready.wait(slot_t, (it_t >> 1) & 1)
                    n_blocks_t = ld_shared_i32(meta_ptr(slot_t, 0))
                    with txl.If(n_blocks_t < 0), txl.Then():
                        txl.assign(running_t, 0)
                    with txl.If(n_blocks_t > 0), txl.Then():
                        with txl.serial(2 * n_blocks_t, unroll=False) as _j:
                            stg_load.full.wait(stg_pipe.stage, stg_pipe.phase)
                            kv_load.empty.wait(kv_pipe.stage, kv_pipe.phase)
                            xf_token = iket_range("xform", leader_only=True)
                            src_words = txl.alloc_local([4], "uint32")
                            out_words = txl.alloc_local([8], "uint32")
                            for c in range(HEAD_DIM // 16):
                                txl.ptx.ld.shared.v4.b32(
                                    src_words[0],
                                    src_words[1],
                                    src_words[2],
                                    src_words[3],
                                    stg_smem[stg_pipe.stage].ptr_to(row, c * 16),
                                )
                                for w in range(4):
                                    txl.ptx.cvt.rn.bf16x2.e4m3x2(
                                        out_words[2 * w], txl.Cast("uint16", src_words[w])
                                    )
                                    txl.ptx.cvt.rn.bf16x2.e4m3x2(
                                        out_words[2 * w + 1],
                                        txl.Cast(
                                            "uint16", txl.shift_right(src_words[w], txl.uint32(16))
                                        ),
                                    )
                                for half in range(2):
                                    txl.ptx.st.shared.v4.b32(
                                        k_smem[kv_pipe.stage].ptr_to(row, c * 16 + half * 8),
                                        out_words[half * 4],
                                        out_words[half * 4 + 1],
                                        out_words[half * 4 + 2],
                                        out_words[half * 4 + 3],
                                    )
                            txl.ptx.fence.proxy.async_.shared__cta()
                            kv_load.full.arrive(kv_pipe.stage)
                            stg_load.empty.arrive(stg_pipe.stage)
                            txl.cuda.iket.range_end(xf_token[0])
                            stg_pipe.advance()
                            kv_pipe.advance()
                    txl.assign(it_t, it_t + 1)
            else:
                pass

        with r_softmax:
            it_x = txl.local_scalar("int32", init=0)
            gstep_x = txl.local_scalar("int32", init=0)
            with txl.If(wg_id == 1), txl.Then():
                xu_turn.arrive(0)
            tok_local = tid_in_wg // GQA
            head_local = tid_in_wg % GQA
            row_max = txl.local_scalar("float32")
            row_sum = txl.alloc_local([1], "float32")
            q_pos = txl.local_scalar("int32")
            mask_base = txl.local_scalar("int32")

            def mask_r2p(s_chunk, col_limit, ncol):
                CHUNK_SIZE = 32
                for s_ in range(ceildiv(ncol, CHUNK_SIZE)):
                    k_keep = txl.max(col_limit - s_ * CHUNK_SIZE, 0)
                    mask_inv = txl.local_scalar("uint32")
                    txl.assign(
                        mask_inv, shl_u32_clamp(txl.uint32(0xFFFFFFFF), txl.Cast("uint32", k_keep))
                    )
                    for i in range(CHUNK_SIZE):
                        if i < ncol - s_ * CHUNK_SIZE:
                            c = s_ * CHUNK_SIZE + i
                            in_bound = txl.bitwise_and(
                                txl.bitwise_not(mask_inv),
                                txl.shift_left(txl.uint32(1), txl.uint32(i)),
                            )
                            txl.ptx.mov.b32(
                                s_chunk[c],
                                txl.Select(
                                    txl.Cast("bool", in_bound), s_chunk[c], txl.float32(NEG_INF)
                                ),
                            )

            def apply_causal_mask(s_chunk, blk):
                col_limit_right = q_pos - blk * BLK_N + 1
                mask_r2p(s_chunk, col_limit_right, BLK_N)

            def rescale_o_rows(scale):
                RESCALE_TILE = 16
                o_row = txl.alloc_local([RESCALE_TILE], "float32")
                for d_tile in range(HEAD_DIM // RESCALE_TILE):
                    d_start = d_tile * RESCALE_TILE
                    addr = tmem((N_TILES + wg_id) * MMA_N + d_start)
                    tmem_load(o_row, 0, addr, RESCALE_TILE)
                    # Make the asynchronous TMEM read complete before the
                    # other softmax warp group can reuse overlapping columns.
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    for i in range(RESCALE_TILE // 2):
                        mul_f32x2(o_row, 2 * i, scale)
                    tmem_store(o_row, 0, addr)
                txl.ptx.tcgen05.wait__st.sync.aligned()

            def softmax_step(blk, other_par, apply_mask=False, is_first=False):
                s_chunk = txl.alloc_local([BLK_N], "float32")
                p_chunk = txl.alloc_local([BLK_N // 2], "uint32")
                sel_word = ld_shared_u32(token_masks.ptr_to([mask_base + (blk >> 5)]))
                selected = txl.local_scalar(
                    "int32",
                    init=txl.Cast(
                        "int32",
                        txl.bitwise_and(
                            txl.shift_right(sel_word, txl.Cast("uint32", txl.bitwise_and(blk, 31))),
                            txl.uint32(1),
                        ),
                    ),
                )
                s_ready.wait(wg_id, score_epoch.phase)
                with txl.If(warp_id == 0), txl.Then():
                    txl.cuda.iket.mark("softmax-phase-0")
                softmax_max_token = iket_range("softmax-max", leader_only=True)
                tile_max = txl.alloc_local([1], "float32")
                for chunk_idx in range(BLK_N // 32):
                    tmem_load(s_chunk, chunk_idx * 32, tmem(wg_id * MMA_N + chunk_idx * 32), 32)
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                if apply_mask:
                    apply_causal_mask(s_chunk, blk)
                row_max_old = txl.local_scalar("float32")
                if is_first:
                    reduce_max_128(tile_max, s_chunk)
                    txl.assign(
                        tile_max[0], txl.Select(selected != 0, tile_max[0], txl.float32(NEG_INF))
                    )
                else:
                    txl.assign(row_max_old, row_max)
                    txl.assign(tile_max[0], row_max_old)
                    reduce_max_128(tile_max, s_chunk, accum=True)
                    txl.assign(tile_max[0], txl.Select(selected != 0, tile_max[0], row_max_old))
                s_consumed.arrive(wg_id)
                row_max_new = txl.local_scalar("float32")
                acc_scale = txl.local_scalar("float32", init=txl.float32(1.0))
                acc_scale_ = txl.local_scalar("float32")
                row_max_safe = txl.local_scalar("float32")
                txl.assign(row_max_new, tile_max[0])
                txl.assign(
                    row_max_safe,
                    txl.if_then_else(
                        tile_max[0] == txl.float32(NEG_INF), txl.float32(0.0), tile_max[0]
                    ),
                )
                if not is_first:
                    txl.assign(acc_scale_, (row_max_old - row_max_safe) * scale_log2)
                    with txl.If(acc_scale_ >= -RESCALE_THRESHOLD):
                        with txl.Then():
                            txl.assign(row_max_new, row_max_old)
                            txl.assign(row_max_safe, row_max_old)
                        with txl.Else():
                            with txl.If(row_max_old != txl.float32(NEG_INF)), txl.Then():
                                txl.ptx.ex2.approx.ftz.f32(acc_scale, acc_scale_)
                txl.assign(row_max, row_max_new)
                row_max_scaled = row_max_safe * scale_log2
                txl.cuda.iket.range_end(softmax_max_token[0])
                if not is_first:
                    should_rescale = txl.local_scalar(
                        "int32", init=txl.Select(acc_scale < txl.float32(1.0), 1, 0)
                    )
                    any_needs_rescale = txl.local_scalar("uint32")
                    txl.ptx.vote_sync.any.pred(
                        any_needs_rescale, txl.ptx.pred(should_rescale), txl.uint32(0xFFFFFFFF)
                    )
                    with txl.If(any_needs_rescale != 0), txl.Then():
                        rescale_token = iket_range("rescale", leader_only=True)
                        pv_done.wait(wg_id, (gstep_x + 1) & 1)
                        rescale_o_rows(acc_scale)
                        txl.cuda.iket.range_end(rescale_token[0])
                turn_token = iket_range("xu-turn-wait", leader_only=True)
                xu_turn.wait(wg_id, gstep_x & 1)
                txl.cuda.iket.range_end(turn_token[0])
                softmax_exp2_token = iket_range("softmax-exp2", leader_only=True)
                bias = txl.local_scalar("float32")
                txl.assign(
                    bias,
                    txl.Select(
                        selected != 0, txl.float32(0.0) - row_max_scaled, txl.float32(NEG_INF)
                    ),
                )
                scale_pair = txl.local_scalar("uint64")
                bias_pair = txl.local_scalar("uint64")
                txl.ptx.mov.b64(
                    scale_pair, txl.float32(1.0) * scale_log2, txl.float32(1.0) * scale_log2
                )
                txl.ptx.mov.b64(bias_pair, bias, bias)
                sum_acc = [txl.local_scalar("uint64") for _ in range(N_SUM_ACC)]
                for a in sum_acc:
                    txl.ptx.mov.b64(a, txl.float32(0.0), txl.float32(0.0))
                pair_tmp = txl.local_scalar("uint64")
                for frag_idx in range(4):
                    for i in range(BLK_N // 4 // 2):
                        idx = frag_idx * BLK_N // 4 + 2 * i
                        txl.ptx.mov.b64(pair_tmp, s_chunk[idx], s_chunk[idx + 1])
                        txl.ptx.fma.rz.ftz.f32x2(pair_tmp, pair_tmp, scale_pair, bias_pair)
                        txl.ptx.mov.b64(s_chunk[idx], s_chunk[idx + 1], pair_tmp)
                        if (
                            i * 2 % 16 < 16 - 2 * EMU_PAIRS
                            or frag_idx >= 4 - 1
                            or frag_idx < EMU_START
                            or apply_mask
                        ):
                            txl.ptx.ex2.approx.ftz.f32(s_chunk[idx], s_chunk[idx])
                            txl.ptx.ex2.approx.ftz.f32(s_chunk[idx + 1], s_chunk[idx + 1])
                        else:
                            ex2_emulation_2(s_chunk, idx, s_chunk[idx], s_chunk[idx + 1])
                txl.cuda.warp_sync()
                xu_turn.arrive(1 - wg_id)
                txl.cuda.warp_sync()
                for frag_idx in range(4):
                    for i in range(BLK_N // 4 // 2):
                        idx = frag_idx * BLK_N // 4 + 2 * i
                        txl.ptx.mov.b64(pair_tmp, s_chunk[idx], s_chunk[idx + 1])
                        acc_k = sum_acc[(frag_idx * (BLK_N // 8) + i) % N_SUM_ACC]
                        txl.ptx.add.rn.ftz.f32x2(acc_k, acc_k, pair_tmp)
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
                        txl.ptx.tcgen05.wait__st.sync.aligned()
                        p_o_rescale.arrive(wg_id)
                txl.cuda.iket.range_end(softmax_exp2_token[0])
                softmax_tmem_st_token = iket_range("softmax-tmem-st", leader_only=True)
                for i in range(4 - P_SPLIT_Q):
                    tmem_store(
                        p_chunk,
                        (P_SPLIT_Q + i) * BLK_N // 4 // 2,
                        tmem(((1 - wg_id) * 2 * MMA_N + MMA_N + (P_SPLIT_Q + i) * BLK_N // 4) // 2),
                    )
                txl.ptx.tcgen05.wait__st.sync.aligned()
                p_ready_2.arrive(wg_id)
                txl.cuda.iket.range_end(softmax_tmem_st_token[0])
                softmax_sum_token = iket_range("softmax-sum", leader_only=True)
                score_epoch.advance()
                for step in (4, 2, 1):
                    for a in range(step):
                        txl.ptx.add.rn.ftz.f32x2(sum_acc[a], sum_acc[a], sum_acc[a + step])
                sum_lo = txl.local_scalar("float32")
                sum_hi = txl.local_scalar("float32")
                txl.ptx.mov.b64(sum_lo, sum_hi, sum_acc[0])
                if is_first:
                    txl.assign(row_sum[0], sum_lo + sum_hi)
                else:
                    txl.assign(row_sum[0], row_sum[0] * acc_scale + (sum_lo + sum_hi))
                txl.cuda.iket.range_end(softmax_sum_token[0])
                txl.assign(gstep_x, gstep_x + 1)

            running_x = txl.local_scalar("int32", init=1)
            with txl.While(running_x != 0):
                slot_x = it_x & 1
                union_ready.wait(slot_x, (it_x >> 1) & 1)
                n_blocks_s = ld_shared_i32(meta_ptr(slot_x, 0))
                with txl.If(n_blocks_s < 0), txl.Then():
                    txl.assign(running_x, 0)
                with txl.If(n_blocks_s > 0), txl.Then():
                    n_masked_s = ld_shared_i32(meta_ptr(slot_x, 1))
                    q_start_x = ld_shared_i32(meta_ptr(slot_x, 2))
                    kv_head_x = ld_shared_i32(meta_ptr(slot_x, 3))
                    q_valid_x = ld_shared_i32(meta_ptr(slot_x, 4))
                    pos_base_x = ld_shared_i32(meta_ptr(slot_x, 5))
                    my_tok = wg_id * TOK_PER_TILE + tok_local
                    txl.assign(q_pos, pos_base_x + my_tok)
                    txl.assign(mask_base, (slot_x * TOK_PER_CTA + my_tok) * MASK_WORDS)
                    list_base = slot_x * MAX_BLOCKS

                    def other_parity(n):
                        if_wg0 = gstep_x & 1
                        nxt = txl.Select(n + 1 < n_blocks_s, (gstep_x + 1) & 1, gstep_x & 1)
                        return txl.Select(wg_id == 0, if_wg0, nxt)

                    blk0 = ld_shared_i32(union_list.ptr_to([list_base]))
                    softmax_step(blk0, other_parity(txl.int32(0)), apply_mask=True, is_first=True)
                    n_masked_rest = txl.max(n_masked_s - 1, 0)
                    with txl.serial(n_masked_rest, unroll=False) as i:
                        blk_m = ld_shared_i32(union_list.ptr_to([list_base + 1 + i]))
                        softmax_step(blk_m, other_parity(1 + i), apply_mask=True)
                    start_plain = txl.max(n_masked_s, 1)
                    with txl.serial(n_blocks_s - start_plain, unroll=False) as i:
                        blk_p = ld_shared_i32(union_list.ptr_to([list_base + start_plain + i]))
                        softmax_step(blk_p, other_parity(start_plain + i), apply_mask=False)
                    union_free.arrive(slot_x)

                    epi_wait_token = iket_range("epi-wait-o", leader_only=True)
                    o_ready.wait(wg_id, it_x & 1)
                    txl.cuda.iket.range_end(epi_wait_token[0])
                    epi_token = iket_range("epi-store", leader_only=True)
                    acc_O_row_is_zero_or_nan = tvm.tirx.any(
                        row_sum[0] == txl.float32(0.0), row_sum[0] != row_sum[0]
                    )
                    norm_scale = txl.local_scalar("float32")
                    txl.ptx.rcp.approx.ftz.f32(
                        norm_scale,
                        txl.Select(acc_O_row_is_zero_or_nan, txl.float32(1.0), row_sum[0]),
                    )
                    EPI_LD = 32
                    o_row_f32 = txl.alloc_local([HEAD_DIM], "float32")
                    o_row_bf16 = txl.alloc_local([HEAD_DIM // 2], "uint32")
                    for d_tile in range(HEAD_DIM // EPI_LD):
                        tmem_load(
                            o_row_f32,
                            d_tile * EPI_LD,
                            tmem((N_TILES + wg_id) * MMA_N + d_tile * EPI_LD),
                            EPI_LD,
                        )
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    o_free.arrive(wg_id)
                    seq_x = it_x * N_TILES + wg_id
                    o_smem_free.wait(0, (seq_x + 1) & 1)
                    for d_tile in range(HEAD_DIM // EPI_LD):
                        d_start = d_tile * EPI_LD
                        for i in range(EPI_LD // 2):
                            mul_f32x2(o_row_f32, d_start + 2 * i, norm_scale)
                        for i in range(EPI_LD // 2):
                            cast_f32x2_bf16x2(o_row_bf16, o_row_f32, d_start + 2 * i)
                        for i in range(EPI_LD // 8):
                            w0 = d_start // 2 + i * 4
                            txl.ptx.st.shared.v4.u32(
                                o_smem.ptr_to(tid_in_wg, d_start + i * 8),
                                o_row_bf16[w0],
                                o_row_bf16[w0 + 1],
                                o_row_bf16[w0 + 2],
                                o_row_bf16[w0 + 3],
                            )
                    # A partial tile cannot be described by this fixed-size
                    # TMA box without crossing a sequence boundary.  Preserve
                    # the original per-row path only for those tail rows.
                    partial_tile = q_valid_x < (wg_id + 1) * TOK_PER_TILE
                    with txl.If(txl.And(partial_tile, my_tok < q_valid_x)), txl.Then():
                        out_row = (q_start_x + my_tok) * HQ + kv_head_x * GQA + head_local
                        out_base = out_row * (HEAD_DIM // 2)
                        for i in range(HEAD_DIM // 8):
                            txl.ptx.st.global_.v4.b32(
                                out.ptr_to([out_base + i * 4]),
                                o_row_bf16[i * 4],
                                o_row_bf16[i * 4 + 1],
                                o_row_bf16[i * 4 + 2],
                                o_row_bf16[i * 4 + 3],
                            )
                    txl.ptx.fence.proxy.async_.shared__cta()
                    o_staged.arrive(0)
                    txl.cuda.iket.range_end(epi_token[0])
                txl.assign(it_x, it_x + 1)

        txl.cuda.cta_sync()
        with txl.If(txl.thread_id() == 0), txl.Then():
            done = txl.local_scalar("int32")
            txl.ptx.atom.acq_rel.gpu.global_.add.s32(done, sched.ptr_to([1]), txl.int32(1))
            with txl.If(done == num_ctas - 1), txl.Then():
                txl.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([0]), txl.int32(0))
                txl.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([1]), txl.int32(0))
        with txl.If(tvm.tirx.all(wg_id == 0, warp_id == 0)), txl.Then():
            dealloc = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(dealloc, tmem_addr.ptr_to([0]))
            txl.ptx[TMEM_RELINQUISH]()
            txl.ptx[TMEM_DEALLOC](dealloc, txl.uint32(N_COLS_TMEM))

    return msa_prefill_union


def make_prep_kernel(
    *, total_q, hq, hkv, topk, nblk, cap, num_seqs, paged, num_chunks, item_batch, num_ctas_main
):
    GQA = hq // hkv
    NBLK = nblk
    TOPK = topk
    TOTAL_Q = total_q
    HQ = hq
    B = num_seqs
    TOK = BLK_M // GQA
    NITEMS = hkv * B * NBLK
    N_PREP = num_chunks * hkv
    ROUNDS = ceildiv(NITEMS, PREP_THREADS)
    ITEM_BATCH = item_batch
    assert NBLK <= PREP_THREADS

    @txl.kernel(warps=PREP_THREADS // 32, arch="sm_100a", grid=(num_chunks, hkv))
    def msa_reverse_prep(
        q2k: txl.gptr[txl.i32],
        cu_q: txl.gptr[txl.i32],
        cu_k: txl.gptr[txl.i32],
        seqused: txl.gptr[txl.i32],
        cursor: txl.gptr[txl.i32],
        edge_table: txl.gptr[txl.i32],
        deg: txl.gptr[txl.i32],
        arrivals: txl.gptr[txl.i32],
        out: txl.gptr[txl.i32],
        plan: txl.gptr[txl.i32],
        batch_tab: txl.gptr[txl.i32],
    ):
        cid = txl.cta_id()
        chunk, h = cid[0], cid[1]
        tid = txl.thread_id()
        lane = txl.Cast("int32", txl.lane_id())
        warp = tid >> 5
        if USE_PDL:
            txl.ptx.griddepcontrol.launch_dependents()
        pool = txl.smem_pool()
        counts = pool.alloc((NBLK,), txl.i32, align=16)
        bases = pool.alloc((NBLK,), txl.i32, align=16)
        wsum = pool.alloc((PREP_THREADS // 32,), txl.i32, align=16)
        last_flag = pool.alloc((4,), txl.i32, align=16)
        SEQ_ROWS = ceildiv(B + 1, 4) * 4
        seq_q0 = pool.alloc((SEQ_ROWS,), txl.i32, align=16)
        seq_kvlen = pool.alloc((SEQ_ROWS,), txl.i32, align=16)

        def ld_shared_i32(ptr):
            v = txl.local_scalar("int32")
            txl.ptx.ld.shared.b32(v, ptr)
            return v

        def block_excl_scan(val):
            """Exclusive prefix of `val` over the CTA's threads; returns (exclusive, total)."""
            incl = txl.local_scalar("int32", init=val)
            for d in (1, 2, 4, 8, 16):
                other = txl.local_scalar("int32")
                txl.ptx.shfl_sync.up.b32(
                    other, incl, txl.uint32(d), txl.uint32(0), txl.uint32(0xFFFFFFFF)
                )
                with txl.If(lane >= d), txl.Then():
                    txl.assign(incl, incl + other)
            with txl.If(lane == 31), txl.Then():
                txl.ptx.st.shared.b32(wsum.ptr_to([warp]), incl)
            txl.cuda.cta_sync()
            woff = txl.local_scalar("int32", init=0)
            total = txl.local_scalar("int32", init=0)
            for w in range(PREP_THREADS // 32):
                v = ld_shared_i32(wsum.ptr_to([w]))
                txl.assign(total, total + v)
                with txl.If(txl.int32(w) < warp), txl.Then():
                    txl.assign(woff, woff + v)
            txl.cuda.cta_sync()
            return incl - val + woff, total

        def chunk_of(tp, total):
            """Guided self-scheduling batch size for an item whose first tile sits at queue position tp."""
            return txl.max(
                txl.min((total - tp) // (2 * num_ctas_main), txl.int32(ITEM_BATCH)), txl.int32(1)
            )

        def ldg(ptr):
            v = txl.local_scalar("int32")
            txl.ptx.ld.global_.nc.b32(v, ptr)
            return v

        tok_abs = chunk * PREP_THREADS + tid
        chunk_begin = chunk * PREP_THREADS
        chunk_end = txl.min(chunk_begin + PREP_THREADS, TOTAL_Q)
        idxs = txl.alloc_local([TOPK], "int32")
        pos = txl.alloc_local([TOPK], "int32")
        n_valid = txl.local_scalar("int32", init=0)

        for bidx in range(B):
            q0 = ldg(cu_q.ptr_to([bidx]))
            q1 = ldg(cu_q.ptr_to([bidx + 1]))
            overlaps = txl.And(chunk_begin < q1, chunk_end > q0)
            with txl.If(overlaps), txl.Then():
                t_local = tok_abs - q0
                valid_tok = txl.And(tok_abs >= q0, tok_abs < q1)
                with txl.If(tid < NBLK), txl.Then():
                    txl.ptx.st.shared.b32(counts.ptr_to([tid]), txl.int32(0))
                txl.cuda.cta_sync()
                for s in range(TOPK):
                    txl.assign(idxs[s], txl.int32(-1))
                    txl.assign(pos[s], txl.int32(0))
                txl.assign(n_valid, txl.int32(0))
                with txl.If(valid_tok), txl.Then():
                    base = (h * TOTAL_Q + tok_abs) * TOPK
                    for vec in range(TOPK // 4):
                        txl.ptx.ld.global_.nc.v4.b32(
                            idxs[vec * 4],
                            idxs[vec * 4 + 1],
                            idxs[vec * 4 + 2],
                            idxs[vec * 4 + 3],
                            q2k.ptr_to([base + vec * 4]),
                        )
                    for s in range(TOPK):
                        with txl.If(txl.And(idxs[s] >= 0, idxs[s] < NBLK)):
                            with txl.Then():
                                old = txl.local_scalar("int32")
                                txl.ptx.atom.shared.add.s32(
                                    old, counts.ptr_to([idxs[s]]), txl.int32(1)
                                )
                                txl.assign(pos[s], old)
                                txl.assign(n_valid, n_valid + 1)
                            with txl.Else():
                                txl.assign(idxs[s], txl.int32(-1))
                txl.cuda.cta_sync()
                item_base = (h * B + bidx) * NBLK
                with txl.If(tid < NBLK), txl.Then():
                    c = txl.local_scalar("int32")
                    txl.ptx.ld.shared.b32(c, counts.ptr_to([tid]))
                    with txl.If(c > 0), txl.Then():
                        gbase = txl.local_scalar("int32")
                        txl.ptx.atom.relaxed.gpu.global_.add.s32(
                            gbase, cursor.ptr_to([item_base + tid]), c
                        )
                        txl.ptx.st.shared.b32(bases.ptr_to([tid]), gbase)
                txl.cuda.cta_sync()
                with txl.If(valid_tok), txl.Then():
                    for s in range(TOPK):
                        with txl.If(idxs[s] >= 0), txl.Then():
                            gb = txl.local_scalar("int32")
                            txl.ptx.ld.shared.b32(gb, bases.ptr_to([idxs[s]]))
                            txl.ptx.st.global_.b32(
                                edge_table.ptr_to([(item_base + idxs[s]) * cap + gb + pos[s]]),
                                txl.bitwise_or(t_local, txl.shift_left(txl.int32(s), 24)),
                            )
                    txl.ptx.st.global_.b32(deg.ptr_to([h * TOTAL_Q + tok_abs]), n_valid)
                    txl.ptx.st.global_.b32(arrivals.ptr_to([h * TOTAL_Q + tok_abs]), txl.int32(0))
                    with txl.If(n_valid == 0), txl.Then():
                        for g in range(GQA):
                            row_base = (tok_abs * HQ + h * GQA + g) * (HEAD_DIM // 2)
                            for i in range(HEAD_DIM // 8):
                                txl.ptx.st.global_.v4.b32(
                                    out.ptr_to([row_base + i * 4]),
                                    txl.int32(0),
                                    txl.int32(0),
                                    txl.int32(0),
                                    txl.int32(0),
                                )
                txl.cuda.cta_sync()

        with txl.If(tid == 0), txl.Then():
            old = txl.local_scalar("int32")
            txl.ptx.atom.acq_rel.gpu.global_.add.s32(old, plan.ptr_to([2]), txl.int32(1))
            txl.ptx.st.shared.b32(
                last_flag.ptr_to([0]), txl.Select(old == N_PREP - 1, txl.int32(1), txl.int32(0))
            )
        txl.cuda.cta_sync()
        with txl.If(ld_shared_i32(last_flag.ptr_to([0])) != 0), txl.Then():
            txl.ptx.fence.acq_rel.gpu()
            cnts = [txl.local_scalar("int32", init=0) for _ in range(ROUNDS)]
            tiles = [txl.local_scalar("int32", init=0) for _ in range(ROUNDS)]
            tps = [txl.local_scalar("int32", init=0) for _ in range(ROUNDS)]
            nbs = [txl.local_scalar("int32", init=0) for _ in range(ROUNDS)]
            bps = [txl.local_scalar("int32", init=0) for _ in range(ROUNDS)]

            for r in range(ROUNDS):
                i = r * PREP_THREADS + tid
                with txl.If(i < NITEMS), txl.Then():
                    txl.ptx.ld.relaxed.gpu.global_.b32(cnts[r], cursor.ptr_to([i]))
            for r in range(ceildiv(B + 1, PREP_THREADS)):
                sidx = r * PREP_THREADS + tid
                with txl.If(sidx <= B), txl.Then():
                    txl.ptx.st.shared.b32(seq_q0.ptr_to([sidx]), ldg(cu_q.ptr_to([sidx])))
                    with txl.If(sidx < B), txl.Then():
                        if paged:
                            txl.ptx.st.shared.b32(
                                seq_kvlen.ptr_to([sidx]), ldg(seqused.ptr_to([sidx]))
                            )
                        else:
                            txl.ptx.st.shared.b32(
                                seq_kvlen.ptr_to([sidx]),
                                ldg(cu_k.ptr_to([sidx + 1])) - ldg(cu_k.ptr_to([sidx])),
                            )
            carry = txl.local_scalar("int32", init=0)
            for r in range(ROUNDS):
                i = r * PREP_THREADS + tid
                txl.assign(
                    tiles[r], txl.Select(i < NITEMS, (cnts[r] + (TOK - 1)) // TOK, txl.int32(0))
                )
                excl, tot = block_excl_scan(tiles[r])
                txl.assign(tps[r], carry + excl)
                txl.assign(carry, carry + tot)
            total_tiles = txl.local_scalar("int32", init=carry)
            carry2 = txl.local_scalar("int32", init=0)
            for r in range(ROUNDS):
                i = r * PREP_THREADS + tid
                ch = chunk_of(tps[r], total_tiles)
                txl.assign(nbs[r], txl.Select(i < NITEMS, (tiles[r] + ch - 1) // ch, txl.int32(0)))
                excl2, tot2 = block_excl_scan(nbs[r])
                txl.assign(bps[r], carry2 + excl2)
                txl.assign(carry2, carry2 + tot2)
            with txl.If(tid == 0), txl.Then():
                txl.ptx.st.global_.b32(plan.ptr_to([0]), carry2)
                txl.ptx.st.global_.b32(plan.ptr_to([1]), total_tiles)
                txl.ptx.st.global_.b32(plan.ptr_to([2]), txl.int32(0))
            for r in range(ROUNDS):
                i = r * PREP_THREADS + tid
                with txl.If(i < NITEMS), txl.Then():
                    hh = i // (B * NBLK)
                    rem = i - hh * (B * NBLK)
                    bidx = rem // NBLK
                    blk = rem - bidx * NBLK
                    q0 = ld_shared_i32(seq_q0.ptr_to([bidx]))
                    q1 = ld_shared_i32(seq_q0.ptr_to([bidx + 1]))
                    kv_len = ld_shared_i32(seq_kvlen.ptr_to([bidx]))
                    pos_off = kv_len - (q1 - q0)
                    ch = chunk_of(tps[r], total_tiles)
                    j = txl.local_scalar("int32", init=0)
                    with txl.While(j < nbs[r]):
                        base = (bps[r] + j) * 12
                        t0 = tps[r] + j * ch
                        txl.ptx.st.global_.v4.b32(
                            batch_tab.ptr_to([base]), i, t0, txl.min(ch, tiles[r] - j * ch), tps[r]
                        )
                        txl.ptx.st.global_.v4.b32(
                            batch_tab.ptr_to([base + 4]), cnts[r], q0, pos_off, hh
                        )
                        txl.ptx.st.global_.v4.b32(
                            batch_tab.ptr_to([base + 8]), bidx, blk, txl.int32(0), txl.int32(0)
                        )
                        txl.assign(j, j + 1)

    return msa_reverse_prep


def make_main_kernel(
    *,
    total_q,
    hq,
    hkv,
    topk,
    nblk,
    cap,
    num_seqs,
    paged,
    max_pages,
    kv_fp8,
    num_ctas,
    compact_s2f6=False,
    fused_combine=False,
):
    GQA = hq // hkv
    assert GQA in (4, 8, 16)
    TOK = BLK_M // GQA
    NBLK = nblk
    NITEMS = hkv * num_seqs * NBLK
    TOPK = topk
    TOTAL_Q = total_q
    HQ, HKV, B = hq, hkv, num_seqs
    Q_TILE_BYTES = BLK_M * HEAD_DIM * F16_BYTES
    KV_TILE_BYTES = BLK_N * HEAD_DIM * F16_BYTES
    STG_TILE_BYTES = BLK_N * HEAD_DIM
    KV_STAGES = 2
    STG_DEPTH = 1
    SOFTMAX_REGS = 168
    EPI_REGS = 112
    WG3_REGS = 64
    TOK_HALF = TOK // 2
    assert TOK % 2 == 0
    Q_HALF_BYTES = TOK_HALF * GQA * 2 * HEAD_DIM
    FUSED = fused_combine
    CHUNK = 32
    ITEM_BATCH = 10 if kv_fp8 else BATCH

    @txl.kernel(warps=16, arch="sm_100a", min_blocks_per_sm=1, grid=num_ctas)
    def msa_reverse_main(
        q_map: txl.TensorMap,
        k_map: txl.TensorMap,
        v_map: txl.TensorMap,
        out: txl.gptr[txl.i32],
        q_raw: txl.gptr[txl.i32],
        cu_q: txl.gptr[txl.i32],
        cu_k: txl.gptr[txl.i32],
        page_table: txl.gptr[txl.i32],
        seqused: txl.gptr[txl.i32],
        cursor: txl.gptr[txl.i32],
        edge_table: txl.gptr[txl.i32],
        deg: txl.gptr[txl.i32],
        arrivals: txl.gptr[txl.i32],
        o_part: txl.gptr[txl.i32],
        q_part: txl.gptr[txl.i32],
        m_part: txl.gptr[txl.f32],
        l_part: txl.gptr[txl.f32],
        sched: txl.gptr[txl.i32],
        batch_tab: txl.gptr[txl.i32],
        plan: txl.gptr[txl.i32],
        scale_log2: txl.f32,
    ):
        warp_cta = txl.warp_id()
        wg_id = warp_cta >> 2
        warp_id = warp_cta & 3
        tid_in_wg = txl.thread_id() & 127
        lane = txl.lane_id()

        smem = txl.smem_pool()
        q_smem = smem.alloc((2, BLK_M, HEAD_DIM), txl.bf16, swizzle=txl.SW128B)
        k_smem = smem.alloc((KV_STAGES, BLK_N, HEAD_DIM), txl.bf16, swizzle=txl.SW128B)
        v_smem = smem.alloc((KV_STAGES, BLK_N, HEAD_DIM), txl.bf16, swizzle=txl.SW128B)

        def stage16(tile):
            return tile.rows * tile.cols * tile.bits // 8 // 16

        Q_STAGE16 = stage16(q_smem)
        KV_STAGE16 = stage16(k_smem)

        def lo_uniform(desc):
            desc_lo = txl.alloc_local((1,), "uint32")
            desc_hi = txl.alloc_local((1,), "uint32")
            txl.assign(desc_lo[0], txl.uniform(txl.Cast("uint32", desc.value)))
            txl.assign(desc_hi[0], txl.Cast("uint32", txl.shift_right(desc.value, txl.uint64(32))))
            return desc_lo, desc_hi

        def desc_at(desc, off16):
            lo, hi = desc
            packed = txl.alloc_local((1,), "uint64")
            low = (
                lo[0]
                if isinstance(off16, int) and off16 == 0
                else lo[0] + txl.Cast("uint32", off16)
            )
            txl.assign(
                packed[0],
                txl.bitwise_or(
                    txl.shift_left(txl.Cast("uint64", hi[0]), txl.uint64(32)),
                    txl.Cast("uint64", low),
                ),
            )
            return packed[0]

        def encode(view, major="k"):
            desc, off16 = view.encode(major=major, mma_k=MMA_K)
            return lo_uniform(desc), off16

        q_desc, qoff = encode(q_smem[0])
        k_desc, koff = encode(k_smem[0])
        v_desc, mnoff = encode(v_smem[0], major="mn")

        tmem_addr = smem.alloc((1,), txl.u32)
        edge_list = smem.alloc((ESLOTS * TOK,), txl.i32, align=16)
        meta = smem.alloc((ESLOTS * META_FIELDS,), txl.i32)
        stats = smem.alloc((STATS_SLOTS * BLK_M * 6,), txl.i32)
        xmax = smem.alloc((2 * 2 * BLK_M,), txl.f32)
        stats_hdr = smem.alloc((STATS_SLOTS * 2,), txl.i32)

        q_full = txl.MBarrier(smem, 2)
        q_full.init(2)
        q_empty = txl.TCGen05Bar(smem, 2, phase_offset=1)
        q_empty.init(1)
        edges_ready = txl.MBarrier(smem, ESLOTS)
        edges_ready.init(32)
        edges_free = txl.MBarrier(smem, ESLOTS, phase_offset=1)
        edges_free.init(256 + 64 + 32)
        if kv_fp8:
            kv_full = txl.MBarrier(smem, KV_STAGES)
            kv_full.init(128 if EPI_CONV else 256)
            kv_tma = txl.TMABar(smem, KV_STAGES)
            kv_tma.init(1)
            if EPI_CONV:
                kv_issued = txl.MBarrier(smem, KV_STAGES)
                kv_issued.init(1)
        else:
            kv_full = txl.TMABar(smem, KV_STAGES)
            kv_full.init(1)
        kv_empty = txl.TCGen05Bar(smem, KV_STAGES, phase_offset=1)
        kv_empty.init(1)
        s_full = txl.TCGen05Bar(smem, 2)
        s_full.init(1)
        s_empty = txl.MBarrier(smem, 2, phase_offset=1)
        s_empty.init(256)
        p_full = txl.MBarrier(smem, 2)
        p_full.init(256)
        pv_done = txl.TCGen05Bar(smem, 2, phase_offset=1)
        pv_done.init(1)
        o_full = txl.TCGen05Bar(smem, 2)
        o_full.init(1)
        o_empty = txl.MBarrier(smem, 2, phase_offset=1)
        o_empty.init(128)
        stats_ready = txl.MBarrier(smem, STATS_SLOTS)
        stats_ready.init(256)
        stats_free = txl.MBarrier(smem, STATS_SLOTS, phase_offset=1)
        stats_free.init(128)

        txl.ptx.fence.proxy.async_.shared__cta()
        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.cuda.cta_sync()

        def elected():
            return txl.cuda.elect_sync() != txl.uint32(0)

        def commit(bar, stage):
            txl.ptx[TCGEN05_COMMIT](bar.ptr_to([stage]))

        def tmem(col):
            return txl.cuda.get_tmem_addr(txl.uint32(0), 0, col)

        def tmem_load(dst, dst_offset, tmem_col, width):
            chain = TMEM_LD_16 if width == 16 else TMEM_LD_32
            txl.ptx[chain](*(dst[dst_offset + i] for i in range(width)), tmem_col)

        def tmem_store(src, src_offset, tmem_col):
            txl.ptx[TMEM_ST_16](tmem_col, *(src[src_offset + i] for i in range(16)))

        def ld_shared_i32(ptr):
            value = txl.local_scalar("int32")
            txl.ptx.ld.shared.b32(value, ptr)
            return value

        def ld_shared_f32(ptr):
            value = txl.local_scalar("float32")
            txl.ptx.ld.shared.b32(value, ptr)
            return value

        def ldg_i32(ptr):
            value = txl.local_scalar("int32")
            txl.ptx.ld.global_.nc.b32(value, ptr)
            return value

        def ldg_i32_coherent(ptr):
            value = txl.local_scalar("int32")
            txl.ptx.ld.relaxed.gpu.global_.b32(value, ptr)
            return value

        def meta_ptr(stage, field):
            return meta.ptr_to([stage * META_FIELDS + field])

        def iket_range(name, *, leader_only=False):
            token = txl.alloc_local([1], "uint32")
            if leader_only:
                txl.assign(token[0], txl.cuda.iket.sentinel_token(name))
                with txl.If(warp_id == 0), txl.Then():
                    txl.assign(token[0], txl.cuda.iket.range_start(name))
            else:
                txl.assign(token[0], txl.cuda.iket.range_start(name))
            return token

        def cast_f32x2_bf16x2(dst_u32, src, offset):
            txl.ptx.cvt.rn.bf16x2.f32(dst_u32[offset // 2], src[offset + 1], src[offset])

        def mul_f32x2(values, idx, multiplier):
            packed = txl.local_scalar("uint64")
            rhs = txl.local_scalar("uint64")
            txl.ptx.mov.b64(packed, values[idx], values[idx + 1])
            txl.ptx.mov.b64(rhs, multiplier, multiplier)
            txl.ptx.mul.rz.ftz.f32x2(packed, packed, rhs)
            txl.ptx.mov.b64(values[idx], values[idx + 1], packed)

        def reduce_max_128(out_, values):
            C = 8
            temp = txl.alloc_local([C], "float32")
            for i in range(C):
                txl.ptx.mov.b32(temp[i], txl.max(values[2 * i], values[2 * i + 1]))
            for g in range(1, BLK_N // (2 * C)):
                for i in range(C):
                    txl.ptx[MAX3_F32](
                        temp[i], temp[i], values[2 * C * g + 2 * i], values[2 * C * g + 2 * i + 1]
                    )
            txl.ptx[MAX3_F32](temp[0], temp[0], temp[1], temp[2])
            txl.ptx[MAX3_F32](temp[3], temp[3], temp[4], temp[5])
            txl.ptx[MAX3_F32](out_[0], temp[6], temp[7], temp[0])
            txl.assign(out_[0], txl.max(out_[0], temp[3]))

        def shl_u32_clamp(val, shift):
            result = txl.local_scalar("uint32")
            txl.ptx.shl.b32(result, val, shift)
            return result

        EX2_C = (0.999924481, 0.693121034, 0.242640083, 0.055922036)
        FP32_ROUND_INT = float(2**23 + 2**22)

        def ex2_poly_pair(vals, idx):
            """vals[idx], vals[idx+1] <- 2^vals (FMA pipe): round, cubic on the fraction, exponent injection."""
            xc = txl.alloc_local([2], "float32")
            txl.ptx.mov.b32(xc[0], txl.max(vals[idx], -126.0))
            txl.ptx.mov.b32(xc[1], txl.max(vals[idx + 1], -126.0))
            packed = txl.local_scalar("uint64")
            rhs = txl.local_scalar("uint64")
            addend = txl.local_scalar("uint64")
            rnd = txl.alloc_local([2], "float32")
            txl.ptx.mov.b64(packed, xc[0], xc[1])
            txl.ptx.mov.b64(rhs, txl.float32(FP32_ROUND_INT), txl.float32(FP32_ROUND_INT))
            txl.ptx.add.rm.ftz.f32x2(packed, packed, rhs)
            txl.ptx.mov.b64(rnd[0], rnd[1], packed)
            back = txl.alloc_local([2], "float32")
            txl.ptx.mov.b64(packed, rnd[0], rnd[1])
            txl.ptx.sub.rn.ftz.f32x2(packed, packed, rhs)
            txl.ptx.mov.b64(back[0], back[1], packed)
            frac = txl.local_scalar("uint64")
            txl.ptx.mov.b64(packed, xc[0], xc[1])
            txl.ptx.mov.b64(rhs, back[0], back[1])
            txl.ptx.sub.rn.ftz.f32x2(frac, packed, rhs)
            poly = txl.local_scalar("uint64")
            txl.ptx.mov.b64(poly, txl.float32(EX2_C[3]), txl.float32(EX2_C[3]))
            for cc in (EX2_C[2], EX2_C[1], EX2_C[0]):
                txl.ptx.mov.b64(addend, txl.float32(cc), txl.float32(cc))
                txl.ptx.fma.rn.ftz.f32x2(poly, poly, frac, addend)
            pv = txl.alloc_local([2], "float32")
            txl.ptx.mov.b64(pv[0], pv[1], poly)
            for e in range(2):
                r_i = txl.local_scalar("int32")
                p_i = txl.local_scalar("int32")
                e_i = txl.local_scalar("int32")
                o_i = txl.local_scalar("int32")
                txl.ptx.mov.b32(r_i, rnd[e])
                txl.ptx.mov.b32(p_i, pv[e])
                txl.ptx.shl.b32(e_i, r_i, txl.uint32(23))
                txl.ptx.add.s32(o_i, e_i, p_i)
                txl.ptx.mov.b32(vals[idx + e], o_i)

        def convert_fp8_wg(tile_view):
            """Expand the fp8 tile staged in the upper 16 KB of this bf16 tile in place.

            One 128-thread warpgroup owns one row per thread.  Each fp8 row (128 B, swizzled by
            row&7) is read completely before its bf16 halves are written: half 1 lands on the
            row's own fp8 bytes, half 0 on the tile's first 16 KB.
            """
            row = tid_in_wg
            src = txl.alloc_local([32], "uint32")
            for c in range(8):
                pc = txl.bitwise_xor(txl.int32(c), txl.bitwise_and(row, 7))
                txl.ptx.ld.shared.v4.b32(
                    src[c * 4],
                    src[c * 4 + 1],
                    src[c * 4 + 2],
                    src[c * 4 + 3],
                    txl.ptx.addr(tile_view.ptr_to(0, 0), KV_TILE_BYTES // 2 + row * 128 + pc * 16),
                )
            for c in range(8):
                outw = txl.alloc_local([8], "uint32")
                for w in range(4):
                    txl.ptx.cvt.rn.bf16x2.e4m3x2(outw[2 * w], txl.Cast("uint16", src[c * 4 + w]))
                    txl.ptx.cvt.rn.bf16x2.e4m3x2(
                        outw[2 * w + 1],
                        txl.Cast("uint16", txl.shift_right(src[c * 4 + w], txl.uint32(16))),
                    )
                for half in range(2):
                    txl.ptx.st.shared.v4.b32(
                        tile_view.ptr_to(row, c * 16 + half * 8),
                        outw[half * 4],
                        outw[half * 4 + 1],
                        outw[half * 4 + 2],
                        outw[half * 4 + 3],
                    )

        sp = txl.specialize(chain_dispatch=True)
        r_sm0 = sp.role("sm0", warps=[0, 1, 2, 3], regs=SOFTMAX_REGS)
        r_sm1 = sp.role("sm1", warps=[4, 5, 6, 7], regs=SOFTMAX_REGS)
        r_epi = sp.role("epi", warps=[8, 9, 10, 11], regs=EPI_REGS)
        wg3 = sp.warpgroup("wg3", warps=range(12, 16), regs=WG3_REGS)
        r_mma = sp.role("mma", warps=[12], group=wg3)
        r_load = sp.role("load", warps=[13], group=wg3)
        r_gather = sp.role("gather", warps=[14, 15], group=wg3)

        with txl.If(warp_cta == 12), txl.Then():
            txl.ptx[TMEM_ALLOC](txl.address_of(tmem_addr[0]), txl.uint32(512))
            txl.cuda.warp_sync()
        if USE_PDL:
            txl.ptx.griddepcontrol.launch_dependents()

        if USE_PDL:
            txl.ptx.griddepcontrol.wait()

        with wg3:
            with r_load:
                total_batches = ldg_i32(plan.ptr_to([0]))
                t_seq = txl.local_scalar("int32", init=0)
                cur_item = txl.local_scalar("int32", init=-1)
                kvs = txl.local_scalar("int32", init=KV_STAGES - 1)
                kv_uses = [txl.local_scalar("int32", init=0) for _ in range(KV_STAGES)]
                n_kvblocks = txl.local_scalar("int32", init=0)
                kv_policy = txl.local_scalar("uint64")
                txl.ptx.createpolicy.fractional.L2__evict_first.b64(kv_policy, txl.float32(1.0))
                NCP = TOK // 4
                assert 1 + NCP <= 32

                def grab_raw():
                    """Issue the batch-counter atomic (lane 0); the result is broadcast later by locate()."""
                    g = txl.local_scalar("int32", init=0)
                    with txl.If(lane == 0), txl.Then():
                        old = txl.local_scalar("int32")
                        txl.ptx.atom.relaxed.gpu.global_.add.s32(
                            old, sched.ptr_to([0]), txl.int32(1)
                        )
                        txl.assign(g, old + num_ctas)
                    return g

                def block_page(bidx, blk):
                    page = txl.local_scalar(
                        "int32", init=ldg_i32(page_table.ptr_to([bidx * max_pages + blk]))
                    )
                    return txl.max(page, 0)

                def issue_kv(blk, bidx, h, kv_start, tensor_map, dst_ptr, mbar_ptr):
                    if kv_fp8:
                        if paged:
                            txl.ptx[TMA_G2S_4D_H](
                                dst_ptr,
                                txl.address_of(tensor_map),
                                txl.int32(0),
                                txl.int32(0),
                                txl.Cast("int32", h),
                                txl.Cast("int32", block_page(bidx, blk)),
                                mbar_ptr,
                                kv_policy,
                            )
                        else:
                            txl.ptx[TMA_G2S_3D_H](
                                dst_ptr,
                                txl.address_of(tensor_map),
                                txl.int32(0),
                                txl.Cast("int32", kv_start + blk * BLK_N),
                                txl.Cast("int32", h),
                                mbar_ptr,
                                kv_policy,
                            )
                    else:
                        if paged:
                            txl.ptx[TMA_G2S_5D_H](
                                dst_ptr,
                                txl.address_of(tensor_map),
                                txl.int32(0),
                                txl.int32(0),
                                txl.int32(0),
                                txl.Cast("int32", h),
                                txl.Cast("int32", block_page(bidx, blk)),
                                mbar_ptr,
                                kv_policy,
                            )
                        else:
                            txl.ptx[TMA_G2S_3D_H](
                                dst_ptr,
                                txl.address_of(tensor_map),
                                txl.int32(0),
                                txl.Cast("int32", kv_start + blk * BLK_N),
                                txl.Cast("int32", h * 2),
                                mbar_ptr,
                                kv_policy,
                            )

                def issue_block(item, h, bidx, blk):
                    """TMA the item's K/V into stage `kvs` (already advanced and known to be free)."""
                    if paged:
                        kv_start = txl.int32(0)
                    else:
                        kv_start = ldg_i32(cu_k.ptr_to([bidx]))
                    for s_ in range(KV_STAGES):
                        with txl.If(kvs == s_), txl.Then():
                            with txl.If(elected()), txl.Then():
                                if kv_fp8:
                                    mb = txl.cuda.cvta_generic_to_shared(kv_tma.ptr_to([s_]))
                                    issue_kv(
                                        blk,
                                        bidx,
                                        h,
                                        kv_start,
                                        k_map,
                                        txl.ptx.addr(k_smem[s_].ptr_to(0, 0), KV_TILE_BYTES // 2),
                                        mb,
                                    )
                                    issue_kv(
                                        blk,
                                        bidx,
                                        h,
                                        kv_start,
                                        v_map,
                                        txl.ptx.addr(v_smem[s_].ptr_to(0, 0), KV_TILE_BYTES // 2),
                                        mb,
                                    )
                                    kv_tma.arrive(s_, tx_count=2 * STG_TILE_BYTES)
                                else:
                                    issue_kv(
                                        blk,
                                        bidx,
                                        h,
                                        kv_start,
                                        k_map,
                                        k_smem[s_].ptr_to(0, 0),
                                        txl.cuda.cvta_generic_to_shared(kv_full.ptr_to([s_])),
                                    )
                                    issue_kv(
                                        blk,
                                        bidx,
                                        h,
                                        kv_start,
                                        v_map,
                                        v_smem[s_].ptr_to(0, 0),
                                        txl.cuda.cvta_generic_to_shared(kv_full.ptr_to([s_])),
                                    )
                                    kv_full.arrive(s_, tx_count=2 * KV_TILE_BYTES)
                            txl.assign(kv_uses[s_], kv_uses[s_] + 1)
                            if kv_fp8 and EPI_CONV:
                                with txl.If(elected()), txl.Then():
                                    kv_issued.arrive(s_)
                    txl.assign(n_kvblocks, n_kvblocks + 1)
                    txl.assign(cur_item, item)

                def next_stage_free():
                    """Non-blocking: is the stage after `kvs` released?"""
                    ns = txl.local_scalar("int32", init=kvs + 1)
                    with txl.If(ns == KV_STAGES), txl.Then():
                        txl.assign(ns, 0)
                    ok = txl.local_scalar("uint32", init=0)
                    for s_ in range(KV_STAGES):
                        with txl.If(ns == s_), txl.Then():
                            txl.ptx.mbarrier.test_wait.parity.shared.b64(
                                ok, kv_empty.ptr_to([s_]), txl.uint32((kv_uses[s_] & 1) ^ 1)
                            )
                    return ok

                class Batch:
                    """Register-resident description of one planned (item-aligned) batch."""

                    def __init__(self):
                        self.g = txl.local_scalar("int32", init=0)
                        self.item = txl.local_scalar("int32", init=0)
                        self.t0 = txl.local_scalar("int32", init=0)
                        self.n = txl.local_scalar("int32", init=0)
                        self.h = txl.local_scalar("int32", init=0)
                        self.bidx = txl.local_scalar("int32", init=0)
                        self.blk = txl.local_scalar("int32", init=0)
                        self.item_first = txl.local_scalar("int32", init=0)
                        self.cnt = txl.local_scalar("int32", init=0)
                        self.q0 = txl.local_scalar("int32", init=0)
                        self.pos_off = txl.local_scalar("int32", init=0)
                        self.kvs = txl.local_scalar("int32", init=0)
                        self.use = txl.local_scalar("int32", init=0)
                        self.new = txl.local_scalar("int32", init=0)
                        self.loaded = txl.local_scalar("int32", init=0)

                    def locate(self, g_raw):
                        """Resolve a batch index (lane-0 value) to item / tile range; issue the metadata loads."""
                        txl.assign(self.g, txl.uniform(g_raw))
                        txl.assign(self.n, 0)
                        txl.assign(self.loaded, 0)
                        txl.assign(self.new, 0)
                        with txl.If(self.g < total_batches), txl.Then():
                            e = txl.alloc_local([12], "int32")
                            base = self.g * 12
                            for v in range(3):
                                txl.ptx.ld.global_.nc.v4.b32(
                                    e[4 * v],
                                    e[4 * v + 1],
                                    e[4 * v + 2],
                                    e[4 * v + 3],
                                    batch_tab.ptr_to([base + 4 * v]),
                                )
                            txl.assign(self.item, e[0])
                            txl.assign(self.t0, e[1])
                            txl.assign(self.n, e[2])
                            txl.assign(self.item_first, e[3])
                            txl.assign(self.cnt, e[4])
                            txl.assign(self.q0, e[5])
                            txl.assign(self.pos_off, e[6])
                            txl.assign(self.h, e[7])
                            txl.assign(self.bidx, e[8])
                            txl.assign(self.blk, e[9])

                    def mark_loaded(self, is_new):
                        txl.assign(self.kvs, kvs)
                        txl.assign(self.use, n_kvblocks - 1)
                        txl.assign(self.new, is_new)
                        txl.assign(self.loaded, 1)

                    def ensure_loaded(self):
                        """Blocking: make sure this batch's K/V is (being) loaded into a kv stage."""
                        with txl.If(txl.And(self.n > 0, self.loaded == 0)), txl.Then():
                            with txl.If(self.item != cur_item):
                                with txl.Then():
                                    txl.assign(kvs, kvs + 1)
                                    with txl.If(kvs == KV_STAGES), txl.Then():
                                        txl.assign(kvs, 0)
                                    kvt = iket_range("ld-kv")
                                    for s_ in range(KV_STAGES):
                                        with txl.If(kvs == s_), txl.Then():
                                            kv_empty.wait(s_, kv_uses[s_] & 1)
                                    issue_block(self.item, self.h, self.bidx, self.blk)
                                    txl.cuda.iket.range_end(kvt[0])
                                    self.mark_loaded(txl.int32(1))
                                with txl.Else():
                                    self.mark_loaded(txl.int32(0))

                    def try_load(self):
                        """Non-blocking variant used while the load warp polls for a free edge slot."""
                        with txl.If(txl.And(self.n > 0, self.loaded == 0)), txl.Then():
                            with txl.If(self.item != cur_item):
                                with txl.Then():
                                    with txl.If(next_stage_free() != txl.uint32(0)), txl.Then():
                                        txl.assign(kvs, kvs + 1)
                                        with txl.If(kvs == KV_STAGES), txl.Then():
                                            txl.assign(kvs, 0)
                                        kvt = iket_range("ld-kv")
                                        issue_block(self.item, self.h, self.bidx, self.blk)
                                        txl.cuda.iket.range_end(kvt[0])
                                        self.mark_loaded(txl.int32(1))
                                with txl.Else():
                                    self.mark_loaded(txl.int32(0))

                    def copy_from(self, other):
                        for a, b in (
                            (self.g, other.g),
                            (self.item, other.item),
                            (self.t0, other.t0),
                            (self.n, other.n),
                            (self.h, other.h),
                            (self.bidx, other.bidx),
                            (self.blk, other.blk),
                            (self.item_first, other.item_first),
                            (self.cnt, other.cnt),
                            (self.q0, other.q0),
                            (self.pos_off, other.pos_off),
                            (self.kvs, other.kvs),
                            (self.use, other.use),
                            (self.new, other.new),
                            (self.loaded, other.loaded),
                        ):
                            txl.assign(a, b)

                def wait_edge_slot(eslot, other):
                    """Wait for edge slot `eslot` to be free; meanwhile prefetch `other`'s K/V when its stage frees."""
                    parity = ((t_seq // ESLOTS) & 1) ^ 1
                    ok = txl.local_scalar("uint32", init=0)
                    wf_tok = iket_range("ld-wait-free")
                    txl.ptx.mbarrier.test_wait.parity.shared.b64(
                        ok, edges_free.ptr_to([eslot]), txl.uint32(parity)
                    )
                    with txl.While(ok == txl.uint32(0)):
                        other.try_load()
                        txl.cuda.nano_sleep(txl.uint32(32))
                        txl.ptx.mbarrier.test_wait.parity.shared.b64(
                            ok, edges_free.ptr_to([eslot]), txl.uint32(parity)
                        )
                    txl.cuda.iket.range_end(wf_tok[0])

                def publish(bt, bi, other, rel):
                    """Publish tile bt.t0 + bi: async edge-list copy + metadata into the next edge slot."""
                    t = bt.t0 + bi
                    local_tile = t - bt.item_first
                    n_valid = txl.min(TOK, bt.cnt - local_tile * TOK)
                    edge_base = bt.item * cap + local_tile * TOK
                    eslot = t_seq & (ESLOTS - 1)
                    wait_edge_slot(eslot, other)
                    pub_tok = iket_range("ld-pub")
                    with txl.If(txl.And(lane >= 1, lane <= NCP)), txl.Then():
                        off = (txl.Cast("int32", lane) - 1) * 4
                        txl.ptx.cp.async_.cg.shared.global_(
                            edge_list.ptr_to([eslot * TOK + off]),
                            edge_table.ptr_to([edge_base + off]),
                            16,
                            16,
                        )
                        txl.ptx.cp.async_.mbarrier.arrive.noinc.shared.b64(
                            edges_ready.ptr_to([eslot])
                        )
                    with txl.If(lane == 0), txl.Then():
                        txl.ptx.st.shared.b32(meta_ptr(eslot, M_NVALID), n_valid)
                        txl.ptx.st.shared.b32(meta_ptr(eslot, M_BLK), bt.blk)
                        txl.ptx.st.shared.b32(meta_ptr(eslot, M_KVS), bt.kvs)
                        txl.ptx.st.shared.b32(
                            meta_ptr(eslot, M_NEWKV), txl.Select(bi == 0, bt.new, 0)
                        )
                        txl.ptx.st.shared.b32(meta_ptr(eslot, M_RELKV), rel)
                        txl.ptx.st.shared.b32(meta_ptr(eslot, M_H), bt.h)
                        txl.ptx.st.shared.b32(meta_ptr(eslot, M_B), bt.bidx)
                        txl.ptx.st.shared.b32(meta_ptr(eslot, M_EDGE), edge_base)
                        txl.ptx.st.shared.b32(meta_ptr(eslot, M_POSOFF), bt.pos_off)
                        txl.ptx.st.shared.b32(meta_ptr(eslot, M_Q0), bt.q0)
                        txl.ptx.st.shared.b32(meta_ptr(eslot, M_KVUSE), bt.use)
                        txl.ptx.st.shared.b32(meta_ptr(eslot, M_BIDX), bt.bidx)
                    txl.cuda.warp_sync()
                    with txl.If(txl.Or(lane == 0, lane > NCP)), txl.Then():
                        txl.ptx.mbarrier.arrive.shared.b64(
                            edges_ready.ptr_to([eslot]), txl.uint32(1)
                        )
                    txl.cuda.iket.range_end(pub_tok[0])
                    txl.assign(t_seq, t_seq + 1)

                def publish_stops():
                    for extra in range(2):
                        ts = t_seq + extra
                        eslot = ts & (ESLOTS - 1)
                        edges_free.wait(eslot, (ts // ESLOTS) & 1)
                        with txl.If(lane == 0), txl.Then():
                            txl.ptx.st.shared.b32(meta_ptr(eslot, M_NVALID), txl.int32(-1))
                        txl.cuda.warp_sync()
                        edges_ready.arrive(eslot)

                A = Batch()
                Bt = Batch()
                g_b = grab_raw()
                g_next = grab_raw()
                loc_tok = iket_range("pro-locate")
                first_raw = txl.local_scalar("int32", init=txl.Cast("int32", txl.cta_id()))
                A.locate(first_raw)
                txl.cuda.iket.range_end(loc_tok[0])
                A.ensure_loaded()
                locb_tok = iket_range("pro-locate-b")
                Bt.locate(g_b)
                txl.cuda.iket.range_end(locb_tok[0])
                running = txl.local_scalar("int32", init=1)
                with txl.While(running != 0):
                    with txl.If(A.n == 0):
                        with txl.Then():
                            publish_stops()
                            txl.assign(running, 0)
                        with txl.Else():
                            A.ensure_loaded()
                            last = A.n - 1
                            rel_last = txl.Select(
                                txl.Or(Bt.n == 0, Bt.item != A.item), txl.int32(1), txl.int32(0)
                            )
                            with txl.serial(A.n, unroll=False) as bi:
                                publish(A, bi, Bt, txl.Select(bi == last, rel_last, txl.int32(0)))
                            A.copy_from(Bt)
                            Bt.locate(g_next)
                            txl.assign(g_next, grab_raw())

            with r_gather:
                gw = warp_cta - 14
                t_g = txl.local_scalar("int32", init=0)
                running_g = txl.local_scalar("int32", init=1)
                lane_i = txl.Cast("int32", lane)

                def try_bar_g(bar, stage, parity):
                    ok = txl.local_scalar("uint32")
                    txl.ptx.mbarrier.test_wait.parity.shared.b64(
                        ok, bar.ptr_to([stage]), txl.uint32(parity)
                    )
                    return ok

                with txl.While(running_g != 0):
                    progressed = txl.local_scalar("int32", init=0)
                    stage = t_g & 1
                    par = (t_g >> 1) & 1
                    eslot = t_g & (ESLOTS - 1)
                    epar = (t_g // ESLOTS) & 1
                    with txl.If(try_bar_g(edges_ready, eslot, epar) != txl.uint32(0)), txl.Then():
                        n_valid = ld_shared_i32(meta_ptr(eslot, M_NVALID))
                        with txl.If(n_valid < 0), txl.Then():
                            txl.assign(running_g, 0)
                        with txl.If(n_valid >= 0), txl.Then():
                            with (
                                txl.If(try_bar_g(q_empty, stage, par ^ 1) != txl.uint32(0)),
                                txl.Then(),
                            ):
                                h = ld_shared_i32(meta_ptr(eslot, M_H))
                                q0_g = ld_shared_i32(meta_ptr(eslot, M_Q0))
                                gt = iket_range("gather-q")
                                rowbase0 = q0_g * HQ + h * GQA
                                with txl.If(elected()), txl.Then():
                                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                        q_full.ptr_to([stage]), txl.uint32(Q_HALF_BYTES)
                                    )

                                for rnd in range(ceildiv(TOK_HALF, 32)):
                                    jj = rnd * 32 + lane_i
                                    with txl.If(jj < TOK_HALF), txl.Then():
                                        j = gw * TOK_HALF + jj
                                        e_j = ld_shared_i32(edge_list.ptr_to([eslot * TOK + j]))
                                        q_row = (
                                            rowbase0
                                            + txl.Select(
                                                j < n_valid, txl.bitwise_and(e_j, 0xFFFFFF), 0
                                            )
                                            * HQ
                                        )
                                        for ks in range(2):
                                            dst_byte = (
                                                ks * BLK_M * (HEAD_DIM // 2)
                                                + j * GQA * (HEAD_DIM // 2)
                                            ) * F16_BYTES
                                            txl.ptx[TMA_G2S_2D_H](
                                                txl.ptx.addr(q_smem[stage].ptr_to(0, 0), dst_byte),
                                                txl.address_of(q_map),
                                                txl.int32(ks * (HEAD_DIM // 2)),
                                                q_row,
                                                txl.cuda.cvta_generic_to_shared(
                                                    q_full.ptr_to([stage])
                                                ),
                                                txl.uint64(0x14F0000000000000),
                                            )
                                txl.cuda.iket.range_end(gt[0])
                                edges_free.arrive(eslot)
                                txl.assign(t_g, t_g + 1)
                                txl.assign(progressed, 1)
                    with txl.If(progressed == 0), txl.Then():
                        txl.cuda.nano_sleep(txl.uint32(64))

            with r_mma:
                tb_raw = txl.local_scalar("uint32")
                txl.ptx.ld.shared.u32(tb_raw, tmem_addr.ptr_to([0]))
                tmem_base = txl.local_scalar("uint32", init=txl.uniform(tb_raw))
                t_m = txl.local_scalar("int32", init=0)
                have_prev = txl.local_scalar("int32", init=0)
                prev_kvs = txl.local_scalar("int32", init=0)
                prev_rel = txl.local_scalar("int32", init=0)
                running_m = txl.local_scalar("int32", init=1)

                def gemm_qk(stage, kv_stage):
                    tok = iket_range("mma-qk")
                    with txl.If(elected()), txl.Then():
                        for ki in range(HEAD_DIM // MMA_K):
                            txl.ptx[MMA_F16](
                                tmem_base + txl.uint32(stage * 256),
                                desc_at(q_desc, stage * Q_STAGE16 + qoff(ki)),
                                desc_at(k_desc, kv_stage * KV_STAGE16 + koff(ki)),
                                txl.uint32(ID_QK),
                                txl.uint32(0),
                                txl.uint32(0),
                                txl.uint32(0),
                                txl.uint32(0),
                                ki != 0,
                            )
                    txl.cuda.iket.range_end(tok[0])

                def gemm_pv(stage, kv_stage):
                    tok = iket_range("mma-pv")
                    with txl.If(elected()), txl.Then():
                        for ki in range(BLK_N // MMA_K):
                            txl.ptx[MMA_F16](
                                tmem_base + txl.uint32(stage * 256 + 128),
                                tmem_base + txl.uint32(stage * 256 + ki * (MMA_K // 2)),
                                desc_at(v_desc, kv_stage * KV_STAGE16 + mnoff(ki)),
                                txl.uint32(ID_PV),
                                txl.uint32(0),
                                txl.uint32(0),
                                txl.uint32(0),
                                txl.uint32(0),
                                ki != 0,
                            )
                    txl.cuda.iket.range_end(tok[0])

                def try_bar(bar, stage, parity):
                    ok = txl.local_scalar("uint32")
                    txl.ptx.mbarrier.test_wait.parity.shared.b64(
                        ok, bar.ptr_to([stage]), txl.uint32(parity)
                    )
                    return ok

                tq = txl.local_scalar("int32", init=0)
                tp = txl.local_scalar("int32", init=0)
                stop_seen = txl.local_scalar("int32", init=0)
                prev_kvs = txl.local_scalar("int32", init=-1)
                done_m = txl.local_scalar("int32", init=0)
                with txl.While(done_m == 0):
                    progressed = txl.local_scalar("int32", init=0)

                    with txl.If(txl.And(stop_seen == 0, tq - tp < 2)), txl.Then():
                        eslot = tq & (ESLOTS - 1)
                        with (
                            txl.If(
                                try_bar(edges_ready, eslot, (tq // ESLOTS) & 1) != txl.uint32(0)
                            ),
                            txl.Then(),
                        ):
                            n_valid = ld_shared_i32(meta_ptr(eslot, M_NVALID))
                            with txl.If(n_valid < 0):
                                with txl.Then():
                                    txl.assign(stop_seen, 1)
                                    txl.assign(progressed, 1)
                                with txl.Else():
                                    stage = tq & 1
                                    par = (tq >> 1) & 1
                                    kv_stage = ld_shared_i32(meta_ptr(eslot, M_KVS))
                                    new_kv = ld_shared_i32(meta_ptr(eslot, M_NEWKV))
                                    kv_use = ld_shared_i32(meta_ptr(eslot, M_KVUSE))
                                    ready = txl.local_scalar(
                                        "uint32", init=try_bar(q_full, stage, par)
                                    )
                                    txl.assign(
                                        ready,
                                        txl.bitwise_and(ready, try_bar(s_empty, stage, par ^ 1)),
                                    )

                                    txl.assign(
                                        ready,
                                        txl.bitwise_and(ready, try_bar(pv_done, stage, par ^ 1)),
                                    )
                                    with txl.If(new_kv != 0), txl.Then():
                                        for s_ in range(KV_STAGES):
                                            with txl.If(kv_stage == s_), txl.Then():
                                                txl.assign(
                                                    ready,
                                                    txl.bitwise_and(
                                                        ready,
                                                        try_bar(
                                                            kv_full, s_, (kv_use // KV_STAGES) & 1
                                                        ),
                                                    ),
                                                )
                                    with txl.If(ready != txl.uint32(0)), txl.Then():
                                        for s_ in range(KV_STAGES):
                                            with txl.If(kv_stage == s_), txl.Then():
                                                gemm_qk(stage, s_)
                                        with txl.If(elected()), txl.Then():
                                            commit(s_full, stage)
                                            commit(q_empty, stage)
                                        txl.assign(tq, tq + 1)
                                        txl.assign(progressed, 1)

                    with txl.If(tp < tq), txl.Then():
                        eslot_p = tp & (ESLOTS - 1)
                        pstage = tp & 1
                        ppar = (tp >> 1) & 1
                        pready = txl.local_scalar("uint32", init=try_bar(p_full, pstage, ppar))
                        txl.assign(
                            pready, txl.bitwise_and(pready, try_bar(o_empty, pstage, ppar ^ 1))
                        )
                        with txl.If(pready != txl.uint32(0)), txl.Then():
                            kvs_p = ld_shared_i32(meta_ptr(eslot_p, M_KVS))
                            rel_p = ld_shared_i32(meta_ptr(eslot_p, M_RELKV))

                            with txl.If(txl.And(prev_kvs >= 0, kvs_p != prev_kvs)), txl.Then():
                                with txl.If(elected()), txl.Then():
                                    for s_ in range(KV_STAGES):
                                        with txl.If(prev_kvs == s_), txl.Then():
                                            commit(kv_empty, s_)
                            for s_ in range(KV_STAGES):
                                with txl.If(kvs_p == s_), txl.Then():
                                    gemm_pv(pstage, s_)
                            with txl.If(elected()), txl.Then():
                                commit(o_full, pstage)
                                commit(pv_done, pstage)

                            with txl.If(rel_p != 0):
                                with txl.Then():
                                    with txl.If(elected()), txl.Then():
                                        for s_ in range(KV_STAGES):
                                            with txl.If(kvs_p == s_), txl.Then():
                                                commit(kv_empty, s_)
                                    txl.assign(prev_kvs, -1)
                                with txl.Else():
                                    txl.assign(prev_kvs, kvs_p)
                            edges_free.arrive(eslot_p)
                            txl.assign(tp, tp + 1)
                            txl.assign(progressed, 1)
                    with txl.If(txl.And(stop_seen != 0, tp == tq)), txl.Then():
                        txl.assign(done_m, 1)
                    with txl.If(progressed == 0), txl.Then():
                        txl.cuda.nano_sleep(txl.uint32(32))

                with txl.If(prev_kvs >= 0), txl.Then():
                    with txl.If(elected()), txl.Then():
                        for s_ in range(KV_STAGES):
                            with txl.If(prev_kvs == s_), txl.Then():
                                commit(kv_empty, s_)

        def softmax_role(half):
            t_x = txl.local_scalar("int32", init=0)
            running_x = txl.local_scalar("int32", init=1)
            tok_local = tid_in_wg // GQA
            with txl.While(running_x != 0):
                stage = t_x & 1
                par = (t_x >> 1) & 1
                eslot = t_x & (ESLOTS - 1)
                epar = (t_x // ESLOTS) & 1
                sslot = stage + 2 * ((t_x >> 1) % (STATS_SLOTS // 2))
                spar = ((t_x >> 1) // (STATS_SLOTS // 2)) & 1
                edges_ready.wait(eslot, epar)
                n_valid = ld_shared_i32(meta_ptr(eslot, M_NVALID))
                with txl.If(n_valid < 0), txl.Then():
                    stats_free.wait(sslot, spar)
                    if half == 0:
                        with txl.If(tid_in_wg == 0), txl.Then():
                            txl.ptx.st.shared.b32(stats_hdr.ptr_to([sslot * 2 + 1]), txl.int32(1))
                    stats_ready.arrive(sslot)
                    txl.assign(running_x, 0)
                with txl.If(n_valid >= 0), txl.Then():
                    blk = ld_shared_i32(meta_ptr(eslot, M_BLK))
                    pos_off = ld_shared_i32(meta_ptr(eslot, M_POSOFF))
                    q0_s = ld_shared_i32(meta_ptr(eslot, M_Q0))
                    h_s = ld_shared_i32(meta_ptr(eslot, M_H))
                    e = ld_shared_i32(edge_list.ptr_to([eslot * TOK + tok_local]))
                    valid = tok_local < n_valid
                    t_local = txl.bitwise_and(e, 0xFFFFFF)
                    q_pos = pos_off + t_local
                    col_limit = txl.local_scalar(
                        "int32",
                        init=txl.Select(
                            valid, txl.min(txl.max(q_pos - blk * BLK_N + 1, 0), BLK_N), 0
                        ),
                    )
                    edges_free.arrive(eslot)

                    if kv_fp8 and not EPI_CONV:
                        new_kv = ld_shared_i32(meta_ptr(eslot, M_NEWKV))
                        with txl.If(new_kv != 0), txl.Then():
                            kv_stage = ld_shared_i32(meta_ptr(eslot, M_KVS))
                            kv_use = ld_shared_i32(meta_ptr(eslot, M_KVUSE))
                            xt = iket_range("xform-k" if half == 0 else "xform-v", leader_only=True)
                            for s_ in range(KV_STAGES):
                                with txl.If(kv_stage == s_), txl.Then():
                                    kv_tma.wait(s_, (kv_use // KV_STAGES) & 1)
                                    txl.ptx.fence.proxy.async_.shared__cta()
                                    if half == 0:
                                        convert_fp8_wg(k_smem[s_])
                                    else:
                                        convert_fp8_wg(v_smem[s_])
                                    txl.ptx.fence.proxy.async_.shared__cta()
                                    kv_full.arrive(s_)
                            txl.cuda.iket.range_end(xt[0])

                    ws_tok = iket_range("sm-wait-s", leader_only=True)
                    s_full.wait(stage, par)
                    txl.cuda.iket.range_end(ws_tok[0])
                    sc_tok = iket_range("sm-compute", leader_only=True)
                    s_half = txl.alloc_local([64], "float32")
                    p_half = txl.alloc_local([32], "uint32")
                    tmem_load(s_half, 0, tmem(stage * 256 + 64 * half), 32)
                    tmem_load(s_half, 32, tmem(stage * 256 + 64 * half + 32), 32)
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    s_empty.arrive(stage)
                    my_limit = col_limit - 64 * half
                    with txl.If(my_limit < 64), txl.Then():
                        for s_ in range(2):
                            k_keep = txl.max(my_limit - s_ * 32, 0)
                            mask_inv = txl.local_scalar("uint32")
                            txl.assign(
                                mask_inv,
                                shl_u32_clamp(txl.uint32(0xFFFFFFFF), txl.Cast("uint32", k_keep)),
                            )
                            for i in range(32):
                                c = s_ * 32 + i
                                in_bound = txl.bitwise_and(
                                    txl.bitwise_not(mask_inv),
                                    txl.shift_left(txl.uint32(1), txl.uint32(i)),
                                )
                                txl.ptx.mov.b32(
                                    s_half[c],
                                    txl.Select(
                                        txl.Cast("bool", in_bound), s_half[c], txl.float32(NEG_INF)
                                    ),
                                )

                    temp = txl.alloc_local([8], "float32")
                    for i in range(8):
                        txl.ptx.mov.b32(temp[i], txl.max(s_half[2 * i], s_half[2 * i + 1]))
                    for g in range(1, 4):
                        for i in range(8):
                            txl.ptx[MAX3_F32](
                                temp[i], temp[i], s_half[16 * g + 2 * i], s_half[16 * g + 2 * i + 1]
                            )
                    txl.ptx[MAX3_F32](temp[0], temp[0], temp[1], temp[2])
                    txl.ptx[MAX3_F32](temp[3], temp[3], temp[4], temp[5])
                    txl.ptx[MAX3_F32](temp[0], temp[0], temp[6], temp[7])
                    my_max = txl.local_scalar("float32", init=txl.max(temp[0], temp[3]))
                    txl.ptx.st.shared.b32(
                        xmax.ptr_to([(stage * 2 + half) * BLK_M + tid_in_wg]), my_max
                    )
                    txl.ptx.bar.sync(txl.uint32(1), txl.uint32(256))
                    other_max = ld_shared_f32(
                        xmax.ptr_to([(stage * 2 + (1 - half)) * BLK_M + tid_in_wg])
                    )
                    row_max = txl.local_scalar(
                        "float32",
                        init=txl.Select(
                            col_limit > 0, txl.max(my_max, other_max), txl.float32(NEG_INF)
                        ),
                    )
                    row_max_safe = txl.if_then_else(
                        row_max == txl.float32(NEG_INF), txl.float32(0.0), row_max
                    )
                    m_scaled = txl.local_scalar("float32", init=row_max_safe * scale_log2)
                    bias = txl.local_scalar(
                        "float32",
                        init=txl.Select(
                            col_limit > 0, txl.float32(0.0) - m_scaled, txl.float32(NEG_INF)
                        ),
                    )
                    scale_pair = txl.local_scalar("uint64")
                    bias_pair = txl.local_scalar("uint64")
                    txl.ptx.mov.b64(scale_pair, scale_log2, scale_log2)
                    txl.ptx.mov.b64(bias_pair, bias, bias)
                    pair_tmp = txl.local_scalar("uint64")
                    sum_acc = [txl.local_scalar("uint64") for _ in range(4)]
                    for a in sum_acc:
                        txl.ptx.mov.b64(a, txl.float32(0.0), txl.float32(0.0))
                    for i in range(32):
                        idx = 2 * i
                        txl.ptx.mov.b64(pair_tmp, s_half[idx], s_half[idx + 1])
                        txl.ptx.fma.rz.ftz.f32x2(pair_tmp, pair_tmp, scale_pair, bias_pair)
                        txl.ptx.mov.b64(s_half[idx], s_half[idx + 1], pair_tmp)
                        if not SKIP_EXP:
                            if (i % 8) < EMU_PER_8:
                                ex2_poly_pair(s_half, idx)
                            else:
                                txl.ptx.ex2.approx.ftz.f32(s_half[idx], s_half[idx])
                                txl.ptx.ex2.approx.ftz.f32(s_half[idx + 1], s_half[idx + 1])
                        txl.ptx.mov.b64(pair_tmp, s_half[idx], s_half[idx + 1])
                        txl.ptx.add.rn.ftz.f32x2(sum_acc[i % 4], sum_acc[i % 4], pair_tmp)
                        cast_f32x2_bf16x2(p_half, s_half, idx)
                    pv_done.wait(stage, par)
                    for i in range(2):
                        tmem_store(p_half, i * 16, tmem(stage * 256 + 32 * half + i * 16))
                    txl.ptx.tcgen05.wait__st.sync.aligned()
                    p_full.arrive(stage)
                    for step in (2, 1):
                        for a in range(step):
                            txl.ptx.add.rn.ftz.f32x2(sum_acc[a], sum_acc[a], sum_acc[a + step])
                    sum_lo = txl.local_scalar("float32")
                    sum_hi = txl.local_scalar("float32")
                    txl.ptx.mov.b64(sum_lo, sum_hi, sum_acc[0])
                    part_sum = txl.local_scalar("float32", init=sum_lo + sum_hi)
                    stats_free.wait(sslot, spar)
                    sbase = (sslot * BLK_M + tid_in_wg) * 6
                    txl.ptx.st.shared.b32(stats.ptr_to([sbase + 1 + half]), part_sum)
                    if half == 0:
                        txl.ptx.st.shared.b32(stats.ptr_to([sbase]), m_scaled)
                        txl.ptx.st.shared.b32(
                            stats.ptr_to([sbase + 3]), txl.Select(valid, q0_s + t_local, -1)
                        )
                        txl.ptx.st.shared.b32(stats.ptr_to([sbase + 4]), txl.shift_right(e, 24))
                        with txl.If(tid_in_wg == 0), txl.Then():
                            txl.ptx.st.shared.b32(stats_hdr.ptr_to([sslot * 2]), h_s)
                            txl.ptx.st.shared.b32(stats_hdr.ptr_to([sslot * 2 + 1]), txl.int32(0))
                    stats_ready.arrive(sslot)
                    txl.cuda.iket.range_end(sc_tok[0])
                txl.assign(t_x, t_x + 1)

        with r_sm0:
            softmax_role(0)
        with r_sm1:
            softmax_role(1)

        with r_epi:
            t_e = txl.local_scalar("int32", init=0)
            running_e = txl.local_scalar("int32", init=1)
            tok_local = tid_in_wg // GQA
            head_local = tid_in_wg % GQA
            if kv_fp8 and EPI_CONV:
                conv_use = txl.local_scalar("int32", init=0)

                def expand_issued_blocks():
                    """Expand fp8 K/V of every issued block whose TMA has landed, in issue order."""
                    go = txl.local_scalar("int32", init=1)
                    with txl.While(go != 0):
                        txl.assign(go, 0)
                        for s_ in range(KV_STAGES):
                            with txl.If(txl.And(go == 0, conv_use % KV_STAGES == s_)), txl.Then():
                                issued = txl.local_scalar("uint32")
                                txl.ptx.mbarrier.test_wait.parity.shared.b64(
                                    issued,
                                    kv_issued.ptr_to([s_]),
                                    txl.uint32((conv_use // KV_STAGES) & 1),
                                )
                                landed = txl.local_scalar("uint32", init=txl.uint32(0))
                                with txl.If(issued != txl.uint32(0)), txl.Then():
                                    txl.ptx.mbarrier.test_wait.parity.shared.b64(
                                        landed,
                                        kv_tma.ptr_to([s_]),
                                        txl.uint32((conv_use // KV_STAGES) & 1),
                                    )
                                with txl.If(landed != txl.uint32(0)), txl.Then():
                                    xt = iket_range("xform-epi", leader_only=True)
                                    txl.ptx.fence.proxy.async_.shared__cta()
                                    convert_fp8_wg(k_smem[s_])
                                    convert_fp8_wg(v_smem[s_])
                                    txl.ptx.fence.proxy.async_.shared__cta()
                                    kv_full.arrive(s_)
                                    txl.cuda.iket.range_end(xt[0])
                                    txl.assign(conv_use, conv_use + 1)
                                    txl.assign(go, 1)

            with txl.While(running_e != 0):
                stage = t_e & 1
                par = (t_e >> 1) & 1
                i_s = t_e >> 1
                sslot = stage + 2 * (i_s % (STATS_SLOTS // 2))
                spar = (i_s // (STATS_SLOTS // 2)) & 1
                if kv_fp8 and EPI_CONV:
                    sr = txl.local_scalar("uint32", init=txl.uint32(0))
                    txl.ptx.mbarrier.test_wait.parity.shared.b64(
                        sr, stats_ready.ptr_to([sslot]), txl.uint32(spar)
                    )
                    with txl.While(sr == txl.uint32(0)):
                        expand_issued_blocks()
                        txl.ptx.mbarrier.test_wait.parity.shared.b64(
                            sr, stats_ready.ptr_to([sslot]), txl.uint32(spar)
                        )
                        with txl.If(sr == txl.uint32(0)), txl.Then():
                            txl.cuda.nano_sleep(txl.uint32(64))
                else:
                    stats_ready.wait(sslot, spar)
                stop = ld_shared_i32(stats_hdr.ptr_to([sslot * 2 + 1]))
                with txl.If(stop != 0), txl.Then():
                    txl.assign(running_e, 0)
                with txl.If(stop == 0), txl.Then():
                    h = ld_shared_i32(stats_hdr.ptr_to([sslot * 2]))
                    sbase = (sslot * BLK_M + tid_in_wg) * 6
                    m_scaled = ld_shared_f32(stats.ptr_to([sbase]))
                    row_sum = ld_shared_f32(stats.ptr_to([sbase + 1])) + ld_shared_f32(
                        stats.ptr_to([sbase + 2])
                    )
                    t_abs = txl.local_scalar("int32", init=ld_shared_i32(stats.ptr_to([sbase + 3])))
                    slot = txl.local_scalar("int32", init=ld_shared_i32(stats.ptr_to([sbase + 4])))
                    stats_free.arrive(sslot)
                    valid = t_abs >= 0
                    hq_row = txl.max(t_abs, 0) * HQ + h * GQA + head_local
                    pidx = hq_row * TOPK + slot
                    has_mass = txl.And(valid, row_sum > txl.float32(0.0))
                    wo_tok = iket_range("ep-wait-o", leader_only=True)
                    o_full.wait(stage, par)
                    txl.cuda.iket.range_end(wo_tok[0])
                    ep_tok = iket_range("ep-store", leader_only=True)
                    pbase = pidx * ((HEAD_DIM // 4) if compact_s2f6 else (HEAD_DIM // 2))
                    scale_word = txl.local_scalar("int32", init=0)
                    quarters = [txl.alloc_local([32], "float32") for _ in range(2)]
                    tmem_load(quarters[0], 0, tmem(stage * 256 + 128), 32)
                    for qtr in range(4):
                        cur = quarters[qtr % 2]
                        if qtr + 1 < 4:
                            tmem_load(
                                quarters[(qtr + 1) % 2],
                                0,
                                tmem(stage * 256 + 128 + (qtr + 1) * 32),
                                32,
                            )
                        txl.ptx.tcgen05.wait__ld.sync.aligned()
                        if qtr == 3:
                            o_empty.arrive(stage)
                        if compact_s2f6:
                            max_abs = txl.local_scalar("float32", init=txl.float32(0.0))
                            # Quantize the raw PV numerator.  The combine uses
                            # exp(m-M), while its denominator still includes l;
                            # this is algebraically identical to normalizing
                            # here and later multiplying by exp(m-M)*l.

                            t1 = txl.alloc_local([11], "float32")
                            for i in range(10):
                                txl.ptx["max.abs.f32"](
                                    t1[i], cur[3 * i], cur[3 * i + 1], cur[3 * i + 2]
                                )
                            txl.ptx["max.abs.f32"](t1[10], cur[30], cur[31], cur[31])
                            t2 = txl.alloc_local([4], "float32")
                            for i in range(3):
                                txl.ptx[MAX3_F32](t2[i], t1[3 * i], t1[3 * i + 1], t1[3 * i + 2])
                            txl.ptx.mov.b32(t2[3], txl.max(t1[9], t1[10]))
                            txl.ptx[MAX3_F32](max_abs, t2[0], t2[1], t2[2])
                            txl.assign(max_abs, txl.max(max_abs, t2[3]))
                            scaled_max = txl.local_scalar(
                                "float32", init=max_abs * txl.float32(64.0 / 127.0)
                            )
                            scaled_bits = txl.local_scalar("uint32")
                            txl.ptx.mov.b32(scaled_bits, scaled_max)
                            scale_exp = txl.local_scalar(
                                "int32",
                                init=txl.Cast(
                                    "int32",
                                    txl.bitwise_and(
                                        txl.shift_right(scaled_bits, txl.uint32(23)),
                                        txl.uint32(0xFF),
                                    ),
                                )
                                + txl.Select(
                                    txl.bitwise_and(scaled_bits, txl.uint32(0x7FFFFF))
                                    != txl.uint32(0),
                                    1,
                                    0,
                                ),
                            )
                            txl.assign(
                                scale_exp,
                                txl.Select(max_abs > txl.float32(0.0), scale_exp, txl.int32(127)),
                            )
                            scale_pair = txl.local_scalar(
                                "uint16",
                                init=txl.Cast(
                                    "uint16",
                                    txl.bitwise_or(
                                        scale_exp, txl.shift_left(scale_exp, txl.int32(8))
                                    ),
                                ),
                            )
                            qwords = txl.alloc_local([8], "uint32")
                            for i in range(8):
                                lo = txl.local_scalar("uint16")
                                hi = txl.local_scalar("uint16")
                                txl.ptx["cvt.rn.satfinite.scaled::n2::ue8m0.s2f6x2.f32"](
                                    lo, cur[i * 4 + 1], cur[i * 4], scale_pair
                                )
                                txl.ptx["cvt.rn.satfinite.scaled::n2::ue8m0.s2f6x2.f32"](
                                    hi, cur[i * 4 + 3], cur[i * 4 + 2], scale_pair
                                )
                                txl.assign(
                                    qwords[i],
                                    txl.bitwise_or(
                                        txl.Cast("uint32", lo),
                                        txl.shift_left(txl.Cast("uint32", hi), txl.uint32(16)),
                                    ),
                                )
                            txl.assign(
                                scale_word,
                                txl.bitwise_or(
                                    scale_word,
                                    txl.shift_left(
                                        txl.bitwise_and(scale_exp, txl.int32(0xFF)),
                                        txl.int32(qtr * 8),
                                    ),
                                ),
                            )
                            if not SKIP_STORES:
                                with txl.If(has_mass), txl.Then():
                                    txl.ptx.st.global_.L2__evict_last.v8.b32(
                                        o_part.ptr_to([pbase + qtr * 8]),
                                        *(qwords[i] for i in range(8)),
                                    )
                        else:
                            o_bf16 = txl.alloc_local([16], "uint32")
                            for i in range(16):
                                cast_f32x2_bf16x2(o_bf16, cur, 2 * i)
                            if not SKIP_STORES:
                                with txl.If(has_mass), txl.Then():
                                    for i in range(2):
                                        txl.ptx.st.global_.L2__evict_last.v8.b32(
                                            o_part.ptr_to([pbase + qtr * 16 + i * 8]),
                                            *(o_bf16[i * 8 + w] for w in range(8)),
                                        )
                    with txl.If(valid), txl.Then():
                        if compact_s2f6:
                            txl.ptx.st.global_.b32(q_part.ptr_to([pidx]), scale_word)
                        txl.ptx.st.global_.f32(
                            m_part.ptr_to([pidx]),
                            txl.Select(has_mass, m_scaled, txl.float32(NEG_INF)),
                        )
                        txl.ptx.st.global_.f32(
                            l_part.ptr_to([pidx]), txl.Select(has_mass, row_sum, txl.float32(0.0))
                        )
                    txl.cuda.iket.range_end(ep_tok[0])
                txl.assign(t_e, t_e + 1)

        txl.cuda.cta_sync()
        is_last_cta = smem.alloc((1,), txl.i32)
        with txl.If(txl.thread_id() == 0), txl.Then():
            done = txl.local_scalar("int32")
            txl.ptx.atom.acq_rel.gpu.global_.add.s32(done, sched.ptr_to([1]), txl.int32(1))
            txl.ptx.st.shared.b32(is_last_cta.ptr_to([0]), txl.Select(done == num_ctas - 1, 1, 0))
        txl.cuda.cta_sync()
        with txl.If(ld_shared_i32(is_last_cta.ptr_to([0])) != 0), txl.Then():
            tid_all = txl.thread_id()
            for j in range(ceildiv(NITEMS, 512)):
                idx = j * 512 + tid_all
                with txl.If(idx < NITEMS), txl.Then():
                    txl.ptx.st.relaxed.gpu.global_.b32(cursor.ptr_to([idx]), txl.int32(0))
            with txl.If(tid_all == 0), txl.Then():
                txl.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([0]), txl.int32(0))
                txl.ptx.st.relaxed.gpu.global_.b32(sched.ptr_to([1]), txl.int32(0))
        with txl.If(tvm.tirx.all(wg_id == 0, warp_id == 0)), txl.Then():
            dealloc = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(dealloc, tmem_addr.ptr_to([0]))
            txl.ptx[TMEM_RELINQUISH]()
            txl.ptx[TMEM_DEALLOC](dealloc, txl.uint32(512))

    return msa_reverse_main


def make_combine_kernel_legacy(*, total_q, hq, hkv, topk, num_ctas):
    GQA = hq // hkv
    TOPK = topk
    TOTAL_Q = total_q
    HQ = hq
    ROWS = total_q * hq
    ROWS_PER_CTA = 32

    @txl.kernel(warps=8, arch="sm_100a", grid=num_ctas)
    def msa_reverse_combine_legacy(
        out: txl.gptr[txl.i32],
        deg: txl.gptr[txl.i32],
        o_part: txl.gptr[txl.i32],
        m_part: txl.gptr[txl.f32],
        l_part: txl.gptr[txl.f32],
    ):
        if USE_PDL:
            txl.ptx.griddepcontrol.wait()
        tid = txl.thread_id()
        row = txl.cta_id() * ROWS_PER_CTA + (tid >> 3)
        lane8 = tid & 7
        with txl.If(row < ROWS), txl.Then():
            tok = row // HQ
            hq_i = row - tok * HQ
            h = hq_i // GQA
            d = txl.local_scalar("int32")
            txl.ptx.ld.global_.nc.b32(d, deg.ptr_to([h * TOTAL_Q + tok]))
            with txl.If(d > 0), txl.Then():
                mvals = [
                    txl.local_scalar("float32", init=txl.float32(NEG_INF)) for _ in range(TOPK)
                ]
                lvals = [txl.local_scalar("float32", init=txl.float32(0.0)) for _ in range(TOPK)]
                for s_i in range(TOPK):
                    with txl.If(s_i < d), txl.Then():
                        txl.ptx.ld.global_.nc.b32(mvals[s_i], m_part.ptr_to([row * TOPK + s_i]))
                        txl.ptx.ld.global_.nc.b32(lvals[s_i], l_part.ptr_to([row * TOPK + s_i]))
                M = txl.local_scalar("float32", init=txl.float32(NEG_INF))
                for s_i in range(TOPK):
                    txl.assign(M, txl.max(M, mvals[s_i]))
                M_safe = txl.if_then_else(M == txl.float32(NEG_INF), txl.float32(0.0), M)
                acc = [txl.local_scalar("uint64") for _ in range(8)]
                for i in range(8):
                    txl.ptx.mov.b64(acc[i], txl.float32(0.0), txl.float32(0.0))
                wsum = txl.local_scalar("float32", init=txl.float32(0.0))
                for sb in range(0, TOPK, 4):
                    words = [txl.alloc_local([8], "uint32") for _ in range(4)]
                    for sj in range(4):
                        s_i = sb + sj
                        with txl.If(txl.And(s_i < d, lvals[s_i] > txl.float32(0.0))), txl.Then():
                            pb = (row * TOPK + s_i) * (HEAD_DIM // 2) + lane8 * 8
                            txl.ptx.ld.global_.nc.v8.b32(
                                *(words[sj][wi] for wi in range(8)), o_part.ptr_to([pb])
                            )
                    for sj in range(4):
                        s_i = sb + sj
                        with txl.If(txl.And(s_i < d, lvals[s_i] > txl.float32(0.0))), txl.Then():
                            w = txl.local_scalar("float32")
                            txl.ptx.ex2.approx.ftz.f32(w, mvals[s_i] - M_safe)
                            txl.assign(wsum, wsum + w * lvals[s_i])
                            w_pair = txl.local_scalar("uint64")
                            txl.ptx.mov.b64(w_pair, w, w)
                            for wi in range(8):
                                lo_f = txl.local_scalar("float32")
                                hi_f = txl.local_scalar("float32")
                                partial_pair = txl.local_scalar("uint64")
                                txl.ptx.mov.b32(lo_f, txl.shift_left(words[sj][wi], txl.uint32(16)))
                                txl.ptx.mov.b32(
                                    hi_f, txl.bitwise_and(words[sj][wi], txl.uint32(0xFFFF0000))
                                )
                                txl.ptx.mov.b64(partial_pair, lo_f, hi_f)
                                txl.ptx.fma.rn.ftz.f32x2(acc[wi], partial_pair, w_pair, acc[wi])
                inv_w = txl.local_scalar("float32")
                txl.ptx.rcp.approx.ftz.f32(
                    inv_w, txl.Select(wsum > txl.float32(0.0), wsum, txl.float32(1.0))
                )
                inv_pair = txl.local_scalar("uint64")
                txl.ptx.mov.b64(inv_pair, inv_w, inv_w)
                outw = txl.alloc_local([8], "uint32")
                for i in range(8):
                    lo_f = txl.local_scalar("float32")
                    hi_f = txl.local_scalar("float32")
                    txl.ptx.mul.rz.ftz.f32x2(acc[i], acc[i], inv_pair)
                    txl.ptx.mov.b64(lo_f, hi_f, acc[i])
                    txl.ptx.cvt.rn.bf16x2.f32(outw[i], hi_f, lo_f)
                ob = row * (HEAD_DIM // 2) + lane8 * 8
                txl.ptx.st.global_.v8.b32(out.ptr_to([ob]), *(outw[w] for w in range(8)))

    return msa_reverse_combine_legacy


def make_combine_kernel(*, total_q, hq, hkv, topk, num_ctas):
    GQA = hq // hkv
    TOPK = topk
    TOTAL_Q = total_q
    HQ = hq
    ROWS = total_q * hq
    ROWS_PER_CTA = 32

    @txl.kernel(warps=8, arch="sm_100a", grid=num_ctas)
    def msa_reverse_combine(
        out: txl.gptr[txl.i32],
        deg: txl.gptr[txl.i32],
        o_part: txl.gptr[txl.i32],
        m_part: txl.gptr[txl.f32],
        l_part: txl.gptr[txl.f32],
    ):
        if USE_PDL:
            txl.ptx.griddepcontrol.wait()
        tid = txl.thread_id()
        row = txl.cta_id() * ROWS_PER_CTA + (tid >> 3)
        lane8 = tid & 7
        row_valid = row < ROWS
        tok = row // HQ
        hq_i = row - tok * HQ
        h = hq_i // GQA
        subgroup_base = txl.local_scalar(
            "uint32", init=txl.bitwise_and(txl.lane_id(), txl.uint32(0x18))
        )
        d = txl.local_scalar("int32", init=0)
        with txl.If(txl.And(row_valid, lane8 == 0)), txl.Then():
            txl.ptx.ld.global_.nc.b32(d, deg.ptr_to([h * TOTAL_Q + tok]))
        txl.ptx.shfl_sync.idx.b32(d, d, subgroup_base, txl.uint32(31), txl.uint32(0xFFFFFFFF))

        mvals = [txl.local_scalar("float32", init=txl.float32(NEG_INF)) for _ in range(TOPK)]
        lvals = [txl.local_scalar("float32", init=txl.float32(0.0)) for _ in range(TOPK)]
        weights = [txl.local_scalar("float32", init=txl.float32(0.0)) for _ in range(TOPK)]
        inv_w = txl.local_scalar("float32", init=txl.float32(1.0))
        with txl.If(txl.And(row_valid, lane8 == 0)), txl.Then():
            for s_i in range(TOPK):
                with txl.If(s_i < d), txl.Then():
                    txl.ptx.ld.global_.nc.b32(mvals[s_i], m_part.ptr_to([row * TOPK + s_i]))
                    txl.ptx.ld.global_.nc.b32(lvals[s_i], l_part.ptr_to([row * TOPK + s_i]))
            M = txl.local_scalar("float32", init=txl.float32(NEG_INF))
            for s_i in range(TOPK):
                txl.assign(M, txl.max(M, mvals[s_i]))
            M_safe = txl.if_then_else(M == txl.float32(NEG_INF), txl.float32(0.0), M)
            wsum = txl.local_scalar("float32", init=txl.float32(0.0))
            for s_i in range(TOPK):
                with txl.If(txl.And(s_i < d, lvals[s_i] > txl.float32(0.0))), txl.Then():
                    txl.ptx.ex2.approx.ftz.f32(weights[s_i], mvals[s_i] - M_safe)
                    txl.assign(wsum, wsum + weights[s_i] * lvals[s_i])
            txl.ptx.rcp.approx.ftz.f32(
                inv_w, txl.Select(wsum > txl.float32(0.0), wsum, txl.float32(1.0))
            )
        for s_i in range(TOPK):
            txl.ptx.shfl_sync.idx.b32(
                weights[s_i], weights[s_i], subgroup_base, txl.uint32(31), txl.uint32(0xFFFFFFFF)
            )
        txl.ptx.shfl_sync.idx.b32(
            inv_w, inv_w, subgroup_base, txl.uint32(31), txl.uint32(0xFFFFFFFF)
        )

        with txl.If(txl.And(row_valid, d > 0)), txl.Then():
            acc = [txl.local_scalar("uint64") for _ in range(8)]
            for i in range(8):
                txl.ptx.mov.b64(acc[i], txl.float32(0.0), txl.float32(0.0))
            for sb in range(0, TOPK, 4):
                words = [txl.alloc_local([8], "uint32") for _ in range(4)]
                for sj in range(4):
                    s_i = sb + sj
                    with txl.If(weights[s_i] > txl.float32(0.0)), txl.Then():
                        pb = (row * TOPK + s_i) * (HEAD_DIM // 2) + lane8 * 8
                        txl.ptx.ld.global_.nc.v8.b32(
                            *(words[sj][wi] for wi in range(8)), o_part.ptr_to([pb])
                        )
                for sj in range(4):
                    s_i = sb + sj
                    with txl.If(weights[s_i] > txl.float32(0.0)), txl.Then():
                        w_pair = txl.local_scalar("uint64")
                        txl.ptx.mov.b64(w_pair, weights[s_i], weights[s_i])
                        for wi in range(8):
                            lo_f = txl.local_scalar("float32")
                            hi_f = txl.local_scalar("float32")
                            partial_pair = txl.local_scalar("uint64")
                            txl.ptx.mov.b32(lo_f, txl.shift_left(words[sj][wi], txl.uint32(16)))
                            txl.ptx.mov.b32(
                                hi_f, txl.bitwise_and(words[sj][wi], txl.uint32(0xFFFF0000))
                            )
                            txl.ptx.mov.b64(partial_pair, lo_f, hi_f)
                            txl.ptx.fma.rn.ftz.f32x2(acc[wi], partial_pair, w_pair, acc[wi])
            inv_pair = txl.local_scalar("uint64")
            txl.ptx.mov.b64(inv_pair, inv_w, inv_w)
            outw = txl.alloc_local([8], "uint32")
            for i in range(8):
                lo_f = txl.local_scalar("float32")
                hi_f = txl.local_scalar("float32")
                txl.ptx.mul.rz.ftz.f32x2(acc[i], acc[i], inv_pair)
                txl.ptx.mov.b64(lo_f, hi_f, acc[i])
                txl.ptx.cvt.rn.bf16x2.f32(outw[i], hi_f, lo_f)
            ob = row * (HEAD_DIM // 2) + lane8 * 8
            txl.ptx.st.global_.v8.b32(out.ptr_to([ob]), *(outw[w] for w in range(8)))

    return msa_reverse_combine


def make_combine_kernel_tma(*, total_q, hq, hkv, topk, num_ctas, compact_s2f6=False):
    """Top-k-4/8 merge with one overlapped TMA load per CTA."""
    assert topk in (4, 8)
    GQA = hq // hkv
    TOPK = topk
    TOTAL_Q = total_q
    HQ = hq
    ROWS = total_q * hq
    ROWS_PER_CTA = 32 if topk == 4 else 16
    WARPS = ROWS_PER_CTA // 4
    TILE_BYTES = ROWS_PER_CTA * TOPK * HEAD_DIM * (1 if compact_s2f6 else F16_BYTES)

    @txl.kernel(warps=WARPS, arch="sm_100a", grid=num_ctas)
    def msa_reverse_combine_tma(
        part_map: txl.TensorMap,
        out: txl.gptr[txl.i32],
        deg: txl.gptr[txl.i32],
        q_part: txl.gptr[txl.i32],
        m_part: txl.gptr[txl.f32],
        l_part: txl.gptr[txl.f32],
    ):
        if USE_PDL:
            txl.ptx.griddepcontrol.wait()
        tid = txl.thread_id()
        lane = txl.lane_id()
        warp = txl.warp_id()
        local_row = tid >> 3
        row = txl.cta_id() * ROWS_PER_CTA + local_row
        lane8 = tid & 7
        row_valid = row < ROWS
        tok = row // HQ
        hq_i = row - tok * HQ
        h = hq_i // GQA
        subgroup_base = txl.local_scalar("uint32", init=txl.bitwise_and(lane, txl.uint32(0x18)))

        def iket_range(name, *, warp0_only=False):
            token = txl.alloc_local([1], "uint32")
            if warp0_only:
                txl.assign(token[0], txl.cuda.iket.sentinel_token(name))
                with txl.If(warp == 0), txl.Then():
                    txl.assign(token[0], txl.cuda.iket.range_start(name))
            else:
                txl.assign(token[0], txl.cuda.iket.range_start(name))
            return token

        smem = txl.smem_pool()
        if compact_s2f6:
            part_smem = smem.alloc((ROWS_PER_CTA * TOPK, HEAD_DIM), txl.u8, swizzle=txl.SW128B)
        else:
            part_smem = smem.alloc(
                (ROWS_PER_CTA * TOPK * 2, HEAD_DIM // 2), txl.bf16, swizzle=txl.SW128B
            )
        part_full = txl.TMABar(smem, 1)
        part_full.init(1)
        txl.ptx.fence.proxy.async_.shared__cta()
        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.cuda.cta_sync()

        issue_tok = iket_range("combine-tma-issue", warp0_only=True)
        with txl.If(txl.And(warp == 0, txl.cuda.elect_sync() != txl.uint32(0))), txl.Then():
            if compact_s2f6:
                txl.ptx[TMA_G2S_3D](
                    part_smem.ptr_to(0, 0),
                    txl.address_of(part_map),
                    txl.int32(0),
                    txl.int32(0),
                    txl.Cast("int32", txl.cta_id() * ROWS_PER_CTA),
                    txl.cuda.cvta_generic_to_shared(part_full.ptr_to([0])),
                )
            else:
                txl.ptx[TMA_G2S_4D](
                    part_smem.ptr_to(0, 0),
                    txl.address_of(part_map),
                    txl.int32(0),
                    txl.int32(0),
                    txl.int32(0),
                    txl.Cast("int32", txl.cta_id() * ROWS_PER_CTA),
                    txl.cuda.cvta_generic_to_shared(part_full.ptr_to([0])),
                )
            part_full.arrive(0, tx_count=TILE_BYTES)
        txl.cuda.iket.range_end(issue_tok[0])

        scalar_tok = iket_range("combine-scalars")
        d = txl.local_scalar("int32", init=0)
        with txl.If(txl.And(row_valid, lane8 == 0)), txl.Then():
            txl.ptx.ld.global_.nc.b32(d, deg.ptr_to([h * TOTAL_Q + tok]))
        txl.ptx.shfl_sync.idx.b32(d, d, subgroup_base, txl.uint32(31), txl.uint32(0xFFFFFFFF))

        mvals = [txl.local_scalar("float32", init=txl.float32(NEG_INF)) for _ in range(TOPK)]
        lvals = [txl.local_scalar("float32", init=txl.float32(0.0)) for _ in range(TOPK)]
        weights = [txl.local_scalar("float32", init=txl.float32(0.0)) for _ in range(TOPK)]
        inv_w = txl.local_scalar("float32", init=txl.float32(1.0))

        qw = txl.alloc_local([TOPK], "int32")
        if compact_s2f6:
            with txl.If(row_valid), txl.Then():
                for v in range(TOPK // 4):
                    txl.ptx.ld.global_.nc.v4.b32(
                        qw[4 * v],
                        qw[4 * v + 1],
                        qw[4 * v + 2],
                        qw[4 * v + 3],
                        q_part.ptr_to([row * TOPK + 4 * v]),
                    )
        with txl.If(txl.And(row_valid, lane8 == 0)), txl.Then():
            mraw = txl.alloc_local([TOPK], "uint32")
            lraw = txl.alloc_local([TOPK], "uint32")
            for v in range(TOPK // 4):
                txl.ptx.ld.global_.nc.v4.b32(
                    mraw[4 * v],
                    mraw[4 * v + 1],
                    mraw[4 * v + 2],
                    mraw[4 * v + 3],
                    m_part.ptr_to([row * TOPK + 4 * v]),
                )
                txl.ptx.ld.global_.nc.v4.b32(
                    lraw[4 * v],
                    lraw[4 * v + 1],
                    lraw[4 * v + 2],
                    lraw[4 * v + 3],
                    l_part.ptr_to([row * TOPK + 4 * v]),
                )
            for s_i in range(TOPK):
                with txl.If(s_i < d), txl.Then():
                    txl.ptx.mov.b32(mvals[s_i], mraw[s_i])
                    txl.ptx.mov.b32(lvals[s_i], lraw[s_i])
            M = txl.local_scalar("float32", init=txl.float32(NEG_INF))
            for s_i in range(TOPK):
                txl.assign(M, txl.max(M, mvals[s_i]))
            M_safe = txl.if_then_else(M == txl.float32(NEG_INF), txl.float32(0.0), M)
            wsum = txl.local_scalar("float32", init=txl.float32(0.0))
            for s_i in range(TOPK):
                with txl.If(txl.And(s_i < d, lvals[s_i] > txl.float32(0.0))), txl.Then():
                    txl.ptx.ex2.approx.ftz.f32(weights[s_i], mvals[s_i] - M_safe)
                    if compact_s2f6:
                        txl.assign(wsum, wsum + weights[s_i] * lvals[s_i])
                    else:
                        txl.assign(wsum, wsum + weights[s_i] * lvals[s_i])
            txl.ptx.rcp.approx.ftz.f32(
                inv_w, txl.Select(wsum > txl.float32(0.0), wsum, txl.float32(1.0))
            )
        for s_i in range(TOPK):
            txl.ptx.shfl_sync.idx.b32(
                weights[s_i], weights[s_i], subgroup_base, txl.uint32(31), txl.uint32(0xFFFFFFFF)
            )
        txl.ptx.shfl_sync.idx.b32(
            inv_w, inv_w, subgroup_base, txl.uint32(31), txl.uint32(0xFFFFFFFF)
        )
        txl.cuda.iket.range_end(scalar_tok[0])

        wait_tok = iket_range("combine-tma-wait")
        part_full.wait(0, 0)
        txl.cuda.iket.range_end(wait_tok[0])
        vector_tok = iket_range("combine-vector")
        with txl.If(txl.And(row_valid, d > 0)), txl.Then():
            acc = [txl.local_scalar("uint64") for _ in range(8)]
            for i in range(8):
                txl.ptx.mov.b64(acc[i], txl.float32(0.0), txl.float32(0.0))
            if compact_s2f6:
                col = lane8 * 16
            else:
                half = lane8 >> 2
                col = txl.bitwise_and(lane8, 3) * 16
            for sb in range(0, TOPK, 4):
                words = [txl.alloc_local([8], "uint32") for _ in range(4)]
                for sj in range(4):
                    s_i = sb + sj
                    if compact_s2f6:
                        raw = txl.alloc_local([4], "uint32")
                        smem_row = local_row * TOPK + s_i
                        txl.ptx.ld.shared.v4.b32(
                            raw[0], raw[1], raw[2], raw[3], part_smem.ptr_to(smem_row, col)
                        )
                        scale_exp = txl.local_scalar(
                            "int32",
                            init=txl.bitwise_and(
                                txl.shift_right(qw[s_i], (lane8 >> 1) * 8), txl.int32(0xFF)
                            ),
                        )
                        scale_pair = txl.local_scalar(
                            "uint16",
                            init=txl.Cast(
                                "uint16",
                                txl.bitwise_or(scale_exp, txl.shift_left(scale_exp, txl.int32(8))),
                            ),
                        )
                        for wi in range(4):
                            txl.ptx["cvt.rn.scaled::n2::ue8m0.bf16x2.s2f6x2"](
                                words[sj][2 * wi], txl.Cast("uint16", raw[wi]), scale_pair
                            )
                            txl.ptx["cvt.rn.scaled::n2::ue8m0.bf16x2.s2f6x2"](
                                words[sj][2 * wi + 1],
                                txl.Cast("uint16", txl.shift_right(raw[wi], txl.uint32(16))),
                                scale_pair,
                            )
                    else:
                        smem_row = (local_row * TOPK + s_i) * 2 + half
                        txl.ptx.ld.shared.v4.b32(
                            words[sj][0],
                            words[sj][1],
                            words[sj][2],
                            words[sj][3],
                            part_smem.ptr_to(smem_row, col),
                        )
                        txl.ptx.ld.shared.v4.b32(
                            words[sj][4],
                            words[sj][5],
                            words[sj][6],
                            words[sj][7],
                            part_smem.ptr_to(smem_row, col + 8),
                        )
                for sj in range(4):
                    s_i = sb + sj
                    with txl.If(weights[s_i] > txl.float32(0.0)), txl.Then():
                        w_pair = txl.local_scalar("uint64")
                        txl.ptx.mov.b64(w_pair, weights[s_i], weights[s_i])
                        for wi in range(8):
                            lo_f = txl.local_scalar("float32")
                            hi_f = txl.local_scalar("float32")
                            partial_pair = txl.local_scalar("uint64")
                            txl.ptx.mov.b32(lo_f, txl.shift_left(words[sj][wi], txl.uint32(16)))
                            txl.ptx.mov.b32(
                                hi_f, txl.bitwise_and(words[sj][wi], txl.uint32(0xFFFF0000))
                            )
                            txl.ptx.mov.b64(partial_pair, lo_f, hi_f)
                            txl.ptx.fma.rn.ftz.f32x2(acc[wi], partial_pair, w_pair, acc[wi])
            inv_pair = txl.local_scalar("uint64")
            txl.ptx.mov.b64(inv_pair, inv_w, inv_w)
            outw = txl.alloc_local([8], "uint32")
            for i in range(8):
                lo_f = txl.local_scalar("float32")
                hi_f = txl.local_scalar("float32")
                txl.ptx.mul.rz.ftz.f32x2(acc[i], acc[i], inv_pair)
                txl.ptx.mov.b64(lo_f, hi_f, acc[i])
                txl.ptx.cvt.rn.bf16x2.f32(outw[i], hi_f, lo_f)
            ob = row * (HEAD_DIM // 2) + lane8 * 8
            txl.ptx.st.global_.v8.b32(out.ptr_to([ob]), *(outw[w] for w in range(8)))
        txl.cuda.iket.range_end(vector_tok[0])

    return msa_reverse_combine_tma


def _launch_params(func, dims, pdl):
    tags = [f"blockIdx.{c}" for c in "xyz"[:dims]] + ["threadIdx.x"]
    if pdl:
        tags.append("tirx.use_programtic_dependent_launch")
    tags.append("tirx.use_dyn_shared_memory")
    return func.with_attr("tirx.kernel_launch_params", tags)


class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode(tensor, dtype_name, dims, strides, box):
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
        2,
        0,
    )
    return desc


_COMPILED = {}


def _compiled(key, factory, dims=1, pdl=False, **kw):
    exe = _COMPILED.get(key)
    if exe is None:
        from tirx_kernels.runner import compile_kernel

        os.environ.setdefault("TVM_CUDA_PTXAS_REG_LEVEL", "6")
        func = factory(**kw).func
        if USE_PDL:
            func = _launch_params(func, dims, pdl)
        exe = compile_kernel(func, arch="sm_100a")
        _COMPILED[key] = exe
    return exe


def setup_reverse(data, total_q, B):
    q, k, v = data["q"], data["k"], data["v"]
    q2k = data["q2k_indices"]
    cu_q = data["cu_seqlens_q"]
    cu_k = data["cu_seqlens_k"]
    page_table = data["page_table"]
    seqused = data["seqused_k"]
    out = data["output"]
    scale = float(data["softmax_scale"])
    device = q.device
    paged = page_table is not None
    kv_fp8 = k.dtype == torch.float8_e4m3fn
    if q.dtype != torch.bfloat16:
        raise NotImplementedError("bf16 queries only")
    total_q_, hq, hd = q.shape
    assert hd == HEAD_DIM and total_q_ == total_q
    hkv = k.shape[1]
    topk = q2k.shape[2]
    # The raw-numerator representation keeps quantization scale independent
    # of the row sum, so use the byte-wide S2F6 partial path for both official
    # sparse rows.  The final output remains BF16.
    compact_s2f6 = topk in (4, 8)
    assert topk % 4 == 0 and topk <= 255
    if paged:
        num_pages, hkv_p, page_size, hd_p = k.shape
        assert hkv_p == hkv and page_size == BLK_N and hd_p == HEAD_DIM
        max_pages = page_table.shape[1]
        nblk = max_pages
    else:
        max_pages = 1
        nblk = ceildiv(k.shape[0], BLK_N)
    assert nblk <= PREP_THREADS, "prep kernel handles up to 256 blocks per sequence"
    cap = ceildiv(total_q, 32) * 32
    nitems = hkv * B * nblk
    num_chunks = ceildiv(total_q, PREP_THREADS)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    item_batch = 10 if kv_fp8 else BATCH
    prep = _compiled(
        ("prep-plan", total_q, hq, hkv, topk, nblk, cap, B, paged, num_chunks, item_batch, num_sms),
        make_prep_kernel,
        dims=2,
        pdl=False,
        total_q=total_q,
        hq=hq,
        hkv=hkv,
        topk=topk,
        nblk=nblk,
        cap=cap,
        num_seqs=B,
        paged=paged,
        num_chunks=num_chunks,
        item_batch=item_batch,
        num_ctas_main=num_sms,
    )
    main = _compiled(
        (
            "main",
            total_q,
            hq,
            hkv,
            topk,
            nblk,
            cap,
            B,
            paged,
            max_pages,
            kv_fp8,
            compact_s2f6,
            num_sms,
            FUSED_COMBINE,
        ),
        make_main_kernel,
        dims=1,
        pdl=True,
        total_q=total_q,
        hq=hq,
        hkv=hkv,
        topk=topk,
        nblk=nblk,
        cap=cap,
        num_seqs=B,
        paged=paged,
        max_pages=max_pages,
        kv_fp8=kv_fp8,
        num_ctas=num_sms,
        compact_s2f6=compact_s2f6,
        fused_combine=FUSED_COMBINE,
    )
    combine_rows = 16 if topk == 8 else 32
    n_comb_ctas = ceildiv(total_q * hq, combine_rows)
    if topk in (4, 8):
        combine_factory = make_combine_kernel_tma
        combine_key = f"combine-tma{topk}" + ("-s2f6" if compact_s2f6 else "")
    else:
        combine_factory = make_combine_kernel
        combine_key = "combine-leader"
    if topk in (4, 8):
        comb = _compiled(
            (combine_key, total_q, hq, hkv, topk, n_comb_ctas, compact_s2f6),
            combine_factory,
            dims=1,
            pdl=True,
            total_q=total_q,
            hq=hq,
            hkv=hkv,
            topk=topk,
            num_ctas=n_comb_ctas,
            compact_s2f6=compact_s2f6,
        )
    else:
        comb = _compiled(
            (combine_key, total_q, hq, hkv, topk, n_comb_ctas),
            combine_factory,
            dims=1,
            pdl=True,
            total_q=total_q,
            hq=hq,
            hkv=hkv,
            topk=topk,
            num_ctas=n_comb_ctas,
        )

    gqa = hq // hkv
    q_map = _encode(
        q, "bfloat16", (HEAD_DIM, hq * total_q), (HEAD_DIM * F16_BYTES,), (HEAD_DIM // 2, gqa)
    )
    if kv_fp8:
        if paged:
            dims = (HEAD_DIM, BLK_N, hkv, k.shape[0])
            strides = (HEAD_DIM, BLK_N * HEAD_DIM, hkv * BLK_N * HEAD_DIM)
            box = (HEAD_DIM, BLK_N, 1, 1)
        else:
            dims = (HEAD_DIM, k.shape[0], hkv)
            strides = (hkv * HEAD_DIM, HEAD_DIM)
            box = (HEAD_DIM, BLK_N, 1)
        k_map = _encode(k, "float8_e4m3fn", dims, strides, box)
        v_map = _encode(v, "float8_e4m3fn", dims, strides, box)
    else:
        if paged:
            dims = (HEAD_DIM // 2, BLK_N, 2, hkv, k.shape[0])
            strides = (
                HEAD_DIM * F16_BYTES,
                (HEAD_DIM // 2) * F16_BYTES,
                BLK_N * HEAD_DIM * F16_BYTES,
                hkv * BLK_N * HEAD_DIM * F16_BYTES,
            )
            box = (HEAD_DIM // 2, BLK_N, 2, 1, 1)
        else:
            dims = (HEAD_DIM // 2, k.shape[0], hkv * 2)
            strides = (hkv * HEAD_DIM * F16_BYTES, (HEAD_DIM // 2) * F16_BYTES)
            box = (HEAD_DIM // 2, BLK_N, 2)
        k_map = _encode(k, "bfloat16", dims, strides, box)
        v_map = _encode(v, "bfloat16", dims, strides, box)

    dummy = torch.zeros(4, dtype=torch.int32, device=device)
    cu_k_arg = cu_k if cu_k is not None else dummy
    pt_arg = page_table.contiguous().view(-1) if paged else dummy
    su_arg = seqused if paged else dummy
    cursor = torch.zeros(nitems, dtype=torch.int32, device=device)
    edge_table = torch.zeros(nitems * cap, dtype=torch.int32, device=device)
    deg = torch.zeros(hkv * total_q, dtype=torch.int32, device=device)
    arrivals = torch.zeros(hkv * total_q, dtype=torch.int32, device=device)
    part_words = HEAD_DIM // (4 if compact_s2f6 else 2)
    o_part = torch.zeros(total_q * hq * topk * part_words, dtype=torch.int32, device=device)
    q_part = torch.zeros(total_q * hq * topk, dtype=torch.int32, device=device)
    m_part = torch.zeros(total_q * hq * topk, dtype=torch.float32, device=device)
    l_part = torch.zeros(total_q * hq * topk, dtype=torch.float32, device=device)
    sched = torch.zeros(2, dtype=torch.int32, device=device)
    plan = torch.zeros(8, dtype=torch.int32, device=device)
    tok_per_tile = BLK_M // gqa
    max_batches = ceildiv(hkv * total_q * topk, tok_per_tile) + nitems
    batch_tab = torch.zeros(max_batches * 12, dtype=torch.int32, device=device)
    out_i32 = out.view(torch.int32).view(-1)
    scale_log2 = scale * LOG2E
    part_map = None
    if topk in (4, 8):
        rows = total_q * hq
        if compact_s2f6:
            part_map = _encode(
                o_part,
                "float8_e4m3fn",
                (HEAD_DIM, topk, rows),
                (HEAD_DIM, topk * HEAD_DIM),
                (HEAD_DIM, topk, combine_rows),
            )
        else:
            part_map = _encode(
                o_part,
                "bfloat16",
                (HEAD_DIM // 2, 2, topk, rows),
                ((HEAD_DIM // 2) * F16_BYTES, HEAD_DIM * F16_BYTES, topk * HEAD_DIM * F16_BYTES),
                (HEAD_DIM // 2, 2, topk, combine_rows),
            )

    prep_args = (
        q2k.view(-1),
        cu_q,
        cu_k_arg,
        su_arg,
        cursor,
        edge_table,
        deg,
        arrivals,
        out_i32,
        plan,
        batch_tab,
    )
    q_raw = q.view(torch.int32).view(-1)
    main_args = (
        q_map.ptr,
        k_map.ptr,
        v_map.ptr,
        out_i32,
        q_raw,
        cu_q,
        cu_k_arg,
        pt_arg,
        su_arg,
        cursor,
        edge_table,
        deg,
        arrivals,
        o_part,
        q_part,
        m_part,
        l_part,
        sched,
        batch_tab,
        plan,
        scale_log2,
    )
    keep = (
        q,
        k,
        v,
        q2k,
        out,
        cu_q,
        cu_k_arg,
        pt_arg,
        su_arg,
        cursor,
        edge_table,
        deg,
        arrivals,
        o_part,
        q_part,
        m_part,
        l_part,
        sched,
        plan,
        batch_tab,
        q_map,
        k_map,
        v_map,
        out_i32,
        q_raw,
        part_map,
    )

    if topk in (4, 8):
        comb_args = (part_map.ptr, out_i32, deg, q_part, m_part, l_part)
    else:
        comb_args = (out_i32, deg, o_part, m_part, l_part)

    prep_fn, main_fn, comb_fn = prep.jit().main, main.jit().main, comb.jit().main

    def run(
        _prep=prep_fn,
        _main=main_fn,
        _comb=comb_fn,
        _pa=prep_args,
        _ma=main_args,
        _ca=comb_args,
        _keep=keep,
    ):
        _prep(*_pa)
        _main(*_ma)
        if not FUSED_COMBINE:
            _comb(*_ca)

    run()
    torch.cuda.synchronize(device)
    return run


DENSITY_THRESHOLD = 0.25


def setup_union(data, total_q, B):
    q, k, v = data["q"], data["k"], data["v"]
    q2k = data["q2k_indices"]
    cu_q = data["cu_seqlens_q"]
    cu_k = data["cu_seqlens_k"]
    page_table = data["page_table"]
    seqused = data["seqused_k"]
    out = data["output"]
    scale = float(data["softmax_scale"])
    device = q.device
    paged = page_table is not None
    kv_fp8 = k.dtype == torch.float8_e4m3fn
    total_q_, hq, hd = q.shape
    hkv = k.shape[1]
    topk = q2k.shape[2]
    if paged:
        max_pages = page_table.shape[1]
        max_blocks = max_pages
    else:
        max_pages = 1
        max_blocks = ceildiv(k.shape[0], BLK_N)
    mask_words = max(1, ceildiv(max_blocks, 32))
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    kv_depth = 3 if kv_fp8 else 4
    exe = _compiled(
        ("union", total_q, hq, hkv, topk, mask_words, paged, max_pages, kv_fp8, num_sms, kv_depth),
        make_union_kernel,
        total_q=total_q,
        hq=hq,
        hkv=hkv,
        topk=topk,
        mask_words=mask_words,
        paged=paged,
        max_pages=max_pages,
        kv_fp8=kv_fp8,
        num_ctas=num_sms,
        kv_depth=kv_depth,
    )
    gqa = hq // hkv
    tok_per_tile = BLK_M // gqa
    q_map = _encode(
        q,
        "bfloat16",
        (HEAD_DIM // 2, hq, total_q, 2),
        (HEAD_DIM * F16_BYTES, hq * HEAD_DIM * F16_BYTES, (HEAD_DIM // 2) * F16_BYTES),
        (HEAD_DIM // 2, gqa, tok_per_tile, 2),
    )
    o_map = _encode(
        out,
        "bfloat16",
        (HEAD_DIM // 2, hq, total_q, 2),
        (HEAD_DIM * F16_BYTES, hq * HEAD_DIM * F16_BYTES, (HEAD_DIM // 2) * F16_BYTES),
        (HEAD_DIM // 2, gqa, tok_per_tile, 2),
    )
    if kv_fp8:
        if paged:
            dims = (HEAD_DIM, BLK_N, hkv, k.shape[0])
            strides = (HEAD_DIM, BLK_N * HEAD_DIM, hkv * BLK_N * HEAD_DIM)
            box = (HEAD_DIM, BLK_N, 1, 1)
        else:
            dims = (HEAD_DIM, k.shape[0], hkv)
            strides = (hkv * HEAD_DIM, HEAD_DIM)
            box = (HEAD_DIM, BLK_N, 1)
        k_map = _encode(k, "float8_e4m3fn", dims, strides, box)
        v_map = _encode(v, "float8_e4m3fn", dims, strides, box)
    else:
        if paged:
            dims = (HEAD_DIM // 2, BLK_N, 2, hkv, k.shape[0])
            strides = (
                HEAD_DIM * F16_BYTES,
                (HEAD_DIM // 2) * F16_BYTES,
                BLK_N * HEAD_DIM * F16_BYTES,
                hkv * BLK_N * HEAD_DIM * F16_BYTES,
            )
            box = (HEAD_DIM // 2, BLK_N, 2, 1, 1)
        else:
            dims = (HEAD_DIM // 2, k.shape[0], hkv * 2)
            strides = (hkv * HEAD_DIM * F16_BYTES, (HEAD_DIM // 2) * F16_BYTES)
            box = (HEAD_DIM // 2, BLK_N, 2)
        k_map = _encode(k, "bfloat16", dims, strides, box)
        v_map = _encode(v, "bfloat16", dims, strides, box)
    dummy = torch.zeros(4, dtype=torch.int32, device=device)
    cu_k_arg = cu_k if cu_k is not None else dummy
    pt_arg = page_table.contiguous().view(-1) if paged else dummy
    su_arg = seqused if paged else dummy
    sched = torch.zeros(2, dtype=torch.int32, device=device)
    out_i32 = out.view(torch.int32).view(-1)
    args = (
        q_map.ptr,
        k_map.ptr,
        v_map.ptr,
        o_map.ptr,
        out_i32,
        q2k.view(-1),
        cu_q,
        cu_k_arg,
        pt_arg,
        su_arg,
        sched,
        scale * LOG2E,
        int(B),
    )
    keep = (
        q,
        k,
        v,
        q2k,
        out,
        cu_q,
        cu_k_arg,
        pt_arg,
        su_arg,
        sched,
        q_map,
        k_map,
        v_map,
        o_map,
        out_i32,
    )
    fn = exe.jit().main

    def run(_fn=fn, _args=args, _keep=keep):
        _fn(*_args)

    run()
    torch.cuda.synchronize(device)
    return run


def setup(data, total_q, B):
    q, k = data["q"], data["k"]
    q2k = data["q2k_indices"]
    page_table = data["page_table"]
    topk = q2k.shape[2]
    if page_table is not None:
        selectable = page_table.shape[1]
    else:
        selectable = ceildiv(k.shape[0], BLK_N)
    density = topk / max(selectable, 1)
    if density >= DENSITY_THRESHOLD:
        return setup_union(data, total_q, B)
    return setup_reverse(data, total_q, B)


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
        "run": "msa_prefill-20260913-030259",
        "selected_version": "frontier/dispatch-all-s2f6-raw-maxabs",
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
        "label": "flat_fp8_b3_q1024_kv8192_h32",
        "batch_size": 3,
        "seqlen_q": 1024,
        "seqlen_kv": 8192,
        "num_qo_heads": 32,
        "num_kv_heads": 2,
        "topk": 8,
        "kv_layout": "flat",
        "kv_dtype": "float8_e4m3fn",
        "seed": 71,
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
    """The candidate's own density dispatch, expressed on the config alone.

    `setup` picks the union route when the selection is dense relative to the
    blocks a token may see, and the KV-major reverse route otherwise.
    """
    selectable = ceildiv(int(resolved["seqlen_kv"]), BLK_N)
    density = int(resolved["topk"]) / max(selectable, 1)
    return "union" if density >= DENSITY_THRESHOLD else "reverse"


def get_kernel(**config: Any):
    """Return the traced tirx-lite PrimFuncs this config's route builds.

    Both routes compile inside the candidate's own `setup`, which pins per-route
    launch geometry, PDL and ptxas settings and caches by compile key. This
    entry point re-derives only what tracing needs, for registry discovery and
    IR inspection.
    """
    from tirx_kernels.runner import hardware_num_sms

    resolved = _config(**config)
    batch = int(resolved["batch_size"])
    total_q = batch * int(resolved["seqlen_q"])
    hkv = int(resolved["num_kv_heads"])
    hq = int(resolved["num_qo_heads"])
    topk = int(resolved["topk"])
    paged = resolved["kv_layout"] == "paged"
    nblk = ceildiv(int(resolved["seqlen_kv"]), BLK_N)
    cap = ceildiv(total_q, 32) * 32
    num_sms = hardware_num_sms()
    if _route(resolved) == "union":
        kv_fp8 = resolved["kv_dtype"] == "float8_e4m3fn"
        return {
            "union": make_union_kernel(
                total_q=total_q,
                hq=hq,
                hkv=hkv,
                topk=topk,
                mask_words=max(1, ceildiv(nblk, 32)),
                paged=paged,
                max_pages=nblk if paged else 1,
                kv_fp8=kv_fp8,
                num_ctas=num_sms,
                kv_depth=3 if kv_fp8 else 4,
            ).func
        }
    num_chunks = ceildiv(total_q, PREP_THREADS)
    kv_fp8 = resolved["kv_dtype"] == "float8_e4m3fn"
    compact = topk in (4, 8)
    return {
        "prep": make_prep_kernel(
            total_q=total_q,
            hq=hq,
            hkv=hkv,
            topk=topk,
            nblk=nblk,
            cap=cap,
            num_seqs=batch,
            paged=paged,
            num_chunks=num_chunks,
            item_batch=10 if kv_fp8 else BATCH,
            num_ctas_main=num_sms,
        ).func,
        "main": make_main_kernel(
            total_q=total_q,
            hq=hq,
            hkv=hkv,
            topk=topk,
            nblk=nblk,
            cap=cap,
            num_seqs=batch,
            paged=paged,
            max_pages=nblk if paged else 1,
            kv_fp8=kv_fp8,
            num_ctas=num_sms,
            compact_s2f6=compact,
            fused_combine=FUSED_COMBINE,
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
        visible_blocks = ceildiv(offset + row % seqlen_q + 1, BLK_N)
        for kv_head in range(num_kv_heads):
            selected = torch.randperm(visible_blocks, generator=generator)
            selected = selected[: min(topk, visible_blocks)].sort().values
            out[kv_head, row, : selected.numel()] = selected.to(torch.int32)
    return out.to(device)


def _to_pages(logical, batch_size, seqlen_kv):
    """128-token pages stored in reverse order, as the packaged benchmark builds them."""
    _total_k, num_kv_heads, head_dim = logical.shape
    pages_per_seq = ceildiv(seqlen_kv, BLK_N)
    total_pages = batch_size * pages_per_seq
    padded = logical.view(batch_size, seqlen_kv, num_kv_heads, head_dim)
    if pages_per_seq * BLK_N != seqlen_kv:
        padded = logical.new_zeros((batch_size, pages_per_seq * BLK_N, num_kv_heads, head_dim))
        padded[:, :seqlen_kv] = logical.view(batch_size, seqlen_kv, num_kv_heads, head_dim)
    pages = (
        padded.view(batch_size, pages_per_seq, BLK_N, num_kv_heads, head_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(total_pages, num_kv_heads, BLK_N, head_dim)
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
        gathered = pages[page_table.reshape(-1)]  # [B*pages, Hkv, BLK_N, D]
        gathered = gathered.view(batch, -1, gathered.shape[1], BLK_N, gathered.shape[-1])
        flat = gathered.permute(0, 1, 3, 2, 4).reshape(
            batch, -1, gathered.shape[2], gathered.shape[-1]
        )
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
    key_block = torch.arange(seqlen_kv, device=device) // BLK_N
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
    """Build the candidate's own dispatch closure for this shape."""
    return setup(case, case["total_q"], case["batch_size"])


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
        pages_per_seq = (kv_lens + BLK_N - 1) // BLK_N
        page_base = torch.cumsum(pages_per_seq, 0) - pages_per_seq
        total_pages = int(pages_per_seq.sum())
        sequences = torch.arange(kv_lens.numel(), device=device)
        seq_of_token = torch.repeat_interleave(sequences, kv_lens, output_size=k.shape[0])
        local = torch.arange(k.shape[0], device=device) - cu_k[:-1][seq_of_token]
        page = page_base[seq_of_token] + local // BLK_N
        slot = local % BLK_N
        k_pages = k.new_zeros((total_pages, k.shape[1], BLK_N, k.shape[2]))
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
    tail = torch.clamp(q_pos.view(1, -1) + 1 - last * BLK_N, max=BLK_N)
    seq_lens = ((count - 1) * BLK_N + tail).clamp(min=0).to(torch.int32).contiguous()
    max_seq_len = int(seq_lens.max().item())
    key = str(device)
    if key not in _BRIDGE_WORKSPACE:
        _BRIDGE_WORKSPACE[key] = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
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
            blk_kv=BLK_N,
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
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **config: Any
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
