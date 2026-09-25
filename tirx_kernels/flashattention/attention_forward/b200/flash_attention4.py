# This file is a TIRx port of code from flash-attention
# (https://github.com/Dao-AILab/flash-attention @ 00756db9), Copyright (c) 2022,
# the respective contributors, as shown by licenses/AUTHORS.flash-attention.txt
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashAttention-4 with K-owned launch, roles, storage, and synchronization.

The mathematical datapath and low-level PTX issue order follow the frozen
FlashAttention-4 implementation. Tensor maps are encoded on the host because
txl.kernel is a device-only entry.

Upstream source: flash_attn/cute/flash_fwd_sm100.py.
"""

# txl.kernel consumes concrete annotations; postponed annotations would stringify them.
import argparse
import ctypes
import math
import os
from functools import partial
from typing import Any

import torch

import tirx_kernels.tirx_lite as txl
import tvm
import tvm.testing
from tirx_kernels.bench.runner import PREPARE_CUDA_ARCH_ENV, PREPARE_NUM_SMS_ENV, bench, cuda_target
from tvm.tirx.cuda import iket
from tvm.tirx.cuda.iket import IketProfiler

IKET_EVENT_NAMES = (
    "correction",
    "epi-ld-tmem",
    "issue-tma-k",
    "issue-tma-q",
    "issue-tma-v",
    "softmax-exp2",
    "softmax-fma",
    "softmax-max",
    "softmax-sum",
    "softmax-tmem-st",
    "tma-store",
    "softmax-baseline",
    "softmax-phase-0",
    "softmax-phase-1",
    "softmax-phase-2",
    "softmax-phase-3",
    "softmax-phase-4",
    "softmax-phase-5",
)

N_COLS_TMEM = 512
TMEM_PIPE_DEPTH = 2
SMEM_PIPE_DEPTH_Q = 2
SMEM_PIPE_DEPTH_KV = 3
BLK_M = 128
BLK_N = 128
SOFTMAX_LD_CHUNK = 32
TMEM_EPI_LD_SIZE = 16
USE_S0_S1_BARRIER = False
MMA_N = 128
MMA_K = 16
F16_BYTES = 2
EMU_PAIRS_CAUSAL = 2
EMU_START_CAUSAL = 0
EMU_PAIRS_NC = 2
EMU_START_NC = 1
MAX_CTAS = 148

TMA_G2S_3D = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
TMA_G2S_4D = (
    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
TMA_S2G_3D = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
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

ID_QK = 136314896
ID_PV = 136380432


def ceildiv(a, b):
    return (a + b - 1) // b


def make_kernel(
    BATCH_SIZE,
    SEQ_LEN_Q,
    SEQ_LEN_KV,
    NUM_QO_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
    is_causal=False,
    tmem_pipe_depth=TMEM_PIPE_DEPTH,
    smem_pipe_depth_kv=SMEM_PIPE_DEPTH_KV,
):
    """Trace the kernel for one specialization. Every ``txl.meta_var`` of the
    original is a plain Python constant here — the host language is the macro
    system (design doc §1)."""
    TMEM_DEPTH = tmem_pipe_depth
    IS_THOR = os.environ.get(PREPARE_CUDA_ARCH_ENV, "sm_100a") == "sm_110a"
    USE_2CTA = IS_THOR and not is_causal
    cta_group = 2 if USE_2CTA else 1
    # A 2-CTA MMA distributes each operand's K dimension across the pair. Each
    # CTA stores a 64-wide slice, so the same shared-memory budget holds twice
    # as many K/V stages.
    KV_DEPTH = smem_pipe_depth_kv * cta_group
    KV_ROWS = BLK_N // cta_group

    GQA_RATIO = NUM_QO_HEADS // NUM_KV_HEADS
    SEQ_Q_PER_TILE = BLK_M // GQA_RATIO
    # Stats handshake width — orig:L340-349.
    STATS_BAR_PAIRWISE = GQA_RATIO == 1
    L2_SIZE = 50 * 1024 * 1024
    SIZE_ONE_KV_HEAD = SEQ_LEN_KV * HEAD_DIM * 2 * F16_BYTES
    L2_SWIZZLE = (
        1 if L2_SIZE < SIZE_ONE_KV_HEAD else 1 << int(math.log2(L2_SIZE // SIZE_ONE_KV_HEAD))
    )
    SSCALE_TOTAL_SIZE = 2 * SMEM_PIPE_DEPTH_Q * BLK_M
    assert TMEM_DEPTH * MMA_N <= N_COLS_TMEM, "TMEM columns exceeded"
    num_q_blocks_total = ceildiv(SEQ_LEN_Q, SEQ_Q_PER_TILE)
    num_q_blocks = ceildiv(num_q_blocks_total, SMEM_PIPE_DEPTH_Q * cta_group)
    num_total_tasks = BATCH_SIZE * NUM_KV_HEADS * num_q_blocks
    num_kv_blocks = ceildiv(SEQ_LEN_KV, BLK_N)
    # orig:L601-620.
    EPI_ON_SOFTMAX = is_causal
    EARLY_Q_RELEASE = not is_causal
    if USE_2CTA:
        max_ctas = int(os.environ.get(PREPARE_NUM_SMS_ENV, "20"))
        cta_count = min(max_ctas, num_total_tasks * cta_group)
        cta_count -= cta_count % cta_group
    else:
        max_ctas = 6 * int(os.environ.get(PREPARE_NUM_SMS_ENV, "20")) if IS_THOR else MAX_CTAS
        cta_count = num_total_tasks if is_causal else min(max_ctas, num_total_tasks)
    scheduler_ctas = cta_count // cta_group
    # PV gemm split point, regime-tuned — orig:L922-930.
    thor_causal_d128 = IS_THOR and is_causal and HEAD_DIM == 128
    K_SPLIT = (4 if is_causal else 6) * MMA_K
    P_SPLIT_Q = 2 if is_causal else 3
    ACC_SCALE_BASE = 0
    ROW_SUM_BASE = 0
    scale_log2 = math.log2(math.e) / math.sqrt(HEAD_DIM)
    rescale_threshold = 8.0
    STEADY_DESC = is_causal and GQA_RATIO > 1
    NEG_INF = -float("inf")
    tma_g2s_3d = (
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
        if USE_2CTA
        else TMA_G2S_3D
    )
    tma_g2s_4d = (
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
        if USE_2CTA
        else TMA_G2S_4D
    )
    mma_f16 = "tcgen05.mma.cta_group::2.kind::f16" if USE_2CTA else MMA_F16
    mma_keep_lanes = (txl.uint32(0),) * (8 if USE_2CTA else 4)
    id_qk = 270532624 if USE_2CTA else ID_QK
    id_pv = 270598160 if USE_2CTA else ID_PV
    tmem_alloc = (
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32" if USE_2CTA else TMEM_ALLOC
    )
    tmem_dealloc = "tcgen05.dealloc.cta_group::2.sync.aligned.b32" if USE_2CTA else TMEM_DEALLOC
    tmem_relinquish = (
        "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned" if USE_2CTA else TMEM_RELINQUISH
    )
    tcgen05_commit = (
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        if USE_2CTA
        else TCGEN05_COMMIT
    )

    # ---- trace-time helpers over the causal block bounds — orig:L138-152 ----

    def n_block_max_of(m_block_idx):
        nbm = num_kv_blocks
        if not is_causal:
            return nbm
        m_idx_max = (m_block_idx + 1) * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
        n_idx = m_idx_max + SEQ_LEN_KV - SEQ_LEN_Q
        return txl.min(nbm, ceildiv(n_idx, BLK_N))

    def n_block_min_causal_of(m_block_idx):
        m_idx_min = m_block_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
        n_idx = m_idx_min + SEQ_LEN_KV - SEQ_LEN_Q
        return txl.max(0, n_idx // BLK_N)

    @txl.kernel(warps=16, arch="sm_100a", min_blocks_per_sm=1, grid=cta_count)
    def flash_attention4(
        Q_tensor_map: txl.TensorMap,
        Q_tensor_map_1: txl.TensorMap,
        K_tensor_map: txl.TensorMap,
        K_tensor_map_1: txl.TensorMap,
        V_tensor_map: txl.TensorMap,
        V_tensor_map_1: txl.TensorMap,
        O_tensor_map: txl.TensorMap,
    ):
        # ---- CTA coordinates — orig:L625-628 ---------------------------------
        cta_rank = txl.cta_id_in_cluster([2], preferred=[2]) if USE_2CTA else txl.int32(0)
        cluster_id = txl.cta_id() // cta_group
        # Materialize the warp-uniform ids once; TIRx expressions are trees and
        # rebuilding these at every use expands address and mask arithmetic.
        warp_cta = txl.warp_id()
        wg_id = warp_cta >> 2
        warp_id = warp_cta & 3
        tid_in_wg = txl.thread_id() & 127

        # ---- shared memory — orig:L629-696 -----------------------------------
        # Declaration order reproduces the frozen kernel's byte layout exactly
        # (audited in the notes: Q@0, K/V@65536, O, sScale, tmem mailbox, then
        # every barrier ring at the original's offset).
        smem = txl.smem_pool()
        # swizzle=txl.SW128B is the same composed layout the original's
        # pool.alloc_tcgen05_mma_AB(..., "float16") picks ("auto" -> 128B atom):
        # both call mma_shared_layout(dtype, SWIZZLE_128B_ATOM, shape) with
        # align=1024.
        q_smem = smem.alloc(
            (SMEM_PIPE_DEPTH_Q, BLK_M * cta_group, HEAD_DIM // cta_group), txl.f16, swizzle=txl.SW128B
        )
        # K and V share one ring: the loader alternates load_k / load_v into
        # successive stages. In the 2-CTA schedule K and V are [128, 64] per
        # CTA, with V consumed through the transposed descriptor view.
        # The backing allocation remains 96KB in both schedules.
        kv_base = SMEM_PIPE_DEPTH_Q * BLK_M * HEAD_DIM * F16_BYTES
        k_smem = smem.alloc((KV_DEPTH, BLK_N, HEAD_DIM // cta_group), txl.f16, swizzle=txl.SW128B)
        smem.pool.move_base_to(kv_base)
        v_smem = smem.alloc((KV_DEPTH, BLK_N, HEAD_DIM // cta_group), txl.f16, swizzle=txl.SW128B)
        smem.pool.move_base_to(kv_base + KV_DEPTH * KV_ROWS * HEAD_DIM * F16_BYTES)
        o_smem = smem.alloc((TMEM_DEPTH, BLK_M, HEAD_DIM), txl.f16, swizzle=txl.SW128B)

        # 16-byte units between two ring stages of a swizzled tile: a stage is a
        # constant translation of the same rows x cols tile.
        def stage16(tile):
            return tile.rows * tile.cols * tile.bits // 8 // 16

        Q_STAGE16 = stage16(q_smem)
        KV_STAGE16 = stage16(k_smem)
        assert KV_STAGE16 == stage16(v_smem)

        # How a descriptor is carried from its encode to the tcgen05.mma
        # operands, by regime. SPLIT_DESC keeps the two halves in plain C
        # scalars; the alternative is K's native uint64 descriptor plus
        # KDesc.__add__, which spells both the split and every step as inline
        # asm (mov.b64-unpack / add.u32 / mov.b64-pack).
        #
        # An asm result is an optimisation barrier, so under the native form
        # ptxas never sees one value flowing into the operands and reschedules
        # the descriptor chain across the whole kernel; the asm add also pins
        # the association to `lo + (stage*STAGE16 + koff)`, costing a ULEA per
        # k-phase because the `lo + stage*STAGE16` base can no longer be
        # hoisted out of the eight-phase walk (ULEA in the issuer region:
        # 15 -> 60). On the long non-causal GQA=1 stream that is worth -3.8%
        # (771.3 -> 742.3us on s8192_h32kv32, paired same-process; the
        # pre-rewrite kernel, which reached the halves through a C union,
        # measures 744.8us there). The GQA-packed causal path measures the
        # other way: with STEADY_DESC keeping eight descriptors live at once,
        # every plain-C spelling tried (split halves; split halves with an asm
        # pack; masking the halves out of the packed value at each use) lands
        # at 112.8-112.9us on s4096_h32kv4_causal against 110.8us for the
        # native form, so that regime keeps the native form. Elsewhere the
        # split is a win or a wash: s4096_h32kv32 218.0 -> 211.2,
        # s4096_h32kv32_causal 125.6 -> 122.9, s1024_h32kv32_causal
        # 26.38 -> 26.06, s8192_h32kv32_causal 424.4 -> 422.9,
        # s2048_h32kv8 60.12 -> 60.02, s1024_h32kv4 23.65 -> 23.85.
        SPLIT_DESC = not STEADY_DESC

        def lo_uniform(desc):
            """Broadcast the descriptor's low half — orig SmemDescriptor.make_lo_uniform.

            Returns whatever :func:`desc_at` wants in this regime: the split
            ``(lo, hi)`` halves, or the ``KDesc`` itself.
            """
            desc_lo = txl.alloc_local((1,), "uint32")
            desc_hi = txl.alloc_local((1,), "uint32")
            if not SPLIT_DESC:
                txl.ptx.mov.b64(desc_lo[0], desc_hi[0], desc.value)
                txl.ptx.mov.b64(desc.value, txl.uniform(desc_lo[0]), desc_hi[0])
                return desc
            txl.assign(desc_lo[0], txl.uniform(txl.Cast("uint32", desc.value)))
            txl.assign(desc_hi[0], txl.Cast("uint32", txl.shift_right(desc.value, txl.uint64(32))))
            return desc_lo, desc_hi

        def desc_at(desc, off16):
            """The descriptor for a 16-byte-unit offset from ``desc``'s base.

            Only the low half moves -- every offset here is a small
            non-negative displacement inside one tile ring, so nothing carries
            into the encoded layout fields -- and the high half is
            loop-invariant, so a step is one 32-bit add plus the pack into the
            operand register pair.
            """
            if not SPLIT_DESC:
                return desc + off16
            lo, hi = desc
            packed = txl.alloc_local((1,), "uint64")
            low = (
                lo[0] if isinstance(off16, int) and off16 == 0 else lo[0] + txl.Cast("uint32", off16)
            )
            txl.assign(
                packed[0],
                txl.bitwise_or(
                    txl.shift_left(txl.Cast("uint64", hi[0]), txl.uint64(32)), txl.Cast("uint64", low)
                ),
            )
            return packed[0]

        def encode(view, major="k", operand=None):
            """One hoisted, lo-uniform tcgen05 matrix descriptor for a stage-0 view.

            Returns ``(halves, off16)``. K derives ldo/sdo/swizzle from the tile it
            allocated and checks them against that layout; the frozen kernel
            hand-writes ldo=1024 sdo=64 swizzle=3 (orig:L635) and K derives
            exactly those three numbers. ``off16(kp)`` is the trace-time 16-byte
            offset of k-tile ``kp``, likewise derived from the layout: for these
            128x128 f16 128B-swizzle tiles it is ``(kp%4)*2 + (kp//4)*1024``
            K-major and ``kp*128`` MN-major -- the frozen kernel's hand-written
            ``ki//4*1024 + ki%4*2`` and ``ki*128`` to the bit.

            ``major`` does not change the emitted encode (the base address, LBO
            and SBO are the same tile either way); it only selects which axis
            ``off16`` walks. So all seven encodes below are the same instruction
            the original emits.
            """
            if USE_2CTA:
                # Each CTA holds one 64-wide slice of the group MMA operand.
                # All three upstream descriptors encode LBO=512/SBO=64, while
                # their K walks differ: Q crosses the second 128-row slab, K
                # crosses the second 64-row slab, and V walks 16 rows at a
                # time along its MN-major axis.
                desc_value = txl.alloc_local((1,), "uint64")
                txl.cuda.tcgen05.encode_matrix_descriptor(
                    txl.address_of(desc_value[0]),
                    view.ptr_to(0, 0),
                    ldo=512,
                    sdo=64,
                    swizzle=txl.SW128B.value,
                )
                desc_lo = txl.alloc_local((1,), "uint32")
                desc_hi = txl.alloc_local((1,), "uint32")
                txl.assign(desc_lo[0], txl.uniform(txl.Cast("uint32", desc_value[0])))
                txl.assign(desc_hi[0], txl.Cast("uint32", txl.shift_right(desc_value[0], txl.uint64(32))))

                if operand == "q":

                    def off16(kp):
                        return (kp % 4) * 2 + (kp // 4) * 1024

                elif operand == "k":

                    def off16(kp):
                        return (kp % 4) * 2 + (kp // 4) * 512

                else:

                    def off16(kp):
                        return kp * 128

                return (desc_lo, desc_hi), off16
            desc, off16 = view.encode(major=major, mma_k=MMA_K)
            return lo_uniform(desc), off16

        # orig:L634-658. k_desc and v_desc are numerically identical (K and V
        # alias); they are separate registers on purpose, and the five *_steady
        # / *_tail copies exist to shorten descriptor live ranges on the causal
        # GQA>1 path (orig's own comment at get_flash_attention4_kernel).
        q_desc, qoff = encode(q_smem[0], operand="q")
        k_desc, koff = encode(k_smem[0], operand="k")
        v_desc, mnoff = encode(v_smem[0], major="mn", operand="v")
        if STEADY_DESC:
            q_desc_steady, _ = encode(q_smem[0], operand="q")
            k_desc_steady, _ = encode(k_smem[0], operand="k")
            v_desc_steady_hi, _ = encode(v_smem[0], major="mn", operand="v")
            v_desc_tail_lo, _ = encode(v_smem[0], major="mn", operand="v")
            v_desc_tail_hi, _ = encode(v_smem[0], major="mn", operand="v")
        else:
            q_desc_steady = q_desc
            k_desc_steady = k_desc
            v_desc_steady_hi = v_desc
            v_desc_tail_lo = v_desc
            v_desc_tail_hi = v_desc

        sScale = smem.alloc((SSCALE_TOTAL_SIZE,), txl.f32, align=1024)
        tmem_addr = smem.alloc((1,), txl.u32)

        # ---- per-thread pipeline state — orig:L663-669 ----------------------
        # KV advances per ring slot. The other protocols unroll their two
        # physical slots, so their state owns only the epoch phase.
        kv_pipe = txl.PipelineState(KV_DEPTH, phase=0)
        softmax_epoch = txl.PipelineState(1, phase=0)
        score_epoch = txl.PipelineState(1, phase=0)
        tmem_epoch = txl.PipelineState(1, phase=0)
        q_epoch = txl.PipelineState(1, phase=0)
        o_epi_epoch = txl.PipelineState(1, phase=0)

        # ---- barrier rings — orig:L670-697 ----------------------------------
        q_load = txl.Pipeline(
            smem, SMEM_PIPE_DEPTH_Q, full="tma", empty="tcgen05", empty_phase_offset=1
        )
        kv_load = txl.Pipeline(smem, KV_DEPTH, full="tma", empty="tcgen05", empty_phase_offset=1)
        p_o_rescale = txl.MBarrier(smem, 2)
        p_o_rescale.init(256 * cta_group)
        # s_ready / o_ready are one-way tcgen05 signals: no slot is recycled, so
        # they are bare barriers rather than Pipelines. The producing side is
        # the matrix engine, so its arrive IS `tcgen05.commit` -- spelled as the
        # instruction it is (see `commit` below), the way the GDN port spells
        # its software arrive on a tcgen05-signalled ring.
        s_ready = txl.MBarrier(smem, 2)
        s_ready.init(1)
        o_ready = txl.MBarrier(smem, 2)
        o_ready.init(1)
        softmax_corr = txl.Pipeline(
            smem, 2, full="mbar", empty="mbar", init_full=128, init_empty=128, empty_phase_offset=1
        )
        corr_epi = txl.Pipeline(
            smem,
            TMEM_DEPTH,
            full="mbar",
            empty="mbar",
            init_full=128,
            init_empty=32,
            empty_phase_offset=1,
        )
        p_ready_2 = txl.MBarrier(smem, 2)
        p_ready_2.init(128 * cta_group)
        # Initialized in the FIRST init group so the single prologue fence
        # covers it, even though USE_S0_S1_BARRIER is off — orig:L692-696.
        bar_s0_s1_sequence = txl.MBarrier(smem, 8)
        bar_s0_s1_sequence.init(32)

        q_load_full_remote = q_load.full.remote_view(0) if USE_2CTA else q_load.full
        kv_load_full_remote = kv_load.full.remote_view(0) if USE_2CTA else kv_load.full
        p_o_rescale_remote = p_o_rescale.remote_view(0) if USE_2CTA else p_o_rescale
        p_ready_2_remote = p_ready_2.remote_view(0) if USE_2CTA else p_ready_2

        txl.ptx.fence.proxy.async_.shared__cta()
        txl.ptx.fence.mbarrier_init.release.cluster()
        if USE_2CTA:
            txl.cuda.cluster_sync()
        else:
            txl.cuda.cta_sync()

        # ---- shared closures -------------------------------------------------

        def elected():
            return txl.cuda.elect_sync() != txl.uint32(0)

        def commit(bar, stage):
            """The matrix engine's arrive on a one-way barrier — orig TCGen05Bar.arrive."""
            if USE_2CTA:
                txl.ptx[tcgen05_commit](bar.ptr_to([stage]), txl.Cast("uint16", txl.uint32(3)))
            else:
                txl.ptx[tcgen05_commit](bar.ptr_to([stage]))

        def tmem(col):
            """A tmem address. The kernel asserts its allocation base is 0 (below)
            and then names every column absolutely — orig:L752-755."""
            return txl.cuda.get_tmem_addr(txl.uint32(0), 0, col)

        def tmem_load(dst, dst_offset, tmem_col, width):
            chain = TMEM_LD_16 if width == 16 else TMEM_LD_32
            txl.ptx[chain](*(dst[dst_offset + i] for i in range(width)), tmem_col)

        def tmem_store(src, src_offset, tmem_col):
            txl.ptx[TMEM_ST_16](tmem_col, *(src[src_offset + i] for i in range(16)))

        def iket_range(name, *, leader_only=False):
            token = txl.alloc_local([1], "uint32")
            if leader_only:
                txl.assign(token[0], txl.cuda.iket.sentinel_token(name))
                with txl.If(warp_id == 0), txl.Then():
                    txl.assign(token[0], txl.cuda.iket.range_start(name))
            else:
                txl.assign(token[0], txl.cuda.iket.range_start(name))
            return token

        def cast_f32x2_f16x2(dst_u32, src, offset):
            """One packed f32x2 -> f16x2 conversion — orig:L95-97.

            Every consumer uses packed uint32 words, so keep that register view
            directly."""
            txl.ptx.cvt.rn.f16x2.f32(dst_u32[offset // 2], src[offset + 1], src[offset])

        def fma_f32x2(values, idx, multiplier, addend_value):
            """A packed f32x2 operand is one 64-bit value."""
            packed = txl.local_scalar("uint64")
            rhs = txl.local_scalar("uint64")
            addend = txl.local_scalar("uint64")
            txl.ptx.mov.b64(packed, values[idx], values[idx + 1])
            txl.ptx.mov.b64(rhs, multiplier, multiplier)
            txl.ptx.mov.b64(addend, addend_value, addend_value)
            txl.ptx.fma.rz.ftz.f32x2(packed, packed, rhs, addend)
            txl.ptx.mov.b64(values[idx], values[idx + 1], packed)

        def mul_f32x2(values, idx, multiplier):
            """orig:L215-223."""
            packed = txl.local_scalar("uint64")
            rhs = txl.local_scalar("uint64")
            txl.ptx.mov.b64(packed, values[idx], values[idx + 1])
            txl.ptx.mov.b64(rhs, multiplier, multiplier)
            txl.ptx.mul.rz.ftz.f32x2(packed, packed, rhs)
            txl.ptx.mov.b64(values[idx], values[idx + 1], packed)

        def reduce_max_128(out, values, accum=False):
            """SM100 three-input max tree — orig:L226-244. The outer walk stays a
            real serial loop (the original's txl.serial(15)); only the 4-wide
            inner fan is expanded, which is what txl.unroll(4) lowers to."""
            temp = txl.alloc_local([4], "float32")
            for i in range(4):
                if accum and i == 0:
                    txl.ptx[MAX3_F32](temp[i], values[2 * i], values[2 * i + 1], out[0])
                else:
                    txl.ptx.mov.b32(temp[i], txl.max(values[2 * i], values[2 * i + 1]))
            with txl.serial(15) as outer:
                for i in range(4):
                    txl.ptx[MAX3_F32](
                        temp[i],
                        temp[i],
                        values[8 * (outer + 1) + 2 * i],
                        values[8 * (outer + 1) + 2 * i + 1],
                    )
            txl.assign(out[0], txl.max(temp[0], temp[1]))
            txl.ptx[MAX3_F32](out[0], out[0], temp[2], temp[3])

        def reduce_sum_128(out, values, accum=False):
            """Packed add tree, accumulator insertion order preserved — orig:L247-276."""
            local_sum = txl.alloc_local([8], "float32")
            packed = txl.local_scalar("uint64")
            rhs = txl.local_scalar("uint64")
            for i in range(8):
                if accum and i == 0:
                    txl.ptx.mov.b32(local_sum[i], values[i] + out[0])
                else:
                    txl.ptx.mov.b32(local_sum[i], values[i])
            with txl.serial(15) as outer:
                for i in range(4):
                    txl.ptx.mov.b64(packed, local_sum[2 * i], local_sum[2 * i + 1])
                    txl.ptx.mov.b64(
                        rhs, values[8 * (outer + 1) + 2 * i], values[8 * (outer + 1) + 2 * i + 1]
                    )
                    txl.ptx.add.rn.ftz.f32x2(packed, packed, rhs)
                    txl.ptx.mov.b64(local_sum[2 * i], local_sum[2 * i + 1], packed)
            for lo, hi in ((0, 2), (4, 6), (0, 4)):
                txl.ptx.mov.b64(packed, local_sum[lo], local_sum[lo + 1])
                txl.ptx.mov.b64(rhs, local_sum[hi], local_sum[hi + 1])
                txl.ptx.add.rn.ftz.f32x2(packed, packed, rhs)
                txl.ptx.mov.b64(local_sum[lo], local_sum[lo + 1], packed)
            txl.assign(out[0], local_sum[0] + local_sum[1])

        def shl_u32_clamp(val, shift):
            """Left shift with PTX clamping (shift>=32 -> 0) — orig:L114-121."""
            result = txl.local_scalar("uint32")
            txl.ptx.shl.b32(result, val, shift)
            return result

        def combine_int_frac_ex2(x_rounded, frac_ex2):
            """orig:L124-135."""
            x_rounded_i = txl.local_scalar("int32")
            frac_ex_i = txl.local_scalar("int32")
            x_rounded_e = txl.local_scalar("int32")
            out_i = txl.local_scalar("int32")
            out = txl.local_scalar("float32")
            txl.ptx.mov.b32(x_rounded_i, x_rounded)
            txl.ptx.mov.b32(frac_ex_i, frac_ex2)
            txl.ptx.shl.b32(x_rounded_e, x_rounded_i, txl.uint32(23))
            txl.ptx.add.s32(out_i, x_rounded_e, frac_ex_i)
            txl.ptx.mov.b32(out, out_i)
            return out

        POLY_EX2_DEG3 = (1.0, 0.6951461434364319, 0.22756439447402954, 0.07711908966302872)
        FP32_ROUND_INT = float(2**23 + 2**22)

        def ex2_emulation_2(out, idx, x, y):
            """Two-lane polynomial ex2 on the packed f32x2 datapath — orig:L155-199."""
            xy_clamped = txl.alloc_local([2], "float32")
            txl.ptx.mov.b32(xy_clamped[0], txl.max(x, -127.0))
            txl.ptx.mov.b32(xy_clamped[1], txl.max(y, -127.0))
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
            txl.ptx.mov.b32(xy_frac_ex2[0], txl.float32(POLY_EX2_DEG3[3]))
            txl.ptx.mov.b32(xy_frac_ex2[1], txl.float32(POLY_EX2_DEG3[3]))
            for coeff in (POLY_EX2_DEG3[2], POLY_EX2_DEG3[1], POLY_EX2_DEG3[0]):
                txl.ptx.mov.b64(rhs, xy_frac[0], xy_frac[1])
                txl.ptx.mov.b64(packed, xy_frac_ex2[0], xy_frac_ex2[1])
                txl.ptx.mov.b64(addend, txl.float32(coeff), txl.float32(coeff))
                txl.ptx.fma.rz.ftz.f32x2(packed, packed, rhs, addend)
                txl.ptx.mov.b64(xy_frac_ex2[0], xy_frac_ex2[1], packed)
            txl.ptx.mov.b32(out[idx], combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0]))
            txl.ptx.mov.b32(out[idx + 1], combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1]))

        # ---- roles — orig:L752/765/1060/1396 ---------------------------------
        # Both role splits consume the complete 65536-register CTA file. The
        # pinned upstream FA4 schedule uses 192/72/56 for causal D128 on SM110.
        softmax_regs = 192 if thor_causal_d128 else 200
        correction_regs = 72 if thor_causal_d128 else 64
        other_regs = 56 if thor_causal_d128 else 48
        sp = txl.specialize(chain_dispatch=True)
        r_softmax = sp.role("softmax", warps=[0, 1, 2, 3, 4, 5, 6, 7], regs=softmax_regs)
        r_correction = sp.role("correction", warps=[8, 9, 10, 11], regs=correction_regs)
        wg3 = sp.warpgroup("wg3", warps=range(12, 16), regs=other_regs)
        r_mma = sp.role("mma", warps=[12], group=wg3, when=cta_rank == 0 if USE_2CTA else None)
        r_load = sp.role("load", warps=[13], group=wg3)
        r_store = sp.role("store", warps=[14], group=wg3)
        r_idle = sp.role("idle", warps=[15], group=wg3)

        # ---- scheduler and prologue — orig:L725-758 --------------------------
        scheduler = (
            txl.FlashAttentionLPTScheduler(
                "fa_scheduler",
                num_batches=BATCH_SIZE,
                num_heads=NUM_KV_HEADS,
                num_m_blocks=num_q_blocks,
                l2_swizzle=L2_SWIZZLE,
            )
            if is_causal
            else txl.FlashAttentionLinearScheduler(
                "fa_scheduler",
                num_batches=BATCH_SIZE,
                num_heads=NUM_KV_HEADS,
                num_m_blocks=num_q_blocks,
                num_ctas=scheduler_ctas,
            )
        )
        scheduler.init(cluster_id)
        # TMEM is allocated by the MMA warp and deliberately sits AFTER the
        # prologue cta_sync: every other warp's first TMEM access is
        # transitively gated behind this warp — orig:L698-751. These three
        # guards are on the raw warp id, not roles: the original runs them
        # once, outside the task loop, while the role blocks (and their
        # setmaxnreg) are inside it.
        with txl.If(warp_cta == 12), txl.Then():
            txl.ptx[tmem_alloc](txl.address_of(tmem_addr[0]), txl.uint32(N_COLS_TMEM))
            txl.cuda.warp_sync()
        with txl.If(tvm.tirx.all(wg_id == 3, warp_id == 0)), txl.Then():
            allocated = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(allocated, tmem_addr.ptr_to([0]))
            txl.cuda.trap_when_assert_failed(allocated == txl.uint32(0))
        with txl.If(wg_id == 2), txl.Then():
            for i_q in range(2):
                p_o_rescale_remote.arrive(i_q)

        # =====================================================================
        # Roles stay inside the task loop, preserving per-task setmaxnreg.
        # =====================================================================
        with txl.While(scheduler.valid()):
            m_block_idx = scheduler.m_block_idx
            batch_idx = scheduler.batch_idx
            kv_head_idx = scheduler.head_idx

            def q_tile_start(i_q):
                return (
                    m_block_idx * SMEM_PIPE_DEPTH_Q * cta_group + i_q * cta_group + cta_rank
                ) * SEQ_Q_PER_TILE

            # =================================================================
            # wg3 — sibling TMA-load, TMA-store, MMA, and idle roles.
            #                                                    orig:L765-1059
            # =================================================================
            # Adjacent role blocks let K specialize the mutually exclusive
            # register lifetimes without hiding any role-local instruction.
            # -------- warp 1: Q/K/V loader — orig:L768-856 -------------------
            with wg3:
                with r_load:

                    def load_q(i_q, tensor_map):
                        q_load.empty.wait(i_q, q_epoch.phase)
                        tma_q_token = iket_range("issue-tma-q")
                        with txl.If(elected()), txl.Then():
                            for k_part in range(cta_group):
                                if GQA_RATIO == 1:
                                    txl.ptx[tma_g2s_3d](
                                        q_smem[i_q].ptr_to(k_part * BLK_M, 0),
                                        txl.address_of(tensor_map),
                                        txl.int32(0),
                                        txl.Cast("int32", q_tile_start(i_q)),
                                        txl.Cast(
                                            "int32",
                                            (batch_idx * NUM_QO_HEADS + kv_head_idx) * 2
                                            + (k_part if USE_2CTA else 0),
                                        ),
                                        txl.cuda.cvta_generic_to_shared(
                                            q_load_full_remote.ptr_to([i_q])
                                        ),
                                    )
                                else:
                                    txl.ptx[tma_g2s_4d](
                                        q_smem[i_q].ptr_to(k_part * BLK_M, 0),
                                        txl.address_of(tensor_map),
                                        txl.int32(0),
                                        txl.Cast("int32", kv_head_idx * GQA_RATIO),
                                        txl.Cast("int32", q_tile_start(i_q)),
                                        txl.Cast(
                                            "int32", batch_idx * 2 + (k_part if USE_2CTA else 0)
                                        ),
                                        txl.cuda.cvta_generic_to_shared(
                                            q_load_full_remote.ptr_to([i_q])
                                        ),
                                    )
                            if USE_2CTA:
                                with txl.If(cta_rank == 0), txl.Then():
                                    q_load_full_remote.arrive(
                                        i_q, tx_count=cta_group * BLK_M * HEAD_DIM * F16_BYTES
                                    )
                            else:
                                q_load_full_remote.arrive(
                                    i_q, tx_count=cta_group * BLK_M * HEAD_DIM * F16_BYTES
                                )
                        txl.cuda.iket.range_end(tma_q_token[0])

                    def load_kv(i_kv, tensor_map, event, is_v=False):
                        """One K or V box into the shared KV ring — orig:L800-840.
                        The two loads are the same instruction on the same ring;
                        only the tensormap differs."""
                        kv_load.empty.wait(kv_pipe.stage, kv_pipe.phase)
                        tma_kv_token = iket_range(event)
                        with txl.If(elected()), txl.Then():
                            for k_part in range(1 if is_v else cta_group):
                                txl.ptx[tma_g2s_3d](
                                    (v_smem if is_v else k_smem)[kv_pipe.stage].ptr_to(
                                        0 if is_v else k_part * KV_ROWS, 0
                                    ),
                                    txl.address_of(tensor_map),
                                    txl.int32(0),
                                    txl.Cast(
                                        "int32",
                                        i_kv * BLK_N + cta_rank * KV_ROWS
                                        if not is_v
                                        else i_kv * BLK_N,
                                    ),
                                    txl.Cast(
                                        "int32",
                                        (batch_idx * NUM_KV_HEADS + kv_head_idx) * 2
                                        + (
                                            cta_rank
                                            if is_v and USE_2CTA
                                            else k_part
                                            if USE_2CTA
                                            else 0
                                        ),
                                    ),
                                    txl.cuda.cvta_generic_to_shared(
                                        kv_load_full_remote.ptr_to([kv_pipe.stage])
                                    ),
                                )
                            if USE_2CTA:
                                with txl.If(cta_rank == 0), txl.Then():
                                    kv_load_full_remote.arrive(
                                        kv_pipe.stage,
                                        tx_count=cta_group * KV_ROWS * HEAD_DIM * F16_BYTES,
                                    )
                            else:
                                kv_load_full_remote.arrive(
                                    kv_pipe.stage,
                                    tx_count=cta_group * KV_ROWS * HEAD_DIM * F16_BYTES,
                                )
                        txl.cuda.iket.range_end(tma_kv_token[0])
                        kv_pipe.advance()

                    load_trip_count = txl.local_scalar("int32")
                    txl.assign(
                        load_trip_count, n_block_max_of(m_block_idx) if is_causal else num_kv_blocks
                    )
                    load_q(0, Q_tensor_map)
                    load_kv(load_trip_count - 1, K_tensor_map, "issue-tma-k")
                    load_q(1, Q_tensor_map_1)
                    q_epoch.advance()
                    load_kv(load_trip_count - 1, V_tensor_map, "issue-tma-v", is_v=True)
                    with txl.serial(load_trip_count - 1, unroll=False) as _i:
                        i_kv = load_trip_count - 2 - _i
                        load_kv(i_kv, K_tensor_map_1, "issue-tma-k")
                        load_kv(i_kv, V_tensor_map_1, "issue-tma-v", is_v=True)

                # -------- warp 2: O store — orig:L857-895 ------------------------
                with r_store:
                    corr_epi.full.wait(0, tmem_epoch.phase)
                    tma_store_token = iket_range("tma-store")
                    for i_q in range(SMEM_PIPE_DEPTH_Q):
                        if i_q != 0:
                            corr_epi.full.wait(i_q, tmem_epoch.phase)
                        m_start_global = q_tile_start(i_q)
                        with txl.If(elected()), txl.Then():
                            if GQA_RATIO == 1:
                                txl.ptx[TMA_S2G_3D](
                                    txl.address_of(O_tensor_map),
                                    txl.int32(0),
                                    txl.Cast("int32", m_start_global),
                                    txl.Cast("int32", (batch_idx * NUM_QO_HEADS + kv_head_idx) * 2),
                                    o_smem[i_q].ptr_to(0, 0),
                                )
                            else:
                                txl.ptx[TMA_S2G_4D](
                                    txl.address_of(O_tensor_map),
                                    txl.int32(0),
                                    txl.Cast("int32", kv_head_idx * GQA_RATIO),
                                    txl.Cast("int32", m_start_global),
                                    txl.Cast("int32", batch_idx * 2),
                                    o_smem[i_q].ptr_to(0, 0),
                                )
                        txl.ptx.cp.async_.bulk.commit_group()
                    # The wait count lives in the instruction text, so it is a
                    # literal per stage rather than a loop variable.
                    txl.ptx.cp.async_.bulk.wait_group(1)
                    corr_epi.empty.arrive(0)
                    txl.ptx.cp.async_.bulk.wait_group(0)
                    corr_epi.empty.arrive(1)
                    txl.cuda.iket.range_end(tma_store_token[0])
                    tmem_epoch.advance()

                # -------- warp 0: MMA issuer — orig:L896-1059 --------------------
                with r_mma:
                    acc = txl.local_scalar("int32", init=0)

                    def gemm_qk(q_stage, kv_stage, qd, kd):
                        """S[q_stage] = Q[q_stage] @ K[kv_stage]^T — orig:L900-920.

                        Eight k-phases of one hoisted descriptor plus a
                        trace-time 16B offset. K derives that offset walk from
                        the tile's own layout: for this 128x128 f16 128B-swizzle
                        tile it is (ki%4)*2 + (ki//4)*1024, which is the frozen
                        kernel's hand-written constant to the bit."""
                        for ki in range(HEAD_DIM // MMA_K):
                            with txl.If(elected()), txl.Then():
                                txl.ptx[mma_f16](
                                    txl.Cast("uint32", q_stage * MMA_N),
                                    desc_at(qd, q_stage * Q_STAGE16 + qoff(ki)),
                                    desc_at(kd, kv_stage * KV_STAGE16 + koff(ki)),
                                    txl.uint32(id_qk),
                                    *mma_keep_lanes,
                                    ki != 0,
                                )
                        with txl.If(elected()), txl.Then():
                            commit(s_ready, q_stage)

                    def gemm_pv_part1(i_q, kv_stage, should_accumulate, vd):
                        """orig:L932-946. A is a tmem address (ts form); B is V,
                        MN-major (ID_PV sets trans_b), so the k walk steps whole
                        rows: 16 rows = 128 16B units."""
                        for ki in range(K_SPLIT // MMA_K):
                            with txl.If(elected()), txl.Then():
                                txl.ptx[mma_f16](
                                    txl.Cast("uint32", (SMEM_PIPE_DEPTH_Q + i_q) * MMA_N),
                                    txl.Cast("uint32", i_q * MMA_N + MMA_N // 2 + ki * (MMA_K // 2)),
                                    desc_at(vd, kv_stage * KV_STAGE16 + mnoff(ki)),
                                    txl.uint32(id_pv),
                                    *mma_keep_lanes,
                                    # any(ki != 0, should_accumulate), folded at
                                    # trace time: ki is a Python int here.
                                    True if ki != 0 else txl.Cast("bool", should_accumulate),
                                )

                    def gemm_pv_part2(i_q, kv_stage, vd):
                        """orig:L948-968."""
                        p_ready_2.wait(i_q, tmem_epoch.phase)
                        for ki in range((BLK_N - K_SPLIT) // MMA_K):
                            with txl.If(elected()), txl.Then():
                                txl.ptx[mma_f16](
                                    txl.Cast("uint32", (SMEM_PIPE_DEPTH_Q + i_q) * MMA_N),
                                    txl.Cast(
                                        "uint32",
                                        i_q * MMA_N + MMA_N // 2 + K_SPLIT // 2 + ki * (MMA_K // 2),
                                    ),
                                    desc_at(
                                        vd, kv_stage * KV_STAGE16 + mnoff(K_SPLIT // MMA_K + ki)
                                    ),
                                    txl.uint32(id_pv),
                                    *mma_keep_lanes,
                                    True,
                                )

                    def gemm_pv(i_q, kv_stage, should_accumulate, v_lo, v_hi):
                        gemm_pv_part1(i_q, kv_stage, should_accumulate, v_lo)
                        gemm_pv_part2(i_q, kv_stage, v_hi)

                    # prologue: both Q stages against the first K — orig:L975-983
                    for i_q in range(SMEM_PIPE_DEPTH_Q):
                        q_load.full.wait(i_q, q_epoch.phase)
                        if i_q == 0:
                            kv_load.full.wait(kv_pipe.stage, kv_pipe.phase)
                        gemm_qk(i_q, kv_pipe.stage, q_desc, k_desc)
                        if i_q == 1:
                            with txl.If(elected()), txl.Then():
                                kv_load.empty.arrive(
                                    kv_pipe.stage,
                                    cta_group=cta_group,
                                    cta_mask=3 if USE_2CTA else None,
                                )
                    kv_pipe.advance()

                    mma_trip_count = txl.local_scalar("int32")
                    txl.assign(
                        mma_trip_count, n_block_max_of(m_block_idx) if is_causal else num_kv_blocks
                    )
                    with txl.serial(mma_trip_count - 1, unroll=False) as i_kv:
                        # stage_v / phase_v are snapshots: kv_pipe.stage is a
                        # mutable Var and advance() rewrites it, so holding the
                        # expression across the advance would name the wrong
                        # stage (the KTileView lifetime rule, in int form).
                        # `txl.local_scalar` is the snapshot: it must copy the
                        # cursor now, because a plain value would re-read the
                        # advanced one below.
                        # stage_k below is deliberately the LIVE expression --
                        # the original's txl.meta_var -- and no advance intervenes
                        # before its last use.
                        stage_v = txl.local_scalar("int32", init=kv_pipe.stage)
                        phase_v = txl.local_scalar("int32", init=kv_pipe.phase)
                        kv_pipe.advance()
                        for i_q in range(SMEM_PIPE_DEPTH_Q):
                            if i_q == 0:
                                kv_load.full.wait(stage_v, phase_v)
                            p_o_rescale.wait(i_q, tmem_epoch.phase)
                            gemm_pv(i_q, stage_v, acc, v_desc, v_desc_steady_hi)
                            if i_q == 1:
                                with txl.If(elected()), txl.Then():
                                    kv_load.empty.arrive(
                                        stage_v,
                                        cta_group=cta_group,
                                        cta_mask=3 if USE_2CTA else None,
                                    )
                            if i_q == 0:
                                kv_load.full.wait(kv_pipe.stage, kv_pipe.phase)
                            gemm_qk(i_q, kv_pipe.stage, q_desc_steady, k_desc_steady)
                            # Early Q release — orig:L1018-1030.
                            if EARLY_Q_RELEASE:
                                with txl.If(i_kv == mma_trip_count - 2), txl.Then():
                                    with txl.If(elected()), txl.Then():
                                        q_load.empty.arrive(
                                            i_q,
                                            cta_group=cta_group,
                                            cta_mask=3 if USE_2CTA else None,
                                        )
                            if i_q == 1:
                                with txl.If(elected()), txl.Then():
                                    kv_load.empty.arrive(
                                        kv_pipe.stage,
                                        cta_group=cta_group,
                                        cta_mask=3 if USE_2CTA else None,
                                    )
                        txl.assign(acc, 1)
                        kv_pipe.advance()
                        tmem_epoch.advance()
                    # tail PV — orig:L1037-1053
                    for i_q in range(SMEM_PIPE_DEPTH_Q):
                        if i_q == 0:
                            kv_load.full.wait(kv_pipe.stage, kv_pipe.phase)
                        p_o_rescale.wait(i_q, tmem_epoch.phase)
                        gemm_pv(i_q, kv_pipe.stage, acc, v_desc_tail_lo, v_desc_tail_hi)
                        if i_q == 1:
                            with txl.If(elected()), txl.Then():
                                kv_load.empty.arrive(
                                    kv_pipe.stage,
                                    cta_group=cta_group,
                                    cta_mask=3 if USE_2CTA else None,
                                )
                        with txl.If(elected()), txl.Then():
                            commit(o_ready, i_q)
                    kv_pipe.advance()
                    tmem_epoch.advance()
                    if not EARLY_Q_RELEASE:
                        for i_q in range(SMEM_PIPE_DEPTH_Q):
                            with txl.If(elected()), txl.Then():
                                q_load.empty.arrive(
                                    i_q, cta_group=cta_group, cta_mask=3 if USE_2CTA else None
                                )
                    q_epoch.advance()

                with r_idle:
                    pass

            # =================================================================
            # softmax warpgroups 0 and 1 — orig:L1060-1395
            # =================================================================
            with r_softmax:
                row_max = txl.local_scalar("float32")
                row_sum = txl.alloc_local([1], "float32")
                with txl.If(warp_id == 0), txl.Then():
                    txl.cuda.iket.mark("softmax-baseline")

                def mask_r2p(s_chunk, col_limit, ncol):
                    """R2P-style column mask — orig:L1069-1103.

                    ~(0xFFFFFFFF << k) is the low-k-bits mask with no `-1`; the
                    ~ fuses into the per-column `& (1<<i)` test as ANDN, and shl
                    clamping (k>=32 -> 0) removes the min. The bit test compiles
                    to R2P."""
                    CHUNK_SIZE = 32
                    for s in range(ceildiv(ncol, CHUNK_SIZE)):
                        k_keep = txl.max(col_limit - s * CHUNK_SIZE, 0)
                        mask_inv = txl.local_scalar("uint32")
                        txl.assign(
                            mask_inv, shl_u32_clamp(txl.uint32(0xFFFFFFFF), txl.Cast("uint32", k_keep))
                        )
                        for i in range(CHUNK_SIZE):
                            if i < ncol - s * CHUNK_SIZE:
                                c = s * CHUNK_SIZE + i
                                in_bound = txl.bitwise_and(
                                    txl.bitwise_not(mask_inv), txl.shift_left(txl.uint32(1), txl.uint32(i))
                                )
                                txl.ptx.mov.b32(
                                    s_chunk[c],
                                    txl.Select(
                                        txl.Cast("bool", in_bound), s_chunk[c], txl.float32(NEG_INF)
                                    ),
                                )

                def apply_causal_mask(s_chunk, m_blk_idx, n_blk_idx):
                    """orig:L1105-1129. `col_limit_right` is read four times
                    (once per 32-column mask chunk); the intermediates are plain
                    values, so ptxas sees the row/offset arithmetic directly."""
                    seq_pos_in_wg = tid_in_wg // GQA_RATIO
                    row_idx = (
                        m_blk_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
                        + wg_id * SEQ_Q_PER_TILE
                        + seq_pos_in_wg
                    )
                    causal_row_offset = 1 + SEQ_LEN_KV - n_blk_idx * BLK_N - SEQ_LEN_Q
                    col_limit_right = row_idx + causal_row_offset
                    mask_r2p(s_chunk, col_limit_right, BLK_N)

                def softmax_step(i_kv, apply_mask=False, is_first=False):
                    """One KV block of the online softmax — orig:L1131-1294."""
                    s_chunk = txl.alloc_local([BLK_N], "float32")
                    # Only the uint32 view of the packed-f16 P fragment is ever
                    # read (tcgen05.st operands), so allocate that directly.
                    p_chunk = txl.alloc_local([BLK_N // 2], "uint32")
                    s_ready.wait(wg_id, score_epoch.phase)
                    with txl.If(warp_id == 0), txl.Then():
                        txl.cuda.iket.mark("softmax-phase-0")
                    softmax_max_token = iket_range("softmax-max", leader_only=True)
                    tile_max = txl.alloc_local([1], "float32")
                    for chunk_idx in range(BLK_N // SOFTMAX_LD_CHUNK):
                        tmem_load(
                            s_chunk,
                            chunk_idx * SOFTMAX_LD_CHUNK,
                            tmem(wg_id * MMA_N + chunk_idx * SOFTMAX_LD_CHUNK),
                            SOFTMAX_LD_CHUNK,
                        )
                    if apply_mask:
                        apply_causal_mask(s_chunk, m_block_idx, i_kv)
                    row_max_old = txl.local_scalar("float32")
                    if is_first:
                        reduce_max_128(tile_max, s_chunk)
                    else:
                        # row_max is initialized by the first step; its load
                        # stays inside the non-first specialization.
                        txl.assign(row_max_old, row_max)
                        txl.assign(tile_max[0], row_max_old)
                        reduce_max_128(tile_max, s_chunk, accum=True)
                    row_max_new = txl.local_scalar("float32")
                    acc_scale = txl.local_scalar("float32")
                    acc_scale_ = txl.local_scalar("float32")
                    row_max_safe = txl.local_scalar("float32")
                    txl.assign(row_max_new, tile_max[0])
                    txl.assign(
                        row_max_safe,
                        txl.if_then_else(
                            tile_max[0] == txl.float32(NEG_INF), txl.float32(0.0), tile_max[0]
                        ),
                    )
                    if is_first:
                        txl.assign(acc_scale, txl.float32(1.0))
                    else:
                        txl.assign(acc_scale_, (row_max_old - row_max_safe) * scale_log2)
                        with txl.If(acc_scale_ >= -rescale_threshold):
                            with txl.Then():
                                txl.assign(row_max_new, row_max_old)
                                txl.assign(row_max_safe, row_max_old)
                                txl.assign(acc_scale, txl.float32(1.0))
                            with txl.Else():
                                txl.ptx.ex2.approx.ftz.f32(acc_scale, acc_scale_)
                    txl.assign(row_max, row_max_new)
                    row_max_scaled = row_max_safe * scale_log2
                    with txl.If(warp_id == 0), txl.Then():
                        txl.cuda.iket.mark("softmax-phase-1")
                    txl.cuda.iket.range_end(softmax_max_token[0])
                    if not is_first:
                        with txl.If(tid_in_wg < BLK_M), txl.Then():
                            sScale_idx = ACC_SCALE_BASE + tid_in_wg + wg_id * BLK_M
                            txl.ptx.st.shared.f32(sScale.ptr_to([sScale_idx]), acc_scale)
                    # Stats-ready handshake to the correction wg over a HW named
                    # barrier — orig:L1192-1203.
                    if STATS_BAR_PAIRWISE:
                        txl.ptx.bar.arrive(txl.Cast("uint32", 1 + wg_id * 4 + warp_id), 64)
                    else:
                        txl.ptx.bar.arrive(txl.Cast("uint32", 1 + wg_id), 256)
                    softmax_fma_token = iket_range("softmax-fma", leader_only=True)
                    for i in range(BLK_N // 2):
                        fma_f32x2(s_chunk, 2 * i, txl.float32(scale_log2), -row_max_scaled)
                    txl.cuda.iket.range_end(softmax_fma_token[0])
                    softmax_exp2_token = iket_range("softmax-exp2", leader_only=True)
                    emu_pairs = EMU_PAIRS_CAUSAL if is_causal else EMU_PAIRS_NC
                    emu_start = (
                        1 if thor_causal_d128 else EMU_START_CAUSAL if is_causal else EMU_START_NC
                    )
                    for frag_idx in range(4):
                        for i in range(BLK_N // 4 // 2):
                            idx = frag_idx * BLK_N // 4 + 2 * i
                            if (
                                i * 2 % 16 < 16 - 2 * emu_pairs
                                or frag_idx >= 4 - 1
                                or frag_idx < emu_start
                                or apply_mask
                            ):
                                txl.ptx.ex2.approx.ftz.f32(s_chunk[idx], s_chunk[idx])
                                txl.ptx.ex2.approx.ftz.f32(s_chunk[idx + 1], s_chunk[idx + 1])
                            else:
                                ex2_emulation_2(s_chunk, idx, s_chunk[idx], s_chunk[idx + 1])
                        for i in range(BLK_N // 4 // 2):
                            idx = frag_idx * BLK_N // 4 + 2 * i
                            cast_f32x2_f16x2(p_chunk, s_chunk, idx)
                    txl.cuda.iket.range_end(softmax_exp2_token[0])
                    softmax_tmem_st_token = iket_range("softmax-tmem-st", leader_only=True)
                    for i in range(P_SPLIT_Q):
                        tmem_store(
                            p_chunk,
                            i * BLK_N // 4 // 2,
                            tmem((wg_id * 2 * MMA_N + MMA_N + i * BLK_N // 4) // 2),
                        )
                    txl.ptx.tcgen05.wait__st.sync.aligned()
                    p_o_rescale_remote.arrive(wg_id)
                    for i in range(4 - P_SPLIT_Q):
                        tmem_store(
                            p_chunk,
                            (P_SPLIT_Q + i) * BLK_N // 4 // 2,
                            tmem((wg_id * 2 * MMA_N + MMA_N + (P_SPLIT_Q + i) * BLK_N // 4) // 2),
                        )
                    with txl.If(warp_id == 0), txl.Then():
                        txl.cuda.iket.mark("softmax-phase-2")
                    txl.ptx.tcgen05.wait__st.sync.aligned()
                    p_ready_2_remote.arrive(wg_id)
                    with txl.If(warp_id == 0), txl.Then():
                        txl.cuda.iket.mark("softmax-phase-3")
                    txl.cuda.iket.range_end(softmax_tmem_st_token[0])
                    softmax_corr.empty.wait(wg_id, softmax_epoch.phase)
                    with txl.If(warp_id == 0), txl.Then():
                        txl.cuda.iket.mark("softmax-phase-4")
                    softmax_sum_token = iket_range("softmax-sum", leader_only=True)
                    score_epoch.advance()
                    softmax_epoch.advance()
                    if is_first:
                        reduce_sum_128(row_sum, s_chunk)
                    else:
                        txl.assign(row_sum[0], row_sum[0] * acc_scale)
                        reduce_sum_128(row_sum, s_chunk, accum=True)
                    with txl.If(warp_id == 0), txl.Then():
                        txl.cuda.iket.mark("softmax-phase-5")
                    txl.cuda.iket.range_end(softmax_sum_token[0])

                # Pre-loop empty.wait guards this task's first sScale write
                # against the PREVIOUS task's tail row_sum read — orig:L1296-1311.
                if not EPI_ON_SOFTMAX:
                    softmax_corr.empty.wait(wg_id, softmax_epoch.phase)
                softmax_epoch.advance()

                # These were `txl.Bind` (one emitted binding, handed back as a Var)
                # because re-emitting the expression at every use once cost 1.3%
                # at the median and 7.3% on s1024_h32kv4_causal: `n_block_max`
                # reaches the innermost mask chunk, so the mask shift amount
                # recomputed `min(n_kv, m_block*2+2)` per masked step.
                # As plain values the block bounds now fold at trace time for
                # every specialization measured, and the SASS instruction count
                # is unchanged on s1024_h32kv4, s1024_h32kv4_causal,
                # s4096_h32kv4_causal and s8192_h32kv32 (schedule differs).
                n_block_max = n_block_max_of(m_block_idx)
                n_block_min_causal = (
                    n_block_min_causal_of(m_block_idx) if is_causal else n_block_max
                )
                softmax_step(n_block_max - 1, apply_mask=is_causal, is_first=True)
                n_block_max_after_p1 = n_block_max - 1
                num_phase2_blocks = txl.max(n_block_max_after_p1 - n_block_min_causal, 0)
                with txl.serial(num_phase2_blocks, unroll=False) as i:
                    softmax_step(n_block_max_after_p1 - 1 - i, apply_mask=True)
                n_block_max_after_p2 = txl.min(n_block_max_after_p1, n_block_min_causal)
                with txl.serial(n_block_max_after_p2, unroll=False) as i:
                    softmax_step(n_block_max_after_p2 - 1 - i, apply_mask=False)

                if EPI_ON_SOFTMAX:
                    # Stage-parallel epilogue on this wg — orig:L1330-1386.
                    EPI_LD_SM = 32
                    o_ready.wait(wg_id, o_epi_epoch.phase)
                    corr_epi.empty.wait(wg_id, o_epi_epoch.phase)
                    epi_ld_tmem_token = iket_range("epi-ld-tmem", leader_only=True)
                    acc_O_row_is_zero_or_nan = tvm.tirx.any(
                        row_sum[0] == txl.float32(0.0), row_sum[0] != row_sum[0]
                    )
                    norm_scale_sm = txl.local_scalar("float32")
                    txl.ptx.rcp.approx.ftz.f32(
                        norm_scale_sm,
                        txl.Select(acc_O_row_is_zero_or_nan, txl.float32(1.0), row_sum[0]),
                    )
                    o_row_f32_sm = txl.alloc_local([EPI_LD_SM], "float32")
                    o_row_f16_sm = txl.alloc_local([EPI_LD_SM // 2], "uint32")
                    for epi_q in range(2):
                        with txl.If(wg_id == epi_q), txl.Then():
                            for d_tile in range(ceildiv(HEAD_DIM, EPI_LD_SM)):
                                d_start = d_tile * EPI_LD_SM
                                tmem_load(
                                    o_row_f32_sm,
                                    0,
                                    tmem((SMEM_PIPE_DEPTH_Q + epi_q) * MMA_N + d_start),
                                    EPI_LD_SM,
                                )
                                for i in range(EPI_LD_SM // 2):
                                    mul_f32x2(o_row_f32_sm, 2 * i, norm_scale_sm)
                                for i in range(EPI_LD_SM // 2):
                                    cast_f32x2_f16x2(o_row_f16_sm, o_row_f32_sm, 2 * i)
                                for i in range(EPI_LD_SM // 8):
                                    txl.ptx.st.shared.v4.u32(
                                        o_smem[epi_q].ptr_to(tid_in_wg, d_start + i * 8),
                                        o_row_f16_sm[i * 4],
                                        o_row_f16_sm[i * 4 + 1],
                                        o_row_f16_sm[i * 4 + 2],
                                        o_row_f16_sm[i * 4 + 3],
                                    )
                    txl.cuda.iket.range_end(epi_ld_tmem_token[0])
                    txl.ptx.fence.proxy.async_.shared__cta()
                    corr_epi.full.arrive(wg_id)
                    p_o_rescale_remote.arrive(wg_id)
                    o_epi_epoch.advance()
                else:
                    with txl.If(tid_in_wg < BLK_M), txl.Then():
                        txl.ptx.st.shared.f32(
                            sScale.ptr_to([ROW_SUM_BASE + tid_in_wg + wg_id * BLK_M]), row_sum[0]
                        )
                    if STATS_BAR_PAIRWISE:
                        txl.ptx.bar.arrive(txl.Cast("uint32", 1 + wg_id * 4 + warp_id), 64)
                    else:
                        txl.ptx.bar.arrive(txl.Cast("uint32", 1 + wg_id), 256)

            # =================================================================
            # correction warpgroup 2 — orig:L1396-1528
            # =================================================================
            with r_correction:

                def stats_sync(i_q):
                    if STATS_BAR_PAIRWISE:
                        txl.ptx.bar.sync(txl.Cast("uint32", 1 + i_q * 4 + warp_id), 64)
                    else:
                        txl.ptx.bar.sync(txl.uint32(1 + i_q), 256)

                stats_sync(0)
                softmax_corr.empty.arrive(0)
                stats_sync(1)
                softmax_epoch.advance()
                corr_trip_count = n_block_max_of(m_block_idx) if is_causal else num_kv_blocks
                with txl.serial(corr_trip_count - 1, unroll=False) as _i_kv:
                    for i_q in range(2):
                        stats_sync(i_q)
                        correction_token = iket_range("correction", leader_only=True)
                        acc_scale = txl.local_scalar("float32")
                        should_rescale = txl.local_scalar("int32")
                        with txl.If(tid_in_wg < BLK_M):
                            with txl.Then():
                                txl.ptx.ld.shared.f32(
                                    acc_scale,
                                    sScale.ptr_to([ACC_SCALE_BASE + tid_in_wg + i_q * BLK_M]),
                                )
                                txl.assign(should_rescale, txl.Select(acc_scale < txl.float32(1.0), 1, 0))
                            with txl.Else():
                                txl.assign(should_rescale, 0)
                        # Materialize the collective before divergence.
                        any_needs_rescale = txl.local_scalar("uint32")
                        txl.ptx.vote_sync.any.pred(
                            any_needs_rescale, txl.ptx.pred(should_rescale), txl.uint32(0xFFFFFFFF)
                        )
                        with txl.If(any_needs_rescale != 0), txl.Then():
                            with txl.If(tid_in_wg < BLK_M), txl.Then():
                                RESCALE_TILE = 16
                                o_row = txl.alloc_local([RESCALE_TILE], "float32")
                                for d_tile in range(ceildiv(HEAD_DIM, RESCALE_TILE)):
                                    d_start = d_tile * RESCALE_TILE
                                    if d_start < HEAD_DIM:
                                        addr = tmem((SMEM_PIPE_DEPTH_Q + i_q) * MMA_N + d_start)
                                        tmem_load(o_row, 0, addr, RESCALE_TILE)
                                        for i in range(RESCALE_TILE // 2):
                                            mul_f32x2(o_row, 2 * i, acc_scale)
                                        tmem_store(o_row, 0, addr)
                                txl.ptx.tcgen05.wait__st.sync.aligned()
                        p_o_rescale_remote.arrive(i_q)
                        softmax_corr.empty.arrive(1 - i_q)
                        txl.cuda.iket.range_end(correction_token[0])
                    softmax_epoch.advance()
                softmax_corr.empty.arrive(1)
                if not EPI_ON_SOFTMAX:
                    for i_q in range(2):
                        stats_sync(i_q)
                        row_sum_c = txl.local_scalar("float32")
                        txl.ptx.ld.shared.f32(
                            row_sum_c, sScale.ptr_to([ROW_SUM_BASE + tid_in_wg + i_q * BLK_M])
                        )
                        softmax_corr.empty.arrive(i_q)
                        o_ready.wait(i_q, tmem_epoch.phase)
                        corr_epi.empty.wait(i_q, tmem_epoch.phase)
                        epi_ld_tmem_token = iket_range("epi-ld-tmem", leader_only=True)
                        zero_or_nan = tvm.tirx.any(
                            row_sum_c == txl.float32(0.0), row_sum_c != row_sum_c
                        )
                        norm_scale = txl.local_scalar("float32")
                        txl.ptx.rcp.approx.ftz.f32(
                            norm_scale, txl.Select(zero_or_nan, txl.float32(1.0), row_sum_c)
                        )
                        o_row_f32 = txl.alloc_local([TMEM_EPI_LD_SIZE], "float32")
                        o_row_f16 = txl.alloc_local([TMEM_EPI_LD_SIZE // 2], "uint32")
                        for d_tile in range(ceildiv(HEAD_DIM, TMEM_EPI_LD_SIZE)):
                            d_start = d_tile * TMEM_EPI_LD_SIZE
                            if d_start < HEAD_DIM:
                                tmem_load(
                                    o_row_f32,
                                    0,
                                    tmem((SMEM_PIPE_DEPTH_Q + i_q) * MMA_N + d_start),
                                    TMEM_EPI_LD_SIZE,
                                )
                                for i in range(TMEM_EPI_LD_SIZE // 2):
                                    mul_f32x2(o_row_f32, 2 * i, norm_scale)
                                for i in range(TMEM_EPI_LD_SIZE // 2):
                                    cast_f32x2_f16x2(o_row_f16, o_row_f32, 2 * i)
                                for i in range(TMEM_EPI_LD_SIZE // 8):
                                    txl.ptx.st.shared.v4.u32(
                                        o_smem[i_q].ptr_to(tid_in_wg, d_start + i * 8),
                                        o_row_f16[i * 4],
                                        o_row_f16[i * 4 + 1],
                                        o_row_f16[i * 4 + 2],
                                        o_row_f16[i * 4 + 3],
                                    )
                        txl.cuda.iket.range_end(epi_ld_tmem_token[0])
                        txl.ptx.fence.proxy.async_.shared__cta()
                        corr_epi.full.arrive(i_q)
                        p_o_rescale_remote.arrive(i_q)
                    tmem_epoch.advance()
                softmax_epoch.advance()

            scheduler.next_tile()

        # Match the allocation scope before TMEM teardown.
        if USE_2CTA:
            txl.cuda.cluster_sync()
        else:
            txl.cuda.cta_sync()
        with txl.If(tvm.tirx.all(wg_id == 0, warp_id == 0)), txl.Then():
            dealloc = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(dealloc, tmem_addr.ptr_to([0]))
            txl.ptx[tmem_relinquish]()
            txl.ptx[tmem_dealloc](dealloc, txl.uint32(N_COLS_TMEM))

    return flash_attention4


class _AlignedTensorMap:
    """Host storage for one 64-byte-aligned, 128-byte TensorMap payload."""

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
        "float16",
        rank,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *dims,
        *strides,
        *box,
        *((1,) * rank),
        0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        3,  # CU_TENSOR_MAP_SWIZZLE_128B
        2,  # CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    )
    return desc


def build_tensor_maps(
    Q,
    Kt,
    V,
    O,
    *,
    batch_size,
    seq_len_q,
    seq_len_kv,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    is_causal=False,
):
    """The seven maps, in the order ``make_kernel``'s parameters declare them."""
    gqa = num_qo_heads // num_kv_heads
    seq_q_per_tile = BLK_M // gqa
    use_2cta = os.environ.get(PREPARE_CUDA_ARCH_ENV, "sm_100a") == "sm_110a" and not is_causal
    cta_group = 2 if use_2cta else 1

    def q_map(t):
        if gqa == 1:
            return _encode(
                t,
                (head_dim // 2, seq_len_q, batch_size * num_qo_heads * 2),
                (num_qo_heads * head_dim * F16_BYTES, head_dim),
                (head_dim // 2, seq_q_per_tile, 1)
                if use_2cta
                else (head_dim // 2, seq_q_per_tile, 2),
            )
        return _encode(
            t,
            (head_dim // 2, num_qo_heads, seq_len_q, batch_size * 2),
            (head_dim * F16_BYTES, num_qo_heads * head_dim * F16_BYTES, head_dim),
            (head_dim // 2, gqa, seq_q_per_tile, 1)
            if use_2cta
            else (head_dim // 2, gqa, seq_q_per_tile, 2),
        )

    def o_map(t):
        if gqa == 1:
            return _encode(
                t,
                (head_dim // 2, seq_len_q, batch_size * num_qo_heads * 2),
                (num_qo_heads * head_dim * F16_BYTES, head_dim),
                (head_dim // 2, seq_q_per_tile, 2),
            )
        return _encode(
            t,
            (head_dim // 2, num_qo_heads, seq_len_q, batch_size * 2),
            (head_dim * F16_BYTES, num_qo_heads * head_dim * F16_BYTES, head_dim),
            (head_dim // 2, gqa, seq_q_per_tile, 2),
        )

    def k_map(t):
        return _encode(
            t,
            (head_dim // 2, seq_len_kv, batch_size * num_kv_heads * 2),
            (num_kv_heads * head_dim * F16_BYTES, head_dim),
            (head_dim // 2, BLK_N // cta_group, 1) if use_2cta else (head_dim // 2, BLK_N, 2),
        )

    def v_map(t):
        return _encode(
            t,
            (head_dim // 2, seq_len_kv, batch_size * num_kv_heads * 2),
            (num_kv_heads * head_dim * F16_BYTES, head_dim),
            (head_dim // 2, BLK_N, 1) if use_2cta else (head_dim // 2, BLK_N, 2),
        )

    return (q_map(Q), q_map(Q), k_map(Kt), k_map(Kt), v_map(V), v_map(V), o_map(O))


def _select_reg_level(
    batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim, is_causal
):
    """Select the measured non-spilling ptxas regime for this K body."""
    override = os.environ.get("FA4_REG_LEVEL", "")
    if override:
        return override
    if is_causal and num_qo_heads == num_kv_heads:
        return "2"
    if is_causal:  # GQA-packed causal
        return "5" if seq_len_q <= 1024 else "6"
    return "10"


def get_flash_attention4_kernel(
    batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim, is_causal=False
):
    os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = _select_reg_level(
        batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim, is_causal
    )
    prepare_arch = os.environ.get(PREPARE_CUDA_ARCH_ENV, "sm_100a")
    deep_o = is_causal and seq_len_q <= 1024 and prepare_arch != "sm_110a"
    return make_kernel(
        batch_size,
        seq_len_q,
        seq_len_kv,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        is_causal=is_causal,
        tmem_pipe_depth=3 if deep_o else TMEM_PIPE_DEPTH,
        smem_pipe_depth_kv=2 if deep_o else SMEM_PIPE_DEPTH_KV,
    ).func


def prepare_data(batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim):
    torch.manual_seed(0)
    q = torch.randn((batch_size, seq_len_q, num_qo_heads, head_dim), dtype=torch.float16)
    k = torch.randn((batch_size, seq_len_kv, num_kv_heads, head_dim), dtype=torch.float16)
    v = torch.randn((batch_size, seq_len_kv, num_kv_heads, head_dim), dtype=torch.float16)
    out = torch.zeros((batch_size, seq_len_q, num_qo_heads, head_dim), dtype=torch.float16)
    return q, k, v, out


KERNEL_META = {
    "name": "flash_attention4",
    "category": "flashattention",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a", "sm_110a"],
    "reference_requirements": (
        {
            "package": "flash-attn-4",
            "git": {
                "url": "https://github.com/Dao-AILab/flash-attention.git",
                "commit": "0251105a2fb19d2957484b7f023cd8c115286ced",
            },
            "import": "flash_attn.cute",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
}
CONFIGS = [
    {
        "batch_size": 1,
        "seq_len": sl,
        "num_qo_heads": 32,
        "num_kv_heads": kv,
        "head_dim": 128,
        "is_causal": causal,
        "label": f"s{sl}_h32kv{kv}{('_causal' if causal else '')}",
    }
    for sl in [1024, 2048, 4096, 8192]
    for kv in [4, 8, 16, 32]
    for causal in [False, True]
]


def get_kernel(
    batch_size, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=False, **kwargs
):
    return get_flash_attention4_kernel(
        batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=is_causal
    )


def _build_launch(
    executable,
    q,
    k,
    v,
    out,
    *,
    batch_size,
    seq_len_q,
    seq_len_kv,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    is_causal=False,
):
    tensor_maps = build_tensor_maps(
        q,
        k,
        v,
        out,
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_causal=is_causal,
    )
    argv = tuple(desc.ptr for desc in tensor_maps)

    def launch():
        executable(*argv)

    launch._fa4_keep_alive = (q, k, v, out, tensor_maps, argv)
    return launch


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.bench.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(batch_size, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=False, **kwargs):
    """Compile, run, and verify FlashAttention-4."""
    from tirx_kernels.bench.runner import compile_kernel

    q, k, v, _ = prepare_data(batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim)
    q_dev, k_dev, v_dev = q.cuda(), k.cuda(), v.cuda()
    out = torch.empty(
        (batch_size, seq_len, num_qo_heads, head_dim), dtype=torch.float16, device="cuda"
    )
    executable = compile_kernel(
        get_flash_attention4_kernel(
            batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=is_causal
        )
    )
    launch = _build_launch(
        executable,
        q_dev,
        k_dev,
        v_dev,
        out,
        batch_size=batch_size,
        seq_len_q=seq_len,
        seq_len_kv=seq_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_causal=is_causal,
    )
    launch()
    torch.cuda.synchronize()

    # The upstream FA4 forward on the same inputs is the arbiter (the sibling
    # backward port validates the same way via flash_attn.cute.interface).
    from flash_attn.cute.interface import _flash_attn_fwd

    with torch.no_grad():
        ref = _flash_attn_fwd(
            q=q_dev, k=k_dev, v=v_dev, softmax_scale=1.0 / math.sqrt(head_dim), causal=is_causal
        )[0]
    if not torch.isfinite(out).all():
        raise AssertionError("output contains a non-finite value")
    torch.testing.assert_close(out, ref, rtol=0.01, atol=0.01)


def run_gpu(
    prepared,
    *,
    warmup=None,
    repeat=None,
    timer=None,  # None inherits the global default (proton); the CuTeDSL flashattn
    # reference cannot be CUDA-graph-captured, so proton (not cudagraph_proton) is what
    # gives an honest ratio here (verified 0.994 vs event's unstable 0.97-1.38).
    **kwargs,
):
    """Benchmark flash attention 4."""
    config = dict(prepared["config"])
    batch_size = config.pop("batch_size")
    seq_len = config.pop("seq_len")
    num_qo_heads = config.pop("num_qo_heads")
    num_kv_heads = config.pop("num_kv_heads")
    head_dim = config.pop("head_dim")
    is_causal = config.pop("is_causal")
    config.update(kwargs)
    kwargs = config
    executable = prepared["executable"]

    ex = executable

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    Q, K, V, _ = prepare_data(batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim)
    Q_cuda = Q.cuda()
    K_cuda = K.cuda()
    V_cuda = V.cuda()
    O_tir = torch.empty(
        (batch_size, seq_len, num_qo_heads, head_dim), dtype=torch.float16, device="cuda"
    )
    launch = _build_launch(
        ex,
        Q_cuda,
        K_cuda,
        V_cuda,
        O_tir,
        batch_size=batch_size,
        seq_len_q=seq_len,
        seq_len_kv=seq_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_causal=is_causal,
    )
    funcs = {"tir": launch}

    def _flashattn_sm100():
        # Use upstream FA4's public adapter so its architecture-specific launch
        # selection remains part of the source baseline.
        from flash_attn.cute.interface import _flash_attn_fwd

        out_fa4 = torch.empty_like(O_tir)
        call_kwargs = {
            "q": Q_cuda,
            "k": K_cuda,
            "v": V_cuda,
            "out": out_fa4,
            "softmax_scale": 1.0 / math.sqrt(head_dim),
            "causal": is_causal,
        }

        def run():
            _flash_attn_fwd(**call_kwargs)

        run._fa4_keep_alive = (out_fa4, call_kwargs)
        return run

    return bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashattn_sm100": _flashattn_sm100},
        **kwargs,
    )


def run_bench(
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    is_causal=False,
    warmup=None,
    repeat=None,
    timer=None,  # None inherits the global default (proton); the CuTeDSL flashattn
    # reference cannot be CUDA-graph-captured, so proton (not cudagraph_proton) is what
    # gives an honest ratio here (verified 0.994 vs event's unstable 0.97-1.38).
    **kwargs,
):
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    prepared = prepare_bench(
        batch_size=batch_size,
        seq_len=seq_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_causal=is_causal,
        **config,
    )
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


def _parse_iket_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the annotated FA4 kernel with NVIDIA IKET"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of traced FA4 launches; setup and compilation remain outside the loop",
    )
    parser.add_argument("--output-dir", default="/tmp/fa4-iket")
    parser.add_argument(
        "--postprocess", choices=("perfetto", "json", "html", "none", "all"), default="all"
    )
    parser.add_argument("--clobber", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--keep", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-ts-cnt-per-warp", type=int, default=None)
    return parser.parse_args()


def _profile_iket_workload(args: argparse.Namespace) -> None:
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    func = get_flash_attention4_kernel(
        args.batch_size,
        args.seq_len,
        args.seq_len,
        args.num_qo_heads,
        args.num_kv_heads,
        args.head_dim,
        is_causal=args.causal,
    )
    executable = IketProfiler().compile(
        tvm.IRModule({"main": func}), target=cuda_target(), tir_pipeline="tirx"
    )

    q, k, v, _ = prepare_data(
        args.batch_size,
        args.seq_len,
        args.seq_len,
        args.num_qo_heads,
        args.num_kv_heads,
        args.head_dim,
    )
    q, k, v = q.cuda(), k.cuda(), v.cuda()
    out = torch.empty(
        (args.batch_size, args.seq_len, args.num_qo_heads, args.head_dim),
        dtype=torch.float16,
        device="cuda",
    )

    launch = _build_launch(
        executable,
        q,
        k,
        v,
        out,
        batch_size=args.batch_size,
        seq_len_q=args.seq_len,
        seq_len_kv=args.seq_len,
        num_qo_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        is_causal=args.causal,
    )
    for _ in range(args.repeat):
        launch()
    torch.cuda.synchronize()


def _print_iket_result(result: iket.IketProfileResult) -> None:
    print(f"IKET output directory: {result.output_dir}")
    for path in (*result.json_traces, *result.perfetto_traces, *result.html_reports):
        print(f"IKET artifact: {path}")


def main() -> None:
    """Profile FA4 when this kernel module is executed directly."""
    args = _parse_iket_args()
    result = iket.run(
        partial(_profile_iket_workload, args),
        output_dir=args.output_dir,
        postprocess=args.postprocess,
        clobber=args.clobber,
        timeout=args.timeout,
        keep=args.keep,
        max_ts_cnt_per_warp=args.max_ts_cnt_per_warp,
    )
    _print_iket_result(result)


if __name__ == "__main__":
    main()
