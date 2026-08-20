# This file is a TIRx port of code from flash-attention
# (https://github.com/Dao-AILab/flash-attention @ d7e4dba3),
# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM100 two-CTA FlashAttention backward in the ``tirx_kernels.kern`` DSL.

The preprocess, two-CTA core, and output cast preserve the original instruction
schedule. Tensor maps are encoded by the host because ``K.kernel`` traces only
device code.

Do not enable postponed annotations: ``K.kernel`` consumes parameter
annotations directly.
"""

import ctypes
import math
from functools import cache

import torch

import tirx_kernels.kern as K
import tvm
from tvm.backend.cuda.cpp.descriptors import encode_smem_descriptor_base_uint64

HEAD_DIM = 128

PRE_THREADS_PER_ROW = 16
PRE_BLOCK = 256
PRE_ROWS_PER_BLOCK = 128
LOG2_E = math.log2(math.e)

CAST_GROUP_WIDTH = 4
CAST_GROUPS_PER_THREAD = 4
CAST_BLOCK = 256


class _AlignedTensorMap:
    """Host storage for one 64-byte-aligned, 128-byte TensorMap payload."""

    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode(tensor, dtype, rank, global_dims, global_strides, box_dims, swizzle, l2_promotion):
    """One ``cuTensorMapEncodeTiled``; element strides are all 1 here."""
    descriptor = _AlignedTensorMap()
    if len(global_dims) != rank or len(global_strides) != rank - 1 or len(box_dims) != rank:
        raise ValueError("TensorMap dimension counts disagree with the rank")
    tvm.get_global_func("runtime.cuTensorMapEncodeTiled")(
        descriptor.ptr,
        dtype,
        rank,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *global_dims,
        *global_strides,
        *box_dims,
        *((1,) * rank),  # elementStrides
        0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        swizzle,
        l2_promotion,
        0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    )
    return descriptor


def build_tensor_maps(Q, Kt, V, dO, dK, dV, dQ_acc, B, H, S, D):
    """The ten descriptors the kernel actually consumes, in parameter order.

    LSE and dPsum use bulk copies from raw global pointers, so they need no
    tensor-map parameters.
    """
    if D != HEAD_DIM:
        raise ValueError("the SM100 2-CTA backward kernel currently requires head_dim=128")
    CTA_N = 128  # BLK_N // CTA_GROUP
    B_N = 64  # BLK_M // CTA_GROUP
    B_N_COL = D // 2
    BLK_M, BLK_N, EPI_N = 128, 256, 64
    DQ_ROWS_PER_STAGE = 8

    # rank-5 row-major stream: (d/2, s, 2, h, b) -- the "two halves of a
    # 128-element head" split the original uses for its row-major boxes.
    row5_dims = (D // 2, S, 2, H, B)
    row5_strides = (H * D * 2, D, D * 2, S * H * D * 2)
    # rank-4 column-major-ish stream: (d, s, h, b)
    col4_dims = (D, S, H, B)
    col4_strides = (H * D * 2, D * 2, S * H * D * 2)

    def row5(t, box_rows):
        return _encode(t, "float16", 5, row5_dims, row5_strides, (D // 2, box_rows, 2, 1, 1), 3, 2)

    def col4(t, box_cols, box_rows):
        return _encode(t, "float16", 4, col4_dims, col4_strides, (box_cols, box_rows, 1, 1), 3, 2)

    # fmt: off
    return (
        row5(Q, B_N),  # q_row
        col4(Q, B_N_COL, BLK_M),  # q_col
        row5(Kt, CTA_N),  # k_row
        col4(Kt, B_N_COL, BLK_N),  # k_col
        row5(V, CTA_N),  # v_row
        row5(dO, B_N),  # do_row
        col4(dO, B_N_COL, BLK_M),  # do_col
        col4(dK, EPI_N, CTA_N),  # dk
        col4(dV, EPI_N, CTA_N),  # dv
        _encode(  # dq: fp32 accumulator, no swizzle on the 128-wide box
            dQ_acc, "float32", 4,
            (D, S, H, B), (D * 4, S * D * 4, H * S * D * 4),
            (D, DQ_ROWS_PER_STAGE, 1, 1), 0, 2,
        ),
    )
    # fmt: on


# ---------------------------------------------------------------------------
# preprocess: dPsum + LSE-log2 + dQ clear
# ---------------------------------------------------------------------------


def build_preprocess(B, S, H, D):
    """dPsum/LSE-log2 and the dQ accumulation clear, in one pass.

    Sixteen lanes own one row, reduce with width-16 shuffles, and clear the
    matching dQ slice.
    """
    if S % PRE_ROWS_PER_BLOCK:
        raise ValueError("the SM100 backward preprocess requires seq_len divisible by 128")
    elems_per_thread = D // PRE_THREADS_PER_ROW
    rows_per_wave = PRE_BLOCK // PRE_THREADS_PER_ROW
    row_iters = PRE_ROWS_PER_BLOCK // rows_per_wave
    nblk = S // PRE_ROWS_PER_BLOCK

    @K.kernel(warps=PRE_BLOCK // 32, arch="sm_100a", grid=(nblk, H, B))
    def preprocess_kernel(
        dO_g: K.gptr[K.f16],
        O_g: K.gptr[K.f16],
        LSE_g: K.gptr[K.f32],
        dpsum_g: K.gptr[K.f32],
        LSE_log2_g: K.gptr[K.f32],
        dQ_accum_g: K.gptr[K.f32],
    ):
        bx, by, bz = K.cta_id()

        tx = K.thread_id()
        col_in_row = tx % PRE_THREADS_PER_ROW
        row_in_wave = tx // PRE_THREADS_PER_ROW
        d_start = col_in_row * elems_per_thread

        def dot_f16x8(dst, lhs, rhs):
            """Accumulate eight f16 products in f32."""
            lhs_words = K.alloc_local([4], "uint32")
            rhs_words = K.alloc_local([4], "uint32")
            lhs_halves = K.alloc_local([2], "uint16")
            rhs_halves = K.alloc_local([2], "uint16")
            lhs_value = K.alloc_local([2], "float32")
            rhs_value = K.alloc_local([2], "float32")
            K.ptx.ld.global_.nc.v4.b32(lhs_words[0], lhs_words[1], lhs_words[2], lhs_words[3], lhs)
            K.ptx.ld.global_.nc.v4.b32(rhs_words[0], rhs_words[1], rhs_words[2], rhs_words[3], rhs)
            K.assign(dst[0], K.float32(0))
            for pair in range(4):
                K.ptx.mov.b32(lhs_halves[0], lhs_halves[1], lhs_words[pair])
                for element in range(2):
                    K.ptx.cvt.f32.f16(lhs_value[element], lhs_halves[element])
                K.ptx.mov.b32(rhs_halves[0], rhs_halves[1], rhs_words[pair])
                for element in range(2):
                    K.ptx.cvt.f32.f16(rhs_value[element], rhs_halves[element])
                for element in range(2):
                    # CUDA's default fast-math path lowers the original fmaf
                    # chain with FTZ; spelling it keeps the same schedule.
                    K.ptx.fma.rn.ftz.f32(dst[0], lhs_value[element], rhs_value[element], dst[0])

        # Overlap the independent LSE load with the O/dO dot products.
        lse_for_log2 = K.alloc_local([1], "float32")
        K.assign(lse_for_log2[0], K.float32(0))
        with K.If(tx < PRE_ROWS_PER_BLOCK), K.Then():
            K.ptx.ld.global_.f32(
                lse_for_log2[0], LSE_g.ptr_to([(bz * H + by) * S + bx * PRE_ROWS_PER_BLOCK + tx])
            )

        acc = K.alloc_local([1], "float32")
        for row_iter in range(row_iters):
            s = bx * PRE_ROWS_PER_BLOCK + row_iter * rows_per_wave + row_in_wave
            row_base = ((bz * S + s) * H + by) * D + d_start
            dot_f16x8(acc, dO_g.ptr_to([row_base]), O_g.ptr_to([row_base]))

            dq_base = ((bz * H + by) * S + s) * D + d_start
            for chunk in range(elems_per_thread // 4):
                K.ptx.st.global_.v4.f32(
                    dQ_accum_g.ptr_to([dq_base + chunk * 4]),
                    K.float32(0),
                    K.float32(0),
                    K.float32(0),
                    K.float32(0),
                )

            # Keep the shuffle collective outside the row-leader guard.
            for delta in (8, 4, 2, 1):
                K.assign(
                    acc[0],
                    acc[0]
                    + K.cuda.__shfl_xor_sync(
                        K.uint32(0xFFFFFFFF), acc[0], delta, PRE_THREADS_PER_ROW
                    ),
                )
            with K.If(col_in_row == 0), K.Then():
                K.ptx.st.global_.f32(dpsum_g.ptr_to([(bz * H + by) * S + s]), acc[0])

        # One independent scalar per row: spread the 128 rows over 128 threads
        # instead of serializing eight conversions on each row leader.
        with K.If(tx < PRE_ROWS_PER_BLOCK), K.Then():
            lse_s = (bz * H + by) * S + bx * PRE_ROWS_PER_BLOCK + tx
            K.ptx.st.global_.f32(
                LSE_log2_g.ptr_to([lse_s]),
                K.if_then_else(
                    lse_for_log2[0] == K.float32(-float("inf")),
                    K.float32(0),
                    lse_for_log2[0] * K.float32(LOG2_E),
                ),
            )

    return preprocess_kernel


# ---------------------------------------------------------------------------
# cast: scale and transpose the head-major dQ accumulation to fp16
# ---------------------------------------------------------------------------


def build_cast_f32_to_f16(B, S, H, D, scale):
    """Scale and transpose the head-major accumulator to sequence-major f16."""
    groups_per_block = CAST_BLOCK * CAST_GROUPS_PER_THREAD
    num_groups = B * S * H * (D // CAST_GROUP_WIDTH)
    nblk = (num_groups + groups_per_block - 1) // groups_per_block

    @K.kernel(warps=CAST_BLOCK // 32, arch="sm_100a", grid=nblk)
    def cast_kernel(src: K.gptr[K.f32], dst: K.gptr[K.f16]):
        bx = K.cta_id()
        tx = K.thread_id()

        def scale_cast_f32x4_f16x4(dst_ptr, src_ptr):
            source = K.alloc_local([4], "float32")
            scaled = K.alloc_local([4], "float32")
            packed = K.alloc_local([2], "uint32")
            K.ptx.ld.global_.v4.f32(source[0], source[1], source[2], source[3], src_ptr)
            for element in range(4):
                K.ptx.mul.rn.f32(scaled[element], source[element], K.float32(scale))
            K.ptx.cvt.rn.f16x2.f32(packed[0], scaled[1], scaled[0])
            K.ptx.cvt.rn.f16x2.f32(packed[1], scaled[3], scaled[2])
            K.ptx.st.global_.v2.b32(dst_ptr, packed[0], packed[1])

        for e in range(CAST_GROUPS_PER_THREAD):
            group = bx * groups_per_block + e * CAST_BLOCK + tx
            with K.If(group < num_groups), K.Then():
                d_group = group % (D // CAST_GROUP_WIDTH)
                h = group // (D // CAST_GROUP_WIDTH) % H
                s = group // (D // CAST_GROUP_WIDTH * H) % S
                b = group // (D // CAST_GROUP_WIDTH * H * S)
                d = d_group * CAST_GROUP_WIDTH
                # The raw tcgen05 dQ accumulator uses the physical 128x128
                # C-fragment bit layout. Decode it while producing the public
                # sequence-major dQ.
                s_in_block = s % 128
                src_s = (
                    s // 128 * 128
                    + ((s_in_block >> 5) & 1)
                    + (((d >> 6) & 1) << 1)
                    + (((d >> 2) & 15) << 2)
                    + (((s_in_block >> 6) & 1) << 6)
                )
                src_d = (s_in_block & 31) << 2
                scale_cast_f32x4_f16x4(
                    dst.ptr_to([((b * S + s) * H + h) * D + d]),
                    src.ptr_to([((b * H + h) * S + src_s) * D + src_d]),
                )

    return cast_kernel


# ---------------------------------------------------------------------------
# Main kernel
# ---------------------------------------------------------------------------


def build_kernel(
    BATCH,
    HEADS_PER_BATCH,
    SEQ_LEN,
    HEAD_DIM=128,
    *,
    causal=False,
    attention_scale=None,
    sm_count=148,
):
    if HEAD_DIM != 128:
        raise ValueError("the SM100 2-CTA backward kernel currently requires head_dim=128")
    if SEQ_LEN % 256:
        raise ValueError("the SM100 2-CTA backward kernel requires seq_len divisible by 256")
    if sm_count < 2:
        raise ValueError("the SM100 2-CTA backward kernel requires at least two SMs")

    # Leave the first KiB to the barriers allocated before the matrix payloads.
    POOL_Q_ROW = 1024

    NUM_HEADS = BATCH * HEADS_PER_BATCH
    CTA_GROUP = 2
    CLUSTER_SIZE = 2
    BLK_M = 128
    BLK_N = 256  # 2 CTAs x 128 rows each
    CTA_N = BLK_N // CTA_GROUP  # 128 per CTA
    MMA_N = 128
    B_N = BLK_M // CTA_GROUP  # 64: per-CTA B rows for row-split
    B_N_COL = HEAD_DIM // CTA_GROUP  # 64: per-CTA B cols for col-split
    EPI_N = 64
    STRIP_SIZE = 64
    DQ_RED_N = 8
    DQ_STAGES = 4
    DQ_M_PER_CTA = 64
    DQ_ROWS_PER_STAGE = DQ_RED_N
    DQ_REDUCE_ITERS = DQ_M_PER_CTA // DQ_ROWS_PER_STAGE

    NUM_M_TILES = SEQ_LEN // BLK_M
    NUM_N_TILES = SEQ_LEN // BLK_N

    softmax_scale = 1.0 / math.sqrt(HEAD_DIM) if attention_scale is None else float(attention_scale)
    log2e = 1.4426950408889634
    scale_log2 = softmax_scale * log2e
    DTYPE_SIZE = 2
    _ = sm_count  # kept for the public signature; the schedule is single-tile

    # TMA byte counts
    CTA_N_BYTES = CTA_N * HEAD_DIM * DTYPE_SIZE  # 32KB per CTA's K or V load
    Q_ROW_BYTES = B_N * HEAD_DIM * DTYPE_SIZE  # 16KB per CTA's Q row-split
    Q_COL_BYTES = BLK_M * B_N_COL * DTYPE_SIZE  # 16KB per CTA's Q col-split
    LSE_BYTES = BLK_M * 4
    DPSUM_BYTES = BLK_M * 4
    K_COL_BYTES = BLK_N * B_N_COL * DTYPE_SIZE  # 32KB
    KV_TOTAL_BYTES = (CTA_N_BYTES * 2 + K_COL_BYTES) * CTA_GROUP
    Q_BATCH_BYTES = Q_ROW_BYTES * CTA_GROUP
    QCOL_BATCH_BYTES = Q_COL_BYTES * CTA_GROUP
    DO_BATCH_BYTES = (Q_ROW_BYTES + Q_COL_BYTES) * CTA_GROUP

    # TMEM packing. dQ aliases the upper half of S/P; dP and dS alias each other
    # after the compute warps drain dP.
    TMEM_OFF_A = 0  # S/P
    TMEM_OFF_DQ = MMA_N // 2  # dQ (64), aliases S/P
    TMEM_OFF_B = MMA_N  # dV accumulator (128)
    TMEM_OFF_DP = 2 * MMA_N  # dP/dS (256)
    TMEM_OFF_C = 3 * MMA_N  # dK accumulator (384)

    # kind::f16 = fp16 A/B into an fp32 accumulator. The .ss and .ts table
    # entries share this chain and are told apart by the A operand's dtype:
    # u64 is a shared-memory descriptor, u32 a TMEM address.
    MMA_F16 = "tcgen05.mma.cta_group::2.kind::f16"
    # Decode the fixed tcgen05 descriptors rather than duplicating their fields.
    ID_SS = 270532624  # 0x10200010: M=256 N=128, A k-major, B k-major
    ID_TS = 270598160  # 0x10210010: M=256 N=128, A in tmem,  B mn-major
    ID_DQ = 136413200  # 0x08218010: M=128 N=128, A mn-major, B mn-major
    DESC_K_ROW = encode_smem_descriptor_base_uint64(1024, 64, K.SW128B.value)
    DESC_ROW = encode_smem_descriptor_base_uint64(512, 64, K.SW128B.value)
    DESC_MN = encode_smem_descriptor_base_uint64(0, 64, K.SW128B.value)

    # cta_group::2 takes an EIGHT-lane disable-output-lane vector; nothing is
    # disabled here, but the operands are not optional.
    MMA_KEEP_ALL_LANES = (0,) * 8

    # 2-SM cluster TMA load: unicast (no .multicast::cluster) and no cache
    # policy, so only the completion and cta_group tokens ride the chain.
    TMA_G2S_2SM = (
        "cp.async.bulk.tensor.{dim}d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
    )
    TMA_S2G = "cp.async.bulk.tensor.{dim}d.global.shared::cta.tile.bulk_group"
    TMA_S2G_REDUCE = "cp.reduce.async.bulk.tensor.{dim}d.global.shared::cta.{redop}.tile.bulk_group"
    BULK_G2S_CTA = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
    BULK_S2C = "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
    # pair_mask names both CTAs of the pair, so the commit is the multicast form.
    TCGEN05_COMMIT = (
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
    )

    XN = NUM_N_TILES * CLUSTER_SIZE

    # fmt: off
    @K.kernel(warps=16, arch="sm_100a", min_blocks_per_sm=1, grid=(XN, NUM_HEADS))
    def kernel(
        Q_g: K.gptr[K.f16],
        K_g: K.gptr[K.f16],
        V_g: K.gptr[K.f16],
        dO_g: K.gptr[K.f16],
        LSE_g: K.gptr[K.f32],
        dpsum_g: K.gptr[K.f32],
        dK_g: K.gptr[K.f16],
        dV_g: K.gptr[K.f16],
        dQ_acc_g: K.gptr[K.f32],
        q_row_map: K.TensorMap,
        q_col_map: K.TensorMap,
        k_row_map: K.TensorMap,
        k_col_map: K.TensorMap,
        v_row_map: K.TensorMap,
        do_row_map: K.TensorMap,
        do_col_map: K.TensorMap,
        dk_map: K.TensorMap,
        dv_map: K.TensorMap,
        dq_map: K.TensorMap,
    ):
        cluster_rank_ = K.cta_id_in_cluster([CLUSTER_SIZE], preferred=[CLUSTER_SIZE])
        bx, by = K.cta_id()
        lane_id = K.lane_id()
        id_in_pair = cluster_rank_ % CTA_GROUP
        pair_leader_rank = cluster_rank_ - id_in_pair

        # ---------------- barriers, in the original's declaration order -----
        smem = K.smem_pool()
        tmem_addr = smem.alloc((1,), K.u32)
        # orig picks wg3/warp0/lane0 = CTA thread 384 as this one's init leader.
        tmem_dealloc_mbar = K.MBarrier(smem, 1, leader=(K.thread_id() == 12 * 32))

        tma_kv = K.TMABar(smem, 1)
        tma_a = K.TMABar(smem, 1)  # depth 1: single buf_A
        tma_q = K.TMABar(smem, 1)  # single-stage in the current 2-CTA schedule
        tma_lse = K.TMABar(smem, 1)
        tma_dpsum = K.TMABar(smem, 1)
        tma_qcol = K.TMABar(smem, 1)  # depth 1: Q_col single-buf
        mma2wg0_s = K.TCGen05Bar(smem, 1)
        mma2wg0_dp = K.TCGen05Bar(smem, 1)
        mma2wg0_dq = K.TCGen05Bar(smem, 1)
        ds_exch_mbar = K.MBarrier(smem, 1)  # DSMEM exchange completion
        ds_exch_consumed = K.MBarrier(smem, 1)  # dQ MMA released the DSMEM buffer
        wg02mma = K.MBarrier(smem, 1)  # softmax fully done (incl DSMEM) -> Phase E
        wg02mma_tmem = K.MBarrier(smem, 1)  # softmax dS in TMEM only -> Phase D
        strip_ready = K.MBarrier(smem, 1)
        s_tmem_consumed = K.MBarrier(smem, 1)  # next S read drained before dQ write
        buf_a_consumed = K.MBarrier(smem, 1)
        q_consumed = K.MBarrier(smem, 1)
        lse_consumed = K.MBarrier(smem, 1)
        dpsum_consumed = K.MBarrier(smem, 1)
        qcol_consumed = K.MBarrier(smem, 1)
        dq_tmem_free = K.MBarrier(smem, 1)
        dv_done = K.TCGen05Bar(smem, 1)  # dV accumulator ready -> epilogue stage 0
        dk_done = K.TCGen05Bar(smem, 1)  # dK accumulator ready -> epilogue stage 1
        # K.smem_pool has no base-move spelling; the in-tree pool it wraps does,
        # and the epilogue tiles below genuinely alias K/V storage.
        smem.pool.move_base_to(POOL_Q_ROW)

        # ---------------- smem plan -----------------------------------------
        # Physical order matches the upstream 2-CTA D=128 SharedStorage:
        # sQ, sK, sV, sdO, sQt, sdOt, sdS_xchg, sKt, sdS.
        # swizzle=K.SW128B is the composed layout the original spells by hand as
        # `tma_shared_layout`: the last two dims tiled into [8, 64] f16 atoms
        # with the xor inside the atom.
        Q_row = smem.alloc((1, B_N, HEAD_DIM), K.f16, swizzle=K.SW128B)  # 16KB
        K_smem = smem.alloc((CTA_N, HEAD_DIM), K.f16, swizzle=K.SW128B)  # 32KB
        V_smem = smem.alloc((CTA_N, HEAD_DIM), K.f16, swizzle=K.SW128B)  # 32KB
        dO_row = smem.alloc((B_N, HEAD_DIM), K.f16, swizzle=K.SW128B)  # 16KB
        Q_col = smem.alloc((BLK_M, B_N_COL), K.f16, swizzle=K.SW128B)  # 16KB
        dO_col = smem.alloc((BLK_M, B_N_COL), K.f16, swizzle=K.SW128B)  # 16KB
        dS_send = smem.alloc((CTA_N, B_N), K.f16, swizzle=K.SW128B)  # 16KB
        K_col = smem.alloc((BLK_N, B_N_COL), K.f16, swizzle=K.SW128B)  # 32KB
        dS_exch = smem.alloc((BLK_N, B_N), K.f16, swizzle=K.SW128B)  # 32KB
        sLSE = smem.alloc((1, BLK_M), K.f32)  # 512B
        sDPsum = smem.alloc((BLK_M,), K.f32)  # 512B
        # One flat row per stage: the original's (BLK_M, DQ_RED_N) row-major
        # tile linearises identically and the writes below are flat anyway.
        dQ_smem = smem.alloc((DQ_STAGES, BLK_M * DQ_RED_N), K.f32, align=1024)
        # The two-stage dKV epilogue reuses sV for dV and sK for dK.
        smem.pool.move_base_to(K_smem.buf.elem_offset * DTYPE_SIZE)
        dK_epi = smem.alloc((2, CTA_N, EPI_N), K.f16, swizzle=K.SW128B)
        smem.pool.move_base_to(V_smem.buf.elem_offset * DTYPE_SIZE)
        dV_epi = smem.alloc((2, CTA_N, EPI_N), K.f16, swizzle=K.SW128B)

        tma_kv.init(1)
        tma_a.init(1)
        tma_q.init(1)
        tma_lse.init(1)
        tma_dpsum.init(1)
        tma_qcol.init(1)
        mma2wg0_s.init(1)
        mma2wg0_dp.init(1)
        mma2wg0_dq.init(1)
        wg02mma.init(CTA_GROUP)
        wg02mma_tmem.init(8 * CTA_GROUP)  # one elected thread per compute warp
        strip_ready.init(8 * CTA_GROUP)
        s_tmem_consumed.init(8 * CTA_GROUP)
        buf_a_consumed.init(1)  # Phase C commit releases dO/dOt
        q_consumed.init(1)  # signaled by tcgen05.commit
        # LSE/dPsum are CTA-local streams: one elected lane in each of the eight
        # compute warps releases the single buffer after its warp consumed it.
        lse_consumed.init(8)
        dpsum_consumed.init(8)
        qcol_consumed.init(1)  # signaled by tcgen05.commit
        dq_tmem_free.init(4 * CTA_GROUP)  # one elected thread per reducer warp
        ds_exch_mbar.init(1)
        ds_exch_consumed.init(1)
        dv_done.init(1)
        dk_done.init(1)
        tmem_dealloc_mbar.init(32)

        pair_mask = K.bitwise_or(
            K.shift_left(K.int32(1), pair_leader_rank),
            K.shift_left(K.int32(1), pair_leader_rank + 1),
        )

        K.ptx.fence.proxy.async_.shared__cta()
        K.ptx.fence.mbarrier_init.release.cluster()
        K.ptx.barrier.cluster.arrive.relaxed()
        K.ptx.barrier.cluster.wait()

        # TMA barrier remote view for the leader's mbar
        tma_kv_cta0 = tma_kv.remote_view(pair_leader_rank)
        tma_q_cta0 = tma_q.remote_view(pair_leader_rank)
        tma_qcol_cta0 = tma_qcol.remote_view(pair_leader_rank)
        tma_a_cta0 = tma_a.remote_view(pair_leader_rank)

        # Every physical 2-CTA cluster owns exactly one logical KV tile for the
        # kernel lifetime (upstream SingleTileScheduler).
        n_tile_idx = bx // CLUSTER_SIZE
        b_idx = by // HEADS_PER_BATCH
        h_idx = by % HEADS_PER_BATCH
        n_st = n_tile_idx * BLK_N
        n_st_cta = n_st + id_in_pair * CTA_N
        # In causal mode a K/V tile n only receives gradients from Q rows at or
        # below its first row; BLK_N is exactly two BLK_M tiles, so skipping the
        # all-zero leading tiles keeps the pipeline trip count even.
        m_tile_start = n_tile_idx * (BLK_N // BLK_M) if causal else 0
        num_m_tiles_this_n = NUM_M_TILES - m_tile_start
        m_st_first = m_tile_start * BLK_M

        # ---------------- shared user closures ------------------------------

        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def tmem_load(width, dst, dst_offset, base, row, col):
            K.ptx[f"tcgen05.ld.sync.aligned.32x32b.x{width}.b32"](
                *[dst[dst_offset + i] for i in range(width)], K.cuda.get_tmem_addr(K.uint32(base), row, col)
            )

        def tmem_store(width, src, src_offset, base, row, col):
            K.ptx[f"tcgen05.st.sync.aligned.32x32b.x{width}.b32"](
                K.cuda.get_tmem_addr(K.uint32(base), row, col), *[src[src_offset + i] for i in range(width)]
            )

        def tma_g2s(dim, dst_ptr, mbar, tensormap, *coords):
            K.ptx[TMA_G2S_2SM.format(dim=dim)](
                dst_ptr, K.address_of(tensormap), *coords, mbar
            )

        def tma_s2g(dim, src_ptr, tensormap, *coords):
            K.ptx[TMA_S2G.format(dim=dim)](K.address_of(tensormap), *coords, src_ptr)

        def tcgen05_commit(mbar_ptr):
            K.ptx[TCGEN05_COMMIT](mbar_ptr, K.Cast("uint16", pair_mask))

        def mma_chain2(d, *, a, b, idesc, accumulate, dol=MMA_KEEP_ALL_LANES):
            """Issue an unpredicated cta_group::2 chain inside ``elect_sync``.

            ``K.idioms.mma_chain`` emits predicated MMAs, so this local expansion
            preserves the original instruction form and descriptor hoisting.
            """
            (b_desc, b_off), n_k = b
            if isinstance(a, int):  # ts-form: one k-tile is 8 tmem columns
                a_ops = [K.uint32(a + kp * 8) for kp in range(n_k)]
            else:
                (a_desc, a_off), _ = a
                a_ops = [a_desc + a_off(kp) for kp in range(n_k)]
            for kp in range(n_k):
                K.ptx[MMA_F16](
                    K.uint32(d),
                    a_ops[kp],
                    b_desc + b_off(kp),
                    K.uint32(idesc),
                    *dol,
                    K.ptx.pred(accumulate if kp == 0 else 1),
                )

        # ================================================================
        # roles. The original's `wg_id == 3` / `1 <= wg_id <= 2` / else split,
        # stated as the warp partition it is. kern checks the exact partition
        # of 0..15, contiguity, regs 8-aligned in [24, 256], budget <= 65536
        # (this kernel uses all of it), and setmaxnreg warpgroup-uniformity.
        # ================================================================
        sp = K.specialize()
        mma = sp.role("mma", warps=[12], regs=104)
        tma = sp.role("tma", warps=[13], regs=104)
        relay = sp.role("relay", warps=[14], regs=104)
        idle = sp.role("idle", warps=[15], regs=104)
        compute = sp.role("compute", warps=list(range(4, 12)), regs=136)
        dq_reduce = sp.role("dq_reduce", warps=[0, 1, 2, 3], regs=136)

        # ================================================================
        # WG3: MMA (warp0) + TMA (warp1) + relay (warp2); warp3 idle.
        # ================================================================
        # ---- TMA warp -----------------------------------------------------
        with tma:
            with K.If(elected()), K.Then():
                for m in (
                    q_row_map, q_col_map, k_row_map, k_col_map, v_row_map,
                    do_row_map, do_col_map, dk_map, dv_map, dq_map,
                ):
                    K.ptx.prefetch.tensormap(K.address_of(m))
            q_cons_ph = K.PipelineState(1, phase=1)
            lse_cons_ph = K.PipelineState(1, phase=1)
            qcol_cons_ph = K.PipelineState(1, phase=1)
            a_cons_ph = K.PipelineState(1, phase=1)
            dpsum_cons_ph = K.PipelineState(1, phase=1)

            def load_q_row(m_st):
                q_consumed.wait(0, q_cons_ph.phase)
                with K.If(elected()), K.Then():
                    tma_g2s(
                        5, Q_row[0].ptr_to(0, 0), tma_q_cta0.ptr_to([0]), q_row_map,
                        0, m_st + id_in_pair * B_N, 0, h_idx, b_idx,
                    )
                    with K.If(id_in_pair == 0), K.Then():
                        tma_q.arrive(0, Q_BATCH_BYTES)
                q_cons_ph.advance()

            def load_lse(m_st):
                lse_consumed.wait(0, lse_cons_ph.phase)
                with K.If(elected()), K.Then():
                    tma_lse.arrive(0, LSE_BYTES)
                    K.ptx[BULK_G2S_CTA](
                        sLSE.ptr_to([0, 0]),
                        LSE_g.ptr_to([(b_idx * HEADS_PER_BATCH + h_idx) * SEQ_LEN + m_st]),
                        K.uint32(LSE_BYTES),
                        tma_lse.ptr_to([0]),
                    )
                lse_cons_ph.advance()

            def load_do(m_st):
                buf_a_consumed.wait(0, a_cons_ph.phase)
                with K.If(elected()), K.Then():
                    tma_g2s(
                        5, dO_row.ptr_to(0, 0), tma_a_cta0.ptr_to([0]), do_row_map,
                        0, m_st + id_in_pair * B_N, 0, h_idx, b_idx,
                    )
                    tma_g2s(
                        4, dO_col.ptr_to(0, 0), tma_a_cta0.ptr_to([0]), do_col_map,
                        id_in_pair * B_N_COL, m_st, h_idx, b_idx,
                    )
                    with K.If(id_in_pair == 0), K.Then():
                        tma_a.arrive(0, DO_BATCH_BYTES)
                a_cons_ph.advance()

            def load_dpsum(m_st):
                dpsum_consumed.wait(0, dpsum_cons_ph.phase)
                with K.If(elected()), K.Then():
                    tma_dpsum.arrive(0, DPSUM_BYTES)
                    K.ptx[BULK_G2S_CTA](
                        sDPsum.ptr_to([0]),
                        dpsum_g.ptr_to([(b_idx * HEADS_PER_BATCH + h_idx) * SEQ_LEN + m_st]),
                        K.uint32(DPSUM_BYTES),
                        tma_dpsum.ptr_to([0]),
                    )
                dpsum_cons_ph.advance()

            def load_q_col(m_st):
                qcol_consumed.wait(0, qcol_cons_ph.phase)
                with K.If(elected()), K.Then():
                    tma_g2s(
                        4, Q_col.ptr_to(0, 0), tma_qcol_cta0.ptr_to([0]), q_col_map,
                        id_in_pair * B_N_COL, m_st, h_idx, b_idx,
                    )
                    with K.If(id_in_pair == 0), K.Then():
                        tma_qcol.arrive(0, QCOL_BATCH_BYTES)
                qcol_cons_ph.advance()

            # K/V: each CTA loads its CTA_N=128 rows; K_col is all BLK_N
            # rows and this CTA's 64 columns.
            with K.If(elected()), K.Then():
                tma_g2s(
                    5, K_smem.ptr_to(0, 0), tma_kv_cta0.ptr_to([0]), k_row_map,
                    0, n_st_cta, 0, h_idx, b_idx,
                )
                tma_g2s(
                    5, V_smem.ptr_to(0, 0), tma_kv_cta0.ptr_to([0]), v_row_map,
                    0, n_st_cta, 0, h_idx, b_idx,
                )
                tma_g2s(
                    4, K_col.ptr_to(0, 0), tma_kv_cta0.ptr_to([0]), k_col_map,
                    id_in_pair * B_N_COL, n_st, h_idx, b_idx,
                )
                with K.If(id_in_pair == 0), K.Then():
                    tma_kv.arrive(0, KV_TOTAL_BYTES)

            # Prologue: Q/Q_col/dO use the cluster MMA pipelines; LSE and
            # dPsum use independent CTA-local bulk-copy pipelines.
            load_q_row(m_st_first)
            load_lse(m_st_first)
            load_do(m_st_first)
            load_dpsum(m_st_first)

            # Every stream is single-stage and released at the earliest
            # observable consumption point. Qt trails the row-major streams
            # by one M tile, so it is issued at the front of the trip.
            with K.serial(
                num_m_tiles_this_n - 1, annotations={"disable_unroll": True}
            ) as i_m:
                m_st_next = (m_tile_start + i_m + 1) * BLK_M
                load_q_col(m_st_next - BLK_M)
                load_q_row(m_st_next)
                load_lse(m_st_next)
                load_do(m_st_next)
                load_dpsum(m_st_next)

            # The producer loop issues Qt for tiles [0, N-2]; fill the final
            # tile for the dK tail once its predecessor released the buffer.
            load_q_col((m_tile_start + num_m_tiles_this_n - 1) * BLK_M)

        # ---- MMA warp -----------------------------------------------------
        with mma:
            # The physical MMA warp owns TMEM allocation in both CTAs.
            K.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                K.address_of(tmem_addr), K.uint32(512)
            )
            K.ptx.barrier.sync(K.uint32(5), 416)

            # Only the pair leader issues the cta_group::2 matrix work.
            with K.If(id_in_pair == 0), K.Then():
                # Every operand is one of three physical layouts. Their fields
                # are compile-time constants; only the start address is live.
                anchor = Q_row[0]
                # Materialized: `descriptor` below reads it once textually but
                # is called eight times, so a lazy binding would re-issue the
                # cvta per descriptor.
                anchor_start = K.local_scalar(
                    "uint64",
                    init=K.cast(
                        K.shift_right(
                            K.cuda.cvta_generic_to_shared(anchor.ptr_to(0, 0)), K.uint32(4)
                        ),
                        "uint64",
                    ),
                )

                def descriptor(base, view):
                    delta16 = (
                        (view.buf.elem_offset - anchor.buf.elem_offset) * DTYPE_SIZE // 16
                    )
                    start = K.bitwise_and(
                        anchor_start + K.uint64(delta16), K.uint64(0x3FFF)
                    )
                    # Plain value: cheap arithmetic over the materialized anchor.
                    # Routing each descriptor through a local instead spills the
                    # MMA warp (STACK 8 -> 136 B on the causal shapes, measured).
                    return K.bitwise_or(K.uint64(base), start)

                def k_major_off(ldo):
                    return lambda kp: (kp % 4) * 2 + (kp // 4) * ldo

                def mn_off(kp):
                    return kp * 128

                k_row_off = k_major_off(1024)
                row_off = k_major_off(512)
                d_k_row = ((descriptor(DESC_K_ROW, K_smem), k_row_off), 8)
                d_v_row = ((descriptor(DESC_K_ROW, V_smem), k_row_off), 8)
                d_do_row = ((descriptor(DESC_ROW, dO_row), row_off), 8)
                d_q_col = ((descriptor(DESC_MN, Q_col), mn_off), 8)
                d_do_col = ((descriptor(DESC_MN, dO_col), mn_off), 8)
                d_k_col = ((descriptor(DESC_MN, K_col), mn_off), 16)
                d_ds_exch = ((descriptor(DESC_MN, dS_exch), mn_off), 16)
                if not causal:
                    d_q_row_hoisted = ((descriptor(DESC_ROW, anchor), row_off), 8)

                def q_row_operand():
                    """The Q-row descriptor for one Phase-A issue.

                    Noncausal shapes hoist it. Causal shapes rebuild it because
                    carrying it across the 2-CTA MMA loop costs uniform registers.
                    """
                    if not causal:
                        return d_q_row_hoisted
                    # Keep the causal Q-row descriptor issue-local: carrying it
                    # across the loop costs uniform registers.
                    q_row_start = K.cast(
                        K.shift_right(
                            K.cuda.cvta_generic_to_shared(anchor.ptr_to(0, 0)), K.uint32(4)
                        ),
                        "uint64",
                    )
                    q_row_desc = K.bitwise_or(
                        K.uint64(DESC_ROW),
                        K.bitwise_and(q_row_start, K.uint64(0x3FFF)),
                    )
                    return ((q_row_desc, row_off), 8)

                kv_ph = K.PipelineState(1, phase=0)
                q_ph = K.PipelineState(1, phase=0)
                qcol_ph = K.PipelineState(1, phase=0)
                a_ph = K.PipelineState(1, phase=0)
                wg0_ph = K.PipelineState(1, phase=0)  # wg02mma_tmem -> Phase D
                wg0_smem_ph = K.PipelineState(1, phase=0)  # peer DSMEM -> Phase E
                strip_ready_ph = K.PipelineState(1, phase=0)
                s_tmem_consumed_ph = K.PipelineState(1, phase=0)
                dq_tmem_free_ph = K.PipelineState(1, phase=1)

                accum_var = K.alloc_local([1], "int32")
                accum_dv = K.alloc_local([1], "int32")
                accum_dk = K.alloc_local([1], "int32")

                def phase_a():
                    """S = K @ Q_row^T, M=256 (128/CTA)."""
                    tma_q.wait(0, q_ph.phase)
                    q_ph.advance()

                def phase_a_issue():
                    K.assign(accum_var[0], 0)
                    with K.If(elected()), K.Then():
                        mma_chain2(
                            TMEM_OFF_A, a=d_k_row, b=q_row_operand(),
                            idesc=ID_SS, accumulate=accum_var[0],
                        )
                    with K.If(elected()), K.Then():
                        mma2wg0_s.arrive(0, cta_group=CTA_GROUP, cta_mask=pair_mask)
                        tcgen05_commit(q_consumed.ptr_to([0]))

                def phase_b():
                    """dP = V @ dO_row^K."""
                    tma_a.wait(0, a_ph.phase)
                    a_ph.advance()
                    K.assign(accum_var[0], 0)
                    with K.If(elected()), K.Then():
                        mma_chain2(
                            TMEM_OFF_DP, a=d_v_row, b=d_do_row,
                            idesc=ID_SS, accumulate=accum_var[0],
                        )
                    with K.If(elected()), K.Then():
                        mma2wg0_dp.arrive(0, cta_group=CTA_GROUP, cta_mask=pair_mask)

                def phase_c():
                    """dV += P^T @ dO_col; waits for the P strips."""
                    strip_ready.wait(0, strip_ready_ph.phase)
                    strip_ready_ph.advance()
                    with K.If(elected()), K.Then():
                        mma_chain2(
                            TMEM_OFF_B, a=TMEM_OFF_A * 2, b=d_do_col,
                            idesc=ID_TS, accumulate=accum_dv[0],
                        )
                    K.assign(accum_dv[0], 1)
                    with K.If(elected()), K.Then():
                        tcgen05_commit(buf_a_consumed.ptr_to([0]))

                def phase_d(*, is_last=False):
                    """dK += dS_tmem @ Q_col^K.

                    Once issued, ordered UMMA execution lets the following dP
                    reuse the dS/dP TMEM region.  ``is_last`` is the post-loop
                    Phase D[N-1]: it releases dK with the same elect, and skips
                    the accumulate flag no later trip reads.
                    """
                    wg02mma_tmem.wait(0, wg0_ph.phase)
                    wg0_ph.advance()
                    tma_qcol.wait(0, qcol_ph.phase)
                    qcol_ph.advance()
                    with K.If(elected()), K.Then():
                        mma_chain2(
                            TMEM_OFF_C, a=TMEM_OFF_DP, b=d_q_col,
                            idesc=ID_TS, accumulate=accum_dk[0],
                        )
                    with K.If(elected()), K.Then():
                        tcgen05_commit(qcol_consumed.ptr_to([0]))
                        if is_last:
                            # sdKVaccum stage 1: release dK independently after
                            # its final update, while the final dQ path is live.
                            tcgen05_commit(dk_done.ptr_to([0]))
                    if not is_last:
                        K.assign(accum_dk[0], 1)

                def phase_e_issue():
                    K.assign(accum_var[0], 0)
                    with K.If(elected()), K.Then():
                        mma_chain2(
                            TMEM_OFF_DQ, a=d_ds_exch, b=d_k_col,
                            idesc=ID_DQ, accumulate=accum_var[0],
                        )
                    with K.If(elected()), K.Then():
                        mma2wg0_dq.arrive(0, cta_group=CTA_GROUP, cta_mask=pair_mask)
                        tcgen05_commit(ds_exch_consumed.ptr_to([0]))

                tma_kv.wait(0, kv_ph.phase)
                kv_ph.advance()
                K.assign(accum_dv[0], 0)
                K.assign(accum_dk[0], 0)

                # ---- special first M-tile (i=0): A, B, C only ----
                phase_a()
                phase_a_issue()
                phase_b()
                phase_c()

                # ---- main loop: M-tiles 1..N-1 ----
                with K.serial(
                    num_m_tiles_this_n - 1, annotations={"disable_unroll": True}
                ) as _i_m:
                    # From the second trip onward Phase A also waits until the
                    # reducer drained the previous dQ tile aliasing the upper
                    # half of S.
                    phase_a()
                    dq_tmem_free.wait(0, dq_tmem_free_ph.phase)
                    dq_tmem_free_ph.advance()
                    phase_a_issue()
                    phase_d()
                    phase_b()
                    # Phase E[i-1]: dQ = dS_exch^T @ K_col. dQ aliases the upper
                    # half of S, so as in FA4's delayed dS producer commit this
                    # requires both the current dS exchange and the next tile's
                    # drained S loads.
                    wg02mma.wait(0, wg0_smem_ph.phase)
                    wg0_smem_ph.advance()
                    s_tmem_consumed.wait(0, s_tmem_consumed_ph.phase)
                    s_tmem_consumed_ph.advance()
                    phase_e_issue()
                    phase_c()

                # dV is complete before the remaining dK/dQ tail. Match
                # sdKVaccum stage 0 by releasing all compute warps now so its
                # epilogue overlaps the tail below.
                with K.If(elected()), K.Then():
                    tcgen05_commit(dv_done.ptr_to([0]))

                # ---- after the loop: Phase D[N-1], Phase E[N-1] ----
                phase_d(is_last=True)

                # The final dQ reuses the same TMEM destination; unlike a loop
                # trip there is no following Phase-A prologue to carry this
                # release wait, so consume it explicitly before the last write.
                dq_tmem_free.wait(0, dq_tmem_free_ph.phase)
                dq_tmem_free_ph.advance()
                wg02mma.wait(0, wg0_smem_ph.phase)
                wg0_smem_ph.advance()
                phase_e_issue()

                # Consume the final reducer release before TMEM teardown.
                dq_tmem_free.wait(0, dq_tmem_free_ph.phase)

            K.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
            K.ptx.barrier.sync(K.uint32(5), 416)
            tmem_dealloc_mbar.arrive(0, remote=1 - id_in_pair, pred=True)
            tmem_dealloc_mbar.wait(0, 0)
            tmem_dealloc_addr = K.alloc_local([1], "uint32")
            K.ptx.ld.shared.u32(tmem_dealloc_addr[0], tmem_addr.ptr_to([0]))
            K.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](
                tmem_dealloc_addr[0], K.uint32(512)
            )

        # ---- relay warp ---------------------------------------------------
        with relay:
            # Relay the per-CTA DSMEM completion to one leader-local
            # barrier, keeping the MMA warp off the cross-CTA wait path.
            relay_ph = K.PipelineState(1, phase=0)
            with K.serial(
                num_m_tiles_this_n, annotations={"disable_unroll": True}
            ) as _r:
                ds_exch_mbar.wait(0, relay_ph.phase)
                relay_ph.advance()
                with K.If(elected()), K.Then():
                    wg02mma.arrive(0, remote=0, pred=True)

        with idle:
            pass

        # ================================================================
        # WG1+WG2: softmax grad + dS + split epilogue
        # ================================================================
        with compute:
            K.ptx.barrier.sync(K.uint32(5), 416)
            role_warp = K.warp_id_in_role()
            compute_wg = role_warp // 4  # WG1 -> 0, WG2 -> 1
            warp_id = role_warp % 4
            wg_bar = compute_wg + 11  # the original's `wg_id + 10`
            row_local = warp_id * 32 + lane_id
            # WG0 -> strip 0 (cols 0:64), WG1 -> strip 1 (cols 64:128)
            strip_off = compute_wg * STRIP_SIZE

            gemm_s_ph = K.PipelineState(1, phase=0)
            lse_ph = K.PipelineState(1, phase=0)
            gemm_dp_ph = K.PipelineState(1, phase=0)
            dpsum_ph = K.PipelineState(1, phase=0)
            dv_done_ph = K.PipelineState(1, phase=0)
            dk_done_ph = K.PipelineState(1, phase=0)
            ds_exch_consumed_ph = K.PipelineState(1, phase=1)

            def fma_scale_sub_f32x2(scores, scale, lse):
                """``scores * scale - lse``, all three packed f32x2."""
                neg_lse = K.alloc_local([1], "uint64")
                result = K.alloc_local([1], "uint64")
                sign_mask = K.bitwise_or(
                    K.shift_left(K.uint64(0x80000000), K.uint64(32)), K.uint64(0x80000000)
                )
                K.ptx.xor.b64(neg_lse[0], lse, sign_mask)
                K.ptx.fma.rn.f32x2(result[0], scores, scale, neg_lse[0])
                return result[0]

            with K.serial(
                num_m_tiles_this_n, annotations={"disable_unroll": True}
            ) as i_m_inner:
                i_m = m_tile_start + i_m_inner
                m_st_val = i_m * BLK_M

                # LSE is an independent CTA-local stream: consumed and released
                # as soon as P is formed, while Q_row is released by Phase A.
                tma_lse.wait(0, lse_ph.phase)

                # ---- wait Phase A: S^T ready in TMEM ----
                mma2wg0_s.wait(0, gemm_s_ph.phase)
                gemm_s_ph.advance()

                S_strip = K.alloc_local([STRIP_SIZE], "float32")
                tmem_s_col = TMEM_OFF_A + strip_off
                for stage in range(STRIP_SIZE // 32):
                    tmem_load(32, S_strip, stage * 32, TMEM_OFF_A, 0, tmem_s_col + stage * 32)
                # D=128 aliases dQ with the upper half of S. Match FA4's delayed
                # dS commit: publish the previous tile's dQ-safe token as soon as
                # this warp drained the next tile's S load, before any P math.
                K.ptx.tcgen05.wait__ld.sync.aligned()
                with K.If((i_m_inner > 0) & (lane_id == 0)), K.Then():
                    s_tmem_consumed.arrive(0, remote=0, pred=True)

                # P^T = exp2(S^T * scale_log2 - LSE_log2[m]); the base conversion
                # is hoisted into preprocess.
                P_f16 = K.alloc_local([STRIP_SIZE], "float16")
                P_f16_u32 = P_f16.view("uint32")
                tmem_p_col = TMEM_OFF_A * 2 + strip_off
                for stage in range(STRIP_SIZE // 32):
                    for j_inner in range(32 // 2):
                        j = stage * (32 // 2) + j_inner
                        lse_pair = K.alloc_local([2], "float32")
                        K.ptx.ld.shared.v2.f32(
                            lse_pair[0], lse_pair[1], sLSE.ptr_to([0, strip_off + 2 * j])
                        )
                        scaled_pair = fma_scale_sub_f32x2(
                            K.cuda.make_float2(S_strip[2 * j], S_strip[2 * j + 1]),
                            K.cuda.make_float2(K.float32(scale_log2), K.float32(scale_log2)),
                            K.cuda.make_float2(lse_pair[0], lse_pair[1]),
                        )
                        K.ptx.mov.b32(S_strip[2 * j], K.cuda.float2_x(scaled_pair))
                        K.ptx.mov.b32(S_strip[2 * j + 1], K.cuda.float2_y(scaled_pair))
                        K.ptx.ex2.approx.ftz.f32(S_strip[2 * j], S_strip[2 * j])
                        K.ptx.ex2.approx.ftz.f32(S_strip[2 * j + 1], S_strip[2 * j + 1])
                    if causal:
                        # Only the first two Q tiles overlapping this 256-row
                        # K/V cluster can intersect the diagonal. The uniform
                        # branch is hoisted out of the unrolled element loop so
                        # later tiles pay one branch per 32-column stage.
                        with K.If(i_m < m_tile_start + BLK_N // BLK_M), K.Then():
                            key_idx = n_st_cta + row_local
                            for j_inner in range(32 // 2):
                                j = stage * (32 // 2) + j_inner
                                query_idx_0 = m_st_val + strip_off + 2 * j
                                K.ptx.mov.b32(S_strip[2 * j], K.if_then_else(
                                    query_idx_0 >= key_idx, S_strip[2 * j], K.float32(0)
                                ))
                                K.ptx.mov.b32(S_strip[2 * j + 1], K.if_then_else(
                                    query_idx_0 + 1 >= key_idx,
                                    S_strip[2 * j + 1],
                                    K.float32(0),
                                ))
                    for j_inner in range(32 // 2):
                        j = stage * (32 // 2) + j_inner
                        K.cuda.float22half2(
                            K.address_of(P_f16[2 * j]), K.address_of(S_strip[2 * j])
                        )

                    # P aliases S in TMEM: once the first 32-column register
                    # stage is ready, drain both S loads and synchronize the
                    # compute warps before the first store.
                    if stage == 0:
                        K.ptx.bar.sync(K.uint32(8), 256)

                    tmem_store(
                        16, P_f16_u32, stage * 16, TMEM_OFF_A, 0,
                        (tmem_p_col + stage * 32) // 2,
                    )
                lse_ph.advance()
                K.ptx.tcgen05.wait__st.sync.aligned()
                K.ptx.fence.proxy.async_.shared__cta()
                K.ptx.bar.sync(K.uint32(8), 256)
                # Bridge the generic LSE reads into the async proxy, then publish
                # the release only after the rendezvous ordered every reader.
                with K.If(lane_id == 0), K.Then():
                    lse_consumed.arrive(0)

                with K.If(elected()), K.Then():
                    strip_ready.arrive(0, remote=0, pred=True)

                # ---- Phase B: wait for dPsum and dP^T, read strip ----
                tma_dpsum.wait(0, dpsum_ph.phase)
                mma2wg0_dp.wait(0, gemm_dp_ph.phase)
                gemm_dp_ph.advance()

                dP_strip = K.alloc_local([STRIP_SIZE], "float32")
                # .f32x2 writes its destination as one packed 64-bit register,
                # so address the same storage in pairs.
                dP_pairs = dP_strip.view("uint64")
                tmem_dp_col = TMEM_OFF_DP + strip_off
                for stage in range(STRIP_SIZE // 32):
                    tmem_load(32, dP_strip, stage * 32, TMEM_OFF_A, 0, tmem_dp_col + stage * 32)

                # dS^T[n,m] = P^T[n,j] * (dP^T[n,j] - dpsum[m])
                for j in range(STRIP_SIZE // 2):
                    dpsum_pair = K.alloc_local([2], "float32")
                    K.ptx.ld.shared.v2.f32(
                        dpsum_pair[0], dpsum_pair[1], sDPsum.ptr_to([strip_off + 2 * j])
                    )
                    K.ptx.sub.rn.ftz.f32x2(
                        dP_pairs[j],
                        K.cuda.make_float2(dP_strip[2 * j], dP_strip[2 * j + 1]),
                        K.cuda.make_float2(dpsum_pair[0], dpsum_pair[1]),
                    )
                    K.ptx.mul.rn.ftz.f32x2(
                        dP_pairs[j],
                        K.cuda.make_float2(S_strip[2 * j], S_strip[2 * j + 1]),
                        K.cuda.make_float2(dP_strip[2 * j], dP_strip[2 * j + 1]),
                    )

                dpsum_ph.advance()
                K.ptx.tcgen05.wait__ld.sync.aligned()

                dS_full_f16 = K.alloc_local([STRIP_SIZE], "float16")
                for j in range(STRIP_SIZE // 2):
                    K.cuda.float22half2(
                        K.address_of(dS_full_f16[2 * j]), K.address_of(dP_strip[2 * j])
                    )

                K.ptx.bar.sync(K.uint32(8), 256)
                tmem_store(
                    32, dS_full_f16.view("uint32"), 0, TMEM_OFF_A, 0,
                    (TMEM_OFF_DP * 2 + strip_off) // 2,
                )
                K.ptx.tcgen05.wait__st.sync.aligned()

                # The dK MMA only consumes dS from TMEM: release that dependency
                # before the independent SMEM visibility / DSMEM exchange path.
                with K.If(elected()), K.Then():
                    wg02mma_tmem.arrive(0, remote=0, pred=True)

                # The DSMEM path is independent of dK. The buffer is
                # single-stage, so wait until the preceding dQ MMA stopped
                # reading it before publishing the next tile.
                ds_exch_consumed.wait(0, ds_exch_consumed_ph.phase)
                # The empty barrier orders TCGEN completion; this proxy fence
                # bridges that completed async-proxy read to the generic stores
                # that reuse dS_exch.
                K.ptx.fence.proxy.async_.shared__cta()
                ds_exch_consumed_ph.advance()
                # Logical (row, col) into the swizzled tile: the composed layout
                # re-applies the xor the original spells by hand with a signed
                # stride table (`RowiseSwizzleOffset`).
                with K.If(compute_wg == id_in_pair):
                    with K.Then():
                        for ni in range(STRIP_SIZE // 8):
                            K.ptx.st.weak.shared__cta.b128(
                                dS_exch.ptr_to(id_in_pair * CTA_N + row_local, ni * 8),
                                dS_full_f16.view("uint128")[ni],
                            )
                    with K.Else():
                        for ni in range(STRIP_SIZE // 8):
                            K.ptx.st.weak.shared__cta.b128(
                                dS_send.ptr_to(row_local, ni * 8),
                                dS_full_f16.view("uint128")[ni],
                            )

                K.ptx.fence.proxy.async_.shared__cta()
                K.ptx.bar.sync(K.uint32(8), 256)

                # This fence/rendezvous bridges both the dS stores used by the
                # DSMEM copy and the completed generic dPsum reads.
                with K.If(lane_id == 0), K.Then():
                    dpsum_consumed.arrive(0)

                # The sender WG starts the peer copy directly. Its source has
                # independent storage, so the load warp can reuse dO as soon as
                # Phase C commits, without a relay-warp drain.
                with K.If((compute_wg != id_in_pair) & (warp_id == 0) & (lane_id == 0)), K.Then():
                    peer_cta = 1 - id_in_pair
                    ds_copy_bytes = CTA_N * B_N * DTYPE_SIZE
                    # mapa writes its result, so the peer-window addresses are
                    # computed into registers first.
                    remote_mbar = K.alloc_local([1], "uint32")
                    K.ptx.mapa.shared__cluster.u32(
                        remote_mbar[0],
                        K.cuda.cvta_generic_to_shared(ds_exch_mbar.ptr_to([0])),
                        K.uint32(peer_cta),
                    )
                    remote_dst = K.alloc_local([1], "uint32")
                    K.ptx.mapa.shared__cluster.u32(
                        remote_dst[0],
                        K.cuda.cvta_generic_to_shared(dS_exch.ptr_to(id_in_pair * CTA_N, 0)),
                        K.uint32(peer_cta),
                    )
                    K.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(
                        remote_mbar[0], K.uint32(ds_copy_bytes), pred=True
                    )
                    K.ptx[BULK_S2C](
                        remote_dst[0],
                        dS_send.ptr_to(0, 0),
                        K.uint32(ds_copy_bytes),
                        remote_mbar[0],
                    )

            # ---- two-stage dKV epilogue ----
            # Both compute WGs split dV by 64-column halves, then dK the same
            # way. Stage 0 overlaps the MMA dK/dQ tail.
            def epilogue_stage(done_bar, done_ph, tmem_off, epi_tile, tensormap, scale_by=None):
                done_bar.wait(0, done_ph.phase)
                done_ph.advance()
                acc = K.alloc_local([EPI_N], "float32")
                acc_pairs = acc.view("uint64")
                out = K.alloc_local([EPI_N], "float16")
                tmem_load(64, acc, 0, TMEM_OFF_A, 0, tmem_off + compute_wg * EPI_N)
                K.ptx.tcgen05.wait__ld.sync.aligned()
                for j in range(EPI_N // 2):
                    if scale_by is not None:
                        K.ptx.mul.rn.ftz.f32x2(
                            acc_pairs[j],
                            K.cuda.make_float2(acc[2 * j], acc[2 * j + 1]),
                            K.cuda.make_float2(K.float32(scale_by), K.float32(scale_by)),
                        )
                    K.cuda.float22half2(K.address_of(out[2 * j]), K.address_of(acc[2 * j]))
                for ni in range(EPI_N // 8):
                    K.ptx.st.weak.shared__cta.b128(
                        epi_tile[compute_wg].ptr_to(row_local, ni * 8),
                        out.view("uint128")[ni],
                    )
                K.ptx.fence.proxy.async_.shared__cta()
                K.ptx.bar.sync(K.Cast("uint32", wg_bar), 128)
                with K.If((warp_id == 0) & (lane_id == 0)), K.Then():
                    tma_s2g(
                        4, epi_tile[compute_wg].ptr_to(0, 0), tensormap,
                        compute_wg * EPI_N, n_st_cta, h_idx, b_idx,
                    )
                with K.If(warp_id == 0), K.Then():
                    K.ptx.bar.arrive(K.Cast("uint32", wg_bar), 160)
                K.ptx.fence.proxy.async_.shared__cta()
                K.ptx.bar.sync(K.Cast("uint32", wg_bar), 160)

            epilogue_stage(dv_done, dv_done_ph, TMEM_OFF_B, dV_epi, dv_map)
            epilogue_stage(dk_done, dk_done_ph, TMEM_OFF_C, dK_epi, dk_map, softmax_scale)
            # Keep both terminal dV/dK stores asynchronous but in one explicit
            # bulk group so their lifetime stays visible; neither source is
            # reused.
            with K.If((warp_id == 0) & (lane_id == 0)), K.Then():
                K.ptx.cp.async_.bulk.commit_group()
            K.ptx.bar.arrive(K.uint32(5), 416)

        # ================================================================
        # WG0: dQ reduce (TMEM -> SMEM -> TMA reduce)
        # ================================================================
        with dq_reduce:
            K.ptx.barrier.sync(K.uint32(5), 416)
            warp_id = K.warp_id_in_role()
            row_local = warp_id * 32 + lane_id
            gemm_dq_ph = K.PipelineState(1, phase=0)

            with K.serial(
                num_m_tiles_this_n, annotations={"disable_unroll": True}
            ) as i_m_inner:
                m_st_val = (m_tile_start + i_m_inner) * BLK_M

                mma2wg0_dq.wait(0, gemm_dq_ph.phase)
                gemm_dq_ph.advance()

                # Datapath-B readback is a physical (128, 64) image of the
                # logical (64, 128) dQ tile.
                dQ_full = K.alloc_local([64], "float32")
                tmem_load(32, dQ_full, 32, TMEM_OFF_DQ, warp_id * 32, 0)
                tmem_load(32, dQ_full, 0, TMEM_OFF_DQ, warp_id * 32, 32)

                # Drain dQ before releasing the shared dP/dQ region.
                K.ptx.tcgen05.wait__ld.sync.aligned()
                K.cuda.warp_sync()
                with K.If(elected()), K.Then():
                    dq_tmem_free.arrive(0, remote=0, pred=True)

                m_st_cta = m_st_val + id_in_pair * DQ_M_PER_CTA
                # Materialize elect_sync while all 32 lanes are converged.
                dq_reduce_elected = K.alloc_local([1], "uint32")
                K.assign(dq_reduce_elected[0], K.cuda.elect_sync())

                for stage in range(DQ_REDUCE_ITERS):
                    smem_slot = stage % DQ_STAGES
                    dq_reg_st = ((stage + DQ_REDUCE_ITERS // 2) % DQ_REDUCE_ITERS) * DQ_RED_N
                    for chunk in range(DQ_RED_N // 4):
                        K.ptx.st.weak.shared__cta.b128(
                            dQ_smem.ptr_to([smem_slot, chunk * BLK_M * 4 + row_local * 4]),
                            dQ_full.view("uint128")[(dq_reg_st + chunk * 4) // 4],
                        )
                    K.ptx.fence.proxy.async_.shared__cta()
                    K.cuda.warpgroup_sync(4)

                    with K.If(warp_id == 0), K.Then():
                        with K.If(dq_reduce_elected[0] != K.uint32(0)), K.Then():
                            K.ptx[TMA_S2G_REDUCE.format(dim=4, redop="add")](
                                K.address_of(dq_map),
                                0,
                                m_st_cta + stage * DQ_ROWS_PER_STAGE,
                                h_idx,
                                b_idx,
                                dQ_smem.ptr_to([smem_slot, 0]),
                            )
                        K.ptx.cp.async_.bulk.commit_group()
                        K.ptx.cp.async_.bulk.wait_group(DQ_STAGES - 1)
                    K.cuda.warpgroup_sync(4)

            # Wait for all pending TMA reduces
            with K.If(warp_id == 0), K.Then():
                K.ptx.cp.async_.bulk.wait_group(0)
            K.cuda.warpgroup_sync(4)
            K.ptx.bar.arrive(K.uint32(5), 416)

    # fmt: on
    return kernel


@cache
def _compile_pipeline(B: int, H: int, S: int, D: int, causal: bool, attention_scale: float):
    return (
        build_preprocess(B, S, H, D).compile(),
        build_kernel(B, H, S, D, causal=causal, attention_scale=attention_scale).compile(),
        build_cast_f32_to_f16(B, S, H, D, attention_scale).compile(),
    )


def setup(data, B, H, S, D, *, executables=None):
    """Prepare GPU resources and return one full backward-pipeline launch."""
    Q = data["Q"]
    K_t = data["K"]
    V = data["V"]
    O = data["O"]
    dO = data["dO"]
    LSE = data["LSE"]
    causal = bool(data.get("causal", False))
    attention_scale = float(data.get("softmax_scale", 1.0 / math.sqrt(D)))

    dpsum = torch.empty(B, H, S, dtype=torch.float32, device="cuda")
    LSE_log2 = torch.empty_like(LSE)
    dQ_acc = torch.empty(B, H, S, D, dtype=torch.float32, device="cuda")
    dK = data["dK"]
    dV = data["dV"]
    dQ = data["dQ"]

    if executables is None:
        executables = _compile_pipeline(B, H, S, D, causal, attention_scale)
    preprocess_ex, kernel_ex, cast_ex = executables

    # The descriptors are by-value kernel parameters here rather than a host
    # prologue inside the PrimFunc, so they are encoded once for these tensors
    # instead of on every launch.
    tensor_maps = build_tensor_maps(Q, K_t, V, dO, dK, dV, dQ_acc, B, H, S, D)

    pre_args = (
        dO.view(-1),
        O.view(-1),
        LSE.view(-1),
        dpsum.view(-1),
        LSE_log2.view(-1),
        dQ_acc.view(-1),
    )
    core_args = (
        Q.view(-1),
        K_t.view(-1),
        V.view(-1),
        dO.view(-1),
        LSE_log2.view(-1),
        dpsum.view(-1),
        dK.view(-1),
        dV.view(-1),
        dQ_acc.view(-1),
        *(m.ptr for m in tensor_maps),
    )
    cast_args = (dQ_acc.view(-1), dQ.view(-1))

    def run_all():
        preprocess_ex(*pre_args)
        kernel_ex(*core_args)
        cast_ex(*cast_args)

    # Launch parameters retain only descriptor addresses; keep their storage
    # alive for every reuse of the returned closure.
    run_all.tensor_maps = tensor_maps

    run_all()

    data["dQ"] = dQ
    data["dK"] = dK
    data["dV"] = dV
    return run_all


KERNEL_META = {
    "name": "flash_attention_backward_sm100",
    "category": "flashattention",
    "compute_capability": 10,
}

CONFIGS = [
    {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_heads": 16,
        "head_dim": 128,
        "is_causal": is_causal,
        "label": (f"b{batch_size}_s{seq_len}_h16_{'causal' if is_causal else 'noncausal'}"),
    }
    for batch_size, seq_len, is_causal in (
        (1, 2048, True),
        (1, 4096, True),
        (2, 4096, True),
        (1, 8192, True),
        (1, 8192, False),
        # flash-attn's default 32k-token sweep yields these batch-four shapes.
        (4, 8192, True),
        (4, 8192, False),
    )
]


def get_kernel(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int = 128,
    is_causal: bool = False,
    **kwargs,
):
    """Return the raw backward core PrimFunc for the registry configuration."""
    return build_kernel(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        causal=is_causal,
        attention_scale=1.0 / math.sqrt(head_dim),
    ).func


def _prepare_official_workload(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    is_causal: bool,
    *,
    compute_backward_reference: bool = True,
):
    """Create saved forward tensors and the current FA4 backward reference."""
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    torch.manual_seed(0)
    shape = (batch_size, seq_len, num_heads, head_dim)
    q = (torch.randn(shape, dtype=torch.float32, device="cuda") * 0.5).half()
    k = (torch.randn(shape, dtype=torch.float32, device="cuda") * 0.5).half()
    v = (torch.randn(shape, dtype=torch.float32, device="cuda") * 0.5).half()
    dout = (torch.randn(shape, dtype=torch.float32, device="cuda") * 0.25).half()
    scale = 1.0 / math.sqrt(head_dim)
    with torch.no_grad():
        out, lse = _flash_attn_fwd(
            q=q, k=k, v=v, softmax_scale=scale, causal=is_causal, return_lse=True
        )[:2]
        expected = (
            _flash_attn_bwd(
                q=q, k=k, v=v, out=out, dout=dout, lse=lse, softmax_scale=scale, causal=is_causal
            )
            if compute_backward_reference
            else None
        )
    data = {
        "Q": q,
        "K": k,
        "V": v,
        "O": out,
        "dO": dout,
        "LSE": lse,
        "dQ": torch.empty_like(q),
        "dK": torch.empty_like(k),
        "dV": torch.empty_like(v),
        "causal": is_causal,
        "softmax_scale": scale,
    }
    return data, expected


def run_test(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int = 128,
    is_causal: bool = False,
    **kwargs,
):
    """Compile the three-kernel pipeline and compare it with current FA4."""
    data, expected = _prepare_official_workload(batch_size, seq_len, num_heads, head_dim, is_causal)
    setup(data, batch_size, num_heads, seq_len, head_dim)
    torch.cuda.synchronize()
    for name, actual, reference in zip(
        ("dQ", "dK", "dV"), (data["dQ"], data["dK"], data["dV"]), expected, strict=True
    ):
        if not torch.isfinite(actual).all():
            raise AssertionError(f"{name} contains a non-finite value")
        matched = torch.isclose(actual, reference, rtol=0.1, atol=0.1).float().mean()
        if matched.item() < 0.995:
            raise AssertionError(f"{name} matched ratio {matched.item():.6f} is below 0.995")


def prepare_bench(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int = 128,
    is_causal: bool = False,
    **kwargs,
):
    """Compile the preprocess/core/cast pipeline before CUDA setup."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    scale = 1.0 / math.sqrt(head_dim)
    state = {
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "is_causal": is_causal,
            **kwargs,
        },
        "executables": _compile_pipeline(
            batch_size, num_heads, seq_len, head_dim, is_causal, scale
        ),
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, **kwargs):
    """Benchmark the full preprocess/core/cast pipeline against current FA4."""
    from flash_attn.cute.interface import _flash_attn_bwd

    from tirx_kernels.runner import bench

    config = {**prepared["config"], **kwargs}
    batch_size = config.pop("batch_size")
    seq_len = config.pop("seq_len")
    num_heads = config.pop("num_heads")
    head_dim = config.pop("head_dim")
    is_causal = config.pop("is_causal")
    data, _ = _prepare_official_workload(
        batch_size, seq_len, num_heads, head_dim, is_causal, compute_backward_reference=False
    )
    candidate = setup(
        data, batch_size, num_heads, seq_len, head_dim, executables=prepared["executables"]
    )

    def official_factory():
        def run():
            return _flash_attn_bwd(
                q=data["Q"],
                k=data["K"],
                v=data["V"],
                out=data["O"],
                dout=data["dO"],
                lse=data["LSE"],
                softmax_scale=data["softmax_scale"],
                causal=data["causal"],
            )

        return run

    return bench(
        {"tir": candidate},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashattn_sm100": official_factory},
        **config,
    )


def run_bench(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int = 128,
    is_causal: bool = False,
    warmup=None,
    repeat=None,
    timer=None,
    **kwargs,
):
    rounds = kwargs.pop("rounds", None)
    cooldown_s = kwargs.pop("cooldown_s", None)
    protocol = {"warmup": warmup, "repeat": repeat, "timer": timer}
    if rounds is not None:
        protocol["rounds"] = rounds
    if cooldown_s is not None:
        protocol["cooldown_s"] = cooldown_s
    return prepare_bench(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        is_causal=is_causal,
        **kwargs,
    ).run_gpu(**protocol)
