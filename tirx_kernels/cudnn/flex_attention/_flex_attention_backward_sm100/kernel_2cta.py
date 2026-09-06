# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ edc7c2833327d699d70b616c6f264b6ae92599b2), Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""CTA-group-two SM100 Flex Attention backward specializations.

The implementation follows
``python/cudnn/flex_attention/kernels/sm100/bwd/backward.py`` and keeps every
shared-memory object in one linear dynamic arena.  The scalar byte formulas and
matrix descriptors below encode the source swizzles and transpose views.
"""

import math

import tirx_kernels.kern as K

from .kernel import (
    _desc_add16,
    _desc_at,
    _epi_bf16_byte,
    _load_i32,
    _packed_binary,
    _packed_fma,
    _PipelinePair,
    _release_inc_i32,
    _tile_byte,
    _tmem_load,
    _tmem_load32,
    _tmem_store16,
    _wait_eq_i32,
)

CTA_GROUP = 2
WARPS = 16
TMEM_COLUMNS = 512
SHARED_BYTES = 231424
BAR_COMPUTE = (3, 256)
BAR_REDUCE = (4, 128)
BAR_TMEM = (5, 416)
MMA_F16 = "tcgen05.mma.cta_group::2.kind::f16"
TMA_S2G = "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
BULK_G2S = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
BULK_S2C = "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
TCGEN_COMMIT = (
    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
)
TMA_CACHE = K.uint64(0)


def _mma2(dest, a, b, idesc, accumulate):
    K.ptx[MMA_F16](
        K.cast(dest, "uint32"),
        a,
        b,
        K.uint32(idesc),
        K.uint32(0),
        K.uint32(0),
        K.uint32(0),
        K.uint32(0),
        K.uint32(0),
        K.uint32(0),
        K.uint32(0),
        K.uint32(0),
        K.ptx.pred(accumulate),
    )


def _mma2_ss(dest, a, b, idesc, a_offsets, b_offsets, accumulate):
    flag = K.local_scalar("uint32", init=K.cast(accumulate, "uint32"))
    for a_offset, b_offset in zip(a_offsets, b_offsets):
        _mma2(dest, _desc_add16(a, a_offset), _desc_add16(b, b_offset), idesc, flag)
        K.assign(flag, K.uint32(1))


def _mma2_dq(dest, a, b, idesc, a_offsets, b_offsets):
    flag = K.local_scalar("uint32", init=K.uint32(0))
    for a_offset, b_offset in zip(a_offsets, b_offsets):
        _mma2(dest, _desc_add16(a, a_offset), _desc_add16(b, b_offset), idesc, flag)
        K.assign(flag, K.uint32(1))


def _mma2_ts(dest, a, b, idesc, b_offsets, accumulate):
    flag = K.local_scalar("uint32", init=K.cast(accumulate, "uint32"))
    for phase, b_offset in enumerate(b_offsets):
        _mma2(
            dest, K.cast(a + K.uint32(phase * 8), "uint32"), _desc_add16(b, b_offset), idesc, flag
        )
        K.assign(flag, K.uint32(1))


def _commit2(barrier, pair_mask):
    K.ptx[TCGEN_COMMIT](barrier, K.cast(pair_mask, "uint16"))


def _issue_cluster_tma(opcode, varlen, dst, tensor_map, feature, seq, head, batch, barrier):
    if varlen:
        K.ptx[opcode](dst, tensor_map, feature, seq, head, barrier, TMA_CACHE)
    else:
        K.ptx[opcode](dst, tensor_map, feature, seq, head, batch, barrier, TMA_CACHE)


def get_kernel_2cta(**config):
    """Build the exact D128 or D192 cooperative specialization."""
    head_dim = int(config["head_dim"])
    head_dim_v = int(config["head_dim_v"])
    if (head_dim, head_dim_v) not in ((128, 128), (192, 128)):
        raise ValueError("cooperative Flex Attention backward requires D128 or D192 with Dv128")
    is_d192 = head_dim == 192

    varlen = bool(config.get("varlen", False))
    tma_g2s = (
        f"cp.async.bulk.tensor.{3 if varlen else 4}d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
    )
    batch = int(config["batch"])
    heads = int(config["num_q_heads"])
    kv_heads = int(config["num_kv_heads"])
    if varlen:
        q_lengths = tuple(int(value) for value in config["seqlen_q"])
        k_lengths = tuple(int(value) for value in config["seqlen_kv"])
    else:
        q_lengths = (int(config["seqlen_q"]),) * batch
        k_lengths = (int(config["seqlen_kv"]),) * batch
    seqlen_q = max(q_lengths)
    seqlen_kv = max(k_lengths)
    q_offsets = tuple(sum(q_lengths[:index]) for index in range(batch)) if varlen else (0,) * batch
    k_offsets = tuple(sum(k_lengths[:index]) for index in range(batch)) if varlen else (0,) * batch
    q_padded_offsets = tuple(
        (q_offsets[index] + index * 128) // 128 * 128 for index in range(batch)
    )
    k_padded_offsets = tuple(
        (k_offsets[index] + index * 256) // 256 * 256 for index in range(batch)
    )
    q_blocks_by_batch = tuple((value + 127) // 128 for value in q_lengths)
    k_pairs_by_batch = tuple((value + 255) // 256 for value in k_lengths)
    pair_offsets = tuple(sum(k_pairs_by_batch[:index]) for index in range(batch))
    q_blocks = max(q_blocks_by_batch)
    tasks = max(k_pairs_by_batch)
    total_tasks = sum(k_pairs_by_batch)
    q128 = q_blocks * 128
    k128 = max((value + 255) // 256 for value in k_lengths) * 256
    q_storage_rows = (
        (sum(q_lengths) + (batch + 1) * 128 - 1) // 128 * 128 if varlen else batch * q128
    )
    k_storage_rows = (
        (sum(k_lengths) + (batch + 1) * 256 - 1) // 256 * 256 if varlen else batch * k128
    )
    qhead_per_kvhead = heads // kv_heads
    deterministic = bool(config.get("deterministic", False))
    dtype = config["dtype"]
    elem_type = {"float16": K.f16, "bfloat16": K.bf16}[dtype]
    cvt_pack = {"float16": "cvt.rn.f16x2.f32", "bfloat16": "cvt.rn.bf16x2.f32"}[dtype]
    dtype_bits = 0x10 if dtype == "float16" else 0x490
    mask_heads = heads if "per_head" in config.get("mask_head_mode", "broadcast") else 1

    sum_plane = heads * q_storage_rows
    dq_base = 2 * sum_plane
    dk_base = dq_base + heads * q_storage_rows * head_dim
    dv_base = dk_base + kv_heads * k_storage_rows * head_dim
    direct_dkv = qhead_per_kvhead == 1
    direct_raw_dkv = direct_dkv and varlen
    fixed_gqa_task_major = (
        not varlen
        and batch == 1
        and q_lengths == (8193,)
        and k_lengths == (16385,)
        and heads == 8
        and kv_heads == 1
        and dtype == "bfloat16"
        and config.get("mask_type") == "mixed"
        and config.get("mask_head_mode", "broadcast") == "per_head"
        and int(config.get("mask_nfunc", 1)) == 19
    )
    p06_longformer_closed_plan = (
        not varlen
        and batch == 1
        and q_lengths == (131072,)
        and k_lengths == (131072,)
        and heads == 4
        and kv_heads == 4
        and (head_dim, head_dim_v) == (128, 128)
        and dtype == "bfloat16"
        and not deterministic
        and config.get("mask_type") == "longformer"
        and config.get("mask_head_mode", "broadcast") == "broadcast"
        and int(config.get("mask_nfunc", 1)) == 1
    )
    p00_causal_closed_plan = (
        not varlen
        and batch == 1
        and q_lengths == (131072,)
        and k_lengths == (131072,)
        and heads == 4
        and kv_heads == 4
        and (head_dim, head_dim_v) == (128, 128)
        and dtype == "bfloat16"
        and not deterministic
        and config.get("mask_type") == "causal"
        and config.get("mask_head_mode", "broadcast") == "broadcast"
        and int(config.get("mask_nfunc", 1)) == 1
    )
    long_irregular_task_major_2d = (
        not varlen
        and batch == 1
        and q_lengths == (131072,)
        and k_lengths == (131072,)
        and heads == 4
        and kv_heads == 4
        and dtype == "bfloat16"
        and not deterministic
        and config.get("mask_head_mode", "broadcast") == "broadcast"
        and int(config.get("mask_nfunc", 1)) == 1
        and (
            (
                (head_dim, head_dim_v) == (128, 128)
                and config.get("mask_type") in {"sink_local", "tree_dfs"}
            )
            or (
                (head_dim, head_dim_v) == (192, 128)
                and config.get("mask_type") == "document_causal"
            )
        )
    )
    use_source_varlen_nondet_schedule = varlen and not deterministic

    dq_ncol = 24 if is_d192 else (16 if deterministic else 8)
    dq_smem_stages = 2 if is_d192 or deterministic else 4
    dq_slices = (head_dim // 2) // dq_ncol
    dq_stage_bytes = 128 * dq_ncol * 4

    if is_d192:
        off_sq = 1024
        off_sk = 25600
        off_sv = 74752
        off_sdo = 107520
        off_sqt = off_sq
        off_sdot = off_sdo
        off_sdsx = 206848
        off_skt = 123904
        off_sds = 173056
        off_slse = 205824
        off_ssum = 206336
        off_sdq = 206848
        t_v = 0
        t_dk = 128
        t_s = 320
        t_p = 320
        t_dp = 384
        t_ds = 384
        t_dq = 416
    else:
        off_sq = 1024
        off_sk = 17408
        off_sv = 50176
        off_sdo = 82944
        off_sqt = 99328
        off_sdot = 115712
        off_sdsx = 132096
        off_skt = 148480
        off_sds = 181248
        off_slse = 214016
        off_ssum = 214528
        off_sdq = 215040
        t_s = 0
        t_p = 0
        t_dq = 64
        t_v = 128
        t_dp = 256
        t_ds = 256
        t_dk = 384
    id_ss = 0x10200000 | dtype_bits
    id_ts = 0x10210000 | dtype_bits
    id_dq = (0x08318000 if is_d192 else 0x08218000) | dtype_bits
    id_dk = (0x10310000 if is_d192 else 0x10210000) | dtype_bits
    # The cooperative D128 source encodes both row-major score operands with
    # LBO=1 descriptor units (low word bit 16).  This differs from the
    # one-CTA descriptor's larger LDO field even though the logical tiles have
    # the same element extents.
    desc_krow_base = 0x4000404000010000
    desc_row_base = 0x4000404000010000
    desc_mn_base = 0x4000404000000000
    krow_offsets = tuple(
        block * 1024 + inner for block in range(3 if is_d192 else 2) for inner in (0, 2, 4, 6)
    )
    row_offsets = tuple(
        block * 512 + inner for block in range(3 if is_d192 else 2) for inner in (0, 2, 4, 6)
    )
    # dP reduces over Dv=128 even when score reduces over D=192.
    dv_krow_offsets = tuple(block * 1024 + inner for block in range(2) for inner in (0, 2, 4, 6))
    do_row_offsets = tuple(block * 512 + inner for block in range(2) for inner in (0, 2, 4, 6))
    mn8_offsets = tuple(phase * 128 for phase in range(8))
    dk_offsets = tuple(phase * (64 if is_d192 else 128) for phase in range(8))
    # dQ reduces over the full 256-key CTA-group tile.  The source emits two
    # eight-instruction unroll groups for both cooperative specializations.
    dq_a_offsets = tuple(phase * 128 for phase in range(16))
    dq_b_offsets = tuple(phase * (64 if is_d192 else 128) for phase in range(16))

    @K.kernel(
        warps=WARPS,
        arch="sm_100a",
        min_blocks_per_sm=1,
        grid=(
            (heads * 2, tasks)
            if long_irregular_task_major_2d
            else (
                (tasks * heads * 2, 1)
                if fixed_gqa_task_major
                else (
                    (total_tasks * heads * 2, 1)
                    if use_source_varlen_nondet_schedule
                    else ((total_tasks if varlen else tasks * batch) * 2, heads)
                )
            )
        ),
    )
    def bwd(
        q_map: K.TensorMap,
        qt_map: K.TensorMap,
        k_map: K.TensorMap,
        kt_map: K.TensorMap,
        v_map: K.TensorMap,
        do_map: K.TensorMap,
        dot_map: K.TensorMap,
        dv_map: K.TensorMap,
        dk_map: K.TensorMap,
        dk_output: K.gptr[elem_type],
        dv_output: K.gptr[elem_type],
        cu_q: K.gptr[K.i32],
        cu_k: K.gptr[K.i32],
        partial_count: K.gptr[K.i32],
        partial_index: K.gptr[K.i32],
        full_count: K.gptr[K.i32],
        full_index: K.gptr[K.i32],
        cu_k_blocks: K.gptr[K.i32],
        dq_write_order: K.gptr[K.i32],
        dq_write_order_full: K.gptr[K.i32],
        partial_offset: K.gptr[K.i32],
        full_offset: K.gptr[K.i32],
        packed_mask: K.gptr[K.u32],
        sequence_desc: K.gptr[K.i32],
        work_desc: K.gptr[K.i32],
        dq_semaphore: K.gptr[K.i32],
        dk_semaphore: K.gptr[K.i32],
        dv_semaphore: K.gptr[K.i32],
        workspace: K.gptr[K.f32],
        softmax_scale: K.f32,
    ):
        # The source materializes ``block_idx_in_cluster`` with
        # ``cute.arch.make_warp_uniform``.  Keep the single broadcast here so
        # every rank-derived hot-loop address remains on the uniform datapath.
        cluster_rank = K.uniform(K.cta_id_in_cluster([2], preferred=[2]))
        physical_block, head_axis = K.cta_id()
        rank = cluster_rank % K.int32(2)
        cluster_block = physical_block // K.int32(2)
        head = K.local_scalar("int32", init=head_axis)
        logical_block = K.local_scalar("int32", init=cluster_block)
        plan_task = K.local_scalar("int32", init=cluster_block)
        if long_irregular_task_major_2d:
            # A 2D grid keeps each CTA pair adjacent in X while submitting
            # all four heads of one irregular tree task before the next task.
            K.assign(head, cluster_block)
            K.assign(logical_block, head_axis)
            K.assign(plan_task, head_axis)
        elif fixed_gqa_task_major:
            # Keep each cooperative CTA pair adjacent while ordering clusters
            # by K task, then by the eight GQA query heads. This matches the
            # source scheduler's adjacent-head K/V reuse for this exact shape.
            K.assign(head, cluster_block % K.int32(heads))
            K.assign(logical_block, cluster_block // K.int32(heads))
        batch_idx = K.local_scalar("int32", init=logical_block // K.int32(tasks))
        task = K.local_scalar("int32", init=logical_block % K.int32(tasks))
        if varlen:
            if use_source_varlen_nondet_schedule:
                cluster_begin = 0
                for sample, sample_tasks in enumerate(k_pairs_by_batch):
                    cluster_end = cluster_begin + sample_tasks * heads
                    with (
                        K.If(
                            (cluster_block >= K.int32(cluster_begin))
                            & (cluster_block < K.int32(cluster_end))
                        ),
                        K.Then(),
                    ):
                        sample_cluster = cluster_block - K.int32(cluster_begin)
                        K.assign(batch_idx, K.int32(sample))
                        K.assign(head, sample_cluster // K.int32(sample_tasks))
                        K.assign(task, sample_cluster % K.int32(sample_tasks))
                        K.assign(plan_task, K.int32(pair_offsets[sample]) + task)
                    cluster_begin = cluster_end
            else:
                K.assign(batch_idx, K.int32(0))
                for sample in range(1, batch):
                    with K.If(logical_block >= _load_i32(cu_k_blocks, K.int32(sample))), K.Then():
                        K.assign(batch_idx, K.int32(sample))
                K.assign(task, logical_block - _load_i32(cu_k_blocks, batch_idx))
            q_offset = _load_i32(cu_q, batch_idx)
            q_length = _load_i32(cu_q, batch_idx + K.int32(1)) - q_offset
            k_offset = _load_i32(cu_k, batch_idx)
            k_length = _load_i32(cu_k, batch_idx + K.int32(1)) - k_offset
            q_padded_offset = (q_offset + batch_idx * K.int32(128)) // K.int32(128) * K.int32(128)
            k_padded_offset = (k_offset + batch_idx * K.int32(256)) // K.int32(256) * K.int32(256)
            q_block_count = (q_length + K.int32(127)) // K.int32(128)
        else:
            q_length = K.int32(seqlen_q)
            k_length = K.int32(seqlen_kv)
            q_offset = K.int32(0)
            k_offset = K.int32(0)
            q_padded_offset = K.int32(0)
            k_padded_offset = K.int32(0)
            q_block_count = K.int32(q_blocks)
        map_batch = K.int32(0) if varlen else batch_idx
        kv_head = head // K.int32(qhead_per_kvhead)
        mask_head = head if mask_heads > 1 else K.int32(0)
        plan_row = (
            mask_head * _load_i32(cu_k_blocks, K.int32(batch)) + plan_task
            if varlen
            else mask_head * K.int32(batch * tasks) + batch_idx * K.int32(tasks) + task
        )
        if p00_causal_closed_plan:
            # For K-pair task t, causal K2Q has two boundary Q blocks
            # (2t, 2t+1) followed by the contiguous full suffix
            # (2t+2, ..., 1023).  Keep the source plan's packed-mask payload
            # offsets while avoiding count/offset/index loads in hot loops.
            partial_n = K.int32(2)
            partial_begin = task * K.int32(2)
            full_n = K.int32(1022) - task * K.int32(2)
            full_begin = task * (K.int32(1023) - task)
        elif p06_longformer_closed_plan:
            # The exact source Longformer K2Q plan has two full Q blocks per
            # task. Every 64th task starting at 32 is global and visits every
            # other Q block; its two neighbors deduplicate one global entry.
            task_mod64 = K.bitwise_and(task, K.int32(63))
            p06_global_task = task_mod64 == K.int32(32)
            p06_adjacent_task = (task_mod64 == K.int32(31)) | (task_mod64 == K.int32(33))
            partial_n = K.Select(
                p06_global_task,
                K.int32(1022),
                K.Select(
                    (task == K.int32(0)) | (task == K.int32(511)),
                    K.int32(10),
                    K.Select(p06_adjacent_task, K.int32(11), K.int32(12)),
                ),
            )
            partial_begin = (
                task * K.int32(12)
                - K.Select(task > K.int32(0), K.int32(2), K.int32(0))
                - K.shift_right(task + K.int32(32), K.int32(6))
                + K.shift_right(task + K.int32(31), K.int32(6)) * K.int32(1010)
                - K.shift_right(task + K.int32(30), K.int32(6))
            )
            full_n = K.int32(2)
            full_begin = task * K.int32(2)
        else:
            partial_n = _load_i32(partial_count, plan_row)
            full_n = _load_i32(full_count, plan_row)
            partial_begin = _load_i32(partial_offset, plan_row)
            full_begin = _load_i32(full_offset, plan_row)
        count = full_n + partial_n
        work = (count > K.int32(0)) & (task * K.int32(256) < k_length)

        def q_block_at(edge):
            value = K.local_scalar("int32")
            if p00_causal_closed_plan:
                K.assign(
                    value,
                    K.Select(
                        edge < full_n,
                        task * K.int32(2) + K.int32(2) + edge,
                        task * K.int32(2) + edge - full_n,
                    ),
                )
            elif p06_longformer_closed_plan:
                partial_edge = K.max(edge - K.int32(2), K.int32(0))
                global_q_block = K.Select(
                    partial_edge < task * K.int32(2), partial_edge, partial_edge + K.int32(2)
                )
                partial_q_block = K.Select(
                    p06_global_task,
                    global_q_block,
                    _load_i32(partial_index, partial_begin + partial_edge),
                )
                K.assign(
                    value, K.Select(edge < K.int32(2), task * K.int32(2) + edge, partial_q_block)
                )
            else:
                with K.If(edge < full_n):
                    with K.Then():
                        K.assign(value, _load_i32(full_index, full_begin + edge))
                    with K.Else():
                        K.assign(value, _load_i32(partial_index, partial_begin + edge - full_n))
            return value

        arena = K.alloc_buffer((SHARED_BYTES,), K.u8, scope="shared.dyn", align=1024)
        pool = K.smem_pool(base=arena).pool
        q_pipe = _PipelinePair(K.TMABar(pool, 1), K.TCGen05Bar(pool, 1))
        do_pipe = _PipelinePair(K.TMABar(pool, 1), K.TCGen05Bar(pool, 1))
        lse_pipe = _PipelinePair(K.TMABar(pool, 1), K.MBarrier(pool, 1))
        sum_pipe = _PipelinePair(K.TMABar(pool, 1), K.MBarrier(pool, 1))
        s_pipe = _PipelinePair(K.TCGen05Bar(pool, 1), K.MBarrier(pool, 1))
        dp_pipe = _PipelinePair(K.TCGen05Bar(pool, 1), K.MBarrier(pool, 1))
        ds_pipe = _PipelinePair(K.MBarrier(pool, 1), K.TCGen05Bar(pool, 1))
        dkdv_pipe = _PipelinePair(K.TCGen05Bar(pool, 2), K.MBarrier(pool, 2))
        dq_pipe = _PipelinePair(K.TCGen05Bar(pool, 1), K.MBarrier(pool, 1))
        tmem_mailbox = pool.alloc((1,), "uint32", align=4)
        tmem_dealloc_bar = K.MBarrier(pool, 1, leader=(K.thread_id() == K.int32(384)))
        if is_d192:
            qt_pipe = q_pipe
            # The source's barrier arena keeps the D128 Qt-pair slot even
            # though D192 aliases Qt to Q.  Preserve the hole so Kt starts at
            # byte 192 and the four cluster barriers remain at 208..232.
            pool.alloc((16,), "uint8", align=8)
        else:
            qt_pipe = _PipelinePair(K.TMABar(pool, 1), K.TCGen05Bar(pool, 1))
        kt_pipe = _PipelinePair(K.TMABar(pool, 1), K.TCGen05Bar(pool, 1))
        ds_empty_cluster = K.MBarrier(pool, 1)
        ds_full_cluster = K.MBarrier(pool, 1)
        ds_leader_cluster = K.MBarrier(pool, 1)
        dq_empty_cluster = K.MBarrier(pool, 1) if is_d192 else None
        assert pool.offset == (240 if is_d192 else 232)

        q_pipe.full.init(1)
        q_pipe.empty.init(1)
        do_pipe.full.init(1)
        do_pipe.empty.init(1)
        lse_pipe.full.init(1)
        lse_pipe.empty.init(8)
        sum_pipe.full.init(1)
        sum_pipe.empty.init(8)
        s_pipe.full.init(1)
        s_pipe.empty.init(16)
        dp_pipe.full.init(1)
        dp_pipe.empty.init(16)
        ds_pipe.full.init(16)
        ds_pipe.empty.init(1)
        dkdv_pipe.full.init(1)
        dkdv_pipe.empty.init(16)
        dq_pipe.full.init(1)
        dq_pipe.empty.init(8)
        if not is_d192:
            qt_pipe.full.init(1)
            qt_pipe.empty.init(1)
        kt_pipe.full.init(1)
        kt_pipe.empty.init(1)
        ds_empty_cluster.init(1)
        ds_full_cluster.init(1)
        ds_leader_cluster.init(2)
        if is_d192:
            dq_empty_cluster.init(4)
        tmem_dealloc_bar.init(32)
        K.ptx.fence.proxy.async_.shared__cta()
        K.ptx.fence.mbarrier_init.release.cluster()
        K.ptx.barrier.cluster.arrive.relaxed()
        K.ptx.barrier.cluster.wait()

        pair_mask = K.bitwise_or(K.int32(1), K.int32(2))
        q_full_leader = q_pipe.full.remote_view(0)
        do_full_leader = do_pipe.full.remote_view(0)
        qt_full_leader = qt_pipe.full.remote_view(0)
        kt_full_leader = kt_pipe.full.remote_view(0)
        smem_base = K.local_scalar("uint32", init=K.cuda.cvta_generic_to_shared(arena.ptr_to([0])))

        is_mha = qhead_per_kvhead == 1
        if is_mha and not deterministic:
            compute_regs, producer_regs, mma_regs = 144, 96, 96
        elif is_mha:
            compute_regs, producer_regs, mma_regs = 136, 96, 112
        else:
            compute_regs, producer_regs, mma_regs = 136, 112, 112
        sp = K.specialize(chain_dispatch=True)
        r_empty = sp.role("empty", warps=[15])
        r_relay = sp.role("relay", warps=[14])
        r_load = sp.role("load", warps=[13])
        r_mma = sp.role("mma", warps=[12])
        r_compute = sp.role("compute", warps=list(range(4, 12)))
        r_reduce = sp.role("reduce", warps=list(range(4)))

        with r_empty:
            K.ptx.setmaxnreg.dec.sync.aligned.u32(K.uint32(24))

        with r_relay:
            K.ptx.setmaxnreg.dec.sync.aligned.u32(K.uint32(producer_regs))
            relay_phase = K.local_scalar("int32", init=K.int32(0))
            edge = K.local_scalar("int32", init=K.int32(0))
            with K.While(edge < count):
                ds_full_cluster.wait(0, relay_phase)
                with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                    ds_leader_cluster.arrive(0, remote=0)
                K.assign(relay_phase, relay_phase ^ K.int32(1))
                K.assign(edge, edge + K.int32(1))

        with r_load:
            K.ptx.setmaxnreg.dec.sync.aligned.u32(K.uint32(producer_regs))
            elected = K.local_scalar("uint32", init=K.cuda.elect_sync())
            with K.If(elected != K.uint32(0)), K.Then():
                for tensor_map in (q_map, qt_map, k_map, kt_map, v_map, do_map, dot_map):
                    K.ptx.prefetch.tensormap(K.address_of(tensor_map))
                if direct_dkv and not direct_raw_dkv:
                    K.ptx.prefetch.tensormap(K.address_of(dv_map))
                    K.ptx.prefetch.tensormap(K.address_of(dk_map))

            q_prod = K.PipelineState(1, phase=1)
            do_prod = K.PipelineState(1, phase=1)
            qt_prod = K.PipelineState(1, phase=1)
            kt_prod = K.PipelineState(1, phase=1)
            lse_prod = K.PipelineState(1, phase=1)
            sum_prod = K.PipelineState(1, phase=1)
            previous_q = K.local_scalar("int32", init=K.int32(0))
            edge = K.local_scalar("int32", init=K.int32(0))
            bh = K.cast(batch_idx, "int64") * K.int64(heads) + K.cast(head, "int64")
            bh_q = (
                K.cast(head, "int64") * K.int64(q_storage_rows) + K.cast(q_padded_offset, "int64")
                if varlen
                else bh * K.int64(q128)
            )

            # D192 aliases Q/Qt and dO/dOt.  Each edge therefore has two
            # epochs on each one-stage barrier: Q then Qt, and dOt then dO.
            # K/V/Kt are resident and join only the first matching epoch.
            if is_d192:
                q192_prod = K.PipelineState(1, phase=1)
                do192_prod = K.PipelineState(1, phase=1)
                lse192_prod = K.PipelineState(1, phase=1)
                sum192_prod = K.PipelineState(1, phase=1)
                edge192 = K.local_scalar("int32", init=K.int32(0))
                with K.While(edge192 < count):
                    q_block = q_block_at(edge192)
                    q_safe = K.Select(q_block < q_block_count, q_block, q_block_count - K.int32(1))
                    first = edge192 == K.int32(0)

                    # The first edge brings in resident K together with Q.
                    with K.If(first), K.Then():
                        q_pipe.empty.wait(q192_prod.stage, q192_prod.phase)
                        with K.If(elected != K.uint32(0)), K.Then():
                            with K.If(rank == K.int32(0)), K.Then():
                                q_pipe.full.arrive(q192_prod.stage, K.uint32(147456))
                            for feature_chunk in range(3):
                                _issue_cluster_tma(
                                    tma_g2s,
                                    varlen,
                                    arena.ptr_to([off_sk + feature_chunk * 16384]),
                                    K.address_of(k_map),
                                    K.int32(feature_chunk * 64),
                                    k_offset + task * K.int32(256) + rank * K.int32(128),
                                    kv_head,
                                    map_batch,
                                    q_full_leader.ptr_to([q192_prod.stage]),
                                )
                                _issue_cluster_tma(
                                    tma_g2s,
                                    varlen,
                                    arena.ptr_to([off_sq + feature_chunk * 8192]),
                                    K.address_of(q_map),
                                    K.int32(feature_chunk * 64),
                                    q_offset + q_safe * K.int32(128) + rank * K.int32(64),
                                    head,
                                    map_batch,
                                    q_full_leader.ptr_to([q192_prod.stage]),
                                )
                        q192_prod.advance()

                    lse_pipe.empty.wait(lse192_prod.stage, lse192_prod.phase)
                    with K.If(elected != K.uint32(0)), K.Then():
                        lse_pipe.full.arrive(lse192_prod.stage, K.uint32(512))
                        K.ptx[BULK_G2S](
                            arena.ptr_to([off_slse]),
                            workspace.ptr_to(
                                [K.int64(sum_plane) + bh_q + K.cast(q_safe, "int64") * K.int64(128)]
                            ),
                            K.uint32(512),
                            lse_pipe.full.ptr_to([lse192_prod.stage]),
                        )
                    lse192_prod.advance()

                    # Later edges bring in Q after LSE, matching the source
                    # overlap.  The byte count is just the three Q chunks.
                    with K.If(first == K.bool(False)), K.Then():
                        q_pipe.empty.wait(q192_prod.stage, q192_prod.phase)
                        with K.If(elected != K.uint32(0)), K.Then():
                            with K.If(rank == K.int32(0)), K.Then():
                                q_pipe.full.arrive(q192_prod.stage, K.uint32(49152))
                            for feature_chunk in range(3):
                                _issue_cluster_tma(
                                    tma_g2s,
                                    varlen,
                                    arena.ptr_to([off_sq + feature_chunk * 8192]),
                                    K.address_of(q_map),
                                    K.int32(feature_chunk * 64),
                                    q_offset + q_safe * K.int32(128) + rank * K.int32(64),
                                    head,
                                    map_batch,
                                    q_full_leader.ptr_to([q192_prod.stage]),
                                )
                        q192_prod.advance()

                    # dOt shares sdO.  Resident V joins the first dOt epoch.
                    do_pipe.empty.wait(do192_prod.stage, do192_prod.phase)
                    with K.If(elected != K.uint32(0)), K.Then():
                        do_bytes = K.local_scalar("uint32", init=K.uint32(32768))
                        with K.If(first), K.Then():
                            K.assign(do_bytes, K.uint32(98304))
                        with K.If(rank == K.int32(0)), K.Then():
                            do_pipe.full.arrive(do192_prod.stage, do_bytes)
                        with K.If(first), K.Then():
                            for feature_chunk in range(2):
                                _issue_cluster_tma(
                                    tma_g2s,
                                    varlen,
                                    arena.ptr_to([off_sv + feature_chunk * 16384]),
                                    K.address_of(v_map),
                                    K.int32(feature_chunk * 64),
                                    k_offset + task * K.int32(256) + rank * K.int32(128),
                                    kv_head,
                                    map_batch,
                                    do_full_leader.ptr_to([do192_prod.stage]),
                                )
                        for feature_chunk in range(2):
                            _issue_cluster_tma(
                                tma_g2s,
                                varlen,
                                arena.ptr_to([off_sdot + feature_chunk * 8192]),
                                K.address_of(dot_map),
                                K.int32(feature_chunk * 64),
                                q_offset + q_safe * K.int32(128) + rank * K.int32(64),
                                head,
                                map_batch,
                                do_full_leader.ptr_to([do192_prod.stage]),
                            )
                    do192_prod.advance()

                    sum_pipe.empty.wait(sum192_prod.stage, sum192_prod.phase)
                    with K.If(elected != K.uint32(0)), K.Then():
                        sum_pipe.full.arrive(sum192_prod.stage, K.uint32(512))
                        K.ptx[BULK_G2S](
                            arena.ptr_to([off_ssum]),
                            workspace.ptr_to([bh_q + K.cast(q_safe, "int64") * K.int64(128)]),
                            K.uint32(512),
                            sum_pipe.full.ptr_to([sum192_prod.stage]),
                        )
                    sum192_prod.advance()

                    # Qt overwrites Q after score consumption.  Kt is loaded
                    # beside it only once and remains resident for all dQ.
                    q_pipe.empty.wait(q192_prod.stage, q192_prod.phase)
                    with K.If(elected != K.uint32(0)), K.Then():
                        with K.If(rank == K.int32(0)), K.Then():
                            q_bytes = K.local_scalar("uint32", init=K.uint32(49152))
                            with K.If(first), K.Then():
                                K.assign(q_bytes, K.uint32(147456))
                            q_pipe.full.arrive(q192_prod.stage, q_bytes)
                        for feature_chunk in range(3):
                            _issue_cluster_tma(
                                tma_g2s,
                                varlen,
                                arena.ptr_to([off_sqt + feature_chunk * 8192]),
                                K.address_of(qt_map),
                                rank * K.int32(96) + K.int32(feature_chunk * 32),
                                q_offset + q_safe * K.int32(128),
                                head,
                                map_batch,
                                q_full_leader.ptr_to([q192_prod.stage]),
                            )
                        with K.If(first), K.Then():
                            for feature_chunk in range(3):
                                _issue_cluster_tma(
                                    tma_g2s,
                                    varlen,
                                    arena.ptr_to([off_skt + feature_chunk * 16384]),
                                    K.address_of(kt_map),
                                    rank * K.int32(96) + K.int32(feature_chunk * 32),
                                    k_offset + task * K.int32(256),
                                    kv_head,
                                    map_batch,
                                    q_full_leader.ptr_to([q192_prod.stage]),
                                )
                    q192_prod.advance()

                    # dO overwrites dOt only after the dP consumer releases it.
                    do_pipe.empty.wait(do192_prod.stage, do192_prod.phase)
                    with K.If(elected != K.uint32(0)), K.Then():
                        with K.If(rank == K.int32(0)), K.Then():
                            do_pipe.full.arrive(do192_prod.stage, K.uint32(32768))
                        _issue_cluster_tma(
                            tma_g2s,
                            varlen,
                            arena.ptr_to([off_sdo]),
                            K.address_of(do_map),
                            rank * K.int32(64),
                            q_offset + q_safe * K.int32(128),
                            head,
                            map_batch,
                            do_full_leader.ptr_to([do192_prod.stage]),
                        )
                    do192_prod.advance()
                    K.assign(edge192, edge192 + K.int32(1))

            d128_count = count if not is_d192 else K.int32(0)
            with K.While(edge < d128_count):
                q_block = q_block_at(edge)
                q_safe = K.Select(q_block < q_block_count, q_block, q_block_count - K.int32(1))
                first = edge == K.int32(0)

                with K.If(edge > K.int32(0)), K.Then():
                    qt_pipe.empty.wait(qt_prod.stage, qt_prod.phase)
                    with K.If(elected != K.uint32(0)), K.Then():
                        with K.If(rank == K.int32(0)), K.Then():
                            qt_pipe.full.arrive(qt_prod.stage, K.uint32(32768))
                        _issue_cluster_tma(
                            tma_g2s,
                            varlen,
                            arena.ptr_to([off_sqt]),
                            K.address_of(qt_map),
                            rank * K.int32(64),
                            q_offset + previous_q * K.int32(128),
                            head,
                            map_batch,
                            qt_full_leader.ptr_to([qt_prod.stage]),
                        )
                    qt_prod.advance()

                q_pipe.empty.wait(q_prod.stage, q_prod.phase)
                with K.If(elected != K.uint32(0)), K.Then():
                    q_bytes = K.local_scalar("uint32", init=K.uint32(32768))
                    with K.If(first), K.Then():
                        K.assign(q_bytes, K.uint32(98304))
                    with K.If(rank == K.int32(0)), K.Then():
                        q_pipe.full.arrive(q_prod.stage, q_bytes)
                    with K.If(first), K.Then():
                        for feature_half in range(2):
                            _issue_cluster_tma(
                                tma_g2s,
                                varlen,
                                arena.ptr_to([off_sk + feature_half * 16384]),
                                K.address_of(k_map),
                                K.int32(feature_half * 64),
                                k_offset + task * K.int32(256) + rank * K.int32(128),
                                kv_head,
                                map_batch,
                                q_full_leader.ptr_to([q_prod.stage]),
                            )
                    for feature_half in range(2):
                        _issue_cluster_tma(
                            tma_g2s,
                            varlen,
                            arena.ptr_to([off_sq + feature_half * 8192]),
                            K.address_of(q_map),
                            K.int32(feature_half * 64),
                            q_offset + q_safe * K.int32(128) + rank * K.int32(64),
                            head,
                            map_batch,
                            q_full_leader.ptr_to([q_prod.stage]),
                        )
                q_prod.advance()

                lse_pipe.empty.wait(lse_prod.stage, lse_prod.phase)
                with K.If(elected != K.uint32(0)), K.Then():
                    lse_pipe.full.arrive(lse_prod.stage, K.uint32(512))
                    K.ptx[BULK_G2S](
                        arena.ptr_to([off_slse]),
                        workspace.ptr_to(
                            [K.int64(sum_plane) + bh_q + K.cast(q_safe, "int64") * K.int64(128)]
                        ),
                        K.uint32(512),
                        lse_pipe.full.ptr_to([lse_prod.stage]),
                    )
                lse_prod.advance()

                do_pipe.empty.wait(do_prod.stage, do_prod.phase)
                with K.If(elected != K.uint32(0)), K.Then():
                    do_bytes = K.local_scalar("uint32", init=K.uint32(65536))
                    with K.If(first), K.Then():
                        K.assign(do_bytes, K.uint32(131072))
                    with K.If(rank == K.int32(0)), K.Then():
                        do_pipe.full.arrive(do_prod.stage, do_bytes)
                    with K.If(first), K.Then():
                        for feature_half in range(2):
                            _issue_cluster_tma(
                                tma_g2s,
                                varlen,
                                arena.ptr_to([off_sv + feature_half * 16384]),
                                K.address_of(v_map),
                                K.int32(feature_half * 64),
                                k_offset + task * K.int32(256) + rank * K.int32(128),
                                kv_head,
                                map_batch,
                                do_full_leader.ptr_to([do_prod.stage]),
                            )
                    _issue_cluster_tma(
                        tma_g2s,
                        varlen,
                        arena.ptr_to([off_sdo]),
                        K.address_of(do_map),
                        rank * K.int32(64),
                        q_offset + q_safe * K.int32(128),
                        head,
                        map_batch,
                        do_full_leader.ptr_to([do_prod.stage]),
                    )
                    for feature_half in range(2):
                        _issue_cluster_tma(
                            tma_g2s,
                            varlen,
                            arena.ptr_to([off_sdot + feature_half * 8192]),
                            K.address_of(dot_map),
                            K.int32(feature_half * 64),
                            q_offset + q_safe * K.int32(128) + rank * K.int32(64),
                            head,
                            map_batch,
                            do_full_leader.ptr_to([do_prod.stage]),
                        )
                do_prod.advance()

                sum_pipe.empty.wait(sum_prod.stage, sum_prod.phase)
                with K.If(elected != K.uint32(0)), K.Then():
                    sum_pipe.full.arrive(sum_prod.stage, K.uint32(512))
                    K.ptx[BULK_G2S](
                        arena.ptr_to([off_ssum]),
                        workspace.ptr_to([bh_q + K.cast(q_safe, "int64") * K.int64(128)]),
                        K.uint32(512),
                        sum_pipe.full.ptr_to([sum_prod.stage]),
                    )
                sum_prod.advance()

                with K.If(first), K.Then():
                    kt_pipe.empty.wait(kt_prod.stage, kt_prod.phase)
                    with K.If(elected != K.uint32(0)), K.Then():
                        with K.If(rank == K.int32(0)), K.Then():
                            kt_pipe.full.arrive(kt_prod.stage, K.uint32(65536))
                        _issue_cluster_tma(
                            tma_g2s,
                            varlen,
                            arena.ptr_to([off_skt]),
                            K.address_of(kt_map),
                            rank * K.int32(64),
                            k_offset + task * K.int32(256),
                            kv_head,
                            map_batch,
                            kt_full_leader.ptr_to([kt_prod.stage]),
                        )
                    kt_prod.advance()
                K.assign(previous_q, q_safe)
                K.assign(edge, edge + K.int32(1))

            if not is_d192:
                with K.If(work), K.Then():
                    qt_pipe.empty.wait(qt_prod.stage, qt_prod.phase)
                    with K.If(elected != K.uint32(0)), K.Then():
                        with K.If(rank == K.int32(0)), K.Then():
                            qt_pipe.full.arrive(qt_prod.stage, K.uint32(32768))
                        _issue_cluster_tma(
                            tma_g2s,
                            varlen,
                            arena.ptr_to([off_sqt]),
                            K.address_of(qt_map),
                            rank * K.int32(64),
                            q_offset + previous_q * K.int32(128),
                            head,
                            map_batch,
                            qt_full_leader.ptr_to([qt_prod.stage]),
                        )

        with r_mma:
            K.ptx.setmaxnreg.dec.sync.aligned.u32(K.uint32(mma_regs))
            K.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                K.address_of(tmem_mailbox), K.uint32(TMEM_COLUMNS)
            )
            K.ptx.barrier.sync(K.uint32(BAR_TMEM[0]), K.uint32(BAR_TMEM[1]))
            tcol = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(tcol, tmem_mailbox.ptr_to([0]))
            elected = K.local_scalar("uint32", init=K.cuda.elect_sync())

            if is_d192:
                with K.If((rank == K.int32(0)) & work), K.Then():
                    d_k = _desc_at(K.uint64(desc_krow_base), smem_base + K.uint32(off_sk))
                    d_v = _desc_at(K.uint64(desc_krow_base), smem_base + K.uint32(off_sv))
                    d_q = _desc_at(K.uint64(desc_row_base), smem_base + K.uint32(off_sq))
                    d_dot = _desc_at(K.uint64(desc_row_base), smem_base + K.uint32(off_sdot))
                    d_do = _desc_at(K.uint64(desc_mn_base), smem_base + K.uint32(off_sdo))
                    d192_high = K.shift_left(K.uint64(0x80004020), K.uint64(32))
                    d_qt = _desc_at(
                        K.bitwise_or(d192_high, K.uint64(0x02000000)), smem_base + K.uint32(off_sqt)
                    )
                    d_kt = _desc_at(
                        K.bitwise_or(d192_high, K.uint64(0x04000000)), smem_base + K.uint32(off_skt)
                    )
                    d_ds = _desc_at(K.uint64(desc_mn_base), smem_base + K.uint32(off_sds))
                    ts = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(t_s))
                    tp = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(t_p))
                    tdq = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(t_dq))
                    tv = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(t_v))
                    tdp = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(t_dp))
                    tds = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(t_ds))
                    tdk = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(t_dk))

                    q_cons = K.PipelineState(1, phase=0)
                    do_cons = K.PipelineState(1, phase=0)
                    s_prod = K.PipelineState(1, phase=0)
                    dp_prod = K.PipelineState(1, phase=0)
                    ds_cons = K.PipelineState(1, phase=0)
                    dq_prod = K.PipelineState(1, phase=0)
                    dkv_prod = K.PipelineState(2, phase=0)
                    dk_accumulate = K.local_scalar("uint32", init=K.uint32(0))
                    dv_accumulate = K.local_scalar("uint32", init=K.uint32(0))
                    edge = K.local_scalar("int32", init=K.int32(0))

                    with K.While(edge < count):
                        q_pipe.full.wait(q_cons.stage, q_cons.phase)
                        dq_pipe.empty.wait(dq_prod.stage, dq_prod.phase ^ 1)
                        with K.If(elected != K.uint32(0)), K.Then():
                            _mma2_ss(ts, d_k, d_q, id_ss, krow_offsets, row_offsets, K.uint32(0))
                            _commit2(s_pipe.full.ptr_to([s_prod.stage]), pair_mask)
                            _commit2(q_pipe.empty.ptr_to([q_cons.stage]), pair_mask)
                        q_cons.advance()
                        s_prod.advance()

                        do_pipe.full.wait(do_cons.stage, do_cons.phase)
                        # S/P share TMEM columns 320..383.  The compute role
                        # releases S only after its score loads, so dP must
                        # acquire that toggled empty phase before overwrite.
                        s_pipe.empty.wait(s_prod.stage, s_prod.phase ^ 1)
                        with K.If(elected != K.uint32(0)), K.Then():
                            _mma2_ss(
                                tdp, d_v, d_dot, id_ss, dv_krow_offsets, do_row_offsets, K.uint32(0)
                            )
                            _commit2(dp_pipe.full.ptr_to([dp_prod.stage]), pair_mask)
                            _commit2(do_pipe.empty.ptr_to([do_cons.stage]), pair_mask)
                        do_cons.advance()
                        dp_prod.advance()

                        q_pipe.full.wait(q_cons.stage, q_cons.phase)
                        dp_pipe.empty.wait(dp_prod.stage, dp_prod.phase ^ 1)
                        with K.If(elected != K.uint32(0)), K.Then():
                            _mma2_ts(tdk, tds, d_qt, id_dk, dk_offsets, dk_accumulate)
                            _commit2(q_pipe.empty.ptr_to([q_cons.stage]), pair_mask)
                        K.assign(dk_accumulate, K.uint32(1))
                        q_cons.advance()

                        do_pipe.full.wait(do_cons.stage, do_cons.phase)
                        with K.If(elected != K.uint32(0)), K.Then():
                            _mma2_ts(tv, tp, d_do, id_ts, mn8_offsets, dv_accumulate)
                            _commit2(do_pipe.empty.ptr_to([do_cons.stage]), pair_mask)
                        K.assign(dv_accumulate, K.uint32(1))
                        do_cons.advance()

                        ds_pipe.full.wait(ds_cons.stage, ds_cons.phase)
                        ds_leader_cluster.wait(0, ds_cons.phase)
                        with K.If(elected != K.uint32(0)), K.Then():
                            _mma2_dq(tdq, d_ds, d_kt, id_dq, dq_a_offsets, dq_b_offsets)
                            _commit2(dq_pipe.full.ptr_to([dq_prod.stage]), pair_mask)
                            _commit2(ds_pipe.empty.ptr_to([ds_cons.stage]), pair_mask)
                        ds_cons.advance()
                        dq_prod.advance()
                        K.assign(edge, edge + K.int32(1))

                    dkdv_pipe.empty.wait(dkv_prod.stage, dkv_prod.phase ^ 1)
                    with K.If(elected != K.uint32(0)), K.Then():
                        _commit2(dkdv_pipe.full.ptr_to([dkv_prod.stage]), pair_mask)
                    dkv_prod.advance()
                    dkdv_pipe.empty.wait(dkv_prod.stage, dkv_prod.phase ^ 1)
                    with K.If(elected != K.uint32(0)), K.Then():
                        _commit2(dkdv_pipe.full.ptr_to([dkv_prod.stage]), pair_mask)
            else:
                with K.If((rank == K.int32(0)) & work), K.Then():
                    d_k = _desc_at(K.uint64(desc_krow_base), smem_base + K.uint32(off_sk))
                    d_v = _desc_at(K.uint64(desc_krow_base), smem_base + K.uint32(off_sv))
                    d_q = _desc_at(K.uint64(desc_row_base), smem_base + K.uint32(off_sq))
                    d_dot = _desc_at(K.uint64(desc_row_base), smem_base + K.uint32(off_sdot))
                    d_do = _desc_at(K.uint64(desc_mn_base), smem_base + K.uint32(off_sdo))
                    d_qt = _desc_at(K.uint64(desc_mn_base), smem_base + K.uint32(off_sqt))
                    d_kt = _desc_at(K.uint64(desc_mn_base), smem_base + K.uint32(off_skt))
                    d_ds = _desc_at(K.uint64(desc_mn_base), smem_base + K.uint32(off_sds))
                    ts = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(t_s))
                    tdq = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(t_dq))
                    tv = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(t_v))
                    tdp = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(t_dp))
                    tds = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(t_ds))
                    tdk = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(t_dk))

                    q_cons = K.PipelineState(1, phase=0)
                    do_cons = K.PipelineState(1, phase=0)
                    qt_cons = K.PipelineState(1, phase=0)
                    kt_cons = K.PipelineState(1, phase=0)
                    s_prod = K.PipelineState(1, phase=0)
                    dp_prod = K.PipelineState(1, phase=0)
                    ds_cons = K.PipelineState(1, phase=0)
                    dq_prod = K.PipelineState(1, phase=0)
                    dkv_prod = K.PipelineState(2, phase=0)

                    q_pipe.full.wait(q_cons.stage, q_cons.phase)
                    s_pipe.empty.wait(s_prod.stage, s_prod.phase ^ 1)
                    with K.If(elected != K.uint32(0)), K.Then():
                        _mma2_ss(ts, d_k, d_q, id_ss, krow_offsets, row_offsets, K.uint32(0))
                        _commit2(s_pipe.full.ptr_to([s_prod.stage]), pair_mask)
                        _commit2(q_pipe.empty.ptr_to([q_cons.stage]), pair_mask)
                    q_cons.advance()
                    s_prod.advance()

                    do_pipe.full.wait(do_cons.stage, do_cons.phase)
                    dp_pipe.empty.wait(dp_prod.stage, dp_prod.phase ^ 1)
                    with K.If(elected != K.uint32(0)), K.Then():
                        _mma2_ss(
                            tdp, d_v, d_dot, id_ss, dv_krow_offsets, do_row_offsets, K.uint32(0)
                        )
                        _commit2(dp_pipe.full.ptr_to([dp_prod.stage]), pair_mask)
                    dp_prod.advance()

                    s_pipe.empty.wait(s_prod.stage, s_prod.phase ^ 1)
                    with K.If(elected != K.uint32(0)), K.Then():
                        _mma2_ts(tv, ts, d_do, id_ts, mn8_offsets, K.uint32(0))
                        _commit2(do_pipe.empty.ptr_to([do_cons.stage]), pair_mask)
                    do_cons.advance()
                    kt_pipe.full.wait(kt_cons.stage, kt_cons.phase)

                    edge = K.local_scalar("int32", init=K.int32(1))
                    dk_accumulate = K.local_scalar("uint32", init=K.uint32(0))
                    with K.While(edge < count):
                        q_pipe.full.wait(q_cons.stage, q_cons.phase)
                        dq_pipe.empty.wait(dq_prod.stage, dq_prod.phase ^ 1)
                        with K.If(elected != K.uint32(0)), K.Then():
                            _mma2_ss(ts, d_k, d_q, id_ss, krow_offsets, row_offsets, K.uint32(0))
                            _commit2(s_pipe.full.ptr_to([s_prod.stage]), pair_mask)
                            _commit2(q_pipe.empty.ptr_to([q_cons.stage]), pair_mask)
                        q_cons.advance()
                        s_prod.advance()

                        qt_pipe.full.wait(qt_cons.stage, qt_cons.phase)
                        dp_pipe.empty.wait(dp_prod.stage, dp_prod.phase ^ 1)
                        ds_pipe.full.wait(ds_cons.stage, ds_cons.phase)
                        with K.If(elected != K.uint32(0)), K.Then():
                            _mma2_ts(tdk, tds, d_qt, id_dk, mn8_offsets, dk_accumulate)
                            _commit2(qt_pipe.empty.ptr_to([qt_cons.stage]), pair_mask)
                        K.assign(dk_accumulate, K.uint32(1))
                        qt_cons.advance()

                        do_pipe.full.wait(do_cons.stage, do_cons.phase)
                        with K.If(elected != K.uint32(0)), K.Then():
                            _mma2_ss(
                                tdp, d_v, d_dot, id_ss, dv_krow_offsets, do_row_offsets, K.uint32(0)
                            )
                            _commit2(dp_pipe.full.ptr_to([dp_prod.stage]), pair_mask)
                        dp_prod.advance()

                        ds_leader_cluster.wait(0, ds_cons.phase)
                        with K.If(elected != K.uint32(0)), K.Then():
                            _mma2_dq(tdq, d_ds, d_kt, id_dq, dq_a_offsets, dq_b_offsets)
                            _commit2(dq_pipe.full.ptr_to([dq_prod.stage]), pair_mask)
                            _commit2(ds_pipe.empty.ptr_to([ds_cons.stage]), pair_mask)
                        ds_cons.advance()
                        dq_prod.advance()

                        s_pipe.empty.wait(s_prod.stage, s_prod.phase ^ 1)
                        with K.If(elected != K.uint32(0)), K.Then():
                            _mma2_ts(tv, ts, d_do, id_ts, mn8_offsets, K.uint32(1))
                            _commit2(do_pipe.empty.ptr_to([do_cons.stage]), pair_mask)
                        do_cons.advance()
                        K.assign(edge, edge + K.int32(1))

                    with K.If(elected != K.uint32(0)), K.Then():
                        _commit2(s_pipe.full.ptr_to([s_prod.stage]), pair_mask)
                    dkdv_pipe.empty.wait(dkv_prod.stage, dkv_prod.phase ^ 1)
                    with K.If(elected != K.uint32(0)), K.Then():
                        _commit2(dkdv_pipe.full.ptr_to([dkv_prod.stage]), pair_mask)
                    dkv_prod.advance()

                    dkdv_pipe.empty.wait(dkv_prod.stage, dkv_prod.phase ^ 1)
                    qt_pipe.full.wait(qt_cons.stage, qt_cons.phase)
                    dp_pipe.empty.wait(dp_prod.stage, dp_prod.phase ^ 1)
                    ds_pipe.full.wait(ds_cons.stage, ds_cons.phase)
                    with K.If(elected != K.uint32(0)), K.Then():
                        _mma2_ts(tdk, tds, d_qt, id_dk, mn8_offsets, dk_accumulate)
                        _commit2(qt_pipe.empty.ptr_to([qt_cons.stage]), pair_mask)
                        _commit2(dkdv_pipe.full.ptr_to([dkv_prod.stage]), pair_mask)
                    qt_cons.advance()
                    dkv_prod.advance()

                    ds_leader_cluster.wait(0, ds_cons.phase)
                    dq_pipe.empty.wait(dq_prod.stage, dq_prod.phase ^ 1)
                    with K.If(elected != K.uint32(0)), K.Then():
                        _mma2_dq(tdq, d_ds, d_kt, id_dq, dq_a_offsets, dq_b_offsets)
                        _commit2(dq_pipe.full.ptr_to([dq_prod.stage]), pair_mask)
                        _commit2(ds_pipe.empty.ptr_to([ds_cons.stage]), pair_mask)
                        _commit2(kt_pipe.empty.ptr_to([kt_cons.stage]), pair_mask)

            K.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
            K.ptx.barrier.sync(K.uint32(BAR_TMEM[0]), K.uint32(BAR_TMEM[1]))
            tmem_dealloc_bar.arrive(0, remote=K.int32(1) - rank, pred=True)
            tmem_dealloc_bar.wait(0, 0)
            K.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](tcol, K.uint32(TMEM_COLUMNS))

        with r_compute:
            K.ptx.setmaxnreg.inc.sync.aligned.u32(K.uint32(compute_regs))
            K.ptx.barrier.sync(K.uint32(BAR_TMEM[0]), K.uint32(BAR_TMEM[1]))
            tcol = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(tcol, tmem_mailbox.ptr_to([0]))
            ctid = K.thread_id() % K.int32(256)
            crow = ctid % K.int32(128)
            wg = ctid // K.int32(128)
            row_group = (crow // K.int32(32)) * K.int32(32)
            warp_in_compute = K.warp_id_in_role()
            warp_in_wg = warp_in_compute % K.int32(4)
            scale_log2 = K.local_scalar("float32")
            K.ptx.mul.f32(scale_log2, softmax_scale, K.float32(1.4426950408889634))
            s_cons = K.PipelineState(1, phase=0)
            lse_cons = K.PipelineState(1, phase=0)
            sum_cons = K.PipelineState(1, phase=0)
            dp_cons = K.PipelineState(1, phase=0)
            ds_prod = K.PipelineState(1, phase=1)
            dkv_cons = K.PipelineState(2, phase=0)

            def compute_group(group_count, group_offset, is_partial):
                group_edge = K.local_scalar("int32", init=K.int32(0))
                with K.While(group_edge < group_count):
                    edge = group_edge + group_offset
                    lse_pipe.full.wait(lse_cons.stage, lse_cons.phase)
                    s_pipe.full.wait(s_cons.stage, s_cons.phase)
                    scores = K.alloc_local((64,), "float32")
                    for rep in range(2):
                        _tmem_load32(
                            scores,
                            rep * 32,
                            K.cuda.get_tmem_addr(
                                tcol, row_group, K.int32(t_s + wg * 32 + rep * 64)
                            ),
                        )
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    if is_d192:
                        # P aliases S in D192, so release the S accumulator as
                        # soon as the TMEM reads have completed, before R2T P.
                        with K.If(K.lane_id() == K.int32(0)), K.Then():
                            s_pipe.empty.arrive(s_cons.stage, remote=0, pred=True)
                        s_cons.advance()
                    else:
                        # D128 publishes dS one iteration late because S and the
                        # exchanged dS lifetime overlap in its TMEM schedule.
                        with K.If(edge > K.int32(0)), K.Then():
                            with K.If(K.lane_id() == K.int32(0)), K.Then():
                                ds_pipe.full.arrive(ds_prod.stage, remote=0)
                            ds_prod.advance()

                    # Partial payloads already include the sequence-tail mask, and
                    # full K2Q entries are complete 256-key tiles by construction.
                    if is_partial:
                        mask_words = K.alloc_local((2,), "uint32")
                        payload = partial_begin + group_edge
                        mask_word = (
                            K.cast(payload, "int64") * K.int64(2) + K.cast(rank, "int64")
                        ) * K.int64(512) + K.cast(ctid, "int64") * K.int64(2)
                        K.ptx.ld.global_.v2.b32(
                            mask_words[0], mask_words[1], packed_mask.ptr_to([mask_word])
                        )
                        for j in range(64):
                            keep = K.bitwise_and(
                                K.shift_right(mask_words[j // 32], K.uint32(j % 32)), K.uint32(1)
                            )
                            score_bits = K.reinterpret(K.u32, scores[j])
                            drop = K.local_scalar("uint32")
                            K.ptx.setp.eq.u32(drop, keep, K.uint32(0))
                            masked_bits = K.local_scalar("uint32")
                            K.ptx.selp.b32(
                                masked_bits, K.uint32(0xFF800000), score_bits, K.ptx.pred(drop)
                            )
                            K.assign(scores[j], K.reinterpret(K.f32, masked_bits))
                    for rep in range(2):
                        for pair in range(16):
                            j = rep * 32 + pair * 2
                            qcol = wg * K.int32(32) + K.int32(rep * 64 + pair * 2)
                            lse0 = K.local_scalar("float32")
                            lse1 = K.local_scalar("float32")
                            K.ptx.ld.shared_.v2.b32(
                                lse0, lse1, arena.ptr_to([off_slse + qcol * K.int32(4)])
                            )
                            neg0 = K.local_scalar("float32")
                            neg1 = K.local_scalar("float32")
                            K.ptx.neg.f32(neg0, lse0)
                            K.ptx.neg.f32(neg1, lse1)
                            lo, hi = _packed_fma(
                                scores[j], scores[j + 1], scale_log2, scale_log2, neg0, neg1
                            )
                            K.ptx.ex2.approx.ftz.f32(scores[j], lo)
                            K.ptx.ex2.approx.ftz.f32(scores[j + 1], hi)
                    packed_p = K.alloc_local((32,), "uint32")
                    for pair in range(32):
                        K.ptx[cvt_pack](packed_p[pair], scores[pair * 2 + 1], scores[pair * 2])
                    K.ptx.bar.sync(K.uint32(BAR_COMPUTE[0]), K.uint32(BAR_COMPUTE[1]))
                    for rep in range(2):
                        _tmem_store16(
                            packed_p,
                            rep * 16,
                            K.cuda.get_tmem_addr(
                                tcol, row_group, K.int32(t_s + wg * 16 + rep * 32)
                            ),
                        )
                    K.ptx["tcgen05.wait::st.sync.aligned"]()
                    K.ptx.fence.proxy.async_.shared__cta()
                    K.ptx.bar.sync(K.uint32(BAR_COMPUTE[0]), K.uint32(BAR_COMPUTE[1]))
                    with K.If(K.lane_id() == K.int32(0)), K.Then():
                        lse_pipe.empty.arrive(lse_cons.stage)
                    if not is_d192:
                        with K.If(K.lane_id() == K.int32(0)), K.Then():
                            s_pipe.empty.arrive(s_cons.stage, remote=0, pred=True)
                        s_cons.advance()
                    lse_cons.advance()

                    sum_pipe.full.wait(sum_cons.stage, sum_cons.phase)
                    dp_pipe.full.wait(dp_cons.stage, dp_cons.phase)
                    exchange = K.alloc_local((16,), "uint32")
                    for rep in range(2):
                        dp_values = K.alloc_local((32,), "float32")
                        _tmem_load32(
                            dp_values,
                            0,
                            K.cuda.get_tmem_addr(
                                tcol, row_group, K.int32(t_dp + wg * 32 + rep * 64)
                            ),
                        )
                        K.ptx["tcgen05.wait::ld.sync.aligned"]()
                        K.ptx.bar.sync(K.uint32(BAR_COMPUTE[0]), K.uint32(BAR_COMPUTE[1]))
                        for pair in range(16):
                            j = pair * 2
                            qcol = wg * K.int32(32) + K.int32(rep * 64 + pair * 2)
                            sum0 = K.local_scalar("float32")
                            sum1 = K.local_scalar("float32")
                            K.ptx.ld.shared_.v2.b32(
                                sum0, sum1, arena.ptr_to([off_ssum + qcol * K.int32(4)])
                            )
                            lo, hi = _packed_binary(
                                "sub.rn.f32x2", dp_values[j], dp_values[j + 1], sum0, sum1
                            )
                            lo, hi = _packed_binary(
                                "mul.rn.f32x2",
                                lo,
                                hi,
                                scores[rep * 32 + j],
                                scores[rep * 32 + j + 1],
                            )
                            K.assign(dp_values[j], lo)
                            K.assign(dp_values[j + 1], hi)
                        packed_ds = K.alloc_local((16,), "uint32")
                        for pair in range(16):
                            K.ptx[cvt_pack](
                                packed_ds[pair], dp_values[pair * 2 + 1], dp_values[pair * 2]
                            )
                        if rep == 0:
                            ds_pipe.empty.wait(0, ds_prod.phase)
                        _tmem_store16(
                            packed_ds,
                            0,
                            K.cuda.get_tmem_addr(
                                tcol, row_group, K.int32(t_ds + wg * 16 + rep * 32)
                            ),
                        )
                        # CTA 0 keeps score stage 0 and exchanges stage 1; CTA 1
                        # keeps stage 1 and exchanges stage 0.  Both compute
                        # warpgroups contribute their 32-column quarter to the
                        # selected 64-column stage.
                        direct_half = K.int32(rep) == rank
                        with K.If(direct_half):
                            with K.Then():
                                for group in range(4):
                                    qcol = wg * K.int32(32) + K.int32(rep * 64 + group * 8)
                                    base = group * 4
                                    K.ptx.st.shared.v4.b32(
                                        arena.ptr_to([_tile_byte(off_sds, crow, qcol)]),
                                        packed_ds[base],
                                        packed_ds[base + 1],
                                        packed_ds[base + 2],
                                        packed_ds[base + 3],
                                    )
                            with K.Else():
                                if is_d192:
                                    for value_index in range(16):
                                        K.assign(exchange[value_index], packed_ds[value_index])
                                else:
                                    for group in range(4):
                                        base = group * 4
                                        K.ptx.st.shared.v4.b32(
                                            arena.ptr_to(
                                                [
                                                    _tile_byte(
                                                        off_sdsx,
                                                        crow,
                                                        wg * K.int32(32) + K.int32(group * 8),
                                                    )
                                                ]
                                            ),
                                            packed_ds[base],
                                            packed_ds[base + 1],
                                            packed_ds[base + 2],
                                            packed_ds[base + 3],
                                        )
                    K.ptx["tcgen05.wait::st.sync.aligned"]()
                    with K.If(K.lane_id() == K.int32(0)), K.Then():
                        dp_pipe.empty.arrive(dp_cons.stage, remote=0, pred=True)
                    dp_cons.advance()
                    if is_d192:
                        # sdSx aliases the dQ reduction arena.  The four reducer
                        # warps release this epoch only after all async reductions
                        # using the prior dQ slice have drained.
                        dq_empty_cluster.wait(0, ds_prod.phase)
                        for group in range(4):
                            base = group * 4
                            K.ptx.st.shared.v4.b32(
                                arena.ptr_to(
                                    [
                                        _tile_byte(
                                            off_sdsx, crow, wg * K.int32(32) + K.int32(group * 8)
                                        )
                                    ]
                                ),
                                exchange[base],
                                exchange[base + 1],
                                exchange[base + 2],
                                exchange[base + 3],
                            )
                    K.ptx.fence.proxy.async_.shared__cta()
                    K.ptx.bar.sync(K.uint32(BAR_COMPUTE[0]), K.uint32(BAR_COMPUTE[1]))
                    with K.If(K.lane_id() == K.int32(0)), K.Then():
                        sum_pipe.empty.arrive(sum_cons.stage)
                    sum_cons.advance()
                    if is_d192:
                        with K.If(K.lane_id() == K.int32(0)), K.Then():
                            ds_pipe.full.arrive(ds_prod.stage, remote=0)
                        ds_prod.advance()
                        exchange_owner = ctid == K.int32(0)
                    else:
                        exchange_owner = (
                            (wg != rank) & (warp_in_wg == K.int32(0)) & (K.lane_id() == K.int32(0))
                        )
                    with K.If(exchange_owner), K.Then():
                        peer = K.int32(1) - rank
                        remote_bar = K.local_scalar("uint32")
                        remote_dst = K.local_scalar("uint32")
                        K.ptx.mapa.shared__cluster.u32(
                            remote_bar,
                            K.cuda.cvta_generic_to_shared(ds_full_cluster.ptr_to([0])),
                            K.cast(peer, "uint32"),
                        )
                        K.ptx.mapa.shared__cluster.u32(
                            remote_dst,
                            K.cuda.cvta_generic_to_shared(
                                arena.ptr_to([off_sds + rank * K.int32(16384)])
                            ),
                            K.cast(peer, "uint32"),
                        )
                        K.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(
                            remote_bar, K.uint32(16384)
                        )
                        K.ptx[BULK_S2C](
                            remote_dst, arena.ptr_to([off_sdsx]), K.uint32(16384), remote_bar
                        )
                    K.assign(group_edge, group_edge + K.int32(1))

            compute_group(full_n, K.int32(0), False)
            compute_group(partial_n, full_n, True)

            with K.If(work), K.Then():
                if not is_d192:
                    with K.If(K.lane_id() == K.int32(0)), K.Then():
                        ds_pipe.full.arrive(ds_prod.stage, remote=0)
                for is_dk in (False, True):
                    dkdv_pipe.full.wait(dkv_cons.stage, dkv_cons.phase)
                    tmem_offset = t_dk if is_dk else t_v
                    epi_base = off_sk if is_dk else off_sv
                    output_dim = head_dim if is_dk else head_dim_v
                    wg_cols = output_dim // 2
                    epi_cols = math.gcd(64 if direct_dkv else 32, wg_cols)
                    epi_stages = wg_cols // epi_cols
                    epi_region_bytes = 128 * epi_cols * (2 if direct_dkv else 4)
                    deterministic_kv = deterministic and qhead_per_kvhead > 1
                    kv_semaphore = dk_semaphore if is_dk else dv_semaphore
                    physical_n = task * K.int32(2) + rank
                    kv_bh = K.cast(batch_idx, "int64") * K.int64(kv_heads) + K.cast(
                        kv_head, "int64"
                    )
                    kv_sem_index = (
                        kv_bh * K.int64((tasks * 2) * 2)
                        + K.cast(physical_n, "int64") * K.int64(2)
                        + K.cast(wg, "int64")
                    )
                    if deterministic_kv:
                        _wait_eq_i32(
                            kv_semaphore,
                            kv_sem_index,
                            head % K.int32(qhead_per_kvhead),
                            crow == K.int32(0),
                        )
                        K.ptx.bar.sync(K.cast(K.int32(1) + wg, "uint32"), K.uint32(128))
                    for epi_stage in range(epi_stages):
                        values = K.alloc_local((epi_cols,), "float32")
                        for load_stage in range(epi_cols // 16):
                            _tmem_load(
                                values,
                                load_stage * 16,
                                K.cuda.get_tmem_addr(
                                    tcol,
                                    row_group,
                                    K.int32(
                                        tmem_offset
                                        + wg * wg_cols
                                        + epi_stage * epi_cols
                                        + load_stage * 16
                                    ),
                                ),
                                16,
                            )
                        K.ptx["tcgen05.wait::ld.sync.aligned"]()
                        # Direct dK bypasses the source postprocess and must
                        # apply the softmax scale here.  GQA writes an FP32
                        # accumulator; the shared source postprocess applies
                        # that scale exactly once after all query heads add.
                        if is_dk and direct_dkv:
                            for pair in range(epi_cols // 2):
                                lo, hi = _packed_binary(
                                    "mul.rn.f32x2",
                                    values[pair * 2],
                                    values[pair * 2 + 1],
                                    softmax_scale,
                                    softmax_scale,
                                )
                                K.assign(values[pair * 2], lo)
                                K.assign(values[pair * 2 + 1], hi)

                        if direct_dkv:
                            packed = K.alloc_local((epi_cols // 2,), "uint32")
                            for pair in range(epi_cols // 2):
                                K.ptx[cvt_pack](
                                    packed[pair], values[pair * 2 + 1], values[pair * 2]
                                )
                            if direct_raw_dkv:
                                seq = task * K.int32(256) + rank * K.int32(128) + crow
                                with K.If(seq < k_length), K.Then():
                                    destination = (
                                        K.cast(k_offset + seq, "int64") * K.int64(kv_heads)
                                        + K.cast(kv_head, "int64")
                                    ) * K.int64(output_dim) + K.cast(
                                        wg * K.int32(wg_cols) + K.int32(epi_stage * epi_cols),
                                        "int64",
                                    )
                                    target = dk_output if is_dk else dv_output
                                    for group in range(epi_cols // 8):
                                        base = group * 4
                                        K.ptx.st.global_.v4.b32(
                                            target.ptr_to([destination + K.int64(group * 8)]),
                                            packed[base],
                                            packed[base + 1],
                                            packed[base + 2],
                                            packed[base + 3],
                                        )
                                K.ptx.bar.warp.sync(K.uint32(0xFFFFFFFF))
                            else:
                                for group in range(epi_cols // 8):
                                    base = group * 4
                                    K.ptx.st.shared.v4.b32(
                                        arena.ptr_to(
                                            [
                                                _epi_bf16_byte(
                                                    epi_base, wg, crow, group * 8, epi_cols
                                                )
                                            ]
                                        ),
                                        packed[base],
                                        packed[base + 1],
                                        packed[base + 2],
                                        packed[base + 3],
                                    )
                                K.ptx.fence.proxy.async_.shared__cta()
                                K.ptx.bar.sync(K.cast(K.int32(1) + wg, "uint32"), K.uint32(128))
                                with (
                                    K.If(
                                        (warp_in_wg == K.int32(0))
                                        & (K.cuda.elect_sync() != K.uint32(0))
                                    ),
                                    K.Then(),
                                ):
                                    target_map = dk_map if is_dk else dv_map
                                    K.ptx[TMA_S2G](
                                        K.address_of(target_map),
                                        wg * K.int32(wg_cols) + K.int32(epi_stage * epi_cols),
                                        task * K.int32(256) + rank * K.int32(128),
                                        kv_head,
                                        batch_idx,
                                        arena.ptr_to([epi_base + wg * K.int32(epi_region_bytes)]),
                                        TMA_CACHE,
                                    )
                        else:
                            for vec in range(epi_cols // 4):
                                base = vec * 4
                                K.ptx.st.shared.v4.b32(
                                    arena.ptr_to(
                                        [
                                            epi_base
                                            + wg * K.int32(epi_region_bytes)
                                            + K.int32(vec * 2048)
                                            + crow * K.int32(16)
                                        ]
                                    ),
                                    values[base],
                                    values[base + 1],
                                    values[base + 2],
                                    values[base + 3],
                                )
                            K.ptx.fence.proxy.async_.shared__cta()
                            K.ptx.bar.sync(K.cast(K.int32(1) + wg, "uint32"), K.uint32(128))
                            with (
                                K.If(
                                    (warp_in_wg == K.int32(0))
                                    & (K.cuda.elect_sync() != K.uint32(0))
                                ),
                                K.Then(),
                            ):
                                output_base = dk_base if is_dk else dv_base
                                dst_head = (
                                    K.cast(kv_head, "int64") * K.int64(k_storage_rows)
                                    + K.cast(k_padded_offset, "int64")
                                    if varlen
                                    else kv_bh * K.int64(k128)
                                )
                                dst = (
                                    K.int64(output_base)
                                    + dst_head * K.int64(output_dim)
                                    + K.cast(physical_n, "int64") * K.int64(128 * output_dim)
                                    + K.cast(
                                        wg * K.int32(wg_cols) + K.int32(epi_stage * epi_cols),
                                        "int64",
                                    )
                                    * K.int64(128)
                                )
                                K.ptx["cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"](
                                    workspace.ptr_to([dst]),
                                    arena.ptr_to([epi_base + wg * K.int32(epi_region_bytes)]),
                                    K.uint32(128 * epi_cols * 4),
                                )

                        if not direct_raw_dkv:
                            with K.If(warp_in_wg == K.int32(0)), K.Then():
                                K.ptx.cp.async_.bulk.commit_group()
                                K.ptx.cp.async_.bulk.wait_group.read(0)
                                K.ptx.bar.arrive(K.cast(K.int32(1) + wg, "uint32"), K.uint32(160))
                            K.ptx.fence.proxy.async_.shared__cta()
                            K.ptx.bar.sync(K.cast(K.int32(1) + wg, "uint32"), K.uint32(160))
                    if deterministic_kv:
                        _release_inc_i32(kv_semaphore, kv_sem_index, crow == K.int32(0))
                    with K.If(K.lane_id() == K.int32(0)), K.Then():
                        dkdv_pipe.empty.arrive(dkv_cons.stage, remote=0, pred=True)
                    dkv_cons.advance()

            with K.If((work == K.bool(False)) & direct_dkv):
                with K.Then():
                    seq = task * K.int32(256) + rank * K.int32(128) + crow
                    with K.If(seq < k_length), K.Then():
                        token = k_offset + seq if varlen else batch_idx * K.int32(seqlen_kv) + seq
                        with K.If(wg == K.int32(0)):
                            with K.Then():
                                destination = (
                                    K.cast(token, "int64") * K.int64(kv_heads)
                                    + K.cast(kv_head, "int64")
                                ) * K.int64(head_dim)
                                for vec in range(head_dim // 8):
                                    K.ptx.st.global_.v4.b32(
                                        dk_output.ptr_to([destination + K.int64(vec * 8)]),
                                        K.uint32(0),
                                        K.uint32(0),
                                        K.uint32(0),
                                        K.uint32(0),
                                    )
                            with K.Else():
                                destination = (
                                    K.cast(token, "int64") * K.int64(kv_heads)
                                    + K.cast(kv_head, "int64")
                                ) * K.int64(head_dim_v)
                                for vec in range(head_dim_v // 8):
                                    K.ptx.st.global_.v4.b32(
                                        dv_output.ptr_to([destination + K.int64(vec * 8)]),
                                        K.uint32(0),
                                        K.uint32(0),
                                        K.uint32(0),
                                        K.uint32(0),
                                    )
            K.ptx.bar.arrive(K.uint32(BAR_TMEM[0]), K.uint32(BAR_TMEM[1]))

        with r_reduce:
            K.ptx.setmaxnreg.inc.sync.aligned.u32(K.uint32(128))
            K.ptx.barrier.sync(K.uint32(BAR_TMEM[0]), K.uint32(BAR_TMEM[1]))
            tcol = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(tcol, tmem_mailbox.ptr_to([0]))
            rtid = K.tid_in_role()
            warp_in_reduce = K.warp_id_in_role()
            row_group = warp_in_reduce * K.int32(32)
            dq_cons = K.PipelineState(1, phase=0)
            store_stage = K.local_scalar("int32", init=K.int32(0))
            bh = K.cast(batch_idx, "int64") * K.int64(heads) + K.cast(head, "int64")
            dq_head_row = (
                K.cast(head, "int64") * K.int64(q_storage_rows) + K.cast(q_padded_offset, "int64")
                if varlen
                else bh * K.int64(q128)
            )

            def reduce_group(group_count, group_offset, is_partial):
                group_edge = K.local_scalar("int32", init=K.int32(0))
                with K.While(group_edge < group_count):
                    edge = group_edge + group_offset
                    dq_pipe.full.wait(dq_cons.stage, dq_cons.phase)
                    q_block = (
                        q_block_at(edge)
                        if p00_causal_closed_plan or p06_longformer_closed_plan
                        else _load_i32(
                            partial_index if is_partial else full_index,
                            (partial_begin if is_partial else full_begin) + group_edge,
                        )
                    )
                    q_safe = K.Select(q_block < q_block_count, q_block, q_block_count - K.int32(1))
                    q_live = q_block < q_block_count
                    values = K.alloc_local((head_dim // 2,), "float32")
                    if is_d192:
                        for load_stage in range(3):
                            _tmem_load32(
                                values,
                                load_stage * 32,
                                K.cuda.get_tmem_addr(
                                    tcol, row_group, K.int32(t_dq + load_stage * 32)
                                ),
                            )
                    else:
                        # D128's cooperative accumulator presents the two halves
                        # in the source's rotated order.
                        _tmem_load32(
                            values, 32, K.cuda.get_tmem_addr(tcol, row_group, K.int32(t_dq))
                        )
                        _tmem_load32(
                            values, 0, K.cuda.get_tmem_addr(tcol, row_group, K.int32(t_dq + 32))
                        )
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    K.ptx.bar.warp.sync(K.uint32(0xFFFFFFFF))
                    with K.If(K.lane_id() == K.int32(0)), K.Then():
                        dq_pipe.empty.arrive(dq_cons.stage, remote=0, pred=True)
                    dq_cons.advance()

                    lock = K.local_scalar("int32", init=K.int32(0))
                    if deterministic:
                        K.assign(
                            lock,
                            _load_i32(
                                dq_write_order if is_partial else dq_write_order_full,
                                (partial_begin if is_partial else full_begin) + group_edge,
                            ),
                        )
                    sem_index = (bh * K.int64(q_blocks) + K.cast(q_safe, "int64")) * K.int64(
                        2
                    ) + K.cast(rank, "int64")
                    for stage in range(dq_slices):
                        smem_stage = store_stage
                        reg_start = (
                            stage * dq_ncol
                            if is_d192
                            else ((stage + dq_slices // 2) % dq_slices) * dq_ncol
                        )
                        for vec in range(dq_ncol // 4):
                            base = reg_start + vec * 4
                            K.ptx.st.shared.v4.b32(
                                arena.ptr_to(
                                    [
                                        off_sdq
                                        + smem_stage * K.int32(dq_stage_bytes)
                                        + K.int32(vec * 2048)
                                        + rtid * K.int32(16)
                                    ]
                                ),
                                values[base],
                                values[base + 1],
                                values[base + 2],
                                values[base + 3],
                            )
                        K.ptx.fence.proxy.async_.shared__cta()
                        if deterministic and stage == 0:
                            with K.If(q_live), K.Then():
                                _wait_eq_i32(dq_semaphore, sem_index, lock, rtid == K.int32(0))
                        K.ptx.bar.sync(K.uint32(BAR_REDUCE[0]), K.uint32(BAR_REDUCE[1]))
                        with K.If((warp_in_reduce == K.int32(0)) & q_live), K.Then():
                            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                                dst = (
                                    K.int64(dq_base)
                                    + dq_head_row * K.int64(head_dim)
                                    + K.cast(q_safe, "int64") * K.int64(128 * head_dim)
                                    + K.cast(
                                        rank * K.int32(head_dim // 2) + K.int32(stage * dq_ncol),
                                        "int64",
                                    )
                                    * K.int64(128)
                                )
                                K.ptx["cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"](
                                    workspace.ptr_to([dst]),
                                    arena.ptr_to([off_sdq + smem_stage * K.int32(dq_stage_bytes)]),
                                    K.uint32(dq_stage_bytes),
                                )
                            K.ptx.cp.async_.bulk.commit_group()
                            K.ptx.cp.async_.bulk.wait_group.read(dq_smem_stages - 1)
                        K.ptx.bar.sync(K.uint32(BAR_REDUCE[0]), K.uint32(BAR_REDUCE[1]))
                        K.assign(
                            store_stage,
                            K.Select(
                                store_stage == K.int32(dq_smem_stages - 1),
                                K.int32(0),
                                store_stage + K.int32(1),
                            ),
                        )
                    if is_d192:
                        # All four slices must be globally consumed before sdQacc
                        # can be reused as the next edge's 16 KiB exchange tile.
                        with K.If(warp_in_reduce == K.int32(0)), K.Then():
                            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                                if deterministic:
                                    K.ptx.cp.async_.bulk.wait_group(0)
                                else:
                                    K.ptx.cp.async_.bulk.wait_group.read(0)
                        K.ptx.bar.sync(K.uint32(BAR_REDUCE[0]), K.uint32(BAR_REDUCE[1]))
                        with K.If(K.lane_id() == K.int32(0)), K.Then():
                            dq_empty_cluster.arrive(0)
                    if deterministic:
                        with K.If(warp_in_reduce == K.int32(0)), K.Then():
                            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                                K.ptx.cp.async_.bulk.wait_group.read(0)
                        K.ptx.bar.sync(K.uint32(BAR_REDUCE[0]), K.uint32(BAR_REDUCE[1]))
                        with K.If(q_live), K.Then():
                            _release_inc_i32(dq_semaphore, sem_index, rtid == K.int32(0))
                    K.assign(group_edge, group_edge + K.int32(1))

            reduce_group(full_n, K.int32(0), False)
            reduce_group(partial_n, full_n, True)
            with K.If(work), K.Then():
                with K.If(warp_in_reduce == K.int32(0)), K.Then():
                    K.ptx.cp.async_.bulk.wait_group.read(0)
                K.ptx.bar.sync(K.uint32(BAR_REDUCE[0]), K.uint32(BAR_REDUCE[1]))
            K.ptx.cp.async_.bulk.wait_group.read(0)
            K.ptx.bar.arrive(K.uint32(BAR_TMEM[0]), K.uint32(BAR_TMEM[1]))

    return bwd.func
