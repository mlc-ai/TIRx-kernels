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

import tirx_kernels.tirx_lite as txl

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
TMA_CACHE = txl.uint64(0)


def _mma2(dest, a, b, idesc, accumulate):
    txl.ptx[MMA_F16](
        txl.cast(dest, "uint32"),
        a,
        b,
        txl.uint32(idesc),
        txl.uint32(0),
        txl.uint32(0),
        txl.uint32(0),
        txl.uint32(0),
        txl.uint32(0),
        txl.uint32(0),
        txl.uint32(0),
        txl.uint32(0),
        txl.ptx.pred(accumulate),
    )


def _mma2_ss(dest, a, b, idesc, a_offsets, b_offsets, accumulate):
    flag = txl.local_scalar("uint32", init=txl.cast(accumulate, "uint32"))
    for a_offset, b_offset in zip(a_offsets, b_offsets):
        _mma2(dest, _desc_add16(a, a_offset), _desc_add16(b, b_offset), idesc, flag)
        txl.assign(flag, txl.uint32(1))


def _mma2_dq(dest, a, b, idesc, a_offsets, b_offsets):
    flag = txl.local_scalar("uint32", init=txl.uint32(0))
    for a_offset, b_offset in zip(a_offsets, b_offsets):
        _mma2(dest, _desc_add16(a, a_offset), _desc_add16(b, b_offset), idesc, flag)
        txl.assign(flag, txl.uint32(1))


def _mma2_ts(dest, a, b, idesc, b_offsets, accumulate):
    flag = txl.local_scalar("uint32", init=txl.cast(accumulate, "uint32"))
    for phase, b_offset in enumerate(b_offsets):
        _mma2(
            dest,
            txl.cast(a + txl.uint32(phase * 8), "uint32"),
            _desc_add16(b, b_offset),
            idesc,
            flag,
        )
        txl.assign(flag, txl.uint32(1))


def _commit2(barrier, pair_mask):
    txl.ptx[TCGEN_COMMIT](barrier, txl.cast(pair_mask, "uint16"))


def _issue_cluster_tma(opcode, varlen, dst, tensor_map, feature, seq, head, batch, barrier):
    if varlen:
        txl.ptx[opcode](dst, tensor_map, feature, seq, head, barrier, TMA_CACHE)
    else:
        txl.ptx[opcode](dst, tensor_map, feature, seq, head, batch, barrier, TMA_CACHE)


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
    elem_type = {"float16": txl.f16, "bfloat16": txl.bf16}[dtype]
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

    @txl.kernel(
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
        q_map: txl.TensorMap,
        qt_map: txl.TensorMap,
        k_map: txl.TensorMap,
        kt_map: txl.TensorMap,
        v_map: txl.TensorMap,
        do_map: txl.TensorMap,
        dot_map: txl.TensorMap,
        dv_map: txl.TensorMap,
        dk_map: txl.TensorMap,
        dk_output: txl.gptr[elem_type],
        dv_output: txl.gptr[elem_type],
        cu_q: txl.gptr[txl.i32],
        cu_k: txl.gptr[txl.i32],
        partial_count: txl.gptr[txl.i32],
        partial_index: txl.gptr[txl.i32],
        full_count: txl.gptr[txl.i32],
        full_index: txl.gptr[txl.i32],
        cu_k_blocks: txl.gptr[txl.i32],
        dq_write_order: txl.gptr[txl.i32],
        dq_write_order_full: txl.gptr[txl.i32],
        partial_offset: txl.gptr[txl.i32],
        full_offset: txl.gptr[txl.i32],
        packed_mask: txl.gptr[txl.u32],
        sequence_desc: txl.gptr[txl.i32],
        work_desc: txl.gptr[txl.i32],
        dq_semaphore: txl.gptr[txl.i32],
        dk_semaphore: txl.gptr[txl.i32],
        dv_semaphore: txl.gptr[txl.i32],
        workspace: txl.gptr[txl.f32],
        softmax_scale: txl.f32,
    ):
        # The source materializes ``block_idx_in_cluster`` with
        # ``cute.arch.make_warp_uniform``.  Keep the single broadcast here so
        # every rank-derived hot-loop address remains on the uniform datapath.
        cluster_rank = txl.uniform(txl.cta_id_in_cluster([2], preferred=[2]))
        physical_block, head_axis = txl.cta_id()
        rank = cluster_rank % txl.int32(2)
        cluster_block = physical_block // txl.int32(2)
        head = txl.local_scalar("int32", init=head_axis)
        logical_block = txl.local_scalar("int32", init=cluster_block)
        plan_task = txl.local_scalar("int32", init=cluster_block)
        if long_irregular_task_major_2d:
            # A 2D grid keeps each CTA pair adjacent in X while submitting
            # all four heads of one irregular tree task before the next task.
            txl.assign(head, cluster_block)
            txl.assign(logical_block, head_axis)
            txl.assign(plan_task, head_axis)
        elif fixed_gqa_task_major:
            # Keep each cooperative CTA pair adjacent while ordering clusters
            # by K task, then by the eight GQA query heads. This matches the
            # source scheduler's adjacent-head K/V reuse for this exact shape.
            txl.assign(head, cluster_block % txl.int32(heads))
            txl.assign(logical_block, cluster_block // txl.int32(heads))
        batch_idx = txl.local_scalar("int32", init=logical_block // txl.int32(tasks))
        task = txl.local_scalar("int32", init=logical_block % txl.int32(tasks))
        if varlen:
            if use_source_varlen_nondet_schedule:
                cluster_begin = 0
                for sample, sample_tasks in enumerate(k_pairs_by_batch):
                    cluster_end = cluster_begin + sample_tasks * heads
                    with (
                        txl.If(
                            (cluster_block >= txl.int32(cluster_begin))
                            & (cluster_block < txl.int32(cluster_end))
                        ),
                        txl.Then(),
                    ):
                        sample_cluster = cluster_block - txl.int32(cluster_begin)
                        txl.assign(batch_idx, txl.int32(sample))
                        txl.assign(head, sample_cluster // txl.int32(sample_tasks))
                        txl.assign(task, sample_cluster % txl.int32(sample_tasks))
                        txl.assign(plan_task, txl.int32(pair_offsets[sample]) + task)
                    cluster_begin = cluster_end
            else:
                txl.assign(batch_idx, txl.int32(0))
                for sample in range(1, batch):
                    with (
                        txl.If(logical_block >= _load_i32(cu_k_blocks, txl.int32(sample))),
                        txl.Then(),
                    ):
                        txl.assign(batch_idx, txl.int32(sample))
                txl.assign(task, logical_block - _load_i32(cu_k_blocks, batch_idx))
            q_offset = _load_i32(cu_q, batch_idx)
            q_length = _load_i32(cu_q, batch_idx + txl.int32(1)) - q_offset
            k_offset = _load_i32(cu_k, batch_idx)
            k_length = _load_i32(cu_k, batch_idx + txl.int32(1)) - k_offset
            q_padded_offset = (
                (q_offset + batch_idx * txl.int32(128)) // txl.int32(128) * txl.int32(128)
            )
            k_padded_offset = (
                (k_offset + batch_idx * txl.int32(256)) // txl.int32(256) * txl.int32(256)
            )
            q_block_count = (q_length + txl.int32(127)) // txl.int32(128)
        else:
            q_length = txl.int32(seqlen_q)
            k_length = txl.int32(seqlen_kv)
            q_offset = txl.int32(0)
            k_offset = txl.int32(0)
            q_padded_offset = txl.int32(0)
            k_padded_offset = txl.int32(0)
            q_block_count = txl.int32(q_blocks)
        map_batch = txl.int32(0) if varlen else batch_idx
        kv_head = head // txl.int32(qhead_per_kvhead)
        mask_head = head if mask_heads > 1 else txl.int32(0)
        plan_row = (
            mask_head * _load_i32(cu_k_blocks, txl.int32(batch)) + plan_task
            if varlen
            else mask_head * txl.int32(batch * tasks) + batch_idx * txl.int32(tasks) + task
        )
        if p00_causal_closed_plan:
            # For K-pair task t, causal K2Q has two boundary Q blocks
            # (2t, 2t+1) followed by the contiguous full suffix
            # (2t+2, ..., 1023).  Keep the source plan's packed-mask payload
            # offsets while avoiding count/offset/index loads in hot loops.
            partial_n = txl.int32(2)
            partial_begin = task * txl.int32(2)
            full_n = txl.int32(1022) - task * txl.int32(2)
            full_begin = task * (txl.int32(1023) - task)
        elif p06_longformer_closed_plan:
            # The exact source Longformer K2Q plan has two full Q blocks per
            # task. Every 64th task starting at 32 is global and visits every
            # other Q block; its two neighbors deduplicate one global entry.
            task_mod64 = txl.bitwise_and(task, txl.int32(63))
            p06_global_task = task_mod64 == txl.int32(32)
            p06_adjacent_task = (task_mod64 == txl.int32(31)) | (task_mod64 == txl.int32(33))
            partial_n = txl.Select(
                p06_global_task,
                txl.int32(1022),
                txl.Select(
                    (task == txl.int32(0)) | (task == txl.int32(511)),
                    txl.int32(10),
                    txl.Select(p06_adjacent_task, txl.int32(11), txl.int32(12)),
                ),
            )
            partial_begin = (
                task * txl.int32(12)
                - txl.Select(task > txl.int32(0), txl.int32(2), txl.int32(0))
                - txl.shift_right(task + txl.int32(32), txl.int32(6))
                + txl.shift_right(task + txl.int32(31), txl.int32(6)) * txl.int32(1010)
                - txl.shift_right(task + txl.int32(30), txl.int32(6))
            )
            full_n = txl.int32(2)
            full_begin = task * txl.int32(2)
        else:
            partial_n = _load_i32(partial_count, plan_row)
            full_n = _load_i32(full_count, plan_row)
            partial_begin = _load_i32(partial_offset, plan_row)
            full_begin = _load_i32(full_offset, plan_row)
        count = full_n + partial_n
        work = (count > txl.int32(0)) & (task * txl.int32(256) < k_length)

        def q_block_at(edge):
            value = txl.local_scalar("int32")
            if p00_causal_closed_plan:
                txl.assign(
                    value,
                    txl.Select(
                        edge < full_n,
                        task * txl.int32(2) + txl.int32(2) + edge,
                        task * txl.int32(2) + edge - full_n,
                    ),
                )
            elif p06_longformer_closed_plan:
                partial_edge = txl.max(edge - txl.int32(2), txl.int32(0))
                global_q_block = txl.Select(
                    partial_edge < task * txl.int32(2), partial_edge, partial_edge + txl.int32(2)
                )
                partial_q_block = txl.Select(
                    p06_global_task,
                    global_q_block,
                    _load_i32(partial_index, partial_begin + partial_edge),
                )
                txl.assign(
                    value,
                    txl.Select(edge < txl.int32(2), task * txl.int32(2) + edge, partial_q_block),
                )
            else:
                with txl.If(edge < full_n):
                    with txl.Then():
                        txl.assign(value, _load_i32(full_index, full_begin + edge))
                    with txl.Else():
                        txl.assign(value, _load_i32(partial_index, partial_begin + edge - full_n))
            return value

        arena = txl.alloc_buffer((SHARED_BYTES,), txl.u8, scope="shared.dyn", align=1024)
        pool = txl.smem_pool(base=arena).pool
        q_pipe = _PipelinePair(txl.TMABar(pool, 1), txl.TCGen05Bar(pool, 1))
        do_pipe = _PipelinePair(txl.TMABar(pool, 1), txl.TCGen05Bar(pool, 1))
        lse_pipe = _PipelinePair(txl.TMABar(pool, 1), txl.MBarrier(pool, 1))
        sum_pipe = _PipelinePair(txl.TMABar(pool, 1), txl.MBarrier(pool, 1))
        s_pipe = _PipelinePair(txl.TCGen05Bar(pool, 1), txl.MBarrier(pool, 1))
        dp_pipe = _PipelinePair(txl.TCGen05Bar(pool, 1), txl.MBarrier(pool, 1))
        ds_pipe = _PipelinePair(txl.MBarrier(pool, 1), txl.TCGen05Bar(pool, 1))
        dkdv_pipe = _PipelinePair(txl.TCGen05Bar(pool, 2), txl.MBarrier(pool, 2))
        dq_pipe = _PipelinePair(txl.TCGen05Bar(pool, 1), txl.MBarrier(pool, 1))
        tmem_mailbox = pool.alloc((1,), "uint32", align=4)
        tmem_dealloc_bar = txl.MBarrier(pool, 1, leader=(txl.thread_id() == txl.int32(384)))
        if is_d192:
            qt_pipe = q_pipe
            # The source's barrier arena keeps the D128 Qt-pair slot even
            # though D192 aliases Qt to Q.  Preserve the hole so Kt starts at
            # byte 192 and the four cluster barriers remain at 208..232.
            pool.alloc((16,), "uint8", align=8)
        else:
            qt_pipe = _PipelinePair(txl.TMABar(pool, 1), txl.TCGen05Bar(pool, 1))
        kt_pipe = _PipelinePair(txl.TMABar(pool, 1), txl.TCGen05Bar(pool, 1))
        ds_empty_cluster = txl.MBarrier(pool, 1)
        ds_full_cluster = txl.MBarrier(pool, 1)
        ds_leader_cluster = txl.MBarrier(pool, 1)
        dq_empty_cluster = txl.MBarrier(pool, 1) if is_d192 else None
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
        txl.ptx.fence.proxy.async_.shared__cta()
        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.barrier.cluster.arrive.relaxed()
        txl.ptx.barrier.cluster.wait()

        pair_mask = txl.bitwise_or(txl.int32(1), txl.int32(2))
        q_full_leader = q_pipe.full.remote_view(0)
        do_full_leader = do_pipe.full.remote_view(0)
        qt_full_leader = qt_pipe.full.remote_view(0)
        kt_full_leader = kt_pipe.full.remote_view(0)
        smem_base = txl.local_scalar(
            "uint32", init=txl.cuda.cvta_generic_to_shared(arena.ptr_to([0]))
        )

        is_mha = qhead_per_kvhead == 1
        if is_mha and not deterministic:
            compute_regs, producer_regs = 144, 96
        else:
            compute_regs, producer_regs = 136, 112
        sp = txl.specialize(chain_dispatch=True)
        r_empty = sp.role("empty", warps=[15], regs=producer_regs)
        r_relay = sp.role("relay", warps=[14], regs=producer_regs)
        r_load = sp.role("load", warps=[13], regs=producer_regs)
        r_mma = sp.role("mma", warps=[12], regs=producer_regs)
        r_compute = sp.role("compute", warps=list(range(4, 12)), regs=compute_regs)
        r_reduce = sp.role("reduce", warps=list(range(4)), regs=128)

        with r_empty:
            pass

        with r_relay:
            relay_phase = txl.local_scalar("int32", init=txl.int32(0))
            edge = txl.local_scalar("int32", init=txl.int32(0))
            with txl.While(edge < count):
                ds_full_cluster.wait(0, relay_phase)
                with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                    ds_leader_cluster.arrive(0, remote=0)
                txl.assign(relay_phase, relay_phase ^ txl.int32(1))
                txl.assign(edge, edge + txl.int32(1))

        with r_load:
            elected = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
            with txl.If(elected != txl.uint32(0)), txl.Then():
                for tensor_map in (q_map, qt_map, k_map, kt_map, v_map, do_map, dot_map):
                    txl.ptx.prefetch.tensormap(txl.address_of(tensor_map))
                if direct_dkv and not direct_raw_dkv:
                    txl.ptx.prefetch.tensormap(txl.address_of(dv_map))
                    txl.ptx.prefetch.tensormap(txl.address_of(dk_map))

            q_prod = txl.PipelineState(1, phase=1)
            do_prod = txl.PipelineState(1, phase=1)
            qt_prod = txl.PipelineState(1, phase=1)
            kt_prod = txl.PipelineState(1, phase=1)
            lse_prod = txl.PipelineState(1, phase=1)
            sum_prod = txl.PipelineState(1, phase=1)
            previous_q = txl.local_scalar("int32", init=txl.int32(0))
            edge = txl.local_scalar("int32", init=txl.int32(0))
            bh = txl.cast(batch_idx, "int64") * txl.int64(heads) + txl.cast(head, "int64")
            bh_q = (
                txl.cast(head, "int64") * txl.int64(q_storage_rows)
                + txl.cast(q_padded_offset, "int64")
                if varlen
                else bh * txl.int64(q128)
            )

            # D192 aliases Q/Qt and dO/dOt.  Each edge therefore has two
            # epochs on each one-stage barrier: Q then Qt, and dOt then dO.
            # K/V/Kt are resident and join only the first matching epoch.
            if is_d192:
                q192_prod = txl.PipelineState(1, phase=1)
                do192_prod = txl.PipelineState(1, phase=1)
                lse192_prod = txl.PipelineState(1, phase=1)
                sum192_prod = txl.PipelineState(1, phase=1)
                edge192 = txl.local_scalar("int32", init=txl.int32(0))
                with txl.While(edge192 < count):
                    q_block = q_block_at(edge192)
                    q_safe = txl.Select(
                        q_block < q_block_count, q_block, q_block_count - txl.int32(1)
                    )
                    first = edge192 == txl.int32(0)

                    # The first edge brings in resident K together with Q.
                    with txl.If(first), txl.Then():
                        q_pipe.empty.wait(q192_prod.stage, q192_prod.phase)
                        with txl.If(elected != txl.uint32(0)), txl.Then():
                            with txl.If(rank == txl.int32(0)), txl.Then():
                                q_pipe.full.arrive(q192_prod.stage, txl.uint32(147456))
                            for feature_chunk in range(3):
                                _issue_cluster_tma(
                                    tma_g2s,
                                    varlen,
                                    arena.ptr_to([off_sk + feature_chunk * 16384]),
                                    txl.address_of(k_map),
                                    txl.int32(feature_chunk * 64),
                                    k_offset + task * txl.int32(256) + rank * txl.int32(128),
                                    kv_head,
                                    map_batch,
                                    q_full_leader.ptr_to([q192_prod.stage]),
                                )
                                _issue_cluster_tma(
                                    tma_g2s,
                                    varlen,
                                    arena.ptr_to([off_sq + feature_chunk * 8192]),
                                    txl.address_of(q_map),
                                    txl.int32(feature_chunk * 64),
                                    q_offset + q_safe * txl.int32(128) + rank * txl.int32(64),
                                    head,
                                    map_batch,
                                    q_full_leader.ptr_to([q192_prod.stage]),
                                )
                        q192_prod.advance()

                    lse_pipe.empty.wait(lse192_prod.stage, lse192_prod.phase)
                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        lse_pipe.full.arrive(lse192_prod.stage, txl.uint32(512))
                        txl.ptx[BULK_G2S](
                            arena.ptr_to([off_slse]),
                            workspace.ptr_to(
                                [
                                    txl.int64(sum_plane)
                                    + bh_q
                                    + txl.cast(q_safe, "int64") * txl.int64(128)
                                ]
                            ),
                            txl.uint32(512),
                            lse_pipe.full.ptr_to([lse192_prod.stage]),
                        )
                    lse192_prod.advance()

                    # Later edges bring in Q after LSE, matching the source
                    # overlap.  The byte count is just the three Q chunks.
                    with txl.If(first == txl.bool(False)), txl.Then():
                        q_pipe.empty.wait(q192_prod.stage, q192_prod.phase)
                        with txl.If(elected != txl.uint32(0)), txl.Then():
                            with txl.If(rank == txl.int32(0)), txl.Then():
                                q_pipe.full.arrive(q192_prod.stage, txl.uint32(49152))
                            for feature_chunk in range(3):
                                _issue_cluster_tma(
                                    tma_g2s,
                                    varlen,
                                    arena.ptr_to([off_sq + feature_chunk * 8192]),
                                    txl.address_of(q_map),
                                    txl.int32(feature_chunk * 64),
                                    q_offset + q_safe * txl.int32(128) + rank * txl.int32(64),
                                    head,
                                    map_batch,
                                    q_full_leader.ptr_to([q192_prod.stage]),
                                )
                        q192_prod.advance()

                    # dOt shares sdO.  Resident V joins the first dOt epoch.
                    do_pipe.empty.wait(do192_prod.stage, do192_prod.phase)
                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        do_bytes = txl.local_scalar("uint32", init=txl.uint32(32768))
                        with txl.If(first), txl.Then():
                            txl.assign(do_bytes, txl.uint32(98304))
                        with txl.If(rank == txl.int32(0)), txl.Then():
                            do_pipe.full.arrive(do192_prod.stage, do_bytes)
                        with txl.If(first), txl.Then():
                            for feature_chunk in range(2):
                                _issue_cluster_tma(
                                    tma_g2s,
                                    varlen,
                                    arena.ptr_to([off_sv + feature_chunk * 16384]),
                                    txl.address_of(v_map),
                                    txl.int32(feature_chunk * 64),
                                    k_offset + task * txl.int32(256) + rank * txl.int32(128),
                                    kv_head,
                                    map_batch,
                                    do_full_leader.ptr_to([do192_prod.stage]),
                                )
                        for feature_chunk in range(2):
                            _issue_cluster_tma(
                                tma_g2s,
                                varlen,
                                arena.ptr_to([off_sdot + feature_chunk * 8192]),
                                txl.address_of(dot_map),
                                txl.int32(feature_chunk * 64),
                                q_offset + q_safe * txl.int32(128) + rank * txl.int32(64),
                                head,
                                map_batch,
                                do_full_leader.ptr_to([do192_prod.stage]),
                            )
                    do192_prod.advance()

                    sum_pipe.empty.wait(sum192_prod.stage, sum192_prod.phase)
                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        sum_pipe.full.arrive(sum192_prod.stage, txl.uint32(512))
                        txl.ptx[BULK_G2S](
                            arena.ptr_to([off_ssum]),
                            workspace.ptr_to([bh_q + txl.cast(q_safe, "int64") * txl.int64(128)]),
                            txl.uint32(512),
                            sum_pipe.full.ptr_to([sum192_prod.stage]),
                        )
                    sum192_prod.advance()

                    # Qt overwrites Q after score consumption.  Kt is loaded
                    # beside it only once and remains resident for all dQ.
                    q_pipe.empty.wait(q192_prod.stage, q192_prod.phase)
                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        with txl.If(rank == txl.int32(0)), txl.Then():
                            q_bytes = txl.local_scalar("uint32", init=txl.uint32(49152))
                            with txl.If(first), txl.Then():
                                txl.assign(q_bytes, txl.uint32(147456))
                            q_pipe.full.arrive(q192_prod.stage, q_bytes)
                        for feature_chunk in range(3):
                            _issue_cluster_tma(
                                tma_g2s,
                                varlen,
                                arena.ptr_to([off_sqt + feature_chunk * 8192]),
                                txl.address_of(qt_map),
                                rank * txl.int32(96) + txl.int32(feature_chunk * 32),
                                q_offset + q_safe * txl.int32(128),
                                head,
                                map_batch,
                                q_full_leader.ptr_to([q192_prod.stage]),
                            )
                        with txl.If(first), txl.Then():
                            for feature_chunk in range(3):
                                _issue_cluster_tma(
                                    tma_g2s,
                                    varlen,
                                    arena.ptr_to([off_skt + feature_chunk * 16384]),
                                    txl.address_of(kt_map),
                                    rank * txl.int32(96) + txl.int32(feature_chunk * 32),
                                    k_offset + task * txl.int32(256),
                                    kv_head,
                                    map_batch,
                                    q_full_leader.ptr_to([q192_prod.stage]),
                                )
                    q192_prod.advance()

                    # dO overwrites dOt only after the dP consumer releases it.
                    do_pipe.empty.wait(do192_prod.stage, do192_prod.phase)
                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        with txl.If(rank == txl.int32(0)), txl.Then():
                            do_pipe.full.arrive(do192_prod.stage, txl.uint32(32768))
                        _issue_cluster_tma(
                            tma_g2s,
                            varlen,
                            arena.ptr_to([off_sdo]),
                            txl.address_of(do_map),
                            rank * txl.int32(64),
                            q_offset + q_safe * txl.int32(128),
                            head,
                            map_batch,
                            do_full_leader.ptr_to([do192_prod.stage]),
                        )
                    do192_prod.advance()
                    txl.assign(edge192, edge192 + txl.int32(1))

            d128_count = count if not is_d192 else txl.int32(0)
            with txl.While(edge < d128_count):
                q_block = q_block_at(edge)
                q_safe = txl.Select(q_block < q_block_count, q_block, q_block_count - txl.int32(1))
                first = edge == txl.int32(0)

                with txl.If(edge > txl.int32(0)), txl.Then():
                    qt_pipe.empty.wait(qt_prod.stage, qt_prod.phase)
                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        with txl.If(rank == txl.int32(0)), txl.Then():
                            qt_pipe.full.arrive(qt_prod.stage, txl.uint32(32768))
                        _issue_cluster_tma(
                            tma_g2s,
                            varlen,
                            arena.ptr_to([off_sqt]),
                            txl.address_of(qt_map),
                            rank * txl.int32(64),
                            q_offset + previous_q * txl.int32(128),
                            head,
                            map_batch,
                            qt_full_leader.ptr_to([qt_prod.stage]),
                        )
                    qt_prod.advance()

                q_pipe.empty.wait(q_prod.stage, q_prod.phase)
                with txl.If(elected != txl.uint32(0)), txl.Then():
                    q_bytes = txl.local_scalar("uint32", init=txl.uint32(32768))
                    with txl.If(first), txl.Then():
                        txl.assign(q_bytes, txl.uint32(98304))
                    with txl.If(rank == txl.int32(0)), txl.Then():
                        q_pipe.full.arrive(q_prod.stage, q_bytes)
                    with txl.If(first), txl.Then():
                        for feature_half in range(2):
                            _issue_cluster_tma(
                                tma_g2s,
                                varlen,
                                arena.ptr_to([off_sk + feature_half * 16384]),
                                txl.address_of(k_map),
                                txl.int32(feature_half * 64),
                                k_offset + task * txl.int32(256) + rank * txl.int32(128),
                                kv_head,
                                map_batch,
                                q_full_leader.ptr_to([q_prod.stage]),
                            )
                    for feature_half in range(2):
                        _issue_cluster_tma(
                            tma_g2s,
                            varlen,
                            arena.ptr_to([off_sq + feature_half * 8192]),
                            txl.address_of(q_map),
                            txl.int32(feature_half * 64),
                            q_offset + q_safe * txl.int32(128) + rank * txl.int32(64),
                            head,
                            map_batch,
                            q_full_leader.ptr_to([q_prod.stage]),
                        )
                q_prod.advance()

                lse_pipe.empty.wait(lse_prod.stage, lse_prod.phase)
                with txl.If(elected != txl.uint32(0)), txl.Then():
                    lse_pipe.full.arrive(lse_prod.stage, txl.uint32(512))
                    txl.ptx[BULK_G2S](
                        arena.ptr_to([off_slse]),
                        workspace.ptr_to(
                            [
                                txl.int64(sum_plane)
                                + bh_q
                                + txl.cast(q_safe, "int64") * txl.int64(128)
                            ]
                        ),
                        txl.uint32(512),
                        lse_pipe.full.ptr_to([lse_prod.stage]),
                    )
                lse_prod.advance()

                do_pipe.empty.wait(do_prod.stage, do_prod.phase)
                with txl.If(elected != txl.uint32(0)), txl.Then():
                    do_bytes = txl.local_scalar("uint32", init=txl.uint32(65536))
                    with txl.If(first), txl.Then():
                        txl.assign(do_bytes, txl.uint32(131072))
                    with txl.If(rank == txl.int32(0)), txl.Then():
                        do_pipe.full.arrive(do_prod.stage, do_bytes)
                    with txl.If(first), txl.Then():
                        for feature_half in range(2):
                            _issue_cluster_tma(
                                tma_g2s,
                                varlen,
                                arena.ptr_to([off_sv + feature_half * 16384]),
                                txl.address_of(v_map),
                                txl.int32(feature_half * 64),
                                k_offset + task * txl.int32(256) + rank * txl.int32(128),
                                kv_head,
                                map_batch,
                                do_full_leader.ptr_to([do_prod.stage]),
                            )
                    _issue_cluster_tma(
                        tma_g2s,
                        varlen,
                        arena.ptr_to([off_sdo]),
                        txl.address_of(do_map),
                        rank * txl.int32(64),
                        q_offset + q_safe * txl.int32(128),
                        head,
                        map_batch,
                        do_full_leader.ptr_to([do_prod.stage]),
                    )
                    for feature_half in range(2):
                        _issue_cluster_tma(
                            tma_g2s,
                            varlen,
                            arena.ptr_to([off_sdot + feature_half * 8192]),
                            txl.address_of(dot_map),
                            txl.int32(feature_half * 64),
                            q_offset + q_safe * txl.int32(128) + rank * txl.int32(64),
                            head,
                            map_batch,
                            do_full_leader.ptr_to([do_prod.stage]),
                        )
                do_prod.advance()

                sum_pipe.empty.wait(sum_prod.stage, sum_prod.phase)
                with txl.If(elected != txl.uint32(0)), txl.Then():
                    sum_pipe.full.arrive(sum_prod.stage, txl.uint32(512))
                    txl.ptx[BULK_G2S](
                        arena.ptr_to([off_ssum]),
                        workspace.ptr_to([bh_q + txl.cast(q_safe, "int64") * txl.int64(128)]),
                        txl.uint32(512),
                        sum_pipe.full.ptr_to([sum_prod.stage]),
                    )
                sum_prod.advance()

                with txl.If(first), txl.Then():
                    kt_pipe.empty.wait(kt_prod.stage, kt_prod.phase)
                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        with txl.If(rank == txl.int32(0)), txl.Then():
                            kt_pipe.full.arrive(kt_prod.stage, txl.uint32(65536))
                        _issue_cluster_tma(
                            tma_g2s,
                            varlen,
                            arena.ptr_to([off_skt]),
                            txl.address_of(kt_map),
                            rank * txl.int32(64),
                            k_offset + task * txl.int32(256),
                            kv_head,
                            map_batch,
                            kt_full_leader.ptr_to([kt_prod.stage]),
                        )
                    kt_prod.advance()
                txl.assign(previous_q, q_safe)
                txl.assign(edge, edge + txl.int32(1))

            if not is_d192:
                with txl.If(work), txl.Then():
                    qt_pipe.empty.wait(qt_prod.stage, qt_prod.phase)
                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        with txl.If(rank == txl.int32(0)), txl.Then():
                            qt_pipe.full.arrive(qt_prod.stage, txl.uint32(32768))
                        _issue_cluster_tma(
                            tma_g2s,
                            varlen,
                            arena.ptr_to([off_sqt]),
                            txl.address_of(qt_map),
                            rank * txl.int32(64),
                            q_offset + previous_q * txl.int32(128),
                            head,
                            map_batch,
                            qt_full_leader.ptr_to([qt_prod.stage]),
                        )

        with r_mma:
            txl.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                txl.address_of(tmem_mailbox), txl.uint32(TMEM_COLUMNS)
            )
            txl.ptx.barrier.sync(txl.uint32(BAR_TMEM[0]), txl.uint32(BAR_TMEM[1]))
            tcol = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(tcol, tmem_mailbox.ptr_to([0]))
            elected = txl.local_scalar("uint32", init=txl.cuda.elect_sync())

            if is_d192:
                with txl.If((rank == txl.int32(0)) & work), txl.Then():
                    d_k = _desc_at(txl.uint64(desc_krow_base), smem_base + txl.uint32(off_sk))
                    d_v = _desc_at(txl.uint64(desc_krow_base), smem_base + txl.uint32(off_sv))
                    d_q = _desc_at(txl.uint64(desc_row_base), smem_base + txl.uint32(off_sq))
                    d_dot = _desc_at(txl.uint64(desc_row_base), smem_base + txl.uint32(off_sdot))
                    d_do = _desc_at(txl.uint64(desc_mn_base), smem_base + txl.uint32(off_sdo))
                    d192_high = txl.shift_left(txl.uint64(0x80004020), txl.uint64(32))
                    d_qt = _desc_at(
                        txl.bitwise_or(d192_high, txl.uint64(0x02000000)),
                        smem_base + txl.uint32(off_sqt),
                    )
                    d_kt = _desc_at(
                        txl.bitwise_or(d192_high, txl.uint64(0x04000000)),
                        smem_base + txl.uint32(off_skt),
                    )
                    d_ds = _desc_at(txl.uint64(desc_mn_base), smem_base + txl.uint32(off_sds))
                    ts = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(t_s))
                    tp = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(t_p))
                    tdq = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(t_dq))
                    tv = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(t_v))
                    tdp = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(t_dp))
                    tds = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(t_ds))
                    tdk = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(t_dk))

                    q_cons = txl.PipelineState(1, phase=0)
                    do_cons = txl.PipelineState(1, phase=0)
                    s_prod = txl.PipelineState(1, phase=0)
                    dp_prod = txl.PipelineState(1, phase=0)
                    ds_cons = txl.PipelineState(1, phase=0)
                    dq_prod = txl.PipelineState(1, phase=0)
                    dkv_prod = txl.PipelineState(2, phase=0)
                    dk_accumulate = txl.local_scalar("uint32", init=txl.uint32(0))
                    dv_accumulate = txl.local_scalar("uint32", init=txl.uint32(0))
                    edge = txl.local_scalar("int32", init=txl.int32(0))

                    with txl.While(edge < count):
                        q_pipe.full.wait(q_cons.stage, q_cons.phase)
                        dq_pipe.empty.wait(dq_prod.stage, dq_prod.phase ^ 1)
                        with txl.If(elected != txl.uint32(0)), txl.Then():
                            _mma2_ss(ts, d_k, d_q, id_ss, krow_offsets, row_offsets, txl.uint32(0))
                            _commit2(s_pipe.full.ptr_to([s_prod.stage]), pair_mask)
                            _commit2(q_pipe.empty.ptr_to([q_cons.stage]), pair_mask)
                        q_cons.advance()
                        s_prod.advance()

                        do_pipe.full.wait(do_cons.stage, do_cons.phase)
                        # S/P share TMEM columns 320..383.  The compute role
                        # releases S only after its score loads, so dP must
                        # acquire that toggled empty phase before overwrite.
                        s_pipe.empty.wait(s_prod.stage, s_prod.phase ^ 1)
                        with txl.If(elected != txl.uint32(0)), txl.Then():
                            _mma2_ss(
                                tdp,
                                d_v,
                                d_dot,
                                id_ss,
                                dv_krow_offsets,
                                do_row_offsets,
                                txl.uint32(0),
                            )
                            _commit2(dp_pipe.full.ptr_to([dp_prod.stage]), pair_mask)
                            _commit2(do_pipe.empty.ptr_to([do_cons.stage]), pair_mask)
                        do_cons.advance()
                        dp_prod.advance()

                        q_pipe.full.wait(q_cons.stage, q_cons.phase)
                        dp_pipe.empty.wait(dp_prod.stage, dp_prod.phase ^ 1)
                        with txl.If(elected != txl.uint32(0)), txl.Then():
                            _mma2_ts(tdk, tds, d_qt, id_dk, dk_offsets, dk_accumulate)
                            _commit2(q_pipe.empty.ptr_to([q_cons.stage]), pair_mask)
                        txl.assign(dk_accumulate, txl.uint32(1))
                        q_cons.advance()

                        do_pipe.full.wait(do_cons.stage, do_cons.phase)
                        with txl.If(elected != txl.uint32(0)), txl.Then():
                            _mma2_ts(tv, tp, d_do, id_ts, mn8_offsets, dv_accumulate)
                            _commit2(do_pipe.empty.ptr_to([do_cons.stage]), pair_mask)
                        txl.assign(dv_accumulate, txl.uint32(1))
                        do_cons.advance()

                        ds_pipe.full.wait(ds_cons.stage, ds_cons.phase)
                        ds_leader_cluster.wait(0, ds_cons.phase)
                        with txl.If(elected != txl.uint32(0)), txl.Then():
                            _mma2_dq(tdq, d_ds, d_kt, id_dq, dq_a_offsets, dq_b_offsets)
                            _commit2(dq_pipe.full.ptr_to([dq_prod.stage]), pair_mask)
                            _commit2(ds_pipe.empty.ptr_to([ds_cons.stage]), pair_mask)
                        ds_cons.advance()
                        dq_prod.advance()
                        txl.assign(edge, edge + txl.int32(1))

                    dkdv_pipe.empty.wait(dkv_prod.stage, dkv_prod.phase ^ 1)
                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        _commit2(dkdv_pipe.full.ptr_to([dkv_prod.stage]), pair_mask)
                    dkv_prod.advance()
                    dkdv_pipe.empty.wait(dkv_prod.stage, dkv_prod.phase ^ 1)
                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        _commit2(dkdv_pipe.full.ptr_to([dkv_prod.stage]), pair_mask)
            else:
                with txl.If((rank == txl.int32(0)) & work), txl.Then():
                    d_k = _desc_at(txl.uint64(desc_krow_base), smem_base + txl.uint32(off_sk))
                    d_v = _desc_at(txl.uint64(desc_krow_base), smem_base + txl.uint32(off_sv))
                    d_q = _desc_at(txl.uint64(desc_row_base), smem_base + txl.uint32(off_sq))
                    d_dot = _desc_at(txl.uint64(desc_row_base), smem_base + txl.uint32(off_sdot))
                    d_do = _desc_at(txl.uint64(desc_mn_base), smem_base + txl.uint32(off_sdo))
                    d_qt = _desc_at(txl.uint64(desc_mn_base), smem_base + txl.uint32(off_sqt))
                    d_kt = _desc_at(txl.uint64(desc_mn_base), smem_base + txl.uint32(off_skt))
                    d_ds = _desc_at(txl.uint64(desc_mn_base), smem_base + txl.uint32(off_sds))
                    ts = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(t_s))
                    tdq = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(t_dq))
                    tv = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(t_v))
                    tdp = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(t_dp))
                    tds = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(t_ds))
                    tdk = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(t_dk))

                    q_cons = txl.PipelineState(1, phase=0)
                    do_cons = txl.PipelineState(1, phase=0)
                    qt_cons = txl.PipelineState(1, phase=0)
                    kt_cons = txl.PipelineState(1, phase=0)
                    s_prod = txl.PipelineState(1, phase=0)
                    dp_prod = txl.PipelineState(1, phase=0)
                    ds_cons = txl.PipelineState(1, phase=0)
                    dq_prod = txl.PipelineState(1, phase=0)
                    dkv_prod = txl.PipelineState(2, phase=0)

                    q_pipe.full.wait(q_cons.stage, q_cons.phase)
                    s_pipe.empty.wait(s_prod.stage, s_prod.phase ^ 1)
                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        _mma2_ss(ts, d_k, d_q, id_ss, krow_offsets, row_offsets, txl.uint32(0))
                        _commit2(s_pipe.full.ptr_to([s_prod.stage]), pair_mask)
                        _commit2(q_pipe.empty.ptr_to([q_cons.stage]), pair_mask)
                    q_cons.advance()
                    s_prod.advance()

                    do_pipe.full.wait(do_cons.stage, do_cons.phase)
                    dp_pipe.empty.wait(dp_prod.stage, dp_prod.phase ^ 1)
                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        _mma2_ss(
                            tdp, d_v, d_dot, id_ss, dv_krow_offsets, do_row_offsets, txl.uint32(0)
                        )
                        _commit2(dp_pipe.full.ptr_to([dp_prod.stage]), pair_mask)
                    dp_prod.advance()

                    s_pipe.empty.wait(s_prod.stage, s_prod.phase ^ 1)
                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        _mma2_ts(tv, ts, d_do, id_ts, mn8_offsets, txl.uint32(0))
                        _commit2(do_pipe.empty.ptr_to([do_cons.stage]), pair_mask)
                    do_cons.advance()
                    kt_pipe.full.wait(kt_cons.stage, kt_cons.phase)

                    edge = txl.local_scalar("int32", init=txl.int32(1))
                    dk_accumulate = txl.local_scalar("uint32", init=txl.uint32(0))
                    with txl.While(edge < count):
                        q_pipe.full.wait(q_cons.stage, q_cons.phase)
                        dq_pipe.empty.wait(dq_prod.stage, dq_prod.phase ^ 1)
                        with txl.If(elected != txl.uint32(0)), txl.Then():
                            _mma2_ss(ts, d_k, d_q, id_ss, krow_offsets, row_offsets, txl.uint32(0))
                            _commit2(s_pipe.full.ptr_to([s_prod.stage]), pair_mask)
                            _commit2(q_pipe.empty.ptr_to([q_cons.stage]), pair_mask)
                        q_cons.advance()
                        s_prod.advance()

                        qt_pipe.full.wait(qt_cons.stage, qt_cons.phase)
                        dp_pipe.empty.wait(dp_prod.stage, dp_prod.phase ^ 1)
                        ds_pipe.full.wait(ds_cons.stage, ds_cons.phase)
                        with txl.If(elected != txl.uint32(0)), txl.Then():
                            _mma2_ts(tdk, tds, d_qt, id_dk, mn8_offsets, dk_accumulate)
                            _commit2(qt_pipe.empty.ptr_to([qt_cons.stage]), pair_mask)
                        txl.assign(dk_accumulate, txl.uint32(1))
                        qt_cons.advance()

                        do_pipe.full.wait(do_cons.stage, do_cons.phase)
                        with txl.If(elected != txl.uint32(0)), txl.Then():
                            _mma2_ss(
                                tdp,
                                d_v,
                                d_dot,
                                id_ss,
                                dv_krow_offsets,
                                do_row_offsets,
                                txl.uint32(0),
                            )
                            _commit2(dp_pipe.full.ptr_to([dp_prod.stage]), pair_mask)
                        dp_prod.advance()

                        ds_leader_cluster.wait(0, ds_cons.phase)
                        with txl.If(elected != txl.uint32(0)), txl.Then():
                            _mma2_dq(tdq, d_ds, d_kt, id_dq, dq_a_offsets, dq_b_offsets)
                            _commit2(dq_pipe.full.ptr_to([dq_prod.stage]), pair_mask)
                            _commit2(ds_pipe.empty.ptr_to([ds_cons.stage]), pair_mask)
                        ds_cons.advance()
                        dq_prod.advance()

                        s_pipe.empty.wait(s_prod.stage, s_prod.phase ^ 1)
                        with txl.If(elected != txl.uint32(0)), txl.Then():
                            _mma2_ts(tv, ts, d_do, id_ts, mn8_offsets, txl.uint32(1))
                            _commit2(do_pipe.empty.ptr_to([do_cons.stage]), pair_mask)
                        do_cons.advance()
                        txl.assign(edge, edge + txl.int32(1))

                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        _commit2(s_pipe.full.ptr_to([s_prod.stage]), pair_mask)
                    dkdv_pipe.empty.wait(dkv_prod.stage, dkv_prod.phase ^ 1)
                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        _commit2(dkdv_pipe.full.ptr_to([dkv_prod.stage]), pair_mask)
                    dkv_prod.advance()

                    dkdv_pipe.empty.wait(dkv_prod.stage, dkv_prod.phase ^ 1)
                    qt_pipe.full.wait(qt_cons.stage, qt_cons.phase)
                    dp_pipe.empty.wait(dp_prod.stage, dp_prod.phase ^ 1)
                    ds_pipe.full.wait(ds_cons.stage, ds_cons.phase)
                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        _mma2_ts(tdk, tds, d_qt, id_dk, mn8_offsets, dk_accumulate)
                        _commit2(qt_pipe.empty.ptr_to([qt_cons.stage]), pair_mask)
                        _commit2(dkdv_pipe.full.ptr_to([dkv_prod.stage]), pair_mask)
                    qt_cons.advance()
                    dkv_prod.advance()

                    ds_leader_cluster.wait(0, ds_cons.phase)
                    dq_pipe.empty.wait(dq_prod.stage, dq_prod.phase ^ 1)
                    with txl.If(elected != txl.uint32(0)), txl.Then():
                        _mma2_dq(tdq, d_ds, d_kt, id_dq, dq_a_offsets, dq_b_offsets)
                        _commit2(dq_pipe.full.ptr_to([dq_prod.stage]), pair_mask)
                        _commit2(ds_pipe.empty.ptr_to([ds_cons.stage]), pair_mask)
                        _commit2(kt_pipe.empty.ptr_to([kt_cons.stage]), pair_mask)

            txl.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
            txl.ptx.barrier.sync(txl.uint32(BAR_TMEM[0]), txl.uint32(BAR_TMEM[1]))
            tmem_dealloc_bar.arrive(0, remote=txl.int32(1) - rank, pred=True)
            tmem_dealloc_bar.wait(0, 0)
            txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](tcol, txl.uint32(TMEM_COLUMNS))

        with r_compute:
            txl.ptx.barrier.sync(txl.uint32(BAR_TMEM[0]), txl.uint32(BAR_TMEM[1]))
            tcol = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(tcol, tmem_mailbox.ptr_to([0]))
            ctid = txl.thread_id() % txl.int32(256)
            crow = ctid % txl.int32(128)
            wg = ctid // txl.int32(128)
            row_group = (crow // txl.int32(32)) * txl.int32(32)
            warp_in_compute = txl.warp_id_in_role()
            warp_in_wg = warp_in_compute % txl.int32(4)
            scale_log2 = txl.local_scalar("float32")
            txl.ptx.mul.f32(scale_log2, softmax_scale, txl.float32(1.4426950408889634))
            s_cons = txl.PipelineState(1, phase=0)
            lse_cons = txl.PipelineState(1, phase=0)
            sum_cons = txl.PipelineState(1, phase=0)
            dp_cons = txl.PipelineState(1, phase=0)
            ds_prod = txl.PipelineState(1, phase=1)
            dkv_cons = txl.PipelineState(2, phase=0)

            def compute_group(group_count, group_offset, is_partial):
                group_edge = txl.local_scalar("int32", init=txl.int32(0))
                with txl.While(group_edge < group_count):
                    edge = group_edge + group_offset
                    lse_pipe.full.wait(lse_cons.stage, lse_cons.phase)
                    s_pipe.full.wait(s_cons.stage, s_cons.phase)
                    scores = txl.alloc_local((64,), "float32")
                    for rep in range(2):
                        _tmem_load32(
                            scores,
                            rep * 32,
                            txl.cuda.get_tmem_addr(
                                tcol, row_group, txl.int32(t_s + wg * 32 + rep * 64)
                            ),
                        )
                    txl.ptx["tcgen05.wait::ld.sync.aligned"]()
                    if is_d192:
                        # P aliases S in D192, so release the S accumulator as
                        # soon as the TMEM reads have completed, before R2T P.
                        with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                            s_pipe.empty.arrive(s_cons.stage, remote=0, pred=True)
                        s_cons.advance()
                    else:
                        # D128 publishes dS one iteration late because S and the
                        # exchanged dS lifetime overlap in its TMEM schedule.
                        with txl.If(edge > txl.int32(0)), txl.Then():
                            with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                                ds_pipe.full.arrive(ds_prod.stage, remote=0)
                            ds_prod.advance()

                    # Partial payloads already include the sequence-tail mask, and
                    # full K2Q entries are complete 256-key tiles by construction.
                    if is_partial:
                        mask_words = txl.alloc_local((2,), "uint32")
                        payload = partial_begin + group_edge
                        mask_word = (
                            txl.cast(payload, "int64") * txl.int64(2) + txl.cast(rank, "int64")
                        ) * txl.int64(512) + txl.cast(ctid, "int64") * txl.int64(2)
                        txl.ptx.ld.global_.v2.b32(
                            mask_words[0], mask_words[1], packed_mask.ptr_to([mask_word])
                        )
                        for j in range(64):
                            keep = txl.bitwise_and(
                                txl.shift_right(mask_words[j // 32], txl.uint32(j % 32)),
                                txl.uint32(1),
                            )
                            score_bits = txl.reinterpret(txl.u32, scores[j])
                            drop = txl.local_scalar("uint32")
                            txl.ptx.setp.eq.u32(drop, keep, txl.uint32(0))
                            masked_bits = txl.local_scalar("uint32")
                            txl.ptx.selp.b32(
                                masked_bits, txl.uint32(0xFF800000), score_bits, txl.ptx.pred(drop)
                            )
                            txl.assign(scores[j], txl.reinterpret(txl.f32, masked_bits))
                    for rep in range(2):
                        for pair in range(16):
                            j = rep * 32 + pair * 2
                            qcol = wg * txl.int32(32) + txl.int32(rep * 64 + pair * 2)
                            lse0 = txl.local_scalar("float32")
                            lse1 = txl.local_scalar("float32")
                            txl.ptx.ld.shared_.v2.b32(
                                lse0, lse1, arena.ptr_to([off_slse + qcol * txl.int32(4)])
                            )
                            neg0 = txl.local_scalar("float32")
                            neg1 = txl.local_scalar("float32")
                            txl.ptx.neg.f32(neg0, lse0)
                            txl.ptx.neg.f32(neg1, lse1)
                            lo, hi = _packed_fma(
                                scores[j], scores[j + 1], scale_log2, scale_log2, neg0, neg1
                            )
                            txl.ptx.ex2.approx.ftz.f32(scores[j], lo)
                            txl.ptx.ex2.approx.ftz.f32(scores[j + 1], hi)
                    packed_p = txl.alloc_local((32,), "uint32")
                    for pair in range(32):
                        txl.ptx[cvt_pack](packed_p[pair], scores[pair * 2 + 1], scores[pair * 2])
                    txl.ptx.bar.sync(txl.uint32(BAR_COMPUTE[0]), txl.uint32(BAR_COMPUTE[1]))
                    for rep in range(2):
                        _tmem_store16(
                            packed_p,
                            rep * 16,
                            txl.cuda.get_tmem_addr(
                                tcol, row_group, txl.int32(t_s + wg * 16 + rep * 32)
                            ),
                        )
                    txl.ptx["tcgen05.wait::st.sync.aligned"]()
                    txl.ptx.fence.proxy.async_.shared__cta()
                    txl.ptx.bar.sync(txl.uint32(BAR_COMPUTE[0]), txl.uint32(BAR_COMPUTE[1]))
                    with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                        lse_pipe.empty.arrive(lse_cons.stage)
                    if not is_d192:
                        with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                            s_pipe.empty.arrive(s_cons.stage, remote=0, pred=True)
                        s_cons.advance()
                    lse_cons.advance()

                    sum_pipe.full.wait(sum_cons.stage, sum_cons.phase)
                    dp_pipe.full.wait(dp_cons.stage, dp_cons.phase)
                    exchange = txl.alloc_local((16,), "uint32")
                    for rep in range(2):
                        dp_values = txl.alloc_local((32,), "float32")
                        _tmem_load32(
                            dp_values,
                            0,
                            txl.cuda.get_tmem_addr(
                                tcol, row_group, txl.int32(t_dp + wg * 32 + rep * 64)
                            ),
                        )
                        txl.ptx["tcgen05.wait::ld.sync.aligned"]()
                        txl.ptx.bar.sync(txl.uint32(BAR_COMPUTE[0]), txl.uint32(BAR_COMPUTE[1]))
                        for pair in range(16):
                            j = pair * 2
                            qcol = wg * txl.int32(32) + txl.int32(rep * 64 + pair * 2)
                            sum0 = txl.local_scalar("float32")
                            sum1 = txl.local_scalar("float32")
                            txl.ptx.ld.shared_.v2.b32(
                                sum0, sum1, arena.ptr_to([off_ssum + qcol * txl.int32(4)])
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
                            txl.assign(dp_values[j], lo)
                            txl.assign(dp_values[j + 1], hi)
                        packed_ds = txl.alloc_local((16,), "uint32")
                        for pair in range(16):
                            txl.ptx[cvt_pack](
                                packed_ds[pair], dp_values[pair * 2 + 1], dp_values[pair * 2]
                            )
                        if rep == 0:
                            ds_pipe.empty.wait(0, ds_prod.phase)
                        _tmem_store16(
                            packed_ds,
                            0,
                            txl.cuda.get_tmem_addr(
                                tcol, row_group, txl.int32(t_ds + wg * 16 + rep * 32)
                            ),
                        )
                        # CTA 0 keeps score stage 0 and exchanges stage 1; CTA 1
                        # keeps stage 1 and exchanges stage 0.  Both compute
                        # warpgroups contribute their 32-column quarter to the
                        # selected 64-column stage.
                        direct_half = txl.int32(rep) == rank
                        with txl.If(direct_half):
                            with txl.Then():
                                for group in range(4):
                                    qcol = wg * txl.int32(32) + txl.int32(rep * 64 + group * 8)
                                    base = group * 4
                                    txl.ptx.st.shared.v4.b32(
                                        arena.ptr_to([_tile_byte(off_sds, crow, qcol)]),
                                        packed_ds[base],
                                        packed_ds[base + 1],
                                        packed_ds[base + 2],
                                        packed_ds[base + 3],
                                    )
                            with txl.Else():
                                if is_d192:
                                    for value_index in range(16):
                                        txl.assign(exchange[value_index], packed_ds[value_index])
                                else:
                                    for group in range(4):
                                        base = group * 4
                                        txl.ptx.st.shared.v4.b32(
                                            arena.ptr_to(
                                                [
                                                    _tile_byte(
                                                        off_sdsx,
                                                        crow,
                                                        wg * txl.int32(32) + txl.int32(group * 8),
                                                    )
                                                ]
                                            ),
                                            packed_ds[base],
                                            packed_ds[base + 1],
                                            packed_ds[base + 2],
                                            packed_ds[base + 3],
                                        )
                    txl.ptx["tcgen05.wait::st.sync.aligned"]()
                    with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                        dp_pipe.empty.arrive(dp_cons.stage, remote=0, pred=True)
                    dp_cons.advance()
                    if is_d192:
                        # sdSx aliases the dQ reduction arena.  The four reducer
                        # warps release this epoch only after all async reductions
                        # using the prior dQ slice have drained.
                        dq_empty_cluster.wait(0, ds_prod.phase)
                        for group in range(4):
                            base = group * 4
                            txl.ptx.st.shared.v4.b32(
                                arena.ptr_to(
                                    [
                                        _tile_byte(
                                            off_sdsx,
                                            crow,
                                            wg * txl.int32(32) + txl.int32(group * 8),
                                        )
                                    ]
                                ),
                                exchange[base],
                                exchange[base + 1],
                                exchange[base + 2],
                                exchange[base + 3],
                            )
                    txl.ptx.fence.proxy.async_.shared__cta()
                    txl.ptx.bar.sync(txl.uint32(BAR_COMPUTE[0]), txl.uint32(BAR_COMPUTE[1]))
                    with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                        sum_pipe.empty.arrive(sum_cons.stage)
                    sum_cons.advance()
                    if is_d192:
                        with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                            ds_pipe.full.arrive(ds_prod.stage, remote=0)
                        ds_prod.advance()
                        exchange_owner = ctid == txl.int32(0)
                    else:
                        exchange_owner = (
                            (wg != rank)
                            & (warp_in_wg == txl.int32(0))
                            & (txl.lane_id() == txl.int32(0))
                        )
                    with txl.If(exchange_owner), txl.Then():
                        peer = txl.int32(1) - rank
                        remote_bar = txl.local_scalar("uint32")
                        remote_dst = txl.local_scalar("uint32")
                        txl.ptx.mapa.shared__cluster.u32(
                            remote_bar,
                            txl.cuda.cvta_generic_to_shared(ds_full_cluster.ptr_to([0])),
                            txl.cast(peer, "uint32"),
                        )
                        txl.ptx.mapa.shared__cluster.u32(
                            remote_dst,
                            txl.cuda.cvta_generic_to_shared(
                                arena.ptr_to([off_sds + rank * txl.int32(16384)])
                            ),
                            txl.cast(peer, "uint32"),
                        )
                        txl.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(
                            remote_bar, txl.uint32(16384)
                        )
                        txl.ptx[BULK_S2C](
                            remote_dst, arena.ptr_to([off_sdsx]), txl.uint32(16384), remote_bar
                        )
                    txl.assign(group_edge, group_edge + txl.int32(1))

            compute_group(full_n, txl.int32(0), False)
            compute_group(partial_n, full_n, True)

            with txl.If(work), txl.Then():
                if not is_d192:
                    with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
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
                    physical_n = task * txl.int32(2) + rank
                    kv_bh = txl.cast(batch_idx, "int64") * txl.int64(kv_heads) + txl.cast(
                        kv_head, "int64"
                    )
                    kv_sem_index = (
                        kv_bh * txl.int64((tasks * 2) * 2)
                        + txl.cast(physical_n, "int64") * txl.int64(2)
                        + txl.cast(wg, "int64")
                    )
                    if deterministic_kv:
                        _wait_eq_i32(
                            kv_semaphore,
                            kv_sem_index,
                            head % txl.int32(qhead_per_kvhead),
                            crow == txl.int32(0),
                        )
                        txl.ptx.bar.sync(txl.cast(txl.int32(1) + wg, "uint32"), txl.uint32(128))
                    for epi_stage in range(epi_stages):
                        values = txl.alloc_local((epi_cols,), "float32")
                        for load_stage in range(epi_cols // 16):
                            _tmem_load(
                                values,
                                load_stage * 16,
                                txl.cuda.get_tmem_addr(
                                    tcol,
                                    row_group,
                                    txl.int32(
                                        tmem_offset
                                        + wg * wg_cols
                                        + epi_stage * epi_cols
                                        + load_stage * 16
                                    ),
                                ),
                                16,
                            )
                        txl.ptx["tcgen05.wait::ld.sync.aligned"]()
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
                                txl.assign(values[pair * 2], lo)
                                txl.assign(values[pair * 2 + 1], hi)

                        if direct_dkv:
                            packed = txl.alloc_local((epi_cols // 2,), "uint32")
                            for pair in range(epi_cols // 2):
                                txl.ptx[cvt_pack](
                                    packed[pair], values[pair * 2 + 1], values[pair * 2]
                                )
                            if direct_raw_dkv:
                                seq = task * txl.int32(256) + rank * txl.int32(128) + crow
                                with txl.If(seq < k_length), txl.Then():
                                    destination = (
                                        txl.cast(k_offset + seq, "int64") * txl.int64(kv_heads)
                                        + txl.cast(kv_head, "int64")
                                    ) * txl.int64(output_dim) + txl.cast(
                                        wg * txl.int32(wg_cols) + txl.int32(epi_stage * epi_cols),
                                        "int64",
                                    )
                                    target = dk_output if is_dk else dv_output
                                    for group in range(epi_cols // 8):
                                        base = group * 4
                                        txl.ptx.st.global_.v4.b32(
                                            target.ptr_to([destination + txl.int64(group * 8)]),
                                            packed[base],
                                            packed[base + 1],
                                            packed[base + 2],
                                            packed[base + 3],
                                        )
                                txl.ptx.bar.warp.sync(txl.uint32(0xFFFFFFFF))
                            else:
                                for group in range(epi_cols // 8):
                                    base = group * 4
                                    txl.ptx.st.shared.v4.b32(
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
                                txl.ptx.fence.proxy.async_.shared__cta()
                                txl.ptx.bar.sync(
                                    txl.cast(txl.int32(1) + wg, "uint32"), txl.uint32(128)
                                )
                                with (
                                    txl.If(
                                        (warp_in_wg == txl.int32(0))
                                        & (txl.cuda.elect_sync() != txl.uint32(0))
                                    ),
                                    txl.Then(),
                                ):
                                    target_map = dk_map if is_dk else dv_map
                                    txl.ptx[TMA_S2G](
                                        txl.address_of(target_map),
                                        wg * txl.int32(wg_cols) + txl.int32(epi_stage * epi_cols),
                                        task * txl.int32(256) + rank * txl.int32(128),
                                        kv_head,
                                        batch_idx,
                                        arena.ptr_to([epi_base + wg * txl.int32(epi_region_bytes)]),
                                        TMA_CACHE,
                                    )
                        else:
                            for vec in range(epi_cols // 4):
                                base = vec * 4
                                txl.ptx.st.shared.v4.b32(
                                    arena.ptr_to(
                                        [
                                            epi_base
                                            + wg * txl.int32(epi_region_bytes)
                                            + txl.int32(vec * 2048)
                                            + crow * txl.int32(16)
                                        ]
                                    ),
                                    values[base],
                                    values[base + 1],
                                    values[base + 2],
                                    values[base + 3],
                                )
                            txl.ptx.fence.proxy.async_.shared__cta()
                            txl.ptx.bar.sync(txl.cast(txl.int32(1) + wg, "uint32"), txl.uint32(128))
                            with (
                                txl.If(
                                    (warp_in_wg == txl.int32(0))
                                    & (txl.cuda.elect_sync() != txl.uint32(0))
                                ),
                                txl.Then(),
                            ):
                                output_base = dk_base if is_dk else dv_base
                                dst_head = (
                                    txl.cast(kv_head, "int64") * txl.int64(k_storage_rows)
                                    + txl.cast(k_padded_offset, "int64")
                                    if varlen
                                    else kv_bh * txl.int64(k128)
                                )
                                dst = (
                                    txl.int64(output_base)
                                    + dst_head * txl.int64(output_dim)
                                    + txl.cast(physical_n, "int64") * txl.int64(128 * output_dim)
                                    + txl.cast(
                                        wg * txl.int32(wg_cols) + txl.int32(epi_stage * epi_cols),
                                        "int64",
                                    )
                                    * txl.int64(128)
                                )
                                txl.ptx[
                                    "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                                ](
                                    workspace.ptr_to([dst]),
                                    arena.ptr_to([epi_base + wg * txl.int32(epi_region_bytes)]),
                                    txl.uint32(128 * epi_cols * 4),
                                )

                        if not direct_raw_dkv:
                            with txl.If(warp_in_wg == txl.int32(0)), txl.Then():
                                txl.ptx.cp.async_.bulk.commit_group()
                                txl.ptx.cp.async_.bulk.wait_group.read(0)
                                txl.ptx.bar.arrive(
                                    txl.cast(txl.int32(1) + wg, "uint32"), txl.uint32(160)
                                )
                            txl.ptx.fence.proxy.async_.shared__cta()
                            txl.ptx.bar.sync(txl.cast(txl.int32(1) + wg, "uint32"), txl.uint32(160))
                    if deterministic_kv:
                        _release_inc_i32(kv_semaphore, kv_sem_index, crow == txl.int32(0))
                    with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                        dkdv_pipe.empty.arrive(dkv_cons.stage, remote=0, pred=True)
                    dkv_cons.advance()

            with txl.If((work == txl.bool(False)) & direct_dkv):
                with txl.Then():
                    seq = task * txl.int32(256) + rank * txl.int32(128) + crow
                    with txl.If(seq < k_length), txl.Then():
                        token = k_offset + seq if varlen else batch_idx * txl.int32(seqlen_kv) + seq
                        with txl.If(wg == txl.int32(0)):
                            with txl.Then():
                                destination = (
                                    txl.cast(token, "int64") * txl.int64(kv_heads)
                                    + txl.cast(kv_head, "int64")
                                ) * txl.int64(head_dim)
                                for vec in range(head_dim // 8):
                                    txl.ptx.st.global_.v4.b32(
                                        dk_output.ptr_to([destination + txl.int64(vec * 8)]),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                    )
                            with txl.Else():
                                destination = (
                                    txl.cast(token, "int64") * txl.int64(kv_heads)
                                    + txl.cast(kv_head, "int64")
                                ) * txl.int64(head_dim_v)
                                for vec in range(head_dim_v // 8):
                                    txl.ptx.st.global_.v4.b32(
                                        dv_output.ptr_to([destination + txl.int64(vec * 8)]),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                    )
            txl.ptx.bar.arrive(txl.uint32(BAR_TMEM[0]), txl.uint32(BAR_TMEM[1]))

        with r_reduce:
            txl.ptx.barrier.sync(txl.uint32(BAR_TMEM[0]), txl.uint32(BAR_TMEM[1]))
            tcol = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(tcol, tmem_mailbox.ptr_to([0]))
            rtid = txl.tid_in_role()
            warp_in_reduce = txl.warp_id_in_role()
            row_group = warp_in_reduce * txl.int32(32)
            dq_cons = txl.PipelineState(1, phase=0)
            store_stage = txl.local_scalar("int32", init=txl.int32(0))
            bh = txl.cast(batch_idx, "int64") * txl.int64(heads) + txl.cast(head, "int64")
            dq_head_row = (
                txl.cast(head, "int64") * txl.int64(q_storage_rows)
                + txl.cast(q_padded_offset, "int64")
                if varlen
                else bh * txl.int64(q128)
            )

            def reduce_group(group_count, group_offset, is_partial):
                group_edge = txl.local_scalar("int32", init=txl.int32(0))
                with txl.While(group_edge < group_count):
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
                    q_safe = txl.Select(
                        q_block < q_block_count, q_block, q_block_count - txl.int32(1)
                    )
                    q_live = q_block < q_block_count
                    values = txl.alloc_local((head_dim // 2,), "float32")
                    if is_d192:
                        for load_stage in range(3):
                            _tmem_load32(
                                values,
                                load_stage * 32,
                                txl.cuda.get_tmem_addr(
                                    tcol, row_group, txl.int32(t_dq + load_stage * 32)
                                ),
                            )
                    else:
                        # D128's cooperative accumulator presents the two halves
                        # in the source's rotated order.
                        _tmem_load32(
                            values, 32, txl.cuda.get_tmem_addr(tcol, row_group, txl.int32(t_dq))
                        )
                        _tmem_load32(
                            values, 0, txl.cuda.get_tmem_addr(tcol, row_group, txl.int32(t_dq + 32))
                        )
                    txl.ptx["tcgen05.wait::ld.sync.aligned"]()
                    txl.ptx.bar.warp.sync(txl.uint32(0xFFFFFFFF))
                    with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                        dq_pipe.empty.arrive(dq_cons.stage, remote=0, pred=True)
                    dq_cons.advance()

                    lock = txl.local_scalar("int32", init=txl.int32(0))
                    if deterministic:
                        txl.assign(
                            lock,
                            _load_i32(
                                dq_write_order if is_partial else dq_write_order_full,
                                (partial_begin if is_partial else full_begin) + group_edge,
                            ),
                        )
                    sem_index = (bh * txl.int64(q_blocks) + txl.cast(q_safe, "int64")) * txl.int64(
                        2
                    ) + txl.cast(rank, "int64")
                    for stage in range(dq_slices):
                        smem_stage = store_stage
                        reg_start = (
                            stage * dq_ncol
                            if is_d192
                            else ((stage + dq_slices // 2) % dq_slices) * dq_ncol
                        )
                        for vec in range(dq_ncol // 4):
                            base = reg_start + vec * 4
                            txl.ptx.st.shared.v4.b32(
                                arena.ptr_to(
                                    [
                                        off_sdq
                                        + smem_stage * txl.int32(dq_stage_bytes)
                                        + txl.int32(vec * 2048)
                                        + rtid * txl.int32(16)
                                    ]
                                ),
                                values[base],
                                values[base + 1],
                                values[base + 2],
                                values[base + 3],
                            )
                        txl.ptx.fence.proxy.async_.shared__cta()
                        if deterministic and stage == 0:
                            with txl.If(q_live), txl.Then():
                                _wait_eq_i32(dq_semaphore, sem_index, lock, rtid == txl.int32(0))
                        txl.ptx.bar.sync(txl.uint32(BAR_REDUCE[0]), txl.uint32(BAR_REDUCE[1]))
                        with txl.If((warp_in_reduce == txl.int32(0)) & q_live), txl.Then():
                            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                                dst = (
                                    txl.int64(dq_base)
                                    + dq_head_row * txl.int64(head_dim)
                                    + txl.cast(q_safe, "int64") * txl.int64(128 * head_dim)
                                    + txl.cast(
                                        rank * txl.int32(head_dim // 2)
                                        + txl.int32(stage * dq_ncol),
                                        "int64",
                                    )
                                    * txl.int64(128)
                                )
                                txl.ptx[
                                    "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                                ](
                                    workspace.ptr_to([dst]),
                                    arena.ptr_to(
                                        [off_sdq + smem_stage * txl.int32(dq_stage_bytes)]
                                    ),
                                    txl.uint32(dq_stage_bytes),
                                )
                            txl.ptx.cp.async_.bulk.commit_group()
                            txl.ptx.cp.async_.bulk.wait_group.read(dq_smem_stages - 1)
                        txl.ptx.bar.sync(txl.uint32(BAR_REDUCE[0]), txl.uint32(BAR_REDUCE[1]))
                        txl.assign(
                            store_stage,
                            txl.Select(
                                store_stage == txl.int32(dq_smem_stages - 1),
                                txl.int32(0),
                                store_stage + txl.int32(1),
                            ),
                        )
                    if is_d192:
                        # All four slices must be globally consumed before sdQacc
                        # can be reused as the next edge's 16 KiB exchange tile.
                        with txl.If(warp_in_reduce == txl.int32(0)), txl.Then():
                            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                                if deterministic:
                                    txl.ptx.cp.async_.bulk.wait_group(0)
                                else:
                                    txl.ptx.cp.async_.bulk.wait_group.read(0)
                        txl.ptx.bar.sync(txl.uint32(BAR_REDUCE[0]), txl.uint32(BAR_REDUCE[1]))
                        with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                            dq_empty_cluster.arrive(0)
                    if deterministic:
                        with txl.If(warp_in_reduce == txl.int32(0)), txl.Then():
                            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                                txl.ptx.cp.async_.bulk.wait_group.read(0)
                        txl.ptx.bar.sync(txl.uint32(BAR_REDUCE[0]), txl.uint32(BAR_REDUCE[1]))
                        with txl.If(q_live), txl.Then():
                            _release_inc_i32(dq_semaphore, sem_index, rtid == txl.int32(0))
                    txl.assign(group_edge, group_edge + txl.int32(1))

            reduce_group(full_n, txl.int32(0), False)
            reduce_group(partial_n, full_n, True)
            with txl.If(work), txl.Then():
                with txl.If(warp_in_reduce == txl.int32(0)), txl.Then():
                    txl.ptx.cp.async_.bulk.wait_group.read(0)
                txl.ptx.bar.sync(txl.uint32(BAR_REDUCE[0]), txl.uint32(BAR_REDUCE[1]))
            txl.ptx.cp.async_.bulk.wait_group.read(0)
            txl.ptx.bar.arrive(txl.uint32(BAR_TMEM[0]), txl.uint32(BAR_TMEM[1]))

    return bwd.func
