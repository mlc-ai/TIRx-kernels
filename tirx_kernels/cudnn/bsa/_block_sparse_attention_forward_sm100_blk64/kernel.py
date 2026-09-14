# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM100 blk64 block-sparse attention forward device programs.

Upstream sources:
``python/cudnn/block_sparse_attention/csrc/fwd/sm100_blk64/bsa_fwd_sm100.py`` and
``python/cudnn/block_sparse_attention/csrc/fwd/sm100_blk64/bsa_fwd_combine.py``.
"""

import math

import tirx_kernels.tirx_lite as txl

from . import source_kernel

HEAD_DIM = 128
QUERY_TILE = 64
SPARSE_BLOCK = 64
LOG2_E = math.log2(math.e)
LN_2 = math.log(2.0)


def _load_bf16(buffer, index):
    bits = txl.local_scalar("uint16")
    txl.ptx.ld.global_.b16(bits, buffer.ptr_to([index]))
    return txl.cast(txl.reinterpret("bfloat16", bits), "float32")


def _load_i32(buffer, index):
    value = txl.local_scalar("int32")
    txl.ptx.ld.global_.s32(value, buffer.ptr_to([index]))
    return value


def _store_bf16(buffer, index, value):
    bits = txl.local_scalar("uint16")
    txl.ptx.cvt.rn.bf16.f32(bits, value)
    txl.ptx.st.global_.b16(buffer.ptr_to([index]), bits)


def _exp2(value):
    out = txl.local_scalar("float32")
    txl.ptx.ex2.approx.ftz.f32(out, value)
    return out


def _log2(value):
    out = txl.local_scalar("float32")
    txl.ptx.lg2.approx.ftz.f32(out, value)
    return out


def _shfl_xor_f32(value, lane_xor):
    out = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.bfly.b32(
        out,
        txl.reinterpret("uint32", value),
        txl.uint32(lane_xor),
        txl.uint32(31),
        txl.uint32(0xFFFFFFFF),
    )
    return txl.reinterpret("float32", out)


def _shfl_xor_i32(value, lane_xor):
    out = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.bfly.b32(
        out,
        txl.reinterpret("uint32", value),
        txl.uint32(lane_xor),
        txl.uint32(31),
        txl.uint32(0xFFFFFFFF),
    )
    return txl.reinterpret("int32", out)


def _warp_sum(value):
    for lane_xor in (16, 8, 4, 2, 1):
        value = value + _shfl_xor_f32(value, lane_xor)
    return value


def _resolve_splits(value):
    if value == "auto":
        return 2
    return int(value)


def make_forward_kernel(**config):
    batch = int(config["batch"])
    num_heads = int(config["num_q_heads"])
    if int(config["num_kv_heads"]) != num_heads:
        raise NotImplementedError(
            "this scalar bring-up kernel indexes K/V by the Q head and has no "
            "grouped head map; the warp-specialized producer is the live path"
        )
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])
    max_blocks = int(config["kv_blocks"])
    num_splits = _resolve_splits(config["kv_splits"])
    has_block_sizes = bool(config["has_block_sizes"])
    has_block_nums = config["block_count_mode"] != "fixed"
    use_clc = bool(config["use_clc"])
    q_blocks = (seqlen_q + QUERY_TILE - 1) // QUERY_TILE
    grid = (q_blocks, num_heads if use_clc else num_heads * num_splits, batch)

    @txl.kernel(warps=16, arch="sm_100a", min_blocks_per_sm=1, grid=grid)
    def forward(
        q: txl.gptr[txl.bf16],
        k: txl.gptr[txl.bf16],
        v: txl.gptr[txl.bf16],
        out_bf16: txl.gptr[txl.bf16],
        out_f32: txl.gptr[txl.f32],
        lse: txl.gptr[txl.f32],
        block_index: txl.gptr[txl.i32],
        block_sizes: txl.gptr[txl.i32],
        block_nums: txl.gptr[txl.i32],
        split_offsets: txl.gptr[txl.i32],
        softmax_scale: txl.f32,
    ):
        # Numerically direct transcription of the sparse block/split contract.
        # The warp-specialized TMA/TMEM producer in ``source_kernel.py`` is the
        # path this module actually launches; this loop is kept only as a
        # readable statement of the same contract.
        if use_clc:
            q_block, head, batch_idx = txl.cta_id_in_cluster([q_blocks, num_heads, batch])
            split = txl.int32(0)
        else:
            q_block, head_split, batch_idx = txl.cta_id([q_blocks, num_heads * num_splits, batch])
            split = head_split // num_heads
            head = head_split - split * num_heads

        warp = txl.warp_id()
        lane = txl.thread_id() & 31
        count_index = (batch_idx * num_heads + head) * q_blocks + q_block
        raw_count = txl.local_scalar("int32")
        split_start = txl.local_scalar("int32", init=0)
        if num_splits > 1:
            split_base = count_index * (num_splits + 1)
            txl.assign(split_start, _load_i32(split_offsets, split_base + split))
            split_end = _load_i32(split_offsets, split_base + split + 1)
            txl.assign(raw_count, split_end - split_start)
        elif has_block_nums:
            txl.assign(raw_count, _load_i32(block_nums, count_index))
        else:
            txl.assign(raw_count, txl.int32(max_blocks))

        with txl.unroll(4) as row_group:
            row = warp + row_group * 16
            query = q_block * QUERY_TILE + row
            with txl.If(query < seqlen_q), txl.Then():
                row_max = txl.local_scalar("float32", init=txl.float32(-float("inf")))
                row_sum = txl.local_scalar("float32", init=txl.float32(0.0))
                accum = txl.alloc_local((4,), "float32")
                with txl.unroll(4) as j:
                    txl.assign(accum[j], txl.float32(0.0))

                block_slot = txl.local_scalar("int32", init=0)
                with txl.While(block_slot < raw_count):
                    meta_index = count_index * max_blocks + split_start + block_slot
                    sparse_id = _load_i32(block_index, meta_index)
                    block_size = txl.local_scalar("int32")
                    if has_block_sizes:
                        txl.assign(block_size, _load_i32(block_sizes, sparse_id))
                    else:
                        txl.assign(block_size, txl.int32(SPARSE_BLOCK))
                    token_in_block = txl.local_scalar("int32", init=0)
                    with txl.While(token_in_block < block_size):
                        token = sparse_id * SPARSE_BLOCK + token_in_block
                        with txl.If(token < seqlen_kv), txl.Then():
                            dot = txl.local_scalar("float32", init=txl.float32(0.0))
                            with txl.unroll(4) as j:
                                dim = lane + j * 32
                                q_idx = (
                                    (batch_idx * num_heads + head) * seqlen_q + query
                                ) * 128 + dim
                                kv_idx = (
                                    (batch_idx * num_heads + head) * seqlen_kv + token
                                ) * 128 + dim
                                txl.assign(dot, dot + _load_bf16(q, q_idx) * _load_bf16(k, kv_idx))
                            txl.assign(dot, _warp_sum(dot))
                            score = dot * softmax_scale
                            new_max = txl.local_scalar("float32")
                            txl.ptx.max.f32(new_max, row_max, score)
                            old_weight = _exp2((row_max - new_max) * txl.float32(LOG2_E))
                            new_weight = _exp2((score - new_max) * txl.float32(LOG2_E))
                            with txl.unroll(4) as j:
                                dim = lane + j * 32
                                value_idx = (
                                    (batch_idx * num_heads + head) * seqlen_kv + token
                                ) * 128 + dim
                                txl.assign(
                                    accum[j],
                                    accum[j] * old_weight + new_weight * _load_bf16(v, value_idx),
                                )
                            txl.assign(row_sum, row_sum * old_weight + new_weight)
                            txl.assign(row_max, new_max)
                        txl.assign(token_in_block, token_in_block + 1)
                    txl.assign(block_slot, block_slot + 1)

                valid_sum = row_sum > txl.float32(0.0)
                inv_sum = txl.if_then_else(valid_sum, txl.float32(1.0) / row_sum, txl.float32(0.0))
                out_head = split * num_heads + head if num_splits > 1 else head
                with txl.unroll(4) as j:
                    dim = lane + j * 32
                    out_idx = (
                        (batch_idx * (num_heads * num_splits) + out_head) * seqlen_q + query
                    ) * 128 + dim
                    if num_splits > 1:
                        txl.ptx.st.global_.f32(out_f32.ptr_to([out_idx]), accum[j] * inv_sum)
                    else:
                        _store_bf16(out_bf16, out_idx, accum[j] * inv_sum)
                with txl.If(lane == 0), txl.Then():
                    lse_idx = (batch_idx * (num_heads * num_splits) + out_head) * seqlen_q + query
                    lse_value = txl.if_then_else(
                        valid_sum,
                        row_max + _log2(row_sum) * txl.float32(LN_2),
                        txl.float32(-float("inf")),
                    )
                    txl.ptx.st.global_.f32(lse.ptr_to([lse_idx]), lse_value)

    return forward


def make_combine_kernel(**config):
    batch = int(config["batch"])
    num_heads = int(config["num_q_heads"])
    seqlen_q = int(config["seqlen_q"])
    num_splits = _resolve_splits(config["kv_splits"])
    max_splits = 1 << (num_splits - 1).bit_length()
    lse_split_storage = max(8, max_splits)
    lse_slots_per_thread = lse_split_storage // 8
    lse_bytes = lse_split_storage * 16 * 4
    max_split_offset = (lse_bytes + 127) // 128 * 128
    o_ring_offset = (max_split_offset + 16 * 4 + 127) // 128 * 128
    smem_bytes = o_ring_offset + 4 * 16 * 64 * 4
    row_tiles = (seqlen_q * num_heads + 15) // 16

    @txl.kernel(warps=4, arch="sm_100a", min_blocks_per_sm=1, grid=(row_tiles, 2, batch))
    def combine(
        out_partial: txl.gptr[txl.f32],
        lse_partial: txl.gptr[txl.f32],
        out: txl.gptr[txl.bf16],
        lse: txl.gptr[txl.f32],
    ):
        row_tile, dim_tile, batch_idx = txl.cta_id([row_tiles, 2, batch])
        tid = txl.thread_id()
        raw = txl.alloc_buffer((smem_bytes,), txl.u8, scope="shared.dyn", align=1024)

        def view(shape, dtype, byte_offset):
            return txl.decl_buffer(
                shape, dtype, data=raw.data, byte_offset=byte_offset, scope="shared.dyn", align=128
            )

        s_lse = view((lse_split_storage * 16,), txl.f32, 0)
        s_max_split = view((16,), txl.i32, max_split_offset)
        s_o = view((4 * 16 * 64,), txl.f32, o_ring_offset)

        def lse_smem_index(split, row):
            linear = split * 16 + row
            return txl.bitwise_xor(linear, txl.bitwise_and(txl.shift_right(linear, 4), 15))

        def partial_lse_index(split, flat_row):
            query = flat_row - (flat_row // seqlen_q) * seqlen_q
            head = flat_row // seqlen_q
            return (
                batch_idx * (num_splits * num_heads) + split * num_heads + head
            ) * seqlen_q + query

        def load_o_stage(split, stage, row0, row1, col):
            for row in (row0, row1):
                flat_row = row_tile * 16 + row
                with txl.If(flat_row < seqlen_q * num_heads), txl.Then():
                    lidx = partial_lse_index(split, flat_row)
                    txl.ptx["cp.async.cg.shared.global"](
                        s_o.ptr_to([stage * 1024 + row * 64 + col]),
                        out_partial.ptr_to([lidx * 128 + dim_tile * 64 + col]),
                        16,
                        16,
                    )

        # LSE staging map: contiguous row lanes and eight split lanes.  Slots
        # beyond the logical split count remain physical -inf entries because
        # the source always allocates at least eight split rows.
        lse_row = tid & 15
        lse_split0 = tid >> 4
        lse_flat_row = row_tile * 16 + lse_row
        with txl.unroll(lse_slots_per_thread) as slot:
            split = lse_split0 + slot * 8
            dst = s_lse.ptr_to([lse_smem_index(split, lse_row)])
            with txl.If((lse_flat_row < seqlen_q * num_heads) & (split < num_splits)):
                with txl.Then():
                    txl.ptx["cp.async.ca.shared.global"](
                        dst, lse_partial.ptr_to([partial_lse_index(split, lse_flat_row)]), 4, 4
                    )
                with txl.Else():
                    txl.ptx.st.shared.u32(dst, txl.uint32(0xFF800000))
        txl.ptx.cp.async_.commit_group()

        row0 = tid >> 4
        row1 = row0 + 8
        col = (tid & 15) * 4
        for stage in range(3):
            if stage < num_splits:
                load_o_stage(stage, stage, row0, row1, col)
            txl.ptx.cp.async_.commit_group()

        txl.ptx.cp.async_.wait_group(3)
        txl.ptx.bar.sync(txl.uint32(0))

        # Transposed LSE readback: eight lanes cooperate on one row, and each
        # lane owns split indices separated by eight.
        stats_row = tid >> 3
        stats_split0 = tid & 7
        lse_regs = txl.alloc_local((lse_slots_per_thread,), "float32")
        with txl.unroll(lse_slots_per_thread) as slot:
            txl.ptx.ld.shared.f32(
                lse_regs[slot], s_lse.ptr_to([lse_smem_index(stats_split0 + slot * 8, stats_row)])
            )

        local_max = txl.local_scalar("float32", init=txl.float32(-float("inf")))
        local_last = txl.local_scalar("int32", init=txl.int32(-1))
        with txl.unroll(lse_slots_per_thread) as slot:
            txl.ptx.max.f32(local_max, local_max, lse_regs[slot])
            with txl.If(lse_regs[slot] != txl.float32(-float("inf"))), txl.Then():
                txl.assign(local_last, stats_split0 + slot * 8)
        for lane_xor in (4, 2, 1):
            other_max = _shfl_xor_f32(local_max, lane_xor)
            txl.ptx.max.f32(local_max, local_max, other_max)
            other_last = _shfl_xor_i32(local_last, lane_xor)
            txl.ptx.max.s32(local_last, local_last, other_last)

        safe_max = txl.if_then_else(
            local_max == txl.float32(-float("inf")), txl.float32(0.0), local_max
        )
        local_sum = txl.local_scalar("float32", init=txl.float32(0.0))
        with txl.unroll(lse_slots_per_thread) as slot:
            txl.assign(lse_regs[slot], _exp2((lse_regs[slot] - safe_max) * txl.float32(LOG2_E)))
            txl.assign(local_sum, local_sum + lse_regs[slot])
        for lane_xor in (4, 2, 1):
            txl.assign(local_sum, local_sum + _shfl_xor_f32(local_sum, lane_xor))

        inv_sum = txl.local_scalar("float32", init=txl.float32(0.0))
        with txl.If((local_last >= 0) & (local_sum > txl.float32(0.0))), txl.Then():
            txl.ptx.rcp.rn.f32(inv_sum, local_sum)
        with txl.unroll(lse_slots_per_thread) as slot:
            txl.ptx.mul.f32(lse_regs[slot], lse_regs[slot], inv_sum)
            txl.ptx.st.shared.f32(
                s_lse.ptr_to([lse_smem_index(stats_split0 + slot * 8, stats_row)]), lse_regs[slot]
            )
        with txl.If(stats_split0 == 0), txl.Then():
            txl.ptx.st.shared.u32(
                s_max_split.ptr_to([stats_row]), txl.reinterpret("uint32", local_last)
            )
            flat_row = row_tile * 16 + stats_row
            with txl.If((dim_tile == 0) & (flat_row < seqlen_q * num_heads)), txl.Then():
                final_lse = txl.local_scalar("float32", init=txl.float32(-float("inf")))
                with txl.If(local_last >= 0), txl.Then():
                    lg = _log2(local_sum)
                    txl.ptx.fma.rn.f32(final_lse, lg, txl.float32(LN_2), local_max)
                query = flat_row - (flat_row // seqlen_q) * seqlen_q
                head = flat_row // seqlen_q
                txl.ptx.st.global_.f32(
                    lse.ptr_to([(batch_idx * num_heads + head) * seqlen_q + query]), final_lse
                )

        txl.ptx.bar.sync(txl.uint32(0))

        max0_bits = txl.local_scalar("uint32")
        max1_bits = txl.local_scalar("uint32")
        txl.ptx.ld.shared.u32(max0_bits, s_max_split.ptr_to([row0]))
        txl.ptx.ld.shared.u32(max1_bits, s_max_split.ptr_to([row1]))
        max_split = txl.local_scalar("int32")
        txl.ptx.max.s32(
            max_split, txl.reinterpret("int32", max0_bits), txl.reinterpret("int32", max1_bits)
        )
        acc0 = txl.alloc_local((4,), "float32")
        acc1 = txl.alloc_local((4,), "float32")
        with txl.unroll(4) as j:
            txl.assign(acc0[j], txl.float32(0.0))
            txl.assign(acc1[j], txl.float32(0.0))
        load_stage = txl.local_scalar("int32", init=3)
        compute_stage = txl.local_scalar("int32", init=0)
        split = txl.local_scalar("int32", init=0)
        with txl.While(split <= max_split):
            with txl.If(split + 3 <= max_split), txl.Then():
                load_o_stage(split + 3, load_stage, row0, row1, col)
            txl.ptx.cp.async_.commit_group()
            weight0 = txl.local_scalar("float32")
            weight1 = txl.local_scalar("float32")
            txl.ptx.ld.shared.f32(weight0, s_lse.ptr_to([lse_smem_index(split, row0)]))
            txl.ptx.ld.shared.f32(weight1, s_lse.ptr_to([lse_smem_index(split, row1)]))
            txl.assign(load_stage, txl.bitwise_and(load_stage + 1, 3))
            txl.ptx.cp.async_.wait_group(3)
            part0 = txl.alloc_local((4,), "float32")
            part1 = txl.alloc_local((4,), "float32")
            part0_lo = txl.local_scalar("uint64")
            part0_hi = txl.local_scalar("uint64")
            part1_lo = txl.local_scalar("uint64")
            part1_hi = txl.local_scalar("uint64")
            txl.ptx["ld.shared.v2.b64"](
                part0_lo, part0_hi, s_o.ptr_to([compute_stage * 1024 + row0 * 64 + col])
            )
            txl.ptx.mov.b64(part0[0], part0[1], part0_lo)
            txl.ptx.mov.b64(part0[2], part0[3], part0_hi)
            txl.ptx["ld.shared.v2.b64"](
                part1_lo, part1_hi, s_o.ptr_to([compute_stage * 1024 + row1 * 64 + col])
            )
            txl.ptx.mov.b64(part1[0], part1[1], part1_lo)
            txl.ptx.mov.b64(part1[2], part1[3], part1_hi)
            txl.assign(compute_stage, txl.bitwise_and(compute_stage + 1, 3))
            for row_acc, part, weight, row in (
                (acc0, part0, weight0, row0),
                (acc1, part1, weight1, row1),
            ):
                with (
                    txl.If(
                        (row_tile * 16 + row < seqlen_q * num_heads) & (weight > txl.float32(0.0))
                    ),
                    txl.Then(),
                ):
                    with txl.unroll(2) as pair:
                        weighted = txl.alloc_local((2,), "float32")
                        packed_weight = txl.local_scalar("uint64")
                        packed_part = txl.local_scalar("uint64")
                        packed_result = txl.local_scalar("uint64")
                        packed_acc = txl.local_scalar("uint64")
                        txl.ptx.mov.b64(packed_weight, weight, weight)
                        txl.ptx.mov.b64(packed_part, part[pair * 2], part[pair * 2 + 1])
                        txl.ptx.mul.rn.f32x2(packed_result, packed_part, packed_weight)
                        txl.ptx.mov.b64(weighted[0], weighted[1], packed_result)
                        txl.ptx.mov.b64(packed_acc, row_acc[pair * 2], row_acc[pair * 2 + 1])
                        txl.ptx.mov.b64(packed_part, weighted[0], weighted[1])
                        txl.ptx.add.rn.f32x2(packed_result, packed_acc, packed_part)
                        txl.ptx.mov.b64(row_acc[pair * 2], row_acc[pair * 2 + 1], packed_result)
            txl.assign(split, split + 1)

        for row_acc, row in ((acc0, row0), (acc1, row1)):
            flat_row = row_tile * 16 + row
            with txl.If(flat_row < seqlen_q * num_heads), txl.Then():
                packed0 = txl.local_scalar("uint32")
                packed1 = txl.local_scalar("uint32")
                txl.ptx.cvt.rn.bf16x2.f32(packed0, row_acc[1], row_acc[0])
                txl.ptx.cvt.rn.bf16x2.f32(packed1, row_acc[3], row_acc[2])
                query = flat_row - (flat_row // seqlen_q) * seqlen_q
                head = flat_row // seqlen_q
                out_index = (
                    ((batch_idx * num_heads + head) * seqlen_q + query) * 128 + dim_tile * 64 + col
                )
                txl.ptx.st.global_.v2.b32(out.ptr_to([out_index]), packed0, packed1)

    return combine


def get_kernel(**config):
    forward = source_kernel.make_forward_kernel(**config).func
    if _resolve_splits(config["kv_splits"]) == 1:
        return [forward]
    return [forward, make_combine_kernel(**config).func]
