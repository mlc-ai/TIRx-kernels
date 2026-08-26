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

import tirx_kernels.kern as K

from . import source_kernel

HEAD_DIM = 128
QUERY_TILE = 64
SPARSE_BLOCK = 64
LOG2_E = math.log2(math.e)
LN_2 = math.log(2.0)


def _load_bf16(buffer, index):
    bits = K.local_scalar("uint16")
    K.ptx.ld.global_.b16(bits, buffer.ptr_to([index]))
    return K.cast(K.reinterpret("bfloat16", bits), "float32")


def _load_i32(buffer, index):
    value = K.local_scalar("int32")
    K.ptx.ld.global_.s32(value, buffer.ptr_to([index]))
    return value


def _store_bf16(buffer, index, value):
    bits = K.local_scalar("uint16")
    K.ptx.cvt.rn.bf16.f32(bits, value)
    K.ptx.st.global_.b16(buffer.ptr_to([index]), bits)


def _exp2(value):
    out = K.local_scalar("float32")
    K.ptx.ex2.approx.ftz.f32(out, value)
    return out


def _log2(value):
    out = K.local_scalar("float32")
    K.ptx.lg2.approx.ftz.f32(out, value)
    return out


def _shfl_xor_f32(value, lane_xor):
    out = K.local_scalar("uint32")
    K.ptx.shfl_sync.bfly.b32(
        out, K.reinterpret("uint32", value), K.uint32(lane_xor), K.uint32(31), K.uint32(0xFFFFFFFF)
    )
    return K.reinterpret("float32", out)


def _shfl_xor_i32(value, lane_xor):
    out = K.local_scalar("uint32")
    K.ptx.shfl_sync.bfly.b32(
        out, K.reinterpret("uint32", value), K.uint32(lane_xor), K.uint32(31), K.uint32(0xFFFFFFFF)
    )
    return K.reinterpret("int32", out)


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
    num_heads = int(config["num_heads"])
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])
    max_blocks = int(config["kv_blocks"])
    num_splits = _resolve_splits(config["kv_splits"])
    has_block_sizes = bool(config["has_block_sizes"])
    has_block_nums = config["block_count_mode"] != "fixed"
    use_clc = bool(config["use_clc"])
    q_blocks = (seqlen_q + QUERY_TILE - 1) // QUERY_TILE
    grid = (q_blocks, num_heads if use_clc else num_heads * num_splits, batch)

    @K.kernel(warps=16, arch="sm_100a", min_blocks_per_sm=1, grid=grid)
    def forward(
        q: K.gptr[K.bf16],
        k: K.gptr[K.bf16],
        v: K.gptr[K.bf16],
        out_bf16: K.gptr[K.bf16],
        out_f32: K.gptr[K.f32],
        lse: K.gptr[K.f32],
        block_index: K.gptr[K.i32],
        block_sizes: K.gptr[K.i32],
        block_nums: K.gptr[K.i32],
        split_offsets: K.gptr[K.i32],
        softmax_scale: K.f32,
    ):
        # Numerically direct bring-up transcription of the sparse block/split
        # contract.  The source-shaped TMA/TMEM datapath replaces this loop in
        # the implementation-review revision.
        if use_clc:
            q_block, head, batch_idx = K.cta_id_in_cluster([q_blocks, num_heads, batch])
            split = K.int32(0)
        else:
            q_block, head_split, batch_idx = K.cta_id([q_blocks, num_heads * num_splits, batch])
            split = head_split // num_heads
            head = head_split - split * num_heads

        warp = K.warp_id()
        lane = K.thread_id() & 31
        count_index = (batch_idx * num_heads + head) * q_blocks + q_block
        raw_count = K.local_scalar("int32")
        split_start = K.local_scalar("int32", init=0)
        if num_splits > 1:
            split_base = count_index * (num_splits + 1)
            K.assign(split_start, _load_i32(split_offsets, split_base + split))
            split_end = _load_i32(split_offsets, split_base + split + 1)
            K.assign(raw_count, split_end - split_start)
        elif has_block_nums:
            K.assign(raw_count, _load_i32(block_nums, count_index))
        else:
            K.assign(raw_count, K.int32(max_blocks))

        with K.unroll(4) as row_group:
            row = warp + row_group * 16
            query = q_block * QUERY_TILE + row
            with K.If(query < seqlen_q), K.Then():
                row_max = K.local_scalar("float32", init=K.float32(-float("inf")))
                row_sum = K.local_scalar("float32", init=K.float32(0.0))
                accum = K.alloc_local((4,), "float32")
                with K.unroll(4) as j:
                    K.assign(accum[j], K.float32(0.0))

                block_slot = K.local_scalar("int32", init=0)
                with K.While(block_slot < raw_count):
                    meta_index = count_index * max_blocks + split_start + block_slot
                    sparse_id = _load_i32(block_index, meta_index)
                    block_size = K.local_scalar("int32")
                    if has_block_sizes:
                        K.assign(block_size, _load_i32(block_sizes, sparse_id))
                    else:
                        K.assign(block_size, K.int32(SPARSE_BLOCK))
                    token_in_block = K.local_scalar("int32", init=0)
                    with K.While(token_in_block < block_size):
                        token = sparse_id * SPARSE_BLOCK + token_in_block
                        with K.If(token < seqlen_kv), K.Then():
                            dot = K.local_scalar("float32", init=K.float32(0.0))
                            with K.unroll(4) as j:
                                dim = lane + j * 32
                                q_idx = (
                                    (batch_idx * num_heads + head) * seqlen_q + query
                                ) * 128 + dim
                                kv_idx = (
                                    (batch_idx * num_heads + head) * seqlen_kv + token
                                ) * 128 + dim
                                K.assign(dot, dot + _load_bf16(q, q_idx) * _load_bf16(k, kv_idx))
                            K.assign(dot, _warp_sum(dot))
                            score = dot * softmax_scale
                            new_max = K.local_scalar("float32")
                            K.ptx.max.f32(new_max, row_max, score)
                            old_weight = _exp2((row_max - new_max) * K.float32(LOG2_E))
                            new_weight = _exp2((score - new_max) * K.float32(LOG2_E))
                            with K.unroll(4) as j:
                                dim = lane + j * 32
                                value_idx = (
                                    (batch_idx * num_heads + head) * seqlen_kv + token
                                ) * 128 + dim
                                K.assign(
                                    accum[j],
                                    accum[j] * old_weight + new_weight * _load_bf16(v, value_idx),
                                )
                            K.assign(row_sum, row_sum * old_weight + new_weight)
                            K.assign(row_max, new_max)
                        K.assign(token_in_block, token_in_block + 1)
                    K.assign(block_slot, block_slot + 1)

                valid_sum = row_sum > K.float32(0.0)
                inv_sum = K.if_then_else(valid_sum, K.float32(1.0) / row_sum, K.float32(0.0))
                out_head = split * num_heads + head if num_splits > 1 else head
                with K.unroll(4) as j:
                    dim = lane + j * 32
                    out_idx = (
                        (batch_idx * (num_heads * num_splits) + out_head) * seqlen_q + query
                    ) * 128 + dim
                    if num_splits > 1:
                        K.ptx.st.global_.f32(out_f32.ptr_to([out_idx]), accum[j] * inv_sum)
                    else:
                        _store_bf16(out_bf16, out_idx, accum[j] * inv_sum)
                with K.If(lane == 0), K.Then():
                    lse_idx = (batch_idx * (num_heads * num_splits) + out_head) * seqlen_q + query
                    lse_value = K.if_then_else(
                        valid_sum,
                        row_max + _log2(row_sum) * K.float32(LN_2),
                        K.float32(-float("inf")),
                    )
                    K.ptx.st.global_.f32(lse.ptr_to([lse_idx]), lse_value)

    return forward


def make_combine_kernel(**config):
    batch = int(config["batch"])
    num_heads = int(config["num_heads"])
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

    @K.kernel(warps=4, arch="sm_100a", min_blocks_per_sm=1, grid=(row_tiles, 2, batch))
    def combine(
        out_partial: K.gptr[K.f32],
        lse_partial: K.gptr[K.f32],
        out: K.gptr[K.bf16],
        lse: K.gptr[K.f32],
    ):
        row_tile, dim_tile, batch_idx = K.cta_id([row_tiles, 2, batch])
        tid = K.thread_id()
        raw = K.alloc_buffer((smem_bytes,), K.u8, scope="shared.dyn", align=1024)

        def view(shape, dtype, byte_offset):
            return K.decl_buffer(
                shape, dtype, data=raw.data, byte_offset=byte_offset, scope="shared.dyn", align=128
            )

        s_lse = view((lse_split_storage * 16,), K.f32, 0)
        s_max_split = view((16,), K.i32, max_split_offset)
        s_o = view((4 * 16 * 64,), K.f32, o_ring_offset)

        def lse_smem_index(split, row):
            linear = split * 16 + row
            return K.bitwise_xor(linear, K.bitwise_and(K.shift_right(linear, 4), 15))

        def partial_lse_index(split, flat_row):
            query = flat_row - (flat_row // seqlen_q) * seqlen_q
            head = flat_row // seqlen_q
            return (
                batch_idx * (num_splits * num_heads) + split * num_heads + head
            ) * seqlen_q + query

        def load_o_stage(split, stage, row0, row1, col):
            for row in (row0, row1):
                flat_row = row_tile * 16 + row
                with K.If(flat_row < seqlen_q * num_heads), K.Then():
                    lidx = partial_lse_index(split, flat_row)
                    K.ptx["cp.async.cg.shared.global"](
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
        with K.unroll(lse_slots_per_thread) as slot:
            split = lse_split0 + slot * 8
            dst = s_lse.ptr_to([lse_smem_index(split, lse_row)])
            with K.If((lse_flat_row < seqlen_q * num_heads) & (split < num_splits)):
                with K.Then():
                    K.ptx["cp.async.ca.shared.global"](
                        dst, lse_partial.ptr_to([partial_lse_index(split, lse_flat_row)]), 4, 4
                    )
                with K.Else():
                    K.ptx.st.shared.u32(dst, K.uint32(0xFF800000))
        K.ptx.cp.async_.commit_group()

        row0 = tid >> 4
        row1 = row0 + 8
        col = (tid & 15) * 4
        for stage in range(3):
            if stage < num_splits:
                load_o_stage(stage, stage, row0, row1, col)
            K.ptx.cp.async_.commit_group()

        K.ptx.cp.async_.wait_group(3)
        K.ptx.bar.sync(K.uint32(0))

        # Transposed LSE readback: eight lanes cooperate on one row, and each
        # lane owns split indices separated by eight.
        stats_row = tid >> 3
        stats_split0 = tid & 7
        lse_regs = K.alloc_local((lse_slots_per_thread,), "float32")
        with K.unroll(lse_slots_per_thread) as slot:
            K.ptx.ld.shared.f32(
                lse_regs[slot], s_lse.ptr_to([lse_smem_index(stats_split0 + slot * 8, stats_row)])
            )

        local_max = K.local_scalar("float32", init=K.float32(-float("inf")))
        local_last = K.local_scalar("int32", init=K.int32(-1))
        with K.unroll(lse_slots_per_thread) as slot:
            K.ptx.max.f32(local_max, local_max, lse_regs[slot])
            with K.If(lse_regs[slot] != K.float32(-float("inf"))), K.Then():
                K.assign(local_last, stats_split0 + slot * 8)
        for lane_xor in (4, 2, 1):
            other_max = _shfl_xor_f32(local_max, lane_xor)
            K.ptx.max.f32(local_max, local_max, other_max)
            other_last = _shfl_xor_i32(local_last, lane_xor)
            K.ptx.max.s32(local_last, local_last, other_last)

        safe_max = K.if_then_else(local_max == K.float32(-float("inf")), K.float32(0.0), local_max)
        local_sum = K.local_scalar("float32", init=K.float32(0.0))
        with K.unroll(lse_slots_per_thread) as slot:
            K.assign(lse_regs[slot], _exp2((lse_regs[slot] - safe_max) * K.float32(LOG2_E)))
            K.assign(local_sum, local_sum + lse_regs[slot])
        for lane_xor in (4, 2, 1):
            K.assign(local_sum, local_sum + _shfl_xor_f32(local_sum, lane_xor))

        inv_sum = K.local_scalar("float32", init=K.float32(0.0))
        with K.If((local_last >= 0) & (local_sum > K.float32(0.0))), K.Then():
            K.ptx.rcp.rn.f32(inv_sum, local_sum)
        with K.unroll(lse_slots_per_thread) as slot:
            K.ptx.mul.f32(lse_regs[slot], lse_regs[slot], inv_sum)
            K.ptx.st.shared.f32(
                s_lse.ptr_to([lse_smem_index(stats_split0 + slot * 8, stats_row)]), lse_regs[slot]
            )
        with K.If(stats_split0 == 0), K.Then():
            K.ptx.st.shared.u32(
                s_max_split.ptr_to([stats_row]), K.reinterpret("uint32", local_last)
            )
            flat_row = row_tile * 16 + stats_row
            with K.If((dim_tile == 0) & (flat_row < seqlen_q * num_heads)), K.Then():
                final_lse = K.local_scalar("float32", init=K.float32(-float("inf")))
                with K.If(local_last >= 0), K.Then():
                    lg = _log2(local_sum)
                    K.ptx.fma.rn.f32(final_lse, lg, K.float32(LN_2), local_max)
                query = flat_row - (flat_row // seqlen_q) * seqlen_q
                head = flat_row // seqlen_q
                K.ptx.st.global_.f32(
                    lse.ptr_to([(batch_idx * num_heads + head) * seqlen_q + query]), final_lse
                )

        K.ptx.bar.sync(K.uint32(0))

        max0_bits = K.local_scalar("uint32")
        max1_bits = K.local_scalar("uint32")
        K.ptx.ld.shared.u32(max0_bits, s_max_split.ptr_to([row0]))
        K.ptx.ld.shared.u32(max1_bits, s_max_split.ptr_to([row1]))
        max_split = K.local_scalar("int32")
        K.ptx.max.s32(
            max_split, K.reinterpret("int32", max0_bits), K.reinterpret("int32", max1_bits)
        )
        acc0 = K.alloc_local((4,), "float32")
        acc1 = K.alloc_local((4,), "float32")
        with K.unroll(4) as j:
            K.assign(acc0[j], K.float32(0.0))
            K.assign(acc1[j], K.float32(0.0))
        load_stage = K.local_scalar("int32", init=3)
        compute_stage = K.local_scalar("int32", init=0)
        split = K.local_scalar("int32", init=0)
        with K.While(split <= max_split):
            with K.If(split + 3 <= max_split), K.Then():
                load_o_stage(split + 3, load_stage, row0, row1, col)
            K.ptx.cp.async_.commit_group()
            weight0 = K.local_scalar("float32")
            weight1 = K.local_scalar("float32")
            K.ptx.ld.shared.f32(weight0, s_lse.ptr_to([lse_smem_index(split, row0)]))
            K.ptx.ld.shared.f32(weight1, s_lse.ptr_to([lse_smem_index(split, row1)]))
            K.assign(load_stage, K.bitwise_and(load_stage + 1, 3))
            K.ptx.cp.async_.wait_group(3)
            part0 = K.alloc_local((4,), "float32")
            part1 = K.alloc_local((4,), "float32")
            part0_lo = K.local_scalar("uint64")
            part0_hi = K.local_scalar("uint64")
            part1_lo = K.local_scalar("uint64")
            part1_hi = K.local_scalar("uint64")
            K.ptx["ld.shared.v2.b64"](
                part0_lo, part0_hi, s_o.ptr_to([compute_stage * 1024 + row0 * 64 + col])
            )
            K.ptx.mov.b64(part0[0], part0[1], part0_lo)
            K.ptx.mov.b64(part0[2], part0[3], part0_hi)
            K.ptx["ld.shared.v2.b64"](
                part1_lo, part1_hi, s_o.ptr_to([compute_stage * 1024 + row1 * 64 + col])
            )
            K.ptx.mov.b64(part1[0], part1[1], part1_lo)
            K.ptx.mov.b64(part1[2], part1[3], part1_hi)
            K.assign(compute_stage, K.bitwise_and(compute_stage + 1, 3))
            for row_acc, part, weight, row in (
                (acc0, part0, weight0, row0),
                (acc1, part1, weight1, row1),
            ):
                with (
                    K.If((row_tile * 16 + row < seqlen_q * num_heads) & (weight > K.float32(0.0))),
                    K.Then(),
                ):
                    with K.unroll(2) as pair:
                        weighted = K.alloc_local((2,), "float32")
                        packed_weight = K.local_scalar("uint64")
                        packed_part = K.local_scalar("uint64")
                        packed_result = K.local_scalar("uint64")
                        packed_acc = K.local_scalar("uint64")
                        K.ptx.mov.b64(packed_weight, weight, weight)
                        K.ptx.mov.b64(packed_part, part[pair * 2], part[pair * 2 + 1])
                        K.ptx.mul.rn.f32x2(packed_result, packed_part, packed_weight)
                        K.ptx.mov.b64(weighted[0], weighted[1], packed_result)
                        K.ptx.mov.b64(packed_acc, row_acc[pair * 2], row_acc[pair * 2 + 1])
                        K.ptx.mov.b64(packed_part, weighted[0], weighted[1])
                        K.ptx.add.rn.f32x2(packed_result, packed_acc, packed_part)
                        K.ptx.mov.b64(row_acc[pair * 2], row_acc[pair * 2 + 1], packed_result)
            K.assign(split, split + 1)

        for row_acc, row in ((acc0, row0), (acc1, row1)):
            flat_row = row_tile * 16 + row
            with K.If(flat_row < seqlen_q * num_heads), K.Then():
                packed0 = K.local_scalar("uint32")
                packed1 = K.local_scalar("uint32")
                K.ptx.cvt.rn.bf16x2.f32(packed0, row_acc[1], row_acc[0])
                K.ptx.cvt.rn.bf16x2.f32(packed1, row_acc[3], row_acc[2])
                query = flat_row - (flat_row // seqlen_q) * seqlen_q
                head = flat_row // seqlen_q
                out_index = (
                    ((batch_idx * num_heads + head) * seqlen_q + query) * 128 + dim_tile * 64 + col
                )
                K.ptx.st.global_.v2.b32(out.ptr_to([out_index]), packed0, packed1)

    return combine


def get_kernel(**config):
    forward = source_kernel.make_forward_kernel(**config).func
    if _resolve_splits(config["kv_splits"]) == 1:
        return [forward]
    return [forward, make_combine_kernel(**config).func]
