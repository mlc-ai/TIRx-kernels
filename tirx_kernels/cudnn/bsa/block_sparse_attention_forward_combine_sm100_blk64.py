# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
# Pradeep Ramani, Tri Dao.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

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
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SPDX-License-Identifier: Apache-2.0 AND MIT AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM100 blk64 split-attention forward combine.

Upstream source:
``python/cudnn/block_sparse_attention/csrc/fwd/sm100_blk64/bsa_fwd_combine.py``.

The public tensor contract is the direct CuTeDSL combine ABI: fp32 partial
outputs and log-sum-exp values are reduced into BSHD bf16 output and BSH fp32
log-sum-exp output.  The surrounding BSA wrapper's final BSHD-to-BHSD copies
are deliberately outside this kernel.
"""

from functools import lru_cache

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "cudnn_sm100_bsa_forward_combine_blk64",
    "category": "cudnn",
    "compute_capability": 10,
}

_HEAD_DIM = 128
_TILE_M = 16
_K_BLOCK = 64
_STAGES = 4
_LOG2_E = 1.4426950408889634
_LN_2 = 0.6931471805599453


def _config(label, *, batch, num_heads, seqlen_q, kv_splits, seed, data_pattern="mixed"):
    return {
        "label": label,
        "batch": batch,
        "num_heads": num_heads,
        "seqlen_q": seqlen_q,
        "kv_splits": kv_splits,
        "seed": seed,
        "data_pattern": data_pattern,
    }


CONFIGS = [
    _config("b1_h1_sq1_s2", batch=1, num_heads=1, seqlen_q=1, kv_splits=2, seed=8601),
    _config("b1_h1_sq15_s2", batch=1, num_heads=1, seqlen_q=15, kv_splits=2, seed=8602),
    _config(
        "b1_h1_sq16_s2_all_dead",
        batch=1,
        num_heads=1,
        seqlen_q=16,
        kv_splits=2,
        seed=8603,
        data_pattern="all_dead",
    ),
    _config("b1_h1_sq17_s3", batch=1, num_heads=1, seqlen_q=17, kv_splits=3, seed=8604),
    _config("b2_h3_sq65_s4", batch=2, num_heads=3, seqlen_q=65, kv_splits=4, seed=8605),
    _config("b1_h4_sq1024_s2", batch=1, num_heads=4, seqlen_q=1024, kv_splits=2, seed=8606),
    _config("b1_h4_sq2048_s4", batch=1, num_heads=4, seqlen_q=2048, kv_splits=4, seed=8607),
    _config("b1_h8_sq2048_s8", batch=1, num_heads=8, seqlen_q=2048, kv_splits=8, seed=8608),
    _config("b1_h1_sq65_s256", batch=1, num_heads=1, seqlen_q=65, kv_splits=256, seed=8609),
]

BENCH_CONFIGS = [
    _config("b1_h4_sq1024_s2", batch=1, num_heads=4, seqlen_q=1024, kv_splits=2, seed=8701),
    _config("b1_h4_sq4097_s2_tail", batch=1, num_heads=4, seqlen_q=4097, kv_splits=2, seed=8702),
    _config("b1_h4_sq2048_s4", batch=1, num_heads=4, seqlen_q=2048, kv_splits=4, seed=8703),
    _config("b1_h8_sq2048_s8", batch=1, num_heads=8, seqlen_q=2048, kv_splits=8, seed=8704),
    _config("b1_h1_sq17_s3_guard", batch=1, num_heads=1, seqlen_q=17, kv_splits=3, seed=8705),
    _config("b1_h1_sq65_s256_guard", batch=1, num_heads=1, seqlen_q=65, kv_splits=256, seed=8706),
]


def _ceil_log2(value):
    if not 2 <= value <= 256:
        raise ValueError(f"kv_splits must be in [2, 256], got {value}")
    return (value - 1).bit_length()


def _fast_divmod(divisor):
    """Return CUTLASS FastDivmod's multiplier and two shifts."""
    if divisor < 1:
        raise ValueError(f"divisor must be positive, got {divisor}")
    if divisor == 1:
        return 1, 0, 0
    shift = max(divisor - 1, 1).bit_length()
    multiplier = ((1 << (32 + shift)) + divisor - 1) // divisor - (1 << 32)
    return multiplier, 1, shift - 1


def _shfl_bfly_f32(value, lane_mask):
    out = K.local_scalar("uint32")
    K.ptx.shfl_sync.bfly.b32(
        out, K.reinterpret("uint32", value), K.uint32(lane_mask), K.uint32(31), K.uint32(0xFFFFFFFF)
    )
    return K.reinterpret("float32", out)


def _shfl_bfly_i32(value, lane_mask):
    out = K.local_scalar("uint32")
    K.ptx.shfl_sync.bfly.b32(
        out, K.reinterpret("uint32", value), K.uint32(lane_mask), K.uint32(31), K.uint32(0xFFFFFFFF)
    )
    return K.reinterpret("int32", out)


@lru_cache(maxsize=8)
def _make_kernel(log_max_splits):
    if not 1 <= log_max_splits <= 8:
        raise ValueError(f"log_max_splits must be in [1, 8], got {log_max_splits}")

    physical_splits = max(8, 1 << log_max_splits)
    slots = physical_splits // 8
    lse_bytes = physical_splits * _TILE_M * 4
    max_split_offset = (lse_bytes + 127) // 128 * 128
    o_ring_offset = (max_split_offset + _TILE_M * 4 + 127) // 128 * 128
    smem_bytes = o_ring_offset + _STAGES * _TILE_M * _K_BLOCK * 4

    def kernel_body(
        o_partial: K.gptr[K.f32],
        lse_partial: K.gptr[K.f32],
        out: K.gptr[K.bf16],
        lse: K.gptr[K.f32],
        batch: K.i32,
        num_heads: K.i32,
        seqlen_q: K.i32,
        num_splits: K.i32,
        seqlen_div_mul: K.i32,
        seqlen_div_s1: K.i32,
        seqlen_div_s2: K.i32,
    ):
        # CUDA TRANSCRIPTION START
        row_tiles = K.ceildiv(seqlen_q * num_heads, _TILE_M)
        row_tile, dim_tile, batch_idx = K.cta_id([row_tiles, _HEAD_DIM // _K_BLOCK, batch])
        tid = K.thread_id()

        arena = K.alloc_buffer((smem_bytes,), K.u8, scope="shared.dyn", align=1024)

        def _view(shape, dtype, byte_offset):
            return K.decl_buffer(
                shape,
                dtype,
                data=arena.data,
                byte_offset=byte_offset,
                scope="shared.dyn",
                align=128,
            )

        s_lse = _view((physical_splits * _TILE_M,), K.f32, 0)
        s_max_valid = _view((_TILE_M,), K.i32, max_split_offset)
        s_o = _view((_STAGES * _TILE_M * _K_BLOCK,), K.f32, o_ring_offset)

        def _lse_index(split, row):
            linear = split * _TILE_M + row
            return K.bitwise_xor(linear, K.bitwise_and(K.shift_right(linear, 4), K.int32(15)))

        def _o_index(stage, row, col):
            return stage * (_TILE_M * _K_BLOCK) + row * _K_BLOCK + col

        def _decode_row(flat):
            q0 = K.local_scalar("uint32")
            K.ptx.mul.hi.u32(
                q0, K.reinterpret("uint32", flat), K.reinterpret("uint32", seqlen_div_mul)
            )
            delta = K.local_scalar("uint32", init=K.reinterpret("uint32", flat) - q0)
            mixed = K.local_scalar(
                "uint32", init=K.shift_right(delta, K.reinterpret("uint32", seqlen_div_s1)) + q0
            )
            head = K.local_scalar(
                "int32",
                init=K.reinterpret(
                    "int32", K.shift_right(mixed, K.reinterpret("uint32", seqlen_div_s2))
                ),
            )
            query = K.local_scalar("int32", init=flat - head * seqlen_q)
            return head, query

        def _partial_row_index(split, flat):
            head, query = _decode_row(flat)
            return (
                batch_idx * (num_splits * num_heads) + split * num_heads + head
            ) * seqlen_q + query

        row0 = K.local_scalar("int32", init=K.shift_right(tid, 4))
        row1 = K.local_scalar("int32", init=row0 + 8)
        col0 = K.local_scalar("int32", init=K.bitwise_and(tid, K.int32(15)) * 4)
        row_extent = K.local_scalar("int32", init=seqlen_q * num_heads)

        def _load_o_stage(split, stage):
            for row in (row0, row1):
                flat = K.local_scalar("int32", init=row_tile * _TILE_M + row)
                with K.If(flat < row_extent):
                    with K.Then():
                        partial_row = _partial_row_index(split, flat)
                        K.ptx["cp.async.cg.shared.global"](
                            s_o.ptr_to([_o_index(stage, row, col0)]),
                            o_partial.ptr_to(
                                [partial_row * _HEAD_DIM + dim_tile * _K_BLOCK + col0]
                            ),
                            16,
                            16,
                        )

        # LSE stage: 16 contiguous row lanes crossed with eight split lanes.
        lse_row = K.local_scalar("int32", init=K.bitwise_and(tid, K.int32(15)))
        lse_split0 = K.local_scalar("int32", init=K.shift_right(tid, 4))
        lse_flat = K.local_scalar("int32", init=row_tile * _TILE_M + lse_row)
        with K.unroll(slots) as slot:
            split = K.local_scalar("int32", init=lse_split0 + slot * 8)
            dst = s_lse.ptr_to([_lse_index(split, lse_row)])
            with K.If((lse_flat < row_extent) & (split < num_splits)):
                with K.Then():
                    K.ptx["cp.async.ca.shared.global"](
                        dst, lse_partial.ptr_to([_partial_row_index(split, lse_flat)]), 4, 4
                    )
                with K.Else():
                    K.ptx.st.shared.b32(dst, K.uint32(0xFF800000))
        K.ptx.cp.async_.commit_group()

        # Prime three generations of the four-stage partial-O ring.
        for stage in range(_STAGES - 1):
            with K.If(K.int32(stage) < num_splits):
                with K.Then():
                    _load_o_stage(K.int32(stage), K.int32(stage))
            K.ptx.cp.async_.commit_group()

        K.ptx.cp.async_.wait_group(_STAGES - 1)
        K.ptx.bar.sync(K.uint32(0))

        # Transposed LSE read: eight lanes cooperate on each stats row.
        stats_row = K.local_scalar("int32", init=K.shift_right(tid, 3))
        stats_split0 = K.local_scalar("int32", init=K.bitwise_and(tid, K.int32(7)))
        lse_regs = K.alloc_local((slots,), "float32")
        with K.unroll(slots) as slot:
            K.ptx.ld.shared.b32(
                lse_regs[slot], s_lse.ptr_to([_lse_index(stats_split0 + slot * 8, stats_row)])
            )

        # The source code generator uses one max.NaN site per pair of local
        # slots.  Its association is immaterial for finite/-inf LSE values, but
        # the instruction family and exact site count are preserved.
        if slots == 1:
            local_max = lse_regs[0]
        elif slots == 2:
            local_max = K.local_scalar("float32")
            K.ptx["max.NaN.f32"](local_max, lse_regs[0], lse_regs[1])
        else:
            local_max = K.local_scalar("float32")
            K.ptx["max.NaN.f32"](local_max, lse_regs[0], lse_regs[1], lse_regs[2])
            for pair in range(1, slots // 2):
                nxt = K.local_scalar("float32")
                if pair == slots // 2 - 1:
                    K.ptx["max.NaN.f32"](nxt, local_max, lse_regs[2 * pair + 1])
                else:
                    K.ptx["max.NaN.f32"](
                        nxt, local_max, lse_regs[2 * pair + 1], lse_regs[2 * pair + 2]
                    )
                local_max = nxt

        peer4 = _shfl_bfly_f32(local_max, 4)
        le4 = K.local_scalar("uint32")
        K.ptx.setp.le.f32(le4, local_max, peer4)
        nan4 = K.local_scalar("uint32")
        K.ptx.setp.nan.f32(nan4, peer4, peer4)
        selected4 = K.local_scalar("float32")
        K.ptx.selp.f32(selected4, peer4, local_max, K.ptx.pred(le4))
        max4 = K.local_scalar("float32")
        K.ptx.selp.f32(max4, peer4, selected4, K.ptx.pred(nan4))
        peer2 = _shfl_bfly_f32(max4, 2)
        max2 = K.local_scalar("float32")
        K.ptx.max.f32(max2, max4, peer2)
        peer1 = _shfl_bfly_f32(max2, 1)
        lse_max = K.local_scalar("float32")
        K.ptx.max.f32(lse_max, max2, peer1)

        local_last = K.local_scalar("int32", init=K.int32(-1))
        with K.unroll(slots) as slot:
            live = K.local_scalar("uint32")
            K.ptx.setp.neu.f32(live, lse_regs[slot], K.float32(-float("inf")))
            coordinate = stats_split0 + slot * 8
            selected = K.local_scalar("int32")
            K.ptx.selp.b32(selected, coordinate, local_last, K.ptx.pred(live))
            K.assign(local_last, selected)
        for lane_mask in (4, 2, 1):
            peer = _shfl_bfly_i32(local_last, lane_mask)
            reduced = K.local_scalar("int32")
            K.ptx.max.s32(reduced, local_last, peer)
            K.assign(local_last, reduced)

        neg_inf = K.reinterpret("float32", K.uint32(0xFF800000))
        is_neg_inf = K.local_scalar("uint32")
        K.ptx.setp.eq.f32(is_neg_inf, lse_max, neg_inf)
        safe_max = K.local_scalar("float32")
        K.ptx.selp.f32(safe_max, K.float32(0.0), lse_max, K.ptx.pred(is_neg_inf))
        safe_max_log2 = K.local_scalar("float32")
        K.ptx.mul.f32(safe_max_log2, safe_max, K.float32(_LOG2_E))

        local_sum = K.local_scalar("float32", init=K.float32(0.0))
        with K.unroll(slots) as slot:
            scaled = K.local_scalar("float32")
            K.ptx.mul.f32(scaled, lse_regs[slot], K.float32(_LOG2_E))
            delta = K.local_scalar("float32")
            K.ptx.sub.f32(delta, scaled, safe_max_log2)
            K.ptx.ex2.approx.ftz.f32(lse_regs[slot], delta)
            summed = K.local_scalar("float32")
            K.ptx.add.f32(summed, local_sum, lse_regs[slot])
            K.assign(local_sum, summed)
        for lane_mask in (4, 2, 1):
            peer = _shfl_bfly_f32(local_sum, lane_mask)
            summed = K.local_scalar("float32")
            K.ptx.add.f32(summed, local_sum, peer)
            K.assign(local_sum, summed)

        log_sum = K.local_scalar("float32")
        K.ptx.lg2.approx.ftz.f32(log_sum, local_sum)
        final_lse = K.local_scalar("float32")
        K.ptx.fma.rn.f32(final_lse, log_sum, K.float32(_LN_2), lse_max)
        bad_sum = K.local_scalar("uint32")
        K.ptx.setp.equ.f32(bad_sum, local_sum, K.float32(0.0))
        reciprocal = K.local_scalar("float32")
        K.ptx.rcp.rn.f32(reciprocal, local_sum)
        inv_sum = K.local_scalar("float32")
        K.ptx.selp.f32(inv_sum, K.float32(0.0), reciprocal, K.ptx.pred(bad_sum))

        if slots == 1:
            normalized = K.local_scalar("float32")
            K.ptx.mul.f32(normalized, lse_regs[0], inv_sum)
            K.assign(lse_regs[0], normalized)
        else:
            for pair in range(slots // 2):
                packed_scale = K.local_scalar("uint64")
                packed_inv = K.local_scalar("uint64")
                packed_result = K.local_scalar("uint64")
                K.ptx.mov.b64(packed_scale, lse_regs[2 * pair], lse_regs[2 * pair + 1])
                K.ptx.mov.b64(packed_inv, inv_sum, inv_sum)
                K.ptx.mul.f32x2(packed_result, packed_scale, packed_inv)
                K.ptx.mov.b64(lse_regs[2 * pair], lse_regs[2 * pair + 1], packed_result)

        with K.unroll(slots) as slot:
            K.ptx.st.shared.b32(
                s_lse.ptr_to([_lse_index(stats_split0 + slot * 8, stats_row)]),
                K.reinterpret("uint32", lse_regs[slot]),
            )
        with K.If(stats_split0 == 0):
            with K.Then():
                K.ptx.st.shared.b32(
                    s_max_valid.ptr_to([stats_row]), K.reinterpret("uint32", local_last)
                )
                stats_flat = K.local_scalar("int32", init=row_tile * _TILE_M + stats_row)
                with K.If((dim_tile == 0) & (stats_flat < row_extent)):
                    with K.Then():
                        head, query = _decode_row(stats_flat)
                        out_index = (batch_idx * seqlen_q + query) * num_heads + head
                        K.ptx.st.global_.b32(
                            lse.ptr_to([out_index]), K.reinterpret("uint32", final_lse)
                        )

        K.ptx.bar.sync(K.uint32(0))

        max0_bits = K.local_scalar("uint32")
        max1_bits = K.local_scalar("uint32")
        K.ptx.ld.shared.b32(max0_bits, s_max_valid.ptr_to([row0]))
        K.ptx.ld.shared.b32(max1_bits, s_max_valid.ptr_to([row1]))
        thread_max = K.local_scalar("int32")
        K.ptx.max.s32(
            thread_max, K.reinterpret("int32", max0_bits), K.reinterpret("int32", max1_bits)
        )

        acc0 = K.alloc_local((4,), "float32")
        acc1 = K.alloc_local((4,), "float32")
        with K.unroll(4) as value:
            K.assign(acc0[value], K.float32(0.0))
            K.assign(acc1[value], K.float32(0.0))

        load_stage = K.local_scalar("int32", init=K.int32(_STAGES - 1))
        compute_stage = K.local_scalar("int32", init=K.int32(0))
        with K.serial(thread_max + 1, unroll=4) as split:
            weight0_bits = K.local_scalar("uint32")
            weight1_bits = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(weight0_bits, s_lse.ptr_to([_lse_index(split, row0)]))
            K.ptx.ld.shared.b32(weight1_bits, s_lse.ptr_to([_lse_index(split, row1)]))
            weight0 = K.reinterpret("float32", weight0_bits)
            weight1 = K.reinterpret("float32", weight1_bits)

            next_split = K.local_scalar("int32", init=split + (_STAGES - 1))
            with K.If(next_split <= thread_max):
                with K.Then():
                    _load_o_stage(next_split, load_stage)
            K.ptx.cp.async_.commit_group()
            K.assign(load_stage, K.bitwise_and(load_stage + 1, K.int32(_STAGES - 1)))
            K.ptx.cp.async_.wait_group(_STAGES - 1)

            part0 = K.alloc_local((4,), "float32")
            part1 = K.alloc_local((4,), "float32")
            p0_lo = K.local_scalar("uint64")
            p0_hi = K.local_scalar("uint64")
            p1_lo = K.local_scalar("uint64")
            p1_hi = K.local_scalar("uint64")
            K.ptx["ld.shared.v2.b64"](
                p0_lo, p0_hi, s_o.ptr_to([_o_index(compute_stage, row0, col0)])
            )
            K.ptx.mov.b64(part0[0], part0[1], p0_lo)
            K.ptx.mov.b64(part0[2], part0[3], p0_hi)
            K.ptx["ld.shared.v2.b64"](
                p1_lo, p1_hi, s_o.ptr_to([_o_index(compute_stage, row1, col0)])
            )
            K.ptx.mov.b64(part1[0], part1[1], p1_lo)
            K.ptx.mov.b64(part1[2], part1[3], p1_hi)
            K.assign(compute_stage, K.bitwise_and(compute_stage + 1, K.int32(_STAGES - 1)))

            for row, weight, part, accum in (
                (row0, weight0, part0, acc0),
                (row1, weight1, part1, acc1),
            ):
                flat = K.local_scalar("int32", init=row_tile * _TILE_M + row)
                with K.If((flat < row_extent) & (weight > K.float32(0.0))):
                    with K.Then():
                        for pair in range(2):
                            packed_weight = K.local_scalar("uint64")
                            packed_part = K.local_scalar("uint64")
                            packed_acc = K.local_scalar("uint64")
                            packed_product = K.local_scalar("uint64")
                            packed_sum = K.local_scalar("uint64")
                            K.ptx.mov.b64(packed_weight, weight, weight)
                            K.ptx.mov.b64(packed_part, part[2 * pair], part[2 * pair + 1])
                            K.ptx.mul.f32x2(packed_product, packed_part, packed_weight)
                            K.ptx.mov.b64(packed_acc, accum[2 * pair], accum[2 * pair + 1])
                            K.ptx.add.f32x2(packed_sum, packed_acc, packed_product)
                            K.ptx.mov.b64(accum[2 * pair], accum[2 * pair + 1], packed_sum)

        packed = [K.local_scalar("uint32") for _ in range(4)]
        K.ptx.cvt.rn.bf16x2.f32(packed[0], acc0[1], acc0[0])
        K.ptx.cvt.rn.bf16x2.f32(packed[1], acc0[3], acc0[2])
        K.ptx.cvt.rn.bf16x2.f32(packed[2], acc1[1], acc1[0])
        K.ptx.cvt.rn.bf16x2.f32(packed[3], acc1[3], acc1[2])

        for row, word0, word1 in ((row0, packed[0], packed[1]), (row1, packed[2], packed[3])):
            flat = K.local_scalar("int32", init=row_tile * _TILE_M + row)
            with K.If(flat < row_extent):
                with K.Then():
                    head, query = _decode_row(flat)
                    out_index = (
                        ((batch_idx * seqlen_q + query) * num_heads + head) * _HEAD_DIM
                        + dim_tile * _K_BLOCK
                        + col0
                    )
                    K.ptx.st.global_.v2.b32(out.ptr_to([out_index]), word0, word1)
        # CUDA TRANSCRIPTION END

    @K.kernel(
        warps=4,
        arch="sm_100a",
        grid=lambda p: [
            K.ceildiv(p["seqlen_q"] * p["num_heads"], _TILE_M),
            _HEAD_DIM // _K_BLOCK,
            p["batch"],
        ],
    )
    def combine(
        o_partial: K.gptr[K.f32],
        lse_partial: K.gptr[K.f32],
        out: K.gptr[K.bf16],
        lse: K.gptr[K.f32],
        batch: K.i32,
        num_heads: K.i32,
        seqlen_q: K.i32,
        num_splits: K.i32,
        seqlen_div_mul: K.i32,
        seqlen_div_s1: K.i32,
        seqlen_div_s2: K.i32,
    ):
        with K.attr({"tirx.required_block_size": 1}):
            kernel_body(
                o_partial,
                lse_partial,
                out,
                lse,
                batch,
                num_heads,
                seqlen_q,
                num_splits,
                seqlen_div_mul,
                seqlen_div_s1,
                seqlen_div_s2,
            )

    return combine


def get_kernel(*, batch, num_heads, seqlen_q, kv_splits, seed=0, data_pattern="mixed"):
    del batch, num_heads, seqlen_q, seed, data_pattern
    return _make_kernel(_ceil_log2(kv_splits)).func.with_attr("global_symbol", KERNEL_META["name"])


def _without_label(config):
    return {key: value for key, value in config.items() if key != "label"}


def _tirx_launch(executable, data):
    config = data["config"]
    output = data["tirx"]
    arguments = (
        data["o_partial"].reshape(-1),
        data["lse_partial"].reshape(-1),
        output["out"].reshape(-1),
        output["lse"].reshape(-1),
        int(config["batch"]),
        int(config["num_heads"]),
        int(config["seqlen_q"]),
        int(config["kv_splits"]),
        *_fast_divmod(int(config["seqlen_q"])),
    )

    def launch():
        executable(*arguments)

    launch._keep_alive = arguments
    return launch


def _source_views(data):
    config = data["config"]
    batch = int(config["batch"])
    num_heads = int(config["num_heads"])
    seqlen_q = int(config["seqlen_q"])
    num_splits = int(config["kv_splits"])
    o_partial = data["o_partial"].as_strided(
        (num_splits, batch, seqlen_q, num_heads, _HEAD_DIM),
        (
            num_heads * seqlen_q * _HEAD_DIM,
            seqlen_q * num_splits * num_heads * _HEAD_DIM,
            _HEAD_DIM,
            seqlen_q * _HEAD_DIM,
            1,
        ),
    )
    lse_partial = data["lse_partial"].as_strided(
        (num_splits, batch, seqlen_q, num_heads),
        (num_heads * seqlen_q, seqlen_q * num_splits * num_heads, 1, seqlen_q),
    )
    return o_partial, lse_partial


def _compile_reference(data):
    import torch

    from tirx_kernels.cudnn._reference import load_reference_module

    interface = load_reference_module("cudnn.block_sparse_attention._interface")
    source = load_reference_module(
        "cudnn.block_sparse_attention.csrc.fwd.sm100_blk64.bsa_fwd_combine"
    )
    config = data["config"]
    num_splits = int(config["kv_splits"])
    o_partial, lse_partial = _source_views(data)
    output = data["source"]
    op = source.BlockSparseAttnForwardCombine(
        dtype=interface.torch2cute_dtype_map[torch.bfloat16],
        head_dim=_HEAD_DIM,
        tile_m=_TILE_M,
        k_block_size=_K_BLOCK,
        log_max_splits=_ceil_log2(num_splits),
        num_threads=128,
        stages=_STAGES,
    )
    compile_args = (
        interface._to_cute_tensor_dynamic_compact_shape(
            o_partial, mode=(0, 1, 2, 3), stride_order=(1, 0, 3, 2, 4)
        ),
        interface._to_cute_tensor_dynamic_compact_shape(
            lse_partial,
            mode=(0, 1, 2, 3),
            assumed_align=4,
            leading_dim=2,
            stride_order=(1, 0, 3, 2),
        ),
        interface._to_cute_tensor_dynamic_compact_shape(
            output["out"], mode=(0, 1, 2), stride_order=(0, 1, 2, 3)
        ),
        interface._to_cute_tensor_dynamic_compact_shape(
            output["lse"], mode=(0, 1, 2), assumed_align=4, stride_order=(0, 1, 2)
        ),
        None,
        None,
        None,
        None,
        None,
        interface.cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
    )
    compiled = interface.cute.compile(op, *compile_args, options="--enable-tvm-ffi")
    current_stream = interface.cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    runtime_args = (
        o_partial,
        lse_partial,
        output["out"],
        output["lse"],
        None,
        None,
        None,
        None,
        None,
        current_stream,
    )

    def launch():
        compiled(*runtime_args)

    launch._keep_alive = (compiled, runtime_args, op, compile_args)
    return launch


def _oracle(data):
    import torch

    config = data["config"]
    batch = int(config["batch"])
    num_heads = int(config["num_heads"])
    seqlen_q = int(config["seqlen_q"])
    num_splits = int(config["kv_splits"])
    lse = (
        data["lse_partial"]
        .view(batch, num_splits, num_heads, seqlen_q)
        .permute(0, 3, 2, 1)
        .double()
    )
    partial = (
        data["o_partial"]
        .view(batch, num_splits, num_heads, seqlen_q, _HEAD_DIM)
        .permute(0, 3, 2, 1, 4)
        .double()
    )
    row_max = lse.amax(dim=-1)
    safe_max = torch.where(torch.isneginf(row_max), torch.zeros_like(row_max), row_max)
    weights = torch.exp(lse - safe_max.unsqueeze(-1))
    denominator = weights.sum(dim=-1)
    live = ~torch.isneginf(lse)
    partial = torch.where(live.unsqueeze(-1), partial, torch.zeros_like(partial))
    numerator = (partial * weights.unsqueeze(-1)).sum(dim=-2)
    out = torch.where(
        denominator.unsqueeze(-1) > 0,
        numerator / denominator.clamp_min(1e-300).unsqueeze(-1),
        torch.zeros_like(numerator),
    )
    final_lse = torch.where(
        denominator > 0,
        torch.log(denominator.clamp_min(1e-300)) + row_max,
        torch.full_like(row_max, float("-inf")),
    )
    return out.to(torch.bfloat16), final_lse.float()


def _validate_outputs(data, *, with_oracle=True):
    import torch

    actual = data["tirx"]
    expected = data["source"]
    actual_o_bits = actual["out"].view(torch.uint16)
    expected_o_bits = expected["out"].view(torch.uint16)
    actual_lse_bits = actual["lse"].view(torch.int32)
    expected_lse_bits = expected["lse"].view(torch.int32)
    o_exact = torch.equal(actual_o_bits, expected_o_bits)
    lse_exact = torch.equal(actual_lse_bits, expected_lse_bits)
    if not o_exact or not lse_exact:
        o_mismatch = int((actual_o_bits != expected_o_bits).sum().item())
        lse_mismatch = int((actual_lse_bits != expected_lse_bits).sum().item())
        o_diff = torch.nan_to_num(
            (actual["out"].float() - expected["out"].float()).abs(), nan=float("inf")
        )
        lse_diff = torch.nan_to_num((actual["lse"] - expected["lse"]).abs(), nan=float("inf"))
        raise AssertionError(
            "TIRx versus pinned source is not bitwise exact: "
            f"O mismatches={o_mismatch}, max_abs={float(o_diff.max().item())}; "
            f"LSE mismatches={lse_mismatch}, max_abs={float(lse_diff.max().item())}"
        )

    if with_oracle:
        oracle_o, oracle_lse = _oracle(data)
        torch.testing.assert_close(
            expected["out"].float(), oracle_o.float(), rtol=0.0, atol=0.0078125, equal_nan=False
        )
        finite = torch.isfinite(oracle_lse)
        if not torch.equal(torch.isneginf(expected["lse"]), torch.isneginf(oracle_lse)):
            raise AssertionError("source/oracle LSE -inf classification mismatch")
        torch.testing.assert_close(
            expected["lse"][finite], oracle_lse[finite], rtol=2e-6, atol=2e-6, equal_nan=False
        )
    return {"o_bitwise": o_exact, "lse_bitwise": lse_exact}


def prepare_data(**config):
    import torch

    config = _without_label(config)
    batch = int(config["batch"])
    num_heads = int(config["num_heads"])
    seqlen_q = int(config["seqlen_q"])
    num_splits = int(config["kv_splits"])
    seed = int(config.get("seed", 0))
    data_pattern = str(config.get("data_pattern", "mixed"))
    _ceil_log2(num_splits)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    logical_lse = (
        torch.randn(
            batch,
            num_splits,
            num_heads,
            seqlen_q,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 2.0
    )
    split = torch.arange(num_splits, device="cuda").view(1, -1, 1, 1)
    head = torch.arange(num_heads, device="cuda").view(1, 1, -1, 1)
    query = torch.arange(seqlen_q, device="cuda").view(1, 1, 1, -1)
    batch_index = torch.arange(batch, device="cuda").view(-1, 1, 1, 1)
    if data_pattern == "all_dead":
        dead = torch.ones_like(logical_lse, dtype=torch.bool)
    elif data_pattern == "mixed":
        dead = (split > 0) & (((3 * split + 5 * head + query + batch_index) % 11) == 0)
    else:
        raise ValueError(f"unknown data_pattern {data_pattern!r}")
    logical_lse = torch.where(dead, torch.full_like(logical_lse, float("-inf")), logical_lse)

    logical_o = torch.randn(
        batch,
        num_splits,
        num_heads,
        seqlen_q,
        _HEAD_DIM,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    logical_o = torch.where(dead.unsqueeze(-1), torch.full_like(logical_o, float("nan")), logical_o)
    o_partial = logical_o.reshape(batch, num_splits * num_heads, seqlen_q, _HEAD_DIM).contiguous()
    lse_partial = logical_lse.reshape(batch, num_splits * num_heads, seqlen_q).contiguous()

    def _outputs():
        return {
            "out": torch.full(
                (batch, seqlen_q, num_heads, _HEAD_DIM),
                float("nan"),
                dtype=torch.bfloat16,
                device="cuda",
            ),
            "lse": torch.full(
                (batch, seqlen_q, num_heads), float("nan"), dtype=torch.float32, device="cuda"
            ),
        }

    return {
        "config": config,
        "o_partial": o_partial,
        "lse_partial": lse_partial,
        "dead": dead,
        "tirx": _outputs(),
        "source": _outputs(),
    }


def run_test(**config):
    import torch

    from tirx_kernels.runner import compile_kernel

    kernel_config = _without_label(config)
    data = prepare_data(**kernel_config)
    tirx_launch = _tirx_launch(compile_kernel(get_kernel(**kernel_config)), data)
    source_launch = _compile_reference(data)
    tirx_launch()
    source_launch()
    torch.cuda.synchronize()
    return _validate_outputs(data)


def prepare_bench(**config):
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    kernel_config = _without_label(config)
    state = {"config": kernel_config, "executable": compile_kernel(get_kernel(**kernel_config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **config):
    from tirx_kernels.runner import bench, defer_gpu_interrupts, external_references_enabled

    with defer_gpu_interrupts():
        import torch

    kernel_config = _without_label({**prepared["config"], **config})
    with_source = external_references_enabled()
    gpu_state = prepared.get("gpu_state")
    if gpu_state is None:
        data = prepare_data(**kernel_config)
        gpu_state = {
            "data": data,
            "tirx_launch": _tirx_launch(prepared["executable"], data),
            "source_launch": None,
            "validated": False,
            "with_source": with_source,
        }
        prepared["gpu_state"] = gpu_state
    elif gpu_state["with_source"] != with_source:
        raise RuntimeError("reference timing mode changed within one prepared benchmark")

    if not gpu_state["validated"]:
        gpu_state["tirx_launch"]()
        torch.cuda.synchronize()
        if with_source:
            with defer_gpu_interrupts():
                gpu_state["source_launch"] = _compile_reference(gpu_state["data"])
                gpu_state["source_launch"]()
                torch.cuda.synchronize()
            _validate_outputs(gpu_state["data"], with_oracle=False)
        gpu_state["validated"] = True

    source_launch = gpu_state["source_launch"]
    references = {"cudnn_frontend": lambda: source_launch} if source_launch is not None else None
    return bench(
        {"tirx": gpu_state["tirx_launch"]},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **config):
    return prepare_bench(**config).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_gpu",
    "run_test",
]
