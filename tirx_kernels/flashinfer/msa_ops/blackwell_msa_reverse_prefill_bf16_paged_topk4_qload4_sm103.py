# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ cc6e8794c49bf66172627bdb9742fcb17d18b839), Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM103a BF16 paged TopK4 reverse-prefill producer/reducer pair.

Upstream sources (FlashInfer @ cc6e8794c49bf66172627bdb9742fcb17d18b839):

- csrc/blackwell_msa/sm103a/blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4.cu
- csrc/blackwell_msa/sm103a/blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_binding.cu
- csrc/blackwell_msa/sm103a/blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_const4_reduce.cu
- csrc/blackwell_msa/sm103a/blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_const4_reduce_binding.cu
- flashinfer/msa_ops/_blackwell_sm100.py
- flashinfer/msa_ops/_blackwell_sm100_reverse_plan.py
"""

import heapq
import math
import os
import random
from functools import lru_cache
from typing import Any

import tirx_kernels.tirx_lite as txl

KERNEL_META = {
    "name": "blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_sm103",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_103a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "cc6e8794c49bf66172627bdb9742fcb17d18b839",
            },
            "import": "flashinfer",
        },
    ),
}

CUDA_ARCH = "sm_103a"
TOTAL_Q = 12288
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 128
PAGE_SIZE = 128
MAX_PAGES = 64
NUM_PAGES = 192
TOPK = 4
PRODUCER_WARPS = 16
REDUCER_WARPS = 8
SMEM_TOTAL = 148480
TMEM_COLS = 512
_PTXAS_REGISTER_USAGE_LEVEL = "5"


def _config(
    label: str,
    *,
    schedule_seed: int,
    page_pattern: str,
    input_pattern: str = "random",
    return_softmax_lse: bool = False,
    return_temperature_lse: bool = False,
) -> dict[str, Any]:
    return {
        "label": label,
        "schedule_seed": schedule_seed,
        "page_pattern": page_pattern,
        "input_pattern": input_pattern,
        "return_softmax_lse": return_softmax_lse,
        "return_temperature_lse": return_temperature_lse,
    }


CONFIGS = [
    _config(
        "b3_q4096_kv8192_h8_hkv2_topk4_seed73_identity_nostats",
        schedule_seed=73,
        page_pattern="identity",
    ),
    _config(
        "b3_q4096_kv8192_h8_hkv2_topk4_seed101_identity_bothlse",
        schedule_seed=101,
        page_pattern="identity",
        return_softmax_lse=True,
        return_temperature_lse=True,
    ),
    _config(
        "b3_q4096_kv8192_h8_hkv2_topk4_seed73_permuted_lse",
        schedule_seed=73,
        page_pattern="permuted",
        return_softmax_lse=True,
    ),
    _config(
        "b3_q4096_kv8192_h8_hkv2_topk4_seed101_reused_negative_tlse",
        schedule_seed=101,
        page_pattern="reused_negative",
        return_temperature_lse=True,
    ),
    _config(
        "b3_q4096_kv8192_h8_hkv2_topk4_seed73_ftz_edges_bothlse",
        schedule_seed=73,
        page_pattern="identity",
        input_pattern="ftz_edges",
        return_softmax_lse=True,
        return_temperature_lse=True,
    ),
]

BENCH_CONFIGS = CONFIGS[:4]


def _without_label(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "label"}


def _validate_config(**config: Any) -> None:
    if int(config["schedule_seed"]) not in (73, 101):
        raise ValueError("this fixed route accepts schedule_seed 73 or 101")
    if config["page_pattern"] not in ("identity", "permuted", "reused_negative"):
        raise ValueError("unknown page pattern")
    if config["input_pattern"] not in ("random", "ftz_edges"):
        raise ValueError("unknown input pattern")


_TMA_INTERLEAVE_NONE = 0
_TMA_SWIZZLE_128B = 3
_TMA_L2_PROMOTION_NONE = 0
_TMA_OOB_FILL_NONE = 0


def _host_prelude(params):
    """Encode the source binding's 2-D gather4 Q map and folded 4-D paged KV maps."""

    def encode(tensor, rank, dims, strides_bytes, box):
        descriptor = txl.stack_alloca("tensormap", 1)
        txl.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            descriptor,
            "bfloat16",
            rank,
            tensor.data,
            *dims,
            *strides_bytes,
            *box,
            *([1] * rank),
            _TMA_INTERLEAVE_NONE,
            _TMA_SWIZZLE_128B,
            _TMA_L2_PROMOTION_NONE,
            _TMA_OOB_FILL_NONE,
        )
        return descriptor

    total_q = params["total_q"]
    num_q_heads = params["num_q_heads"]
    num_kv_heads = params["num_kv_heads"]
    q_map = encode(params["q"], 2, (HEAD_DIM, total_q * num_q_heads), (HEAD_DIM * 2,), (64, 1))
    page_heads = params["k"].shape[0] * num_kv_heads
    kv_dims = (64, PAGE_SIZE, HEAD_DIM // 64, page_heads)
    kv_strides = (HEAD_DIM * 2, 64 * 2, PAGE_SIZE * HEAD_DIM * 2)
    kv_box = (64, 64, 1, 1)
    k_map = encode(params["k"], 4, kv_dims, kv_strides, kv_box)
    v_map = encode(params["v"], 4, kv_dims, kv_strides, kv_box)
    return q_map, k_map, v_map


_MBAR_Q_FULL = 0
_MBAR_Q_EMPTY = 16
_MBAR_K_FULL = 32
_MBAR_V_FULL = 40
_MBAR_FP8_K_FULL = 48
_MBAR_FP8_V_FULL = 56
_MBAR_FP8_EMPTY = 64
_MBAR_S_FULL = 72
_MBAR_S_EMPTY = 88
_MBAR_P_FULL = 104
_MBAR_P_EMPTY = 120
_MBAR_O_FULL = 136
_MBAR_O_EMPTY = 152
_SMEM_TMEM_MAILBOX = 168
_SMEM_Q = 1024
_SMEM_K = 66560
_SMEM_V = 99328
_MBAR_INIT = (
    (_MBAR_Q_FULL, 4),
    (_MBAR_Q_FULL + 8, 4),
    (_MBAR_Q_EMPTY, 1),
    (_MBAR_Q_EMPTY + 8, 1),
    (_MBAR_K_FULL, 1),
    (_MBAR_V_FULL, 1),
    (_MBAR_FP8_K_FULL, 1),
    (_MBAR_FP8_V_FULL, 1),
    (_MBAR_FP8_EMPTY, 1),
    (_MBAR_S_FULL, 1),
    (_MBAR_S_FULL + 8, 1),
    (_MBAR_S_EMPTY, 128),
    (_MBAR_S_EMPTY + 8, 128),
    (_MBAR_P_FULL, 128),
    (_MBAR_P_FULL + 8, 128),
    (_MBAR_P_EMPTY, 1),
    (_MBAR_P_EMPTY + 8, 1),
    (_MBAR_O_FULL, 1),
    (_MBAR_O_FULL + 8, 1),
    (_MBAR_O_EMPTY, 128),
    (_MBAR_O_EMPTY + 8, 128),
)
_TMA_GATHER4_2D = (
    "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes"
)
_TMA_G2S_4D = "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes"
_MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
_TCGEN05_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
_TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
_TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
_TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
_TMEM_LD_X64 = "tcgen05.ld.sync.aligned.32x32b.x64.b32"
_TMEM_LD_X16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
_TMEM_ST_X32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"
_DESC_HI = 0x40004040
_QK_IDESC = 0x08200490
_PV_IDESC = 0x08210490
_QK_DELTAS = (0, 2, 4, 6, 1024, 1026, 1028, 1030)
_PV_TMEM_DELTAS = (32, 40, 48, 56, 0, 8, 16, 24)
_PV_SMEM_DELTAS = (512, 640, 768, 896, 0, 128, 256, 384)
_NEG_INF = float("-inf")
_FULL_MASK = 0xFFFFFFFF
_LN2 = 0.6931471805599453
_LOG2E = 1.4426950408889634


def _u32(value):
    return txl.uint32(value)


def _i32(value):
    return txl.int32(value)


def _i64(value):
    return txl.int64(value)


def _f32(value):
    return txl.float32(value)


def _pack2(lo, hi):
    packed = txl.local_scalar("uint64")
    txl.ptx.mov.b64(packed, lo, hi)
    return packed


def _mbar_wait(addr, phase):
    txl.cuda.mbarrier_wait(addr, phase)


def _mbar_arrive(addr):
    txl.ptx.mbarrier.arrive.release.cta.shared__cta.b64(addr)


def _mbar_expect_tx(addr, tx_bytes):
    txl.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(addr, _u32(tx_bytes))


def _ld_global_i32(buffer, index):
    value = txl.local_scalar("int32")
    txl.ptx.ld.global_.nc.b32(value, buffer.ptr_to([index]))
    return value


def _max_f32(lhs, rhs):
    value = txl.local_scalar("float32")
    txl.ptx.max.f32(value, lhs, rhs)
    return value


def _min_ftz_f32(lhs, rhs):
    value = txl.local_scalar("float32")
    txl.ptx.min.ftz.f32(value, lhs, rhs)
    return value


def _shfl_idx_i32(value, source_lane):
    out = txl.local_scalar("int32")
    txl.ptx.shfl_sync.idx.b32(out, value, source_lane, _u32(31), _u32(_FULL_MASK))
    return out


def _shfl_idx_f32(value, source_lane):
    out = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.idx.b32(
        out, txl.reinterpret("uint32", value), source_lane, _u32(31), _u32(_FULL_MASK)
    )
    return txl.reinterpret("float32", out)


def _shfl_xor_f32(value, lane_xor):
    out = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.bfly.b32(
        out, txl.reinterpret("uint32", value), _u32(lane_xor), _u32(31), _u32(_FULL_MASK)
    )
    return txl.reinterpret("float32", out)


def _packed_fma_inplace(values, base, scale_pair, bias_pair):
    result = txl.local_scalar("uint64")
    txl.ptx.fma.rn.ftz.f32x2(result, _pack2(values[base], values[base + 1]), scale_pair, bias_pair)
    txl.ptx.mov.b64(values[base], values[base + 1], result)


def _packed_mul_inplace(values, base, scale_pair):
    result = txl.local_scalar("uint64")
    txl.ptx.mul.rn.ftz.f32x2(result, _pack2(values[base], values[base + 1]), scale_pair)
    txl.ptx.mov.b64(values[base], values[base + 1], result)


def _prefix_mask(limit):
    mask = txl.local_scalar("uint32", init=_u32(0))
    with txl.If(limit <= _i32(0)):
        with txl.Then():
            txl.assign(mask, _u32(0))
        with txl.Else():
            with txl.If(limit >= _i32(32)):
                with txl.Then():
                    txl.assign(mask, _u32(_FULL_MASK))
                with txl.Else():
                    shifted = txl.local_scalar("uint32")
                    txl.ptx.shl.b32(shifted, _u32(1), txl.cast(limit, "uint32"))
                    txl.ptx.add.u32(mask, shifted, _u32(_FULL_MASK))
    return mask


def _mask_64(values, valid):
    mask_lo = _prefix_mask(valid)
    mask_hi = _prefix_mask(valid - _i32(32))
    for index in range(64):
        bit = txl.local_scalar("uint32")
        txl.ptx.and_.b32(bit, mask_lo if index < 32 else mask_hi, _u32(1 << (index % 32)))
        keep = txl.local_scalar("uint32")
        txl.ptx.setp.ne.b32(keep, bit, _u32(0))
        selected = txl.local_scalar("float32")
        txl.ptx.selp.f32(selected, values[index], _f32(_NEG_INF), txl.ptx.pred(keep))
        txl.assign(values[index], selected)


def _row_max_64(values):
    acc0 = txl.local_scalar("float32", init=_f32(_NEG_INF))
    acc1 = txl.local_scalar("float32", init=_f32(_NEG_INF))
    for half in range(2):
        for pair in range(16):
            pair_max = _max_f32(values[half * 32 + pair * 2], values[half * 32 + pair * 2 + 1])
            if pair % 2 == 0:
                txl.assign(acc0, _max_f32(acc0, pair_max))
            else:
                txl.assign(acc1, _max_f32(acc1, pair_max))
    return _max_f32(acc0, acc1)


def _sum_64(values):
    packed_sum = txl.local_scalar("uint64")
    txl.ptx.mov.b64(packed_sum, _f32(0.0), _f32(0.0))
    for pair in range(32):
        txl.ptx.add.f32x2(packed_sum, packed_sum, _pack2(values[pair * 2], values[pair * 2 + 1]))
    sum_lo = txl.local_scalar("float32")
    sum_hi = txl.local_scalar("float32")
    txl.ptx.mov.b64(sum_lo, sum_hi, packed_sum)
    total = txl.local_scalar("float32")
    txl.ptx.add.ftz.f32(total, sum_lo, sum_hi)
    return total


def _tmem_load_x64(values, addr):
    txl.ptx[_TMEM_LD_X64](*(values[index] for index in range(64)), addr)


def _tmem_load_x16(values, addr):
    txl.ptx[_TMEM_LD_X16](*(values[index] for index in range(16)), addr)


def _build_producer_kernel():
    @txl.kernel(
        warps=PRODUCER_WARPS,
        arch=CUDA_ARCH,
        min_blocks_per_sm=1,
        grid=lambda p: [p["num_work_items"]],
        host_prelude=_host_prelude,
    )
    def blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_sm103_producer(
        q: txl.gptr[txl.bf16, 3],
        k: txl.gptr[txl.bf16, 4],
        v: txl.gptr[txl.bf16, 4],
        scheduler_metadata: txl.gptr[txl.i32],
        k2q_row_ptr: txl.gptr[txl.i32],
        k2q_qsplit_indices: txl.gptr[txl.i32],
        partial_o: txl.gptr[txl.u8],
        partial_scale: txl.gptr[txl.f32],
        partial_lse: txl.gptr[txl.f32],
        partial_temperature_lse: txl.gptr[txl.f32],
        cu_seqlens_q: txl.gptr[txl.i32],
        cu_seqlens_k: txl.gptr[txl.i32],
        q_offsets: txl.gptr[txl.i32],
        kv_lens: txl.gptr[txl.i32],
        page_table: txl.gptr[txl.i32],
        q_group_segment_end_21: txl.i32,
        q_group_segment_end_20: txl.i32,
        q_group_segment_end_19: txl.i32,
        q_group_segment_end_18: txl.i32,
        q_group_segment_end_17: txl.i32,
        q_group_segment_end_16: txl.i32,
        q_group_segment_end_15: txl.i32,
        q_group_segment_end_14: txl.i32,
        q_group_segment_end_13: txl.i32,
        q_group_segment_end_12: txl.i32,
        q_group_segment_end_11: txl.i32,
        q_group_segment_end_10: txl.i32,
        q_group_segment_end_9: txl.i32,
        q_group_segment_end_8: txl.i32,
        q_group_segment_end_7: txl.i32,
        q_group_segment_end_6: txl.i32,
        q_group_segment_end_5: txl.i32,
        q_group_segment_end_4: txl.i32,
        q_group_segment_end_3: txl.i32,
        q_group_segment_end_2: txl.i32,
        total_q: txl.i32,
        num_q_heads: txl.i32,
        num_kv_heads: txl.i32,
        total_rows: txl.i32,
        nnz_per_head: txl.i32,
        work_capacity: txl.i32,
        num_work_items: txl.i32,
        topk: txl.i32,
        max_pages: txl.i32,
        causal: txl.i32,
        derive_q_offset: txl.i32,
        softmax_scale_log2: txl.f32,
        lse_temperature_scale: txl.f32,
        return_temperature_lse: txl.i32,
        *,
        host,
    ):
        q_map, k_map, v_map = host
        del q, k, v, partial_temperature_lse, work_capacity, num_work_items
        del lse_temperature_scale, return_temperature_lse
        work = txl.cta_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        arena = txl.alloc_buffer((SMEM_TOTAL,), txl.u8, scope="shared.dyn", align=1024)
        smem = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(arena.ptr_to([0])))

        def bar(offset):
            return smem + _u32(offset)

        with txl.If(warp == _i32(0)), txl.Then():
            leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
            with txl.If(leader != _u32(0)), txl.Then():
                for offset, count in _MBAR_INIT:
                    txl.ptx.mbarrier.init.shared__cta.b64(bar(offset), _u32(count))
                txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.warp.sync(_u32(_FULL_MASK))
        with txl.If(warp == _i32(0)), txl.Then():
            txl.ptx[_TMEM_ALLOC](bar(_SMEM_TMEM_MAILBOX), _u32(TMEM_COLS))
            txl.ptx[_TMEM_RELINQUISH]()
        txl.cuda.cta_sync()
        txl.ptx["tcgen05.fence::after_thread_sync"]()
        taddr = txl.local_scalar("uint32")
        txl.ptx.ld.volatile.shared.b32(taddr, bar(_SMEM_TMEM_MAILBOX))

        segment_ends = (
            q_group_segment_end_21,
            q_group_segment_end_20,
            q_group_segment_end_19,
            q_group_segment_end_18,
            q_group_segment_end_17,
            q_group_segment_end_16,
            q_group_segment_end_15,
            q_group_segment_end_14,
            q_group_segment_end_13,
            q_group_segment_end_12,
            q_group_segment_end_11,
            q_group_segment_end_10,
            q_group_segment_end_9,
            q_group_segment_end_8,
            q_group_segment_end_7,
            q_group_segment_end_6,
            q_group_segment_end_5,
            q_group_segment_end_4,
            q_group_segment_end_3,
            q_group_segment_end_2,
        )

        def decode_group_count():
            group_count = txl.local_scalar("int32", init=_i32(21))
            for index, segment_end in enumerate(segment_ends):
                with txl.If(work >= segment_end), txl.Then():
                    txl.assign(group_count, _i32(20 - index))
            return group_count

        def metadata_field(index):
            return _ld_global_i32(scheduler_metadata, work * _i32(6) + _i32(index))

        def decode_softmax_record():
            group_count = decode_group_count()
            head_kv = metadata_field(0)
            row_linear = metadata_field(1)
            q_begin = metadata_field(2)
            q_count = metadata_field(3)
            batch = metadata_field(4)
            kv_block = metadata_field(5)
            row_start = (
                _ld_global_i32(k2q_row_ptr, head_kv * (total_rows + _i32(1)) + row_linear) + q_begin
            )
            q_batch = _ld_global_i32(cu_seqlens_q, batch)
            k_batch = _ld_global_i32(cu_seqlens_k, batch)
            kv_len = _ld_global_i32(kv_lens, batch)
            with txl.If(max_pages == _i32(0)), txl.Then():
                txl.assign(kv_len, _ld_global_i32(cu_seqlens_k, batch + _i32(1)) - k_batch)
            query_offset = _ld_global_i32(q_offsets, batch)
            with txl.If(derive_q_offset != _i32(0)), txl.Then():
                txl.assign(
                    query_offset, kv_len - (_ld_global_i32(cu_seqlens_q, batch + _i32(1)) - q_batch)
                )
            return (
                group_count,
                head_kv,
                row_start,
                q_count,
                kv_block,
                q_batch,
                kv_len,
                query_offset,
            )

        def decode_qload_record():
            group_count = decode_group_count()
            head_kv = metadata_field(0)
            row_linear = metadata_field(1)
            q_begin = metadata_field(2)
            q_count = metadata_field(3)
            batch = metadata_field(4)
            row_start = (
                _ld_global_i32(k2q_row_ptr, head_kv * (total_rows + _i32(1)) + row_linear) + q_begin
            )
            q_batch = _ld_global_i32(cu_seqlens_q, batch)
            return group_count, head_kv, row_start, q_count, q_batch

        def decode_loader_record():
            return metadata_field(0), metadata_field(4), metadata_field(5)

        roles = txl.specialize(chain_dispatch=False)
        common_regs = roles.register_scope("common", warps=range(12, 16), regs=48)
        r_even = roles.role("softmax_even", warps=range(0, 4), regs=176)
        r_odd = roles.role("softmax_odd", warps=range(4, 8), regs=176)
        r_qload = roles.role("qload", warps=range(8, 12), regs=112)
        r_mma = roles.role("mma", warps=[12], register_scope=common_regs)
        r_transform = roles.role("transform", warps=range(13, 15), register_scope=common_regs)
        r_load = roles.role("load", warps=[15], register_scope=common_regs)

        with txl.If(txl.And(warp >= _i32(12), warp <= _i32(15))), txl.Then():
            common_regs.emit()

        def emit_softmax(stage, warp_in_group, first_group, record):
            (group_count, head_kv, row_start, q_count, kv_block, q_batch, kv_len, query_offset) = (
                record
            )
            my_row = warp_in_group * _i32(32) + lane
            tmem_row = txl.shift_left(txl.cast(warp_in_group * _i32(32), "uint32"), _u32(16))
            iteration = txl.local_scalar("int32", init=_i32(0))
            trip_count = (
                (group_count + _i32(1)) // _i32(2) if first_group == 0 else group_count // _i32(2)
            )
            with txl.While(iteration < trip_count):
                group = iteration * _i32(2) + _i32(first_group)
                phase = iteration & _i32(1)
                _mbar_wait(bar(_MBAR_S_FULL + 8 * stage), phase)
                token_in_group = my_row // _i32(4)
                edge = group * _i32(32) + token_in_group
                row_valid = txl.local_scalar("int32", init=txl.cast(edge < q_count, "int32"))
                owner_lane = (lane // _i32(4)) * _i32(4)
                owned_packed = txl.local_scalar("int32", init=_i32(-1))
                with txl.If(txl.And(lane == owner_lane, edge < q_count)), txl.Then():
                    txl.assign(
                        owned_packed,
                        _ld_global_i32(
                            k2q_qsplit_indices, head_kv * nnz_per_head + row_start + edge
                        ),
                    )
                packed_q = _shfl_idx_i32(owned_packed, txl.cast(owner_lane, "uint32"))
                q_index = packed_q & _i32(0x00FFFFFF)
                valid_cols = txl.local_scalar("int32", init=_i32(0))
                with txl.If(row_valid != _i32(0)), txl.Then():
                    txl.assign(valid_cols, kv_len - kv_block * _i32(PAGE_SIZE))
                    with txl.If(valid_cols > _i32(PAGE_SIZE)), txl.Then():
                        txl.assign(valid_cols, _i32(PAGE_SIZE))
                    with txl.If(causal != _i32(0)), txl.Then():
                        causal_cols = query_offset + q_index - kv_block * _i32(PAGE_SIZE) + _i32(1)
                        with txl.If(valid_cols > causal_cols), txl.Then():
                            txl.assign(valid_cols, causal_cols)
                    with txl.If(valid_cols < _i32(0)), txl.Then():
                        txl.assign(valid_cols, _i32(0))

                score0 = txl.alloc_local((64,), "float32")
                score1 = txl.alloc_local((64,), "float32")
                score_addr = taddr + _u32(stage * 128) + tmem_row
                _tmem_load_x64(score0, score_addr)
                body_valid = txl.local_scalar("int32", init=valid_cols)
                with txl.If(body_valid < _i32(0)), txl.Then():
                    txl.assign(body_valid, _i32(0))
                with txl.If(txl.And(body_valid > _i32(0), body_valid < _i32(64))), txl.Then():
                    _mask_64(score0, body_valid)
                body_max = txl.local_scalar("float32", init=_row_max_64(score0))
                with txl.If(body_valid <= _i32(0)), txl.Then():
                    txl.assign(body_max, _f32(_NEG_INF))

                _tmem_load_x64(score1, score_addr + _u32(64))
                tail_valid = txl.local_scalar("int32", init=valid_cols - _i32(64))
                with txl.If(tail_valid < _i32(0)), txl.Then():
                    txl.assign(tail_valid, _i32(0))
                with txl.If(txl.And(valid_cols > _i32(0), tail_valid < _i32(64))), txl.Then():
                    _mask_64(score1, tail_valid)
                tail_max = txl.local_scalar("float32", init=_row_max_64(score1))
                with txl.If(tail_valid <= _i32(0)), txl.Then():
                    txl.assign(tail_max, _f32(_NEG_INF))
                row_max = _max_f32(body_max, tail_max)
                is_neg_inf = txl.local_scalar("uint32")
                txl.ptx.setp.eq.ftz.f32(is_neg_inf, row_max, _f32(_NEG_INF))
                safe_max = txl.local_scalar("float32")
                txl.ptx.selp.f32(safe_max, _f32(0.0), row_max, txl.ptx.pred(is_neg_inf))
                neg_safe_max = txl.local_scalar("float32")
                txl.ptx.neg.ftz.f32(neg_safe_max, safe_max)
                score_bias = txl.local_scalar("float32", init=_f32(_NEG_INF))
                with txl.If(valid_cols > _i32(0)), txl.Then():
                    txl.ptx.mul.ftz.f32(score_bias, neg_safe_max, softmax_scale_log2)

                _mbar_wait(bar(_MBAR_P_EMPTY + 8 * stage), phase ^ _i32(1))
                _tmem_load_x64(score0, score_addr)
                with txl.If(txl.And(body_valid > _i32(0), body_valid < _i32(64))), txl.Then():
                    _mask_64(score0, body_valid)
                scale_pair = _pack2(softmax_scale_log2, softmax_scale_log2)
                bias_pair = _pack2(score_bias, score_bias)
                for pair in range(32):
                    _packed_fma_inplace(score0, pair * 2, scale_pair, bias_pair)
                for element in range(64):
                    txl.ptx.ex2.approx.ftz.f32(score0[element], score0[element])
                row_sum = txl.local_scalar("float32", init=_sum_64(score0))
                packed_probability = txl.alloc_local((32,), "uint32")
                for pair in range(32):
                    txl.ptx.cvt.rn.bf16x2.f32(
                        packed_probability[pair], score0[pair * 2 + 1], score0[pair * 2]
                    )
                probability_addr = score_addr + _u32(64)
                txl.ptx[_TMEM_ST_X32](
                    probability_addr, *(packed_probability[index] for index in range(32))
                )

                # Keep the original tail scores: the P store has reused their TMEM columns.
                with txl.If(txl.And(valid_cols > _i32(0), tail_valid < _i32(64))), txl.Then():
                    _mask_64(score1, tail_valid)
                for pair in range(32):
                    _packed_fma_inplace(score1, pair * 2, scale_pair, bias_pair)
                for element in range(64):
                    txl.ptx.ex2.approx.ftz.f32(score1[element], score1[element])
                tail_sum = _sum_64(score1)
                txl.ptx.add.ftz.f32(row_sum, row_sum, tail_sum)
                for pair in range(32):
                    txl.ptx.cvt.rn.bf16x2.f32(
                        packed_probability[pair], score1[pair * 2 + 1], score1[pair * 2]
                    )
                txl.ptx[_TMEM_ST_X32](
                    probability_addr + _u32(32), *(packed_probability[index] for index in range(32))
                )
                txl.ptx.tcgen05.wait__st.sync.aligned()
                _mbar_arrive(bar(_MBAR_P_FULL + 8 * stage))
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                _mbar_arrive(bar(_MBAR_S_EMPTY + 8 * stage))
                _mbar_wait(bar(_MBAR_O_FULL + 8 * stage), phase)

                q_head_local = my_row - token_in_group * _i32(4)
                split_slot = (packed_q >> _i32(24)) & _i32(255)
                output_valid = txl.local_scalar(
                    "int32",
                    init=txl.cast(
                        txl.And(edge < q_count, txl.And(split_slot >= _i32(0), split_slot < topk)),
                        "int32",
                    ),
                )
                partial_row = txl.local_scalar("int64", init=_i64(0))
                inv_sum = txl.local_scalar("float32", init=_f32(0.0))
                with txl.If(output_valid != _i32(0)), txl.Then():
                    q_abs = q_batch + q_index
                    q_head = head_kv * _i32(4) + q_head_local
                    txl.assign(
                        partial_row,
                        txl.cast(split_slot, "int64") * txl.cast(total_q * num_q_heads, "int64")
                        + txl.cast(q_abs * num_q_heads + q_head, "int64"),
                    )
                    reciprocal = txl.local_scalar("float32")
                    txl.ptx.rcp.approx.ftz.f32(reciprocal, row_sum)
                    txl.assign(
                        inv_sum,
                        txl.if_then_else(
                            txl.And(row_sum > _f32(0.0), row_sum == row_sum), reciprocal, _f32(0.0)
                        ),
                    )

                output_addr = taddr + _u32(256 + stage * 128) + tmem_row
                output_fragment = txl.alloc_local((16,), "float32")
                row_abs_max = txl.local_scalar("float32", init=_f32(0.0))
                segment = txl.local_scalar("int32", init=_i32(0))
                with txl.While(segment < _i32(8)):
                    _tmem_load_x16(
                        output_fragment, output_addr + txl.cast(segment * _i32(16), "uint32")
                    )
                    segment_max = txl.local_scalar("float32", init=output_fragment[0])
                    segment_min = txl.local_scalar("float32", init=output_fragment[0])
                    for element in range(1, 16):
                        txl.assign(segment_max, _max_f32(segment_max, output_fragment[element]))
                        txl.assign(segment_min, _min_ftz_f32(segment_min, output_fragment[element]))
                    neg_min = txl.local_scalar("float32")
                    txl.ptx.neg.ftz.f32(neg_min, segment_min)
                    txl.assign(row_abs_max, _max_f32(row_abs_max, _max_f32(segment_max, neg_min)))
                    txl.assign(segment, segment + _i32(1))
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                dequant_scale = txl.local_scalar("float32", init=_f32(0.0))
                quant_scale = txl.local_scalar("float32", init=_f32(0.0))
                with (
                    txl.If(txl.And(row_abs_max > _f32(0.0), row_abs_max == row_abs_max)),
                    txl.Then(),
                ):
                    scaled_abs = txl.local_scalar("float32")
                    txl.ptx.mul.ftz.f32(scaled_abs, row_abs_max, inv_sum)
                    txl.ptx.mul.ftz.f32(dequant_scale, scaled_abs, _f32(1.0 / 448.0))
                    txl.ptx.div.approx.ftz.f32(quant_scale, _f32(448.0), row_abs_max)
                with txl.If(output_valid != _i32(0)), txl.Then():
                    txl.ptx.st.global_.b32(partial_scale.ptr_to([partial_row]), dequant_scale)

                segment_q = txl.local_scalar("int32", init=_i32(0))
                with txl.While(segment_q < _i32(8)):
                    _tmem_load_x16(
                        output_fragment, output_addr + txl.cast(segment_q * _i32(16), "uint32")
                    )
                    with txl.If(output_valid != _i32(0)), txl.Then():
                        quant_pair = _pack2(quant_scale, quant_scale)
                        for pair in range(8):
                            _packed_mul_inplace(output_fragment, pair * 2, quant_pair)
                        packed_fp8 = txl.alloc_local((4,), "uint32")
                        for word in range(4):
                            low = txl.local_scalar("uint16")
                            high = txl.local_scalar("uint16")
                            txl.ptx.cvt.rn.satfinite.e4m3x2.f32(
                                low, output_fragment[word * 4 + 1], output_fragment[word * 4]
                            )
                            txl.ptx.cvt.rn.satfinite.e4m3x2.f32(
                                high, output_fragment[word * 4 + 3], output_fragment[word * 4 + 2]
                            )
                            txl.assign(
                                packed_fp8[word],
                                txl.bitwise_or(
                                    txl.cast(low, "uint32"),
                                    txl.shift_left(txl.cast(high, "uint32"), _u32(16)),
                                ),
                            )
                        partial_base = partial_row * _i64(HEAD_DIM) + txl.cast(
                            segment_q * _i32(16), "int64"
                        )
                        txl.ptx.st.global_.v4.b32(
                            partial_o.ptr_to([partial_base]),
                            packed_fp8[0],
                            packed_fp8[1],
                            packed_fp8[2],
                            packed_fp8[3],
                        )
                    txl.assign(segment_q, segment_q + _i32(1))

                with txl.If(output_valid != _i32(0)), txl.Then():
                    log_sum = txl.local_scalar("float32")
                    txl.ptx.lg2.approx.ftz.f32(log_sum, row_sum)
                    max_scaled = txl.local_scalar("float32")
                    log_scaled = txl.local_scalar("float32")
                    lse_value = txl.local_scalar("float32")
                    txl.ptx.mul.ftz.f32(max_scaled, row_max, softmax_scale_log2)
                    txl.ptx.mul.ftz.f32(log_scaled, log_sum, _f32(_LN2))
                    txl.ptx.fma.rn.ftz.f32(lse_value, max_scaled, _f32(_LN2), log_scaled)
                    txl.ptx.st.global_.b32(
                        partial_lse.ptr_to([partial_row]),
                        txl.if_then_else(row_sum > _f32(0.0), lse_value, _f32(_NEG_INF)),
                    )
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                _mbar_arrive(bar(_MBAR_O_EMPTY + 8 * stage))
                txl.assign(iteration, iteration + _i32(1))

        with r_even:
            emit_softmax(0, warp, 0, decode_softmax_record())

        with r_odd:
            emit_softmax(1, warp - _i32(4), 1, decode_softmax_record())

        with r_qload:
            (group_count_q, head_kv_q, row_start_q, q_count_q, q_batch_q) = decode_qload_record()
            qload_warp = warp - _i32(8)
            group_q = txl.local_scalar("int32", init=_i32(0))
            with txl.While(group_q < group_count_q):
                stage_q = group_q & _i32(1)
                phase_q = (group_q // _i32(2)) & _i32(1)
                _mbar_wait(bar(_MBAR_Q_EMPTY) + txl.cast(stage_q, "uint32") * _u32(8), phase_q ^ 1)
                elected = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
                with txl.If(elected != _u32(0)), txl.Then():
                    _mbar_expect_tx(bar(_MBAR_Q_FULL) + txl.cast(stage_q, "uint32") * _u32(8), 8192)
                elected_load = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
                with txl.If(elected_load != _u32(0)), txl.Then():
                    q_stage_addr = bar(_SMEM_Q) + txl.cast(stage_q, "uint32") * _u32(32768)
                    for local_edge in range(8):
                        edge_q = group_q * _i32(32) + qload_warp * _i32(8) + _i32(local_edge)
                        safe_edge = txl.local_scalar(
                            "int32", init=txl.if_then_else(edge_q < q_count_q, edge_q, _i32(0))
                        )
                        packed_q = _ld_global_i32(
                            k2q_qsplit_indices, head_kv_q * nnz_per_head + row_start_q + safe_edge
                        )
                        q_abs = txl.local_scalar(
                            "int32",
                            init=txl.if_then_else(
                                edge_q < q_count_q,
                                q_batch_q + (packed_q & _i32(0x00FFFFFF)),
                                _i32(0),
                            ),
                        )
                        row_base = q_abs * num_q_heads + head_kv_q * _i32(4)
                        dst = (
                            q_stage_addr
                            + _u32((local_edge) * 512)
                            + txl.cast(qload_warp * _i32(8 * 512), "uint32")
                        )
                        q_bar = bar(_MBAR_Q_FULL) + txl.cast(stage_q, "uint32") * _u32(8)
                        txl.ptx[_TMA_GATHER4_2D](
                            dst,
                            txl.address_of(q_map),
                            _i32(0),
                            row_base,
                            row_base + _i32(1),
                            row_base + _i32(2),
                            row_base + _i32(3),
                            q_bar,
                        )
                        txl.ptx[_TMA_GATHER4_2D](
                            dst + _u32(16384),
                            txl.address_of(q_map),
                            _i32(64),
                            row_base,
                            row_base + _i32(1),
                            row_base + _i32(2),
                            row_base + _i32(3),
                            q_bar,
                        )
                txl.assign(group_q, group_q + _i32(1))

        with r_mma:
            group_count_m = decode_group_count()
            phase_k = txl.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_K_FULL), phase_k)

            def issue_qk(group):
                stage_m = group & _i32(1)
                phase_m = (group // _i32(2)) & _i32(1)
                _mbar_wait(bar(_MBAR_Q_FULL) + txl.cast(stage_m, "uint32") * _u32(8), phase_m)
                _mbar_wait(
                    bar(_MBAR_S_EMPTY) + txl.cast(stage_m, "uint32") * _u32(8), phase_m ^ _i32(1)
                )
                q_lo = txl.local_scalar(
                    "uint32",
                    init=txl.uniform(
                        ((bar(_SMEM_Q) >> _u32(4)) & _u32(0x3FFF))
                        + txl.cast(stage_m, "uint32") * _u32(2048)
                    ),
                )
                k_lo = txl.local_scalar(
                    "uint32", init=txl.uniform((bar(_SMEM_K) >> _u32(4)) & _u32(0x3FFF))
                )
                leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
                for atom, delta in enumerate(_QK_DELTAS):
                    txl.ptx[_MMA_F16](
                        taddr + txl.cast(stage_m, "uint32") * _u32(128),
                        _pack2(q_lo + _u32(delta), _u32(_DESC_HI)),
                        _pack2(k_lo + _u32(delta), _u32(_DESC_HI)),
                        _u32(_QK_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        txl.ptx.pred(_u32(0 if atom == 0 else 1)),
                        pred=leader,
                    )
                leader_s = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
                txl.ptx[_TCGEN05_COMMIT](
                    bar(_MBAR_S_FULL) + txl.cast(stage_m, "uint32") * _u32(8), pred=leader_s
                )
                leader_q = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
                txl.ptx[_TCGEN05_COMMIT](
                    bar(_MBAR_Q_EMPTY) + txl.cast(stage_m, "uint32") * _u32(8), pred=leader_q
                )

            def issue_pv(group):
                stage_pv = group & _i32(1)
                phase_pv = (group // _i32(2)) & _i32(1)
                _mbar_wait(bar(_MBAR_P_FULL) + txl.cast(stage_pv, "uint32") * _u32(8), phase_pv)
                _mbar_wait(
                    bar(_MBAR_O_EMPTY) + txl.cast(stage_pv, "uint32") * _u32(8), phase_pv ^ _i32(1)
                )
                v_lo = txl.local_scalar(
                    "uint32",
                    init=txl.uniform(((bar(_SMEM_V) >> _u32(4)) & _u32(0x3FFF)) | _u32(0x4000000)),
                )
                leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
                probability_addr = taddr + txl.cast(stage_pv, "uint32") * _u32(128) + _u32(64)
                output_addr = taddr + _u32(256) + txl.cast(stage_pv, "uint32") * _u32(128)
                for atom, (ta_delta, v_delta) in enumerate(
                    zip(_PV_TMEM_DELTAS, _PV_SMEM_DELTAS, strict=True)
                ):
                    txl.ptx[_MMA_F16](
                        output_addr,
                        probability_addr + _u32(ta_delta),
                        _pack2(v_lo + _u32(v_delta), _u32(_DESC_HI)),
                        _u32(_PV_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        txl.ptx.pred(_u32(0 if atom == 0 else 1)),
                        pred=leader,
                    )
                leader_o = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
                txl.ptx[_TCGEN05_COMMIT](
                    bar(_MBAR_O_FULL) + txl.cast(stage_pv, "uint32") * _u32(8), pred=leader_o
                )
                leader_p = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
                txl.ptx[_TCGEN05_COMMIT](
                    bar(_MBAR_P_EMPTY) + txl.cast(stage_pv, "uint32") * _u32(8), pred=leader_p
                )

            issue_qk(_i32(0))
            with txl.If(group_count_m > _i32(1)), txl.Then():
                issue_qk(_i32(1))
            phase_v = txl.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_V_FULL), phase_v)
            group_m = txl.local_scalar("int32", init=_i32(2))
            with txl.While(group_m < group_count_m):
                issue_pv(group_m - _i32(2))
                issue_qk(group_m)
                txl.assign(group_m, group_m + _i32(1))
            drain_start = txl.local_scalar(
                "int32",
                init=txl.if_then_else(group_count_m == _i32(1), _i32(0), group_count_m - _i32(2)),
            )
            drain = txl.local_scalar("int32", init=drain_start)
            with txl.While(drain < group_count_m):
                issue_pv(drain)
                txl.assign(drain, drain + _i32(1))
            completed = txl.local_scalar("int32", init=drain_start)
            with txl.While(completed < group_count_m):
                completed_stage = completed & _i32(1)
                completed_phase = (completed // _i32(2)) & _i32(1)
                _mbar_wait(
                    bar(_MBAR_O_EMPTY) + txl.cast(completed_stage, "uint32") * _u32(8),
                    completed_phase,
                )
                txl.assign(completed, completed + _i32(1))
            tmem_dealloc = txl.local_scalar("uint32")
            txl.ptx.ld.volatile.shared.b32(tmem_dealloc, bar(_SMEM_TMEM_MAILBOX))
            txl.ptx[_TMEM_DEALLOC](tmem_dealloc, _u32(TMEM_COLS))

        with r_transform:
            pass

        with r_load:
            head_kv_l, batch_l, kv_block_l = decode_loader_record()
            physical_page = _ld_global_i32(page_table, batch_l * max_pages + kv_block_l)
            with txl.If(physical_page < _i32(0)), txl.Then():
                txl.assign(physical_page, _i32(0))
            page_head = physical_page * num_kv_heads + head_kv_l
            elected = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
            with txl.If(elected != _u32(0)), txl.Then():
                _mbar_expect_tx(bar(_MBAR_K_FULL), 32768)
                for dim_half in range(2):
                    for token_half in range(2):
                        txl.ptx[_TMA_G2S_4D](
                            bar(_SMEM_K + dim_half * 16384 + token_half * 8192),
                            txl.address_of(k_map),
                            _i32(0),
                            _i32(token_half * 64),
                            _i32(dim_half),
                            page_head,
                            bar(_MBAR_K_FULL),
                        )
                _mbar_expect_tx(bar(_MBAR_V_FULL), 32768)
                for dim_half in range(2):
                    for token_half in range(2):
                        txl.ptx[_TMA_G2S_4D](
                            bar(_SMEM_V + dim_half * 16384 + token_half * 8192),
                            txl.address_of(v_map),
                            _i32(0),
                            _i32(token_half * 64),
                            _i32(dim_half),
                            page_head,
                            bar(_MBAR_V_FULL),
                        )

    return blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_sm103_producer


def _build_reducer_kernel():
    @txl.kernel(
        warps=REDUCER_WARPS,
        arch=CUDA_ARCH,
        grid=lambda p: [(p["total_q"] * p["num_q_heads"] + 31) // 32],
    )
    def blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_sm103_reduce(
        partial_o: txl.gptr[txl.u8],
        partial_scale: txl.gptr[txl.f32],
        partial_lse: txl.gptr[txl.f32],
        partial_temperature_lse: txl.gptr[txl.f32],
        split_counts: txl.gptr[txl.i32],
        out: txl.gptr[txl.bf16],
        lse: txl.gptr[txl.f32],
        temperature_lse: txl.gptr[txl.f32],
        total_q: txl.i32,
        num_q_heads: txl.i32,
        num_kv_heads: txl.i32,
        qhead_per_kv: txl.i32,
        topk: txl.i32,
        return_softmax_lse: txl.i32,
        return_temperature_lse: txl.i32,
    ):
        del partial_temperature_lse, split_counts, num_kv_heads, qhead_per_kv, topk
        tid = txl.thread_id()
        lane = txl.lane_id()
        row_group = tid // _i32(8)
        lane_in_row = tid & _i32(7)
        leader_lane = row_group * _i32(8)
        row = txl.cta_id() * _i32(32) + row_group
        total_rows_out = total_q * num_q_heads
        row_valid = txl.local_scalar("int32", init=txl.cast(row < total_rows_out, "int32"))
        lane_lse = txl.local_scalar("float32", init=_f32(_NEG_INF))
        lane_scale = txl.local_scalar("float32", init=_f32(0.0))
        with txl.If(txl.And(row_valid != _i32(0), lane_in_row < _i32(4))), txl.Then():
            split_row = txl.cast(lane_in_row, "int64") * txl.cast(
                total_rows_out, "int64"
            ) + txl.cast(row, "int64")
            txl.ptx.ld.global_.nc.b32(lane_lse, partial_lse.ptr_to([split_row]))
            txl.ptx.ld.global_.nc.b32(lane_scale, partial_scale.ptr_to([split_row]))
        lse_max = txl.local_scalar("float32", init=lane_lse)
        for delta in (1, 2, 4):
            txl.assign(lse_max, _max_f32(lse_max, _shfl_xor_f32(lse_max, delta)))
        safe_lse_max = txl.local_scalar(
            "float32", init=txl.if_then_else(lse_max == _f32(_NEG_INF), _f32(0.0), lse_max)
        )
        lane_weight = txl.local_scalar("float32", init=_f32(0.0))
        with txl.If(lane_in_row < _i32(4)), txl.Then():
            difference = txl.local_scalar("float32")
            scaled = txl.local_scalar("float32")
            txl.ptx.sub.ftz.f32(difference, lane_lse, safe_lse_max)
            txl.ptx.mul.ftz.f32(scaled, difference, _f32(_LOG2E))
            txl.ptx.ex2.approx.ftz.f32(lane_weight, scaled)
            with txl.If(lane_lse == _f32(_NEG_INF)), txl.Then():
                txl.assign(lane_weight, _f32(0.0))
        lse_sum = txl.local_scalar("float32", init=lane_weight)
        for delta in (1, 2, 4):
            peer = _shfl_xor_f32(lse_sum, delta)
            txl.ptx.add.ftz.f32(lse_sum, lse_sum, peer)
        reciprocal = txl.local_scalar("float32")
        txl.ptx.rcp.approx.ftz.f32(reciprocal, lse_sum)
        inv_sum = txl.local_scalar(
            "float32",
            init=txl.if_then_else(
                txl.And(lse_sum > _f32(0.0), lse_sum == lse_sum), reciprocal, _f32(0.0)
            ),
        )
        txl.ptx.mul.ftz.f32(lane_weight, lane_weight, inv_sum)
        txl.ptx.mul.ftz.f32(lane_weight, lane_weight, lane_scale)
        weights = txl.alloc_local((4,), "float32")
        for split in range(4):
            txl.assign(
                weights[split],
                _shfl_idx_f32(lane_weight, txl.cast(leader_lane + _i32(split), "uint32")),
            )

        with txl.If(txl.And(lane_in_row == _i32(0), row_valid != _i32(0))), txl.Then():
            final_lse = txl.local_scalar("float32", init=_f32(_NEG_INF))
            with (
                txl.If(txl.Or(return_softmax_lse != _i32(0), return_temperature_lse != _i32(0))),
                txl.Then(),
            ):
                log_sum = txl.local_scalar("float32")
                txl.ptx.lg2.approx.ftz.f32(log_sum, lse_sum)
                with txl.If(lse_sum > _f32(0.0)), txl.Then():
                    txl.ptx.fma.rn.ftz.f32(final_lse, log_sum, _f32(_LN2), safe_lse_max)
            with txl.If(return_softmax_lse != _i32(0)), txl.Then():
                txl.ptx.st.global_.b32(lse.ptr_to([row]), final_lse)
            with txl.If(return_temperature_lse != _i32(0)), txl.Then():
                txl.ptx.st.global_.b32(temperature_lse.ptr_to([row]), final_lse)

        with txl.If(row_valid != _i32(0)), txl.Then():
            col = lane_in_row * _i32(16)
            accum = txl.alloc_local((16,), "float32")
            values = txl.alloc_local((16,), "float32")
            for split in range(4):
                base = (
                    txl.cast(split, "int64") * txl.cast(total_rows_out, "int64")
                    + txl.cast(row, "int64")
                ) * _i64(HEAD_DIM) + txl.cast(col, "int64")
                raw = txl.alloc_local((2,), "uint64")
                txl.ptx.ld.global_.nc.b64(raw[0], partial_o.ptr_to([base]))
                txl.ptx.ld.global_.nc.b64(raw[1], partial_o.ptr_to([base + _i64(8)]))
                for chunk in range(2):
                    for pair in range(4):
                        pair_bits = txl.local_scalar("uint16")
                        txl.ptx.mov.b16(
                            pair_bits,
                            txl.cast(
                                txl.bitwise_and(
                                    txl.shift_right(raw[chunk], _u32(pair * 16)), txl.uint64(0xFFFF)
                                ),
                                "uint16",
                            ),
                        )
                        half_pair = txl.local_scalar("uint32")
                        txl.ptx.cvt.rn.f16x2.e4m3x2(half_pair, pair_bits)
                        for half in range(2):
                            half_bits = txl.local_scalar("uint16")
                            txl.ptx.mov.b16(
                                half_bits,
                                txl.cast(
                                    txl.bitwise_and(
                                        txl.shift_right(half_pair, _u32(half * 16)), _u32(0xFFFF)
                                    ),
                                    "uint16",
                                ),
                            )
                            txl.ptx.cvt.f32.f16(values[chunk * 8 + pair * 2 + half], half_bits)
                for element in range(16):
                    txl.ptx.fma.rn.ftz.f32(
                        accum[element],
                        values[element],
                        weights[split],
                        _f32(0.0) if split == 0 else accum[element],
                    )
            packed_bf16 = txl.alloc_local((8,), "uint32")
            for pair in range(8):
                txl.ptx.cvt.rn.bf16x2.f32(packed_bf16[pair], accum[pair * 2 + 1], accum[pair * 2])
            output_base = txl.cast(row * _i32(HEAD_DIM) + col, "int64")
            txl.ptx.st.global_.v4.b32(
                out.ptr_to([output_base]),
                packed_bf16[0],
                packed_bf16[1],
                packed_bf16[2],
                packed_bf16[3],
            )
            txl.ptx.st.global_.v4.b32(
                out.ptr_to([output_base + _i64(8)]),
                packed_bf16[4],
                packed_bf16[5],
                packed_bf16[6],
                packed_bf16[7],
            )

    return blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_sm103_reduce


@lru_cache(maxsize=1)
def _kernels():
    return (_build_producer_kernel(), _build_reducer_kernel())


def get_kernel(**config: Any):
    del config
    return [kernel.func for kernel in _kernels()]


@lru_cache(maxsize=1)
def _compiled_kernels():
    from tirx_kernels.runner import compile_kernel

    previous = os.environ.get("TVM_CUDA_PTXAS_REG_LEVEL")
    os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = _PTXAS_REGISTER_USAGE_LEVEL
    try:
        return tuple(compile_kernel(func, cuda_compile_mode="nvcc") for func in get_kernel())
    finally:
        if previous is None:
            os.environ.pop("TVM_CUDA_PTXAS_REG_LEVEL", None)
        else:
            os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = previous


@lru_cache(maxsize=1)
def _source_modules():
    """Load the exact FlashInfer producer and fixed-four reducer specializations."""
    from flashinfer.msa_ops import _blackwell_sm100

    return (
        _blackwell_sm100._get_module("reverse_prefill_bf16_paged_topk4_qload4", "sm103a"),
        _blackwell_sm100._get_module(
            "reverse_prefill_bf16_paged_topk4_qload4_const4_reduce", "sm103a"
        ),
    )


def _target_block_counts(hot: bool) -> list[int]:
    if not hot:
        return [256] * MAX_PAGES
    counts = [16 + 32 * index for index in range(11)] + [385]
    counts.extend([270] * 51)
    counts.append(293)
    if len(counts) != MAX_PAGES or sum(counts) != 4096 * TOPK:
        raise AssertionError("invalid fixed reverse-prefill degree sequence")
    return counts


def _realize_query_block_degrees(counts: list[int], rng: random.Random) -> list[list[int]]:
    """Realize 4096 degree-four query rows with distinct logical blocks."""
    permutation = list(range(MAX_PAGES))
    rng.shuffle(permutation)
    heap = [(-counts[index], rng.random(), permutation[index]) for index in range(MAX_PAGES)]
    heapq.heapify(heap)
    rows: list[list[int]] = []
    for _ in range(4096):
        selected = [heapq.heappop(heap) for _ in range(TOPK)]
        rows.append([entry[2] for entry in selected])
        for negative_count, _, block in selected:
            remaining = -negative_count - 1
            heapq.heappush(heap, (-remaining, rng.random(), block))
    if any(-entry[0] for entry in heap):
        raise AssertionError("reverse-prefill degree realization left unmatched edges")
    return rows


def _make_q2k_cpu(seed: int):
    import torch

    rng = random.Random(seed)
    q2k = torch.empty((NUM_KV_HEADS, TOTAL_Q, TOPK), dtype=torch.int32)
    combination = 0
    for head in range(NUM_KV_HEADS):
        for batch in range(3):
            rows = _realize_query_block_degrees(_target_block_counts(combination < 5), rng)
            begin = batch * 4096
            q2k[head, begin : begin + 4096] = torch.tensor(rows, dtype=torch.int32)
            combination += 1
    return q2k.contiguous()


def _make_page_table(pattern: str, seed: int):
    import torch

    page_table = torch.empty((3, MAX_PAGES), dtype=torch.int32)
    if pattern == "identity":
        for batch in range(3):
            page_table[batch] = torch.arange(
                batch * MAX_PAGES, (batch + 1) * MAX_PAGES, dtype=torch.int32
            )
    elif pattern == "permuted":
        generator = torch.Generator().manual_seed(seed + 911)
        for batch in range(3):
            page_table[batch] = (
                torch.randperm(MAX_PAGES, generator=generator, dtype=torch.int64).to(torch.int32)
                + batch * MAX_PAGES
            )
    elif pattern == "reused_negative":
        for batch in range(3):
            logical = torch.arange(MAX_PAGES, dtype=torch.int32)
            physical = batch * MAX_PAGES + ((logical * 17 + 9) % MAX_PAGES)
            repeated = torch.arange(0, MAX_PAGES, 7, dtype=torch.int64)
            physical[repeated] = physical[(repeated + 1) % MAX_PAGES]
            physical[5::11] = -1
            page_table[batch] = physical
    else:
        raise ValueError(f"unknown page pattern {pattern!r}")
    return page_table.contiguous()


_GUARD_ELEMS = 64
_U8_GUARD = 0x6D
_U8_SENTINEL = 0xA5
_BF16_GUARD = 42.5
_BF16_SENTINEL = -13.25
_F32_GUARD = -54321.25
_F32_SENTINEL = 12345.25
_I32_GUARD = -777777


def _guarded_tensor(shape, dtype, *, fill, guard, device):
    import torch

    elements = math.prod(shape)
    storage = torch.full((elements + 2 * _GUARD_ELEMS,), guard, dtype=dtype, device=device)
    view = storage[_GUARD_ELEMS : _GUARD_ELEMS + elements].view(shape)
    view.fill_(fill)
    return view, storage


def _new_buffers(device):
    import torch

    partial_o, partial_o_storage = _guarded_tensor(
        (TOPK, TOTAL_Q, NUM_Q_HEADS, HEAD_DIM),
        torch.uint8,
        fill=_U8_SENTINEL,
        guard=_U8_GUARD,
        device=device,
    )
    partial_scale, partial_scale_storage = _guarded_tensor(
        (TOPK, TOTAL_Q, NUM_Q_HEADS, TOPK),
        torch.float32,
        fill=_F32_SENTINEL,
        guard=_F32_GUARD,
        device=device,
    )
    partial_lse, partial_lse_storage = _guarded_tensor(
        (TOPK, TOTAL_Q, NUM_Q_HEADS),
        torch.float32,
        fill=_F32_SENTINEL,
        guard=_F32_GUARD,
        device=device,
    )
    partial_tlse, partial_tlse_storage = _guarded_tensor(
        (TOPK, TOTAL_Q, NUM_Q_HEADS),
        torch.float32,
        fill=_F32_SENTINEL,
        guard=_F32_GUARD,
        device=device,
    )
    out, out_storage = _guarded_tensor(
        (TOTAL_Q, NUM_Q_HEADS, HEAD_DIM),
        torch.bfloat16,
        fill=_BF16_SENTINEL,
        guard=_BF16_GUARD,
        device=device,
    )
    lse, lse_storage = _guarded_tensor(
        (TOTAL_Q, NUM_Q_HEADS), torch.float32, fill=_F32_SENTINEL, guard=_F32_GUARD, device=device
    )
    tlse, tlse_storage = _guarded_tensor(
        (TOTAL_Q, NUM_Q_HEADS), torch.float32, fill=_F32_SENTINEL, guard=_F32_GUARD, device=device
    )
    return {
        "partial_o": partial_o,
        "partial_scale": partial_scale,
        "partial_lse": partial_lse,
        "partial_tlse": partial_tlse,
        "out": out,
        "lse": lse,
        "tlse": tlse,
        "guards": {
            "partial_o": partial_o_storage,
            "partial_scale": partial_scale_storage,
            "partial_lse": partial_lse_storage,
            "partial_tlse": partial_tlse_storage,
            "out": out_storage,
            "lse": lse_storage,
            "tlse": tlse_storage,
        },
    }


def _fill_ftz_edges(*tensors) -> None:
    import torch

    edge_values = torch.tensor(
        [0.0, -0.0, 2.0**-133, -(2.0**-133), 2.0**-126, -(2.0**-126)],
        dtype=torch.float32,
        device=tensors[0].device,
    ).to(torch.bfloat16)
    for tensor in tensors:
        flat = tensor.view(-1)
        repeats = min(4096, flat.numel() // edge_values.numel())
        flat[: repeats * edge_values.numel()] = edge_values.repeat(repeats)


def prepare_data(**config: Any) -> dict[str, Any]:
    """Create the fixed route, exact reverse plan, and independent guarded workspaces."""
    import torch
    from flashinfer.msa_ops._blackwell_sm100_reverse_plan import build_bf16_paged_topk4_plan

    config = _without_label(config)
    _validate_config(**config)
    device = torch.device("cuda")
    seed = int(config["schedule_seed"])
    generator = torch.Generator(device=device).manual_seed(seed + 4000)
    q = torch.randn(
        (TOTAL_Q, NUM_Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device, generator=generator
    )
    k = torch.randn(
        (NUM_PAGES, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    v = torch.randn(
        (NUM_PAGES, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    if config["input_pattern"] == "ftz_edges":
        _fill_ftz_edges(q, k, v)

    q2k_cpu = _make_q2k_cpu(seed)
    plan_cpu = build_bf16_paged_topk4_plan(q2k_cpu, sm_count=152)
    geometry = plan_cpu["geometry"]
    if geometry.schedule_capacity != 640 or geometry.work_count != 389:
        raise AssertionError("fixed route must resolve to capacity/work 640/389")
    group_counts = {(item.q_count + 31) // 32 for item in geometry.work_items}
    if group_counts != set(range(1, 13)):
        raise AssertionError(f"schedule must cover group counts 1..12, got {group_counts}")

    q2k = q2k_cpu.to(device)
    page_table = _make_page_table(config["page_pattern"], seed).to(device)
    cu_q = torch.tensor((0, 4096, 8192, 12288), dtype=torch.int32, device=device)
    cu_k = torch.tensor((0, 8192, 16384, 24576), dtype=torch.int32, device=device)
    kv_lens = torch.full((3,), 8192, dtype=torch.int32, device=device)
    plan = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in plan_cpu.items()
    }
    return_lse = bool(config["return_softmax_lse"])
    return_tlse = bool(config["return_temperature_lse"])
    immutable = {
        "q": q.clone(),
        "k": k.clone(),
        "v": v.clone(),
        "q2k": q2k.clone(),
        "page_table": page_table.clone(),
        "cu_q": cu_q.clone(),
        "cu_k": cu_k.clone(),
        "kv_lens": kv_lens.clone(),
        "scheduler_metadata": plan["scheduler_metadata"].clone(),
        "k2q_row_ptr": plan["k2q_row_ptr"].clone(),
        "k2q_qsplit_indices": plan["k2q_qsplit_indices"].clone(),
        "split_counts": plan["split_counts"].clone(),
    }
    return {
        "config": config,
        "q": q,
        "k": k,
        "v": v,
        "q2k": q2k,
        "page_table": page_table,
        "cu_q": cu_q,
        "cu_k": cu_k,
        "kv_lens": kv_lens,
        "plan": plan,
        "geometry": geometry,
        "return_lse": return_lse,
        "return_tlse": return_tlse,
        "softmax_scale_log2": (HEAD_DIM**-0.5) / math.log(2.0),
        "immutable": immutable,
        "tirx": _new_buffers(device),
        "source": _new_buffers(device),
    }


def _producer_args(data: dict[str, Any], slot: str) -> tuple[Any, ...]:
    buffers = data[slot]
    plan = data["plan"]
    geometry = data["geometry"]
    return (
        data["q"],
        data["k"],
        data["v"],
        plan["scheduler_metadata"].view(-1),
        plan["k2q_row_ptr"].view(-1),
        plan["k2q_qsplit_indices"].view(-1),
        buffers["partial_o"].view(-1),
        buffers["partial_scale"].view(-1),
        buffers["partial_lse"].view(-1),
        buffers["partial_tlse"].view(-1),
        data["cu_q"].view(-1),
        data["cu_k"].view(-1),
        data["cu_k"].view(-1),
        data["kv_lens"].view(-1),
        data["page_table"].view(-1),
        *plan["group_segment_ends"],
        TOTAL_Q,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        int(geometry.total_rows),
        TOTAL_Q * TOPK,
        int(geometry.schedule_capacity),
        int(geometry.work_count),
        TOPK,
        MAX_PAGES,
        1,
        1,
        data["softmax_scale_log2"],
        1.0,
        int(data["return_tlse"]),
    )


def _reducer_args(data: dict[str, Any], slot: str) -> tuple[Any, ...]:
    buffers = data[slot]
    return (
        buffers["partial_o"].view(-1),
        buffers["partial_scale"].view(-1),
        buffers["partial_lse"].view(-1),
        buffers["partial_tlse"].view(-1),
        data["plan"]["split_counts"].view(-1),
        buffers["out"].view(-1),
        buffers["lse"].view(-1),
        buffers["tlse"].view(-1),
        TOTAL_Q,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        4,
        TOPK,
        int(data["return_lse"] or data["return_tlse"]),
        int(data["return_tlse"]),
    )


def _tirx_launch(executables, data: dict[str, Any]):
    producer_args = _producer_args(data, "tirx")
    reducer_args = _reducer_args(data, "tirx")

    def launch():
        executables[0](*producer_args)
        executables[1](*reducer_args)

    launch._keep_alive = (executables, producer_args, reducer_args)
    return launch


def _source_launch(data: dict[str, Any]):
    import torch

    modules = _source_modules()
    producer_args = _producer_args(data, "source")
    reducer_args = _reducer_args(data, "source")
    stream = int(torch.cuda.current_stream(data["q"].device).cuda_stream)
    producer_grid = (int(data["geometry"].work_count), 1, 1)
    reducer_grid = ((TOTAL_Q * NUM_Q_HEADS + 31) // 32, 1, 1)

    def launch():
        modules[0].run(*producer_args, *producer_grid, stream)
        modules[1].run(*reducer_args, *reducer_grid, stream)

    launch._keep_alive = (modules, producer_args, reducer_args, producer_grid, reducer_grid)
    return launch


def _guard_value(name: str):
    if name == "partial_o":
        return _U8_GUARD
    if name == "out":
        return _BF16_GUARD
    return _F32_GUARD


def _assert_guards(slot: str, buffers: dict[str, Any]) -> None:
    import torch

    for name, storage in buffers["guards"].items():
        value = _guard_value(name)
        expected = torch.full((_GUARD_ELEMS,), value, dtype=storage.dtype, device=storage.device)
        if not torch.equal(storage[:_GUARD_ELEMS], expected):
            raise AssertionError(f"{slot}.{name} prefix guard was modified")
        if not torch.equal(storage[-_GUARD_ELEMS:], expected):
            raise AssertionError(f"{slot}.{name} suffix guard was modified")


def _bits(tensor):
    import torch

    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16)
    if tensor.dtype == torch.float32:
        return tensor.view(torch.int32)
    return tensor


def _assert_bitwise(name: str, ours, expected) -> None:
    import torch

    ours_bits = _bits(ours)
    expected_bits = _bits(expected)
    equal = ours_bits == expected_bits
    if bool(equal.all()):
        return
    first = int((~equal).view(-1).nonzero()[0].item())
    ours_value = float(ours.view(-1)[first].item())
    expected_value = float(expected.view(-1)[first].item())
    finite = torch.isfinite(ours.float()) & torch.isfinite(expected.float())
    max_abs = (
        float((ours.float()[finite] - expected.float()[finite]).abs().max().item())
        if bool(finite.any())
        else float("nan")
    )
    raise AssertionError(
        f"{name} is not bitwise equal at flat index {first}: "
        f"tirx={ours_value}, source={expected_value}, max_finite_abs={max_abs}"
    )


def _assert_immutable(data: dict[str, Any]) -> None:
    import torch

    current = {
        "q": data["q"],
        "k": data["k"],
        "v": data["v"],
        "q2k": data["q2k"],
        "page_table": data["page_table"],
        "cu_q": data["cu_q"],
        "cu_k": data["cu_k"],
        "kv_lens": data["kv_lens"],
        "scheduler_metadata": data["plan"]["scheduler_metadata"],
        "k2q_row_ptr": data["plan"]["k2q_row_ptr"],
        "k2q_qsplit_indices": data["plan"]["k2q_qsplit_indices"],
        "split_counts": data["plan"]["split_counts"],
    }
    for name, tensor in current.items():
        if not torch.equal(_bits(tensor), _bits(data["immutable"][name])):
            raise AssertionError(f"immutable input {name} was modified")


def _assert_dead_and_disabled(data: dict[str, Any], slot: str) -> None:
    buffers = data[slot]
    if not bool((buffers["partial_tlse"] == _F32_SENTINEL).all()):
        raise AssertionError(f"{slot}.partial_temperature_lse was modified")
    live_scale = TOPK * TOTAL_Q * NUM_Q_HEADS
    if not bool((buffers["partial_scale"].view(-1)[live_scale:] == _F32_SENTINEL).all()):
        raise AssertionError(f"{slot}.partial_scale dead capacity was modified")
    if not data["return_lse"] and not data["return_tlse"]:
        if not bool((buffers["lse"] == _F32_SENTINEL).all()):
            raise AssertionError(f"{slot}.disabled lse was modified")
    if not data["return_tlse"]:
        if not bool((buffers["tlse"] == _F32_SENTINEL).all()):
            raise AssertionError(f"{slot}.disabled temperature_lse was modified")


def _validate_outputs(data: dict[str, Any]) -> dict[str, float]:
    for slot in ("tirx", "source"):
        _assert_guards(slot, data[slot])
        _assert_dead_and_disabled(data, slot)
    _assert_immutable(data)
    for name in ("partial_o", "partial_scale", "partial_lse", "partial_tlse", "out"):
        _assert_bitwise(name, data["tirx"][name], data["source"][name])
    if data["return_lse"] or data["return_tlse"]:
        _assert_bitwise("lse", data["tirx"]["lse"], data["source"]["lse"])
    if data["return_tlse"]:
        _assert_bitwise("temperature_lse", data["tirx"]["tlse"], data["source"]["tlse"])
    return {
        "partial_o_mismatch": 0.0,
        "partial_scale_max_abs": 0.0,
        "partial_lse_max_abs": 0.0,
        "out_max_abs": 0.0,
        "lse_max_abs": 0.0,
        "temperature_lse_max_abs": 0.0,
    }


def _skip_unless_supported() -> None:
    import unittest

    import torch

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is unavailable")
    if torch.cuda.get_device_capability() != (10, 3):
        raise unittest.SkipTest("this kernel requires compute capability 10.3")
    if torch.cuda.get_device_properties(0).multi_processor_count != 152:
        raise unittest.SkipTest("this fixed route requires the 152-SM GB300 schedule")


def run_test(**config: Any) -> dict[str, float]:
    import torch

    _skip_unless_supported()
    data = prepare_data(**config)
    tirx_launch = _tirx_launch(_compiled_kernels(), data)
    source_launch = _source_launch(data)
    tirx_launch()
    source_launch()
    torch.cuda.synchronize()
    result = _validate_outputs(data)
    deterministic = {
        name: tensor.clone()
        for name, tensor in data["tirx"].items()
        if name != "guards" and hasattr(tensor, "clone")
    }
    tirx_launch()
    torch.cuda.synchronize()
    for name, expected in deterministic.items():
        _assert_bitwise(f"deterministic.{name}", data["tirx"][name], expected)
    _assert_guards("tirx", data["tirx"])
    _assert_immutable(data)
    return result


def prepare_bench(**config: Any):
    from tirx_kernels.runner import prepared_gpu_benchmark

    kernel_config = _without_label(config)
    _validate_config(**kernel_config)
    state = {"config": kernel_config, "executables": _compiled_kernels()}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs: Any
):
    from tirx_kernels.runner import bench, defer_gpu_interrupts, external_references_enabled

    with defer_gpu_interrupts():
        import torch

    config = _without_label({**prepared["config"], **kwargs})
    with_source = external_references_enabled()
    gpu_state = prepared.get("gpu_state")
    if gpu_state is None:
        _skip_unless_supported()
        data = prepare_data(**config)
        gpu_state = {
            "data": data,
            "tirx_launch": _tirx_launch(prepared["executables"], data),
            "source_launch": None,
            "validated": False,
            "with_source": with_source,
        }
        prepared["gpu_state"] = gpu_state
    elif gpu_state["with_source"] != with_source:
        raise RuntimeError("reference timing mode changed within one prepared benchmark")

    data = gpu_state["data"]
    tirx_launch = gpu_state["tirx_launch"]
    if not gpu_state["validated"]:
        tirx_launch()
        torch.cuda.synchronize()
        if with_source:
            with defer_gpu_interrupts():
                source_launch = _source_launch(data)
                gpu_state["source_launch"] = source_launch
                source_launch()
                torch.cuda.synchronize()
            _validate_outputs(data)
        gpu_state["validated"] = True

    source_launch = gpu_state["source_launch"]
    references = {"flashinfer": lambda: source_launch} if source_launch is not None else None
    return bench(
        {"tirx": tirx_launch},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config: Any):
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
