# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ 9f5051736e9fd5cab41c06118a7d4b5c1de23a6d), Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Blackwell MSA long paged BF16 GQA16 direct-group prefill on SM100a.

Upstream sources (FlashInfer @ 9f5051736e9fd5cab41c06118a7d4b5c1de23a6d):

- ``csrc/blackwell_msa/sm100a/blackwell_msa_long_prefill_paged_bf16_gqa16_direct_group_sm100.cu``
- ``csrc/blackwell_msa/sm100a/blackwell_msa_long_prefill_paged_bf16_gqa16_direct_group_sm100_binding.cu``
- ``flashinfer/msa_ops/_blackwell_sm100.py``
"""

import math
import os
from functools import lru_cache
from typing import Any

import tirx_kernels.tirx_lite as txl

KERNEL_META = {
    "name": "blackwell_msa_long_prefill_paged_bf16_gqa16_direct_group_sm100",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_100a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "9f5051736e9fd5cab41c06118a7d4b5c1de23a6d",
            },
            "import": "flashinfer",
        },
    ),
}

HEAD_DIM = 128
PAGE_SIZE = 128
TOPK = 16
GQA_RATIO = 16
THREADS = 512
NUM_WARPS = THREADS // 32
SMEM_TOTAL = 148480
TMEM_COLS = 512
MAX_PAGES = 8192
CUDA_ARCH = "sm_100a"
_PTXAS_REG_LEVEL = "2"


def _config(
    label: str,
    *,
    q_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    selection: str = "balanced",
    page_layout: str = "identity",
    return_temperature_lse: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    return {
        "label": label,
        "q_len": q_len,
        "kv_len": kv_len,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "selection": selection,
        "page_layout": page_layout,
        "return_temperature_lse": return_temperature_lse,
        "seed": seed,
    }


_BENCH_CONFIGS = [
    _config(
        "q8192_kv1048576_h16_hkv1_balanced",
        q_len=8192,
        kv_len=1048576,
        num_q_heads=16,
        num_kv_heads=1,
        seed=101,
    ),
    _config(
        "q8193_kv1048576_h16_hkv1_tail",
        q_len=8193,
        kv_len=1048576,
        num_q_heads=16,
        num_kv_heads=1,
        seed=102,
    ),
    _config(
        "q16384_kv1048576_h16_hkv1_balanced",
        q_len=16384,
        kv_len=1048576,
        num_q_heads=16,
        num_kv_heads=1,
        seed=103,
    ),
    _config(
        "q32768_kv1048576_h16_hkv1_balanced",
        q_len=32768,
        kv_len=1048576,
        num_q_heads=16,
        num_kv_heads=1,
        seed=104,
    ),
    _config(
        "q65536_kv1048576_h16_hkv1_balanced",
        q_len=65536,
        kv_len=1048576,
        num_q_heads=16,
        num_kv_heads=1,
        seed=105,
    ),
    _config(
        "q8192_kv1048576_h64_hkv4_balanced",
        q_len=8192,
        kv_len=1048576,
        num_q_heads=64,
        num_kv_heads=4,
        seed=106,
    ),
    _config(
        "q16384_kv1048576_h16_hkv1_hot_g128",
        q_len=16384,
        kv_len=1048576,
        num_q_heads=16,
        num_kv_heads=1,
        selection="hot_g128",
        seed=107,
    ),
    _config(
        "q16384_kv1048576_h16_hkv1_bucket_ladder",
        q_len=16384,
        kv_len=1048576,
        num_q_heads=16,
        num_kv_heads=1,
        selection="bucket_ladder",
        seed=108,
    ),
    _config(
        "q8192_kv8192_h16_hkv1_single_future_tlse",
        q_len=8192,
        kv_len=8192,
        num_q_heads=16,
        num_kv_heads=1,
        selection="single_future",
        return_temperature_lse=True,
        seed=109,
    ),
    _config(
        "q16384_kv1048576_h16_hkv1_balanced_permuted_pages",
        q_len=16384,
        kv_len=1048576,
        num_q_heads=16,
        num_kv_heads=1,
        page_layout="permuted",
        seed=103,
    ),
]

BENCH_CONFIGS = list(_BENCH_CONFIGS)
_CORRECTNESS_LABELS = {
    "q8192_kv1048576_h16_hkv1_balanced",
    "q8193_kv1048576_h16_hkv1_tail",
    "q8192_kv1048576_h64_hkv4_balanced",
    "q16384_kv1048576_h16_hkv1_hot_g128",
    "q16384_kv1048576_h16_hkv1_bucket_ladder",
    "q8192_kv8192_h16_hkv1_single_future_tlse",
    "q16384_kv1048576_h16_hkv1_balanced_permuted_pages",
}
CONFIGS = [c for c in BENCH_CONFIGS if c["label"] in _CORRECTNESS_LABELS]


def _without_label(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "label"}


def _validate_config(config: dict[str, Any]) -> None:
    if int(config["q_len"]) < 8192:
        raise ValueError("this long-prefill specialization requires q_len >= 8192")
    if int(config["kv_len"]) <= 0 or int(config["kv_len"]) > MAX_PAGES * PAGE_SIZE:
        raise ValueError("kv_len must be in [1, 1048576]")
    if int(config["num_q_heads"]) != GQA_RATIO * int(config["num_kv_heads"]):
        raise ValueError("this specialization requires GQA ratio 16")
    if config["selection"] not in {"balanced", "hot_g128", "bucket_ladder", "single_future"}:
        raise ValueError(f"unknown selection pattern {config['selection']!r}")
    if config["page_layout"] not in {"identity", "permuted"}:
        raise ValueError(f"unknown page layout {config['page_layout']!r}")


# cuTensorMapEncodeTiled enum values.
_TMA_INTERLEAVE_NONE = 0
_TMA_SWIZZLE_128B = 3
_TMA_L2_PROMOTION_NONE = 0
_TMA_OOB_FILL_NONE = 0


def _host_prelude(params):
    """Encode the exact three BF16 rank-four TensorMaps from the source binding."""

    hq = params["num_q_heads"]
    hkv = params["num_kv_heads"]
    total_q = params["total_q"]
    physical_pages = params["k"].shape[0]

    def encode(tensor, dims, strides, box):
        descriptor = txl.stack_alloca("tensormap", 1)
        txl.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            descriptor,
            "bfloat16",
            4,
            tensor.data,
            *dims,
            *strides,
            *box,
            1,
            1,
            1,
            1,
            _TMA_INTERLEAVE_NONE,
            _TMA_SWIZZLE_128B,
            _TMA_L2_PROMOTION_NONE,
            _TMA_OOB_FILL_NONE,
        )
        return descriptor

    q_map = encode(
        params["q"],
        (64, hq, HEAD_DIM // 64, total_q),
        (HEAD_DIM * 2, 64 * 2, hq * HEAD_DIM * 2),
        (64, GQA_RATIO, 1, 1),
    )
    kv_dims = (64, PAGE_SIZE, HEAD_DIM // 64, physical_pages * hkv)
    kv_strides = (HEAD_DIM * 2, 64 * 2, PAGE_SIZE * HEAD_DIM * 2)
    kv_box = (64, 64, 1, 1)
    return (
        q_map,
        encode(params["k"], kv_dims, kv_strides, kv_box),
        encode(params["v"], kv_dims, kv_strides, kv_box),
    )


# Linear SMEM layout and exact barrier initial states from the source.
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
_MBAR_P_FULL_2 = 120
_MBAR_P_EMPTY = 136
_MBAR_O_FULL = 152
_MBAR_O_EMPTY = 168
_SMEM_TMEM_MAILBOX = 184
_SMEM_Q = 1024
_SMEM_K = 66560
_SMEM_V = 99328
_MBARRIER_INIT = (
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
    (_MBAR_P_FULL_2, 128),
    (_MBAR_P_FULL_2 + 8, 128),
    (_MBAR_P_EMPTY, 1),
    (_MBAR_P_EMPTY + 8, 1),
    (_MBAR_O_FULL, 1),
    (_MBAR_O_FULL + 8, 1),
    (_MBAR_O_EMPTY, 128),
    (_MBAR_O_EMPTY + 8, 128),
)

_TMEM_SCORE = (0, 128)
_TMEM_OUTPUT = (256, 384)
_TMEM_P_OFFSET = 64
_DESC_HI = 0x40004040
_QK_IDESC = 136316048
_PV_IDESC = 136381584
_QK_STEPS = (2, 2, 2, 1018, 2, 2, 2)
_V_LBO_BIT = 0x04000000
_FULL_MASK = 0xFFFFFFFF
_NEG_INF = float("-inf")
_LN2 = 0.6931471805599453

_TMA_G2S_4D = "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes"
_TMA_G2S_4D_L2 = _TMA_G2S_4D + ".L2::cache_hint"
_MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
_TCGEN05_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
_TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
_TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
_TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
_TMEM_LD_X32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
_TMEM_ST_X16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
_TMEM_ST_X32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"


def _u32(value):
    return txl.uint32(value)


def _i32(value):
    return txl.int32(value)


def _f32(value):
    return txl.float32(value)


def _pack2(lo, hi):
    result = txl.local_scalar("uint64")
    txl.ptx.mov.b64(result, lo, hi)
    return result


def _mbar_wait(addr, phase):
    txl.cuda.mbarrier_wait(addr, phase)


def _mbar_arrive(addr):
    txl.ptx.mbarrier.arrive.release.cta.shared__cta.b64(addr)


def _mbar_expect_tx(addr, count):
    txl.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(addr, _u32(count))


def _ld_global_i32(buffer, index):
    result = txl.local_scalar("int32")
    txl.ptx.ld.global_.nc.b32(result, buffer.ptr_to([index]))
    return result


def _max_f32(lhs, rhs):
    result = txl.local_scalar("float32")
    txl.ptx.max.f32(result, lhs, rhs)
    return result


def _min_ftz_f32(lhs, rhs):
    result = txl.local_scalar("float32")
    txl.ptx.min.ftz.f32(result, lhs, rhs)
    return result


def _packed_fma(values, base, scale, bias):
    packed = txl.local_scalar("uint64")
    txl.ptx.fma.rn.ftz.f32x2(
        packed, _pack2(values[base], values[base + 1]), _pack2(scale, scale), _pack2(bias, bias)
    )
    txl.ptx.mov.b64(values[base], values[base + 1], packed)


def _packed_mul(values, base, scale):
    packed = txl.local_scalar("uint64")
    txl.ptx.mul.rn.ftz.f32x2(packed, _pack2(values[base], values[base + 1]), _pack2(scale, scale))
    txl.ptx.mov.b64(values[base], values[base + 1], packed)


def _tmem_load_x32(values, base, addr):
    txl.ptx[_TMEM_LD_X32](*(values[base + i] for i in range(32)), addr)


def _tmem_store_x16(addr, words, base=0):
    txl.ptx[_TMEM_ST_X16](addr, *(words[base + i] for i in range(16)))


def _tmem_store_x32(addr, words):
    txl.ptx[_TMEM_ST_X32](addr, *(words[i] for i in range(32)))


def _row_max_128(values):
    acc0 = txl.local_scalar("float32", init=_f32(_NEG_INF))
    acc1 = txl.local_scalar("float32", init=_f32(_NEG_INF))
    for quarter in range(4):
        for pair in range(16):
            pair_max = _max_f32(
                values[quarter * 32 + pair * 2], values[quarter * 32 + pair * 2 + 1]
            )
            if pair % 2 == 0:
                txl.assign(acc0, _max_f32(acc0, pair_max))
            else:
                txl.assign(acc1, _max_f32(acc1, pair_max))
    return _max_f32(acc0, acc1)


def _row_max_32(values):
    acc0 = txl.local_scalar("float32", init=_f32(_NEG_INF))
    acc1 = txl.local_scalar("float32", init=_f32(_NEG_INF))
    for pair in range(16):
        pair_max = _max_f32(values[2 * pair], values[2 * pair + 1])
        if pair % 2 == 0:
            txl.assign(acc0, _max_f32(acc0, pair_max))
        else:
            txl.assign(acc1, _max_f32(acc1, pair_max))
    return _max_f32(acc0, acc1)


def _row_min_32(values):
    result = txl.local_scalar("float32", init=values[0])
    for index in range(1, 32):
        txl.assign(result, _min_ftz_f32(result, values[index]))
    return result


def _block_sum_128(values):
    packed = txl.local_scalar("uint64")
    txl.ptx.mov.b64(packed, _f32(0.0), _f32(0.0))
    for pair in range(64):
        txl.ptx.add.f32x2(packed, packed, _pack2(values[2 * pair], values[2 * pair + 1]))
    lo = txl.local_scalar("float32")
    hi = txl.local_scalar("float32")
    txl.ptx.mov.b64(lo, hi, packed)
    result = txl.local_scalar("float32")
    txl.ptx.add.ftz.f32(result, lo, hi)
    return result


def _combine_int_frac_exp2(rounded, fraction):
    rounded_i = txl.local_scalar("int32")
    fraction_i = txl.local_scalar("int32")
    exponent = txl.local_scalar("int32")
    bits = txl.local_scalar("int32")
    result = txl.local_scalar("float32")
    txl.ptx.mov.b32(rounded_i, rounded)
    txl.ptx.mov.b32(fraction_i, fraction)
    txl.ptx.shl.b32(exponent, rounded_i, _u32(23))
    txl.ptx.add.s32(bits, exponent, fraction_i)
    txl.ptx.mov.b32(result, bits)
    return result


def _exp2_emulation_pair(values, base):
    clamped = txl.alloc_local((2,), "float32")
    txl.ptx.mov.b32(clamped[0], _max_f32(values[base], _f32(-127.0)))
    txl.ptx.mov.b32(clamped[1], _max_f32(values[base + 1], _f32(-127.0)))
    packed = txl.local_scalar("uint64")
    rhs = txl.local_scalar("uint64")
    addend = txl.local_scalar("uint64")
    rounded = txl.alloc_local((2,), "float32")
    rounded_back = txl.alloc_local((2,), "float32")
    fraction = txl.alloc_local((2,), "float32")
    polynomial = txl.alloc_local((2,), "float32")
    magic = _f32(12582912.0)
    txl.ptx.add.rm.ftz.f32x2(packed, _pack2(clamped[0], clamped[1]), _pack2(magic, magic))
    txl.ptx.mov.b64(rounded[0], rounded[1], packed)
    txl.ptx.sub.rn.ftz.f32x2(packed, _pack2(rounded[0], rounded[1]), _pack2(magic, magic))
    txl.ptx.mov.b64(rounded_back[0], rounded_back[1], packed)
    txl.ptx.sub.rn.ftz.f32x2(
        packed, _pack2(clamped[0], clamped[1]), _pack2(rounded_back[0], rounded_back[1])
    )
    txl.ptx.mov.b64(fraction[0], fraction[1], packed)
    txl.ptx.mov.b32(polynomial[0], _f32(0.07711908966302872))
    txl.ptx.mov.b32(polynomial[1], _f32(0.07711908966302872))
    for coefficient in (0.22756439447402954, 0.6951461434364319, 1.0):
        txl.ptx.fma.rn.ftz.f32x2(
            packed,
            _pack2(polynomial[0], polynomial[1]),
            _pack2(fraction[0], fraction[1]),
            _pack2(_f32(coefficient), _f32(coefficient)),
        )
        txl.ptx.mov.b64(polynomial[0], polynomial[1], packed)
    txl.ptx.mov.b32(values[base], _combine_int_frac_exp2(rounded[0], polynomial[0]))
    txl.ptx.mov.b32(values[base + 1], _combine_int_frac_exp2(rounded[1], polynomial[1]))


def _pack_fp8x4(word, f0, f1, f2, f3):
    lo = txl.local_scalar("uint16")
    hi = txl.local_scalar("uint16")
    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(lo, f1, f0)
    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(hi, f3, f2)
    txl.ptx.mov.b32(word, lo, hi)


def _commit(addr):
    leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
    txl.ptx[_TCGEN05_COMMIT](addr, pred=leader)


def _build_kernel():
    @txl.kernel(
        warps=NUM_WARPS,
        arch=CUDA_ARCH,
        min_blocks_per_sm=1,
        grid=lambda p: [p["num_work_items"]],
        host_prelude=_host_prelude,
    )
    def blackwell_msa_long_prefill_paged_bf16_gqa16_direct_group_sm100(
        q: txl.gptr[txl.bf16, 3],
        k: txl.gptr[txl.bf16, 4],
        v: txl.gptr[txl.bf16, 4],
        scheduler_metadata: txl.gptr[txl.i32],
        k2q_row_ptr: txl.gptr[txl.i32],
        k2q_qsplit_indices: txl.gptr[txl.i32],
        partial_o: txl.gptr[txl.u8],
        partial_scale: txl.gptr[txl.bf16],
        partial_lse: txl.gptr[txl.f32],
        partial_temperature_lse: txl.gptr[txl.f32],
        out: txl.gptr[txl.bf16],
        cu_seqlens_q: txl.gptr[txl.i32],
        cu_seqlens_k: txl.gptr[txl.i32],
        q_offsets: txl.gptr[txl.i32],
        kv_lens: txl.gptr[txl.i32],
        page_table: txl.gptr[txl.i32],
        q_group_segment_end_128: txl.i32,
        q_group_segment_end_64: txl.i32,
        q_group_segment_end_32: txl.i32,
        q_group_segment_end_16: txl.i32,
        q_group_segment_end_8: txl.i32,
        q_group_segment_end_4: txl.i32,
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
        del (
            q,
            k,
            v,
            out,
            q_group_segment_end_128,
            q_group_segment_end_64,
            q_group_segment_end_32,
            q_group_segment_end_16,
            q_group_segment_end_8,
            q_group_segment_end_4,
            q_group_segment_end_2,
            work_capacity,
            num_work_items,
            lse_temperature_scale,
        )

        tid = txl.thread_id()
        lane = tid % _i32(32)
        raw_warp = tid // _i32(32)
        warp = txl.local_scalar("int32")
        txl.ptx.shfl_sync.idx.b32(warp, raw_warp, _u32(0), _u32(31), _u32(_FULL_MASK))
        arena = txl.alloc_buffer((SMEM_TOTAL,), txl.u8, scope="shared.dyn", align=1024)
        smem = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(arena.ptr_to([0])))

        def bar(offset):
            return smem + _u32(offset)

        with txl.If(warp == _i32(0)), txl.Then():
            init_leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
            with txl.If(init_leader != _u32(0)), txl.Then():
                for offset, count in _MBARRIER_INIT:
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

        def decode_work():
            metadata = txl.cta_id() * _i32(6)
            head_kv = _ld_global_i32(scheduler_metadata, metadata)
            row_linear = _ld_global_i32(scheduler_metadata, metadata + _i32(1))
            q_begin = _ld_global_i32(scheduler_metadata, metadata + _i32(2))
            q_count = _ld_global_i32(scheduler_metadata, metadata + _i32(3))
            batch = _ld_global_i32(scheduler_metadata, metadata + _i32(4))
            kv_block = _ld_global_i32(scheduler_metadata, metadata + _i32(5))
            row_ptr_base = head_kv * (total_rows + _i32(1)) + row_linear
            row_start = _ld_global_i32(k2q_row_ptr, row_ptr_base) + q_begin
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
                head_kv,
                row_start,
                q_count,
                batch,
                kv_block,
                q_batch,
                k_batch,
                kv_len,
                query_offset,
            )

        def emit_softmax(stage, warp_origin):
            head_kv, row_start, q_count, batch, kv_block, q_batch, _, kv_len, query_offset = (
                decode_work()
            )
            del batch
            group_count = (q_count + _i32(7)) // _i32(8)
            local_warp = warp - _i32(warp_origin)
            my_row = local_warp * _i32(32) + lane
            row_bits = txl.shift_left(txl.cast(local_warp * _i32(32), "uint32"), _u32(16))
            iteration = txl.local_scalar("int32", init=_i32(0))
            iteration_count = (
                (group_count + _i32(1)) // _i32(2) if stage == 0 else group_count // _i32(2)
            )
            with txl.While(iteration < iteration_count):
                group = iteration * _i32(2) + _i32(stage)
                phase = iteration & _i32(1)
                _mbar_wait(bar(_MBAR_S_FULL + 8 * stage), phase)
                token_in_group = my_row // _i32(16)
                edge = group * _i32(8) + token_in_group
                row_valid = txl.local_scalar("int32", init=txl.cast(edge < q_count, "int32"))
                owner_lane = (lane // _i32(16)) * _i32(16)
                owned_packed = txl.local_scalar("int32", init=_i32(-1))
                with txl.If(txl.And(lane == owner_lane, edge < q_count)), txl.Then():
                    txl.assign(
                        owned_packed,
                        _ld_global_i32(
                            k2q_qsplit_indices, head_kv * nnz_per_head + row_start + edge
                        ),
                    )
                packed_q = txl.local_scalar("int32")
                txl.ptx.shfl_sync.idx.b32(
                    packed_q, owned_packed, txl.cast(owner_lane, "uint32"), _u32(31), _u32(_FULL_MASK)
                )
                q_idx = packed_q & _i32(0xFFFFFF)
                valid_cols = txl.local_scalar("int32", init=_i32(0))
                with txl.If(row_valid != _i32(0)), txl.Then():
                    txl.assign(valid_cols, kv_len - kv_block * _i32(PAGE_SIZE))
                    with txl.If(valid_cols > _i32(PAGE_SIZE)), txl.Then():
                        txl.assign(valid_cols, _i32(PAGE_SIZE))
                    with txl.If(causal != _i32(0)), txl.Then():
                        causal_cols = query_offset + q_idx - kv_block * _i32(PAGE_SIZE) + _i32(1)
                        with txl.If(valid_cols > causal_cols), txl.Then():
                            txl.assign(valid_cols, causal_cols)
                    with txl.If(valid_cols < _i32(0)), txl.Then():
                        txl.assign(valid_cols, _i32(0))

                scores = txl.alloc_local((128,), "float32")
                score_addr = taddr + _u32(_TMEM_SCORE[stage]) + row_bits
                for quarter in range(4):
                    _tmem_load_x32(scores, 32 * quarter, score_addr + _u32(32 * quarter))
                with txl.If(valid_cols < _i32(128)), txl.Then():
                    for quarter in range(4):
                        limit = valid_cols - _i32(32 * quarter)
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
                        for element in range(32):
                            txl.assign(
                                scores[32 * quarter + element],
                                txl.if_then_else(
                                    (mask & _u32(1 << element)) != _u32(0),
                                    scores[32 * quarter + element],
                                    _f32(_NEG_INF),
                                ),
                            )
                row_max = _row_max_128(scores)
                safe_max = txl.local_scalar(
                    "float32", init=txl.if_then_else(row_max == _f32(_NEG_INF), _f32(0.0), row_max)
                )
                neg_safe = txl.local_scalar("float32")
                scaled_neg_safe = txl.local_scalar("float32")
                txl.ptx.neg.ftz.f32(neg_safe, safe_max)
                txl.ptx.mul.ftz.f32(scaled_neg_safe, neg_safe, softmax_scale_log2)
                score_bias = txl.local_scalar(
                    "float32",
                    init=txl.if_then_else(valid_cols > _i32(0), scaled_neg_safe, _f32(_NEG_INF)),
                )
                _mbar_wait(bar(_MBAR_P_EMPTY + 8 * stage), phase ^ _i32(1))
                for pair in range(64):
                    _packed_fma(scores, 2 * pair, softmax_scale_log2, score_bias)

                lower_words = txl.alloc_local((16,), "uint32")
                for segment in range(2):
                    base = 32 * segment
                    for element in range(32):
                        txl.ptx.ex2.approx.ftz.f32(scores[base + element], scores[base + element])
                    for pair in range(16):
                        txl.ptx.cvt.rn.bf16x2.f32(
                            lower_words[pair], scores[base + 2 * pair + 1], scores[base + 2 * pair]
                        )
                    _tmem_store_x16(
                        taddr + _u32(_TMEM_SCORE[stage] + _TMEM_P_OFFSET + 16 * segment) + row_bits,
                        lower_words,
                    )
                txl.ptx.tcgen05.wait__st.sync.aligned()
                _mbar_arrive(bar(_MBAR_P_FULL + 8 * stage))

                upper_words = txl.alloc_local((32,), "uint32")
                for segment in range(2):
                    base = 64 + 32 * segment
                    for element in range(24):
                        txl.ptx.ex2.approx.ftz.f32(scores[base + element], scores[base + element])
                    for pair in range(12, 16):
                        _exp2_emulation_pair(scores, base + 2 * pair)
                    for pair in range(16):
                        txl.ptx.cvt.rn.bf16x2.f32(
                            upper_words[16 * segment + pair],
                            scores[base + 2 * pair + 1],
                            scores[base + 2 * pair],
                        )
                _tmem_store_x32(
                    taddr + _u32(_TMEM_SCORE[stage] + _TMEM_P_OFFSET + 32) + row_bits, upper_words
                )
                row_sum = _block_sum_128(scores)
                txl.ptx.tcgen05.wait__st.sync.aligned()
                _mbar_arrive(bar(_MBAR_P_FULL_2 + 8 * stage))
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                _mbar_arrive(bar(_MBAR_S_EMPTY + 8 * stage))

                _mbar_wait(bar(_MBAR_O_FULL + 8 * stage), phase)
                q_head_local = my_row - token_in_group * _i32(16)
                slot = (packed_q >> _i32(24)) & _i32(15)
                output_valid = txl.local_scalar(
                    "int32", init=txl.cast(txl.And(edge < q_count, slot < topk), "int32")
                )
                partial_row = txl.local_scalar("int64", init=txl.int64(0))
                inv_sum = txl.local_scalar("float32", init=_f32(0.0))
                with txl.If(output_valid != _i32(0)), txl.Then():
                    q_abs = q_batch + q_idx
                    q_head = head_kv * _i32(GQA_RATIO) + q_head_local
                    txl.assign(
                        partial_row,
                        txl.cast(slot, "int64")
                        * txl.cast(total_q, "int64")
                        * txl.cast(num_q_heads, "int64")
                        + txl.cast(q_abs, "int64") * txl.cast(num_q_heads, "int64")
                        + txl.cast(q_head, "int64"),
                    )
                    reciprocal = txl.local_scalar("float32")
                    txl.ptx.rcp.approx.ftz.f32(reciprocal, row_sum)
                    txl.assign(
                        inv_sum,
                        txl.if_then_else(
                            txl.And(row_sum > _f32(0.0), row_sum == row_sum), reciprocal, _f32(0.0)
                        ),
                    )
                partial_base = partial_row * txl.int64(HEAD_DIM)
                output_addr = taddr + _u32(_TMEM_OUTPUT[stage]) + row_bits
                pending_scale = txl.local_scalar("float32", init=_f32(0.0))
                segment = txl.local_scalar("int32", init=_i32(0))
                with txl.While(segment < _i32(4)):
                    values = txl.alloc_local((32,), "float32")
                    _tmem_load_x32(values, 0, output_addr + txl.cast(segment * _i32(32), "uint32"))
                    segment_min = _row_min_32(values)
                    segment_max = _row_max_32(values)
                    neg_min = txl.local_scalar("float32")
                    txl.ptx.neg.ftz.f32(neg_min, segment_min)
                    residual = _max_f32(segment_max, neg_min)
                    dequant_scale = txl.local_scalar("float32", init=_f32(0.0))
                    quant_scale = txl.local_scalar("float32", init=_f32(0.0))
                    with txl.If(txl.And(residual > _f32(0.0), residual == residual)), txl.Then():
                        scaled = txl.local_scalar("float32")
                        txl.ptx.mul.ftz.f32(scaled, residual, inv_sum)
                        txl.ptx.mul.ftz.f32(dequant_scale, scaled, _f32(0.002232142857142857))
                        txl.ptx["div.approx.ftz.f32"](quant_scale, _f32(448.0), residual)
                    with txl.If(output_valid != _i32(0)), txl.Then():
                        with txl.If((segment & _i32(1)) == _i32(0)):
                            with txl.Then():
                                txl.assign(pending_scale, dequant_scale)
                            with txl.Else():
                                scale_word = txl.local_scalar("uint32")
                                txl.ptx.cvt.rn.bf16x2.f32(scale_word, dequant_scale, pending_scale)
                                txl.ptx.st.global_.b32(
                                    partial_scale.ptr_to(
                                        [
                                            partial_row * txl.int64(4)
                                            + txl.cast(segment - _i32(1), "int64")
                                        ]
                                    ),
                                    scale_word,
                                )
                        for pair in range(16):
                            _packed_mul(values, 2 * pair, quant_scale)
                        for half in range(2):
                            words = txl.alloc_local((4,), "uint32")
                            for word in range(4):
                                base = 16 * half + 4 * word
                                _pack_fp8x4(
                                    words[word],
                                    values[base],
                                    values[base + 1],
                                    values[base + 2],
                                    values[base + 3],
                                )
                            txl.ptx.st.global_.v4.b32(
                                partial_o.ptr_to(
                                    [
                                        partial_base
                                        + txl.cast(segment, "int64") * txl.int64(32)
                                        + txl.int64(16 * half)
                                    ]
                                ),
                                words[0],
                                words[1],
                                words[2],
                                words[3],
                            )
                    txl.assign(segment, segment + _i32(1))

                with txl.If(output_valid != _i32(0)), txl.Then():
                    log2_sum = txl.local_scalar("float32")
                    txl.ptx.lg2.approx.ftz.f32(log2_sum, row_sum)
                    max_scaled = txl.local_scalar("float32")
                    log_scaled = txl.local_scalar("float32")
                    candidate = txl.local_scalar("float32")
                    txl.ptx.mul.ftz.f32(max_scaled, row_max, softmax_scale_log2)
                    txl.ptx.mul.ftz.f32(log_scaled, log2_sum, _f32(_LN2))
                    txl.ptx.fma.rn.ftz.f32(candidate, max_scaled, _f32(_LN2), log_scaled)
                    lse_value = txl.if_then_else(row_sum > _f32(0.0), candidate, _f32(_NEG_INF))
                    txl.ptx.st.global_.b32(partial_lse.ptr_to([partial_row]), lse_value)
                    with txl.If(return_temperature_lse != _i32(0)), txl.Then():
                        txl.ptx.st.global_.b32(
                            partial_temperature_lse.ptr_to([partial_row]), lse_value
                        )
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                _mbar_arrive(bar(_MBAR_O_EMPTY + 8 * stage))
                txl.assign(iteration, iteration + _i32(1))

        def issue_qk(stage):
            a_lo = txl.local_scalar(
                "uint32", init=txl.uniform((bar(_SMEM_Q + stage * 32768) >> _u32(4)) & _u32(0x3FFF))
            )
            b_lo = txl.local_scalar(
                "uint32", init=txl.uniform((bar(_SMEM_K) >> _u32(4)) & _u32(0x3FFF))
            )
            leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
            for k16 in range(8):
                txl.ptx[_MMA_F16](
                    taddr + _u32(_TMEM_SCORE[stage]),
                    _pack2(a_lo, _u32(_DESC_HI)),
                    _pack2(b_lo, _u32(_DESC_HI)),
                    _u32(_QK_IDESC),
                    _u32(0),
                    _u32(0),
                    _u32(0),
                    _u32(0),
                    txl.ptx.pred(_u32(0 if k16 == 0 else 1)),
                    pred=leader,
                )
                if k16 < 7:
                    txl.assign(a_lo, a_lo + _u32(_QK_STEPS[k16]))
                    txl.assign(b_lo, b_lo + _u32(_QK_STEPS[k16]))

        def issue_pv(stage):
            v_lo = txl.local_scalar(
                "uint32",
                init=txl.uniform(((bar(_SMEM_V) >> _u32(4)) & _u32(0x3FFF)) | _u32(_V_LBO_BIT)),
            )
            first_leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
            for k16 in range(4):
                txl.ptx[_MMA_F16](
                    taddr + _u32(_TMEM_OUTPUT[stage]),
                    taddr + _u32(_TMEM_SCORE[stage] + _TMEM_P_OFFSET + 8 * k16),
                    _pack2(v_lo + _u32(128 * k16), _u32(_DESC_HI)),
                    _u32(_PV_IDESC),
                    _u32(0),
                    _u32(0),
                    _u32(0),
                    _u32(0),
                    txl.ptx.pred(_u32(0 if k16 == 0 else 1)),
                    pred=first_leader,
                )
            second_leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
            for k16 in range(4):
                txl.ptx[_MMA_F16](
                    taddr + _u32(_TMEM_OUTPUT[stage]),
                    taddr + _u32(_TMEM_SCORE[stage] + _TMEM_P_OFFSET + 32 + 8 * k16),
                    _pack2(v_lo + _u32(512 + 128 * k16), _u32(_DESC_HI)),
                    _u32(_PV_IDESC),
                    _u32(0),
                    _u32(0),
                    _u32(0),
                    _u32(0),
                    txl.ptx.pred(_u32(1)),
                    pred=second_leader,
                )

        sp = txl.specialize(chain_dispatch=False)
        r_even = sp.role("softmax_even", warps=range(0, 4), regs=200)
        r_odd = sp.role("softmax_odd", warps=range(4, 8), regs=200)
        r_qload = sp.role("qload", warps=range(8, 12), regs=64)
        other_regs = sp.register_scope("other", warps=range(12, 16), regs=48)
        r_mma = sp.role("mma", warps=[12], register_scope=other_regs)
        r_transform = sp.role("transform", warps=[13, 14], register_scope=other_regs)
        r_load = sp.role("load", warps=[15], register_scope=other_regs)

        # Match the source ordering: warps 12..15 relinquish registers before
        # any softmax or Q-load warp requests its larger register allocation.
        with txl.If(txl.And(warp >= _i32(12), warp <= _i32(15))), txl.Then():
            other_regs.emit()

        with r_even:
            emit_softmax(0, 0)

        with r_odd:
            emit_softmax(1, 4)

        with r_qload:
            head_kv, row_start, q_count, _, _, q_batch, _, _, _ = decode_work()
            group_count = (q_count + _i32(7)) // _i32(8)
            group = txl.local_scalar("int32", init=_i32(0))
            with txl.While(group < group_count):
                stage = group & _i32(1)
                phase = (group // _i32(2)) & _i32(1)
                _mbar_wait(bar(_MBAR_Q_EMPTY) + txl.cast(stage * _i32(8), "uint32"), phase ^ _i32(1))
                leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
                with txl.If(leader != _u32(0)), txl.Then():
                    q_full_addr = bar(_MBAR_Q_FULL) + txl.cast(stage * _i32(8), "uint32")
                    _mbar_expect_tx(q_full_addr, 8192)
                    for local_token in range(2):
                        token = (warp - _i32(8)) * _i32(2) + _i32(local_token)
                        edge = group * _i32(8) + token
                        edge_valid = txl.local_scalar("int32", init=txl.cast(edge < q_count, "int32"))
                        safe_edge = txl.if_then_else(edge_valid != _i32(0), edge, _i32(0))
                        packed = _ld_global_i32(
                            k2q_qsplit_indices, head_kv * nnz_per_head + row_start + safe_edge
                        )
                        decoded = q_batch + (packed & _i32(0xFFFFFF))
                        q_abs = txl.if_then_else(edge_valid != _i32(0), decoded, _i32(0))
                        dst = bar(_SMEM_Q) + txl.cast(
                            stage * _i32(32768) + token * _i32(2048), "uint32"
                        )
                        txl.ptx[_TMA_G2S_4D](
                            dst,
                            txl.address_of(q_map),
                            _i32(0),
                            head_kv * _i32(GQA_RATIO),
                            _i32(0),
                            q_abs,
                            q_full_addr,
                        )
                        txl.ptx[_TMA_G2S_4D](
                            dst + _u32(16384),
                            txl.address_of(q_map),
                            _i32(0),
                            head_kv * _i32(GQA_RATIO),
                            _i32(1),
                            q_abs,
                            q_full_addr,
                        )
                txl.assign(group, group + _i32(1))

        with r_mma:
            _, _, q_count, _, _, _, _, _, _ = decode_work()
            group_count = (q_count + _i32(7)) // _i32(8)
            _mbar_wait(bar(_MBAR_K_FULL), _i32(0))
            _mbar_wait(bar(_MBAR_Q_FULL), _i32(0))
            _mbar_wait(bar(_MBAR_S_EMPTY), _i32(1))
            issue_qk(0)
            _commit(bar(_MBAR_S_FULL))
            _commit(bar(_MBAR_Q_EMPTY))
            with txl.If(group_count > _i32(1)), txl.Then():
                _mbar_wait(bar(_MBAR_Q_FULL + 8), _i32(0))
                _mbar_wait(bar(_MBAR_S_EMPTY + 8), _i32(1))
                issue_qk(1)
                _commit(bar(_MBAR_S_FULL + 8))
                _commit(bar(_MBAR_Q_EMPTY + 8))
            _mbar_wait(bar(_MBAR_V_FULL), _i32(0))

            group = txl.local_scalar("int32", init=_i32(2))
            with txl.While(group < group_count):
                pv_group = group - _i32(2)
                stage = pv_group & _i32(1)
                phase = (pv_group // _i32(2)) & _i32(1)
                _mbar_wait(bar(_MBAR_P_FULL) + txl.cast(stage * _i32(8), "uint32"), phase)
                _mbar_wait(bar(_MBAR_O_EMPTY) + txl.cast(stage * _i32(8), "uint32"), phase ^ _i32(1))
                # The source issues the two PV halves on opposite sides of P-full-2.
                v_lo = txl.local_scalar(
                    "uint32",
                    init=txl.uniform(((bar(_SMEM_V) >> _u32(4)) & _u32(0x3FFF)) | _u32(_V_LBO_BIT)),
                )
                first_leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
                for k16 in range(4):
                    txl.ptx[_MMA_F16](
                        taddr + _u32(256) + txl.cast(stage * _i32(128), "uint32"),
                        taddr + _u32(64) + txl.cast(stage * _i32(128), "uint32") + _u32(8 * k16),
                        _pack2(v_lo + _u32(128 * k16), _u32(_DESC_HI)),
                        _u32(_PV_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        txl.ptx.pred(_u32(0 if k16 == 0 else 1)),
                        pred=first_leader,
                    )
                _mbar_wait(bar(_MBAR_P_FULL_2) + txl.cast(stage * _i32(8), "uint32"), phase)
                second_leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
                for k16 in range(4):
                    txl.ptx[_MMA_F16](
                        taddr + _u32(256) + txl.cast(stage * _i32(128), "uint32"),
                        taddr + _u32(96) + txl.cast(stage * _i32(128), "uint32") + _u32(8 * k16),
                        _pack2(v_lo + _u32(512 + 128 * k16), _u32(_DESC_HI)),
                        _u32(_PV_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        txl.ptx.pred(_u32(1)),
                        pred=second_leader,
                    )
                _commit(bar(_MBAR_O_FULL) + txl.cast(stage * _i32(8), "uint32"))
                _commit(bar(_MBAR_P_EMPTY) + txl.cast(stage * _i32(8), "uint32"))

                q_stage = group & _i32(1)
                q_phase = (group // _i32(2)) & _i32(1)
                _mbar_wait(bar(_MBAR_Q_FULL) + txl.cast(q_stage * _i32(8), "uint32"), q_phase)
                _mbar_wait(
                    bar(_MBAR_S_EMPTY) + txl.cast(q_stage * _i32(8), "uint32"), q_phase ^ _i32(1)
                )
                # Runtime stages need the same descriptor walk as the two explicit primes.
                a_lo = txl.local_scalar(
                    "uint32",
                    init=txl.uniform(
                        ((bar(_SMEM_Q) >> _u32(4)) & _u32(0x3FFF))
                        + txl.cast(q_stage, "uint32") * _u32(2048)
                    ),
                )
                b_lo = txl.local_scalar(
                    "uint32", init=txl.uniform((bar(_SMEM_K) >> _u32(4)) & _u32(0x3FFF))
                )
                qk_leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
                for k16 in range(8):
                    txl.ptx[_MMA_F16](
                        taddr + txl.cast(q_stage, "uint32") * _u32(128),
                        _pack2(a_lo, _u32(_DESC_HI)),
                        _pack2(b_lo, _u32(_DESC_HI)),
                        _u32(_QK_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        txl.ptx.pred(_u32(0 if k16 == 0 else 1)),
                        pred=qk_leader,
                    )
                    if k16 < 7:
                        txl.assign(a_lo, a_lo + _u32(_QK_STEPS[k16]))
                        txl.assign(b_lo, b_lo + _u32(_QK_STEPS[k16]))
                _commit(bar(_MBAR_S_FULL) + txl.cast(q_stage * _i32(8), "uint32"))
                _commit(bar(_MBAR_Q_EMPTY) + txl.cast(q_stage * _i32(8), "uint32"))
                txl.assign(group, group + _i32(1))

            drain_start = txl.local_scalar(
                "int32", init=txl.if_then_else(group_count == _i32(1), _i32(0), group_count - _i32(2))
            )
            pv_group = txl.local_scalar("int32", init=drain_start)
            with txl.While(pv_group < group_count):
                stage = pv_group & _i32(1)
                phase = (pv_group // _i32(2)) & _i32(1)
                _mbar_wait(bar(_MBAR_P_FULL) + txl.cast(stage * _i32(8), "uint32"), phase)
                _mbar_wait(bar(_MBAR_O_EMPTY) + txl.cast(stage * _i32(8), "uint32"), phase ^ _i32(1))
                v_lo = txl.local_scalar(
                    "uint32",
                    init=txl.uniform(((bar(_SMEM_V) >> _u32(4)) & _u32(0x3FFF)) | _u32(_V_LBO_BIT)),
                )
                first_leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
                for k16 in range(4):
                    txl.ptx[_MMA_F16](
                        taddr + _u32(256) + txl.cast(stage, "uint32") * _u32(128),
                        taddr + _u32(64) + txl.cast(stage, "uint32") * _u32(128) + _u32(8 * k16),
                        _pack2(v_lo + _u32(128 * k16), _u32(_DESC_HI)),
                        _u32(_PV_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        txl.ptx.pred(_u32(0 if k16 == 0 else 1)),
                        pred=first_leader,
                    )
                _mbar_wait(bar(_MBAR_P_FULL_2) + txl.cast(stage * _i32(8), "uint32"), phase)
                second_leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
                for k16 in range(4):
                    txl.ptx[_MMA_F16](
                        taddr + _u32(256) + txl.cast(stage, "uint32") * _u32(128),
                        taddr + _u32(96) + txl.cast(stage, "uint32") * _u32(128) + _u32(8 * k16),
                        _pack2(v_lo + _u32(512 + 128 * k16), _u32(_DESC_HI)),
                        _u32(_PV_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        txl.ptx.pred(_u32(1)),
                        pred=second_leader,
                    )
                _commit(bar(_MBAR_O_FULL) + txl.cast(stage * _i32(8), "uint32"))
                _commit(bar(_MBAR_P_EMPTY) + txl.cast(stage * _i32(8), "uint32"))
                txl.assign(pv_group, pv_group + _i32(1))

            completed = txl.local_scalar("int32", init=drain_start)
            with txl.While(completed < group_count):
                stage = completed & _i32(1)
                phase = (completed // _i32(2)) & _i32(1)
                _mbar_wait(bar(_MBAR_O_EMPTY) + txl.cast(stage * _i32(8), "uint32"), phase)
                txl.assign(completed, completed + _i32(1))
            dealloc_addr = txl.local_scalar("uint32")
            txl.ptx.ld.volatile.shared.b32(dealloc_addr, bar(_SMEM_TMEM_MAILBOX))
            txl.ptx[_TMEM_DEALLOC](dealloc_addr, _u32(TMEM_COLS))

        with r_transform:
            decode_work()

        with r_load:
            launch_leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
            with txl.If(launch_leader != _u32(0)), txl.Then():
                txl.ptx.griddepcontrol.launch_dependents()
            head_kv, _, _, batch, kv_block, _, _, _, _ = decode_work()
            physical_page = _ld_global_i32(page_table, batch * max_pages + kv_block)
            with txl.If(physical_page < _i32(0)), txl.Then():
                txl.assign(physical_page, _i32(0))
            page_head = physical_page * num_kv_heads + head_kv
            load_leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
            with txl.If(load_leader != _u32(0)), txl.Then():
                _mbar_expect_tx(bar(_MBAR_K_FULL), 32768)
                for dim_half in range(2):
                    for token_half in range(2):
                        txl.ptx[_TMA_G2S_4D_L2](
                            bar(_SMEM_K + dim_half * 16384 + token_half * 8192),
                            txl.address_of(k_map),
                            _i32(0),
                            _i32(token_half * 64),
                            _i32(dim_half),
                            page_head,
                            bar(_MBAR_K_FULL),
                            txl.uint64(0x12F0000000000000),
                        )
                _mbar_expect_tx(bar(_MBAR_V_FULL), 32768)
                for dim_half in range(2):
                    for token_half in range(2):
                        txl.ptx[_TMA_G2S_4D_L2](
                            bar(_SMEM_V + dim_half * 16384 + token_half * 8192),
                            txl.address_of(v_map),
                            _i32(0),
                            _i32(token_half * 64),
                            _i32(dim_half),
                            page_head,
                            bar(_MBAR_V_FULL),
                            txl.uint64(0x12F0000000000000),
                        )

    return blackwell_msa_long_prefill_paged_bf16_gqa16_direct_group_sm100


@lru_cache(maxsize=1)
def _kernel():
    return _build_kernel()


def get_kernel(**config: Any):
    if config:
        _validate_config(_without_label(config))
    return _kernel().func


def _build_long_prefill_plan(q2k_indices, *, total_k: int, num_sms: int) -> dict[str, Any]:
    """Local copy of FlashInfer's exact GQA16 paged long-prefill decomposition."""
    import torch

    num_kv_heads, total_q, topk = (int(value) for value in q2k_indices.shape)
    if topk != TOPK:
        raise ValueError("long prefill requires topk=16")
    total_rows = (total_k + PAGE_SIZE - 1) // PAGE_SIZE
    nnz_per_head = total_q * topk
    q_tokens_per_group = PAGE_SIZE // GQA_RATIO
    total_groups_upper = (
        num_kv_heads * total_q * topk + q_tokens_per_group - 1
    ) // q_tokens_per_group
    target = max(1, (total_groups_upper + 2 * num_sms - 1) // (2 * num_sms))
    work_group_cap = min(128, 1 << (target - 1).bit_length())
    buckets = tuple(value for value in (128, 64, 32, 16, 8, 4, 2, 1) if value <= work_group_cap)

    row_ptr = torch.zeros(
        (num_kv_heads, total_rows + 1), dtype=torch.int32, device=q2k_indices.device
    )
    qsplit = torch.full(
        (num_kv_heads, nnz_per_head), -1, dtype=torch.int32, device=q2k_indices.device
    )
    split_counts = ((q2k_indices >= 0) & (q2k_indices < total_rows)).sum(dim=2, dtype=torch.int32)
    counts_host: list[list[int]] = []
    for head in range(num_kv_heads):
        flat = q2k_indices[head].reshape(-1)
        valid = (flat >= 0) & (flat < total_rows)
        positions = torch.nonzero(valid, as_tuple=False).flatten()
        blocks = flat.index_select(0, positions)
        counts = torch.bincount(blocks.to(torch.int64), minlength=total_rows).to(torch.int32)
        row_ptr[head, 1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
        counts_host.append([int(value) for value in counts.cpu().tolist()])
        order = torch.argsort(blocks, stable=True)
        sorted_positions = positions.index_select(0, order)
        q_indices = torch.div(sorted_positions, topk, rounding_mode="floor")
        slots = sorted_positions - q_indices * topk
        packed = q_indices.to(torch.int32) | (slots.to(torch.int32) << 24)
        packed |= (split_counts[head].index_select(0, q_indices) == 1).to(torch.int32) << 28
        qsplit[head, : packed.numel()] = packed

    work: list[tuple[int, tuple[int, int, int, int, int, int]]] = []
    for head, counts in enumerate(counts_host):
        for kv_block, row_count in enumerate(counts):
            q_begin = 0
            remaining = row_count
            for group_count in buckets:
                capacity = group_count * q_tokens_per_group
                while (remaining + q_tokens_per_group - 1) // q_tokens_per_group >= group_count:
                    q_count = min(capacity, remaining)
                    work.append((group_count, (head, kv_block, q_begin, q_count, 0, kv_block)))
                    q_begin += q_count
                    remaining -= q_count
            if remaining:
                raise AssertionError("long-prefill work decomposition failed")
    if not work:
        raise ValueError("long prefill requires at least one selected edge")
    work.sort(key=lambda item: item[0], reverse=True)
    metadata = torch.tensor(
        [entry for _group, entry in work], dtype=torch.int32, device=q2k_indices.device
    ).contiguous()
    counts_by_group = [0] * 129
    for group_count, _entry in work:
        counts_by_group[group_count] += 1
    running = 0
    end_by_group: dict[int, int] = {}
    for group_count in range(128, 1, -1):
        running += counts_by_group[group_count]
        end_by_group[group_count] = running
    return {
        "scheduler_metadata": metadata,
        "row_ptr": row_ptr.contiguous(),
        "qsplit": qsplit.contiguous(),
        "split_counts": split_counts.transpose(0, 1).contiguous(),
        "group_segment_ends": tuple(end_by_group[value] for value in (128, 64, 32, 16, 8, 4, 2)),
        "work_count": len(work),
        "total_rows": total_rows,
        "nnz_per_head": nnz_per_head,
    }


def _make_q2k_indices(config: dict[str, Any], *, total_rows: int):
    import torch

    q_len = int(config["q_len"])
    num_kv_heads = int(config["num_kv_heads"])
    selection = str(config["selection"])
    queries = torch.arange(q_len, dtype=torch.int64).view(q_len, 1)
    slots = torch.arange(TOPK, dtype=torch.int64).view(1, TOPK)
    result = torch.empty((num_kv_heads, q_len, TOPK), dtype=torch.int32)
    if selection == "balanced":
        for head in range(num_kv_heads):
            result[head] = ((queries * TOPK + slots + head * 257) % total_rows).to(torch.int32)
    elif selection == "hot_g128":
        result.zero_()
    elif selection == "bucket_ladder":
        bucket_counts = [
            bucket * (PAGE_SIZE // GQA_RATIO) for bucket in (128, 64, 32, 16, 8, 4, 2, 1)
        ]
        reserved = len(bucket_counts)
        exact = [
            torch.full((count,), total_rows - reserved + i, dtype=torch.int32)
            for i, count in enumerate(bucket_counts)
        ]
        used = sum(bucket_counts)
        remaining = q_len * TOPK - used
        if remaining < 0:
            raise ValueError("bucket_ladder needs a larger q_len")
        balanced = (
            torch.arange(remaining, dtype=torch.int64)
            .remainder(total_rows - reserved)
            .to(torch.int32)
        )
        flat = torch.cat((*exact, balanced))
        generator = torch.Generator().manual_seed(int(config["seed"]) + 701)
        flat = flat[torch.randperm(flat.numel(), generator=generator)]
        for head in range(num_kv_heads):
            result[head] = flat.roll(head * 17).view(q_len, TOPK)
    elif selection == "single_future":
        result.fill_(-1)
        next_page = queries[:, 0] // PAGE_SIZE + 1
        next_page = torch.where(next_page < total_rows, next_page, torch.full_like(next_page, -1))
        result[:, :, 0] = next_page.to(torch.int32)
    else:
        raise ValueError(f"unknown selection pattern {selection!r}")
    return result.contiguous()


_GUARD_ELEMS = 64
_BF16_INPUT_GUARD = 77.25
_I32_INPUT_GUARD = 324508639
_PARTIAL_O_FILL = 0xA5
_PARTIAL_O_GUARD = 0x5A
_PARTIAL_SCALE_FILL = 123.0
_PARTIAL_SCALE_GUARD = -77.0
_PARTIAL_LSE_FILL = 12345.25
_PARTIAL_LSE_GUARD = -54321.25
_PARTIAL_TLSE_FILL = -23456.5
_DEAD_OUT_FILL = 19.5
_DEAD_OUT_GUARD = -31.5


def _guarded_empty(shape, dtype, *, fill, guard, device):
    import torch

    elements = math.prod(shape)
    storage = torch.full((elements + 2 * _GUARD_ELEMS,), guard, dtype=dtype, device=device)
    view = storage[_GUARD_ELEMS : _GUARD_ELEMS + elements].view(shape)
    view.fill_(fill)
    return view, storage


def _guarded_copy(tensor, *, guard):
    import torch

    storage = torch.full(
        (tensor.numel() + 2 * _GUARD_ELEMS,), guard, dtype=tensor.dtype, device=tensor.device
    )
    view = storage[_GUARD_ELEMS : _GUARD_ELEMS + tensor.numel()].view(tensor.shape)
    view.copy_(tensor)
    return view, storage


def _outputs(total_q: int, num_q_heads: int, *, return_tlse: bool, device):
    import torch

    partial_o, partial_o_storage = _guarded_empty(
        (TOPK, total_q, num_q_heads, HEAD_DIM),
        torch.uint8,
        fill=_PARTIAL_O_FILL,
        guard=_PARTIAL_O_GUARD,
        device=device,
    )
    partial_scale, partial_scale_storage = _guarded_empty(
        (TOPK, total_q, num_q_heads, 4),
        torch.bfloat16,
        fill=_PARTIAL_SCALE_FILL,
        guard=_PARTIAL_SCALE_GUARD,
        device=device,
    )
    partial_lse, partial_lse_storage = _guarded_empty(
        (TOPK, total_q, num_q_heads),
        torch.float32,
        fill=_PARTIAL_LSE_FILL,
        guard=_PARTIAL_LSE_GUARD,
        device=device,
    )
    partial_tlse, partial_tlse_storage = _guarded_empty(
        (TOPK, total_q, num_q_heads),
        torch.float32,
        fill=_PARTIAL_TLSE_FILL,
        guard=_PARTIAL_LSE_GUARD,
        device=device,
    )
    dead_out, dead_out_storage = _guarded_empty(
        (1,), torch.bfloat16, fill=_DEAD_OUT_FILL, guard=_DEAD_OUT_GUARD, device=device
    )
    return {
        "partial_o": partial_o,
        "partial_scale": partial_scale,
        "partial_lse": partial_lse,
        "partial_tlse": partial_tlse,
        "partial_tlse_arg": partial_tlse if return_tlse else partial_lse,
        "out": dead_out,
        "guards": {
            "partial_o": (partial_o_storage, _PARTIAL_O_GUARD),
            "partial_scale": (partial_scale_storage, _PARTIAL_SCALE_GUARD),
            "partial_lse": (partial_lse_storage, _PARTIAL_LSE_GUARD),
            "partial_tlse": (partial_tlse_storage, _PARTIAL_LSE_GUARD),
            "out": (dead_out_storage, _DEAD_OUT_GUARD),
        },
    }


def prepare_data(**config: Any) -> dict[str, Any]:
    """Build deterministic direct-route inputs, exact plan, and guarded producer outputs."""
    import torch

    config = _without_label(config)
    _validate_config(config)
    q_len = int(config["q_len"])
    kv_len = int(config["kv_len"])
    hq = int(config["num_q_heads"])
    hkv = int(config["num_kv_heads"])
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]))
    physical_pages = MAX_PAGES

    q_raw = torch.randn(
        (q_len, hq, HEAD_DIM), dtype=torch.bfloat16, device=device, generator=generator
    )
    k_raw = torch.randn(
        (physical_pages, hkv, PAGE_SIZE, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    v_raw = torch.randn(
        (physical_pages, hkv, PAGE_SIZE, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    q, q_storage = _guarded_copy(q_raw, guard=_BF16_INPUT_GUARD)
    k, k_storage = _guarded_copy(k_raw, guard=_BF16_INPUT_GUARD)
    v, v_storage = _guarded_copy(v_raw, guard=_BF16_INPUT_GUARD)
    del q_raw, k_raw, v_raw

    q2k_raw = _make_q2k_indices(config, total_rows=MAX_PAGES)
    q2k_indices, q2k_storage = _guarded_copy(q2k_raw.to(device), guard=_I32_INPUT_GUARD)
    del q2k_raw
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    plan = _build_long_prefill_plan(q2k_indices, total_k=MAX_PAGES * PAGE_SIZE, num_sms=num_sms)
    for key in ("scheduler_metadata", "row_ptr", "qsplit"):
        guarded, storage = _guarded_copy(plan[key], guard=_I32_INPUT_GUARD)
        plan[key] = guarded
        plan[f"{key}_storage"] = storage

    page_table_cpu = torch.arange(physical_pages, dtype=torch.int32)
    if config["page_layout"] == "permuted":
        cpu_generator = torch.Generator().manual_seed(int(config["seed"]) + 991)
        page_table_cpu = page_table_cpu[torch.randperm(physical_pages, generator=cpu_generator)]
    page_table_raw = page_table_cpu.view(1, physical_pages).to(device)
    page_table, page_table_storage = _guarded_copy(page_table_raw, guard=_I32_INPUT_GUARD)
    cu_q, cu_q_storage = _guarded_copy(
        torch.tensor((0, q_len), dtype=torch.int32, device=device), guard=_I32_INPUT_GUARD
    )
    cu_k, cu_k_storage = _guarded_copy(
        torch.tensor((0, kv_len), dtype=torch.int32, device=device), guard=_I32_INPUT_GUARD
    )
    q_offsets, q_offsets_storage = _guarded_copy(
        torch.tensor((0,), dtype=torch.int32, device=device), guard=_I32_INPUT_GUARD
    )
    kv_lens, kv_lens_storage = _guarded_copy(
        torch.tensor((kv_len,), dtype=torch.int32, device=device), guard=_I32_INPUT_GUARD
    )
    return_tlse = bool(config["return_temperature_lse"])
    input_guards = {
        "q": (q_storage, _BF16_INPUT_GUARD),
        "k": (k_storage, _BF16_INPUT_GUARD),
        "v": (v_storage, _BF16_INPUT_GUARD),
        "q2k_indices": (q2k_storage, _I32_INPUT_GUARD),
        "scheduler_metadata": (plan["scheduler_metadata_storage"], _I32_INPUT_GUARD),
        "row_ptr": (plan["row_ptr_storage"], _I32_INPUT_GUARD),
        "qsplit": (plan["qsplit_storage"], _I32_INPUT_GUARD),
        "page_table": (page_table_storage, _I32_INPUT_GUARD),
        "cu_q": (cu_q_storage, _I32_INPUT_GUARD),
        "cu_k": (cu_k_storage, _I32_INPUT_GUARD),
        "q_offsets": (q_offsets_storage, _I32_INPUT_GUARD),
        "kv_lens": (kv_lens_storage, _I32_INPUT_GUARD),
    }
    return {
        "config": config,
        "q": q,
        "k": k,
        "v": v,
        "q2k_indices": q2k_indices,
        "page_table": page_table,
        "cu_q": cu_q,
        "cu_k": cu_k,
        "q_offsets": q_offsets,
        "kv_lens": kv_lens,
        "plan": plan,
        "total_q": q_len,
        "num_q_heads": hq,
        "num_kv_heads": hkv,
        "max_pages": MAX_PAGES,
        "softmax_scale_log2": (HEAD_DIM**-0.5) / math.log(2.0),
        "lse_temperature_scale": 1.0,
        "return_temperature_lse": return_tlse,
        "tirx": _outputs(q_len, hq, return_tlse=return_tlse, device=device),
        "source": _outputs(q_len, hq, return_tlse=return_tlse, device=device),
        "input_guards": input_guards,
    }


def _forward_scalars(data: dict[str, Any]) -> tuple[Any, ...]:
    plan = data["plan"]
    return (
        *plan["group_segment_ends"],
        data["total_q"],
        data["num_q_heads"],
        data["num_kv_heads"],
        int(plan["total_rows"]),
        int(plan["nnz_per_head"]),
        int(plan["work_count"]),
        int(plan["work_count"]),
        TOPK,
        data["max_pages"],
        1,
        1,
        data["softmax_scale_log2"],
        data["lse_temperature_scale"],
        int(data["return_temperature_lse"]),
    )


def _forward_tensors(data: dict[str, Any], slot: str) -> tuple[Any, ...]:
    plan = data["plan"]
    buffers = data[slot]
    return (
        data["q"],
        data["k"],
        data["v"],
        plan["scheduler_metadata"],
        plan["row_ptr"],
        plan["qsplit"],
        buffers["partial_o"],
        buffers["partial_scale"],
        buffers["partial_lse"],
        buffers["partial_tlse_arg"],
        buffers["out"],
        data["cu_q"],
        data["cu_k"],
        data["q_offsets"],
        data["kv_lens"],
        data["page_table"],
    )


@lru_cache(maxsize=1)
def _compiled_kernel():
    from tirx_kernels.runner import compile_kernel

    # CUDA 13.2 emits PTX ISA 9.2 for sm_100a, matching the pinned source module.
    # Level 2 is the measured winner for this zero-spill, 128-register kernel;
    # restore the caller's compiler environment exactly after compilation.
    previous = os.environ.get("TVM_CUDA_PTXAS_REG_LEVEL")
    os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = _PTXAS_REG_LEVEL
    try:
        return compile_kernel(get_kernel(), cuda_compile_mode="nvcc")
    finally:
        if previous is None:
            os.environ.pop("TVM_CUDA_PTXAS_REG_LEVEL", None)
        else:
            os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = previous


def _tirx_launch(executable, data: dict[str, Any]):
    tensors = _forward_tensors(data, "tirx")
    arguments = (
        *tensors[:3],
        *(tensor.view(-1) for tensor in tensors[3:]),
        *_forward_scalars(data),
    )

    def launch():
        executable(*arguments)

    launch._keep_alive = arguments
    return launch


@lru_cache(maxsize=1)
def _source_module():
    from flashinfer.msa_ops import _blackwell_sm100

    return _blackwell_sm100._get_module(
        "long_prefill_paged_bf16_gqa16_direct_group_sm100", "sm100a"
    )


def _source_launch(data: dict[str, Any]):
    import torch

    module = _source_module()
    arguments = (
        *_forward_tensors(data, "source"),
        *_forward_scalars(data),
        int(data["plan"]["work_count"]),
        1,
        1,
        int(torch.cuda.current_stream(data["q"].device).cuda_stream),
    )

    def launch():
        module.run(*arguments)

    launch._keep_alive = (module, arguments)
    return launch


def _assert_direct_public_route(data: dict[str, Any]) -> None:
    from flashinfer.msa_ops import _blackwell_sm100

    target = _blackwell_sm100._select_target(data["q"].device)
    selected = _blackwell_sm100._should_use_long_prefill(
        requested_schedule="auto",
        batch_size=1,
        total_q=data["total_q"],
        paged=True,
        group_size=GQA_RATIO,
        max_pages=data["max_pages"],
        k_outer_dim=data["k"].shape[0],
        q_dtype=data["q"].dtype,
        k_dtype=data["k"].dtype,
        v_dtype=data["v"].dtype,
        causal=True,
        q_offset_is_none=True,
        return_temperature_lse=data["return_temperature_lse"],
        lse_temperature_scale=1.0,
    )
    if target != "sm100a" or not selected or data["max_pages"] != MAX_PAGES:
        raise AssertionError(
            f"configuration does not select the direct SM100 long-prefill route: target={target}, selected={selected}"
        )


def _assert_guards(name: str, guards: dict[str, tuple[Any, Any]]) -> None:
    import torch

    for key, (storage, guard) in guards.items():
        expected = torch.full((_GUARD_ELEMS,), guard, dtype=storage.dtype, device=storage.device)
        if not torch.equal(storage[:_GUARD_ELEMS], expected):
            raise AssertionError(f"{name}.{key} prefix guard was modified")
        if not torch.equal(storage[-_GUARD_ELEMS:], expected):
            raise AssertionError(f"{name}.{key} suffix guard was modified")


def _bits(tensor):
    import torch

    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16)
    if tensor.dtype == torch.float32:
        return tensor.view(torch.int32)
    if tensor.dtype == torch.int32:
        return tensor
    raise TypeError(f"unsupported exact-compare dtype {tensor.dtype}")


def _assert_bitwise(name: str, actual, expected) -> None:
    import torch

    actual_bits = _bits(actual).reshape(-1)
    expected_bits = _bits(expected).reshape(-1)
    equal = actual_bits == expected_bits
    if bool(equal.all()):
        return
    first = int((~equal).nonzero()[0].item())
    actual_value = actual.reshape(-1)[first].item()
    expected_value = expected.reshape(-1)[first].item()
    finite = torch.isfinite(actual.float()) & torch.isfinite(expected.float())
    max_abs = (
        float((actual.float()[finite] - expected.float()[finite]).abs().max().item())
        if bool(finite.any())
        else float("nan")
    )
    mismatches = int((~equal).sum().item())
    raise AssertionError(
        f"{name} is not bitwise equal at flat index {first}: tirx={actual_value}, "
        f"expected={expected_value}, mismatches={mismatches}, max_finite_abs={max_abs}"
    )


def _validate_outputs(data: dict[str, Any]) -> dict[str, float]:
    _assert_guards("inputs", data["input_guards"])
    _assert_guards("tirx", data["tirx"]["guards"])
    _assert_guards("source", data["source"]["guards"])
    for key in ("partial_o", "partial_scale", "partial_lse", "partial_tlse", "out"):
        _assert_bitwise(key, data["tirx"][key], data["source"][key])
    return {
        "partial_o_max_abs": 0.0,
        "partial_scale_max_abs": 0.0,
        "partial_lse_max_abs": 0.0,
        "partial_temperature_lse_max_abs": 0.0,
    }


def _reset_outputs(buffers: dict[str, Any], *, return_tlse: bool) -> None:
    buffers["partial_o"].fill_(_PARTIAL_O_FILL)
    buffers["partial_scale"].fill_(_PARTIAL_SCALE_FILL)
    buffers["partial_lse"].fill_(_PARTIAL_LSE_FILL)
    buffers["partial_tlse"].fill_(_PARTIAL_TLSE_FILL)
    buffers["out"].fill_(_DEAD_OUT_FILL)
    buffers["partial_tlse_arg"] = buffers["partial_tlse"] if return_tlse else buffers["partial_lse"]


def _skip_unless_supported() -> None:
    import unittest

    import torch

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is unavailable")
    if torch.cuda.get_device_capability() != (10, 0):
        raise unittest.SkipTest("this kernel requires compute capability 10.0")


def run_test(**config: Any) -> dict[str, float]:
    import torch

    _skip_unless_supported()
    data = prepare_data(**config)
    _assert_direct_public_route(data)
    input_names = (
        "q",
        "k",
        "v",
        "q2k_indices",
        "page_table",
        "cu_q",
        "cu_k",
        "q_offsets",
        "kv_lens",
    )
    input_snapshots = {name: data[name].clone() for name in input_names}
    for name in ("scheduler_metadata", "row_ptr", "qsplit"):
        input_snapshots[name] = data["plan"][name].clone()
    executable = _compiled_kernel()
    tirx_launch = _tirx_launch(executable, data)
    source_launch = _source_launch(data)
    tirx_launch()
    source_launch()
    torch.cuda.synchronize()
    metrics = _validate_outputs(data)
    first = {
        key: data["tirx"][key].clone()
        for key in ("partial_o", "partial_scale", "partial_lse", "partial_tlse", "out")
    }
    _reset_outputs(data["tirx"], return_tlse=data["return_temperature_lse"])
    tirx_launch = _tirx_launch(executable, data)
    tirx_launch()
    torch.cuda.synchronize()
    for key, expected in first.items():
        _assert_bitwise(f"determinism.{key}", data["tirx"][key], expected)
    for name, expected in input_snapshots.items():
        actual = data["plan"][name] if name in data["plan"] else data[name]
        _assert_bitwise(f"input_immutability.{name}", actual, expected)
    _assert_guards("inputs", data["input_guards"])
    _assert_guards("tirx", data["tirx"]["guards"])
    return metrics


def prepare_bench(**config: Any):
    from tirx_kernels.runner import prepared_gpu_benchmark

    kernel_config = _without_label(config)
    _validate_config(kernel_config)
    return prepared_gpu_benchmark(
        run_gpu, {"config": kernel_config, "executable": _compiled_kernel()}
    )


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs):
    from tirx_kernels.runner import bench, defer_gpu_interrupts, external_references_enabled

    with defer_gpu_interrupts():
        import torch

    config = _without_label({**prepared["config"], **kwargs})
    with_source = external_references_enabled()
    gpu_state = prepared.get("gpu_state")
    if gpu_state is None:
        _skip_unless_supported()
        data = prepare_data(**config)
        _assert_direct_public_route(data)
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

    data = gpu_state["data"]
    if not gpu_state["validated"]:
        gpu_state["tirx_launch"]()
        torch.cuda.synchronize()
        if with_source:
            with defer_gpu_interrupts():
                gpu_state["source_launch"] = _source_launch(data)
                gpu_state["source_launch"]()
                torch.cuda.synchronize()
            _validate_outputs(data)
        gpu_state["validated"] = True
    source_launch = gpu_state["source_launch"]
    references = {"flashinfer": lambda: source_launch} if source_launch is not None else None
    return bench(
        {"tirx": gpu_state["tirx_launch"]},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **config: Any):
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
