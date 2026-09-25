# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

import ctypes
import math
import random
from dataclasses import dataclass, fields
from enum import Enum
from functools import lru_cache
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.tirx_lite as txl
import tvm
from tvm.ir import PointerType, PrimType

B_H = 64
B_TOPK = 64
D_V = 512
NUM_BUFS = 2
NUM_INDEX_BUFS = 4
NUM_THREADS = 384
BF16_BYTES = 2
MAX_INIT_VAL = -1.0e30
LOG_2_E = math.log2(math.e)
LN_2 = math.log(2.0)

# config.h's unscoped NamedBarriers enum is passed to CUTLASS's uint32_t
# user-barrier overload.  barrier.h therefore adds FirstUserBarrier (8) before
# emitting PTX; use those physical IDs rather than the logical enum values.
CUTLASS_USER_BARRIER_BASE = 8
BAR_EVERYONE_SYNC = CUTLASS_USER_BARRIER_BASE + 4
BAR_WG0_SYNC = CUTLASS_USER_BARRIER_BASE + 1
BAR_WG0_WARP02 = CUTLASS_USER_BARRIER_BASE + 2

LAUNCH_TAGS = (
    "blockIdx.x",
    "blockIdx.y",
    "blockIdx.z",
    "threadIdx.x",
    "tirx.use_dyn_shared_memory",
)
COMBINE_LAUNCH_TAGS = ("blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x")
COMBINE_PDL_LAUNCH_TAGS = (
    "blockIdx.x",
    "blockIdx.y",
    "blockIdx.z",
    "threadIdx.x",
    "tirx.use_programtic_dependent_launch",
)
MAIN_OPTIONAL_BUFFER_PARAMS = (
    "topk_length_h",
    "attn_sink_h",
    "extra_kv_h",
    "extra_indices_h",
    "extra_topk_length_h",
)
COMBINE_OPTIONAL_BUFFER_PARAMS = ("attn_sink_h",)
MainPresenceMask = tuple[bool, bool, bool, bool, bool]
MAIN_OPTIONAL_ARG_INDICES = (3, 4, 11, 12, 13)
COMBINE_OPTIONAL_ARG_INDICES = (5,)

_TMEM_LD_32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
_TMEM_LD_64 = "tcgen05.ld.sync.aligned.32x32b.x64.b32"
_TMEM_ST_64 = "tcgen05.st.sync.aligned.32x32b.x64.b32"
_TMA_G2S_4D_CACHE = (
    "cp.async.bulk.tensor.4d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
)
_TMA_G2S_5D_CACHE = (
    "cp.async.bulk.tensor.5d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
)
_TMA_GATHER4_2D_CACHE = (
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
)
_TMA_S2G_4D = "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
_TCGEN_CP_128X256 = "tcgen05.cp.cta_group::1.128x256b"
_MMA_WS_F16 = "tcgen05.mma.ws.cta_group::1.kind::f16"
_Q_TMA_CACHE_HINT = txl.uint64(0x12F0000000000000)
_KV_TMA_CACHE_HINT = txl.uint64(0x14F0000000000000)


def _tmem_load(dst, tmem_col, width):
    chain = _TMEM_LD_32 if width == 32 else _TMEM_LD_64
    return txl.ptx[chain](*[dst[i] for i in range(width)], tmem_col)


def _tmem_store(src, tmem_col, width=64):
    assert width == 64
    return txl.ptx[_TMEM_ST_64](tmem_col, *[src[i] for i in range(width)])


def _load_scaled_tmem_chunk(dst, tmem_col, scale_pair):
    """Read one 64-column TMEM chunk into ``dst`` and scale it in place.

    The scale is applied as f32x2 against a packed pair, so the 64 accumulator
    words move through 32 mul.f32x2 rather than 64 scalar multiplies.
    """
    _tmem_load(dst, tmem_col, 64)
    txl.ptx.tcgen05.wait__ld.sync.aligned()
    scaled_pair = txl.local_scalar("uint64")
    with txl.unroll(64 // 2) as scale_i:
        txl.ptx.mul.f32x2(
            scaled_pair, txl.cuda.make_float2(dst[scale_i * 2], dst[scale_i * 2 + 1]), scale_pair
        )
        txl.ptx.mov.b32(dst[scale_i * 2], txl.cuda.float2_x(scaled_pair))
        txl.ptx.mov.b32(dst[scale_i * 2 + 1], txl.cuda.float2_y(scaled_pair))


def _desc_add_16B_offset(desc, offset):
    """Add a 16-byte-unit offset to the low half of an SMEM descriptor.

    Same wrap-in-the-low-32-bits arithmetic as ``txl.SmemDescriptor``'s own
    stepper, but spelled as an expression instead of a ``mov.b64`` unpack /
    ``add.u32`` / ``mov.b64`` repack.  The inline-asm round trip is opaque to
    ptxas' uniform-datapath promotion, so every one of this kernel's 26
    per-block MMA operands paid a vector IADD3 plus an R2UR to reach the
    uniform register the MMA reads; as plain arithmetic the whole descriptor
    chain stays uniform.
    """
    low = txl.cast(txl.cast(desc, "uint32") + txl.cast(offset, "uint32"), "uint64")
    return txl.bitwise_or(txl.bitwise_and(desc, txl.bitwise_not(txl.uint64(0xFFFFFFFF))), low)


def _replace_smem_desc_addr(desc, smem_ptr):
    start_addr = txl.cast(
        txl.bitwise_and(
            txl.shift_right(txl.cuda.cvta_generic_to_shared(smem_ptr), txl.uint32(4)), txl.uint32(0x3FFF)
        ),
        "uint64",
    )
    return txl.bitwise_or(txl.bitwise_and(desc, txl.bitwise_not(txl.uint64(0x3FFF))), start_addr)


class ModelType(str, Enum):
    """The two CUDA template instances of the single head64 implementation."""

    V32 = "V32"
    MODEL1 = "MODEL1"


@dataclass(frozen=True)
class SparseFlashMLADecodeHead64Config:
    label: str
    model_type: ModelType | str
    b: int
    s_q: int
    s_kv: int
    topk: int
    page_block_size: int
    h_q: int = B_H
    h_kv: int = 1
    d_v: int = D_V
    have_attn_sink: bool = False
    have_topk_length: bool = False
    inject_invalid_indices: bool = False
    is_varlen: bool = True
    is_all_indices_invalid: bool = False
    have_zero_seqlen_k: bool = False
    extra_s_kv: int = 0
    extra_topk: int = 0
    extra_page_block_size: int = 0
    have_extra_topk_length: bool = False
    seed: int = 0

    @property
    def normalized_model_type(self) -> ModelType:
        return ModelType(self.model_type)

    @property
    def d_qk(self) -> int:
        # d_qk is deliberately derived from MODEL_TYPE, matching config.h.
        return 576 if self.normalized_model_type is ModelType.V32 else 512

    def validate(self) -> None:
        if self.h_kv != 1 or self.d_v != D_V:
            raise ValueError("head64 sparse decode requires h_kv=1 and d_v=512")
        if self.h_q == 128 and self.normalized_model_type is not ModelType.V32:
            raise ValueError("h_q=128,d_qk=512 dispatches to the out-of-scope head128 kernel")
        if self.h_q not in (B_H, 2 * B_H):
            raise ValueError("this port covers h_q=64 direct and V32 h_q=128 head64x2 dispatch")
        if self.b <= 0 or self.s_q <= 0 or self.s_kv <= 0 or self.page_block_size <= 0:
            raise ValueError("b, s_q, s_kv, and page_block_size must be positive")
        if self.topk <= 0 or self.topk % B_TOPK != 0:
            raise ValueError("topk must be a positive multiple of 64")
        if self.extra_topk % B_TOPK != 0:
            raise ValueError("extra_topk must be a multiple of 64")
        if self.extra_topk and not self.extra_s_kv:
            raise ValueError("extra_s_kv is required when extra_topk is nonzero")
        if self.extra_topk and self.extra_page_block_size <= 0:
            raise ValueError("extra_page_block_size must be positive with an extra KV cache")
        if self.have_extra_topk_length and not self.extra_topk:
            raise ValueError("extra_topk_length requires an extra KV cache")


# The MODEL1 rows sweep one batch list across two extra-KV variants: a plain
# 512-token extra cache, and a 1024-token one whose length is read per request.
_MODEL1_BATCH_SWEEP = (2, 64, 74, 128, 148, 256)
_MODEL1_EXTRA_VARIANTS = (
    (
        "xsk16384_xtopk512_xp64",
        {"extra_s_kv": 16384, "extra_topk": 512, "extra_page_block_size": 64},
    ),
    (
        "xsk16384_xtopk1024_xp2_xtopklen",
        {
            "extra_s_kv": 16384,
            "extra_topk": 1024,
            "extra_page_block_size": 2,
            "have_extra_topk_length": True,
        },
    ),
)

CONFIGS = [
    {
        "label": "deepseek_v4_v32_b128_sq2_sk32768_topk2048_p64",
        "model_type": "V32",
        "b": 128,
        "s_q": 2,
        "s_kv": 32768,
        "topk": 2048,
        "page_block_size": 64,
        "have_attn_sink": True,
    },
    *(
        {
            "label": f"model1_b{b}_sq2_sk16384_topk128_p256_{suffix}",
            "model_type": "MODEL1",
            "b": b,
            "s_q": 2,
            "s_kv": 16384,
            "topk": 128,
            "page_block_size": 256,
            "have_attn_sink": True,
            **extra,
        }
        for suffix, extra in _MODEL1_EXTRA_VARIANTS
        for b in _MODEL1_BATCH_SWEEP
    ),
    *(
        {
            "label": f"{prefix}_b148_sq2_sk32768_topk16384_p64",
            "model_type": model_type,
            "b": 148,
            "s_q": 2,
            "s_kv": 32768,
            "topk": 16384,
            "page_block_size": 64,
            "have_attn_sink": True,
        }
        for prefix, model_type in (("model1", "MODEL1"), ("v32", "V32"))
    ),
]


KERNEL_META = {
    "name": "sparse_flashmla_decode_head64",
    "category": "flashmla",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "reference_requirements": (
        {
            "package": "flash-mla",
            "git": {
                "url": "https://github.com/deepseek-ai/FlashMLA.git",
                "commit": "9241ae3ef9bac614dd25e45e507e089f888280e0",
            },
            "import": "flash_mla",
        },
    ),
}


def _cfg(**kwargs: Any) -> SparseFlashMLADecodeHead64Config:
    cfg_fields = {field.name for field in fields(SparseFlashMLADecodeHead64Config)}
    cfg_kwargs = {key: value for key, value in kwargs.items() if key in cfg_fields}
    cfg_kwargs.setdefault("label", "custom")
    cfg = SparseFlashMLADecodeHead64Config(**cfg_kwargs)
    cfg.validate()
    return cfg


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _kv_storage_spec(
    model_type: ModelType, num_blocks: int, page_block_size: int
) -> tuple[int, int, int, int]:
    """Return API bytes/token, TMA stride, block stride, and TMA rows."""

    if model_type is ModelType.V32:
        bytes_per_token = 512 + 4 * 4 + 64 * BF16_BYTES
        tma_k_stride = 656
        # tests/quant.py intentionally allocates one padding row per block.
        stride_kv_block = (page_block_size + 1) * tma_k_stride
    else:
        bytes_per_token = 448 + 64 * BF16_BYTES + 7 + 1
        tma_k_stride = 576
        stride_kv_block = _ceil_div(page_block_size * bytes_per_token, tma_k_stride) * tma_k_stride
    num_tma_rows = num_blocks * (stride_kv_block // tma_k_stride)
    return bytes_per_token, tma_k_stride, stride_kv_block, num_tma_rows


def make_main_kernel(model_type, presence, use_pdl=False):
    is_v32 = model_type is ModelType.V32
    (
        have_topk_length,
        have_attn_sink,
        have_extra_kv,
        have_extra_indices,
        have_extra_topk_length,
    ) = presence
    d_qk = 576 if is_v32 else 512
    d_nope = 512 if is_v32 else 448
    num_scales = 4 if is_v32 else 8
    tma_k_stride = 656 if is_v32 else 576
    q_tail_start = 256 if is_v32 else 224
    rope_tile = 32 if is_v32 else 64
    rows_per_group = B_TOPK // (128 // 8)
    cols_per_group = d_nope // (8 * 8)
    kv_rope_start = (d_nope + (16 if is_v32 else 0)) // BF16_BYTES
    source_smem_size = 232192 if is_v32 else 218848

    @txl.kernel(warps=12, arch="sm_100a", min_blocks_per_sm=1, grid=("s_q", "num_sm_parts", 1))
    def sparse_flashmla_decode_head64_main(
        q: txl.gptr[txl.bf16],
        kv: txl.gptr[txl.bf16],
        indices: txl.gptr[txl.i32],
        topk_length: txl.gptr[txl.i32],
        attn_sink: txl.gptr[txl.f32],
        lse: txl.gptr[txl.f32],
        out: txl.gptr[txl.bf16],
        lse_accum: txl.gptr[txl.f32],
        o_accum: txl.gptr[txl.f32],
        tile_scheduler_metadata: txl.gptr[txl.i32],
        num_splits: txl.gptr[txl.i32],
        extra_kv: txl.gptr[txl.bf16],
        extra_indices: txl.gptr[txl.i32],
        extra_topk_length: txl.gptr[txl.i32],
        kv_rope_tensormap: txl.TensorMap,
        kv_nope_tensormap: txl.TensorMap,
        extra_kv_rope_tensormap: txl.TensorMap,
        extra_kv_nope_tensormap: txl.TensorMap,
        q_strided_tensormap: txl.TensorMap,
        q_tail_tensormap: txl.TensorMap,
        out_tensormap: txl.TensorMap,
        sm_scale_div_log2: txl.f32,
        stride_q_b: txl.i32,
        stride_q_s_q: txl.i32,
        stride_q_h_q: txl.i32,
        stride_kv_block: txl.i32,
        stride_kv_row: txl.i32,
        stride_indices_b: txl.i32,
        stride_indices_s_q: txl.i32,
        stride_lse_b: txl.i32,
        stride_lse_s_q: txl.i32,
        stride_o_b: txl.i32,
        stride_o_s_q: txl.i32,
        stride_o_h_q: txl.i32,
        stride_extra_kv_block: txl.i32,
        stride_extra_kv_row: txl.i32,
        tma_coords_step_per_block: txl.i32,
        tma_coords_step_per_extra_block: txl.i32,
        stride_extra_indices_b: txl.i32,
        stride_extra_indices_s_q: txl.i32,
        stride_lse_accum_split: txl.i32,
        stride_lse_accum_s_q: txl.i32,
        stride_o_accum_split: txl.i32,
        stride_o_accum_s_q: txl.i32,
        stride_o_accum_h_q: txl.i32,
        b: txl.i32,
        s_q: txl.i32,
        topk: txl.i32,
        extra_topk: txl.i32,
        num_blocks: txl.i32,
        extra_num_blocks: txl.i32,
        page_block_size: txl.i32,
        extra_page_block_size: txl.i32,
        num_sm_parts: txl.i32,
    ):
        s_q_idx, partition_idx, _ = txl.cta_id()
        warp_idx = txl.warp_id()
        lane_idx = txl.lane_id()
        idx_in_warpgroup = txl.thread_id_in_wg([128])
        with txl.If(warp_idx == 0), txl.Then():
            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                for _prefetch_i in range(8):
                    txl.ptx.prefetch.tensormap(txl.address_of(out_tensormap))
                txl.ptx.prefetch.tensormap(txl.address_of(q_strided_tensormap))
                if is_v32:
                    txl.ptx.prefetch.tensormap(txl.address_of(q_tail_tensormap))
                txl.ptx.prefetch.tensormap(txl.address_of(kv_nope_tensormap))
                txl.ptx.prefetch.tensormap(txl.address_of(kv_rope_tensormap))
        smem = txl.smem_pool()
        pool = smem.pool
        u_base = pool.offset
        if is_v32:
            k_union = smem.alloc((NUM_BUFS, B_TOPK, D_V + 64), "bfloat16", swizzle=txl.SW128B).buf
            k_union_end = pool.offset
            k_full = k_union
            pool.move_base_to(u_base)
            k_rope = smem.alloc((NUM_BUFS, B_TOPK, D_V + 64), "bfloat16", swizzle=txl.SW64B).buf
            pool.move_base_to(k_union_end)
        else:
            k_full = smem.alloc((NUM_BUFS, B_TOPK, D_V), "bfloat16", swizzle=txl.SW128B).buf
            k_union_end = pool.offset
            pool.move_base_to(u_base)
            k_rope = smem.alloc((NUM_BUFS, B_TOPK, 64), "bfloat16", swizzle=txl.SW128B).buf
            pool.move_base_to(k_union_end)
        raw_nope = pool.alloc((NUM_BUFS, B_TOPK, d_nope // 8), "uint64", align=1024)
        kv_union_end = pool.offset
        pool.move_base_to(u_base)
        q_sw128 = smem.alloc((B_H, 512), "bfloat16", swizzle=txl.SW128B).buf
        q_sw128_end = pool.offset
        if is_v32:
            pool.move_base_to(q_sw128_end)
            q_sw64 = smem.alloc((B_H, 64), "bfloat16", swizzle=txl.SW64B).buf
        o_union_base = pool.offset
        o_smem = smem.alloc((B_H, D_V), "bfloat16", swizzle=txl.SW128B).buf
        o_bf16_end = pool.offset
        pool.move_base_to(o_union_base)
        o_accum_storage = pool.alloc(((B_H - 1) * (D_V + 8) + D_V,), "float32", align=1024)
        qo_union_end = pool.offset
        pool.move_base_to(max(kv_union_end, qo_union_end, o_bf16_end))
        sp_union_base = pool.offset
        p_exchange = pool.alloc((4, 32 * (B_TOPK // 2)), "float32", align=16)
        sp_union_end = pool.offset
        pool.move_base_to(sp_union_base)
        s_smem_gemm = smem.alloc((B_H, B_TOPK), "bfloat16", align=1024)
        pool.move_base_to(sp_union_end)
        pv_b_lo_desc = txl.SmemDescriptor()
        pv_b_lo_desc.init(k_full.ptr_to([0, 0, 0]), ldo=512, sdo=64, swizzle=3)
        pv_b_hi_desc = txl.SmemDescriptor()
        pv_b_hi_desc.init(k_full.ptr_to([0, 0, 0]), ldo=512, sdo=64, swizzle=3)
        pv_a_lo_desc = txl.SmemDescriptor()
        pv_a_lo_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
        pv_a_hi_desc = txl.SmemDescriptor()
        pv_a_hi_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
        q_main_cp_desc = txl.local_scalar("uint64")
        txl.cuda.tcgen05.encode_matrix_descriptor(
            txl.address_of(q_main_cp_desc), txl.reinterpret(txl.handle().ty, txl.uint64(0)), 1, 64, 3
        )
        if is_v32:
            q_tail_cp_desc = txl.local_scalar("uint64")
            txl.cuda.tcgen05.encode_matrix_descriptor(
                txl.address_of(q_tail_cp_desc), txl.reinterpret(txl.handle().ty, txl.uint64(0)), 1, 32, 2
            )
        rowwise_buf = pool.alloc((128,), "float32", align=16)
        is_token_valid = pool.alloc((NUM_INDEX_BUFS, B_TOPK // 8), "int8", align=16)
        tma_coord = pool.alloc((NUM_INDEX_BUFS, B_TOPK), "int32", align=16)
        scales_e8m0 = pool.alloc((NUM_INDEX_BUFS, B_TOPK * num_scales), "uint8", align=16)
        tmem_start_addr = pool.alloc((4,), "uint32", align=16)
        bar_last_store_done = txl.MBarrier(pool, 1)
        bar_q_tma = txl.TMABar(pool, 1)
        bar_q_utccp = txl.TCGen05Bar(pool, 1)
        bar_rope_ready = txl.TMABar(pool, NUM_BUFS)
        bar_nope_ready = txl.MBarrier(pool, NUM_BUFS)
        bar_raw_ready = txl.TMABar(pool, NUM_BUFS)
        bar_raw_free = txl.MBarrier(pool, NUM_BUFS)
        bar_valid_ready = txl.MBarrier(pool, NUM_INDEX_BUFS)
        bar_valid_free = txl.MBarrier(pool, NUM_INDEX_BUFS)
        bar_qk_done = txl.TCGen05Bar(pool, NUM_BUFS)
        bar_so_ready = txl.MBarrier(pool, NUM_BUFS)
        bar_sv_done = txl.TCGen05Bar(pool, NUM_BUFS)
        smem.commit(size=source_smem_size)

        def load_scheduler_meta(dst):
            # kernel.cuh:80-88 / KU_LDG_256.  Keep one 32-byte operation,
            # including its cache operators and L2 prefetch size; the eighth
            # int32 word is intentionally loaded even though it is reserved.
            txl.ptx["ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v4.u64"](
                dst[0], dst[1], dst[2], dst[3], tile_scheduler_metadata.ptr_to([partition_idx * 8])
            )

        def unpack_scheduler_meta(word_count):
            # The roles that never take the split path stop after the block
            # range; only the scale/exp role needs the three split words.
            sched_words = txl.alloc_local((4,), "uint64")
            load_scheduler_meta(sched_words)
            sched_i32 = sched_words.view("int32")
            return [sched_i32[i] for i in range(word_count)]

        def batch_block_range(
            batch_idx, sched_begin_req, sched_end_req, sched_begin_block, sched_end_block
        ):
            # kernel.cuh:89-118.  The padded top-k this batch spans and the
            # [start_block, end_block) slice of it this partition owns.  All
            # three roles walk the schedule with exactly this arithmetic.
            topk_len = txl.local_scalar("int32", init=topk)
            with txl.If(have_topk_length), txl.Then():
                txl.ptx.ld.global_.nc.s32(topk_len, topk_length.ptr_to([batch_idx]))
            orig_topk_padded = txl.max(((topk_len + B_TOPK - 1) // B_TOPK) * B_TOPK, B_TOPK)
            extra_topk_len = txl.local_scalar("int32", init=extra_topk)
            with txl.If(have_extra_topk_length), txl.Then():
                txl.ptx.ld.global_.nc.s32(extra_topk_len, extra_topk_length.ptr_to([batch_idx]))
            total_topk_padded = (
                orig_topk_padded + ((extra_topk_len + B_TOPK - 1) // B_TOPK) * B_TOPK
            )
            start_block = txl.if_then_else(batch_idx == sched_begin_req, sched_begin_block, 0)
            end_block = txl.if_then_else(
                batch_idx == sched_end_req, sched_end_block, total_topk_padded // B_TOPK
            )
            return topk_len, extra_topk_len, orig_topk_padded, start_block, end_block

        def dequant_st128(smem_addr, raw, scale_bits):
            scale = txl.reinterpret("bfloat16", scale_bits)
            packed = txl.alloc_local((4,), "uint32")
            with txl.unroll(4) as pair_i:
                raw_pair = txl.cast(txl.shift_right(raw, txl.cast(pair_i * 16, "uint64")), "uint16")
                rounded_bits = txl.local_scalar("uint32")
                txl.ptx.cvt.rn.bf16x2.e4m3x2(rounded_bits, raw_pair)
                rounded = txl.reinterpret("bfloat16x2", rounded_bits)
                scaled_lo = txl.Shuffle([rounded], [0]) * scale
                scaled_hi = txl.Shuffle([rounded], [1]) * scale
                txl.ptx.mov.b32(
                    packed[pair_i],
                    txl.reinterpret("uint32", txl.Shuffle([scaled_lo, scaled_hi], [0, 1])),
                )
            # One 128-bit store: the four packed words are read through a b128 view.
            txl.ptx.st.weak.shared__cta.b128(smem_addr, packed.view("uint128")[0])

        def initialize_protocol():
            # kernel.cuh:35-67.  Each copy site requests the lowering's ordinary
            # descriptor prefetch.  tma_explicit deduplicates the two normal KV
            # descriptors and intentionally does not prefetch src_selector
            # candidates, matching the source's normal-only KV prefetch.
            with txl.If(warp_idx == 0), txl.Then():
                with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                    txl.ptx.mbarrier.init.shared.b64(bar_last_store_done.ptr_to([0]), txl.uint32(128))
                    txl.ptx.mbarrier.init.shared.b64(bar_q_tma.ptr_to([0]), txl.uint32(1))
                    txl.ptx.mbarrier.init.shared.b64(bar_q_utccp.ptr_to([0]), txl.uint32(1))
                    with txl.unroll(NUM_BUFS) as stage:
                        txl.ptx.mbarrier.init.shared.b64(bar_rope_ready.ptr_to([stage]), txl.uint32(1))
                        txl.ptx.mbarrier.init.shared.b64(
                            bar_nope_ready.ptr_to([stage]), txl.uint32(128)
                        )
                        txl.ptx.mbarrier.init.shared.b64(bar_raw_ready.ptr_to([stage]), txl.uint32(1))
                        txl.ptx.mbarrier.init.shared.b64(bar_raw_free.ptr_to([stage]), txl.uint32(128))
                        txl.ptx.mbarrier.init.shared.b64(bar_qk_done.ptr_to([stage]), txl.uint32(1))
                        txl.ptx.mbarrier.init.shared.b64(bar_so_ready.ptr_to([stage]), txl.uint32(128))
                        txl.ptx.mbarrier.init.shared.b64(bar_sv_done.ptr_to([stage]), txl.uint32(1))
                    with txl.unroll(NUM_INDEX_BUFS) as index_stage:
                        txl.ptx.mbarrier.init.shared.b64(
                            bar_valid_ready.ptr_to([index_stage]), txl.uint32(32)
                        )
                        txl.ptx.mbarrier.init.shared.b64(
                            bar_valid_free.ptr_to([index_stage]), txl.uint32(258)
                        )
                    txl.ptx.fence.mbarrier_init.release.cluster()
                txl.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    txl.address_of(tmem_start_addr[0]), txl.uint32(512)
                )
                allocated_tmem_start = txl.local_scalar("uint32")
                txl.ptx.ld.shared.u32(allocated_tmem_start, tmem_start_addr.ptr_to([0]))
                txl.cuda.trap_when_assert_failed(allocated_tmem_start == txl.uint32(0))
                txl.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            txl.cuda.cta_sync()

        initialize_protocol()

        def scale_exp_output():
            # kernel.cuh:134-150.  Scale/exp warpgroup and its 224-register
            # allocation.  The output and S register/shared layouts match the
            # fixed dual-GEMM TMEM datapath used by the CUDA source.
            rs_buf = txl.PipelineState(NUM_BUFS, phase=0)
            rs_index = txl.PipelineState(NUM_INDEX_BUFS, phase=0)
            scale_pair = txl.cuda.make_float2(sm_scale_div_log2, sm_scale_div_log2)
            attn_sink_log2 = txl.local_scalar("float32", init=txl.float32(-float("inf")))
            with txl.If(have_attn_sink), txl.Then():
                attn_sink_val = txl.local_scalar("float32")
                txl.ptx.ld.global_.nc.f32(attn_sink_val, attn_sink.ptr_to([idx_in_warpgroup % B_H]))
                txl.assign(attn_sink_log2, attn_sink_val * LOG_2_E)

            # kernel.cuh:77-118 expanded at the role call site to avoid the
            # register spilling explicitly called out by the CUDA source.
            (
                sched_begin_req,
                sched_end_req,
                sched_begin_block,
                sched_end_block,
                sched_begin_split,
                sched_first_split,
                sched_last_split,
            ) = unpack_scheduler_meta(7)
            # The CUDA return exits only the local run_main_loop lambda.  Guard
            # its body so inactive partitions still reach WG0's TMEM dealloc.
            with txl.If(sched_begin_req < b), txl.Then():
                with txl.serial(sched_begin_req, sched_end_req + 1, unroll=False) as batch_idx:
                    _, _, _, start_block, end_block = batch_block_range(
                        batch_idx,
                        sched_begin_req,
                        sched_end_req,
                        sched_begin_block,
                        sched_end_block,
                    )
                    is_split = txl.cast(
                        txl.if_then_else(
                            batch_idx == sched_begin_req,
                            sched_first_split,
                            txl.if_then_else(batch_idx == sched_end_req, sched_last_split, 0),
                        ),
                        "bool",
                    )
                    is_no_split = txl.Not(is_split)
                    # Both LSE and O epilogues address their split row with
                    # this index; keep the num_splits fetch off the store's
                    # dependence chain by loading it once here.
                    batch_num_splits = txl.local_scalar("int32")
                    txl.ptx.ld.global_.nc.s32(batch_num_splits, num_splits.ptr_to([batch_idx]))
                    n_split_idx = txl.local_scalar(
                        "int32",
                        init=(
                            txl.if_then_else(
                                batch_idx == sched_begin_req,
                                batch_num_splits + sched_begin_split,
                                batch_num_splits,
                            )
                        ),
                    )
                    is_last_batch = batch_idx == sched_end_req

                    # kernel.cuh:151-159.  Retire prior TMA stores before the
                    # aliased Q/O shared region is reused for this batch.
                    txl.ptx.cp.async_.bulk.wait_group.read(0)
                    bar_last_store_done.arrive(0)
                    mi = txl.local_scalar("float32", init=MAX_INIT_VAL)
                    li = txl.local_scalar("float32", init=0.0)
                    real_mi = txl.local_scalar("float32", init=txl.float32(-float("inf")))

                    # kernel.cuh:160-299.  P load, dual-warp exchange, mask,
                    # online softmax, S staging, and conditional O rescale.
                    with txl.serial(start_block, end_block, unroll=False) as block_idx:
                        txl.ptx.bar.sync(txl.uint32(BAR_WG0_SYNC), 128)
                        bar_valid_ready.wait(rs_index.stage, rs_index.phase)
                        bar_qk_done.wait(rs_buf.stage, rs_buf.phase)
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        # A later QK commit is ordered after the preceding SxV
                        # operation from this TCGEN issuer.  Its completion thus
                        # retires the prior async read of s_smem_gemm; bridge that
                        # read before p_exchange overwrites the aliased union.
                        txl.ptx.fence.proxy.async_.shared__cta()

                        p = txl.alloc_local((B_TOPK // 2,), "float32")
                        p_peer = txl.alloc_local((B_TOPK // 2,), "float32")
                        with txl.If(warp_idx < 2):
                            with txl.Then():
                                _tmem_load(p, txl.uint32(400), B_TOPK // 2)
                                _tmem_load(
                                    p_peer,
                                    txl.cuda.get_tmem_addr(txl.uint32(400), 0, B_TOPK // 2),
                                    B_TOPK // 2,
                                )
                            with txl.Else():
                                _tmem_load(p_peer, txl.uint32(400), B_TOPK // 2)
                                _tmem_load(
                                    p,
                                    txl.cuda.get_tmem_addr(txl.uint32(400), 0, B_TOPK // 2),
                                    B_TOPK // 2,
                                )
                        txl.ptx.tcgen05.wait__ld.sync.aligned()
                        txl.ptx.tcgen05.fence__before_thread_sync()

                        with txl.unroll((B_TOPK // 2) // 4) as exchange_i:
                            exchange_offset = exchange_i * 32 * 4 + lane_idx * 4
                            p_peer_words = p_peer.view("uint32")
                            peer_word = exchange_i * 4
                            txl.ptx.st.shared.v4.u32(
                                p_exchange.view("uint32").ptr_to([warp_idx ^ 2, exchange_offset]),
                                p_peer_words[peer_word],
                                p_peer_words[peer_word + 1],
                                p_peer_words[peer_word + 2],
                                p_peer_words[peer_word + 3],
                            )
                        txl.ptx.bar.sync(
                            txl.uint32(BAR_WG0_WARP02 + txl.bitwise_and(warp_idx, txl.int32(1))), 64
                        )
                        with txl.unroll((B_TOPK // 2) // 4) as exchange_i:
                            exchange_offset = exchange_i * 32 * 4 + lane_idx * 4
                            peer_tmp = txl.alloc_local((4,), "float32")
                            peer_tmp_words = peer_tmp.view("uint32")
                            txl.ptx.ld.shared.v4.u32(
                                peer_tmp_words[0],
                                peer_tmp_words[1],
                                peer_tmp_words[2],
                                peer_tmp_words[3],
                                p_exchange.view("uint32").ptr_to([warp_idx, exchange_offset]),
                            )
                            pair0 = txl.local_scalar("uint64")
                            pair1 = txl.local_scalar("uint64")
                            txl.ptx.add.f32x2(
                                pair0,
                                txl.cuda.make_float2(p[exchange_i * 4], p[exchange_i * 4 + 1]),
                                txl.cuda.make_float2(peer_tmp[0], peer_tmp[1]),
                            )
                            txl.ptx.add.f32x2(
                                pair1,
                                txl.cuda.make_float2(p[exchange_i * 4 + 2], p[exchange_i * 4 + 3]),
                                txl.cuda.make_float2(peer_tmp[2], peer_tmp[3]),
                            )
                            txl.ptx.mov.b32(p[exchange_i * 4], txl.cuda.float2_x(pair0))
                            txl.ptx.mov.b32(p[exchange_i * 4 + 1], txl.cuda.float2_y(pair0))
                            txl.ptx.mov.b32(p[exchange_i * 4 + 2], txl.cuda.float2_x(pair1))
                            txl.ptx.mov.b32(p[exchange_i * 4 + 3], txl.cuda.float2_y(pair1))

                        valid_word = txl.local_scalar("uint32")
                        txl.ptx.ld.shared.u32(
                            valid_word,
                            is_token_valid.view("uint32").ptr_to(
                                [rs_index.stage, txl.if_then_else(idx_in_warpgroup >= 64, 1, 0)]
                            ),
                        )
                        with txl.unroll(B_TOPK // 2) as p_i:
                            with (
                                txl.If(
                                    txl.bitwise_and(
                                        txl.shift_right(valid_word, txl.cast(p_i, "uint32")),
                                        txl.uint32(1),
                                    )
                                    == txl.uint32(0)
                                ),
                                txl.Then(),
                            ):
                                txl.ptx.mov.b32(p[p_i], txl.float32(-float("inf")))

                        cur_pi_max = txl.local_scalar("float32", init=txl.float32(-float("inf")))
                        with txl.unroll(B_TOPK // 2) as p_i:
                            txl.assign(cur_pi_max, txl.max(cur_pi_max, p[p_i]))
                        txl.assign(cur_pi_max, cur_pi_max * sm_scale_div_log2)
                        txl.ptx.st.shared.f32(rowwise_buf.ptr_to([idx_in_warpgroup]), cur_pi_max)
                        txl.ptx.bar.sync(txl.uint32(BAR_WG0_SYNC), 128)
                        bar_valid_free.arrive(rs_index.stage)
                        peer_pi_max = txl.local_scalar("float32")
                        txl.ptx.ld.shared.f32(
                            peer_pi_max, rowwise_buf.ptr_to([idx_in_warpgroup ^ 64])
                        )
                        txl.assign(cur_pi_max, txl.max(cur_pi_max, peer_pi_max))
                        txl.assign(real_mi, txl.max(real_mi, cur_pi_max))
                        should_scale_o = txl.local_scalar("uint32")
                        txl.ptx.vote_sync.any.pred(
                            should_scale_o, cur_pi_max - mi > txl.float32(6.0), txl.uint32(0xFFFFFFFF)
                        )
                        new_max = txl.local_scalar("float32")
                        scale_for_old = txl.local_scalar("float32")
                        with txl.If(should_scale_o == txl.uint32(0)):
                            with txl.Then():
                                txl.assign(scale_for_old, 1.0)
                                txl.assign(new_max, mi)
                            with txl.Else():
                                txl.assign(new_max, txl.max(cur_pi_max, mi))
                                txl.ptx.ex2.approx.ftz.f32(scale_for_old, mi - new_max)
                        txl.assign(mi, new_max)

                        s_frag = txl.alloc_local((B_TOPK // 2,), "bfloat16")
                        s_pack = s_frag.view("uint32")
                        cur_sum_pair = txl.local_scalar("uint64", init=txl.cuda.make_float2(0.0, 0.0))
                        neg_max_pair = txl.cuda.make_float2(-new_max, -new_max)
                        with txl.unroll((B_TOPK // 2) // 2) as s_i:
                            p_pair = txl.cuda.make_float2(p[s_i * 2], p[s_i * 2 + 1])
                            soft_pair = txl.local_scalar("uint64")
                            txl.ptx.fma.rn.f32x2(soft_pair, p_pair, scale_pair, neg_max_pair)
                            sx = txl.local_scalar("float32")
                            sy = txl.local_scalar("float32")
                            txl.ptx.ex2.approx.ftz.f32(sx, txl.cuda.float2_x(soft_pair))
                            txl.ptx.ex2.approx.ftz.f32(sy, txl.cuda.float2_y(soft_pair))
                            txl.ptx.add.f32x2(cur_sum_pair, cur_sum_pair, txl.cuda.make_float2(sx, sy))
                            txl.ptx.mov.b32(s_pack[s_i], txl.cuda.float22bfloat162_rn(sx, sy))
                        cur_sum = txl.cuda.float2_x(cur_sum_pair) + txl.cuda.float2_y(cur_sum_pair)
                        li_next = txl.local_scalar("float32")
                        txl.ptx.fma.rn.f32(li_next, li, scale_for_old, cur_sum)
                        txl.assign(li, li_next)

                        s_base = idx_in_warpgroup // 64 * 2048 + idx_in_warpgroup % 64 * 8
                        s_words = s_frag.view("uint32")
                        with txl.unroll(4) as s_store_i:
                            s_ptr = txl.ptr_byte_offset(
                                s_smem_gemm.ptr_to([0, 0]),
                                (s_base + s_store_i * 512) * BF16_BYTES,
                                "bfloat16",
                            )
                            s_word = s_store_i * 4
                            txl.ptx.st.shared.v4.u32(
                                s_ptr,
                                s_words[s_word],
                                s_words[s_word + 1],
                                s_words[s_word + 2],
                                s_words[s_word + 3],
                            )
                        with (
                            txl.If(txl.And(block_idx != start_block, should_scale_o != txl.uint32(0))),
                            txl.Then(),
                        ):
                            scale_for_old_pair = txl.cuda.make_float2(scale_for_old, scale_for_old)
                            txl.ptx.tcgen05.fence__after_thread_sync()
                            o_rescale = txl.alloc_local((64,), "float32")
                            with txl.unroll((D_V // 2) // 64) as o_chunk:
                                _load_scaled_tmem_chunk(
                                    o_rescale,
                                    txl.cuda.get_tmem_addr(txl.uint32(0), 0, o_chunk * 64),
                                    scale_for_old_pair,
                                )
                                _tmem_store(
                                    o_rescale, txl.cuda.get_tmem_addr(txl.uint32(0), 0, o_chunk * 64)
                                )
                                txl.ptx.tcgen05.wait__st.sync.aligned()
                            txl.ptx.tcgen05.fence__before_thread_sync()

                        txl.ptx.fence.proxy.async_.shared__cta()
                        bar_so_ready.arrive(rs_buf.stage)
                        with txl.If(block_idx != end_block - 1), txl.Then():
                            rs_buf.advance()
                            rs_index.advance()

                    # kernel.cuh:301-333.  Empty-row repair, li exchange, LSE
                    # store, final SV wait, and ring advance.
                    with txl.If(real_mi == txl.float32(-float("inf"))), txl.Then():
                        txl.assign(li, 0.0)
                        txl.assign(mi, txl.float32(-float("inf")))
                    # Every WG0 warp read its peer's per-block maximum from this
                    # allocation above.  Do not let a faster warp reuse the same
                    # locations for ``li`` until all of those reads have retired.
                    txl.ptx.bar.sync(txl.uint32(BAR_WG0_SYNC), 128)
                    txl.ptx.st.shared.f32(rowwise_buf.ptr_to([idx_in_warpgroup]), li)
                    txl.ptx.bar.sync(txl.uint32(BAR_WG0_SYNC), 128)
                    peer_li = txl.local_scalar("float32")
                    txl.ptx.ld.shared.f32(peer_li, rowwise_buf.ptr_to([idx_in_warpgroup ^ 64]))
                    txl.assign(li, li + peer_li)
                    with txl.If(idx_in_warpgroup < B_H), txl.Then():
                        with txl.If(is_no_split):
                            with txl.Then():
                                cur_lse = txl.local_scalar("float32")
                                txl.ptx.fma.rn.f32(cur_lse, mi, txl.float32(LN_2), txl.log(li))
                                txl.ptx.st.global_.f32(
                                    lse.ptr_to(
                                        [
                                            batch_idx * stride_lse_b
                                            + s_q_idx * stride_lse_s_q
                                            + idx_in_warpgroup
                                        ]
                                    ),
                                    txl.if_then_else(
                                        cur_lse == txl.float32(-float("inf")),
                                        txl.float32(float("inf")),
                                        cur_lse,
                                    ),
                                )
                            with txl.Else():
                                txl.ptx.st.global_.f32(
                                    lse_accum.ptr_to(
                                        [
                                            n_split_idx * stride_lse_accum_split
                                            + s_q_idx * stride_lse_accum_s_q
                                            + idx_in_warpgroup
                                        ]
                                    ),
                                    txl.log2(li) + mi,
                                )
                    bar_sv_done.wait(rs_buf.stage, rs_buf.phase)
                    rs_buf.advance()
                    rs_index.advance()
                    txl.ptx.tcgen05.fence__after_thread_sync()
                    with txl.If(txl.And(use_pdl, is_last_batch)), txl.Then():
                        txl.ptx.griddepcontrol.launch_dependents()

                    # kernel.cuh:335-421.  Keep no-split TMA output and split
                    # fp32 bulk output as distinct epilogues; attn_sink is only
                    # applied here for no-split and is deferred to combine for
                    # split output exactly as in the CUDA source.
                    with txl.If(is_no_split):
                        with txl.Then():
                            sink_exp = txl.local_scalar("float32")
                            txl.ptx.ex2.approx.ftz.f32(sink_exp, attn_sink_log2 - mi)
                            output_scale = txl.local_scalar(
                                "float32",
                                init=txl.if_then_else(
                                    li == 0.0, 0.0, txl.cuda.fdividef(1.0, li + sink_exp)
                                ),
                            )
                            output_scale_pair = txl.local_scalar(
                                "uint64", init=txl.cuda.make_float2(output_scale, output_scale)
                            )
                            o_epi = txl.alloc_local((64,), "float32")
                            o_epi_bf16 = txl.alloc_local((64,), "bfloat16")

                            def emit_no_split_epilogue(epi_i: txl.constexpr):
                                _load_scaled_tmem_chunk(
                                    o_epi,
                                    txl.cuda.get_tmem_addr(txl.uint32(0), 0, epi_i * 64),
                                    output_scale_pair,
                                )
                                o_epi_words = o_epi_bf16.view("uint32")
                                with txl.unroll(64 // 2) as cast_i:
                                    txl.ptx.cvt.rn.bf16x2.f32(
                                        o_epi_words[cast_i],
                                        o_epi[cast_i * 2 + 1],
                                        o_epi[cast_i * 2],
                                    )
                                col_base = (D_V // 2 if epi_i * 64 >= D_V // 4 else 0) + (
                                    epi_i * 64
                                ) % (D_V // 4)
                                with txl.unroll(8) as o_store_i:
                                    o_smem_offset = (
                                        col_base * B_H
                                        + idx_in_warpgroup // 64 * 8192
                                        + idx_in_warpgroup % 64 * 64
                                        + txl.bitwise_xor(
                                            o_store_i * 8,
                                            txl.shift_left(
                                                txl.bitwise_and(
                                                    idx_in_warpgroup // 64 * 128
                                                    + idx_in_warpgroup % 64,
                                                    7,
                                                ),
                                                3,
                                            ),
                                        )
                                    )
                                    o_smem_ptr = txl.ptr_byte_offset(
                                        o_smem.ptr_to([0, 0]),
                                        o_smem_offset * BF16_BYTES,
                                        "bfloat16",
                                    )
                                    o_word = o_store_i * 4
                                    txl.ptx.st.shared.v4.u32(
                                        o_smem_ptr,
                                        o_epi_words[o_word],
                                        o_epi_words[o_word + 1],
                                        o_epi_words[o_word + 2],
                                        o_epi_words[o_word + 3],
                                    )
                                txl.ptx.fence.proxy.async_.shared__cta()
                                txl.ptx.bar.sync(txl.uint32(BAR_WG0_SYNC), 128)
                                with txl.If(warp_idx == 0), txl.Then():
                                    with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                                        txl.ptx[_TMA_S2G_4D](
                                            txl.address_of(out_tensormap),
                                            txl.int32(col_base),
                                            txl.int32(0),
                                            txl.cast(s_q_idx, "int32"),
                                            txl.cast(batch_idx, "int32"),
                                            txl.cuda.cvta_generic_to_shared(
                                                txl.ptr_byte_offset(
                                                    o_smem.ptr_to([0, 0]),
                                                    col_base * B_H * BF16_BYTES,
                                                    "bfloat16",
                                                )
                                            ),
                                        )
                                warp1_col_base = col_base + D_V // 4
                                with txl.If(warp_idx == 1), txl.Then():
                                    with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                                        txl.ptx[_TMA_S2G_4D](
                                            txl.address_of(out_tensormap),
                                            txl.int32(warp1_col_base),
                                            txl.int32(0),
                                            txl.cast(s_q_idx, "int32"),
                                            txl.cast(batch_idx, "int32"),
                                            txl.cuda.cvta_generic_to_shared(
                                                txl.ptr_byte_offset(
                                                    o_smem.ptr_to([0, 0]),
                                                    warp1_col_base * B_H * BF16_BYTES,
                                                    "bfloat16",
                                                )
                                            ),
                                        )

                            emit_no_split_epilogue(0)
                            emit_no_split_epilogue(1)
                            emit_no_split_epilogue(2)
                            emit_no_split_epilogue(3)
                            txl.ptx.cp.async_.bulk.commit_group()
                        with txl.Else():
                            output_scale = txl.local_scalar(
                                "float32",
                                init=txl.if_then_else(li == 0.0, 0.0, txl.cuda.fdividef(1.0, li)),
                            )
                            output_scale_pair = txl.local_scalar(
                                "uint64", init=txl.cuda.make_float2(output_scale, output_scale)
                            )
                            split_local = txl.alloc_local((64,), "float32")
                            with txl.unroll((D_V // 2) // 64) as epi_i:
                                _load_scaled_tmem_chunk(
                                    split_local,
                                    txl.cuda.get_tmem_addr(txl.uint32(0), 0, epi_i * 64),
                                    output_scale_pair,
                                )
                                col_base = (
                                    (idx_in_warpgroup // 64) * 128
                                    + txl.if_then_else(epi_i * 64 >= D_V // 4, D_V // 2, 0)
                                    + (epi_i * 64) % (D_V // 4)
                                )
                                split_words = split_local.view("uint32")
                                with txl.unroll(64 // 4) as j:
                                    split_word = j * 4
                                    txl.ptx.st.shared.v4.u32(
                                        o_accum_storage.ptr_to(
                                            [(idx_in_warpgroup % 64) * (D_V + 8) + col_base + j * 4]
                                        ),
                                        split_words[split_word],
                                        split_words[split_word + 1],
                                        split_words[split_word + 2],
                                        split_words[split_word + 3],
                                    )
                            txl.ptx.fence.proxy.async_.shared__cta()
                            txl.ptx.bar.sync(txl.uint32(BAR_WG0_SYNC), 128)
                            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                                # One int32 row index per store: handing
                                # ptr_to the open sum lets the index
                                # widening distribute over it, so each of
                                # the 16 stores sign-extends three
                                # products and adds them in 64 bits.
                                o_accum_row = txl.local_scalar("int32")
                                with txl.unroll(B_H // 4) as local_row:
                                    smem_row = local_row * 4 + warp_idx
                                    txl.assign(
                                        o_accum_row,
                                        n_split_idx * stride_o_accum_split
                                        + s_q_idx * stride_o_accum_s_q
                                        + smem_row * stride_o_accum_h_q,
                                    )
                                    txl.ptx["cp.async.bulk.global.shared::cta.bulk_group"](
                                        o_accum.ptr_to([o_accum_row]),
                                        o_accum_storage.ptr_to([smem_row * (D_V + 8)]),
                                        txl.uint32(D_V * 4),
                                    )
                                txl.ptx.cp.async_.bulk.commit_group()

                    # kernel.cuh:116 uses the unaligned spelling because the
                    # elected WG1 producer lanes reach this named barrier via
                    # control flow distinct from the empty-role lanes.
                    txl.ptx.barrier.sync(txl.uint32(BAR_EVERYONE_SYNC), txl.uint32(NUM_THREADS))

            with txl.If(warp_idx == 0), txl.Then():
                txl.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(txl.uint32(0), txl.uint32(512))

        def producer_mma(selected_wg1_role):
            # kernel.cuh:431-746.  Mirror each CUDA run_main_loop(lambda)
            # call as a separate parser-time specialization.  This keeps the
            # scheduler and its registers inside the K-selected warp role.
            def run_wg1_role(role: txl.constexpr):
                rs_buf = txl.PipelineState(NUM_BUFS, phase=0)
                rs_index = txl.PipelineState(NUM_INDEX_BUFS, phase=0)

                qk_nope_desc = txl.SmemDescriptor()
                qk_rope_desc = txl.SmemDescriptor()
                if role == 4:
                    qk_nope_desc.init(k_full.ptr_to([0, 0, 0]), ldo=1024, sdo=64, swizzle=3)
                    if is_v32:
                        qk_rope_desc.init(k_rope.ptr_to([0, 0, D_V]), ldo=0, sdo=32, swizzle=2)

                # kernel.cuh:657-667.  These warp-7 invariants are deliberately
                # materialized before the scheduler traversal.  Pointer bases are
                # held as byte addresses so the per-token paths only add offsets.
                if role == 7:
                    tma_coords_step_per_token = (656 if is_v32 else 576) // tma_k_stride
                    k_scales_ptr_u64 = txl.reinterpret(
                        "uint64",
                        (
                            kv.ptr_to([d_nope // BF16_BYTES])
                            if is_v32
                            else kv.ptr_to([page_block_size * (tma_k_stride // BF16_BYTES)])
                        ),
                    )
                    extra_k_scales_ptr_u64 = txl.local_scalar("uint64", init=txl.uint64(0))
                    with txl.If(have_extra_kv), txl.Then():
                        txl.assign(
                            extra_k_scales_ptr_u64,
                            txl.reinterpret(
                                "uint64",
                                (
                                    extra_kv.ptr_to([d_nope // BF16_BYTES])
                                    if is_v32
                                    else extra_kv.ptr_to(
                                        [extra_page_block_size * (tma_k_stride // BF16_BYTES)]
                                    )
                                ),
                            ),
                        )
                # kernel.cuh:77-118, expanded once for all WG1 threads.  Non-elected
                # lanes still execute the empty role and participate in the 384-way
                # per-batch named barrier, matching the CUDA else branch at 744.
                (sched_begin_req, sched_end_req, sched_begin_block, sched_end_block) = (
                    unpack_scheduler_meta(4)
                )
                batch_epoch = txl.PipelineState(1, phase=0)

                # The CUDA return exits only this role's run_main_loop lambda.
                with txl.If(sched_begin_req < b), txl.Then():
                    with txl.serial(sched_begin_req, sched_end_req + 1, unroll=False) as batch_idx:
                        (topk_len, extra_topk_len, orig_topk_padded, start_block, end_block) = (
                            batch_block_range(
                                batch_idx,
                                sched_begin_req,
                                sched_end_req,
                                sched_begin_block,
                                sched_end_block,
                            )
                        )
                        num_orig_blocks = orig_topk_padded // B_TOPK

                        if role == 4:
                            # kernel.cuh:431-527.  Warp 4 issues both Q TMAs, then
                            # the same SW128/SW64 UTCCP deposits at TMEM Q=256.
                            txl.cuda.trap_when_assert_failed(start_block < end_block)
                            # SmemLayoutQ_SW128 is tile_to_shape of a 64x64
                            # atom.  CuTe therefore partitions the 64x512 Q
                            # copy into eight source-order TMA boxes instead of
                            # issuing one monolithic 64 KiB transaction.
                            with txl.unroll(512 // 64) as q_tile:
                                txl.ptx[_TMA_G2S_4D_CACHE](
                                    txl.ptr_byte_offset(
                                        q_sw128.ptr_to([0, 0]),
                                        q_tile * B_H * 64 * BF16_BYTES,
                                        "bfloat16",
                                    ),
                                    txl.address_of(q_strided_tensormap),
                                    txl.cast(q_tile * 64, "int32"),
                                    txl.int32(0),
                                    txl.cast(s_q_idx, "int32"),
                                    txl.cast(batch_idx, "int32"),
                                    txl.cuda.cvta_generic_to_shared(bar_q_tma.ptr_to([0])),
                                    _Q_TMA_CACHE_HINT,
                                )
                            if is_v32:
                                txl.ptx[_TMA_G2S_5D_CACHE](
                                    q_sw64.ptr_to([0, 0]),
                                    txl.address_of(q_tail_tensormap),
                                    txl.int32(0),
                                    txl.int32(0),
                                    txl.int32(16),
                                    txl.cast(s_q_idx, "int32"),
                                    txl.cast(batch_idx, "int32"),
                                    txl.cuda.cvta_generic_to_shared(bar_q_tma.ptr_to([0])),
                                    _Q_TMA_CACHE_HINT,
                                )
                            bar_q_tma.arrive(0, tx_count=B_H * d_qk * BF16_BYTES)
                            bar_q_tma.wait(0, batch_epoch.phase)
                            txl.ptx.tcgen05.fence__after_thread_sync()
                            q_main_cp_view = q_sw128.view(B_H, 4, 2, 64)
                            with txl.unroll(16) as q_main_flat:
                                q_main_src = txl.ptr_byte_offset(
                                    q_main_cp_view.ptr_to([0, 0, 0, 0]),
                                    (q_main_flat % 4 * 1024 + q_main_flat // 4 % 4 * 2) * 16,
                                    "bfloat16",
                                )
                                txl.ptx[_TCGEN_CP_128X256](
                                    txl.cast(
                                        256 + q_main_flat % 4 * 32 + q_main_flat // 4 % 4 * 8,
                                        "uint32",
                                    ),
                                    _replace_smem_desc_addr(q_main_cp_desc, q_main_src),
                                )
                            if is_v32:
                                with txl.unroll(2) as q_tail_flat:
                                    q_tail_src = txl.ptr_byte_offset(
                                        q_sw64.ptr_to([0, 0]), q_tail_flat % 2 * 2 * 16, "bfloat16"
                                    )
                                    txl.ptx[_TCGEN_CP_128X256](
                                        txl.cast(384 + q_tail_flat % 2 * 8, "uint32"),
                                        _replace_smem_desc_addr(q_tail_cp_desc, q_tail_src),
                                    )
                            bar_q_utccp.arrive(0)
                            bar_q_utccp.wait(0, batch_epoch.phase)
                            txl.ptx.tcgen05.fence__after_thread_sync()

                            # kernel.cuh:529-584.  MODEL_TYPE only selects how the
                            # shared K latent is interpreted; both instances issue
                            # the same dual-head P and SxV pipelines.
                            with txl.serial(start_block, end_block, unroll=False) as block_idx:
                                k_stage_elems = B_TOPK * (D_V + 64) if is_v32 else B_TOPK * D_V
                                if is_v32:
                                    bar_rope_ready.wait(rs_buf.stage, rs_buf.phase)
                                    txl.ptx.tcgen05.fence__after_thread_sync()
                                    with txl.unroll(2) as qk_rope_ki:
                                        qk_rope_offset = (
                                            rs_buf.stage * k_stage_elems + qk_rope_ki * 16
                                        ) // 8
                                        txl.ptx[_MMA_WS_F16](
                                            txl.uint32(400),
                                            txl.cast(384 + qk_rope_ki * 8, "uint32"),
                                            _desc_add_16B_offset(qk_rope_desc.desc, qk_rope_offset),
                                            txl.uint32(69207184),
                                            txl.ptx.pred(txl.Cast("bool", qk_rope_ki != 0)),
                                            txl.uint64(0),
                                        )
                                    bar_nope_ready.wait(rs_buf.stage, rs_buf.phase)
                                    txl.ptx.tcgen05.fence__after_thread_sync()
                                else:
                                    bar_rope_ready.wait(rs_buf.stage, rs_buf.phase)
                                    bar_nope_ready.wait(rs_buf.stage, rs_buf.phase)
                                    txl.ptx.tcgen05.fence__after_thread_sync()

                                with txl.unroll(16) as qk_nope_ki:
                                    qk_nope_offset = (
                                        qk_nope_ki // 2048 * k_stage_elems
                                        + rs_buf.stage * k_stage_elems
                                        + qk_nope_ki % 16 // 4 * 8192
                                        + qk_nope_ki % 2048 // 16 * 64
                                        + qk_nope_ki % 4 * 16
                                    ) // 8
                                    txl.ptx[_MMA_WS_F16](
                                        txl.uint32(400),
                                        txl.cast(256 + qk_nope_ki * 8, "uint32"),
                                        _desc_add_16B_offset(qk_nope_desc.desc, qk_nope_offset),
                                        txl.uint32(69207184),
                                        txl.ptx.pred(
                                            txl.Cast(
                                                "bool",
                                                txl.Or(
                                                    qk_nope_ki != 0,
                                                    txl.cast(txl.uint32(1 if is_v32 else 0), "bool"),
                                                ),
                                            )
                                        ),
                                        txl.uint64(0),
                                    )
                                bar_qk_done.arrive(rs_buf.stage)

                                bar_so_ready.wait(rs_buf.stage, rs_buf.phase)
                                txl.ptx.tcgen05.fence__after_thread_sync()
                                mma_o_accum = txl.if_then_else(
                                    block_idx == start_block, txl.uint32(0), txl.uint32(1)
                                )
                                with txl.unroll(4) as pv_ki:
                                    pv_a_offset = (pv_ki % 4 * 1024 + pv_ki // 4 * 8) // 8
                                    pv_b_lo_offset = (
                                        (pv_ki * 16) // 64 * k_stage_elems
                                        + rs_buf.stage * k_stage_elems
                                        + (pv_ki * 16) % 64 * 64
                                    ) // 8
                                    txl.ptx[_MMA_WS_F16](
                                        txl.uint32(0),
                                        _desc_add_16B_offset(pv_a_lo_desc.desc, pv_a_offset),
                                        _desc_add_16B_offset(pv_b_lo_desc.desc, pv_b_lo_offset),
                                        txl.uint32(71369872),
                                        txl.ptx.pred(
                                            txl.Cast(
                                                "bool",
                                                txl.Or(pv_ki != 0, txl.cast(mma_o_accum, "bool")),
                                            )
                                        ),
                                        txl.uint64(0),
                                    )
                                with txl.unroll(4) as pv_ki:
                                    pv_a_offset = (pv_ki % 4 * 1024 + pv_ki // 4 * 8) // 8
                                    pv_b_hi_offset = (
                                        (pv_ki * 16) // 64 * k_stage_elems
                                        + rs_buf.stage * k_stage_elems
                                        + (pv_ki * 16) % 64 * 64
                                        + 16384
                                    ) // 8
                                    txl.ptx[_MMA_WS_F16](
                                        txl.uint32(128),
                                        _desc_add_16B_offset(pv_a_hi_desc.desc, pv_a_offset),
                                        _desc_add_16B_offset(pv_b_hi_desc.desc, pv_b_hi_offset),
                                        txl.uint32(71369872),
                                        txl.ptx.pred(
                                            txl.Cast(
                                                "bool",
                                                txl.Or(pv_ki != 0, txl.cast(mma_o_accum, "bool")),
                                            )
                                        ),
                                        txl.uint64(0),
                                    )
                                bar_sv_done.arrive(rs_buf.stage)
                                rs_buf.advance()
                                rs_index.advance()
                        elif role == 5:
                            # kernel.cuh:586-615.  One gather4 producer loads raw
                            # fp8 NoPE as int64, retaining the two-stage raw ring.
                            bar_q_utccp.wait(0, batch_epoch.phase)
                            bar_last_store_done.wait(0, batch_epoch.phase)
                            with txl.serial(start_block, end_block, unroll=False) as block_idx:
                                bar_valid_ready.wait(rs_index.stage, rs_index.phase)
                                bar_raw_free.wait(rs_buf.stage, rs_buf.phase ^ 1)
                                cur_indices = txl.alloc_local((4,), "int32")
                                next_indices = txl.alloc_local((4,), "int32")
                                cur_index_words = cur_indices.view("uint32")
                                txl.ptx.ld.shared.v4.u32(
                                    cur_index_words[0],
                                    cur_index_words[1],
                                    cur_index_words[2],
                                    cur_index_words[3],
                                    tma_coord.view("uint32").ptr_to([rs_index.stage, 0]),
                                )
                                with txl.unroll(B_TOPK // 4) as row4:
                                    row = row4 * 4
                                    with txl.If(row + 4 < B_TOPK), txl.Then():
                                        next_index_words = next_indices.view("uint32")
                                        txl.ptx.ld.shared.v4.u32(
                                            next_index_words[0],
                                            next_index_words[1],
                                            next_index_words[2],
                                            next_index_words[3],
                                            tma_coord.view("uint32").ptr_to(
                                                [rs_index.stage, row + 4]
                                            ),
                                        )
                                    selected_nope_tensormap = txl.local_scalar(
                                        "uint64",
                                        init=txl.reinterpret(
                                            "uint64", txl.address_of(kv_nope_tensormap)
                                        ),
                                    )
                                    with txl.If(have_extra_kv), txl.Then():
                                        txl.assign(
                                            selected_nope_tensormap,
                                            (
                                                txl.if_then_else(
                                                    block_idx >= num_orig_blocks,
                                                    txl.reinterpret(
                                                        "uint64",
                                                        txl.address_of(extra_kv_nope_tensormap),
                                                    ),
                                                    txl.reinterpret(
                                                        "uint64", txl.address_of(kv_nope_tensormap)
                                                    ),
                                                )
                                            ),
                                        )
                                    txl.ptx[_TMA_GATHER4_2D_CACHE](
                                        raw_nope.ptr_to([rs_buf.stage, row, 0]),
                                        txl.reinterpret(txl.handle().ty, selected_nope_tensormap),
                                        txl.int32(0),
                                        cur_indices[0],
                                        cur_indices[1],
                                        cur_indices[2],
                                        cur_indices[3],
                                        txl.cuda.cvta_generic_to_shared(
                                            bar_raw_ready.ptr_to([rs_buf.stage])
                                        ),
                                        _KV_TMA_CACHE_HINT,
                                    )
                                    txl.ptx.mov.b32(cur_indices[0], next_indices[0])
                                    txl.ptx.mov.b32(cur_indices[1], next_indices[1])
                                    txl.ptx.mov.b32(cur_indices[2], next_indices[2])
                                    txl.ptx.mov.b32(cur_indices[3], next_indices[3])
                                bar_raw_ready.arrive(rs_buf.stage, tx_count=B_TOPK * d_nope)
                                bar_valid_free.arrive(rs_index.stage)
                                rs_buf.advance()
                                rs_index.advance()
                        elif role == 6:
                            # kernel.cuh:616-652.  RoPE remains bf16 and uses the
                            # model-specific SW64 (two 32-col gathers) or SW128
                            # (one 64-col gather) destination.
                            bar_q_utccp.wait(0, batch_epoch.phase)
                            bar_last_store_done.wait(0, batch_epoch.phase)
                            with txl.serial(start_block, end_block, unroll=False) as block_idx:
                                bar_valid_ready.wait(rs_index.stage, rs_index.phase)
                                if is_v32:
                                    bar_qk_done.wait(rs_buf.stage, rs_buf.phase ^ 1)
                                else:
                                    bar_sv_done.wait(rs_buf.stage, rs_buf.phase ^ 1)
                                cur_indices = txl.alloc_local((4,), "int32")
                                next_indices = txl.alloc_local((4,), "int32")
                                cur_index_words = cur_indices.view("uint32")
                                txl.ptx.ld.shared.v4.u32(
                                    cur_index_words[0],
                                    cur_index_words[1],
                                    cur_index_words[2],
                                    cur_index_words[3],
                                    tma_coord.view("uint32").ptr_to([rs_index.stage, 0]),
                                )
                                with txl.unroll(B_TOPK // 4) as row4:
                                    row = row4 * 4
                                    with txl.If(row + 4 < B_TOPK), txl.Then():
                                        next_index_words = next_indices.view("uint32")
                                        txl.ptx.ld.shared.v4.u32(
                                            next_index_words[0],
                                            next_index_words[1],
                                            next_index_words[2],
                                            next_index_words[3],
                                            tma_coord.view("uint32").ptr_to(
                                                [rs_index.stage, row + 4]
                                            ),
                                        )
                                    with txl.unroll(64 // rope_tile) as rope_part:
                                        if is_v32:
                                            rope_tma_dst = txl.ptr_byte_offset(
                                                k_union.ptr_to([0, 0, 0]),
                                                (
                                                    rs_buf.stage * B_TOPK * (D_V + 64)
                                                    + (D_V + rope_part * rope_tile) * B_TOPK
                                                    + row * rope_tile
                                                )
                                                * BF16_BYTES,
                                                "bfloat16",
                                            )
                                        else:
                                            rope_tma_dst = txl.ptr_byte_offset(
                                                k_full.ptr_to([0, 0, 0]),
                                                (
                                                    rs_buf.stage * B_TOPK * D_V
                                                    + d_nope * B_TOPK
                                                    + row * 64
                                                )
                                                * BF16_BYTES,
                                                "bfloat16",
                                            )
                                        selected_rope_tensormap = txl.local_scalar(
                                            "uint64",
                                            init=(
                                                txl.reinterpret(
                                                    "uint64", txl.address_of(kv_rope_tensormap)
                                                )
                                            ),
                                        )
                                        with txl.If(have_extra_kv), txl.Then():
                                            txl.assign(
                                                selected_rope_tensormap,
                                                txl.if_then_else(
                                                    block_idx >= num_orig_blocks,
                                                    txl.reinterpret(
                                                        "uint64",
                                                        txl.address_of(extra_kv_rope_tensormap),
                                                    ),
                                                    txl.reinterpret(
                                                        "uint64", txl.address_of(kv_rope_tensormap)
                                                    ),
                                                ),
                                            )
                                        txl.ptx[_TMA_GATHER4_2D_CACHE](
                                            rope_tma_dst,
                                            txl.reinterpret(txl.handle().ty, selected_rope_tensormap),
                                            txl.cast(rope_part * rope_tile, "int32"),
                                            cur_indices[0],
                                            cur_indices[1],
                                            cur_indices[2],
                                            cur_indices[3],
                                            txl.cuda.cvta_generic_to_shared(
                                                bar_rope_ready.ptr_to([rs_buf.stage])
                                            ),
                                            _KV_TMA_CACHE_HINT,
                                        )
                                    txl.ptx.mov.b32(cur_indices[0], next_indices[0])
                                    txl.ptx.mov.b32(cur_indices[1], next_indices[1])
                                    txl.ptx.mov.b32(cur_indices[2], next_indices[2])
                                    txl.ptx.mov.b32(cur_indices[3], next_indices[3])
                                bar_rope_ready.arrive(
                                    rs_buf.stage, tx_count=B_TOPK * 64 * BF16_BYTES
                                )
                                bar_valid_free.arrive(rs_index.stage)
                                rs_buf.advance()
                                rs_index.advance()
                        elif role == 7:
                            # kernel.cuh:653-743.  All 32 lanes transform exactly
                            # two indices, form TMA coordinates, load/convert the
                            # model-specific scales, and construct each 8-bit mask.
                            indices_base = (
                                batch_idx * stride_indices_b + s_q_idx * stride_indices_s_q
                            )
                            extra_indices_base = (
                                batch_idx * stride_extra_indices_b
                                + s_q_idx * stride_extra_indices_s_q
                            )

                            def process_index_block(cur_block, is_extra: txl.constexpr):
                                abs_pos = txl.if_then_else(
                                    is_extra,
                                    (cur_block - num_orig_blocks) * B_TOPK + lane_idx * 2,
                                    cur_block * B_TOPK + lane_idx * 2,
                                )
                                cur_page_size = txl.if_then_else(
                                    is_extra, extra_page_block_size, page_block_size
                                )
                                cur_block_stride = txl.if_then_else(
                                    is_extra, stride_extra_kv_block, stride_kv_block
                                )
                                cur_row_stride = txl.if_then_else(
                                    is_extra, stride_extra_kv_row, stride_kv_row
                                )
                                cur_length = txl.if_then_else(is_extra, extra_topk_len, topk_len)
                                cur_k_scales_ptr_u64 = txl.if_then_else(
                                    is_extra, extra_k_scales_ptr_u64, k_scales_ptr_u64
                                )
                                cur_tma_coords_step_per_block = txl.if_then_else(
                                    is_extra,
                                    tma_coords_step_per_extra_block,
                                    tma_coords_step_per_block,
                                )

                                pair_indices = txl.alloc_local((2,), "int32")
                                pair_index_words = pair_indices.view("uint32")
                                with txl.If(is_extra):
                                    with txl.Then():
                                        txl.ptx.ld.global_.nc.v2.u32(
                                            pair_index_words[0],
                                            pair_index_words[1],
                                            extra_indices.view("uint32").ptr_to(
                                                [extra_indices_base + abs_pos]
                                            ),
                                        )
                                    with txl.Else():
                                        txl.ptx.ld.global_.nc.v2.u32(
                                            pair_index_words[0],
                                            pair_index_words[1],
                                            indices.view("uint32").ptr_to([indices_base + abs_pos]),
                                        )
                                bar_valid_free.wait(rs_index.stage, rs_index.phase ^ 1)
                                coords = txl.alloc_local((2,), "int32")
                                cache_blocks = txl.alloc_local((2,), "uint32")
                                indices_in_block = txl.alloc_local((2,), "uint32")
                                scale_words = txl.alloc_local((2,), "uint64")
                                pair_token_valid = txl.alloc_local((2,), "uint32")
                                scale_f32 = txl.alloc_local((2, 4), "float32")
                                scale_byte_offsets = txl.alloc_local((2,), "uint64")

                                def load_token_scales(
                                    pair_i: txl.constexpr,
                                    token_valid,
                                    cache_block,
                                    index_in_block,
                                    block_stride,
                                    row_stride,
                                    scales_ptr_u64,
                                    byte_offsets,
                                    words,
                                    values,
                                ):
                                    if is_v32:
                                        # Invalid V32 entries still issue token-0's
                                        # float4 load, then zero the converted word.
                                        txl.ptx.mov.b64(
                                            byte_offsets[pair_i],
                                            (
                                                txl.if_then_else(
                                                    txl.Cast("bool", token_valid),
                                                    txl.cast(cache_block, "uint64")
                                                    * txl.cast(block_stride, "int64")
                                                    + txl.cast(index_in_block, "uint64")
                                                    * txl.cast(row_stride, "int64"),
                                                    txl.uint64(0),
                                                )
                                            ),
                                        )
                                        txl.ptx.ld.global_.nc.v4.f32(
                                            values[pair_i, 0],
                                            values[pair_i, 1],
                                            values[pair_i, 2],
                                            values[pair_i, 3],
                                            txl.reinterpret(
                                                PointerType(PrimType("float32")),
                                                scales_ptr_u64 + byte_offsets[pair_i],
                                            ),
                                        )
                                    else:
                                        txl.ptx.mov.b64(
                                            byte_offsets[pair_i],
                                            (
                                                txl.cast(cache_block, "uint64")
                                                * txl.cast(block_stride, "int64")
                                                + txl.cast(index_in_block, "uint64") * 8
                                            ),
                                        )
                                        # The offset is unguarded here (unlike the
                                        # V32 arm, which zeroes it), so the load stays
                                        # inside the validity branch.
                                        with txl.If(txl.Cast("bool", token_valid)):
                                            with txl.Then():
                                                txl.ptx.ld.global_.nc.u64(
                                                    words[pair_i],
                                                    txl.reinterpret(
                                                        PointerType(PrimType("uint64")),
                                                        scales_ptr_u64 + byte_offsets[pair_i],
                                                    ),
                                                )
                                            with txl.Else():
                                                txl.ptx.mov.b64(words[pair_i], txl.uint64(0))

                                valid_mask = txl.local_scalar("int8", init=txl.int8(0))
                                with txl.unroll(2) as pair_i:
                                    index_u32 = txl.cast(pair_indices[pair_i], "uint32")
                                    txl.ptx.mov.b32(
                                        cache_blocks[pair_i],
                                        (index_u32 // txl.cast(cur_page_size, "uint32")),
                                    )
                                    txl.ptx.mov.b32(
                                        indices_in_block[pair_i],
                                        (index_u32 % txl.cast(cur_page_size, "uint32")),
                                    )
                                    token_valid = txl.And(
                                        pair_indices[pair_i] != -1, abs_pos + pair_i < cur_length
                                    )
                                    txl.ptx.mov.pred(
                                        pair_token_valid[pair_i], txl.Cast("bool", token_valid)
                                    )
                                    txl.assign(
                                        valid_mask,
                                        txl.cast(
                                            txl.bitwise_or(
                                                txl.cast(valid_mask, "int32"),
                                                txl.shift_left(
                                                    txl.cast(token_valid, "int32"),
                                                    txl.cast(pair_i, "int32"),
                                                ),
                                            ),
                                            "int8",
                                        ),
                                    )
                                    txl.ptx.mov.b32(
                                        coords[pair_i],
                                        txl.if_then_else(
                                            txl.Cast("bool", pair_token_valid[pair_i]),
                                            txl.cast(cache_blocks[pair_i], "int32")
                                            * cur_tma_coords_step_per_block
                                            + txl.cast(indices_in_block[pair_i], "int32")
                                            * tma_coords_step_per_token,
                                            -1,
                                        ),
                                    )
                                    # The source-unrolled loop issues both random
                                    # scale loads before either V32 conversion.
                                    load_token_scales(
                                        pair_i,
                                        pair_token_valid[pair_i],
                                        cache_blocks[pair_i],
                                        indices_in_block[pair_i],
                                        cur_block_stride,
                                        cur_row_stride,
                                        cur_k_scales_ptr_u64,
                                        scale_byte_offsets,
                                        scale_words,
                                        scale_f32,
                                    )

                                if is_v32:
                                    with txl.unroll(2) as pair_i:
                                        lo = txl.local_scalar("uint16")
                                        txl.ptx.cvt.rz.ue8m0x2.f32(
                                            lo, scale_f32[pair_i, 1], scale_f32[pair_i, 0]
                                        )
                                        hi = txl.local_scalar("uint16")
                                        txl.ptx.cvt.rz.ue8m0x2.f32(
                                            hi, scale_f32[pair_i, 3], scale_f32[pair_i, 2]
                                        )
                                        packed_scale = txl.bitwise_or(
                                            txl.cast(lo, "uint32"),
                                            txl.shift_left(txl.cast(hi, "uint32"), txl.uint32(16)),
                                        )
                                        txl.ptx.mov.b64(
                                            scale_words[pair_i],
                                            (
                                                txl.if_then_else(
                                                    txl.Cast("bool", pair_token_valid[pair_i]),
                                                    txl.cast(packed_scale, "uint64"),
                                                    txl.uint64(0),
                                                )
                                            ),
                                        )

                                txl.assign(
                                    valid_mask,
                                    txl.cast(
                                        txl.shift_left(
                                            txl.cast(valid_mask, "int32"),
                                            txl.cast((lane_idx % 4) * 2, "int32"),
                                        ),
                                        "int8",
                                    ),
                                )
                                peer_valid_mask = txl.local_scalar("int32")
                                txl.ptx.shfl_sync.bfly.b32(
                                    peer_valid_mask,
                                    txl.cast(valid_mask, "int32"),
                                    txl.uint32(1),
                                    txl.uint32(0x1F),
                                    txl.uint32(0xFFFFFFFF),
                                )
                                txl.assign(
                                    valid_mask,
                                    txl.cast(
                                        txl.bitwise_or(txl.cast(valid_mask, "int32"), peer_valid_mask),
                                        "int8",
                                    ),
                                )
                                peer_valid_mask = txl.local_scalar("int32")
                                txl.ptx.shfl_sync.bfly.b32(
                                    peer_valid_mask,
                                    txl.cast(valid_mask, "int32"),
                                    txl.uint32(2),
                                    txl.uint32(0x1F),
                                    txl.uint32(0xFFFFFFFF),
                                )
                                txl.assign(
                                    valid_mask,
                                    txl.cast(
                                        txl.bitwise_or(txl.cast(valid_mask, "int32"), peer_valid_mask),
                                        "int8",
                                    ),
                                )
                                if is_v32:
                                    txl.ptx.st.shared.u64(
                                        scales_e8m0.view("uint64").ptr_to(
                                            [rs_index.stage, lane_idx]
                                        ),
                                        txl.bitwise_or(
                                            scale_words[0],
                                            txl.shift_left(scale_words[1], txl.uint64(32)),
                                        ),
                                    )
                                else:
                                    scale_word_bits = scale_words.view("uint32")
                                    txl.ptx.st.shared.v4.u32(
                                        scales_e8m0.view("uint32").ptr_to(
                                            [rs_index.stage, lane_idx * 4]
                                        ),
                                        scale_word_bits[0],
                                        scale_word_bits[1],
                                        scale_word_bits[2],
                                        scale_word_bits[3],
                                    )
                                coord_words = coords.view("uint32")
                                txl.ptx.st.shared.v2.u32(
                                    tma_coord.view("uint32").ptr_to([rs_index.stage, lane_idx * 2]),
                                    coord_words[0],
                                    coord_words[1],
                                )
                                with txl.If(lane_idx % 4 == 0), txl.Then():
                                    txl.ptx.st.shared.b8(
                                        is_token_valid.ptr_to([rs_index.stage, lane_idx // 4]),
                                        txl.reinterpret("uint8", valid_mask),
                                    )
                                bar_valid_ready.arrive(rs_index.stage)
                                rs_buf.advance()
                                rs_index.advance()

                            with txl.serial(
                                start_block, txl.min(num_orig_blocks, end_block), unroll=False
                            ) as block_idx:
                                process_index_block(block_idx, False)
                            with txl.If(txl.And(have_extra_kv, have_extra_indices)), txl.Then():
                                with txl.serial(
                                    txl.max(start_block, num_orig_blocks), end_block, unroll=False
                                ) as block_idx:
                                    process_index_block(block_idx, True)

                        txl.ptx.barrier.sync(txl.uint32(BAR_EVERYONE_SYNC), txl.uint32(NUM_THREADS))
                        batch_epoch.advance()

            with txl.If(selected_wg1_role == 4):
                with txl.Then():
                    run_wg1_role(4)
                with txl.Else():
                    with txl.If(selected_wg1_role == 5):
                        with txl.Then():
                            run_wg1_role(5)
                        with txl.Else():
                            with txl.If(selected_wg1_role == 6):
                                with txl.Then():
                                    run_wg1_role(6)
                                with txl.Else():
                                    with txl.If(selected_wg1_role == 7):
                                        with txl.Then():
                                            run_wg1_role(7)
                                        with txl.Else():
                                            run_wg1_role(-1)

        def dequant():
            # kernel.cuh:747-759.  The dequant warpgroup keeps 208 registers
            # and assigns exactly eight threads per token group.
            rs_buf = txl.PipelineState(NUM_BUFS, phase=0)
            rs_index = txl.PipelineState(NUM_INDEX_BUFS, phase=0)
            group_idx = idx_in_warpgroup // 8
            idx_in_group = idx_in_warpgroup % 8

            # kernel.cuh:751-758.  Keep both dequant-stage and raw-stage
            # per-thread bases live across the scheduler loop.  The source
            # selects one pointer from each pair once per block.
            nope0_base_u64 = txl.reinterpret(
                "uint64", k_full.ptr_to([0, group_idx, idx_in_group * 8])
            )
            nope1_base_u64 = txl.reinterpret(
                "uint64", k_full.ptr_to([1, group_idx, idx_in_group * 8])
            )
            raw_nope0_base_u64 = txl.reinterpret(
                "uint64", raw_nope.ptr_to([0, group_idx, idx_in_group])
            )
            raw_nope1_base_u64 = txl.reinterpret(
                "uint64", raw_nope.ptr_to([1, group_idx, idx_in_group])
            )

            # kernel.cuh:77-118 expanded for WG2, preserving scheduler loads
            # in this specialization to avoid cross-role register pressure.
            (sched_begin_req, sched_end_req, sched_begin_block, sched_end_block) = (
                unpack_scheduler_meta(4)
            )
            batch_epoch = txl.PipelineState(1, phase=0)

            # The CUDA return exits only this role's run_main_loop lambda.
            with txl.If(sched_begin_req < b), txl.Then():
                with txl.serial(sched_begin_req, sched_end_req + 1, unroll=False) as batch_idx:
                    _, _, _, start_block, end_block = batch_block_range(
                        batch_idx,
                        sched_begin_req,
                        sched_end_req,
                        sched_begin_block,
                        sched_end_block,
                    )

                    # kernel.cuh:760-840.  Wait on Q, raw fp8 and the previous
                    # SxV use, then convert each fp8x8 with the exact ue8m0
                    # scale and weak shared b128 store from the source.
                    bar_q_utccp.wait(0, batch_epoch.phase)
                    with txl.serial(start_block, end_block, unroll=False) as block_idx:
                        bar_valid_ready.wait(rs_index.stage, rs_index.phase)
                        bar_raw_ready.wait(rs_buf.stage, rs_buf.phase)
                        bar_sv_done.wait(rs_buf.stage, rs_buf.phase ^ 1)
                        # On the first block, bridge the completed UTCCP read of
                        # q_sw128 before generic stores reuse its k_full alias.  On
                        # later ring turns, bridge the completed SxV read of this
                        # stage before the same generic stores overwrite it.
                        txl.ptx.fence.proxy.async_.shared__cta()
                        cur_nope_base_u64 = txl.if_then_else(
                            rs_buf.stage == 0, nope0_base_u64, nope1_base_u64
                        )
                        cur_raw_nope_base_u64 = txl.if_then_else(
                            rs_buf.stage == 0, raw_nope0_base_u64, raw_nope1_base_u64
                        )
                        cur_nope_base_uint_addr = txl.cuda.cvta_generic_to_shared(
                            txl.reinterpret(PointerType(PrimType("bfloat16")), cur_nope_base_u64)
                        )
                        cur_raw_nope_base_uint_addr = txl.cuda.cvta_generic_to_shared(
                            txl.reinterpret(PointerType(PrimType("uint64")), cur_raw_nope_base_u64)
                        )
                        with txl.unroll(rows_per_group) as local_row:
                            row_idx = local_row * (128 // 8) + group_idx
                            scales_bf16_bits = txl.alloc_local((num_scales,), "uint16")
                            if is_v32:
                                packed_scales = txl.local_scalar("uint32")
                                txl.ptx.ld.shared.u32(
                                    packed_scales,
                                    scales_e8m0.view("uint32").ptr_to([rs_index.stage, row_idx]),
                                )
                                with txl.unroll(2) as scale_pair_idx:
                                    converted_pair = txl.local_scalar("uint32")
                                    txl.ptx.cvt.rn.bf16x2.ue8m0x2(
                                        converted_pair,
                                        txl.cast(
                                            txl.shift_right(
                                                packed_scales, txl.cast(scale_pair_idx * 16, "uint32")
                                            ),
                                            "uint16",
                                        ),
                                    )
                                    txl.ptx.mov.b16(
                                        scales_bf16_bits[scale_pair_idx * 2],
                                        txl.cast(converted_pair, "uint16"),
                                    )
                                    txl.ptx.mov.b16(
                                        scales_bf16_bits[scale_pair_idx * 2 + 1],
                                        txl.cast(
                                            txl.shift_right(converted_pair, txl.uint32(16)), "uint16"
                                        ),
                                    )
                            else:
                                packed_scales = txl.local_scalar("uint64")
                                txl.ptx.ld.shared.u64(
                                    packed_scales,
                                    scales_e8m0.view("uint64").ptr_to([rs_index.stage, row_idx]),
                                )
                                with txl.unroll(4) as scale_pair_idx:
                                    converted_pair = txl.local_scalar("uint32")
                                    txl.ptx.cvt.rn.bf16x2.ue8m0x2(
                                        converted_pair,
                                        txl.cast(
                                            txl.shift_right(
                                                packed_scales, txl.cast(scale_pair_idx * 16, "uint64")
                                            ),
                                            "uint16",
                                        ),
                                    )
                                    txl.ptx.mov.b16(
                                        scales_bf16_bits[scale_pair_idx * 2],
                                        txl.cast(converted_pair, "uint16"),
                                    )
                                    txl.ptx.mov.b16(
                                        scales_bf16_bits[scale_pair_idx * 2 + 1],
                                        txl.cast(
                                            txl.shift_right(converted_pair, txl.uint32(16)), "uint16"
                                        ),
                                    )

                            cur_raw_fp8x8 = txl.local_scalar("uint64")
                            txl.ptx.ld.shared.u64(
                                cur_raw_fp8x8,
                                cur_raw_nope_base_uint_addr
                                + txl.cast(local_row * (128 // 8) * d_nope, "uint32"),
                            )
                            with txl.unroll(cols_per_group) as local_col:
                                raw_fp8x8 = txl.local_scalar("uint64")
                                txl.ptx.mov.b64(raw_fp8x8, cur_raw_fp8x8)
                                with txl.If(local_col + 1 < cols_per_group), txl.Then():
                                    txl.ptx.ld.shared.u64(
                                        cur_raw_fp8x8,
                                        cur_raw_nope_base_uint_addr
                                        + txl.cast(
                                            local_row * (128 // 8) * d_nope
                                            + (local_col + 1) * (8 * 8),
                                            "uint32",
                                        ),
                                    )
                                scale_idx = (
                                    local_col // (cols_per_group // 4) if is_v32 else local_col
                                )
                                dequant_st128(
                                    cur_nope_base_uint_addr
                                    + txl.cast(
                                        BF16_BYTES
                                        * (local_row * (128 // 8) * 64 + local_col * B_TOPK * 64),
                                        "uint32",
                                    ),
                                    raw_fp8x8,
                                    scales_bf16_bits[scale_idx],
                                )
                        txl.ptx.fence.proxy.async_.shared__cta()
                        bar_nope_ready.arrive(rs_buf.stage)
                        bar_raw_free.arrive(rs_buf.stage)
                        bar_valid_free.arrive(rs_index.stage)
                        rs_buf.advance()
                        rs_index.advance()

                    txl.ptx.barrier.sync(txl.uint32(BAR_EVERYONE_SYNC), txl.uint32(NUM_THREADS))
                    batch_epoch.advance()

        roles = txl.specialize(chain_dispatch=True)
        scale = roles.role("scale_exp_output", warps=range(0, 4), regs=224)
        producer = roles.warpgroup("producer_mma", warps=range(4, 8), regs=72)
        q_mma = roles.role("q_mma", warps=4, group=producer)
        raw_nope_tma = roles.role("raw_nope_tma", warps=5, group=producer)
        rope_tma = roles.role("rope_tma", warps=6, group=producer)
        index_scale = roles.role("index_scale", warps=7, group=producer)
        convert = roles.role("dequant", warps=range(8, 12), regs=208)
        with scale:
            scale_exp_output()
        with producer:
            # kernel.cuh:431/586/616/653/744.  K owns the four producer warp
            # predicates.  Election remains lane-local: every non-elected lane
            # takes the one shared empty scheduler and still reaches the
            # 384-thread per-batch barrier.
            selected_wg1_role = txl.local_scalar("int32", init=-1)
            with q_mma:
                with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                    txl.assign(selected_wg1_role, 4)
            with raw_nope_tma:
                with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                    txl.assign(selected_wg1_role, 5)
            with rope_tma:
                with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                    txl.assign(selected_wg1_role, 6)
            with index_scale:
                txl.assign(selected_wg1_role, 7)
            producer_mma(selected_wg1_role)
        with convert:
            dequant()

    return sparse_flashmla_decode_head64_main


def make_combine_kernel(max_splits, have_attn_sink, use_pdl=False):
    @txl.kernel(warps=8, arch="sm_100a", grid=False)
    def sparse_decode_head64_combine(
        lse: txl.gptr[txl.f32],
        out: txl.gptr[txl.bf16],
        lse_accum: txl.gptr[txl.f32],
        o_accum: txl.gptr[txl.f32],
        num_splits: txl.gptr[txl.i32],
        attn_sink: txl.gptr[txl.f32],
        stride_lse_b: txl.i32,
        stride_lse_s_q: txl.i32,
        stride_o_b: txl.i32,
        stride_o_s_q: txl.i32,
        stride_o_h_q: txl.i32,
        stride_lse_accum_split: txl.i32,
        stride_lse_accum_s_q: txl.i32,
        stride_o_accum_split: txl.i32,
        stride_o_accum_s_q: txl.i32,
        stride_o_accum_h_q: txl.i32,
        b: txl.i32,
        s_q: txl.i32,
        h_q: txl.i32,
        d_v: txl.i32,
        num_sm_parts: txl.i32,
    ):
        smem = txl.smem_pool()
        lse_scales = smem.alloc((8, max_splits), "float32")

        # combine.cu:18-43. One warp per head, eight heads per CTA.
        batch_s_q_idx, _, h_block_idx = txl.cta_id([b * s_q, 1, (h_q + 7) // 8])
        thread_idx = txl.thread_id()
        warp_idx = txl.local_scalar("int32", init=thread_idx // 32)
        lane_idx = txl.local_scalar("int32", init=thread_idx % 32)
        batch_idx = txl.local_scalar("int32", init=batch_s_q_idx // s_q)
        query_idx = txl.local_scalar("int32", init=batch_s_q_idx - batch_idx * s_q)
        h_block_base = txl.local_scalar("int32", init=h_block_idx * 8)
        head_idx = txl.local_scalar("int32", init=h_block_base + warp_idx)
        num_valid_heads = txl.local_scalar("int32", init=txl.min(8, h_q - h_block_base))
        with txl.If(warp_idx >= num_valid_heads), txl.Then():
            txl.Return(txl.int32(0))

        start_split = txl.local_scalar("int32")
        txl.ptx.ld.global_.nc.s32(start_split, num_splits.ptr_to([batch_idx]))
        end_split = txl.local_scalar("int32")
        txl.ptx.ld.global_.nc.s32(end_split, num_splits.ptr_to([batch_idx + 1]))
        my_num_splits = txl.local_scalar("int32", init=end_split - start_split)
        with txl.If(my_num_splits == 1), txl.Then():
            txl.Return(txl.int32(0))

        txl.cuda.trap_when_assert_failed(my_num_splits <= max_splits)

        # combine.cu:45-54. Preserve the source base views.
        g_lse_accum_offset = txl.local_scalar(
            "int32",
            init=(
                start_split * stride_lse_accum_split
                + query_idx * stride_lse_accum_s_q
                + h_block_base
            ),
        )
        g_lse_offset = txl.local_scalar(
            "int32", init=batch_idx * stride_lse_b + query_idx * stride_lse_s_q + h_block_base
        )
        g_lse_accum = txl.decl_buffer(
            (max_splits * stride_lse_accum_split + 8,),
            "float32",
            data=lse_accum.data,
            scope="global",
            elem_offset=g_lse_accum_offset,
        )
        g_lse = txl.decl_buffer(
            (8,), "float32", data=lse.data, scope="global", elem_offset=g_lse_offset
        )

        # combine.cu:58-69. PDL wait follows both early returns.
        if use_pdl:
            txl.ptx.griddepcontrol.wait()
        oaccum_offset = txl.local_scalar(
            "int32",
            init=(
                start_split * stride_o_accum_split
                + query_idx * stride_o_accum_s_q
                + head_idx * stride_o_accum_h_q
            ),
        )
        oaccum_ptr = txl.decl_buffer(
            (num_sm_parts * stride_o_accum_split + D_V,),
            "float32",
            data=o_accum.data,
            scope="global",
            elem_offset=oaccum_offset,
        )
        datas = txl.alloc_local((D_V // (32 * 4), 4), "float32")
        data_words = datas.view("uint32")
        with txl.unroll(D_V // (32 * 4)) as elem_i:
            txl.ptx.ld.global_.v4.u32(
                data_words[elem_i, 0],
                data_words[elem_i, 1],
                data_words[elem_i, 2],
                data_words[elem_i, 3],
                oaccum_ptr.view("uint32").ptr_to([lane_idx * 4 + elem_i * 128]),
            )

        # combine.cu:71-119. Gather LSE, reduce, and normalize.
        lse_fragments = (max_splits + 31) // 32
        local_lse = txl.alloc_local((lse_fragments,), "float32")
        with txl.unroll(lse_fragments) as lse_i:
            split_idx = txl.local_scalar("int32", init=lse_i * 32 + lane_idx)
            txl.ptx.mov.b32(local_lse[lse_i], txl.float32(-float("inf")))
            with txl.If(split_idx < my_num_splits), txl.Then():
                txl.ptx.ld.global_.f32(
                    local_lse[lse_i],
                    g_lse_accum.ptr_to([split_idx * stride_lse_accum_split + warp_idx]),
                )
        max_lse = txl.alloc_local((1,), "float32")
        txl.assign(max_lse[0], txl.float32(-float("inf")))
        with txl.unroll(lse_fragments) as lse_i:
            txl.assign(max_lse[0], txl.max(max_lse[0], local_lse[lse_i]))
        with txl.unroll(5) as reduce_i:
            xor_offset = txl.local_scalar("int32", init=16 >> reduce_i)
            peer_max_lse = txl.local_scalar("float32")
            txl.ptx.shfl_sync.bfly.b32(
                peer_max_lse,
                max_lse[0],
                txl.cast(xor_offset, "uint32"),
                txl.uint32(0x1F),
                txl.uint32(0xFFFFFFFF),
            )
            txl.assign(max_lse[0], txl.max(max_lse[0], peer_max_lse))
        txl.assign(
            max_lse[0],
            txl.if_then_else(max_lse[0] == txl.float32(-float("inf")), txl.float32(0.0), max_lse[0]),
        )
        sum_lse = txl.alloc_local((1,), "float32")
        lse_exp = txl.alloc_local((1,), "float32")
        txl.assign(sum_lse[0], txl.float32(0.0))
        with txl.unroll(lse_fragments) as lse_i:
            txl.ptx.ex2.approx.ftz.f32(lse_exp[0], local_lse[lse_i] - max_lse[0])
            txl.assign(sum_lse[0], sum_lse[0] + lse_exp[0])
        with txl.unroll(5) as reduce_i:
            xor_offset = txl.local_scalar("int32", init=16 >> reduce_i)
            peer_sum_lse = txl.local_scalar("float32")
            txl.ptx.shfl_sync.bfly.b32(
                peer_sum_lse,
                sum_lse[0],
                txl.cast(xor_offset, "uint32"),
                txl.uint32(0x1F),
                txl.uint32(0xFFFFFFFF),
            )
            txl.assign(sum_lse[0], sum_lse[0] + peer_sum_lse)
        global_lse = txl.alloc_local((1,), "float32")
        txl.assign(
            global_lse[0],
            txl.if_then_else(
                txl.Or(sum_lse[0] == 0.0, sum_lse[0] == txl.float32(-float("inf"))),
                txl.float32(float("inf")),
                txl.log2(sum_lse[0]) + max_lse[0],
            ),
        )
        with txl.If(lane_idx == 0), txl.Then():
            txl.ptx.st.global_.f32(g_lse.ptr_to([warp_idx]), global_lse[0] / txl.float32(LOG_2_E))

        if have_attn_sink:
            sink = txl.local_scalar("float32")
            txl.ptx.ld.global_.nc.f32(sink, attn_sink.ptr_to([head_idx]))
            with txl.If(global_lse[0] != txl.float32(float("inf"))):
                with txl.Then():
                    sink_lse_exp = txl.alloc_local((1,), "float32")
                    txl.ptx.ex2.approx.ftz.f32(sink_lse_exp[0], sink * LOG_2_E - global_lse[0])
                    txl.assign(global_lse[0], global_lse[0] + txl.log2(1.0 + sink_lse_exp[0]))
                with txl.Else():
                    txl.assign(
                        global_lse[0],
                        txl.if_then_else(
                            sink == txl.float32(-float("inf")),
                            txl.float32(float("inf")),
                            sink * LOG_2_E,
                        ),
                    )
        lse_scale_value = txl.alloc_local((1,), "float32")
        with txl.unroll(lse_fragments) as lse_i:
            split_idx = txl.local_scalar("int32", init=lse_i * 32 + lane_idx)
            txl.ptx.ex2.approx.ftz.f32(lse_scale_value[0], local_lse[lse_i] - global_lse[0])
            txl.ptx.st.shared.f32(lse_scales.ptr_to([warp_idx, split_idx]), lse_scale_value[0])
        txl.cuda.warp_sync()

        # combine.cu:123-160. Keep the serial traversal and next-split prefetch.
        result = txl.alloc_local((D_V // (32 * 4), 4), "float32")
        with txl.unroll(D_V // (32 * 4)) as elem_i:
            with txl.unroll(4) as vec_i:
                txl.ptx.mov.b32(result[elem_i, vec_i], txl.float32(0.0))
        lse_scale = txl.alloc_local((1,), "float32")
        with txl.serial(my_num_splits, unroll=False) as split_idx:
            txl.ptx.ld.shared.f32(lse_scale[0], lse_scales.ptr_to([warp_idx, split_idx]))
            with txl.unroll(D_V // (32 * 4)) as elem_i:
                with txl.unroll(4) as vec_i:
                    txl.ptx.fma.rn.f32(
                        result[elem_i, vec_i],
                        lse_scale[0],
                        datas[elem_i, vec_i],
                        result[elem_i, vec_i],
                    )
                with txl.If(split_idx != my_num_splits - 1), txl.Then():
                    txl.ptx.ld.global_.v4.u32(
                        data_words[elem_i, 0],
                        data_words[elem_i, 1],
                        data_words[elem_i, 2],
                        data_words[elem_i, 3],
                        oaccum_ptr.view("uint32").ptr_to(
                            [(split_idx + 1) * stride_o_accum_split + lane_idx * 4 + elem_i * 128]
                        ),
                    )

        out_offset = txl.local_scalar(
            "int32",
            init=batch_idx * stride_o_b + query_idx * stride_o_s_q + head_idx * stride_o_h_q,
        )
        o_ptr = txl.decl_buffer(
            (D_V,), "bfloat16", data=out.data, scope="global", elem_offset=out_offset
        )
        with txl.unroll(D_V // (32 * 4)) as elem_i:
            data_converted = txl.alloc_local((4,), "uint16")
            txl.ptx.cvt.rn.bf16.f32(data_converted[0], result[elem_i, 0])
            txl.ptx.cvt.rn.bf16.f32(data_converted[1], result[elem_i, 1])
            txl.ptx.cvt.rn.bf16.f32(data_converted[2], result[elem_i, 2])
            txl.ptx.cvt.rn.bf16.f32(data_converted[3], result[elem_i, 3])
            txl.ptx.st.global_.u64(
                o_ptr.view("uint64").ptr_to([(lane_idx * 4 + elem_i * 128) // 4]),
                data_converted.view("uint64")[0],
            )

    return sparse_decode_head64_combine


def _kernel_shape_params(
    cfg: SparseFlashMLADecodeHead64Config,
    device: torch.device | str,
    *,
    prepared_num_sms: int | None = None,
    prepared_num_blocks: int | None = None,
    prepared_extra_num_blocks: int | None = None,
) -> dict[str, int]:
    if prepared_num_sms is None:
        device_obj = torch.device(device)
        device_index = device_obj.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        num_sms = int(torch.cuda.get_device_properties(device_index).multi_processor_count)
    else:
        num_sms = prepared_num_sms
    num_sm_parts = num_sms // cfg.s_q
    num_sm_parts = max(num_sm_parts, 1)
    num_blocks = (
        prepared_num_blocks
        if prepared_num_blocks is not None
        else _ceil_div(cfg.s_kv, cfg.page_block_size)
    )
    _, tma_k_stride, stride_kv_block, num_tma_rows = _kv_storage_spec(
        cfg.normalized_model_type, num_blocks, cfg.page_block_size
    )

    if cfg.extra_topk:
        extra_num_blocks = (
            prepared_extra_num_blocks
            if prepared_extra_num_blocks is not None
            else _ceil_div(cfg.extra_s_kv, cfg.extra_page_block_size)
        )
        (_, extra_tma_k_stride, stride_extra_kv_block, extra_num_tma_rows) = _kv_storage_spec(
            cfg.normalized_model_type, extra_num_blocks, cfg.extra_page_block_size
        )
        if extra_tma_k_stride != tma_k_stride:
            raise AssertionError("original and extra KV caches must use one MODEL_TYPE")
        extra_page_block_size = cfg.extra_page_block_size
    else:
        # kernel.cuh:934-950 leaves all optional runtime shape/stride fields at
        # zero when extra KV is absent.  Optional specialization removes the
        # extra buffer views and their generated descriptors entirely.
        extra_num_blocks = 0
        extra_page_block_size = 0
        stride_extra_kv_block = 0
        extra_num_tma_rows = 0

    max_splits = next((bucket for bucket in (32, 64, 96, 128, 160) if num_sm_parts <= bucket), None)
    if max_splits is None:
        raise ValueError(f"FlashMLA combine supports at most 160 SM partitions, got {num_sm_parts}")

    return {
        "num_sm_parts": num_sm_parts,
        "num_blocks": num_blocks,
        "stride_kv_block": stride_kv_block,
        "num_tma_rows": num_tma_rows,
        "kv_bytes": num_blocks * stride_kv_block,
        "extra_num_blocks": extra_num_blocks,
        "extra_page_block_size": extra_page_block_size,
        "stride_extra_kv_block": stride_extra_kv_block,
        "extra_num_tma_rows": extra_num_tma_rows,
        "extra_kv_bytes": extra_num_blocks * stride_extra_kv_block,
        "extra_indices_elems": cfg.b * cfg.s_q * cfg.extra_topk,
        "split_rows": cfg.b + num_sm_parts,
        "max_splits": max_splits,
    }


def _main_presence_mask(cfg: SparseFlashMLADecodeHead64Config) -> MainPresenceMask:
    have_extra_kv = cfg.extra_topk != 0
    return (
        cfg.have_topk_length,
        cfg.have_attn_sink,
        have_extra_kv,
        have_extra_kv,
        cfg.have_extra_topk_length,
    )


def _absent_specialization_kwargs(
    optional_names: tuple[str, ...], presence: tuple[bool, ...]
) -> dict[str, None]:
    return {
        name: None
        for name, is_present in zip(optional_names, presence, strict=True)
        if not is_present
    }


@lru_cache(maxsize=64)
def _specialized_main_kernel(
    model_type: ModelType, presence: MainPresenceMask, use_pdl: bool = False
):
    return (
        make_main_kernel(model_type, presence, use_pdl)
        .func.with_attr("global_symbol", KERNEL_META["name"])
        .with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))
    )


@lru_cache(maxsize=20)
def _specialized_combine_kernel(max_splits: int, have_attn_sink: bool, use_pdl: bool = False):
    return (
        make_combine_kernel(max_splits, have_attn_sink, use_pdl)
        .func.with_attr("global_symbol", "sparse_flashmla_decode_head64_combine")
        .with_attr(
            "tirx.kernel_launch_params",
            list(COMBINE_PDL_LAUNCH_TAGS if use_pdl else COMBINE_LAUNCH_TAGS),
        )
    )


def _specialized_decode_kernels(
    model_type: ModelType, max_splits: int, presence: MainPresenceMask, use_pdl: bool = False
):
    if not use_pdl:
        return (
            _specialized_main_kernel(model_type, presence),
            _specialized_combine_kernel(max_splits, presence[1]),
        )
    return (
        _specialized_main_kernel(model_type, presence, True),
        _specialized_combine_kernel(max_splits, presence[1], True),
    )


def get_kernel(**kwargs: Any):
    cfg = _cfg(**kwargs)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA decode")
    device = kwargs.get("device", "cuda")
    shape = _kernel_shape_params(cfg, device)
    return list(
        _specialized_decode_kernels(
            cfg.normalized_model_type, shape["max_splits"], _main_presence_mask(cfg), cfg.b == 2
        )
    )


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA decode")
    device = torch.device(kwargs.get("device", "cuda"))
    props = torch.cuda.get_device_properties(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    if props.major != 10:
        raise SkipTest(f"SM100f is required, got compute capability {props.major}.{props.minor}")

    device_generator = torch.Generator(device=device)
    device_generator.manual_seed(cfg.seed)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(cfg.seed)
    python_rng = random.Random(cfg.seed)

    q_contiguous = torch.randn(
        (cfg.b, cfg.s_q, cfg.h_q, cfg.d_qk),
        dtype=torch.bfloat16,
        device=device,
        generator=device_generator,
    )
    q_contiguous.clamp_(min=-1.0, max=1.0)

    if cfg.have_attn_sink:
        attn_sink = torch.randn(
            (cfg.h_q,), dtype=torch.float32, device=device, generator=device_generator
        )
        inf_mask = torch.randn(
            (cfg.h_q,), dtype=torch.float32, device=device, generator=device_generator
        )
        attn_sink[inf_mask > 0.5] = float("inf")
        attn_sink[inf_mask < -0.5] = -float("inf")
    else:
        attn_sink = torch.zeros((cfg.h_q,), dtype=torch.float32, device=device)

    scope_specs = [(cfg.s_kv, cfg.topk, cfg.page_block_size, cfg.have_topk_length)]
    if cfg.extra_topk:
        scope_specs.append(
            (cfg.extra_s_kv, cfg.extra_topk, cfg.extra_page_block_size, cfg.have_extra_topk_length)
        )

    prepared_scopes = []
    for s_kv, topk, page_block_size, have_topk_length in scope_specs:
        cache_seqlens_cpu = torch.full((cfg.b,), s_kv, dtype=torch.int32)
        if cfg.is_varlen:
            for batch_idx in range(cfg.b):
                cache_seqlens_cpu[batch_idx] = int(
                    max(python_rng.normalvariate(s_kv, s_kv / 2), cfg.s_q)
                )
        if cfg.have_zero_seqlen_k:
            zero_mask = torch.randn((cfg.b,), dtype=torch.float32, generator=cpu_generator) > 0
            cache_seqlens_cpu[zero_mask] = 0

        max_seqlen_alignment = 4 * page_block_size
        max_seqlen_pad = (
            max(_ceil_div(int(cache_seqlens_cpu.max().item()), max_seqlen_alignment), 1)
            * max_seqlen_alignment
        )
        blocks_per_sequence = max_seqlen_pad // page_block_size
        num_blocks = cfg.b * blocks_per_sequence

        block_ids = torch.arange(num_blocks, dtype=torch.int32, device=device)
        block_table = block_ids.index_select(
            0, torch.randperm(num_blocks, device=device, generator=device_generator)
        ).view(cfg.b, blocks_per_sequence)

        source_shape = (num_blocks, page_block_size, cfg.h_kv, cfg.d_qk)
        source_storage = torch.randn(
            tuple(
                dim + 128 if dim_idx == len(source_shape) - 1 else dim + 1
                for dim_idx, dim in enumerate(source_shape)
            ),
            dtype=torch.bfloat16,
            device=device,
            generator=device_generator,
        )
        source = source_storage[tuple(slice(0, dim) for dim in source_shape)] / 10
        source.clamp_(min=-1.0, max=1.0)

        if cfg.is_all_indices_invalid:
            absolute_indices = torch.full(
                (cfg.b, cfg.s_q, topk), -1, dtype=torch.int32, device=device
            )
        else:
            permutation_ranges = cache_seqlens_cpu.to(device=device).repeat_interleave(cfg.s_q)
            max_range = max(int(permutation_ranges.max().item()), topk)
            random_values = torch.rand(
                (permutation_ranges.numel(), max_range),
                dtype=torch.float32,
                device=device,
                generator=device_generator,
            )
            positions = torch.arange(max_range, device=device)
            random_values.masked_fill_(
                positions.view(1, -1) >= permutation_ranges.view(-1, 1), -math.inf
            )
            absolute_indices = (
                random_values.topk(topk, dim=-1, sorted=True)
                .indices.to(torch.int32)
                .view(cfg.b, cfg.s_q, topk)
            )
            absolute_indices.masked_fill_(
                absolute_indices >= permutation_ranges.view(cfg.b, cfg.s_q, 1), -1
            )

        safe_indices = absolute_indices.clamp_min(0)
        batch_block_offsets = (
            torch.arange(cfg.b, dtype=torch.int32, device=device) * blocks_per_sequence
        ).view(cfg.b, 1, 1)
        block_lookup = safe_indices // page_block_size + batch_block_offsets
        physical_blocks = block_table.view(-1).index_select(0, block_lookup.view(-1).long())
        indices = (
            physical_blocks.view(cfg.b, cfg.s_q, topk) * page_block_size
            + safe_indices % page_block_size
        )
        indices.masked_fill_(absolute_indices < 0, -1)

        if have_topk_length:
            topk_length = torch.randint(
                0, topk + 1, (cfg.b,), dtype=torch.int32, device=device, generator=device_generator
            )
            topk_length_cpu = topk_length.cpu()
            masked_indices = indices.clone()
            masked_indices.masked_fill_(
                torch.arange(topk, device=device).view(1, 1, topk) >= topk_length.view(cfg.b, 1, 1),
                -1,
            )
        else:
            topk_length = torch.zeros((cfg.b,), dtype=torch.int32, device=device)
            topk_length_cpu = torch.zeros((cfg.b,), dtype=torch.int32)
            masked_indices = indices

        nonused_tokens = torch.ones(
            (num_blocks * page_block_size,), dtype=torch.bool, device=device
        )
        nonused_tokens[masked_indices.long()] = False
        source.view(-1, cfg.d_qk)[nonused_tokens] = float("nan")

        bytes_per_token, _, stride_kv_block, num_tma_rows = _kv_storage_spec(
            cfg.normalized_model_type, num_blocks, page_block_size
        )
        kv_storage = torch.empty((num_blocks * stride_kv_block,), dtype=torch.uint8, device=device)
        source_rows = source[:, :, 0, :]
        if cfg.normalized_model_type is ModelType.V32:
            d_nope, tile_size, num_tiles = 512, 128, 4
            physical_rows = kv_storage.as_strided(
                (num_blocks, page_block_size, 656), (stride_kv_block, 656, 1)
            )
            scale_view = physical_rows[:, :, 512:528].view(torch.float32)
            physical_rows[:, :, 528:656].view(torch.bfloat16).copy_(source_rows[:, :, d_nope:])
            for tile_idx in range(num_tiles):
                values = source_rows[
                    :, :, tile_idx * tile_size : (tile_idx + 1) * tile_size
                ].float()
                scale = torch.pow(
                    2.0, (values.abs().amax(dim=-1) / 448.0).clamp_min(1.0e-4).log2().ceil()
                )
                physical_rows[:, :, tile_idx * tile_size : (tile_idx + 1) * tile_size].copy_(
                    (values / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(torch.uint8)
                )
                scale_view[:, :, tile_idx].copy_(scale)
        else:
            d_nope, tile_size, num_tiles = 448, 64, 7
            physical_rows = kv_storage.as_strided(
                (num_blocks, page_block_size, 576), (stride_kv_block, 576, 1)
            )
            scale_rows = kv_storage.as_strided(
                (num_blocks, page_block_size, 8),
                (stride_kv_block, 8, 1),
                storage_offset=page_block_size * 576,
            )
            physical_rows[:, :, d_nope:576].view(torch.bfloat16).copy_(source_rows[:, :, d_nope:])
            for tile_idx in range(num_tiles):
                values = source_rows[
                    :, :, tile_idx * tile_size : (tile_idx + 1) * tile_size
                ].float()
                scale = torch.pow(
                    2.0, (values.abs().amax(dim=-1) / 448.0).clamp_min(1.0e-4).log2().ceil()
                )
                physical_rows[:, :, tile_idx * tile_size : (tile_idx + 1) * tile_size].copy_(
                    (values / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(torch.uint8)
                )
                scale_rows[:, :, tile_idx].copy_(scale.to(torch.float8_e8m0fnu).view(torch.uint8))
        del source

        kv = kv_storage.view(torch.float8_e4m3fn).as_strided(
            (num_blocks, page_block_size, 1, bytes_per_token),
            (stride_kv_block, bytes_per_token, bytes_per_token, 1),
        )
        indices_storage = torch.empty(
            (cfg.b + 1, cfg.s_q + 1, topk + 128), dtype=torch.int32, device=device
        )
        indices_view = indices_storage[: cfg.b, : cfg.s_q, :topk]
        indices_view.copy_(indices)
        prepared_scopes.append(
            {
                "kv": kv,
                "kv_storage": kv_storage,
                "indices": indices_view,
                "topk_length": topk_length,
                "topk_length_cpu": topk_length_cpu,
                "cache_seqlens_cpu": cache_seqlens_cpu,
                "num_blocks": num_blocks,
                "stride_kv_block": stride_kv_block,
                "num_tma_rows": num_tma_rows,
            }
        )

    kv_scope = prepared_scopes[0]
    extra_scope = prepared_scopes[1] if cfg.extra_topk else None

    shape = _kernel_shape_params(
        cfg,
        device,
        prepared_num_blocks=kv_scope["num_blocks"],
        prepared_extra_num_blocks=(extra_scope["num_blocks"] if extra_scope is not None else None),
    )
    q_storage = torch.empty(
        (cfg.b + 1, cfg.s_q + 1, cfg.h_q + 1, cfg.d_qk + 128), dtype=torch.bfloat16, device=device
    )
    q = q_storage[: cfg.b, : cfg.s_q, : cfg.h_q, : cfg.d_qk]
    q.copy_(q_contiguous)
    del q_contiguous
    indices = kv_scope["indices"]
    if cfg.inject_invalid_indices:
        indices[:, :, 0] = -1
        indices[:, :, -1] = -1
        if extra_scope is not None:
            extra_scope["indices"][:, :, 0] = -1
            extra_scope["indices"][:, :, -1] = -1

    topk_length_cpu = kv_scope["topk_length_cpu"]
    extra_topk_length_cpu = (
        extra_scope["topk_length_cpu"]
        if extra_scope is not None
        else torch.zeros((cfg.b,), dtype=torch.int32)
    )

    block_size_n = B_TOPK
    fixed_overhead_num_blocks = 5
    seqlens_k = []
    num_blocks_per_request = []
    first_block_idx = []
    last_block_idx = []
    total_num_blocks = 0
    for batch_idx in range(cfg.b):
        cur_s_k = int(topk_length_cpu[batch_idx]) if cfg.have_topk_length else cfg.topk
        if cur_s_k == 0:
            cur_s_k = 1
        if cfg.extra_topk:
            cur_s_k = _ceil_div(cur_s_k, block_size_n) * block_size_n
            cur_s_k += (
                int(extra_topk_length_cpu[batch_idx])
                if cfg.have_extra_topk_length
                else cfg.extra_topk
            )
        seqlens_k.append(cur_s_k)
        last = max(cur_s_k - 1, 0) // block_size_n
        blocks = last + 1
        first_block_idx.append(0)
        last_block_idx.append(last)
        num_blocks_per_request.append(blocks)
        total_num_blocks += blocks + fixed_overhead_num_blocks

    num_sm_parts = shape["num_sm_parts"]
    payload = _ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks
    tile_scheduler_metadata = torch.zeros((num_sm_parts, 8), dtype=torch.int32)
    num_splits = torch.zeros((cfg.b + 1,), dtype=torch.int32)
    now_req_idx = 0
    now_block = 0
    now_n_split_idx = 0
    cum_num_splits = 0
    for partition_idx in range(num_sm_parts):
        if now_req_idx >= cfg.b:
            tile_scheduler_metadata[partition_idx, 0] = cfg.b
            continue

        begin_req_idx = now_req_idx
        begin_block_idx = now_block + first_block_idx[now_req_idx]
        begin_split_idx = now_n_split_idx
        is_first_req_splitted = int(now_block != 0)
        remain_payload = payload
        while now_req_idx < cfg.b:
            now_remain_blocks = num_blocks_per_request[now_req_idx] - now_block
            if remain_payload >= now_remain_blocks + fixed_overhead_num_blocks:
                cum_num_splits += now_n_split_idx + 1
                num_splits[now_req_idx + 1] = cum_num_splits
                remain_payload -= now_remain_blocks + fixed_overhead_num_blocks
                now_req_idx += 1
                now_block = 0
                now_n_split_idx = 0
            else:
                if remain_payload - fixed_overhead_num_blocks > 0:
                    now_block += remain_payload - fixed_overhead_num_blocks
                    now_n_split_idx += 1
                break

        end_req_idx = now_req_idx if now_block > 0 else now_req_idx - 1
        if now_block > 0:
            end_block_idx = now_block + first_block_idx[now_req_idx]
        else:
            prev_req_idx = now_req_idx - 1
            end_block_idx = 0 if seqlens_k[prev_req_idx] == 0 else last_block_idx[prev_req_idx] + 1
        is_last_req_splitted = int(
            end_block_idx != last_block_idx[end_req_idx] + 1 and seqlens_k[end_req_idx] != 0
        )
        if begin_req_idx == end_req_idx:
            split = int(bool(is_first_req_splitted or is_last_req_splitted))
            is_first_req_splitted = split
            is_last_req_splitted = split
        tile_scheduler_metadata[partition_idx] = torch.tensor(
            [
                begin_req_idx,
                end_req_idx,
                begin_block_idx,
                end_block_idx,
                begin_split_idx,
                is_first_req_splitted,
                is_last_req_splitted,
                0,
            ],
            dtype=torch.int32,
        )

    if not (now_req_idx == cfg.b and now_block == 0 and now_n_split_idx == 0):
        raise RuntimeError("host scheduler did not consume every sparse decode request")
    tile_scheduler_metadata = tile_scheduler_metadata.to(device=device)
    num_splits = num_splits.to(device=device)

    out_elements = cfg.b * cfg.s_q * cfg.h_q * D_V
    out_storage = torch.empty((out_elements + cfg.h_q * D_V,), dtype=torch.bfloat16, device=device)
    out = out_storage[:out_elements].view(cfg.b, cfg.s_q, cfg.h_q, D_V)
    lse = torch.empty((cfg.b, cfg.s_q, cfg.h_q), dtype=torch.float32, device=device)
    lse_accum = torch.empty(
        (shape["split_rows"], cfg.s_q, cfg.h_q), dtype=torch.float32, device=device
    )
    o_accum = torch.empty(
        (shape["split_rows"], cfg.s_q, cfg.h_q, D_V), dtype=torch.float32, device=device
    )
    sm_scale = cfg.d_qk**-0.55
    case = {
        "config": cfg,
        "shape": shape,
        "q": q,
        "kv": kv_scope["kv"],
        "kv_storage": kv_scope["kv_storage"],
        "indices": indices,
        "topk_length": kv_scope["topk_length"],
        "cache_seqlens_cpu": kv_scope["cache_seqlens_cpu"],
        "attn_sink": attn_sink,
        "lse": lse,
        "out": out,
        "lse_accum": lse_accum,
        "o_accum": o_accum,
        "tile_scheduler_metadata": tile_scheduler_metadata,
        "num_splits": num_splits,
        "extra_kv": extra_scope["kv"] if extra_scope is not None else None,
        "extra_kv_storage": extra_scope["kv_storage"] if extra_scope is not None else None,
        "extra_indices": extra_scope["indices"] if extra_scope is not None else None,
        "extra_topk_length": (
            extra_scope["topk_length"]
            if extra_scope is not None
            else torch.zeros((cfg.b,), dtype=torch.int32, device=device)
        ),
        "extra_cache_seqlens_cpu": (
            extra_scope["cache_seqlens_cpu"] if extra_scope is not None else None
        ),
        "sm_scale": sm_scale,
        "sm_scale_div_log2": sm_scale * LOG_2_E,
        "stride_q_b": q.stride(0),
        "stride_q_s_q": q.stride(1),
        "stride_q_h_q": q.stride(2),
        "stride_kv_block": kv_scope["kv"].stride(0),
        "stride_kv_row": kv_scope["kv"].stride(1),
        "stride_indices_b": indices.stride(0),
        "stride_indices_s_q": indices.stride(1),
        "stride_lse_b": lse.stride(0),
        "stride_lse_s_q": lse.stride(1),
        "stride_o_b": out.stride(0),
        "stride_o_s_q": out.stride(1),
        "stride_o_h_q": out.stride(2),
        "stride_extra_kv_block": extra_scope["kv"].stride(0) if extra_scope is not None else 0,
        "stride_extra_kv_row": extra_scope["kv"].stride(1) if extra_scope is not None else 0,
        "stride_extra_indices_b": (
            extra_scope["indices"].stride(0) if extra_scope is not None else 0
        ),
        "stride_extra_indices_s_q": (
            extra_scope["indices"].stride(1) if extra_scope is not None else 0
        ),
        "stride_lse_accum_split": lse_accum.stride(0),
        "stride_lse_accum_s_q": lse_accum.stride(1),
        "stride_o_accum_split": o_accum.stride(0),
        "stride_o_accum_s_q": o_accum.stride(1),
        "stride_o_accum_h_q": o_accum.stride(2),
    }
    _validate_tirx_launch_case(case)
    return case


def _validate_tirx_launch_case(case: dict[str, Any]) -> None:
    """Validate the runtime storage assumptions encoded by the TMA views."""

    cfg: SparseFlashMLADecodeHead64Config = case["config"]
    if cfg.normalized_model_type is ModelType.MODEL1 and case["stride_kv_row"] != 584:
        raise ValueError("MODEL1 sparse FP8 decode requires stride_kv_row == 584 bytes")
    if case["kv_storage"].data_ptr() % 16 != 0:
        raise ValueError("KV cache base must be 16-byte aligned")
    tma_k_stride = 656 if cfg.normalized_model_type is ModelType.V32 else 576
    if case["stride_kv_block"] % tma_k_stride:
        raise ValueError("KV block stride must be divisible by MODEL_TYPE TMA_K_STRIDE")

    if cfg.extra_topk:
        if case["extra_kv_storage"].data_ptr() % 16 != 0:
            raise ValueError("extra KV cache base must be 16-byte aligned")
        if case["stride_extra_kv_block"] % tma_k_stride:
            raise ValueError("extra KV block stride must be divisible by TMA_K_STRIDE")


def _flat_storage_alias(
    tensor: torch.Tensor, *, element_offset: int = 0, extent: int | None = None
) -> torch.Tensor:
    """Expose a raw-pointer span while retaining the source tensor's storage."""

    if extent is None:
        extent = tensor.numel() - element_offset
    return tensor.as_strided(
        (extent,), (1,), storage_offset=tensor.storage_offset() + element_offset
    )


def _present_runtime_args(
    args: tuple[Any, ...], optional_indices: tuple[int, ...], presence: tuple[bool, ...]
) -> tuple[Any, ...]:
    absent_indices = {
        index
        for index, is_present in zip(optional_indices, presence, strict=True)
        if not is_present
    }
    return tuple(arg for index, arg in enumerate(args) if index not in absent_indices)


class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_tensor_map(ptr, dtype, dims, strides, box, *, swizzle, l2_promotion=2):
    desc = _AlignedTensorMap()
    rank = len(dims)
    assert len(strides) == rank - 1 and len(box) == rank
    tvm.get_global_func("runtime.cuTensorMapEncodeTiled")(
        desc.ptr,
        dtype,
        rank,
        ctypes.c_void_p(int(ptr)),
        *dims,
        *strides,
        *box,
        *((1,) * rank),
        0,
        swizzle,
        l2_promotion,
        0,
    )
    return desc


def _main_tensor_maps(case, start_head_idx, q_arg, kv_arg, extra_kv_arg, out_arg):
    cache = case.setdefault("_main_tensor_maps", {})
    if start_head_idx in cache:
        return cache[start_head_idx]
    cfg: SparseFlashMLADecodeHead64Config = case["config"]
    is_v32 = cfg.normalized_model_type is ModelType.V32
    d_qk = cfg.d_qk
    d_nope = 512 if is_v32 else 448
    tma_k_stride = 656 if is_v32 else 576
    rope_tile = 32 if is_v32 else 64
    kv_rope_start = (d_nope + (16 if is_v32 else 0)) // BF16_BYTES

    def kv_maps(tensor, num_rows):
        ptr = int(tensor.data_ptr())
        rope = _encode_tensor_map(
            ptr + kv_rope_start * BF16_BYTES,
            "bfloat16",
            (64, num_rows),
            (tma_k_stride,),
            (rope_tile, 1),
            swizzle=2 if is_v32 else 3,
        )
        nope = _encode_tensor_map(
            ptr, "int64", (d_nope // 8, num_rows), (tma_k_stride,), (d_nope // 8, 1), swizzle=0
        )
        return rope, nope

    kv_rope, kv_nope = kv_maps(kv_arg, case["shape"]["num_tma_rows"])
    if cfg.extra_topk:
        extra_rope, extra_nope = kv_maps(extra_kv_arg, case["shape"]["extra_num_tma_rows"])
    else:
        extra_rope, extra_nope = kv_rope, kv_nope
    q_ptr = int(q_arg.data_ptr())
    q_strided = _encode_tensor_map(
        q_ptr,
        "bfloat16",
        (d_qk, B_H, cfg.s_q, cfg.b),
        (
            case["stride_q_h_q"] * BF16_BYTES,
            case["stride_q_s_q"] * BF16_BYTES,
            case["stride_q_b"] * BF16_BYTES,
        ),
        (64, B_H, 1, 1),
        swizzle=3,
    )
    if is_v32:
        q_tail = _encode_tensor_map(
            q_ptr,
            "bfloat16",
            (32, B_H, d_qk // 32, cfg.s_q, cfg.b),
            (
                case["stride_q_h_q"] * BF16_BYTES,
                32 * BF16_BYTES,
                case["stride_q_s_q"] * BF16_BYTES,
                case["stride_q_b"] * BF16_BYTES,
            ),
            (32, B_H, 2, 1, 1),
            swizzle=2,
        )
    else:
        q_tail = q_strided
    out_map = _encode_tensor_map(
        int(out_arg.data_ptr()),
        "bfloat16",
        (D_V, B_H, cfg.s_q, cfg.b),
        (
            case["stride_o_h_q"] * BF16_BYTES,
            case["stride_o_s_q"] * BF16_BYTES,
            case["stride_o_b"] * BF16_BYTES,
        ),
        (64, B_H, 1, 1),
        swizzle=3,
    )
    maps = (kv_rope, kv_nope, extra_rope, extra_nope, q_strided, q_tail, out_map)
    cache[start_head_idx] = maps
    return maps


def _tirx_main_args(case: dict[str, Any], start_head_idx: int) -> tuple[Any, ...]:
    cfg: SparseFlashMLADecodeHead64Config = case["config"]
    tma_k_stride = 656 if cfg.normalized_model_type is ModelType.V32 else 576
    if start_head_idx % B_H or start_head_idx + B_H > cfg.h_q:
        raise ValueError(f"invalid head64 slice {start_head_idx} for h_q={cfg.h_q}")
    q_extent = cfg.b * case["stride_q_b"]
    indices_extent = (
        (cfg.b - 1) * case["stride_indices_b"]
        + (cfg.s_q - 1) * case["stride_indices_s_q"]
        + cfg.topk
    )
    lse_extent = (cfg.b - 1) * case["stride_lse_b"] + (cfg.s_q - 1) * case["stride_lse_s_q"] + B_H
    out_extent = cfg.b * case["stride_o_b"]
    lse_accum_extent = (
        (case["shape"]["split_rows"] - 1) * case["stride_lse_accum_split"]
        + (cfg.s_q - 1) * case["stride_lse_accum_s_q"]
        + B_H
    )
    o_accum_extent = (
        (case["shape"]["split_rows"] - 1) * case["stride_o_accum_split"]
        + (cfg.s_q - 1) * case["stride_o_accum_s_q"]
        + (B_H - 1) * case["stride_o_accum_h_q"]
        + D_V
    )
    extra_indices_extent = max(
        1,
        (cfg.b - 1) * case["stride_extra_indices_b"]
        + (cfg.s_q - 1) * case["stride_extra_indices_s_q"]
        + cfg.extra_topk,
    )
    q_arg = _flat_storage_alias(
        case["q"], element_offset=start_head_idx * case["stride_q_h_q"], extent=q_extent
    )
    kv_arg = case["kv_storage"].view(torch.bfloat16)
    indices_arg = _flat_storage_alias(case["indices"], extent=indices_extent)
    topk_length_arg = case["topk_length"]
    attn_sink_arg = case["attn_sink"][start_head_idx : start_head_idx + B_H]
    lse_arg = _flat_storage_alias(case["lse"], element_offset=start_head_idx, extent=lse_extent)
    out_arg = _flat_storage_alias(
        case["out"], element_offset=start_head_idx * case["stride_o_h_q"], extent=out_extent
    )
    lse_accum_arg = _flat_storage_alias(
        case["lse_accum"], element_offset=start_head_idx, extent=lse_accum_extent
    )
    o_accum_arg = _flat_storage_alias(
        case["o_accum"],
        element_offset=start_head_idx * case["stride_o_accum_h_q"],
        extent=o_accum_extent,
    )
    if case["extra_kv_storage"] is not None:
        extra_kv_arg = case["extra_kv_storage"].view(torch.bfloat16)
        extra_indices_arg = _flat_storage_alias(case["extra_indices"], extent=extra_indices_extent)
    else:
        extra_kv_arg = kv_arg
        extra_indices_arg = indices_arg
    extra_topk_length_arg = (
        case["extra_topk_length"] if case["extra_topk_length"] is not None else topk_length_arg
    )
    maps = _main_tensor_maps(case, start_head_idx, q_arg, kv_arg, extra_kv_arg, out_arg)
    return (
        q_arg,
        kv_arg,
        indices_arg,
        topk_length_arg,
        attn_sink_arg,
        lse_arg,
        out_arg,
        lse_accum_arg,
        o_accum_arg,
        case["tile_scheduler_metadata"].reshape(-1),
        case["num_splits"],
        extra_kv_arg,
        extra_indices_arg,
        extra_topk_length_arg,
        *(tensor_map.ptr for tensor_map in maps),
        case["sm_scale_div_log2"],
        case["stride_q_b"],
        case["stride_q_s_q"],
        case["stride_q_h_q"],
        case["stride_kv_block"],
        case["stride_kv_row"],
        case["stride_indices_b"],
        case["stride_indices_s_q"],
        case["stride_lse_b"],
        case["stride_lse_s_q"],
        case["stride_o_b"],
        case["stride_o_s_q"],
        case["stride_o_h_q"],
        case["stride_extra_kv_block"],
        case["stride_extra_kv_row"],
        case["stride_kv_block"] // tma_k_stride,
        case["stride_extra_kv_block"] // tma_k_stride,
        case["stride_extra_indices_b"],
        case["stride_extra_indices_s_q"],
        case["stride_lse_accum_split"],
        case["stride_lse_accum_s_q"],
        case["stride_o_accum_split"],
        case["stride_o_accum_s_q"],
        case["stride_o_accum_h_q"],
        cfg.b,
        cfg.s_q,
        cfg.topk,
        cfg.extra_topk,
        case["shape"]["num_blocks"],
        case["shape"]["extra_num_blocks"],
        cfg.page_block_size,
        case["shape"]["extra_page_block_size"],
        case["shape"]["num_sm_parts"],
    )


def _tirx_combine_args(case: dict[str, Any]) -> tuple[Any, ...]:
    cfg: SparseFlashMLADecodeHead64Config = case["config"]
    args = (
        case["lse"].reshape(-1),
        case["out"].reshape(-1),
        case["lse_accum"].reshape(-1),
        case["o_accum"].reshape(-1),
        case["num_splits"],
        case["attn_sink"],
        case["stride_lse_b"],
        case["stride_lse_s_q"],
        case["stride_o_b"],
        case["stride_o_s_q"],
        case["stride_o_h_q"],
        case["stride_lse_accum_split"],
        case["stride_lse_accum_s_q"],
        case["stride_o_accum_split"],
        case["stride_o_accum_s_q"],
        case["stride_o_accum_h_q"],
        cfg.b,
        cfg.s_q,
        cfg.h_q,
        cfg.d_v,
        case["shape"]["num_sm_parts"],
    )
    return args


@lru_cache(maxsize=128)
def _compile_main_kernel_cached(model_type: ModelType, presence: MainPresenceMask, use_pdl: bool):
    from tirx_kernels.runner import compile_kernel

    return compile_kernel(_specialized_main_kernel(model_type, presence, use_pdl))


@lru_cache(maxsize=20)
def _compile_combine_kernel_cached(max_splits: int, have_attn_sink: bool, use_pdl: bool):
    from tirx_kernels.runner import compile_kernel

    return compile_kernel(_specialized_combine_kernel(max_splits, have_attn_sink, use_pdl))


def _compile_decode_kernels(**kwargs: Any):
    from tirx_kernels.runner import hardware_num_sms

    cfg = _cfg(**kwargs)
    device = kwargs.get("device", "cuda")
    shape = _kernel_shape_params(cfg, device, prepared_num_sms=hardware_num_sms())
    presence = _main_presence_mask(cfg)
    use_pdl = cfg.b == 2
    return (
        _compile_main_kernel_cached(cfg.normalized_model_type, presence, use_pdl),
        _compile_combine_kernel_cached(shape["max_splits"], presence[1], use_pdl),
    )


def prepare_bench(**kwargs: Any):
    """Compile both sparse-decode executables without touching CUDA."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executables": _compile_decode_kernels(**kwargs)}
    return prepared_gpu_benchmark(run_gpu, state)


def _launch_tirx(case: dict[str, Any], executables: tuple[Any, Any]) -> None:
    main_ex, combine_ex = executables
    _validate_tirx_launch_case(case)
    cfg: SparseFlashMLADecodeHead64Config = case["config"]
    for start_head_idx in range(0, cfg.h_q, B_H):
        main_ex(*_tirx_main_args(case, start_head_idx))
    combine_ex(*_tirx_combine_args(case))


def run_test(**kwargs: Any) -> None:
    cfg = _cfg(**kwargs)
    # Upstream clears the allocator before every generated case; keep the
    # 15-case performance sweep from retaining cached pressure-shape blocks.
    torch.cuda.empty_cache()
    case = prepare_data(**kwargs)
    executables = _compile_decode_kernels(**kwargs)

    from tirx_kernels.flashmla.utils._flashmla_bench import (
        _import_flash_mla,
        run_flashmla_sparse_decode,
    )

    flash_mla = _import_flash_mla()
    sched_meta, _ = flash_mla.get_mla_metadata()
    ref_out, ref_lse = run_flashmla_sparse_decode(case, sched_meta)
    torch.cuda.synchronize()

    # Validate the host replica against the actual CUDA scheduler.  Inactive
    # rows only have a defined begin_req_idx; the remaining CUDA fields may
    # contain shared-memory tail values and are never consumed.
    ref_metadata = sched_meta.tile_scheduler_metadata
    ref_num_splits = sched_meta.num_splits
    if ref_metadata is None or ref_num_splits is None:
        raise AssertionError("FlashMLA did not initialize decode scheduler metadata")
    ours_metadata = case["tile_scheduler_metadata"]
    torch.testing.assert_close(ours_metadata[:, 0], ref_metadata[:, 0], rtol=0, atol=0)
    active = ours_metadata[:, 0] < cfg.b
    torch.testing.assert_close(ours_metadata[active, :7], ref_metadata[active, :7], rtol=0, atol=0)
    torch.testing.assert_close(case["num_splits"], ref_num_splits, rtol=0, atol=0)

    case["out"].fill_(float("nan"))
    case["lse"].fill_(float("nan"))
    _launch_tirx(case, executables)
    torch.cuda.synchronize()
    torch.testing.assert_close(case["out"], ref_out, rtol=2.01 / 128, atol=1.0e-3)
    torch.testing.assert_close(case["lse"], ref_lse.transpose(1, 2), rtol=8.01 / 65536, atol=1.0e-6)
    cfg.validate()


def run_gpu(
    prepared,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    kwargs = {**prepared["config"], **kwargs}
    rounds = kwargs.pop("rounds", 1)
    cooldown_s = kwargs.pop("cooldown_s", 1.0)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA decode benchmark")

    from tirx_kernels.runner import bench

    executables = prepared["executables"]
    # Allocate once outside both timed regions; both paths launch the exact
    # split-KV main kernel followed by their separate combine kernel.
    case = prepare_data(**kwargs)

    def tirx_decode():
        _launch_tirx(case, executables)

    from tirx_kernels.flashmla.utils._flashmla_bench import flashmla_decode_reference_builder

    return bench(
        {"tirx": tirx_decode},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashmla": lambda: flashmla_decode_reference_builder(case)},
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    rounds = kwargs.pop("rounds", 1)
    cooldown_s = kwargs.pop("cooldown_s", 1.0)
    return prepare_bench(**kwargs).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "CONFIGS",
    "KERNEL_META",
    "ModelType",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
