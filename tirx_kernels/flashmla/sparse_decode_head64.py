# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

import math
import random
from dataclasses import dataclass, fields
from enum import Enum
from functools import lru_cache
from typing import Any
from unittest import SkipTest

import torch

from tirx_kernels.flashmla.utils._ir_builder import (
    MBarrier,
    PipelineState,
    SmemDescriptor,
    TCGen05Bar,
    TMABar,
)
from tirx_kernels.flashmla.utils._ir_builder import builder_alloc_scalar as _builder_alloc_scalar
from tirx_kernels.flashmla.utils._ir_builder import builder_assign as _builder_assign
from tirx_kernels.flashmla.utils._ir_builder import builder_bind as _builder_bind
from tirx_kernels.flashmla.utils._ir_builder import builder_emit as _builder_emit
from tirx_kernels.flashmla.utils._ir_builder import builder_enter as _builder_enter
from tirx_kernels.flashmla.utils._ir_builder import builder_scalar as _builder_scalar
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode
from tvm.ir import PointerType, PrimType
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T
from tvm.tirx.layout import S, TileLayout, laneid, wid_in_wg

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
_Q_TMA_CACHE_HINT = T.uint64(0x12F0000000000000)
_KV_TMA_CACHE_HINT = T.uint64(0x14F0000000000000)


def _tmem_load(dst, tmem_col, width):
    chain = _TMEM_LD_32 if width == 32 else _TMEM_LD_64
    return T.ptx[chain](*[dst[i] for i in range(width)], tmem_col)


def _tmem_store(src, tmem_col, width=64):
    assert width == 64
    return T.ptx[_TMEM_ST_64](tmem_col, *[src[i] for i in range(width)])


def _cast_f32x2_bf16x2(dst, src, offset):
    dst_words = dst.view("uint32")
    return T.ptx.cvt.rn.bf16x2.f32(dst_words[offset // 2], src[offset + 1], src[offset])


def _replace_smem_desc_addr(desc, smem_ptr):
    start_addr = T.cast(
        T.bitwise_and(
            T.shift_right(T.cuda.cvta_generic_to_shared(smem_ptr), T.uint32(4)), T.uint32(0x3FFF)
        ),
        "uint64",
    )
    return T.bitwise_or(T.bitwise_and(desc, T.bitwise_not(T.uint64(0x3FFF))), start_addr)


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
    {
        "label": "model1_b2_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64",
        "model_type": "MODEL1",
        "b": 2,
        "s_q": 2,
        "s_kv": 16384,
        "topk": 128,
        "page_block_size": 256,
        "have_attn_sink": True,
        "extra_s_kv": 16384,
        "extra_topk": 512,
        "extra_page_block_size": 64,
    },
    {
        "label": "model1_b64_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64",
        "model_type": "MODEL1",
        "b": 64,
        "s_q": 2,
        "s_kv": 16384,
        "topk": 128,
        "page_block_size": 256,
        "have_attn_sink": True,
        "extra_s_kv": 16384,
        "extra_topk": 512,
        "extra_page_block_size": 64,
    },
    {
        "label": "model1_b74_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64",
        "model_type": "MODEL1",
        "b": 74,
        "s_q": 2,
        "s_kv": 16384,
        "topk": 128,
        "page_block_size": 256,
        "have_attn_sink": True,
        "extra_s_kv": 16384,
        "extra_topk": 512,
        "extra_page_block_size": 64,
    },
    {
        "label": "model1_b128_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64",
        "model_type": "MODEL1",
        "b": 128,
        "s_q": 2,
        "s_kv": 16384,
        "topk": 128,
        "page_block_size": 256,
        "have_attn_sink": True,
        "extra_s_kv": 16384,
        "extra_topk": 512,
        "extra_page_block_size": 64,
    },
    {
        "label": "model1_b148_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64",
        "model_type": "MODEL1",
        "b": 148,
        "s_q": 2,
        "s_kv": 16384,
        "topk": 128,
        "page_block_size": 256,
        "have_attn_sink": True,
        "extra_s_kv": 16384,
        "extra_topk": 512,
        "extra_page_block_size": 64,
    },
    {
        "label": "model1_b256_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64",
        "model_type": "MODEL1",
        "b": 256,
        "s_q": 2,
        "s_kv": 16384,
        "topk": 128,
        "page_block_size": 256,
        "have_attn_sink": True,
        "extra_s_kv": 16384,
        "extra_topk": 512,
        "extra_page_block_size": 64,
    },
    {
        "label": "model1_b2_sq2_sk16384_topk128_p256_xsk16384_xtopk1024_xp2_xtopklen",
        "model_type": "MODEL1",
        "b": 2,
        "s_q": 2,
        "s_kv": 16384,
        "topk": 128,
        "page_block_size": 256,
        "have_attn_sink": True,
        "extra_s_kv": 16384,
        "extra_topk": 1024,
        "extra_page_block_size": 2,
        "have_extra_topk_length": True,
    },
    {
        "label": "model1_b64_sq2_sk16384_topk128_p256_xsk16384_xtopk1024_xp2_xtopklen",
        "model_type": "MODEL1",
        "b": 64,
        "s_q": 2,
        "s_kv": 16384,
        "topk": 128,
        "page_block_size": 256,
        "have_attn_sink": True,
        "extra_s_kv": 16384,
        "extra_topk": 1024,
        "extra_page_block_size": 2,
        "have_extra_topk_length": True,
    },
    {
        "label": "model1_b74_sq2_sk16384_topk128_p256_xsk16384_xtopk1024_xp2_xtopklen",
        "model_type": "MODEL1",
        "b": 74,
        "s_q": 2,
        "s_kv": 16384,
        "topk": 128,
        "page_block_size": 256,
        "have_attn_sink": True,
        "extra_s_kv": 16384,
        "extra_topk": 1024,
        "extra_page_block_size": 2,
        "have_extra_topk_length": True,
    },
    {
        "label": "model1_b128_sq2_sk16384_topk128_p256_xsk16384_xtopk1024_xp2_xtopklen",
        "model_type": "MODEL1",
        "b": 128,
        "s_q": 2,
        "s_kv": 16384,
        "topk": 128,
        "page_block_size": 256,
        "have_attn_sink": True,
        "extra_s_kv": 16384,
        "extra_topk": 1024,
        "extra_page_block_size": 2,
        "have_extra_topk_length": True,
    },
    {
        "label": "model1_b148_sq2_sk16384_topk128_p256_xsk16384_xtopk1024_xp2_xtopklen",
        "model_type": "MODEL1",
        "b": 148,
        "s_q": 2,
        "s_kv": 16384,
        "topk": 128,
        "page_block_size": 256,
        "have_attn_sink": True,
        "extra_s_kv": 16384,
        "extra_topk": 1024,
        "extra_page_block_size": 2,
        "have_extra_topk_length": True,
    },
    {
        "label": "model1_b256_sq2_sk16384_topk128_p256_xsk16384_xtopk1024_xp2_xtopklen",
        "model_type": "MODEL1",
        "b": 256,
        "s_q": 2,
        "s_kv": 16384,
        "topk": 128,
        "page_block_size": 256,
        "have_attn_sink": True,
        "extra_s_kv": 16384,
        "extra_topk": 1024,
        "extra_page_block_size": 2,
        "have_extra_topk_length": True,
    },
    {
        "label": "model1_b148_sq2_sk32768_topk16384_p64",
        "model_type": "MODEL1",
        "b": 148,
        "s_q": 2,
        "s_kv": 32768,
        "topk": 16384,
        "page_block_size": 64,
        "have_attn_sink": True,
    },
    {
        "label": "v32_b148_sq2_sk32768_topk16384_p64",
        "model_type": "V32",
        "b": 148,
        "s_q": 2,
        "s_kv": 32768,
        "topk": 16384,
        "page_block_size": 64,
        "have_attn_sink": True,
    },
]


KERNEL_META = {
    "name": "sparse_flashmla_decode_head64",
    "category": "flashmla",
    "compute_capability": 10,
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


def _build_kernel(
    *,
    model_type: T.constexpr,
    use_pdl: T.constexpr,
    _have_topk_length_h=True,
    _have_attn_sink_h=True,
    _have_extra_kv_h=True,
    _have_extra_indices_h=True,
    _have_extra_topk_length_h=True,
):
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("_kernel")
            q_h = T.arg("q_h", T.handle())
            kv_h = T.arg("kv_h", T.handle())
            indices_h = T.arg("indices_h", T.handle())
            if _have_topk_length_h:
                topk_length_h = T.arg("topk_length_h", T.handle())
            else:
                topk_length_h = None
            if _have_attn_sink_h:
                attn_sink_h = T.arg("attn_sink_h", T.handle())
            else:
                attn_sink_h = None
            lse_h = T.arg("lse_h", T.handle())
            out_h = T.arg("out_h", T.handle())
            lse_accum_h = T.arg("lse_accum_h", T.handle())
            o_accum_h = T.arg("o_accum_h", T.handle())
            tile_scheduler_metadata_h = T.arg("tile_scheduler_metadata_h", T.handle())
            num_splits_h = T.arg("num_splits_h", T.handle())
            if _have_extra_kv_h:
                extra_kv_h = T.arg("extra_kv_h", T.handle())
            else:
                extra_kv_h = None
            if _have_extra_indices_h:
                extra_indices_h = T.arg("extra_indices_h", T.handle())
            else:
                extra_indices_h = None
            if _have_extra_topk_length_h:
                extra_topk_length_h = T.arg("extra_topk_length_h", T.handle())
            else:
                extra_topk_length_h = None
            sm_scale_div_log2 = T.arg("sm_scale_div_log2", T.float32())
            stride_q_b = T.arg("stride_q_b", T.int32())
            stride_q_s_q = T.arg("stride_q_s_q", T.int32())
            stride_q_h_q = T.arg("stride_q_h_q", T.int32())
            stride_kv_block = T.arg("stride_kv_block", T.int32())
            stride_kv_row = T.arg("stride_kv_row", T.int32())
            stride_indices_b = T.arg("stride_indices_b", T.int32())
            stride_indices_s_q = T.arg("stride_indices_s_q", T.int32())
            stride_lse_b = T.arg("stride_lse_b", T.int32())
            stride_lse_s_q = T.arg("stride_lse_s_q", T.int32())
            stride_o_b = T.arg("stride_o_b", T.int32())
            stride_o_s_q = T.arg("stride_o_s_q", T.int32())
            stride_o_h_q = T.arg("stride_o_h_q", T.int32())
            stride_extra_kv_block = T.arg("stride_extra_kv_block", T.int32())
            stride_extra_kv_row = T.arg("stride_extra_kv_row", T.int32())
            stride_extra_indices_b = T.arg("stride_extra_indices_b", T.int32())
            stride_extra_indices_s_q = T.arg("stride_extra_indices_s_q", T.int32())
            stride_lse_accum_split = T.arg("stride_lse_accum_split", T.int32())
            stride_lse_accum_s_q = T.arg("stride_lse_accum_s_q", T.int32())
            stride_o_accum_split = T.arg("stride_o_accum_split", T.int32())
            stride_o_accum_s_q = T.arg("stride_o_accum_s_q", T.int32())
            stride_o_accum_h_q = T.arg("stride_o_accum_h_q", T.int32())
            b = T.arg("b", T.int32())
            s_q = T.arg("s_q", T.int32())
            topk = T.arg("topk", T.int32())
            extra_topk = T.arg("extra_topk", T.int32())
            num_blocks = T.arg("num_blocks", T.int32())
            extra_num_blocks = T.arg("extra_num_blocks", T.int32())
            page_block_size = T.arg("page_block_size", T.int32())
            extra_page_block_size = T.arg("extra_page_block_size", T.int32())
            num_sm_parts = T.arg("num_sm_parts", T.int32())
            is_v32 = model_type is ModelType.V32
            d_qk = 576 if is_v32 else 512
            d_nope = 512 if is_v32 else 448
            num_scales = 4 if is_v32 else 8
            tma_k_stride = 656 if is_v32 else 576
            q_tail_start = 256 if is_v32 else 224
            rope_tile = 32 if is_v32 else 64
            rows_per_group = B_TOPK // (128 // 8)
            cols_per_group = d_nope // (8 * 8)
            q = _builder_assign(
                "q",
                T.match_buffer(
                    q_h,
                    (b, stride_q_b // stride_q_s_q, stride_q_s_q // stride_q_h_q, stride_q_h_q),
                    "bfloat16",
                    scope="global",
                ),
            )
            kv = _builder_assign(
                "kv",
                T.match_buffer(
                    kv_h,
                    (num_blocks * (stride_kv_block // tma_k_stride), tma_k_stride // BF16_BYTES),
                    "bfloat16",
                    scope="global",
                ),
            )
            indices = _builder_assign(
                "indices",
                T.match_buffer(
                    indices_h,
                    ((b - 1) * stride_indices_b + (s_q - 1) * stride_indices_s_q + topk,),
                    "int32",
                    scope="global",
                ),
            )
            if topk_length_h is not None:
                topk_length = _builder_assign(
                    "topk_length", T.match_buffer(topk_length_h, (b,), "int32", scope="global")
                )
            if attn_sink_h is not None:
                attn_sink = _builder_assign(
                    "attn_sink", T.match_buffer(attn_sink_h, (B_H,), "float32", scope="global")
                )
            lse = _builder_assign(
                "lse",
                T.match_buffer(
                    lse_h,
                    ((b - 1) * stride_lse_b + (s_q - 1) * stride_lse_s_q + B_H,),
                    "float32",
                    scope="global",
                ),
            )
            out = _builder_assign(
                "out",
                T.match_buffer(
                    out_h,
                    (b, stride_o_b // stride_o_s_q, stride_o_s_q // stride_o_h_q, stride_o_h_q),
                    "bfloat16",
                    scope="global",
                ),
            )
            lse_accum = _builder_assign(
                "lse_accum",
                T.match_buffer(
                    lse_accum_h,
                    (
                        (b + num_sm_parts - 1) * stride_lse_accum_split
                        + (s_q - 1) * stride_lse_accum_s_q
                        + B_H,
                    ),
                    "float32",
                    scope="global",
                ),
            )
            o_accum = _builder_assign(
                "o_accum",
                T.match_buffer(
                    o_accum_h,
                    (
                        (b + num_sm_parts - 1) * stride_o_accum_split
                        + (s_q - 1) * stride_o_accum_s_q
                        + (B_H - 1) * stride_o_accum_h_q
                        + D_V,
                    ),
                    "float32",
                    scope="global",
                ),
            )
            tile_scheduler_metadata = _builder_assign(
                "tile_scheduler_metadata",
                T.match_buffer(
                    tile_scheduler_metadata_h, (num_sm_parts, 8), "int32", scope="global"
                ),
            )
            num_splits = _builder_assign(
                "num_splits", T.match_buffer(num_splits_h, (b + 1,), "int32", scope="global")
            )
            if extra_kv_h is not None:
                extra_kv = _builder_assign(
                    "extra_kv",
                    T.match_buffer(
                        extra_kv_h,
                        (
                            extra_num_blocks * (stride_extra_kv_block // tma_k_stride),
                            tma_k_stride // BF16_BYTES,
                        ),
                        "bfloat16",
                        scope="global",
                    ),
                )
            if extra_indices_h is not None:
                extra_indices = _builder_assign(
                    "extra_indices",
                    T.match_buffer(
                        extra_indices_h,
                        (
                            (b - 1) * stride_extra_indices_b
                            + (s_q - 1) * stride_extra_indices_s_q
                            + extra_topk,
                        ),
                        "int32",
                        scope="global",
                    ),
                )
            if extra_topk_length_h is not None:
                extra_topk_length = _builder_assign(
                    "extra_topk_length",
                    T.match_buffer(extra_topk_length_h, (b,), "int32", scope="global"),
                )
            kv_nope_tma = _builder_assign("kv_nope_tma", kv.view("int64").sub[:, : d_nope // 8])
            kv_rope_start = (d_nope + (16 if is_v32 else 0)) // BF16_BYTES
            kv_rope_tma = _builder_assign(
                "kv_rope_tma", kv.sub[:, kv_rope_start : kv_rope_start + 64]
            )
            if extra_kv_h is not None:
                extra_kv_nope_tma = _builder_assign(
                    "extra_kv_nope_tma", extra_kv.view("int64").sub[:, : d_nope // 8]
                )
                extra_kv_rope_tma = _builder_assign(
                    "extra_kv_rope_tma", extra_kv.sub[:, kv_rope_start : kv_rope_start + 64]
                )
            kv_rope_tensormap = _builder_bind(
                "kv_rope_tensormap",
                T.tvm_stack_alloca("tensormap", 1),
                type_annotation=T.TensorMap(),
            )
            _builder_emit(
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    kv_rope_tensormap,
                    "bfloat16",
                    2,
                    T.handle_add_byte_offset(kv.data, kv_rope_start * BF16_BYTES),
                    64,
                    num_blocks * (stride_kv_block // tma_k_stride),
                    tma_k_stride,
                    rope_tile,
                    1,
                    1,
                    1,
                    0,
                    2 if is_v32 else 3,
                    2,
                    0,
                )
            )
            kv_nope_tensormap = _builder_bind(
                "kv_nope_tensormap",
                T.tvm_stack_alloca("tensormap", 1),
                type_annotation=T.TensorMap(),
            )
            _builder_emit(
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    kv_nope_tensormap,
                    "int64",
                    2,
                    kv.data,
                    d_nope // 8,
                    num_blocks * (stride_kv_block // tma_k_stride),
                    tma_k_stride,
                    d_nope // 8,
                    1,
                    1,
                    1,
                    0,
                    0,
                    2,
                    0,
                )
            )
            if extra_kv_h is not None:
                extra_kv_rope_tensormap = _builder_bind(
                    "extra_kv_rope_tensormap",
                    T.tvm_stack_alloca("tensormap", 1),
                    type_annotation=T.TensorMap(),
                )
                _builder_emit(
                    T.call_packed(
                        "runtime.cuTensorMapEncodeTiled",
                        extra_kv_rope_tensormap,
                        "bfloat16",
                        2,
                        T.handle_add_byte_offset(extra_kv.data, kv_rope_start * BF16_BYTES),
                        64,
                        extra_num_blocks * (stride_extra_kv_block // tma_k_stride),
                        tma_k_stride,
                        rope_tile,
                        1,
                        1,
                        1,
                        0,
                        2 if is_v32 else 3,
                        2,
                        0,
                    )
                )
                extra_kv_nope_tensormap = _builder_bind(
                    "extra_kv_nope_tensormap",
                    T.tvm_stack_alloca("tensormap", 1),
                    type_annotation=T.TensorMap(),
                )
                _builder_emit(
                    T.call_packed(
                        "runtime.cuTensorMapEncodeTiled",
                        extra_kv_nope_tensormap,
                        "int64",
                        2,
                        extra_kv.data,
                        d_nope // 8,
                        extra_num_blocks * (stride_extra_kv_block // tma_k_stride),
                        tma_k_stride,
                        d_nope // 8,
                        1,
                        1,
                        1,
                        0,
                        0,
                        2,
                        0,
                    )
                )
            q_strided_tensormap = _builder_bind(
                "q_strided_tensormap",
                T.tvm_stack_alloca("tensormap", 1),
                type_annotation=T.TensorMap(),
            )
            _builder_emit(
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    q_strided_tensormap,
                    "bfloat16",
                    4,
                    q.data,
                    d_qk,
                    B_H,
                    s_q,
                    b,
                    stride_q_h_q * BF16_BYTES,
                    stride_q_s_q * BF16_BYTES,
                    stride_q_b * BF16_BYTES,
                    64,
                    B_H,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    3,
                    2,
                    0,
                )
            )
            if is_v32:
                q_tail_tensormap = _builder_bind(
                    "q_tail_tensormap",
                    T.tvm_stack_alloca("tensormap", 1),
                    type_annotation=T.TensorMap(),
                )
                _builder_emit(
                    T.call_packed(
                        "runtime.cuTensorMapEncodeTiled",
                        q_tail_tensormap,
                        "bfloat16",
                        5,
                        q.data,
                        32,
                        B_H,
                        d_qk // 32,
                        s_q,
                        b,
                        stride_q_h_q * BF16_BYTES,
                        32 * BF16_BYTES,
                        stride_q_s_q * BF16_BYTES,
                        stride_q_b * BF16_BYTES,
                        32,
                        B_H,
                        2,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        2,
                        2,
                        0,
                    )
                )
            out_tensormap = _builder_bind(
                "out_tensormap", T.tvm_stack_alloca("tensormap", 1), type_annotation=T.TensorMap()
            )
            _builder_emit(
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    out_tensormap,
                    "bfloat16",
                    4,
                    out.data,
                    D_V,
                    B_H,
                    s_q,
                    b,
                    stride_o_h_q * BF16_BYTES,
                    stride_o_s_q * BF16_BYTES,
                    stride_o_b * BF16_BYTES,
                    64,
                    B_H,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    3,
                    2,
                    0,
                )
            )
            _builder_emit(T.device_entry())
            _builder_enter(
                T.attr(
                    {
                        "tirx.launch_bounds_min_blocks_per_sm": 1,
                        "tirx.launch_bounds_max_blocks_per_cluster": 1,
                    }
                )
            )
            source_smem_size = 232192 if is_v32 else 218848
            _builder_values_330 = T.cta_id([s_q, num_sm_parts, 1])
            s_q_idx, partition_idx, _ = _builder_values_330
            IRBuilder.name("_", _)
            IRBuilder.name("partition_idx", partition_idx)
            IRBuilder.name("s_q_idx", s_q_idx)
            thread_idx = _builder_assign("thread_idx", T.thread_id([NUM_THREADS]))
            warpgroup_idx = _builder_assign("warpgroup_idx", T.warpgroup_id([3]))
            warp_idx_in_wg = _builder_assign("warp_idx_in_wg", T.warp_id_in_wg([4]))
            lane_idx = _builder_assign("lane_idx", T.lane_id([32]))
            idx_in_warpgroup = _builder_assign("idx_in_warpgroup", T.thread_id_in_wg([128]))
            warp_idx = _builder_bind("warp_idx", warpgroup_idx * 4 + warp_idx_in_wg)
            with T.If(warp_idx == 0):
                with T.Then():
                    with T.If(T.cuda.elect_sync() != T.uint32(0)):
                        with T.Then():
                            with T.unroll(8) as _prefetch_i:
                                _builder_emit(
                                    T.evaluate(
                                        T.ptx.prefetch.tensormap(T.address_of(out_tensormap))
                                    )
                                )
                            _builder_emit(
                                T.evaluate(
                                    T.ptx.prefetch.tensormap(T.address_of(q_strided_tensormap))
                                )
                            )
                            if is_v32:
                                _builder_emit(
                                    T.evaluate(
                                        T.ptx.prefetch.tensormap(T.address_of(q_tail_tensormap))
                                    )
                                )
                            _builder_emit(
                                T.evaluate(
                                    T.ptx.prefetch.tensormap(T.address_of(kv_nope_tensormap))
                                )
                            )
                            _builder_emit(
                                T.evaluate(
                                    T.ptx.prefetch.tensormap(T.address_of(kv_rope_tensormap))
                                )
                            )
            pool = _builder_assign("pool", T.SMEMPool())
            u_base = pool.offset
            if is_v32:
                k_union = _builder_assign(
                    "k_union", pool.alloc_tcgen05_mma_AB((NUM_BUFS, B_TOPK, D_V + 64), "bfloat16")
                )
                k_union_end = pool.offset
                k_full = _builder_assign("k_full", k_union.sub[:, :, :D_V])
                _builder_emit(pool.move_base_to(u_base))
                k_rope = _builder_assign(
                    "k_rope",
                    pool.alloc_tcgen05_mma_AB(
                        (NUM_BUFS, B_TOPK, D_V + 64),
                        "bfloat16",
                        swizzle_mode=SwizzleMode.SWIZZLE_64B_ATOM,
                    ).sub[:, :, D_V : D_V + 64],
                )
                _builder_emit(pool.move_base_to(k_union_end))
            else:
                k_full = _builder_assign(
                    "k_full", pool.alloc_tcgen05_mma_AB((NUM_BUFS, B_TOPK, D_V), "bfloat16")
                )
                k_union_end = pool.offset
                _builder_emit(pool.move_base_to(u_base))
                k_rope = _builder_assign(
                    "k_rope",
                    pool.alloc_tcgen05_mma_AB(
                        (NUM_BUFS, B_TOPK, 64),
                        "bfloat16",
                        swizzle_mode=SwizzleMode.SWIZZLE_128B_ATOM,
                    ),
                )
                _builder_emit(pool.move_base_to(k_union_end))
            k_rope_tma = k_rope if is_v32 else k_full.sub[:, :, d_nope : d_nope + 64]
            raw_nope = _builder_assign(
                "raw_nope", pool.alloc((NUM_BUFS, B_TOPK, d_nope // 8), "uint64", align=1024)
            )
            kv_union_end = pool.offset
            _builder_emit(pool.move_base_to(u_base))
            q_sw128 = _builder_assign("q_sw128", pool.alloc_tcgen05_mma_AB((B_H, 512), "bfloat16"))
            q_sw128_end = pool.offset
            if is_v32:
                _builder_emit(pool.move_base_to(q_sw128_end))
                q_sw64 = _builder_assign(
                    "q_sw64",
                    pool.alloc_tcgen05_mma_AB(
                        (B_H, 64), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_64B_ATOM
                    ),
                )
            o_union_base = pool.offset
            o_smem = _builder_assign("o_smem", pool.alloc_tcgen05_mma_AB((B_H, D_V), "bfloat16"))
            o_bf16_end = pool.offset
            _builder_emit(pool.move_base_to(o_union_base))
            o_accum_storage = _builder_assign(
                "o_accum_storage", pool.alloc(((B_H - 1) * (D_V + 8) + D_V,), "float32", align=1024)
            )
            o_accum_smem = _builder_assign(
                "o_accum_smem",
                o_accum_storage.view(B_H, D_V, layout=TileLayout(S[(B_H, D_V) : (D_V + 8, 1)])),
            )
            qo_union_end = pool.offset
            _builder_emit(pool.move_base_to(max(kv_union_end, qo_union_end, o_bf16_end)))
            sp_union_base = pool.offset
            p_exchange = _builder_assign(
                "p_exchange", pool.alloc((4, 32 * (B_TOPK // 2)), "float32", align=16)
            )
            sp_union_end = pool.offset
            _builder_emit(pool.move_base_to(sp_union_base))
            s_smem_gemm = _builder_assign(
                "s_smem_gemm",
                pool.alloc_tcgen05_mma_AB(
                    (B_H, B_TOPK), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_NONE
                ),
            )
            _builder_emit(pool.move_base_to(sp_union_end))
            pv_b_lo_desc = _builder_assign("pv_b_lo_desc", SmemDescriptor())
            _builder_emit(pv_b_lo_desc.init(k_full.ptr_to([0, 0, 0]), ldo=512, sdo=64, swizzle=3))
            pv_b_hi_desc = _builder_assign("pv_b_hi_desc", SmemDescriptor())
            _builder_emit(pv_b_hi_desc.init(k_full.ptr_to([0, 0, 0]), ldo=512, sdo=64, swizzle=3))
            pv_a_lo_desc = _builder_assign("pv_a_lo_desc", SmemDescriptor())
            _builder_emit(pv_a_lo_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0))
            pv_a_hi_desc = _builder_assign("pv_a_hi_desc", SmemDescriptor())
            _builder_emit(pv_a_hi_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0))
            q_main_cp_desc = _builder_alloc_scalar("q_main_cp_desc", "uint64")
            _builder_emit(
                T.cuda.tcgen05.encode_matrix_descriptor(
                    T.address_of(q_main_cp_desc),
                    T.reinterpret(T.handle().ty, T.uint64(0)),
                    1,
                    64,
                    3,
                )
            )
            if is_v32:
                q_tail_cp_desc = _builder_alloc_scalar("q_tail_cp_desc", "uint64")
                _builder_emit(
                    T.cuda.tcgen05.encode_matrix_descriptor(
                        T.address_of(q_tail_cp_desc),
                        T.reinterpret(T.handle().ty, T.uint64(0)),
                        1,
                        32,
                        2,
                    )
                )
            rowwise_buf = _builder_assign("rowwise_buf", pool.alloc((128,), "float32", align=16))
            is_token_valid = _builder_assign(
                "is_token_valid", pool.alloc((NUM_INDEX_BUFS, B_TOPK // 8), "int8", align=16)
            )
            tma_coord = _builder_assign(
                "tma_coord", pool.alloc((NUM_INDEX_BUFS, B_TOPK), "int32", align=16)
            )
            scales_e8m0 = _builder_assign(
                "scales_e8m0", pool.alloc((NUM_INDEX_BUFS, B_TOPK * num_scales), "uint8", align=16)
            )
            tmem_start_addr = _builder_assign(
                "tmem_start_addr", pool.alloc((4,), "uint32", align=16)
            )
            bar_last_store_done = _builder_assign("bar_last_store_done", MBarrier(pool, 1))
            bar_q_tma = _builder_assign("bar_q_tma", TMABar(pool, 1))
            bar_q_utccp = _builder_assign("bar_q_utccp", TCGen05Bar(pool, 1))
            bar_rope_ready = _builder_assign("bar_rope_ready", TMABar(pool, NUM_BUFS))
            bar_nope_ready = _builder_assign("bar_nope_ready", MBarrier(pool, NUM_BUFS))
            bar_raw_ready = _builder_assign("bar_raw_ready", TMABar(pool, NUM_BUFS))
            bar_raw_free = _builder_assign("bar_raw_free", MBarrier(pool, NUM_BUFS))
            bar_valid_ready = _builder_assign("bar_valid_ready", MBarrier(pool, NUM_INDEX_BUFS))
            bar_valid_free = _builder_assign("bar_valid_free", MBarrier(pool, NUM_INDEX_BUFS))
            bar_qk_done = _builder_assign("bar_qk_done", TCGen05Bar(pool, NUM_BUFS))
            bar_so_ready = _builder_assign("bar_so_ready", MBarrier(pool, NUM_BUFS))
            bar_sv_done = _builder_assign("bar_sv_done", TCGen05Bar(pool, NUM_BUFS))
            _builder_emit(pool.commit(size=source_smem_size))
            tmem_pool = _builder_assign(
                "tmem_pool",
                T.TMEMPool(
                    pool,
                    total_cols=512,
                    cta_group=1,
                    tmem_addr=tmem_start_addr,
                    sync_after_alloc=False,
                ),
            )
            o_tmem = _builder_assign(
                "o_tmem",
                tmem_pool.alloc_tcgen05_mma_D(
                    (B_H, D_V), "float32", M=64, cta_group=1, ws=True, group=(2, 2, 128)
                ),
            )
            _builder_emit(tmem_pool.move_base_to(256))
            q_tmem = _builder_assign(
                "q_tmem",
                tmem_pool.alloc_tcgen05_mma_A(
                    (2, B_H, d_qk // 2), "bfloat16", M=64, cta_group=1, ws=True
                ),
            )
            _builder_emit(tmem_pool.move_base_to(400))
            p_tmem = _builder_assign(
                "p_tmem",
                tmem_pool.alloc_tcgen05_mma_D(
                    (2, B_H, B_TOPK), "float32", M=64, cta_group=1, ws=True
                ),
            )

            def load_scheduler_meta(dst):
                _builder_emit(
                    T.ptx["ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v4.u64"](
                        dst[0],
                        dst[1],
                        dst[2],
                        dst[3],
                        tile_scheduler_metadata.view("uint64").ptr_to([partition_idx, 0]),
                    )
                )

            def dequant_st128(smem_addr, raw, scale_bits):
                scale = _builder_bind("scale", T.reinterpret("bfloat16", scale_bits))
                packed = _builder_assign("packed", T.alloc_local((4,), "uint32"))
                with T.unroll(4) as pair_i:
                    raw_pair = _builder_bind(
                        "raw_pair",
                        T.cast(T.shift_right(raw, T.cast(pair_i * 16, "uint64")), "uint16"),
                    )
                    rounded_bits = _builder_assign("rounded_bits", T.local_scalar("uint32"))
                    _builder_emit(T.ptx.cvt.rn.bf16x2.e4m3x2(rounded_bits, raw_pair))
                    rounded = _builder_bind("rounded", T.reinterpret("bfloat16x2", rounded_bits))
                    scaled_lo = _builder_bind("scaled_lo", T.Shuffle([rounded], [0]) * scale)
                    scaled_hi = _builder_bind("scaled_hi", T.Shuffle([rounded], [1]) * scale)
                    T.buffer_store(
                        packed,
                        T.reinterpret("uint32", T.Shuffle([scaled_lo, scaled_hi], [0, 1])),
                        [pair_i],
                    )
                _builder_emit(T.ptx.st.weak.shared__cta.b128(smem_addr, packed.view("uint128")[0]))

            with T.If(warp_idx == 0):
                with T.Then():
                    with T.If(T.cuda.elect_sync() != T.uint32(0)):
                        with T.Then():
                            _builder_emit(
                                T.ptx.mbarrier.init.shared.b64(
                                    bar_last_store_done.ptr_to([0]), T.uint32(128)
                                )
                            )
                            _builder_emit(
                                T.ptx.mbarrier.init.shared.b64(bar_q_tma.ptr_to([0]), T.uint32(1))
                            )
                            _builder_emit(
                                T.ptx.mbarrier.init.shared.b64(bar_q_utccp.ptr_to([0]), T.uint32(1))
                            )
                            with T.unroll(NUM_BUFS) as stage:
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_rope_ready.ptr_to([stage]), T.uint32(1)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_nope_ready.ptr_to([stage]), T.uint32(128)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_raw_ready.ptr_to([stage]), T.uint32(1)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_raw_free.ptr_to([stage]), T.uint32(128)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_qk_done.ptr_to([stage]), T.uint32(1)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_so_ready.ptr_to([stage]), T.uint32(128)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_sv_done.ptr_to([stage]), T.uint32(1)
                                    )
                                )
                            with T.unroll(NUM_INDEX_BUFS) as index_stage:
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_valid_ready.ptr_to([index_stage]), T.uint32(32)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_valid_free.ptr_to([index_stage]), T.uint32(258)
                                    )
                                )
                            _builder_emit(T.ptx.fence.mbarrier_init.release.cluster())
                    _builder_emit(
                        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                            T.address_of(tmem_start_addr[0]), T.uint32(512)
                        )
                    )
                    allocated_tmem_start = _builder_alloc_scalar("allocated_tmem_start", "uint32")
                    _builder_emit(
                        T.ptx.ld.shared.u32(allocated_tmem_start, tmem_start_addr.ptr_to([0]))
                    )
                    _builder_emit(
                        T.cuda.trap_when_assert_failed(allocated_tmem_start == T.uint32(0))
                    )
                    _builder_emit(T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned())
            _builder_emit(T.cuda.cta_sync())
            with T.If(warpgroup_idx == 0):
                with T.Then():
                    _builder_emit(T.ptx.setmaxnreg.inc.sync.aligned.u32(224))
                    rs_buf = _builder_assign("rs_buf", PipelineState(NUM_BUFS, phase=0))
                    rs_index = _builder_assign("rs_index", PipelineState(NUM_INDEX_BUFS, phase=0))
                    s_frag_layout = _builder_assign(
                        "s_frag_layout",
                        TileLayout(
                            S[(2, 32, 2, 32) : (1 @ wid_in_wg, 1 @ laneid, 2 @ wid_in_wg, 1)]
                        ),
                    )
                    o_smem_win = _builder_assign(
                        "o_smem_win", o_smem.rearrange("h (a b c) -> (b h) (a c)", a=2, b=2, c=128)
                    )
                    scale_pair = _builder_bind(
                        "scale_pair", T.cuda.make_float2(sm_scale_div_log2, sm_scale_div_log2)
                    )
                    attn_sink_log2 = _builder_scalar(
                        "attn_sink_log2", T.float32(-float("inf")), dtype="float32"
                    )
                    if attn_sink_h is not None:
                        T.buffer_store(
                            attn_sink_log2.buffer,
                            T.cuda.ldg(attn_sink.ptr_to([idx_in_warpgroup % B_H]), "float32")
                            * LOG_2_E,
                            [0],
                        )
                    sched_words = _builder_assign("sched_words", T.alloc_local((4,), "uint64"))
                    _builder_emit(load_scheduler_meta(sched_words))
                    sched_i32 = _builder_assign("sched_i32", sched_words.view("int32"))
                    sched_begin_req = _builder_bind("sched_begin_req", sched_i32[0])
                    sched_end_req = _builder_bind("sched_end_req", sched_i32[1])
                    sched_begin_block = _builder_bind("sched_begin_block", sched_i32[2])
                    sched_end_block = _builder_bind("sched_end_block", sched_i32[3])
                    sched_begin_split = _builder_bind("sched_begin_split", sched_i32[4])
                    sched_first_split = _builder_bind("sched_first_split", sched_i32[5])
                    sched_last_split = _builder_bind("sched_last_split", sched_i32[6])
                    batch_bar_phase = _builder_scalar("batch_bar_phase", 0, dtype="int32")
                    with T.If(sched_begin_req < b):
                        with T.Then():
                            with T.serial(
                                sched_begin_req, sched_end_req + 1, unroll=False
                            ) as batch_idx:
                                topk_len = _builder_scalar("topk_len", topk, dtype="int32")
                                if topk_length_h is not None:
                                    T.buffer_store(
                                        topk_len.buffer,
                                        T.cuda.ldg(topk_length.ptr_to([batch_idx]), "int32"),
                                        [0],
                                    )
                                orig_topk_padded = _builder_bind(
                                    "orig_topk_padded",
                                    T.max((topk_len + B_TOPK - 1) // B_TOPK * B_TOPK, B_TOPK),
                                )
                                extra_topk_len = _builder_scalar(
                                    "extra_topk_len", extra_topk, dtype="int32"
                                )
                                if extra_topk_length_h is not None:
                                    T.buffer_store(
                                        extra_topk_len.buffer,
                                        T.cuda.ldg(extra_topk_length.ptr_to([batch_idx]), "int32"),
                                        [0],
                                    )
                                total_topk_padded = _builder_bind(
                                    "total_topk_padded",
                                    orig_topk_padded
                                    + (extra_topk_len + B_TOPK - 1) // B_TOPK * B_TOPK,
                                )
                                start_block = _builder_bind(
                                    "start_block",
                                    T.if_then_else(
                                        batch_idx == sched_begin_req, sched_begin_block, 0
                                    ),
                                )
                                end_block = _builder_bind(
                                    "end_block",
                                    T.if_then_else(
                                        batch_idx == sched_end_req,
                                        sched_end_block,
                                        total_topk_padded // B_TOPK,
                                    ),
                                )
                                is_split = _builder_scalar(
                                    "is_split",
                                    T.cast(
                                        T.if_then_else(
                                            batch_idx == sched_begin_req,
                                            sched_first_split,
                                            T.if_then_else(
                                                batch_idx == sched_end_req, sched_last_split, 0
                                            ),
                                        ),
                                        "bool",
                                    ),
                                    dtype="bool",
                                )
                                is_no_split = _builder_scalar(
                                    "is_no_split", T.Not(is_split), dtype="bool"
                                )
                                n_split_idx = _builder_bind(
                                    "n_split_idx",
                                    T.if_then_else(
                                        batch_idx == sched_begin_req,
                                        T.cuda.ldg(num_splits.ptr_to([batch_idx]), "int32")
                                        + sched_begin_split,
                                        T.cuda.ldg(num_splits.ptr_to([batch_idx]), "int32"),
                                    ),
                                )
                                num_orig_blocks = _builder_bind(
                                    "num_orig_blocks", orig_topk_padded // B_TOPK
                                )
                                is_last_batch = _builder_scalar(
                                    "is_last_batch", batch_idx == sched_end_req, dtype="bool"
                                )
                                _builder_emit(T.ptx.cp.async_.bulk.wait_group.read(0))
                                _builder_emit(bar_last_store_done.arrive(0))
                                mi = _builder_scalar("mi", MAX_INIT_VAL, dtype="float32")
                                li = _builder_scalar("li", 0.0, dtype="float32")
                                real_mi = _builder_scalar(
                                    "real_mi", T.float32(-float("inf")), dtype="float32"
                                )
                                with T.serial(start_block, end_block, unroll=False) as block_idx:
                                    _builder_emit(T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128))
                                    _builder_emit(
                                        bar_valid_ready.wait(rs_index.stage, rs_index.phase)
                                    )
                                    _builder_emit(bar_qk_done.wait(rs_buf.stage, rs_buf.phase))
                                    _builder_emit(T.ptx.tcgen05.fence__after_thread_sync())
                                    _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                    p_frag = _builder_assign(
                                        "p_frag",
                                        T.alloc_tcgen05_ldst_frag(
                                            "32x32b", (128, B_TOPK // 2), "float32"
                                        ),
                                    )
                                    p_peer_frag = _builder_assign(
                                        "p_peer_frag",
                                        T.alloc_tcgen05_ldst_frag(
                                            "32x32b", (128, B_TOPK // 2), "float32"
                                        ),
                                    )
                                    p = _builder_assign("p", p_frag.local())
                                    p_peer = _builder_assign("p_peer", p_peer_frag.local())
                                    with T.If(warp_idx < 2):
                                        with T.Then():
                                            _builder_emit(
                                                T.evaluate(
                                                    _tmem_load(p, T.uint32(400), B_TOPK // 2)
                                                )
                                            )
                                            _builder_emit(
                                                T.evaluate(
                                                    _tmem_load(
                                                        p_peer,
                                                        T.cuda.get_tmem_addr(
                                                            T.uint32(400), 0, B_TOPK // 2
                                                        ),
                                                        B_TOPK // 2,
                                                    )
                                                )
                                            )
                                        with T.Else():
                                            _builder_emit(
                                                T.evaluate(
                                                    _tmem_load(p_peer, T.uint32(400), B_TOPK // 2)
                                                )
                                            )
                                            _builder_emit(
                                                T.evaluate(
                                                    _tmem_load(
                                                        p,
                                                        T.cuda.get_tmem_addr(
                                                            T.uint32(400), 0, B_TOPK // 2
                                                        ),
                                                        B_TOPK // 2,
                                                    )
                                                )
                                            )
                                    _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                                    _builder_emit(T.ptx.tcgen05.fence__before_thread_sync())
                                    with T.unroll(B_TOPK // 2 // 4) as exchange_i:
                                        exchange_offset = _builder_bind(
                                            "exchange_offset", exchange_i * 32 * 4 + lane_idx * 4
                                        )
                                        p_peer_words = _builder_assign(
                                            "p_peer_words", p_peer.view("uint32")
                                        )
                                        peer_word = _builder_bind("peer_word", exchange_i * 4)
                                        _builder_emit(
                                            T.ptx.st.shared.v4.u32(
                                                p_exchange.view("uint32").ptr_to(
                                                    [warp_idx ^ 2, exchange_offset]
                                                ),
                                                p_peer_words[peer_word],
                                                p_peer_words[peer_word + 1],
                                                p_peer_words[peer_word + 2],
                                                p_peer_words[peer_word + 3],
                                            )
                                        )
                                    _builder_emit(
                                        T.ptx.bar.sync(
                                            T.uint32(
                                                BAR_WG0_WARP02 + T.bitwise_and(warp_idx, T.int32(1))
                                            ),
                                            64,
                                        )
                                    )
                                    with T.unroll(B_TOPK // 2 // 4) as exchange_i:
                                        exchange_offset = _builder_bind(
                                            "exchange_offset", exchange_i * 32 * 4 + lane_idx * 4
                                        )
                                        peer_tmp = _builder_assign(
                                            "peer_tmp", T.alloc_local((4,), "float32")
                                        )
                                        peer_tmp_words = _builder_assign(
                                            "peer_tmp_words", peer_tmp.view("uint32")
                                        )
                                        _builder_emit(
                                            T.ptx.ld.shared.v4.u32(
                                                peer_tmp_words[0],
                                                peer_tmp_words[1],
                                                peer_tmp_words[2],
                                                peer_tmp_words[3],
                                                p_exchange.view("uint32").ptr_to(
                                                    [warp_idx, exchange_offset]
                                                ),
                                            )
                                        )
                                        pair0 = _builder_alloc_scalar("pair0", "uint64")
                                        pair1 = _builder_alloc_scalar("pair1", "uint64")
                                        _builder_emit(
                                            T.ptx.add.f32x2(
                                                pair0,
                                                T.cuda.make_float2(
                                                    p[exchange_i * 4], p[exchange_i * 4 + 1]
                                                ),
                                                T.cuda.make_float2(peer_tmp[0], peer_tmp[1]),
                                            )
                                        )
                                        _builder_emit(
                                            T.ptx.add.f32x2(
                                                pair1,
                                                T.cuda.make_float2(
                                                    p[exchange_i * 4 + 2], p[exchange_i * 4 + 3]
                                                ),
                                                T.cuda.make_float2(peer_tmp[2], peer_tmp[3]),
                                            )
                                        )
                                        T.buffer_store(p, T.cuda.float2_x(pair0), [exchange_i * 4])
                                        T.buffer_store(
                                            p, T.cuda.float2_y(pair0), [exchange_i * 4 + 1]
                                        )
                                        T.buffer_store(
                                            p, T.cuda.float2_x(pair1), [exchange_i * 4 + 2]
                                        )
                                        T.buffer_store(
                                            p, T.cuda.float2_y(pair1), [exchange_i * 4 + 3]
                                        )
                                    valid_word = _builder_alloc_scalar("valid_word", "uint32")
                                    _builder_emit(
                                        T.ptx.ld.shared.u32(
                                            valid_word,
                                            is_token_valid.view("uint32").ptr_to(
                                                [
                                                    rs_index.stage,
                                                    T.if_then_else(idx_in_warpgroup >= 64, 1, 0),
                                                ]
                                            ),
                                        )
                                    )
                                    with T.unroll(B_TOPK // 2) as p_i:
                                        with T.If(
                                            T.bitwise_and(
                                                T.shift_right(valid_word, T.cast(p_i, "uint32")),
                                                T.uint32(1),
                                            )
                                            == T.uint32(0)
                                        ):
                                            with T.Then():
                                                T.buffer_store(p, T.float32(-float("inf")), [p_i])
                                    cur_pi_max = _builder_scalar(
                                        "cur_pi_max", T.float32(-float("inf")), dtype="float32"
                                    )
                                    with T.unroll(B_TOPK // 2) as p_i:
                                        T.buffer_store(
                                            cur_pi_max.buffer, T.max(cur_pi_max, p[p_i]), [0]
                                        )
                                    T.buffer_store(
                                        cur_pi_max.buffer, cur_pi_max * sm_scale_div_log2, [0]
                                    )
                                    _builder_emit(
                                        T.ptx.st.shared.f32(
                                            rowwise_buf.ptr_to([idx_in_warpgroup]), cur_pi_max
                                        )
                                    )
                                    _builder_emit(T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128))
                                    _builder_emit(bar_valid_free.arrive(rs_index.stage))
                                    peer_pi_max = _builder_alloc_scalar("peer_pi_max", "float32")
                                    _builder_emit(
                                        T.ptx.ld.shared.f32(
                                            peer_pi_max, rowwise_buf.ptr_to([idx_in_warpgroup ^ 64])
                                        )
                                    )
                                    T.buffer_store(
                                        cur_pi_max.buffer, T.max(cur_pi_max, peer_pi_max), [0]
                                    )
                                    T.buffer_store(real_mi.buffer, T.max(real_mi, cur_pi_max), [0])
                                    should_scale_o = _builder_bind(
                                        "should_scale_o",
                                        T.cuda.any_sync(T.uint32(4294967295), cur_pi_max - mi > 6.0)
                                        != 0,
                                    )
                                    new_max = _builder_alloc_scalar("new_max", "float32")
                                    scale_for_old = _builder_alloc_scalar(
                                        "scale_for_old", "float32"
                                    )
                                    with T.If(T.Not(should_scale_o)):
                                        with T.Then():
                                            T.buffer_store(scale_for_old.buffer, 1.0, [0])
                                            T.buffer_store(new_max.buffer, mi, [0])
                                        with T.Else():
                                            T.buffer_store(
                                                new_max.buffer, T.max(cur_pi_max, mi), [0]
                                            )
                                            _builder_emit(
                                                T.ptx.ex2.approx.ftz.f32(
                                                    scale_for_old, mi - new_max
                                                )
                                            )
                                    T.buffer_store(mi.buffer, new_max, [0])
                                    s_frag = _builder_assign(
                                        "s_frag",
                                        T.alloc_buffer(
                                            (B_H, B_TOPK),
                                            "bfloat16",
                                            scope="local",
                                            layout=s_frag_layout,
                                        ),
                                    )
                                    s_pack = _builder_assign(
                                        "s_pack", s_frag.local().view("uint32")
                                    )
                                    cur_sum_pair = _builder_scalar(
                                        "cur_sum_pair", T.cuda.make_float2(0.0, 0.0), dtype="uint64"
                                    )
                                    neg_max_pair = _builder_bind(
                                        "neg_max_pair", T.cuda.make_float2(-new_max, -new_max)
                                    )
                                    with T.unroll(B_TOPK // 2 // 2) as s_i:
                                        p_pair = _builder_bind(
                                            "p_pair", T.cuda.make_float2(p[s_i * 2], p[s_i * 2 + 1])
                                        )
                                        soft_pair = _builder_alloc_scalar("soft_pair", "uint64")
                                        _builder_emit(
                                            T.ptx.fma.rn.f32x2(
                                                soft_pair, p_pair, scale_pair, neg_max_pair
                                            )
                                        )
                                        sx = _builder_alloc_scalar("sx", "float32")
                                        sy = _builder_alloc_scalar("sy", "float32")
                                        _builder_emit(
                                            T.ptx.ex2.approx.ftz.f32(sx, T.cuda.float2_x(soft_pair))
                                        )
                                        _builder_emit(
                                            T.ptx.ex2.approx.ftz.f32(sy, T.cuda.float2_y(soft_pair))
                                        )
                                        _builder_emit(
                                            T.ptx.add.f32x2(
                                                cur_sum_pair,
                                                cur_sum_pair,
                                                T.cuda.make_float2(sx, sy),
                                            )
                                        )
                                        T.buffer_store(
                                            s_pack, T.cuda.float22bfloat162_rn(sx, sy), [s_i]
                                        )
                                    cur_sum = _builder_bind(
                                        "cur_sum",
                                        T.cuda.float2_x(cur_sum_pair)
                                        + T.cuda.float2_y(cur_sum_pair),
                                    )
                                    li_next = _builder_alloc_scalar("li_next", "float32")
                                    _builder_emit(
                                        T.ptx.fma.rn.f32(li_next, li, scale_for_old, cur_sum)
                                    )
                                    T.buffer_store(li.buffer, li_next, [0])
                                    s_base = _builder_bind(
                                        "s_base",
                                        idx_in_warpgroup // 64 * 2048 + idx_in_warpgroup % 64 * 8,
                                    )
                                    s_words = _builder_assign(
                                        "s_words", s_frag.local().view("uint32")
                                    )
                                    with T.unroll(4) as s_store_i:
                                        s_ptr = _builder_bind(
                                            "s_ptr",
                                            T.ptr_byte_offset(
                                                s_smem_gemm.ptr_to([0, 0]),
                                                (s_base + s_store_i * 512) * BF16_BYTES,
                                                "bfloat16",
                                            ),
                                        )
                                        s_word = _builder_bind("s_word", s_store_i * 4)
                                        _builder_emit(
                                            T.ptx.st.shared.v4.u32(
                                                s_ptr,
                                                s_words[s_word],
                                                s_words[s_word + 1],
                                                s_words[s_word + 2],
                                                s_words[s_word + 3],
                                            )
                                        )
                                    with T.If(T.And(block_idx != start_block, should_scale_o)):
                                        with T.Then():
                                            scale_for_old_pair = _builder_bind(
                                                "scale_for_old_pair",
                                                T.cuda.make_float2(scale_for_old, scale_for_old),
                                            )
                                            _builder_emit(T.ptx.tcgen05.fence__after_thread_sync())
                                            o_rescale_frag = _builder_assign(
                                                "o_rescale_frag",
                                                T.alloc_tcgen05_ldst_frag(
                                                    "32x32b", (128, 64), "float32"
                                                ),
                                            )
                                            o_rescale = _builder_assign(
                                                "o_rescale", o_rescale_frag.local()
                                            )
                                            with T.unroll(D_V // 2 // 64) as o_chunk:
                                                _builder_emit(
                                                    T.evaluate(
                                                        _tmem_load(
                                                            o_rescale,
                                                            T.cuda.get_tmem_addr(
                                                                T.uint32(0), 0, o_chunk * 64
                                                            ),
                                                            64,
                                                        )
                                                    )
                                                )
                                                _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                                                scaled_pair = _builder_alloc_scalar(
                                                    "scaled_pair", "uint64"
                                                )
                                                with T.unroll(64 // 2) as scale_i:
                                                    _builder_emit(
                                                        T.ptx.mul.f32x2(
                                                            scaled_pair,
                                                            T.cuda.make_float2(
                                                                o_rescale[scale_i * 2],
                                                                o_rescale[scale_i * 2 + 1],
                                                            ),
                                                            scale_for_old_pair,
                                                        )
                                                    )
                                                    T.buffer_store(
                                                        o_rescale,
                                                        T.cuda.float2_x(scaled_pair),
                                                        [scale_i * 2],
                                                    )
                                                    T.buffer_store(
                                                        o_rescale,
                                                        T.cuda.float2_y(scaled_pair),
                                                        [scale_i * 2 + 1],
                                                    )
                                                _builder_emit(
                                                    T.evaluate(
                                                        _tmem_store(
                                                            o_rescale,
                                                            T.cuda.get_tmem_addr(
                                                                T.uint32(0), 0, o_chunk * 64
                                                            ),
                                                        )
                                                    )
                                                )
                                                _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
                                            _builder_emit(T.ptx.tcgen05.fence__before_thread_sync())
                                    _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                    _builder_emit(bar_so_ready.arrive(rs_buf.stage))
                                    with T.If(block_idx != end_block - 1):
                                        with T.Then():
                                            _builder_emit(rs_buf.advance())
                                            _builder_emit(rs_index.advance())
                                with T.If(real_mi == T.float32(-float("inf"))):
                                    with T.Then():
                                        T.buffer_store(li.buffer, 0.0, [0])
                                        T.buffer_store(mi.buffer, T.float32(-float("inf")), [0])
                                _builder_emit(T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128))
                                _builder_emit(
                                    T.ptx.st.shared.f32(rowwise_buf.ptr_to([idx_in_warpgroup]), li)
                                )
                                _builder_emit(T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128))
                                peer_li = _builder_alloc_scalar("peer_li", "float32")
                                _builder_emit(
                                    T.ptx.ld.shared.f32(
                                        peer_li, rowwise_buf.ptr_to([idx_in_warpgroup ^ 64])
                                    )
                                )
                                T.buffer_store(li.buffer, li + peer_li, [0])
                                with T.If(idx_in_warpgroup < B_H):
                                    with T.Then():
                                        with T.If(is_no_split):
                                            with T.Then():
                                                cur_lse = _builder_alloc_scalar(
                                                    "cur_lse", "float32"
                                                )
                                                _builder_emit(
                                                    T.ptx.fma.rn.f32(
                                                        cur_lse, mi, T.float32(LN_2), T.log(li)
                                                    )
                                                )
                                                _builder_emit(
                                                    T.ptx.st.global_.f32(
                                                        lse.ptr_to(
                                                            [
                                                                batch_idx * stride_lse_b
                                                                + s_q_idx * stride_lse_s_q
                                                                + idx_in_warpgroup
                                                            ]
                                                        ),
                                                        T.if_then_else(
                                                            cur_lse == T.float32(-float("inf")),
                                                            T.float32(float("inf")),
                                                            cur_lse,
                                                        ),
                                                    )
                                                )
                                            with T.Else():
                                                _builder_emit(
                                                    T.ptx.st.global_.f32(
                                                        lse_accum.ptr_to(
                                                            [
                                                                n_split_idx * stride_lse_accum_split
                                                                + s_q_idx * stride_lse_accum_s_q
                                                                + idx_in_warpgroup
                                                            ]
                                                        ),
                                                        T.log2(li) + mi,
                                                    )
                                                )
                                _builder_emit(bar_sv_done.wait(rs_buf.stage, rs_buf.phase))
                                _builder_emit(rs_buf.advance())
                                _builder_emit(rs_index.advance())
                                _builder_emit(T.ptx.tcgen05.fence__after_thread_sync())
                                if use_pdl:
                                    with T.If(T.And(True, is_last_batch)):
                                        with T.Then():
                                            _builder_emit(T.ptx.griddepcontrol.launch_dependents())
                                with T.If(is_no_split):
                                    with T.Then():
                                        sink_exp = _builder_alloc_scalar("sink_exp", "float32")
                                        _builder_emit(
                                            T.ptx.ex2.approx.ftz.f32(sink_exp, attn_sink_log2 - mi)
                                        )
                                        output_scale = _builder_bind(
                                            "output_scale",
                                            T.if_then_else(
                                                li == 0.0, 0.0, T.cuda.fdividef(1.0, li + sink_exp)
                                            ),
                                        )
                                        output_scale_pair = _builder_bind(
                                            "output_scale_pair",
                                            T.cuda.make_float2(output_scale, output_scale),
                                        )
                                        o_epi_frag = _builder_assign(
                                            "o_epi_frag",
                                            T.alloc_tcgen05_ldst_frag(
                                                "32x32b", (128, 64), "float32"
                                            ),
                                        )
                                        o_epi_bf16_frag = _builder_assign(
                                            "o_epi_bf16_frag",
                                            T.alloc_tcgen05_ldst_frag(
                                                "32x32b", (128, 64), "bfloat16"
                                            ),
                                        )
                                        o_epi = _builder_assign("o_epi", o_epi_frag.local())

                                        def emit_no_split_epilogue(epi_i: T.constexpr):
                                            _builder_emit(
                                                T.evaluate(
                                                    _tmem_load(
                                                        o_epi,
                                                        T.cuda.get_tmem_addr(
                                                            T.uint32(0), 0, epi_i * 64
                                                        ),
                                                        64,
                                                    )
                                                )
                                            )
                                            _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                                            scaled_pair = _builder_alloc_scalar(
                                                "scaled_pair", "uint64"
                                            )
                                            with T.unroll(64 // 2) as scale_i:
                                                _builder_emit(
                                                    T.ptx.mul.f32x2(
                                                        scaled_pair,
                                                        T.cuda.make_float2(
                                                            o_epi[scale_i * 2],
                                                            o_epi[scale_i * 2 + 1],
                                                        ),
                                                        output_scale_pair,
                                                    )
                                                )
                                                T.buffer_store(
                                                    o_epi,
                                                    T.cuda.float2_x(scaled_pair),
                                                    [scale_i * 2],
                                                )
                                                T.buffer_store(
                                                    o_epi,
                                                    T.cuda.float2_y(scaled_pair),
                                                    [scale_i * 2 + 1],
                                                )
                                            o_epi_bf16 = _builder_assign(
                                                "o_epi_bf16", o_epi_bf16_frag.local()
                                            )
                                            with T.unroll(64 // 2) as cast_i:
                                                _builder_emit(
                                                    T.evaluate(
                                                        _cast_f32x2_bf16x2(
                                                            o_epi_bf16, o_epi, cast_i * 2
                                                        )
                                                    )
                                                )
                                            col_base = (
                                                D_V // 2 if epi_i * 64 >= D_V // 4 else 0
                                            ) + epi_i * 64 % (D_V // 4)
                                            o_epi_words = _builder_assign(
                                                "o_epi_words", o_epi_bf16.view("uint32")
                                            )
                                            with T.unroll(8) as o_store_i:
                                                o_smem_offset = _builder_bind(
                                                    "o_smem_offset",
                                                    col_base * B_H
                                                    + idx_in_warpgroup // 64 * 8192
                                                    + idx_in_warpgroup % 64 * 64
                                                    + T.bitwise_xor(
                                                        o_store_i * 8,
                                                        T.shift_left(
                                                            T.bitwise_and(
                                                                idx_in_warpgroup // 64 * 128
                                                                + idx_in_warpgroup % 64,
                                                                7,
                                                            ),
                                                            3,
                                                        ),
                                                    ),
                                                )
                                                o_smem_ptr = _builder_bind(
                                                    "o_smem_ptr",
                                                    T.ptr_byte_offset(
                                                        o_smem_win.ptr_to([0, 0]),
                                                        o_smem_offset * BF16_BYTES,
                                                        "bfloat16",
                                                    ),
                                                )
                                                o_word = _builder_bind("o_word", o_store_i * 4)
                                                _builder_emit(
                                                    T.ptx.st.shared.v4.u32(
                                                        o_smem_ptr,
                                                        o_epi_words[o_word],
                                                        o_epi_words[o_word + 1],
                                                        o_epi_words[o_word + 2],
                                                        o_epi_words[o_word + 3],
                                                    )
                                                )
                                            _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                            _builder_emit(
                                                T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128)
                                            )
                                            with T.If(warp_idx == 0):
                                                with T.Then():
                                                    with T.If(T.cuda.elect_sync() != T.uint32(0)):
                                                        with T.Then():
                                                            _builder_emit(
                                                                T.evaluate(
                                                                    T.ptx[_TMA_S2G_4D](
                                                                        T.address_of(out_tensormap),
                                                                        T.int32(col_base),
                                                                        T.int32(0),
                                                                        T.cast(s_q_idx, "int32"),
                                                                        T.cast(batch_idx, "int32"),
                                                                        T.cuda.cvta_generic_to_shared(
                                                                            T.ptr_byte_offset(
                                                                                o_smem.ptr_to(
                                                                                    [0, 0]
                                                                                ),
                                                                                col_base
                                                                                * B_H
                                                                                * BF16_BYTES,
                                                                                "bfloat16",
                                                                            )
                                                                        ),
                                                                    )
                                                                )
                                                            )
                                            warp1_col_base = col_base + D_V // 4
                                            with T.If(warp_idx == 1):
                                                with T.Then():
                                                    with T.If(T.cuda.elect_sync() != T.uint32(0)):
                                                        with T.Then():
                                                            _builder_emit(
                                                                T.evaluate(
                                                                    T.ptx[_TMA_S2G_4D](
                                                                        T.address_of(out_tensormap),
                                                                        T.int32(warp1_col_base),
                                                                        T.int32(0),
                                                                        T.cast(s_q_idx, "int32"),
                                                                        T.cast(batch_idx, "int32"),
                                                                        T.cuda.cvta_generic_to_shared(
                                                                            T.ptr_byte_offset(
                                                                                o_smem.ptr_to(
                                                                                    [0, 0]
                                                                                ),
                                                                                warp1_col_base
                                                                                * B_H
                                                                                * BF16_BYTES,
                                                                                "bfloat16",
                                                                            )
                                                                        ),
                                                                    )
                                                                )
                                                            )

                                        _builder_emit(emit_no_split_epilogue(0))
                                        _builder_emit(emit_no_split_epilogue(1))
                                        _builder_emit(emit_no_split_epilogue(2))
                                        _builder_emit(emit_no_split_epilogue(3))
                                        _builder_emit(T.ptx.cp.async_.bulk.commit_group())
                                    with T.Else():
                                        output_scale = _builder_bind(
                                            "output_scale",
                                            T.if_then_else(
                                                li == 0.0, 0.0, T.cuda.fdividef(1.0, li)
                                            ),
                                        )
                                        output_scale_pair = _builder_bind(
                                            "output_scale_pair",
                                            T.cuda.make_float2(output_scale, output_scale),
                                        )
                                        split_frag = _builder_assign(
                                            "split_frag",
                                            T.alloc_tcgen05_ldst_frag(
                                                "32x32b", (128, 64), "float32"
                                            ),
                                        )
                                        split_local = _builder_assign(
                                            "split_local", split_frag.local()
                                        )
                                        with T.unroll(D_V // 2 // 64) as epi_i:
                                            _builder_emit(
                                                T.evaluate(
                                                    _tmem_load(
                                                        split_local,
                                                        T.cuda.get_tmem_addr(
                                                            T.uint32(0), 0, epi_i * 64
                                                        ),
                                                        64,
                                                    )
                                                )
                                            )
                                            _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                                            scaled_pair = _builder_alloc_scalar(
                                                "scaled_pair", "uint64"
                                            )
                                            with T.unroll(64 // 2) as scale_i:
                                                _builder_emit(
                                                    T.ptx.mul.f32x2(
                                                        scaled_pair,
                                                        T.cuda.make_float2(
                                                            split_local[scale_i * 2],
                                                            split_local[scale_i * 2 + 1],
                                                        ),
                                                        output_scale_pair,
                                                    )
                                                )
                                                T.buffer_store(
                                                    split_local,
                                                    T.cuda.float2_x(scaled_pair),
                                                    [scale_i * 2],
                                                )
                                                T.buffer_store(
                                                    split_local,
                                                    T.cuda.float2_y(scaled_pair),
                                                    [scale_i * 2 + 1],
                                                )
                                            col_base = _builder_bind(
                                                "col_base",
                                                idx_in_warpgroup // 64 * 128
                                                + T.if_then_else(
                                                    epi_i * 64 >= D_V // 4, D_V // 2, 0
                                                )
                                                + epi_i * 64 % (D_V // 4),
                                            )
                                            split_words = _builder_assign(
                                                "split_words", split_local.view("uint32")
                                            )
                                            with T.unroll(64 // 4) as j:
                                                split_word = _builder_bind("split_word", j * 4)
                                                _builder_emit(
                                                    T.ptx.st.shared.v4.u32(
                                                        o_accum_smem.view("uint32").ptr_to(
                                                            [
                                                                idx_in_warpgroup % 64,
                                                                col_base + j * 4,
                                                            ]
                                                        ),
                                                        split_words[split_word],
                                                        split_words[split_word + 1],
                                                        split_words[split_word + 2],
                                                        split_words[split_word + 3],
                                                    )
                                                )
                                        _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                        _builder_emit(T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128))
                                        with T.If(T.cuda.elect_sync() != T.uint32(0)):
                                            with T.Then():
                                                with T.unroll(B_H // 4) as local_row:
                                                    smem_row = _builder_bind(
                                                        "smem_row", local_row * 4 + warp_idx
                                                    )
                                                    _builder_emit(
                                                        T.ptx[
                                                            "cp.async.bulk.global.shared::cta.bulk_group"
                                                        ](
                                                            o_accum.ptr_to(
                                                                [
                                                                    n_split_idx
                                                                    * stride_o_accum_split
                                                                    + s_q_idx * stride_o_accum_s_q
                                                                    + smem_row * stride_o_accum_h_q
                                                                ]
                                                            ),
                                                            o_accum_smem.ptr_to([smem_row, 0]),
                                                            T.uint32(D_V * 4),
                                                        )
                                                    )
                                                _builder_emit(T.ptx.cp.async_.bulk.commit_group())
                                _builder_emit(
                                    T.ptx.barrier.sync(
                                        T.uint32(BAR_EVERYONE_SYNC), T.uint32(NUM_THREADS)
                                    )
                                )
                                T.buffer_store(batch_bar_phase.buffer, batch_bar_phase ^ 1, [0])
                    with T.If(warp_idx == 0):
                        with T.Then():
                            _builder_emit(
                                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                                    T.uint32(0), T.uint32(512)
                                )
                            )
                with T.Else():
                    with T.If(warpgroup_idx == 1):
                        with T.Then():
                            _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(72))
                            wg1_warp_idx = _builder_bind(
                                "wg1_warp_idx",
                                T.tvm_warp_shuffle(
                                    T.uint32(4294967295), T.cuda.thread_rank() // 32, 0, 32, 32
                                ),
                            )

                            def run_wg1_role(role: T.constexpr):
                                rs_buf = _builder_assign("rs_buf", PipelineState(NUM_BUFS, phase=0))
                                rs_index = _builder_assign(
                                    "rs_index", PipelineState(NUM_INDEX_BUFS, phase=0)
                                )
                                q_sw128_tmem = _builder_assign(
                                    "q_sw128_tmem", q_tmem.sub[:, :, 0:256]
                                )
                                k_full_tiled = _builder_assign(
                                    "k_full_tiled",
                                    k_full.rearrange(
                                        "b r (dc h ci) -> b (h r) (dc ci)", dc=4, h=2, ci=64
                                    ),
                                )
                                q_tail_tmem = _builder_assign(
                                    "q_tail_tmem",
                                    q_tmem.sub[:, :, q_tail_start : q_tail_start + 32],
                                )
                                k_rope_tiled = _builder_assign(
                                    "k_rope_tiled",
                                    k_rope.rearrange("b r (h ci) -> b (h r) ci", h=2, ci=32),
                                )
                                qk_nope_desc = _builder_assign("qk_nope_desc", SmemDescriptor())
                                qk_rope_desc = _builder_assign("qk_rope_desc", SmemDescriptor())
                                if role == 4:
                                    _builder_emit(
                                        qk_nope_desc.init(
                                            k_full_tiled.ptr_to([0, 0, 0]),
                                            ldo=1024,
                                            sdo=64,
                                            swizzle=3,
                                        )
                                    )
                                    if is_v32:
                                        _builder_emit(
                                            qk_rope_desc.init(
                                                k_rope_tiled.ptr_to([0, 0, 0]),
                                                ldo=0,
                                                sdo=32,
                                                swizzle=2,
                                            )
                                        )
                                if role == 7:
                                    tma_coords_step_per_token = _builder_bind(
                                        "tma_coords_step_per_token",
                                        (656 if is_v32 else 576) // tma_k_stride,
                                    )
                                    tma_coords_step_per_block = _builder_bind(
                                        "tma_coords_step_per_block", stride_kv_block // tma_k_stride
                                    )
                                    tma_coords_step_per_extra_block = _builder_bind(
                                        "tma_coords_step_per_extra_block",
                                        stride_extra_kv_block // tma_k_stride,
                                    )
                                    k_scales_ptr_u64 = _builder_bind(
                                        "k_scales_ptr_u64",
                                        T.reinterpret(
                                            "uint64",
                                            kv.ptr_to([0, d_nope // BF16_BYTES])
                                            if is_v32
                                            else kv.ptr_to([page_block_size, 0]),
                                        ),
                                    )
                                    extra_k_scales_ptr_u64 = _builder_scalar(
                                        "extra_k_scales_ptr_u64", T.uint64(0), dtype="uint64"
                                    )
                                    if extra_kv_h is not None:
                                        T.buffer_store(
                                            extra_k_scales_ptr_u64.buffer,
                                            T.reinterpret(
                                                "uint64",
                                                extra_kv.ptr_to([0, d_nope // BF16_BYTES])
                                                if is_v32
                                                else extra_kv.ptr_to([extra_page_block_size, 0]),
                                            ),
                                            [0],
                                        )
                                sched_words = _builder_assign(
                                    "sched_words", T.alloc_local((4,), "uint64")
                                )
                                _builder_emit(load_scheduler_meta(sched_words))
                                sched_i32 = _builder_assign("sched_i32", sched_words.view("int32"))
                                sched_begin_req = _builder_bind("sched_begin_req", sched_i32[0])
                                sched_end_req = _builder_bind("sched_end_req", sched_i32[1])
                                sched_begin_block = _builder_bind("sched_begin_block", sched_i32[2])
                                sched_end_block = _builder_bind("sched_end_block", sched_i32[3])
                                sched_begin_split = _builder_bind("sched_begin_split", sched_i32[4])
                                sched_first_split = _builder_bind("sched_first_split", sched_i32[5])
                                sched_last_split = _builder_bind("sched_last_split", sched_i32[6])
                                batch_bar_phase = _builder_scalar(
                                    "batch_bar_phase", 0, dtype="int32"
                                )
                                with T.If(sched_begin_req < b):
                                    with T.Then():
                                        with T.serial(
                                            sched_begin_req, sched_end_req + 1, unroll=False
                                        ) as batch_idx:
                                            topk_len = _builder_scalar(
                                                "topk_len", topk, dtype="int32"
                                            )
                                            if topk_length_h is not None:
                                                T.buffer_store(
                                                    topk_len.buffer,
                                                    T.cuda.ldg(
                                                        topk_length.ptr_to([batch_idx]), "int32"
                                                    ),
                                                    [0],
                                                )
                                            orig_topk_padded = _builder_bind(
                                                "orig_topk_padded",
                                                T.max(
                                                    (topk_len + B_TOPK - 1) // B_TOPK * B_TOPK,
                                                    B_TOPK,
                                                ),
                                            )
                                            extra_topk_len = _builder_scalar(
                                                "extra_topk_len", extra_topk, dtype="int32"
                                            )
                                            if extra_topk_length_h is not None:
                                                T.buffer_store(
                                                    extra_topk_len.buffer,
                                                    T.cuda.ldg(
                                                        extra_topk_length.ptr_to([batch_idx]),
                                                        "int32",
                                                    ),
                                                    [0],
                                                )
                                            total_topk_padded = _builder_bind(
                                                "total_topk_padded",
                                                orig_topk_padded
                                                + (extra_topk_len + B_TOPK - 1) // B_TOPK * B_TOPK,
                                            )
                                            start_block = _builder_bind(
                                                "start_block",
                                                T.if_then_else(
                                                    batch_idx == sched_begin_req,
                                                    sched_begin_block,
                                                    0,
                                                ),
                                            )
                                            end_block = _builder_bind(
                                                "end_block",
                                                T.if_then_else(
                                                    batch_idx == sched_end_req,
                                                    sched_end_block,
                                                    total_topk_padded // B_TOPK,
                                                ),
                                            )
                                            is_split = _builder_scalar(
                                                "is_split",
                                                T.cast(
                                                    T.if_then_else(
                                                        batch_idx == sched_begin_req,
                                                        sched_first_split,
                                                        T.if_then_else(
                                                            batch_idx == sched_end_req,
                                                            sched_last_split,
                                                            0,
                                                        ),
                                                    ),
                                                    "bool",
                                                ),
                                                dtype="bool",
                                            )
                                            is_no_split = _builder_scalar(
                                                "is_no_split", T.Not(is_split), dtype="bool"
                                            )
                                            num_orig_blocks = _builder_bind(
                                                "num_orig_blocks", orig_topk_padded // B_TOPK
                                            )
                                            n_split_idx = _builder_bind(
                                                "n_split_idx",
                                                T.if_then_else(
                                                    batch_idx == sched_begin_req,
                                                    T.cuda.ldg(
                                                        num_splits.ptr_to([batch_idx]), "int32"
                                                    )
                                                    + sched_begin_split,
                                                    T.cuda.ldg(
                                                        num_splits.ptr_to([batch_idx]), "int32"
                                                    ),
                                                ),
                                            )
                                            is_last_batch = _builder_scalar(
                                                "is_last_batch",
                                                batch_idx == sched_end_req,
                                                dtype="bool",
                                            )
                                            if role == 4:
                                                _builder_emit(
                                                    T.cuda.trap_when_assert_failed(
                                                        start_block < end_block
                                                    )
                                                )
                                                with T.unroll(512 // 64) as q_tile:
                                                    _builder_emit(
                                                        T.evaluate(
                                                            T.ptx[_TMA_G2S_4D_CACHE](
                                                                T.ptr_byte_offset(
                                                                    q_sw128.ptr_to([0, 0]),
                                                                    q_tile * B_H * 64 * BF16_BYTES,
                                                                    "bfloat16",
                                                                ),
                                                                T.address_of(q_strided_tensormap),
                                                                T.cast(q_tile * 64, "int32"),
                                                                T.int32(0),
                                                                T.cast(s_q_idx, "int32"),
                                                                T.cast(batch_idx, "int32"),
                                                                T.cuda.cvta_generic_to_shared(
                                                                    bar_q_tma.ptr_to([0])
                                                                ),
                                                                _Q_TMA_CACHE_HINT,
                                                            )
                                                        )
                                                    )
                                                if is_v32:
                                                    _builder_emit(
                                                        T.evaluate(
                                                            T.ptx[_TMA_G2S_5D_CACHE](
                                                                q_sw64.ptr_to([0, 0]),
                                                                T.address_of(q_tail_tensormap),
                                                                T.int32(0),
                                                                T.int32(0),
                                                                T.int32(16),
                                                                T.cast(s_q_idx, "int32"),
                                                                T.cast(batch_idx, "int32"),
                                                                T.cuda.cvta_generic_to_shared(
                                                                    bar_q_tma.ptr_to([0])
                                                                ),
                                                                _Q_TMA_CACHE_HINT,
                                                            )
                                                        )
                                                    )
                                                _builder_emit(
                                                    bar_q_tma.arrive(
                                                        0, tx_count=B_H * d_qk * BF16_BYTES
                                                    )
                                                )
                                                _builder_emit(bar_q_tma.wait(0, batch_bar_phase))
                                                _builder_emit(
                                                    T.ptx.tcgen05.fence__after_thread_sync()
                                                )
                                                q_main_cp_view = _builder_assign(
                                                    "q_main_cp_view", q_sw128.view(B_H, 4, 2, 64)
                                                )
                                                with T.unroll(16) as q_main_flat:
                                                    q_main_src = _builder_bind(
                                                        "q_main_src",
                                                        T.ptr_byte_offset(
                                                            q_main_cp_view.ptr_to([0, 0, 0, 0]),
                                                            (
                                                                q_main_flat % 4 * 1024
                                                                + q_main_flat // 4 % 4 * 2
                                                            )
                                                            * 16,
                                                            "bfloat16",
                                                        ),
                                                    )
                                                    _builder_emit(
                                                        T.evaluate(
                                                            T.ptx[_TCGEN_CP_128X256](
                                                                T.cast(
                                                                    256
                                                                    + q_main_flat % 4 * 32
                                                                    + q_main_flat // 4 % 4 * 8,
                                                                    "uint32",
                                                                ),
                                                                _replace_smem_desc_addr(
                                                                    q_main_cp_desc, q_main_src
                                                                ),
                                                            )
                                                        )
                                                    )
                                                if is_v32:
                                                    with T.unroll(2) as q_tail_flat:
                                                        q_tail_src = _builder_bind(
                                                            "q_tail_src",
                                                            T.ptr_byte_offset(
                                                                q_sw64.ptr_to([0, 0]),
                                                                q_tail_flat % 2 * 2 * 16,
                                                                "bfloat16",
                                                            ),
                                                        )
                                                        _builder_emit(
                                                            T.evaluate(
                                                                T.ptx[_TCGEN_CP_128X256](
                                                                    T.cast(
                                                                        384 + q_tail_flat % 2 * 8,
                                                                        "uint32",
                                                                    ),
                                                                    _replace_smem_desc_addr(
                                                                        q_tail_cp_desc, q_tail_src
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                _builder_emit(bar_q_utccp.arrive(0))
                                                _builder_emit(bar_q_utccp.wait(0, batch_bar_phase))
                                                _builder_emit(
                                                    T.ptx.tcgen05.fence__after_thread_sync()
                                                )
                                                with T.serial(
                                                    start_block, end_block, unroll=False
                                                ) as block_idx:
                                                    k_stage_elems = (
                                                        B_TOPK * (D_V + 64)
                                                        if is_v32
                                                        else B_TOPK * D_V
                                                    )
                                                    if is_v32:
                                                        _builder_emit(
                                                            bar_rope_ready.wait(
                                                                rs_buf.stage, rs_buf.phase
                                                            )
                                                        )
                                                        _builder_emit(
                                                            T.ptx.tcgen05.fence__after_thread_sync()
                                                        )
                                                        with T.unroll(2) as qk_rope_ki:
                                                            qk_rope_offset = _builder_bind(
                                                                "qk_rope_offset",
                                                                (
                                                                    rs_buf.stage * k_stage_elems
                                                                    + qk_rope_ki * 16
                                                                )
                                                                // 8,
                                                            )
                                                            _builder_emit(
                                                                T.evaluate(
                                                                    T.ptx[_MMA_WS_F16](
                                                                        T.uint32(400),
                                                                        T.cast(
                                                                            384 + qk_rope_ki * 8,
                                                                            "uint32",
                                                                        ),
                                                                        qk_rope_desc.add_16B_offset(
                                                                            qk_rope_offset
                                                                        ),
                                                                        T.uint32(69207184),
                                                                        (
                                                                            qk_rope_ki != 0
                                                                        ).asobject(),
                                                                        T.uint64(0),
                                                                    )
                                                                )
                                                            )
                                                        _builder_emit(
                                                            bar_nope_ready.wait(
                                                                rs_buf.stage, rs_buf.phase
                                                            )
                                                        )
                                                        _builder_emit(
                                                            T.ptx.tcgen05.fence__after_thread_sync()
                                                        )
                                                    else:
                                                        _builder_emit(
                                                            bar_rope_ready.wait(
                                                                rs_buf.stage, rs_buf.phase
                                                            )
                                                        )
                                                        _builder_emit(
                                                            bar_nope_ready.wait(
                                                                rs_buf.stage, rs_buf.phase
                                                            )
                                                        )
                                                        _builder_emit(
                                                            T.ptx.tcgen05.fence__after_thread_sync()
                                                        )
                                                    with T.unroll(16) as qk_nope_ki:
                                                        qk_nope_offset = _builder_bind(
                                                            "qk_nope_offset",
                                                            (
                                                                qk_nope_ki // 2048 * k_stage_elems
                                                                + rs_buf.stage * k_stage_elems
                                                                + qk_nope_ki % 16 // 4 * 8192
                                                                + qk_nope_ki % 2048 // 16 * 64
                                                                + qk_nope_ki % 4 * 16
                                                            )
                                                            // 8,
                                                        )
                                                        _builder_emit(
                                                            T.evaluate(
                                                                T.ptx[_MMA_WS_F16](
                                                                    T.uint32(400),
                                                                    T.cast(
                                                                        256 + qk_nope_ki * 8,
                                                                        "uint32",
                                                                    ),
                                                                    qk_nope_desc.add_16B_offset(
                                                                        qk_nope_offset
                                                                    ),
                                                                    T.uint32(69207184),
                                                                    T.Or(
                                                                        qk_nope_ki != 0,
                                                                        T.cast(
                                                                            T.uint32(
                                                                                1 if is_v32 else 0
                                                                            ),
                                                                            "bool",
                                                                        ),
                                                                    ),
                                                                    T.uint64(0),
                                                                )
                                                            )
                                                        )
                                                    _builder_emit(bar_qk_done.arrive(rs_buf.stage))
                                                    _builder_emit(
                                                        bar_so_ready.wait(
                                                            rs_buf.stage, rs_buf.phase
                                                        )
                                                    )
                                                    _builder_emit(
                                                        T.ptx.tcgen05.fence__after_thread_sync()
                                                    )
                                                    mma_o_accum = _builder_bind(
                                                        "mma_o_accum",
                                                        T.if_then_else(
                                                            block_idx == start_block,
                                                            T.uint32(0),
                                                            T.uint32(1),
                                                        ),
                                                    )
                                                    with T.unroll(4) as pv_ki:
                                                        pv_a_offset = _builder_bind(
                                                            "pv_a_offset",
                                                            (pv_ki % 4 * 1024 + pv_ki // 4 * 8)
                                                            // 8,
                                                        )
                                                        pv_b_lo_offset = _builder_bind(
                                                            "pv_b_lo_offset",
                                                            (
                                                                pv_ki * 16 // 64 * k_stage_elems
                                                                + rs_buf.stage * k_stage_elems
                                                                + pv_ki * 16 % 64 * 64
                                                            )
                                                            // 8,
                                                        )
                                                        _builder_emit(
                                                            T.evaluate(
                                                                T.ptx[_MMA_WS_F16](
                                                                    T.uint32(0),
                                                                    pv_a_lo_desc.add_16B_offset(
                                                                        pv_a_offset
                                                                    ),
                                                                    pv_b_lo_desc.add_16B_offset(
                                                                        pv_b_lo_offset
                                                                    ),
                                                                    T.uint32(71369872),
                                                                    T.Or(
                                                                        pv_ki != 0,
                                                                        T.cast(mma_o_accum, "bool"),
                                                                    ),
                                                                    T.uint64(0),
                                                                )
                                                            )
                                                        )
                                                    with T.unroll(4) as pv_ki:
                                                        pv_a_offset = _builder_bind(
                                                            "pv_a_offset",
                                                            (pv_ki % 4 * 1024 + pv_ki // 4 * 8)
                                                            // 8,
                                                        )
                                                        pv_b_hi_offset = _builder_bind(
                                                            "pv_b_hi_offset",
                                                            (
                                                                pv_ki * 16 // 64 * k_stage_elems
                                                                + rs_buf.stage * k_stage_elems
                                                                + pv_ki * 16 % 64 * 64
                                                                + 16384
                                                            )
                                                            // 8,
                                                        )
                                                        _builder_emit(
                                                            T.evaluate(
                                                                T.ptx[_MMA_WS_F16](
                                                                    T.uint32(128),
                                                                    pv_a_hi_desc.add_16B_offset(
                                                                        pv_a_offset
                                                                    ),
                                                                    pv_b_hi_desc.add_16B_offset(
                                                                        pv_b_hi_offset
                                                                    ),
                                                                    T.uint32(71369872),
                                                                    T.Or(
                                                                        pv_ki != 0,
                                                                        T.cast(mma_o_accum, "bool"),
                                                                    ),
                                                                    T.uint64(0),
                                                                )
                                                            )
                                                        )
                                                    _builder_emit(bar_sv_done.arrive(rs_buf.stage))
                                                    _builder_emit(rs_buf.advance())
                                                    _builder_emit(rs_index.advance())
                                            elif role == 5:
                                                _builder_emit(bar_q_utccp.wait(0, batch_bar_phase))
                                                _builder_emit(
                                                    bar_last_store_done.wait(0, batch_bar_phase)
                                                )
                                                with T.serial(
                                                    start_block, end_block, unroll=False
                                                ) as block_idx:
                                                    _builder_emit(
                                                        bar_valid_ready.wait(
                                                            rs_index.stage, rs_index.phase
                                                        )
                                                    )
                                                    _builder_emit(
                                                        bar_raw_free.wait(
                                                            rs_buf.stage, rs_buf.phase ^ 1
                                                        )
                                                    )
                                                    cur_indices = _builder_assign(
                                                        "cur_indices", T.alloc_local((4,), "int32")
                                                    )
                                                    next_indices = _builder_assign(
                                                        "next_indices", T.alloc_local((4,), "int32")
                                                    )
                                                    cur_index_words = _builder_assign(
                                                        "cur_index_words",
                                                        cur_indices.view("uint32"),
                                                    )
                                                    _builder_emit(
                                                        T.ptx.ld.shared.v4.u32(
                                                            cur_index_words[0],
                                                            cur_index_words[1],
                                                            cur_index_words[2],
                                                            cur_index_words[3],
                                                            tma_coord.view("uint32").ptr_to(
                                                                [rs_index.stage, 0]
                                                            ),
                                                        )
                                                    )
                                                    with T.unroll(B_TOPK // 4) as row4:
                                                        row = _builder_bind("row", row4 * 4)
                                                        with T.If(row + 4 < B_TOPK):
                                                            with T.Then():
                                                                next_index_words = _builder_assign(
                                                                    "next_index_words",
                                                                    next_indices.view("uint32"),
                                                                )
                                                                _builder_emit(
                                                                    T.ptx.ld.shared.v4.u32(
                                                                        next_index_words[0],
                                                                        next_index_words[1],
                                                                        next_index_words[2],
                                                                        next_index_words[3],
                                                                        tma_coord.view(
                                                                            "uint32"
                                                                        ).ptr_to(
                                                                            [
                                                                                rs_index.stage,
                                                                                row + 4,
                                                                            ]
                                                                        ),
                                                                    )
                                                                )
                                                        selected_nope_tensormap = _builder_scalar(
                                                            "selected_nope_tensormap",
                                                            T.reinterpret(
                                                                "uint64",
                                                                T.address_of(kv_nope_tensormap),
                                                            ),
                                                            dtype="uint64",
                                                        )
                                                        if extra_kv_h is not None:
                                                            T.buffer_store(
                                                                selected_nope_tensormap.buffer,
                                                                T.if_then_else(
                                                                    block_idx >= num_orig_blocks,
                                                                    T.reinterpret(
                                                                        "uint64",
                                                                        T.address_of(
                                                                            extra_kv_nope_tensormap
                                                                        ),
                                                                    ),
                                                                    T.reinterpret(
                                                                        "uint64",
                                                                        T.address_of(
                                                                            kv_nope_tensormap
                                                                        ),
                                                                    ),
                                                                ),
                                                                [0],
                                                            )
                                                        _builder_emit(
                                                            T.evaluate(
                                                                T.ptx[_TMA_GATHER4_2D_CACHE](
                                                                    raw_nope.ptr_to(
                                                                        [rs_buf.stage, row, 0]
                                                                    ),
                                                                    T.reinterpret(
                                                                        T.handle().ty,
                                                                        selected_nope_tensormap,
                                                                    ),
                                                                    T.int32(0),
                                                                    cur_indices[0],
                                                                    cur_indices[1],
                                                                    cur_indices[2],
                                                                    cur_indices[3],
                                                                    T.cuda.cvta_generic_to_shared(
                                                                        bar_raw_ready.ptr_to(
                                                                            [rs_buf.stage]
                                                                        )
                                                                    ),
                                                                    _KV_TMA_CACHE_HINT,
                                                                )
                                                            )
                                                        )
                                                        T.buffer_store(
                                                            cur_indices, next_indices[0], [0]
                                                        )
                                                        T.buffer_store(
                                                            cur_indices, next_indices[1], [1]
                                                        )
                                                        T.buffer_store(
                                                            cur_indices, next_indices[2], [2]
                                                        )
                                                        T.buffer_store(
                                                            cur_indices, next_indices[3], [3]
                                                        )
                                                    _builder_emit(
                                                        bar_raw_ready.arrive(
                                                            rs_buf.stage, tx_count=B_TOPK * d_nope
                                                        )
                                                    )
                                                    _builder_emit(
                                                        bar_valid_free.arrive(rs_index.stage)
                                                    )
                                                    _builder_emit(rs_buf.advance())
                                                    _builder_emit(rs_index.advance())
                                            elif role == 6:
                                                _builder_emit(bar_q_utccp.wait(0, batch_bar_phase))
                                                _builder_emit(
                                                    bar_last_store_done.wait(0, batch_bar_phase)
                                                )
                                                with T.serial(
                                                    start_block, end_block, unroll=False
                                                ) as block_idx:
                                                    _builder_emit(
                                                        bar_valid_ready.wait(
                                                            rs_index.stage, rs_index.phase
                                                        )
                                                    )
                                                    if is_v32:
                                                        _builder_emit(
                                                            bar_qk_done.wait(
                                                                rs_buf.stage, rs_buf.phase ^ 1
                                                            )
                                                        )
                                                    else:
                                                        _builder_emit(
                                                            bar_sv_done.wait(
                                                                rs_buf.stage, rs_buf.phase ^ 1
                                                            )
                                                        )
                                                    cur_indices = _builder_assign(
                                                        "cur_indices", T.alloc_local((4,), "int32")
                                                    )
                                                    next_indices = _builder_assign(
                                                        "next_indices", T.alloc_local((4,), "int32")
                                                    )
                                                    cur_index_words = _builder_assign(
                                                        "cur_index_words",
                                                        cur_indices.view("uint32"),
                                                    )
                                                    _builder_emit(
                                                        T.ptx.ld.shared.v4.u32(
                                                            cur_index_words[0],
                                                            cur_index_words[1],
                                                            cur_index_words[2],
                                                            cur_index_words[3],
                                                            tma_coord.view("uint32").ptr_to(
                                                                [rs_index.stage, 0]
                                                            ),
                                                        )
                                                    )
                                                    with T.unroll(B_TOPK // 4) as row4:
                                                        row = _builder_bind("row", row4 * 4)
                                                        with T.If(row + 4 < B_TOPK):
                                                            with T.Then():
                                                                next_index_words = _builder_assign(
                                                                    "next_index_words",
                                                                    next_indices.view("uint32"),
                                                                )
                                                                _builder_emit(
                                                                    T.ptx.ld.shared.v4.u32(
                                                                        next_index_words[0],
                                                                        next_index_words[1],
                                                                        next_index_words[2],
                                                                        next_index_words[3],
                                                                        tma_coord.view(
                                                                            "uint32"
                                                                        ).ptr_to(
                                                                            [
                                                                                rs_index.stage,
                                                                                row + 4,
                                                                            ]
                                                                        ),
                                                                    )
                                                                )
                                                        with T.unroll(64 // rope_tile) as rope_part:
                                                            if is_v32:
                                                                rope_tma_dst = _builder_bind(
                                                                    "rope_tma_dst",
                                                                    T.ptr_byte_offset(
                                                                        k_union.ptr_to([0, 0, 0]),
                                                                        (
                                                                            rs_buf.stage
                                                                            * B_TOPK
                                                                            * (D_V + 64)
                                                                            + (
                                                                                D_V
                                                                                + rope_part
                                                                                * rope_tile
                                                                            )
                                                                            * B_TOPK
                                                                            + row * rope_tile
                                                                        )
                                                                        * BF16_BYTES,
                                                                        "bfloat16",
                                                                    ),
                                                                )
                                                            else:
                                                                rope_tma_dst = _builder_bind(
                                                                    "rope_tma_dst",
                                                                    T.ptr_byte_offset(
                                                                        k_full.ptr_to([0, 0, 0]),
                                                                        (
                                                                            rs_buf.stage
                                                                            * B_TOPK
                                                                            * D_V
                                                                            + d_nope * B_TOPK
                                                                            + row * 64
                                                                        )
                                                                        * BF16_BYTES,
                                                                        "bfloat16",
                                                                    ),
                                                                )
                                                            selected_rope_tensormap = (
                                                                _builder_scalar(
                                                                    "selected_rope_tensormap",
                                                                    T.reinterpret(
                                                                        "uint64",
                                                                        T.address_of(
                                                                            kv_rope_tensormap
                                                                        ),
                                                                    ),
                                                                    dtype="uint64",
                                                                )
                                                            )
                                                            if extra_kv_h is not None:
                                                                T.buffer_store(
                                                                    selected_rope_tensormap.buffer,
                                                                    T.if_then_else(
                                                                        block_idx
                                                                        >= num_orig_blocks,
                                                                        T.reinterpret(
                                                                            "uint64",
                                                                            T.address_of(
                                                                                extra_kv_rope_tensormap
                                                                            ),
                                                                        ),
                                                                        T.reinterpret(
                                                                            "uint64",
                                                                            T.address_of(
                                                                                kv_rope_tensormap
                                                                            ),
                                                                        ),
                                                                    ),
                                                                    [0],
                                                                )
                                                            _builder_emit(
                                                                T.evaluate(
                                                                    T.ptx[_TMA_GATHER4_2D_CACHE](
                                                                        rope_tma_dst,
                                                                        T.reinterpret(
                                                                            T.handle().ty,
                                                                            selected_rope_tensormap,
                                                                        ),
                                                                        T.cast(
                                                                            rope_part * rope_tile,
                                                                            "int32",
                                                                        ),
                                                                        cur_indices[0],
                                                                        cur_indices[1],
                                                                        cur_indices[2],
                                                                        cur_indices[3],
                                                                        T.cuda.cvta_generic_to_shared(
                                                                            bar_rope_ready.ptr_to(
                                                                                [rs_buf.stage]
                                                                            )
                                                                        ),
                                                                        _KV_TMA_CACHE_HINT,
                                                                    )
                                                                )
                                                            )
                                                        T.buffer_store(
                                                            cur_indices, next_indices[0], [0]
                                                        )
                                                        T.buffer_store(
                                                            cur_indices, next_indices[1], [1]
                                                        )
                                                        T.buffer_store(
                                                            cur_indices, next_indices[2], [2]
                                                        )
                                                        T.buffer_store(
                                                            cur_indices, next_indices[3], [3]
                                                        )
                                                    _builder_emit(
                                                        bar_rope_ready.arrive(
                                                            rs_buf.stage,
                                                            tx_count=B_TOPK * 64 * BF16_BYTES,
                                                        )
                                                    )
                                                    _builder_emit(
                                                        bar_valid_free.arrive(rs_index.stage)
                                                    )
                                                    _builder_emit(rs_buf.advance())
                                                    _builder_emit(rs_index.advance())
                                            elif role == 7:
                                                indices_base = _builder_bind(
                                                    "indices_base",
                                                    batch_idx * stride_indices_b
                                                    + s_q_idx * stride_indices_s_q,
                                                )
                                                extra_indices_base = _builder_bind(
                                                    "extra_indices_base",
                                                    batch_idx * stride_extra_indices_b
                                                    + s_q_idx * stride_extra_indices_s_q,
                                                )

                                                def process_index_block(
                                                    cur_block, is_extra: T.constexpr
                                                ):
                                                    abs_pos = _builder_bind(
                                                        "abs_pos",
                                                        T.if_then_else(
                                                            is_extra,
                                                            (cur_block - num_orig_blocks) * B_TOPK
                                                            + lane_idx * 2,
                                                            cur_block * B_TOPK + lane_idx * 2,
                                                        ),
                                                    )
                                                    cur_page_size = _builder_bind(
                                                        "cur_page_size",
                                                        T.if_then_else(
                                                            is_extra,
                                                            extra_page_block_size,
                                                            page_block_size,
                                                        ),
                                                    )
                                                    cur_block_stride = _builder_bind(
                                                        "cur_block_stride",
                                                        T.if_then_else(
                                                            is_extra,
                                                            stride_extra_kv_block,
                                                            stride_kv_block,
                                                        ),
                                                    )
                                                    cur_row_stride = _builder_bind(
                                                        "cur_row_stride",
                                                        T.if_then_else(
                                                            is_extra,
                                                            stride_extra_kv_row,
                                                            stride_kv_row,
                                                        ),
                                                    )
                                                    cur_length = _builder_bind(
                                                        "cur_length",
                                                        T.if_then_else(
                                                            is_extra, extra_topk_len, topk_len
                                                        ),
                                                    )
                                                    cur_k_scales_ptr_u64 = _builder_bind(
                                                        "cur_k_scales_ptr_u64",
                                                        T.if_then_else(
                                                            is_extra,
                                                            extra_k_scales_ptr_u64,
                                                            k_scales_ptr_u64,
                                                        ),
                                                    )
                                                    cur_tma_coords_step_per_block = _builder_bind(
                                                        "cur_tma_coords_step_per_block",
                                                        T.if_then_else(
                                                            is_extra,
                                                            tma_coords_step_per_extra_block,
                                                            tma_coords_step_per_block,
                                                        ),
                                                    )
                                                    pair_indices = _builder_assign(
                                                        "pair_indices", T.alloc_local((2,), "int32")
                                                    )
                                                    pair_index_words = _builder_assign(
                                                        "pair_index_words",
                                                        pair_indices.view("uint32"),
                                                    )
                                                    if is_extra:
                                                        _builder_emit(
                                                            T.ptx.ld.global_.nc.v2.u32(
                                                                pair_index_words[0],
                                                                pair_index_words[1],
                                                                extra_indices.view("uint32").ptr_to(
                                                                    [extra_indices_base + abs_pos]
                                                                ),
                                                            )
                                                        )
                                                    else:
                                                        _builder_emit(
                                                            T.ptx.ld.global_.nc.v2.u32(
                                                                pair_index_words[0],
                                                                pair_index_words[1],
                                                                indices.view("uint32").ptr_to(
                                                                    [indices_base + abs_pos]
                                                                ),
                                                            )
                                                        )
                                                    _builder_emit(
                                                        bar_valid_free.wait(
                                                            rs_index.stage, rs_index.phase ^ 1
                                                        )
                                                    )
                                                    coords = _builder_assign(
                                                        "coords", T.alloc_local((2,), "int32")
                                                    )
                                                    cache_blocks = _builder_assign(
                                                        "cache_blocks",
                                                        T.alloc_local((2,), "uint32"),
                                                    )
                                                    indices_in_block = _builder_assign(
                                                        "indices_in_block",
                                                        T.alloc_local((2,), "uint32"),
                                                    )
                                                    scale_words = _builder_assign(
                                                        "scale_words", T.alloc_local((2,), "uint64")
                                                    )
                                                    pair_token_valid = _builder_assign(
                                                        "pair_token_valid",
                                                        T.alloc_local((2,), "bool"),
                                                    )
                                                    scale_f32 = _builder_assign(
                                                        "scale_f32",
                                                        T.alloc_local((2, 4), "float32"),
                                                    )
                                                    scale_byte_offsets = _builder_assign(
                                                        "scale_byte_offsets",
                                                        T.alloc_local((2,), "uint64"),
                                                    )

                                                    def load_token_scales(
                                                        pair_i: T.constexpr,
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
                                                            T.buffer_store(
                                                                byte_offsets,
                                                                T.if_then_else(
                                                                    token_valid,
                                                                    T.cast(cache_block, "uint64")
                                                                    * T.cast(block_stride, "int64")
                                                                    + T.cast(
                                                                        index_in_block, "uint64"
                                                                    )
                                                                    * T.cast(row_stride, "int64"),
                                                                    T.uint64(0),
                                                                ),
                                                                [pair_i],
                                                            )
                                                            _builder_emit(
                                                                T.cuda.ldg(
                                                                    T.reinterpret(
                                                                        PointerType(
                                                                            PrimType("float32")
                                                                        ),
                                                                        scales_ptr_u64
                                                                        + byte_offsets[pair_i],
                                                                    ),
                                                                    "float32",
                                                                    dst=(
                                                                        values.ptr_to([pair_i, 0]),
                                                                        values.ptr_to([pair_i, 1]),
                                                                        values.ptr_to([pair_i, 2]),
                                                                        values.ptr_to([pair_i, 3]),
                                                                    ),
                                                                    vec="v4",
                                                                )
                                                            )
                                                        else:
                                                            T.buffer_store(
                                                                byte_offsets,
                                                                T.cast(cache_block, "uint64")
                                                                * T.cast(block_stride, "int64")
                                                                + T.cast(index_in_block, "uint64")
                                                                * 8,
                                                                [pair_i],
                                                            )
                                                            T.buffer_store(
                                                                words,
                                                                T.if_then_else(
                                                                    token_valid,
                                                                    T.cuda.ldg(
                                                                        T.reinterpret(
                                                                            PointerType(
                                                                                PrimType("uint64")
                                                                            ),
                                                                            scales_ptr_u64
                                                                            + byte_offsets[pair_i],
                                                                        ),
                                                                        "uint64",
                                                                    ),
                                                                    T.uint64(0),
                                                                ),
                                                                [pair_i],
                                                            )

                                                    valid_mask = _builder_scalar(
                                                        "valid_mask", T.int8(0), dtype="int8"
                                                    )
                                                    with T.unroll(2) as pair_i:
                                                        index_u32 = _builder_bind(
                                                            "index_u32",
                                                            T.cast(pair_indices[pair_i], "uint32"),
                                                        )
                                                        T.buffer_store(
                                                            cache_blocks,
                                                            index_u32
                                                            // T.cast(cur_page_size, "uint32"),
                                                            [pair_i],
                                                        )
                                                        T.buffer_store(
                                                            indices_in_block,
                                                            index_u32
                                                            % T.cast(cur_page_size, "uint32"),
                                                            [pair_i],
                                                        )
                                                        token_valid = _builder_bind(
                                                            "token_valid",
                                                            T.And(
                                                                pair_indices[pair_i] != -1,
                                                                abs_pos + pair_i < cur_length,
                                                            ),
                                                        )
                                                        T.buffer_store(
                                                            pair_token_valid, token_valid, [pair_i]
                                                        )
                                                        T.buffer_store(
                                                            valid_mask.buffer,
                                                            T.cast(
                                                                T.bitwise_or(
                                                                    T.cast(valid_mask, "int32"),
                                                                    T.shift_left(
                                                                        T.cast(
                                                                            token_valid, "int32"
                                                                        ),
                                                                        T.cast(pair_i, "int32"),
                                                                    ),
                                                                ),
                                                                "int8",
                                                            ),
                                                            [0],
                                                        )
                                                        T.buffer_store(
                                                            coords,
                                                            T.if_then_else(
                                                                pair_token_valid[pair_i],
                                                                T.cast(
                                                                    cache_blocks[pair_i], "int32"
                                                                )
                                                                * cur_tma_coords_step_per_block
                                                                + T.cast(
                                                                    indices_in_block[pair_i],
                                                                    "int32",
                                                                )
                                                                * tma_coords_step_per_token,
                                                                -1,
                                                            ),
                                                            [pair_i],
                                                        )
                                                        _builder_emit(
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
                                                        )
                                                    if is_v32:
                                                        with T.unroll(2) as pair_i:
                                                            lo = _builder_assign(
                                                                "lo", T.local_scalar("uint16")
                                                            )
                                                            _builder_emit(
                                                                T.ptx.cvt.rz.ue8m0x2.f32(
                                                                    lo,
                                                                    scale_f32[pair_i, 1],
                                                                    scale_f32[pair_i, 0],
                                                                )
                                                            )
                                                            hi = _builder_assign(
                                                                "hi", T.local_scalar("uint16")
                                                            )
                                                            _builder_emit(
                                                                T.ptx.cvt.rz.ue8m0x2.f32(
                                                                    hi,
                                                                    scale_f32[pair_i, 3],
                                                                    scale_f32[pair_i, 2],
                                                                )
                                                            )
                                                            packed_scale = _builder_bind(
                                                                "packed_scale",
                                                                T.bitwise_or(
                                                                    T.cast(lo, "uint32"),
                                                                    T.shift_left(
                                                                        T.cast(hi, "uint32"),
                                                                        T.uint32(16),
                                                                    ),
                                                                ),
                                                            )
                                                            T.buffer_store(
                                                                scale_words,
                                                                T.if_then_else(
                                                                    pair_token_valid[pair_i],
                                                                    T.cast(packed_scale, "uint64"),
                                                                    T.uint64(0),
                                                                ),
                                                                [pair_i],
                                                            )
                                                    T.buffer_store(
                                                        valid_mask.buffer,
                                                        T.cast(
                                                            T.shift_left(
                                                                T.cast(valid_mask, "int32"),
                                                                T.cast(lane_idx % 4 * 2, "int32"),
                                                            ),
                                                            "int8",
                                                        ),
                                                        [0],
                                                    )
                                                    T.buffer_store(
                                                        valid_mask.buffer,
                                                        T.cast(
                                                            T.bitwise_or(
                                                                T.cast(valid_mask, "int32"),
                                                                T.cuda.__shfl_xor_sync(
                                                                    T.uint32(4294967295),
                                                                    T.cast(valid_mask, "int32"),
                                                                    1,
                                                                    32,
                                                                ),
                                                            ),
                                                            "int8",
                                                        ),
                                                        [0],
                                                    )
                                                    T.buffer_store(
                                                        valid_mask.buffer,
                                                        T.cast(
                                                            T.bitwise_or(
                                                                T.cast(valid_mask, "int32"),
                                                                T.cuda.__shfl_xor_sync(
                                                                    T.uint32(4294967295),
                                                                    T.cast(valid_mask, "int32"),
                                                                    2,
                                                                    32,
                                                                ),
                                                            ),
                                                            "int8",
                                                        ),
                                                        [0],
                                                    )
                                                    if is_v32:
                                                        _builder_emit(
                                                            T.ptx.st.shared.u64(
                                                                scales_e8m0.view("uint64").ptr_to(
                                                                    [rs_index.stage, lane_idx]
                                                                ),
                                                                T.bitwise_or(
                                                                    scale_words[0],
                                                                    T.shift_left(
                                                                        scale_words[1], T.uint64(32)
                                                                    ),
                                                                ),
                                                            )
                                                        )
                                                    else:
                                                        scale_word_bits = _builder_assign(
                                                            "scale_word_bits",
                                                            scale_words.view("uint32"),
                                                        )
                                                        _builder_emit(
                                                            T.ptx.st.shared.v4.u32(
                                                                scales_e8m0.view("uint32").ptr_to(
                                                                    [rs_index.stage, lane_idx * 4]
                                                                ),
                                                                scale_word_bits[0],
                                                                scale_word_bits[1],
                                                                scale_word_bits[2],
                                                                scale_word_bits[3],
                                                            )
                                                        )
                                                    coord_words = _builder_assign(
                                                        "coord_words", coords.view("uint32")
                                                    )
                                                    _builder_emit(
                                                        T.ptx.st.shared.v2.u32(
                                                            tma_coord.view("uint32").ptr_to(
                                                                [rs_index.stage, lane_idx * 2]
                                                            ),
                                                            coord_words[0],
                                                            coord_words[1],
                                                        )
                                                    )
                                                    with T.If(lane_idx % 4 == 0):
                                                        with T.Then():
                                                            _builder_emit(
                                                                T.ptx.st.shared.b8(
                                                                    is_token_valid.ptr_to(
                                                                        [
                                                                            rs_index.stage,
                                                                            lane_idx // 4,
                                                                        ]
                                                                    ),
                                                                    T.reinterpret(
                                                                        "uint8", valid_mask
                                                                    ),
                                                                )
                                                            )
                                                    _builder_emit(
                                                        bar_valid_ready.arrive(rs_index.stage)
                                                    )
                                                    _builder_emit(rs_buf.advance())
                                                    _builder_emit(rs_index.advance())

                                                with T.serial(
                                                    start_block,
                                                    T.min(num_orig_blocks, end_block),
                                                    unroll=False,
                                                ) as block_idx:
                                                    _builder_emit(
                                                        process_index_block(block_idx, False)
                                                    )
                                                if (
                                                    extra_kv_h is not None
                                                    and extra_indices_h is not None
                                                ):
                                                    with T.serial(
                                                        T.max(start_block, num_orig_blocks),
                                                        end_block,
                                                        unroll=False,
                                                    ) as block_idx:
                                                        _builder_emit(
                                                            process_index_block(block_idx, True)
                                                        )
                                            _builder_emit(
                                                T.ptx.barrier.sync(
                                                    T.uint32(BAR_EVERYONE_SYNC),
                                                    T.uint32(NUM_THREADS),
                                                )
                                            )
                                            T.buffer_store(
                                                batch_bar_phase.buffer, batch_bar_phase ^ 1, [0]
                                            )

                            selected_wg1_role = _builder_scalar(
                                "selected_wg1_role", -1, dtype="int32"
                            )
                            with T.If(wg1_warp_idx == 4):
                                with T.Then():
                                    with T.If(T.cuda.elect_sync() != T.uint32(0)):
                                        with T.Then():
                                            T.buffer_store(selected_wg1_role.buffer, 4, [0])
                                with T.Else():
                                    with T.If(wg1_warp_idx == 5):
                                        with T.Then():
                                            with T.If(T.cuda.elect_sync() != T.uint32(0)):
                                                with T.Then():
                                                    T.buffer_store(selected_wg1_role.buffer, 5, [0])
                                        with T.Else():
                                            with T.If(wg1_warp_idx == 6):
                                                with T.Then():
                                                    with T.If(T.cuda.elect_sync() != T.uint32(0)):
                                                        with T.Then():
                                                            T.buffer_store(
                                                                selected_wg1_role.buffer, 6, [0]
                                                            )
                                                with T.Else():
                                                    with T.If(wg1_warp_idx == 7):
                                                        with T.Then():
                                                            T.buffer_store(
                                                                selected_wg1_role.buffer, 7, [0]
                                                            )
                            with T.If(selected_wg1_role == 4):
                                with T.Then():
                                    _builder_emit(run_wg1_role(4))
                                with T.Else():
                                    with T.If(selected_wg1_role == 5):
                                        with T.Then():
                                            _builder_emit(run_wg1_role(5))
                                        with T.Else():
                                            with T.If(selected_wg1_role == 6):
                                                with T.Then():
                                                    _builder_emit(run_wg1_role(6))
                                                with T.Else():
                                                    with T.If(selected_wg1_role == 7):
                                                        with T.Then():
                                                            _builder_emit(run_wg1_role(7))
                                                        with T.Else():
                                                            _builder_emit(run_wg1_role(-1))
                        with T.Else():
                            _builder_emit(T.ptx.setmaxnreg.inc.sync.aligned.u32(208))
                            rs_buf = _builder_assign("rs_buf", PipelineState(NUM_BUFS, phase=0))
                            rs_index = _builder_assign(
                                "rs_index", PipelineState(NUM_INDEX_BUFS, phase=0)
                            )
                            group_idx = _builder_bind("group_idx", idx_in_warpgroup // 8)
                            idx_in_group = _builder_bind("idx_in_group", idx_in_warpgroup % 8)
                            nope0_base_u64 = _builder_bind(
                                "nope0_base_u64",
                                T.reinterpret(
                                    "uint64", k_full.ptr_to([0, group_idx, idx_in_group * 8])
                                ),
                            )
                            nope1_base_u64 = _builder_bind(
                                "nope1_base_u64",
                                T.reinterpret(
                                    "uint64", k_full.ptr_to([1, group_idx, idx_in_group * 8])
                                ),
                            )
                            raw_nope0_base_u64 = _builder_bind(
                                "raw_nope0_base_u64",
                                T.reinterpret(
                                    "uint64", raw_nope.ptr_to([0, group_idx, idx_in_group])
                                ),
                            )
                            raw_nope1_base_u64 = _builder_bind(
                                "raw_nope1_base_u64",
                                T.reinterpret(
                                    "uint64", raw_nope.ptr_to([1, group_idx, idx_in_group])
                                ),
                            )
                            sched_words = _builder_assign(
                                "sched_words", T.alloc_local((4,), "uint64")
                            )
                            _builder_emit(load_scheduler_meta(sched_words))
                            sched_i32 = _builder_assign("sched_i32", sched_words.view("int32"))
                            sched_begin_req = _builder_bind("sched_begin_req", sched_i32[0])
                            sched_end_req = _builder_bind("sched_end_req", sched_i32[1])
                            sched_begin_block = _builder_bind("sched_begin_block", sched_i32[2])
                            sched_end_block = _builder_bind("sched_end_block", sched_i32[3])
                            sched_begin_split = _builder_bind("sched_begin_split", sched_i32[4])
                            sched_first_split = _builder_bind("sched_first_split", sched_i32[5])
                            sched_last_split = _builder_bind("sched_last_split", sched_i32[6])
                            batch_bar_phase = _builder_scalar("batch_bar_phase", 0, dtype="int32")
                            with T.If(sched_begin_req < b):
                                with T.Then():
                                    with T.serial(
                                        sched_begin_req, sched_end_req + 1, unroll=False
                                    ) as batch_idx:
                                        topk_len = _builder_scalar("topk_len", topk, dtype="int32")
                                        if topk_length_h is not None:
                                            T.buffer_store(
                                                topk_len.buffer,
                                                T.cuda.ldg(
                                                    topk_length.ptr_to([batch_idx]), "int32"
                                                ),
                                                [0],
                                            )
                                        orig_topk_padded = _builder_bind(
                                            "orig_topk_padded",
                                            T.max(
                                                (topk_len + B_TOPK - 1) // B_TOPK * B_TOPK, B_TOPK
                                            ),
                                        )
                                        extra_topk_len = _builder_scalar(
                                            "extra_topk_len", extra_topk, dtype="int32"
                                        )
                                        if extra_topk_length_h is not None:
                                            T.buffer_store(
                                                extra_topk_len.buffer,
                                                T.cuda.ldg(
                                                    extra_topk_length.ptr_to([batch_idx]), "int32"
                                                ),
                                                [0],
                                            )
                                        total_topk_padded = _builder_bind(
                                            "total_topk_padded",
                                            orig_topk_padded
                                            + (extra_topk_len + B_TOPK - 1) // B_TOPK * B_TOPK,
                                        )
                                        start_block = _builder_bind(
                                            "start_block",
                                            T.if_then_else(
                                                batch_idx == sched_begin_req, sched_begin_block, 0
                                            ),
                                        )
                                        end_block = _builder_bind(
                                            "end_block",
                                            T.if_then_else(
                                                batch_idx == sched_end_req,
                                                sched_end_block,
                                                total_topk_padded // B_TOPK,
                                            ),
                                        )
                                        is_split = _builder_scalar(
                                            "is_split",
                                            T.cast(
                                                T.if_then_else(
                                                    batch_idx == sched_begin_req,
                                                    sched_first_split,
                                                    T.if_then_else(
                                                        batch_idx == sched_end_req,
                                                        sched_last_split,
                                                        0,
                                                    ),
                                                ),
                                                "bool",
                                            ),
                                            dtype="bool",
                                        )
                                        is_no_split = _builder_scalar(
                                            "is_no_split", T.Not(is_split), dtype="bool"
                                        )
                                        n_split_idx = _builder_bind(
                                            "n_split_idx",
                                            T.if_then_else(
                                                batch_idx == sched_begin_req,
                                                T.cuda.ldg(num_splits.ptr_to([batch_idx]), "int32")
                                                + sched_begin_split,
                                                T.cuda.ldg(num_splits.ptr_to([batch_idx]), "int32"),
                                            ),
                                        )
                                        num_orig_blocks = _builder_bind(
                                            "num_orig_blocks", orig_topk_padded // B_TOPK
                                        )
                                        is_last_batch = _builder_scalar(
                                            "is_last_batch",
                                            batch_idx == sched_end_req,
                                            dtype="bool",
                                        )
                                        _builder_emit(bar_q_utccp.wait(0, batch_bar_phase))
                                        with T.serial(
                                            start_block, end_block, unroll=False
                                        ) as block_idx:
                                            _builder_emit(
                                                bar_valid_ready.wait(rs_index.stage, rs_index.phase)
                                            )
                                            _builder_emit(
                                                bar_raw_ready.wait(rs_buf.stage, rs_buf.phase)
                                            )
                                            _builder_emit(
                                                bar_sv_done.wait(rs_buf.stage, rs_buf.phase ^ 1)
                                            )
                                            _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                            cur_nope_base_u64 = _builder_bind(
                                                "cur_nope_base_u64",
                                                T.if_then_else(
                                                    rs_buf.stage == 0,
                                                    nope0_base_u64,
                                                    nope1_base_u64,
                                                ),
                                            )
                                            cur_raw_nope_base_u64 = _builder_bind(
                                                "cur_raw_nope_base_u64",
                                                T.if_then_else(
                                                    rs_buf.stage == 0,
                                                    raw_nope0_base_u64,
                                                    raw_nope1_base_u64,
                                                ),
                                            )
                                            cur_nope_base_uint_addr = _builder_bind(
                                                "cur_nope_base_uint_addr",
                                                T.cuda.cvta_generic_to_shared(
                                                    T.reinterpret(
                                                        PointerType(PrimType("bfloat16")),
                                                        cur_nope_base_u64,
                                                    )
                                                ),
                                            )
                                            cur_raw_nope_base_uint_addr = _builder_bind(
                                                "cur_raw_nope_base_uint_addr",
                                                T.cuda.cvta_generic_to_shared(
                                                    T.reinterpret(
                                                        PointerType(PrimType("uint64")),
                                                        cur_raw_nope_base_u64,
                                                    )
                                                ),
                                            )
                                            with T.unroll(rows_per_group) as local_row:
                                                row_idx = _builder_bind(
                                                    "row_idx", local_row * (128 // 8) + group_idx
                                                )
                                                scales_bf16_bits = _builder_assign(
                                                    "scales_bf16_bits",
                                                    T.alloc_local((num_scales,), "uint16"),
                                                )
                                                if is_v32:
                                                    packed_scales = _builder_alloc_scalar(
                                                        "packed_scales", "uint32"
                                                    )
                                                    _builder_emit(
                                                        T.ptx.ld.shared.u32(
                                                            packed_scales,
                                                            scales_e8m0.view("uint32").ptr_to(
                                                                [rs_index.stage, row_idx]
                                                            ),
                                                        )
                                                    )
                                                    with T.unroll(2) as scale_pair_idx:
                                                        converted_pair = _builder_assign(
                                                            "converted_pair",
                                                            T.local_scalar("uint32"),
                                                        )
                                                        _builder_emit(
                                                            T.ptx.cvt.rn.bf16x2.ue8m0x2(
                                                                converted_pair,
                                                                T.cast(
                                                                    T.shift_right(
                                                                        packed_scales,
                                                                        T.cast(
                                                                            scale_pair_idx * 16,
                                                                            "uint32",
                                                                        ),
                                                                    ),
                                                                    "uint16",
                                                                ),
                                                            )
                                                        )
                                                        T.buffer_store(
                                                            scales_bf16_bits,
                                                            T.cast(converted_pair, "uint16"),
                                                            [scale_pair_idx * 2],
                                                        )
                                                        T.buffer_store(
                                                            scales_bf16_bits,
                                                            T.cast(
                                                                T.shift_right(
                                                                    converted_pair, T.uint32(16)
                                                                ),
                                                                "uint16",
                                                            ),
                                                            [scale_pair_idx * 2 + 1],
                                                        )
                                                else:
                                                    packed_scales = _builder_alloc_scalar(
                                                        "packed_scales", "uint64"
                                                    )
                                                    _builder_emit(
                                                        T.ptx.ld.shared.u64(
                                                            packed_scales,
                                                            scales_e8m0.view("uint64").ptr_to(
                                                                [rs_index.stage, row_idx]
                                                            ),
                                                        )
                                                    )
                                                    with T.unroll(4) as scale_pair_idx:
                                                        converted_pair = _builder_assign(
                                                            "converted_pair",
                                                            T.local_scalar("uint32"),
                                                        )
                                                        _builder_emit(
                                                            T.ptx.cvt.rn.bf16x2.ue8m0x2(
                                                                converted_pair,
                                                                T.cast(
                                                                    T.shift_right(
                                                                        packed_scales,
                                                                        T.cast(
                                                                            scale_pair_idx * 16,
                                                                            "uint64",
                                                                        ),
                                                                    ),
                                                                    "uint16",
                                                                ),
                                                            )
                                                        )
                                                        T.buffer_store(
                                                            scales_bf16_bits,
                                                            T.cast(converted_pair, "uint16"),
                                                            [scale_pair_idx * 2],
                                                        )
                                                        T.buffer_store(
                                                            scales_bf16_bits,
                                                            T.cast(
                                                                T.shift_right(
                                                                    converted_pair, T.uint32(16)
                                                                ),
                                                                "uint16",
                                                            ),
                                                            [scale_pair_idx * 2 + 1],
                                                        )
                                                cur_raw_fp8x8 = _builder_alloc_scalar(
                                                    "cur_raw_fp8x8", "uint64"
                                                )
                                                _builder_emit(
                                                    T.ptx.ld.shared.u64(
                                                        cur_raw_fp8x8,
                                                        cur_raw_nope_base_uint_addr
                                                        + T.cast(
                                                            local_row * (128 // 8) * d_nope,
                                                            "uint32",
                                                        ),
                                                    )
                                                )
                                                with T.unroll(cols_per_group) as local_col:
                                                    raw_fp8x8 = _builder_bind(
                                                        "raw_fp8x8", cur_raw_fp8x8
                                                    )
                                                    with T.If(local_col + 1 < cols_per_group):
                                                        with T.Then():
                                                            _builder_emit(
                                                                T.ptx.ld.shared.u64(
                                                                    cur_raw_fp8x8,
                                                                    cur_raw_nope_base_uint_addr
                                                                    + T.cast(
                                                                        local_row
                                                                        * (128 // 8)
                                                                        * d_nope
                                                                        + (local_col + 1) * (8 * 8),
                                                                        "uint32",
                                                                    ),
                                                                )
                                                            )
                                                    scale_idx = _builder_bind(
                                                        "scale_idx",
                                                        local_col // (cols_per_group // 4)
                                                        if is_v32
                                                        else local_col,
                                                    )
                                                    _builder_emit(
                                                        dequant_st128(
                                                            cur_nope_base_uint_addr
                                                            + T.cast(
                                                                BF16_BYTES
                                                                * (
                                                                    local_row * (128 // 8) * 64
                                                                    + local_col * B_TOPK * 64
                                                                ),
                                                                "uint32",
                                                            ),
                                                            raw_fp8x8,
                                                            scales_bf16_bits[scale_idx],
                                                        )
                                                    )
                                            _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                            _builder_emit(bar_nope_ready.arrive(rs_buf.stage))
                                            _builder_emit(bar_raw_free.arrive(rs_buf.stage))
                                            _builder_emit(bar_valid_free.arrive(rs_index.stage))
                                            _builder_emit(rs_buf.advance())
                                            _builder_emit(rs_index.advance())
                                        _builder_emit(
                                            T.ptx.barrier.sync(
                                                T.uint32(BAR_EVERYONE_SYNC), T.uint32(NUM_THREADS)
                                            )
                                        )
                                        T.buffer_store(
                                            batch_bar_phase.buffer, batch_bar_phase ^ 1, [0]
                                        )
    return builder.get()


def _build_sparse_decode_head64_combine_kernel(
    *, max_splits: T.constexpr, use_pdl: T.constexpr, _have_attn_sink_h=True
):
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("_sparse_decode_head64_combine_kernel")
            lse_h = T.arg("lse_h", T.handle())
            out_h = T.arg("out_h", T.handle())
            lse_accum_h = T.arg("lse_accum_h", T.handle())
            o_accum_h = T.arg("o_accum_h", T.handle())
            num_splits_h = T.arg("num_splits_h", T.handle())
            if _have_attn_sink_h:
                attn_sink_h = T.arg("attn_sink_h", T.handle())
            else:
                attn_sink_h = None
            stride_lse_b = T.arg("stride_lse_b", T.int32())
            stride_lse_s_q = T.arg("stride_lse_s_q", T.int32())
            stride_o_b = T.arg("stride_o_b", T.int32())
            stride_o_s_q = T.arg("stride_o_s_q", T.int32())
            stride_o_h_q = T.arg("stride_o_h_q", T.int32())
            stride_lse_accum_split = T.arg("stride_lse_accum_split", T.int32())
            stride_lse_accum_s_q = T.arg("stride_lse_accum_s_q", T.int32())
            stride_o_accum_split = T.arg("stride_o_accum_split", T.int32())
            stride_o_accum_s_q = T.arg("stride_o_accum_s_q", T.int32())
            stride_o_accum_h_q = T.arg("stride_o_accum_h_q", T.int32())
            b = T.arg("b", T.int32())
            s_q = T.arg("s_q", T.int32())
            h_q = T.arg("h_q", T.int32())
            d_v = T.arg("d_v", T.int32())
            num_sm_parts = T.arg("num_sm_parts", T.int32())
            lse = _builder_assign(
                "lse", T.match_buffer(lse_h, (b * s_q * h_q,), "float32", scope="global")
            )
            out = _builder_assign(
                "out", T.match_buffer(out_h, (b * s_q * h_q * d_v,), "bfloat16", scope="global")
            )
            lse_accum = _builder_assign(
                "lse_accum",
                T.match_buffer(
                    lse_accum_h, ((b + num_sm_parts) * s_q * h_q,), "float32", scope="global"
                ),
            )
            o_accum = _builder_assign(
                "o_accum",
                T.match_buffer(
                    o_accum_h, ((b + num_sm_parts) * s_q * h_q * d_v,), "float32", scope="global"
                ),
            )
            num_splits = _builder_assign(
                "num_splits", T.match_buffer(num_splits_h, (b + 1,), "int32", scope="global")
            )
            if attn_sink_h is not None:
                attn_sink = _builder_assign(
                    "attn_sink", T.match_buffer(attn_sink_h, (h_q,), "float32", scope="global")
                )
            _builder_emit(T.device_entry())
            _builder_values_41 = T.cta_id([b * s_q, 1, (h_q + 7) // 8])
            batch_s_q_idx_expr, _, h_block_idx_expr = _builder_values_41
            IRBuilder.name("_", _)
            IRBuilder.name("batch_s_q_idx_expr", batch_s_q_idx_expr)
            IRBuilder.name("h_block_idx_expr", h_block_idx_expr)
            thread_idx_expr = _builder_assign("thread_idx_expr", T.thread_id([8 * 32]))
            batch_s_q_idx = _builder_scalar("batch_s_q_idx", batch_s_q_idx_expr, dtype="int32")
            h_block_idx = _builder_scalar("h_block_idx", h_block_idx_expr, dtype="int32")
            thread_idx = _builder_scalar("thread_idx", thread_idx_expr, dtype="int32")
            warp_idx = _builder_scalar("warp_idx", thread_idx // 32, dtype="int32")
            lane_idx = _builder_scalar("lane_idx", thread_idx % 32, dtype="int32")
            batch_idx = _builder_scalar("batch_idx", batch_s_q_idx // s_q, dtype="int32")
            query_idx = _builder_scalar("query_idx", batch_s_q_idx - batch_idx * s_q, dtype="int32")
            h_block_base = _builder_scalar("h_block_base", h_block_idx * 8, dtype="int32")
            head_idx = _builder_scalar("head_idx", h_block_base + warp_idx, dtype="int32")
            num_valid_heads = _builder_scalar(
                "num_valid_heads", T.min(8, h_q - h_block_base), dtype="int32"
            )
            with T.If(warp_idx >= num_valid_heads):
                with T.Then():
                    T.Return(0)
            start_split = _builder_scalar(
                "start_split", T.cuda.ldg(num_splits.ptr_to([batch_idx]), "int32"), dtype="int32"
            )
            end_split = _builder_scalar(
                "end_split", T.cuda.ldg(num_splits.ptr_to([batch_idx + 1]), "int32"), dtype="int32"
            )
            my_num_splits = _builder_scalar("my_num_splits", end_split - start_split, dtype="int32")
            with T.If(my_num_splits == 1):
                with T.Then():
                    T.Return(0)
            _builder_emit(T.cuda.trap_when_assert_failed(my_num_splits <= max_splits))
            g_lse_accum_offset = _builder_scalar(
                "g_lse_accum_offset",
                start_split * stride_lse_accum_split
                + query_idx * stride_lse_accum_s_q
                + h_block_base,
                dtype="int32",
            )
            g_lse_offset = _builder_scalar(
                "g_lse_offset",
                batch_idx * stride_lse_b + query_idx * stride_lse_s_q + h_block_base,
                dtype="int32",
            )
            g_lse_accum = _builder_assign(
                "g_lse_accum",
                T.decl_buffer(
                    (max_splits, 8),
                    "float32",
                    data=lse_accum.data,
                    scope="global",
                    elem_offset=g_lse_accum_offset,
                    layout=TileLayout(S[(max_splits, 8) : (stride_lse_accum_split, 1)]),
                ),
            )
            g_lse = _builder_assign(
                "g_lse",
                T.decl_buffer(
                    (8,), "float32", data=lse.data, scope="global", elem_offset=g_lse_offset
                ),
            )
            lse_scales = _builder_assign(
                "lse_scales", T.alloc_buffer((8, max_splits), "float32", scope="shared")
            )
            if use_pdl:
                _builder_emit(T.evaluate(T.ptx.griddepcontrol.wait()))
            oaccum_offset = _builder_scalar(
                "oaccum_offset",
                start_split * stride_o_accum_split
                + query_idx * stride_o_accum_s_q
                + head_idx * stride_o_accum_h_q,
                dtype="int32",
            )
            oaccum_ptr = _builder_assign(
                "oaccum_ptr",
                T.decl_buffer(
                    (num_sm_parts * stride_o_accum_split + D_V,),
                    "float32",
                    data=o_accum.data,
                    scope="global",
                    elem_offset=oaccum_offset,
                ),
            )
            datas = _builder_assign("datas", T.alloc_local((D_V // (32 * 4), 4), "float32"))
            data_words = _builder_assign("data_words", datas.view("uint32"))
            with T.unroll(D_V // (32 * 4)) as elem_i:
                _builder_emit(
                    T.ptx.ld.global_.v4.u32(
                        data_words[elem_i, 0],
                        data_words[elem_i, 1],
                        data_words[elem_i, 2],
                        data_words[elem_i, 3],
                        oaccum_ptr.view("uint32").ptr_to([lane_idx * 4 + elem_i * 128]),
                    )
                )
            local_lse = _builder_assign(
                "local_lse", T.alloc_local(((max_splits + 31) // 32,), "float32")
            )
            with T.unroll((max_splits + 31) // 32) as lse_i:
                split_idx = _builder_bind("split_idx", lse_i * 32 + lane_idx)
                T.buffer_store(local_lse, T.float32(-float("inf")), [lse_i])
                with T.If(split_idx < my_num_splits):
                    with T.Then():
                        _builder_emit(
                            T.ptx.ld.global_.f32(
                                local_lse[lse_i], g_lse_accum.ptr_to([split_idx, warp_idx])
                            )
                        )
            max_lse = _builder_scalar("max_lse", T.float32(-float("inf")), dtype="float32")
            with T.unroll((max_splits + 31) // 32) as lse_i:
                T.buffer_store(max_lse.buffer, T.max(max_lse, local_lse[lse_i]), [0])
            with T.unroll(5) as reduce_i:
                xor_offset = _builder_bind("xor_offset", 16 >> reduce_i)
                T.buffer_store(
                    max_lse.buffer,
                    T.max(
                        max_lse,
                        T.cuda.__shfl_xor_sync(T.uint32(4294967295), max_lse, xor_offset, 32),
                    ),
                    [0],
                )
            T.buffer_store(
                max_lse.buffer,
                T.if_then_else(max_lse == T.float32(-float("inf")), 0.0, max_lse),
                [0],
            )
            sum_lse = _builder_scalar("sum_lse", 0.0, dtype="float32")
            lse_exp = _builder_alloc_scalar("lse_exp", "float32")
            with T.unroll((max_splits + 31) // 32) as lse_i:
                _builder_emit(T.ptx.ex2.approx.ftz.f32(lse_exp, local_lse[lse_i] - max_lse))
                T.buffer_store(sum_lse.buffer, sum_lse + lse_exp, [0])
            with T.unroll(5) as reduce_i:
                xor_offset = _builder_bind("xor_offset", 16 >> reduce_i)
                T.buffer_store(
                    sum_lse.buffer,
                    sum_lse + T.cuda.__shfl_xor_sync(T.uint32(4294967295), sum_lse, xor_offset, 32),
                    [0],
                )
            global_lse = _builder_scalar(
                "global_lse",
                T.if_then_else(
                    T.Or(sum_lse == 0.0, sum_lse == T.float32(-float("inf"))),
                    T.float32(float("inf")),
                    T.log2(sum_lse) + max_lse,
                ),
                dtype="float32",
            )
            with T.If(lane_idx == 0):
                with T.Then():
                    _builder_emit(
                        T.ptx.st.global_.f32(
                            g_lse.ptr_to([warp_idx]), global_lse / T.float32(LOG_2_E)
                        )
                    )
            if attn_sink_h is not None:
                sink = _builder_bind("sink", T.cuda.ldg(attn_sink.ptr_to([head_idx]), "float32"))
                with T.If(global_lse != T.float32(float("inf"))):
                    with T.Then():
                        sink_lse_exp = _builder_alloc_scalar("sink_lse_exp", "float32")
                        _builder_emit(
                            T.ptx.ex2.approx.ftz.f32(sink_lse_exp, sink * LOG_2_E - global_lse)
                        )
                        T.buffer_store(
                            global_lse.buffer, global_lse + T.log2(1.0 + sink_lse_exp), [0]
                        )
                    with T.Else():
                        T.buffer_store(
                            global_lse.buffer,
                            T.if_then_else(
                                sink == T.float32(-float("inf")),
                                T.float32(float("inf")),
                                sink * LOG_2_E,
                            ),
                            [0],
                        )
            with T.unroll((max_splits + 31) // 32) as lse_i:
                split_idx = _builder_bind("split_idx", lse_i * 32 + lane_idx)
                lse_scale_value = _builder_alloc_scalar("lse_scale_value", "float32")
                _builder_emit(
                    T.ptx.ex2.approx.ftz.f32(lse_scale_value, local_lse[lse_i] - global_lse)
                )
                _builder_emit(
                    T.ptx.st.shared.f32(lse_scales.ptr_to([warp_idx, split_idx]), lse_scale_value)
                )
            _builder_emit(T.cuda.warp_sync())
            result = _builder_assign("result", T.alloc_local((D_V // (32 * 4), 4), "float32"))
            with T.unroll(D_V // (32 * 4)) as elem_i:
                with T.unroll(4) as vec_i:
                    T.buffer_store(result, 0.0, [elem_i, vec_i])
            with T.serial(0, my_num_splits, unroll=False) as split_idx:
                lse_scale = _builder_alloc_scalar("lse_scale", "float32")
                _builder_emit(
                    T.ptx.ld.shared.f32(lse_scale, lse_scales.ptr_to([warp_idx, split_idx]))
                )
                with T.unroll(D_V // (32 * 4)) as elem_i:
                    with T.unroll(4) as vec_i:
                        T.buffer_store(
                            result,
                            result[elem_i, vec_i] + lse_scale * datas[elem_i, vec_i],
                            [elem_i, vec_i],
                        )
                    with T.If(split_idx != my_num_splits - 1):
                        with T.Then():
                            _builder_emit(
                                T.ptx.ld.global_.v4.u32(
                                    data_words[elem_i, 0],
                                    data_words[elem_i, 1],
                                    data_words[elem_i, 2],
                                    data_words[elem_i, 3],
                                    oaccum_ptr.view("uint32").ptr_to(
                                        [
                                            (split_idx + 1) * stride_o_accum_split
                                            + lane_idx * 4
                                            + elem_i * 128
                                        ]
                                    ),
                                )
                            )
            out_offset = _builder_scalar(
                "out_offset",
                batch_idx * stride_o_b + query_idx * stride_o_s_q + head_idx * stride_o_h_q,
                dtype="int32",
            )
            o_ptr = _builder_assign(
                "o_ptr",
                T.decl_buffer(
                    (D_V,), "bfloat16", data=out.data, scope="global", elem_offset=out_offset
                ),
            )
            with T.unroll(D_V // (32 * 4)) as elem_i:
                data_converted = _builder_assign("data_converted", T.alloc_local((4,), "bfloat16"))
                T.buffer_store(data_converted, result[elem_i, 0], [0])
                T.buffer_store(data_converted, result[elem_i, 1], [1])
                T.buffer_store(data_converted, result[elem_i, 2], [2])
                T.buffer_store(data_converted, result[elem_i, 3], [3])
                _builder_emit(
                    T.ptx.st.global_.u64(
                        o_ptr.view("uint64").ptr_to([(lane_idx * 4 + elem_i * 128) // 4]),
                        data_converted.view("uint64")[0],
                    )
                )
    return builder.get()


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


@lru_cache(maxsize=64)
def _specialized_main_kernel(
    model_type: ModelType, presence: MainPresenceMask, use_pdl: bool = False
):
    specialization = {
        "_have_topk_length_h": presence[0],
        "_have_attn_sink_h": presence[1],
        "_have_extra_kv_h": presence[2],
        "_have_extra_indices_h": presence[3],
        "_have_extra_topk_length_h": presence[4],
    }
    return _build_kernel(model_type=model_type, use_pdl=use_pdl, **specialization).with_attr(
        "tirx.kernel_launch_params", list(LAUNCH_TAGS)
    )


@lru_cache(maxsize=20)
def _specialized_combine_kernel(max_splits: int, have_attn_sink: bool, use_pdl: bool = False):
    return _build_sparse_decode_head64_combine_kernel(
        max_splits=max_splits, use_pdl=use_pdl, _have_attn_sink_h=have_attn_sink
    ).with_attr(
        "tirx.kernel_launch_params",
        list(COMBINE_PDL_LAUNCH_TAGS if use_pdl else COMBINE_LAUNCH_TAGS),
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


def _tirx_main_args(case: dict[str, Any], start_head_idx: int) -> tuple[Any, ...]:
    cfg: SparseFlashMLADecodeHead64Config = case["config"]
    if start_head_idx % B_H or start_head_idx + B_H > cfg.h_q:
        raise ValueError(f"invalid head64 slice {start_head_idx} for h_q={cfg.h_q}")
    tma_k_stride = 656 if cfg.normalized_model_type is ModelType.V32 else 576

    q_storage_shape = (
        cfg.b,
        case["stride_q_b"] // case["stride_q_s_q"],
        case["stride_q_s_q"] // case["stride_q_h_q"],
        case["stride_q_h_q"],
    )
    q_extent = math.prod(q_storage_shape)
    indices_extent = (
        (cfg.b - 1) * case["stride_indices_b"]
        + (cfg.s_q - 1) * case["stride_indices_s_q"]
        + cfg.topk
    )
    lse_extent = (cfg.b - 1) * case["stride_lse_b"] + (cfg.s_q - 1) * case["stride_lse_s_q"] + B_H
    out_storage_shape = (
        cfg.b,
        case["stride_o_b"] // case["stride_o_s_q"],
        case["stride_o_s_q"] // case["stride_o_h_q"],
        case["stride_o_h_q"],
    )
    out_extent = math.prod(out_storage_shape)
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
    extra_indices_extent = (
        (cfg.b - 1) * case["stride_extra_indices_b"]
        + (cfg.s_q - 1) * case["stride_extra_indices_s_q"]
        + cfg.extra_topk
    )
    args = (
        _flat_storage_alias(
            case["q"], element_offset=start_head_idx * case["stride_q_h_q"], extent=q_extent
        ).view(q_storage_shape),
        case["kv_storage"]
        .view(torch.bfloat16)
        .view(case["shape"]["num_tma_rows"], tma_k_stride // BF16_BYTES),
        _flat_storage_alias(case["indices"], extent=indices_extent),
        case["topk_length"] if cfg.have_topk_length else None,
        (case["attn_sink"][start_head_idx : start_head_idx + B_H] if cfg.have_attn_sink else None),
        _flat_storage_alias(case["lse"], element_offset=start_head_idx, extent=lse_extent),
        _flat_storage_alias(
            case["out"], element_offset=start_head_idx * case["stride_o_h_q"], extent=out_extent
        ).view(out_storage_shape),
        _flat_storage_alias(
            case["lse_accum"], element_offset=start_head_idx, extent=lse_accum_extent
        ),
        _flat_storage_alias(
            case["o_accum"],
            element_offset=start_head_idx * case["stride_o_accum_h_q"],
            extent=o_accum_extent,
        ),
        case["tile_scheduler_metadata"],
        case["num_splits"],
        (
            case["extra_kv_storage"]
            .view(torch.bfloat16)
            .view(case["shape"]["extra_num_tma_rows"], tma_k_stride // BF16_BYTES)
            if case["extra_kv_storage"] is not None
            else None
        ),
        (
            _flat_storage_alias(case["extra_indices"], extent=extra_indices_extent)
            if case["extra_indices"] is not None
            else None
        ),
        case["extra_topk_length"] if cfg.have_extra_topk_length else None,
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
    presence = _main_presence_mask(cfg)
    return _present_runtime_args(args, MAIN_OPTIONAL_ARG_INDICES, presence)


def _tirx_combine_args(case: dict[str, Any]) -> tuple[Any, ...]:
    cfg: SparseFlashMLADecodeHead64Config = case["config"]
    args = (
        case["lse"].reshape(-1),
        case["out"].reshape(-1),
        case["lse_accum"].reshape(-1),
        case["o_accum"].reshape(-1),
        case["num_splits"],
        case["attn_sink"] if cfg.have_attn_sink else None,
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
    presence = (cfg.have_attn_sink,)
    return _present_runtime_args(args, COMBINE_OPTIONAL_ARG_INDICES, presence)


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
