# This file is a TIRx port of code from flash-attention
# (https://github.com/Dao-AILab/flash-attention @ 00756db9), Copyright (c) 2022,
# the respective contributors, as shown by licenses/AUTHORS.flash-attention.txt
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashAttention-4 TIRx kernel and direct NVIDIA IKET profiling entry point.

Run ``python -m tirx_kernels.flashattention.flash_attention4`` to profile the
annotated kernel.  Correctness and ordinary benchmarks remain exposed through
``run_test`` and ``run_bench``.

Upstream source: flash_attn/cute/flash_fwd_sm100.py.
"""

from __future__ import annotations

import argparse
import math
import os
from functools import partial
from typing import Any

import numpy as np
import torch

import tvm
import tvm.testing
from tirx_kernels.runner import bench
from tirx_kernels.flashmla.utils._ir_builder import (
    MBarrier,
    PipelineState,
    SmemDescriptor,
    TCGen05Bar,
    TMABar,
)
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T
from tvm.tirx.cuda import iket
from tvm.tirx.cuda.iket import IketProfiler

M_CLUSTER = 1
N_CLUSTER = 1
IKET_EVENT_NAMES = (
    "correction",
    "epi-ld-tmem",
    "issue-tma-k",
    "issue-tma-q",
    "issue-tma-v",
    "softmax-exp2",
    "softmax-fma",
    "softmax-max",
    "softmax-sum",
    "softmax-tmem-st",
    "tma-store",
    "softmax-baseline",
    "softmax-phase-0",
    "softmax-phase-1",
    "softmax-phase-2",
    "softmax-phase-3",
    "softmax-phase-4",
    "softmax-phase-5",
)
# ex2-emulation ratio per regime (grid-searched under the bench protocol;
# heavier and lighter both measured worse on their respective regimes):
# emulate elements with (i*2 % 16) >= 16 - 2*PAIRS in fragments
# [START, 3). Causal keeps fragment 0; non-causal skips it.
EMU_PAIRS_CAUSAL = 2
EMU_START_CAUSAL = 0
EMU_PAIRS_NC = 2
EMU_START_NC = 1

_TMA_G2S_3D = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
_TMA_G2S_4D = (
    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
_TMA_S2G_3D = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
_TMA_S2G_4D = "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
_MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
_TMEM_LD_16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
_TMEM_LD_32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
_TMEM_ST_16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"


def _smem_desc_add_16B_offset(desc_base, offset):
    # Update the address lane with uint32 wraparound without carrying into the descriptor bits.
    desc_halves = T.reinterpret("uint32x2", desc_base)
    desc_lo = T.Shuffle([desc_halves], [0]) + T.cast(offset, "uint32")
    desc_hi = T.Shuffle([desc_halves], [1])
    return T.reinterpret("uint64", T.Shuffle([desc_lo, desc_hi], [0, 1]))


def _cast_f32x2_f16x2(dst, src, offset):
    dst_words = dst.view("uint32")
    return T.ptx.cvt.rn.f16x2.f32(dst_words[offset // 2], src[offset + 1], src[offset])


def _tmem_load(dst, dst_offset, tmem_col, width):
    chain = _TMEM_LD_16 if width == 16 else _TMEM_LD_32
    return T.ptx[chain](*[dst[dst_offset + i] for i in range(width)], tmem_col)


def _tmem_store(src, src_offset, tmem_col, width=16):
    assert width == 16
    return T.ptx[_TMEM_ST_16](tmem_col, *[src[src_offset + i] for i in range(width)])


def ceildiv(a, b):
    return (a + b - 1) // b


def shl_u32_clamp(val, shift):
    """Left shift with PTX clamping (shift>=32 -> 0). Lets the causal mask keep
    the fused ``max(col_limit - s*CHUNK, 0)`` (one VIADDMNMX, like cutedsl) and
    drop the ``min(.,CHUNK)`` clamp (which adds a VIMNMX above cutedsl's count):
    ``~shl_clamp(0xFFFFFFFF, k)`` is the low-k-bits mask for any k, no min."""
    result = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.shl.b32(result[0], val, shift))
    return result[0]


def combine_int_frac_ex2(x_rounded, frac_ex2):
    x_rounded_i = T.alloc_local((1,), "int32")
    frac_ex_i = T.alloc_local((1,), "int32")
    x_rounded_e = T.alloc_local((1,), "int32")
    out_i = T.alloc_local((1,), "int32")
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.mov.b32(x_rounded_i[0], x_rounded))
    T.evaluate(T.ptx.mov.b32(frac_ex_i[0], frac_ex2))
    T.evaluate(T.ptx.shl.b32(x_rounded_e[0], x_rounded_i[0], T.uint32(23)))
    T.evaluate(T.ptx.add.s32(out_i[0], x_rounded_e[0], frac_ex_i[0]))
    T.evaluate(T.ptx.mov.b32(out[0], out_i[0]))
    return out[0]


def get_n_block_max(m_block_idx, causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE):
    """Maximum KV block index (exclusive) for this Q block."""
    n_block_max = ceildiv(SEQ_LEN_KV, BLK_N)
    if not causal:
        return n_block_max
    m_idx_max = (m_block_idx + 1) * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
    n_idx = m_idx_max + SEQ_LEN_KV - SEQ_LEN_Q
    return T.min(n_block_max, ceildiv(n_idx, BLK_N))


def get_n_block_min_causal_mask(m_block_idx, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE):
    """KV block index where causal masking stops being needed."""
    m_idx_min = m_block_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
    n_idx = m_idx_min + SEQ_LEN_KV - SEQ_LEN_Q
    return T.max(0, n_idx // BLK_N)


def ex2_emulation_2(out, idx, x, y):
    poly_ex2_deg3 = T.meta_var(
        (1.0, 0.6951461434364319, 0.22756439447402954, 0.07711908966302872)
    ).value
    fp32_round_int = T.meta_var(float(2**23 + 2**22)).value
    xy_clamped = _builder_name("xy_clamped", T.alloc_local((2,), "float32"))
    T.buffer_store(xy_clamped, T.max(x, -127.0), [0])
    T.buffer_store(xy_clamped, T.max(y, -127.0), [1])
    packed = _builder_alloc_scalar("packed", "uint64")
    rhs = _builder_alloc_scalar("rhs", "uint64")
    addend = _builder_alloc_scalar("addend", "uint64")
    xy_rounded = _builder_name("xy_rounded", T.alloc_local((2,), "float32"))
    _builder_emit(T.ptx.mov.b64(packed, xy_clamped[0], xy_clamped[1]))
    _builder_emit(T.ptx.mov.b64(rhs, T.float32(fp32_round_int), T.float32(fp32_round_int)))
    _builder_emit(T.ptx.add.rm.ftz.f32x2(packed, packed, rhs))
    _builder_emit(T.ptx.mov.b64(xy_rounded[0], xy_rounded[1], packed))
    xy_rounded_back = _builder_name("xy_rounded_back", T.alloc_local((2,), "float32"))
    _builder_emit(T.ptx.mov.b64(packed, xy_rounded[0], xy_rounded[1]))
    _builder_emit(T.ptx.mov.b64(rhs, T.float32(fp32_round_int), T.float32(fp32_round_int)))
    _builder_emit(T.ptx.sub.rn.ftz.f32x2(packed, packed, rhs))
    _builder_emit(T.ptx.mov.b64(xy_rounded_back[0], xy_rounded_back[1], packed))
    xy_frac = _builder_name("xy_frac", T.alloc_local((2,), "float32"))
    _builder_emit(T.ptx.mov.b64(packed, xy_clamped[0], xy_clamped[1]))
    _builder_emit(T.ptx.mov.b64(rhs, xy_rounded_back[0], xy_rounded_back[1]))
    _builder_emit(T.ptx.sub.rn.ftz.f32x2(packed, packed, rhs))
    _builder_emit(T.ptx.mov.b64(xy_frac[0], xy_frac[1], packed))
    xy_frac_ex2 = _builder_name("xy_frac_ex2", T.alloc_local((2,), "float32"))
    T.buffer_store(xy_frac_ex2, poly_ex2_deg3[3], [0])
    T.buffer_store(xy_frac_ex2, poly_ex2_deg3[3], [1])
    _builder_emit(T.ptx.mov.b64(rhs, xy_frac[0], xy_frac[1]))
    _builder_emit(T.ptx.mov.b64(packed, xy_frac_ex2[0], xy_frac_ex2[1]))
    _builder_emit(T.ptx.mov.b64(addend, T.float32(poly_ex2_deg3[2]), T.float32(poly_ex2_deg3[2])))
    _builder_emit(T.ptx.fma.rz.ftz.f32x2(packed, packed, rhs, addend))
    _builder_emit(T.ptx.mov.b64(xy_frac_ex2[0], xy_frac_ex2[1], packed))
    _builder_emit(T.ptx.mov.b64(packed, xy_frac_ex2[0], xy_frac_ex2[1]))
    _builder_emit(T.ptx.mov.b64(rhs, xy_frac[0], xy_frac[1]))
    _builder_emit(T.ptx.mov.b64(addend, T.float32(poly_ex2_deg3[1]), T.float32(poly_ex2_deg3[1])))
    _builder_emit(T.ptx.fma.rz.ftz.f32x2(packed, packed, rhs, addend))
    _builder_emit(T.ptx.mov.b64(xy_frac_ex2[0], xy_frac_ex2[1], packed))
    _builder_emit(T.ptx.mov.b64(packed, xy_frac_ex2[0], xy_frac_ex2[1]))
    _builder_emit(T.ptx.mov.b64(rhs, xy_frac[0], xy_frac[1]))
    _builder_emit(T.ptx.mov.b64(addend, T.float32(poly_ex2_deg3[0]), T.float32(poly_ex2_deg3[0])))
    _builder_emit(T.ptx.fma.rz.ftz.f32x2(packed, packed, rhs, addend))
    _builder_emit(T.ptx.mov.b64(xy_frac_ex2[0], xy_frac_ex2[1], packed))
    T.buffer_store(out, combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0]), [idx])
    T.buffer_store(out, combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1]), [idx + 1])


def fma_f32x2(values, idx, multiplier, addend_value):
    """Apply the packed f32x2 FMA emitted by the former tile primitive."""
    packed = _builder_alloc_scalar("packed", "uint64")
    rhs = _builder_alloc_scalar("rhs", "uint64")
    addend = _builder_alloc_scalar("addend", "uint64")
    _builder_emit(T.ptx.mov.b64(packed, values[idx], values[idx + 1]))
    _builder_emit(T.ptx.mov.b64(rhs, multiplier, multiplier))
    _builder_emit(T.ptx.mov.b64(addend, addend_value, addend_value))
    _builder_emit(T.ptx.fma.rz.ftz.f32x2(packed, packed, rhs, addend))
    _builder_emit(T.ptx.mov.b64(values[idx], values[idx + 1], packed))


def mul_f32x2(values, idx, multiplier):
    """Apply the packed f32x2 multiply emitted by the former tile primitive."""
    packed = _builder_alloc_scalar("packed", "uint64")
    rhs = _builder_alloc_scalar("rhs", "uint64")
    _builder_emit(T.ptx.mov.b64(packed, values[idx], values[idx + 1]))
    _builder_emit(T.ptx.mov.b64(rhs, multiplier, multiplier))
    _builder_emit(T.ptx.mul.rz.ftz.f32x2(packed, packed, rhs))
    _builder_emit(T.ptx.mov.b64(values[idx], values[idx + 1], packed))


def reduce_max_128(out, values, accum=False):
    """SM100 three-input max tree used by the former tile reduction."""
    temp = _builder_name("temp", T.alloc_local((4,), "float32"))
    with T.unroll(4) as i:
        if accum:
            _builder_if_5_8 = _builder_scope_enter(T.If(T.And(T.bool(True), i == 0)))
            _builder_then_5_8 = _builder_scope_enter(T.Then())
            _builder_emit(T.ptx["max.f32"](temp[i], values[2 * i], values[2 * i + 1], out[0]))
            _builder_scope_exit(_builder_then_5_8)
            _builder_else_5_8 = _builder_scope_enter(T.Else())
            T.buffer_store(temp, T.max(values[2 * i], values[2 * i + 1]), [i])
            _builder_scope_exit(_builder_else_5_8)
            _builder_scope_exit(_builder_if_5_8)
        else:
            T.buffer_store(temp, T.max(values[2 * i], values[2 * i + 1]), [i])
    with T.serial(15) as outer:
        with T.unroll(4) as i:
            _builder_emit(
                T.ptx["max.f32"](
                    temp[i],
                    temp[i],
                    values[8 * (outer + 1) + 2 * i],
                    values[8 * (outer + 1) + 2 * i + 1],
                )
            )
    T.buffer_store(out, T.max(temp[0], temp[1]), [0])
    _builder_emit(T.ptx["max.f32"](out[0], out[0], temp[2], temp[3]))


def reduce_sum_128(out, values, accum=False):
    """Preserve the packed add tree and accumulator insertion order."""
    local_sum = _builder_name("local_sum", T.alloc_local((8,), "float32"))
    packed = _builder_alloc_scalar("packed", "uint64")
    rhs = _builder_alloc_scalar("rhs", "uint64")
    with T.unroll(8) as i:
        if accum:
            _builder_if_7_8 = _builder_scope_enter(T.If(T.And(T.bool(True), i == 0)))
            _builder_then_7_8 = _builder_scope_enter(T.Then())
            T.buffer_store(local_sum, values[i] + out[0], [i])
            _builder_scope_exit(_builder_then_7_8)
            _builder_else_7_8 = _builder_scope_enter(T.Else())
            T.buffer_store(local_sum, values[i], [i])
            _builder_scope_exit(_builder_else_7_8)
            _builder_scope_exit(_builder_if_7_8)
        else:
            T.buffer_store(local_sum, values[i], [i])
    with T.serial(15) as outer:
        with T.unroll(4) as i:
            _builder_emit(T.ptx.mov.b64(packed, local_sum[2 * i], local_sum[2 * i + 1]))
            _builder_emit(
                T.ptx.mov.b64(
                    rhs, values[8 * (outer + 1) + 2 * i], values[8 * (outer + 1) + 2 * i + 1]
                )
            )
            _builder_emit(T.ptx.add.rn.ftz.f32x2(packed, packed, rhs))
            _builder_emit(T.ptx.mov.b64(local_sum[2 * i], local_sum[2 * i + 1], packed))
    _builder_emit(T.ptx.mov.b64(packed, local_sum[0], local_sum[1]))
    _builder_emit(T.ptx.mov.b64(rhs, local_sum[2], local_sum[3]))
    _builder_emit(T.ptx.add.rn.ftz.f32x2(packed, packed, rhs))
    _builder_emit(T.ptx.mov.b64(local_sum[0], local_sum[1], packed))
    _builder_emit(T.ptx.mov.b64(packed, local_sum[4], local_sum[5]))
    _builder_emit(T.ptx.mov.b64(rhs, local_sum[6], local_sum[7]))
    _builder_emit(T.ptx.add.rn.ftz.f32x2(packed, packed, rhs))
    _builder_emit(T.ptx.mov.b64(local_sum[4], local_sum[5], packed))
    _builder_emit(T.ptx.mov.b64(packed, local_sum[0], local_sum[1]))
    _builder_emit(T.ptx.mov.b64(rhs, local_sum[4], local_sum[5]))
    _builder_emit(T.ptx.add.rn.ftz.f32x2(packed, packed, rhs))
    _builder_emit(T.ptx.mov.b64(local_sum[0], local_sum[1], packed))
    T.buffer_store(out, local_sum[0] + local_sum[1], [0])


WG_NUMBER = 4
WARP_NUMBER = 4
NUM_THREADS = 32 * WARP_NUMBER * WG_NUMBER
N_COLS_TMEM = 512
TMEM_PIPE_DEPTH = 2
SMEM_PIPE_DEPTH_Q = 2
SMEM_PIPE_DEPTH_KV = 3
BLK_M = 128
BLK_N = 128
BLK_K = 64
SOFTMAX_LD_CHUNK = 32
SOFTMAX_ST_CHUNK = 32
EPI_TILE = 64
TMEM_EPI_LD_SIZE = 16
USE_S0_S1_BARRIER = False
MMA_M = 128
MMA_N = 128
MMA_K = 16
F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16
a_type_qk = tvm.DataType("float16")
b_type_qk = tvm.DataType("float16")
d_type_qk = tvm.DataType("float32")
a_type_pv = tvm.DataType("float16")
b_type_pv = tvm.DataType("float16")
d_type_pv = tvm.DataType("float32")


@T.meta_class
class Pipeline:
    """Builder-native composition of the canonical full/empty barrier pair."""

    def __init__(
        self,
        pool,
        stages,
        *,
        full,
        empty,
        init_full=1,
        init_empty=1,
        empty_phase_offset=0,
        leader=None,
    ):
        barrier_kinds = {"tma": TMABar, "tcgen05": TCGen05Bar, "mbar": MBarrier}
        self.stages = stages
        self.full = barrier_kinds[full](pool, stages, leader=leader)
        self.full.init(init_full)
        self.empty = barrier_kinds[empty](
            pool, stages, phase_offset=empty_phase_offset, leader=leader
        )
        self.empty.init(init_empty)


@T.meta_class
class FlashAttentionLinearScheduler:
    """Builder-native form of TVM's linear FlashAttention scheduler."""

    def __init__(self, prefix, num_batches, num_heads, num_m_blocks, num_ctas):
        self._prefix = prefix
        self._num_batches = num_batches
        self._num_heads = num_heads
        self._num_m_blocks = num_m_blocks
        self._num_ctas = num_ctas
        self._total_tasks = num_batches * num_heads * num_m_blocks
        self.m_idx = T.local_scalar("int32")
        self.n_idx = T.local_scalar("int32")
        self.linear_idx = T.local_scalar("int32")
        self.batch_idx = T.local_scalar("int32")
        self.head_idx = T.local_scalar("int32")
        self.m_block_idx = T.local_scalar("int32")

    def _update(self, linear_idx):
        head_m_product = self._num_heads * self._num_m_blocks
        T.buffer_store(self.batch_idx.buffer, linear_idx // head_m_product, [0])
        T.buffer_store(self.head_idx.buffer, linear_idx % head_m_product // self._num_m_blocks, [0])
        T.buffer_store(self.m_block_idx.buffer, linear_idx % self._num_m_blocks, [0])

    def init(self, cta_id):
        T.buffer_store(self.linear_idx.buffer, cta_id, [0])
        self._update(cta_id)

    def next_tile(self):
        T.buffer_store(self.linear_idx.buffer, self.linear_idx + self._num_ctas, [0])
        self._update(self.linear_idx)

    def valid(self):
        return self.linear_idx < self._total_tasks


@T.meta_class
class FlashAttentionLPTScheduler:
    """Builder-native form of TVM's causal LPT/L2 FlashAttention scheduler."""

    def __init__(self, prefix, num_batches, num_heads, num_m_blocks, l2_swizzle, num_ctas=None):
        self._prefix = prefix
        self._num_batches = num_batches
        self._num_heads = num_heads
        self._num_m_blocks = num_m_blocks
        self._l2_swizzle = l2_swizzle
        self._num_ctas = num_ctas
        self._total_tasks = num_batches * num_heads * num_m_blocks
        self._num_hb = num_batches * num_heads
        self._l2_major = l2_swizzle * num_m_blocks
        self._num_hb_quotient = self._num_hb // l2_swizzle
        self.m_idx = T.local_scalar("int32")
        self.n_idx = T.local_scalar("int32")
        self.linear_idx = T.local_scalar("int32")
        self.batch_idx = T.local_scalar("int32")
        self.head_idx = T.local_scalar("int32")
        self.m_block_idx = T.local_scalar("int32")

    def _update(self, linear_idx):
        bidhb = _builder_bind("bidhb", linear_idx // self._l2_major)
        l2_mod = _builder_bind("l2_mod", linear_idx % self._l2_major)
        num_hb_remainder = _builder_bind(
            "num_hb_remainder", T.max(self._num_hb % self._l2_swizzle, 1)
        )
        m_block_raw = _builder_bind(
            "m_block_raw",
            T.Select(
                bidhb < self._num_hb_quotient,
                l2_mod // self._l2_swizzle,
                l2_mod // num_hb_remainder,
            ),
        )
        bidhb_residual = _builder_bind(
            "bidhb_residual",
            T.Select(
                bidhb < self._num_hb_quotient, l2_mod % self._l2_swizzle, l2_mod % num_hb_remainder
            ),
        )
        bidhb_actual = _builder_bind("bidhb_actual", bidhb * self._l2_swizzle + bidhb_residual)
        T.buffer_store(self.batch_idx.buffer, bidhb_actual // self._num_heads, [0])
        T.buffer_store(self.head_idx.buffer, bidhb_actual % self._num_heads, [0])
        T.buffer_store(self.m_block_idx.buffer, self._num_m_blocks - 1 - m_block_raw, [0])

    def init(self, cta_id):
        T.buffer_store(self.linear_idx.buffer, cta_id, [0])
        self._update(cta_id)

    def next_tile(self):
        if self._num_ctas is None:
            T.buffer_store(self.linear_idx.buffer, self._total_tasks, [0])
        else:
            T.buffer_store(self.linear_idx.buffer, self.linear_idx + self._num_ctas, [0])
            self._update(self.linear_idx)

    def valid(self):
        return self.linear_idx < self._total_tasks


def _build_kernel(
    *,
    BATCH_SIZE: T.constexpr,
    SEQ_LEN_Q: T.constexpr,
    SEQ_LEN_KV: T.constexpr,
    NUM_QO_HEADS: T.constexpr,
    NUM_KV_HEADS: T.constexpr,
    HEAD_DIM: T.constexpr,
    is_causal: T.constexpr,
    CTA_GROUP: T.constexpr,
    TMEM_PIPE_DEPTH: T.constexpr,
    SMEM_PIPE_DEPTH_KV: T.constexpr,
):
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("_kernel")
            Q = T.arg("Q", T.Buffer((BATCH_SIZE, SEQ_LEN_Q, NUM_QO_HEADS, HEAD_DIM), "float16"))
            K = T.arg("K", T.Buffer((BATCH_SIZE, SEQ_LEN_KV, NUM_KV_HEADS, HEAD_DIM), "float16"))
            V = T.arg("V", T.Buffer((BATCH_SIZE, SEQ_LEN_KV, NUM_KV_HEADS, HEAD_DIM), "float16"))
            O = T.arg("O", T.Buffer((BATCH_SIZE, SEQ_LEN_Q, NUM_QO_HEADS, HEAD_DIM), "float16"))
            GQA_RATIO = T.meta_var(NUM_QO_HEADS // NUM_KV_HEADS).value
            SEQ_Q_PER_TILE = T.meta_var(BLK_M // GQA_RATIO).value
            STATS_BAR_PAIRWISE = T.meta_var(GQA_RATIO == 1).value
            L2_SIZE = T.meta_var(50 * 1024 * 1024).value
            SIZE_ONE_KV_HEAD = T.meta_var(SEQ_LEN_KV * HEAD_DIM * 2 * F16_BYTES).value
            L2_SWIZZLE = T.meta_var(
                1
                if L2_SIZE < SIZE_ONE_KV_HEAD
                else 1 << int(math.log2(L2_SIZE // SIZE_ONE_KV_HEAD))
            ).value
            SSCALE_TOTAL_SIZE = T.meta_var(2 * SMEM_PIPE_DEPTH_Q * BLK_M).value
            with T.Assert(T.bool(TMEM_PIPE_DEPTH * MMA_N <= N_COLS_TMEM), "TMEM columns exceeded"):
                T.evaluate(0)
            num_q_blocks_total = T.meta_var(ceildiv(SEQ_LEN_Q, SEQ_Q_PER_TILE)).value
            num_q_blocks = T.meta_var(ceildiv(num_q_blocks_total, SMEM_PIPE_DEPTH_Q)).value
            num_total_tasks = T.meta_var(BATCH_SIZE * NUM_KV_HEADS * num_q_blocks).value
            Q_tensor_map = _builder_bind(
                "Q_tensor_map", T.tvm_stack_alloca("tensormap", 1), T.TensorMap()
            )
            Q_tensor_map_1 = _builder_bind(
                "Q_tensor_map_1", T.tvm_stack_alloca("tensormap", 1), T.TensorMap()
            )
            K_tensor_map = _builder_bind(
                "K_tensor_map", T.tvm_stack_alloca("tensormap", 1), T.TensorMap()
            )
            K_tensor_map_1 = _builder_bind(
                "K_tensor_map_1", T.tvm_stack_alloca("tensormap", 1), T.TensorMap()
            )
            V_tensor_map = _builder_bind(
                "V_tensor_map", T.tvm_stack_alloca("tensormap", 1), T.TensorMap()
            )
            V_tensor_map_1 = _builder_bind(
                "V_tensor_map_1", T.tvm_stack_alloca("tensormap", 1), T.TensorMap()
            )
            O_tensor_map = _builder_bind(
                "O_tensor_map", T.tvm_stack_alloca("tensormap", 1), T.TensorMap()
            )
            if GQA_RATIO == 1:
                _builder_emit(
                    T.call_packed(
                        "runtime.cuTensorMapEncodeTiled",
                        Q_tensor_map,
                        "float16",
                        3,
                        Q.data,
                        HEAD_DIM // 2,
                        SEQ_LEN_Q,
                        BATCH_SIZE * NUM_QO_HEADS * 2,
                        NUM_QO_HEADS * HEAD_DIM * F16_BYTES,
                        HEAD_DIM,
                        HEAD_DIM // 2,
                        SEQ_Q_PER_TILE,
                        2,
                        1,
                        1,
                        1,
                        0,
                        3,
                        2,
                        0,
                    )
                )
                _builder_emit(
                    T.call_packed(
                        "runtime.cuTensorMapEncodeTiled",
                        Q_tensor_map_1,
                        "float16",
                        3,
                        Q.data,
                        HEAD_DIM // 2,
                        SEQ_LEN_Q,
                        BATCH_SIZE * NUM_QO_HEADS * 2,
                        NUM_QO_HEADS * HEAD_DIM * F16_BYTES,
                        HEAD_DIM,
                        HEAD_DIM // 2,
                        SEQ_Q_PER_TILE,
                        2,
                        1,
                        1,
                        1,
                        0,
                        3,
                        2,
                        0,
                    )
                )
                _builder_emit(
                    T.call_packed(
                        "runtime.cuTensorMapEncodeTiled",
                        O_tensor_map,
                        "float16",
                        3,
                        O.data,
                        HEAD_DIM // 2,
                        SEQ_LEN_Q,
                        BATCH_SIZE * NUM_QO_HEADS * 2,
                        NUM_QO_HEADS * HEAD_DIM * F16_BYTES,
                        HEAD_DIM,
                        HEAD_DIM // 2,
                        SEQ_Q_PER_TILE,
                        2,
                        1,
                        1,
                        1,
                        0,
                        3,
                        2,
                        0,
                    )
                )
            else:
                _builder_emit(
                    T.call_packed(
                        "runtime.cuTensorMapEncodeTiled",
                        Q_tensor_map,
                        "float16",
                        4,
                        Q.data,
                        HEAD_DIM // 2,
                        NUM_QO_HEADS,
                        SEQ_LEN_Q,
                        BATCH_SIZE * 2,
                        HEAD_DIM * F16_BYTES,
                        NUM_QO_HEADS * HEAD_DIM * F16_BYTES,
                        HEAD_DIM,
                        HEAD_DIM // 2,
                        GQA_RATIO,
                        SEQ_Q_PER_TILE,
                        2,
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
                _builder_emit(
                    T.call_packed(
                        "runtime.cuTensorMapEncodeTiled",
                        Q_tensor_map_1,
                        "float16",
                        4,
                        Q.data,
                        HEAD_DIM // 2,
                        NUM_QO_HEADS,
                        SEQ_LEN_Q,
                        BATCH_SIZE * 2,
                        HEAD_DIM * F16_BYTES,
                        NUM_QO_HEADS * HEAD_DIM * F16_BYTES,
                        HEAD_DIM,
                        HEAD_DIM // 2,
                        GQA_RATIO,
                        SEQ_Q_PER_TILE,
                        2,
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
                _builder_emit(
                    T.call_packed(
                        "runtime.cuTensorMapEncodeTiled",
                        O_tensor_map,
                        "float16",
                        4,
                        O.data,
                        HEAD_DIM // 2,
                        NUM_QO_HEADS,
                        SEQ_LEN_Q,
                        BATCH_SIZE * 2,
                        HEAD_DIM * F16_BYTES,
                        NUM_QO_HEADS * HEAD_DIM * F16_BYTES,
                        HEAD_DIM,
                        HEAD_DIM // 2,
                        GQA_RATIO,
                        SEQ_Q_PER_TILE,
                        2,
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
            _builder_emit(
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    K_tensor_map,
                    "float16",
                    3,
                    K.data,
                    HEAD_DIM // 2,
                    SEQ_LEN_KV,
                    BATCH_SIZE * NUM_KV_HEADS * 2,
                    NUM_KV_HEADS * HEAD_DIM * F16_BYTES,
                    HEAD_DIM,
                    HEAD_DIM // 2,
                    BLK_N,
                    2,
                    1,
                    1,
                    1,
                    0,
                    3,
                    2,
                    0,
                )
            )
            _builder_emit(
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    K_tensor_map_1,
                    "float16",
                    3,
                    K.data,
                    HEAD_DIM // 2,
                    SEQ_LEN_KV,
                    BATCH_SIZE * NUM_KV_HEADS * 2,
                    NUM_KV_HEADS * HEAD_DIM * F16_BYTES,
                    HEAD_DIM,
                    HEAD_DIM // 2,
                    BLK_N,
                    2,
                    1,
                    1,
                    1,
                    0,
                    3,
                    2,
                    0,
                )
            )
            _builder_emit(
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    V_tensor_map,
                    "float16",
                    3,
                    V.data,
                    HEAD_DIM // 2,
                    SEQ_LEN_KV,
                    BATCH_SIZE * NUM_KV_HEADS * 2,
                    NUM_KV_HEADS * HEAD_DIM * F16_BYTES,
                    HEAD_DIM,
                    HEAD_DIM // 2,
                    BLK_N,
                    2,
                    1,
                    1,
                    1,
                    0,
                    3,
                    2,
                    0,
                )
            )
            _builder_emit(
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    V_tensor_map_1,
                    "float16",
                    3,
                    V.data,
                    HEAD_DIM // 2,
                    SEQ_LEN_KV,
                    BATCH_SIZE * NUM_KV_HEADS * 2,
                    NUM_KV_HEADS * HEAD_DIM * F16_BYTES,
                    HEAD_DIM,
                    HEAD_DIM // 2,
                    BLK_N,
                    2,
                    1,
                    1,
                    1,
                    0,
                    3,
                    2,
                    0,
                )
            )
            EPI_ON_SOFTMAX = T.meta_var(is_causal).value
            EARLY_Q_RELEASE = T.meta_var(not is_causal).value
            max_ctas = _builder_bind("max_ctas", 148)
            cta_count = _builder_bind(
                "cta_count", T.min(max_ctas, num_total_tasks) if not is_causal else num_total_tasks
            )
            _builder_emit(T.device_entry())
            _builder_enter(T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1}))
            bx = _builder_name("bx", T.cta_id([cta_count]))
            wg_id = _builder_name("wg_id", T.warpgroup_id([4]))
            warp_id = _builder_name("warp_id", T.warp_id_in_wg([4]))
            tid_in_wg = _builder_name("tid_in_wg", T.thread_id_in_wg([128]))
            pool = _builder_meta("pool", T.SMEMPool())
            Q_smem = _builder_scalar(
                "Q_smem", pool.alloc_tcgen05_mma_AB((SMEM_PIPE_DEPTH_Q, BLK_M, HEAD_DIM), "float16")
            )
            K_smem = _builder_scalar(
                "K_smem",
                pool.alloc_tcgen05_mma_AB((SMEM_PIPE_DEPTH_KV, BLK_N, HEAD_DIM), "float16"),
            )
            V_smem = _builder_scalar("V_smem", K_smem.view(SMEM_PIPE_DEPTH_KV, BLK_N, HEAD_DIM))
            O_smem = _builder_scalar(
                "O_smem", pool.alloc_tcgen05_mma_AB((TMEM_PIPE_DEPTH, BLK_M, HEAD_DIM), "float16")
            )
            q_desc = _builder_scalar("q_desc", SmemDescriptor())
            _builder_emit(q_desc.init(Q_smem.ptr_to([0, 0, 0]), ldo=1024, sdo=64, swizzle=3))
            _builder_emit(q_desc.make_lo_uniform())
            k_desc = _builder_scalar("k_desc", SmemDescriptor())
            _builder_emit(k_desc.init(K_smem.ptr_to([0, 0, 0]), ldo=1024, sdo=64, swizzle=3))
            _builder_emit(k_desc.make_lo_uniform())
            v_desc = _builder_scalar("v_desc", SmemDescriptor())
            _builder_emit(v_desc.init(V_smem.ptr_to([0, 0, 0]), ldo=1024, sdo=64, swizzle=3))
            _builder_emit(v_desc.make_lo_uniform())
            q_desc_steady = _builder_scalar("q_desc_steady", SmemDescriptor())
            k_desc_steady = _builder_scalar("k_desc_steady", SmemDescriptor())
            v_desc_steady_hi = _builder_scalar("v_desc_steady_hi", SmemDescriptor())
            v_desc_tail_lo = _builder_scalar("v_desc_tail_lo", SmemDescriptor())
            v_desc_tail_hi = _builder_scalar("v_desc_tail_hi", SmemDescriptor())
            if is_causal and GQA_RATIO > 1:
                _builder_emit(
                    q_desc_steady.init(Q_smem.ptr_to([0, 0, 0]), ldo=1024, sdo=64, swizzle=3)
                )
                _builder_emit(q_desc_steady.make_lo_uniform())
                _builder_emit(
                    k_desc_steady.init(K_smem.ptr_to([0, 0, 0]), ldo=1024, sdo=64, swizzle=3)
                )
                _builder_emit(k_desc_steady.make_lo_uniform())
                _builder_emit(
                    v_desc_steady_hi.init(V_smem.ptr_to([0, 0, 0]), ldo=1024, sdo=64, swizzle=3)
                )
                _builder_emit(v_desc_steady_hi.make_lo_uniform())
                _builder_emit(
                    v_desc_tail_lo.init(V_smem.ptr_to([0, 0, 0]), ldo=1024, sdo=64, swizzle=3)
                )
                _builder_emit(v_desc_tail_lo.make_lo_uniform())
                _builder_emit(
                    v_desc_tail_hi.init(V_smem.ptr_to([0, 0, 0]), ldo=1024, sdo=64, swizzle=3)
                )
                _builder_emit(v_desc_tail_hi.make_lo_uniform())
            sScale = _builder_name(
                "sScale", pool.alloc((SSCALE_TOTAL_SIZE,), "float32", align=1024)
            )
            tmem_addr = _builder_name("tmem_addr", pool.alloc([1], "uint32"))
            ACC_SCALE_BASE = _builder_bind("ACC_SCALE_BASE", 0)
            ROW_SUM_BASE = _builder_bind("ROW_SUM_BASE", 0)
            kv_pipe = _builder_scalar("kv_pipe", PipelineState(SMEM_PIPE_DEPTH_KV))
            phase_q = _builder_alloc_scalar("phase_q", "int32")
            phase_s_full = _builder_alloc_scalar("phase_s_full", "int32")
            phase_tmem = _builder_alloc_scalar("phase_tmem", "int32")
            phase_s0_s1 = _builder_alloc_scalar("phase_s0_s1", "int32")
            phase_q_load = _builder_alloc_scalar("phase_q_load", "int32")
            phase_oepi = _builder_alloc_scalar("phase_oepi", "int32")
            q_load = _builder_scalar(
                "q_load",
                Pipeline(
                    pool, SMEM_PIPE_DEPTH_Q, full="tma", empty="tcgen05", empty_phase_offset=1
                ),
            )
            kv_load = _builder_scalar(
                "kv_load",
                Pipeline(
                    pool, SMEM_PIPE_DEPTH_KV, full="tma", empty="tcgen05", empty_phase_offset=1
                ),
            )
            p_o_rescale = _builder_scalar("p_o_rescale", MBarrier(pool, 2))
            _builder_emit(p_o_rescale.init(256))
            s_ready = _builder_scalar("s_ready", TCGen05Bar(pool, 2))
            _builder_emit(s_ready.init(1))
            o_ready = _builder_scalar("o_ready", TCGen05Bar(pool, 2))
            _builder_emit(o_ready.init(1))
            softmax_corr = _builder_scalar(
                "softmax_corr",
                Pipeline(
                    pool,
                    2,
                    full="mbar",
                    empty="mbar",
                    init_full=128,
                    init_empty=128,
                    empty_phase_offset=1,
                ),
            )
            corr_epi = _builder_scalar(
                "corr_epi",
                Pipeline(
                    pool,
                    TMEM_PIPE_DEPTH,
                    full="mbar",
                    empty="mbar",
                    init_full=128,
                    init_empty=32,
                    empty_phase_offset=1,
                ),
            )
            p_ready_2 = _builder_scalar("p_ready_2", MBarrier(pool, 2))
            _builder_emit(p_ready_2.init(128))
            bar_s0_s1_sequence = _builder_scalar("bar_s0_s1_sequence", MBarrier(pool, 8))
            _builder_emit(bar_s0_s1_sequence.init(32))
            _builder_emit(pool.commit())
            iket = _builder_scalar("iket", IketProfiler())
            tmem_pool = _builder_scalar(
                "tmem_pool",
                T.TMEMPool(
                    pool,
                    total_cols=N_COLS_TMEM,
                    cta_group=CTA_GROUP,
                    tmem_addr=tmem_addr,
                    alloc_warp=12,
                    dealloc_warp=0,
                ),
            )
            _tmem_f32 = _builder_scalar("_tmem_f32", tmem_pool.alloc((128, N_COLS_TMEM), "float32"))
            _builder_emit(tmem_pool.move_base_to(0))
            _tmem_f16 = _builder_scalar(
                "_tmem_f16", tmem_pool.alloc((128, N_COLS_TMEM * 2), "float16")
            )
            _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
            _builder_emit(T.ptx.fence.mbarrier_init.release.cluster())
            _builder_emit(T.cuda.cta_sync())
            scheduler = _builder_scalar(
                "scheduler",
                FlashAttentionLPTScheduler(
                    "fa_scheduler",
                    num_batches=BATCH_SIZE,
                    num_heads=NUM_KV_HEADS,
                    num_m_blocks=num_q_blocks,
                    l2_swizzle=L2_SWIZZLE,
                )
                if is_causal
                else FlashAttentionLinearScheduler(
                    "fa_scheduler",
                    num_batches=BATCH_SIZE,
                    num_heads=NUM_KV_HEADS,
                    num_m_blocks=num_q_blocks,
                    num_ctas=cta_count,
                ),
            )
            _builder_emit(scheduler.init(bx))
            _builder_emit(kv_pipe.init(0))
            T.buffer_store(phase_q.buffer, 0, [0])
            T.buffer_store(phase_oepi.buffer, 0, [0])
            T.buffer_store(phase_tmem.buffer, 0, [0])
            T.buffer_store(phase_s_full.buffer, 0, [0])
            if USE_S0_S1_BARRIER:
                T.buffer_store(phase_s0_s1.buffer, T.if_then_else(wg_id == 1, 0, 1), [0])
            T.buffer_store(phase_q_load.buffer, 0, [0])
            _builder_emit(tmem_pool.commit())
            _builder_if_444_4 = _builder_scope_enter(T.If((wg_id == 3) & (warp_id == 0)))
            _builder_then_444_4 = _builder_scope_enter(T.Then())
            allocated_tmem_addr = _builder_alloc_scalar("allocated_tmem_addr", "uint32")
            _builder_emit(T.ptx.ld.shared.u32(allocated_tmem_addr, tmem_addr.ptr_to([0])))
            _builder_emit(T.cuda.trap_when_assert_failed(allocated_tmem_addr == T.uint32(0)))
            _builder_scope_exit(_builder_then_444_4)
            _builder_scope_exit(_builder_if_444_4)
            _builder_if_448_4 = _builder_scope_enter(T.If(wg_id == 2))
            _builder_then_448_4 = _builder_scope_enter(T.Then())
            with T.unroll(2) as i_q:
                _builder_emit(p_o_rescale.arrive(i_q))
            _builder_scope_exit(_builder_then_448_4)
            _builder_scope_exit(_builder_if_448_4)
            num_kv_blocks = _builder_bind("num_kv_blocks", ceildiv(SEQ_LEN_KV, BLK_N))
            with T.While(scheduler.valid()):
                m_block_idx = T.meta_var(scheduler.m_block_idx).value
                batch_idx = T.meta_var(scheduler.batch_idx).value
                kv_head_idx = T.meta_var(scheduler.head_idx).value
                m_start = T.meta_var(m_block_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q).value
                _builder_if_457_8 = _builder_scope_enter(T.If(wg_id == 3))
                _builder_then_457_8 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(48))
                _builder_if_459_12 = _builder_scope_enter(T.If(warp_id == 1))
                _builder_then_459_12 = _builder_scope_enter(T.Then())

                def load_q(i_q, tensor_map):
                    _builder_emit(q_load.empty.wait(i_q, phase_q_load))
                    tma_q_token = _builder_scalar("tma_q_token", iket.range_start("issue-tma-q"))
                    _builder_if_465_20 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                    _builder_then_465_20 = _builder_scope_enter(T.Then())
                    if GQA_RATIO == 1:
                        _builder_emit(
                            T.evaluate(
                                T.ptx[_TMA_G2S_3D](
                                    Q_smem.ptr_to([i_q, 0, 0]),
                                    T.address_of(tensor_map),
                                    T.int32(0),
                                    T.cast(m_start + i_q * SEQ_Q_PER_TILE, "int32"),
                                    T.cast((batch_idx * NUM_QO_HEADS + kv_head_idx) * 2, "int32"),
                                    T.cuda.cvta_generic_to_shared(q_load.full.buf.ptr_to([i_q])),
                                )
                            )
                        )
                    else:
                        _builder_emit(
                            T.evaluate(
                                T.ptx[_TMA_G2S_4D](
                                    Q_smem.ptr_to([i_q, 0, 0]),
                                    T.address_of(tensor_map),
                                    T.int32(0),
                                    T.cast(kv_head_idx * GQA_RATIO, "int32"),
                                    T.cast(m_start + i_q * SEQ_Q_PER_TILE, "int32"),
                                    T.cast(batch_idx * 2, "int32"),
                                    T.cuda.cvta_generic_to_shared(q_load.full.buf.ptr_to([i_q])),
                                )
                            )
                        )
                    _builder_emit(q_load.full.arrive(i_q, CTA_GROUP * BLK_M * HEAD_DIM * F16_BYTES))
                    _builder_scope_exit(_builder_then_465_20)
                    _builder_scope_exit(_builder_if_465_20)
                    _builder_emit(iket.range_end(tma_q_token))

                def load_k(i_kv, tensor_map):
                    _builder_emit(kv_load.empty.wait(kv_pipe.stage, kv_pipe.phase))
                    tma_k_token = _builder_scalar("tma_k_token", iket.range_start("issue-tma-k"))
                    _builder_if_496_20 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                    _builder_then_496_20 = _builder_scope_enter(T.Then())
                    _builder_emit(
                        T.evaluate(
                            T.ptx[_TMA_G2S_3D](
                                K_smem.ptr_to([kv_pipe.stage, 0, 0]),
                                T.address_of(tensor_map),
                                T.int32(0),
                                T.cast(i_kv * BLK_N, "int32"),
                                T.cast((batch_idx * NUM_KV_HEADS + kv_head_idx) * 2, "int32"),
                                T.cuda.cvta_generic_to_shared(
                                    kv_load.full.buf.ptr_to([kv_pipe.stage])
                                ),
                            )
                        )
                    )
                    _builder_emit(
                        kv_load.full.arrive(kv_pipe.stage, CTA_GROUP * BLK_N * HEAD_DIM * F16_BYTES)
                    )
                    _builder_scope_exit(_builder_then_496_20)
                    _builder_scope_exit(_builder_if_496_20)
                    _builder_emit(iket.range_end(tma_k_token))
                    _builder_emit(kv_pipe.advance())

                def load_v(i_kv, tensor_map):
                    _builder_emit(kv_load.empty.wait(kv_pipe.stage, kv_pipe.phase))
                    tma_v_token = _builder_scalar("tma_v_token", iket.range_start("issue-tma-v"))
                    _builder_if_517_20 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                    _builder_then_517_20 = _builder_scope_enter(T.Then())
                    _builder_emit(
                        T.evaluate(
                            T.ptx[_TMA_G2S_3D](
                                V_smem.ptr_to([kv_pipe.stage, 0, 0]),
                                T.address_of(tensor_map),
                                T.int32(0),
                                T.cast(i_kv * BLK_N, "int32"),
                                T.cast((batch_idx * NUM_KV_HEADS + kv_head_idx) * 2, "int32"),
                                T.cuda.cvta_generic_to_shared(
                                    kv_load.full.buf.ptr_to([kv_pipe.stage])
                                ),
                            )
                        )
                    )
                    _builder_emit(
                        kv_load.full.arrive(kv_pipe.stage, CTA_GROUP * BLK_N * HEAD_DIM * F16_BYTES)
                    )
                    _builder_scope_exit(_builder_then_517_20)
                    _builder_scope_exit(_builder_if_517_20)
                    _builder_emit(iket.range_end(tma_v_token))
                    _builder_emit(kv_pipe.advance())

                load_trip_count = _builder_alloc_scalar("load_trip_count", "int32")
                T.buffer_store(
                    load_trip_count.buffer,
                    get_n_block_max(m_block_idx, is_causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE)
                    if is_causal
                    else num_kv_blocks,
                    [0],
                )
                _builder_emit(load_q(0, Q_tensor_map))
                _builder_emit(load_k(load_trip_count - 1, K_tensor_map))
                _builder_emit(load_q(1, Q_tensor_map_1))
                T.buffer_store(phase_q_load.buffer, phase_q_load ^ 1, [0])
                _builder_emit(load_v(load_trip_count - 1, V_tensor_map))
                with T.serial(load_trip_count - 1, unroll=False) as _i:
                    i_kv = _builder_bind("i_kv", load_trip_count - 2 - _i)
                    _builder_emit(load_k(i_kv, K_tensor_map_1))
                    _builder_emit(load_v(i_kv, V_tensor_map_1))
                _builder_scope_exit(_builder_then_459_12)
                _builder_scope_exit(_builder_if_459_12)
                _builder_if_549_12 = _builder_scope_enter(T.If(warp_id == 2))
                _builder_then_549_12 = _builder_scope_enter(T.Then())
                _builder_emit(corr_epi.full.wait(0, phase_tmem))
                tma_store_token = _builder_scalar("tma_store_token", iket.range_start("tma-store"))
                with T.unroll(SMEM_PIPE_DEPTH_Q) as i_q:
                    _builder_if_553_20 = _builder_scope_enter(T.If(i_q != 0))
                    _builder_then_553_20 = _builder_scope_enter(T.Then())
                    _builder_emit(corr_epi.full.wait(i_q, phase_tmem))
                    _builder_scope_exit(_builder_then_553_20)
                    _builder_scope_exit(_builder_if_553_20)
                    m_start_global = T.meta_var(m_start + i_q * SEQ_Q_PER_TILE).value
                    _builder_if_556_20 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                    _builder_then_556_20 = _builder_scope_enter(T.Then())
                    if GQA_RATIO == 1:
                        _builder_emit(
                            T.evaluate(
                                T.ptx[_TMA_S2G_3D](
                                    T.address_of(O_tensor_map),
                                    T.int32(0),
                                    T.cast(m_start_global, "int32"),
                                    T.cast((batch_idx * NUM_QO_HEADS + kv_head_idx) * 2, "int32"),
                                    O_smem.ptr_to([i_q, 0, 0]),
                                )
                            )
                        )
                    else:
                        _builder_emit(
                            T.evaluate(
                                T.ptx[_TMA_S2G_4D](
                                    T.address_of(O_tensor_map),
                                    T.int32(0),
                                    T.cast(kv_head_idx * GQA_RATIO, "int32"),
                                    T.cast(m_start_global, "int32"),
                                    T.cast(batch_idx * 2, "int32"),
                                    O_smem.ptr_to([i_q, 0, 0]),
                                )
                            )
                        )
                    _builder_scope_exit(_builder_then_556_20)
                    _builder_scope_exit(_builder_if_556_20)
                    _builder_emit(T.ptx.cp.async_.bulk.commit_group())
                _builder_emit(T.ptx.cp.async_.bulk.wait_group(1))
                _builder_emit(corr_epi.empty.arrive(0))
                _builder_emit(T.ptx.cp.async_.bulk.wait_group(0))
                _builder_emit(corr_epi.empty.arrive(1))
                _builder_emit(iket.range_end(tma_store_token))
                T.buffer_store(phase_tmem.buffer, phase_tmem ^ 1, [0])
                _builder_scope_exit(_builder_then_549_12)
                _builder_scope_exit(_builder_if_549_12)
                _builder_if_588_12 = _builder_scope_enter(T.If(warp_id == 0))
                _builder_then_588_12 = _builder_scope_enter(T.Then())
                acc = _builder_alloc_scalar("acc", "int32")
                T.buffer_store(acc.buffer, 0, [0])

                def gemm_qk(q_stage, kv_stage, q_desc_value, k_desc_value):
                    with T.unroll(HEAD_DIM // MMA_K) as ki:
                        _builder_if_595_24 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                        _builder_then_595_24 = _builder_scope_enter(T.Then())
                        _builder_emit(
                            T.ptx[_MMA_F16](
                                T.cast(q_stage * MMA_N, "uint32"),
                                _smem_desc_add_16B_offset(
                                    q_desc_value, q_stage * 2048 + ki // 4 * 1024 + ki % 4 * 2
                                ),
                                _smem_desc_add_16B_offset(
                                    k_desc_value, kv_stage * 2048 + ki // 4 * 1024 + ki % 4 * 2
                                ),
                                T.uint32(136314896),
                                T.uint32(0),
                                T.uint32(0),
                                T.uint32(0),
                                T.uint32(0),
                                T.cast(ki != 0, "bool"),
                            )
                        )
                        _builder_scope_exit(_builder_then_595_24)
                        _builder_scope_exit(_builder_if_595_24)
                    _builder_if_611_20 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                    _builder_then_611_20 = _builder_scope_enter(T.Then())
                    _builder_emit(s_ready.arrive(q_stage))
                    _builder_scope_exit(_builder_then_611_20)
                    _builder_scope_exit(_builder_if_611_20)

                K_SPLIT = T.meta_var((4 if is_causal else 6) * MMA_K).value

                def gemm_pv_part1(i_q, kv_stage, should_accumulate, v_desc_value):
                    with T.unroll(K_SPLIT // MMA_K) as ki:
                        _builder_if_627_24 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                        _builder_then_627_24 = _builder_scope_enter(T.Then())
                        _builder_emit(
                            T.ptx[_MMA_F16](
                                T.cast((SMEM_PIPE_DEPTH_Q + i_q) * MMA_N, "uint32"),
                                T.cast(i_q * MMA_N + MMA_N // 2 + ki * (MMA_K // 2), "uint32"),
                                _smem_desc_add_16B_offset(v_desc_value, kv_stage * 2048 + ki * 128),
                                T.uint32(136380432),
                                T.uint32(0),
                                T.uint32(0),
                                T.uint32(0),
                                T.uint32(0),
                                T.cast(
                                    tvm.tirx.any(ki != 0, T.cast(should_accumulate, "bool")), "bool"
                                ),
                            )
                        )
                        _builder_scope_exit(_builder_then_627_24)
                        _builder_scope_exit(_builder_if_627_24)

                def gemm_pv_part2(i_q, kv_stage, v_desc_value):
                    _builder_emit(p_ready_2.wait(i_q, phase_tmem))
                    with T.unroll((BLK_N - K_SPLIT) // MMA_K) as ki:
                        _builder_if_644_24 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                        _builder_then_644_24 = _builder_scope_enter(T.Then())
                        _builder_emit(
                            T.ptx[_MMA_F16](
                                T.cast((SMEM_PIPE_DEPTH_Q + i_q) * MMA_N, "uint32"),
                                T.cast(
                                    i_q * MMA_N + MMA_N // 2 + K_SPLIT // 2 + ki * (MMA_K // 2),
                                    "uint32",
                                ),
                                _smem_desc_add_16B_offset(
                                    v_desc_value, kv_stage * 2048 + K_SPLIT * 8 + ki * 128
                                ),
                                T.uint32(136380432),
                                T.uint32(0),
                                T.uint32(0),
                                T.uint32(0),
                                T.uint32(0),
                                T.bool(True),
                            )
                        )
                        _builder_scope_exit(_builder_then_644_24)
                        _builder_scope_exit(_builder_if_644_24)

                def gemm_pv(i_q, kv_stage, should_accumulate, v_desc_lo, v_desc_hi):
                    _builder_emit(gemm_pv_part1(i_q, kv_stage, should_accumulate, v_desc_lo))
                    _builder_emit(gemm_pv_part2(i_q, kv_stage, v_desc_hi))

                with T.unroll(SMEM_PIPE_DEPTH_Q) as i_q:
                    _builder_emit(q_load.full.wait(i_q, phase_q_load))
                    _builder_if_669_20 = _builder_scope_enter(T.If(i_q == 0))
                    _builder_then_669_20 = _builder_scope_enter(T.Then())
                    _builder_emit(kv_load.full.wait(kv_pipe.stage, kv_pipe.phase))
                    _builder_scope_exit(_builder_then_669_20)
                    _builder_scope_exit(_builder_if_669_20)
                    _builder_emit(gemm_qk(i_q, kv_pipe.stage, q_desc.desc, k_desc.desc))
                    _builder_if_672_20 = _builder_scope_enter(T.If(i_q == 1))
                    _builder_then_672_20 = _builder_scope_enter(T.Then())
                    _builder_if_673_24 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                    _builder_then_673_24 = _builder_scope_enter(T.Then())
                    _builder_emit(kv_load.empty.arrive(kv_pipe.stage))
                    _builder_scope_exit(_builder_then_673_24)
                    _builder_scope_exit(_builder_if_673_24)
                    _builder_scope_exit(_builder_then_672_20)
                    _builder_scope_exit(_builder_if_672_20)
                _builder_emit(kv_pipe.advance())
                mma_trip_count = _builder_alloc_scalar("mma_trip_count", "int32")
                T.buffer_store(
                    mma_trip_count.buffer,
                    get_n_block_max(m_block_idx, is_causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE)
                    if is_causal
                    else num_kv_blocks,
                    [0],
                )
                with T.serial(mma_trip_count - 1, unroll=False) as i_kv:
                    stage_v = _builder_bind("stage_v", kv_pipe.stage)
                    phase_v = _builder_bind("phase_v", kv_pipe.phase)
                    _builder_emit(kv_pipe.advance())
                    stage_k = T.meta_var(kv_pipe.stage).value
                    phase_k = T.meta_var(kv_pipe.phase).value
                    with T.unroll(SMEM_PIPE_DEPTH_Q) as i_q:
                        _builder_if_689_24 = _builder_scope_enter(T.If(i_q == 0))
                        _builder_then_689_24 = _builder_scope_enter(T.Then())
                        _builder_emit(kv_load.full.wait(stage_v, phase_v))
                        _builder_scope_exit(_builder_then_689_24)
                        _builder_scope_exit(_builder_if_689_24)
                        _builder_emit(p_o_rescale.wait(i_q, phase_tmem))
                        _builder_emit(
                            gemm_pv(
                                i_q,
                                stage_v,
                                acc,
                                v_desc.desc,
                                v_desc_steady_hi.desc
                                if is_causal and GQA_RATIO > 1
                                else v_desc.desc,
                            )
                        )
                        _builder_if_699_24 = _builder_scope_enter(T.If(i_q == 1))
                        _builder_then_699_24 = _builder_scope_enter(T.Then())
                        _builder_if_700_28 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                        _builder_then_700_28 = _builder_scope_enter(T.Then())
                        _builder_emit(kv_load.empty.arrive(stage_v))
                        _builder_scope_exit(_builder_then_700_28)
                        _builder_scope_exit(_builder_if_700_28)
                        _builder_scope_exit(_builder_then_699_24)
                        _builder_scope_exit(_builder_if_699_24)
                        _builder_if_702_24 = _builder_scope_enter(T.If(i_q == 0))
                        _builder_then_702_24 = _builder_scope_enter(T.Then())
                        _builder_emit(kv_load.full.wait(stage_k, phase_k))
                        _builder_scope_exit(_builder_then_702_24)
                        _builder_scope_exit(_builder_if_702_24)
                        _builder_emit(
                            gemm_qk(
                                i_q,
                                stage_k,
                                q_desc_steady.desc if is_causal and GQA_RATIO > 1 else q_desc.desc,
                                k_desc_steady.desc if is_causal and GQA_RATIO > 1 else k_desc.desc,
                            )
                        )
                        if EARLY_Q_RELEASE:
                            _builder_if_720_28 = _builder_scope_enter(
                                T.If(i_kv == mma_trip_count - 2)
                            )
                            _builder_then_720_28 = _builder_scope_enter(T.Then())
                            _builder_if_721_32 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                            _builder_then_721_32 = _builder_scope_enter(T.Then())
                            _builder_emit(q_load.empty.arrive(i_q))
                            _builder_scope_exit(_builder_then_721_32)
                            _builder_scope_exit(_builder_if_721_32)
                            _builder_scope_exit(_builder_then_720_28)
                            _builder_scope_exit(_builder_if_720_28)
                        _builder_if_723_24 = _builder_scope_enter(T.If(i_q == 1))
                        _builder_then_723_24 = _builder_scope_enter(T.Then())
                        _builder_if_724_28 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                        _builder_then_724_28 = _builder_scope_enter(T.Then())
                        _builder_emit(kv_load.empty.arrive(stage_k))
                        _builder_scope_exit(_builder_then_724_28)
                        _builder_scope_exit(_builder_if_724_28)
                        _builder_scope_exit(_builder_then_723_24)
                        _builder_scope_exit(_builder_if_723_24)
                    T.buffer_store(acc.buffer, 1, [0])
                    _builder_emit(kv_pipe.advance())
                    T.buffer_store(phase_tmem.buffer, phase_tmem ^ 1, [0])
                with T.unroll(SMEM_PIPE_DEPTH_Q) as i_q:
                    _builder_if_730_20 = _builder_scope_enter(T.If(i_q == 0))
                    _builder_then_730_20 = _builder_scope_enter(T.Then())
                    _builder_emit(kv_load.full.wait(kv_pipe.stage, kv_pipe.phase))
                    _builder_scope_exit(_builder_then_730_20)
                    _builder_scope_exit(_builder_if_730_20)
                    _builder_emit(p_o_rescale.wait(i_q, phase_tmem))
                    _builder_emit(
                        gemm_pv(
                            i_q,
                            kv_pipe.stage,
                            acc,
                            v_desc_tail_lo.desc if is_causal and GQA_RATIO > 1 else v_desc.desc,
                            v_desc_tail_hi.desc if is_causal and GQA_RATIO > 1 else v_desc.desc,
                        )
                    )
                    _builder_if_740_20 = _builder_scope_enter(T.If(i_q == 1))
                    _builder_then_740_20 = _builder_scope_enter(T.Then())
                    _builder_if_741_24 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                    _builder_then_741_24 = _builder_scope_enter(T.Then())
                    _builder_emit(kv_load.empty.arrive(kv_pipe.stage))
                    _builder_scope_exit(_builder_then_741_24)
                    _builder_scope_exit(_builder_if_741_24)
                    _builder_scope_exit(_builder_then_740_20)
                    _builder_scope_exit(_builder_if_740_20)
                    _builder_if_743_20 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                    _builder_then_743_20 = _builder_scope_enter(T.Then())
                    _builder_emit(o_ready.arrive(i_q))
                    _builder_scope_exit(_builder_then_743_20)
                    _builder_scope_exit(_builder_if_743_20)
                _builder_emit(kv_pipe.advance())
                T.buffer_store(phase_tmem.buffer, phase_tmem ^ 1, [0])
                if not EARLY_Q_RELEASE:
                    with T.unroll(SMEM_PIPE_DEPTH_Q) as i_q:
                        _builder_if_749_24 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                        _builder_then_749_24 = _builder_scope_enter(T.Then())
                        _builder_emit(q_load.empty.arrive(i_q))
                        _builder_scope_exit(_builder_then_749_24)
                        _builder_scope_exit(_builder_if_749_24)
                T.buffer_store(phase_q_load.buffer, phase_q_load ^ 1, [0])
                _builder_scope_exit(_builder_then_588_12)
                _builder_scope_exit(_builder_if_588_12)
                _builder_scope_exit(_builder_then_457_8)
                _builder_else_457_8 = _builder_scope_enter(T.Else())
                _builder_if_752_8 = _builder_scope_enter(T.If(wg_id < 2))
                _builder_then_752_8 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.setmaxnreg.inc.sync.aligned.u32(200))
                scale_log2 = T.meta_var(math.log2(math.e) / math.sqrt(HEAD_DIM)).value
                rescale_threshold = T.meta_var(8.0).value
                row_max = _builder_name("row_max", T.alloc_local((1,), "float32"))
                row_sum = _builder_name("row_sum", T.alloc_local((1,), "float32"))
                _builder_if_758_12 = _builder_scope_enter(T.If(warp_id == 0))
                _builder_then_758_12 = _builder_scope_enter(T.Then())
                _builder_emit(iket.mark("softmax-baseline"))
                _builder_scope_exit(_builder_then_758_12)
                _builder_scope_exit(_builder_if_758_12)

                def mask_r2p(s_chunk_buf, col_limit, ncol: T.int32):
                    """Apply mask using R2P-style bit manipulation.

                    Optimizes: for j in range(N): buf[j] = -inf if j >= col_limit else buf[j]
                    Into: bitmask operations that compile to R2P PTX instruction.

                    Following flash_attn/cute/mask.py mask_r2p(): process in 32-col
                    chunks (shl_u32_clamp tolerates shift>=32, so no 24-col split).

                    The bit test `mask & (1 << i)` compiles to the R2P (Register to Predicate)
                    PTX instruction, which is more efficient than per-column comparisons.
                    """
                    ncol = T.meta_var(ncol).value
                    CHUNK_SIZE = _builder_bind("CHUNK_SIZE", 32)
                    num_chunks = _builder_bind("num_chunks", ceildiv(ncol, CHUNK_SIZE))
                    s_chunk_local = _builder_scalar("s_chunk_local", s_chunk_buf.local(ncol))
                    with T.unroll(num_chunks) as s:
                        k_keep = _builder_bind("k_keep", T.max(col_limit - s * CHUNK_SIZE, 0))
                        mask_inv = _builder_alloc_scalar("mask_inv", "uint32")
                        T.buffer_store(
                            mask_inv.buffer,
                            shl_u32_clamp(T.uint32(4294967295), T.cast(k_keep, "uint32")),
                            [0],
                        )
                        with T.unroll(CHUNK_SIZE) as i:
                            _builder_if_788_24 = _builder_scope_enter(
                                T.If(i < ncol - s * CHUNK_SIZE)
                            )
                            _builder_then_788_24 = _builder_scope_enter(T.Then())
                            c = _builder_bind("c", s * CHUNK_SIZE + i)
                            in_bound = _builder_bind(
                                "in_bound",
                                T.bitwise_and(
                                    T.bitwise_not(mask_inv), T.shift_left(T.uint32(1), i)
                                ),
                            )
                            T.buffer_store(
                                s_chunk_local,
                                T.Select(
                                    T.cast(in_bound, "bool"),
                                    s_chunk_local[c],
                                    T.float32(-float("inf")),
                                ),
                                [c],
                            )
                            _builder_scope_exit(_builder_then_788_24)
                            _builder_scope_exit(_builder_if_788_24)

                def apply_causal_mask(s_chunk_buf, m_blk_idx, n_blk_idx):
                    """Apply causal mask to attention scores.

                    Following flash_attn/cute/mask.py apply_mask_sm100() lines 384-400:
                    causal_row_offset = 1 + seqlen_k - n_block * tile_n - seqlen_q
                    row_idx = thread_row + m_block * tile_m
                    col_limit_right = row_idx + causal_row_offset
                    Mask if col >= col_limit_right

                    Coordinate Mapping:
                    - BLK_M = 128 packed rows per tile
                    - SEQ_Q_PER_TILE = BLK_M // GQA_RATIO (e.g., 32 for GQA_RATIO=4)
                    - Each warpgroup handles one Q stage with SEQ_Q_PER_TILE sequence positions
                    - tid_in_wg (0-127) maps to packed rows: (seq_pos, head) = (tid//GQA_RATIO, tid%GQA_RATIO)
                    """
                    seq_pos_in_wg = _builder_bind("seq_pos_in_wg", tid_in_wg // GQA_RATIO)
                    row_idx = _builder_bind(
                        "row_idx",
                        m_blk_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
                        + wg_id * SEQ_Q_PER_TILE
                        + seq_pos_in_wg,
                    )
                    causal_row_offset = _builder_bind(
                        "causal_row_offset", 1 + SEQ_LEN_KV - n_blk_idx * BLK_N - SEQ_LEN_Q
                    )
                    col_limit_right = _builder_bind("col_limit_right", row_idx + causal_row_offset)
                    _builder_emit(mask_r2p(s_chunk_buf, col_limit_right, BLK_N))

                def softmax_step(i_kv, apply_mask=False, is_first=False):
                    s_chunk_buf = _builder_name("s_chunk_buf", T.alloc_local((BLK_N,), "float32"))
                    p_chunk_buf_f32 = _builder_name(
                        "p_chunk_buf_f32", T.alloc_local((BLK_N // 2,), "float32")
                    )
                    p_chunk_buf = _builder_name(
                        "p_chunk_buf",
                        T.decl_buffer((BLK_N,), dtype="float16", data=p_chunk_buf_f32.data),
                    )
                    p_chunk_u32 = _builder_scalar("p_chunk_u32", p_chunk_buf.view("uint32"))
                    _builder_emit(s_ready.wait(wg_id, phase_s_full))
                    _builder_if_830_16 = _builder_scope_enter(T.If(warp_id == 0))
                    _builder_then_830_16 = _builder_scope_enter(T.Then())
                    _builder_emit(iket.mark("softmax-phase-0"))
                    _builder_scope_exit(_builder_then_830_16)
                    _builder_scope_exit(_builder_if_830_16)
                    softmax_max_token = _builder_scalar(
                        "softmax_max_token", iket.sentinel_token("softmax-max")
                    )
                    _builder_if_833_16 = _builder_scope_enter(T.If(warp_id == 0))
                    _builder_then_833_16 = _builder_scope_enter(T.Then())
                    T.buffer_store(softmax_max_token.buffer, iket.range_start("softmax-max"), [0])
                    _builder_scope_exit(_builder_then_833_16)
                    _builder_scope_exit(_builder_if_833_16)
                    tile_max = _builder_name("tile_max", T.alloc_local((1,), "float32"))
                    with T.unroll(BLK_N // SOFTMAX_LD_CHUNK) as chunk_idx:
                        _builder_emit(
                            T.evaluate(
                                _tmem_load(
                                    s_chunk_buf,
                                    chunk_idx * SOFTMAX_LD_CHUNK,
                                    T.cuda.get_tmem_addr(
                                        T.uint32(0), 0, wg_id * MMA_N + chunk_idx * SOFTMAX_LD_CHUNK
                                    ),
                                    SOFTMAX_LD_CHUNK,
                                )
                            )
                        )
                    if apply_mask:
                        _builder_emit(apply_causal_mask(s_chunk_buf, m_block_idx, i_kv))
                    row_max_old = _builder_alloc_scalar("row_max_old", "f32")
                    if is_first:
                        _builder_emit(reduce_max_128(tile_max, s_chunk_buf))
                    else:
                        T.buffer_store(row_max_old.buffer, row_max[0], [0])
                        T.buffer_store(tile_max, row_max_old, [0])
                        _builder_emit(reduce_max_128(tile_max, s_chunk_buf, accum=True))
                    row_max_new = _builder_alloc_scalar("row_max_new", "f32")
                    acc_scale = _builder_alloc_scalar("acc_scale", "f32")
                    acc_scale_ = _builder_alloc_scalar("acc_scale_", "f32")
                    row_max_safe = _builder_alloc_scalar("row_max_safe", "f32")
                    T.buffer_store(row_max_new.buffer, tile_max[0], [0])
                    T.buffer_store(
                        row_max_safe.buffer,
                        T.if_then_else(tile_max[0] == -float("inf"), 0.0, tile_max[0]),
                        [0],
                    )
                    if is_first:
                        T.buffer_store(acc_scale.buffer, T.float32(1.0), [0])
                    else:
                        T.buffer_store(
                            acc_scale_.buffer, (row_max_old - row_max_safe) * scale_log2, [0]
                        )
                        _builder_if_870_20 = _builder_scope_enter(
                            T.If(acc_scale_ >= -rescale_threshold)
                        )
                        _builder_then_870_20 = _builder_scope_enter(T.Then())
                        T.buffer_store(row_max_new.buffer, row_max_old, [0])
                        T.buffer_store(row_max_safe.buffer, row_max_old, [0])
                        T.buffer_store(acc_scale.buffer, T.float32(1.0), [0])
                        _builder_scope_exit(_builder_then_870_20)
                        _builder_else_870_20 = _builder_scope_enter(T.Else())
                        _builder_emit(T.ptx.ex2.approx.ftz.f32(acc_scale, acc_scale_))
                        _builder_scope_exit(_builder_else_870_20)
                        _builder_scope_exit(_builder_if_870_20)
                    T.buffer_store(row_max, row_max_new, [0])
                    row_max_scaled = _builder_bind("row_max_scaled", row_max_safe * scale_log2)
                    _builder_if_878_16 = _builder_scope_enter(T.If(warp_id == 0))
                    _builder_then_878_16 = _builder_scope_enter(T.Then())
                    _builder_emit(iket.mark("softmax-phase-1"))
                    _builder_scope_exit(_builder_then_878_16)
                    _builder_scope_exit(_builder_if_878_16)
                    _builder_emit(iket.range_end(softmax_max_token))
                    _builder_if_881_16 = _builder_scope_enter(
                        T.If(T.And(tid_in_wg < BLK_M, T.bool(not is_first)))
                    )
                    _builder_then_881_16 = _builder_scope_enter(T.Then())
                    sScale_idx = _builder_bind(
                        "sScale_idx", ACC_SCALE_BASE + tid_in_wg + wg_id * BLK_M
                    )
                    _builder_emit(T.ptx.st.shared.f32(sScale.ptr_to([sScale_idx]), acc_scale))
                    _builder_scope_exit(_builder_then_881_16)
                    _builder_scope_exit(_builder_if_881_16)
                    if STATS_BAR_PAIRWISE:
                        _builder_emit(T.ptx.bar.arrive(T.uint32(1 + wg_id * 4 + warp_id), 64))
                    else:
                        _builder_emit(T.ptx.bar.arrive(T.uint32(1 + wg_id), 256))
                    softmax_fma_token = _builder_scalar(
                        "softmax_fma_token", iket.sentinel_token("softmax-fma")
                    )
                    _builder_if_897_16 = _builder_scope_enter(T.If(warp_id == 0))
                    _builder_then_897_16 = _builder_scope_enter(T.Then())
                    T.buffer_store(softmax_fma_token.buffer, iket.range_start("softmax-fma"), [0])
                    _builder_scope_exit(_builder_then_897_16)
                    _builder_scope_exit(_builder_if_897_16)
                    with T.unroll(BLK_N // 2) as i:
                        _builder_emit(
                            fma_f32x2(s_chunk_buf, 2 * i, T.float32(scale_log2), -row_max_scaled)
                        )
                    _builder_emit(iket.range_end(softmax_fma_token))
                    if USE_S0_S1_BARRIER:
                        _builder_emit(bar_s0_s1_sequence.wait(wg_id * 4 + warp_id, phase_s0_s1))
                    softmax_exp2_token = _builder_scalar(
                        "softmax_exp2_token", iket.sentinel_token("softmax-exp2")
                    )
                    _builder_if_905_16 = _builder_scope_enter(T.If(warp_id == 0))
                    _builder_then_905_16 = _builder_scope_enter(T.Then())
                    T.buffer_store(softmax_exp2_token.buffer, iket.range_start("softmax-exp2"), [0])
                    _builder_scope_exit(_builder_then_905_16)
                    _builder_scope_exit(_builder_if_905_16)
                    with T.unroll(4) as frag_idx:
                        s_chunk_local = _builder_scalar("s_chunk_local", s_chunk_buf.local(BLK_N))
                        with T.unroll(BLK_N // 4 // 2) as i:
                            idx = T.meta_var(frag_idx * BLK_N // 4 + 2 * i).value
                            emu_pairs = T.meta_var(
                                EMU_PAIRS_CAUSAL if is_causal else EMU_PAIRS_NC
                            ).value
                            emu_start = T.meta_var(
                                EMU_START_CAUSAL if is_causal else EMU_START_NC
                            ).value
                            _builder_if_916_24 = _builder_scope_enter(
                                T.If(
                                    T.Or(
                                        T.Or(
                                            T.Or(
                                                i * 2 % 16 < 16 - 2 * emu_pairs, frag_idx >= 4 - 1
                                            ),
                                            frag_idx < emu_start,
                                        ),
                                        apply_mask,
                                    )
                                )
                            )
                            _builder_then_916_24 = _builder_scope_enter(T.Then())
                            _builder_emit(
                                T.ptx.ex2.approx.ftz.f32(s_chunk_local[idx], s_chunk_local[idx])
                            )
                            _builder_emit(
                                T.ptx.ex2.approx.ftz.f32(
                                    s_chunk_local[idx + 1], s_chunk_local[idx + 1]
                                )
                            )
                            _builder_scope_exit(_builder_then_916_24)
                            _builder_else_916_24 = _builder_scope_enter(T.Else())
                            _builder_emit(
                                ex2_emulation_2(
                                    s_chunk_local, idx, s_chunk_local[idx], s_chunk_local[idx + 1]
                                )
                            )
                            _builder_scope_exit(_builder_else_916_24)
                            _builder_scope_exit(_builder_if_916_24)
                        with T.unroll(BLK_N // 4 // 2) as i:
                            idx = T.meta_var(frag_idx * BLK_N // 4 + 2 * i).value
                            _builder_emit(
                                T.evaluate(_cast_f32x2_f16x2(p_chunk_buf, s_chunk_buf, idx))
                            )
                    if USE_S0_S1_BARRIER:
                        _builder_emit(bar_s0_s1_sequence.arrive((1 - wg_id) * 4 + warp_id))
                    _builder_emit(iket.range_end(softmax_exp2_token))
                    softmax_tmem_st_token = _builder_scalar(
                        "softmax_tmem_st_token", iket.sentinel_token("softmax-tmem-st")
                    )
                    _builder_if_935_16 = _builder_scope_enter(T.If(warp_id == 0))
                    _builder_then_935_16 = _builder_scope_enter(T.Then())
                    T.buffer_store(
                        softmax_tmem_st_token.buffer, iket.range_start("softmax-tmem-st"), [0]
                    )
                    _builder_scope_exit(_builder_then_935_16)
                    _builder_scope_exit(_builder_if_935_16)
                    P_SPLIT_Q = T.meta_var(2 if is_causal else 3).value
                    with T.unroll(P_SPLIT_Q) as i:
                        _builder_emit(
                            T.evaluate(
                                _tmem_store(
                                    p_chunk_u32,
                                    i * BLK_N // 4 // 2,
                                    T.cuda.get_tmem_addr(
                                        T.uint32(0),
                                        0,
                                        (wg_id * 2 * MMA_N + MMA_N + i * BLK_N // 4) // 2,
                                    ),
                                )
                            )
                        )
                    _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
                    _builder_emit(p_o_rescale.arrive(wg_id))
                    with T.unroll(4 - P_SPLIT_Q) as i:
                        _builder_emit(
                            T.evaluate(
                                _tmem_store(
                                    p_chunk_u32,
                                    (P_SPLIT_Q + i) * BLK_N // 4 // 2,
                                    T.cuda.get_tmem_addr(
                                        T.uint32(0),
                                        0,
                                        (wg_id * 2 * MMA_N + MMA_N + (P_SPLIT_Q + i) * BLK_N // 4)
                                        // 2,
                                    ),
                                )
                            )
                        )
                    _builder_if_962_16 = _builder_scope_enter(T.If(warp_id == 0))
                    _builder_then_962_16 = _builder_scope_enter(T.Then())
                    _builder_emit(iket.mark("softmax-phase-2"))
                    _builder_scope_exit(_builder_then_962_16)
                    _builder_scope_exit(_builder_if_962_16)
                    _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
                    _builder_emit(p_ready_2.arrive(wg_id))
                    _builder_if_966_16 = _builder_scope_enter(T.If(warp_id == 0))
                    _builder_then_966_16 = _builder_scope_enter(T.Then())
                    _builder_emit(iket.mark("softmax-phase-3"))
                    _builder_scope_exit(_builder_then_966_16)
                    _builder_scope_exit(_builder_if_966_16)
                    _builder_emit(iket.range_end(softmax_tmem_st_token))
                    _builder_emit(softmax_corr.empty.wait(wg_id, phase_q))
                    _builder_if_970_16 = _builder_scope_enter(T.If(warp_id == 0))
                    _builder_then_970_16 = _builder_scope_enter(T.Then())
                    _builder_emit(iket.mark("softmax-phase-4"))
                    _builder_scope_exit(_builder_then_970_16)
                    _builder_scope_exit(_builder_if_970_16)
                    softmax_sum_token = _builder_scalar(
                        "softmax_sum_token", iket.sentinel_token("softmax-sum")
                    )
                    _builder_if_973_16 = _builder_scope_enter(T.If(warp_id == 0))
                    _builder_then_973_16 = _builder_scope_enter(T.Then())
                    T.buffer_store(softmax_sum_token.buffer, iket.range_start("softmax-sum"), [0])
                    _builder_scope_exit(_builder_then_973_16)
                    _builder_scope_exit(_builder_if_973_16)
                    T.buffer_store(phase_s_full.buffer, phase_s_full ^ 1, [0])
                    T.buffer_store(phase_q.buffer, phase_q ^ 1, [0])
                    if is_first:
                        _builder_emit(reduce_sum_128(row_sum, s_chunk_buf))
                    else:
                        T.buffer_store(row_sum, row_sum[0] * acc_scale, [0])
                        _builder_emit(reduce_sum_128(row_sum, s_chunk_buf, accum=True))
                    _builder_if_982_16 = _builder_scope_enter(T.If(warp_id == 0))
                    _builder_then_982_16 = _builder_scope_enter(T.Then())
                    _builder_emit(iket.mark("softmax-phase-5"))
                    _builder_scope_exit(_builder_then_982_16)
                    _builder_scope_exit(_builder_if_982_16)
                    _builder_emit(iket.range_end(softmax_sum_token))
                    if USE_S0_S1_BARRIER:
                        T.buffer_store(phase_s0_s1.buffer, phase_s0_s1 ^ 1, [0])

                if not EPI_ON_SOFTMAX:
                    _builder_emit(softmax_corr.empty.wait(wg_id, phase_q))
                T.buffer_store(phase_q.buffer, phase_q ^ 1, [0])
                n_block_max = _builder_bind(
                    "n_block_max",
                    get_n_block_max(m_block_idx, is_causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE),
                )
                n_block_min_causal = _builder_bind(
                    "n_block_min_causal",
                    get_n_block_min_causal_mask(m_block_idx, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE)
                    if is_causal
                    else n_block_max,
                )
                _builder_emit(softmax_step(n_block_max - 1, apply_mask=is_causal, is_first=True))
                n_block_max_after_p1 = _builder_bind("n_block_max_after_p1", n_block_max - 1)
                num_phase2_blocks = _builder_bind(
                    "num_phase2_blocks", T.max(n_block_max_after_p1 - n_block_min_causal, 0)
                )
                with T.serial(num_phase2_blocks, unroll=False) as i:
                    n_block = _builder_bind("n_block", n_block_max_after_p1 - 1 - i)
                    _builder_emit(softmax_step(n_block, apply_mask=True))
                n_block_max_after_p2 = _builder_bind(
                    "n_block_max_after_p2", T.min(n_block_max_after_p1, n_block_min_causal)
                )
                with T.serial(n_block_max_after_p2, unroll=False) as i:
                    n_block = _builder_bind("n_block", n_block_max_after_p2 - 1 - i)
                    _builder_emit(softmax_step(n_block, apply_mask=False))
                if EPI_ON_SOFTMAX:
                    EPI_LD_SM = T.meta_var(32).value
                    _builder_emit(o_ready.wait(wg_id, phase_oepi))
                    _builder_emit(corr_epi.empty.wait(wg_id, phase_oepi))
                    epi_ld_tmem_token = _builder_scalar(
                        "epi_ld_tmem_token", iket.sentinel_token("epi-ld-tmem")
                    )
                    _builder_if_1032_16 = _builder_scope_enter(T.If(warp_id == 0))
                    _builder_then_1032_16 = _builder_scope_enter(T.Then())
                    T.buffer_store(epi_ld_tmem_token.buffer, iket.range_start("epi-ld-tmem"), [0])
                    _builder_scope_exit(_builder_then_1032_16)
                    _builder_scope_exit(_builder_if_1032_16)
                    acc_O_row_is_zero_or_nan = _builder_bind(
                        "acc_O_row_is_zero_or_nan",
                        tvm.tirx.any(row_sum[0] == T.float32(0.0), row_sum[0] != row_sum[0]),
                    )
                    norm_scale_sm = _builder_alloc_scalar("norm_scale_sm", "float32")
                    _builder_emit(
                        T.ptx.rcp.approx.ftz.f32(
                            norm_scale_sm,
                            T.Select(acc_O_row_is_zero_or_nan, T.float32(1.0), row_sum[0]),
                        )
                    )
                    o_row_f32_sm = _builder_name(
                        "o_row_f32_sm", T.alloc_local((EPI_LD_SM,), "float32")
                    )
                    o_row_f16_sm = _builder_name(
                        "o_row_f16_sm", T.alloc_local((EPI_LD_SM,), "float16")
                    )
                    o_row_f16_sm_u32 = _builder_name(
                        "o_row_f16_sm_u32",
                        T.decl_buffer((EPI_LD_SM // 2,), "uint32", data=o_row_f16_sm.data),
                    )
                    with T.unroll(2) as epi_q:
                        _builder_if_1047_20 = _builder_scope_enter(T.If(wg_id == epi_q))
                        _builder_then_1047_20 = _builder_scope_enter(T.Then())
                        with T.unroll(ceildiv(HEAD_DIM, EPI_LD_SM)) as d_tile:
                            d_start = _builder_bind("d_start", d_tile * EPI_LD_SM)
                            _builder_emit(
                                T.evaluate(
                                    _tmem_load(
                                        o_row_f32_sm,
                                        0,
                                        T.cuda.get_tmem_addr(
                                            T.uint32(0),
                                            0,
                                            (SMEM_PIPE_DEPTH_Q + epi_q) * MMA_N + d_start,
                                        ),
                                        EPI_LD_SM,
                                    )
                                )
                            )
                            with T.unroll(EPI_LD_SM // 2) as i:
                                _builder_emit(mul_f32x2(o_row_f32_sm, 2 * i, norm_scale_sm))
                            with T.unroll(EPI_LD_SM // 2) as i:
                                _builder_emit(
                                    T.evaluate(_cast_f32x2_f16x2(o_row_f16_sm, o_row_f32_sm, 2 * i))
                                )
                            with T.unroll(EPI_LD_SM // 8) as i:
                                _builder_emit(
                                    T.ptx.st.shared.v4.u32(
                                        O_smem.ptr_to([epi_q, tid_in_wg, d_start + i * 8]),
                                        o_row_f16_sm_u32[i * 4],
                                        o_row_f16_sm_u32[i * 4 + 1],
                                        o_row_f16_sm_u32[i * 4 + 2],
                                        o_row_f16_sm_u32[i * 4 + 3],
                                    )
                                )
                        _builder_scope_exit(_builder_then_1047_20)
                        _builder_scope_exit(_builder_if_1047_20)
                    _builder_emit(iket.range_end(epi_ld_tmem_token))
                    _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                    _builder_emit(corr_epi.full.arrive(wg_id))
                    _builder_emit(p_o_rescale.arrive(wg_id))
                    T.buffer_store(phase_oepi.buffer, phase_oepi ^ 1, [0])
                else:
                    _builder_if_1080_16 = _builder_scope_enter(T.If(tid_in_wg < BLK_M))
                    _builder_then_1080_16 = _builder_scope_enter(T.Then())
                    _builder_emit(
                        T.ptx.st.shared.f32(
                            sScale.ptr_to([ROW_SUM_BASE + tid_in_wg + wg_id * BLK_M]), row_sum[0]
                        )
                    )
                    _builder_scope_exit(_builder_then_1080_16)
                    _builder_scope_exit(_builder_if_1080_16)
                    if STATS_BAR_PAIRWISE:
                        _builder_emit(T.ptx.bar.arrive(T.uint32(1 + wg_id * 4 + warp_id), 64))
                    else:
                        _builder_emit(T.ptx.bar.arrive(T.uint32(1 + wg_id), 256))
                _builder_scope_exit(_builder_then_752_8)
                _builder_scope_exit(_builder_if_752_8)
                _builder_scope_exit(_builder_else_457_8)
                _builder_scope_exit(_builder_if_457_8)
                _builder_if_1088_8 = _builder_scope_enter(T.If(wg_id == 2))
                _builder_then_1088_8 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(64))
                if STATS_BAR_PAIRWISE:
                    _builder_emit(T.ptx.bar.sync(T.uint32(1 + 0 * 4 + warp_id), 64))
                else:
                    _builder_emit(T.ptx.bar.sync(T.uint32(1 + 0), 256))
                _builder_emit(softmax_corr.empty.arrive(0))
                if STATS_BAR_PAIRWISE:
                    _builder_emit(T.ptx.bar.sync(T.uint32(1 + 1 * 4 + warp_id), 64))
                else:
                    _builder_emit(T.ptx.bar.sync(T.uint32(1 + 1), 256))
                T.buffer_store(phase_q.buffer, phase_q ^ 1, [0])
                corr_trip_count = _builder_bind(
                    "corr_trip_count",
                    get_n_block_max(m_block_idx, is_causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE)
                    if is_causal
                    else num_kv_blocks,
                )
                with T.serial(corr_trip_count - 1, unroll=False) as i_kv:
                    with T.unroll(2) as i_q:
                        if STATS_BAR_PAIRWISE:
                            _builder_emit(T.ptx.bar.sync(T.uint32(1 + i_q * 4 + warp_id), 64))
                        else:
                            _builder_emit(T.ptx.bar.sync(T.uint32(1 + i_q), 256))
                        correction_token = _builder_scalar(
                            "correction_token", iket.sentinel_token("correction")
                        )
                        _builder_if_1112_20 = _builder_scope_enter(T.If(warp_id == 0))
                        _builder_then_1112_20 = _builder_scope_enter(T.Then())
                        T.buffer_store(correction_token.buffer, iket.range_start("correction"), [0])
                        _builder_scope_exit(_builder_then_1112_20)
                        _builder_scope_exit(_builder_if_1112_20)
                        acc_scale = _builder_alloc_scalar("acc_scale", "f32")
                        should_rescale = _builder_alloc_scalar("should_rescale", "i32")
                        _builder_if_1116_20 = _builder_scope_enter(T.If(tid_in_wg < BLK_M))
                        _builder_then_1116_20 = _builder_scope_enter(T.Then())
                        _builder_emit(
                            T.ptx.ld.shared.f32(
                                acc_scale, sScale.ptr_to([ACC_SCALE_BASE + tid_in_wg + i_q * BLK_M])
                            )
                        )
                        T.buffer_store(
                            should_rescale.buffer, T.Select(acc_scale < T.float32(1.0), 1, 0), [0]
                        )
                        _builder_scope_exit(_builder_then_1116_20)
                        _builder_else_1116_20 = _builder_scope_enter(T.Else())
                        T.buffer_store(should_rescale.buffer, 0, [0])
                        _builder_scope_exit(_builder_else_1116_20)
                        _builder_scope_exit(_builder_if_1116_20)
                        any_needs_rescale = _builder_bind(
                            "any_needs_rescale", T.cuda.any_sync(4294967295, should_rescale)
                        )
                        _builder_if_1124_20 = _builder_scope_enter(T.If(any_needs_rescale != 0))
                        _builder_then_1124_20 = _builder_scope_enter(T.Then())
                        _builder_if_1125_24 = _builder_scope_enter(T.If(tid_in_wg < BLK_M))
                        _builder_then_1125_24 = _builder_scope_enter(T.Then())
                        RESCALE_TILE = T.meta_var(16).value
                        o_row = _builder_name("o_row", T.alloc_local((RESCALE_TILE,), "float32"))
                        with T.unroll(ceildiv(HEAD_DIM, RESCALE_TILE)) as d_tile:
                            d_start = _builder_bind("d_start", d_tile * RESCALE_TILE)
                            _builder_if_1130_32 = _builder_scope_enter(T.If(d_start < HEAD_DIM))
                            _builder_then_1130_32 = _builder_scope_enter(T.Then())
                            _builder_emit(
                                T.evaluate(
                                    _tmem_load(
                                        o_row,
                                        0,
                                        T.cuda.get_tmem_addr(
                                            T.uint32(0),
                                            0,
                                            (SMEM_PIPE_DEPTH_Q + i_q) * MMA_N + d_start,
                                        ),
                                        RESCALE_TILE,
                                    )
                                )
                            )
                            with T.unroll(RESCALE_TILE // 2) as i:
                                _builder_emit(mul_f32x2(o_row, 2 * i, acc_scale))
                            _builder_emit(
                                T.evaluate(
                                    _tmem_store(
                                        o_row,
                                        0,
                                        T.cuda.get_tmem_addr(
                                            T.uint32(0),
                                            0,
                                            (SMEM_PIPE_DEPTH_Q + i_q) * MMA_N + d_start,
                                        ),
                                    )
                                )
                            )
                            _builder_scope_exit(_builder_then_1130_32)
                            _builder_scope_exit(_builder_if_1130_32)
                        _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
                        _builder_scope_exit(_builder_then_1125_24)
                        _builder_scope_exit(_builder_if_1125_24)
                        _builder_scope_exit(_builder_then_1124_20)
                        _builder_scope_exit(_builder_if_1124_20)
                        _builder_emit(p_o_rescale.arrive(i_q))
                        _builder_emit(softmax_corr.empty.arrive(1 - i_q))
                        _builder_emit(iket.range_end(correction_token))
                    T.buffer_store(phase_q.buffer, phase_q ^ 1, [0])
                _builder_emit(softmax_corr.empty.arrive(1))
                if not EPI_ON_SOFTMAX:
                    with T.unroll(2) as i_q:
                        if STATS_BAR_PAIRWISE:
                            _builder_emit(T.ptx.bar.sync(T.uint32(1 + i_q * 4 + warp_id), 64))
                        else:
                            _builder_emit(T.ptx.bar.sync(T.uint32(1 + i_q), 256))
                        row_sum = _builder_alloc_scalar("row_sum", "f32")
                        _builder_emit(
                            T.ptx.ld.shared.f32(
                                row_sum, sScale.ptr_to([ROW_SUM_BASE + tid_in_wg + i_q * BLK_M])
                            )
                        )
                        _builder_emit(softmax_corr.empty.arrive(i_q))
                        _builder_emit(o_ready.wait(i_q, phase_tmem))
                        _builder_emit(corr_epi.empty.wait(i_q, phase_tmem))
                        epi_ld_tmem_token = _builder_scalar(
                            "epi_ld_tmem_token", iket.sentinel_token("epi-ld-tmem")
                        )
                        _builder_if_1176_20 = _builder_scope_enter(T.If(warp_id == 0))
                        _builder_then_1176_20 = _builder_scope_enter(T.Then())
                        T.buffer_store(
                            epi_ld_tmem_token.buffer, iket.range_start("epi-ld-tmem"), [0]
                        )
                        _builder_scope_exit(_builder_then_1176_20)
                        _builder_scope_exit(_builder_if_1176_20)
                        acc_O_mn_row_is_zero_or_nan = _builder_bind(
                            "acc_O_mn_row_is_zero_or_nan",
                            tvm.tirx.any(row_sum == T.float32(0.0), row_sum != row_sum),
                        )
                        norm_scale = _builder_alloc_scalar("norm_scale", "float32")
                        _builder_emit(
                            T.ptx.rcp.approx.ftz.f32(
                                norm_scale,
                                T.Select(acc_O_mn_row_is_zero_or_nan, T.float32(1.0), row_sum),
                            )
                        )
                        o_row_f32 = _builder_name(
                            "o_row_f32", T.alloc_local((TMEM_EPI_LD_SIZE,), "float32")
                        )
                        o_row_f16 = _builder_name(
                            "o_row_f16", T.alloc_local((TMEM_EPI_LD_SIZE,), "float16")
                        )
                        o_row_f16_u32 = _builder_name(
                            "o_row_f16_u32",
                            T.decl_buffer((TMEM_EPI_LD_SIZE // 2,), "uint32", data=o_row_f16.data),
                        )
                        with T.unroll(ceildiv(HEAD_DIM, TMEM_EPI_LD_SIZE)) as d_tile:
                            d_start = _builder_bind("d_start", d_tile * TMEM_EPI_LD_SIZE)
                            _builder_if_1192_24 = _builder_scope_enter(T.If(d_start < HEAD_DIM))
                            _builder_then_1192_24 = _builder_scope_enter(T.Then())
                            _builder_emit(
                                T.evaluate(
                                    _tmem_load(
                                        o_row_f32,
                                        0,
                                        T.cuda.get_tmem_addr(
                                            T.uint32(0),
                                            0,
                                            (SMEM_PIPE_DEPTH_Q + i_q) * MMA_N + d_start,
                                        ),
                                        TMEM_EPI_LD_SIZE,
                                    )
                                )
                            )
                            with T.unroll(TMEM_EPI_LD_SIZE // 2) as i:
                                _builder_emit(mul_f32x2(o_row_f32, 2 * i, norm_scale))
                            with T.unroll(TMEM_EPI_LD_SIZE // 2) as i:
                                _builder_emit(
                                    T.evaluate(_cast_f32x2_f16x2(o_row_f16, o_row_f32, 2 * i))
                                )
                            with T.unroll(TMEM_EPI_LD_SIZE // 8) as i:
                                _builder_emit(
                                    T.ptx.st.shared.v4.u32(
                                        O_smem.ptr_to([i_q, tid_in_wg, d_start + i * 8]),
                                        o_row_f16_u32[i * 4],
                                        o_row_f16_u32[i * 4 + 1],
                                        o_row_f16_u32[i * 4 + 2],
                                        o_row_f16_u32[i * 4 + 3],
                                    )
                                )
                            _builder_scope_exit(_builder_then_1192_24)
                            _builder_scope_exit(_builder_if_1192_24)
                        _builder_emit(iket.range_end(epi_ld_tmem_token))
                        _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                        _builder_emit(corr_epi.full.arrive(i_q))
                        _builder_emit(p_o_rescale.arrive(i_q))
                    T.buffer_store(phase_tmem.buffer, phase_tmem ^ 1, [0])
                T.buffer_store(phase_q.buffer, phase_q ^ 1, [0])
                _builder_scope_exit(_builder_then_1088_8)
                _builder_scope_exit(_builder_if_1088_8)
                _builder_emit(scheduler.next_tile())
            # Every TMEM consumer is inside the tile loop above. Synchronize
            # all CTA warps after they leave it and before warp 0 deallocates
            # the shared TMEM allocation.
            _builder_emit(T.cuda.cta_sync())
            _builder_if_1222_4 = _builder_scope_enter(T.If((wg_id == 0) & (warp_id == 0)))
            _builder_then_1222_4 = _builder_scope_enter(T.Then())
            dealloc_tmem_addr = _builder_alloc_scalar("dealloc_tmem_addr", "uint32")
            _builder_emit(T.ptx.ld.shared.u32(dealloc_tmem_addr, tmem_addr.ptr_to([0])))
            _builder_emit(
                T.ptx[f"tcgen05.relinquish_alloc_permit.cta_group::{CTA_GROUP}.sync.aligned"]()
            )
            _builder_emit(
                T.ptx[f"tcgen05.dealloc.cta_group::{CTA_GROUP}.sync.aligned.b32"](
                    dealloc_tmem_addr, T.uint32(N_COLS_TMEM)
                )
            )
            _builder_scope_exit(_builder_then_1222_4)
            _builder_scope_exit(_builder_if_1222_4)
    return builder.get()


def get_flash_attention4_kernel(
    batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim, is_causal=False
):
    # ptxas --register-usage-level: the default 10 over-allocates under the
    # setmaxnreg caps and SPILLS (35K LDL/STL; cutedsl has 0). On the latency-
    # bound causal GQA=1 (kv32) shapes those spills sit on the critical path.
    # Clean isolated run_bench A/B (warm-L2) picks the best NON-spilling level
    # per regime (swept 3..8; >=6 re-spills): multi-wave s2048/s4096_h32kv32_c
    # want 5 (0.967->0.992, 0.974->0.993); the single-wave s1024_h32kv32_c
    # wants 4 (0.954->0.962, won all 3 rounds, s2048/s4096 unaffected ~tie).
    # After shortening the descriptor live ranges, the two benchmarked short
    # causal GQA paths (kv4/kv8) are spill-free again at level 10. A paired
    # same-process A/B selected level 10 for both; the other short causal
    # regimes keep their previously validated level. Throughput-bound
    # non-causal variants retain level 10. Read by tir support/nvcc.py via the
    # env var; set per-shape here because each bench config compiles in its own
    # process.
    # FA4_REG_LEVEL overrides (tuning).
    _reg_override = os.environ.get("FA4_REG_LEVEL", "")
    if _reg_override:
        _reg_level = _reg_override
    elif (
        is_causal
        and batch_size == 1
        and seq_len_q == 1024
        and seq_len_kv == 1024
        and num_qo_heads == 32
        and num_kv_heads in (4, 8)
        and head_dim == 128
    ):
        _reg_level = "10"
    elif is_causal and num_qo_heads == num_kv_heads:
        _reg_level = "4" if seq_len_q <= 1024 else "5"
    elif is_causal and seq_len_q <= 1024:
        _reg_level = "5"
    else:
        _reg_level = "10"
    os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = _reg_level
    # Pipeline-depth split by wave count. Single-wave causal shapes (s1024) are
    # warpgroup-handshake-bound (stall-PC: ~40% mbarrier sync); a DEEPER
    # O-accumulator TMEM pipeline (TMEM_PIPE_DEPTH 3, funded by SMEM_PIPE_DEPTH_KV
    # 2 — SMEM-neutral) hides the correction-wg handshake bubble. Clean A/B
    # (warmup=100): s1024_h32kv32_c 0.953->0.998, s1024_h32kv16_c 0.985->1.002.
    # Multi-wave shapes (s2048+) instead need KV depth 3 for the long KV stream:
    # 3/2 REGRESSES s2048 (0.991->0.951) and s4096 (0.992->0.968), so they keep
    # the default 2/3. KV depth 1 deadlocks (pipeline needs >=2 stages).
    _deep_o = is_causal and seq_len_q <= 1024
    _tmem_depth = 3 if _deep_o else TMEM_PIPE_DEPTH
    _kv_depth = 2 if _deep_o else SMEM_PIPE_DEPTH_KV
    return _build_kernel(
        BATCH_SIZE=batch_size,
        SEQ_LEN_Q=seq_len_q,
        SEQ_LEN_KV=seq_len_kv,
        NUM_QO_HEADS=num_qo_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        is_causal=is_causal,
        CTA_GROUP=1,
        TMEM_PIPE_DEPTH=_tmem_depth,
        SMEM_PIPE_DEPTH_KV=_kv_depth,
    )


def prepare_data(batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim):
    torch.manual_seed(0)
    Q = torch.randn((batch_size, seq_len_q, num_qo_heads, head_dim), dtype=torch.float16)
    K = torch.randn((batch_size, seq_len_kv, num_kv_heads, head_dim), dtype=torch.float16)
    V = torch.randn((batch_size, seq_len_kv, num_kv_heads, head_dim), dtype=torch.float16)
    O = torch.zeros((batch_size, seq_len_q, num_qo_heads, head_dim), dtype=torch.float16)
    return (Q, K, V, O)


def _builder_name(name: str, value):
    """Name a directly constructed builder value and return it."""
    try:
        return IRBuilder.name(name, value)
    except (TypeError, ValueError):
        return value


def _builder_meta(name: str, value):
    """Name resources owned by an IR-builder meta-class instance."""
    from tvm.tirx.script.builder.ir import name_meta_class_value

    name_meta_class_value(name, value)
    return value


def _builder_scalar(name: str, value, dtype: str | None = None):
    """Materialize the mutable scalar semantics used by the former parser."""
    if isinstance(value, tvm.tirx.Buffer):
        return _builder_name(name, value)
    value_type = getattr(value, "ty", None)
    if value_type is None:
        return _builder_meta(name, value)
    if value_type is not None and not isinstance(value_type, tvm.ir.PrimType):
        return _builder_bind(name, value, value.ty)
    if dtype is None:
        dtype = str(value.ty.dtype)
    scalar = T.alloc_scalar(dtype=dtype, scope="local")
    IRBuilder.name(name, scalar.scalar.buffer)
    T.buffer_store(scalar.scalar.buffer, value, [0])
    return scalar.scalar


def _builder_alloc_scalar(name: str, dtype: str):
    """Allocate a mutable scalar without inventing an initializer."""
    scalar = T.alloc_scalar(dtype=dtype, scope="local")
    IRBuilder.name(name, scalar.scalar.buffer)
    return scalar.scalar


def _builder_bind(name: str, value, type_annotation=None):
    """Emit and name an immutable builder Bind."""
    result = T.Bind(value, type_annotation)
    IRBuilder.name(name, result)
    return result


def _builder_enter(frame):
    """Enter a flat builder frame until its enclosing PrimFunc completes."""
    frame.add_callback(lambda: frame.__exit__(None, None, None))
    frame.__enter__()


def _builder_scope_enter(frame):
    """Enter a builder frame without adding Python source nesting."""
    frame.__enter__()
    return frame


def _builder_scope_exit(frame):
    """Exit a frame entered by :func:`_builder_scope_enter`."""
    frame.__exit__(None, None, None)


def _builder_emit(value):
    """Match TVMScript expression-statement emission in direct builder code."""
    if value is None or isinstance(value, tvm.ir.Var):
        return
    if tvm.ir.is_prim_expr(value) or isinstance(value, tvm.ir.Call):
        T.evaluate(value)


KERNEL_META = {"name": "flash_attention4", "category": "flashattention", "compute_capability": 10}
CONFIGS = [
    {
        "batch_size": 1,
        "seq_len": sl,
        "num_qo_heads": 32,
        "num_kv_heads": kv,
        "head_dim": 128,
        "is_causal": causal,
        "label": f"s{sl}_h32kv{kv}{('_causal' if causal else '')}",
    }
    for sl in [1024, 2048, 4096, 8192]
    for kv in [4, 8, 16, 32]
    for causal in [False, True]
]


def get_kernel(
    batch_size, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=False, **kwargs
):
    return get_flash_attention4_kernel(
        batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=is_causal
    )


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(batch_size, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=False, **kwargs):
    """Compile, run, and verify flash attention 4 kernel."""
    from tirx_kernels.runner import compile_kernel

    Q, K, V, _ = prepare_data(batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim)
    prim_func = get_flash_attention4_kernel(
        batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=is_causal
    )
    ex = compile_kernel(prim_func)
    Q_tir = Q.cuda()
    K_tir = K.cuda()
    V_tir = V.cuda()
    O_tir = torch.empty(
        (batch_size, seq_len, num_qo_heads, head_dim), dtype=torch.float16, device="cuda"
    )
    ex(Q_tir, K_tir, V_tir, O_tir)
    torch.cuda.synchronize()
    Q_t = Q.float().transpose(1, 2)
    K_t = K.float().transpose(1, 2)
    V_t = V.float().transpose(1, 2)
    if num_qo_heads != num_kv_heads:
        repeat_factor = num_qo_heads // num_kv_heads
        K_t = K_t.repeat_interleave(repeat_factor, dim=1)
        V_t = V_t.repeat_interleave(repeat_factor, dim=1)
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(Q_t, K_t.transpose(-2, -1)) * scale
    if is_causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    ref = torch.matmul(attn, V_t).transpose(1, 2).to(torch.float16)
    np.testing.assert_allclose(O_tir.cpu().numpy(), ref.cpu().numpy(), rtol=0.01, atol=0.01)


def run_gpu(
    prepared,
    *,
    warmup=None,
    repeat=None,
    timer=None,  # None inherits the global default (proton); the CuTeDSL flashattn
    # reference cannot be CUDA-graph-captured, so proton (not cudagraph_proton) is what
    # gives an honest ratio here (verified 0.994 vs event's unstable 0.97-1.38).
    **kwargs,
):
    """Benchmark flash attention 4."""
    config = dict(prepared["config"])
    batch_size = config.pop("batch_size")
    seq_len = config.pop("seq_len")
    num_qo_heads = config.pop("num_qo_heads")
    num_kv_heads = config.pop("num_kv_heads")
    head_dim = config.pop("head_dim")
    is_causal = config.pop("is_causal")
    config.update(kwargs)
    kwargs = config
    executable = prepared["executable"]

    ex = executable

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    Q, K, V, _ = prepare_data(batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim)
    Q_cuda = Q.cuda()
    K_cuda = K.cuda()
    V_cuda = V.cuda()
    O_tir = torch.empty(
        (batch_size, seq_len, num_qo_heads, head_dim), dtype=torch.float16, device="cuda"
    )
    funcs = {"tir": lambda: ex(Q_cuda, K_cuda, V_cuda, O_tir)}

    def _flashattn_sm100():
        # Flash-Attention SM100 (CuTeDSL FA4) baseline.
        #
        # CUTe-DSL hard rule (discovered by experiment): every `cute_tensor_like`
        # call must happen BEFORE `cute.compile`. Wrapping new tensors after
        # compile poisons the host-side `cuTensorMapEncodeTiled` path (it starts
        # failing ~hundreds of launches later anywhere in the process, including
        # in unrelated TIR kernels). So we wrap one FA tensor set up-front, then
        # compile exactly once using it.
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
        from flash_attn.cute.utils import AuxData

        Qi, Ki, Vi, _ = prepare_data(
            batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim
        )
        Qf = Qi.cuda().contiguous()
        Kf = Ki.cuda().contiguous()
        Vf = Vi.cuda().contiguous()
        Of = torch.zeros_like(Qf)
        q_t, q_th = cutlass_torch.cute_tensor_like(
            Qf, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )
        k_t, k_th = cutlass_torch.cute_tensor_like(
            Kf, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )
        v_t, v_th = cutlass_torch.cute_tensor_like(
            Vf, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )
        o_t, o_th = cutlass_torch.cute_tensor_like(
            Of, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
        )

        fa_fwd = FlashAttentionForwardSm100(
            head_dim=head_dim,
            head_dim_v=head_dim,
            qhead_per_kvhead=num_qo_heads // num_kv_heads,
            is_causal=is_causal,
            is_local=False,
            pack_gqa=False,
            m_block_size=128,
            n_block_size=128,
            is_persistent=True,
        )
        _stream_fa = cutlass_torch.default_stream()
        _scale_fa = 1.0 / math.sqrt(head_dim)
        compiled_fa = cute.compile(
            fa_fwd,
            q_t,
            k_t,
            v_t,
            o_t,
            None,  # mLSE
            _scale_fa,  # softmax_scale
            None,  # mCuSeqlensQ
            None,  # mCuSeqlensK
            None,  # mSeqUsedQ
            None,  # mSeqUsedK
            None,  # mPageTable
            None,  # window_size_left
            None,  # window_size_right
            None,  # learnable_sink
            None,  # descale_tensors
            None,  # blocksparse_tensors
            AuxData(),  # aux_data (FA4 takes an AuxData, not None)
            _stream_fa,  # stream (FA4 sm100 keeps stream as the LAST positional)
        )

        def run():
            compiled_fa(
                q_t,
                k_t,
                v_t,
                o_t,
                None,  # mLSE
                _scale_fa,
                None,  # mCuSeqlensQ
                None,  # mCuSeqlensK
                None,  # mSeqUsedQ
                None,  # mSeqUsedK
                None,  # mPageTable
                None,  # window_size_left
                None,  # window_size_right
                None,  # learnable_sink
                None,  # descale_tensors
                None,  # blocksparse_tensors
                AuxData(),  # aux_data (FA4 takes an AuxData, not None)
                _stream_fa,  # stream (FA4 sm100 keeps stream as the LAST positional)
            )

        # Keep the backing torch storage alive for the run's lifetime
        # (the cute tensors alias it).
        run._fa_keep_alive = (q_th, k_th, v_th, o_th, Qf, Kf, Vf, Of)
        return run

    return bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashattn_sm100": _flashattn_sm100},
        **kwargs,
    )


def run_bench(
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    is_causal=False,
    warmup=None,
    repeat=None,
    timer=None,  # None inherits the global default (proton); the CuTeDSL flashattn
    # reference cannot be CUDA-graph-captured, so proton (not cudagraph_proton) is what
    # gives an honest ratio here (verified 0.994 vs event's unstable 0.97-1.38).
    **kwargs,
):
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    prepared = prepare_bench(
        batch_size=batch_size,
        seq_len=seq_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_causal=is_causal,
        **config,
    )
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


def _parse_iket_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the annotated FA4 kernel with NVIDIA IKET"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of traced FA4 launches; setup and compilation remain outside the loop",
    )
    parser.add_argument("--output-dir", default="/tmp/fa4-iket")
    parser.add_argument(
        "--postprocess", choices=("perfetto", "json", "html", "none", "all"), default="all"
    )
    parser.add_argument("--clobber", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--keep", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-ts-cnt-per-warp", type=int, default=None)
    return parser.parse_args()


def _profile_iket_workload(args: argparse.Namespace) -> None:
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    func = get_flash_attention4_kernel(
        args.batch_size,
        args.seq_len,
        args.seq_len,
        args.num_qo_heads,
        args.num_kv_heads,
        args.head_dim,
        is_causal=args.causal,
    )
    executable = IketProfiler().compile(
        tvm.IRModule({"main": func}),
        target=tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}),
        tir_pipeline="tirx",
    )

    q, k, v, _ = prepare_data(
        args.batch_size,
        args.seq_len,
        args.seq_len,
        args.num_qo_heads,
        args.num_kv_heads,
        args.head_dim,
    )
    q, k, v = q.cuda(), k.cuda(), v.cuda()
    out = torch.empty(
        (args.batch_size, args.seq_len, args.num_qo_heads, args.head_dim),
        dtype=torch.float16,
        device="cuda",
    )

    for _ in range(args.repeat):
        executable(q, k, v, out)
    torch.cuda.synchronize()


def _print_iket_result(result: iket.IketProfileResult) -> None:
    print(f"IKET output directory: {result.output_dir}")
    for path in (*result.json_traces, *result.perfetto_traces, *result.html_reports):
        print(f"IKET artifact: {path}")


def main() -> None:
    """Profile FA4 when this kernel module is executed directly."""
    args = _parse_iket_args()
    result = iket.run(
        partial(_profile_iket_workload, args),
        output_dir=args.output_dir,
        postprocess=args.postprocess,
        clobber=args.clobber,
        timeout=args.timeout,
        keep=args.keep,
        max_ts_cnt_per_warp=args.max_ts_cnt_per_warp,
    )
    _print_iket_result(result)


if __name__ == "__main__":
    main()
