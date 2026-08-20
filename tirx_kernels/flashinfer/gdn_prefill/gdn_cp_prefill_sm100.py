# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

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
# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400).
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer SM100 chunk-parallel delta-rule prefill port.

The public operation is a four-launch chain: T precompute, M/N precompute,
chunk-state fixup, and CP prefill.  The implementation keeps those launch and
workspace boundaries explicit so the source and TIRx paths exercise the same
context-parallel decomposition.
"""

import ctypes
import math
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any
from unittest import SkipTest

import torch
import torch.nn.functional as F

import tirx_kernels.kern as K

D_HEAD = 128
T_BLOCK = 64
DESCRIPTOR_SLOT_BYTES = 128
DESCRIPTOR_SLOTS = 5
TMEM_COLUMNS = 512

# CP M/N precompute keeps both affine recurrences in TMEM and dedicates one
# warp each to the two independent UMMA issue streams.

MN_OPT_TMEM_M_COL = 0
MN_OPT_TMEM_N_COL = 128
MN_OPT_TMEM_SCRATCH_COL = 256
MN_OPT_TMEM_M_INPUT_COL = 320
MN_OPT_TMEM_N_INPUT_COL = 384
MN_OPT_TMEM_XY_COL = 448

MN_OPT_TMEM_ALLOC_BARRIER = 1
MN_OPT_TMEM_DEALLOC_BARRIER = 2

# CP prefill uses all 512 TMEM columns; K owns its barrier and shared-memory
# layout below.

PREFILL_OPT_TMEM_STATE_COL = 0
PREFILL_OPT_TMEM_Q_STATE_COL = 128
PREFILL_OPT_TMEM_STATE_INPUT_COL = 192
PREFILL_OPT_TMEM_CG0_ACC_COL = 256
PREFILL_OPT_TMEM_CG1_ACC_COL = 384
PREFILL_OPT_TMEM_SHARED_INPUT_COL = 448

PREFILL_OPT_TMEM_ALLOC_BARRIER = 1
PREFILL_OPT_T_STORE_BARRIER = 2
PREFILL_OPT_TMEM_DEALLOC_BARRIER = 3
PREFILL_OPT_INITIAL_STATE_BARRIER = 4

KERNEL_META = {"name": "gdn_cp_prefill_sm100", "category": "flashinfer", "compute_capability": 10}


BENCH_CONFIGS = [
    {
        "label": "fp16_q1_k1_v1_s2048_none_i32",
        "dtype": "float16",
        "q_heads": 1,
        "k_heads": 1,
        "v_heads": 1,
        "seq_lens": (2048,),
        "cu_seqlens_dtype": "int32",
        "state_mode": "none",
        "state_dtype": "float32",
        "indexed_state": False,
        "cp_chunk_len": None,
        "gate_baseline": 0.99,
        "scale": "auto",
        "seed": 17001,
    },
    {
        "label": "bf16_q4_k1_v1_s8193_final_f32_i64",
        "dtype": "bfloat16",
        "q_heads": 4,
        "k_heads": 1,
        "v_heads": 1,
        "seq_lens": (8193,),
        "cu_seqlens_dtype": "int64",
        "state_mode": "final",
        "state_dtype": "float32",
        "indexed_state": False,
        "cp_chunk_len": None,
        "gate_baseline": 0.99,
        "scale": "auto",
        "seed": 17002,
    },
    {
        "label": "bf16_q1_k1_v4_s9999+65530_initfinal_bf16_i32",
        "dtype": "bfloat16",
        "q_heads": 1,
        "k_heads": 1,
        "v_heads": 4,
        "seq_lens": (9999, 65530),
        "cu_seqlens_dtype": "int32",
        "state_mode": "initial_final",
        "state_dtype": "bfloat16",
        "indexed_state": False,
        "cp_chunk_len": None,
        "gate_baseline": 0.99,
        "scale": "auto",
        "seed": 17003,
    },
    {
        "label": "fp16_q16_k16_v16_s4096+4096_init_f16_i64",
        "dtype": "float16",
        "q_heads": 16,
        "k_heads": 16,
        "v_heads": 16,
        "seq_lens": (4096, 4096),
        "cu_seqlens_dtype": "int64",
        "state_mode": "initial",
        "state_dtype": "float16",
        "indexed_state": False,
        "cp_chunk_len": None,
        "gate_baseline": 0.99,
        "scale": "auto",
        "seed": 17004,
    },
    {
        "label": "bf16_q2_k2_v8_s2048x4_final_f32_i32",
        "dtype": "bfloat16",
        "q_heads": 2,
        "k_heads": 2,
        "v_heads": 8,
        "seq_lens": (2048, 2048, 2048, 2048),
        "cu_seqlens_dtype": "int32",
        "state_mode": "final",
        "state_dtype": "float32",
        "indexed_state": False,
        "cp_chunk_len": None,
        "gate_baseline": 0.99,
        "scale": "auto",
        "seed": 17005,
    },
    {
        "label": "bf16_q16_k16_v16_s128+192_indexed_bf16_i32",
        "dtype": "bfloat16",
        "q_heads": 16,
        "k_heads": 16,
        "v_heads": 16,
        "seq_lens": (128, 192),
        "cu_seqlens_dtype": "int32",
        "state_mode": "initial_final",
        "state_dtype": "bfloat16",
        "indexed_state": True,
        "cp_chunk_len": None,
        "gate_baseline": 0.99,
        "scale": "auto",
        "seed": 17006,
    },
    {
        "label": "fp16_q16_k16_v64_s192+64_initfinal_f16_i64",
        "dtype": "float16",
        "q_heads": 16,
        "k_heads": 16,
        "v_heads": 64,
        "seq_lens": (192, 64),
        "cu_seqlens_dtype": "int64",
        "state_mode": "initial_final",
        "state_dtype": "float16",
        "indexed_state": False,
        "cp_chunk_len": None,
        "gate_baseline": 0.99,
        "scale": "auto",
        "seed": 17007,
    },
]


CONFIGS = [
    *BENCH_CONFIGS,
    {
        "label": "bf16_q1_k1_v1_s96_c128_tail_i64",
        "dtype": "bfloat16",
        "q_heads": 1,
        "k_heads": 1,
        "v_heads": 1,
        "seq_lens": (96,),
        "cu_seqlens_dtype": "int64",
        "state_mode": "none",
        "state_dtype": "float32",
        "indexed_state": False,
        "cp_chunk_len": 128,
        "gate_baseline": 1.0,
        "scale": 1.0,
        "seed": 17101,
    },
    {
        "label": "fp16_q4_k1_v1_s64+192_c64_initfinal_f32_i32",
        "dtype": "float16",
        "q_heads": 4,
        "k_heads": 1,
        "v_heads": 1,
        "seq_lens": (64, 192),
        "cu_seqlens_dtype": "int32",
        "state_mode": "initial_final",
        "state_dtype": "float32",
        "indexed_state": False,
        "cp_chunk_len": 64,
        "gate_baseline": 0.9,
        "scale": 1.0,
        "seed": 17102,
    },
    {
        "label": "bf16_q1_k1_v1_s256+0_c128_final_f32_i64",
        "dtype": "bfloat16",
        "q_heads": 1,
        "k_heads": 1,
        "v_heads": 1,
        "seq_lens": (256, 0),
        "cu_seqlens_dtype": "int64",
        "state_mode": "final",
        "state_dtype": "float32",
        "indexed_state": False,
        "cp_chunk_len": 128,
        "gate_baseline": 0.9995,
        "scale": 1.0,
        "seed": 17103,
    },
]


# KERNEL_SKETCH_START: cp_delta_rule_dsl_sm100


def _device_chunk_bound(seq_idx, total, chunk_size):
    clipped = K.min(seq_idx, total)
    return clipped + (total - clipped) // chunk_size


def _load_shared_f32(dst, address):
    K.ptx.ld.shared.f32(dst, address)


def _load_global_as_f32(values, value_index, source, source_index, SOURCE_DTYPE):
    if SOURCE_DTYPE == "float32":
        K.ptx.ld.global_.f32(values[value_index], source.ptr_to([source_index]))
    else:
        bits = K.alloc_local((1,), "uint16")
        K.ptx.ld.global_.b16(bits[0], source.ptr_to([source_index]))
        K.ptx.mov.b32(values[value_index], K.cast(K.reinterpret(SOURCE_DTYPE, bits[0]), "float32"))


def _store_global_from_f32(output, output_index, value, OUTPUT_DTYPE):
    if OUTPUT_DTYPE == "float32":
        K.ptx.st.global_.f32(output.ptr_to([output_index]), value)
    else:
        K.ptx.st.global_.b16(
            output.ptr_to([output_index]), K.reinterpret("uint16", K.cast(value, OUTPUT_DTYPE))
        )


def _load_sequence_bounds(cu_seqlens, seq_idx, bounds, CU_DTYPE):
    if CU_DTYPE == "int32":
        K.ptx.ld.global_.nc.b32(bounds[0], cu_seqlens.ptr_to([seq_idx]))
        K.ptx.ld.global_.nc.b32(bounds[1], cu_seqlens.ptr_to([seq_idx + 1]))
    else:
        raw = K.alloc_local((2,), "int64")
        K.ptx.ld.global_.nc.b64(raw[0], cu_seqlens.ptr_to([seq_idx]))
        K.ptx.ld.global_.nc.b64(raw[1], cu_seqlens.ptr_to([seq_idx + 1]))
        K.ptx.mov.b32(bounds[0], K.cast(raw[0], "int32"))
        K.ptx.mov.b32(bounds[1], K.cast(raw[1], "int32"))


def _lg2_approx_ftz(value):
    result = K.alloc_local((1,), "float32")
    K.ptx.lg2.approx.ftz.f32(result[0], value)
    return result[0]


def _ex2_approx_ftz(value):
    result = K.alloc_local((1,), "float32")
    K.ptx.ex2.approx.ftz.f32(result[0], value)
    return result[0]


def _prefill_predicated_gamma(s_addr, t_addr, pred):
    s_log = K.alloc_local((1,), "float32")
    t_log = K.alloc_local((1,), "float32")
    gamma = K.alloc_local((1,), "float32")
    predicate = K.cast(pred, "uint32")
    # Both coordinates always name the allocated shared tile; the predicate
    # only masks the triangular/tail result.  Load and exponentiate directly,
    # then select zero, which preserves the source semantics without a
    # destination-predicated instruction.
    K.ptx.ld.shared.f32(s_log[0], s_addr)
    K.ptx.ld.shared.f32(t_log[0], t_addr)
    K.ptx.sub.f32(gamma[0], s_log[0], t_log[0])
    K.ptx.ex2.approx.ftz.f32(gamma[0], gamma[0])
    K.ptx.selp.f32(gamma[0], gamma[0], K.float32(0), K.ptx.pred(predicate))
    return gamma[0]


def _t_matrix_ptr(storage, row, col):
    return storage.ptr_to(row, col)


def _t_k_matrix_ptr(storage, row, col):
    return storage.ptr_to(row, col)


def _t_pack_f16x2(a, b):
    packed = K.alloc_local((1,), "uint32")
    K.evaluate(K.ptx.cvt.rn.f16x2.f32(packed[0], b, a))
    return packed[0]


def _t_sub_zero_pack_f16x2(a, b, dst, dst_index):
    negated = K.alloc_local((1,), "uint64")
    K.ptx.sub.rn.f32x2(
        negated[0], K.cuda.make_float2(K.float32(0.0), K.float32(0.0)), K.cuda.make_float2(a, b)
    )
    K.ptx.mov.b32(
        dst[dst_index], _t_pack_f16x2(K.cuda.float2_x(negated[0]), K.cuda.float2_y(negated[0]))
    )


_T_MMA_ZERO_C = [K.float32(0.0)] * 4


def _t_mma_m16n8k16_f16_zero(acc, a, b, acc_off, b_off):
    K.ptx["mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"](
        *[acc[acc_off + i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[b_off + i] for i in range(2)],
        *_T_MMA_ZERO_C,
    )


def _t_mma_m16n8k16_f16_acc(acc, a, b, acc_off, b_off):
    K.ptx["mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"](
        *[acc[acc_off + i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[b_off + i] for i in range(2)],
        *[acc[acc_off + i] for i in range(4)],
    )


def _t_mma_m16n8k16_bf16_zero(acc, a, b, acc_off, b_off):
    K.ptx["mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"](
        *[acc[acc_off + i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[b_off + i] for i in range(2)],
        *_T_MMA_ZERO_C,
    )


def _t_mma_m16n8k16_bf16_acc(acc, a, b, acc_off, b_off):
    K.ptx["mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"](
        *[acc[acc_off + i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[b_off + i] for i in range(2)],
        *[acc[acc_off + i] for i in range(4)],
    )


def _t_mma_m16n8k8_f16_zero(acc, a, b):
    K.ptx["mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"](
        *[acc[i] for i in range(4)], *[a[i] for i in range(2)], b[0], *_T_MMA_ZERO_C
    )


def _t_ldmatrix_x4(storage, base_row, base_col, lane, transpose, dst):
    lane_matrix = K.local_scalar("int32")
    K.assign(lane_matrix, lane >> 3)
    row = K.local_scalar("int32")
    K.assign(row, base_row + (lane & 7) + (lane_matrix & 1) * 8)
    col = K.local_scalar("int32")
    K.assign(col, base_col + (lane_matrix >> 1) * 8)
    K.ptx[f"ldmatrix.sync.aligned.m8n8.x4{'.trans' if transpose else ''}.shared.b16"](
        *[dst[i] for i in range(4)], _t_matrix_ptr(storage, row, col)
    )


def _t_ldmatrix_x4_k_a(smem_raw, base_row, base_col, lane, dst):
    lane_matrix = K.local_scalar("int32")
    K.assign(lane_matrix, lane >> 3)
    row = K.local_scalar("int32")
    K.assign(row, base_row + (lane & 7) + (lane_matrix & 1) * 8)
    col = K.local_scalar("int32")
    K.assign(col, base_col + (lane_matrix >> 1) * 8)
    K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
        dst[0], dst[1], dst[2], dst[3], _t_k_matrix_ptr(smem_raw, row, col)
    )


def _t_ldmatrix_x4_k_b(smem_raw, base_row, base_col, lane, dst):
    row = K.local_scalar("int32")
    K.assign(row, base_row + (lane & 7) + ((lane >> 4) & 1) * 8)
    col = K.local_scalar("int32")
    K.assign(col, base_col + ((lane >> 3) & 1) * 8)
    K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
        dst[0], dst[1], dst[2], dst[3], _t_k_matrix_ptr(smem_raw, row, col)
    )


def _t_stmatrix_x4(storage, base_row, base_col, lane, src):
    lane_matrix = K.local_scalar("int32")
    K.assign(lane_matrix, lane >> 3)
    row = K.local_scalar("int32")
    K.assign(row, base_row + (lane & 7) + (lane_matrix & 1) * 8)
    col = K.local_scalar("int32")
    K.assign(col, base_col + (lane_matrix >> 1) * 8)
    K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
        _t_matrix_ptr(storage, row, col), *[src[i] for i in range(4)]
    )


def _t_store_t_fragment(
    inverse_frag, beta_storage, t, t_base, valid_len, local_warp, lane, n_group, IO_DTYPE
):
    row_base = K.local_scalar("int32")
    K.assign(row_base, local_warp * 16 + (lane >> 2))
    col_base = K.local_scalar("int32")
    K.assign(col_base, n_group * 16 + (lane & 3) * 2)
    with K.unroll(4) as pair:
        row = K.local_scalar("int32")
        K.assign(row, row_base + (pair & 1) * 8)
        col = K.local_scalar("int32")
        K.assign(col, col_base + (pair >> 1) * 8)
        word = K.local_scalar("uint32")
        K.assign(word, inverse_frag[pair])
        inverse_lo = K.local_scalar("float32")
        K.assign(
            inverse_lo,
            K.cast(K.reinterpret("float16", K.cast(word & K.uint32(0xFFFF), "uint16")), "float32"),
        )
        inverse_hi = K.local_scalar("float32")
        K.assign(
            inverse_hi, K.cast(K.reinterpret("float16", K.cast(word >> 16, "uint16")), "float32")
        )
        output_lo = K.local_scalar("float32")
        K.assign(output_lo, 0.0)
        output_hi = K.local_scalar("float32")
        K.assign(output_hi, 0.0)
        beta_value = K.local_scalar("float32")
        with K.If(K.And(row < valid_len, col < valid_len)):
            with K.Then():
                _load_shared_f32(beta_value, beta_storage.ptr_to([col]))
                K.assign(output_lo, -beta_value * inverse_lo)
        with K.If(K.And(row < valid_len, col + 1 < valid_len)):
            with K.Then():
                _load_shared_f32(beta_value, beta_storage.ptr_to([col + 1]))
                K.assign(output_hi, -beta_value * inverse_hi)
        K.ptx.st.global_.b16(
            t.ptr_to([t_base + col * T_BLOCK + row]),
            K.reinterpret("uint16", K.cast(output_lo, IO_DTYPE)),
        )
        K.ptx.st.global_.b16(
            t.ptr_to([t_base + (col + 1) * T_BLOCK + row]),
            K.reinterpret("uint16", K.cast(output_hi, IO_DTYPE)),
        )


def _t_inverse_8_to_16(storage, block16, lane):
    a = K.alloc_local((2,), "uint32")
    b = K.alloc_local((1,), "uint32")
    acc = K.alloc_local((4,), "float32")
    word = K.alloc_local((1,), "uint32")
    K.ptx.ldmatrix.sync.aligned.m8n8.x1.shared.b16(
        word[0], _t_matrix_ptr(storage, block16 + 8 + (lane & 7), block16 + 8)
    )
    K.ptx.mov.b32(a[0], word[0])
    K.ptx.mov.b32(a[1], word[0])
    K.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(
        b[0], _t_matrix_ptr(storage, block16 + 8 + (lane & 7), block16)
    )
    _t_mma_m16n8k8_f16_zero(acc, a, b)
    _t_sub_zero_pack_f16x2(acc[0], acc[1], a, 0)
    _t_sub_zero_pack_f16x2(acc[2], acc[3], a, 1)
    K.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(
        b[0], _t_matrix_ptr(storage, block16 + (lane & 7), block16)
    )
    _t_mma_m16n8k8_f16_zero(acc, a, b)
    K.assign(word[0], _t_pack_f16x2(acc[0], acc[1]))
    K.ptx.stmatrix.sync.aligned.m8n8.x1.shared.b16(
        _t_matrix_ptr(storage, block16 + 8 + (lane & 7), block16), word[0]
    )


def _t_inverse_16_to_32(storage, block32, lane):
    a = K.alloc_local((4,), "uint32")
    b = K.alloc_local((4,), "uint32")
    acc = K.alloc_local((8,), "float32")
    packed = K.alloc_local((4,), "uint32")
    _t_ldmatrix_x4(storage, block32 + 16, block32 + 16, lane, False, a)
    _t_ldmatrix_x4(storage, block32 + 16, block32, lane, True, b)
    _t_mma_m16n8k16_f16_zero(acc, a, b, 0, 0)
    _t_mma_m16n8k16_f16_zero(acc, a, b, 4, 2)
    with K.unroll(4) as pair:
        _t_sub_zero_pack_f16x2(acc[pair * 2], acc[pair * 2 + 1], a, pair)
    _t_ldmatrix_x4(storage, block32, block32, lane, True, b)
    _t_mma_m16n8k16_f16_zero(acc, a, b, 0, 0)
    _t_mma_m16n8k16_f16_zero(acc, a, b, 4, 2)
    with K.unroll(4) as pair:
        K.ptx.mov.b32(packed[pair], _t_pack_f16x2(acc[pair * 2], acc[pair * 2 + 1]))
    _t_stmatrix_x4(storage, block32 + 16, block32, lane, packed)


def _t_inverse_32_to_64(storage, local_warp, lane):
    # CollectiveInverse splits the final 32-wide K reduction between warp
    # pairs.  Each partial result is rounded to FP16 before the x=1 warp adds
    # the x=0 contribution in FP16.
    x = K.local_scalar("int32")
    K.assign(x, local_warp >> 1)
    y = K.local_scalar("int32")
    K.assign(y, local_warp & 1)
    row_base = K.local_scalar("int32")
    K.assign(row_base, 32 + y * 16)
    split = K.local_scalar("int32")
    K.assign(split, x * 16)
    d0 = K.alloc_local((4,), "uint32")
    d1 = K.alloc_local((4,), "uint32")
    c0 = K.alloc_local((4,), "uint32")
    c1 = K.alloc_local((4,), "uint32")
    ainv0 = K.alloc_local((4,), "uint32")
    ainv1 = K.alloc_local((4,), "uint32")
    temp = K.alloc_local((8,), "float32")
    output = K.alloc_local((16,), "float32")
    temp_f16 = K.alloc_local((4,), "uint32")
    output0_f16 = K.alloc_local((4,), "uint32")
    output1_f16 = K.alloc_local((4,), "uint32")
    reduced0 = K.alloc_local((4,), "uint32")
    reduced1 = K.alloc_local((4,), "uint32")

    _t_ldmatrix_x4(storage, row_base, 32, lane, False, d0)
    _t_ldmatrix_x4(storage, row_base, 48, lane, False, d1)
    _t_ldmatrix_x4(storage, 32, split, lane, True, c0)
    _t_ldmatrix_x4(storage, 48, split, lane, True, c1)
    _t_mma_m16n8k16_f16_zero(temp, d0, c0, 0, 0)
    _t_mma_m16n8k16_f16_zero(temp, d0, c0, 4, 2)
    _t_mma_m16n8k16_f16_acc(temp, d1, c1, 0, 0)
    _t_mma_m16n8k16_f16_acc(temp, d1, c1, 4, 2)
    with K.unroll(4) as pair:
        _t_sub_zero_pack_f16x2(temp[pair * 2], temp[pair * 2 + 1], temp_f16, pair)

    _t_ldmatrix_x4(storage, split, 0, lane, True, ainv0)
    _t_ldmatrix_x4(storage, split, 16, lane, True, ainv1)
    _t_mma_m16n8k16_f16_zero(output, temp_f16, ainv0, 0, 0)
    _t_mma_m16n8k16_f16_zero(output, temp_f16, ainv0, 4, 2)
    _t_mma_m16n8k16_f16_zero(output, temp_f16, ainv1, 8, 0)
    _t_mma_m16n8k16_f16_zero(output, temp_f16, ainv1, 12, 2)
    with K.unroll(4) as pair:
        K.ptx.mov.b32(output0_f16[pair], _t_pack_f16x2(output[pair * 2], output[pair * 2 + 1]))
        K.ptx.mov.b32(
            output1_f16[pair], _t_pack_f16x2(output[8 + pair * 2], output[8 + pair * 2 + 1])
        )

    K.cuda.cta_sync()
    with K.If(x == 0):
        with K.Then():
            _t_stmatrix_x4(storage, row_base, 0, lane, output0_f16)
            _t_stmatrix_x4(storage, row_base, 16, lane, output1_f16)
    K.cuda.cta_sync()
    with K.If(x == 1):
        with K.Then():
            _t_ldmatrix_x4(storage, row_base, 0, lane, False, reduced0)
            _t_ldmatrix_x4(storage, row_base, 16, lane, False, reduced1)
            with K.unroll(4) as pair:
                sum_lo0 = K.local_scalar("uint16")
                sum_hi0 = K.local_scalar("uint16")
                sum_lo1 = K.local_scalar("uint16")
                sum_hi1 = K.local_scalar("uint16")
                K.ptx.add.f16(
                    sum_lo0,
                    K.cast(output0_f16[pair] & K.uint32(0xFFFF), "uint16"),
                    K.cast(reduced0[pair] & K.uint32(0xFFFF), "uint16"),
                )
                K.ptx.add.f16(
                    sum_hi0,
                    K.cast(output0_f16[pair] >> 16, "uint16"),
                    K.cast(reduced0[pair] >> 16, "uint16"),
                )
                K.ptx.add.f16(
                    sum_lo1,
                    K.cast(output1_f16[pair] & K.uint32(0xFFFF), "uint16"),
                    K.cast(reduced1[pair] & K.uint32(0xFFFF), "uint16"),
                )
                K.ptx.add.f16(
                    sum_hi1,
                    K.cast(output1_f16[pair] >> 16, "uint16"),
                    K.cast(reduced1[pair] >> 16, "uint16"),
                )
                K.ptx.mov.b32(
                    output0_f16[pair], K.cast(sum_lo0, "uint32") | (K.cast(sum_hi0, "uint32") << 16)
                )
                K.ptx.mov.b32(
                    output1_f16[pair], K.cast(sum_lo1, "uint32") | (K.cast(sum_hi1, "uint32") << 16)
                )
            _t_stmatrix_x4(storage, row_base, 0, lane, output0_f16)
            _t_stmatrix_x4(storage, row_base, 16, lane, output1_f16)


def _mn_opt_pack_iox2(a, b, IO_DTYPE):
    packed = K.alloc_local((1,), "uint32")
    if IO_DTYPE == "float16":
        K.evaluate(K.ptx.cvt.rn.f16x2.f32(packed[0], b, a))
    else:
        K.evaluate(K.ptx.cvt.rn.bf16x2.f32(packed[0], b, a))
    return packed[0]


def _mn_opt_unpack_io_lo(word, IO_DTYPE):
    raw: K.uint16 = K.cast(K.bitwise_and(word, K.uint32(0xFFFF)), "uint16")
    if IO_DTYPE == "float16":
        return K.cast(K.reinterpret("float16", raw), "float32")
    return K.cast(K.reinterpret("bfloat16", raw), "float32")


def _mn_opt_unpack_io_hi(word, IO_DTYPE):
    raw: K.uint16 = K.cast(K.shift_right(word, K.uint32(16)), "uint16")
    if IO_DTYPE == "float16":
        return K.cast(K.reinterpret("float16", raw), "float32")
    return K.cast(K.reinterpret("bfloat16", raw), "float32")


def _mn_opt_tmem_row_bits(thread):
    return K.bitwise_and(thread << 16, K.int32(0x600000))


def _mn_opt_tmem_ld_matrix_sub(tmem_base, column, thread, sub, values, value_offset):
    addr = K.local_scalar("int32")
    K.assign(addr, tmem_base + _mn_opt_tmem_row_bits(thread) + column + sub * 32)
    K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
        *[values[value_offset + i] for i in range(32)], K.cast(addr, "uint32")
    )


def _mn_opt_tmem_st_matrix_sub(tmem_base, column, thread, sub, values, value_offset):
    addr = K.local_scalar("int32")
    K.assign(addr, tmem_base + _mn_opt_tmem_row_bits(thread) + column + sub * 32)
    K.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
        K.cast(addr, "uint32"), *[values[value_offset + i] for i in range(32)]
    )


def _mn_opt_tmem_st_matrix_io_sub(tmem_base, column, thread, sub, values):
    addr = K.local_scalar("int32")
    K.assign(addr, tmem_base + _mn_opt_tmem_row_bits(thread) + column + sub * 16)
    K.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
        K.cast(addr, "uint32"), *[values[sub * 16 + i] for i in range(16)]
    )


def _mn_opt_tmem_ld_128x64(tmem_base, column, thread, values):
    addr = K.local_scalar("int32")
    K.assign(addr, tmem_base + _mn_opt_tmem_row_bits(thread) + column)
    K.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
        *[values[i] for i in range(32)], K.cast(addr, "uint32")
    )
    K.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
        *[values[32 + i] for i in range(32)], K.cast(addr + K.int32(0x100000), "uint32")
    )


def _mn_opt_tmem_st_128x64_io_half(tmem_base, column, thread, half, values):
    addr = K.local_scalar("int32")
    K.assign(addr, tmem_base + _mn_opt_tmem_row_bits(thread) + column)
    K.ptx["tcgen05.st.sync.aligned.16x128b.x8.b32"](
        K.cast(addr + half * K.int32(0x100000), "uint32"),
        *[values[half * 16 + i] for i in range(16)],
    )


def _mn_opt_tmem_st_128x64_io(tmem_base, column, thread, values):
    _mn_opt_tmem_st_128x64_io_half(tmem_base, column, thread, 0, values)
    _mn_opt_tmem_st_128x64_io_half(tmem_base, column, thread, 1, values)


def _mn_opt_load_128x64_fragment(tile, stage, thread, values):
    for half_idx in range(2):
        for band in range(4):
            row = K.local_scalar("int32")
            K.assign(row, (thread & 7) | ((thread & 16) >> 1) | (thread & 64) | (band << 4))
            col = K.local_scalar("int32")
            K.assign(col, (thread & 40) | (half_idx << 4))
            offset = half_idx * 16 + band * 4
            K.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                *[values[offset + i] for i in range(4)], tile[stage].ptr_to(row, col)
            )


def _mn_opt_store_128x64_fragment(tile, stage, thread, values):
    for half_idx in range(2):
        for band in range(4):
            row = K.local_scalar("int32")
            K.assign(row, (thread & 7) | ((thread & 16) >> 1) | (thread & 64) | (band << 4))
            col = K.local_scalar("int32")
            K.assign(col, (thread & 40) | (half_idx << 4))
            offset = half_idx * 16 + band * 4
            K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                tile[stage].ptr_to(row, col), *[values[offset + i] for i in range(4)]
            )


def _mn_opt_initialize_matrix(tmem_base, column, thread, identity):
    values = K.alloc_local((32,), "float32")
    for sub in range(4):
        with K.unroll(32) as i:
            col = K.local_scalar("int32")
            K.assign(col, sub * 32 + i)
            K.ptx.mov.b32(
                values[i],
                K.if_then_else(K.And(identity, thread == col), K.float32(1.0), K.float32(0.0)),
            )
        _mn_opt_tmem_st_matrix_sub(tmem_base, column, thread, sub, values, 0)
    K.ptx.tcgen05.wait__st.sync.aligned()


def _mn_opt_scale_matrix(tmem_base, column, thread, scale):
    values = K.alloc_local((32,), "float32")
    for sub in range(4):
        _mn_opt_tmem_ld_matrix_sub(tmem_base, column, thread, sub, values, 0)
        with K.unroll(32) as i:
            K.ptx.mov.b32(values[i], values[i] * scale)
        _mn_opt_tmem_st_matrix_sub(tmem_base, column, thread, sub, values, 0)
    K.ptx.tcgen05.wait__st.sync.aligned()


def _mn_opt_matrix_to_io_input(tmem_base, src_column, dst_column, thread, IO_DTYPE):
    values = K.alloc_local((32,), "float32")
    packed = K.alloc_local((64,), "uint32")
    for sub in range(4):
        _mn_opt_tmem_ld_matrix_sub(tmem_base, src_column, thread, sub, values, 0)
        with K.unroll(16) as pair:
            K.ptx.mov.b32(
                packed[sub * 16 + pair],
                _mn_opt_pack_iox2(values[pair * 2], values[pair * 2 + 1], IO_DTYPE),
            )
        _mn_opt_tmem_st_matrix_io_sub(tmem_base, dst_column, thread, sub, packed)
    K.ptx.tcgen05.wait__st.sync.aligned()


def _mn_opt_scratch_to_io_input(tmem_base, dst_column, thread, IO_DTYPE):
    values = K.alloc_local((64,), "float32")
    packed = K.alloc_local((32,), "uint32")
    _mn_opt_tmem_ld_128x64(tmem_base, MN_OPT_TMEM_SCRATCH_COL, thread, values)
    with K.unroll(32) as pair:
        K.ptx.mov.b32(
            packed[pair], _mn_opt_pack_iox2(values[pair * 2], values[pair * 2 + 1], IO_DTYPE)
        )
    _mn_opt_tmem_st_128x64_io(tmem_base, dst_column, thread, packed)
    K.ptx.tcgen05.wait__st.sync.aligned()


def _mn_opt_materialize_x(tmem_base, x_tile, stage, thread, IO_DTYPE):
    values = K.alloc_local((64,), "float32")
    packed = K.alloc_local((32,), "uint32")
    _mn_opt_tmem_ld_128x64(tmem_base, MN_OPT_TMEM_XY_COL, thread, values)
    K.ptx.tcgen05.wait__ld.sync.aligned()
    with K.unroll(32) as pair:
        K.ptx.mov.b32(
            packed[pair], _mn_opt_pack_iox2(values[pair * 2], values[pair * 2 + 1], IO_DTYPE)
        )
    _mn_opt_store_128x64_fragment(x_tile, stage, thread, packed)
    K.ptx.fence.proxy.async_.shared__cta()


def _mn_opt_store_matrix_global(tmem_base, column, output, base, thread):
    values = K.alloc_local((32,), "uint32")
    thread_base = K.local_scalar("int64")
    K.assign(thread_base, base + K.cast(thread, "int64") * D_HEAD)
    for sub in range(4):
        _mn_opt_tmem_ld_matrix_sub(tmem_base, column, thread, sub, values, 0)
        for vector in range(8):
            K.ptx["st.global.L1::no_allocate.v4.b32"](
                output.ptr_to([thread_base + sub * 32 + vector * 4]),
                values[vector * 4],
                values[vector * 4 + 1],
                values[vector * 4 + 2],
                values[vector * 4 + 3],
            )


def _mn_opt_smem_desc_k(smem_addr):
    desc_lo = K.cast(
        K.bitwise_and(K.shift_right(smem_addr, K.uint32(4)), K.uint32(0x3FFF)), "uint64"
    )
    return K.bitwise_or(K.uint64(0x4000404000010000), desc_lo)


def _mn_opt_smem_desc_mn(smem_addr):
    desc_lo = K.cast(
        K.bitwise_and(K.shift_right(smem_addr, K.uint32(4)), K.uint32(0x3FFF)), "uint64"
    )
    return K.bitwise_or(K.uint64(0x4000404002000000), desc_lo)


def _mn_opt_mma_descriptor(base, IO_DTYPE):
    return K.uint32(base + (0x480 if IO_DTYPE == "bfloat16" else 0))


_MN_OPT_MMA_CHAIN = "tcgen05.mma.cta_group::1.kind::f16"
_MN_OPT_ZERO_MASKS = [K.uint32(0)] * 4


def _mn_opt_mma_commit(barrier):
    K.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
        barrier, pred=K.cuda.elect_sync()
    )


# CP state fixup: the UTC64/UTC128 source specializations share one pipeline
# skeleton but use different TMEM copies, register ownership, and M-ring depth.
_FIXUP_TMA_G2S_4D = (
    "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
)
_FIXUP_MMA_TF32 = "tcgen05.mma.cta_group::1.kind::tf32"
_FIXUP_ZERO_MASKS = [K.uint32(0)] * 4
_FIXUP_TMEM_ACC_COL = 0
_FIXUP_TMEM_OPERAND_COL = 128
_FIXUP_TMEM_COLUMNS = 256
_FIXUP_TMEM_ALLOC_BARRIER = 1


def _fixup_tmem_row_bits(thread):
    return K.bitwise_and(thread << 16, K.int32(0x600000))


def _fixup_tmem_ld_sub(tmem_base, column, thread, sub, words, value_offset, ROWS):
    addr = K.local_scalar("int32")
    K.assign(addr, tmem_base + _fixup_tmem_row_bits(thread) + column + sub * 32)
    if ROWS == 128:
        K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
            *[words[value_offset + i] for i in range(32)], K.cast(addr, "uint32")
        )
    else:
        K.ptx["tcgen05.ld.sync.aligned.16x32bx2.x16.b32"](
            *[words[value_offset + i] for i in range(16)], K.cast(addr, "uint32"), 16
        )


def _fixup_tmem_st_sub(tmem_base, column, thread, sub, words, value_offset, ROWS):
    addr = K.local_scalar("int32")
    K.assign(addr, tmem_base + _fixup_tmem_row_bits(thread) + column + sub * 32)
    if ROWS == 128:
        K.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
            K.cast(addr, "uint32"), *[words[value_offset + i] for i in range(32)]
        )
    else:
        K.ptx["tcgen05.st.sync.aligned.16x32bx2.x16.b32"](
            K.cast(addr, "uint32"), 16, *[words[value_offset + i] for i in range(16)]
        )


def _fixup_tmem_ld(tmem_base, column, thread, words, ROWS):
    values_per_sub = 32 if ROWS == 128 else 16
    for sub in range(4):
        _fixup_tmem_ld_sub(tmem_base, column, thread, sub, words, sub * values_per_sub, ROWS)


def _fixup_tmem_st(tmem_base, column, thread, words, ROWS):
    values_per_sub = 32 if ROWS == 128 else 16
    for sub in range(4):
        _fixup_tmem_st_sub(tmem_base, column, thread, sub, words, sub * values_per_sub, ROWS)


def _fixup_load_n_to_tmem(tmem_base, n_tile, thread, ROWS):
    words = K.alloc_local((128,), "uint32")
    if ROWS == 128:
        for sub in range(4):
            row = K.local_scalar("int32")
            K.assign(row, thread)
            for vector in range(8):
                K.ptx.ld.shared.v4.b32(
                    words[sub * 32 + vector * 4],
                    words[sub * 32 + vector * 4 + 1],
                    words[sub * 32 + vector * 4 + 2],
                    words[sub * 32 + vector * 4 + 3],
                    n_tile[sub].ptr_to(row, vector * 4),
                )
    else:
        for sub in range(4):
            row = K.local_scalar("int32")
            K.assign(row, (thread >> 5) * 16 + (thread & 15))
            col = K.local_scalar("int32")
            K.assign(col, thread & 16)
            for vector in range(4):
                K.ptx.ld.shared.v4.b32(
                    words[sub * 16 + vector * 4],
                    words[sub * 16 + vector * 4 + 1],
                    words[sub * 16 + vector * 4 + 2],
                    words[sub * 16 + vector * 4 + 3],
                    n_tile[sub].ptr_to(row, col + vector * 4),
                )
    _fixup_tmem_st(tmem_base, _FIXUP_TMEM_ACC_COL, thread, words, ROWS)


def _fixup_load_initial_to_tmem(tmem_base, initial_state, base, thread, ROWS, STATE_DTYPE):
    values = K.alloc_local((128,), "float32")
    words = values.view("uint32")
    if ROWS == 128:
        for sub in range(4):
            with K.unroll(32) as i:
                _load_global_as_f32(
                    values,
                    sub * 32 + i,
                    initial_state,
                    base + K.cast(thread, "int64") * D_HEAD + sub * 32 + i,
                    STATE_DTYPE,
                )
    else:
        local_row = K.local_scalar("int32")
        K.assign(local_row, (thread >> 5) * 16 + (thread & 15))
        col_base = K.local_scalar("int32")
        K.assign(col_base, thread & 16)
        for sub in range(4):
            with K.unroll(16) as i:
                _load_global_as_f32(
                    values,
                    sub * 16 + i,
                    initial_state,
                    base + K.cast(local_row, "int64") * D_HEAD + col_base + sub * 32 + i,
                    STATE_DTYPE,
                )
    _fixup_tmem_st(tmem_base, _FIXUP_TMEM_ACC_COL, thread, words, ROWS)


def _fixup_store_f32(words, output, base, thread, ROWS):
    if ROWS == 128:
        for sub in range(4):
            thread_base = K.local_scalar("int64")
            K.assign(thread_base, base + K.cast(thread, "int64") * D_HEAD + sub * 32)
            for vector in range(8):
                K.ptx.st.global_.v4.b32(
                    output.ptr_to([thread_base + vector * 4]),
                    words[sub * 32 + vector * 4],
                    words[sub * 32 + vector * 4 + 1],
                    words[sub * 32 + vector * 4 + 2],
                    words[sub * 32 + vector * 4 + 3],
                )
    else:
        local_row = K.local_scalar("int32")
        K.assign(local_row, (thread >> 5) * 16 + (thread & 15))
        col_base = K.local_scalar("int32")
        K.assign(col_base, thread & 16)
        for sub in range(4):
            thread_base = K.local_scalar("int64")
            K.assign(
                thread_base, (base + K.cast(local_row, "int64") * D_HEAD + col_base + sub * 32)
            )
            for vector in range(4):
                K.ptx.st.global_.v4.b32(
                    output.ptr_to([thread_base + vector * 4]),
                    words[sub * 16 + vector * 4],
                    words[sub * 16 + vector * 4 + 1],
                    words[sub * 16 + vector * 4 + 2],
                    words[sub * 16 + vector * 4 + 3],
                )


def _fixup_store_state(values, output, base, thread, ROWS, STATE_DTYPE):
    if STATE_DTYPE == "float32":
        _fixup_store_f32(values.view("uint32"), output, base, thread, ROWS)
    else:
        packed = K.alloc_local((64,), "uint32")
        if ROWS == 128:
            for sub in range(4):
                with K.unroll(16) as pair:
                    K.ptx.mov.b32(
                        packed[sub * 16 + pair],
                        _mn_opt_pack_iox2(
                            values[sub * 32 + pair * 2],
                            values[sub * 32 + pair * 2 + 1],
                            STATE_DTYPE,
                        ),
                    )
                thread_base = K.local_scalar("int64")
                K.assign(thread_base, base + K.cast(thread, "int64") * D_HEAD + sub * 32)
                for vector in range(4):
                    K.ptx["st.global.L1::no_allocate.v4.b32"](
                        output.ptr_to([thread_base + vector * 8]),
                        packed[sub * 16 + vector * 4],
                        packed[sub * 16 + vector * 4 + 1],
                        packed[sub * 16 + vector * 4 + 2],
                        packed[sub * 16 + vector * 4 + 3],
                    )
        else:
            local_row = K.local_scalar("int32")
            K.assign(local_row, (thread >> 5) * 16 + (thread & 15))
            col_base = K.local_scalar("int32")
            K.assign(col_base, thread & 16)
            for sub in range(4):
                with K.unroll(8) as pair:
                    K.ptx.mov.b32(
                        packed[sub * 8 + pair],
                        _mn_opt_pack_iox2(
                            values[sub * 16 + pair * 2],
                            values[sub * 16 + pair * 2 + 1],
                            STATE_DTYPE,
                        ),
                    )
                thread_base = K.local_scalar("int64")
                K.assign(
                    thread_base, base + K.cast(local_row, "int64") * D_HEAD + col_base + sub * 32
                )
                for vector in range(2):
                    K.ptx["st.global.L1::no_allocate.v4.b32"](
                        output.ptr_to([thread_base + vector * 8]),
                        packed[sub * 8 + vector * 4],
                        packed[sub * 8 + vector * 4 + 1],
                        packed[sub * 8 + vector * 4 + 2],
                        packed[sub * 8 + vector * 4 + 3],
                    )


def _fixup_acc_to_tf32(tmem_base, thread, ROWS):
    values = K.alloc_local((128,), "float32")
    words = values.view("uint32")
    tf32_words = K.alloc_local((128,), "uint32")
    _fixup_tmem_ld(tmem_base, _FIXUP_TMEM_ACC_COL, thread, words, ROWS)
    K.ptx.tcgen05.wait__ld.sync.aligned()
    if ROWS == 128:
        with K.unroll(128) as i:
            K.ptx.cvt.rna.tf32.f32(tf32_words[i], values[i])
    else:
        with K.unroll(64) as i:
            K.ptx.cvt.rna.tf32.f32(tf32_words[i], values[i])
    _fixup_tmem_st(tmem_base, _FIXUP_TMEM_OPERAND_COL, thread, tf32_words, ROWS)
    K.ptx.tcgen05.wait__st.sync.aligned()


def _fixup_smem_desc_m(smem_addr, M_OFF):
    desc_lo = K.cast(
        K.bitwise_and(K.shift_right(smem_addr + K.uint32(M_OFF), K.uint32(4)), K.uint32(0x3FFF)),
        "uint64",
    )
    return K.bitwise_or(K.uint64(0x2000402004000000), desc_lo)


def _fixup_mma(tmem_base, m_desc, m_stage, ROWS, done_full, ready_empty, m_empty):
    instr_desc = K.local_scalar("uint32")
    K.assign(instr_desc, K.uint32(K.if_then_else(ROWS == 128, 0x08110910, 0x04110910)))
    for kphase in range(16):
        for n_half in range(2):
            K.evaluate(
                K.ptx[_FIXUP_MMA_TF32](
                    K.cast(tmem_base + n_half * 64, "uint32"),
                    K.cast(tmem_base + _FIXUP_TMEM_OPERAND_COL + kphase * 8, "uint32"),
                    m_desc + K.uint64(m_stage * 4096 + kphase * 64 + n_half * 2048),
                    instr_desc,
                    *_FIXUP_ZERO_MASKS,
                    True,
                    pred=K.cuda.elect_sync(),
                )
            )
    _mn_opt_mma_commit(done_full)
    _mn_opt_mma_commit(ready_empty)
    _mn_opt_mma_commit(m_empty)


def _mn_opt_mma_ss_128x64_k64(tmem_d, a_desc_base, b_desc_base, full_barrier, IO_DTYPE):
    descriptor = K.local_scalar("uint32")
    K.assign(descriptor, _mn_opt_mma_descriptor(0x08108010, IO_DTYPE))
    for kphase in range(4):
        K.evaluate(
            K.ptx[_MN_OPT_MMA_CHAIN](
                K.cast(tmem_d, "uint32"),
                a_desc_base + K.uint64(kphase * 128),
                b_desc_base + K.uint64(kphase * 2),
                descriptor,
                *_MN_OPT_ZERO_MASKS,
                K.ptx.pred(K.if_then_else(kphase == 0, 0, 1)),
                pred=K.cuda.elect_sync(),
            )
        )
    _mn_opt_mma_commit(full_barrier)


def _mn_opt_mma_ts_128x64_k128(tmem_d, tmem_a, b_desc_base, full_barrier, IO_DTYPE):
    descriptor = K.local_scalar("uint32")
    K.assign(descriptor, _mn_opt_mma_descriptor(0x08100010, IO_DTYPE))
    for kphase in range(8):
        phase_off = K.local_scalar("uint64")
        K.assign(phase_off, K.uint64((kphase % 4) * 2 + (kphase // 4) * 512))
        K.evaluate(
            K.ptx[_MN_OPT_MMA_CHAIN](
                K.cast(tmem_d, "uint32"),
                K.cast(tmem_a + kphase * 8, "uint32"),
                b_desc_base + phase_off,
                descriptor,
                *_MN_OPT_ZERO_MASKS,
                K.ptx.pred(K.if_then_else(kphase == 0, 0, 1)),
                pred=K.cuda.elect_sync(),
            )
        )
    _mn_opt_mma_commit(full_barrier)


def _mn_opt_mma_ss_128x128_k64(tmem_d, a_desc_base, b_desc_base, full_barrier, IO_DTYPE):
    descriptor = K.local_scalar("uint32")
    K.assign(descriptor, _mn_opt_mma_descriptor(0x08218010, IO_DTYPE))
    for kphase in range(4):
        K.evaluate(
            K.ptx[_MN_OPT_MMA_CHAIN](
                K.cast(tmem_d, "uint32"),
                a_desc_base + K.uint64(kphase * 128),
                b_desc_base + K.uint64(kphase * 128),
                descriptor,
                *_MN_OPT_ZERO_MASKS,
                True,
                pred=K.cuda.elect_sync(),
            )
        )
    _mn_opt_mma_commit(full_barrier)


def _mn_opt_mma_ts_128x128_k64(tmem_d, tmem_a, b_desc_base, full_barrier, IO_DTYPE):
    descriptor = K.local_scalar("uint32")
    K.assign(descriptor, _mn_opt_mma_descriptor(0x08210010, IO_DTYPE))
    for kphase in range(4):
        K.evaluate(
            K.ptx[_MN_OPT_MMA_CHAIN](
                K.cast(tmem_d, "uint32"),
                K.cast(tmem_a + kphase * 8, "uint32"),
                b_desc_base + K.uint64(kphase * 128),
                descriptor,
                *_MN_OPT_ZERO_MASKS,
                True,
                pred=K.cuda.elect_sync(),
            )
        )
    _mn_opt_mma_commit(full_barrier)


_MN_OPT_TMA_G2S_3D = (
    "cp.async.bulk.tensor.3d.shared::cta.global.tile"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
)
_MN_OPT_TMA_G2S_4D = (
    "cp.async.bulk.tensor.4d.shared::cta.global.tile"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
)


def _mn_opt_process_y(
    tmem_base, v_tile, alpha_tile, v_stage, alpha_stage, block_coeff, thread, IO_DTYPE
):
    y_values = K.alloc_local((64,), "float32")
    v_words = K.alloc_local((32,), "uint32")
    y_words = K.alloc_local((32,), "uint32")
    _mn_opt_tmem_ld_128x64(tmem_base, MN_OPT_TMEM_XY_COL, thread, y_values)
    _mn_opt_load_128x64_fragment(v_tile, v_stage, thread, v_words)
    factor_col_base = K.local_scalar("int32")
    K.assign(factor_col_base, K.bitwise_and(thread << 1, K.int32(6)))
    neg_factors = K.alloc_local((16,), "float32")
    with K.unroll(8) as factor_group:
        factor_col = K.local_scalar("int32")
        K.assign(factor_col, factor_col_base + factor_group * 8)
        K.ptx.ld.shared.v2.f32(
            neg_factors[factor_group * 2],
            neg_factors[factor_group * 2 + 1],
            alpha_tile.ptr_to([alpha_stage, 2, factor_col]),
        )
    with K.unroll(2) as row_half:
        with K.unroll(8) as factor_group:
            with K.unroll(2) as factor_repeat:
                pair = K.local_scalar("int32")
                K.assign(pair, row_half * 16 + factor_group * 2 + factor_repeat)
                v0 = K.local_scalar("float32")
                K.assign(v0, _mn_opt_unpack_io_lo(v_words[pair], IO_DTYPE))
                v1 = K.local_scalar("float32")
                K.assign(v1, _mn_opt_unpack_io_hi(v_words[pair], IO_DTYPE))
                updated = K.local_scalar("uint64")
                K.ptx.mul.rn.f32x2(
                    updated,
                    K.cuda.make_float2(v0, v1),
                    K.cuda.make_float2(
                        neg_factors[factor_group * 2], neg_factors[factor_group * 2 + 1]
                    ),
                )
                K.ptx.mov.b32(
                    y_values[pair * 2], block_coeff * y_values[pair * 2] + K.cuda.float2_x(updated)
                )
                K.ptx.mov.b32(
                    y_values[pair * 2 + 1],
                    block_coeff * y_values[pair * 2 + 1] + K.cuda.float2_y(updated),
                )
    with K.unroll(32) as pair:
        K.ptx.mov.b32(
            y_words[pair], _mn_opt_pack_iox2(y_values[pair * 2], y_values[pair * 2 + 1], IO_DTYPE)
        )
    _mn_opt_tmem_st_128x64_io(tmem_base, MN_OPT_TMEM_N_INPUT_COL, thread, y_words)
    K.ptx.tcgen05.wait__st.sync.aligned()


def _prefill_opt_cg0_tmem_ld(tmem_base, stage, thread, values):
    row_bits = K.local_scalar("int32")
    K.assign(row_bits, K.bitwise_and(thread << 16, K.int32(0x600000)))
    addr = K.local_scalar("int32")
    K.assign(addr, tmem_base + PREFILL_OPT_TMEM_CG0_ACC_COL + stage * 64 + row_bits)
    K.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
        *[values[i] for i in range(32)], K.cast(addr, "uint32")
    )


def _prefill_tmem_st_128x64_f32(tmem_base, column, thread, values):
    addr = K.local_scalar("int32")
    K.assign(addr, tmem_base + _mn_opt_tmem_row_bits(thread) + column)
    K.ptx["tcgen05.st.sync.aligned.16x256b.x8.b32"](
        K.cast(addr, "uint32"), *[values[i] for i in range(32)]
    )
    K.ptx["tcgen05.st.sync.aligned.16x256b.x8.b32"](
        K.cast(addr + K.int32(0x100000), "uint32"), *[values[32 + i] for i in range(32)]
    )


def _prefill_opt_load_t_fragment(tile, stage, thread, values, IO_DTYPE):
    lane_byte = K.local_scalar("int32")
    K.assign(
        lane_byte,
        (
            K.bitwise_or(
                K.bitwise_or(
                    K.bitwise_and(thread << 6, K.int32(960)), K.bitwise_and(thread >> 1, K.int32(8))
                ),
                K.bitwise_and(thread << 5, K.int32(3072)),
            )
            << 1
        ),
    )
    packed = K.alloc_local((16,), "uint32")
    for band in range(4):
        linear = K.local_scalar("int32")
        K.assign(linear, lane_byte // 2 + band * 16)
        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
            *[packed[band * 4 + i] for i in range(4)],
            tile[stage].ptr_to(linear // T_BLOCK, linear % T_BLOCK),
        )
    with K.unroll(16) as pair:
        K.ptx.mov.b32(values[pair * 2], _mn_opt_unpack_io_lo(packed[pair], IO_DTYPE))
        K.ptx.mov.b32(values[pair * 2 + 1], _mn_opt_unpack_io_hi(packed[pair], IO_DTYPE))


def _prefill_opt_store_ainv_fragment(tile, stage, thread, values, IO_DTYPE):
    # Ainv carries the transposed STMatrix fragment produced by the T transform.
    a = K.local_scalar("int32")
    K.assign(a, K.bitwise_and(thread << 6, K.int32(448)))
    c = K.local_scalar("int32")
    K.assign(
        c,
        K.bitwise_or(
            K.bitwise_or(a, K.bitwise_and(thread >> 1, K.int32(48))),
            K.bitwise_and(thread, K.int32(8)),
        ),
    )
    lane_byte = K.local_scalar("int32")
    K.assign(
        lane_byte,
        K.bitwise_or(K.bitwise_and(thread << 6, K.int32(1024)), K.bitwise_xor(a >> 3, c) << 1),
    )
    packed = K.alloc_local((16,), "uint32")
    with K.unroll(16) as pair:
        K.ptx.mov.b32(
            packed[pair], _mn_opt_pack_iox2(values[pair * 2], values[pair * 2 + 1], IO_DTYPE)
        )
    base = K.local_scalar("uint32")
    K.assign(base, K.cuda.cvta_generic_to_shared(tile[0].ptr_to(0, 0)))
    for band in range(4):
        K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            base + stage * 8192 + lane_byte + band * 2048, *[packed[band * 4 + i] for i in range(4)]
        )


def _prefill_opt_store_qk_fragment(tile, stage, thread, values, IO_DTYPE):
    # QK carries the non-transposed TMEM accumulator fragment; it is not Ainv's layout.
    a = K.local_scalar("int32")
    K.assign(a, K.bitwise_and(thread << 6, K.int32(448)))
    x = K.local_scalar("int32")
    K.assign(x, K.bitwise_or(a, K.bitwise_and(thread >> 1, K.int32(8))))
    g = K.local_scalar("int32")
    K.assign(g, K.bitwise_xor(K.bitwise_and(x >> 3, K.int32(56)), x))
    hi = K.local_scalar("int32")
    K.assign(
        hi,
        K.bitwise_or(
            K.bitwise_and(thread << 6, K.int32(512)), K.bitwise_and(thread << 5, K.int32(3072))
        ),
    )
    lane_byte = K.local_scalar("int32")
    K.assign(lane_byte, K.bitwise_or(hi, g) << 1)
    delta1 = K.local_scalar("int32")
    K.assign(delta1, K.if_then_else(K.bitwise_and(x, K.int32(128)) == 0, K.int32(16), K.int32(-16)))
    delta2 = K.local_scalar("int32")
    K.assign(delta2, K.if_then_else(K.bitwise_and(x, K.int32(256)) == 0, K.int32(32), K.int32(-32)))
    packed = K.alloc_local((16,), "uint32")
    with K.unroll(16) as pair:
        K.ptx.mov.b32(
            packed[pair], _mn_opt_pack_iox2(values[pair * 2], values[pair * 2 + 1], IO_DTYPE)
        )
    base = K.local_scalar("uint32")
    K.assign(base, K.cuda.cvta_generic_to_shared(tile[0].ptr_to(0, 0)))
    K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
        base + stage * 8192 + lane_byte, *[packed[i] for i in range(4)]
    )
    K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
        base + stage * 8192 + lane_byte + 2 * delta1, *[packed[4 + i] for i in range(4)]
    )
    K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
        base + stage * 8192 + lane_byte + 2 * delta2, *[packed[8 + i] for i in range(4)]
    )
    K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
        base + stage * 8192 + lane_byte + 2 * (delta1 + delta2), *[packed[12 + i] for i in range(4)]
    )


def _prefill_opt_transform_t(
    t_tile,
    ainv_tile,
    s_cumsumlog,
    t_stage,
    ainv_stage,
    gate_stage,
    thread,
    is_final_block,
    valid_tokens,
    IO_DTYPE,
):
    values = K.alloc_local((32,), "float32")
    _prefill_opt_load_t_fragment(t_tile, t_stage, thread, values, IO_DTYPE)
    row_base = K.local_scalar("int32")
    K.assign(
        row_base,
        K.bitwise_or(
            K.bitwise_and(thread >> 2, K.int32(7)), K.bitwise_and(thread >> 1, K.int32(48))
        ),
    )
    col_base = K.local_scalar("int32")
    K.assign(col_base, K.bitwise_and(thread << 1, K.int32(6)))
    with K.unroll(32) as i:
        t_coord = K.local_scalar("int32")
        K.assign(t_coord, row_base + K.bitwise_and(i >> 1, K.int32(1)) * 8)
        s_coord = K.local_scalar("int32")
        K.assign(
            s_coord,
            (
                col_base
                + K.bitwise_and(i, K.int32(1))
                + K.bitwise_and(i >> 2, K.int32(1)) * 8
                + (i >> 3) * 16
            ),
        )
        valid = K.local_scalar("bool")
        K.assign(valid, s_coord >= t_coord)
        with K.If(is_final_block):
            with K.Then():
                K.assign(valid, K.And(K.And(valid, s_coord < valid_tokens), t_coord < valid_tokens))
        gamma = K.local_scalar("float32")
        K.assign(
            gamma,
            _prefill_predicated_gamma(
                K.cuda.cvta_generic_to_shared(s_cumsumlog.ptr_to([gate_stage, s_coord])),
                K.cuda.cvta_generic_to_shared(s_cumsumlog.ptr_to([gate_stage, t_coord])),
                valid,
            ),
        )
        K.ptx.mov.b32(values[i], -gamma * values[i])
    _prefill_opt_store_ainv_fragment(ainv_tile, ainv_stage, thread, values, IO_DTYPE)


def _prefill_opt_load_v_fragment(tile, stage, thread, values):
    _mn_opt_load_128x64_fragment(tile, stage, thread, values)


def _prefill_opt_store_o_fragment(tile, stage, thread, values):
    _mn_opt_store_128x64_fragment(tile, stage, thread, values)


def _prefill_opt_sub_iox2(lhs, rhs, IO_DTYPE):
    result = K.alloc_local((1,), "uint32")
    if IO_DTYPE == "float16":
        K.evaluate(K.ptx.sub.f16x2(result[0], lhs, rhs))
    else:
        K.evaluate(K.ptx["sub.bf16x2"](result[0], lhs, rhs))
    return result[0]


_PREFILL_OPT_MMA_CHAIN = "tcgen05.mma.cta_group::1.kind::f16"
_PREFILL_OPT_ZERO_MASKS = [K.uint32(0)] * 4


def _prefill_opt_mma_descriptor(base, IO_DTYPE):
    return K.uint32(base + (0x480 if IO_DTYPE == "bfloat16" else 0))


def _prefill_opt_mma_commit(barrier):
    K.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
        barrier, pred=K.cuda.elect_sync()
    )


def _prefill_opt_mma_ss_64x64_k128(tmem_d, a_desc_base, b_desc_base, full_barrier, IO_DTYPE):
    descriptor = K.local_scalar("uint32")
    K.assign(descriptor, _prefill_opt_mma_descriptor(0x04100010, IO_DTYPE))
    for kphase in range(8):
        phase_off = K.local_scalar("uint64")
        K.assign(phase_off, K.uint64((kphase % 4) * 2 + (kphase // 4) * 512))
        K.evaluate(
            K.ptx[_PREFILL_OPT_MMA_CHAIN](
                K.cast(tmem_d, "uint32"),
                a_desc_base + phase_off,
                b_desc_base + phase_off,
                descriptor,
                *_PREFILL_OPT_ZERO_MASKS,
                K.ptx.pred(K.if_then_else(kphase == 0, 0, 1)),
                pred=K.cuda.elect_sync(),
            )
        )
    _prefill_opt_mma_commit(full_barrier)


def _prefill_opt_mma_ts_128x64_k128(tmem_d, tmem_a, b_desc_base, full_barrier, IO_DTYPE):
    descriptor = K.local_scalar("uint32")
    K.assign(descriptor, _prefill_opt_mma_descriptor(0x08100010, IO_DTYPE))
    for kphase in range(8):
        phase_off = K.local_scalar("uint64")
        K.assign(phase_off, K.uint64((kphase % 4) * 2 + (kphase // 4) * 512))
        K.evaluate(
            K.ptx[_PREFILL_OPT_MMA_CHAIN](
                K.cast(tmem_d, "uint32"),
                K.cast(tmem_a + kphase * 8, "uint32"),
                b_desc_base + phase_off,
                descriptor,
                *_PREFILL_OPT_ZERO_MASKS,
                K.ptx.pred(K.if_then_else(kphase == 0, 0, 1)),
                pred=K.cuda.elect_sync(),
            )
        )
    _prefill_opt_mma_commit(full_barrier)


def _prefill_opt_mma_ts_128x64_k64(tmem_d, tmem_a, b_desc_base, accumulate, IO_DTYPE):
    descriptor = K.local_scalar("uint32")
    K.assign(descriptor, _prefill_opt_mma_descriptor(0x08100010, IO_DTYPE))
    for kphase in range(4):
        K.evaluate(
            K.ptx[_PREFILL_OPT_MMA_CHAIN](
                K.cast(tmem_d, "uint32"),
                K.cast(tmem_a + kphase * 8, "uint32"),
                b_desc_base + K.uint64(kphase * 2),
                descriptor,
                *_PREFILL_OPT_ZERO_MASKS,
                K.ptx.pred(K.if_then_else(kphase == 0, accumulate, 1)),
                pred=K.cuda.elect_sync(),
            )
        )


def _prefill_opt_mma_ts_128x128_k64(tmem_d, tmem_a, b_desc_base, full_barrier, IO_DTYPE):
    descriptor = K.local_scalar("uint32")
    K.assign(descriptor, _prefill_opt_mma_descriptor(0x08210010, IO_DTYPE))
    for kphase in range(4):
        K.evaluate(
            K.ptx[_PREFILL_OPT_MMA_CHAIN](
                K.cast(tmem_d, "uint32"),
                K.cast(tmem_a + kphase * 8, "uint32"),
                b_desc_base + K.uint64(kphase * 128),
                descriptor,
                *_PREFILL_OPT_ZERO_MASKS,
                True,
                pred=K.cuda.elect_sync(),
            )
        )
    _prefill_opt_mma_commit(full_barrier)


_PREFILL_OPT_TMA_G2S = {
    dim: (
        f"cp.async.bulk.tensor.{dim}d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes.L2::cache_hint"
    )
    for dim in (3, 4, 5)
}
_PREFILL_OPT_TMA_S2G_4D = (
    "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
)


def _make_t_precompute(spec):
    io_dtype = spec["IO_DTYPE"]
    cu_dtype = spec["CU_DTYPE"]
    num_sequences = spec["NUM_SEQUENCES"]
    k_heads = spec["K_HEADS"]
    state_heads = spec["STATE_HEADS"]
    max_t_blocks = spec["MAX_T_BLOCKS"]
    grid_x = state_heads * max_t_blocks

    @K.kernel(warps=4, arch="sm_100a", min_blocks_per_sm=8, grid=(grid_x, num_sequences))
    def t_precompute(
        k: K.gptr[io_dtype],
        beta: K.gptr[K.f32],
        t: K.gptr[io_dtype],
        cu_seqlens: K.gptr[cu_dtype],
        k_map: K.TensorMap,
    ):
        bx, seq_idx = K.cta_id()
        roles = K.specialize()
        compute = roles.role("compute", warps=range(4))
        smem = K.smem_pool()
        k_ready = K.TMABar(smem, 1)
        beta_ready = K.MBarrier(smem, 1)
        k_ready.init(1)
        beta_ready.init(32)
        s_k = smem.alloc((T_BLOCK, D_HEAD), io_dtype, swizzle=K.SW128B)
        s_inv = smem.alloc((T_BLOCK, T_BLOCK), K.f16, swizzle=K.SW128B)
        s_beta = smem.alloc((T_BLOCK,), K.f32, align=16)
        with K.If(K.thread_id() == 0), K.Then():
            K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        with compute:
            tid = K.thread_id()
            lane = K.lane_id()
            warp = K.warp_id_in_role()
            state_head = bx % state_heads
            block_in_seq = bx // state_heads
            k_head = state_head * k_heads // state_heads
            sequence_bounds = K.alloc_local((2,), "int32")
            _load_sequence_bounds(cu_seqlens, seq_idx, sequence_bounds, cu_dtype)
            seq_start = sequence_bounds[0]
            seq_end = sequence_bounds[1]
            num_blocks = (seq_end - seq_start + T_BLOCK - 1) // T_BLOCK

            with K.If(block_in_seq < num_blocks), K.Then():
                token_start = seq_start + block_in_seq * T_BLOCK
                valid_len = K.min(T_BLOCK, seq_end - token_start)
                t_block = _device_chunk_bound(seq_idx, seq_start, T_BLOCK) + block_in_seq

                with K.If(warp == 1):
                    with K.Then(), K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx.prefetch.tensormap(K.address_of(k_map))
                        k_ready.arrive(0, tx_count=16384)
                        for d_coord in range(0, D_HEAD, 64):
                            K.ptx[_MN_OPT_TMA_G2S_3D](
                                s_k.ptr_to(0, d_coord),
                                K.address_of(k_map),
                                K.int32(d_coord),
                                K.Cast("int32", token_start),
                                k_head,
                                k_ready.ptr_to([0]),
                                K.uint64(0),
                            )
                    with K.Else(), K.If(warp == 2), K.Then():
                        with K.unroll(2) as beta_half:
                            beta_row = lane + beta_half * 32
                            beta_value = K.alloc_local((1,), "float32")
                            K.assign(beta_value[0], 0.0)
                            with K.If(beta_row < valid_len), K.Then():
                                K.ptx.ld.global_.nc.f32(
                                    beta_value[0],
                                    beta.ptr_to(
                                        [(token_start + beta_row) * state_heads + state_head]
                                    ),
                                )
                            K.ptx.st.shared.f32(s_beta.ptr_to([beta_row]), beta_value[0])
                        K.ptx.fence.proxy.async_.shared__cta()
                        beta_ready.arrive(0)

                k_ready.wait(0, 0)
                beta_ready.wait(0, 0)
                K.ptx.fence.proxy.async_.shared__cta()
                K.cuda.cta_sync()

                a_regs = K.alloc_local((4,), "uint32")
                b_regs = K.alloc_local((4,), "uint32")
                kk_acc = K.alloc_local((32,), "float32")
                packed_kk = K.alloc_local((4,), "uint32")
                with K.unroll(32) as acc_idx:
                    K.ptx.mov.b32(kk_acc[acc_idx], K.float32(0.0))
                with K.unroll(D_HEAD // 16) as k_tile:
                    _t_ldmatrix_x4_k_a(s_k, warp * 16, k_tile * 16, lane, a_regs)
                    with K.unroll(T_BLOCK // 16) as n_group:
                        _t_ldmatrix_x4_k_b(s_k, n_group * 16, k_tile * 16, lane, b_regs)
                        if io_dtype == "float16":
                            with K.If(k_tile == 0):
                                with K.Then():
                                    _t_mma_m16n8k16_f16_zero(kk_acc, a_regs, b_regs, n_group * 8, 0)
                                    _t_mma_m16n8k16_f16_zero(
                                        kk_acc, a_regs, b_regs, n_group * 8 + 4, 2
                                    )
                                with K.Else():
                                    _t_mma_m16n8k16_f16_acc(kk_acc, a_regs, b_regs, n_group * 8, 0)
                                    _t_mma_m16n8k16_f16_acc(
                                        kk_acc, a_regs, b_regs, n_group * 8 + 4, 2
                                    )
                        else:
                            with K.If(k_tile == 0):
                                with K.Then():
                                    _t_mma_m16n8k16_bf16_zero(
                                        kk_acc, a_regs, b_regs, n_group * 8, 0
                                    )
                                    _t_mma_m16n8k16_bf16_zero(
                                        kk_acc, a_regs, b_regs, n_group * 8 + 4, 2
                                    )
                                with K.Else():
                                    _t_mma_m16n8k16_bf16_acc(kk_acc, a_regs, b_regs, n_group * 8, 0)
                                    _t_mma_m16n8k16_bf16_acc(
                                        kk_acc, a_regs, b_regs, n_group * 8 + 4, 2
                                    )
                K.cuda.cta_sync()

                with K.unroll(T_BLOCK // 16) as n_group:
                    with K.unroll(8) as element:
                        within_mma = element & 3
                        row_kk = warp * 16 + (lane >> 2) + (within_mma >> 1) * 8
                        col_kk = (
                            n_group * 16 + (element >> 2) * 8 + (lane & 3) * 2 + (within_mma & 1)
                        )
                        value_kk = K.alloc_local((1,), "float32")
                        K.assign(value_kk[0], 0.0)
                        with K.If(row_kk > col_kk), K.Then():
                            beta_value = K.alloc_local((1,), "float32")
                            K.ptx.ld.shared.f32(beta_value[0], s_beta.ptr_to([row_kk]))
                            K.assign(value_kk[0], kk_acc[n_group * 8 + element] * beta_value[0])
                        K.ptx.mov.b32(kk_acc[n_group * 8 + element], value_kk[0])
                    with K.unroll(4) as pair:
                        K.ptx.mov.b32(
                            packed_kk[pair],
                            _t_pack_f16x2(
                                kk_acc[n_group * 8 + pair * 2], kk_acc[n_group * 8 + pair * 2 + 1]
                            ),
                        )
                    _t_stmatrix_x4(s_inv, warp * 16, n_group * 16, lane, packed_kk)
                K.cuda.cta_sync()

                with K.If(tid < 64), K.Then():
                    block8 = (tid >> 3) * 8
                    row8 = tid & 7
                    inverse_words = K.alloc_local((4,), "uint32")
                    inverse_row = K.alloc_local((8,), "float32")
                    K.ptx.ld.shared.v4.u32(
                        inverse_words[0],
                        inverse_words[1],
                        inverse_words[2],
                        inverse_words[3],
                        s_inv.ptr_to(block8 + row8, block8),
                    )
                    with K.unroll(4) as pair8:
                        K.ptx.mov.b32(
                            inverse_row[pair8 * 2],
                            K.Cast(
                                "float32",
                                K.reinterpret(
                                    "float16",
                                    K.Cast("uint16", inverse_words[pair8] & K.uint32(0xFFFF)),
                                ),
                            ),
                        )
                        K.ptx.mov.b32(
                            inverse_row[pair8 * 2 + 1],
                            K.Cast(
                                "float32",
                                K.reinterpret(
                                    "float16", K.Cast("uint16", inverse_words[pair8] >> 16)
                                ),
                            ),
                        )
                    with K.unroll(8) as col8:
                        raw_value = inverse_row[col8]
                        with K.If(row8 == col8):
                            with K.Then():
                                K.ptx.mov.b32(inverse_row[col8], K.float32(1.0))
                            with K.Else(), K.If(row8 < col8):
                                with K.Then():
                                    K.ptx.mov.b32(inverse_row[col8], K.float32(0.0))
                                with K.Else():
                                    K.ptx.mov.b32(inverse_row[col8], raw_value)
                    with K.unroll(7) as src_row:
                        row_scale = K.alloc_local((1,), "float32")
                        K.ptx.neg.f32(row_scale[0], inverse_row[src_row])
                        with K.unroll(7) as inverse_col:
                            with K.If(inverse_col < src_row), K.Then():
                                pivot = K.alloc_local((1,), "float32")
                                K.assign(
                                    pivot[0],
                                    K.cuda._shfl_sync(
                                        K.uint32(0xFFFFFFFF), inverse_row[inverse_col], src_row, 8
                                    ),
                                )
                                with K.If(row8 > src_row), K.Then():
                                    K.ptx.mov.b32(
                                        inverse_row[inverse_col],
                                        (inverse_row[inverse_col] + row_scale[0] * pivot[0]),
                                    )
                        with K.If(row8 > src_row), K.Then():
                            K.ptx.mov.b32(inverse_row[src_row], row_scale[0])
                    with K.unroll(4) as pair8:
                        K.ptx.mov.b32(
                            inverse_words[pair8],
                            _t_pack_f16x2(inverse_row[pair8 * 2], inverse_row[pair8 * 2 + 1]),
                        )
                    K.ptx.st.shared.v4.u32(
                        s_inv.ptr_to(block8 + row8, block8),
                        inverse_words[0],
                        inverse_words[1],
                        inverse_words[2],
                        inverse_words[3],
                    )
                K.cuda.cta_sync()
                _t_inverse_8_to_16(s_inv, warp * 16, lane)
                K.cuda.cta_sync()
                with K.If(tid < 64), K.Then():
                    _t_inverse_16_to_32(s_inv, warp * 32, lane)
                K.cuda.cta_sync()
                _t_inverse_32_to_64(s_inv, warp, lane)
                K.cuda.cta_sync()

                inverse_t0 = K.alloc_local((4,), "uint32")
                inverse_t1 = K.alloc_local((4,), "uint32")
                inverse_t2 = K.alloc_local((4,), "uint32")
                inverse_t3 = K.alloc_local((4,), "uint32")
                _t_ldmatrix_x4(s_inv, warp * 16, 0, lane, False, inverse_t0)
                _t_ldmatrix_x4(s_inv, warp * 16, 16, lane, False, inverse_t1)
                _t_ldmatrix_x4(s_inv, warp * 16, 32, lane, False, inverse_t2)
                _t_ldmatrix_x4(s_inv, warp * 16, 48, lane, False, inverse_t3)
                t_base = K.Cast("int64", (t_block * state_heads + state_head) * T_BLOCK * T_BLOCK)
                _t_store_t_fragment(
                    inverse_t0, s_beta, t, t_base, valid_len, warp, lane, 0, io_dtype
                )
                _t_store_t_fragment(
                    inverse_t1, s_beta, t, t_base, valid_len, warp, lane, 1, io_dtype
                )
                _t_store_t_fragment(
                    inverse_t2, s_beta, t, t_base, valid_len, warp, lane, 2, io_dtype
                )
                _t_store_t_fragment(
                    inverse_t3, s_beta, t, t_base, valid_len, warp, lane, 3, io_dtype
                )

    return t_precompute


def _make_fixup_simt(spec):
    cu_dtype = spec["CU_DTYPE"]
    state_dtype = spec["STATE_DTYPE"]
    num_sequences = spec["NUM_SEQUENCES"]
    state_heads = spec["STATE_HEADS"]
    total_cp_chunks = spec["TOTAL_CP_CHUNKS"]
    cp_chunk_len = spec["CP_CHUNK_LEN"]
    rows_per_cta = spec["FIXUP_SIMT_ROWS"]
    needs_initial_state = spec["NEEDS_INITIAL_STATE"]
    store_final_state = spec["STORE_FINAL_STATE"]
    use_state_indices = spec["USE_STATE_INDICES"]
    row_ctas = D_HEAD // rows_per_cta

    @K.kernel(
        warps=4, arch="sm_100a", min_blocks_per_sm=2, grid=num_sequences * state_heads * row_ctas
    )
    def fixup_simt(
        transfer: K.gptr[K.f32],
        local_state: K.gptr[K.f32],
        initial_state: K.gptr[state_dtype],
        initial_state_workspace: K.gptr[K.f32],
        fixed_state: K.gptr[K.f32],
        final_state: K.gptr[state_dtype],
        state_indices: K.gptr[K.i32],
        cu_seqlens: K.gptr[cu_dtype],
    ):
        roles = K.specialize()
        compute = roles.role("compute", warps=range(4), regs=256)
        smem = K.smem_pool()
        shared_state = smem.alloc((rows_per_cta, D_HEAD), K.f32, align=128)

        with compute:
            block = K.cta_id()
            col = K.thread_id()
            row_cta = block % row_ctas
            head_seq = block // row_ctas
            state_head = head_seq % state_heads
            seq_idx = head_seq // state_heads
            row_start = row_cta * rows_per_cta
            sequence_bounds = K.alloc_local((2,), "int32")
            _load_sequence_bounds(cu_seqlens, seq_idx, sequence_bounds, cu_dtype)
            seq_start = sequence_bounds[0]
            seq_end = sequence_bounds[1]
            seq_len = seq_end - seq_start
            num_chunks = (seq_len + cp_chunk_len - 1) // cp_chunk_len
            chunk_start = _device_chunk_bound(seq_idx, seq_start, cp_chunk_len)
            gap_start = chunk_start + num_chunks
            gap_end = K.alloc_local((1,), "int32")
            K.assign(gap_end[0], total_cp_chunks)
            with K.If(seq_idx + 1 < num_sequences), K.Then():
                K.assign(gap_end[0], _device_chunk_bound(seq_idx + 1, seq_end, cp_chunk_len))
            state_slot = K.alloc_local((1,), "int32")
            K.assign(state_slot[0], seq_idx)
            if use_state_indices:
                K.ptx.ld.global_.nc.b32(state_slot[0], state_indices.ptr_to([seq_idx]))

            with K.If(num_chunks > 0), K.Then():
                start = 0
                if needs_initial_state:
                    initial_values = K.alloc_local((rows_per_cta,), "float32")
                    with K.unroll(rows_per_cta) as local_row:
                        initial_index = (
                            (state_slot[0] * state_heads + state_head) * D_HEAD
                            + row_start
                            + local_row
                        ) * D_HEAD + col
                        _load_global_as_f32(
                            initial_values, local_row, initial_state, initial_index, state_dtype
                        )
                        K.ptx.st.shared.f32(
                            shared_state.ptr_to([local_row, col]), initial_values[local_row]
                        )
                        workspace_index = (
                            (seq_idx * state_heads + state_head) * D_HEAD + row_start + local_row
                        ) * D_HEAD + col
                        K.ptx.st.global_.f32(
                            initial_state_workspace.ptr_to([workspace_index]),
                            initial_values[local_row],
                        )
                else:
                    start = 1
                    first_slot = chunk_start
                    with K.unroll(rows_per_cta) as local_row:
                        first_index = (
                            (first_slot * state_heads + state_head) * D_HEAD + row_start + local_row
                        ) * D_HEAD + col
                        first_value = K.alloc_local((1,), "float32")
                        K.ptx.ld.global_.nc.f32(first_value[0], local_state.ptr_to([first_index]))
                        K.ptx.st.shared.f32(shared_state.ptr_to([local_row, col]), first_value[0])
                        K.ptx.st.global_.f32(fixed_state.ptr_to([first_index]), first_value[0])
                K.cuda.cta_sync()

                accum = K.alloc_local((rows_per_cta,), "float32")
                accum_next = K.alloc_local((rows_per_cta,), "float32")
                m_values = K.alloc_local((16,), "float32")
                m_next = K.alloc_local((16,), "float32")
                with K.If(start < num_chunks), K.Then():
                    first_work_slot = chunk_start + start
                    with K.unroll(rows_per_cta) as local_row:
                        first_work_index = (
                            (first_work_slot * state_heads + state_head) * D_HEAD
                            + row_start
                            + local_row
                        ) * D_HEAD + col
                        K.ptx.ld.global_.nc.f32(
                            accum[local_row], local_state.ptr_to([first_work_index])
                        )
                    with K.unroll(16) as inner:
                        transfer_index = (
                            (first_work_slot * state_heads + state_head) * D_HEAD + inner
                        ) * D_HEAD + col
                        K.ptx.ld.global_.nc.f32(m_values[inner], transfer.ptr_to([transfer_index]))

                with K.serial(start, num_chunks) as chunk:
                    cp_slot = chunk_start + chunk
                    next_chunk = chunk + 1
                    with K.unroll(7) as k_tile:
                        with K.unroll(16) as inner:
                            transfer_next_index = (
                                (cp_slot * state_heads + state_head) * D_HEAD
                                + (k_tile + 1) * 16
                                + inner
                            ) * D_HEAD + col
                            K.ptx.ld.global_.nc.f32(
                                m_next[inner], transfer.ptr_to([transfer_next_index])
                            )
                        with K.unroll(rows_per_cta) as local_row:
                            with K.unroll(16) as inner:
                                shared_value = K.alloc_local((1,), "float32")
                                K.ptx.ld.shared.f32(
                                    shared_value[0],
                                    shared_state.ptr_to([local_row, k_tile * 16 + inner]),
                                )
                                K.ptx.fma.rn.f32(
                                    accum[local_row],
                                    shared_value[0],
                                    m_values[inner],
                                    accum[local_row],
                                )
                        with K.unroll(16) as inner:
                            K.ptx.mov.b32(m_values[inner], m_next[inner])

                    with K.If(next_chunk < num_chunks), K.Then():
                        next_slot = chunk_start + next_chunk
                        with K.unroll(rows_per_cta) as local_row:
                            next_state_index = (
                                (next_slot * state_heads + state_head) * D_HEAD
                                + row_start
                                + local_row
                            ) * D_HEAD + col
                            K.ptx.ld.global_.nc.f32(
                                accum_next[local_row], local_state.ptr_to([next_state_index])
                            )
                        with K.unroll(16) as inner:
                            next_transfer_index = (
                                (next_slot * state_heads + state_head) * D_HEAD + inner
                            ) * D_HEAD + col
                            K.ptx.ld.global_.nc.f32(
                                m_next[inner], transfer.ptr_to([next_transfer_index])
                            )
                    with K.unroll(rows_per_cta) as local_row:
                        with K.unroll(16) as inner:
                            shared_value = K.alloc_local((1,), "float32")
                            K.ptx.ld.shared.f32(
                                shared_value[0], shared_state.ptr_to([local_row, 112 + inner])
                            )
                            K.ptx.fma.rn.f32(
                                accum[local_row], shared_value[0], m_values[inner], accum[local_row]
                            )
                    K.cuda.cta_sync()
                    with K.unroll(rows_per_cta) as local_row:
                        K.ptx.st.shared.f32(shared_state.ptr_to([local_row, col]), accum[local_row])
                        fixed_index = (
                            (cp_slot * state_heads + state_head) * D_HEAD + row_start + local_row
                        ) * D_HEAD + col
                        K.ptx.st.global_.f32(fixed_state.ptr_to([fixed_index]), accum[local_row])
                    K.cuda.cta_sync()
                    with K.If(next_chunk < num_chunks), K.Then():
                        with K.unroll(rows_per_cta) as local_row:
                            K.ptx.mov.b32(accum[local_row], accum_next[local_row])
                        with K.unroll(16) as inner:
                            K.ptx.mov.b32(m_values[inner], m_next[inner])

                if store_final_state:
                    with K.unroll(rows_per_cta) as local_row:
                        final_index = (
                            (state_slot[0] * state_heads + state_head) * D_HEAD
                            + row_start
                            + local_row
                        ) * D_HEAD + col
                        final_value = K.alloc_local((1,), "float32")
                        K.ptx.ld.shared.f32(final_value[0], shared_state.ptr_to([local_row, col]))
                        _store_global_from_f32(
                            final_state, final_index, final_value[0], state_dtype
                        )

            with K.serial(gap_start, gap_end[0]) as gap_slot:
                with K.unroll(rows_per_cta) as local_row:
                    gap_index = (
                        (gap_slot * state_heads + state_head) * D_HEAD + row_start + local_row
                    ) * D_HEAD + col
                    K.ptx.st.global_.f32(fixed_state.ptr_to([gap_index]), K.float32(0.0))

    return fixup_simt


def _make_fixup_utcmma(spec, rows, m_stages, compute_regs):
    cu_dtype = spec["CU_DTYPE"]
    state_dtype = spec["STATE_DTYPE"]
    num_sequences = spec["NUM_SEQUENCES"]
    state_heads = spec["STATE_HEADS"]
    cp_chunk_len = spec["CP_CHUNK_LEN"]
    needs_initial_state = spec["NEEDS_INITIAL_STATE"]
    store_final_state = spec["STORE_FINAL_STATE"]
    use_state_indices = spec["USE_STATE_INDICES"]
    row_ctas = D_HEAD // rows

    @K.kernel(
        warps=8, arch="sm_100a", min_blocks_per_sm=1, grid=num_sequences * state_heads * row_ctas
    )
    def fixup_utcmma(
        transfer: K.gptr[K.f32],
        local_state: K.gptr[K.f32],
        initial_state: K.gptr[state_dtype],
        initial_state_workspace: K.gptr[K.f32],
        fixed_state: K.gptr[K.f32],
        final_state: K.gptr[state_dtype],
        state_indices: K.gptr[K.i32],
        cu_seqlens: K.gptr[cu_dtype],
        transfer_map: K.TensorMap,
        local_state_map: K.TensorMap,
    ):
        roles = K.specialize()
        compute = roles.role("compute", warps=range(4))
        mma = roles.role("mma", warps=[4], regs=32)
        tma = roles.role("tma", warps=[5], regs=32)
        idle = roles.role("idle", warps=[6, 7], regs=32)
        empty_sequence_regs = roles.register_scope("empty_sequence_regs", warps=range(8), regs=32)
        smem = K.smem_pool()
        p_m = K.Pipeline(smem, m_stages, full="tma", empty="tcgen05")
        p_n = K.Pipeline(smem, 1, full="tma", empty="mbar", init_empty=4)
        p_ready = K.Pipeline(smem, 1, full="mbar", empty="tcgen05", init_full=128)
        p_done = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=128)
        tmem_holding = smem.alloc((1,), K.i32, align=4)
        s_m = smem.alloc((m_stages * 4, D_HEAD, 32), K.f32, swizzle=K.SW128B)
        s_n = smem.alloc((4, rows, 32), K.f32, swizzle=K.SW128B)
        with K.If(K.thread_id() == 0), K.Then():
            K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        block = K.cta_id()
        row_cta = block % row_ctas
        head_seq = block // row_ctas
        state_head = head_seq % state_heads
        seq_idx = head_seq // state_heads
        row_start = row_cta * rows
        sequence_bounds = K.alloc_local((2,), "int32")
        _load_sequence_bounds(cu_seqlens, seq_idx, sequence_bounds, cu_dtype)
        seq_start = sequence_bounds[0]
        seq_end = sequence_bounds[1]
        num_chunks = (seq_end - seq_start + cp_chunk_len - 1) // cp_chunk_len
        chunk_start = _device_chunk_bound(seq_idx, seq_start, cp_chunk_len)
        start = 0 if needs_initial_state else 1
        state_slot = K.alloc_local((1,), "int32")
        K.assign(state_slot[0], seq_idx)
        if use_state_indices:
            K.ptx.ld.global_.nc.b32(state_slot[0], state_indices.ptr_to([seq_idx]))

        with K.If(num_chunks == 0):
            with K.Then():
                warp = K.Cast(
                    "int32",
                    K.cuda._shfl_sync(
                        K.uint32(0xFFFFFFFF), K.Cast("uint32", K.thread_id() >> 5), 0, 32
                    ),
                )
                empty_sequence_regs.emit()
                with K.If(warp == 0), K.Then():
                    K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                        K.address_of(tmem_holding[0]), K.uint32(_FIXUP_TMEM_COLUMNS)
                    )
                tmem_base_invalid = K.alloc_local((1,), "int32")
                K.assign(tmem_base_invalid[0], 0)
                with K.If(warp <= 4), K.Then():
                    K.ptx.bar.sync(K.uint32(_FIXUP_TMEM_ALLOC_BARRIER), K.uint32(160))
                    K.ptx.ld.volatile.shared.s32(
                        tmem_base_invalid[0], K.address_of(tmem_holding[0])
                    )
                with K.If(warp == 0), K.Then():
                    K.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                    K.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                        K.Cast("uint32", tmem_base_invalid[0]), K.uint32(_FIXUP_TMEM_COLUMNS)
                    )
            with K.Else():
                with compute:
                    K.ptx.setmaxnreg.inc.sync.aligned.u32(compute_regs)
                    compute_thread = K.tid_in_role()
                    lane = K.lane_id()
                    with K.If(K.warp_id_in_role() == 0), K.Then():
                        K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                            K.address_of(tmem_holding[0]), K.uint32(_FIXUP_TMEM_COLUMNS)
                        )
                    K.ptx.bar.sync(K.uint32(_FIXUP_TMEM_ALLOC_BARRIER), K.uint32(160))
                    tmem_base = K.alloc_local((1,), "int32")
                    K.ptx.ld.volatile.shared.s32(tmem_base[0], K.address_of(tmem_holding[0]))
                    n_state = K.PipelineState(1, phase=0)
                    ready_state = K.PipelineState(1, phase=1)
                    done_state = K.PipelineState(1, phase=0)
                    state_base = (
                        K.Cast("int64", state_slot[0] * state_heads + state_head) * D_HEAD * D_HEAD
                        + row_start * D_HEAD
                    )
                    workspace_base = (
                        K.Cast("int64", seq_idx * state_heads + state_head) * D_HEAD * D_HEAD
                        + row_start * D_HEAD
                    )
                    fixed_base = (
                        K.Cast("int64", chunk_start * state_heads + state_head) * D_HEAD * D_HEAD
                        + row_start * D_HEAD
                    )
                    values = K.alloc_local((128,), "float32")
                    words = values.view("uint32")

                    if needs_initial_state:
                        _fixup_load_initial_to_tmem(
                            tmem_base[0],
                            initial_state,
                            state_base,
                            compute_thread,
                            rows,
                            state_dtype,
                        )
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        _fixup_tmem_ld(
                            tmem_base[0], _FIXUP_TMEM_ACC_COL, compute_thread, words, rows
                        )
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                        _fixup_store_f32(
                            words, initial_state_workspace, workspace_base, compute_thread, rows
                        )
                    else:
                        p_n.full.wait(n_state.stage, n_state.phase)
                        _fixup_load_n_to_tmem(tmem_base[0], s_n, compute_thread, rows)
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        _fixup_tmem_ld(
                            tmem_base[0], _FIXUP_TMEM_ACC_COL, compute_thread, words, rows
                        )
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                        _fixup_store_f32(words, fixed_state, fixed_base, compute_thread, rows)
                        with K.If(lane == 0), K.Then():
                            p_n.empty.arrive(n_state.stage)
                        n_state.advance()

                    with K.serial(start, num_chunks) as chunk:
                        _fixup_acc_to_tf32(tmem_base[0], compute_thread, rows)
                        p_n.full.wait(n_state.stage, n_state.phase)
                        p_ready.empty.wait(ready_state.stage, ready_state.phase)
                        _fixup_load_n_to_tmem(tmem_base[0], s_n, compute_thread, rows)
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        p_ready.full.arrive(ready_state.stage)
                        with K.If(lane == 0), K.Then():
                            p_n.empty.arrive(n_state.stage)
                        n_state.advance()
                        ready_state.advance()

                        p_done.full.wait(done_state.stage, done_state.phase)
                        _fixup_tmem_ld(
                            tmem_base[0], _FIXUP_TMEM_ACC_COL, compute_thread, words, rows
                        )
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                        chunk_base = (
                            K.Cast("int64", (chunk_start + chunk) * state_heads + state_head)
                            * D_HEAD
                            * D_HEAD
                            + row_start * D_HEAD
                        )
                        _fixup_store_f32(words, fixed_state, chunk_base, compute_thread, rows)
                        p_done.empty.arrive(done_state.stage)
                        done_state.advance()

                    if store_final_state:
                        _fixup_tmem_ld(
                            tmem_base[0], _FIXUP_TMEM_ACC_COL, compute_thread, words, rows
                        )
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                        last_fixed_base = (
                            K.Cast(
                                "int64", (chunk_start + num_chunks - 1) * state_heads + state_head
                            )
                            * D_HEAD
                            * D_HEAD
                            + row_start * D_HEAD
                        )
                        _fixup_store_f32(words, fixed_state, last_fixed_base, compute_thread, rows)
                        _fixup_store_state(
                            values, final_state, state_base, compute_thread, rows, state_dtype
                        )

                    with K.If(K.warp_id_in_role() == 0), K.Then():
                        K.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                        K.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                            K.Cast("uint32", tmem_base[0]), K.uint32(_FIXUP_TMEM_COLUMNS)
                        )

                with mma:
                    K.ptx.bar.sync(K.uint32(_FIXUP_TMEM_ALLOC_BARRIER), K.uint32(160))
                    tmem_base = K.alloc_local((1,), "int32")
                    K.ptx.ld.volatile.shared.s32(tmem_base[0], K.address_of(tmem_holding[0]))
                    m_state = K.PipelineState(m_stages, phase=0)
                    ready_state = K.PipelineState(1, phase=0)
                    done_state = K.PipelineState(1, phase=1)
                    with K.serial(num_chunks - start):
                        p_m.full.wait(m_state.stage, m_state.phase)
                        p_ready.full.wait(ready_state.stage, ready_state.phase)
                        p_done.empty.wait(done_state.stage, done_state.phase)
                        m_desc = _fixup_smem_desc_m(
                            K.cuda.cvta_generic_to_shared(s_m[m_state.stage * 4].ptr_to(0, 0)), 0
                        )
                        _fixup_mma(
                            tmem_base[0],
                            m_desc,
                            0,
                            rows,
                            p_done.full.ptr_to([done_state.stage]),
                            p_ready.empty.ptr_to([ready_state.stage]),
                            p_m.empty.ptr_to([m_state.stage]),
                        )
                        m_state.advance()
                        ready_state.advance()
                        done_state.advance()

                with tma:
                    K.ptx.prefetch.tensormap(K.address_of(transfer_map))
                    K.ptx.prefetch.tensormap(K.address_of(local_state_map))
                    n_state = K.PipelineState(1, phase=1)
                    m_state = K.PipelineState(m_stages, phase=1)
                    with K.serial(num_chunks) as chunk:
                        cp_slot = chunk_start + chunk
                        p_n.empty.wait(n_state.stage, n_state.phase)
                        with K.If(K.cuda.elect_sync()), K.Then():
                            p_n.full.arrive(n_state.stage, tx_count=rows * D_HEAD * 4)
                            for part in range(4):
                                K.ptx[_FIXUP_TMA_G2S_4D](
                                    s_n[part].ptr_to(0, 0),
                                    K.address_of(local_state_map),
                                    K.int32(part * 32),
                                    K.Cast("int32", row_start),
                                    state_head,
                                    cp_slot,
                                    p_n.full.ptr_to([n_state.stage]),
                                    K.uint64(0),
                                )
                        n_state.advance()
                        with K.If(chunk >= start), K.Then():
                            p_m.empty.wait(m_state.stage, m_state.phase)
                            with K.If(K.cuda.elect_sync()), K.Then():
                                p_m.full.arrive(m_state.stage, tx_count=D_HEAD * D_HEAD * 4)
                                for part in range(4):
                                    K.ptx[_FIXUP_TMA_G2S_4D](
                                        s_m[m_state.stage * 4 + part].ptr_to(0, 0),
                                        K.address_of(transfer_map),
                                        K.int32(part * 32),
                                        K.int32(0),
                                        state_head,
                                        cp_slot,
                                        p_m.full.ptr_to([m_state.stage]),
                                        K.uint64(0),
                                    )
                            m_state.advance()

                with idle:
                    K.evaluate(0)

    return fixup_utcmma


def _make_mn_precompute(spec):
    io_dtype = spec["IO_DTYPE"]
    cu_dtype = spec["CU_DTYPE"]
    num_sequences = spec["NUM_SEQUENCES"]
    k_heads = spec["K_HEADS"]
    v_heads = spec["V_HEADS"]
    state_heads = spec["STATE_HEADS"]
    max_cp_chunks = spec["MAX_CP_CHUNKS"]
    cp_chunk_len = spec["CP_CHUNK_LEN"]
    grid_x = state_heads * max_cp_chunks

    @K.kernel(warps=12, arch="sm_100a", min_blocks_per_sm=1, grid=(grid_x, num_sequences))
    def mn_precompute(
        k: K.gptr[io_dtype],
        v: K.gptr[io_dtype],
        t: K.gptr[io_dtype],
        alpha: K.gptr[K.f32],
        transfer: K.gptr[K.f32],
        local_state: K.gptr[K.f32],
        cu_seqlens: K.gptr[cu_dtype],
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        t_map: K.TensorMap,
    ):
        bx, seq_idx = K.cta_id()
        roles = K.specialize()
        cg0 = roles.role("cg0", warps=range(4), regs=216)
        cg1 = roles.role("cg1", warps=range(4, 8), regs=216)
        mma_m = roles.role("mma_m", warps=[8], regs=72)
        tma = roles.role("tma", warps=[9], regs=72)
        alpha_role = roles.role("alpha", warps=[10], regs=72)
        mma_n = roles.role("mma_n", warps=[11], regs=72)
        invalid_chunk_regs = roles.register_scope("invalid_chunk_regs", warps=range(12), regs=24)
        smem = K.smem_pool()
        p_k = K.Pipeline(smem, 3, full="tma", empty="tcgen05", init_empty=2)
        p_v = K.Pipeline(smem, 3, full="tma", empty="mbar", init_empty=4)
        p_t = K.Pipeline(smem, 3, full="tma", empty="tcgen05")
        p_alpha = K.Pipeline(smem, 4, full="mbar", empty="mbar", init_full=32, init_empty=256)
        p_m_init = K.Pipeline(smem, 1, full="mbar", empty="mbar", init_full=128)
        p_n_init = K.Pipeline(smem, 1, full="mbar", empty="mbar", init_full=128)
        p_x_acc = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=128)
        p_x_ready = K.Pipeline(smem, 2, full="mbar", empty="tcgen05", init_full=128, init_empty=2)
        p_m_input = K.Pipeline(smem, 1, full="mbar", empty="tcgen05", init_full=128)
        p_n_input = K.Pipeline(smem, 1, full="mbar", empty="tcgen05", init_full=128)
        p_z_acc = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=128)
        p_z_ready = K.Pipeline(smem, 1, full="mbar", empty="mbar", init_full=128)
        p_m_acc = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=128)
        p_y_acc = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=128)
        p_y_ready = K.Pipeline(smem, 1, full="mbar", empty="mbar", init_full=128)
        p_n_acc = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=128)
        p_done = K.Pipeline(smem, 1, full="mbar", empty="mbar", init_full=128, init_empty=128)
        tmem_holding = smem.alloc((1,), K.i32, align=4)
        s_k = smem.alloc((3, D_HEAD, T_BLOCK), io_dtype, swizzle=K.SW128B)
        s_v = smem.alloc((3, D_HEAD, T_BLOCK), io_dtype, swizzle=K.SW128B)
        s_t = smem.alloc((3, T_BLOCK, T_BLOCK), io_dtype, swizzle=K.SW128B)
        s_x = smem.alloc((2, D_HEAD, T_BLOCK), io_dtype, swizzle=K.SW128B)
        s_alpha = smem.alloc((4, 3, T_BLOCK), K.f32, align=4)
        with K.If(K.thread_id() == 0), K.Then():
            K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        tid = K.thread_id()
        lane = K.lane_id()
        warp = K.Cast(
            "int32", K.cuda._shfl_sync(K.uint32(0xFFFFFFFF), K.Cast("uint32", tid >> 5), 0, 32)
        )
        state_head = bx % state_heads
        chunk_in_seq = bx // state_heads
        k_head = state_head * k_heads // state_heads
        v_head = state_head * v_heads // state_heads
        sequence_bounds = K.alloc_local((2,), "int32")
        _load_sequence_bounds(cu_seqlens, seq_idx, sequence_bounds, cu_dtype)
        seq_start = sequence_bounds[0]
        seq_end = sequence_bounds[1]
        seq_len = seq_end - seq_start
        num_chunks = (seq_len + cp_chunk_len - 1) // cp_chunk_len
        valid_chunk = K.Cast("int32", chunk_in_seq < num_chunks)
        token_start = seq_start + chunk_in_seq * cp_chunk_len
        valid_len = K.min(cp_chunk_len, seq_end - token_start)
        num_blocks = (valid_len + T_BLOCK - 1) // T_BLOCK
        t_block_start = _device_chunk_bound(seq_idx, seq_start, T_BLOCK) + chunk_in_seq * (
            cp_chunk_len // T_BLOCK
        )

        with K.If(valid_chunk == 0):
            with K.Then():
                invalid_chunk_regs.emit()
                with K.If(warp == 4), K.Then():
                    K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                        K.address_of(tmem_holding[0]), K.uint32(TMEM_COLUMNS)
                    )
                tmem_base_invalid = K.alloc_local((1,), "int32")
                K.assign(tmem_base_invalid[0], 0)
                with K.If((warp <= 8) | (warp == 11)), K.Then():
                    K.ptx.bar.sync(K.uint32(MN_OPT_TMEM_ALLOC_BARRIER), K.uint32(320))
                    K.ptx.ld.volatile.shared.s32(
                        tmem_base_invalid[0], K.address_of(tmem_holding[0])
                    )
                with K.If(warp == 4), K.Then():
                    K.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                    K.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                        K.Cast("uint32", tmem_base_invalid[0]), K.uint32(TMEM_COLUMNS)
                    )
            with K.Else():
                with cg0:
                    K.ptx.barrier.sync(K.uint32(MN_OPT_TMEM_ALLOC_BARRIER), K.uint32(320))
                    tmem_base = K.alloc_local((1,), "int32")
                    K.ptx.ld.volatile.shared.s32(tmem_base[0], K.address_of(tmem_holding[0]))
                    thread = K.tid_in_role()
                    st_alpha = K.PipelineState(4, phase=0)
                    st_x_acc = K.PipelineState(1, phase=0)
                    st_x_ready = K.PipelineState(2, phase=1)
                    st_m_input = K.PipelineState(1, phase=1)
                    st_z_acc = K.PipelineState(1, phase=0)
                    st_z_ready = K.PipelineState(1, phase=1)
                    st_m_acc = K.PipelineState(1, phase=0)
                    st_m_init = K.PipelineState(1, phase=1)
                    st_done = K.PipelineState(1, phase=1)
                    _mn_opt_initialize_matrix(tmem_base[0], MN_OPT_TMEM_M_COL, thread, True)
                    p_m_init.empty.wait(st_m_init.stage, st_m_init.phase)
                    p_m_init.full.arrive(st_m_init.stage)
                    st_m_init.advance()

                    with K.serial(num_blocks) as block:
                        p_alpha.full.wait(st_alpha.stage, st_alpha.phase)
                        p_x_acc.full.wait(st_x_acc.stage, st_x_acc.phase)
                        p_x_ready.empty.wait(st_x_ready.stage, st_x_ready.phase)
                        _mn_opt_materialize_x(tmem_base[0], s_x, st_x_ready.stage, thread, io_dtype)
                        p_x_acc.empty.arrive(st_x_acc.stage)
                        p_x_ready.full.arrive(st_x_ready.stage)
                        st_x_acc.advance()
                        st_x_ready.advance()

                        with K.If(block > 0), K.Then():
                            p_m_input.empty.wait(st_m_input.stage, st_m_input.phase)
                            _mn_opt_matrix_to_io_input(
                                tmem_base[0],
                                MN_OPT_TMEM_M_COL,
                                MN_OPT_TMEM_M_INPUT_COL,
                                thread,
                                io_dtype,
                            )
                            p_m_input.full.arrive(st_m_input.stage)
                            p_z_acc.full.wait(st_z_acc.stage, st_z_acc.phase)
                            _mn_opt_scratch_to_io_input(
                                tmem_base[0], MN_OPT_TMEM_M_INPUT_COL, thread, io_dtype
                            )
                            p_z_acc.empty.arrive(st_z_acc.stage)
                            p_z_ready.empty.wait(st_z_ready.stage, st_z_ready.phase)
                            p_z_ready.full.arrive(st_z_ready.stage)
                            st_m_input.advance()
                            st_z_acc.advance()
                            st_z_ready.advance()

                        block_coeff = K.alloc_local((1,), "float32")
                        K.ptx.ld.shared.f32(
                            block_coeff[0], s_alpha.ptr_to([st_alpha.stage, 1, T_BLOCK - 1])
                        )
                        p_m_acc.full.wait(st_m_acc.stage, st_m_acc.phase)
                        _mn_opt_scale_matrix(
                            tmem_base[0], MN_OPT_TMEM_M_COL, thread, block_coeff[0]
                        )
                        p_m_acc.empty.arrive(st_m_acc.stage)
                        p_alpha.empty.arrive(st_alpha.stage)
                        st_m_acc.advance()
                        st_alpha.advance()

                    cp_slot = _device_chunk_bound(seq_idx, seq_start, cp_chunk_len) + chunk_in_seq
                    output_base = (
                        K.Cast("int64", cp_slot * state_heads + state_head) * D_HEAD * D_HEAD
                    )
                    _mn_opt_store_matrix_global(
                        tmem_base[0], MN_OPT_TMEM_M_COL, transfer, output_base, thread
                    )
                    p_done.empty.wait(st_done.stage, st_done.phase)
                    p_done.full.arrive(st_done.stage)
                    st_done.advance()

                with cg1:
                    with K.If(K.warp_id_in_role() == 0), K.Then():
                        K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                            K.address_of(tmem_holding[0]), K.uint32(TMEM_COLUMNS)
                        )
                    K.ptx.barrier.sync(K.uint32(MN_OPT_TMEM_ALLOC_BARRIER), K.uint32(320))
                    tmem_base = K.alloc_local((1,), "int32")
                    K.ptx.ld.volatile.shared.s32(tmem_base[0], K.address_of(tmem_holding[0]))
                    thread = K.tid_in_role()
                    st_v = K.PipelineState(3, phase=0)
                    st_alpha = K.PipelineState(4, phase=0)
                    st_n_input = K.PipelineState(1, phase=1)
                    st_y_acc = K.PipelineState(1, phase=0)
                    st_y_ready = K.PipelineState(1, phase=1)
                    st_n_acc = K.PipelineState(1, phase=0)
                    st_n_init = K.PipelineState(1, phase=1)
                    st_done = K.PipelineState(1, phase=0)
                    _mn_opt_initialize_matrix(tmem_base[0], MN_OPT_TMEM_N_COL, thread, False)
                    p_n_init.empty.wait(st_n_init.stage, st_n_init.phase)
                    p_n_init.full.arrive(st_n_init.stage)
                    st_n_init.advance()

                    with K.serial(num_blocks):
                        p_v.full.wait(st_v.stage, st_v.phase)
                        p_alpha.full.wait(st_alpha.stage, st_alpha.phase)
                        _mn_opt_matrix_to_io_input(
                            tmem_base[0],
                            MN_OPT_TMEM_N_COL,
                            MN_OPT_TMEM_N_INPUT_COL,
                            thread,
                            io_dtype,
                        )
                        p_n_input.empty.wait(st_n_input.stage, st_n_input.phase)
                        p_n_input.full.arrive(st_n_input.stage)
                        p_y_acc.full.wait(st_y_acc.stage, st_y_acc.phase)
                        block_coeff = K.alloc_local((1,), "float32")
                        K.ptx.ld.shared.f32(
                            block_coeff[0], s_alpha.ptr_to([st_alpha.stage, 1, T_BLOCK - 1])
                        )
                        _mn_opt_process_y(
                            tmem_base[0],
                            s_v,
                            s_alpha,
                            st_v.stage,
                            st_alpha.stage,
                            block_coeff[0],
                            thread,
                            io_dtype,
                        )
                        p_y_acc.empty.arrive(st_y_acc.stage)
                        _mn_opt_scale_matrix(
                            tmem_base[0], MN_OPT_TMEM_N_COL, thread, block_coeff[0]
                        )
                        p_y_ready.empty.wait(st_y_ready.stage, st_y_ready.phase)
                        p_y_ready.full.arrive(st_y_ready.stage)
                        p_n_acc.full.wait(st_n_acc.stage, st_n_acc.phase)
                        p_n_acc.empty.arrive(st_n_acc.stage)
                        with K.If(lane == 0), K.Then():
                            p_v.empty.arrive(st_v.stage)
                        p_alpha.empty.arrive(st_alpha.stage)
                        st_v.advance()
                        st_alpha.advance()
                        st_n_input.advance()
                        st_y_acc.advance()
                        st_y_ready.advance()
                        st_n_acc.advance()

                    cp_slot = _device_chunk_bound(seq_idx, seq_start, cp_chunk_len) + chunk_in_seq
                    output_base = (
                        K.Cast("int64", cp_slot * state_heads + state_head) * D_HEAD * D_HEAD
                    )
                    _mn_opt_store_matrix_global(
                        tmem_base[0], MN_OPT_TMEM_N_COL, local_state, output_base, thread
                    )
                    p_done.full.wait(st_done.stage, st_done.phase)
                    p_done.empty.arrive(st_done.stage)
                    K.cuda.warpgroup_sync(MN_OPT_TMEM_DEALLOC_BARRIER)
                    with K.If(K.warp_id_in_role() == 0), K.Then():
                        K.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                        K.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                            K.Cast("uint32", tmem_base[0]), K.uint32(TMEM_COLUMNS)
                        )

                with mma_m:
                    K.ptx.prefetch.tensormap(K.address_of(k_map))
                    K.ptx.prefetch.tensormap(K.address_of(v_map))
                    K.ptx.prefetch.tensormap(K.address_of(t_map))
                    K.ptx.barrier.sync(K.uint32(MN_OPT_TMEM_ALLOC_BARRIER), K.uint32(320))
                    tmem_base = K.alloc_local((1,), "int32")
                    K.ptx.ld.volatile.shared.s32(tmem_base[0], K.address_of(tmem_holding[0]))
                    st_m_init = K.PipelineState(1, phase=0)
                    st_k = K.PipelineState(3, phase=0)
                    st_m_input = K.PipelineState(1, phase=0)
                    st_z_acc = K.PipelineState(1, phase=1)
                    st_z_ready = K.PipelineState(1, phase=0)
                    st_x_ready = K.PipelineState(2, phase=0)
                    st_m_acc = K.PipelineState(1, phase=1)
                    p_m_init.full.wait(st_m_init.stage, st_m_init.phase)
                    with K.If(K.cuda.elect_sync()), K.Then():
                        p_m_init.empty.arrive(st_m_init.stage)
                    st_m_init.advance()

                    with K.serial(num_blocks) as block:
                        p_k.full.wait(st_k.stage, st_k.phase)
                        k_desc = s_k[st_k.stage].mma_desc(major="k").value
                        k_desc_mn = _mn_opt_smem_desc_mn(
                            K.cuda.cvta_generic_to_shared(s_k[st_k.stage].ptr_to(0, 0))
                        )
                        with K.If(block > 0), K.Then():
                            p_m_input.full.wait(st_m_input.stage, st_m_input.phase)
                            p_z_acc.empty.wait(st_z_acc.stage, st_z_acc.phase)
                            _mn_opt_mma_ts_128x64_k128(
                                tmem_base[0] + MN_OPT_TMEM_SCRATCH_COL,
                                tmem_base[0] + MN_OPT_TMEM_M_INPUT_COL,
                                k_desc,
                                p_z_acc.full.ptr_to([st_z_acc.stage]),
                                io_dtype,
                            )
                            p_z_ready.full.wait(st_z_ready.stage, st_z_ready.phase)
                            with K.If(K.cuda.elect_sync()), K.Then():
                                p_z_ready.empty.arrive(st_z_ready.stage)
                            _mn_opt_mma_commit(p_m_input.empty.ptr_to([st_m_input.stage]))
                            st_m_input.advance()
                            st_z_acc.advance()
                            st_z_ready.advance()

                        p_x_ready.full.wait(st_x_ready.stage, st_x_ready.phase)
                        p_m_acc.empty.wait(st_m_acc.stage, st_m_acc.phase)
                        x_desc = _mn_opt_smem_desc_mn(
                            K.cuda.cvta_generic_to_shared(s_x[st_x_ready.stage].ptr_to(0, 0))
                        )
                        with K.If(block == 0):
                            with K.Then():
                                _mn_opt_mma_ss_128x128_k64(
                                    tmem_base[0] + MN_OPT_TMEM_M_COL,
                                    k_desc_mn,
                                    x_desc,
                                    p_m_acc.full.ptr_to([st_m_acc.stage]),
                                    io_dtype,
                                )
                            with K.Else():
                                _mn_opt_mma_ts_128x128_k64(
                                    tmem_base[0] + MN_OPT_TMEM_M_COL,
                                    tmem_base[0] + MN_OPT_TMEM_M_INPUT_COL,
                                    x_desc,
                                    p_m_acc.full.ptr_to([st_m_acc.stage]),
                                    io_dtype,
                                )
                        _mn_opt_mma_commit(p_x_ready.empty.ptr_to([st_x_ready.stage]))
                        _mn_opt_mma_commit(p_k.empty.ptr_to([st_k.stage]))
                        st_k.advance()
                        st_x_ready.advance()
                        st_m_acc.advance()

                with mma_n:
                    K.ptx.barrier.sync(K.uint32(MN_OPT_TMEM_ALLOC_BARRIER), K.uint32(320))
                    tmem_base = K.alloc_local((1,), "int32")
                    K.ptx.ld.volatile.shared.s32(tmem_base[0], K.address_of(tmem_holding[0]))
                    st_n_init = K.PipelineState(1, phase=0)
                    st_k = K.PipelineState(3, phase=0)
                    st_t = K.PipelineState(3, phase=0)
                    st_x_acc = K.PipelineState(1, phase=1)
                    st_x_ready = K.PipelineState(2, phase=0)
                    st_n_input = K.PipelineState(1, phase=0)
                    st_y_acc = K.PipelineState(1, phase=1)
                    st_y_ready = K.PipelineState(1, phase=0)
                    st_n_acc = K.PipelineState(1, phase=1)
                    p_n_init.full.wait(st_n_init.stage, st_n_init.phase)
                    with K.If(K.cuda.elect_sync()), K.Then():
                        p_n_init.empty.arrive(st_n_init.stage)
                    st_n_init.advance()

                    with K.serial(num_blocks):
                        p_k.full.wait(st_k.stage, st_k.phase)
                        p_t.full.wait(st_t.stage, st_t.phase)
                        p_x_acc.empty.wait(st_x_acc.stage, st_x_acc.phase)
                        k_desc = s_k[st_k.stage].mma_desc(major="k").value
                        k_desc_mn = _mn_opt_smem_desc_mn(
                            K.cuda.cvta_generic_to_shared(s_k[st_k.stage].ptr_to(0, 0))
                        )
                        t_desc = s_t[st_t.stage].mma_desc(major="k").value
                        _mn_opt_mma_ss_128x64_k64(
                            tmem_base[0] + MN_OPT_TMEM_XY_COL,
                            k_desc_mn,
                            t_desc,
                            p_x_acc.full.ptr_to([st_x_acc.stage]),
                            io_dtype,
                        )
                        _mn_opt_mma_commit(p_t.empty.ptr_to([st_t.stage]))
                        p_x_ready.full.wait(st_x_ready.stage, st_x_ready.phase)
                        p_n_input.full.wait(st_n_input.stage, st_n_input.phase)
                        p_y_acc.empty.wait(st_y_acc.stage, st_y_acc.phase)
                        _mn_opt_mma_ts_128x64_k128(
                            tmem_base[0] + MN_OPT_TMEM_XY_COL,
                            tmem_base[0] + MN_OPT_TMEM_N_INPUT_COL,
                            k_desc,
                            p_y_acc.full.ptr_to([st_y_acc.stage]),
                            io_dtype,
                        )
                        _mn_opt_mma_commit(p_n_input.empty.ptr_to([st_n_input.stage]))
                        _mn_opt_mma_commit(p_k.empty.ptr_to([st_k.stage]))
                        p_y_ready.full.wait(st_y_ready.stage, st_y_ready.phase)
                        with K.If(K.cuda.elect_sync()), K.Then():
                            p_y_ready.empty.arrive(st_y_ready.stage)
                        p_n_acc.empty.wait(st_n_acc.stage, st_n_acc.phase)
                        x_desc = _mn_opt_smem_desc_mn(
                            K.cuda.cvta_generic_to_shared(s_x[st_x_ready.stage].ptr_to(0, 0))
                        )
                        _mn_opt_mma_ts_128x128_k64(
                            tmem_base[0] + MN_OPT_TMEM_N_COL,
                            tmem_base[0] + MN_OPT_TMEM_N_INPUT_COL,
                            x_desc,
                            p_n_acc.full.ptr_to([st_n_acc.stage]),
                            io_dtype,
                        )
                        _mn_opt_mma_commit(p_x_ready.empty.ptr_to([st_x_ready.stage]))
                        st_k.advance()
                        st_t.advance()
                        st_x_acc.advance()
                        st_x_ready.advance()
                        st_n_input.advance()
                        st_y_acc.advance()
                        st_y_ready.advance()
                        st_n_acc.advance()

                with tma:
                    st_k = K.PipelineState(3, phase=1)
                    st_v = K.PipelineState(3, phase=1)
                    st_t = K.PipelineState(3, phase=1)
                    with K.serial(num_blocks) as block:
                        p_k.empty.wait(st_k.stage, st_k.phase)
                        p_v.empty.wait(st_v.stage, st_v.phase)
                        p_t.empty.wait(st_t.stage, st_t.phase)
                        with K.If(K.cuda.elect_sync()), K.Then():
                            p_k.full.arrive(st_k.stage, tx_count=16384)
                            p_v.full.arrive(st_v.stage, tx_count=16384)
                            p_t.full.arrive(st_t.stage, tx_count=8192)
                            for d_coord in range(0, D_HEAD, 64):
                                K.ptx[_MN_OPT_TMA_G2S_3D](
                                    s_k[st_k.stage].ptr_to(d_coord, 0),
                                    K.address_of(k_map),
                                    K.int32(d_coord),
                                    K.Cast("int32", token_start + block * T_BLOCK),
                                    k_head,
                                    p_k.full.ptr_to([st_k.stage]),
                                    K.uint64(0),
                                )
                                K.ptx[_MN_OPT_TMA_G2S_3D](
                                    s_v[st_v.stage].ptr_to(d_coord, 0),
                                    K.address_of(v_map),
                                    K.int32(d_coord),
                                    K.Cast("int32", token_start + block * T_BLOCK),
                                    v_head,
                                    p_v.full.ptr_to([st_v.stage]),
                                    K.uint64(0),
                                )
                            K.ptx[_MN_OPT_TMA_G2S_4D](
                                s_t[st_t.stage].ptr_to(0, 0),
                                K.address_of(t_map),
                                K.int32(0),
                                K.int32(0),
                                state_head,
                                t_block_start + block,
                                p_t.full.ptr_to([st_t.stage]),
                                K.uint64(0),
                            )
                        st_k.advance()
                        st_v.advance()
                        st_t.advance()

                with alpha_role:
                    st_alpha = K.PipelineState(4, phase=1)
                    with K.serial(num_blocks) as block:
                        p_alpha.empty.wait(st_alpha.stage, st_alpha.phase)
                        token0 = block * T_BLOCK + lane
                        token1 = token0 + 32
                        alpha_values = K.alloc_local((2,), "float32")
                        K.ptx.mov.b32(alpha_values[0], K.float32(1.0))
                        K.ptx.mov.b32(alpha_values[1], K.float32(1.0))
                        with K.If(token0 < valid_len), K.Then():
                            K.ptx.ld.global_.f32(
                                alpha_values[0],
                                alpha.ptr_to(
                                    [
                                        K.Cast("int64", token_start + token0) * state_heads
                                        + state_head
                                    ]
                                ),
                            )
                        with K.If(token1 < valid_len), K.Then():
                            K.ptx.ld.global_.f32(
                                alpha_values[1],
                                alpha.ptr_to(
                                    [
                                        K.Cast("int64", token_start + token1) * state_heads
                                        + state_head
                                    ]
                                ),
                            )
                        logs = K.alloc_local((2,), "float32")
                        K.ptx.mov.b32(
                            logs[0], _lg2_approx_ftz(alpha_values[0] + K.float32(1.0e-10))
                        )
                        K.ptx.mov.b32(
                            logs[1], _lg2_approx_ftz(alpha_values[1] + K.float32(1.0e-10))
                        )
                        with K.unroll(5) as scan_step:
                            scan_offset = 1 << scan_step
                            prior = K.alloc_local((2,), "float32")
                            K.ptx.mov.b32(
                                prior[0],
                                K.cuda._shfl_up_sync(
                                    K.uint32(0xFFFFFFFF), logs[0], scan_offset, 32
                                ),
                            )
                            K.ptx.mov.b32(
                                prior[1],
                                K.cuda._shfl_up_sync(
                                    K.uint32(0xFFFFFFFF), logs[1], scan_offset, 32
                                ),
                            )
                            with K.If(lane >= scan_offset), K.Then():
                                K.ptx.mov.b32(logs[0], logs[0] + prior[0])
                                K.ptx.mov.b32(logs[1], logs[1] + prior[1])
                        K.ptx.mov.b32(
                            logs[1],
                            logs[1] + K.cuda._shfl_sync(K.uint32(0xFFFFFFFF), logs[0], 31, 32),
                        )
                        end_log = K.alloc_local((1,), "float32")
                        K.assign(
                            end_log[0], K.cuda._shfl_sync(K.uint32(0xFFFFFFFF), logs[1], 31, 32)
                        )
                        cumprod0 = _ex2_approx_ftz(logs[0])
                        cumprod1 = _ex2_approx_ftz(logs[1])
                        neg = K.alloc_local((2,), "float32")
                        K.ptx.neg.f32(neg[0], _ex2_approx_ftz(end_log[0] - logs[0]))
                        K.ptx.neg.f32(neg[1], _ex2_approx_ftz(end_log[0] - logs[1]))
                        with K.If(token0 >= valid_len), K.Then():
                            K.ptx.mov.b32(neg[0], K.float32(0.0))
                        with K.If(token1 >= valid_len), K.Then():
                            K.ptx.mov.b32(neg[1], K.float32(0.0))
                        K.ptx.st.shared.f32(s_alpha.ptr_to([st_alpha.stage, 0, lane]), logs[0])
                        K.ptx.st.shared.f32(s_alpha.ptr_to([st_alpha.stage, 0, lane + 32]), logs[1])
                        K.ptx.st.shared.f32(s_alpha.ptr_to([st_alpha.stage, 1, lane]), cumprod0)
                        K.ptx.st.shared.f32(
                            s_alpha.ptr_to([st_alpha.stage, 1, lane + 32]), cumprod1
                        )
                        K.ptx.st.shared.f32(s_alpha.ptr_to([st_alpha.stage, 2, lane]), neg[0])
                        K.ptx.st.shared.f32(s_alpha.ptr_to([st_alpha.stage, 2, lane + 32]), neg[1])
                        K.ptx.fence.proxy.async_.shared__cta()
                        p_alpha.full.arrive(st_alpha.stage)
                        st_alpha.advance()

    return mn_precompute


def _make_prefill(spec):
    io_dtype = spec["IO_DTYPE"]
    cu_dtype = spec["CU_DTYPE"]
    num_sequences = spec["NUM_SEQUENCES"]
    q_heads = spec["Q_HEADS"]
    k_heads = spec["K_HEADS"]
    v_heads = spec["V_HEADS"]
    state_heads = spec["STATE_HEADS"]
    max_cp_chunks = spec["MAX_CP_CHUNKS"]
    cp_chunk_len = spec["CP_CHUNK_LEN"]
    needs_initial_state = spec["NEEDS_INITIAL_STATE"]
    is_gqa = spec["IS_GQA"]
    head_base = spec["HEAD_BASE"]
    head_ratio = spec["HEAD_RATIO"]
    grid_x = state_heads * max_cp_chunks

    def copy_descriptor(src_map, dst):
        payload = K.alloc_local((4,), "uint64")
        src_u64: K.uint64 = K.reinterpret("uint64", K.address_of(src_map))
        dst_u64: K.uint64 = K.reinterpret("uint64", dst)
        for word_half in range(2):
            off: K.uint64 = K.uint64(word_half * 32)
            K.ptx.ld.global_.v4.b64(
                payload[0],
                payload[1],
                payload[2],
                payload[3],
                K.reinterpret("handle", src_u64 + off),
            )
            K.ptx.st.global_.v4.b64(
                K.reinterpret("handle", dst_u64 + off),
                payload[0],
                payload[1],
                payload[2],
                payload[3],
            )

    def replace_descriptor(desc, address, dim1, dim2, dim3, stride0, stride1, stride2):
        K.ptx.tensormap_replace.tile.global_address.global_.b1024.b64(
            desc, K.reinterpret("uint64", address)
        )
        K.ptx.tensormap_replace.tile.global_dim.global_.b1024.b32(desc, 0, K.uint32(128))
        K.ptx.tensormap_replace.tile.global_dim.global_.b1024.b32(desc, 1, K.cast(dim1, "uint32"))
        K.ptx.tensormap_replace.tile.global_stride.global_.b1024.b64(
            desc, 0, K.cast(stride0, "uint64")
        )
        K.ptx.tensormap_replace.tile.global_dim.global_.b1024.b32(desc, 2, K.cast(dim2, "uint32"))
        K.ptx.tensormap_replace.tile.global_stride.global_.b1024.b64(
            desc, 1, K.cast(stride1, "uint64")
        )
        K.ptx.tensormap_replace.tile.global_dim.global_.b1024.b32(desc, 3, K.cast(dim3, "uint32"))
        K.ptx.tensormap_replace.tile.global_stride.global_.b1024.b64(
            desc, 2, K.cast(stride2, "uint64")
        )
        K.ptx.tensormap_replace.tile.global_dim.global_.b1024.b32(desc, 4, K.uint32(1))
        K.ptx.tensormap_replace.tile.global_stride.global_.b1024.b64(desc, 3, K.uint64(0))

    @K.kernel(warps=12, arch="sm_100a", min_blocks_per_sm=1, grid=(grid_x, num_sequences))
    def prefill(
        q: K.gptr[io_dtype],
        k: K.gptr[io_dtype],
        v: K.gptr[io_dtype],
        alpha: K.gptr[K.f32],
        t: K.gptr[io_dtype],
        fixed_state: K.gptr[K.f32],
        initial_state_workspace: K.gptr[K.f32],
        o: K.gptr[io_dtype],
        cu_seqlens: K.gptr[cu_dtype],
        scale: K.f32,
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        t_map: K.TensorMap,
        o_map: K.TensorMap,
        descriptor_workspace: K.gptr[K.i8],
    ):
        bx, seq_idx = K.cta_id()
        roles = K.specialize()
        cg0 = roles.role("cg0", warps=range(4), regs=224)
        cg1 = roles.role("cg1", warps=range(4, 8), regs=256)
        mma0 = roles.role("mma0", warps=[8], regs=24)
        tma = roles.role("tma", warps=[9], regs=24)
        mma1 = roles.role("mma1", warps=[10], regs=24)
        aux = roles.role("aux", warps=[11], regs=24)
        smem = K.smem_pool()
        p_k = K.Pipeline(smem, 3, full="tma", empty="tcgen05", init_empty=2)
        p_q = K.Pipeline(smem, 2, full="tma", empty="tcgen05", init_empty=2)
        p_v = K.Pipeline(smem, 3, full="tma", empty="mbar", init_empty=4)
        p_gate = K.Pipeline(smem, 5, full="mbar", empty="mbar", init_full=32, init_empty=256)
        p_t = K.Pipeline(smem, 2, full="tma", empty="mbar", init_empty=4)
        p_qstate = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=128)
        p_kv = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=128)
        p_cg0 = K.Pipeline(smem, 2, full="tcgen05", empty="mbar", init_empty=128)
        p_cg1 = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=128)
        p_ainv = K.Pipeline(smem, 3, full="mbar", empty="tcgen05", init_full=128)
        p_qk = K.Pipeline(smem, 2, full="mbar", empty="tcgen05", init_full=128)
        p_state_input = K.Pipeline(smem, 1, full="mbar", empty="tcgen05", init_full=128)
        m_vks = K.MBarrier(smem, 1)
        m_nv = K.MBarrier(smem, 1)
        m_decay = K.MBarrier(smem, 1)
        m_vks.init(128)
        m_nv.init(128)
        m_decay.init(128)
        p_o = K.Pipeline(smem, 2, full="mbar", empty="mbar", init_full=128, init_empty=32)
        tmem_holding = smem.alloc((1,), K.i32, align=4)
        s_q = smem.alloc((2, T_BLOCK, D_HEAD), io_dtype, swizzle=K.SW128B)
        s_k = smem.alloc((3, T_BLOCK, D_HEAD), io_dtype, swizzle=K.SW128B)
        s_v = smem.alloc((3, D_HEAD, T_BLOCK), io_dtype, swizzle=K.SW128B)
        s_t = smem.alloc((2, T_BLOCK, T_BLOCK), io_dtype, swizzle=K.SW128B)
        s_ainv = smem.alloc((3, T_BLOCK, T_BLOCK), io_dtype, swizzle=K.SW128B)
        s_qk = smem.alloc((2, T_BLOCK, T_BLOCK), io_dtype, swizzle=K.SW128B)
        s_o = smem.alloc((2, D_HEAD, T_BLOCK), io_dtype, swizzle=K.SW128B)
        s_cumsumlog = smem.alloc((5, T_BLOCK), K.f32, align=16)
        s_cumprod = smem.alloc((5, T_BLOCK), K.f32, align=16)
        with K.If(K.thread_id() == 0), K.Then():
            K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        thread = K.thread_id()
        lane = K.lane_id()
        state_head = bx % state_heads
        chunk_in_seq = bx // state_heads
        sequence_bounds = K.alloc_local((2,), "int32")
        _load_sequence_bounds(cu_seqlens, seq_idx, sequence_bounds, cu_dtype)
        seq_start = sequence_bounds[0]
        seq_end = sequence_bounds[1]
        seq_len = seq_end - seq_start
        num_cp_chunks = (seq_len + cp_chunk_len - 1) // cp_chunk_len
        chunk_len = K.alloc_local((1,), "int32")
        K.assign(chunk_len[0], 0)
        with K.If(chunk_in_seq < num_cp_chunks), K.Then():
            K.assign(chunk_len[0], K.min(cp_chunk_len, seq_len - chunk_in_seq * cp_chunk_len))
        chunk_start = seq_start + chunk_in_seq * cp_chunk_len
        chunk_end = chunk_start + chunk_len[0]
        cp_slot = _device_chunk_bound(seq_idx, seq_start, cp_chunk_len) + chunk_in_seq
        t_block_start = _device_chunk_bound(seq_idx, seq_start, T_BLOCK) + chunk_in_seq * (
            cp_chunk_len // T_BLOCK
        )
        num_valid_chunks = (chunk_len[0] + T_BLOCK - 1) // T_BLOCK
        padded_chunks = ((chunk_len[0] + 2 * T_BLOCK - 1) // (2 * T_BLOCK)) * 2
        subhead = state_head % head_ratio
        base_head = state_head // head_ratio
        k_head = state_head * k_heads // state_heads
        descriptor_base = K.Cast("int64", seq_idx * grid_x + bx) * K.int64(
            DESCRIPTOR_SLOTS * DESCRIPTOR_SLOT_BYTES
        )
        descriptor_q = descriptor_workspace.ptr_to([descriptor_base])
        descriptor_k = descriptor_workspace.ptr_to([descriptor_base + 128])
        descriptor_v = descriptor_workspace.ptr_to([descriptor_base + 256])
        descriptor_o = descriptor_workspace.ptr_to([descriptor_base + 512])

        with cg0:
            K.ptx.barrier.sync(K.uint32(PREFILL_OPT_TMEM_ALLOC_BARRIER), K.uint32(320))
            tmem_base = K.alloc_local((1,), "int32")
            K.ptx.ld.volatile.shared.s32(tmem_base[0], K.address_of(tmem_holding[0]))
            st_gate = K.PipelineState(5, phase=0)
            st_t = K.PipelineState(2, phase=0)
            st_acc = K.PipelineState(2, phase=0)
            st_ainv = K.PipelineState(3, phase=1)
            st_qk = K.PipelineState(2, phase=1)
            with K.serial(padded_chunks) as chunk:
                p_gate.full.wait(st_gate.stage, st_gate.phase)
                p_t.full.wait(st_t.stage, st_t.phase)
                p_ainv.empty.wait(st_ainv.stage, st_ainv.phase)
                _prefill_opt_transform_t(
                    s_t,
                    s_ainv,
                    s_cumsumlog,
                    st_t.stage,
                    st_ainv.stage,
                    st_gate.stage,
                    thread,
                    chunk >= num_valid_chunks - 1,
                    chunk_len[0] - chunk * T_BLOCK,
                    io_dtype,
                )
                K.ptx.fence.proxy.async_.shared__cta()
                K.ptx.bar.sync(K.uint32(PREFILL_OPT_T_STORE_BARRIER), K.uint32(128))
                with K.If(lane == 0), K.Then():
                    p_t.empty.arrive(st_t.stage)
                p_ainv.full.arrive(st_ainv.stage)

                p_qk.empty.wait(st_qk.stage, st_qk.phase)
                p_cg0.full.wait(st_acc.stage, st_acc.phase)
                qk_values = K.alloc_local((32,), "float32")
                _prefill_opt_cg0_tmem_ld(tmem_base[0], st_acc.stage, thread, qk_values)
                row_base = K.bitwise_or(
                    K.bitwise_and(thread >> 2, K.int32(7)), K.bitwise_and(thread >> 1, K.int32(48))
                )
                col_base = K.bitwise_and(thread << 1, K.int32(6))
                with K.unroll(32) as frag:
                    score_s = row_base + K.bitwise_and(frag >> 1, K.int32(1)) * 8
                    score_t = (
                        col_base
                        + K.bitwise_and(frag, K.int32(1))
                        + K.bitwise_and(frag >> 2, K.int32(1)) * 8
                        + (frag >> 3) * 16
                    )
                    valid = K.alloc_local((1,), "bool")
                    K.assign(valid[0], score_s >= score_t)
                    with K.If(chunk >= num_valid_chunks - 1), K.Then():
                        K.assign(
                            valid[0],
                            (
                                valid[0]
                                & (score_s < chunk_len[0] - chunk * T_BLOCK)
                                & (score_t < chunk_len[0] - chunk * T_BLOCK)
                            ),
                        )
                    gamma = _prefill_predicated_gamma(
                        K.cuda.cvta_generic_to_shared(s_cumsumlog.ptr_to([st_gate.stage, score_s])),
                        K.cuda.cvta_generic_to_shared(s_cumsumlog.ptr_to([st_gate.stage, score_t])),
                        valid[0],
                    )
                    K.ptx.mov.b32(qk_values[frag], qk_values[frag] * gamma * scale)
                _prefill_opt_store_qk_fragment(s_qk, st_qk.stage, thread, qk_values, io_dtype)
                K.ptx.fence.proxy.async_.shared__cta()
                K.ptx.tcgen05.wait__ld.sync.aligned()
                p_cg0.empty.arrive(st_acc.stage)
                p_qk.full.arrive(st_qk.stage)
                p_gate.empty.arrive(st_gate.stage)
                st_gate.advance()
                st_t.advance()
                st_acc.advance()
                st_ainv.advance()
                st_qk.advance()
            for _ in range(3):
                p_ainv.empty.wait(st_ainv.stage, st_ainv.phase)
                st_ainv.advance()
            for _ in range(2):
                p_qk.empty.wait(st_qk.stage, st_qk.phase)
                st_qk.advance()

        with cg1:
            with K.If(K.warp_id_in_role() == 0), K.Then():
                K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    K.address_of(tmem_holding[0]), K.uint32(TMEM_COLUMNS)
                )
            K.ptx.barrier.sync(K.uint32(PREFILL_OPT_TMEM_ALLOC_BARRIER), K.uint32(320))
            tmem_base = K.alloc_local((1,), "int32")
            K.ptx.ld.volatile.shared.s32(tmem_base[0], K.address_of(tmem_holding[0]))
            st_v = K.PipelineState(3, phase=0)
            st_gate = K.PipelineState(5, phase=0)
            st_shared = K.PipelineState(1, phase=0)
            st_kv_c = K.PipelineState(1, phase=0)
            st_qstate_c = K.PipelineState(1, phase=0)
            st_kv_p = K.PipelineState(1, phase=1)
            st_state_input = K.PipelineState(1, phase=1)
            st_vks = K.PipelineState(1, phase=1)
            st_nv = K.PipelineState(1, phase=1)
            st_decay = K.PipelineState(1, phase=1)
            st_o = K.PipelineState(2, phase=1)
            with K.If(chunk_len[0] > 0), K.Then():
                p_kv.empty.wait(st_kv_p.stage, st_kv_p.phase)
                cg1_thread = K.tid_in_role()
                state_words = K.alloc_local((32,), "uint32")
                for state_sub in range(4):
                    for vector in range(8):
                        word_offset = state_sub * 32 + vector * 4
                        with K.If(chunk_in_seq > 0):
                            with K.Then():
                                base = (
                                    K.Cast("int64", (cp_slot - 1) * state_heads + state_head)
                                    * D_HEAD
                                    + K.Cast("int64", cg1_thread)
                                ) * D_HEAD + word_offset
                                K.ptx["ld.global.L1::no_allocate.v4.b32"](
                                    state_words[vector * 4],
                                    state_words[vector * 4 + 1],
                                    state_words[vector * 4 + 2],
                                    state_words[vector * 4 + 3],
                                    fixed_state.ptr_to([base]),
                                )
                            with K.Else():
                                if needs_initial_state:
                                    base = (
                                        K.Cast("int64", seq_idx * state_heads + state_head) * D_HEAD
                                        + K.Cast("int64", cg1_thread)
                                    ) * D_HEAD + word_offset
                                    K.ptx["ld.global.L1::no_allocate.v4.b32"](
                                        state_words[vector * 4],
                                        state_words[vector * 4 + 1],
                                        state_words[vector * 4 + 2],
                                        state_words[vector * 4 + 3],
                                        initial_state_workspace.ptr_to([base]),
                                    )
                                else:
                                    K.ptx.mov.b32(state_words[vector * 4], K.uint32(0))
                                    K.ptx.mov.b32(state_words[vector * 4 + 1], K.uint32(0))
                                    K.ptx.mov.b32(state_words[vector * 4 + 2], K.uint32(0))
                                    K.ptx.mov.b32(state_words[vector * 4 + 3], K.uint32(0))
                    _mn_opt_tmem_st_matrix_sub(
                        tmem_base[0],
                        PREFILL_OPT_TMEM_STATE_COL,
                        cg1_thread,
                        state_sub,
                        state_words,
                        0,
                    )
                K.ptx.tcgen05.wait__st.sync.aligned()
                K.ptx.bar.sync(K.uint32(PREFILL_OPT_INITIAL_STATE_BARRIER), K.uint32(128))
                with K.If(cg1_thread == 0), K.Then():
                    p_kv.full.arrive(st_kv_p.stage)
                st_kv_p.advance()

                with K.serial(padded_chunks) as chunk:
                    with K.If((chunk & 1) == 0), K.Then():
                        st_kv_p.advance()
                        st_kv_p.advance()
                    p_gate.full.wait(st_gate.stage, st_gate.phase)
                    cumprod_total = K.alloc_local((1,), "float32")
                    K.ptx.ld.shared.f32(
                        cumprod_total[0], s_cumprod.ptr_to([st_gate.stage, T_BLOCK - 1])
                    )

                    p_kv.full.wait(st_kv_c.stage, st_kv_c.phase)
                    p_state_input.empty.wait(st_state_input.stage, st_state_input.phase)
                    state_values = K.alloc_local((128,), "float32")
                    state_input_words = K.alloc_local((64,), "uint32")
                    for state_sub in range(4):
                        _mn_opt_tmem_ld_matrix_sub(
                            tmem_base[0],
                            PREFILL_OPT_TMEM_STATE_COL,
                            cg1_thread,
                            state_sub,
                            state_values,
                            state_sub * 32,
                        )
                    with K.unroll(64) as pair:
                        K.ptx.mov.b32(
                            state_input_words[pair],
                            _mn_opt_pack_iox2(
                                state_values[pair * 2], state_values[pair * 2 + 1], io_dtype
                            ),
                        )
                    for state_sub in range(4):
                        _mn_opt_tmem_st_matrix_io_sub(
                            tmem_base[0],
                            PREFILL_OPT_TMEM_STATE_INPUT_COL,
                            cg1_thread,
                            state_sub,
                            state_input_words,
                        )
                    K.ptx.tcgen05.wait__st.sync.aligned()
                    p_state_input.full.arrive(st_state_input.stage)
                    with K.unroll(64) as pair:
                        state_mul = K.alloc_local((1,), "uint64")
                        K.ptx.mul.rn.f32x2(
                            state_mul[0],
                            K.cuda.make_float2(state_values[pair * 2], state_values[pair * 2 + 1]),
                            K.cuda.make_float2(cumprod_total[0], cumprod_total[0]),
                        )
                        K.ptx.mov.b32(state_values[pair * 2], K.cuda.float2_x(state_mul[0]))
                        K.ptx.mov.b32(state_values[pair * 2 + 1], K.cuda.float2_y(state_mul[0]))
                    for state_sub in range(4):
                        _mn_opt_tmem_st_matrix_sub(
                            tmem_base[0],
                            PREFILL_OPT_TMEM_STATE_COL,
                            cg1_thread,
                            state_sub,
                            state_values,
                            state_sub * 32,
                        )
                    K.ptx.tcgen05.wait__st.sync.aligned()
                    p_kv.empty.arrive(st_kv_c.stage)

                    cumprod_factor = K.alloc_local((16,), "float32")
                    decay_factor = K.alloc_local((16,), "float32")
                    factor_col_base = K.bitwise_and(cg1_thread << 1, K.int32(6))
                    last_log = K.alloc_local((1,), "float32")
                    K.ptx.ld.shared.f32(
                        last_log[0], s_cumsumlog.ptr_to([st_gate.stage, T_BLOCK - 1])
                    )
                    with K.unroll(8) as factor_group:
                        factor_col = factor_col_base + factor_group * 8
                        K.ptx.ld.shared.v2.f32(
                            cumprod_factor[factor_group * 2],
                            cumprod_factor[factor_group * 2 + 1],
                            s_cumprod.ptr_to([st_gate.stage, factor_col]),
                        )
                        log_pair = K.alloc_local((2,), "float32")
                        K.ptx.ld.shared.v2.f32(
                            log_pair[0],
                            log_pair[1],
                            s_cumsumlog.ptr_to([st_gate.stage, factor_col]),
                        )
                        diff = K.alloc_local((1,), "uint64")
                        K.ptx.sub.rn.f32x2(
                            diff[0],
                            K.cuda.make_float2(last_log[0], last_log[0]),
                            K.cuda.make_float2(log_pair[0], log_pair[1]),
                        )
                        K.ptx.ex2.approx.ftz.f32(
                            decay_factor[factor_group * 2], K.cuda.float2_x(diff[0])
                        )
                        K.ptx.ex2.approx.ftz.f32(
                            decay_factor[factor_group * 2 + 1], K.cuda.float2_y(diff[0])
                        )
                    p_gate.empty.arrive(st_gate.stage)

                    p_v.full.wait(st_v.stage, st_v.phase)
                    v_words = K.alloc_local((32,), "uint32")
                    _prefill_opt_load_v_fragment(s_v, st_v.stage, cg1_thread, v_words)
                    p_cg1.full.wait(st_shared.stage, st_shared.phase)
                    fragment = K.alloc_local((64,), "float32")
                    _mn_opt_tmem_ld_128x64(
                        tmem_base[0], PREFILL_OPT_TMEM_CG1_ACC_COL, cg1_thread, fragment
                    )
                    with K.unroll(2) as row_half:
                        with K.unroll(8) as factor_group:
                            with K.unroll(2) as factor_repeat:
                                pair = row_half * 16 + factor_group * 2 + factor_repeat
                                mul = K.alloc_local((1,), "uint64")
                                K.ptx.mul.rn.f32x2(
                                    mul[0],
                                    K.cuda.make_float2(fragment[pair * 2], fragment[pair * 2 + 1]),
                                    K.cuda.make_float2(
                                        cumprod_factor[factor_group * 2],
                                        cumprod_factor[factor_group * 2 + 1],
                                    ),
                                )
                                K.ptx.mov.b32(fragment[pair * 2], K.cuda.float2_x(mul[0]))
                                K.ptx.mov.b32(fragment[pair * 2 + 1], K.cuda.float2_y(mul[0]))
                    p_cg1.empty.arrive(st_shared.stage)
                    st_shared.advance()
                    with K.unroll(32) as pair:
                        ks_word = _mn_opt_pack_iox2(
                            fragment[pair * 2], fragment[pair * 2 + 1], io_dtype
                        )
                        K.ptx.mov.b32(
                            v_words[pair], _prefill_opt_sub_iox2(v_words[pair], ks_word, io_dtype)
                        )
                    _mn_opt_tmem_st_128x64_io(
                        tmem_base[0], PREFILL_OPT_TMEM_SHARED_INPUT_COL, cg1_thread, v_words
                    )
                    K.ptx.tcgen05.wait__st.sync.aligned()
                    m_vks.arrive(st_vks.stage)

                    p_qstate.full.wait(st_qstate_c.stage, st_qstate_c.phase)
                    _mn_opt_tmem_ld_128x64(
                        tmem_base[0], PREFILL_OPT_TMEM_Q_STATE_COL, cg1_thread, fragment
                    )
                    with K.unroll(2) as row_half:
                        with K.unroll(8) as factor_group:
                            with K.unroll(2) as factor_repeat:
                                pair = row_half * 16 + factor_group * 2 + factor_repeat
                                mul = K.alloc_local((1,), "uint64")
                                K.ptx.mul.rn.f32x2(
                                    mul[0],
                                    K.cuda.make_float2(fragment[pair * 2], fragment[pair * 2 + 1]),
                                    K.cuda.make_float2(
                                        cumprod_factor[factor_group * 2],
                                        cumprod_factor[factor_group * 2 + 1],
                                    ),
                                )
                                K.ptx.mul.rn.f32x2(mul[0], mul[0], K.cuda.make_float2(scale, scale))
                                K.ptx.mov.b32(fragment[pair * 2], K.cuda.float2_x(mul[0]))
                                K.ptx.mov.b32(fragment[pair * 2 + 1], K.cuda.float2_y(mul[0]))
                    _prefill_tmem_st_128x64_f32(
                        tmem_base[0], PREFILL_OPT_TMEM_Q_STATE_COL, cg1_thread, fragment
                    )
                    K.ptx.tcgen05.wait__st.sync.aligned()
                    p_qstate.empty.arrive(st_qstate_c.stage)
                    st_qstate_c.advance()

                    p_cg1.full.wait(st_shared.stage, st_shared.phase)
                    with K.If(lane == 0), K.Then():
                        p_v.empty.arrive(st_v.stage)
                    _mn_opt_tmem_ld_128x64(
                        tmem_base[0], PREFILL_OPT_TMEM_CG1_ACC_COL, cg1_thread, fragment
                    )
                    nv_words = K.alloc_local((32,), "uint32")
                    with K.unroll(32) as pair:
                        K.ptx.mov.b32(
                            nv_words[pair],
                            _mn_opt_pack_iox2(fragment[pair * 2], fragment[pair * 2 + 1], io_dtype),
                        )
                    p_cg1.empty.arrive(st_shared.stage)
                    st_shared.advance()
                    with K.unroll(2) as row_half:
                        with K.unroll(8) as factor_group:
                            with K.unroll(2) as factor_repeat:
                                pair = row_half * 16 + factor_group * 2 + factor_repeat
                                mul = K.alloc_local((1,), "uint64")
                                K.ptx.mul.rn.f32x2(
                                    mul[0],
                                    K.cuda.make_float2(fragment[pair * 2], fragment[pair * 2 + 1]),
                                    K.cuda.make_float2(
                                        decay_factor[factor_group * 2],
                                        decay_factor[factor_group * 2 + 1],
                                    ),
                                )
                                K.ptx.mov.b32(fragment[pair * 2], K.cuda.float2_x(mul[0]))
                                K.ptx.mov.b32(fragment[pair * 2 + 1], K.cuda.float2_y(mul[0]))
                    decay_words = K.alloc_local((32,), "uint32")
                    for row_half in range(2):
                        _mn_opt_tmem_st_128x64_io_half(
                            tmem_base[0],
                            PREFILL_OPT_TMEM_SHARED_INPUT_COL,
                            cg1_thread,
                            row_half,
                            nv_words,
                        )
                        with K.unroll(16) as pair_in_half:
                            pair = row_half * 16 + pair_in_half
                            K.ptx.mov.b32(
                                decay_words[pair],
                                _mn_opt_pack_iox2(
                                    fragment[pair * 2], fragment[pair * 2 + 1], io_dtype
                                ),
                            )
                        _mn_opt_tmem_st_128x64_io_half(
                            tmem_base[0],
                            PREFILL_OPT_TMEM_SHARED_INPUT_COL + 32,
                            cg1_thread,
                            row_half,
                            decay_words,
                        )
                    K.ptx.tcgen05.wait__st.sync.aligned()
                    m_nv.arrive(st_nv.stage)
                    m_decay.arrive(st_decay.stage)

                    p_o.empty.wait(st_o.stage, st_o.phase)
                    p_qstate.full.wait(st_qstate_c.stage, st_qstate_c.phase)
                    _mn_opt_tmem_ld_128x64(
                        tmem_base[0], PREFILL_OPT_TMEM_Q_STATE_COL, cg1_thread, fragment
                    )
                    o_words = K.alloc_local((32,), "uint32")
                    with K.unroll(32) as pair:
                        K.ptx.mov.b32(
                            o_words[pair],
                            _mn_opt_pack_iox2(fragment[pair * 2], fragment[pair * 2 + 1], io_dtype),
                        )
                    _prefill_opt_store_o_fragment(s_o, st_o.stage, cg1_thread, o_words)
                    K.ptx.fence.proxy.async_.shared__cta()
                    p_qstate.empty.arrive(st_qstate_c.stage)
                    st_qstate_c.advance()
                    p_o.full.arrive(st_o.stage)

                    st_gate.advance()
                    st_v.advance()
                    st_kv_c.advance()
                    st_state_input.advance()
                    st_vks.advance()
                    st_nv.advance()
                    st_decay.advance()
                    st_o.advance()

                p_kv.full.wait(st_kv_c.stage, st_kv_c.phase)
                final_state_drain = K.alloc_local((128,), "float32")
                for state_sub in range(4):
                    _mn_opt_tmem_ld_matrix_sub(
                        tmem_base[0],
                        PREFILL_OPT_TMEM_STATE_COL,
                        cg1_thread,
                        state_sub,
                        final_state_drain,
                        state_sub * 32,
                    )
                p_kv.empty.arrive(st_kv_c.stage)

            K.cuda.warpgroup_sync(PREFILL_OPT_TMEM_DEALLOC_BARRIER)
            with K.If(K.warp_id_in_role() == 0), K.Then():
                K.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                K.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    K.Cast("uint32", tmem_base[0]), K.uint32(TMEM_COLUMNS)
                )
            for _ in range(2):
                p_o.empty.wait(st_o.stage, st_o.phase)
                st_o.advance()
            p_state_input.empty.wait(st_state_input.stage, st_state_input.phase)

        with mma0:
            K.ptx.prefetch.tensormap(K.address_of(q_map))
            K.ptx.prefetch.tensormap(K.address_of(k_map))
            K.ptx.prefetch.tensormap(K.address_of(v_map))
            K.ptx.prefetch.tensormap(K.address_of(t_map))
            K.ptx.prefetch.tensormap(K.address_of(o_map))
            K.ptx.barrier.sync(K.uint32(PREFILL_OPT_TMEM_ALLOC_BARRIER), K.uint32(320))
            tmem_base = K.alloc_local((1,), "int32")
            K.ptx.ld.volatile.shared.s32(tmem_base[0], K.address_of(tmem_holding[0]))
            st_acc = K.PipelineState(2, phase=1)
            st_k = K.PipelineState(3, phase=0)
            st_q = K.PipelineState(2, phase=0)
            with K.serial(padded_chunks):
                p_cg0.empty.wait(st_acc.stage, st_acc.phase)
                p_k.full.wait(st_k.stage, st_k.phase)
                p_q.full.wait(st_q.stage, st_q.phase)
                k_desc = _mn_opt_smem_desc_k(
                    K.cuda.cvta_generic_to_shared(s_k[st_k.stage].ptr_to(0, 0))
                )
                q_desc = _mn_opt_smem_desc_k(
                    K.cuda.cvta_generic_to_shared(s_q[st_q.stage].ptr_to(0, 0))
                )
                _prefill_opt_mma_ss_64x64_k128(
                    tmem_base[0] + PREFILL_OPT_TMEM_CG0_ACC_COL + st_acc.stage * 64,
                    q_desc,
                    k_desc,
                    p_cg0.full.ptr_to([st_acc.stage]),
                    io_dtype,
                )
                _prefill_opt_mma_commit(p_q.empty.ptr_to([st_q.stage]))
                _prefill_opt_mma_commit(p_k.empty.ptr_to([st_k.stage]))
                st_acc.advance()
                st_k.advance()
                st_q.advance()
            for _ in range(2):
                p_cg0.empty.wait(st_acc.stage, st_acc.phase)
                st_acc.advance()

        with mma1:
            K.ptx.barrier.sync(K.uint32(PREFILL_OPT_TMEM_ALLOC_BARRIER), K.uint32(320))
            tmem_base = K.alloc_local((1,), "int32")
            K.ptx.ld.volatile.shared.s32(tmem_base[0], K.address_of(tmem_holding[0]))
            st_cg1 = K.PipelineState(1, phase=1)
            st_qstate = K.PipelineState(1, phase=1)
            st_kv = K.PipelineState(1, phase=1)
            st_k = K.PipelineState(3, phase=0)
            st_q = K.PipelineState(2, phase=0)
            st_ainv = K.PipelineState(3, phase=0)
            st_qk = K.PipelineState(2, phase=0)
            st_state_input = K.PipelineState(1, phase=0)
            st_vks = K.PipelineState(1, phase=0)
            st_nv = K.PipelineState(1, phase=0)
            st_decay = K.PipelineState(1, phase=0)
            with K.serial(padded_chunks) as chunk:
                p_k.full.wait(st_k.stage, st_k.phase)
                p_q.full.wait(st_q.stage, st_q.phase)
                k_desc = _mn_opt_smem_desc_k(
                    K.cuda.cvta_generic_to_shared(s_k[st_k.stage].ptr_to(0, 0))
                )
                q_desc = _mn_opt_smem_desc_k(
                    K.cuda.cvta_generic_to_shared(s_q[st_q.stage].ptr_to(0, 0))
                )

                p_cg1.empty.wait(st_cg1.stage, st_cg1.phase)
                p_state_input.full.wait(st_state_input.stage, st_state_input.phase)
                _prefill_opt_mma_ts_128x64_k128(
                    tmem_base[0] + PREFILL_OPT_TMEM_CG1_ACC_COL,
                    tmem_base[0] + PREFILL_OPT_TMEM_STATE_INPUT_COL,
                    k_desc,
                    p_cg1.full.ptr_to([st_cg1.stage]),
                    io_dtype,
                )
                st_cg1.advance()

                p_qstate.empty.wait(st_qstate.stage, st_qstate.phase)
                _prefill_opt_mma_ts_128x64_k128(
                    tmem_base[0] + PREFILL_OPT_TMEM_Q_STATE_COL,
                    tmem_base[0] + PREFILL_OPT_TMEM_STATE_INPUT_COL,
                    q_desc,
                    p_qstate.full.ptr_to([st_qstate.stage]),
                    io_dtype,
                )
                _prefill_opt_mma_commit(p_state_input.empty.ptr_to([st_state_input.stage]))
                _prefill_opt_mma_commit(p_q.empty.ptr_to([st_q.stage]))
                st_qstate.advance()
                st_state_input.advance()

                p_cg1.empty.wait(st_cg1.stage, st_cg1.phase)
                m_vks.wait(st_vks.stage, st_vks.phase)
                p_ainv.full.wait(st_ainv.stage, st_ainv.phase)
                ainv_desc = _mn_opt_smem_desc_k(
                    K.cuda.cvta_generic_to_shared(s_ainv[st_ainv.stage].ptr_to(0, 0))
                )
                _prefill_opt_mma_ts_128x64_k64(
                    tmem_base[0] + PREFILL_OPT_TMEM_CG1_ACC_COL,
                    tmem_base[0] + PREFILL_OPT_TMEM_SHARED_INPUT_COL,
                    ainv_desc,
                    0,
                    io_dtype,
                )
                _prefill_opt_mma_commit(p_cg1.full.ptr_to([st_cg1.stage]))
                _prefill_opt_mma_commit(p_ainv.empty.ptr_to([st_ainv.stage]))
                st_cg1.advance()
                st_vks.advance()
                st_ainv.advance()

                p_qstate.empty.wait(st_qstate.stage, st_qstate.phase)
                p_qk.full.wait(st_qk.stage, st_qk.phase)
                m_nv.wait(st_nv.stage, st_nv.phase)
                qk_desc = _mn_opt_smem_desc_k(
                    K.cuda.cvta_generic_to_shared(s_qk[st_qk.stage].ptr_to(0, 0))
                )
                _prefill_opt_mma_ts_128x64_k64(
                    tmem_base[0] + PREFILL_OPT_TMEM_Q_STATE_COL,
                    tmem_base[0] + PREFILL_OPT_TMEM_SHARED_INPUT_COL,
                    qk_desc,
                    1,
                    io_dtype,
                )
                _prefill_opt_mma_commit(p_qk.empty.ptr_to([st_qk.stage]))
                _prefill_opt_mma_commit(p_qstate.full.ptr_to([st_qstate.stage]))
                st_qstate.advance()
                st_qk.advance()
                st_nv.advance()

                with K.If(chunk == 0), K.Then():
                    st_kv.advance()
                p_kv.empty.wait(st_kv.stage, st_kv.phase)
                m_decay.wait(st_decay.stage, st_decay.phase)
                k_desc_mn = _mn_opt_smem_desc_mn(
                    K.cuda.cvta_generic_to_shared(s_k[st_k.stage].ptr_to(0, 0))
                )
                _prefill_opt_mma_ts_128x128_k64(
                    tmem_base[0] + PREFILL_OPT_TMEM_STATE_COL,
                    tmem_base[0] + PREFILL_OPT_TMEM_SHARED_INPUT_COL + 32,
                    k_desc_mn,
                    p_kv.full.ptr_to([st_kv.stage]),
                    io_dtype,
                )
                _prefill_opt_mma_commit(p_k.empty.ptr_to([st_k.stage]))
                st_kv.advance()
                st_k.advance()
                st_q.advance()
                st_decay.advance()
            p_cg1.empty.wait(st_cg1.stage, st_cg1.phase)
            p_qstate.empty.wait(st_qstate.stage, st_qstate.phase)
            p_kv.empty.wait(st_kv.stage, st_kv.phase)

        with tma:
            st_q = K.PipelineState(2, phase=1)
            st_k = K.PipelineState(3, phase=1)
            st_v = K.PipelineState(3, phase=1)
            st_t = K.PipelineState(2, phase=1)
            with K.If(K.cuda.elect_sync()), K.Then():
                copy_descriptor(q_map, descriptor_q)
            K.cuda.warp_sync()
            with K.If(K.cuda.elect_sync()), K.Then():
                copy_descriptor(k_map, descriptor_k)
            K.cuda.warp_sync()
            with K.If(K.cuda.elect_sync()), K.Then():
                copy_descriptor(v_map, descriptor_v)
            K.cuda.warp_sync()
            K.ptx.fence.acq_rel.cta()
            with K.If(K.cuda.elect_sync()), K.Then():
                K.ptx.cp.async_.bulk.commit_group()
                K.ptx.cp.async_.bulk.wait_group.read(0)
            K.cuda.warp_sync()
            with K.If(K.cuda.elect_sync()), K.Then():
                if is_gqa:
                    replace_descriptor(
                        descriptor_q,
                        K.address_of(q[0]),
                        chunk_end,
                        head_ratio,
                        head_base,
                        2 * D_HEAD * q_heads,
                        2 * D_HEAD,
                        2 * D_HEAD * head_ratio,
                    )
                    replace_descriptor(
                        descriptor_k,
                        K.address_of(k[0]),
                        chunk_end,
                        k_heads,
                        1,
                        2 * D_HEAD * k_heads,
                        2 * D_HEAD,
                        0,
                    )
                    replace_descriptor(
                        descriptor_v,
                        K.address_of(v[0]),
                        chunk_end,
                        head_base,
                        1,
                        2 * D_HEAD * v_heads,
                        2 * D_HEAD,
                        0,
                    )
                else:
                    replace_descriptor(
                        descriptor_q,
                        K.address_of(q[0]),
                        chunk_end,
                        head_base,
                        1,
                        2 * D_HEAD * q_heads,
                        2 * D_HEAD,
                        0,
                    )
                    replace_descriptor(
                        descriptor_k,
                        K.address_of(k[0]),
                        chunk_end,
                        k_heads,
                        1,
                        2 * D_HEAD * k_heads,
                        2 * D_HEAD,
                        0,
                    )
                    replace_descriptor(
                        descriptor_v,
                        K.address_of(v[0]),
                        chunk_end,
                        head_ratio,
                        head_base,
                        2 * D_HEAD * v_heads,
                        2 * D_HEAD,
                        2 * D_HEAD * head_ratio,
                    )
            K.cuda.warp_sync()
            K.ptx.fence.proxy.tensormap__generic.release.gpu()

            with K.serial(padded_chunks) as chunk:
                token = chunk_start + chunk * T_BLOCK
                p_k.empty.wait(st_k.stage, st_k.phase)
                with K.If(chunk == 0), K.Then():
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx.fence.proxy.tensormap__generic.acquire.gpu(descriptor_k)
                with K.If(K.cuda.elect_sync()), K.Then():
                    p_k.full.arrive(st_k.stage, tx_count=16384)
                    for d_coord in range(0, D_HEAD, 64):
                        K.ptx[_PREFILL_OPT_TMA_G2S[3]](
                            s_k[st_k.stage].ptr_to(0, d_coord),
                            descriptor_k,
                            K.int32(d_coord),
                            K.Cast("int32", token),
                            k_head,
                            p_k.full.ptr_to([st_k.stage]),
                            K.uint64(0),
                        )
                st_k.advance()

                p_q.empty.wait(st_q.stage, st_q.phase)
                with K.If(chunk == 0), K.Then():
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx.fence.proxy.tensormap__generic.acquire.gpu(descriptor_q)
                with K.If(K.cuda.elect_sync()), K.Then():
                    p_q.full.arrive(st_q.stage, tx_count=16384)
                    for d_coord in range(0, D_HEAD, 64):
                        if is_gqa:
                            K.ptx[_PREFILL_OPT_TMA_G2S[4]](
                                s_q[st_q.stage].ptr_to(0, d_coord),
                                descriptor_q,
                                K.int32(d_coord),
                                K.Cast("int32", token),
                                subhead,
                                base_head,
                                p_q.full.ptr_to([st_q.stage]),
                                K.uint64(0),
                            )
                        else:
                            K.ptx[_PREFILL_OPT_TMA_G2S[3]](
                                s_q[st_q.stage].ptr_to(0, d_coord),
                                descriptor_q,
                                K.int32(d_coord),
                                K.Cast("int32", token),
                                base_head,
                                p_q.full.ptr_to([st_q.stage]),
                                K.uint64(0),
                            )
                st_q.advance()

                p_v.empty.wait(st_v.stage, st_v.phase)
                with K.If(chunk == 0), K.Then():
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx.fence.proxy.tensormap__generic.acquire.gpu(descriptor_v)
                with K.If(K.cuda.elect_sync()), K.Then():
                    p_v.full.arrive(st_v.stage, tx_count=16384)
                    for d_coord in range(0, D_HEAD, 64):
                        if is_gqa:
                            K.ptx[_PREFILL_OPT_TMA_G2S[3]](
                                s_v[st_v.stage].ptr_to(d_coord, 0),
                                descriptor_v,
                                K.int32(d_coord),
                                K.Cast("int32", token),
                                base_head,
                                p_v.full.ptr_to([st_v.stage]),
                                K.uint64(0),
                            )
                        else:
                            K.ptx[_PREFILL_OPT_TMA_G2S[4]](
                                s_v[st_v.stage].ptr_to(d_coord, 0),
                                descriptor_v,
                                K.int32(d_coord),
                                K.Cast("int32", token),
                                subhead,
                                base_head,
                                p_v.full.ptr_to([st_v.stage]),
                                K.uint64(0),
                            )
                st_v.advance()

                p_t.empty.wait(st_t.stage, st_t.phase)
                t_chunk = K.alloc_local((1,), "int32")
                K.assign(t_chunk[0], chunk)
                with K.If(chunk >= num_valid_chunks), K.Then():
                    K.assign(t_chunk[0], num_valid_chunks - 1)
                with K.If(K.cuda.elect_sync()), K.Then():
                    p_t.full.arrive(st_t.stage, tx_count=8192)
                    K.ptx[_PREFILL_OPT_TMA_G2S[5]](
                        s_t[st_t.stage].ptr_to(0, 0),
                        K.address_of(t_map),
                        K.int32(0),
                        K.int32(0),
                        subhead,
                        base_head,
                        t_block_start + t_chunk[0],
                        p_t.full.ptr_to([st_t.stage]),
                        K.uint64(0),
                    )
                st_t.advance()
            for _ in range(2):
                p_q.empty.wait(st_q.stage, st_q.phase)
                st_q.advance()
            for _ in range(3):
                p_k.empty.wait(st_k.stage, st_k.phase)
                st_k.advance()
            for _ in range(3):
                p_v.empty.wait(st_v.stage, st_v.phase)
                st_v.advance()
            for _ in range(2):
                p_t.empty.wait(st_t.stage, st_t.phase)
                st_t.advance()

        with aux:
            st_gate = K.PipelineState(5, phase=1)
            st_o = K.PipelineState(2, phase=0)

            def load_gate(chunk_offset, is_last_tile):
                pos0 = K.Cast("int64", chunk_offset) + K.Cast("int64", lane)
                pos1 = K.Cast("int64", chunk_offset) + K.Cast("int64", lane + 32)
                gate = K.alloc_local((2,), "float32")
                K.ptx.mov.b32(gate[0], K.float32(1.0))
                K.ptx.mov.b32(gate[1], K.float32(1.0))
                with K.If(is_last_tile):
                    with K.Then():
                        with K.If(pos0 < K.Cast("int64", chunk_end)), K.Then():
                            K.ptx.ld.global_.f32(
                                gate[0],
                                alpha.ptr_to([pos0 * K.Cast("int64", state_heads) + state_head]),
                            )
                        with K.If(pos1 < K.Cast("int64", chunk_end)), K.Then():
                            K.ptx.ld.global_.f32(
                                gate[1],
                                alpha.ptr_to([pos1 * K.Cast("int64", state_heads) + state_head]),
                            )
                    with K.Else():
                        K.ptx.ld.global_.f32(
                            gate[0],
                            alpha.ptr_to([pos0 * K.Cast("int64", state_heads) + state_head]),
                        )
                        K.ptx.ld.global_.f32(
                            gate[1],
                            alpha.ptr_to([pos1 * K.Cast("int64", state_heads) + state_head]),
                        )
                K.ptx.lg2.approx.ftz.f32(gate[0], gate[0] + K.float32(1.0e-10))
                K.ptx.lg2.approx.ftz.f32(gate[1], gate[1] + K.float32(1.0e-10))
                with K.unroll(5) as scan_step:
                    scan_offset = 1 << scan_step
                    prior = K.alloc_local((2,), "float32")
                    K.ptx.mov.b32(
                        prior[0],
                        K.tvm_warp_shuffle_up(K.uint32(0xFFFFFFFF), gate[0], scan_offset, 32, 32),
                    )
                    K.ptx.mov.b32(
                        prior[1],
                        K.tvm_warp_shuffle_up(K.uint32(0xFFFFFFFF), gate[1], scan_offset, 32, 32),
                    )
                    with K.If(lane >= scan_offset), K.Then():
                        K.ptx.mov.b32(gate[0], gate[0] + prior[0])
                        K.ptx.mov.b32(gate[1], gate[1] + prior[1])
                K.ptx.mov.b32(
                    gate[1], gate[1] + K.cuda._shfl_sync(K.uint32(0xFFFFFFFF), gate[0], 31, 32)
                )
                cumprod = K.alloc_local((2,), "float32")
                K.ptx.ex2.approx.ftz.f32(cumprod[0], gate[0])
                K.ptx.ex2.approx.ftz.f32(cumprod[1], gate[1])
                p_gate.empty.wait(st_gate.stage, st_gate.phase)
                K.ptx.st.shared.f32(s_cumsumlog.ptr_to([st_gate.stage, lane]), gate[0])
                K.ptx.st.shared.f32(s_cumsumlog.ptr_to([st_gate.stage, lane + 32]), gate[1])
                K.ptx.st.shared.f32(s_cumprod.ptr_to([st_gate.stage, lane]), cumprod[0])
                K.ptx.st.shared.f32(s_cumprod.ptr_to([st_gate.stage, lane + 32]), cumprod[1])
                p_gate.full.arrive(st_gate.stage)
                st_gate.advance()

            def store_o(chunk_offset):
                p_o.full.wait(st_o.stage, st_o.phase)
                with K.If(K.cuda.elect_sync()), K.Then():
                    for d_coord in range(0, D_HEAD, 64):
                        K.ptx[_PREFILL_OPT_TMA_S2G_4D](
                            descriptor_o,
                            K.int32(d_coord),
                            K.Cast("int32", chunk_offset),
                            subhead,
                            base_head,
                            s_o[st_o.stage].ptr_to(d_coord, 0),
                            K.uint64(0),
                        )
                    K.ptx.cp.async_.bulk.commit_group()
                    K.ptx.cp.async_.bulk.wait_group.read(0)
                p_o.empty.arrive(st_o.stage)
                st_o.advance()

            with K.If(K.cuda.elect_sync()), K.Then():
                copy_descriptor(o_map, descriptor_o)
            K.cuda.warp_sync()
            K.ptx.fence.acq_rel.cta()
            with K.If(K.cuda.elect_sync()), K.Then():
                K.ptx.cp.async_.bulk.commit_group()
                K.ptx.cp.async_.bulk.wait_group.read(0)
            K.cuda.warp_sync()
            with K.If(K.cuda.elect_sync()), K.Then():
                replace_descriptor(
                    descriptor_o,
                    K.address_of(o[0]),
                    chunk_end,
                    head_ratio,
                    head_base,
                    2 * D_HEAD * state_heads,
                    2 * D_HEAD,
                    2 * D_HEAD * head_ratio,
                )
            K.cuda.warp_sync()
            K.ptx.fence.proxy.tensormap__generic.release.gpu()
            with K.If(K.cuda.elect_sync()), K.Then():
                K.ptx.fence.proxy.tensormap__generic.acquire.gpu(descriptor_o)
            with K.If(chunk_len[0] > 0), K.Then():
                with K.unroll(2) as prefetch:
                    load_gate(chunk_start + prefetch * T_BLOCK, prefetch >= num_valid_chunks - 1)
                with K.If(padded_chunks > 2), K.Then():
                    with K.unroll(2, 4) as prefetch:
                        load_gate(
                            chunk_start + prefetch * T_BLOCK, prefetch >= num_valid_chunks - 1
                        )
                with K.serial(padded_chunks) as chunk:
                    future = chunk + 4
                    with K.If(future < padded_chunks), K.Then():
                        load_gate(chunk_start + future * T_BLOCK, future >= num_valid_chunks - 1)
                    store_o(chunk_start + chunk * T_BLOCK)
            for _ in range(5):
                p_gate.empty.wait(st_gate.stage, st_gate.phase)
                st_gate.advance()

    return prefill


@dataclass(frozen=True, slots=True)
class GDNCPPrefillSM100Config:
    label: str
    dtype: str
    q_heads: int
    k_heads: int
    v_heads: int
    seq_lens: tuple[int, ...]
    cu_seqlens_dtype: str
    state_mode: str
    state_dtype: str
    indexed_state: bool
    cp_chunk_len: int | None
    gate_baseline: float
    scale: str | float
    seed: int

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)

    @property
    def num_sequences(self) -> int:
        return len(self.seq_lens)

    @property
    def state_heads(self) -> int:
        return max(self.q_heads, self.v_heads)

    @property
    def needs_initial_state(self) -> bool:
        return self.state_mode in ("initial", "initial_final")

    @property
    def store_final_state(self) -> bool:
        return self.state_mode in ("final", "initial_final")

    def validate(self) -> None:
        if self.dtype not in ("float16", "bfloat16"):
            raise ValueError(f"unsupported IO dtype {self.dtype}")
        if self.cu_seqlens_dtype not in ("int32", "int64"):
            raise ValueError(f"unsupported cu_seqlens dtype {self.cu_seqlens_dtype}")
        if self.state_dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError(f"unsupported state dtype {self.state_dtype}")
        if not self.seq_lens or any(length < 0 for length in self.seq_lens):
            raise ValueError("seq_lens must be non-empty and non-negative")
        if max(self.seq_lens) <= 0:
            raise ValueError("at least one sequence must be non-empty")
        is_gqa = self.k_heads == self.v_heads and self.q_heads % self.k_heads == 0
        is_gva = self.k_heads == self.q_heads and self.v_heads % self.k_heads == 0
        if not (is_gqa or is_gva):
            raise ValueError(
                "head counts must use GQA (K=V, Q multiple of K) or "
                "GVA (K=Q, V multiple of K) topology"
            )
        if self.state_heads % self.q_heads or self.state_heads % self.v_heads:
            raise ValueError("Q/V heads must divide the expanded state-head count")
        if self.indexed_state and not (self.needs_initial_state or self.store_final_state):
            raise ValueError("indexed state requires an initial or final state")
        if self.cp_chunk_len is not None and (
            self.cp_chunk_len <= 0 or self.cp_chunk_len % T_BLOCK
        ):
            raise ValueError("cp_chunk_len must be a positive multiple of 64")


_CONFIG_FIELDS = {field.name for field in fields(GDNCPPrefillSM100Config)}
_TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _cfg(**kwargs: Any) -> GDNCPPrefillSM100Config:
    data = {key: value for key, value in kwargs.items() if key in _CONFIG_FIELDS}
    # The shared runner removes the presentation-only label before dispatching
    # run_test/run_bench.  Keep it out of the semantic configuration by using a
    # stable internal value when callers do not provide one.
    data.setdefault("label", "runtime")
    if "seq_lens" in data:
        data["seq_lens"] = tuple(int(length) for length in data["seq_lens"])
    cfg = GDNCPPrefillSM100Config(**data)
    cfg.validate()
    return cfg


class _AlignedTensorMap:
    """Host storage for one 64-byte-aligned, 128-byte TensorMap payload."""

    def __init__(self) -> None:
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_tensor_map(
    tensor: torch.Tensor,
    *,
    dtype: str,
    global_dims: tuple[int, ...],
    global_strides_bytes: tuple[int, ...],
    box_dims: tuple[int, ...] | None = None,
    swizzle: int = 3,
) -> _AlignedTensorMap:
    """Encode one source-shaped swizzled TMA tile."""
    import tvm

    rank = len(global_dims)
    if rank not in (3, 4, 5):
        raise ValueError(f"GDN CP TensorMap rank must be 3, 4, or 5, got {rank}")
    if len(global_strides_bytes) != rank - 1:
        raise ValueError("TensorMap global stride count must be rank - 1")
    if box_dims is None:
        box_dims = (64, 64, *((1,) * (rank - 2)))
    if len(box_dims) != rank:
        raise ValueError("TensorMap box dimension count must match rank")
    descriptor = _AlignedTensorMap()
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    encode(
        descriptor.ptr,
        dtype,
        rank,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *global_dims,
        *global_strides_bytes,
        *box_dims,
        *((1,) * rank),
        0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        swizzle,
        3,  # CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    )
    return descriptor


def _build_mn_tensor_maps(case: dict[str, Any]) -> dict[str, _AlignedTensorMap]:
    """Build the three immutable descriptors consumed by MN precompute."""
    cfg: GDNCPPrefillSM100Config = case["config"]
    spec = case["spec"]
    element_bytes = 2
    k_dims = (D_HEAD, cfg.total_tokens, cfg.k_heads)
    k_strides = (element_bytes * D_HEAD * cfg.k_heads, element_bytes * D_HEAD)
    v_dims = (D_HEAD, cfg.total_tokens, cfg.v_heads)
    v_strides = (element_bytes * D_HEAD * cfg.v_heads, element_bytes * D_HEAD)
    t_dims = (T_BLOCK, T_BLOCK, cfg.state_heads, spec["TOTAL_T_BLOCKS"])
    t_strides = (
        element_bytes * T_BLOCK,
        element_bytes * T_BLOCK * T_BLOCK,
        element_bytes * T_BLOCK * T_BLOCK * cfg.state_heads,
    )
    return {
        "k": _encode_tensor_map(
            case["k"], dtype=cfg.dtype, global_dims=k_dims, global_strides_bytes=k_strides
        ),
        "v": _encode_tensor_map(
            case["v"], dtype=cfg.dtype, global_dims=v_dims, global_strides_bytes=v_strides
        ),
        "t": _encode_tensor_map(
            case["t"], dtype=cfg.dtype, global_dims=t_dims, global_strides_bytes=t_strides
        ),
    }


def _build_fixup_tensor_maps(case: dict[str, Any]) -> dict[str, _AlignedTensorMap]:
    """Build the FP32 M/N descriptors consumed by UTC64 or UTC128 fixup."""
    cfg: GDNCPPrefillSM100Config = case["config"]
    spec = case["spec"]
    rows = 64 if spec["FIXUP_KIND"] == "fixup_utcmma64" else 128
    global_dims = (D_HEAD, D_HEAD, cfg.state_heads, spec["TOTAL_CP_CHUNKS"])
    global_strides = (4 * D_HEAD, 4 * D_HEAD * D_HEAD, 4 * D_HEAD * D_HEAD * cfg.state_heads)
    return {
        "transfer": _encode_tensor_map(
            case["transfer"],
            dtype="float32",
            global_dims=global_dims,
            global_strides_bytes=global_strides,
            box_dims=(32, D_HEAD, 1, 1),
            swizzle=4,  # CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B
        ),
        "local_state": _encode_tensor_map(
            case["local_state"],
            dtype="float32",
            global_dims=global_dims,
            global_strides_bytes=global_strides,
            box_dims=(32, rows, 1, 1),
        ),
    }


def _build_prefill_tensor_maps(case: dict[str, Any]) -> dict[str, _AlignedTensorMap]:
    """Build the five launch descriptors used by the CP prefill CTA."""
    cfg: GDNCPPrefillSM100Config = case["config"]
    spec = case["spec"]
    element_bytes = 2
    state_heads = cfg.state_heads
    base_heads = min(cfg.q_heads, cfg.v_heads)
    head_ratio = state_heads // base_heads
    is_gqa = cfg.q_heads >= cfg.v_heads

    if is_gqa:
        q_dims = (D_HEAD, cfg.total_tokens, head_ratio, base_heads)
        q_strides = (
            element_bytes * D_HEAD * cfg.q_heads,
            element_bytes * D_HEAD,
            element_bytes * D_HEAD * head_ratio,
        )
        v_dims = (D_HEAD, cfg.total_tokens, cfg.v_heads)
        v_strides = (element_bytes * D_HEAD * cfg.v_heads, element_bytes * D_HEAD)
    else:
        q_dims = (D_HEAD, cfg.total_tokens, cfg.q_heads)
        q_strides = (element_bytes * D_HEAD * cfg.q_heads, element_bytes * D_HEAD)
        v_dims = (D_HEAD, cfg.total_tokens, head_ratio, base_heads)
        v_strides = (
            element_bytes * D_HEAD * cfg.v_heads,
            element_bytes * D_HEAD,
            element_bytes * D_HEAD * head_ratio,
        )

    k_dims = (D_HEAD, cfg.total_tokens, cfg.k_heads)
    k_strides = (element_bytes * D_HEAD * cfg.k_heads, element_bytes * D_HEAD)
    t_dims = (T_BLOCK, T_BLOCK, head_ratio, base_heads, spec["TOTAL_T_BLOCKS"])
    t_strides = (
        element_bytes * T_BLOCK,
        element_bytes * T_BLOCK * T_BLOCK,
        element_bytes * T_BLOCK * T_BLOCK * head_ratio,
        element_bytes * T_BLOCK * T_BLOCK * state_heads,
    )
    o_dims = (D_HEAD, cfg.total_tokens, head_ratio, base_heads)
    o_strides = (
        element_bytes * D_HEAD * state_heads,
        element_bytes * D_HEAD,
        element_bytes * D_HEAD * head_ratio,
    )
    return {
        "q": _encode_tensor_map(
            case["q"], dtype=cfg.dtype, global_dims=q_dims, global_strides_bytes=q_strides
        ),
        "k": _encode_tensor_map(
            case["k"], dtype=cfg.dtype, global_dims=k_dims, global_strides_bytes=k_strides
        ),
        "v": _encode_tensor_map(
            case["v"], dtype=cfg.dtype, global_dims=v_dims, global_strides_bytes=v_strides
        ),
        "t": _encode_tensor_map(
            case["t"], dtype=cfg.dtype, global_dims=t_dims, global_strides_bytes=t_strides
        ),
        "o": _encode_tensor_map(
            case["output"], dtype=cfg.dtype, global_dims=o_dims, global_strides_bytes=o_strides
        ),
    }


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _chunk_bound(num_items: int, total: int, chunk_size: int) -> int:
    clipped = min(num_items, total)
    return clipped + (total - clipped) // chunk_size


def _choose_cp_chunk_len(cfg: GDNCPPrefillSM100Config, num_sms: int) -> int:
    if cfg.cp_chunk_len is not None:
        return cfg.cp_chunk_len
    granularity = 512
    max_seqlen = max(cfg.seq_lens)
    approx_ctas = _ceil_div(cfg.total_tokens, granularity) * cfg.state_heads
    # B200 is HBM: preserve the source 1/2 short-workload threshold.
    if approx_ctas * 2 < num_sms:
        square = max_seqlen * T_BLOCK
        balanced = math.isqrt(square)
        if balanced * balanced < square:
            balanced += 1
        return max(T_BLOCK, _ceil_div(balanced, T_BLOCK) * T_BLOCK)

    target_chunks = max(1, num_sms // cfg.state_heads)
    remaining_tokens = max(0, cfg.total_tokens - max_seqlen)
    remaining_sequences = max(0, cfg.num_sequences - 1)
    lo = 1
    hi = max(1, _ceil_div(max_seqlen, granularity))
    while lo < hi:
        mid = (lo + hi) // 2
        candidate = mid * granularity
        bounded = _ceil_div(max_seqlen, candidate) + _chunk_bound(
            remaining_sequences, remaining_tokens, candidate
        )
        if bounded <= target_chunks:
            hi = mid
        else:
            lo = mid + 1
    return lo * granularity


def _fixup_kind(cfg: GDNCPPrefillSM100Config, num_sms: int) -> str:
    parallel_states = cfg.num_sequences * cfg.state_heads
    if parallel_states <= num_sms * 2 // (D_HEAD // 4):
        return "fixup_simt_row4"
    if parallel_states <= num_sms // (D_HEAD // 64):
        return "fixup_utcmma64"
    return "fixup_utcmma128"


def _fixup_simt_rows(cfg: GDNCPPrefillSM100Config, num_sms: int) -> int:
    parallel_states = cfg.num_sequences * cfg.state_heads
    for rows_per_cta in (4, 2, 1):
        if parallel_states * (D_HEAD // rows_per_cta) >= num_sms:
            return rows_per_cta
    return 1


def _specialization(cfg: GDNCPPrefillSM100Config, device: str = "cuda") -> dict[str, Any]:
    del device
    from tirx_kernels.runner import hardware_num_sms

    num_sms = hardware_num_sms()
    cp_chunk_len = _choose_cp_chunk_len(cfg, num_sms)
    return {
        "IO_DTYPE": cfg.dtype,
        "CU_DTYPE": cfg.cu_seqlens_dtype,
        "STATE_DTYPE": cfg.state_dtype,
        "TOTAL_TOKENS": cfg.total_tokens,
        "NUM_SEQUENCES": cfg.num_sequences,
        "Q_HEADS": cfg.q_heads,
        "K_HEADS": cfg.k_heads,
        "V_HEADS": cfg.v_heads,
        "STATE_HEADS": cfg.state_heads,
        "STATE_POOL": cfg.num_sequences + 2 if cfg.indexed_state else cfg.num_sequences,
        "TOTAL_T_BLOCKS": _chunk_bound(cfg.num_sequences, cfg.total_tokens, T_BLOCK),
        "MAX_T_BLOCKS": _ceil_div(max(cfg.seq_lens), T_BLOCK),
        "TOTAL_CP_CHUNKS": _chunk_bound(cfg.num_sequences, cfg.total_tokens, cp_chunk_len),
        "MAX_CP_CHUNKS": _ceil_div(max(cfg.seq_lens), cp_chunk_len),
        "CP_CHUNK_LEN": cp_chunk_len,
        "IS_GQA": cfg.q_heads >= cfg.v_heads,
        "HEAD_BASE": min(cfg.q_heads, cfg.v_heads),
        "HEAD_RATIO": cfg.state_heads // min(cfg.q_heads, cfg.v_heads),
        "NEEDS_INITIAL_STATE": cfg.needs_initial_state,
        "STORE_FINAL_STATE": cfg.store_final_state,
        "USE_STATE_INDICES": cfg.indexed_state,
        "FIXUP_KIND": _fixup_kind(cfg, num_sms),
        "FIXUP_SIMT_ROWS": _fixup_simt_rows(cfg, num_sms),
    }


def get_kernel(**kwargs: Any) -> dict[str, Any]:
    """Build the six K-owned device variants used by the four-launch chain."""
    cfg = _cfg(**kwargs)
    spec = _specialization(cfg, kwargs.get("device", "cuda"))
    return {
        "t_precompute": _make_t_precompute(spec).func,
        "mn_precompute": _make_mn_precompute(spec).func,
        "fixup_simt_row4": _make_fixup_simt(spec).func,
        "fixup_utcmma64": _make_fixup_utcmma(spec, rows=64, m_stages=2, compute_regs=120).func,
        "fixup_utcmma128": _make_fixup_utcmma(spec, rows=128, m_stages=1, compute_regs=256).func,
        "prefill": _make_prefill(spec).func,
    }


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Allocate a deterministic CP chain and all persistent workspaces."""
    cfg = _cfg(**kwargs)
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for GDN CP prefill SM100")
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 0):
        raise SkipTest(f"GDN CP prefill requires SM100/B200, got {capability}")

    spec = _specialization(cfg, device)
    io_dtype = _TORCH_DTYPES[cfg.dtype]
    state_dtype = _TORCH_DTYPES[cfg.state_dtype]
    cu_dtype = _TORCH_DTYPES[cfg.cu_seqlens_dtype]
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    q = torch.randn(
        (cfg.total_tokens, cfg.q_heads, D_HEAD), dtype=io_dtype, device=device, generator=generator
    )
    k = F.normalize(
        torch.randn(
            (cfg.total_tokens, cfg.k_heads, D_HEAD),
            dtype=torch.float32,
            device=device,
            generator=generator,
        ),
        p=2.0,
        dim=-1,
    ).to(io_dtype)
    v = torch.randn(
        (cfg.total_tokens, cfg.v_heads, D_HEAD), dtype=io_dtype, device=device, generator=generator
    )
    alpha = cfg.gate_baseline + (1.0 - cfg.gate_baseline) * torch.rand(
        (cfg.total_tokens, cfg.state_heads), dtype=torch.float32, device=device, generator=generator
    )
    beta = cfg.gate_baseline + (1.0 - cfg.gate_baseline) * torch.rand(
        (cfg.total_tokens, cfg.state_heads), dtype=torch.float32, device=device, generator=generator
    )
    endpoints = [0]
    for length in cfg.seq_lens:
        endpoints.append(endpoints[-1] + length)
    cu_seqlens = torch.tensor(endpoints, dtype=cu_dtype, device=device)

    state_pool = spec["STATE_POOL"]
    state_shape = (state_pool, cfg.state_heads, D_HEAD, D_HEAD)
    initial_state = torch.zeros(state_shape, dtype=state_dtype, device=device)
    if cfg.needs_initial_state:
        initial_state.copy_(
            (
                torch.randn(state_shape, dtype=torch.float32, device=device, generator=generator)
                * 0.02
            ).to(state_dtype)
        )
    final_state = torch.zeros(state_shape, dtype=state_dtype, device=device)
    if cfg.indexed_state:
        indices = list(range(cfg.num_sequences))
        indices = [cfg.num_sequences + 1 - index for index in range(cfg.num_sequences)]
        state_indices = torch.tensor(indices, dtype=torch.int32, device=device)
    else:
        state_indices = torch.arange(cfg.num_sequences, dtype=torch.int32, device=device)

    t_workspace = torch.empty(
        (spec["TOTAL_T_BLOCKS"], cfg.state_heads, T_BLOCK, T_BLOCK), dtype=io_dtype, device=device
    )
    cp_shape = (spec["TOTAL_CP_CHUNKS"], cfg.state_heads, D_HEAD, D_HEAD)
    transfer = torch.empty(cp_shape, dtype=torch.float32, device=device)
    local_state = torch.empty_like(transfer)
    fixed_state = torch.zeros_like(transfer)
    initial_state_workspace = torch.zeros(
        (cfg.num_sequences, cfg.state_heads, D_HEAD, D_HEAD), dtype=torch.float32, device=device
    )
    descriptor_workspace = torch.empty(
        (
            cfg.num_sequences
            * cfg.state_heads
            * spec["MAX_CP_CHUNKS"]
            * DESCRIPTOR_SLOTS
            * DESCRIPTOR_SLOT_BYTES,
        ),
        dtype=torch.int8,
        device=device,
    )
    output = torch.empty((cfg.total_tokens, cfg.state_heads, D_HEAD), dtype=io_dtype, device=device)
    scale = 1.0 / math.sqrt(D_HEAD) if cfg.scale == "auto" else float(cfg.scale)
    case = {
        "config": cfg,
        "spec": spec,
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "alpha": alpha.contiguous(),
        "beta": beta.contiguous(),
        "cu_seqlens": cu_seqlens,
        "initial_state": initial_state,
        "final_state": final_state,
        "state_indices": state_indices,
        "t": t_workspace,
        "transfer": transfer,
        "local_state": local_state,
        "fixed_state": fixed_state,
        "initial_state_workspace": initial_state_workspace,
        "descriptor_workspace": descriptor_workspace,
        "output": output,
        "scale": scale,
    }
    case["mn_tensor_maps"] = _build_mn_tensor_maps(case)
    if spec["FIXUP_KIND"] != "fixup_simt_row4":
        case["fixup_tensor_maps"] = _build_fixup_tensor_maps(case)
    case["prefill_tensor_maps"] = _build_prefill_tensor_maps(case)
    return case


def _stage_args(case: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    mn_tensor_maps = case["mn_tensor_maps"]
    prefill_tensor_maps = case["prefill_tensor_maps"]
    fixup_args: tuple[Any, ...] = (
        case["transfer"].view(-1),
        case["local_state"].view(-1),
        case["initial_state"].view(-1),
        case["initial_state_workspace"].view(-1),
        case["fixed_state"].view(-1),
        case["final_state"].view(-1),
        case["state_indices"],
        case["cu_seqlens"],
    )
    if case["spec"]["FIXUP_KIND"] != "fixup_simt_row4":
        fixup_tensor_maps = case["fixup_tensor_maps"]
        fixup_args = (
            *fixup_args,
            fixup_tensor_maps["transfer"].ptr,
            fixup_tensor_maps["local_state"].ptr,
        )
    return {
        "t_precompute": (
            case["k"].view(-1),
            case["beta"].view(-1),
            case["t"].view(-1),
            case["cu_seqlens"],
            mn_tensor_maps["k"].ptr,
        ),
        "mn_precompute": (
            case["k"].view(-1),
            case["v"].view(-1),
            case["t"].view(-1),
            case["alpha"].view(-1),
            case["transfer"].view(-1),
            case["local_state"].view(-1),
            case["cu_seqlens"],
            mn_tensor_maps["k"].ptr,
            mn_tensor_maps["v"].ptr,
            mn_tensor_maps["t"].ptr,
        ),
        "fixup": fixup_args,
        "prefill": (
            case["q"].view(-1),
            case["k"].view(-1),
            case["v"].view(-1),
            case["alpha"].view(-1),
            case["t"].view(-1),
            case["fixed_state"].view(-1),
            case["initial_state_workspace"].view(-1),
            case["output"].view(-1),
            case["cu_seqlens"],
            case["scale"],
            prefill_tensor_maps["q"].ptr,
            prefill_tensor_maps["k"].ptr,
            prefill_tensor_maps["v"].ptr,
            prefill_tensor_maps["t"].ptr,
            prefill_tensor_maps["o"].ptr,
            case["descriptor_workspace"],
        ),
    }


@lru_cache(maxsize=1)
def _load_oracle():
    from flashinfer.gdn_kernels.blackwell.gdn_cp_prefill import cp_delta_rule_dsl_sm100

    return cp_delta_rule_dsl_sm100


def _run_oracle(
    case: dict[str, Any], output: torch.Tensor, final_state: torch.Tensor | None
) -> None:
    cfg: GDNCPPrefillSM100Config = case["config"]
    _load_oracle()(
        output,
        final_state if cfg.store_final_state else None,
        case["q"],
        case["k"],
        case["v"],
        case["alpha"],
        case["beta"],
        case["cu_seqlens"],
        case["scale"],
        initial_state=case["initial_state"] if cfg.needs_initial_state else None,
        state_indices=case["state_indices"] if cfg.indexed_state else None,
        max_seqlen=max(cfg.seq_lens),
        cp_chunk_len=case["spec"]["CP_CHUNK_LEN"],
    )


def _compile_selected(case: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    from tirx_kernels.runner import compile_kernel

    kernels = get_kernel(**kwargs)
    fixup_name = case["spec"]["FIXUP_KIND"]
    return {
        "t_precompute": compile_kernel(kernels["t_precompute"]),
        "mn_precompute": compile_kernel(kernels["mn_precompute"]),
        fixup_name: compile_kernel(kernels[fixup_name]),
        "prefill": compile_kernel(kernels["prefill"]),
    }


def _compile_for_config(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    spec = _specialization(cfg, kwargs.get("device", "cuda"))
    kernels = get_kernel(**kwargs)
    fixup_name = spec["FIXUP_KIND"]
    from tirx_kernels.runner import compile_kernel

    return {
        "t_precompute": compile_kernel(kernels["t_precompute"]),
        "mn_precompute": compile_kernel(kernels["mn_precompute"]),
        fixup_name: compile_kernel(kernels[fixup_name]),
        "prefill": compile_kernel(kernels["prefill"]),
    }


def _launch_chain(case: dict[str, Any], executable: dict[str, Any]) -> None:
    args = _stage_args(case)
    fixup_name = case["spec"]["FIXUP_KIND"]
    executable["t_precompute"](*args["t_precompute"])
    executable["mn_precompute"](*args["mn_precompute"])
    executable[fixup_name](*args["fixup"])
    executable["prefill"](*args["prefill"])


def run_test(**kwargs: Any) -> None:
    """Compile and compare the full four-launch chain with frozen FlashInfer."""
    case = prepare_data(**kwargs)
    executable = _compile_selected(case, **kwargs)
    _launch_chain(case, executable)
    torch.cuda.synchronize()

    reference_output = torch.empty_like(case["output"])
    reference_state = torch.zeros_like(case["final_state"])
    _run_oracle(case, reference_output, reference_state)
    torch.cuda.synchronize()

    for name, tensor in (("TIRx output", case["output"]), ("source output", reference_output)):
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} contains non-finite values")
    io_atol = 2e-2 if case["config"].dtype == "bfloat16" else 8e-3
    io_rtol = 2e-2 if case["config"].dtype == "bfloat16" else 8e-3
    torch.testing.assert_close(case["output"], reference_output, atol=io_atol, rtol=io_rtol)

    if case["config"].store_final_state:
        if case["config"].indexed_state:
            selected = case["state_indices"].to(torch.int64)
            got_state = case["final_state"].index_select(0, selected)
            expected_state = reference_state.index_select(0, selected)
        else:
            got_state = case["final_state"]
            expected_state = reference_state
        if not torch.isfinite(got_state).all() or not torch.isfinite(expected_state).all():
            raise AssertionError("final state contains non-finite values")
        state_atol = 4e-2 if case["config"].state_dtype != "float32" else 1e-2
        state_rtol = 4e-2 if case["config"].state_dtype != "float32" else 1e-2
        torch.testing.assert_close(got_state, expected_state, atol=state_atol, rtol=state_rtol)


def prepare_bench(**kwargs: Any):
    """Compile the selected four-launch chain before CUDA assignment."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": _compile_for_config(**kwargs)}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 1,
    cooldown_s: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Benchmark the prepared four-launch chain against the source chain."""
    kwargs = {**prepared["config"], **kwargs}
    from tirx_kernels.runner import bench

    case = prepare_data(**kwargs)
    executable = prepared["executable"]

    def source_builder():
        source_output = torch.empty_like(case["output"])
        source_state = torch.zeros_like(case["final_state"])
        _run_oracle(case, source_output, source_state)
        _run_oracle(case, source_output, source_state)
        torch.cuda.synchronize()

        def launch():
            _run_oracle(case, source_output, source_state)

        return launch

    return bench(
        {"tirx": lambda: _launch_chain(case, executable)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashinfer_cutedsl": source_builder},
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 1,
    cooldown_s: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Prepare and benchmark the complete four-launch chain."""
    return prepare_bench(**kwargs).run_gpu(
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
    "run_test",
]
