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

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, fields
from typing import Any
from unittest import SkipTest

import torch
import torch.nn.functional as F

from tvm.script import tirx as T

from . import gdn_prefill_sm100 as _base_prefill

D_HEAD = 128
T_BLOCK = 64
T_THREADS = 128
MN_THREADS = 384
PREFILL_THREADS = 384
DESCRIPTOR_SLOT_BYTES = 128
DESCRIPTOR_SLOTS = 5

# CP M/N precompute: the source keeps both affine recurrences in TMEM and
# dedicates one warp each to the two independent UMMA issue streams.  These
# offsets are the literal SharedStorage member offsets after its seventeen
# full/empty barrier rings.
MN_OPT_SMEM_TOTAL = 159744
MN_OPT_TMEM_COLUMNS = 512
MN_OPT_TMEM_HOLDING_OFF = 432
MN_OPT_K_OFF = 1024
MN_OPT_V_OFF = 50176
MN_OPT_T_OFF = 99328
MN_OPT_X_OFF = 123904
MN_OPT_ALPHA_OFF = 156672

MN_OPT_TMEM_M_COL = 0
MN_OPT_TMEM_N_COL = 128
MN_OPT_TMEM_SCRATCH_COL = 256
MN_OPT_TMEM_M_INPUT_COL = 320
MN_OPT_TMEM_N_INPUT_COL = 384
MN_OPT_TMEM_XY_COL = 448

MN_OPT_TMEM_ALLOC_BARRIER = 1
MN_OPT_TMEM_DEALLOC_BARRIER = 2

MN_OPT_PIPELINES = (
    # name, stages, full byte offset, empty byte offset, producers, consumers
    ("load_k", 3, 0, 24, 1, 2),
    ("load_v", 3, 48, 72, 1, 4),
    ("load_t", 3, 96, 120, 1, 1),
    ("alpha", 4, 144, 176, 32, 256),
    ("m_init", 1, 208, 216, 128, 1),
    ("n_init", 1, 224, 232, 128, 1),
    ("x_acc", 1, 240, 248, 1, 128),
    ("x_ready", 2, 256, 272, 128, 2),
    ("m_input", 1, 288, 296, 128, 1),
    ("n_input", 1, 304, 312, 128, 1),
    ("z_acc", 1, 320, 328, 1, 128),
    ("z_ready", 1, 336, 344, 128, 1),
    ("m_acc", 1, 352, 360, 1, 128),
    ("y_acc", 1, 368, 376, 1, 128),
    ("y_ready", 1, 384, 392, 128, 1),
    ("n_acc", 1, 400, 408, 1, 128),
    ("done", 1, 416, 424, 128, 128),
)

# CP prefill: exact SharedStorage order from the reviewed source sketch.  The
# first 496 bytes hold the sixteen full/empty mbarrier rings; matrix storage is
# aligned to 1024 bytes and TMEM uses all 512 columns.
PREFILL_OPT_SMEM_TOTAL = 224768
PREFILL_OPT_TMEM_COLUMNS = 512
PREFILL_OPT_TMEM_HOLDING_OFF = 496
PREFILL_OPT_Q_OFF = 1024
PREFILL_OPT_K_OFF = 33792
PREFILL_OPT_V_OFF = 82944
PREFILL_OPT_T_OFF = 132096
PREFILL_OPT_AINV_OFF = 148480
PREFILL_OPT_QK_OFF = 173056
PREFILL_OPT_O_OFF = 189440
PREFILL_OPT_CUMSUMLOG_OFF = 222208
PREFILL_OPT_CUMPROD_OFF = 223488

PREFILL_OPT_TMEM_STATE_COL = 0
PREFILL_OPT_TMEM_Q_STATE_COL = 128
PREFILL_OPT_TMEM_STATE_INPUT_COL = 192
PREFILL_OPT_TMEM_CG0_ACC_COL = 256
PREFILL_OPT_TMEM_CG1_ACC_COL = 384
PREFILL_OPT_TMEM_SHARED_INPUT_COL = 448

_PREFILL_PREDICATED_GAMMA_SRC = r"""
__forceinline__ __device__ float gdn_cp_prefill_predicated_gamma(
        uint32_t s_addr, uint32_t t_addr, uint32_t pred) {
    float gamma;
    asm volatile(
        "{ .reg .pred p; .reg .f32 s_log; .reg .f32 t_log; "
        "mov.f32 %0, 0f00000000; "
        "setp.ne.b32 p, %1, 0; "
        "@p ld.shared.f32 s_log, [%2]; "
        "@p ld.shared.f32 t_log, [%3]; "
        "@p sub.f32 %0, s_log, t_log; "
        "@p ex2.approx.ftz.f32 %0, %0; }"
        : "=f"(gamma)
        : "r"(pred), "r"(s_addr), "r"(t_addr)
        : "memory");
    return gamma;
}
"""

PREFILL_OPT_TMEM_ALLOC_BARRIER = 1
PREFILL_OPT_T_STORE_BARRIER = 2
PREFILL_OPT_TMEM_DEALLOC_BARRIER = 3
PREFILL_OPT_INITIAL_STATE_BARRIER = 4

PREFILL_OPT_PIPELINES = (
    # name, stages, full byte offset, empty byte offset, producers, consumers
    ("load_k", 3, 0, 24, 1, 2),
    ("load_q", 2, 48, 64, 1, 2),
    ("load_v", 3, 80, 104, 1, 4),
    ("load_gate", 5, 128, 168, 32, 256),
    ("load_t", 2, 208, 224, 1, 4),
    ("q_state_acc", 1, 240, 248, 1, 128),
    ("kv_acc", 1, 256, 264, 1, 128),
    ("cg0_acc", 2, 272, 288, 1, 128),
    ("cg1_acc", 1, 304, 312, 1, 128),
    ("ainv_ready", 3, 320, 344, 128, 1),
    ("qk_ready", 2, 368, 384, 128, 1),
    ("state_input", 1, 400, 408, 128, 1),
    ("vks_ready", 1, 416, 424, 128, 1),
    ("nv_ready", 1, 432, 440, 128, 1),
    ("decay_ready", 1, 448, 456, 128, 1),
    ("o_store", 2, 464, 480, 128, 32),
)

# The mature non-CP SM100 port already carries the exact low-level pipeline,
# TensorMap, TMEM-copy, and descriptor helpers used by the CP specialization.
# They are storage-offset agnostic (or share the same TMEM column ABI), so keep
# one implementation of these ISA spellings and specialize only the CP dataflow.
_pf_make_warp_uniform = _base_prefill._make_warp_uniform
_pf_byte_ptr = _base_prefill._byte_ptr
_pf_pipe_next_index = _base_prefill._pipe_next_index
_pf_pipe_next_phase = _base_prefill._pipe_next_phase
_pf_pipe_full_addr = _base_prefill._pipe_full_addr
_pf_pipe_empty_addr = _base_prefill._pipe_empty_addr
_pf_producer_acquire = _base_prefill._producer_acquire
_pf_consumer_wait = _base_prefill._consumer_wait
_pf_software_commit = _base_prefill._software_commit
_pf_consumer_release = _base_prefill._consumer_release
_pf_producer_acquire_state = _base_prefill._producer_acquire_state
_pf_consumer_wait_state = _base_prefill._consumer_wait_state
_pf_software_commit_state = _base_prefill._software_commit_state
_pf_consumer_release_state = _base_prefill._consumer_release_state
_pf_producer_tail = _base_prefill._producer_tail
_pf_producer_tail_state = _base_prefill._producer_tail_state
_pf_descriptor_copy_payload = _base_prefill._descriptor_copy_payload
_pf_replace_descriptor = _base_prefill._replace_descriptor
_pf_tensormap_release = _base_prefill._tensormap_release
_pf_tensormap_acquire = _base_prefill._tensormap_acquire
_pf_shared_addr = _base_prefill._shared_addr
_pf_smem_desc_b128 = _base_prefill._smem_desc_b128
_pf_smem_desc_k_trans_b128 = _base_prefill._smem_desc_k_trans_b128
_pf_cg0_tmem_ld = _base_prefill._cg0_tmem_ld
_pf_state_tmem_ld_sub = _base_prefill._state_tmem_ld_sub
_pf_state_tmem_st_sub = _base_prefill._state_tmem_st_sub
_pf_state_input_tmem_st_sub = _base_prefill._state_input_tmem_st_sub
_pf_cg1_tmem_ld_f32 = _base_prefill._cg1_tmem_ld_f32
_pf_cg1_tmem_st_f32 = _base_prefill._cg1_tmem_st_f32
_pf_cg1_tmem_st_io_half = _base_prefill._cg1_tmem_st_f16_half
_pf_cg1_tmem_st_io = _base_prefill._cg1_tmem_st_f16
_pf_cg1_smem_lane_byte = _base_prefill._cg1_smem_lane_byte
_pf_cg1_smem_second_half_delta = _base_prefill._cg1_smem_second_half_delta


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
    clipped = T.min(seq_idx, total)
    return clipped + (total - clipped) // chunk_size


def _load_global_s32(dst, address):
    return T.ptx.ld.global_.nc.b32(dst, address)


def _load_global_f32(dst, address):
    return T.ptx.ld.global_.nc.f32(dst, address)


def _store_global_f32(address, value):
    return T.ptx.st.global_.f32(address, value)


def _load_shared_f32(dst, address):
    return T.ptx.ld.shared.f32(dst, address)


def _store_shared_f32(address, value):
    return T.ptx.st.shared.f32(address, value)


@T.inline
def _load_global_as_f32(values, value_index, source, source_index, SOURCE_DTYPE):
    if SOURCE_DTYPE == "float32":
        T.ptx.ld.global_.f32(values[value_index], source.ptr_to([source_index]))
    else:
        bits = T.alloc_local((1,), "uint16")
        T.ptx.ld.global_.b16(bits[0], source.ptr_to([source_index]))
        values[value_index] = T.cast(T.reinterpret(SOURCE_DTYPE, bits[0]), "float32")


def _store_global_from_f32(output, output_index, value, OUTPUT_DTYPE):
    if OUTPUT_DTYPE == "float32":
        return T.ptx.st.global_.f32(output.ptr_to([output_index]), value)
    return T.ptx.st.global_.b16(
        output.ptr_to([output_index]), T.reinterpret("uint16", T.cast(value, OUTPUT_DTYPE))
    )


@T.inline
def _load_sequence_bounds(cu_seqlens, seq_idx, bounds, CU_DTYPE):
    if CU_DTYPE == "int32":
        T.ptx.ld.global_.nc.b32(bounds[0], cu_seqlens.ptr_to([seq_idx]))
        T.ptx.ld.global_.nc.b32(bounds[1], cu_seqlens.ptr_to([seq_idx + 1]))
    else:
        raw = T.alloc_local((2,), "int64")
        T.ptx.ld.global_.nc.b64(raw[0], cu_seqlens.ptr_to([seq_idx]))
        T.ptx.ld.global_.nc.b64(raw[1], cu_seqlens.ptr_to([seq_idx + 1]))
        bounds[0] = T.cast(raw[0], "int32")
        bounds[1] = T.cast(raw[1], "int32")


def _lg2_approx_ftz(value):
    result = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.lg2.approx.ftz.f32(result[0], value))
    return result[0]


def _ex2_approx_ftz(value):
    result = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.ex2.approx.ftz.f32(result[0], value))
    return result[0]


def _prefill_predicated_gamma(s_addr, t_addr, pred):
    return T.cuda.func_call(
        "gdn_cp_prefill_predicated_gamma",
        s_addr,
        t_addr,
        T.cast(pred, "uint32"),
        source_code=_PREFILL_PREDICATED_GAMMA_SRC,
        return_type="float32",
    )


@T.inline
def _compose_inverse_level(lower, inverse, inverse_tmp, tid, level, half, level_blocks):
    """Compose neighboring lower-triangular inverse tiles at one fixed level."""
    offdiag_elements = level_blocks * half * half
    for work in T.serial((offdiag_elements + T_THREADS - 1) // T_THREADS):
        linear_off: T.int32 = tid + work * T_THREADS
        if linear_off < offdiag_elements:
            block_level: T.int32 = linear_off // (half * half)
            within: T.int32 = linear_off % (half * half)
            row_half: T.int32 = within // half
            col_half: T.int32 = within % half
            row_global: T.int32 = block_level * level + half + row_half
            col_global: T.int32 = block_level * level + col_half
            first_product: T.float32 = 0.0
            for inner_half in T.serial(half):
                first_product = first_product + T.cast(
                    inverse[row_global * T_BLOCK + block_level * level + half + inner_half],
                    "float32",
                ) * T.cast(
                    lower[(block_level * level + half + inner_half) * T_BLOCK + col_global],
                    "float32",
                )
            inverse_tmp[row_global * T_BLOCK + col_global] = T.cast(-first_product, "float16")
    T.cuda.cta_sync()
    for work in T.serial((offdiag_elements + T_THREADS - 1) // T_THREADS):
        linear_off: T.int32 = tid + work * T_THREADS
        if linear_off < offdiag_elements:
            block_level: T.int32 = linear_off // (half * half)
            within: T.int32 = linear_off % (half * half)
            row_half: T.int32 = within // half
            col_half: T.int32 = within % half
            row_global: T.int32 = block_level * level + half + row_half
            col_global: T.int32 = block_level * level + col_half
            second_product: T.float32 = 0.0
            for inner_half in T.serial(half):
                second_product = second_product + T.cast(
                    inverse_tmp[row_global * T_BLOCK + block_level * level + inner_half], "float32"
                ) * T.cast(
                    inverse[(block_level * level + inner_half) * T_BLOCK + col_global], "float32"
                )
            inverse[row_global * T_BLOCK + col_global] = T.cast(second_product, "float16")
    T.cuda.cta_sync()


def _t_matrix_index(row, col):
    """K_SW128 element offset for a 64-row, one- or two-tile matrix."""
    tile_col: T.int32 = col & 63
    byte_offset: T.int32 = (row * T_BLOCK + tile_col) * 2
    swizzled: T.int32 = T.bitwise_xor(
        byte_offset, T.bitwise_and(T.shift_right(byte_offset, T.int32(3)), T.int32(112))
    )
    return (col >> 6) * (T_BLOCK * T_BLOCK) + swizzled // 2


def _t_matrix_ptr(storage, row, col):
    return storage.ptr_to([_t_matrix_index(row, col)])


def _t_k_matrix_ptr(smem_raw, row, col):
    """Source K_SW128 address, including the 128-byte SharedStorage offset."""
    byte_offset: T.int32 = (
        128 + (col >> 6) * (T_BLOCK * T_BLOCK * 2) + (row * T_BLOCK + (col & 63)) * 2
    )
    swizzled: T.int32 = T.bitwise_xor(
        byte_offset, T.bitwise_and(T.shift_right(byte_offset, T.int32(3)), T.int32(112))
    )
    return smem_raw.ptr_to([swizzled])


def _t_pack_f16x2(a, b):
    packed = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.cvt.rn.f16x2.f32(packed[0], b, a))
    return packed[0]


@T.inline
def _t_sub_zero_pack_f16x2(a, b, dst, dst_index):
    negated: T.uint64[1]
    T.ptx.sub.rn.f32x2(
        negated[0], T.cuda.make_float2(T.float32(0.0), T.float32(0.0)), T.cuda.make_float2(a, b)
    )
    dst[dst_index] = _t_pack_f16x2(T.cuda.float2_x(negated[0]), T.cuda.float2_y(negated[0]))


_T_MMA_ZERO_C = [T.float32(0.0)] * 4


def _t_mma_m16n8k16_f16_zero(acc, a, b, acc_off, b_off):
    return T.ptx["mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"](
        *[acc[acc_off + i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[b_off + i] for i in range(2)],
        *_T_MMA_ZERO_C,
    )


def _t_mma_m16n8k16_f16_acc(acc, a, b, acc_off, b_off):
    return T.ptx["mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"](
        *[acc[acc_off + i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[b_off + i] for i in range(2)],
        *[acc[acc_off + i] for i in range(4)],
    )


def _t_mma_m16n8k16_bf16_zero(acc, a, b, acc_off, b_off):
    return T.ptx["mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"](
        *[acc[acc_off + i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[b_off + i] for i in range(2)],
        *_T_MMA_ZERO_C,
    )


def _t_mma_m16n8k16_bf16_acc(acc, a, b, acc_off, b_off):
    return T.ptx["mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"](
        *[acc[acc_off + i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[b_off + i] for i in range(2)],
        *[acc[acc_off + i] for i in range(4)],
    )


def _t_mma_m16n8k8_f16_zero(acc, a, b):
    return T.ptx["mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"](
        *[acc[i] for i in range(4)], *[a[i] for i in range(2)], b[0], *_T_MMA_ZERO_C
    )


@T.inline
def _t_ldmatrix_x4(storage, base_row, base_col, lane, transpose, dst):
    lane_matrix: T.int32 = lane >> 3
    row: T.int32 = base_row + (lane & 7) + (lane_matrix & 1) * 8
    col: T.int32 = base_col + (lane_matrix >> 1) * 8
    T.ptx[f"ldmatrix.sync.aligned.m8n8.x4{'.trans' if transpose else ''}.shared.b16"](
        *[dst[i] for i in range(4)], _t_matrix_ptr(storage, row, col)
    )


@T.inline
def _t_ldmatrix_x4_kdot_b(storage, base_row, base_col, lane, dst):
    # CuTe's B partition swaps the lane bits selecting the second 8-row and
    # 8-column subtile.  The underlying ldmatrix remains non-transpose.
    row: T.int32 = base_row + (lane & 7) + ((lane >> 4) & 1) * 8
    col: T.int32 = base_col + ((lane >> 3) & 1) * 8
    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
        dst[0], dst[1], dst[2], dst[3], _t_matrix_ptr(storage, row, col)
    )


@T.inline
def _t_ldmatrix_x4_k_a(smem_raw, base_row, base_col, lane, dst):
    lane_matrix: T.int32 = lane >> 3
    row: T.int32 = base_row + (lane & 7) + (lane_matrix & 1) * 8
    col: T.int32 = base_col + (lane_matrix >> 1) * 8
    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
        dst[0], dst[1], dst[2], dst[3], _t_k_matrix_ptr(smem_raw, row, col)
    )


@T.inline
def _t_ldmatrix_x4_k_b(smem_raw, base_row, base_col, lane, dst):
    row: T.int32 = base_row + (lane & 7) + ((lane >> 4) & 1) * 8
    col: T.int32 = base_col + ((lane >> 3) & 1) * 8
    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
        dst[0], dst[1], dst[2], dst[3], _t_k_matrix_ptr(smem_raw, row, col)
    )


@T.inline
def _t_stmatrix_x4(storage, base_row, base_col, lane, src):
    lane_matrix: T.int32 = lane >> 3
    row: T.int32 = base_row + (lane & 7) + (lane_matrix & 1) * 8
    col: T.int32 = base_col + (lane_matrix >> 1) * 8
    T.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
        _t_matrix_ptr(storage, row, col), *[src[i] for i in range(4)]
    )


@T.inline
def _t_store_t_fragment(
    inverse_frag, beta_storage, t, t_base, valid_len, local_warp, lane, n_group, IO_DTYPE
):
    row_base: T.int32 = local_warp * 16 + (lane >> 2)
    col_base: T.int32 = n_group * 16 + (lane & 3) * 2
    for pair in T.unroll(4):
        row: T.int32 = row_base + (pair & 1) * 8
        col: T.int32 = col_base + (pair >> 1) * 8
        word: T.uint32 = inverse_frag[pair]
        inverse_lo: T.float32 = T.cast(
            T.reinterpret("float16", T.cast(word & T.uint32(0xFFFF), "uint16")), "float32"
        )
        inverse_hi: T.float32 = T.cast(
            T.reinterpret("float16", T.cast(word >> 16, "uint16")), "float32"
        )
        output_lo: T.float32 = 0.0
        output_hi: T.float32 = 0.0
        beta_value: T.float32
        if row < valid_len and col < valid_len:
            _load_shared_f32(beta_value, beta_storage.ptr_to([col]))
            output_lo = -beta_value * inverse_lo
        if row < valid_len and col + 1 < valid_len:
            _load_shared_f32(beta_value, beta_storage.ptr_to([col + 1]))
            output_hi = -beta_value * inverse_hi
        T.ptx.st.global_.b16(
            t.ptr_to([t_base + col * T_BLOCK + row]),
            T.reinterpret("uint16", T.cast(output_lo, IO_DTYPE)),
        )
        T.ptx.st.global_.b16(
            t.ptr_to([t_base + (col + 1) * T_BLOCK + row]),
            T.reinterpret("uint16", T.cast(output_hi, IO_DTYPE)),
        )


@T.inline
def _t_inverse_8_to_16(storage, block16, lane):
    a: T.uint32[2]
    b: T.uint32[1]
    acc: T.float32[4]
    word: T.uint32[1]
    T.ptx.ldmatrix.sync.aligned.m8n8.x1.shared.b16(
        word[0], _t_matrix_ptr(storage, block16 + 8 + (lane & 7), block16 + 8)
    )
    a[0] = word[0]
    a[1] = word[0]
    T.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(
        b[0], _t_matrix_ptr(storage, block16 + 8 + (lane & 7), block16)
    )
    _t_mma_m16n8k8_f16_zero(acc, a, b)
    _t_sub_zero_pack_f16x2(acc[0], acc[1], a, 0)
    _t_sub_zero_pack_f16x2(acc[2], acc[3], a, 1)
    T.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(
        b[0], _t_matrix_ptr(storage, block16 + (lane & 7), block16)
    )
    _t_mma_m16n8k8_f16_zero(acc, a, b)
    word[0] = _t_pack_f16x2(acc[0], acc[1])
    T.ptx.stmatrix.sync.aligned.m8n8.x1.shared.b16(
        _t_matrix_ptr(storage, block16 + 8 + (lane & 7), block16), word[0]
    )


@T.inline
def _t_inverse_16_to_32(storage, block32, lane):
    a: T.uint32[4]
    b: T.uint32[4]
    acc: T.float32[8]
    packed: T.uint32[4]
    _t_ldmatrix_x4(storage, block32 + 16, block32 + 16, lane, False, a)
    _t_ldmatrix_x4(storage, block32 + 16, block32, lane, True, b)
    _t_mma_m16n8k16_f16_zero(acc, a, b, 0, 0)
    _t_mma_m16n8k16_f16_zero(acc, a, b, 4, 2)
    for pair in T.unroll(4):
        _t_sub_zero_pack_f16x2(acc[pair * 2], acc[pair * 2 + 1], a, pair)
    _t_ldmatrix_x4(storage, block32, block32, lane, True, b)
    _t_mma_m16n8k16_f16_zero(acc, a, b, 0, 0)
    _t_mma_m16n8k16_f16_zero(acc, a, b, 4, 2)
    for pair in T.unroll(4):
        packed[pair] = _t_pack_f16x2(acc[pair * 2], acc[pair * 2 + 1])
    _t_stmatrix_x4(storage, block32 + 16, block32, lane, packed)


@T.inline
def _t_inverse_32_to_64(storage, local_warp, lane):
    # CollectiveInverse splits the final 32-wide K reduction between warp
    # pairs.  Each partial result is rounded to FP16 before the x=1 warp adds
    # the x=0 contribution in FP16.
    x: T.int32 = local_warp >> 1
    y: T.int32 = local_warp & 1
    row_base: T.int32 = 32 + y * 16
    split: T.int32 = x * 16
    d0: T.uint32[4]
    d1: T.uint32[4]
    c0: T.uint32[4]
    c1: T.uint32[4]
    ainv0: T.uint32[4]
    ainv1: T.uint32[4]
    temp: T.float32[8]
    output: T.float32[16]
    temp_f16: T.uint32[4]
    output0_f16: T.uint32[4]
    output1_f16: T.uint32[4]
    reduced0: T.uint32[4]
    reduced1: T.uint32[4]

    _t_ldmatrix_x4(storage, row_base, 32, lane, False, d0)
    _t_ldmatrix_x4(storage, row_base, 48, lane, False, d1)
    _t_ldmatrix_x4(storage, 32, split, lane, True, c0)
    _t_ldmatrix_x4(storage, 48, split, lane, True, c1)
    _t_mma_m16n8k16_f16_zero(temp, d0, c0, 0, 0)
    _t_mma_m16n8k16_f16_zero(temp, d0, c0, 4, 2)
    _t_mma_m16n8k16_f16_acc(temp, d1, c1, 0, 0)
    _t_mma_m16n8k16_f16_acc(temp, d1, c1, 4, 2)
    for pair in T.unroll(4):
        _t_sub_zero_pack_f16x2(temp[pair * 2], temp[pair * 2 + 1], temp_f16, pair)

    _t_ldmatrix_x4(storage, split, 0, lane, True, ainv0)
    _t_ldmatrix_x4(storage, split, 16, lane, True, ainv1)
    _t_mma_m16n8k16_f16_zero(output, temp_f16, ainv0, 0, 0)
    _t_mma_m16n8k16_f16_zero(output, temp_f16, ainv0, 4, 2)
    _t_mma_m16n8k16_f16_zero(output, temp_f16, ainv1, 8, 0)
    _t_mma_m16n8k16_f16_zero(output, temp_f16, ainv1, 12, 2)
    for pair in T.unroll(4):
        output0_f16[pair] = _t_pack_f16x2(output[pair * 2], output[pair * 2 + 1])
        output1_f16[pair] = _t_pack_f16x2(output[8 + pair * 2], output[8 + pair * 2 + 1])

    T.cuda.cta_sync()
    if x == 0:
        _t_stmatrix_x4(storage, row_base, 0, lane, output0_f16)
        _t_stmatrix_x4(storage, row_base, 16, lane, output1_f16)
    T.cuda.cta_sync()
    if x == 1:
        _t_ldmatrix_x4(storage, row_base, 0, lane, False, reduced0)
        _t_ldmatrix_x4(storage, row_base, 16, lane, False, reduced1)
        for pair in T.unroll(4):
            sum_lo0: T.uint16
            sum_hi0: T.uint16
            sum_lo1: T.uint16
            sum_hi1: T.uint16
            T.ptx.add.f16(
                sum_lo0,
                T.cast(output0_f16[pair] & T.uint32(0xFFFF), "uint16"),
                T.cast(reduced0[pair] & T.uint32(0xFFFF), "uint16"),
            )
            T.ptx.add.f16(
                sum_hi0,
                T.cast(output0_f16[pair] >> 16, "uint16"),
                T.cast(reduced0[pair] >> 16, "uint16"),
            )
            T.ptx.add.f16(
                sum_lo1,
                T.cast(output1_f16[pair] & T.uint32(0xFFFF), "uint16"),
                T.cast(reduced1[pair] & T.uint32(0xFFFF), "uint16"),
            )
            T.ptx.add.f16(
                sum_hi1,
                T.cast(output1_f16[pair] >> 16, "uint16"),
                T.cast(reduced1[pair] >> 16, "uint16"),
            )
            output0_f16[pair] = T.cast(sum_lo0, "uint32") | (T.cast(sum_hi0, "uint32") << 16)
            output1_f16[pair] = T.cast(sum_lo1, "uint32") | (T.cast(sum_hi1, "uint32") << 16)
        _t_stmatrix_x4(storage, row_base, 0, lane, output0_f16)
        _t_stmatrix_x4(storage, row_base, 16, lane, output1_f16)


def _mn_matrix_index(row, col):
    """K_SW128 offset for a 128x64 IO matrix split into two row tiles."""
    local_row: T.int32 = row & 63
    byte_offset: T.int32 = (local_row * T_BLOCK + col) * 2
    swizzled: T.int32 = T.bitwise_xor(
        byte_offset, T.bitwise_and(T.shift_right(byte_offset, T.int32(3)), T.int32(112))
    )
    return (row >> 6) * (T_BLOCK * T_BLOCK) + swizzled // 2


def _mn_matrix_ptr(storage, row, col):
    return storage.ptr_to([_mn_matrix_index(row, col)])


@T.inline
def _mn_ldmatrix_x4_a(storage, base_row, base_col, lane, dst):
    lane_matrix: T.int32 = lane >> 3
    row: T.int32 = base_row + (lane & 7) + (lane_matrix & 1) * 8
    col: T.int32 = base_col + (lane_matrix >> 1) * 8
    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
        dst[0], dst[1], dst[2], dst[3], _mn_matrix_ptr(storage, row, col)
    )


@T.inline
def _mn_ldmatrix_x4_b(storage, base_row, base_col, lane, dst):
    row: T.int32 = base_row + (lane & 7) + ((lane >> 4) & 1) * 8
    col: T.int32 = base_col + ((lane >> 3) & 1) * 8
    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
        dst[0], dst[1], dst[2], dst[3], _mn_matrix_ptr(storage, row, col)
    )


def _mn_opt_shared_addr(smem_base_addr, byte_offset):
    return smem_base_addr + T.cast(byte_offset, "uint32")


def _mn_opt_stage(count, stages):
    return T.cast(T.cast(count, "uint32") % T.uint32(stages), "int32")


def _mn_opt_phase(count, stages, initial_phase):
    turns: T.int32 = T.cast(
        T.bitwise_and(T.cast(count, "uint32") // T.uint32(stages), T.uint32(1)), "int32"
    )
    return T.bitwise_xor(turns, T.int32(initial_phase))


def _mn_opt_full_addr(smem_base_addr, full_off, count, stages):
    return _mn_opt_shared_addr(smem_base_addr, full_off + _mn_opt_stage(count, stages) * 8)


def _mn_opt_empty_addr(smem_base_addr, empty_off, count, stages):
    return _mn_opt_shared_addr(smem_base_addr, empty_off + _mn_opt_stage(count, stages) * 8)


def _mn_opt_producer_acquire(smem_base_addr, empty_off, count, stages):
    return T.cuda.mbarrier_wait(
        _mn_opt_empty_addr(smem_base_addr, empty_off, count, stages),
        _mn_opt_phase(count, stages, 1),
    )


def _mn_opt_consumer_wait(smem_base_addr, full_off, count, stages):
    return T.cuda.mbarrier_wait(
        _mn_opt_full_addr(smem_base_addr, full_off, count, stages), _mn_opt_phase(count, stages, 0)
    )


def _mn_opt_commit(smem_base_addr, full_off, count, stages):
    return T.ptx.mbarrier.arrive.shared.b64(
        _mn_opt_full_addr(smem_base_addr, full_off, count, stages), T.uint32(1)
    )


def _mn_opt_release(smem_base_addr, empty_off, count, stages):
    return T.ptx.mbarrier.arrive.shared.b64(
        _mn_opt_empty_addr(smem_base_addr, empty_off, count, stages), T.uint32(1)
    )


@T.inline
def _mn_opt_init_pipeline(smem_raw, full_off, empty_off, stages, producers, consumers):
    for stage in range(stages):
        T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([full_off + stage * 8]), T.uint32(producers))
        T.ptx.mbarrier.init.shared.b64(
            smem_raw.ptr_to([empty_off + stage * 8]), T.uint32(consumers)
        )


def _mn_opt_init_all_pipelines(smem_raw):
    for _, stages, full_off, empty_off, producers, consumers in MN_OPT_PIPELINES:
        _mn_opt_init_pipeline(smem_raw, full_off, empty_off, stages, producers, consumers)


def _mn_opt_b128_swizzle(byte_offset):
    return T.bitwise_xor(
        byte_offset, T.bitwise_and(T.shift_right(byte_offset, T.int32(3)), T.int32(112))
    )


def _mn_opt_tile_ptr(smem_raw, base, stage_stride, stage, row, col):
    byte_offset: T.int32 = base + stage * stage_stride + row * 128 + col * 2
    return smem_raw.ptr_to([_mn_opt_b128_swizzle(byte_offset)])


def _mn_opt_pack_iox2(a, b, IO_DTYPE):
    packed = T.alloc_local((1,), "uint32")
    if IO_DTYPE == "float16":
        T.evaluate(T.ptx.cvt.rn.f16x2.f32(packed[0], b, a))
    else:
        T.evaluate(T.ptx.cvt.rn.bf16x2.f32(packed[0], b, a))
    return packed[0]


def _mn_opt_unpack_io_lo(word, IO_DTYPE):
    raw: T.uint16 = T.cast(T.bitwise_and(word, T.uint32(0xFFFF)), "uint16")
    if IO_DTYPE == "float16":
        return T.cast(T.reinterpret("float16", raw), "float32")
    return T.cast(T.reinterpret("bfloat16", raw), "float32")


def _mn_opt_unpack_io_hi(word, IO_DTYPE):
    raw: T.uint16 = T.cast(T.shift_right(word, T.uint32(16)), "uint16")
    if IO_DTYPE == "float16":
        return T.cast(T.reinterpret("float16", raw), "float32")
    return T.cast(T.reinterpret("bfloat16", raw), "float32")


def _mn_opt_tmem_row_bits(thread):
    return T.bitwise_and(thread << 16, T.int32(0x600000))


@T.inline
def _mn_opt_tmem_ld_matrix_sub(tmem_base, column, thread, sub, values, value_offset):
    addr: T.int32 = tmem_base + _mn_opt_tmem_row_bits(thread) + column + sub * 32
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
        *[values[value_offset + i] for i in range(32)], T.cast(addr, "uint32")
    )


@T.inline
def _mn_opt_tmem_st_matrix_sub(tmem_base, column, thread, sub, values, value_offset):
    addr: T.int32 = tmem_base + _mn_opt_tmem_row_bits(thread) + column + sub * 32
    T.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
        T.cast(addr, "uint32"), *[values[value_offset + i] for i in range(32)]
    )


@T.inline
def _mn_opt_tmem_st_matrix_io_sub(tmem_base, column, thread, sub, values):
    addr: T.int32 = tmem_base + _mn_opt_tmem_row_bits(thread) + column + sub * 16
    T.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
        T.cast(addr, "uint32"), *[values[sub * 16 + i] for i in range(16)]
    )


@T.inline
def _mn_opt_tmem_ld_128x64(tmem_base, column, thread, values):
    addr: T.int32 = tmem_base + _mn_opt_tmem_row_bits(thread) + column
    T.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
        *[values[i] for i in range(32)], T.cast(addr, "uint32")
    )
    T.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
        *[values[32 + i] for i in range(32)], T.cast(addr + T.int32(0x100000), "uint32")
    )


@T.inline
def _mn_opt_tmem_st_128x64_io_half(tmem_base, column, thread, half, values):
    addr: T.int32 = tmem_base + _mn_opt_tmem_row_bits(thread) + column
    T.ptx["tcgen05.st.sync.aligned.16x128b.x8.b32"](
        T.cast(addr + half * T.int32(0x100000), "uint32"),
        *[values[half * 16 + i] for i in range(16)],
    )


@T.inline
def _mn_opt_tmem_st_128x64_io(tmem_base, column, thread, values):
    _mn_opt_tmem_st_128x64_io_half(tmem_base, column, thread, 0, values)
    _mn_opt_tmem_st_128x64_io_half(tmem_base, column, thread, 1, values)


def _mn_opt_fragment_lane_byte(thread):
    a: T.int32 = T.bitwise_and(thread << 6, T.int32(448))
    b: T.int32 = T.bitwise_and(thread, T.int32(40))
    c: T.int32 = T.bitwise_or(b, a)
    d: T.int32 = T.bitwise_and(thread << 5, T.int32(512))
    e: T.int32 = T.bitwise_and(thread << 6, T.int32(4096))
    return T.bitwise_or(T.bitwise_or(d, e), T.bitwise_xor(a >> 3, c)) << 1


def _mn_opt_fragment_second_half_delta(thread):
    c: T.int32 = T.bitwise_or(
        T.bitwise_and(thread, T.int32(40)), T.bitwise_and(thread << 6, T.int32(448))
    )
    return T.if_then_else(T.bitwise_and(c, T.int32(128)) == 0, T.int32(32), T.int32(-32))


@T.inline
def _mn_opt_load_128x64_fragment(smem_raw, base, stage, thread, values):
    lane_byte: T.int32 = _mn_opt_fragment_lane_byte(thread)
    stage_byte: T.int32 = stage * 16384
    for band in range(4):
        T.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            *[values[band * 4 + i] for i in range(4)],
            smem_raw.ptr_to([base + stage_byte + lane_byte + band * 2048]),
        )
    lane_byte = lane_byte + _mn_opt_fragment_second_half_delta(thread)
    for band in range(4):
        T.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            *[values[16 + band * 4 + i] for i in range(4)],
            smem_raw.ptr_to([base + stage_byte + lane_byte + band * 2048]),
        )


@T.inline
def _mn_opt_store_128x64_fragment(smem_raw, base, stage, thread, values):
    lane_byte: T.int32 = _mn_opt_fragment_lane_byte(thread)
    stage_byte: T.int32 = stage * 16384
    for band in range(4):
        T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            smem_raw.ptr_to([base + stage_byte + lane_byte + band * 2048]),
            *[values[band * 4 + i] for i in range(4)],
        )
    lane_byte = lane_byte + _mn_opt_fragment_second_half_delta(thread)
    for band in range(4):
        T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            smem_raw.ptr_to([base + stage_byte + lane_byte + band * 2048]),
            *[values[16 + band * 4 + i] for i in range(4)],
        )


@T.inline
def _mn_opt_initialize_matrix(tmem_base, column, thread, identity):
    values: T.float32[32]
    for sub in range(4):
        for i in T.unroll(32):
            col: T.int32 = sub * 32 + i
            values[i] = T.if_then_else(identity and thread == col, T.float32(1.0), T.float32(0.0))
        _mn_opt_tmem_st_matrix_sub(tmem_base, column, thread, sub, values, 0)
    T.ptx.tcgen05.wait__st.sync.aligned()


@T.inline
def _mn_opt_scale_matrix(tmem_base, column, thread, scale):
    values: T.float32[32]
    for sub in range(4):
        _mn_opt_tmem_ld_matrix_sub(tmem_base, column, thread, sub, values, 0)
        for i in T.unroll(32):
            values[i] = values[i] * scale
        _mn_opt_tmem_st_matrix_sub(tmem_base, column, thread, sub, values, 0)
    T.ptx.tcgen05.wait__st.sync.aligned()


@T.inline
def _mn_opt_matrix_to_io_input(tmem_base, src_column, dst_column, thread, IO_DTYPE):
    values: T.float32[32]
    packed: T.uint32[64]
    for sub in range(4):
        _mn_opt_tmem_ld_matrix_sub(tmem_base, src_column, thread, sub, values, 0)
        for pair in T.unroll(16):
            packed[sub * 16 + pair] = _mn_opt_pack_iox2(
                values[pair * 2], values[pair * 2 + 1], IO_DTYPE
            )
        _mn_opt_tmem_st_matrix_io_sub(tmem_base, dst_column, thread, sub, packed)
    T.ptx.tcgen05.wait__st.sync.aligned()


@T.inline
def _mn_opt_scratch_to_io_input(tmem_base, dst_column, thread, IO_DTYPE):
    values: T.float32[64]
    packed: T.uint32[32]
    _mn_opt_tmem_ld_128x64(tmem_base, MN_OPT_TMEM_SCRATCH_COL, thread, values)
    for pair in T.unroll(32):
        packed[pair] = _mn_opt_pack_iox2(values[pair * 2], values[pair * 2 + 1], IO_DTYPE)
    _mn_opt_tmem_st_128x64_io(tmem_base, dst_column, thread, packed)
    T.ptx.tcgen05.wait__st.sync.aligned()


@T.inline
def _mn_opt_materialize_x(tmem_base, smem_raw, stage, thread, IO_DTYPE):
    values: T.float32[64]
    packed: T.uint32[32]
    _mn_opt_tmem_ld_128x64(tmem_base, MN_OPT_TMEM_XY_COL, thread, values)
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for pair in T.unroll(32):
        packed[pair] = _mn_opt_pack_iox2(values[pair * 2], values[pair * 2 + 1], IO_DTYPE)
    _mn_opt_store_128x64_fragment(smem_raw, MN_OPT_X_OFF, stage, thread, packed)
    T.ptx.fence.proxy.async_.shared__cta()


@T.inline
def _mn_opt_store_matrix_global(tmem_base, column, output, base, thread):
    values: T.uint32[32]
    thread_base: T.int64 = base + T.cast(thread, "int64") * D_HEAD
    for sub in range(4):
        _mn_opt_tmem_ld_matrix_sub(tmem_base, column, thread, sub, values, 0)
        for vector in range(8):
            T.ptx["st.global.L1::no_allocate.v4.b32"](
                output.ptr_to([thread_base + sub * 32 + vector * 4]),
                values[vector * 4],
                values[vector * 4 + 1],
                values[vector * 4 + 2],
                values[vector * 4 + 3],
            )


def _mn_opt_smem_desc_k(smem_addr):
    desc_lo = T.cast(
        T.bitwise_and(T.shift_right(smem_addr, T.uint32(4)), T.uint32(0x3FFF)), "uint64"
    )
    return T.bitwise_or(T.uint64(0x4000404000010000), desc_lo)


def _mn_opt_smem_desc_mn(smem_addr):
    desc_lo = T.cast(
        T.bitwise_and(T.shift_right(smem_addr, T.uint32(4)), T.uint32(0x3FFF)), "uint64"
    )
    return T.bitwise_or(T.uint64(0x4000404002000000), desc_lo)


def _mn_opt_mma_descriptor(base, IO_DTYPE):
    return T.uint32(base + (0x480 if IO_DTYPE == "bfloat16" else 0))


_MN_OPT_MMA_CHAIN = "tcgen05.mma.cta_group::1.kind::f16"
_MN_OPT_ZERO_MASKS = [T.uint32(0)] * 4


def _mn_opt_mma_commit(barrier):
    return T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
        barrier, pred=T.cuda.elect_sync()
    )


# CP state fixup: the UTC64/UTC128 source specializations share one pipeline
# skeleton but use different TMEM copies, register ownership, and M-ring depth.
_FIXUP_TMA_G2S_4D = (
    "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
)
_FIXUP_MMA_TF32 = "tcgen05.mma.cta_group::1.kind::tf32"
_FIXUP_ZERO_MASKS = [T.uint32(0)] * 4
_FIXUP_TMEM_ACC_COL = 0
_FIXUP_TMEM_OPERAND_COL = 128
_FIXUP_TMEM_COLUMNS = 256
_FIXUP_TMEM_ALLOC_BARRIER = 1


def _fixup_tmem_row_bits(thread):
    return T.bitwise_and(thread << 16, T.int32(0x600000))


@T.inline
def _fixup_tmem_ld_sub(tmem_base, column, thread, sub, words, value_offset, ROWS):
    addr: T.int32 = tmem_base + _fixup_tmem_row_bits(thread) + column + sub * 32
    if ROWS == 128:
        T.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
            *[words[value_offset + i] for i in range(32)], T.cast(addr, "uint32")
        )
    else:
        T.ptx["tcgen05.ld.sync.aligned.16x32bx2.x16.b32"](
            *[words[value_offset + i] for i in range(16)], T.cast(addr, "uint32"), 16
        )


@T.inline
def _fixup_tmem_st_sub(tmem_base, column, thread, sub, words, value_offset, ROWS):
    addr: T.int32 = tmem_base + _fixup_tmem_row_bits(thread) + column + sub * 32
    if ROWS == 128:
        T.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
            T.cast(addr, "uint32"), *[words[value_offset + i] for i in range(32)]
        )
    else:
        T.ptx["tcgen05.st.sync.aligned.16x32bx2.x16.b32"](
            T.cast(addr, "uint32"), 16, *[words[value_offset + i] for i in range(16)]
        )


@T.inline
def _fixup_tmem_ld(tmem_base, column, thread, words, ROWS):
    values_per_sub = 32 if ROWS == 128 else 16
    for sub in range(4):
        _fixup_tmem_ld_sub(tmem_base, column, thread, sub, words, sub * values_per_sub, ROWS)


@T.inline
def _fixup_tmem_st(tmem_base, column, thread, words, ROWS):
    values_per_sub = 32 if ROWS == 128 else 16
    for sub in range(4):
        _fixup_tmem_st_sub(tmem_base, column, thread, sub, words, sub * values_per_sub, ROWS)


@T.inline
def _fixup_load_n_to_tmem(tmem_base, smem_raw, thread, ROWS, N_OFF):
    words: T.uint32[128]
    if ROWS == 128:
        for sub in range(4):
            lane_byte: T.int32 = N_OFF + (thread >> 5) * 4096 + (thread & 31) * 128 + sub * 16384
            for vector in range(8):
                T.ptx.ld.shared.v4.b32(
                    words[sub * 32 + vector * 4],
                    words[sub * 32 + vector * 4 + 1],
                    words[sub * 32 + vector * 4 + 2],
                    words[sub * 32 + vector * 4 + 3],
                    smem_raw.ptr_to([_mn_opt_b128_swizzle(lane_byte + vector * 16)]),
                )
    else:
        for sub in range(4):
            lane_byte: T.int32 = (
                N_OFF + (thread >> 5) * 2048 + (thread & 15) * 128 + (thread & 16) * 4 + sub * 8192
            )
            for vector in range(4):
                T.ptx.ld.shared.v4.b32(
                    words[sub * 16 + vector * 4],
                    words[sub * 16 + vector * 4 + 1],
                    words[sub * 16 + vector * 4 + 2],
                    words[sub * 16 + vector * 4 + 3],
                    smem_raw.ptr_to([_mn_opt_b128_swizzle(lane_byte + vector * 16)]),
                )
    _fixup_tmem_st(tmem_base, _FIXUP_TMEM_ACC_COL, thread, words, ROWS)


@T.inline
def _fixup_load_initial_to_tmem(tmem_base, initial_state, base, thread, ROWS, STATE_DTYPE):
    values = T.alloc_local((128,), "float32")
    words = values.view("uint32")
    if ROWS == 128:
        for sub in range(4):
            for i in T.unroll(32):
                _load_global_as_f32(
                    values,
                    sub * 32 + i,
                    initial_state,
                    base + T.cast(thread, "int64") * D_HEAD + sub * 32 + i,
                    STATE_DTYPE,
                )
    else:
        local_row: T.int32 = (thread >> 5) * 16 + (thread & 15)
        col_base: T.int32 = thread & 16
        for sub in range(4):
            for i in T.unroll(16):
                _load_global_as_f32(
                    values,
                    sub * 16 + i,
                    initial_state,
                    base + T.cast(local_row, "int64") * D_HEAD + col_base + sub * 32 + i,
                    STATE_DTYPE,
                )
    _fixup_tmem_st(tmem_base, _FIXUP_TMEM_ACC_COL, thread, words, ROWS)


@T.inline
def _fixup_store_f32(words, output, base, thread, ROWS):
    if ROWS == 128:
        for sub in range(4):
            thread_base: T.int64 = base + T.cast(thread, "int64") * D_HEAD + sub * 32
            for vector in range(8):
                T.ptx.st.global_.v4.b32(
                    output.ptr_to([thread_base + vector * 4]),
                    words[sub * 32 + vector * 4],
                    words[sub * 32 + vector * 4 + 1],
                    words[sub * 32 + vector * 4 + 2],
                    words[sub * 32 + vector * 4 + 3],
                )
    else:
        local_row: T.int32 = (thread >> 5) * 16 + (thread & 15)
        col_base: T.int32 = thread & 16
        for sub in range(4):
            thread_base: T.int64 = base + T.cast(local_row, "int64") * D_HEAD + col_base + sub * 32
            for vector in range(4):
                T.ptx.st.global_.v4.b32(
                    output.ptr_to([thread_base + vector * 4]),
                    words[sub * 16 + vector * 4],
                    words[sub * 16 + vector * 4 + 1],
                    words[sub * 16 + vector * 4 + 2],
                    words[sub * 16 + vector * 4 + 3],
                )


@T.inline
def _fixup_store_state(values, output, base, thread, ROWS, STATE_DTYPE):
    if STATE_DTYPE == "float32":
        _fixup_store_f32(values.view("uint32"), output, base, thread, ROWS)
    else:
        packed: T.uint32[64]
        if ROWS == 128:
            for sub in range(4):
                for pair in T.unroll(16):
                    packed[sub * 16 + pair] = _mn_opt_pack_iox2(
                        values[sub * 32 + pair * 2], values[sub * 32 + pair * 2 + 1], STATE_DTYPE
                    )
                thread_base: T.int64 = base + T.cast(thread, "int64") * D_HEAD + sub * 32
                for vector in range(4):
                    T.ptx["st.global.L1::no_allocate.v4.b32"](
                        output.ptr_to([thread_base + vector * 8]),
                        packed[sub * 16 + vector * 4],
                        packed[sub * 16 + vector * 4 + 1],
                        packed[sub * 16 + vector * 4 + 2],
                        packed[sub * 16 + vector * 4 + 3],
                    )
        else:
            local_row: T.int32 = (thread >> 5) * 16 + (thread & 15)
            col_base: T.int32 = thread & 16
            for sub in range(4):
                for pair in T.unroll(8):
                    packed[sub * 8 + pair] = _mn_opt_pack_iox2(
                        values[sub * 16 + pair * 2], values[sub * 16 + pair * 2 + 1], STATE_DTYPE
                    )
                thread_base: T.int64 = (
                    base + T.cast(local_row, "int64") * D_HEAD + col_base + sub * 32
                )
                for vector in range(2):
                    T.ptx["st.global.L1::no_allocate.v4.b32"](
                        output.ptr_to([thread_base + vector * 8]),
                        packed[sub * 8 + vector * 4],
                        packed[sub * 8 + vector * 4 + 1],
                        packed[sub * 8 + vector * 4 + 2],
                        packed[sub * 8 + vector * 4 + 3],
                    )


@T.inline
def _fixup_acc_to_tf32(tmem_base, thread, ROWS):
    values = T.alloc_local((128,), "float32")
    words = values.view("uint32")
    tf32_words: T.uint32[128]
    _fixup_tmem_ld(tmem_base, _FIXUP_TMEM_ACC_COL, thread, words, ROWS)
    T.ptx.tcgen05.wait__ld.sync.aligned()
    if ROWS == 128:
        for i in T.unroll(128):
            T.ptx.cvt.rna.tf32.f32(tf32_words[i], values[i])
    else:
        for i in T.unroll(64):
            T.ptx.cvt.rna.tf32.f32(tf32_words[i], values[i])
    _fixup_tmem_st(tmem_base, _FIXUP_TMEM_OPERAND_COL, thread, tf32_words, ROWS)
    T.ptx.tcgen05.wait__st.sync.aligned()


def _fixup_smem_desc_m(smem_addr, M_OFF):
    desc_lo = T.cast(
        T.bitwise_and(T.shift_right(smem_addr + T.uint32(M_OFF), T.uint32(4)), T.uint32(0x3FFF)),
        "uint64",
    )
    return T.bitwise_or(T.uint64(0x2000402004000000), desc_lo)


@T.inline
def _fixup_mma(tmem_base, m_desc, m_stage, ROWS, done_full, ready_empty, m_empty):
    instr_desc: T.uint32 = T.uint32(0x08110910 if ROWS == 128 else 0x04110910)
    for kphase in range(16):
        for n_half in range(2):
            T.evaluate(
                T.ptx[_FIXUP_MMA_TF32](
                    T.cast(tmem_base + n_half * 64, "uint32"),
                    T.cast(tmem_base + _FIXUP_TMEM_OPERAND_COL + kphase * 8, "uint32"),
                    m_desc + T.uint64(m_stage * 4096 + kphase * 64 + n_half * 2048),
                    instr_desc,
                    *_FIXUP_ZERO_MASKS,
                    True,
                    pred=T.cuda.elect_sync(),
                )
            )
    _mn_opt_mma_commit(done_full)
    _mn_opt_mma_commit(ready_empty)
    _mn_opt_mma_commit(m_empty)


@T.inline
def _fixup_tma_matrix(smem_addr, smem_off, descriptor, barrier, row_coord, head, chunk, ROWS, is_m):
    part_bytes = 16384 if is_m or ROWS == 128 else 8192
    for part in range(4):
        T.ptx[_FIXUP_TMA_G2S_4D](
            _mn_opt_shared_addr(smem_addr, smem_off + part * part_bytes),
            descriptor,
            T.int32(part * 32),
            T.cast(row_coord, "int32"),
            head,
            chunk,
            barrier,
            T.uint64(0),
        )


@T.inline
def _mn_opt_mma_ss_128x64_k64(tmem_d, a_desc_base, b_desc_base, full_barrier, IO_DTYPE):
    descriptor: T.uint32 = _mn_opt_mma_descriptor(0x08108010, IO_DTYPE)
    for kphase in range(4):
        T.evaluate(
            T.ptx[_MN_OPT_MMA_CHAIN](
                T.cast(tmem_d, "uint32"),
                a_desc_base + T.uint64(kphase * 128),
                b_desc_base + T.uint64(kphase * 2),
                descriptor,
                *_MN_OPT_ZERO_MASKS,
                T.ptx.pred(0 if kphase == 0 else 1),
                pred=T.cuda.elect_sync(),
            )
        )
    _mn_opt_mma_commit(full_barrier)


@T.inline
def _mn_opt_mma_ts_128x64_k128(tmem_d, tmem_a, b_desc_base, full_barrier, IO_DTYPE):
    descriptor: T.uint32 = _mn_opt_mma_descriptor(0x08100010, IO_DTYPE)
    for kphase in range(8):
        phase_off: T.uint64 = T.uint64((kphase % 4) * 2 + (kphase // 4) * 512)
        T.evaluate(
            T.ptx[_MN_OPT_MMA_CHAIN](
                T.cast(tmem_d, "uint32"),
                T.cast(tmem_a + kphase * 8, "uint32"),
                b_desc_base + phase_off,
                descriptor,
                *_MN_OPT_ZERO_MASKS,
                T.ptx.pred(0 if kphase == 0 else 1),
                pred=T.cuda.elect_sync(),
            )
        )
    _mn_opt_mma_commit(full_barrier)


@T.inline
def _mn_opt_mma_ss_128x128_k64(tmem_d, a_desc_base, b_desc_base, full_barrier, IO_DTYPE):
    descriptor: T.uint32 = _mn_opt_mma_descriptor(0x08218010, IO_DTYPE)
    for kphase in range(4):
        T.evaluate(
            T.ptx[_MN_OPT_MMA_CHAIN](
                T.cast(tmem_d, "uint32"),
                a_desc_base + T.uint64(kphase * 128),
                b_desc_base + T.uint64(kphase * 128),
                descriptor,
                *_MN_OPT_ZERO_MASKS,
                True,
                pred=T.cuda.elect_sync(),
            )
        )
    _mn_opt_mma_commit(full_barrier)


@T.inline
def _mn_opt_mma_ts_128x128_k64(tmem_d, tmem_a, b_desc_base, full_barrier, IO_DTYPE):
    descriptor: T.uint32 = _mn_opt_mma_descriptor(0x08210010, IO_DTYPE)
    for kphase in range(4):
        T.evaluate(
            T.ptx[_MN_OPT_MMA_CHAIN](
                T.cast(tmem_d, "uint32"),
                T.cast(tmem_a + kphase * 8, "uint32"),
                b_desc_base + T.uint64(kphase * 128),
                descriptor,
                *_MN_OPT_ZERO_MASKS,
                True,
                pred=T.cuda.elect_sync(),
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


@T.inline
def _mn_opt_tma_kv(smem_base_addr, smem_off, descriptor, barrier, token, head):
    for d_coord in range(0, D_HEAD, 64):
        T.ptx[_MN_OPT_TMA_G2S_3D](
            _mn_opt_shared_addr(smem_base_addr, smem_off + d_coord * 128),
            descriptor,
            T.int32(d_coord),
            T.cast(token, "int32"),
            head,
            barrier,
            T.uint64(0),
        )


@T.inline
def _mn_opt_tma_t(smem_base_addr, smem_off, descriptor, barrier, t_block, state_head):
    T.ptx[_MN_OPT_TMA_G2S_4D](
        _mn_opt_shared_addr(smem_base_addr, smem_off),
        descriptor,
        T.int32(0),
        T.int32(0),
        state_head,
        t_block,
        barrier,
        T.uint64(0),
    )


@T.inline
def _mn_opt_process_y(tmem_base, smem_raw, v_stage, alpha_stage, block_coeff, thread, IO_DTYPE):
    y_values: T.float32[64]
    v_words: T.uint32[32]
    y_words: T.uint32[32]
    _mn_opt_tmem_ld_128x64(tmem_base, MN_OPT_TMEM_XY_COL, thread, y_values)
    _mn_opt_load_128x64_fragment(smem_raw, MN_OPT_V_OFF, v_stage, thread, v_words)
    factor_col_base: T.int32 = T.bitwise_and(thread << 1, T.int32(6))
    neg_factors: T.float32[16]
    for factor_group in T.unroll(8):
        factor_col: T.int32 = factor_col_base + factor_group * 8
        T.ptx.ld.shared.v2.f32(
            neg_factors[factor_group * 2],
            neg_factors[factor_group * 2 + 1],
            smem_raw.ptr_to(
                [MN_OPT_ALPHA_OFF + (alpha_stage * T_BLOCK * 3 + T_BLOCK * 2 + factor_col) * 4]
            ),
        )
    for row_half in T.unroll(2):
        for factor_group in T.unroll(8):
            for factor_repeat in T.unroll(2):
                pair: T.int32 = row_half * 16 + factor_group * 2 + factor_repeat
                v0: T.float32 = _mn_opt_unpack_io_lo(v_words[pair], IO_DTYPE)
                v1: T.float32 = _mn_opt_unpack_io_hi(v_words[pair], IO_DTYPE)
                updated: T.uint64
                T.ptx.mul.rn.f32x2(
                    updated,
                    T.cuda.make_float2(v0, v1),
                    T.cuda.make_float2(
                        neg_factors[factor_group * 2], neg_factors[factor_group * 2 + 1]
                    ),
                )
                y_values[pair * 2] = block_coeff * y_values[pair * 2] + T.cuda.float2_x(updated)
                y_values[pair * 2 + 1] = block_coeff * y_values[pair * 2 + 1] + T.cuda.float2_y(
                    updated
                )
    for pair in T.unroll(32):
        y_words[pair] = _mn_opt_pack_iox2(y_values[pair * 2], y_values[pair * 2 + 1], IO_DTYPE)
    _mn_opt_tmem_st_128x64_io(tmem_base, MN_OPT_TMEM_N_INPUT_COL, thread, y_words)
    T.ptx.tcgen05.wait__st.sync.aligned()


@T.inline
def _prefill_opt_init_pipeline(smem_raw, full_off, empty_off, stages, producers, consumers):
    for stage in range(stages):
        T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([full_off + stage * 8]), T.uint32(producers))
        T.ptx.mbarrier.init.shared.b64(
            smem_raw.ptr_to([empty_off + stage * 8]), T.uint32(consumers)
        )


def _prefill_opt_init_all_pipelines(smem_raw):
    for _, stages, full_off, empty_off, producers, consumers in PREFILL_OPT_PIPELINES:
        _prefill_opt_init_pipeline(smem_raw, full_off, empty_off, stages, producers, consumers)


@T.inline
def _prefill_opt_cg0_tmem_ld(tmem_base, stage, thread, values):
    row_bits: T.int32 = T.bitwise_and(thread << 16, T.int32(0x600000))
    addr: T.int32 = tmem_base + PREFILL_OPT_TMEM_CG0_ACC_COL + stage * 64 + row_bits
    T.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
        *[values[i] for i in range(32)], T.cast(addr, "uint32")
    )


@T.inline
def _prefill_opt_load_t_fragment(smem_raw, stage, thread, values, IO_DTYPE):
    lane_byte: T.int32 = (
        T.bitwise_or(
            T.bitwise_or(
                T.bitwise_and(thread << 6, T.int32(960)), T.bitwise_and(thread >> 1, T.int32(8))
            ),
            T.bitwise_and(thread << 5, T.int32(3072)),
        )
        << 1
    )
    packed: T.uint32[16]
    for band in range(4):
        T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
            *[packed[band * 4 + i] for i in range(4)],
            smem_raw.ptr_to(
                [PREFILL_OPT_T_OFF + stage * 8192 + _mn_opt_b128_swizzle(lane_byte + band * 32)]
            ),
        )
    for pair in T.unroll(16):
        values[pair * 2] = _mn_opt_unpack_io_lo(packed[pair], IO_DTYPE)
        values[pair * 2 + 1] = _mn_opt_unpack_io_hi(packed[pair], IO_DTYPE)


@T.inline
def _prefill_opt_store_ainv_fragment(smem_raw, stage, thread, values, IO_DTYPE):
    a: T.int32 = T.bitwise_and(thread << 6, T.int32(448))
    c: T.int32 = T.bitwise_or(
        T.bitwise_or(a, T.bitwise_and(thread >> 1, T.int32(48))), T.bitwise_and(thread, T.int32(8))
    )
    lane_byte: T.int32 = T.bitwise_or(
        T.bitwise_and(thread << 6, T.int32(1024)), T.bitwise_xor(a >> 3, c) << 1
    )
    packed: T.uint32[16]
    for pair in T.unroll(16):
        packed[pair] = _mn_opt_pack_iox2(values[pair * 2], values[pair * 2 + 1], IO_DTYPE)
    for band in range(4):
        T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            smem_raw.ptr_to([PREFILL_OPT_AINV_OFF + stage * 8192 + lane_byte + band * 2048]),
            *[packed[band * 4 + i] for i in range(4)],
        )


@T.inline
def _prefill_opt_store_qk_fragment(smem_raw, stage, thread, values, IO_DTYPE):
    a: T.int32 = T.bitwise_and(thread << 6, T.int32(448))
    x: T.int32 = T.bitwise_or(a, T.bitwise_and(thread >> 1, T.int32(8)))
    g: T.int32 = T.bitwise_xor(T.bitwise_and(x >> 3, T.int32(56)), x)
    hi: T.int32 = T.bitwise_or(
        T.bitwise_and(thread << 6, T.int32(512)), T.bitwise_and(thread << 5, T.int32(3072))
    )
    lane_byte: T.int32 = T.bitwise_or(hi, g) << 1
    delta1: T.int32 = T.if_then_else(T.bitwise_and(x, T.int32(128)) == 0, T.int32(16), T.int32(-16))
    delta2: T.int32 = T.if_then_else(T.bitwise_and(x, T.int32(256)) == 0, T.int32(32), T.int32(-32))
    packed: T.uint32[16]
    for pair in T.unroll(16):
        packed[pair] = _mn_opt_pack_iox2(values[pair * 2], values[pair * 2 + 1], IO_DTYPE)
    T.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
        smem_raw.ptr_to([PREFILL_OPT_QK_OFF + stage * 8192 + lane_byte]),
        *[packed[i] for i in range(4)],
    )
    T.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
        smem_raw.ptr_to([PREFILL_OPT_QK_OFF + stage * 8192 + lane_byte + 2 * delta1]),
        *[packed[4 + i] for i in range(4)],
    )
    T.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
        smem_raw.ptr_to([PREFILL_OPT_QK_OFF + stage * 8192 + lane_byte + 2 * delta2]),
        *[packed[8 + i] for i in range(4)],
    )
    T.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
        smem_raw.ptr_to([PREFILL_OPT_QK_OFF + stage * 8192 + lane_byte + 2 * (delta1 + delta2)]),
        *[packed[12 + i] for i in range(4)],
    )


@T.inline
def _prefill_opt_transform_t(
    smem_raw,
    smem_addr,
    s_cumsumlog,
    t_stage,
    ainv_stage,
    gate_stage,
    thread,
    is_final_block,
    valid_tokens,
    IO_DTYPE,
):
    values: T.float32[32]
    _prefill_opt_load_t_fragment(smem_raw, t_stage, thread, values, IO_DTYPE)
    row_base: T.int32 = T.bitwise_or(
        T.bitwise_and(thread >> 2, T.int32(7)), T.bitwise_and(thread >> 1, T.int32(48))
    )
    col_base: T.int32 = T.bitwise_and(thread << 1, T.int32(6))
    for i in T.unroll(32):
        t_coord: T.int32 = row_base + T.bitwise_and(i >> 1, T.int32(1)) * 8
        s_coord: T.int32 = (
            col_base
            + T.bitwise_and(i, T.int32(1))
            + T.bitwise_and(i >> 2, T.int32(1)) * 8
            + (i >> 3) * 16
        )
        valid: T.bool = s_coord >= t_coord
        if is_final_block:
            valid = valid and s_coord < valid_tokens and t_coord < valid_tokens
        gamma: T.float32 = _prefill_predicated_gamma(
            _pf_shared_addr(
                smem_addr, PREFILL_OPT_CUMSUMLOG_OFF + (gate_stage * T_BLOCK + s_coord) * 4
            ),
            _pf_shared_addr(
                smem_addr, PREFILL_OPT_CUMSUMLOG_OFF + (gate_stage * T_BLOCK + t_coord) * 4
            ),
            valid,
        )
        values[i] = -gamma * values[i]
    _prefill_opt_store_ainv_fragment(smem_raw, ainv_stage, thread, values, IO_DTYPE)


@T.inline
def _prefill_opt_load_v_fragment(smem_raw, stage, thread, values):
    lane_byte: T.int32 = _pf_cg1_smem_lane_byte(thread)
    stage_byte: T.int32 = stage * 16384
    for band in range(4):
        T.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            *[values[band * 4 + i] for i in range(4)],
            smem_raw.ptr_to([PREFILL_OPT_V_OFF + stage_byte + lane_byte + band * 2048]),
        )
    lane_byte = lane_byte + _pf_cg1_smem_second_half_delta(thread)
    for band in range(4):
        T.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            *[values[16 + band * 4 + i] for i in range(4)],
            smem_raw.ptr_to([PREFILL_OPT_V_OFF + stage_byte + lane_byte + band * 2048]),
        )


@T.inline
def _prefill_opt_store_o_fragment(smem_raw, stage, thread, values):
    lane_byte: T.int32 = _pf_cg1_smem_lane_byte(thread)
    stage_byte: T.int32 = stage * 16384
    for band in range(4):
        T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            smem_raw.ptr_to([PREFILL_OPT_O_OFF + stage_byte + lane_byte + band * 2048]),
            *[values[band * 4 + i] for i in range(4)],
        )
    lane_byte = lane_byte + _pf_cg1_smem_second_half_delta(thread)
    for band in range(4):
        T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            smem_raw.ptr_to([PREFILL_OPT_O_OFF + stage_byte + lane_byte + band * 2048]),
            *[values[16 + band * 4 + i] for i in range(4)],
        )


def _prefill_opt_sub_iox2(lhs, rhs, IO_DTYPE):
    result = T.alloc_local((1,), "uint32")
    if IO_DTYPE == "float16":
        T.evaluate(T.ptx.sub.f16x2(result[0], lhs, rhs))
    else:
        T.evaluate(T.ptx["sub.bf16x2"](result[0], lhs, rhs))
    return result[0]


_PREFILL_OPT_MMA_CHAIN = "tcgen05.mma.cta_group::1.kind::f16"
_PREFILL_OPT_ZERO_MASKS = [T.uint32(0)] * 4


def _prefill_opt_mma_descriptor(base, IO_DTYPE):
    return T.uint32(base + (0x480 if IO_DTYPE == "bfloat16" else 0))


def _prefill_opt_mma_commit(barrier):
    return T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
        barrier, pred=T.cuda.elect_sync()
    )


@T.inline
def _prefill_opt_mma_ss_64x64_k128(tmem_d, a_desc_base, b_desc_base, full_barrier, IO_DTYPE):
    descriptor: T.uint32 = _prefill_opt_mma_descriptor(0x04100010, IO_DTYPE)
    for kphase in range(8):
        phase_off: T.uint64 = T.uint64((kphase % 4) * 2 + (kphase // 4) * 512)
        T.evaluate(
            T.ptx[_PREFILL_OPT_MMA_CHAIN](
                T.cast(tmem_d, "uint32"),
                a_desc_base + phase_off,
                b_desc_base + phase_off,
                descriptor,
                *_PREFILL_OPT_ZERO_MASKS,
                T.ptx.pred(0 if kphase == 0 else 1),
                pred=T.cuda.elect_sync(),
            )
        )
    _prefill_opt_mma_commit(full_barrier)


@T.inline
def _prefill_opt_mma_ts_128x64_k128(tmem_d, tmem_a, b_desc_base, full_barrier, IO_DTYPE):
    descriptor: T.uint32 = _prefill_opt_mma_descriptor(0x08100010, IO_DTYPE)
    for kphase in range(8):
        phase_off: T.uint64 = T.uint64((kphase % 4) * 2 + (kphase // 4) * 512)
        T.evaluate(
            T.ptx[_PREFILL_OPT_MMA_CHAIN](
                T.cast(tmem_d, "uint32"),
                T.cast(tmem_a + kphase * 8, "uint32"),
                b_desc_base + phase_off,
                descriptor,
                *_PREFILL_OPT_ZERO_MASKS,
                T.ptx.pred(0 if kphase == 0 else 1),
                pred=T.cuda.elect_sync(),
            )
        )
    _prefill_opt_mma_commit(full_barrier)


@T.inline
def _prefill_opt_mma_ts_128x64_k64(tmem_d, tmem_a, b_desc_base, accumulate, IO_DTYPE):
    descriptor: T.uint32 = _prefill_opt_mma_descriptor(0x08100010, IO_DTYPE)
    for kphase in range(4):
        T.evaluate(
            T.ptx[_PREFILL_OPT_MMA_CHAIN](
                T.cast(tmem_d, "uint32"),
                T.cast(tmem_a + kphase * 8, "uint32"),
                b_desc_base + T.uint64(kphase * 2),
                descriptor,
                *_PREFILL_OPT_ZERO_MASKS,
                T.ptx.pred(accumulate if kphase == 0 else 1),
                pred=T.cuda.elect_sync(),
            )
        )


@T.inline
def _prefill_opt_mma_ts_128x128_k64(tmem_d, tmem_a, b_desc_base, full_barrier, IO_DTYPE):
    descriptor: T.uint32 = _prefill_opt_mma_descriptor(0x08210010, IO_DTYPE)
    for kphase in range(4):
        T.evaluate(
            T.ptx[_PREFILL_OPT_MMA_CHAIN](
                T.cast(tmem_d, "uint32"),
                T.cast(tmem_a + kphase * 8, "uint32"),
                b_desc_base + T.uint64(kphase * 128),
                descriptor,
                *_PREFILL_OPT_ZERO_MASKS,
                True,
                pred=T.cuda.elect_sync(),
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


@T.inline
def _prefill_opt_tma_q(
    smem_base_addr, smem_off, descriptor, barrier, token, subhead, base_head, IS_GQA
):
    for d_coord in range(0, D_HEAD, 64):
        if IS_GQA:
            T.ptx[_PREFILL_OPT_TMA_G2S[4]](
                _pf_shared_addr(smem_base_addr, smem_off + d_coord * 128),
                descriptor,
                T.int32(d_coord),
                T.cast(token, "int32"),
                subhead,
                base_head,
                barrier,
                T.uint64(0),
            )
        else:
            T.ptx[_PREFILL_OPT_TMA_G2S[3]](
                _pf_shared_addr(smem_base_addr, smem_off + d_coord * 128),
                descriptor,
                T.int32(d_coord),
                T.cast(token, "int32"),
                base_head,
                barrier,
                T.uint64(0),
            )


@T.inline
def _prefill_opt_tma_k(smem_base_addr, smem_off, descriptor, barrier, token, base_head):
    for d_coord in range(0, D_HEAD, 64):
        T.ptx[_PREFILL_OPT_TMA_G2S[3]](
            _pf_shared_addr(smem_base_addr, smem_off + d_coord * 128),
            descriptor,
            T.int32(d_coord),
            T.cast(token, "int32"),
            base_head,
            barrier,
            T.uint64(0),
        )


@T.inline
def _prefill_opt_tma_v(
    smem_base_addr, smem_off, descriptor, barrier, token, subhead, base_head, IS_GQA
):
    for d_coord in range(0, D_HEAD, 64):
        if IS_GQA:
            T.ptx[_PREFILL_OPT_TMA_G2S[3]](
                _pf_shared_addr(smem_base_addr, smem_off + d_coord * 128),
                descriptor,
                T.int32(d_coord),
                T.cast(token, "int32"),
                base_head,
                barrier,
                T.uint64(0),
            )
        else:
            T.ptx[_PREFILL_OPT_TMA_G2S[4]](
                _pf_shared_addr(smem_base_addr, smem_off + d_coord * 128),
                descriptor,
                T.int32(d_coord),
                T.cast(token, "int32"),
                subhead,
                base_head,
                barrier,
                T.uint64(0),
            )


@T.inline
def _prefill_opt_tma_t(smem_base_addr, smem_off, descriptor, barrier, t_block, subhead, base_head):
    T.ptx[_PREFILL_OPT_TMA_G2S[5]](
        _pf_shared_addr(smem_base_addr, smem_off),
        descriptor,
        T.int32(0),
        T.int32(0),
        subhead,
        base_head,
        t_block,
        barrier,
        T.uint64(0),
    )


@T.inline
def _prefill_opt_load_gate(
    smem_base_addr,
    s_cumsumlog,
    s_cumprod,
    alpha,
    chunk_offset,
    state_head,
    is_last_tile,
    chunk_end,
    lane,
    gate_index,
    gate_phase,
    STATE_HEADS,
):
    pos0: T.int64 = T.cast(chunk_offset, "int64") + T.cast(lane, "int64")
    pos1: T.int64 = T.cast(chunk_offset, "int64") + T.cast(lane + 32, "int64")
    valid0: T.int32 = 1
    valid1: T.int32 = 1
    gate0: T.float32 = T.float32(1.0)
    gate1: T.float32 = T.float32(1.0)
    if is_last_tile:
        valid0 = T.cast(pos0 < T.cast(chunk_end, "int64"), "int32")
        valid1 = T.cast(pos1 < T.cast(chunk_end, "int64"), "int32")
        if valid0 != 0:
            T.ptx.ld.global_.f32(
                gate0,
                alpha.ptr_to([pos0 * T.cast(STATE_HEADS, "int64") + T.cast(state_head, "int64")]),
            )
        if valid1 != 0:
            T.ptx.ld.global_.f32(
                gate1,
                alpha.ptr_to([pos1 * T.cast(STATE_HEADS, "int64") + T.cast(state_head, "int64")]),
            )
    else:
        T.ptx.ld.global_.f32(
            gate0, alpha.ptr_to([pos0 * T.cast(STATE_HEADS, "int64") + T.cast(state_head, "int64")])
        )
        T.ptx.ld.global_.f32(
            gate1, alpha.ptr_to([pos1 * T.cast(STATE_HEADS, "int64") + T.cast(state_head, "int64")])
        )
    T.ptx.lg2.approx.ftz.f32(gate0, gate0 + T.float32(1.0e-10))
    T.ptx.lg2.approx.ftz.f32(gate1, gate1 + T.float32(1.0e-10))
    for scan_step in T.unroll(5):
        scan_offset: T.int32 = 1 << scan_step
        prior0: T.float32 = T.tvm_warp_shuffle_up(T.uint32(0xFFFFFFFF), gate0, scan_offset, 32, 32)
        prior1: T.float32 = T.tvm_warp_shuffle_up(T.uint32(0xFFFFFFFF), gate1, scan_offset, 32, 32)
        if lane >= scan_offset:
            gate0 = gate0 + prior0
            gate1 = gate1 + prior1
    gate1 = gate1 + T.cuda._shfl_sync(T.uint32(0xFFFFFFFF), gate0, 31, 32)
    cumprod0: T.float32
    cumprod1: T.float32
    T.ptx.ex2.approx.ftz.f32(cumprod0, gate0)
    T.ptx.ex2.approx.ftz.f32(cumprod1, gate1)
    _pf_producer_acquire_state(smem_base_addr, 168, gate_index, gate_phase)
    T.ptx.st.shared.f32(s_cumsumlog.ptr_to([gate_index * T_BLOCK + lane]), gate0)
    T.ptx.st.shared.f32(s_cumsumlog.ptr_to([gate_index * T_BLOCK + lane + 32]), gate1)
    T.ptx.st.shared.f32(s_cumprod.ptr_to([gate_index * T_BLOCK + lane]), cumprod0)
    T.ptx.st.shared.f32(s_cumprod.ptr_to([gate_index * T_BLOCK + lane + 32]), cumprod1)
    _pf_software_commit_state(smem_base_addr, 128, gate_index)


@T.inline
def _prefill_opt_store_o(
    smem_base_addr, descriptor, chunk_offset, subhead, base_head, o_index, o_phase
):
    _pf_consumer_wait_state(smem_base_addr, 464, o_index, o_phase)
    if T.cuda.elect_sync():
        for d_coord in range(0, D_HEAD, 64):
            T.ptx[_PREFILL_OPT_TMA_S2G_4D](
                descriptor,
                T.int32(d_coord),
                T.cast(chunk_offset, "int32"),
                subhead,
                base_head,
                _pf_shared_addr(
                    smem_base_addr, PREFILL_OPT_O_OFF + o_index * 16384 + d_coord * 128
                ),
                T.uint64(0),
            )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
    _pf_consumer_release_state(smem_base_addr, 480, o_index)


@T.jit
def _t_precompute_sm100(
    k_h: T.handle,
    beta_h: T.handle,
    t_h: T.handle,
    cu_seqlens_h: T.handle,
    k_map: T.TensorMap(),
    *,
    IO_DTYPE: T.constexpr,
    CU_DTYPE: T.constexpr,
    TOTAL_TOKENS: T.constexpr,
    NUM_SEQUENCES: T.constexpr,
    K_HEADS: T.constexpr,
    STATE_HEADS: T.constexpr,
    TOTAL_T_BLOCKS: T.constexpr,
    MAX_T_BLOCKS: T.constexpr,
):
    """Form the signed beta-folded 64-token triangular-solve tiles."""
    k = T.match_buffer(k_h, (TOTAL_TOKENS * K_HEADS * D_HEAD,), IO_DTYPE, scope="global")
    beta = T.match_buffer(beta_h, (TOTAL_TOKENS * STATE_HEADS,), "float32", scope="global")
    t = T.match_buffer(
        t_h, (TOTAL_T_BLOCKS * STATE_HEADS * T_BLOCK * T_BLOCK,), IO_DTYPE, scope="global"
    )
    cu_seqlens = T.match_buffer(cu_seqlens_h, (NUM_SEQUENCES + 1,), CU_DTYPE, scope="global")
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 8})
    # TIRX_TRANSCRIBE_START cp_delta_rule_t_precompute_sm100

    bx, seq_idx = T.cta_id([STATE_HEADS * MAX_T_BLOCKS, NUM_SEQUENCES])
    tid = T.thread_id([T_THREADS])
    state_head: T.int32 = bx % STATE_HEADS
    block_in_seq: T.int32 = bx // STATE_HEADS
    k_head: T.int32 = state_head * K_HEADS // STATE_HEADS
    sequence_bounds = T.alloc_local((2,), "int32")
    _load_sequence_bounds(cu_seqlens, seq_idx, sequence_bounds, CU_DTYPE)
    seq_start: T.int32 = sequence_bounds[0]
    seq_end: T.int32 = sequence_bounds[1]
    seq_len: T.int32 = seq_end - seq_start
    num_blocks: T.int32 = (seq_len + T_BLOCK - 1) // T_BLOCK

    if block_in_seq < num_blocks:
        token_start: T.int32 = seq_start + block_in_seq * T_BLOCK
        valid_len: T.int32 = T.min(T_BLOCK, seq_end - token_start)
        t_block: T.int32 = _device_chunk_bound(seq_idx, seq_start, T_BLOCK) + block_in_seq
        lane: T.int32 = tid & 31
        warp: T.int32 = tid >> 5
        pool = T.SMEMPool()
        smem_raw = pool.alloc((24960,), "uint8", align=1024)
        inverse_storage = T.decl_buffer(
            (T_BLOCK * T_BLOCK,),
            "uint16",
            data=smem_raw.data,
            scope="shared.dyn",
            byte_offset=16512,
            align=16,
        )
        beta_storage = T.decl_buffer(
            (T_BLOCK,),
            "float32",
            data=smem_raw.data,
            scope="shared.dyn",
            byte_offset=24704,
            align=16,
        )
        pool.commit()
        smem_addr: T.uint32 = T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0]))

        if tid == 0:
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([8]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([16]), T.uint32(32))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([24]), T.uint32(128))
            T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cta_sync()

        if warp == 1:
            if T.cuda.elect_sync():
                T.evaluate(T.ptx.prefetch.tensormap(T.address_of(k_map)))
                k_full: T.uint32 = _mn_opt_shared_addr(smem_addr, 0)
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(k_full, T.uint32(16384))
                _mn_opt_tma_kv(smem_addr, 128, T.address_of(k_map), k_full, token_start, k_head)
        elif warp == 2:
            for beta_half in T.unroll(2):
                beta_row: T.int32 = lane + beta_half * 32
                beta_value: T.float32 = 0.0
                if beta_row < valid_len:
                    _load_global_f32(
                        beta_value,
                        beta.ptr_to([(token_start + beta_row) * STATE_HEADS + state_head]),
                    )
                _store_shared_f32(beta_storage.ptr_to([beta_row]), beta_value)
            T.ptx.fence.proxy.async_.shared__cta()
            _mn_opt_commit(smem_addr, 16, 0, 1)

        _mn_opt_consumer_wait(smem_addr, 0, 0, 1)
        _mn_opt_consumer_wait(smem_addr, 16, 0, 1)
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        a_regs: T.uint32[4]
        b_regs: T.uint32[4]
        kk_acc: T.float32[32]
        packed_kk: T.uint32[4]
        for acc_idx in T.unroll(32):
            kk_acc[acc_idx] = 0.0
        for k_tile in T.unroll(D_HEAD // 16):
            _t_ldmatrix_x4_k_a(smem_raw, warp * 16, k_tile * 16, lane, a_regs)
            for n_group in T.unroll(T_BLOCK // 16):
                _t_ldmatrix_x4_k_b(smem_raw, n_group * 16, k_tile * 16, lane, b_regs)
                if IO_DTYPE == "float16":
                    if k_tile == 0:
                        _t_mma_m16n8k16_f16_zero(kk_acc, a_regs, b_regs, n_group * 8, 0)
                        _t_mma_m16n8k16_f16_zero(kk_acc, a_regs, b_regs, n_group * 8 + 4, 2)
                    else:
                        _t_mma_m16n8k16_f16_acc(kk_acc, a_regs, b_regs, n_group * 8, 0)
                        _t_mma_m16n8k16_f16_acc(kk_acc, a_regs, b_regs, n_group * 8 + 4, 2)
                else:
                    if k_tile == 0:
                        _t_mma_m16n8k16_bf16_zero(kk_acc, a_regs, b_regs, n_group * 8, 0)
                        _t_mma_m16n8k16_bf16_zero(kk_acc, a_regs, b_regs, n_group * 8 + 4, 2)
                    else:
                        _t_mma_m16n8k16_bf16_acc(kk_acc, a_regs, b_regs, n_group * 8, 0)
                        _t_mma_m16n8k16_bf16_acc(kk_acc, a_regs, b_regs, n_group * 8 + 4, 2)
        T.cuda.cta_sync()

        for n_group in T.unroll(T_BLOCK // 16):
            for element in T.unroll(8):
                within_mma: T.int32 = element & 3
                row_kk: T.int32 = warp * 16 + (lane >> 2) + (within_mma >> 1) * 8
                col_kk: T.int32 = (
                    n_group * 16 + (element >> 2) * 8 + (lane & 3) * 2 + (within_mma & 1)
                )
                value_kk: T.float32 = 0.0
                if row_kk > col_kk:
                    beta_value: T.float32
                    _load_shared_f32(beta_value, beta_storage.ptr_to([row_kk]))
                    value_kk = kk_acc[n_group * 8 + element] * beta_value
                kk_acc[n_group * 8 + element] = value_kk
            for pair in T.unroll(4):
                packed_kk[pair] = _t_pack_f16x2(
                    kk_acc[n_group * 8 + pair * 2], kk_acc[n_group * 8 + pair * 2 + 1]
                )
            _t_stmatrix_x4(inverse_storage, warp * 16, n_group * 16, lane, packed_kk)
        if lane == 0:
            _mn_opt_release(smem_addr, 8, 0, 1)
        T.cuda.cta_sync()

        # Match CollectiveInverse exactly: independent 8x8 elimination,
        # followed by HMMA block composition at 16, 32, and 64.
        if tid < 64:
            block8: T.int32 = (tid >> 3) * 8
            row8: T.int32 = tid & 7
            inverse_words: T.uint32[4]
            inverse_row: T.float32[8]
            T.ptx.ld.shared.v4.u32(
                inverse_words[0],
                inverse_words[1],
                inverse_words[2],
                inverse_words[3],
                _t_matrix_ptr(inverse_storage, block8 + row8, block8),
            )
            for pair8 in T.unroll(4):
                inverse_row[pair8 * 2] = T.cast(
                    T.reinterpret(
                        "float16", T.cast(inverse_words[pair8] & T.uint32(0xFFFF), "uint16")
                    ),
                    "float32",
                )
                inverse_row[pair8 * 2 + 1] = T.cast(
                    T.reinterpret("float16", T.cast(inverse_words[pair8] >> 16, "uint16")),
                    "float32",
                )
            for col8 in T.unroll(8):
                raw_value: T.float32 = inverse_row[col8]
                if row8 == col8:
                    inverse_row[col8] = 1.0
                elif row8 < col8:
                    inverse_row[col8] = 0.0
                else:
                    inverse_row[col8] = raw_value
            for src_row in T.unroll(7):
                row_scale: T.float32
                T.ptx.neg.f32(row_scale, inverse_row[src_row])
                for inverse_col in T.unroll(7):
                    if inverse_col < src_row:
                        pivot: T.float32 = T.cuda._shfl_sync(
                            T.uint32(0xFFFFFFFF), inverse_row[inverse_col], src_row, 8
                        )
                        if row8 > src_row:
                            inverse_row[inverse_col] = inverse_row[inverse_col] + row_scale * pivot
                if row8 > src_row:
                    inverse_row[src_row] = row_scale
            for pair8 in T.unroll(4):
                inverse_words[pair8] = _t_pack_f16x2(
                    inverse_row[pair8 * 2], inverse_row[pair8 * 2 + 1]
                )
            T.ptx.st.shared.v4.u32(
                _t_matrix_ptr(inverse_storage, block8 + row8, block8),
                inverse_words[0],
                inverse_words[1],
                inverse_words[2],
                inverse_words[3],
            )
        T.cuda.cta_sync()
        _t_inverse_8_to_16(inverse_storage, warp * 16, lane)
        T.cuda.cta_sync()
        if tid < 64:
            _t_inverse_16_to_32(inverse_storage, warp * 32, lane)
        T.cuda.cta_sync()
        _t_inverse_32_to_64(inverse_storage, warp, lane)
        T.cuda.cta_sync()

        inverse_t0: T.uint32[4]
        inverse_t1: T.uint32[4]
        inverse_t2: T.uint32[4]
        inverse_t3: T.uint32[4]
        _t_ldmatrix_x4(inverse_storage, warp * 16, 0, lane, False, inverse_t0)
        _t_ldmatrix_x4(inverse_storage, warp * 16, 16, lane, False, inverse_t1)
        _t_ldmatrix_x4(inverse_storage, warp * 16, 32, lane, False, inverse_t2)
        _t_ldmatrix_x4(inverse_storage, warp * 16, 48, lane, False, inverse_t3)
        t_base: T.int64 = T.cast((t_block * STATE_HEADS + state_head) * T_BLOCK * T_BLOCK, "int64")
        _t_store_t_fragment(inverse_t0, beta_storage, t, t_base, valid_len, warp, lane, 0, IO_DTYPE)
        _t_store_t_fragment(inverse_t1, beta_storage, t, t_base, valid_len, warp, lane, 1, IO_DTYPE)
        _t_store_t_fragment(inverse_t2, beta_storage, t, t_base, valid_len, warp, lane, 2, IO_DTYPE)
        _t_store_t_fragment(inverse_t3, beta_storage, t, t_base, valid_len, warp, lane, 3, IO_DTYPE)
        _mn_opt_release(smem_addr, 24, 0, 1)


@T.jit
def _mn_precompute_sm100(
    k_h: T.handle,
    v_h: T.handle,
    t_h: T.handle,
    alpha_h: T.handle,
    transfer_h: T.handle,
    local_state_h: T.handle,
    cu_seqlens_h: T.handle,
    k_map: T.TensorMap(),
    v_map: T.TensorMap(),
    t_map: T.TensorMap(),
    *,
    IO_DTYPE: T.constexpr,
    CU_DTYPE: T.constexpr,
    TOTAL_TOKENS: T.constexpr,
    NUM_SEQUENCES: T.constexpr,
    K_HEADS: T.constexpr,
    V_HEADS: T.constexpr,
    STATE_HEADS: T.constexpr,
    TOTAL_T_BLOCKS: T.constexpr,
    TOTAL_CP_CHUNKS: T.constexpr,
    MAX_CP_CHUNKS: T.constexpr,
    CP_CHUNK_LEN: T.constexpr,
):
    """Build each CP affine map with the source SM100 TMA/TCGEN schedule."""
    k = T.match_buffer(k_h, (TOTAL_TOKENS * K_HEADS * D_HEAD,), IO_DTYPE, scope="global")
    v = T.match_buffer(v_h, (TOTAL_TOKENS * V_HEADS * D_HEAD,), IO_DTYPE, scope="global")
    t = T.match_buffer(
        t_h, (TOTAL_T_BLOCKS * STATE_HEADS * T_BLOCK * T_BLOCK,), IO_DTYPE, scope="global"
    )
    alpha = T.match_buffer(alpha_h, (TOTAL_TOKENS * STATE_HEADS,), "float32", scope="global")
    transfer = T.match_buffer(
        transfer_h, (TOTAL_CP_CHUNKS * STATE_HEADS * D_HEAD * D_HEAD,), "float32", scope="global"
    )
    local_state = T.match_buffer(
        local_state_h, (TOTAL_CP_CHUNKS * STATE_HEADS * D_HEAD * D_HEAD,), "float32", scope="global"
    )
    cu_seqlens = T.match_buffer(cu_seqlens_h, (NUM_SEQUENCES + 1,), CU_DTYPE, scope="global")
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    # TIRX_TRANSCRIBE_START cp_delta_rule_mn_precompute_sm100

    bx, seq_idx = T.cta_id([STATE_HEADS * MAX_CP_CHUNKS, NUM_SEQUENCES])
    tid = T.thread_id([MN_THREADS])
    warp: T.int32 = T.cast(
        T.cuda._shfl_sync(T.uint32(0xFFFFFFFF), T.cast(tid >> 5, "uint32"), 0, 32), "int32"
    )
    lane: T.int32 = tid & 31
    state_head: T.int32 = bx % STATE_HEADS
    chunk_in_seq: T.int32 = bx // STATE_HEADS
    k_head: T.int32 = state_head * K_HEADS // STATE_HEADS
    v_head: T.int32 = state_head * V_HEADS // STATE_HEADS
    sequence_bounds = T.alloc_local((2,), "int32")
    _load_sequence_bounds(cu_seqlens, seq_idx, sequence_bounds, CU_DTYPE)
    seq_start: T.int32 = sequence_bounds[0]
    seq_end: T.int32 = sequence_bounds[1]
    seq_len: T.int32 = seq_end - seq_start
    num_chunks: T.int32 = (seq_len + CP_CHUNK_LEN - 1) // CP_CHUNK_LEN
    valid_chunk: T.int32 = T.cast(chunk_in_seq < num_chunks, "int32")
    token_start: T.int32 = seq_start + chunk_in_seq * CP_CHUNK_LEN
    valid_len: T.int32 = T.min(CP_CHUNK_LEN, seq_end - token_start)
    num_blocks: T.int32 = (valid_len + T_BLOCK - 1) // T_BLOCK
    t_blocks_per_chunk: T.int32 = CP_CHUNK_LEN // T_BLOCK
    t_block_start: T.int32 = (
        _device_chunk_bound(seq_idx, seq_start, T_BLOCK) + chunk_in_seq * t_blocks_per_chunk
    )

    pool = T.SMEMPool()
    smem_raw = pool.alloc((MN_OPT_SMEM_TOTAL,), "uint8", align=1024)
    tmem_holding = T.decl_buffer(
        (1,),
        "int32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=MN_OPT_TMEM_HOLDING_OFF,
        align=4,
    )
    pool.commit()
    smem_addr: T.uint32 = T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0]))

    if warp == 0:
        if T.cuda.elect_sync():
            _mn_opt_init_all_pipelines(smem_raw)
            T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 8:
        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(k_map)))
        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(v_map)))
        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(t_map)))

    if warp == 4:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
            T.address_of(tmem_holding[0]), T.uint32(MN_OPT_TMEM_COLUMNS)
        )

    if valid_chunk != 0 and warp >= 8:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(72)

    if valid_chunk == 0:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
        tmem_base_invalid: T.int32 = 0
        if warp <= 8 or warp == 11:
            T.ptx.bar.sync(T.uint32(MN_OPT_TMEM_ALLOC_BARRIER), T.uint32(320))
            T.ptx.ld.volatile.shared.s32(tmem_base_invalid, T.address_of(tmem_holding[0]))
        if warp == 4:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                T.cast(tmem_base_invalid, "uint32"), T.uint32(MN_OPT_TMEM_COLUMNS)
            )
    elif warp <= 3:
        T.ptx.setmaxnreg.inc.sync.aligned.u32(216)
        T.ptx.barrier.sync(T.uint32(MN_OPT_TMEM_ALLOC_BARRIER), T.uint32(320))
        tmem_base_cg0: T.int32
        T.ptx.ld.volatile.shared.s32(tmem_base_cg0, T.address_of(tmem_holding[0]))
        cg0_thread: T.int32 = tid

        _mn_opt_initialize_matrix(tmem_base_cg0, MN_OPT_TMEM_M_COL, cg0_thread, True)
        _mn_opt_producer_acquire(smem_addr, 216, 0, 1)
        _mn_opt_commit(smem_addr, 208, 0, 1)

        for block in T.serial(num_blocks):
            _mn_opt_consumer_wait(smem_addr, 144, block, 4)
            _mn_opt_consumer_wait(smem_addr, 240, block, 1)
            _mn_opt_producer_acquire(smem_addr, 272, block, 2)
            _mn_opt_materialize_x(
                tmem_base_cg0, smem_raw, _mn_opt_stage(block, 2), cg0_thread, IO_DTYPE
            )
            _mn_opt_release(smem_addr, 248, block, 1)
            _mn_opt_commit(smem_addr, 256, block, 2)

            if block > 0:
                recurrence: T.int32 = block - 1
                _mn_opt_producer_acquire(smem_addr, 296, recurrence, 1)
                _mn_opt_matrix_to_io_input(
                    tmem_base_cg0, MN_OPT_TMEM_M_COL, MN_OPT_TMEM_M_INPUT_COL, cg0_thread, IO_DTYPE
                )
                _mn_opt_commit(smem_addr, 288, recurrence, 1)
                _mn_opt_consumer_wait(smem_addr, 320, recurrence, 1)
                _mn_opt_scratch_to_io_input(
                    tmem_base_cg0, MN_OPT_TMEM_M_INPUT_COL, cg0_thread, IO_DTYPE
                )
                _mn_opt_release(smem_addr, 328, recurrence, 1)
                _mn_opt_producer_acquire(smem_addr, 344, recurrence, 1)
                _mn_opt_commit(smem_addr, 336, recurrence, 1)

            block_coeff_cg0: T.float32
            alpha_stage_cg0: T.int32 = _mn_opt_stage(block, 4)
            T.ptx.ld.shared.f32(
                block_coeff_cg0,
                smem_raw.ptr_to(
                    [MN_OPT_ALPHA_OFF + (alpha_stage_cg0 * T_BLOCK * 3 + T_BLOCK + T_BLOCK - 1) * 4]
                ),
            )
            _mn_opt_consumer_wait(smem_addr, 352, block, 1)
            _mn_opt_scale_matrix(tmem_base_cg0, MN_OPT_TMEM_M_COL, cg0_thread, block_coeff_cg0)
            _mn_opt_release(smem_addr, 360, block, 1)
            _mn_opt_release(smem_addr, 176, block, 4)

        cp_slot_cg0: T.int32 = _device_chunk_bound(seq_idx, seq_start, CP_CHUNK_LEN) + chunk_in_seq
        output_base_cg0: T.int64 = (
            T.cast(cp_slot_cg0 * STATE_HEADS + state_head, "int64") * D_HEAD * D_HEAD
        )
        _mn_opt_store_matrix_global(
            tmem_base_cg0, MN_OPT_TMEM_M_COL, transfer, output_base_cg0, cg0_thread
        )
        _mn_opt_producer_acquire(smem_addr, 424, 0, 1)
        _mn_opt_commit(smem_addr, 416, 0, 1)

    elif warp <= 7:
        T.ptx.setmaxnreg.inc.sync.aligned.u32(216)
        T.ptx.barrier.sync(T.uint32(MN_OPT_TMEM_ALLOC_BARRIER), T.uint32(320))
        tmem_base_cg1: T.int32
        T.ptx.ld.volatile.shared.s32(tmem_base_cg1, T.address_of(tmem_holding[0]))
        cg1_thread: T.int32 = tid & 127

        _mn_opt_initialize_matrix(tmem_base_cg1, MN_OPT_TMEM_N_COL, cg1_thread, False)
        _mn_opt_producer_acquire(smem_addr, 232, 0, 1)
        _mn_opt_commit(smem_addr, 224, 0, 1)

        for block in T.serial(num_blocks):
            _mn_opt_consumer_wait(smem_addr, 48, block, 3)
            _mn_opt_consumer_wait(smem_addr, 144, block, 4)
            _mn_opt_matrix_to_io_input(
                tmem_base_cg1, MN_OPT_TMEM_N_COL, MN_OPT_TMEM_N_INPUT_COL, cg1_thread, IO_DTYPE
            )
            _mn_opt_producer_acquire(smem_addr, 312, block, 1)
            _mn_opt_commit(smem_addr, 304, block, 1)
            _mn_opt_consumer_wait(smem_addr, 368, block, 1)

            block_coeff_cg1: T.float32
            alpha_stage_cg1: T.int32 = _mn_opt_stage(block, 4)
            T.ptx.ld.shared.f32(
                block_coeff_cg1,
                smem_raw.ptr_to(
                    [MN_OPT_ALPHA_OFF + (alpha_stage_cg1 * T_BLOCK * 3 + T_BLOCK + T_BLOCK - 1) * 4]
                ),
            )
            _mn_opt_process_y(
                tmem_base_cg1,
                smem_raw,
                _mn_opt_stage(block, 3),
                alpha_stage_cg1,
                block_coeff_cg1,
                cg1_thread,
                IO_DTYPE,
            )
            _mn_opt_release(smem_addr, 376, block, 1)
            _mn_opt_scale_matrix(tmem_base_cg1, MN_OPT_TMEM_N_COL, cg1_thread, block_coeff_cg1)
            _mn_opt_producer_acquire(smem_addr, 392, block, 1)
            _mn_opt_commit(smem_addr, 384, block, 1)
            _mn_opt_consumer_wait(smem_addr, 400, block, 1)
            _mn_opt_release(smem_addr, 408, block, 1)
            if lane == 0:
                _mn_opt_release(smem_addr, 72, block, 3)
            _mn_opt_release(smem_addr, 176, block, 4)

        cp_slot_cg1: T.int32 = _device_chunk_bound(seq_idx, seq_start, CP_CHUNK_LEN) + chunk_in_seq
        output_base_cg1: T.int64 = (
            T.cast(cp_slot_cg1 * STATE_HEADS + state_head, "int64") * D_HEAD * D_HEAD
        )
        _mn_opt_store_matrix_global(
            tmem_base_cg1, MN_OPT_TMEM_N_COL, local_state, output_base_cg1, cg1_thread
        )
        _mn_opt_consumer_wait(smem_addr, 416, 0, 1)
        _mn_opt_release(smem_addr, 424, 0, 1)
        T.cuda.warpgroup_sync(MN_OPT_TMEM_DEALLOC_BARRIER)
        if warp == 4:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                T.cast(tmem_base_cg1, "uint32"), T.uint32(MN_OPT_TMEM_COLUMNS)
            )

    elif warp == 8:
        T.ptx.barrier.sync(T.uint32(MN_OPT_TMEM_ALLOC_BARRIER), T.uint32(320))
        tmem_base_transfer: T.int32
        T.ptx.ld.volatile.shared.s32(tmem_base_transfer, T.address_of(tmem_holding[0]))
        _mn_opt_consumer_wait(smem_addr, 208, 0, 1)
        if T.cuda.elect_sync():
            _mn_opt_release(smem_addr, 216, 0, 1)

        for block in T.serial(num_blocks):
            k_stage_transfer: T.int32 = _mn_opt_stage(block, 3)
            x_stage_transfer: T.int32 = _mn_opt_stage(block, 2)
            _mn_opt_consumer_wait(smem_addr, 0, block, 3)
            k_addr_transfer: T.uint32 = _mn_opt_shared_addr(
                smem_addr, MN_OPT_K_OFF + k_stage_transfer * 16384
            )
            k_desc_transfer: T.uint64 = _mn_opt_smem_desc_k(k_addr_transfer)
            k_desc_transfer_mn: T.uint64 = _mn_opt_smem_desc_mn(k_addr_transfer)
            if block > 0:
                recurrence: T.int32 = block - 1
                _mn_opt_consumer_wait(smem_addr, 288, recurrence, 1)
                _mn_opt_producer_acquire(smem_addr, 328, recurrence, 1)
                _mn_opt_mma_ts_128x64_k128(
                    tmem_base_transfer + MN_OPT_TMEM_SCRATCH_COL,
                    tmem_base_transfer + MN_OPT_TMEM_M_INPUT_COL,
                    k_desc_transfer,
                    _mn_opt_full_addr(smem_addr, 320, recurrence, 1),
                    IO_DTYPE,
                )
                _mn_opt_consumer_wait(smem_addr, 336, recurrence, 1)
                if T.cuda.elect_sync():
                    _mn_opt_release(smem_addr, 344, recurrence, 1)
                _mn_opt_mma_commit(_mn_opt_empty_addr(smem_addr, 296, recurrence, 1))

            _mn_opt_consumer_wait(smem_addr, 256, block, 2)
            _mn_opt_producer_acquire(smem_addr, 360, block, 1)
            x_desc_transfer: T.uint64 = _mn_opt_smem_desc_mn(
                _mn_opt_shared_addr(smem_addr, MN_OPT_X_OFF + x_stage_transfer * 16384)
            )
            if block == 0:
                _mn_opt_mma_ss_128x128_k64(
                    tmem_base_transfer + MN_OPT_TMEM_M_COL,
                    k_desc_transfer_mn,
                    x_desc_transfer,
                    _mn_opt_full_addr(smem_addr, 352, block, 1),
                    IO_DTYPE,
                )
            else:
                _mn_opt_mma_ts_128x128_k64(
                    tmem_base_transfer + MN_OPT_TMEM_M_COL,
                    tmem_base_transfer + MN_OPT_TMEM_M_INPUT_COL,
                    x_desc_transfer,
                    _mn_opt_full_addr(smem_addr, 352, block, 1),
                    IO_DTYPE,
                )
            _mn_opt_mma_commit(_mn_opt_empty_addr(smem_addr, 272, block, 2))
            _mn_opt_mma_commit(_mn_opt_empty_addr(smem_addr, 24, block, 3))

    elif warp == 11:
        T.ptx.barrier.sync(T.uint32(MN_OPT_TMEM_ALLOC_BARRIER), T.uint32(320))
        tmem_base_state: T.int32
        T.ptx.ld.volatile.shared.s32(tmem_base_state, T.address_of(tmem_holding[0]))
        _mn_opt_consumer_wait(smem_addr, 224, 0, 1)
        if T.cuda.elect_sync():
            _mn_opt_release(smem_addr, 232, 0, 1)

        for block in T.serial(num_blocks):
            k_stage_state: T.int32 = _mn_opt_stage(block, 3)
            t_stage_state: T.int32 = _mn_opt_stage(block, 3)
            x_stage_state: T.int32 = _mn_opt_stage(block, 2)
            _mn_opt_consumer_wait(smem_addr, 0, block, 3)
            _mn_opt_consumer_wait(smem_addr, 96, block, 3)
            _mn_opt_producer_acquire(smem_addr, 248, block, 1)
            k_addr_state: T.uint32 = _mn_opt_shared_addr(
                smem_addr, MN_OPT_K_OFF + k_stage_state * 16384
            )
            k_desc_state: T.uint64 = _mn_opt_smem_desc_k(k_addr_state)
            k_desc_state_mn: T.uint64 = _mn_opt_smem_desc_mn(k_addr_state)
            t_desc_state: T.uint64 = _mn_opt_smem_desc_k(
                _mn_opt_shared_addr(smem_addr, MN_OPT_T_OFF + t_stage_state * 8192)
            )
            _mn_opt_mma_ss_128x64_k64(
                tmem_base_state + MN_OPT_TMEM_XY_COL,
                k_desc_state_mn,
                t_desc_state,
                _mn_opt_full_addr(smem_addr, 240, block, 1),
                IO_DTYPE,
            )
            _mn_opt_mma_commit(_mn_opt_empty_addr(smem_addr, 120, block, 3))
            _mn_opt_consumer_wait(smem_addr, 256, block, 2)

            _mn_opt_consumer_wait(smem_addr, 304, block, 1)
            _mn_opt_producer_acquire(smem_addr, 376, block, 1)
            _mn_opt_mma_ts_128x64_k128(
                tmem_base_state + MN_OPT_TMEM_XY_COL,
                tmem_base_state + MN_OPT_TMEM_N_INPUT_COL,
                k_desc_state,
                _mn_opt_full_addr(smem_addr, 368, block, 1),
                IO_DTYPE,
            )
            _mn_opt_mma_commit(_mn_opt_empty_addr(smem_addr, 312, block, 1))
            _mn_opt_mma_commit(_mn_opt_empty_addr(smem_addr, 24, block, 3))

            _mn_opt_consumer_wait(smem_addr, 384, block, 1)
            if T.cuda.elect_sync():
                _mn_opt_release(smem_addr, 392, block, 1)
            _mn_opt_producer_acquire(smem_addr, 408, block, 1)
            x_desc_state: T.uint64 = _mn_opt_smem_desc_mn(
                _mn_opt_shared_addr(smem_addr, MN_OPT_X_OFF + x_stage_state * 16384)
            )
            _mn_opt_mma_ts_128x128_k64(
                tmem_base_state + MN_OPT_TMEM_N_COL,
                tmem_base_state + MN_OPT_TMEM_N_INPUT_COL,
                x_desc_state,
                _mn_opt_full_addr(smem_addr, 400, block, 1),
                IO_DTYPE,
            )
            _mn_opt_mma_commit(_mn_opt_empty_addr(smem_addr, 272, block, 2))

    elif warp == 9:
        for block in T.serial(num_blocks):
            k_stage_tma: T.int32 = _mn_opt_stage(block, 3)
            v_stage_tma: T.int32 = _mn_opt_stage(block, 3)
            t_stage_tma: T.int32 = _mn_opt_stage(block, 3)
            _mn_opt_producer_acquire(smem_addr, 24, block, 3)
            _mn_opt_producer_acquire(smem_addr, 72, block, 3)
            _mn_opt_producer_acquire(smem_addr, 120, block, 3)
            if T.cuda.elect_sync():
                k_full_tma: T.uint32 = _mn_opt_full_addr(smem_addr, 0, block, 3)
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(k_full_tma, T.uint32(16384))
                _mn_opt_tma_kv(
                    smem_addr,
                    MN_OPT_K_OFF + k_stage_tma * 16384,
                    T.address_of(k_map),
                    k_full_tma,
                    token_start + block * T_BLOCK,
                    k_head,
                )
                v_full_tma: T.uint32 = _mn_opt_full_addr(smem_addr, 48, block, 3)
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(v_full_tma, T.uint32(16384))
                _mn_opt_tma_kv(
                    smem_addr,
                    MN_OPT_V_OFF + v_stage_tma * 16384,
                    T.address_of(v_map),
                    v_full_tma,
                    token_start + block * T_BLOCK,
                    v_head,
                )
                t_full_tma: T.uint32 = _mn_opt_full_addr(smem_addr, 96, block, 3)
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(t_full_tma, T.uint32(8192))
                _mn_opt_tma_t(
                    smem_addr,
                    MN_OPT_T_OFF + t_stage_tma * 8192,
                    T.address_of(t_map),
                    t_full_tma,
                    t_block_start + block,
                    state_head,
                )

    elif warp == 10:
        for block in T.serial(num_blocks):
            _mn_opt_producer_acquire(smem_addr, 176, block, 4)
            token0: T.int32 = block * T_BLOCK + lane
            token1: T.int32 = token0 + 32
            alpha0: T.float32 = T.float32(1.0)
            alpha1: T.float32 = T.float32(1.0)
            if token0 < valid_len:
                T.ptx.ld.global_.f32(
                    alpha0,
                    alpha.ptr_to(
                        [T.cast(token_start + token0, "int64") * STATE_HEADS + state_head]
                    ),
                )
            if token1 < valid_len:
                T.ptx.ld.global_.f32(
                    alpha1,
                    alpha.ptr_to(
                        [T.cast(token_start + token1, "int64") * STATE_HEADS + state_head]
                    ),
                )
            log0: T.float32 = _lg2_approx_ftz(alpha0 + T.float32(1.0e-10))
            log1: T.float32 = _lg2_approx_ftz(alpha1 + T.float32(1.0e-10))
            for scan_step in T.unroll(5):
                scan_offset: T.int32 = 1 << scan_step
                prior0: T.float32 = T.cuda._shfl_up_sync(
                    T.uint32(0xFFFFFFFF), log0, scan_offset, 32
                )
                prior1: T.float32 = T.cuda._shfl_up_sync(
                    T.uint32(0xFFFFFFFF), log1, scan_offset, 32
                )
                if lane >= scan_offset:
                    log0 = log0 + prior0
                    log1 = log1 + prior1
            log1 = log1 + T.cuda._shfl_sync(T.uint32(0xFFFFFFFF), log0, 31, 32)
            end_log: T.float32 = T.cuda._shfl_sync(T.uint32(0xFFFFFFFF), log1, 31, 32)
            cumprod0: T.float32 = _ex2_approx_ftz(log0)
            cumprod1: T.float32 = _ex2_approx_ftz(log1)
            neg0: T.float32
            neg1: T.float32
            T.ptx.neg.f32(neg0, _ex2_approx_ftz(end_log - log0))
            T.ptx.neg.f32(neg1, _ex2_approx_ftz(end_log - log1))
            if token0 >= valid_len:
                neg0 = T.float32(0.0)
            if token1 >= valid_len:
                neg1 = T.float32(0.0)
            alpha_stage_loader: T.int32 = _mn_opt_stage(block, 4)
            alpha_stage_base: T.int32 = MN_OPT_ALPHA_OFF + alpha_stage_loader * T_BLOCK * 3 * 4
            T.ptx.st.shared.f32(smem_raw.ptr_to([alpha_stage_base + lane * 4]), log0)
            T.ptx.st.shared.f32(smem_raw.ptr_to([alpha_stage_base + (lane + 32) * 4]), log1)
            T.ptx.st.shared.f32(
                smem_raw.ptr_to([alpha_stage_base + (T_BLOCK + lane) * 4]), cumprod0
            )
            T.ptx.st.shared.f32(
                smem_raw.ptr_to([alpha_stage_base + (T_BLOCK + lane + 32) * 4]), cumprod1
            )
            T.ptx.st.shared.f32(
                smem_raw.ptr_to([alpha_stage_base + (T_BLOCK * 2 + lane) * 4]), neg0
            )
            T.ptx.st.shared.f32(
                smem_raw.ptr_to([alpha_stage_base + (T_BLOCK * 2 + lane + 32) * 4]), neg1
            )
            T.ptx.fence.proxy.async_.shared__cta()
            _mn_opt_commit(smem_addr, 144, block, 4)


@T.jit
def _mn_precompute_scalar_sm100(
    k_h: T.handle,
    v_h: T.handle,
    t_h: T.handle,
    alpha_h: T.handle,
    transfer_h: T.handle,
    local_state_h: T.handle,
    cu_seqlens_h: T.handle,
    *,
    IO_DTYPE: T.constexpr,
    CU_DTYPE: T.constexpr,
    TOTAL_TOKENS: T.constexpr,
    NUM_SEQUENCES: T.constexpr,
    K_HEADS: T.constexpr,
    V_HEADS: T.constexpr,
    STATE_HEADS: T.constexpr,
    TOTAL_T_BLOCKS: T.constexpr,
    TOTAL_CP_CHUNKS: T.constexpr,
    MAX_CP_CHUNKS: T.constexpr,
    CP_CHUNK_LEN: T.constexpr,
):
    """Build one transposed affine state map for every CP chunk."""
    k = T.match_buffer(k_h, (TOTAL_TOKENS * K_HEADS * D_HEAD,), IO_DTYPE, scope="global")
    v = T.match_buffer(v_h, (TOTAL_TOKENS * V_HEADS * D_HEAD,), IO_DTYPE, scope="global")
    t = T.match_buffer(
        t_h, (TOTAL_T_BLOCKS * STATE_HEADS * T_BLOCK * T_BLOCK,), IO_DTYPE, scope="global"
    )
    alpha = T.match_buffer(alpha_h, (TOTAL_TOKENS * STATE_HEADS,), "float32", scope="global")
    transfer = T.match_buffer(
        transfer_h, (TOTAL_CP_CHUNKS * STATE_HEADS * D_HEAD * D_HEAD,), "float32", scope="global"
    )
    local_state = T.match_buffer(
        local_state_h, (TOTAL_CP_CHUNKS * STATE_HEADS * D_HEAD * D_HEAD,), "float32", scope="global"
    )
    cu_seqlens = T.match_buffer(cu_seqlens_h, (NUM_SEQUENCES + 1,), CU_DTYPE, scope="global")
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    # TIRX_TRANSCRIBE_START cp_delta_rule_mn_precompute_sm100

    bx, seq_idx = T.cta_id([STATE_HEADS * MAX_CP_CHUNKS, NUM_SEQUENCES])
    tid = T.thread_id([MN_THREADS])
    state_head: T.int32 = bx % STATE_HEADS
    chunk_in_seq: T.int32 = bx // STATE_HEADS
    k_head: T.int32 = state_head * K_HEADS // STATE_HEADS
    v_head: T.int32 = state_head * V_HEADS // STATE_HEADS
    seq_start: T.int32 = T.cast(cu_seqlens[seq_idx], "int32")
    seq_end: T.int32 = T.cast(cu_seqlens[seq_idx + 1], "int32")
    seq_len: T.int32 = seq_end - seq_start
    num_chunks: T.int32 = (seq_len + CP_CHUNK_LEN - 1) // CP_CHUNK_LEN

    if chunk_in_seq < num_chunks:
        x_shared = T.alloc_buffer((T_BLOCK * D_HEAD,), IO_DTYPE, scope="shared", align=128)
        y_shared = T.alloc_buffer((T_BLOCK * D_HEAD,), IO_DTYPE, scope="shared", align=128)
        n_first_shared = T.alloc_buffer((D_HEAD * D_HEAD,), "float32", scope="shared", align=128)
        alpha_shared = T.alloc_buffer((T_BLOCK * 2,), "float32", scope="shared", align=16)
        matrix_kind: T.int32 = tid // D_HEAD
        row: T.int32 = tid % D_HEAD
        values: T.float32[D_HEAD]
        block_operand = T.alloc_local((T_BLOCK,), IO_DTYPE)
        if tid < 256:
            for col in T.serial(D_HEAD):
                values[col] = T.if_then_else(
                    matrix_kind == 0 and row == col, T.float32(1.0), T.float32(0.0)
                )

        token_start: T.int32 = seq_start + chunk_in_seq * CP_CHUNK_LEN
        valid_len: T.int32 = T.min(CP_CHUNK_LEN, seq_end - token_start)
        t_start: T.int32 = _device_chunk_bound(seq_idx, seq_start, T_BLOCK)
        num_blocks_in_chunk: T.int32 = (valid_len + T_BLOCK - 1) // T_BLOCK
        for block_local in T.serial(num_blocks_in_chunk):
            block_token_start: T.int32 = token_start + block_local * T_BLOCK
            block_valid: T.int32 = T.min(T_BLOCK, seq_end - block_token_start)
            t_block: T.int32 = t_start + chunk_in_seq * (CP_CHUNK_LEN // T_BLOCK) + block_local

            # AlphaProcessor's two-half warp scan and its exact fast-math
            # channels: gamma and -gamma_end/gamma.
            if tid < 32:
                alpha0: T.float32 = 1.0
                alpha1: T.float32 = 1.0
                if tid < block_valid:
                    alpha0 = alpha[(block_token_start + tid) * STATE_HEADS + state_head]
                if tid + 32 < block_valid:
                    alpha1 = alpha[(block_token_start + tid + 32) * STATE_HEADS + state_head]
                log0: T.float32 = _lg2_approx_ftz(alpha0 + 1.0e-10)
                log1: T.float32 = _lg2_approx_ftz(alpha1 + 1.0e-10)
                for scan_step in T.unroll(5):
                    delta: T.int32 = 1 << scan_step
                    peer0: T.float32 = T.cuda._shfl_up_sync(T.uint32(0xFFFFFFFF), log0, delta, 32)
                    peer1: T.float32 = T.cuda._shfl_up_sync(T.uint32(0xFFFFFFFF), log1, delta, 32)
                    if tid >= delta:
                        log0 = log0 + peer0
                        log1 = log1 + peer1
                carry: T.float32 = T.cuda._shfl_sync(T.uint32(0xFFFFFFFF), log0, 31, 32)
                log1 = log1 + carry
                end_log: T.float32 = T.cuda._shfl_sync(T.uint32(0xFFFFFFFF), log1, 31, 32)
                gamma0: T.float32 = _ex2_approx_ftz(log0)
                gamma1: T.float32 = _ex2_approx_ftz(log1)
                neg0: T.float32
                neg1: T.float32
                T.ptx.neg.f32(neg0, _ex2_approx_ftz(end_log - log0))
                T.ptx.neg.f32(neg1, _ex2_approx_ftz(end_log - log1))
                if tid >= block_valid:
                    neg0 = 0.0
                if tid + 32 >= block_valid:
                    neg1 = 0.0
                alpha_shared[tid] = gamma0
                alpha_shared[tid + 32] = gamma1
                alpha_shared[T_BLOCK + tid] = neg0
                alpha_shared[T_BLOCK + tid + 32] = neg1
            T.cuda.cta_sync()

            # X = T @ K.  It is rounded to IO before either M or N consumes it.
            for work in T.serial((T_BLOCK * D_HEAD + MN_THREADS - 1) // MN_THREADS):
                linear_x: T.int32 = tid + work * MN_THREADS
                if linear_x < T_BLOCK * D_HEAD:
                    output_token: T.int32 = linear_x // D_HEAD
                    key_col: T.int32 = linear_x % D_HEAD
                    x_value: T.float32 = 0.0
                    for input_token in T.serial(T_BLOCK):
                        key_value: T.float32 = 0.0
                        if input_token < block_valid:
                            key_value = T.cast(
                                k[
                                    (block_token_start + input_token) * K_HEADS * D_HEAD
                                    + k_head * D_HEAD
                                    + key_col
                                ],
                                "float32",
                            )
                        x_value = (
                            x_value
                            + T.cast(
                                t[
                                    (t_block * STATE_HEADS + state_head) * T_BLOCK * T_BLOCK
                                    + output_token * T_BLOCK
                                    + input_token
                                ],
                                "float32",
                            )
                            * key_value
                        )
                    x_shared[_mn_matrix_index(key_col, output_token)] = T.cast(x_value, IO_DTYPE)
            T.cuda.cta_sync()

            # The first-block N update is one 128x128x64 Tensor Core GEMM.
            # Preserve that accumulation order because its FP32 result is the
            # IO bridge feeding every later block in the CP chunk.
            if block_local == 0:
                for work in T.serial((D_HEAD * T_BLOCK + MN_THREADS - 1) // MN_THREADS):
                    linear_y: T.int32 = tid + work * MN_THREADS
                    if linear_y < D_HEAD * T_BLOCK:
                        value_row: T.int32 = linear_y // T_BLOCK
                        token_y: T.int32 = linear_y % T_BLOCK
                        y_value: T.float32 = 0.0
                        if token_y < block_valid:
                            y_value = (
                                T.cast(
                                    v[
                                        (block_token_start + token_y) * V_HEADS * D_HEAD
                                        + v_head * D_HEAD
                                        + value_row
                                    ],
                                    "float32",
                                )
                                * alpha_shared[T_BLOCK + token_y]
                            )
                        y_shared[_mn_matrix_index(value_row, token_y)] = T.cast(y_value, IO_DTYPE)
                T.cuda.cta_sync()
                if tid < 256:
                    mma_lane: T.int32 = tid & 31
                    mma_warp: T.int32 = tid >> 5
                    y_regs: T.uint32[4]
                    x_regs: T.uint32[4]
                    n_acc: T.float32[8]
                    for n_group in T.unroll(D_HEAD // 16):
                        for k_tile in T.unroll(T_BLOCK // 16):
                            _mn_ldmatrix_x4_a(
                                y_shared, mma_warp * 16, k_tile * 16, mma_lane, y_regs
                            )
                            _mn_ldmatrix_x4_b(x_shared, n_group * 16, k_tile * 16, mma_lane, x_regs)
                            if IO_DTYPE == "float16":
                                if k_tile == 0:
                                    _t_mma_m16n8k16_f16_zero(n_acc, y_regs, x_regs, 0, 0)
                                    _t_mma_m16n8k16_f16_zero(n_acc, y_regs, x_regs, 4, 2)
                                else:
                                    _t_mma_m16n8k16_f16_acc(n_acc, y_regs, x_regs, 0, 0)
                                    _t_mma_m16n8k16_f16_acc(n_acc, y_regs, x_regs, 4, 2)
                            else:
                                if k_tile == 0:
                                    _t_mma_m16n8k16_bf16_zero(n_acc, y_regs, x_regs, 0, 0)
                                    _t_mma_m16n8k16_bf16_zero(n_acc, y_regs, x_regs, 4, 2)
                                else:
                                    _t_mma_m16n8k16_bf16_acc(n_acc, y_regs, x_regs, 0, 0)
                                    _t_mma_m16n8k16_bf16_acc(n_acc, y_regs, x_regs, 4, 2)
                        for element in T.unroll(8):
                            within_mma: T.int32 = element & 3
                            output_row: T.int32 = (
                                mma_warp * 16 + (mma_lane >> 2) + (within_mma >> 1) * 8
                            )
                            output_col: T.int32 = (
                                n_group * 16
                                + (element >> 2) * 8
                                + (mma_lane & 3) * 2
                                + (within_mma & 1)
                            )
                            n_first_shared[output_row * D_HEAD + output_col] = n_acc[element]
                T.cuda.cta_sync()

            if tid < 256:
                block_coeff: T.float32 = alpha_shared[T_BLOCK - 1]
                for output_token in T.serial(T_BLOCK):
                    operand_value: T.float32 = 0.0
                    if matrix_kind == 0 and block_local == 0:
                        if output_token < block_valid:
                            operand_value = T.cast(
                                k[
                                    (block_token_start + output_token) * K_HEADS * D_HEAD
                                    + k_head * D_HEAD
                                    + row
                                ],
                                "float32",
                            )
                    else:
                        for inner in T.serial(D_HEAD):
                            input_value: T.float32 = T.cast(
                                T.cast(values[inner], IO_DTYPE), "float32"
                            )
                            key_value: T.float32 = 0.0
                            if output_token < block_valid:
                                key_value = T.cast(
                                    k[
                                        (block_token_start + output_token) * K_HEADS * D_HEAD
                                        + k_head * D_HEAD
                                        + inner
                                    ],
                                    "float32",
                                )
                            operand_value = operand_value + input_value * key_value
                    if matrix_kind == 1:
                        v_term: T.float32 = 0.0
                        if output_token < block_valid:
                            v_term = (
                                T.cast(
                                    v[
                                        (block_token_start + output_token) * V_HEADS * D_HEAD
                                        + v_head * D_HEAD
                                        + row
                                    ],
                                    "float32",
                                )
                                * alpha_shared[T_BLOCK + output_token]
                            )
                        if block_local == 0:
                            operand_value = v_term
                        else:
                            operand_value = block_coeff * operand_value + v_term
                    block_operand[output_token] = T.cast(operand_value, IO_DTYPE)

                for col in T.serial(D_HEAD):
                    updated: T.float32 = values[col]
                    if matrix_kind == 1:
                        updated = block_coeff * updated
                    for output_token in T.serial(T_BLOCK):
                        updated = updated + T.cast(block_operand[output_token], "float32") * T.cast(
                            x_shared[_mn_matrix_index(col, output_token)], "float32"
                        )
                    if matrix_kind == 0:
                        updated = block_coeff * updated
                    values[col] = updated
                if matrix_kind == 1 and block_local == 0:
                    for col in T.serial(D_HEAD):
                        values[col] = n_first_shared[row * D_HEAD + col]
            T.cuda.cta_sync()

        if tid < 256:
            cp_slot: T.int32 = _device_chunk_bound(seq_idx, seq_start, CP_CHUNK_LEN) + chunk_in_seq
            base: T.int64 = (
                (T.cast(cp_slot, "int64") * STATE_HEADS + state_head) * D_HEAD + row
            ) * D_HEAD
            for col in T.serial(D_HEAD):
                if matrix_kind == 0:
                    transfer[base + col] = values[col]
                else:
                    local_state[base + col] = values[col]


@T.jit
def _fixup_sm100(
    transfer_h: T.handle,
    local_state_h: T.handle,
    initial_state_h: T.handle,
    initial_state_workspace_h: T.handle,
    fixed_state_h: T.handle,
    final_state_h: T.handle,
    state_indices_h: T.handle,
    cu_seqlens_h: T.handle,
    *,
    CU_DTYPE: T.constexpr,
    STATE_DTYPE: T.constexpr,
    NUM_SEQUENCES: T.constexpr,
    STATE_HEADS: T.constexpr,
    STATE_POOL: T.constexpr,
    TOTAL_CP_CHUNKS: T.constexpr,
    CP_CHUNK_LEN: T.constexpr,
    ROWS_PER_CTA: T.constexpr,
    NEEDS_INITIAL_STATE: T.constexpr,
    STORE_FINAL_STATE: T.constexpr,
    USE_STATE_INDICES: T.constexpr,
):
    """Source SIMT fixup: one thread owns one column and a small row group."""
    transfer = T.match_buffer(
        transfer_h, (TOTAL_CP_CHUNKS * STATE_HEADS * D_HEAD * D_HEAD,), "float32", scope="global"
    )
    local_state = T.match_buffer(
        local_state_h, (TOTAL_CP_CHUNKS * STATE_HEADS * D_HEAD * D_HEAD,), "float32", scope="global"
    )
    initial_state = T.match_buffer(
        initial_state_h, (STATE_POOL * STATE_HEADS * D_HEAD * D_HEAD,), STATE_DTYPE, scope="global"
    )
    initial_state_workspace = T.match_buffer(
        initial_state_workspace_h,
        (NUM_SEQUENCES * STATE_HEADS * D_HEAD * D_HEAD,),
        "float32",
        scope="global",
    )
    fixed_state = T.match_buffer(
        fixed_state_h, (TOTAL_CP_CHUNKS * STATE_HEADS * D_HEAD * D_HEAD,), "float32", scope="global"
    )
    final_state = T.match_buffer(
        final_state_h, (STATE_POOL * STATE_HEADS * D_HEAD * D_HEAD,), STATE_DTYPE, scope="global"
    )
    state_indices = T.match_buffer(state_indices_h, (NUM_SEQUENCES,), "int32", scope="global")
    cu_seqlens = T.match_buffer(cu_seqlens_h, (NUM_SEQUENCES + 1,), CU_DTYPE, scope="global")
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 2})
    # TIRX_TRANSCRIBE_START cp_delta_rule_fixup_sm100

    row_ctas = T.meta_var(D_HEAD // ROWS_PER_CTA)
    block = T.cta_id([NUM_SEQUENCES * STATE_HEADS * row_ctas])
    col = T.thread_id([T_THREADS])
    row_cta: T.int32 = block % row_ctas
    head_seq: T.int32 = block // row_ctas
    state_head: T.int32 = head_seq % STATE_HEADS
    seq_idx: T.int32 = head_seq // STATE_HEADS
    row_start: T.int32 = row_cta * ROWS_PER_CTA
    sequence_bounds = T.alloc_local((2,), "int32")
    _load_sequence_bounds(cu_seqlens, seq_idx, sequence_bounds, CU_DTYPE)
    seq_start: T.int32 = sequence_bounds[0]
    seq_end: T.int32 = sequence_bounds[1]
    seq_len: T.int32 = seq_end - seq_start
    num_chunks: T.int32 = (seq_len + CP_CHUNK_LEN - 1) // CP_CHUNK_LEN
    chunk_start: T.int32 = _device_chunk_bound(seq_idx, seq_start, CP_CHUNK_LEN)
    gap_start: T.int32 = chunk_start + num_chunks
    gap_end: T.int32 = TOTAL_CP_CHUNKS
    if seq_idx + 1 < NUM_SEQUENCES:
        gap_end = _device_chunk_bound(seq_idx + 1, seq_end, CP_CHUNK_LEN)
    state_slot: T.int32 = seq_idx
    if USE_STATE_INDICES:
        _load_global_s32(state_slot, state_indices.ptr_to([seq_idx]))

    shared_state = T.alloc_buffer(
        (ROWS_PER_CTA * D_HEAD,), "float32", scope="shared", align=128
    )
    T.ptx.setmaxnreg.inc.sync.aligned.u32(256)
    T.cuda.cta_sync()
    if num_chunks > 0:
        start: T.int32 = 0
        if NEEDS_INITIAL_STATE:
            initial_values = T.alloc_local((ROWS_PER_CTA,), "float32")
            for local_row in T.unroll(ROWS_PER_CTA):
                initial_index: T.int32 = (
                    ((state_slot * STATE_HEADS + state_head) * D_HEAD + row_start + local_row)
                    * D_HEAD
                    + col
                )
                _load_global_as_f32(
                    initial_values,
                    local_row,
                    initial_state,
                    initial_index,
                    STATE_DTYPE,
                )
                _store_shared_f32(
                    shared_state.ptr_to([local_row * D_HEAD + col]), initial_values[local_row]
                )
                workspace_index: T.int32 = (
                    ((seq_idx * STATE_HEADS + state_head) * D_HEAD + row_start + local_row)
                    * D_HEAD
                    + col
                )
                _store_global_f32(
                    initial_state_workspace.ptr_to([workspace_index]), initial_values[local_row]
                )
        else:
            start = 1
            first_slot: T.int32 = chunk_start
            for local_row in T.unroll(ROWS_PER_CTA):
                first_index: T.int32 = (
                    ((first_slot * STATE_HEADS + state_head) * D_HEAD + row_start + local_row)
                    * D_HEAD
                    + col
                )
                first_value: T.float32
                _load_global_f32(first_value, local_state.ptr_to([first_index]))
                _store_shared_f32(
                    shared_state.ptr_to([local_row * D_HEAD + col]), first_value
                )
                fixed_index: T.int32 = (
                    ((first_slot * STATE_HEADS + state_head) * D_HEAD + row_start + local_row)
                    * D_HEAD
                    + col
                )
                _store_global_f32(fixed_state.ptr_to([fixed_index]), first_value)
        T.cuda.cta_sync()

        accum = T.alloc_local((ROWS_PER_CTA,), "float32")
        accum_next = T.alloc_local((ROWS_PER_CTA,), "float32")
        m_values: T.float32[16]
        m_next: T.float32[16]
        if start < num_chunks:
            first_work_slot: T.int32 = chunk_start + start
            for local_row in T.unroll(ROWS_PER_CTA):
                first_work_index: T.int32 = (
                    ((first_work_slot * STATE_HEADS + state_head) * D_HEAD + row_start + local_row)
                    * D_HEAD
                    + col
                )
                _load_global_f32(accum[local_row], local_state.ptr_to([first_work_index]))
            for inner in T.unroll(16):
                transfer_index: T.int32 = (
                    ((first_work_slot * STATE_HEADS + state_head) * D_HEAD + inner) * D_HEAD + col
                )
                _load_global_f32(m_values[inner], transfer.ptr_to([transfer_index]))

        for chunk in T.serial(start, num_chunks):
            cp_slot: T.int32 = chunk_start + chunk
            next_chunk: T.int32 = chunk + 1
            for k_tile in T.unroll(7):
                for inner in T.unroll(16):
                    transfer_next_index: T.int32 = (
                        ((cp_slot * STATE_HEADS + state_head) * D_HEAD + (k_tile + 1) * 16 + inner)
                        * D_HEAD
                        + col
                    )
                    _load_global_f32(m_next[inner], transfer.ptr_to([transfer_next_index]))
                for local_row in T.unroll(ROWS_PER_CTA):
                    for inner in T.unroll(16):
                        shared_value_main: T.float32
                        _load_shared_f32(
                            shared_value_main,
                            shared_state.ptr_to(
                                [local_row * D_HEAD + k_tile * 16 + inner]
                            ),
                        )
                        T.ptx.fma.rn.f32(
                            accum[local_row],
                            shared_value_main,
                            m_values[inner],
                            accum[local_row],
                        )
                for inner in T.unroll(16):
                    m_values[inner] = m_next[inner]

            if next_chunk < num_chunks:
                next_slot: T.int32 = chunk_start + next_chunk
                for local_row in T.unroll(ROWS_PER_CTA):
                    next_state_index: T.int32 = (
                        ((next_slot * STATE_HEADS + state_head) * D_HEAD + row_start + local_row)
                        * D_HEAD
                        + col
                    )
                    _load_global_f32(
                        accum_next[local_row], local_state.ptr_to([next_state_index])
                    )
                for inner in T.unroll(16):
                    next_transfer_index: T.int32 = (
                        ((next_slot * STATE_HEADS + state_head) * D_HEAD + inner) * D_HEAD + col
                    )
                    _load_global_f32(m_next[inner], transfer.ptr_to([next_transfer_index]))
            for local_row in T.unroll(ROWS_PER_CTA):
                for inner in T.unroll(16):
                    shared_value_tail: T.float32
                    _load_shared_f32(
                        shared_value_tail,
                        shared_state.ptr_to([local_row * D_HEAD + 112 + inner]),
                    )
                    T.ptx.fma.rn.f32(
                        accum[local_row],
                        shared_value_tail,
                        m_values[inner],
                        accum[local_row],
                    )
            T.cuda.cta_sync()
            for local_row in T.unroll(ROWS_PER_CTA):
                _store_shared_f32(
                    shared_state.ptr_to([local_row * D_HEAD + col]), accum[local_row]
                )
                fixed_index: T.int32 = (
                    ((cp_slot * STATE_HEADS + state_head) * D_HEAD + row_start + local_row) * D_HEAD
                    + col
                )
                _store_global_f32(fixed_state.ptr_to([fixed_index]), accum[local_row])
            T.cuda.cta_sync()
            if next_chunk < num_chunks:
                for local_row in T.unroll(ROWS_PER_CTA):
                    accum[local_row] = accum_next[local_row]
                for inner in T.unroll(16):
                    m_values[inner] = m_next[inner]

        if STORE_FINAL_STATE:
            for local_row in T.unroll(ROWS_PER_CTA):
                final_index: T.int32 = (
                    ((state_slot * STATE_HEADS + state_head) * D_HEAD + row_start + local_row)
                    * D_HEAD
                    + col
                )
                final_value: T.float32
                _load_shared_f32(
                    final_value, shared_state.ptr_to([local_row * D_HEAD + col])
                )
                _store_global_from_f32(
                    final_state, final_index, final_value, STATE_DTYPE
                )

    for gap_slot in T.serial(gap_start, gap_end):
        for local_row in T.unroll(ROWS_PER_CTA):
            gap_index: T.int32 = (
                ((gap_slot * STATE_HEADS + state_head) * D_HEAD + row_start + local_row) * D_HEAD
                + col
            )
            _store_global_f32(fixed_state.ptr_to([gap_index]), T.float32(0.0))


@T.jit
def _fixup_utcmma_sm100(
    transfer_h: T.handle,
    local_state_h: T.handle,
    initial_state_h: T.handle,
    initial_state_workspace_h: T.handle,
    fixed_state_h: T.handle,
    final_state_h: T.handle,
    state_indices_h: T.handle,
    cu_seqlens_h: T.handle,
    transfer_map: T.TensorMap(),
    local_state_map: T.TensorMap(),
    *,
    CU_DTYPE: T.constexpr,
    STATE_DTYPE: T.constexpr,
    NUM_SEQUENCES: T.constexpr,
    STATE_HEADS: T.constexpr,
    STATE_POOL: T.constexpr,
    TOTAL_CP_CHUNKS: T.constexpr,
    CP_CHUNK_LEN: T.constexpr,
    ROWS: T.constexpr,
    M_STAGES: T.constexpr,
    COMPUTE_REGS: T.constexpr,
    SMEM_TOTAL: T.constexpr,
    TMEM_HOLDING_OFF: T.constexpr,
    M_OFF: T.constexpr,
    N_OFF: T.constexpr,
    M_FULL_OFF: T.constexpr,
    M_EMPTY_OFF: T.constexpr,
    N_FULL_OFF: T.constexpr,
    N_EMPTY_OFF: T.constexpr,
    READY_FULL_OFF: T.constexpr,
    READY_EMPTY_OFF: T.constexpr,
    DONE_FULL_OFF: T.constexpr,
    DONE_EMPTY_OFF: T.constexpr,
    NEEDS_INITIAL_STATE: T.constexpr,
    STORE_FINAL_STATE: T.constexpr,
    USE_STATE_INDICES: T.constexpr,
):
    """Source UTCMMA fixup with TMA M/N rings and an FP32 TMEM recurrence."""
    transfer = T.match_buffer(
        transfer_h, (TOTAL_CP_CHUNKS * STATE_HEADS * D_HEAD * D_HEAD,), "float32", scope="global"
    )
    local_state = T.match_buffer(
        local_state_h, (TOTAL_CP_CHUNKS * STATE_HEADS * D_HEAD * D_HEAD,), "float32", scope="global"
    )
    initial_state = T.match_buffer(
        initial_state_h, (STATE_POOL * STATE_HEADS * D_HEAD * D_HEAD,), STATE_DTYPE, scope="global"
    )
    initial_state_workspace = T.match_buffer(
        initial_state_workspace_h,
        (NUM_SEQUENCES * STATE_HEADS * D_HEAD * D_HEAD,),
        "float32",
        scope="global",
    )
    fixed_state = T.match_buffer(
        fixed_state_h, (TOTAL_CP_CHUNKS * STATE_HEADS * D_HEAD * D_HEAD,), "float32", scope="global"
    )
    final_state = T.match_buffer(
        final_state_h, (STATE_POOL * STATE_HEADS * D_HEAD * D_HEAD,), STATE_DTYPE, scope="global"
    )
    state_indices = T.match_buffer(state_indices_h, (NUM_SEQUENCES,), "int32", scope="global")
    cu_seqlens = T.match_buffer(cu_seqlens_h, (NUM_SEQUENCES + 1,), CU_DTYPE, scope="global")
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    # TIRX_TRANSCRIBE_START cp_delta_rule_fixup_utcmma_sm100

    row_ctas = T.meta_var(D_HEAD // ROWS)
    block = T.cta_id([NUM_SEQUENCES * STATE_HEADS * row_ctas])
    tid = T.thread_id([256])
    warp: T.int32 = T.cast(
        T.cuda._shfl_sync(T.uint32(0xFFFFFFFF), T.cast(tid >> 5, "uint32"), 0, 32), "int32"
    )
    lane: T.int32 = tid & 31
    row_cta: T.int32 = block % row_ctas
    head_seq: T.int32 = block // row_ctas
    state_head: T.int32 = head_seq % STATE_HEADS
    seq_idx: T.int32 = head_seq // STATE_HEADS
    row_start: T.int32 = row_cta * ROWS
    sequence_bounds = T.alloc_local((2,), "int32")
    _load_sequence_bounds(cu_seqlens, seq_idx, sequence_bounds, CU_DTYPE)
    seq_start: T.int32 = sequence_bounds[0]
    seq_end: T.int32 = sequence_bounds[1]
    seq_len: T.int32 = seq_end - seq_start
    num_chunks: T.int32 = (seq_len + CP_CHUNK_LEN - 1) // CP_CHUNK_LEN
    chunk_start: T.int32 = _device_chunk_bound(seq_idx, seq_start, CP_CHUNK_LEN)
    start: T.int32 = 0 if NEEDS_INITIAL_STATE else 1
    num_iters: T.int32 = num_chunks - start
    state_slot: T.int32 = seq_idx
    if USE_STATE_INDICES:
        _load_global_s32(state_slot, state_indices.ptr_to([seq_idx]))

    pool = T.SMEMPool()
    smem_raw = pool.alloc((SMEM_TOTAL,), "uint8", align=1024)
    tmem_holding = T.decl_buffer(
        (1,), "int32", data=smem_raw.data, scope="shared.dyn", byte_offset=TMEM_HOLDING_OFF, align=4
    )
    pool.commit()
    smem_addr: T.uint32 = T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0]))

    if warp == 0:
        if T.cuda.elect_sync():
            _mn_opt_init_pipeline(smem_raw, M_FULL_OFF, M_EMPTY_OFF, M_STAGES, 1, 1)
            _mn_opt_init_pipeline(smem_raw, N_FULL_OFF, N_EMPTY_OFF, 1, 1, 4)
            _mn_opt_init_pipeline(smem_raw, READY_FULL_OFF, READY_EMPTY_OFF, 1, 128, 1)
            _mn_opt_init_pipeline(smem_raw, DONE_FULL_OFF, DONE_EMPTY_OFF, 1, 1, 128)
            T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 5:
        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(transfer_map)))
        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(local_state_map)))
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
            T.address_of(tmem_holding[0]), T.uint32(_FIXUP_TMEM_COLUMNS)
        )

    tmem_base: T.int32 = 0
    if warp <= 4:
        T.ptx.bar.sync(T.uint32(_FIXUP_TMEM_ALLOC_BARRIER), T.uint32(160))
        T.ptx.ld.volatile.shared.s32(tmem_base, T.address_of(tmem_holding[0]))

    if num_chunks == 0:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(32)
        if warp == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                T.cast(tmem_base, "uint32"), T.uint32(_FIXUP_TMEM_COLUMNS)
            )
    elif warp <= 3:
        T.ptx.setmaxnreg.inc.sync.aligned.u32(COMPUTE_REGS)
        compute_thread: T.int32 = tid
        state_base: T.int64 = (
            T.cast(state_slot * STATE_HEADS + state_head, "int64") * D_HEAD * D_HEAD
            + row_start * D_HEAD
        )
        workspace_base: T.int64 = (
            T.cast(seq_idx * STATE_HEADS + state_head, "int64") * D_HEAD * D_HEAD
            + row_start * D_HEAD
        )
        fixed_base: T.int64 = (
            T.cast(chunk_start * STATE_HEADS + state_head, "int64") * D_HEAD * D_HEAD
            + row_start * D_HEAD
        )
        values = T.alloc_local((128,), "float32")
        words = values.view("uint32")
        n_count: T.int32 = 0
        ready_count: T.int32 = 0
        done_count: T.int32 = 0

        if NEEDS_INITIAL_STATE:
            _fixup_load_initial_to_tmem(
                tmem_base, initial_state, state_base, compute_thread, ROWS, STATE_DTYPE
            )
            T.ptx.tcgen05.wait__st.sync.aligned()
            _fixup_tmem_ld(tmem_base, _FIXUP_TMEM_ACC_COL, compute_thread, words, ROWS)
            T.ptx.tcgen05.wait__ld.sync.aligned()
            _fixup_store_f32(words, initial_state_workspace, workspace_base, compute_thread, ROWS)
        else:
            _mn_opt_consumer_wait(smem_addr, N_FULL_OFF, n_count, 1)
            _fixup_load_n_to_tmem(tmem_base, smem_raw, compute_thread, ROWS, N_OFF)
            T.ptx.tcgen05.wait__st.sync.aligned()
            _fixup_tmem_ld(tmem_base, _FIXUP_TMEM_ACC_COL, compute_thread, words, ROWS)
            T.ptx.tcgen05.wait__ld.sync.aligned()
            _fixup_store_f32(words, fixed_state, fixed_base, compute_thread, ROWS)
            if lane == 0:
                _mn_opt_release(smem_addr, N_EMPTY_OFF, n_count, 1)
            n_count = n_count + 1

        for chunk in T.serial(start, num_chunks):
            _fixup_acc_to_tf32(tmem_base, compute_thread, ROWS)
            _mn_opt_consumer_wait(smem_addr, N_FULL_OFF, n_count, 1)
            _mn_opt_producer_acquire(smem_addr, READY_EMPTY_OFF, ready_count, 1)
            _fixup_load_n_to_tmem(tmem_base, smem_raw, compute_thread, ROWS, N_OFF)
            T.ptx.tcgen05.wait__st.sync.aligned()
            _mn_opt_commit(smem_addr, READY_FULL_OFF, ready_count, 1)
            if lane == 0:
                _mn_opt_release(smem_addr, N_EMPTY_OFF, n_count, 1)
            n_count = n_count + 1
            ready_count = ready_count + 1

            _mn_opt_consumer_wait(smem_addr, DONE_FULL_OFF, done_count, 1)
            _fixup_tmem_ld(tmem_base, _FIXUP_TMEM_ACC_COL, compute_thread, words, ROWS)
            T.ptx.tcgen05.wait__ld.sync.aligned()
            chunk_base: T.int64 = (
                T.cast((chunk_start + chunk) * STATE_HEADS + state_head, "int64") * D_HEAD * D_HEAD
                + row_start * D_HEAD
            )
            _fixup_store_f32(words, fixed_state, chunk_base, compute_thread, ROWS)
            _mn_opt_release(smem_addr, DONE_EMPTY_OFF, done_count, 1)
            done_count = done_count + 1

        if STORE_FINAL_STATE:
            _fixup_tmem_ld(tmem_base, _FIXUP_TMEM_ACC_COL, compute_thread, words, ROWS)
            T.ptx.tcgen05.wait__ld.sync.aligned()
            last_fixed_base: T.int64 = (
                T.cast((chunk_start + num_chunks - 1) * STATE_HEADS + state_head, "int64")
                * D_HEAD
                * D_HEAD
                + row_start * D_HEAD
            )
            _fixup_store_f32(words, fixed_state, last_fixed_base, compute_thread, ROWS)
            _fixup_store_state(values, final_state, state_base, compute_thread, ROWS, STATE_DTYPE)

        if warp == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                T.cast(tmem_base, "uint32"), T.uint32(_FIXUP_TMEM_COLUMNS)
            )
    elif warp == 4:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(32)
        m_desc: T.uint64 = _fixup_smem_desc_m(smem_addr, M_OFF)
        for iteration in T.serial(num_iters):
            _mn_opt_consumer_wait(smem_addr, M_FULL_OFF, iteration, M_STAGES)
            _mn_opt_consumer_wait(smem_addr, READY_FULL_OFF, iteration, 1)
            _mn_opt_producer_acquire(smem_addr, DONE_EMPTY_OFF, iteration, 1)
            _fixup_mma(
                tmem_base,
                m_desc,
                _mn_opt_stage(iteration, M_STAGES),
                ROWS,
                _mn_opt_full_addr(smem_addr, DONE_FULL_OFF, iteration, 1),
                _mn_opt_empty_addr(smem_addr, READY_EMPTY_OFF, iteration, 1),
                _mn_opt_empty_addr(smem_addr, M_EMPTY_OFF, iteration, M_STAGES),
            )
    elif warp == 5:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(32)
        m_count: T.int32 = 0
        n_count_tma: T.int32 = 0
        for chunk in T.serial(num_chunks):
            cp_slot: T.int32 = chunk_start + chunk
            _mn_opt_producer_acquire(smem_addr, N_EMPTY_OFF, n_count_tma, 1)
            if T.cuda.elect_sync():
                n_full: T.uint32 = _mn_opt_full_addr(smem_addr, N_FULL_OFF, n_count_tma, 1)
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(n_full, T.uint32(ROWS * D_HEAD * 4))
                _fixup_tma_matrix(
                    smem_addr,
                    N_OFF,
                    T.address_of(local_state_map),
                    n_full,
                    row_start,
                    state_head,
                    cp_slot,
                    ROWS,
                    False,
                )
            n_count_tma = n_count_tma + 1
            if chunk >= start:
                _mn_opt_producer_acquire(smem_addr, M_EMPTY_OFF, m_count, M_STAGES)
                if T.cuda.elect_sync():
                    m_full: T.uint32 = _mn_opt_full_addr(smem_addr, M_FULL_OFF, m_count, M_STAGES)
                    T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        m_full, T.uint32(D_HEAD * D_HEAD * 4)
                    )
                    _fixup_tma_matrix(
                        smem_addr,
                        M_OFF + _mn_opt_stage(m_count, M_STAGES) * D_HEAD * D_HEAD * 4,
                        T.address_of(transfer_map),
                        m_full,
                        0,
                        state_head,
                        cp_slot,
                        ROWS,
                        True,
                    )
                m_count = m_count + 1
    else:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(32)


@T.jit
def _prefill_sm100(
    q_h: T.handle,
    k_h: T.handle,
    v_h: T.handle,
    alpha_h: T.handle,
    t_h: T.handle,
    fixed_state_h: T.handle,
    initial_state_workspace_h: T.handle,
    o_h: T.handle,
    cu_seqlens_h: T.handle,
    scale: T.float32,
    q_map: T.TensorMap(),
    k_map: T.TensorMap(),
    v_map: T.TensorMap(),
    t_map: T.TensorMap(),
    o_map: T.TensorMap(),
    descriptor_workspace_h: T.handle,
    *,
    IO_DTYPE: T.constexpr,
    CU_DTYPE: T.constexpr,
    TOTAL_TOKENS: T.constexpr,
    NUM_SEQUENCES: T.constexpr,
    Q_HEADS: T.constexpr,
    K_HEADS: T.constexpr,
    V_HEADS: T.constexpr,
    STATE_HEADS: T.constexpr,
    TOTAL_T_BLOCKS: T.constexpr,
    TOTAL_CP_CHUNKS: T.constexpr,
    MAX_CP_CHUNKS: T.constexpr,
    CP_CHUNK_LEN: T.constexpr,
    NEEDS_INITIAL_STATE: T.constexpr,
    IS_GQA: T.constexpr,
    HEAD_BASE: T.constexpr,
    HEAD_RATIO: T.constexpr,
):
    """Source-shaped SM100 CP prefill with TMA, TMEM, and tcgen05."""
    q = T.match_buffer(q_h, (TOTAL_TOKENS * Q_HEADS * D_HEAD,), IO_DTYPE, scope="global")
    k = T.match_buffer(k_h, (TOTAL_TOKENS * K_HEADS * D_HEAD,), IO_DTYPE, scope="global")
    v = T.match_buffer(v_h, (TOTAL_TOKENS * V_HEADS * D_HEAD,), IO_DTYPE, scope="global")
    alpha = T.match_buffer(alpha_h, (TOTAL_TOKENS * STATE_HEADS,), "float32", scope="global")
    t = T.match_buffer(
        t_h, (TOTAL_T_BLOCKS * STATE_HEADS * T_BLOCK * T_BLOCK,), IO_DTYPE, scope="global"
    )
    fixed_state = T.match_buffer(
        fixed_state_h, (TOTAL_CP_CHUNKS * STATE_HEADS * D_HEAD * D_HEAD,), "float32", scope="global"
    )
    initial_state_workspace = T.match_buffer(
        initial_state_workspace_h,
        (NUM_SEQUENCES * STATE_HEADS * D_HEAD * D_HEAD,),
        "float32",
        scope="global",
    )
    o = T.match_buffer(o_h, (TOTAL_TOKENS * STATE_HEADS * D_HEAD,), IO_DTYPE, scope="global")
    cu_seqlens = T.match_buffer(cu_seqlens_h, (NUM_SEQUENCES + 1,), CU_DTYPE, scope="global")
    descriptor_workspace = T.match_buffer(
        descriptor_workspace_h,
        (NUM_SEQUENCES * STATE_HEADS * MAX_CP_CHUNKS * DESCRIPTOR_SLOTS * DESCRIPTOR_SLOT_BYTES,),
        "int8",
        scope="global",
    )
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})

    bx, seq_idx = T.cta_id([STATE_HEADS * MAX_CP_CHUNKS, NUM_SEQUENCES])
    thread = T.thread_id([PREFILL_THREADS])
    warp = _pf_make_warp_uniform(T.cast(thread // 32, "uint32"))
    lane: T.int32 = thread & 31
    state_head: T.int32 = bx % STATE_HEADS
    chunk_in_seq: T.int32 = bx // STATE_HEADS
    sequence_bounds = T.alloc_local((2,), "int32")
    _load_sequence_bounds(cu_seqlens, seq_idx, sequence_bounds, CU_DTYPE)
    seq_start: T.int32 = sequence_bounds[0]
    seq_end: T.int32 = sequence_bounds[1]
    seq_len: T.int32 = seq_end - seq_start
    num_cp_chunks: T.int32 = (seq_len + CP_CHUNK_LEN - 1) // CP_CHUNK_LEN
    chunk_len: T.int32 = 0
    if chunk_in_seq < num_cp_chunks:
        chunk_len = T.min(CP_CHUNK_LEN, seq_len - chunk_in_seq * CP_CHUNK_LEN)
    chunk_start: T.int32 = seq_start + chunk_in_seq * CP_CHUNK_LEN
    chunk_end: T.int32 = chunk_start + chunk_len
    cp_slot_start: T.int32 = _device_chunk_bound(seq_idx, seq_start, CP_CHUNK_LEN)
    cp_slot: T.int32 = cp_slot_start + chunk_in_seq
    t_block_start: T.int32 = _device_chunk_bound(seq_idx, seq_start, T_BLOCK) + chunk_in_seq * (
        CP_CHUNK_LEN // T_BLOCK
    )
    num_valid_chunks: T.int32 = (chunk_len + T_BLOCK - 1) // T_BLOCK
    num_pairs: T.int32 = (chunk_len + 2 * T_BLOCK - 1) // (2 * T_BLOCK)
    padded_chunks: T.int32 = num_pairs * 2
    subhead: T.int32 = state_head % HEAD_RATIO
    base_head: T.int32 = state_head // HEAD_RATIO
    k_head: T.int32 = state_head * K_HEADS // STATE_HEADS

    pool = T.SMEMPool()
    smem_raw = pool.alloc((PREFILL_OPT_SMEM_TOTAL,), "uint8", align=1024)
    tmem_holding = T.decl_buffer(
        (1,),
        "int32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=PREFILL_OPT_TMEM_HOLDING_OFF,
        align=4,
    )
    s_cumsumlog = T.decl_buffer(
        (T_BLOCK * 5,),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=PREFILL_OPT_CUMSUMLOG_OFF,
        align=16,
    )
    s_cumprod = T.decl_buffer(
        (T_BLOCK * 5,),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=PREFILL_OPT_CUMPROD_OFF,
        align=16,
    )
    pool.commit()
    smem_addr: T.uint32 = T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0]))

    cta_linear: T.int64 = T.cast(seq_idx * (STATE_HEADS * MAX_CP_CHUNKS) + bx, "int64")
    descriptor_base: T.int64 = cta_linear * T.int64(DESCRIPTOR_SLOTS * DESCRIPTOR_SLOT_BYTES)
    descriptor_q = descriptor_workspace.ptr_to([descriptor_base])
    descriptor_k = descriptor_workspace.ptr_to([descriptor_base + 128])
    descriptor_v = descriptor_workspace.ptr_to([descriptor_base + 256])
    descriptor_o = descriptor_workspace.ptr_to([descriptor_base + 512])

    if warp == 0:
        if T.cuda.elect_sync():
            _prefill_opt_init_all_pipelines(smem_raw)
            T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 8:
        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(q_map)))
        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(k_map)))
        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(v_map)))
        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(t_map)))
        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(o_map)))

    # CG0: transform the precomputed T tile and materialize gated QK.
    if warp <= 3:
        T.ptx.setmaxnreg.inc.sync.aligned.u32(224)
        T.ptx.barrier.sync(T.uint32(PREFILL_OPT_TMEM_ALLOC_BARRIER), T.uint32(320))
        tmem_base_cg0: T.int32
        T.ptx.ld.volatile.shared.s32(tmem_base_cg0, T.address_of(tmem_holding[0]))
        gate_index_cg0: T.int32 = 0
        gate_phase_cg0: T.int32 = 0
        t_index_cg0: T.int32 = 0
        t_phase_cg0: T.int32 = 0
        acc_index_cg0: T.int32 = 0
        acc_phase_cg0: T.int32 = 0
        ainv_index_cg0: T.int32 = 0
        ainv_phase_cg0: T.int32 = 1
        qk_index_cg0: T.int32 = 0
        qk_phase_cg0: T.int32 = 1
        for chunk_cg0 in T.serial(padded_chunks):
            gate_count_cg0: T.int32 = gate_index_cg0
            gate_count_phase_cg0: T.int32 = gate_phase_cg0
            _pf_consumer_wait_state(smem_addr, 128, gate_count_cg0, gate_count_phase_cg0)
            gate_index_cg0 = _pf_pipe_next_index(gate_count_cg0, 5)
            gate_phase_cg0 = _pf_pipe_next_phase(gate_count_cg0, gate_count_phase_cg0, 5)
            t_count_cg0: T.int32 = t_index_cg0
            t_count_phase_cg0: T.int32 = t_phase_cg0
            _pf_consumer_wait_state(smem_addr, 208, t_count_cg0, t_count_phase_cg0)
            t_index_cg0 = _pf_pipe_next_index(t_count_cg0, 2)
            t_phase_cg0 = _pf_pipe_next_phase(t_count_cg0, t_count_phase_cg0, 2)

            ainv_count_cg0: T.int32 = ainv_index_cg0
            ainv_count_phase_cg0: T.int32 = ainv_phase_cg0
            _pf_producer_acquire_state(smem_addr, 344, ainv_count_cg0, ainv_count_phase_cg0)
            ainv_index_cg0 = _pf_pipe_next_index(ainv_count_cg0, 3)
            ainv_phase_cg0 = _pf_pipe_next_phase(ainv_count_cg0, ainv_count_phase_cg0, 3)
            _prefill_opt_transform_t(
                smem_raw,
                smem_addr,
                s_cumsumlog,
                t_count_cg0,
                ainv_count_cg0,
                gate_count_cg0,
                thread,
                chunk_cg0 >= num_valid_chunks - 1,
                chunk_len - chunk_cg0 * T_BLOCK,
                IO_DTYPE,
            )
            T.ptx.fence.proxy.async_.shared__cta()
            T.ptx.bar.sync(T.uint32(PREFILL_OPT_T_STORE_BARRIER), T.uint32(128))
            if lane == 0:
                _pf_consumer_release_state(smem_addr, 224, t_count_cg0)
            _pf_software_commit_state(smem_addr, 320, ainv_count_cg0)

            qk_count_cg0: T.int32 = qk_index_cg0
            qk_count_phase_cg0: T.int32 = qk_phase_cg0
            _pf_producer_acquire_state(smem_addr, 384, qk_count_cg0, qk_count_phase_cg0)
            qk_index_cg0 = _pf_pipe_next_index(qk_count_cg0, 2)
            qk_phase_cg0 = _pf_pipe_next_phase(qk_count_cg0, qk_count_phase_cg0, 2)
            acc_count_cg0: T.int32 = acc_index_cg0
            acc_count_phase_cg0: T.int32 = acc_phase_cg0
            _pf_consumer_wait_state(smem_addr, 272, acc_count_cg0, acc_count_phase_cg0)
            acc_index_cg0 = _pf_pipe_next_index(acc_count_cg0, 2)
            acc_phase_cg0 = _pf_pipe_next_phase(acc_count_cg0, acc_count_phase_cg0, 2)
            qk_values_cg0: T.float32[32]
            _prefill_opt_cg0_tmem_ld(tmem_base_cg0, acc_count_cg0, thread, qk_values_cg0)
            row_base_cg0: T.int32 = T.bitwise_or(
                T.bitwise_and(thread >> 2, T.int32(7)), T.bitwise_and(thread >> 1, T.int32(48))
            )
            col_base_cg0: T.int32 = T.bitwise_and(thread << 1, T.int32(6))
            for frag_cg0 in T.unroll(32):
                score_s_cg0: T.int32 = row_base_cg0 + T.bitwise_and(frag_cg0 >> 1, T.int32(1)) * 8
                score_t_cg0: T.int32 = (
                    col_base_cg0
                    + T.bitwise_and(frag_cg0, T.int32(1))
                    + T.bitwise_and(frag_cg0 >> 2, T.int32(1)) * 8
                    + (frag_cg0 >> 3) * 16
                )
                valid_cg0: T.bool = score_s_cg0 >= score_t_cg0
                if chunk_cg0 >= num_valid_chunks - 1:
                    valid_cg0 = (
                        valid_cg0
                        and score_s_cg0 < chunk_len - chunk_cg0 * T_BLOCK
                        and score_t_cg0 < chunk_len - chunk_cg0 * T_BLOCK
                    )
                gamma_cg0: T.float32 = _prefill_predicated_gamma(
                    _pf_shared_addr(
                        smem_addr,
                        PREFILL_OPT_CUMSUMLOG_OFF + (gate_count_cg0 * T_BLOCK + score_s_cg0) * 4,
                    ),
                    _pf_shared_addr(
                        smem_addr,
                        PREFILL_OPT_CUMSUMLOG_OFF + (gate_count_cg0 * T_BLOCK + score_t_cg0) * 4,
                    ),
                    valid_cg0,
                )
                qk_values_cg0[frag_cg0] = qk_values_cg0[frag_cg0] * gamma_cg0 * scale
            _prefill_opt_store_qk_fragment(smem_raw, qk_count_cg0, thread, qk_values_cg0, IO_DTYPE)
            T.ptx.fence.proxy.async_.shared__cta()
            T.ptx.tcgen05.wait__ld.sync.aligned()
            _pf_consumer_release_state(smem_addr, 288, acc_count_cg0)
            _pf_software_commit_state(smem_addr, 368, qk_count_cg0)
            _pf_consumer_release_state(smem_addr, 168, gate_count_cg0)
        _pf_producer_tail_state(smem_addr, 344, ainv_index_cg0, ainv_phase_cg0, 3)
        _pf_producer_tail_state(smem_addr, 384, qk_index_cg0, qk_phase_cg0, 2)

    # CG1: recurrent state, VKS/NV preparation, and O materialization.
    if warp >= 4 and warp <= 7:
        T.ptx.setmaxnreg.inc.sync.aligned.u32(256)
        if warp == 4:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_holding[0]), T.uint32(PREFILL_OPT_TMEM_COLUMNS)
            )
        T.ptx.barrier.sync(T.uint32(PREFILL_OPT_TMEM_ALLOC_BARRIER), T.uint32(320))
        tmem_base_cg1: T.int32
        T.ptx.ld.volatile.shared.s32(tmem_base_cg1, T.address_of(tmem_holding[0]))
        v_index_cg1: T.int32 = 0
        v_phase_cg1: T.int32 = 0
        gate_index_cg1: T.int32 = 0
        gate_phase_cg1: T.int32 = 0
        shared_consumer_cg1: T.int32 = 0
        kv_consumer_cg1: T.int32 = 0
        qstate_consumer_cg1: T.int32 = 0
        kv_producer_cg1: T.int32 = 0
        state_input_producer_cg1: T.int32 = 0
        vks_producer_cg1: T.int32 = 0
        nv_producer_cg1: T.int32 = 0
        decay_producer_cg1: T.int32 = 0
        o_index_cg1: T.int32 = 0
        o_phase_cg1: T.int32 = 1

        if chunk_len > 0:
            # CP always publishes an explicit generation-zero state.  A first
            # chunk receives the user state (already converted by fixup) or
            # zero; later chunks receive the fixed previous boundary.
            initial_count_cg1: T.int32 = kv_producer_cg1
            _pf_producer_acquire(smem_addr, 264, initial_count_cg1, 1)
            cg1_thread: T.int32 = thread & 127
            state_words_cg1: T.uint32[32]
            previous_slot_cg1: T.int32 = cp_slot - 1
            for state_sub_cg1 in range(4):
                for vector_cg1 in range(8):
                    word_offset_cg1: T.int32 = state_sub_cg1 * 32 + vector_cg1 * 4
                    if chunk_in_seq > 0:
                        fixed_base_cg1: T.int64 = (
                            T.cast(previous_slot_cg1 * STATE_HEADS + state_head, "int64") * D_HEAD
                            + T.cast(cg1_thread, "int64")
                        ) * D_HEAD + word_offset_cg1
                        T.ptx["ld.global.L1::no_allocate.v4.b32"](
                            state_words_cg1[vector_cg1 * 4],
                            state_words_cg1[vector_cg1 * 4 + 1],
                            state_words_cg1[vector_cg1 * 4 + 2],
                            state_words_cg1[vector_cg1 * 4 + 3],
                            fixed_state.ptr_to([fixed_base_cg1]),
                        )
                    elif NEEDS_INITIAL_STATE:
                        initial_base_cg1: T.int64 = (
                            T.cast(seq_idx * STATE_HEADS + state_head, "int64") * D_HEAD
                            + T.cast(cg1_thread, "int64")
                        ) * D_HEAD + word_offset_cg1
                        T.ptx["ld.global.L1::no_allocate.v4.b32"](
                            state_words_cg1[vector_cg1 * 4],
                            state_words_cg1[vector_cg1 * 4 + 1],
                            state_words_cg1[vector_cg1 * 4 + 2],
                            state_words_cg1[vector_cg1 * 4 + 3],
                            initial_state_workspace.ptr_to([initial_base_cg1]),
                        )
                    else:
                        state_words_cg1[vector_cg1 * 4] = T.uint32(0)
                        state_words_cg1[vector_cg1 * 4 + 1] = T.uint32(0)
                        state_words_cg1[vector_cg1 * 4 + 2] = T.uint32(0)
                        state_words_cg1[vector_cg1 * 4 + 3] = T.uint32(0)
                _pf_state_tmem_st_sub(tmem_base_cg1, thread, state_sub_cg1, state_words_cg1, 0)
            T.ptx.tcgen05.wait__st.sync.aligned()
            T.ptx.bar.sync(T.uint32(PREFILL_OPT_INITIAL_STATE_BARRIER), T.uint32(128))
            if cg1_thread == 0:
                _pf_software_commit(smem_addr, 256, initial_count_cg1, 1)
            kv_producer_cg1 = kv_producer_cg1 + 1

            for chunk_cg1 in T.serial(padded_chunks):
                if (chunk_cg1 & 1) == 0:
                    kv_producer_cg1 = kv_producer_cg1 + 1
                    kv_producer_cg1 = kv_producer_cg1 + 1

                gate_count_cg1: T.int32 = gate_index_cg1
                gate_count_phase_cg1: T.int32 = gate_phase_cg1
                _pf_consumer_wait_state(smem_addr, 128, gate_count_cg1, gate_count_phase_cg1)
                gate_index_cg1 = _pf_pipe_next_index(gate_count_cg1, 5)
                gate_phase_cg1 = _pf_pipe_next_phase(gate_count_cg1, gate_count_phase_cg1, 5)
                cumprod_total_cg1: T.float32
                T.ptx.ld.shared.f32(
                    cumprod_total_cg1, s_cumprod.ptr_to([gate_count_cg1 * T_BLOCK + T_BLOCK - 1])
                )

                kv_previous_cg1: T.int32 = kv_consumer_cg1
                _pf_consumer_wait(smem_addr, 256, kv_previous_cg1, 1)
                kv_consumer_cg1 = kv_consumer_cg1 + 1
                state_input_count_cg1: T.int32 = state_input_producer_cg1
                _pf_producer_acquire(smem_addr, 408, state_input_count_cg1, 1)
                state_input_producer_cg1 = state_input_producer_cg1 + 1
                state_values_cg1: T.float32[128]
                state_input_words_cg1: T.uint32[64]
                for state_sub_cg1 in range(4):
                    _pf_state_tmem_ld_sub(
                        tmem_base_cg1, thread, state_sub_cg1, state_values_cg1, state_sub_cg1 * 32
                    )
                for state_pair_cg1 in T.unroll(64):
                    state_input_words_cg1[state_pair_cg1] = _mn_opt_pack_iox2(
                        state_values_cg1[state_pair_cg1 * 2],
                        state_values_cg1[state_pair_cg1 * 2 + 1],
                        IO_DTYPE,
                    )
                for state_sub_cg1 in range(4):
                    _pf_state_input_tmem_st_sub(
                        tmem_base_cg1, thread, state_sub_cg1, state_input_words_cg1
                    )
                T.ptx.tcgen05.wait__st.sync.aligned()
                _pf_software_commit(smem_addr, 400, state_input_count_cg1, 1)

                for state_pair_cg1 in T.unroll(64):
                    state_mul_cg1: T.uint64
                    T.ptx.mul.rn.f32x2(
                        state_mul_cg1,
                        T.cuda.make_float2(
                            state_values_cg1[state_pair_cg1 * 2],
                            state_values_cg1[state_pair_cg1 * 2 + 1],
                        ),
                        T.cuda.make_float2(cumprod_total_cg1, cumprod_total_cg1),
                    )
                    state_values_cg1[state_pair_cg1 * 2] = T.cuda.float2_x(state_mul_cg1)
                    state_values_cg1[state_pair_cg1 * 2 + 1] = T.cuda.float2_y(state_mul_cg1)
                for state_sub_cg1 in range(4):
                    _pf_state_tmem_st_sub(
                        tmem_base_cg1, thread, state_sub_cg1, state_values_cg1, state_sub_cg1 * 32
                    )
                T.ptx.tcgen05.wait__st.sync.aligned()
                _pf_consumer_release(smem_addr, 264, kv_previous_cg1, 1)

                cumprod_factor_cg1: T.float32[16]
                decay_factor_cg1: T.float32[16]
                factor_col_base_cg1: T.int32 = T.bitwise_and(thread << 1, T.int32(6))
                last_log_cg1: T.float32
                T.ptx.ld.shared.f32(
                    last_log_cg1, s_cumsumlog.ptr_to([gate_count_cg1 * T_BLOCK + T_BLOCK - 1])
                )
                for factor_group_cg1 in T.unroll(8):
                    factor_col_cg1: T.int32 = factor_col_base_cg1 + factor_group_cg1 * 8
                    T.ptx.ld.shared.v2.f32(
                        cumprod_factor_cg1[factor_group_cg1 * 2],
                        cumprod_factor_cg1[factor_group_cg1 * 2 + 1],
                        s_cumprod.ptr_to([gate_count_cg1 * T_BLOCK + factor_col_cg1]),
                    )
                    log_pair_cg1: T.float32[2]
                    T.ptx.ld.shared.v2.f32(
                        log_pair_cg1[0],
                        log_pair_cg1[1],
                        s_cumsumlog.ptr_to([gate_count_cg1 * T_BLOCK + factor_col_cg1]),
                    )
                    neg_log_pair_cg1: T.float32[2]
                    T.ptx.neg.f32(neg_log_pair_cg1[0], log_pair_cg1[0])
                    T.ptx.neg.f32(neg_log_pair_cg1[1], log_pair_cg1[1])
                    decay_diff_cg1: T.uint64
                    T.ptx.add.rn.f32x2(
                        decay_diff_cg1,
                        T.cuda.make_float2(last_log_cg1, last_log_cg1),
                        T.cuda.make_float2(neg_log_pair_cg1[0], neg_log_pair_cg1[1]),
                    )
                    T.ptx.ex2.approx.ftz.f32(
                        decay_factor_cg1[factor_group_cg1 * 2], T.cuda.float2_x(decay_diff_cg1)
                    )
                    T.ptx.ex2.approx.ftz.f32(
                        decay_factor_cg1[factor_group_cg1 * 2 + 1], T.cuda.float2_y(decay_diff_cg1)
                    )
                _pf_consumer_release_state(smem_addr, 168, gate_count_cg1)

                vks_count_cg1: T.int32 = vks_producer_cg1
                vks_producer_cg1 = vks_producer_cg1 + 1
                v_count_cg1: T.int32 = v_index_cg1
                v_count_phase_cg1: T.int32 = v_phase_cg1
                _pf_consumer_wait_state(smem_addr, 80, v_count_cg1, v_count_phase_cg1)
                v_index_cg1 = _pf_pipe_next_index(v_count_cg1, 3)
                v_phase_cg1 = _pf_pipe_next_phase(v_count_cg1, v_count_phase_cg1, 3)
                v_words_cg1: T.uint32[32]
                _prefill_opt_load_v_fragment(smem_raw, v_count_cg1, thread, v_words_cg1)

                ks_count_cg1: T.int32 = shared_consumer_cg1
                _pf_consumer_wait(smem_addr, 304, ks_count_cg1, 1)
                shared_consumer_cg1 = shared_consumer_cg1 + 1
                fragment_cg1: T.float32[64]
                _pf_cg1_tmem_ld_f32(
                    tmem_base_cg1, PREFILL_OPT_TMEM_CG1_ACC_COL, thread, fragment_cg1
                )
                for row_half_cg1 in T.unroll(2):
                    for factor_group_cg1 in T.unroll(8):
                        for factor_repeat_cg1 in T.unroll(2):
                            pair_cg1: T.int32 = (
                                row_half_cg1 * 16 + factor_group_cg1 * 2 + factor_repeat_cg1
                            )
                            ks_mul_cg1: T.uint64
                            T.ptx.mul.rn.f32x2(
                                ks_mul_cg1,
                                T.cuda.make_float2(
                                    fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1]
                                ),
                                T.cuda.make_float2(
                                    cumprod_factor_cg1[factor_group_cg1 * 2],
                                    cumprod_factor_cg1[factor_group_cg1 * 2 + 1],
                                ),
                            )
                            fragment_cg1[pair_cg1 * 2] = T.cuda.float2_x(ks_mul_cg1)
                            fragment_cg1[pair_cg1 * 2 + 1] = T.cuda.float2_y(ks_mul_cg1)
                _pf_consumer_release(smem_addr, 312, ks_count_cg1, 1)
                for pair_cg1 in T.unroll(32):
                    ks_word_cg1: T.uint32 = _mn_opt_pack_iox2(
                        fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1], IO_DTYPE
                    )
                    v_words_cg1[pair_cg1] = _prefill_opt_sub_iox2(
                        v_words_cg1[pair_cg1], ks_word_cg1, IO_DTYPE
                    )
                _pf_cg1_tmem_st_io(
                    tmem_base_cg1, PREFILL_OPT_TMEM_SHARED_INPUT_COL, thread, v_words_cg1
                )
                T.ptx.tcgen05.wait__st.sync.aligned()
                _pf_software_commit(smem_addr, 416, vks_count_cg1, 1)

                qs_count_cg1: T.int32 = qstate_consumer_cg1
                _pf_consumer_wait(smem_addr, 240, qs_count_cg1, 1)
                qstate_consumer_cg1 = qstate_consumer_cg1 + 1
                _pf_cg1_tmem_ld_f32(
                    tmem_base_cg1, PREFILL_OPT_TMEM_Q_STATE_COL, thread, fragment_cg1
                )
                for row_half_cg1 in T.unroll(2):
                    for factor_group_cg1 in T.unroll(8):
                        for factor_repeat_cg1 in T.unroll(2):
                            pair_cg1: T.int32 = (
                                row_half_cg1 * 16 + factor_group_cg1 * 2 + factor_repeat_cg1
                            )
                            qs_mul_cg1: T.uint64
                            T.ptx.mul.rn.f32x2(
                                qs_mul_cg1,
                                T.cuda.make_float2(
                                    fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1]
                                ),
                                T.cuda.make_float2(
                                    cumprod_factor_cg1[factor_group_cg1 * 2],
                                    cumprod_factor_cg1[factor_group_cg1 * 2 + 1],
                                ),
                            )
                            T.ptx.mul.rn.f32x2(
                                qs_mul_cg1, qs_mul_cg1, T.cuda.make_float2(scale, scale)
                            )
                            fragment_cg1[pair_cg1 * 2] = T.cuda.float2_x(qs_mul_cg1)
                            fragment_cg1[pair_cg1 * 2 + 1] = T.cuda.float2_y(qs_mul_cg1)
                _pf_cg1_tmem_st_f32(
                    tmem_base_cg1, PREFILL_OPT_TMEM_Q_STATE_COL, thread, fragment_cg1
                )
                T.ptx.tcgen05.wait__st.sync.aligned()
                _pf_consumer_release(smem_addr, 248, qs_count_cg1, 1)

                nv_acc_count_cg1: T.int32 = shared_consumer_cg1
                _pf_consumer_wait(smem_addr, 304, nv_acc_count_cg1, 1)
                shared_consumer_cg1 = shared_consumer_cg1 + 1
                if lane == 0:
                    _pf_consumer_release_state(smem_addr, 104, v_count_cg1)
                _pf_cg1_tmem_ld_f32(
                    tmem_base_cg1, PREFILL_OPT_TMEM_CG1_ACC_COL, thread, fragment_cg1
                )
                nv_words_cg1: T.uint32[32]
                for pair_cg1 in T.unroll(32):
                    nv_words_cg1[pair_cg1] = _mn_opt_pack_iox2(
                        fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1], IO_DTYPE
                    )
                _pf_consumer_release(smem_addr, 312, nv_acc_count_cg1, 1)

                for row_half_cg1 in T.unroll(2):
                    for factor_group_cg1 in T.unroll(8):
                        for factor_repeat_cg1 in T.unroll(2):
                            pair_cg1: T.int32 = (
                                row_half_cg1 * 16 + factor_group_cg1 * 2 + factor_repeat_cg1
                            )
                            decay_mul_cg1: T.uint64
                            T.ptx.mul.rn.f32x2(
                                decay_mul_cg1,
                                T.cuda.make_float2(
                                    fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1]
                                ),
                                T.cuda.make_float2(
                                    decay_factor_cg1[factor_group_cg1 * 2],
                                    decay_factor_cg1[factor_group_cg1 * 2 + 1],
                                ),
                            )
                            fragment_cg1[pair_cg1 * 2] = T.cuda.float2_x(decay_mul_cg1)
                            fragment_cg1[pair_cg1 * 2 + 1] = T.cuda.float2_y(decay_mul_cg1)

                nv_count_cg1: T.int32 = nv_producer_cg1
                nv_producer_cg1 = nv_producer_cg1 + 1
                decay_count_cg1: T.int32 = decay_producer_cg1
                decay_producer_cg1 = decay_producer_cg1 + 1
                decay_words_cg1: T.uint32[32]
                for row_half_cg1 in range(2):
                    _pf_cg1_tmem_st_io_half(
                        tmem_base_cg1,
                        PREFILL_OPT_TMEM_SHARED_INPUT_COL,
                        thread,
                        row_half_cg1,
                        nv_words_cg1,
                    )
                    for pair_in_half_cg1 in T.unroll(16):
                        pair_cg1: T.int32 = row_half_cg1 * 16 + pair_in_half_cg1
                        decay_words_cg1[pair_cg1] = _mn_opt_pack_iox2(
                            fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1], IO_DTYPE
                        )
                    _pf_cg1_tmem_st_io_half(
                        tmem_base_cg1,
                        PREFILL_OPT_TMEM_SHARED_INPUT_COL + 32,
                        thread,
                        row_half_cg1,
                        decay_words_cg1,
                    )
                T.ptx.tcgen05.wait__st.sync.aligned()
                _pf_software_commit(smem_addr, 432, nv_count_cg1, 1)
                _pf_software_commit(smem_addr, 448, decay_count_cg1, 1)

                o_count_cg1: T.int32 = o_index_cg1
                o_count_phase_cg1: T.int32 = o_phase_cg1
                _pf_producer_acquire_state(smem_addr, 480, o_count_cg1, o_count_phase_cg1)
                o_index_cg1 = _pf_pipe_next_index(o_count_cg1, 2)
                o_phase_cg1 = _pf_pipe_next_phase(o_count_cg1, o_count_phase_cg1, 2)
                qkv_count_cg1: T.int32 = qstate_consumer_cg1
                _pf_consumer_wait(smem_addr, 240, qkv_count_cg1, 1)
                qstate_consumer_cg1 = qstate_consumer_cg1 + 1
                _pf_cg1_tmem_ld_f32(
                    tmem_base_cg1, PREFILL_OPT_TMEM_Q_STATE_COL, thread, fragment_cg1
                )
                o_words_cg1: T.uint32[32]
                for pair_cg1 in T.unroll(32):
                    o_words_cg1[pair_cg1] = _mn_opt_pack_iox2(
                        fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1], IO_DTYPE
                    )
                _prefill_opt_store_o_fragment(smem_raw, o_count_cg1, thread, o_words_cg1)
                T.ptx.fence.proxy.async_.shared__cta()
                _pf_consumer_release(smem_addr, 248, qkv_count_cg1, 1)
                _pf_software_commit_state(smem_addr, 464, o_count_cg1)

            final_kv_count_cg1: T.int32 = kv_consumer_cg1
            _pf_consumer_wait(smem_addr, 256, final_kv_count_cg1, 1)
            kv_consumer_cg1 = kv_consumer_cg1 + 1
            final_state_drain_cg1: T.float32[128]
            for state_sub_cg1 in range(4):
                _pf_state_tmem_ld_sub(
                    tmem_base_cg1, thread, state_sub_cg1, final_state_drain_cg1, state_sub_cg1 * 32
                )
            _pf_consumer_release(smem_addr, 264, final_kv_count_cg1, 1)

        T.cuda.warpgroup_sync(PREFILL_OPT_TMEM_DEALLOC_BARRIER)
        if warp == 4:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                T.cast(tmem_base_cg1, "uint32"), T.uint32(PREFILL_OPT_TMEM_COLUMNS)
            )
        _pf_producer_tail_state(smem_addr, 480, o_index_cg1, o_phase_cg1, 2)
        _pf_producer_tail(smem_addr, 408, state_input_producer_cg1, 1)

    # Issuer and load roles are mutually exclusive with CG1, matching source.
    elif warp == 8:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
        T.ptx.barrier.sync(T.uint32(PREFILL_OPT_TMEM_ALLOC_BARRIER), T.uint32(320))
        tmem_base_i0: T.int32
        T.ptx.ld.volatile.shared.s32(tmem_base_i0, T.address_of(tmem_holding[0]))
        acc_index_i0: T.int32 = 0
        acc_phase_i0: T.int32 = 1
        k_index_i0: T.int32 = 0
        k_phase_i0: T.int32 = 0
        q_index_i0: T.int32 = 0
        q_phase_i0: T.int32 = 0
        for chunk_i0 in T.serial(padded_chunks):
            acc_count_i0: T.int32 = acc_index_i0
            acc_count_phase_i0: T.int32 = acc_phase_i0
            _pf_producer_acquire_state(smem_addr, 288, acc_count_i0, acc_count_phase_i0)
            acc_index_i0 = _pf_pipe_next_index(acc_count_i0, 2)
            acc_phase_i0 = _pf_pipe_next_phase(acc_count_i0, acc_count_phase_i0, 2)
            k_count_i0: T.int32 = k_index_i0
            k_count_phase_i0: T.int32 = k_phase_i0
            _pf_consumer_wait_state(smem_addr, 0, k_count_i0, k_count_phase_i0)
            k_index_i0 = _pf_pipe_next_index(k_count_i0, 3)
            k_phase_i0 = _pf_pipe_next_phase(k_count_i0, k_count_phase_i0, 3)
            q_count_i0: T.int32 = q_index_i0
            q_count_phase_i0: T.int32 = q_phase_i0
            _pf_consumer_wait_state(smem_addr, 48, q_count_i0, q_count_phase_i0)
            q_index_i0 = _pf_pipe_next_index(q_count_i0, 2)
            q_phase_i0 = _pf_pipe_next_phase(q_count_i0, q_count_phase_i0, 2)
            k_desc_i0: T.uint64 = _pf_smem_desc_b128(
                _pf_shared_addr(smem_addr, PREFILL_OPT_K_OFF + k_count_i0 * 16384)
            )
            q_desc_i0: T.uint64 = _pf_smem_desc_b128(
                _pf_shared_addr(smem_addr, PREFILL_OPT_Q_OFF + q_count_i0 * 16384)
            )
            _prefill_opt_mma_ss_64x64_k128(
                tmem_base_i0 + PREFILL_OPT_TMEM_CG0_ACC_COL + acc_count_i0 * 64,
                q_desc_i0,
                k_desc_i0,
                _pf_shared_addr(smem_addr, 272 + acc_count_i0 * 8),
                IO_DTYPE,
            )
            _prefill_opt_mma_commit(_pf_shared_addr(smem_addr, 64 + q_count_i0 * 8))
            _prefill_opt_mma_commit(_pf_shared_addr(smem_addr, 24 + k_count_i0 * 8))
        _pf_producer_tail_state(smem_addr, 288, acc_index_i0, acc_phase_i0, 2)

    elif warp == 10:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
        T.ptx.barrier.sync(T.uint32(PREFILL_OPT_TMEM_ALLOC_BARRIER), T.uint32(320))
        tmem_base_i1: T.int32
        T.ptx.ld.volatile.shared.s32(tmem_base_i1, T.address_of(tmem_holding[0]))
        cg1_producer_i1: T.int32 = 0
        qstate_producer_i1: T.int32 = 0
        kv_producer_i1: T.int32 = 0
        k_index_i1: T.int32 = 0
        k_phase_i1: T.int32 = 0
        q_index_i1: T.int32 = 0
        q_phase_i1: T.int32 = 0
        ainv_index_i1: T.int32 = 0
        ainv_phase_i1: T.int32 = 0
        qk_index_i1: T.int32 = 0
        qk_phase_i1: T.int32 = 0
        state_input_consumer_i1: T.int32 = 0
        vks_consumer_i1: T.int32 = 0
        nv_consumer_i1: T.int32 = 0
        decay_consumer_i1: T.int32 = 0
        for chunk_i1 in T.serial(padded_chunks):
            k_count_i1: T.int32 = k_index_i1
            k_count_phase_i1: T.int32 = k_phase_i1
            _pf_consumer_wait_state(smem_addr, 0, k_count_i1, k_count_phase_i1)
            q_count_i1: T.int32 = q_index_i1
            q_count_phase_i1: T.int32 = q_phase_i1
            _pf_consumer_wait_state(smem_addr, 48, q_count_i1, q_count_phase_i1)
            k_index_i1 = _pf_pipe_next_index(k_count_i1, 3)
            k_phase_i1 = _pf_pipe_next_phase(k_count_i1, k_count_phase_i1, 3)
            q_index_i1 = _pf_pipe_next_index(q_count_i1, 2)
            q_phase_i1 = _pf_pipe_next_phase(q_count_i1, q_count_phase_i1, 2)
            k_desc_i1: T.uint64 = _pf_smem_desc_b128(
                _pf_shared_addr(smem_addr, PREFILL_OPT_K_OFF + k_count_i1 * 16384)
            )
            q_desc_i1: T.uint64 = _pf_smem_desc_b128(
                _pf_shared_addr(smem_addr, PREFILL_OPT_Q_OFF + q_count_i1 * 16384)
            )

            ks_count_i1: T.int32 = cg1_producer_i1
            _pf_producer_acquire(smem_addr, 312, ks_count_i1, 1)
            state_input_count_i1: T.int32 = state_input_consumer_i1
            _pf_consumer_wait(smem_addr, 400, state_input_count_i1, 1)
            _prefill_opt_mma_ts_128x64_k128(
                tmem_base_i1 + PREFILL_OPT_TMEM_CG1_ACC_COL,
                tmem_base_i1 + PREFILL_OPT_TMEM_STATE_INPUT_COL,
                k_desc_i1,
                _pf_pipe_full_addr(smem_addr, 304, ks_count_i1, 1),
                IO_DTYPE,
            )
            cg1_producer_i1 = cg1_producer_i1 + 1
            state_input_consumer_i1 = state_input_consumer_i1 + 1

            qs_count_i1: T.int32 = qstate_producer_i1
            _pf_producer_acquire(smem_addr, 248, qs_count_i1, 1)
            _prefill_opt_mma_ts_128x64_k128(
                tmem_base_i1 + PREFILL_OPT_TMEM_Q_STATE_COL,
                tmem_base_i1 + PREFILL_OPT_TMEM_STATE_INPUT_COL,
                q_desc_i1,
                _pf_pipe_full_addr(smem_addr, 240, qs_count_i1, 1),
                IO_DTYPE,
            )
            qstate_producer_i1 = qstate_producer_i1 + 1
            _prefill_opt_mma_commit(_pf_pipe_empty_addr(smem_addr, 408, state_input_count_i1, 1))
            _prefill_opt_mma_commit(_pf_shared_addr(smem_addr, 64 + q_count_i1 * 8))

            nv_acc_count_i1: T.int32 = cg1_producer_i1
            _pf_producer_acquire(smem_addr, 312, nv_acc_count_i1, 1)
            _pf_consumer_wait(smem_addr, 416, vks_consumer_i1, 1)
            vks_consumer_i1 = vks_consumer_i1 + 1
            ainv_count_i1: T.int32 = ainv_index_i1
            ainv_count_phase_i1: T.int32 = ainv_phase_i1
            _pf_consumer_wait_state(smem_addr, 320, ainv_count_i1, ainv_count_phase_i1)
            ainv_desc_i1: T.uint64 = _pf_smem_desc_b128(
                _pf_shared_addr(smem_addr, PREFILL_OPT_AINV_OFF + ainv_count_i1 * 8192)
            )
            _prefill_opt_mma_ts_128x64_k64(
                tmem_base_i1 + PREFILL_OPT_TMEM_CG1_ACC_COL,
                tmem_base_i1 + PREFILL_OPT_TMEM_SHARED_INPUT_COL,
                ainv_desc_i1,
                0,
                IO_DTYPE,
            )
            _prefill_opt_mma_commit(_pf_pipe_full_addr(smem_addr, 304, nv_acc_count_i1, 1))
            cg1_producer_i1 = cg1_producer_i1 + 1
            ainv_index_i1 = _pf_pipe_next_index(ainv_count_i1, 3)
            ainv_phase_i1 = _pf_pipe_next_phase(ainv_count_i1, ainv_count_phase_i1, 3)
            _prefill_opt_mma_commit(_pf_shared_addr(smem_addr, 344 + ainv_count_i1 * 8))

            qkv_count_i1: T.int32 = qstate_producer_i1
            _pf_producer_acquire(smem_addr, 248, qkv_count_i1, 1)
            qk_count_i1: T.int32 = qk_index_i1
            qk_count_phase_i1: T.int32 = qk_phase_i1
            _pf_consumer_wait_state(smem_addr, 368, qk_count_i1, qk_count_phase_i1)
            qk_desc_i1: T.uint64 = _pf_smem_desc_b128(
                _pf_shared_addr(smem_addr, PREFILL_OPT_QK_OFF + qk_count_i1 * 8192)
            )
            _pf_consumer_wait(smem_addr, 432, nv_consumer_i1, 1)
            nv_consumer_i1 = nv_consumer_i1 + 1
            _prefill_opt_mma_ts_128x64_k64(
                tmem_base_i1 + PREFILL_OPT_TMEM_Q_STATE_COL,
                tmem_base_i1 + PREFILL_OPT_TMEM_SHARED_INPUT_COL,
                qk_desc_i1,
                1,
                IO_DTYPE,
            )
            qk_index_i1 = _pf_pipe_next_index(qk_count_i1, 2)
            qk_phase_i1 = _pf_pipe_next_phase(qk_count_i1, qk_count_phase_i1, 2)
            _prefill_opt_mma_commit(_pf_shared_addr(smem_addr, 384 + qk_count_i1 * 8))
            _prefill_opt_mma_commit(_pf_pipe_full_addr(smem_addr, 240, qkv_count_i1, 1))
            qstate_producer_i1 = qstate_producer_i1 + 1

            if chunk_i1 == 0:
                kv_producer_i1 = kv_producer_i1 + 1
            kv_count_i1: T.int32 = kv_producer_i1
            _pf_producer_acquire(smem_addr, 264, kv_count_i1, 1)
            _pf_consumer_wait(smem_addr, 448, decay_consumer_i1, 1)
            decay_consumer_i1 = decay_consumer_i1 + 1
            kt_desc_i1: T.uint64 = _mn_opt_smem_desc_mn(
                _pf_shared_addr(smem_addr, PREFILL_OPT_K_OFF + k_count_i1 * 16384)
            )
            _prefill_opt_mma_ts_128x128_k64(
                tmem_base_i1 + PREFILL_OPT_TMEM_STATE_COL,
                tmem_base_i1 + PREFILL_OPT_TMEM_SHARED_INPUT_COL + 32,
                kt_desc_i1,
                _pf_pipe_full_addr(smem_addr, 256, kv_count_i1, 1),
                IO_DTYPE,
            )
            kv_producer_i1 = kv_producer_i1 + 1
            _prefill_opt_mma_commit(_pf_shared_addr(smem_addr, 24 + k_count_i1 * 8))
        _pf_producer_tail(smem_addr, 312, cg1_producer_i1, 1)
        _pf_producer_tail(smem_addr, 248, qstate_producer_i1, 1)
        _pf_producer_tail(smem_addr, 264, kv_producer_i1, 1)

    elif warp == 9:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
        q_index_tma: T.int32 = 0
        q_phase_tma: T.int32 = 1
        k_index_tma: T.int32 = 0
        k_phase_tma: T.int32 = 1
        v_index_tma: T.int32 = 0
        v_phase_tma: T.int32 = 1
        t_index_tma: T.int32 = 0
        t_phase_tma: T.int32 = 1

        if T.cuda.elect_sync():
            _pf_descriptor_copy_payload(q_map, descriptor_q)
        T.cuda.warp_sync()
        if T.cuda.elect_sync():
            _pf_descriptor_copy_payload(k_map, descriptor_k)
        T.cuda.warp_sync()
        if T.cuda.elect_sync():
            _pf_descriptor_copy_payload(v_map, descriptor_v)
        T.cuda.warp_sync()
        T.ptx.fence.acq_rel.cta()
        if T.cuda.elect_sync():
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group.read(0)
        T.cuda.warp_sync()
        if T.cuda.elect_sync():
            if IS_GQA:
                _pf_replace_descriptor(
                    descriptor_q,
                    q.data,
                    chunk_end,
                    HEAD_RATIO,
                    HEAD_BASE,
                    2 * D_HEAD * Q_HEADS,
                    2 * D_HEAD,
                    2 * D_HEAD * HEAD_RATIO,
                )
                _pf_replace_descriptor(
                    descriptor_k, k.data, chunk_end, K_HEADS, 1, 2 * D_HEAD * K_HEADS, 2 * D_HEAD, 0
                )
                _pf_replace_descriptor(
                    descriptor_v,
                    v.data,
                    chunk_end,
                    HEAD_BASE,
                    1,
                    2 * D_HEAD * V_HEADS,
                    2 * D_HEAD,
                    0,
                )
            else:
                _pf_replace_descriptor(
                    descriptor_q,
                    q.data,
                    chunk_end,
                    HEAD_BASE,
                    1,
                    2 * D_HEAD * Q_HEADS,
                    2 * D_HEAD,
                    0,
                )
                _pf_replace_descriptor(
                    descriptor_k, k.data, chunk_end, K_HEADS, 1, 2 * D_HEAD * K_HEADS, 2 * D_HEAD, 0
                )
                _pf_replace_descriptor(
                    descriptor_v,
                    v.data,
                    chunk_end,
                    HEAD_RATIO,
                    HEAD_BASE,
                    2 * D_HEAD * V_HEADS,
                    2 * D_HEAD,
                    2 * D_HEAD * HEAD_RATIO,
                )
        T.cuda.warp_sync()
        _pf_tensormap_release()

        for chunk_tma in T.serial(padded_chunks):
            token_tma: T.int32 = chunk_start + chunk_tma * T_BLOCK

            k_count_tma: T.int32 = k_index_tma
            k_count_phase_tma: T.int32 = k_phase_tma
            _pf_producer_acquire_state(smem_addr, 24, k_count_tma, k_count_phase_tma)
            k_full_tma: T.uint32 = _pf_shared_addr(smem_addr, k_count_tma * 8)
            if chunk_tma == 0:
                if T.cuda.elect_sync():
                    _pf_tensormap_acquire(descriptor_k)
            if T.cuda.elect_sync():
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(k_full_tma, T.uint32(16384))
                _prefill_opt_tma_k(
                    smem_addr,
                    PREFILL_OPT_K_OFF + k_count_tma * 16384,
                    descriptor_k,
                    k_full_tma,
                    token_tma,
                    k_head,
                )
            k_index_tma = _pf_pipe_next_index(k_count_tma, 3)
            k_phase_tma = _pf_pipe_next_phase(k_count_tma, k_count_phase_tma, 3)

            q_count_tma: T.int32 = q_index_tma
            q_count_phase_tma: T.int32 = q_phase_tma
            _pf_producer_acquire_state(smem_addr, 64, q_count_tma, q_count_phase_tma)
            q_full_tma: T.uint32 = _pf_shared_addr(smem_addr, 48 + q_count_tma * 8)
            if chunk_tma == 0:
                if T.cuda.elect_sync():
                    _pf_tensormap_acquire(descriptor_q)
            if T.cuda.elect_sync():
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(q_full_tma, T.uint32(16384))
                _prefill_opt_tma_q(
                    smem_addr,
                    PREFILL_OPT_Q_OFF + q_count_tma * 16384,
                    descriptor_q,
                    q_full_tma,
                    token_tma,
                    subhead,
                    base_head,
                    IS_GQA,
                )
            q_index_tma = _pf_pipe_next_index(q_count_tma, 2)
            q_phase_tma = _pf_pipe_next_phase(q_count_tma, q_count_phase_tma, 2)

            v_count_tma: T.int32 = v_index_tma
            v_count_phase_tma: T.int32 = v_phase_tma
            _pf_producer_acquire_state(smem_addr, 104, v_count_tma, v_count_phase_tma)
            v_full_tma: T.uint32 = _pf_shared_addr(smem_addr, 80 + v_count_tma * 8)
            if chunk_tma == 0:
                if T.cuda.elect_sync():
                    _pf_tensormap_acquire(descriptor_v)
            if T.cuda.elect_sync():
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(v_full_tma, T.uint32(16384))
                _prefill_opt_tma_v(
                    smem_addr,
                    PREFILL_OPT_V_OFF + v_count_tma * 16384,
                    descriptor_v,
                    v_full_tma,
                    token_tma,
                    subhead,
                    base_head,
                    IS_GQA,
                )
            v_index_tma = _pf_pipe_next_index(v_count_tma, 3)
            v_phase_tma = _pf_pipe_next_phase(v_count_tma, v_count_phase_tma, 3)

            t_count_tma: T.int32 = t_index_tma
            t_count_phase_tma: T.int32 = t_phase_tma
            _pf_producer_acquire_state(smem_addr, 224, t_count_tma, t_count_phase_tma)
            t_full_tma: T.uint32 = _pf_shared_addr(smem_addr, 208 + t_count_tma * 8)
            t_chunk_tma: T.int32 = chunk_tma
            if chunk_tma >= num_valid_chunks:
                t_chunk_tma = num_valid_chunks - 1
            if T.cuda.elect_sync():
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(t_full_tma, T.uint32(8192))
                _prefill_opt_tma_t(
                    smem_addr,
                    PREFILL_OPT_T_OFF + t_count_tma * 8192,
                    T.address_of(t_map),
                    t_full_tma,
                    t_block_start + t_chunk_tma,
                    subhead,
                    base_head,
                )
            t_index_tma = _pf_pipe_next_index(t_count_tma, 2)
            t_phase_tma = _pf_pipe_next_phase(t_count_tma, t_count_phase_tma, 2)

        _pf_producer_tail_state(smem_addr, 64, q_index_tma, q_phase_tma, 2)
        _pf_producer_tail_state(smem_addr, 24, k_index_tma, k_phase_tma, 3)
        _pf_producer_tail_state(smem_addr, 104, v_index_tma, v_phase_tma, 3)
        _pf_producer_tail_state(smem_addr, 224, t_index_tma, t_phase_tma, 2)

    # Warp 11 independently owns gate lookahead and the O TensorMap.
    if warp == 11:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
        gate_index_epi: T.int32 = 0
        gate_phase_epi: T.int32 = 1
        o_index_epi: T.int32 = 0
        o_phase_epi: T.int32 = 0
        if T.cuda.elect_sync():
            _pf_descriptor_copy_payload(o_map, descriptor_o)
        T.cuda.warp_sync()
        T.ptx.fence.acq_rel.cta()
        if T.cuda.elect_sync():
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group.read(0)
        T.cuda.warp_sync()
        if T.cuda.elect_sync():
            _pf_replace_descriptor(
                descriptor_o,
                o.data,
                chunk_end,
                HEAD_RATIO,
                HEAD_BASE,
                2 * D_HEAD * STATE_HEADS,
                2 * D_HEAD,
                2 * D_HEAD * HEAD_RATIO,
            )
        T.cuda.warp_sync()
        _pf_tensormap_release()
        if T.cuda.elect_sync():
            _pf_tensormap_acquire(descriptor_o)

        if chunk_len > 0:
            for prefetch_epi in T.unroll(2):
                _prefill_opt_load_gate(
                    smem_addr,
                    s_cumsumlog,
                    s_cumprod,
                    alpha,
                    chunk_start + prefetch_epi * T_BLOCK,
                    state_head,
                    prefetch_epi >= num_valid_chunks - 1,
                    chunk_end,
                    lane,
                    gate_index_epi,
                    gate_phase_epi,
                    STATE_HEADS,
                )
                previous_gate_epi: T.int32 = gate_index_epi
                previous_gate_phase_epi: T.int32 = gate_phase_epi
                gate_index_epi = _pf_pipe_next_index(previous_gate_epi, 5)
                gate_phase_epi = _pf_pipe_next_phase(previous_gate_epi, previous_gate_phase_epi, 5)
            if padded_chunks > 2:
                for prefetch_epi in T.unroll(2, 4):
                    _prefill_opt_load_gate(
                        smem_addr,
                        s_cumsumlog,
                        s_cumprod,
                        alpha,
                        chunk_start + prefetch_epi * T_BLOCK,
                        state_head,
                        prefetch_epi >= num_valid_chunks - 1,
                        chunk_end,
                        lane,
                        gate_index_epi,
                        gate_phase_epi,
                        STATE_HEADS,
                    )
                    previous_gate_epi: T.int32 = gate_index_epi
                    previous_gate_phase_epi: T.int32 = gate_phase_epi
                    gate_index_epi = _pf_pipe_next_index(previous_gate_epi, 5)
                    gate_phase_epi = _pf_pipe_next_phase(
                        previous_gate_epi, previous_gate_phase_epi, 5
                    )

            for chunk_epi in T.serial(padded_chunks):
                future_epi: T.int32 = chunk_epi + 4
                if future_epi < padded_chunks:
                    _prefill_opt_load_gate(
                        smem_addr,
                        s_cumsumlog,
                        s_cumprod,
                        alpha,
                        chunk_start + future_epi * T_BLOCK,
                        state_head,
                        future_epi >= num_valid_chunks - 1,
                        chunk_end,
                        lane,
                        gate_index_epi,
                        gate_phase_epi,
                        STATE_HEADS,
                    )
                    previous_gate_epi: T.int32 = gate_index_epi
                    previous_gate_phase_epi: T.int32 = gate_phase_epi
                    gate_index_epi = _pf_pipe_next_index(previous_gate_epi, 5)
                    gate_phase_epi = _pf_pipe_next_phase(
                        previous_gate_epi, previous_gate_phase_epi, 5
                    )
                o_count_epi: T.int32 = o_index_epi
                o_count_phase_epi: T.int32 = o_phase_epi
                _prefill_opt_store_o(
                    smem_addr,
                    descriptor_o,
                    chunk_start + chunk_epi * T_BLOCK,
                    subhead,
                    base_head,
                    o_count_epi,
                    o_count_phase_epi,
                )
                o_index_epi = _pf_pipe_next_index(o_count_epi, 2)
                o_phase_epi = _pf_pipe_next_phase(o_count_epi, o_count_phase_epi, 2)
        _pf_producer_tail_state(smem_addr, 168, gate_index_epi, gate_phase_epi, 5)


@T.jit
def _prefill_scalar_sm100(
    q_h: T.handle,
    k_h: T.handle,
    v_h: T.handle,
    alpha_h: T.handle,
    t_h: T.handle,
    fixed_state_h: T.handle,
    initial_state_workspace_h: T.handle,
    o_h: T.handle,
    cu_seqlens_h: T.handle,
    scale: T.float32,
    *,
    IO_DTYPE: T.constexpr,
    CU_DTYPE: T.constexpr,
    TOTAL_TOKENS: T.constexpr,
    NUM_SEQUENCES: T.constexpr,
    Q_HEADS: T.constexpr,
    K_HEADS: T.constexpr,
    V_HEADS: T.constexpr,
    STATE_HEADS: T.constexpr,
    TOTAL_T_BLOCKS: T.constexpr,
    TOTAL_CP_CHUNKS: T.constexpr,
    MAX_CP_CHUNKS: T.constexpr,
    CP_CHUNK_LEN: T.constexpr,
    NEEDS_INITIAL_STATE: T.constexpr,
):
    """Evaluate each CP chunk independently from its fixed boundary state."""
    q = T.match_buffer(q_h, (TOTAL_TOKENS * Q_HEADS * D_HEAD,), IO_DTYPE, scope="global")
    k = T.match_buffer(k_h, (TOTAL_TOKENS * K_HEADS * D_HEAD,), IO_DTYPE, scope="global")
    v = T.match_buffer(v_h, (TOTAL_TOKENS * V_HEADS * D_HEAD,), IO_DTYPE, scope="global")
    alpha = T.match_buffer(alpha_h, (TOTAL_TOKENS * STATE_HEADS,), "float32", scope="global")
    t = T.match_buffer(
        t_h, (TOTAL_T_BLOCKS * STATE_HEADS * T_BLOCK * T_BLOCK,), IO_DTYPE, scope="global"
    )
    fixed_state = T.match_buffer(
        fixed_state_h, (TOTAL_CP_CHUNKS * STATE_HEADS * D_HEAD * D_HEAD,), "float32", scope="global"
    )
    initial_state_workspace = T.match_buffer(
        initial_state_workspace_h,
        (NUM_SEQUENCES * STATE_HEADS * D_HEAD * D_HEAD,),
        "float32",
        scope="global",
    )
    o = T.match_buffer(o_h, (TOTAL_TOKENS * STATE_HEADS * D_HEAD,), IO_DTYPE, scope="global")
    cu_seqlens = T.match_buffer(cu_seqlens_h, (NUM_SEQUENCES + 1,), CU_DTYPE, scope="global")
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    # TIRX_TRANSCRIBE_START cp_delta_rule_prefill_sm100

    bx, seq_idx = T.cta_id([STATE_HEADS * MAX_CP_CHUNKS, NUM_SEQUENCES])
    tid = T.thread_id([PREFILL_THREADS])
    state_head: T.int32 = bx % STATE_HEADS
    chunk_in_seq: T.int32 = bx // STATE_HEADS
    q_head: T.int32 = state_head * Q_HEADS // STATE_HEADS
    k_head: T.int32 = state_head * K_HEADS // STATE_HEADS
    v_head: T.int32 = state_head * V_HEADS // STATE_HEADS
    seq_start: T.int32 = T.cast(cu_seqlens[seq_idx], "int32")
    seq_end: T.int32 = T.cast(cu_seqlens[seq_idx + 1], "int32")
    seq_len: T.int32 = seq_end - seq_start
    num_chunks: T.int32 = (seq_len + CP_CHUNK_LEN - 1) // CP_CHUNK_LEN

    if chunk_in_seq < num_chunks and tid < D_HEAD:
        value_row: T.int32 = tid
        state_values: T.float32[D_HEAD]
        cp_start: T.int32 = _device_chunk_bound(seq_idx, seq_start, CP_CHUNK_LEN)
        if chunk_in_seq == 0:
            for key_col in T.serial(D_HEAD):
                if NEEDS_INITIAL_STATE:
                    state_values[key_col] = initial_state_workspace[
                        ((seq_idx * STATE_HEADS + state_head) * D_HEAD + value_row) * D_HEAD
                        + key_col
                    ]
                else:
                    state_values[key_col] = 0.0
        else:
            previous_slot: T.int32 = cp_start + chunk_in_seq - 1
            for key_col in T.serial(D_HEAD):
                state_values[key_col] = fixed_state[
                    ((previous_slot * STATE_HEADS + state_head) * D_HEAD + value_row) * D_HEAD
                    + key_col
                ]

        token_start: T.int32 = seq_start + chunk_in_seq * CP_CHUNK_LEN
        valid_len: T.int32 = T.min(CP_CHUNK_LEN, seq_end - token_start)
        t_start: T.int32 = _device_chunk_bound(seq_idx, seq_start, T_BLOCK)
        num_t_blocks: T.int32 = (valid_len + T_BLOCK - 1) // T_BLOCK
        if CP_CHUNK_LEN == T_BLOCK:
            # The one-block specialization has no intra-CTA state handoff.
            # Retain the source-equivalent token recurrence here; this avoids
            # introducing an extra IO round trip for a state that is consumed
            # only once.
            for token_local in T.serial(valid_len):
                token: T.int32 = token_start + token_local
                t_inner: T.int32 = token_local
                t_block: T.int32 = t_start + chunk_in_seq
                beta_value: T.float32 = -T.cast(
                    t[
                        (t_block * STATE_HEADS + state_head) * T_BLOCK * T_BLOCK
                        + t_inner * T_BLOCK
                        + t_inner
                    ],
                    "float32",
                )
                alpha_value: T.float32 = alpha[token * STATE_HEADS + state_head]
                projected: T.float32 = 0.0
                for key_col in T.serial(D_HEAD):
                    projected = projected + state_values[key_col] * T.cast(
                        k[token * K_HEADS * D_HEAD + k_head * D_HEAD + key_col], "float32"
                    )
                old_value: T.float32 = alpha_value * projected
                input_value: T.float32 = T.cast(
                    v[token * V_HEADS * D_HEAD + v_head * D_HEAD + value_row], "float32"
                )
                new_value: T.float32 = beta_value * input_value + (1.0 - beta_value) * old_value
                delta_value: T.float32 = new_value - old_value
                for key_col in T.serial(D_HEAD):
                    key_value: T.float32 = T.cast(
                        k[token * K_HEADS * D_HEAD + k_head * D_HEAD + key_col], "float32"
                    )
                    state_values[key_col] = (
                        alpha_value * state_values[key_col] + delta_value * key_value
                    )
                output_value: T.float32 = 0.0
                for key_col in T.serial(D_HEAD):
                    output_value = output_value + state_values[key_col] * T.cast(
                        q[token * Q_HEADS * D_HEAD + q_head * D_HEAD + key_col], "float32"
                    )
                o[token * STATE_HEADS * D_HEAD + state_head * D_HEAD + value_row] = T.cast(
                    scale * output_value, IO_DTYPE
                )
            num_t_blocks = 0
        for block_local in T.serial(num_t_blocks):
            block_token_start: T.int32 = token_start + block_local * T_BLOCK
            block_valid: T.int32 = T.min(T_BLOCK, seq_end - block_token_start)
            t_block: T.int32 = t_start + chunk_in_seq * (CP_CHUNK_LEN // T_BLOCK) + block_local
            valid_state: T.bool = NEEDS_INITIAL_STATE or chunk_in_seq > 0 or block_local > 0

            cumsum_log: T.float32[T_BLOCK]
            running_log: T.float32 = 0.0
            for token_col in T.serial(T_BLOCK):
                gate_value: T.float32 = 1.0
                if token_col < block_valid:
                    gate_value = alpha[(block_token_start + token_col) * STATE_HEADS + state_head]
                running_log = running_log + _lg2_approx_ftz(gate_value + 1.0e-10)
                cumsum_log[token_col] = running_log

            vks = T.alloc_local((T_BLOCK,), IO_DTYPE)
            nv: T.float32[T_BLOCK]
            for token_col in T.serial(T_BLOCK):
                vks_value: T.float32 = 0.0
                if token_col < block_valid:
                    token: T.int32 = block_token_start + token_col
                    vks_value = T.cast(
                        v[token * V_HEADS * D_HEAD + v_head * D_HEAD + value_row], "float32"
                    )
                    if valid_state:
                        ks_value: T.float32 = 0.0
                        for key_col in T.serial(D_HEAD):
                            state_io: T.float32 = T.cast(
                                T.cast(state_values[key_col], IO_DTYPE), "float32"
                            )
                            ks_value = ks_value + state_io * T.cast(
                                k[token * K_HEADS * D_HEAD + k_head * D_HEAD + key_col], "float32"
                            )
                        ks_io: T.float32 = T.cast(
                            T.cast(ks_value * _ex2_approx_ftz(cumsum_log[token_col]), IO_DTYPE),
                            "float32",
                        )
                        vks_value = T.cast(T.cast(vks_value - ks_io, IO_DTYPE), "float32")
                vks[token_col] = T.cast(vks_value, IO_DTYPE)

            # NV = (V - gamma*K*S) @ Ainv.  Ainv is the signed T tile
            # transformed by the source gate sandwich and rounded to IO.
            for output_token in T.serial(T_BLOCK):
                nv_value: T.float32 = 0.0
                if output_token < block_valid:
                    for input_token in T.serial(T_BLOCK):
                        if input_token <= output_token:
                            gamma_ratio: T.float32 = _ex2_approx_ftz(
                                cumsum_log[output_token] - cumsum_log[input_token]
                            )
                            ainv_io: T.float32 = T.cast(
                                T.cast(
                                    -gamma_ratio
                                    * T.cast(
                                        t[
                                            (t_block * STATE_HEADS + state_head) * T_BLOCK * T_BLOCK
                                            + input_token * T_BLOCK
                                            + output_token
                                        ],
                                        "float32",
                                    ),
                                    IO_DTYPE,
                                ),
                                "float32",
                            )
                            nv_value = nv_value + T.cast(vks[input_token], "float32") * ainv_io
                nv[output_token] = nv_value

            # Q-state plus the lower-triangular NV @ QK path.
            for output_token in T.serial(T_BLOCK):
                if output_token < block_valid:
                    output_position: T.int32 = block_token_start + output_token
                    output_value: T.float32 = 0.0
                    if valid_state:
                        for key_col in T.serial(D_HEAD):
                            output_value = output_value + T.cast(
                                T.cast(state_values[key_col], IO_DTYPE), "float32"
                            ) * T.cast(
                                q[output_position * Q_HEADS * D_HEAD + q_head * D_HEAD + key_col],
                                "float32",
                            )
                        output_value = (
                            output_value * _ex2_approx_ftz(cumsum_log[output_token]) * scale
                        )
                    for input_token in T.serial(T_BLOCK):
                        if input_token <= output_token:
                            input_position: T.int32 = block_token_start + input_token
                            qk_value: T.float32 = 0.0
                            for key_col in T.serial(D_HEAD):
                                qk_value = qk_value + T.cast(
                                    q[
                                        output_position * Q_HEADS * D_HEAD
                                        + q_head * D_HEAD
                                        + key_col
                                    ],
                                    "float32",
                                ) * T.cast(
                                    k[
                                        input_position * K_HEADS * D_HEAD
                                        + k_head * D_HEAD
                                        + key_col
                                    ],
                                    "float32",
                                )
                            qk_io: T.float32 = T.cast(
                                T.cast(
                                    qk_value
                                    * _ex2_approx_ftz(
                                        cumsum_log[output_token] - cumsum_log[input_token]
                                    )
                                    * scale,
                                    IO_DTYPE,
                                ),
                                "float32",
                            )
                            output_value = (
                                output_value
                                + T.cast(T.cast(nv[input_token], IO_DTYPE), "float32") * qk_io
                            )
                    o[output_position * STATE_HEADS * D_HEAD + state_head * D_HEAD + value_row] = (
                        T.cast(output_value, IO_DTYPE)
                    )

            total_decay: T.float32 = _ex2_approx_ftz(cumsum_log[T_BLOCK - 1])
            for key_col in T.serial(D_HEAD):
                next_state: T.float32 = total_decay * state_values[key_col]
                for input_token in T.serial(T_BLOCK):
                    if input_token < block_valid:
                        input_position: T.int32 = block_token_start + input_token
                        decay_io: T.float32 = T.cast(
                            T.cast(
                                nv[input_token]
                                * _ex2_approx_ftz(
                                    cumsum_log[T_BLOCK - 1] - cumsum_log[input_token]
                                ),
                                IO_DTYPE,
                            ),
                            "float32",
                        )
                        next_state = next_state + decay_io * T.cast(
                            k[input_position * K_HEADS * D_HEAD + k_head * D_HEAD + key_col],
                            "float32",
                        )
                state_values[key_col] = next_state


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
    """Return all six source-shaped specializations for this configuration."""
    cfg = _cfg(**kwargs)
    spec = _specialization(cfg, kwargs.get("device", "cuda"))
    t_kernel = _t_precompute_sm100.specialize(
        **{
            key: spec[key]
            for key in (
                "IO_DTYPE",
                "CU_DTYPE",
                "TOTAL_TOKENS",
                "NUM_SEQUENCES",
                "K_HEADS",
                "STATE_HEADS",
                "TOTAL_T_BLOCKS",
                "MAX_T_BLOCKS",
            )
        }
    )
    mn_kernel = _mn_precompute_sm100.specialize(
        **{
            key: spec[key]
            for key in (
                "IO_DTYPE",
                "CU_DTYPE",
                "TOTAL_TOKENS",
                "NUM_SEQUENCES",
                "K_HEADS",
                "V_HEADS",
                "STATE_HEADS",
                "TOTAL_T_BLOCKS",
                "TOTAL_CP_CHUNKS",
                "MAX_CP_CHUNKS",
                "CP_CHUNK_LEN",
            )
        }
    )
    fixup_common = {
        key: spec[key]
        for key in (
            "CU_DTYPE",
            "STATE_DTYPE",
            "NUM_SEQUENCES",
            "STATE_HEADS",
            "STATE_POOL",
            "TOTAL_CP_CHUNKS",
            "CP_CHUNK_LEN",
            "NEEDS_INITIAL_STATE",
            "STORE_FINAL_STATE",
            "USE_STATE_INDICES",
        )
    }
    prefill_kernel = _prefill_sm100.specialize(
        **{
            key: spec[key]
            for key in (
                "IO_DTYPE",
                "CU_DTYPE",
                "TOTAL_TOKENS",
                "NUM_SEQUENCES",
                "Q_HEADS",
                "K_HEADS",
                "V_HEADS",
                "STATE_HEADS",
                "TOTAL_T_BLOCKS",
                "TOTAL_CP_CHUNKS",
                "MAX_CP_CHUNKS",
                "CP_CHUNK_LEN",
                "NEEDS_INITIAL_STATE",
                "IS_GQA",
                "HEAD_BASE",
                "HEAD_RATIO",
            )
        }
    )
    return {
        "t_precompute": t_kernel,
        "mn_precompute": mn_kernel,
        "fixup_simt_row4": _fixup_sm100.specialize(
            **fixup_common, ROWS_PER_CTA=spec["FIXUP_SIMT_ROWS"]
        ),
        "fixup_utcmma64": _fixup_utcmma_sm100.specialize(
            **fixup_common,
            ROWS=64,
            M_STAGES=2,
            COMPUTE_REGS=120,
            SMEM_TOTAL=164864,
            TMEM_HOLDING_OFF=80,
            M_OFF=1024,
            N_OFF=132096,
            M_FULL_OFF=0,
            M_EMPTY_OFF=16,
            N_FULL_OFF=32,
            N_EMPTY_OFF=40,
            READY_FULL_OFF=48,
            READY_EMPTY_OFF=56,
            DONE_FULL_OFF=64,
            DONE_EMPTY_OFF=72,
        ),
        "fixup_utcmma128": _fixup_utcmma_sm100.specialize(
            **fixup_common,
            ROWS=128,
            M_STAGES=1,
            COMPUTE_REGS=256,
            SMEM_TOTAL=132096,
            TMEM_HOLDING_OFF=64,
            M_OFF=1024,
            N_OFF=66560,
            M_FULL_OFF=0,
            M_EMPTY_OFF=8,
            N_FULL_OFF=16,
            N_EMPTY_OFF=24,
            READY_FULL_OFF=32,
            READY_EMPTY_OFF=40,
            DONE_FULL_OFF=48,
            DONE_EMPTY_OFF=56,
        ),
        "prefill": prefill_kernel,
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

    q = 0.25 * torch.randn(
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
    v = 0.25 * torch.randn(
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


def _run_oracle(
    case: dict[str, Any], output: torch.Tensor, final_state: torch.Tensor | None
) -> None:
    from tirx_kernels.flashinfer._gdn_reference import gated_delta_rule_prefill

    cfg: GDNCPPrefillSM100Config = case["config"]
    gated_delta_rule_prefill(
        output=output,
        final_state=final_state if cfg.store_final_state else None,
        q=case["q"],
        k=case["k"],
        v=case["v"],
        alpha=case["alpha"],
        beta=case["beta"],
        cu_seqlens=case["cu_seqlens"],
        scale=case["scale"],
        initial_state=case["initial_state"] if cfg.needs_initial_state else None,
        state_indices=case["state_indices"] if cfg.indexed_state else None,
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
        from flashinfer.gdn_kernels.blackwell.gdn_cp_prefill import cp_delta_rule_dsl_sm100

        source_output = torch.empty_like(case["output"])
        source_state = torch.zeros_like(case["final_state"])
        cfg: GDNCPPrefillSM100Config = case["config"]

        def launch():
            cp_delta_rule_dsl_sm100(
                source_output,
                source_state if cfg.store_final_state else None,
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

        launch()
        launch()
        torch.cuda.synchronize()
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
