# Copyright (c) 2025 by FlashInfer team.
# Copyright (c) 2026 The TIRX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TIRx port of FlashInfer's selective-state-update STP simple kernel."""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any
from unittest import SkipTest

import torch

from tvm.script import tirx as T

KERNEL_META = {
    "name": "selective_state_update_stp_simple",
    "category": "flashinfer",
    "compute_capability": 10,
}

FROZEN_FLASHINFER_SOURCE_SHA256 = "c0e13b64bf42f4f8155058dc9f5877f7aca90832f50a1e7602863894908e89fd"
_LOG2_E = 1.4426950408889634
_LN_2 = 0.6931471805599453
_FLT_LOWEST = -3.4028234663852886e38


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _ptx_unary(chain: str, value, dtype: str = "float32"):
    out = T.alloc_local((1,), dtype)
    T.evaluate(T.ptx[chain](out[0], value))
    return out[0]


def _ptx_binary(chain: str, lhs, rhs, dtype: str = "float32"):
    out = T.alloc_local((1,), dtype)
    T.evaluate(T.ptx[chain](out[0], lhs, rhs))
    return out[0]


def _ptx_ternary(chain: str, lhs, rhs, acc, dtype: str = "float32"):
    out = T.alloc_local((1,), dtype)
    T.evaluate(T.ptx[chain](out[0], lhs, rhs, acc))
    return out[0]


def _mul(lhs, rhs):
    return _ptx_binary("mul.ftz.f32", lhs, rhs)


def _add(lhs, rhs):
    return _ptx_binary("add.ftz.f32", lhs, rhs)


def _sub(lhs, rhs):
    return _ptx_binary("sub.ftz.f32", lhs, rhs)


def _fma(lhs, rhs, acc):
    return _ptx_ternary("fma.rn.ftz.f32", lhs, rhs, acc)


def _max(lhs, rhs):
    return _ptx_binary("max.ftz.f32", lhs, rhs)


def _min(lhs, rhs):
    return _ptx_binary("min.ftz.f32", lhs, rhs)


def _abs(value):
    return _ptx_unary("abs.ftz.f32", value)


def _exp2(value):
    return _ptx_unary("ex2.approx.ftz.f32", value)


def _log2(value):
    return _ptx_unary("lg2.approx.ftz.f32", value)


def _div(lhs, rhs):
    return _ptx_binary("div.approx.ftz.f32", lhs, rhs)


def _rcp(value):
    return _ptx_unary("rcp.approx.ftz.f32", value)


def _prmt_5410(lhs, rhs):
    return _ptx_ternary(
        "prmt.b32", T.cast(lhs, "uint32"), T.cast(rhs, "uint32"), T.uint32(0x5410), dtype="uint32"
    )


def _mul_hi_u32(lhs, rhs):
    return _ptx_binary("mul.hi.u32", lhs, rhs, dtype="uint32")


def _mul_lo_s32(lhs, rhs):
    return _ptx_binary("mul.lo.s32", lhs, rhs, dtype="int32")


def _add_s32(lhs, rhs):
    return _ptx_binary("add.s32", lhs, rhs, dtype="int32")


def _lane_mask(raw_lane):
    return T.cast(T.bitwise_and(T.cast(raw_lane, "uint32"), T.uint32(31)), "int32")


def _global_load_u16(buffer, index):
    out = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.ld.global_.b16(out[0], buffer.ptr_to([index])))
    return out[0]


def _global_load_u32(buffer, index):
    out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.ld.global_.b32(out[0], buffer.ptr_to([index])))
    return out[0]


def _shared_load_u16(buffer, index):
    out = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.ld.shared.b16(out[0], buffer.ptr_to([index])))
    return out[0]


def _shared_load_u32(buffer, index):
    out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.ld.shared.b32(out[0], buffer.ptr_to([index])))
    return out[0]


def _bf16_to_f32(bits):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.cvt.f32.bf16(out[0], T.cast(bits, "uint16")))
    return out[0]


def _f16_to_f32(bits):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.cvt.f32.f16(out[0], T.cast(bits, "uint16")))
    return out[0]


def _i16_to_f32(bits):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.cvt.rn.f32.s16(out[0], T.reinterpret("int16", T.cast(bits, "uint16"))))
    return out[0]


def _f32_to_bf16(value):
    out = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.cvt.rn.bf16.f32(out[0], value))
    return out[0]


def _f32_to_f16(value):
    out = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.cvt.rn.f16.f32(out[0], value))
    return out[0]


def _load_weight(buffer, index, dtype: str):
    if dtype == "float32":
        return T.reinterpret("float32", _global_load_u32(buffer, index))
    return _bf16_to_f32(_global_load_u16(buffer, index))


def _state_bits_to_f32(bits, dtype: str):
    if dtype == "bfloat16":
        return _bf16_to_f32(bits)
    if dtype == "float16":
        return _f16_to_f32(bits)
    if dtype == "int16":
        return _i16_to_f32(bits)
    return T.reinterpret("float32", T.cast(bits, "uint32"))


def _f32_to_state_bits(value, dtype: str):
    if dtype == "bfloat16":
        return _f32_to_bf16(value)
    if dtype == "float16":
        return _f32_to_f16(value)
    return T.reinterpret("uint32", value)


def _load_two_byte_vector(buffer, index, count: int, scope: str):
    bits = T.alloc_local((count,), "uint16")
    prefix = f"ld.{scope}"
    if count == 2:
        T.evaluate(T.ptx[f"{prefix}.v2.b16"](bits[0], bits[1], buffer.ptr_to([index])))
    elif count == 3:
        for e in range(3):
            T.evaluate(T.ptx[f"{prefix}.b16"](bits[e], buffer.ptr_to([index + e])))
    elif count == 4:
        T.evaluate(
            T.ptx[f"{prefix}.v4.b16"](bits[0], bits[1], bits[2], bits[3], buffer.ptr_to([index]))
        )
    else:
        words = T.alloc_local((4,), "uint32")
        T.evaluate(
            T.ptx[f"{prefix}.v4.b32"](
                words[0], words[1], words[2], words[3], buffer.ptr_to([index])
            )
        )
        for pair in range(4):
            T.buffer_store(
                bits, T.cast(T.bitwise_and(words[pair], T.uint32(0xFFFF)), "uint16"), [2 * pair]
            )
            T.buffer_store(
                bits, T.cast(T.shift_right(words[pair], T.uint32(16)), "uint16"), [2 * pair + 1]
            )
    return bits


def _store_two_byte_vector(buffer, index, bits, count: int, scope: str = "global_"):
    prefix = f"st.{scope}"
    if count == 2:
        T.evaluate(T.ptx[f"{prefix}.v2.b16"](buffer.ptr_to([index]), bits[0], bits[1]))
    elif count == 3:
        for e in range(3):
            T.evaluate(T.ptx[f"{prefix}.b16"](buffer.ptr_to([index + e]), bits[e]))
    elif count == 4:
        T.evaluate(
            T.ptx[f"{prefix}.v4.b16"](buffer.ptr_to([index]), bits[0], bits[1], bits[2], bits[3])
        )
    else:
        words = T.alloc_local((4,), "uint32")
        for pair in range(4):
            T.buffer_store(
                words,
                T.bitwise_or(
                    T.cast(bits[2 * pair], "uint32"),
                    T.shift_left(T.cast(bits[2 * pair + 1], "uint32"), T.uint32(16)),
                ),
                [pair],
            )
        T.evaluate(
            T.ptx[f"{prefix}.v4.b32"](
                buffer.ptr_to([index]), words[0], words[1], words[2], words[3]
            )
        )


def _load_f32_vector(buffer, index, count: int):
    words = T.alloc_local((count,), "uint32")
    if count == 2:
        T.evaluate(T.ptx.ld.global_.v2.b32(words[0], words[1], buffer.ptr_to([index])))
    elif count == 3:
        for e in range(3):
            T.evaluate(T.ptx.ld.global_.b32(words[e], buffer.ptr_to([index + e])))
    else:
        T.evaluate(
            T.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
        )
    return words


def _store_f32_vector(buffer, index, words, count: int):
    if count == 2:
        T.evaluate(T.ptx.st.global_.v2.b32(buffer.ptr_to([index]), words[0], words[1]))
    elif count == 3:
        for e in range(3):
            T.evaluate(T.ptx.st.global_.b32(buffer.ptr_to([index + e]), words[e]))
    else:
        T.evaluate(
            T.ptx.st.global_.v4.b32(buffer.ptr_to([index]), words[0], words[1], words[2], words[3])
        )


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "label": label,
        "batch": 64,
        "nheads": 64,
        "dim": 64,
        "dstate": 128,
        "ngroups": 8,
        "input_dtype": "bfloat16",
        "state_dtype": "bfloat16",
        "weight_dtype": "float32",
        "matrix_a_dtype": "float32",
        "index_dtype": "int64",
        "has_state_indices": True,
        "has_dst_indices": False,
        "index_rank": 1,
        "has_z": False,
        "has_d": True,
        "has_dt_bias": True,
        "dt_softplus": True,
        "update_state": True,
        "state_stride_factor": 1,
        "pad_every": 0,
        "use_out_tensor": True,
        "philox_rounds": 0,
        "seed": 0,
    }
    config.update(overrides)
    return config


# Every performance row changes one source branch or specialization from the
# base case.  The two simple launch modes are represented by base and batch=1.
BENCH_CONFIGS = [
    _case("b64_h64_d64_s128_r8_base"),
    _case("b1_h64_d64_s128_r8_tiled", batch=1),
    _case("b64_h8_d64_s128_r1", nheads=8),
    _case("b64_h64_d128_s128_r8", dim=128),
    _case("b64_h64_d256_s128_r8", dim=256),
    _case("b64_h64_d64_s64_r8", dstate=64),
    _case("b64_h64_d64_s96_r8", dstate=96),
    _case("b64_h64_d64_s256_r8", dstate=256),
    _case("b64_h64_d64_s128_r8_statef16", state_dtype="float16"),
    _case("b64_h64_d64_s128_r8_statef32", state_dtype="float32"),
    _case("b64_h64_d64_s128_r8_weightbf16", weight_dtype="bfloat16"),
    _case("b64_h64_d64_s128_r1", ngroups=64),
    _case("b64_h64_d64_s128_r2", ngroups=32),
    _case("b64_h64_d64_s128_r4", ngroups=16),
    _case("b64_h64_d64_s128_r16", ngroups=4),
    _case("b64_h64_d64_s128_r32", ngroups=2),
    _case("b64_h64_d64_s128_r64", ngroups=1),
    _case("b64_h64_d64_s128_r8_z", has_z=True),
    _case("b64_h64_d64_s128_r8_no_dt_bias", has_dt_bias=False),
    _case("b64_h64_d64_s128_r8_no_softplus", dt_softplus=False),
    _case("b64_h64_d64_s128_r8_no_update", update_state=False),
    _case("b64_h64_d64_s128_r8_no_indices", has_state_indices=False, index_dtype="int32"),
    _case("b64_h64_d64_s128_r8_indices_i32", index_dtype="int32"),
    _case("b64_h64_d64_s128_r8_stride2", state_stride_factor=2),
    _case("b64_h64_d64_s128_r8_dst2d", has_dst_indices=True, index_rank=2, index_dtype="int32"),
    _case("b64_h64_d64_s128_r8_pad4", pad_every=4, index_dtype="int32"),
    _case("b64_h64_d64_s128_r8_int16", state_dtype="int16"),
    _case("b64_h64_d64_s128_r8_philox10", state_dtype="float16", philox_rounds=10, seed=42),
    _case(
        "b64_h64_d64_s64_r8_philox10", dstate=64, state_dtype="float16", philox_rounds=10, seed=42
    ),
]


# Correctness includes every benchmark specialization plus the additional
# one-axis rows covered by FlashInfer's upstream STP tests.  The public
# FlashInfer API requires D, so its nullable device branch is correctness-only:
# a source/TIRx benchmark row could not exercise matching implementation paths.
CONFIGS = [dict(config) for config in BENCH_CONFIGS] + [
    _case("b64_h64_d64_s128_r8_no_d", has_d=False),
    _case("b64_h64_d64_s128_r8_out_allocated", use_out_tensor=False),
    *[
        _case(
            f"b{batch}_h64_d64_s128_r8_dst1d",
            batch=batch,
            has_dst_indices=True,
            index_dtype="int32",
        )
        for batch in (1, 4, 32, 64)
    ],
    *[
        _case(
            f"b{batch}_h64_d64_s128_r8_dst2d_correctness",
            batch=batch,
            has_dst_indices=True,
            index_rank=2,
            index_dtype="int32",
        )
        for batch in (1, 16, 64)
    ],
    _case("b1_h64_d64_s128_r8_int16", batch=1, state_dtype="int16"),
    _case("b64_h8_d64_s128_r1_int16", nheads=8, state_dtype="int16"),
    _case("b64_h64_d128_s128_r8_int16", dim=128, state_dtype="int16"),
    _case("b64_h64_d64_s64_r8_int16", dstate=64, state_dtype="int16"),
    _case("b64_h64_d64_s256_r8_int16", dstate=256, state_dtype="int16"),
    _case("b64_h64_d64_s128_r8_int16_weightbf16", state_dtype="int16", weight_dtype="bfloat16"),
]


@T.jit
def _selective_state_update_stp_simple(
    state_h: T.handle,
    state_scale_h: T.handle,
    x_h: T.handle,
    dt_h: T.handle,
    matrix_a_h: T.handle,
    matrix_b_h: T.handle,
    matrix_c_h: T.handle,
    d_h: T.handle,
    z_h: T.handle,
    dt_bias_h: T.handle,
    state_indices_h: T.handle,
    dst_indices_h: T.handle,
    rand_seed_h: T.handle,
    output_h: T.handle,
    state_stride_batch: T.int64,
    state_scale_stride_batch: T.int64,
    x_stride_batch: T.int64,
    dt_stride_batch: T.int64,
    b_stride_batch: T.int64,
    c_stride_batch: T.int64,
    z_stride_batch: T.int64,
    out_stride_batch: T.int64,
    state_indices_stride_batch: T.int64,
    dst_indices_stride_batch: T.int64,
    nheads_runtime: T.int32,
    ngroups_runtime: T.int32,
    dt_softplus: T.int32,
    update_state: T.int32,
    pad_slot_id: T.int32,
    dim_tiles_runtime: T.int32,
    *,
    BATCH: T.constexpr,
    NHEADS: T.constexpr,
    DIM: T.constexpr,
    DSTATE: T.constexpr,
    ROWS_PER_BLOCK: T.constexpr,
    STATE_DTYPE: T.constexpr,
    WEIGHT_DTYPE: T.constexpr,
    INDEX_DTYPE: T.constexpr,
    STATE_ELEMENTS: T.constexpr,
    SCALE_ELEMENTS: T.constexpr,
    X_ELEMENTS: T.constexpr,
    DT_ELEMENTS: T.constexpr,
    BC_ELEMENTS: T.constexpr,
    INDEX_ELEMENTS: T.constexpr,
    HAS_STATE_INDICES: T.constexpr,
    HAS_DST_INDICES: T.constexpr,
    HAS_Z: T.constexpr,
    HAS_D: T.constexpr,
    HAS_DT_BIAS: T.constexpr,
    SCALE_STATE: T.constexpr,
    PHILOX_ROUNDS: T.constexpr,
    STATE_BYTES: T.constexpr,
    STATE_VECTOR: T.constexpr,
    STATE_ITERATIONS: T.constexpr,
    LANE_STATE_COUNT: T.constexpr,
    OFF_X: T.constexpr,
    OFF_Z: T.constexpr,
    OFF_B: T.constexpr,
    OFF_C: T.constexpr,
    OFF_OUT: T.constexpr,
    OFF_SCALE: T.constexpr,
    SHARED_BYTES: T.constexpr,
):
    state = T.match_buffer(state_h, (STATE_ELEMENTS,), STATE_DTYPE, scope="global")
    state_scale = T.match_buffer(state_scale_h, (SCALE_ELEMENTS,), "float32", scope="global")
    x = T.match_buffer(x_h, (X_ELEMENTS,), "bfloat16", scope="global")
    dt = T.match_buffer(dt_h, (DT_ELEMENTS,), WEIGHT_DTYPE, scope="global")
    matrix_a = T.match_buffer(matrix_a_h, (NHEADS,), "float32", scope="global")
    matrix_b = T.match_buffer(matrix_b_h, (BC_ELEMENTS,), "bfloat16", scope="global")
    matrix_c = T.match_buffer(matrix_c_h, (BC_ELEMENTS,), "bfloat16", scope="global")
    d_weight = T.match_buffer(d_h, (NHEADS,), WEIGHT_DTYPE, scope="global")
    z = T.match_buffer(z_h, (X_ELEMENTS,), "bfloat16", scope="global")
    dt_bias = T.match_buffer(dt_bias_h, (NHEADS,), WEIGHT_DTYPE, scope="global")
    state_indices = T.match_buffer(state_indices_h, (INDEX_ELEMENTS,), INDEX_DTYPE, scope="global")
    dst_indices = T.match_buffer(dst_indices_h, (INDEX_ELEMENTS,), INDEX_DTYPE, scope="global")
    rand_seed = T.match_buffer(rand_seed_h, (1,), "int64", scope="global")
    output = T.match_buffer(output_h, (X_ELEMENTS,), "bfloat16", scope="global")
    T.device_entry()
    # TIRX_TRANSCRIBE_START selective_state_update_stp_simple

    random_seed: T.int64 = 0
    if PHILOX_ROUNDS > 0 and not SCALE_STATE:
        random_seed = rand_seed[0]

    batch_i, head, dim_tile = T.cta_id([BATCH, NHEADS, dim_tiles_runtime])
    lane_axis, warp = T.thread_id([32, 4])
    lane: T.int32 = _lane_mask(lane_axis)
    dim_offset: T.int32 = dim_tile * ROWS_PER_BLOCK
    group: T.int32 = head // (nheads_runtime // ngroups_runtime)
    rows_per_warp: T.int32 = (ROWS_PER_BLOCK + 3) // 4

    state_batch: T.int64
    if HAS_STATE_INDICES:
        state_batch = T.cast(state_indices[batch_i * state_indices_stride_batch], "int64")
    else:
        state_batch = T.cast(batch_i, "int64")
    dst_state_batch: T.int64
    if HAS_DST_INDICES:
        dst_state_batch = T.cast(dst_indices[batch_i * dst_indices_stride_batch], "int64")
    else:
        dst_state_batch = state_batch

    state_head_offset: T.int64 = state_batch * state_stride_batch + T.cast(
        head * DIM * DSTATE, "int64"
    )
    dst_state_head_offset: T.int64 = dst_state_batch * state_stride_batch + T.cast(
        head * DIM * DSTATE, "int64"
    )
    scale_head_offset: T.int64 = state_batch * state_scale_stride_batch + T.cast(
        head * DIM, "int64"
    )
    dst_scale_head_offset: T.int64 = dst_state_batch * state_scale_stride_batch + T.cast(
        head * DIM, "int64"
    )

    shared_raw = T.alloc_buffer((SHARED_BYTES,), "uint8", scope="shared", align=16)
    s_x = T.decl_buffer(
        (ROWS_PER_BLOCK,),
        "bfloat16",
        data=shared_raw.data,
        scope="shared",
        byte_offset=OFF_X,
        align=16,
    )
    s_z = T.decl_buffer(
        (ROWS_PER_BLOCK,),
        "bfloat16",
        data=shared_raw.data,
        scope="shared",
        byte_offset=OFF_Z,
        align=16,
    )
    s_b = T.decl_buffer(
        (DSTATE,), "bfloat16", data=shared_raw.data, scope="shared", byte_offset=OFF_B, align=16
    )
    s_c = T.decl_buffer(
        (DSTATE,), "bfloat16", data=shared_raw.data, scope="shared", byte_offset=OFF_C, align=16
    )
    s_out = T.decl_buffer(
        (ROWS_PER_BLOCK,),
        "float32",
        data=shared_raw.data,
        scope="shared",
        byte_offset=OFF_OUT,
        align=4,
    )
    s_scale = T.decl_buffer(
        (ROWS_PER_BLOCK,),
        "float32",
        data=shared_raw.data,
        scope="shared",
        byte_offset=OFF_SCALE,
        align=16,
    )

    a_value: T.float32 = T.reinterpret("float32", _global_load_u32(matrix_a, head))
    dt_value: T.float32 = _load_weight(
        dt, T.cast(batch_i, "int64") * dt_stride_batch + head, WEIGHT_DTYPE
    )
    if HAS_DT_BIAS:
        bias_value: T.float32 = _load_weight(dt_bias, head, WEIGHT_DTYPE)
        dt_value = _add(dt_value, bias_value)
    if dt_softplus != 0:
        if dt_value <= T.float32(20.0):
            softplus_exp_arg: T.float32 = _mul(dt_value, T.float32(_LOG2_E))
            softplus_exp: T.float32 = _exp2(softplus_exp_arg)
            softplus_sum: T.float32 = _add(T.float32(1.0), softplus_exp)
            softplus_log2: T.float32 = _log2(softplus_sum)
            dt_value = _mul(softplus_log2, T.float32(_LN_2))
    a_dt: T.float32 = _mul(a_value, dt_value)
    da_exp_arg: T.float32 = _mul(a_dt, T.float32(_LOG2_E))
    da_value: T.float32 = _exp2(da_exp_arg)
    d_value: T.float32 = 0.0
    if HAS_D:
        d_value = _load_weight(d_weight, head, WEIGHT_DTYPE)

    if warp == 0:
        for preload_iter in T.serial((ROWS_PER_BLOCK + 31) // 32):
            local_row: T.int32 = lane + preload_iter * 32
            row_d: T.int32 = dim_offset + local_row
            if local_row < ROWS_PER_BLOCK and row_d < DIM:
                x_bits: T.uint16 = _global_load_u16(
                    x, T.cast(batch_i, "int64") * x_stride_batch + head * DIM + row_d
                )
                T.evaluate(T.ptx.st.shared.b16(s_x.ptr_to([local_row]), x_bits))
        if SCALE_STATE:
            for scale_iter in T.serial((ROWS_PER_BLOCK + 31) // 32):
                local_row: T.int32 = lane + scale_iter * 32
                row_d: T.int32 = dim_offset + local_row
                if local_row < ROWS_PER_BLOCK and row_d < DIM:
                    scale_bits: T.uint32 = _global_load_u32(state_scale, scale_head_offset + row_d)
                    T.evaluate(T.ptx.st.shared.b32(s_scale.ptr_to([local_row]), scale_bits))
    elif warp == 1:
        bc_i: T.int32 = lane * 8
        if bc_i < DSTATE:
            b_words = T.alloc_local((4,), "uint32")
            T.evaluate(
                T.ptx.ld.global_.v4.b32(
                    b_words[0],
                    b_words[1],
                    b_words[2],
                    b_words[3],
                    matrix_b.ptr_to(
                        [T.cast(batch_i, "int64") * b_stride_batch + group * DSTATE + bc_i]
                    ),
                )
            )
            T.evaluate(
                T.ptx.st.shared.v4.b32(
                    s_b.ptr_to([bc_i]), b_words[0], b_words[1], b_words[2], b_words[3]
                )
            )
    elif warp == 2:
        if HAS_Z:
            for preload_iter in T.serial((ROWS_PER_BLOCK + 31) // 32):
                local_row: T.int32 = lane + preload_iter * 32
                row_d: T.int32 = dim_offset + local_row
                if local_row < ROWS_PER_BLOCK and row_d < DIM:
                    z_bits: T.uint16 = _global_load_u16(
                        z, T.cast(batch_i, "int64") * z_stride_batch + head * DIM + row_d
                    )
                    T.evaluate(T.ptx.st.shared.b16(s_z.ptr_to([local_row]), z_bits))
        else:
            for preload_iter in T.serial((ROWS_PER_BLOCK + 31) // 32):
                local_row: T.int32 = lane + preload_iter * 32
                row_d: T.int32 = dim_offset + local_row
                if local_row < ROWS_PER_BLOCK and row_d < DIM:
                    T.evaluate(T.ptx.st.shared.b16(s_z.ptr_to([local_row]), T.uint16(0)))
    else:
        bc_i: T.int32 = lane * 8
        if bc_i < DSTATE:
            c_words = T.alloc_local((4,), "uint32")
            T.evaluate(
                T.ptx.ld.global_.v4.b32(
                    c_words[0],
                    c_words[1],
                    c_words[2],
                    c_words[3],
                    matrix_c.ptr_to(
                        [T.cast(batch_i, "int64") * c_stride_batch + group * DSTATE + bc_i]
                    ),
                )
            )
            T.evaluate(
                T.ptx.st.shared.v4.b32(
                    s_c.ptr_to([bc_i]), c_words[0], c_words[1], c_words[2], c_words[3]
                )
            )
    T.cuda.cta_sync()

    for row_in_warp in T.serial(rows_per_warp):
        local_row: T.int32 = warp * rows_per_warp + row_in_warp
        row_d: T.int32 = dim_offset + local_row
        if row_d < DIM:
            x_value: T.float32 = _bf16_to_f32(_shared_load_u16(s_x, local_row))
            decode_scale: T.float32 = 1.0
            new_state_max: T.float32 = T.float32(_FLT_LOWEST)
            if SCALE_STATE:
                decode_scale = T.reinterpret("float32", _shared_load_u32(s_scale, local_row))
            d_times_x: T.float32 = _mul(d_value, x_value)
            out_value: T.float32 = T.if_then_else(lane == 0, d_times_x, T.float32(0.0))
            new_states = T.alloc_local((LANE_STATE_COUNT,), "float32")
            for state_iter in T.unroll(STATE_ITERATIONS):
                state_i: T.int32 = (state_iter * 32 + lane) * STATE_VECTOR
                if STATE_BYTES == 2:
                    r_state = T.alloc_local((STATE_VECTOR,), "uint16")
                    for e in T.unroll(STATE_VECTOR):
                        r_state[e] = T.uint16(0)
                    if state_batch != T.cast(pad_slot_id, "int64"):
                        loaded_state = _load_two_byte_vector(
                            state,
                            state_head_offset + row_d * DSTATE + state_i,
                            STATE_VECTOR,
                            "global",
                        )
                        for e in T.unroll(STATE_VECTOR):
                            r_state[e] = loaded_state[e]
                else:
                    r_state = T.alloc_local((STATE_VECTOR,), "uint32")
                    for e in T.unroll(STATE_VECTOR):
                        r_state[e] = T.uint32(0)
                    if state_batch != T.cast(pad_slot_id, "int64"):
                        loaded_state = _load_f32_vector(
                            state, state_head_offset + row_d * DSTATE + state_i, STATE_VECTOR
                        )
                        for e in T.unroll(STATE_VECTOR):
                            r_state[e] = loaded_state[e]

                b_bits = T.alloc_local((STATE_VECTOR,), "uint16")
                c_bits = T.alloc_local((STATE_VECTOR,), "uint16")
                random_words = T.alloc_local((4,), "uint32")
                sr_raw = T.alloc_local((STATE_VECTOR,), "uint32")
                for e in T.unroll(STATE_VECTOR):
                    if PHILOX_ROUNDS > 0 and not SCALE_STATE and e % 4 == 0:
                        random_offset: T.uint64 = T.cast(
                            state_head_offset + row_d * DSTATE + state_i + e, "uint64"
                        )
                        c0: T.uint32 = T.cast(random_offset, "uint32")
                        c1: T.uint32 = T.cast(T.shift_right(random_offset, T.uint64(32)), "uint32")
                        c2: T.uint32 = 0
                        c3: T.uint32 = 0
                        k0: T.uint32 = T.cast(T.reinterpret("uint64", random_seed), "uint32")
                        k1: T.uint32 = T.cast(
                            T.shift_right(T.reinterpret("uint64", random_seed), T.uint64(32)),
                            "uint32",
                        )
                        for philox_round in T.unroll(10):
                            old_c0: T.uint32 = c0
                            old_c2: T.uint32 = c2
                            hi_b: T.uint32 = _mul_hi_u32(T.uint32(0xCD9E8D57), old_c2)
                            next_c0: T.uint32 = T.bitwise_xor(T.bitwise_xor(hi_b, c1), k0)
                            hi_a: T.uint32 = _mul_hi_u32(T.uint32(0xD2511F53), old_c0)
                            next_c2: T.uint32 = T.bitwise_xor(T.bitwise_xor(hi_a, c3), k1)
                            next_c1_s: T.int32 = _mul_lo_s32(
                                T.int32(-845247145), T.reinterpret("int32", old_c2)
                            )
                            next_c3_s: T.int32 = _mul_lo_s32(
                                T.int32(-766435501), T.reinterpret("int32", old_c0)
                            )
                            next_k0_s: T.int32 = _add_s32(
                                T.reinterpret("int32", k0), T.int32(-1640531527)
                            )
                            next_k1_s: T.int32 = _add_s32(
                                T.reinterpret("int32", k1), T.int32(-1150833019)
                            )
                            c0 = next_c0
                            c1 = T.reinterpret("uint32", next_c1_s)
                            c2 = next_c2
                            c3 = T.reinterpret("uint32", next_c3_s)
                            k0 = T.reinterpret("uint32", next_k0_s)
                            k1 = T.reinterpret("uint32", next_k1_s)
                        random_words[0] = c0
                        random_words[1] = c1
                        random_words[2] = c2
                        random_words[3] = c3

                    state_value: T.float32 = _state_bits_to_f32(r_state[e], STATE_DTYPE)
                    if SCALE_STATE:
                        state_value = _mul(state_value, decode_scale)
                    if STATE_VECTOR == 3:
                        b_bits[e] = _shared_load_u16(s_b, state_i + e)
                        b_value: T.float32 = _bf16_to_f32(b_bits[e])
                        c_bits[e] = _shared_load_u16(s_c, state_i + e)
                        c_value: T.float32 = _bf16_to_f32(c_bits[e])
                    else:
                        if e == 0:
                            loaded_b = _load_two_byte_vector(s_b, state_i, STATE_VECTOR, "shared")
                            for copy_e in T.unroll(STATE_VECTOR):
                                b_bits[copy_e] = loaded_b[copy_e]
                        b_value: T.float32 = _bf16_to_f32(b_bits[e])
                        if e == 0:
                            loaded_c = _load_two_byte_vector(s_c, state_i, STATE_VECTOR, "shared")
                            for copy_e in T.unroll(STATE_VECTOR):
                                c_bits[copy_e] = loaded_c[copy_e]
                        c_value: T.float32 = _bf16_to_f32(c_bits[e])

                    db_value: T.float32 = _mul(b_value, dt_value)
                    db_x: T.float32 = _mul(db_value, x_value)
                    new_state: T.float32 = _fma(state_value, da_value, db_x)
                    if SCALE_STATE:
                        magnitude: T.float32 = _abs(new_state)
                        new_state_max = _max(new_state_max, magnitude)
                        new_states[state_iter * STATE_VECTOR + e] = new_state
                    elif PHILOX_ROUNDS > 0:
                        random13: T.uint32 = T.bitwise_and(random_words[e % 4], T.uint32(0x1FFF))
                        T.evaluate(
                            T.ptx.cvt.rs.f16x2.f32(sr_raw[e], T.float32(0.0), new_state, random13)
                        )
                    else:
                        r_state[e] = _f32_to_state_bits(new_state, STATE_DTYPE)
                    out_value = _fma(new_state, c_value, out_value)

                if (
                    not SCALE_STATE
                    and update_state != 0
                    and state_batch != T.cast(pad_slot_id, "int64")
                ):
                    if PHILOX_ROUNDS > 0:
                        sr_words = T.alloc_local((STATE_VECTOR // 2,), "uint32")
                        for pair in T.unroll(STATE_VECTOR // 2):
                            sr_words[pair] = _prmt_5410(sr_raw[2 * pair], sr_raw[2 * pair + 1])
                        if STATE_VECTOR == 2:
                            T.evaluate(
                                T.ptx.st.global_.b32(
                                    state.ptr_to(
                                        [dst_state_head_offset + row_d * DSTATE + state_i]
                                    ),
                                    sr_words[0],
                                )
                            )
                        else:
                            T.evaluate(
                                T.ptx.st.global_.v2.b32(
                                    state.ptr_to(
                                        [dst_state_head_offset + row_d * DSTATE + state_i]
                                    ),
                                    sr_words[0],
                                    sr_words[1],
                                )
                            )
                    elif STATE_BYTES == 2:
                        _store_two_byte_vector(
                            state,
                            dst_state_head_offset + row_d * DSTATE + state_i,
                            r_state,
                            STATE_VECTOR,
                        )
                    else:
                        _store_f32_vector(
                            state,
                            dst_state_head_offset + row_d * DSTATE + state_i,
                            r_state,
                            STATE_VECTOR,
                        )

            for delta_i in T.unroll(5):
                delta: T.int32 = T.shift_right(T.int32(16), delta_i)
                peer_out: T.float32 = T.cuda.__shfl_down_sync(
                    T.uint32(0xFFFFFFFF), out_value, delta, 32
                )
                out_value = _add(out_value, peer_out)
            if lane == 0:
                T.evaluate(
                    T.ptx.st.shared.b32(
                        s_out.ptr_to([local_row]), T.reinterpret("uint32", out_value)
                    )
                )

            if SCALE_STATE and update_state != 0 and state_batch != T.cast(pad_slot_id, "int64"):
                for delta_i in T.unroll(5):
                    delta: T.int32 = T.shift_right(T.int32(16), delta_i)
                    peer_max: T.float32 = T.cuda.__shfl_down_sync(
                        T.uint32(0xFFFFFFFF), new_state_max, delta, 32
                    )
                    new_state_max = _max(new_state_max, peer_max)
                T.cuda.warp_sync()
                new_state_max = T.cuda.__shfl_sync(T.uint32(0xFFFFFFFF), new_state_max, 0, 32)
                encode_scale: T.float32 = 1.0
                if new_state_max != T.float32(0.0):
                    encode_scale = _div(T.float32(32767.0), new_state_max)
                new_decode_scale: T.float32 = _rcp(encode_scale)
                for state_iter in T.unroll(STATE_ITERATIONS):
                    state_i: T.int32 = (state_iter * 32 + lane) * STATE_VECTOR
                    quantized = T.alloc_local((STATE_VECTOR,), "int32")
                    packed_quantized = T.alloc_local((STATE_VECTOR // 2,), "uint32")
                    for e in T.unroll(STATE_VECTOR):
                        scaled: T.float32 = _mul(
                            new_states[state_iter * STATE_VECTOR + e], encode_scale
                        )
                        clipped_low: T.float32 = _max(scaled, T.float32(-32767.0))
                        clipped: T.float32 = _min(clipped_low, T.float32(32767.0))
                        T.evaluate(T.ptx.cvt.rni.ftz.s32.f32(quantized[e], clipped))
                    for pair in T.unroll(STATE_VECTOR // 2):
                        packed_quantized[pair] = _prmt_5410(
                            T.reinterpret("uint32", quantized[2 * pair]),
                            T.reinterpret("uint32", quantized[2 * pair + 1]),
                        )
                    if STATE_VECTOR == 2:
                        T.evaluate(
                            T.ptx.st.global_.b32(
                                state.ptr_to([dst_state_head_offset + row_d * DSTATE + state_i]),
                                packed_quantized[0],
                            )
                        )
                    elif STATE_VECTOR == 4:
                        T.evaluate(
                            T.ptx.st.global_.v2.b32(
                                state.ptr_to([dst_state_head_offset + row_d * DSTATE + state_i]),
                                packed_quantized[0],
                                packed_quantized[1],
                            )
                        )
                    else:
                        T.evaluate(
                            T.ptx.st.global_.v4.b32(
                                state.ptr_to([dst_state_head_offset + row_d * DSTATE + state_i]),
                                packed_quantized[0],
                                packed_quantized[1],
                                packed_quantized[2],
                                packed_quantized[3],
                            )
                        )
                if lane == 0:
                    T.evaluate(
                        T.ptx.st.shared.b32(
                            s_scale.ptr_to([local_row]), T.reinterpret("uint32", new_decode_scale)
                        )
                    )

    T.cuda.cta_sync()
    for output_iter in T.serial((ROWS_PER_BLOCK + 127) // 128):
        row_in_warp: T.int32 = lane + output_iter * 32
        local_row: T.int32 = warp * rows_per_warp + row_in_warp
        row_d: T.int32 = dim_offset + local_row
        if row_in_warp < rows_per_warp and row_d < DIM:
            out_value: T.float32 = T.reinterpret("float32", _shared_load_u32(s_out, local_row))
            if HAS_Z:
                z_value: T.float32 = _bf16_to_f32(_shared_load_u16(s_z, local_row))
                neg_z: T.float32 = _sub(T.float32(0.0), z_value)
                z_exp_arg: T.float32 = _mul(neg_z, T.float32(_LOG2_E))
                exp_neg_z: T.float32 = _exp2(z_exp_arg)
                denominator: T.float32 = _add(T.float32(1.0), exp_neg_z)
                sigmoid_z: T.float32 = _div(T.float32(1.0), denominator)
                silu_z: T.float32 = _mul(z_value, sigmoid_z)
                out_value = _mul(out_value, silu_z)
            output_bits: T.uint16 = _f32_to_bf16(out_value)
            T.evaluate(
                T.ptx.st.global_.b16(
                    output.ptr_to(
                        [T.cast(batch_i, "int64") * out_stride_batch + head * DIM + row_d]
                    ),
                    output_bits,
                )
            )
    if SCALE_STATE and update_state != 0 and state_batch != T.cast(pad_slot_id, "int64"):
        for scale_iter in T.serial((ROWS_PER_BLOCK + 127) // 128):
            row_in_warp: T.int32 = lane + scale_iter * 32
            local_row: T.int32 = warp * rows_per_warp + row_in_warp
            row_d: T.int32 = dim_offset + local_row
            if row_in_warp < rows_per_warp and row_d < DIM:
                scale_bits: T.uint32 = _shared_load_u32(s_scale, local_row)
                T.evaluate(
                    T.ptx.st.global_.b32(
                        state_scale.ptr_to([dst_scale_head_offset + row_d]), scale_bits
                    )
                )


def _num_sms(device: str | torch.device = "cuda") -> int:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(device).multi_processor_count
    return 148


def _specialization(kwargs: dict[str, Any]) -> dict[str, Any]:
    batch = int(kwargs["batch"])
    nheads = int(kwargs["nheads"])
    dim = int(kwargs["dim"])
    dstate = int(kwargs["dstate"])
    ngroups = int(kwargs["ngroups"])
    state_dtype = str(kwargs["state_dtype"])
    state_stride_factor = int(kwargs.get("state_stride_factor", 1))
    has_dst_indices = bool(kwargs.get("has_dst_indices", False))
    state_slots = max((2 if has_dst_indices else 1) * batch + 8, 16)
    state_stride = nheads * dim * dstate * state_stride_factor
    scale_stride = nheads * dim
    index_elements = batch * (2 if int(kwargs.get("index_rank", 1)) == 2 else 1)
    rows_per_block = 4 if batch * nheads < 2 * _num_sms(kwargs.get("device", "cuda")) else dim
    scale_state = state_dtype == "int16"
    state_bytes = 4 if state_dtype == "float32" else 2
    state_vector = min(16 // state_bytes, dstate // 32)
    off_x = 0
    off_z = _align_up(2 * rows_per_block, 16)
    off_b = _align_up(off_z + 2 * rows_per_block, 16)
    off_c = _align_up(off_b + 2 * dstate, 16)
    off_out = off_c + 2 * dstate
    off_scale = _align_up(off_out + 4 * rows_per_block, 16)
    scale_tail_bytes = 4 * rows_per_block if scale_state else rows_per_block
    philox_rounds = int(kwargs.get("philox_rounds", 0))
    if scale_state and dstate not in (64, 128, 256):
        raise ValueError("int16 simple specializations require dstate in {64, 128, 256}")
    if philox_rounds not in (0, 10):
        raise ValueError("simple stochastic rounding supports philox_rounds in {0, 10}")
    if philox_rounds and (state_dtype != "float16" or dstate not in (64, 128)):
        raise ValueError("philox10 is scoped to float16 state with dstate 64 or 128")
    return {
        "BATCH": batch,
        "NHEADS": nheads,
        "DIM": dim,
        "DSTATE": dstate,
        "ROWS_PER_BLOCK": rows_per_block,
        "STATE_DTYPE": state_dtype,
        "WEIGHT_DTYPE": str(kwargs["weight_dtype"]),
        "INDEX_DTYPE": str(kwargs["index_dtype"]),
        "STATE_ELEMENTS": state_slots * state_stride,
        "SCALE_ELEMENTS": state_slots * scale_stride if scale_state else 1,
        "X_ELEMENTS": batch * nheads * dim,
        "DT_ELEMENTS": batch * nheads,
        "BC_ELEMENTS": batch * ngroups * dstate,
        "INDEX_ELEMENTS": max(index_elements, 1),
        "HAS_STATE_INDICES": bool(kwargs.get("has_state_indices", True)),
        "HAS_DST_INDICES": has_dst_indices,
        "HAS_Z": bool(kwargs.get("has_z", False)),
        "HAS_D": bool(kwargs.get("has_d", True)),
        "HAS_DT_BIAS": bool(kwargs.get("has_dt_bias", True)),
        "SCALE_STATE": scale_state,
        "PHILOX_ROUNDS": philox_rounds,
        "STATE_BYTES": state_bytes,
        "STATE_VECTOR": state_vector,
        "STATE_ITERATIONS": dstate // (32 * state_vector),
        "LANE_STATE_COUNT": dstate // 32,
        "OFF_X": off_x,
        "OFF_Z": off_z,
        "OFF_B": off_b,
        "OFF_C": off_c,
        "OFF_OUT": off_out,
        "OFF_SCALE": off_scale,
        "SHARED_BYTES": _align_up(off_scale + scale_tail_bytes, 16),
    }


def get_kernel(**kwargs: Any):
    """Return the source-shaped plain-TIRx simple specialization."""
    return _selective_state_update_stp_simple.specialize(**_specialization(kwargs))


_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}


@functools.cache
def _load_frozen_oracle():
    from flashinfer.jit.env import FLASHINFER_INCLUDE_DIR
    from flashinfer.mamba import selective_state_update

    source_path = (
        Path(FLASHINFER_INCLUDE_DIR) / "flashinfer/mamba/kernel_selective_state_update_stp.cuh"
    )
    if not source_path.is_file():
        raise RuntimeError(f"missing frozen FlashInfer source: {source_path}")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != FROZEN_FLASHINFER_SOURCE_SHA256:
        raise RuntimeError(
            "selective-state-update oracle does not match the reviewed source: "
            f"{source_path} sha256={digest}"
        )

    return selective_state_update


def _view_state(raw: torch.Tensor, spec: dict[str, Any], state_stride: int) -> torch.Tensor:
    return raw.as_strided(
        (spec["STATE_ELEMENTS"] // state_stride, spec["NHEADS"], spec["DIM"], spec["DSTATE"]),
        (state_stride, spec["DIM"] * spec["DSTATE"], spec["DSTATE"], 1),
    )


def _view_scale(raw: torch.Tensor, spec: dict[str, Any], scale_stride: int) -> torch.Tensor:
    return raw.as_strided(
        (spec["SCALE_ELEMENTS"] // scale_stride, spec["NHEADS"], spec["DIM"]),
        (scale_stride, spec["DIM"], 1),
    )


def _index_tensor(
    values: torch.Tensor, *, rank: int, total_elements: int, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if rank == 1:
        shaped = values.contiguous()
        return shaped, shaped.reshape(-1), 1
    shaped = torch.empty((values.numel(), 2), dtype=values.dtype, device=device)
    shaped[:, 0] = values
    shaped[:, 1] = values
    flat = shaped.reshape(-1)
    if flat.numel() != total_elements:
        raise AssertionError((flat.numel(), total_elements))
    return shaped, flat, 2


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Create independent mutable TIRx/source cases for one specialization."""
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for selective-state-update STP simple")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(f"STP simple SM100 requires compute capability 10.x, got {capability}")

    spec = _specialization(kwargs)
    batch = spec["BATCH"]
    nheads = spec["NHEADS"]
    dim = spec["DIM"]
    dstate = spec["DSTATE"]
    ngroups = int(kwargs["ngroups"])
    state_dtype = _TORCH_DTYPES[str(kwargs["state_dtype"])]
    weight_dtype = _TORCH_DTYPES[str(kwargs["weight_dtype"])]
    index_dtype = _TORCH_DTYPES[str(kwargs["index_dtype"])]
    state_stride = nheads * dim * dstate * int(kwargs.get("state_stride_factor", 1))
    scale_stride = nheads * dim
    state_slots = spec["STATE_ELEMENTS"] // state_stride
    generator = torch.Generator(device=device)
    generator.manual_seed(int(kwargs.get("seed", 0)) + 20260808)

    if state_dtype == torch.int16:
        logical_f32 = torch.randn(
            (state_slots, nheads, dim, dstate),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        amax = logical_f32.abs().amax(dim=-1)
        encode = torch.where(amax == 0, torch.ones_like(amax), 32767.0 / amax)
        quantized = (logical_f32 * encode[..., None]).round().clamp(-32767, 32767).to(torch.int16)
        initial_state_raw = torch.zeros(spec["STATE_ELEMENTS"], dtype=torch.int16, device=device)
        initial_state_view = _view_state(initial_state_raw, spec, state_stride)
        initial_state_view.copy_(quantized)
        initial_scale_raw = torch.zeros(spec["SCALE_ELEMENTS"], dtype=torch.float32, device=device)
        _view_scale(initial_scale_raw, spec, scale_stride).copy_(1.0 / encode)
        del logical_f32, quantized, amax, encode
    else:
        initial_state_raw = torch.randn(
            (spec["STATE_ELEMENTS"],), dtype=state_dtype, device=device, generator=generator
        )
        initial_scale_raw = torch.zeros((1,), dtype=torch.float32, device=device)

    x = torch.randn((batch, nheads, dim), dtype=torch.bfloat16, device=device, generator=generator)
    dt_base = torch.randn((batch, nheads), dtype=weight_dtype, device=device, generator=generator)
    dt_view = dt_base.as_strided((batch, nheads, dim), (nheads, 1, 0))
    matrix_a_base = (
        -torch.rand((nheads,), dtype=torch.float32, device=device, generator=generator) - 1.0
    )
    matrix_a_view = matrix_a_base.as_strided((nheads, dim, dstate), (1, 0, 0))
    matrix_b = torch.randn(
        (batch, ngroups, dstate), dtype=torch.bfloat16, device=device, generator=generator
    )
    matrix_c = torch.randn(
        (batch, ngroups, dstate), dtype=torch.bfloat16, device=device, generator=generator
    )
    d_base = torch.randn((nheads,), dtype=weight_dtype, device=device, generator=generator)
    if not bool(kwargs.get("has_d", True)):
        d_base.zero_()
    d_view = d_base.as_strided((nheads, dim), (1, 0))
    bias_base = torch.rand((nheads,), dtype=weight_dtype, device=device, generator=generator) - 4.0
    bias_view = bias_base.as_strided((nheads, dim), (1, 0))
    z = torch.randn((batch, nheads, dim), dtype=torch.bfloat16, device=device, generator=generator)

    rank = int(kwargs.get("index_rank", 1))
    if bool(kwargs.get("has_dst_indices", False)):
        state_values = torch.arange(batch, dtype=index_dtype, device=device)
        dst_values = torch.arange(batch, dtype=index_dtype, device=device) + batch
    else:
        state_values = torch.randperm(state_slots, device=device, generator=generator)[:batch].to(
            index_dtype
        )
        dst_values = state_values.clone()
    pad_every = int(kwargs.get("pad_every", 0))
    pad_slot_id = -1
    if pad_every:
        state_values[::pad_every] = pad_slot_id
    state_indices, state_indices_flat, state_index_stride = _index_tensor(
        state_values, rank=rank, total_elements=spec["INDEX_ELEMENTS"], device=device
    )
    dst_indices, dst_indices_flat, dst_index_stride = _index_tensor(
        dst_values, rank=rank, total_elements=spec["INDEX_ELEMENTS"], device=device
    )
    seed = torch.tensor([int(kwargs.get("seed", 0))], dtype=torch.int64, device=device)

    tirx_state_raw = initial_state_raw.clone()
    reference_state_raw = initial_state_raw.clone()
    tirx_scale_raw = initial_scale_raw.clone()
    reference_scale_raw = initial_scale_raw.clone()
    tirx_output = torch.empty((batch, nheads, dim), dtype=torch.bfloat16, device=device)
    reference_output = torch.empty_like(tirx_output)
    dummy_index = torch.zeros((spec["INDEX_ELEMENTS"],), dtype=index_dtype, device=device)

    case = {
        "kwargs": dict(kwargs),
        "spec": spec,
        "state_stride": state_stride,
        "scale_stride": scale_stride,
        "initial_state_raw": initial_state_raw,
        "initial_scale_raw": initial_scale_raw,
        "tirx_state_raw": tirx_state_raw,
        "reference_state_raw": reference_state_raw,
        "tirx_scale_raw": tirx_scale_raw,
        "reference_scale_raw": reference_scale_raw,
        "tirx_output": tirx_output,
        "reference_output": reference_output,
        "x": x,
        "dt_base": dt_base,
        "dt_view": dt_view,
        "matrix_a_base": matrix_a_base,
        "matrix_a_view": matrix_a_view,
        "matrix_b": matrix_b,
        "matrix_c": matrix_c,
        "d_base": d_base,
        "d_view": d_view,
        "bias_base": bias_base,
        "bias_view": bias_view,
        "z": z,
        "state_indices": state_indices,
        "state_indices_flat": state_indices_flat,
        "dst_indices": dst_indices,
        "dst_indices_flat": dst_indices_flat,
        "state_index_stride": state_index_stride,
        "dst_index_stride": dst_index_stride,
        "dummy_index": dummy_index,
        "seed": seed,
        "pad_slot_id": pad_slot_id,
    }
    return case


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    kwargs = case["kwargs"]
    spec = case["spec"]
    batch, nheads, dim = spec["BATCH"], spec["NHEADS"], spec["DIM"]
    ngroups, dstate = int(kwargs["ngroups"]), spec["DSTATE"]
    has_state_indices = bool(kwargs.get("has_state_indices", True))
    has_dst_indices = bool(kwargs.get("has_dst_indices", False))
    return (
        case["tirx_state_raw"],
        case["tirx_scale_raw"],
        case["x"].reshape(-1),
        case["dt_base"].reshape(-1),
        case["matrix_a_base"],
        case["matrix_b"].reshape(-1),
        case["matrix_c"].reshape(-1),
        case["d_base"],
        case["z"].reshape(-1),
        case["bias_base"],
        case["state_indices_flat"] if has_state_indices else case["dummy_index"],
        case["dst_indices_flat"] if has_dst_indices else case["dummy_index"],
        case["seed"],
        case["tirx_output"].reshape(-1),
        case["state_stride"],
        case["scale_stride"] if spec["SCALE_STATE"] else 0,
        nheads * dim,
        nheads,
        ngroups * dstate,
        ngroups * dstate,
        nheads * dim,
        nheads * dim,
        case["state_index_stride"] if has_state_indices else 1,
        case["dst_index_stride"] if has_dst_indices else 0,
        nheads,
        ngroups,
        int(bool(kwargs.get("dt_softplus", False))),
        int(bool(kwargs.get("update_state", True))),
        case["pad_slot_id"],
        (dim + spec["ROWS_PER_BLOCK"] - 1) // spec["ROWS_PER_BLOCK"],
    )


def _run_reference(case: dict[str, Any]) -> torch.Tensor:
    kwargs = case["kwargs"]
    spec = case["spec"]
    oracle = _load_frozen_oracle()
    state_view = _view_state(case["reference_state_raw"], spec, case["state_stride"])
    state_scale = (
        _view_scale(case["reference_scale_raw"], spec, case["scale_stride"])
        if spec["SCALE_STATE"]
        else None
    )
    source_out = case["reference_output"] if bool(kwargs.get("use_out_tensor", True)) else None
    result = oracle(
        state_view,
        case["x"],
        case["dt_view"],
        case["matrix_a_view"],
        case["matrix_b"],
        case["matrix_c"],
        case["d_view"],
        z=case["z"] if bool(kwargs.get("has_z", False)) else None,
        dt_bias=case["bias_view"] if bool(kwargs.get("has_dt_bias", True)) else None,
        dt_softplus=bool(kwargs.get("dt_softplus", False)),
        state_batch_indices=(
            case["state_indices"] if bool(kwargs.get("has_state_indices", True)) else None
        ),
        dst_state_batch_indices=(
            case["dst_indices"] if bool(kwargs.get("has_dst_indices", False)) else None
        ),
        pad_slot_id=case["pad_slot_id"],
        state_scale=state_scale,
        out=source_out,
        disable_state_update=not bool(kwargs.get("update_state", True)),
        rand_seed=case["seed"] if spec["PHILOX_ROUNDS"] else None,
        philox_rounds=spec["PHILOX_ROUNDS"],
        algorithm="simple",
    )
    if source_out is None:
        case["reference_output"].copy_(result)
    return result


def _written_slots(case: dict[str, Any]) -> list[int]:
    kwargs = case["kwargs"]
    batch = case["spec"]["BATCH"]
    if not bool(kwargs.get("update_state", True)):
        return []
    if bool(kwargs.get("has_state_indices", True)):
        read = case["state_indices"].reshape(batch, -1)[:, 0]
    else:
        read = torch.arange(batch, device=case["x"].device)
    if bool(kwargs.get("has_dst_indices", False)):
        dst = case["dst_indices"].reshape(batch, -1)[:, 0]
    else:
        dst = read
    valid = read != case["pad_slot_id"]
    return sorted({int(value) for value in dst[valid].tolist()})


def _assert_case_close(case: dict[str, Any]) -> None:
    kwargs = case["kwargs"]
    spec = case["spec"]
    for name, tensor in (
        ("TIRx output", case["tirx_output"]),
        ("FlashInfer output", case["reference_output"]),
    ):
        if not torch.isfinite(tensor.float()).all():
            raise AssertionError(f"{name} contains non-finite values")
    atol = 0.1 if spec["SCALE_STATE"] else 2e-2
    rtol = 1e-2 if spec["SCALE_STATE"] else 2e-2
    torch.testing.assert_close(case["tirx_output"], case["reference_output"], atol=atol, rtol=rtol)

    tirx_state = _view_state(case["tirx_state_raw"], spec, case["state_stride"])
    reference_state = _view_state(case["reference_state_raw"], spec, case["state_stride"])
    slots = _written_slots(case)
    if slots:
        slot_index = torch.tensor(slots, dtype=torch.int64, device=tirx_state.device)
        tirx_rows = tirx_state.index_select(0, slot_index)
        reference_rows = reference_state.index_select(0, slot_index)
        if spec["SCALE_STATE"]:
            tirx_scale = _view_scale(case["tirx_scale_raw"], spec, case["scale_stride"])
            reference_scale = _view_scale(case["reference_scale_raw"], spec, case["scale_stride"])
            tirx_scale_rows = tirx_scale.index_select(0, slot_index)
            reference_scale_rows = reference_scale.index_select(0, slot_index)
            torch.testing.assert_close(tirx_scale_rows, reference_scale_rows, atol=2e-5, rtol=2e-4)
            tirx_rows = tirx_rows.float() * tirx_scale_rows[..., None]
            reference_rows = reference_rows.float() * reference_scale_rows[..., None]
            torch.testing.assert_close(tirx_rows, reference_rows, atol=0.1, rtol=1e-2)
        else:
            state_atol = 2e-3 if spec["STATE_DTYPE"] == "float32" else 2e-2
            torch.testing.assert_close(tirx_rows, reference_rows, atol=state_atol, rtol=2e-2)
    elif not bool(kwargs.get("update_state", True)):
        torch.testing.assert_close(
            case["tirx_state_raw"], case["initial_state_raw"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            case["reference_state_raw"], case["initial_state_raw"], atol=0, rtol=0
        )


def run_test(**kwargs: Any) -> None:
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    executable = compile_kernel(get_kernel(**kwargs))
    executable(*_tirx_args(case))
    _run_reference(case)
    torch.cuda.synchronize()
    _assert_case_close(case)


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    rounds = int(kwargs.pop("rounds", 5))
    cooldown_s = float(kwargs.pop("cooldown_s", 1.0))
    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench

    case = prepare_data(**kwargs)
    executable = compile_kernel(get_kernel(**kwargs))
    args = _tirx_args(case)
    executable(*args)
    _run_reference(case)
    torch.cuda.synchronize()
    _assert_case_close(case)

    def source_builder():
        for _ in range(2):
            _run_reference(case)
        torch.cuda.synchronize()

        def launch():
            _run_reference(case)

        return launch

    return bench(
        {"tirx": lambda: executable(*args)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashinfer_cuda": source_builder},
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
