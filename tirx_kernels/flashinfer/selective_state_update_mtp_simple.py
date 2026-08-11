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
"""TIRx port of FlashInfer's selective-state-update MTP simple kernel."""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any
from unittest import SkipTest

import torch

from tvm.script import tirx as T

KERNEL_META = {
    "name": "selective_state_update_mtp_simple",
    "category": "flashinfer",
    "compute_capability": 10,
}

FROZEN_FLASHINFER_COMMIT = "f2e04400e330fb2debe0bf8730d9424a1d37927f"
FROZEN_FLASHINFER_SOURCE_SHA256 = "64b189b642d05970202964bdd3accf7055d34d6a50b7953a64c8a05a3e0a0dbe"
_LOG2_E = 1.4426950408889634
_LN_2 = 0.6931471805599453
_FLT_LOWEST = -3.4028234663852886e38


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


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


def _extract_u16(word, high: bool):
    if high:
        return T.cast(T.shift_right(word, T.uint32(16)), "uint16")
    return T.cast(T.bitwise_and(word, T.uint32(0xFFFF)), "uint16")


def _bf16_word_to_f32x2(word):
    low_bits = T.shift_left(word, T.uint32(16))
    high_bits = T.bitwise_and(word, T.uint32(0xFFFF0000))
    return T.cuda.make_float2(
        T.reinterpret("float32", low_bits), T.reinterpret("float32", high_bits)
    )


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "label": label,
        "batch": 64,
        "nheads": 64,
        "dim": 64,
        "dstate": 128,
        "tokens": 4,
        "heads_per_group": 8,
        "input_dtype": "bfloat16",
        "state_dtype": "bfloat16",
        "weight_dtype": "float32",
        "matrix_a_dtype": "float32",
        "index_dtype": "int64",
        "index_rank": 1,
        "cu_seqlens_dtype": "int32",
        "accepted_dtype": "int64",
        "mode": "fixed",
        "has_state_indices": True,
        "has_dst_indices": False,
        "has_intermediate_states": False,
        "has_num_accepted_tokens": False,
        "has_z": False,
        "has_d": True,
        "has_dt_bias": True,
        "dt_softplus": True,
        "update_state": True,
        "state_stride_factor": 1,
        "pad_every": 0,
        "use_out_tensor": True,
        "philox_rounds": 0,
        "shared_state_slot": False,
        "seed": 0,
    }
    config.update(overrides)
    return config


# FlashInfer's official MTP sweep: powers-of-two batch sizes through 2048,
# T=6, and BF16/FP32 state.  State update is disabled and all requests share a
# read-only cache slot.  Storage still contains the API-required batch number
# of slots, while every index points at slot zero.
BENCH_CONFIGS = [
    _case(
        f"b{batch}_h64_d64_s128_t6_r8_state{state_tag}_official",
        batch=batch,
        tokens=6,
        state_dtype=state_dtype,
        update_state=False,
        shared_state_slot=True,
    )
    for state_tag, state_dtype in (("bf16", "bfloat16"), ("f32", "float32"))
    for batch in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
] + [
    _case("b64_h64_d64_s128_t4_r8_update"),
    _case("b64_h64_d64_s128_t1_r8", tokens=1),
    _case("b64_h64_d64_s128_t2_r8", tokens=2),
    _case("b64_h64_d64_s128_t8_r8", tokens=8),
    _case("b64_h64_d128_s128_t4_r8", dim=128),
    _case("b64_h64_d64_s64_t4_r8", dstate=64),
    _case("b64_h64_d64_s96_t4_r8", dstate=96),
    _case("b64_h64_d64_s128_t4_r1", heads_per_group=1),
    _case("b64_h64_d64_s128_t4_r16", heads_per_group=16),
    _case("b64_h64_d64_s128_t4_r64", heads_per_group=64),
    _case("b64_h64_d64_s128_t4_r8_statef16", state_dtype="float16"),
    _case("b64_h64_d64_s128_t4_r8_weightbf16", weight_dtype="bfloat16"),
    _case("b64_h64_d64_s128_t4_r8_indices_i32", index_dtype="int32"),
    _case("b64_h64_d64_s128_t4_r8_intermediate", has_intermediate_states=True, update_state=False),
    _case("b64_h64_d64_s128_t4_r8_z", has_z=True),
    _case("b64_h64_d64_s128_t4_r8_int16", state_dtype="int16"),
    _case("b64_h64_d64_s128_t4_r8_philox10", state_dtype="float16", philox_rounds=10, seed=42),
    _case(
        "n8_h64_d64_s128_t4_r8_varlen_uniform",
        batch=8,
        mode="varlen_uniform",
        has_dst_indices=True,
        has_num_accepted_tokens=True,
        index_dtype="int32",
        index_rank=2,
    ),
    _case(
        "n8_h64_d64_s128_t6_r8_varlen_variable",
        batch=8,
        tokens=6,
        mode="varlen_variable",
        has_dst_indices=True,
        has_num_accepted_tokens=True,
        index_dtype="int32",
        index_rank=2,
    ),
]


# Correctness is an explicit one-variable-at-a-time matrix.  Rejection cases
# for the other algorithms live in their modules because every row here is in
# the simple kernel's real dispatch domain.
CONFIGS = [
    _case("b64_h64_d64_s128_t4_r8_base"),
    *[_case(f"b{batch}_h64_d64_s128_t4_r8", batch=batch) for batch in (1, 4, 16, 32, 256)],
    *[_case(f"b64_h64_d64_s128_t{tokens}_r8", tokens=tokens) for tokens in (1, 2, 6, 8)],
    _case("b64_h64_d128_s128_t4_r8", dim=128),
    _case("b64_h64_d64_s64_t4_r8", dstate=64),
    _case("b64_h64_d64_s96_t4_r8", dstate=96),
    *[
        _case(f"b64_h64_d64_s128_t4_r{ratio}", heads_per_group=ratio)
        for ratio in (1, 2, 4, 16, 32, 64)
    ],
    _case("b64_h64_d64_s128_t4_r8_statef16", state_dtype="float16"),
    _case("b64_h64_d64_s128_t4_r8_statef32", state_dtype="float32"),
    _case("b64_h64_d64_s128_t4_r8_weightbf16", weight_dtype="bfloat16"),
    _case("b64_h64_d64_s128_t4_r8_indices_i32", index_dtype="int32"),
    _case("b64_h64_d64_s128_t4_r8_dst1d", has_dst_indices=True, index_dtype="int32"),
    _case("b64_h64_d64_s128_t4_r8_dst2d", has_dst_indices=True, index_dtype="int32", index_rank=2),
    _case("b64_h64_d64_s128_t4_r8_pad4", pad_every=4, index_dtype="int32"),
    _case("b64_h64_d64_s128_t4_r8_stride2", state_stride_factor=2),
    _case("b64_h64_d64_s128_t4_r8_z", has_z=True),
    _case("b64_h64_d64_s128_t4_r8_no_d", has_d=False),
    _case("b64_h64_d64_s128_t4_r8_no_dt_bias", has_dt_bias=False),
    _case("b64_h64_d64_s128_t4_r8_no_softplus", dt_softplus=False),
    _case("b64_h64_d64_s128_t4_r8_no_update", update_state=False),
    _case("b64_h64_d64_s128_t4_r8_out_allocated", use_out_tensor=False),
    _case("b64_h64_d64_s128_t4_r8_intermediate", has_intermediate_states=True, update_state=False),
    _case("b64_h64_d64_s128_t4_r8_int16", state_dtype="int16"),
    _case(
        "b64_h64_d64_s128_t4_r8_int16_intermediate",
        state_dtype="int16",
        has_intermediate_states=True,
        update_state=False,
    ),
    _case("b64_h64_d64_s128_t4_r8_philox10", state_dtype="float16", philox_rounds=10, seed=42),
    _case(
        "b64_h64_d64_s128_t4_r8_philox10_intermediate",
        state_dtype="float16",
        philox_rounds=10,
        has_intermediate_states=True,
        update_state=False,
        seed=42,
    ),
    _case(
        "n4_h64_d64_s128_t4_r8_varlen_uniform",
        batch=4,
        mode="varlen_uniform",
        has_dst_indices=True,
        has_num_accepted_tokens=True,
        index_dtype="int32",
        index_rank=2,
    ),
    _case(
        "n8_h64_d64_s128_t6_r8_varlen_variable",
        batch=8,
        tokens=6,
        mode="varlen_variable",
        has_dst_indices=True,
        has_num_accepted_tokens=True,
        index_dtype="int32",
        index_rank=2,
    ),
    _case(
        "n4_h64_d64_s128_t4_r8_varlen_empty",
        batch=4,
        mode="varlen_empty",
        has_dst_indices=True,
        has_num_accepted_tokens=True,
        index_dtype="int32",
        index_rank=2,
    ),
    _case(
        "n8_h64_d64_s128_t4_r8_accepted_i32",
        batch=8,
        mode="varlen_uniform",
        has_dst_indices=True,
        has_num_accepted_tokens=True,
        accepted_dtype="int32",
        index_dtype="int32",
        index_rank=2,
    ),
    _case(
        "n8_h64_d64_s128_t4_r8_accepted_i64",
        batch=8,
        mode="varlen_uniform",
        has_dst_indices=True,
        has_num_accepted_tokens=True,
        accepted_dtype="int64",
        index_dtype="int64",
        index_rank=2,
    ),
]


_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _sequence_lengths(config: dict[str, Any], device: str | torch.device) -> torch.Tensor:
    batch = int(config["batch"])
    tokens = int(config["tokens"])
    mode = str(config["mode"])
    if mode == "varlen_variable":
        return torch.tensor(
            [(seq % tokens) + 1 for seq in range(batch)], dtype=torch.int64, device=device
        )
    if mode == "varlen_empty":
        return torch.tensor([0, *([tokens] * (batch - 1))], dtype=torch.int64, device=device)
    return torch.full((batch,), tokens, dtype=torch.int64, device=device)


def _total_tokens(config: dict[str, Any]) -> int:
    mode = str(config["mode"])
    batch = int(config["batch"])
    tokens = int(config["tokens"])
    if mode == "varlen_variable":
        return sum((seq % tokens) + 1 for seq in range(batch))
    if mode == "varlen_empty":
        return max(batch - 1, 0) * tokens
    return batch * tokens


@T.jit
def _selective_state_update_mtp_simple(
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
    intermediate_states_h: T.handle,
    intermediate_indices_h: T.handle,
    intermediate_scales_h: T.handle,
    cu_seqlens_h: T.handle,
    num_accepted_tokens_h: T.handle,
    rand_seed_h: T.handle,
    output_h: T.handle,
    state_stride_batch: T.int64,
    state_scale_stride_batch: T.int64,
    x_stride_batch: T.int64,
    x_stride_mtp: T.int64,
    dt_stride_batch: T.int64,
    dt_stride_mtp: T.int64,
    b_stride_batch: T.int64,
    b_stride_mtp: T.int64,
    c_stride_batch: T.int64,
    c_stride_mtp: T.int64,
    z_stride_batch: T.int64,
    z_stride_mtp: T.int64,
    out_stride_batch: T.int64,
    out_stride_mtp: T.int64,
    state_indices_stride_batch: T.int64,
    state_indices_stride_t: T.int64,
    dst_indices_stride_batch: T.int64,
    dst_indices_stride_t: T.int64,
    cache_steps: T.int32,
    nheads_runtime: T.int32,
    ngroups_runtime: T.int32,
    dt_softplus: T.int32,
    update_state: T.int32,
    pad_slot_id: T.int32,
    *,
    BATCH: T.constexpr,
    NHEADS: T.constexpr,
    DIM: T.constexpr,
    DSTATE: T.constexpr,
    NTOKENS: T.constexpr,
    HEADS_PER_GROUP: T.constexpr,
    CTAS_PER_HEAD: T.constexpr,
    DIM_PER_CTA: T.constexpr,
    DSTATE_PAD: T.constexpr,
    NUM_PASSES: T.constexpr,
    STATE_STAGES: T.constexpr,
    STATE_BYTES: T.constexpr,
    ELEMS_PER_TILE_MEMBER: T.constexpr,
    PAIRS_PER_TILE_MEMBER: T.constexpr,
    ELEMS_PER_TILE: T.constexpr,
    NUM_TILES: T.constexpr,
    HAS_STATE_INDICES: T.constexpr,
    HAS_DST_INDICES: T.constexpr,
    HAS_INTERMEDIATE_STATES: T.constexpr,
    HAS_INTERMEDIATE_INDICES: T.constexpr,
    HAS_CU_SEQLENS: T.constexpr,
    HAS_NUM_ACCEPTED_TOKENS: T.constexpr,
    HAS_Z: T.constexpr,
    HAS_D: T.constexpr,
    HAS_DT_BIAS: T.constexpr,
    SCALE_STATE: T.constexpr,
    PHILOX_ROUNDS: T.constexpr,
    OFF_B: T.constexpr,
    OFF_C: T.constexpr,
    OFF_X: T.constexpr,
    OFF_DT: T.constexpr,
    OFF_OUT: T.constexpr,
    OFF_DST_SLOTS: T.constexpr,
    OFF_STATE_IN: T.constexpr,
    SHARED_BYTES: T.constexpr,
    STATE_ELEMENTS: T.constexpr,
    SCALE_ELEMENTS: T.constexpr,
    X_ELEMENTS: T.constexpr,
    DT_ELEMENTS: T.constexpr,
    BC_ELEMENTS: T.constexpr,
    INDEX_ELEMENTS: T.constexpr,
    INTERMEDIATE_ELEMENTS: T.constexpr,
    INTERMEDIATE_SCALE_ELEMENTS: T.constexpr,
    CU_SEQLENS_ELEMENTS: T.constexpr,
    ACCEPTED_ELEMENTS: T.constexpr,
    STATE_DTYPE: T.constexpr,
    WEIGHT_DTYPE: T.constexpr,
    INDEX_DTYPE: T.constexpr,
    CU_SEQLENS_DTYPE: T.constexpr,
    ACCEPTED_DTYPE: T.constexpr,
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
    intermediate_states = T.match_buffer(
        intermediate_states_h, (INTERMEDIATE_ELEMENTS,), STATE_DTYPE, scope="global"
    )
    intermediate_indices = T.match_buffer(
        intermediate_indices_h, (ACCEPTED_ELEMENTS,), INDEX_DTYPE, scope="global"
    )
    intermediate_scales = T.match_buffer(
        intermediate_scales_h, (INTERMEDIATE_SCALE_ELEMENTS,), "float32", scope="global"
    )
    cu_seqlens = T.match_buffer(
        cu_seqlens_h, (CU_SEQLENS_ELEMENTS,), CU_SEQLENS_DTYPE, scope="global"
    )
    num_accepted_tokens = T.match_buffer(
        num_accepted_tokens_h, (ACCEPTED_ELEMENTS,), ACCEPTED_DTYPE, scope="global"
    )
    rand_seed = T.match_buffer(rand_seed_h, (1,), "int64", scope="global")
    output = T.match_buffer(output_h, (X_ELEMENTS,), "bfloat16", scope="global")
    T.device_entry()
    # TIRX_TRANSCRIBE_START selective_state_update_mtp_simple

    seq_idx, head, cta_z = T.cta_id([BATCH, NHEADS, CTAS_PER_HEAD])
    lane, warp = T.thread_id([32, 4])
    flat_tid: T.int32 = warp * 32 + lane
    dim_offset: T.int32 = cta_z * DIM_PER_CTA
    kv_group: T.int32 = head // HEADS_PER_GROUP

    bos: T.int32 = 0
    seq_len: T.int32 = NTOKENS
    if HAS_CU_SEQLENS:
        bos = T.cast(cu_seqlens[seq_idx], "int32")
        eos: T.int32 = T.cast(cu_seqlens[seq_idx + 1], "int32")
        seq_len = eos - bos

    if seq_len > 0:
        init_token_idx: T.int32 = 0
        if HAS_NUM_ACCEPTED_TOKENS:
            accepted: T.int32 = T.cast(num_accepted_tokens[seq_idx], "int32")
            init_token_idx = T.if_then_else(accepted > 1, accepted - 1, 0)

        state_batch: T.int64
        if HAS_STATE_INDICES:
            state_batch = T.cast(
                state_indices[
                    T.cast(seq_idx, "int64") * state_indices_stride_batch
                    + T.cast(init_token_idx, "int64") * state_indices_stride_t
                ],
                "int64",
            )
        else:
            state_batch = T.cast(seq_idx, "int64")
        is_pad: T.int32 = T.if_then_else(state_batch != T.cast(pad_slot_id, "int64"), 0, 1)

        state_head_offset: T.int64 = state_batch * state_stride_batch + T.cast(
            head * DIM * DSTATE, "int64"
        )
        state_ptr_offset_i32: T.int32 = T.cast(state_head_offset, "int32")

        shared_raw = T.alloc_buffer((SHARED_BYTES,), "uint8", scope="shared", align=128)
        s_b = T.decl_buffer(
            (NTOKENS * DSTATE_PAD,),
            "bfloat16",
            data=shared_raw.data,
            scope="shared",
            byte_offset=OFF_B,
            align=128,
        )
        s_c = T.decl_buffer(
            (NTOKENS * DSTATE_PAD,),
            "bfloat16",
            data=shared_raw.data,
            scope="shared",
            byte_offset=OFF_C,
            align=128,
        )
        s_x = T.decl_buffer(
            (NTOKENS * DIM_PER_CTA,),
            "bfloat16",
            data=shared_raw.data,
            scope="shared",
            byte_offset=OFF_X,
            align=128,
        )
        s_dt = T.decl_buffer(
            (NTOKENS,), "float32", data=shared_raw.data, scope="shared", byte_offset=OFF_DT, align=4
        )
        s_out = T.decl_buffer(
            (NTOKENS * DIM_PER_CTA,),
            "float32",
            data=shared_raw.data,
            scope="shared",
            byte_offset=OFF_OUT,
            align=4,
        )
        s_dst_slots = T.decl_buffer(
            (NTOKENS,),
            "int64",
            data=shared_raw.data,
            scope="shared",
            byte_offset=OFF_DST_SLOTS,
            align=8,
        )
        s_state = T.decl_buffer(
            (STATE_STAGES * 16 * DSTATE_PAD,),
            STATE_DTYPE,
            data=shared_raw.data,
            scope="shared",
            byte_offset=OFF_STATE_IN,
            align=128,
        )

        a_value: T.float32 = T.reinterpret("float32", _global_load_u32(matrix_a, head))
        d_value: T.float32 = 0.0
        if HAS_D:
            d_value = _load_weight(d_weight, head, WEIGHT_DTYPE)

        @T.inline
        def run_simple(IS_PAD: T.constexpr):
            if HAS_CU_SEQLENS:
                b_base: T.int64 = T.cast(bos, "int64") * b_stride_batch
                b_tstride: T.int64 = b_stride_batch
                c_base: T.int64 = T.cast(bos, "int64") * c_stride_batch
                c_tstride: T.int64 = c_stride_batch
                x_base: T.int64 = T.cast(bos, "int64") * x_stride_batch
                x_tstride: T.int64 = x_stride_batch
                dt_base: T.int64 = T.cast(bos, "int64") * dt_stride_batch
                dt_tstride: T.int64 = dt_stride_batch
            else:
                b_base = T.cast(seq_idx, "int64") * b_stride_batch
                b_tstride = b_stride_mtp
                c_base = T.cast(seq_idx, "int64") * c_stride_batch
                c_tstride = c_stride_mtp
                x_base = T.cast(seq_idx, "int64") * x_stride_batch
                x_tstride = x_stride_mtp
                dt_base = T.cast(seq_idx, "int64") * dt_stride_batch
                dt_tstride = dt_stride_mtp

            if warp == 0:
                for load_iter in T.serial((NTOKENS * DSTATE // 8 + 31) // 32):
                    packed_i: T.int32 = lane + load_iter * 32
                    if packed_i < NTOKENS * DSTATE // 8:
                        step: T.int32 = packed_i // (DSTATE // 8)
                        col: T.int32 = packed_i % (DSTATE // 8) * 8
                        if step < seq_len:
                            T.ptx["cp.async.cg.shared.global"](
                                s_b.ptr_to([step * DSTATE_PAD + col]),
                                matrix_b.ptr_to(
                                    [
                                        b_base
                                        + T.cast(step, "int64") * b_tstride
                                        + kv_group * DSTATE
                                        + col
                                    ]
                                ),
                                16,
                            )
            elif warp == 1:
                for load_iter in T.serial((NTOKENS * DSTATE // 8 + 31) // 32):
                    packed_i: T.int32 = lane + load_iter * 32
                    if packed_i < NTOKENS * DSTATE // 8:
                        step: T.int32 = packed_i // (DSTATE // 8)
                        col: T.int32 = packed_i % (DSTATE // 8) * 8
                        if step < seq_len:
                            T.ptx["cp.async.cg.shared.global"](
                                s_c.ptr_to([step * DSTATE_PAD + col]),
                                matrix_c.ptr_to(
                                    [
                                        c_base
                                        + T.cast(step, "int64") * c_tstride
                                        + kv_group * DSTATE
                                        + col
                                    ]
                                ),
                                16,
                            )

            for step_iter in T.serial((NTOKENS + 3) // 4):
                step: T.int32 = warp + step_iter * 4
                if step < seq_len:
                    for col_iter in T.serial((DIM_PER_CTA // 8 + 31) // 32):
                        col: T.int32 = (lane + col_iter * 32) * 8
                        if col < DIM_PER_CTA:
                            T.ptx["cp.async.cg.shared.global"](
                                s_x.ptr_to([step * DIM_PER_CTA + col]),
                                x.ptr_to(
                                    [
                                        x_base
                                        + T.cast(step, "int64") * x_tstride
                                        + head * DIM
                                        + dim_offset
                                        + col
                                    ]
                                ),
                                16,
                            )

            if not IS_PAD:
                for state_load_iter in T.serial((16 * DSTATE // (16 // STATE_BYTES) + 127) // 128):
                    packed_i: T.int32 = flat_tid + state_load_iter * 128
                    if packed_i < 16 * DSTATE // (16 // STATE_BYTES):
                        state_row: T.int32 = packed_i // (DSTATE // (16 // STATE_BYTES))
                        state_col: T.int32 = (
                            packed_i % (DSTATE // (16 // STATE_BYTES)) * (16 // STATE_BYTES)
                        )
                        T.ptx["cp.async.cg.shared.global"](
                            s_state.ptr_to([state_row * DSTATE_PAD + state_col]),
                            state.ptr_to(
                                [state_head_offset + (dim_offset + state_row) * DSTATE + state_col]
                            ),
                            16,
                        )

            if flat_tid < seq_len:
                dt_value: T.float32 = _load_weight(
                    dt, dt_base + T.cast(flat_tid, "int64") * dt_tstride + head, WEIGHT_DTYPE
                )
                if HAS_DT_BIAS:
                    dt_value = _add(dt_value, _load_weight(dt_bias, head, WEIGHT_DTYPE))
                if dt_softplus != 0:
                    if dt_value <= T.float32(20.0):
                        dt_exp: T.float32 = _exp2(_mul(dt_value, T.float32(_LOG2_E)))
                        dt_value = _mul(_log2(_add(T.float32(1.0), dt_exp)), T.float32(_LN_2))
                T.evaluate(
                    T.ptx.st.shared.b32(s_dt.ptr_to([flat_tid]), T.reinterpret("uint32", dt_value))
                )

            if flat_tid < NTOKENS:
                step: T.int32 = flat_tid
                dst_slot: T.int64 = -1
                if not IS_PAD and step < seq_len:
                    if HAS_DST_INDICES:
                        dst_index: T.int64 = T.cast(
                            dst_indices[
                                T.cast(seq_idx, "int64") * dst_indices_stride_batch
                                + T.cast(step, "int64") * dst_indices_stride_t
                            ],
                            "int64",
                        )
                        if dst_index != T.cast(pad_slot_id, "int64"):
                            dst_slot = dst_index
                    elif HAS_INTERMEDIATE_STATES:
                        intermediate_index: T.int64 = state_batch
                        if HAS_INTERMEDIATE_INDICES:
                            intermediate_index = T.cast(intermediate_indices[seq_idx], "int64")
                        dst_slot = intermediate_index * T.cast(cache_steps, "int64") + step
                    elif step == seq_len - 1 and update_state != 0:
                        dst_slot = state_batch
                s_dst_slots[step] = dst_slot

            T.ptx.cp.async_.commit_group()
            T.ptx.cp.async_.wait_group(0)
            T.ptx.bar.sync(T.uint32(0))

            random_seed: T.int64 = 0
            if PHILOX_ROUNDS > 0:
                random_seed = rand_seed[0]

            member: T.int32 = lane % 8
            row_group: T.int32 = lane // 8
            for pass_idx in T.serial(NUM_PASSES):
                pass_row: T.int32 = warp * 4 + row_group
                local_row: T.int32 = pass_idx * 16 + pass_row
                row_d: T.int32 = dim_offset + local_row
                state_stage: T.int32 = pass_idx % STATE_STAGES
                decode_scale: T.float32 = 1.0
                if SCALE_STATE and not IS_PAD:
                    decode_scale = T.reinterpret(
                        "float32",
                        _global_load_u32(
                            state_scale, state_batch * state_scale_stride_batch + head * DIM + row_d
                        ),
                    )

                r_state = T.alloc_local((NUM_TILES * PAIRS_PER_TILE_MEMBER,), "uint64")
                for tile_idx in T.unroll(NUM_TILES):
                    member_col: T.int32 = tile_idx * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER
                    if member_col < DSTATE and not IS_PAD:
                        state_words = T.alloc_local((4,), "uint32")
                        T.evaluate(
                            T.ptx.ld.shared.v4.b32(
                                state_words[0],
                                state_words[1],
                                state_words[2],
                                state_words[3],
                                s_state.ptr_to(
                                    [(state_stage * 16 + pass_row) * DSTATE_PAD + member_col]
                                ),
                            )
                        )
                        for pair_idx in T.unroll(PAIRS_PER_TILE_MEMBER):
                            if STATE_DTYPE == "bfloat16":
                                state_pair: T.uint64 = _bf16_word_to_f32x2(state_words[pair_idx])
                            elif STATE_DTYPE == "float16":
                                state_pair = T.cuda.make_float2(
                                    _f16_to_f32(_extract_u16(state_words[pair_idx], False)),
                                    _f16_to_f32(_extract_u16(state_words[pair_idx], True)),
                                )
                            elif STATE_DTYPE == "int16":
                                state_pair = T.cuda.make_float2(
                                    _i16_to_f32(_extract_u16(state_words[pair_idx], False)),
                                    _i16_to_f32(_extract_u16(state_words[pair_idx], True)),
                                )
                            else:
                                state_pair = T.cuda.make_float2(
                                    T.reinterpret("float32", state_words[pair_idx * 2]),
                                    T.reinterpret("float32", state_words[pair_idx * 2 + 1]),
                                )
                            if SCALE_STATE:
                                T.ptx.mul.f32x2(
                                    state_pair,
                                    state_pair,
                                    T.cuda.make_float2(decode_scale, decode_scale),
                                )
                            r_state[tile_idx * PAIRS_PER_TILE_MEMBER + pair_idx] = state_pair
                    else:
                        for pair_idx in T.unroll(PAIRS_PER_TILE_MEMBER):
                            r_state[tile_idx * PAIRS_PER_TILE_MEMBER + pair_idx] = (
                                T.cuda.make_float2(T.float32(0.0), T.float32(0.0))
                            )

                b_step: T.int32 = 0
                c_step: T.int32 = 0
                x_step: T.int32 = 0
                dt_step: T.int32 = 0
                out_step: T.int32 = 0
                for step in T.serial(NTOKENS):
                    if step < seq_len:
                        dst_slot: T.int64 = s_dst_slots[step]
                        dt_value: T.float32 = T.reinterpret(
                            "float32", _shared_load_u32(s_dt, dt_step)
                        )
                        da_value: T.float32 = _exp2(
                            _mul(_mul(a_value, dt_value), T.float32(_LOG2_E))
                        )
                        x_value: T.float32 = _bf16_to_f32(_shared_load_u16(s_x, x_step + local_row))
                        dtx_value: T.float32 = _mul(dt_value, x_value)
                        out_pair: T.uint64 = T.cuda.make_float2(T.float32(0.0), T.float32(0.0))

                        for tile_idx in T.unroll(NUM_TILES):
                            member_col: T.int32 = (
                                tile_idx * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER
                            )
                            if member_col < DSTATE:
                                b_words = T.alloc_local((4,), "uint32")
                                c_words = T.alloc_local((4,), "uint32")
                                if ELEMS_PER_TILE_MEMBER == 4:
                                    T.evaluate(
                                        T.ptx.ld.shared.v2.b32(
                                            b_words[0],
                                            b_words[1],
                                            s_b.ptr_to([b_step + member_col]),
                                        )
                                    )
                                    T.evaluate(
                                        T.ptx.ld.shared.v2.b32(
                                            c_words[0],
                                            c_words[1],
                                            s_c.ptr_to([c_step + member_col]),
                                        )
                                    )
                                else:
                                    T.evaluate(
                                        T.ptx.ld.shared.v4.b32(
                                            b_words[0],
                                            b_words[1],
                                            b_words[2],
                                            b_words[3],
                                            s_b.ptr_to([b_step + member_col]),
                                        )
                                    )
                                    T.evaluate(
                                        T.ptx.ld.shared.v4.b32(
                                            c_words[0],
                                            c_words[1],
                                            c_words[2],
                                            c_words[3],
                                            s_c.ptr_to([c_step + member_col]),
                                        )
                                    )
                                for pair_idx in T.unroll(PAIRS_PER_TILE_MEMBER):
                                    b_pair: T.uint64 = _bf16_word_to_f32x2(b_words[pair_idx])
                                    c_pair: T.uint64 = _bf16_word_to_f32x2(c_words[pair_idx])
                                    dbx_pair: T.uint64
                                    T.ptx.mul.f32x2(
                                        dbx_pair, b_pair, T.cuda.make_float2(dtx_value, dtx_value)
                                    )
                                    pair_index: T.int32 = (
                                        tile_idx * PAIRS_PER_TILE_MEMBER + pair_idx
                                    )
                                    updated_state: T.uint64
                                    T.ptx.fma.rn.f32x2(
                                        updated_state,
                                        T.cuda.make_float2(da_value, da_value),
                                        r_state[pair_index],
                                        dbx_pair,
                                    )
                                    r_state[pair_index] = updated_state
                                    T.ptx.fma.rn.f32x2(out_pair, updated_state, c_pair, out_pair)

                        out_value: T.float32 = _add(
                            T.cuda.float2_x(out_pair), T.cuda.float2_y(out_pair)
                        )
                        for delta_idx in T.unroll(3):
                            delta: T.int32 = T.shift_right(T.int32(4), delta_idx)
                            peer_out: T.float32 = T.cuda.__shfl_down_sync(
                                T.uint32(0xFFFFFFFF), out_value, delta, 32
                            )
                            out_value = _add(out_value, peer_out)
                        if member == 0:
                            row_output: T.float32 = _fma(d_value, x_value, out_value)
                            T.evaluate(
                                T.ptx.st.shared.b32(
                                    s_out.ptr_to([out_step + local_row]),
                                    T.reinterpret("uint32", row_output),
                                )
                            )

                        b_step = b_step + DSTATE_PAD
                        c_step = c_step + DSTATE_PAD
                        x_step = x_step + DIM_PER_CTA
                        dt_step = dt_step + 1
                        out_step = out_step + DIM_PER_CTA

                        if dst_slot != T.int64(-1):
                            encode_scale: T.float32 = 1.0
                            if SCALE_STATE:
                                local_max: T.float32 = T.float32(_FLT_LOWEST)
                                for tile_idx in T.unroll(NUM_TILES):
                                    for pair_idx in T.unroll(PAIRS_PER_TILE_MEMBER):
                                        col0: T.int32 = (
                                            tile_idx * ELEMS_PER_TILE
                                            + member * ELEMS_PER_TILE_MEMBER
                                            + pair_idx * 2
                                        )
                                        if col0 < DSTATE:
                                            state_pair: T.uint64 = r_state[
                                                tile_idx * PAIRS_PER_TILE_MEMBER + pair_idx
                                            ]
                                            local_max = _max(
                                                local_max,
                                                _max(
                                                    _abs(T.cuda.float2_x(state_pair)),
                                                    _abs(T.cuda.float2_y(state_pair)),
                                                ),
                                            )
                                for delta_idx in T.unroll(3):
                                    delta: T.int32 = T.shift_right(T.int32(4), delta_idx)
                                    peer_max: T.float32 = T.cuda.__shfl_down_sync(
                                        T.uint32(0xFFFFFFFF), local_max, delta, 32
                                    )
                                    local_max = _max(local_max, peer_max)
                                leader_lane: T.int32 = T.bitwise_and(lane, T.int32(-8))
                                local_max = T.cuda.__shfl_sync(
                                    T.uint32(0xFFFFFFFF), local_max, leader_lane, 32
                                )
                                if local_max != T.float32(0.0):
                                    encode_scale = _div(T.float32(32767.0), local_max)

                            if HAS_INTERMEDIATE_STATES:
                                dst_base: T.int64 = (
                                    dst_slot * T.cast(NHEADS * DIM * DSTATE, "int64")
                                    + head * DIM * DSTATE
                                    + row_d * DSTATE
                                )
                            else:
                                dst_base = (
                                    dst_slot * state_stride_batch
                                    + head * DIM * DSTATE
                                    + row_d * DSTATE
                                )

                            for tile_idx in T.unroll(NUM_TILES):
                                member_col: T.int32 = (
                                    tile_idx * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER
                                )
                                if member_col < DSTATE:
                                    store_words = T.alloc_local((4,), "uint32")
                                    random_words = T.alloc_local((4,), "uint32")
                                    for pair_idx in T.unroll(PAIRS_PER_TILE_MEMBER):
                                        pair_index: T.int32 = (
                                            tile_idx * PAIRS_PER_TILE_MEMBER + pair_idx
                                        )
                                        state_pair: T.uint64 = r_state[pair_index]
                                        if SCALE_STATE:
                                            T.ptx.mul.f32x2(
                                                state_pair,
                                                state_pair,
                                                T.cuda.make_float2(encode_scale, encode_scale),
                                            )
                                            low_scaled: T.float32 = _min(
                                                _max(
                                                    T.cuda.float2_x(state_pair), T.float32(-32767.0)
                                                ),
                                                T.float32(32767.0),
                                            )
                                            high_scaled: T.float32 = _min(
                                                _max(
                                                    T.cuda.float2_y(state_pair), T.float32(-32767.0)
                                                ),
                                                T.float32(32767.0),
                                            )
                                            low_i32 = T.alloc_local((1,), "int32")
                                            high_i32 = T.alloc_local((1,), "int32")
                                            T.evaluate(
                                                T.ptx.cvt.rni.ftz.s32.f32(low_i32[0], low_scaled)
                                            )
                                            T.evaluate(
                                                T.ptx.cvt.rni.ftz.s32.f32(high_i32[0], high_scaled)
                                            )
                                            store_words[pair_idx] = _prmt_5410(
                                                T.reinterpret("uint32", low_i32[0]),
                                                T.reinterpret("uint32", high_i32[0]),
                                            )
                                        elif PHILOX_ROUNDS > 0:
                                            element_idx: T.int32 = pair_idx * 2
                                            if pair_idx % 2 == 0:
                                                offset_mad = T.alloc_local((1,), "int32")
                                                T.ptx.mad.lo.s32(
                                                    offset_mad[0],
                                                    row_d,
                                                    T.int32(DSTATE),
                                                    state_ptr_offset_i32,
                                                )
                                                random_offset: T.int32 = _add_s32(
                                                    offset_mad[0], member_col + element_idx
                                                )
                                                c0: T.uint32 = T.reinterpret(
                                                    "uint32", random_offset
                                                )
                                                c1_signed = T.alloc_local((1,), "int32")
                                                T.ptx.shr.s32(
                                                    c1_signed[0], random_offset, T.uint32(31)
                                                )
                                                c1: T.uint32 = T.reinterpret("uint32", c1_signed[0])
                                                c2: T.uint32 = 0
                                                c3: T.uint32 = 0
                                                seed_u64: T.uint64 = T.reinterpret(
                                                    "uint64", random_seed
                                                )
                                                k0: T.uint32 = T.cast(seed_u64, "uint32")
                                                k1: T.uint32 = T.cast(
                                                    T.shift_right(seed_u64, T.uint64(32)), "uint32"
                                                )
                                                for philox_round in T.unroll(10):
                                                    old_c0: T.uint32 = c0
                                                    old_c2: T.uint32 = c2
                                                    next_c0: T.uint32 = T.bitwise_xor(
                                                        T.bitwise_xor(
                                                            _mul_hi_u32(
                                                                T.uint32(0xCD9E8D57), old_c2
                                                            ),
                                                            c1,
                                                        ),
                                                        k0,
                                                    )
                                                    next_c2: T.uint32 = T.bitwise_xor(
                                                        T.bitwise_xor(
                                                            _mul_hi_u32(
                                                                T.uint32(0xD2511F53), old_c0
                                                            ),
                                                            c3,
                                                        ),
                                                        k1,
                                                    )
                                                    next_c1: T.int32 = _mul_lo_s32(
                                                        T.int32(-845247145),
                                                        T.reinterpret("int32", old_c2),
                                                    )
                                                    next_c3: T.int32 = _mul_lo_s32(
                                                        T.int32(-766435501),
                                                        T.reinterpret("int32", old_c0),
                                                    )
                                                    next_k0: T.int32 = _add_s32(
                                                        T.reinterpret("int32", k0),
                                                        T.int32(-1640531527),
                                                    )
                                                    next_k1: T.int32 = _add_s32(
                                                        T.reinterpret("int32", k1),
                                                        T.int32(-1150833019),
                                                    )
                                                    c0 = next_c0
                                                    c1 = T.reinterpret("uint32", next_c1)
                                                    c2 = next_c2
                                                    c3 = T.reinterpret("uint32", next_c3)
                                                    k0 = T.reinterpret("uint32", next_k0)
                                                    k1 = T.reinterpret("uint32", next_k1)
                                                random_words[0] = c0
                                                random_words[1] = c1
                                                random_words[2] = c2
                                                random_words[3] = c3
                                            packed_f16 = T.alloc_local((1,), "uint32")
                                            T.evaluate(
                                                T.ptx.cvt.rs.f16x2.f32(
                                                    packed_f16[0],
                                                    T.cuda.float2_y(state_pair),
                                                    T.cuda.float2_x(state_pair),
                                                    random_words[pair_idx % 2],
                                                )
                                            )
                                            store_words[pair_idx] = packed_f16[0]
                                        elif STATE_DTYPE == "bfloat16":
                                            low_bits: T.uint16 = _f32_to_bf16(
                                                T.cuda.float2_x(state_pair)
                                            )
                                            high_bits: T.uint16 = _f32_to_bf16(
                                                T.cuda.float2_y(state_pair)
                                            )
                                            T.evaluate(
                                                T.ptx.mov.b32(
                                                    store_words[pair_idx], low_bits, high_bits
                                                )
                                            )
                                        elif STATE_DTYPE == "float16":
                                            low_bits = _f32_to_f16(T.cuda.float2_x(state_pair))
                                            high_bits = _f32_to_f16(T.cuda.float2_y(state_pair))
                                            T.evaluate(
                                                T.ptx.mov.b32(
                                                    store_words[pair_idx], low_bits, high_bits
                                                )
                                            )
                                        else:
                                            store_words[pair_idx * 2] = T.reinterpret(
                                                "uint32", T.cuda.float2_x(state_pair)
                                            )
                                            store_words[pair_idx * 2 + 1] = T.reinterpret(
                                                "uint32", T.cuda.float2_y(state_pair)
                                            )
                                    if HAS_INTERMEDIATE_STATES:
                                        T.evaluate(
                                            T.ptx.st.global_.v4.b32(
                                                intermediate_states.ptr_to([dst_base + member_col]),
                                                store_words[0],
                                                store_words[1],
                                                store_words[2],
                                                store_words[3],
                                            )
                                        )
                                    else:
                                        T.evaluate(
                                            T.ptx.st.global_.v4.b32(
                                                state.ptr_to([dst_base + member_col]),
                                                store_words[0],
                                                store_words[1],
                                                store_words[2],
                                                store_words[3],
                                            )
                                        )

                            if SCALE_STATE and member == 0:
                                new_decode_scale: T.float32 = _rcp(encode_scale)
                                if HAS_INTERMEDIATE_STATES:
                                    scale_offset: T.int64 = (
                                        dst_slot * T.cast(NHEADS * DIM, "int64")
                                        + head * DIM
                                        + row_d
                                    )
                                    T.evaluate(
                                        T.ptx.st.global_.b32(
                                            intermediate_scales.ptr_to([scale_offset]),
                                            T.reinterpret("uint32", new_decode_scale),
                                        )
                                    )
                                else:
                                    scale_offset = (
                                        dst_slot * state_scale_stride_batch + head * DIM + row_d
                                    )
                                    T.evaluate(
                                        T.ptx.st.global_.b32(
                                            state_scale.ptr_to([scale_offset]),
                                            T.reinterpret("uint32", new_decode_scale),
                                        )
                                    )

                if NUM_PASSES > 1 and pass_idx < NUM_PASSES - 1:
                    next_stage: T.int32 = (pass_idx + 1) % STATE_STAGES
                    next_dim_base: T.int32 = dim_offset + (pass_idx + 1) * 16
                    if not IS_PAD:
                        for state_load_iter in T.serial(
                            (16 * DSTATE // (16 // STATE_BYTES) + 127) // 128
                        ):
                            packed_i: T.int32 = flat_tid + state_load_iter * 128
                            if packed_i < 16 * DSTATE // (16 // STATE_BYTES):
                                state_row: T.int32 = packed_i // (DSTATE // (16 // STATE_BYTES))
                                state_col: T.int32 = (
                                    packed_i % (DSTATE // (16 // STATE_BYTES)) * (16 // STATE_BYTES)
                                )
                                T.ptx["cp.async.cg.shared.global"](
                                    s_state.ptr_to(
                                        [(next_stage * 16 + state_row) * DSTATE_PAD + state_col]
                                    ),
                                    state.ptr_to(
                                        [
                                            state_head_offset
                                            + (next_dim_base + state_row) * DSTATE
                                            + state_col
                                        ]
                                    ),
                                    16,
                                )
                    T.ptx.cp.async_.commit_group()
                    T.ptx.cp.async_.wait_group(0)
                    T.ptx.bar.sync(T.uint32(0))

            T.ptx.bar.sync(T.uint32(0))
            for output_iter in T.serial((NTOKENS + 3) // 4):
                step: T.int32 = warp + output_iter * 4
                if step < seq_len:
                    if HAS_CU_SEQLENS:
                        out_base: T.int64 = (
                            T.cast(bos + step, "int64") * out_stride_batch + head * DIM + dim_offset
                        )
                        z_base: T.int64 = (
                            T.cast(bos + step, "int64") * z_stride_batch + head * DIM + dim_offset
                        )
                    else:
                        out_base = (
                            T.cast(seq_idx, "int64") * out_stride_batch
                            + T.cast(step, "int64") * out_stride_mtp
                            + head * DIM
                            + dim_offset
                        )
                        z_base = (
                            T.cast(seq_idx, "int64") * z_stride_batch
                            + T.cast(step, "int64") * z_stride_mtp
                            + head * DIM
                            + dim_offset
                        )

                    if DIM_PER_CTA >= 32:
                        output_count: T.int32 = DIM_PER_CTA // 32
                        local_col: T.int32 = lane * output_count
                        out_words = T.alloc_local((4,), "uint32")
                        if DIM_PER_CTA == 32:
                            out_words[0] = _shared_load_u32(s_out, step * DIM_PER_CTA + local_col)
                        elif DIM_PER_CTA == 64:
                            if OFF_OUT % 8 == 0:
                                T.evaluate(
                                    T.ptx.ld.shared.v2.b32(
                                        out_words[0],
                                        out_words[1],
                                        s_out.ptr_to([step * DIM_PER_CTA + local_col]),
                                    )
                                )
                            else:
                                out_words[0] = _shared_load_u32(
                                    s_out, step * DIM_PER_CTA + local_col
                                )
                                out_words[1] = _shared_load_u32(
                                    s_out, step * DIM_PER_CTA + local_col + 1
                                )
                        else:
                            T.evaluate(
                                T.ptx.ld.shared.v4.b32(
                                    out_words[0],
                                    out_words[1],
                                    out_words[2],
                                    out_words[3],
                                    s_out.ptr_to([step * DIM_PER_CTA + local_col]),
                                )
                            )
                        z_bits = T.alloc_local((4,), "uint16")
                        if HAS_Z:
                            if DIM_PER_CTA == 32:
                                z_bits[0] = _global_load_u16(z, z_base + local_col)
                            elif DIM_PER_CTA == 64:
                                T.evaluate(
                                    T.ptx.ld.global_.v2.b16(
                                        z_bits[0], z_bits[1], z.ptr_to([z_base + local_col])
                                    )
                                )
                            else:
                                T.evaluate(
                                    T.ptx.ld.global_.v4.b16(
                                        z_bits[0],
                                        z_bits[1],
                                        z_bits[2],
                                        z_bits[3],
                                        z.ptr_to([z_base + local_col]),
                                    )
                                )
                        output_bits = T.alloc_local((4,), "uint16")
                        for element in T.unroll(DIM_PER_CTA // 32):
                            value: T.float32 = T.reinterpret("float32", out_words[element])
                            if HAS_Z:
                                z_value: T.float32 = _bf16_to_f32(z_bits[element])
                                exp_neg_z: T.float32 = _exp2(
                                    _mul(_sub(T.float32(0.0), z_value), T.float32(_LOG2_E))
                                )
                                sigmoid_z: T.float32 = _div(
                                    T.float32(1.0), _add(T.float32(1.0), exp_neg_z)
                                )
                                value = _mul(value, _mul(z_value, sigmoid_z))
                            output_bits[element] = _f32_to_bf16(value)
                        if DIM_PER_CTA == 32:
                            T.evaluate(
                                T.ptx.st.global_.b16(
                                    output.ptr_to([out_base + local_col]), output_bits[0]
                                )
                            )
                        elif DIM_PER_CTA == 64:
                            T.evaluate(
                                T.ptx.st.global_.v2.b16(
                                    output.ptr_to([out_base + local_col]),
                                    output_bits[0],
                                    output_bits[1],
                                )
                            )
                        else:
                            T.evaluate(
                                T.ptx.st.global_.v4.b16(
                                    output.ptr_to([out_base + local_col]),
                                    output_bits[0],
                                    output_bits[1],
                                    output_bits[2],
                                    output_bits[3],
                                )
                            )
                    elif lane < DIM_PER_CTA:
                        value: T.float32 = T.reinterpret(
                            "float32", _shared_load_u32(s_out, step * DIM_PER_CTA + lane)
                        )
                        if HAS_Z:
                            z_value: T.float32 = _bf16_to_f32(_global_load_u16(z, z_base + lane))
                            exp_neg_z: T.float32 = _exp2(
                                _mul(_sub(T.float32(0.0), z_value), T.float32(_LOG2_E))
                            )
                            sigmoid_z: T.float32 = _div(
                                T.float32(1.0), _add(T.float32(1.0), exp_neg_z)
                            )
                            value = _mul(value, _mul(z_value, sigmoid_z))
                        output_bit: T.uint16 = _f32_to_bf16(value)
                        T.evaluate(
                            T.ptx.st.global_.b16(output.ptr_to([out_base + lane]), output_bit)
                        )

        if is_pad != 0:
            run_simple(True)
        else:
            run_simple(False)


def _num_sms(device: str | torch.device = "cuda") -> int:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(device).multi_processor_count
    return 148


def _specialization(config: dict[str, Any]) -> dict[str, Any]:
    batch = int(config["batch"])
    nheads = int(config["nheads"])
    dim = int(config["dim"])
    dstate = int(config["dstate"])
    tokens = int(config["tokens"])
    total_tokens = _total_tokens(config)
    heads_per_group = int(config["heads_per_group"])
    ngroups = nheads // heads_per_group
    is_varlen = str(config["mode"]).startswith("varlen")
    logical_slots = max(batch * tokens if is_varlen else batch, 1)
    if bool(config.get("has_dst_indices", False)):
        logical_slots *= 2
    state_slots = logical_slots
    state_stride_factor = int(config.get("state_stride_factor", 1))
    index_elements = batch * tokens if int(config["index_rank"]) == 2 else batch
    intermediate_elements = (
        batch * tokens * nheads * dim * dstate if bool(config["has_intermediate_states"]) else 1
    )
    state_dtype = str(config["state_dtype"])
    state_bytes = 4 if state_dtype == "float32" else 2
    scale_state = state_dtype == "int16"
    philox_rounds = int(config.get("philox_rounds", 0))
    if philox_rounds not in (0, 10):
        raise ValueError("MTP simple stochastic rounding supports philox_rounds in {0, 10}")
    if philox_rounds and state_dtype != "float16":
        raise ValueError("MTP simple Philox is restricted to float16 state")

    target_ctas = _num_sms(config.get("device", "cuda")) * 10
    total_tiles = max(batch * nheads, 1)
    requested_ctas = max(1, min(target_ctas // total_tiles, dim // 16))
    ctas_per_head = 4 if requested_ctas >= 4 else 2 if requested_ctas >= 2 else 1
    dim_per_cta = dim // ctas_per_head
    if dim % ctas_per_head or dim_per_cta % 16:
        raise ValueError("MTP simple requires DIM_PER_CTA to be a multiple of 16")
    num_passes = dim_per_cta // 16
    state_stages = 1 if num_passes == 1 else 2

    dstate_pad = _align_up(dstate * 2, 128) // 2
    elems_per_tile_member = 16 // state_bytes
    pairs_per_tile_member = elems_per_tile_member // 2
    elems_per_tile = elems_per_tile_member * 8
    num_tiles = (_next_power_of_two(dstate) // 8) // elems_per_tile_member

    off_b = 0
    off_c = _align_up(off_b + tokens * dstate_pad * 2, 128)
    off_x = _align_up(off_c + tokens * dstate_pad * 2, 128)
    off_dt = _align_up(off_x + tokens * dim_per_cta * 2, 4)
    off_out = off_dt + tokens * 4
    off_dst_slots = _align_up(off_out + tokens * dim_per_cta * 4, 8)
    off_state_in = _align_up(off_dst_slots + tokens * 8, 128)
    shared_bytes = _align_up(off_state_in + state_stages * 16 * dstate_pad * state_bytes, 128)

    return {
        "BATCH": batch,
        "NHEADS": nheads,
        "DIM": dim,
        "DSTATE": dstate,
        "NTOKENS": tokens,
        "HEADS_PER_GROUP": heads_per_group,
        "CTAS_PER_HEAD": ctas_per_head,
        "DIM_PER_CTA": dim_per_cta,
        "DSTATE_PAD": dstate_pad,
        "NUM_PASSES": num_passes,
        "STATE_STAGES": state_stages,
        "STATE_BYTES": state_bytes,
        "ELEMS_PER_TILE_MEMBER": elems_per_tile_member,
        "PAIRS_PER_TILE_MEMBER": pairs_per_tile_member,
        "ELEMS_PER_TILE": elems_per_tile,
        "NUM_TILES": num_tiles,
        "HAS_STATE_INDICES": bool(config.get("has_state_indices", True)),
        "HAS_DST_INDICES": bool(config.get("has_dst_indices", False)),
        "HAS_INTERMEDIATE_STATES": bool(config.get("has_intermediate_states", False)),
        "HAS_INTERMEDIATE_INDICES": bool(config.get("has_intermediate_states", False)),
        "HAS_CU_SEQLENS": is_varlen,
        "HAS_NUM_ACCEPTED_TOKENS": bool(config.get("has_num_accepted_tokens", False)),
        "HAS_Z": bool(config.get("has_z", False)),
        "HAS_D": bool(config.get("has_d", True)),
        "HAS_DT_BIAS": bool(config.get("has_dt_bias", True)),
        "SCALE_STATE": scale_state,
        "PHILOX_ROUNDS": philox_rounds,
        "OFF_B": off_b,
        "OFF_C": off_c,
        "OFF_X": off_x,
        "OFF_DT": off_dt,
        "OFF_OUT": off_out,
        "OFF_DST_SLOTS": off_dst_slots,
        "OFF_STATE_IN": off_state_in,
        "SHARED_BYTES": shared_bytes,
        "STATE_ELEMENTS": state_slots * state_stride_factor * nheads * dim * dstate,
        "SCALE_ELEMENTS": state_slots * nheads * dim if scale_state else 1,
        "X_ELEMENTS": total_tokens * nheads * dim,
        "DT_ELEMENTS": total_tokens * nheads,
        "BC_ELEMENTS": total_tokens * ngroups * dstate,
        "INDEX_ELEMENTS": max(index_elements, 1),
        "INTERMEDIATE_ELEMENTS": max(intermediate_elements, 1),
        "INTERMEDIATE_SCALE_ELEMENTS": (
            batch * tokens * nheads * dim
            if bool(config["has_intermediate_states"]) and scale_state
            else 1
        ),
        "CU_SEQLENS_ELEMENTS": batch + 1,
        "ACCEPTED_ELEMENTS": max(batch, 1),
        "STATE_DTYPE": state_dtype,
        "WEIGHT_DTYPE": str(config["weight_dtype"]),
        "INDEX_DTYPE": str(config["index_dtype"]),
        "CU_SEQLENS_DTYPE": str(config["cu_seqlens_dtype"]),
        "ACCEPTED_DTYPE": str(config["accepted_dtype"]),
    }


def get_kernel(**kwargs: Any):
    """Return the specialized plain-TIRx MTP simple kernel."""
    return _selective_state_update_mtp_simple.specialize(**_specialization(kwargs))


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Allocate deterministic, independent TIRx and FlashInfer MTP cases."""
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for selective-state-update MTP simple")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(f"MTP simple SM100 requires compute capability 10.x, got {capability}")

    batch = int(kwargs["batch"])
    nheads = int(kwargs["nheads"])
    dim = int(kwargs["dim"])
    dstate = int(kwargs["dstate"])
    tokens = int(kwargs["tokens"])
    heads_per_group = int(kwargs["heads_per_group"])
    if nheads % heads_per_group:
        raise ValueError("nheads must be divisible by heads_per_group")
    ngroups = nheads // heads_per_group
    state_dtype = _TORCH_DTYPES[str(kwargs["state_dtype"])]
    weight_dtype = _TORCH_DTYPES[str(kwargs["weight_dtype"])]
    index_dtype = _TORCH_DTYPES[str(kwargs["index_dtype"])]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(kwargs.get("seed", 0)) + 20260810)

    sequence_lengths = _sequence_lengths(kwargs, device)
    total_tokens = int(sequence_lengths.sum().item())
    is_varlen = str(kwargs["mode"]).startswith("varlen")
    logical_slots = max(batch * tokens if is_varlen else batch, 1)
    if bool(kwargs.get("has_dst_indices", False)):
        logical_slots *= 2
    state_cache_size = logical_slots
    stride_factor = int(kwargs.get("state_stride_factor", 1))
    state_storage_shape = (state_cache_size * stride_factor, nheads, dim, dstate)

    if state_dtype == torch.int16:
        logical_state = torch.randn(
            (state_cache_size, nheads, dim, dstate),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        amax = logical_state.abs().amax(dim=-1)
        encode = torch.where(amax == 0, torch.ones_like(amax), 32767.0 / amax)
        quantized = (logical_state * encode[..., None]).round().clamp(-32767, 32767).to(torch.int16)
        initial_state_storage = torch.zeros(state_storage_shape, dtype=state_dtype, device=device)
        initial_state = initial_state_storage[::stride_factor]
        initial_state.copy_(quantized)
        initial_state_scale = 1.0 / encode
    else:
        initial_state_storage = torch.randn(
            state_storage_shape, dtype=state_dtype, device=device, generator=generator
        )
        initial_state = initial_state_storage[::stride_factor]
        initial_state_scale = torch.ones((1,), dtype=torch.float32, device=device)

    if is_varlen:
        x = torch.randn(
            (total_tokens, nheads, dim), dtype=torch.bfloat16, device=device, generator=generator
        )
        dt_base = torch.randn(
            (total_tokens, nheads), dtype=weight_dtype, device=device, generator=generator
        )
        dt = dt_base.as_strided((total_tokens, nheads, dim), (nheads, 1, 0))
        matrix_b = torch.randn(
            (total_tokens, ngroups, dstate),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        matrix_c = torch.randn_like(matrix_b)
    else:
        x = torch.randn(
            (batch, tokens, nheads, dim), dtype=torch.bfloat16, device=device, generator=generator
        )
        dt_base = torch.randn(
            (batch, tokens, nheads), dtype=weight_dtype, device=device, generator=generator
        )
        dt = dt_base.as_strided((batch, tokens, nheads, dim), (tokens * nheads, nheads, 1, 0))
        matrix_b = torch.randn(
            (batch, tokens, ngroups, dstate),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        matrix_c = torch.randn_like(matrix_b)

    matrix_a_base = (
        -torch.rand((nheads,), dtype=torch.float32, device=device, generator=generator) - 1.0
    )
    matrix_a = matrix_a_base.as_strided((nheads, dim, dstate), (1, 0, 0))
    d_base = torch.randn((nheads,), dtype=weight_dtype, device=device, generator=generator)
    if not bool(kwargs.get("has_d", True)):
        d_base.zero_()
    d_weight = d_base.as_strided((nheads, dim), (1, 0))
    bias_base = torch.rand((nheads,), dtype=weight_dtype, device=device, generator=generator) - 4.0
    dt_bias = bias_base.as_strided((nheads, dim), (1, 0))
    z = torch.randn_like(x) if bool(kwargs.get("has_z", False)) else None

    if bool(kwargs.get("shared_state_slot", False)):
        source_indices = torch.zeros((batch, tokens), dtype=index_dtype, device=device)
    elif is_varlen:
        source_indices = torch.arange(batch * tokens, dtype=index_dtype, device=device).reshape(
            batch, tokens
        )
    else:
        source_indices = (
            torch.arange(batch, dtype=index_dtype, device=device)[:, None]
            .expand(batch, tokens)
            .clone()
        )
    destination_indices = source_indices.clone()
    if bool(kwargs.get("has_dst_indices", False)):
        destination_indices = source_indices + logical_slots // 2
    pad_every = int(kwargs.get("pad_every", 0))
    if pad_every:
        source_indices.reshape(-1)[::pad_every] = -1
    if is_varlen:
        for seq, length in enumerate(sequence_lengths.tolist()):
            source_indices[seq, length:] = -1
            destination_indices[seq, length:] = -1

    index_rank = int(kwargs.get("index_rank", 1))
    state_indices = source_indices if index_rank == 2 else source_indices[:, 0].contiguous()
    dst_indices = destination_indices if index_rank == 2 else destination_indices[:, 0].contiguous()
    cu_seqlens = torch.zeros(
        (batch + 1,), dtype=_TORCH_DTYPES[str(kwargs["cu_seqlens_dtype"])], device=device
    )
    cu_seqlens[1:] = torch.cumsum(sequence_lengths, dim=0).to(cu_seqlens.dtype)
    accepted_dtype = _TORCH_DTYPES[str(kwargs["accepted_dtype"])]
    num_accepted_tokens = torch.ones((batch,), dtype=accepted_dtype, device=device)
    if bool(kwargs.get("has_num_accepted_tokens", False)):
        num_accepted_tokens.copy_(
            torch.clamp(sequence_lengths, min=1, max=tokens).to(accepted_dtype)
        )

    intermediate_states = None
    intermediate_scales = None
    intermediate_indices = None
    if bool(kwargs.get("has_intermediate_states", False)):
        intermediate_states = torch.zeros(
            (batch, tokens, nheads, dim, dstate), dtype=state_dtype, device=device
        )
        intermediate_indices = torch.arange(batch, dtype=index_dtype, device=device)
        if state_dtype == torch.int16:
            intermediate_scales = torch.zeros(
                (batch, tokens, nheads, dim), dtype=torch.float32, device=device
            )

    tirx_output = torch.empty_like(x)
    flashinfer_output = torch.empty_like(x)
    return {
        "config": dict(kwargs),
        "spec": _specialization(kwargs),
        "sequence_lengths": sequence_lengths,
        "tirx_state_storage": initial_state_storage.clone(),
        "flashinfer_state_storage": initial_state_storage.clone(),
        "initial_state_storage": initial_state_storage,
        "tirx_state_scale": initial_state_scale.clone(),
        "flashinfer_state_scale": initial_state_scale.clone(),
        "x": x,
        "dt": dt,
        "dt_base": dt_base,
        "matrix_a": matrix_a,
        "matrix_a_base": matrix_a_base,
        "matrix_b": matrix_b,
        "matrix_c": matrix_c,
        "d_weight": d_weight,
        "d_base": d_base,
        "z": z,
        "dt_bias": dt_bias,
        "dt_bias_base": bias_base,
        "state_indices": state_indices,
        "dst_indices": dst_indices,
        "cu_seqlens": cu_seqlens,
        "num_accepted_tokens": num_accepted_tokens,
        "tirx_intermediate_states": (
            intermediate_states.clone()
            if intermediate_states is not None
            else torch.zeros((1,), dtype=state_dtype, device=device)
        ),
        "flashinfer_intermediate_states": (
            intermediate_states.clone()
            if intermediate_states is not None
            else torch.zeros((1,), dtype=state_dtype, device=device)
        ),
        "intermediate_state_indices": intermediate_indices,
        "tirx_intermediate_state_scales": (
            intermediate_scales.clone()
            if intermediate_scales is not None
            else torch.zeros((1,), dtype=torch.float32, device=device)
        ),
        "flashinfer_intermediate_state_scales": (
            intermediate_scales.clone()
            if intermediate_scales is not None
            else torch.zeros((1,), dtype=torch.float32, device=device)
        ),
        "rand_seed": torch.tensor([int(kwargs.get("seed", 0))], dtype=torch.int64, device=device),
        "tirx_output": tirx_output,
        "flashinfer_output": flashinfer_output,
    }


@functools.cache
def _load_frozen_oracle():
    from flashinfer.jit.env import FLASHINFER_INCLUDE_DIR
    from flashinfer.mamba import selective_state_update

    source_path = (
        Path(FLASHINFER_INCLUDE_DIR)
        / "flashinfer/mamba/kernel_selective_state_update_mtp_simple.cuh"
    )
    if not source_path.is_file():
        raise RuntimeError(f"missing frozen FlashInfer source: {source_path}")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != FROZEN_FLASHINFER_SOURCE_SHA256:
        raise RuntimeError(
            "selective-state-update MTP oracle does not match the reviewed source: "
            f"{source_path} sha256={digest}"
        )
    return selective_state_update


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    config = case["config"]
    nheads = int(config["nheads"])
    dim = int(config["dim"])
    dstate = int(config["dstate"])
    ngroups = nheads // int(config["heads_per_group"])
    tokens = int(config["tokens"])
    stride_factor = int(config.get("state_stride_factor", 1))
    is_varlen = str(config["mode"]).startswith("varlen")

    x = case["x"]
    dt = case["dt"]
    matrix_b = case["matrix_b"]
    matrix_c = case["matrix_c"]
    output = case["tirx_output"]
    if is_varlen:
        x_stride_batch = x.stride(0)
        x_stride_mtp = x.stride(0)
        dt_stride_batch = dt.stride(0)
        dt_stride_mtp = dt.stride(0)
        b_stride_batch = matrix_b.stride(0)
        b_stride_mtp = matrix_b.stride(0)
        c_stride_batch = matrix_c.stride(0)
        c_stride_mtp = matrix_c.stride(0)
        out_stride_batch = output.stride(0)
        out_stride_mtp = output.stride(0)
    else:
        x_stride_batch, x_stride_mtp = x.stride(0), x.stride(1)
        dt_stride_batch, dt_stride_mtp = dt.stride(0), dt.stride(1)
        b_stride_batch, b_stride_mtp = matrix_b.stride(0), matrix_b.stride(1)
        c_stride_batch, c_stride_mtp = matrix_c.stride(0), matrix_c.stride(1)
        out_stride_batch, out_stride_mtp = output.stride(0), output.stride(1)

    if case["z"] is not None:
        z_arg = case["z"]
        if is_varlen:
            z_stride_batch = z_arg.stride(0)
            z_stride_mtp = z_arg.stride(0)
        else:
            z_stride_batch, z_stride_mtp = z_arg.stride(0), z_arg.stride(1)
    else:
        z_arg = x
        z_stride_batch, z_stride_mtp = x_stride_batch, x_stride_mtp

    state_indices = case["state_indices"]
    dst_indices = case["dst_indices"]
    state_indices_stride_batch = state_indices.stride(0)
    state_indices_stride_t = state_indices.stride(1) if state_indices.ndim == 2 else 0
    dst_indices_stride_batch = dst_indices.stride(0)
    dst_indices_stride_t = dst_indices.stride(1) if dst_indices.ndim == 2 else 0
    intermediate_indices = case["intermediate_state_indices"]
    if intermediate_indices is None:
        intermediate_indices = torch.zeros(
            (int(config["batch"]),),
            dtype=_TORCH_DTYPES[str(config["index_dtype"])],
            device=x.device,
        )

    return (
        case["tirx_state_storage"].reshape(-1),
        case["tirx_state_scale"].reshape(-1),
        x.reshape(-1),
        case["dt_base"].reshape(-1),
        case["matrix_a_base"],
        matrix_b.reshape(-1),
        matrix_c.reshape(-1),
        case["d_base"],
        z_arg.reshape(-1),
        case["dt_bias_base"],
        state_indices.reshape(-1),
        dst_indices.reshape(-1),
        case["tirx_intermediate_states"].reshape(-1),
        intermediate_indices.reshape(-1),
        case["tirx_intermediate_state_scales"].reshape(-1),
        case["cu_seqlens"].reshape(-1),
        case["num_accepted_tokens"].reshape(-1),
        case["rand_seed"],
        output.reshape(-1),
        stride_factor * nheads * dim * dstate,
        nheads * dim if str(config["state_dtype"]) == "int16" else 0,
        x_stride_batch,
        x_stride_mtp,
        dt_stride_batch,
        dt_stride_mtp,
        b_stride_batch,
        b_stride_mtp,
        c_stride_batch,
        c_stride_mtp,
        z_stride_batch,
        z_stride_mtp,
        out_stride_batch,
        out_stride_mtp,
        state_indices_stride_batch,
        state_indices_stride_t,
        dst_indices_stride_batch,
        dst_indices_stride_t,
        tokens,
        nheads,
        ngroups,
        int(bool(config.get("dt_softplus", False))),
        int(bool(config.get("update_state", True))),
        -1,
    )


def _run_reference(case: dict[str, Any]) -> torch.Tensor:
    config = case["config"]
    stride_factor = int(config.get("state_stride_factor", 1))
    state_view = case["flashinfer_state_storage"][::stride_factor]
    scale_state = str(config["state_dtype"]) == "int16"
    source_out = case["flashinfer_output"] if bool(config.get("use_out_tensor", True)) else None
    oracle = _load_frozen_oracle()
    result = oracle(
        state_view,
        case["x"],
        case["dt"],
        case["matrix_a"],
        case["matrix_b"],
        case["matrix_c"],
        case["d_weight"],
        z=case["z"] if bool(config.get("has_z", False)) else None,
        dt_bias=case["dt_bias"] if bool(config.get("has_dt_bias", True)) else None,
        dt_softplus=bool(config.get("dt_softplus", False)),
        state_batch_indices=(
            case["state_indices"] if bool(config.get("has_state_indices", True)) else None
        ),
        dst_state_batch_indices=(
            case["dst_indices"] if bool(config.get("has_dst_indices", False)) else None
        ),
        pad_slot_id=-1,
        state_scale=case["flashinfer_state_scale"] if scale_state else None,
        out=source_out,
        disable_state_update=not bool(config.get("update_state", True)),
        intermediate_states_buffer=(
            case["flashinfer_intermediate_states"]
            if bool(config.get("has_intermediate_states", False))
            else None
        ),
        intermediate_state_indices=(
            case["intermediate_state_indices"]
            if bool(config.get("has_intermediate_states", False))
            else None
        ),
        intermediate_state_scales=(
            case["flashinfer_intermediate_state_scales"]
            if bool(config.get("has_intermediate_states", False)) and scale_state
            else None
        ),
        rand_seed=case["rand_seed"] if int(config.get("philox_rounds", 0)) else None,
        philox_rounds=int(config.get("philox_rounds", 0)),
        cache_steps=int(config["tokens"]),
        algorithm="simple",
        cu_seqlens=(case["cu_seqlens"] if str(config["mode"]).startswith("varlen") else None),
        num_accepted_tokens=(
            case["num_accepted_tokens"]
            if bool(config.get("has_num_accepted_tokens", False))
            else None
        ),
    )
    if source_out is None:
        case["flashinfer_output"].copy_(result)
    return result


def _assert_case_close(case: dict[str, Any]) -> None:
    config = case["config"]
    scale_state = str(config["state_dtype"]) == "int16"
    atol = 0.1 if scale_state else 2e-2
    rtol = 1e-2 if scale_state else 2e-2
    torch.testing.assert_close(case["tirx_output"], case["flashinfer_output"], atol=atol, rtol=rtol)

    stride_factor = int(config.get("state_stride_factor", 1))
    tirx_state = case["tirx_state_storage"][::stride_factor]
    reference_state = case["flashinfer_state_storage"][::stride_factor]
    if scale_state:
        torch.testing.assert_close(
            case["tirx_state_scale"], case["flashinfer_state_scale"], atol=2e-5, rtol=2e-4
        )
        tirx_state = tirx_state.float() * case["tirx_state_scale"][..., None]
        reference_state = reference_state.float() * case["flashinfer_state_scale"][..., None]
        torch.testing.assert_close(tirx_state, reference_state, atol=0.1, rtol=1e-2)
    else:
        state_atol = 2e-3 if str(config["state_dtype"]) == "float32" else 2e-2
        torch.testing.assert_close(tirx_state, reference_state, atol=state_atol, rtol=2e-2)

    if bool(config.get("has_intermediate_states", False)):
        tirx_intermediate = case["tirx_intermediate_states"]
        reference_intermediate = case["flashinfer_intermediate_states"]
        if scale_state:
            torch.testing.assert_close(
                case["tirx_intermediate_state_scales"],
                case["flashinfer_intermediate_state_scales"],
                atol=2e-5,
                rtol=2e-4,
            )
            tirx_intermediate = (
                tirx_intermediate.float() * case["tirx_intermediate_state_scales"][..., None]
            )
            reference_intermediate = (
                reference_intermediate.float()
                * case["flashinfer_intermediate_state_scales"][..., None]
            )
            torch.testing.assert_close(
                tirx_intermediate, reference_intermediate, atol=0.1, rtol=1e-2
            )
        else:
            torch.testing.assert_close(
                tirx_intermediate, reference_intermediate, atol=2e-2, rtol=2e-2
            )


def run_test(**kwargs: Any) -> None:
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    executable = compile_kernel(get_kernel(**kwargs))
    executable(*_tirx_args(case))
    torch.cuda.synchronize()
    _run_reference(case)
    torch.cuda.synchronize()
    _assert_case_close(case)


def run_bench(
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 1,
    cooldown_s: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
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
