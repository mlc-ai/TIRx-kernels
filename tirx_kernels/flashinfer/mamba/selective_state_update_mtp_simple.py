# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's selective-state-update MTP simple kernel.

Upstream source: include/flashinfer/mamba/kernel_selective_state_update_mtp_simple.cuh.
"""

from __future__ import annotations

import functools
from typing import Any
from unittest import SkipTest

import torch

import tvm
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T


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


def _builder_enter(frame):
    """Enter a flat builder frame until its enclosing PrimFunc completes."""
    frame.add_callback(lambda: frame.__exit__(None, None, None))
    frame.__enter__()


def _builder_emit(value):
    """Match TVMScript expression-statement emission in direct builder code."""
    if value is None or isinstance(value, tvm.ir.Var):
        return
    if tvm.ir.is_prim_expr(value) or isinstance(value, tvm.ir.Call):
        T.evaluate(value)


KERNEL_META = {
    "name": "selective_state_update_mtp_simple",
    "category": "flashinfer",
    "compute_capability": 10,
}

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


def _build_selective_state_update_mtp_simple(
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
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("_selective_state_update_mtp_simple")
            state_h = T.arg("state_h", T.handle())
            state_scale_h = T.arg("state_scale_h", T.handle())
            x_h = T.arg("x_h", T.handle())
            dt_h = T.arg("dt_h", T.handle())
            matrix_a_h = T.arg("matrix_a_h", T.handle())
            matrix_b_h = T.arg("matrix_b_h", T.handle())
            matrix_c_h = T.arg("matrix_c_h", T.handle())
            d_h = T.arg("d_h", T.handle())
            z_h = T.arg("z_h", T.handle())
            dt_bias_h = T.arg("dt_bias_h", T.handle())
            state_indices_h = T.arg("state_indices_h", T.handle())
            dst_indices_h = T.arg("dst_indices_h", T.handle())
            intermediate_states_h = T.arg("intermediate_states_h", T.handle())
            intermediate_indices_h = T.arg("intermediate_indices_h", T.handle())
            intermediate_scales_h = T.arg("intermediate_scales_h", T.handle())
            cu_seqlens_h = T.arg("cu_seqlens_h", T.handle())
            num_accepted_tokens_h = T.arg("num_accepted_tokens_h", T.handle())
            rand_seed_h = T.arg("rand_seed_h", T.handle())
            output_h = T.arg("output_h", T.handle())
            state_stride_batch = T.arg("state_stride_batch", T.int64())
            state_scale_stride_batch = T.arg("state_scale_stride_batch", T.int64())
            x_stride_batch = T.arg("x_stride_batch", T.int64())
            x_stride_mtp = T.arg("x_stride_mtp", T.int64())
            dt_stride_batch = T.arg("dt_stride_batch", T.int64())
            dt_stride_mtp = T.arg("dt_stride_mtp", T.int64())
            b_stride_batch = T.arg("b_stride_batch", T.int64())
            b_stride_mtp = T.arg("b_stride_mtp", T.int64())
            c_stride_batch = T.arg("c_stride_batch", T.int64())
            c_stride_mtp = T.arg("c_stride_mtp", T.int64())
            z_stride_batch = T.arg("z_stride_batch", T.int64())
            z_stride_mtp = T.arg("z_stride_mtp", T.int64())
            out_stride_batch = T.arg("out_stride_batch", T.int64())
            out_stride_mtp = T.arg("out_stride_mtp", T.int64())
            state_indices_stride_batch = T.arg("state_indices_stride_batch", T.int64())
            state_indices_stride_t = T.arg("state_indices_stride_t", T.int64())
            dst_indices_stride_batch = T.arg("dst_indices_stride_batch", T.int64())
            dst_indices_stride_t = T.arg("dst_indices_stride_t", T.int64())
            cache_steps = T.arg("cache_steps", T.int32())
            nheads_runtime = T.arg("nheads_runtime", T.int32())
            ngroups_runtime = T.arg("ngroups_runtime", T.int32())
            dt_softplus = T.arg("dt_softplus", T.int32())
            update_state = T.arg("update_state", T.int32())
            pad_slot_id = T.arg("pad_slot_id", T.int32())
            state = _builder_name(
                "state", T.match_buffer(state_h, (STATE_ELEMENTS,), STATE_DTYPE, scope="global")
            )
            state_scale = _builder_name(
                "state_scale",
                T.match_buffer(state_scale_h, (SCALE_ELEMENTS,), "float32", scope="global"),
            )
            x = _builder_name("x", T.match_buffer(x_h, (X_ELEMENTS,), "bfloat16", scope="global"))
            dt = _builder_name(
                "dt", T.match_buffer(dt_h, (DT_ELEMENTS,), WEIGHT_DTYPE, scope="global")
            )
            matrix_a = _builder_name(
                "matrix_a", T.match_buffer(matrix_a_h, (NHEADS,), "float32", scope="global")
            )
            matrix_b = _builder_name(
                "matrix_b", T.match_buffer(matrix_b_h, (BC_ELEMENTS,), "bfloat16", scope="global")
            )
            matrix_c = _builder_name(
                "matrix_c", T.match_buffer(matrix_c_h, (BC_ELEMENTS,), "bfloat16", scope="global")
            )
            d_weight = _builder_name(
                "d_weight", T.match_buffer(d_h, (NHEADS,), WEIGHT_DTYPE, scope="global")
            )
            z = _builder_name("z", T.match_buffer(z_h, (X_ELEMENTS,), "bfloat16", scope="global"))
            dt_bias = _builder_name(
                "dt_bias", T.match_buffer(dt_bias_h, (NHEADS,), WEIGHT_DTYPE, scope="global")
            )
            state_indices = _builder_name(
                "state_indices",
                T.match_buffer(state_indices_h, (INDEX_ELEMENTS,), INDEX_DTYPE, scope="global"),
            )
            dst_indices = _builder_name(
                "dst_indices",
                T.match_buffer(dst_indices_h, (INDEX_ELEMENTS,), INDEX_DTYPE, scope="global"),
            )
            intermediate_states = _builder_name(
                "intermediate_states",
                T.match_buffer(
                    intermediate_states_h, (INTERMEDIATE_ELEMENTS,), STATE_DTYPE, scope="global"
                ),
            )
            intermediate_indices = _builder_name(
                "intermediate_indices",
                T.match_buffer(
                    intermediate_indices_h, (ACCEPTED_ELEMENTS,), INDEX_DTYPE, scope="global"
                ),
            )
            intermediate_scales = _builder_name(
                "intermediate_scales",
                T.match_buffer(
                    intermediate_scales_h, (INTERMEDIATE_SCALE_ELEMENTS,), "float32", scope="global"
                ),
            )
            cu_seqlens = _builder_name(
                "cu_seqlens",
                T.match_buffer(
                    cu_seqlens_h, (CU_SEQLENS_ELEMENTS,), CU_SEQLENS_DTYPE, scope="global"
                ),
            )
            num_accepted_tokens = _builder_name(
                "num_accepted_tokens",
                T.match_buffer(
                    num_accepted_tokens_h, (ACCEPTED_ELEMENTS,), ACCEPTED_DTYPE, scope="global"
                ),
            )
            rand_seed = _builder_name(
                "rand_seed", T.match_buffer(rand_seed_h, (1,), "int64", scope="global")
            )
            output = _builder_name(
                "output", T.match_buffer(output_h, (X_ELEMENTS,), "bfloat16", scope="global")
            )
            _builder_emit(T.device_entry())
            _builder_values_129 = T.cta_id([BATCH, NHEADS, CTAS_PER_HEAD])
            seq_idx, head, cta_z = _builder_values_129
            IRBuilder.name("cta_z", cta_z)
            IRBuilder.name("head", head)
            IRBuilder.name("seq_idx", seq_idx)
            _builder_values_130 = T.thread_id([32, 4])
            lane, warp = _builder_values_130
            IRBuilder.name("lane", lane)
            IRBuilder.name("warp", warp)
            flat_tid = _builder_scalar("flat_tid", warp * 32 + lane, dtype="int32")
            dim_offset = _builder_scalar("dim_offset", cta_z * DIM_PER_CTA, dtype="int32")
            kv_group = _builder_scalar("kv_group", head // HEADS_PER_GROUP, dtype="int32")
            bos = _builder_scalar("bos", 0, dtype="int32")
            seq_len = _builder_scalar("seq_len", NTOKENS, dtype="int32")
            if HAS_CU_SEQLENS:
                T.buffer_store(bos.buffer, T.cast(cu_seqlens[seq_idx], "int32"), [0])
                eos = _builder_scalar(
                    "eos", T.cast(cu_seqlens[seq_idx + 1], "int32"), dtype="int32"
                )
                T.buffer_store(seq_len.buffer, eos - bos, [0])
            with T.If(seq_len > 0):
                with T.Then():
                    init_token_idx = _builder_scalar("init_token_idx", 0, dtype="int32")
                    if HAS_NUM_ACCEPTED_TOKENS:
                        accepted = _builder_scalar(
                            "accepted", T.cast(num_accepted_tokens[seq_idx], "int32"), dtype="int32"
                        )
                        T.buffer_store(
                            init_token_idx.buffer,
                            T.if_then_else(accepted > 1, accepted - 1, 0),
                            [0],
                        )
                    state_batch = _builder_alloc_scalar("state_batch", "int64")
                    if HAS_STATE_INDICES:
                        T.buffer_store(
                            state_batch.buffer,
                            T.cast(
                                state_indices[
                                    T.cast(seq_idx, "int64") * state_indices_stride_batch
                                    + T.cast(init_token_idx, "int64") * state_indices_stride_t
                                ],
                                "int64",
                            ),
                            [0],
                        )
                    else:
                        T.buffer_store(state_batch.buffer, T.cast(seq_idx, "int64"), [0])
                    is_pad = _builder_scalar(
                        "is_pad",
                        T.if_then_else(state_batch != T.cast(pad_slot_id, "int64"), 0, 1),
                        dtype="int32",
                    )
                    state_head_offset = _builder_scalar(
                        "state_head_offset",
                        state_batch * state_stride_batch + T.cast(head * DIM * DSTATE, "int64"),
                        dtype="int64",
                    )
                    state_ptr_offset_i32 = _builder_scalar(
                        "state_ptr_offset_i32", T.cast(state_head_offset, "int32"), dtype="int32"
                    )
                    shared_raw = _builder_name(
                        "shared_raw",
                        T.alloc_buffer((SHARED_BYTES,), "uint8", scope="shared", align=128),
                    )
                    s_b = _builder_name(
                        "s_b",
                        T.decl_buffer(
                            (NTOKENS * DSTATE_PAD,),
                            "bfloat16",
                            data=shared_raw.data,
                            scope="shared",
                            byte_offset=OFF_B,
                            align=128,
                        ),
                    )
                    s_c = _builder_name(
                        "s_c",
                        T.decl_buffer(
                            (NTOKENS * DSTATE_PAD,),
                            "bfloat16",
                            data=shared_raw.data,
                            scope="shared",
                            byte_offset=OFF_C,
                            align=128,
                        ),
                    )
                    s_x = _builder_name(
                        "s_x",
                        T.decl_buffer(
                            (NTOKENS * DIM_PER_CTA,),
                            "bfloat16",
                            data=shared_raw.data,
                            scope="shared",
                            byte_offset=OFF_X,
                            align=128,
                        ),
                    )
                    s_dt = _builder_name(
                        "s_dt",
                        T.decl_buffer(
                            (NTOKENS,),
                            "float32",
                            data=shared_raw.data,
                            scope="shared",
                            byte_offset=OFF_DT,
                            align=4,
                        ),
                    )
                    s_out = _builder_name(
                        "s_out",
                        T.decl_buffer(
                            (NTOKENS * DIM_PER_CTA,),
                            "float32",
                            data=shared_raw.data,
                            scope="shared",
                            byte_offset=OFF_OUT,
                            align=4,
                        ),
                    )
                    s_dst_slots = _builder_name(
                        "s_dst_slots",
                        T.decl_buffer(
                            (NTOKENS,),
                            "int64",
                            data=shared_raw.data,
                            scope="shared",
                            byte_offset=OFF_DST_SLOTS,
                            align=8,
                        ),
                    )
                    s_state = _builder_name(
                        "s_state",
                        T.decl_buffer(
                            (STATE_STAGES * 16 * DSTATE_PAD,),
                            STATE_DTYPE,
                            data=shared_raw.data,
                            scope="shared",
                            byte_offset=OFF_STATE_IN,
                            align=128,
                        ),
                    )
                    a_value = _builder_scalar(
                        "a_value",
                        T.reinterpret("float32", _global_load_u32(matrix_a, head)),
                        dtype="float32",
                    )
                    d_value = _builder_scalar("d_value", 0.0, dtype="float32")
                    if HAS_D:
                        T.buffer_store(
                            d_value.buffer, _load_weight(d_weight, head, WEIGHT_DTYPE), [0]
                        )

                    def run_simple(IS_PAD: T.constexpr):
                        if HAS_CU_SEQLENS:
                            b_base = _builder_scalar(
                                "b_base", T.cast(bos, "int64") * b_stride_batch, dtype="int64"
                            )
                            b_tstride = _builder_scalar("b_tstride", b_stride_batch, dtype="int64")
                            c_base = _builder_scalar(
                                "c_base", T.cast(bos, "int64") * c_stride_batch, dtype="int64"
                            )
                            c_tstride = _builder_scalar("c_tstride", c_stride_batch, dtype="int64")
                            x_base = _builder_scalar(
                                "x_base", T.cast(bos, "int64") * x_stride_batch, dtype="int64"
                            )
                            x_tstride = _builder_scalar("x_tstride", x_stride_batch, dtype="int64")
                            dt_base = _builder_scalar(
                                "dt_base", T.cast(bos, "int64") * dt_stride_batch, dtype="int64"
                            )
                            dt_tstride = _builder_scalar(
                                "dt_tstride", dt_stride_batch, dtype="int64"
                            )
                        else:
                            b_base = _builder_scalar(
                                "b_base", T.cast(seq_idx, "int64") * b_stride_batch, dtype="int64"
                            )
                            b_tstride = _builder_scalar("b_tstride", b_stride_mtp, dtype="int64")
                            c_base = _builder_scalar(
                                "c_base", T.cast(seq_idx, "int64") * c_stride_batch, dtype="int64"
                            )
                            c_tstride = _builder_scalar("c_tstride", c_stride_mtp, dtype="int64")
                            x_base = _builder_scalar(
                                "x_base", T.cast(seq_idx, "int64") * x_stride_batch, dtype="int64"
                            )
                            x_tstride = _builder_scalar("x_tstride", x_stride_mtp, dtype="int64")
                            dt_base = _builder_scalar(
                                "dt_base", T.cast(seq_idx, "int64") * dt_stride_batch, dtype="int64"
                            )
                            dt_tstride = _builder_scalar("dt_tstride", dt_stride_mtp, dtype="int64")
                        with T.If(warp == 0):
                            with T.Then():
                                with T.serial((NTOKENS * DSTATE // 8 + 31) // 32) as load_iter:
                                    packed_i = _builder_scalar(
                                        "packed_i", lane + load_iter * 32, dtype="int32"
                                    )
                                    with T.If(packed_i < NTOKENS * DSTATE // 8):
                                        with T.Then():
                                            step = _builder_scalar(
                                                "step", packed_i // (DSTATE // 8), dtype="int32"
                                            )
                                            col = _builder_scalar(
                                                "col", packed_i % (DSTATE // 8) * 8, dtype="int32"
                                            )
                                            with T.If(step < seq_len):
                                                with T.Then():
                                                    _builder_emit(
                                                        T.ptx["cp.async.cg.shared.global"](
                                                            s_b.ptr_to([step * DSTATE_PAD + col]),
                                                            matrix_b.ptr_to(
                                                                [
                                                                    b_base
                                                                    + T.cast(step, "int64")
                                                                    * b_tstride
                                                                    + kv_group * DSTATE
                                                                    + col
                                                                ]
                                                            ),
                                                            16,
                                                        )
                                                    )
                            with T.Else():
                                with T.If(warp == 1):
                                    with T.Then():
                                        with T.serial(
                                            (NTOKENS * DSTATE // 8 + 31) // 32
                                        ) as load_iter:
                                            packed_i = _builder_scalar(
                                                "packed_i", lane + load_iter * 32, dtype="int32"
                                            )
                                            with T.If(packed_i < NTOKENS * DSTATE // 8):
                                                with T.Then():
                                                    step = _builder_scalar(
                                                        "step",
                                                        packed_i // (DSTATE // 8),
                                                        dtype="int32",
                                                    )
                                                    col = _builder_scalar(
                                                        "col",
                                                        packed_i % (DSTATE // 8) * 8,
                                                        dtype="int32",
                                                    )
                                                    with T.If(step < seq_len):
                                                        with T.Then():
                                                            _builder_emit(
                                                                T.ptx["cp.async.cg.shared.global"](
                                                                    s_c.ptr_to(
                                                                        [step * DSTATE_PAD + col]
                                                                    ),
                                                                    matrix_c.ptr_to(
                                                                        [
                                                                            c_base
                                                                            + T.cast(step, "int64")
                                                                            * c_tstride
                                                                            + kv_group * DSTATE
                                                                            + col
                                                                        ]
                                                                    ),
                                                                    16,
                                                                )
                                                            )
                        with T.serial((NTOKENS + 3) // 4) as step_iter:
                            step = _builder_scalar("step", warp + step_iter * 4, dtype="int32")
                            with T.If(step < seq_len):
                                with T.Then():
                                    with T.serial((DIM_PER_CTA // 8 + 31) // 32) as col_iter:
                                        col = _builder_scalar(
                                            "col", (lane + col_iter * 32) * 8, dtype="int32"
                                        )
                                        with T.If(col < DIM_PER_CTA):
                                            with T.Then():
                                                _builder_emit(
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
                                                )
                        if not IS_PAD:
                            with T.serial(
                                (16 * DSTATE // (16 // STATE_BYTES) + 127) // 128
                            ) as state_load_iter:
                                packed_i = _builder_scalar(
                                    "packed_i", flat_tid + state_load_iter * 128, dtype="int32"
                                )
                                with T.If(packed_i < 16 * DSTATE // (16 // STATE_BYTES)):
                                    with T.Then():
                                        state_row = _builder_scalar(
                                            "state_row",
                                            packed_i // (DSTATE // (16 // STATE_BYTES)),
                                            dtype="int32",
                                        )
                                        state_col = _builder_scalar(
                                            "state_col",
                                            packed_i
                                            % (DSTATE // (16 // STATE_BYTES))
                                            * (16 // STATE_BYTES),
                                            dtype="int32",
                                        )
                                        _builder_emit(
                                            T.ptx["cp.async.cg.shared.global"](
                                                s_state.ptr_to(
                                                    [state_row * DSTATE_PAD + state_col]
                                                ),
                                                state.ptr_to(
                                                    [
                                                        state_head_offset
                                                        + (dim_offset + state_row) * DSTATE
                                                        + state_col
                                                    ]
                                                ),
                                                16,
                                            )
                                        )
                        with T.If(flat_tid < seq_len):
                            with T.Then():
                                dt_value = _builder_scalar(
                                    "dt_value",
                                    _load_weight(
                                        dt,
                                        dt_base + T.cast(flat_tid, "int64") * dt_tstride + head,
                                        WEIGHT_DTYPE,
                                    ),
                                    dtype="float32",
                                )
                                if HAS_DT_BIAS:
                                    T.buffer_store(
                                        dt_value.buffer,
                                        _add(dt_value, _load_weight(dt_bias, head, WEIGHT_DTYPE)),
                                        [0],
                                    )
                                with T.If(dt_softplus != 0):
                                    with T.Then():
                                        with T.If(dt_value <= T.float32(20.0)):
                                            with T.Then():
                                                dt_exp = _builder_scalar(
                                                    "dt_exp",
                                                    _exp2(_mul(dt_value, T.float32(_LOG2_E))),
                                                    dtype="float32",
                                                )
                                                T.buffer_store(
                                                    dt_value.buffer,
                                                    _mul(
                                                        _log2(_add(T.float32(1.0), dt_exp)),
                                                        T.float32(_LN_2),
                                                    ),
                                                    [0],
                                                )
                                _builder_emit(
                                    T.evaluate(
                                        T.ptx.st.shared.b32(
                                            s_dt.ptr_to([flat_tid]),
                                            T.reinterpret("uint32", dt_value),
                                        )
                                    )
                                )
                        with T.If(flat_tid < NTOKENS):
                            with T.Then():
                                step = _builder_scalar("step", flat_tid, dtype="int32")
                                dst_slot = _builder_scalar("dst_slot", -1, dtype="int64")
                                if not IS_PAD:
                                    with T.If(T.And(T.bool(True), step < seq_len)):
                                        with T.Then():
                                            if HAS_DST_INDICES:
                                                dst_index = _builder_scalar(
                                                    "dst_index",
                                                    T.cast(
                                                        dst_indices[
                                                            T.cast(seq_idx, "int64")
                                                            * dst_indices_stride_batch
                                                            + T.cast(step, "int64")
                                                            * dst_indices_stride_t
                                                        ],
                                                        "int64",
                                                    ),
                                                    dtype="int64",
                                                )
                                                with T.If(
                                                    dst_index != T.cast(pad_slot_id, "int64")
                                                ):
                                                    with T.Then():
                                                        T.buffer_store(
                                                            dst_slot.buffer, dst_index, [0]
                                                        )
                                            elif HAS_INTERMEDIATE_STATES:
                                                intermediate_index = _builder_scalar(
                                                    "intermediate_index", state_batch, dtype="int64"
                                                )
                                                if HAS_INTERMEDIATE_INDICES:
                                                    T.buffer_store(
                                                        intermediate_index.buffer,
                                                        T.cast(
                                                            intermediate_indices[seq_idx], "int64"
                                                        ),
                                                        [0],
                                                    )
                                                T.buffer_store(
                                                    dst_slot.buffer,
                                                    intermediate_index
                                                    * T.cast(cache_steps, "int64")
                                                    + step,
                                                    [0],
                                                )
                                            else:
                                                with T.If(
                                                    T.And(step == seq_len - 1, update_state != 0)
                                                ):
                                                    with T.Then():
                                                        T.buffer_store(
                                                            dst_slot.buffer, state_batch, [0]
                                                        )
                                T.buffer_store(s_dst_slots, dst_slot, [step])
                        _builder_emit(T.ptx.cp.async_.commit_group())
                        _builder_emit(T.ptx.cp.async_.wait_group(0))
                        _builder_emit(T.ptx.bar.sync(T.uint32(0)))
                        random_seed = _builder_scalar("random_seed", 0, dtype="int64")
                        if PHILOX_ROUNDS > 0:
                            T.buffer_store(random_seed.buffer, rand_seed[0], [0])
                        member = _builder_scalar("member", lane % 8, dtype="int32")
                        row_group = _builder_scalar("row_group", lane // 8, dtype="int32")
                        with T.serial(NUM_PASSES) as pass_idx:
                            pass_row = _builder_scalar(
                                "pass_row", warp * 4 + row_group, dtype="int32"
                            )
                            local_row = _builder_scalar(
                                "local_row", pass_idx * 16 + pass_row, dtype="int32"
                            )
                            row_d = _builder_scalar("row_d", dim_offset + local_row, dtype="int32")
                            state_stage = _builder_scalar(
                                "state_stage", pass_idx % STATE_STAGES, dtype="int32"
                            )
                            decode_scale = _builder_scalar("decode_scale", 1.0, dtype="float32")
                            if SCALE_STATE and not IS_PAD:
                                T.buffer_store(
                                    decode_scale.buffer,
                                    T.reinterpret(
                                        "float32",
                                        _global_load_u32(
                                            state_scale,
                                            state_batch * state_scale_stride_batch
                                            + head * DIM
                                            + row_d,
                                        ),
                                    ),
                                    [0],
                                )
                            r_state = _builder_name(
                                "r_state",
                                T.alloc_local((NUM_TILES * PAIRS_PER_TILE_MEMBER,), "uint64"),
                            )
                            with T.unroll(NUM_TILES) as tile_idx:
                                member_col = _builder_scalar(
                                    "member_col",
                                    tile_idx * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER,
                                    dtype="int32",
                                )
                                with T.If(T.And(member_col < DSTATE, T.bool(not IS_PAD))):
                                    with T.Then():
                                        state_words = _builder_name(
                                            "state_words", T.alloc_local((4,), "uint32")
                                        )
                                        _builder_emit(
                                            T.evaluate(
                                                T.ptx.ld.shared.v4.b32(
                                                    state_words[0],
                                                    state_words[1],
                                                    state_words[2],
                                                    state_words[3],
                                                    s_state.ptr_to(
                                                        [
                                                            (state_stage * 16 + pass_row)
                                                            * DSTATE_PAD
                                                            + member_col
                                                        ]
                                                    ),
                                                )
                                            )
                                        )
                                        with T.unroll(PAIRS_PER_TILE_MEMBER) as pair_idx:
                                            if STATE_DTYPE == "bfloat16":
                                                state_pair = _builder_scalar(
                                                    "state_pair",
                                                    _bf16_word_to_f32x2(state_words[pair_idx]),
                                                    dtype="uint64",
                                                )
                                            elif STATE_DTYPE == "float16":
                                                state_pair = _builder_scalar(
                                                    "state_pair",
                                                    T.cuda.make_float2(
                                                        _f16_to_f32(
                                                            _extract_u16(
                                                                state_words[pair_idx], False
                                                            )
                                                        ),
                                                        _f16_to_f32(
                                                            _extract_u16(
                                                                state_words[pair_idx], True
                                                            )
                                                        ),
                                                    ),
                                                    dtype="uint64",
                                                )
                                            elif STATE_DTYPE == "int16":
                                                state_pair = _builder_scalar(
                                                    "state_pair",
                                                    T.cuda.make_float2(
                                                        _i16_to_f32(
                                                            _extract_u16(
                                                                state_words[pair_idx], False
                                                            )
                                                        ),
                                                        _i16_to_f32(
                                                            _extract_u16(
                                                                state_words[pair_idx], True
                                                            )
                                                        ),
                                                    ),
                                                    dtype="uint64",
                                                )
                                            else:
                                                state_pair = _builder_scalar(
                                                    "state_pair",
                                                    T.cuda.make_float2(
                                                        T.reinterpret(
                                                            "float32", state_words[pair_idx * 2]
                                                        ),
                                                        T.reinterpret(
                                                            "float32", state_words[pair_idx * 2 + 1]
                                                        ),
                                                    ),
                                                    dtype="uint64",
                                                )
                                            if SCALE_STATE:
                                                _builder_emit(
                                                    T.ptx.mul.f32x2(
                                                        state_pair,
                                                        state_pair,
                                                        T.cuda.make_float2(
                                                            decode_scale, decode_scale
                                                        ),
                                                    )
                                                )
                                            T.buffer_store(
                                                r_state,
                                                state_pair,
                                                [tile_idx * PAIRS_PER_TILE_MEMBER + pair_idx],
                                            )
                                    with T.Else():
                                        with T.unroll(PAIRS_PER_TILE_MEMBER) as pair_idx:
                                            T.buffer_store(
                                                r_state,
                                                T.cuda.make_float2(T.float32(0.0), T.float32(0.0)),
                                                [tile_idx * PAIRS_PER_TILE_MEMBER + pair_idx],
                                            )
                            b_step = _builder_scalar("b_step", 0, dtype="int32")
                            c_step = _builder_scalar("c_step", 0, dtype="int32")
                            x_step = _builder_scalar("x_step", 0, dtype="int32")
                            dt_step = _builder_scalar("dt_step", 0, dtype="int32")
                            out_step = _builder_scalar("out_step", 0, dtype="int32")
                            with T.serial(NTOKENS) as step:
                                with T.If(step < seq_len):
                                    with T.Then():
                                        dst_slot = _builder_scalar(
                                            "dst_slot", s_dst_slots[step], dtype="int64"
                                        )
                                        dt_value = _builder_scalar(
                                            "dt_value",
                                            T.reinterpret(
                                                "float32", _shared_load_u32(s_dt, dt_step)
                                            ),
                                            dtype="float32",
                                        )
                                        da_value = _builder_scalar(
                                            "da_value",
                                            _exp2(
                                                _mul(_mul(a_value, dt_value), T.float32(_LOG2_E))
                                            ),
                                            dtype="float32",
                                        )
                                        x_value = _builder_scalar(
                                            "x_value",
                                            _bf16_to_f32(_shared_load_u16(s_x, x_step + local_row)),
                                            dtype="float32",
                                        )
                                        dtx_value = _builder_scalar(
                                            "dtx_value", _mul(dt_value, x_value), dtype="float32"
                                        )
                                        out_pair = _builder_scalar(
                                            "out_pair",
                                            T.cuda.make_float2(T.float32(0.0), T.float32(0.0)),
                                            dtype="uint64",
                                        )
                                        with T.unroll(NUM_TILES) as tile_idx:
                                            member_col = _builder_scalar(
                                                "member_col",
                                                tile_idx * ELEMS_PER_TILE
                                                + member * ELEMS_PER_TILE_MEMBER,
                                                dtype="int32",
                                            )
                                            with T.If(member_col < DSTATE):
                                                with T.Then():
                                                    b_words = _builder_name(
                                                        "b_words", T.alloc_local((4,), "uint32")
                                                    )
                                                    c_words = _builder_name(
                                                        "c_words", T.alloc_local((4,), "uint32")
                                                    )
                                                    if ELEMS_PER_TILE_MEMBER == 4:
                                                        _builder_emit(
                                                            T.evaluate(
                                                                T.ptx.ld.shared.v2.b32(
                                                                    b_words[0],
                                                                    b_words[1],
                                                                    s_b.ptr_to(
                                                                        [b_step + member_col]
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                        _builder_emit(
                                                            T.evaluate(
                                                                T.ptx.ld.shared.v2.b32(
                                                                    c_words[0],
                                                                    c_words[1],
                                                                    s_c.ptr_to(
                                                                        [c_step + member_col]
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                    else:
                                                        _builder_emit(
                                                            T.evaluate(
                                                                T.ptx.ld.shared.v4.b32(
                                                                    b_words[0],
                                                                    b_words[1],
                                                                    b_words[2],
                                                                    b_words[3],
                                                                    s_b.ptr_to(
                                                                        [b_step + member_col]
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                        _builder_emit(
                                                            T.evaluate(
                                                                T.ptx.ld.shared.v4.b32(
                                                                    c_words[0],
                                                                    c_words[1],
                                                                    c_words[2],
                                                                    c_words[3],
                                                                    s_c.ptr_to(
                                                                        [c_step + member_col]
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                    with T.unroll(
                                                        PAIRS_PER_TILE_MEMBER
                                                    ) as pair_idx:
                                                        b_pair = _builder_scalar(
                                                            "b_pair",
                                                            _bf16_word_to_f32x2(b_words[pair_idx]),
                                                            dtype="uint64",
                                                        )
                                                        c_pair = _builder_scalar(
                                                            "c_pair",
                                                            _bf16_word_to_f32x2(c_words[pair_idx]),
                                                            dtype="uint64",
                                                        )
                                                        dbx_pair = _builder_alloc_scalar(
                                                            "dbx_pair", "uint64"
                                                        )
                                                        _builder_emit(
                                                            T.ptx.mul.f32x2(
                                                                dbx_pair,
                                                                b_pair,
                                                                T.cuda.make_float2(
                                                                    dtx_value, dtx_value
                                                                ),
                                                            )
                                                        )
                                                        pair_index = _builder_scalar(
                                                            "pair_index",
                                                            tile_idx * PAIRS_PER_TILE_MEMBER
                                                            + pair_idx,
                                                            dtype="int32",
                                                        )
                                                        updated_state = _builder_alloc_scalar(
                                                            "updated_state", "uint64"
                                                        )
                                                        _builder_emit(
                                                            T.ptx.fma.rn.f32x2(
                                                                updated_state,
                                                                T.cuda.make_float2(
                                                                    da_value, da_value
                                                                ),
                                                                r_state[pair_index],
                                                                dbx_pair,
                                                            )
                                                        )
                                                        T.buffer_store(
                                                            r_state, updated_state, [pair_index]
                                                        )
                                                        _builder_emit(
                                                            T.ptx.fma.rn.f32x2(
                                                                out_pair,
                                                                updated_state,
                                                                c_pair,
                                                                out_pair,
                                                            )
                                                        )
                                        out_value = _builder_scalar(
                                            "out_value",
                                            _add(
                                                T.cuda.float2_x(out_pair), T.cuda.float2_y(out_pair)
                                            ),
                                            dtype="float32",
                                        )
                                        with T.unroll(3) as delta_idx:
                                            delta = _builder_scalar(
                                                "delta",
                                                T.shift_right(T.int32(4), delta_idx),
                                                dtype="int32",
                                            )
                                            peer_out = _builder_scalar(
                                                "peer_out",
                                                T.cuda.__shfl_down_sync(
                                                    T.uint32(4294967295), out_value, delta, 32
                                                ),
                                                dtype="float32",
                                            )
                                            T.buffer_store(
                                                out_value.buffer, _add(out_value, peer_out), [0]
                                            )
                                        with T.If(member == 0):
                                            with T.Then():
                                                row_output = _builder_scalar(
                                                    "row_output",
                                                    _fma(d_value, x_value, out_value),
                                                    dtype="float32",
                                                )
                                                _builder_emit(
                                                    T.evaluate(
                                                        T.ptx.st.shared.b32(
                                                            s_out.ptr_to([out_step + local_row]),
                                                            T.reinterpret("uint32", row_output),
                                                        )
                                                    )
                                                )
                                        T.buffer_store(b_step.buffer, b_step + DSTATE_PAD, [0])
                                        T.buffer_store(c_step.buffer, c_step + DSTATE_PAD, [0])
                                        T.buffer_store(x_step.buffer, x_step + DIM_PER_CTA, [0])
                                        T.buffer_store(dt_step.buffer, dt_step + 1, [0])
                                        T.buffer_store(out_step.buffer, out_step + DIM_PER_CTA, [0])
                                        with T.If(dst_slot != T.int64(-1)):
                                            with T.Then():
                                                encode_scale = _builder_scalar(
                                                    "encode_scale", 1.0, dtype="float32"
                                                )
                                                if SCALE_STATE:
                                                    local_max = _builder_scalar(
                                                        "local_max",
                                                        T.float32(_FLT_LOWEST),
                                                        dtype="float32",
                                                    )
                                                    with T.unroll(NUM_TILES) as tile_idx:
                                                        with T.unroll(
                                                            PAIRS_PER_TILE_MEMBER
                                                        ) as pair_idx:
                                                            col0 = _builder_scalar(
                                                                "col0",
                                                                tile_idx * ELEMS_PER_TILE
                                                                + member * ELEMS_PER_TILE_MEMBER
                                                                + pair_idx * 2,
                                                                dtype="int32",
                                                            )
                                                            with T.If(col0 < DSTATE):
                                                                with T.Then():
                                                                    state_pair = _builder_scalar(
                                                                        "state_pair",
                                                                        r_state[
                                                                            tile_idx
                                                                            * PAIRS_PER_TILE_MEMBER
                                                                            + pair_idx
                                                                        ],
                                                                        dtype="uint64",
                                                                    )
                                                                    T.buffer_store(
                                                                        local_max.buffer,
                                                                        _max(
                                                                            local_max,
                                                                            _max(
                                                                                _abs(
                                                                                    T.cuda.float2_x(
                                                                                        state_pair
                                                                                    )
                                                                                ),
                                                                                _abs(
                                                                                    T.cuda.float2_y(
                                                                                        state_pair
                                                                                    )
                                                                                ),
                                                                            ),
                                                                        ),
                                                                        [0],
                                                                    )
                                                    with T.unroll(3) as delta_idx:
                                                        delta = _builder_scalar(
                                                            "delta",
                                                            T.shift_right(T.int32(4), delta_idx),
                                                            dtype="int32",
                                                        )
                                                        peer_max = _builder_scalar(
                                                            "peer_max",
                                                            T.cuda.__shfl_down_sync(
                                                                T.uint32(4294967295),
                                                                local_max,
                                                                delta,
                                                                32,
                                                            ),
                                                            dtype="float32",
                                                        )
                                                        T.buffer_store(
                                                            local_max.buffer,
                                                            _max(local_max, peer_max),
                                                            [0],
                                                        )
                                                    leader_lane = _builder_scalar(
                                                        "leader_lane",
                                                        T.bitwise_and(lane, T.int32(-8)),
                                                        dtype="int32",
                                                    )
                                                    T.buffer_store(
                                                        local_max.buffer,
                                                        T.cuda.__shfl_sync(
                                                            T.uint32(4294967295),
                                                            local_max,
                                                            leader_lane,
                                                            32,
                                                        ),
                                                        [0],
                                                    )
                                                    with T.If(local_max != T.float32(0.0)):
                                                        with T.Then():
                                                            T.buffer_store(
                                                                encode_scale.buffer,
                                                                _div(T.float32(32767.0), local_max),
                                                                [0],
                                                            )
                                                if HAS_INTERMEDIATE_STATES:
                                                    dst_base = _builder_scalar(
                                                        "dst_base",
                                                        dst_slot
                                                        * T.cast(NHEADS * DIM * DSTATE, "int64")
                                                        + head * DIM * DSTATE
                                                        + row_d * DSTATE,
                                                        dtype="int64",
                                                    )
                                                else:
                                                    dst_base = _builder_scalar(
                                                        "dst_base",
                                                        dst_slot * state_stride_batch
                                                        + head * DIM * DSTATE
                                                        + row_d * DSTATE,
                                                        dtype="int64",
                                                    )
                                                with T.unroll(NUM_TILES) as tile_idx:
                                                    member_col = _builder_scalar(
                                                        "member_col",
                                                        tile_idx * ELEMS_PER_TILE
                                                        + member * ELEMS_PER_TILE_MEMBER,
                                                        dtype="int32",
                                                    )
                                                    with T.If(member_col < DSTATE):
                                                        with T.Then():
                                                            store_words = _builder_name(
                                                                "store_words",
                                                                T.alloc_local((4,), "uint32"),
                                                            )
                                                            random_words = _builder_name(
                                                                "random_words",
                                                                T.alloc_local((4,), "uint32"),
                                                            )
                                                            with T.unroll(
                                                                PAIRS_PER_TILE_MEMBER
                                                            ) as pair_idx:
                                                                pair_index = _builder_scalar(
                                                                    "pair_index",
                                                                    tile_idx * PAIRS_PER_TILE_MEMBER
                                                                    + pair_idx,
                                                                    dtype="int32",
                                                                )
                                                                state_pair = _builder_scalar(
                                                                    "state_pair",
                                                                    r_state[pair_index],
                                                                    dtype="uint64",
                                                                )
                                                                if SCALE_STATE:
                                                                    _builder_emit(
                                                                        T.ptx.mul.f32x2(
                                                                            state_pair,
                                                                            state_pair,
                                                                            T.cuda.make_float2(
                                                                                encode_scale,
                                                                                encode_scale,
                                                                            ),
                                                                        )
                                                                    )
                                                                    low_scaled = _builder_scalar(
                                                                        "low_scaled",
                                                                        _min(
                                                                            _max(
                                                                                T.cuda.float2_x(
                                                                                    state_pair
                                                                                ),
                                                                                T.float32(-32767.0),
                                                                            ),
                                                                            T.float32(32767.0),
                                                                        ),
                                                                        dtype="float32",
                                                                    )
                                                                    high_scaled = _builder_scalar(
                                                                        "high_scaled",
                                                                        _min(
                                                                            _max(
                                                                                T.cuda.float2_y(
                                                                                    state_pair
                                                                                ),
                                                                                T.float32(-32767.0),
                                                                            ),
                                                                            T.float32(32767.0),
                                                                        ),
                                                                        dtype="float32",
                                                                    )
                                                                    low_i32 = _builder_name(
                                                                        "low_i32",
                                                                        T.alloc_local(
                                                                            (1,), "int32"
                                                                        ),
                                                                    )
                                                                    high_i32 = _builder_name(
                                                                        "high_i32",
                                                                        T.alloc_local(
                                                                            (1,), "int32"
                                                                        ),
                                                                    )
                                                                    _builder_emit(
                                                                        T.evaluate(
                                                                            T.ptx.cvt.rni.ftz.s32.f32(
                                                                                low_i32[0],
                                                                                low_scaled,
                                                                            )
                                                                        )
                                                                    )
                                                                    _builder_emit(
                                                                        T.evaluate(
                                                                            T.ptx.cvt.rni.ftz.s32.f32(
                                                                                high_i32[0],
                                                                                high_scaled,
                                                                            )
                                                                        )
                                                                    )
                                                                    T.buffer_store(
                                                                        store_words,
                                                                        _prmt_5410(
                                                                            T.reinterpret(
                                                                                "uint32", low_i32[0]
                                                                            ),
                                                                            T.reinterpret(
                                                                                "uint32",
                                                                                high_i32[0],
                                                                            ),
                                                                        ),
                                                                        [pair_idx],
                                                                    )
                                                                elif PHILOX_ROUNDS > 0:
                                                                    element_idx = _builder_scalar(
                                                                        "element_idx",
                                                                        pair_idx * 2,
                                                                        dtype="int32",
                                                                    )
                                                                    with T.If(pair_idx % 2 == 0):
                                                                        with T.Then():
                                                                            offset_mad = (
                                                                                _builder_name(
                                                                                    "offset_mad",
                                                                                    T.alloc_local(
                                                                                        (1,),
                                                                                        "int32",
                                                                                    ),
                                                                                )
                                                                            )
                                                                            _builder_emit(
                                                                                T.ptx.mad.lo.s32(
                                                                                    offset_mad[0],
                                                                                    row_d,
                                                                                    T.int32(DSTATE),
                                                                                    state_ptr_offset_i32,
                                                                                )
                                                                            )
                                                                            random_offset = _builder_scalar(
                                                                                "random_offset",
                                                                                _add_s32(
                                                                                    offset_mad[0],
                                                                                    member_col
                                                                                    + element_idx,
                                                                                ),
                                                                                dtype="int32",
                                                                            )
                                                                            c0 = _builder_scalar(
                                                                                "c0",
                                                                                T.reinterpret(
                                                                                    "uint32",
                                                                                    random_offset,
                                                                                ),
                                                                                dtype="uint32",
                                                                            )
                                                                            c1_signed = (
                                                                                _builder_name(
                                                                                    "c1_signed",
                                                                                    T.alloc_local(
                                                                                        (1,),
                                                                                        "int32",
                                                                                    ),
                                                                                )
                                                                            )
                                                                            _builder_emit(
                                                                                T.ptx.shr.s32(
                                                                                    c1_signed[0],
                                                                                    random_offset,
                                                                                    T.uint32(31),
                                                                                )
                                                                            )
                                                                            c1 = _builder_scalar(
                                                                                "c1",
                                                                                T.reinterpret(
                                                                                    "uint32",
                                                                                    c1_signed[0],
                                                                                ),
                                                                                dtype="uint32",
                                                                            )
                                                                            c2 = _builder_scalar(
                                                                                "c2",
                                                                                0,
                                                                                dtype="uint32",
                                                                            )
                                                                            c3 = _builder_scalar(
                                                                                "c3",
                                                                                0,
                                                                                dtype="uint32",
                                                                            )
                                                                            seed_u64 = (
                                                                                _builder_scalar(
                                                                                    "seed_u64",
                                                                                    T.reinterpret(
                                                                                        "uint64",
                                                                                        random_seed,
                                                                                    ),
                                                                                    dtype="uint64",
                                                                                )
                                                                            )
                                                                            k0 = _builder_scalar(
                                                                                "k0",
                                                                                T.cast(
                                                                                    seed_u64,
                                                                                    "uint32",
                                                                                ),
                                                                                dtype="uint32",
                                                                            )
                                                                            k1 = _builder_scalar(
                                                                                "k1",
                                                                                T.cast(
                                                                                    T.shift_right(
                                                                                        seed_u64,
                                                                                        T.uint64(
                                                                                            32
                                                                                        ),
                                                                                    ),
                                                                                    "uint32",
                                                                                ),
                                                                                dtype="uint32",
                                                                            )
                                                                            with T.unroll(
                                                                                10
                                                                            ) as philox_round:
                                                                                old_c0 = _builder_scalar(
                                                                                    "old_c0",
                                                                                    c0,
                                                                                    dtype="uint32",
                                                                                )
                                                                                old_c2 = _builder_scalar(
                                                                                    "old_c2",
                                                                                    c2,
                                                                                    dtype="uint32",
                                                                                )
                                                                                next_c0 = _builder_scalar(
                                                                                    "next_c0",
                                                                                    T.bitwise_xor(
                                                                                        T.bitwise_xor(
                                                                                            _mul_hi_u32(
                                                                                                T.uint32(
                                                                                                    3449720151
                                                                                                ),
                                                                                                old_c2,
                                                                                            ),
                                                                                            c1,
                                                                                        ),
                                                                                        k0,
                                                                                    ),
                                                                                    dtype="uint32",
                                                                                )
                                                                                next_c2 = _builder_scalar(
                                                                                    "next_c2",
                                                                                    T.bitwise_xor(
                                                                                        T.bitwise_xor(
                                                                                            _mul_hi_u32(
                                                                                                T.uint32(
                                                                                                    3528531795
                                                                                                ),
                                                                                                old_c0,
                                                                                            ),
                                                                                            c3,
                                                                                        ),
                                                                                        k1,
                                                                                    ),
                                                                                    dtype="uint32",
                                                                                )
                                                                                next_c1 = _builder_scalar(
                                                                                    "next_c1",
                                                                                    _mul_lo_s32(
                                                                                        T.int32(
                                                                                            -845247145
                                                                                        ),
                                                                                        T.reinterpret(
                                                                                            "int32",
                                                                                            old_c2,
                                                                                        ),
                                                                                    ),
                                                                                    dtype="int32",
                                                                                )
                                                                                next_c3 = _builder_scalar(
                                                                                    "next_c3",
                                                                                    _mul_lo_s32(
                                                                                        T.int32(
                                                                                            -766435501
                                                                                        ),
                                                                                        T.reinterpret(
                                                                                            "int32",
                                                                                            old_c0,
                                                                                        ),
                                                                                    ),
                                                                                    dtype="int32",
                                                                                )
                                                                                next_k0 = _builder_scalar(
                                                                                    "next_k0",
                                                                                    _add_s32(
                                                                                        T.reinterpret(
                                                                                            "int32",
                                                                                            k0,
                                                                                        ),
                                                                                        T.int32(
                                                                                            -1640531527
                                                                                        ),
                                                                                    ),
                                                                                    dtype="int32",
                                                                                )
                                                                                next_k1 = _builder_scalar(
                                                                                    "next_k1",
                                                                                    _add_s32(
                                                                                        T.reinterpret(
                                                                                            "int32",
                                                                                            k1,
                                                                                        ),
                                                                                        T.int32(
                                                                                            -1150833019
                                                                                        ),
                                                                                    ),
                                                                                    dtype="int32",
                                                                                )
                                                                                T.buffer_store(
                                                                                    c0.buffer,
                                                                                    next_c0,
                                                                                    [0],
                                                                                )
                                                                                T.buffer_store(
                                                                                    c1.buffer,
                                                                                    T.reinterpret(
                                                                                        "uint32",
                                                                                        next_c1,
                                                                                    ),
                                                                                    [0],
                                                                                )
                                                                                T.buffer_store(
                                                                                    c2.buffer,
                                                                                    next_c2,
                                                                                    [0],
                                                                                )
                                                                                T.buffer_store(
                                                                                    c3.buffer,
                                                                                    T.reinterpret(
                                                                                        "uint32",
                                                                                        next_c3,
                                                                                    ),
                                                                                    [0],
                                                                                )
                                                                                T.buffer_store(
                                                                                    k0.buffer,
                                                                                    T.reinterpret(
                                                                                        "uint32",
                                                                                        next_k0,
                                                                                    ),
                                                                                    [0],
                                                                                )
                                                                                T.buffer_store(
                                                                                    k1.buffer,
                                                                                    T.reinterpret(
                                                                                        "uint32",
                                                                                        next_k1,
                                                                                    ),
                                                                                    [0],
                                                                                )
                                                                            T.buffer_store(
                                                                                random_words,
                                                                                c0,
                                                                                [0],
                                                                            )
                                                                            T.buffer_store(
                                                                                random_words,
                                                                                c1,
                                                                                [1],
                                                                            )
                                                                            T.buffer_store(
                                                                                random_words,
                                                                                c2,
                                                                                [2],
                                                                            )
                                                                            T.buffer_store(
                                                                                random_words,
                                                                                c3,
                                                                                [3],
                                                                            )
                                                                    packed_f16 = _builder_name(
                                                                        "packed_f16",
                                                                        T.alloc_local(
                                                                            (1,), "uint32"
                                                                        ),
                                                                    )
                                                                    _builder_emit(
                                                                        T.evaluate(
                                                                            T.ptx.cvt.rs.f16x2.f32(
                                                                                packed_f16[0],
                                                                                T.cuda.float2_y(
                                                                                    state_pair
                                                                                ),
                                                                                T.cuda.float2_x(
                                                                                    state_pair
                                                                                ),
                                                                                random_words[
                                                                                    pair_idx % 2
                                                                                ],
                                                                            )
                                                                        )
                                                                    )
                                                                    T.buffer_store(
                                                                        store_words,
                                                                        packed_f16[0],
                                                                        [pair_idx],
                                                                    )
                                                                elif STATE_DTYPE == "bfloat16":
                                                                    low_bits = _builder_scalar(
                                                                        "low_bits",
                                                                        _f32_to_bf16(
                                                                            T.cuda.float2_x(
                                                                                state_pair
                                                                            )
                                                                        ),
                                                                        dtype="uint16",
                                                                    )
                                                                    high_bits = _builder_scalar(
                                                                        "high_bits",
                                                                        _f32_to_bf16(
                                                                            T.cuda.float2_y(
                                                                                state_pair
                                                                            )
                                                                        ),
                                                                        dtype="uint16",
                                                                    )
                                                                    _builder_emit(
                                                                        T.evaluate(
                                                                            T.ptx.mov.b32(
                                                                                store_words[
                                                                                    pair_idx
                                                                                ],
                                                                                low_bits,
                                                                                high_bits,
                                                                            )
                                                                        )
                                                                    )
                                                                elif STATE_DTYPE == "float16":
                                                                    low_bits = _builder_scalar(
                                                                        "low_bits",
                                                                        _f32_to_f16(
                                                                            T.cuda.float2_x(
                                                                                state_pair
                                                                            )
                                                                        ),
                                                                        dtype="uint16",
                                                                    )
                                                                    high_bits = _builder_scalar(
                                                                        "high_bits",
                                                                        _f32_to_f16(
                                                                            T.cuda.float2_y(
                                                                                state_pair
                                                                            )
                                                                        ),
                                                                        dtype="uint16",
                                                                    )
                                                                    _builder_emit(
                                                                        T.evaluate(
                                                                            T.ptx.mov.b32(
                                                                                store_words[
                                                                                    pair_idx
                                                                                ],
                                                                                low_bits,
                                                                                high_bits,
                                                                            )
                                                                        )
                                                                    )
                                                                else:
                                                                    T.buffer_store(
                                                                        store_words,
                                                                        T.reinterpret(
                                                                            "uint32",
                                                                            T.cuda.float2_x(
                                                                                state_pair
                                                                            ),
                                                                        ),
                                                                        [pair_idx * 2],
                                                                    )
                                                                    T.buffer_store(
                                                                        store_words,
                                                                        T.reinterpret(
                                                                            "uint32",
                                                                            T.cuda.float2_y(
                                                                                state_pair
                                                                            ),
                                                                        ),
                                                                        [pair_idx * 2 + 1],
                                                                    )
                                                            if HAS_INTERMEDIATE_STATES:
                                                                _builder_emit(
                                                                    T.evaluate(
                                                                        T.ptx.st.global_.v4.b32(
                                                                            intermediate_states.ptr_to(
                                                                                [
                                                                                    dst_base
                                                                                    + member_col
                                                                                ]
                                                                            ),
                                                                            store_words[0],
                                                                            store_words[1],
                                                                            store_words[2],
                                                                            store_words[3],
                                                                        )
                                                                    )
                                                                )
                                                            else:
                                                                _builder_emit(
                                                                    T.evaluate(
                                                                        T.ptx.st.global_.v4.b32(
                                                                            state.ptr_to(
                                                                                [
                                                                                    dst_base
                                                                                    + member_col
                                                                                ]
                                                                            ),
                                                                            store_words[0],
                                                                            store_words[1],
                                                                            store_words[2],
                                                                            store_words[3],
                                                                        )
                                                                    )
                                                                )
                                                if SCALE_STATE:
                                                    with T.If(T.And(T.bool(True), member == 0)):
                                                        with T.Then():
                                                            new_decode_scale = _builder_scalar(
                                                                "new_decode_scale",
                                                                _rcp(encode_scale),
                                                                dtype="float32",
                                                            )
                                                            if HAS_INTERMEDIATE_STATES:
                                                                scale_offset = _builder_scalar(
                                                                    "scale_offset",
                                                                    dst_slot
                                                                    * T.cast(NHEADS * DIM, "int64")
                                                                    + head * DIM
                                                                    + row_d,
                                                                    dtype="int64",
                                                                )
                                                                _builder_emit(
                                                                    T.evaluate(
                                                                        T.ptx.st.global_.b32(
                                                                            intermediate_scales.ptr_to(
                                                                                [scale_offset]
                                                                            ),
                                                                            T.reinterpret(
                                                                                "uint32",
                                                                                new_decode_scale,
                                                                            ),
                                                                        )
                                                                    )
                                                                )
                                                            else:
                                                                scale_offset = _builder_scalar(
                                                                    "scale_offset",
                                                                    dst_slot
                                                                    * state_scale_stride_batch
                                                                    + head * DIM
                                                                    + row_d,
                                                                    dtype="int64",
                                                                )
                                                                _builder_emit(
                                                                    T.evaluate(
                                                                        T.ptx.st.global_.b32(
                                                                            state_scale.ptr_to(
                                                                                [scale_offset]
                                                                            ),
                                                                            T.reinterpret(
                                                                                "uint32",
                                                                                new_decode_scale,
                                                                            ),
                                                                        )
                                                                    )
                                                                )
                            if NUM_PASSES > 1:
                                with T.If(T.And(T.bool(True), pass_idx < NUM_PASSES - 1)):
                                    with T.Then():
                                        next_stage = _builder_scalar(
                                            "next_stage",
                                            (pass_idx + 1) % STATE_STAGES,
                                            dtype="int32",
                                        )
                                        next_dim_base = _builder_scalar(
                                            "next_dim_base",
                                            dim_offset + (pass_idx + 1) * 16,
                                            dtype="int32",
                                        )
                                        if not IS_PAD:
                                            with T.serial(
                                                (16 * DSTATE // (16 // STATE_BYTES) + 127) // 128
                                            ) as state_load_iter:
                                                packed_i = _builder_scalar(
                                                    "packed_i",
                                                    flat_tid + state_load_iter * 128,
                                                    dtype="int32",
                                                )
                                                with T.If(
                                                    packed_i < 16 * DSTATE // (16 // STATE_BYTES)
                                                ):
                                                    with T.Then():
                                                        state_row = _builder_scalar(
                                                            "state_row",
                                                            packed_i
                                                            // (DSTATE // (16 // STATE_BYTES)),
                                                            dtype="int32",
                                                        )
                                                        state_col = _builder_scalar(
                                                            "state_col",
                                                            packed_i
                                                            % (DSTATE // (16 // STATE_BYTES))
                                                            * (16 // STATE_BYTES),
                                                            dtype="int32",
                                                        )
                                                        _builder_emit(
                                                            T.ptx["cp.async.cg.shared.global"](
                                                                s_state.ptr_to(
                                                                    [
                                                                        (
                                                                            next_stage * 16
                                                                            + state_row
                                                                        )
                                                                        * DSTATE_PAD
                                                                        + state_col
                                                                    ]
                                                                ),
                                                                state.ptr_to(
                                                                    [
                                                                        state_head_offset
                                                                        + (
                                                                            next_dim_base
                                                                            + state_row
                                                                        )
                                                                        * DSTATE
                                                                        + state_col
                                                                    ]
                                                                ),
                                                                16,
                                                            )
                                                        )
                                        _builder_emit(T.ptx.cp.async_.commit_group())
                                        _builder_emit(T.ptx.cp.async_.wait_group(0))
                                        _builder_emit(T.ptx.bar.sync(T.uint32(0)))
                        _builder_emit(T.ptx.bar.sync(T.uint32(0)))
                        with T.serial((NTOKENS + 3) // 4) as output_iter:
                            step = _builder_scalar("step", warp + output_iter * 4, dtype="int32")
                            with T.If(step < seq_len):
                                with T.Then():
                                    if HAS_CU_SEQLENS:
                                        out_base = _builder_scalar(
                                            "out_base",
                                            T.cast(bos + step, "int64") * out_stride_batch
                                            + head * DIM
                                            + dim_offset,
                                            dtype="int64",
                                        )
                                        z_base = _builder_scalar(
                                            "z_base",
                                            T.cast(bos + step, "int64") * z_stride_batch
                                            + head * DIM
                                            + dim_offset,
                                            dtype="int64",
                                        )
                                    else:
                                        out_base = _builder_scalar(
                                            "out_base",
                                            T.cast(seq_idx, "int64") * out_stride_batch
                                            + T.cast(step, "int64") * out_stride_mtp
                                            + head * DIM
                                            + dim_offset,
                                            dtype="int64",
                                        )
                                        z_base = _builder_scalar(
                                            "z_base",
                                            T.cast(seq_idx, "int64") * z_stride_batch
                                            + T.cast(step, "int64") * z_stride_mtp
                                            + head * DIM
                                            + dim_offset,
                                            dtype="int64",
                                        )
                                    if DIM_PER_CTA >= 32:
                                        output_count = _builder_scalar(
                                            "output_count", DIM_PER_CTA // 32, dtype="int32"
                                        )
                                        local_col = _builder_scalar(
                                            "local_col", lane * output_count, dtype="int32"
                                        )
                                        out_words = _builder_name(
                                            "out_words", T.alloc_local((4,), "uint32")
                                        )
                                        if DIM_PER_CTA == 32:
                                            T.buffer_store(
                                                out_words,
                                                _shared_load_u32(
                                                    s_out, step * DIM_PER_CTA + local_col
                                                ),
                                                [0],
                                            )
                                        elif DIM_PER_CTA == 64:
                                            if OFF_OUT % 8 == 0:
                                                _builder_emit(
                                                    T.evaluate(
                                                        T.ptx.ld.shared.v2.b32(
                                                            out_words[0],
                                                            out_words[1],
                                                            s_out.ptr_to(
                                                                [step * DIM_PER_CTA + local_col]
                                                            ),
                                                        )
                                                    )
                                                )
                                            else:
                                                T.buffer_store(
                                                    out_words,
                                                    _shared_load_u32(
                                                        s_out, step * DIM_PER_CTA + local_col
                                                    ),
                                                    [0],
                                                )
                                                T.buffer_store(
                                                    out_words,
                                                    _shared_load_u32(
                                                        s_out, step * DIM_PER_CTA + local_col + 1
                                                    ),
                                                    [1],
                                                )
                                        else:
                                            _builder_emit(
                                                T.evaluate(
                                                    T.ptx.ld.shared.v4.b32(
                                                        out_words[0],
                                                        out_words[1],
                                                        out_words[2],
                                                        out_words[3],
                                                        s_out.ptr_to(
                                                            [step * DIM_PER_CTA + local_col]
                                                        ),
                                                    )
                                                )
                                            )
                                        z_bits = _builder_name(
                                            "z_bits", T.alloc_local((4,), "uint16")
                                        )
                                        if HAS_Z:
                                            if DIM_PER_CTA == 32:
                                                T.buffer_store(
                                                    z_bits,
                                                    _global_load_u16(z, z_base + local_col),
                                                    [0],
                                                )
                                            elif DIM_PER_CTA == 64:
                                                _builder_emit(
                                                    T.evaluate(
                                                        T.ptx.ld.global_.v2.b16(
                                                            z_bits[0],
                                                            z_bits[1],
                                                            z.ptr_to([z_base + local_col]),
                                                        )
                                                    )
                                                )
                                            else:
                                                _builder_emit(
                                                    T.evaluate(
                                                        T.ptx.ld.global_.v4.b16(
                                                            z_bits[0],
                                                            z_bits[1],
                                                            z_bits[2],
                                                            z_bits[3],
                                                            z.ptr_to([z_base + local_col]),
                                                        )
                                                    )
                                                )
                                        output_bits = _builder_name(
                                            "output_bits", T.alloc_local((4,), "uint16")
                                        )
                                        with T.unroll(DIM_PER_CTA // 32) as element:
                                            value = _builder_scalar(
                                                "value",
                                                T.reinterpret("float32", out_words[element]),
                                                dtype="float32",
                                            )
                                            if HAS_Z:
                                                z_value = _builder_scalar(
                                                    "z_value",
                                                    _bf16_to_f32(z_bits[element]),
                                                    dtype="float32",
                                                )
                                                exp_neg_z = _builder_scalar(
                                                    "exp_neg_z",
                                                    _exp2(
                                                        _mul(
                                                            _sub(T.float32(0.0), z_value),
                                                            T.float32(_LOG2_E),
                                                        )
                                                    ),
                                                    dtype="float32",
                                                )
                                                sigmoid_z = _builder_scalar(
                                                    "sigmoid_z",
                                                    _div(
                                                        T.float32(1.0),
                                                        _add(T.float32(1.0), exp_neg_z),
                                                    ),
                                                    dtype="float32",
                                                )
                                                T.buffer_store(
                                                    value.buffer,
                                                    _mul(value, _mul(z_value, sigmoid_z)),
                                                    [0],
                                                )
                                            T.buffer_store(
                                                output_bits, _f32_to_bf16(value), [element]
                                            )
                                        if DIM_PER_CTA == 32:
                                            _builder_emit(
                                                T.evaluate(
                                                    T.ptx.st.global_.b16(
                                                        output.ptr_to([out_base + local_col]),
                                                        output_bits[0],
                                                    )
                                                )
                                            )
                                        elif DIM_PER_CTA == 64:
                                            _builder_emit(
                                                T.evaluate(
                                                    T.ptx.st.global_.v2.b16(
                                                        output.ptr_to([out_base + local_col]),
                                                        output_bits[0],
                                                        output_bits[1],
                                                    )
                                                )
                                            )
                                        else:
                                            _builder_emit(
                                                T.evaluate(
                                                    T.ptx.st.global_.v4.b16(
                                                        output.ptr_to([out_base + local_col]),
                                                        output_bits[0],
                                                        output_bits[1],
                                                        output_bits[2],
                                                        output_bits[3],
                                                    )
                                                )
                                            )
                                    else:
                                        with T.If(lane < DIM_PER_CTA):
                                            with T.Then():
                                                value = _builder_scalar(
                                                    "value",
                                                    T.reinterpret(
                                                        "float32",
                                                        _shared_load_u32(
                                                            s_out, step * DIM_PER_CTA + lane
                                                        ),
                                                    ),
                                                    dtype="float32",
                                                )
                                                if HAS_Z:
                                                    z_value = _builder_scalar(
                                                        "z_value",
                                                        _bf16_to_f32(
                                                            _global_load_u16(z, z_base + lane)
                                                        ),
                                                        dtype="float32",
                                                    )
                                                    exp_neg_z = _builder_scalar(
                                                        "exp_neg_z",
                                                        _exp2(
                                                            _mul(
                                                                _sub(T.float32(0.0), z_value),
                                                                T.float32(_LOG2_E),
                                                            )
                                                        ),
                                                        dtype="float32",
                                                    )
                                                    sigmoid_z = _builder_scalar(
                                                        "sigmoid_z",
                                                        _div(
                                                            T.float32(1.0),
                                                            _add(T.float32(1.0), exp_neg_z),
                                                        ),
                                                        dtype="float32",
                                                    )
                                                    T.buffer_store(
                                                        value.buffer,
                                                        _mul(value, _mul(z_value, sigmoid_z)),
                                                        [0],
                                                    )
                                                output_bit = _builder_scalar(
                                                    "output_bit",
                                                    _f32_to_bf16(value),
                                                    dtype="uint16",
                                                )
                                                _builder_emit(
                                                    T.evaluate(
                                                        T.ptx.st.global_.b16(
                                                            output.ptr_to([out_base + lane]),
                                                            output_bit,
                                                        )
                                                    )
                                                )

                    with T.If(is_pad != 0):
                        with T.Then():
                            _builder_emit(run_simple(True))
                        with T.Else():
                            _builder_emit(run_simple(False))
    return builder.get()


def _num_sms(device: str | torch.device = "cuda") -> int:
    del device
    from tirx_kernels.runner import hardware_num_sms

    return hardware_num_sms()


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
    return _build_selective_state_update_mtp_simple(**_specialization(kwargs))


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
def _load_oracle():
    from flashinfer.mamba import selective_state_update

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
    oracle = _load_oracle()
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


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(**kwargs: Any) -> None:
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    executable = compile_kernel(get_kernel(**kwargs))
    executable(*_tirx_args(case))
    torch.cuda.synchronize()
    _run_reference(case)
    torch.cuda.synchronize()
    _assert_case_close(case)


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
    config = dict(prepared["config"])
    config.update(kwargs)
    kwargs = config
    executable = prepared["executable"]
    from tirx_kernels.runner import bench

    case = prepare_data(**kwargs)
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


def run_bench(
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 1,
    cooldown_s: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    return prepare_bench(**kwargs).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
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
