# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Integration scaffold for FlashInfer's SM100 FP32-state MTP warp kernel.

Upstream source: flashinfer/gdn_kernels/gdn_decode_mtp.py.
"""

from __future__ import annotations

import functools
import os
from typing import Any
from unittest import SkipTest

import torch

from tvm.script import tirx as T

KERNEL_META = {
    "name": "gdn_decode_fp32_mtp_warp",
    "category": "flashinfer",
    "compute_capability": 10,
}

K = 128
V = 128
THREADS = 128
VEC_SIZE = 4
SCALE = K**-0.5
NUM_WARPS = 4
NUM_GROUPS = 4
LANES_PER_GROUP = 32
SOURCE_NUM_SMS = 148
LOG2_E = 1.4426950408889634
LN_2 = 0.6931471805599453
_HAS_NATIVE_PTX_ADDR = hasattr(T.ptx, "addr")


def _ptx_unary(chain: str, value):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx[chain](out[0], value))
    return out[0]


def _ptx_binary(chain: str, lhs, rhs, dtype: str = "float32"):
    out = T.alloc_local((1,), dtype)
    T.evaluate(T.ptx[chain](out[0], lhs, rhs))
    return out[0]


def _ptx_ternary(chain: str, lhs, rhs, acc):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx[chain](out[0], lhs, rhs, acc))
    return out[0]


def _add_f32(lhs, rhs):
    return _ptx_binary("add.f32", lhs, rhs)


def _sub_f32(lhs, rhs):
    return _ptx_binary("sub.f32", lhs, rhs)


def _mul_f32(lhs, rhs):
    return _ptx_binary("mul.f32", lhs, rhs)


def _fma_f32(lhs, rhs, acc):
    return _ptx_ternary("fma.rn.f32", lhs, rhs, acc)


def _exp2_f32(value):
    return _ptx_unary("ex2.approx.ftz.f32", value)


def _log2_f32(value):
    return _ptx_unary("lg2.approx.ftz.f32", value)


def _rcp_f32(value):
    return _ptx_unary("rcp.rn.f32", value)


def _rsqrt_f32(value):
    return _ptx_unary("rsqrt.approx.ftz.f32", value)


def _global_load_u16(buffer, index):
    return _global_load_u16_ptr(buffer.ptr_to([index]))


def _global_load_u16_ptr(ptr):
    out = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.ld.global_.b16(out[0], ptr))
    return out[0]


def _global_load_u16_ptr_offset(ptr, byte_offset: int, native_offset: bool = True):
    if not native_offset:
        return _global_load_u16_ptr(T.ptr_byte_offset(ptr, byte_offset, "bfloat16"))
    out = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.ld.global_.b16(out[0], T.ptx.addr(ptr, byte_offset)))
    return out[0]


def _global_load_s32(buffer, index):
    out = T.alloc_local((1,), "int32")
    T.evaluate(T.ptx.ld.global_.s32(out[0], buffer.ptr_to([index])))
    return out[0]


def _global_load_f32(buffer, index):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.ld.global_.b32(out[0], buffer.ptr_to([index])))
    return out[0]


def _bf16_to_f32(bits):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.cvt.f32.bf16(out[0], T.cast(bits, "uint16")))
    return out[0]


def _f32_to_bf16(value):
    out = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.cvt.rn.bf16.f32(out[0], value))
    return out[0]


@T.inline
def _add_f32_store(out, index: T.int32, lhs, rhs):
    T.evaluate(T.ptx.add.f32(out[index], lhs, rhs))


@T.inline
def _mul_f32_store(out, index: T.int32, lhs, rhs):
    T.evaluate(T.ptx.mul.f32(out[index], lhs, rhs))


@T.inline
def _mixed_sub_bf16_f32_store(out, index: T.int32, bits, value):
    T.evaluate(T.ptx.sub.rn.f32.bf16(out[index], T.cast(bits, "uint16"), value))


@T.inline
def _sub_f32_store(out, index: T.int32, lhs, rhs):
    T.evaluate(T.ptx.sub.f32(out[index], lhs, rhs))


@T.inline
def _f32_to_bf16_store(out, index: T.int32, value):
    T.evaluate(T.ptx.cvt.rn.bf16.f32(out[index], value))


def _mixed_add_bf16_f32(bits, value):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.add.rn.f32.bf16(out[0], T.cast(bits, "uint16"), value))
    return out[0]


def _mixed_sub_bf16_f32(bits, value):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.sub.rn.f32.bf16(out[0], T.cast(bits, "uint16"), value))
    return out[0]


def _bf16_square_fma(bits, acc):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.fma.rn.f32.bf16(out[0], T.cast(bits, "uint16"), T.cast(bits, "uint16"), acc))
    return out[0]


def _shared_store_f32(buffer, index, value):
    _shared_store_f32_ptr(buffer.ptr_to([index]), value)


def _shared_store_f32_ptr(ptr, value):
    T.evaluate(T.ptx.st.shared.b32(ptr, T.reinterpret("uint32", value)))


def _shared_store_f32_ptr_offset(ptr, byte_offset: int, value, native_offset: bool = True):
    if not native_offset:
        _shared_store_f32_ptr(T.ptr_byte_offset(ptr, byte_offset, "float32"), value)
        return
    T.evaluate(T.ptx.st.shared.b32(T.ptx.addr(ptr, byte_offset), T.reinterpret("uint32", value)))


def _shared_load_f32(buffer, index):
    return _shared_load_f32_ptr(buffer.ptr_to([index]))


def _shared_load_f32_ptr(ptr):
    word = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.ld.shared.b32(word[0], ptr))
    return T.reinterpret("float32", word[0])


def _shared_load_f32_ptr_offset(ptr, byte_offset: int, native_offset: bool = True):
    if not native_offset:
        return _shared_load_f32_ptr(T.ptr_byte_offset(ptr, byte_offset, "float32"))
    word = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.ld.shared.b32(word[0], T.ptx.addr(ptr, byte_offset)))
    return T.reinterpret("float32", word[0])


def _shared_store_u16(buffer, index, value):
    _shared_store_u16_ptr(buffer.ptr_to([index]), value)


def _shared_store_u16_ptr(ptr, value):
    T.evaluate(T.ptx.st.shared.b16(ptr, value))


def _shared_store_u16_ptr_offset(ptr, byte_offset: int, value, native_offset: bool = True):
    if not native_offset:
        _shared_store_u16_ptr(T.ptr_byte_offset(ptr, byte_offset, "bfloat16"), value)
        return
    T.evaluate(T.ptx.st.shared.b16(T.ptx.addr(ptr, byte_offset), value))


def _shared_load_u16(buffer, index):
    return _shared_load_u16_ptr(buffer.ptr_to([index]))


def _shared_load_u16_ptr(ptr):
    out = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.ld.shared.b16(out[0], ptr))
    return out[0]


def _shared_load_u16_ptr_offset(ptr, byte_offset: int, native_offset: bool = True):
    if not native_offset:
        return _shared_load_u16_ptr(T.ptr_byte_offset(ptr, byte_offset, "bfloat16"))
    out = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.ld.shared.b16(out[0], T.ptx.addr(ptr, byte_offset)))
    return out[0]


@T.inline
def _shared_load_f32x4(buffer, index, values):
    words = T.alloc_local((4,), "uint32", align=16)
    T.evaluate(
        T.ptx.ld.shared.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
    )
    for i in T.unroll(4):
        values[i] = T.reinterpret("float32", words[i])


@T.inline
def _shared_load_f32x4_ptr(ptr, values):
    words = T.alloc_local((4,), "uint32", align=16)
    T.evaluate(T.ptx.ld.shared.v4.b32(words[0], words[1], words[2], words[3], ptr))
    for i in T.unroll(4):
        values[i] = T.reinterpret("float32", words[i])


@T.inline
def _shared_load_f32x4_ptr_offset(ptr, byte_offset: T.int32, values, native_offset: bool = True):
    if not native_offset:
        _shared_load_f32x4_ptr(T.ptr_byte_offset(ptr, byte_offset, "float32"), values)
    else:
        words = T.alloc_local((4,), "uint32", align=16)
        T.evaluate(
            T.ptx.ld.shared.v4.b32(
                words[0], words[1], words[2], words[3], T.ptx.addr(ptr, byte_offset)
            )
        )
        for i in T.unroll(4):
            values[i] = T.reinterpret("float32", words[i])


@T.inline
def _shared_load_f32x4_b64(buffer, index, values):
    pairs = T.alloc_local((2,), "uint64", align=16)
    T.evaluate(T.ptx.ld.shared.v2.b64(pairs[0], pairs[1], buffer.ptr_to([index])))
    values[0] = T.cuda.float2_x(pairs[0])
    values[1] = T.cuda.float2_y(pairs[0])
    values[2] = T.cuda.float2_x(pairs[1])
    values[3] = T.cuda.float2_y(pairs[1])


@T.inline
def _shared_load_f32x4_b64_ptr(ptr, values):
    pairs = T.alloc_local((2,), "uint64", align=16)
    T.evaluate(T.ptx.ld.shared.v2.b64(pairs[0], pairs[1], ptr))
    values[0] = T.cuda.float2_x(pairs[0])
    values[1] = T.cuda.float2_y(pairs[0])
    values[2] = T.cuda.float2_x(pairs[1])
    values[3] = T.cuda.float2_y(pairs[1])


@T.inline
def _shared_load_f32x4_b64_ptr_offset(
    ptr, byte_offset: T.int32, values, native_offset: bool = True
):
    if not native_offset:
        _shared_load_f32x4_b64_ptr(T.ptr_byte_offset(ptr, byte_offset, "float32"), values)
    else:
        pairs = T.alloc_local((2,), "uint64", align=16)
        T.evaluate(T.ptx.ld.shared.v2.b64(pairs[0], pairs[1], T.ptx.addr(ptr, byte_offset)))
        values[0] = T.cuda.float2_x(pairs[0])
        values[1] = T.cuda.float2_y(pairs[0])
        values[2] = T.cuda.float2_x(pairs[1])
        values[3] = T.cuda.float2_y(pairs[1])


@T.inline
def _global_load_f32x4(buffer, index, values, value_offset: T.int32):
    _global_load_f32x4_ptr(buffer.ptr_to([index]), values, value_offset)


@T.inline
def _global_load_f32x4_ptr(ptr, values, value_offset: T.int32):
    words = T.alloc_local((4,), "uint32", align=16)
    T.evaluate(T.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], ptr))
    for i in T.unroll(4):
        values[value_offset + i] = T.reinterpret("float32", words[i])


@T.inline
def _global_load_f32x4_ptr_offset(
    ptr, byte_offset: T.int32, values, value_offset: T.int32, native_offset: bool = True
):
    if not native_offset:
        _global_load_f32x4_ptr(T.ptr_byte_offset(ptr, byte_offset, "float32"), values, value_offset)
    else:
        words = T.alloc_local((4,), "uint32", align=16)
        T.evaluate(
            T.ptx.ld.global_.v4.b32(
                words[0], words[1], words[2], words[3], T.ptx.addr(ptr, byte_offset)
            )
        )
        for i in T.unroll(4):
            values[value_offset + i] = T.reinterpret("float32", words[i])


@T.inline
def _global_load_f32x4_b64(buffer, index, values, value_offset: T.int32):
    _global_load_f32x4_b64_ptr(buffer.ptr_to([index]), values, value_offset)


@T.inline
def _global_load_f32x4_b64_ptr(ptr, values, value_offset: T.int32):
    pairs = T.alloc_local((2,), "uint64", align=16)
    T.evaluate(T.ptx.ld.global_.v2.b64(pairs[0], pairs[1], ptr))
    values[value_offset] = T.cuda.float2_x(pairs[0])
    values[value_offset + 1] = T.cuda.float2_y(pairs[0])
    values[value_offset + 2] = T.cuda.float2_x(pairs[1])
    values[value_offset + 3] = T.cuda.float2_y(pairs[1])


@T.inline
def _global_load_f32x4_b64_ptr_offset(
    ptr, byte_offset: T.int32, values, value_offset: T.int32, native_offset: bool = True
):
    if not native_offset:
        _global_load_f32x4_b64_ptr(
            T.ptr_byte_offset(ptr, byte_offset, "float32"), values, value_offset
        )
    else:
        pairs = T.alloc_local((2,), "uint64", align=16)
        T.evaluate(T.ptx.ld.global_.v2.b64(pairs[0], pairs[1], T.ptx.addr(ptr, byte_offset)))
        values[value_offset] = T.cuda.float2_x(pairs[0])
        values[value_offset + 1] = T.cuda.float2_y(pairs[0])
        values[value_offset + 2] = T.cuda.float2_x(pairs[1])
        values[value_offset + 3] = T.cuda.float2_y(pairs[1])


@T.inline
def _global_store_f32x4(buffer, index, values, value_offset: T.int32):
    _global_store_f32x4_ptr(buffer.ptr_to([index]), values, value_offset)


@T.inline
def _global_store_f32x4_ptr(ptr, values, value_offset: T.int32):
    T.evaluate(
        T.ptx.st.global_.v4.b32(
            ptr,
            T.reinterpret("uint32", values[value_offset]),
            T.reinterpret("uint32", values[value_offset + 1]),
            T.reinterpret("uint32", values[value_offset + 2]),
            T.reinterpret("uint32", values[value_offset + 3]),
        )
    )


@T.inline
def _global_store_f32x4_ptr_offset(
    ptr, byte_offset: T.int32, values, value_offset: T.int32, native_offset: bool = True
):
    if not native_offset:
        _global_store_f32x4_ptr(
            T.ptr_byte_offset(ptr, byte_offset, "float32"), values, value_offset
        )
    else:
        T.evaluate(
            T.ptx.st.global_.v4.b32(
                T.ptx.addr(ptr, byte_offset),
                T.reinterpret("uint32", values[value_offset]),
                T.reinterpret("uint32", values[value_offset + 1]),
                T.reinterpret("uint32", values[value_offset + 2]),
                T.reinterpret("uint32", values[value_offset + 3]),
            )
        )


@T.inline
def _global_store_f32x4_b64(buffer, index, values, value_offset: T.int32):
    _global_store_f32x4_b64_ptr(buffer.ptr_to([index]), values, value_offset)


@T.inline
def _global_store_f32x4_b64_ptr(ptr, values, value_offset: T.int32):
    T.evaluate(
        T.ptx.st.global_.v2.b64(
            ptr,
            T.cuda.make_float2(values[value_offset], values[value_offset + 1]),
            T.cuda.make_float2(values[value_offset + 2], values[value_offset + 3]),
        )
    )


@T.inline
def _global_store_f32x4_b64_ptr_offset(
    ptr, byte_offset: T.int32, values, value_offset: T.int32, native_offset: bool = True
):
    if not native_offset:
        _global_store_f32x4_b64_ptr(
            T.ptr_byte_offset(ptr, byte_offset, "float32"), values, value_offset
        )
    else:
        T.evaluate(
            T.ptx.st.global_.v2.b64(
                T.ptx.addr(ptr, byte_offset),
                T.cuda.make_float2(values[value_offset], values[value_offset + 1]),
                T.cuda.make_float2(values[value_offset + 2], values[value_offset + 3]),
            )
        )


def _global_store_u16(buffer, index, value):
    _global_store_u16_ptr(buffer.ptr_to([index]), value)


def _global_store_u16_ptr(ptr, value):
    T.evaluate(T.ptx.st.global_.b16(ptr, value))


def _global_store_u16_ptr_offset(ptr, byte_offset: int, value, native_offset: bool = True):
    if not native_offset:
        _global_store_u16_ptr(T.ptr_byte_offset(ptr, byte_offset, "bfloat16"), value)
        return
    T.evaluate(T.ptx.st.global_.b16(T.ptx.addr(ptr, byte_offset), value))


def _global_store_u32_ptr_offset(ptr, byte_offset: int, value, native_offset: bool = True):
    if not native_offset:
        T.evaluate(T.ptx.st.global_.b32(T.ptr_byte_offset(ptr, byte_offset, "bfloat16"), value))
        return
    T.evaluate(T.ptx.st.global_.b32(T.ptx.addr(ptr, byte_offset), value))


def _shared_store_u32_ptr_offset(ptr, byte_offset: int, value, native_offset: bool = True):
    if not native_offset:
        T.evaluate(T.ptx.st.shared.b32(T.ptr_byte_offset(ptr, byte_offset, "bfloat16"), value))
        return
    T.evaluate(T.ptx.st.shared.b32(T.ptx.addr(ptr, byte_offset), value))


def _packed_fma(lhs0, lhs1, rhs0, rhs1, acc0, acc1):
    out = T.alloc_local((1,), "uint64")
    T.evaluate(
        T.ptx.fma.rn.f32x2(
            out[0],
            T.cuda.make_float2(lhs0, lhs1),
            T.cuda.make_float2(rhs0, rhs1),
            T.cuda.make_float2(acc0, acc1),
        )
    )
    return out[0]


@T.inline
def _packed_fma_store(out, lhs0, lhs1, rhs0, rhs1, acc0, acc1):
    T.evaluate(
        T.ptx.fma.rn.f32x2(
            out[0],
            T.cuda.make_float2(lhs0, lhs1),
            T.cuda.make_float2(rhs0, rhs1),
            T.cuda.make_float2(acc0, acc1),
        )
    )


def _gate_pair(a_bits, b_bits, exp_A_value, dt_value):
    b_value: T.float32 = _bf16_to_f32(b_bits)
    x_value: T.float32 = _mixed_add_bf16_f32(a_bits, dt_value)
    softplus_exp: T.float32 = _exp2_f32(_mul_f32(x_value, T.float32(LOG2_E)))
    softplus_value: T.float32 = _mul_f32(
        _log2_f32(_add_f32(T.float32(1.0), softplus_exp)), T.float32(LN_2)
    )
    use_softplus: T.float32 = T.if_then_else(
        x_value <= T.float32(20.0), T.float32(1.0), T.float32(0.0)
    )
    softplus_x: T.float32 = _fma_f32(
        softplus_value, use_softplus, _mul_f32(x_value, _sub_f32(T.float32(1.0), use_softplus))
    )
    decay_positive: T.float32 = _mul_f32(exp_A_value, softplus_x)
    beta: T.float32 = _rcp_f32(
        _add_f32(T.float32(1.0), _exp2_f32(_mul_f32(b_value, T.float32(-LOG2_E))))
    )
    g_value: T.float32 = _exp2_f32(_mul_f32(decay_positive, T.float32(-LOG2_E)))
    return T.cuda.make_float2(g_value, beta)


@T.inline
def _gate_pair_store(out, scratch, a_bits, b_bits, exp_A_value, dt_value):
    T.evaluate(T.ptx.cvt.f32.bf16(scratch[0], T.cast(b_bits, "uint16")))
    T.evaluate(T.ptx.add.rn.f32.bf16(scratch[1], T.cast(a_bits, "uint16"), dt_value))
    T.evaluate(T.ptx.mul.f32(scratch[2], scratch[1], T.float32(LOG2_E)))
    T.evaluate(T.ptx.ex2.approx.ftz.f32(scratch[2], scratch[2]))
    T.evaluate(T.ptx.add.f32(scratch[2], T.float32(1.0), scratch[2]))
    T.evaluate(T.ptx.lg2.approx.ftz.f32(scratch[2], scratch[2]))
    T.evaluate(T.ptx.mul.f32(scratch[2], scratch[2], T.float32(LN_2)))
    scratch[3] = T.if_then_else(scratch[1] <= T.float32(20.0), T.float32(1.0), T.float32(0.0))
    T.evaluate(T.ptx.sub.f32(scratch[4], T.float32(1.0), scratch[3]))
    T.evaluate(T.ptx.mul.f32(scratch[4], scratch[1], scratch[4]))
    T.evaluate(T.ptx.fma.rn.f32(scratch[2], scratch[2], scratch[3], scratch[4]))
    T.evaluate(T.ptx.mul.f32(scratch[4], exp_A_value, scratch[2]))
    T.evaluate(T.ptx.mul.f32(scratch[0], scratch[0], T.float32(-LOG2_E)))
    T.evaluate(T.ptx.ex2.approx.ftz.f32(scratch[0], scratch[0]))
    T.evaluate(T.ptx.add.f32(scratch[0], T.float32(1.0), scratch[0]))
    T.evaluate(T.ptx.rcp.rn.f32(scratch[0], scratch[0]))
    T.evaluate(T.ptx.mul.f32(scratch[4], scratch[4], T.float32(-LOG2_E)))
    T.evaluate(T.ptx.ex2.approx.ftz.f32(scratch[4], scratch[4]))
    out[0] = T.cuda.make_float2(scratch[4], scratch[0])


def _make_warp_uniform(value):
    return T.cuda._shfl_sync(T.uint32(0xFFFFFFFF), value, 0, 32)


def _source_config(
    batch: int, seq_len: int, num_v_heads: int, *, disable_state_update: bool
) -> tuple[int, int, bool]:
    """Mirror the frozen source's reachable warp-kernel picker."""
    work_units = batch * num_v_heads
    if work_units <= 64:
        tile_v, ilp_rows, use_smem_v = 8, 2, False
    elif work_units <= 128:
        tile_v, ilp_rows, use_smem_v = 16, 4, False
    elif work_units <= 448:
        if seq_len <= 2:
            tile_v, ilp_rows, use_smem_v = 16, 2, False
        else:
            tile_v, ilp_rows, use_smem_v = 32, 4, False
    elif work_units <= 1024:
        tile_v, ilp_rows, use_smem_v = 32, 4, False
    else:
        tile_v, ilp_rows, use_smem_v = 64, 4, True
        if not disable_state_update and seq_len <= 2:
            ilp_rows = 8
            use_smem_v = False
    return min(tile_v, V), ilp_rows, use_smem_v


def _target_config(
    batch: int, seq_len: int, num_heads: int, num_v_heads: int, *, disable_state_update: bool
) -> tuple[int, int, bool]:
    tile_v, ilp_rows, use_smem_v = _source_config(
        batch, seq_len, num_v_heads, disable_state_update=disable_state_update
    )
    if 128 < batch * num_v_heads <= 448 and seq_len == 2:
        return 32, 4, False
    if num_heads == 16 and 448 < batch * num_v_heads <= 1024 and 3 <= seq_len <= 7:
        return 32, 4, True
    return tile_v, ilp_rows, use_smem_v


def _case(
    label: str,
    *,
    batch: int = 4,
    seq_len: int = 4,
    num_heads: int = 16,
    num_v_heads: int = 64,
    use_qk_l2norm: bool = True,
    disable_state_update: bool = False,
    cache_intermediate_states: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    tile_v, ilp_rows, use_smem_v = _source_config(
        batch, seq_len, num_v_heads, disable_state_update=disable_state_update
    )
    return {
        "label": label,
        "batch": batch,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "num_v_heads": num_v_heads,
        "tile_v": tile_v,
        "ilp_rows": ilp_rows,
        "use_smem_v": use_smem_v,
        "use_qk_l2norm": use_qk_l2norm,
        "disable_state_update": disable_state_update,
        "cache_intermediate_states": cache_intermediate_states,
        **kwargs,
    }


CONFIGS = [
    _case("t2_wu256_ilp2", seq_len=2),
    _case("t3_wu256_ilp4", seq_len=3),
    _case("t4_wu512", batch=8, seq_len=4),
    _case("t5_wu1024", batch=16, seq_len=5),
    _case("t6_wu256", seq_len=6),
    _case("t7_wu256", seq_len=7),
    _case("t8_wu256", seq_len=8),
    _case("t2_wu2048_ilp8", batch=32, seq_len=2),
    _case("t4_wu2048_smem_v", batch=32, seq_len=4),
    _case("t8_wu2048_smem_v", batch=32, seq_len=8),
    _case("t4_l2off", use_qk_l2norm=False),
    _case("t4_disable_update", disable_state_update=True),
    _case("t4_cache_update", cache_intermediate_states=True),
    _case("t4_cache_disable_update", disable_state_update=True, cache_intermediate_states=True),
    _case("t4_split_pool", same_pool=False),
    _case("t4_negative_read", negative_read_index=True),
    _case("t4_negative_write", same_pool=False, negative_write_index=True),
    _case("t4_padded_pool", padded_pool=True),
    _case("t4_padded_split", padded_pool=True, same_pool=False),
    _case("t4_packed_qkv", packed_qkv=True),
    _case("t4_scatter_flat", per_token_pool_scatter=True),
    _case("t4_scatter_padded", padded_pool=True, per_token_pool_scatter=True),
    _case(
        "t8_scatter_i64_stress",
        batch=128,
        seq_len=8,
        num_heads=16,
        num_v_heads=64,
        per_token_pool_scatter=True,
    ),
    _case("t4_tp2", batch=8, num_heads=8, num_v_heads=32),
    _case("t4_tp4", batch=16, num_heads=4, num_v_heads=16),
    _case("t4_tp8", batch=32, num_heads=2, num_v_heads=8),
]


def _bench_case(seq_len: int, batch: int, num_heads: int, num_v_heads: int) -> dict[str, Any]:
    tile_v, ilp_rows, use_smem_v = _source_config(
        batch, seq_len, num_v_heads, disable_state_update=False
    )
    label = (
        f"t{seq_len}_b{batch}_h{num_heads}_hv{num_v_heads}_"
        f"tv{tile_v}_ilp{ilp_rows}_sv{int(use_smem_v)}"
    )
    return _case(
        label,
        batch=batch,
        seq_len=seq_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        cache_intermediate_states=True,
    )


_TP1_BATCHES = (4, 8, 16, 32, 64, 128, 256)
_TP_BOUNDARY_WORK_UNITS = (256, 512, 1024, 2048)
_TP_BOUNDARY_SEQ_LENS = (2, 3, 4, 8)
_TP_HEAD_CONFIGS = ((8, 32), (4, 16), (2, 8))

BENCH_CONFIGS = [
    _bench_case(seq_len, batch, 16, 64) for seq_len in range(2, 9) for batch in _TP1_BATCHES
] + [
    _bench_case(seq_len, work_units // num_v_heads, num_heads, num_v_heads)
    for num_heads, num_v_heads in _TP_HEAD_CONFIGS
    for work_units in _TP_BOUNDARY_WORK_UNITS
    for seq_len in _TP_BOUNDARY_SEQ_LENS
]

assert len(BENCH_CONFIGS) == 97
assert len({config["label"] for config in BENCH_CONFIGS}) == len(BENCH_CONFIGS)


def _require_supported_config(config: dict[str, Any]) -> None:
    batch = int(config["batch"])
    seq_len = int(config["seq_len"])
    num_heads = int(config["num_heads"])
    num_v_heads = int(config["num_v_heads"])
    disable_state_update = bool(config.get("disable_state_update", False))
    if batch < 1:
        raise ValueError("batch must be positive")
    if not 2 <= seq_len <= 8:
        raise ValueError("FP32 MTP warp port requires seq_len in [2, 8]")
    if batch * num_v_heads <= 128:
        raise ValueError("FP32 MTP warp port requires batch * num_v_heads > 128")
    if (num_heads, num_v_heads) not in ((16, 64), (8, 32), (4, 16), (2, 8)):
        raise ValueError("unsupported Qwen3.5 TP head configuration")
    if num_v_heads % num_heads:
        raise ValueError("num_v_heads must be divisible by num_heads")
    expected = _source_config(
        batch, seq_len, num_v_heads, disable_state_update=disable_state_update
    )
    actual = (int(config["tile_v"]), int(config["ilp_rows"]), bool(config["use_smem_v"]))
    if actual != expected:
        raise ValueError(f"config does not match frozen source picker: {actual} != {expected}")
    scatter = bool(config.get("per_token_pool_scatter", False))
    cache = bool(config.get("cache_intermediate_states", False))
    if scatter and (cache or disable_state_update):
        raise ValueError("per-token scatter requires cache off and state update on")


@T.jit
def _gdn_decode_fp32_mtp_warp(
    state_h: T.handle,
    intermediate_h: T.handle,
    A_log_h: T.handle,
    a_h: T.handle,
    dt_bias_h: T.handle,
    q_h: T.handle,
    k_h: T.handle,
    v_h: T.handle,
    b_h: T.handle,
    output_h: T.handle,
    read_indices_h: T.handle,
    write_indices_h: T.handle,
    ssm_state_indices_h: T.handle,
    state_slot_stride: T.int64,
    state_head_stride: T.int64,
    q_batch_stride: T.int64,
    k_batch_stride: T.int64,
    v_batch_stride: T.int64,
    batch: T.int32,
    *,
    SEQ_LEN: T.constexpr,
    NUM_HEADS: T.constexpr,
    NUM_V_HEADS: T.constexpr,
    TILE_V: T.constexpr,
    NUM_V_TILES: T.constexpr,
    ILP_ROWS: T.constexpr,
    USE_SMEM_V: T.constexpr,
    USE_QK_L2NORM: T.constexpr,
    USE_NATIVE_OFFSETS: T.constexpr,
    RELOAD_K_FOR_OUTPUT: T.constexpr,
    USE_CANONICAL_WARP_ID: T.constexpr,
    USE_PACKED_OUTPUT: T.constexpr,
    DISABLE_STATE_UPDATE: T.constexpr,
    CACHE_INTERMEDIATE_STATES: T.constexpr,
    SAME_POOL: T.constexpr,
    PER_TOKEN_POOL_SCATTER: T.constexpr,
    PER_TOKEN_POOL_SCATTER_FLAT: T.constexpr,
    PADDED_POOL: T.constexpr,
    PACKED_QKV: T.constexpr,
    POOL_FACTOR: T.constexpr,
    INTERMEDIATE_BATCH_STRIDE: T.constexpr,
    INTERMEDIATE_DUMMY_ELEMENTS: T.constexpr,
    SSM_BATCH_STRIDE: T.constexpr,
    SSM_DUMMY_ELEMENTS: T.constexpr,
    SHARED_BYTES: T.constexpr,
    S_K_BYTE_OFFSET: T.constexpr,
    S_G_BYTE_OFFSET: T.constexpr,
    S_BETA_BYTE_OFFSET: T.constexpr,
    S_V_BYTE_OFFSET: T.constexpr,
    S_OUTPUT_BYTE_OFFSET: T.constexpr,
    ROWS_PER_GROUP: T.constexpr,
    ITERS_PER_GROUP: T.constexpr,
    PREFETCH_ROWS: T.constexpr,
):
    # TIRX_KERNEL_SKETCH_START
    state = T.match_buffer(
        state_h,
        (state_slot_stride * T.cast(batch * POOL_FACTOR, "int64"),),
        "float32",
        scope="global",
    )
    intermediate = T.match_buffer(
        intermediate_h,
        (T.cast(batch * INTERMEDIATE_BATCH_STRIDE + INTERMEDIATE_DUMMY_ELEMENTS, "int64"),),
        "float32",
        scope="global",
    )
    A_log = T.match_buffer(A_log_h, (NUM_V_HEADS,), "float32", scope="global")
    a = T.match_buffer(
        a_h, (T.cast(batch * SEQ_LEN * NUM_V_HEADS, "int64"),), "bfloat16", scope="global"
    )
    dt_bias = T.match_buffer(dt_bias_h, (NUM_V_HEADS,), "float32", scope="global")
    q = T.match_buffer(
        q_h,
        (q_batch_stride * T.cast(batch - 1, "int64") + T.cast(SEQ_LEN * NUM_HEADS * K, "int64"),),
        "bfloat16",
        scope="global",
    )
    k = T.match_buffer(
        k_h,
        (k_batch_stride * T.cast(batch - 1, "int64") + T.cast(SEQ_LEN * NUM_HEADS * K, "int64"),),
        "bfloat16",
        scope="global",
    )
    v = T.match_buffer(
        v_h,
        (v_batch_stride * T.cast(batch - 1, "int64") + T.cast(SEQ_LEN * NUM_V_HEADS * V, "int64"),),
        "bfloat16",
        scope="global",
    )
    b_gate = T.match_buffer(
        b_h, (T.cast(batch * SEQ_LEN * NUM_V_HEADS, "int64"),), "bfloat16", scope="global"
    )
    output = T.match_buffer(
        output_h, (T.cast(batch * SEQ_LEN * NUM_V_HEADS * V, "int64"),), "bfloat16", scope="global"
    )
    read_indices = T.match_buffer(
        read_indices_h, (T.cast(batch, "int64"),), "int32", scope="global"
    )
    write_indices = T.match_buffer(
        write_indices_h, (T.cast(batch, "int64"),), "int32", scope="global"
    )
    ssm_state_indices = T.match_buffer(
        ssm_state_indices_h,
        (T.cast(batch * SSM_BATCH_STRIDE + SSM_DUMMY_ELEMENTS, "int64"),),
        "int32",
        scope="global",
    )
    T.device_entry()
    if ILP_ROWS == 4 and SEQ_LEN == 8:
        if USE_SMEM_V and NUM_HEADS >= 8:
            T.attr({"tirx.launch_bounds_min_blocks_per_sm": 9})
        else:
            T.attr({"tirx.launch_bounds_min_blocks_per_sm": 8})
    linear_cta = T.cta_id([batch * NUM_V_HEADS * NUM_V_TILES])
    if USE_CANONICAL_WARP_ID:
        warp: T.int32 = T.warp_id([NUM_GROUPS])
        lane: T.int32 = T.lane_id([LANES_PER_GROUP])
        tid: T.int32 = warp * LANES_PER_GROUP + lane
    else:
        tid = T.thread_id([THREADS])
        warp_raw: T.int32 = tid // LANES_PER_GROUP
        warp = _make_warp_uniform(warp_raw)
        lane = tid % LANES_PER_GROUP
    k_start: T.int32 = lane * VEC_SIZE

    v_tile: T.let[T.int32] = linear_cta % NUM_V_TILES
    cta_head: T.let[T.int32] = linear_cta // NUM_V_TILES
    hv: T.let[T.int32] = cta_head % NUM_V_HEADS
    n: T.let[T.int32] = cta_head // NUM_V_HEADS
    h: T.let[T.int32] = hv // (NUM_V_HEADS // NUM_HEADS)

    effective_state_slot_stride: T.let[T.int64] = T.if_then_else(
        PADDED_POOL, state_slot_stride, T.int64(NUM_V_HEADS * V * K)
    )
    effective_state_head_stride: T.let[T.int64] = T.if_then_else(
        PADDED_POOL, state_head_stride, T.int64(V * K)
    )
    effective_q_batch_stride: T.let[T.int64] = T.if_then_else(
        PACKED_QKV, q_batch_stride, T.int64(SEQ_LEN * NUM_HEADS * K)
    )
    effective_k_batch_stride: T.let[T.int64] = T.if_then_else(
        PACKED_QKV, k_batch_stride, T.int64(SEQ_LEN * NUM_HEADS * K)
    )
    effective_v_batch_stride: T.let[T.int64] = T.if_then_else(
        PACKED_QKV, v_batch_stride, T.int64(SEQ_LEN * NUM_V_HEADS * V)
    )

    read_slot_raw: T.int32 = _global_load_s32(read_indices, n)
    A_value: T.float32 = _global_load_f32(A_log, hv)
    dt_value: T.float32 = _global_load_f32(dt_bias, hv)

    pool = T.SMEMPool()
    smem_raw = pool.alloc((SHARED_BYTES,), "uint8", align=16)
    s_q = T.decl_buffer(
        (SEQ_LEN * (K + 8),),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=0,
        align=16,
    )
    s_k = T.decl_buffer(
        (SEQ_LEN * (K + 8),),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=S_K_BYTE_OFFSET,
        align=16,
    )
    s_g = T.decl_buffer(
        (SEQ_LEN,),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=S_G_BYTE_OFFSET,
        align=16,
    )
    s_beta = T.decl_buffer(
        (SEQ_LEN,),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=S_BETA_BYTE_OFFSET,
        align=16,
    )
    s_v = T.decl_buffer(
        (SEQ_LEN * TILE_V,),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=S_V_BYTE_OFFSET,
        align=16,
    )
    s_output = T.decl_buffer(
        (SEQ_LEN * TILE_V,),
        "bfloat16",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=S_OUTPUT_BYTE_OFFSET,
        align=16,
    )
    pool.commit()

    r_h = T.alloc_local((ILP_ROWS * VEC_SIZE,), "float32", align=16)
    r_q = T.alloc_local((VEC_SIZE,), "float32", align=16)
    r_k = T.alloc_local((VEC_SIZE,), "float32", align=16)
    r_k_output = T.alloc_local((VEC_SIZE,), "float32", align=16)
    r_q_bits = T.alloc_local((VEC_SIZE,), "uint16")
    r_k_bits = T.alloc_local((VEC_SIZE,), "uint16")
    gate_scratch = T.alloc_local((5,), "float32")
    gate_pair_value = T.alloc_local((1,), "uint64")

    if read_slot_raw >= 0:
        write_slot_raw: T.int32 = read_slot_raw
        if not SAME_POOL:
            write_slot_raw = _global_load_s32(write_indices, n)
        write_slot: T.int32 = T.if_then_else(write_slot_raw < 0, read_slot_raw, write_slot_raw)

        read_state_base: T.let[T.int64] = (
            T.cast(read_slot_raw, "int64") * effective_state_slot_stride
            + T.cast(hv, "int64") * effective_state_head_stride
        )
        write_state_base: T.int64 = read_state_base
        if not SAME_POOL:
            write_state_base = (
                T.cast(write_slot, "int64") * effective_state_slot_stride
                + T.cast(hv, "int64") * effective_state_head_stride
            )

        if warp == 0:
            exp_A_value: T.float32 = _exp2_f32(_mul_f32(A_value, T.float32(LOG2_E)))
            for t in T.unroll(SEQ_LEN):
                q_base: T.let[T.int64] = T.cast(n, "int64") * effective_q_batch_stride + T.cast(
                    (t * NUM_HEADS + h) * K + k_start, "int64"
                )
                k_base: T.let[T.int64] = T.cast(n, "int64") * effective_k_batch_stride + T.cast(
                    (t * NUM_HEADS + h) * K + k_start, "int64"
                )
                q_input_ptr: T.let = q.ptr_to([q_base])
                k_input_ptr: T.let = k.ptr_to([k_base])
                for elem in T.unroll(VEC_SIZE):
                    r_q_bits[elem] = _global_load_u16_ptr_offset(
                        q_input_ptr, elem * 2, USE_NATIVE_OFFSETS
                    )
                for elem in T.unroll(VEC_SIZE):
                    r_k_bits[elem] = _global_load_u16_ptr_offset(
                        k_input_ptr, elem * 2, USE_NATIVE_OFFSETS
                    )
                for elem in T.unroll(VEC_SIZE):
                    r_q[elem] = _bf16_to_f32(r_q_bits[elem])
                    r_k[elem] = _bf16_to_f32(r_k_bits[elem])

                if USE_QK_L2NORM:
                    sum_q: T.float32 = T.float32(0.0)
                    sum_k: T.float32 = T.float32(0.0)
                    for elem in T.unroll(VEC_SIZE):
                        sum_q = _bf16_square_fma(r_q_bits[elem], sum_q)
                        sum_k = _bf16_square_fma(r_k_bits[elem], sum_k)
                    for delta_index in T.unroll(5):
                        delta: T.int32 = T.shift_right(T.int32(16), delta_index)
                        sum_q = _add_f32(
                            sum_q, T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), sum_q, delta, 32)
                        )
                        sum_k = _add_f32(
                            sum_k, T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), sum_k, delta, 32)
                        )
                    q_factor: T.float32 = _mul_f32(
                        _rsqrt_f32(_add_f32(sum_q, T.float32(1.0e-6))), T.float32(SCALE)
                    )
                    k_factor: T.float32 = _rsqrt_f32(_add_f32(sum_k, T.float32(1.0e-6)))
                    for elem in T.unroll(VEC_SIZE):
                        r_q[elem] = _mul_f32(r_q[elem], q_factor)
                        r_k[elem] = _mul_f32(r_k[elem], k_factor)
                else:
                    for elem in T.unroll(VEC_SIZE):
                        r_q[elem] = _mul_f32(r_q[elem], T.float32(SCALE))

                shared_base: T.let[T.int32] = t * (K + 8) + k_start
                shared_q_ptr: T.let = s_q.ptr_to([shared_base])
                for elem in T.unroll(VEC_SIZE):
                    _shared_store_f32_ptr_offset(
                        shared_q_ptr, elem * 4, r_q[elem], USE_NATIVE_OFFSETS
                    )
                    _shared_store_f32_ptr_offset(
                        shared_q_ptr, S_K_BYTE_OFFSET + elem * 4, r_k[elem], USE_NATIVE_OFFSETS
                    )

                gate_index: T.let[T.int32] = (n * SEQ_LEN + t) * NUM_V_HEADS + hv
                a_bits: T.uint16 = _global_load_u16(a, gate_index)
                b_bits: T.uint16 = _global_load_u16(b_gate, gate_index)
                _gate_pair_store(
                    gate_pair_value, gate_scratch, a_bits, b_bits, exp_A_value, dt_value
                )
                shared_g_ptr: T.let = s_g.ptr_to([t])
                _shared_store_f32_ptr(shared_g_ptr, T.cuda.float2_x(gate_pair_value[0]))
                _shared_store_f32_ptr_offset(
                    shared_g_ptr,
                    S_BETA_BYTE_OFFSET - S_G_BYTE_OFFSET,
                    T.cuda.float2_y(gate_pair_value[0]),
                    USE_NATIVE_OFFSETS,
                )

                if USE_SMEM_V and tid < TILE_V:
                    v_input_base: T.let[T.int64] = T.cast(
                        n, "int64"
                    ) * effective_v_batch_stride + T.cast(
                        (t * NUM_V_HEADS + hv) * V + v_tile * TILE_V + tid, "int64"
                    )
                    v_input_ptr: T.let = v.ptr_to([v_input_base])
                    v_bits: T.uint16 = _global_load_u16_ptr(v_input_ptr)
                    _shared_store_f32(s_v, t * TILE_V + tid, _bf16_to_f32(v_bits))
        else:
            if PREFETCH_ROWS > 0:
                pre_v_base: T.let[T.int32] = v_tile * TILE_V + warp * ROWS_PER_GROUP
                prefetch_base: T.let[T.int64] = read_state_base + T.cast(
                    pre_v_base * K + k_start, "int64"
                )
                prefetch_ptr: T.let = state.ptr_to([prefetch_base])
                for row in T.unroll(PREFETCH_ROWS):
                    _global_load_f32x4_b64_ptr_offset(
                        prefetch_ptr, row * K * 4, r_h, row * VEC_SIZE, USE_NATIVE_OFFSETS
                    )
            if USE_SMEM_V:
                for t in T.unroll(SEQ_LEN):
                    if tid < TILE_V:
                        v_input_base: T.let[T.int64] = T.cast(
                            n, "int64"
                        ) * effective_v_batch_stride + T.cast(
                            (t * NUM_V_HEADS + hv) * V + v_tile * TILE_V + tid, "int64"
                        )
                        v_input_ptr: T.let = v.ptr_to([v_input_base])
                        v_bits: T.uint16 = _global_load_u16_ptr(v_input_ptr)
                        _shared_store_f32(s_v, t * TILE_V + tid, _bf16_to_f32(v_bits))

        T.cuda.cta_sync()

        for iter_index in T.unroll(ITERS_PER_GROUP):
            v_base: T.let[T.int32] = v_tile * TILE_V + warp * ROWS_PER_GROUP + iter_index * ILP_ROWS
            read_offset: T.int64 = read_state_base + T.cast(v_base * K + k_start, "int64")
            if ILP_ROWS == 8 or warp == 0 or iter_index > 0:
                read_ptr: T.let = state.ptr_to([read_offset])
                for row in T.unroll(ILP_ROWS):
                    if ILP_ROWS < 8 and iter_index == 0:
                        _global_load_f32x4_b64_ptr_offset(
                            read_ptr, row * K * 4, r_h, row * VEC_SIZE, USE_NATIVE_OFFSETS
                        )
                    else:
                        _global_load_f32x4_ptr_offset(
                            read_ptr, row * K * 4, r_h, row * VEC_SIZE, USE_NATIVE_OFFSETS
                        )

            sums = T.alloc_local((ILP_ROWS,), "float32")
            residuals = T.alloc_local((ILP_ROWS,), "float32")
            output_sums = T.alloc_local((ILP_ROWS,), "float32")
            sum_lo = T.alloc_local((4,), "float32")
            sum_hi = T.alloc_local((4,), "float32")
            output_lo = T.alloc_local((4,), "float32")
            output_hi = T.alloc_local((4,), "float32")
            packed_value = T.alloc_local((1,), "uint64")
            output_pair_bits = T.alloc_local((1,), "uint32")
            output_scalar_bits = T.alloc_local((ILP_ROWS,), "uint16")

            for t in T.unroll(SEQ_LEN):
                shared_q_ptr: T.let = s_q.ptr_to([t * (K + 8) + k_start])
                if ILP_ROWS == 4:
                    _shared_load_f32x4_b64_ptr_offset(
                        shared_q_ptr, S_K_BYTE_OFFSET, r_k, USE_NATIVE_OFFSETS
                    )
                else:
                    _shared_load_f32x4_ptr(shared_q_ptr, r_q)
                    _shared_load_f32x4_ptr_offset(
                        shared_q_ptr, S_K_BYTE_OFFSET, r_k, USE_NATIVE_OFFSETS
                    )
                shared_g_ptr: T.let = s_g.ptr_to([t])
                g_value: T.float32 = _shared_load_f32_ptr(shared_g_ptr)
                beta: T.float32 = _shared_load_f32_ptr_offset(
                    shared_g_ptr, S_BETA_BYTE_OFFSET - S_G_BYTE_OFFSET, USE_NATIVE_OFFSETS
                )

                if ILP_ROWS == 4:
                    for row in T.unroll(4):
                        sum_lo[row] = T.float32(0.0)
                        sum_hi[row] = T.float32(0.0)
                    for pair in T.unroll(2):
                        for row in T.unroll(4):
                            base: T.int32 = row * VEC_SIZE + pair * 2
                            _mul_f32_store(r_h, base, r_h[base], g_value)
                            _mul_f32_store(r_h, base + 1, r_h[base + 1], g_value)
                            _packed_fma_store(
                                packed_value,
                                r_h[base],
                                r_h[base + 1],
                                r_k[pair * 2],
                                r_k[pair * 2 + 1],
                                sum_lo[row],
                                sum_hi[row],
                            )
                            sum_lo[row] = T.cuda.float2_x(packed_value[0])
                            sum_hi[row] = T.cuda.float2_y(packed_value[0])
                    for row in T.unroll(4):
                        _add_f32_store(sum_lo, row, sum_lo[row], sum_hi[row])
                else:
                    for row in T.unroll(ILP_ROWS):
                        sums[row] = T.float32(0.0)
                    for elem in T.unroll(VEC_SIZE):
                        for row in T.unroll(ILP_ROWS):
                            index: T.int32 = row * VEC_SIZE + elem
                            r_h[index] = _mul_f32(r_h[index], g_value)
                            sums[row] = _fma_f32(r_h[index], r_k[elem], sums[row])

                for delta_index in T.unroll(5):
                    delta: T.int32 = T.shift_right(T.int32(16), delta_index)
                    for row in T.unroll(ILP_ROWS):
                        if ILP_ROWS == 4:
                            _add_f32_store(
                                sum_lo,
                                row,
                                sum_lo[row],
                                T.cuda.__shfl_xor_sync(
                                    T.uint32(0xFFFFFFFF), sum_lo[row], delta, 32
                                ),
                            )
                        else:
                            sums[row] = _add_f32(
                                sums[row],
                                T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), sums[row], delta, 32),
                            )

                v_input_base: T.let[T.int64] = T.cast(
                    n, "int64"
                ) * effective_v_batch_stride + T.cast((t * NUM_V_HEADS + hv) * V + v_base, "int64")
                v_input_ptr: T.let = v.ptr_to([v_input_base])
                shared_v_ptr: T.let = s_v.ptr_to([t * TILE_V + v_base - v_tile * TILE_V])
                for row in T.unroll(ILP_ROWS):
                    if USE_SMEM_V:
                        v_value: T.float32 = _shared_load_f32_ptr_offset(
                            shared_v_ptr, row * 4, USE_NATIVE_OFFSETS
                        )
                        if ILP_ROWS == 4:
                            _sub_f32_store(residuals, row, v_value, sum_lo[row])
                            _mul_f32_store(residuals, row, residuals[row], beta)
                        else:
                            residuals[row] = _mul_f32(_sub_f32(v_value, sums[row]), beta)
                    else:
                        v_bits: T.uint16 = _global_load_u16_ptr_offset(
                            v_input_ptr, row * 2, USE_NATIVE_OFFSETS
                        )
                        if ILP_ROWS == 4:
                            _mixed_sub_bf16_f32_store(residuals, row, v_bits, sum_lo[row])
                            _mul_f32_store(residuals, row, residuals[row], beta)
                        else:
                            residuals[row] = _mul_f32(_mixed_sub_bf16_f32(v_bits, sums[row]), beta)

                if ILP_ROWS == 4:
                    _shared_load_f32x4_b64_ptr(shared_q_ptr, r_q)
                    if RELOAD_K_FOR_OUTPUT:
                        _shared_load_f32x4_b64_ptr_offset(
                            shared_q_ptr, S_K_BYTE_OFFSET, r_k_output, USE_NATIVE_OFFSETS
                        )
                    for row in T.unroll(4):
                        output_lo[row] = T.float32(0.0)
                        output_hi[row] = T.float32(0.0)
                    for pair in T.unroll(2):
                        for row in T.unroll(4):
                            base: T.int32 = row * VEC_SIZE + pair * 2
                            if RELOAD_K_FOR_OUTPUT:
                                _packed_fma_store(
                                    packed_value,
                                    r_k_output[pair * 2],
                                    r_k_output[pair * 2 + 1],
                                    residuals[row],
                                    residuals[row],
                                    r_h[base],
                                    r_h[base + 1],
                                )
                            else:
                                _packed_fma_store(
                                    packed_value,
                                    r_k[pair * 2],
                                    r_k[pair * 2 + 1],
                                    residuals[row],
                                    residuals[row],
                                    r_h[base],
                                    r_h[base + 1],
                                )
                            r_h[base] = T.cuda.float2_x(packed_value[0])
                            r_h[base + 1] = T.cuda.float2_y(packed_value[0])
                            _packed_fma_store(
                                packed_value,
                                r_h[base],
                                r_h[base + 1],
                                r_q[pair * 2],
                                r_q[pair * 2 + 1],
                                output_lo[row],
                                output_hi[row],
                            )
                            output_lo[row] = T.cuda.float2_x(packed_value[0])
                            output_hi[row] = T.cuda.float2_y(packed_value[0])
                    for row in T.unroll(4):
                        _add_f32_store(output_lo, row, output_lo[row], output_hi[row])
                else:
                    for row in T.unroll(ILP_ROWS):
                        output_sums[row] = T.float32(0.0)
                    for elem in T.unroll(VEC_SIZE):
                        for row in T.unroll(ILP_ROWS):
                            index: T.int32 = row * VEC_SIZE + elem
                            r_h[index] = _fma_f32(r_k[elem], residuals[row], r_h[index])
                            output_sums[row] = _fma_f32(r_h[index], r_q[elem], output_sums[row])

                if CACHE_INTERMEDIATE_STATES and ILP_ROWS != 4:
                    intermediate_base: T.let[T.int64] = T.cast(
                        ((n * SEQ_LEN + t) * NUM_V_HEADS + hv) * V * K + v_base * K + k_start,
                        "int64",
                    )
                    intermediate_ptr: T.let = intermediate.ptr_to([intermediate_base])
                    for row in T.unroll(ILP_ROWS):
                        if ILP_ROWS == 2 and t > 0:
                            _global_store_f32x4_b64_ptr_offset(
                                intermediate_ptr,
                                row * K * 4,
                                r_h,
                                row * VEC_SIZE,
                                USE_NATIVE_OFFSETS,
                            )
                        else:
                            _global_store_f32x4_ptr_offset(
                                intermediate_ptr,
                                row * K * 4,
                                r_h,
                                row * VEC_SIZE,
                                USE_NATIVE_OFFSETS,
                            )
                if PER_TOKEN_POOL_SCATTER and ILP_ROWS != 4:
                    scatter_slot: T.int32 = _global_load_s32(ssm_state_indices, n * SEQ_LEN + t)
                    scatter_base: T.let[T.int64] = (
                        T.cast(scatter_slot, "int64") * effective_state_slot_stride
                        + T.cast(hv, "int64") * effective_state_head_stride
                        + T.cast(v_base * K + k_start, "int64")
                    )
                    scatter_ptr: T.let = state.ptr_to([scatter_base])
                    for row in T.unroll(ILP_ROWS):
                        _global_store_f32x4_ptr_offset(
                            scatter_ptr, row * K * 4, r_h, row * VEC_SIZE, USE_NATIVE_OFFSETS
                        )

                for delta_index in T.unroll(5):
                    delta: T.int32 = T.shift_right(T.int32(16), delta_index)
                    for row in T.unroll(ILP_ROWS):
                        if ILP_ROWS == 4:
                            _add_f32_store(
                                output_lo,
                                row,
                                output_lo[row],
                                T.cuda.__shfl_xor_sync(
                                    T.uint32(0xFFFFFFFF), output_lo[row], delta, 32
                                ),
                            )
                        else:
                            output_sums[row] = _add_f32(
                                output_sums[row],
                                T.cuda.__shfl_xor_sync(
                                    T.uint32(0xFFFFFFFF), output_sums[row], delta, 32
                                ),
                            )

                if lane == 0:
                    output_base: T.let[T.int64] = T.cast(
                        ((n * SEQ_LEN + t) * NUM_V_HEADS + hv) * V + v_base, "int64"
                    )
                    output_ptr: T.let = output.ptr_to([output_base])
                    shared_output_ptr: T.let = s_output.ptr_to(
                        [t * TILE_V + v_base - v_tile * TILE_V]
                    )
                    if ILP_ROWS == 2:
                        T.evaluate(
                            T.ptx.cvt.rn.bf16x2.f32(
                                output_pair_bits[0], output_sums[1], output_sums[0]
                            )
                        )
                        T.evaluate(T.ptx.st.global_.b32(output_ptr, output_pair_bits[0]))
                    elif USE_PACKED_OUTPUT:
                        for pair in T.unroll(ILP_ROWS // 2):
                            output_value_0: T.float32 = output_sums[pair * 2]
                            output_value_1: T.float32 = output_sums[pair * 2 + 1]
                            if ILP_ROWS == 4:
                                output_value_0 = output_lo[pair * 2]
                                output_value_1 = output_lo[pair * 2 + 1]
                            T.evaluate(
                                T.ptx.cvt.rn.bf16x2.f32(
                                    output_pair_bits[0], output_value_1, output_value_0
                                )
                            )
                            if USE_SMEM_V:
                                _shared_store_u32_ptr_offset(
                                    shared_output_ptr,
                                    pair * 4,
                                    output_pair_bits[0],
                                    USE_NATIVE_OFFSETS,
                                )
                            else:
                                _global_store_u32_ptr_offset(
                                    output_ptr, pair * 4, output_pair_bits[0], USE_NATIVE_OFFSETS
                                )
                    else:
                        for row in T.unroll(ILP_ROWS):
                            output_value: T.float32 = output_sums[row]
                            if ILP_ROWS == 4:
                                output_value = output_lo[row]
                            _f32_to_bf16_store(output_scalar_bits, row, output_value)
                            if USE_SMEM_V:
                                _shared_store_u16_ptr_offset(
                                    shared_output_ptr,
                                    row * 2,
                                    output_scalar_bits[row],
                                    USE_NATIVE_OFFSETS,
                                )
                            else:
                                _global_store_u16_ptr_offset(
                                    output_ptr, row * 2, output_scalar_bits[row], USE_NATIVE_OFFSETS
                                )

                if CACHE_INTERMEDIATE_STATES and ILP_ROWS == 4:
                    intermediate_base: T.let[T.int64] = T.cast(
                        ((n * SEQ_LEN + t) * NUM_V_HEADS + hv) * V * K + v_base * K + k_start,
                        "int64",
                    )
                    intermediate_ptr: T.let = intermediate.ptr_to([intermediate_base])
                    for row in T.unroll(4):
                        _global_store_f32x4_b64_ptr_offset(
                            intermediate_ptr, row * K * 4, r_h, row * VEC_SIZE, USE_NATIVE_OFFSETS
                        )
                if PER_TOKEN_POOL_SCATTER and ILP_ROWS == 4:
                    scatter_slot: T.int32 = _global_load_s32(ssm_state_indices, n * SEQ_LEN + t)
                    scatter_base: T.let[T.int64] = (
                        T.cast(scatter_slot, "int64") * effective_state_slot_stride
                        + T.cast(hv, "int64") * effective_state_head_stride
                        + T.cast(v_base * K + k_start, "int64")
                    )
                    scatter_ptr: T.let = state.ptr_to([scatter_base])
                    for row in T.unroll(4):
                        _global_store_f32x4_b64_ptr_offset(
                            scatter_ptr, row * K * 4, r_h, row * VEC_SIZE, USE_NATIVE_OFFSETS
                        )

            if not DISABLE_STATE_UPDATE and not PER_TOKEN_POOL_SCATTER:
                if write_slot_raw >= 0:
                    write_offset: T.int64 = write_state_base + T.cast(v_base * K + k_start, "int64")
                    write_ptr: T.let = state.ptr_to([write_offset])
                    for row in T.unroll(ILP_ROWS):
                        if ILP_ROWS == 4 or (ILP_ROWS == 2 and CACHE_INTERMEDIATE_STATES):
                            _global_store_f32x4_b64_ptr_offset(
                                write_ptr, row * K * 4, r_h, row * VEC_SIZE, USE_NATIVE_OFFSETS
                            )
                        else:
                            _global_store_f32x4_ptr_offset(
                                write_ptr, row * K * 4, r_h, row * VEC_SIZE, USE_NATIVE_OFFSETS
                            )

        if USE_SMEM_V:
            T.cuda.cta_sync()
            output_tile_base: T.let[T.int64] = T.cast(
                ((n * SEQ_LEN) * NUM_V_HEADS + hv) * V + v_tile * TILE_V, "int64"
            )
            if tid < TILE_V:
                output_tile_ptr: T.let = output.ptr_to([output_tile_base + tid])
                shared_output_ptr: T.let = s_output.ptr_to([tid])
                for t in T.unroll(SEQ_LEN):
                    output_bits: T.uint16 = _shared_load_u16_ptr_offset(
                        shared_output_ptr, t * TILE_V * 2, USE_NATIVE_OFFSETS
                    )
                    _global_store_u16_ptr_offset(
                        output_tile_ptr, t * NUM_V_HEADS * V * 2, output_bits, USE_NATIVE_OFFSETS
                    )


def _pool_factor(config: dict[str, Any]) -> int:
    factor = 1
    if config.get("negative_read_index", False) or config.get("negative_write_index", False):
        factor += 1
    if config.get("per_token_pool_scatter", False):
        factor += int(config["seq_len"])
    if not config.get("same_pool", True):
        factor += 1
    return factor


def get_kernel(**kwargs: Any):
    """Return the source-specialized scaffold PrimFunc."""
    config = dict(kwargs)
    _require_supported_config(config)
    seq_len = int(config["seq_len"])
    num_v_heads = int(config["num_v_heads"])
    work_units = int(config["batch"]) * num_v_heads
    tile_v, ilp_rows, use_smem_v = _target_config(
        int(config["batch"]),
        seq_len,
        int(config["num_heads"]),
        num_v_heads,
        disable_state_update=bool(config.get("disable_state_update", False)),
    )
    scatter = bool(config.get("per_token_pool_scatter", False))
    scatter_flat = scatter and not bool(config.get("padded_pool", False))
    cache = bool(config.get("cache_intermediate_states", False))
    pool_factor = int(config.get("pool_factor_override", _pool_factor(config)))
    intermediate_batch_stride = 0
    if cache:
        intermediate_batch_stride = seq_len * num_v_heads * V * K
    qk_bytes = 4 * ((seq_len - 1) * (K + 8) + K)
    gate_bytes_aligned = ((4 * seq_len + 15) // 16) * 16
    s_g_byte_offset = 2 * qk_bytes
    s_v_byte_offset = s_g_byte_offset + 2 * gate_bytes_aligned
    kernel = _gdn_decode_fp32_mtp_warp.specialize(
        SEQ_LEN=seq_len,
        NUM_HEADS=int(config["num_heads"]),
        NUM_V_HEADS=num_v_heads,
        TILE_V=tile_v,
        NUM_V_TILES=V // tile_v,
        ILP_ROWS=ilp_rows,
        USE_SMEM_V=use_smem_v,
        USE_QK_L2NORM=bool(config.get("use_qk_l2norm", True)),
        USE_NATIVE_OFFSETS=_HAS_NATIVE_PTX_ADDR
        and (seq_len == 2 or 3 < seq_len < 8)
        and not (seq_len == 4 and int(config["num_heads"]) <= 8 and work_units == 1024),
        RELOAD_K_FOR_OUTPUT=seq_len == 8 and not use_smem_v,
        USE_CANONICAL_WARP_ID=seq_len == 4,
        USE_PACKED_OUTPUT=not (
            (seq_len in (5, 7) and work_units == 512)
            or (seq_len == 8 and int(config["num_heads"]) <= 4)
        ),
        DISABLE_STATE_UPDATE=bool(config.get("disable_state_update", False)),
        CACHE_INTERMEDIATE_STATES=cache,
        SAME_POOL=bool(config.get("same_pool", True)),
        PER_TOKEN_POOL_SCATTER=scatter,
        PER_TOKEN_POOL_SCATTER_FLAT=scatter_flat,
        PADDED_POOL=bool(config.get("padded_pool", False)),
        PACKED_QKV=bool(config.get("packed_qkv", False)),
        POOL_FACTOR=pool_factor,
        INTERMEDIATE_BATCH_STRIDE=intermediate_batch_stride,
        INTERMEDIATE_DUMMY_ELEMENTS=0 if cache else 1,
        SSM_BATCH_STRIDE=seq_len if scatter else 0,
        SSM_DUMMY_ELEMENTS=0 if scatter else 1,
        SHARED_BYTES=8 * seq_len * (K + 8) + 8 * seq_len + 6 * seq_len * tile_v + 128,
        S_K_BYTE_OFFSET=qk_bytes,
        S_G_BYTE_OFFSET=s_g_byte_offset,
        S_BETA_BYTE_OFFSET=s_g_byte_offset + gate_bytes_aligned,
        S_V_BYTE_OFFSET=s_v_byte_offset,
        S_OUTPUT_BYTE_OFFSET=s_v_byte_offset + 4 * seq_len * tile_v,
        ROWS_PER_GROUP=tile_v // NUM_GROUPS,
        ITERS_PER_GROUP=(tile_v // NUM_GROUPS) // ilp_rows,
        PREFETCH_ROWS=0 if ilp_rows == 8 else ilp_rows,
    )
    kernel = kernel.with_attr(
        "tirx.kernel_launch_params", ["blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory"]
    )
    return kernel


def _allocate_pool(
    pool_slots: int,
    num_v_heads: int,
    *,
    padded: bool,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if padded:
        backing = (
            torch.randn(
                (pool_slots, num_v_heads * 2 + 1, V, K),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.05
        )
        return backing[:, : num_v_heads * 2 : 2], backing
    backing = (
        torch.randn(
            (pool_slots, num_v_heads, V, K), dtype=torch.float32, device=device, generator=generator
        )
        * 0.05
    )
    return backing, backing


def _clone_pool_layout(pool: torch.Tensor, *, padded: bool) -> tuple[torch.Tensor, torch.Tensor]:
    pool_slots, num_v_heads = pool.shape[:2]
    backing = torch.empty(
        (pool_slots, num_v_heads * 2 + 1 if padded else num_v_heads, V, K),
        dtype=pool.dtype,
        device=pool.device,
    )
    view = backing[:, : num_v_heads * 2 : 2] if padded else backing
    view.copy_(pool)
    return view, backing


def _make_qkv(
    batch: int,
    seq_len: int,
    num_heads: int,
    num_v_heads: int,
    *,
    packed: bool,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not packed:
        q = (
            torch.randn(
                (batch, seq_len, num_heads, K),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            * 0.05
        )
        k = (
            torch.randn(
                (batch, seq_len, num_heads, K),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            * 0.05
        )
        v = (
            torch.randn(
                (batch, seq_len, num_v_heads, V),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            * 0.05
        )
        return q, k, v, None

    q_elements = seq_len * num_heads * K
    k_elements = seq_len * num_heads * K
    v_elements = seq_len * num_v_heads * V
    row_elements = q_elements + k_elements + v_elements + 128
    backing = (
        torch.randn((batch, row_elements), dtype=torch.bfloat16, device=device, generator=generator)
        * 0.05
    )
    q = backing.as_strided((batch, seq_len, num_heads, K), (row_elements, num_heads * K, K, 1), 0)
    k = backing.as_strided(
        (batch, seq_len, num_heads, K), (row_elements, num_heads * K, K, 1), q_elements
    )
    v = backing.as_strided(
        (batch, seq_len, num_v_heads, V),
        (row_elements, num_v_heads * V, V, 1),
        q_elements + k_elements,
    )
    return q, k, v, backing


def _device_from_config(config: dict[str, Any]) -> torch.device:
    configured_device = config.get("device")
    device = (
        torch.device(configured_device)
        if configured_device is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SkipTest("CUDA is required for FP32 MTP warp GDN decode")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(f"FP32 MTP warp GDN decode requires SM100, got {capability}")
    return device


@functools.cache
def _compile_tirx(
    work_units: int,
    seq_len: int,
    num_heads: int,
    num_v_heads: int,
    tile_v: int,
    ilp_rows: int,
    use_smem_v: bool,
    pool_factor: int,
    use_qk_l2norm: bool,
    disable_state_update: bool,
    cache_intermediate_states: bool,
    same_pool: bool,
    per_token_pool_scatter: bool,
    padded_pool: bool,
    packed_qkv: bool,
):
    from tirx_kernels.runner import compile_kernel

    representative_work_units = work_units
    config = {
        "seq_len": seq_len,
        "batch": representative_work_units // num_v_heads,
        "num_heads": num_heads,
        "num_v_heads": num_v_heads,
        "tile_v": tile_v,
        "ilp_rows": ilp_rows,
        "use_smem_v": use_smem_v,
        "pool_factor_override": pool_factor,
        "use_qk_l2norm": use_qk_l2norm,
        "disable_state_update": disable_state_update,
        "cache_intermediate_states": cache_intermediate_states,
        "same_pool": same_pool,
        "per_token_pool_scatter": per_token_pool_scatter,
        "padded_pool": padded_pool,
        "packed_qkv": packed_qkv,
    }
    reg_level: int | None = None
    if seq_len == 2:
        reg_level = 4 if tile_v == 16 or work_units == 2048 else 0
    elif seq_len == 4 and num_heads == 2 and tile_v == 32:
        reg_level = 4
    elif seq_len in (3, 4) and num_heads < 16 and tile_v == 32:
        reg_level = 4
    elif seq_len == 4 and num_heads == 4 and tile_v == 64:
        reg_level = 4
    elif seq_len == 5 and tile_v == 32:
        reg_level = 4
    elif seq_len == 7 and tile_v == 32:
        reg_level = 0
    elif seq_len == 8 and tile_v == 64 and num_heads >= 8:
        reg_level = 0
    elif seq_len == 8 and num_heads <= 4:
        reg_level = 4
    if reg_level is None:
        os.environ.pop("TVM_CUDA_PTXAS_REG_LEVEL", None)
    else:
        os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = str(reg_level)
    return compile_kernel(get_kernel(**config))


def _compile_tirx_for_config(config: dict[str, Any]):
    return _compile_tirx(
        int(config["batch"]) * int(config["num_v_heads"]),
        int(config["seq_len"]),
        int(config["num_heads"]),
        int(config["num_v_heads"]),
        int(config["tile_v"]),
        int(config["ilp_rows"]),
        bool(config["use_smem_v"]),
        _pool_factor(config),
        bool(config.get("use_qk_l2norm", True)),
        bool(config.get("disable_state_update", False)),
        bool(config.get("cache_intermediate_states", False)),
        bool(config.get("same_pool", True)),
        bool(config.get("per_token_pool_scatter", False)),
        bool(config.get("padded_pool", False)),
        bool(config.get("packed_qkv", False)),
    )


def _tirx_executable(case: dict[str, Any]):
    return _compile_tirx_for_config(case["config"])


def _storage_span(tensor: torch.Tensor, elements: int) -> torch.Tensor:
    return tensor.as_strided((elements,), (1,))


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    config = case["config"]
    state = case["tirx_state"]
    q = case["q"]
    k = case["k"]
    v = case["v"]
    batch = int(config["batch"])
    state_elements = int(state.stride(0)) * case["pool_slots"]
    return (
        _storage_span(state, state_elements),
        _storage_span(case["tirx_intermediate"], case["tirx_intermediate"].numel()),
        case["A_log"],
        case["a"].reshape(-1),
        case["dt_bias"],
        _storage_span(q, int(q.stride(0)) * (batch - 1) + int(q[0].numel())),
        _storage_span(k, int(k.stride(0)) * (batch - 1) + int(k[0].numel())),
        _storage_span(v, int(v.stride(0)) * (batch - 1) + int(v[0].numel())),
        case["b_gate"].reshape(-1),
        case["tirx_output"].reshape(-1),
        case["read_indices"],
        case["write_indices"],
        case["ssm_state_indices"].reshape(-1),
        int(state.stride(0)),
        int(state.stride(1)),
        int(q.stride(0)),
        int(k.stride(0)),
        int(v.stride(0)),
        batch,
    )


def _run_reference(case: dict[str, Any]) -> torch.Tensor:
    from tirx_kernels.flashinfer._gdn_reference import gated_delta_rule_decode

    config = case["config"]
    return gated_delta_rule_decode(
        q=case["q"],
        k=case["k"],
        v=case["v"],
        state_pool=case["source_state"],
        read_indices=case["read_indices"],
        write_indices=(
            case["read_indices"] if bool(config.get("same_pool", True)) else case["write_indices"]
        ),
        A_log=case["A_log"],
        a=case["a"],
        dt_bias=case["dt_bias"],
        b=case["b_gate"],
        scale=SCALE,
        output=case["source_output"],
        intermediate_states=(
            case["source_intermediate"]
            if bool(config.get("cache_intermediate_states", False))
            else None
        ),
        ssm_state_indices=(
            case["ssm_state_indices"] if bool(config.get("per_token_pool_scatter", False)) else None
        ),
        disable_state_update=bool(config.get("disable_state_update", False)),
        use_qk_l2norm=bool(config.get("use_qk_l2norm", True)),
    )


def _assert_case_close(case: dict[str, Any]) -> None:
    config = case["config"]
    torch.testing.assert_close(
        case["tirx_output"].float(), case["source_output"].float(), atol=1.0e-3, rtol=5.0e-3
    )
    torch.testing.assert_close(case["tirx_state"], case["source_state"], atol=2.0e-5, rtol=2.0e-5)
    if bool(config.get("cache_intermediate_states", False)):
        torch.testing.assert_close(
            case["tirx_intermediate"], case["source_intermediate"], atol=2.0e-5, rtol=2.0e-5
        )
    if case["qkv_backing"] is not None:
        torch.testing.assert_close(case["qkv_backing"], case["qkv_snapshot"], atol=0, rtol=0)
    if bool(config.get("padded_pool", False)):
        backing = case["tirx_state_backing"]
        snapshot = case["tirx_state_backing_snapshot"]
        torch.testing.assert_close(backing[:, 1::2], snapshot[:, 1::2], atol=0, rtol=0)
        torch.testing.assert_close(backing[:, -1], snapshot[:, -1], atol=0, rtol=0)


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Create deterministic, independently mutable TIRx and FlashInfer cases."""
    config = dict(kwargs)
    _require_supported_config(config)
    device = _device_from_config(config)
    batch = int(config["batch"])
    seq_len = int(config["seq_len"])
    num_heads = int(config["num_heads"])
    num_v_heads = int(config["num_v_heads"])
    padded_pool = bool(config.get("padded_pool", False))
    same_pool = bool(config.get("same_pool", True))
    scatter = bool(config.get("per_token_pool_scatter", False))
    negative_read = bool(config.get("negative_read_index", False))
    negative_write = bool(config.get("negative_write_index", False))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.get("seed", 0)) + 20260813)

    pool_factor = _pool_factor(config)
    pool_slots = batch * pool_factor
    initial_pool, initial_backing = _allocate_pool(
        pool_slots, num_v_heads, padded=padded_pool, device=device, generator=generator
    )
    tirx_state, tirx_state_backing = _clone_pool_layout(initial_pool, padded=padded_pool)
    if padded_pool and scatter:
        # The frozen source cannot construct native-4D + scatter.  Its flat
        # source specialization is algorithmically identical for the logical
        # pool and remains the trusted oracle for this combined target case.
        source_state = initial_pool.contiguous()
        source_state_backing = source_state
    else:
        source_state, source_state_backing = _clone_pool_layout(initial_pool, padded=padded_pool)

    slot_offset = 1 if (negative_read or negative_write) else 0
    read_indices = torch.arange(batch, dtype=torch.int32, device=device) + slot_offset
    if negative_read:
        read_indices[-1] = -1

    next_slot = slot_offset + batch
    if scatter:
        scatter_indices = torch.arange(
            next_slot, next_slot + batch * seq_len, dtype=torch.int32, device=device
        ).reshape(batch, seq_len)
        next_slot += batch * seq_len
    else:
        scatter_indices = None
    if same_pool:
        write_indices = read_indices.clone()
    else:
        write_indices = torch.arange(next_slot, next_slot + batch, dtype=torch.int32, device=device)
        if negative_write:
            write_indices[-1] = -1

    q, k, v, qkv_backing = _make_qkv(
        batch,
        seq_len,
        num_heads,
        num_v_heads,
        packed=bool(config.get("packed_qkv", False)),
        device=device,
        generator=generator,
    )
    A_log = (
        torch.randn((num_v_heads,), dtype=torch.float32, device=device, generator=generator) * 0.1
    )
    dt_bias = (
        torch.randn((num_v_heads,), dtype=torch.float32, device=device, generator=generator) * 0.1
    )
    a = (
        torch.randn(
            (batch, seq_len, num_v_heads), dtype=torch.bfloat16, device=device, generator=generator
        )
        * 0.05
    )
    b_gate = (
        torch.randn(
            (batch, seq_len, num_v_heads), dtype=torch.bfloat16, device=device, generator=generator
        )
        * 0.05
    )
    output_initial = torch.randn(
        (batch, seq_len, num_v_heads, V), dtype=torch.bfloat16, device=device, generator=generator
    )
    if bool(config.get("cache_intermediate_states", False)):
        intermediate_initial = (
            torch.randn(
                (batch, seq_len, num_v_heads, V, K),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.05
        )
        tirx_intermediate = intermediate_initial.clone()
        source_intermediate = intermediate_initial.clone()
    else:
        tirx_intermediate = torch.zeros((1,), dtype=torch.float32, device=device)
        source_intermediate = torch.zeros((1,), dtype=torch.float32, device=device)

    return {
        "config": config,
        "pool_slots": pool_slots,
        "initial_pool": initial_pool.clone(),
        "initial_backing": initial_backing,
        "tirx_state": tirx_state,
        "tirx_state_backing": tirx_state_backing,
        "tirx_state_backing_snapshot": tirx_state_backing.clone(),
        "source_state": source_state,
        "source_state_backing": source_state_backing,
        "read_indices": read_indices,
        "write_indices": write_indices,
        "ssm_state_indices": (
            scatter_indices
            if scatter_indices is not None
            else torch.zeros((1,), dtype=torch.int32, device=device)
        ),
        "q": q,
        "k": k,
        "v": v,
        "qkv_backing": qkv_backing,
        "qkv_snapshot": qkv_backing.clone() if qkv_backing is not None else None,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "a": a,
        "b_gate": b_gate,
        "tirx_output": output_initial.clone(),
        "source_output": output_initial.clone(),
        "tirx_intermediate": tirx_intermediate,
        "source_intermediate": source_intermediate,
    }


def run_test(**kwargs: Any) -> None:
    case = prepare_data(**kwargs)
    executable = _tirx_executable(case)
    executable(*_tirx_args(case))
    torch.cuda.synchronize(case["tirx_state"].device)
    _run_reference(case)
    torch.cuda.synchronize(case["tirx_state"].device)
    _assert_case_close(case)


def prepare_bench(**kwargs: Any):
    """Compile the selected FP32 MTP warp specialization before CUDA setup."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    config = dict(kwargs)
    _require_supported_config(config)
    executable = _compile_tirx_for_config(config)
    return prepared_gpu_benchmark(run_gpu, {"config": dict(kwargs), "executable": executable})


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
    from tirx_kernels.runner import bench

    kwargs = {**prepared["config"], **kwargs}
    case = prepare_data(**kwargs)
    executable = prepared["executable"]
    args = _tirx_args(case)

    def source_builder():
        import flashinfer.gdn_decode as public

        config = case["config"]

        def launch():
            public.gated_delta_rule_mtp(
                q=case["q"],
                k=case["k"],
                v=case["v"],
                initial_state=case["source_state"],
                initial_state_indices=case["read_indices"],
                A_log=case["A_log"],
                a=case["a"],
                dt_bias=case["dt_bias"],
                b=case["b_gate"],
                scale=SCALE,
                output=case["source_output"],
                intermediate_states_buffer=(
                    case["source_intermediate"]
                    if bool(config.get("cache_intermediate_states", False))
                    else None
                ),
                ssm_state_indices=(
                    case["ssm_state_indices"]
                    if bool(config.get("per_token_pool_scatter", False))
                    else None
                ),
                disable_state_update=bool(config.get("disable_state_update", False)),
                use_qk_l2norm=bool(config.get("use_qk_l2norm", True)),
                output_state_indices=(
                    None if bool(config.get("same_pool", True)) else case["write_indices"]
                ),
            )

        executable(*args)
        launch()
        torch.cuda.synchronize(case["tirx_state"].device)
        _assert_case_close(case)
        for _ in range(2):
            launch()
        torch.cuda.synchronize(case["source_state"].device)
        return launch

    return bench(
        {"tirx": lambda: executable(*args)},
        references={"flashinfer_cutedsl": source_builder},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
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
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_test",
]
