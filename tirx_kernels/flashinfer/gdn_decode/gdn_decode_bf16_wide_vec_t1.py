# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM100 BF16 wide-vector single-token GDN decode kernel.

Upstream source: flashinfer/gdn_kernels/gdn_decode_bf16_state.py.
"""

from __future__ import annotations

import functools
import math
from typing import Any
from unittest import SkipTest

import torch

from tirx_kernels.runner import bench
from tvm.script import tirx as T

KERNEL_META = {
    "name": "gdn_decode_bf16_wide_vec_t1",
    "category": "flashinfer",
    "compute_capability": 10,
}


SEQ_LEN = 1
K = 128
V = 128
THREADS = 128
NUM_WARPS = 4
NUM_GROUPS = 8
LANES_PER_GROUP = 16
ELEMS_PER_LANE = 8
ILP_ROWS = 4
SHARED_BYTES = 1356
LOG2_E = 1.4426950408889634
LN_2 = 0.6931471805599453
SCALE = 1.0 / math.sqrt(K)


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


def _max_s32(lhs, rhs):
    return _ptx_binary("max.s32", lhs, rhs, dtype="int32")


def _global_load_u16(buffer, index):
    out = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.ld.global_.b16(out[0], buffer.ptr_to([index])))
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


def _shared_load_f32(buffer, index):
    out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.ld.shared.b32(out[0], buffer.ptr_to([index])))
    return T.reinterpret("float32", out[0])


def _shared_store_f32(buffer, index, value):
    T.evaluate(T.ptx.st.shared.b32(buffer.ptr_to([index]), T.reinterpret("uint32", value)))


@T.inline
def _load_state_bf16x8(buffer, index, values, value_offset):
    words = T.alloc_local((4,), "uint32", align=16)
    T.evaluate(
        T.ptx["ld.global.L1::evict_first.v4.b32"](
            words[0], words[1], words[2], words[3], buffer.ptr_to([index])
        )
    )
    for pair in T.unroll(4):
        values[value_offset + pair * 2] = T.cuda.uint_as_float(
            T.shift_left(words[pair], T.uint32(16))
        )
        values[value_offset + pair * 2 + 1] = T.cuda.uint_as_float(
            T.bitwise_and(words[pair], T.uint32(0xFFFF0000))
        )


@T.inline
def _load_state_bf16x8_evict_first(buffer, index, values, value_offset):
    words = T.alloc_local((4,), "uint32", align=16)
    T.evaluate(
        T.ptx["ld.global.L1::evict_first.v4.b32"](
            words[0], words[1], words[2], words[3], buffer.ptr_to([index])
        )
    )
    for pair in T.unroll(4):
        values[value_offset + pair * 2] = T.cuda.uint_as_float(
            T.shift_left(words[pair], T.uint32(16))
        )
        values[value_offset + pair * 2 + 1] = T.cuda.uint_as_float(
            T.bitwise_and(words[pair], T.uint32(0xFFFF0000))
        )


@T.inline
def _load_state_bf16x8_vector_buffer(buffer, index, values, value_offset):
    words = T.alloc_local((4,), "uint32", align=16)
    T.evaluate(
        T.ptx.ld.global_.v4.b32(
            words[0], words[1], words[2], words[3], buffer.ptr_to([index // ELEMS_PER_LANE])
        )
    )
    for pair in T.unroll(4):
        values[value_offset + pair * 2] = T.cuda.uint_as_float(
            T.shift_left(words[pair], T.uint32(16))
        )
        values[value_offset + pair * 2 + 1] = T.cuda.uint_as_float(
            T.bitwise_and(words[pair], T.uint32(0xFFFF0000))
        )


@T.inline
def _load_bf16x8_bits(buffer, index, values):
    words = T.alloc_local((4,), "uint32", align=16)
    T.evaluate(
        T.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
    )
    for pair in T.unroll(4):
        values[pair * 2] = T.cast(T.bitwise_and(words[pair], T.uint32(0xFFFF)), "uint16")
        values[pair * 2 + 1] = T.cast(T.shift_right(words[pair], T.uint32(16)), "uint16")


@T.inline
def _load_bf16x4_bits(buffer, index, values):
    words = T.alloc_local((2,), "uint32", align=8)
    T.evaluate(T.ptx.ld.global_.v2.b32(words[0], words[1], buffer.ptr_to([index])))
    for pair in T.unroll(2):
        values[pair * 2] = T.cast(T.bitwise_and(words[pair], T.uint32(0xFFFF)), "uint16")
        values[pair * 2 + 1] = T.cast(T.shift_right(words[pair], T.uint32(16)), "uint16")


@T.inline
def _store_state_f32x8(buffer, index, values, value_offset):
    words = T.alloc_local((4,), "uint32", align=16)
    for pair in T.unroll(4):
        words[pair] = T.cuda.float22bfloat162_rn(
            values[value_offset + pair * 2], values[value_offset + pair * 2 + 1]
        )
    T.evaluate(
        T.ptx.st.global_.v4.b32(buffer.ptr_to([index]), words[0], words[1], words[2], words[3])
    )


@T.inline
def _store_state_f32x8_vector_buffer(buffer, index, values, value_offset):
    words = T.alloc_local((4,), "uint32", align=16)
    for pair in T.unroll(4):
        words[pair] = T.cuda.float22bfloat162_rn(
            values[value_offset + pair * 2], values[value_offset + pair * 2 + 1]
        )
    T.evaluate(
        T.ptx["st.global.L1::evict_first.v4.b32"](
            buffer.ptr_to([index // ELEMS_PER_LANE]), words[0], words[1], words[2], words[3]
        )
    )


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


def _case(label: str, **kwargs: Any) -> dict[str, Any]:
    return {"label": label, **kwargs}


_TP_HEADS = ((16, 32), (8, 16), (4, 8), (2, 4))
_BATCH_SWEEP = (1, 4, 8, 16, 32, 64, 128, 256, 512)


def _production_tile_v(batch: int, num_v_heads: int) -> int | None:
    work_units = batch * num_v_heads
    if work_units >= 1024:
        return 128
    if work_units >= 512:
        return 64
    return None


BENCH_CONFIGS = [
    _case(
        f"b{batch}_h{num_heads}_hv{num_v_heads}_tv{tile_v}",
        batch=batch,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        tile_v=tile_v,
        seed=10000 + head_idx * len(_BATCH_SWEEP) + batch_idx,
    )
    for head_idx, (num_heads, num_v_heads) in enumerate(_TP_HEADS)
    for batch_idx, batch in enumerate(_BATCH_SWEEP)
    if (tile_v := _production_tile_v(batch, num_v_heads)) is not None
]

_CORRECTNESS_MODES = (
    ("l2off", {"use_qk_l2norm": False}),
    ("split", {"same_pool": False}),
    ("split_l2off", {"same_pool": False, "use_qk_l2norm": False}),
    ("padded", {"padded_pool": True}),
    ("padded_split", {"padded_pool": True, "same_pool": False}),
    ("packed_qkv", {"packed_qkv": True}),
    ("negative_read", {"negative_read_index": True}),
    ("negative_write", {"same_pool": False, "negative_write_index": True}),
    ("disable_update", {"disable_state_update": True}),
    ("disable_update_split", {"same_pool": False, "disable_state_update": True}),
    ("cache", {"cache_intermediate_states": True}),
    ("cache_split", {"same_pool": False, "cache_intermediate_states": True}),
)

_BOUNDARY_CASES = ((16, 16, 32, 64), (32, 16, 32, 128))

CONFIGS = [
    _case(
        f"b{batch}_h{num_heads}_hv{num_v_heads}_tv{tile_v}_{mode}",
        batch=batch,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        tile_v=tile_v,
        seed=20000 + boundary_idx * len(_CORRECTNESS_MODES) + mode_idx,
        **mode_kwargs,
    )
    for boundary_idx, (batch, num_heads, num_v_heads, tile_v) in enumerate(_BOUNDARY_CASES)
    for mode_idx, (mode, mode_kwargs) in enumerate(_CORRECTNESS_MODES)
]


@T.jit
def _gdn_decode_bf16_wide_vec_t1(
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
    state_slot_stride: T.int64,
    q_batch_stride: T.int64,
    k_batch_stride: T.int64,
    v_batch_stride: T.int64,
    batch: T.int32,
    *,
    NUM_HEADS: T.constexpr,
    NUM_V_HEADS: T.constexpr,
    TILE_V: T.constexpr,
    NUM_V_TILES: T.constexpr,
    ROWS_PER_GROUP: T.constexpr,
    ITERS_PER_GROUP: T.constexpr,
    POOL_FACTOR: T.constexpr,
    INTERMEDIATE_BATCH_STRIDE: T.constexpr,
    INTERMEDIATE_DUMMY_ELEMENTS: T.constexpr,
    USE_QK_L2NORM: T.constexpr,
    DISABLE_STATE_UPDATE: T.constexpr,
    CACHE_INTERMEDIATE_STATES: T.constexpr,
    SAME_POOL: T.constexpr,
):
    # TIRX_KERNEL_SKETCH_START
    state = T.match_buffer(
        state_h,
        (state_slot_stride * T.cast(batch * POOL_FACTOR, "int64"),),
        "bfloat16",
        scope="global",
    )
    intermediate = T.match_buffer(
        intermediate_h,
        (T.cast(batch * INTERMEDIATE_BATCH_STRIDE + INTERMEDIATE_DUMMY_ELEMENTS, "int64"),),
        "bfloat16",
        scope="global",
    )
    A_log = T.match_buffer(A_log_h, (NUM_V_HEADS,), "float32", scope="global")
    a = T.match_buffer(a_h, (T.cast(batch * NUM_V_HEADS, "int64"),), "bfloat16", scope="global")
    dt_bias = T.match_buffer(dt_bias_h, (NUM_V_HEADS,), "float32", scope="global")
    q = T.match_buffer(
        q_h,
        (q_batch_stride * T.cast(batch - 1, "int64") + T.cast(NUM_HEADS * K, "int64"),),
        "bfloat16",
        scope="global",
    )
    k = T.match_buffer(
        k_h,
        (k_batch_stride * T.cast(batch - 1, "int64") + T.cast(NUM_HEADS * K, "int64"),),
        "bfloat16",
        scope="global",
    )
    v = T.match_buffer(
        v_h,
        (v_batch_stride * T.cast(batch - 1, "int64") + T.cast(NUM_V_HEADS * V, "int64"),),
        "bfloat16",
        scope="global",
    )
    b_gate = T.match_buffer(
        b_h, (T.cast(batch * NUM_V_HEADS, "int64"),), "bfloat16", scope="global"
    )
    output = T.match_buffer(
        output_h, (T.cast(batch * NUM_V_HEADS * V, "int64"),), "bfloat16", scope="global"
    )
    read_indices = T.match_buffer(
        read_indices_h, (T.cast(batch, "int64"),), "int32", scope="global"
    )
    write_indices = T.match_buffer(
        write_indices_h, (T.cast(batch, "int64"),), "int32", scope="global"
    )
    state_vector = T.decl_buffer(
        (state_slot_stride * T.cast(batch * POOL_FACTOR, "int64") // ELEMS_PER_LANE,),
        "uint32x4",
        data=state.data,
        scope="global",
        align=32,
    )
    T.device_entry()

    linear_cta = T.cta_id([batch * NUM_V_HEADS * NUM_V_TILES])
    tid = T.thread_id([THREADS])
    warp_raw: T.int32 = tid // 32
    lane_in_warp: T.int32 = tid % 32
    group: T.int32 = tid // LANES_PER_GROUP
    lane: T.int32 = tid % LANES_PER_GROUP
    k_start: T.int32 = lane * ELEMS_PER_LANE

    v_tile: T.int32 = linear_cta % NUM_V_TILES
    cta_head: T.int32 = linear_cta // NUM_V_TILES
    hv: T.int32 = cta_head % NUM_V_HEADS
    n: T.int32 = cta_head // NUM_V_HEADS
    h: T.int32 = hv // (NUM_V_HEADS // NUM_HEADS)

    read_slot_raw: T.int32 = _global_load_s32(read_indices, n)

    pool = T.SMEMPool()
    smem_raw = pool.alloc((SHARED_BYTES,), "uint8", align=16)
    s_q = T.decl_buffer(
        (K,), "float32", data=smem_raw.data, scope="shared.dyn", byte_offset=0, align=16
    )
    s_k = T.decl_buffer(
        (K,), "float32", data=smem_raw.data, scope="shared.dyn", byte_offset=512, align=16
    )
    s_gb = T.decl_buffer(
        (3,), "float32", data=smem_raw.data, scope="shared.dyn", byte_offset=1024, align=16
    )
    pool.commit()

    read_slot: T.int32 = _max_s32(read_slot_raw, T.int32(0))
    write_slot: T.int32 = read_slot
    if not SAME_POOL:
        write_slot = _max_s32(_global_load_s32(write_indices, n), T.int32(0))

    read_state_base: T.int64 = T.cast(read_slot, "int64") * state_slot_stride + T.cast(
        hv * V * K, "int64"
    )
    write_state_base: T.int64 = read_state_base
    if not SAME_POOL:
        write_state_base = T.cast(write_slot, "int64") * state_slot_stride + T.cast(
            hv * V * K, "int64"
        )
    # Phase 0: with the T=1-only kq value dead, independent warps can publish
    # normalized Q, normalized K, and the scalar gates concurrently.
    if warp_raw == 0:
        member_pre: T.int32 = lane_in_warp % LANES_PER_GROUP
        k_pre: T.int32 = member_pre * ELEMS_PER_LANE
        q_bits = T.alloc_local((ELEMS_PER_LANE,), "uint16")
        r_q = T.alloc_local((ELEMS_PER_LANE,), "float32")
        q_base: T.int64 = T.cast(n, "int64") * q_batch_stride + h * K + k_pre
        _load_bf16x8_bits(q, q_base, q_bits)
        for i in T.unroll(ELEMS_PER_LANE):
            r_q[i] = _bf16_to_f32(q_bits[i])

        if USE_QK_L2NORM:
            sum_q: T.float32 = 0.0
            for i in T.unroll(ELEMS_PER_LANE):
                sum_q = _bf16_square_fma(q_bits[i], sum_q)
            for delta_index in T.unroll(4):
                delta: T.int32 = T.shift_right(T.int32(8), delta_index)
                sum_q = _add_f32(
                    sum_q, T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), sum_q, delta, 32)
                )
            q_factor: T.float32 = _mul_f32(
                _rsqrt_f32(_add_f32(sum_q, T.float32(1.0e-6))), T.float32(SCALE)
            )
            for i in T.unroll(ELEMS_PER_LANE):
                r_q[i] = _mul_f32(r_q[i], q_factor)
        else:
            for i in T.unroll(ELEMS_PER_LANE):
                r_q[i] = _mul_f32(r_q[i], T.float32(SCALE))

        for i in T.unroll(ELEMS_PER_LANE):
            _shared_store_f32(s_q, k_pre + i, r_q[i])

    if warp_raw == 1:
        member_pre: T.int32 = lane_in_warp % LANES_PER_GROUP
        k_pre: T.int32 = member_pre * ELEMS_PER_LANE
        k_bits = T.alloc_local((ELEMS_PER_LANE,), "uint16")
        r_k = T.alloc_local((ELEMS_PER_LANE,), "float32")
        k_base: T.int64 = T.cast(n, "int64") * k_batch_stride + h * K + k_pre
        _load_bf16x8_bits(k, k_base, k_bits)
        for i in T.unroll(ELEMS_PER_LANE):
            r_k[i] = _bf16_to_f32(k_bits[i])

        if USE_QK_L2NORM:
            sum_k: T.float32 = 0.0
            for i in T.unroll(ELEMS_PER_LANE):
                sum_k = _bf16_square_fma(k_bits[i], sum_k)
            for delta_index in T.unroll(4):
                delta: T.int32 = T.shift_right(T.int32(8), delta_index)
                sum_k = _add_f32(
                    sum_k, T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), sum_k, delta, 32)
                )
            k_factor: T.float32 = _rsqrt_f32(_add_f32(sum_k, T.float32(1.0e-6)))
            for i in T.unroll(ELEMS_PER_LANE):
                r_k[i] = _mul_f32(r_k[i], k_factor)

        for i in T.unroll(ELEMS_PER_LANE):
            _shared_store_f32(s_k, k_pre + i, r_k[i])

    if warp_raw == 2:
        A_value: T.float32 = _global_load_f32(A_log, hv)
        dt_value: T.float32 = _global_load_f32(dt_bias, hv)
        a_bits: T.uint16 = _global_load_u16(a, n * NUM_V_HEADS + hv)
        x_value: T.float32 = _mixed_add_bf16_f32(a_bits, dt_value)
        softplus_exp: T.float32 = _exp2_f32(_mul_f32(x_value, T.float32(LOG2_E)))
        softplus_log2: T.float32 = _log2_f32(_add_f32(T.float32(1.0), softplus_exp))
        softplus_value: T.float32 = _mul_f32(softplus_log2, T.float32(LN_2))
        use_softplus: T.float32 = T.if_then_else(
            x_value <= T.float32(20.0), T.float32(1.0), T.float32(0.0)
        )
        direct_weight: T.float32 = _sub_f32(T.float32(1.0), use_softplus)
        softplus_x: T.float32 = _fma_f32(
            softplus_value, use_softplus, _mul_f32(x_value, direct_weight)
        )
        exp_A: T.float32 = _exp2_f32(_mul_f32(A_value, T.float32(LOG2_E)))
        gate_exponent: T.float32 = _mul_f32(_sub_f32(T.float32(0.0), exp_A), softplus_x)

        if lane_in_warp == 0:
            g: T.float32 = _exp2_f32(_mul_f32(gate_exponent, T.float32(LOG2_E)))
            _shared_store_f32(s_gb, 0, g)

    if warp_raw == 3 and lane_in_warp == 0:
        b_bits: T.uint16 = _global_load_u16(b_gate, n * NUM_V_HEADS + hv)
        b_value: T.float32 = _bf16_to_f32(b_bits)
        exp_neg_b: T.float32 = _exp2_f32(_mul_f32(b_value, T.float32(-LOG2_E)))
        beta: T.float32 = _rcp_f32(_add_f32(T.float32(1.0), exp_neg_b))
        _shared_store_f32(s_gb, 1, beta)

    T.cuda.cta_sync()

    # Phase 1: eight independent 16-lane groups.  Each source constexpr
    # iteration is physically unrolled and carries four state rows in registers.
    # The source compiler retains the same-token Q/K/g/beta shared values across
    # every unrolled V-row body, so materialize that physical register lifetime
    # explicitly: inline PTX shared loads are opaque to nvcc's CSE.
    g_value: T.float32 = _shared_load_f32(s_gb, 0)
    beta_value: T.float32 = _shared_load_f32(s_gb, 1)
    r_k_main = T.alloc_local((ELEMS_PER_LANE,), "float32")
    r_q_main = T.alloc_local((ELEMS_PER_LANE,), "float32")
    for i in T.unroll(ELEMS_PER_LANE):
        r_k_main[i] = _shared_load_f32(s_k, k_start + i)
        r_q_main[i] = _shared_load_f32(s_q, k_start + i)

    for iter_index in T.unroll(ITERS_PER_GROUP):
        v_base: T.int32 = v_tile * TILE_V + group * ROWS_PER_GROUP + iter_index * ILP_ROWS
        read_state_offset: T.int64 = read_state_base + T.cast(v_base * K + k_start, "int64")
        r_h = T.alloc_local((ILP_ROWS * ELEMS_PER_LANE,), "float32")
        state_bits = T.alloc_local((ILP_ROWS * ELEMS_PER_LANE,), "uint16", align=16)
        for row in T.unroll(ILP_ROWS):
            if TILE_V == 64:
                _load_state_bf16x8_vector_buffer(
                    state_vector, read_state_offset + row * K, r_h, row * ELEMS_PER_LANE
                )
            else:
                _load_state_bf16x8(state, read_state_offset + row * K, r_h, row * ELEMS_PER_LANE)

        sums = T.alloc_local((ILP_ROWS,), "float32")
        for row in T.unroll(ILP_ROWS):
            sums[row] = T.float32(0.0)

        for pair in T.unroll(ELEMS_PER_LANE // 2):
            for row in T.unroll(ILP_ROWS):
                pair_value: T.uint64 = _packed_fma(
                    r_h[row * ELEMS_PER_LANE + pair * 2],
                    r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                    g_value,
                    g_value,
                    T.float32(0.0),
                    T.float32(0.0),
                )
                r_h[row * ELEMS_PER_LANE + pair * 2] = T.cuda.float2_x(pair_value)
                r_h[row * ELEMS_PER_LANE + pair * 2 + 1] = T.cuda.float2_y(pair_value)
                sums[row] = _fma_f32(
                    r_h[row * ELEMS_PER_LANE + pair * 2], r_k_main[pair * 2], sums[row]
                )
                sums[row] = _fma_f32(
                    r_h[row * ELEMS_PER_LANE + pair * 2 + 1], r_k_main[pair * 2 + 1], sums[row]
                )

        for delta_index in T.unroll(4):
            delta: T.int32 = T.shift_right(T.int32(8), delta_index)
            for row in T.unroll(ILP_ROWS):
                sums[row] = _add_f32(
                    sums[row], T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), sums[row], delta, 32)
                )

        values = T.alloc_local((ILP_ROWS,), "float32")
        value_bits = T.alloc_local((ILP_ROWS,), "uint16")
        v_input_base: T.int64 = T.cast(n, "int64") * v_batch_stride + hv * V + v_base
        if TILE_V == 128:
            _load_bf16x4_bits(v, v_input_base, value_bits)
        else:
            for row in T.unroll(ILP_ROWS):
                value_bits[row] = _global_load_u16(v, v_input_base + row)
        for row in T.unroll(ILP_ROWS):
            values[row] = _mul_f32(_mixed_sub_bf16_f32(value_bits[row], sums[row]), beta_value)

        for row in T.unroll(ILP_ROWS):
            sums[row] = T.float32(0.0)
        for pair in T.unroll(ELEMS_PER_LANE // 2):
            q0: T.float32 = r_q_main[pair * 2]
            q1: T.float32 = r_q_main[pair * 2 + 1]
            for row in T.unroll(ILP_ROWS):
                pair_value: T.uint64 = _packed_fma(
                    r_k_main[pair * 2],
                    r_k_main[pair * 2 + 1],
                    values[row],
                    values[row],
                    r_h[row * ELEMS_PER_LANE + pair * 2],
                    r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                )
                r_h[row * ELEMS_PER_LANE + pair * 2] = T.cuda.float2_x(pair_value)
                r_h[row * ELEMS_PER_LANE + pair * 2 + 1] = T.cuda.float2_y(pair_value)
                sums[row] = _fma_f32(r_h[row * ELEMS_PER_LANE + pair * 2], q0, sums[row])
                sums[row] = _fma_f32(r_h[row * ELEMS_PER_LANE + pair * 2 + 1], q1, sums[row])

        for delta_index in T.unroll(4):
            delta: T.int32 = T.shift_right(T.int32(8), delta_index)
            for row in T.unroll(ILP_ROWS):
                sums[row] = _add_f32(
                    sums[row], T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), sums[row], delta, 32)
                )

        if lane == 0:
            output_base: T.int64 = T.cast((n * NUM_V_HEADS + hv) * V + v_base, "int64")
            for row in T.unroll(ILP_ROWS):
                T.evaluate(
                    T.ptx.st.global_.b16(
                        output.ptr_to([output_base + row]), _f32_to_bf16(sums[row])
                    )
                )

        if CACHE_INTERMEDIATE_STATES:
            intermediate_base: T.int64 = T.cast(
                (n * NUM_V_HEADS + hv) * V * K + v_base * K + k_start, "int64"
            )
            for row in T.unroll(ILP_ROWS):
                _store_state_f32x8(
                    intermediate, intermediate_base + row * K, r_h, row * ELEMS_PER_LANE
                )

        if not DISABLE_STATE_UPDATE and not CACHE_INTERMEDIATE_STATES:
            write_state_offset: T.int64 = (
                read_state_offset
                if SAME_POOL
                else write_state_base + T.cast(v_base * K + k_start, "int64")
            )
            for row in T.unroll(ILP_ROWS):
                if TILE_V == 64:
                    _store_state_f32x8_vector_buffer(
                        state_vector, write_state_offset + row * K, r_h, row * ELEMS_PER_LANE
                    )
                else:
                    _store_state_f32x8(
                        state, write_state_offset + row * K, r_h, row * ELEMS_PER_LANE
                    )


def get_kernel(**kwargs: Any):
    """Return the source-specialized TIRx PrimFunc."""
    tile_v = int(kwargs["tile_v"])
    same_pool = bool(kwargs.get("same_pool", True))
    kernel = _gdn_decode_bf16_wide_vec_t1.specialize(
        NUM_HEADS=kwargs["num_heads"],
        NUM_V_HEADS=kwargs["num_v_heads"],
        TILE_V=tile_v,
        NUM_V_TILES=V // tile_v,
        ROWS_PER_GROUP=tile_v // NUM_GROUPS,
        ITERS_PER_GROUP=(tile_v // NUM_GROUPS) // ILP_ROWS,
        POOL_FACTOR=1 if same_pool else 2,
        INTERMEDIATE_BATCH_STRIDE=(
            int(kwargs["num_v_heads"]) * V * K
            if kwargs.get("cache_intermediate_states", False)
            else 0
        ),
        INTERMEDIATE_DUMMY_ELEMENTS=(0 if kwargs.get("cache_intermediate_states", False) else 1),
        USE_QK_L2NORM=kwargs.get("use_qk_l2norm", True),
        DISABLE_STATE_UPDATE=kwargs.get("disable_state_update", False),
        CACHE_INTERMEDIATE_STATES=kwargs.get("cache_intermediate_states", False),
        SAME_POOL=same_pool,
    )
    return kernel.with_attr(
        "tirx.kernel_launch_params", ["blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory"]
    )


def _require_supported_config(config: dict[str, Any]) -> None:
    batch = int(config["batch"])
    num_heads = int(config["num_heads"])
    num_v_heads = int(config["num_v_heads"])
    tile_v = int(config["tile_v"])
    if batch < 1:
        raise ValueError("batch must be positive")
    if (num_heads, num_v_heads) not in _TP_HEADS:
        raise ValueError(f"unsupported Qwen3-Next TP head pair {(num_heads, num_v_heads)}")
    if num_v_heads % num_heads:
        raise ValueError("num_v_heads must be divisible by num_heads")
    if tile_v not in (64, 128) or V % tile_v:
        raise ValueError("wide-vector T1 port requires tile_v in {64, 128}")
    if (tile_v // NUM_GROUPS) % ILP_ROWS:
        raise ValueError("tile_v is incompatible with the 8-group, ILP4 mapping")


def _allocate_pool(
    pool_slots: int,
    num_v_heads: int,
    *,
    padded: bool,
    device: str | torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    padding_heads = 1 if padded else 0
    backing = (
        torch.randn(
            (pool_slots, num_v_heads + padding_heads, V, K),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.1
    )
    return backing[:, :num_v_heads], backing


def _clone_pool_layout(pool: torch.Tensor, *, padded: bool) -> tuple[torch.Tensor, torch.Tensor]:
    pool_slots, num_v_heads = pool.shape[:2]
    padding_heads = 1 if padded else 0
    backing = torch.empty(
        (pool_slots, num_v_heads + padding_heads, V, K), dtype=pool.dtype, device=pool.device
    )
    view = backing[:, :num_v_heads]
    view.copy_(pool)
    return view, backing


def _make_qkv(
    batch: int,
    num_heads: int,
    num_v_heads: int,
    *,
    packed: bool,
    device: str | torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not packed:
        q = (
            torch.randn(
                (batch, SEQ_LEN, num_heads, K),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            * 0.05
        )
        k = torch.randn_like(q)
        v = (
            torch.randn(
                (batch, SEQ_LEN, num_v_heads, V),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            * 0.05
        )
        return q, k, v, None

    q_elements = num_heads * K
    k_elements = num_heads * K
    v_elements = num_v_heads * V
    row_elements = q_elements + k_elements + v_elements + 128
    backing = (
        torch.randn((batch, row_elements), dtype=torch.bfloat16, device=device, generator=generator)
        * 0.05
    )
    q = backing.as_strided((batch, SEQ_LEN, num_heads, K), (row_elements, row_elements, K, 1), 0)
    k = backing.as_strided(
        (batch, SEQ_LEN, num_heads, K), (row_elements, row_elements, K, 1), q_elements
    )
    v = backing.as_strided(
        (batch, SEQ_LEN, num_v_heads, V),
        (row_elements, row_elements, V, 1),
        q_elements + k_elements,
    )
    return q, k, v, backing


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Create deterministic independent mutable TIRx and FlashInfer cases."""
    config = dict(kwargs)
    _require_supported_config(config)
    device = config.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for BF16 wide-vector GDN decode")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(f"BF16 wide-vector GDN decode requires SM100, got {capability}")

    batch = int(config["batch"])
    num_heads = int(config["num_heads"])
    num_v_heads = int(config["num_v_heads"])
    same_pool = bool(config.get("same_pool", True))
    padded_pool = bool(config.get("padded_pool", False))
    negative_read = bool(config.get("negative_read_index", False))
    negative_write = bool(config.get("negative_write_index", False))
    packed_qkv = bool(config.get("packed_qkv", False))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.get("seed", 0)) + 20260811)

    extra_null_slot = negative_read or negative_write
    pool_slots = batch * (1 if same_pool else 2) + int(extra_null_slot)
    initial_pool, initial_backing = _allocate_pool(
        pool_slots, num_v_heads, padded=padded_pool, device=device, generator=generator
    )
    tirx_state, tirx_state_backing = _clone_pool_layout(initial_pool, padded=padded_pool)
    source_state, source_state_backing = _clone_pool_layout(initial_pool, padded=padded_pool)

    slot_offset = 1 if extra_null_slot else 0
    read_indices = torch.arange(batch, dtype=torch.int32, device=device) + slot_offset
    if negative_read:
        read_indices[-1] = -1
    if same_pool:
        write_indices = read_indices
    else:
        write_indices = torch.arange(batch, dtype=torch.int32, device=device) + slot_offset + batch
        if negative_write:
            write_indices[-1] = -1

    q, k, v, qkv_backing = _make_qkv(
        batch, num_heads, num_v_heads, packed=packed_qkv, device=device, generator=generator
    )
    A_log = (
        torch.randn((num_v_heads,), dtype=torch.float32, device=device, generator=generator) * 0.1
    )
    dt_bias = (
        torch.randn((num_v_heads,), dtype=torch.float32, device=device, generator=generator) * 0.1
    )
    a = (
        torch.randn(
            (batch, SEQ_LEN, num_v_heads), dtype=torch.bfloat16, device=device, generator=generator
        )
        * 0.05
    )
    b_gate = torch.randn_like(a) * 0.05

    cache = bool(config.get("cache_intermediate_states", False))
    cache_shape = (batch, SEQ_LEN, num_v_heads, V, K)
    tirx_intermediate = (
        torch.empty(cache_shape, dtype=torch.bfloat16, device=device)
        if cache
        else torch.zeros((1,), dtype=torch.bfloat16, device=device)
    )
    source_intermediate = (
        torch.empty(cache_shape, dtype=torch.bfloat16, device=device)
        if cache
        else torch.zeros((1,), dtype=torch.bfloat16, device=device)
    )

    return {
        "config": config,
        "initial_pool": initial_pool.clone(),
        "initial_backing": initial_backing,
        "tirx_state": tirx_state,
        "tirx_state_backing": tirx_state_backing,
        "source_state": source_state,
        "source_state_backing": source_state_backing,
        "read_indices": read_indices,
        "write_indices": write_indices,
        "q": q,
        "k": k,
        "v": v,
        "qkv_backing": qkv_backing,
        "qkv_snapshot": qkv_backing.clone() if qkv_backing is not None else None,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "a": a,
        "b_gate": b_gate,
        "tirx_output": torch.empty(
            (batch, SEQ_LEN, num_v_heads, V), dtype=torch.bfloat16, device=device
        ),
        "source_output": torch.empty(
            (batch, SEQ_LEN, num_v_heads, V), dtype=torch.bfloat16, device=device
        ),
        "tirx_intermediate": tirx_intermediate,
        "source_intermediate": source_intermediate,
    }


@functools.cache
def _compile_tirx(
    num_heads: int,
    num_v_heads: int,
    tile_v: int,
    use_qk_l2norm: bool,
    disable_state_update: bool,
    cache_intermediate_states: bool,
    same_pool: bool,
):
    from tirx_kernels.runner import compile_kernel

    return compile_kernel(
        get_kernel(
            batch=1,
            num_heads=num_heads,
            num_v_heads=num_v_heads,
            tile_v=tile_v,
            use_qk_l2norm=use_qk_l2norm,
            disable_state_update=disable_state_update,
            cache_intermediate_states=cache_intermediate_states,
            same_pool=same_pool,
        )
    )


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    config = case["config"]
    state = case["tirx_state"]
    q = case["q"]
    k = case["k"]
    v = case["v"]
    batch = int(config["batch"])
    pool_factor = 1 if bool(config.get("same_pool", True)) else 2

    def storage_span(tensor: torch.Tensor, elements: int) -> torch.Tensor:
        return tensor.as_strided((elements,), (1,))

    return (
        # Negative-index cases own one extra physical null slot, but the kernel
        # ABI is deliberately the production B (or 2B) pool span.  All clamped
        # and shifted indices remain inside that declared span.
        storage_span(state, int(state.stride(0)) * batch * pool_factor),
        storage_span(case["tirx_intermediate"], case["tirx_intermediate"].numel()),
        case["A_log"],
        case["a"].reshape(-1),
        case["dt_bias"],
        storage_span(q, int(q.stride(0)) * (batch - 1) + int(q.shape[2] * q.shape[3])),
        storage_span(k, int(k.stride(0)) * (batch - 1) + int(k.shape[2] * k.shape[3])),
        storage_span(v, int(v.stride(0)) * (batch - 1) + int(v.shape[2] * v.shape[3])),
        case["b_gate"].reshape(-1),
        case["tirx_output"].reshape(-1),
        case["read_indices"],
        case["write_indices"],
        int(state.stride(0)),
        int(q.stride(0)),
        int(k.stride(0)),
        int(v.stride(0)),
        batch,
    )


def _tirx_executable(case: dict[str, Any]):
    config = case["config"]
    return _compile_tirx(
        int(config["num_heads"]),
        int(config["num_v_heads"]),
        int(config["tile_v"]),
        bool(config.get("use_qk_l2norm", True)),
        bool(config.get("disable_state_update", False)),
        bool(config.get("cache_intermediate_states", False)),
        bool(config.get("same_pool", True)),
    )


def _run_reference(case: dict[str, Any]) -> torch.Tensor:
    from tirx_kernels.flashinfer._gdn_reference import gated_delta_rule_decode

    config = case["config"]
    cache = bool(config.get("cache_intermediate_states", False))
    same_pool = bool(config.get("same_pool", True))
    return gated_delta_rule_decode(
        A_log=case["A_log"],
        a=case["a"],
        dt_bias=case["dt_bias"],
        q=case["q"],
        k=case["k"],
        v=case["v"],
        b=case["b_gate"],
        state_pool=case["source_state"],
        read_indices=case["read_indices"],
        write_indices=case["read_indices"] if same_pool else case["write_indices"],
        intermediate_states=case["source_intermediate"] if cache else None,
        disable_state_update=bool(config.get("disable_state_update", False)),
        use_qk_l2norm=bool(config.get("use_qk_l2norm", True)),
        scale=SCALE,
        output=case["source_output"],
    )


def _assert_untouched_slots(case: dict[str, Any], state_key: str) -> None:
    config = case["config"]
    pool = case[state_key]
    initial = case["initial_pool"]
    update_pool = not bool(config.get("disable_state_update", False)) and not bool(
        config.get("cache_intermediate_states", False)
    )
    touched = torch.zeros((pool.shape[0],), dtype=torch.bool, device=pool.device)
    if update_pool:
        indices = (
            case["read_indices"] if bool(config.get("same_pool", True)) else case["write_indices"]
        )
        touched[torch.clamp(indices, min=0).to(torch.int64)] = True
    torch.testing.assert_close(pool[~touched], initial[~touched], atol=0.0, rtol=0.0)


def _assert_case_close(case: dict[str, Any]) -> None:
    torch.testing.assert_close(
        case["tirx_output"].float(), case["source_output"].float(), atol=1.0e-3, rtol=5.0e-3
    )
    torch.testing.assert_close(
        case["tirx_state"].float(), case["source_state"].float(), atol=2.0e-2, rtol=1.0e-2
    )
    _assert_untouched_slots(case, "tirx_state")
    _assert_untouched_slots(case, "source_state")
    if bool(case["config"].get("cache_intermediate_states", False)):
        torch.testing.assert_close(
            case["tirx_intermediate"].float(),
            case["source_intermediate"].float(),
            atol=2.0e-2,
            rtol=1.0e-2,
        )
    if case["qkv_backing"] is not None:
        torch.testing.assert_close(case["qkv_backing"], case["qkv_snapshot"], atol=0.0, rtol=0.0)


def run_test(**kwargs: Any) -> None:
    case = prepare_data(**kwargs)
    executable = _tirx_executable(case)
    executable(*_tirx_args(case))
    torch.cuda.synchronize()
    _run_reference(case)
    torch.cuda.synchronize()
    _assert_case_close(case)


def prepare_bench(**kwargs: Any):
    """Compile the selected wide-vector specialization before CUDA setup."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    config = dict(kwargs)
    _require_supported_config(config)
    executable = _compile_tirx(
        int(config["num_heads"]),
        int(config["num_v_heads"]),
        int(config["tile_v"]),
        bool(config.get("use_qk_l2norm", True)),
        bool(config.get("disable_state_update", False)),
        bool(config.get("cache_intermediate_states", False)),
        bool(config.get("same_pool", True)),
    )
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
    kwargs = {**prepared["config"], **kwargs}
    case = prepare_data(**kwargs)
    executable = prepared["executable"]
    args = _tirx_args(case)

    def source_builder():
        from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_t1_wide_vec

        config = case["config"]

        def launch():
            gated_delta_rule_t1_wide_vec(
                A_log=case["A_log"],
                a=case["a"],
                dt_bias=case["dt_bias"],
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=case["q"],
                k=case["k"],
                v=case["v"],
                b=case["b_gate"],
                initial_state_source=case["source_state"],
                initial_state_indices=case["read_indices"],
                output_state_indices=(
                    case["read_indices"]
                    if bool(config.get("same_pool", True))
                    else case["write_indices"]
                ),
                intermediate_states_buffer=(
                    case["source_intermediate"]
                    if bool(config.get("cache_intermediate_states", False))
                    else None
                ),
                disable_state_update=bool(config.get("disable_state_update", False)),
                use_qk_l2norm_in_kernel=bool(config.get("use_qk_l2norm", True)),
                scale=SCALE,
                output=case["source_output"],
                tile_v=int(config["tile_v"]),
            )

        executable(*args)
        launch()
        torch.cuda.synchronize()
        _assert_case_close(case)
        for _ in range(2):
            launch()
        torch.cuda.synchronize()
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
    "prepare_data",
    "run_bench",
    "run_test",
]
