# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM100 BF16 wide-vector multi-token GDN decode kernel.

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
    "name": "gdn_decode_bf16_wide_vec_mtp",
    "category": "flashinfer",
    "compute_capability": 10,
}


K = 128
V = 128
THREADS = 128
NUM_WARPS = 4
NUM_GROUPS = 8
LANES_PER_GROUP = 16
ELEMS_PER_LANE = 8
ILP_ROWS = 4
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


def _global_store_u16_pred(buffer, index, value, pred):
    T.evaluate(T.ptx.st.global_.b16(buffer.ptr_to([index]), value, pred=pred))


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


@T.inline
def _shared_load_f32x2(buffer, index, values):
    words = T.alloc_local((2,), "uint32", align=8)
    T.evaluate(T.ptx.ld.shared.v2.b32(words[0], words[1], buffer.ptr_to([index])))
    values[0] = T.reinterpret("float32", words[0])
    values[1] = T.reinterpret("float32", words[1])


def _shared_store_f32(buffer, index, value):
    T.evaluate(T.ptx.st.shared.b32(buffer.ptr_to([index]), T.reinterpret("uint32", value)))


def _shared_store_f32x2(buffer, index, value0, value1):
    T.evaluate(
        T.ptx.st.shared.v2.b32(
            buffer.ptr_to([index]), T.reinterpret("uint32", value0), T.reinterpret("uint32", value1)
        )
    )


@T.inline
def _load_state_bf16x32(buffer, index, values):
    words = T.alloc_local((ILP_ROWS * 4,), "uint32", align=16)
    for row in T.unroll(ILP_ROWS):
        word_offset: T.int32 = row * 4
        T.evaluate(
            T.ptx["ld.global.L1::evict_first.v4.b32"](
                words[word_offset],
                words[word_offset + 1],
                words[word_offset + 2],
                words[word_offset + 3],
                buffer.ptr_to([index + row * K]),
            )
        )
    for pair in T.unroll(4):
        for row in T.unroll(ILP_ROWS):
            word: T.uint32 = words[row * 4 + pair]
            values[row * ELEMS_PER_LANE + pair * 2] = T.cuda.uint_as_float(
                T.shift_left(word, T.uint32(16))
            )
            values[row * ELEMS_PER_LANE + pair * 2 + 1] = T.cuda.uint_as_float(
                T.bitwise_and(word, T.uint32(0xFFFF0000))
            )


@T.inline
def _store_state_f32x32(buffer, index, values):
    words = T.alloc_local((ILP_ROWS * 4,), "uint32", align=16)
    for pair in T.unroll(4):
        for row in T.unroll(ILP_ROWS):
            words[row * 4 + pair] = T.cuda.float22bfloat162_rn(
                values[row * ELEMS_PER_LANE + pair * 2], values[row * ELEMS_PER_LANE + pair * 2 + 1]
            )
    for row in T.unroll(ILP_ROWS):
        word_offset: T.int32 = row * 4
        T.evaluate(
            T.ptx.st.global_.v4.b32(
                buffer.ptr_to([index + row * K]),
                words[word_offset],
                words[word_offset + 1],
                words[word_offset + 2],
                words[word_offset + 3],
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


def _make_warp_uniform(value):
    return T.cuda._shfl_sync(T.uint32(0xFFFFFFFF), value, 0, 32)


_SEQ_LENS = (2, 4, 8)
_HEAD_CONFIGS = ((16, 32), (8, 16), (4, 8), (2, 4))
_BATCH_SIZES = (1, 4, 8, 16, 32, 64, 128, 256, 512)


def _select_tile_v(batch: int, num_v_heads: int) -> int | None:
    work_units = batch * num_v_heads
    if work_units >= 1024:
        return 128
    if work_units >= 512:
        return 64
    if work_units >= 128:
        return 32
    return None


def _production_case(seq_len: int, batch: int, num_heads: int, num_v_heads: int) -> dict[str, Any]:
    tile_v = _select_tile_v(batch, num_v_heads)
    if tile_v is None:
        raise ValueError("wide-vector MTP requires batch * num_v_heads >= 128")
    return {
        "label": f"t{seq_len}_b{batch}_h{num_heads}_hv{num_v_heads}_tv{tile_v}",
        "seq_len": seq_len,
        "batch": batch,
        "num_heads": num_heads,
        "num_v_heads": num_v_heads,
        "tile_v": tile_v,
        "use_qk_l2norm": True,
        "disable_state_update": True,
        "cache_intermediate_states": True,
        "disable_output": False,
    }


BENCH_CONFIGS = [
    _production_case(seq_len, batch, num_heads, num_v_heads)
    for seq_len in _SEQ_LENS
    for num_heads, num_v_heads in _HEAD_CONFIGS
    for batch in _BATCH_SIZES
    if _select_tile_v(batch, num_v_heads) is not None
]


def _correctness_case(
    label: str,
    *,
    seq_len: int = 4,
    batch: int = 2,
    num_heads: int = 16,
    num_v_heads: int = 32,
    tile_v: int = 32,
    **kwargs: Any,
) -> dict[str, Any]:
    return {
        "label": label,
        "seq_len": seq_len,
        "batch": batch,
        "num_heads": num_heads,
        "num_v_heads": num_v_heads,
        "tile_v": tile_v,
        **kwargs,
    }


CONFIGS = [
    _correctness_case("t2_tv32_base", seq_len=2, batch=1),
    _correctness_case("t3_tv64_tail", seq_len=3, batch=1, num_heads=8, num_v_heads=16, tile_v=64),
    _correctness_case("t5_tv128_tail", seq_len=5, batch=1, num_heads=4, num_v_heads=8, tile_v=128),
    _correctness_case("t7_tv32_tail", seq_len=7, num_heads=2, num_v_heads=4),
    _correctness_case("t8_tv128_base", seq_len=8, tile_v=128),
    _correctness_case("t4_l2off", use_qk_l2norm=False),
    _correctness_case("t4_split", same_pool=False),
    _correctness_case("t6_split_l2off", seq_len=6, tile_v=64, same_pool=False, use_qk_l2norm=False),
    _correctness_case("t4_padded", padded_pool=True),
    _correctness_case("t4_padded_split", padded_pool=True, same_pool=False),
    _correctness_case("t3_packed_qkv", seq_len=3, packed_qkv=True),
    _correctness_case("t4_negative_read", negative_read_index=True),
    _correctness_case("t4_negative_write", same_pool=False, negative_write_index=True),
    _correctness_case("t4_disable_update", disable_state_update=True),
    _correctness_case("t3_disable_output", seq_len=3, disable_output=True),
    _correctness_case("t4_cache", cache_intermediate_states=True),
    _correctness_case("t4_cache_split", cache_intermediate_states=True, same_pool=False),
    _correctness_case(
        "t3_cache_no_output", seq_len=3, cache_intermediate_states=True, disable_output=True
    ),
    _correctness_case("t4_recovery1", recovery_steps=1),
    _correctness_case(
        "t8_recovery4_split", seq_len=8, tile_v=128, recovery_steps=4, same_pool=False
    ),
    _correctness_case("t4_recovery_all", recovery_steps=4),
    _correctness_case(
        "t8_accepted_fused",
        seq_len=8,
        tile_v=64,
        per_request_accepted_steps=True,
        accepted_steps=(1, 6),
    ),
    _correctness_case(
        "t8_accepted_fused_split",
        seq_len=8,
        tile_v=128,
        same_pool=False,
        per_request_accepted_steps=True,
        accepted_steps=(0, 7),
    ),
    _correctness_case(
        "t6_accepted_disable_update",
        seq_len=6,
        disable_state_update=True,
        per_request_accepted_steps=True,
        accepted_steps=(1, 5),
    ),
    _correctness_case(
        "t5_accepted_cache",
        seq_len=5,
        cache_intermediate_states=True,
        per_request_accepted_steps=True,
        accepted_steps=(0, 4),
    ),
    _correctness_case(
        "t7_accepted_no_output",
        seq_len=7,
        disable_output=True,
        per_request_accepted_steps=True,
        accepted_steps=(2, 6),
    ),
    _correctness_case("t4_scatter_flat", per_token_pool_scatter=True),
    _correctness_case("t4_scatter_flat_split", same_pool=False, per_token_pool_scatter=True),
    _correctness_case("t4_scatter_padded", padded_pool=True, per_token_pool_scatter=True),
    _correctness_case(
        "t4_scatter_padded_split", padded_pool=True, same_pool=False, per_token_pool_scatter=True
    ),
    _correctness_case(
        "t8_accepted_scatter",
        seq_len=8,
        tile_v=64,
        per_request_accepted_steps=True,
        accepted_steps=(2, 7),
        per_token_pool_scatter=True,
    ),
    _correctness_case(
        "threshold_tv32",
        seq_len=2,
        batch=4,
        tile_v=32,
        disable_state_update=True,
        cache_intermediate_states=True,
    ),
    _correctness_case(
        "threshold_tv64",
        seq_len=4,
        batch=16,
        tile_v=64,
        disable_state_update=True,
        cache_intermediate_states=True,
    ),
    _correctness_case(
        "threshold_tv128",
        seq_len=8,
        batch=32,
        tile_v=128,
        disable_state_update=True,
        cache_intermediate_states=True,
    ),
]

assert len(BENCH_CONFIGS) == 78


@T.jit
def _gdn_decode_bf16_wide_vec_mtp(
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
    accepted_steps_h: T.handle,
    ssm_state_indices_h: T.handle,
    state_slot_stride: T.int64,
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
    ROWS_PER_GROUP: T.constexpr,
    ITERS_PER_GROUP: T.constexpr,
    POOL_FACTOR: T.constexpr,
    INTERMEDIATE_BATCH_STRIDE: T.constexpr,
    INTERMEDIATE_DUMMY_ELEMENTS: T.constexpr,
    ACCEPTED_BATCH_STRIDE: T.constexpr,
    ACCEPTED_DUMMY_ELEMENTS: T.constexpr,
    SSM_BATCH_STRIDE: T.constexpr,
    SSM_DUMMY_ELEMENTS: T.constexpr,
    SHARED_BYTES: T.constexpr,
    S_K_BYTE_OFFSET: T.constexpr,
    S_GB_BYTE_OFFSET: T.constexpr,
    USE_QK_L2NORM: T.constexpr,
    DISABLE_STATE_UPDATE: T.constexpr,
    CACHE_INTERMEDIATE_STATES: T.constexpr,
    SAME_POOL: T.constexpr,
    DISABLE_OUTPUT: T.constexpr,
    RECOVERY_STEPS: T.constexpr,
    PER_REQUEST_ACCEPTED_STEPS: T.constexpr,
    PER_TOKEN_POOL_SCATTER: T.constexpr,
    PER_TOKEN_POOL_SCATTER_FLAT: T.constexpr,
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
    accepted_steps = T.match_buffer(
        accepted_steps_h,
        (T.cast(batch * ACCEPTED_BATCH_STRIDE + ACCEPTED_DUMMY_ELEMENTS, "int64"),),
        "int32",
        scope="global",
    )
    ssm_state_indices = T.match_buffer(
        ssm_state_indices_h,
        (T.cast(batch * SSM_BATCH_STRIDE + SSM_DUMMY_ELEMENTS, "int64"),),
        "int32",
        scope="global",
    )
    T.device_entry()

    linear_cta = T.cta_id([batch * NUM_V_HEADS * NUM_V_TILES])
    tid = T.thread_id([THREADS])
    warp_raw: T.int32 = tid // 32
    warp: T.int32 = _make_warp_uniform(warp_raw)
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
    s_gb = T.decl_buffer(
        (SEQ_LEN * 2,),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=S_GB_BYTE_OFFSET,
        align=16,
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

    accepted: T.int32 = T.int32(SEQ_LEN - 1)
    if PER_REQUEST_ACCEPTED_STEPS:
        accepted = _global_load_s32(accepted_steps, n)

    phase_a_bound: T.int32 = T.int32(RECOVERY_STEPS)
    phase_b_begin: T.int32 = T.int32(RECOVERY_STEPS)
    phase_b_bound: T.int32 = T.int32(SEQ_LEN - RECOVERY_STEPS)
    if (
        PER_REQUEST_ACCEPTED_STEPS
        and not DISABLE_OUTPUT
        and not DISABLE_STATE_UPDATE
        and not PER_TOKEN_POOL_SCATTER
    ):
        phase_a_bound = accepted + T.int32(1)
        phase_b_begin = accepted + T.int32(1)
        phase_b_bound = T.int32(SEQ_LEN) - accepted - T.int32(1)
    elif PER_REQUEST_ACCEPTED_STEPS:
        phase_b_bound = accepted + T.int32(1 - RECOVERY_STEPS)

    prefetch_words = T.alloc_local((ILP_ROWS * 4,), "uint32", align=16)
    if TILE_V == 32:
        pre_v_base: T.int32 = v_tile * TILE_V + group * ROWS_PER_GROUP
        for row in T.unroll(ILP_ROWS):
            word_offset: T.int32 = row * 4
            prefetch_offset: T.int64 = read_state_base + T.cast(
                (pre_v_base + row) * K + k_start, "int64"
            )
            T.evaluate(
                T.ptx["ld.global.L1::evict_first.v4.b32"](
                    prefetch_words[word_offset],
                    prefetch_words[word_offset + 1],
                    prefetch_words[word_offset + 2],
                    prefetch_words[word_offset + 3],
                    state.ptr_to([prefetch_offset]),
                )
            )

    for pass_index in T.unroll((SEQ_LEN + NUM_WARPS - 1) // NUM_WARPS):
        t_pre: T.int32 = pass_index * NUM_WARPS + warp
        member_pre: T.int32 = lane_in_warp % LANES_PER_GROUP
        k_pre: T.int32 = member_pre * ELEMS_PER_LANE
        if t_pre < SEQ_LEN:
            q_bits = T.alloc_local((ELEMS_PER_LANE,), "uint16")
            k_bits = T.alloc_local((ELEMS_PER_LANE,), "uint16")
            r_q = T.alloc_local((ELEMS_PER_LANE,), "float32")
            r_k = T.alloc_local((ELEMS_PER_LANE,), "float32")
            if not DISABLE_OUTPUT and (pass_index + 1) * NUM_WARPS > RECOVERY_STEPS:
                q_base: T.int64 = T.cast(n, "int64") * q_batch_stride + T.cast(
                    (t_pre * NUM_HEADS + h) * K + k_pre, "int64"
                )
                for i in T.unroll(ELEMS_PER_LANE):
                    q_bits[i] = _global_load_u16(q, q_base + i)
                    r_q[i] = _bf16_to_f32(q_bits[i])

            k_base: T.int64 = T.cast(n, "int64") * k_batch_stride + T.cast(
                (t_pre * NUM_HEADS + h) * K + k_pre, "int64"
            )
            for i in T.unroll(ELEMS_PER_LANE):
                k_bits[i] = _global_load_u16(k, k_base + i)
                r_k[i] = _bf16_to_f32(k_bits[i])

            if USE_QK_L2NORM:
                sum_k: T.float32 = T.float32(0.0)
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

                if not DISABLE_OUTPUT and (pass_index + 1) * NUM_WARPS > RECOVERY_STEPS:
                    sum_q: T.float32 = T.float32(0.0)
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
            elif not DISABLE_OUTPUT and (pass_index + 1) * NUM_WARPS > RECOVERY_STEPS:
                for i in T.unroll(ELEMS_PER_LANE):
                    r_q[i] = _mul_f32(r_q[i], T.float32(SCALE))

            shared_base: T.int32 = t_pre * (K + 8) + k_pre
            if not DISABLE_OUTPUT and (pass_index + 1) * NUM_WARPS > RECOVERY_STEPS:
                for i in T.unroll(ELEMS_PER_LANE):
                    _shared_store_f32(s_q, shared_base + i, r_q[i])
            for i in T.unroll(ELEMS_PER_LANE):
                _shared_store_f32(s_k, shared_base + i, r_k[i])

            a_bits: T.uint16 = _global_load_u16(a, (n * SEQ_LEN + t_pre) * NUM_V_HEADS + hv)
            x_value: T.float32 = _mixed_add_bf16_f32(a_bits, dt_value)
            softplus_exp: T.float32 = _exp2_f32(_mul_f32(x_value, T.float32(LOG2_E)))
            softplus_value: T.float32 = _mul_f32(
                _log2_f32(_add_f32(T.float32(1.0), softplus_exp)), T.float32(LN_2)
            )
            use_softplus: T.float32 = T.if_then_else(
                x_value <= T.float32(20.0), T.float32(1.0), T.float32(0.0)
            )
            softplus_x: T.float32 = _fma_f32(
                softplus_value,
                use_softplus,
                _mul_f32(x_value, _sub_f32(T.float32(1.0), use_softplus)),
            )
            exp_A: T.float32 = _exp2_f32(_mul_f32(A_value, T.float32(LOG2_E)))
            gate_exponent: T.float32 = _mul_f32(_sub_f32(T.float32(0.0), exp_A), softplus_x)
            if lane_in_warp == 0:
                b_bits: T.uint16 = _global_load_u16(
                    b_gate, (n * SEQ_LEN + t_pre) * NUM_V_HEADS + hv
                )
                b_value: T.float32 = _bf16_to_f32(b_bits)
                beta: T.float32 = _rcp_f32(
                    _add_f32(T.float32(1.0), _exp2_f32(_mul_f32(b_value, T.float32(-LOG2_E))))
                )
                g_value: T.float32 = _exp2_f32(_mul_f32(gate_exponent, T.float32(LOG2_E)))
                _shared_store_f32x2(s_gb, t_pre * 2, g_value, beta)
        T.cuda.cta_sync()

    for iter_index in T.unroll(ITERS_PER_GROUP):
        v_base: T.int32 = v_tile * TILE_V + group * ROWS_PER_GROUP + iter_index * ILP_ROWS
        read_offset: T.int64 = read_state_base + T.cast(v_base * K + k_start, "int64")
        r_h = T.alloc_local((ILP_ROWS * ELEMS_PER_LANE,), "float32")
        if TILE_V == 32:
            for pair in T.unroll(4):
                for row in T.unroll(ILP_ROWS):
                    word: T.uint32 = prefetch_words[row * 4 + pair]
                    r_h[row * ELEMS_PER_LANE + pair * 2] = T.cuda.uint_as_float(
                        T.shift_left(word, T.uint32(16))
                    )
                    r_h[row * ELEMS_PER_LANE + pair * 2 + 1] = T.cuda.uint_as_float(
                        T.bitwise_and(word, T.uint32(0xFFFF0000))
                    )
        else:
            _load_state_bf16x32(state, read_offset, r_h)

        r_k_main = T.alloc_local((ELEMS_PER_LANE,), "float32")
        gate_pair = T.alloc_local((2,), "float32", align=8)
        sums = T.alloc_local((ILP_ROWS,), "float32")
        values = T.alloc_local((ILP_ROWS,), "float32")

        for t in T.serial(0, phase_a_bound, unroll=False):
            _shared_load_f32x2(s_gb, t * 2, gate_pair)
            g_value: T.float32 = gate_pair[0]
            beta: T.float32 = gate_pair[1]
            for row in T.unroll(ILP_ROWS):
                sums[row] = T.float32(0.0)
            for pair in T.unroll(ELEMS_PER_LANE // 2):
                r_k_main[pair * 2] = _shared_load_f32(s_k, t * (K + 8) + k_start + pair * 2)
                r_k_main[pair * 2 + 1] = _shared_load_f32(s_k, t * (K + 8) + k_start + pair * 2 + 1)
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
                        sums[row],
                        T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), sums[row], delta, 32),
                    )
            v_base_input: T.int64 = T.cast(n, "int64") * v_batch_stride + T.cast(
                (t * NUM_V_HEADS + hv) * V + v_base, "int64"
            )
            for row in T.unroll(ILP_ROWS):
                v_bits: T.uint16 = _global_load_u16(v, v_base_input + row)
                values[row] = _mul_f32(_mixed_sub_bf16_f32(v_bits, sums[row]), beta)
            for pair in T.unroll(ELEMS_PER_LANE // 2):
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

        if (
            (
                RECOVERY_STEPS > 0
                or (
                    PER_REQUEST_ACCEPTED_STEPS
                    and not DISABLE_OUTPUT
                    and not DISABLE_STATE_UPDATE
                    and not PER_TOKEN_POOL_SCATTER
                )
            )
            and not DISABLE_STATE_UPDATE
            and not CACHE_INTERMEDIATE_STATES
        ):
            write_offset: T.int64 = write_state_base + T.cast(v_base * K + k_start, "int64")
            _store_state_f32x32(state, write_offset, r_h)

        r_q_main = T.alloc_local((ELEMS_PER_LANE,), "float32")
        output_sums = T.alloc_local((ILP_ROWS,), "float32")
        for t_offset in T.serial(0, phase_b_bound, unroll=False):
            t: T.int32 = phase_b_begin + t_offset
            _shared_load_f32x2(s_gb, t * 2, gate_pair)
            g_value: T.float32 = gate_pair[0]
            beta: T.float32 = gate_pair[1]
            for row in T.unroll(ILP_ROWS):
                sums[row] = T.float32(0.0)
            for pair in T.unroll(ELEMS_PER_LANE // 2):
                r_k_main[pair * 2] = _shared_load_f32(s_k, t * (K + 8) + k_start + pair * 2)
                r_k_main[pair * 2 + 1] = _shared_load_f32(s_k, t * (K + 8) + k_start + pair * 2 + 1)
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
                        sums[row],
                        T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), sums[row], delta, 32),
                    )
            v_base_input: T.int64 = T.cast(n, "int64") * v_batch_stride + T.cast(
                (t * NUM_V_HEADS + hv) * V + v_base, "int64"
            )
            for row in T.unroll(ILP_ROWS):
                v_bits: T.uint16 = _global_load_u16(v, v_base_input + row)
                values[row] = _mul_f32(_mixed_sub_bf16_f32(v_bits, sums[row]), beta)
                output_sums[row] = T.float32(0.0)
            for pair in T.unroll(ELEMS_PER_LANE // 2):
                if not DISABLE_OUTPUT:
                    r_q_main[pair * 2] = _shared_load_f32(s_q, t * (K + 8) + k_start + pair * 2)
                    r_q_main[pair * 2 + 1] = _shared_load_f32(
                        s_q, t * (K + 8) + k_start + pair * 2 + 1
                    )
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
                    if not DISABLE_OUTPUT:
                        output_sums[row] = _fma_f32(
                            r_h[row * ELEMS_PER_LANE + pair * 2],
                            r_q_main[pair * 2],
                            output_sums[row],
                        )
                        output_sums[row] = _fma_f32(
                            r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                            r_q_main[pair * 2 + 1],
                            output_sums[row],
                        )
            if not DISABLE_OUTPUT:
                for delta_index in T.unroll(4):
                    delta: T.int32 = T.shift_right(T.int32(8), delta_index)
                    for row in T.unroll(ILP_ROWS):
                        output_sums[row] = _add_f32(
                            output_sums[row],
                            T.cuda.__shfl_xor_sync(
                                T.uint32(0xFFFFFFFF), output_sums[row], delta, 32
                            ),
                        )
                output_base: T.int64 = T.cast(
                    ((n * SEQ_LEN + t) * NUM_V_HEADS + hv) * V + v_base, "int64"
                )
                output_bits = T.alloc_local((ILP_ROWS,), "uint16")
                for row in T.unroll(ILP_ROWS):
                    output_bits[row] = _f32_to_bf16(output_sums[row])
                for row in T.unroll(ILP_ROWS):
                    _global_store_u16_pred(output, output_base + row, output_bits[row], lane == 0)

            if CACHE_INTERMEDIATE_STATES or PER_TOKEN_POOL_SCATTER:
                if PER_TOKEN_POOL_SCATTER:
                    scatter_slot: T.int32 = _global_load_s32(ssm_state_indices, n * SEQ_LEN + t)
                    if PER_TOKEN_POOL_SCATTER_FLAT:
                        state_write_base: T.int64 = (
                            T.cast(scatter_slot, "int64") * NUM_V_HEADS + hv
                        ) * V * K + T.cast(v_base * K + k_start, "int64")
                    else:
                        state_write_base = T.cast(
                            scatter_slot, "int64"
                        ) * state_slot_stride + T.cast(hv * V * K + v_base * K + k_start, "int64")
                else:
                    state_write_base = T.cast(
                        (((n * SEQ_LEN + t) * NUM_V_HEADS + hv) * V + v_base) * K + k_start, "int64"
                    )
                if PER_TOKEN_POOL_SCATTER and not PER_TOKEN_POOL_SCATTER_FLAT:
                    _store_state_f32x32(state, state_write_base, r_h)
                else:
                    _store_state_f32x32(intermediate, state_write_base, r_h)

        if (
            not DISABLE_STATE_UPDATE
            and not CACHE_INTERMEDIATE_STATES
            and RECOVERY_STEPS == 0
            and not (
                PER_REQUEST_ACCEPTED_STEPS
                and not DISABLE_OUTPUT
                and not DISABLE_STATE_UPDATE
                and not PER_TOKEN_POOL_SCATTER
            )
            and not (PER_TOKEN_POOL_SCATTER and SAME_POOL)
        ):
            write_offset: T.int64 = write_state_base + T.cast(v_base * K + k_start, "int64")
            _store_state_f32x32(state, write_offset, r_h)


def _pool_factor(config: dict[str, Any]) -> int:
    factor = 1
    if config.get("per_token_pool_scatter", False):
        factor += int(config["seq_len"])
    if not config.get("same_pool", True):
        factor += 1
    if config.get("negative_read_index", False) or config.get("negative_write_index", False):
        factor += 1
    return factor


def _require_supported_config(config: dict[str, Any]) -> None:
    seq_len = int(config["seq_len"])
    num_heads = int(config["num_heads"])
    num_v_heads = int(config["num_v_heads"])
    tile_v = int(config["tile_v"])
    if int(config["batch"]) < 1:
        raise ValueError("batch must be positive")
    if seq_len < 2 or seq_len > 8:
        raise ValueError("wide-vector MTP port requires seq_len in [2, 8]")
    if (num_heads, num_v_heads) not in _HEAD_CONFIGS:
        raise ValueError(f"unsupported Qwen3-Next TP head pair {(num_heads, num_v_heads)}")
    if num_v_heads % num_heads:
        raise ValueError("num_v_heads must be divisible by num_heads")
    if tile_v not in (32, 64, 128) or V % tile_v:
        raise ValueError("wide-vector MTP port requires tile_v in {32, 64, 128}")
    if (tile_v // NUM_GROUPS) % ILP_ROWS:
        raise ValueError("tile_v is incompatible with the 8-group, ILP4 mapping")

    recovery_steps = int(config.get("recovery_steps", 0))
    cache = bool(config.get("cache_intermediate_states", False))
    disable_update = bool(config.get("disable_state_update", False))
    scatter = bool(config.get("per_token_pool_scatter", False))
    if not 0 <= recovery_steps <= seq_len:
        raise ValueError("recovery_steps must be in [0, seq_len]")
    if recovery_steps and (cache or disable_update):
        raise ValueError("positive recovery requires state update and forbids dense cache")
    if scatter and (cache or disable_update or recovery_steps or seq_len < 2):
        raise ValueError("per-token scatter requires update, no cache/recovery, and T >= 2")


def get_kernel(**kwargs: Any):
    """Return the source-specialized TIRx PrimFunc."""
    config = dict(kwargs)
    _require_supported_config(config)
    seq_len = int(config["seq_len"])
    num_v_heads = int(config["num_v_heads"])
    tile_v = int(config["tile_v"])
    cache = bool(config.get("cache_intermediate_states", False))
    scatter = bool(config.get("per_token_pool_scatter", False))
    scatter_flat = scatter and not bool(config.get("padded_pool", False))
    effective_disable_update = bool(config.get("disable_state_update", False)) or cache
    pool_factor = int(config.get("pool_factor_override", _pool_factor(config)))
    intermediate_batch_stride = 0
    if cache:
        intermediate_batch_stride = seq_len * num_v_heads * V * K
    elif scatter_flat:
        intermediate_batch_stride = pool_factor * num_v_heads * V * K

    cosize_qk_bytes = 4 * ((seq_len - 1) * (K + 8) + K)
    kernel = _gdn_decode_bf16_wide_vec_mtp.specialize(
        SEQ_LEN=seq_len,
        NUM_HEADS=int(config["num_heads"]),
        NUM_V_HEADS=num_v_heads,
        TILE_V=tile_v,
        NUM_V_TILES=V // tile_v,
        ROWS_PER_GROUP=tile_v // NUM_GROUPS,
        ITERS_PER_GROUP=(tile_v // NUM_GROUPS) // ILP_ROWS,
        POOL_FACTOR=pool_factor,
        INTERMEDIATE_BATCH_STRIDE=intermediate_batch_stride,
        INTERMEDIATE_DUMMY_ELEMENTS=0 if (cache or scatter_flat) else 1,
        ACCEPTED_BATCH_STRIDE=1 if config.get("per_request_accepted_steps", False) else 0,
        ACCEPTED_DUMMY_ELEMENTS=0 if config.get("per_request_accepted_steps", False) else 1,
        SSM_BATCH_STRIDE=seq_len if scatter else 0,
        SSM_DUMMY_ELEMENTS=0 if scatter else 1,
        SHARED_BYTES=1096 * seq_len + 256,
        S_K_BYTE_OFFSET=cosize_qk_bytes,
        S_GB_BYTE_OFFSET=2 * cosize_qk_bytes,
        USE_QK_L2NORM=bool(config.get("use_qk_l2norm", True)),
        DISABLE_STATE_UPDATE=effective_disable_update,
        CACHE_INTERMEDIATE_STATES=cache,
        SAME_POOL=bool(config.get("same_pool", True)),
        DISABLE_OUTPUT=bool(config.get("disable_output", False)),
        RECOVERY_STEPS=int(config.get("recovery_steps", 0)),
        PER_REQUEST_ACCEPTED_STEPS=bool(config.get("per_request_accepted_steps", False)),
        PER_TOKEN_POOL_SCATTER=scatter,
        PER_TOKEN_POOL_SCATTER_FLAT=scatter_flat,
    )
    return kernel.with_attr(
        "tirx.kernel_launch_params", ["blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory"]
    )


def _allocate_pool(
    pool_slots: int,
    num_v_heads: int,
    *,
    padded: bool,
    device: torch.device,
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
        * 0.05
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
        k = torch.randn_like(q) * 0.05
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
        raise SkipTest("CUDA is required for BF16 wide-vector GDN MTP decode")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(f"BF16 wide-vector GDN MTP decode requires SM100, got {capability}")
    return device


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
    generator.manual_seed(int(config.get("seed", 0)) + 20260812)

    pool_factor = _pool_factor(config)
    pool_slots = batch * pool_factor
    initial_pool, initial_backing = _allocate_pool(
        pool_slots, num_v_heads, padded=padded_pool, device=device, generator=generator
    )
    tirx_state, tirx_state_backing = _clone_pool_layout(initial_pool, padded=padded_pool)
    source_state, source_state_backing = _clone_pool_layout(initial_pool, padded=padded_pool)

    slot_offset = 1 if negative_read or negative_write else 0
    read_indices = torch.arange(batch, dtype=torch.int32, device=device) + slot_offset
    if negative_read:
        read_indices[-1] = -1

    scatter_indices: torch.Tensor | None = None
    next_slot = slot_offset + batch
    if scatter:
        scatter_indices = torch.arange(
            next_slot, next_slot + batch * seq_len, dtype=torch.int32, device=device
        ).reshape(batch, seq_len)
        next_slot += batch * seq_len
    if same_pool:
        write_indices = read_indices
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
    b_gate = torch.randn_like(a) * 0.05

    output_initial = torch.randn(
        (batch, seq_len, num_v_heads, V), dtype=torch.bfloat16, device=device, generator=generator
    )
    cache = bool(config.get("cache_intermediate_states", False))
    if cache:
        intermediate_initial = torch.randn(
            (batch, seq_len, num_v_heads, V, K),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        tirx_intermediate = intermediate_initial.clone()
        source_intermediate = intermediate_initial.clone()
    else:
        tirx_intermediate = torch.zeros((1,), dtype=torch.bfloat16, device=device)
        source_intermediate = torch.zeros((1,), dtype=torch.bfloat16, device=device)

    if config.get("per_request_accepted_steps", False):
        accepted_values = tuple(int(value) for value in config.get("accepted_steps", (0,)))
        if len(accepted_values) == 1:
            accepted_values = accepted_values * batch
        if len(accepted_values) != batch:
            raise ValueError("accepted_steps must contain one value per batch row")
        accepted_steps = torch.tensor(accepted_values, dtype=torch.int32, device=device)
    else:
        accepted_steps = torch.zeros((1,), dtype=torch.int32, device=device)
    ssm_arg = (
        scatter_indices
        if scatter_indices is not None
        else torch.zeros((1,), dtype=torch.int32, device=device)
    )

    return {
        "config": config,
        "pool_slots": pool_slots,
        "initial_pool": initial_pool.clone(),
        "initial_backing": initial_backing,
        "tirx_state": tirx_state,
        "tirx_state_backing": tirx_state_backing,
        "source_state": source_state,
        "source_state_backing": source_state_backing,
        "read_indices": read_indices,
        "write_indices": write_indices,
        "accepted_steps": accepted_steps,
        "ssm_state_indices": ssm_arg,
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


@functools.cache
def _compile_tirx(
    seq_len: int,
    num_heads: int,
    num_v_heads: int,
    tile_v: int,
    pool_factor: int,
    use_qk_l2norm: bool,
    disable_state_update: bool,
    cache_intermediate_states: bool,
    same_pool: bool,
    disable_output: bool,
    recovery_steps: int,
    per_request_accepted_steps: bool,
    per_token_pool_scatter: bool,
    padded_pool: bool,
):
    from tirx_kernels.runner import compile_kernel

    config = {
        "seq_len": seq_len,
        "batch": 1,
        "num_heads": num_heads,
        "num_v_heads": num_v_heads,
        "tile_v": tile_v,
        "use_qk_l2norm": use_qk_l2norm,
        "disable_state_update": disable_state_update,
        "cache_intermediate_states": cache_intermediate_states,
        "same_pool": same_pool,
        "disable_output": disable_output,
        "recovery_steps": recovery_steps,
        "per_request_accepted_steps": per_request_accepted_steps,
        "per_token_pool_scatter": per_token_pool_scatter,
        "padded_pool": padded_pool,
        "pool_factor_override": pool_factor,
    }
    return compile_kernel(get_kernel(**config))


def _compile_tirx_for_config(config: dict[str, Any]):
    return _compile_tirx(
        int(config["seq_len"]),
        int(config["num_heads"]),
        int(config["num_v_heads"]),
        int(config["tile_v"]),
        _pool_factor(config),
        bool(config.get("use_qk_l2norm", True)),
        bool(config.get("disable_state_update", False)),
        bool(config.get("cache_intermediate_states", False)),
        bool(config.get("same_pool", True)),
        bool(config.get("disable_output", False)),
        int(config.get("recovery_steps", 0)),
        bool(config.get("per_request_accepted_steps", False)),
        bool(config.get("per_token_pool_scatter", False)),
        bool(config.get("padded_pool", False)),
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
    cache = bool(config.get("cache_intermediate_states", False))
    scatter = bool(config.get("per_token_pool_scatter", False))
    scatter_flat = scatter and not bool(config.get("padded_pool", False))
    if scatter_flat:
        intermediate = _storage_span(state, int(state.stride(0)) * case["pool_slots"])
    else:
        intermediate = _storage_span(case["tirx_intermediate"], case["tirx_intermediate"].numel())
    if not cache and not scatter_flat:
        intermediate = intermediate[:1]
    return (
        _storage_span(state, int(state.stride(0)) * case["pool_slots"]),
        intermediate,
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
        case["accepted_steps"],
        case["ssm_state_indices"].reshape(-1),
        int(state.stride(0)),
        int(q.stride(0)),
        int(k.stride(0)),
        int(v.stride(0)),
        batch,
    )


def _run_reference(case: dict[str, Any]) -> torch.Tensor:
    from tirx_kernels.flashinfer._gdn_reference import gated_delta_rule_decode

    config = case["config"]
    accepted_steps = (
        case["accepted_steps"] if config.get("per_request_accepted_steps", False) else None
    )
    scatter = case["ssm_state_indices"] if config.get("per_token_pool_scatter", False) else None
    cache_intermediate_states = bool(config.get("cache_intermediate_states", False))
    effective_disable_state_update = bool(config.get("disable_state_update", False)) or (
        cache_intermediate_states
    )
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
        write_indices=(
            case["read_indices"] if bool(config.get("same_pool", True)) else case["write_indices"]
        ),
        intermediate_states=(
            case["source_intermediate"] if cache_intermediate_states else None
        ),
        accepted_steps=accepted_steps,
        ssm_state_indices=scatter,
        disable_state_update=effective_disable_state_update,
        use_qk_l2norm=bool(config.get("use_qk_l2norm", True)),
        scale=SCALE,
        output=case["source_output"],
        disable_output=bool(config.get("disable_output", False)),
        recovery_steps=int(config.get("recovery_steps", 0)),
        fused_accepted_steps=(
            accepted_steps is not None
            and not bool(config.get("disable_output", False))
            and not effective_disable_state_update
            and scatter is None
        ),
    )


def _assert_case_close(case: dict[str, Any]) -> None:
    config = case["config"]
    if bool(config.get("disable_output", False)):
        torch.testing.assert_close(case["tirx_output"], case["source_output"], atol=0, rtol=0)
    else:
        torch.testing.assert_close(
            case["tirx_output"].float(), case["source_output"].float(), atol=1.0e-3, rtol=5.0e-3
        )
    torch.testing.assert_close(
        case["tirx_state"].float(), case["source_state"].float(), atol=2.0e-2, rtol=1.0e-2
    )
    if bool(config.get("cache_intermediate_states", False)):
        torch.testing.assert_close(
            case["tirx_intermediate"].float(),
            case["source_intermediate"].float(),
            atol=2.0e-2,
            rtol=1.0e-2,
        )
    if case["qkv_backing"] is not None:
        torch.testing.assert_close(case["qkv_backing"], case["qkv_snapshot"], atol=0, rtol=0)


def run_test(**kwargs: Any) -> None:
    case = prepare_data(**kwargs)
    executable = _tirx_executable(case)
    executable(*_tirx_args(case))
    torch.cuda.synchronize(case["tirx_state"].device)
    _run_reference(case)
    torch.cuda.synchronize(case["tirx_state"].device)
    _assert_case_close(case)


def prepare_bench(**kwargs: Any):
    """Compile the selected wide-vector MTP specialization before CUDA setup."""
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
    kwargs = {**prepared["config"], **kwargs}
    case = prepare_data(**kwargs)
    executable = prepared["executable"]
    args = _tirx_args(case)

    def source_builder():
        from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp_wide_vec

        config = case["config"]

        def launch():
            gated_delta_rule_mtp_wide_vec(
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
                accepted_steps=(
                    case["accepted_steps"]
                    if config.get("per_request_accepted_steps", False)
                    else None
                ),
                ssm_state_indices=(
                    case["ssm_state_indices"]
                    if config.get("per_token_pool_scatter", False)
                    else None
                ),
                disable_state_update=bool(config.get("disable_state_update", False)),
                use_qk_l2norm_in_kernel=bool(config.get("use_qk_l2norm", True)),
                scale=SCALE,
                output=case["source_output"],
                tile_v=int(config["tile_v"]),
                disable_output=bool(config.get("disable_output", False)),
                recovery_steps=int(config.get("recovery_steps", 0)),
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
    "prepare_data",
    "run_bench",
    "run_test",
]
