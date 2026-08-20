# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Integration scaffold for FlashInfer's SM100 BF16 ILP4 GDN decode kernel.

Upstream source: flashinfer/gdn_kernels/gdn_decode_bf16_state.py.
"""

import functools
import math
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as TK
from tirx_kernels.runner import bench

KERNEL_META = {"name": "gdn_decode_bf16_ilp4", "category": "flashinfer", "compute_capability": 10}


K = 128
V = 128
THREADS = 128
NUM_WARPS = 4
NUM_GROUPS = 4
LANES_PER_GROUP = 32
ELEMS_PER_LANE = 4
ILP_ROWS = 4
SOURCE_NUM_SMS = 148
LOG2_E = 1.4426950408889634
LN_2 = 0.6931471805599453
SCALE = 1.0 / math.sqrt(K)


def _local_scalar(dtype: str, value):
    out = TK.alloc_local((1,), dtype)
    TK.assign(out[0], value)
    return out


def _ptx_unary(chain: str, value):
    out = TK.alloc_local((1,), "float32")
    TK.ptx[chain](out[0], value)
    return out[0]


def _ptx_binary(chain: str, lhs, rhs, dtype: str = "float32"):
    out = TK.alloc_local((1,), dtype)
    TK.ptx[chain](out[0], lhs, rhs)
    return out[0]


def _ptx_ternary(chain: str, lhs, rhs, acc):
    out = TK.alloc_local((1,), "float32")
    TK.ptx[chain](out[0], lhs, rhs, acc)
    return out[0]


def _add_f32(lhs, rhs):
    return _ptx_binary("add.f32", lhs, rhs)


def _sub_f32(lhs, rhs):
    return _ptx_binary("sub.f32", lhs, rhs)


def _neg_f32(value):
    return _ptx_unary("neg.f32", value)


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
    out = TK.alloc_local((1,), "uint16")
    TK.evaluate(TK.ptx.ld.global_.b16(out[0], buffer.ptr_to([index])))
    return out[0]


def _global_load_s32(buffer, index):
    out = TK.alloc_local((1,), "int32")
    TK.evaluate(TK.ptx.ld.global_.s32(out[0], buffer.ptr_to([index])))
    return out[0]


def _global_load_f32(buffer, index):
    out = TK.alloc_local((1,), "float32")
    TK.evaluate(TK.ptx.ld.global_.b32(out[0], buffer.ptr_to([index])))
    return out[0]


def _bf16_to_f32(bits):
    out = TK.alloc_local((1,), "float32")
    TK.evaluate(TK.ptx.cvt.f32.bf16(out[0], TK.cast(bits, "uint16")))
    return out[0]


def _f32_to_bf16(value):
    out = TK.alloc_local((1,), "uint16")
    TK.evaluate(TK.ptx.cvt.rn.bf16.f32(out[0], value))
    return out[0]


def _mixed_add_bf16_f32(bits, value):
    out = TK.alloc_local((1,), "float32")
    TK.evaluate(TK.ptx.add.rn.f32.bf16(out[0], TK.cast(bits, "uint16"), value))
    return out[0]


def _mixed_sub_bf16_f32(bits, value):
    out = TK.alloc_local((1,), "float32")
    TK.evaluate(TK.ptx.sub.rn.f32.bf16(out[0], TK.cast(bits, "uint16"), value))
    return out[0]


def _bf16_square_fma(bits, acc):
    out = TK.alloc_local((1,), "float32")
    TK.evaluate(
        TK.ptx.fma.rn.f32.bf16(out[0], TK.cast(bits, "uint16"), TK.cast(bits, "uint16"), acc)
    )
    return out[0]


def _shared_store_f32(buffer, index, value):
    TK.evaluate(TK.ptx.st.shared.b32(buffer.ptr_to([index]), TK.reinterpret("uint32", value)))


def _shared_store_f32x2(buffer, index, value0, value1):
    TK.evaluate(
        TK.ptx.st.shared.v2.b32(
            buffer.ptr_to([index]),
            TK.reinterpret("uint32", value0),
            TK.reinterpret("uint32", value1),
        )
    )


def _global_store_u16x4(buffer, index, values):
    base: TK.let = buffer.ptr_to([index])
    TK.evaluate(TK.ptx.st.global_.b16(base, values[0]))
    TK.evaluate(TK.ptx.st.global_.b16(TK.ptr_byte_offset(base, 2, "bfloat16"), values[1]))
    TK.evaluate(TK.ptx.st.global_.b16(TK.ptr_byte_offset(base, 4, "bfloat16"), values[2]))
    TK.evaluate(TK.ptx.st.global_.b16(TK.ptr_byte_offset(base, 6, "bfloat16"), values[3]))


def _shared_load_f32x2(buffer, index, values):
    words = TK.alloc_local((2,), "uint32", align=8)
    TK.evaluate(TK.ptx.ld.shared.v2.b32(words[0], words[1], buffer.ptr_to([index])))
    TK.ptx.mov.b32(values[0], TK.reinterpret("float32", words[0]))
    TK.ptx.mov.b32(values[1], TK.reinterpret("float32", words[1]))


def _shared_load_f32x4(buffer, index, values):
    pairs = TK.alloc_local((2,), "uint64", align=16)
    TK.evaluate(TK.ptx.ld.shared.v2.b64(pairs[0], pairs[1], buffer.ptr_to([index])))
    TK.ptx.mov.b32(values[0], TK.cuda.float2_x(pairs[0]))
    TK.ptx.mov.b32(values[1], TK.cuda.float2_y(pairs[0]))
    TK.ptx.mov.b32(values[2], TK.cuda.float2_x(pairs[1]))
    TK.ptx.mov.b32(values[3], TK.cuda.float2_y(pairs[1]))


def _load_state_bf16x16(buffer, index, values):
    bits = TK.alloc_local((ILP_ROWS * ELEMS_PER_LANE,), "uint16", align=8)
    with TK.unroll(ILP_ROWS) as row:
        base = TK.local_scalar("int32")
        TK.assign(base, row * ELEMS_PER_LANE)
        TK.evaluate(
            TK.ptx.ld.global_.v4.b16(
                bits[base],
                bits[base + 1],
                bits[base + 2],
                bits[base + 3],
                buffer.ptr_to([index + row * K]),
            )
        )
    with TK.unroll(ELEMS_PER_LANE) as elem:
        with TK.unroll(ILP_ROWS) as row:
            TK.ptx.mov.b32(
                values[row * ELEMS_PER_LANE + elem], _bf16_to_f32(bits[row * ELEMS_PER_LANE + elem])
            )


def _store_state_f32x16(buffer, index, values):
    words = TK.alloc_local((ILP_ROWS * 2,), "uint32", align=8)
    with TK.unroll(ELEMS_PER_LANE // 2) as pair:
        with TK.unroll(ILP_ROWS) as row:
            TK.ptx.mov.b32(
                words[row * 2 + pair],
                TK.cuda.float22bfloat162_rn(
                    values[row * ELEMS_PER_LANE + pair * 2],
                    values[row * ELEMS_PER_LANE + pair * 2 + 1],
                ),
            )
    with TK.unroll(ILP_ROWS) as row:
        TK.evaluate(
            TK.ptx.st.global_.v2.b32(
                buffer.ptr_to([index + row * K]), words[row * 2], words[row * 2 + 1]
            )
        )


def _packed_fma(lhs0, lhs1, rhs0, rhs1, acc0, acc1):
    out = TK.alloc_local((1,), "uint64")
    TK.evaluate(
        TK.ptx.fma.rn.f32x2(
            out[0],
            TK.cuda.make_float2(lhs0, lhs1),
            TK.cuda.make_float2(rhs0, rhs1),
            TK.cuda.make_float2(acc0, acc1),
        )
    )
    return out[0]


def _gate_pair(a, b_gate, A_value, dt_value, index):
    a_bits: TK.uint16 = _global_load_u16(a, index)
    b_bits: TK.uint16 = _global_load_u16(b_gate, index)
    b_value: TK.float32 = _bf16_to_f32(b_bits)
    x_value: TK.float32 = _mixed_add_bf16_f32(a_bits, dt_value)
    softplus_exp: TK.float32 = _exp2_f32(_mul_f32(x_value, TK.float32(LOG2_E)))
    softplus_value: TK.float32 = _mul_f32(
        _log2_f32(_add_f32(TK.float32(1.0), softplus_exp)), TK.float32(LN_2)
    )
    use_softplus: TK.float32 = TK.if_then_else(
        x_value <= TK.float32(20.0), TK.float32(1.0), TK.float32(0.0)
    )
    softplus_x: TK.float32 = _fma_f32(
        softplus_value, use_softplus, _mul_f32(x_value, _sub_f32(TK.float32(1.0), use_softplus))
    )
    exp_A: TK.float32 = _exp2_f32(_mul_f32(A_value, TK.float32(LOG2_E)))
    gate_exponent: TK.float32 = _mul_f32(_neg_f32(exp_A), softplus_x)
    beta: TK.float32 = _rcp_f32(
        _add_f32(TK.float32(1.0), _exp2_f32(_mul_f32(b_value, TK.float32(-LOG2_E))))
    )
    g_value: TK.float32 = _exp2_f32(_mul_f32(gate_exponent, TK.float32(LOG2_E)))
    return TK.cuda.make_float2(g_value, beta)


_SEQ_LENS = (1, 2, 4, 8)
_HEAD_CONFIGS = ((16, 32), (8, 16), (4, 8), (2, 4))
_BATCH_SIZES = (1, 4, 8, 16, 32, 64, 128, 256, 512)


def _production_tile_v(seq_len: int, batch: int, num_v_heads: int) -> int | None:
    work_units = batch * num_v_heads
    if seq_len == 1:
        if work_units >= 512:
            return None
        if work_units <= 128:
            return 16
        if work_units * 2 >= 4 * SOURCE_NUM_SMS:
            return 64
        return 32
    return 16 if work_units < 128 else None


def _production_case(seq_len: int, batch: int, num_heads: int, num_v_heads: int) -> dict[str, Any]:
    tile_v = _production_tile_v(seq_len, batch, num_v_heads)
    if tile_v is None:
        raise ValueError("shape does not dispatch to the BF16 ILP4 source kernel")
    return {
        "label": f"t{seq_len}_b{batch}_h{num_heads}_hv{num_v_heads}_tv{tile_v}",
        "seq_len": seq_len,
        "batch": batch,
        "num_heads": num_heads,
        "num_v_heads": num_v_heads,
        "tile_v": tile_v,
        "use_qk_l2norm": True,
        "disable_state_update": seq_len > 1,
        "cache_intermediate_states": seq_len > 1,
        "disable_output": False,
    }


BENCH_CONFIGS = [
    _production_case(seq_len, batch, num_heads, num_v_heads)
    for seq_len in _SEQ_LENS
    for num_heads, num_v_heads in _HEAD_CONFIGS
    for batch in _BATCH_SIZES
    if _production_tile_v(seq_len, batch, num_v_heads) is not None
]


def _correctness_case(
    label: str,
    *,
    seq_len: int = 4,
    batch: int = 2,
    num_heads: int = 16,
    num_v_heads: int = 32,
    tile_v: int = 16,
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


CONFIGS = [dict(config) for config in BENCH_CONFIGS] + [
    _correctness_case(
        "t1_tv64_source_picker", seq_len=1, batch=96, num_heads=2, num_v_heads=4, tile_v=64
    ),
    _correctness_case("t1_l2off", seq_len=1, use_qk_l2norm=False),
    _correctness_case(
        "t1_split_padded_negative",
        seq_len=1,
        same_pool=False,
        padded_pool=True,
        negative_read_index=True,
        negative_write_index=True,
    ),
    _correctness_case("t1_packed_qkv", seq_len=1, packed_qkv=True),
    _correctness_case(
        "t1_accepted", seq_len=1, per_request_accepted_steps=True, accepted_steps=(0, 0)
    ),
    _correctness_case("t2_base", seq_len=2),
    _correctness_case("t2_state_only", seq_len=2, disable_output=True),
    _correctness_case("t3_precompute_tail", seq_len=3),
    _correctness_case(
        "t5_accepted", seq_len=5, per_request_accepted_steps=True, accepted_steps=(1, 4)
    ),
    _correctness_case("t4_cache_update", cache_intermediate_states=True),
    _correctness_case("t4_split", same_pool=False),
    _correctness_case("t4_padded", padded_pool=True),
    _correctness_case(
        "t3_cache_no_output", seq_len=3, cache_intermediate_states=True, disable_output=True
    ),
    _correctness_case("t4_scatter_flat", per_token_pool_scatter=True),
    _correctness_case("t4_scatter_flat_split", same_pool=False, per_token_pool_scatter=True),
    _correctness_case("t4_scatter_padded", padded_pool=True, per_token_pool_scatter=True),
    _correctness_case(
        "t4_scatter_padded_split", padded_pool=True, same_pool=False, per_token_pool_scatter=True
    ),
    _correctness_case(
        "t5_accepted_scatter",
        seq_len=5,
        per_request_accepted_steps=True,
        accepted_steps=(2, 4),
        per_token_pool_scatter=True,
    ),
    _correctness_case("t4_disable_update", disable_state_update=True),
]

assert len(BENCH_CONFIGS) == 48


def _make_gdn_decode_bf16_ilp4(
    *,
    SEQ_LEN,
    NUM_HEADS,
    NUM_V_HEADS,
    TILE_V,
    NUM_V_TILES,
    ROWS_PER_GROUP,
    ITERS_PER_GROUP,
    POOL_FACTOR,
    INTERMEDIATE_BATCH_STRIDE,
    INTERMEDIATE_DUMMY_ELEMENTS,
    ACCEPTED_BATCH_STRIDE,
    ACCEPTED_DUMMY_ELEMENTS,
    SSM_BATCH_STRIDE,
    SSM_DUMMY_ELEMENTS,
    SHARED_BYTES,
    S_K_BYTE_OFFSET,
    S_GB_BYTE_OFFSET,
    USE_QK_L2NORM,
    DISABLE_STATE_UPDATE,
    CACHE_INTERMEDIATE_STATES,
    SAME_POOL,
    DISABLE_OUTPUT,
    PER_REQUEST_ACCEPTED_STEPS,
    PER_TOKEN_POOL_SCATTER,
    PER_TOKEN_POOL_SCATTER_FLAT,
):
    @TK.kernel(
        warps=NUM_WARPS, arch="sm_100a", min_blocks_per_sm=8, grid=False, thread_layout=False
    )
    def gdn_decode_bf16_ilp4(
        state: TK.gptr[TK.bf16],
        intermediate: TK.gptr[TK.bf16],
        A_log: TK.gptr[TK.f32],
        a: TK.gptr[TK.bf16],
        dt_bias: TK.gptr[TK.f32],
        q: TK.gptr[TK.bf16],
        k: TK.gptr[TK.bf16],
        v: TK.gptr[TK.bf16],
        b_gate: TK.gptr[TK.bf16],
        output: TK.gptr[TK.bf16],
        read_indices: TK.gptr[TK.i32],
        write_indices: TK.gptr[TK.i32],
        accepted_steps: TK.gptr[TK.i32],
        ssm_state_indices: TK.gptr[TK.i32],
        state_slot_stride: TK.i64,
        q_batch_stride: TK.i64,
        k_batch_stride: TK.i64,
        v_batch_stride: TK.i64,
        batch: TK.i32,
    ):
        smem = TK.smem_pool()
        if SEQ_LEN > 1:
            s_q = smem.alloc((S_K_BYTE_OFFSET // 4,), TK.f32, align=16)
            s_k = smem.alloc(((S_GB_BYTE_OFFSET - S_K_BYTE_OFFSET) // 4,), TK.f32, align=16)
            s_gb = smem.alloc((SEQ_LEN * 2,), TK.f32, align=16)
        else:
            scratch = smem.alloc((SHARED_BYTES,), TK.u8, align=16)
            s_q = s_k = s_gb = scratch
        smem.commit(SHARED_BYTES)
        linear_cta = TK.cta_id([batch * NUM_V_HEADS * NUM_V_TILES])
        tid = TK.thread_id([THREADS])
        warp = _local_scalar("int32", tid // LANES_PER_GROUP)
        lane = _local_scalar("int32", tid % LANES_PER_GROUP)
        k_start = _local_scalar("int32", lane[0] * ELEMS_PER_LANE)
        v_tile = _local_scalar("int32", linear_cta % NUM_V_TILES)
        cta_head = _local_scalar("int32", linear_cta // NUM_V_TILES)
        hv = _local_scalar("int32", cta_head[0] % NUM_V_HEADS)
        n = _local_scalar("int32", cta_head[0] // NUM_V_HEADS)
        h = _local_scalar("int32", hv[0] // (NUM_V_HEADS // NUM_HEADS))
        read_slot = _local_scalar(
            "int32", _max_s32(_global_load_s32(read_indices, n[0]), TK.int32(0))
        )
        write_slot = _local_scalar("int32", read_slot[0])
        if not SAME_POOL:
            TK.assign(write_slot[0], _max_s32(_global_load_s32(write_indices, n[0]), TK.int32(0)))
        read_state_base = _local_scalar(
            "int64",
            TK.cast(read_slot[0], "int64") * state_slot_stride + TK.cast(hv[0] * V * K, "int64"),
        )
        write_state_base = _local_scalar("int64", read_state_base[0])
        if not SAME_POOL:
            TK.assign(
                write_state_base[0],
                TK.cast(write_slot[0], "int64") * state_slot_stride
                + TK.cast(hv[0] * V * K, "int64"),
            )
        A_value = _local_scalar("float32", _global_load_f32(A_log, hv[0]))
        dt_value = _local_scalar("float32", _global_load_f32(dt_bias, hv[0]))
        token_bound = _local_scalar("int32", TK.int32(SEQ_LEN))
        if PER_REQUEST_ACCEPTED_STEPS:
            TK.assign(token_bound[0], _global_load_s32(accepted_steps, n[0]) + TK.int32(1))
        if SEQ_LEN > 1:
            for pass_index in range((SEQ_LEN + NUM_WARPS - 1) // NUM_WARPS):
                t_pre = _local_scalar("int32", pass_index * NUM_WARPS + warp[0])
                with TK.If(t_pre[0] < SEQ_LEN), TK.Then():
                    q_bits = TK.alloc_local((ELEMS_PER_LANE,), "uint16")
                    k_bits = TK.alloc_local((ELEMS_PER_LANE,), "uint16")
                    r_q_pre = TK.alloc_local((ELEMS_PER_LANE,), "float32")
                    r_k_pre = TK.alloc_local((ELEMS_PER_LANE,), "float32")
                    q_base = _local_scalar(
                        "int64",
                        TK.cast(n[0], "int64") * q_batch_stride
                        + TK.cast((t_pre[0] * NUM_HEADS + h[0]) * K + k_start[0], "int64"),
                    )
                    k_base = _local_scalar(
                        "int64",
                        TK.cast(n[0], "int64") * k_batch_stride
                        + TK.cast((t_pre[0] * NUM_HEADS + h[0]) * K + k_start[0], "int64"),
                    )
                    if not DISABLE_OUTPUT:
                        for elem in range(ELEMS_PER_LANE):
                            TK.ptx.mov.b16(q_bits[elem], _global_load_u16(q, q_base[0] + elem))
                    for elem in range(ELEMS_PER_LANE):
                        TK.ptx.mov.b16(k_bits[elem], _global_load_u16(k, k_base[0] + elem))
                    if not DISABLE_OUTPUT:
                        for elem in range(ELEMS_PER_LANE):
                            TK.ptx.mov.b32(r_q_pre[elem], _bf16_to_f32(q_bits[elem]))
                    for elem in range(ELEMS_PER_LANE):
                        TK.ptx.mov.b32(r_k_pre[elem], _bf16_to_f32(k_bits[elem]))
                    if USE_QK_L2NORM:
                        sum_k = _local_scalar("float32", TK.float32(0.0))
                        for elem in range(ELEMS_PER_LANE):
                            TK.assign(sum_k[0], _bf16_square_fma(k_bits[elem], sum_k[0]))
                        for delta_index in range(5):
                            delta = _local_scalar(
                                "int32", TK.shift_right(TK.int32(16), delta_index)
                            )
                            TK.assign(
                                sum_k[0],
                                _add_f32(
                                    sum_k[0],
                                    TK.cuda.__shfl_xor_sync(
                                        TK.uint32(4294967295), sum_k[0], delta[0], LANES_PER_GROUP
                                    ),
                                ),
                            )
                        k_factor = _local_scalar(
                            "float32", _rsqrt_f32(_add_f32(sum_k[0], TK.float32(1e-06)))
                        )
                        for elem in range(ELEMS_PER_LANE):
                            TK.ptx.mov.b32(r_k_pre[elem], _mul_f32(r_k_pre[elem], k_factor[0]))
                        if not DISABLE_OUTPUT:
                            sum_q = _local_scalar("float32", TK.float32(0.0))
                            for elem in range(ELEMS_PER_LANE):
                                TK.assign(sum_q[0], _bf16_square_fma(q_bits[elem], sum_q[0]))
                            for delta_index in range(5):
                                delta = _local_scalar(
                                    "int32", TK.shift_right(TK.int32(16), delta_index)
                                )
                                TK.assign(
                                    sum_q[0],
                                    _add_f32(
                                        sum_q[0],
                                        TK.cuda.__shfl_xor_sync(
                                            TK.uint32(4294967295),
                                            sum_q[0],
                                            delta[0],
                                            LANES_PER_GROUP,
                                        ),
                                    ),
                                )
                            q_factor = _local_scalar(
                                "float32",
                                _mul_f32(
                                    _rsqrt_f32(_add_f32(sum_q[0], TK.float32(1e-06))),
                                    TK.float32(SCALE),
                                ),
                            )
                            for elem in range(ELEMS_PER_LANE):
                                TK.ptx.mov.b32(r_q_pre[elem], _mul_f32(r_q_pre[elem], q_factor[0]))
                    elif not DISABLE_OUTPUT:
                        for elem in range(ELEMS_PER_LANE):
                            TK.ptx.mov.b32(
                                r_q_pre[elem], _mul_f32(r_q_pre[elem], TK.float32(SCALE))
                            )
                    shared_base = _local_scalar("int32", t_pre[0] * (K + 8) + k_start[0])
                    if not DISABLE_OUTPUT:
                        for elem in range(ELEMS_PER_LANE):
                            _shared_store_f32(s_q, shared_base[0] + elem, r_q_pre[elem])
                    for elem in range(ELEMS_PER_LANE):
                        _shared_store_f32(s_k, shared_base[0] + elem, r_k_pre[elem])
                    if SEQ_LEN > 2:
                        gate = _local_scalar(
                            "uint64",
                            _gate_pair(
                                a,
                                b_gate,
                                A_value[0],
                                dt_value[0],
                                (n[0] * SEQ_LEN + t_pre[0]) * NUM_V_HEADS + hv[0],
                            ),
                        )
                        with TK.If(lane[0] == 0), TK.Then():
                            _shared_store_f32x2(
                                s_gb,
                                t_pre[0] * 2,
                                TK.cuda.float2_x(gate[0]),
                                TK.cuda.float2_y(gate[0]),
                            )
                TK.cuda.cta_sync()
        for iter_index in range(ITERS_PER_GROUP):
            v_base = _local_scalar(
                "int32", v_tile[0] * TILE_V + warp[0] * ROWS_PER_GROUP + iter_index * ILP_ROWS
            )
            state_offset = _local_scalar(
                "int64", read_state_base[0] + TK.cast(v_base[0] * K + k_start[0], "int64")
            )
            r_h = TK.alloc_local((ILP_ROWS * ELEMS_PER_LANE,), "float32")
            _load_state_bf16x16(state, state_offset[0], r_h)
            r_q = TK.alloc_local((ELEMS_PER_LANE,), "float32", align=16)
            r_k = TK.alloc_local((ELEMS_PER_LANE,), "float32", align=16)
            gate_values = TK.alloc_local((2,), "float32", align=8)
            sums = TK.alloc_local((ILP_ROWS,), "float32")
            residuals = TK.alloc_local((ILP_ROWS,), "float32")
            output_sums = TK.alloc_local((ILP_ROWS,), "float32")
            with TK.serial(0, token_bound[0], unroll=4 if SEQ_LEN > 4 else False) as t:
                if SEQ_LEN > 1:
                    shared_base = _local_scalar("int32", t * (K + 8) + k_start[0])
                    if not DISABLE_OUTPUT:
                        _shared_load_f32x4(s_q, shared_base[0], r_q)
                    _shared_load_f32x4(s_k, shared_base[0], r_k)
                    if SEQ_LEN > 2:
                        _shared_load_f32x2(s_gb, t * 2, gate_values)
                    else:
                        gate = _local_scalar(
                            "uint64",
                            _gate_pair(
                                a,
                                b_gate,
                                A_value[0],
                                dt_value[0],
                                (n[0] * SEQ_LEN + t) * NUM_V_HEADS + hv[0],
                            ),
                        )
                        TK.ptx.mov.b32(gate_values[0], TK.cuda.float2_x(gate[0]))
                        TK.ptx.mov.b32(gate_values[1], TK.cuda.float2_y(gate[0]))
                else:
                    q_bits = TK.alloc_local((ELEMS_PER_LANE,), "uint16")
                    k_bits = TK.alloc_local((ELEMS_PER_LANE,), "uint16")
                    q_base = _local_scalar(
                        "int64",
                        TK.cast(n[0], "int64") * q_batch_stride
                        + TK.cast((t * NUM_HEADS + h[0]) * K + k_start[0], "int64"),
                    )
                    k_base = _local_scalar(
                        "int64",
                        TK.cast(n[0], "int64") * k_batch_stride
                        + TK.cast((t * NUM_HEADS + h[0]) * K + k_start[0], "int64"),
                    )
                    for elem in range(ELEMS_PER_LANE):
                        TK.ptx.mov.b16(q_bits[elem], _global_load_u16(q, q_base[0] + elem))
                    for elem in range(ELEMS_PER_LANE):
                        TK.ptx.mov.b16(k_bits[elem], _global_load_u16(k, k_base[0] + elem))
                    for elem in range(ELEMS_PER_LANE):
                        TK.ptx.mov.b32(r_q[elem], _bf16_to_f32(q_bits[elem]))
                        TK.ptx.mov.b32(r_k[elem], _bf16_to_f32(k_bits[elem]))
                    if USE_QK_L2NORM:
                        sum_q = _local_scalar("float32", TK.float32(0.0))
                        sum_k = _local_scalar("float32", TK.float32(0.0))
                        for elem in range(ELEMS_PER_LANE):
                            TK.assign(sum_q[0], _bf16_square_fma(q_bits[elem], sum_q[0]))
                            TK.assign(sum_k[0], _bf16_square_fma(k_bits[elem], sum_k[0]))
                        for delta_index in range(5):
                            delta = _local_scalar(
                                "int32", TK.shift_right(TK.int32(16), delta_index)
                            )
                            TK.assign(
                                sum_q[0],
                                _add_f32(
                                    sum_q[0],
                                    TK.cuda.__shfl_xor_sync(
                                        TK.uint32(4294967295), sum_q[0], delta[0], LANES_PER_GROUP
                                    ),
                                ),
                            )
                            TK.assign(
                                sum_k[0],
                                _add_f32(
                                    sum_k[0],
                                    TK.cuda.__shfl_xor_sync(
                                        TK.uint32(4294967295), sum_k[0], delta[0], LANES_PER_GROUP
                                    ),
                                ),
                            )
                        q_factor = _local_scalar(
                            "float32",
                            _mul_f32(
                                _rsqrt_f32(_add_f32(sum_q[0], TK.float32(1e-06))), TK.float32(SCALE)
                            ),
                        )
                        k_factor = _local_scalar(
                            "float32", _rsqrt_f32(_add_f32(sum_k[0], TK.float32(1e-06)))
                        )
                        for elem in range(ELEMS_PER_LANE):
                            TK.ptx.mov.b32(r_q[elem], _mul_f32(r_q[elem], q_factor[0]))
                            TK.ptx.mov.b32(r_k[elem], _mul_f32(r_k[elem], k_factor[0]))
                    else:
                        for elem in range(ELEMS_PER_LANE):
                            TK.ptx.mov.b32(r_q[elem], _mul_f32(r_q[elem], TK.float32(SCALE)))
                    gate = _local_scalar(
                        "uint64",
                        _gate_pair(
                            a,
                            b_gate,
                            A_value[0],
                            dt_value[0],
                            (n[0] * SEQ_LEN + t) * NUM_V_HEADS + hv[0],
                        ),
                    )
                    TK.ptx.mov.b32(gate_values[0], TK.cuda.float2_x(gate[0]))
                    TK.ptx.mov.b32(gate_values[1], TK.cuda.float2_y(gate[0]))
                g_value = _local_scalar("float32", gate_values[0])
                beta = _local_scalar("float32", gate_values[1])
                sum_lo = TK.alloc_local((ILP_ROWS,), "float32")
                sum_hi = TK.alloc_local((ILP_ROWS,), "float32")
                for row in range(ILP_ROWS):
                    TK.ptx.mov.b32(sum_lo[row], TK.float32(0.0))
                    TK.ptx.mov.b32(sum_hi[row], TK.float32(0.0))
                for pair in range(ELEMS_PER_LANE // 2):
                    for row in range(ILP_ROWS):
                        TK.ptx.mov.b32(
                            r_h[row * ELEMS_PER_LANE + pair * 2],
                            _mul_f32(r_h[row * ELEMS_PER_LANE + pair * 2], g_value[0]),
                        )
                        TK.ptx.mov.b32(
                            r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                            _mul_f32(r_h[row * ELEMS_PER_LANE + pair * 2 + 1], g_value[0]),
                        )
                    for row in range(ILP_ROWS):
                        packed = _local_scalar(
                            "uint64",
                            _packed_fma(
                                r_h[row * ELEMS_PER_LANE + pair * 2],
                                r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                                r_k[pair * 2],
                                r_k[pair * 2 + 1],
                                sum_lo[row],
                                sum_hi[row],
                            ),
                        )
                        TK.ptx.mov.b32(sum_lo[row], TK.cuda.float2_x(packed[0]))
                        TK.ptx.mov.b32(sum_hi[row], TK.cuda.float2_y(packed[0]))
                for row in range(ILP_ROWS):
                    TK.ptx.mov.b32(sums[row], _add_f32(sum_lo[row], sum_hi[row]))
                for delta_index in range(5):
                    delta = _local_scalar("int32", TK.shift_right(TK.int32(16), delta_index))
                    for row in range(ILP_ROWS):
                        TK.ptx.mov.b32(
                            sums[row],
                            _add_f32(
                                sums[row],
                                TK.cuda.__shfl_xor_sync(
                                    TK.uint32(4294967295), sums[row], delta[0], LANES_PER_GROUP
                                ),
                            ),
                        )
                v_input_base = _local_scalar(
                    "int64",
                    TK.cast(n[0], "int64") * v_batch_stride
                    + TK.cast((t * NUM_V_HEADS + hv[0]) * V + v_base[0], "int64"),
                )
                for row in range(ILP_ROWS):
                    v_bits = _local_scalar("uint16", _global_load_u16(v, v_input_base[0] + row))
                    TK.ptx.mov.b32(
                        residuals[row], _mul_f32(_mixed_sub_bf16_f32(v_bits[0], sums[row]), beta[0])
                    )
                output_lo = TK.alloc_local((ILP_ROWS,), "float32")
                output_hi = TK.alloc_local((ILP_ROWS,), "float32")
                for row in range(ILP_ROWS):
                    TK.ptx.mov.b32(output_lo[row], TK.float32(0.0))
                    TK.ptx.mov.b32(output_hi[row], TK.float32(0.0))
                for pair in range(ELEMS_PER_LANE // 2):
                    for row in range(ILP_ROWS):
                        packed = _local_scalar(
                            "uint64",
                            _packed_fma(
                                r_k[pair * 2],
                                r_k[pair * 2 + 1],
                                residuals[row],
                                residuals[row],
                                r_h[row * ELEMS_PER_LANE + pair * 2],
                                r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                            ),
                        )
                        TK.ptx.mov.b32(
                            r_h[row * ELEMS_PER_LANE + pair * 2], TK.cuda.float2_x(packed[0])
                        )
                        TK.ptx.mov.b32(
                            r_h[row * ELEMS_PER_LANE + pair * 2 + 1], TK.cuda.float2_y(packed[0])
                        )
                    if not DISABLE_OUTPUT:
                        for row in range(ILP_ROWS):
                            packed = _local_scalar(
                                "uint64",
                                _packed_fma(
                                    r_h[row * ELEMS_PER_LANE + pair * 2],
                                    r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                                    r_q[pair * 2],
                                    r_q[pair * 2 + 1],
                                    output_lo[row],
                                    output_hi[row],
                                ),
                            )
                            TK.ptx.mov.b32(output_lo[row], TK.cuda.float2_x(packed[0]))
                            TK.ptx.mov.b32(output_hi[row], TK.cuda.float2_y(packed[0]))
                if not DISABLE_OUTPUT:
                    for row in range(ILP_ROWS):
                        TK.ptx.mov.b32(output_sums[row], _add_f32(output_lo[row], output_hi[row]))
                if CACHE_INTERMEDIATE_STATES or PER_TOKEN_POOL_SCATTER:
                    state_write_base = TK.alloc_local((1,), "int64")
                    if PER_TOKEN_POOL_SCATTER:
                        scatter_slot = _local_scalar(
                            "int32", _global_load_s32(ssm_state_indices, n[0] * SEQ_LEN + t)
                        )
                        if PER_TOKEN_POOL_SCATTER_FLAT:
                            TK.assign(
                                state_write_base[0],
                                TK.cast(
                                    (TK.cast(scatter_slot[0], "int64") * NUM_V_HEADS + hv[0])
                                    * V
                                    * K
                                    + v_base[0] * K
                                    + k_start[0],
                                    "int64",
                                ),
                            )
                            _store_state_f32x16(intermediate, state_write_base[0], r_h)
                        else:
                            TK.assign(
                                state_write_base[0],
                                TK.cast(scatter_slot[0], "int64") * state_slot_stride
                                + TK.cast(hv[0] * V * K + v_base[0] * K + k_start[0], "int64"),
                            )
                            _store_state_f32x16(state, state_write_base[0], r_h)
                    else:
                        TK.assign(
                            state_write_base[0],
                            TK.cast(
                                (((n[0] * SEQ_LEN + t) * NUM_V_HEADS + hv[0]) * V + v_base[0]) * K
                                + k_start[0],
                                "int64",
                            ),
                        )
                        _store_state_f32x16(intermediate, state_write_base[0], r_h)
                if not DISABLE_OUTPUT:
                    for delta_index in range(5):
                        delta = _local_scalar("int32", TK.shift_right(TK.int32(16), delta_index))
                        for row in range(ILP_ROWS):
                            TK.ptx.mov.b32(
                                output_sums[row],
                                _add_f32(
                                    output_sums[row],
                                    TK.cuda.__shfl_xor_sync(
                                        TK.uint32(4294967295),
                                        output_sums[row],
                                        delta[0],
                                        LANES_PER_GROUP,
                                    ),
                                ),
                            )
                    output_base = _local_scalar(
                        "int64",
                        TK.cast(
                            ((n[0] * SEQ_LEN + t) * NUM_V_HEADS + hv[0]) * V + v_base[0], "int64"
                        ),
                    )
                    with TK.If(lane[0] == 0), TK.Then():
                        output_bits = TK.alloc_local((ILP_ROWS,), "uint16")
                        for row in range(ILP_ROWS):
                            TK.ptx.mov.b16(output_bits[row], _f32_to_bf16(output_sums[row]))
                        _global_store_u16x4(output, output_base[0], output_bits)
            if not DISABLE_STATE_UPDATE and (not (PER_TOKEN_POOL_SCATTER and SAME_POOL)):
                final_offset = _local_scalar(
                    "int64", write_state_base[0] + TK.cast(v_base[0] * K + k_start[0], "int64")
                )
                _store_state_f32x16(state, final_offset[0], r_h)

    return gdn_decode_bf16_ilp4.func


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
    if seq_len < 1 or seq_len > 8:
        raise ValueError("BF16 ILP4 port requires seq_len in [1, 8]")
    if (num_heads, num_v_heads) not in _HEAD_CONFIGS:
        raise ValueError(f"unsupported Qwen3-Next TP head pair {(num_heads, num_v_heads)}")
    if num_v_heads % num_heads:
        raise ValueError("num_v_heads must be divisible by num_heads")
    if tile_v not in (16, 32, 64) or V % tile_v:
        raise ValueError("BF16 ILP4 port requires tile_v in {16, 32, 64}")
    if (tile_v // NUM_GROUPS) % ILP_ROWS:
        raise ValueError("tile_v is incompatible with the four-warp ILP4 mapping")
    if int(config.get("recovery_steps", 0)) != 0:
        raise ValueError("FlashInfer's BF16 ILP4 path does not support recovery_steps")

    cache = bool(config.get("cache_intermediate_states", False))
    disable_update = bool(config.get("disable_state_update", False))
    scatter = bool(config.get("per_token_pool_scatter", False))
    if scatter and (cache or disable_update or seq_len < 2):
        raise ValueError("per-token scatter requires update, no dense cache, and T >= 2")
    if config.get("per_request_accepted_steps", False):
        accepted = tuple(int(value) for value in config.get("accepted_steps", (0,)))
        if any(value < 0 or value >= seq_len for value in accepted):
            raise ValueError("accepted_steps values must be in [0, seq_len)")


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
    pool_factor = int(config.get("pool_factor_override", _pool_factor(config)))
    intermediate_batch_stride = 0
    if cache:
        intermediate_batch_stride = seq_len * num_v_heads * V * K
    elif scatter_flat:
        intermediate_batch_stride = pool_factor * num_v_heads * V * K

    cosize_qk_bytes = 4 * ((seq_len - 1) * (K + 8) + K)
    shared_bytes = 128 if seq_len == 1 else 1096 * seq_len + 128
    return _make_gdn_decode_bf16_ilp4(
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
        SHARED_BYTES=shared_bytes,
        S_K_BYTE_OFFSET=cosize_qk_bytes,
        S_GB_BYTE_OFFSET=2 * cosize_qk_bytes,
        USE_QK_L2NORM=bool(config.get("use_qk_l2norm", True)),
        DISABLE_STATE_UPDATE=bool(config.get("disable_state_update", False)),
        CACHE_INTERMEDIATE_STATES=cache,
        SAME_POOL=bool(config.get("same_pool", True)),
        DISABLE_OUTPUT=bool(config.get("disable_output", False)),
        PER_REQUEST_ACCEPTED_STEPS=bool(config.get("per_request_accepted_steps", False)),
        PER_TOKEN_POOL_SCATTER=scatter,
        PER_TOKEN_POOL_SCATTER_FLAT=scatter_flat,
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
        raise SkipTest("CUDA is required for BF16 ILP4 GDN decode")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(f"BF16 ILP4 GDN decode requires SM100, got {capability}")
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
def _load_oracle():
    import flashinfer.gdn_kernels.gdn_decode_bf16_state as source_module

    return source_module.gated_delta_rule_mtp


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
    config = case["config"]
    oracle = _load_oracle()
    return oracle(
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
            case["read_indices"] if bool(config.get("same_pool", True)) else case["write_indices"]
        ),
        intermediate_states_buffer=(
            case["source_intermediate"]
            if bool(config.get("cache_intermediate_states", False))
            else None
        ),
        accepted_steps=(
            case["accepted_steps"] if config.get("per_request_accepted_steps", False) else None
        ),
        ssm_state_indices=(
            case["ssm_state_indices"] if config.get("per_token_pool_scatter", False) else None
        ),
        disable_state_update=bool(config.get("disable_state_update", False)),
        use_qk_l2norm_in_kernel=bool(config.get("use_qk_l2norm", True)),
        scale=SCALE,
        output=case["source_output"],
        disable_output=bool(config.get("disable_output", False)),
        recovery_steps=0,
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
    """Compile the selected ILP4 specialization before CUDA setup."""
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
        executable(*args)
        _run_reference(case)
        torch.cuda.synchronize(case["tirx_state"].device)
        _assert_case_close(case)
        for _ in range(2):
            _run_reference(case)
        torch.cuda.synchronize(case["source_state"].device)

        def launch():
            _run_reference(case)

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
