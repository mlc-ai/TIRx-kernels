# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM100 BF16 wide-vector multi-token GDN decode kernel.

Upstream source: flashinfer/gdn_kernels/gdn_decode_bf16_state.py.
"""

import functools
import math
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.tirx_lite as txl
from tirx_kernels.bench.runner import bench

KERNEL_META = {
    "name": "gdn_decode_bf16_wide_vec_mtp",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "f2e04400e330fb2debe0bf8730d9424a1d37927f",
            },
            "import": "flashinfer",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
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


def _shfl_bfly_f32(value, lane_xor):
    """``shfl.sync.bfly.b32`` at width 32: clamp/segmask 31, full member mask.

    DPS: the destination pins the warp collective to the call site, so the
    shuffle is emitted once here rather than re-emitted at every textual use
    of the returned value.
    """
    shfl_bfly = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.bfly.b32(
        shfl_bfly,
        txl.reinterpret("uint32", value),
        txl.cast(lane_xor, "uint32"),
        txl.uint32(31),
        txl.uint32(4294967295),
    )
    return txl.reinterpret("float32", shfl_bfly)


def _local_scalar(dtype: str, value):
    out = txl.alloc_local((1,), dtype)
    txl.assign(out[0], value)
    return out


def _shared_load_f32x2(buffer, index, values):
    words = txl.alloc_local((2,), "uint32", align=8)
    txl.ptx.ld.shared.v2.b32(words[0], words[1], buffer.ptr_to([index]))
    txl.ptx.mov.b32(values[0], txl.reinterpret("float32", words[0]))
    txl.ptx.mov.b32(values[1], txl.reinterpret("float32", words[1]))


def _load_state_bf16x32(buffer, index, values):
    words = txl.alloc_local((ILP_ROWS * 4,), "uint32", align=16)
    with txl.unroll(ILP_ROWS) as row:
        word_offset = txl.local_scalar("int32", init=row * 4)
        txl.ptx["ld.global.L1::evict_first.v4.b32"](
            words[word_offset],
            words[word_offset + 1],
            words[word_offset + 2],
            words[word_offset + 3],
            buffer.ptr_to([index + row * K]),
        )
    with txl.unroll(4) as pair:
        with txl.unroll(ILP_ROWS) as row:
            word = txl.local_scalar("uint32", init=words[row * 4 + pair])
            txl.ptx.mov.b32(
                values[row * ELEMS_PER_LANE + pair * 2],
                txl.cuda.uint_as_float(txl.shift_left(word, txl.uint32(16))),
            )
            txl.ptx.mov.b32(
                values[row * ELEMS_PER_LANE + pair * 2 + 1],
                txl.cuda.uint_as_float(txl.bitwise_and(word, txl.uint32(0xFFFF0000))),
            )


def _store_state_f32x32(buffer, index, values):
    words = txl.alloc_local((ILP_ROWS * 4,), "uint32", align=16)
    with txl.unroll(4) as pair:
        with txl.unroll(ILP_ROWS) as row:
            txl.ptx.mov.b32(
                words[row * 4 + pair],
                txl.cuda.float22bfloat162_rn(
                    values[row * ELEMS_PER_LANE + pair * 2],
                    values[row * ELEMS_PER_LANE + pair * 2 + 1],
                ),
            )
    with txl.unroll(ILP_ROWS) as row:
        word_offset = txl.local_scalar("int32", init=row * 4)
        txl.ptx.st.global_.v4.b32(
            buffer.ptr_to([index + row * K]),
            words[word_offset],
            words[word_offset + 1],
            words[word_offset + 2],
            words[word_offset + 3],
        )


def _packed_fma(lhs0, lhs1, rhs0, rhs1, acc0, acc1):
    out = txl.local_scalar("uint64")
    txl.ptx.fma.rn.f32x2(
        out,
        txl.cuda.make_float2(lhs0, lhs1),
        txl.cuda.make_float2(rhs0, rhs1),
        txl.cuda.make_float2(acc0, acc1),
    )
    return out


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


def _make_gdn_decode_bf16_wide_vec_mtp(
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
    RECOVERY_STEPS,
    PER_REQUEST_ACCEPTED_STEPS,
    PER_TOKEN_POOL_SCATTER,
    PER_TOKEN_POOL_SCATTER_FLAT,
):
    @txl.kernel(
        warps=NUM_WARPS, arch="sm_100a", grid=lambda p: p["batch"] * NUM_V_HEADS * NUM_V_TILES
    )
    def gdn_decode_bf16_wide_vec_mtp(
        state: txl.gptr[txl.bf16],
        intermediate: txl.gptr[txl.bf16],
        A_log: txl.gptr[txl.f32],
        a: txl.gptr[txl.bf16],
        dt_bias: txl.gptr[txl.f32],
        q: txl.gptr[txl.bf16],
        k: txl.gptr[txl.bf16],
        v: txl.gptr[txl.bf16],
        b_gate: txl.gptr[txl.bf16],
        output: txl.gptr[txl.bf16],
        read_indices: txl.gptr[txl.i32],
        write_indices: txl.gptr[txl.i32],
        accepted_steps: txl.gptr[txl.i32],
        ssm_state_indices: txl.gptr[txl.i32],
        state_slot_stride: txl.i64,
        q_batch_stride: txl.i64,
        k_batch_stride: txl.i64,
        v_batch_stride: txl.i64,
        batch: txl.i32,
    ):
        smem = txl.smem_pool()
        s_q = smem.alloc((S_K_BYTE_OFFSET // 4,), txl.f32, align=16)
        s_k = smem.alloc(((S_GB_BYTE_OFFSET - S_K_BYTE_OFFSET) // 4,), txl.f32, align=16)
        s_gb = smem.alloc((SEQ_LEN * 2,), txl.f32, align=16)
        smem.commit(SHARED_BYTES)
        linear_cta = txl.cta_id()
        tid = txl.thread_id()
        warp = txl.warp_id()
        lane_in_warp = txl.lane_id()
        group = _local_scalar("int32", tid // LANES_PER_GROUP)
        lane = _local_scalar("int32", tid % LANES_PER_GROUP)
        k_start = _local_scalar("int32", lane[0] * ELEMS_PER_LANE)
        v_tile = _local_scalar("int32", linear_cta % NUM_V_TILES)
        cta_head = _local_scalar("int32", linear_cta // NUM_V_TILES)
        hv = _local_scalar("int32", cta_head[0] % NUM_V_HEADS)
        n = _local_scalar("int32", cta_head[0] // NUM_V_HEADS)
        h = _local_scalar("int32", hv[0] // (NUM_V_HEADS // NUM_HEADS))
        read_slot_raw = txl.local_scalar("int32")
        txl.ptx.ld.global_.s32(read_slot_raw, read_indices.ptr_to([n[0]]))
        A_value = txl.local_scalar("float32")
        txl.ptx.ld.global_.b32(A_value, A_log.ptr_to([hv[0]]))
        dt_value = txl.local_scalar("float32")
        txl.ptx.ld.global_.b32(dt_value, dt_bias.ptr_to([hv[0]]))
        _max = txl.local_scalar("int32")
        txl.ptx["max.s32"](_max, read_slot_raw, txl.int32(0))
        read_slot = _local_scalar("int32", _max)
        write_slot = _local_scalar("int32", read_slot[0])
        if not SAME_POOL:
            _ldg32 = txl.local_scalar("int32")
            txl.ptx.ld.global_.s32(_ldg32, write_indices.ptr_to([n[0]]))
            txl.ptx["max.s32"](write_slot[0], _ldg32, txl.int32(0))
        read_state_base = _local_scalar(
            "int64",
            txl.cast(read_slot[0], "int64") * state_slot_stride + txl.cast(hv[0] * V * K, "int64"),
        )
        write_state_base = _local_scalar("int64", read_state_base[0])
        if not SAME_POOL:
            txl.assign(
                write_state_base[0],
                txl.cast(write_slot[0], "int64") * state_slot_stride
                + txl.cast(hv[0] * V * K, "int64"),
            )
        accepted = _local_scalar("int32", txl.int32(SEQ_LEN - 1))
        if PER_REQUEST_ACCEPTED_STEPS:
            txl.ptx.ld.global_.s32(accepted[0], accepted_steps.ptr_to([n[0]]))
        phase_a_bound = _local_scalar("int32", txl.int32(RECOVERY_STEPS))
        phase_b_begin = _local_scalar("int32", txl.int32(RECOVERY_STEPS))
        phase_b_bound = _local_scalar("int32", txl.int32(SEQ_LEN - RECOVERY_STEPS))
        if (
            PER_REQUEST_ACCEPTED_STEPS
            and (not DISABLE_OUTPUT)
            and (not DISABLE_STATE_UPDATE)
            and (not PER_TOKEN_POOL_SCATTER)
        ):
            txl.assign(phase_a_bound[0], accepted[0] + txl.int32(1))
            txl.assign(phase_b_begin[0], accepted[0] + txl.int32(1))
            txl.assign(phase_b_bound[0], txl.int32(SEQ_LEN) - accepted[0] - txl.int32(1))
        elif PER_REQUEST_ACCEPTED_STEPS:
            txl.assign(phase_b_bound[0], accepted[0] + txl.int32(1 - RECOVERY_STEPS))
        prefetch_words = txl.alloc_local((ILP_ROWS * 4,), "uint32", align=16)
        if TILE_V == 32:
            pre_v_base = _local_scalar("int32", v_tile[0] * TILE_V + group[0] * ROWS_PER_GROUP)
            for row in range(ILP_ROWS):
                word_offset = _local_scalar("int32", row * 4)
                prefetch_offset = _local_scalar(
                    "int64",
                    read_state_base[0] + txl.cast((pre_v_base[0] + row) * K + k_start[0], "int64"),
                )
                txl.ptx["ld.global.L1::evict_first.v4.b32"](
                    prefetch_words[word_offset[0]],
                    prefetch_words[word_offset[0] + 1],
                    prefetch_words[word_offset[0] + 2],
                    prefetch_words[word_offset[0] + 3],
                    state.ptr_to([prefetch_offset[0]]),
                )
        for pass_index in range((SEQ_LEN + NUM_WARPS - 1) // NUM_WARPS):
            t_pre = _local_scalar("int32", pass_index * NUM_WARPS + warp)
            member_pre = _local_scalar("int32", lane_in_warp % LANES_PER_GROUP)
            k_pre = _local_scalar("int32", member_pre[0] * ELEMS_PER_LANE)
            with txl.If(t_pre[0] < SEQ_LEN), txl.Then():
                q_bits = txl.alloc_local((ELEMS_PER_LANE,), "uint16")
                k_bits = txl.alloc_local((ELEMS_PER_LANE,), "uint16")
                r_q = txl.alloc_local((ELEMS_PER_LANE,), "float32")
                r_k = txl.alloc_local((ELEMS_PER_LANE,), "float32")
                if not DISABLE_OUTPUT and (pass_index + 1) * NUM_WARPS > RECOVERY_STEPS:
                    q_base = _local_scalar(
                        "int64",
                        txl.cast(n[0], "int64") * q_batch_stride
                        + txl.cast((t_pre[0] * NUM_HEADS + h[0]) * K + k_pre[0], "int64"),
                    )
                    for i in range(ELEMS_PER_LANE):
                        txl.ptx.ld.global_.b16(q_bits[i], q.ptr_to([q_base[0] + i]))
                        txl.ptx.cvt.f32.bf16(r_q[i], txl.cast(q_bits[i], "uint16"))
                k_base = _local_scalar(
                    "int64",
                    txl.cast(n[0], "int64") * k_batch_stride
                    + txl.cast((t_pre[0] * NUM_HEADS + h[0]) * K + k_pre[0], "int64"),
                )
                for i in range(ELEMS_PER_LANE):
                    txl.ptx.ld.global_.b16(k_bits[i], k.ptr_to([k_base[0] + i]))
                    txl.ptx.cvt.f32.bf16(r_k[i], txl.cast(k_bits[i], "uint16"))
                if USE_QK_L2NORM:
                    sum_k = _local_scalar("float32", txl.float32(0.0))
                    for i in range(ELEMS_PER_LANE):
                        txl.ptx.fma.rn.f32.bf16(
                            sum_k[0],
                            txl.cast(k_bits[i], "uint16"),
                            txl.cast(k_bits[i], "uint16"),
                            sum_k[0],
                        )
                    for delta_index in range(4):
                        delta = _local_scalar("int32", txl.shift_right(txl.int32(8), delta_index))
                        txl.ptx["add.f32"](sum_k[0], sum_k[0], _shfl_bfly_f32(sum_k[0], delta[0]))
                    _add = txl.local_scalar("float32")
                    txl.ptx["add.f32"](_add, sum_k[0], txl.float32(1e-06))
                    _rsqrt = txl.local_scalar("float32")
                    txl.ptx["rsqrt.approx.ftz.f32"](_rsqrt, _add)
                    k_factor = _local_scalar("float32", _rsqrt)
                    for i in range(ELEMS_PER_LANE):
                        txl.ptx["mul.f32"](r_k[i], r_k[i], k_factor[0])
                    if not DISABLE_OUTPUT and (pass_index + 1) * NUM_WARPS > RECOVERY_STEPS:
                        sum_q = _local_scalar("float32", txl.float32(0.0))
                        for i in range(ELEMS_PER_LANE):
                            txl.ptx.fma.rn.f32.bf16(
                                sum_q[0],
                                txl.cast(q_bits[i], "uint16"),
                                txl.cast(q_bits[i], "uint16"),
                                sum_q[0],
                            )
                        for delta_index in range(4):
                            delta = _local_scalar("int32", txl.shift_right(txl.int32(8), delta_index))
                            txl.ptx["add.f32"](
                                sum_q[0], sum_q[0], _shfl_bfly_f32(sum_q[0], delta[0])
                            )
                        _add2 = txl.local_scalar("float32")
                        txl.ptx["add.f32"](_add2, sum_q[0], txl.float32(1e-06))
                        _rsqrt2 = txl.local_scalar("float32")
                        txl.ptx["rsqrt.approx.ftz.f32"](_rsqrt2, _add2)
                        _mul = txl.local_scalar("float32")
                        txl.ptx["mul.f32"](_mul, _rsqrt2, txl.float32(SCALE))
                        q_factor = _local_scalar("float32", _mul)
                        for i in range(ELEMS_PER_LANE):
                            txl.ptx["mul.f32"](r_q[i], r_q[i], q_factor[0])
                elif not DISABLE_OUTPUT and (pass_index + 1) * NUM_WARPS > RECOVERY_STEPS:
                    for i in range(ELEMS_PER_LANE):
                        txl.ptx["mul.f32"](r_q[i], r_q[i], txl.float32(SCALE))
                shared_base = _local_scalar("int32", t_pre[0] * (K + 8) + k_pre[0])
                if not DISABLE_OUTPUT and (pass_index + 1) * NUM_WARPS > RECOVERY_STEPS:
                    for i in range(ELEMS_PER_LANE):
                        txl.ptx.st.shared.b32(
                            s_q.ptr_to([shared_base[0] + i]), txl.reinterpret("uint32", r_q[i])
                        )
                for i in range(ELEMS_PER_LANE):
                    txl.ptx.st.shared.b32(
                        s_k.ptr_to([shared_base[0] + i]), txl.reinterpret("uint32", r_k[i])
                    )
                a_bits = txl.local_scalar("uint16")
                txl.ptx.ld.global_.b16(
                    a_bits, a.ptr_to([(n[0] * SEQ_LEN + t_pre[0]) * NUM_V_HEADS + hv[0]])
                )
                _addbf = txl.local_scalar("float32")
                txl.ptx.add.rn.f32.bf16(_addbf, txl.cast(a_bits, "uint16"), dt_value)
                x_value = _local_scalar("float32", _addbf)
                _mul2 = txl.local_scalar("float32")
                txl.ptx["mul.f32"](_mul2, x_value[0], txl.float32(LOG2_E))
                _exp2 = txl.local_scalar("float32")
                txl.ptx["ex2.approx.ftz.f32"](_exp2, _mul2)
                softplus_exp = _local_scalar("float32", _exp2)
                _add3 = txl.local_scalar("float32")
                txl.ptx["add.f32"](_add3, txl.float32(1.0), softplus_exp[0])
                _log2 = txl.local_scalar("float32")
                txl.ptx["lg2.approx.ftz.f32"](_log2, _add3)
                _mul3 = txl.local_scalar("float32")
                txl.ptx["mul.f32"](_mul3, _log2, txl.float32(LN_2))
                softplus_value = _local_scalar("float32", _mul3)
                use_softplus = _local_scalar(
                    "float32",
                    txl.if_then_else(
                        x_value[0] <= txl.float32(20.0), txl.float32(1.0), txl.float32(0.0)
                    ),
                )
                _sub = txl.local_scalar("float32")
                txl.ptx["sub.f32"](_sub, txl.float32(1.0), use_softplus[0])
                _mul4 = txl.local_scalar("float32")
                txl.ptx["mul.f32"](_mul4, x_value[0], _sub)
                _fma = txl.local_scalar("float32")
                txl.ptx["fma.rn.f32"](_fma, softplus_value[0], use_softplus[0], _mul4)
                softplus_x = _local_scalar("float32", _fma)
                _mul5 = txl.local_scalar("float32")
                txl.ptx["mul.f32"](_mul5, A_value, txl.float32(LOG2_E))
                _exp2_2 = txl.local_scalar("float32")
                txl.ptx["ex2.approx.ftz.f32"](_exp2_2, _mul5)
                exp_A = _local_scalar("float32", _exp2_2)
                _sub2 = txl.local_scalar("float32")
                txl.ptx["sub.f32"](_sub2, txl.float32(0.0), exp_A[0])
                _mul6 = txl.local_scalar("float32")
                txl.ptx["mul.f32"](_mul6, _sub2, softplus_x[0])
                gate_exponent = _local_scalar("float32", _mul6)
                with txl.If(lane_in_warp == 0), txl.Then():
                    b_bits = txl.local_scalar("uint16")
                    txl.ptx.ld.global_.b16(
                        b_bits, b_gate.ptr_to([(n[0] * SEQ_LEN + t_pre[0]) * NUM_V_HEADS + hv[0]])
                    )
                    b_value = txl.local_scalar("float32")
                    txl.ptx.cvt.f32.bf16(b_value, txl.cast(b_bits, "uint16"))
                    _mul7 = txl.local_scalar("float32")
                    txl.ptx["mul.f32"](_mul7, b_value, txl.float32(-LOG2_E))
                    _exp2_3 = txl.local_scalar("float32")
                    txl.ptx["ex2.approx.ftz.f32"](_exp2_3, _mul7)
                    _add4 = txl.local_scalar("float32")
                    txl.ptx["add.f32"](_add4, txl.float32(1.0), _exp2_3)
                    _rcp = txl.local_scalar("float32")
                    txl.ptx["rcp.rn.f32"](_rcp, _add4)
                    beta = _local_scalar("float32", _rcp)
                    _mul8 = txl.local_scalar("float32")
                    txl.ptx["mul.f32"](_mul8, gate_exponent[0], txl.float32(LOG2_E))
                    _exp2_4 = txl.local_scalar("float32")
                    txl.ptx["ex2.approx.ftz.f32"](_exp2_4, _mul8)
                    g_value = _local_scalar("float32", _exp2_4)
                    txl.ptx.st.shared.v2.b32(
                        s_gb.ptr_to([t_pre[0] * 2]),
                        txl.reinterpret("uint32", g_value[0]),
                        txl.reinterpret("uint32", beta[0]),
                    )
            txl.cuda.cta_sync()
        for iter_index in range(ITERS_PER_GROUP):
            v_base = _local_scalar(
                "int32", v_tile[0] * TILE_V + group[0] * ROWS_PER_GROUP + iter_index * ILP_ROWS
            )
            read_offset = _local_scalar(
                "int64", read_state_base[0] + txl.cast(v_base[0] * K + k_start[0], "int64")
            )
            r_h = txl.alloc_local((ILP_ROWS * ELEMS_PER_LANE,), "float32")
            if TILE_V == 32:
                for pair in range(4):
                    for row in range(ILP_ROWS):
                        word = _local_scalar("uint32", prefetch_words[row * 4 + pair])
                        txl.ptx.mov.b32(
                            r_h[row * ELEMS_PER_LANE + pair * 2],
                            txl.cuda.uint_as_float(txl.shift_left(word[0], txl.uint32(16))),
                        )
                        txl.ptx.mov.b32(
                            r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                            txl.cuda.uint_as_float(txl.bitwise_and(word[0], txl.uint32(4294901760))),
                        )
            else:
                _load_state_bf16x32(state, read_offset[0], r_h)
            r_k_main = txl.alloc_local((ELEMS_PER_LANE,), "float32")
            gate_pair = txl.alloc_local((2,), "float32", align=8)
            sums = txl.alloc_local((ILP_ROWS,), "float32")
            values = txl.alloc_local((ILP_ROWS,), "float32")
            with txl.serial(0, phase_a_bound[0], unroll=False) as t:
                _shared_load_f32x2(s_gb, t * 2, gate_pair)
                g_value = _local_scalar("float32", gate_pair[0])
                beta = _local_scalar("float32", gate_pair[1])
                for row in range(ILP_ROWS):
                    txl.ptx.mov.b32(sums[row], txl.float32(0.0))
                for pair in range(ELEMS_PER_LANE // 2):
                    _lds32 = txl.local_scalar("uint32")
                    txl.ptx.ld.shared.b32(_lds32, s_k.ptr_to([t * (K + 8) + k_start[0] + pair * 2]))
                    txl.ptx.mov.b32(r_k_main[pair * 2], txl.reinterpret("float32", _lds32))
                    _lds32_2 = txl.local_scalar("uint32")
                    txl.ptx.ld.shared.b32(
                        _lds32_2, s_k.ptr_to([t * (K + 8) + k_start[0] + pair * 2 + 1])
                    )
                    txl.ptx.mov.b32(r_k_main[pair * 2 + 1], txl.reinterpret("float32", _lds32_2))
                    for row in range(ILP_ROWS):
                        pair_value = _local_scalar(
                            "uint64",
                            _packed_fma(
                                r_h[row * ELEMS_PER_LANE + pair * 2],
                                r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                                g_value[0],
                                g_value[0],
                                txl.float32(0.0),
                                txl.float32(0.0),
                            ),
                        )
                        txl.ptx.mov.b32(
                            r_h[row * ELEMS_PER_LANE + pair * 2], txl.cuda.float2_x(pair_value[0])
                        )
                        txl.ptx.mov.b32(
                            r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                            txl.cuda.float2_y(pair_value[0]),
                        )
                        txl.ptx["fma.rn.f32"](
                            sums[row],
                            r_h[row * ELEMS_PER_LANE + pair * 2],
                            r_k_main[pair * 2],
                            sums[row],
                        )
                        txl.ptx["fma.rn.f32"](
                            sums[row],
                            r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                            r_k_main[pair * 2 + 1],
                            sums[row],
                        )
                for delta_index in range(4):
                    delta = _local_scalar("int32", txl.shift_right(txl.int32(8), delta_index))
                    for row in range(ILP_ROWS):
                        txl.ptx["add.f32"](sums[row], sums[row], _shfl_bfly_f32(sums[row], delta[0]))
                v_base_input = _local_scalar(
                    "int64",
                    txl.cast(n[0], "int64") * v_batch_stride
                    + txl.cast((t * NUM_V_HEADS + hv[0]) * V + v_base[0], "int64"),
                )
                for row in range(ILP_ROWS):
                    v_bits = txl.alloc_local((1,), "uint16")
                    txl.ptx.ld.global_.b16(v_bits[0], v.ptr_to([v_base_input[0] + row]))
                    _subbf = txl.local_scalar("float32")
                    txl.ptx.sub.rn.f32.bf16(_subbf, txl.cast(v_bits[0], "uint16"), sums[row])
                    txl.ptx["mul.f32"](values[row], _subbf, beta[0])
                for pair in range(ELEMS_PER_LANE // 2):
                    for row in range(ILP_ROWS):
                        pair_value = _local_scalar(
                            "uint64",
                            _packed_fma(
                                r_k_main[pair * 2],
                                r_k_main[pair * 2 + 1],
                                values[row],
                                values[row],
                                r_h[row * ELEMS_PER_LANE + pair * 2],
                                r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                            ),
                        )
                        txl.ptx.mov.b32(
                            r_h[row * ELEMS_PER_LANE + pair * 2], txl.cuda.float2_x(pair_value[0])
                        )
                        txl.ptx.mov.b32(
                            r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                            txl.cuda.float2_y(pair_value[0]),
                        )
            if (
                (
                    RECOVERY_STEPS > 0
                    or (
                        PER_REQUEST_ACCEPTED_STEPS
                        and (not DISABLE_OUTPUT)
                        and (not DISABLE_STATE_UPDATE)
                        and (not PER_TOKEN_POOL_SCATTER)
                    )
                )
                and (not DISABLE_STATE_UPDATE)
                and (not CACHE_INTERMEDIATE_STATES)
            ):
                write_offset = _local_scalar(
                    "int64", write_state_base[0] + txl.cast(v_base[0] * K + k_start[0], "int64")
                )
                _store_state_f32x32(state, write_offset[0], r_h)
            r_q_main = txl.alloc_local((ELEMS_PER_LANE,), "float32")
            output_sums = txl.alloc_local((ILP_ROWS,), "float32")
            with txl.serial(0, phase_b_bound[0], unroll=False) as t_offset:
                t = _local_scalar("int32", phase_b_begin[0] + t_offset)
                _shared_load_f32x2(s_gb, t[0] * 2, gate_pair)
                g_value = _local_scalar("float32", gate_pair[0])
                beta = _local_scalar("float32", gate_pair[1])
                for row in range(ILP_ROWS):
                    txl.ptx.mov.b32(sums[row], txl.float32(0.0))
                for pair in range(ELEMS_PER_LANE // 2):
                    _lds32_3 = txl.local_scalar("uint32")
                    txl.ptx.ld.shared.b32(
                        _lds32_3, s_k.ptr_to([t[0] * (K + 8) + k_start[0] + pair * 2])
                    )
                    txl.ptx.mov.b32(r_k_main[pair * 2], txl.reinterpret("float32", _lds32_3))
                    _lds32_4 = txl.local_scalar("uint32")
                    txl.ptx.ld.shared.b32(
                        _lds32_4, s_k.ptr_to([t[0] * (K + 8) + k_start[0] + pair * 2 + 1])
                    )
                    txl.ptx.mov.b32(r_k_main[pair * 2 + 1], txl.reinterpret("float32", _lds32_4))
                    for row in range(ILP_ROWS):
                        pair_value = _local_scalar(
                            "uint64",
                            _packed_fma(
                                r_h[row * ELEMS_PER_LANE + pair * 2],
                                r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                                g_value[0],
                                g_value[0],
                                txl.float32(0.0),
                                txl.float32(0.0),
                            ),
                        )
                        txl.ptx.mov.b32(
                            r_h[row * ELEMS_PER_LANE + pair * 2], txl.cuda.float2_x(pair_value[0])
                        )
                        txl.ptx.mov.b32(
                            r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                            txl.cuda.float2_y(pair_value[0]),
                        )
                        txl.ptx["fma.rn.f32"](
                            sums[row],
                            r_h[row * ELEMS_PER_LANE + pair * 2],
                            r_k_main[pair * 2],
                            sums[row],
                        )
                        txl.ptx["fma.rn.f32"](
                            sums[row],
                            r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                            r_k_main[pair * 2 + 1],
                            sums[row],
                        )
                for delta_index in range(4):
                    delta = _local_scalar("int32", txl.shift_right(txl.int32(8), delta_index))
                    for row in range(ILP_ROWS):
                        txl.ptx["add.f32"](sums[row], sums[row], _shfl_bfly_f32(sums[row], delta[0]))
                v_base_input = _local_scalar(
                    "int64",
                    txl.cast(n[0], "int64") * v_batch_stride
                    + txl.cast((t[0] * NUM_V_HEADS + hv[0]) * V + v_base[0], "int64"),
                )
                for row in range(ILP_ROWS):
                    v_bits = txl.alloc_local((1,), "uint16")
                    txl.ptx.ld.global_.b16(v_bits[0], v.ptr_to([v_base_input[0] + row]))
                    _subbf2 = txl.local_scalar("float32")
                    txl.ptx.sub.rn.f32.bf16(_subbf2, txl.cast(v_bits[0], "uint16"), sums[row])
                    txl.ptx["mul.f32"](values[row], _subbf2, beta[0])
                    txl.ptx.mov.b32(output_sums[row], txl.float32(0.0))
                for pair in range(ELEMS_PER_LANE // 2):
                    if not DISABLE_OUTPUT:
                        _lds32_5 = txl.local_scalar("uint32")
                        txl.ptx.ld.shared.b32(
                            _lds32_5, s_q.ptr_to([t[0] * (K + 8) + k_start[0] + pair * 2])
                        )
                        txl.ptx.mov.b32(r_q_main[pair * 2], txl.reinterpret("float32", _lds32_5))
                        _lds32_6 = txl.local_scalar("uint32")
                        txl.ptx.ld.shared.b32(
                            _lds32_6, s_q.ptr_to([t[0] * (K + 8) + k_start[0] + pair * 2 + 1])
                        )
                        txl.ptx.mov.b32(r_q_main[pair * 2 + 1], txl.reinterpret("float32", _lds32_6))
                    for row in range(ILP_ROWS):
                        pair_value = _local_scalar(
                            "uint64",
                            _packed_fma(
                                r_k_main[pair * 2],
                                r_k_main[pair * 2 + 1],
                                values[row],
                                values[row],
                                r_h[row * ELEMS_PER_LANE + pair * 2],
                                r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                            ),
                        )
                        txl.ptx.mov.b32(
                            r_h[row * ELEMS_PER_LANE + pair * 2], txl.cuda.float2_x(pair_value[0])
                        )
                        txl.ptx.mov.b32(
                            r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                            txl.cuda.float2_y(pair_value[0]),
                        )
                        if not DISABLE_OUTPUT:
                            txl.ptx["fma.rn.f32"](
                                output_sums[row],
                                r_h[row * ELEMS_PER_LANE + pair * 2],
                                r_q_main[pair * 2],
                                output_sums[row],
                            )
                            txl.ptx["fma.rn.f32"](
                                output_sums[row],
                                r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                                r_q_main[pair * 2 + 1],
                                output_sums[row],
                            )
                if not DISABLE_OUTPUT:
                    for delta_index in range(4):
                        delta = _local_scalar("int32", txl.shift_right(txl.int32(8), delta_index))
                        for row in range(ILP_ROWS):
                            txl.ptx["add.f32"](
                                output_sums[row],
                                output_sums[row],
                                _shfl_bfly_f32(output_sums[row], delta[0]),
                            )
                    output_base = _local_scalar(
                        "int64",
                        txl.cast(
                            ((n[0] * SEQ_LEN + t[0]) * NUM_V_HEADS + hv[0]) * V + v_base[0], "int64"
                        ),
                    )
                    output_bits = txl.alloc_local((ILP_ROWS,), "uint16")
                    for row in range(ILP_ROWS):
                        txl.ptx.cvt.rn.bf16.f32(output_bits[row], output_sums[row])
                    for row in range(ILP_ROWS):
                        txl.ptx.st.global_.b16(
                            output.ptr_to([output_base[0] + row]),
                            output_bits[row],
                            pred=txl.EQ(lane[0], txl.int32(0)),
                        )
                if CACHE_INTERMEDIATE_STATES or PER_TOKEN_POOL_SCATTER:
                    state_write_base = txl.local_scalar("int64")
                    if PER_TOKEN_POOL_SCATTER:
                        scatter_slot = txl.local_scalar("int32")
                        txl.ptx.ld.global_.s32(
                            scatter_slot, ssm_state_indices.ptr_to([n[0] * SEQ_LEN + t[0]])
                        )
                        if PER_TOKEN_POOL_SCATTER_FLAT:
                            txl.assign(
                                state_write_base,
                                (txl.cast(scatter_slot, "int64") * NUM_V_HEADS + hv[0]) * V * K
                                + txl.cast(v_base[0] * K + k_start[0], "int64"),
                            )
                        else:
                            txl.assign(
                                state_write_base,
                                txl.cast(scatter_slot, "int64") * state_slot_stride
                                + txl.cast(hv[0] * V * K + v_base[0] * K + k_start[0], "int64"),
                            )
                    else:
                        txl.assign(
                            state_write_base,
                            txl.cast(
                                (((n[0] * SEQ_LEN + t[0]) * NUM_V_HEADS + hv[0]) * V + v_base[0])
                                * K
                                + k_start[0],
                                "int64",
                            ),
                        )
                    if PER_TOKEN_POOL_SCATTER and (not PER_TOKEN_POOL_SCATTER_FLAT):
                        _store_state_f32x32(state, state_write_base, r_h)
                    else:
                        _store_state_f32x32(intermediate, state_write_base, r_h)
            if (
                not DISABLE_STATE_UPDATE
                and (not CACHE_INTERMEDIATE_STATES)
                and (RECOVERY_STEPS == 0)
                and (
                    not (
                        PER_REQUEST_ACCEPTED_STEPS
                        and (not DISABLE_OUTPUT)
                        and (not DISABLE_STATE_UPDATE)
                        and (not PER_TOKEN_POOL_SCATTER)
                    )
                )
                and (not (PER_TOKEN_POOL_SCATTER and SAME_POOL))
            ):
                write_offset = _local_scalar(
                    "int64", write_state_base[0] + txl.cast(v_base[0] * K + k_start[0], "int64")
                )
                _store_state_f32x32(state, write_offset[0], r_h)

    return gdn_decode_bf16_wide_vec_mtp.func


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
    return _make_gdn_decode_bf16_wide_vec_mtp(
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
def _load_oracle():
    import flashinfer.gdn_kernels.gdn_decode_bf16_state as source_module

    return source_module.gated_delta_rule_mtp_wide_vec


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
    from tirx_kernels.bench.runner import compile_kernel

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
        tile_v=int(config["tile_v"]),
        disable_output=bool(config.get("disable_output", False)),
        recovery_steps=int(config.get("recovery_steps", 0)),
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
    from tirx_kernels.bench.runner import prepared_gpu_benchmark

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
