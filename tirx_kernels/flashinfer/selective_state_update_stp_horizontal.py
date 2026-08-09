# Copyright (c) 2026 The TIRX Authors
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
"""Integration scaffold for FlashInfer's STP producer-consumer horizontal kernel."""

from __future__ import annotations

import ctypes
from typing import Any

import torch

from tvm.script import tirx as T

from . import selective_state_update_stp_simple as _simple
from . import selective_state_update_stp_vertical as _vertical

KERNEL_META = {
    "name": "selective_state_update_stp_horizontal",
    "category": "flashinfer",
    "compute_capability": 10,
}

FROZEN_FLASHINFER_COMMIT = "f2e04400e330fb2debe0bf8730d9424a1d37927f"
FROZEN_FLASHINFER_SOURCE_SHA256 = "c0e13b64bf42f4f8155058dc9f5877f7aca90832f50a1e7602863894908e89fd"

_LOG2_E = _simple._LOG2_E
_LN_2 = _simple._LN_2

_mul = _simple._mul
_add = _simple._add
_sub = _simple._sub
_fma = _simple._fma
_exp2 = _simple._exp2
_log2 = _simple._log2
_div = _simple._div
_prmt_5410 = _simple._prmt_5410
_mul_hi_u32 = _simple._mul_hi_u32
_global_load_u16 = _simple._global_load_u16
_global_load_u32 = _simple._global_load_u32
_shared_load_u16 = _simple._shared_load_u16
_shared_load_u32 = _simple._shared_load_u32
_bf16_to_f32 = _simple._bf16_to_f32
_state_bits_to_f32 = _simple._state_bits_to_f32
_f32_to_state_bits = _simple._f32_to_state_bits
_f32_to_bf16 = _simple._f32_to_bf16
_load_two_byte_vector = _simple._load_two_byte_vector
_store_two_byte_vector = _simple._store_two_byte_vector

_mbarrier_arrive_wait = _vertical._mbarrier_arrive_wait
_mbarrier_arrive = _vertical._mbarrier_arrive
_mbarrier_expect_tx = _vertical._mbarrier_expect_tx

_TMA_G2S_4D = "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
_TMA_S2G_4D = "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"


def _global_load_s32_to_s64(buffer, index):
    out = T.alloc_local((1,), "int64")
    T.evaluate(T.ptx.ld.global_.s32(out[0], buffer.ptr_to([index])))
    return out[0]


def _global_load_s64(buffer, index):
    out = T.alloc_local((1,), "int64")
    T.evaluate(T.ptx.ld.global_.b64(out[0], buffer.ptr_to([index])))
    return out[0]


def _global_load_i32(buffer, index):
    out = T.alloc_local((1,), "int32")
    T.evaluate(T.ptx.ld.global_.b32(out[0], buffer.ptr_to([index])))
    return out[0]


def _load_weight(buffer, index, dtype: str):
    if dtype == "float32":
        return T.reinterpret("float32", _global_load_u32(buffer, index))
    return _bf16_to_f32(_global_load_u16(buffer, index))


def _lane_mask(raw_lane):
    return T.cast(T.bitwise_and(T.cast(raw_lane, "uint32"), T.uint32(31)), "int32")


def _copy_bf16x8_g2s(source, source_index, destination, destination_index):
    words = T.alloc_local((4,), "uint32")
    T.evaluate(
        T.ptx.ld.global_.v4.b32(
            words[0], words[1], words[2], words[3], source.ptr_to([source_index])
        )
    )
    T.evaluate(
        T.ptx.st.shared.v4.b32(
            destination.ptr_to([destination_index]), words[0], words[1], words[2], words[3]
        )
    )


def _tma_g2s_horizontal(
    smem_raw, dst_offset, tensor_state, column, head, state_batch, barrier_offset
):
    T.evaluate(
        T.ptx[_TMA_G2S_4D](
            smem_raw.ptr_to([dst_offset]),
            T.address_of(tensor_state),
            T.cast(column, "int32"),
            T.int32(0),
            T.cast(head, "int32"),
            T.cast(state_batch, "int32"),
            smem_raw.ptr_to([barrier_offset]),
        )
    )


def _tma_s2g_horizontal(smem_raw, src_offset, tensor_state, column, head, state_batch):
    T.evaluate(
        T.ptx[_TMA_S2G_4D](
            T.address_of(tensor_state),
            T.cast(column, "int32"),
            T.int32(0),
            T.cast(head, "int32"),
            T.cast(state_batch, "int32"),
            smem_raw.ptr_to([src_offset]),
        )
    )


@T.inline
def _producer_horizontal(
    smem_raw,
    tensor_state,
    smem_addr,
    state_batch,
    dst_state_batch,
    head,
    *,
    DIM,
    DSTATE,
    STAGE_COLS,
    NUM_STAGES,
    READ_STATE,
    WRITE_STATE,
    STATE_STAGE_VALUES,
    STATE_STAGE_BYTES,
    OFF_STATE,
    OFF_EMPTY,
    OFF_FULL,
):
    for fill_iter in T.unroll(NUM_STAGES):
        fill_stage: T.int32 = fill_iter
        fill_column: T.int32 = fill_iter * STAGE_COLS
        _mbarrier_arrive_wait(smem_addr + T.cast(OFF_EMPTY + fill_stage * 8, "uint32"))
        if READ_STATE:
            _tma_g2s_horizontal(
                smem_raw,
                OFF_STATE + fill_stage * STATE_STAGE_BYTES,
                tensor_state,
                fill_column,
                head,
                state_batch,
                OFF_FULL + fill_stage * 8,
            )
            _mbarrier_expect_tx(smem_raw, OFF_FULL + fill_stage * 8, STATE_STAGE_BYTES)
        else:
            _mbarrier_arrive(smem_raw, OFF_FULL + fill_stage * 8)

    for steady_iter in T.unroll(DSTATE // STAGE_COLS - NUM_STAGES):
        steady_stage: T.int32 = (NUM_STAGES + steady_iter) % NUM_STAGES
        read_column: T.int32 = (NUM_STAGES + steady_iter) * STAGE_COLS
        write_column: T.int32 = steady_iter * STAGE_COLS
        _mbarrier_arrive_wait(smem_addr + T.cast(OFF_EMPTY + steady_stage * 8, "uint32"))
        if READ_STATE or WRITE_STATE:
            T.evaluate(T.ptx.fence.proxy.async_.shared__cta())
            if WRITE_STATE:
                _tma_s2g_horizontal(
                    smem_raw,
                    OFF_STATE + steady_stage * STATE_STAGE_BYTES,
                    tensor_state,
                    write_column,
                    head,
                    dst_state_batch,
                )
                T.evaluate(T.ptx.cp.async_.bulk.commit_group())
                T.evaluate(T.ptx.cp.async_.bulk.wait_group.read(0))
            if READ_STATE:
                _tma_g2s_horizontal(
                    smem_raw,
                    OFF_STATE + steady_stage * STATE_STAGE_BYTES,
                    tensor_state,
                    read_column,
                    head,
                    state_batch,
                    OFF_FULL + steady_stage * 8,
                )
                _mbarrier_expect_tx(smem_raw, OFF_FULL + steady_stage * 8, STATE_STAGE_BYTES)
            else:
                _mbarrier_arrive(smem_raw, OFF_FULL + steady_stage * 8)
        else:
            _mbarrier_arrive(smem_raw, OFF_FULL + steady_stage * 8)

    for drain_iter in T.unroll(NUM_STAGES):
        drain_stage: T.int32 = (
            NUM_STAGES + (DSTATE // STAGE_COLS - NUM_STAGES) + drain_iter
        ) % NUM_STAGES
        write_column: T.int32 = (DSTATE // STAGE_COLS - NUM_STAGES + drain_iter) * STAGE_COLS
        _mbarrier_arrive_wait(smem_addr + T.cast(OFF_EMPTY + drain_stage * 8, "uint32"))
        if WRITE_STATE:
            T.evaluate(T.ptx.fence.proxy.async_.shared__cta())
            _tma_s2g_horizontal(
                smem_raw,
                OFF_STATE + drain_stage * STATE_STAGE_BYTES,
                tensor_state,
                write_column,
                head,
                dst_state_batch,
            )
            T.evaluate(T.ptx.cp.async_.bulk.commit_group())
            T.evaluate(T.ptx.cp.async_.bulk.wait_group.read(0))


@T.inline
def _dispatch_producer_horizontal(
    smem_raw,
    tensor_state,
    smem_addr,
    state_batch,
    dst_state_batch,
    head,
    update_state,
    pad_slot_id,
    *,
    DIM,
    DSTATE,
    STAGE_COLS,
    NUM_STAGES,
    STATE_STAGE_VALUES,
    STATE_STAGE_BYTES,
    OFF_STATE,
    OFF_EMPTY,
    OFF_FULL,
):
    read_state: T.bool = state_batch != T.cast(pad_slot_id, "int64")
    if read_state:
        if update_state != 0:
            _producer_horizontal(
                smem_raw,
                tensor_state,
                smem_addr,
                state_batch,
                dst_state_batch,
                head,
                DIM=DIM,
                DSTATE=DSTATE,
                STAGE_COLS=STAGE_COLS,
                NUM_STAGES=NUM_STAGES,
                READ_STATE=True,
                WRITE_STATE=True,
                STATE_STAGE_VALUES=STATE_STAGE_VALUES,
                STATE_STAGE_BYTES=STATE_STAGE_BYTES,
                OFF_STATE=OFF_STATE,
                OFF_EMPTY=OFF_EMPTY,
                OFF_FULL=OFF_FULL,
            )
        else:
            _producer_horizontal(
                smem_raw,
                tensor_state,
                smem_addr,
                state_batch,
                dst_state_batch,
                head,
                DIM=DIM,
                DSTATE=DSTATE,
                STAGE_COLS=STAGE_COLS,
                NUM_STAGES=NUM_STAGES,
                READ_STATE=True,
                WRITE_STATE=False,
                STATE_STAGE_VALUES=STATE_STAGE_VALUES,
                STATE_STAGE_BYTES=STATE_STAGE_BYTES,
                OFF_STATE=OFF_STATE,
                OFF_EMPTY=OFF_EMPTY,
                OFF_FULL=OFF_FULL,
            )
    else:
        _producer_horizontal(
            smem_raw,
            tensor_state,
            smem_addr,
            state_batch,
            dst_state_batch,
            head,
            DIM=DIM,
            DSTATE=DSTATE,
            STAGE_COLS=STAGE_COLS,
            NUM_STAGES=NUM_STAGES,
            READ_STATE=False,
            WRITE_STATE=False,
            STATE_STAGE_VALUES=STATE_STAGE_VALUES,
            STATE_STAGE_BYTES=STATE_STAGE_BYTES,
            OFF_STATE=OFF_STATE,
            OFF_EMPTY=OFF_EMPTY,
            OFF_FULL=OFF_FULL,
        )


@T.inline
def _philox4x32_horizontal(random_words, random_seed, random_offset, *, PHILOX_ROUNDS):
    c0: T.uint32 = T.cast(random_offset, "uint32")
    c1: T.uint32 = T.cast(T.shift_right(T.cast(random_offset, "uint64"), T.uint64(32)), "uint32")
    c2: T.uint32 = 0
    c3: T.uint32 = 0
    k0: T.uint32 = T.cast(T.reinterpret("uint64", random_seed), "uint32")
    k1: T.uint32 = T.cast(
        T.shift_right(T.reinterpret("uint64", random_seed), T.uint64(32)), "uint32"
    )
    for _round in T.unroll(PHILOX_ROUNDS):
        old_c0: T.uint32 = c0
        old_c2: T.uint32 = c2
        hi_b: T.uint32 = _mul_hi_u32(T.uint32(0xCD9E8D57), old_c2)
        next_c0: T.uint32 = T.bitwise_xor(T.bitwise_xor(hi_b, c1), k0)
        hi_a: T.uint32 = _mul_hi_u32(T.uint32(0xD2511F53), old_c0)
        next_c2: T.uint32 = T.bitwise_xor(T.bitwise_xor(hi_a, c3), k1)
        next_c1: T.uint32 = old_c2 * T.uint32(0xCD9E8D57)
        next_c3: T.uint32 = old_c0 * T.uint32(0xD2511F53)
        c0 = next_c0
        c1 = next_c1
        c2 = next_c2
        c3 = next_c3
        k0 = k0 + T.uint32(0x9E3779B9)
        k1 = k1 + T.uint32(0xBB67AE85)
    random_words[0] = c0
    random_words[1] = c1
    random_words[2] = c2
    random_words[3] = c3


@T.inline
def _consumer_horizontal(
    smem_raw,
    s_state,
    s_b,
    s_c,
    smem_addr,
    out_accum,
    d,
    member,
    row_group,
    a_value,
    dt_value,
    x_value,
    random_seed,
    state_ptr_offset,
    *,
    DIM,
    DSTATE,
    STATE_DTYPE,
    STATE_BYTES,
    STATE_VALUES_PER_BANK,
    STAGE_COLS,
    NUM_STAGES,
    ITEMS_PER_THREAD,
    STATE_STAGE_VALUES,
    PHILOX_ROUNDS,
    USE_STATE_CACHE,
    OFF_EMPTY,
    OFF_FULL,
):
    a_dt: T.float32 = _mul(a_value, dt_value)
    a_dt_exp_arg: T.float32 = _mul(a_dt, T.float32(_LOG2_E))
    d_a: T.float32 = _exp2(a_dt_exp_arg)
    padded_state_d_a: T.float32 = 0.0
    if not USE_STATE_CACHE:
        padded_state_d_a = _mul(d_a, T.float32(0.0))

    out_value: T.float32 = 0.0
    random_words = T.alloc_local((4,), "uint32")
    i_begin: T.int32 = 0
    stage: T.int32 = 0
    while i_begin < DSTATE:
        _mbarrier_arrive_wait(smem_addr + T.cast(OFF_FULL + stage * 8, "uint32"))
        for item_iter in T.unroll(ITEMS_PER_THREAD // STATE_VALUES_PER_BANK):
            item: T.int32 = item_iter * STATE_VALUES_PER_BANK
            base_column: T.int32 = item + member * ITEMS_PER_THREAD
            sequence_index: T.int32 = row_group * STAGE_COLS + base_column
            bank_cycle: T.int32 = (sequence_index // STATE_VALUES_PER_BANK) // 32
            ii: T.int32 = (base_column + STATE_VALUES_PER_BANK * bank_cycle) % STAGE_COLS
            state_column: T.int32 = i_begin + ii
            state_index: T.int32 = stage * STATE_STAGE_VALUES + d * STAGE_COLS + ii

            if STATE_BYTES == 2:
                r_state = _load_two_byte_vector(
                    s_state, state_index, STATE_VALUES_PER_BANK, "shared"
                )
                b_bits = _load_two_byte_vector(s_b, state_column, STATE_VALUES_PER_BANK, "shared")
                c_bits = _load_two_byte_vector(s_c, state_column, STATE_VALUES_PER_BANK, "shared")
                if PHILOX_ROUNDS > 0 and item_iter % 2 == 0:
                    random_offset: T.int64 = state_ptr_offset + T.cast(
                        d * DSTATE + state_column, "int64"
                    )
                    _philox4x32_horizontal(
                        random_words, random_seed, random_offset, PHILOX_ROUNDS=PHILOX_ROUNDS
                    )
                sr_raw = T.alloc_local((STATE_VALUES_PER_BANK,), "uint32")
                for e in T.unroll(STATE_VALUES_PER_BANK):
                    state_value: T.float32 = 0.0
                    if USE_STATE_CACHE:
                        state_value = _state_bits_to_f32(r_state[e], STATE_DTYPE)
                    b_value: T.float32 = _bf16_to_f32(b_bits[e])
                    c_value: T.float32 = _bf16_to_f32(c_bits[e])
                    d_b: T.float32 = _mul(b_value, dt_value)
                    state_d_a: T.float32 = padded_state_d_a
                    if USE_STATE_CACHE:
                        state_d_a = _mul(state_value, d_a)
                    new_state: T.float32 = _fma(x_value, d_b, state_d_a)
                    if PHILOX_ROUNDS > 0:
                        random13: T.uint32 = T.bitwise_and(
                            random_words[(item + e) % 4], T.uint32(0x1FFF)
                        )
                        T.evaluate(
                            T.ptx.cvt.rs.f16x2.f32(sr_raw[e], T.float32(0.0), new_state, random13)
                        )
                    else:
                        r_state[e] = _f32_to_state_bits(new_state, STATE_DTYPE)
                    out_value = _fma(c_value, new_state, out_value)

                if PHILOX_ROUNDS > 0:
                    packed_state: T.uint32 = _prmt_5410(sr_raw[0], sr_raw[1])
                    T.evaluate(T.ptx.st.shared.b32(s_state.ptr_to([state_index]), packed_state))
                else:
                    _store_two_byte_vector(
                        s_state, state_index, r_state, STATE_VALUES_PER_BANK, "shared"
                    )
            else:
                state_word: T.uint32 = _shared_load_u32(s_state, state_index)
                state_value: T.float32 = 0.0
                if USE_STATE_CACHE:
                    state_value = T.reinterpret("float32", state_word)
                b_value: T.float32 = _bf16_to_f32(_shared_load_u16(s_b, state_column))
                c_value: T.float32 = _bf16_to_f32(_shared_load_u16(s_c, state_column))
                d_b: T.float32 = _mul(b_value, dt_value)
                state_d_a: T.float32 = padded_state_d_a
                if USE_STATE_CACHE:
                    state_d_a = _mul(state_value, d_a)
                new_state: T.float32 = _fma(x_value, d_b, state_d_a)
                out_value = _fma(c_value, new_state, out_value)
                T.evaluate(
                    T.ptx.st.shared.b32(
                        s_state.ptr_to([state_index]), T.reinterpret("uint32", new_state)
                    )
                )

        _mbarrier_arrive(smem_raw, OFF_EMPTY + stage * 8)
        i_begin = i_begin + STAGE_COLS
        stage = (stage + 1) % NUM_STAGES

    out_accum[0] = out_value


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


# Performance rows vary every source branch or compile-time specialization that
# is meaningful for the explicit horizontal oracle.  Batch=1 and nullable-D are
# kept in CONFIGS as correctness-only rows.
BENCH_CONFIGS = [
    _case("b64_h64_d64_s128_r8_base"),
    _case("b64_h8_d64_s128_r1", nheads=8),
    _case("b64_h64_d128_s128_r8", dim=128),
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
    _case("b64_h64_d64_s128_r8_philox10", state_dtype="float16", philox_rounds=10, seed=42),
    _case(
        "b64_h64_d64_s64_r8_philox10", dstate=64, state_dtype="float16", philox_rounds=10, seed=42
    ),
]


CONFIGS = [dict(config) for config in BENCH_CONFIGS] + [
    _case("b1_h64_d64_s128_r8", batch=1),
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
        for batch in (1, 16)
    ],
]


@T.jit
def _selective_state_update_stp_horizontal(
    tensor_state: T.TensorMap(),
    state_h: T.handle,
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
    x_stride_batch: T.int64,
    dt_stride_batch: T.int64,
    b_stride_batch: T.int64,
    c_stride_batch: T.int64,
    z_stride_batch: T.int64,
    out_stride_batch: T.int64,
    state_indices_stride_batch: T.int64,
    dst_indices_stride_batch: T.int64,
    dt_softplus: T.int32,
    update_state: T.int32,
    pad_slot_id: T.int32,
    *,
    BATCH: T.constexpr,
    NHEADS: T.constexpr,
    DIM: T.constexpr,
    DSTATE: T.constexpr,
    HEADS_GROUP_RATIO: T.constexpr,
    CONSUMER_WARPS: T.constexpr,
    NUM_WARPS: T.constexpr,
    MIN_BLOCKS_PER_SM: T.constexpr,
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
    STATE_VALUES_PER_BANK: T.constexpr,
    STAGE_COLS: T.constexpr,
    NUM_STAGES: T.constexpr,
    ITEMS_PER_THREAD: T.constexpr,
    STATE_STAGE_VALUES: T.constexpr,
    STATE_STAGE_BYTES: T.constexpr,
    OFF_STATE: T.constexpr,
    OFF_B: T.constexpr,
    OFF_C: T.constexpr,
    OFF_EMPTY: T.constexpr,
    OFF_FULL: T.constexpr,
    OFF_CONSUMERS: T.constexpr,
    SHARED_BYTES: T.constexpr,
):
    state = T.match_buffer(state_h, (STATE_ELEMENTS,), STATE_DTYPE, scope="global")
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
    # Codegen-only occupancy steering: DIM64 needs the nine-block register cap,
    # while DIM128 must avoid that pressure at its 288-thread CTA size.
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": MIN_BLOCKS_PER_SM})
    # TIRX_TRANSCRIBE_START selective_state_update_stp_horizontal

    random_seed: T.int64 = 0
    if PHILOX_ROUNDS > 0:
        random_seed = rand_seed[0]

    batch_i, head = T.cta_id([BATCH, NHEADS])
    raw_lane, warp = T.thread_id([32, NUM_WARPS])
    lane: T.int32 = _lane_mask(raw_lane)
    group: T.int32 = head // HEADS_GROUP_RATIO

    state_batch: T.int64
    if HAS_STATE_INDICES:
        if INDEX_DTYPE == "int32":
            state_batch = _global_load_s32_to_s64(
                state_indices, batch_i * state_indices_stride_batch
            )
        else:
            state_batch = _global_load_s64(state_indices, batch_i * state_indices_stride_batch)
    else:
        state_batch = T.cast(batch_i, "int64")

    dst_state_batch_i32: T.int32 = 0
    dst_state_batch_i64: T.int64 = state_batch
    if HAS_DST_INDICES:
        if INDEX_DTYPE == "int32":
            dst_state_batch_i32 = _global_load_i32(dst_indices, batch_i * dst_indices_stride_batch)
        else:
            dst_state_batch_i64 = _global_load_s64(dst_indices, batch_i * dst_indices_stride_batch)

    state_ptr_offset: T.int64 = state_batch * state_stride_batch + T.cast(
        head * DIM * DSTATE, "int64"
    )

    pool = T.SMEMPool()
    smem_raw = pool.alloc((SHARED_BYTES,), "uint8", align=128)
    s_state = T.decl_buffer(
        (NUM_STAGES * STATE_STAGE_VALUES,),
        STATE_DTYPE,
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=OFF_STATE,
        align=128,
    )
    s_b = T.decl_buffer(
        (DSTATE,), "bfloat16", data=smem_raw.data, scope="shared.dyn", byte_offset=OFF_B, align=16
    )
    s_c = T.decl_buffer(
        (DSTATE,), "bfloat16", data=smem_raw.data, scope="shared.dyn", byte_offset=OFF_C, align=16
    )
    pool.commit()
    smem_addr: T.uint32 = T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0]))
    T.evaluate(state.data)

    init_stage: T.int32 = warp
    while init_stage < NUM_STAGES:
        if lane == 0:
            T.evaluate(
                T.ptx.mbarrier.init.shared.b64(
                    smem_raw.ptr_to([OFF_EMPTY + init_stage * 8]), T.uint32(1 + CONSUMER_WARPS * 32)
                )
            )
            T.evaluate(
                T.ptx.mbarrier.init.shared.b64(
                    smem_raw.ptr_to([OFF_FULL + init_stage * 8]), T.uint32(1 + CONSUMER_WARPS * 32)
                )
            )
            T.evaluate(T.ptx.fence.proxy.async_.shared__cta())
        init_stage = init_stage + NUM_WARPS
    if warp == 0 and lane == 0:
        T.evaluate(
            T.ptx.mbarrier.init.shared.b64(
                smem_raw.ptr_to([OFF_CONSUMERS]), T.uint32(CONSUMER_WARPS * 32)
            )
        )
    T.cuda.cta_sync()

    if warp == CONSUMER_WARPS:
        if T.cuda.elect_sync():
            if HAS_DST_INDICES and INDEX_DTYPE == "int32":
                _dispatch_producer_horizontal(
                    smem_raw,
                    tensor_state,
                    smem_addr,
                    state_batch,
                    dst_state_batch_i32,
                    head,
                    update_state,
                    pad_slot_id,
                    DIM=DIM,
                    DSTATE=DSTATE,
                    STAGE_COLS=STAGE_COLS,
                    NUM_STAGES=NUM_STAGES,
                    STATE_STAGE_VALUES=STATE_STAGE_VALUES,
                    STATE_STAGE_BYTES=STATE_STAGE_BYTES,
                    OFF_STATE=OFF_STATE,
                    OFF_EMPTY=OFF_EMPTY,
                    OFF_FULL=OFF_FULL,
                )
            else:
                _dispatch_producer_horizontal(
                    smem_raw,
                    tensor_state,
                    smem_addr,
                    state_batch,
                    dst_state_batch_i64,
                    head,
                    update_state,
                    pad_slot_id,
                    DIM=DIM,
                    DSTATE=DSTATE,
                    STAGE_COLS=STAGE_COLS,
                    NUM_STAGES=NUM_STAGES,
                    STATE_STAGE_VALUES=STATE_STAGE_VALUES,
                    STATE_STAGE_BYTES=STATE_STAGE_BYTES,
                    OFF_STATE=OFF_STATE,
                    OFF_EMPTY=OFF_EMPTY,
                    OFF_FULL=OFF_FULL,
                )
    else:
        for arrive_stage in T.unroll(NUM_STAGES):
            _mbarrier_arrive(smem_raw, OFF_EMPTY + arrive_stage * 8)

        a_value: T.float32 = T.reinterpret("float32", _global_load_u32(matrix_a, head))
        d_value: T.float32 = 0.0
        if HAS_D:
            d_value = _load_weight(d_weight, head, WEIGHT_DTYPE)
        dt_value: T.float32 = _load_weight(
            dt, T.cast(batch_i, "int64") * dt_stride_batch + head, WEIGHT_DTYPE
        )
        if HAS_DT_BIAS:
            bias_value: T.float32 = _load_weight(dt_bias, head, WEIGHT_DTYPE)
            dt_value = _add(dt_value, bias_value)
        if dt_softplus != 0:
            if dt_value <= T.float32(20.0):
                dt_exp_arg: T.float32 = _mul(dt_value, T.float32(_LOG2_E))
                dt_exp: T.float32 = _exp2(dt_exp_arg)
                dt_one_plus: T.float32 = _add(T.float32(1.0), dt_exp)
                dt_log2: T.float32 = _log2(dt_one_plus)
                dt_value = _mul(dt_log2, T.float32(_LN_2))

        if warp == 0:
            b_column: T.int32 = lane * 8
            while b_column < DSTATE:
                _copy_bf16x8_g2s(
                    matrix_b,
                    T.cast(batch_i, "int64") * b_stride_batch + group * DSTATE + b_column,
                    s_b,
                    b_column,
                )
                b_column = b_column + 32 * 8
        elif warp == 1:
            c_column: T.int32 = lane * 8
            while c_column < DSTATE:
                _copy_bf16x8_g2s(
                    matrix_c,
                    T.cast(batch_i, "int64") * c_stride_batch + group * DSTATE + c_column,
                    s_c,
                    c_column,
                )
                c_column = c_column + 32 * 8

        row_group: T.int32 = lane % 16
        member: T.int32 = lane // 16
        d: T.int32 = warp * 16 + row_group
        x_value: T.float32 = _bf16_to_f32(
            _global_load_u16(x, T.cast(batch_i, "int64") * x_stride_batch + head * DIM + d)
        )
        z_value: T.float32 = 0.0
        if HAS_Z:
            z_value = _bf16_to_f32(
                _global_load_u16(z, T.cast(batch_i, "int64") * z_stride_batch + head * DIM + d)
            )

        _mbarrier_arrive_wait(smem_addr + T.uint32(OFF_CONSUMERS))
        out_accum = T.alloc_local((1,), "float32")
        out_accum[0] = 0.0
        if state_batch != T.cast(pad_slot_id, "int64"):
            _consumer_horizontal(
                smem_raw,
                s_state,
                s_b,
                s_c,
                smem_addr,
                out_accum,
                d,
                member,
                row_group,
                a_value,
                dt_value,
                x_value,
                random_seed,
                state_ptr_offset,
                DIM=DIM,
                DSTATE=DSTATE,
                STATE_DTYPE=STATE_DTYPE,
                STATE_BYTES=STATE_BYTES,
                STATE_VALUES_PER_BANK=STATE_VALUES_PER_BANK,
                STAGE_COLS=STAGE_COLS,
                NUM_STAGES=NUM_STAGES,
                ITEMS_PER_THREAD=ITEMS_PER_THREAD,
                STATE_STAGE_VALUES=STATE_STAGE_VALUES,
                PHILOX_ROUNDS=PHILOX_ROUNDS,
                USE_STATE_CACHE=True,
                OFF_EMPTY=OFF_EMPTY,
                OFF_FULL=OFF_FULL,
            )
        else:
            _consumer_horizontal(
                smem_raw,
                s_state,
                s_b,
                s_c,
                smem_addr,
                out_accum,
                d,
                member,
                row_group,
                a_value,
                dt_value,
                x_value,
                random_seed,
                state_ptr_offset,
                DIM=DIM,
                DSTATE=DSTATE,
                STATE_DTYPE=STATE_DTYPE,
                STATE_BYTES=STATE_BYTES,
                STATE_VALUES_PER_BANK=STATE_VALUES_PER_BANK,
                STAGE_COLS=STAGE_COLS,
                NUM_STAGES=NUM_STAGES,
                ITEMS_PER_THREAD=ITEMS_PER_THREAD,
                STATE_STAGE_VALUES=STATE_STAGE_VALUES,
                PHILOX_ROUNDS=PHILOX_ROUNDS,
                USE_STATE_CACHE=False,
                OFF_EMPTY=OFF_EMPTY,
                OFF_FULL=OFF_FULL,
            )

        out_value: T.float32 = out_accum[0]
        peer_value: T.float32 = T.cuda.__shfl_down_sync(T.uint32(0xFFFFFFFF), out_value, 16, 32)
        out_value = _add(out_value, peer_value)
        if member == 0:
            out_value = _fma(d_value, x_value, out_value)
            if HAS_Z:
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
                    output.ptr_to([T.cast(batch_i, "int64") * out_stride_batch + head * DIM + d]),
                    output_bits,
                )
            )


def _specialization(kwargs: dict[str, Any]) -> dict[str, Any]:
    batch = int(kwargs["batch"])
    nheads = int(kwargs["nheads"])
    dim = int(kwargs["dim"])
    dstate = int(kwargs["dstate"])
    ngroups = int(kwargs["ngroups"])
    state_dtype = str(kwargs["state_dtype"])
    weight_dtype = str(kwargs["weight_dtype"])
    index_dtype = str(kwargs["index_dtype"])
    state_stride_factor = int(kwargs.get("state_stride_factor", 1))
    has_dst_indices = bool(kwargs.get("has_dst_indices", False))
    if str(kwargs.get("input_dtype", "bfloat16")) != "bfloat16":
        raise ValueError("horizontal STP is scoped to bfloat16 input")
    if str(kwargs.get("matrix_a_dtype", "float32")) != "float32":
        raise ValueError("horizontal STP is scoped to float32 matrix A")
    if state_dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError("horizontal STP supports bfloat16, float16, or float32 state")
    if weight_dtype not in ("float32", "bfloat16"):
        raise ValueError("horizontal STP supports float32 or bfloat16 weights")
    if index_dtype not in ("int32", "int64"):
        raise ValueError("horizontal STP supports int32 or int64 indices")
    if dim not in (64, 128):
        raise ValueError("horizontal STP requires dim in {64, 128}")
    if dstate not in (64, 96, 128, 256):
        raise ValueError("horizontal STP requires dstate in {64, 96, 128, 256}")
    if nheads % ngroups != 0:
        raise ValueError("nheads must be divisible by ngroups")
    heads_group_ratio = nheads // ngroups
    if heads_group_ratio not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError("horizontal STP group ratio must be a supported power of two")
    if state_stride_factor < 1:
        raise ValueError("state_stride_factor must be positive")

    state_bytes = 4 if state_dtype == "float32" else 2
    stage_cols = 64 // state_bytes
    if dstate % stage_cols:
        raise ValueError("dstate must be divisible by the horizontal stage width")
    total_stages = dstate // stage_cols
    num_stages = min(4, total_stages)
    consumer_warps = (dim // 64) * 4
    state_values_per_bank = 4 // state_bytes
    state_stage_values = dim * stage_cols
    state_stage_bytes = state_stage_values * state_bytes
    items_per_thread = stage_cols // 2

    state_slots = max((2 if has_dst_indices else 1) * batch + 8, 16)
    state_stride = nheads * dim * dstate * state_stride_factor
    index_elements = batch * (2 if int(kwargs.get("index_rank", 1)) == 2 else 1)
    philox_rounds = int(kwargs.get("philox_rounds", 0))
    if philox_rounds not in (0, 10):
        raise ValueError("horizontal stochastic rounding supports philox_rounds in {0, 10}")
    if philox_rounds and (state_dtype != "float16" or dstate not in (64, 128)):
        raise ValueError("philox10 is scoped to float16 state with dstate 64 or 128")

    off_state = 0
    off_b = num_stages * state_stage_bytes
    off_c = off_b + dstate * 2
    off_empty = off_c + dstate * 2
    off_full = off_empty + num_stages * 8
    off_consumers = off_full + num_stages * 8
    shared_bytes = _simple._align_up(off_consumers + 8, 128)

    for name, stride_bytes in (
        ("x", nheads * dim * 2),
        ("z", nheads * dim * 2),
        ("B", ngroups * dstate * 2),
        ("C", ngroups * dstate * 2),
    ):
        if stride_bytes % 16 != 0:
            raise ValueError(f"{name} batch stride must be 16-byte aligned, got {stride_bytes}")

    return {
        "BATCH": batch,
        "NHEADS": nheads,
        "DIM": dim,
        "DSTATE": dstate,
        "HEADS_GROUP_RATIO": heads_group_ratio,
        "CONSUMER_WARPS": consumer_warps,
        "NUM_WARPS": consumer_warps + 1,
        "MIN_BLOCKS_PER_SM": 1 if dim == 128 else 9,
        "STATE_DTYPE": state_dtype,
        "WEIGHT_DTYPE": weight_dtype,
        "INDEX_DTYPE": index_dtype,
        "STATE_ELEMENTS": state_slots * state_stride,
        "SCALE_ELEMENTS": 1,
        "X_ELEMENTS": batch * nheads * dim,
        "DT_ELEMENTS": batch * nheads,
        "BC_ELEMENTS": batch * ngroups * dstate,
        "INDEX_ELEMENTS": max(index_elements, 1),
        "HAS_STATE_INDICES": bool(kwargs.get("has_state_indices", True)),
        "HAS_DST_INDICES": has_dst_indices,
        "HAS_Z": bool(kwargs.get("has_z", False)),
        "HAS_D": bool(kwargs.get("has_d", True)),
        "HAS_DT_BIAS": bool(kwargs.get("has_dt_bias", True)),
        "SCALE_STATE": False,
        "PHILOX_ROUNDS": philox_rounds,
        "STATE_BYTES": state_bytes,
        "STATE_VALUES_PER_BANK": state_values_per_bank,
        "STAGE_COLS": stage_cols,
        "NUM_STAGES": num_stages,
        "ITEMS_PER_THREAD": items_per_thread,
        "STATE_STAGE_VALUES": state_stage_values,
        "STATE_STAGE_BYTES": state_stage_bytes,
        "OFF_STATE": off_state,
        "OFF_B": off_b,
        "OFF_C": off_c,
        "OFF_EMPTY": off_empty,
        "OFF_FULL": off_full,
        "OFF_CONSUMERS": off_consumers,
        "SHARED_BYTES": shared_bytes,
    }


def get_kernel(**kwargs: Any):
    """Return the reviewed source-shaped plain-TIRx horizontal specialization."""
    kernel = _selective_state_update_stp_horizontal.specialize(**_specialization(kwargs))
    return kernel.with_attr(
        "tirx.kernel_launch_params",
        ["blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y", "tirx.use_dyn_shared_memory"],
    )


class _AlignedTensorMap:
    """Host storage for one 128-byte-aligned TensorMap payload."""

    def __init__(self) -> None:
        self._storage = ctypes.create_string_buffer(128 + 128)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 127) & ~127)


def _encode_state_tensor_map(
    state: torch.Tensor, spec: dict[str, Any], state_stride: int
) -> _AlignedTensorMap:
    import tvm

    if int(state.data_ptr()) % 128:
        raise ValueError("horizontal state TensorMap base must be 128-byte aligned")
    descriptor = _AlignedTensorMap()
    dstate = spec["DSTATE"]
    dim = spec["DIM"]
    nheads = spec["NHEADS"]
    state_slots = spec["STATE_ELEMENTS"] // state_stride
    state_bytes = spec["STATE_BYTES"]
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    encode(
        descriptor.ptr,
        spec["STATE_DTYPE"],
        4,
        ctypes.c_void_p(int(state.data_ptr())),
        dstate,
        dim,
        nheads,
        state_slots,
        dstate * state_bytes,
        dstate * dim * state_bytes,
        state_stride * state_bytes,
        spec["STAGE_COLS"],
        dim,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        2,
        0,
    )
    return descriptor


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Create independent mutable TIRx/source cases and the state TensorMap."""
    case = _simple.prepare_data(**kwargs)
    spec = _specialization(kwargs)
    case["spec"] = spec
    for name, tensor in (
        ("x", case["x"]),
        ("z", case["z"]),
        ("B", case["matrix_b"]),
        ("C", case["matrix_c"]),
    ):
        if int(tensor.data_ptr()) % 16:
            raise ValueError(f"horizontal {name} base must be 16-byte aligned")
    case["tensor_state"] = _encode_state_tensor_map(
        case["tirx_state_raw"], spec, case["state_stride"]
    )
    return case


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    kwargs = case["kwargs"]
    spec = case["spec"]
    nheads, dim = spec["NHEADS"], spec["DIM"]
    ngroups, dstate = int(kwargs["ngroups"]), spec["DSTATE"]
    has_state_indices = bool(kwargs.get("has_state_indices", True))
    has_dst_indices = bool(kwargs.get("has_dst_indices", False))
    return (
        case["tensor_state"].ptr,
        case["tirx_state_raw"],
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
        nheads * dim,
        nheads,
        ngroups * dstate,
        ngroups * dstate,
        nheads * dim,
        nheads * dim,
        case["state_index_stride"] if has_state_indices else 1,
        case["dst_index_stride"] if has_dst_indices else 0,
        int(bool(kwargs.get("dt_softplus", False))),
        int(bool(kwargs.get("update_state", True))),
        case["pad_slot_id"],
    )


def _run_reference(case: dict[str, Any]) -> torch.Tensor:
    kwargs = case["kwargs"]
    spec = case["spec"]
    oracle = _simple._load_frozen_oracle()
    state_view = _simple._view_state(case["reference_state_raw"], spec, case["state_stride"])
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
        out=source_out,
        disable_state_update=not bool(kwargs.get("update_state", True)),
        rand_seed=case["seed"] if spec["PHILOX_ROUNDS"] else None,
        philox_rounds=spec["PHILOX_ROUNDS"],
        algorithm="horizontal",
    )
    if source_out is None:
        case["reference_output"].copy_(result)
    return result


def run_test(**kwargs: Any) -> None:
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    executable = compile_kernel(get_kernel(**kwargs))
    executable(*_tirx_args(case))
    _run_reference(case)
    torch.cuda.synchronize()
    _simple._assert_case_close(case)


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
    _simple._assert_case_close(case)

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
