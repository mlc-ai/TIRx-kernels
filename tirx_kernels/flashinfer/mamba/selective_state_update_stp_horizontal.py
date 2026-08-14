# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Integration scaffold for FlashInfer's STP producer-consumer horizontal kernel.

Upstream source: include/flashinfer/mamba/kernel_selective_state_update_stp.cuh.
"""

from __future__ import annotations

import ctypes
from typing import Any

import torch

import tvm
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T

from . import selective_state_update_stp_simple as _simple
from . import selective_state_update_stp_vertical as _vertical


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
    "name": "selective_state_update_stp_horizontal",
    "category": "flashinfer",
    "compute_capability": 10,
}


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
    with T.unroll(NUM_STAGES) as fill_iter:
        fill_stage = _builder_scalar("fill_stage", fill_iter, dtype="int32")
        fill_column = _builder_scalar("fill_column", fill_iter * STAGE_COLS, dtype="int32")
        _builder_emit(
            _mbarrier_arrive_wait(smem_addr + T.cast(OFF_EMPTY + fill_stage * 8, "uint32"))
        )
        if READ_STATE:
            _builder_emit(
                _tma_g2s_horizontal(
                    smem_raw,
                    OFF_STATE + fill_stage * STATE_STAGE_BYTES,
                    tensor_state,
                    fill_column,
                    head,
                    state_batch,
                    OFF_FULL + fill_stage * 8,
                )
            )
            _builder_emit(
                _mbarrier_expect_tx(smem_raw, OFF_FULL + fill_stage * 8, STATE_STAGE_BYTES)
            )
        else:
            _builder_emit(_mbarrier_arrive(smem_raw, OFF_FULL + fill_stage * 8))
    with T.unroll(DSTATE // STAGE_COLS - NUM_STAGES) as steady_iter:
        steady_stage = _builder_scalar(
            "steady_stage", (NUM_STAGES + steady_iter) % NUM_STAGES, dtype="int32"
        )
        read_column = _builder_scalar(
            "read_column", (NUM_STAGES + steady_iter) * STAGE_COLS, dtype="int32"
        )
        write_column = _builder_scalar("write_column", steady_iter * STAGE_COLS, dtype="int32")
        _builder_emit(
            _mbarrier_arrive_wait(smem_addr + T.cast(OFF_EMPTY + steady_stage * 8, "uint32"))
        )
        if READ_STATE or WRITE_STATE:
            _builder_emit(T.evaluate(T.ptx.fence.proxy.async_.shared__cta()))
            if WRITE_STATE:
                _builder_emit(
                    _tma_s2g_horizontal(
                        smem_raw,
                        OFF_STATE + steady_stage * STATE_STAGE_BYTES,
                        tensor_state,
                        write_column,
                        head,
                        dst_state_batch,
                    )
                )
                _builder_emit(T.evaluate(T.ptx.cp.async_.bulk.commit_group()))
                _builder_emit(T.evaluate(T.ptx.cp.async_.bulk.wait_group.read(0)))
            if READ_STATE:
                _builder_emit(
                    _tma_g2s_horizontal(
                        smem_raw,
                        OFF_STATE + steady_stage * STATE_STAGE_BYTES,
                        tensor_state,
                        read_column,
                        head,
                        state_batch,
                        OFF_FULL + steady_stage * 8,
                    )
                )
                _builder_emit(
                    _mbarrier_expect_tx(smem_raw, OFF_FULL + steady_stage * 8, STATE_STAGE_BYTES)
                )
            else:
                _builder_emit(_mbarrier_arrive(smem_raw, OFF_FULL + steady_stage * 8))
        else:
            _builder_emit(_mbarrier_arrive(smem_raw, OFF_FULL + steady_stage * 8))
    with T.unroll(NUM_STAGES) as drain_iter:
        drain_stage = _builder_scalar(
            "drain_stage",
            (NUM_STAGES + (DSTATE // STAGE_COLS - NUM_STAGES) + drain_iter) % NUM_STAGES,
            dtype="int32",
        )
        write_column = _builder_scalar(
            "write_column",
            (DSTATE // STAGE_COLS - NUM_STAGES + drain_iter) * STAGE_COLS,
            dtype="int32",
        )
        _builder_emit(
            _mbarrier_arrive_wait(smem_addr + T.cast(OFF_EMPTY + drain_stage * 8, "uint32"))
        )
        if WRITE_STATE:
            _builder_emit(T.evaluate(T.ptx.fence.proxy.async_.shared__cta()))
            _builder_emit(
                _tma_s2g_horizontal(
                    smem_raw,
                    OFF_STATE + drain_stage * STATE_STAGE_BYTES,
                    tensor_state,
                    write_column,
                    head,
                    dst_state_batch,
                )
            )
            _builder_emit(T.evaluate(T.ptx.cp.async_.bulk.commit_group()))
            _builder_emit(T.evaluate(T.ptx.cp.async_.bulk.wait_group.read(0)))


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
    read_state = _builder_scalar(
        "read_state", state_batch != T.cast(pad_slot_id, "int64"), dtype="bool"
    )
    with T.If(read_state):
        with T.Then():
            with T.If(update_state != 0):
                with T.Then():
                    _builder_emit(
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
                    )
                with T.Else():
                    _builder_emit(
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
                    )
        with T.Else():
            _builder_emit(
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
            )


def _philox4x32_horizontal(random_words, random_seed, random_offset, *, PHILOX_ROUNDS):
    c0 = _builder_scalar("c0", T.cast(random_offset, "uint32"), dtype="uint32")
    c1 = _builder_scalar(
        "c1",
        T.cast(T.shift_right(T.cast(random_offset, "uint64"), T.uint64(32)), "uint32"),
        dtype="uint32",
    )
    c2 = _builder_scalar("c2", 0, dtype="uint32")
    c3 = _builder_scalar("c3", 0, dtype="uint32")
    k0 = _builder_scalar(
        "k0", T.cast(T.reinterpret("uint64", random_seed), "uint32"), dtype="uint32"
    )
    k1 = _builder_scalar(
        "k1",
        T.cast(T.shift_right(T.reinterpret("uint64", random_seed), T.uint64(32)), "uint32"),
        dtype="uint32",
    )
    with T.unroll(PHILOX_ROUNDS) as _round:
        old_c0 = _builder_scalar("old_c0", c0, dtype="uint32")
        old_c2 = _builder_scalar("old_c2", c2, dtype="uint32")
        hi_b = _builder_scalar("hi_b", _mul_hi_u32(T.uint32(3449720151), old_c2), dtype="uint32")
        next_c0 = _builder_scalar(
            "next_c0", T.bitwise_xor(T.bitwise_xor(hi_b, c1), k0), dtype="uint32"
        )
        hi_a = _builder_scalar("hi_a", _mul_hi_u32(T.uint32(3528531795), old_c0), dtype="uint32")
        next_c2 = _builder_scalar(
            "next_c2", T.bitwise_xor(T.bitwise_xor(hi_a, c3), k1), dtype="uint32"
        )
        next_c1 = _builder_scalar("next_c1", old_c2 * T.uint32(3449720151), dtype="uint32")
        next_c3 = _builder_scalar("next_c3", old_c0 * T.uint32(3528531795), dtype="uint32")
        T.buffer_store(c0.buffer, next_c0, [0])
        T.buffer_store(c1.buffer, next_c1, [0])
        T.buffer_store(c2.buffer, next_c2, [0])
        T.buffer_store(c3.buffer, next_c3, [0])
        T.buffer_store(k0.buffer, k0 + T.uint32(2654435769), [0])
        T.buffer_store(k1.buffer, k1 + T.uint32(3144134277), [0])
    T.buffer_store(random_words, c0, [0])
    T.buffer_store(random_words, c1, [1])
    T.buffer_store(random_words, c2, [2])
    T.buffer_store(random_words, c3, [3])


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
    a_dt = _builder_scalar("a_dt", _mul(a_value, dt_value), dtype="float32")
    a_dt_exp_arg = _builder_scalar("a_dt_exp_arg", _mul(a_dt, T.float32(_LOG2_E)), dtype="float32")
    d_a = _builder_scalar("d_a", _exp2(a_dt_exp_arg), dtype="float32")
    padded_state_d_a = _builder_scalar("padded_state_d_a", 0.0, dtype="float32")
    if not USE_STATE_CACHE:
        T.buffer_store(padded_state_d_a.buffer, _mul(d_a, T.float32(0.0)), [0])
    out_value = _builder_scalar("out_value", 0.0, dtype="float32")
    random_words = _builder_name("random_words", T.alloc_local((4,), "uint32"))
    i_begin = _builder_scalar("i_begin", 0, dtype="int32")
    stage = _builder_scalar("stage", 0, dtype="int32")
    with T.While(i_begin < DSTATE):
        _builder_emit(_mbarrier_arrive_wait(smem_addr + T.cast(OFF_FULL + stage * 8, "uint32")))
        with T.unroll(ITEMS_PER_THREAD // STATE_VALUES_PER_BANK) as item_iter:
            item = _builder_scalar("item", item_iter * STATE_VALUES_PER_BANK, dtype="int32")
            base_column = _builder_scalar(
                "base_column", item + member * ITEMS_PER_THREAD, dtype="int32"
            )
            sequence_index = _builder_scalar(
                "sequence_index", row_group * STAGE_COLS + base_column, dtype="int32"
            )
            bank_cycle = _builder_scalar(
                "bank_cycle", sequence_index // STATE_VALUES_PER_BANK // 32, dtype="int32"
            )
            ii = _builder_scalar(
                "ii", (base_column + STATE_VALUES_PER_BANK * bank_cycle) % STAGE_COLS, dtype="int32"
            )
            state_column = _builder_scalar("state_column", i_begin + ii, dtype="int32")
            state_index = _builder_scalar(
                "state_index", stage * STATE_STAGE_VALUES + d * STAGE_COLS + ii, dtype="int32"
            )
            if STATE_BYTES == 2:
                r_state = _builder_name(
                    "r_state",
                    _load_two_byte_vector(s_state, state_index, STATE_VALUES_PER_BANK, "shared"),
                )
                b_bits = _builder_name(
                    "b_bits",
                    _load_two_byte_vector(s_b, state_column, STATE_VALUES_PER_BANK, "shared"),
                )
                c_bits = _builder_name(
                    "c_bits",
                    _load_two_byte_vector(s_c, state_column, STATE_VALUES_PER_BANK, "shared"),
                )
                if PHILOX_ROUNDS > 0:
                    with T.If(T.And(T.bool(True), item_iter % 2 == 0)):
                        with T.Then():
                            random_offset = _builder_scalar(
                                "random_offset",
                                state_ptr_offset + T.cast(d * DSTATE + state_column, "int64"),
                                dtype="int64",
                            )
                            _builder_emit(
                                _philox4x32_horizontal(
                                    random_words,
                                    random_seed,
                                    random_offset,
                                    PHILOX_ROUNDS=PHILOX_ROUNDS,
                                )
                            )
                sr_raw = _builder_name("sr_raw", T.alloc_local((STATE_VALUES_PER_BANK,), "uint32"))
                with T.unroll(STATE_VALUES_PER_BANK) as e:
                    state_value = _builder_scalar("state_value", 0.0, dtype="float32")
                    if USE_STATE_CACHE:
                        T.buffer_store(
                            state_value.buffer, _state_bits_to_f32(r_state[e], STATE_DTYPE), [0]
                        )
                    b_value = _builder_scalar("b_value", _bf16_to_f32(b_bits[e]), dtype="float32")
                    c_value = _builder_scalar("c_value", _bf16_to_f32(c_bits[e]), dtype="float32")
                    d_b = _builder_scalar("d_b", _mul(b_value, dt_value), dtype="float32")
                    state_d_a = _builder_scalar("state_d_a", padded_state_d_a, dtype="float32")
                    if USE_STATE_CACHE:
                        T.buffer_store(state_d_a.buffer, _mul(state_value, d_a), [0])
                    new_state = _builder_scalar(
                        "new_state", _fma(x_value, d_b, state_d_a), dtype="float32"
                    )
                    if PHILOX_ROUNDS > 0:
                        random13 = _builder_scalar(
                            "random13",
                            T.bitwise_and(random_words[(item + e) % 4], T.uint32(8191)),
                            dtype="uint32",
                        )
                        _builder_emit(
                            T.evaluate(
                                T.ptx.cvt.rs.f16x2.f32(
                                    sr_raw[e], T.float32(0.0), new_state, random13
                                )
                            )
                        )
                    else:
                        T.buffer_store(r_state, _f32_to_state_bits(new_state, STATE_DTYPE), [e])
                    T.buffer_store(out_value.buffer, _fma(c_value, new_state, out_value), [0])
                if PHILOX_ROUNDS > 0:
                    packed_state = _builder_scalar(
                        "packed_state", _prmt_5410(sr_raw[0], sr_raw[1]), dtype="uint32"
                    )
                    _builder_emit(
                        T.evaluate(T.ptx.st.shared.b32(s_state.ptr_to([state_index]), packed_state))
                    )
                else:
                    _builder_emit(
                        _store_two_byte_vector(
                            s_state, state_index, r_state, STATE_VALUES_PER_BANK, "shared"
                        )
                    )
            else:
                state_word = _builder_scalar(
                    "state_word", _shared_load_u32(s_state, state_index), dtype="uint32"
                )
                state_value = _builder_scalar("state_value", 0.0, dtype="float32")
                if USE_STATE_CACHE:
                    T.buffer_store(state_value.buffer, T.reinterpret("float32", state_word), [0])
                b_value = _builder_scalar(
                    "b_value", _bf16_to_f32(_shared_load_u16(s_b, state_column)), dtype="float32"
                )
                c_value = _builder_scalar(
                    "c_value", _bf16_to_f32(_shared_load_u16(s_c, state_column)), dtype="float32"
                )
                d_b = _builder_scalar("d_b", _mul(b_value, dt_value), dtype="float32")
                state_d_a = _builder_scalar("state_d_a", padded_state_d_a, dtype="float32")
                if USE_STATE_CACHE:
                    T.buffer_store(state_d_a.buffer, _mul(state_value, d_a), [0])
                new_state = _builder_scalar(
                    "new_state", _fma(x_value, d_b, state_d_a), dtype="float32"
                )
                T.buffer_store(out_value.buffer, _fma(c_value, new_state, out_value), [0])
                _builder_emit(
                    T.evaluate(
                        T.ptx.st.shared.b32(
                            s_state.ptr_to([state_index]), T.reinterpret("uint32", new_state)
                        )
                    )
                )
        _builder_emit(_mbarrier_arrive(smem_raw, OFF_EMPTY + stage * 8))
        T.buffer_store(i_begin.buffer, i_begin + STAGE_COLS, [0])
        T.buffer_store(stage.buffer, (stage + 1) % NUM_STAGES, [0])
    T.buffer_store(out_accum, out_value, [0])


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


def _build_selective_state_update_stp_horizontal(
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
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("_selective_state_update_stp_horizontal")
            tensor_state = T.arg("tensor_state", T.TensorMap())
            state_h = T.arg("state_h", T.handle())
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
            rand_seed_h = T.arg("rand_seed_h", T.handle())
            output_h = T.arg("output_h", T.handle())
            state_stride_batch = T.arg("state_stride_batch", T.int64())
            x_stride_batch = T.arg("x_stride_batch", T.int64())
            dt_stride_batch = T.arg("dt_stride_batch", T.int64())
            b_stride_batch = T.arg("b_stride_batch", T.int64())
            c_stride_batch = T.arg("c_stride_batch", T.int64())
            z_stride_batch = T.arg("z_stride_batch", T.int64())
            out_stride_batch = T.arg("out_stride_batch", T.int64())
            state_indices_stride_batch = T.arg("state_indices_stride_batch", T.int64())
            dst_indices_stride_batch = T.arg("dst_indices_stride_batch", T.int64())
            dt_softplus = T.arg("dt_softplus", T.int32())
            update_state = T.arg("update_state", T.int32())
            pad_slot_id = T.arg("pad_slot_id", T.int32())
            state = _builder_name(
                "state", T.match_buffer(state_h, (STATE_ELEMENTS,), STATE_DTYPE, scope="global")
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
            rand_seed = _builder_name(
                "rand_seed", T.match_buffer(rand_seed_h, (1,), "int64", scope="global")
            )
            output = _builder_name(
                "output", T.match_buffer(output_h, (X_ELEMENTS,), "bfloat16", scope="global")
            )
            _builder_emit(T.device_entry())
            _builder_enter(T.attr({"tirx.launch_bounds_min_blocks_per_sm": MIN_BLOCKS_PER_SM}))
            random_seed = _builder_scalar("random_seed", 0, dtype="int64")
            if PHILOX_ROUNDS > 0:
                T.buffer_store(random_seed.buffer, rand_seed[0], [0])
            _builder_values_91 = T.cta_id([BATCH, NHEADS])
            batch_i, head = _builder_values_91
            IRBuilder.name("batch_i", batch_i)
            IRBuilder.name("head", head)
            _builder_values_92 = T.thread_id([32, NUM_WARPS])
            raw_lane, warp = _builder_values_92
            IRBuilder.name("raw_lane", raw_lane)
            IRBuilder.name("warp", warp)
            lane = _builder_scalar("lane", _lane_mask(raw_lane), dtype="int32")
            group = _builder_scalar("group", head // HEADS_GROUP_RATIO, dtype="int32")
            state_batch = _builder_alloc_scalar("state_batch", "int64")
            if HAS_STATE_INDICES:
                if INDEX_DTYPE == "int32":
                    T.buffer_store(
                        state_batch.buffer,
                        _global_load_s32_to_s64(
                            state_indices, batch_i * state_indices_stride_batch
                        ),
                        [0],
                    )
                else:
                    T.buffer_store(
                        state_batch.buffer,
                        _global_load_s64(state_indices, batch_i * state_indices_stride_batch),
                        [0],
                    )
            else:
                T.buffer_store(state_batch.buffer, T.cast(batch_i, "int64"), [0])
            dst_state_batch_i32 = _builder_scalar("dst_state_batch_i32", 0, dtype="int32")
            dst_state_batch_i64 = _builder_scalar("dst_state_batch_i64", state_batch, dtype="int64")
            if HAS_DST_INDICES:
                if INDEX_DTYPE == "int32":
                    T.buffer_store(
                        dst_state_batch_i32.buffer,
                        _global_load_i32(dst_indices, batch_i * dst_indices_stride_batch),
                        [0],
                    )
                else:
                    T.buffer_store(
                        dst_state_batch_i64.buffer,
                        _global_load_s64(dst_indices, batch_i * dst_indices_stride_batch),
                        [0],
                    )
            state_ptr_offset = _builder_scalar(
                "state_ptr_offset",
                state_batch * state_stride_batch + T.cast(head * DIM * DSTATE, "int64"),
                dtype="int64",
            )
            pool = _builder_meta("pool", T.SMEMPool())
            smem_raw = _builder_name("smem_raw", pool.alloc((SHARED_BYTES,), "uint8", align=128))
            s_state = _builder_name(
                "s_state",
                T.decl_buffer(
                    (NUM_STAGES * STATE_STAGE_VALUES,),
                    STATE_DTYPE,
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=OFF_STATE,
                    align=128,
                ),
            )
            s_b = _builder_name(
                "s_b",
                T.decl_buffer(
                    (DSTATE,),
                    "bfloat16",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=OFF_B,
                    align=16,
                ),
            )
            s_c = _builder_name(
                "s_c",
                T.decl_buffer(
                    (DSTATE,),
                    "bfloat16",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=OFF_C,
                    align=16,
                ),
            )
            _builder_emit(pool.commit())
            smem_addr = _builder_scalar(
                "smem_addr", T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0])), dtype="uint32"
            )
            _builder_emit(T.evaluate(state.data))
            init_stage = _builder_scalar("init_stage", warp, dtype="int32")
            with T.While(init_stage < NUM_STAGES):
                with T.If(lane == 0):
                    with T.Then():
                        _builder_emit(
                            T.evaluate(
                                T.ptx.mbarrier.init.shared.b64(
                                    smem_raw.ptr_to([OFF_EMPTY + init_stage * 8]),
                                    T.uint32(1 + CONSUMER_WARPS * 32),
                                )
                            )
                        )
                        _builder_emit(
                            T.evaluate(
                                T.ptx.mbarrier.init.shared.b64(
                                    smem_raw.ptr_to([OFF_FULL + init_stage * 8]),
                                    T.uint32(1 + CONSUMER_WARPS * 32),
                                )
                            )
                        )
                        _builder_emit(T.evaluate(T.ptx.fence.proxy.async_.shared__cta()))
                T.buffer_store(init_stage.buffer, init_stage + NUM_WARPS, [0])
            with T.If(T.And(warp == 0, lane == 0)):
                with T.Then():
                    _builder_emit(
                        T.evaluate(
                            T.ptx.mbarrier.init.shared.b64(
                                smem_raw.ptr_to([OFF_CONSUMERS]), T.uint32(CONSUMER_WARPS * 32)
                            )
                        )
                    )
            _builder_emit(T.cuda.cta_sync())
            with T.If(warp == CONSUMER_WARPS):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            if HAS_DST_INDICES and INDEX_DTYPE == "int32":
                                _builder_emit(
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
                                )
                            else:
                                _builder_emit(
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
                                )
                with T.Else():
                    with T.unroll(NUM_STAGES) as arrive_stage:
                        _builder_emit(_mbarrier_arrive(smem_raw, OFF_EMPTY + arrive_stage * 8))
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
                    dt_value = _builder_scalar(
                        "dt_value",
                        _load_weight(
                            dt, T.cast(batch_i, "int64") * dt_stride_batch + head, WEIGHT_DTYPE
                        ),
                        dtype="float32",
                    )
                    if HAS_DT_BIAS:
                        bias_value = _builder_scalar(
                            "bias_value", _load_weight(dt_bias, head, WEIGHT_DTYPE), dtype="float32"
                        )
                        T.buffer_store(dt_value.buffer, _add(dt_value, bias_value), [0])
                    with T.If(dt_softplus != 0):
                        with T.Then():
                            with T.If(dt_value <= T.float32(20.0)):
                                with T.Then():
                                    dt_exp_arg = _builder_scalar(
                                        "dt_exp_arg",
                                        _mul(dt_value, T.float32(_LOG2_E)),
                                        dtype="float32",
                                    )
                                    dt_exp = _builder_scalar(
                                        "dt_exp", _exp2(dt_exp_arg), dtype="float32"
                                    )
                                    dt_one_plus = _builder_scalar(
                                        "dt_one_plus", _add(T.float32(1.0), dt_exp), dtype="float32"
                                    )
                                    dt_log2 = _builder_scalar(
                                        "dt_log2", _log2(dt_one_plus), dtype="float32"
                                    )
                                    T.buffer_store(
                                        dt_value.buffer, _mul(dt_log2, T.float32(_LN_2)), [0]
                                    )
                    with T.If(warp == 0):
                        with T.Then():
                            b_column = _builder_scalar("b_column", lane * 8, dtype="int32")
                            with T.While(b_column < DSTATE):
                                _builder_emit(
                                    _copy_bf16x8_g2s(
                                        matrix_b,
                                        T.cast(batch_i, "int64") * b_stride_batch
                                        + group * DSTATE
                                        + b_column,
                                        s_b,
                                        b_column,
                                    )
                                )
                                T.buffer_store(b_column.buffer, b_column + 32 * 8, [0])
                        with T.Else():
                            with T.If(warp == 1):
                                with T.Then():
                                    c_column = _builder_scalar("c_column", lane * 8, dtype="int32")
                                    with T.While(c_column < DSTATE):
                                        _builder_emit(
                                            _copy_bf16x8_g2s(
                                                matrix_c,
                                                T.cast(batch_i, "int64") * c_stride_batch
                                                + group * DSTATE
                                                + c_column,
                                                s_c,
                                                c_column,
                                            )
                                        )
                                        T.buffer_store(c_column.buffer, c_column + 32 * 8, [0])
                    row_group = _builder_scalar("row_group", lane % 16, dtype="int32")
                    member = _builder_scalar("member", lane // 16, dtype="int32")
                    d = _builder_scalar("d", warp * 16 + row_group, dtype="int32")
                    x_value = _builder_scalar(
                        "x_value",
                        _bf16_to_f32(
                            _global_load_u16(
                                x, T.cast(batch_i, "int64") * x_stride_batch + head * DIM + d
                            )
                        ),
                        dtype="float32",
                    )
                    z_value = _builder_scalar("z_value", 0.0, dtype="float32")
                    if HAS_Z:
                        T.buffer_store(
                            z_value.buffer,
                            _bf16_to_f32(
                                _global_load_u16(
                                    z, T.cast(batch_i, "int64") * z_stride_batch + head * DIM + d
                                )
                            ),
                            [0],
                        )
                    _builder_emit(_mbarrier_arrive_wait(smem_addr + T.uint32(OFF_CONSUMERS)))
                    out_accum = _builder_name("out_accum", T.alloc_local((1,), "float32"))
                    T.buffer_store(out_accum, 0.0, [0])
                    with T.If(state_batch != T.cast(pad_slot_id, "int64")):
                        with T.Then():
                            _builder_emit(
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
                            )
                        with T.Else():
                            _builder_emit(
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
                            )
                    out_value = _builder_scalar("out_value", out_accum[0], dtype="float32")
                    peer_value = _builder_scalar(
                        "peer_value",
                        T.cuda.__shfl_down_sync(T.uint32(4294967295), out_value, 16, 32),
                        dtype="float32",
                    )
                    T.buffer_store(out_value.buffer, _add(out_value, peer_value), [0])
                    with T.If(member == 0):
                        with T.Then():
                            T.buffer_store(out_value.buffer, _fma(d_value, x_value, out_value), [0])
                            if HAS_Z:
                                neg_z = _builder_scalar(
                                    "neg_z", _sub(T.float32(0.0), z_value), dtype="float32"
                                )
                                z_exp_arg = _builder_scalar(
                                    "z_exp_arg", _mul(neg_z, T.float32(_LOG2_E)), dtype="float32"
                                )
                                exp_neg_z = _builder_scalar(
                                    "exp_neg_z", _exp2(z_exp_arg), dtype="float32"
                                )
                                denominator = _builder_scalar(
                                    "denominator", _add(T.float32(1.0), exp_neg_z), dtype="float32"
                                )
                                sigmoid_z = _builder_scalar(
                                    "sigmoid_z", _div(T.float32(1.0), denominator), dtype="float32"
                                )
                                silu_z = _builder_scalar(
                                    "silu_z", _mul(z_value, sigmoid_z), dtype="float32"
                                )
                                T.buffer_store(out_value.buffer, _mul(out_value, silu_z), [0])
                            output_bits = _builder_scalar(
                                "output_bits", _f32_to_bf16(out_value), dtype="uint16"
                            )
                            _builder_emit(
                                T.evaluate(
                                    T.ptx.st.global_.b16(
                                        output.ptr_to(
                                            [
                                                T.cast(batch_i, "int64") * out_stride_batch
                                                + head * DIM
                                                + d
                                            ]
                                        ),
                                        output_bits,
                                    )
                                )
                            )
    return builder.get()


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
    kernel = _build_selective_state_update_stp_horizontal(**_specialization(kwargs))
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
    oracle = _simple._load_oracle()
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
    _run_reference(case)
    torch.cuda.synchronize()
    _simple._assert_case_close(case)


def run_gpu(
    prepared,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    config = dict(prepared["config"])
    config.update(kwargs)
    kwargs = config
    executable = prepared["executable"]
    rounds = int(kwargs.pop("rounds", 5))
    cooldown_s = float(kwargs.pop("cooldown_s", 1.0))
    from tirx_kernels.runner import bench

    case = prepare_data(**kwargs)
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


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    prepared = prepare_bench(**config)
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
