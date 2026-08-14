# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's selective-state-update MTP horizontal kernel.

Upstream source: include/flashinfer/mamba/kernel_selective_state_update_mtp_horizontal.cuh.
"""

from __future__ import annotations

import ctypes
import functools
from typing import Any

import torch

import tvm
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T

from . import selective_state_update_mtp_simple as _simple
from . import selective_state_update_mtp_vertical as _vertical
from .selective_state_update_mtp_simple import _case


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
    "name": "selective_state_update_mtp_horizontal",
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
_global_load_u16 = _simple._global_load_u16
_global_load_u32 = _simple._global_load_u32
_shared_load_u16 = _simple._shared_load_u16
_shared_load_u32 = _simple._shared_load_u32
_bf16_to_f32 = _simple._bf16_to_f32
_f16_to_f32 = _simple._f16_to_f32
_f32_to_bf16 = _simple._f32_to_bf16
_f32_to_f16 = _simple._f32_to_f16
_load_weight = _simple._load_weight
_extract_u16 = _simple._extract_u16
_bf16_word_to_f32x2 = _simple._bf16_word_to_f32x2
_global_load_nc_s32 = _vertical._global_load_nc_s32
_global_load_nc_s64 = _vertical._global_load_nc_s64
_philox4x32 = _vertical._philox4x32
_tma_g2s_4d = _vertical._tma_g2s_4d


def _mbarrier_arrive(smem_raw, offset):
    T.evaluate(T.ptx.mbarrier.arrive.shared__cta.b64(smem_raw.ptr_to([offset])))


def _mbarrier_arrive_expect_tx(smem_raw, offset, num_bytes):
    T.evaluate(
        T.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
            smem_raw.ptr_to([offset]), T.uint32(num_bytes)
        )
    )


def _mbarrier_arrive_wait_parity(smem_raw, offset, parity):
    _builder_emit(_mbarrier_arrive(smem_raw, offset))
    ready = _builder_name("ready", T.alloc_local((1,), "uint32"))
    T.buffer_store(ready, T.uint32(0), [0])
    with T.While(True):
        _builder_emit(
            T.evaluate(
                T.ptx.mbarrier.try_wait.parity.shared__cta.b64(
                    ready[0], smem_raw.ptr_to([offset]), T.uint32(parity)
                )
            )
        )
        with T.If(ready[0] != T.uint32(0)):
            with T.Then():
                T.evaluate(T.break_loop())


def _shared_load_v2_b32(buffer, index, values):
    T.evaluate(T.ptx.ld.shared.v2.b32(values[0], values[1], buffer.ptr_to([index])))


def _shared_load_v4_b32(buffer, index, values):
    T.evaluate(
        T.ptx.ld.shared.v4.b32(values[0], values[1], values[2], values[3], buffer.ptr_to([index]))
    )


def _global_load_v2_b16(buffer, index, values):
    T.evaluate(T.ptx.ld.global_.v2.b16(values[0], values[1], buffer.ptr_to([index])))


def _global_load_v4_b16(buffer, index, values):
    T.evaluate(
        T.ptx.ld.global_.v4.b16(values[0], values[1], values[2], values[3], buffer.ptr_to([index]))
    )


def _store_state_tile(
    values,
    pair_base,
    random_words,
    random_base,
    destination,
    destination_base,
    *,
    STATE_DTYPE,
    PAIRS_PER_TILE_MEMBER,
    PHILOX_ROUNDS,
):
    if STATE_DTYPE == "float32":
        words = _builder_name("words", T.alloc_local((4,), "uint32"))
        with T.unroll(2) as pair:
            packed = _builder_scalar("packed", values[pair_base + pair])
            T.buffer_store(words, T.reinterpret("uint32", T.cuda.float2_x(packed)), [pair * 2])
            T.buffer_store(words, T.reinterpret("uint32", T.cuda.float2_y(packed)), [pair * 2 + 1])
        _builder_emit(
            T.evaluate(
                T.ptx.st.global_.v4.b32(
                    destination.ptr_to([destination_base]), words[0], words[1], words[2], words[3]
                )
            )
        )
    elif PHILOX_ROUNDS > 0:
        packed_words = _builder_name("packed_words", T.alloc_local((4,), "uint32"))
        with T.unroll(4) as pair:
            packed = _builder_scalar("packed", values[pair_base + pair], dtype="uint64")
            _builder_emit(
                T.evaluate(
                    T.ptx.cvt.rs.f16x2.f32(
                        packed_words[pair],
                        T.cuda.float2_y(packed),
                        T.cuda.float2_x(packed),
                        random_words[random_base + pair // 2 * 4 + pair % 2],
                    )
                )
            )
        _builder_emit(
            T.evaluate(
                T.ptx.st.global_.v4.b32(
                    destination.ptr_to([destination_base]),
                    packed_words[0],
                    packed_words[1],
                    packed_words[2],
                    packed_words[3],
                )
            )
        )
    else:
        bits = _builder_name("bits", T.alloc_local((8,), "uint16"))
        words = _builder_name("words", T.alloc_local((4,), "uint32"))
        with T.unroll(PAIRS_PER_TILE_MEMBER) as pair:
            packed = _builder_scalar("packed", values[pair_base + pair], dtype="uint64")
            if STATE_DTYPE == "bfloat16":
                T.buffer_store(bits, _f32_to_bf16(T.cuda.float2_x(packed)), [pair * 2])
                T.buffer_store(bits, _f32_to_bf16(T.cuda.float2_y(packed)), [pair * 2 + 1])
            else:
                T.buffer_store(bits, _f32_to_f16(T.cuda.float2_x(packed)), [pair * 2])
                T.buffer_store(bits, _f32_to_f16(T.cuda.float2_y(packed)), [pair * 2 + 1])
        with T.unroll(4) as word:
            _builder_emit(
                T.evaluate(T.ptx.mov.b32(words[word], bits[word * 2], bits[word * 2 + 1]))
            )
        _builder_emit(
            T.evaluate(
                T.ptx.st.global_.v4.b32(
                    destination.ptr_to([destination_base]), words[0], words[1], words[2], words[3]
                )
            )
        )


def _role_load_horizontal(
    smem_raw,
    tensor_state,
    tensor_b,
    tensor_c,
    tensor_x,
    lane,
    batch_i,
    head,
    kv_group,
    state_batch,
    *,
    IS_PAD,
    NTOKENS,
    NUM_TMA_LOADS,
    DSTATE_PAD,
    STATE_STAGE_BYTES,
    STATE_BYTES,
    DIM,
    OFF_B,
    OFF_C,
    OFF_STATE,
    OFF_X,
    OFF_EMPTY,
    OFF_FULL,
):
    with T.If(lane == 0):
        with T.Then():
            _builder_emit(_tma_g2s_4d(smem_raw, OFF_B, tensor_b, 0, kv_group, 0, batch_i, OFF_FULL))
            _builder_emit(_tma_g2s_4d(smem_raw, OFF_C, tensor_c, 0, kv_group, 0, batch_i, OFF_FULL))
            _builder_emit(_tma_g2s_4d(smem_raw, OFF_X, tensor_x, 0, head, 0, batch_i, OFF_FULL))
    bcx_bytes = _builder_scalar(
        "bcx_bytes", 2 * NTOKENS * DSTATE_PAD * 2 + NTOKENS * DIM * 2, dtype="int32"
    )
    with T.unroll(NUM_TMA_LOADS) as tl:
        stage = _builder_scalar("stage", tl % 2, dtype="int32")
        parity = _builder_scalar("parity", tl // 2, dtype="int32")
        _builder_emit(_mbarrier_arrive_wait_parity(smem_raw, OFF_EMPTY + stage * 8, parity))
        with T.If(lane == 0):
            with T.Then():
                if not IS_PAD:
                    _builder_emit(
                        _tma_g2s_4d(
                            smem_raw,
                            OFF_STATE + stage * STATE_STAGE_BYTES,
                            tensor_state,
                            0,
                            tl * 32,
                            head,
                            state_batch,
                            OFF_FULL + stage * 8,
                        )
                    )
                    transaction_bytes = _builder_scalar(
                        "transaction_bytes", STATE_STAGE_BYTES, dtype="int32"
                    )
                    with T.If(tl == 0):
                        with T.Then():
                            T.buffer_store(
                                transaction_bytes.buffer, transaction_bytes + bcx_bytes, [0]
                            )
                    _builder_emit(
                        _mbarrier_arrive_expect_tx(
                            smem_raw, OFF_FULL + stage * 8, transaction_bytes
                        )
                    )
                else:
                    transaction_bytes = _builder_scalar("transaction_bytes", 0, dtype="int32")
                    with T.If(tl == 0):
                        with T.Then():
                            T.buffer_store(transaction_bytes.buffer, bcx_bytes, [0])
                    _builder_emit(
                        _mbarrier_arrive_expect_tx(
                            smem_raw, OFF_FULL + stage * 8, transaction_bytes
                        )
                    )


def _role_update_horizontal(
    smem_raw,
    s_u16,
    s_u32,
    state,
    intermediate_states,
    dt,
    matrix_a,
    d_weight,
    z,
    dt_bias,
    intermediate_indices,
    rand_seed,
    output,
    lane,
    compute_warp,
    batch_i,
    head,
    state_batch,
    state_stride_batch,
    dt_stride_batch,
    dt_stride_mtp,
    z_stride_batch,
    z_stride_mtp,
    out_stride_batch,
    out_stride_mtp,
    dt_softplus,
    update_state,
    *,
    IS_PAD,
    NHEADS,
    DIM,
    DSTATE,
    NTOKENS,
    NUM_TMA_LOADS,
    DSTATE_PAD,
    STATE_DTYPE,
    WEIGHT_DTYPE,
    STATE_BYTES,
    ELEMS_PER_TILE_MEMBER,
    PAIRS_PER_TILE_MEMBER,
    ELEMS_PER_TILE,
    NUM_TILES,
    HAS_INTERMEDIATE_STATES,
    HAS_Z,
    HAS_D,
    HAS_DT_BIAS,
    UPDATE_STATE,
    PHILOX_ROUNDS,
    OFF_B,
    OFF_C,
    OFF_STATE,
    OFF_X,
    OFF_DT,
    OFF_OUT,
    OFF_EMPTY,
    OFF_FULL,
    OFF_OUT_READY,
    STATE_STAGE_BYTES,
):
    random_seed = _builder_scalar("random_seed", 0, dtype="int64")
    if PHILOX_ROUNDS > 0 and (not IS_PAD):
        T.buffer_store(random_seed.buffer, rand_seed[0], [0])
    icache_idx = _builder_scalar("icache_idx", state_batch, dtype="int64")
    if HAS_INTERMEDIATE_STATES and (not IS_PAD):
        T.buffer_store(icache_idx.buffer, T.cast(intermediate_indices[batch_i], "int64"), [0])
    a_value = _builder_scalar(
        "a_value", T.reinterpret("float32", _global_load_u32(matrix_a, head)), dtype="float32"
    )
    d_value = _builder_scalar("d_value", 0.0, dtype="float32")
    if HAS_D:
        T.buffer_store(d_value.buffer, _load_weight(d_weight, head, WEIGHT_DTYPE), [0])
    bias_value = _builder_scalar("bias_value", 0.0, dtype="float32")
    if HAS_DT_BIAS:
        T.buffer_store(bias_value.buffer, _load_weight(dt_bias, head, WEIGHT_DTYPE), [0])
    _builder_emit(_mbarrier_arrive(smem_raw, OFF_EMPTY))
    _builder_emit(_mbarrier_arrive(smem_raw, OFF_EMPTY + 8))
    flat_thread = _builder_scalar("flat_thread", compute_warp * 32 + lane, dtype="int32")
    with T.If(flat_thread < NTOKENS):
        with T.Then():
            dt_value = _builder_scalar(
                "dt_value",
                _load_weight(
                    dt,
                    T.cast(batch_i, "int64") * dt_stride_batch
                    + T.cast(flat_thread, "int64") * dt_stride_mtp
                    + head,
                    WEIGHT_DTYPE,
                ),
                dtype="float32",
            )
            if HAS_DT_BIAS:
                T.buffer_store(dt_value.buffer, _add(dt_value, bias_value), [0])
            with T.If(T.And(dt_softplus != 0, dt_value <= T.float32(20.0))):
                with T.Then():
                    exp_value = _builder_scalar(
                        "exp_value", _exp2(_mul(dt_value, T.float32(_LOG2_E))), dtype="float32"
                    )
                    log_value = _builder_scalar(
                        "log_value", _log2(_add(T.float32(1.0), exp_value)), dtype="float32"
                    )
                    T.buffer_store(dt_value.buffer, _mul(log_value, T.float32(_LN_2)), [0])
            _builder_emit(
                T.evaluate(
                    T.ptx.st.shared.b32(
                        s_u32.ptr_to([OFF_DT // 4 + flat_thread]), T.reinterpret("uint32", dt_value)
                    )
                )
            )
    member = _builder_scalar("member", lane % 8, dtype="int32")
    row_group = _builder_scalar("row_group", lane // 8, dtype="int32")
    state_ptr_offset_i32 = _builder_scalar(
        "state_ptr_offset_i32",
        T.cast(state_batch * state_stride_batch + T.cast(head * DIM * DSTATE, "int64"), "int32"),
        dtype="int32",
    )
    with T.unroll(NUM_TMA_LOADS) as tl:
        stage = _builder_scalar("stage", tl % 2, dtype="int32")
        parity = _builder_scalar("parity", tl // 2, dtype="int32")
        _builder_emit(_mbarrier_arrive_wait_parity(smem_raw, OFF_FULL + stage * 8, parity))
        with T.serial(2) as sp:
            sram_row = _builder_scalar(
                "sram_row", sp * 16 + compute_warp * 4 + row_group, dtype="int32"
            )
            dd = _builder_scalar("dd", tl * 32 + sram_row, dtype="int32")
            state_values = _builder_name(
                "state_values", T.alloc_local((NUM_TILES * PAIRS_PER_TILE_MEMBER,), "uint64")
            )
            with T.unroll(NUM_TILES) as tile:
                member_col = _builder_scalar(
                    "member_col",
                    tile * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER,
                    dtype="int32",
                )
                pair_base = _builder_scalar(
                    "pair_base", tile * PAIRS_PER_TILE_MEMBER, dtype="int32"
                )
                if IS_PAD:
                    with T.unroll(PAIRS_PER_TILE_MEMBER) as pair:
                        T.buffer_store(
                            state_values,
                            T.cuda.make_float2(T.float32(0.0), T.float32(0.0)),
                            [pair_base + pair],
                        )
                else:
                    with T.If(T.And(T.bool(True), member_col < DSTATE)):
                        with T.Then():
                            state_words = _builder_name(
                                "state_words", T.alloc_local((4,), "uint32")
                            )
                            state_word_index = _builder_scalar(
                                "state_word_index",
                                (
                                    OFF_STATE
                                    + stage * STATE_STAGE_BYTES
                                    + (sram_row * DSTATE_PAD + member_col) * STATE_BYTES
                                )
                                // 4,
                                dtype="int32",
                            )
                            _builder_emit(_shared_load_v4_b32(s_u32, state_word_index, state_words))
                            with T.unroll(PAIRS_PER_TILE_MEMBER) as pair:
                                if STATE_DTYPE == "bfloat16":
                                    T.buffer_store(
                                        state_values,
                                        _bf16_word_to_f32x2(state_words[pair]),
                                        [pair_base + pair],
                                    )
                                elif STATE_DTYPE == "float16":
                                    T.buffer_store(
                                        state_values,
                                        T.cuda.make_float2(
                                            _f16_to_f32(_extract_u16(state_words[pair], False)),
                                            _f16_to_f32(_extract_u16(state_words[pair], True)),
                                        ),
                                        [pair_base + pair],
                                    )
                                else:
                                    T.buffer_store(
                                        state_values,
                                        T.cuda.make_float2(
                                            T.reinterpret("float32", state_words[pair * 2]),
                                            T.reinterpret("float32", state_words[pair * 2 + 1]),
                                        ),
                                        [pair_base + pair],
                                    )
                        with T.Else():
                            with T.unroll(PAIRS_PER_TILE_MEMBER) as pair:
                                T.buffer_store(
                                    state_values,
                                    T.cuda.make_float2(T.float32(0.0), T.float32(0.0)),
                                    [pair_base + pair],
                                )
            random_words = _builder_name("random_words", T.alloc_local((16,), "uint32"))
            if PHILOX_ROUNDS > 0 and (not IS_PAD):
                seed_lo = _builder_scalar("seed_lo", T.cast(random_seed, "uint32"), dtype="uint32")
                seed_hi = _builder_scalar(
                    "seed_hi",
                    T.cast(
                        T.shift_right(T.reinterpret("uint64", random_seed), T.uint32(32)), "uint32"
                    ),
                    dtype="uint32",
                )
                with T.unroll(4) as random_group:
                    random_tile = _builder_scalar("random_tile", random_group // 2, dtype="int32")
                    random_e = _builder_scalar("random_e", random_group % 2 * 4, dtype="int32")
                    random_col = _builder_scalar(
                        "random_col",
                        random_tile * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER + random_e,
                        dtype="int32",
                    )
                    counter = _builder_scalar(
                        "counter", state_ptr_offset_i32 + dd * DSTATE + random_col, dtype="int32"
                    )
                    group_words = _builder_name("group_words", T.alloc_local((4,), "uint32"))
                    _builder_emit(
                        _philox4x32(
                            group_words, seed_lo, seed_hi, counter, PHILOX_ROUNDS=PHILOX_ROUNDS
                        )
                    )
                    with T.unroll(4) as random_word:
                        T.buffer_store(
                            random_words, group_words[random_word], [random_group * 4 + random_word]
                        )
            bc_step_words = _builder_scalar("bc_step_words", 0, dtype="int32")
            x_step = _builder_scalar("x_step", 0, dtype="int32")
            dt_step = _builder_scalar("dt_step", 0, dtype="int32")
            out_step = _builder_scalar("out_step", 0, dtype="int32")
            intermediate_step_base = _builder_scalar(
                "intermediate_step_base",
                icache_idx * T.int64(NTOKENS * NHEADS * DIM * DSTATE)
                + T.cast(head * DIM * DSTATE + dd * DSTATE, "int64"),
                dtype="int64",
            )
            with T.serial(NTOKENS) as step:
                dt_value = _builder_scalar(
                    "dt_value",
                    T.reinterpret("float32", _shared_load_u32(s_u32, OFF_DT // 4 + dt_step)),
                    dtype="float32",
                )
                da_value = _builder_scalar(
                    "da_value",
                    _exp2(_mul(_mul(a_value, dt_value), T.float32(_LOG2_E))),
                    dtype="float32",
                )
                x_value = _builder_scalar(
                    "x_value",
                    _bf16_to_f32(_shared_load_u16(s_u16, OFF_X // 2 + x_step + dd)),
                    dtype="float32",
                )
                dtx_value = _builder_scalar("dtx_value", _mul(dt_value, x_value), dtype="float32")
                out_pair = _builder_scalar(
                    "out_pair", T.cuda.make_float2(T.float32(0.0), T.float32(0.0)), dtype="uint64"
                )
                with T.unroll(NUM_TILES) as tile:
                    member_col = _builder_scalar(
                        "member_col",
                        tile * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER,
                        dtype="int32",
                    )
                    with T.If(member_col < DSTATE):
                        with T.Then():
                            b_words = _builder_name("b_words", T.alloc_local((4,), "uint32"))
                            c_words = _builder_name("c_words", T.alloc_local((4,), "uint32"))
                            b_word_index = _builder_scalar(
                                "b_word_index",
                                OFF_B // 4 + bc_step_words + member_col // 2,
                                dtype="int32",
                            )
                            c_word_index = _builder_scalar(
                                "c_word_index",
                                OFF_C // 4 + bc_step_words + member_col // 2,
                                dtype="int32",
                            )
                            if PAIRS_PER_TILE_MEMBER == 2:
                                _builder_emit(_shared_load_v2_b32(s_u32, b_word_index, b_words))
                                _builder_emit(_shared_load_v2_b32(s_u32, c_word_index, c_words))
                            else:
                                _builder_emit(_shared_load_v4_b32(s_u32, b_word_index, b_words))
                                _builder_emit(_shared_load_v4_b32(s_u32, c_word_index, c_words))
                            with T.unroll(PAIRS_PER_TILE_MEMBER) as pair:
                                b_pair = _builder_scalar(
                                    "b_pair", _bf16_word_to_f32x2(b_words[pair]), dtype="uint64"
                                )
                                c_pair = _builder_scalar(
                                    "c_pair", _bf16_word_to_f32x2(c_words[pair]), dtype="uint64"
                                )
                                dbx_pair = _builder_alloc_scalar("dbx_pair", "uint64")
                                _builder_emit(
                                    T.ptx.mul.f32x2(
                                        dbx_pair, b_pair, T.cuda.make_float2(dtx_value, dtx_value)
                                    )
                                )
                                pair_index = _builder_scalar(
                                    "pair_index", tile * PAIRS_PER_TILE_MEMBER + pair, dtype="int32"
                                )
                                updated_state = _builder_alloc_scalar("updated_state", "uint64")
                                _builder_emit(
                                    T.ptx.fma.rn.f32x2(
                                        updated_state,
                                        T.cuda.make_float2(da_value, da_value),
                                        state_values[pair_index],
                                        dbx_pair,
                                    )
                                )
                                T.buffer_store(state_values, updated_state, [pair_index])
                                _builder_emit(
                                    T.ptx.fma.rn.f32x2(out_pair, updated_state, c_pair, out_pair)
                                )
                out_value = _builder_scalar(
                    "out_value",
                    _add(T.cuda.float2_x(out_pair), T.cuda.float2_y(out_pair)),
                    dtype="float32",
                )
                with T.unroll(3) as delta_idx:
                    peer_out = _builder_scalar(
                        "peer_out",
                        T.cuda.__shfl_down_sync(
                            T.uint32(4294967295),
                            out_value,
                            T.shift_right(T.int32(4), delta_idx),
                            32,
                        ),
                        dtype="float32",
                    )
                    T.buffer_store(out_value.buffer, _add(out_value, peer_out), [0])
                with T.If(member == 0):
                    with T.Then():
                        _builder_emit(
                            T.evaluate(
                                T.ptx.st.shared.b32(
                                    s_u32.ptr_to([OFF_OUT // 4 + out_step + dd]),
                                    T.reinterpret("uint32", _fma(d_value, x_value, out_value)),
                                )
                            )
                        )
                T.buffer_store(bc_step_words.buffer, bc_step_words + DSTATE_PAD // 2, [0])
                T.buffer_store(x_step.buffer, x_step + DIM, [0])
                T.buffer_store(dt_step.buffer, dt_step + 1, [0])
                T.buffer_store(out_step.buffer, out_step + DIM, [0])
                if HAS_INTERMEDIATE_STATES and (not IS_PAD):
                    with T.unroll(NUM_TILES) as tile:
                        member_col = _builder_scalar(
                            "member_col",
                            tile * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER,
                            dtype="int32",
                        )
                        with T.If(member_col < DSTATE):
                            with T.Then():
                                _builder_emit(
                                    _store_state_tile(
                                        state_values,
                                        tile * PAIRS_PER_TILE_MEMBER,
                                        random_words,
                                        tile * 8,
                                        intermediate_states,
                                        intermediate_step_base + member_col,
                                        STATE_DTYPE=STATE_DTYPE,
                                        PAIRS_PER_TILE_MEMBER=PAIRS_PER_TILE_MEMBER,
                                        PHILOX_ROUNDS=PHILOX_ROUNDS,
                                    )
                                )
                    T.buffer_store(
                        intermediate_step_base.buffer,
                        intermediate_step_base + T.int64(NHEADS * DIM * DSTATE),
                        [0],
                    )
                if UPDATE_STATE:
                    with T.If(T.And(T.And(T.bool(True), step == NTOKENS - 1), T.bool(not IS_PAD))):
                        with T.Then():
                            final_base = _builder_scalar(
                                "final_base",
                                state_batch * state_stride_batch
                                + T.cast(head * DIM * DSTATE + dd * DSTATE, "int64"),
                                dtype="int64",
                            )
                            with T.unroll(NUM_TILES) as tile:
                                member_col = _builder_scalar(
                                    "member_col",
                                    tile * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER,
                                    dtype="int32",
                                )
                                with T.If(member_col < DSTATE):
                                    with T.Then():
                                        _builder_emit(
                                            _store_state_tile(
                                                state_values,
                                                tile * PAIRS_PER_TILE_MEMBER,
                                                random_words,
                                                tile * 8,
                                                state,
                                                final_base + member_col,
                                                STATE_DTYPE=STATE_DTYPE,
                                                PAIRS_PER_TILE_MEMBER=PAIRS_PER_TILE_MEMBER,
                                                PHILOX_ROUNDS=PHILOX_ROUNDS,
                                            )
                                        )
        _builder_emit(_mbarrier_arrive(smem_raw, OFF_EMPTY + stage * 8))
    _builder_emit(_mbarrier_arrive_wait_parity(smem_raw, OFF_OUT_READY, 0))
    with T.unroll((NTOKENS + 3) // 4) as episode:
        step = _builder_scalar("step", compute_warp + episode * 4, dtype="int32")
        with T.If(step < NTOKENS):
            with T.Then():
                out_base = _builder_scalar(
                    "out_base",
                    T.cast(batch_i, "int64") * out_stride_batch
                    + T.cast(step, "int64") * out_stride_mtp
                    + head * DIM,
                    dtype="int64",
                )
                z_base = _builder_scalar(
                    "z_base",
                    T.cast(batch_i, "int64") * z_stride_batch
                    + T.cast(step, "int64") * z_stride_mtp
                    + head * DIM,
                    dtype="int64",
                )
                if DIM == 64:
                    d = _builder_scalar("d", lane * 2, dtype="int32")
                    out_words = _builder_name("out_words", T.alloc_local((2,), "uint32"))
                    if NTOKENS == 1:
                        T.buffer_store(
                            out_words, _shared_load_u32(s_u32, OFF_OUT // 4 + step * DIM + d), [0]
                        )
                        T.buffer_store(
                            out_words,
                            _shared_load_u32(s_u32, OFF_OUT // 4 + step * DIM + d + 1),
                            [1],
                        )
                    else:
                        _builder_emit(
                            _shared_load_v2_b32(s_u32, OFF_OUT // 4 + step * DIM + d, out_words)
                        )
                    z_bits = _builder_name("z_bits", T.alloc_local((2,), "uint16"))
                    output_bits = _builder_name("output_bits", T.alloc_local((2,), "uint16"))
                    if HAS_Z:
                        _builder_emit(_global_load_v2_b16(z, z_base + d, z_bits))
                    with T.unroll(2) as k:
                        out_value = _builder_scalar(
                            "out_value", T.reinterpret("float32", out_words[k]), dtype="float32"
                        )
                        if HAS_Z:
                            z_value = _builder_scalar(
                                "z_value", _bf16_to_f32(z_bits[k]), dtype="float32"
                            )
                            exp_neg = _builder_scalar(
                                "exp_neg",
                                _exp2(_mul(_sub(T.float32(0.0), z_value), T.float32(_LOG2_E))),
                                dtype="float32",
                            )
                            sigmoid = _builder_scalar(
                                "sigmoid",
                                _div(T.float32(1.0), _add(T.float32(1.0), exp_neg)),
                                dtype="float32",
                            )
                            T.buffer_store(
                                out_value.buffer, _mul(out_value, _mul(z_value, sigmoid)), [0]
                            )
                        T.buffer_store(output_bits, _f32_to_bf16(out_value), [k])
                    _builder_emit(
                        T.evaluate(
                            T.ptx.st.global_.v2.b16(
                                output.ptr_to([out_base + d]), output_bits[0], output_bits[1]
                            )
                        )
                    )
                else:
                    d = _builder_scalar("d", lane * 4, dtype="int32")
                    out_words = _builder_name("out_words", T.alloc_local((4,), "uint32"))
                    _builder_emit(
                        _shared_load_v4_b32(s_u32, OFF_OUT // 4 + step * DIM + d, out_words)
                    )
                    z_bits = _builder_name("z_bits", T.alloc_local((4,), "uint16"))
                    output_bits = _builder_name("output_bits", T.alloc_local((4,), "uint16"))
                    if HAS_Z:
                        _builder_emit(_global_load_v4_b16(z, z_base + d, z_bits))
                    with T.unroll(4) as k:
                        out_value = _builder_scalar(
                            "out_value", T.reinterpret("float32", out_words[k]), dtype="float32"
                        )
                        if HAS_Z:
                            z_value = _builder_scalar(
                                "z_value", _bf16_to_f32(z_bits[k]), dtype="float32"
                            )
                            exp_neg = _builder_scalar(
                                "exp_neg",
                                _exp2(_mul(_sub(T.float32(0.0), z_value), T.float32(_LOG2_E))),
                                dtype="float32",
                            )
                            sigmoid = _builder_scalar(
                                "sigmoid",
                                _div(T.float32(1.0), _add(T.float32(1.0), exp_neg)),
                                dtype="float32",
                            )
                            T.buffer_store(
                                out_value.buffer, _mul(out_value, _mul(z_value, sigmoid)), [0]
                            )
                        T.buffer_store(output_bits, _f32_to_bf16(out_value), [k])
                    _builder_emit(
                        T.evaluate(
                            T.ptx.st.global_.v4.b16(
                                output.ptr_to([out_base + d]),
                                output_bits[0],
                                output_bits[1],
                                output_bits[2],
                                output_bits[3],
                            )
                        )
                    )


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
    _case("b64_h64_d64_s128_t4_r8_philox10", state_dtype="float16", philox_rounds=10, seed=42),
]


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
    _case("b64_h64_d64_s128_t4_r8_philox10", state_dtype="float16", philox_rounds=10, seed=42),
    _case(
        "b64_h64_d64_s128_t4_r8_philox10_intermediate",
        state_dtype="float16",
        philox_rounds=10,
        has_intermediate_states=True,
        update_state=False,
        seed=42,
    ),
]

REJECTION_CONFIGS = [
    _case("reject_scaled_state", state_dtype="int16", expected_rejection="scaled state"),
    _case(
        "reject_varlen",
        batch=4,
        mode="varlen_uniform",
        has_dst_indices=True,
        has_num_accepted_tokens=True,
        index_dtype="int32",
        index_rank=2,
        expected_rejection="varlen",
    ),
]


def _build_selective_state_update_mtp_horizontal(
    *,
    BATCH: T.constexpr,
    NHEADS: T.constexpr,
    DIM: T.constexpr,
    DSTATE: T.constexpr,
    NTOKENS: T.constexpr,
    HEADS_PER_GROUP: T.constexpr,
    NUM_TMA_LOADS: T.constexpr,
    DSTATE_PAD: T.constexpr,
    STATE_BYTES: T.constexpr,
    STATE_STAGE_BYTES: T.constexpr,
    ELEMS_PER_TILE_MEMBER: T.constexpr,
    PAIRS_PER_TILE_MEMBER: T.constexpr,
    ELEMS_PER_TILE: T.constexpr,
    NUM_TILES: T.constexpr,
    HAS_STATE_INDICES: T.constexpr,
    HAS_INTERMEDIATE_STATES: T.constexpr,
    HAS_Z: T.constexpr,
    HAS_D: T.constexpr,
    HAS_DT_BIAS: T.constexpr,
    UPDATE_STATE: T.constexpr,
    PHILOX_ROUNDS: T.constexpr,
    OFF_B: T.constexpr,
    OFF_C: T.constexpr,
    OFF_STATE: T.constexpr,
    OFF_X: T.constexpr,
    OFF_DT: T.constexpr,
    OFF_OUT: T.constexpr,
    OFF_EMPTY: T.constexpr,
    OFF_FULL: T.constexpr,
    OFF_OUT_READY: T.constexpr,
    SHARED_BYTES: T.constexpr,
    STATE_ELEMENTS: T.constexpr,
    X_ELEMENTS: T.constexpr,
    DT_ELEMENTS: T.constexpr,
    BC_ELEMENTS: T.constexpr,
    INDEX_ELEMENTS: T.constexpr,
    INTERMEDIATE_ELEMENTS: T.constexpr,
    ACCEPTED_ELEMENTS: T.constexpr,
    STATE_DTYPE: T.constexpr,
    WEIGHT_DTYPE: T.constexpr,
    INDEX_DTYPE: T.constexpr,
):
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("_selective_state_update_mtp_horizontal")
            tensor_state = T.arg("tensor_state", T.TensorMap())
            tensor_b = T.arg("tensor_b", T.TensorMap())
            tensor_c = T.arg("tensor_c", T.TensorMap())
            tensor_x = T.arg("tensor_x", T.TensorMap())
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
            rand_seed = _builder_name(
                "rand_seed", T.match_buffer(rand_seed_h, (1,), "int64", scope="global")
            )
            output = _builder_name(
                "output", T.match_buffer(output_h, (X_ELEMENTS,), "bfloat16", scope="global")
            )
            _builder_emit(T.device_entry())
            _builder_enter(T.attr({"tirx.launch_bounds_min_blocks_per_sm": 6}))
            _builder_values_114 = T.cta_id([BATCH, NHEADS])
            batch_i, head = _builder_values_114
            IRBuilder.name("batch_i", batch_i)
            IRBuilder.name("head", head)
            _builder_values_115 = T.thread_id([32, 5])
            lane, warp = _builder_values_115
            IRBuilder.name("lane", lane)
            IRBuilder.name("warp", warp)
            state_batch = _builder_alloc_scalar("state_batch", "int64")
            if HAS_STATE_INDICES:
                if INDEX_DTYPE == "int32":
                    T.buffer_store(
                        state_batch.buffer,
                        T.cast(_global_load_nc_s32(state_indices, batch_i), "int64"),
                        [0],
                    )
                else:
                    T.buffer_store(
                        state_batch.buffer, _global_load_nc_s64(state_indices, batch_i), [0]
                    )
            else:
                T.buffer_store(state_batch.buffer, T.cast(batch_i, "int64"), [0])
            smem_raw = _builder_name(
                "smem_raw", T.alloc_buffer((SHARED_BYTES,), "uint8", scope="shared.dyn", align=128)
            )
            s_u16 = _builder_name(
                "s_u16",
                T.decl_buffer(
                    (SHARED_BYTES // 2,),
                    "uint16",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=0,
                    align=128,
                ),
            )
            s_u32 = _builder_name(
                "s_u32",
                T.decl_buffer(
                    (SHARED_BYTES // 4,),
                    "uint32",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=0,
                    align=128,
                ),
            )
            with T.If(T.And(warp == 0, lane == 0)):
                with T.Then():
                    with T.unroll(2) as stage:
                        _builder_emit(
                            T.evaluate(
                                T.ptx.mbarrier.init.shared.b64(
                                    smem_raw.ptr_to([OFF_EMPTY + stage * 8]), T.uint32(160)
                                )
                            )
                        )
                        _builder_emit(
                            T.evaluate(
                                T.ptx.mbarrier.init.shared.b64(
                                    smem_raw.ptr_to([OFF_FULL + stage * 8]), T.uint32(129)
                                )
                            )
                        )
                    _builder_emit(
                        T.evaluate(
                            T.ptx.mbarrier.init.shared.b64(
                                smem_raw.ptr_to([OFF_OUT_READY]), T.uint32(128)
                            )
                        )
                    )
            _builder_emit(T.cuda.cta_sync())

            def dispatch_pad(*, IS_PAD):
                with T.If(warp < 4):
                    with T.Then():
                        _builder_emit(
                            _role_update_horizontal(
                                smem_raw,
                                s_u16,
                                s_u32,
                                state,
                                intermediate_states,
                                dt,
                                matrix_a,
                                d_weight,
                                z,
                                dt_bias,
                                intermediate_indices,
                                rand_seed,
                                output,
                                lane,
                                warp,
                                batch_i,
                                head,
                                state_batch,
                                state_stride_batch,
                                dt_stride_batch,
                                dt_stride_mtp,
                                z_stride_batch,
                                z_stride_mtp,
                                out_stride_batch,
                                out_stride_mtp,
                                dt_softplus,
                                update_state,
                                IS_PAD=IS_PAD,
                                NHEADS=NHEADS,
                                DIM=DIM,
                                DSTATE=DSTATE,
                                NTOKENS=NTOKENS,
                                NUM_TMA_LOADS=NUM_TMA_LOADS,
                                DSTATE_PAD=DSTATE_PAD,
                                STATE_DTYPE=STATE_DTYPE,
                                WEIGHT_DTYPE=WEIGHT_DTYPE,
                                STATE_BYTES=STATE_BYTES,
                                ELEMS_PER_TILE_MEMBER=ELEMS_PER_TILE_MEMBER,
                                PAIRS_PER_TILE_MEMBER=PAIRS_PER_TILE_MEMBER,
                                ELEMS_PER_TILE=ELEMS_PER_TILE,
                                NUM_TILES=NUM_TILES,
                                HAS_INTERMEDIATE_STATES=HAS_INTERMEDIATE_STATES,
                                HAS_Z=HAS_Z,
                                HAS_D=HAS_D,
                                HAS_DT_BIAS=HAS_DT_BIAS,
                                UPDATE_STATE=UPDATE_STATE,
                                PHILOX_ROUNDS=PHILOX_ROUNDS,
                                OFF_B=OFF_B,
                                OFF_C=OFF_C,
                                OFF_STATE=OFF_STATE,
                                OFF_X=OFF_X,
                                OFF_DT=OFF_DT,
                                OFF_OUT=OFF_OUT,
                                OFF_EMPTY=OFF_EMPTY,
                                OFF_FULL=OFF_FULL,
                                OFF_OUT_READY=OFF_OUT_READY,
                                STATE_STAGE_BYTES=STATE_STAGE_BYTES,
                            )
                        )
                    with T.Else():
                        _builder_emit(
                            _role_load_horizontal(
                                smem_raw,
                                tensor_state,
                                tensor_b,
                                tensor_c,
                                tensor_x,
                                lane,
                                batch_i,
                                head,
                                head // HEADS_PER_GROUP,
                                state_batch,
                                IS_PAD=IS_PAD,
                                NTOKENS=NTOKENS,
                                NUM_TMA_LOADS=NUM_TMA_LOADS,
                                DSTATE_PAD=DSTATE_PAD,
                                STATE_STAGE_BYTES=STATE_STAGE_BYTES,
                                STATE_BYTES=STATE_BYTES,
                                DIM=DIM,
                                OFF_B=OFF_B,
                                OFF_C=OFF_C,
                                OFF_STATE=OFF_STATE,
                                OFF_X=OFF_X,
                                OFF_EMPTY=OFF_EMPTY,
                                OFF_FULL=OFF_FULL,
                            )
                        )

            with T.If(state_batch == T.cast(pad_slot_id, "int64")):
                with T.Then():
                    _builder_emit(dispatch_pad(IS_PAD=True))
                with T.Else():
                    _builder_emit(dispatch_pad(IS_PAD=False))
    return builder.get()


def _validate_dispatch(config: dict[str, Any]) -> None:
    if str(config.get("mode", "fixed")).startswith("varlen"):
        raise ValueError("MTP horizontal does not support varlen inputs")
    if str(config.get("state_dtype")) == "int16":
        raise ValueError("MTP horizontal does not support scaled state")
    if str(config.get("input_dtype", "bfloat16")) != "bfloat16":
        raise ValueError("MTP horizontal is scoped to bfloat16 input")
    if str(config.get("matrix_a_dtype", "float32")) != "float32":
        raise ValueError("MTP horizontal is scoped to float32 matrix A")
    dim = int(config["dim"])
    dstate = int(config["dstate"])
    nheads = int(config["nheads"])
    heads_per_group = int(config["heads_per_group"])
    if dim not in (64, 128) or dim % 32:
        raise ValueError("MTP horizontal requires DIM in {64, 128} and divisible by 32")
    if dstate not in (64, 96, 128) or dstate % 8:
        raise ValueError("MTP horizontal requires DSTATE in {64, 96, 128} and divisible by 8")
    if nheads % heads_per_group:
        raise ValueError("nheads must be divisible by heads_per_group")
    philox_rounds = int(config.get("philox_rounds", 0))
    if philox_rounds not in (0, 10):
        raise ValueError("MTP horizontal stochastic rounding supports philox_rounds in {0, 10}")
    if philox_rounds and str(config["state_dtype"]) != "float16":
        raise ValueError("MTP horizontal Philox is restricted to float16 state")


def _specialization(config: dict[str, Any]) -> dict[str, Any]:
    _validate_dispatch(config)
    base = _simple._specialization(config)
    batch = int(config["batch"])
    nheads = int(config["nheads"])
    dim = int(config["dim"])
    dstate = int(config["dstate"])
    tokens = int(config["tokens"])
    heads_per_group = int(config["heads_per_group"])
    state_dtype = str(config["state_dtype"])
    state_bytes = 4 if state_dtype == "float32" else 2
    dstate_pad = _simple._align_up(dstate * state_bytes, 128) // state_bytes
    elems_per_tile_member = 16 // state_bytes
    pairs_per_tile_member = elems_per_tile_member // 2
    elems_per_tile = elems_per_tile_member * 8
    num_tiles = (dstate_pad // 8) // elems_per_tile_member
    state_stage_bytes = 32 * dstate_pad * state_bytes

    off_b = 0
    off_c = _simple._align_up(off_b + tokens * dstate_pad * 2, 128)
    off_state = _simple._align_up(off_c + tokens * dstate_pad * 2, 128)
    off_x = _simple._align_up(off_state + 2 * state_stage_bytes, 128)
    off_dt = off_x + tokens * dim * 2
    off_out = off_dt + tokens * 4
    off_empty = _simple._align_up(off_out + tokens * dim * 4, 8)
    off_full = off_empty + 16
    off_out_ready = off_full + 16
    shared_bytes = _simple._align_up(off_out_ready + 8, 128)

    return {
        "BATCH": batch,
        "NHEADS": nheads,
        "DIM": dim,
        "DSTATE": dstate,
        "NTOKENS": tokens,
        "HEADS_PER_GROUP": heads_per_group,
        "NUM_TMA_LOADS": dim // 32,
        "DSTATE_PAD": dstate_pad,
        "STATE_BYTES": state_bytes,
        "STATE_STAGE_BYTES": state_stage_bytes,
        "ELEMS_PER_TILE_MEMBER": elems_per_tile_member,
        "PAIRS_PER_TILE_MEMBER": pairs_per_tile_member,
        "ELEMS_PER_TILE": elems_per_tile,
        "NUM_TILES": num_tiles,
        "HAS_STATE_INDICES": bool(config.get("has_state_indices", True)),
        "HAS_INTERMEDIATE_STATES": bool(config.get("has_intermediate_states", False)),
        "HAS_Z": bool(config.get("has_z", False)),
        "HAS_D": bool(config.get("has_d", True)),
        "HAS_DT_BIAS": bool(config.get("has_dt_bias", True)),
        "UPDATE_STATE": bool(config.get("update_state", True)),
        "PHILOX_ROUNDS": int(config.get("philox_rounds", 0)),
        "OFF_B": off_b,
        "OFF_C": off_c,
        "OFF_STATE": off_state,
        "OFF_X": off_x,
        "OFF_DT": off_dt,
        "OFF_OUT": off_out,
        "OFF_EMPTY": off_empty,
        "OFF_FULL": off_full,
        "OFF_OUT_READY": off_out_ready,
        "SHARED_BYTES": shared_bytes,
        "STATE_ELEMENTS": base["STATE_ELEMENTS"],
        "X_ELEMENTS": base["X_ELEMENTS"],
        "DT_ELEMENTS": base["DT_ELEMENTS"],
        "BC_ELEMENTS": base["BC_ELEMENTS"],
        "INDEX_ELEMENTS": base["INDEX_ELEMENTS"],
        "INTERMEDIATE_ELEMENTS": base["INTERMEDIATE_ELEMENTS"],
        "ACCEPTED_ELEMENTS": base["ACCEPTED_ELEMENTS"],
        "STATE_DTYPE": state_dtype,
        "WEIGHT_DTYPE": str(config["weight_dtype"]),
        "INDEX_DTYPE": str(config["index_dtype"]),
    }


def get_kernel(**kwargs: Any):
    kernel = _build_selective_state_update_mtp_horizontal(**_specialization(kwargs))
    return kernel.with_attr(
        "tirx.kernel_launch_params",
        ["blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y", "tirx.use_dyn_shared_memory"],
    )


class _AlignedTensorMap:
    """Host storage for one 128-byte-aligned CUtensorMap payload."""

    def __init__(self) -> None:
        self._storage = ctypes.create_string_buffer(256)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 127) & ~127)


def _encode_tensor_map(
    tensor: torch.Tensor,
    *,
    dtype: str,
    shape: tuple[int, int, int, int],
    strides: tuple[int, int, int, int],
    box: tuple[int, int, int, int],
    name: str,
) -> _AlignedTensorMap:
    import tvm

    if int(tensor.data_ptr()) % 128:
        raise ValueError(f"horizontal {name} TensorMap base must be 128-byte aligned")
    if strides[0] != 1:
        raise ValueError(f"horizontal {name} TensorMap innermost stride must be one")
    element_bytes = tensor.element_size()
    for axis, stride in enumerate(strides[1:], start=1):
        if stride * element_bytes % 16:
            raise ValueError(
                f"horizontal {name} TensorMap byte stride {axis} must be 16-byte aligned"
            )
    descriptor = _AlignedTensorMap()
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    encode(
        descriptor.ptr,
        dtype,
        4,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *shape,
        strides[1] * element_bytes,
        strides[2] * element_bytes,
        strides[3] * element_bytes,
        *box,
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


def _build_tensor_maps(case: dict[str, Any]) -> dict[str, _AlignedTensorMap]:
    config = case["config"]
    spec = case["spec"]
    nheads = spec["NHEADS"]
    dim = spec["DIM"]
    dstate = spec["DSTATE"]
    tokens = spec["NTOKENS"]
    ngroups = nheads // spec["HEADS_PER_GROUP"]
    state_stride = int(config.get("state_stride_factor", 1)) * nheads * dim * dstate
    state_slots = spec["STATE_ELEMENTS"] // state_stride
    matrix_b = case["matrix_b"]
    matrix_c = case["matrix_c"]
    x = case["x"]
    return {
        "state": _encode_tensor_map(
            case["tirx_state_storage"],
            dtype=spec["STATE_DTYPE"],
            shape=(dstate, dim, nheads, state_slots),
            strides=(1, dstate, dstate * dim, state_stride),
            box=(spec["DSTATE_PAD"], 32, 1, 1),
            name="state",
        ),
        "b": _encode_tensor_map(
            matrix_b,
            dtype="bfloat16",
            shape=(dstate, ngroups, tokens, spec["BATCH"]),
            strides=(1, matrix_b.stride(2), matrix_b.stride(1), matrix_b.stride(0)),
            box=(spec["DSTATE_PAD"], 1, tokens, 1),
            name="B",
        ),
        "c": _encode_tensor_map(
            matrix_c,
            dtype="bfloat16",
            shape=(dstate, ngroups, tokens, spec["BATCH"]),
            strides=(1, matrix_c.stride(2), matrix_c.stride(1), matrix_c.stride(0)),
            box=(spec["DSTATE_PAD"], 1, tokens, 1),
            name="C",
        ),
        "x": _encode_tensor_map(
            x,
            dtype="bfloat16",
            shape=(dim, nheads, tokens, spec["BATCH"]),
            strides=(1, x.stride(2), x.stride(1), x.stride(0)),
            box=(dim, 1, tokens, 1),
            name="x",
        ),
    }


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Allocate independent TIRx/reference cases and four TensorMaps."""
    spec = _specialization(kwargs)
    case = _simple.prepare_data(**kwargs)
    if int(kwargs.get("index_rank", 1)) == 2:
        flat_indices = case["state_indices"].reshape(-1)
        flat_indices[: spec["BATCH"]] = torch.arange(
            spec["BATCH"], dtype=flat_indices.dtype, device=flat_indices.device
        )
    case["spec"] = spec
    case["tensor_maps"] = _build_tensor_maps(case)
    return case


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    maps = case["tensor_maps"]
    return (
        maps["state"].ptr,
        maps["b"].ptr,
        maps["c"].ptr,
        maps["x"].ptr,
        *_simple._tirx_args(case),
    )


@functools.cache
def _load_oracle():
    from flashinfer.mamba import selective_state_update

    return selective_state_update


def _run_reference(case: dict[str, Any]) -> torch.Tensor:
    config = case["config"]
    stride_factor = int(config.get("state_stride_factor", 1))
    source_out = case["flashinfer_output"] if bool(config.get("use_out_tensor", True)) else None
    result = _load_oracle()(
        case["flashinfer_state_storage"][::stride_factor],
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
        state_scale=None,
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
        intermediate_state_scales=None,
        rand_seed=case["rand_seed"] if int(config.get("philox_rounds", 0)) else None,
        philox_rounds=int(config.get("philox_rounds", 0)),
        cache_steps=int(config["tokens"]),
        algorithm="horizontal",
        cu_seqlens=None,
        num_accepted_tokens=None,
    )
    if source_out is None:
        case["flashinfer_output"].copy_(result)
    return result


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(**kwargs: Any) -> None:
    expected_rejection = kwargs.pop("expected_rejection", None)
    if expected_rejection is not None:
        try:
            _specialization(kwargs)
        except ValueError as error:
            if expected_rejection not in str(error):
                raise AssertionError(
                    f"expected rejection containing {expected_rejection!r}, got {error!r}"
                ) from error
            return
        raise AssertionError(f"expected horizontal rejection containing {expected_rejection!r}")
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    executable = compile_kernel(get_kernel(**kwargs))
    executable(*_tirx_args(case))
    torch.cuda.synchronize()
    _run_reference(case)
    torch.cuda.synchronize()
    _simple._assert_case_close(case)


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
    "REJECTION_CONFIGS",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
