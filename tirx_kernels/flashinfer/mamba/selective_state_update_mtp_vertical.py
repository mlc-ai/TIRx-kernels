# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's selective-state-update MTP vertical kernel.

Upstream source: include/flashinfer/mamba/kernel_selective_state_update_mtp_vertical.cuh.
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
    "name": "selective_state_update_mtp_vertical",
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
_mul_hi_u32 = _simple._mul_hi_u32
_mul_lo_s32 = _simple._mul_lo_s32
_add_s32 = _simple._add_s32
_global_load_u16 = _simple._global_load_u16
_global_load_u32 = _simple._global_load_u32
_shared_load_u16 = _simple._shared_load_u16
_shared_load_u32 = _simple._shared_load_u32
_bf16_to_f32 = _simple._bf16_to_f32
_f16_to_f32 = _simple._f16_to_f32
_f32_to_bf16 = _simple._f32_to_bf16
_f32_to_f16 = _simple._f32_to_f16
_load_weight = _simple._load_weight

_TMA_G2S_4D = "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"


# FlashInfer's official MTP sweep, pinned to algorithm="vertical" by the
# reference interface that will replace the scaffolding stub.
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


# One-variable-at-a-time correctness domain for the vertical dispatch.  Scaled
# state and varlen rejection cases are explicit because FlashInfer rejects both
# before launch for this algorithm.
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


def _shared_load_v2_b16(buffer, index, values):
    T.evaluate(T.ptx.ld.shared.v2.b16(values[0], values[1], buffer.ptr_to([index])))


def _shared_load_v4_b16(buffer, index, values):
    T.evaluate(
        T.ptx.ld.shared.v4.b16(values[0], values[1], values[2], values[3], buffer.ptr_to([index]))
    )


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


def _global_load_nc_s32(buffer, index):
    out = T.alloc_local((1,), "int32")
    T.evaluate(T.ptx.ld.global_.nc.b32(out[0], buffer.ptr_to([index])))
    return out[0]


def _global_load_nc_s64(buffer, index):
    out = T.alloc_local((1,), "int64")
    T.evaluate(T.ptx.ld.global_.nc.b64(out[0], buffer.ptr_to([index])))
    return out[0]


def _mbarrier_arrive_wait(bar_addr):
    token = _builder_name("token", T.alloc_local((1,), "uint64"))
    done = _builder_name("done", T.alloc_local((1,), "uint32"))
    _builder_emit(
        T.evaluate(
            T.ptx.mbarrier.arrive.shared__cta.b64(token[0], T.cast(bar_addr, "uint32"), T.uint32(1))
        )
    )
    with T.While(True):
        _builder_emit(
            T.evaluate(
                T.ptx.mbarrier.try_wait.shared__cta.b64(
                    done[0], T.cast(bar_addr, "uint32"), token[0]
                )
            )
        )
        with T.If(done[0] != T.uint32(0)):
            with T.Then():
                T.evaluate(T.break_loop())


def _mbarrier_arrive(smem_raw, offset):
    T.evaluate(T.ptx.mbarrier.arrive.shared__cta.b64(smem_raw.ptr_to([offset]), T.uint32(1)))


def _mbarrier_arrive_count_release(smem_raw, offset, count):
    T.evaluate(
        T.ptx.mbarrier.arrive.release.cta.shared__cta.b64(
            smem_raw.ptr_to([offset]), T.uint32(count)
        )
    )


def _mbarrier_expect_tx_relaxed(smem_raw, offset, num_bytes):
    T.evaluate(
        T.ptx.mbarrier.expect_tx.relaxed.cta.shared__cta.b64(
            smem_raw.ptr_to([offset]), T.uint32(num_bytes)
        )
    )


def _mbarrier_arrive_expect_tx(smem_raw, offset, num_bytes):
    T.evaluate(
        T.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
            smem_raw.ptr_to([offset]), T.uint32(num_bytes)
        )
    )


def _tma_g2s_4d(smem_raw, dst_offset, tensor_map, c0, c1, c2, c3, barrier_offset):
    T.evaluate(
        T.ptx[_TMA_G2S_4D](
            smem_raw.ptr_to([dst_offset]),
            T.address_of(tensor_map),
            T.cast(c0, "int32"),
            T.cast(c1, "int32"),
            T.cast(c2, "int32"),
            T.cast(c3, "int32"),
            smem_raw.ptr_to([barrier_offset]),
        )
    )


def _philox4x32(random_words, seed_lo, seed_hi, counter, *, PHILOX_ROUNDS):
    c0 = _builder_scalar("c0", T.reinterpret("uint32", counter), dtype="uint32")
    high_signed = _builder_name("high_signed", T.alloc_local((1,), "int32"))
    _builder_emit(T.evaluate(T.ptx.shr.s32(high_signed[0], counter, T.uint32(31))))
    c1 = _builder_scalar("c1", T.reinterpret("uint32", high_signed[0]), dtype="uint32")
    c2 = _builder_scalar("c2", 0, dtype="uint32")
    c3 = _builder_scalar("c3", 0, dtype="uint32")
    k0 = _builder_scalar("k0", seed_lo, dtype="uint32")
    k1 = _builder_scalar("k1", seed_hi, dtype="uint32")
    with T.unroll(PHILOX_ROUNDS) as _round:
        old_c0 = _builder_scalar("old_c0", c0, dtype="uint32")
        old_c2 = _builder_scalar("old_c2", c2, dtype="uint32")
        next_c0 = _builder_scalar(
            "next_c0",
            T.bitwise_xor(T.bitwise_xor(_mul_hi_u32(T.uint32(3449720151), old_c2), c1), k0),
            dtype="uint32",
        )
        next_c2 = _builder_scalar(
            "next_c2",
            T.bitwise_xor(T.bitwise_xor(_mul_hi_u32(T.uint32(3528531795), old_c0), c3), k1),
            dtype="uint32",
        )
        next_c1 = _builder_scalar(
            "next_c1",
            _mul_lo_s32(T.int32(-845247145), T.reinterpret("int32", old_c2)),
            dtype="int32",
        )
        next_c3 = _builder_scalar(
            "next_c3",
            _mul_lo_s32(T.int32(-766435501), T.reinterpret("int32", old_c0)),
            dtype="int32",
        )
        next_k0 = _builder_scalar(
            "next_k0", _add_s32(T.reinterpret("int32", k0), T.int32(-1640531527)), dtype="int32"
        )
        next_k1 = _builder_scalar(
            "next_k1", _add_s32(T.reinterpret("int32", k1), T.int32(-1150833019)), dtype="int32"
        )
        T.buffer_store(c0.buffer, next_c0, [0])
        T.buffer_store(c1.buffer, T.reinterpret("uint32", next_c1), [0])
        T.buffer_store(c2.buffer, next_c2, [0])
        T.buffer_store(c3.buffer, T.reinterpret("uint32", next_c3), [0])
        T.buffer_store(k0.buffer, T.reinterpret("uint32", next_k0), [0])
        T.buffer_store(k1.buffer, T.reinterpret("uint32", next_k1), [0])
    T.buffer_store(random_words, c0, [0])
    T.buffer_store(random_words, c1, [1])
    T.buffer_store(random_words, c2, [2])
    T.buffer_store(random_words, c3, [3])


def _role_load(
    smem_raw,
    smem_addr,
    tensor_state,
    tensor_b,
    tensor_c,
    tensor_x,
    lane,
    batch_i,
    head,
    kv_group,
    state_batch,
    group_base,
    *,
    IS_PAD,
    NTOKENS,
    DIM,
    DSTATE,
    STATE_BYTES,
    OFF_B,
    OFF_C,
    OFF_STATE,
    OFF_X,
    OFF_BAR_BC,
    OFF_BAR_EMPTY,
    OFF_BAR_FULL,
):
    with T.If(lane == 0):
        with T.Then():
            _builder_emit(
                _tma_g2s_4d(
                    smem_raw,
                    group_base + OFF_B,
                    tensor_b,
                    0,
                    kv_group,
                    0,
                    batch_i,
                    group_base + OFF_BAR_BC,
                )
            )
            _builder_emit(
                _tma_g2s_4d(
                    smem_raw,
                    group_base + OFF_C,
                    tensor_c,
                    0,
                    kv_group,
                    0,
                    batch_i,
                    group_base + OFF_BAR_BC,
                )
            )
            _builder_emit(
                _mbarrier_expect_tx_relaxed(
                    smem_raw, group_base + OFF_BAR_BC, 2 * NTOKENS * DSTATE * 2
                )
            )
            _builder_emit(_mbarrier_arrive_count_release(smem_raw, group_base + OFF_BAR_BC, 32))
    _builder_emit(_mbarrier_arrive_wait(smem_addr + T.cast(group_base + OFF_BAR_EMPTY, "uint32")))
    with T.If(lane == 0):
        with T.Then():
            if not IS_PAD:
                _builder_emit(
                    _tma_g2s_4d(
                        smem_raw,
                        group_base + OFF_STATE,
                        tensor_state,
                        0,
                        0,
                        head,
                        state_batch,
                        group_base + OFF_BAR_FULL,
                    )
                )
            _builder_emit(
                _tma_g2s_4d(
                    smem_raw,
                    group_base + OFF_X,
                    tensor_x,
                    0,
                    head,
                    0,
                    batch_i,
                    group_base + OFF_BAR_FULL,
                )
            )
            if IS_PAD:
                _builder_emit(
                    _mbarrier_arrive_expect_tx(
                        smem_raw, group_base + OFF_BAR_FULL, NTOKENS * DIM * 2
                    )
                )
            else:
                _builder_emit(
                    _mbarrier_arrive_expect_tx(
                        smem_raw,
                        group_base + OFF_BAR_FULL,
                        DIM * DSTATE * STATE_BYTES + NTOKENS * DIM * 2,
                    )
                )


def _store_state_row(
    values,
    wr,
    random_words,
    state,
    intermediate_states,
    intermediate_base,
    final_base,
    write_final,
    *,
    DSTATE,
    STATE_DTYPE,
    STATE_VALUES_PER_THREAD,
    HAS_INTERMEDIATE_STATES,
    PHILOX_ROUNDS,
):
    if PHILOX_ROUNDS > 0:
        pair0 = _builder_name("pair0", T.alloc_local((1,), "uint32"))
        pair1 = _builder_name("pair1", T.alloc_local((1,), "uint32"))
        _builder_emit(
            T.evaluate(
                T.ptx.cvt.rs.f16x2.f32(pair0[0], values[wr, 1], values[wr, 0], random_words[0])
            )
        )
        _builder_emit(
            T.evaluate(
                T.ptx.cvt.rs.f16x2.f32(pair1[0], values[wr, 3], values[wr, 2], random_words[1])
            )
        )
        if HAS_INTERMEDIATE_STATES:
            _builder_emit(
                T.evaluate(
                    T.ptx.st.global_.v2.b32(
                        intermediate_states.ptr_to([intermediate_base]), pair0[0], pair1[0]
                    )
                )
            )
            with T.If(T.NE(write_final, T.int32(0))):
                with T.Then():
                    _builder_emit(
                        T.evaluate(
                            T.ptx.st.global_.v2.b32(state.ptr_to([final_base]), pair0[0], pair1[0])
                        )
                    )
        else:
            with T.If(T.NE(write_final, T.int32(0))):
                with T.Then():
                    _builder_emit(
                        T.evaluate(
                            T.ptx.st.global_.v2.b32(state.ptr_to([final_base]), pair0[0], pair1[0])
                        )
                    )
    elif STATE_DTYPE == "float32":
        words = _builder_name("words", T.alloc_local((4,), "uint32"))
        with T.unroll(4) as k:
            T.buffer_store(words, T.reinterpret("uint32", values[wr, k]), [k])
        if HAS_INTERMEDIATE_STATES:
            with T.If(T.NE(write_final, T.int32(0))):
                with T.Then():
                    lo = _builder_name("lo", T.alloc_local((1,), "uint64"))
                    hi = _builder_name("hi", T.alloc_local((1,), "uint64"))
                    _builder_emit(T.evaluate(T.ptx.mov.b64(lo[0], words[0], words[1])))
                    _builder_emit(T.evaluate(T.ptx.mov.b64(hi[0], words[2], words[3])))
                    _builder_emit(
                        T.evaluate(
                            T.ptx.st.global_.v2.b64(
                                intermediate_states.ptr_to([intermediate_base]), lo[0], hi[0]
                            )
                        )
                    )
                    _builder_emit(
                        T.evaluate(
                            T.ptx.st.global_.v2.b64(state.ptr_to([final_base]), lo[0], hi[0])
                        )
                    )
                with T.Else():
                    _builder_emit(
                        T.evaluate(
                            T.ptx.st.global_.v4.b32(
                                intermediate_states.ptr_to([intermediate_base]),
                                words[0],
                                words[1],
                                words[2],
                                words[3],
                            )
                        )
                    )
        else:
            with T.If(T.NE(write_final, T.int32(0))):
                with T.Then():
                    _builder_emit(
                        T.evaluate(
                            T.ptx.st.global_.v4.b32(
                                state.ptr_to([final_base]), words[0], words[1], words[2], words[3]
                            )
                        )
                    )
    else:
        bits = _builder_name("bits", T.alloc_local((4,), "uint16"))
        with T.unroll(STATE_VALUES_PER_THREAD) as k:
            if STATE_DTYPE == "bfloat16":
                T.buffer_store(bits, _f32_to_bf16(values[wr, k]), [k])
            else:
                T.buffer_store(bits, _f32_to_f16(values[wr, k]), [k])
        if DSTATE == 64:
            if HAS_INTERMEDIATE_STATES:
                with T.If(T.NE(write_final, T.int32(0))):
                    with T.Then():
                        word = _builder_name("word", T.alloc_local((1,), "uint32"))
                        _builder_emit(T.evaluate(T.ptx.mov.b32(word[0], bits[0], bits[1])))
                        _builder_emit(
                            T.evaluate(
                                T.ptx.st.global_.b32(
                                    intermediate_states.ptr_to([intermediate_base]), word[0]
                                )
                            )
                        )
                        _builder_emit(
                            T.evaluate(T.ptx.st.global_.b32(state.ptr_to([final_base]), word[0]))
                        )
                    with T.Else():
                        _builder_emit(
                            T.evaluate(
                                T.ptx.st.global_.v2.b16(
                                    intermediate_states.ptr_to([intermediate_base]),
                                    bits[0],
                                    bits[1],
                                )
                            )
                        )
            else:
                with T.If(T.NE(write_final, T.int32(0))):
                    with T.Then():
                        _builder_emit(
                            T.evaluate(
                                T.ptx.st.global_.v2.b16(
                                    state.ptr_to([final_base]), bits[0], bits[1]
                                )
                            )
                        )
        elif DSTATE == 96:
            if HAS_INTERMEDIATE_STATES:
                with T.unroll(3) as k:
                    _builder_emit(
                        T.evaluate(
                            T.ptx.st.global_.b16(
                                intermediate_states.ptr_to([intermediate_base + k]), bits[k]
                            )
                        )
                    )
            with T.If(T.NE(write_final, T.int32(0))):
                with T.Then():
                    with T.unroll(3) as k:
                        _builder_emit(
                            T.evaluate(
                                T.ptx.st.global_.b16(state.ptr_to([final_base + k]), bits[k])
                            )
                        )
        elif HAS_INTERMEDIATE_STATES:
            with T.If(T.NE(write_final, T.int32(0))):
                with T.Then():
                    word0 = _builder_name("word0", T.alloc_local((1,), "uint32"))
                    word1 = _builder_name("word1", T.alloc_local((1,), "uint32"))
                    _builder_emit(T.evaluate(T.ptx.mov.b32(word1[0], bits[2], bits[3])))
                    _builder_emit(T.evaluate(T.ptx.mov.b32(word0[0], bits[0], bits[1])))
                    _builder_emit(
                        T.evaluate(
                            T.ptx.st.global_.v2.b32(
                                intermediate_states.ptr_to([intermediate_base]), word0[0], word1[0]
                            )
                        )
                    )
                    _builder_emit(
                        T.evaluate(
                            T.ptx.st.global_.v2.b32(state.ptr_to([final_base]), word0[0], word1[0])
                        )
                    )
                with T.Else():
                    _builder_emit(
                        T.evaluate(
                            T.ptx.st.global_.v4.b16(
                                intermediate_states.ptr_to([intermediate_base]),
                                bits[0],
                                bits[1],
                                bits[2],
                                bits[3],
                            )
                        )
                    )
        else:
            with T.If(T.NE(write_final, T.int32(0))):
                with T.Then():
                    _builder_emit(
                        T.evaluate(
                            T.ptx.st.global_.v4.b16(
                                state.ptr_to([final_base]), bits[0], bits[1], bits[2], bits[3]
                            )
                        )
                    )


def _role_update_state(
    smem_raw,
    s_u16,
    s_u32,
    state,
    intermediate_states,
    dt,
    matrix_a,
    d_weight,
    dt_bias,
    intermediate_indices,
    rand_seed,
    smem_addr,
    lane,
    compute_warp,
    batch_i,
    head,
    state_batch,
    group_base,
    state_stride_batch,
    dt_stride_batch,
    dt_stride_mtp,
    intermediate_state_stride_batch,
    dt_softplus,
    update_state,
    *,
    IS_PAD,
    NHEADS,
    DIM,
    DSTATE,
    NTOKENS,
    NUM_PASSES,
    STATE_DTYPE,
    WEIGHT_DTYPE,
    STATE_BYTES,
    STATE_VALUES_PER_THREAD,
    HAS_INTERMEDIATE_STATES,
    HAS_D,
    HAS_DT_BIAS,
    PHILOX_ROUNDS,
    OFF_B,
    OFF_C,
    OFF_DT,
    OFF_STATE,
    OFF_X,
    OFF_OUT,
    OFF_BAR_BC,
    OFF_BAR_EMPTY,
    OFF_BAR_FULL,
    OFF_BAR_OUT,
    OFF_BAR_DONE,
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
    _builder_emit(_mbarrier_arrive(smem_raw, group_base + OFF_BAR_EMPTY))
    _builder_emit(_mbarrier_arrive_wait(smem_addr + T.cast(group_base + OFF_BAR_BC, "uint32")))
    with T.serial((NTOKENS + 3) // 4) as step_iter:
        dt_step = _builder_scalar("dt_step", compute_warp + step_iter * 4, dtype="int32")
        with T.If(T.And(dt_step < NTOKENS, lane == 0)):
            with T.Then():
                dt_value = _builder_scalar(
                    "dt_value",
                    _load_weight(
                        dt,
                        T.cast(batch_i, "int64") * dt_stride_batch
                        + T.cast(dt_step, "int64") * dt_stride_mtp
                        + head,
                        WEIGHT_DTYPE,
                    ),
                    dtype="float32",
                )
                if HAS_DT_BIAS:
                    T.buffer_store(dt_value.buffer, _add(dt_value, bias_value), [0])
                with T.If(T.And(dt_softplus != 0, dt_value <= T.float32(20.0))):
                    with T.Then():
                        exp_arg = _builder_scalar(
                            "exp_arg", _mul(dt_value, T.float32(_LOG2_E)), dtype="float32"
                        )
                        exp_value = _builder_scalar("exp_value", _exp2(exp_arg), dtype="float32")
                        log_value = _builder_scalar(
                            "log_value", _log2(_add(T.float32(1.0), exp_value)), dtype="float32"
                        )
                        T.buffer_store(dt_value.buffer, _mul(log_value, T.float32(_LN_2)), [0])
                _builder_emit(
                    T.evaluate(
                        T.ptx.st.shared.b32(
                            s_u32.ptr_to([(group_base + OFF_DT) // 4 + dt_step]),
                            T.reinterpret("uint32", dt_value),
                        )
                    )
                )
    _builder_emit(_mbarrier_arrive_wait(smem_addr + T.cast(group_base + OFF_BAR_FULL, "uint32")))
    lane_indicator = _builder_scalar(
        "lane_indicator", T.if_then_else(lane == 0, T.float32(1.0), T.float32(0.0)), dtype="float32"
    )
    seed_u64 = _builder_scalar("seed_u64", T.reinterpret("uint64", random_seed), dtype="uint64")
    seed_lo = _builder_scalar("seed_lo", T.cast(seed_u64, "uint32"), dtype="uint32")
    seed_hi = _builder_scalar(
        "seed_hi", T.cast(T.shift_right(seed_u64, T.uint64(32)), "uint32"), dtype="uint32"
    )
    state_head_i32 = _builder_scalar(
        "state_head_i32",
        T.cast(state_batch * state_stride_batch + T.cast(head * DIM * DSTATE, "int64"), "int32"),
        dtype="int32",
    )
    pass_idx = _builder_scalar("pass_idx", 0, dtype="int32")
    with T.While(pass_idx < NUM_PASSES):
        row_offset = _builder_scalar(
            "row_offset", compute_warp * (DIM // 4) + pass_idx * 4, dtype="int32"
        )
        r_state = _builder_name("r_state", T.alloc_local((4, STATE_VALUES_PER_THREAD), "float32"))
        with T.unroll(4) as wr:
            dd = _builder_scalar("dd", row_offset + wr, dtype="int32")
            if IS_PAD:
                with T.unroll(STATE_VALUES_PER_THREAD) as ii:
                    T.buffer_store(r_state, 0.0, [wr, ii])
            elif STATE_DTYPE == "float32":
                state_words = _builder_name("state_words", T.alloc_local((4,), "uint32"))
                _builder_emit(
                    _shared_load_v4_b32(
                        s_u32,
                        (group_base + OFF_STATE) // 4
                        + dd * DSTATE
                        + lane * STATE_VALUES_PER_THREAD,
                        state_words,
                    )
                )
                with T.unroll(4) as ii:
                    T.buffer_store(r_state, T.reinterpret("float32", state_words[ii]), [wr, ii])
            else:
                state_bits = _builder_name("state_bits", T.alloc_local((4,), "uint16"))
                state_index = _builder_scalar(
                    "state_index",
                    (group_base + OFF_STATE) // 2 + dd * DSTATE + lane * STATE_VALUES_PER_THREAD,
                    dtype="int32",
                )
                if DSTATE == 64:
                    _builder_emit(_shared_load_v2_b16(s_u16, state_index, state_bits))
                elif DSTATE == 96:
                    with T.unroll(3) as ii:
                        T.buffer_store(state_bits, _shared_load_u16(s_u16, state_index + ii), [ii])
                else:
                    _builder_emit(_shared_load_v4_b16(s_u16, state_index, state_bits))
                with T.unroll(STATE_VALUES_PER_THREAD) as ii:
                    if STATE_DTYPE == "bfloat16":
                        T.buffer_store(r_state, _bf16_to_f32(state_bits[ii]), [wr, ii])
                    else:
                        T.buffer_store(r_state, _f16_to_f32(state_bits[ii]), [wr, ii])
        row_random = _builder_name("row_random", T.alloc_local((4, 4), "uint32"))
        if PHILOX_ROUNDS > 0 and (not IS_PAD):
            with T.unroll(4) as wr:
                dd = _builder_scalar("dd", row_offset + wr, dtype="int32")
                random_counter = _builder_scalar(
                    "random_counter",
                    _add_s32(state_head_i32, dd * DSTATE + lane * STATE_VALUES_PER_THREAD),
                    dtype="int32",
                )
                random_words = _builder_name("random_words", T.alloc_local((4,), "uint32"))
                _builder_emit(
                    _philox4x32(
                        random_words, seed_lo, seed_hi, random_counter, PHILOX_ROUNDS=PHILOX_ROUNDS
                    )
                )
                with T.unroll(4) as ri:
                    T.buffer_store(row_random, random_words[ri], [wr, ri])
        step = _builder_scalar("step", 0, dtype="int32")
        with T.While(step < NTOKENS):
            shared_dt = _builder_scalar(
                "shared_dt",
                T.reinterpret(
                    "float32", _shared_load_u32(s_u32, (group_base + OFF_DT) // 4 + step)
                ),
                dtype="float32",
            )
            da_value = _builder_scalar(
                "da_value",
                _exp2(_mul(_mul(a_value, shared_dt), T.float32(_LOG2_E))),
                dtype="float32",
            )
            b_values = _builder_name(
                "b_values", T.alloc_local((STATE_VALUES_PER_THREAD,), "float32")
            )
            c_values = _builder_name(
                "c_values", T.alloc_local((STATE_VALUES_PER_THREAD,), "float32")
            )
            b_bits = _builder_name("b_bits", T.alloc_local((4,), "uint16"))
            c_bits = _builder_name("c_bits", T.alloc_local((4,), "uint16"))
            bc_col = _builder_scalar("bc_col", lane * STATE_VALUES_PER_THREAD, dtype="int32")
            b_index = _builder_scalar(
                "b_index", (group_base + OFF_B) // 2 + step * DSTATE + bc_col, dtype="int32"
            )
            c_index = _builder_scalar(
                "c_index", (group_base + OFF_C) // 2 + step * DSTATE + bc_col, dtype="int32"
            )
            if DSTATE == 64:
                _builder_emit(_shared_load_v2_b16(s_u16, b_index, b_bits))
                _builder_emit(_shared_load_v2_b16(s_u16, c_index, c_bits))
            elif DSTATE == 96:
                with T.unroll(3) as ii:
                    T.buffer_store(b_bits, _shared_load_u16(s_u16, b_index + ii), [ii])
                    T.buffer_store(c_bits, _shared_load_u16(s_u16, c_index + ii), [ii])
            else:
                _builder_emit(_shared_load_v4_b16(s_u16, b_index, b_bits))
                _builder_emit(_shared_load_v4_b16(s_u16, c_index, c_bits))
            with T.unroll(STATE_VALUES_PER_THREAD) as ii:
                T.buffer_store(b_values, _bf16_to_f32(b_bits[ii]), [ii])
                T.buffer_store(c_values, _bf16_to_f32(c_bits[ii]), [ii])
            with T.unroll(4) as wr:
                dd = _builder_scalar("dd", row_offset + wr, dtype="int32")
                x_value = _builder_scalar(
                    "x_value",
                    _bf16_to_f32(
                        _shared_load_u16(s_u16, (group_base + OFF_X) // 2 + step * DIM + dd)
                    ),
                    dtype="float32",
                )
                d_times_x = _builder_scalar("d_times_x", _mul(d_value, x_value), dtype="float32")
                out_value = _builder_scalar("out_value", 0.0, dtype="float32")
                with T.unroll(STATE_VALUES_PER_THREAD) as ii:
                    db_value = _builder_scalar(
                        "db_value", _mul(b_values[ii], shared_dt), dtype="float32"
                    )
                    db_x = _builder_scalar("db_x", _mul(db_value, x_value), dtype="float32")
                    new_state = _builder_scalar(
                        "new_state", _fma(r_state[wr, ii], da_value, db_x), dtype="float32"
                    )
                    T.buffer_store(r_state, new_state, [wr, ii])
                    with T.If(ii == 0):
                        with T.Then():
                            state_c = _builder_scalar(
                                "state_c", _mul(new_state, c_values[ii]), dtype="float32"
                            )
                            T.buffer_store(
                                out_value.buffer, _fma(d_times_x, lane_indicator, state_c), [0]
                            )
                        with T.Else():
                            T.buffer_store(
                                out_value.buffer, _fma(new_state, c_values[ii], out_value), [0]
                            )
                with T.unroll(5) as delta_i:
                    delta = _builder_scalar(
                        "delta", T.shift_right(T.int32(16), delta_i), dtype="int32"
                    )
                    peer = _builder_scalar(
                        "peer",
                        T.cuda.__shfl_down_sync(T.uint32(4294967295), out_value, delta, 32),
                        dtype="float32",
                    )
                    T.buffer_store(out_value.buffer, _add(out_value, peer), [0])
                with T.If(lane == 0):
                    with T.Then():
                        _builder_emit(
                            T.evaluate(
                                T.ptx.st.shared.b32(
                                    s_u32.ptr_to([(group_base + OFF_OUT) // 4 + step * DIM + dd]),
                                    T.reinterpret("uint32", out_value),
                                )
                            )
                        )
            if not IS_PAD:
                write_final = _builder_scalar(
                    "write_final",
                    T.if_then_else(step == NTOKENS - 1, T.if_then_else(update_state != 0, 1, 0), 0),
                    dtype="int32",
                )
                if HAS_INTERMEDIATE_STATES:
                    with T.If(write_final != 0):
                        with T.Then():
                            with T.unroll(4) as wr:
                                dd = _builder_scalar("dd", row_offset + wr, dtype="int32")
                                intermediate_base = _builder_scalar(
                                    "intermediate_base",
                                    icache_idx * intermediate_state_stride_batch
                                    + T.cast(step * NHEADS * DIM * DSTATE, "int64")
                                    + head * DIM * DSTATE
                                    + dd * DSTATE
                                    + lane * STATE_VALUES_PER_THREAD,
                                    dtype="int64",
                                )
                                final_base = _builder_scalar(
                                    "final_base",
                                    state_batch * state_stride_batch
                                    + head * DIM * DSTATE
                                    + dd * DSTATE
                                    + lane * STATE_VALUES_PER_THREAD,
                                    dtype="int64",
                                )
                                random_words = _builder_name(
                                    "random_words", T.alloc_local((4,), "uint32")
                                )
                                with T.unroll(4) as ri:
                                    T.buffer_store(random_words, row_random[wr, ri], [ri])
                                _builder_emit(
                                    _store_state_row(
                                        r_state,
                                        wr,
                                        random_words,
                                        state,
                                        intermediate_states,
                                        intermediate_base,
                                        final_base,
                                        T.int32(1),
                                        DSTATE=DSTATE,
                                        STATE_DTYPE=STATE_DTYPE,
                                        STATE_VALUES_PER_THREAD=STATE_VALUES_PER_THREAD,
                                        HAS_INTERMEDIATE_STATES=True,
                                        PHILOX_ROUNDS=PHILOX_ROUNDS,
                                    )
                                )
                        with T.Else():
                            with T.unroll(4) as wr:
                                dd = _builder_scalar("dd", row_offset + wr, dtype="int32")
                                intermediate_base = _builder_scalar(
                                    "intermediate_base",
                                    icache_idx * intermediate_state_stride_batch
                                    + T.cast(step * NHEADS * DIM * DSTATE, "int64")
                                    + head * DIM * DSTATE
                                    + dd * DSTATE
                                    + lane * STATE_VALUES_PER_THREAD,
                                    dtype="int64",
                                )
                                random_words = _builder_name(
                                    "random_words", T.alloc_local((4,), "uint32")
                                )
                                with T.unroll(4) as ri:
                                    T.buffer_store(random_words, row_random[wr, ri], [ri])
                                _builder_emit(
                                    _store_state_row(
                                        r_state,
                                        wr,
                                        random_words,
                                        state,
                                        intermediate_states,
                                        intermediate_base,
                                        T.int64(0),
                                        T.int32(0),
                                        DSTATE=DSTATE,
                                        STATE_DTYPE=STATE_DTYPE,
                                        STATE_VALUES_PER_THREAD=STATE_VALUES_PER_THREAD,
                                        HAS_INTERMEDIATE_STATES=True,
                                        PHILOX_ROUNDS=PHILOX_ROUNDS,
                                    )
                                )
                else:
                    with T.If(write_final != 0):
                        with T.Then():
                            with T.unroll(4) as wr:
                                dd = _builder_scalar("dd", row_offset + wr, dtype="int32")
                                final_base = _builder_scalar(
                                    "final_base",
                                    state_batch * state_stride_batch
                                    + head * DIM * DSTATE
                                    + dd * DSTATE
                                    + lane * STATE_VALUES_PER_THREAD,
                                    dtype="int64",
                                )
                                random_words = _builder_name(
                                    "random_words", T.alloc_local((4,), "uint32")
                                )
                                with T.unroll(4) as ri:
                                    T.buffer_store(random_words, row_random[wr, ri], [ri])
                                _builder_emit(
                                    _store_state_row(
                                        r_state,
                                        wr,
                                        random_words,
                                        state,
                                        intermediate_states,
                                        T.int64(0),
                                        final_base,
                                        T.int32(1),
                                        DSTATE=DSTATE,
                                        STATE_DTYPE=STATE_DTYPE,
                                        STATE_VALUES_PER_THREAD=STATE_VALUES_PER_THREAD,
                                        HAS_INTERMEDIATE_STATES=False,
                                        PHILOX_ROUNDS=PHILOX_ROUNDS,
                                    )
                                )
            T.buffer_store(step.buffer, step + 1, [0])
        T.buffer_store(pass_idx.buffer, pass_idx + 1, [0])
    _builder_emit(_mbarrier_arrive(smem_raw, group_base + OFF_BAR_OUT))
    _builder_emit(_mbarrier_arrive_wait(smem_addr + T.cast(group_base + OFF_BAR_DONE, "uint32")))


def _role_epilogue(
    smem_raw,
    s_u16,
    s_u32,
    z,
    output,
    smem_addr,
    lane,
    batch_i,
    head_base,
    z_stride_batch,
    z_stride_mtp,
    out_stride_batch,
    out_stride_mtp,
    *,
    NHEADS,
    DIM,
    NTOKENS,
    HAS_Z,
    GROUP_BYTES,
    OFF_OUT,
    OFF_BAR_OUT,
    OFF_BAR_DONE,
):
    with T.serial(3) as group:
        head = _builder_scalar("head", head_base + group, dtype="int32")
        with T.If(head < NHEADS):
            with T.Then():
                group_base = _builder_scalar("group_base", group * GROUP_BYTES, dtype="int32")
                _builder_emit(
                    _mbarrier_arrive_wait(smem_addr + T.cast(group_base + OFF_BAR_OUT, "uint32"))
                )
                with T.unroll(NTOKENS) as step:
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
                    out_words = _builder_name("out_words", T.alloc_local((4,), "uint32"))
                    z_bits = _builder_name("z_bits", T.alloc_local((4,), "uint16"))
                    output_bits = _builder_name("output_bits", T.alloc_local((4,), "uint16"))
                    if DIM == 64:
                        d = _builder_scalar("d", lane * 2, dtype="int32")
                        _builder_emit(
                            T.evaluate(
                                T.ptx.ld.shared.v2.b32(
                                    out_words[0],
                                    out_words[1],
                                    s_u32.ptr_to([(group_base + OFF_OUT) // 4 + step * DIM + d]),
                                )
                            )
                        )
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
                                exp_neg_z = _builder_scalar(
                                    "exp_neg_z",
                                    _exp2(_mul(_sub(T.float32(0.0), z_value), T.float32(_LOG2_E))),
                                    dtype="float32",
                                )
                                sigmoid_z = _builder_scalar(
                                    "sigmoid_z",
                                    _div(T.float32(1.0), _add(T.float32(1.0), exp_neg_z)),
                                    dtype="float32",
                                )
                                T.buffer_store(
                                    out_value.buffer, _mul(out_value, _mul(z_value, sigmoid_z)), [0]
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
                        _builder_emit(
                            _shared_load_v4_b32(
                                s_u32, (group_base + OFF_OUT) // 4 + step * DIM + d, out_words
                            )
                        )
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
                                exp_neg_z = _builder_scalar(
                                    "exp_neg_z",
                                    _exp2(_mul(_sub(T.float32(0.0), z_value), T.float32(_LOG2_E))),
                                    dtype="float32",
                                )
                                sigmoid_z = _builder_scalar(
                                    "sigmoid_z",
                                    _div(T.float32(1.0), _add(T.float32(1.0), exp_neg_z)),
                                    dtype="float32",
                                )
                                T.buffer_store(
                                    out_value.buffer, _mul(out_value, _mul(z_value, sigmoid_z)), [0]
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
                _builder_emit(_mbarrier_arrive(smem_raw, group_base + OFF_BAR_DONE))


def _build_selective_state_update_mtp_vertical(
    *,
    BATCH: T.constexpr,
    NHEADS: T.constexpr,
    NUM_HEAD_CHUNKS: T.constexpr,
    DIM: T.constexpr,
    DSTATE: T.constexpr,
    NTOKENS: T.constexpr,
    HEADS_PER_GROUP: T.constexpr,
    NUM_PASSES: T.constexpr,
    STATE_BYTES: T.constexpr,
    STATE_VALUES_PER_THREAD: T.constexpr,
    HAS_STATE_INDICES: T.constexpr,
    HAS_INTERMEDIATE_STATES: T.constexpr,
    HAS_Z: T.constexpr,
    HAS_D: T.constexpr,
    HAS_DT_BIAS: T.constexpr,
    PHILOX_ROUNDS: T.constexpr,
    OFF_B: T.constexpr,
    OFF_C: T.constexpr,
    OFF_DT: T.constexpr,
    OFF_STATE: T.constexpr,
    OFF_X: T.constexpr,
    OFF_OUT: T.constexpr,
    OFF_BAR_BC: T.constexpr,
    OFF_BAR_EMPTY: T.constexpr,
    OFF_BAR_FULL: T.constexpr,
    OFF_BAR_OUT: T.constexpr,
    OFF_BAR_DONE: T.constexpr,
    GROUP_BYTES: T.constexpr,
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
            T.func_name("_selective_state_update_mtp_vertical")
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
            _builder_enter(T.attr({"tirx.launch_bounds_min_blocks_per_sm": 2}))
            _builder_values_112 = T.cta_id([BATCH, NUM_HEAD_CHUNKS])
            batch_i, head_chunk = _builder_values_112
            IRBuilder.name("batch_i", batch_i)
            IRBuilder.name("head_chunk", head_chunk)
            _builder_values_113 = T.thread_id([32, 16])
            lane, warp = _builder_values_113
            IRBuilder.name("lane", lane)
            IRBuilder.name("warp", warp)
            head_base = _builder_scalar("head_base", head_chunk * 3, dtype="int32")
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
            pool = _builder_meta("pool", T.SMEMPool())
            smem_raw = _builder_name("smem_raw", pool.alloc((SHARED_BYTES,), "uint8", align=128))
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
            _builder_emit(pool.commit())
            smem_addr = _builder_scalar(
                "smem_addr", T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0])), dtype="uint32"
            )
            with T.If(T.And(warp == 0, lane == 0)):
                with T.Then():
                    with T.serial(3) as group:
                        with T.If(head_base + group < NHEADS):
                            with T.Then():
                                group_base = _builder_scalar(
                                    "group_base", group * GROUP_BYTES, dtype="int32"
                                )
                                _builder_emit(
                                    T.evaluate(
                                        T.ptx.mbarrier.init.shared.b64(
                                            smem_raw.ptr_to([group_base + OFF_BAR_BC]),
                                            T.uint32(160),
                                        )
                                    )
                                )
                                _builder_emit(
                                    T.evaluate(
                                        T.ptx.mbarrier.init.shared.b64(
                                            smem_raw.ptr_to([group_base + OFF_BAR_EMPTY]),
                                            T.uint32(160),
                                        )
                                    )
                                )
                                _builder_emit(
                                    T.evaluate(
                                        T.ptx.mbarrier.init.shared.b64(
                                            smem_raw.ptr_to([group_base + OFF_BAR_FULL]),
                                            T.uint32(129),
                                        )
                                    )
                                )
                                _builder_emit(
                                    T.evaluate(
                                        T.ptx.mbarrier.init.shared.b64(
                                            smem_raw.ptr_to([group_base + OFF_BAR_OUT]),
                                            T.uint32(160),
                                        )
                                    )
                                )
                                _builder_emit(
                                    T.evaluate(
                                        T.ptx.mbarrier.init.shared.b64(
                                            smem_raw.ptr_to([group_base + OFF_BAR_DONE]),
                                            T.uint32(160),
                                        )
                                    )
                                )
            _builder_emit(T.cuda.cta_sync())

            def dispatch_pad(*, IS_PAD):
                with T.If(warp < 12):
                    with T.Then():
                        group = _builder_scalar("group", warp // 4, dtype="int32")
                        head = _builder_scalar("head", head_base + group, dtype="int32")
                        with T.If(head < NHEADS):
                            with T.Then():
                                _builder_emit(
                                    _role_update_state(
                                        smem_raw,
                                        s_u16,
                                        s_u32,
                                        state,
                                        intermediate_states,
                                        dt,
                                        matrix_a,
                                        d_weight,
                                        dt_bias,
                                        intermediate_indices,
                                        rand_seed,
                                        smem_addr,
                                        lane,
                                        warp % 4,
                                        batch_i,
                                        head,
                                        state_batch,
                                        group * GROUP_BYTES,
                                        state_stride_batch,
                                        dt_stride_batch,
                                        dt_stride_mtp,
                                        T.int64(NTOKENS * NHEADS * DIM * DSTATE),
                                        dt_softplus,
                                        update_state,
                                        IS_PAD=IS_PAD,
                                        NHEADS=NHEADS,
                                        DIM=DIM,
                                        DSTATE=DSTATE,
                                        NTOKENS=NTOKENS,
                                        NUM_PASSES=NUM_PASSES,
                                        STATE_DTYPE=STATE_DTYPE,
                                        WEIGHT_DTYPE=WEIGHT_DTYPE,
                                        STATE_BYTES=STATE_BYTES,
                                        STATE_VALUES_PER_THREAD=STATE_VALUES_PER_THREAD,
                                        HAS_INTERMEDIATE_STATES=HAS_INTERMEDIATE_STATES,
                                        HAS_D=HAS_D,
                                        HAS_DT_BIAS=HAS_DT_BIAS,
                                        PHILOX_ROUNDS=PHILOX_ROUNDS,
                                        OFF_B=OFF_B,
                                        OFF_C=OFF_C,
                                        OFF_DT=OFF_DT,
                                        OFF_STATE=OFF_STATE,
                                        OFF_X=OFF_X,
                                        OFF_OUT=OFF_OUT,
                                        OFF_BAR_BC=OFF_BAR_BC,
                                        OFF_BAR_EMPTY=OFF_BAR_EMPTY,
                                        OFF_BAR_FULL=OFF_BAR_FULL,
                                        OFF_BAR_OUT=OFF_BAR_OUT,
                                        OFF_BAR_DONE=OFF_BAR_DONE,
                                    )
                                )
                    with T.Else():
                        with T.If(warp < 15):
                            with T.Then():
                                group = _builder_scalar("group", warp - 12, dtype="int32")
                                head = _builder_scalar("head", head_base + group, dtype="int32")
                                with T.If(head < NHEADS):
                                    with T.Then():
                                        _builder_emit(
                                            _role_load(
                                                smem_raw,
                                                smem_addr,
                                                tensor_state,
                                                tensor_b,
                                                tensor_c,
                                                tensor_x,
                                                lane,
                                                batch_i,
                                                head,
                                                head // HEADS_PER_GROUP,
                                                state_batch,
                                                group * GROUP_BYTES,
                                                IS_PAD=IS_PAD,
                                                NTOKENS=NTOKENS,
                                                DIM=DIM,
                                                DSTATE=DSTATE,
                                                STATE_BYTES=STATE_BYTES,
                                                OFF_B=OFF_B,
                                                OFF_C=OFF_C,
                                                OFF_STATE=OFF_STATE,
                                                OFF_X=OFF_X,
                                                OFF_BAR_BC=OFF_BAR_BC,
                                                OFF_BAR_EMPTY=OFF_BAR_EMPTY,
                                                OFF_BAR_FULL=OFF_BAR_FULL,
                                            )
                                        )
                            with T.Else():
                                _builder_emit(
                                    _role_epilogue(
                                        smem_raw,
                                        s_u16,
                                        s_u32,
                                        z,
                                        output,
                                        smem_addr,
                                        lane,
                                        batch_i,
                                        head_base,
                                        z_stride_batch,
                                        z_stride_mtp,
                                        out_stride_batch,
                                        out_stride_mtp,
                                        NHEADS=NHEADS,
                                        DIM=DIM,
                                        NTOKENS=NTOKENS,
                                        HAS_Z=HAS_Z,
                                        GROUP_BYTES=GROUP_BYTES,
                                        OFF_OUT=OFF_OUT,
                                        OFF_BAR_OUT=OFF_BAR_OUT,
                                        OFF_BAR_DONE=OFF_BAR_DONE,
                                    )
                                )

            with T.If(state_batch == T.cast(pad_slot_id, "int64")):
                with T.Then():
                    _builder_emit(dispatch_pad(IS_PAD=True))
                with T.Else():
                    _builder_emit(dispatch_pad(IS_PAD=False))
    return builder.get()


def _specialization(config: dict[str, Any]) -> dict[str, Any]:
    if str(config.get("mode", "fixed")).startswith("varlen"):
        raise ValueError("MTP vertical does not support varlen inputs")
    if str(config.get("state_dtype")) == "int16":
        raise ValueError("MTP vertical does not support scaled state")
    if str(config.get("input_dtype", "bfloat16")) != "bfloat16":
        raise ValueError("MTP vertical is scoped to bfloat16 input")
    if str(config.get("matrix_a_dtype", "float32")) != "float32":
        raise ValueError("MTP vertical is scoped to float32 matrix A")

    base = _simple._specialization(config)
    nheads = int(config["nheads"])
    dim = int(config["dim"])
    dstate = int(config["dstate"])
    tokens = int(config["tokens"])
    heads_per_group = int(config["heads_per_group"])
    state_dtype = str(config["state_dtype"])
    philox_rounds = int(config.get("philox_rounds", 0))
    if nheads % heads_per_group:
        raise ValueError("nheads must be divisible by heads_per_group")
    if dim not in (64, 128):
        raise ValueError("MTP vertical requires DIM in {64, 128}")
    if dstate not in (64, 96, 128):
        raise ValueError("MTP vertical requires DSTATE in {64, 96, 128}")
    if philox_rounds not in (0, 10):
        raise ValueError("MTP vertical stochastic rounding supports philox_rounds in {0, 10}")
    if philox_rounds and state_dtype != "float16":
        raise ValueError("MTP vertical Philox is restricted to float16 state")

    state_bytes = 4 if state_dtype == "float32" else 2
    off_b = 0
    off_c = _simple._align_up(off_b + tokens * dstate * 2, 128)
    off_dt = off_c + tokens * dstate * 2
    off_state = _simple._align_up(off_dt + tokens * 4, 128)
    off_x = _simple._align_up(off_state + dim * dstate * state_bytes, 128)
    off_out = off_x + tokens * dim * 2
    off_bar_bc = off_out + tokens * dim * 4
    off_bar_empty = off_bar_bc + 8
    off_bar_full = off_bar_empty + 8
    off_bar_out = off_bar_full + 8
    off_bar_done = off_bar_out + 8
    group_bytes = _simple._align_up(off_bar_done + 8, 128)

    return {
        "BATCH": int(config["batch"]),
        "NHEADS": nheads,
        "NUM_HEAD_CHUNKS": (nheads + 2) // 3,
        "DIM": dim,
        "DSTATE": dstate,
        "NTOKENS": tokens,
        "HEADS_PER_GROUP": heads_per_group,
        "NUM_PASSES": dim // 16,
        "STATE_BYTES": state_bytes,
        "STATE_VALUES_PER_THREAD": dstate // 32,
        "HAS_STATE_INDICES": bool(config.get("has_state_indices", True)),
        "HAS_INTERMEDIATE_STATES": bool(config.get("has_intermediate_states", False)),
        "HAS_Z": bool(config.get("has_z", False)),
        "HAS_D": bool(config.get("has_d", True)),
        "HAS_DT_BIAS": bool(config.get("has_dt_bias", True)),
        "PHILOX_ROUNDS": philox_rounds,
        "OFF_B": off_b,
        "OFF_C": off_c,
        "OFF_DT": off_dt,
        "OFF_STATE": off_state,
        "OFF_X": off_x,
        "OFF_OUT": off_out,
        "OFF_BAR_BC": off_bar_bc,
        "OFF_BAR_EMPTY": off_bar_empty,
        "OFF_BAR_FULL": off_bar_full,
        "OFF_BAR_OUT": off_bar_out,
        "OFF_BAR_DONE": off_bar_done,
        "GROUP_BYTES": group_bytes,
        "SHARED_BYTES": 3 * group_bytes,
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
    kernel = _build_selective_state_update_mtp_vertical(**_specialization(kwargs))
    return kernel.with_attr(
        "tirx.kernel_launch_params",
        ["blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y", "tirx.use_dyn_shared_memory"],
    )


class _AlignedTensorMap:
    """Host storage for one 128-byte-aligned CUtensorMap payload."""

    def __init__(self) -> None:
        self._storage = ctypes.create_string_buffer(128 + 128)
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
        raise ValueError(f"vertical {name} TensorMap base must be 128-byte aligned")
    if strides[0] != 1:
        raise ValueError(f"vertical {name} TensorMap innermost stride must be one")
    element_bytes = tensor.element_size()
    for axis, stride in enumerate(strides[1:], start=1):
        if stride * element_bytes % 16:
            raise ValueError(
                f"vertical {name} TensorMap byte stride {axis} must be 16-byte aligned"
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
        0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        0,  # CU_TENSOR_MAP_SWIZZLE_NONE
        2,  # CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
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
    state_dtype = spec["STATE_DTYPE"]

    state = case["tirx_state_storage"]
    matrix_b = case["matrix_b"]
    matrix_c = case["matrix_c"]
    x = case["x"]
    return {
        "state": _encode_tensor_map(
            state,
            dtype=state_dtype,
            shape=(dstate, dim, nheads, state_slots),
            strides=(1, dstate, dstate * dim, state_stride),
            box=(dstate, dim, 1, 1),
            name="state",
        ),
        "b": _encode_tensor_map(
            matrix_b,
            dtype="bfloat16",
            shape=(dstate, ngroups, tokens, spec["BATCH"]),
            strides=(1, matrix_b.stride(2), matrix_b.stride(1), matrix_b.stride(0)),
            box=(dstate, 1, tokens, 1),
            name="B",
        ),
        "c": _encode_tensor_map(
            matrix_c,
            dtype="bfloat16",
            shape=(dstate, ngroups, tokens, spec["BATCH"]),
            strides=(1, matrix_c.stride(2), matrix_c.stride(1), matrix_c.stride(0)),
            box=(dstate, 1, tokens, 1),
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
        # The frozen vertical kernel intentionally treats the index pointer as
        # flat and reads element ``batch``.  Give those first B elements unique
        # slots so the final-state oracle is deterministic rather than a race
        # between four batches that inherited the same repeated row value.
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
    oracle = _load_oracle()
    result = oracle(
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
        rand_seed=(case["rand_seed"] if int(config.get("philox_rounds", 0)) else None),
        philox_rounds=int(config.get("philox_rounds", 0)),
        cache_steps=int(config["tokens"]),
        algorithm="vertical",
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
        raise AssertionError(f"expected vertical rejection containing {expected_rejection!r}")

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
