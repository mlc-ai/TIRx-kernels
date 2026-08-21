# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's selective-state-update MTP vertical kernel.

Upstream source: include/flashinfer/mamba/kernel_selective_state_update_mtp_vertical.cuh.
"""

import ctypes
import functools
from typing import Any

import torch

import tirx_kernels.kern as K

from . import selective_state_update_mtp_simple as _simple
from .selective_state_update_mtp_simple import _case, _shfl_down_f32

KERNEL_META = {
    "name": "selective_state_update_mtp_vertical",
    "category": "flashinfer",
    "compute_capability": 10,
}


_LOG2_E = _simple._LOG2_E
_LN_2 = _simple._LN_2
_global_load_index_s64 = _simple._global_load_index_s64
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


def _mbarrier_arrive_wait(barrier):
    token = K.local_scalar("uint64")
    done = K.local_scalar("uint32")
    K.ptx.mbarrier.arrive.shared__cta.b64(token, barrier, K.uint32(1))
    with K.While(True):
        K.ptx.mbarrier.try_wait.shared__cta.b64(done, barrier, token)
        with K.If(done != K.uint32(0)), K.Then():
            K.Break()


def _tma_g2s_4d(dst, tensor_map, c0, c1, c2, c3, barrier):
    K.ptx[_TMA_G2S_4D](
        dst,
        K.address_of(tensor_map),
        K.cast(c0, "int32"),
        K.cast(c1, "int32"),
        K.cast(c2, "int32"),
        K.cast(c3, "int32"),
        barrier,
    )


def _philox4x32(random_words, seed_lo, seed_hi, counter, *, PHILOX_ROUNDS):
    c0 = K.local_scalar("uint32", init=K.reinterpret("uint32", counter))
    high_signed = K.local_scalar("int32")
    K.ptx.shr.s32(high_signed, counter, K.uint32(31))
    c1 = K.local_scalar("uint32", init=K.reinterpret("uint32", high_signed))
    c2 = K.local_scalar("uint32", init=0)
    c3 = K.local_scalar("uint32", init=0)
    k0 = K.local_scalar("uint32", init=seed_lo)
    k1 = K.local_scalar("uint32", init=seed_hi)
    with K.unroll(PHILOX_ROUNDS) as _round:
        old_c0 = K.local_scalar("uint32", init=c0)
        old_c2 = K.local_scalar("uint32", init=c2)
        next_c0 = K.local_scalar("uint32")
        mul_hi_0 = K.local_scalar("uint32")
        K.ptx["mul.hi.u32"](mul_hi_0, K.uint32(0xCD9E8D57), old_c2)
        K.assign(next_c0, K.bitwise_xor(K.bitwise_xor(mul_hi_0, c1), k0))
        next_c2 = K.local_scalar("uint32")
        mul_hi_1 = K.local_scalar("uint32")
        K.ptx["mul.hi.u32"](mul_hi_1, K.uint32(0xD2511F53), old_c0)
        K.assign(next_c2, K.bitwise_xor(K.bitwise_xor(mul_hi_1, c3), k1))
        next_c1 = K.local_scalar("int32")
        K.ptx["mul.lo.s32"](next_c1, K.int32(-845247145), K.reinterpret("int32", old_c2))
        next_c3 = K.local_scalar("int32")
        K.ptx["mul.lo.s32"](next_c3, K.int32(-766435501), K.reinterpret("int32", old_c0))
        next_k0 = K.local_scalar("int32")
        K.ptx["add.s32"](next_k0, K.reinterpret("int32", k0), K.int32(-1640531527))
        next_k1 = K.local_scalar("int32")
        K.ptx["add.s32"](next_k1, K.reinterpret("int32", k1), K.int32(-1150833019))
        K.assign(c0, next_c0)
        K.assign(c1, K.reinterpret("uint32", next_c1))
        K.assign(c2, next_c2)
        K.assign(c3, K.reinterpret("uint32", next_c3))
        K.assign(k0, K.reinterpret("uint32", next_k0))
        K.assign(k1, K.reinterpret("uint32", next_k1))
    K.ptx.mov.b32(random_words[0], c0)
    K.ptx.mov.b32(random_words[1], c1)
    K.ptx.mov.b32(random_words[2], c2)
    K.ptx.mov.b32(random_words[3], c3)


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
        pair0 = K.local_scalar("uint32")
        pair1 = K.local_scalar("uint32")
        K.ptx.cvt.rs.f16x2.f32(pair0, values[wr, 1], values[wr, 0], random_words[0])
        K.ptx.cvt.rs.f16x2.f32(pair1, values[wr, 3], values[wr, 2], random_words[1])
        if HAS_INTERMEDIATE_STATES:
            K.ptx.st.global_.v2.b32(intermediate_states.ptr_to([intermediate_base]), pair0, pair1)
            with K.If(write_final != 0), K.Then():
                K.ptx.st.global_.v2.b32(state.ptr_to([final_base]), pair0, pair1)
        else:
            with K.If(write_final != 0), K.Then():
                K.ptx.st.global_.v2.b32(state.ptr_to([final_base]), pair0, pair1)
    elif STATE_DTYPE == "float32":
        words = K.alloc_local((4,), "uint32")
        with K.unroll(4) as k:
            K.ptx.mov.b32(words[k], K.reinterpret("uint32", values[wr, k]))
        if HAS_INTERMEDIATE_STATES:
            with K.If(write_final != 0):
                with K.Then():
                    lo = K.local_scalar("uint64")
                    hi = K.local_scalar("uint64")
                    K.ptx.mov.b64(lo, words[0], words[1])
                    K.ptx.mov.b64(hi, words[2], words[3])
                    K.ptx.st.global_.v2.b64(intermediate_states.ptr_to([intermediate_base]), lo, hi)
                    K.ptx.st.global_.v2.b64(state.ptr_to([final_base]), lo, hi)
                with K.Else():
                    K.ptx.st.global_.v4.b32(
                        intermediate_states.ptr_to([intermediate_base]),
                        words[0],
                        words[1],
                        words[2],
                        words[3],
                    )
        else:
            with K.If(write_final != 0), K.Then():
                K.ptx.st.global_.v4.b32(
                    state.ptr_to([final_base]), words[0], words[1], words[2], words[3]
                )
    else:
        bits = K.alloc_local((4,), "uint16")
        with K.unroll(STATE_VALUES_PER_THREAD) as k:
            if STATE_DTYPE == "bfloat16":
                K.ptx.cvt.rn.bf16.f32(bits[k], values[wr, k])
            else:
                K.ptx.cvt.rn.f16.f32(bits[k], values[wr, k])
        if DSTATE == 64:
            if HAS_INTERMEDIATE_STATES:
                with K.If(write_final != 0):
                    with K.Then():
                        word = K.local_scalar("uint32")
                        K.ptx.mov.b32(word, bits[0], bits[1])
                        K.ptx.st.global_.b32(intermediate_states.ptr_to([intermediate_base]), word)
                        K.ptx.st.global_.b32(state.ptr_to([final_base]), word)
                    with K.Else():
                        K.ptx.st.global_.v2.b16(
                            intermediate_states.ptr_to([intermediate_base]), bits[0], bits[1]
                        )
            else:
                with K.If(write_final != 0), K.Then():
                    K.ptx.st.global_.v2.b16(state.ptr_to([final_base]), bits[0], bits[1])
        elif DSTATE == 96:
            if HAS_INTERMEDIATE_STATES:
                with K.unroll(3) as k:
                    K.ptx.st.global_.b16(
                        intermediate_states.ptr_to([intermediate_base + k]), bits[k]
                    )
            with K.If(write_final != 0), K.Then():
                with K.unroll(3) as k:
                    K.ptx.st.global_.b16(state.ptr_to([final_base + k]), bits[k])
        elif HAS_INTERMEDIATE_STATES:
            with K.If(write_final != 0):
                with K.Then():
                    word0 = K.local_scalar("uint32")
                    word1 = K.local_scalar("uint32")
                    K.ptx.mov.b32(word1, bits[2], bits[3])
                    K.ptx.mov.b32(word0, bits[0], bits[1])
                    K.ptx.st.global_.v2.b32(
                        intermediate_states.ptr_to([intermediate_base]), word0, word1
                    )
                    K.ptx.st.global_.v2.b32(state.ptr_to([final_base]), word0, word1)
                with K.Else():
                    K.ptx.st.global_.v4.b16(
                        intermediate_states.ptr_to([intermediate_base]),
                        bits[0],
                        bits[1],
                        bits[2],
                        bits[3],
                    )
        else:
            with K.If(write_final != 0), K.Then():
                K.ptx.st.global_.v4.b16(
                    state.ptr_to([final_base]), bits[0], bits[1], bits[2], bits[3]
                )


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
    spec = _specialization(kwargs)
    NHEADS = spec["NHEADS"]
    DIM = spec["DIM"]
    DSTATE = spec["DSTATE"]
    NTOKENS = spec["NTOKENS"]
    HEADS_PER_GROUP = spec["HEADS_PER_GROUP"]
    NUM_PASSES = spec["NUM_PASSES"]
    STATE_BYTES = spec["STATE_BYTES"]
    STATE_STAGE_VALUES = DIM * DSTATE
    STATE_VALUES_PER_THREAD = spec["STATE_VALUES_PER_THREAD"]
    HAS_INTERMEDIATE_STATES = spec["HAS_INTERMEDIATE_STATES"]
    HAS_Z = spec["HAS_Z"]
    HAS_D = spec["HAS_D"]
    HAS_DT_BIAS = spec["HAS_DT_BIAS"]
    PHILOX_ROUNDS = spec["PHILOX_ROUNDS"]
    STATE_DTYPE = spec["STATE_DTYPE"]
    WEIGHT_DTYPE = spec["WEIGHT_DTYPE"]
    INDEX_DTYPE = spec["INDEX_DTYPE"]

    @K.kernel(
        warps=16, arch="sm_100a", min_blocks_per_sm=2, grid=(spec["BATCH"], spec["NUM_HEAD_CHUNKS"])
    )
    def selective_state_update_mtp_vertical(
        tensor_state: K.TensorMap,
        tensor_b: K.TensorMap,
        tensor_c: K.TensorMap,
        tensor_x: K.TensorMap,
        state: K.gptr[spec["STATE_DTYPE"]],
        state_scale: K.gptr[K.f32],
        x: K.gptr[K.bf16],
        dt: K.gptr[spec["WEIGHT_DTYPE"]],
        matrix_a: K.gptr[K.f32],
        matrix_b: K.gptr[K.bf16],
        matrix_c: K.gptr[K.bf16],
        d_weight: K.gptr[spec["WEIGHT_DTYPE"]],
        z: K.gptr[K.bf16],
        dt_bias: K.gptr[spec["WEIGHT_DTYPE"]],
        state_indices: K.gptr[spec["INDEX_DTYPE"]],
        dst_indices: K.gptr[spec["INDEX_DTYPE"]],
        intermediate_states: K.gptr[spec["STATE_DTYPE"]],
        intermediate_indices: K.gptr[spec["INDEX_DTYPE"]],
        intermediate_scales: K.gptr[K.f32],
        cu_seqlens: K.gptr[str(kwargs["cu_seqlens_dtype"])],
        num_accepted_tokens: K.gptr[str(kwargs["accepted_dtype"])],
        rand_seed: K.gptr[K.i64],
        output: K.gptr[K.bf16],
        state_stride_batch: K.i64,
        state_scale_stride_batch: K.i64,
        x_stride_batch: K.i64,
        x_stride_mtp: K.i64,
        dt_stride_batch: K.i64,
        dt_stride_mtp: K.i64,
        b_stride_batch: K.i64,
        b_stride_mtp: K.i64,
        c_stride_batch: K.i64,
        c_stride_mtp: K.i64,
        z_stride_batch: K.i64,
        z_stride_mtp: K.i64,
        out_stride_batch: K.i64,
        out_stride_mtp: K.i64,
        state_indices_stride_batch: K.i64,
        state_indices_stride_t: K.i64,
        dst_indices_stride_batch: K.i64,
        dst_indices_stride_t: K.i64,
        cache_steps: K.i32,
        nheads_runtime: K.i32,
        ngroups_runtime: K.i32,
        dt_softplus: K.i32,
        update_state: K.i32,
        pad_slot_id: K.i32,
    ):
        batch_i, head_chunk = K.cta_id()
        head_base = head_chunk * 3
        if spec["HAS_STATE_INDICES"]:
            if spec["INDEX_DTYPE"] == "int32":
                _t1 = K.local_scalar("int32")
                K.ptx.ld.global_.nc.b32(_t1, state_indices.ptr_to([batch_i]))
                state_batch = K.cast(_t1, "int64")
            else:
                _t2 = K.local_scalar("int64")
                K.ptx.ld.global_.nc.b64(_t2, state_indices.ptr_to([batch_i]))
                state_batch = _t2
        else:
            state_batch = K.cast(batch_i, "int64")

        smem = K.smem_pool()
        s_b = smem.alloc((3 * spec["NTOKENS"] * spec["DSTATE"],), K.bf16, align=128)
        s_c = smem.alloc((3 * spec["NTOKENS"] * spec["DSTATE"],), K.bf16, align=128)
        s_dt = smem.alloc((3 * spec["NTOKENS"],), K.f32, align=128)
        s_state = smem.alloc((3 * spec["DIM"] * spec["DSTATE"],), spec["STATE_DTYPE"], align=128)
        s_x = smem.alloc((3 * spec["NTOKENS"] * spec["DIM"],), K.bf16, align=128)
        s_out = smem.alloc((3 * spec["NTOKENS"] * spec["DIM"],), K.f32, align=128)
        bar_bc = K.MBarrier(smem, 3)
        bar_empty = K.MBarrier(smem, 3)
        bar_full = K.MBarrier(smem, 3)
        bar_out = K.MBarrier(smem, 3)
        bar_done = K.MBarrier(smem, 3)
        bar_bc.init(160)
        bar_empty.init(160)
        bar_full.init(129)
        bar_out.init(160)
        bar_done.init(160)
        K.cuda.cta_sync()

        roles = K.specialize()
        update = roles.role("update", warps=list(range(12)))
        load = roles.role("load", warps=[12, 13, 14])
        epilogue = roles.role("epilogue", warps=[15])

        s_state_u16 = s_state.view("uint16")
        s_state_u32 = s_state.view("uint32")
        bar_bc_buf = bar_bc.buf
        bar_empty_buf = bar_empty.buf
        bar_full_buf = bar_full.buf
        bar_out_buf = bar_out.buf
        bar_done_buf = bar_done.buf
        intermediate_state_stride_batch = K.int64(NTOKENS * NHEADS * DIM * DSTATE)

        def update_head(group, head, IS_PAD: K.constexpr):
            lane: K.int32 = K.tid_in_role() & 31
            compute_warp: K.int32 = K.warp_id_in_role() % 4
            random_seed = K.local_scalar("int64", init=0)
            if PHILOX_ROUNDS > 0 and not IS_PAD:
                K.ptx.ld.global_.s64(random_seed, rand_seed.ptr_to([0]))
            icache_idx = K.local_scalar("int64", init=state_batch)
            if HAS_INTERMEDIATE_STATES and not IS_PAD:
                K.assign(
                    icache_idx, _global_load_index_s64(intermediate_indices, batch_i, INDEX_DTYPE)
                )

            gload_0 = K.local_scalar("uint32")
            K.ptx.ld.global_.b32(gload_0, matrix_a.ptr_to([head]))
            a_value: K.float32 = K.reinterpret("float32", gload_0)
            d_value = K.local_scalar("float32", init=0.0)
            if HAS_D:
                K.assign(d_value, _load_weight(d_weight, head, WEIGHT_DTYPE))
            bias_value = K.local_scalar("float32", init=0.0)
            if HAS_DT_BIAS:
                K.assign(bias_value, _load_weight(dt_bias, head, WEIGHT_DTYPE))

            K.ptx.mbarrier.arrive.shared__cta.b64(bar_empty_buf.ptr_to([group]), K.uint32(1))
            _mbarrier_arrive_wait(bar_bc_buf.ptr_to([group]))

            with K.serial((NTOKENS + 3) // 4) as step_iter:
                dt_step: K.int32 = compute_warp + step_iter * 4
                with K.If(K.And(dt_step < NTOKENS, lane == 0)), K.Then():
                    dt_value = K.local_scalar("float32")
                    K.assign(
                        dt_value,
                        _load_weight(
                            dt,
                            K.cast(batch_i, "int64") * dt_stride_batch
                            + K.cast(dt_step, "int64") * dt_stride_mtp
                            + head,
                            WEIGHT_DTYPE,
                        ),
                    )
                    if HAS_DT_BIAS:
                        K.ptx["add.ftz.f32"](dt_value, dt_value, bias_value)
                    with K.If(K.And(dt_softplus != 0, dt_value <= K.float32(20.0))), K.Then():
                        mul_0 = K.local_scalar("float32")
                        K.ptx["mul.ftz.f32"](mul_0, dt_value, K.float32(_LOG2_E))
                        exp_arg: K.float32 = mul_0
                        exp2_0 = K.local_scalar("float32")
                        K.ptx["ex2.approx.ftz.f32"](exp2_0, exp_arg)
                        exp_value: K.float32 = exp2_0
                        add_0 = K.local_scalar("float32")
                        K.ptx["add.ftz.f32"](add_0, K.float32(1.0), exp_value)
                        log2_0 = K.local_scalar("float32")
                        K.ptx["lg2.approx.ftz.f32"](log2_0, add_0)
                        log_value: K.float32 = log2_0
                        K.ptx["mul.ftz.f32"](dt_value, log_value, K.float32(_LN_2))
                    K.ptx.st.shared.b32(
                        s_dt.ptr_to([group * NTOKENS + dt_step]), K.reinterpret("uint32", dt_value)
                    )

            _mbarrier_arrive_wait(bar_full_buf.ptr_to([group]))
            lane_indicator: K.float32 = K.if_then_else(lane == 0, K.float32(1.0), K.float32(0.0))
            seed_u64: K.uint64 = K.reinterpret("uint64", random_seed)
            seed_lo: K.uint32 = K.cast(seed_u64, "uint32")
            seed_hi: K.uint32 = K.cast(K.shift_right(seed_u64, K.uint64(32)), "uint32")
            state_head_i32: K.int32 = K.cast(
                state_batch * state_stride_batch + K.cast(head * DIM * DSTATE, "int64"), "int32"
            )

            pass_idx = K.local_scalar("int32", init=0)
            with K.While(pass_idx < NUM_PASSES):
                row_offset: K.int32 = compute_warp * (DIM // 4) + pass_idx * 4
                r_state = K.alloc_local((4, STATE_VALUES_PER_THREAD), "float32")
                with K.unroll(4) as wr:
                    dd: K.int32 = row_offset + wr
                    if IS_PAD:
                        with K.unroll(STATE_VALUES_PER_THREAD) as ii:
                            K.ptx.mov.b32(r_state[wr, ii], K.float32(0.0))
                    elif STATE_DTYPE == "float32":
                        state_words = K.alloc_local((4,), "uint32")
                        K.ptx.ld.shared.v4.b32(
                            state_words[0],
                            state_words[1],
                            state_words[2],
                            state_words[3],
                            s_state_u32.ptr_to(
                                [
                                    (
                                        group * DIM * DSTATE
                                        + dd * DSTATE
                                        + lane * STATE_VALUES_PER_THREAD
                                    )
                                ]
                            ),
                        )
                        with K.unroll(4) as ii:
                            K.ptx.mov.b32(
                                r_state[wr, ii], K.reinterpret("float32", state_words[ii])
                            )
                    else:
                        state_bits = K.alloc_local((4,), "uint16")
                        state_index: K.int32 = (
                            group * DIM * DSTATE + dd * DSTATE + lane * STATE_VALUES_PER_THREAD
                        )
                        if DSTATE == 64:
                            K.ptx.ld.shared.v2.b16(
                                state_bits[0], state_bits[1], s_state_u16.ptr_to([state_index])
                            )
                        elif DSTATE == 96:
                            with K.unroll(3) as ii:
                                sload_0 = K.local_scalar("uint16")
                                K.ptx.ld.shared.b16(sload_0, s_state_u16.ptr_to([state_index + ii]))
                                K.ptx.mov.b16(state_bits[ii], sload_0)
                        else:
                            K.ptx.ld.shared.v4.b16(
                                state_bits[0],
                                state_bits[1],
                                state_bits[2],
                                state_bits[3],
                                s_state_u16.ptr_to([state_index]),
                            )
                        with K.unroll(STATE_VALUES_PER_THREAD) as ii:
                            if STATE_DTYPE == "bfloat16":
                                bf16_f32_0 = K.local_scalar("float32")
                                K.ptx.cvt.f32.bf16(bf16_f32_0, K.cast(state_bits[ii], "uint16"))
                                K.ptx.mov.b32(r_state[wr, ii], bf16_f32_0)
                            else:
                                f16_f32_0 = K.local_scalar("float32")
                                K.ptx.cvt.f32.f16(f16_f32_0, K.cast(state_bits[ii], "uint16"))
                                K.ptx.mov.b32(r_state[wr, ii], f16_f32_0)

                row_random = K.alloc_local((4, 4), "uint32")
                if PHILOX_ROUNDS > 0 and not IS_PAD:
                    with K.unroll(4) as wr:
                        dd: K.int32 = row_offset + wr
                        add_s32_0 = K.local_scalar("int32")
                        K.ptx["add.s32"](
                            add_s32_0, state_head_i32, dd * DSTATE + lane * STATE_VALUES_PER_THREAD
                        )
                        random_counter: K.int32 = add_s32_0
                        random_words = K.alloc_local((4,), "uint32")
                        _philox4x32(
                            random_words,
                            seed_lo,
                            seed_hi,
                            random_counter,
                            PHILOX_ROUNDS=PHILOX_ROUNDS,
                        )
                        with K.unroll(4) as ri:
                            K.ptx.mov.b32(row_random[wr, ri], random_words[ri])

                step = K.local_scalar("int32", init=0)
                with K.While(step < NTOKENS):
                    sload_1 = K.local_scalar("uint32")
                    K.ptx.ld.shared.b32(sload_1, s_dt.ptr_to([group * NTOKENS + step]))
                    shared_dt: K.float32 = K.reinterpret("float32", sload_1)
                    mul_1 = K.local_scalar("float32")
                    K.ptx["mul.ftz.f32"](mul_1, a_value, shared_dt)
                    mul_2 = K.local_scalar("float32")
                    K.ptx["mul.ftz.f32"](mul_2, mul_1, K.float32(_LOG2_E))
                    exp2_1 = K.local_scalar("float32")
                    K.ptx["ex2.approx.ftz.f32"](exp2_1, mul_2)
                    da_value: K.float32 = exp2_1
                    b_values = K.alloc_local((STATE_VALUES_PER_THREAD,), "float32")
                    c_values = K.alloc_local((STATE_VALUES_PER_THREAD,), "float32")
                    b_bits = K.alloc_local((4,), "uint16")
                    c_bits = K.alloc_local((4,), "uint16")
                    bc_col: K.int32 = lane * STATE_VALUES_PER_THREAD
                    b_index: K.int32 = group * NTOKENS * DSTATE + step * DSTATE + bc_col
                    c_index: K.int32 = group * NTOKENS * DSTATE + step * DSTATE + bc_col
                    if DSTATE == 64:
                        K.ptx.ld.shared.v2.b16(b_bits[0], b_bits[1], s_b.ptr_to([b_index]))
                        K.ptx.ld.shared.v2.b16(c_bits[0], c_bits[1], s_c.ptr_to([c_index]))
                    elif DSTATE == 96:
                        with K.unroll(3) as ii:
                            K.ptx.ld.shared.b16(b_bits[ii], s_b.ptr_to([b_index + ii]))
                            K.ptx.ld.shared.b16(c_bits[ii], s_c.ptr_to([c_index + ii]))
                    else:
                        K.ptx.ld.shared.v4.b16(
                            b_bits[0], b_bits[1], b_bits[2], b_bits[3], s_b.ptr_to([b_index])
                        )
                        K.ptx.ld.shared.v4.b16(
                            c_bits[0], c_bits[1], c_bits[2], c_bits[3], s_c.ptr_to([c_index])
                        )
                    with K.unroll(STATE_VALUES_PER_THREAD) as ii:
                        K.ptx.cvt.f32.bf16(b_values[ii], K.cast(b_bits[ii], "uint16"))
                        K.ptx.cvt.f32.bf16(c_values[ii], K.cast(c_bits[ii], "uint16"))

                    with K.unroll(4) as wr:
                        dd: K.int32 = row_offset + wr
                        sload_2 = K.local_scalar("uint16")
                        K.ptx.ld.shared.b16(
                            sload_2, s_x.ptr_to([group * NTOKENS * DIM + step * DIM + dd])
                        )
                        bf16_f32_1 = K.local_scalar("float32")
                        K.ptx.cvt.f32.bf16(bf16_f32_1, K.cast(sload_2, "uint16"))
                        x_value: K.float32 = bf16_f32_1
                        mul_3 = K.local_scalar("float32")
                        K.ptx["mul.ftz.f32"](mul_3, d_value, x_value)
                        d_times_x: K.float32 = mul_3
                        out_value = K.local_scalar("float32", init=0.0)
                        with K.unroll(STATE_VALUES_PER_THREAD) as ii:
                            mul_4 = K.local_scalar("float32")
                            K.ptx["mul.ftz.f32"](mul_4, b_values[ii], shared_dt)
                            db_value: K.float32 = mul_4
                            mul_5 = K.local_scalar("float32")
                            K.ptx["mul.ftz.f32"](mul_5, db_value, x_value)
                            db_x: K.float32 = mul_5
                            fma_0 = K.local_scalar("float32")
                            K.ptx["fma.rn.ftz.f32"](fma_0, r_state[wr, ii], da_value, db_x)
                            new_state: K.float32 = fma_0
                            K.ptx.mov.b32(r_state[wr, ii], new_state)
                            with K.If(ii == 0):
                                with K.Then():
                                    mul_6 = K.local_scalar("float32")
                                    K.ptx["mul.ftz.f32"](mul_6, new_state, c_values[ii])
                                    state_c: K.float32 = mul_6
                                    K.ptx["fma.rn.ftz.f32"](
                                        out_value, d_times_x, lane_indicator, state_c
                                    )
                                with K.Else():
                                    K.ptx["fma.rn.ftz.f32"](
                                        out_value, new_state, c_values[ii], out_value
                                    )
                        with K.unroll(5) as delta_i:
                            delta: K.int32 = K.shift_right(K.int32(16), delta_i)
                            K.ptx["add.ftz.f32"](
                                out_value, out_value, _shfl_down_f32(out_value, delta)
                            )
                        with K.If(lane == 0), K.Then():
                            K.ptx.st.shared.b32(
                                s_out.ptr_to([group * NTOKENS * DIM + step * DIM + dd]),
                                K.reinterpret("uint32", out_value),
                            )

                    if not IS_PAD:
                        write_final: K.int32 = K.if_then_else(
                            step == NTOKENS - 1, K.if_then_else(update_state != 0, 1, 0), 0
                        )
                        if HAS_INTERMEDIATE_STATES:
                            with K.If(write_final != 0):
                                with K.Then():
                                    with K.unroll(4) as wr:
                                        dd: K.int32 = row_offset + wr
                                        intermediate_base: K.int64 = (
                                            icache_idx * intermediate_state_stride_batch
                                            + K.cast(step * NHEADS * DIM * DSTATE, "int64")
                                            + head * DIM * DSTATE
                                            + dd * DSTATE
                                            + lane * STATE_VALUES_PER_THREAD
                                        )
                                        final_base: K.int64 = (
                                            state_batch * state_stride_batch
                                            + head * DIM * DSTATE
                                            + dd * DSTATE
                                            + lane * STATE_VALUES_PER_THREAD
                                        )
                                        random_words = K.alloc_local((4,), "uint32")
                                        with K.unroll(4) as ri:
                                            K.ptx.mov.b32(random_words[ri], row_random[wr, ri])
                                        _store_state_row(
                                            r_state,
                                            wr,
                                            random_words,
                                            state,
                                            intermediate_states,
                                            intermediate_base,
                                            final_base,
                                            K.int32(1),
                                            DSTATE=DSTATE,
                                            STATE_DTYPE=STATE_DTYPE,
                                            STATE_VALUES_PER_THREAD=STATE_VALUES_PER_THREAD,
                                            HAS_INTERMEDIATE_STATES=True,
                                            PHILOX_ROUNDS=PHILOX_ROUNDS,
                                        )
                                with K.Else():
                                    with K.unroll(4) as wr:
                                        dd: K.int32 = row_offset + wr
                                        intermediate_base: K.int64 = (
                                            icache_idx * intermediate_state_stride_batch
                                            + K.cast(step * NHEADS * DIM * DSTATE, "int64")
                                            + head * DIM * DSTATE
                                            + dd * DSTATE
                                            + lane * STATE_VALUES_PER_THREAD
                                        )
                                        random_words = K.alloc_local((4,), "uint32")
                                        with K.unroll(4) as ri:
                                            K.ptx.mov.b32(random_words[ri], row_random[wr, ri])
                                        _store_state_row(
                                            r_state,
                                            wr,
                                            random_words,
                                            state,
                                            intermediate_states,
                                            intermediate_base,
                                            K.int64(0),
                                            K.int32(0),
                                            DSTATE=DSTATE,
                                            STATE_DTYPE=STATE_DTYPE,
                                            STATE_VALUES_PER_THREAD=STATE_VALUES_PER_THREAD,
                                            HAS_INTERMEDIATE_STATES=True,
                                            PHILOX_ROUNDS=PHILOX_ROUNDS,
                                        )
                        else:
                            with K.If(write_final != 0), K.Then():
                                with K.unroll(4) as wr:
                                    dd: K.int32 = row_offset + wr
                                    final_base: K.int64 = (
                                        state_batch * state_stride_batch
                                        + head * DIM * DSTATE
                                        + dd * DSTATE
                                        + lane * STATE_VALUES_PER_THREAD
                                    )
                                    random_words = K.alloc_local((4,), "uint32")
                                    with K.unroll(4) as ri:
                                        K.ptx.mov.b32(random_words[ri], row_random[wr, ri])
                                    _store_state_row(
                                        r_state,
                                        wr,
                                        random_words,
                                        state,
                                        intermediate_states,
                                        K.int64(0),
                                        final_base,
                                        K.int32(1),
                                        DSTATE=DSTATE,
                                        STATE_DTYPE=STATE_DTYPE,
                                        STATE_VALUES_PER_THREAD=STATE_VALUES_PER_THREAD,
                                        HAS_INTERMEDIATE_STATES=False,
                                        PHILOX_ROUNDS=PHILOX_ROUNDS,
                                    )
                    K.assign(step, step + 1)
                K.assign(pass_idx, pass_idx + 1)

            K.ptx.mbarrier.arrive.shared__cta.b64(bar_out_buf.ptr_to([group]), K.uint32(1))
            _mbarrier_arrive_wait(bar_done_buf.ptr_to([group]))

        def run_update(is_pad):
            group = K.warp_id_in_role() // 4
            head = head_base + group
            with K.If(head < NHEADS), K.Then():
                update_head(group, head, is_pad)

        def load_head(group, head, IS_PAD: K.constexpr):
            lane: K.int32 = K.tid_in_role() & 31
            kv_group: K.int32 = head // HEADS_PER_GROUP
            with K.If(lane == 0), K.Then():
                _tma_g2s_4d(
                    s_b.ptr_to([group * NTOKENS * DSTATE]),
                    tensor_b,
                    0,
                    kv_group,
                    0,
                    batch_i,
                    bar_bc_buf.ptr_to([group]),
                )
                _tma_g2s_4d(
                    s_c.ptr_to([group * NTOKENS * DSTATE]),
                    tensor_c,
                    0,
                    kv_group,
                    0,
                    batch_i,
                    bar_bc_buf.ptr_to([group]),
                )
                K.ptx.mbarrier.expect_tx.relaxed.cta.shared__cta.b64(
                    bar_bc_buf.ptr_to([group]), K.uint32(2 * NTOKENS * DSTATE * 2)
                )
                K.ptx.mbarrier.arrive.release.cta.shared__cta.b64(
                    bar_bc_buf.ptr_to([group]), K.uint32(32)
                )

            _mbarrier_arrive_wait(bar_empty_buf.ptr_to([group]))
            with K.If(lane == 0), K.Then():
                if not IS_PAD:
                    _tma_g2s_4d(
                        s_state.ptr_to([group * STATE_STAGE_VALUES]),
                        tensor_state,
                        0,
                        0,
                        head,
                        state_batch,
                        bar_full_buf.ptr_to([group]),
                    )
                _tma_g2s_4d(
                    s_x.ptr_to([group * NTOKENS * DIM]),
                    tensor_x,
                    0,
                    head,
                    0,
                    batch_i,
                    bar_full_buf.ptr_to([group]),
                )
                if IS_PAD:
                    K.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
                        bar_full_buf.ptr_to([group]), K.uint32(NTOKENS * DIM * 2)
                    )
                else:
                    K.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
                        bar_full_buf.ptr_to([group]),
                        K.uint32(DIM * DSTATE * STATE_BYTES + NTOKENS * DIM * 2),
                    )

        def run_load(is_pad):
            group = K.warp_id_in_role()
            head = head_base + group
            with K.If(head < NHEADS), K.Then():
                load_head(group, head, is_pad)

        def run_epilogue():
            lane: K.int32 = K.tid_in_role()
            with K.serial(3) as group:
                head: K.int32 = head_base + group
                with K.If(head < NHEADS), K.Then():
                    _mbarrier_arrive_wait(bar_out_buf.ptr_to([group]))
                    with K.unroll(NTOKENS) as step:
                        out_base: K.int64 = (
                            K.cast(batch_i, "int64") * out_stride_batch
                            + K.cast(step, "int64") * out_stride_mtp
                            + head * DIM
                        )
                        z_base: K.int64 = (
                            K.cast(batch_i, "int64") * z_stride_batch
                            + K.cast(step, "int64") * z_stride_mtp
                            + head * DIM
                        )
                        out_words = K.alloc_local((4,), "uint32")
                        z_bits = K.alloc_local((4,), "uint16")
                        output_bits = K.alloc_local((4,), "uint16")
                        if DIM == 64:
                            d: K.int32 = lane * 2
                            K.ptx.ld.shared.v2.b32(
                                out_words[0],
                                out_words[1],
                                s_out.ptr_to([group * NTOKENS * DIM + step * DIM + d]),
                            )
                            if HAS_Z:
                                K.ptx.ld.global_.v2.b16(
                                    z_bits[0], z_bits[1], z.ptr_to([(z_base + d)])
                                )
                            with K.unroll(2) as k:
                                out_value = K.local_scalar(
                                    "float32", init=K.reinterpret("float32", out_words[k])
                                )
                                if HAS_Z:
                                    bf16_f32_2 = K.local_scalar("float32")
                                    K.ptx.cvt.f32.bf16(bf16_f32_2, K.cast(z_bits[k], "uint16"))
                                    z_value: K.float32 = bf16_f32_2
                                    sub_0 = K.local_scalar("float32")
                                    K.ptx["sub.ftz.f32"](sub_0, K.float32(0.0), z_value)
                                    mul_7 = K.local_scalar("float32")
                                    K.ptx["mul.ftz.f32"](mul_7, sub_0, K.float32(_LOG2_E))
                                    exp2_2 = K.local_scalar("float32")
                                    K.ptx["ex2.approx.ftz.f32"](exp2_2, mul_7)
                                    exp_neg_z: K.float32 = exp2_2
                                    add_1 = K.local_scalar("float32")
                                    K.ptx["add.ftz.f32"](add_1, K.float32(1.0), exp_neg_z)
                                    div_0 = K.local_scalar("float32")
                                    K.ptx["div.approx.ftz.f32"](div_0, K.float32(1.0), add_1)
                                    sigmoid_z: K.float32 = div_0
                                    mul_8 = K.local_scalar("float32")
                                    K.ptx["mul.ftz.f32"](mul_8, z_value, sigmoid_z)
                                    K.ptx["mul.ftz.f32"](out_value, out_value, mul_8)
                                K.ptx.cvt.rn.bf16.f32(output_bits[k], out_value)
                            K.ptx.st.global_.v2.b16(
                                output.ptr_to([out_base + d]), output_bits[0], output_bits[1]
                            )
                        else:
                            d: K.int32 = lane * 4
                            K.ptx.ld.shared.v4.b32(
                                out_words[0],
                                out_words[1],
                                out_words[2],
                                out_words[3],
                                s_out.ptr_to([(group * NTOKENS * DIM + step * DIM + d)]),
                            )
                            if HAS_Z:
                                K.ptx.ld.global_.v4.b16(
                                    z_bits[0],
                                    z_bits[1],
                                    z_bits[2],
                                    z_bits[3],
                                    z.ptr_to([(z_base + d)]),
                                )
                            with K.unroll(4) as k:
                                out_value = K.local_scalar(
                                    "float32", init=K.reinterpret("float32", out_words[k])
                                )
                                if HAS_Z:
                                    bf16_f32_3 = K.local_scalar("float32")
                                    K.ptx.cvt.f32.bf16(bf16_f32_3, K.cast(z_bits[k], "uint16"))
                                    z_value: K.float32 = bf16_f32_3
                                    sub_1 = K.local_scalar("float32")
                                    K.ptx["sub.ftz.f32"](sub_1, K.float32(0.0), z_value)
                                    mul_9 = K.local_scalar("float32")
                                    K.ptx["mul.ftz.f32"](mul_9, sub_1, K.float32(_LOG2_E))
                                    exp2_3 = K.local_scalar("float32")
                                    K.ptx["ex2.approx.ftz.f32"](exp2_3, mul_9)
                                    exp_neg_z: K.float32 = exp2_3
                                    add_2 = K.local_scalar("float32")
                                    K.ptx["add.ftz.f32"](add_2, K.float32(1.0), exp_neg_z)
                                    div_1 = K.local_scalar("float32")
                                    K.ptx["div.approx.ftz.f32"](div_1, K.float32(1.0), add_2)
                                    sigmoid_z: K.float32 = div_1
                                    mul_10 = K.local_scalar("float32")
                                    K.ptx["mul.ftz.f32"](mul_10, z_value, sigmoid_z)
                                    K.ptx["mul.ftz.f32"](out_value, out_value, mul_10)
                                K.ptx.cvt.rn.bf16.f32(output_bits[k], out_value)
                            K.ptx.st.global_.v4.b16(
                                output.ptr_to([out_base + d]),
                                output_bits[0],
                                output_bits[1],
                                output_bits[2],
                                output_bits[3],
                            )
                    K.ptx.mbarrier.arrive.shared__cta.b64(bar_done_buf.ptr_to([group]), K.uint32(1))

        with update:
            with K.If(state_batch == K.cast(pad_slot_id, "int64")):
                with K.Then():
                    run_update(True)
                with K.Else():
                    run_update(False)
        with load:
            with K.If(state_batch == K.cast(pad_slot_id, "int64")):
                with K.Then():
                    run_load(True)
                with K.Else():
                    run_load(False)
        with epilogue:
            run_epilogue()

    return selective_state_update_mtp_vertical.func


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

    def source_builder():
        executable(*args)
        _run_reference(case)
        torch.cuda.synchronize()
        _simple._assert_case_close(case)
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
