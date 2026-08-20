# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's selective-state-update MTP horizontal kernel.

Upstream source: include/flashinfer/mamba/kernel_selective_state_update_mtp_horizontal.cuh.
"""

import ctypes
import functools
from typing import Any

import torch

import tirx_kernels.kern as K

from . import selective_state_update_mtp_simple as _simple
from . import selective_state_update_mtp_vertical as _vertical
from .selective_state_update_mtp_simple import _case

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
_global_load_s64 = _simple._global_load_s64
_global_load_index_s64 = _simple._global_load_index_s64
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


def _mbarrier_arrive(barrier):
    K.evaluate(K.ptx.mbarrier.arrive.shared__cta.b64(barrier))


def _mbarrier_arrive_expect_tx(barrier, num_bytes):
    K.evaluate(
        K.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(barrier, K.uint32(num_bytes))
    )


def _mbarrier_arrive_wait_parity(barrier, parity):
    _mbarrier_arrive(barrier)
    ready = K.alloc_local((1,), "uint32")
    K.assign(ready[0], K.uint32(0))
    with K.While(True):
        K.evaluate(
            K.ptx.mbarrier.try_wait.parity.shared__cta.b64(ready[0], barrier, K.uint32(parity))
        )
        with K.If(ready[0] != K.uint32(0)):
            with K.Then():
                K.Break()


def _tma_g2s_4d(dst, tensor_map, c0, c1, c2, c3, barrier):
    K.evaluate(
        K.ptx[_vertical._TMA_G2S_4D](
            dst,
            K.address_of(tensor_map),
            K.cast(c0, "int32"),
            K.cast(c1, "int32"),
            K.cast(c2, "int32"),
            K.cast(c3, "int32"),
            barrier,
        )
    )


def _shared_load_v2_b32(buffer, index, values):
    K.evaluate(K.ptx.ld.shared.v2.b32(values[0], values[1], buffer.ptr_to([index])))


def _shared_load_v4_b32(buffer, index, values):
    K.evaluate(
        K.ptx.ld.shared.v4.b32(values[0], values[1], values[2], values[3], buffer.ptr_to([index]))
    )


def _global_load_v2_b16(buffer, index, values):
    K.evaluate(K.ptx.ld.global_.v2.b16(values[0], values[1], buffer.ptr_to([index])))


def _global_load_v4_b16(buffer, index, values):
    K.evaluate(
        K.ptx.ld.global_.v4.b16(values[0], values[1], values[2], values[3], buffer.ptr_to([index]))
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
        words = K.alloc_local((4,), "uint32")
        with K.unroll(2) as pair:
            packed = values[pair_base + pair]
            K.ptx.mov.b32(words[pair * 2], K.reinterpret("uint32", K.cuda.float2_x(packed)))
            K.ptx.mov.b32(words[pair * 2 + 1], K.reinterpret("uint32", K.cuda.float2_y(packed)))
        K.evaluate(
            K.ptx.st.global_.v4.b32(
                destination.ptr_to([destination_base]), words[0], words[1], words[2], words[3]
            )
        )
    else:
        if PHILOX_ROUNDS > 0:
            packed_words = K.alloc_local((4,), "uint32")
            with K.unroll(4) as pair:
                packed = values[pair_base + pair]
                K.evaluate(
                    K.ptx.cvt.rs.f16x2.f32(
                        packed_words[pair],
                        K.cuda.float2_y(packed),
                        K.cuda.float2_x(packed),
                        random_words[random_base + pair // 2 * 4 + pair % 2],
                    )
                )
            K.evaluate(
                K.ptx.st.global_.v4.b32(
                    destination.ptr_to([destination_base]),
                    packed_words[0],
                    packed_words[1],
                    packed_words[2],
                    packed_words[3],
                )
            )
        else:
            bits = K.alloc_local((8,), "uint16")
            words = K.alloc_local((4,), "uint32")
            with K.unroll(PAIRS_PER_TILE_MEMBER) as pair:
                packed = values[pair_base + pair]
                if STATE_DTYPE == "bfloat16":
                    K.ptx.mov.b16(bits[pair * 2], _f32_to_bf16(K.cuda.float2_x(packed)))
                    K.ptx.mov.b16(bits[pair * 2 + 1], _f32_to_bf16(K.cuda.float2_y(packed)))
                else:
                    K.ptx.mov.b16(bits[pair * 2], _f32_to_f16(K.cuda.float2_x(packed)))
                    K.ptx.mov.b16(bits[pair * 2 + 1], _f32_to_f16(K.cuda.float2_y(packed)))
            with K.unroll(4) as word:
                K.evaluate(K.ptx.mov.b32(words[word], bits[word * 2], bits[word * 2 + 1]))
            K.evaluate(
                K.ptx.st.global_.v4.b32(
                    destination.ptr_to([destination_base]), words[0], words[1], words[2], words[3]
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
    NUM_TMA_LOADS = spec["NUM_TMA_LOADS"]
    DSTATE_PAD = spec["DSTATE_PAD"]
    STATE_BYTES = spec["STATE_BYTES"]
    STATE_STAGE_BYTES = spec["STATE_STAGE_BYTES"]
    STATE_STAGE_WORDS = STATE_STAGE_BYTES // 4
    ELEMS_PER_TILE_MEMBER = spec["ELEMS_PER_TILE_MEMBER"]
    PAIRS_PER_TILE_MEMBER = spec["PAIRS_PER_TILE_MEMBER"]
    ELEMS_PER_TILE = spec["ELEMS_PER_TILE"]
    NUM_TILES = spec["NUM_TILES"]
    HAS_INTERMEDIATE_STATES = spec["HAS_INTERMEDIATE_STATES"]
    HAS_Z = spec["HAS_Z"]
    HAS_D = spec["HAS_D"]
    HAS_DT_BIAS = spec["HAS_DT_BIAS"]
    UPDATE_STATE = spec["UPDATE_STATE"]
    PHILOX_ROUNDS = spec["PHILOX_ROUNDS"]
    STATE_DTYPE = spec["STATE_DTYPE"]
    WEIGHT_DTYPE = spec["WEIGHT_DTYPE"]
    INDEX_DTYPE = spec["INDEX_DTYPE"]

    @K.kernel(
        warps=5,
        arch="sm_100a",
        min_blocks_per_sm=7,
        grid=(spec["BATCH"], spec["NHEADS"]),
        thread_layout="lane_warp",
    )
    def selective_state_update_mtp_horizontal(
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
        batch_i, head = K.cta_id()
        if spec["HAS_STATE_INDICES"]:
            if spec["INDEX_DTYPE"] == "int32":
                state_batch = K.cast(_global_load_nc_s32(state_indices, batch_i), "int64")
            else:
                state_batch = _global_load_nc_s64(state_indices, batch_i)
        else:
            state_batch = K.cast(batch_i, "int64")

        smem = K.smem_pool()
        s_b = smem.alloc((spec["NTOKENS"] * spec["DSTATE_PAD"],), K.bf16, align=128)
        s_c = smem.alloc((spec["NTOKENS"] * spec["DSTATE_PAD"],), K.bf16, align=128)
        s_state_words = smem.alloc((2 * spec["STATE_STAGE_BYTES"] // 4,), K.u32, align=128)
        s_x = smem.alloc((spec["NTOKENS"] * spec["DIM"],), K.bf16, align=128)
        s_dt = smem.alloc((spec["NTOKENS"],), K.f32, align=4)
        s_out = smem.alloc((spec["NTOKENS"] * spec["DIM"],), K.f32, align=4)
        empty = K.MBarrier(smem, 2, leader=True)
        full = K.MBarrier(smem, 2, leader=True)
        out_ready = K.MBarrier(smem, 1, leader=True)
        with K.If(K.thread_id() == 0):
            with K.Then():
                empty.init(160)
                full.init(129)
                out_ready.init(128)
        K.cuda.cta_sync()

        roles = K.specialize()
        update = roles.role("update", warps=[0, 1, 2, 3])
        load = roles.role("load", warps=[4])

        s_b_words = s_b.view("uint32")
        s_c_words = s_c.view("uint32")
        empty_barriers = empty.buf
        full_barriers = full.buf
        out_ready_barrier = out_ready.buf

        def run_update(IS_PAD: K.constexpr):
            lane: K.int32 = K.lane_id()
            compute_warp: K.int32 = K.warp_id_in_role()
            random_seed = K.local_scalar("int64")
            K.assign(random_seed, 0)
            if PHILOX_ROUNDS > 0 and not IS_PAD:
                K.assign(random_seed, _global_load_s64(rand_seed, 0))

            icache_idx = K.local_scalar("int64")
            K.assign(icache_idx, state_batch)
            if HAS_INTERMEDIATE_STATES and not IS_PAD:
                K.assign(
                    icache_idx, _global_load_index_s64(intermediate_indices, batch_i, INDEX_DTYPE)
                )

            a_value: K.float32 = K.reinterpret("float32", _global_load_u32(matrix_a, head))
            d_value = K.local_scalar("float32")
            K.assign(d_value, 0.0)
            if HAS_D:
                K.assign(d_value, _load_weight(d_weight, head, WEIGHT_DTYPE))
            bias_value = K.local_scalar("float32")
            K.assign(bias_value, 0.0)
            if HAS_DT_BIAS:
                K.assign(bias_value, _load_weight(dt_bias, head, WEIGHT_DTYPE))

            _mbarrier_arrive(empty_barriers.ptr_to([0]))
            _mbarrier_arrive(empty_barriers.ptr_to([1]))

            flat_thread: K.int32 = compute_warp * 32 + lane
            with K.If(flat_thread < NTOKENS):
                with K.Then():
                    dt_value = K.local_scalar("float32")
                    K.assign(
                        dt_value,
                        _load_weight(
                            dt,
                            K.cast(batch_i, "int64") * dt_stride_batch
                            + K.cast(flat_thread, "int64") * dt_stride_mtp
                            + head,
                            WEIGHT_DTYPE,
                        ),
                    )
                    if HAS_DT_BIAS:
                        K.assign(dt_value, _add(dt_value, bias_value))
                    with K.If(K.And(dt_softplus != 0, dt_value <= K.float32(20.0))):
                        with K.Then():
                            exp_value: K.float32 = _exp2(_mul(dt_value, K.float32(_LOG2_E)))
                            log_value: K.float32 = _log2(_add(K.float32(1.0), exp_value))
                            K.assign(dt_value, _mul(log_value, K.float32(_LN_2)))
                    K.evaluate(
                        K.ptx.st.shared.b32(
                            s_dt.ptr_to([flat_thread]), K.reinterpret("uint32", dt_value)
                        )
                    )

            member: K.int32 = lane % 8
            row_group: K.int32 = lane // 8
            state_ptr_offset_i32: K.int32 = K.cast(
                state_batch * state_stride_batch + K.cast(head * DIM * DSTATE, "int64"), "int32"
            )

            state_pipe = K.PipelineState(2, phase=0)
            with K.unroll(NUM_TMA_LOADS) as tl:
                _mbarrier_arrive_wait_parity(
                    full_barriers.ptr_to([state_pipe.stage]), state_pipe.phase
                )

                with K.serial(2) as sp:
                    sram_row: K.int32 = sp * 16 + compute_warp * 4 + row_group
                    dd: K.int32 = tl * 32 + sram_row
                    state_values = K.alloc_local((NUM_TILES * PAIRS_PER_TILE_MEMBER,), "uint64")

                    with K.unroll(NUM_TILES) as tile:
                        member_col: K.int32 = tile * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER
                        pair_base: K.int32 = tile * PAIRS_PER_TILE_MEMBER
                        with K.If(K.And(K.Not(IS_PAD), member_col < DSTATE)):
                            with K.Then():
                                state_words = K.alloc_local((4,), "uint32")
                                state_word_index: K.int32 = (
                                    state_pipe.stage * STATE_STAGE_BYTES
                                    + (sram_row * DSTATE_PAD + member_col) * STATE_BYTES
                                ) // 4
                                _shared_load_v4_b32(s_state_words, state_word_index, state_words)
                                with K.unroll(PAIRS_PER_TILE_MEMBER) as pair:
                                    if STATE_DTYPE == "bfloat16":
                                        K.assign(
                                            state_values[pair_base + pair],
                                            _bf16_word_to_f32x2(state_words[pair]),
                                        )
                                    else:
                                        if STATE_DTYPE == "float16":
                                            K.assign(
                                                state_values[pair_base + pair],
                                                (
                                                    K.cuda.make_float2(
                                                        _f16_to_f32(
                                                            _extract_u16(state_words[pair], False)
                                                        ),
                                                        _f16_to_f32(
                                                            _extract_u16(state_words[pair], True)
                                                        ),
                                                    )
                                                ),
                                            )
                                        else:
                                            K.assign(
                                                state_values[pair_base + pair],
                                                (
                                                    K.cuda.make_float2(
                                                        K.reinterpret(
                                                            "float32", state_words[pair * 2]
                                                        ),
                                                        K.reinterpret(
                                                            "float32", state_words[pair * 2 + 1]
                                                        ),
                                                    )
                                                ),
                                            )
                            with K.Else():
                                with K.unroll(PAIRS_PER_TILE_MEMBER) as pair:
                                    K.assign(
                                        state_values[pair_base + pair],
                                        K.cuda.make_float2(K.float32(0.0), K.float32(0.0)),
                                    )

                    random_words = K.alloc_local((16,), "uint32")
                    if PHILOX_ROUNDS > 0 and not IS_PAD:
                        seed_lo: K.uint32 = K.cast(random_seed, "uint32")
                        seed_hi: K.uint32 = K.cast(
                            K.shift_right(K.reinterpret("uint64", random_seed), K.uint32(32)),
                            "uint32",
                        )
                        with K.unroll(4) as random_group:
                            random_tile: K.int32 = random_group // 2
                            random_e: K.int32 = (random_group % 2) * 4
                            random_col: K.int32 = (
                                random_tile * ELEMS_PER_TILE
                                + member * ELEMS_PER_TILE_MEMBER
                                + random_e
                            )
                            counter: K.int32 = state_ptr_offset_i32 + dd * DSTATE + random_col
                            group_words = K.alloc_local((4,), "uint32")
                            _philox4x32(
                                group_words, seed_lo, seed_hi, counter, PHILOX_ROUNDS=PHILOX_ROUNDS
                            )
                            with K.unroll(4) as random_word:
                                K.ptx.mov.b32(
                                    random_words[random_group * 4 + random_word],
                                    group_words[random_word],
                                )

                    bc_step_words = K.local_scalar("int32")
                    K.assign(bc_step_words, 0)
                    x_step = K.local_scalar("int32")
                    K.assign(x_step, 0)
                    dt_step = K.local_scalar("int32")
                    K.assign(dt_step, 0)
                    out_step = K.local_scalar("int32")
                    K.assign(out_step, 0)
                    intermediate_step_base = K.local_scalar("int64")
                    K.assign(
                        intermediate_step_base,
                        icache_idx * K.int64(NTOKENS * NHEADS * DIM * DSTATE)
                        + K.cast(head * DIM * DSTATE + dd * DSTATE, "int64"),
                    )

                    with K.serial(NTOKENS) as step:
                        dt_value = K.local_scalar("float32")
                        K.assign(
                            dt_value, K.reinterpret("float32", _shared_load_u32(s_dt, dt_step))
                        )
                        da_value: K.float32 = _exp2(
                            _mul(_mul(a_value, dt_value), K.float32(_LOG2_E))
                        )
                        x_value: K.float32 = _bf16_to_f32(_shared_load_u16(s_x, x_step + dd))
                        dtx_value: K.float32 = _mul(dt_value, x_value)
                        out_pair = K.local_scalar("uint64")
                        K.assign(out_pair, K.cuda.make_float2(K.float32(0.0), K.float32(0.0)))

                        with K.unroll(NUM_TILES) as tile:
                            member_col: K.int32 = (
                                tile * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER
                            )
                            with K.If(member_col < DSTATE):
                                with K.Then():
                                    b_words = K.alloc_local((4,), "uint32")
                                    c_words = K.alloc_local((4,), "uint32")
                                    b_word_index: K.int32 = bc_step_words + member_col // 2
                                    c_word_index: K.int32 = bc_step_words + member_col // 2
                                    if PAIRS_PER_TILE_MEMBER == 2:
                                        _shared_load_v2_b32(s_b_words, b_word_index, b_words)
                                        _shared_load_v2_b32(s_c_words, c_word_index, c_words)
                                    else:
                                        _shared_load_v4_b32(s_b_words, b_word_index, b_words)
                                        _shared_load_v4_b32(s_c_words, c_word_index, c_words)
                                    with K.unroll(PAIRS_PER_TILE_MEMBER) as pair:
                                        b_pair: K.uint64 = _bf16_word_to_f32x2(b_words[pair])
                                        c_pair: K.uint64 = _bf16_word_to_f32x2(c_words[pair])
                                        dbx_pair = K.local_scalar("uint64")
                                        K.ptx.mul.f32x2(
                                            dbx_pair,
                                            b_pair,
                                            K.cuda.make_float2(dtx_value, dtx_value),
                                        )
                                        pair_index: K.int32 = tile * PAIRS_PER_TILE_MEMBER + pair
                                        updated_state = K.local_scalar("uint64")
                                        K.ptx.fma.rn.f32x2(
                                            updated_state,
                                            K.cuda.make_float2(da_value, da_value),
                                            state_values[pair_index],
                                            dbx_pair,
                                        )
                                        K.ptx.mov.b64(state_values[pair_index], updated_state)
                                        K.ptx.fma.rn.f32x2(
                                            out_pair, updated_state, c_pair, out_pair
                                        )

                        out_value = K.local_scalar("float32")
                        K.assign(
                            out_value, _add(K.cuda.float2_x(out_pair), K.cuda.float2_y(out_pair))
                        )
                        with K.unroll(3) as delta_idx:
                            peer_out: K.float32 = K.cuda.__shfl_down_sync(
                                K.uint32(0xFFFFFFFF),
                                out_value,
                                K.shift_right(K.int32(4), delta_idx),
                                32,
                            )
                            K.assign(out_value, _add(out_value, peer_out))
                        with K.If(member == 0):
                            with K.Then():
                                K.evaluate(
                                    K.ptx.st.shared.b32(
                                        s_out.ptr_to([out_step + dd]),
                                        K.reinterpret("uint32", _fma(d_value, x_value, out_value)),
                                    )
                                )

                        K.assign(bc_step_words, bc_step_words + DSTATE_PAD // 2)
                        K.assign(x_step, x_step + DIM)
                        K.assign(dt_step, dt_step + 1)
                        K.assign(out_step, out_step + DIM)

                        if HAS_INTERMEDIATE_STATES and not IS_PAD:
                            with K.unroll(NUM_TILES) as tile:
                                member_col: K.int32 = (
                                    tile * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER
                                )
                                with K.If(member_col < DSTATE):
                                    with K.Then():
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
                            K.assign(
                                intermediate_step_base,
                                intermediate_step_base + K.int64(NHEADS * DIM * DSTATE),
                            )

                        with K.If(K.And(K.And(UPDATE_STATE, step == NTOKENS - 1), K.Not(IS_PAD))):
                            with K.Then():
                                final_base: K.int64 = state_batch * state_stride_batch + K.cast(
                                    head * DIM * DSTATE + dd * DSTATE, "int64"
                                )
                                with K.unroll(NUM_TILES) as tile:
                                    member_col: K.int32 = (
                                        tile * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER
                                    )
                                    with K.If(member_col < DSTATE):
                                        with K.Then():
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

                _mbarrier_arrive(empty_barriers.ptr_to([state_pipe.stage]))
                state_pipe.advance()

            _mbarrier_arrive_wait_parity(out_ready_barrier.ptr_to([0]), 0)
            with K.unroll((NTOKENS + 3) // 4) as episode:
                step: K.int32 = compute_warp + episode * 4
                with K.If(step < NTOKENS):
                    with K.Then():
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
                        if DIM == 64:
                            d: K.int32 = lane * 2
                            out_words = K.alloc_local((2,), "uint32")
                            if NTOKENS == 1:
                                K.ptx.mov.b32(out_words[0], _shared_load_u32(s_out, step * DIM + d))
                                K.ptx.mov.b32(
                                    out_words[1], _shared_load_u32(s_out, step * DIM + d + 1)
                                )
                            else:
                                _shared_load_v2_b32(s_out, step * DIM + d, out_words)
                            z_bits = K.alloc_local((2,), "uint16")
                            output_bits = K.alloc_local((2,), "uint16")
                            if HAS_Z:
                                _global_load_v2_b16(z, z_base + d, z_bits)
                            with K.unroll(2) as k:
                                out_value = K.local_scalar("float32")
                                K.assign(out_value, K.reinterpret("float32", out_words[k]))
                                if HAS_Z:
                                    z_value: K.float32 = _bf16_to_f32(z_bits[k])
                                    exp_neg: K.float32 = _exp2(
                                        _mul(_sub(K.float32(0.0), z_value), K.float32(_LOG2_E))
                                    )
                                    sigmoid: K.float32 = _div(
                                        K.float32(1.0), _add(K.float32(1.0), exp_neg)
                                    )
                                    K.assign(out_value, _mul(out_value, _mul(z_value, sigmoid)))
                                K.ptx.mov.b16(output_bits[k], _f32_to_bf16(out_value))
                            K.evaluate(
                                K.ptx.st.global_.v2.b16(
                                    output.ptr_to([out_base + d]), output_bits[0], output_bits[1]
                                )
                            )
                        else:
                            d: K.int32 = lane * 4
                            out_words = K.alloc_local((4,), "uint32")
                            _shared_load_v4_b32(s_out, step * DIM + d, out_words)
                            z_bits = K.alloc_local((4,), "uint16")
                            output_bits = K.alloc_local((4,), "uint16")
                            if HAS_Z:
                                _global_load_v4_b16(z, z_base + d, z_bits)
                            with K.unroll(4) as k:
                                out_value = K.local_scalar("float32")
                                K.assign(out_value, K.reinterpret("float32", out_words[k]))
                                if HAS_Z:
                                    z_value: K.float32 = _bf16_to_f32(z_bits[k])
                                    exp_neg: K.float32 = _exp2(
                                        _mul(_sub(K.float32(0.0), z_value), K.float32(_LOG2_E))
                                    )
                                    sigmoid: K.float32 = _div(
                                        K.float32(1.0), _add(K.float32(1.0), exp_neg)
                                    )
                                    K.assign(out_value, _mul(out_value, _mul(z_value, sigmoid)))
                                K.ptx.mov.b16(output_bits[k], _f32_to_bf16(out_value))
                            K.evaluate(
                                K.ptx.st.global_.v4.b16(
                                    output.ptr_to([out_base + d]),
                                    output_bits[0],
                                    output_bits[1],
                                    output_bits[2],
                                    output_bits[3],
                                )
                            )

        def run_load(IS_PAD: K.constexpr):
            lane: K.int32 = K.tid_in_role()
            kv_group: K.int32 = head // HEADS_PER_GROUP
            with K.If(lane == 0):
                with K.Then():
                    _tma_g2s_4d(
                        s_b.ptr_to([0]),
                        tensor_b,
                        0,
                        kv_group,
                        0,
                        batch_i,
                        full_barriers.ptr_to([0]),
                    )
                    _tma_g2s_4d(
                        s_c.ptr_to([0]),
                        tensor_c,
                        0,
                        kv_group,
                        0,
                        batch_i,
                        full_barriers.ptr_to([0]),
                    )
                    _tma_g2s_4d(
                        s_x.ptr_to([0]), tensor_x, 0, head, 0, batch_i, full_barriers.ptr_to([0])
                    )

            bcx_bytes: K.int32 = 2 * NTOKENS * DSTATE_PAD * 2 + NTOKENS * DIM * 2
            state_pipe = K.PipelineState(2, phase=0)
            with K.unroll(NUM_TMA_LOADS) as tl:
                _mbarrier_arrive_wait_parity(
                    empty_barriers.ptr_to([state_pipe.stage]), state_pipe.phase
                )
                with K.If(lane == 0):
                    with K.Then():
                        if not IS_PAD:
                            _tma_g2s_4d(
                                s_state_words.ptr_to([state_pipe.stage * STATE_STAGE_WORDS]),
                                tensor_state,
                                0,
                                tl * 32,
                                head,
                                state_batch,
                                full_barriers.ptr_to([state_pipe.stage]),
                            )
                            transaction_bytes = K.local_scalar("int32")
                            K.assign(transaction_bytes, STATE_STAGE_BYTES)
                            with K.If(tl == 0):
                                with K.Then():
                                    K.assign(transaction_bytes, transaction_bytes + bcx_bytes)
                            _mbarrier_arrive_expect_tx(
                                full_barriers.ptr_to([state_pipe.stage]), transaction_bytes
                            )
                        else:
                            transaction_bytes = K.local_scalar("int32")
                            K.assign(transaction_bytes, 0)
                            with K.If(tl == 0):
                                with K.Then():
                                    K.assign(transaction_bytes, bcx_bytes)
                            _mbarrier_arrive_expect_tx(
                                full_barriers.ptr_to([state_pipe.stage]), transaction_bytes
                            )
                state_pipe.advance()

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

    return selective_state_update_mtp_horizontal.func


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
