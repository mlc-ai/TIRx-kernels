# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's selective-state-update STP simple kernel.

Upstream source: include/flashinfer/mamba/kernel_selective_state_update_stp.cuh.
"""

import functools
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "selective_state_update_stp_simple",
    "category": "flashinfer",
    "compute_capability": 10,
}

_LOG2_E = 1.4426950408889634
_LN_2 = 0.6931471805599453
_FLT_LOWEST = -3.4028234663852886e38


def _lane_mask(raw_lane):
    """Preserve the shared lane-normalization leaf used by sibling STP ports."""
    return K.cast(K.bitwise_and(K.cast(raw_lane, "uint32"), K.uint32(31)), "int32")


def _global_load_index_s64(buffer, index, dtype):
    if dtype == "int32":
        gload_0 = K.alloc_local((1,), "int32")
        K.evaluate(K.ptx.ld.global_.s32(gload_0[0], buffer.ptr_to([index])))
        return K.cast(gload_0[0], "int64")
    gload_1 = K.alloc_local((1,), "int64")
    K.evaluate(K.ptx.ld.global_.s64(gload_1[0], buffer.ptr_to([index])))
    return gload_1[0]


def _load_weight(buffer, index, dtype: str):
    if dtype == "float32":
        gload_2 = K.alloc_local((1,), "uint32")
        K.evaluate(K.ptx.ld.global_.b32(gload_2[0], buffer.ptr_to([index])))
        return K.reinterpret("float32", gload_2[0])
    gload_3 = K.alloc_local((1,), "uint16")
    K.evaluate(K.ptx.ld.global_.b16(gload_3[0], buffer.ptr_to([index])))
    bf16_f32_0 = K.alloc_local((1,), "float32")
    K.evaluate(K.ptx.cvt.f32.bf16(bf16_f32_0[0], K.cast(gload_3[0], "uint16")))
    return bf16_f32_0[0]


def _state_bits_to_f32(bits, dtype: str):
    if dtype == "bfloat16":
        bf16_f32_1 = K.alloc_local((1,), "float32")
        K.evaluate(K.ptx.cvt.f32.bf16(bf16_f32_1[0], K.cast(bits, "uint16")))
        return bf16_f32_1[0]
    if dtype == "float16":
        f16_f32_0 = K.alloc_local((1,), "float32")
        K.evaluate(K.ptx.cvt.f32.f16(f16_f32_0[0], K.cast(bits, "uint16")))
        return f16_f32_0[0]
    if dtype == "int16":
        i16_f32_0 = K.alloc_local((1,), "float32")
        K.evaluate(
            K.ptx.cvt.rn.f32.s16(i16_f32_0[0], K.reinterpret("int16", K.cast(bits, "uint16")))
        )
        return i16_f32_0[0]
    return K.reinterpret("float32", K.cast(bits, "uint32"))


def _f32_to_state_bits(value, dtype: str):
    if dtype == "bfloat16":
        f32_bf16_0 = K.alloc_local((1,), "uint16")
        K.evaluate(K.ptx.cvt.rn.bf16.f32(f32_bf16_0[0], value))
        return f32_bf16_0[0]
    if dtype == "float16":
        f32_f16_0 = K.alloc_local((1,), "uint16")
        K.evaluate(K.ptx.cvt.rn.f16.f32(f32_f16_0[0], value))
        return f32_f16_0[0]
    return K.reinterpret("uint32", value)


def _load_two_byte_vector(buffer, index, count: int, scope: str):
    bits = K.alloc_local((count,), "uint16")
    prefix = f"ld.{scope}"
    if count == 2:
        K.evaluate(K.ptx[f"{prefix}.v2.b16"](bits[0], bits[1], buffer.ptr_to([index])))
    elif count == 3:
        for e in range(3):
            K.evaluate(K.ptx[f"{prefix}.b16"](bits[e], buffer.ptr_to([index + e])))
    elif count == 4:
        K.evaluate(
            K.ptx[f"{prefix}.v4.b16"](bits[0], bits[1], bits[2], bits[3], buffer.ptr_to([index]))
        )
    else:
        words = K.alloc_local((4,), "uint32")
        K.evaluate(
            K.ptx[f"{prefix}.v4.b32"](
                words[0], words[1], words[2], words[3], buffer.ptr_to([index])
            )
        )
        for pair in range(4):
            K.buffer_store(
                bits, K.cast(K.bitwise_and(words[pair], K.uint32(0xFFFF)), "uint16"), [2 * pair]
            )
            K.buffer_store(
                bits, K.cast(K.shift_right(words[pair], K.uint32(16)), "uint16"), [2 * pair + 1]
            )
    return bits


def _store_two_byte_vector(buffer, index, bits, count: int, scope: str = "global_"):
    prefix = f"st.{scope}"
    if count == 2:
        K.evaluate(K.ptx[f"{prefix}.v2.b16"](buffer.ptr_to([index]), bits[0], bits[1]))
    elif count == 3:
        for e in range(3):
            K.evaluate(K.ptx[f"{prefix}.b16"](buffer.ptr_to([index + e]), bits[e]))
    elif count == 4:
        K.evaluate(
            K.ptx[f"{prefix}.v4.b16"](buffer.ptr_to([index]), bits[0], bits[1], bits[2], bits[3])
        )
    else:
        words = K.alloc_local((4,), "uint32")
        for pair in range(4):
            K.buffer_store(
                words,
                K.bitwise_or(
                    K.cast(bits[2 * pair], "uint32"),
                    K.shift_left(K.cast(bits[2 * pair + 1], "uint32"), K.uint32(16)),
                ),
                [pair],
            )
        K.evaluate(
            K.ptx[f"{prefix}.v4.b32"](
                buffer.ptr_to([index]), words[0], words[1], words[2], words[3]
            )
        )


def _load_f32_vector(buffer, index, count: int):
    words = K.alloc_local((count,), "uint32")
    if count == 2:
        K.evaluate(K.ptx.ld.global_.v2.b32(words[0], words[1], buffer.ptr_to([index])))
    elif count == 3:
        for e in range(3):
            K.evaluate(K.ptx.ld.global_.b32(words[e], buffer.ptr_to([index + e])))
    else:
        K.evaluate(
            K.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
        )
    return words


def _store_f32_vector(buffer, index, words, count: int):
    if count == 2:
        K.evaluate(K.ptx.st.global_.v2.b32(buffer.ptr_to([index]), words[0], words[1]))
    elif count == 3:
        for e in range(3):
            K.evaluate(K.ptx.st.global_.b32(buffer.ptr_to([index + e]), words[e]))
    else:
        K.evaluate(
            K.ptx.st.global_.v4.b32(buffer.ptr_to([index]), words[0], words[1], words[2], words[3])
        )


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


# Every performance row changes one source branch or specialization from the
# base case.  The two simple launch modes are represented by base and batch=1.
BENCH_CONFIGS = [
    _case("b64_h64_d64_s128_r8_base"),
    _case("b1_h64_d64_s128_r8_tiled", batch=1),
    _case("b64_h8_d64_s128_r1", nheads=8),
    _case("b64_h64_d128_s128_r8", dim=128),
    _case("b64_h64_d256_s128_r8", dim=256),
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
    _case("b64_h64_d64_s128_r8_int16", state_dtype="int16"),
    _case("b64_h64_d64_s128_r8_philox10", state_dtype="float16", philox_rounds=10, seed=42),
    _case(
        "b64_h64_d64_s64_r8_philox10", dstate=64, state_dtype="float16", philox_rounds=10, seed=42
    ),
]


# Correctness includes every benchmark specialization plus the additional
# one-axis rows covered by FlashInfer's upstream STP tests.  The public
# FlashInfer API requires D, so its nullable device branch is correctness-only:
# a source/TIRx benchmark row could not exercise matching implementation paths.
CONFIGS = [dict(config) for config in BENCH_CONFIGS] + [
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
        for batch in (1, 16, 64)
    ],
    _case("b1_h64_d64_s128_r8_int16", batch=1, state_dtype="int16"),
    _case("b64_h8_d64_s128_r1_int16", nheads=8, state_dtype="int16"),
    _case("b64_h64_d128_s128_r8_int16", dim=128, state_dtype="int16"),
    _case("b64_h64_d64_s64_r8_int16", dstate=64, state_dtype="int16"),
    _case("b64_h64_d64_s256_r8_int16", dstate=256, state_dtype="int16"),
    _case("b64_h64_d64_s128_r8_int16_weightbf16", state_dtype="int16", weight_dtype="bfloat16"),
]


def _num_sms(device: str | torch.device = "cuda") -> int:
    del device
    from tirx_kernels.runner import hardware_num_sms

    return hardware_num_sms()


def _specialization(kwargs: dict[str, Any]) -> dict[str, Any]:
    batch = int(kwargs["batch"])
    nheads = int(kwargs["nheads"])
    dim = int(kwargs["dim"])
    dstate = int(kwargs["dstate"])
    ngroups = int(kwargs["ngroups"])
    state_dtype = str(kwargs["state_dtype"])
    state_stride_factor = int(kwargs.get("state_stride_factor", 1))
    has_dst_indices = bool(kwargs.get("has_dst_indices", False))
    state_slots = max((2 if has_dst_indices else 1) * batch + 8, 16)
    state_stride = nheads * dim * dstate * state_stride_factor
    scale_stride = nheads * dim
    index_elements = batch * (2 if int(kwargs.get("index_rank", 1)) == 2 else 1)
    rows_per_block = 4 if batch * nheads < 2 * _num_sms(kwargs.get("device", "cuda")) else dim
    scale_state = state_dtype == "int16"
    state_bytes = 4 if state_dtype == "float32" else 2
    state_vector = min(16 // state_bytes, dstate // 32)
    philox_rounds = int(kwargs.get("philox_rounds", 0))
    if scale_state and dstate not in (64, 128, 256):
        raise ValueError("int16 simple specializations require dstate in {64, 128, 256}")
    if philox_rounds not in (0, 10):
        raise ValueError("simple stochastic rounding supports philox_rounds in {0, 10}")
    if philox_rounds and (state_dtype != "float16" or dstate not in (64, 128)):
        raise ValueError("philox10 is scoped to float16 state with dstate 64 or 128")
    return {
        "BATCH": batch,
        "NHEADS": nheads,
        "DIM": dim,
        "DSTATE": dstate,
        "ROWS_PER_BLOCK": rows_per_block,
        "STATE_DTYPE": state_dtype,
        "WEIGHT_DTYPE": str(kwargs["weight_dtype"]),
        "INDEX_DTYPE": str(kwargs["index_dtype"]),
        "STATE_ELEMENTS": state_slots * state_stride,
        "SCALE_ELEMENTS": state_slots * scale_stride if scale_state else 1,
        "X_ELEMENTS": batch * nheads * dim,
        "DT_ELEMENTS": batch * nheads,
        "BC_ELEMENTS": batch * ngroups * dstate,
        "INDEX_ELEMENTS": max(index_elements, 1),
        "HAS_STATE_INDICES": bool(kwargs.get("has_state_indices", True)),
        "HAS_DST_INDICES": has_dst_indices,
        "HAS_Z": bool(kwargs.get("has_z", False)),
        "HAS_D": bool(kwargs.get("has_d", True)),
        "HAS_DT_BIAS": bool(kwargs.get("has_dt_bias", True)),
        "SCALE_STATE": scale_state,
        "PHILOX_ROUNDS": philox_rounds,
        "STATE_BYTES": state_bytes,
        "STATE_VECTOR": state_vector,
        "STATE_ITERATIONS": dstate // (32 * state_vector),
        "LANE_STATE_COUNT": dstate // 32,
    }


def get_kernel(**kwargs: Any):
    """Build the K entry for one simple specialization."""
    spec = _specialization(kwargs)

    DIM = spec["DIM"]
    DSTATE = spec["DSTATE"]
    HAS_D = spec["HAS_D"]
    HAS_DST_INDICES = spec["HAS_DST_INDICES"]
    HAS_DT_BIAS = spec["HAS_DT_BIAS"]
    HAS_STATE_INDICES = spec["HAS_STATE_INDICES"]
    HAS_Z = spec["HAS_Z"]
    INDEX_DTYPE = spec["INDEX_DTYPE"]
    LANE_STATE_COUNT = spec["LANE_STATE_COUNT"]
    PHILOX_ROUNDS = spec["PHILOX_ROUNDS"]
    ROWS_PER_BLOCK = spec["ROWS_PER_BLOCK"]
    SCALE_STATE = spec["SCALE_STATE"]
    STATE_BYTES = spec["STATE_BYTES"]
    STATE_DTYPE = spec["STATE_DTYPE"]
    STATE_ITERATIONS = spec["STATE_ITERATIONS"]
    STATE_VECTOR = spec["STATE_VECTOR"]
    WEIGHT_DTYPE = spec["WEIGHT_DTYPE"]

    @K.kernel(
        warps=4,
        arch="sm_100a",
        grid=(spec["BATCH"], spec["NHEADS"], "dim_tiles_runtime"),
        thread_layout="lane_warp",
    )
    def selective_state_update_stp_simple(
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
        rand_seed: K.gptr[K.i64],
        output: K.gptr[K.bf16],
        state_stride_batch: K.i64,
        state_scale_stride_batch: K.i64,
        x_stride_batch: K.i64,
        dt_stride_batch: K.i64,
        b_stride_batch: K.i64,
        c_stride_batch: K.i64,
        z_stride_batch: K.i64,
        out_stride_batch: K.i64,
        state_indices_stride_batch: K.i64,
        dst_indices_stride_batch: K.i64,
        nheads_runtime: K.i32,
        ngroups_runtime: K.i32,
        dt_softplus: K.i32,
        update_state: K.i32,
        pad_slot_id: K.i32,
        dim_tiles_runtime: K.i32,
    ):
        batch_i, head, dim_tile = K.cta_id()
        smem = K.smem_pool()
        s_x = smem.alloc((spec["ROWS_PER_BLOCK"],), K.bf16, align=16)
        s_z = smem.alloc((spec["ROWS_PER_BLOCK"],), K.bf16, align=16)
        s_b = smem.alloc((spec["DSTATE"],), K.bf16, align=16)
        s_c = smem.alloc((spec["DSTATE"],), K.bf16, align=16)
        s_out = smem.alloc((spec["ROWS_PER_BLOCK"],), K.f32, align=4)
        s_scale = (
            smem.alloc((spec["ROWS_PER_BLOCK"],), K.f32, align=16) if spec["SCALE_STATE"] else s_out
        )
        roles = K.specialize()
        load_x = roles.role("load_x_and_scale", warps=[0])
        load_b = roles.role("load_b", warps=[1])
        load_z = roles.role("load_z", warps=[2])
        load_c = roles.role("load_c", warps=[3])

        lane_ctx = K.alloc_local((1,), K.i32)
        dim_offset_ctx = K.alloc_local((1,), K.i32)
        K.assign(lane_ctx[0], K.lane_id())
        K.assign(dim_offset_ctx[0], dim_tile * ROWS_PER_BLOCK)
        lane = lane_ctx[0]
        warp = K.warp_id()
        dim_offset = dim_offset_ctx[0]
        rows_per_warp = (ROWS_PER_BLOCK + 3) // 4
        group_ctx = K.alloc_local((1,), K.i32)
        random_seed_ctx = K.alloc_local((1,), K.i64)
        state_batch_ctx = K.alloc_local((1,), K.i64)
        state_head_offset_ctx = K.alloc_local((1,), K.i64)
        dst_state_head_offset_ctx = K.alloc_local((1,), K.i64)
        scale_head_offset_ctx = K.alloc_local((1,), K.i64)
        dst_scale_head_offset_ctx = K.alloc_local((1,), K.i64)
        dt_value_ctx = K.alloc_local((1,), K.f32)
        da_value_ctx = K.alloc_local((1,), K.f32)
        d_value_ctx = K.alloc_local((1,), K.f32)

        def prepare_cta():
            # TIRX_TRANSCRIBE_START selective_state_update_stp_simple

            random_seed = K.local_scalar("int64")
            K.assign(random_seed, 0)
            if PHILOX_ROUNDS > 0 and not SCALE_STATE:
                K.evaluate(K.ptx.ld.global_.s64(random_seed, rand_seed.ptr_to([0])))

            state_batch = K.local_scalar("int64")
            if HAS_STATE_INDICES:
                K.assign(
                    state_batch,
                    _global_load_index_s64(
                        state_indices, batch_i * state_indices_stride_batch, INDEX_DTYPE
                    ),
                )
            else:
                K.assign(state_batch, K.cast(batch_i, "int64"))
            dst_state_batch = K.local_scalar("int64")
            if HAS_DST_INDICES:
                K.assign(
                    dst_state_batch,
                    _global_load_index_s64(
                        dst_indices, batch_i * dst_indices_stride_batch, INDEX_DTYPE
                    ),
                )
            else:
                K.assign(dst_state_batch, state_batch)

            state_head_offset: K.int64 = state_batch * state_stride_batch + K.cast(
                head * DIM * DSTATE, "int64"
            )
            dst_state_head_offset: K.int64 = dst_state_batch * state_stride_batch + K.cast(
                head * DIM * DSTATE, "int64"
            )
            scale_head_offset: K.int64 = state_batch * state_scale_stride_batch + K.cast(
                head * DIM, "int64"
            )
            dst_scale_head_offset: K.int64 = dst_state_batch * state_scale_stride_batch + K.cast(
                head * DIM, "int64"
            )

            gload_4 = K.alloc_local((1,), "uint32")
            K.evaluate(K.ptx.ld.global_.b32(gload_4[0], matrix_a.ptr_to([head])))
            a_value: K.float32 = K.reinterpret("float32", gload_4[0])
            dt_value = K.local_scalar("float32")
            K.assign(
                dt_value,
                _load_weight(dt, K.cast(batch_i, "int64") * dt_stride_batch + head, WEIGHT_DTYPE),
            )
            if HAS_DT_BIAS:
                K.evaluate(
                    K.ptx["add.ftz.f32"](
                        dt_value, dt_value, _load_weight(dt_bias, head, WEIGHT_DTYPE)
                    )
                )
            with K.If(dt_softplus != 0), K.Then():
                with K.If(dt_value <= K.float32(20.0)), K.Then():
                    mul_0 = K.alloc_local((1,), "float32")
                    K.evaluate(K.ptx["mul.ftz.f32"](mul_0[0], dt_value, K.float32(_LOG2_E)))
                    exp2_0 = K.alloc_local((1,), "float32")
                    K.evaluate(K.ptx["ex2.approx.ftz.f32"](exp2_0[0], mul_0[0]))
                    softplus_exp: K.float32 = exp2_0[0]
                    add_0 = K.alloc_local((1,), "float32")
                    K.evaluate(K.ptx["add.ftz.f32"](add_0[0], K.float32(1.0), softplus_exp))
                    log2_0 = K.alloc_local((1,), "float32")
                    K.evaluate(K.ptx["lg2.approx.ftz.f32"](log2_0[0], add_0[0]))
                    K.evaluate(K.ptx["mul.ftz.f32"](dt_value, log2_0[0], K.float32(_LN_2)))
            mul_1 = K.alloc_local((1,), "float32")
            K.evaluate(K.ptx["mul.ftz.f32"](mul_1[0], a_value, dt_value))
            mul_2 = K.alloc_local((1,), "float32")
            K.evaluate(K.ptx["mul.ftz.f32"](mul_2[0], mul_1[0], K.float32(_LOG2_E)))
            exp2_1 = K.alloc_local((1,), "float32")
            K.evaluate(K.ptx["ex2.approx.ftz.f32"](exp2_1[0], mul_2[0]))
            da_value: K.float32 = exp2_1[0]
            d_value = K.local_scalar("float32")
            K.assign(d_value, 0.0)
            if HAS_D:
                K.assign(d_value, _load_weight(d_weight, head, WEIGHT_DTYPE))

            K.assign(group_ctx[0], head // (nheads_runtime // ngroups_runtime))
            K.assign(random_seed_ctx[0], random_seed)
            K.assign(state_batch_ctx[0], state_batch)
            K.assign(state_head_offset_ctx[0], state_head_offset)
            K.assign(dst_state_head_offset_ctx[0], dst_state_head_offset)
            K.assign(scale_head_offset_ctx[0], scale_head_offset)
            K.assign(dst_scale_head_offset_ctx[0], dst_scale_head_offset)
            K.assign(dt_value_ctx[0], dt_value)
            K.assign(da_value_ctx[0], da_value)
            K.assign(d_value_ctx[0], d_value)

        def load_x_and_scale():
            scale_head_offset: K.int64 = scale_head_offset_ctx[0]
            with K.serial((ROWS_PER_BLOCK + 31) // 32) as preload_iter:
                local_row: K.int32 = lane + preload_iter * 32
                row_d: K.int32 = dim_offset + local_row
                with K.If(K.And(local_row < ROWS_PER_BLOCK, row_d < DIM)), K.Then():
                    gload_5 = K.alloc_local((1,), "uint16")
                    K.evaluate(
                        K.ptx.ld.global_.b16(
                            gload_5[0],
                            x.ptr_to(
                                [K.cast(batch_i, "int64") * x_stride_batch + head * DIM + row_d]
                            ),
                        )
                    )
                    x_bits: K.uint16 = gload_5[0]
                    K.evaluate(K.ptx.st.shared.b16(s_x.ptr_to([local_row]), x_bits))
            if SCALE_STATE:
                with K.serial((ROWS_PER_BLOCK + 31) // 32) as scale_iter:
                    local_row: K.int32 = lane + scale_iter * 32
                    row_d: K.int32 = dim_offset + local_row
                    with K.If(K.And(local_row < ROWS_PER_BLOCK, row_d < DIM)), K.Then():
                        gload_6 = K.alloc_local((1,), "uint32")
                        K.evaluate(
                            K.ptx.ld.global_.b32(
                                gload_6[0], state_scale.ptr_to([scale_head_offset + row_d])
                            )
                        )
                        scale_bits: K.uint32 = gload_6[0]
                        K.evaluate(K.ptx.st.shared.b32(s_scale.ptr_to([local_row]), scale_bits))

        def load_bc_values(s_dst, src, src_stride):
            group: K.int32 = group_ctx[0]
            bc_i: K.int32 = lane * 8
            with K.If(bc_i < DSTATE), K.Then():
                bc_words = K.alloc_local((4,), "uint32")
                K.evaluate(
                    K.ptx.ld.global_.v4.b32(
                        bc_words[0],
                        bc_words[1],
                        bc_words[2],
                        bc_words[3],
                        src.ptr_to([K.cast(batch_i, "int64") * src_stride + group * DSTATE + bc_i]),
                    )
                )
                K.evaluate(
                    K.ptx.st.shared.v4.b32(
                        s_dst.ptr_to([bc_i]), bc_words[0], bc_words[1], bc_words[2], bc_words[3]
                    )
                )

        def load_z_values():
            with K.serial((ROWS_PER_BLOCK + 31) // 32) as preload_iter:
                local_row: K.int32 = lane + preload_iter * 32
                row_d: K.int32 = dim_offset + local_row
                with K.If(K.And(local_row < ROWS_PER_BLOCK, row_d < DIM)), K.Then():
                    if HAS_Z:
                        gload_7 = K.alloc_local((1,), "uint16")
                        K.evaluate(
                            K.ptx.ld.global_.b16(
                                gload_7[0],
                                z.ptr_to(
                                    [K.cast(batch_i, "int64") * z_stride_batch + head * DIM + row_d]
                                ),
                            )
                        )
                        z_bits: K.uint16 = gload_7[0]
                        K.evaluate(K.ptx.st.shared.b16(s_z.ptr_to([local_row]), z_bits))
                    else:
                        K.evaluate(K.ptx.st.shared.b16(s_z.ptr_to([local_row]), K.uint16(0)))

        def update_rows():
            random_seed: K.int64 = random_seed_ctx[0]
            state_batch: K.int64 = state_batch_ctx[0]
            state_head_offset: K.int64 = state_head_offset_ctx[0]
            dst_state_head_offset: K.int64 = dst_state_head_offset_ctx[0]
            dt_value: K.float32 = dt_value_ctx[0]
            da_value: K.float32 = da_value_ctx[0]
            d_value: K.float32 = d_value_ctx[0]

            with K.serial(rows_per_warp) as row_in_warp:
                local_row_ctx = K.local_scalar("int32")
                K.assign(local_row_ctx, warp * rows_per_warp + row_in_warp)
                local_row: K.int32 = local_row_ctx
                row_d_ctx = K.local_scalar("int32")
                K.assign(row_d_ctx, dim_offset + local_row)
                row_d: K.int32 = row_d_ctx
                with K.If(row_d < DIM), K.Then():
                    sload_0 = K.alloc_local((1,), "uint16")
                    K.evaluate(K.ptx.ld.shared.b16(sload_0[0], s_x.ptr_to([local_row])))
                    bf16_f32_2 = K.alloc_local((1,), "float32")
                    K.evaluate(K.ptx.cvt.f32.bf16(bf16_f32_2[0], K.cast(sload_0[0], "uint16")))
                    x_value: K.float32 = bf16_f32_2[0]
                    decode_scale = K.local_scalar("float32")
                    K.assign(decode_scale, 1.0)
                    new_state_max = K.local_scalar("float32")
                    K.assign(new_state_max, K.float32(_FLT_LOWEST))
                    if SCALE_STATE:
                        sload_1 = K.alloc_local((1,), "uint32")
                        K.evaluate(K.ptx.ld.shared.b32(sload_1[0], s_scale.ptr_to([local_row])))
                        K.assign(decode_scale, K.reinterpret("float32", sload_1[0]))
                    mul_3 = K.alloc_local((1,), "float32")
                    K.evaluate(K.ptx["mul.ftz.f32"](mul_3[0], d_value, x_value))
                    d_times_x: K.float32 = mul_3[0]
                    out_value = K.local_scalar("float32")
                    K.assign(out_value, K.if_then_else(lane == 0, d_times_x, K.float32(0.0)))
                    new_states = K.alloc_local((LANE_STATE_COUNT,), "float32")
                    with K.unroll(STATE_ITERATIONS) as state_iter:
                        state_i_ctx = K.local_scalar("int32")
                        K.assign(state_i_ctx, (state_iter * 32 + lane) * STATE_VECTOR)
                        state_i: K.int32 = state_i_ctx
                        if STATE_BYTES == 2:
                            r_state = K.alloc_local((STATE_VECTOR,), "uint16")
                            with K.unroll(STATE_VECTOR) as e:
                                K.ptx.mov.b16(r_state[e], K.uint16(0))
                            with K.If(state_batch != K.cast(pad_slot_id, "int64")), K.Then():
                                loaded_state = _load_two_byte_vector(
                                    state,
                                    state_head_offset + row_d * DSTATE + state_i,
                                    STATE_VECTOR,
                                    "global",
                                )
                                with K.unroll(STATE_VECTOR) as e:
                                    K.ptx.mov.b16(r_state[e], loaded_state[e])
                        else:
                            r_state = K.alloc_local((STATE_VECTOR,), "uint32")
                            with K.unroll(STATE_VECTOR) as e:
                                K.ptx.mov.b32(r_state[e], K.uint32(0))
                            with K.If(state_batch != K.cast(pad_slot_id, "int64")), K.Then():
                                loaded_state = _load_f32_vector(
                                    state,
                                    state_head_offset + row_d * DSTATE + state_i,
                                    STATE_VECTOR,
                                )
                                with K.unroll(STATE_VECTOR) as e:
                                    K.ptx.mov.b32(r_state[e], loaded_state[e])

                        b_bits = K.alloc_local((STATE_VECTOR,), "uint16")
                        c_bits = K.alloc_local((STATE_VECTOR,), "uint16")
                        random_words = K.alloc_local((4,), "uint32")
                        sr_raw = K.alloc_local((STATE_VECTOR,), "uint32")
                        with K.unroll(STATE_VECTOR) as e:
                            with (
                                K.If(
                                    K.And(K.And(PHILOX_ROUNDS > 0, K.Not(SCALE_STATE)), e % 4 == 0)
                                ),
                                K.Then(),
                            ):
                                random_offset: K.uint64 = K.cast(
                                    state_head_offset + row_d * DSTATE + state_i + e, "uint64"
                                )
                                c0 = K.local_scalar("uint32")
                                K.assign(c0, K.cast(random_offset, "uint32"))
                                c1 = K.local_scalar("uint32")
                                K.assign(
                                    c1, K.cast(K.shift_right(random_offset, K.uint64(32)), "uint32")
                                )
                                c2 = K.local_scalar("uint32")
                                K.assign(c2, 0)
                                c3 = K.local_scalar("uint32")
                                K.assign(c3, 0)
                                k0 = K.local_scalar("uint32")
                                K.assign(k0, K.cast(K.reinterpret("uint64", random_seed), "uint32"))
                                k1 = K.local_scalar("uint32")
                                K.assign(
                                    k1,
                                    K.cast(
                                        K.shift_right(
                                            K.reinterpret("uint64", random_seed), K.uint64(32)
                                        ),
                                        "uint32",
                                    ),
                                )
                                with K.unroll(10) as philox_round:
                                    old_c0: K.uint32 = c0
                                    old_c2: K.uint32 = c2
                                    mul_hi_0 = K.alloc_local((1,), "uint32")
                                    K.evaluate(
                                        K.ptx["mul.hi.u32"](
                                            mul_hi_0[0], K.uint32(0xCD9E8D57), old_c2
                                        )
                                    )
                                    hi_b: K.uint32 = mul_hi_0[0]
                                    next_c0: K.uint32 = K.bitwise_xor(K.bitwise_xor(hi_b, c1), k0)
                                    mul_hi_1 = K.alloc_local((1,), "uint32")
                                    K.evaluate(
                                        K.ptx["mul.hi.u32"](
                                            mul_hi_1[0], K.uint32(0xD2511F53), old_c0
                                        )
                                    )
                                    hi_a: K.uint32 = mul_hi_1[0]
                                    next_c2: K.uint32 = K.bitwise_xor(K.bitwise_xor(hi_a, c3), k1)
                                    mul_lo_0 = K.alloc_local((1,), "int32")
                                    K.evaluate(
                                        K.ptx["mul.lo.s32"](
                                            mul_lo_0[0],
                                            K.int32(-845247145),
                                            K.reinterpret("int32", old_c2),
                                        )
                                    )
                                    next_c1_s: K.int32 = mul_lo_0[0]
                                    mul_lo_1 = K.alloc_local((1,), "int32")
                                    K.evaluate(
                                        K.ptx["mul.lo.s32"](
                                            mul_lo_1[0],
                                            K.int32(-766435501),
                                            K.reinterpret("int32", old_c0),
                                        )
                                    )
                                    next_c3_s: K.int32 = mul_lo_1[0]
                                    add_s32_0 = K.alloc_local((1,), "int32")
                                    K.evaluate(
                                        K.ptx["add.s32"](
                                            add_s32_0[0],
                                            K.reinterpret("int32", k0),
                                            K.int32(-1640531527),
                                        )
                                    )
                                    next_k0_s: K.int32 = add_s32_0[0]
                                    add_s32_1 = K.alloc_local((1,), "int32")
                                    K.evaluate(
                                        K.ptx["add.s32"](
                                            add_s32_1[0],
                                            K.reinterpret("int32", k1),
                                            K.int32(-1150833019),
                                        )
                                    )
                                    next_k1_s: K.int32 = add_s32_1[0]
                                    K.assign(c0, next_c0)
                                    K.assign(c1, K.reinterpret("uint32", next_c1_s))
                                    K.assign(c2, next_c2)
                                    K.assign(c3, K.reinterpret("uint32", next_c3_s))
                                    K.assign(k0, K.reinterpret("uint32", next_k0_s))
                                    K.assign(k1, K.reinterpret("uint32", next_k1_s))
                                K.ptx.mov.b32(random_words[0], c0)
                                K.ptx.mov.b32(random_words[1], c1)
                                K.ptx.mov.b32(random_words[2], c2)
                                K.ptx.mov.b32(random_words[3], c3)

                            state_value = K.local_scalar("float32")
                            K.assign(state_value, _state_bits_to_f32(r_state[e], STATE_DTYPE))
                            if SCALE_STATE:
                                K.evaluate(
                                    K.ptx["mul.ftz.f32"](state_value, state_value, decode_scale)
                                )
                            if STATE_VECTOR == 3:
                                K.evaluate(
                                    K.ptx.ld.shared.b16(b_bits[e], s_b.ptr_to([state_i + e]))
                                )
                                bf16_f32_3 = K.alloc_local((1,), "float32")
                                K.evaluate(
                                    K.ptx.cvt.f32.bf16(bf16_f32_3[0], K.cast(b_bits[e], "uint16"))
                                )
                                b_value: K.float32 = bf16_f32_3[0]
                                K.evaluate(
                                    K.ptx.ld.shared.b16(c_bits[e], s_c.ptr_to([state_i + e]))
                                )
                                bf16_f32_4 = K.alloc_local((1,), "float32")
                                K.evaluate(
                                    K.ptx.cvt.f32.bf16(bf16_f32_4[0], K.cast(c_bits[e], "uint16"))
                                )
                                c_value: K.float32 = bf16_f32_4[0]
                            else:
                                with K.If(e == 0), K.Then():
                                    loaded_b = _load_two_byte_vector(
                                        s_b, state_i, STATE_VECTOR, "shared"
                                    )
                                    with K.unroll(STATE_VECTOR) as copy_e:
                                        K.ptx.mov.b16(b_bits[copy_e], loaded_b[copy_e])
                                bf16_f32_5 = K.alloc_local((1,), "float32")
                                K.evaluate(
                                    K.ptx.cvt.f32.bf16(bf16_f32_5[0], K.cast(b_bits[e], "uint16"))
                                )
                                b_value: K.float32 = bf16_f32_5[0]
                                with K.If(e == 0), K.Then():
                                    loaded_c = _load_two_byte_vector(
                                        s_c, state_i, STATE_VECTOR, "shared"
                                    )
                                    with K.unroll(STATE_VECTOR) as copy_e:
                                        K.ptx.mov.b16(c_bits[copy_e], loaded_c[copy_e])
                                bf16_f32_6 = K.alloc_local((1,), "float32")
                                K.evaluate(
                                    K.ptx.cvt.f32.bf16(bf16_f32_6[0], K.cast(c_bits[e], "uint16"))
                                )
                                c_value: K.float32 = bf16_f32_6[0]

                            mul_4 = K.alloc_local((1,), "float32")
                            K.evaluate(K.ptx["mul.ftz.f32"](mul_4[0], b_value, dt_value))
                            db_value: K.float32 = mul_4[0]
                            mul_5 = K.alloc_local((1,), "float32")
                            K.evaluate(K.ptx["mul.ftz.f32"](mul_5[0], db_value, x_value))
                            db_x: K.float32 = mul_5[0]
                            fma_0 = K.alloc_local((1,), "float32")
                            K.evaluate(
                                K.ptx["fma.rn.ftz.f32"](fma_0[0], state_value, da_value, db_x)
                            )
                            new_state: K.float32 = fma_0[0]
                            if SCALE_STATE:
                                abs_0 = K.alloc_local((1,), "float32")
                                K.evaluate(K.ptx["abs.ftz.f32"](abs_0[0], new_state))
                                magnitude: K.float32 = abs_0[0]
                                K.evaluate(
                                    K.ptx["max.ftz.f32"](new_state_max, new_state_max, magnitude)
                                )
                                K.ptx.mov.b32(new_states[state_iter * STATE_VECTOR + e], new_state)
                            elif PHILOX_ROUNDS > 0:
                                random13: K.uint32 = K.bitwise_and(
                                    random_words[e % 4], K.uint32(0x1FFF)
                                )
                                K.evaluate(
                                    K.ptx.cvt.rs.f16x2.f32(
                                        sr_raw[e], K.float32(0.0), new_state, random13
                                    )
                                )
                            elif STATE_BYTES == 2:
                                K.ptx.mov.b16(
                                    r_state[e], _f32_to_state_bits(new_state, STATE_DTYPE)
                                )
                            else:
                                K.ptx.mov.b32(
                                    r_state[e], _f32_to_state_bits(new_state, STATE_DTYPE)
                                )
                            K.evaluate(
                                K.ptx["fma.rn.ftz.f32"](out_value, new_state, c_value, out_value)
                            )

                        with (
                            K.If(
                                K.And(
                                    K.And(K.Not(SCALE_STATE), update_state != 0),
                                    state_batch != K.cast(pad_slot_id, "int64"),
                                )
                            ),
                            K.Then(),
                        ):
                            if PHILOX_ROUNDS > 0:
                                sr_words = K.alloc_local((STATE_VECTOR // 2,), "uint32")
                                with K.unroll(STATE_VECTOR // 2) as pair:
                                    prmt_0 = K.alloc_local((1,), "uint32")
                                    K.evaluate(
                                        K.ptx["prmt.b32"](
                                            prmt_0[0],
                                            K.cast(sr_raw[2 * pair], "uint32"),
                                            K.cast(sr_raw[2 * pair + 1], "uint32"),
                                            K.uint32(0x5410),
                                        )
                                    )
                                    K.ptx.mov.b32(sr_words[pair], prmt_0[0])
                                if STATE_VECTOR == 2:
                                    K.evaluate(
                                        K.ptx.st.global_.b32(
                                            state.ptr_to(
                                                [dst_state_head_offset + row_d * DSTATE + state_i]
                                            ),
                                            sr_words[0],
                                        )
                                    )
                                else:
                                    K.evaluate(
                                        K.ptx.st.global_.v2.b32(
                                            state.ptr_to(
                                                [dst_state_head_offset + row_d * DSTATE + state_i]
                                            ),
                                            sr_words[0],
                                            sr_words[1],
                                        )
                                    )
                            elif STATE_BYTES == 2:
                                _store_two_byte_vector(
                                    state,
                                    dst_state_head_offset + row_d * DSTATE + state_i,
                                    r_state,
                                    STATE_VECTOR,
                                )
                            else:
                                _store_f32_vector(
                                    state,
                                    dst_state_head_offset + row_d * DSTATE + state_i,
                                    r_state,
                                    STATE_VECTOR,
                                )

                    with K.unroll(5) as delta_i:
                        delta: K.int32 = K.shift_right(K.int32(16), delta_i)
                        peer_out: K.float32 = K.cuda.__shfl_down_sync(
                            K.uint32(0xFFFFFFFF), out_value, delta, 32
                        )
                        K.evaluate(K.ptx["add.ftz.f32"](out_value, out_value, peer_out))
                    with K.If(lane == 0), K.Then():
                        K.evaluate(
                            K.ptx.st.shared.b32(
                                s_out.ptr_to([local_row]), K.reinterpret("uint32", out_value)
                            )
                        )

                    with (
                        K.If(
                            K.And(
                                K.And(SCALE_STATE, update_state != 0),
                                state_batch != K.cast(pad_slot_id, "int64"),
                            )
                        ),
                        K.Then(),
                    ):
                        with K.unroll(5) as delta_i:
                            delta: K.int32 = K.shift_right(K.int32(16), delta_i)
                            peer_max: K.float32 = K.cuda.__shfl_down_sync(
                                K.uint32(0xFFFFFFFF), new_state_max, delta, 32
                            )
                            K.evaluate(K.ptx["max.ftz.f32"](new_state_max, new_state_max, peer_max))
                        K.cuda.warp_sync()
                        K.assign(
                            new_state_max,
                            K.cuda.__shfl_sync(K.uint32(0xFFFFFFFF), new_state_max, 0, 32),
                        )
                        encode_scale = K.local_scalar("float32")
                        K.assign(encode_scale, 1.0)
                        with K.If(new_state_max != K.float32(0.0)), K.Then():
                            K.evaluate(
                                K.ptx["div.approx.ftz.f32"](
                                    encode_scale, K.float32(32767.0), new_state_max
                                )
                            )
                        rcp_0 = K.alloc_local((1,), "float32")
                        K.evaluate(K.ptx["rcp.approx.ftz.f32"](rcp_0[0], encode_scale))
                        new_decode_scale: K.float32 = rcp_0[0]
                        with K.unroll(STATE_ITERATIONS) as state_iter:
                            state_i: K.int32 = (state_iter * 32 + lane) * STATE_VECTOR
                            quantized = K.alloc_local((STATE_VECTOR,), "int32")
                            packed_quantized = K.alloc_local((STATE_VECTOR // 2,), "uint32")
                            with K.unroll(STATE_VECTOR) as e:
                                mul_6 = K.alloc_local((1,), "float32")
                                K.evaluate(
                                    K.ptx["mul.ftz.f32"](
                                        mul_6[0],
                                        new_states[state_iter * STATE_VECTOR + e],
                                        encode_scale,
                                    )
                                )
                                scaled: K.float32 = mul_6[0]
                                max_0 = K.alloc_local((1,), "float32")
                                K.evaluate(
                                    K.ptx["max.ftz.f32"](max_0[0], scaled, K.float32(-32767.0))
                                )
                                clipped_low: K.float32 = max_0[0]
                                min_0 = K.alloc_local((1,), "float32")
                                K.evaluate(
                                    K.ptx["min.ftz.f32"](min_0[0], clipped_low, K.float32(32767.0))
                                )
                                clipped: K.float32 = min_0[0]
                                K.evaluate(K.ptx.cvt.rni.ftz.s32.f32(quantized[e], clipped))
                            with K.unroll(STATE_VECTOR // 2) as pair:
                                prmt_1 = K.alloc_local((1,), "uint32")
                                K.evaluate(
                                    K.ptx["prmt.b32"](
                                        prmt_1[0],
                                        K.cast(
                                            K.reinterpret("uint32", quantized[2 * pair]), "uint32"
                                        ),
                                        K.cast(
                                            K.reinterpret("uint32", quantized[2 * pair + 1]),
                                            "uint32",
                                        ),
                                        K.uint32(0x5410),
                                    )
                                )
                                K.ptx.mov.b32(packed_quantized[pair], prmt_1[0])
                            if STATE_VECTOR == 2:
                                K.evaluate(
                                    K.ptx.st.global_.b32(
                                        state.ptr_to(
                                            [dst_state_head_offset + row_d * DSTATE + state_i]
                                        ),
                                        packed_quantized[0],
                                    )
                                )
                            elif STATE_VECTOR == 4:
                                K.evaluate(
                                    K.ptx.st.global_.v2.b32(
                                        state.ptr_to(
                                            [dst_state_head_offset + row_d * DSTATE + state_i]
                                        ),
                                        packed_quantized[0],
                                        packed_quantized[1],
                                    )
                                )
                            else:
                                K.evaluate(
                                    K.ptx.st.global_.v4.b32(
                                        state.ptr_to(
                                            [dst_state_head_offset + row_d * DSTATE + state_i]
                                        ),
                                        packed_quantized[0],
                                        packed_quantized[1],
                                        packed_quantized[2],
                                        packed_quantized[3],
                                    )
                                )
                        with K.If(lane == 0), K.Then():
                            K.evaluate(
                                K.ptx.st.shared.b32(
                                    s_scale.ptr_to([local_row]),
                                    K.reinterpret("uint32", new_decode_scale),
                                )
                            )

        def store_outputs():
            state_batch: K.int64 = state_batch_ctx[0]
            dst_scale_head_offset: K.int64 = dst_scale_head_offset_ctx[0]
            with K.serial((ROWS_PER_BLOCK + 127) // 128) as output_iter:
                row_in_warp: K.int32 = lane + output_iter * 32
                local_row: K.int32 = warp * rows_per_warp + row_in_warp
                row_d: K.int32 = dim_offset + local_row
                with K.If(K.And(row_in_warp < rows_per_warp, row_d < DIM)), K.Then():
                    out_value = K.local_scalar("float32")
                    sload_2 = K.alloc_local((1,), "uint32")
                    K.evaluate(K.ptx.ld.shared.b32(sload_2[0], s_out.ptr_to([local_row])))
                    K.assign(out_value, K.reinterpret("float32", sload_2[0]))
                    if HAS_Z:
                        sload_3 = K.alloc_local((1,), "uint16")
                        K.evaluate(K.ptx.ld.shared.b16(sload_3[0], s_z.ptr_to([local_row])))
                        bf16_f32_7 = K.alloc_local((1,), "float32")
                        K.evaluate(K.ptx.cvt.f32.bf16(bf16_f32_7[0], K.cast(sload_3[0], "uint16")))
                        z_value: K.float32 = bf16_f32_7[0]
                        sub_0 = K.alloc_local((1,), "float32")
                        K.evaluate(K.ptx["sub.ftz.f32"](sub_0[0], K.float32(0.0), z_value))
                        neg_z: K.float32 = sub_0[0]
                        mul_7 = K.alloc_local((1,), "float32")
                        K.evaluate(K.ptx["mul.ftz.f32"](mul_7[0], neg_z, K.float32(_LOG2_E)))
                        z_exp_arg: K.float32 = mul_7[0]
                        exp2_2 = K.alloc_local((1,), "float32")
                        K.evaluate(K.ptx["ex2.approx.ftz.f32"](exp2_2[0], z_exp_arg))
                        exp_neg_z: K.float32 = exp2_2[0]
                        add_1 = K.alloc_local((1,), "float32")
                        K.evaluate(K.ptx["add.ftz.f32"](add_1[0], K.float32(1.0), exp_neg_z))
                        denominator: K.float32 = add_1[0]
                        div_0 = K.alloc_local((1,), "float32")
                        K.evaluate(
                            K.ptx["div.approx.ftz.f32"](div_0[0], K.float32(1.0), denominator)
                        )
                        sigmoid_z: K.float32 = div_0[0]
                        mul_8 = K.alloc_local((1,), "float32")
                        K.evaluate(K.ptx["mul.ftz.f32"](mul_8[0], z_value, sigmoid_z))
                        silu_z: K.float32 = mul_8[0]
                        K.evaluate(K.ptx["mul.ftz.f32"](out_value, out_value, silu_z))
                    f32_bf16_1 = K.alloc_local((1,), "uint16")
                    K.evaluate(K.ptx.cvt.rn.bf16.f32(f32_bf16_1[0], out_value))
                    output_bits: K.uint16 = f32_bf16_1[0]
                    K.evaluate(
                        K.ptx.st.global_.b16(
                            output.ptr_to(
                                [K.cast(batch_i, "int64") * out_stride_batch + head * DIM + row_d]
                            ),
                            output_bits,
                        )
                    )
            with (
                K.If(
                    K.And(
                        K.And(SCALE_STATE, update_state != 0),
                        state_batch != K.cast(pad_slot_id, "int64"),
                    )
                ),
                K.Then(),
            ):
                with K.serial((ROWS_PER_BLOCK + 127) // 128) as scale_iter:
                    row_in_warp: K.int32 = lane + scale_iter * 32
                    local_row: K.int32 = warp * rows_per_warp + row_in_warp
                    row_d: K.int32 = dim_offset + local_row
                    with K.If(K.And(row_in_warp < rows_per_warp, row_d < DIM)), K.Then():
                        sload_4 = K.alloc_local((1,), "uint32")
                        K.evaluate(K.ptx.ld.shared.b32(sload_4[0], s_scale.ptr_to([local_row])))
                        scale_bits: K.uint32 = sload_4[0]
                        K.evaluate(
                            K.ptx.st.global_.b32(
                                state_scale.ptr_to([dst_scale_head_offset + row_d]), scale_bits
                            )
                        )

        prepare_cta()
        with load_x:
            load_x_and_scale()
        with load_b:
            load_bc_values(s_b, matrix_b, b_stride_batch)
        with load_z:
            load_z_values()
        with load_c:
            load_bc_values(s_c, matrix_c, c_stride_batch)
        K.cuda.cta_sync()
        update_rows()
        K.cuda.cta_sync()
        store_outputs()

    return selective_state_update_stp_simple.func


_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}


@functools.cache
def _load_oracle():
    from flashinfer.mamba import selective_state_update

    return selective_state_update


def _view_state(raw: torch.Tensor, spec: dict[str, Any], state_stride: int) -> torch.Tensor:
    return raw.as_strided(
        (spec["STATE_ELEMENTS"] // state_stride, spec["NHEADS"], spec["DIM"], spec["DSTATE"]),
        (state_stride, spec["DIM"] * spec["DSTATE"], spec["DSTATE"], 1),
    )


def _view_scale(raw: torch.Tensor, spec: dict[str, Any], scale_stride: int) -> torch.Tensor:
    return raw.as_strided(
        (spec["SCALE_ELEMENTS"] // scale_stride, spec["NHEADS"], spec["DIM"]),
        (scale_stride, spec["DIM"], 1),
    )


def _index_tensor(
    values: torch.Tensor, *, rank: int, total_elements: int, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if rank == 1:
        shaped = values.contiguous()
        return shaped, shaped.reshape(-1), 1
    shaped = torch.empty((values.numel(), 2), dtype=values.dtype, device=device)
    shaped[:, 0] = values
    shaped[:, 1] = values
    flat = shaped.reshape(-1)
    if flat.numel() != total_elements:
        raise AssertionError((flat.numel(), total_elements))
    return shaped, flat, 2


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Create independent mutable TIRx/source cases for one specialization."""
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for selective-state-update STP simple")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(f"STP simple SM100 requires compute capability 10.x, got {capability}")

    spec = _specialization(kwargs)
    batch = spec["BATCH"]
    nheads = spec["NHEADS"]
    dim = spec["DIM"]
    dstate = spec["DSTATE"]
    ngroups = int(kwargs["ngroups"])
    state_dtype = _TORCH_DTYPES[str(kwargs["state_dtype"])]
    weight_dtype = _TORCH_DTYPES[str(kwargs["weight_dtype"])]
    index_dtype = _TORCH_DTYPES[str(kwargs["index_dtype"])]
    state_stride = nheads * dim * dstate * int(kwargs.get("state_stride_factor", 1))
    scale_stride = nheads * dim
    state_slots = spec["STATE_ELEMENTS"] // state_stride
    generator = torch.Generator(device=device)
    generator.manual_seed(int(kwargs.get("seed", 0)) + 20260808)

    if state_dtype == torch.int16:
        logical_f32 = torch.randn(
            (state_slots, nheads, dim, dstate),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        amax = logical_f32.abs().amax(dim=-1)
        encode = torch.where(amax == 0, torch.ones_like(amax), 32767.0 / amax)
        quantized = (logical_f32 * encode[..., None]).round().clamp(-32767, 32767).to(torch.int16)
        initial_state_raw = torch.zeros(spec["STATE_ELEMENTS"], dtype=torch.int16, device=device)
        initial_state_view = _view_state(initial_state_raw, spec, state_stride)
        initial_state_view.copy_(quantized)
        initial_scale_raw = torch.zeros(spec["SCALE_ELEMENTS"], dtype=torch.float32, device=device)
        _view_scale(initial_scale_raw, spec, scale_stride).copy_(1.0 / encode)
        del logical_f32, quantized, amax, encode
    else:
        initial_state_raw = torch.randn(
            (spec["STATE_ELEMENTS"],), dtype=state_dtype, device=device, generator=generator
        )
        initial_scale_raw = torch.zeros((1,), dtype=torch.float32, device=device)

    x = torch.randn((batch, nheads, dim), dtype=torch.bfloat16, device=device, generator=generator)
    dt_base = torch.randn((batch, nheads), dtype=weight_dtype, device=device, generator=generator)
    dt_view = dt_base.as_strided((batch, nheads, dim), (nheads, 1, 0))
    matrix_a_base = (
        -torch.rand((nheads,), dtype=torch.float32, device=device, generator=generator) - 1.0
    )
    matrix_a_view = matrix_a_base.as_strided((nheads, dim, dstate), (1, 0, 0))
    matrix_b = torch.randn(
        (batch, ngroups, dstate), dtype=torch.bfloat16, device=device, generator=generator
    )
    matrix_c = torch.randn(
        (batch, ngroups, dstate), dtype=torch.bfloat16, device=device, generator=generator
    )
    d_base = torch.randn((nheads,), dtype=weight_dtype, device=device, generator=generator)
    if not bool(kwargs.get("has_d", True)):
        d_base.zero_()
    d_view = d_base.as_strided((nheads, dim), (1, 0))
    bias_base = torch.rand((nheads,), dtype=weight_dtype, device=device, generator=generator) - 4.0
    bias_view = bias_base.as_strided((nheads, dim), (1, 0))
    z = torch.randn((batch, nheads, dim), dtype=torch.bfloat16, device=device, generator=generator)

    rank = int(kwargs.get("index_rank", 1))
    if bool(kwargs.get("has_dst_indices", False)):
        state_values = torch.arange(batch, dtype=index_dtype, device=device)
        dst_values = torch.arange(batch, dtype=index_dtype, device=device) + batch
    else:
        state_values = torch.randperm(state_slots, device=device, generator=generator)[:batch].to(
            index_dtype
        )
        dst_values = state_values.clone()
    pad_every = int(kwargs.get("pad_every", 0))
    pad_slot_id = -1
    if pad_every:
        state_values[::pad_every] = pad_slot_id
    state_indices, state_indices_flat, state_index_stride = _index_tensor(
        state_values, rank=rank, total_elements=spec["INDEX_ELEMENTS"], device=device
    )
    dst_indices, dst_indices_flat, dst_index_stride = _index_tensor(
        dst_values, rank=rank, total_elements=spec["INDEX_ELEMENTS"], device=device
    )
    seed = torch.tensor([int(kwargs.get("seed", 0))], dtype=torch.int64, device=device)

    tirx_state_raw = initial_state_raw.clone()
    reference_state_raw = initial_state_raw.clone()
    tirx_scale_raw = initial_scale_raw.clone()
    reference_scale_raw = initial_scale_raw.clone()
    tirx_output = torch.empty((batch, nheads, dim), dtype=torch.bfloat16, device=device)
    reference_output = torch.empty_like(tirx_output)
    dummy_index = torch.zeros((spec["INDEX_ELEMENTS"],), dtype=index_dtype, device=device)

    case = {
        "kwargs": dict(kwargs),
        "spec": spec,
        "state_stride": state_stride,
        "scale_stride": scale_stride,
        "initial_state_raw": initial_state_raw,
        "initial_scale_raw": initial_scale_raw,
        "tirx_state_raw": tirx_state_raw,
        "reference_state_raw": reference_state_raw,
        "tirx_scale_raw": tirx_scale_raw,
        "reference_scale_raw": reference_scale_raw,
        "tirx_output": tirx_output,
        "reference_output": reference_output,
        "x": x,
        "dt_base": dt_base,
        "dt_view": dt_view,
        "matrix_a_base": matrix_a_base,
        "matrix_a_view": matrix_a_view,
        "matrix_b": matrix_b,
        "matrix_c": matrix_c,
        "d_base": d_base,
        "d_view": d_view,
        "bias_base": bias_base,
        "bias_view": bias_view,
        "z": z,
        "state_indices": state_indices,
        "state_indices_flat": state_indices_flat,
        "dst_indices": dst_indices,
        "dst_indices_flat": dst_indices_flat,
        "state_index_stride": state_index_stride,
        "dst_index_stride": dst_index_stride,
        "dummy_index": dummy_index,
        "seed": seed,
        "pad_slot_id": pad_slot_id,
    }
    return case


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    kwargs = case["kwargs"]
    spec = case["spec"]
    batch, nheads, dim = spec["BATCH"], spec["NHEADS"], spec["DIM"]
    ngroups, dstate = int(kwargs["ngroups"]), spec["DSTATE"]
    has_state_indices = bool(kwargs.get("has_state_indices", True))
    has_dst_indices = bool(kwargs.get("has_dst_indices", False))
    return (
        case["tirx_state_raw"],
        case["tirx_scale_raw"],
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
        case["scale_stride"] if spec["SCALE_STATE"] else 0,
        nheads * dim,
        nheads,
        ngroups * dstate,
        ngroups * dstate,
        nheads * dim,
        nheads * dim,
        case["state_index_stride"] if has_state_indices else 1,
        case["dst_index_stride"] if has_dst_indices else 0,
        nheads,
        ngroups,
        int(bool(kwargs.get("dt_softplus", False))),
        int(bool(kwargs.get("update_state", True))),
        case["pad_slot_id"],
        (dim + spec["ROWS_PER_BLOCK"] - 1) // spec["ROWS_PER_BLOCK"],
    )


def _run_reference(case: dict[str, Any]) -> torch.Tensor:
    kwargs = case["kwargs"]
    spec = case["spec"]
    oracle = _load_oracle()
    state_view = _view_state(case["reference_state_raw"], spec, case["state_stride"])
    state_scale = (
        _view_scale(case["reference_scale_raw"], spec, case["scale_stride"])
        if spec["SCALE_STATE"]
        else None
    )
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
        state_scale=state_scale,
        out=source_out,
        disable_state_update=not bool(kwargs.get("update_state", True)),
        rand_seed=case["seed"] if spec["PHILOX_ROUNDS"] else None,
        philox_rounds=spec["PHILOX_ROUNDS"],
        algorithm="simple",
    )
    if source_out is None:
        case["reference_output"].copy_(result)
    return result


def _written_slots(case: dict[str, Any]) -> list[int]:
    kwargs = case["kwargs"]
    batch = case["spec"]["BATCH"]
    if not bool(kwargs.get("update_state", True)):
        return []
    if bool(kwargs.get("has_state_indices", True)):
        read = case["state_indices"].reshape(batch, -1)[:, 0]
    else:
        read = torch.arange(batch, device=case["x"].device)
    if bool(kwargs.get("has_dst_indices", False)):
        dst = case["dst_indices"].reshape(batch, -1)[:, 0]
    else:
        dst = read
    valid = read != case["pad_slot_id"]
    return sorted({int(value) for value in dst[valid].tolist()})


def _assert_case_close(case: dict[str, Any]) -> None:
    kwargs = case["kwargs"]
    spec = case["spec"]
    for name, tensor in (
        ("TIRx output", case["tirx_output"]),
        ("FlashInfer output", case["reference_output"]),
    ):
        if not torch.isfinite(tensor.float()).all():
            raise AssertionError(f"{name} contains non-finite values")
    atol = 0.1 if spec["SCALE_STATE"] else 2e-2
    rtol = 1e-2 if spec["SCALE_STATE"] else 2e-2
    torch.testing.assert_close(case["tirx_output"], case["reference_output"], atol=atol, rtol=rtol)

    tirx_state = _view_state(case["tirx_state_raw"], spec, case["state_stride"])
    reference_state = _view_state(case["reference_state_raw"], spec, case["state_stride"])
    slots = _written_slots(case)
    if slots:
        slot_index = torch.tensor(slots, dtype=torch.int64, device=tirx_state.device)
        tirx_rows = tirx_state.index_select(0, slot_index)
        reference_rows = reference_state.index_select(0, slot_index)
        if spec["SCALE_STATE"]:
            tirx_scale = _view_scale(case["tirx_scale_raw"], spec, case["scale_stride"])
            reference_scale = _view_scale(case["reference_scale_raw"], spec, case["scale_stride"])
            tirx_scale_rows = tirx_scale.index_select(0, slot_index)
            reference_scale_rows = reference_scale.index_select(0, slot_index)
            torch.testing.assert_close(tirx_scale_rows, reference_scale_rows, atol=2e-5, rtol=2e-4)
            tirx_rows = tirx_rows.float() * tirx_scale_rows[..., None]
            reference_rows = reference_rows.float() * reference_scale_rows[..., None]
            torch.testing.assert_close(tirx_rows, reference_rows, atol=0.1, rtol=1e-2)
        else:
            state_atol = 2e-3 if spec["STATE_DTYPE"] == "float32" else 2e-2
            torch.testing.assert_close(tirx_rows, reference_rows, atol=state_atol, rtol=2e-2)
    elif not bool(kwargs.get("update_state", True)):
        torch.testing.assert_close(
            case["tirx_state_raw"], case["initial_state_raw"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            case["reference_state_raw"], case["initial_state_raw"], atol=0, rtol=0
        )


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
    _assert_case_close(case)


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

    def source_builder():
        executable(*args)
        _run_reference(case)
        torch.cuda.synchronize()
        _assert_case_close(case)
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
