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

import tirx_kernels.tirx_lite as txl

KERNEL_META = {
    "name": "selective_state_update_stp_simple",
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

_LOG2_E = 1.4426950408889634
_LN_2 = 0.6931471805599453
_FLT_LOWEST = -3.4028234663852886e38


def _lane_mask(raw_lane):
    """Preserve the shared lane-normalization leaf used by sibling STP ports."""
    return txl.cast(txl.bitwise_and(txl.cast(raw_lane, "uint32"), txl.uint32(31)), "int32")


def _shfl_down_f32(value, delta):
    """``shfl.sync.down.b32`` at width 32: clamp/segmask 31, full member mask.

    DPS: the destination pins the warp collective to the call site, so the
    shuffle is emitted once here rather than re-emitted at every textual use
    of the returned value.
    """
    shfl_down = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.down.b32(
        shfl_down,
        txl.reinterpret("uint32", value),
        txl.cast(delta, "uint32"),
        txl.uint32(31),
        txl.uint32(0xFFFFFFFF),
    )
    return txl.reinterpret("float32", shfl_down)


def _shfl_idx_f32(value, source_lane):
    """``shfl.sync.idx.b32`` at width 32: clamp/segmask 31, full member mask."""
    shfl_idx = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.idx.b32(
        shfl_idx,
        txl.reinterpret("uint32", value),
        txl.cast(source_lane, "uint32"),
        txl.uint32(31),
        txl.uint32(0xFFFFFFFF),
    )
    return txl.reinterpret("float32", shfl_idx)


def _global_load_index_s64(buffer, index, dtype):
    if dtype == "int32":
        gload_0 = txl.local_scalar("int32")
        txl.ptx.ld.global_.s32(gload_0, buffer.ptr_to([index]))
        return txl.cast(gload_0, "int64")
    gload_1 = txl.local_scalar("int64")
    txl.ptx.ld.global_.s64(gload_1, buffer.ptr_to([index]))
    return gload_1


def _load_weight(buffer, index, dtype: str):
    if dtype == "float32":
        gload_2 = txl.local_scalar("uint32")
        txl.ptx.ld.global_.b32(gload_2, buffer.ptr_to([index]))
        return txl.reinterpret("float32", gload_2)
    gload_3 = txl.local_scalar("uint16")
    txl.ptx.ld.global_.b16(gload_3, buffer.ptr_to([index]))
    bf16_f32_0 = txl.local_scalar("float32")
    txl.ptx.cvt.f32.bf16(bf16_f32_0, txl.cast(gload_3, "uint16"))
    return bf16_f32_0


def _state_bits_to_f32(bits, dtype: str):
    if dtype == "bfloat16":
        bf16_f32_1 = txl.local_scalar("float32")
        txl.ptx.cvt.f32.bf16(bf16_f32_1, txl.cast(bits, "uint16"))
        return bf16_f32_1
    if dtype == "float16":
        f16_f32_0 = txl.local_scalar("float32")
        txl.ptx.cvt.f32.f16(f16_f32_0, txl.cast(bits, "uint16"))
        return f16_f32_0
    if dtype == "int16":
        i16_f32_0 = txl.local_scalar("float32")
        txl.ptx.cvt.rn.f32.s16(i16_f32_0, txl.reinterpret("int16", txl.cast(bits, "uint16")))
        return i16_f32_0
    return txl.reinterpret("float32", txl.cast(bits, "uint32"))


def _f32_to_state_bits(value, dtype: str):
    if dtype == "bfloat16":
        f32_bf16_0 = txl.local_scalar("uint16")
        txl.ptx.cvt.rn.bf16.f32(f32_bf16_0, value)
        return f32_bf16_0
    if dtype == "float16":
        f32_f16_0 = txl.local_scalar("uint16")
        txl.ptx.cvt.rn.f16.f32(f32_f16_0, value)
        return f32_f16_0
    return txl.reinterpret("uint32", value)


def _load_two_byte_vector(buffer, index, count: int, scope: str):
    bits = txl.alloc_local((count,), "uint16")
    prefix = f"ld.{scope}"
    if count == 2:
        txl.ptx[f"{prefix}.v2.b16"](bits[0], bits[1], buffer.ptr_to([index]))
    elif count == 3:
        for e in range(3):
            txl.ptx[f"{prefix}.b16"](bits[e], buffer.ptr_to([index + e]))
    elif count == 4:
        txl.ptx[f"{prefix}.v4.b16"](bits[0], bits[1], bits[2], bits[3], buffer.ptr_to([index]))
    else:
        words = txl.alloc_local((4,), "uint32")
        txl.ptx[f"{prefix}.v4.b32"](words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
        for pair in range(4):
            txl.buffer_store(
                bits,
                txl.cast(txl.bitwise_and(words[pair], txl.uint32(0xFFFF)), "uint16"),
                [2 * pair],
            )
            txl.buffer_store(
                bits,
                txl.cast(txl.shift_right(words[pair], txl.uint32(16)), "uint16"),
                [2 * pair + 1],
            )
    return bits


def _store_two_byte_vector(buffer, index, bits, count: int, scope: str = "global_"):
    prefix = f"st.{scope}"
    if count == 2:
        txl.ptx[f"{prefix}.v2.b16"](buffer.ptr_to([index]), bits[0], bits[1])
    elif count == 3:
        for e in range(3):
            txl.ptx[f"{prefix}.b16"](buffer.ptr_to([index + e]), bits[e])
    elif count == 4:
        txl.ptx[f"{prefix}.v4.b16"](buffer.ptr_to([index]), bits[0], bits[1], bits[2], bits[3])
    else:
        words = txl.alloc_local((4,), "uint32")
        for pair in range(4):
            txl.buffer_store(
                words,
                txl.bitwise_or(
                    txl.cast(bits[2 * pair], "uint32"),
                    txl.shift_left(txl.cast(bits[2 * pair + 1], "uint32"), txl.uint32(16)),
                ),
                [pair],
            )
        txl.ptx[f"{prefix}.v4.b32"](buffer.ptr_to([index]), words[0], words[1], words[2], words[3])


def _load_f32_vector(buffer, index, count: int):
    words = txl.alloc_local((count,), "uint32")
    if count == 2:
        txl.ptx.ld.global_.v2.b32(words[0], words[1], buffer.ptr_to([index]))
    elif count == 3:
        for e in range(3):
            txl.ptx.ld.global_.b32(words[e], buffer.ptr_to([index + e]))
    else:
        txl.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
    return words


def _store_f32_vector(buffer, index, words, count: int):
    if count == 2:
        txl.ptx.st.global_.v2.b32(buffer.ptr_to([index]), words[0], words[1])
    elif count == 3:
        for e in range(3):
            txl.ptx.st.global_.b32(buffer.ptr_to([index + e]), words[e])
    else:
        txl.ptx.st.global_.v4.b32(buffer.ptr_to([index]), words[0], words[1], words[2], words[3])


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

    @txl.kernel(warps=4, arch="sm_100a", grid=(spec["BATCH"], spec["NHEADS"], "dim_tiles_runtime"))
    def selective_state_update_stp_simple(
        state: txl.gptr[spec["STATE_DTYPE"]],
        state_scale: txl.gptr[txl.f32],
        x: txl.gptr[txl.bf16],
        dt: txl.gptr[spec["WEIGHT_DTYPE"]],
        matrix_a: txl.gptr[txl.f32],
        matrix_b: txl.gptr[txl.bf16],
        matrix_c: txl.gptr[txl.bf16],
        d_weight: txl.gptr[spec["WEIGHT_DTYPE"]],
        z: txl.gptr[txl.bf16],
        dt_bias: txl.gptr[spec["WEIGHT_DTYPE"]],
        state_indices: txl.gptr[spec["INDEX_DTYPE"]],
        dst_indices: txl.gptr[spec["INDEX_DTYPE"]],
        rand_seed: txl.gptr[txl.i64],
        output: txl.gptr[txl.bf16],
        state_stride_batch: txl.i64,
        state_scale_stride_batch: txl.i64,
        x_stride_batch: txl.i64,
        dt_stride_batch: txl.i64,
        b_stride_batch: txl.i64,
        c_stride_batch: txl.i64,
        z_stride_batch: txl.i64,
        out_stride_batch: txl.i64,
        state_indices_stride_batch: txl.i64,
        dst_indices_stride_batch: txl.i64,
        nheads_runtime: txl.i32,
        ngroups_runtime: txl.i32,
        dt_softplus: txl.i32,
        update_state: txl.i32,
        pad_slot_id: txl.i32,
        dim_tiles_runtime: txl.i32,
    ):
        batch_i, head, dim_tile = txl.cta_id()
        smem = txl.smem_pool()
        s_x = smem.alloc((spec["ROWS_PER_BLOCK"],), txl.bf16, align=16)
        s_z = smem.alloc((spec["ROWS_PER_BLOCK"],), txl.bf16, align=16)
        s_b = smem.alloc((spec["DSTATE"],), txl.bf16, align=16)
        s_c = smem.alloc((spec["DSTATE"],), txl.bf16, align=16)
        s_out = smem.alloc((spec["ROWS_PER_BLOCK"],), txl.f32, align=4)
        s_scale = (
            smem.alloc((spec["ROWS_PER_BLOCK"],), txl.f32, align=16)
            if spec["SCALE_STATE"]
            else s_out
        )
        roles = txl.specialize()
        load_x = roles.role("load_x_and_scale", warps=[0])
        load_b = roles.role("load_b", warps=[1])
        load_z = roles.role("load_z", warps=[2])
        load_c = roles.role("load_c", warps=[3])

        lane_ctx = txl.local_scalar(txl.i32)
        dim_offset_ctx = txl.local_scalar(txl.i32)
        txl.assign(lane_ctx, txl.lane_id())
        txl.assign(dim_offset_ctx, dim_tile * ROWS_PER_BLOCK)
        lane = lane_ctx
        warp = txl.warp_id()
        dim_offset = dim_offset_ctx
        rows_per_warp = (ROWS_PER_BLOCK + 3) // 4
        group_ctx = txl.local_scalar(txl.i32)
        random_seed_ctx = txl.local_scalar(txl.i64)
        state_batch_ctx = txl.local_scalar(txl.i64)
        state_head_offset_ctx = txl.local_scalar(txl.i64)
        dst_state_head_offset_ctx = txl.local_scalar(txl.i64)
        scale_head_offset_ctx = txl.local_scalar(txl.i64)
        dst_scale_head_offset_ctx = txl.local_scalar(txl.i64)
        dt_value_ctx = txl.local_scalar(txl.f32)
        da_value_ctx = txl.local_scalar(txl.f32)
        d_value_ctx = txl.local_scalar(txl.f32)

        def prepare_cta():
            # TIRX_TRANSCRIBE_START selective_state_update_stp_simple

            random_seed = txl.local_scalar("int64", init=0)
            if PHILOX_ROUNDS > 0 and not SCALE_STATE:
                txl.ptx.ld.global_.s64(random_seed, rand_seed.ptr_to([0]))

            state_batch = txl.local_scalar("int64")
            if HAS_STATE_INDICES:
                txl.assign(
                    state_batch,
                    _global_load_index_s64(
                        state_indices, batch_i * state_indices_stride_batch, INDEX_DTYPE
                    ),
                )
            else:
                txl.assign(state_batch, txl.cast(batch_i, "int64"))
            dst_state_batch = txl.local_scalar("int64")
            if HAS_DST_INDICES:
                txl.assign(
                    dst_state_batch,
                    _global_load_index_s64(
                        dst_indices, batch_i * dst_indices_stride_batch, INDEX_DTYPE
                    ),
                )
            else:
                txl.assign(dst_state_batch, state_batch)

            state_head_offset: txl.int64 = state_batch * state_stride_batch + txl.cast(
                head * DIM * DSTATE, "int64"
            )
            dst_state_head_offset: txl.int64 = dst_state_batch * state_stride_batch + txl.cast(
                head * DIM * DSTATE, "int64"
            )
            scale_head_offset: txl.int64 = state_batch * state_scale_stride_batch + txl.cast(
                head * DIM, "int64"
            )
            dst_scale_head_offset: txl.int64 = (
                dst_state_batch * state_scale_stride_batch + txl.cast(head * DIM, "int64")
            )

            gload_4 = txl.local_scalar("uint32")
            txl.ptx.ld.global_.b32(gload_4, matrix_a.ptr_to([head]))
            a_value: txl.float32 = txl.reinterpret("float32", gload_4)
            dt_value = txl.local_scalar("float32")
            txl.assign(
                dt_value,
                _load_weight(dt, txl.cast(batch_i, "int64") * dt_stride_batch + head, WEIGHT_DTYPE),
            )
            if HAS_DT_BIAS:
                txl.ptx["add.ftz.f32"](
                    dt_value, dt_value, _load_weight(dt_bias, head, WEIGHT_DTYPE)
                )
            with txl.If(dt_softplus != 0), txl.Then():
                with txl.If(dt_value <= txl.float32(20.0)), txl.Then():
                    mul_0 = txl.local_scalar("float32")
                    txl.ptx["mul.ftz.f32"](mul_0, dt_value, txl.float32(_LOG2_E))
                    exp2_0 = txl.local_scalar("float32")
                    txl.ptx["ex2.approx.ftz.f32"](exp2_0, mul_0)
                    softplus_exp: txl.float32 = exp2_0
                    add_0 = txl.local_scalar("float32")
                    txl.ptx["add.ftz.f32"](add_0, txl.float32(1.0), softplus_exp)
                    log2_0 = txl.local_scalar("float32")
                    txl.ptx["lg2.approx.ftz.f32"](log2_0, add_0)
                    txl.ptx["mul.ftz.f32"](dt_value, log2_0, txl.float32(_LN_2))
            mul_1 = txl.local_scalar("float32")
            txl.ptx["mul.ftz.f32"](mul_1, a_value, dt_value)
            mul_2 = txl.local_scalar("float32")
            txl.ptx["mul.ftz.f32"](mul_2, mul_1, txl.float32(_LOG2_E))
            exp2_1 = txl.local_scalar("float32")
            txl.ptx["ex2.approx.ftz.f32"](exp2_1, mul_2)
            da_value: txl.float32 = exp2_1
            d_value = txl.local_scalar("float32", init=0.0)
            if HAS_D:
                txl.assign(d_value, _load_weight(d_weight, head, WEIGHT_DTYPE))

            txl.assign(group_ctx, head // (nheads_runtime // ngroups_runtime))
            txl.assign(random_seed_ctx, random_seed)
            txl.assign(state_batch_ctx, state_batch)
            txl.assign(state_head_offset_ctx, state_head_offset)
            txl.assign(dst_state_head_offset_ctx, dst_state_head_offset)
            txl.assign(scale_head_offset_ctx, scale_head_offset)
            txl.assign(dst_scale_head_offset_ctx, dst_scale_head_offset)
            txl.assign(dt_value_ctx, dt_value)
            txl.assign(da_value_ctx, da_value)
            txl.assign(d_value_ctx, d_value)

        def load_x_and_scale():
            scale_head_offset: txl.int64 = scale_head_offset_ctx
            with txl.serial((ROWS_PER_BLOCK + 31) // 32) as preload_iter:
                local_row: txl.int32 = lane + preload_iter * 32
                row_d: txl.int32 = dim_offset + local_row
                with txl.If(txl.And(local_row < ROWS_PER_BLOCK, row_d < DIM)), txl.Then():
                    gload_5 = txl.local_scalar("uint16")
                    txl.ptx.ld.global_.b16(
                        gload_5,
                        x.ptr_to(
                            [txl.cast(batch_i, "int64") * x_stride_batch + head * DIM + row_d]
                        ),
                    )
                    x_bits: txl.uint16 = gload_5
                    txl.ptx.st.shared.b16(s_x.ptr_to([local_row]), x_bits)
            if SCALE_STATE:
                with txl.serial((ROWS_PER_BLOCK + 31) // 32) as scale_iter:
                    local_row: txl.int32 = lane + scale_iter * 32
                    row_d: txl.int32 = dim_offset + local_row
                    with txl.If(txl.And(local_row < ROWS_PER_BLOCK, row_d < DIM)), txl.Then():
                        gload_6 = txl.local_scalar("uint32")
                        txl.ptx.ld.global_.b32(
                            gload_6, state_scale.ptr_to([scale_head_offset + row_d])
                        )
                        scale_bits: txl.uint32 = gload_6
                        txl.ptx.st.shared.b32(s_scale.ptr_to([local_row]), scale_bits)

        def load_bc_values(s_dst, src, src_stride):
            group: txl.int32 = group_ctx
            bc_i: txl.int32 = lane * 8
            with txl.If(bc_i < DSTATE), txl.Then():
                bc_words = txl.alloc_local((4,), "uint32")
                txl.ptx.ld.global_.v4.b32(
                    bc_words[0],
                    bc_words[1],
                    bc_words[2],
                    bc_words[3],
                    src.ptr_to([txl.cast(batch_i, "int64") * src_stride + group * DSTATE + bc_i]),
                )
                txl.ptx.st.shared.v4.b32(
                    s_dst.ptr_to([bc_i]), bc_words[0], bc_words[1], bc_words[2], bc_words[3]
                )

        def load_z_values():
            with txl.serial((ROWS_PER_BLOCK + 31) // 32) as preload_iter:
                local_row: txl.int32 = lane + preload_iter * 32
                row_d: txl.int32 = dim_offset + local_row
                with txl.If(txl.And(local_row < ROWS_PER_BLOCK, row_d < DIM)), txl.Then():
                    if HAS_Z:
                        gload_7 = txl.local_scalar("uint16")
                        txl.ptx.ld.global_.b16(
                            gload_7,
                            z.ptr_to(
                                [txl.cast(batch_i, "int64") * z_stride_batch + head * DIM + row_d]
                            ),
                        )
                        z_bits: txl.uint16 = gload_7
                        txl.ptx.st.shared.b16(s_z.ptr_to([local_row]), z_bits)
                    else:
                        txl.ptx.st.shared.b16(s_z.ptr_to([local_row]), txl.uint16(0))

        def update_rows():
            random_seed: txl.int64 = random_seed_ctx
            state_batch: txl.int64 = state_batch_ctx
            state_head_offset: txl.int64 = state_head_offset_ctx
            dst_state_head_offset: txl.int64 = dst_state_head_offset_ctx
            dt_value: txl.float32 = dt_value_ctx
            da_value: txl.float32 = da_value_ctx
            d_value: txl.float32 = d_value_ctx

            with txl.serial(rows_per_warp) as row_in_warp:
                local_row_ctx = txl.local_scalar("int32", init=warp * rows_per_warp + row_in_warp)
                local_row: txl.int32 = local_row_ctx
                row_d_ctx = txl.local_scalar("int32", init=dim_offset + local_row)
                row_d: txl.int32 = row_d_ctx
                with txl.If(row_d < DIM), txl.Then():
                    sload_0 = txl.local_scalar("uint16")
                    txl.ptx.ld.shared.b16(sload_0, s_x.ptr_to([local_row]))
                    bf16_f32_2 = txl.local_scalar("float32")
                    txl.ptx.cvt.f32.bf16(bf16_f32_2, txl.cast(sload_0, "uint16"))
                    x_value: txl.float32 = bf16_f32_2
                    decode_scale = txl.local_scalar("float32", init=1.0)
                    new_state_max = txl.local_scalar("float32", init=txl.float32(_FLT_LOWEST))
                    if SCALE_STATE:
                        sload_1 = txl.local_scalar("uint32")
                        txl.ptx.ld.shared.b32(sload_1, s_scale.ptr_to([local_row]))
                        txl.assign(decode_scale, txl.reinterpret("float32", sload_1))
                    mul_3 = txl.local_scalar("float32")
                    txl.ptx["mul.ftz.f32"](mul_3, d_value, x_value)
                    d_times_x: txl.float32 = mul_3
                    out_value = txl.local_scalar(
                        "float32", init=txl.if_then_else(lane == 0, d_times_x, txl.float32(0.0))
                    )
                    new_states = txl.alloc_local((LANE_STATE_COUNT,), "float32")
                    with txl.unroll(STATE_ITERATIONS) as state_iter:
                        state_i_ctx = txl.local_scalar(
                            "int32", init=(state_iter * 32 + lane) * STATE_VECTOR
                        )
                        state_i: txl.int32 = state_i_ctx
                        if STATE_BYTES == 2:
                            r_state = txl.alloc_local((STATE_VECTOR,), "uint16")
                            with txl.unroll(STATE_VECTOR) as e:
                                txl.ptx.mov.b16(r_state[e], txl.uint16(0))
                            with txl.If(state_batch != txl.cast(pad_slot_id, "int64")), txl.Then():
                                loaded_state = _load_two_byte_vector(
                                    state,
                                    state_head_offset + row_d * DSTATE + state_i,
                                    STATE_VECTOR,
                                    "global",
                                )
                                with txl.unroll(STATE_VECTOR) as e:
                                    txl.ptx.mov.b16(r_state[e], loaded_state[e])
                        else:
                            r_state = txl.alloc_local((STATE_VECTOR,), "uint32")
                            with txl.unroll(STATE_VECTOR) as e:
                                txl.ptx.mov.b32(r_state[e], txl.uint32(0))
                            with txl.If(state_batch != txl.cast(pad_slot_id, "int64")), txl.Then():
                                loaded_state = _load_f32_vector(
                                    state,
                                    state_head_offset + row_d * DSTATE + state_i,
                                    STATE_VECTOR,
                                )
                                with txl.unroll(STATE_VECTOR) as e:
                                    txl.ptx.mov.b32(r_state[e], loaded_state[e])

                        b_bits = txl.alloc_local((STATE_VECTOR,), "uint16")
                        c_bits = txl.alloc_local((STATE_VECTOR,), "uint16")
                        random_words = txl.alloc_local((4,), "uint32")
                        sr_raw = txl.alloc_local((STATE_VECTOR,), "uint32")
                        with txl.unroll(STATE_VECTOR) as e:
                            with (
                                txl.If(
                                    txl.And(
                                        txl.And(PHILOX_ROUNDS > 0, txl.Not(SCALE_STATE)), e % 4 == 0
                                    )
                                ),
                                txl.Then(),
                            ):
                                random_offset: txl.uint64 = txl.cast(
                                    state_head_offset + row_d * DSTATE + state_i + e, "uint64"
                                )
                                c0 = txl.local_scalar(
                                    "uint32", init=txl.cast(random_offset, "uint32")
                                )
                                c1 = txl.local_scalar(
                                    "uint32",
                                    init=txl.cast(
                                        txl.shift_right(random_offset, txl.uint64(32)), "uint32"
                                    ),
                                )
                                c2 = txl.local_scalar("uint32", init=0)
                                c3 = txl.local_scalar("uint32", init=0)
                                k0 = txl.local_scalar(
                                    "uint32",
                                    init=txl.cast(txl.reinterpret("uint64", random_seed), "uint32"),
                                )
                                k1 = txl.local_scalar(
                                    "uint32",
                                    init=txl.cast(
                                        txl.shift_right(
                                            txl.reinterpret("uint64", random_seed), txl.uint64(32)
                                        ),
                                        "uint32",
                                    ),
                                )
                                with txl.unroll(10) as philox_round:
                                    old_c0: txl.uint32 = c0
                                    old_c2: txl.uint32 = c2
                                    mul_hi_0 = txl.local_scalar("uint32")
                                    txl.ptx["mul.hi.u32"](mul_hi_0, txl.uint32(0xCD9E8D57), old_c2)
                                    hi_b: txl.uint32 = mul_hi_0
                                    next_c0: txl.uint32 = txl.bitwise_xor(
                                        txl.bitwise_xor(hi_b, c1), k0
                                    )
                                    mul_hi_1 = txl.local_scalar("uint32")
                                    txl.ptx["mul.hi.u32"](mul_hi_1, txl.uint32(0xD2511F53), old_c0)
                                    hi_a: txl.uint32 = mul_hi_1
                                    next_c2: txl.uint32 = txl.bitwise_xor(
                                        txl.bitwise_xor(hi_a, c3), k1
                                    )
                                    mul_lo_0 = txl.local_scalar("int32")
                                    txl.ptx["mul.lo.s32"](
                                        mul_lo_0,
                                        txl.int32(-845247145),
                                        txl.reinterpret("int32", old_c2),
                                    )
                                    next_c1_s: txl.int32 = mul_lo_0
                                    mul_lo_1 = txl.local_scalar("int32")
                                    txl.ptx["mul.lo.s32"](
                                        mul_lo_1,
                                        txl.int32(-766435501),
                                        txl.reinterpret("int32", old_c0),
                                    )
                                    next_c3_s: txl.int32 = mul_lo_1
                                    add_s32_0 = txl.local_scalar("int32")
                                    txl.ptx["add.s32"](
                                        add_s32_0,
                                        txl.reinterpret("int32", k0),
                                        txl.int32(-1640531527),
                                    )
                                    next_k0_s: txl.int32 = add_s32_0
                                    add_s32_1 = txl.local_scalar("int32")
                                    txl.ptx["add.s32"](
                                        add_s32_1,
                                        txl.reinterpret("int32", k1),
                                        txl.int32(-1150833019),
                                    )
                                    next_k1_s: txl.int32 = add_s32_1
                                    txl.assign(c0, next_c0)
                                    txl.assign(c1, txl.reinterpret("uint32", next_c1_s))
                                    txl.assign(c2, next_c2)
                                    txl.assign(c3, txl.reinterpret("uint32", next_c3_s))
                                    txl.assign(k0, txl.reinterpret("uint32", next_k0_s))
                                    txl.assign(k1, txl.reinterpret("uint32", next_k1_s))
                                txl.ptx.mov.b32(random_words[0], c0)
                                txl.ptx.mov.b32(random_words[1], c1)
                                txl.ptx.mov.b32(random_words[2], c2)
                                txl.ptx.mov.b32(random_words[3], c3)

                            state_value = txl.local_scalar("float32")
                            txl.assign(state_value, _state_bits_to_f32(r_state[e], STATE_DTYPE))
                            if SCALE_STATE:
                                txl.ptx["mul.ftz.f32"](state_value, state_value, decode_scale)
                            if STATE_VECTOR == 3:
                                txl.ptx.ld.shared.b16(b_bits[e], s_b.ptr_to([state_i + e]))
                                bf16_f32_3 = txl.local_scalar("float32")
                                txl.ptx.cvt.f32.bf16(bf16_f32_3, txl.cast(b_bits[e], "uint16"))
                                b_value: txl.float32 = bf16_f32_3
                                txl.ptx.ld.shared.b16(c_bits[e], s_c.ptr_to([state_i + e]))
                                bf16_f32_4 = txl.local_scalar("float32")
                                txl.ptx.cvt.f32.bf16(bf16_f32_4, txl.cast(c_bits[e], "uint16"))
                                c_value: txl.float32 = bf16_f32_4
                            else:
                                with txl.If(e == 0), txl.Then():
                                    loaded_b = _load_two_byte_vector(
                                        s_b, state_i, STATE_VECTOR, "shared"
                                    )
                                    with txl.unroll(STATE_VECTOR) as copy_e:
                                        txl.ptx.mov.b16(b_bits[copy_e], loaded_b[copy_e])
                                bf16_f32_5 = txl.local_scalar("float32")
                                txl.ptx.cvt.f32.bf16(bf16_f32_5, txl.cast(b_bits[e], "uint16"))
                                b_value: txl.float32 = bf16_f32_5
                                with txl.If(e == 0), txl.Then():
                                    loaded_c = _load_two_byte_vector(
                                        s_c, state_i, STATE_VECTOR, "shared"
                                    )
                                    with txl.unroll(STATE_VECTOR) as copy_e:
                                        txl.ptx.mov.b16(c_bits[copy_e], loaded_c[copy_e])
                                bf16_f32_6 = txl.local_scalar("float32")
                                txl.ptx.cvt.f32.bf16(bf16_f32_6, txl.cast(c_bits[e], "uint16"))
                                c_value: txl.float32 = bf16_f32_6

                            mul_4 = txl.local_scalar("float32")
                            txl.ptx["mul.ftz.f32"](mul_4, b_value, dt_value)
                            db_value: txl.float32 = mul_4
                            mul_5 = txl.local_scalar("float32")
                            txl.ptx["mul.ftz.f32"](mul_5, db_value, x_value)
                            db_x: txl.float32 = mul_5
                            fma_0 = txl.local_scalar("float32")
                            txl.ptx["fma.rn.ftz.f32"](fma_0, state_value, da_value, db_x)
                            new_state: txl.float32 = fma_0
                            if SCALE_STATE:
                                abs_0 = txl.local_scalar("float32")
                                txl.ptx["abs.ftz.f32"](abs_0, new_state)
                                magnitude: txl.float32 = abs_0
                                txl.ptx["max.ftz.f32"](new_state_max, new_state_max, magnitude)
                                txl.ptx.mov.b32(
                                    new_states[state_iter * STATE_VECTOR + e], new_state
                                )
                            elif PHILOX_ROUNDS > 0:
                                random13: txl.uint32 = txl.bitwise_and(
                                    random_words[e % 4], txl.uint32(0x1FFF)
                                )
                                txl.ptx.cvt.rs.f16x2.f32(
                                    sr_raw[e], txl.float32(0.0), new_state, random13
                                )
                            elif STATE_BYTES == 2:
                                txl.ptx.mov.b16(
                                    r_state[e], _f32_to_state_bits(new_state, STATE_DTYPE)
                                )
                            else:
                                txl.ptx.mov.b32(
                                    r_state[e], _f32_to_state_bits(new_state, STATE_DTYPE)
                                )
                            txl.ptx["fma.rn.ftz.f32"](out_value, new_state, c_value, out_value)

                        with (
                            txl.If(
                                txl.And(
                                    txl.And(txl.Not(SCALE_STATE), update_state != 0),
                                    state_batch != txl.cast(pad_slot_id, "int64"),
                                )
                            ),
                            txl.Then(),
                        ):
                            if PHILOX_ROUNDS > 0:
                                sr_words = txl.alloc_local((STATE_VECTOR // 2,), "uint32")
                                with txl.unroll(STATE_VECTOR // 2) as pair:
                                    prmt_0 = txl.local_scalar("uint32")
                                    txl.ptx["prmt.b32"](
                                        prmt_0,
                                        txl.cast(sr_raw[2 * pair], "uint32"),
                                        txl.cast(sr_raw[2 * pair + 1], "uint32"),
                                        txl.uint32(0x5410),
                                    )
                                    txl.ptx.mov.b32(sr_words[pair], prmt_0)
                                if STATE_VECTOR == 2:
                                    txl.ptx.st.global_.b32(
                                        state.ptr_to(
                                            [dst_state_head_offset + row_d * DSTATE + state_i]
                                        ),
                                        sr_words[0],
                                    )
                                else:
                                    txl.ptx.st.global_.v2.b32(
                                        state.ptr_to(
                                            [dst_state_head_offset + row_d * DSTATE + state_i]
                                        ),
                                        sr_words[0],
                                        sr_words[1],
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

                    with txl.unroll(5) as delta_i:
                        delta: txl.int32 = txl.shift_right(txl.int32(16), delta_i)
                        txl.ptx["add.ftz.f32"](
                            out_value, out_value, _shfl_down_f32(out_value, delta)
                        )
                    with txl.If(lane == 0), txl.Then():
                        txl.ptx.st.shared.b32(
                            s_out.ptr_to([local_row]), txl.reinterpret("uint32", out_value)
                        )

                    with (
                        txl.If(
                            txl.And(
                                txl.And(SCALE_STATE, update_state != 0),
                                state_batch != txl.cast(pad_slot_id, "int64"),
                            )
                        ),
                        txl.Then(),
                    ):
                        with txl.unroll(5) as delta_i:
                            delta: txl.int32 = txl.shift_right(txl.int32(16), delta_i)
                            txl.ptx["max.ftz.f32"](
                                new_state_max, new_state_max, _shfl_down_f32(new_state_max, delta)
                            )
                        txl.cuda.warp_sync()
                        txl.assign(new_state_max, _shfl_idx_f32(new_state_max, txl.int32(0)))
                        encode_scale = txl.local_scalar("float32", init=1.0)
                        with txl.If(new_state_max != txl.float32(0.0)), txl.Then():
                            txl.ptx["div.approx.ftz.f32"](
                                encode_scale, txl.float32(32767.0), new_state_max
                            )
                        rcp_0 = txl.local_scalar("float32")
                        txl.ptx["rcp.approx.ftz.f32"](rcp_0, encode_scale)
                        new_decode_scale: txl.float32 = rcp_0
                        with txl.unroll(STATE_ITERATIONS) as state_iter:
                            state_i: txl.int32 = (state_iter * 32 + lane) * STATE_VECTOR
                            quantized = txl.alloc_local((STATE_VECTOR,), "int32")
                            packed_quantized = txl.alloc_local((STATE_VECTOR // 2,), "uint32")
                            with txl.unroll(STATE_VECTOR) as e:
                                mul_6 = txl.local_scalar("float32")
                                txl.ptx["mul.ftz.f32"](
                                    mul_6, new_states[state_iter * STATE_VECTOR + e], encode_scale
                                )
                                scaled: txl.float32 = mul_6
                                max_0 = txl.local_scalar("float32")
                                txl.ptx["max.ftz.f32"](max_0, scaled, txl.float32(-32767.0))
                                clipped_low: txl.float32 = max_0
                                min_0 = txl.local_scalar("float32")
                                txl.ptx["min.ftz.f32"](min_0, clipped_low, txl.float32(32767.0))
                                clipped: txl.float32 = min_0
                                txl.ptx.cvt.rni.ftz.s32.f32(quantized[e], clipped)
                            with txl.unroll(STATE_VECTOR // 2) as pair:
                                prmt_1 = txl.local_scalar("uint32")
                                txl.ptx["prmt.b32"](
                                    prmt_1,
                                    txl.cast(
                                        txl.reinterpret("uint32", quantized[2 * pair]), "uint32"
                                    ),
                                    txl.cast(
                                        txl.reinterpret("uint32", quantized[2 * pair + 1]), "uint32"
                                    ),
                                    txl.uint32(0x5410),
                                )
                                txl.ptx.mov.b32(packed_quantized[pair], prmt_1)
                            if STATE_VECTOR == 2:
                                txl.ptx.st.global_.b32(
                                    state.ptr_to(
                                        [dst_state_head_offset + row_d * DSTATE + state_i]
                                    ),
                                    packed_quantized[0],
                                )
                            elif STATE_VECTOR == 4:
                                txl.ptx.st.global_.v2.b32(
                                    state.ptr_to(
                                        [dst_state_head_offset + row_d * DSTATE + state_i]
                                    ),
                                    packed_quantized[0],
                                    packed_quantized[1],
                                )
                            else:
                                txl.ptx.st.global_.v4.b32(
                                    state.ptr_to(
                                        [dst_state_head_offset + row_d * DSTATE + state_i]
                                    ),
                                    packed_quantized[0],
                                    packed_quantized[1],
                                    packed_quantized[2],
                                    packed_quantized[3],
                                )
                        with txl.If(lane == 0), txl.Then():
                            txl.ptx.st.shared.b32(
                                s_scale.ptr_to([local_row]),
                                txl.reinterpret("uint32", new_decode_scale),
                            )

        def store_outputs():
            state_batch: txl.int64 = state_batch_ctx
            dst_scale_head_offset: txl.int64 = dst_scale_head_offset_ctx
            with txl.serial((ROWS_PER_BLOCK + 127) // 128) as output_iter:
                row_in_warp: txl.int32 = lane + output_iter * 32
                local_row: txl.int32 = warp * rows_per_warp + row_in_warp
                row_d: txl.int32 = dim_offset + local_row
                with txl.If(txl.And(row_in_warp < rows_per_warp, row_d < DIM)), txl.Then():
                    out_value = txl.local_scalar("float32")
                    sload_2 = txl.local_scalar("uint32")
                    txl.ptx.ld.shared.b32(sload_2, s_out.ptr_to([local_row]))
                    txl.assign(out_value, txl.reinterpret("float32", sload_2))
                    if HAS_Z:
                        sload_3 = txl.local_scalar("uint16")
                        txl.ptx.ld.shared.b16(sload_3, s_z.ptr_to([local_row]))
                        bf16_f32_7 = txl.local_scalar("float32")
                        txl.ptx.cvt.f32.bf16(bf16_f32_7, txl.cast(sload_3, "uint16"))
                        z_value: txl.float32 = bf16_f32_7
                        sub_0 = txl.local_scalar("float32")
                        txl.ptx["sub.ftz.f32"](sub_0, txl.float32(0.0), z_value)
                        neg_z: txl.float32 = sub_0
                        mul_7 = txl.local_scalar("float32")
                        txl.ptx["mul.ftz.f32"](mul_7, neg_z, txl.float32(_LOG2_E))
                        z_exp_arg: txl.float32 = mul_7
                        exp2_2 = txl.local_scalar("float32")
                        txl.ptx["ex2.approx.ftz.f32"](exp2_2, z_exp_arg)
                        exp_neg_z: txl.float32 = exp2_2
                        add_1 = txl.local_scalar("float32")
                        txl.ptx["add.ftz.f32"](add_1, txl.float32(1.0), exp_neg_z)
                        denominator: txl.float32 = add_1
                        div_0 = txl.local_scalar("float32")
                        txl.ptx["div.approx.ftz.f32"](div_0, txl.float32(1.0), denominator)
                        sigmoid_z: txl.float32 = div_0
                        mul_8 = txl.local_scalar("float32")
                        txl.ptx["mul.ftz.f32"](mul_8, z_value, sigmoid_z)
                        silu_z: txl.float32 = mul_8
                        txl.ptx["mul.ftz.f32"](out_value, out_value, silu_z)
                    f32_bf16_1 = txl.local_scalar("uint16")
                    txl.ptx.cvt.rn.bf16.f32(f32_bf16_1, out_value)
                    output_bits: txl.uint16 = f32_bf16_1
                    txl.ptx.st.global_.b16(
                        output.ptr_to(
                            [txl.cast(batch_i, "int64") * out_stride_batch + head * DIM + row_d]
                        ),
                        output_bits,
                    )
            with (
                txl.If(
                    txl.And(
                        txl.And(SCALE_STATE, update_state != 0),
                        state_batch != txl.cast(pad_slot_id, "int64"),
                    )
                ),
                txl.Then(),
            ):
                with txl.serial((ROWS_PER_BLOCK + 127) // 128) as scale_iter:
                    row_in_warp: txl.int32 = lane + scale_iter * 32
                    local_row: txl.int32 = warp * rows_per_warp + row_in_warp
                    row_d: txl.int32 = dim_offset + local_row
                    with txl.If(txl.And(row_in_warp < rows_per_warp, row_d < DIM)), txl.Then():
                        sload_4 = txl.local_scalar("uint32")
                        txl.ptx.ld.shared.b32(sload_4, s_scale.ptr_to([local_row]))
                        scale_bits: txl.uint32 = sload_4
                        txl.ptx.st.global_.b32(
                            state_scale.ptr_to([dst_scale_head_offset + row_d]), scale_bits
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
        txl.cuda.cta_sync()
        update_rows()
        txl.cuda.cta_sync()
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
