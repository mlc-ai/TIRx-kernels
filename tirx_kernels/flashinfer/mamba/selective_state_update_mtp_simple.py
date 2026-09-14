# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's selective-state-update MTP simple kernel.

Upstream source: include/flashinfer/mamba/kernel_selective_state_update_mtp_simple.cuh.
"""

import functools
import os
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.tirx_lite as txl

KERNEL_META = {
    "name": "selective_state_update_mtp_simple",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a", "sm_110a"],
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


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _cvt_rs_f16x2_f32(dst, a, b, random_bits):
    """Use FlashInfer's integer stochastic-conversion fallback on Thor."""
    from tirx_kernels.runner import PREPARE_CUDA_ARCH_ENV

    if os.environ.get(PREPARE_CUDA_ARCH_ENV, "sm_100a") != "sm_110a":
        txl.ptx.cvt.rs.f16x2.f32(dst, a, b, random_bits)
        return

    def cvt_rs_f16_sw(value, random13):
        materialized = txl.local_scalar("float32", init=value)
        bits = txl.reinterpret("uint32", materialized)
        sign = txl.bitwise_and(bits, txl.uint32(0x80000000))
        abs_bits = txl.bitwise_and(bits, txl.uint32(0x7FFFFFFF)) + txl.bitwise_and(
            random13, txl.uint32(0x1FFF)
        )
        f32_exp = txl.bitwise_and(txl.shift_right(abs_bits, txl.uint32(23)), txl.uint32(0xFF))
        f32_mantissa = txl.bitwise_and(abs_bits, txl.uint32(0x7FFFFF))
        normal = txl.bitwise_or(
            txl.shift_left(f32_exp - txl.uint32(112), txl.uint32(10)),
            txl.bitwise_and(txl.shift_right(abs_bits, txl.uint32(13)), txl.uint32(0x3FF)),
        )
        magnitude = txl.if_then_else(
            f32_exp == txl.uint32(0xFF),
            txl.if_then_else(f32_mantissa != txl.uint32(0), txl.uint32(0x7E00), txl.uint32(0x7C00)),
            txl.if_then_else(
                f32_exp > txl.uint32(142),
                txl.uint32(0x7C00),
                txl.if_then_else(f32_exp < txl.uint32(113), txl.uint32(0), normal),
            ),
        )
        return txl.bitwise_or(txl.shift_right(sign, txl.uint32(16)), magnitude)

    low = cvt_rs_f16_sw(b, txl.bitwise_and(random_bits, txl.uint32(0x1FFF)))
    high = cvt_rs_f16_sw(
        a, txl.bitwise_and(txl.shift_right(random_bits, txl.uint32(16)), txl.uint32(0x1FFF))
    )
    packed = txl.bitwise_or(low, txl.shift_left(high, txl.uint32(16)))
    txl.ptx.mov.b32(dst, packed)


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


def _extract_u16(word, high: bool):
    if high:
        return txl.cast(txl.shift_right(word, txl.uint32(16)), "uint16")
    return txl.cast(txl.bitwise_and(word, txl.uint32(0xFFFF)), "uint16")


def _bf16_word_to_f32x2(word):
    low_bits = txl.shift_left(word, txl.uint32(16))
    high_bits = txl.bitwise_and(word, txl.uint32(0xFFFF0000))
    return txl.cuda.make_float2(
        txl.reinterpret("float32", low_bits), txl.reinterpret("float32", high_bits)
    )


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "label": label,
        "batch": 64,
        "nheads": 64,
        "dim": 64,
        "dstate": 128,
        "tokens": 4,
        "heads_per_group": 8,
        "input_dtype": "bfloat16",
        "state_dtype": "bfloat16",
        "weight_dtype": "float32",
        "matrix_a_dtype": "float32",
        "index_dtype": "int64",
        "index_rank": 1,
        "cu_seqlens_dtype": "int32",
        "accepted_dtype": "int64",
        "mode": "fixed",
        "has_state_indices": True,
        "has_dst_indices": False,
        "has_intermediate_states": False,
        "has_num_accepted_tokens": False,
        "has_z": False,
        "has_d": True,
        "has_dt_bias": True,
        "dt_softplus": True,
        "update_state": True,
        "state_stride_factor": 1,
        "pad_every": 0,
        "use_out_tensor": True,
        "philox_rounds": 0,
        "shared_state_slot": False,
        "seed": 0,
    }
    config.update(overrides)
    return config


# FlashInfer's official MTP sweep: powers-of-two batch sizes through 2048,
# T=6, and BF16/FP32 state.  State update is disabled and all requests share a
# read-only cache slot.  Storage still contains the API-required batch number
# of slots, while every index points at slot zero.
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
    _case("b64_h64_d64_s128_t4_r8_int16", state_dtype="int16"),
    _case("b64_h64_d64_s128_t4_r8_philox10", state_dtype="float16", philox_rounds=10, seed=42),
    _case(
        "n8_h64_d64_s128_t4_r8_varlen_uniform",
        batch=8,
        mode="varlen_uniform",
        has_dst_indices=True,
        has_num_accepted_tokens=True,
        index_dtype="int32",
        index_rank=2,
    ),
    _case(
        "n8_h64_d64_s128_t6_r8_varlen_variable",
        batch=8,
        tokens=6,
        mode="varlen_variable",
        has_dst_indices=True,
        has_num_accepted_tokens=True,
        index_dtype="int32",
        index_rank=2,
    ),
]


# Correctness is an explicit one-variable-at-a-time matrix.  Rejection cases
# for the other algorithms live in their modules because every row here is in
# the simple kernel's real dispatch domain.
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
    _case("b64_h64_d64_s128_t4_r8_int16", state_dtype="int16"),
    _case(
        "b64_h64_d64_s128_t4_r8_int16_intermediate",
        state_dtype="int16",
        has_intermediate_states=True,
        update_state=False,
    ),
    _case("b64_h64_d64_s128_t4_r8_philox10", state_dtype="float16", philox_rounds=10, seed=42),
    _case(
        "b64_h64_d64_s128_t4_r8_philox10_intermediate",
        state_dtype="float16",
        philox_rounds=10,
        has_intermediate_states=True,
        update_state=False,
        seed=42,
    ),
    _case(
        "n4_h64_d64_s128_t4_r8_varlen_uniform",
        batch=4,
        mode="varlen_uniform",
        has_dst_indices=True,
        has_num_accepted_tokens=True,
        index_dtype="int32",
        index_rank=2,
    ),
    _case(
        "n8_h64_d64_s128_t6_r8_varlen_variable",
        batch=8,
        tokens=6,
        mode="varlen_variable",
        has_dst_indices=True,
        has_num_accepted_tokens=True,
        index_dtype="int32",
        index_rank=2,
    ),
    _case(
        "n4_h64_d64_s128_t4_r8_varlen_empty",
        batch=4,
        mode="varlen_empty",
        has_dst_indices=True,
        has_num_accepted_tokens=True,
        index_dtype="int32",
        index_rank=2,
    ),
    _case(
        "n8_h64_d64_s128_t4_r8_accepted_i32",
        batch=8,
        mode="varlen_uniform",
        has_dst_indices=True,
        has_num_accepted_tokens=True,
        accepted_dtype="int32",
        index_dtype="int32",
        index_rank=2,
    ),
    _case(
        "n8_h64_d64_s128_t4_r8_accepted_i64",
        batch=8,
        mode="varlen_uniform",
        has_dst_indices=True,
        has_num_accepted_tokens=True,
        accepted_dtype="int64",
        index_dtype="int64",
        index_rank=2,
    ),
]


_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _sequence_lengths(config: dict[str, Any], device: str | torch.device) -> torch.Tensor:
    batch = int(config["batch"])
    tokens = int(config["tokens"])
    mode = str(config["mode"])
    if mode == "varlen_variable":
        return torch.tensor(
            [(seq % tokens) + 1 for seq in range(batch)], dtype=torch.int64, device=device
        )
    if mode == "varlen_empty":
        return torch.tensor([0, *([tokens] * (batch - 1))], dtype=torch.int64, device=device)
    return torch.full((batch,), tokens, dtype=torch.int64, device=device)


def _total_tokens(config: dict[str, Any]) -> int:
    mode = str(config["mode"])
    batch = int(config["batch"])
    tokens = int(config["tokens"])
    if mode == "varlen_variable":
        return sum((seq % tokens) + 1 for seq in range(batch))
    if mode == "varlen_empty":
        return max(batch - 1, 0) * tokens
    return batch * tokens


def _num_sms(device: str | torch.device = "cuda") -> int:
    del device
    from tirx_kernels.runner import hardware_num_sms

    return hardware_num_sms()


def _specialization(config: dict[str, Any]) -> dict[str, Any]:
    batch = int(config["batch"])
    nheads = int(config["nheads"])
    dim = int(config["dim"])
    dstate = int(config["dstate"])
    tokens = int(config["tokens"])
    total_tokens = _total_tokens(config)
    heads_per_group = int(config["heads_per_group"])
    ngroups = nheads // heads_per_group
    is_varlen = str(config["mode"]).startswith("varlen")
    logical_slots = max(batch * tokens if is_varlen else batch, 1)
    if bool(config.get("has_dst_indices", False)):
        logical_slots *= 2
    state_slots = logical_slots
    state_stride_factor = int(config.get("state_stride_factor", 1))
    index_elements = batch * tokens if int(config["index_rank"]) == 2 else batch
    intermediate_elements = (
        batch * tokens * nheads * dim * dstate if bool(config["has_intermediate_states"]) else 1
    )
    state_dtype = str(config["state_dtype"])
    state_bytes = 4 if state_dtype == "float32" else 2
    scale_state = state_dtype == "int16"
    philox_rounds = int(config.get("philox_rounds", 0))
    if philox_rounds not in (0, 10):
        raise ValueError("MTP simple stochastic rounding supports philox_rounds in {0, 10}")
    if philox_rounds and state_dtype != "float16":
        raise ValueError("MTP simple Philox is restricted to float16 state")

    target_ctas = _num_sms(config.get("device", "cuda")) * 10
    total_tiles = max(batch * nheads, 1)
    requested_ctas = max(1, min(target_ctas // total_tiles, dim // 16))
    ctas_per_head = 4 if requested_ctas >= 4 else 2 if requested_ctas >= 2 else 1
    if "_ctas_per_head" in config:
        ctas_per_head = int(config["_ctas_per_head"])
    dim_per_cta = dim // ctas_per_head
    if ctas_per_head not in (1, 2, 4) or dim % ctas_per_head or dim_per_cta % 16:
        raise ValueError("MTP simple requires DIM_PER_CTA to be a multiple of 16")
    num_passes = dim_per_cta // 16
    state_stages = 1 if num_passes == 1 else 2

    dstate_pad = _align_up(dstate * 2, 128) // 2
    elems_per_tile_member = 16 // state_bytes
    pairs_per_tile_member = elems_per_tile_member // 2
    elems_per_tile = elems_per_tile_member * 8
    num_tiles = (_next_power_of_two(dstate) // 8) // elems_per_tile_member

    return {
        "BATCH": batch,
        "NHEADS": nheads,
        "DIM": dim,
        "DSTATE": dstate,
        "NTOKENS": tokens,
        "HEADS_PER_GROUP": heads_per_group,
        "CTAS_PER_HEAD": ctas_per_head,
        "DIM_PER_CTA": dim_per_cta,
        "DSTATE_PAD": dstate_pad,
        "NUM_PASSES": num_passes,
        "STATE_STAGES": state_stages,
        "STATE_BYTES": state_bytes,
        "ELEMS_PER_TILE_MEMBER": elems_per_tile_member,
        "PAIRS_PER_TILE_MEMBER": pairs_per_tile_member,
        "ELEMS_PER_TILE": elems_per_tile,
        "NUM_TILES": num_tiles,
        "HAS_STATE_INDICES": bool(config.get("has_state_indices", True)),
        "ASSUME_NO_PAD": bool(config.get("_assume_no_pad", False)),
        "HAS_DST_INDICES": bool(config.get("has_dst_indices", False)),
        "HAS_INTERMEDIATE_STATES": bool(config.get("has_intermediate_states", False)),
        "HAS_INTERMEDIATE_INDICES": bool(config.get("has_intermediate_states", False)),
        "HAS_CU_SEQLENS": is_varlen,
        "HAS_NUM_ACCEPTED_TOKENS": bool(config.get("has_num_accepted_tokens", False)),
        "HAS_Z": bool(config.get("has_z", False)),
        "HAS_D": bool(config.get("has_d", True)),
        "HAS_DT_BIAS": bool(config.get("has_dt_bias", True)),
        "DT_SOFTPLUS": bool(config.get("dt_softplus", True)),
        "UPDATE_STATE": bool(config.get("update_state", True)),
        "SCALE_STATE": scale_state,
        "PHILOX_ROUNDS": philox_rounds,
        # K allocates s_out immediately after a 128-byte-aligned x tile and
        # NTOKENS float32 dt values, so paired loads are aligned iff NTOKENS is even.
        "OUT_ALIGNED": tokens % 2 == 0,
        "STATE_ELEMENTS": state_slots * state_stride_factor * nheads * dim * dstate,
        "SCALE_ELEMENTS": state_slots * nheads * dim if scale_state else 1,
        "X_ELEMENTS": total_tokens * nheads * dim,
        "DT_ELEMENTS": total_tokens * nheads,
        "BC_ELEMENTS": total_tokens * ngroups * dstate,
        "INDEX_ELEMENTS": max(index_elements, 1),
        "INTERMEDIATE_ELEMENTS": max(intermediate_elements, 1),
        "INTERMEDIATE_SCALE_ELEMENTS": (
            batch * tokens * nheads * dim
            if bool(config["has_intermediate_states"]) and scale_state
            else 1
        ),
        "CU_SEQLENS_ELEMENTS": batch + 1,
        "ACCEPTED_ELEMENTS": max(batch, 1),
        "STATE_DTYPE": state_dtype,
        "WEIGHT_DTYPE": str(config["weight_dtype"]),
        "INDEX_DTYPE": str(config["index_dtype"]),
        "CU_SEQLENS_DTYPE": str(config["cu_seqlens_dtype"]),
        "ACCEPTED_DTYPE": str(config["accepted_dtype"]),
    }


def get_kernel(**kwargs: Any):
    """Build the K entry for one MTP simple specialization."""
    spec = _specialization(kwargs)
    schedule_heads_first = bool(kwargs.get("_schedule_heads_first", False))
    min_blocks_per_sm = int(kwargs.get("_min_blocks_per_sm", 0))

    ACCEPTED_DTYPE = spec["ACCEPTED_DTYPE"]
    ASSUME_NO_PAD = spec["ASSUME_NO_PAD"]
    CU_SEQLENS_DTYPE = spec["CU_SEQLENS_DTYPE"]
    DIM = spec["DIM"]
    DIM_PER_CTA = spec["DIM_PER_CTA"]
    DSTATE = spec["DSTATE"]
    DSTATE_PAD = spec["DSTATE_PAD"]
    DT_SOFTPLUS = spec["DT_SOFTPLUS"]
    ELEMS_PER_TILE = spec["ELEMS_PER_TILE"]
    ELEMS_PER_TILE_MEMBER = spec["ELEMS_PER_TILE_MEMBER"]
    HAS_CU_SEQLENS = spec["HAS_CU_SEQLENS"]
    HAS_D = spec["HAS_D"]
    HAS_DST_INDICES = spec["HAS_DST_INDICES"]
    HAS_DT_BIAS = spec["HAS_DT_BIAS"]
    HAS_INTERMEDIATE_INDICES = spec["HAS_INTERMEDIATE_INDICES"]
    HAS_INTERMEDIATE_STATES = spec["HAS_INTERMEDIATE_STATES"]
    HAS_NUM_ACCEPTED_TOKENS = spec["HAS_NUM_ACCEPTED_TOKENS"]
    HAS_STATE_INDICES = spec["HAS_STATE_INDICES"]
    HAS_Z = spec["HAS_Z"]
    HEADS_PER_GROUP = spec["HEADS_PER_GROUP"]
    INDEX_DTYPE = spec["INDEX_DTYPE"]
    NHEADS = spec["NHEADS"]
    NTOKENS = spec["NTOKENS"]
    NUM_PASSES = spec["NUM_PASSES"]
    NUM_TILES = spec["NUM_TILES"]
    OUT_ALIGNED = spec["OUT_ALIGNED"]
    PAIRS_PER_TILE_MEMBER = spec["PAIRS_PER_TILE_MEMBER"]
    PHILOX_ROUNDS = spec["PHILOX_ROUNDS"]
    SCALE_STATE = spec["SCALE_STATE"]
    STATE_BYTES = spec["STATE_BYTES"]
    STATE_DTYPE = spec["STATE_DTYPE"]
    STATE_STAGES = spec["STATE_STAGES"]
    UPDATE_STATE = spec["UPDATE_STATE"]
    WEIGHT_DTYPE = spec["WEIGHT_DTYPE"]

    kernel_options = {
        "warps": 4,
        "arch": "sm_100a",
        "grid": (
            (spec["NHEADS"], spec["BATCH"], spec["CTAS_PER_HEAD"])
            if schedule_heads_first
            else (spec["BATCH"], spec["NHEADS"], spec["CTAS_PER_HEAD"])
        ),
    }
    if min_blocks_per_sm:
        kernel_options["min_blocks_per_sm"] = min_blocks_per_sm

    @txl.kernel(**kernel_options)
    def selective_state_update_mtp_simple(
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
        intermediate_states: txl.gptr[spec["STATE_DTYPE"]],
        intermediate_indices: txl.gptr[spec["INDEX_DTYPE"]],
        intermediate_scales: txl.gptr[txl.f32],
        cu_seqlens: txl.gptr[spec["CU_SEQLENS_DTYPE"]],
        num_accepted_tokens: txl.gptr[spec["ACCEPTED_DTYPE"]],
        rand_seed: txl.gptr[txl.i64],
        output: txl.gptr[txl.bf16],
        state_stride_batch: txl.i64,
        state_scale_stride_batch: txl.i64,
        x_stride_batch: txl.i64,
        x_stride_mtp: txl.i64,
        dt_stride_batch: txl.i64,
        dt_stride_mtp: txl.i64,
        b_stride_batch: txl.i64,
        b_stride_mtp: txl.i64,
        c_stride_batch: txl.i64,
        c_stride_mtp: txl.i64,
        z_stride_batch: txl.i64,
        z_stride_mtp: txl.i64,
        out_stride_batch: txl.i64,
        out_stride_mtp: txl.i64,
        state_indices_stride_batch: txl.i64,
        state_indices_stride_t: txl.i64,
        dst_indices_stride_batch: txl.i64,
        dst_indices_stride_t: txl.i64,
        cache_steps: txl.i32,
        nheads_runtime: txl.i32,
        ngroups_runtime: txl.i32,
        dt_softplus: txl.i32,
        update_state: txl.i32,
        pad_slot_id: txl.i32,
    ):
        cta_x, cta_y, cta_z = txl.cta_id()
        if schedule_heads_first:
            head, seq_idx = cta_x, cta_y
        else:
            seq_idx, head = cta_x, cta_y
        smem = txl.smem_pool()
        s_b = smem.alloc((spec["NTOKENS"] * spec["DSTATE_PAD"],), txl.bf16, align=128)
        s_c = smem.alloc((spec["NTOKENS"] * spec["DSTATE_PAD"],), txl.bf16, align=128)
        s_x = smem.alloc((spec["NTOKENS"] * spec["DIM_PER_CTA"],), txl.bf16, align=128)
        s_dt = smem.alloc((spec["NTOKENS"],), txl.f32, align=4)
        s_out = smem.alloc((spec["NTOKENS"] * spec["DIM_PER_CTA"],), txl.f32, align=4)
        s_dst_slots = smem.alloc((spec["NTOKENS"],), txl.i64, align=8)
        s_state = smem.alloc(
            (spec["STATE_STAGES"] * 16 * spec["DSTATE_PAD"],), spec["STATE_DTYPE"], align=128
        )
        roles = txl.specialize()
        load_b = roles.role("load_b", warps=[0])
        load_c = roles.role("load_c", warps=[1])
        roles.role("common_only", warps=[2, 3])

        lane = txl.lane_id()
        warp = txl.warp_id()
        flat_tid = txl.thread_id()
        dim_offset = cta_z * DIM_PER_CTA
        kv_group = head // HEADS_PER_GROUP
        bos = txl.local_scalar("int32")
        seq_len = txl.local_scalar("int32")
        active_seq_len = seq_len if HAS_CU_SEQLENS else NTOKENS
        is_pad = txl.local_scalar("int32")
        state_ptr_offset_i32 = txl.local_scalar("int32")
        state_batch = txl.local_scalar("int64")
        state_head_offset = txl.local_scalar("int64")
        b_base = txl.local_scalar("int64")
        b_tstride = txl.local_scalar("int64")
        c_base = txl.local_scalar("int64")
        c_tstride = txl.local_scalar("int64")
        x_base = txl.local_scalar("int64")
        x_tstride = txl.local_scalar("int64")
        dt_base = txl.local_scalar("int64")
        dt_tstride = txl.local_scalar("int64")
        a_value = txl.local_scalar("float32")
        d_value = txl.local_scalar("float32")

        def prepare_sequence():
            # TIRX_TRANSCRIBE_START selective_state_update_mtp_simple

            txl.assign(bos, 0)
            txl.assign(seq_len, NTOKENS)
            if HAS_CU_SEQLENS:
                txl.assign(
                    bos,
                    txl.cast(_global_load_index_s64(cu_seqlens, seq_idx, CU_SEQLENS_DTYPE), "int32"),
                )
                eos: txl.int32 = txl.cast(
                    _global_load_index_s64(cu_seqlens, seq_idx + 1, CU_SEQLENS_DTYPE), "int32"
                )
                txl.assign(seq_len, eos - bos)

        def prepare_active_sequence():
            init_token_idx = txl.local_scalar("int32", init=0)
            if HAS_NUM_ACCEPTED_TOKENS:
                accepted: txl.int32 = txl.cast(
                    _global_load_index_s64(num_accepted_tokens, seq_idx, ACCEPTED_DTYPE), "int32"
                )
                txl.assign(init_token_idx, txl.if_then_else(accepted > 1, accepted - 1, 0))

            if HAS_STATE_INDICES:
                txl.assign(
                    state_batch,
                    _global_load_index_s64(
                        state_indices,
                        txl.cast(seq_idx, "int64") * state_indices_stride_batch
                        + txl.cast(init_token_idx, "int64") * state_indices_stride_t,
                        INDEX_DTYPE,
                    ),
                )
            else:
                txl.assign(state_batch, txl.cast(seq_idx, "int64"))
            if ASSUME_NO_PAD:
                txl.assign(is_pad, 0)
            else:
                txl.assign(is_pad, txl.if_then_else(state_batch != txl.cast(pad_slot_id, "int64"), 0, 1))
            txl.assign(
                state_head_offset,
                state_batch * state_stride_batch + txl.cast(head * DIM * DSTATE, "int64"),
            )
            txl.assign(state_ptr_offset_i32, txl.cast(state_head_offset, "int32"))

            gload_4 = txl.local_scalar("uint32")
            txl.ptx.ld.global_.b32(gload_4, matrix_a.ptr_to([head]))
            txl.assign(a_value, txl.reinterpret("float32", gload_4))
            txl.assign(d_value, 0.0)
            if HAS_D:
                txl.assign(d_value, _load_weight(d_weight, head, WEIGHT_DTYPE))

            if HAS_CU_SEQLENS:
                txl.assign(b_base, txl.cast(bos, "int64") * b_stride_batch)
                txl.assign(b_tstride, b_stride_batch)
                txl.assign(c_base, txl.cast(bos, "int64") * c_stride_batch)
                txl.assign(c_tstride, c_stride_batch)
                txl.assign(x_base, txl.cast(bos, "int64") * x_stride_batch)
                txl.assign(x_tstride, x_stride_batch)
                txl.assign(dt_base, txl.cast(bos, "int64") * dt_stride_batch)
                txl.assign(dt_tstride, dt_stride_batch)
            else:
                txl.assign(b_base, txl.cast(seq_idx, "int64") * b_stride_batch)
                txl.assign(b_tstride, b_stride_mtp)
                txl.assign(c_base, txl.cast(seq_idx, "int64") * c_stride_batch)
                txl.assign(c_tstride, c_stride_mtp)
                txl.assign(x_base, txl.cast(seq_idx, "int64") * x_stride_batch)
                txl.assign(x_tstride, x_stride_mtp)
                txl.assign(dt_base, txl.cast(seq_idx, "int64") * dt_stride_batch)
                txl.assign(dt_tstride, dt_stride_mtp)

        def load_bc_values(s_dst, src, src_base, src_tstride):
            with txl.serial((NTOKENS * DSTATE // 8 + 31) // 32) as load_iter:
                packed_i: txl.int32 = lane + load_iter * 32
                with txl.If(packed_i < NTOKENS * DSTATE // 8), txl.Then():
                    step: txl.int32 = packed_i // (DSTATE // 8)
                    col: txl.int32 = packed_i % (DSTATE // 8) * 8
                    with txl.If(step < active_seq_len), txl.Then():
                        txl.ptx["cp.async.cg.shared.global"](
                            s_dst.ptr_to([step * DSTATE_PAD + col]),
                            src.ptr_to(
                                [
                                    src_base
                                    + txl.cast(step, "int64") * src_tstride
                                    + kv_group * DSTATE
                                    + col
                                ]
                            ),
                            16,
                            16,
                        )

        def apply_z_gate(value, z_value):
            sub_0 = txl.local_scalar("float32")
            txl.ptx["sub.ftz.f32"](sub_0, txl.float32(0.0), z_value)
            mul_0 = txl.local_scalar("float32")
            txl.ptx["mul.ftz.f32"](mul_0, sub_0, txl.float32(_LOG2_E))
            exp2_0 = txl.local_scalar("float32")
            txl.ptx["ex2.approx.ftz.f32"](exp2_0, mul_0)
            exp_neg_z: txl.float32 = exp2_0
            add_0 = txl.local_scalar("float32")
            txl.ptx["add.ftz.f32"](add_0, txl.float32(1.0), exp_neg_z)
            div_0 = txl.local_scalar("float32")
            txl.ptx["div.approx.ftz.f32"](div_0, txl.float32(1.0), add_0)
            sigmoid_z: txl.float32 = div_0
            mul_1 = txl.local_scalar("float32")
            txl.ptx["mul.ftz.f32"](mul_1, z_value, sigmoid_z)
            txl.ptx["mul.ftz.f32"](value, value, mul_1)

        def update_sequence(IS_PAD: txl.constexpr):
            with txl.serial((NTOKENS + 3) // 4) as step_iter:
                step: txl.int32 = warp + step_iter * 4
                with txl.If(step < active_seq_len), txl.Then():
                    with txl.serial((DIM_PER_CTA // 8 + 31) // 32) as col_iter:
                        col: txl.int32 = (lane + col_iter * 32) * 8
                        with txl.If(col < DIM_PER_CTA), txl.Then():
                            txl.ptx["cp.async.cg.shared.global"](
                                s_x.ptr_to([step * DIM_PER_CTA + col]),
                                x.ptr_to(
                                    [
                                        x_base
                                        + txl.cast(step, "int64") * x_tstride
                                        + head * DIM
                                        + dim_offset
                                        + col
                                    ]
                                ),
                                16,
                                16,
                            )

            if not IS_PAD:
                with txl.serial((16 * DSTATE // (16 // STATE_BYTES) + 127) // 128) as state_load_iter:
                    packed_i: txl.int32 = flat_tid + state_load_iter * 128
                    with txl.If(packed_i < 16 * DSTATE // (16 // STATE_BYTES)), txl.Then():
                        state_row: txl.int32 = packed_i // (DSTATE // (16 // STATE_BYTES))
                        state_col: txl.int32 = (
                            packed_i % (DSTATE // (16 // STATE_BYTES)) * (16 // STATE_BYTES)
                        )
                        txl.ptx["cp.async.cg.shared.global"](
                            s_state.ptr_to([state_row * DSTATE_PAD + state_col]),
                            state.ptr_to(
                                [state_head_offset + (dim_offset + state_row) * DSTATE + state_col]
                            ),
                            16,
                            16,
                        )

            with txl.If(flat_tid < active_seq_len), txl.Then():
                dt_value = txl.local_scalar("float32")
                txl.assign(
                    dt_value,
                    _load_weight(
                        dt, dt_base + txl.cast(flat_tid, "int64") * dt_tstride + head, WEIGHT_DTYPE
                    ),
                )
                if HAS_DT_BIAS:
                    txl.ptx["add.ftz.f32"](
                        dt_value, dt_value, _load_weight(dt_bias, head, WEIGHT_DTYPE)
                    )
                if DT_SOFTPLUS:
                    with txl.If(dt_value <= txl.float32(20.0)), txl.Then():
                        mul_2 = txl.local_scalar("float32")
                        txl.ptx["mul.ftz.f32"](mul_2, dt_value, txl.float32(_LOG2_E))
                        exp2_1 = txl.local_scalar("float32")
                        txl.ptx["ex2.approx.ftz.f32"](exp2_1, mul_2)
                        dt_exp: txl.float32 = exp2_1
                        add_1 = txl.local_scalar("float32")
                        txl.ptx["add.ftz.f32"](add_1, txl.float32(1.0), dt_exp)
                        log2_0 = txl.local_scalar("float32")
                        txl.ptx["lg2.approx.ftz.f32"](log2_0, add_1)
                        txl.ptx["mul.ftz.f32"](dt_value, log2_0, txl.float32(_LN_2))
                txl.ptx.st.shared.b32(s_dt.ptr_to([flat_tid]), txl.reinterpret("uint32", dt_value))

            with txl.If(flat_tid < NTOKENS), txl.Then():
                step: txl.int32 = flat_tid
                dst_slot = txl.local_scalar("int64", init=-1)
                with txl.If(txl.And(txl.Not(IS_PAD), step < active_seq_len)), txl.Then():
                    if HAS_DST_INDICES:
                        dst_index: txl.int64 = _global_load_index_s64(
                            dst_indices,
                            txl.cast(seq_idx, "int64") * dst_indices_stride_batch
                            + txl.cast(step, "int64") * dst_indices_stride_t,
                            INDEX_DTYPE,
                        )
                        with txl.If(dst_index != txl.cast(pad_slot_id, "int64")), txl.Then():
                            txl.assign(dst_slot, dst_index)
                    elif HAS_INTERMEDIATE_STATES:
                        intermediate_index = txl.local_scalar("int64", init=state_batch)
                        if HAS_INTERMEDIATE_INDICES:
                            txl.assign(
                                intermediate_index,
                                _global_load_index_s64(intermediate_indices, seq_idx, INDEX_DTYPE),
                            )
                        txl.assign(
                            dst_slot, (intermediate_index * txl.cast(cache_steps, "int64") + step)
                        )
                    elif UPDATE_STATE:
                        with txl.If(step == active_seq_len - 1), txl.Then():
                            txl.assign(dst_slot, state_batch)
                txl.ptx.st.shared.b64(s_dst_slots.ptr_to([step]), dst_slot)

            txl.ptx.cp.async_.commit_group()
            txl.ptx.cp.async_.wait_group(0)
            txl.ptx.bar.sync(txl.uint32(0))

            random_seed = txl.local_scalar("int64", init=0)
            if PHILOX_ROUNDS > 0:
                txl.ptx.ld.global_.s64(random_seed, rand_seed.ptr_to([0]))

            member: txl.int32 = lane % 8
            row_group: txl.int32 = lane // 8
            with txl.serial(NUM_PASSES) as pass_idx:
                pass_row: txl.int32 = warp * 4 + row_group
                local_row: txl.int32 = pass_idx * 16 + pass_row
                row_d: txl.int32 = dim_offset + local_row
                state_stage: txl.int32 = pass_idx % STATE_STAGES
                decode_scale = txl.local_scalar("float32", init=1.0)
                if SCALE_STATE and not IS_PAD:
                    gload_5 = txl.local_scalar("uint32")
                    txl.ptx.ld.global_.b32(
                        gload_5,
                        state_scale.ptr_to(
                            [state_batch * state_scale_stride_batch + head * DIM + row_d]
                        ),
                    )
                    txl.assign(decode_scale, txl.reinterpret("float32", gload_5))

                r_state = txl.alloc_local((NUM_TILES * PAIRS_PER_TILE_MEMBER,), "uint64")
                with txl.unroll(NUM_TILES) as tile_idx:
                    member_col: txl.int32 = tile_idx * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER
                    with txl.If(txl.And(member_col < DSTATE, txl.Not(IS_PAD))):
                        with txl.Then():
                            state_words = txl.alloc_local((4,), "uint32")
                            txl.ptx.ld.shared.v4.b32(
                                state_words[0],
                                state_words[1],
                                state_words[2],
                                state_words[3],
                                s_state.ptr_to(
                                    [(state_stage * 16 + pass_row) * DSTATE_PAD + member_col]
                                ),
                            )
                            with txl.unroll(PAIRS_PER_TILE_MEMBER) as pair_idx:
                                state_pair = txl.local_scalar("uint64")
                                if STATE_DTYPE == "bfloat16":
                                    txl.assign(state_pair, _bf16_word_to_f32x2(state_words[pair_idx]))
                                elif STATE_DTYPE == "float16":
                                    f16_f32_0 = txl.local_scalar("float32")
                                    txl.ptx.cvt.f32.f16(
                                        f16_f32_0,
                                        txl.cast(
                                            _extract_u16(state_words[pair_idx], False), "uint16"
                                        ),
                                    )
                                    f16_f32_1 = txl.local_scalar("float32")
                                    txl.ptx.cvt.f32.f16(
                                        f16_f32_1,
                                        txl.cast(_extract_u16(state_words[pair_idx], True), "uint16"),
                                    )
                                    txl.assign(state_pair, txl.cuda.make_float2(f16_f32_0, f16_f32_1))
                                elif STATE_DTYPE == "int16":
                                    i16_f32_0 = txl.local_scalar("float32")
                                    txl.ptx.cvt.rn.f32.s16(
                                        i16_f32_0,
                                        txl.reinterpret(
                                            "int16",
                                            txl.cast(
                                                _extract_u16(state_words[pair_idx], False), "uint16"
                                            ),
                                        ),
                                    )
                                    i16_f32_1 = txl.local_scalar("float32")
                                    txl.ptx.cvt.rn.f32.s16(
                                        i16_f32_1,
                                        txl.reinterpret(
                                            "int16",
                                            txl.cast(
                                                _extract_u16(state_words[pair_idx], True), "uint16"
                                            ),
                                        ),
                                    )
                                    txl.assign(state_pair, txl.cuda.make_float2(i16_f32_0, i16_f32_1))
                                else:
                                    txl.assign(
                                        state_pair,
                                        txl.cuda.make_float2(
                                            txl.reinterpret("float32", state_words[pair_idx * 2]),
                                            txl.reinterpret("float32", state_words[pair_idx * 2 + 1]),
                                        ),
                                    )
                                if SCALE_STATE:
                                    txl.ptx.mul.f32x2(
                                        state_pair,
                                        state_pair,
                                        txl.cuda.make_float2(decode_scale, decode_scale),
                                    )
                                txl.ptx.mov.b64(
                                    r_state[tile_idx * PAIRS_PER_TILE_MEMBER + pair_idx], state_pair
                                )
                        with txl.Else():
                            with txl.unroll(PAIRS_PER_TILE_MEMBER) as pair_idx:
                                txl.ptx.mov.b64(
                                    r_state[tile_idx * PAIRS_PER_TILE_MEMBER + pair_idx],
                                    (txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0))),
                                )

                b_step = txl.local_scalar("int32", init=0)
                c_step = txl.local_scalar("int32", init=0)
                x_step = txl.local_scalar("int32", init=0)
                dt_step = txl.local_scalar("int32", init=0)
                out_step = txl.local_scalar("int32", init=0)
                with txl.serial(NTOKENS) as step:
                    with txl.If(step < active_seq_len), txl.Then():
                        dst_slot = txl.local_scalar("int64")
                        txl.ptx.ld.shared.b64(dst_slot, s_dst_slots.ptr_to([step]))
                        dt_value = txl.local_scalar("float32")
                        sload_0 = txl.local_scalar("uint32")
                        txl.ptx.ld.shared.b32(sload_0, s_dt.ptr_to([dt_step]))
                        txl.assign(dt_value, txl.reinterpret("float32", sload_0))
                        mul_3 = txl.local_scalar("float32")
                        txl.ptx["mul.ftz.f32"](mul_3, a_value, dt_value)
                        mul_4 = txl.local_scalar("float32")
                        txl.ptx["mul.ftz.f32"](mul_4, mul_3, txl.float32(_LOG2_E))
                        exp2_2 = txl.local_scalar("float32")
                        txl.ptx["ex2.approx.ftz.f32"](exp2_2, mul_4)
                        da_value: txl.float32 = exp2_2
                        sload_1 = txl.local_scalar("uint16")
                        txl.ptx.ld.shared.b16(sload_1, s_x.ptr_to([x_step + local_row]))
                        bf16_f32_1 = txl.local_scalar("float32")
                        txl.ptx.cvt.f32.bf16(bf16_f32_1, txl.cast(sload_1, "uint16"))
                        x_value: txl.float32 = bf16_f32_1
                        mul_5 = txl.local_scalar("float32")
                        txl.ptx["mul.ftz.f32"](mul_5, dt_value, x_value)
                        dtx_value: txl.float32 = mul_5
                        out_pair = txl.local_scalar("uint64")
                        txl.assign(out_pair, txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0)))

                        with txl.unroll(NUM_TILES) as tile_idx:
                            member_col: txl.int32 = (
                                tile_idx * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER
                            )
                            with txl.If(member_col < DSTATE), txl.Then():
                                b_words = txl.alloc_local((4,), "uint32")
                                c_words = txl.alloc_local((4,), "uint32")
                                if ELEMS_PER_TILE_MEMBER == 4:
                                    txl.ptx.ld.shared.v2.b32(
                                        b_words[0], b_words[1], s_b.ptr_to([b_step + member_col])
                                    )
                                    txl.ptx.ld.shared.v2.b32(
                                        c_words[0], c_words[1], s_c.ptr_to([c_step + member_col])
                                    )
                                else:
                                    txl.ptx.ld.shared.v4.b32(
                                        b_words[0],
                                        b_words[1],
                                        b_words[2],
                                        b_words[3],
                                        s_b.ptr_to([b_step + member_col]),
                                    )
                                    txl.ptx.ld.shared.v4.b32(
                                        c_words[0],
                                        c_words[1],
                                        c_words[2],
                                        c_words[3],
                                        s_c.ptr_to([c_step + member_col]),
                                    )
                                with txl.unroll(PAIRS_PER_TILE_MEMBER) as pair_idx:
                                    b_pair: txl.uint64 = _bf16_word_to_f32x2(b_words[pair_idx])
                                    c_pair: txl.uint64 = _bf16_word_to_f32x2(c_words[pair_idx])
                                    dbx_pair = txl.local_scalar("uint64")
                                    txl.ptx.mul.f32x2(
                                        dbx_pair, b_pair, txl.cuda.make_float2(dtx_value, dtx_value)
                                    )
                                    pair_index: txl.int32 = (
                                        tile_idx * PAIRS_PER_TILE_MEMBER + pair_idx
                                    )
                                    updated_state = txl.local_scalar("uint64")
                                    txl.ptx.fma.rn.f32x2(
                                        updated_state,
                                        txl.cuda.make_float2(da_value, da_value),
                                        r_state[pair_index],
                                        dbx_pair,
                                    )
                                    txl.ptx.mov.b64(r_state[pair_index], updated_state)
                                    txl.ptx.fma.rn.f32x2(out_pair, updated_state, c_pair, out_pair)

                        out_value = txl.local_scalar("float32")
                        txl.ptx["add.ftz.f32"](
                            out_value, txl.cuda.float2_x(out_pair), txl.cuda.float2_y(out_pair)
                        )
                        with txl.unroll(3) as delta_idx:
                            delta: txl.int32 = txl.shift_right(txl.int32(4), delta_idx)
                            txl.ptx["add.ftz.f32"](
                                out_value, out_value, _shfl_down_f32(out_value, delta)
                            )
                        with txl.If(member == 0), txl.Then():
                            fma_0 = txl.local_scalar("float32")
                            txl.ptx["fma.rn.ftz.f32"](fma_0, d_value, x_value, out_value)
                            row_output: txl.float32 = fma_0
                            txl.ptx.st.shared.b32(
                                s_out.ptr_to([out_step + local_row]),
                                txl.reinterpret("uint32", row_output),
                            )

                        txl.assign(b_step, b_step + DSTATE_PAD)
                        txl.assign(c_step, c_step + DSTATE_PAD)
                        txl.assign(x_step, x_step + DIM_PER_CTA)
                        txl.assign(dt_step, dt_step + 1)
                        txl.assign(out_step, out_step + DIM_PER_CTA)

                        with txl.If(dst_slot != txl.int64(-1)), txl.Then():
                            encode_scale = txl.local_scalar("float32", init=1.0)
                            if SCALE_STATE:
                                local_max = txl.local_scalar("float32", init=txl.float32(_FLT_LOWEST))
                                with txl.unroll(NUM_TILES) as tile_idx:
                                    with txl.unroll(PAIRS_PER_TILE_MEMBER) as pair_idx:
                                        col0: txl.int32 = (
                                            tile_idx * ELEMS_PER_TILE
                                            + member * ELEMS_PER_TILE_MEMBER
                                            + pair_idx * 2
                                        )
                                        with txl.If(col0 < DSTATE), txl.Then():
                                            state_pair = txl.local_scalar(
                                                "uint64",
                                                init=r_state[
                                                    tile_idx * PAIRS_PER_TILE_MEMBER + pair_idx
                                                ],
                                            )
                                            abs_0 = txl.local_scalar("float32")
                                            txl.ptx["abs.ftz.f32"](abs_0, txl.cuda.float2_x(state_pair))
                                            abs_1 = txl.local_scalar("float32")
                                            txl.ptx["abs.ftz.f32"](abs_1, txl.cuda.float2_y(state_pair))
                                            max_0 = txl.local_scalar("float32")
                                            txl.ptx["max.ftz.f32"](max_0, abs_0, abs_1)
                                            max_1 = txl.local_scalar("float32")
                                            txl.ptx["max.ftz.f32"](max_1, local_max, max_0)
                                            txl.assign(local_max, max_1)
                                with txl.unroll(3) as delta_idx:
                                    delta: txl.int32 = txl.shift_right(txl.int32(4), delta_idx)
                                    txl.ptx["max.ftz.f32"](
                                        local_max, local_max, _shfl_down_f32(local_max, delta)
                                    )
                                leader_lane: txl.int32 = txl.bitwise_and(lane, txl.int32(-8))
                                txl.assign(local_max, _shfl_idx_f32(local_max, leader_lane))
                                with txl.If(local_max != txl.float32(0.0)), txl.Then():
                                    txl.ptx["div.approx.ftz.f32"](
                                        encode_scale, txl.float32(32767.0), local_max
                                    )

                            dst_base = txl.local_scalar("int64")
                            if HAS_INTERMEDIATE_STATES:
                                txl.assign(
                                    dst_base,
                                    (
                                        dst_slot * txl.cast(NHEADS * DIM * DSTATE, "int64")
                                        + head * DIM * DSTATE
                                        + row_d * DSTATE
                                    ),
                                )
                            else:
                                txl.assign(
                                    dst_base,
                                    (
                                        dst_slot * state_stride_batch
                                        + head * DIM * DSTATE
                                        + row_d * DSTATE
                                    ),
                                )

                            with txl.unroll(NUM_TILES) as tile_idx:
                                member_col: txl.int32 = (
                                    tile_idx * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER
                                )
                                with txl.If(member_col < DSTATE), txl.Then():
                                    store_words = txl.alloc_local((4,), "uint32")
                                    random_words = txl.alloc_local((4,), "uint32")
                                    with txl.unroll(PAIRS_PER_TILE_MEMBER) as pair_idx:
                                        pair_index: txl.int32 = (
                                            tile_idx * PAIRS_PER_TILE_MEMBER + pair_idx
                                        )
                                        state_pair = txl.local_scalar(
                                            "uint64", init=r_state[pair_index]
                                        )
                                        if SCALE_STATE:
                                            txl.ptx.mul.f32x2(
                                                state_pair,
                                                state_pair,
                                                txl.cuda.make_float2(encode_scale, encode_scale),
                                            )
                                            max_2 = txl.local_scalar("float32")
                                            txl.ptx["max.ftz.f32"](
                                                max_2,
                                                txl.cuda.float2_x(state_pair),
                                                txl.float32(-32767.0),
                                            )
                                            min_0 = txl.local_scalar("float32")
                                            txl.ptx["min.ftz.f32"](min_0, max_2, txl.float32(32767.0))
                                            low_scaled: txl.float32 = min_0
                                            max_3 = txl.local_scalar("float32")
                                            txl.ptx["max.ftz.f32"](
                                                max_3,
                                                txl.cuda.float2_y(state_pair),
                                                txl.float32(-32767.0),
                                            )
                                            min_1 = txl.local_scalar("float32")
                                            txl.ptx["min.ftz.f32"](min_1, max_3, txl.float32(32767.0))
                                            high_scaled: txl.float32 = min_1
                                            low_i32 = txl.local_scalar("int32")
                                            high_i32 = txl.local_scalar("int32")
                                            txl.ptx.cvt.rni.ftz.s32.f32(low_i32, low_scaled)
                                            txl.ptx.cvt.rni.ftz.s32.f32(high_i32, high_scaled)
                                            prmt_0 = txl.local_scalar("uint32")
                                            txl.ptx["prmt.b32"](
                                                prmt_0,
                                                txl.cast(txl.reinterpret("uint32", low_i32), "uint32"),
                                                txl.cast(txl.reinterpret("uint32", high_i32), "uint32"),
                                                txl.uint32(0x5410),
                                            )
                                            txl.ptx.mov.b32(store_words[pair_idx], prmt_0)
                                        elif PHILOX_ROUNDS > 0:
                                            element_idx: txl.int32 = pair_idx * 2
                                            with txl.If(pair_idx % 2 == 0), txl.Then():
                                                offset_mad = txl.local_scalar("int32")
                                                txl.ptx.mad.lo.s32(
                                                    offset_mad,
                                                    row_d,
                                                    txl.int32(DSTATE),
                                                    state_ptr_offset_i32,
                                                )
                                                add_s32_0 = txl.local_scalar("int32")
                                                txl.ptx["add.s32"](
                                                    add_s32_0, offset_mad, member_col + element_idx
                                                )
                                                random_offset: txl.int32 = add_s32_0
                                                c0 = txl.local_scalar(
                                                    "uint32",
                                                    init=txl.reinterpret("uint32", random_offset),
                                                )
                                                c1_signed = txl.local_scalar("int32")
                                                txl.ptx.shr.s32(
                                                    c1_signed, random_offset, txl.uint32(31)
                                                )
                                                c1 = txl.local_scalar(
                                                    "uint32",
                                                    init=txl.reinterpret("uint32", c1_signed),
                                                )
                                                c2 = txl.local_scalar("uint32", init=0)
                                                c3 = txl.local_scalar("uint32", init=0)
                                                seed_u64: txl.uint64 = txl.reinterpret(
                                                    "uint64", random_seed
                                                )
                                                k0 = txl.local_scalar(
                                                    "uint32", init=txl.cast(seed_u64, "uint32")
                                                )
                                                k1 = txl.local_scalar(
                                                    "uint32",
                                                    init=txl.cast(
                                                        txl.shift_right(seed_u64, txl.uint64(32)),
                                                        "uint32",
                                                    ),
                                                )
                                                with txl.unroll(10) as philox_round:
                                                    old_c0: txl.uint32 = c0
                                                    old_c2: txl.uint32 = c2
                                                    mul_hi_0 = txl.local_scalar("uint32")
                                                    txl.ptx["mul.hi.u32"](
                                                        mul_hi_0, txl.uint32(0xCD9E8D57), old_c2
                                                    )
                                                    next_c0: txl.uint32 = txl.bitwise_xor(
                                                        txl.bitwise_xor(mul_hi_0, c1), k0
                                                    )
                                                    mul_hi_1 = txl.local_scalar("uint32")
                                                    txl.ptx["mul.hi.u32"](
                                                        mul_hi_1, txl.uint32(0xD2511F53), old_c0
                                                    )
                                                    next_c2: txl.uint32 = txl.bitwise_xor(
                                                        txl.bitwise_xor(mul_hi_1, c3), k1
                                                    )
                                                    mul_lo_0 = txl.local_scalar("int32")
                                                    txl.ptx["mul.lo.s32"](
                                                        mul_lo_0,
                                                        txl.int32(-845247145),
                                                        txl.reinterpret("int32", old_c2),
                                                    )
                                                    next_c1: txl.int32 = mul_lo_0
                                                    mul_lo_1 = txl.local_scalar("int32")
                                                    txl.ptx["mul.lo.s32"](
                                                        mul_lo_1,
                                                        txl.int32(-766435501),
                                                        txl.reinterpret("int32", old_c0),
                                                    )
                                                    next_c3: txl.int32 = mul_lo_1
                                                    add_s32_1 = txl.local_scalar("int32")
                                                    txl.ptx["add.s32"](
                                                        add_s32_1,
                                                        txl.reinterpret("int32", k0),
                                                        txl.int32(-1640531527),
                                                    )
                                                    next_k0: txl.int32 = add_s32_1
                                                    add_s32_2 = txl.local_scalar("int32")
                                                    txl.ptx["add.s32"](
                                                        add_s32_2,
                                                        txl.reinterpret("int32", k1),
                                                        txl.int32(-1150833019),
                                                    )
                                                    next_k1: txl.int32 = add_s32_2
                                                    txl.assign(c0, next_c0)
                                                    txl.assign(c1, txl.reinterpret("uint32", next_c1))
                                                    txl.assign(c2, next_c2)
                                                    txl.assign(c3, txl.reinterpret("uint32", next_c3))
                                                    txl.assign(k0, txl.reinterpret("uint32", next_k0))
                                                    txl.assign(k1, txl.reinterpret("uint32", next_k1))
                                                txl.ptx.mov.b32(random_words[0], c0)
                                                txl.ptx.mov.b32(random_words[1], c1)
                                                txl.ptx.mov.b32(random_words[2], c2)
                                                txl.ptx.mov.b32(random_words[3], c3)
                                            packed_f16 = txl.local_scalar("uint32")
                                            _cvt_rs_f16x2_f32(
                                                packed_f16,
                                                txl.cuda.float2_y(state_pair),
                                                txl.cuda.float2_x(state_pair),
                                                random_words[pair_idx % 2],
                                            )
                                            txl.ptx.mov.b32(store_words[pair_idx], (packed_f16))
                                        elif STATE_DTYPE == "bfloat16":
                                            low_bits = txl.local_scalar("uint16")
                                            f32_bf16_0 = txl.local_scalar("uint16")
                                            txl.ptx.cvt.rn.bf16.f32(
                                                f32_bf16_0, txl.cuda.float2_x(state_pair)
                                            )
                                            txl.assign(low_bits, (f32_bf16_0))
                                            high_bits = txl.local_scalar("uint16")
                                            f32_bf16_1 = txl.local_scalar("uint16")
                                            txl.ptx.cvt.rn.bf16.f32(
                                                f32_bf16_1, txl.cuda.float2_y(state_pair)
                                            )
                                            txl.assign(high_bits, (f32_bf16_1))
                                            txl.ptx.mov.b32(
                                                store_words[pair_idx], low_bits, high_bits
                                            )
                                        elif STATE_DTYPE == "float16":
                                            low_bits = txl.local_scalar("uint16")
                                            high_bits = txl.local_scalar("uint16")
                                            txl.ptx.cvt.rn.f16.f32(
                                                low_bits, txl.cuda.float2_x(state_pair)
                                            )
                                            txl.ptx.cvt.rn.f16.f32(
                                                high_bits, txl.cuda.float2_y(state_pair)
                                            )
                                            txl.ptx.mov.b32(
                                                store_words[pair_idx], low_bits, high_bits
                                            )
                                        else:
                                            txl.ptx.mov.b32(
                                                store_words[pair_idx * 2],
                                                txl.reinterpret(
                                                    "uint32", txl.cuda.float2_x(state_pair)
                                                ),
                                            )
                                            txl.ptx.mov.b32(
                                                store_words[pair_idx * 2 + 1],
                                                txl.reinterpret(
                                                    "uint32", txl.cuda.float2_y(state_pair)
                                                ),
                                            )
                                    if HAS_INTERMEDIATE_STATES:
                                        txl.ptx.st.global_.v4.b32(
                                            intermediate_states.ptr_to([dst_base + member_col]),
                                            store_words[0],
                                            store_words[1],
                                            store_words[2],
                                            store_words[3],
                                        )
                                    else:
                                        txl.ptx.st.global_.v4.b32(
                                            state.ptr_to([dst_base + member_col]),
                                            store_words[0],
                                            store_words[1],
                                            store_words[2],
                                            store_words[3],
                                        )

                            with txl.If(txl.And(SCALE_STATE, member == 0)), txl.Then():
                                rcp_0 = txl.local_scalar("float32")
                                txl.ptx["rcp.approx.ftz.f32"](rcp_0, encode_scale)
                                new_decode_scale: txl.float32 = rcp_0
                                scale_offset = txl.local_scalar("int64")
                                if HAS_INTERMEDIATE_STATES:
                                    txl.assign(
                                        scale_offset,
                                        (
                                            dst_slot * txl.cast(NHEADS * DIM, "int64")
                                            + head * DIM
                                            + row_d
                                        ),
                                    )
                                    txl.ptx.st.global_.b32(
                                        intermediate_scales.ptr_to([scale_offset]),
                                        txl.reinterpret("uint32", new_decode_scale),
                                    )
                                else:
                                    txl.assign(
                                        scale_offset,
                                        (dst_slot * state_scale_stride_batch + head * DIM + row_d),
                                    )
                                    txl.ptx.st.global_.b32(
                                        state_scale.ptr_to([scale_offset]),
                                        txl.reinterpret("uint32", new_decode_scale),
                                    )

                with txl.If(txl.And(NUM_PASSES > 1, pass_idx < NUM_PASSES - 1)), txl.Then():
                    next_stage: txl.int32 = (pass_idx + 1) % STATE_STAGES
                    next_dim_base: txl.int32 = dim_offset + (pass_idx + 1) * 16
                    if not IS_PAD:
                        with txl.serial(
                            (16 * DSTATE // (16 // STATE_BYTES) + 127) // 128
                        ) as state_load_iter:
                            packed_i: txl.int32 = flat_tid + state_load_iter * 128
                            with txl.If(packed_i < 16 * DSTATE // (16 // STATE_BYTES)), txl.Then():
                                state_row: txl.int32 = packed_i // (DSTATE // (16 // STATE_BYTES))
                                state_col: txl.int32 = (
                                    packed_i % (DSTATE // (16 // STATE_BYTES)) * (16 // STATE_BYTES)
                                )
                                txl.ptx["cp.async.cg.shared.global"](
                                    s_state.ptr_to(
                                        [(next_stage * 16 + state_row) * DSTATE_PAD + state_col]
                                    ),
                                    state.ptr_to(
                                        [
                                            state_head_offset
                                            + (next_dim_base + state_row) * DSTATE
                                            + state_col
                                        ]
                                    ),
                                    16,
                                    16,
                                )
                    txl.ptx.cp.async_.commit_group()
                    txl.ptx.cp.async_.wait_group(0)
                    txl.ptx.bar.sync(txl.uint32(0))

            txl.ptx.bar.sync(txl.uint32(0))
            with txl.serial((NTOKENS + 3) // 4) as output_iter:
                step: txl.int32 = warp + output_iter * 4
                with txl.If(step < active_seq_len), txl.Then():
                    out_base = txl.local_scalar("int64")
                    z_base = txl.local_scalar("int64")
                    if HAS_CU_SEQLENS:
                        txl.assign(
                            out_base,
                            (
                                txl.cast(bos + step, "int64") * out_stride_batch
                                + head * DIM
                                + dim_offset
                            ),
                        )
                        txl.assign(
                            z_base,
                            (
                                txl.cast(bos + step, "int64") * z_stride_batch
                                + head * DIM
                                + dim_offset
                            ),
                        )
                    else:
                        txl.assign(
                            out_base,
                            (
                                txl.cast(seq_idx, "int64") * out_stride_batch
                                + txl.cast(step, "int64") * out_stride_mtp
                                + head * DIM
                                + dim_offset
                            ),
                        )
                        txl.assign(
                            z_base,
                            (
                                txl.cast(seq_idx, "int64") * z_stride_batch
                                + txl.cast(step, "int64") * z_stride_mtp
                                + head * DIM
                                + dim_offset
                            ),
                        )

                    if DIM_PER_CTA >= 32:
                        output_count: txl.int32 = DIM_PER_CTA // 32
                        local_col: txl.int32 = lane * output_count
                        out_words = txl.alloc_local((4,), "uint32")
                        if DIM_PER_CTA == 32:
                            sload_2 = txl.local_scalar("uint32")
                            txl.ptx.ld.shared.b32(
                                sload_2, s_out.ptr_to([step * DIM_PER_CTA + local_col])
                            )
                            txl.ptx.mov.b32(out_words[0], sload_2)
                        elif DIM_PER_CTA == 64:
                            if OUT_ALIGNED:
                                txl.ptx.ld.shared.v2.b32(
                                    out_words[0],
                                    out_words[1],
                                    s_out.ptr_to([step * DIM_PER_CTA + local_col]),
                                )
                            else:
                                sload_3 = txl.local_scalar("uint32")
                                txl.ptx.ld.shared.b32(
                                    sload_3, s_out.ptr_to([step * DIM_PER_CTA + local_col])
                                )
                                txl.ptx.mov.b32(out_words[0], sload_3)
                                sload_4 = txl.local_scalar("uint32")
                                txl.ptx.ld.shared.b32(
                                    sload_4, s_out.ptr_to([step * DIM_PER_CTA + local_col + 1])
                                )
                                txl.ptx.mov.b32(out_words[1], sload_4)
                        else:
                            txl.ptx.ld.shared.v4.b32(
                                out_words[0],
                                out_words[1],
                                out_words[2],
                                out_words[3],
                                s_out.ptr_to([step * DIM_PER_CTA + local_col]),
                            )
                        z_bits = txl.alloc_local((4,), "uint16")
                        if HAS_Z:
                            if DIM_PER_CTA == 32:
                                txl.ptx.ld.global_.b16(z_bits[0], z.ptr_to([z_base + local_col]))
                            elif DIM_PER_CTA == 64:
                                txl.ptx.ld.global_.v2.b16(
                                    z_bits[0], z_bits[1], z.ptr_to([z_base + local_col])
                                )
                            else:
                                txl.ptx.ld.global_.v4.b16(
                                    z_bits[0],
                                    z_bits[1],
                                    z_bits[2],
                                    z_bits[3],
                                    z.ptr_to([z_base + local_col]),
                                )
                        output_bits = txl.alloc_local((4,), "uint16")
                        with txl.unroll(DIM_PER_CTA // 32) as element:
                            value = txl.local_scalar(
                                "float32", init=txl.reinterpret("float32", out_words[element])
                            )
                            if HAS_Z:
                                bf16_f32_2 = txl.local_scalar("float32")
                                txl.ptx.cvt.f32.bf16(bf16_f32_2, txl.cast(z_bits[element], "uint16"))
                                z_value: txl.float32 = bf16_f32_2
                                apply_z_gate(value, z_value)
                            txl.ptx.cvt.rn.bf16.f32(output_bits[element], value)
                        if DIM_PER_CTA == 32:
                            txl.ptx.st.global_.b16(
                                output.ptr_to([out_base + local_col]), output_bits[0]
                            )
                        elif DIM_PER_CTA == 64:
                            txl.ptx.st.global_.v2.b16(
                                output.ptr_to([out_base + local_col]),
                                output_bits[0],
                                output_bits[1],
                            )
                        else:
                            txl.ptx.st.global_.v4.b16(
                                output.ptr_to([out_base + local_col]),
                                output_bits[0],
                                output_bits[1],
                                output_bits[2],
                                output_bits[3],
                            )
                    else:
                        with txl.If(lane < DIM_PER_CTA), txl.Then():
                            value = txl.local_scalar("float32")
                            sload_5 = txl.local_scalar("uint32")
                            txl.ptx.ld.shared.b32(sload_5, s_out.ptr_to([step * DIM_PER_CTA + lane]))
                            txl.assign(value, txl.reinterpret("float32", sload_5))
                            if HAS_Z:
                                gload_6 = txl.local_scalar("uint16")
                                txl.ptx.ld.global_.b16(gload_6, z.ptr_to([z_base + lane]))
                                bf16_f32_3 = txl.local_scalar("float32")
                                txl.ptx.cvt.f32.bf16(bf16_f32_3, txl.cast(gload_6, "uint16"))
                                z_value: txl.float32 = bf16_f32_3
                                apply_z_gate(value, z_value)
                            f32_bf16_2 = txl.local_scalar("uint16")
                            txl.ptx.cvt.rn.bf16.f32(f32_bf16_2, value)
                            output_bit: txl.uint16 = f32_bf16_2
                            txl.ptx.st.global_.b16(output.ptr_to([out_base + lane]), output_bit)

        def run_active_sequence():
            prepare_active_sequence()
            with load_b:
                load_bc_values(s_b, matrix_b, b_base, b_tstride)
            with load_c:
                load_bc_values(s_c, matrix_c, c_base, c_tstride)
            if ASSUME_NO_PAD:
                update_sequence(False)
            else:
                with txl.If(is_pad != 0):
                    with txl.Then():
                        update_sequence(True)
                    with txl.Else():
                        update_sequence(False)

        prepare_sequence()
        if HAS_CU_SEQLENS:
            with txl.If(seq_len > 0), txl.Then():
                run_active_sequence()
        else:
            run_active_sequence()

    return selective_state_update_mtp_simple.func


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Allocate deterministic, independent TIRx and FlashInfer MTP cases."""
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for selective-state-update MTP simple")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10 and capability != (11, 0):
        raise SkipTest(f"MTP simple requires SM100 or Thor, got {capability}")

    batch = int(kwargs["batch"])
    nheads = int(kwargs["nheads"])
    dim = int(kwargs["dim"])
    dstate = int(kwargs["dstate"])
    tokens = int(kwargs["tokens"])
    heads_per_group = int(kwargs["heads_per_group"])
    if nheads % heads_per_group:
        raise ValueError("nheads must be divisible by heads_per_group")
    ngroups = nheads // heads_per_group
    state_dtype = _TORCH_DTYPES[str(kwargs["state_dtype"])]
    weight_dtype = _TORCH_DTYPES[str(kwargs["weight_dtype"])]
    index_dtype = _TORCH_DTYPES[str(kwargs["index_dtype"])]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(kwargs.get("seed", 0)) + 20260810)

    sequence_lengths = _sequence_lengths(kwargs, device)
    total_tokens = int(sequence_lengths.sum().item())
    is_varlen = str(kwargs["mode"]).startswith("varlen")
    logical_slots = max(batch * tokens if is_varlen else batch, 1)
    if bool(kwargs.get("has_dst_indices", False)):
        logical_slots *= 2
    state_cache_size = logical_slots
    stride_factor = int(kwargs.get("state_stride_factor", 1))
    state_storage_shape = (state_cache_size * stride_factor, nheads, dim, dstate)

    if state_dtype == torch.int16:
        logical_state = torch.randn(
            (state_cache_size, nheads, dim, dstate),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        amax = logical_state.abs().amax(dim=-1)
        encode = torch.where(amax == 0, torch.ones_like(amax), 32767.0 / amax)
        quantized = (logical_state * encode[..., None]).round().clamp(-32767, 32767).to(torch.int16)
        initial_state_storage = torch.zeros(state_storage_shape, dtype=state_dtype, device=device)
        initial_state = initial_state_storage[::stride_factor]
        initial_state.copy_(quantized)
        initial_state_scale = 1.0 / encode
    else:
        initial_state_storage = torch.randn(
            state_storage_shape, dtype=state_dtype, device=device, generator=generator
        )
        initial_state = initial_state_storage[::stride_factor]
        initial_state_scale = torch.ones((1,), dtype=torch.float32, device=device)

    if is_varlen:
        x = torch.randn(
            (total_tokens, nheads, dim), dtype=torch.bfloat16, device=device, generator=generator
        )
        dt_base = torch.randn(
            (total_tokens, nheads), dtype=weight_dtype, device=device, generator=generator
        )
        dt = dt_base.as_strided((total_tokens, nheads, dim), (nheads, 1, 0))
        matrix_b = torch.randn(
            (total_tokens, ngroups, dstate),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        matrix_c = torch.randn_like(matrix_b)
    else:
        x = torch.randn(
            (batch, tokens, nheads, dim), dtype=torch.bfloat16, device=device, generator=generator
        )
        dt_base = torch.randn(
            (batch, tokens, nheads), dtype=weight_dtype, device=device, generator=generator
        )
        dt = dt_base.as_strided((batch, tokens, nheads, dim), (tokens * nheads, nheads, 1, 0))
        matrix_b = torch.randn(
            (batch, tokens, ngroups, dstate),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        matrix_c = torch.randn_like(matrix_b)

    matrix_a_base = (
        -torch.rand((nheads,), dtype=torch.float32, device=device, generator=generator) - 1.0
    )
    matrix_a = matrix_a_base.as_strided((nheads, dim, dstate), (1, 0, 0))
    d_base = torch.randn((nheads,), dtype=weight_dtype, device=device, generator=generator)
    if not bool(kwargs.get("has_d", True)):
        d_base.zero_()
    d_weight = d_base.as_strided((nheads, dim), (1, 0))
    bias_base = torch.rand((nheads,), dtype=weight_dtype, device=device, generator=generator) - 4.0
    dt_bias = bias_base.as_strided((nheads, dim), (1, 0))
    z = torch.randn_like(x) if bool(kwargs.get("has_z", False)) else None

    if bool(kwargs.get("shared_state_slot", False)):
        source_indices = torch.zeros((batch, tokens), dtype=index_dtype, device=device)
    elif is_varlen:
        source_indices = torch.arange(batch * tokens, dtype=index_dtype, device=device).reshape(
            batch, tokens
        )
    else:
        source_indices = (
            torch.arange(batch, dtype=index_dtype, device=device)[:, None]
            .expand(batch, tokens)
            .clone()
        )
    destination_indices = source_indices.clone()
    if bool(kwargs.get("has_dst_indices", False)):
        destination_indices = source_indices + logical_slots // 2
    pad_every = int(kwargs.get("pad_every", 0))
    if pad_every:
        source_indices.reshape(-1)[::pad_every] = -1
    if is_varlen:
        for seq, length in enumerate(sequence_lengths.tolist()):
            source_indices[seq, length:] = -1
            destination_indices[seq, length:] = -1

    index_rank = int(kwargs.get("index_rank", 1))
    state_indices = source_indices if index_rank == 2 else source_indices[:, 0].contiguous()
    dst_indices = destination_indices if index_rank == 2 else destination_indices[:, 0].contiguous()
    cu_seqlens = torch.zeros(
        (batch + 1,), dtype=_TORCH_DTYPES[str(kwargs["cu_seqlens_dtype"])], device=device
    )
    cu_seqlens[1:] = torch.cumsum(sequence_lengths, dim=0).to(cu_seqlens.dtype)
    accepted_dtype = _TORCH_DTYPES[str(kwargs["accepted_dtype"])]
    num_accepted_tokens = torch.ones((batch,), dtype=accepted_dtype, device=device)
    if bool(kwargs.get("has_num_accepted_tokens", False)):
        num_accepted_tokens.copy_(
            torch.clamp(sequence_lengths, min=1, max=tokens).to(accepted_dtype)
        )

    intermediate_states = None
    intermediate_scales = None
    intermediate_indices = None
    if bool(kwargs.get("has_intermediate_states", False)):
        intermediate_states = torch.zeros(
            (batch, tokens, nheads, dim, dstate), dtype=state_dtype, device=device
        )
        intermediate_indices = torch.arange(batch, dtype=index_dtype, device=device)
        if state_dtype == torch.int16:
            intermediate_scales = torch.zeros(
                (batch, tokens, nheads, dim), dtype=torch.float32, device=device
            )

    tirx_output = torch.empty_like(x)
    flashinfer_output = torch.empty_like(x)
    return {
        "config": dict(kwargs),
        "spec": _specialization(kwargs),
        "sequence_lengths": sequence_lengths,
        "tirx_state_storage": initial_state_storage.clone(),
        "flashinfer_state_storage": initial_state_storage.clone(),
        "initial_state_storage": initial_state_storage,
        "tirx_state_scale": initial_state_scale.clone(),
        "flashinfer_state_scale": initial_state_scale.clone(),
        "x": x,
        "dt": dt,
        "dt_base": dt_base,
        "matrix_a": matrix_a,
        "matrix_a_base": matrix_a_base,
        "matrix_b": matrix_b,
        "matrix_c": matrix_c,
        "d_weight": d_weight,
        "d_base": d_base,
        "z": z,
        "dt_bias": dt_bias,
        "dt_bias_base": bias_base,
        "state_indices": state_indices,
        "dst_indices": dst_indices,
        "cu_seqlens": cu_seqlens,
        "num_accepted_tokens": num_accepted_tokens,
        "tirx_intermediate_states": (
            intermediate_states.clone()
            if intermediate_states is not None
            else torch.zeros((1,), dtype=state_dtype, device=device)
        ),
        "flashinfer_intermediate_states": (
            intermediate_states.clone()
            if intermediate_states is not None
            else torch.zeros((1,), dtype=state_dtype, device=device)
        ),
        "intermediate_state_indices": intermediate_indices,
        "tirx_intermediate_state_scales": (
            intermediate_scales.clone()
            if intermediate_scales is not None
            else torch.zeros((1,), dtype=torch.float32, device=device)
        ),
        "flashinfer_intermediate_state_scales": (
            intermediate_scales.clone()
            if intermediate_scales is not None
            else torch.zeros((1,), dtype=torch.float32, device=device)
        ),
        "rand_seed": torch.tensor([int(kwargs.get("seed", 0))], dtype=torch.int64, device=device),
        "tirx_output": tirx_output,
        "flashinfer_output": flashinfer_output,
    }


@functools.cache
def _load_oracle():
    from flashinfer.mamba import selective_state_update

    return selective_state_update


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    config = case["config"]
    nheads = int(config["nheads"])
    dim = int(config["dim"])
    dstate = int(config["dstate"])
    ngroups = nheads // int(config["heads_per_group"])
    tokens = int(config["tokens"])
    stride_factor = int(config.get("state_stride_factor", 1))
    is_varlen = str(config["mode"]).startswith("varlen")

    x = case["x"]
    dt = case["dt"]
    matrix_b = case["matrix_b"]
    matrix_c = case["matrix_c"]
    output = case["tirx_output"]
    if is_varlen:
        x_stride_batch = x.stride(0)
        x_stride_mtp = x.stride(0)
        dt_stride_batch = dt.stride(0)
        dt_stride_mtp = dt.stride(0)
        b_stride_batch = matrix_b.stride(0)
        b_stride_mtp = matrix_b.stride(0)
        c_stride_batch = matrix_c.stride(0)
        c_stride_mtp = matrix_c.stride(0)
        out_stride_batch = output.stride(0)
        out_stride_mtp = output.stride(0)
    else:
        x_stride_batch, x_stride_mtp = x.stride(0), x.stride(1)
        dt_stride_batch, dt_stride_mtp = dt.stride(0), dt.stride(1)
        b_stride_batch, b_stride_mtp = matrix_b.stride(0), matrix_b.stride(1)
        c_stride_batch, c_stride_mtp = matrix_c.stride(0), matrix_c.stride(1)
        out_stride_batch, out_stride_mtp = output.stride(0), output.stride(1)

    if case["z"] is not None:
        z_arg = case["z"]
        if is_varlen:
            z_stride_batch = z_arg.stride(0)
            z_stride_mtp = z_arg.stride(0)
        else:
            z_stride_batch, z_stride_mtp = z_arg.stride(0), z_arg.stride(1)
    else:
        z_arg = x
        z_stride_batch, z_stride_mtp = x_stride_batch, x_stride_mtp

    state_indices = case["state_indices"]
    dst_indices = case["dst_indices"]
    state_indices_stride_batch = state_indices.stride(0)
    state_indices_stride_t = state_indices.stride(1) if state_indices.ndim == 2 else 0
    dst_indices_stride_batch = dst_indices.stride(0)
    dst_indices_stride_t = dst_indices.stride(1) if dst_indices.ndim == 2 else 0
    intermediate_indices = case["intermediate_state_indices"]
    if intermediate_indices is None:
        intermediate_indices = torch.zeros(
            (int(config["batch"]),),
            dtype=_TORCH_DTYPES[str(config["index_dtype"])],
            device=x.device,
        )

    return (
        case["tirx_state_storage"].reshape(-1),
        case["tirx_state_scale"].reshape(-1),
        x.reshape(-1),
        case["dt_base"].reshape(-1),
        case["matrix_a_base"],
        matrix_b.reshape(-1),
        matrix_c.reshape(-1),
        case["d_base"],
        z_arg.reshape(-1),
        case["dt_bias_base"],
        state_indices.reshape(-1),
        dst_indices.reshape(-1),
        case["tirx_intermediate_states"].reshape(-1),
        intermediate_indices.reshape(-1),
        case["tirx_intermediate_state_scales"].reshape(-1),
        case["cu_seqlens"].reshape(-1),
        case["num_accepted_tokens"].reshape(-1),
        case["rand_seed"],
        output.reshape(-1),
        stride_factor * nheads * dim * dstate,
        nheads * dim if str(config["state_dtype"]) == "int16" else 0,
        x_stride_batch,
        x_stride_mtp,
        dt_stride_batch,
        dt_stride_mtp,
        b_stride_batch,
        b_stride_mtp,
        c_stride_batch,
        c_stride_mtp,
        z_stride_batch,
        z_stride_mtp,
        out_stride_batch,
        out_stride_mtp,
        state_indices_stride_batch,
        state_indices_stride_t,
        dst_indices_stride_batch,
        dst_indices_stride_t,
        tokens,
        nheads,
        ngroups,
        int(bool(config.get("dt_softplus", False))),
        int(bool(config.get("update_state", True))),
        -1,
    )


def _run_reference(case: dict[str, Any]) -> torch.Tensor:
    config = case["config"]
    stride_factor = int(config.get("state_stride_factor", 1))
    state_view = case["flashinfer_state_storage"][::stride_factor]
    scale_state = str(config["state_dtype"]) == "int16"
    source_out = case["flashinfer_output"] if bool(config.get("use_out_tensor", True)) else None
    oracle = _load_oracle()
    result = oracle(
        state_view,
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
        state_scale=case["flashinfer_state_scale"] if scale_state else None,
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
        intermediate_state_scales=(
            case["flashinfer_intermediate_state_scales"]
            if bool(config.get("has_intermediate_states", False)) and scale_state
            else None
        ),
        rand_seed=case["rand_seed"] if int(config.get("philox_rounds", 0)) else None,
        philox_rounds=int(config.get("philox_rounds", 0)),
        cache_steps=int(config["tokens"]),
        algorithm="simple",
        cu_seqlens=(case["cu_seqlens"] if str(config["mode"]).startswith("varlen") else None),
        num_accepted_tokens=(
            case["num_accepted_tokens"]
            if bool(config.get("has_num_accepted_tokens", False))
            else None
        ),
    )
    if source_out is None:
        case["flashinfer_output"].copy_(result)
    return result


def _assert_case_close(case: dict[str, Any]) -> None:
    config = case["config"]
    scale_state = str(config["state_dtype"]) == "int16"
    atol = 0.1 if scale_state else 2e-2
    rtol = 1e-2 if scale_state else 2e-2
    torch.testing.assert_close(case["tirx_output"], case["flashinfer_output"], atol=atol, rtol=rtol)

    stride_factor = int(config.get("state_stride_factor", 1))
    tirx_state = case["tirx_state_storage"][::stride_factor]
    reference_state = case["flashinfer_state_storage"][::stride_factor]
    if scale_state:
        torch.testing.assert_close(
            case["tirx_state_scale"], case["flashinfer_state_scale"], atol=2e-5, rtol=2e-4
        )
        tirx_state = tirx_state.float() * case["tirx_state_scale"][..., None]
        reference_state = reference_state.float() * case["flashinfer_state_scale"][..., None]
        torch.testing.assert_close(tirx_state, reference_state, atol=0.1, rtol=1e-2)
    else:
        state_atol = 2e-3 if str(config["state_dtype"]) == "float32" else 2e-2
        torch.testing.assert_close(tirx_state, reference_state, atol=state_atol, rtol=2e-2)

    if bool(config.get("has_intermediate_states", False)):
        tirx_intermediate = case["tirx_intermediate_states"]
        reference_intermediate = case["flashinfer_intermediate_states"]
        if scale_state:
            torch.testing.assert_close(
                case["tirx_intermediate_state_scales"],
                case["flashinfer_intermediate_state_scales"],
                atol=2e-5,
                rtol=2e-4,
            )
            tirx_intermediate = (
                tirx_intermediate.float() * case["tirx_intermediate_state_scales"][..., None]
            )
            reference_intermediate = (
                reference_intermediate.float()
                * case["flashinfer_intermediate_state_scales"][..., None]
            )
            torch.testing.assert_close(
                tirx_intermediate, reference_intermediate, atol=0.1, rtol=1e-2
            )
        else:
            torch.testing.assert_close(
                tirx_intermediate, reference_intermediate, atol=2e-2, rtol=2e-2
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
    torch.cuda.synchronize()
    _run_reference(case)
    torch.cuda.synchronize()
    _assert_case_close(case)


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
