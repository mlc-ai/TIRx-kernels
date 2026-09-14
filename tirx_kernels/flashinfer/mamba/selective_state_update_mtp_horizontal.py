# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's selective-state-update MTP horizontal kernel.

Upstream source: include/flashinfer/mamba/kernel_selective_state_update_mtp_horizontal.cuh.
"""

import ctypes
import functools
import os
from typing import Any

import torch

import tirx_kernels.tirx_lite as txl
from tirx_kernels.runner import PREPARE_CUDA_ARCH_ENV
from tvm.ir import PointerType, PrimType

from . import selective_state_update_mtp_simple as _simple
from . import selective_state_update_mtp_vertical as _vertical
from .selective_state_update_mtp_simple import _case, _shfl_down_f32

KERNEL_META = {
    "name": "selective_state_update_mtp_horizontal",
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


_LOG2_E = _simple._LOG2_E
_LN_2 = _simple._LN_2
_global_load_index_s64 = _simple._global_load_index_s64
_load_weight = _simple._load_weight
_extract_u16 = _simple._extract_u16
_bf16_word_to_f32x2 = _simple._bf16_word_to_f32x2
_philox4x32 = _vertical._philox4x32


def _mbarrier_arrive_wait_parity(barrier, parity, *, yield_while_waiting=False):
    txl.ptx.mbarrier.arrive.shared__cta.b64(barrier)
    ready = txl.local_scalar("uint32", init=txl.uint32(0))
    with txl.While(True):
        if yield_while_waiting:
            txl.cuda.nano_sleep(0)
        txl.ptx.mbarrier.try_wait.parity.shared__cta.b64(ready, barrier, txl.uint32(parity))
        with txl.If(ready != txl.uint32(0)), txl.Then():
            txl.Break()


def _tma_g2s_4d(dst, tensor_map, c0, c1, c2, c3, barrier):
    txl.ptx[_vertical._TMA_G2S_4D](
        dst,
        txl.address_of(tensor_map),
        txl.cast(c0, "int32"),
        txl.cast(c1, "int32"),
        txl.cast(c2, "int32"),
        txl.cast(c3, "int32"),
        barrier,
    )


def _store_state_tile(
    values,
    pair_base,
    random_words,
    random_base,
    destination_ptr,
    *,
    STATE_DTYPE,
    PAIRS_PER_TILE_MEMBER,
    PHILOX_ROUNDS,
):
    if STATE_DTYPE == "float32":
        words = txl.alloc_local((4,), "uint32")
        with txl.unroll(2) as pair:
            packed = values[pair_base + pair]
            txl.ptx.mov.b32(words[pair * 2], txl.reinterpret("uint32", txl.cuda.float2_x(packed)))
            txl.ptx.mov.b32(
                words[pair * 2 + 1], txl.reinterpret("uint32", txl.cuda.float2_y(packed))
            )
        txl.ptx.st.global_.v4.b32(destination_ptr, words[0], words[1], words[2], words[3])
    elif PHILOX_ROUNDS > 0:
        packed_words = txl.alloc_local((4,), "uint32")
        with txl.unroll(4) as pair:
            packed = values[pair_base + pair]
            _simple._cvt_rs_f16x2_f32(
                packed_words[pair],
                txl.cuda.float2_y(packed),
                txl.cuda.float2_x(packed),
                random_words[random_base + pair // 2 * 4 + pair % 2],
            )
        txl.ptx.st.global_.v4.b32(
            destination_ptr, packed_words[0], packed_words[1], packed_words[2], packed_words[3]
        )
    else:
        bits = txl.alloc_local((8,), "uint16")
        words = txl.alloc_local((4,), "uint32")
        with txl.unroll(PAIRS_PER_TILE_MEMBER) as pair:
            packed = values[pair_base + pair]
            if STATE_DTYPE == "bfloat16":
                f32_bf16_0 = txl.local_scalar("uint16")
                txl.ptx.cvt.rn.bf16.f32(f32_bf16_0, txl.cuda.float2_x(packed))
                txl.ptx.mov.b16(bits[pair * 2], f32_bf16_0)
                f32_bf16_1 = txl.local_scalar("uint16")
                txl.ptx.cvt.rn.bf16.f32(f32_bf16_1, txl.cuda.float2_y(packed))
                txl.ptx.mov.b16(bits[pair * 2 + 1], f32_bf16_1)
            else:
                f32_f16_0 = txl.local_scalar("uint16")
                txl.ptx.cvt.rn.f16.f32(f32_f16_0, txl.cuda.float2_x(packed))
                txl.ptx.mov.b16(bits[pair * 2], f32_f16_0)
                f32_f16_1 = txl.local_scalar("uint16")
                txl.ptx.cvt.rn.f16.f32(f32_f16_1, txl.cuda.float2_y(packed))
                txl.ptx.mov.b16(bits[pair * 2 + 1], f32_f16_1)
        with txl.unroll(4) as word:
            txl.ptx.mov.b32(words[word], bits[word * 2], bits[word * 2 + 1])
        txl.ptx.st.global_.v4.b32(destination_ptr, words[0], words[1], words[2], words[3])


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
        "HAS_DST_INDICES": bool(config.get("has_dst_indices", False)),
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


def _use_simple_dispatch(spec: dict[str, Any], arch: str) -> bool:
    if not arch.startswith("sm_110") or spec["HAS_DST_INDICES"]:
        return False
    shape = (
        spec["BATCH"],
        spec["NHEADS"],
        spec["DIM"],
        spec["DSTATE"],
        spec["NTOKENS"],
        spec["HEADS_PER_GROUP"],
        spec["STATE_DTYPE"],
    )
    return (
        shape == (1, 64, 64, 128, 6, 8, "bfloat16")
        or (
            shape in ((2, 64, 64, 128, 6, 8, "bfloat16"), (128, 64, 64, 128, 6, 8, "bfloat16"))
            and not spec["UPDATE_STATE"]
        )
        or (
            shape
            in (
                (1, 64, 64, 128, 6, 8, "float32"),
                (2, 64, 64, 128, 6, 8, "float32"),
                (4, 64, 64, 128, 6, 8, "float32"),
            )
            and not spec["UPDATE_STATE"]
        )
        or (
            spec["BATCH"],
            spec["NHEADS"],
            spec["DIM"],
            spec["DSTATE"],
            spec["NTOKENS"],
            spec["HEADS_PER_GROUP"],
            spec["STATE_DTYPE"],
            spec["WEIGHT_DTYPE"],
        )
        == (64, 64, 64, 128, 4, 8, "bfloat16", "bfloat16")
        or (
            spec["BATCH"],
            spec["NHEADS"],
            spec["DIM"],
            spec["DSTATE"],
            spec["NTOKENS"],
            spec["HEADS_PER_GROUP"],
            spec["STATE_DTYPE"],
        )
        in (
            (64, 64, 128, 128, 4, 8, "bfloat16"),
            (64, 64, 64, 128, 4, 8, "bfloat16"),
            (64, 64, 64, 128, 2, 8, "bfloat16"),
            (64, 64, 64, 128, 8, 8, "bfloat16"),
            (64, 64, 64, 128, 4, 1, "bfloat16"),
            (64, 64, 64, 96, 4, 8, "bfloat16"),
            (64, 64, 64, 64, 4, 8, "bfloat16"),
        )
        or (shape == (64, 64, 64, 128, 4, 8, "float16") and spec["PHILOX_ROUNDS"] == 0)
        or (
            (spec["INDEX_DTYPE"] == "int32" or spec["HAS_INTERMEDIATE_STATES"])
            and (
                spec["BATCH"],
                spec["NHEADS"],
                spec["DIM"],
                spec["DSTATE"],
                spec["NTOKENS"],
                spec["HEADS_PER_GROUP"],
                spec["STATE_DTYPE"],
            )
            == (64, 64, 64, 128, 4, 8, "bfloat16")
        )
    )


def _simple_ptxas_level(spec: dict[str, Any], arch: str) -> str | None:
    if not arch.startswith("sm_110"):
        return None
    shape = (
        spec["BATCH"],
        spec["NHEADS"],
        spec["DIM"],
        spec["DSTATE"],
        spec["NTOKENS"],
        spec["HEADS_PER_GROUP"],
        spec["STATE_DTYPE"],
    )
    if shape == (64, 64, 64, 128, 8, 8, "bfloat16"):
        return "5"
    if shape == (64, 64, 64, 128, 4, 1, "bfloat16"):
        return "2"
    if shape == (64, 64, 64, 96, 4, 8, "bfloat16"):
        return "5"
    if shape == (64, 64, 64, 128, 4, 8, "bfloat16") and spec["HAS_INTERMEDIATE_STATES"]:
        return "9"
    if shape == (64, 64, 64, 64, 4, 8, "bfloat16") or (
        shape == (64, 64, 64, 128, 4, 8, "bfloat16") and spec["INDEX_DTYPE"] == "int32"
    ):
        return "7"
    return None


def _simple_min_blocks_per_sm(spec: dict[str, Any], arch: str) -> int | None:
    if not arch.startswith("sm_110"):
        return None
    shape = (
        spec["BATCH"],
        spec["NHEADS"],
        spec["DIM"],
        spec["DSTATE"],
        spec["NTOKENS"],
        spec["HEADS_PER_GROUP"],
        spec["STATE_DTYPE"],
    )
    if shape == (64, 64, 64, 128, 4, 8, "bfloat16") and spec["HAS_INTERMEDIATE_STATES"]:
        return 2
    if shape == (64, 64, 64, 128, 4, 8, "bfloat16") and spec["UPDATE_STATE"]:
        return 2
    return None


def _ptxas_level(spec: dict[str, Any], arch: str) -> str:
    ptxas_level = "10"
    if (
        arch.startswith("sm_110")
        and spec["HAS_INTERMEDIATE_STATES"]
        and (spec["BATCH"], spec["DIM"], spec["DSTATE"], spec["NTOKENS"], spec["HEADS_PER_GROUP"])
        == (64, 64, 128, 4, 8)
    ):
        ptxas_level = "6"
    elif (
        arch.startswith("sm_110")
        and spec["WEIGHT_DTYPE"] == "bfloat16"
        and (spec["BATCH"], spec["DIM"], spec["DSTATE"], spec["NTOKENS"], spec["HEADS_PER_GROUP"])
        == (64, 64, 128, 4, 8)
    ):
        ptxas_level = "9"
    elif (
        arch.startswith("sm_110")
        and spec["INDEX_DTYPE"] == "int32"
        and (spec["BATCH"], spec["DIM"], spec["DSTATE"], spec["NTOKENS"], spec["HEADS_PER_GROUP"])
        == (64, 64, 128, 4, 8)
    ):
        ptxas_level = "9"
    elif (
        arch.startswith("sm_110")
        and spec["HAS_Z"]
        and (spec["BATCH"], spec["DIM"], spec["DSTATE"], spec["NTOKENS"], spec["HEADS_PER_GROUP"])
        == (64, 64, 128, 4, 8)
    ):
        ptxas_level = "8"
    elif (
        arch.startswith("sm_110")
        and spec["STATE_DTYPE"] == "bfloat16"
        and (spec["BATCH"], spec["DIM"], spec["DSTATE"], spec["NTOKENS"], spec["HEADS_PER_GROUP"])
        == (64, 64, 64, 4, 8)
    ):
        ptxas_level = "9"
    elif arch.startswith("sm_110") and (
        spec["BATCH"],
        spec["DIM"],
        spec["DSTATE"],
        spec["NTOKENS"],
        spec["HEADS_PER_GROUP"],
        spec["STATE_DTYPE"],
    ) == (64, 128, 128, 4, 8, "bfloat16"):
        ptxas_level = "6"
    elif arch.startswith("sm_110") and (
        spec["BATCH"],
        spec["DIM"],
        spec["DSTATE"],
        spec["NTOKENS"],
        spec["HEADS_PER_GROUP"],
    ) == (64, 64, 128, 8, 8):
        ptxas_level = "3"
    elif arch.startswith("sm_110") and (
        spec["BATCH"],
        spec["DIM"],
        spec["DSTATE"],
        spec["NTOKENS"],
        spec["HEADS_PER_GROUP"],
    ) == (64, 64, 128, 4, 16):
        ptxas_level = "2"
    elif arch.startswith("sm_110") and (
        spec["BATCH"],
        spec["DIM"],
        spec["DSTATE"],
        spec["NTOKENS"],
        spec["HEADS_PER_GROUP"],
    ) == (64, 64, 128, 4, 64):
        ptxas_level = "5"
    elif arch.startswith("sm_110") and (
        spec["BATCH"],
        spec["DIM"],
        spec["DSTATE"],
        spec["NTOKENS"],
        spec["HEADS_PER_GROUP"],
    ) == (64, 64, 128, 4, 1):
        ptxas_level = "5"
    elif arch.startswith("sm_110") and (
        spec["BATCH"],
        spec["DIM"],
        spec["DSTATE"],
        spec["NTOKENS"],
        spec["HEADS_PER_GROUP"],
    ) == (64, 64, 128, 2, 8):
        ptxas_level = "4"
    elif arch.startswith("sm_110") and (
        spec["BATCH"],
        spec["DIM"],
        spec["DSTATE"],
        spec["NTOKENS"],
        spec["HEADS_PER_GROUP"],
    ) == (64, 64, 128, 1, 8):
        ptxas_level = "0"
    elif arch.startswith("sm_110") and (
        spec["BATCH"],
        spec["NHEADS"],
        spec["DIM"],
        spec["DSTATE"],
        spec["NTOKENS"],
        spec["HEADS_PER_GROUP"],
        spec["STATE_DTYPE"],
    ) == (1, 64, 64, 128, 6, 8, "float32"):
        ptxas_level = "5"
    elif arch.startswith("sm_110") and (
        spec["BATCH"],
        spec["DIM"],
        spec["DSTATE"],
        spec["NTOKENS"],
        spec["HEADS_PER_GROUP"],
        spec["STATE_DTYPE"],
    ) == (64, 64, 128, 4, 8, "bfloat16"):
        ptxas_level = "0"
    elif arch.startswith("sm_110") and spec["STATE_DTYPE"] == "float16":
        ptxas_level = "9"
    return ptxas_level


def get_kernel(**kwargs: Any):
    spec = _specialization(kwargs)
    arch = os.environ.get(PREPARE_CUDA_ARCH_ENV, "")
    if _use_simple_dispatch(spec, arch):
        simple_kwargs = dict(kwargs)
        simple_kwargs["_assume_no_pad"] = int(kwargs.get("pad_every", 0)) == 0
        simple_kwargs["_schedule_heads_first"] = spec["BATCH"] >= 64
        if (
            spec["BATCH"],
            spec["NHEADS"],
            spec["DIM"],
            spec["DSTATE"],
            spec["NTOKENS"],
            spec["HEADS_PER_GROUP"],
            spec["STATE_DTYPE"],
        ) == (1, 64, 64, 128, 6, 8, "float32"):
            simple_kwargs["_ctas_per_head"] = 1
        min_blocks = _simple_min_blocks_per_sm(spec, arch)
        if min_blocks is not None:
            simple_kwargs["_min_blocks_per_sm"] = min_blocks
        return _simple.get_kernel(**simple_kwargs)
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
    thor_bf16_shape = arch.startswith("sm_110") and STATE_DTYPE == "bfloat16"
    use_direct_state_index = thor_bf16_shape and (
        (spec["BATCH"], DIM, DSTATE, NTOKENS, HEADS_PER_GROUP)
        in ((64, 128, 128, 4, 8), (64, 64, 128, 4, 16), (64, 64, 64, 4, 8))
        or (
            INDEX_DTYPE == "int32"
            and (spec["BATCH"], DIM, DSTATE, NTOKENS, HEADS_PER_GROUP) == (64, 64, 128, 4, 8)
        )
    )
    use_direct_bc_index = thor_bf16_shape and (
        (spec["BATCH"], DIM, DSTATE, NTOKENS, HEADS_PER_GROUP)
        in ((64, 128, 128, 4, 8), (64, 64, 128, 4, 16), (64, 64, 64, 4, 8))
        or (
            (HAS_INTERMEDIATE_STATES or WEIGHT_DTYPE == "bfloat16")
            and (spec["BATCH"], DIM, DSTATE, NTOKENS, HEADS_PER_GROUP) == (64, 64, 128, 4, 8)
        )
    )
    # A zero-duration nanosleep yields the polling warp so the producer can
    # advance on the Thor shapes where the mbarrier handoff is latency-bound.
    use_wait_yield = thor_bf16_shape and (
        (spec["BATCH"], DIM, DSTATE, NTOKENS, HEADS_PER_GROUP)
        in (
            (64, 128, 128, 4, 8),
            (64, 64, 128, 4, 16),
            (64, 64, 128, 4, 64),
            (64, 64, 64, 4, 8),
            (64, 64, 96, 4, 8),
            (64, 64, 128, 2, 8),
        )
        or (
            (INDEX_DTYPE == "int32" or WEIGHT_DTYPE == "bfloat16" or HAS_Z)
            and (spec["BATCH"], DIM, DSTATE, NTOKENS, HEADS_PER_GROUP) == (64, 64, 128, 4, 8)
        )
    )
    min_blocks = 7
    if arch.startswith("sm_110"):
        if HAS_INTERMEDIATE_STATES:
            min_blocks = 6
        elif (
            STATE_DTYPE == "float16"
            and PHILOX_ROUNDS == 0
            and (spec["BATCH"], DIM, DSTATE, NTOKENS, HEADS_PER_GROUP) == (64, 64, 128, 4, 8)
        ):
            min_blocks = 5
        elif (
            spec["BATCH"],
            DIM,
            DSTATE,
            NTOKENS,
            HEADS_PER_GROUP,
            STATE_DTYPE,
            WEIGHT_DTYPE,
            INDEX_DTYPE,
            HAS_Z,
            HAS_D,
            HAS_DT_BIAS,
            UPDATE_STATE,
        ) == (64, 64, 128, 4, 8, "bfloat16", "float32", "int64", False, True, True, True):
            min_blocks = 10
        elif (
            STATE_DTYPE == "bfloat16"
            and WEIGHT_DTYPE == "bfloat16"
            and (spec["BATCH"], DIM, DSTATE, NTOKENS, HEADS_PER_GROUP) == (64, 64, 128, 4, 8)
        ):
            min_blocks = 10
        elif (
            STATE_DTYPE == "bfloat16"
            and INDEX_DTYPE == "int32"
            and (spec["BATCH"], DIM, DSTATE, NTOKENS, HEADS_PER_GROUP) == (64, 64, 128, 4, 8)
        ):
            min_blocks = 5
        elif (
            STATE_DTYPE == "bfloat16"
            and HAS_Z
            and (spec["BATCH"], DIM, DSTATE, NTOKENS, HEADS_PER_GROUP) == (64, 64, 128, 4, 8)
        ):
            min_blocks = 5
        elif STATE_DTYPE == "bfloat16" and (spec["BATCH"], DIM, DSTATE, NTOKENS) in (
            (64, 64, 64, 4),
            (64, 64, 128, 2),
        ):
            min_blocks = 7 if DSTATE == 64 else 8
        elif STATE_DTYPE == "bfloat16" and (
            spec["BATCH"],
            DIM,
            DSTATE,
            NTOKENS,
            HEADS_PER_GROUP,
        ) == (64, 64, 128, 8, 8):
            min_blocks = 9
        elif STATE_DTYPE == "bfloat16" and (
            spec["BATCH"],
            DIM,
            DSTATE,
            NTOKENS,
            HEADS_PER_GROUP,
        ) == (64, 64, 128, 4, 1):
            min_blocks = 8
        elif STATE_DTYPE == "bfloat16" and (
            spec["BATCH"],
            DIM,
            DSTATE,
            NTOKENS,
            HEADS_PER_GROUP,
        ) == (64, 64, 128, 4, 16):
            min_blocks = 6
        elif STATE_DTYPE == "bfloat16" and (
            spec["BATCH"],
            DIM,
            DSTATE,
            NTOKENS,
            HEADS_PER_GROUP,
        ) == (64, 128, 128, 4, 8):
            min_blocks = 10
        elif STATE_DTYPE == "bfloat16" and (
            spec["BATCH"],
            DIM,
            DSTATE,
            NTOKENS,
            HEADS_PER_GROUP,
        ) == (64, 64, 128, 4, 64):
            min_blocks = 5

    @txl.kernel(
        warps=5, arch="sm_100a", min_blocks_per_sm=min_blocks, grid=(spec["BATCH"], spec["NHEADS"])
    )
    def selective_state_update_mtp_horizontal(
        tensor_state: txl.TensorMap,
        tensor_b: txl.TensorMap,
        tensor_c: txl.TensorMap,
        tensor_x: txl.TensorMap,
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
        cu_seqlens: txl.gptr[str(kwargs["cu_seqlens_dtype"])],
        num_accepted_tokens: txl.gptr[str(kwargs["accepted_dtype"])],
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
        batch_i, head = txl.cta_id()
        if spec["HAS_STATE_INDICES"]:
            if spec["INDEX_DTYPE"] == "int32":
                _t1 = txl.local_scalar("int32")
                txl.ptx.ld.global_.nc.b32(_t1, state_indices.ptr_to([batch_i]))
                state_batch = txl.cast(_t1, "int64")
            else:
                _t2 = txl.local_scalar("int64")
                txl.ptx.ld.global_.nc.b64(_t2, state_indices.ptr_to([batch_i]))
                state_batch = _t2
        else:
            state_batch = txl.cast(batch_i, "int64")

        smem = txl.smem_pool()
        s_b = smem.alloc((spec["NTOKENS"] * spec["DSTATE_PAD"],), txl.bf16, align=128)
        s_c = smem.alloc((spec["NTOKENS"] * spec["DSTATE_PAD"],), txl.bf16, align=128)
        s_state_words = smem.alloc((2 * spec["STATE_STAGE_BYTES"] // 4,), txl.u32, align=128)
        s_x = smem.alloc((spec["NTOKENS"] * spec["DIM"],), txl.bf16, align=128)
        s_dt = smem.alloc((spec["NTOKENS"],), txl.f32, align=4)
        s_out = smem.alloc((spec["NTOKENS"] * spec["DIM"],), txl.f32, align=4)
        empty = txl.MBarrier(smem, 2, leader=True)
        full = txl.MBarrier(smem, 2, leader=True)
        out_ready = txl.MBarrier(smem, 1, leader=True)
        with txl.If(txl.thread_id() == 0), txl.Then():
            empty.init(160)
            full.init(129)
            out_ready.init(128)
        txl.cuda.cta_sync()

        roles = txl.specialize()
        update = roles.role("update", warps=[0, 1, 2, 3])
        load = roles.role("load", warps=[4])

        s_b_words = s_b.view("uint32")
        s_c_words = s_c.view("uint32")
        if use_direct_state_index:
            s_state_values = s_state_words.view(STATE_DTYPE)
        empty_barriers = empty.buf
        full_barriers = full.buf
        out_ready_barrier = out_ready.buf

        def run_update(IS_PAD: txl.constexpr):
            lane: txl.int32 = txl.lane_id()
            compute_warp: txl.int32 = txl.warp_id_in_role()
            random_seed = txl.local_scalar("int64", init=0)
            if PHILOX_ROUNDS > 0 and not IS_PAD:
                txl.ptx.ld.global_.s64(random_seed, rand_seed.ptr_to([0]))

            if HAS_INTERMEDIATE_STATES and not IS_PAD:
                icache_idx = txl.local_scalar(
                    "int64", init=_global_load_index_s64(intermediate_indices, batch_i, INDEX_DTYPE)
                )

            gload_0 = txl.local_scalar("uint32")
            txl.ptx.ld.global_.b32(gload_0, matrix_a.ptr_to([head]))
            a_value: txl.float32 = txl.reinterpret("float32", gload_0)
            d_value = txl.local_scalar("float32", init=0.0)
            if HAS_D:
                txl.assign(d_value, _load_weight(d_weight, head, WEIGHT_DTYPE))
            bias_value = txl.local_scalar("float32", init=0.0)
            if HAS_DT_BIAS:
                txl.assign(bias_value, _load_weight(dt_bias, head, WEIGHT_DTYPE))

            txl.ptx.mbarrier.arrive.shared__cta.b64(empty_barriers.ptr_to([0]))
            txl.ptx.mbarrier.arrive.shared__cta.b64(empty_barriers.ptr_to([1]))

            flat_thread: txl.int32 = compute_warp * 32 + lane
            with txl.If(flat_thread < NTOKENS), txl.Then():
                dt_value = txl.local_scalar("float32")
                txl.assign(
                    dt_value,
                    _load_weight(
                        dt,
                        txl.cast(batch_i, "int64") * dt_stride_batch
                        + txl.cast(flat_thread, "int64") * dt_stride_mtp
                        + head,
                        WEIGHT_DTYPE,
                    ),
                )
                if HAS_DT_BIAS:
                    txl.ptx["add.ftz.f32"](dt_value, dt_value, bias_value)
                with txl.If(txl.And(dt_softplus != 0, dt_value <= txl.float32(20.0))), txl.Then():
                    mul_0 = txl.local_scalar("float32")
                    txl.ptx["mul.ftz.f32"](mul_0, dt_value, txl.float32(_LOG2_E))
                    exp2_0 = txl.local_scalar("float32")
                    txl.ptx["ex2.approx.ftz.f32"](exp2_0, mul_0)
                    exp_value: txl.float32 = exp2_0
                    add_0 = txl.local_scalar("float32")
                    txl.ptx["add.ftz.f32"](add_0, txl.float32(1.0), exp_value)
                    log2_0 = txl.local_scalar("float32")
                    txl.ptx["lg2.approx.ftz.f32"](log2_0, add_0)
                    log_value: txl.float32 = log2_0
                    txl.ptx["mul.ftz.f32"](dt_value, log_value, txl.float32(_LN_2))
                txl.ptx.st.shared.b32(
                    s_dt.ptr_to([flat_thread]), txl.reinterpret("uint32", dt_value)
                )

            member: txl.int32 = lane % 8
            row_group: txl.int32 = lane // 8
            state_ptr_offset_i32: txl.int32 = txl.cast(
                state_batch * state_stride_batch + txl.cast(head * DIM * DSTATE, "int64"), "int32"
            )

            state_pipe = txl.PipelineState(2, phase=0)
            with txl.unroll(NUM_TMA_LOADS) as tl:
                _mbarrier_arrive_wait_parity(
                    full_barriers.ptr_to([state_pipe.stage]),
                    state_pipe.phase,
                    yield_while_waiting=use_wait_yield,
                )

                with txl.serial(2) as sp:
                    sram_row: txl.int32 = sp * 16 + compute_warp * 4 + row_group
                    dd: txl.int32 = tl * 32 + sram_row
                    state_values = txl.alloc_local((NUM_TILES * PAIRS_PER_TILE_MEMBER,), "uint64")

                    with txl.unroll(NUM_TILES) as tile:
                        member_col: txl.int32 = (
                            tile * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER
                        )
                        pair_base: txl.int32 = tile * PAIRS_PER_TILE_MEMBER
                        with txl.If(txl.And(txl.Not(IS_PAD), member_col < DSTATE)):
                            with txl.Then():
                                state_words = txl.alloc_local((4,), "uint32")
                                if use_direct_state_index:
                                    state_address = s_state_values.ptr_to(
                                        [
                                            state_pipe.stage * (STATE_STAGE_BYTES // STATE_BYTES)
                                            + sram_row * DSTATE_PAD
                                            + member_col
                                        ]
                                    )
                                else:
                                    state_word_index: txl.int32 = (
                                        state_pipe.stage * STATE_STAGE_BYTES
                                        + (sram_row * DSTATE_PAD + member_col) * STATE_BYTES
                                    ) // 4
                                    state_address = s_state_words.ptr_to([state_word_index])
                                txl.ptx.ld.shared.v4.b32(
                                    state_words[0],
                                    state_words[1],
                                    state_words[2],
                                    state_words[3],
                                    state_address,
                                )
                                with txl.unroll(PAIRS_PER_TILE_MEMBER) as pair:
                                    if STATE_DTYPE == "bfloat16":
                                        txl.assign(
                                            state_values[pair_base + pair],
                                            _bf16_word_to_f32x2(state_words[pair]),
                                        )
                                    elif STATE_DTYPE == "float16":
                                        f16_f32_0 = txl.local_scalar("float32")
                                        txl.ptx.cvt.f32.f16(
                                            f16_f32_0,
                                            txl.cast(
                                                _extract_u16(state_words[pair], False), "uint16"
                                            ),
                                        )
                                        f16_f32_1 = txl.local_scalar("float32")
                                        txl.ptx.cvt.f32.f16(
                                            f16_f32_1,
                                            txl.cast(
                                                _extract_u16(state_words[pair], True), "uint16"
                                            ),
                                        )
                                        txl.assign(
                                            state_values[pair_base + pair],
                                            (txl.cuda.make_float2(f16_f32_0, f16_f32_1)),
                                        )
                                    else:
                                        txl.assign(
                                            state_values[pair_base + pair],
                                            (
                                                txl.cuda.make_float2(
                                                    txl.reinterpret(
                                                        "float32", state_words[pair * 2]
                                                    ),
                                                    txl.reinterpret(
                                                        "float32", state_words[pair * 2 + 1]
                                                    ),
                                                )
                                            ),
                                        )
                            with txl.Else():
                                with txl.unroll(PAIRS_PER_TILE_MEMBER) as pair:
                                    txl.assign(
                                        state_values[pair_base + pair],
                                        txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0)),
                                    )

                    random_words = txl.alloc_local((16,), "uint32")
                    if PHILOX_ROUNDS > 0 and not IS_PAD:
                        seed_lo: txl.uint32 = txl.cast(random_seed, "uint32")
                        seed_hi: txl.uint32 = txl.cast(
                            txl.shift_right(txl.reinterpret("uint64", random_seed), txl.uint32(32)),
                            "uint32",
                        )
                        with txl.unroll(4) as random_group:
                            random_tile: txl.int32 = random_group // 2
                            random_e: txl.int32 = (random_group % 2) * 4
                            random_col: txl.int32 = (
                                random_tile * ELEMS_PER_TILE
                                + member * ELEMS_PER_TILE_MEMBER
                                + random_e
                            )
                            counter: txl.int32 = state_ptr_offset_i32 + dd * DSTATE + random_col
                            group_words = txl.alloc_local((4,), "uint32")
                            _philox4x32(
                                group_words, seed_lo, seed_hi, counter, PHILOX_ROUNDS=PHILOX_ROUNDS
                            )
                            with txl.unroll(4) as random_word:
                                txl.ptx.mov.b32(
                                    random_words[random_group * 4 + random_word],
                                    group_words[random_word],
                                )

                    bc_step = txl.local_scalar("int32", init=0)
                    x_step = txl.local_scalar("int32", init=0)
                    dt_step = txl.local_scalar("int32", init=0)
                    out_step = txl.local_scalar("int32", init=0)
                    if HAS_INTERMEDIATE_STATES and not IS_PAD:
                        intermediate_step_addr = txl.local_scalar(
                            "uint64",
                            init=txl.reinterpret(
                                "uint64",
                                intermediate_states.ptr_to(
                                    [
                                        icache_idx * txl.int64(NTOKENS * NHEADS * DIM * DSTATE)
                                        + txl.cast(head * DIM * DSTATE + dd * DSTATE, "int64")
                                    ]
                                ),
                            ),
                        )

                    with txl.serial(NTOKENS) as step:
                        dt_value = txl.local_scalar("float32")
                        sload_0 = txl.local_scalar("uint32")
                        txl.ptx.ld.shared.b32(sload_0, s_dt.ptr_to([dt_step]))
                        txl.assign(dt_value, txl.reinterpret("float32", sload_0))
                        mul_1 = txl.local_scalar("float32")
                        txl.ptx["mul.ftz.f32"](mul_1, a_value, dt_value)
                        mul_2 = txl.local_scalar("float32")
                        txl.ptx["mul.ftz.f32"](mul_2, mul_1, txl.float32(_LOG2_E))
                        exp2_1 = txl.local_scalar("float32")
                        txl.ptx["ex2.approx.ftz.f32"](exp2_1, mul_2)
                        da_value: txl.float32 = exp2_1
                        sload_1 = txl.local_scalar("uint16")
                        txl.ptx.ld.shared.b16(sload_1, s_x.ptr_to([x_step + dd]))
                        bf16_f32_0 = txl.local_scalar("float32")
                        txl.ptx.cvt.f32.bf16(bf16_f32_0, txl.cast(sload_1, "uint16"))
                        x_value: txl.float32 = bf16_f32_0
                        mul_3 = txl.local_scalar("float32")
                        txl.ptx["mul.ftz.f32"](mul_3, dt_value, x_value)
                        dtx_value: txl.float32 = mul_3
                        out_pair = txl.local_scalar("uint64")
                        txl.assign(
                            out_pair, txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0))
                        )

                        with txl.unroll(NUM_TILES) as tile:
                            member_col: txl.int32 = (
                                tile * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER
                            )
                            with txl.If(member_col < DSTATE), txl.Then():
                                b_words = txl.alloc_local((4,), "uint32")
                                c_words = txl.alloc_local((4,), "uint32")
                                if use_direct_bc_index:
                                    b_address = s_b.ptr_to([bc_step + member_col])
                                    c_address = s_c.ptr_to([bc_step + member_col])
                                else:
                                    b_word_index: txl.int32 = bc_step + member_col // 2
                                    c_word_index: txl.int32 = bc_step + member_col // 2
                                    b_address = s_b_words.ptr_to([b_word_index])
                                    c_address = s_c_words.ptr_to([c_word_index])
                                if PAIRS_PER_TILE_MEMBER == 2:
                                    txl.ptx.ld.shared.v2.b32(b_words[0], b_words[1], b_address)
                                    txl.ptx.ld.shared.v2.b32(c_words[0], c_words[1], c_address)
                                else:
                                    txl.ptx.ld.shared.v4.b32(
                                        b_words[0], b_words[1], b_words[2], b_words[3], b_address
                                    )
                                    txl.ptx.ld.shared.v4.b32(
                                        c_words[0], c_words[1], c_words[2], c_words[3], c_address
                                    )
                                with txl.unroll(PAIRS_PER_TILE_MEMBER) as pair:
                                    b_pair: txl.uint64 = _bf16_word_to_f32x2(b_words[pair])
                                    c_pair: txl.uint64 = _bf16_word_to_f32x2(c_words[pair])
                                    dbx_pair = txl.local_scalar("uint64")
                                    txl.ptx.mul.f32x2(
                                        dbx_pair, b_pair, txl.cuda.make_float2(dtx_value, dtx_value)
                                    )
                                    pair_index: txl.int32 = tile * PAIRS_PER_TILE_MEMBER + pair
                                    updated_state = txl.local_scalar("uint64")
                                    txl.ptx.fma.rn.f32x2(
                                        updated_state,
                                        txl.cuda.make_float2(da_value, da_value),
                                        state_values[pair_index],
                                        dbx_pair,
                                    )
                                    txl.ptx.mov.b64(state_values[pair_index], updated_state)
                                    txl.ptx.fma.rn.f32x2(out_pair, updated_state, c_pair, out_pair)

                        out_value = txl.local_scalar("float32")
                        txl.ptx["add.ftz.f32"](
                            out_value, txl.cuda.float2_x(out_pair), txl.cuda.float2_y(out_pair)
                        )
                        with txl.unroll(3) as delta_idx:
                            txl.ptx["add.ftz.f32"](
                                out_value,
                                out_value,
                                _shfl_down_f32(out_value, txl.shift_right(txl.int32(4), delta_idx)),
                            )
                        with txl.If(member == 0), txl.Then():
                            fma_0 = txl.local_scalar("float32")
                            txl.ptx["fma.rn.ftz.f32"](fma_0, d_value, x_value, out_value)
                            txl.ptx.st.shared.b32(
                                s_out.ptr_to([out_step + dd]), txl.reinterpret("uint32", fma_0)
                            )

                        txl.assign(
                            bc_step,
                            bc_step + (DSTATE_PAD if use_direct_bc_index else DSTATE_PAD // 2),
                        )
                        txl.assign(x_step, x_step + DIM)
                        txl.assign(dt_step, dt_step + 1)
                        txl.assign(out_step, out_step + DIM)

                        if HAS_INTERMEDIATE_STATES and not IS_PAD:
                            with txl.unroll(NUM_TILES) as tile:
                                member_col: txl.int32 = (
                                    tile * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER
                                )
                                with txl.If(member_col < DSTATE), txl.Then():
                                    _store_state_tile(
                                        state_values,
                                        tile * PAIRS_PER_TILE_MEMBER,
                                        random_words,
                                        tile * 8,
                                        txl.reinterpret(
                                            PointerType(PrimType(STATE_DTYPE), "global"),
                                            intermediate_step_addr
                                            + txl.cast(member_col * STATE_BYTES, "uint64"),
                                        ),
                                        STATE_DTYPE=STATE_DTYPE,
                                        PAIRS_PER_TILE_MEMBER=PAIRS_PER_TILE_MEMBER,
                                        PHILOX_ROUNDS=PHILOX_ROUNDS,
                                    )
                            txl.assign(
                                intermediate_step_addr,
                                intermediate_step_addr
                                + txl.uint64(NHEADS * DIM * DSTATE * STATE_BYTES),
                            )

                        if UPDATE_STATE and not IS_PAD:
                            with txl.If(step == NTOKENS - 1), txl.Then():
                                final_base: txl.int64 = state_batch * state_stride_batch + txl.cast(
                                    head * DIM * DSTATE + dd * DSTATE, "int64"
                                )
                                with txl.unroll(NUM_TILES) as tile:
                                    member_col: txl.int32 = (
                                        tile * ELEMS_PER_TILE + member * ELEMS_PER_TILE_MEMBER
                                    )
                                    with txl.If(member_col < DSTATE), txl.Then():
                                        _store_state_tile(
                                            state_values,
                                            tile * PAIRS_PER_TILE_MEMBER,
                                            random_words,
                                            tile * 8,
                                            state.ptr_to([final_base + member_col]),
                                            STATE_DTYPE=STATE_DTYPE,
                                            PAIRS_PER_TILE_MEMBER=PAIRS_PER_TILE_MEMBER,
                                            PHILOX_ROUNDS=PHILOX_ROUNDS,
                                        )

                txl.ptx.mbarrier.arrive.shared__cta.b64(empty_barriers.ptr_to([state_pipe.stage]))
                state_pipe.advance()

            _mbarrier_arrive_wait_parity(
                out_ready_barrier.ptr_to([0]), 0, yield_while_waiting=use_wait_yield
            )
            with txl.unroll((NTOKENS + 3) // 4) as episode:
                step: txl.int32 = compute_warp + episode * 4
                with txl.If(step < NTOKENS), txl.Then():
                    out_base: txl.int64 = (
                        txl.cast(batch_i, "int64") * out_stride_batch
                        + txl.cast(step, "int64") * out_stride_mtp
                        + head * DIM
                    )
                    z_base: txl.int64 = (
                        txl.cast(batch_i, "int64") * z_stride_batch
                        + txl.cast(step, "int64") * z_stride_mtp
                        + head * DIM
                    )
                    if DIM == 64:
                        d: txl.int32 = lane * 2
                        out_words = txl.alloc_local((2,), "uint32")
                        if NTOKENS == 1:
                            txl.ptx.ld.shared.b32(out_words[0], s_out.ptr_to([step * DIM + d]))
                            txl.ptx.ld.shared.b32(out_words[1], s_out.ptr_to([step * DIM + d + 1]))
                        else:
                            txl.ptx.ld.shared.v2.b32(
                                out_words[0], out_words[1], s_out.ptr_to([(step * DIM + d)])
                            )
                        z_bits = txl.alloc_local((2,), "uint16")
                        output_bits = txl.alloc_local((2,), "uint16")
                        if HAS_Z:
                            txl.ptx.ld.global_.v2.b16(
                                z_bits[0], z_bits[1], z.ptr_to([(z_base + d)])
                            )
                        with txl.unroll(2) as k:
                            out_value = txl.local_scalar(
                                "float32", init=txl.reinterpret("float32", out_words[k])
                            )
                            if HAS_Z:
                                bf16_f32_1 = txl.local_scalar("float32")
                                txl.ptx.cvt.f32.bf16(bf16_f32_1, txl.cast(z_bits[k], "uint16"))
                                z_value: txl.float32 = bf16_f32_1
                                sub_0 = txl.local_scalar("float32")
                                txl.ptx["sub.ftz.f32"](sub_0, txl.float32(0.0), z_value)
                                mul_4 = txl.local_scalar("float32")
                                txl.ptx["mul.ftz.f32"](mul_4, sub_0, txl.float32(_LOG2_E))
                                exp2_2 = txl.local_scalar("float32")
                                txl.ptx["ex2.approx.ftz.f32"](exp2_2, mul_4)
                                exp_neg: txl.float32 = exp2_2
                                add_1 = txl.local_scalar("float32")
                                txl.ptx["add.ftz.f32"](add_1, txl.float32(1.0), exp_neg)
                                div_0 = txl.local_scalar("float32")
                                txl.ptx["div.approx.ftz.f32"](div_0, txl.float32(1.0), add_1)
                                sigmoid: txl.float32 = div_0
                                mul_5 = txl.local_scalar("float32")
                                txl.ptx["mul.ftz.f32"](mul_5, z_value, sigmoid)
                                txl.ptx["mul.ftz.f32"](out_value, out_value, mul_5)
                            txl.ptx.cvt.rn.bf16.f32(output_bits[k], out_value)
                        txl.ptx.st.global_.v2.b16(
                            output.ptr_to([out_base + d]), output_bits[0], output_bits[1]
                        )
                    else:
                        d: txl.int32 = lane * 4
                        out_words = txl.alloc_local((4,), "uint32")
                        txl.ptx.ld.shared.v4.b32(
                            out_words[0],
                            out_words[1],
                            out_words[2],
                            out_words[3],
                            s_out.ptr_to([(step * DIM + d)]),
                        )
                        z_bits = txl.alloc_local((4,), "uint16")
                        output_bits = txl.alloc_local((4,), "uint16")
                        if HAS_Z:
                            txl.ptx.ld.global_.v4.b16(
                                z_bits[0], z_bits[1], z_bits[2], z_bits[3], z.ptr_to([(z_base + d)])
                            )
                        with txl.unroll(4) as k:
                            out_value = txl.local_scalar(
                                "float32", init=txl.reinterpret("float32", out_words[k])
                            )
                            if HAS_Z:
                                bf16_f32_2 = txl.local_scalar("float32")
                                txl.ptx.cvt.f32.bf16(bf16_f32_2, txl.cast(z_bits[k], "uint16"))
                                z_value: txl.float32 = bf16_f32_2
                                sub_1 = txl.local_scalar("float32")
                                txl.ptx["sub.ftz.f32"](sub_1, txl.float32(0.0), z_value)
                                mul_6 = txl.local_scalar("float32")
                                txl.ptx["mul.ftz.f32"](mul_6, sub_1, txl.float32(_LOG2_E))
                                exp2_3 = txl.local_scalar("float32")
                                txl.ptx["ex2.approx.ftz.f32"](exp2_3, mul_6)
                                exp_neg: txl.float32 = exp2_3
                                add_2 = txl.local_scalar("float32")
                                txl.ptx["add.ftz.f32"](add_2, txl.float32(1.0), exp_neg)
                                div_1 = txl.local_scalar("float32")
                                txl.ptx["div.approx.ftz.f32"](div_1, txl.float32(1.0), add_2)
                                sigmoid: txl.float32 = div_1
                                mul_7 = txl.local_scalar("float32")
                                txl.ptx["mul.ftz.f32"](mul_7, z_value, sigmoid)
                                txl.ptx["mul.ftz.f32"](out_value, out_value, mul_7)
                            txl.ptx.cvt.rn.bf16.f32(output_bits[k], out_value)
                        txl.ptx.st.global_.v4.b16(
                            output.ptr_to([out_base + d]),
                            output_bits[0],
                            output_bits[1],
                            output_bits[2],
                            output_bits[3],
                        )

        def run_load(IS_PAD: txl.constexpr):
            lane: txl.int32 = txl.tid_in_role()
            kv_group: txl.int32 = head // HEADS_PER_GROUP
            with txl.If(lane == 0), txl.Then():
                _tma_g2s_4d(
                    s_b.ptr_to([0]), tensor_b, 0, kv_group, 0, batch_i, full_barriers.ptr_to([0])
                )
                _tma_g2s_4d(
                    s_c.ptr_to([0]), tensor_c, 0, kv_group, 0, batch_i, full_barriers.ptr_to([0])
                )
                _tma_g2s_4d(
                    s_x.ptr_to([0]), tensor_x, 0, head, 0, batch_i, full_barriers.ptr_to([0])
                )

            bcx_bytes: txl.int32 = 2 * NTOKENS * DSTATE_PAD * 2 + NTOKENS * DIM * 2
            state_pipe = txl.PipelineState(2, phase=0)
            with txl.unroll(NUM_TMA_LOADS) as tl:
                _mbarrier_arrive_wait_parity(
                    empty_barriers.ptr_to([state_pipe.stage]),
                    state_pipe.phase,
                    yield_while_waiting=use_wait_yield,
                )
                with txl.If(lane == 0), txl.Then():
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
                        transaction_bytes = txl.local_scalar("int32", init=STATE_STAGE_BYTES)
                        with txl.If(tl == 0), txl.Then():
                            txl.assign(transaction_bytes, transaction_bytes + bcx_bytes)
                        txl.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
                            full_barriers.ptr_to([state_pipe.stage]), txl.uint32(transaction_bytes)
                        )
                    else:
                        transaction_bytes = txl.local_scalar("int32", init=0)
                        with txl.If(tl == 0), txl.Then():
                            txl.assign(transaction_bytes, bcx_bytes)
                        txl.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
                            full_barriers.ptr_to([state_pipe.stage]), txl.uint32(transaction_bytes)
                        )
                state_pipe.advance()

        with update:
            with txl.If(state_batch == txl.cast(pad_slot_id, "int64")):
                with txl.Then():
                    run_update(True)
                with txl.Else():
                    run_update(False)
        with load:
            with txl.If(state_batch == txl.cast(pad_slot_id, "int64")):
                with txl.Then():
                    run_load(True)
                with txl.Else():
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
    if _use_simple_dispatch(case["spec"], os.environ.get(PREPARE_CUDA_ARCH_ENV, "")):
        return _simple._tirx_args(case)
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


def _compile_tirx(config: dict[str, Any]):
    from tirx_kernels.runner import compile_kernel

    arch = os.environ.get(PREPARE_CUDA_ARCH_ENV, "")
    spec = _specialization(config)
    ptxas_level = (
        _simple_ptxas_level(spec, arch)
        if _use_simple_dispatch(spec, arch)
        else _ptxas_level(spec, arch)
    )
    previous = os.environ.get("TVM_CUDA_PTXAS_REG_LEVEL")
    if ptxas_level is not None:
        os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = ptxas_level
    try:
        return compile_kernel(get_kernel(**config))
    finally:
        if previous is None:
            os.environ.pop("TVM_CUDA_PTXAS_REG_LEVEL", None)
        else:
            os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = previous


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": _compile_tirx(kwargs)}
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
    case = prepare_data(**kwargs)
    executable = _compile_tirx(kwargs)
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
